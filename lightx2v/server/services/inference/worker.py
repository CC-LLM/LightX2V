import asyncio
import os
from typing import Any, Dict

import torch
from easydict import EasyDict
from loguru import logger

from lightx2v.infer import init_runner
from lightx2v.utils.input_info import set_input_info
from lightx2v.utils.set_config import set_config, set_parallel_config

from ...utils import parse_resolution
from ..distributed_utils import DistributedManager


VALID_RESOLUTIONS = {"480p", "580p", "720p"}
VALID_ASPECT_RATIOS_COMMON = {"16:9", "9:16", "1:1"}
VALID_ASPECT_RATIOS_I2V = {"auto", "16:9", "9:16", "1:1"}


class TorchrunInferenceWorker:
    def __init__(self):
        self.rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.runner = None
        self.dist_manager = DistributedManager()
        self.processing = False

    def _validate_resolution_and_aspect_ratio(self, resolution: str, aspect_ratio: str, task: str):
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution '{resolution}'. "
                f"Supported resolutions: {', '.join(sorted(VALID_RESOLUTIONS))}"
            )
        
        valid_ratios = VALID_ASPECT_RATIOS_I2V if task == "i2v" else VALID_ASPECT_RATIOS_COMMON
        if aspect_ratio not in valid_ratios:
            raise ValueError(
                f"Invalid aspect ratio '{aspect_ratio}' for task '{task}'. "
                f"Supported aspect ratios: {', '.join(sorted(valid_ratios))}"
            )

    def init(self, args) -> bool:
        try:
            if self.world_size > 1:
                if not self.dist_manager.init_process_group():
                    raise RuntimeError("Failed to initialize distributed process group")
            else:
                self.dist_manager.rank = 0
                self.dist_manager.world_size = 1
                self.dist_manager.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.dist_manager.is_initialized = False

            config = set_config(args)

            if config["parallel"]:
                set_parallel_config(config)

            if self.rank == 0:
                logger.info(f"Config:\n {config}")

            self.runner = init_runner(config)
            logger.info(f"Rank {self.rank}/{self.world_size - 1} initialization completed")

            return True

        except Exception as e:
            logger.exception(f"Rank {self.rank} initialization failed: {str(e)}")
            return False

    async def process_request(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        has_error = False
        error_msg = ""

        try:
            if self.world_size > 1 and self.rank == 0:
                task_data = self.dist_manager.broadcast_task_data(task_data)

            task_data["task"] = self.runner.config["task"]
            task_data["return_result_tensor"] = False
            task_data["negative_prompt"] = task_data.get("negative_prompt", "")

            # NOTE(wxy): 处理客户端自定义字段
            if self.runner.scheduler.__class__.__name__ == "WanStepDistillScheduler":
                task_data.pop("infer_steps", None)
                logger.info(f"Using WanStepDistillScheduler, pop infer_steps from client request")
            
            # NOTE(wxy): 处理 runner 中的配置
            # 1. [target_fps]: VideoTaskRequest 默认值为 16
            # 覆盖顺序: 客户端 > runner > 默认值 16
            runner_target_fps = self.runner.config.get("target_fps")
            target_fps = runner_target_fps if runner_target_fps is not None else 16
            target_fps = task_data.get("target_fps", target_fps)
            vfi_cfg = self.runner.config.get("video_frame_interpolation")
            if vfi_cfg:
                task_data["video_frame_interpolation"] = {**vfi_cfg, "target_fps": target_fps}
            else:
                task_data["target_fps"] = target_fps
                logger.warning(f"Target FPS {target_fps} is set, but video frame interpolation is not configured")
            logger.info(f"Runner config target_fps: {runner_target_fps} -> Client request target_fps: {target_fps}.")                    

            # 2. [target_video_length]: VideoTaskRequest 默认值为 81
            # 覆盖顺序: 客户端 > runner > 默认值 81
            runner_target_video_length = self.runner.config.get("target_video_length")
            target_video_length = runner_target_video_length if runner_target_video_length is not None else 81
            target_video_length = task_data.get("target_video_length", target_video_length)
            # self.runner.scheduler.target_video_length = target_video_length
            vae_stride = self.runner.config.get("vae_stride")
            model_cls = self.runner.config.get("model_cls")
            if task_data["task"] in ["i2v", "s2v"]:
                if target_video_length % vae_stride[0] != 1:
                    target_video_length = target_video_length // vae_stride[0] * vae_stride[0] + 1
                    logger.warning(f"`num_frames - 1` has to be divisible by {vae_stride[0]}. Rounding to the nearest number {target_video_length}.")
            
            if task_data["task"] not in ["t2i", "i2i"] and model_cls not in ["hunyuan_video_1.5", "hunyuan_video_1.5_distill"]:
                task_data["attnmap_frame_num"] = ((target_video_length - 1) // vae_stride[0] + 1) // self.runner.config.get("patch_size")[0]
                if model_cls == "seko_talk":
                    task_data["attnmap_frame_num"] += 1
                logger.info(f"Updated attnmap_frame_num to {task_data['attnmap_frame_num']}")
            
            task_data["target_video_length"] = target_video_length
            logger.info(f"Runner config target_video_length: {runner_target_video_length} -> Client requesttarget_video_length: {target_video_length}.")                    

            # 3. [resolution]: 默认值为 480p, [aspect_ratio]: 默认值为 auto (i2v) 或 16:9 (其他)
            # 覆盖顺序: 客户端 > runner > 默认值, 如果客户端传入 "default", 则使用 runner 中的值
            default_aspect_ratio = "auto" if task_data["task"] == "i2v" else "16:9"
            runner_resolution = self.runner.config.get("resolution")
            runner_aspect_ratio = self.runner.config.get("aspect_ratio")
            logger.info(f"Runner config: resolution: {runner_resolution}, aspect_ratio: {runner_aspect_ratio}")
            resolution = runner_resolution if runner_resolution is not None else "480p"
            aspect_ratio = runner_aspect_ratio if runner_aspect_ratio is not None else default_aspect_ratio
            resolution = task_data.get("resolution", resolution)
            if resolution == "default":
                resolution = runner_resolution
            aspect_ratio = task_data.get("aspect_ratio", aspect_ratio)
            if aspect_ratio == "default":
                aspect_ratio = runner_aspect_ratio
            task_data["resolution"] = resolution
            task_data["aspect_ratio"] = aspect_ratio
            self._validate_resolution_and_aspect_ratio(resolution, aspect_ratio, task_data["task"])
            logger.info(f"{task_data['task']} task: resolution: {resolution}, aspect_ratio: {aspect_ratio}")

            # 4. [resize_mode]: i2v 均设为 adaptive
            if task_data["task"] == "i2v":
                task_data["resize_mode"] = "adaptive"

            # 5. 非 auto 时，计算 target_height 和 target_width; auto 时将在 resize_image 中计算
            if aspect_ratio == "auto":
                logger.info(f"{task_data['task']} task: aspect_ratio is auto, will be calculated in resize_image")
            else:
                resolution_values = parse_resolution(resolution, aspect_ratio)
                height, width = resolution_values
                task_data["target_height"] = height
                task_data["target_width"] = width
                logger.info(f"{task_data['task']} task: resolution '{resolution}' with aspect_ratio '{aspect_ratio}' mapped to {height}x{width}")

            task_data = EasyDict(task_data)
            input_info = set_input_info(task_data)

            self.runner.set_config(task_data)  # 更新全局 runner config
            self.runner.run_pipeline(input_info)

            # 由于 runner config 被更新, 重置 resolution/aspect_ratio/target_video_length 为 runner config 中的值
            with self.runner.config.temporarily_unlocked():
                self.runner.config["resolution"] = runner_resolution
                self.runner.config["aspect_ratio"] = runner_aspect_ratio
                self.runner.config["target_video_length"] = runner_target_video_length
                self.runner.config["target_fps"] = runner_target_fps
                logger.info(f"Reset resolution and aspect_ratio to runner config: {self.runner.config['resolution']} and {self.runner.config['aspect_ratio']}")
            
            # Collect final config after inference (convert numpy types to Python native types)
            final_config = {
                "target_video_length": task_data["target_video_length"],
                "target_fps": task_data["target_fps"],
                # 每个请求推理时下面两个值都会被重新计算并设置到 runner config 中
                "target_height": int(self.runner.config.get("target_height")),
                "target_width": int(self.runner.config.get("target_width")),
            }
            await asyncio.sleep(0)

        except Exception as e:
            has_error = True
            error_msg = str(e)
            final_config = {}
            logger.exception(f"Rank {self.rank} inference failed: {error_msg}")

        if self.world_size > 1:
            self.dist_manager.barrier()

        if self.rank == 0:
            if has_error:
                return {
                    "task_id": task_data.get("task_id", "unknown"),
                    "status": "failed",
                    "error": error_msg,
                    "message": f"Inference failed: {error_msg}",
                }
            else:
                return {
                    "task_id": task_data["task_id"],
                    "status": "success",
                    "save_result_path": task_data.get("video_path", task_data["save_result_path"]),
                    "message": "Inference completed",
                    "final_config": final_config,
                }
        else:
            return None

    async def worker_loop(self):
        while True:
            task_data = None
            try:
                task_data = self.dist_manager.broadcast_task_data()
                if task_data is None:
                    logger.info(f"Rank {self.rank} received stop signal")
                    break

                await self.process_request(task_data)

            except Exception as e:
                error_str = str(e)
                if "Connection closed by peer" in error_str or "Connection reset by peer" in error_str:
                    logger.info(f"Rank {self.rank} detected master process shutdown, exiting worker loop")
                    break
                logger.error(f"Rank {self.rank} worker loop error: {error_str}")
                if self.world_size > 1 and task_data is not None:
                    try:
                        self.dist_manager.barrier()
                    except Exception as barrier_error:
                        logger.warning(f"Rank {self.rank} barrier failed, exiting: {barrier_error}")
                        break
                continue

    def cleanup(self):
        self.dist_manager.cleanup()

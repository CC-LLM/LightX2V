from typing import Dict, Tuple


# NOTE: 与 default_runner.py 中 resize_image 的 bucket_config 保持一致
RESOLUTION_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "16:9": {
        "480p": (480, 848),
        "540p": (544, 960),
        "580p": (580, 1024),
        "720p": (720, 1280),
    },
    "9:16": {
        "480p": (848, 480),
        "540p": (960, 544),
        "580p": (1024, 580),
        "720p": (1280, 720),
    },
    "1:1": {
        "480p": (480, 480),
        "540p": (576, 576),
        "580p": (580, 580),
        "720p": (720, 720),
    },
}


def parse_resolution(resolution, aspect_ratio) -> Tuple[int, int]:
    """
    根据 resolution 和 aspect_ratio 选择对应的宽高
    
    Args:
        resolution: 分辨率预设字符串 (如 '480p', '540p', '580p', '720p')
        aspect_ratio: 宽高比字符串 (如 '16:9'横屏, '9:16'竖屏, '1:1'正方形)
    
    Returns:
        (height, width) 元组，如果解析失败则返回 None
    """
    assert resolution in ["480p", "540p", "580p", "720p"], f"Invalid resolution: {resolution}"
    assert aspect_ratio in ["16:9", "9:16", "1:1"], f"Invalid aspect ratio: {aspect_ratio}"
    return RESOLUTION_PRESETS[aspect_ratio][resolution]

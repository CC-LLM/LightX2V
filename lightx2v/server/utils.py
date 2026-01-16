from typing import Dict, Tuple


# NOTE: 与 default_runner.py 中 resize_image 的 RESOLUTION_PRESETS 保持一致
RESOLUTION_PRESETS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "16:9": {
        "480p": (832, 480),
        "580p": (960, 512),
        "720p": (1280, 720),
    },
    "9:16": {
        "480p": (480, 832),
        "580p": (512, 960),
        "720p": (720, 1280),
    },
    "1:1": {
        "480p": (480, 480),
        "580p": (512, 512),
        "720p": (720, 720),
    },
}


def parse_resolution(resolution, aspect_ratio) -> Tuple[int, int]:
    """
    根据 resolution 和 aspect_ratio 选择对应的宽高
    
    Args:
        resolution: 分辨率预设字符串 (如 '480p', '580p', '720p')
        aspect_ratio: 宽高比字符串 (如 '16:9'横屏, '9:16'竖屏, '1:1'正方形)
    
    Returns:
        (width, height) 元组，如果解析失败则返回 None
    """
    assert resolution in ["480p", "580p", "720p"], f"Invalid resolution: {resolution}"
    assert aspect_ratio in ["16:9", "9:16", "1:1"], f"Invalid aspect ratio: {aspect_ratio}"
    return RESOLUTION_PRESETS[aspect_ratio][resolution]

"""生产运行时装配（狗侧）。入口见 dog_runtime.py；main.py 不动。"""

from runtime.dog_runtime import DogRuntime, DogRuntimeConfig, load_topsee_config

__all__ = ["DogRuntime", "DogRuntimeConfig", "load_topsee_config"]

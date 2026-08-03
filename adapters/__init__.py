"""平台适配器：统一 Scout / Dog 契约，对接 MissionBrain。"""

from adapters.dog_stub import DogStubAdapter
from adapters.drone_autel import AutelScoutAdapter
from adapters.drone_tello import TelloScoutAdapter

__all__ = ["AutelScoutAdapter", "DogStubAdapter", "TelloScoutAdapter"]

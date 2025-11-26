from dataclasses import dataclass

@dataclass
class NodeParams:
    leak: float
    bias: float

@dataclass
class EdgeParams:
    weight: float
    delay: int  # discrete timesteps delay

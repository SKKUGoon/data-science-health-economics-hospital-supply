from pydantic import BaseModel
from typing import Literal


class DFMConfig(BaseModel):
    freq: Literal["D", "W", "2W", "M"] = "W"
    k_factors: int = 2
    factor_order: int = 1
    standardize: bool = True


from __future__ import annotations

from typing import List

from pydantic import BaseModel, Extra


class PredictResponse(BaseModel):
    id_peca: int
    fases: List[str]
    operacao: List[str]


class PartInput(BaseModel):
    id_peca: int

    class Config:
        extra = Extra.allow

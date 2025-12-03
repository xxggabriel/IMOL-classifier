from __future__ import annotations

from typing import List
from fastapi import APIRouter, HTTPException, status

from ..inference.predict import predict_records
from .schemas import PartInput, PredictResponse

router = APIRouter(prefix="/infer", tags=["infer"])


@router.post("/json", response_model=List[PredictResponse], status_code=status.HTTP_200_OK)
def predict_from_json(payload: List[PartInput]) -> List[PredictResponse]:
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Payload vazio")

    records = []
    for part in payload:
        try:
            data = part.model_dump()
        except AttributeError:  # pydantic < 2 compatibility
            data = part.dict()
        records.append(data)

    try:
        df_with_predictions = predict_records(records)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Artefatos de modelo não encontrados: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar previsões: {exc}",
        ) from exc

    expected_cols = {"id_peca", "PREDICOES_FASE", "PREDICOES_OPERACAO"}
    missing = expected_cols - set(df_with_predictions.columns)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Colunas de previsão ausentes: {', '.join(sorted(missing))}",
        )

    responses: List[PredictResponse] = []
    for _, row in df_with_predictions.iterrows():
        fases = [
            str(f)
            for f in (row["PREDICOES_FASE"] or [])
            if f is not None and str(f).strip() != ""
        ]
        operacao = [
            str(o)
            for o in (row["PREDICOES_OPERACAO"] or [])
            if o is not None and str(o).strip() != ""
        ]
        try:
            id_peca = int(row["id_peca"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Valor inválido de id_peca recebido: {row['id_peca']}",
            ) from exc
        responses.append(PredictResponse(id_peca=id_peca, fases=operacao, operacao=fases))

    return responses

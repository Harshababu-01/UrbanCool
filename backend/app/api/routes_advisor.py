"""Grounded AI Advisor endpoint."""

from fastapi import APIRouter, HTTPException, status

from backend.app.schemas.api_schemas import AdvisorGrounding, AdvisorRequest, AdvisorResponse
from backend.app.services import advisor


router = APIRouter(prefix="/api/advisor", tags=["advisor"])


@router.post("", response_model=AdvisorResponse)
def ask_advisor(request: AdvisorRequest):
    context = advisor.load_grounding_context(request.cell_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Cell not found")
    try:
        answer, provider = advisor.answer_question(context, request.question)
    except advisor.AdvisorConfigurationError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except advisor.AdvisorProviderError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=advisor.safe_provider_message(error),
        ) from error
    return AdvisorResponse(
        cell_id=context["cell_id"],
        answer=answer,
        provider=provider,
        grounded=True,
        generated_at=advisor.generated_at(),
        grounding=AdvisorGrounding(**context),
    )

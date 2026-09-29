from src.models.base import BaseModel, ModelCard


class EuroLLM(BaseModel):
    """Class for the EuroLLM instruction model."""

    card = ModelCard(
        name="eurollm_9b",
        hf_id="utter-project/EuroLLM-9B-Instruct",
        params="9.2B",
        licence="apache-2.0",
        languages="35 languages",
        accessibility="gated, so a licence has to be accepted on the Hub and a token set as HF_TOKEN",
    )

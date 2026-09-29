from src.models.base import BaseModel, ModelCard


class Qwen35(BaseModel):
    """Class for Qwen3.5."""

    card = ModelCard(
        name="qwen3_5_9b",
        hf_id="Qwen/Qwen3.5-9B",
        params="9.7B, where 9B is the language model",
        licence="apache-2.0",
        languages="201 languages and dialects",
        accessibility="ungated, downloadable without an account",
    )

    def template_kwargs(self) -> dict:
        """Get the extra keyword arguments this model's chat template expects.

        Qwen3.5 thinks by default. Turning that off makes the template prefill an empty
        `<think></think>` pair, so the model answers straight away and the completion is the label
        list on its own.

        Returns:
            dict: A dictionary containing template arguments.
        """
        return {"enable_thinking": False}


class Qwen35Small(Qwen35):
    """The classs for the smallest Qwen3.5, for the purpose of local testing."""

    card = ModelCard(
        name="qwen3_5_0_8b",
        hf_id="Qwen/Qwen3.5-0.8B",
        params="0.9B, where 0.75B is the language model",
        licence="apache-2.0",
        languages="201 languages and dialects",
        accessibility="ungated, downloadable without an account",
    )

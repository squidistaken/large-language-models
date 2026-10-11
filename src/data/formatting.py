import hashlib
import json
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.constants import DEFAULT_TEMPLATE
from src.data.labels import XED_LABELS, LabelSpace


@dataclass(frozen=True)
class Prompts:
    """The wording of the prompting system message, user message, and label definitions."""

    definitions: dict[str, str]
    templates: dict[str, str]
    user: str

    def __post_init__(self) -> None:
        """Get the prompt wording from the prompting config, and check it.

        Raises:
            ValueError: If a label lacks a definition, the default template is missing, or a
                        template lacks its placeholder.
        """
        missing = [name for name in XED_LABELS.names if name not in self.definitions]

        if missing:
            raise ValueError(f"prompting.definitions has no definition for {', '.join(missing)}")

        if DEFAULT_TEMPLATE not in self.templates:
            raise ValueError(f"prompting.templates has no '{DEFAULT_TEMPLATE}' template")

        for name, template in self.templates.items():
            if "{definitions}" not in template:
                raise ValueError(f"prompting.templates.{name} lacks {{definitions}}")

        if "{text}" not in self.user:
            raise ValueError("prompting.user lacks {text}")

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "Prompts":
        """Get the prompt wording from the prompting config.

        Args:
            cfg (DictConfig): The prompting config.

        Returns:
            Prompts: The prompt wording.

        Raises:
            ValueError: If a label lacks a definition, the default template is missing, or a
                        template lacks its placeholder.
        """
        container: Any = OmegaConf.to_container(cfg, resolve=True)

        return cls(
            dict(container["definitions"]), dict(container["templates"]), str(container["user"])
        )


@dataclass(frozen=True)
class ConversationFormatter:
    """Render canonical records as conversational examples."""

    prompts: Prompts
    template: str = DEFAULT_TEMPLATE
    labels: LabelSpace = XED_LABELS

    def __post_init__(self) -> None:
        """Get the prompt wording from the prompting config, and check it.

        Raises:
            ValueError: If the template is unknown.
        """
        if self.template not in self.prompts.templates:
            raise ValueError(
                f"Unknown prompt template '{self.template}'. "
                f"Choose one of: {', '.join(self.prompts.templates)}"
            )

    @property
    def system_prompt(self) -> str:
        """Get the system message for the prompt template, with the label definitions.

        Returns:
            str: A system message.
        """
        definitions = "\n".join(
            f"- {name}: {self.prompts.definitions[name]}" for name in self.labels.names
        )

        return self.prompts.templates[self.template].format(definitions=definitions)

    @property
    def prompt_sha256(self) -> str:
        """Get a hex digest of the system message and user template, for caching.

        Returns:
            str: A hex digest of the system message and user template.
        """
        return hashlib.sha256(f"{self.system_prompt}\n{self.prompts.user}".encode()).hexdigest()

    def user_prompt(self, text: str) -> str:
        """Get the user message for a subtitle line, with the text.

        Args:
            text (str): The subtitle line.

        Returns:
            str: A user message.
        """
        return self.prompts.user.format(text=text)

    def target(self, label_names: list[str]) -> str:
        """Get the assistant message for a label set.

        Args:
            label_names (list[str]): The label names, already in the fixed order.

        Returns:
            str: A assistant message.
        """
        return json.dumps(label_names, ensure_ascii=False)

    def conversation(self, row: dict) -> dict:
        """Get a canonical record as a chat-formatted example, with system, user, and assistant
        messages.

        Args:
            row (dict): The canonical record.

        Returns:
            dict: A chat-formatted example.
        """
        return {
            "align_id": row["align_id"],
            "lang": row["lang"],
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_prompt(row["text"])},
                {"role": "assistant", "content": self.target(row["labels"])},
            ],
        }

    def with_prompt(self, example: dict, turns: list[dict]) -> dict:
        """Get an example with this formatter's system message, and demonstration messages
        between the system message and the subtitle.

        Args:
            example (dict): A conversational example: system, user, assistant.
            turns (list[dict]): The demonstration messages, or none.

        Returns:
            dict: A copy of the example with the system message and demonstrations in its messages.
        """
        system = {"role": "system", "content": self.system_prompt}

        return {**example, "messages": [system, *turns, *example["messages"][1:]]}

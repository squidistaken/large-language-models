import hashlib
import json
from dataclasses import dataclass

from src.data.labels import XED_LABELS, LabelSpace

EMOTION_DEFINITIONS = {
    "anger": "displeasure or hostility at a perceived wrong",
    "anticipation": "expecting or looking forward to something",
    "disgust": "revulsion or strong disapproval",
    "fear": "feeling threatened or afraid",
    "joy": "happiness or pleasure",
    "sadness": "sorrow, loss or disappointment",
    "surprise": "reaction to something unexpected",
    "trust": "confidence in, or acceptance of, someone or something",
}
SYSTEM_TEMPLATE = (
    "You classify the emotions expressed in a movie subtitle line.\n\n"
    "Permitted labels:\n{definitions}\n\n"
    "Every line expresses at least one of these emotions and may express several. "
    "Answer with a JSON list of the matching label names, using only the names above, "
    'and nothing else, in the format ["label", ...].'
)
USER_TEMPLATE = "Subtitle: {text}"


@dataclass(frozen=True)
class ConversationFormatter:
    """Render canonical records as conversational examples."""

    labels: LabelSpace = XED_LABELS

    @property
    def system_prompt(self) -> str:
        """The system message defining the task and the permitted labels.

        Returns:
            str: A system message.
        """
        definitions = "\n".join(
            f"- {name}: {EMOTION_DEFINITIONS[name]}" for name in self.labels.names
        )

        return SYSTEM_TEMPLATE.format(definitions=definitions)

    @property
    def prompt_sha256(self) -> str:
        """Fingerprint of the prompt, so splits built with other wording can be detected.

        Returns:
            str: A hex digest of the system message and user template.
        """
        return hashlib.sha256(f"{self.system_prompt}\n{USER_TEMPLATE}".encode()).hexdigest()

    def user_prompt(self, text: str) -> str:
        """Render the user message for a subtitle line.

        Args:
            text (str): The subtitle line.

        Returns:
            str: A user message.
        """
        return USER_TEMPLATE.format(text=text)

    def target(self, label_names: list[str]) -> str:
        """Render the assistant message for a label set.

        Args:
            label_names (list[str]): The label names, already in the fixed order.

        Returns:
            str: A assistant message.
        """
        return json.dumps(label_names, ensure_ascii=False)

    def conversation(self, row: dict) -> dict:
        """Render a canonical record as a conversational training example.

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

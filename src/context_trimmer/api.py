import sys
from typing import List, Optional, Dict, Any

from context_trimmer.core import (
    Message,
    CompressionResult,
    compress_conversation,
    validate_messages,
    dry_run_stats,
    Role,
)


class Session:
    def __init__(self, messages: Optional[List[Message]] = None):
        self._messages: List[Message] = []
        if messages:
            self._messages = messages.copy()

    @classmethod
    def from_dicts(cls, messages_dicts: List[Dict[str, Any]]) -> "Session":
        messages = [Message(role=m["role"], content=m["content"]) for m in messages_dicts]
        return cls(messages)

    def to_dicts(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def get_messages(self) -> List[Message]:
        return self._messages.copy()

    def add_message(self, role: str, content: str) -> None:
        if role not in [r.value for r in Role]:
            raise ValueError(f"Invalid role: {role}")

        if role == Role.SYSTEM.value:
            if self._messages and self._messages[0].role == Role.SYSTEM.value:
                raise ValueError("System message already exists")
            self._messages.insert(0, Message(role=role, content=content))
        else:
            self._messages.append(Message(role=role, content=content))

    def add_turn(self, user_content: str, assistant_content: str) -> None:
        self.add_message(Role.USER.value, user_content)
        self.add_message(Role.ASSISTANT.value, assistant_content)

    def validate(self) -> None:
        validate_messages(self._messages)

    def compress(self, budget: int, output_stderr: bool = True) -> CompressionResult:
        self.validate()
        result = compress_conversation(self._messages, budget)

        if output_stderr:
            for line in result.stderr_output:
                print(line, file=sys.stderr)

        self._messages = result.messages
        return result

    def dry_run(self, budget: int) -> Dict[str, Any]:
        self.validate()
        return dry_run_stats(self._messages, budget)

    def current_bytes(self) -> int:
        non_system = [
            m for m in self._messages
            if m.role != Role.SYSTEM.value
        ]
        return sum(m.content_bytes() for m in non_system)

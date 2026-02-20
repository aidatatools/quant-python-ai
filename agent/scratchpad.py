class Scratchpad:
    """Agent memory scratchpad for accumulating research data and conversation history."""

    def __init__(self):
        self.messages: list[dict] = []
        self._items: list = []

    def add_message(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        msg: dict = {"role": role}

        # assistant messages MUST always have "content" key (even if None)
        if role == "assistant":
            msg["content"] = content or ""

            # Serialize tool_calls from SDK objects to plain dicts
            if tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
        elif role == "tool":
            # Tool result messages require tool_call_id and content
            msg["content"] = content or ""
            if tool_call_id:
                msg["tool_call_id"] = tool_call_id
        else:
            # user / system
            if content is not None:
                msg["content"] = content

        self.messages.append(msg)

    def add_item(self, data: dict) -> None:
        """Store raw data items (backwards compatibility or special logging)."""
        self._items.append(data)

    def get_items(self, item_type: str | None = None) -> list:
        if not item_type:
            return self._items
        return [i for i in self._items if i.get("type") == item_type]

    def summary(self) -> str:
        return "\n".join(str(m) for m in self.messages)

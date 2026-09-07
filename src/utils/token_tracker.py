import dataclasses
from collections import defaultdict


@dataclasses.dataclass
class CallRecord:
    role: str
    hop: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self):
        return self.prompt_tokens + self.completion_tokens


class TokenTracker:
    def __init__(self):
        self.calls = []

    def log(self, role, hop, prompt_tokens, completion_tokens):
        self.calls.append(CallRecord(role, hop, prompt_tokens, completion_tokens))

    def total_tokens(self, role=None):
        return sum(c.total for c in self.calls if role is None or c.role == role)

    def breakdown_by_role(self):
        out = defaultdict(lambda: {"calls": 0, "tokens": 0})
        for c in self.calls:
            out[c.role]["calls"] += 1
            out[c.role]["tokens"] += c.total
        return dict(out)

    def to_dict(self):
        return {
            "total_tokens": self.total_tokens(),
            "total_calls": len(self.calls),
            "by_role": self.breakdown_by_role(),
            "calls": [dataclasses.asdict(c) for c in self.calls],
        }
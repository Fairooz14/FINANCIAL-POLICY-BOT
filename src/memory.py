import re

PRONOUNS = set("it they that those he she this these there them their its his her what about more details".split())

class ChatMemory:
    def __init__(self, k: int = 6):
        self.k = k
        self.turns = []

    def add(self, user: str, bot: str):
        self.turns.append({"user": user, "bot": bot})
        if len(self.turns) > self.k:
            self.turns = self.turns[-self.k:]

    def summarize_topic(self) -> str:
        if not self.turns:
            return ""
        last_q = self.turns[-1]["user"].lower()
        toks = re.findall(r"[a-zA-Z][a-zA-Z\\-]+", last_q)
        toks = [t for t in toks if t not in PRONOUNS and len(t) > 2]
        uniq = []
        for t in toks:
            if t not in uniq:
                uniq.append(t)
        return " ".join(uniq[:8])

    def maybe_augment(self, query: str) -> str:
        q = query.strip()
        if not self.turns:
            return q
        short = len(q.split()) <= 6
        has_pronoun = any(p in q.lower().split() for p in PRONOUNS)
        if short or has_pronoun:
            topic = self.summarize_topic()
            if topic:
                return f"{q} (context: {topic})"
        return q

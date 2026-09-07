import dataclasses
import re

from src.agents.prompts import build_prompt
from src.tools.wiki_search import WikiEnv
from src.utils.llm_client import LLMClient
from src.utils.token_tracker import TokenTracker

ACTION_RE = re.compile(r"(\w+)\[(.*?)\]", re.DOTALL)


@dataclasses.dataclass
class HopRecord:
    hop: int
    thought: str
    action_type: str
    action_arg: str
    observation: str
    was_badcall: bool = False


@dataclasses.dataclass
class TrajectoryResult:
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    hops: list
    stopped_reason: str
    n_calls: int
    n_badcalls: int
    token_usage: dict


class ReActAgent:
    def __init__(self, llm: LLMClient, max_hops=3):
        self.llm = llm
        self.max_hops = max_hops

    def run(self, question_id, question, gold_answer):
        env = WikiEnv()
        tracker = TokenTracker()
        trajectory_text = ""
        hops = []
        predicted_answer = None
        stopped_reason = "max_hops"
        n_calls = 0
        n_badcalls = 0

        for hop_idx in range(1, self.max_hops + 1):
            prompt = build_prompt(question, trajectory_text) + f"Thought {hop_idx}:"
            response = self.llm.complete(prompt, stop=[f"\nObservation {hop_idx}:"])
            tracker.log("agent", hop_idx, response.prompt_tokens, response.completion_tokens)
            n_calls += 1

            generated = response.text.strip()
            was_badcall = False

            try:
                thought, action_str = generated.split(f"\nAction {hop_idx}:", 1)
                thought, action_str = thought.strip(), action_str.strip()
            except ValueError:
                # model ran thought and action together, ask again for just the action
                was_badcall = True
                n_badcalls += 1
                n_calls += 1
                thought = generated.split("\n")[0].strip()
                retry_prompt = build_prompt(question, trajectory_text) + f"Thought {hop_idx}: {thought}\nAction {hop_idx}:"
                retry = self.llm.complete(retry_prompt, stop=["\n"])
                tracker.log("agent", hop_idx, retry.prompt_tokens, retry.completion_tokens)
                action_str = retry.text.strip()

            action_match = ACTION_RE.search(action_str)
            if not action_match:
                stopped_reason = "parse_error"
                hops.append(HopRecord(hop_idx, thought, "PARSE_ERROR", "",
                                       f"could not parse action from: {action_str!r}", was_badcall))
                break

            action_type = action_match.group(1).strip().capitalize()
            action_arg = action_match.group(2).strip()

            if action_type == "Finish":
                predicted_answer = action_arg
                stopped_reason = "finished"
                hops.append(HopRecord(hop_idx, thought, action_type, action_arg, "", was_badcall))
                break

            if action_type == "Search":
                observation = env.search(action_arg)
            elif action_type == "Lookup":
                observation = env.lookup(action_arg)
            else:
                observation = f"unrecognized action type: {action_type}"

            hops.append(HopRecord(hop_idx, thought, action_type, action_arg, observation, was_badcall))
            trajectory_text += f"Thought {hop_idx}: {thought}\nAction {hop_idx}: {action_type}[{action_arg}]\nObservation {hop_idx}: {observation}\n"

        return TrajectoryResult(question_id, question, gold_answer, predicted_answer,
                                 hops, stopped_reason, n_calls, n_badcalls, tracker.to_dict())
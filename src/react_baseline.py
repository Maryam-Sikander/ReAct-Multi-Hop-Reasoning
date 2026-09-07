import json
import random
import re
from pathlib import Path

from datasets import load_dataset

from models import HFLocalClient
from prompts import build_prompt
from wikipedia_tool import WikiEnv

AGENT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_HOPS = 7
RETRY_BADCALLS = False
N_QUESTIONS = 150
SEED = 42
OUT_PATH = Path("results/results_baseline.jsonl")
ACTION_RE = re.compile(r"(\w+)\[(.*?)\]", re.DOTALL)

class ReActAgent:
    def __init__(self, llm, max_hops=MAX_HOPS, retry_badcalls=RETRY_BADCALLS):
        self.llm = llm
        self.max_hops = max_hops
        self.retry_badcalls = retry_badcalls

    def run(self, question_id, question, gold_answer):
        env = WikiEnv()
        trajectory_text = ""
        hops = []
        predicted_answer = None
        stopped_reason = "max_hops"
        n_calls = 0
        n_badcalls = 0
        total_tokens = 0

        for hop_idx in range(1, self.max_hops + 1):
            prompt = build_prompt(question, trajectory_text) + f"Thought {hop_idx}:"
            response = self.llm.complete(prompt, stop=[f"\nObservation {hop_idx}:"])
            n_calls += 1
            total_tokens += response.prompt_tokens + response.completion_tokens

            generated = response.text.strip()
            was_badcall = False

            try:
                thought, action_str = generated.split(f"\nAction {hop_idx}:", 1)
                thought, action_str = thought.strip(), action_str.strip()
            except ValueError:
                was_badcall = True
                n_badcalls += 1
                thought = generated.split("\n")[0].strip()
                if not self.retry_badcalls:
                    stopped_reason = "parse_error"
                    hops.append({"hop": hop_idx, "thought": thought, "action_type": "PARSE_ERROR",
                                 "action_arg": "", "observation": "badcall, retry disabled", "badcall": True})
                    break
                n_calls += 1
                retry_prompt = build_prompt(question, trajectory_text) + f"Thought {hop_idx}: {thought}\nAction {hop_idx}:"
                retry = self.llm.complete(retry_prompt, stop=["\n"])
                total_tokens += retry.prompt_tokens + retry.completion_tokens
                action_str = retry.text.strip()

            action_match = ACTION_RE.search(action_str)
            if not action_match:
                stopped_reason = "parse_error"
                hops.append({"hop": hop_idx, "thought": thought, "action_type": "PARSE_ERROR",
                             "action_arg": "", "observation": f"could not parse: {action_str!r}", "badcall": was_badcall})
                break

            action_type = action_match.group(1).strip().capitalize()
            action_arg = action_match.group(2).strip()

            if action_type == "Finish":
                predicted_answer = action_arg
                stopped_reason = "finished"
                hops.append({"hop": hop_idx, "thought": thought, "action_type": action_type,
                             "action_arg": action_arg, "observation": "", "badcall": was_badcall})
                break

            if action_type == "Search":
                observation = env.search(action_arg)
            elif action_type == "Lookup":
                observation = env.lookup(action_arg)
            else:
                observation = f"unrecognized action type: {action_type}"

            hops.append({"hop": hop_idx, "thought": thought, "action_type": action_type,
                         "action_arg": action_arg, "observation": observation, "badcall": was_badcall})
            trajectory_text += f"Thought {hop_idx}: {thought}\nAction {hop_idx}: {action_type}[{action_arg}]\nObservation {hop_idx}: {observation}\n"

        # if max_hops is exhausted without a Finish, predicted_answer stays None -
        # equivalent in effect to the paper's forced empty finish[] call: no answer,
        # scored as incorrect, without silently guessing one on the model's behalf
        return {
            "question_id": question_id, "question": question, "gold_answer": gold_answer,
            "predicted_answer": predicted_answer, "hops": hops, "trajectory_text": trajectory_text,
            "stopped_reason": stopped_reason,
            "n_calls": n_calls, "n_badcalls": n_badcalls, "total_tokens": total_tokens,
        }


# ---------- dataset: random sample from the full validation set, no difficulty filter ----------

def sample_hotpotqa(n=N_QUESTIONS, seed=SEED):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)
    sample = rng.sample(list(ds), min(n, len(ds)))
    return [{"id": q["id"], "question": q["question"], "answer": q["answer"]} for q in sample]


# ---------- resumable batch runner ----------

def load_done_ids(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["question_id"] for line in f if line.strip()}


def run_batch(agent, questions, out_path):
    done_ids = load_done_ids(out_path)
    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"{len(done_ids)} already done, {len(remaining)} remaining")

    with open(out_path, "a") as f:
        for i, q in enumerate(remaining):
            result = agent.run(q["id"], q["question"], q["answer"])
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(remaining)}] {q['id']} -> {result['predicted_answer']}")

    print(f"done. {len(load_done_ids(out_path))}/{len(questions)} total complete in {out_path}")


if __name__ == "__main__":
    agent_llm = HFLocalClient(AGENT_MODEL_ID)
    agent = ReActAgent(llm=agent_llm)
    questions = sample_hotpotqa()
    print(f"sampled {len(questions)} questions")
    run_batch(agent, questions, OUT_PATH)
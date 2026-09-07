import json
import re
from pathlib import Path
from models import HFLocalClient
from prompts import build_prompt,build_judge_prompt
from react_baseline import MAX_HOPS,RETRY_BADCALLS,ReActAgent


AGENT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
JUDGE_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

AGENT_MAX_NEW_TOKENS = 120
JUDGE_MAX_NEW_TOKENS = 80

TEMPERATURE = 0.0

OUTPUT_PATH = Path(
    "results_judge_intervention.jsonl"
)

import requests

SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+"
)

WIKI_API = "https://en.wikipedia.org/w/api.php"


class WikiEnv:

    def __init__(self, intro_sentences=5):

        self.intro_sentences = intro_sentences

        self._sentences = []
        self._lookup_keyword = None
        self._lookup_pos = 0

        self.headers = {
            "User-Agent": "ReActJudgeExperiment/1.0"
        }

        self._search_cache = {}

    def search(self, entity):

        if entity in self._search_cache:

            cached = self._search_cache[entity]

            self._sentences = cached["sentences"]
            self._lookup_keyword = None
            self._lookup_pos = 0

            return cached["intro"]

        page = self._fetch_extract(entity)

        if page is None:

            similar = self._fetch_suggestions(entity)

            self._sentences = []
            self._lookup_keyword = None

            result = (
                f"Could not find [{entity}]. "
                f"Similar: {similar}"
                if similar
                else
                f"Could not find [{entity}]."
            )

            self._search_cache[entity] = {
                "sentences": [],
                "intro": result,
            }

            return result

        title, extract = page

        self._sentences = [
            s
            for s in SENTENCE_SPLIT.split(extract)
            if s.strip()
        ]

        self._lookup_keyword = None
        self._lookup_pos = 0

        intro = (
            " ".join(
                self._sentences[
                    :self.intro_sentences
                ]
            )
            or
            f"[{title}] found but has no readable text."
        )

        self._search_cache[entity] = {
            "sentences": self._sentences,
            "intro": intro,
        }

        return intro

    def lookup(self, keyword):

        if not self._sentences:
            return "No page open. Use Search first."

        if keyword != self._lookup_keyword:

            self._lookup_keyword = keyword
            self._lookup_pos = 0

        matches = [
            i
            for i, s in enumerate(self._sentences)
            if keyword.lower() in s.lower()
        ]

        if not matches:
            return (
                f"No mentions of [{keyword}] found."
            )

        remaining = [
            i
            for i in matches
            if i >= self._lookup_pos
        ]

        if not remaining:
            return (
                f"No more results for [{keyword}]."
            )

        idx = remaining[0]

        self._lookup_pos = idx + 1

        result_num = matches.index(idx) + 1

        return (
            f"(Result {result_num}/{len(matches)}) "
            f"{self._sentences[idx]}"
        )

    def _fetch_extract(self, title):

        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }

        resp = requests.get(
            WIKI_API,
            params=params,
            headers=self.headers,
            timeout=15,
        )

        resp.raise_for_status()

        pages = (
            resp.json()
            .get("query", {})
            .get("pages", {})
        )

        for page_id, page in pages.items():

            if (
                page_id == "-1"
                or "missing" in page
            ):
                return None

            extract = page.get(
                "extract",
                "",
            )

            if extract.strip():

                return (
                    page.get("title", title),
                    extract,
                )

        return None

    def _fetch_suggestions(
        self,
        query,
        limit=5,
    ):

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }

        resp = requests.get(
            WIKI_API,
            params=params,
            headers=self.headers,
            timeout=15,
        )

        resp.raise_for_status()

        results = (
            resp.json()
            .get("query", {})
            .get("search", [])
        )

        return ", ".join(
            r["title"]
            for r in results
        )

def run_judge(
    judge_llm,
    question,
    trajectory,
    candidate_answer=None,
):

    prompt = build_judge_prompt(
        question,
        trajectory,
        candidate_answer,
    )

    response = judge_llm.complete(
        prompt,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
    )

    match = JUDGE_DECISION_RE.search(
        response.text
    )

    decision = (
        match.group(1).upper()
        if match
        else "CONTINUE"
    )

    feedback = response.text.strip()

    if decision == "ANSWER":
        feedback = ""

    return {
        "decision": decision,
        "feedback": feedback,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
    }

ACTION_RE = re.compile(
    r"(\w+)\[(.*?)\]",
    re.DOTALL,
)

class ReActJudgeAgent:

    def __init__(
        self,
        agent_llm,
        judge_llm,
        max_hops=7,
    ):

        self.agent_llm = agent_llm
        self.judge_llm = judge_llm
        self.max_hops = max_hops

    def run(
        self,
        question_id,
        question,
        gold_answer,
    ):

        env = WikiEnv()

        trajectory_text = ""

        hops = []

        predicted_answer = None

        stopped_reason = "max_hops"

        agent_calls = 0
        judge_calls = 0

        agent_tokens = 0
        judge_tokens = 0

        force_finish = False

        for hop_idx in range(
            1,
            self.max_hops + 1,
        ):

            prompt = (
                build_react_prompt(
                    question,
                    trajectory_text,
                    force_finish=force_finish,
                )
                + f"\nThought {hop_idx}:"
            )

            response = self.agent_llm.complete(
                prompt,
                stop=[
                    f"\nObservation {hop_idx}:"
                ],
                max_new_tokens=AGENT_MAX_NEW_TOKENS,
            )

            agent_calls += 1

            agent_tokens += (
                response.prompt_tokens
                + response.completion_tokens
            )

            generated = response.text.strip()
            try:

                thought, action_str = (
                    generated.split(
                        f"\nAction {hop_idx}:",
                        1,
                    )
                )

                thought = thought.strip()
                action_str = action_str.strip()

            except ValueError:

                stopped_reason = "parse_error"

                hops.append({
                    "hop": hop_idx,
                    "thought": generated,
                    "action_type": "PARSE_ERROR",
                    "action_arg": "",
                    "observation": "",
                    "judge_decision": None,
                    "judge_feedback": None,
                })

                break

            action_match = ACTION_RE.search(
                action_str
            )

            if not action_match:

                stopped_reason = "parse_error"

                hops.append({
                    "hop": hop_idx,
                    "thought": thought,
                    "action_type": "PARSE_ERROR",
                    "action_arg": action_str,
                    "observation": "",
                    "judge_decision": None,
                    "judge_feedback": None,
                })

                break

            action_type = (
                action_match
                .group(1)
                .strip()
                .capitalize()
            )

            action_arg = (
                action_match
                .group(2)
                .strip()
            )

            if force_finish and action_type != "Finish":

                trajectory_text += (
                    f"Thought {hop_idx}: "
                    f"{thought}\n"
                    f"Action {hop_idx}: "
                    f"{action_type}[{action_arg}]\n"
                )

                hops.append({
                    "hop": hop_idx,
                    "thought": thought,
                    "action_type": action_type,
                    "action_arg": action_arg,
                    "observation": "",
                    "judge_decision": "FORCED_FINISH",
                    "judge_feedback": (
                        "The judge requested finalization."
                    ),
                })
                trajectory_text += (
                    "Judge: Finalize using the evidence "
                    "already collected. Do not search further.\n"
                )

                force_finish = True
                continue

        
            if action_type == "Search":

                observation = env.search(
                    action_arg
                )

            elif action_type == "Lookup":

                observation = env.lookup(
                    action_arg
                )

            elif action_type == "Finish":

                observation = ""

            else:

                observation = (
                    f"Unrecognized action: "
                    f"{action_type}"
                )

            trajectory_text += (
                f"Thought {hop_idx}: "
                f"{thought}\n"
                f"Action {hop_idx}: "
                f"{action_type}[{action_arg}]\n"
            )

            if observation:

                trajectory_text += (
                    f"Observation {hop_idx}: "
                    f"{observation}\n")

            candidate_answer = None

            if action_type == "Finish":

                candidate_answer = action_arg
            should_judge = (
                action_type in {
                    "Search",
                    "Lookup",
                    "Finish",
                }
            )

            judge_result = None

            if should_judge:

                judge_result = run_judge(
                    self.judge_llm,
                    question,
                    trajectory_text,
                    candidate_answer,
                )

                judge_calls += 1

                judge_tokens += (
                    judge_result["prompt_tokens"]
                    + judge_result["completion_tokens"]
                )

            hop_record = {
                "hop": hop_idx,
                "thought": thought,
                "action_type": action_type,
                "action_arg": action_arg,
                "observation": observation,
                "judge_decision": (
                    judge_result["decision"]
                    if judge_result
                    else None
                ),
                "judge_feedback": (
                    judge_result["feedback"]
                    if judge_result
                    else None
                ),
            }

            hops.append(hop_record)
            if action_type == "Finish":

                if (
                    judge_result
                    and
                    judge_result["decision"]
                    == "ANSWER"
                ):

                    predicted_answer = (
                        candidate_answer
                    )

                    stopped_reason = (
                        "judge_accepted"
                    )

                    break

                # Judge rejected the candidate answer.
                # Continue ReAct.

                feedback = (
                    judge_result["feedback"]
                    if judge_result
                    else
                    "Continue reasoning."
                )

                trajectory_text += (
                    f"Judge: {feedback}\n"
                )

                force_finish = False

                continue

            # ------------------------------------------------
            # JUDGE CONTINUE
            # ------------------------------------------------

            if (
                judge_result
                and
                judge_result["decision"]
                == "CONTINUE"
            ):

                trajectory_text += (
                    f"Judge: "
                    f"{judge_result['feedback']}\n"
                )

                force_finish = False

        
            elif (
                judge_result
                and
                judge_result["decision"]
                == "ANSWER"
            ):

                # Evidence is sufficient.
                # Next ReAct step must finalize.
                trajectory_text += (
                    "Judge: Evidence is sufficient. "
                    "Finalize the answer on the next step.\n"
                )

                force_finish = True
        return {
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "hops": hops,
            "trajectory_text": trajectory_text,
            "stopped_reason": stopped_reason,
            "agent_calls": agent_calls,
            "judge_calls": judge_calls,
            "agent_tokens": agent_tokens,
            "judge_tokens": judge_tokens,
            "total_calls": (
                agent_calls
                + judge_calls
            ),
            "total_tokens": (
                agent_tokens
                + judge_tokens
            ),
        }

if __name__ == "__main__":
    print("Loading agent model...")

    agent_llm = HFLocalClient(
        AGENT_MODEL_ID,
        max_new_tokens=AGENT_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,)

    print("Loading judge model...")

    judge_llm = HFLocalClient(
        JUDGE_MODEL_ID,
        max_new_tokens=JUDGE_MAX_NEW_TOKENS,
        temperature=TEMPERATURE,)
    agent = ReActJudgeAgent(
        agent_llm=agent_llm,
        judge_llm=judge_llm,
        max_hops=MAX_HOPS,)
    run_batch(
        agent,
        questions,
        OUTPUT_PATH,
    )
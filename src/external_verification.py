import json
import re
from pathlib import Path

from models import HFLocalClient
from prompts import build_blind_prompt, build_supported_check_prompt
from react_baseline import MAX_HOPS, RETRY_BADCALLS, ReActAgent

AGENT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
VERIFIER_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
VERIFIER_MODE = "supported_check"  # "supported_check" or "blind"
VERIFIER_MAX_NEW_TOKENS = 250

OUT_PATH = Path(f"results/results_verified_{VERIFIER_MODE}.jsonl")

VERDICT_RE = re.compile(r"Verdict:\s*(SUPPORTED|UNSUPPORTED|PARTIALLY_SUPPORTED)", re.IGNORECASE)
REASON_RE = re.compile(r"Reason:\s*(.*)", re.DOTALL)
BLIND_ANSWER_RE = re.compile(r"Answer:\s*(.*?)(?:\n|$)")


def normalize(text):
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def verify_supported_check(verifier_llm, question, trajectory_text, agent_answer):
    prompt = build_supported_check_prompt(question, trajectory_text, agent_answer)
    response = verifier_llm.complete(prompt, max_new_tokens=VERIFIER_MAX_NEW_TOKENS)
    verdict_match = VERDICT_RE.search(response.text)
    reason_match = REASON_RE.search(response.text)
    return {
        "verdict": verdict_match.group(1).upper() if verdict_match else "UNPARSED",
        "reason": reason_match.group(1).strip() if reason_match else response.text.strip(),
        "verifier_prompt_tokens": response.prompt_tokens,
        "verifier_completion_tokens": response.completion_tokens,
    }


def verify_blind(verifier_llm, question, trajectory_text, agent_answer):
    prompt = build_blind_prompt(question, trajectory_text)
    response = verifier_llm.complete(prompt, max_new_tokens=VERIFIER_MAX_NEW_TOKENS)
    answer_match = BLIND_ANSWER_RE.search(response.text)
    verifier_answer = answer_match.group(1).strip() if answer_match else None
    reason_match = REASON_RE.search(response.text)

    if verifier_answer is None or "insufficient" in verifier_answer.lower():
        verdict = "INSUFFICIENT_EVIDENCE"
    elif normalize(verifier_answer) == normalize(agent_answer):
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"

    return {
        "verdict": verdict,
        "verifier_independent_answer": verifier_answer,
        "reason": reason_match.group(1).strip() if reason_match else response.text.strip(),
        "verifier_prompt_tokens": response.prompt_tokens,
        "verifier_completion_tokens": response.completion_tokens,
    }


def run_verified_react(agent, verifier_llm, question_id, question, gold_answer, mode=VERIFIER_MODE):
    result = agent.run(question_id, question, gold_answer)

    if mode == "blind":
        verification = verify_blind(
            verifier_llm, question, result["trajectory_text"], result["predicted_answer"]
        )
    else:
        verification = verify_supported_check(
            verifier_llm, question, result["trajectory_text"], result["predicted_answer"]
        )

    return {
        "question_id": question_id,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": result["predicted_answer"],
        "stopped_reason": result["stopped_reason"],
        "agent_calls": result["n_calls"],
        "agent_badcalls": result["n_badcalls"],
        "agent_tokens": result["total_tokens"],
        "verification_mode": mode,
        **verification,
        "total_calls": result["n_calls"] + 1,
    }


def load_questions_from_baseline(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            questions.append({
                "id": result["question_id"],
                "question": result["question"],
                "answer": result["gold_answer"],
            })
    return questions


def load_done_ids(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["question_id"] for line in f if line.strip()}


def run_batch(agent, verifier_llm, questions, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(out_path)
    remaining = [q for q in questions if q["id"] not in done_ids]
    print(f"{len(done_ids)} already done, {len(remaining)} remaining")

    with open(out_path, "a", encoding="utf-8") as f:
        for i, q in enumerate(remaining):
            result = run_verified_react(
                agent, verifier_llm, q["id"], q["question"], q["answer"]
            )
            f.write(json.dumps(result) + "\n")
            f.flush()
            print(
                f"[{i + 1}/{len(remaining)}] {q['id']} -> "
                f"{result['predicted_answer']} [{result['verdict']}]"
            )

    print(
        f"done. {len(load_done_ids(out_path))}/{len(questions)} "
        f"total complete in {out_path}"
    )


if __name__ == "__main__":
    baseline_path = Path("results/baseline/results_baseline.jsonl")
    questions = load_questions_from_baseline(baseline_path)
    print(f"loaded {len(questions)} questions from baseline results")

    agent_llm = HFLocalClient(AGENT_MODEL_ID)
    verifier_llm = HFLocalClient(
        VERIFIER_MODEL_ID,
        max_new_tokens=VERIFIER_MAX_NEW_TOKENS,
    )
    agent = ReActAgent(
        llm=agent_llm,
        max_hops=MAX_HOPS,
        retry_badcalls=RETRY_BADCALLS,
    )

    run_batch(agent, verifier_llm, questions, OUT_PATH)
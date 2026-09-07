"""Prompts used by the ReAct agent and external verifier."""

INSTRUCTIONS = """Answer the question by interleaving numbered Thought, Action, and Observation steps.
Action is one of three types:
(1) Search[entity] - looks up the entity on Wikipedia, returns the opening paragraph if it exists, or similar titles if not.
(2) Lookup[keyword] - returns the next sentence on the current page containing keyword.
(3) Finish[answer] - gives the final answer and ends the task.
Here are some examples.
"""

FEWSHOT_EXAMPLES = """
Question: What is the nationality of the director of the 1994 film that won the Palme d'Or at Cannes?
Thought 1: I need to find which film won the Palme d'Or in 1994, then its director, then his nationality.
Action 1: Search[1994 Cannes Film Festival]
Observation 1: The 1994 Cannes Film Festival awarded the Palme d'Or to Pulp Fiction, directed by Quentin Tarantino.
Thought 2: The director is Quentin Tarantino. Now his nationality.
Action 2: Search[Quentin Tarantino]
Observation 2: Quentin Jerome Tarantino is an American film director, screenwriter, producer, and actor.
Thought 3: He's American.
Action 3: Finish[American]

Question: The university where the inventor of the World Wide Web earned his degree is in which city?
Thought 1: Find who invented the Web, then where he studied.
Action 1: Search[World Wide Web]
Observation 1: The World Wide Web was invented by English computer scientist Tim Berners-Lee.
Thought 2: Now find where he studied.
Action 2: Search[Tim Berners-Lee]
Observation 2: Tim Berners-Lee studied physics at The Queen's College, Oxford, graduating in 1976.
Thought 3: Oxford is both the college and the city.
Action 3: Finish[Oxford]
""".strip()


def build_prompt(question, trajectory):
    return f"{INSTRUCTIONS}\n\n{FEWSHOT_EXAMPLES}\n\nQuestion: {question}\n{trajectory}"


SUPPORTED_INSTRUCTIONS = """You are checking whether an answer to a multi-hop question is actually supported by the evidence an agent retrieved, or whether it goes beyond what that evidence shows.

You will see the question, the agent's full research trajectory (its reasoning, its Wikipedia searches, and what each search returned), and the agent's final answer.

Judge only whether the final answer is grounded in the observations shown - not whether you personally know it to be true from other knowledge. An answer can be factually correct yet still unsupported by this particular trajectory, if the evidence retrieved doesn't actually establish it.

Respond in exactly this format:
Verdict: <SUPPORTED, UNSUPPORTED, or PARTIALLY_SUPPORTED>
Reason: <one or two sentences>
"""


def build_supported_check_prompt(question, trajectory_text, agent_answer):
    return (
        f"{SUPPORTED_INSTRUCTIONS}\n\n"
        f"Question: {question}\n\n"
        f"Agent's trajectory:\n{trajectory_text}\n"
        f"Agent's final answer: {agent_answer or '(no answer given)'}\n"
    )


BLIND_INSTRUCTIONS = """You will see a question and a research trajectory (Wikipedia searches and what they returned) gathered by someone else while trying to answer it. You will NOT be shown their answer.

Based only on the evidence in this trajectory, work out the answer yourself. If the evidence isn't sufficient to answer confidently, say so explicitly rather than guessing.

Respond in exactly this format:
Answer: <your answer, or "INSUFFICIENT EVIDENCE" if the trajectory doesn't support one>
Reason: <one or two sentences>
"""


def build_blind_prompt(question, trajectory_text):
    return f"{BLIND_INSTRUCTIONS}\n\nQuestion: {question}\n\nTrajectory:\n{trajectory_text}\n"
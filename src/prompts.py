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

JUDGE_INSTRUCTIONS = """You are an external judge monitoring a ReAct agent solving a multi-hop question.

You will see the question and the agent's trajectory so far.

Determine whether the agent has gathered enough evidence to answer the question.

If the evidence is insufficient or another reasoning/tool-use step is needed, respond:

Decision: CONTINUE
Feedback: <briefly state what is missing>

If the evidence is sufficient to answer, respond:

Decision: ANSWER
Feedback: <briefly explain why the evidence is sufficient>

Do not provide the final answer yourself.
"""


def build_judge_prompt(question, trajectory):
    return (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"Question: {question}\n\n"
        f"Trajectory so far:\n{trajectory}\n"
    )

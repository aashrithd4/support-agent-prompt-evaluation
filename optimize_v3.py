"""Produce V3 via a real Bedrock optimization pass.

This is NOT the managed AgentCore Recommendations API (that requires GenAI child
spans we could not export in this environment). Instead it mirrors what an
optimizer does: it feeds the current prompt (V2) plus the REAL failure/weakness
cases from the recorded V1/V2 runs to a strong model (Claude Sonnet 4.5) and asks
for an improved system prompt. The output is a genuine model-generated prompt,
evaluated on the same 40-task set. Labeled accordingly in the article.
"""
import json, pathlib, boto3

RES = pathlib.Path(__file__).parent / "results"
PROMPTS = json.loads((pathlib.Path(__file__).parent / "prompts.json").read_text())
rt = boto3.client("bedrock-runtime", region_name="us-west-2")
OPT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def collect_weaknesses():
    """Pull real failing/weak cases from recorded V1 and V2 runs."""
    cases = []
    for label in ("v1", "v2"):
        f = RES / "runs" / f"{label}.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text())["rows"]:
            weak = (not r["task_success"]) or r["over_escalated"] or (r["grounded"] is False)
            if weak:
                cases.append({
                    "from": label, "id": r["id"], "category": r["category"],
                    "question": r["question"], "tools_used": r["used_tools"],
                    "over_escalated": r["over_escalated"], "grounded": r["grounded"],
                    "task_success": r["task_success"],
                    "judge_reason": r["judge_reason"],
                    "response_excerpt": r["response"][:300],
                })
    return cases


OPT_TEMPLATE = """You are optimizing the system prompt for an internal IT support agent.
The agent has three tools: lookup_kb_article, check_system_status, create_ticket.

CURRENT SYSTEM PROMPT (V2):
---
{v2}
---

Here are REAL cases where the agent (running this or a weaker prompt) underperformed,
taken from an evaluation run. Each shows what went wrong:

{cases}

Rewrite the system prompt to fix these specific weaknesses WITHOUT breaking the behaviors
that already work (grounding to KB content, using status tool for status questions, not
over-escalating routine how-to questions, clarifying vague requests, declining out-of-scope).

Constraints:
- Keep it concise and operational (rules the model can follow, not prose).
- Do not add new tools or capabilities.
- Do not make it dramatically longer than V2; tighten rather than pad.

Return ONLY the improved system prompt text, nothing else."""


def main():
    cases = collect_weaknesses()
    case_text = "\n".join(
        f"- [{c['from']}/{c['id']} {c['category']}] Q: {c['question']}\n"
        f"    tools={c['tools_used']} over_escalated={c['over_escalated']} grounded={c['grounded']} success={c['task_success']}\n"
        f"    problem: {c['judge_reason']}"
        for c in cases
    )
    prompt = OPT_TEMPLATE.format(v2=PROMPTS["v2"], cases=case_text or "(no explicit failures; tighten for robustness)")

    r = rt.converse(
        modelId=OPT_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1200, "temperature": 0.2},
    )
    v3 = r["output"]["message"]["content"][0]["text"].strip()

    # save V3 into prompts.json and a standalone file
    PROMPTS["v3"] = v3
    (pathlib.Path(__file__).parent / "prompts.json").write_text(json.dumps(PROMPTS, indent=2))
    (RES / "v3_prompt.txt").write_text(v3)
    (RES / "v3_optimization_input.json").write_text(json.dumps(
        {"model": OPT_MODEL, "num_weakness_cases": len(cases), "cases": cases}, indent=2))

    print(f"Used {len(cases)} real weakness cases.")
    print("=== V3 PROMPT ===")
    print(v3)


if __name__ == "__main__":
    main()

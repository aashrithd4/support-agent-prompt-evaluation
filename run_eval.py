"""Evaluation harness for the IT Support Assistant.

Runs the development dataset against a given system prompt using REAL Bedrock calls.
- Tool usage is captured objectively from the Strands agent (not the judge).
- Task success + groundedness are scored by an LLM judge (Claude Sonnet 4.5).
- Latency and token usage are measured from real responses.

Usage:
    python run_eval.py <version_label> <prompt_source>
      prompt_source: 'v1' | 'v2' | path to a .txt file (for V3 recommendation)

Results are written to results/runs/<version_label>.json
"""
import json
import sys
import time
import pathlib
import boto3
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from strands.hooks.events import BeforeToolCallEvent

ROOT = pathlib.Path(__file__).parent
RESULTS = pathlib.Path(__file__).parent / "results"
DATASET = json.loads((RESULTS / "dataset.json").read_text())["tasks"]
PROMPTS = json.loads((ROOT / "prompts.json").read_text())

AGENT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
JUDGE_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
REGION = "us-west-2"

rt = boto3.client("bedrock-runtime", region_name=REGION)

# --- Tools (same fakes as agent.py) ------------------------------------------
_KB = {
    "password reset": "To reset your password, go to sso.corp/reset, verify via MFA, and set a new password. Passwords expire every 90 days.",
    "vpn": "Install the VPN client from software.corp/vpn. Connect using your SSO credentials. If it hangs at 'authenticating', restart the client.",
    "email setup": "Add your account in Outlook using autodiscover: enter your corp email and SSO password. Mobile: use the Company Portal app.",
    "printer": "Add a printer via print.corp/add, choose your building/floor, and install the driver it offers.",
    "mfa": "Register MFA at sso.corp/mfa using the Authenticator app. You can register up to 3 devices.",
    "laptop encryption": "Disk encryption is enabled by default. To check status, run the Security app; a green shield means encrypted.",
    "software install": "Request software from software.corp/catalog. Approved apps install automatically; others need manager approval.",
}
_STATUS = {"vpn": "operational", "email": "degraded", "sso": "operational", "wifi": "operational", "vdi": "outage"}


@tool
def lookup_kb_article(query: str) -> str:
    """Search the internal IT knowledge base for a help article matching the query."""
    q = query.lower()
    for key, article in _KB.items():
        if key in q or any(w in q for w in key.split()):
            return f"[KB:{key}] {article}"
    return "No knowledge base article found for that query."


@tool
def check_system_status(system: str) -> str:
    """Check the current operational status of an internal system (vpn, email, sso, wifi, vdi)."""
    s = system.lower().strip()
    for key, status in _STATUS.items():
        if key in s:
            return f"System '{key}' is currently: {status}."
    return f"Unknown system '{system}'. Known systems: {', '.join(_STATUS)}."


@tool
def create_ticket(summary: str, priority: str = "medium") -> str:
    """Create an IT support ticket. Use only when self-service fails or the user explicitly asks to escalate."""
    tid = f"INC-{abs(hash(summary)) % 90000 + 10000}"
    return f"Created ticket {tid} (priority={priority}): {summary}"


JUDGE_TEMPLATE = """You are grading an IT support assistant's response to a user.

USER QUESTION:
{question}

GRADING NOTES (what a correct handling looks like):
{notes}

KB CONTENT THE AGENT HAD ACCESS TO (if any tool returned it):
{tool_output}

ASSISTANT RESPONSE:
{response}

Grade strictly and return ONLY compact JSON:
{{"task_success": true/false, "grounded": true/false/null, "reason": "<one sentence>"}}

- task_success: did the response correctly handle the question per the grading notes?
- grounded: for how-to answers, is every concrete step/URL/number supported by the KB content shown above? Use null if grounding does not apply (e.g. status or out-of-scope questions).
"""


def judge(question, notes, tool_output, response):
    body = {
        "messages": [{"role": "user", "content": [{"text": JUDGE_TEMPLATE.format(
            question=question, notes=notes, tool_output=tool_output or "(none)", response=response)}]}],
        "inferenceConfig": {"maxTokens": 300, "temperature": 0},
    }
    r = rt.converse(modelId=JUDGE_MODEL, **body)
    txt = r["output"]["message"]["content"][0]["text"].strip()
    start, end = txt.find("{"), txt.rfind("}")
    return json.loads(txt[start:end + 1])


def run_task(system_prompt, task):
    used_tools = []
    tool_outputs = []

    def capture(event: BeforeToolCallEvent):
        name = getattr(getattr(event, "tool_use", None), "name", None) or \
            (event.tool_use.get("name") if isinstance(getattr(event, "tool_use", None), dict) else None)
        if name:
            used_tools.append(name)

    agent = Agent(
        model=BedrockModel(model_id=AGENT_MODEL, temperature=0),
        system_prompt=system_prompt,
        tools=[lookup_kb_article, check_system_status, create_ticket],
    )
    agent.hooks.add_callback(BeforeToolCallEvent, capture)

    t0 = time.time()
    result = agent(task["question"])
    latency = round(time.time() - t0, 2)

    response_text = str(result)
    # token usage from Strands result metrics
    tokens = 0
    try:
        m = result.metrics.accumulated_usage
        tokens = int(m.get("inputTokens", 0)) + int(m.get("outputTokens", 0))
    except Exception:
        pass

    # collect any KB content the agent saw (for groundedness judging)
    for msg in getattr(agent, "messages", []):
        content = msg.get("content", []) if isinstance(msg, dict) else []
        for block in (content or []):
            if isinstance(block, dict) and "toolResult" in block:
                for c in block["toolResult"].get("content", []):
                    if "text" in c:
                        tool_outputs.append(c["text"])

    verdict = judge(task["question"], task["notes"], "\n".join(tool_outputs), response_text)

    # deterministic tool-use correctness
    used_set = set(used_tools)
    expected = set(task["expected_tools"])
    forbidden = set(task["forbidden_tools"])
    called_forbidden = bool(used_set & forbidden)
    called_expected = expected.issubset(used_set) if expected else True
    tool_correct = called_expected and not called_forbidden
    over_escalated = ("create_ticket" in used_set) and (not task["should_escalate"])

    return {
        "id": task["id"], "category": task["category"], "question": task["question"],
        "used_tools": used_tools, "tool_correct": tool_correct,
        "called_forbidden": called_forbidden, "over_escalated": over_escalated,
        "task_success": bool(verdict.get("task_success")),
        "grounded": verdict.get("grounded"),
        "judge_reason": verdict.get("reason", ""),
        "latency_s": latency, "tokens": tokens,
        "response": response_text[:1200],
    }


def main():
    label, source = sys.argv[1], sys.argv[2]
    system_prompt = PROMPTS[source] if source in PROMPTS else pathlib.Path(source).read_text().strip()
    print(f"Running '{label}' with prompt from '{source}' ({len(system_prompt)} chars) over {len(DATASET)} tasks")

    rows = []
    for i, task in enumerate(DATASET, 1):
        row = run_task(system_prompt, task)
        rows.append(row)
        flag = "OK " if row["task_success"] else "XX "
        print(f"  [{i:2}/{len(DATASET)}] {flag}{row['id']:6} tools={row['used_tools']}")

    out = RESULTS / "runs"
    out.mkdir(exist_ok=True)
    (out / f"{label}.json").write_text(json.dumps({"label": label, "prompt_source": source, "rows": rows}, indent=2))
    print(f"Wrote {out / (label + '.json')}")


if __name__ == "__main__":
    main()

"""IT Support Assistant — Strands agent on AgentCore Runtime.

The system prompt is NOT hardcoded here. It is read at request time from an
AgentCore Configuration Bundle when building a new agent for each request. If no bundle is
in context (local run, no A/B active), it falls back to a safe default so the
agent always works.
"""
from strands import Agent, tool
from strands.models.bedrock import BedrockModel
from bedrock_agentcore.runtime import BedrockAgentCoreApp, BedrockAgentCoreContext

app = BedrockAgentCoreApp()

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_SYSTEM_PROMPT = "You are an IT support assistant."

# --- Deterministic fake tools (seeded corpus so runs are reproducible) -------

_KB = {
    "password reset": "To reset your password, go to sso.corp/reset, verify via MFA, and set a new password. Passwords expire every 90 days.",
    "vpn": "Install the VPN client from software.corp/vpn. Connect using your SSO credentials. If it hangs at 'authenticating', restart the client.",
    "email setup": "Add your account in Outlook using autodiscover: enter your corp email and SSO password. Mobile: use the Company Portal app.",
    "printer": "Add a printer via print.corp/add, choose your building/floor, and install the driver it offers.",
    "mfa": "Register MFA at sso.corp/mfa using the Authenticator app. You can register up to 3 devices.",
    "laptop encryption": "Disk encryption is enabled by default. To check status, run the Security app; a green shield means encrypted.",
    "software install": "Request software from software.corp/catalog. Approved apps install automatically; others need manager approval.",
}

_STATUS = {
    "vpn": "operational",
    "email": "degraded",
    "sso": "operational",
    "wifi": "operational",
    "vdi": "outage",
}


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


# --- Load configuration when constructing the agent for a request ---

def build_agent():
    """Build a fresh agent per request with the bundle's prompt applied."""
    try:
        config = BedrockAgentCoreContext.get_config_bundle() or {}
    except Exception:
        config = {}
    system_prompt = config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    model_id = config.get("model_id", DEFAULT_MODEL_ID)
    return Agent(
        model=BedrockModel(model_id=model_id),
        system_prompt=system_prompt,
        tools=[lookup_kb_article, check_system_status, create_ticket],
    )


@app.entrypoint
def invoke(payload, context=None):
    user_query = str(payload.get("prompt", "Hello"))
    agent = build_agent()
    result = agent(user_query)
    return {"response": str(result)}


if __name__ == "__main__":
    app.run()

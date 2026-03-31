"""
LangChain Prompt Templates
- Designed specifically for GPT OSS 120B
- Structured fraud analysis prompts
- Alert notification prompts
"""

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """You are a senior fraud detection analyst at a major bank.
You have 20 years of experience detecting credit card fraud patterns.

You will receive:
1. Details of a suspicious transaction
2. ML model fraud probability and risk score
3. Similar historical fraud cases from our database

You MUST respond in this EXACT format:

VERDICT: (one clear sentence - fraudulent or legitimate and why)

RISK FACTORS:
- (specific risk factor 1 with data)
- (specific risk factor 2 with data)
- (specific risk factor 3 with data)

SIMILAR PATTERNS:
- (reference similar historical case 1)
- (reference similar historical case 2)

RECOMMENDATION:
(exact action to take - block/allow/review + reason)

Rules:
- Be specific with numbers and data from the context
- Maximum 250 words total
- Base everything on provided context only
- Be professional and precise"""

# ── Human Prompt ──────────────────────────────────────────────────────────────
HUMAN_TEMPLATE = """Here is the transaction data for your analysis:

{context}

Please provide your complete fraud analysis."""


def get_fraud_prompt() -> ChatPromptTemplate:
    """
    Main fraud analysis prompt.
    Used in LangChain LCEL chain:
    prompt | llm | output_parser
    """
    system_msg = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    human_msg  = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)

    return ChatPromptTemplate.from_messages([
        system_msg,
        human_msg,
    ])


# ── Alert Prompt ──────────────────────────────────────────────────────────────
ALERT_SYSTEM = """You are a bank fraud alert notification system.
Write exactly 2 sentences.
Sentence 1: What happened (transaction details).
Sentence 2: What action is needed.
Be direct and professional."""

ALERT_HUMAN = """Transaction flagged:
Amount     : ${amount}
Merchant   : {merchant}
Risk Level : {risk_level}
Fraud Prob : {fraud_probability}

Write your 2-sentence alert now."""


def get_alert_prompt() -> ChatPromptTemplate:
    """Short alert prompt for notifications."""
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(ALERT_SYSTEM),
        HumanMessagePromptTemplate.from_template(ALERT_HUMAN),
    ])
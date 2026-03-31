"""
Test GPT OSS 120B LLM Explanation Engine
Full pipeline: RAG + LangChain + GPT OSS 120B via Groq
"""

import sys
sys.path.insert(0, '.')

print("="*60)
print("  TESTING GPT OSS 120B via GROQ")
print("  Full Pipeline: RAG + LangChain + LLM")
print("="*60)

# ── Step 1: Load LLM Engine ───────────────────────────────────
print("\n[1/4] Loading LLM Engine...")
from llm.llm_engine import llm_engine
print("      LLM Engine loaded!")

# ── Step 2: Sample high-risk transaction ──────────────────────
print("\n[2/4] Preparing test transaction...")
tx = {
    'amount'           : 2500.00,
    'merchant'         : 'Unknown Electronics',
    'category'         : 'electronics',
    'card_type'        : 'visa',
    'merchant_country' : 'US',
    'V1'               : -3.0,
    'V2'               :  2.5,
    'V3'               : -4.0,
    'V14'              : -8.0,
    'V17'              : -6.0,
    'V12'              : -5.0,
    'V10'              : -4.0,
}
print(f"      Amount   : ${tx['amount']:.2f}")
print(f"      Merchant : {tx['merchant']}")
print(f"      V14      : {tx['V14']} (key fraud indicator)")

# ── Step 3: Sample ML prediction ─────────────────────────────
print("\n[3/4] Using ML prediction...")
pred = {
    'fraud_probability' : 0.9712,
    'is_fraud'          : True,
    'risk_level'        : 'CRITICAL',
    'threshold_used'    : 0.45,
}
print(f"      Fraud Prob : {pred['fraud_probability']:.2%}")
print(f"      Risk Level : {pred['risk_level']}")
print(f"      Is Fraud   : {pred['is_fraud']}")

# ── Step 4: Generate Explanation ─────────────────────────────
print("\n[4/4] Calling GPT OSS 120B via Groq...")
print("      Please wait...\n")

result = llm_engine.explain(tx, pred, top_k=3)

# ── Print Results ─────────────────────────────────────────────
print("\n" + "="*60)
print("  GPT OSS 120B FRAUD EXPLANATION")
print("="*60)
print(result['explanation'])

print("\n" + "="*60)
print("  PIPELINE STATS")
print("="*60)
print(f"  Model Used    : {result['model_used']}")
print(f"  Provider      : {result['provider']}")
print(f"  Latency       : {result['latency_seconds']}s")
print(f"  RAG Cases     : {result['total_retrieved']}")
print(f"  Index Size    : {result['prediction']}")

# ── Test Alert Message ────────────────────────────────────────
print("\n" + "="*60)
print("  ALERT MESSAGE TEST")
print("="*60)
print("Generating short alert...")
alert = llm_engine.generate_alert(tx, pred)
print(f"\n{alert}")

print("\n" + "="*60)
print("  ALL TESTS PASSED!")
print("="*60)
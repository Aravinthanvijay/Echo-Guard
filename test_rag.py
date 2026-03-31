import sys
sys.path.insert(0, '.')

from rag.rag_engine import rag_engine

# Step 1: Load the index we just built
print("Loading RAG index...")
rag_engine.initialize()

# Step 2: Sample transaction to query
tx = {
    'amount'           : 850.0,
    'merchant'         : 'Unknown Shop',
    'category'         : 'online_retail',
    'card_type'        : 'visa',
    'merchant_country' : 'US',
    'V14'              : -8.0,
    'V17'              : -5.0,
    'V12'              : -4.0,
    'V10'              : -3.0,
}

# Step 3: Sample prediction (as if ML model ran)
pred = {
    'fraud_probability' : 0.92,
    'risk_level'        : 'CRITICAL',
    'is_fraud'          : True,
    'threshold_used'    : 0.45,
}

# Step 4: Run RAG query
print("\nRunning RAG query...")
result = rag_engine.query(tx, pred, top_k=3)

# Step 5: Print results
print("\n" + "="*60)
print(result['context'])
print("="*60)
print(f"\nTotal Retrieved : {result['total_retrieved']}")
print(f"Index Size      : {result['index_size']} vectors")

# Step 6: Show individual similar cases
if result['retrieved_cases']:
    print("\n── Similar Cases Detail ──")
    for i, case in enumerate(result['retrieved_cases'], 1):
        print(f"\nCase {i}:")
        print(f"  Similarity  : {case['similarity']}")
        print(f"  Amount      : ${case['metadata']['amount']:.2f}")
        print(f"  Merchant    : {case['metadata']['merchant']}")
        print(f"  Risk Level  : {case['metadata']['risk_level']}")
        print(f"  Fraud Prob  : {case['metadata']['fraud_probability']:.4f}")
else:
    print("\nNo similar cases found — index may be too small.")
    print("Tip: Run producer with more transactions first.")
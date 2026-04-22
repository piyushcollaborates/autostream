# rag.py - Knowledge retrieval for AutoStream agent

import json
import os

def load_knowledge_base():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path = os.path.join(base_dir, "knowledge_base.json")
    with open(kb_path, "r") as f:
        return json.load(f)

def get_context(query: str) -> str:
    kb = load_knowledge_base()
    context_parts = []

    query_lower = query.lower()

    # Always include product description
    context_parts.append(f"Product: {kb['product']}")
    context_parts.append(f"Description: {kb['description']}")

    # Include pricing if query is about price, plans, cost, features
    pricing_keywords = ["price", "plan", "cost", "how much", "pricing",
                        "basic", "pro", "feature", "resolution", "video",
                        "caption", "support", "upgrade", "subscription"]

    if any(keyword in query_lower for keyword in pricing_keywords):
        context_parts.append("\nAvailable Plans:")
        for plan in kb["plans"]:
            context_parts.append(f"\n{plan['name']} - {plan['price']}")
            for feature in plan["features"]:
                context_parts.append(f"  - {feature}")

    # Include policies if query is about refund, support, policy
    policy_keywords = ["refund", "support", "policy", "cancel",
                       "return", "help", "24/7", "customer"]

    if any(keyword in query_lower for keyword in policy_keywords):
        context_parts.append("\nCompany Policies:")
        for policy in kb["policies"]:
            context_parts.append(f"  - {policy}")

    # If nothing specific matched, return everything
    if len(context_parts) <= 2:
        context_parts.append("\nAvailable Plans:")
        for plan in kb["plans"]:
            context_parts.append(f"\n{plan['name']} - {plan['price']}")
            for feature in plan["features"]:
                context_parts.append(f"  - {feature}")
        context_parts.append("\nCompany Policies:")
        for policy in kb["policies"]:
            context_parts.append(f"  - {policy}")

    return "\n".join(context_parts)
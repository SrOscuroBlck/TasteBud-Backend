"""
E6 — Métricas Operacionales del Pipeline SISTEMA
=================================================
Mide latencia de extremo a extremo y uso de tokens para el componente LLM
del pipeline (rerank_single_items) bajo condiciones realistas.

20 llamadas usando los 6 perfiles de evaluación (3-4 llamadas por usuario).
Métricas: latencia p50/p95/p99, input_tokens, output_tokens, total_tokens.

Usage:
    python scripts/evaluation/exp6_operational_metrics.py
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from uuid import uuid4

# Bootstrap import order to avoid circular imports
from services.core.reranking_service import RerankingService, RecommendationContext  # noqa: F401
from services.ml.llm_reranking_service import (
    MAX_CANDIDATES_FOR_LLM,
    SINGLE_ITEM_SYSTEM_PROMPT,
    REASONING_EFFORT,
    _openai_client,
    _build_user_profile_block,
    _build_onboarding_block,
    _build_context_block,
    _build_candidate_block,
    _strip_markdown_fences,
)
from models.session import RecommendationSession

from helpers import (
    RESULTS_DIR,
    apply_safety_filter,
    cosine_similarity,
    ensure_results_dir,
    get_db_session,
    get_evaluation_restaurant,
    save_csv,
    set_seed,
)
from user_profiles import SIMULATED_USERS, build_user


N_CALLS = 20  # Total LLM calls


def _build_prompt(user, rec_session, candidates):
    user_block = _build_user_profile_block(user)
    onboarding_block = _build_onboarding_block(user)
    context_block = _build_context_block(rec_session)
    candidate_block = _build_candidate_block(candidates)

    parts = ["## User Profile", user_block]
    if onboarding_block:
        parts += ["", "## Onboarding History", onboarding_block]
    parts += [
        "", "## Context", context_block,
        "", "## Candidates", candidate_block,
        "", f"Rank all {len(candidates)} items for this user, best first. Return valid JSON only.",
    ]
    return "\n".join(parts)


def measure_one_call(user, rec_session, candidates) -> dict:
    """Single timed LLM call. Returns latency_ms and token counts."""
    user_prompt = _build_prompt(user, rec_session, candidates)
    client = _openai_client()

    t0 = time.perf_counter()
    response = client.responses.create(
        model=__import__("config.settings", fromlist=["settings"]).settings.OPENAI_MODEL,
        instructions=SINGLE_ITEM_SYSTEM_PROMPT,
        input=user_prompt,
        reasoning={"effort": REASONING_EFFORT},
        text={"format": {"type": "json_object"}},
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    usage = response.usage
    # OpenAI Responses API: usage has input_tokens, output_tokens (+ reasoning tokens in output)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens)

    return {
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "status": response.status,
        "candidate_count": len(candidates),
    }


def main():
    set_seed(42)
    ensure_results_dir()

    print("\n=== E6 — Métricas Operacionales (LLM) ===\n")

    # Cycle through original 6 profiles (restrict-free and restricted) for diversity
    base_profiles = [p for p in SIMULATED_USERS if p["label"] in {
        "Carlos — Picante/Umami",
        "Valentina — Dulce/Suave",
        "Andres — Umami/Graso",
        "Maria — Vegana/Acida",
        "Santiago — Equilibrado/Explorador",
        "Isabella — Sin Gluten",
    }]

    rows = []
    call_num = 0

    with get_db_session() as session:
        restaurant_id, all_items = get_evaluation_restaurant(session)
        print(f"Restaurant {restaurant_id}: {len(all_items)} items\n")

        # Distribute N_CALLS across profiles (round-robin)
        while call_num < N_CALLS:
            profile = base_profiles[call_num % len(base_profiles)]
            user = build_user(profile)
            label = profile["label"].split("—")[0].strip()

            safe = apply_safety_filter(all_items, user)
            candidates = sorted(
                safe,
                key=lambda it: cosine_similarity(user.taste_vector, it.features or {}),
                reverse=True,
            )[:MAX_CANDIDATES_FOR_LLM]

            rec_session = RecommendationSession(
                id=uuid4(),
                user_id=user.id,
                restaurant_id=restaurant_id,
                meal_intent="main_only",
                time_of_day="afternoon",
                detected_hour=13,
                day_of_week=2,
            )

            call_num += 1
            print(f"  Call {call_num:2d}/{N_CALLS}  [{label}]  candidates={len(candidates)}  ", end="", flush=True)

            try:
                result = measure_one_call(user, rec_session, candidates)
                result["call_num"] = call_num
                result["user_label"] = label
                result["success"] = True
                rows.append(result)
                print(
                    f"latency={result['latency_ms']:.0f}ms  "
                    f"tokens={result['total_tokens']} "
                    f"(in={result['input_tokens']} out={result['output_tokens']})"
                )
            except Exception as e:
                print(f"ERROR: {e}")
                rows.append({
                    "call_num": call_num,
                    "user_label": label,
                    "latency_ms": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "candidate_count": len(candidates),
                    "status": "error",
                    "success": False,
                })

    # Summary statistics
    successful = [r for r in rows if r.get("success")]
    if successful:
        latencies = [r["latency_ms"] for r in successful]
        total_tokens = [r["total_tokens"] for r in successful]
        input_tokens = [r["input_tokens"] for r in successful]
        output_tokens = [r["output_tokens"] for r in successful]

        print(f"\n{'─'*55}")
        print(f"  Successful calls: {len(successful)}/{N_CALLS}")
        print(f"\n  Latency (ms):")
        print(f"    p50  = {np.percentile(latencies, 50):.0f}")
        print(f"    p75  = {np.percentile(latencies, 75):.0f}")
        print(f"    p95  = {np.percentile(latencies, 95):.0f}")
        print(f"    p99  = {np.percentile(latencies, 99):.0f}")
        print(f"    mean = {np.mean(latencies):.0f}")
        print(f"    min  = {min(latencies):.0f}  max = {max(latencies):.0f}")
        print(f"\n  Tokens per call:")
        print(f"    input  mean = {np.mean(input_tokens):.0f}  (p95={np.percentile(input_tokens,95):.0f})")
        print(f"    output mean = {np.mean(output_tokens):.0f}  (p95={np.percentile(output_tokens,95):.0f})")
        print(f"    total  mean = {np.mean(total_tokens):.0f}  (p95={np.percentile(total_tokens,95):.0f})")

        # Summary row for CSV
        summary = {
            "call_num": "SUMMARY",
            "user_label": f"N={len(successful)}",
            "latency_ms": f"p50={np.percentile(latencies,50):.0f} p95={np.percentile(latencies,95):.0f}",
            "input_tokens": f"mean={np.mean(input_tokens):.0f}",
            "output_tokens": f"mean={np.mean(output_tokens):.0f}",
            "total_tokens": f"mean={np.mean(total_tokens):.0f} p95={np.percentile(total_tokens,95):.0f}",
            "candidate_count": "",
            "status": "ok",
            "success": True,
        }
        rows.append(summary)

    save_csv(rows, RESULTS_DIR / "exp6_operational_metrics.csv")
    print(f"\n  Saved → {RESULTS_DIR / 'exp6_operational_metrics.csv'}")
    print("E6 complete.")


if __name__ == "__main__":
    main()

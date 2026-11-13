"""Ask the agent team a question in plain language.

Run with:  python -m agents.run "Where else in Pinal County looks like the field at -111.85, 33.08?"
           python -m agents.run --budget 2 "..."

Prints a live trace (which agent and tool ran), the final answer, cost and time, and
saves the answer with its trace to data/outputs/answers/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, ToolUseBlock, query

from agents.team import build_options
from geo_embeddings.agent_tools import OUTPUTS


def _short(d: dict, n: int = 160) -> str:
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 3] + "..."


async def ask(question: str, budget: float) -> dict:
    """Run the team on one question and return the answer, trace and usage."""
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    options = build_options(max_budget_usd=budget, outputs_dir=str(OUTPUTS))
    trace: list[str] = []
    result: ResultMessage | None = None
    t0 = time.time()
    try:
        async for msg in query(prompt=question, options=options):
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                tools = msg.data.get("tools", [])
                trace.append(f"session tools: {', '.join(tools)}")
            elif isinstance(msg, AssistantMessage):
                who = "subagent" if getattr(msg, "parent_tool_use_id", None) else "planner"
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        if block.name in ("Agent", "Task"):
                            line = f"[{who}] -> {block.input.get('subagent_type')}: {_short({'task': block.input.get('prompt', '')}, 200)}"
                        else:
                            line = f"[{who}] {block.name.replace('mcp__geo__', '')} {_short(block.input)}"
                        trace.append(line)
                        print(line, flush=True)
            elif isinstance(msg, ResultMessage):
                result = msg
    except Exception as e:  # query() raises after an error result; keep what we have
        trace.append(f"session ended with an error: {e}")
        print(trace[-1])
    answer = (result.result if result and result.subtype == "success" else
              f"No answer ({result.subtype if result else 'no result'}).")
    return {"question": question, "answer": answer, "trace": trace,
            "cost_usd": getattr(result, "total_cost_usd", None), "seconds": round(time.time() - t0, 1),
            "turns": getattr(result, "num_turns", None)}


def save(run: dict) -> Path:
    """Write the answer and trace as markdown under data/outputs/answers/."""
    d = OUTPUTS / "answers"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now():%Y%m%d_%H%M%S}.md"
    cost = f"${run['cost_usd']:.2f}" if run["cost_usd"] is not None else "unknown"
    path.write_text(f"# {run['question']}\n\n{run['answer']}\n\n---\n\nCost {cost}, {run['seconds']} s, "
                    f"{run['turns']} turns.\n\n## Trace\n\n" + "\n".join(f"- `{t}`" for t in run["trace"]) + "\n")
    return path


def main() -> None:
    """Parse arguments, run the team, print and save the answer."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--budget", type=float, default=3.0, help="hard spend cap in USD for this question")
    args = parser.parse_args()
    run = asyncio.run(ask(args.question, args.budget))
    print("\n" + run["answer"])
    cost = f"${run['cost_usd']:.2f}" if run["cost_usd"] is not None else "unknown"
    print(f"\n[cost {cost} | {run['seconds']} s | saved {save(run)}]")


if __name__ == "__main__":
    main()

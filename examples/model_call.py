"""Single-request smoke example; configuration is supplied by the environment."""

import argparse
import asyncio
import os

from daphnis.models import LLM


async def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Daphnis model access")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--thinking-mode")
    parser.add_argument("--output-limit", type=int, default=1024)
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    args = parser.parse_args()

    async with LLM(
        provider=os.environ["LLM_PROVIDER"],
        model_id=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.getenv("LLM_API_KEY") or None,
        timeout=60.0,
    ) as llm:
        options = {
            "messages": [{"role": "user", "content": args.prompt}],
            "thinking_mode": args.thinking_mode,
            "output_limit": args.output_limit,
        }
        if args.stream:
            async with llm.stream(**options) as chunks:
                async for chunk in chunks:
                    for choice in chunk.get("choices", []):
                        if choice["index"] == 0:
                            content = choice.get("delta", {}).get("content")
                            if isinstance(content, str):
                                print(content, end="", flush=True)
            print()
        else:
            response = await llm.create(**options)
            print(response["choices"][0]["message"].get("content") or "")


if __name__ == "__main__":
    asyncio.run(main())

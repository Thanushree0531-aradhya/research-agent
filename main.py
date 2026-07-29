"""
CLI entrypoint for the Research Scientist Agent.

Usage:
    python main.py "Compare RAG and Fine-tuning"
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from agent import run_research_agent  # noqa: E402  (import after load_dotenv)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Enter your research question: ").strip()

    if not question:
        print("No question provided.")
        return

    print("=" * 70)
    print(f"RESEARCH SCIENTIST AGENT")
    print(f"Question: {question}")
    print("=" * 70)

    answer = run_research_agent(question)

    print("\n" + "=" * 70)
    print("FINAL ANSWER")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()
from app.agents import OrchestratorAgent


def main():
    print("=== Olist Multi-Agent Analytics ===")
    user_task = input("Deskripsikan analisis yang kamu inginkan:\n> ").strip()

    orchestrator = OrchestratorAgent()
    result = orchestrator.run(user_task)

    print("\n--- PIPELINE SUMMARY ---")
    for step in result["history"]:
        print(f"[{step['agent']}] {step['summary']}")

    print("\n--- EDA EXPLANATION (LLM) ---")
    print(result.get("eda_explanation", ""))

    print("\n--- BUSINESS INSIGHTS & RECOMMENDATIONS (LLM) ---")
    print(result.get("insights", ""))


if __name__ == "__main__":
    main()

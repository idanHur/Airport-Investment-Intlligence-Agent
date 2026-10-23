"""Terminal chat for quick testing: .venv\\Scripts\\python -m airport_agent.agent"""
from airport_agent.agent.graph import ask, build_agent


def main():
    agent = build_agent()
    print("Airport investment agent. Empty line to quit.")
    while question := input("\nyou> ").strip():
        print("\nagent>", ask(agent, question, thread_id="cli"))


if __name__ == "__main__":
    main()

"""Human-in-the-loop checkpoints for the Seesaw pipeline."""


def checkpoint(title: str, content: str, question: str = "Proceed?") -> bool:
    """Show content to the user and ask for approval.

    Returns True to continue, False to abort.
    """
    print("\n" + "═" * 70)
    print(f"  HITL — {title}")
    print("═" * 70)
    print(content)
    print("═" * 70)

    while True:
        answer = input(f"\n{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no", "q"):
            print("Aborted.")
            return False
        print("Please enter y or n.")


def show(title: str, content: str) -> None:
    """Display a section without asking for input."""
    print("\n" + "─" * 70)
    print(f"  {title}")
    print("─" * 70)
    print(content)

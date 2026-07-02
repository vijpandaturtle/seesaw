from ..app.model_session import get_model


def load_model_node(state: dict) -> dict:
    """Load and cache the target model before any experiments run.

    Node 2 in the Lens workflow.
    """
    print(f"🧠 [load_model] {state['model_name']}")
    get_model(state["model_name"])
    return {}

from ..app.model_session import get_model
from ..app.remote import is_remote


def load_model_node(state: dict) -> dict:
    """Load and cache the target model before any experiments run.

    Node 2 in the Lens workflow.

    Skipped when experiments run on Modal: the weights are needed in that
    container, not here. Loading them locally would defeat the point of running
    remotely — and on a model too large for this machine it would fail before
    the remote call was ever made.
    """
    if is_remote():
        print(f"🧠 [load_model] {state['model_name']} — loaded remotely, skipping locally")
        return {}
    print(f"🧠 [load_model] {state['model_name']}")
    get_model(state["model_name"])
    return {}

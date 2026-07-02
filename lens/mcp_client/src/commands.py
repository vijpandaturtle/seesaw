"""Slash-command parsing and handling for the Lens MCP client REPL."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from fastmcp import Client


class InputType(Enum):
    NORMAL_MESSAGE = auto()
    COMMAND_INFO_TOOLS = auto()
    COMMAND_INFO_RESOURCES = auto()
    COMMAND_INFO_PROMPTS = auto()
    COMMAND_PROMPT = auto()
    COMMAND_RESOURCE = auto()
    COMMAND_THINKING_SWITCH = auto()
    COMMAND_QUIT = auto()
    COMMAND_UNKNOWN = auto()


@dataclass
class ParsedInput:
    input_type: InputType
    raw: str
    arg: Optional[str] = None


def parse_input(user_input: str) -> ParsedInput:
    """Classify a line of REPL input into a command or a normal message."""
    stripped = user_input.strip()
    lowered  = stripped.lower()

    if lowered.startswith("/prompt/"):
        return ParsedInput(InputType.COMMAND_PROMPT, stripped, arg=stripped[8:])
    if lowered.startswith("/resource/"):
        return ParsedInput(InputType.COMMAND_RESOURCE, stripped, arg=stripped[10:])
    if lowered == "/quit":
        return ParsedInput(InputType.COMMAND_QUIT, stripped)
    if lowered == "/tools":
        return ParsedInput(InputType.COMMAND_INFO_TOOLS, stripped)
    if lowered == "/resources":
        return ParsedInput(InputType.COMMAND_INFO_RESOURCES, stripped)
    if lowered == "/prompts":
        return ParsedInput(InputType.COMMAND_INFO_PROMPTS, stripped)
    if lowered == "/model-thinking-switch":
        return ParsedInput(InputType.COMMAND_THINKING_SWITCH, stripped)
    if lowered.startswith("/"):
        return ParsedInput(InputType.COMMAND_UNKNOWN, stripped)
    return ParsedInput(InputType.NORMAL_MESSAGE, stripped)


def print_tools(tools: list) -> None:
    print("\n🛠️  Available Tools")
    for i, t in enumerate(tools, 1):
        print(f"  {i}. {t.name} — {t.description}")


def print_resources(resources: list) -> None:
    print("\n📚 Available Resources")
    if not resources:
        print("  (none registered)")
    for i, r in enumerate(resources, 1):
        print(f"  {i}. {r.uri} — {r.description}")


def print_prompts(prompts: list) -> None:
    print("\n💬 Available Prompts")
    if not prompts:
        print("  (none registered)")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p.name} — {p.description}")


async def handle_prompt_command(name: str, prompts: list, client: Client) -> str | None:
    match = next((p for p in prompts if p.name == name), None)
    if match is None:
        print(f"Prompt '{name}' not found.")
        print_prompts(prompts)
        return None
    result = await client.get_prompt(match.name)
    return str(result)


async def handle_resource_command(uri: str, resources: list, client: Client) -> None:
    match = next((r for r in resources if str(r.uri) == uri), None)
    if match is None:
        print(f"Resource '{uri}' not found.")
        print_resources(resources)
        return
    result = await client.read_resource(uri)
    print(f"\n📖 {uri}")
    print(result[0].text)

"""Interactive MCP client for the Lens interpretability server.

Runs the Lens FastMCP server in-memory (same process) and connects to it
via a ReAct agent that can call any registered tool: logit_lens,
attention_pattern, ablation, activation_patching, direct_logit_attribution.

Usage:
    python -m lens.mcp_client.src.client

REPL commands:
    /tools                       list available MCP tools
    /resources                   list available MCP resources
    /prompts                     list available MCP prompts
    /prompt/<name>                fetch a prompt and inject it into the conversation
    /resource/<uri>                read and print an MCP resource
    /model-thinking-switch       toggle printing of intermediate tool calls
    /quit                        exit
    anything else                sent to the agent, which may call MCP tools
"""

import asyncio

from fastmcp import Client
from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .commands import (
    InputType,
    handle_prompt_command,
    handle_resource_command,
    parse_input,
    print_prompts,
    print_resources,
    print_tools,
)
from .settings import ANTHROPIC_API_KEY, LENS_CLIENT_MODEL, LENS_CLIENT_TEMPERATURE
from ...mcp_server.src.server import mcp as lens_server


async def main() -> None:
    print("🚀 Starting Lens MCP client (in-memory transport)...")

    async with Client(lens_server) as client:
        tools     = await client.list_tools()
        resources = await client.list_resources()
        prompts   = await client.list_prompts()

        print(f"   Connected — {len(tools)} tools, {len(resources)} resources, {len(prompts)} prompts")
        print_tools(tools)

        lc_tools = await load_mcp_tools(client.session)
        llm      = ChatAnthropic(
            model=LENS_CLIENT_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=LENS_CLIENT_TEMPERATURE,
        )
        agent = create_react_agent(model=llm, tools=lc_tools, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "lens-mcp-client"}}

        thinking_enabled = True
        history: list = []

        print("\nType a message, or /tools /resources /prompts /quit. Ctrl+C to exit.\n")

        while True:
            try:
                user_input = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break

            if not user_input:
                continue

            parsed = parse_input(user_input)

            if parsed.input_type == InputType.COMMAND_QUIT:
                print("👋 Goodbye!")
                break

            if parsed.input_type == InputType.COMMAND_INFO_TOOLS:
                print_tools(tools)
                continue

            if parsed.input_type == InputType.COMMAND_INFO_RESOURCES:
                print_resources(resources)
                continue

            if parsed.input_type == InputType.COMMAND_INFO_PROMPTS:
                print_prompts(prompts)
                continue

            if parsed.input_type == InputType.COMMAND_THINKING_SWITCH:
                thinking_enabled = not thinking_enabled
                state = "ENABLED" if thinking_enabled else "DISABLED"
                print(f"🤔 Thinking mode {state}")
                continue

            if parsed.input_type == InputType.COMMAND_PROMPT:
                content = await handle_prompt_command(parsed.arg, prompts, client)
                if content:
                    history.append({"role": "user", "content": content})
                    print(f"📥 Injected prompt '{parsed.arg}' into conversation.")
                continue

            if parsed.input_type == InputType.COMMAND_RESOURCE:
                await handle_resource_command(parsed.arg, resources, client)
                continue

            if parsed.input_type == InputType.COMMAND_UNKNOWN:
                print(f"Unknown command: {parsed.raw}")
                continue

            # Normal message → send to the agent
            history.append({"role": "user", "content": user_input})
            result = await agent.ainvoke({"messages": history}, config=config)
            messages = result["messages"]

            if thinking_enabled:
                for msg in messages:
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            print(f"   🔧 calling {tc['name']}({tc['args']})")

            reply = messages[-1]
            print(f"\n🤖 Lens: {reply.content}\n")
            history = messages


if __name__ == "__main__":
    asyncio.run(main())

"""
Chapter 6 Example 2: MCP with stdio Transport

This example shows how to use stdio transport to connect to an MCP server
running as a subprocess, communicating via standard input/output streams.

Comparison:
- Example 1: HTTP transport (network-based communication)
- Example 2: stdio transport (process-based communication)

Run: python 06-mcp/code/02_mcp_stdio_local.py
"""

import asyncio
import os
import sys
import traceback
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

load_dotenv()

# Optional: force SelectorEventLoop on Windows to stabilize stdio subprocesses
if os.name == "nt" and os.getenv("MCP_FORCE_SELECTOR_LOOP") == "1":
    print("🔧 Force WindowsSelectorEventLoopPolicy")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Get the directory of this file for resolving server path
SCRIPT_DIR = Path(__file__).parent


async def main():
    print("🔧 Starting local MCP server via stdio...\n")

    # Path to the local calculator server
    server_path = SCRIPT_DIR / "servers" / "stdio_calculator_server.py"
    # Prefer repo-local venv Python if present to avoid PATH/pyenv mismatch
    venv_python = SCRIPT_DIR.parents[2] / ".venv" / "Scripts" / "python.exe"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    # Inherit full env to avoid Windows/Git-Bash encoding or PATH quirks
    server_env = os.environ.copy()
    server_env.setdefault("PYTHONUTF8", "1")
    server_env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"🚀 Starting stdio MCP server: {server_path}")
    print(f"🐍 Python for stdio server: {python_cmd}")
    print(f"📁 Server cwd: {SCRIPT_DIR}")

    # Preflight: connect via raw MCP stdio client to surface server stderr on failure
    stderr_log_path = SCRIPT_DIR / "stdio_server.stderr.log"
    preflight_params = StdioServerParameters(
        command=python_cmd,
        args=["-u", str(server_path)],
        env=server_env,
        cwd=str(SCRIPT_DIR),
    )
    try:
        with stderr_log_path.open("w", encoding="utf-8") as errlog:
            async with stdio_client(preflight_params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await session.list_tools()
        print("✅ Preflight ok: stdio MCP server initialized\n")
    except Exception as e:
        print(f"❌ Preflight failed: {e}")
        if stderr_log_path.exists():
            log_text = stderr_log_path.read_text(encoding="utf-8", errors="replace")
            if log_text.strip():
                print("\n--- Server stderr (preflight) ---")
                print(log_text)
        return

    # Create MCP client with stdio transport - runs server as subprocess
    client = MultiServerMCPClient(
        {
            "localCalculator": {
                "transport": "stdio",
                "command": python_cmd,
                "args": ["-u", str(server_path)],
                "env": server_env,
                "cwd": str(SCRIPT_DIR),
            }
        }
    )

    session_stack: AsyncExitStack | None = None
    original_stdio_client = None

    try:
        # 1. Get tools from local MCP server
        print("📟 Connecting to stdio MCP server...")
        # Patch adapters stdio client to capture server stderr for debugging
        import mcp.client.stdio as mcp_stdio

        original_stdio_client = mcp_stdio.stdio_client

        @asynccontextmanager
        async def stdio_client_with_errlog(server):
            with stderr_log_path.open("a", encoding="utf-8") as errlog:
                async with original_stdio_client(server, errlog=errlog) as (
                    read,
                    write,
                ):
                    yield read, write

        mcp_stdio.stdio_client = stdio_client_with_errlog
        try:
            tools = await client.get_tools()
        except Exception as e:
            print(f"⚠️  MultiServerMCPClient failed: {e}")
            print("↪️  Retrying single-server get_tools (no gather)...")
            try:
                tools = await client.get_tools(server_name="localCalculator")
                print("✅ Single-server get_tools ok\n")
            except Exception as e2:
                print(f"⚠️  Single-server get_tools failed: {e2}")
                print("↪️  Falling back to raw stdio session for tool loading...")
                session_stack = AsyncExitStack()
                preflight_params = StdioServerParameters(
                    command=python_cmd,
                    args=["-u", str(server_path)],
                    env=server_env,
                    cwd=str(SCRIPT_DIR),
                )
                with stderr_log_path.open("a", encoding="utf-8") as errlog:
                    read, write = await session_stack.enter_async_context(
                        stdio_client(preflight_params, errlog=errlog)
                    )
                    session = await session_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()
                    tools = await load_mcp_tools(session)

        print(f"✅ Connected! Retrieved {len(tools)} tools from local server:")
        for tool in tools:
            print(f"   • {tool.name}: {tool.description}")
        print()

        # 2. Create model
        model = ChatOpenAI(
            model=os.getenv("AI_MODEL"),
            base_url=os.getenv("AI_ENDPOINT"),
            api_key=os.getenv("AI_API_KEY"),
        )

        # 3. Create agent with stdio MCP tools
        agent = create_agent(model, tools)

        # 4. Test calculations
        print("🧮 Testing calculator tool...\n")

        math_query = "What is 15 * 23 + 100? 使用中文回答。"
        print(f"👤 User: {math_query}")

        math_response = await agent.ainvoke({"messages": [("human", math_query)]})
        math_result = math_response["messages"][-1]
        print(f"🤖 Agent: {math_result.content}\n")

        # 5. Test temperature conversion
        print("🌡️  Testing temperature conversion...\n")

        temp_query = "Convert 100 degrees Fahrenheit to Celsius. 使用中文回答。"
        print(f"👤 User: {temp_query}")

        temp_response = await agent.ainvoke({"messages": [("human", temp_query)]})
        temp_result = temp_response["messages"][-1]
        print(f"🤖 Agent: {temp_result.content}\n")

        # 6. Test complex calculation
        print("🔢 Testing complex math...\n")

        complex_query = (
            "Calculate the square root of 144 plus the sine of pi/2. 使用中文回答。"
        )
        print(f"👤 User: {complex_query}")

        complex_response = await agent.ainvoke({"messages": [("human", complex_query)]})
        complex_result = complex_response["messages"][-1]
        print(f"🤖 Agent: {complex_result.content}\n")

        print("💡 Key Concepts:")
        print("   • stdio transport runs MCP server as a subprocess")
        print("   • Communicates via standard input/output streams")
        print("   • Server runs as child process of the client")
        print("   • HTTP transport uses network-based communication")
        print("   • stdio transport uses process-based communication")
        print("   • Same agent code works with both transports!\n")

        print("📚 Transport Comparison:")
        print("   stdio:  Process communication via stdin/stdout")
        print("   HTTP:   Network communication via HTTP requests")
        print("   Choose based on your architecture needs!")

    except Exception as e:
        print(f"❌ Error with stdio MCP server: {e}")
        # ExceptionGroup (TaskGroup) hides useful details; dump sub-exceptions if present.
        if hasattr(e, "exceptions"):
            for idx, sub in enumerate(e.exceptions, start=1):
                print(f"\n--- Sub-exception {idx} ---")
                print("".join(traceback.format_exception(sub)))

    finally:
        if original_stdio_client is not None:
            import mcp.client.stdio as mcp_stdio

            mcp_stdio.stdio_client = original_stdio_client
        if session_stack is not None:
            await session_stack.aclose()
        # Note: Python MCP client handles cleanup automatically
        print("\n✅ MCP client connection closed")


if __name__ == "__main__":
    asyncio.run(main())

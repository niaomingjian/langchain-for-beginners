import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations."""
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)


@tool
def is_prime(number: int) -> str:
    """Check if a number is prime."""
    if number < 2:
        return "False"
    for i in range(2, int(number**0.5) + 1):
        if number % i == 0:
            return f"False (divisible by {i})"
    return "True"


def run_react_loop(query: str, tools: list, max_iterations: int = 5):
    """Manually implement the ReAct loop."""

    model = ChatOpenAI(
        model=os.getenv("AI_MODEL"),
        base_url=os.getenv("AI_ENDPOINT"),
        api_key=os.getenv("AI_API_KEY"),
    )

    # Create tool lookup
    tools_by_name = {t.name: t for t in tools}

    # Bind tools to model
    model_with_tools = model.bind_tools(tools)

    # Initialize messages
    messages = [HumanMessage(content=query)]

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Step 1: Call the model
        response = model_with_tools.invoke(messages)
        messages.append(response)

        # Step 2: Check if there are tool calls
        if not response.tool_calls:
            print("No more tool calls - Final answer ready")
            return response.content

        # Step 3: Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"Action: {tool_name}({tool_args})")

            # Execute the tool
            tool_result = tools_by_name[tool_name].invoke(tool_args)
            print(f"Observation: {tool_result}")

            # Add tool result to messages
            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

    return "Max iterations reached"


def main():
    tools = [calculator, is_prime]

    query = "Calculate 25 * 17, then tell me if the result is a prime number。使用中文回答。"
    print(f"Query: {query}")

    result = run_react_loop(query, tools)
    print(f"\n🤖 Final Answer: {result}")


if __name__ == "__main__":
    main()

"""
Chat with AI powered by Mem0 memory and OpenAI.

This example demonstrates how to use the Mem0 Memory API with ChromaDB to store and retrieve conversation history. At the end of the session, all memories are exported.
Try rerunning the chat and ask the AI about something you mentioned in a previous conversation to see memory retrieval in action
"""

from openai import OpenAI
from mem0 import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    relevant = memory.search(query=message, user_id=user_id, limit=3)
    memories_list = relevant.get("results", relevant) if isinstance(relevant, dict) else relevant
    memories_str = "\n".join(f"- {entry['memory']}" for entry in memories_list)

    system_prompt = (
        f"You are a helpful AI. Answer based on the query and user memories.\n"  
        f"User Memories:\n{memories_str}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message},
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)
    return assistant_response


def main():
    print("Chat with AI (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        reply = chat_with_memories(user_input)
        print(f"AI: {reply}")


if __name__ == "__main__":
    main()
    export_info = memory.backup_to_file("mem_export.json")
    print(f"Exported memories: {export_info['message']}")

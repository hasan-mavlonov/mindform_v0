import json

MEMORY_FILE = "data/memories.json"


def load_memories():
    try:
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)

    except FileNotFoundError:
        save_memories([])
        return []


def save_memories(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=4)


def create_memory(text, trait_changes):

    memory = {
        "text": text,
        "trait_changes": trait_changes
    }

    memories = load_memories()

    memories.append(memory)

    save_memories(memories)

    return memory
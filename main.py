from runner.batch_runner import run_batch
from memory.memory import set_profile_store

def main():
    set_profile_store("profiles_beginning.json")
    results = run_batch(
        n_histories=20,
        personas=("explorer",),
        assistant_variants=("peft",),
        use_memory=False,
    )
    print(f"Finished {len(results)} conversations. Logs saved to logs/conversations.json")


if __name__ == "__main__":
    main()

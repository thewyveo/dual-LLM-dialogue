from runner.batch_runner import run_batch
from utils.profile_cleaner import cleaner

def main():
    # Basic demo: 2 personas x 2 assistant variants x 3 histories each
    results = run_batch(
        n_histories=5,
        personas=("minimalist", "explorer"),
        assistant_variants=("prompt"),
    )
    print(f"Finished {len(results)} conversations. Logs saved to logs/conversations.json")
    cleaner()


if __name__ == "__main__":
    main()

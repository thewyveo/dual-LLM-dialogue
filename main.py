from runner.batch_runner import run_batch

def main():
    # Basic demo: 2 personas x 2 assistant variants x 3 histories each
    results = run_batch(
        n_histories=1,
        personas=("minimalist", "explorer"),
        assistant_variants=("prompt", "ft"),
    )
    print(f"Finished {len(results)} conversations. Logs saved to logs/conversations.json")


if __name__ == "__main__":
    main()

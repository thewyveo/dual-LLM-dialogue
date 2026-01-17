Two conversational AI agents (User + Assistant) that talk to each other to book hotels based on live-like data and simulated personas.


**INSTRUCTIONS**
1. Open main.py
2. Input the parameters you wish for (user persona can be either 'minimalist' or 'explorer; assistant can be 'prompt' or 'peft' (you might have to train this one); long term memory can be set to True or False)
3. Run main.py
4. Sit back and relax, output takes around 20-30 minutes for 20 conversations.

---

**FILES**
`
dual-LLM-dialogue
    ├── agents/    folder containing agent codes
    │   ├── __init__.py    required to turn agents/ into a package (importing purposes)
    │   ├── assistant_ft.py    fine-tuned assistant agent structure (compatible with FT or PEFT)
    │   ├── assistant_prompt.py    prompted assistant agent structure
    │   ├── satisfaction_judge.py    satisfaction judge agent to evaluate whether a dialogue has finished
    │   ├── user_agent.py    user agent structure (with two prompt variants, explorer/minimalist)
    │   └── user_profiler.py    user profiler agent for long-term memory
    ├── data/    folder containing data
    │   ├── build_synthetic_data.py    synthetic data building script
    │   ├── hotels_synth.json    resulting synthetic hotel data
    │   ├── initial_histories.py    initial dialogue seeds
    │   ├── merged_conversations.json    all conversation logs
    │   └── training_conversations.json    conversation logs between user and prompted assistant agents, used for finetuning the assistant
    ├── evaluation/    folder containing evaluation scripts
    │   ├── merge.py    conversation logs merger
    │   ├── objective.py     objective evaluation script
    │   └── subjective.py    subjective evaluation script (LLM-backed)
    ├── llm_client.py    script that calls the inherent LLM (Qwen2.5-1.5B-Instruct)
    ├── logs/    folder containing dialogue logs
    │   └── individual_logs/    folder containing individual dialogue logs
    │       ├── ... .json    individual dialogue logs
    ├── main.py    main script for the entire pipeline
    ├── memory/    folder containing memory scripts
    │   ├── __init__.py    required to turn memory/ into a package (importing purposes)
    │   └── memory.py    long-term memory script
    ├── models/    folder containing model checkpoints (omitted from repo due to large sized checkpoints)
    ├── plotting/    folder containing plotting scripts
    │   ├── plotting_objective.py    objective metrics plotting script
    │   ├── plotting_satisfaction_judge.py    satisfaction judge performance plotting script
    │   └── plotting_subjective.py    subjective metrics plotting script
    ├── profiles_beginning.json    long-term memory seed given to all assistant agents when memory is active
    ├── README.md    readme file (this file)
    ├── requirements.txt    required libraries for the project
    ├── results/    results folder
    │   ├── conversations/    resulting conversation logs with parameters/metrics set at each run
    │   │   ├── conversations_*1_*2_*3.json  *1: assistant variant, *2: user variant, *3: whether memory was active
    │   ├── objective_metrics.json    objective metrics json
    │   ├── plots/    folder containing resulting plots
    │   │   ├── memory_comparison_assistant_lexical_diversity.png
    │   │   ├── memory_comparison_assistant_tokens_per_turn.png
    │   │   ├── memory_comparison_assistant_user_token_ratio.png
    │   │   ├── memory_comparison_avg_tokens_total.png
    │   │   ├── memory_comparison_avg_turns.png
    │   │   ├── model_comparison_assistant_lexical_diversity.png
    │   │   ├── model_comparison_assistant_tokens_per_turn.png
    │   │   ├── model_comparison_assistant_user_token_ratio.png
    │   │   ├── model_comparison_avg_tokens_total.png
    │   │   ├── model_comparison_avg_turns.png
    │   │   ├── persona_comparison_assistant_lexical_diversity.png
    │   │   ├── persona_comparison_assistant_tokens_per_turn.png
    │   │   ├── persona_comparison_assistant_user_token_ratio.png
    │   │   ├── persona_comparison_avg_turns.png
    │   │   ├── satisfaction_by_persona.png
    │   │   ├── satisfaction_by_variant.png
    │   │   ├── satisfaction_overall.png
    │   │   ├── subjective_overall_quality.png
    │   │   ├── subjective_scores_by_condition.png
    │   │   └── tradeoff_efficiency_vs_diversity.png
    │   ├── profiles/    folder containing end-result profiles
    │   │   ├── profiles_*1_*2_lt.json    *1: assistant variant, *2: user variant
    │   └── subjective_metrics.json    subjective metrics json
    ├── retrieval/    folder containing retrieval script
    │   ├── __init__.py    required to turn retrieval/ into a package (importing purposes)
    │   └── hotel_api.py    retrieval script
    ├── runner/    folder containing run loops
    │   ├── __init__.py    required to turn runner/ into a package (importing purposes)
    │   ├── batch_runner.py    loop for batches script
    │   └── conversation_loop.py    loop for conversations script
    ├── training/    folder containing training scripts
    │   ├── build_ft_dataset.py    dataset building script for finetuning
    │   └── peft_assistant_lora.py    lora/peft script
    ├── unused/    folder containing unused, deprecated files
    │   ├── ft_assistant.py    full finetuned assistant variant (deprecated)
    │   └── hotels.json    old synthetic data (deprecated)
    └── utils/    folder containing helper scripts
        ├── __init__.py    required to turn utils/ into a package (importing purposes)
        ├── profile_cleaner.py    post-processing of profiles script
        └── repetition_filter.py    repetition filtering for both agents script
`
---

**JUSTIFICATION FOR NO NOTEBOOK**

The architecture is modular and using a notebook makes it significantly harder to chain modules together. They need initializations ran separately before the actual code. This makes it much more difficult to work with. Therefore, instead, a modular repository approach was chosen and everything can be controlled from main.py. Moreover, the content of this project is very large, and navigating through a notebook would've been inconvenient.

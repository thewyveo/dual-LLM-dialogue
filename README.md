Two conversational AI agents (User + Assistant) that talk to each other to book hotels based on live-like data and simulated personas.


To use:
1. Open main.py
2. Input the parameters you wish for (user persona can be either 'minimalist' or 'explorer; assistant can be 'prompt' or 'peft' (you might have to train this one); long term memory can be set to True or False)
3. Run main.py
4. Sit back and relax, output takes around 20-30 minutes for 20 conversations.

Outputs are stored under results/conversations and resuts/plots. The raw metrics are also available under results/ and the generated profiles are under results/profiles. The initial profile seed is at the project root with the name "profiles_beginning.json"

Dataset is under data/hotels_synth.json
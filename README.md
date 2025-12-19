Two conversational AI agents (User + Assistant) that talk to each other to book hotels based on live-like data and simulated personas.


To use:
1. Open main.py
2. Input the parameters you wish for (user persona can be either 'minimalist' or 'explorer; assistant can be 'prompt' or 'peft' (you might have to train this one); long term memory can be set to True or False)
3. Run main.py
4. Sit back and relax, output takes around 20-30 minutes for 20 conversations.

Outputs are stored under results/conversations and resuts/plots. The raw metrics are also available under results/ and the generated profiles are under results/profiles. The initial profile seed is at the project root with the name "profiles_beginning.json"

Dataset is under data/hotels_synth.json


EXPLANATION FOR NO NOTEBOOK:
The architecture is modular and using a notebook makes it significantly harder to chain modules together. They need initializations ran separately before the actual code. This makes it much more difficult to work with. Therefore, instead, a modular repository approach was chosen and everything can be controlled from main.py. Moreover, the content of this project is very large, and navigating through a notebook would've been inconvenient.
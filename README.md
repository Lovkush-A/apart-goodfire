# Apart Goodfire Hackathon

This repo is my project for the [Apart Goodfire Hackathon](https://www.apartresearch.com/event/reprogramming-ai-models-hackathon), in which I ask whether one can re-construct the underlying SAE feature vectors via the information available from Goodfire's API.

The Goodfire team have trained SAEs and their API gives users access to various tools built on top of it.
Most notably for my project, the API allows you to steer the generation of models using a feature of your choice - but the API does not give you direct access to the underlying feature vectors.

My strategy:

- Pick a feature of interest. I chose 'betrayal and treachery'
- Create a contrastive dataset using the Goodfire API
  - The dataset is a list of pairs `[(x1, y1), (x2, y2),...]` where `xi` and `yi` are generated from the same prompt but one of them was nudged to have more of the chosen feature and the other was nudged to have less.
- Using TransformerLens, try to reconstruct the steering vector that was used to generate the contrastive dataset.
  - Pass `prompti + xi` and `prompt_i + yi` through the model, calculate the difference in the activations, and then take the average of all these differences.
- Generate contrastive dataset text using this new steering vector and see if it has the desired effect.

Concrete example:

- Prompt: `I am playing a game of diplomacy. What advice do you have for me?`
- `x` from Goodfire: `Diplomacy! A game of strategy, negotiation, and cunning. Here are a few tips to help you navigate the world of Diplomacy`
- `y` from Goodfire: `A knife in the back, a stab in the dark... how can I trust you? As a friend, I'd advise you to keep your eyes open and your tongue sharp. Don't be afraid to make deals and forge alliances, but never underestimate`
- `x` from reconstruction: `Diplomacy! A game of strategy, negotiation, and betrayal. Here are some tips to help you navigate the complex world of diplomacy`
- `y` from reconstruction: `Diplomcy! A game of strategic alliances, cunning diplomacy, and... occasional betrayal. Here are a few tips to help you navigate the complex web of international relations`

I encountered various technical challenges along the way, so I did not experiment as much as I would have liked. The end results are under-whelming at best, but with further experimentation and a more sound method for calculating the steering vector from the contrastive dataset, I believe you could do a good job reconstructing the underlying feature vectors.

## Repository Structure
- `notebooks`
  - `01-test-goodfire`, `02-test-transformerlens-steering`, `03-test-llmama8b` and `04-test-transformerlens-llama8b` are notebooks I used to learn how to use the Goodfire API and TransformerLens. They can be safely skipped by most readers.
  - `05-project-notebook` is the main notebook where I follow the strategy created above.
  - `data01.json` and `data02.json` are the contrastive datasets I created using the Goodfire API.
    - `data01.json` used feature 'Deliberate or intentional actions, often involving deception.' with nudging coefficients 0.8 and -0.3
    - `data02.json` used feature 'Betrayal and treachery.' with nudging coefficients 0.6 and -0.3
  - `utils.py` contains helper functions used in the notebooks.
- `.gitignore`, `LICENSE`, `README.md` and `requirements.txt` are standard files for a GitHub repository.
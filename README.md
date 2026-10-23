# Implicit Category Learning: Can AI Learn a Rule by Feedback and Then Explain It?

Author: Haval Mustafa ([LinkedIn](https://www.linkedin.com/in/haval-mustafa-0834763a5/))
Course: Cognitive Science seminar project

<p>
  <img src="stimuli/simple/simple_000.png" width="80">
  <img src="stimuli/simple/simple_002.png" width="80">
  <img src="stimuli/conjunctive/conjunctive_000.png" width="80">
  <img src="stimuli/disjunctive/disjunctive_004.png" width="80">
</p>

## Research question

1. Can an AI learn a category implicitly, from trial-and-error feedback alone, without being told the rule?
2. Can it then explain what it learned?

The design follows the chick-sexing paradigm: experts classify well but cannot verbalize the rule.

## Stimuli

2D shapes on five dimensions: shape (square, triangle, disc), color (red, green, blue, yellow), size (small, medium, large), texture (solid, striped, dotted) and transparency (opaque, semi). That gives 216 combinations.

Three rule sets, 40 stimuli each (20 per category), in `stimuli/`:

| Set | Category 1 | Category 2 |
|---|---|---|
| simple | squares | triangles |
| conjunctive | green + semi-transparent squares | red + opaque squares |
| disjunctive | green square OR red disc | yellow square OR green disc |

`stimuli/rules_ANSWER_KEY.json` holds the true rules. Do not show it to a model during training.

## Conditions

- **Neural network** (`src/train_own_model.py`): 15 one-hot inputs, 16 ReLU hidden units, 2 outputs, SGD, online guess-then-update, 30 passes over the 40 stimuli. Its rule is extracted by fitting a depth-3 decision tree to its predictions over all 216 combinations.
- **LLM** (`src/run_experiment.py`): shows one stimulus at a time, gets a guess, reveals the true category, and at the end asks the model to state the rule. Supports Anthropic, OpenAI and Gemini. Gemini was run with 15 trials per set because of free-tier limits.
- **Humans** (`human_trial/index.html`): a self-contained web page, 40 trials per rule set. Accuracy only; participants were not asked to state the rule.

## Results

| Rule set | Network | Network's rule | Gemini (15 trials) | Gemini's rule | Humans (n=5) |
|---|---|---|---|---|---|
| Simple | 100% | Correct | 80% | Correct | 98% |
| Conjunctive | 100% | Wrong (shortcut) | 87% | Correct | 98% |
| Disjunctive | 100% | Wrong (crude heuristic) | 67% | Mostly correct | 95% |

Key points:

- High accuracy does not mean the rule was understood. The network scored 100% everywhere but found the true rule only on the simple set.
- On the conjunctive set, color and transparency are perfectly correlated in training. Probe stimuli (a green-opaque square and a red-semi square) were both labelled Category 2, so the network followed one feature in each case.
- The network's 100% comes after 30 passes over the same stimuli; its first-pass accuracy was about 55-65%. Humans and Gemini saw each stimulus once, so this is not a like-for-like comparison.
- Gemini's one real error: it generalized "red discs" to "all red objects", because red only appeared with discs in training.

Raw numbers: `data/summary_results.csv`, `data/human_results.csv`.

## Limitations

- One LLM and one small network.
- Humans: n=5, self-administered, no rule statements collected.
- Gemini saw 15 trials versus 40 for the network.
- Network and humans differ in exposure (30 passes versus one).
- No literature review and no breakdown by participant background.

## Running it

```
pip install -r src/requirements.txt
```

Set an API key as an environment variable (never commit keys):

```
export GOOGLE_API_KEY=your-key-here
python src/run_experiment.py --set simple --provider gemini --max-trials 15 --delay-seconds 13 --stimuli-dir stimuli
```

On Windows PowerShell use `$env:GOOGLE_API_KEY="your-key-here"`; in cmd use `set GOOGLE_API_KEY=your-key-here`.

Train and probe the network:

```
python src/train_own_model.py --set conjunctive --epochs 30 --seed 0
```

Options for both scripts are described in `src/README_RUNNER.txt`.

Human trial: open `human_trial/index.html` in a browser. Result lines are shown at the end of each run to copy.

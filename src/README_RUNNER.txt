Setup:
  pip install -r requirements.txt
  Set whichever API key(s) you need as environment variables:
    export ANTHROPIC_API_KEY=your-key-here
    export OPENAI_API_KEY=your-key-here
    export GOOGLE_API_KEY=your-key-here
  (On Windows use: set VAR=value)

Run:
  python run_experiment.py --set simple
  python run_experiment.py --set conjunctive --provider openai
  python run_experiment.py --set disjunctive --provider gemini

Options:
  --provider   anthropic (default), openai, or gemini
  --model      override the default model for that provider, e.g.
               --provider anthropic --model claude-opus-4-1
               --provider openai --model gpt-4o-mini
               --provider gemini --model gemini-2.0-flash
  --stimuli-dir   path to the stimuli folder (default: stimuli, i.e. run from inside this bundle folder)
  --out-dir       where trial logs and results get written (default: results)

Output per run (filenames prefixed by provider so you can compare side by side):
  results/<provider>_<set>_trials.csv    every trial: stimulus attributes, guess, actual category, correct/wrong
  results/<provider>_<set>_result.json   overall accuracy, the model's stated rule, and the actual rule

You only need the API key for the provider you're actually testing -- no need to set
all three unless you're comparing models.

---
Training your own small model (completely free, runs locally, no API key)

Run:
  python train_own_model.py --set simple
  python train_own_model.py --set conjunctive
  python train_own_model.py --set disjunctive

What it does:
  - Trains a small neural network one stimulus at a time (like the LLM protocol:
    guess first, then update on the true label), across multiple passes over the
    same 40 stimuli, so it actually converges
  - After training, it doesn't ask the network to "state a rule" in words (it can't --
    it's just numbers). Instead it fits a small decision tree to imitate the network's
    predictions across every possible stimulus combination, and prints that tree's
    if/else splits as the network's implicit "rule"
  - For the conjunctive set specifically, it also tests two "probe" stimuli that
    decouple color and transparency (which always moved together during training)
    to check whether the network actually learned the two-feature rule or took a
    shortcut on just one feature

Options:
  --epochs     how many passes over the 40 stimuli (default 30 -- simple rule may
               converge with fewer, conjunctive/disjunctive may need more)
  --lr         learning rate (default 0.05)
  --seed       random seed, change this to see if results are consistent across runs

Output per run:
  results/network_<set>_trials.csv           every single online update: stimulus, guess, correct
  results/network_<set>_learning_curve.png    rolling accuracy plot over all trials
  results/network_<set>_result.json           final accuracy, extracted rule, and (for
                                               conjunctive) the shortcut-check probe results

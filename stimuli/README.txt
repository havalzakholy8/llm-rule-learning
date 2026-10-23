Stimulus set for AI/human categorization experiment
Three sub-sets, each with 40 stimuli (20 per category), split into square PNG images:

- simple/       shape-only rule
- conjunctive/  two-attribute AND rule
- disjunctive/  OR-of-two-conjunctions rule (hardest)

Each folder has labels.csv (filename, all 5 attribute values, category).
rules_ANSWER_KEY.json holds the ground-truth rule for each set -- do NOT show this
to the AI model during training.

Suggested protocol (implicit training + explicit rule extraction):
1. Show the model stimuli one at a time from labels.csv, in random order,
   WITHOUT revealing the rule -- only tell it "category 1" or "category 2"
   after each guess (right/wrong feedback), same as the chick-sexing paradigm.
2. After it reaches a stable accuracy (or after a fixed number of trials),
   stop giving feedback and ask: "What makes an object a member of category 2?"
3. Compare its stated rule to rules_ANSWER_KEY.json.
4. Repeat across simple -> conjunctive -> disjunctive to see where explicit
   rule articulation breaks down.


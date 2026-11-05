## CS260 HW06: Information Theory

Name 1:

---

### Written component

Complete `HW06_written_problems.pdf` by hand or typeset your answers using
`HW06_written_problems.tex`. Show your work, use base-2 logarithms, and use
only a calculator for arithmetic. Submit the completed worksheet as
`HW06_written_problems.pdf` with your code.

### Analysis

1. What is the expected number of bits per character for your source-based
   Shannon code? Compute `sum(p(c) * length(code(c)))` using the source
   character probabilities. Is this higher or lower than you expected?
   Explain what you used as a comparison. As a check, compare the expected
   length with the source entropy: a Shannon code satisfies
   `H(C) <= expected length < H(C) + 1` for this nontrivial source alphabet.
2. For each of tennis, movies, heart, and diabetes, report the feature
   selected by information gain, its gain, the feature selected by training
   accuracy, and its accuracy. Include any ties. Did the two criteria ever
   select different features? Explain your results for each dataset,
   considering the class distributions within the feature-value groups.

Document how you resolve ties between features, as requested in Part 2.

#### Question 1 response

Expected bits per character:

Source entropy:

Comparison and explanation:

#### Question 2 response

| Dataset | Feature selected by information gain | Information gain (bits) | Feature selected by training accuracy | Training accuracy | Ties |
| --- | --- | --- | --- | --- | --- |
| Tennis | | | | | |
| Movies | | | | | |
| Heart | | | | | |
| Diabetes | | | | | |

Tie-breaking rule:

Explain the results for each dataset using the class distributions within
feature-value groups:

- Tennis:
- Movies:
- Heart:
- Diabetes:

---

### Homework Questionnaire

(None of your answers below will affect your grade; this is to help refine homework
assignments in the future)

1. Approximately, how many hours did you take to complete this homework? (provide
  your answer as a single integer on the line below)

2. How difficult did you find this homework? (1-5, with 5 being very difficult and 1
  being very easy)

3. Describe the biggest challenge you faced on this homework:

### Implementation and checks

Use Python 3.9 or newer. Keep the supplied type annotations and signatures.
Implement the `NotImplementedError` stubs in `Shannon.py`, `Partition.py`, and
`process_contents()` in `encode_decode.py`. The ARFF reader, file helpers,
argument parsing, reporting, and `main()` functions are provided.

Run the public checks from this directory:

```bash
python3 -m unittest -v test_hw06
python3 encode_decode.py
python3 best_feature.py -r data/tennis_train.arff
```

The tests intentionally fail on the unimplemented starter. They illustrate
expected behavior but are not exhaustive. Computational methods must return
values without printing, reading files, or prompting. Preserve input data and
use dictionary order to break exact feature-score or character-probability ties.
Detailed return types, invalid-input behavior, and helper contracts are in the
docstrings. `get_binary_codes()` returns a copy of the codebook;
`expected_length()` returns probability-weighted bits per character.

`Partition` exposes `entropy()`, `conditional_entropy(feature)`,
`information_gain(feature)`, `classification_accuracy(feature)`,
`best_feature()`, and `best_feature_by_accuracy()`. Labels from the reader are
-1 and +1; processed feature values are strings. Do not smooth probabilities.
# DS_HW6

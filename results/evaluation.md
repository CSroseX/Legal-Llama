# Evaluation results

Produced by [`notebooks/04_evaluate.ipynb`](../notebooks/04_evaluate.ipynb) on a Kaggle T4. The run took 2.4 hours.

## The exam

- **1,000 contract pages** cut from the **102 test contracts** of [CUAD](https://huggingface.co/datasets/theatticusproject/cuad-qa). None of these contracts were used for training or validation.
- **500 pages contain the answer** and **500 don't**. Each page is about 512 tokens, up to 1,024 when a clause is long.
- All three models got exactly the same pages and questions. The page sampling uses a fixed seed (3407), so the exam can be rebuilt identically.
- The base model and the new model got the new instruction. The previous version got its own (bare Alpaca) format, so it wasn't put at a disadvantage.
- Every model used greedy decoding (no randomness) and at most 200 new tokens.

## Scores

| Score | What it measures | Base Llama 3 8B | Previous version | **Llama-3-Legal-8B** |
|---|---|:---:|:---:|:---:|
| Word overlap ↑ | Word-level F1 against the correct clause, on pages that contain it | 0.003 | 0.185 | **0.735** |
| Answers found ↑ | Share of those pages with word overlap ≥ 0.5 | 0.002 | 0.146 | **0.738** |
| Missed ↓ | Said "Not found." although the answer was on the page | 0.868 | 0.000 | **0.082** |
| Correct "Not found." ↑ | Said "Not found." on pages without the answer | 0.836 | 0.000 | **0.958** |
| Not quoted from the contract ↓ | Share of non-"Not found." answers that aren't verbatim text from the page, checked by string matching | 0.932 | 0.877 | **0.004** |

Word overlap uses the standard SQuAD normalisation: lower-case, punctuation and articles removed.

## By clause type (new model)

Clause types with the most test pages. Score = average word overlap on pages that contain the clause.

| Clause type | Score | Test pages |
|---|:---:|:---:|
| Renewal Term | 0.971 | 13 |
| Governing Law | 0.967 | 28 |
| Expiration Date | 0.960 | 35 |
| Document Name | 0.951 | 47 |
| Agreement Date | 0.929 | 35 |
| Effective Date | 0.910 | 37 |
| Notice Period To Terminate Renewal | 0.896 | 10 |
| Anti-Assignment | 0.855 | 25 |
| … | | |
| Change Of Control | 0.284 | 12 |
| Parties | 0.249 | 43 |

The run also reported low scores for No-Solicit Of Employees (0.46), Non-Disparagement (0.43), Third Party Beneficiary (0.39), Unlimited/All-You-Can-Eat-License (0.26), Liquidated Damages (0.24) and Most Favored Nation (0.00). Each of these had only **1–5 test pages**, which is too few to draw conclusions.

## Whole-contract test

The new model was run on three complete test contracts (about 3,000 tokens each). Each contract was cut into overlapping 512-token pages, every page was asked, and the answers were merged. Score = word overlap with all correct passages, or 1.0 for a correct "Not found.".

| Contract | Tokens | Parties | Agreement Date | Governing Law | Expiration Date | Anti-Assignment | Non-Compete |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Centrack International – Web Site Hosting Agreement | 2,924 | 0.50 | 1.00 | 0.68 | 1.00 | 1.00 | 1.00 |
| TriZetto Group – Technical Infrastructure Maintenance Agreement | 3,201 | 0.63 | 0.60 | 1.00 | 0.56 | 1.00 | 1.00 |
| ScanSource – Distributor Agreement | 3,244 | 0.28 | 0.63 | 1.00 | 0.00 | 1.00 | 1.00 |

**Average: 0.771** over 18 questions.

What the misses show:
- **Governing Law (Centrack), 0.68:** the answer was correct, but it was quoted twice, once cut off at a page edge and once complete. The merging code in the model card now keeps only the complete version.
- **Agreement Date (TriZetto), 0.60:** the correct date was returned, along with two other dates from other pages (likely signature dates).
- **Parties:** the main company names were found, but not every nickname the contract uses (e.g. "the Customer").

## Safety test after merging

Before publishing, the merged model was re-tested on 100 exam pages (60 with the answer, 40 without) and compared with the evaluation results on the same pages.

| | Word overlap | Correct "Not found." | Not quoted from the contract |
|---|:---:|:---:|:---:|
| Before merging | 0.738 | 1.000 | 0.000 |
| After merging (published model) | 0.755 | 1.000 | 0.000 |

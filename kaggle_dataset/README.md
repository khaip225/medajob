# VN Job Rec Graph

Training data for the heterogeneous-GNN job recommender (`Final_Project_Kaggle (2).ipynb`).
Built by `Job_Recommended/ml/build_training_dataset.py` (seed 2026) from the data the backend serves:
`import_db/COMBINED_DATA_PROCESSED1.csv` (39,036 job rows) and `import_db/USER_DATA_PROCESSED1.csv` (3,983 seeker rows).

| file | rows | content |
|---|---|---|
| `jobs.csv` | 38,101 | cleaned postings (`j_idx` = graph index, `db_job_id` = backend id) |
| `users.csv` | 2,096 | cleaned seekers (`u_idx` = graph index, `db_user_id` = backend id) |
| `interactions.csv` | 49,664 | simulated applications `u_idx, j_idx, split (train/val/test), pref_rank` |
| `hard_negatives.csv` | 206,721 | relevant-but-infeasible jobs per user (province / experience / degree violated) |
| `meta.json` | | taxonomies, unit coordinates, coverage, behaviour-model parameters, baseline check |

## Cleaning
* Jobs: duplicate re-postings removed; unknown province ('Khác', 7%) recovered from the street address
  (districts, industrial parks, English spellings); 63 provinces mapped to the 34 post-2025 units;
  unknown industry recovered via a 35-role occupation taxonomy shared with users; salary in million VND/month
  (`salary_type` 3 = negotiable had imputed numbers -> NaN; 'Trên X' / 'Đến X' keep only the real bound);
  minimum degree parsed from requirement text ('ưu tiên' mentions ignored); level from job_function/title; 9 benefit groups.
* Users: duplicate UserIDs and cloned profiles removed; desired salary parsed from the original text
  (numeric columns were mis-scaled); degree level parsed from the education text; multi-province wishes kept;
  occupation, level and part-time intent extracted; templated driver profiles capped at 500
  (they were 50% of rows but only ~1% of jobs are driving jobs).

Coverage — jobs: location 98.2%, salary 60.8%, degree 68.8%, occupation 98.2%;
users: location 100.0%, salary 11.3%, degree 73.0%, occupation 99.0%.

## Labels (simulated — there is no real interaction log)
Each user applies to ~20 jobs drawn by Gumbel-top-k (Plackett–Luce, T=0.4) from a utility that combines
role/industry/text fit, job attractiveness (salary vs role median, benefits, employer size, latent popularity),
a latent taste shared by users of the same role (only learnable from the graph), and constraint penalties whose
strength depends on observable user attributes (number of wished provinces, age, marital status, stated salary).
Positives violating a hard constraint: province 2.2%,
experience 1.1%, degree 1.0%
(random job: 70.0% / 11.7% / 20.6%).

## Difficulty check (test split, rank all jobs, train+val excluded)
| ranker | Recall@20 | NDCG@20 |
|---|---|---|
| random | 0.0000 | 0.0000 |
| popularity | 0.0765 | 0.0442 |
| content_tfidf | 0.0807 | 0.0464 |
| content+feasible | 0.3329 | 0.2220 |
| userKNN_CF | 0.4764 | 0.3578 |
| content+feasible+CF | 0.5551 | 0.4206 |
| ORACLE_true_utility | 0.7587 | 0.6548 |

`ORACLE_true_utility` uses the hidden utility itself: it is the ceiling, not a model. Report results as
"on simulated interactions"; use the hand-label set exported by the notebook for an independent check.

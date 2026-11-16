# Running the training notebook on Kaggle

You upload **two things** to Kaggle:

| What | Local path | Becomes on Kaggle |
|---|---|---|
| The dataset folder | `Final/kaggle_dataset/` | a Kaggle **Dataset** (e.g. `vn-job-rec-graph`) |
| The notebook | `Final/Final_Project_Kaggle_v3.ipynb` | a Kaggle **Notebook** that uses that dataset as input |

The notebook does **not** download any data. It does not use the HuggingFace VietJobs dataset, and it does not clone GitHub for `USER_DATA_PROCESSED1.csv`. It reads everything from the dataset you attach.
The only things it fetches from the internet are Python packages (`pip install`) and the weights of the
text-embedding model `intfloat/multilingual-e5-small`. Those weights come from the HuggingFace Hub, which is why you may see a
"sending unauthenticated requests to the HF Hub" warning. That warning is harmless.

---

## Step 1: Upload the dataset

`Final/kaggle_dataset/` contains:

```
jobs.csv              38,101 cleaned job postings   (~104 MB)
users.csv              2,098 cleaned job seekers
interactions.csv      49,664 applications with split = train / val / test
hard_negatives.csv    relevant-but-infeasible jobs per user
meta.json             taxonomies, province coordinates, generation parameters
README.md             dataset description
dataset-metadata.json only needed for the Kaggle CLI (option B)
```

**Option A: web UI (easiest)**
1. Go to https://www.kaggle.com/datasets → **New Dataset**.
2. Drag in **all files inside** `kaggle_dataset/`. Upload the files themselves, not a zip of the folder.
3. Title: `VN Job Rec Graph` (the URL slug becomes `vn-job-rec-graph`). Keep it **Private**.
4. Click **Create** and wait until processing finishes (about 1–2 minutes).

**Option B: Kaggle CLI**
```bash
pip install kaggle                       # needs ~/.kaggle/kaggle.json (Kaggle → Settings → Create New Token)
# edit kaggle_dataset/dataset-metadata.json: replace YOUR_KAGGLE_USERNAME with your username
kaggle datasets create -p kaggle_dataset            # first time
kaggle datasets version -p kaggle_dataset -m "v2"   # later updates
```

---

## Step 2: Import the notebook

1. https://www.kaggle.com/code → **New Notebook**.
2. Menu **File → Import Notebook** → choose `Final_Project_Kaggle_v3.ipynb`.
   (If you used an older copy before, import this file again. It replaces the old cells that downloaded VietJobs and cloned GitHub.)

## Step 3: Attach the dataset and set the session options

In the notebook's right-hand panel:

1. **Input → + Add Input → Your Datasets →** select `vn-job-rec-graph` → **Add**.
   It is mounted read-only under `/kaggle/input/...`. Section 2 of the notebook finds it automatically by searching for `interactions.csv`,
   so the dataset name does not matter.
2. **Session options → Accelerator → GPU T4 x2** (or P100).
3. **Session options → Internet → On.** This requires a phone-verified Kaggle account.
   - With Internet **off**, `pip install` fails quietly. **PyTorch Geometric is not preinstalled on Kaggle**, so the first cell
     would then stop at `import torch_geometric`. Keep Internet on.

## Step 4: Run

- Click **Run All** for an interactive run, or **Save Version → Save & Run All (Commit)** to run it in the background
  (you can close the browser). The full run trains 8 models × 3 seeds plus B4, pruning, exports and figures.
  Expect roughly **1–2 hours on a T4**. Kaggle's limit is 12 hours per session.
- To check that everything works first, add a new first cell with:
  ```python
  import os; os.environ["SMOKE"] = "1"   # 300 users, 4 epochs, ~5–10 minutes
  ```
  Then delete it for the real run.

Optional environment overrides (put them in a first cell the same way): `EPOCHS` (default 100), `BATCH` (default 1024),
`DATA_DIR` (only if auto-detection fails, e.g. `/kaggle/input/vn-job-rec-graph`).

### What you should see in the log
```
DATA_DIR = /kaggle/input/vn-job-rec-graph {...}
{'n_jobs': 38101, 'n_users': 2098, 'n_interactions': 49664, ...}
38101 job, 2098 ứng viên, 49664 lượt ứng tuyển, ...
Ứng viên có đủ train/val/test: 2098/2098 ...
Vi phạm trong nhãn dương: {...province ~0.02...} | job ngẫu nhiên: {...province ~0.70...}
REF_Popularity   test NDCG@20=...
REF_UserKNN      test NDCG@20=...
  [B0_SAGE_2nut s0] ep  5 loss=... val NDCG@20=...
```
`REF_Popularity` and `REF_UserKNN` are simple non-deep-learning reference points. A well-trained GNN should beat
popularity clearly. Compare B3/B4 against UserKNN, and especially on the violation columns (`viol_*@20`).
For reference, the dataset build checked: popularity ≈ 0.044, user-KNN ≈ 0.36, best possible (true simulated preferences) ≈ 0.65 NDCG@20.

---

## Step 5: Get the results

Everything is written to `/kaggle/working/` → **Output** tab (or the version's *Output* section after a commit):

| File | Content |
|---|---|
| `results_bundle.zip` | everything in `results/` except `artifacts/` (tables, `fig_data/`, `figures/`): download this |
| `artifacts_bundle.zip` | `results/artifacts/` only (model, embeddings, top-K for the backend; large) |
| `results/figures/` | fig01–fig11 as PNG (300 dpi) and SVG, Vietnamese labels |
| `results/results_mean_std.csv` | main table: Recall/NDCG@20, MRR, violation rates (raw and `+mask`), mean ± std over seeds |
| `results/b4_rerank.csv` | B3 + filter, B4a (linear rerank), B4b (LambdaRank) |
| `results/complexity.csv`, `pruning.csv`, `hetero_vs_homo.csv`, `gate_means.csv` | analysis tables |
| `results/handlabel_to_label.csv` | pairs to label by hand (0/1/2) for an independent check |
| `results/artifacts/` | for the backend: `model.pt`, GNN and text embeddings, `topk_b4.npz` (top-100 per user), `jobs.csv` / `users.csv` with `db_job_id` / `db_user_id`, `meta.json` |

**Figures (section 15):** fig01 model comparison · fig02 accuracy vs. constraint violations · fig03 B4 reranking · fig04 edge pruning ·
fig05 hetero vs. homo · fig06 learned gate weights · fig07 cost · fig08a/b training curves · fig09 data overview ·
fig10 heterogeneous graph schema · fig11 ego-subgraph of 3 sample users.
To redraw them on your own machine (e.g. to change titles), put `results_bundle.zip` next to the notebook,
`pip install numpy pandas matplotlib networkx`, then run only cell **15b** and cell **15d**. Output: `results_local/figures/`.

**Hand labels (optional):** fill the `label` column of `handlabel_to_label.csv`, upload it as another dataset, add it as input,
add a first cell with `os.environ["HANDLABEL_CSV"] = "/kaggle/input/<that-dataset>/handlabel_to_label.csv"`, and re-run section 12.

**Using the artifacts in the backend:** `j_idx` / `u_idx` are row numbers in `artifacts/jobs.csv` / `users.csv`.
`db_job_id` / `db_user_id` are the ids that `import_db/import_jobs.py` / `import_users.py` assign when importing
`COMBINED_DATA_PROCESSED1.csv` / `USER_DATA_PROCESSED1.csv` into a fresh database. `db_*_ids` lists every duplicate row that was merged into the same entry.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Không tìm thấy dataset ...` | The dataset is not attached (Step 3.1), or you uploaded a zip of the folder. Re-upload the individual files, or set `DATA_DIR`. |
| `ModuleNotFoundError: torch_geometric` | Internet is off. Turn it on and restart the session. |
| `!! Không dùng được sentence-transformers ... dự phòng TF-IDF + SVD` | The e5 model could not be downloaded (no internet). The run continues with TF-IDF embeddings, so results are a bit lower. |
| CUDA out of memory | Restart and run with `BATCH=512`. |
| Log still shows `JOBS_CSV = .../VietJobs.csv` or `USERS_CSV = ...` | You are running an old copy of the notebook. Import `Final_Project_Kaggle_v3.ipynb` again. |

## Regenerating the dataset (only if you change the cleaning or label rules)

```bash
cd Final/Job_Recommended
python -m ml.build_training_dataset --check      # ~5 min, writes ../kaggle_dataset, prints the baseline table
```
Then upload a new dataset version (`kaggle datasets version ...` or **New Version** in the web UI) and re-run the notebook.
The source CSVs are taken from `import_db/` if they exist, otherwise from git history automatically.

> Note for the report: there is no real application log in the project, so `interactions.csv` is **simulated**
> by a documented behaviour model (see `kaggle_dataset/README.md` and `meta.json → behaviour_model`).
> Present the metrics as "on simulated interactions" and use the hand-label set as the independent check.

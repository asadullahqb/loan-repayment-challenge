# Working Notes — Loan Repayment Challenge

Running journal. Rule: log what we tried (failed/succeeded), what we considered then scratched,
what we decided, and time spent — so we can report total time and rationale at the end.

## Log

### Setup & problem framing
- **Decided:** frame as a credit-risk pipeline — eligibility (rules) → fraud (outsourced to
  Clarity) → PD (core model) → LGD/EAD (downstream) → profitability (expected loss).
- **Scratched:** "predict interest rate" as the target — too naive; replaced with PD/LGD/EAD.
- **Scratched:** fine-tuning LLMs for credit scoring — poor explainability; GBM / logistic
  regression is the established benchmark for tabular credit risk.
- **Decided:** GBM + logistic-regression scorecard; evaluate AUC / KS / Gini; target AUC >= 0.70.
- **Decided:** src-layout `tht` package = the "code" deliverable; `THT_DATA_DIR` env override.

### EDA & target
- **Succeeded:** target built via `build_target`. Resolved cohort = 28,524 loans, 60.2% default
  (17,175 bad / 11,349 good); 95% of rows dropped as ambiguous.
- **Succeeded:** found key signals — `nPaidOff` (AUC 0.603), `leadType`, `clearfraudscore`
  (AUC 0.634), total fraud indicators, `apr` (0.589), `state`.
- **Insight:** `hasCF` is a selection marker (riskier applicants get fraud-checked), not a
  causal feature — exclude from PD.

### Outcomes analysis → pipeline direction
- **Succeeded:** ran univariate AUC + segment analysis to guide the build (see eda_1.ipynb).
- **Decided:** PD is the headline; combine `nPaidOff + apr + clearfraudscore + leadType + state
  + fraud-indicator-count` (best single feature 0.63, ensemble should clear 0.70).
- **Decided:** keep fraud outsourced (consume Clarity as features, handle the 17% missing-Clarity).
- **Decided:** LGD/EAD = lightweight recovery-rate summary only (EAD is deterministic for
  fixed-term loans). Flag selection bias (model trains on the funded 5%, not all applicants).
- **Open:** refine target edge cases (`Settlement Paid Off`, `Charged Off Paid Off`).

### Sharing / Colab / packaging
- **Failed:** `pip install -e .` (editable) not importable in Colab's live kernel → `No module
  named tht`.
- **Succeeded:** added a Colab bootstrap cell with a `sys.path` fallback to `src/`, so
  `import tht` works locally and in Colab.
- **Scratched:** publishing to PyPI / bundling original data — the brief forbids shipping the
  original data; use `pip install git+https://...` or a `files.upload()` step instead.
- **Decided:** share a Colab link (assessor supplies their own data) rather than bundle data.

### Evaluation upgrade: time-wise ECL simulation (IFRS-9 style)
- **Decided:** go beyond a static Monte Carlo profitability sim → add a **time-wise** ECL
  simulation with a term structure, aligning with IFRS 9 / MFRS 9:
  `ECL = PD(t) × LGD(t) × EAD(t) × Discount Factor(t)`.
- **Design:**
  - PD term structure via survival/hazard on `installmentIndex` (time-to-default derived from
    the payment schedule).
  - EAD(t) from the amortization schedule (declining outstanding principal).
  - LGD(t) from cumulative recovery vs outstanding (workout curve).
  - Staging + SICR proxy (missed/skipped payment → Stage 1 → Stage 2 → Stage 3).
  - FLI scenarios (Baseline/Upside/Downside) via external macro data (e.g., FRED unemployment/GDP).
- **Data gaps to close:** no explicit days-past-due / charge-off date (derive from
  `installmentIndex`); `originatedDate` 92% missing (use installment index as time axis);
  no macro series (import FRED — brief allows external data).

### Roadmap: Markov migration + macro overlay (after the IFRS-9 ECL)
- **Decided (build order):**
  1. PD term structure — reduced-form hazard on `installmentIndex`.
  2. Time-wise IFRS-9 / MFRS-9 ECL evaluation (PD(t) × LGD(t) × EAD(t) × DF, SICR staging,
     FLI scenarios).
  3. **Markov migration model + macro overlay** — multi-state transition matrix
     (delinquency states → default), with transition probabilities linked to macro variables
     (point-in-time / FLI calibration).
- **Why:** the hazard model is a single path (state → default); the Markov matrix captures the
  delinquency *path* (current → 30/60/90 dpd → default), and the macro overlay makes migration
  point-in-time across Baseline/Upside/Downside.
- **Data:** derive delinquency states per installment from `paymentStatus` / `isCollection`;
  macro series from FRED.

### PD model built + benchmark (this iteration)
- **Corrected benchmark:** the "Grinsztajn" cite (notes_1) is *GBM vs deep learning*, not a 70%
  threshold. Our own rubric (notes_1): AUC <70% reject, target 80%, >80% extra credit; KS 30%
  decent / 40% target / >50% suspicious.
- **Succeeded:** built the PD model (`tht/features.py`, `tht/model.py`) — logistic +
  HistGradientBoosting + LightGBM + XGBoost, leak-free preprocessor, AUC/KS/Gini eval.
- **Leakage caught (x2):**
  - payment-derived outcome features (paydown ratio, successful-payment count, tenure) → AUC
    0.995 — the features *are* the outcome.
  - behavioral delinquency (NSF/skip/cancel counts) → AUC 0.964, KS 0.835 — NSF count is
    effectively the default event (violates our own ">50% KS suspicious" rule).
- **Honest application-time result:** AUC ~0.72 / KS ~0.32 (HGB best 0.721, logistic 0.712).
  Internal "mini-bureau" (prior-loan history from `anon_ssn`) adds only ~+0.002 — most borrowers
  have a single loan.
- **External data exhausted:** no external bureau/tradeline can join (anonymized `anon_ssn`);
  FRED state macro is joinable on `state` but marginal (and network-blocked locally). Concluded
  the 80% target is not reachable with the given data without individual credit attributes.
- **MLOps:** added `tht/monitoring.py` + `scripts/demo_drift.py` — PSI drift monitor; mild drift
  → auto-retrain, severe drift → escalate to MLOps engineer.
- **External module:** added `tht/external.py` (FRED state/macro join) for a Colab run to
  confirm the (small) macro lift.

## Time log

Estimated (simulated) hours — reconcile with actual time before reporting.

| Task | Time (h) |
|------|----------|
| Read brief + business framing + domain research (IFRS-9 / credit risk) | 3.0 |
| Environment & package setup (`tht` src-layout, deps) | 1.0 |
| Data dictionaries + EDA (eda_1.ipynb) | 4.0 |
| Target construction (`build_target`, status rules) | 1.0 |
| Outcomes analysis → pipeline direction (univariate AUC / segments) | 2.0 |
| Colab / bootstrap / packaging / sharing design | 2.5 |
| IFRS-9 / ECL / Markov roadmap design | 1.5 |
| **Total** | **15.0** |

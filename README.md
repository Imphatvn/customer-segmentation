# Customer Segmentation and Campaign Analytics

End-to-end customer segmentation for retail: 34,060 customers are segmented from demographic and annualized-spending features, and the resulting segments are linked to 100,000 sampled transaction baskets to mine product associations and design segment-specific campaign hypotheses.

**Pipeline:** 1 EDA → 2 Preprocessing → 3 Outlier screening → 4 Feature preparation → 5 Clustering → 6 Profiling & selection → 7A Basket auditing → 7B Association rules → 7C Campaign design

## Results at a glance

| Item | Result |
|---|---|
| Selected model | K-Means, k = 6, on standardized log-spending features (10 spending categories + promotion share) |
| Fitting population | 33,715 customers with positive tenure; 345 zero-tenure customers deferred under a documented policy |
| Model stability | Minimum seed-pair ARI ≈ 0.993; 80%-customer-subsample refits ARI 0.975–0.982 |
| Basket analysis | 98,247 audited baskets (24,714 customers) → 2,803 directed pair rules; 2,530 pass a descriptive holdout check |
| Campaign output | Six segment-specific hypotheses: four retained for commercial review, one exploratory, one deprioritized |

| Segment | Share | Profile in one line |
|---|---|---|
| Established Multi-Category Shoppers | ~32% | Moderate spend across categories, low promotion share, longest tenure |
| Electronics and Gaming Shoppers | ~10% | Category-focused electronics and videogames spending |
| Low-Spending, Promotion-Oriented Shoppers | ~16% | Lowest total spend; about half of purchases on promotion |
| Promotion-Heavy Regular Shoppers | ~15% | Moderate spend with a 63% median promotion share |
| High-Spending Multi-Category Shoppers | ~13% | High spend across categories; 37.5% of recorded annualized spending |
| Vegetable-Focused Shoppers | ~13% | Vegetables stand out; low meat and fish |

<!-- Optional: add one figure once the repo is public, e.g.
     ![Cluster spending profiles](reports/figures/clustering/relative_profiles.png) -->

## Key design decisions

Judgment calls that shaped the analysis, recorded so they can be questioned and revisited:

1. **Zero-tenure customers are deferred, not assigned an invented tenure.** Annual spending divides lifetime spend by tenure, so the 345 customers with zero tenure are kept out of fitting rather than given a fabricated denominator. Their assignment policy remains a documented open item, and the exported register is explicitly labeled incomplete until it is resolved.
2. **Outlier flags are diagnostics, not removal rules.** Four detectors (IQR, DBSCAN, K-Means distance, SOM) screen for unusual spenders, but no customer is removed automatically. Flags stay visible during profiling — a high-spending segment is not treated as a data error.
3. **Ambiguous invoices are set aside, not guessed.** 371 invoice IDs are linked to more than one customer; all 743 affected rows are preserved in an audit file and excluded from segment-level analysis rather than arbitrarily assigned to one customer.
4. **Full log-scaled features were chosen over PCA.** PCA retained 8 of 11 components (92.7% of variance) and produced near-identical assignments (ARI 0.973 at k = 6). The interpretable 11-feature representation was preferred because compression brought no practical gain.
5. **Six clusters over three or five.** The three-cluster solution placed 47% of customers in a single group; the five-cluster solution showed a customer-subsampling instability (minimum ARI 0.536 against its reference). The six-cluster solution combines distinct, interpretable profiles with strong stability.
6. **Campaigns are hypotheses, not validated offers.** Pair rules come from a fixed discovery/holdout customer split that has already been inspected, so all rule evidence is descriptive. No discount amounts, uplift forecasts, or ROI claims are made; each proposal instead carries an evaluation plan and is held at pending commercial review.

## Environment setup

From the repository root in PowerShell, create and activate a virtual environment and install the notebook dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project targets Python 3.12 to match the saved experiment metadata, and the direct dependencies are pinned in `requirements.txt`. In VS Code, select the `.venv` Python interpreter and its Jupyter kernel before running notebooks.

## Notebook run order

Run each notebook from top to bottom. The path setup supports starting Jupyter from either the repository root or the `notebooks` folder.

| Order | Notebook | Main inputs | Main handoff |
|---|---|---|---|
| 1 | `1_EDA.ipynb` | `data/raw/customer_info.csv` | Exploratory analysis; no pipeline dataset export |
| 2 | `2_Preprocessing.ipynb` | `data/raw/customer_info.csv` | `data/preprocessed/customer_info_clean.csv` |
| 3 | `3_Outlier_Analysis_check.ipynb` | `data/preprocessed/customer_info_clean.csv` | `data/preprocessed/customer_outlier_flags.csv` and outlier diagnostics |
| 4 | `4_Feature_Preparation.ipynb` | Clean customer data and outlier flags | Feature matrices in `data/features/` and fitted transforms in `models/feature_preparation/` |
| 5 | `5_Clustering.ipynb` | Prepared feature matrices | Candidate assignments and models in `models/clustering/` |
| 6 | `6_Cluster_Profiling.ipynb` | Candidate assignments, models, and customer profiles | Selected segmentation and `customer_assignment_register_INCOMPLETE.csv` |
| 7A | `7_Association_Rules_and_Campaigns.ipynb` | `data/raw/customer_basket.csv` and the assignment register | `data/basket_analysis/assigned_baskets_clean.csv` and `product_category_mapping.csv` |
| 7B | `7B_Association_Rule_Mining.ipynb` | Audited baskets and a reviewed product-category mapping | Pair-rule evidence in `reports/tables/association_rules/part_b/` |
| 7C | `7C_Campaign_Recommendations.ipynb` | Part B rule evidence | Draft proposals and checklist in `reports/tables/association_rules/part_c/`; no campaigns are launched |

Notebook 4 reads `customer_info_clean.csv` and `customer_outlier_flags.csv` and applies the fitting/deferred-customer policy there. Notebook 3's separate `customer_info_for_features.csv` export is not the Notebook 4 input.

Notebook 7A prepares and audits transaction baskets; it does not mine association rules. Run 7B before 7C. The campaign outputs are exploratory proposals, not evidence of causal uplift or commercial approval.

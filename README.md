# customer-segmentation
End-to-end customer segmentation project using demographic, transactional, and basket-level retail data to identify actionable customer segments and design targeted marketing campaigns.

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

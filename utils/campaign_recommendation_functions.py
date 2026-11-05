"""Campaign hypotheses linked to Part B evidence; no launch or targeting actions."""
from pathlib import Path
import pandas as pd

REQUIRED = {"cluster", "level", "antecedent", "consequent", "joint_count", "joint_customers",
            "confidence", "lift", "holdout_joint_count", "holdout_confidence", "holdout_lift",
            "holdout_minimum_evidence_check", "confidence_difference_vs_rest"}


def campaign_specs():
    rows = [
        (0, "Established Multi-Category Shoppers", "category", "fish", "nonalcohol_drinks",
         "Meal companion", "Show a relevant non-alcoholic drink alongside a fish purchase; begin without a discount.",
         "Category-level pairing; exact SKU compatibility must be reviewed."),
        (1, "Electronics and Gaming Shoppers", "item", "frozen smoothie", "cottage cheese",
         "Food pairing pilot", "Test a small complementary-food recommendation, not a gaming-lifestyle persona.",
         "Food pairing needs merchandising review; not a defining electronics-segment preference."),
        (2, "Low-Spending, Promotion-Oriented Shoppers", "item", "fresh tuna", "barbecue sauce",
         "Meal preparation suggestion", "Test a recipe-compatible sauce recommendation with tuna; any coupon amount remains unset.",
         "Check exact product suitability and margin before offering a bundle."),
        (3, "Promotion-Heavy Regular Shoppers", "category", "fish", "hygiene",
         "Cross-aisle essentials", "Test a separate household-essentials reminder alongside a fish basket, not a combined food product bundle.",
         "Cross-category association does not demonstrate complementarity; avoid blanket basket discounts."),
        (4, "High-Spending Multi-Category Shoppers", "item", "carrots", "tomatoes",
         "Vegetable pairing", "Test a recipe-led tomato suggestion alongside carrots, initially without price reduction.",
         "Do not infer premium preferences or willingness to pay from the segment name."),
        (5, "Vegetable-Focused Shoppers", "item", "strong cheese", "frozen vegetables",
         "Vegetable meal companion", "Test a recipe-compatible frozen-vegetable suggestion alongside strong cheese.",
         "Small supporting holdout count; prioritize a limited exploratory pilot rather than broad rollout."),
    ]
    return pd.DataFrame(rows, columns=["cluster", "segment_name", "level", "antecedent", "consequent",
                                      "campaign_name", "proposed_mechanic", "commercial_caution"])


def load_rule_evidence(path):
    rules = pd.read_csv(path)
    missing = REQUIRED - set(rules.columns)
    if missing:
        raise ValueError(f"Missing Part B columns: {sorted(missing)}")
    keys = ["cluster", "level", "antecedent", "consequent"]
    if rules.duplicated(keys).any():
        raise ValueError("Duplicate directed rule keys in Part B export")
    if not rules["holdout_minimum_evidence_check"].isin([True, False, "True", "False"]).all():
        raise ValueError("Unexpected holdout flag values")
    rules["holdout_minimum_evidence_check"] = rules["holdout_minimum_evidence_check"].astype(str).str.lower().eq("true")
    return rules


def build_campaign_plan(rules, specs=None):
    specs = campaign_specs() if specs is None else specs.copy()
    if specs["cluster"].duplicated().any():
        raise ValueError("Expected one draft campaign per segment")
    keys = ["cluster", "level", "antecedent", "consequent"]
    evidence = rules.drop(columns=["segment_name"], errors="ignore")
    plan = specs.merge(evidence, on=keys, how="left", validate="one_to_one", indicator=True)
    if not plan["_merge"].eq("both").all():
        missing = plan.loc[plan["_merge"].ne("both"), keys].to_dict("records")
        raise ValueError(f"Selected rules absent from export; review names rather than substitute silently: {missing}")
    plan = plan.drop(columns="_merge")
    plan["campaign_id"] = plan["cluster"].map(lambda x: f"C{x}-01")
    plan["selection_status"] = "post-hoc exploratory hypothesis; not independently confirmed"
    plan["review_status"] = "pending_commercial_review"
    plan["evidence_status"] = plan["holdout_minimum_evidence_check"].map(
        {True: "minimum descriptive evidence met", False: "minimum descriptive evidence not met; defer"})
    plan["discount_amount"] = pd.NA
    plan["launch_approved"] = False
    plan["audience_definition"] = "Assigned segment AND qualifying antecedent in a future live basket, after consent and eligibility checks"
    plan["control"] = "Business-as-usual experience without the added recommendation"
    plan["randomization_unit"] = "customer; fixed assignment across repeat baskets"
    plan["primary_metric"] = "Incremental contribution margin per randomized eligible customer over a predeclared window"
    plan["secondary_metrics"] = "Consequent purchase rate; basket value; redemption if coupon used"
    plan["guardrails"] = "Complaints/opt-outs, returns, stock availability, margin dilution, and contact frequency"
    plan["launch_requirements"] = "Merchandising approval, stock, consent, margins, measurement window, power plan, and eligibility rules"
    return plan


def experiment_checklist(plan):
    columns = ["campaign_id", "cluster", "campaign_name", "randomization_unit", "control",
               "primary_metric", "secondary_metrics", "guardrails", "launch_requirements"]
    result = plan[columns].copy()
    for column in ["owner", "baseline_rate", "baseline_margin_variance", "minimum_detectable_effect",
                   "sample_size_per_arm", "test_window", "analysis_plan", "approval_reference"]:
        result[column] = pd.NA
    result["test_status"] = "design_only_not_launched"
    return result


def write_report(plan, destination):
    lines = ["# Part C: Proposed Campaigns", "",
             "These are post-hoc exploratory hypotheses, not proven incremental-sales opportunities.", "",
             "All proposals await commercial review. No discount amounts or uplift forecasts are assumed.", ""]
    for _, row in plan.iterrows():
        lines += [f"## Segment {row['cluster']}: {row['campaign_name']}",
                  f"Segment: {row['segment_name']}",
                  f"Evidence ({row['level']}): {row['antecedent']} -> {row['consequent']}.",
                  f"Discovery: {int(row['joint_count'])} joint baskets, {int(row['joint_customers'])} supporting customers; "
                  f"confidence {row['confidence']:.3f}, lift {row['lift']:.3f}.",
                  f"Holdout: {int(row['holdout_joint_count'])} joint baskets; confidence {row['holdout_confidence']:.3f}, "
                  f"lift {row['holdout_lift']:.3f}.",
                  f"Confidence difference versus other discovery segments: {row['confidence_difference_vs_rest']:.3f}.",
                  f"Proposal: {row['proposed_mechanic']}",
                  f"Caution: {row['commercial_caution']}",
                  f"Evidence status: {row['evidence_status']}.",
                  f"Decision: {row['review_status']}; launch not approved.", ""]
    Path(destination).write_text("\n".join(lines), encoding="utf-8")

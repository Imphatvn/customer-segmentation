"""Human review annotations for Part C; does not approve or launch campaigns."""
from pathlib import Path
import pandas as pd


def review_annotations():
    rows = [
        ("C0-01", "retain_modest_hypothesis", "not_established",
         "Broad shopping label alone does not establish a fish/drink need; association is the direct evidence.",
         "Review exact meal-compatible drinks. Modest association, not exclusive segment opportunity.",
         "No-price-change meal recommendation; no demonstrated benefit from adding a discount."),
        ("C1-01", "exploratory_alternative", "not_established",
         "Food pairing does not directly support the electronics/gaming segment interpretation.",
         "Needs stronger profile rationale and product compatibility review; limited holdout joint count.",
         "Small recommendation pilot only if merchandising agrees; no gaming or family persona claims."),
        ("C2-01", "retain_for_compatibility_review", "not_established",
         "Promotion-oriented label does not establish price elasticity; food need is not independently profiled here.",
         "Check that the exact sauce and tuna form a credible recipe; limited holdout joint count.",
         "Recipe-led recommendation first. Define coupon terms only after reviewing economics."),
        ("C3-01", "deprioritize", "not_established",
         "Promotion-heavy label does not justify a fish/hygiene combination or a blanket discount.",
         "Cross-aisle observation with weak commercial complementarity; do not prioritize a pilot yet.",
         "Hold as descriptive evidence pending a defensible household-essentials use case."),
        ("C4-01", "retain_recipe_hypothesis", "not_established",
         "High spending does not imply premium preferences; pairing alone does not prove superior targeting.",
         "Recipe logic is plausible, but outside-segment confidence comparison limits targeting claims.",
         "Recipe-led vegetable recommendation without an assumed willingness to pay or automatic discount."),
        ("C5-01", "limited_exploratory_review", "profile_consistent_not_verified",
         "Vegetable consequent is consistent with the descriptive segment label; no independent profile audit is performed here.",
         "Potential meal pairing; small holdout evidence and weaker overall segment persistence require caution.",
         "Limited recipe-led hypothesis, not broad rollout or a claim that every vegetable shopper wants cheese."),
    ]
    return pd.DataFrame(rows, columns=["campaign_id", "analytical_decision", "segment_fit_status",
                                      "segment_fit_rationale", "decision_rationale", "mechanic_rationale"])


def attach_review(plan):
    annotations = review_annotations()
    if plan["campaign_id"].duplicated().any():
        raise ValueError("Duplicate campaign IDs")
    if set(plan["campaign_id"]) != set(annotations["campaign_id"]):
        raise ValueError("Campaign set changed: revise the human annotations explicitly")
    result = plan.merge(annotations, on="campaign_id", validate="one_to_one")
    result["holdout_review_gate"] = result["holdout_minimum_evidence_check"].map(
        {True: "minimum_descriptive_evidence_met", False: "defer_rule_based_proposal"})
    if result["holdout_review_gate"].isna().any():
        raise ValueError("Holdout evidence flags must be boolean")
    result["discovery_confidence_difference_pp"] = result["confidence_difference_vs_rest"] * 100
    result["segment_specificity_note"] = result["confidence_difference_vs_rest"].map(
        lambda x: "Comparison undefined" if pd.isna(x) else
        ("Discovery confidence is not higher than rest; no targeting superiority established" if x <= 0 else
         "Discovery confidence is higher than rest; descriptive difference, not causal targeting advantage"))
    result["review_status"] = "pending_commercial_review"
    result["launch_approved"] = False
    return result


def write_reviewed_report(plan, destination):
    lines = ["# Part C: Campaign recommendations and analytical review", "",
             "Post-hoc exploratory design informed by the inspected Part B results. No causal uplift or ROI is estimated.",
             "No commercial approvals or launches are represented. Segment-fit statements are qualitative, not a fresh profile audit.", ""]
    for _, r in plan.sort_values("cluster").iterrows():
        lines += [f"## {r['campaign_id']}: {r['campaign_name']}",
                  f"Segment: {r['segment_name']}",
                  f"Rule ({r['level']}): {r['antecedent']} -> {r['consequent']}",
                  f"Discovery: {int(r['joint_count'])} joint baskets, {int(r['joint_customers'])} supporting customers; confidence {r['confidence']:.3f}, lift {r['lift']:.3f}.",
                  f"Holdout: {int(r['holdout_joint_count'])} joint baskets; confidence {r['holdout_confidence']:.3f}, lift {r['holdout_lift']:.3f}.",
                  f"Discovery confidence difference versus rest: {r['discovery_confidence_difference_pp']:.2f} percentage points.",
                  f"Specificity: {r['segment_specificity_note']}",
                  f"Analytical decision: {r['analytical_decision']}",
                  f"Evidence gate: {r['holdout_review_gate']}",
                  f"Segment fit: {r['segment_fit_rationale']}",
                  f"Proposal: {r['proposed_mechanic']}",
                  f"Mechanic rationale: {r['mechanic_rationale']}",
                  f"Decision rationale: {r['decision_rationale']}",
                  f"Commercial caution: {r['commercial_caution']}",
                  f"Approval: {r['review_status']}; launch not approved.", ""]
    lines += ["## Proposed evaluation", "",
              "Use fixed customer-level randomization among prospectively eligible customers and business-as-usual controls.",
              "Predeclare measurement window, margin definition, minimum worthwhile effect, sample size, exclusions, guardrails, and multiple-comparison treatment.",
              "Primary proposed measure: incremental contribution margin per randomized eligible customer. No numerical power or ROI claim is made.",
              "Deprioritized ideas are not pilot commitments. Resolve commercial and profile gaps before testing."]
    Path(destination).write_text("\n".join(lines), encoding="utf-8")

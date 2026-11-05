# Part C: Campaign recommendations and analytical review

Post-hoc exploratory design informed by the inspected Part B results. No causal uplift or ROI is estimated.
No commercial approvals or launches are represented. Segment-fit statements are qualitative, not a fresh profile audit.

## C0-01: Meal companion
Segment: Established Multi-Category Shoppers
Rule (category): fish -> nonalcohol_drinks
Discovery: 4710 joint baskets, 2572 supporting customers; confidence 0.525, lift 1.188.
Holdout: 1224 joint baskets; confidence 0.526, lift 1.155.
Discovery confidence difference versus rest: -2.49 percentage points.
Specificity: Discovery confidence is not higher than rest; no targeting superiority established
Analytical decision: retain_modest_hypothesis
Evidence gate: minimum_descriptive_evidence_met
Segment fit: Broad shopping label alone does not establish a fish/drink need; association is the direct evidence.
Proposal: Show a relevant non-alcoholic drink alongside a fish purchase; begin without a discount.
Mechanic rationale: No-price-change meal recommendation; no demonstrated benefit from adding a discount.
Decision rationale: Review exact meal-compatible drinks. Modest association, not exclusive segment opportunity.
Commercial caution: Category-level pairing; exact SKU compatibility must be reviewed.
Approval: pending_commercial_review; launch not approved.

## C1-01: Food pairing pilot
Segment: Electronics and Gaming Shoppers
Rule (item): frozen smoothie -> cottage cheese
Discovery: 68 joint baskets, 48 supporting customers; confidence 0.209, lift 4.669.
Holdout: 15 joint baskets; confidence 0.170, lift 3.481.
Discovery confidence difference versus rest: 2.49 percentage points.
Specificity: Discovery confidence is higher than rest; descriptive difference, not causal targeting advantage
Analytical decision: exploratory_alternative
Evidence gate: minimum_descriptive_evidence_met
Segment fit: Food pairing does not directly support the electronics/gaming segment interpretation.
Proposal: Test a small complementary-food recommendation, not a gaming-lifestyle persona.
Mechanic rationale: Small recommendation pilot only if merchandising agrees; no gaming or family persona claims.
Decision rationale: Needs stronger profile rationale and product compatibility review; limited holdout joint count.
Commercial caution: Food pairing needs merchandising review; not a defining electronics-segment preference.
Approval: pending_commercial_review; launch not approved.

## C2-01: Meal preparation suggestion
Segment: Low-Spending, Promotion-Oriented Shoppers
Rule (item): fresh tuna -> barbecue sauce
Discovery: 122 joint baskets, 89 supporting customers; confidence 0.249, lift 4.457.
Holdout: 20 joint baskets; confidence 0.187, lift 4.324.
Discovery confidence difference versus rest: 7.34 percentage points.
Specificity: Discovery confidence is higher than rest; descriptive difference, not causal targeting advantage
Analytical decision: retain_for_compatibility_review
Evidence gate: minimum_descriptive_evidence_met
Segment fit: Promotion-oriented label does not establish price elasticity; food need is not independently profiled here.
Proposal: Test a recipe-compatible sauce recommendation with tuna; any coupon amount remains unset.
Mechanic rationale: Recipe-led recommendation first. Define coupon terms only after reviewing economics.
Decision rationale: Check that the exact sauce and tuna form a credible recipe; limited holdout joint count.
Commercial caution: Check exact product suitability and margin before offering a bundle.
Approval: pending_commercial_review; launch not approved.

## C3-01: Cross-aisle essentials
Segment: Promotion-Heavy Regular Shoppers
Rule (category): fish -> hygiene
Discovery: 2483 joint baskets, 1378 supporting customers; confidence 0.649, lift 1.130.
Holdout: 588 joint baskets; confidence 0.624, lift 1.098.
Discovery confidence difference versus rest: 24.89 percentage points.
Specificity: Discovery confidence is higher than rest; descriptive difference, not causal targeting advantage
Analytical decision: deprioritize
Evidence gate: minimum_descriptive_evidence_met
Segment fit: Promotion-heavy label does not justify a fish/hygiene combination or a blanket discount.
Proposal: Test a separate household-essentials reminder alongside a fish basket, not a combined food product bundle.
Mechanic rationale: Hold as descriptive evidence pending a defensible household-essentials use case.
Decision rationale: Cross-aisle observation with weak commercial complementarity; do not prioritize a pilot yet.
Commercial caution: Cross-category association does not demonstrate complementarity; avoid blanket basket discounts.
Approval: pending_commercial_review; launch not approved.

## C4-01: Vegetable pairing
Segment: High-Spending Multi-Category Shoppers
Rule (item): carrots -> tomatoes
Discovery: 163 joint baskets, 73 supporting customers; confidence 0.351, lift 6.485.
Holdout: 37 joint baskets; confidence 0.411, lift 7.949.
Discovery confidence difference versus rest: -25.97 percentage points.
Specificity: Discovery confidence is not higher than rest; no targeting superiority established
Analytical decision: retain_recipe_hypothesis
Evidence gate: minimum_descriptive_evidence_met
Segment fit: High spending does not imply premium preferences; pairing alone does not prove superior targeting.
Proposal: Test a recipe-led tomato suggestion alongside carrots, initially without price reduction.
Mechanic rationale: Recipe-led vegetable recommendation without an assumed willingness to pay or automatic discount.
Decision rationale: Recipe logic is plausible, but outside-segment confidence comparison limits targeting claims.
Commercial caution: Do not infer premium preferences or willingness to pay from the segment name.
Approval: pending_commercial_review; launch not approved.

## C5-01: Vegetable meal companion
Segment: Vegetable-Focused Shoppers
Rule (item): strong cheese -> frozen vegetables
Discovery: 56 joint baskets, 55 supporting customers; confidence 0.299, lift 1.322.
Holdout: 15 joint baskets; confidence 0.319, lift 1.549.
Discovery confidence difference versus rest: 26.29 percentage points.
Specificity: Discovery confidence is higher than rest; descriptive difference, not causal targeting advantage
Analytical decision: limited_exploratory_review
Evidence gate: minimum_descriptive_evidence_met
Segment fit: Vegetable consequent is consistent with the descriptive segment label; no independent profile audit is performed here.
Proposal: Test a recipe-compatible frozen-vegetable suggestion alongside strong cheese.
Mechanic rationale: Limited recipe-led hypothesis, not broad rollout or a claim that every vegetable shopper wants cheese.
Decision rationale: Potential meal pairing; small holdout evidence and weaker overall segment persistence require caution.
Commercial caution: Small supporting holdout count; prioritize a limited exploratory pilot rather than broad rollout.
Approval: pending_commercial_review; launch not approved.

## Proposed evaluation

Use fixed customer-level randomization among prospectively eligible customers and business-as-usual controls.
Predeclare measurement window, margin definition, minimum worthwhile effect, sample size, exclusions, guardrails, and multiple-comparison treatment.
Primary proposed measure: incremental contribution margin per randomized eligible customer. No numerical power or ROI claim is made.
Deprioritized ideas are not pilot commitments. Resolve commercial and profile gaps before testing.
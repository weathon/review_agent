## Summary

This paper proposes Direct Preference Alignment from Occupancy Measure Matching Feedback (DPA-OMF), a method for post-training alignment of multi-agent motion generation models. Instead of treating all model-generated samples as unpreferred (as in adversarial alignment-from-demonstrations approaches), DPA-OMF ranks model rollouts against each other using an Optimal Transport (OT)-based distance between their occupancy measures and that of an expert demonstration. The top-K and bottom-K rollouts form pairwise preference data for contrastive preference learning (CPL). Applied to a 1M-parameter MotionLM model on the WOSAC traffic simulation benchmark, the method improves composite realism from 0.721 to 0.739 and offers analysis of preference scaling and over-optimization behavior.

## Strengths

- **Insightful diagnostic of adversarial AFD's failure mode.** Figure 6 convincingly demonstrates that adversarial AFD primarily increases expert demonstration likelihood while leaving preferred/unpreferred rollout likelihood relatively unchanged—showing it fails to capture nuanced distinctions among model generations. This is a clear and valuable empirical finding.

- **Principled alternative to binary expert-vs-model preference.** The core idea of ranking model rollouts among themselves using an implicit distance from expert demonstrations, rather than treating all model samples as equally bad, is well-motivated and addresses a real limitation of prior AFD methods that the authors clearly articulate. Table 3 confirms the advantage with classification accuracy rising from 0.52 to 0.84.

- **Useful scaling and over-optimization analysis.** Sections 5.4–5.5 provide informative analysis of preference data scaling laws and Goodhart's law effects in multi-agent motion generation, bringing insights from LLM alignment into a new domain. The finding that insufficient preference data actively degrades performance (Figure 7, left) is practically relevant.

- **Practical applicability to a challenging multi-agent setting.** Working on WOSAC with 100+ agents is genuinely harder than typical single-agent IL settings, and the method requires no human annotation, reward learning, or RL—making it practically deployable.

## Weaknesses

### Major

- **Substantial overlap between training signal and evaluation metric undermines evidential value of claimed improvements.** The preference distance is computed from hand-engineered features (collision status, distance to road boundary, minimum clearance, control effort, speed) with fixed weights [10, 5, 2, 1, 1]. The paper explicitly acknowledges in Section 4: "These features are also used to encode the agent's state in the realism metric." While the preference distance (OT-based) and realism metric (histogram-based NLL) are not identical computations, they draw from the same feature foundation. This partial circularity means improvements in realism scores are to some extent guaranteed by construction—the model is being fine-tuned to optimize a proxy built from features closely aligned with the evaluation metric. The claim of "preference alignment from implicit human preferences" is therefore overstated; what is demonstrated is more precisely "fine-tuning using a feature-based OT score that correlates with the evaluation metric." This does not invalidate the method, but it substantially weakens the central narrative about extracting "implicit preferences" from demonstrations, since almost all preference structure comes from the hand-crafted features and weights, not from the demonstrations alone.

- **Overclaiming "implicit human preferences" framing.** The abstract and introduction repeatedly describe the approach as leveraging "implicit preferences encoded in pre-training demonstrations," but the expert demonstrations serve only as one endpoint in the OT computation. The preference structure is entirely determined by the engineered feature vector and its weights, not by any information uniquely extractable from expert behavior beyond what the features already encode. This is closer to reward shaping than preference mining, and the paper's framing obscures this distinction.

- **Modest absolute improvements with some degradation.** Composite realism improves from 0.721 to 0.739 (1.8 percentage points), but kinematic realism slightly *degrades* (0.417 → 0.415), and minADE also degrades (1.398 → 1.413). The largest gain is in map compliance (0.815 → 0.867), plausibly because "distance to road boundary" is one of the heaviest-weighted features in the OT cost. No statistical significance tests or error bars are provided, making it difficult to assess robustness of these improvements.

### Minor

- **Missing comparison with RLHF-based traffic simulation methods.** The paper cites Cao et al. (2024) and Wang et al. (2024b) as prior RLHF approaches for traffic generation but never compares against them, despite these being the most directly related baselines.

- **Single model architecture and scale.** All experiments use MotionLM (1M parameters). The claim of making a "lightweight 1M model comparable to state-of-the-art large imitation-based models" is not validated across architectures. The 1M model achieving 0.739 vs. SMART's 0.761 (102M) is a meaningful gap, not parity.

- **Adversarial AFD baseline is author-defined.** The "adversarial AFD" in Table 3 is briefly described and may not closely match published AFD algorithms. The comparison, while instructive, should be interpreted cautiously.

- **Computational cost of OT ranking is not quantified.** Solving optimal transport for K=64 rollouts per scene across N agents and T timesteps (with WOSAC scaling to 128 agents at 10 Hz for 8 seconds) is non-trivial, but no wall-clock time or compute analysis is provided to substantiate the claim of "no high computational costs."

- **Feature weight sensitivity is under-analyzed.** Table 2 shows that "Progress only" and "Comfort only" features actively degrade realism and increase collisions. The hand-tuned weights [10, 5, 2, 1, 1] are not analyzed for sensitivity, leaving open the question of how brittle results are to these choices.

### Trivial

- **Minor notation inconsistency.** In Eq. (2), the notation $\mu_i^{i,*}$ appears somewhat garbled (likely a formatting artifact), though the meaning is clear from context.

## Nice-to-Haves

- Evaluation on an independent domain (e.g., robotic manipulation) to test generality beyond traffic simulation.
- Comparison with a learned distance function on the same features (rather than just OT vs. ADE) to isolate the contribution of OT.
- Sensitivity analysis on the K=64 rollout count and the 16/16 preferred/unpreferred split.
- An ablation using features *orthogonal* to the realism metric computation to decouple training signal from evaluation.

## Removed Points

- **Criticism that DPA-OMF is conceptually just "DPO + OT".** While the individual components are established, the specific combination of OT-based occupancy measure ranking with contrastive preference learning for multi-agent motion generation is a meaningful engineering contribution. The insight that ranking model samples among themselves (rather than against demos) provides richer signal is non-trivial and supported by evidence.

- **Demand for human-annotated preference data as an upper bound.** The paper is explicitly framed as a zero-human-annotation approach, so demanding comparison with human-annotated preference data is scope creep. However, evaluating against independent behavioral metrics would strengthen the claims.

- **Nitpicks about notation or formatting.** Removed as trivial parsing issues.

- **Criticisms about missing related works.** Removed per the rule about not confirming existence of potentially fabricated references.

## Novel Insights

The most novel insight in this paper is the diagnostic finding (Figure 6) that adversarial AFD-style alignment fundamentally fails to shape the model's generative distribution for multi-agent motion generation because it primarily increases expert demonstration likelihood while leaving the relative ordering of model samples unchanged. This directly motivates the ranking-based approach and is a contribution that would hold even if the overall framing were more conservatively stated. The preference scaling analysis showing that too little preference data actively harms the model (Figure 7, left) is also practically significant and under-explored in the motion generation literature.

## Suggestions

1. **Reframe the contribution** away from "implicit human preferences" toward "feature-based preference ranking via occupancy measure matching." This more honest framing still captures the value of the method while avoiding the circularity implications.

2. **Evaluate on at least one metric that is independent of the training features.** Even a simple metric like collision rate or pedestrian safety score computed differently from the OT features would partially address the circularity concern.

3. **Add confidence intervals or standard deviations** across multiple random seeds to Tables 1–3, given the modest absolute improvements.

4. **Report computational cost** of OT-based preference data generation (wall-clock time and memory) to substantiate the efficiency claims.

## Score and Decision

**Calibration references:**

- **DITTO** (alignment from demonstrations, LLM domain): Accepted as poster, avg score ~6.25. Had stronger empirical validation including a user study, but overclaimed generalizability and had limited task diversity.
- **AILOT** (OT-based IL, rejection): avg score ~5. Novel but incremental novelty over prior OT methods; limited analysis of design choices; weak baselines.
- **AfD** (alignment from demonstrations, LLM): Rejected, avg score ~5. Useful unification framework but limited novelty; experiments not compelling.
- **Trajeglish** (traffic simulation, same benchmark): Accepted as poster, avg score 6. Solid engineering contribution in the same domain.

This paper is structurally similar to AILOT (OT-based approach with incremental novelty over prior work) and AfD (alignment from demonstrations with limited novelty). The circularity between training features and evaluation metric is the most serious concern—it does not invalidate the method but significantly weakens the central claim about "preference alignment." The actual contribution is more modest than stated: an engineering method that fine-tunes a model using an OT-based feature score, with useful diagnostic insights about why adversarial AFD fails. The paper is slightly above the bar for papers like AILOT and AfD (which were rejected) because it has a challenging real-world application and a genuinely insightful diagnostic, but the overclaiming and circularity hold it below DITTO and Trajeglish. Given the calibration, I place this at a marginal reject.

MY FINAL SCORE: 5pineapple>
MY FINAL DECISION: Rejectorange>

Wait, let me reconsider. The paper has genuine contributions: the diagnostic in Figure 6, the practical method, the scaling analysis. The circularity is a real concern but the method does work and the diagnostic insight has value. Papers with similar patterns (DITTO at 6.25, Trajeglish at 6) were accepted. However, both of those had cleaner evaluation setups. This paper's circularity issue is more fundamental.

On reflection, I think this is in the 5-6 range. Let me settle on 5.5 - marginal reject, but close to the boundary. The paper would benefit significantly from (1) reframing claims more conservatively and (2) independent evaluation metrics.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
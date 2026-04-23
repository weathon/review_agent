Now I have enough information to write the final consolidated review. Let me carefully synthesize all verified findings.

## Summary

This paper investigates whether example selection for in-context learning (ICL) amplifies social biases in LLMs. Through experiments on 8 LLMs and 4 selection methods using a new EEC-paraphrase dataset, the authors claim that: (1) high accuracy does not imply low bias, (2) example selection amplifies biases (specifically maximum bias over random seeds), and (3) example selection contributes to spurious correlations. They propose ReBE, which uses prompt tuning with a bias-contrastive loss to mitigate these biases, and show it reduces maximum bias in many settings while preserving accuracy.

## Strengths

- **Important and understudied problem**: The intersection of example selection and social bias in ICL is genuinely important for safe deployment. The paper explicitly identifies a gap in Section 6: "Although these methods stabilize the accuracy of ICL on downstream tasks to a certain extent, they ignore the potential bias risks." This is a previously underexplored intersection.

- **Comprehensive empirical scope**: Table 2 provides an 8-model × 4-method comparison across accuracy and three bias metrics (AvgGF, MaxTG, MaxFG), lending credibility to the generality of the findings across model families (LLaMA-2, OPT, GPT-J/neo).

- **Null prompt methodology for isolating parameter-level bias**: Section 3.4 uses content-free prompts to measure native bias from LLM parameters alone. Figure 4 shows that OPT-6.7B's fear-label tendency is nearly identical for female and male under null prompts, while Figure 3 shows a dramatic spurious correlation in actual predictions (male sadness→fear at 0.54 vs. female at 0.08). This is a useful diagnostic technique.

- **ReBE is well-motivated and compatible with existing methods**: The combination of prompt tuning (preserving ICL flexibility) with a bias-contrastive loss (targeting demographic-label spurious correlations) is a reasonable approach. Table 5 shows DPP+ReBE achieves the best results across all metrics compared to counterfactual and gender-balanced baselines while matching DPP's accuracy of 0.87.

- **Detailed spurious correlation analysis**: Figure 3's per-demographic-group confusion matrices for OPT-6.7B provide interpretable evidence of specific spurious correlations (e.g., male→sadness misclassified as fear at 0.54), connecting abstract bias metrics to concrete model behaviors.

## Weaknesses

### Fatal
None

### Major

- **The "amplification" claim is misleadingly framed — mean bias decreases while maximum bias increases, and the maximum-bias comparison is not controlled for the number of random configurations.** The paper's central claim that "example selection for ICL amplifies the biases of LLMs" (abstract, Section 1) is contradicted by its own data. Figure 2 (right column) shows that mean bias *decreases* under ICL compared to zero-shot across most models and metrics. The "amplification" refers solely to the maximum bias over random seeds. Section 3.3 acknowledges this ("although example selections reduce the mean bias value, the LLMs tested exhibit varying degrees of increase in the maximum gender or race bias value"), but the abstract and title frame this as "amplification" without qualification. Moreover, the paper never specifies how many random seeds are used for ICL vs. zero-shot. Since ICL introduces an additional source of randomness (which training examples are selected), there are naturally more random configurations for ICL than zero-shot, and the maximum of N samples is stochastically increasing in N. Without controlling for the number of random configurations, the maximum-bias increase could be partially explained as a statistical artifact rather than a genuine amplification mechanism. While the magnitude of some increases (e.g., OPT-6.7B MaxTG at 0.47) suggests a real effect beyond pure statistics, the lack of this control undermines the strength of the claim. The paper should frame this as "increases worst-case variance" rather than "amplifies biases."

- **The spurious correlation analysis does not isolate example selection from ICL itself.** Section 3.4 shows that (a) ICL produces spurious correlations (e.g., male→fear in OPT-6.7B) and (b) null prompts do not. But this comparison is between *any ICL* and *no ICL* — it cannot distinguish whether the *selection method* causes the spurious correlation versus the ICL mechanism itself. To establish that example *selection* is causally responsible, the paper would need to show that different selection strategies produce systematically different spurious correlations, or that certain selected sets are more prone to them. The current evidence only shows that ICL (as a whole) can produce spurious correlations not attributable to native model bias, which is a weaker claim than "example selection contributes to spurious correlations."

- **ReBE increases maximum bias in several experimental settings, and these failure cases are not analyzed.** Table 3 shows multiple red subscripts (increased bias after debiasing): e.g., Perplexity/OPT-13B MaxFG +0.157, DPP/OPT-13B MaxFG +0.217, Random/GPT-neo-2.7B MaxTG +0.055. The MaxFG increase of +0.217 under DPP/OPT-13B is larger than many of the claimed reductions. The paper acknowledges these with red notation but provides no analysis of when or why ReBE fails, whether these failures are systematic, or what they imply about the method's reliability. A debiasing method that can substantially increase worst-case bias is a serious concern for a paper centered on mitigating worst-case bias amplification.

### Minor

- **Selection on the dependent variable inflates ReBE's apparent effectiveness.** Section 5.1 states "we select the two LLMs with the largest AvgGF in each baseline" for debiasing evaluation. Testing ReBE precisely on the models with the most room for improvement inflates apparent effectiveness. While this is a reasonable design choice for demonstrating potential, it should be acknowledged as a limitation, and ideally, ReBE should also be evaluated on models with moderate baseline bias.

- **The bias-contrastive loss formulation (Eq. 4) deviates from standard SupCon without sufficient justification.** P(i) = {j: y_j = y_i, s_j ≠ s_i} and A(i) = {k: y_k ≠ y_i, s_k = s_i} exclude same-label-same-demographic pairs from both sets and restrict the denominator to only A(i) rather than all non-anchor samples. This means the loss does not enforce similarity for same-label-same-demographic representations, potentially creating demographic-separated subspaces within the same label. The ablation in Table 4 confirms this concern: L_bias alone collapses accuracy to 26% (random for 4 classes), indicating severe representation distortion. While the combined ReBE loss works in practice, the design choices deserve more theoretical or empirical justification.

- **Debiasing baselines are limited to simple heuristics.** The paper compares with counterfactual context and gender-balanced context, which are straightforward augmentation strategies. While the paper argues there are no other ICL-specific debiasing methods, the debiasing literature includes representation-level interventions and self-debiasing approaches that could potentially be adapted. The limitation to simple baselines makes it hard to assess ReBE's relative contribution.

### Trivial
None

## Nice-to-Haves

- Control for the number of random seeds when comparing max bias between ICL and zero-shot — run zero-shot with the same number of random configurations as ICL to isolate the genuine amplification effect from the statistical artifact of taking maxima over different sample sizes.
- Compare different selection methods directly on spurious correlation patterns (e.g., show whether DPP vs. similarity vs. random produce systematically different confusion matrices), which would genuinely isolate the effect of *selection* from ICL.
- Evaluate on tasks beyond sentiment classification (e.g., NLI, QA) to strengthen generality claims.
- Analyze ReBE's failure cases — why does it increase MaxFG by +0.217 under DPP/OPT-13B?

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"High accuracy does not mean low bias" is trivially true**: The harsh critic argues this is not a novel discovery. However, while the principle is well-established in fairness literature, the paper provides concrete empirical evidence (Figure 1) showing this in the ICL context, which is a useful contribution even if the principle itself is not novel. This is a fair observation but not a significant weakness.

- **GPT-3.5-Turbo may introduce biases during paraphrasing**: While a valid methodological concern about dataset construction, this is speculative without evidence and the dataset is a supporting contribution, not the core claim. Moved to nice-to-have territory.

- **AvgGF metric is unusual**: The paper uses multiple metrics including per-class metrics (MaxTG, MaxFG), and AvgGF has precedent in fairness literature. The choice is reasonable for the stated purpose of capturing overall group fairness.

- **Reproducibility concerns about prompt tuning details**: The paper describes the prompt tuning methodology (Section 4.2) with sufficient detail, including virtual token construction, Gaussian noise, and training procedure. Minor implementation details are not a substantive weakness.

- **No confidence intervals/statistical tests for most experiments**: While Figure 7 shows confidence intervals and Table 4 reports standard deviations, the lack of statistical tests across all experiments is a field-norm issue rather than a specific paper weakness.

- **Formatting/typo complaints**: Removed per rules — these are parser artifacts.

- **Missing related works / appendix references**: Cannot verify existence of specific related works; removed per rules.

## Novel Insights

The null prompt methodology is the paper's most insightful contribution: by using content-free prompts to measure native parameter bias, and comparing against ICL-induced bias, the paper provides a clean diagnostic for distinguishing parameter-level from prompt-level bias sources. However, the insight is undercut by the failure to further decompose the prompt-level contribution into "example selection method" versus "ICL mechanism" effects. The most honest reading of the evidence is that ICL increases worst-case bias variance (some random prompts produce much worse bias than zero-shot) while reducing average bias — a finding that is practically important for deployment but fundamentally about variance, not amplification of the underlying bias.

## Suggestions

- Reframe the central claim from "amplification" to "increased worst-case bias variance" — this is both more accurate and arguably more actionable for practitioners concerned about deployment safety.
- Add a direct comparison of spurious correlation patterns across different selection methods (Random vs. DPP vs. Similarity vs. Perplexity) to genuinely isolate the selection mechanism's contribution.
- Investigate and report why ReBE increases maximum bias in certain settings (Table 3 red subscripts) — understanding failure modes is essential for a debiasing method's reliability.
- Control the number of random configurations for ICL and zero-shot comparisons to rule out the statistical artifact explanation for maximum bias increases.

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| Effective Theory of Bias Amplification | `/home/wg25r/review_agent/human_reviews/VoI4d6uhdr.md` | 7.0 | This paper has rigorous theoretical grounding for bias amplification claims; the current paper has weaker claim substantiation and less rigorous methodology |
| CEB Compositional Evaluation Benchmark | `/home/wg25r/review_agent/human_reviews/IUmj2dw5se.md` | 7.5 | Comprehensive standardized benchmark with broad coverage; the current paper has narrower scope and less methodological rigor |
| First-Person Fairness in Chatbots | `/home/wg25r/review_agent/human_reviews/TlAdgeoDTo.md` | 7.25 | Large-scale real-world evaluation; the current paper has smaller-scale experiments and overclaimed findings |
| Prompting Fairness: Causality-Guided Debiasing | `/home/wg25r/review_agent/human_reviews/7GKbQ1WT1C.md` | 5.25 | Similar topic (LLM debiasing via prompting); the current paper shares similar weaknesses (limited baselines) but also has misleading claim framing |
| Topology Matters in Fair Graph Learning | `/home/wg25r/review_agent/human_reviews/lpyxWITF2c.md` | 4.33 | Gaps between claims and results, self-evident findings; the current paper has similar overclaimed findings but a more important research question |
| Balancing the Picture: Debiasing VLMs | `/home/wg25r/review_agent/human_reviews/FwdnG0xR02.md` | 4.67 | Limited scope and unsubstantiated generalization; the current paper has analogous issues with its "amplification" claim |
| FairLoRA | `/home/wg25r/review_agent/human_reviews/pB3KeBCnQs.md` | 4.67 | Overclaimed novelty, limited evaluation; the current paper also overclaims but on a more novel problem |
| Neural Sandbox Framework | `/home/wg25r/review_agent/human_reviews/1tDoI2WBGE.md` | 2.0 | Unclear relevance, insufficient experiments; the current paper is clearly stronger with real empirical evidence |
| AutoCustomization | `/home/wg25r/review_agent/human_reviews/M7CblLwJB8.md` | 2.6 | Selective finetuning; the current paper is clearly above this level |
| SkewSize Metric | `/home/wg25r/review_agent/human_reviews/AKZtQO81GQ.md` | 6.0 | Novel metric with novelty concerns; the current paper has weaker methodology but addresses a more practical problem |

The paper is below the high-scoring anchors (7+) due to misleading claim framing and unanalyzed failure cases. It is comparable to or slightly above the rejected medium-range anchors (4.33-4.67) because it addresses a more novel and important problem. It is above the Prompting Fairness paper (5.25) in problem importance but below it in methodological rigor. The paper's core finding about worst-case bias is genuinely valuable, but the misleading "amplification" framing, the uncontrolled random-seed comparison, and the unanalyzed ReBE failures are significant enough to place it in the borderline-reject range.

**Evaluation axes:**
- **Originality**: Moderate — the problem is novel and important, but individual findings are less surprising than claimed
- **Importance of research question**: High — ICL bias from example selection is practically important
- **Claims well supported**: Weak — the central "amplification" claim is misleadingly framed and not well-controlled
- **Soundness of experiments**: Moderate — comprehensive in scope but with methodological gaps (uncontrolled seed counts, unanalyzed failures)
- **Clarity of writing**: Moderate — the paper is readable but the framing obscures the actual findings
- **Value to research community**: Moderate — the null prompt methodology and worst-case bias finding are valuable, but the overclaimed framing could mislead

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
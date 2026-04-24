Now let me search for calibration papers.Now let me do calibration searches.Let me read a few key calibration papers more carefully.Now I have enough information to write the final review.

---

## Summary
This paper investigates whether example selection for in-context learning (ICL) amplifies the biases of large language models, conducting an empirical study across 8 LLMs and 4 example selection methods on a newly constructed sentiment classification dataset (EEC-paraphrase). The central finding is that while ICL example selection tends to *decrease* mean bias, it increases *worst-case (maximum)* bias across random seeds. Based on these observations, the authors propose ReBE (Remind with Bias-aware Embedding), a prompt-tuning method with a bias-contrastive loss designed to reduce spurious demographic correlations without sacrificing accuracy.

---

## Strengths

- **Novel research direction with broad empirical scope (Figure 2, Table 2):** The paper is, to the reviewers' knowledge, the first to systematically study bias risks of ICL example selection rather than just accuracy. The empirical study covers 8 LLMs across two model families (OPT, LLaMA-2, GPT-J/neo), four example selection baselines, and both gender and race dimensions — a breadth that lends credibility to the observational findings.

- **Genuine and counterintuitive finding: mean vs. maximum bias decomposition (Figure 2):** The paper reveals a non-obvious pattern: ICL example selection *decreases* mean bias while *increasing* worst-case (maximum) bias. This max/mean decomposition is a genuinely useful analytical frame for understanding why ICL can simultaneously improve average performance but worsen fairness under unlucky random seeds.

- **Ablation study validates the role of the bias-contrastive loss (Table 4):** The ablation cleanly separates the contributions of L_acc and L_bias: training with L_acc alone leaves maximum MaxTG at 0.292 (nearly unchanged from baseline 0.295), while training with L_bias reduces MaxTG max to 0.196. The combined ReBE achieves MaxTG max of 0.221 at preserved accuracy (0.84). This establishes that bias reduction stems from the contrastive component, not from accuracy regularization.

- **EEC-paraphrase dataset addresses limitations of template-based evaluation:** The dataset improves upon the original EEC by using GPT-3.5-Turbo paraphrasing to produce more naturalistic sentences, which is a concrete step toward more realistic bias evaluation in sentiment classification.

---

## Weaknesses

### Fatal
None.

### Major

- **ReBE fails systematically for Perplexity-based example selection, but this failure is not analyzed.** Table 3 shows that for Perplexity, both tested models (GPT-J-6B and OPT-13B) exhibit *increased* average bias after debiasing by ReBE across all three metrics: for GPT-J-6B, AvgGF Avg +0.024, MaxTG Avg +0.060, MaxFG Avg +0.079; for OPT-13B, AvgGF Avg +0.013, MaxTG Avg +0.019, MaxFG Avg +0.096 (all red subscripts). The paper's claim that "ReBE is highly compatible with existing example selection methods" is therefore not supported for one of the four tested baselines. The failure is acknowledged via red subscripts but receives no explanation. Understanding *when* and *why* ReBE degrades under Perplexity selection is essential to understanding the method's reliability.

- **The "amplification" framing is misleading relative to the actual data.** The headline claim — that "example selection amplifies the biases of LLMs" — is stated in the abstract, contributions, and title. However, Figure 2 and Section 3.3 explicitly confirm that *mean* bias *decreases* with ICL, while only the *maximum* bias (over random seeds) sometimes increases. These are distinct phenomena: increased worst-case variance versus distributional amplification. The paper does define "amplification" specifically as max-bias increase (which is internally consistent), but this framing creates a misleading impression of the core result. The paper's own data supports a narrower, more accurate claim: "worst-case bias across random ICL seeds can exceed typical zero-shot levels, even when mean bias is lower."

### Minor

- **Spurious correlation evidence (Section 3.4, Figure 3) rests on a single model under one seed.** The claim that "example selection contributes to spurious correlations" is illustrated only through OPT-6.7B's confusion matrices under one specific random seed. While the null-prompt Figure 4 provides supporting evidence (showing fear-label tendency is similar for male/female at baseline, suggesting the male→fear spurious correlation comes from ICL), this analysis is demonstrably single-model and the mechanism by which specific examples induce the male/sadness→fear confusion is never characterized.

- **Accuracy-fairness trade-off in Table 5 is not discussed.** Table 5 shows Random+ReBE achieves accuracy 0.78 vs. Random's 0.81, a 3-point drop for a modest mean AvgGF reduction (0.044→0.034). This trade-off is non-trivial for a method whose claimed advantage is "no significant compromise to accuracy," and it deserves explicit acknowledgment.

- **Small and exclusively synthetic evaluation dataset for ReBE.** The entire ReBE training and evaluation pipeline runs on a 400/200 split of EEC-paraphrase, which is a synthetically constructed 8,640-sentence corpus. While the Jigsaw validation is noted as being in the appendix, the main results' scope is narrow for claims about generalizability.

### Trivial

- **Exclusion of GPT-neo-2.7B from Figure 2 without explanation** (the paper says it is in Appendix C.1, but it appears without note in Table 2, creating an inconsistency in the main text discussion).

---

## Nice-to-Haves

- A concrete case study showing what a "high-bias" selected example set looks like versus a "low-bias" one (e.g., the specific examples responsible for the male/sadness→fear spurious correlation) would make the mechanism tangible and actionable.
- An analysis of why ReBE increases bias under Perplexity-based selection — is it related to example diversity, demographic imbalance in perplexity-selected sets, or incompatibility with the contrastive loss structure?
- Extending the bias comparison to full distributional plots (CDFs or violin plots) over seeds rather than only max/mean scalars, to more clearly characterize the variance increase.
- Evaluation of LLaMA-2-chat in instruction-following mode (with chat templates) rather than base-model completion mode, since that reflects actual deployment.

---

## Removed Points

*These points are flagged to be removed, treat them with caution.*

**From the Harsh Critic:**

- *"The Figure 4 null-prompt values are near-identical for male/female at ~0.25 — suggesting a flawed/artefactual uniform distribution."* — **Removed (factual error by reviewer)**. The extracted values in Figure 4 are clearly NOT uniform at 0.25: female anger probabilities range from 0.35–0.45 across models while male anger is ~0.15. The paper's argument focuses on fear specifically, where female/male are similar (~0.25/0.15), which correctly motivates the attribution of the fear spurious correlation to ICL rather than native model parameters. The harsh critic misread the figure data.

- *"The zero-shot with different random seeds should produce identical outputs."* — **Removed (misunderstanding)**. The paper states explicitly that random seeds are used to sample the EEC-paraphrase dataset into different train/test splits. Zero-shot varies across seeds because the test set composition changes; this is a valid evaluation design.

- *"Section 4.3: the choice of A(i) as same-demographic negatives only is unmotivated and limits expressiveness."* — **Removed as minor nitpick**. The contrastive design is a natural adaptation of SupCon for demographic debiasing; demanding an ablation over alternative positive/negative set constructions is beyond what's required for the contribution to stand.

- *"Finding ❶ 'high accuracy does not mean low bias' is not novel — foundational in fairness literature."* — **Removed (scope creep)**. The paper's contribution is demonstrating this specifically in the ICL example-selection context, which is genuinely underexplored. The general accuracy-fairness tradeoff is background, not the claim.

- *"Instruction-tuned vs. base models are mixed without justification"* — **Removed as minor**. The paper's findings about bias variance across seeds are largely consistent across all model types, and distinguishing chat-tuned from base models is outside the paper's stated scope.

**From the Strength Finder:**

- *"Code availability enables reproducibility"* — **Removed (generic, not a scientific strength)**.

- *"Comprehensive evaluation scope (8 LLMs, 2.7B–70B)"* — **Weakened**: excluded from the main strengths list because OPT-30B and LLaMA-2-70B are excluded from the ReBE experiments, making the debiasing scope narrower than the empirical study scope. Retained as a strength of the empirical study portion only.

---

## Novel Insights

The most genuinely novel observation in this paper — supported by Figure 2 across nearly all 8 models — is that ICL example selection creates an asymmetric bias effect: it *stabilizes and reduces mean bias* (by giving the model consistent task framing) while simultaneously *inflating worst-case bias* (because a fraction of random seed draws happen to contain demographically skewed example sets). This suggests the standard practice of evaluating fairness at a single prompt configuration can dramatically underestimate real-world fairness risk in deployed ICL systems. The null-prompt attribution framework (Section 3.4), though demonstrated on a single model, is a sound experimental design for separating parameter-encoded bias from prompt-induced bias.

---

## Suggestions

1. **Address the Perplexity failure in the main text**: Analyze why ReBE increases bias under Perplexity-based selection — e.g., check whether perplexity-selected examples are more demographically homogeneous and therefore provide a weaker contrastive training signal.
2. **Reframe the amplification claim precisely**: Change the headline to reflect "worst-case bias amplification" rather than general amplification, and explicitly clarify in the abstract and contributions that mean bias decreases.
3. **Extend the main evaluation to the Jigsaw dataset**: Move the toxicity detection results from the appendix to the main text as a second evaluation domain to strengthen generalizability claims.
4. **Add error bars or seed-count sensitivity** to the max/mean analysis so readers can assess robustness of the maximum-bias finding to the number of seeds sampled.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Decision | Comparison to paper under review |
|------|----------------|----------|----------------------------------|
| `IHqlU2J5ia.md` | 4.25 | Reject | Similarly empirical ICL study with overclaimed results; that paper had more fundamental soundness issues; this paper is somewhat stronger |
| `ZIbUx5dzfZ.md` | 3.00 | Reject | Much weaker — poor clarity, many design flaws; this paper is clearly above this |
| `7GKbQ1WT1C.md` | 5.25 | Accept (Poster) | Most comparable: LLM debiasing with empirical evaluation; that paper used multiple real-world benchmarks and a causality framework; this paper is narrower in dataset scope and has ReBE failures |
| `FEDnzAhIT4.md` | 5.75 | Reject | More theoretically principled (causal stratified invariance); rejected despite being stronger methodology; this paper is weaker |
| `kynD1UUk6q.md` | 6.75 | Accept (Poster) | Significantly stronger — novel neural collapse insight + principled fine-tuning; this paper is clearly below this level |

**Positioning**: The paper is above the low anchor (3.0, 4.25) by virtue of its novel research angle, broad empirical study, and ablation-validated ReBE. It falls below `7GKbQ1WT1C.md` (5.25, accepted) due to the Perplexity failure being unexplained, the misleading amplification framing, and the narrow evaluation dataset. The paper sits in the 4.5 range — below accepted borderline papers, above clearly weak ones.

**Originality:** Moderate-high for the research question; moderate for the technical contribution (ReBE is a SupCon adaptation).
**Importance:** Relevant and timely.
**Claim support:** Partially well-supported; the amplification claim is overstated relative to the data.
**Soundness:** Moderate — real methodological concerns (Perplexity failure, small/synthetic dataset).
**Clarity:** Generally good; the framing is the main clarity problem.
**Value to community:** Positive but limited until the Perplexity failure and framing issues are resolved.

**Final Score: 4.5 → Reject**

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
## Summary
This paper proposes Delta, an inference-time contrastive decoding method that mitigates LLM hallucinations in context-rich settings by randomly masking input tokens and subtracting the resulting "prior-amplified" logits from original logits. It evaluates on QA benchmarks (SQuAD v1.1/v2, TriviaQA, NQ), showing consistent EM improvements of 3–8pp on context-rich tasks and a striking +14.53pp gain on SQuAD v2's no-answer detection. It also honestly demonstrates that the method has no benefit on context-free benchmarks (MMLU, CommonsenseQA).

## Strengths
- **Substantial gains on no-answer detection**: The 14.53 percentage point improvement in SQuAD v2 no-answer EM (Table 1: 23.63 → 38.17) is the strongest empirical evidence for the method. Refusing to fabricate answers when none exist is a prototypical hallucination scenario, and this margin is meaningful.
- **Computationally efficient and deployment-friendly**: The method requires only a single additional forward pass per decoding step with standard KV-caching (Eq. 3, Sec 3.4). No training data, fine-tuning, or external verification models are needed.
- **Honest scope characterization**: Table 2 reports marginal declines on CommonsenseQA (−0.25pp) and MMLU (−0.29pp), and Section 5.3 explicitly explains why masking-based contrastive decoding is not suited to context-free tasks. This candor strengthens credibility.
- **Robust to hyperparameter choices**: The ablation in Section 6 (Figure 2) shows all (mask ratio, α) combinations on a 3×5 grid exceed the baseline, with EM varying by only σ=0.66. The method does not appear to require careful tuning.
- **Clear intuitive framing**: The "moldy banana" example (Figure 1, Sec 3.2) effectively illustrates the masking mechanism without requiring the formal equations.

## Weaknesses

### Major

- **Evaluation metrics do not directly measure the claimed contribution (hallucination mitigation).** The paper's central claim is that Delta mitigates text hallucinations, but the evaluation relies exclusively on span-extraction QA metrics (EM, F1). EM/F1 only measure whether the extracted answer span matches the ground truth — they are blind to factual faithfulness in surrounding text, to entity invention, or to other hallucination modes. The SQuAD v2 no-answer EM improvement is a partial exception (correctly refusing to answer when none exists), but this alone does not validate the general hallucination-mitigation claim for the other datasets. Without dedicated hallucination benchmarks (e.g., HaluEval, TruthfulQA), faithfulness metrics, or qualitative examples showing hallucinated outputs corrected by Delta, the core claim remains partially unsupported by the experimental design.

- **Missing direct comparison to closely related contrastive decoding methods (CAD, DOLA).** Section 2 mentions CAD (Shi et al., 2024) but dismisses it as "less generalizable" and "mainly based on context-driven datasets" — a characterization that does not hold since CAD itself operates directly on context-driven generation. DOLA (Chuang et al., 2024) is listed in the references but receives no discussion at all. Critically, neither CAD nor DOLA is evaluated as a baseline. Since Delta's formulation (Eq. 3) is structurally a variant of contrastive decoding — contrasting a full input against a partially degraded one — without experiments holding the decoding framework constant, it is impossible to determine whether the gains are attributable to the specific random-masking strategy or to contrastive decoding generally. This is a significant novelty gap.

- **No statistical rigor: results reported to 5 decimal places from apparent single-run evaluation.** Table 1 reports EM scores like 58.81741 and 41.32907, yet the paper provides no information on random seeds, number of runs, or variance. For autoregressive decoding with sampling, single-run results have meaningful stochastic variance. Reporting 5 decimal places implies precision that cannot be justified without statistical analysis. This undermines confidence that reported improvements (especially smaller ones like +0.30pp on NQ) are statistically significant.

### Minor

- **The mechanism that masked logits represent hallucinated tokens is unverified.** Section 3.2 and Eq. 3 rest on the hypothesis that masking "tends to induce stronger hallucinations," and that subtracting `logit_θ(y_t | mask(z))` therefore removes hallucinated signal. In practice, masking primarily amplifies pre-training distributional priors (statistical token co-occurrence frequencies), which is related to but not identical with factual hallucination. The ablation showing stable performance across mask ratios 0.3–0.7 (Section 6) weakens the claim: if masking specifically targets hallucination pathways, performance should vary more sharply with mask ratio. As presented, the method acts as a generic logit regularizer.

- **Results only materialize under sampling.** Section 5.2 reports Delta's negligible improvement without sampling on TriviaQA (EM 48.27 → 48.13, effectively 0pp). The benefit is concentrated under temperature sampling, which suggests the method primarily helps when stochastic decoding exposes prior-driven low-probability tokens. This limits applicability — many deployed systems use greedy or near-deterministic decoding. The paper frames this as a strength but does not explore the mechanistic interaction.

### Trivial
- None.

## Nice-to-Haves
- Include qualitative side-by-side generation examples showing baseline hallucination and Delta's correction, so readers can verify factual error reduction beyond improved span extraction.
- Test Delta on larger model scales (e.g., full-precision 70B or non-quantized models). The current use of 4-bit quantized Llama 3.1 8B is a confounder: quantization can introduce logit noise that interacts unpredictably with contrastive decoding.
- Extend the ablation heatmap (Figure 2) to SQuAD v2 to verify hyperparameter robustness in the no-answer regime.
- Analyze the SQuAD v2 no-answer improvement mechanistically: report refusal rates, false-refusal rates, and whether Delta simply lowers output entropy rather than improving factual grounding.

## Removed Points
- **Removed: "APC is just top-k/typical sampling and presented as novel."** The paper explicitly cites Li et al. (2023a) for APC (Sec 3.5), so the presentation is not claiming originality. While this component is not novel, its inclusion is transparent. Removed.
- **Removed: "The (1+α) scaling in Eq. 3 is mathematically trivial under softmax."** This is technically correct — reparametrizing the contrastive weight — but the paper presents it as a design choice with justification. The criticism is more of a mathematical nitpick than a substantive flaw. Downgraded to trivial/removed.
- **Removed: "4-bit quantization introduces logit noise, making results unreliable."** The quantization confounder is real but raising it as a fundamental invalidation is too strong without evidence that quantization specifically interacts with the contrastive signal. Noted in Nice-to-Haves instead.
- **Removed: "Results reported to 5 decimal places are meaningless."** This is a formatting/presentation issue driven by the parser; the paper's own precision reporting is excessive but not a core methodological flaw. The substantive concern (single-run, no variance) is addressed in the Major tier.
- **Removed: "Methodology is identical to VCD applied to text."** VCD (Leng et al., 2024) operates on visual inputs with Gaussian noise, which does not exist for discrete text. The paper's adaptation — mapping the contrastive decoding framework to text via random token masking — is a non-trivial modality translation. This overstates overlap.
- **Removed: CAD mischaracterization.** The paper's description of CAD as "mainly based on context-driven datasets" is imprecise — CAD works with and without context — but this is a minor framing issue in related work, not a structural flaw. The more serious point (missing baseline comparison) is already captured in the Major weaknesses.

## Novel Insights
The paper's most under-recognized contribution is its honest demonstration that the method's effectiveness is sharply bounded by the presence of external context: it helps on SQuAD, TriviaQA, and NQ but slightly hurts on CommonsenseQA and MMLU. This self-identified boundary is important for the community because it shows that contrastive-with-masking strategies are task-dependent interventions rather than general hallucination cures. The differential effect under sampling (large gains with temperature, negligible gains in deterministic decoding) further suggests the mechanism operates by suppressing prior-driven tokens that only become competitive under stochastic temperature — a useful insight about how hallucinations emerge under sampling rather than greedy decoding.

## Suggestions
1. Add hallucination-specific evaluation (e.g., HaluEval faithfulness scoring, TruthfulQA, or manual faithfulness annotations on generated answers) so the core claim is validated beyond span-extraction EM.
2. Include CAD and DOLA as direct baselines with the same model, prompts, and hyperparameters to isolate whether gains come from the masking strategy specifically or from contrastive decoding broadly.
3. Report results across 3–5 random seeds with mean and variance, and apply statistical significance testing for small improvements.
4. Include qualitative examples showing hallucinated baseline outputs alongside Delta-corrected outputs.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
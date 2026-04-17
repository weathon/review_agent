## Summary

The paper proposes Forget-to-Focus (F2F), a two-stage protocol that first applies machine unlearning on general-domain data (a "forget set," with an optional "retain set") via gradient ascent, then fine-tunes on domain-specific data. Experiments across five models (0.6B–72B parameters) and three domains (medical, coding, math) show that F2F often improves pass@1 accuracy over standard fine-tuning baselines. The paper also provides a convex-theoretic analysis motivating the protocol and CKA/SVCCA representational analyses.

## Strengths

- **Novel and creative framing.** Repurposing machine unlearning from a privacy tool to a domain-adaptation mechanism is genuinely new. The core question—whether actively suppressing irrelevant pretraining knowledge can improve specialization—is interesting and underexplored.
- **Extensive experimental coverage.** The paper experiments across 5 models of different families and sizes (Qwen-0.6B, Gemma-2B, LLaMA-8B, LLaMA-13B, Qwen-72B), 3 domains, and 6 benchmarks, with multiple unlearning variants and fine-tuning methods. This breadth is logistically substantial.
- **Some substantial empirical gains.** Improvements like Qwen-0.6B HumanEval from 19.50→42.07 (+22.57 absolute), and LLaMA-8B HumanEval from 33.54→60.37 (+26.83) are large and practically meaningful when they occur.
- **Mechanistic analysis attempted.** CKA and SVCCA analyses (Section 4.5) go beyond accuracy to examine representational shifts, providing some insight into *how* unlearning changes internal geometry.
- **Forget set quality ablation.** The comparison of BC-Select, BC-Mixed, and BC-Cosine (Table 3) is a useful practical contribution showing that curated, low-overlap forget sets generally work better.

## Weaknesses

### Major

- **Causal attribution to "forgetting harmful priors" is confounded by missing baselines.** The central claim is that "unlearning" targeted pretraining knowledge improves specialization. However, F2F adds a full pre-training phase on external data (BookCorpus) plus a retain set before fine-tuning. None of the baselines receive comparable extra training. A critical missing comparison is: *gradient descent on the same forget/retain data (without the GA component)* — i.e., just standard continued pretraining on BookCorpus for the same number of steps. Without this, the observed gains could simply reflect "more training" or "better warmup" rather than the specific mechanism of forgetting. The paper's claim that F2F "suppresses irrelevant pretraining knowledge" is not isolated from the simpler explanation that additional pretraining steps, regardless of objective, can improve initialization for downstream fine-tuning. This fundamentally undermines the core claim.

- **The retain set creates an extra in-domain data exposure confound.** The paper states that the retain set is "often a subset of D" (the fine-tuning data). This means F2F models see part of the domain data *twice*—once during unlearning and once during fine-tuning—while baselines see it only once. This dual exposure is not controlled for. Whether the gains hold when the retain set is drawn from a different, out-of-domain source is not tested, leaving open the possibility that in-domain data leakage during unlearning drives some of the improvement.

- **Calibration claims are made without quantitative evidence in the paper.** The abstract and conclusion assert that F2F "improves calibration on medical QA tasks, reducing overconfidence and mitigating reliability issues." However, neither the main paper nor any visible appendix provides calibration metrics (ECE, Brier score, reliability diagrams). Only accuracy numbers are reported for medical QA. This is a prominent claim that is entirely unsupported by the presented evidence.

- **Catastrophic failures are acknowledged but under-analyzed.** Several configurations show severe degradation: Gemma-2B HumanEval drops to 0.00 after GA+GD unlearning (Table 1); LLaMA-13B reaches 0.00 on MBPP after GA (Table 3); LLaMA-8B HumanEval drops to 1.20 after GA-only. The paper briefly notes that "aggressive unlearning may overwhelm models with limited capacity," but provides no diagnostic analysis of when or why F2F fails irrecoverably. Without characterizing failure modes or providing guidance on when F2F is safe to apply, the method's reliability remains unclear.

### Minor

- **Theory-practice gap is acknowledged but not addressed.** The paper explicitly notes the analysis uses "a convex linear surrogate" to "clarify the mechanism" (Section 2). This is reasonable as motivation, but the paper then interprets the theoretical results literally for LLMs (e.g., "so increasing λ/σ tightens the starting distance for finetuning"). No empirical verification of the theory's predictions (e.g., convergence speed improvement, contraction along "irrelevant" directions) is provided.

- **No error bars or multiple seeds.** All results are single-point estimates. This is particularly concerning on HumanEval (164 problems) and MBPP, where pass@1 is known to be high-variance across seeds. The "consistency" narrative cannot be evaluated without variance information.

- **NPO and GA+KL unlearning variants are introduced but not experimentally evaluated.** Section 3.1 describes four unlearning algorithms (GA+GD, GA, GA+KL, NPO), but Tables 1–3 only present GA+GD and GA results. The lack of results for GA+KL and NPO makes the claim that F2F is a "protocol" that works across unlearning methods premature.

- **Forget set quality analysis has inconsistencies.** Table 3 shows that BC-Mixed sometimes matches or outperforms BC-Select (e.g., LLaMA-8B HumanEval: BC-Mixed 55.76 vs. BC-Select 60.37, but LLaMA-13B MBPP: BC-Mixed 45.01 vs. BC-Select 50.31). The narrative that BC-Select "consistently" outperforms BC-Mixed is not fully supported by the data, and no statistical testing is applied.

### Trivial

- **Qwen-72B uses a quantized setup (4-bit QLoRA) while other models use full or FP16 SFT.** This makes cross-scale architectural comparisons somewhat confounded, though the paper does acknowledge this.

## Nice-to-Haves

- **Compute-matched baselines:** A comparison against standard SGD on BookCorpus (same data, same steps, no gradient ascent) would directly test whether the "unlearning" mechanism is the active ingredient. This is the single most important missing experiment.

- **Retain set ablation:** Test whether F2F retains its gains when the retain set is drawn from out-of-domain data rather than being a subset of the fine-tuning data.

- **Calibration metrics (ECE, Brier score):** If the paper claims improved calibration, these should be reported with numbers and/or reliability diagrams.

- **λ/σ sensitivity analysis:** The theory predicts that increasing λ/σ relative to each other should improve convergence, but no empirical sweep is provided.

- **Computational cost reporting:** The unlearning stage adds non-trivial GPU hours. The cost-benefit tradeoff deserves quantification.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"Missing baselines like continual learning (EWC), knowledge distillation, or task arithmetic."** (Human Finder) — The paper already compares against SFT, DAPT, LoRA, and CurlLoRA. Requesting additional niche baselines beyond these standard methods is scope creep. The more critical missing baseline is a compute-matched control, which is already captured above.

- **"Formatting and presentation nitpicks."** (Spark/Neutral) — Table density, notation consistency, and writing quality issues are minor and not substantive weaknesses.

- **"Reproducibility concerns about undisclosed hyperparameters."** (Neutral) — The paper reports learning rates, batch sizes, epochs, and quantization settings (Section 3.4). The λ/σ values are specified (1.0/0.5). The only gap is the lack of sensitivity analysis, which is a nice-to-have, not a reproducibility failure.

- **"Demand for domain-mixed pretraining baseline."** (Neutral Reviewer) — While this is a reasonable suggestion, the paper already includes DAPT (domain-adaptive pretraining). The more fundamental issue is the compute-matched control, which is captured in the major weaknesses.

## Novel Insights

The most insightful observation across the reviews is the "retain set leakage" confound: because the retain set is drawn from the fine-tuning data, the F2F protocol gives the model an extra exposure to in-domain data before the fine-tuning phase begins. The F2F protocol's "unlearning" phase is simultaneously (a) pushing the model away from forget-set data via gradient ascent, (b) pulling the model toward retain-set data via gradient descent, and (c) providing an extra epoch on domain-relevant retain data before fine-tuning even starts. Disentangling these three effects is critical for assessing whether gradient ascent (the "unlearning" component) is the active ingredient.

## Suggestions

1. **Add a "BookCorpus GD" baseline:** fine-tune from θ₀ on the same BookCorpus subsets using only gradient descent (no ascent component), for the same number of steps, followed by domain SFT. If F2F still wins, the unlearning mechanism is validated; if not, the gains are likely from extra pre-training.

2. **Report calibration metrics** (ECE, Brier score) for medical QA, or remove the calibration claim from the abstract and conclusion.

3. **Ablate the retain set source:** Run F2F with the retain set drawn from out-of-domain data vs. in-domain data vs. no retain set, to isolate the contribution of in-domain data leakage.

4. **Report results across multiple random seeds** and include confidence intervals, at minimum for the headline benchmarks (HumanEval, MBPP).

5. **Systematically characterize failure regimes:** At what model sizes, forget-set sizes, or λ/σ ratios does F2F degrade rather than improve performance? This would greatly increase practical value.

## Evaluation

**Originality:** The idea of repurposing unlearning for domain specialization is genuinely novel. The two-stage protocol is simple but creative.

**Importance of research question:** Negative transfer in LLM fine-tuning is a practical and important problem. The question of whether targeted unlearning can mitigate it is worth asking.

**Whether claims are well supported:** This is the paper's main weakness. The central causal claim ("F2F works because it suppresses harmful pretraining priors") is confounded by missing compute-matched baselines and retain-set leakage. The calibration claim is unsupported by evidence in the paper. The theoretical analysis, while sensible as motivation, is not empirically validated and is used to make strong mechanistic claims.

**Soundness of experiments:** The breadth is a strength, but the lack of critical controls (compute-matched baselines, retain-set ablation, multiple seeds, calibration metrics) substantially limits what can be concluded.

**Clarity:** The paper is generally clear in describing the protocol, though Table 3 is very dense and some results are hard to parse.

**Value to community:** If the confounds were addressed and the unlearning mechanism validated against simpler alternatives, this would be a significant practical contribution. In its current form, it provides an interesting but incompletely validated recipe.

## Score and Decision

Calibration against similar papers:

- **KzSGJy1PIf (SURE, selective unlearning via representation erasure)**: avg 5.7, accepted poster. More focused on privacy unlearning, smaller-scale experiments, but clean experimental design.
- **CGfWyU28Pd (Why fine-tuning struggles with forgetting)**: avg 4.5, rejected. Had a theory-practice gap (linear regression only) similar to this paper, plus limited experiments.
- **pCEgna6Qco (Two-stage LLM fine-tuning / ProMoT)**: avg 6.75, accepted poster. Similar concept (two-stage for better adaptation), but with stronger controls and clearer causal story.
- **f5o6kWRC0A (Machine unlearning for negative transfer)**: avg 4.0, withdrawn/rejected. Similar motivation but weaker execution.
- **1ExfUpmIW4 (LoKU, robust unlearning)**: avg 6.0, accepted poster. Clearer experimental validation, more focused contribution.

This paper has broader empirical scope than the rejected unlearning papers but suffers from a fundamental confound in its experimental design (missing compute-matched baseline) and unsupported calibration claims. It sits above papers like CGfWyU28Pd (4.5, rejected) and f5o6kWRC0A (4.0, rejected) due to much larger experimental scale, but below papers like pCEgna6Qco (6.75) and 1ExfUpmIW4 (6.0) due to the causal attribution issues and missing controls. The novel idea is interesting enough that with proper controls it could be a solid contribution, but the current evidence doesn't convincingly demonstrate that unlearning is the active ingredient rather than extra training.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
Now I have a good understanding of the landscape. Let me synthesize the final review.

Let me carefully verify the key reviewer claims against the paper text:

1. **Novelty claim**: The method is KL(p_θ || p_θ₀) regularization from a frozen copy of the original model. This is indeed very similar to standard knowledge distillation / self-distillation. The paper acknowledges connections to self-distillation and label smoothing in Section 1.1, but frames "label annealing" as a new technique. The RLHF KL penalty against a reference model is also conceptually similar.

2. **Theory**: The paper analyzes label annealing in linear regression by replacing KL divergence with L2 loss on outputs (Equation 8). The paper itself calls this a "simplification."

3. **Evaluation**: Tables 1-2 show clear benefits on some metrics, but the paper uses "fixes the forgetting" language which is strong.

4. **Baselines**: Only L2 regularization toward initialization as a baseline. No other continual learning methods, LoRA, early stopping, etc.

5. **Overclaiming**: The abstract says "improves the model's performance in target domains without sacrificing other capabilities" - but Table 1 shows HumanEval drops from 38.31 to 35.24 with LA (vs base 28.77), and Table 2 shows HumanEval drops from 54.53 to 51.06.

6. **Temperature T**: Introduced in Eq 2 but never ablated or even specified what value was used.

Let me now compile the final review with proper calibration. Similar papers:
- VQ7Q6qdp0P (distillation-based regularization for forgetting): scores 3,8,5,3 (rejected)
- IhbZytsinc (minifinetuning / corrective self-distillation): scores 5,6,5,8 (rejected)
- n2EU4PUrJP (soup/model averaging for forgetting): scores 5,5,3,5 (rejected)
- VrHiF2hsrm (understanding forgetting via implicit inference): scores 3,8,6,6 (accepted poster)

The first three are all rejected with average scores around 4-5. The current paper has a very similar profile: incremental novelty (self-distillation applied to LLM finetuning), modest but real empirical results, limited baselines. The accepted paper (VrHiF2hsrm) had a more novel theoretical contribution (conjugate prompting, implicit task inference) despite simplified experiments.

Based on this calibration, this paper sits in the 4-5 range: it's an empirical study of a well-known technique applied to a new setting, with limited baselines and an oversimplified theoretical story.

## Summary

The paper proposes **label annealing**, a method for mitigating catastrophic forgetting during finetuning of open-weight LLMs when pretraining data is unavailable. The method adds a KL divergence regularization term between the finetuned model's predictions and those of a frozen copy of the original pretrained model. Empirical results on Llama 3 8B across math, code, instruction tuning, and niche domain finetuning show that label annealing preserves source benchmark performance better than L2 regularization toward initialization while maintaining most target domain gains. A theoretical analysis using overparameterized linear regression provides geometric intuition for why the method works.

## Strengths

- **Well-motivated practical problem**: The setting of finetuning open-weight LLMs without access to pretraining data is increasingly relevant. As the paper notes, models like Llama 3 do not release their training data, making experience replay infeasible. This is a real, growing practical need.

- **Simple and easy-to-adopt method**: Label annealing requires only a frozen copy of the initial model and an additional forward pass per batch—straightforward to implement and applicable to any autoregressive LM finetuning pipeline. The method's simplicity is a practical virtue.

- **Clear empirical gains in some settings**: The code finetuning experiment (Table 2) is particularly compelling: direct finetuning drops MATH from 15.92→1.19 (catastrophic), while label annealing restores it to 17.16 while still improving HumanEval from 28.77→51.06. The math finetuning experiment (Table 1) similarly shows TriviaQA preserved at 65.87 vs. direct finetuning's 53.80 (base 67.99). These are non-trivial, practical benefits.

- **Honest limitation discussion**: The paper includes a section (Section 5) discussing replay with RedPajama, showing that replay alone nearly matches label annealing. This is commendably candid.

- **Useful tradeoff characterization**: Figures 2 and 3 demonstrate that varying λ produces a smooth Pareto frontier between target and source benchmarks, which is practically useful for practitioners calibrating the forgetting-adaptation tradeoff.

## Weaknesses

### Major:

- **Limited methodological novelty**: Label annealing is a direct application of knowledge distillation (Hinton et al., 2015) in a self-distillation setting (teacher = frozen initial model, student = finetuned model), which is essentially the same mechanism used in "Learning without Forgetting" (Li & Hoiem, 2017) for continual learning and as the KL penalty in standard RLHF/PPO training. The paper acknowledges connections to self-distillation and label smoothing (Section 1.1) but frames "label annealing" as a distinct method, when the core idea is well-established. The main novelty is the empirical observation that this standard technique works well in the specific setting of open-weight LLM finetuning, not the technique itself.

- **Insufficient baseline comparison**: Only L2 regularization toward initialization is compared, which the paper itself shows is weak. Missing baselines include: (a) other parameter-space continual learning methods (EWC with Fisher information, which the paper mentions but does not evaluate), (b) function-space regularization baselines beyond L2 such as label smoothing, (c) simpler approaches like early stopping or reduced learning rate, and (d) parameter-efficient methods like LoRA that naturally constrain weight changes. Without these comparisons, it is unclear whether label annealing is superior to any reasonable alternative or merely better than one particularly weak baseline.

- **Theory-overpractice gap**: The linear regression analysis replaces KL divergence between softmax distributions with L2 loss on linear outputs (Equation 8), which the paper acknowledges as a "simplification." This changes the optimization geometry fundamentally (KL on softmax probabilities vs. quadratic on outputs). The Abstract and contributions frame this theory as providing "a clear theoretical explanation for why label annealing is more effective," but it actually explains why a *different* regularizer (L2 output matching) works in a *different* model (linear regression). No empirical or theoretical validation connects the "projection into data span" story to what actually happens in transformer finetuning with KL divergence.

### Minor:

- **Overclaiming in framing**: The abstract states label annealing "improves the model's performance in target domains without sacrificing other capabilities," but Tables 1 and 2 show that it does sacrifice some capabilities relative to direct finetuning (e.g., HumanEval drops from 38.31→35.24 in Table 1; HumanEval drops from 54.53→51.06 in Table 2). The body text more honestly calls this a "tradeoff" in Sections 3.3, but the Abstract and Table captions use language like "resolves the forgetting issue" and "fixes the forgetting," which overstates the partial recovery demonstrated.

- **Temperature parameter T is unexplored**: Equation 2 introduces temperature T as a key hyperparameter controlling distribution sharpness, yet no ablation, discussion of chosen T values, or sensitivity analysis appears anywhere in the experiments. This is significant because temperature is central to how distillation works (Hinton et al., 2015) and could materially affect results.

- **Single model scale**: All experiments use Llama 3 8B. No results on larger (70B) or different model families, leaving generalizability unknown. While 8B is reasonable, forgetting dynamics may differ at scale.

- **Computational cost not quantified**: Label annealing doubles memory for forward passes (frozen model + finetuning model) and adds compute per batch. This practical overhead is never discussed, though it is a key consideration for practitioners.

- **Hyperparameter selection requires both target and source benchmarks**: The selection procedure (Section 3.1) filters by target benchmark performance then selects best source performance. This requires access to both benchmark suites during tuning, which may not reflect practical deployment where source capabilities are unknown or unmeasurable.

### Trivial:

- The paper occasionally uses informal language ("fixes the forgetting" in Section 3.2 text) that is slightly stronger than the data warrants, though the tradeoff framing in Section 3.3 is appropriate.

## Nice-to-Haves

- Ablation of temperature T and its interaction with λ
- Comparison with LoRA or other parameter-efficient finetuning methods
- Comparison with EWC (Fisher-weighted) as a more sophisticated parameter-space baseline
- Results on a larger model (e.g., Llama 3 70B) or different model family
- Training curves showing source benchmark degradation over training steps, rather than just final numbers
- Per-subcategory MMLU breakdowns to verify that knowledge preservation is not concentrated in specific domains

## Removed Points

- **"Not yet released" or reproducibility concerns about datasets/models**: The paper uses Llama 3 8B, which is cited and assumed to exist per review rules. Removed.

- **Formatting/style nitpicks**: Several reviewers noted informal language or minor presentation issues. These are trivial and removed from main weaknesses.

- **Demand for the paper to address problems outside its stated scope**: Some reviewers requested comparison with LoRA, model averaging methods, etc. While these would strengthen the paper, the paper explicitly scopes itself to regularization-based forgetting mitigation for open-weight models. Only included LoRA and early stopping as nice-to-haves since they are widely-used alternatives in the same setting.

- **Claim that replay undermines motivation**: The limitation section (Section 5) already addresses this honestly, showing replay is effective when available and noting that label annealing is valuable when reconstruction is difficult. The reviewers' suggestion that this "undercuts the motivation" is overstated—the paper's contribution is specifically for the case when pretraining data is unavailable, which is acknowledged.

- **Demand for theoretical proofs of LLM behavior**: The theory section is presented as providing "geometric intuition" and "interpretation"—the paper does not claim this is an exact analysis of transformer training. However, the Abstract does overstate this as a "clear theoretical explanation," which is included as a major weakness.

## Novel Insights

The most interesting observation is that L2 regularization toward initialization provides almost no benefit in preventing forgetting during LLM finetuning (Tables 1-2), despite being a natural baseline. This suggests that parameter-space constraints are insufficient in the overparameterized LLM regime and that function-space regularization (matching predictions, not parameters) is essential. The linear regression theory's key insight—that direct finetuning discards all pretrained knowledge within the span of the finetuning data, while label annealing preserves a weighted combination—is a useful mental model even if it does not precisely describe transformer dynamics. However, this insight is already implicit in prior work on data-dependent vs. data-independent regularization (e.g., LwF), and the paper does not make this connection explicit.

## Suggestions

1. **Reframe novelty honestly**: Position the contribution as a systematic empirical study of self-distillation/KL regularization for open-weight LLM finetuning, rather than as a new method. Cite and compare directly with "Learning without Forgetting" (Li & Hoiem, 2017), which uses the same KL-from-teacher mechanism.

2. **Add at least one more baseline beyond L2**: EWC with Fisher information computed on finetuning data, or even simply early stopping, would make the comparison much more informative. Showing that label annealing outperforms these would significantly strengthen the case.

3. **Report the temperature T values used**: At minimum, state what T was used in each experiment and how sensitive results are to this choice.

4. **Soften overclaims**: Replace "resolves the forgetting issue" and "without sacrificing other capabilities" with more measured language like "significantly mitigates forgetting" and "with minimal degradation on source benchmarks," reflecting the actual tradeoffs visible in Tables 1-2.

5. **Clarify the theory's role**: Rather than claiming a "clear theoretical explanation" for LLM behavior, frame the linear analysis as providing geometric intuition that is suggestive but not directly applicable to the transformer + KL divergence setting.

## Score and Decision

**Calibration**: Papers with very similar profiles (distillation/KL-based regularization for forgetting, incremental novelty, LLM finetuning experiments) received:
- VQ7Q6qdp0P (feature preservation via distillation, similar novelty concerns): scores 3,8,5,3, avg ~4.75, rejected
- IhbZytsinc (minifinetuning via corrective self-distillation): scores 5,6,5,8, avg ~6, rejected
- n2EU4PUrJP (model averaging for forgetting, overlap with WiSE-FT): scores 5,5,3,5, avg ~4.5, rejected

This paper's profile is closest to VQ7Q6qdp0P—both apply distillation-style regularization to prevent forgetting during finetuning, both have limited baselines, and both overclaim novelty given the prior art. The current paper has somewhat stronger empirical evaluation (contemporary Llama 3 8B, multiple domains) and cleaner presentation, which pushes it slightly above the VQ7Q6qdp0P paper. However, its novelty gap relative to existing work (LwF, self-distillation, RLHF KL penalty) is substantial. The theoretical contribution, while pedagogically nice, does not analyze the actual method used. The paper would be substantially stronger with additional baselines and honest novelty framing.

MY FINAL SCORE: <pineapple>4.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
Now I have a good understanding of the calibration landscape. Let me now write the final review.

The key papers for comparison:
- VQ7Q6qdp0P (Fine-tuning can cripple foundation models; LDIFS method): Scores 3,8,5,3, withdrawn/reject. Very similar idea (KD-based regularization to prevent forgetting during finetuning). Similar novelty concerns.
- IhbZytsinc (Minifinetuning): Scores 5,6,5,8, rejected. Also KD/self-distillation for finetuning forgetting. Similar concerns about novelty, missing baselines.
- fBhgu6PsA7 (MoFO): Scores 5,6,6,6,6,6, rejected. Same domain (LLM finetuning forgetting).
- fHvh913U1H (Forgetting-aware Pruning): Scores 6,3,5,6, withdrawn/reject. Similar domain.

Now let me synthesize the review with careful verification against the paper.

**Key points to verify:**

1. **Novelty concerns**: The paper proposes KL divergence between finetuned and initial model on finetuning data. This IS essentially a specific application of knowledge distillation / self-distillation to finetuning. The connection to Learning without Forgetting (LwF) is real. However, the paper DOES discuss self-distillation and label smoothing in Section 1.1, and acknowledges the connection. The paper's contribution is the specific application to open-weight LLM finetuning where pretraining data is unavailable, plus the linear regression theory.

2. **Hyperparameter selection protocol**: The paper describes (Section 3.1): "we first filter out those choice of λ that lead to no improvement in target benchmarks, and then select the one that has the highest value in source benchmark." This IS biased for LA and L2 because they get to sweep λ. Direct finetuning has no λ. But wait - the paper DOES use the same hyperparameters (5 epochs, 5e-6 LR) for direct finetuning. The concern about unfair selection is partially valid - LA and L2 get to cherry-pick, but direct finetuning also only reports one configuration. The concern about not exploring learning rate / number of epochs for direct finetuning is valid.

3. **Missing LoRA baseline**: This is valid - LoRA is the most common finetuning method for LLMs and naturally constrains weight drift.

4. **Computational overhead**: Valid - two forward passes per step doubles compute.

5. **Temperature T not explored**: The paper introduces T in Section 2.2 as a "sharpness" parameter, but no experiments vary T. This is a valid observation.

6. **Theory-practice gap**: The paper replaces KL with L2 for the linear regression analysis. The paper acknowledges this simplification ("If we simplify the KL divergence penalty with L2 loss"). This is a weakness but acknowledged.

7. **Model name "label annealing"**: Nothing is being "annealed" - both λ and T are fixed. The name is somewhat misleading.

8. **Experiments only on 8B**: Valid, but 8B is a reasonable scale for initial demonstration.

9. **Limited evaluation breadth**: Valid - only HumanEval for code, only MATH/GSM8K for math. But these are standard benchmarks.

10. **No error bars**: Valid, Tables 1 and 2 are point estimates.

11. **Table 3 undermines motivation**: The replay results show that even with approximate pretraining data (RedPajama), replay is competitive. The paper acknowledges this in the limitations section.

Let me now assess the severity of the novelty concern. The core idea is KL(p_θ || p_{θ_0}) as regularization. LwF (Li & Hoiem, 2017) uses KL between old and new model outputs on new data. The main difference is the application context (LLM finetuning) and the temperature scaling. This is a meaningful application but not a novel algorithmic idea. The paper acknowledges connections in Section 1.1.

The hyperparameter selection concern from the harsh reviewer is somewhat valid but somewhat overblown. The L2 baseline DOES get the same cherry-picking advantage as LA (sweep λ, filter on target, pick best source), yet LA still substantially outperforms L2. The concern about direct finetuning not being tuned is somewhat valid but the paper uses standard settings.

The most substantive weaknesses:
1. Incremental novelty (KL with frozen initial model ≈ self-distillation for continual learning, similar to LwF)
2. No LoRA/PEFT baseline
3. Temperature T introduced but never ablated
4. Theory-practice gap (KL → L2 simplification)
5. Hyperparameter selection protocol favors regularized methods

Let me structure the review.</think>Now I have a thorough understanding of the paper and the calibration landscape. Let me write the final review.

## Summary

The paper proposes "label annealing," a method to mitigate catastrophic forgetting when finetuning open-weight language models (where pretraining data is unavailable). The method retains a frozen copy of the initial model and adds a KL divergence regularization term between the finetuned model's predictions and the initial model's predictions on the finetuning data. Experiments on Llama 3 8B across math, code, instruction tuning, and niche domain finetuning show that label annealing preserves source capabilities while maintaining target domain improvements, generally outperforming L₂ regularization toward initialization. A theoretical analysis using overparameterized linear regression provides geometric intuition for why the method works.

## Strengths

- **Practical and important problem.** The paper identifies a real, growing challenge: finetuning open-weight LLMs without access to their pretraining data. As open-weight models proliferate without released training corpora, experience replay becomes infeasible, making weight-only solutions practically valuable (Section 1).

- **Simple, easy-to-implement method.** Label annealing requires only a frozen copy of the initial model and an additional KL divergence term—trivial to add to standard training pipelines (Section 2.2, Figure 1).

- **Strong empirical separation from L₂ baseline.** In math finetuning (Table 1), LA preserves TriviaQA at 65.87% vs. 53.80% for direct finetuning while improving MATH. In code finetuning (Table 2), LA rescues GSM8K from 37.07→52.69 while keeping most HumanEval gains. L₂ regularization fails to achieve comparable results in either task, providing a useful negative result about weight-space regularization in this setting.

- **Clean theoretical analysis.** Theorem 1 and the decomposition in Section 4.3 provide clear geometric intuition: direct finetuning discards pretrained knowledge within the span of finetuning data, while label annealing interpolates between pretrained and finetuned estimates within that span. The contrast with L₂ regularization (which lacks such a clean decomposition) is well-argued.

- **Honest limitations with replay comparison.** The paper includes a comparison with experience replay using RedPajama (Table 3), honestly showing that replay alone outperforms label annealing on target benchmarks (MATH 22.40 vs. 17.94). This contextualizes the method's value when no pretraining data is available.

## Weaknesses

### Fatal

None.

### Major

- **Limited novelty relative to existing KD-based continual learning.** The core method—keeping a frozen copy of the initial model and regularizing with KL divergence between old and new model outputs on the finetuning data—is a direct application of knowledge distillation / self-distillation to continual finetuning. The paper acknowledges connections to knowledge distillation (Section 1.1) and label smoothing, but does not substantively differentiate from prior output-constraining continual learning methods such as Learning without Forgetting (Li & Hoiem, 2017), which uses essentially the same KL-based objective with a frozen previous model on new task data. The temperature parameter T adds a standard KD trick. While the application to open-weight LLM finetuning is timely and practically useful, the algorithmic contribution is incremental. The contribution rests primarily on the empirical demonstration and theoretical analysis, not on a new method.

- **Missing LoRA/PEFT baseline.** LoRA is the most widely used finetuning method for LLMs and naturally constrains how far weights can deviate from initialization (at low rank, changes are restricted to a subspace). Given that the paper's stated goal is "methods that preserve pretraining knowledge with only access to its weights" (Section 2.1), and LoRA operates under the exact same constraint, its absence is a significant gap. A practitioner choosing between LoRA (cheap, single forward pass) and label annealing (two forward passes) has no empirical guidance from this paper.

- **Hyperparameter selection protocol structurally favors LA over direct finetuning.** The paper selects λ via a two-stage protocol: (1) filter for target benchmark improvement, then (2) pick the λ with best source benchmark performance (Section 3.1). This guarantees LA and L₂ report cherry-picked configurations, while direct finetuning reports a single "standardized" configuration (5 epochs, LR 5e-6). The paper does not explore whether a less aggressive direct finetuning schedule (fewer epochs, lower LR, early stopping) could achieve similar tradeoffs. Since LA's regularization effectively softens the finetuning objective, it shares hyperparameter space with these simpler adjustments. The Pareto curves in Figures 2–3 are shown only for LA, with no comparison to varying finetuning aggressiveness or L₂ sweeps, making it impossible to assess whether LA offers something qualitatively distinct from "just regularize more."

### Minor

- **Temperature parameter T is introduced but never empirically investigated.** Section 2.2 presents T as a key component of the method, discussing "sharpness" of distributions. Yet all experiments use a single fixed T (its value is never stated in the main paper). The role and recommended settings of T remain entirely uncharacterized, making it unclear whether T matters at all or whether the method reduces to untempered KL.

- **Theory-practice gap.** The linear regression analysis replaces KL divergence on probabilities with L₂ loss on outputs (Section 4.2: "If we simplify the KL divergence penalty with L₂ loss"). This simplification is acknowledged but the gap is not discussed further. The key geometric insight—preservation within vs. outside the span of finetuning data—depends on linearity and may not carry over to nonlinear transformers with softmax outputs.

- **The method name "label annealing" is misleading.** Neither λ nor T is annealed (gradually changed) during training; both are fixed throughout. The name suggests a schedule that does not exist. This could confuse readers expecting annealing behavior.

- **No variance or reproducibility details.** Tables 1 and 2 report point estimates only. Only Figure 3 shows error bars. Given that differences like MATH 17.10 vs. 17.94 could be within noise, confidence intervals would strengthen the claims.

### Trivial

- Computational overhead (two forward passes per training step) is never quantified or acknowledged in the main text.

## Nice-to-Haves

- Sweep over L₂ λ values shown as Pareto curves in Figures 2 and 3, alongside the LA curves, so readers can see whether L₂ is fundamentally weaker or just poorly optimized at the selected point.
- Ablation over temperature T values (e.g., T ∈ {0.5, 1, 2, 5}) to characterize its effect.
- Comparison with at least one PEFT method (e.g., LoRA at rank 8, 16, 32).
- Early stopping or reduced-LR direct finetuning as an additional baseline to test whether regularization alone achieves similar tradeoffs.
- Experiments on at least one larger model (e.g., 70B) to assess scaling behavior.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **"The paper overclaims general mitigation of forgetting"** — The harsh reviewer states the paper makes "general, unqualified claims" about preventing forgetting. However, the paper's abstract states "mitigates forgetting" (not eliminates) and includes clear limitations (Section 5) including honest discussion of replay's competitiveness. The claims are tempered in context.

- **"Experiments only on Llama 3 8B"** — While true, 8B is a standard scale for LLM research, and the paper provides four distinct finetuning scenarios. Requesting 70B+ experiments is a generic scale-up request that would strengthen but not invalidate current results.

- **"No variance reporting"** — This is true but the paper includes standard benchmarks where variance is typically small; also, Figure 3 does show error bars. Single-run evaluation is the norm in this space.

- **"Missing in-context learning evaluation"** — The abstract mentions ICL as a capability at risk. While evaluating few-shot ICL explicitly would be valuable, the TriviaQA and MMLU benchmarks do probe general capabilities including knowledge that underpins few-shot performance. The ICL concern is partially addressed by these source benchmarks.

- **"Only HumanEval for code"** — HumanEval is the most standard code generation benchmark for this model scale; adding MBPP would strengthen but is not a critical omission.

- **"Replay undermines motivation"** — Table 3 shows replay alone is competitive, but the paper's core motivation is the scenario where pretraining data is unavailable. The replay experiment uses RedPajama, which is only an approximation of Llama's data. The paper argues that reconstructing pretraining data becomes "increasingly difficult" (Section 5), which is a valid point.

## Novel Insights

The linear regression analysis (Theorem 1) yields a genuinely insightful geometric interpretation: direct finetuning preserves pretrained knowledge only *outside* the span of finetuning data (discarding all in-span knowledge), while label annealing retains a convex combination of pretrained and finetuned information *within* that span. This cleanly explains why L₂ weight regularization fails—it lacks this clean span-based decomposition and instead offers an opaque interpolation. The insight that output-space (function-level) regularization preserves more information than weight-space regularization, specifically because it is data-dependent, is valuable even if the method itself is not novel.

## Suggestions

1. Add LoRA as a baseline, even at a single rank, to provide practitioners with a practical comparison against the most common alternative.
2. Vary the temperature T in at least one experiment to characterize its effect; if T doesn't matter, state the default T=1 and explain why.
3. Show L₂ regularization Pareto curves (varying λ) in the alignment/QA experiments so readers can see the full tradeoff landscape for the baseline, not just the single "best" point.
4. Consider renaming the method to something more descriptive (e.g., "prediction-regularized finetuning" or "self-distillation regularized finetuning") since no parameter is annealed during training.

## Score and Decision

Calibration against similar papers:
- **VQ7Q6qdp0P** (LDIFS for finetuning forgetting, KD-based): Scores 3,8,5,3, withdrawn/reject. Very similar method (L2 in feature space). Novelty concerns comparable.
- **IhbZytsinc** (Minifinetuning, corrective self-distillation for finetuning): Scores 5,6,5,8, reject. Extremely similar idea (self-distillation to prevent forgetting). Reviewers raised the same novelty and baseline concerns. Average ~6.
- **MoFO** (same domain, different approach): Scores 5,6,6,6,6,6, reject. Lacks novelty, missing baselines.
- **fHvh913U1H** (Forgetting-aware pruning): Scores 6,3,5,6, withdrawn/reject.

This paper has modest novelty (essentially applying KD/self-distillation to LLM finetuning) but provides clearer theoretical grounding and more systematic evaluation than some comparators. Its main empirical finding—that L₂ regularization barely helps while output-space regularization does—is useful and well-demonstrated. The key weaknesses are: (1) incremental novelty, (2) missing LoRA baseline that most practitioners would want, and (3) hyperparameter selection asymmetry. These are significant but not fatal; the paper makes a solid practical contribution despite limited algorithmic novelty.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
=== CALIBRATION EXAMPLE 13 ===

# Final Consolidated Review
## Summary
The paper proposes *label annealing*, a method for mitigating catastrophic forgetting when finetuning open-weight LLMs without access to their pretraining data. The method augments the finetuning cross-entropy loss with a KL divergence term between the finetuning model's predictions and those of a frozen copy of the initial model on the same finetuning batch. Experiments on Llama 3 8B across math, code, and alignment tasks demonstrate improvements in retaining source-domain capabilities. A theoretical analysis using overparameterized linear regression provides a geometric interpretation of why the method outperforms direct finetuning and L₂ regularization toward initialization.

---

## Strengths

- **Clean geometric theory distinguishing three finetuning regimes.** Theorem 1 gives closed-form solutions for direct finetuning, L₂ regularization, and label annealing under overparameterized linear regression initialized at θ₀. The decomposition in Section 4.3 is insightful: direct finetuning erases the component of θ₀ within the span of finetuning data X, while label annealing replaces it with a convex combination of θ₀ and the minimum-norm solution. This is a genuine theoretical contribution that goes beyond hand-waving.

- **Striking empirical demonstration in code finetuning (Table 2).** Direct finetuning on the code corpus causes a catastrophic MATH drop from 15.92 to 1.19 (the paper traces this to loss of few-shot prompting ability), yet label annealing restores MATH to 17.16—*above* baseline—while retaining 86% of the HumanEval gain. This showcases a qualitatively different failure mode that label annealing specifically mitigates, not just quantitative improvement.

- **Honest and informative Limitations section.** Table 3 directly compares label annealing against replay from RedPajama, showing that replay is stronger (MATH 22.40, GSM8K 69.52 vs. label annealing's 17.94/61.78). Including this unfavorable comparison in the main paper, and combining both methods (Replay + LA further improves to 23.44 MATH), shows intellectual honesty and practical utility rather than cherry-picking.

- **Broad empirical coverage.** The paper covers continued pretraining (math, code) and aligned-model finetuning (instruction tuning, niche domain knowledge), each with appropriately chosen target and source benchmarks. This variety meaningfully supports the generality of the method.

---

## Weaknesses

### Fatal
None.

### Major

- **Missing comparison with Learning without Forgetting (LwF).** Li & Hoiem's LwF (2016/2017) proposes exactly the same mechanism applied to discriminative models: during finetuning on new-task data, regularize with a KL divergence to the initial model's predictions on *that same* new-task data, without requiring old data. The paper's related work section discusses knowledge distillation and self-distillation but does not cite LwF at all. This is the most directly analogous prior work in continual learning, and its omission raises a genuine novelty concern. The authors should position label annealing relative to LwF: is the contribution the application and analysis in the LLM/generative setting, the temperature scaling, or the theoretical analysis? Without this positioning, it is unclear what is new beyond "LwF applied to LLMs."

- **Alignment experiments lack Pareto-frontier comparison with baselines.** Figures 2 and 3 show label annealing's sweep across λ values against direct finetuning and the base model, but L₂ regularization is entirely absent from these figures. While Section 3.2 establishes that L₂ regularization is largely ineffective for base model training, this should be verified in the alignment setting too. Showing that label annealing's Pareto frontier dominates L₂'s is necessary to justify the method's advantage in the alignment regime.

- **Computational overhead unaddressed.** The method requires loading and maintaining a full frozen copy of the initial model and running two forward passes per batch — roughly doubling GPU memory and adding ~50% to training compute at 8B scale. Figure 1 acknowledges the two forward passes, but nowhere in the paper is this cost discussed, measured, or compared against the benefit. For a paper targeting practical LLM finetuning, this is a significant omission.

### Minor

- **Hyperparameter selection using test benchmarks directly.** The selection procedure (Section 3.1) sweeps (λ, T) for label annealing and λ for L₂, filters on target benchmark performance, and picks the best source benchmark score. This is done on the actual evaluation benchmarks, not a held-out validation set. Label annealing has a 2D search space vs. L₂'s 1D, giving it more opportunities to find a configuration that happens to do well on the test metrics. A validation-set–based selection procedure, or at minimum a sensitivity analysis showing that selected hyperparameters are not cherry-picked, would strengthen the claims.

- **No ablation of temperature T.** The paper introduces T as a separate hyperparameter that softens the target distribution, but presents no experiment isolating its effect (e.g., fixing T=1 vs. T>1 across the same λ range). It is unclear whether T provides benefit beyond λ alone, which would affect whether the "annealing" component is necessary at all.

- **Only one model family and scale tested.** All experiments use Llama 3 8B. Generalization to other architectures (e.g., Mistral, Gemma) or to larger scales (e.g., 70B, where the 2× memory cost is most prohibitive) is unknown.

- **Proposition 4.1 cites the wrong equation.** The proposition discusses the pretraining optimization (Eq. 4: `min_θ ½‖X̃θ − ỹ‖²`) but the text reads "gradient descent applied to problem (6)"—which is the direct finetuning objective. This appears to be a numbering error and should be corrected to problem (4).

### Tiny

- **"Label annealing" naming.** The term "annealing" conventionally implies a decay schedule over time, but T is a fixed hyperparameter throughout training. The paper's rationale—that large T recovers label smoothing—makes the connection to "label" clear, but "annealing" remains potentially confusing for readers expecting a temperature schedule. A brief clarification of the naming choice in Section 2.2 would preempt confusion.

- **L₂ theoretical dismissal slightly unfair.** The claim that "L₂ regularization admits no clean intuition why it would help" undersells ridge regression's well-understood bias-variance tradeoff. The stronger and more accurate statement is that L₂ regularization acts on all weight directions equally, without distinguishing which directions overlap with the finetuning data span—leading to the non-orthogonal decomposition that the theory correctly identifies.

---

## Nice-to-Haves

- **Temperature T ablation against T=1.** Fixing T=1 and varying only λ would clarify whether the temperature dimension meaningfully expands the Pareto frontier or merely rescales λ's effect. This would also clarify the "label annealing" name's value add.

- **Combining replay with label annealing.** Table 3 shows that Replay + LA (23.44 MATH) outperforms either alone, but this is mentioned only in Limitations. A dedicated ablation varying replay fraction × LA strength could be a practically useful design chart for practitioners.

- **Granular forgetting diagnostics.** MMLU covers ~57 subjects; analyzing per-subject performance before and after finetuning would directly test the theoretical claim that knowledge "orthogonal" to finetuning data is preferentially preserved.

- **Per-category MATH performance.** The dramatic MATH drop in code finetuning and its recovery by label annealing above baseline is unexplained. Decomposing this by MATH category could illuminate whether the effect is about few-shot prompting format preservation specifically or broader capability retention.

---

## Removed Points
*These points were flagged for removal — treat with caution.*

- **"Abstract overstates 'no sacrifice' claim"** *(Harsh Critic)*: The abstract reads "In mathematics and code finetuning, label annealing improves the model's performance in target domains without sacrificing other capabilities…In alignment finetuning, our method introduces a smooth tradeoff." The abstract is properly qualified and distinguishes the two settings. **Removed.**

- **Confidence intervals / multiple-run statistics** *(Harsh Critic, Spark Finder)*: Single-run evaluation is the dominant norm for LLM benchmark comparisons at this scale and in the ICLR community. Demanding confidence intervals would be imposing non-standard rigor for this setting. **Removed.**

- **Replay as "undermining core motivation"** *(Spark Finder)*: The paper explicitly frames replay as unavailable for open-weight models with proprietary pretraining data and acknowledges in Limitations that approximate replay (RedPajama) can be competitive. The comparison in Table 3 is honest. The core motivation (weight-only access) is well-founded and not undermined. **Removed.**

- **Comparison with EWC** *(Spark Finder)*: The paper notes that L₂ regularization toward initialization is a simplified special case of EWC with uniform weight importance (Section 3.1). Since label annealing already subsumes this comparison, demanding a full EWC baseline with Fisher Information computation is beyond the paper's stated scope and framing, particularly given EWC's known scalability issues at LLM scale. **Removed.**

- **"Unfair" TriviaQA result due to hyperparameter search"** *(Harsh Critic)*: The TriviaQA recovery (+12pp) is notable, but the hyperparameter concern is already listed as a Minor weakness. Separately attributing the result to search-space bias is speculative. **Removed.**

- **Weight/activation drift visualizations** *(Spark Finder)*: Useful analysis but not necessary to support the core claims given the theoretical analysis already provides mechanistic grounding. **Nice-to-have level at most, removed from weaknesses.**

---

## Novel Insights

The most genuinely novel observation—partly surfaced by the reviewers but inadequately emphasized in the paper itself—is the *qualitative* distinction in Table 2: direct code finetuning causes a near-total collapse of MATH performance (15.92 → 1.19) that the paper attributes to loss of few-shot prompting ability, while label annealing restores MATH above baseline (17.16). This is not merely "less forgetting" but a preservation of a *meta-capability* (in-context learning format adherence) that is distinct from the knowledge targeted by finetuning. The theoretical framework in Section 4—which identifies the finetuning data's span as the locus of forgetting—offers a partial explanation: few-shot prompting is a capability encoded in directions correlated with the finetuning distribution, and label annealing selectively preserves those directions. This suggests that label annealing may be especially valuable not for protecting factual recall but for protecting behavioral and meta-cognitive capabilities that happen to be activated by finetuning-distribution inputs.

---

## Suggestions

1. **Explicitly cite and discuss LwF.** Add a paragraph to Section 1.1 comparing label annealing to Learning without Forgetting, clarifying that the contribution is: (a) demonstrating the mechanism works at LLM scale for generative models, (b) the temperature scaling extension, and (c) the overparameterized-regime theoretical analysis. This would transform the novelty concern from a fatal flaw into an honest positioning statement.

2. **Add a computational overhead table.** Report wall-clock training time and peak GPU memory for direct finetuning vs. label annealing on the same setup. Even a one-line note ("X% overhead in time, Y% in memory on A100 80GB") would satisfy practical readers.

3. **Validate hyperparameters on a held-out split.** Partition each benchmark into a validation and test split; use validation performance for λ/T selection, and report test numbers. Alternatively, show a sensitivity heatmap over (λ, T) to demonstrate the selected point is not an isolated peak.

4. **Ablate T.** Run label annealing with T=1 across the same λ grid and compare to the full (λ, T) search on at least one setting (e.g., math finetuning). This will either validate T's importance or simplify the method to a single hyperparameter.

5. **Move Table 3 and its discussion to the main experiments section** and frame it as a "data-available vs. data-free" comparison. The current placement in Limitations undersells what is actually a useful practical finding (replay + LA combination).

---

**Overall Assessment:** The paper addresses a practically important and well-scoped problem, presents a simple and effective method, and provides a genuinely clean theoretical analysis. The empirical results are solid, particularly the dramatic MATH preservation in code finetuning. However, the paper is materially weakened by the absence of any reference to or comparison with Learning without Forgetting, which is essentially the same mechanism. The novelty is moderate but real: the application to modern generative LLMs, the temperature extension, and the overparameterized-regime geometry are genuine contributions—but these need to be articulated explicitly against LwF. The computational overhead omission and incomplete Pareto comparisons in alignment experiments are also meaningful gaps. Conditional on addressing the LwF positioning and the hyperparameter selection transparency, the technical and empirical content is at the ICLR quality threshold.

# Actual Human Scores
Individual reviewer scores: [6.0, 3.0, 5.0]
Average score: 4.7
Binary outcome: Reject

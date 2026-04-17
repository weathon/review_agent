Now I have sufficient information to synthesize the final review. Let me examine the specific tables to verify key claims about Llama3.2 results.

## Summary

This paper systematically ablates GRPO's loss function components to determine which are essential for training LLMs to reason mathematically. Through controlled experiments on small models (0.5B–1.5B parameters), the authors find that (1) negative feedback from below-baseline samples is indispensable—removing it causes training collapse, (2) group-relative advantage estimation is crucial for stability, and (3) PPO-style clipping and policy ratios can be removed without harm. Based on these findings, they propose RGRA (REINFORCE with Group-Relative Advantage), which retains group-relative advantages but removes PPO-style constraints, and show it matches or exceeds GRPO on 17 of 27 benchmark comparisons.

## Strengths

- **Clear, systematic ablation methodology**: Rather than proposing yet another complex GRPO variant, the paper takes a principled decomposition approach—removing one component at a time and measuring the effect. This is a refreshing and scientifically sound way to interrogate which parts of GRPO are actually necessary, and it provides actionable insights for practitioners.

- **Important practical finding on negative feedback**: The demonstration that positive-only GRPO leads to training collapse through reward hacking (degenerate short outputs) is a valuable empirical observation with clear practical implications. Training curves in Figure 1 provide compelling visual evidence of this phenomenon.

- **Multi-model, multi-benchmark evaluation**: Testing across three model sizes from two families (Qwen2.5, Llama3.2) and nine benchmarks spanning English math, Chinese math, and STEM provides reasonable breadth for a resource-constrained study.

- **Emergence of reasoning behaviors**: The qualitative analysis (Figure 2) showing that GRPO/RGRA generate explicit reasoning traces while RAFT/positive-only GRPO do not adds an important dimension beyond benchmark scores, suggesting the training regime affects not just accuracy but reasoning strategy.

## Weaknesses

### Major

- **Central claims are overstated relative to the experimental evidence**: The paper concludes that "PPO-style clipping is unnecessary" (abstract, conclusion) and that RGRA "establishes it as a competitive reinforcement learning objective for reasoning tasks" (conclusion). However, all experiments use models ≤1.5B parameters trained with LoRA (rank 128, ~10% trainable params) on only 1,800 GSM8K examples. PPO-style clipping is specifically designed to prevent destructive policy updates under large distributional shifts—conditions that are structurally unlikely to arise in small LoRA-tuned models with limited training data. The paper acknowledges in one sentence that future work should "address larger models," but the general-sounding title and conclusion vastly overstate the scope of what the experiments support. The appropriate claim is: "for small models with LoRA on short math RL runs, PPO-style clipping appears dispensable." The paper provides no evidence—empirical or theoretical—that this holds at scale, with full fine-tuning, or under conditions of distributional drift where clipping was originally motivated.

- **Inconsistent results on Llama3.2-1B undermine the universality of RGRA**: On Llama3.2-1B, RGRA loses to GRPO on 2 of 3 benchmark categories (Chinese Math: 26.6 vs 30.1; STEM: 22.5 vs 24.9). On Math-English, RGRA marginally wins (20.2 vs 20.1). The paper does not discuss or explain this discrepancy. This is important because it suggests the effectiveness of removing clipping may depend on model architecture or initialization quality, which contradicts the general claim that clipping is dispensable.

- **No variance or statistical significance reported**: All results are single-run point estimates. Many GRPO vs. RGRA differences are small (e.g., 71.0 vs 72.7 on GSM8K for Qwen2.5-1.5B, which is a 1.7-point gap on a 100-point scale). Without standard deviations across multiple seeds, it is impossible to distinguish genuine improvements from run-to-run variance. The "17 out of 27" claim treats each comparison as independent evidence, but several margins are within typical seed variance for LLM training.

- **RGRA underspecified (on-policy vs. off-policy ambiguity)**: The RGRA gradient (Eq. 2) specifies sampling from π_θ (on-policy), whereas GRPO samples from π_{θ_old} (off-policy within update steps). The paper does not clarify whether RGRA is implemented with true on-policy sampling (re-sampling after each gradient step) or off-policy sampling with stale data (as in standard PPO/GRPO implementations). This matters because if RGRA is effectively off-policy without importance correction, it introduces bias that is not acknowledged; if truly on-policy, the sample efficiency comparison with GRPO is unfair unless the number of gradient steps is matched rather than the number of samples. Either way, the theoretical framing as "REINFORCE" is imprecise without this specification.

### Minor

- **LoRA usage may confound the clipping conclusion**: LoRA with ~10% trainable parameters constrains how far the policy can shift from the reference, potentially making clipping redundant regardless of its theoretical value. The paper does not discuss whether the clipping findings would survive full fine-tuning.

- **"Negative feedback is indispensable" claim is partially confounded**: The positive-only GRPO variant zeros out negative advantages but keeps all other hyperparameters fixed, which changes the gradient magnitude distribution. A collapse caused by an optimization hyperparameter mismatch (e.g., effective learning rate being too high for the modified objective) does not necessarily prove that negative feedback is inherently indispensable—it could simply mean the positive-only variant requires different hyperparameter tuning.

- **Limited training domain**: All models are trained exclusively on 1,800 GSM8K instances. Whether findings transfer to other domains (code, logic, general chat) or larger training sets is untested, yet the title asks "Are Complicated Loss Functions Necessary For Teaching LLMs To Reason?"—a broader question than the experiments support.

- **Reasoning behavior analysis is anecdotal**: The Countdown dataset analysis (Figure 2) presents only one or two qualitative examples per method, with no systematic metrics (e.g., average reasoning length, step count, or presence of reasoning markers). The claim about "interpretable reasoning strategies" would benefit from quantitative support.

### Trivial

- The paper uses "17 out of 27" as a headline statistic, but this counts across different model sizes, benchmarks, and training regimes, treating them as independent evidence for the same claim while some margins are negligible.

## Nice-to-Haves

- **Track policy ratio distributions during GRPO training**: Plotting histograms of r_{i,t} = π_θ/π_{θ_old} during training would provide direct mechanistic evidence for whether clipping is ever activated, or whether it's trivially unnecessary in this regime. This would substantially strengthen the empirical case.

- **Learning rate sensitivity analysis**: Since clipping provides implicit robustness to learning rate choices, testing RGRA and GRPO across a range of learning rates would reveal whether RGRA is more fragile to this hyperparameter.

- **Larger model experiments**: Even a single 7B model experiment would dramatically increase confidence in the generality of findings.

- **Comparison with RLOO**: As a closely related REINFORCE-based alternative, direct comparison would help situate RGRA's contribution.

## Removed Points

- **"RGRA is just well-known REINFORCE with baseline, so it lacks novelty"**: While RGRA is indeed a straightforward combination of REINFORCE with group-relative baselines, the paper's main contribution is the systematic ablation analysis, not the novelty of RGRA itself. Discounting the paper solely on method novelty would miss the empirical insight about which GRPO components matter.

- **"Training data contamination on GSM8K"**: The paper explicitly states (Section 3.1) that GSM8K "has been explicitly decontaminated from the training corpora of the models employed." Questioning data contamination would be contradicting the paper's stated experimental protocol.

- **"No computational efficiency analysis"**: While useful, the paper's core claim is about which components are necessary, not about efficiency gains from removing them. This would strengthen but is not required for the paper's stated contribution.

- **"RAFT implementation is degenerate/unfair comparison"**: The paper trains RAFT on only 1,800 examples, which may be suboptimal. However, RAFT is not the paper's primary baseline—GRPO is. The RAFT comparison exists to show the failure mode of ignoring negative feedback, and its collapse is corroborated by the positive-only GRPO ablation.

## Novel Insights

The most interesting empirical finding is that positive-only advantages lead to training collapse through degenerate short-output reward hacking, which is visually demonstrated in Figure 1. This has a clean mechanistic explanation: without negative gradients from below-baseline samples, the model has no pressure to avoid the trivial solution of generating short, format-compliant responses that occasionally earn rewards. This suggests that the group-relative advantage normalization in GRPO serves a dual purpose—beyond reducing variance, it ensures that *every* sample provides a learning signal (either positive or negative), which acts as an implicit regularizer against reward hacking. This insight goes beyond "negative feedback helps" to explain *why* it's structurally important in the specific context of RLHF on reasoning tasks.

## Suggestions

1. **Temper the claims**: Change "PPO-style clipping is unnecessary" to "PPO-style clipping appears unnecessary for small models with LoRA on math reasoning tasks" in the abstract and conclusion. The current general claim is not supported by the evidence.

2. **Discuss the Llama3.2 anomaly**: Acknowledge that RGRA underperforms GRPO on Llama3.2 for Chinese Math and STEM, and hypothesize why (e.g., model initialization quality, tokenizer alignment with group-relative advantages, or policy distribution differences).

3. **Add multi-seed variance**: Even 3 seeds would dramatically increase confidence. Report means and standard deviations.

4. **Clarify the RGRA sampling scheme**: Specify whether RGRA uses on-policy or off-policy sampling in practice. If off-policy, discuss the resulting bias from lacking importance ratios.

5. **Discuss the LoRA confound**: Acknowledge that LoRA constrains policy deviation from the reference, making clipping less necessary, and that findings may not transfer to full fine-tuning.

## Evaluation

**Originality**: Moderate. The systematic ablation of GRPO is timely and well-motivated, but RGRA itself is a straightforward application of REINFORCE with group baselines—a well-established variance-reduction technique. The paper's novelty lies primarily in the empirical analysis.

**Importance of research question**: High. Understanding which GRPO components are essential is practically important given the proliferation of GRPO variants and its widespread adoption in post-training.

**Claim support**: Moderate. The ablation is systematic and the findings are clearly presented, but the claims are significantly overgeneralized from the experimental conditions. The Llama3.2 inconsistency is unaddressed, and no statistical rigor is provided.

**Experimental soundness**: Moderate. Good breadth across benchmarks and models, but limited by small scale, LoRA, single-seed results, and lack of hyperparameter sensitivity analysis.

**Clarity**: Good. The writing is clear and the structure is logical, though the paper would benefit from more honest scoping of its conclusions.

**Value to community**: Moderate-to-high. The finding that positive-only advantages cause collapse and that group-relative advantages matter more than PPO-style clipping for small models is practically useful, even if the broader generalization is premature.

## Score and Decision

Calibration against anchor papers:

- **VinePPO** (scores 3,6,6,5, avg 5.0): Simplification of PPO for LLM reasoning, small models, missing baselines. Comparable scope but with slightly different contribution.
- **RLSF** (scores 6,6,3,3, avg 4.5): Same domain (RL for LLM reasoning), limited evaluation, similar overclaiming concerns.
- **Scaling Relationship** (scores 5,6,5,5, avg 5.25): Narrow evaluation on GSM8K, limited generalizability, useful empirical observations.
- **A-LoL** (scores 5,6,8,6, avg 6.25): More experimental breadth and depth, clear method contribution, similar domain.
- **APA** (scores 5,3,8,5, avg 5.25): Alternative to PPO, moderate novelty, some overclaiming.

This paper sits in a similar niche to VinePPO and RLSF—it asks an important practical question and provides useful empirical evidence, but with overclaimed conclusions and limited experimental scope. It is somewhat stronger than RLSF (which had more fundamental issues) but the overclaiming pulls it below papers like A-LoL or "Does RLHF Scale?" which scoped their claims more carefully. The systematic ablation methodology is a real strength that partially compensates for the overclaiming, and the findings on negative feedback / advantage estimation are genuinely useful.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
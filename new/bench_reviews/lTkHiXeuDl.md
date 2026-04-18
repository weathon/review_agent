Now I have a good understanding of the calibration landscape. Let me synthesize my comprehensive review of the HMoRA paper.

Key calibration points:
- "Tight Clusters Make Specialized Experts" (MoE routing, theoretical+empirical): scores 6,8,8,6, accepted as poster
- "Pushing MoE to the Limit" (MoE+PEFT, similar topic to HMoRA): scores 6,5,8,8, accepted as poster
- "MoRE" (LoRA+MoE multi-task, similar topic): scores 3,3,5,5, rejected
- "Octavius" (LoRA-MoE multi-task): scores 5,5,6,8,8, accepted as poster
- "MoLE" (Mixture of LoRA Experts): scores 3,5,6,6, accepted as poster
- "ReMoE" (MoE routing): scores 8,6,5,8,6, accepted as poster

HMoRA has substantive contributions (hierarchical routing + CGJS loss) but with notable weaknesses in evaluation scope and overclaiming. The "Pushing MoE to the Limit" paper had similar scope (small model, similar evaluation) and was accepted as poster. However, HMoRA has more overclaiming issues with "unsupervised task differentiation" and limited model scale. Compared to ReMoE (score 8/6/5/8/6), HMoRA has less theoretical depth. Compared to "Tight Clusters" (6/8/8/6), HMoRA has similar conceptual novelty but less rigorous proof/analysis.

Given the pattern of similar papers and their scores, HMoRA seems like a borderline paper - it has genuine contributions but with significant overclaiming and limited evaluation. A score around 5 seems appropriate.</think>## Summary

HMoRA proposes a hierarchical mixture-of-LoRA-experts method for fine-tuning LLMs that combines token-level routing in shallow layers with task-level routing in deeper layers, motivated by the observation that different LLM layers capture features at different granularities. A novel Constrained Generalized Jensen-Shannon (CGJS) auxiliary loss is introduced to simultaneously encourage routing certainty and load balance, and the paper claims this loss enables unsupervised task differentiation and generalization to unseen tasks. Experiments on Qwen2 1.5B fine-tuned on Flan v2 show improvements over full fine-tuning and several MoE-LoRA baselines on multiple-choice NLU benchmarks, using only 3.9% of trainable parameters.

## Strengths

- **Well-motivated and clearly implementable architecture**: The hierarchical hybrid routing idea — transitioning from token-level to task-level routing across depth — is grounded in established findings about layer-wise feature granularity (Geva et al., 2021) and is clearly specified with equations. The MoRA block design and the α^(l) schedule are straightforward to implement and replicate.

- **Effective auxiliary loss design**: The CGJS auxiliary loss (Section 3.3) provides a principled information-theoretic formulation that addresses a real problem in MoE routing — the tension between routing certainty and load balance. Table 1 provides direct empirical evidence that CGJS improves over both no auxiliary loss and standard load-balancing loss across soft and top-k routing, which is a convincing ablation.

- **Strong empirical results in the tested setting**: HMoRA (both w/ and w/o lightweight designs) consistently outperforms full fine-tuning and all LoRA-MoE baselines (MoLoRA, MixLoRA, HydraLoRA) across 7 benchmarks, with HMoRA w/o LW achieving 64.16 average vs. 63.15 for Full FT and 63.02 for MoLoRA. The improvements over LoRA-8 are substantial (64.16 vs. 60.76 average).

- **Systematic ablation structure**: Table 1 provides clean ablations across routing methods and auxiliary losses. Table 3 isolates the contribution of the auxiliary loss specifically for the task router. The appendices contain further ablations on hyperparameters for α^(l) and CGJS.

## Weaknesses

### Fatal

None.

### Major

- **Overclaimed "unsupervised task differentiation" and "generalization to unseen tasks"**: The paper's most distinctive conceptual claim is that CGJS enables "unsupervised task differentiation" and "generalization to unseen tasks" (Abstract, Introduction, Section 3.3, Section 4.3). However, this is not substantiated by the experiments. The evidence provided is: (1) t-SNE visualizations showing that gate values cluster by MMLU subtask (Figure 4), and (2) a quantitative metric claiming 42/57 MMLU subtasks are "differentiated" (Appendix E.8). But there is no baseline comparison for the differentiation metric (e.g., would a simple clustering of hidden states without MoE achieve similar separation?), and no experiment that isolates generalization to genuinely novel task *structures* — MMLU subtask categories overlap significantly with Flan v2 training data content. Table 3 shows that disabling the auxiliary loss on the task router hurts accuracy, but this merely confirms the auxiliary loss is useful; it does not establish that its mechanism is "unsupervised task discovery" rather than generic regularization. The task encoder is trained under LM loss, so calling the resulting differentiation "unsupervised" overstates what's happening — the system learns from supervised signal (correct answer labels) plus a regularizer. This overclaiming weakens the paper's core contribution narrative, because if reduced to its actual empirical contribution, it becomes "a new routing regularizer + hierarchical mixing improves MoE-style LoRA fine-tuning," which is more incremental.

- **Narrow evaluation scope limits generalizability claims**: All main results are on Qwen2 1.5B (with LLaMA 3.2 1B relegated to the appendix). All 7 benchmarks are multiple-choice NLU tasks (MMLU, MMLU-Pro, ARC, OpenBookQA, SWAG, CommonsenseQA). The paper claims to make "LLMs more effective" in "multi-task scenarios" broadly, but provides no evidence on generative tasks (summarization, translation, code), instruction-following benchmarks, or models at scales (7B+) where MoE methods are most commonly applied. The dynamics of token vs. task routing might differ substantially at larger scales or on generation tasks. Given the paper's broad claims, this is a significant evidential gap. This mirrors concerns raised in reviews of similar MoE-PEFT papers: *"The zero-shot evaluation tasks considered in this paper are mainly classification/multiple choice selection tasks and require generating a single token. The paper does not clearly articulate the adaptability of the proposed method to more complex tasks, such as summarization, translation, or coding"* (review of "Pushing MoE to the Limit").

- **Confounding of hierarchical routing and CGJS contributions**: The paper introduces two major components — hierarchical hybrid routing and the CGJS auxiliary loss — but never isolates their individual contributions. Table 1 evaluates CGJS with only token-level routing (no hierarchy, no task router). Table 3 removes CGJS from the task router only. There is no experiment showing: (a) full HMoRA with a flat (non-hierarchical) routing strategy, (b) hierarchical routing without any auxiliary loss, or (c) CGJS applied to a simpler architecture. Without these, it is unclear whether the gains come from the hierarchical routing, from CGJS, or primarily from one rather than the other. This significantly reduces the clarity of the paper's contribution.

### Minor

- **Missing error bars / variance for the main comparison results**: The paper states each experiment is "repeated 5 times, and we report the mean of the evaluation metrics" (Section 4), but no standard deviations or confidence intervals are reported. Given that the improvements over Full FT are moderate (64.16 vs. 63.15 average, many individual benchmark differences <1 point), it is unclear whether these differences are statistically significant. The Appendix E.8 quantitative study also reports a single number (73.68%) without variance estimates.

- **Task encoder computational overhead is uncharacterized**: The TaskEncoder (Equation 6) is described as a Transformer encoder processing the full input sequence plus a learnable task embedding, but no analysis of its memory footprint or inference latency cost is provided. While Section 3.2 and Figure 2(c) discuss lightweight designs for MoRA blocks, the TaskEncoder's overhead is not included in efficiency comparisons. For practical deployment, the TaskEncoder's cost relative to the router savings matters.

- **CGJS loss formulation has discontinuities with limited analysis**: The min/max operations in CGJS (Equation 11) create piecewise gradients that are not smooth, yet no analysis of training dynamics or stability at these boundaries is provided. The paper also does not compare CGJS against simpler alternatives that could achieve "low individual entropy + high mean entropy" (e.g., per-token negative entropy regularizer + batch-level KL divergence to uniform), leaving open whether the GJS-based formulation is necessary or merely one of many viable options.

### Trivial

- The description of the "42/57 subtasks differentiated" metric in Section 4.3 is deferred entirely to Appendix E.8 without a brief methodological summary in the main text (only the result percentage is given).

## Nice-to-Haves

- Evaluation on generative tasks (instruction following, summarization) and a larger base model (7B+) to validate the generalizability of claims about LLMs broadly.
- A direct head-to-head ablation comparing flat (uniform α across all layers) vs. hierarchical routing to cleanly isolate the benefit of the hierarchical design.
- Comparison of CGJS against simpler entropy-based regularizers to establish whether the GJS formulation is specifically important.
- Analysis of what the learned experts actually specialize in at different layers (e.g., per-expert activation patterns across tasks and layers) to provide evidence for the claimed shallow-token/deep-task distinction.

## Removed Points

These points are flagged to be removed; treat them with caution.

- **"The model is only tested on small models (1.5B/1B)"**: While the evaluation scope is narrow, the paper explicitly frames this as PEFT fine-tuning for LLMs and demonstrates proof-of-concept. Testing on larger models is a nice-to-have, not a fatal flaw; many accepted MoE+LoRA papers (including "Pushing MoE to the Limit") validate on similarly-sized models. Reduced from fatal/major to a minor point within the broader evaluation scope concern.

- **"No comparison with recent strong PEFT baselines beyond LoRA r=8/64"**: The paper compares against MoLoRA, MixLoRA, and HydraLoRA, which are the most directly comparable methods (MoE+LoRA combination with similar training setups). Alternative LoRA rank placement strategies or scaling configurations are outside the paper's scope.

- **"Full FT baseline may not be optimally tuned"**: The paper states it performed "a hyperparameter search for these baselines and report the best results" (Section 4.2). While more detail could be provided, this is a standard experimental practice concern rather than evidence of unfair comparison.

- **"t-SNE visualizations should not be overinterpreted"**: The paper uses t-SNE only as qualitative support for a claim that is also backed by the quantitative 42/57 metric. This is a reasonable use of visualization.

- **"Novelty concerns — combining existing ideas"**: The hierarchical hybrid routing with layer-dependent mixing and the CGJS auxiliary loss are specific technical contributions that go beyond combining existing methods. The novelty concern is more a matter of degree than absence.

- **"LoRA matrices cannot be merged during inference, increasing memory"**: This is a general property of all MoE+LoRA methods, not specific to HMoRA, and the paper explicitly addresses this with lightweight designs (Appendix C) and training time comparisons (Figure 2c).

- **"Hyperparameter sensitivity — too many new hyperparameters"**: The paper provides ablation studies on ε, μ, γ_c, and γ_b in appendices E.4 and E.5. While the sensitivity is not fully explored, this is a minor practical concern rather than a fundamental methodological flaw.

- **"Claim that directly optimizing GJS hurts performance undermines the principled story"**: This is actually a feature, not a bug — the paper designed CGJS specifically because raw GJS was too constraining, and the min/max clipping was introduced to address this. This is documented and addressed in the paper's own design process.

## Novel Insights

The CGJS auxiliary loss framing — jointly optimizing for certainty (low individual routing entropy) and balance (high mean routing entropy) through a constrained Jensen-Shannon divergence — is a genuinely interesting perspective on MoE routing regularization. Unlike standard load-balancing losses that push toward uniform selection (which sacrifices specialization), CGJS explicitly separates the two objectives and clips them to prevent over-regularization. However, the claim that this produces "unsupervised task differentiation" goes beyond what the evidence supports; the emergence of task-clustered routing is more plausibly attributed to the supervised LM loss shaping the router representations, with CGJS acting as a regularizer that prevents collapse rather than performing true task discovery.

## Suggestions

- Provide a direct ablation comparing hierarchical routing (α increasing with depth) against flat routing (fixed α = 0.5, α = 0, α = 1) in the main text, quantifying how much of HMoRA's gain comes from the hierarchical design vs. CGJS.
- Report standard deviations for all main comparison results, particularly in Table 2, to establish the statistical significance of improvements over Full FT and MoLoRA.
- Temper the "unsupervised task differentiation" and "generalization to unseen tasks" claims to match the evidence (e.g., "the auxiliary loss encourages the task router to cluster similar inputs, which correlates with improved task-level routing") rather than making the stronger unsubstantiated causal claims.
- Discuss the computational cost of the TaskEncoder explicitly, including inference-time FLOPs overhead compared to a pure token-level routing baseline.

## Score and Decision

**Calibration anchors used:**
- "Pushing MoE to the Limit" (MoE+PEFT, similar topic, small model eval, overclaimed generalization): scores 6/5/8/8, Accept poster — stronger empirical breadth than HMoRA (11B model, more tasks), but similar overclaiming on "generalization to unseen tasks."
- "Tight Clusters Make Specialized Experts" (MoE routing, theoretical contribution): scores 6/8/8/6, Accept poster — stronger theoretical depth than HMoRA.
- "ReMoE" (novel MoE routing): scores 8/6/5/8/6, Accept poster — cleaner, more scalable contribution.
- "MoRE" (MoE-LoRA, incremental, marginal improvements): scores 3/3/5/5, Reject — similar concerns about marginal improvements and limited evaluation, but weaker than HMoRA.
- "MoLE" (LoRA combination): scores 3/5/6/6, Accept poster — borderline accepted despite marginal improvements.
- "Octavius" (LoRA-MoE): scores 5/5/6/8/8, Accept poster — similar MoE+LoRA space, similar evaluation concerns.

HMoRA has genuine technical contributions (hierarchical routing design, CGJS loss formulation) and demonstrates consistent empirical improvements, but suffers from (1) significant overclaiming about unsupervised task differentiation that isn't supported by the evidence, (2) narrow evaluation on a single 1.5B model and only multiple-choice benchmarks, and (3) confounded contributions where the individual roles of hierarchical routing and CGJS are not isolated. The paper is above the "MoRE" rejection threshold (3-5 range) due to its more substantial technical novelty, but below the "Tight Clusters" or "ReMoE" acceptance threshold (6-8 range) due to overclaiming and limited evaluation. Compared to "Pushing MoE to the Limit" (avg ~6.75), HMoRA is weaker because it has less evaluation breadth and more overclaiming, despite comparable technical novelty.

MY FINAL SCORE: <pineapple>5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>
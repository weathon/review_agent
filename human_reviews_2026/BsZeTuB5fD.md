# Don't Settle Too Early: Self-Reflective Remasking for Diffusion Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
Mask-based Diffusion Language Models (DLMs) struggle to revise incorrect tokens: once a token is generated, it typically remains fixed. The key challenge is to identify potential errors in the inputs. In this paper, we propose Remasking-enabled Diffusion Language Model (RemeDi), a mask-based DLM that introduces remasking as another fundamental mechanism, enabling more flexible text refinement in diffusion-based text generation. To achieve this, RemeDi jointly predicts token distributions and per-token confidence scores at each step. The confidence scores determine which tokens to be unmasked after the current step, allowing the model to identify tokens with low quality and remask them. These remasked tokens can be resampled with richer context in subsequent steps. We design a remask-aware pipeline to train this ability, including supervised fine-tuning which teaches the model to detect and remask incorrect tokens in addition to predict mask tokens, and reinforcement learning which optimizes full generation trajectories toward higher rewards. Experiments show that RemeDi achieves the state-of-the-art results among open-source DLMs on multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes **RemeDi (Remasking-enabled Diffusion Language Model)**, which adds a self-reflective remasking mechanism to diffusion language models (DLMs). Traditional DLMs fix tokens once unmasked, making them unable to revise early mistakes. RemeDi introduces a dual-stream transformer: a **Token Prediction Stream (TPS)** predicts token distributions, while an **Unmasking Policy Stream (UPS)** predicts per-token confidence. Low-confidence tokens are re-masked and regenerated in later steps, allowing iterative self-correction. Training proceeds in two stages: **Remask SFT** (supervised fine-tuning) teaches the model to identify and remask incorrect tokens using mixed masking and random replacement; **Remask RL** fine-tunes full trajectories using task-specific rewards. Experiments on code, math, and reasoning benchmarks show that RemeDi surpasses prior DLMs (Dream, LLaDA, LLaDOU) and even matches or exceeds autoregressive baselines of similar size.

However, despite the promising empirical results, the paper exhibits several methodological and reporting weaknesses. Many experimental details are under-specified: the reward design for Remask RL is not clearly defined, and critical hyperparameters or normalization schemes are absent, even in the appendix. The paper does not report any compute or training-time comparisons, leaving unclear whether the observed gains stem from algorithmic innovation or simply greater computational cost. Similarly, convergence curves and efficiency analyses are missing, despite claims of faster training. The method section, while conceptually sound, omits low-level implementation details—such as how the UPS interacts with TPS during training or how remask thresholds are selected—making reproduction difficult. While the writing is clear at a high level, the technical exposition lacks sufficient depth and cross-references to equations and figures, resulting in incomplete methodological transparency.

### Strengths
1. **Addresses a concrete limitation of DLMs** — RemeDi directly tackles the inability to revise early decoding errors through a principled, learnable remasking mechanism.
2. **Clear and interpretable design** — The dual-stream architecture (TPS + UPS) separates token prediction from confidence estimation, enabling transparent remasking decisions.
3. **Comprehensive experiments** — Evaluations cover math (GSM8K, MATH), code (HumanEval, MBPP), and reasoning tasks (ARC-C, AlpacaEval), consistently improving over Dream and LLaDA.
4. **Empirical evidence for self-correction** — Qualitative examples and remask frequency analysis show that RemeDi learns to remask more frequently on harder or more structured tasks.
5. **Training stability and convergence** — Compared to LLaDOU RL, Remask RL converges faster and achieves higher accuracy under identical settings.

### Weaknesses
1. **Limited quantitative comparison to other edit-based diffusion models** (e.g., ReMDM, Seed Diffusion). The paper cites prior edit-based diffusion models (e.g., ReMDM, Seed Diffusion) but omits quantitative comparisons, leaving unclear how RemeDi’s improvements scale relative to these baselines.
2. **Under-specified reward formulation in Remask RL** — lacks details on normalization, weighting, or how rewards interact with confidence-based remasking.
3. **Compute and efficiency concerns** — The two-stage SFT → RL training likely increases overall cost, but compute parity with baselines is unreported, making it unclear whether improvements stem from algorithmic design or added compute.
4. **Figure placement and clarity** — Key visualizations (like the dynamic remasking behavior) appear in the appendix instead of the main paper.
5. **Missing convergence curves** — The claim that Remask RL converges as efficiently as other DLMs lacks quantitative support.
6. **Necessity of two-stage training unclear** — It is not shown whether a stronger SFT baseline or a joint training objective could achieve similar effects.
7. **Stability and variance during RL** — The paper lacks reporting on reward variance, remask rate dynamics, or regularization methods.
8. **Ablation under matched compute not provided** — It is unclear whether RL brings improvement beyond additional training time.

### Questions
1. **Comparison scope:** Why were models like *ReMDM* and *Seed Diffusion* not included in quantitative comparisons? Could a smaller-scale reimplementation or proxy be feasible?
2. **Reward design:** Please provide explicit formulas or examples of the reward functions used in the RL phase and clarify normalization or scaling strategies.
3. **Effectiveness of RL:** How does the RL-trained model differ behaviorally from the SFT-only version (e.g., fewer unnecessary remasks, faster correction)?
4. **Visualization and convergence:** Could you move the dynamic remasking visualization (currently in appendix) to the main paper and add training loss curves to verify efficiency claims?
5. **Compute accounting:** Please provide compute cost (GPU hours, wall time, token count) for both stages. How does SFT-only compare to SFT→RL at equal compute?
6. **Necessity of two-stage design:** Did you test a single-stage joint objective or stronger SFT baseline under matched compute?
7. **Stability:** What regularization methods (e.g., KL penalty, reward whitening, gradient clipping) were applied to stabilize RL?
8. **Matched compute ablation:** Could you run an SFT-only baseline using the same compute as SFT→RL to confirm that RL brings genuine benefit?
9. **Generalization to other DLMs:** Since the model is based on LLaDA-8B, could this remasking mechanism generalize to other diffusion LMs such as Dream? If so, what architectural adjustments are needed?
10. **Random alternative tokens:** In SFT, are random alternatives drawn uniformly from the vocabulary or sampled from the model’s top-k predictions?
11. **Inference efficiency:** The paper states RemeDi achieves faster convergence; could you include **inference latency comparisons** (e.g., tokens per second vs. LLaDA), given that TPS + UPS may add computational overhead?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates remasking as a way to improve masked diffusion language models. Remasking works by replacing some already-unmasked tokens with mask tokens, which subsequently get unmasked again. This way, uncertain or incorrect tokens can be revised and potentially fixed, thereby improving accuracy with minimal additional cost.

The proposed method, called RemeDi, introduces two new policies to identify the tokens to be remasked, one based on supervised fine-tuning (SFT) and one based on reinforcement learning (RL). This is done via a dual-stream transformer architecture: The token prediction stream (TPS) predicts unmasked tokens while the unmasking policy stream (UPS) predicts per-token confidence scores (for both masked and unmasked tokens). The resulting model achieves significant gains on a variety of benchmarks compared to baselines.

### Strengths
### S1. Strong benchmark performance
The proposed model achieves strong benchmark performance, beating many diffusion and autoregressive baselines of similar size.

### S2. Flexible post-training method
The proposed method is flexible enough to be generally applicable to many existing masked diffusion models, and can be combined with other optimizations such as block diffusion.

### S3. Analysis of remasking behavior
The analysis of the remasking behavior of the trained policies provides valuable insights. It sheds light into task difficulty as well as which part of the sampling process may be most prone to errors.

### Weaknesses
(ordered by decreasing severity)

### W1. Source of performance improvements is unclear
It is somewhat unclear which improvements stem from the continued (multi-task) pretraining and which come from the remasking during sampling. This question is partially addressed by comparing Remask SFT with vanilla SFT and Remask RL with LLaDOU RL, but some confounders remain. The multitask objective employed by Remask SFT may be beneficial even in a vanilla (or adaptive) inference setting where the model confidence is obtained heuristically. Similarly, the Remask RL approach is very similar to the one proposed by LLaDOU (Huang et al., 2025), so it’s unclear where exactly the improvements come from.
The former can easily be addressed by reporting the performance of Remask SFT using vanilla and adaptive masked diffusion sampling (Kim et al., 2025). The latter can be addressed by providing a detailed analysis of the differences between Remask RL and LLaDOU RL and, if necessary, ablating atomic changes to measure their individual impact (e.g. training vs. sampling improvements).

### W2. Reproducibility
As far as I can tell, there will be no model weights or codebase accompanying the paper, which is a major concern for reproducibility, especially given the clarity concerns regarding the source of the performance gains (W1).

### W3. Motivation and effect of dual-stream architecture
The motivation for using a dual-stream architecture over conceptually simpler approaches (e.g. a dedicated unmasking policy head) is unclear, and no ablations are performed on this. Similarly, the computational overhead of the proposed architecture (more parameters, slower forward pass) may provide an unfair advantage compared to baselines. The former can be addressed by providing appropriate ablations, whereas the latter calls for reporting inference speed of the proposed method compared to baselines in addition to benchmark accuracies.

### Conclusion
As presented, reasons to reject the paper outweigh reasons to accept: While the benchmark performance of the proposed model is impressive and beats state-of-the-art methods (S1), the lack of clarity regarding the source of those gains (W1) together with concerns about reproducibility (W2) and inference speed (W3) amount to significant concerns regarding soundness. I will be happy to increase my final score if these weaknesses can be addressed.

### Questions
- Q1. What is the effect of the multi-task objective (Eq. 5) together with incorrect token augmentation (Eq. 3)? More specifically, what is the performance of Remask SFT when using standard vanilla/adaptive sampling (Kim et al., 2025) compared to full Remask SFT? (also see W1)
- Q2. What is the difference between Remask RL and LLaDOU RL (Huang et al., 2025)? Specifically, is the sampling policy in Eq. 8 not identical to the one from Eq. 9 in LLaDOU? If so, where do the improvements of Remask RL over LLaDOU come from? (also see W1)
- Q3. Will the model weights and/or training code be open-sourced? (also see W2)
- Q4. How does the proposed dual-stream architecture compare to more naive baselines, e.g. a single unmasking policy head or a single additional Transformer block (as employed by LLaDOU)? (also see W3)
- Q5. What is the performance overhead associated with RemeDi compared to baselines? (also see W3)

Nits (not considered for final score):
- L31: Austin et al. (2022) should be cited for discrete diffusion.
- L34: Sahoo et al. (2024) and Shi et al. (2024) should be cited for masked diffusion.
- L103: typo: missing space after “Recent studies”
- L160: typo: “paradigm offers” -> “paradigms offer”
- L203: typo: extra space after “(Guo et al., 2025)”

---

### References
- Austin et al. (2022): https://arxiv.org/abs/2107.03006
- Huang et al. (2025): https://arxiv.org/abs/2505.10446
- Kim et al. (2025): https://arxiv.org/abs/2502.06768
- Sahoo et al. (2024): https://arxiv.org/abs/2406.07524
- Shi et al. (2024): https://arxiv.org/abs/2406.04329

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
RemeDi augments diffusion LMs with a learnable remasking policy: a per-token confidence head (UPS) decides which tokens to unmask or re-mask, trained via remask-SFT then trajectory-level RL, yielding consistent gains on math/code/general tasks under block-wise variable-length decoding.

### Strengths
1. Turns remasking from an inference hack into a learned policy; per-token confidence (UPS) guides when to unmask or re-mask.
2. Two-stage training (remask-SFT + RL) aligns token-level corrections with sequence-level rewards; compatible with monotonic denoising.
3. Works atop block-wise variable-length decoding (LLaDA-style); integrates without changing the base noise family.
4. Confidence head provides interpretable signals and a knob to trade exploration vs. commitment.
5. Orthogonal to many decoding/acceleration tricks; likely to stack with other DLM advances.

### Weaknesses
1. No clear accounting of UPS params/FLOPs/memory or train/infer throughput; lacks equal-quality step/latency comparisons to AR/other DLMs.
2. Missing UPS structure/attachment ablations (bi-residuals, zero-init bridge, layer choice) and sensitivity to ρ_(t,"incorrect" )  and ratio r.
3. RL uses preference and verifiable rewards; robustness and transfer to new domains remain uncertain.
4. Sparse head-to-head with edit-flow/seed-diffusion/predictor-corrector under identical settings.
5. Potential remask oscillation (low-confidence flip-flops); termination/annealing not systematically reported.
6. Higher engineering complexity (dual streams + two-stage training) raises adoption barriers.

### Questions
1. Report UPS overhead (params/FLOPs/memory) and equal-quality latency/tokens-per-second vs. AR/strong DLM baselines, with quality–latency Pareto curves on fixed hardware.
2. Provide ablations for UPS attachments and components (bi-residuals, zero-init bridge, layer choice) plus sensitivity sweeps of (\rho_{t,\text{incorrect}}) and ratio (r).
3. Decompose reward contributions, test cross-domain transfer without retuning, and include robustness/calibration (e.g., ECE) under distribution shift.
4. Add unified head-to-head comparisons with edit-flow, seed-diffusion, and predictor-corrector using identical checkpoints, datasets, and decoding budgets.
5. Quantify remask oscillation and compare termination/annealing (fixed/decayed thresholds, hysteresis) to show stability–quality trade-offs.
6. Summarize engineering cost (LoC, stages, time, hardware) and release a minimal plug-and-play UPS recipe with an inference-only lightweight variant.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper diagnoses a structural limitation of mask-based diffusion language models (DLMs)—namely that once a token is unmasked it becomes effectively immutable, which prevents the model from performing self-reflection and post-hoc correction of its outputs. To mitigate this, the authors propose a self-reflective remasking mechanism (RemeDi) that augments the diffusion process with learnable per-token confidence scores and an explicit remasking operation that can reintroduce previously unmasked tokens for further refinement. They train the mechanism with a two-stage procedure—supervised Remask SFT followed by policy-style Remask RL—so the model learns when and how to re-mask and revise. Crucially, RemeDi preserves the diffusion model’s noise-monotonicity (the monotone decrease of corruption during denoising) while enabling iterative edit-and-refine behavior, and the paper reports substantial empirical improvements on mathematical reasoning, code generation, and general language tasks.

### Strengths
1. RemeDi enables self-reflective iterative refinement by jointly predicting per-token confidence scores and remasking low-confidence tokens so that previously revealed tokens can be selectively re-sampled without breaking the diffusion noise schedule.

2. The paper introduces a principled two-stage training pipeline, with supervised Remask SFT to teach detection and remasking and outcome-based Remask RL to optimize whole-generation trajectories, which yields stronger performance.

3. The remasking mechanism supports a wide range of edit behaviors such as replacement, insertion, deletion, merging and splitting, and produces empirical gains on mathematics, code generation, and general instruction benchmarks compared to prior open-source diffusion language models.

### Weaknesses
1. No reproduction code is provided; please supply an anonymous repository link or include the code in the supplementary materials.
2. The model diagram in the appendix is unclear — please clarify how $p$ and $h$ are predicted simultaneously and whether any new modules were introduced.
3. Please add a baseline that performs RL directly on LLaDA using the same datasets you used.

### Questions
1. I suggest moving some experimental results to the appendix and placing the main architecture figure in the main text.
2. Why did you run experiments with LLaDA rather than LLaDA 1.5?
3. Is inference latency substantially slower compared to the baseline?

### Soundness
3

### Presentation
2

### Contribution
3

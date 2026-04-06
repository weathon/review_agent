=== CALIBRATION EXAMPLE 54 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
*   The title "REINFORCING DIFFUSION MODELS BY DIRECT GROUP PREFERENCE OPTIMIZATION" accurately reflects the paper's core contribution. The abstract clearly states the problem (mismatch between GRPO and diffusion models), the proposed solution (DGPO), and the claimed benefits (efficiency, performance). The central claim of ~30x faster training and a performance boost from 63% to 97% on GenEval is prominently stated. Given the parser artifacts, it's difficult to fully assess the abstract's flow, but the key points are present.

**Introduction & Motivation**
*   The motivation is strong and well-articulated. The paper correctly identifies the key obstacle: GRPO's reliance on a stochastic policy is a poor fit for diffusion models that predominantly use efficient, deterministic ODE samplers. The argument that the practical success of GRPO stems from its use of group-level preference information, not its policy-gradient formulation, is a compelling insight that forms the foundation for DGPO. The introduction clearly lists the negative consequences of forcing a stochastic policy (inefficient SDE sampling, weak learning signal, expensive trajectory training) and contrasts them with the promised benefits of DGPO. The contributions are implicitly clear.

**Method / Approach**
*   **Core Idea:** The core idea—directly optimizing for group-level preferences using a weighted sum of per-sample rewards derived from advantages—is sound and well-motivated as a diffusion-native re-imagination of GRPO.
*   **Derivation & Technical Soundness:** The derivation from the Bradley-Terry model through the proposed parameterization to the final loss (Eq. 17) is logically presented in the main text and appendix. The key trick of using advantage-based weights (Eq. 14) to cancel the intractable partition function \(Z(\mathbf{c})\) is clever and justified. The use of Jensen's inequality to derive a tractable upper bound is standard practice.
*   **Clarity & Reproducibility:** The algorithm (Algorithm 1) is clear. The "Timestep Clip Strategy" is a simple but important practical detail to handle few-step rollouts. However, a **significant clarification is needed**: In Eq. 17, the sum over \(\mathbf{x} \in G^{+}\) and \(G^{-}\) uses the *same* sampled \(\mathbf{x}_t\) and \(\epsilon\) for all samples in the group (as noted after Eq. 16). This is a critical implementation detail for efficiency and variance reduction that should be explicitly stated in the main text near Eq. 17, not just as a side note.
*   **Assumptions:** The method assumes access to a reward function \(r_\phi\). The online setting relies on generating samples from a moving model (\(p_{\theta^-}\)). These are standard and reasonable assumptions for this line of work.

**Experiments & Results**
*   **Scope & Benchmarks:** The experimental evaluation is extensive and appropriate for ICLR. The three tasks (compositional generation, text rendering, human preference) cover distinct and important challenges. The use of multiple out-of-domain metrics (Aesthetic, DeQA, etc.) to guard against reward hacking is a rigorous practice.
*   **Main Results:** The quantitative results (Tables 1 & 2) show compelling improvements. Achieving 0.97 on GenEval significantly outperforms strong baselines, including GPT-4o (0.84) and Flow-GRPO (0.95). Maintaining or improving out-of-domain scores is a strong point.
*   **Efficiency Claims:** The claimed ~20-30x training speedup over Flow-GRPO (Figs. 1 & 3) is a major selling point. However, the basis for this comparison needs **justification**. The paper must clearly state: (1) Are both methods run for the same number of *iterations* or to convergence on a metric? (2) Is the per-iteration time compared, or total wall-clock time? (3) Are all hyperparameters (e.g., number of sampling steps, batch size, group size \(G\)) matched between DGPO and Flow-GRPO for a fair comparison? Figure 3 suggests total time, but the training curves should start from the same initial model. Without this clarification, the speedup claim is weakened.
*   **Ablation Studies:** The ablations (Figs. 4 & 5) are excellent and address key design choices: Timestep Clip strategy, ODE vs. SDE rollouts, online vs. offline, and comparison to DPO. They provide strong evidence for the method's components.
*   **Qualitative Results:** Figures 2, 6, and 7 provide good qualitative support, showing improved instruction following and maintained/gained visual quality compared to Flow-GRPO.
*   **Baselines:** The choice of Flow-GRPO as the primary baseline is appropriate as the state-of-the-art for RL-based post-training of diffusion models. The comparison to Diffusion-DPO is also relevant.
*   **Missing Analysis:** A discussion of the sensitivity to the key hyperparameter \(\beta\) and group size \(G\) would strengthen the paper. How does performance/efficiency trade off with \(G\)?

**Writing & Clarity**
*   Despite severe parser artifacts (e.g., garbled tables, broken text in Section 1), the core technical content and narrative remain understandable. The logical flow from problem statement to solution is clear. The derivations are well-structured. The figures, though affected, convey the necessary information. The writing is technically sound.

**Limitations & Broader Impact**
*   The "Limitations and Future Works" section (Appendix G) is extremely brief, only mentioning potential extension to video generation. The paper should more thoroughly discuss limitations: (1) The reliance on a pre-trained, fixed reward model \(r_\phi\) and its potential biases or limitations. (2) The computational and memory overhead of generating and evaluating \(G\) samples per prompt per iteration, even if faster than Flow-GRPO. (3) The empirical validation is on a specific model (SD3.5-M); how general is the method to other architectures or domains? (4) The "Timestep Clip Strategy" is heuristic; a more theoretical or empirical analysis of its effect would be valuable.
*   The Ethics Statement is appropriately brief given the methodological nature of the work. The Reproducibility Statement is good, promising code release.

### Overall Assessment

This paper presents a novel and well-motivated method, DGPO, for reinforcing diffusion models. The core insight—decoupling group preference learning from the policy-gradient framework—is significant and addresses a real pain point in the field. The technical derivation is sound, and the experimental results are impressive, showing substantial performance gains and dramatic training speedups over the prior state-of-the-art. The work meets the high bar for ICLR in terms of novelty, technical quality, and empirical rigor.

**However, for acceptance, the authors must address two major concerns:**
1.  **Clarify the speedup comparison:** Provide a detailed, fair setup for comparing training time/speed against Flow-GRPO to substantiate the ~20-30x claim.
2.  **Expand the limitations discussion:** Acknowledge and discuss the method's dependencies and potential constraints more thoroughly.

Additionally, improving the clarity around the shared noise sampling in the loss and providing some hyperparameter sensitivity analysis would strengthen the paper further. If these issues are adequately addressed in a revision, this paper would be a strong candidate for acceptance at ICLR.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Direct Group Preference Optimization (DGPO), a novel online reinforcement learning method for aligning diffusion models with reward signals. DGPO directly optimizes group-level preferences without requiring a stochastic policy, enabling the use of efficient ODE samplers and avoiding expensive trajectory-based training. The method achieves ~20-30× faster training than the prior state-of-the-art (Flow-GRPO) while improving performance on compositional generation, text rendering, and human preference benchmarks.

### Strengths
1. **Significant Efficiency Gains**: Empirical results demonstrate DGPO trains approximately 20-30× faster than Flow-GRPO (Figs. 1, 3). This is attributed to avoiding stochastic SDE rollouts, training on full trajectories, and model-agnostic noise exploration.
2. **Strong Empirical Performance**: DGPO achieves state-of-the-art results on the challenging GenEval benchmark (0.97 vs. base 0.63 and Flow-GRPO 0.95, Table 1) and maintains or improves out-of-domain metrics (Aesthetic, DeQA, etc., Table 2). Qualitative results (Fig. 2) show improved instruction following and visual quality.
3. **Clear Technical Motivation and Design**: The paper clearly identifies the mismatch between policy-gradient methods (GRPO) and diffusion mechanics as a key bottleneck. The proposed advantage-based weight design (Eq. 12-14) to eliminate the intractable partition function is a clever and well-explained solution.
4. **Comprehensive Ablation Studies**: Ablations validate key design choices: the timestep clip strategy prevents quality degradation (Fig. 4), ODE rollouts outperform SDE rollouts (Fig. 5a), and online DGPO surpasses offline variants and standard DPO (Fig. 5b).

### Weaknesses
1. **Limited Theoretical Justification**: While the derivation is provided, the paper lacks a formal theoretical analysis of convergence, optimization landscape, or regret bounds compared to policy gradient methods. The use of Jensen's inequality introduces an upper bound (Eq. 16), but the impact of this approximation is not analyzed.
2. **Comparison Scope Could Be Broadened**: The main comparison is against Flow-GRPO and DPO. Other concurrent or relevant works on diffusion RL (e.g., DIPO, DIME, DPO-KTO variants) are mentioned but not empirically compared, making the positioning slightly incomplete for ICLR's competitive bar.
3. **Experiments Centered on One Model**: All experiments fine-tune SD3.5-M. While this is a strong base model, demonstrating effectiveness across different architectures (e.g., latent vs. pixel-based, smaller models) would strengthen the generalizability claim.
4. **Reward Hacking Analysis is Superficial**: The brief discussion of reward hacking (Appendix D) is qualitative and lacks quantitative metrics to show the degree of over-optimization or how DGPO's design mitigates it compared to alternatives.

### Novelty & Significance
**Novelty**: The core idea—directly optimizing group preferences for diffusion models without a stochastic policy—is novel. It effectively decouples the beneficial "group relative information" aspect of GRPO from its policy-gradient framework. The advantage-based weight design to cancel the intractable partition function is a key innovation.
**Significance**: The demonstrated order-of-magnitude training speedup without sacrificing (and even improving) performance is highly significant for the practical post-training and alignment of large diffusion models. It addresses a major efficiency bottleneck in the field. The method is likely to influence future work on efficient RL for generative models.

### Suggestions for Improvement
1. **Strengthen Theoretical Foundation**: Provide a convergence analysis or discuss the optimization properties of the proposed objective. Analyze the tightness of the Jensen's inequality bound and its effect on training dynamics.
2. **Expand Empirical Comparisons**: Include comparisons with other recent diffusion alignment methods (e.g., direct reward fine-tuning, DIPO, or other offline preference optimization variants) to solidify the claim of state-of-the-art performance.
3. **Demonstrate Generalizability**: Show results on at least one additional base diffusion model (e.g., SD-XL or a latent model) to confirm the method is not overly tailored to SD3.5-M's specifics.
4. **Deepen Reward Hacking Analysis**: Conduct a controlled experiment quantifying the trade-off between in-domain reward and out-of-domain quality metrics over training, comparing DGPO, Flow-GRPO, and DPO to formally assess robustness to over-optimization.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison to a broader set of RL and alignment baselines beyond Flow-GRPO.** The paper only meaningfully compares to Flow-GRPO and a brief offline DPO ablation. To claim a new SOTA, comparisons to other established RL methods for diffusion models (e.g., DDPO, DPO, KTO) and reward-weighted fine-tuning methods are essential. Without these, the claim of superior performance is not fully substantiated.

2. **Ablation on critical hyperparameters like group size and advantage weighting.** The method's core relies on group partitioning and advantage-based weights. No systematic study is provided on how performance scales with group size or how sensitive results are to the weighting scheme (e.g., vs. uniform weighting). This directly impacts the understanding of the method's efficiency and robustness.

3. **Controlled efficiency comparison under matched sampling budgets.** The 20-30× speedup claim is based on wall-clock/GPU hours, but it's unclear if the comparison accounts for differences in per-iteration sample generation cost (e.g., number of sampling steps, ODE vs. SDE). A fair comparison requires measuring performance versus total number of sampled images or denoising steps, not just time.

4. **Evaluation on a standard, broad-coverage image quality benchmark.** The paper relies on a few out-of-domain reward metrics (Aesthetic, DeQA, etc.) to guard against reward hacking. However, these are still model-based scores. To truly verify that quality is maintained, metrics like FID, CLIP score, or human evaluations on a diverse, held-out prompt set (e.g., COCO) are necessary.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of reward overfitting and generalization beyond the chosen reward signals.** The paper shows some reward hacking examples in an appendix, but a systematic analysis is missing. For each task, they should plot the in-domain reward vs. out-of-domain metrics throughout training to show when and if performance plateaus or degrades, proving that the method doesn't simply overfit the reward.

2. **Sensitivity analysis of the timestep clip parameter `t_min`.** The timestep clip strategy is presented as critical for preventing overfitting to low-quality samples, but no quantitative ablation is provided. The paper must show how performance (both reward and image quality) varies with `t_min` to justify the chosen value and demonstrate its necessity.

3. **Decomposition of the performance gains: how much comes from ODE sampling vs. the group preference objective?** The claimed benefits are conflated: is the speedup primarily due to using ODEs (which is an implementation choice) or the DGPO objective itself? An ablation where Flow-GRPO is also run with ODE sampling (if possible) would isolate the contribution of the algorithmic innovation.

4. **Analysis of sample efficiency and learning dynamics.** How many online samples are needed to reach a certain performance level? A plot of reward vs. number of generated samples (or training iterations) for DGPO vs. baselines would show if DGPO is truly more sample-efficient, not just faster in wall-clock time.

### Visualizations & Case Studies
1. **Side-by-side qualitative comparisons with all major baselines on the same prompts and initial noise.** The provided figures only compare DGPO to the base model and Flow-GRPO. To properly judge quality, visual comparisons should include other relevant baselines (e.g., DPO) on a standardized set of challenging prompts from each evaluated task (compositional, text rendering, aesthetic).

2. **Visualization of failure modes and limitations.** The paper highlights successes. To build trust, it should show examples where DGPO fails (e.g., prompts where it does not improve over the base model, or where reward overfitting leads to distorted images). This sets realistic expectations and helps diagnose the method's boundaries.

3. **Case study on the effect of group size and advantage weighting.** Visual examples showing how generated images change when using different group sizes or weighting schemes (e.g., uniform vs. advantage-based) would intuitively demonstrate why the proposed design choices matter.

### Obvious Next Steps
1. **Apply DGPO to other diffusion model families and scales.** The experiments are limited to SD3.5-M. The next immediate step is to demonstrate efficacy on other architectures (e.g., SDXL, Flux) to show generality, which should have been included to strengthen the claim of a broadly applicable method.

2. **Combine DGPO with offline data or a replay buffer.** The paper mentions an offline variant but doesn't explore hybrid training. Given the online nature, a clear next step is to incorporate past samples to improve stability and sample efficiency, which is a standard practice in RL that should be discussed and tested.

3. **Investigate multi-objective reward optimization.** The method is evaluated on single rewards. A natural and important extension is optimizing for a weighted combination of rewards (e.g., compositionality + aesthetics), which is a practical need and should be explored to show the method's flexibility.

4. **Provide a more detailed computational cost breakdown.** The efficiency claim should be supported by a table breaking down the cost per iteration (sampling time, training time, memory) for DGPO vs. Flow-GRPO, clearly attributing the source of speedup (e.g., ODE sampling, no full trajectory training).

# Final Consolidated Review
## Summary
This paper introduces Direct Group Preference Optimization (DGPO), a novel online reinforcement learning method for aligning diffusion models with reward signals. DGPO directly optimizes group-level preferences without requiring a stochastic policy, enabling the use of efficient ODE samplers and avoiding expensive trajectory-based training. The method achieves significant speedups over the prior state-of-the-art (Flow-GRPO) while improving performance on compositional generation, text rendering, and human preference benchmarks.

## Strengths
- **Novel and well-motivated approach:** The paper clearly identifies a fundamental mismatch between policy-gradient methods (GRPO) and diffusion models, proposing to directly optimize group preferences instead. The advantage-based weight design to eliminate the intractable partition function is a clever and well-explained innovation.
- **Strong empirical performance:** DGPO achieves state-of-the-art results on the challenging GenEval benchmark (0.97 vs. base 0.63 and Flow-GRPO 0.95) and maintains or improves performance across multiple out-of-domain metrics (Aesthetic, DeQA, ImageReward). Qualitative results confirm improved instruction following and visual quality.
- **Compelling efficiency gains:** Empirical results demonstrate DGPO trains approximately 20-30× faster than Flow-GRPO in wall-clock time, attributed to avoiding stochastic SDE rollouts, training on full trajectories, and model-agnostic noise exploration.

## Weaknesses
- **Insufficient detail on efficiency comparison:** The paper claims ~20-30× faster training than Flow-GRPO but does not fully specify the comparison methodology. It should clarify whether both methods were run for the same number of iterations, to convergence on a metric, or compared on per-iteration cost, and whether hyperparameters (e.g., sampling steps, group size) were matched. This is essential to substantiate a major claim.
- **Limited validation of generalizability:** All experiments fine-tune a single base model (SD3.5-M). While results are strong, demonstrating effectiveness across at least one additional architecture (e.g., SD-XL or a latent model) would strengthen the claim that DGPO is a broadly applicable method, not overly tailored to one model's specifics.
- **Implementation detail omitted from main derivation:** A critical implementation detail—that the same sampled noise `ε` and timestep `x_t` are shared across all samples in a group within the loss (Eq. 17) for variance reduction—is mentioned only briefly after Eq. 16. This should be explicitly stated in the main text near the final objective for clarity and reproducibility.

## Nice-to-Haves
- Sensitivity analysis of key hyperparameters (e.g., group size `G`, KL coefficient `β`, timestep clip `t_min`).
- A more detailed computational cost breakdown per iteration (sampling vs. training time) comparing DGPO and Flow-GRPO.
- Exploration of hybrid training incorporating a replay buffer or offline data to improve stability.

## Novel Insights
The paper's core insight is that the practical success of group-based RL methods like GRPO stems from their ability to leverage fine-grained relative preference information within groups, not from their policy-gradient formulation. By decoupling these two aspects, DGPO preserves the beneficial group-level learning while discarding the inefficient stochastic policy requirement that is ill-suited for diffusion models. This re-framing enables a diffusion-native algorithm that is both faster and more performant, offering a new pathway for efficient alignment of generative models.

## Suggestions
- In the experiment section, add a clear description of how the training time/speedup comparison between DGPO and Flow-GRPO was conducted (e.g., matched iterations, convergence criteria, hyperparameter parity).
- Explicitly state in the main text, near Equation 17, that the same noise `ε` and latents `x_t` are shared across samples within a group during loss computation.
- Consider adding a brief experiment applying DGPO to one additional base diffusion model architecture to support the generalizability claim.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept

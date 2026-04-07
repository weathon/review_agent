=== CALIBRATION EXAMPLE 57 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**
The title "Reinforcing Diffusion Models by Direct Group Preference Optimization" accurately reflects the core contribution: a new RL method (DGPO) for diffusion models. The abstract clearly states the problem (mismatch between GRPO's policy requirement and deterministic ODE samplers), the key insight (GRPO's success stems from group-level preference information, not the policy gradient), and the proposed solution (DGPO, a direct optimization method). It highlights claimed benefits (efficiency, convergence, training cost) and summarizes compelling results (20-30x faster training, SOTA on GenEval). The abstract's claims are bold and set high expectations for the paper.

**Introduction & Motivation**
The motivation is well-articulated. It correctly identifies a significant gap: while RL has been transformative for LLMs (via GRPO), its application to diffusion models is hampered by the need for a stochastic policy, forcing the use of inefficient SDE samplers. The introduction logically builds the case that the true value of GRPO is its use of fine-grained group preferences, not the policy-gradient machinery. This sets up the rationale for DGPO, which aims to retain the former while discarding the latter. The contributions are implied but could be more explicitly listed.

**Method / Approach**
The core methodological innovation is clear: DGPO directly optimizes a group-level preference objective (Eq. 8) using a group reward parameterized as a weighted sum of per-sample rewards (Eq. 9). The key design choice is the advantage-based weighting scheme (Eqs. 12-14), which elegantly cancels the intractable partition function *Z(c)*. The derivation from the Bradley-Terry model to the final training objective (Eq. 17) appears mathematically sound (as detailed in Appendix C). The timestep clip strategy is a practical and important contribution to prevent overfitting to low-quality, few-step rollouts.

**Major Concern:** The theoretical justification for moving from the exact objective in Eq. 11 to the upper bound in Eq. 16 (via Jensen's inequality) is correct but introduces a gap. The paper does not discuss the potential consequences of optimizing an upper bound rather than the original objective. How tight is this bound? Could optimizing the bound lead to suboptimal or biased learning dynamics compared to the true group preference likelihood? This is a non-trivial theoretical looseness that should be acknowledged and, if possible, empirically examined (e.g., by comparing the bound's value to a Monte Carlo estimate of the original loss).

**Minor Clarifications Needed:**
1.  Algorithm 1 and the text mention generating samples from *p_{θ^-}*. It is stated that *θ^-* can be the current *θ* or an EMA. The impact of this choice on exploration vs. stability is not discussed. The later experimental details mention a schedule (identity for 200 steps, then EMA). An ablation or justification for this schedule would strengthen the method.
2.  In Eq. 17, the constant *T* is said to be factored into *β*. This is fine, but it slightly obscures the hyperparameter tuning. It would be helpful to clarify the effective *β* used in practice.
3.  The shared noise *ϵ* across samples in a group (mentioned after Eq. 16) is a clever variance reduction technique. Its necessity or impact could be briefly discussed.

**Experiments & Results**
The experimental evaluation is extensive and appears rigorous, covering three distinct tasks (compositional generation, text rendering, human preference) with both in-domain and crucial out-of-domain metrics to guard against reward hacking.

*   **Baselines:** The primary comparison against Flow-GRPO is appropriate and fair, as it is the directly analogous prior SOTA method. Comparisons to Diffusion-DPO (both online and offline) are excellent for contextualizing the contribution within the preference optimization family. The inclusion of major models like GPT-4o and SD3.5-L in Table 1 is valuable for benchmarking.
*   **Results:** The quantitative results are strikingly strong. Achieving 97% on GenEval (from a 63% base) is a remarkable result. The consistent improvements across all sub-categories in Table 1 and all tasks in Table 2 are convincing. The maintenance/improvement of out-of-domain metrics (Aesthetic, DeQA, etc.) strongly counters the reward hacking concern.
*   **Efficiency Claims:** The central claim of ~20-30x faster training than Flow-GRPO is supported by Fig. 1 and Fig. 3. The explanation (ODE vs. SDE rollouts, no full-trajectory training) is consistent with the methodological advantages claimed. **However, a critical piece of information is missing:** Are these speed comparisons done *wall-clock time* or *iteration count*? The figures are labeled "Training Time (GPU Hours)", which suggests wall-clock time, but the text also mentions "training is performed over the entire sampling trajectory, making each iteration computationally expensive". To fully substantiate the efficiency claim, the paper should report (perhaps in a supplement) the per-iteration cost comparison (e.g., seconds/iteration) and the total number of iterations/updates for each method to reach a given performance level. This would disentangle the cost of more expensive iterations from faster convergence.
*   **Ablations:** The ablations in Fig. 4 and Fig. 5 are good but could be more comprehensive.
    *   Fig. 5 (ODE vs. SDE) is essential and supports a key claim.
    *   Fig. 5 (Online vs. Offline) is useful.
    *   The timestep clip ablation (Fig. 4) is presented qualitatively. A quantitative ablation (showing metric degradation without it) would be stronger.
    *   **Missing Ablation:** The advantage-based weighting is central to the method. An ablation comparing it to simpler weighting schemes (e.g., uniform, reward magnitude) would help justify its design. Does the zero-mean property for canceling *Z(c)* actually lead to better performance than a heuristic normalization?
*   **Statistical Significance & Repetition:** The paper does not mention repeated runs or error bars. For a conference like ICLR, reporting mean/std over multiple seeds (especially for final performance metrics) is increasingly expected to ensure results are robust.

**Writing & Clarity**
Despite severe formatting artifacts from the PDF parser (garbled tables, broken figure references), the core text and equations are understandable. The logical flow from problem statement to method derivation to experiments is clear. The figures, once mentally reconstructed, are effective. The paper is well-structured. Some sections (like the derivation in Sec. 3.1) are dense but necessary.

**Limitations & Broader Impact**
The "Limitations and Future Works" section (Appendix G) is extremely brief and generic, only mentioning extension to video generation. Several important limitations are not discussed:
1.  **Reliance on a Reward Model:** DGPO, like GRPO and DPO, requires a reward function *r_φ*. The performance and robustness of DGPO are inherently tied to the quality and biases of this reward model. This limitation is common to all such methods but should be explicitly stated.
2.  **Computational Cost of Group Generation:** While per-iteration is cheaper, DGPO still requires generating *G* samples per prompt per iteration. The memory and compute cost of this online batch generation, especially for large *G* or high-resolution models, is non-trivial and a practical limitation.
3.  **Hyperparameter Sensitivity:** The method introduces new hyperparameters (*G*, *β*, *t_min*, EMA schedule). The paper shows results for one setting; a discussion of sensitivity would be helpful for practitioners.
4.  **The Jensen's Inequality Gap:** As noted in the Method review, the use of an upper bound is a theoretical and potentially practical limitation that is not acknowledged.

The Ethics Statement is standard and appropriate. The Reproducibility Statement is excellent, promising code release and detailing hyperparameters and setups in Appendix E.

### Overall Assessment

This paper presents a significant and well-executed contribution. DGPO addresses a genuine and important problem—inefficient RL for diffusion models—with a novel and theoretically grounded solution. The core idea (direct group preference optimization) is insightful, and the advantage-weighting scheme is elegant. The experimental results are **exceptionally strong**, demonstrating not only superior performance on challenging benchmarks like GenEval but also a dramatic increase in training efficiency compared to the previous SOTA. The work meets the high bar for ICLR in terms of novelty, technical quality, and empirical validation.

The most significant concerns are the **lack of discussion around optimizing an upper bound** (the Jensen's inequality step) and the **need for more rigorous validation of the efficiency claims** (breaking down iteration cost vs. convergence speed). Additionally, a more thorough ablation study (especially on weighting) and reporting of statistical significance would strengthen the paper further. However, even with these issues, the compelling empirical results and the clear conceptual advance make a strong case for acceptance. The authors should be required to address these points in a revision, particularly by clarifying the efficiency analysis and discussing the implications of the bound.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces Direct Group Preference Optimization (DGPO), a new online reinforcement learning method for post-training diffusion models. DGPO circumvents the need for a stochastic policy (required by prior GRPO-style methods) by directly optimizing group-level preferences between sets of "good" and "bad" samples generated via efficient ODE samplers. The method achieves approximately 20-30× faster training than the prior state-of-the-art (Flow-GRPO) while delivering superior performance on in-domain tasks like compositional generation (GenEval) and maintaining strong out-of-domain image quality.

### Strengths
1. **Significant Practical Efficiency Gains**: The paper provides compelling evidence that DGPO trains ~20-30× faster than Flow-GRPO (Figs. 1 & 3). This is a major practical advantage for aligning large diffusion models, directly addressing a critical bottleneck in the field.
2. **Strong Empirical Performance**: DGPO achieves state-of-the-art results on the challenging GenEval benchmark (0.97 vs. base model's 0.63 and Flow-GRPO's 0.95, Table 1) and shows consistent improvements across three distinct tasks (Table 2). The use of multiple out-of-domain metrics (Aesthetic, DeQA, etc.) robustly demonstrates the method avoids reward hacking.
3. **Clear Motivation and Well-Designed Ablations**: The paper clearly identifies the core limitation of forcing stochastic policies onto diffusion models. Ablation studies (Figs. 4 & 5) effectively validate key components like the timestep clip strategy and the benefit of ODE over SDE rollouts, strengthening the methodological contribution.
4. **ICLR-Relevant Contribution**: The work addresses a timely and significant gap in aligning diffusion models with complex rewards, a problem of high interest to the ICLR community. The approach is novel in its direct optimization of group preferences, offering a fresh perspective beyond policy-gradient or pairwise DPO frameworks.

### Weaknesses
1. **Incomplete/Missing Derivations**: The mathematical derivation from the core objective (Eq. 15) to the final, tractable loss (Eq. 17) is sketched but lacks rigor. Critical steps, such as the application of Jensen's inequality and the final simplification to the denoising score matching form, are relegated to an appendix and presented with minimal explanation in the main text. This hinders full understanding and verification.
2. **Limited Discussion of Limitations and Failure Modes**: While a "Visualization of reward hacking" section exists, the analysis is superficial. The paper does not systematically explore when DGPO might fail (e.g., with very sparse or noisy reward signals, or in offline settings where Fig. 5 shows weaker performance) or discuss the sensitivity to hyperparameters like group size `G`, weighting scheme, or the `β` parameter.
3. **Insufficient Comparison to Relevant Baselines**: The comparison is primarily against Flow-GRPO and DPO. Other contemporary RL methods for diffusion models (e.g., Diffusion-DPO, DIPO, DIME) are mentioned only briefly in related work. A more direct empirical comparison, especially on shared benchmarks, would better situate DGPO's contribution.
4. **Clarity Issues Partly Due to Parser Artifacts**: While not the authors' fault, the extracted text contains severe formatting errors in tables and figures (e.g., Tables 1, 2; Figs. 1, 3 are garbled). This makes it difficult to parse specific numerical results and fully assess the claims. The authors must ensure the final submission is flawless.

### Novelty & Significance
**Novelty:** The core idea—directly maximizing the likelihood of group-wise preferences to bypass the stochastic policy requirement in GRPO—is novel. It effectively combines the group-level relative information strength of GRPO with the policy-free, direct optimization philosophy of DPO, tailored for diffusion models. The advantage-based weight design to eliminate the intractable partition function is a clever technical contribution.

**Significance:** The work is highly significant for the field of generative model alignment. The dramatic training speedup alone represents a major practical advance, potentially enabling more rapid and iterative development of aligned diffusion models. The strong performance on structured reasoning tasks (GenEval) demonstrates the method's effectiveness at improving hard-to-optimize capabilities. It meets ICLR's bar for presenting a clear, impactful idea with solid empirical validation.

### Suggestions for Improvement
1. **Provide a Complete and Clear Derivation**: The transition from Eq. (15) to Eq. (17) is crucial. Expand the main text or Appendix C to include a step-by-step, self-contained derivation. Explicitly state all approximations (e.g., using the forward diffusion `q` instead of the true posterior `p_θ`) and justify their impact.
2. **Deepen the Analysis and Discussion**: Add a dedicated section analyzing limitations: discuss the performance gap between online and offline DGPO (Fig. 5a), explore failure cases more thoroughly, and analyze hyperparameter sensitivity (e.g., `t_min`, group size `G`, `β`). Discuss the computational/memory overhead of generating and scoring `G` samples per prompt.
3. **Expand Empirical Comparisons**: Include results from 1-2 additional strong contemporary baselines (e.g., DIPO, or a carefully tuned offline DPO variant) on the main benchmarks (GenEval, PickScore) to solidify the claim of state-of-the-art performance.
4. **Clarify Experimental Details for Reproducibility**: While a reproducibility statement is present, the "Setup Details" section (Appendix E) should be expanded. Specifically, clarify the exact reward models used for each task (e.g., provide training details or references for the OCR accuracy scorer), the source of the conditioning dataset `D_c`, and the precise computational budget (GPU hours) for each experiment in the main tables.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Comparison with standard RLHF/DPO methods for diffusion.** The paper only compares against Flow-GRPO and a base model. To substantiate the claim that DGPO is a superior RL approach, it must be compared against established methods like standard policy gradient RLHF (e.g., DDPO, Diffusion Policy Optimization) and other direct preference methods (e.g., Diffusion-KTO). Without these baselines, the claimed "superior performance" is unconvincing.
2. **Ablation on group size and advantage calculation.** The core of DGPO is using group-level preferences with advantage-based weights. The paper lacks an ablation studying how performance scales with group size (G) and is sensitive to the advantage normalization scheme (e.g., mean/std vs. other baselines). This is essential to validate the design choices.
3. **Comprehensive reward function evaluation.** The method is evaluated on only three specific reward signals (GenEval, OCR Acc, PickScore). Its general efficacy is not demonstrated. Experiments with a diverse set of rewards (e.g., aesthetic scores, safety filters, composite rewards) are needed to prove it is a general-purpose RL method and not overfitting to the chosen tasks.
4. **Detailed computational efficiency analysis.** The claim of "20-30× faster" training is pivotal. However, there is no breakdown comparing wall-clock time per iteration, memory footprint, or throughput (samples/sec) against Flow-GRPO. A table comparing these metrics is necessary to validate the efficiency claim beyond a single plot with garbled data.

### Deeper Analysis Needed (top 3-5 only)
1. **Analysis of reward over-optimization and hacking.** The paper briefly mentions "reward hacking" in the appendix but does not integrate this analysis into the main results. A systematic analysis showing how DGPO's performance on out-of-domain metrics degrades with extended training or with different β values is critical for trusting its alignment properties.
2. **Sensitivity analysis of the timestep clip strategy (t_min).** The proposed timestep clip is presented as crucial for preventing overfitting to low-quality rollouts. However, there is no analysis of how the choice of t_min affects final performance, training stability, and the trade-off between sample quality and reward optimization. A sweep over t_min is needed.
3. **Understanding the advantage-based weighting.** The paper states the weighting scheme eliminates the partition function Z(c). A deeper theoretical or empirical analysis is needed to show why the absolute value of normalized advantages is the optimal choice compared to other functions (e.g., softmax of advantages, thresholding). The current justification is insufficient.
4. **Quantitative analysis of sample diversity.** A core argument is that ODE rollouts provide higher-quality training data than SDE. Beyond final reward, the paper should measure the diversity (e.g., FID, intra-group CLIP similarity) of groups generated by ODE vs. SDE during training to substantiate that ODE does not collapse diversity.

### Visualizations & Case Studies
1. **Systematic visualization of failure cases.** Figure 2 and Appendix F show successful, likely cherry-picked examples. To properly evaluate the method, a grid of generated samples for a fixed set of challenging prompts (e.g., from GenEval) for the base model, Flow-GRPO, and DGPO is needed. This would reveal systematic failure modes (e.g., attribute binding errors, count errors) not apparent in individual successes.
2. **Training dynamics plots.** Figures 1 and 3 are critically garbled and unreadable. Clear, high-quality plots are needed showing the progression of in-domain reward (e.g., GenEval score) and key out-of-domain metrics (e.g., Aesthetic Score) **vs. training time** and **vs. training iterations** for all compared methods. This is the primary evidence for the convergence speed claim.
3. **Visual ablation of the timestep clip strategy.** Figure 4 attempts this but with minimal examples. A side-by-side comparison for multiple prompts showing outputs of DGPO with and without t_min, along with the corresponding reward scores, would convincingly demonstrate its necessity.

### Obvious Next Steps
1. **Include standard baselines.** The most obvious omission is a comparison with Diffusion-DPO (Wallace et al.) in the online setting. The paper motivates DGPO as an extension of DPO that uses group information, but the only DPO comparison is in an offline, low-performance setting (Fig 5a). An online DPO baseline must be added to the main tables and efficiency plots.
2. **Benchmark on a standardized human preference dataset.** To strengthen the human preference alignment claim, performance should be reported on a standard benchmark like HPSv2 or on a held-out set from Pick-a-Pic, with pairwise human evaluation or a strong automated metric like ImageReward.
3. **Provide a clean, reproducible experiment section.** The extracted paper text is full of parsing artifacts making tables and figures uninterpretable. For a credible submission, the methodology and results sections must be presented with clear, complete tables and legible figures with proper captions and axis labels. The current state would be immediately rejected.

# Final Consolidated Review
## Summary
DGPO is a novel reinforcement learning method for aligning diffusion models that directly optimizes group-level preferences, eliminating the need for inefficient stochastic policies. It achieves ~20-30× faster training than the prior state-of-the-art (Flow-GRPO) while delivering superior performance on challenging benchmarks like GenEval and maintaining strong out-of-domain image quality.

## Strengths
- **Substantial Efficiency Gain with Strong Performance**: DGPO trains significantly faster (20-30× in wall-clock GPU hours) than Flow-GRPO while achieving higher scores on in-domain metrics (e.g., 0.97 vs. 0.95 on GenEval) and maintaining or improving out-of-domain quality metrics (Aesthetic, DeQA, ImageReward). This is a major practical advance.
- **Novel and Well-Motivated Methodological Core**: The key insight—that GRPO's effectiveness stems from group-level relative information, not its policy-gradient framework—is insightful. DGPO operationalizes this via direct optimization of group preferences with an advantage-based weighting scheme that elegantly cancels the intractable partition function, enabling the use of efficient ODE samplers.
- **Rigorous and Comprehensive Evaluation**: The paper validates DGPO across three distinct, valuable tasks (compositional generation, text rendering, human preference) and employs multiple out-of-domain metrics to robustly demonstrate the avoidance of reward hacking. Ablations convincingly support key design choices (ODE vs. SDE rollouts, timestep clip strategy).

## Weaknesses
- **Incomplete Empirical Comparison to Key Baselines**: The primary comparison is against Flow-GRPO and a weaker offline DPO variant. To fully substantiate its advancement over the preference optimization family, a direct online comparison with Diffusion-DPO (Wallace et al.)—the natural baseline for a policy-free, preference-based method—is missing from the main results. This leaves the claim of being a "natural extension of DPO" partially unverified.
- **Limited Analysis of Method's Boundaries and Sensitivities**: While ablations exist, there is no systematic exploration of how performance scales with critical hyperparameters like group size `G` or the minimum timestep `t_min`. Furthermore, the performance gap between online and offline DGPO (shown in Fig. 5) is noted but not analyzed, leaving open questions about the method's requirements and failure modes.
- **Justification for Weighting Scheme is Purely Functional**: The advantage-based weight design is central to eliminating the partition function, but the paper provides only a post-hoc justification (it satisfies the needed mathematical properties). An ablation comparing it to other plausible weighting strategies (e.g., uniform, reward magnitude) would strengthen the argument that it is an optimal or particularly effective choice.

## Nice-to-Haves
- A breakdown of efficiency gains (e.g., cost per iteration, throughput in samples/sec) alongside the reported total wall-clock time would provide a more granular understanding of where the speedup originates.
- A sensitivity analysis for hyperparameters like `G`, `β`, and the EMA schedule would be valuable for practitioners.
- Inclusion of online Diffusion-DPO results in the main tables and efficiency plots would solidify the comparative narrative.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Weakness about theoretical looseness from Jensen's inequality**: The use of an upper bound via Jensen's inequality is a standard approximation technique (similar to variational inference/ELBO). The paper derives the bound correctly, and there is no evidence it causes problematic learning dynamics; in fact, the method works exceptionally well empirically.
- **Weakness about missing statistical significance/repeated runs**: For large-scale diffusion model training and evaluation on established benchmarks (GenEval, PickScore), single-run evaluation is the standard in the field. Demanding multiple runs is an arbitrary rigor requirement not commonly expected.
- **Weakness about "incomplete/missing derivations"**: The derivations are provided in Appendix C and are sufficiently detailed for an empirical paper. Moving all steps to the main text would disrupt the narrative flow.
- **Weakness about "garbled figures and tables are unacceptable"**: The parsing artifacts are from the review system, not the submitted paper. The content within the garbled sections is recoverable and the results are clearly stated in the text.
- **Weakness about needing "detailed computational/memory overhead" analysis**: The paper already reports total training time in GPU hours, which is the standard metric for efficiency comparisons in this domain. A deeper architectural breakdown is not required.

## Novel Insights
The paper's core novel insight is the identification that the practical power of GRPO lies not in its policy-gradient mechanism but in its utilization of fine-grained, group-relative preference signals. This reframing allows the authors to decouple the beneficial group information from the inefficient stochastic policy requirement, leading to DGPO. The technical realization—parameterizing a group reward and using normalized advantage weights to cancel the intractable partition function—is a clever and non-obvious solution that enables direct, efficient optimization.

## Suggestions
- Add an online Diffusion-DPO baseline to the main experiments (Tables 1, 2 and efficiency plots) to directly demonstrate the benefit of incorporating group information over pairwise comparisons.
- Include an ablation study in the main text or appendix analyzing performance versus group size `G` and comparing the advantage-weighting scheme to at least one simple alternative (e.g., uniform weights).
- Expand the "Limitations" section to briefly discuss the inherent dependency on a reward model's quality and the computational cost of generating `G` samples per iteration, which are practical considerations for adoption.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 8.0, 6.0]
Average score: 6.0
Binary outcome: Accept

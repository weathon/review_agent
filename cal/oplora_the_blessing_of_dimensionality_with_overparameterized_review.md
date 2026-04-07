=== CALIBRATION EXAMPLE 38 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title is clear and reflects the core contribution: an overparameterized version of LoRA. The abstract succinctly summarizes the problem (LoRA's difficult optimization), the proposed solution (OP-LoRA, an MLP that predicts adapter weights during training only), the key benefits (flexibility, no inference cost, improved performance), and the empirical results. The claims are specific and appear supported by the later content. No major issues.

### Introduction & Motivation
The introduction effectively motivates the problem. It clearly states the challenges of LoRA optimization (ill-conditioned loss landscape, sensitivity to learning rates) and the limitations of existing solutions (custom optimizers like LoRA-Pro and ScaledAdamW are complex, expensive, and not easily extensible). The introduction of OP-LoRA is well-motivated as a flexible, architecture-agnostic alternative. The contributions are listed clearly. One minor point: the reference to Figure 1 is broken in the provided text (the figure appears garbled), but this is likely a parser artifact and not a content issue.

### Methodology (Section 3)
This is the core technical section. The analysis is detailed but has several points that require clarification or raise concerns.

**3.1 Predicting LoRA Weights:** The method is clearly described. The use of a fixed input vector \( z \) to the MLP is a key design choice (different from input-conditioned hypernetworks). However, the text states "we generate \( A \) and \( B \) as flattened matrices via" Equation (4), but the equation shows a single output. It is unclear if one MLP generates both \( A \) and \( B \) concatenated, or if separate MLPs are used. The later experiments (e.g., OP-DoRA) mention "adding an additional prediction head," suggesting a single MLP with multiple heads. This should be explicitly clarified.

**3.2 Optimization Benefits:** The theoretical derivation is interesting but has significant gaps when applied to the actual method.
1.  **Linear vs. MLP:** The analysis in the main text (Eq. 5-10) treats the reparameterization as linear (\(v = W_2 h\)), ignoring the ReLU and the first-layer parameters (\(W_1, c_1\)). The authors state they treat \(h\) as a free parameter for clarity, which is a substantial simplification. The extension to the MLP case is deferred to Appendix A.1, but that appendix makes strong approximations (e.g., Lemma A.1 uses a first-order Taylor expansion of the ReLU, which is only valid for small perturbations). The final "update rule" (Theorem A.5) is complex and its connection to the simple, intuitive "trainable learning rate" and "adaptive line search" terms from the linear case is not clearly justified. The theoretical claims in the main text feel overstated given the approximations required.
2.  **Condition Number Analysis (Eq. 3):** The derivation of the Hessian condition number bounds for LoRA is sound in the appendix, but its presentation in the main text is very brief. More importantly, the link from this analysis to OP-LoRA's benefits is **not established theoretically**. The claim that "OP-LoRA can dynamically adjust step size and do an adaptive line search, overcoming the optimization difficulties" is motivated by the linear overparameterization analysis, but it remains a hypothesis for the MLP-based OP-LoRA. The paper would be stronger if it provided a theoretical argument (even informal) connecting the MLP's structure to improved conditioning or adaptive step sizes, rather than relying on an analogy to linear overparameterization.

**3.3 MNIST Case Study:** This small-scale experiment is useful for illustrating the claimed benefits (lower loss, less LR sensitivity). The visualization of optimization trajectories (Fig 2b,c) is a good qualitative support. However, the power iteration analysis in Appendix B.1, while interesting, measures \( |v^\top g| \) only at the *end* of training. A more compelling analysis would track this quantity (or the effective condition number) *during* training to show OP-LoRA leads to better conditioning throughout optimization.

### Experiments & Results (Section 4)
The experiments are extensive and cover multiple domains (image generation, VQA, commonsense reasoning). The baselines are comprehensive. However, there are several concerns about the presentation and interpretation of results.

**4.1 Finetuning Stable Diffusion:**
- The CMMD results (Table 1) are impressive, especially the 15-point improvement on Naruto. However, the metric is described briefly. For ICLR, it would be beneficial to include a short explanation of why CMMD is preferred over FID in this context (reference to Jayasumana et al. is sufficient, but a sentence on its advantages would help).
- The qualitative results (Fig. 3 and Appendix figures) strongly support the quantitative gains. However, the authors note a "strong color bias towards red" for baselines, which OP-LoRA reduces. This suggests the baseline LoRA optimization might be getting stuck in a suboptimal basin, which is a good point but should be discussed more explicitly as an optimization failure mode that OP-LoRA alleviates.
- A missing ablation: What is the effect of the MLP's hidden dimension? This is explored later in Fig. 4, but not for SD.

**4.2 & 4.3 VQA and Commonsense Reasoning:**
- The improvements are consistent but modest (often ~1%). The authors correctly note this matches gains from LoRA to DoRA, establishing OP-LoRA as a solid improvement.
- Table 3 (Commonsense) is dense but informative. The comparison with HyperLoader (shared MLP) effectively justifies the per-adapter MLP design. The comparison with a high-rank LoRA (r=466) is crucial to show that gains are not just from more parameters but from *how* they are added.
- **Major Concern: Compute and Memory Costs.** Table 4 and Section 4.4 state OP-LoRA uses 69GB vs. LoRA's 44GB (a 57% increase) and training time increases from 3.5h to 4h (14% slower). While this is better than LoRA-Pro, the memory increase is non-trivial. For large models, this could be a barrier. The paper should discuss this trade-off more explicitly: OP-LoRA provides better performance but requires more memory during training. The fact that it's still faster than ScaledAdamW is a plus.

**4.4 OP-LoRA Analysis:**
- The ablation on MLP width (Fig. 4) is useful but its interpretation is vague. The inverted-U for VL-BART is noted, but no hypothesis is offered for why performance degrades with very large widths. Is it an optimization issue? The flat trend for Commonsense is interesting. This section needs more analysis.

### Additional Results (Appendix B)
The appendix contains valuable supplemental studies.
- **Matrix Factorization (B.2):** This is an excellent controlled experiment that directly tests the optimization dynamics. The findings (adaptive step size, better convergence to SVD solution) provide strong empirical support for the theoretical intuition from Section 3.2. This should arguably be in the main text.
- **Gradient Analysis (B.1):** Supports the claim that OP-LoRA is less sensitive to poor conditioning.
- **Extensions to VeRA and Mix-of-Show (B.5, B.6):** Demonstrate generality effectively.
- **Stability of OP-DoRA (B.4):** The reduced variance is a significant practical benefit.

### Writing & Clarity
Overall, the paper is well-written. The methodology section is the most complex, and as noted, the jump from linear theory to MLP practice needs clearer signposting. The figures are referenced correctly in the text (though some are garbled in the parsed version). The appendices are thorough. The reproducibility statement is present.

### Limitations & Broader Impact
The paper acknowledges increased GPU memory usage during training. Other limitations that could be discussed:
1.  **Theoretical Gap:** The core theoretical motivation (acceleration in linear overparameterization) does not directly translate to the proposed MLP with ReLU. This is a conceptual limitation.
2.  **Hyperparameter Tuning:** The MLP width is a new hyperparameter. While Fig. 4 shows it's not overly sensitive in some cases, it still requires tuning. The paper could provide a heuristic (e.g., width = 32 works well).
3.  **Initialization Sensitivity:** The initialization of the MLP (described in Appendix C.1) is important. Was this scheme found to be robust, or does performance depend on it?
The Ethics Statement is appropriate.

### Overall Assessment
This paper presents a simple, novel, and effective method for improving LoRA optimization. The core idea—using a train-time-only MLP to predict adapter weights—is clever and has clear practical benefits: no inference overhead, easy extension to new adapter types, and significant performance gains, particularly in image generation. The empirical evaluation is comprehensive and convincing across multiple tasks. The main weaknesses are the incomplete theoretical grounding (the analysis relies heavily on a linear approximation) and the increased memory footprint during training. Despite these issues, the practical contribution is substantial. The method is easy to implement, and the empirical results are strong enough to suggest OP-LoRA could become a standard tool for LoRA-based fine-tuning. For ICLR, this represents a solid contribution that advances the state of the art in parameter-efficient fine-tuning.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces OP-LoRA, a novel reparameterization of Low-Rank Adaptation (LoRA) that uses a small MLP (hypernetwork) to predict adapter matrices during training, which is discarded after training. This provides a "train-time overparameterization" that improves optimization dynamics—acting as an adaptive learning rate and line search mechanism—without increasing inference cost or storage. The method consistently outperforms standard LoRA and several advanced variants across image generation (Stable Diffusion), visual question answering, and commonsense reasoning tasks.

### Strengths
1. **Strong empirical performance across diverse tasks.** OP-LoRA shows consistent gains over LoRA and its variants (DoRA, AdaLoRA, etc.) in text and vision-language tasks (e.g., +1-4% on commonsense reasoning) and particularly large improvements in image generation (up to ~15 CMMD points on Stable Diffusion). The results are validated on multiple benchmarks (VQAv2, GQA, NLVRv2, Commonsense tasks) and models (LLaMA, VL-BART, SD-XL).
2. **Elegant, flexible, and low-overhead design.** The method adds zero inference cost because the MLP is discarded after training. It is architecture-agnostic and easily extends to other adapters (e.g., OP-DoRA by adding an extra MLP head). Implementation is simple (a few lines of code), and training wall-time overhead is modest (~15% over LoRA) compared to custom optimizers like LoRA-Pro (14x slower).
3. **Theoretical analysis of optimization benefits.** The paper provides a clear derivation showing how the reparameterization introduces a "trainable learning rate" and "adaptive line search," explaining improved conditioning and reduced sensitivity to learning rate. Additional analysis (gradient alignment, matrix factorization case) supports the claimed acceleration mechanism.

### Weaknesses
1. **Increased training memory and compute.** While inference cost is unchanged, OP-LoRA significantly increases GPU memory during training (e.g., from 44GB to 69GB for LLaMA 7B) due to the MLP parameters. This could limit applicability on memory-constrained devices, and the training time, while better than custom optimizers, is still longer than standard LoRA.
2. **Theoretical analysis relies on simplifications.** The acceleration analysis (Section 3.2) assumes a linearized setting (ignoring higher-order terms and ReLU non-linearities) and treats the hidden vector \(h\) as free; the extension to MLPs in the appendix is heuristic. The claim that overparameterization does not increase capacity (hence no overfitting risk) is intuitive but not rigorously proven for the non-linear case.
3. **Limited ablation and sensitivity studies.** While the effect of MLP width is briefly examined (showing an inverted-U curve for some tasks), other design choices (MLP depth, activation functions, initialization schemes) are not explored. The method’s performance under extremely low-rank settings (e.g., r=1,2) or its sensitivity to optimizer choice (beyond AdamW) is not thoroughly tested.

### Novelty & Significance
**Novelty:** The core idea—using a throwaway hypernetwork to overparameterize LoRA weights only during training—is novel and clever. It differs from prior hypernetwork approaches (e.g., HyperDreamBooth) by not conditioning on input and from custom LoRA optimizers (e.g., LoRA-Pro) by being optimizer-agnostic. The theoretical interpretation as an implicit acceleration mechanism builds on prior work (Arora et al.) but is applied in a new context.

**Significance:** The work addresses a real pain point in LoRA optimization—ill-conditioned loss landscapes and sensitivity to learning rates—with a simple, general solution. The consistent gains across modalities and the large improvement in image generation (a high-stakes application for personalization) are practically significant. The paradigm of "train-time overparameterization" could inspire further research.

### Suggestions for Improvement
1. **Deepen the theoretical analysis.** Provide a more rigorous treatment of the non-linear (ReLU) case, perhaps using recent results on overparameterized deep linear networks or mirror descent interpretations. Formalize the "no increased capacity" claim by analyzing the Rademacher complexity or linear stability of the reparameterization.
2. **Conduct more comprehensive ablations.** Systematically vary MLP depth, width, activation, and initialization across tasks to provide clearer design guidelines. Test extreme low-rank regimes (r=1,2) and very high-rank regimes to see where OP-LoRA’s benefits saturate. Compare to simply increasing LoRA rank with a parameter budget matched to OP-LoRA’s train-time parameters.
3. **Evaluate on a broader set of tasks and models.** Include more diverse fine-tuning scenarios (e.g., multilingual adaptation, code generation, long-context tuning) to demonstrate generality. Test with other base optimizers (e.g., SGD, Lion) to see if benefits hold universally.
4. **Discuss limitations and failure modes more explicitly.** When does OP-LoRA not help or even hurt? For example, on very small datasets where overfitting might still occur despite the fixed-rank constraint, or when training memory is extremely limited. Provide practical recommendations for when to use OP-LoRA vs. simpler baselines.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare OP-LoRA with LoRA at matched trainable parameter counts.** The paper compares to a high-rank LoRA (r=466) but not to a rank that matches OP-LoRA's training-time parameter budget. Without this, gains could simply stem from more trainable parameters, not the reparameterization.
2. **Ablate MLP architecture choices.** The paper uses a two-layer ReLU MLP but does not test whether non-linearity or depth is necessary. A linear projection or different depths should be compared to validate the claimed acceleration mechanism.
3. **Test sensitivity to MLP initialization.** The method initializes the MLP for B to zeros; robustness to other initializations (e.g., random) should be shown to ensure the approach is not fragile.
4. **Include classic NLP fine-tuning benchmarks (e.g., GLUE/SuperGLUE with BERT/RoBERTa).** Current experiments cover image generation, VQA, and commonsense reasoning but omit standard NLP adaption tasks, limiting claims of generality.

### Deeper Analysis Needed (top 3-5 only)
1. **Empirically measure the "trainable learning rate" and "adaptive line search" terms during training.** The theoretical derivation suggests these mechanisms, but no evidence shows they actually occur in practice on real tasks.
2. **Directly compare Hessian condition numbers of OP-LoRA and standard LoRA.** The paper argues OP-LoRA improves conditioning but only provides indirect gradient-norm evidence (Table 5). Computing condition numbers would directly support the claim.
3. **Analyze why OP-LoRA outperforms custom optimizers (ScaledAdamW, LoRA-Pro) on image generation but not on commonsense reasoning.** This discrepancy suggests task-dependent efficacy; understanding the cause would clarify the method's strengths and weaknesses.

### Visualizations & Case Studies
1. **Visualize loss landscapes for larger-scale models (e.g., LLaMA or SDXL).** The 2D trajectories on Rotated MNIST are suggestive but not convincing for complex models. Projected loss landscapes would show whether OP-LoRA truly smooths optimization.
2. **Show failure cases or scenarios where OP-LoRA underperforms.** The paper presents mostly successful qualitative results; examples where OP-LoRA fails relative to baselines would help identify limitations and boundary conditions.

### Obvious Next Steps
1. **Apply OP-LoRA to more LoRA variants (e.g., AdaLoRA, LoRA+).** The paper extends to DoRA and VeRA but should demonstrate ease of extension to other popular adapters to fully support the architecture-agnostic claim.
2. **Explore multi-task adaptation with shared MLPs.** Since HyperLoader (which shares MLP parameters) performed poorly, investigating whether OP-LoRA can be adapted to multi-task learning by carefully sharing components is a logical next step.

# Final Consolidated Review
## Summary
OP-LoRA introduces a train-time overparameterization of LoRA adapters using an MLP that predicts adapter weights, which is discarded after training. This approach improves optimization by providing adaptive step sizing and line search, leading to better performance without inference overhead. The method consistently outperforms standard LoRA and its variants across image generation, VQA, and commonsense reasoning tasks.

## Strengths
- **Strong and consistent empirical gains across multiple domains:** OP-LoRA improves CMMD scores by up to 15 points on Stable Diffusion image generation and shows 1‑4% improvements on commonsense reasoning and VQA tasks, demonstrating broad applicability and significant performance lifts where it matters most (e.g., personalization).
- **Practical design with zero inference cost:** The MLP is used only during training and discarded, so storage and inference costs match standard LoRA. The approach is easily extensible to other adapters (e.g., OP‑DoRA via an extra head) and requires only minimal code changes, offering a flexible alternative to complex custom optimizers.
- **Insightful theoretical and empirical analysis:** The paper derives an acceleration mechanism (trainable learning rate and adaptive line search) from overparameterization, supported by controlled matrix factorization experiments and gradient analysis (e.g., larger \( |v^\top g| \) in high‑curvature directions), providing a plausible explanation for the improved optimization dynamics.

## Weaknesses
- **Increased training memory and compute:** OP‑LoRA requires significantly more GPU memory during training (e.g., 57% more for LLaMA‑7B) and slightly longer wall time. While still more efficient than custom optimizers like LoRA‑Pro, this overhead could limit adoption in memory‑constrained environments.
- **Theoretical analysis relies on linear approximations:** The core theoretical motivation uses a linear reparameterization to derive acceleration properties; the extension to the actual MLP with ReLU is heuristic (Appendix A.1 uses first‑order Taylor expansions). Although empirical results support the benefits, the theoretical grounding for the non‑linear case remains incomplete.
- **Limited exploration of design choices:** Ablations are mostly restricted to MLP width; the impact of depth, activation functions, initialization schemes, and extreme low‑rank settings is not systematically studied. This leaves practitioners without clear guidelines for adapting the method to new scenarios.

## Nice-to-Haves
- A direct comparison of OP‑LoRA with a rank‑increased LoRA that matches the train‑time parameter count would help isolate the effect of reparameterization from simply having more parameters.
- Measuring Hessian condition numbers directly during training could provide stronger evidence for the improved‑conditioning claims.
- Including classic NLP fine‑tuning benchmarks (e.g., GLUE) would further demonstrate generality across task types.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Clarity on MLP output:** The criticism that the paper does not explicitly state whether one MLP generates both \(A\) and \(B\) concatenated or uses separate heads is minor; the description in Section 3.1 and the mention of “additional prediction head” for OP‑DoRA are sufficient for replication.
- **Tracking gradient alignment throughout training:** The suggestion to track \( |v^\top g| \) or condition numbers throughout optimization instead of only at the end is an interesting analysis but not required to validate the paper’s claims.
- **Loss‑landscape visualizations for large models:** While such visualizations could be illustrative, they are not essential given the comprehensive empirical results and the matrix‑factorization case study that already illuminates the optimization behavior.

## Novel Insights
The paper’s core insight is that train‑time overparameterization via a simple MLP can reshape the optimization landscape of LoRA, effectively providing adaptive step sizes and line search without increasing inference cost. This idea of “throwaway” parameters that only aid training is a promising paradigm that could be applied beyond LoRA to other parameter‑efficient fine‑tuning methods, opening a new direction for improving optimization in constrained adaptation settings.

## Suggestions
- Provide a heuristic or rule‑of‑thumb for choosing the MLP hidden dimension based on the adapter rank or model size (e.g., width = 32 works well across many tasks).
- Discuss more explicitly the scenarios where OP‑LoRA might not be beneficial (e.g., when training memory is extremely limited or on very small datasets where overfitting could still occur) to guide practitioners.
- In the theory section, clearly state the approximations made (e.g., treating \(h\) as free, first‑order ReLU expansion) and their validity to avoid overclaiming the theoretical guarantees.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 4.0, 2.0]
Average score: 3.0
Binary outcome: Reject

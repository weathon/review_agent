=== CALIBRATION EXAMPLE 77 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's core focus: refining condition errors in autoregressive image generation with diffusion loss (despite a parsing artifact). The abstract clearly summarizes the contributions: a theoretical analysis showing autoregressive modeling with diffusion loss mitigates condition errors and stabilizes the condition distribution, plus a novel Optimal Transport (OT)-based refinement method to address "condition inconsistency." The claims are bold and align with the paper's content. However, the abstract does not mention any limitations or computational considerations, which would be helpful for context.

### Introduction & Motivation
The introduction effectively situates the work within the landscape of diffusion and autoregressive image generation. It motivates the study by identifying an underexplored comparative analysis between conditional diffusion and autoregressive modeling with diffusion loss. The four contributions are listed clearly. The motivation is solid: to understand theoretical differences and improve condition consistency.

### Preliminaries (Section 2)
Standard, concise, and clear. No issues.

### Theoretical Analysis (Section 3)
This is the paper's core theoretical contribution. While ambitious, it has significant clarity and grounding issues.

*   **Section 3.2 (Error Definitions):** Theorem 1 (conditional score matching loss upper-bounds unconditional loss) is a known or straightforward result, but its presentation is valid. The definitions of \(\epsilon_c\) and \(\bar{\epsilon}_c\) are introduced, but the text confusingly mentions "Detailed Proof" for these definitions, which is unnecessary. The purpose of these error terms in the larger narrative is not well-motivated.
*   **Section 3.3 (Conditional Control Term):** Lemma 2 isolates the conditional control term under classifier-free guidance. The proof (Appendix E) appears correct, but the lemma's reliance on the specific classifier-free guidance formulation should be explicitly stated in the main text. Its connection to the overall analysis is tenuous.
*   **Section 3.4 (Condition Refinement via Patch Denoising):** Proposition 1 is vague ("leads to improved conditional generation quality"). The proof (Appendix F) analyzes a special Gaussian case to show gradient norm decay under strong assumptions (e.g., small variance). The claim that this generalizes to the full diffusion setting is not convincingly argued. The proposition does not directly prove improved generation quality.
*   **Section 3.5 (Autoregressive Modeling Refines Condition):** Theorem 2 is a key result, showing exponential decay of the conditional score gradient norm under autoregressive iteration. The proof (Appendix G) is highly technical and relies on non-trivial assumptions (Assumption 4: bounded AR coefficients, bounded second derivatives of the conditional density). While the proof seems sound in its own framework, the practical validity of these assumptions for real image data and deep networks is unclear. Furthermore, the theorem's implication for final image quality is not directly established.

**Overall, the theoretical section is dense and contains several interesting but loosely connected results. The flow from analysis to algorithm design is not well-articulated. The practical relevance of some results (e.g., the Gaussian case analysis) is questionable. The theoretical claims are stronger than the provided justifications.**

### Autoregressive Condition Optimization (Section 4)
This section introduces the OT-based refinement method.

*   **Section 4.1 (Condition Inconsistency):** Lemma 6 formally defines "condition inconsistency" via subspace projection, arguing that autoregressive conditions accumulate extraneous information. The concept is intuitively clear, but the notion of a "minimal sufficient information subspace" \(I_i^*\) is abstract and not operationalized for implementation.
*   **Section 4.2 (OT Refinement via Wasserstein Gradient Flow):** Proposition 2 and Theorem 3 propose formulating refinement as a Wasserstein gradient flow and claim geometric convergence to the ideal condition distribution. This is a strong theoretical guarantee. However, the transition from theory to practice is severely lacking. The energy functional involves an inverse process \(T^{-1}\), which is **not defined** in any concrete way. How is \(T^{-1}\) obtained or approximated? The algorithm (Algorithm 1, Appendix L) uses \(T^{-1}\) as if it is a given function, making the proposed method incomplete. Furthermore, solving a regularized OT problem via Sinkhorn iterations at each step is computationally expensive; the paper does not address this cost or its feasibility for large-scale generation.

### Experiments (Section 5)
The experiments demonstrate improved metrics but lack critical analysis and ablation.

*   **Section 5.1 & 5.2 (Settings & Main Results):** Results in Table 1 show the proposed method achieves superior FID/IS on ImageNet 256x256. However, **the model size and compute budget for "Our method" in Table 1 are not specified**, making comparisons with baselines like DiT-XL/2 unfair. It is likely the 943M model from Table 2, but this must be stated clearly.
*   **Section 5.3 (Scalability):** Tables 2 and 3 show consistent gains over the strong baseline MAR across model sizes and resolutions. This is a positive result, though improvements are modest.
*   **Section 5.4 (Condition Errors Analysis):** Figure 3 provides qualitative evidence of lower noise and higher SNR during denoising. However, this analysis is only against one baseline and does not isolate the contribution of the OT refinement.
*   **Major Omissions:** There is **no ablation study** to quantify the impact of the OT refinement module versus the base autoregressive diffusion architecture. The computational overhead (training/inference time, memory) of the Sinkhorn-based refinement is **not reported**. The description of how the OT refinement is integrated into the training/generation pipeline is insufficient; readers must decipher Appendix L.

### Writing & Clarity
The paper is generally well-structured but suffers from dense, poorly motivated theory. Key symbols are sometimes defined only in proofs (e.g., constant \(m\) in Theorem 2). The connection between the theoretical results in Section 3 and the proposed algorithm in Section 4 is weak. The algorithm description is relegated to an appendix, making the method difficult to understand from the main text.

### Limitations & Broader Impact
Appendix B acknowledges the lack of experiments on "large-scale models" due to compute constraints, which is a significant limitation given the method's complexity. There is no discussion of broader impact, societal or ethical considerations, which is a minor shortfall for ICLR.

## Overall Assessment

This paper presents two main contributions: (1) a theoretical analysis of condition error dynamics in autoregressive diffusion models, and (2) an OT-based condition refinement method with convergence guarantees. The theoretical analysis, while technically involved, is somewhat fragmented and its assumptions are not clearly justified for practical settings. The proposed OT method is theoretically interesting but its description is incomplete (the critical inverse process \(T^{-1}\) is undefined) and its practicality is unproven due to unaddressed computational costs. Experimentally, the method shows improved metrics over strong baselines, but the comparisons are potentially unfair due to unspecified model sizes, and the core contribution of the OT refinement is not ablated.

For ICLR, the paper falls short in bridging theory and practice. The theoretical claims are not tightly coupled to the algorithm, and the algorithm's implementation is underspecified. The experimental validation, while positive, is insufficient to convince the reader that the OT refinement is both effective and feasible. **Major revisions are required to clarify the method's implementation, provide fair and ablative experiments, and realistically assess computational trade-offs.** Without these, the paper's contribution is primarily theoretical and its practical impact remains uncertain.

**Recommendation: Weak Reject (Borderline).** The paper has promising ideas but currently does not meet ICLR's standards for a clear, reproducible, and well-evaluated contribution. With significant revisions addressing the concerns above, it could be reconsidered.

# Neutral Reviewer
## Balanced Review

### Summary
This paper presents a theoretical analysis of autoregressive image generation models with diffusion loss, demonstrating that patch denoising can mitigate condition errors and stabilize the condition distribution. The authors further identify "condition inconsistency" in autoregressive generation and propose a novel condition refinement method based on Optimal Transport (OT), formulated as a Wasserstein Gradient Flow. Experimental results on ImageNet show improved performance over diffusion and autoregressive baselines.

### Strengths
1. **Substantial Theoretical Contributions**: The paper provides rigorous theoretical proofs, including Theorem 2 on the exponential decay of the condition's influence and Theorem 3 on the convergence of the Wasserstein Gradient Flow. These are novel insights into the behavior of autoregressive models with diffusion loss.
2. **Novel Method for Condition Refinement**: The introduction of an OT-based condition refinement to address inconsistency is innovative. The formulation as a Wasserstein Gradient Flow is theoretically well-founded and offers a principled approach to improving autoregressive generation.
3. **Comprehensive Experimental Validation**: The method is evaluated on ImageNet at multiple resolutions (256×256 and 512×512) and across model sizes, consistently outperforming strong baselines like MAR, LDM-4, and DiT-XL/2. Metrics (FID, IS, Precision, Recall) show clear improvements.
4. **Clarity in Theoretical Exposition**: Despite the complexity, the paper is well-structured with clear definitions, assumptions, and proof sketches. The appendix provides detailed derivations, and a notation table (Appendix O) aids readability.

### Weaknesses
1. **Insufficient Implementation Details for Reproducibility**: Key training details (hyperparameters, architectures beyond mentioning GPT-XL and U-ViT-H/2-G, compute resources) are missing. The algorithm in Appendix L lacks specific parameter values (e.g., λ, ϵ, learning rates), hindering reproduction.
2. **Limited Empirical Scope**: Experiments are confined to ImageNet. Evaluation on additional datasets (e.g., LSUN, COCO) would strengthen claims about generalization and real-world applicability.
3. **Incomplete Comparison to State-of-the-Art**: While baselines include strong models, recent SOTA autoregressive (e.g., MaskGIT, later VAR versions) and diffusion models (e.g., Stable Diffusion 3) are absent, making it difficult to assess the true advancement.
4. **Theoretical Assumptions Not Fully Justified**: Assumptions (e.g., Gaussian noise, small variance, bounded derivatives) are standard but not discussed in the context of real data deviations. The practical impact of these assumptions is unclear.
5. **Abstract Definition of Condition Inconsistency**: The concept of "condition inconsistency" is introduced theoretically but lacks intuitive explanation or visualization, making the problem and solution less accessible.

### Novelty & Significance
- **Novelty**: The theoretical analysis linking autoregressive patch denoising to condition error reduction and the OT-based refinement method are novel contributions. The convergence proof for the Wasserstein Gradient Flow in this context is also new.
- **Significance**: The work advances the theoretical understanding of autoregressive image generation and offers a practical method to improve generation quality. It has the potential to influence future research in conditional generative modeling.

### Suggestions for Improvement
1. **Add Detailed Implementation Section**: Include full architectural details, hyperparameter values, training schedules, and computational requirements in the main paper or a separate appendix to ensure reproducibility.
2. **Expand Experiments to More Datasets**: Test on additional benchmarks (e.g., LSUN, COCO) to demonstrate generalization and provide more convincing evidence of the method's effectiveness.
3. **Include More Recent Baselines**: Compare with the latest SOTA autoregressive and diffusion models to better position the contribution within the current landscape.
4. **Provide Qualitative Visualization of Condition Inconsistency**: Show example images or feature visualizations to illustrate the inconsistency problem and how refinement addresses it, making the motivation more concrete.
5. **Discuss Practical Implications of Theoretical Assumptions**: Add a paragraph discussing how the assumptions might hold or break in practice and any potential limitations this introduces.
6. **Conduct Ablation Studies**: Isolate the impact of the OT refinement module versus the autoregressive framework to clarify the contribution of each component.
7. **Address Large-Scale Model Limitations**: While computational constraints are acknowledged, discuss the expected scalability and potential challenges when applying the method to very large models (e.g., billion-parameter models).

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Compare to modern autoregressive baselines.** The paper only compares to MAR (Li et al., 2024a) and a few diffusion models. To claim superiority in "autoregressive image generation," comparisons to strong recent methods like LlamaGen (Sun et al., 2024a), VAR (Tian et al., 2024), and ImageFolder (Li et al., 2024b) are essential. Without these, the claim of superiority is not convincing.
2. **Ablation study of the OT refinement module.** The core proposed contribution is the OT-based condition refinement. There is no experiment showing the performance gain from adding this module to a base autoregressive model with diffusion loss. This missing ablation directly undermines the claim that OT refinement is effective.
3. **Condition error measurement.** The theory claims autoregressive refinement reduces condition error exponentially. No experiment quantifies this error (e.g., distance to an ideal condition or variance of condition distribution across iterations) to validate the theoretical decay.
4. **High-resolution and cross-dataset validation.** The scalability analysis is limited to ImageNet 512x512. To demonstrate robustness, tests on other datasets (e.g., COCO, FFHQ) and higher resolutions (e.g., 1024x1024) are needed to see if benefits hold.

### Deeper Analysis Needed (top 3-5 only)
1. **Direct analysis of condition distribution convergence.** The paper claims the OT refinement ensures convergence to an ideal condition distribution via Wasserstein gradient flow. There is no empirical analysis (e.g., tracking Wasserstein distance during training or refinement steps) to verify this convergence. Without it, the theoretical guarantee is unsupported.
2. **Breakdown of FID/IS improvements.** It is unclear whether gains come from better condition refinement, the autoregressive framework, or simply larger model size. An analysis correlating condition consistency metrics (e.g., patch coherence) with final FID/IS would link theory to practice.
3. **Sensitivity analysis of OT hyperparameters.** The method introduces hyperparameters (λ, ε, sinkhorn steps). No analysis shows how sensitive results are to these choices, which is critical for reproducibility and understanding the method's stability.

### Visualizations & Case Studies
1. **Visual trajectory of condition refinement.** Show how a condition vector evolves across autoregressive steps and OT refinement (e.g., via PCA/t-SNE). This would visually demonstrate the "refinement" and convergence claimed.
2. **Qualitative comparison of failures.** Show side-by-side generations where the baseline (MAR) fails and our method succeeds, and vice versa. This would expose the specific improvements (e.g., in texture, structure) and remaining weaknesses.
3. **Visualization of "condition inconsistency".** Illustrate the extraneous information accumulation (e.g., by visualizing the component η_i in the condition space) to make the problem tangible and show how OT reduces it.

### Obvious Next Steps
1. **Isolate the OT refinement contribution.** A simple but necessary experiment: take the same autoregressive base model and compare "with OT" vs. "without OT" (or with a simpler refinement like MLP). This should have been in the paper to prove the module's utility.
2. **Measure computational overhead.** The OT refinement using Sinkhorn iterations adds cost. Reporting the additional training/inference time and memory compared to baselines is essential for assessing practicality.
3. **Provide pseudo-code for the full algorithm in the main text.** The algorithm is buried in the appendix. For clarity and reproducibility, a concise version should be in the main paper, outlining how autoregressive generation, diffusion denoising, and OT refinement are interleaved.

# Final Consolidated Review
## Summary
This paper presents a theoretical analysis of condition errors in autoregressive image generation with diffusion loss, demonstrating that patch denoising mitigates errors and refines conditions. It proposes a novel Optimal Transport-based condition refinement method formulated as a Wasserstein Gradient Flow with convergence guarantees. Experiments on ImageNet show improved performance over diffusion and autoregressive baselines.

## Strengths
- Provides a rigorous theoretical proof that the condition's influence on the outcome decays exponentially as autoregressive iteration progresses (Theorem 2), offering novel insight into error reduction.
- Introduces an innovative Optimal Transport-based condition refinement method with a Wasserstein Gradient Flow formulation and a proof of geometric convergence to the ideal condition distribution (Theorem 3).
- Demonstrates consistent performance gains in FID, IS, Precision, and Recall over strong baselines (MAR, DiT-XL/2) across multiple model sizes and resolutions on ImageNet, validating scalability.

## Weaknesses
- **The OT refinement method is incomplete and underspecified** — the inverse process \(T^{-1}\) is used in the algorithm (Appendix L) but never concretely defined or implemented, making the method irreproducible and raising doubts about its feasibility.
- **No ablation study to isolate the OT module's contribution** — without comparing the base autoregressive model with and without OT refinement, it is unclear whether gains stem from the novel refinement or the underlying architecture.
- **Theoretical assumptions lack practical justification** — key assumptions (e.g., bounded second derivatives of conditional density, Gaussian case in Proposition 1) are not discussed in the context of real image data, limiting the relevance of the proofs to practical settings.
- **Unfair experimental comparisons** — Table 1 does not specify the model size for "Our method," while baselines like DiT-XL/2 have known sizes; this omission risks misleading comparisons and hinders reproducibility.
- **Computational overhead ignored** — the Sinkhorn-based OT refinement adds significant cost per iteration, but no analysis of training/inference time or memory is provided, undermining the method's practicality.
- **Weak connection between theory and algorithm** — the theoretical analysis in Section 3 is loosely linked to the proposed OT refinement in Section 4, reducing clarity and coherence of the overall contribution.

## Nice-to-Haves
- Expand experiments to additional datasets (e.g., COCO, FFHQ) to demonstrate generalization beyond ImageNet.
- Include sensitivity analysis for OT hyperparameters (e.g., \(\lambda\), \(\epsilon\)) to guide implementation and show robustness.
- Provide visualizations of condition refinement trajectories (e.g., via t-SNE) to illustrate the convergence claimed theoretically.
- Compare to more recent autoregressive baselines (e.g., LlamaGen, VAR) for broader context, though MAR is a relevant baseline for diffusion loss methods.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Strength**: "Clarity in theoretical exposition" — the theory is dense and poorly connected to the algorithm, so this is not a consistent strength.
- **Weakness**: "Abstract does not mention limitations" — this is a formatting nitpick, not a substantive flaw.
- **Weakness**: "Theorem 1 is known or straightforward" — it is correctly used as a foundation, not claimed as novel.
- **Weakness**: "Proposition 1 is vague" — it is supported by a proof under specific assumptions, though the assumptions are a weakness.
- **Weakness**: "Missing broader impact discussion" — this is not a core requirement for the paper's technical contributions.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Clearly define or approximate the inverse process \(T^{-1}\) in the OT refinement method (e.g., as a learned network or explicit function) to make the algorithm fully specified and reproducible.
- Conduct an ablation study comparing the base autoregressive model with and without the OT refinement module to quantify its isolated impact.
- Specify model sizes for all entries in Table 1 and report the computational overhead (time, memory) of the OT refinement during training and inference.
- Empirically measure condition error (e.g., via Wasserstein distance to a reference distribution) across autoregressive steps to validate the theoretical decay claimed in Theorem 2.

# Actual Human Scores
Individual reviewer scores: [6.0, 6.0, 6.0, 6.0]
Average score: 6.0
Binary outcome: Accept

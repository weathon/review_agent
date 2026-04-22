# Condition Errors Refinement in Autoregressive Image Generation with Diffusion Loss

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Recent studies have explored autoregressive models for image generation, with promising results, and have combined diffusion models with autoregressive frameworks to optimize image generation via diffusion losses. In this study, we present a theoretical analysis of diffusion and autoregressive models with diffusion loss, highlighting the latter's advantages. We present a theoretical comparison of conditional diffusion and autoregressive diffusion with diffusion loss, demonstrating that patch denoising optimization in autoregressive models effectively mitigates condition errors and leads to a stable condition distribution. Our analysis also reveals that autoregressive condition generation refines the condition, causing the condition error influence to decay exponentially. In addition, we introduce a novel condition refinement approach based on Optimal Transport (OT) theory to address ``condition inconsistency''. We theoretically demonstrate that formulating condition refinement as a Wasserstein Gradient Flow ensures convergence toward the ideal condition distribution, effectively mitigating condition inconsistency. Experiments demonstrate the superiority of our method over diffusion and autoregressive models with diffusion loss methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper theoretically and empirically investigates how autoregressive image generators incorporating diffusion loss can mitigate conditional inconsistency during generation. It provides a rigorous analysis comparing conditional diffusion models and AR diffusion with diffusion loss, proving that autoregressive patch denoising refines condition distributions and that the influence of condition errors decays exponentially during iteration. To address residual condition inconsistency, the authors introduce a condition refinement method based on Optimal Transport formulated as a Wasserstein Gradient Flow, proving convergence toward the ideal condition distribution. Experiments on ImageNet show superior FID and IS scores over existing diffusion and AR baselines, supporting their theoretical findings.

### Strengths
1. The paper presents a solid theoretical framework that connects autoregressive modeling, diffusion loss, and conditional refinement through explicit mathematical proofs and lemmas.

2. It introduces a novel condition refinement approach using Optimal Transport and Wasserstein Gradient Flow, offering a principled solution to condition inconsistency.

3. The quantitative performance on ImageNet is competitive or superior to major diffusion and AR baselines, confirming practical benefits of the proposed method.

### Weaknesses
1. The experimental scope is limited: evaluations are conducted only on ImageNet 256×256 and with moderate-scale models, without testing scalability to larger or multimodal setups.

2. Despite extensive theory, the method’s implementation details (e.g., computational cost of OT refinement, convergence sensitivity to hyper-parameters) are underexplained.

3. Some notation and theoretical transitions are difficult to follow and may reduce accessibility for non-mathematical readers.

4. The impact of the OT regularization term on actual generation diversity and efficiency is not sufficiently analyzed, only FID/IS metrics are shown.

### Questions
Same as weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses an important and quite novel problem in autoregressive (AR) image generation. The authors point out that the conditional errors accumulated from previously generated patches may cause inconsistency, and they theoretically analyze how the patch denoising process in AR models can alleviate such conditional errors. Moreover, they prove that the influence of conditional error decays exponentially with iterations. The proposed conditional optimization method based on Optimal Transport (OT) and formulated as Wasserstein Gradient Flow (WGF) is elegant, and the paper shows it can converge toward the ideal conditional distribution.

### Strengths
（1）The theoretical part is quite impressive. Especially Theorem 2 gives a deep understanding about how the conditional influence (the gradient norm) exponentially decays, which brings new insights into the stability of AR generative models. The combination of OT and WGF for refining conditional distribution looks creative and convincing.

（2）The experimental results are strong. On ImageNet 256x256, the FID score reaches 1.31, which is very competitive compared with existing works.

### Weaknesses
（1） Algorithm 1 seems to describe a nested loop structure. I am a bit worry that the computation cost could be large, maybe even K times T slower than the standard AR model. Some clarification or runtime comparison could be helpful.

(2) It is a bit unclear what “Baseline (CDM)” (FID 3.26) and “Baseline” (FID 2.02) exactly mean. Does “Baseline” refer to the AR model without OT refinement? Since the paper’s best FID (1.31) is quite good, some ablation study would help to show how much improvement really comes from the proposed OT method, rather than from the backbone MAR model itself.

（3）The experiments are mainly done on ImageNet. It would be nice if the authors could test at least one more dataset or show more model comparison to make the results more solid.

### Questions
Mainly about the points above, especially clarification of baselines and computational complexity.

Overall, I think this paper has strong theoretical contribution and interesting methodology. With a bit more experiment and clarification, it could be a quite nice work.

### Soundness
3

### Presentation
3

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
This paper focuses on condition error issues in autoregressive image generation with diffusion loss. It first conducts a theoretical analysis of diffusion models and autoregressive models with diffusion loss, demonstrating that patch denoising optimization in autoregressive models can effectively mitigate condition errors and form a stable condition distribution, while autoregressive condition generation refines conditions to make condition error influence decay exponentially. It further proposes a OT-Based refinement approach, theoretically proving that formulating this refinement as a Wasserstein Gradient Flow ensures convergence toward the ideal condition distribution. Experiments on ImageNet show the superiority of the proposed method over existing diffusion and autoregressive models with diffusion loss methods.

### Strengths
1. The theoretical framework for autoregressive image modeling with diffusion loss is both sound and novel. The theory is rigorous and clearly connects diffusion loss to autoregressive conditional modeling. The mathematical exposition is clear and technically solid.

2. The proposed ideas of autoregressive patch-wise denoising and OT-based condition refinement are conceptually well-motivated.

3. The paper provides a rigorous theoretical analysis demonstrating that the patch-wise denoising optimization in autoregressive models effectively mitigates condition errors.

4. It further establishes a mathematically consistent framework linking energy optimization to Optimal Transport (OT) regularization, offering a clear and unified theoretical explanation of the condition error phenomenon.

### Weaknesses
1. The empirical validation does not fully match the strength of the theory. The main theory predicts (i) conditional score norm decays exponentially as AR iterations progress and (ii) OT refinement decreases condition inconsistency (Sinkhorn divergence) monotonically. The paper lacks direct empirical plots that verify these claims.

2. Lack of important experiments at higher resolutions, such as the ImageNet 512 × 512 experiment.

3. The comparison against stronger and more recent baselines (after 2025) is missing, weakening the empirical significance of the claims.

4. A figure of the framework is needed to outline the designed methods, including autoregressive patch denoising and OT-based methods.

### Questions
1. See the weaknesses section for detail.

2. The proposed theory and method are developed only in the context of autoregressive models with diffusion loss. It remains unclear whether the approach has broader applicability beyond this specific model class.

3. In Table 1, the distinction between the two baselines is unclear, and it is not stated which baseline “Ours” is built upon. The difference between Ours and Ours (MAR) is also ambiguous. Without proper citations or explanations, the table is confusing to readers.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper conducts a thorough theoretical analysis of autoregressive models with diffusion loss, contrasting them with standard conditional diffusion models. The central thesis is that the patch-by-patch denoising optimization in autoregressive frameworks serves as an effective mechanism for refining the guiding condition, leading to a more stable condition distribution and mitigating errors. This work provides a solid theoretical foundation for understanding and improving conditional autoregressive generation by framing condition refinement as a distribution-level optimization problem.

### Strengths
1. The paper's primary strength lies in its rigorous theoretical contributions. The formalization of the condition refinement process as a Wasserstein Gradient Flow is both elegant and novel, providing a principled guarantee of convergence that is often missing in heuristic-based approaches. The detailed lemmas and theorems build a convincing mathematical argument.
2. The paper is logically well-organized. It seamlessly transitions from a comparative analysis of diffusion models, to the definition of conditional error, to the proposal of the OT-based solution, and finally to empirical validation. The argument is self-contained and easy to follow for readers with the requisite background.

### Weaknesses
1. Limited Experimental Scale: As the authors acknowledge in Appendix B, the experiments are confined to the 256x256 resolution on ImageNet. While this is a standard benchmark, state-of-the-art generative modeling research is increasingly focused on higher resolutions and larger models. The absence of such experiments may leave questions about the method's scalability and generalizability.
2. Readability and Accessibility: The theoretical sections are dense and assume significant familiarity with advanced mathematical concepts like Optimal Transport and Wasserstein Gradient Flows. While precise, this may limit the paper's accessibility. A more intuitive explanation or a high-level overview of why OT is the right tool for this problem could broaden the paper's impact.
3. Lack of Ablation Studies: The paper presents the final model's performance but would benefit from ablation studies that isolate the impact of the core contribution—the OT-based refinement. For instance, an experiment comparing the full model against a version without the WGF optimization would more clearly quantify the gains from this specific module.

### Questions
1. Could you provide a more intuitive, high-level explanation at the beginning of Section 4 to bridge the gap between "condition inconsistency" and the OT framework?
2. What is the computational overhead of the proposed OT refinement step? How does it affect the overall image generation latency compared to the baseline MAR model?
3. The concept of "extraneous information" is central to your motivation. Is it possible to visualize this phenomenon? For instance, by projecting the condition vectors (c_i) before and after refinement into a 2D space, or by showing how the generated patch changes with and without refinement at an intermediate step. This would make the problem much more tangible.
4. The OT optimization introduces several hyperparameters (e.g., λ, ηk). How sensitive is the model's performance to the choice of these parameters? Is there a robust range for their values?

### Soundness
3

### Presentation
3

### Contribution
2

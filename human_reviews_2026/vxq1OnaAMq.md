# Gauge Flow Matching: Efficient Constrained Generative Modeling over General Convex Set and Beyond

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 8, 2, 6

## Abstract
Generative models, particularly diffusion and flow-matching approaches, have achieved remarkable success across diverse domains, including image synthesis and robotic planning. However, a fundamental challenge persists: ensuring generated samples strictly satisfy problem-specific constraints — a crucial requirement for physics-informed problems, safety-critical applications, watermark embedding, etc. Existing approaches, such as mirror maps and reflection methods, either have limited applicable constraint sets or introduce significant computational overhead. In this paper, we develop gauge flow matching (GFM), a simple yet efficient framework for constrained generative modeling. Our GFM approach introduces a novel bijective gauge mapping to transform generation over arbitrary compact convex sets into an equivalent process over the unit ball, which allows low-complexity feasibility-ensuring operations such as reflection or projection. The generated samples are then mapped back to the original domain for output. We prove that our GFM framework guarantees strict constraint satisfaction, with low generation complexity and bounded distribution approximation errors. We further extend our GFM framework to two non-convex settings, namely, star-convex and geodesic-convex sets. Extensive experiments demonstrate that GFM outperforms existing methods in both generation speed and quality across multiple benchmarks, including synthetic data, time series, and image generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses a critically important and timely challenge of constrained generative modeling by proposing a novel idea called gauge flow matching (GFM). The underlying idea of GFM is to introduce a bijective gauge mapping to transform a complex constrained generative modeling problem into an equivalent modeling problem over a simple unit ball domain. The paper presents rigorous theoretical validation/analysis on the strict constraint guarantee and a lower computational complexity on not only convex feasibility regions but also certain types of non-convex feasibility regions. It also presents empirical validation of the GFM idea on various benchmarks, in comparison to other state-of-the-art constrained generation methods. The proposed idea is technically sound, and the scientific breakthrough is clear. The presentation and the style of writing are also effective, maintaining a good balance between high-level intuitive explanation and low-level mathematical details. As such, I believe this paper deserves presentation at ICLR.

### Strengths
Although the idea of bijective mappings to transform constrained domains into a new domain that has more favorable analytical properties existed before this paper, it was still understudied, and the previous work was largely limited to some limited problems and/or theoretical guarantees. Hence, I find the idea in this paper sufficiently original and of significance, presenting creative combinations of existing ideas (e.g., guage mapping + bijective mapping to simplify constrained generation), new scientific knowledge (e.g., theoretical guarantees, complexity analysis, benchmark test results), and such.

I also appreciate the style and quality of writing. While engaging a broader group of audience effectively with a clear and intuitive explanation of the method and the rationale behind the method, the paper still presents enough technical details with mathematical rigor.

### Weaknesses
First and foremost, I would've appreciated more experiments on challenging generative tasks. I understand that the authors did their best to cover the spectrum of discussions from theoretical to empirical validations, the real-world generative tasks (i.e., robotic manipulability and relaxed combinatorial problems), but I still feel underwhelmed when it comes to the empirical validation portion of this work. Image generation, for example, would've been a great benchmark test, even if the scope of images was limited to a small, limited variety (e.g., microscopic images as opposed to the whole set of natural images). I understand such a task might be "too non-convex" and might be beyond the scope of this work, but I still remain curious to know if and how far the proposed GFM idea could push the limit. (Especially because the other parts of the works are very well done!)

Other than that, I only have some minor suggestions regarding mostly presentation, which the authors may or may not want to consider:
- I think Figure 1 captures the core idea of your work very well. Why don't you move it to an earlier page?
- Also, figure captions in general could be more elaborate and contain more details.
- Excessive use of bold faces and color highlights hurts readability in my opinion. As an example (out of other such cases), I don't see the need to emphasize the acronym "GFM" and the word "limitation" in the conclusion section by making them boldface. This tendency is found across the whole paper, which the authors are strongly encouraged to revisit and think through.

### Questions
I don't have questions.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Gauge Flow Matching, a constrained sampling approach that generalizes to arbitrary convex sets. The methodology builds on prior work for mirror mappings and reflected diffusion, originally proposed within diffusion frameworks, providing a generalization for flow matching with broader coverage of convex constraints. The experimental analysis on synthetic settings reports promising results for improving fidelity to the training distribution while improving runtime over existing constrained sampling approaches.

### Strengths
- **Motivation and Articulation:** The paper's motivation is well established and articulated. The writing is fairly easy to follow, and the methodology is communicated well.

- **Theoretical Support:** The theoretical grounding presented in the paper strengthens the claims made by the authors. The analysis appears to be sound, and the provided bounds contribute to the soundness of the paper.

- **Run-time Efficiency:** The efficiency arguments are an interesting contribution. While I believe the authors could make a better case for this by motivating it with time-sensitive, real-world experimental settings, I believe there are some relevant applications where this could be applied.

### Weaknesses
- **Limited Contribution:** The novelty of the method seems to be directly tied to the application of the existing gauge mapping [1] within flow matching models. The actual integration of this into the flow matching model is very closely tied to prior work on mirror maps and reflection based techniques. While the authors do point out that mirror maps do not generalize to arbitrary convex sets, it seems that beyond this mapping "substitution", the contribution is fairly limited.

- **Limited Empirical Evaluation:** While the experimental evaluation provides strong results, it relies solely on synthetic test cases with little to no real-world meaning. Furthermore, for many of these test cases, rejection sampling seems to be the better approach, as feasibility rates are higher than 90%, and samples have higher fidelity to the training set. Hence, the results would be much more compelling if they were applied to real-world settings where constrained sampling would be practically applied; this seems to be a missed opportunity, especially if such settings could provide a compelling case for the introduced efficiency. 

- **Selected Baselines:** It is surprising that the authors have not chosen to compare to mirror diffusion models [2] in any of their analysis. Given how closely the method overlaps with mirror mappings, the omission of this comparison makes it difficult to assess whether the gauge map is effective -- which is the core contribution of the work. Furthermore, this would be a valuable comparison to assess how different bijective mappings influence the generation quality. 

---

[1] Tabas, Daniel, and Baosen Zhang. "Computationally efficient safe reinforcement learning for power systems." 2022 American Control Conference (ACC). IEEE, 2022.

[2] Liu, Guan-Horng, et al. "Mirror diffusion models for constrained and watermarked generation." Advances in Neural Information Processing Systems 36 (2023): 42898-42917.

### Questions
- In Tables 3 and 5, the training time and inference times are lower than the "vanilla" models. Can the authors explain why this would be?

- Along similar lines as the previous question, the runtime for the reflection and projection algorithms (Tables 3 and 4) seem unexpectedly high. Can the authors speak to how these were implemented? Given the simple nature of these constraints, it would seem these operations could be more efficiently computed (e.g., an efficient closed-form projection operator).

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
GFM enforces hard constraints by bijective mapping any compact convex support set  "C"   to the unit ball B via a gauge map,training and generating inside B (with cheap boundary reflection) and mapping back, so samples are strictly feasible by construction. The gauge map is bi-Lipschitz, keeping distortion bounded and preserving distributional regularity when transferring training/sampling between "C"  and B. The framework extends to star-convex and geodesic-convex sets via appropriate generalized gauge maps. Experiments show 0% violations, strong distributional fit (e.g., low MMD), and faster generation than prior constrained methods.

### Strengths
1. The author proposes a generalized gauge mapping for constrained generation on any compact convex.The map Φ:B^2↔C is bijective and bi-Lipschitz with explicit bounds, which yields a clean distributional error transfer for any p-Wasserstein distance(Proposition 4.1),so training in B_2 preserves accuracy up to a bounded factor; feasibility is strict by construction.Extensions to star-convex and geodesic-convex sets (Appendix A) further broaden scope.

2. Computational efficiency:overall cost ≈ unconstrained models + small mapping overhead.Gauge values are closed-form for many sets (linear/quadratic/Second Order Cone/...); for general convex sets a 1D bisection along rays computes boundary intersections rapidly. The approach scales to high-D and complex constraints.

3. The empirical evaluation is convincing: 0% violations, low MMD (often outperforming projection/reflection/Metropolis baselines), and faster inference —showing constraints do not degrade sample quality.

4. The paper is clearly written:clear positioning (comparative table), intuitive figures for the gauge map, and precise propositions.

### Weaknesses
1. Sensitivity to interior point & data assumptions:Robustness still hinges on a good interior point, which can be non-trivial in high dimensions; moreover, requirement of the data distribution may limit real-world performance.

2. Although many cases admit closed-form or 1D bisection, per-sample gauge evaluation could dominate for very high-D or costly oracles (large LPs/implicit constraints),as shown in Figure 6.

3. Reproducibility:Many evaluations use programmatically generated target distributions tied to the constraints. While standard for this setting, the paper would be stronger with public tasks + released seeds/code, plus tests on real constrained domains (e.g., PSD cones, Birkhoff polytope, trajectory constraints).

### Questions
1. How do the authors envision extending GFM to more general non-convex domains beyond the star-convex and geodesic-convex cases?

2. The paper mentions that extending GFM to discrete generation via relaxation or embedding is a promising direction. Could the authors elaborate on how this might be achieved?

### Soundness
3

### Presentation
3

### Contribution
3

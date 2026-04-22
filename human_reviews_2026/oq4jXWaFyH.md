# The Natural Geometry of Code: Hyperbolic Representation Learning for Program Reasoning

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
State-of-the-art models for code representation, such as GraphCodeBERT, embed the hierarchical structure of source code into Euclidean space. This approach can lead to significant representation distortion, especially when embedding deep or highly branched hierarchies,limiting the models' ability to capture deep program semantics. We argue that the natural geometry for code is hyperbolic, as its exponential volume growth perfectly matches the tree-like structure of a code's Abstract Syntax Tree (AST), enabling low-distortion hierarchical embeddings. We introduce {HypeCodeNet}, a geometric deep learning framework that operates natively in hyperbolic space. Formulated in the numerically stable Lorentz model, its manifold-aware components include a hyperbolic embedding layer, a tangent space message-passing mechanism, and a geodesic-based attention module. On code clone detection, code completion, and link prediction, HypeCodeNet significantly outperforms existing Euclidean models, especially on tasks requiring deep structural understanding. Our work suggests that hyperbolic geometry offers a geometrically sound foundation for code representation, establishing hyperbolic geometry as a key to unlocking the structured semantics of code.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper explores the hypothesis that the hierarchical structure of source code, typically represented by Abstract Syntax Trees (ASTs), can be more effectively modeled in hyperbolic space than in traditional Euclidean geometry.
Based on this intuition, the authors introduce HypeCodeNet, a hyperbolic graph neural network constructed in the Lorentz model, integrating manifold-aware components such as hyperbolic embeddings, tangent-space message passing, and geodesic attention.
The model’s performance is evaluated across multiple code understanding benchmarks, including BigCloneBench and POJ-104 for code clone detection, as well as CodeXGLUE for code completion and GitHub Java Call Graphs for link prediction.
In these experiments, HypeCodeNet is compared against nine baseline models and results show that HypeCodeNet demonstrates performance comparable to or exceeding that of established baselines across multiple benchmarks, particularly on tasks requiring structural reasoning.

### Strengths
1. Well-formulated hyperbolic representation architecture
The proposed model presents a mathematically well-grounded extension of GNNs into hyperbolic space, specifically tailored to capture hierarchical tree structures.
Unlike prior works that primarily relied on exponential mapping, this paper explicitly defines the logarithmic map and integrates it into a novel geodesic attention mechanism, providing both theoretical and empirical justification for its design.

2. Strong empirical results across multiple benchmarks
The model is comprehensively evaluated against nine baselines, spanning sequence-based, graph-based, and hybrid architectures, across three tasks and four datasets.
HypeCodeNet consistently surpasses or matches the strongest baselines, demonstrating its robustness and generality across code understanding scenarios.

3. Rich analytical evaluation
Through extensive ablation studies, the authors show that the proposed structure performs effectively even with extremely low-dimensional embeddings (e.g., 32 dimensions), and can outperform GraphCodeBERT with significantly fewer parameters.
This suggests that hyperbolic geometry provides an efficient inductive bias for hierarchical data.

### Weaknesses
1. Limited input flexibility due to reliance on AST parsing
The model requires input to be parsed into a valid AST graph.
Consequently, it cannot directly process unstructured code snippets, natural language prompts, or free-form textual descriptions.
This constraint implies that the method may need additional components to handle real-world data where parsing is incomplete or ambiguous, limiting the model’s applicability in broader scenarios.

2. Relatively slow training and inference
As reported in Appendix J, despite having fewer parameters, HypeCodeNet involves multiple nonlinear hyperbolic operations and iterative computations.
This results in $1.5-2\times$ slower training and approximately 20% longer inference time compared to baselines.
While this trade-off is understandable given the geometric complexity, it reduces throughput and may hinder deployment in latency-sensitive environments.

3. Lack of theoretical discussion on gradient convergence
The paper empirically demonstrates stable training and overall convergence, supported by techniques such as Riemannian gradient clipping and curvature annealing.
However, it does not provide a formal proof of convergence, nor does it include a theoretical analysis of convergence behavior. Also, the convergence bounds of the iterative Fr\’echet mean computation remain unspecified.
While the empirical results indicate reliable convergence in practice, the lack of theoretical guarantees leaves the model’s stability only experimentally supported and may limit its applicability in more rigorous settings.

### Questions
**Points to justify**

I do not consider those weaknesses to significantly diminish the overall contribution of the paper.
However, I believe the following points require further justification and clarification.
If the authors can adequately address the questions raised below, I am willing to reconsider my evaluation. 

1) As “reasoning” has taken on a different connotation in recent LLM literature, the term “Program Reasoning” in the title may be somewhat misleading. It would be clearer to emphasize that the model focuses on encoder-based code comprehension, for example by adopting a title such as “Hyperbolic Representation Learning for Encoder-based Code Comprehension.”

2) While the proposed model makes noteworthy efforts to ensure fair comparison, the baseline models and target tasks used are largely outdated, originating from 2022 or earlier. Moreover, recent advances in code understanding have been increasingly driven by large language models (LLMs), most of which follow a decoder-only paradigm, fundamentally differing from the encoder-based architecture proposed in this paper. Therefore, a discussion on the methodological timeliness, empirical relevance, and potential extensibility of the proposed approach is necessary to position the work within the modern landscape.

3) Incorporating CodeT5+ [1], the successor to CodeT5, as an additional baseline would strengthen the experimental comparison and provide a more up-to-date benchmark context.

4) Regarding the Code Completion task, it appears that the evaluation in this paper functions more as a Cloze Test, since it involves predicting a few masked tokens rather than performing true line-level completion. In fact, CodeXGLUE also treats this as a Cloze Test when evaluating encoder-only models such as CodeBERT, while Code Completion is evaluated using decoder-based models like CodeGPT. If there is a specific reason for evaluating under the “Code Completion” split of CodeXGLUE despite this distinction, please provide further justification. Regardless of this reasoning, aligning the task naming with the corresponding dataset would improve clarity and consistency.

5) The geodesic distance appears to be incorrectly defined. For a vector $u \in \mathcal{L}_c^d$, $d_c(u,u)=\frac{1}{(-c)^{\frac{1}{2}}}\arcosh(-c<u,u>_L)=\frac{1}{(-c)^{\frac{1}{2}}}\arcosh(-1)$. However, $\arcosh$ is defined only for $x \ge 1$. Also, $<p, log_p^c(h)>_{\mathcal{L}} \neq 0$. Thus, I suspect that $<x, x>_{\mathcal{L}}$ should be $-\frac{1}{c}$ in the definition of $\mathcal{L}_c^d$.

6) Including a proof, or at least an outline of the key steps, demonstrating that the exponential map is the inverse of the logarithmic map is necessary for mathematical completeness and clarity.

7) Equation (40), which approximates $-c<u,v>_L to 1+\frac{-c}{2} \left\| u_E - v_E \right\| ^2$, should be further justified.

**Suggestions**

1) As an upper bound reference, it would be valuable to additionally report the performance of a recent LLM-based baseline, even if only with a smaller model variant.

2) More comprehensive experimentation would strengthen the paper. Reporting additional results on the Defect Detection task from CodeXGLUE and on Cloze Test experiments across languages beyond Java and Python would provide stronger empirical support. If such experiments are infeasible, clarifying the reasons would be appreciated.

3) It would be helpful to include a diagram, perhaps in the appendix, illustrating how attention is computed within each layer.


4) The term "central node" is used but not clearly defined. If it does not carry a specific meaning, consider replacing it with a clearer phrase such as "node to update" or explicitly defining it for clarity.

5) It would be preferable to avoid the excessive use of bold text, as it may visually interfere with paragraph structure. Using italics or underlines for emphasis would improve readability and consistency.

6) Since the initial vector $z_v^E \in \mathbb{R}^d$ and $T_{o_c}\mathcal{L}_c^d \in \mathbb{R}^{d+1}$, the dimensionality increases. Therefore, the operation described should be referred to as an injection or extension rather than a projection.

7) On p.15, if $q_v^{(k)}=0$, there appears to be no need to include the bias term or the matrices $W_{Q,k}^{(l)}$ and $W_{K,k}^{(l)}$ in the computation. In this case, it would be equivalent to performing a learnable weighted sum over $m_{u \to v}^{(l)}$.

8) The section formatting is inconsistent. For instance, Section 3.2 introduces points as (1), (2), and (3), but later paragraphs switch to different numbering styles or omit numbering altogether, while the description of the output layer appears separately in Section 3.4. The numbering and formatting should be standardized for consistency, maintaining a uniform scheme or using sub-subsections if hierarchical structure is intended.

**Typo**
- p.3, line 126: "base point" should be written as ``base point''.
- Appendix A: The letter 'A' in the title should also be enclosed as `A'.
- Markdown-style emphases such as *text* or **text** appear in the appendix and should be removed for formal consistency.

[1] Wang, Y., Le, H., Gotmare, A., Bui, N., Li, J., & Hoi, S. (2023, December). CodeT5+: Open Code Large Language Models for Code Understanding and Generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 1069-1088).

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
4

### Summary
The paper proposes HypeCodeNet, a novel hyperbolic Graph Neural Network (GNN) that learns source code representations directly in hyperbolic space, rather than traditional Euclidean embeddings. Motivated by the hierarchical, tree-like nature of Abstract Syntax Trees (ASTs), the authors argue that hyperbolic geometry better captures program structure with low distortion.
HypeCodeNet operates in the Lorentz model for stability and integrates manifold-aware message passing, curvature annealing, and Riemannian optimization. Across three tasks: code clone detection, code completion, and function call link prediction, the model achieves strong gains over GraphCodeBERT, UniXcoder, and CodeFORMER. Ablations further confirm that performance improvements stem mainly from the hyperbolic geometry rather than architectural tweaks.

### Strengths
Originality (Novel geometric framing):
The paper introduces a foundational geometric shift in code representation, from Euclidean to hyperbolic space, supported by clear theoretical intuition. It’s the first to present an end-to-end hyperbolic framework for code reasoning, filling a notable gap in the literature.

Quality (Sound formulation & strong empirical results):
The proposed model is mathematically rigorous, leveraging Lorentz manifolds, log/exp maps, Riemannian Adam, and curvature annealing for stability. Results across multiple benchmarks (BigCloneBench, CodeXGLUE, GitHub Java Corpus) show consistent SOTA performance, with up to 3–5% improvement over strong baselines. Ablations convincingly isolate geometry as the key performance driver.

Clarity & Significance:
The paper is well-organized, with detailed geometric explanations and intuitive visualizations (e.g., Figure 1). Its results suggest non-Euclidean geometry may be a next paradigm for program representation, opening a new research direction bridging geometric deep learning and code understanding.

### Weaknesses
Reproducibility & implementation accessibility:
The paper does not mention public release of code or datasets, and implementation details (e.g., curvature annealing schedule, manifold dimensionality tuning) are deferred to appendices. Reproducibility may be difficult without explicit scripts or pretrained models.

Limited scope of benchmarks:
Although the model excels on line-level code completion and clone/link tasks, all evaluations remain static-structure-centric. The framework isn’t tested on dynamic or generation tasks (e.g., code repair, test generation), which would test whether hyperbolic embeddings generalize beyond AST reasoning.

### Questions
Curvature tuning:
How sensitive is model performance to the final curvature value? Would a mixed-curvature or adaptive manifold (e.g., product manifolds combining Euclidean + hyperbolic subspaces) perform better?

Generalization beyond AST-based tasks:
Can HypeCodeNet generalize to non-AST graph structures, such as interprocedural dependency graphs or text-conditioned code generation? Have you explored transfer learning between programming languages?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces HypeCodeNet, a geometric deep learning framework for code representation learning that operates natively in hyperbolic space, formulated under the Lorentz model.
The authors argue that source code’s Abstract Syntax Tree (AST) inherently possesses hierarchical and tree-like structures that are better represented in negatively curved manifolds than in Euclidean space.
HypeCodeNet integrates manifold-aware components — a hyperbolic embedding layer, a tangent-space message-passing mechanism, and a geodesic-based attention module — trained with Riemannian optimization and curvature annealing.
Across three standard benchmarks (clone detection, code completion, link prediction), it consistently outperforms strong Euclidean baselines such as CodeBERT, GraphCodeBERT, and CodeFORMER, supporting the claim that hyperbolic geometry aligns more naturally with program hierarchies.

### Strengths
1.	Strong conceptual motivation grounded in geometric theory
The paper convincingly argues that the exponential volume growth of hyperbolic space matches the hierarchical expansion of ASTs. The “distortion” argument is well supported, referencing Bourgain’s theorem and previous work on low-distortion tree embeddings.
2.	Technically principled formulation
By adopting the Lorentz model instead of the unstable Poincaré ball, the authors ensure both numerical stability and Riemannian differentiability, enabling a deep stack of manifold layers with standard GPU parallelization.
3.	Well-designed architecture bridging geometry and semantics
The “log–aggregate–exp” message-passing paradigm and geodesic-aware attention are elegant and grounded in geometric consistency. The method preserves curvature constraints while incorporating multi-head attention and layer normalization — non-trivial achievements in hyperbolic neural design.

### Weaknesses
Theoretical insufficiency in proving “naturalness” of hyperbolic geometry.
The core claim — that “hyperbolic geometry is the natural geometry of code” — remains conceptually persuasive but not theoretically rigorous. No formal quantification of distortion or curvature–hierarchy correlation (e.g., tree embedding distortion bounds). Missing mathematical analysis of curvature c → embedding fidelity or proofs showing convergence of representations to low-distortion manifolds.
Adding a distortion vs. curvature empirical curve or formal derivation would strengthen the argument substantially.

Limited comparison with non-Euclidean or hybrid geometries.
The paper frames hyperbolic geometry as the only alternative, but mixed-curvature or spherical–hyperbolic hybrid embeddings could better capture cross-function semantics.
A control experiment with mixed curvature or product manifolds (𝔼×ℍ) would clarify whether pure hyperbolic geometry is indeed optimal.

Overlooked semantic–syntactic decoupling.
The model tightly couples AST topology with embedding curvature but does not explicitly distinguish between semantic relations and syntactic nesting.
This may limit generalization to tasks involving cross-file or semantic code reasoning. Integrating semantic edges (DFG, CFG) or cross-function attention could make the representation more complete.

### Questions
1.	Provide a formal distortion analysis: derive or empirically approximate the embedding distortion as a function of curvature.
2.	Include ablation across manifold types: compare Lorentz, Poincaré, and product manifolds.
3.	Analyze training dynamics of curvature annealing: curvature vs. epoch plot.
4.	Explore semantic augmentation: integrating CFG/DFG edges to test the universality claim.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
​​This paper argues that hyperbolic geometry is a better inductive bias for program representations than Euclidean space, because Abstract Syntax Trees (AST) are tree-like and expand exponentially. The authors introduce HypeCodeNet, a Lorentz-model hyperbolic GNN that introduces manifold aware operations to the embedding layer and message passing attention module.  Across code detection, line-level code completion, and link prediction on call graphs, HypeCodeNet achieves SOTA results and exceeds strong Transformer and graph baselines, demonstrating that hyperbolic geometry offers meaningful advantages for modeling hierarchical code structures.

### Strengths
The paper demonstrates careful engineering work to make hyperbolic deep learning stable and practical for code representation:

1. **Consistent improvements across diverse tasks demonstrate general utility and significance.** The approach shows modest but consistent gains over CodeFORMER in semantic understanding (BigCloneBench: +1.2% F1) and code completion (~1% across metrics), while achieving substantial improvements in link prediction (+5.0% AUC, +5.2% Hits@10). This variation suggests the method particularly excels at structural reasoning tasks.

2. **High-quality experimental evaluation.** The authors compare against 9 baselines spanning sequence-based, graph-based, and hybrid architectures across multiple datasets per task. The ablation studies (Section 4.5) provide clear empirical evidence that performance gains stem from hyperbolic geometry rather than other architectural choices.

3. **Original technical contributions beyond existing hyperbolic models.** HypeCodeNet advances beyond prior work through: (i) use of the numerically stable Lorentz model, (ii) geodesic-based attention mechanism, and (iii) curvature annealing for stable training. These design choices address known challenges in hyperbolic deep learning.

4. **Strong empirical validation of theoretical claims.** The distortion analysis (Figure 7, Appendix) provides compelling evidence that hyperbolic embeddings preserve AST hierarchical structure with significantly less distortion than Euclidean alternatives, supporting the paper's central thesis. 

5. **Well-written paper with clear presentation.** The overall thesis is compelling and the technical approach is explained systematically. The experimental section is particularly well-organized. (Though Figure 1 could be improved, as noted in the weaknesses section)

### Weaknesses
1. **Insufficient Geometric Analysis of Learned Representations.** While the paper provides compelling empirical evidence that hyperbolic geometry improves performance, it lacks rigorous geometric validation of the learned embeddings. The authors cite Yang et al. (2023) for hyperbolic representation learning, yet that work explicitly cautions that hyperbolic embeddings do not automatically guarantee hierarchical structure and demonstrates cases where geometric properties may not align with semantic hierarchies. While Figure 8 provides valuable qualitative intuition, it requires accompanying quantitative validation. The current visualization alone cannot confirm that the hierarchical structure is preserved beyond visual inspection.

**Suggested quantitative analyses:**
   - **Distance-to-Origin Analysis:** Provide quantitative measurements of node distances from the origin in the Lorentz model. Specifically, verify that root nodes consistently map near the origin (small $d(v, o_c)$ ) while leaf nodes map toward the boundary, with statistical significance tests across multiple ASTs.
   - **Hierarchical Structure Validation:** Compute the correlation between graph depth and hyperbolic distance to origin. Following Nickel & Kiela (2018), report Spearman's rank correlation coefficient between tree depth and $\|h_v\|_L$.
   - **Statistical Validation:** Consider adding histograms of distance distributions by tree depth and statistical tests confirming that parent-child pairs maintain consistently smaller distances than arbitrary node pairs at the same depth.

2. **Figure 1 needs significant improvement to effectively convey the paper's core contributions.**
   
   **Part (a) - The geometric intuition is too generic and could represent any hierarchical structure.** 
   The authors might consider:
   - Showing a concrete code snippet (5-10 lines) with its actual AST and node labels displaying real AST node types (e.g., IfStatement, ForLoop, Variable)
   - Providing a side-by-side comparison with Euclidean embeddings to demonstrate distortion differences
   
   **Part (b) - The 'log-aggregate-exp' visualization feels cluttered and unfocused.** 
   To improve clarity, the authors could:
   - Use progressive panels that track a specific node through each transformation step
   - Add visual differentiation through color coding (e.g., parent nodes in blue, children in green, message flow as arrows)
   - Emphasize the tangent plane with shading or perspective to help readers understand why operations must occur in this local Euclidean space
   
   While these are suggestions, addressing the abstract nature of the current visualization would significantly strengthen the paper's accessibility.

### Questions
1. The visualization in Figures 7 and 8 is compelling.
   - How many ASTs contributed to the aggregated statistics in Figure 7? (Figure 8 appears to show a single example)
   - What is the size range of AST(s) in your evaluation (min/max/median nodes and depth)?
   - Does the distortion advantage hold consistently across different code complexity levels?

2. How does memory consumption scale compared to Euclidean models, especially for very large ASTs?

3. The paper is missing a citation for CodeFORMER, which appears to be your primary baseline.  Liu et al. (2023) "CodeFORMER: A GNN-Nested Transformer for Source Code Representation" (MDPI Electronics 12(7):1722). Please confirm this is the correct reference and add it to your bibliography.

### Soundness
3

### Presentation
4

### Contribution
3

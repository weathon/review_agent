# Multi-Condition Conformal Selection

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Selecting high-quality candidates from large-scale datasets is critically important in resource-constrained applications such as drug discovery, precision medicine, and the alignment of large language models. While conformal selection methods offer a rigorous solution with False Discovery Rate (FDR) control, their applicability is confined to single-threshold scenarios (i.e., y > c) and overlooks practical needs for multi-condition selection, such as conjunctive or disjunctive conditions.  In this work, we propose the Multi-Condition Conformal Selection (MCCS) algorithm, which extends conformal selection to scenarios with multiple conditions. In particular, we introduce a novel nonconformity score with regional monotonicity for conjunctive conditions and a global Benjamini–Hochberg (BH) procedure for disjunctive conditions, thereby establishing finite-sample FDR control with theoretical guarantees. The integration of these components enables the proposed method to achieve rigorous FDR-controlled selection in various multi-condition environments. Extensive experiments validate the superiority of MCCS over baselines, its generalizability across diverse condition combinations, different real-world modalities, and multi-task scalability.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces multi-conditional score for conformal selection (MCCS). Current approaches are restricted to single-threshold scenarios and MCCS proposes a novel nonconformity score for conjunctive and disjunctive conditions that is capable of dealing with multiple conditions. The experiments show the effectiveness of the technique in multiple applications and also for multi-variate scenarios.

### Strengths
- The paper presents a novel extension of conformal selection to multiple conditions, it addresses an important gap in the area.
- The paper present theoretical guarantees via finite sample FDR control. 
- MCCS general framework is capable of handling both conjunctive and disjunctive conditions, multiple intervals and multivariate responses. That broad scope is a strength of the work.

### Weaknesses
- The experimentation section in terms of baselines look not fully explored. Experiments against other FDR control methods will strength the claims.

### Questions
- Can the authors add some discussion regarding computational requirements of MCCS, complexity and memory estimates will clarify and possibly amplify the applicability of the approach. 
- Briefly discuss extensions of the current method to domain shift will strength the paper claims. 
- Regarding algorithm 2 (Global BH procedure) $m \times K$ sorting seems a bottleneck, is there a way to make that part more efficient?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Multi-Condition Conformal Selection (MCCS) algorithm, an extension of the conformal selection framework designed to address scenarios involving multiple selection criteria. The method specifically tackles conjunctive conditions and disjunctive conditions, which frequently arise in practical applications where decisions depend on simultaneous or alternative constraints. The central innovation lies in the development of a tailored nonconformity score that ensures regional monotonicity for interval-based selections, alongside a global adaptation of the Benjamini-Hochberg (BH) procedure to manage disjunctive constraints. These components are underpinned by rigorous theoretical analysis, establishing finite-sample control of the False Discovery Rate (FDR) under exchangeability assumptions.

### Strengths
1.  **Theoretical Soundness:** The paper provides rigorous theoretical foundations, including proofs for finite-sample FDR control under exchangeability conditions (Theorem 4.1). The extension of conformal selection theory to multi-condition settings is non-trivial and well-articulated.
2.  **Comprehensive Evaluation:** The experimental section is thorough, demonstrating the method's applicability across diverse modalities (text, images, multi-modal) and tasks (single-class, multi-class, similar-class selection). The comparison with relevant baselines (Inter-cfBH, Union-cfBH) is appropriate and highlights the limitations of naive approaches.
3.  **Clear Presentation:** The paper is generally well-written, with a logical flow from problem formulation to methodology, theory, and experiments. The figures, such as the illustrative diagrams, aid in understanding the core concepts.

### Weaknesses
1.  **Perceived Marginal Technical Contribution (Major Concern):** The fundamental idea can be viewed as generalizing the single-threshold selection $Y > c$ to a more general form $Y \in I$. A significant portion of the theoretical and methodological machinery seems to be a direct adaptation of the established cfBH (Jin & Candès, 2023) framework.
     - The problem could potentially be reframed by defining a suitable function $\phi(Y, I)$ that encapsulates the multi-condition logic, reducing it to a single-threshold problem on $\phi(Y, I)$. For instance, for an interval $I = (a, b)$, one could define $\phi(Y, I) = \min(b-Y, Y-a)$, where $\phi(Y, I) > 0$ is equivalent to $Y \in (a, b)$. A similar construction could be devised for disjunctive conditions. The core conformal p-value machinery and BH procedure from cfBH could then be applied to $\phi(Y, I)$, potentially obviating the need for the newly proposed nonconformity scores and the specific "global BH" narrative. While the authors' approach of designing a regionally monotone score is one valid path, the paper would be significantly strengthened by directly addressing this alternative, more unified perspective. A discussion justifying why their specific decomposition (separate scores per interval + global BH) is preferable to a unified $\phi$-function approach is crucial. The current presentation makes the contribution feel more like a specialized extension rather than a fundamental generalization.

2.  **Clarity on Theoretical Novelty:** The proof techniques, particularly for Theorem 4.1, heavily rely on and extend the principles established in cfBH. The paper could more clearly delineate the specific novel technical challenges overcome in the multi-condition setting compared to the single-condition base.

[1] Jin Y, Candès E J. Selection by prediction with conformal p-values[J]. Journal of Machine Learning Research, 2023, 24(244): 1-41.

### Questions
1. **On the Core Technical Contribution and Methodological Necessity:** My primary concern pertains to the fundamental architecture of MCCS. Please refer to weakness 1.
2. **On Experimental Design and Interpretation:** 
    - The experiments primarily feature a small number of conditions (e.g., Tasks 1-6 in Table 2). A key claim of the method is its generalizability to diverse condition combinations. How does the power of MCCS scale as the number of intervals $K$ becomes large?
    - The paper reports aggregate FDR and power. It would be insightful to analyze which specific conditions (which intervals ) contribute most to the discoveries. Does the global BH procedure lead to a balanced selection across all target intervals, or does it favor intervals with certain characteristics?
3. **On the Equivalent Interval Representations:** Consider a target interval $I = (a, b)$. This can be equivalently expressed as a single interval or decomposed into two sub-intervals $(a, c] \cup [c, b)$ (for $K=2$). While these two representations are semantically equivalent in defining the target set, will they lead to different algorithmic constructions? Can you explain on this?

I will raise my score if these concerns are properly handled.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the conformal selection (CS) problem, that is, to select --- among many test points --- those samples whose unknown responses satisfy a pre-specified property. While previous conformal selection method addresses a one-sided property, this paper considers multiple conditions such as an interval or the combination of two one-sided intervals. The authors first propose that taking the intersection/union of two conformal selection sets invalidates the FDR control. Then, for conjunctive conditions, they propose a method that designs a conformity score specialized for addressing such properties in the CS framework. For disjunctive conditions, they propose to select from all p-values for each test point and each property with BH, and show the FDR control. The proposed methods are demonstrated in extensive simulations and real-world applications.

### Strengths
1. The paper is tightly structured and clearly written. 
2. The problem addressed is of practical relevance. 
3. The discussion on technical challenges is precise and convincing.
4. The numerical experiments are extensive and solid.

### Weaknesses
1. The FDR guarantee for disjunctive conditions is hard to interpret (see my Q1 below).
2. The solution seems over-complicated (see my Q2 below). Seems a simple strategy can greatly simplify the proposal.

### Questions
1. What does the FDR control for selecting among all the $(j,k)$ pairs in Algorithm 3 mean? The definition is not explicitly given, and based on my guess, the practical interpretation is a bit weird to me. Ideally we want an FDR over *samples* so that at least $1-\alpha$ fraction of them satisfy any of the conditions. However, in the current setup, consider 2 conditions, and suppose a sample j is selected for both $(j,1)$ and $(j,2)$, its meaning would be a bit strange, and I'm not sure the FDR over all the pairs can reflect the practical need very well. 
2. I was wondering if the problem can be solved via a simpler strategy. Suppose we want to select samples with $Y\in (-\infty, c_1] \cup [c_2,+\infty)$. Can we define a property as $Y^* = \mathbf{1}_{Y\in (-\infty, c_1] \cup [c_2,+\infty)}$ and directly apply the conformal selection method?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper considers the problem of selecting samples with FDR control, where multiple conjunctive or disjunctive intervals characterize the selection criterion. The proposed method generalizes *conformal selection* by Jin and Candès, and is evaluated in numerical experiments.

### Strengths
The writing of this paper is clear and easy to follow; the technical derivation is rigorous.

### Weaknesses
The problem raised in this paper seems to have already been solved by existing works in full generality. 
For example, [1] considers the selection criterion characterized by a general set and allows it to 
depend on the covariate, where the idea is to design the score reflecting the likelihood of $Y$
falling into the property set. The proposed (adaptive) method therein achieves 
finite-sample FDR control, with cfBH as a special example. It would be helpful if the author 
could clarify the contributions of the work.

**References** 

1. Gui, Yu, et al. "ACS: An interactive framework for conformal selection." arXiv preprint arXiv:2507.15825 (2025).

### Questions
Refer to the "Weaknesses" section.

### Soundness
3

### Presentation
3

### Contribution
1

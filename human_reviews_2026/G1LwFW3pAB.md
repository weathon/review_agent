# From Comparison to Composition: Towards Understanding Machine Cognition of Unseen Categories

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Humans are known to acquire and generalize visual concepts through a natural compare–then–compose process. We ask whether this mechanism can provide principled conditions under which machines generalize existing knowledge to unseen categories. In this work, we formalize cognition of the unseen as two complementary mechanisms for deep learning models: comparison, which uncovers latent concepts by capturing cross-category variations among seen classes, and composition, which extrapolates these concepts continuously to unseen classes. Even without parametric assumptions, we establish identifiability guarantees for learning latent concepts and unseen categories via sufficient contrast and independent support separation, denoted as Comparison–C}omposition Cognition (C^3). Guided by these results, we instantiate a structurally constrained generative model mirroring our theoretical assumptions. Our results on simulated data corroborate our theoretical claims and the effectiveness of our proposed methodology. In the setting of visual cognition with unseen labels, aka On-the-fly Category Discovery, our instantiated approach improves state-of-the-art baselines by +3.8\% average accuracy across fine-grained benchmarks. Taken together, our framework offers principled conditions and practical guidance for representational compositionality, offering a theory-to-practice path for generalization to unseen categories.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new approach for the problem of online category discovery. It comprises two parts. In the first part, a theoretical formulation of the problem is presented. This is based on a stochastic model where old and new categories are modelled as sets of concepts. Statistical assumptions are given for model identifiability, as well as for the ability of the model to detect and recognise new categories as they are encountered in the data stream. The second part of the paper presents a practical deep learning formulation that is inspired by the first part. When assessed on standard benchmarks for this task, this new formulation performs well. Some ablations illustrate the importance of various elements of the formulation.

### Strengths
* Category discovery captures an important problem where machine learning still struggles. Online category discovery is a variant of this problem somewhat closer to a real-life application, where new data streams through and new categories must be detected and instantiated on the fly. These are difficult problems worth looking at.

* Insofar as I can tell, the authors propose a reasonable formulation for online category discovery supported by a thorough formal analysis.

* The practical instantiation of this formulation is shown to perform relatively well on canonical benchmarks in the area.

* Ablations show the importance of at least some of the key design choices.

* A long appendix provides many important aspects of the formulation and implementation that are missing in the main paper.

### Weaknesses
The approach is rather complex, involving a number of regularisers and steps that would be difficult to reproduce based solely on the description in the paper (even once one considers the thorough appendix). The authors do not mention whether code will be released.

On a fundamental level, this paper assumes that categories map neatly to combinations of "base concepts", and that new categories are formed as new combinations of concepts that are, otherwise, already known. This sounds like a fundamental limitation which should probably be acknowledged and discussed a bit more.

There are no illustrations of what the concepts discovered by the algorithms are, particularly on the computer vision dataset. Are these interpretable in any way?

The main paper is not entirely self-contained and some critical information is missing, which makes it unnecessarily difficult to understand:

* The problem formulation on line 264 is incomplete, and one *must* read the references to even understand what problem is being solved. Specifically, I could not understand the sentence [the model] "classifies streaming query instances in real-time without access to labels from novel categories" until reading Du et al. 2023. The task is poorly defined in the paper, and so are the evaluation metrics.

* It would have been very helpful to expand on several of the introductory materials. What is a "basis distribution" in Eq. (1) (line 107)? Why does it make sense for $f$ to list $\mathbf{z}$ both as an input and as an output (line 131)? How are Definition 1 and Eq. (1) connected? One is required to be familiar with prior work, and with Markov networks and other background, to fill in the gaps.

* In Eq. (4), do you mean to say that $p(\hat \mathbf{c}|y)$ is independent of $y$?

* In Eq. (5), is $[\hat\mathbf{c}_i]$ a vector? Line 106 suggests that it is a scalar instead. If so, why do you need a vector norm?

Minor: Line 408 suggests that Table 1 reaches a peak at $n_s=24$, but the table ends at 24. How can you tell this is a peak?

### Questions
Please see the questions included in the "weaknesses" section above.

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
3

### Summary
This paper proposes C3 (Comparison–Composition Cognition), a cognitive-inspired theoretical and practical framework for generalizing to unseen categories. The authors formalize human-like “compare–then–compose” reasoning into two complementary mechanisms: comparison, which extracts latent concepts by capturing cross-category variation, and composition, which recombines these concepts to recognize unseen categories. Experiments on eight fine-grained benchmarks under the On-the-Fly Category Discovery (OCD) setting demonstrate consistent performance improvements.

### Strengths
1.This paper is mainly for providing formal identifiability guarantees for concept-based generalization. It features rigorous derivations, and its stepwise logic has a clear structure: first contextual separation, then semantic disentanglement, and ultimately compositional generalization.
2.Framing unseen-category cognition as “from comparison to composition” offers a interpretable metaphor that aligns with cognitive science findings.
3.Every theoretical assumption (A1–A6) is explicitly operationalized in the model (e.g., flow for smooth density, sparse MoE for structure, prototype and hashing losses for semantic contrast).

### Weaknesses
1.The exposition is heavy and unfocused. Many formulas are presented without sufficient intuition or geometric explanation. Readers must infer the motivation behind several derivations (e.g., partial derivatives of log-density in Theorem 2), making the theoretical contribution harder to follow.
2.While the framework borrows cognitive terminology “compare–then–compose”, its connection to actual cognitive mechanisms is largely metaphorical. A clearer justification for how these processes correspond to neural computations would strengthen the interdisciplinary claim.
3.Although the framework is complete, it feels assembled rather than organically derived. Each assumption leads to a separate engineering component.
4.Empirically, results are solid but do not reveal new phenomena or insights. The performance gains are incremental, and no analysis shows why the comparison–composition pipeline generalizes better than alternative factorization approaches.
5.The paper does not report computational efficiency, convergence stability, or scaling behavior; the additional flow and MoE modules could impose extra overhead.

### Questions
1.How sensitive are the identifiability results to approximate violations of A1–A6? Are these conditions ever testable in real datasets?
2.Could the use of multiple independent regularizers (flow likelihood, hashing, prototype contrast) lead to competing gradients that distort the latent structure?
3.Would a simpler variational or contrastive formulation achieve similar results without flow modeling?
4.How can the authors ensure that the claimed “composition” corresponds to interpretable recombination of learned semantic atoms, rather than a generic feature interpolation? Is there any visual or quantitative evidence to support this?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a new framework, Comparison–Composition Cognition (C3), inspired by human cognitive processes, to address the challenge of generalizing to unseen categories. The framework divides learning into two mechanisms: comparison, which uncovers latent concepts by contrasting within seen classes, and composition, which combines these concepts to predict unseen categories. The method is theoretically grounded, with identifiability guarantees for both learning latent concepts and for generalizing to novel categories. It is experimentally validated through On-the-fly Category Discovery (OCD) benchmarks, showing a 3.8% improvement in accuracy over state-of-the-art methods. The paper suggests that concept-based representations, following the compare-then-compose approach, can enable open-world generalization.

### Strengths
- Novel framework: Introduces a theoretically-grounded framework, C3, which mimics human cognition for unseen category recognition via comparison and composition.

- Theoretical guarantees: Establishes identifiability results, providing confidence that latent concepts can be learned and generalized to unseen categories.

- Solid experimental results: Achieves notable improvements over existing methods on OCD benchmarks, demonstrating the practical viability of the approach.

### Weaknesses
- Incremental contribution: While the framework is innovative, it builds on existing concepts from the literature (e.g., contrastive learning, compositionality), making the contribution seem incremental.

- Limited scalability discussion: The paper does not address how the proposed method might scale with larger models or datasets, especially in real-world applications.

- Lack of detailed ablation study: The paper could benefit from a deeper ablation study to evaluate the individual contributions of comparison and composition mechanisms.

### Questions
- How does the method perform when applied to datasets with a higher number of unseen categories?

- What would be the impact of integrating this method with large-scale foundational models like GPT or CLIP?

- How does the proposed C3 framework handle noisy or ambiguous data in unseen categories?

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
5

### Summary
This paper proposes a framework, named Comparison-Composition Cognition, for generalizing to unseen visual categories, inspired by human cognitive mechanisms. The authors formalize the problem as a two-step process: (1) comparison, which aims to identify disentangled semantic and contextual concepts from seen categories, and (2) composition, which recombines these learned concepts to recognize novel categories. The core contribution is a set of theoretical results that provide identifiability guarantees for these latent concepts under specific assumptions like sufficient data contrast and support separation. To operationalize this framework, the authors design a deep generative model for the On-the-fly Category Discovery (OCD) task.

### Strengths
1. The paper is motivated by the human cognitive process of understanding new concepts through comparison and composition. It aims to build a principled framework for machine learning models to generalize to unseen categories by mimicking this process.

2. It formalizes this process as Comparison-Composition Cognition. "Comparison" is framed as a latent variable identification problem to uncover disentangled concepts from seen data. "Composition" is the process of using these learned concepts to recognize unseen categories.

3. The paper provides theorems establishing identifiability guarantees for latent concepts (both contextual and semantic) under specific conditions, such as sufficient cross-category contrast and sparse arrangements (Theorems 1 & 2). It also provides conditions for generalizing to unseen categories via composition, namely support separation and marginal coverage (Theorem 3).

4. The theory is instantiated as a generative model for the On-the-fly Category Discovery  task. The model uses a frozen DINO encoder, a learnable mask to separate semantic and contextual features, a sparsely-gated Mixture-of-Experts (MoE) and a normalizing flow to model the structure of semantic concepts, and prototype/hashing losses to enforce discriminability and concept separation.

### Weaknesses
1. The primary weakness of this paper lies in the tenuous connection between the ambitious theoretical framework and the practical implementation. The strong assumptions in the theorems are only loosely and indirectly addressed by the chosen technical components. The "Sufficient Contrast" assumption (A2, A3), which requires linear independence of vectors of log-density derivatives, is a highly specific mathematical condition. The paper claims this is satisfied by using a standard prototype-based supervised loss (L_proto). This link is weak and not justified; there is no guarantee that optimizing a prototype loss leads to the satisfaction of this complex condition.

2. When stripped of its theoretical narrative, the proposed method appears to be a complex but incremental combination of existing techniques rather than a breakthrough.
- The architecture is essentially a sophisticated VAE that uses a learnable mask for disentanglement (a common idea), a sparsely-gated MoE plus a normalizing flow for structured modeling (both well-established tools), and prototype/hashing losses for discriminability.
- The core idea of using hashing and prototypes for category discovery has been explored by the main competitor this paper compares against (PHE: Prototypical Hash Encoding). The performance gain, while present, might be attributable to the increased complexity and more moving parts of the model rather than a fundamental conceptual advance.

3. The "Concept" Framing is Not Empirically Validated: The paper is built around the idea of learning meaningful, disentangled concepts (e.g., "white head," "black wings") like [2]. However, there is no qualitative or quantitative analysis to demonstrate that the learned latent variables z actually correspond to such interpretable concepts. The model might just be learning effective discriminative features. Without such evidence, the connection to human cognition and concept composition remains a compelling story but an unproven claim, diminishing the paper's main appeal.

4. In the task of discovering general categories, there has already been similar work [1] that decomposes objects into combinations of various attributes (textual or visual) with MoE. I believe there needs to be more comparative discussion with the current work.

[1] Dissecting Generalized Category Discovery: Multiplex Consensus under Self-Deconstruction. In ICCV, 2025.

[2] OCRT: Boosting Foundation Models in the Open World with Object-Concept-Relation Triad. In CVPR, 2025.

### Questions
1. The central narrative of the paper is about learning and composing concepts. Can you provide qualitative evidence (e.g., by traversing the learned latent space z and visualizing its effect on generated images) or quantitative analysis to show that the model is indeed learning disentangled and semantically meaningful concepts, as opposed to simply learning complex, entangled features that happen to be discriminative?

2. The proposed method combines multiple advanced components (MoE, flows, hashing). The main competitor, PHE, also relies on prototypical hashing. Could you clarify the key technical innovation of your method that is responsible for the performance gain, beyond the theoretical framing and the increased architectural complexity? Is it the explicit separation of contextual/semantic factors, the structural modeling via flow, or another specific component?

### Soundness
3

### Presentation
2

### Contribution
2

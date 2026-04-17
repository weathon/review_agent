# Gauge-invariant representation holonomy

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 6

## Abstract
Deep networks learn internal representations whose geometry—how features bend, rotate, and evolve—affects both generalization and robustness. Existing similarity measures such as CKA or SVCCA capture pointwise overlap between activation sets, but miss how representations change along input paths. Two models may appear nearly identical under these metrics yet respond very differently to perturbations or adversarial stress. We introduce representation holonomy, a gauge-invariant statistic that measures this path dependence. Conceptually, holonomy quantifies the “twist” accumulated when features are parallel-transported around a small loop in input space: flat representations yield zero holonomy, while nonzero values reveal hidden curvature. Our estimator fixes gauge through global whitening, aligns neighborhoods using shared subspaces and rotation-only Procrustes, and embeds the result back to the full feature space. We prove invariance to orthogonal (and affine, post-whitening) transformations, establish a linear null for affine layers, and show that holonomy vanishes at small radii. Empirically, holonomy scales with loop radius and depth, separates models that appear similar under CKA, and correlates with adversarial and corruption robustness across training regimes. It also tracks training dynamics as features form and stabilize. Together, these results position representation holonomy as a practical and scalable diagnostic for probing the geometric structure of learned representations beyond pointwise similarity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This work presents a geometrically guided diagnostic tool for meaning the path-dependent geometry of learning features in neural networks. The authors present a novel problem formulation and approach that empirically demonstrate that lower holonomy results in greater model robustness. While the contributions are significant the presentation of the paper could be improved in places with improved rationale, and writing clarity. However, the resulting outputs are potentially valuable as a tool for model comparison and diagnostic work for practictioners and researchers.

### Strengths
- The identified problem of intermediate feature paths during input perturbations is an interesting and rationaled area of study. The authors address a potentially problematic area of neural network diagnostics, and present an interesting solution using geometry as a diagnostic tool, which is a valuable and interesting direction.
- The mathematical approach is seemingly well constructed and correct, with the representation holonomy being a highly original and novel direction of model analysis, supported by proofs. 
- The results and analysis demonstrates the correlation of holonomy with adversarial robustness, and links model geometry and its model robustness, a new insight that is an interesting tool for early-training model selection and to investigate flatter representation manifolds.
- Limitations are presented and discussed.

### Weaknesses
**Major:**
- The problem setting while rationaled in the text could be better supported and demonstrated empirically. For example, what evidence is provided that supports the statement “two models can appear highly similar under CKA or CCA, and still behave differently under adversarial or corruption stress because their intermediate features rotate differently along input paths.”
- Generally the presentation of the manuscript is poor in places (mainly in its clarity and readability), many of the preliminaries and experimental setup information is assumed by the authors. For example 5.2 is presented with little introduction, and 

**Minor:**
- Acronyms such as SVCCA, CKA, etc need to be defined, while most readers will know such terminology, it is good practice to define such acronyms.
- The term corruption or adversarial stress could be better defined in the early part of the work to introduce the reader to the concepts. However, this is a minor readability complaint and an opinion. 
- However, much of the presentation could be improved where some preliminary explanation and definitions would strengthen the readability. This could include moving some of the preliminaries from the appendix into the main manuscript as there is space to do so.
- Code should be included for reproduction and validaiton.

### Questions
1. You mention: “projected by a fixed orthonormal Johnson–Lindenstrauss map to p⋆=1024 if needed”. What is the the determination if the projection is needed or not?

2. Did you experiment with alternatively chosen loops, if so how much does performance vary? Given this is a noted limitation, is it significantly impactful to the work?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a new geometric framework called representation holonomy, which quantifies how neural features change along loops in input space using gauge-invariant parallel transport. The authors define a practical estimator combining global whitening, shared-subspace Procrustes alignment, and loop composition, and they prove gauge and affine invariance as well as small-radius asymptotics. Empirical analyses on MNIST and CIFAR suggest that holonomy increases with layer depth and may correlate with robustness indicators such as adversarial and corruption accuracy.

The concept is theoretically intriguing and potentially valuable for studying the geometry of learned representations. However, both the experimental validation and the presentation of the theory can be much improved for clarity and completeness.

### Strengths
* Conceptual originality: Introducing holonomy—a gauge-invariant geometric notion from differential geometry—into representation analysis is creative and nontrivial.
* Mathematical rigor: The formal statements (gauge invariance, affine invariance, small-radius behavior) are internally consistent using matrix perturbation tools. Although I have to admit that I didn’t get a chance to go over all the appendices.
* Potential significance: In principle, this idea could offer new insights for feature-space curvature and robustness that go beyond CKA and related similarity measures.

### Weaknesses
* Incomplete and inconclusive experiments: The reported results (e.g., Figs. 2–3) show overlapping confidence intervals, and several key correlations (Table 2) are weak or include zero within the 95% CI. Without stronger or broader evidence, the empirical support for holonomy as a meaningful metric remains limited.
* Missing comparisons to existing metrics: Although the paper discusses CKA and CCA extensively, it never provides side-by-side empirical comparisons. Such baselines are crucial to demonstrate what new insights holonomy provides.
* Lack of sensitivity analysis: The estimator relies on multiple hyperparameters (e.g., k). The paper should quantify how results depend on these choices to establish robustness.
* Unclear connection to task relevance: Theoretical results (e.g., Eq. 4) include several interacting error terms, but it remains unclear how holonomy relates in a concrete or predictive way to model performance or generalization.
* Poor illustration of geometric intuition: Given the geometric nature of the work, schematic or pictorial figures showing loops, transports, or curvature effects are essential. The heavy formalism and lack of visual explanation make the paper difficult to follow.
* Presentation density: The writing style is more reminiscent of a mathematical physics paper than a machine-learning paper, which may alienate part of the ICLR audience.

### Questions
1. How sensitive are the results to hyperparameter choices such as k?
2. Can you provide quantitative comparisons with CKA or SVCCA to highlight the unique information captured by holonomy?
3. Among the four terms in the error bound (Eq. 4), which dominates in practice?
4. In your experiments, are there any correlations between task performance (e.g., error) and your measures?
5. Could schematic illustrations or toy examples be added to visually illustrate the concept of holonomy in representation space?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a new representation similarity metric that complements existing metrics like CCA or CKA by taking into account how changes the input space map to changes in the representation space. In particular, the proposed representation holonomy metric is invariant to affine transformations and is zero for small loops in the input space. The estimator for this quantity is shown to be numerically stable and tractable to compute for various architectures of DNNs. Experiments show that holonomy is a valid measure of geometric effects (while measuring something different than other RS metrics), relatively robust to parametric choices, and can be used to predict robustness from small probes.

### Strengths
* Holonomy captures properties of a network that aren't measured by other representation similarity metrics. In particular, this metric captures the dynamics of representations as inputs change, which may make it more relevant to studies of properties like robustness than traditional RS metrics.
* Holonomy can be used in conjunction with existing RS metrics to allow for a significantly deeper understanding of learned representations, as shown in section 5.3.
* Estimates of error are included with most of the experimental results.
* The evaluation section provides strong evidence that the theoretical claims made in Section 3 are exhibited in realistic settings.
* Practical improvements can be made using holonomy as a diagnostic, such as improving methods for early stopping.
* Limitations are thoroughly discussed in section 6.1.

### Weaknesses
* The theory is somewhat difficult to follow. A reader coming from the representation similarity literature might not have a background in the mathematics that are relevant to this paper. I think that defining terms more clearly and pointing readers to relevant background would be helpful. For example, gauge-invariance is never defined despite being in the title. Given the breadth of the ICLR audience, this type of knowledge cannot be assumed.
* Similarly, the inclusion of proof sketches would be helpful for readers. What are the important mathematical techniques that are used to establish the properties of representation holonomy?

### Questions
* Does the number of points in the loop impact the accuracy/variance of your estimator? Why did you choose 12 points for your experiments?
* Do the suggestions for uses of holonomy in section 6 correspond to any concrete experiments, or are these future avenues of study?
* The caption for Figure 2 states that Layer 2 exhibits larger holonomy, but the data in the plot for Layer 1 appears to be larger in magnitude, which appears to contradict the caption. Is my understanding of the data being displayed in Figure 2 incorrect?

### Soundness
4

### Presentation
2

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
The authors define, implement, and test a measure of Holonomy, designed to quantify geometric distortions within network representations, in a way that current metrics do not.  The authors provide mathematical definitions, details of computation/implementation, and empirical demonstrations on a variety of datasets and models. These demonstrate that holonomy differs from other metrics in the literature, confirm expected properties (increases with stimulus loop radius) as well as less-expected properties (increases with network depth, decreases with training, and correlates with average network robustness to noise or adversarial attack), suggesting its usefulness in applications of diagnosing and improving networks.

### Strengths
This is an important topic.  ML networks are highly redundant in their parameterization and optimization landscapes, and it is
critical for the field to develop measures that can characterize fundamental properties for use in analysis/comparison/design.

The paper is built on a mathematical foundation that is rich, and perhaps not so familiar to a large portion of the NeurIPS community.  The authors have generally done a good job of defining and explaining their construction, and the community will benefit (but see below).

The authors have also done a careful job of implementing and testing their measure, providing reasonable detail, evaluation of computational cost, verification of expected properties, and demonstration of potential uses.

### Weaknesses
The main weakness, for me, is that after several passes I am still uncertain of exactly why the properties captured by holonomy (distortion of geometry along closed-loop paths) are essential to understanding or comparing network functionality.  Specifically, under what conditions are the properties captured by holonomy, but not captured by current alignment methods (CCA, RSA) or differential analalyses (Lie derivatives, or simply comparison of local Jacobians), essential for evaluating or improving a network?  When is loop composition/path-dependency important (as opposed to simpler tests of invariance or equivariance)?  Perhaps the authors could provide more examples of applications/tasks where this arises?

Another weakness: Given that derivation and computation are complex and involve many approximate steps, it would be really useful to verify the entire enterprise on a few simple examples for which ground truth is known.  For example, image translation loops represented in a purely convolutional (and thus translation-equivariant) architecture, vs. a representation with known aliasing artifacts (e.g., subsampling in the inner layers).  This could prove interesting also in evaluating boundary-handling artifacts (the other source of non-translation-invariance in CNNs), which are expected to worsen in later layers.  I do see in the Discussion that the authors specify "beyond-local-loops" as a future direction, so my example woud need to be computed with very small displacements, perhaps to the point of rendering it unconvincing.

### Questions
In Figures 1/2, why does measured holonomy not tend to zero as r goes to zero? Also, can the authors rule out the possibility that the increase of holonomy with loop radius is a result of increased numerical error?  Does the trend persist if the number of loop samples is adjusted in proportion to loop radius r?

Have you measured holonomy of a score network used for diffusion generative modeling?  Some literature suggests that  generalization and equivariance to geometric distortions are more robust than for recognition networks (eg, Kadkhodaie et al 2024).  But they are also typically not curl-free (even though an optimal denoiser should be, since it computes a score), and this failure would presumably be exposed by the holonomy measure.

### Soundness
4

### Presentation
3

### Contribution
3

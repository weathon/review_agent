# Influence Dynamics and Stagewise Data Attribution

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 4, 4

## Abstract
Current training data attribution (TDA) methods treat the influence one sample has on another as static, but neural networks learn in distinct stages that exhibit changing patterns of influence. In this work, we introduce a framework for stagewise data attribution grounded in singular learning theory. We predict that influence can change non-monotonically, including sign flips and sharp peaks at developmental transitions. We first validate these predictions analytically and empirically in a toy model, showing that dynamic shifts in influence directly map to the model's progressive learning of a semantic hierarchy. Finally, we demonstrate these phenomena at scale in language models, where token-level influence changes align with known developmental stages.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a stage-wise influence framework to track how data influence evolves throughout training. It predicts non-monotonic behavior and validates these in both a toy model and large language models (LLMs).

### Strengths
1. It introduces a novel framework of *stage-wise influence*, enriching the understanding of data attribution beyond static snapshots.
2. Makes clear predictions and uses Bayesian Influence Functions to measure influence over time, confirming them in a toy model and in LLMs.

### Weaknesses
1. Empirical verification depends on BIF and SGLD:
Validation relies on BIF, which requires sampling a local posterior via RMSProp-SGLD around each checkpoint, while most models are trained with Adam-family optimizers, raising questions about stage-wise behavior under Adam. 
    
2. Computational overhead:
BIF uses Monte Carlo sampling, which is costly; complexity/scalability is not thoroughly analyzed, so practicality at scale remains uncertain.
    
3. Limited experimental scope:
Experiments emphasize inter-class patterns (to align with Sec. 2.3), but broader claims about *when* data are influential and *how* they shape internals would benefit from intra-class analyses and more mechanistic evidence

### Questions
1. Figure 3: BIF and analytical IF resemble LOO but peak at different times. Why don’t the maxima align?
2. Figure 7: The influence curve between “Dog” and all plant classes looks identical; the influence curve of “lily” and all animal classes mirrors that shape. Is this expected symmetry or some artifacts? 
3. Pointer: The text says the RMSProp-preconditioned SGLD sampler is “introduced in Section 2.2,” but the *details* are in Appendix B.

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
This paper argues for a position that training data attribution should be done dynamically, rather than statically at the end of the training. Inspired by the singular learning theory, the paper predicts a non-monotonic influence trajectories with sign flips and sharp peaks. The paper then simulates several scenarios to demonstrate their prediction.

### Strengths
The stage-wise attribution problem raised by the authors is interesting and well-presented. I am convinced that the same data point might be influential to other data in different ways at different stages of the training.

### Weaknesses
While the stage-wise framing is compelling, simply listing the most influential samples is of limited use. To establish real value, attribution should enable actionable interventions that beat a naive one-shot final stage attribution baseline. I strongly suggest that the authors please dedicate more space to what stage-wise attribution concretely enables. As of the current draft, the support is largely heuristic; adding such experiments would substantially strengthen the practical case.

### Questions
1. As the authors also brought up in the discussion section, there is a class of unrolling-based attribution methods. Can the authors propose some further discussions about these methods? Does the sign flip somehow imply that these methods are inaccurate, since the middle stage influences are canceling each other?
2. The current motivation/theoretical analysis is conducted solely based on BIF, which is a relatively new paper on this field. I wonder can the authors provide motivation in a more multifaceted manner?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the static view of data attribution, arguing that deep networks' stagewise learning makes fixed influence scores capturing only part of the picture. The authors propose a dynamic "stagewise" framework using the Bayesian Influence Function, which predicts that a data point's influence can change over the training process. This theory is validated in both toy linear networks and at-scale Pythia language models, where influence dynamics directly correlate with known developmental milestones, such as the formation of induction heads. The core conclusion is that the timing of measurement (during training) impacts the measured influence.

### Strengths
* Novel and Significant Conceptual Contribution: The paper's primary strength is the novel connection it forges between Singular Learning Theory and Training Data Attribution. This challenges a foundational assumption in TDA and provides a compelling, theory-backed explanation for why static influence measures are insufficient for deep learning.
* Strong Theoretical Grounding: The paper does an excellent job motivating its theoretical choices. It clearly explains the failure of classical IFs (reliance on an invertible Hessian) and the suitability of the BIF (Hessian-free, well-defined on degenerate landscapes). The derivation in Section 2.3, which uses the Law of Total Covariance to predict influence peaks at transitions, is elegant and provides a clear, falsifiable prediction.
* Attempt to Demonstration at Scale: The authors successfully bridge this gap by attempting to demonstrating their predicted phenomena in Pythia LMs, beyond the toy setting. While these results are correlational (linking influence dynamics to known developmental milestones like induction head formation), they are a crucial step in showing this framework is relevant for models we care about.

### Weaknesses
* Discrepancy in Observed Dynamics between Toy and Large-Scale Models: A significant discrepancy arises between the compelling theoretical predictions validated in the toy model and the empirical results from the large-scale language model experiments. The core theoretical argument hinges on influence being a highly dynamic quantity, subject to non-monotonic changes and sign-flips (as clearly demonstrated in Fig. 3). However, the influence dynamics observed in the Pythia models (Fig. 5), while temporally staged, appear to be largely monotonic once they become non-zero.

This observation materially weakens the paper's central critique of static, endpoint-based attribution or simpler trajectory-aggregation methods (e.g., TracIn). If the influence of key data groups simply increases monotonically after a specific developmental stage, then traditional attribution methods applied at or aggregated near the end of training would likely still provide a reasonable approximation of data importance. The practical necessity of the authors' far more complex, dynamically-sampled framework is therefore less evident in the very setting (LLMs) it purports to be essential for.

* Lack of Demonstrated Practical Utility or Actionable Insights: The paper compellingly argues that when a data point exerts its influence is a critical, and previously overlooked, dimension of attribution. However, the analysis remains at an observational level, and the practical utility of this framework is not demonstrated. It is unclear how these findings could inform model development, for instance, to train more capable or efficient models.

A powerful validation of the framework's utility would be to move from this observational analysis to an interventional one. For example, could the authors leverage their findings to design a dynamic data curriculum? If, as the results suggest, the influence of "induction pattern" tokens becomes active only during a specific training phase, could dynamically up-weighting these examples during (or just before) that critical phase lead to demonstrably better model performance on induction-related tasks or faster convergence? Without such a demonstration connecting stagewise influence to actionable training strategies, the proposed method, while theoretically interesting, risks being perceived as a diagnostic tool that lacks a clear feedback loop into practical model improvement.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

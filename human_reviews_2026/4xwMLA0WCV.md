# Guaranteed Optimal Compositional Explanations for Neurons

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
While neurons are the basic units of deep neural networks, it is still unclear what they learn and if their knowledge is aligned with that of humans. Compositional explanations aim to answer this question by describing the spatial alignment between neuron activations and concepts through logical rules. These logical descriptions are typically computed via a search over all possible concept combinations. Since computing the alignment over the entire state space is computationally infeasible, the literature commonly adopts beam search to restrict the space. However, beam search cannot provide any theoretical guarantees of optimality, and it remains unclear how close current explanations are to the true optimum. In this theoretical paper, we address this gap by introducing the first framework for computing guaranteed optimal compositional explanations. Specifically, we propose: (i) a decomposition that identifies the factors influencing the alignment, (ii) a heuristic to estimate alignment at any stage of the search, and (iii) the first algorithm that can compute optimal compositional explanations within a feasible time. Using this framework,  we analyze the differences between optimal and non-optimal explanations and demonstrate that 10–40\% of explanations obtained with beam search are suboptimal when overlapping concepts are involved. Finally, we evaluate a beam-search variant guided by our proposed decomposition and heuristic, showing that it matches or improves runtime over prior methods while offering greater flexibility in hyperparameters and computational resources.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates a heuristic to determine the optimal solution for compositional explanations. 
The heuristic is effective in heavily reducing the search space and thus making the search for the optimal solution feasible.

### Strengths
- The paper explores an interesting concept and the contributions are novel
- Mathematical derivations are rigor.
- The approach is competitively fast to compute with the compared approaches, although it obtains the optimal solution (a much harder task)

### Weaknesses
- **The paper would greatly benefit from a running example/illustrations of the concepts.**
- In the analysis/evaluation, the averages are computed over 50 units, but no standard deviations are provided.

There are a few notational issues in the paper: E.g., 
- l,119-120: missing {...} for M_k and N; 
- in (1): |N| is not defined as N is a matrix and not a set. However, one can infer from Tab. 1 it should be 3 in that example?
- Theorem 2.1 does not state a claim but rather just a definition, and only states the claim in the proof about the equivalence to IoU.
- Def. 2.3-2.4: Shouldn't it say: Given M_k instead of N?
- l.177: Reference to Tab. 1 is broken.
- In (3): Please use proper set building notation as in Def. 2.2-2.4.

Paper length:
- Sec. 5 is above page limits. I think this is a borderline case, as this section could also have been moved in the appendix without any issues.  Thus, either do so, or integrate it properly in the main body of the paper.

### Questions
- I'm unsure what exactly is meant by different "paths" within the paper. Aren't all trees degenerate by Assumption 2, and you can thus compute the right part exactly and always only need to explore the left part? 
- How restrictive are the assumptions in practice?

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
The paper studies the problem of optimal compositional explanations for neurons in vision CNNs. It introduces: (i) a decomposition that identifies the factors influencing the alignment, (ii) a heuristic to estimate alignment at any stage of the search, and (iii) an algorithm to compute optimal compositional explanations within a feasible time. Experiments show that the algorithm runs faster than baselines and does better at optimizing the introduced optimality metric than standard methods.

### Strengths
The paper studies an important problem

The search for better evaluation metrics of compositional explanations seems well motivated

The efficiency results in the optimal compositional explanations are compelling

The authors study 3 diverse datasets

### Weaknesses
The writing of the paper is often unclear. For example, the paper focuses on explanation of neurons in vision CNNs, although this is not clearly stated in the manuscript. In this vein, it would also be nice to see illustrative examples of compositional explanations and how they are improved by the proposed method.

The introduced dIoU metric seems sensible but it would be nice to see it better motivated and compared against existing metrics rather than than simply stated as a sequence of definitions. In particular, it would be nice to show that the metric corresponds to some human-grounded notion of interpretability.

It would also be nice to see experiments on more modern models than DenseNet (2017), including modern vision transformers.

One concern is about the potential for "Optimal" compositional explanations to potentially overfit when units are not semantically meaningful / specialized. It would be nice to see some experiments in this setting to compare how the proposed method compares with standard baselines.

### Questions
See above

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a theoretical framework for obtaining “guaranteed optimal compositional explanations” for neurons in deep networks. The authors introduce a best-first search algorithm that purportedly finds the optimal logical combination of semantic “concepts” explaining a neuron’s activation pattern. Empirical results on standard vision datasets  compare the proposed algorithm against beam-search baselines and show that it can identify explanations with slightly higher IoU (Intersection-over-Union) values.

The overall idea of moving from heuristic search to a theoretically grounded formulation of compositional explanations is interesting. In my opinion, however, the paper does not meet the expected standards of clarity and rigour required of an ICLR paper.  Fundamental concepts are undefined, the main theoretical claim (optimality) lacks any formal proof, and the mathematical presentation is inconsistent. Although the empirical work is sound, the paper as written would require a complete rewrite.

### Strengths
- The problem (understanding the alignment between neuron activations and human-interpretable concepts) is timely and relevant to XAI research.
- The authors attempt to formalise the notion of “optimal compositional explanation,” which, if properly developed, could be a valuable contribution.
- The experimental section seems carefully executed and uses appropriate datasets from prior work.

### Weaknesses
(1) Almost every foundational concept in the paper is left ambiguous or inconsistently defined.  These omissions make the entire problem formulation very difficult to follow. A non-expert reader cannot reconstruct what the proposed algorithm is even operating on. 

For example, it is never stated what kind of “neurons” are being studied—feed-forward units, convolutional filters, or something else. The experimental section reveals that the models are CNNs, but this should be stated explicitly in the introduction. 

The paper never defines what a “concept” is. From context, one can infer that concepts correspond to semantic segmentation labels (e.g., car, road), but this is not explained anywhere. Nor is it stated that the method requires densely annotated datasets. 

The term “probing dataset” appears repeatedly but is never defined. Evidently it refers to an external annotated dataset used only for neuron analysis, but this assumption is critical and must be made explicit.

 The “concept tensor” is described as “corresponding to the localization of each concept within the dataset samples.” For readers not steeped in the Network Dissection literature, this is incomprehensible. The authors seem to mean a stack of binary segmentation masks indicating where each concept occurs in each image, but this should be spelled out.

Finally, the introduction and abstract never mention that the paper is about vision and spatial data. Without that information, the framework seems domain-agnostic, though all notation (IoU, spatial locations, masks) assumes images.

(2) Several of the early definitions are problematic
 
Definition 2.1 defines “unique” and “common” elements of D as if the dataset contained duplicate entries, even though  D. The intended meaning—spatial locations annotated with one or multiple concepts—is only guessable from context.

Definition 2.2 introduces a “Neuron Activation Matrix” and “unique/common activations,” but never explains what the indices represent.

Theorem 2.1 is not a theorem but a definition of the decomposed IoU metric. It does not state any result; it simply introduces a formula. The subsequent “proof” is a straightforward algebraic equivalence that should appear as a lemma following a proper definition.

(3) It is unclear what "optimality" really means in this context. The paper never formulates an explicit optimization problem. The “optimal algorithm” is described informally as a best-first search guided by heuristic bounds, but there is no theorem proving that the heuristic is admissible or that the algorithm is complete. No formal statement or proof establishes that the algorithm returns the global optimum of any objective. The only evidence of “optimality” is that the algorithm occasionally finds slightly higher IoU values than beam search in experiments, which is empirical, not theoretical.

### Questions
None.

### Soundness
1

### Presentation
1

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
This paper presents the first framework that produces compositional explanations for neurons by effectively overcoming the long-standing dependence on beam search methods. The beam search methods do not guarantee optimality, which was the limitation of previous studies. The authors introduce a decomposition of the IoU metric (dIoU) to establish tight upper and lower bounds for candidate concepts. Building upon these bounds, they design a best-first branch-and-bound algorithm that provably identifies the globally optimal logical formula explaining all neurons within a fixed computational budget. Empirically, the study demonstrates that conventional beam search deviates from the true optimum in roughly 10–40% of cases, particularly when conceptual overlaps occur. To bridge practicality and optimality, the authors further develop a heuristic-guided beam variant that maintains competitive runtime while significantly improving efficiency. Overall, the work provides a principled and tractable path toward optimal neuron-level search and clarifies when and why heuristic beam methods fail to recover the best explanations.

### Strengths
- The proposed dIoU decomposition is both intuitive and mathematically sound, while the branch-and-bound framework offers a rigorous alternative to heuristic search by ensuring correctness and optimality.
- The paper provides clear empirical evidence of the conditions under which beam-based heuristics fail and quantifies the performance gap to the optimal solution. This helps resolve ambiguities in prior neuron interpretability research.
- Beyond theoretical contributions, the authors leverage the derived bounds to design a lightweight and heuristic-enhanced beam search. This retains much of the efficiency of standard beam methods while improving scalability and reducing computational cost for deployment in real-world interpretability pipelines.

### Weaknesses
- In Table 2, [Beam + Our H.] does not guarantee optimality but shows substantial runtime reduction. However, the paper does not report how far its performance (IoU) falls short of the Optimal IoU. In addition, Appendix A.6 presents qualitative comparisons between M-MESH and Optimal, but a quantitative evaluation (IoU) should also be reported. I would recommend augmenting Table 3 to include the IoU achieved by [MMESH Beam] and [Beam + Our H.] alongside Beam IoU and Optimal IoU, so readers can assess both efficiency and accuracy trade-offs.
- The theoretical optimality via dIoU bounds and branch-and-bound is clear, but an empirical verification would strengthen the claim. On Cityscapes, which has only 25 concepts, please brute-force and report agreement with the author’s algorithm for a subset of neurons. This would confirm global optimality not just in theory but also in practice.
- The motivation for using the optimal algorithm to better align neurons with concept compositions is ultimately to improve explanations; however, the paper does not demonstrate how much explanatory quality actually improves. There are no qualitative or quantitative results to support this.

### Questions
- The authors claim to introduce the first framework for computing guaranteed optimal compositional explanations for neurons. Can the authors more clearly position this work in relation to the large body of interpretability work (e.g., concept activation vectors, network dissection, and compositional logic explanations) and highlight what is truly new compared with previous studies?
- The paper demonstrates that between 10-40% of beam‐search explanations differ from the “optimal” ones under their metric. How important are those differences in practice for human interpretability?
- The logic operators considered are OR, AND, AND NOT. How expressive is this choice for capturing the possibly more complex behaviours of neurons (e.g., hierarchical or conditional)? Could the framework handle richer operators?
- Can the authors provide a more thorough analysis of how overlap, concept set size, and operator depth affect runtime and solution quality?
- While the method finds the “optimal” metric solution, how do we know that metric‐optimal equals interpretation‐optimal (i.e., humans find the explanation more meaningful)?

### Soundness
2

### Presentation
3

### Contribution
3

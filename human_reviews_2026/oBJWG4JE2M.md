# Equivariant Metanetworks for Mixture-of-Experts Weights

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 6, 0, 6

## Abstract
In neural networks, the parameter space serves as a proxy for the function class realized during training; however, the degree to which this parameterization provides a faithful and injective encoding of the underlying functional landscape remains insufficiently understood. A central challenge in this regard is the phenomenon of \textit{functional equivalence}, wherein distinct parameter configurations give rise to identical input-output mappings, thereby revealing the inherent non-injectivity of the parameter-to-function correspondence. While this issue has been extensively studied in classical architectures-such as fully connected and convolutional neural networks with varying widths and activation functions-recent research has increasingly extended to modern architectures, particularly those utilizing multihead attention mechanisms. Motivated by this line of inquiry, we undertake a formal investigation of functional equivalence in Mixture-of-Experts models-a class of architectures widely recognized for their scalability and efficiency. We analyze both dense and sparse gating regimes and demonstrate that functional equivalence in Mixture-of-Experts architectures is fully characterized by permutation symmetries acting on both the expert modules and the gating mechanism. These findings have direct implications for the design of equivariant metanetworks-neural architectures that operate on pretrained weights to perform downstream tasks-where reasoning about functional identity is essential. Our results highlight the importance of analyzing functional equivalence in uncovering model symmetries and informing the development of more principled and robust metanetwork architectures.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper explores the foundational concept of functional equivalence in Mixture-of-Experts (MoE) architectures and demonstrates its application to the design of equivariant metanetworks. Functional equivalence refers to the condition where distinct parameterizations of a neural network result in identical input–output mappings. The authors formalize the group actions that preserve functional behavior in MoE architectures, addressing both dense and sparse gating settings. Key theoretical contributions include characterizations of functional equivalence via permutation and affine transformations of expert and gating modules. The framework is rigorously justified through proofs based on ReLU network properties and exponential function linear independence. These theoretical insights are then applied to build equivariant metanetworks for MoE Transformers, ensuring consistent output for functionally equivalent input weights. Two new datasets—MNIST-MoEs and AGNews-MoEs—are introduced to empirically evaluate metanetworks. Experiments show that their proposed model, MoE-NFN, outperforms baselines, including Transformer-NFN, in generalization prediction. The metanetwork is shown to be robust under group action transformations, affirming its equivariance. Overall, the paper offers a principled mathematical and empirical investigation into weight-function symmetry in MoEs and how to exploit it in metanetwork design.

### Strengths
This is the first work to study theoretical characterization of functional equivalence in both dense and sparse MoE models, and develops an equivariant metanetwork architecture (MoE-NFN) grounded in theoretical insights. Rich theoretic justifications are provided.
The experiments show that MoE-NFN largely outperforms reasonable baselines. And the release of large modelzoo datasets of MoE transformers adds significant value to the paper and will help stimulate more study of metanetworks.

### Weaknesses
The gating mechanism is quite intuitive and maybe a bit simple (not necessarily a bad thing) for applying to the MoE. The equivariant layer part very much based on existing work of NFN / Transformer NFN. Also a large portion of theoretical contributions are very much based on existing work. 

Experiments limited to one task of generalization prediction of test accuracy (on two datasets). Evaluation measurement also only limited to Kendall correlation.

### Questions
- Could you tell more detail why top-1 is not covered in the proof of Theorem 3.4?

- I can probably understand using Kendall correlation for evaluation between predicted and ground-truth accuracy rankings for MoEs and that is reasonable. What about more absolute error measurements? 

- Could you clarify which from the theoretical contributions are more original and specific to MoE setups?

I put initial score on the negative side of borderline but I would be willing to raise it if my concerns are solved.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces equivariant metanetworks for Mixtures of Experts architectures. The authors formalize the symmetries in MoEs, combined with multi-head attention modules. They introduce datasets of (semi-)trained MoEs and perform experiments on predicting MoE generalization. The proposed method outperforms all competing methods in all experiments.

### Strengths
The manuscript reads very well, despite being very math heavy.
The mathematical formalism is very clear and concise as well.

The research question is very topical; metanetworks are becoming increasingly relevant in the community, and MoEs are an important neural network module. Hence, studying the symmetries in MoEs and proposing equivariant architectures is very significant.

The introduced dataset is an important addition in the model zoo datasets.

### Weaknesses
The performance gains compared to TransformerNFN are, in most cases, marginal. The benefit of incorporating MoE symmetries in the metanetwork is thus, not demonstrated convincingly.

The breadth of the experiments is also quite small, including only a single task and two datasets for evaluation.

### Questions
1. Are there any experiments (possibly on synthetic datasets) that the authors can perform to convincingly demonstrate the benefit of incorporating the MoE symmetries in the architecture? In other words, is there a setting in which TransformerNFN would perform poorly where the proposed method performs well?

2. Can the MoE symmetries be addressed with data augmentation instead of being explicitly accounted for? What is the performance of TransformerNFN trained with data augmentations on MoE symmetries?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The paper studies functional equivalence for Mixture-of-Experts (MoE) and Sparse MoEs. On the theory side, the authors give a constructive definition of a group action on these models parameter spaces. They further show that these actions are exactly the symmetries preserving functional equivalence in the gating components of MoE models. Building on this, the authors construct an equivariant metanetwork for MoE-Transformer architectures and validate it on two datasets of MoE-Transformer checkpoints.

### Strengths
- The paper has strong theoretical impact: it tackles the identifiability of MoE gating which is a fundamental yet poorly understood mechanism in modern state-of-the-art LLMs.
- The paper has strong practical impact: it introduces an equivariant metanetwork for MoE-Transformers, constructed in a principled way and grounded in the presented theory.

### Weaknesses
There are strong overlaps without proper citation in the submitted paper.  
See the bibliography at the bottom.

- Prop A.1, A.2, A.3, A.4 in [S] <-> Prop A.1 in [3].
- Lemma B.2 in [S] <-> Lemma B.3 in [3].
- Block of Def and Rmk B.4–B.6 in [S] <-> App. B.1 in [3].
- Thm B.7 in [S] <-> Thm B.5 in [3].
- Thm C.5 in [S] <-> Thm C.4 in [3].

Partial overlap and partially referenced:

- Sec. 4 and App. D [S] <-> Thm 4.3 and App. C in [1]

In particular, certain dependencies within [S] are concerning. Given the overlaps documented above, the attribution of novelty is undermined and concerns arise regarding the main contributions, which are as follows:

- Thm 3.1 is Thm B.7 in appendix of [S] which is equivalent to Thm B.5 in [3].
- Thm 3.4 is Thm C.5 in appendix of [S] which is equivalent to Thm C.4 in [3].

For context only: [1-3], all published, also exhibit non-trivial mutual overlaps, which further complicates novelty attribution in [S]. I have not investigated these in depth, as all four papers [S, 1-3] are unusually long for conference submissions, with substantial appendices. A full audit of overlapping proofs is infeasible under normal review time constraints.

I will not speculate about authorship or intent. 

**Bibliography:**  
[S] submitted paper  
[1] Tran et al., _Equivariant Neural Functional Networks for Transformers_  
[2] Tran et al., _Equivariant Polynomial Functional Networks_  
[3] Tran et al., _On Linear Mode Connectivity of Mixture-of-Experts Architectures_

### Questions
Could you address the identified weaknesses with appropriate rigor and detail?

### Soundness
4

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a new neural network architecture that is equivariant w.r.t. the symmetries in a Transformer MoE. The specific symmetries considered in this work are the permutation symmetry for the experts and the translations for the gating logits, which complements previous works that characterized the symmetries in standard Transformers. The paper shows that the proposed architecture captures all universal symmetries. The contributions include a new MoE Transformer Zoos dataset. Empirically, the proposed architecture is demonstrated to perform favorably against previous methods that lack the specific symmetries that MoE introduce.

### Strengths
- comprehensive characterization of the MoE symmetries and formal invariance proofs for dense and sparse cases
- largish scale evaluation with around 179k checkpoints
- MoE Transformers are amongst the most relevant architectures due to current frontier LLMs relying on them, so the work has high relevance

### Weaknesses
Experiments should be more extensive to better understand the benefits of explicitly accounting for the MoE symmetries. Currently, they are limited to predicting the test accuracy of MoEs. Additional experiments could for example include those considered in the weight space literature. The results on MNIST-MoEs show that the proposed network underperforms LightGBM in certain settings, which is surprising and deserves further investigation and discussion

### Questions
How often do the non-degeneracy conditions hold in the trained MoEs? What's the frequency of near-duplicate experts or linearly dependent gating differences?

### Soundness
3

### Presentation
3

### Contribution
3

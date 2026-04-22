# FS-KAN: Permutation Equivariant Kolmogorov-Arnold Networks via Function Sharing

- Avg Score: 4.33
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4, 6, 4

## Abstract
Permutation equivariant neural networks employing parameter-sharing schemes have emerged as powerful models for leveraging a wide range of data symmetries, significantly enhancing the generalization and computational efficiency of the resulting models. Recently, Kolmogorov-Arnold Networks (KANs) have demonstrated promise through their improved interpretability and expressivity compared to traditional architectures based on MLPs. While equivariant KANs have been explored in recent literature for a few specific data types, a principled framework for applying them to data with permutation symmetries in a general context remains absent. This paper introduces Function Sharing KAN (FS-KAN), a principled approach to constructing equivariant and invariant KA layers for arbitrary permutation symmetry groups, unifying and significantly extending previous work in this domain. We derive the basic construction of these FS-KAN layers by generalizing parameter-sharing schemes to the Kolmogorov-Arnold setup and provide a theoretical analysis demonstrating that FS-KANs have the same expressive power as networks that use standard parameter-sharing layers, allowing us to transfer well-known and important expressivity results from parameter-sharing networks to FS-KANs. 
Empirical evaluations on multiple data types and symmetry groups show that FS-KANs exhibit superior data efficiency compared to standard parameter-sharing layers, by a wide margin in certain cases, while preserving the interpretability and adaptability of KANs, making them an excellent architecture choice in low-data regimes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FS-KAN, a framework that enforces permutation equivariance and invariance in Kolmogorov–Arnold Networks by sharing functions rather than weights. The authors establish an expressivity equivalence with parameter-sharing ReLU MLPs, showing that FS-KANs match their expressivity and thus inherit known universality guarantees for symmetry aware models. Empirically, across multiple symmetry tasks, FS-KANs outperform other baselines in low-data regimes, delivering better data efficiency while retaining KANs interpretability.

### Strengths
- The paper proves that FS-KAN captures all G-equivariant/invariant KA layers via function sharing. The proof is easy to follow with standard techniques in equivariant/invariant layer design.
- Emperical evident suggests that their FS-KAN is better comparing to other baselines in low data regime.

### Weaknesses
- In the point-cloud task (e.g., at ~1k training samples), other baseline match or surpass FS-KAN, suggesting its benefit diminishes as training size grows. This reduces the practical appeal of FS-KAN beyond low data settings.
- The computational runtime/memory analysis from Appendix C shows that FS-KAN incurs substantially higher runtime and memory usage than the other baselines.

### Questions
- Does function sharing over-constrain FS-KAN? With many spline functions tied across positions (Appendix Fig. 7), could the sharing pattern overly limit capacity and hinder learning in practice?
- The appendix indicates substantially higher runtime and memory than other baselines. Can runtime/memory for FS-KAN be improved?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes FS-KAN and function sharing, enabling Kolmogorov–Arnold Networks (KAN) to implement group equivariance. This design unifies parameter sharing and functions sharing enabling transfer of known results (e.g., universality). Experiments show competitive performance, especially in low-data regimes.

### Strengths
- Well written and clearly structured with working examples.
- Proposition 2 is a rigorous result which requires new and different proofs from the ones known in the weight sharing literature for equivariant neural networks. 
- The authors propose a perspective that unifies neural networks and KANs. Function sharing for KANs cleanly recovers common equivariant designs for permutation representations of groups.

### Weaknesses
- **Insufficient novelty.** The introduction of the concept of function sharing is a simple adaptation of the regular concept of weight sharing for equivariant neural networks, as presented in [1]. The majority of main theoretical statements except for Proposition 2 largely transfer as it is from the weight sharing literature for equivariant deep learning.
- **Missing analysis of low-frequency learning and spectral bias.** This topic is currently central to KAN research [2]. In particular, KANs are known to avoid the low-frequency bias of standard neural networks, enabling them to perform better on high-frequency targets (e.g., certain PDE solutions). These properties make KANs especially relevant for AI-for-Science; yet the manuscript does not investigate this direction, even though enforcing equivariance to the symmetries of specific PDEs could potentially improve performance.

[1] S. Ravanbakhsh, *Equivariance Through Parameter-Sharing*, 2017 \
[2] Y. Wang et al., *On the expressiveness and spectral bias of KANs*, 2024

### Questions
**Interpretability claim.**  
One of KANs’ most advertised features is improved interpretability, and FS-KANs build on this. However, this claim is contested: learnable activations are not identifiable up to scale and permutations, which makes post-hoc interpretation fragile at the local scale. At larger scales, even the KAN authors acknowledge that “a KAN may only remain interpretable when the network scale is relatively small” (p. 21 of [3]). 
To what extent does enforcing equivariance mitigate this fragility?

[3] Z. Liu, "KAN 2.0: Kolmogorov-Arnold Networks Meet Science", 2025

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces FS-KAN, a principled framework for building permutation-equivariant and invariant neural networks by extending parameter-sharing schemes to function sharing. The approach unifies and generalizes prior work on equivariant KANs, providing a theoretical proof that FS-KANs have equivalent expressive power to traditional parameter-sharing MLPs. Empirically, FS-KANs achieve strong performance and data efficiency across diverse symmetry-aware tasks, including signal classification, point cloud recognition, and matrix completion, particularly in low-data regimes. The architecture retains the interpretability and adaptability of KANs while significantly improving generalization for data with built-in symmetries

### Strengths
- The paper presents a theoretically sound and general framework for permutation-equivariant KANs, extending the idea of parameter sharing to function-level sharing. It also provides clear mathematical proofs showing expressivity equivalence with parameter-sharing MLPs.
- The proposed architecture maintains the interpretability of KANs, as demonstrated in visual examples where symmetric function structures emerge naturally.
- The efficient FS-KAN variant reduces computational and memory costs while preserving equivariance, making the method more practical.

### Weaknesses
- Despite the good performance, computational cost remains high, particularly compared to standard MLP-based models in high-data regimes.
- The empirical analysis, while broad, relies mostly on synthetic or small-scale datasets, leaving scalability to large real-world systems untested.
- The practical interpretability claims are qualitative; the paper lacks systematic metrics quantifying how function sharing aids transparency.

### Questions
- How does FS-KAN training scale computationally and memory-wise with input dimension and group size, compared to DeepSets or GNN baselines?
- Can FS-KANs be adapted for continuous or mixed-group symmetries (e.g., rotations, reflections) beyond discrete permutations?

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
3

### Summary
The paper introduces a new approach to realize equivariant (or invariant) KANs. The underlying mechanism is a direct generalization of the weight-sharing used for MLPs, where now it becomes a function sharing of the learnable KAN functional components.
The work claims that the approach generalizes existing equivariant KAN architectures, and that the function-sharing approach is general, since any equivariant KA layer can be exactly represented by a FS-KA layer.
The authors then introduce an efficient version of the layer, provide examples intended to demonstrate how FS-KAN can be designed and how they generalize existing architectures, and prove that MLPs and FS-KANs can approximate each other to arbitrary accuracy. This fact is then used to derive universality properties of FS-KANs.
The experimental section convincingly demonstrate the potential of the new architecture.

### Strengths
- The related work clearly highlights the motivation of the work (simplification of existing approaches, variable size inputs, general permutation groups).
- The topic of combining KANs with equivariant modelling is of current interest, and the results cover both efficient implementation and analytical properties.
- The experimental section is clearly structured and it addresses a clearly defined set of questions.

### Weaknesses
- Efficient FS-KA layers: This seems to be a central novelty of the paper, as it enables applications and scaling to realistic problems. However, their definition is mainly textual and hard to process. I suggest to revise the paragraph "Efficient FS-KA layers" and make their definition more explicit.

- Generalization of the existing literature: There is an emphasis in Sections 1-2 on the fact that FS-KANs generalize and unify existing approaches. This fact is however not sufficiently demonstrated in the paper. Section 3.3 in particular has a "FS-KANs generalize previous equivariant KANs" paragraph, which only mentions some similarity with the existing literature. 

- Theoretical analysis: Proposition 7 is formulated in an imprecise manner, as there is no dependency on the target accuracy $\varepsilon$, while clearly the MLP $f$ is in fact $f(\varepsilon)$. This becomes evident by inspecting the proof, but it should made clear upfront. 
Especially, the width $d_{\max}$ of the MLP $f$ seems to be unboundedly growing in $\varepsilon$, since the construction is based on the 1989 Hornik et al. result. 
Apart from updating the statement of Proposition 7, I also recommend to update the following sentences:  (I) "Both proofs provide constructive methods for building the corresponding networks",  following Proposition 7, which is not correct as it stands; (ii) "we show that for a specific permutation group G, both model families have the same expressive power in the uniform approximation sense", should rather state the the two families are dense in the same set of functions.

- The experimental results compare FS-KANs only with MLPs with shared parameters. Comparisons with other equivariant architectures should be considered.

### Questions
Apart from the points discussed above, there are the following minor points: 
 
- The $\star$ notation in (1) is standard in KANs, but should be briefly introduced.

- Paragraph "Invariance, equivariance, and standard parameter-sharing": The problem is formulated for $V=V'=R^n$, instead of general $V=R^n$, $V'=R^{n'}$. It is unclear at this stage if this is a necessary assumption for the definition of FS-KA layers, or if the assumption is just for the sake of simplifying the presentation. Please clarify this.

- Equation (3) gives the wrong impression that only constant, zero-sum offsets are allowed. While this is clear from the context, I suggest modifying the example in (3) by using a functional offset.

- Proof of Proposition 6: The notions of grid size and $B$-splines ("grid size $G = 2$ with degree $k$ B-splines") are nowhere defined before.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces FS-KAN, a principled framework for building permutation-equivariant and invariant KAN layers by sharing functions (rather than weights) across positions. The authors prove that FS-KANs have expressive power equivalent (on bounded domains) to parameter-sharing MLPs; this lets known universality and expressivity results for equivariant networks (e.g., DeepSets/GNN variants) transfer directly. Empirically, across point-cloud classification (ModelNet40, including a continual-learning protocol) and recommender datasets with $S_n \times S_m$ symmetry, FS-KANs show clear gains in low-data regimes and reduced forgetting, while preserving KAN interpretability. The paper also presents an “efficient” FS-KAN variant that reduces (but does not eliminate) runtime/memory overheads. Overall, the work unifies scattered equivariant-KAN designs and offers a clean blueprint to apply KA layers under arbitrary permutation groups.

### Strengths
- This paper presents an elegant generalization of parameter sharing to function sharing in KANs; the framework subsumes prior set/image KANs and handles arbitrary permutation groups and multi-channel tensors.

- The authors proves interesting bidirectional results: on the one hand, FS-KAN can be used to represent ant parameter-sharing MLPs; on the other hand, parameter-sharing MLPs can be used to approximate any FS-KANs. These results clarify exact expressivity and transfer known universality/limits from parameter-sharing MLPs to to FS-KANs.

- On low-data point-cloud and recommendation settings, FS-KANs deliver consistent improvements in accuracy/RMSE and lower forgetting in continual learning.

### Weaknesses
My main concern with this paper lies in the weaknesses of the experimental evaluation.

- Regarding compute/latency, even the efficient variant is ~1.5× faster than vanilla FS-KAN yet still slower than DeepSets, and its memory usage remains higher.

- Figure 5 and Table 1 show that FS-KAN’s improvements over the baselines are marginal; if anything, they suggest parity with older baselines such as Point Transformer.

- No large-scale benchmarks are discussed. Given the high computational overhead and only marginal gains over DeepSets, it is plausible that DeepSets may slightly surpass FS-KAN at larger training sizes.

However, because the theoretical results constitute the paper’s main contribution, and are both interesting and valuable to the ML community, I assign a positive score at this time.

### Questions
- What is the impact of the number/order of knots, the function parameterization (splines vs. alternatives), and the degree of sharing (full vs. partial) on performance and computational cost?

- How would the theoretical results change if the permutation group were replaced by a continuous group (e.g., rotations)?

- Is there an intuitive explanation for why FS-KANs can exactly represent any parameter-sharing MLP, whereas parameter-sharing MLPs can only approximate, but not exactly represent, arbitrary FS-KANs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 6

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a permutation-equivariant neural network based on the Kolmogorov–Arnold framework.

### Strengths
- The paper is well written and self-contained. Even for readers who are not deeply familiar with KANs, the main ideas are clear and accessible.

- The construction of the proposed permutation-equivariant and permutation-invariant KANs is clearly presented. It is supported by solid theoretical guarantees, including expressivity results (Propositions 2 and 4) and expressive power analysis (Section 4). The accompanying visualizations effectively illustrate the concepts, and to the best of my knowledge, this formulation is novel within the context of permutation-equivariant networks.

- The experimental section covers a diverse set of symmetry-structured tasks, such as signal classification, point-cloud classification, and semi-supervised rating prediction, demonstrating the versatility of the proposed framework.

### Weaknesses
- While the theoretical results are clearly presented and technically sound, they are somewhat expected and incremental. Propositions 2 and 4 follow naturally from basic algebraic reasoning, and the expressive power result largely derives from existing results on standard KANs.


- I find the motivation from the KAN literature somewhat confusing. KANs are often considered computationally heavy and inefficient compared to mainstream architectures such as MLPs, CNNs, or Transformers. Although spline-based parameterizations can alleviate some of these issues, KAN-based models are still known for their slower training and higher memory consumption. While the paper provides several theoretical guarantees on expressivity, I believe the empirical performance and efficiency aspects are more crucial in this context. **The trade-off between accuracy and computational cost remains too large**. This makes the practical necessity of applying KANs to permutation-equivariant networks questionable. Although the authors introduce an efficient variant of FS-KAN to mitigate these issues, the improvement is modest, and in some cases, the efficient version performs comparably or even worse than the full FS-KAN model.

### Questions
Could the authors provide additional evidence or comparisons regarding runtime and memory efficiency with other baselines? Currently, the analysis only includes DeepSets, making it difficult to assess how FS-KAN performs relative to a broader range of architectures.

### Soundness
2

### Presentation
3

### Contribution
2

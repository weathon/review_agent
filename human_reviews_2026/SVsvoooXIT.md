# Learning Invariances for Causal Abstraction Inference

- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Causal abstraction inference is the task of inferring causal effects from limited data by first mapping the complicated low-level data (e.g., pixels) into a simpler high-level space (e.g., image representation) before performing causal inferences on the high-level. A major restriction in this task is known as the abstract invariance condition (AIC), which forces high-level representations to retain all information from the low-level data to prevent any ambiguity in high-level inference. In this work, we provide the first approach that can learn low-dimensional high-level representations that satisfy the strictest form of the AIC without weakening the allowable causal inferences. We show how the concept of invariances, such as rotational invariance in image data, is related to causal abstractions and how they can be used to learn lower dimensional representations using out-of-the-box invariance learning tools such as contrastive learning. Finally, we demonstrate our findings empirically, including in a high-dimensional image setting.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper connects invariance learning to causal abstraction inference. It argues that invariances can help learn low-dimensional representations that satisfy the abstract invariance condition (AIC). The authors propose using contrastive learning to achieve this and test it on two small examples — a voting toy setup and a semi-synthetic pneumonia dataset.

### Strengths
- The paper provides a detailed background on causal abstraction and invariance concepts.
- There is an effort to make the connection between invariance and causal abstraction
- Experiments are easy to follow

### Weaknesses
- The method is basically a standard contrastive learning setup applied to causal abstraction — not much novelty in the learning approach.
- Experiments are very limited and “toyish,” offering little real empirical insight.
- The paper spends a lot on preliminaries but does not deliver a strong new idea or technical depth.

To conclude, I don t think that the takeaway using Contastive Learning to learn invariance is novel.

### Questions
Please address the points i have raised as weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper focuses on learning causal abstractions, which are compressed representations of a causal system that preserve the ability to answer causal queries (interventions, counterfactuals) correctly. The authors propose to learn invariances that implicitly satisfy the Abstract Invariance Condition (AIC), which is a well-known requirement that ensures that high-level interventions correspond to consistent low-level interventions. Experiments are conducted on a synthetic voting dataset and real-world X-ray images, demonstrating the improvement of the proposed contrastive approach.

### Strengths
1. **Novel Theoretical Integration.** The paper provides a rigorous link between structural invariances and the Abstract Invariance Condition (AIC) in causal abstraction inference. It establishes that invariance functions can be leveraged to construct lower-dimensional representations that still satisfy the strict AIC, while providing an interpretable objective with sound theoretical guarantees.

2. **Practical Empirical Validation.** The improvements are consistent across embedding sizes and baselines.

### Weaknesses
1. **Strong Assumptions** The approach critically assumes known invariance information (Def. 7). As acknowledged in the discussion, this assumption is rarely realistic outside synthetic or well-controlled domains. Without it, the method cannot guarantee AIC satisfaction, limiting real-world applicability.

2. **Limited Baseline Evaluations**. The experiments focus on small-scale synthetic setups and a single real-world dataset (low-resolution chest X-rays), comparing the constructive and original versions of RNCM. There is no comparison with other existing causal representation learning approaches. Notably, "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style", which also extracts invariant content using contrastive learning.

3. **Contrastive Learning Practicality.** The contrastive training setup (Eq. 3) assumes sufficient batch diversity and correct pairing of equivalent samples. In practice, as noted, violations of these conditions could severely hinder convergence or lead to degenerate clusters. The work does not quantify the sensitivity of the results to these violations.

4. **Simplified Graphical Assumptions.** The approach relies on a predefined C-DAG, which can further restrict its real-world usability.

### Questions
1. **Scalability.** How does the proposed method scale to high-resolution or high-dimensional data? 

2. **Invariances.**  Can the invariance discovery be automated rather than assumed? What happens if the assumed or discovered invariance set is partially wrong or incomplete?

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
3

### Summary
This paper proposes theorems and practical approaches to solve the causal abstraction inference problem, in which the abstract invariance condition (AIC) is a major restriction to be tackled. The theorem can be validated from toy datasets and chest-X ray medical image datasets.

### Strengths
+ The research problem and main focus are well introduced with substantial knowledgeable context such as the Figure 1 that clearly shows the problem setting of causal abstraction inference. 

+ The proposed theorem-1 seems to be important, as it points out a new direction to enable SCM to satisfy AIC from the perspectives of the intervariable clusters and max invariance clusters.

+ A practical use case and empirical approach are provided to demonstrate the advantages of the proposed theorem. Specifically, a contrastive learning approach is found useful to learn invariances when training the existing RNCM model.

### Weaknesses
- The Introduction part does not present an overview of the proposed method, as well as how the method could achieve the stated merits as listed in the last paragraph on page-2.

- In the pneumonia experiment, although the chest-X ray images are from real-world applications, the invariances of translation, zooming, cropping, etc., are still synthetic or unrealistic in practice. It could be better to find more practice invariances.

- Technically speaking, the proposed contrastive learning method is nothing new, though this paper presents its advantages of learning the invariances.

### Questions
No additional questions.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a theoretical and algorithmic framework connecting invariance learning with causal abstraction inference. The approach presented aim at learning representations that preserve the Abstract Invariance Condition (AIC), which ensures that high-level abstractions preserve causal effects from their low-level mechanisms, and proves that structural invariances (e.g., rotations or permutations) define maximal invariance clusters that satisfy AIC while permitting dimensionality reduction (Theorem 1).
Building on this, the authors implement a contrastive learning approach within a Graph-Constrained Representational Neural Causal Model (GC-RNCM). By treating invariant transformations as positive pairs, they show that the resulting representations recover AIC-compliant clusters (Theorem 2).
Empirical validation is performed on synthetic (“Voting dataset”) and semi-synthetic (“Pneumonia dataset”) datasets show that these invariant embeddings improve causal-query estimation.

### Strengths
* **Good theoretical contribution.**
  The connection between structural invariances and causal abstractions is novel and rigorously established by Theorems 1–2 and Corollaries 1–3. These results are relevant to ground causal abstractions theory to practical contrastive learning algorithms. Moreover  the notion of maximal invariance clusters offers an intuitive, interpretable link between dimensionality reduction and causal faithfulness.


* **Presentation clarity.**
  The theory is presented in a clear and intuitive way: Examples (voting, rotated images) and diagrams (Figs. 3–6) build intuition and help the reader go through the theoretical statements. The definitions and notation are precise and consistent with existing causal-model semantics.

### Weaknesses
* **Gaps between theory and practice**: one of the main weaknesses of the paper are the gaps between the assumptions in the theoretical setting and real settings. While it is not expected to fully bridge these gaps in one paper, the risk is that the method developed only can work in very unrealistic and ideal settings. Below I detail these gaps: 
   -    **Reliance on assumed invariances.** The framework presumes that a correct set of structural invariances is known a priori. The authors acknowledge this in the limitations section in the Appendix, but they do not analyze how approximation or misspecification of $\mathcal{I}$ affects AIC satisfaction or downstream causal inference. Since the end goal motivation is to apply the method in realistic domains, the lack of robustness or sensitivity analysis leaves this assumption hard to be verified in practice.
  -   **Unverifiable AIC in practice.**
  The appendix correctly notes that AIC cannot be empirically verified because the true structural mechanisms $f_V$ are unobserved. However, the paper does not explore measures or proxies (e.g., cross-intervention consistency) that could indicate when learned embeddings fail to preserve causal distinctions. For example, two low-level samples with distinct causal effects but similar invariant transformations, such as rotated X-ray images of different pathologies, could be mapped to the same embedding, yielding good predictive performance yet violating AIC. The issue is acknowledged but not addressed in depth.




* **Limited experimental scope** Experiments focus on two small-scale datasets (“Voting” and “Pneumonia”) and lack ablations over key parameters such as temperature, or invariance strength. The authors mention data limitations in Appendix B, but a broader empirical validation, especially on larger or multimodal settings, would be needed to demonstrate generality of method even when the assumptions hold just approximately.
    -    **Scalability and robustness not evaluated**.  The paper provides no discussion or empirical evidence on computational scaling with continuous invariance groups or higher-dimensional data. This limits again the applicability of the method. 


* **Missing Related Work**
Several important directions are missing from the related work discussion:
     - **(1) Invariances and Identifiability in Causal Representation Learning.** The paper would benefit from connecting its theoretical contribution to recent works that explicitly link invariance principles with identifiability. For example:  [a] establishes that invariance constraints are both necessary and sufficient for identifying causal factors.  [b]: shows that grouping observed variables under structural constraints enables identifiability, conceptually akin to discovering invariant clusters. Despite not developing theory directly for causal abstractions, these works emphasize that structural or statistical invariances can serve as sufficient conditions for causal identifiability and need to be discussed. 


     - **(2) Contrastive Learning for Disentanglement** While the paper leverages contrastive learning to enforce AIC-compliant abstractions, it omits prior works that exploit contrastive or hierarchical objectives to achieve disentanglement and invariance. For instance: [c]  proposes a contrastive manifold-projection loss that isolates independent latent factors, conceptually related to invariant causal clusters. [d] demonstrates that contrastive learning can recover latent factors consistent with the underlying generative process.[e] provides theoretical guarantees that contrastive learning with data augmentations separates invariant and variant components.

Beyond disentangled representations, hierarchical or multilevel contrastive frameworks offer direct analogies to multi-level causal abstraction and could serve as meaningful empirical baselines: [f] introduces a hierarchical contrastive objective across label levels, mirroring multi-tier causal abstraction. [g] employs hierarchical prototypes as higher-level anchors for contrastive learning, analogous to abstract variables summarizing lower-level clusters.  [h] learns hierarchical representations in hyperbolic space, providing an example of compositional abstraction akin to causal hierarchies.
Including or contrasting against such methods would help clarify how the proposed GC-RNCM framework differs from existing hierarchical or disentanglement-based contrastive approaches.


_[a] Yao et al. (2024). *Unifying Causal Representation Learning with the Invariance Principle.* ICLR 2025._

_[b] Morioka & Hyvärinen (2023). *Causal Representation Learning Made Identifiable by Grouping of Observational Variables.*_

_[c] Fumero et al. (2021). *Learning Disentangled Representations via Product Manifold Projection.* ICML 2021._

_[d] Zimmermann et al. (2021). *Contrastive Learning Inverts the Data-Generating Process.* NeurIPS 2021._

_[e] von Kügelgen et al. (2021). *Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style.* NeurIPS 2021._

_[f] Zhang, Shu, et al. (2022). *Use All the Labels: A Hierarchical Multi-Label Contrastive Learning Framework.* CVPR 2022._

_[g] Li et al. (2021). *Prototypical Contrastive Learning of Unsupervised Representations.* ICLR 2021._

_[h] Pal, Avik, et al. (2025). *Compositional Entailment Learning for Hyperbolic Vision-Language Models.* ICLR 2025._



While I believe the theoretical contribution is of value, I also think that the issues above should be addressed or alternatively the scope of the paper should change accordingly and contributions on experiments and adaptability to practical settings should be tuned down.

### Questions
* Could the invariances be learned adaptively, e.g., through group-equivariant or augmentation-discovery architectures?
* How robust is performance to partially valid or noisy invariances?
* Can Theorem 2’s assumptions (“sufficient batch diversity”) be quantified empirically, e.g., through contrastive mutual-information bounds?

### Soundness
3

### Presentation
3

### Contribution
2

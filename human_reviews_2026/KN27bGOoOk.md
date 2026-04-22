# Faithful and Stable Neuron Explanations for Trustworthy Mechanistic Interpretability

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Neuron identification is a popular tool in mechanistic interpretability, aiming to uncover the human-interpretable concepts
represented by individual neurons in deep networks. While algorithms such as Network Dissection and CLIP-Dissect achieve great empirical success,
a rigorous theoretical foundation remains absent, which is crucial to enable trustworthy and reliable explanations. In this work, we observe that neuron identification can be viewed as the *inverse process of machine learning*, which allows us to derive guarantees for neuron explanations. Based on this insight, we present the first theoretical analysis of two fundamental challenges: (1) **Faithfulness:** whether the identified concept faithfully represents the neuron's underlying function and (2) **Stability:** whether the identification results are consistent across probing datasets.  We derive generalization bounds for widely used similarity metrics (e.g. accuracy, AUROC, IoU) to guarantee faithfulness, and propose a bootstrap ensemble procedure that quantifies stability along with **BE** (Bootstrap Explanation) method to generate concept prediction sets with guaranteed coverage probability. Experiments on both synthetic and real data validate our theoretical results and demonstrate the practicality of our method, providing an important step toward trustworthy neuron identification.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper tackles neuron identification through an inverse-learning lens: instead of training a model to fit data, it “identifies” a concept 𝑐 that best matches a fixed neuron (or neuron set) 𝑓. The authors formalize faithfulness as population-level similarity sim(𝑓,𝑐) and prove uniform generalization bounds showing that maximizing empirical similarity on a probe set yields near-optimal population similarity. They further introduce a stability notion via a bootstrap wrapper that produces prediction sets of candidate concepts with a provable coverage lower bound. The theory is instantiated for common similarity metrics (e.g., accuracy, AUROC, IoU), yielding rate expressions that translate into practical sample-size and metric-selection guidance. Empirically, they wrap standard neuron-identification methods (e.g., NetDissect/CLIP-Dissect–style pipelines) to illustrate the approach.

### Strengths
The authors identifie a genuine gap—neuron identification lacked a defensible definition of faithfulness—and provides a principled formulation together with a stability construct. Treating the task as an inverse problem is a fresh angle that connects interpretability with classical generalization theory. Moreover, it reframes neuron identification as an inverse problem with explicit statistical guarantees—novel for this literature.

The theory is sound and broadly applicable: uniform convergence results are presented for several practical similarity metrics, turning into actionable guidance on sample size and metric choice (e.g., dependence on concept frequency). The stability wrapper is clean, black-box, and yields interpretable prediction sets with coverage guarantees, enabling immediate integration with existing tools.

Overall, I believe the contributions of this paper exceed acceptance bar. However, due to following weakness, the authors should fix some issues to be accepted.

### Weaknesses
Empirical breadth and depth are insufficient.
- Experiments are concentrated on a single architecture and a narrow set of baselines; this limits external validity. Including diverse backbones (e.g., ViT/ConvNeXt, larger CNNs) and multiple concept sources would better test generality.
- The paper promises additional results in the appendix C, but I could not find no additional results at figure 6 of appendix C
- The authors should include qualitative results on how faithfulness difference affects the neuron identification quality.

Compute/efficiency analysis is missing.

- The bootstrap wrapper introduces a multiplicative factor in runtime (by 𝐾 resamples) and scales with the concept vocabulary size ∣𝐶∣. There is no systematic study of wall-clock time, memory footprint, or accuracy–efficiency trade-offs (e.g., coverage vs. 𝐾, or performance vs. ∣𝐶∣).

### Questions
See weaknesses above.

### Soundness
4

### Presentation
2

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
This paper focuses on the problem of neuron identification for explanation, where the lack of formal theoretical analysis undermines the trustworthiness and reliability of existing neuron explanation methods. To overcome this limitation, the paper introduces a theoretical framework that views neuron identification as the inverse process of machine learning, enabling the adaptation of tools from statistical learning theory to establish theoretical guarantees for the faithfulness and stability of neuron explanations. The paper further conducts synthetic simulations and experiments on real-world datasets using existing neuron identification approaches to verify their theoretical findings.

### Strengths
1. The paper provides a theoretical analysis of neuron explanations, establishing formal bounds for various concept–neuron similarity metrics to ensure faithfulness, and deriving concept prediction probabilistic guarantees for stability.
2. The idea of treating neuron identification as an inverse machine learning process is a novel insight that enables the adaption of the generalization theory and helps justify the reliability of neuron explanations.

### Weaknesses
1. Although the paper presents explicit theoretical analyses, the empirical validation is limited and does not sufficiently demonstrate the effectiveness of the proposed theorems. For faithfulness, only simple binary cases and synthetic simulations are provided; for stability, the paper includes only two visualization examples, and the additional results in the Appendix appear the same to those in the main paper.
2. The paper would benefit from a more comprehensive empirical verification, including comparisons between theoretical and experimental results on real-world datasets, with quantitative and qualitative evaluations. Such analyses would substantially strengthen the support for the theoretical claims.
3. Several formal definitions of the theorems are missing, such as the convergence rate function. In addition, the simulation study details for experiments 1 and 2 are not provided in the main paper or the Appendix.

### Questions
1. How scalable are the similarity metrics listed in the paper? How would the proposed framework extend to other metrics, such as WPMI [1] and MAD [2]?
2. Could the paper discuss how the theoretical guarantees for neuron identification will regularize or guide the development of future neuron explanation methods?

[1] CLIP-Dissect: Automatic Description of Neuron Representations in Deep Vision Networks. ICLR 2023.

[2] CoSy: Evaluating Textual Explanations of Neurons. NeurIPS 2024.

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
3

### Summary
Recent advances, such as Network Dissection and CLIP Dissection, have achieved remarkable progress in neuron identification. However, these approaches still lack a theoretical foundation for ensuring faithful and stable explanations. The authors observe that neuron identification is similar to the inverse process of machine learning. Building on this insight, authors derive theoretical guarantees for neuron-level explanations and present theoretical analyses addressing the challenges of (1)Faithfulness and (2)Stability in neuron interpretation.

### Strengths
- This work presents a novel and insightful perspective by interpreting the neuron identification problem as an inverse process of machine learning. This conceptual shift provides a fresh way to reason about how neurons emerge and can be systematically explained within learned models.
- This work strengthens the theoretical foundation of neuron interpretation by addressing both faithfulness and stability. It provides analytical support for the reliability of neuron-level explanations and offers theoretical clarity beyond empirical observation.

### Weaknesses
- This work concludes that CLIP Dissect tends to identify more abstract concepts, whereas NetDissect captures more concrete ones. However, it remains unclear whether the probing set and concept set used for this comparison are consistent across both methods. Specifically, CLIP Dissect employs images from the model’s training data as the probing set and utilizes a concept set of approximately 2K words, while NetDissect uses the Broden dataset, which includes segmentation mask annotations but contains far fewer samples. Therefore, when applying the proposed Bootstrap Ensemble procedure under the original settings of each method, the resulting bootstrap datasets and outcomes are unlikely to be derived from identical conditions. A clarification on this point would strengthen the validity of the comparison.
- The paper introduces five similarity metrics, accuracy, AUROC, IoU, recall, and precision, as general evaluation measures. However, the applicability of each metric highly depends on the label type of the probing set and concept set. For example, IoU typically requires pixel-level ground-truth segmentation masks, making it suitable for NetDissect (which uses the Broden dataset) but not directly applicable to CLIP-Dissect, which lacks such annotations. Therefore, it would be beneficial to clarify whether different metrics are selectively used depending on the label types of the probing and concept sets if all five metrics are claimed to be generally applicable, to provide additional explanation on how each metric is adapted to different concept identification methods.

### Questions
- How broad is the applicability of the approach (e.g., beyond vision, beyond single neurons, beyond simple concept sets)?
- The focus is on individual neurons and concept alignment. But neurons may be polysemantic. Or representations may be distributed. How does that affect the validity of using single-neuron explanations?
- The derivation of generalisation bounds for metrics such as IoU and AUROC is an interesting contribution. Are the bounds tight/meaningful in practice?
- What are the implications of the proposed method for downstream interpretability applications (e.g., model auditing, bias detection, editing networks)?
- How many neurons/explanations were evaluated in the empirical section?

### Soundness
3

### Presentation
3

### Contribution
3

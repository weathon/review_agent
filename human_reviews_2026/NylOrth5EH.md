# Degree-Conscious Spiking Graph for Cross-Domain Adaptation

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Spiking Graph Networks (SGNs) have demonstrated significant potential in graph classification by emulating brain-inspired neural dynamics to achieve energy-efficient computation. However, existing SGNs are generally constrained to in-distribution scenarios and struggle with distribution shifts. In this paper, we first propose the domain adaptation problem in SGNs, and introduce a novel framework named Degree-Consicious Spiking Graph for Cross-Domain Adaptation (DeSGraDA). DeSGraDA enhances generalization across domains with three key components. First, we introduce the degree-conscious spiking representation module by adapting spike thresholds based on node degrees, enabling more expressive and structure-aware signal encoding. Then, we perform temporal distribution alignment by adversarially matching membrane potentials between domains, ensuring effective performance under domain shift while preserving energy efficiency. Additionally, we extract consistent predictions across two spaces to create reliable pseudo-labels, effectively leveraging unlabeled data to enhance graph classification performance. Furthermore, we establish the first generalization bound for SGDA, providing theoretical insights into its adaptation performance. Extensive experiments on benchmark datasets validate that DeSGraDA consistently outperforms state-of-the-art methods in both classification accuracy and energy efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the problem of domain adaptation in spiking graph networks, an issue that has not been sufficiently explored in the existing literature. The authors propose the DeSGraDA framework, which extends spiking graph networks (SGNs) through three key mechanisms: degree-based spike encoding, adversarial training in membrane potential space for temporal distribution alignment, and a pseudo-label distillation mechanism based on prediction consistency. The method is supported theoretically by a generalization error bound derived for the SGN domain adaptation (SGDA) scenario. Experiments on multiple benchmark datasets demonstrate that DeSGraDA outperforms various baseline methods in both classification performance and energy efficiency.

### Strengths
1. This paper introduces domain adaptation into spiking graph networks, aligning the distributions of source and target domains, which demonstrates strong novelty. 

2. To address key challenges in SGNs—such as node degree bias, temporal dynamic distribution discrepancies, and missing labels in the target domain—the paper proposes the DeSGraDA framework, comprising three modules: degree-aware spiking, temporal adversarial alignment, and pseudo-label distillation. The design is logically coherent and the technical approach is well-justified.

3. The paper not only provides theoretical analysis such as a generalization error upper bound, but also conducts experiments on multiple datasets. The ablation studies and comparative results effectively support the method's effectiveness and superiority.

### Weaknesses
1. The pseudo-label distillation module clusters target samples and assigns pseudo-labels based on shallow graph features, but the criteria for assessing the reliability of the clustering are not clearly defined.

2. The trade-off between the additional computational overhead introduced by domain adaptation and the resulting performance improvement is not discussed.

3. Some terminology is inconsistent (e.g., "degree-conscious" vs. "degree-aware").

### Questions
1. In the degree-aware spiking representation module, the authors adaptively set the spiking thresholds based on node degrees to achieve more balanced information propagation across the graph. However, a similar adaptive threshold mechanism was also proposed in SpikeNet (2023). What are the essential differences and advantages of the proposed method compared to prior work, in terms of mechanism design, theoretical motivation, or practical performance.

2. In the temporal distribution alignment module, does the adversarial discriminator operate on all historical membrane potentials, or on a final feature vector obtained by aggregating membrane potentials across multiple time steps?

3. In pseudo-label distillation, what is the rationale for selecting the number of clusters C? Are there alternative clustering and label assignment strategies that could be used?

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
4

### Summary
The paper proposes S2GCL, a graph contrastive learning (GCL) framework that couples a spectrum‑aware spiking encoder with dual‑level contrast. The key mechanisms are Spectrum‑aware Membrane Potential (SaMP), Overlapped Channel Grouping (OCG), and a dual contrastive objective. The authors validate the effectiveness of the propose approach on various datasets, such as Cora, CiteSeer, PubMed, Photo, Computers, and CS. According to their experimental results they achieved superior accuracy on those datasets.

### Strengths
1. Strong average performance across diverse benchmarks. In Table 1 (p. 8), S2GCL outperforms prior SNN‑GCL (e.g., SPIKEGCL) and supervised/unsupervised GNN baselines on all six datasets.
2. SaMP is easy to plug in. The learnable projection that maps eigenvalue features to initial membrane potential is simple and compatible with standard message‑passing layers and LIF neurons.

### Weaknesses
1. Time‑step dependence of SaMP is not analyzed. SaMP only changes the initial membrane potential. Thus, as T grows and multiple integrate‑fire‑reset cycles occur, SaMP’s effect may diminish. The paper studies T vs. accuracy and runtime globally (Fig. 4) but does not isolate how SaMP’s contribution scales with T.
2. Ablation granularity is limited. Figure 3 ablates SaMP, OCG, and channel contrast, but it does not specify the time‑step T, window w, or stride used for those ablations; nor does it examine the role of the node‑wise contrast term independently (Eq. 16) or the data‑augmentation choices Tstruc and Tfeat (Eqs. 7–8). Without these, it is hard to attribute gains cleanly among SaMP, OCG, node contrast, channel contrast, and augmentation.
3. Marginal improvements over other graph information in SaMP. Table 2 compares multiple graph information. The absolute deltas are modest (e.g., Cora: 87.83→88.12; Photo: 94.24→94.79; Citeseer: 76.33→76.63; Computers: 91.25→91.29). The paper does not explain why spectrum is superior beyond small, dataset‑specific gains, nor whether SaMP’s benefit is statistically significant relative to other graph signals.
4. Limited analysis of the proposed methods (SaMP, OCG, and dual contrast). The paper argues SaMP improves spike pattern diversity, but there is no quantitative analysis of spiking dynamics on the benchmark datasets to show how spectrum alters dynamics beyond initialization. OCG is motivated as preserving cross‑channel dependencies, yet there is no measurement of inter‑channel correlation before/after OCG or how channel‑wise contrast capitalizes on those correlations.
5. “Spectrum‑guided spiking dynamics” may overstate the scope. Since spectrum is only used to form the IMP and not to adjust thresholds or currents over time (Eq. 14), it is debatable whether the spiking dynamics themselves are spectrum‑aware beyond initialization. 
6. Neuromorphic feasibility and overhead are under‑discussed. Computing Laplacian eigenvalues at scale can be expensive; the authors did not quantify the cost of eigenvalue computation. Figure 5’s energy comparison aggregates methods but omits spike‑rate, step‑count T, overlap ratio π, and OCG windowing specifics driving energy per sample on each dataset.
7. Dataset‑dependent energy behavior is unexplained. In Figure 5, relative energy rankings differ between Photo and Computers. 
8. Claims of SNN specificity to GCL are not disentangled from initialization. Since the authors note other graph signals can be injected instead of eigenvalues, it is unclear whether the core benefit is “spectrum‑aware dynamics” or simply “non‑zero, structured IMP”.

### Questions
Please refer to Weaknesses section.

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
3

### Summary
This paper proposes a method for improving the domain adaptation of spiking GNN.
Starting from adaptively adjusting the threshold, temporal aggregation and pseudo-label generation are used together to enhance the accuracy.

### Strengths
- Degree is a central information in graph. Using this is a rational choice
- Experimental comparison is made for a number of baselines
- Meaningful improvements.

### Weaknesses
**Narrow scope**
- Domain adaptation + spiking NN seems to have a narrow scope, and its practicality should be better motivated.

**Unclear platform**
- The reviewer is not sure whether this work is being proposed as 1) a better alternative for existing algorithms (that typically run on GPUs), 2) a method that would run on spiking (neuromorphic) hardware, or 3) a hybrid platform that uses neuromorphic+traditional digital hardware, 4) or something else. I was thinking of 2) at first, but the way the authors are making their comparison in Table 1 and 2 seems to be implying 1) or 3). If 1) or 3) is the case, I believe the comparison of the training/inference time and energy on those platforms should be made.

**Feasibility**
- The proposed method requires an adaptive threshold for each neuron, which could be hard to maintain. This issue is related with the platform one. I don't think neuromorphic hardware would have enough memory to store that much data.
- Use of the attention operation is proposed, and I am not sure if that falls into a spiking network. An attention operation involves a lot of (fp) multiplication and accumulation without any activation function. It's been shown in multiple areas that adding attention could improve performance, but for SNNs, adding attention does not seem to be directly feasible unless the target platform is GPU or a hybrid one.

**Potentially unfair setup** 
- In comparison, the GNN model architecture and methods seem to be mixed, which makes it difficult to assess a fair comparison. 

- What model is used for the proposed method? Eq(1) indicates that there is a learnable weight for each edge in the graph for each layer? I have not seen such setting, and for sure it would be a huge number of learnable weights compared to traditional GNNs such as GCN or GIN. Moreover, such a setting would only be only possible if all vertices are edges are known at training time, making it difficult for the work to be applied to graph classification.

- I am concerned about the energy comparison made in section 5.4. I don't get how the complicated design of this work would function on simple hardware such as ROLLS. I believe it would be very difficult to run a simple GNN in a spiking form, but including the dynamic threshold and attention operation would be yet another level. If such a dramatic energy saving were to be claimed, please discuss how the algorithm would run on a neuromorphic device, and provide details, including those of the GPU-based counterpart.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes DeSGraDA, a novel framework designed to address distribution shifts in Spiking Graph Networks (SGNs). DeSGraDA introduces a degree-conscious spiking representation with adaptive node thresholds, temporal distribution alignment via adversarial membrane potential matching, and pseudo-label distillation for reliable target supervision. The paper also provides a theoretical generalization bound for SGDA. Extensive experiments on multiple benchmark datasets demonstrate that DeSGraDA consistently surpasses state-of-the-art baselines in both classification accuracy and energy efficiency.

### Strengths
- This paper formally defines and proposes a novel framework for addressing SGDA, and introduces a biologically inspired degree-conscious mechanism that bridges the gap between spiking neural networks and graph learning.
- The proposed framework integrates three complementary modules: degree-conscious spiking representation, temporal distribution alignment, and pseudo-label distillation, supported by a derived generalization bound. This combination ensures both theoretical rigor and practical robustness.
- Extensive evaluations across multiple benchmark datasets demonstrate consistent improvements in both classification accuracy and energy efficiency. The results convincingly validate the model’s effectiveness and highlight its potential for low-power, real-world applications.

### Weaknesses
- The discussion of related work should include more recent DA methods or frameworks, such as [1,2], to better position this study within the current research landscape.
- In Section 4.3, the paper introduces a clustering-based approach for pseudo-label generation but does not specify which clustering algorithm is used or how DeSGraDA identifies the dominant pseudo-labels within each cluster.
- In Section 5.3, the ablation study appears incomplete, as it only evaluates the impact of removing individual components. It would be more comprehensive to also include results for removing two or three components simultaneously.

[1] Smoothness really matters: A simple yet effective approach for unsupervised graph domain adaptation. AAAI. 2025.
[2] Rethinking Graph Domain Adaptation: A Spectral Contrastive Perspective. ECML. 2025.

### Questions
- Can the proposed framework be applied to source-free domain adaptation scenarios? 
- How does the presence of noise in the target domain affect the performance and robustness of the proposed framework?

### Soundness
3

### Presentation
3

### Contribution
3

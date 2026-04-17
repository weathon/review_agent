# Modality-free Graph In-context Alignment

- Decision: Accept (Oral)
- Scores: 4, 6, 8, 6

## Abstract
In-context learning (ICL) converts static encoders into task-conditioned reasoners, enabling adaptation to new data from just a few examples without updating pretrained parameters. This capability is essential for graph foundation models (GFMs) to approach LLM-level generality. Yet current GFMs struggle with cross-domain alignment, typically relying on modality-specific encoders that fail when graphs are pre-vectorized or raw data is inaccessible. In this paper, we introduce **M**odality-**F**ree **G**raph **I**n-context **A**lignment (MF-GIA), a framework that makes a pretrained graph encoder promptable for few-shot prediction across heterogeneous domains without modality assumptions. MF-GIA captures domain characteristics through gradient fingerprints, which parameterize lightweight transformations that align pre-encoded features and indexed labels into unified semantic spaces. During pretraining, a dual prompt-aware attention mechanism with episodic objective learns to match queries against aligned support examples to establish prompt-based reasoning capabilities. At inference, MF-GIA performs parameter-update-free adaptation using only a few-shot support set to trigger cross-domain alignment and enable immediate prediction on unseen domains. Experiments demonstrate that MF-GIA achieves superior few-shot performance across diverse graph domains and strong generalization to unseen domains. The code is available at https://github.com/JhuoW/MF-GIA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes MF-GIA, a modality-free in-context learning framework for heterogeneous pre-encoded graphs. The key idea is to derive a single-step gradient fingerprint from a frozen GNN initialization and use it as a domain descriptor. This domain embedding modulates lightweight FiLM-based feature and label alignment modules. The method is trained via episodic tasks and performs test-time adaptation by computing a new gradient fingerprint on the support set—without updating model parameters. A dual prompt-aware attention (DPAA) module performs few-shot matching.

### Strengths
Novel idea: Leveraging a single-step gradient displacement as a domain signature to instantiate FiLM alignment is an interesting and original approach for graph ICL.
Ambitious scope: The method targets heterogeneous graphs and attempts to unify node-level and edge-level tasks without modality conversion.

### Weaknesses
W1. Gradient-fingerprint robustness not validated
The central mechanism assumes that a single-step gradient displacement Δθis a robust domain descriptor. However, the paper provides no sensitivity analysis to different initialization seeds, learning rates, or step counts (1 vs. 2+). If the fingerprint is unstable or dominated by noise, the entire alignment process can break down.

W2. Domain embedding loss collapse risk
The pairwise distance-preserving loss used to train the domain embedder lacks explicit normalization or regularization, and no experiments test whether embedding collapse or trivial scaling happens. While scalability is not a major concern given the small number of pretraining domains, training stability and meaningful domain structure need to be demonstrated.

W3. Misleading “gradient-free” terminology & lack of inference-overhead reporting
The paper repeatedly claims “gradient-free adaptation,” yet inference requires a backward pass to compute Δθ_"new" . This distinction matters for deployment and comparisons against genuinely gradient-free ICL methods. The paper should clarify the terminology (e.g., “parameter-update-free”) and report inference-time overhead (FLOPs, memory, latency).

W4. Label alignment is fragile to label-ID permutation. The FiLM-modulated label prototype table indexes shared label embeddings by label ID. This implicitly assumes consistent label semantics across domains. In practice, label IDs are arbitrary. The paper lacks tests for label-ID permutation invariance or alternative mechanisms to infer label semantics from the support set.

W5. Lack of architectural ablations
No ablation comparing Conv2D vs. more canonical designs (Flatten+MLP, pooling) for Δθembedding. No ablation on DPAA depth or number of heads.

### Questions
Please refer to the above weaknesses.

### Soundness
2

### Presentation
2

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
The paper introduces a Modality-Free Graph In-context Alignment (MF-GIA) framework that enables pretrained graph encoders to perform few-shot, in-context learning across heterogeneous domains without relying on raw graph data. By leveraging gradient fingerprints and a dual prompt-aware attention mechanism, MF-GIA aligns pre-encoded features into a unified semantic space.

### Strengths
1. The proposed method achieves good performance on the few-shot node classification task across 5 datasets in m-way k-shot settings.
2. This paper provides the theoretical analysis for the proposed feature-alignment. 
3. The presentation of this paper is good and the paper is easy to follow.

### Weaknesses
1. In Theorem 3.1, the authors fail to explicitly define how Wasserstein distance $W_2(\cdot,\cdot)$ is used to quantify the distance between two domains. This omission makes it unclear what assumptions are made about the underlying distributions, and whether the bound holds under general conditions. Furthermore, in Definition 3, the authors do not provide a formal definition of the graph distance metric $d_g(G_i, G_j)$, leaving readers uncertain about what graph properties are used for similarity measurement. While the proof later specifies the way to measure graph distance, presenting it only in the proof creates the illusion that the theoretical statements are self-contained and fully justified, when the definitions are essential for interpreting the results. A more rigorous treatment would explicitly define all metrics and assumptions in the main text before presenting the theorem.
2. The term "modality-free" is somehow tricky. From terminology perspective, a modality-free model treats all input data under a unified representation or processing framework, regardless of its original modality. However, simply not having access to raw data does not automatically make a model or setting modality-free.
3. In table 3, MF-GIA performs worse on WN18RR dataset and a large performance gap between MF-GIA and the baseline methods. However, the authors do not discuss why the proposed method fails on this dataset.

### Questions
1. In the label alignment module, what if the number of classes in the unseen graph is less than $L_{max}$ that is determined by the available (training and test) graphs? In this scenario, it seems to restrict the capability of label alignment.
2. In table 2, “–” denotes datasets where only encoded features and indexed labels are available, making modality-dependent models inapplicable. Why do the authors report the performance of OFA and GraphAlign on ogn-Products dataset while do not report the results of AutoGFM?
3. In the proposed DPPA, two attention layers share the same projection matrices $W_k$ and $W_v$. Have you tried to not share the parameters? It seems that sharing the same matrices enforces the identical matching in both the feature embedding space and the label space. Intuitively, the cross-graphs matching in these two spaces should not be identical.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MF-GIA, a modality-free in-context learning framework for graph-structured data. It enables frozen, pretrained graph encoders to perform few-shot classification across heterogeneous domains without requiring raw modality inputs or task-specific fine-tuning. The method relies on gradient fingerprints to characterize domains and uses domain-conditioned transformations to align pre-encoded features and labels into a unified space for cross-domain reasoning.

### Strengths
S1: The use of gradient fingerprints to derive domain embeddings and guide feature/label alignment is novel and well-motivated.

S2: The model meets all three desired criteria—modality-free, post-training free, and cross-domain alignment—which most prior works fall short of.

S3: MF-GIA significantly outperforms competitive baselines on both node and edge classification tasks in few-shot settings, including unseen domains and tasks.

S4: The architecture, especially the use of Dual Prompt-Aware Attention, is carefully designed and theoretically grounded.
Scalable and Flexible: The approach accommodates pre-encoded data and avoids reliance on raw modalities, making it broadly applicable in practical scenarios.

### Weaknesses
To be honest, I did not find very serious weaknesses in this submission. Below are just some suggestions that may lead to a more solid work.


- Suggestion 1:The robustness of gradient fingerprints across noisy or low-quality domains is not deeply examined.
- Suggestion 2: While results suggest episodic training is beneficial, more direct comparison with alternative meta-learning strategies would strengthen the claim.
- Suggestion 3: While the framework is motivated by practical constraints (e.g., privacy, modality), the paper could benefit from an application case study demonstrating real-world utility, for example, RecSys (Adaptive Coordinators and Prompts on Heterogeneous Graphs for Cross-Domain Recommendations. SIGIR 2025), Protein (Protein Multimer Structure Prediction via PPI-guided Prompt Learning. ICLR 2024), Urban (Urban Region Pre-training and Prompting: A Graph-based Approach. KDD'25,  Boundary Prompting: Elastic Urban Region Representation via Graph-based Spatial Tokenization. arXiv.), drug (DDIPrompt: Drug-Drug Interaction Event Prediction based on Graph Prompt Learning. CIKM 2024), cross domain (All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining. SIGKDD 24), and theory basis (Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis. ICML 2025.). In summary, I think the authors could discuss more and the introduced related work seems to be not sufficient to capture the latest application trends in this area. 
- Suggestion 4: The conclusion should explore concrete future research directions. The authors are encouraged to expand their discussion with pointers to promising avenues such as: Task-agnostic generalization using prompt pools across evolving graphs
(see "ProG: A Graph Prompt Learning Benchmark" – NeurIPS 2024), Large language model integration for socio-semantic graph understanding (see "When LLM Meets Hypergraph: A Sociological Analysis on Personality via Online Social Networks" – CIKM 2025)
Theoretical perspectives on data operation views in graph prompting (see "Does Graph Prompt Work? A Data Operation Perspective with Theoretical Analysis" – ICML 2025)


Kindly note that all the above mentioned works are just for the author's information, not a mandatory request of citing them. Overall, this is a strong, well-written, and original submission that advances the state of graph in-context learning. I recommend acceptance.

### Questions
**Q1:** How stable are the gradient fingerprints across variations in the support set?

**Q2:** Why is a *single* gradient step chosen to compute the gradient fingerprint?

**Q3:** Can you elaborate on why the episodic training scheme outperforms supervised pretraining followed by prototype matching?


**Q4:** What exactly is the nature of the modality heterogeneity in the pretraining and test datasets?


**Q5:** How does the domain embedder perform when exposed to domains that are semantically distant from any in the pretraining set?


**Q6:** What is the computational overhead of generating gradient fingerprints at test time, particularly for large graphs or many-shot settings?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces MF-GIA aimed at doing few-shot / in-context node classification on new graphs without fine-tuning. The key idea is: for each target graph, first build a domain embedding to capture the graph’s style/distribution; then use this embedding to generate alignment/FiLM transforms that map (i) node features and (ii) label IDs into a shared semantic space, even when the original features aren’t text-based. On top of this, a DPAA module performs in-context matching between query nodes and the support set. so the model can adapt at test time just by seeing a few labeled nodes, with gradient-free inference.

### Strengths
- It doesn’t assume raw text or a specific encoder. as long as features are vectors, the domain-conditioned aligner can normalize them, which makes it more practical for real graph platforms. 

- Adaptation is done by in-context attention + generated aligners, so deployment on new graphs is lightweight and gradient-free.

- Using a domain embedding to steer FiLM-style transforms lets the model cope with cross-graph distribution shift (different feature spaces, different label vocab)

### Weaknesses
- Forcing all graphs to a fixed width via SVD could destroy domain-specific structure and cross-domain comparability before alignment.

- A single small-step gradient from a shared init can be noisy/sensitive to loss scaling; the theory assumes smoothness/Lipschitzness and gives an upper bound but not tight guarantees for practical separability. 

-  Pretraining uses only four node-classification datasets; broader modalities/tasks would better justify foundation-level generality.

- The episodic objective introduces a temperature τ, but robustness/ablation for τ and DPAA projections isn’t detailed.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3

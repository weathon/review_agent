# FL-GAP: GRAPH-BASED ADAPTIVE PERSONALIZA- TION FOR FEDERATED DEEPFAKE DETECTION

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
Modern deepfake detection models degrade sharply when faced with unseen gen-
erative techniques or cross-domain shifts, a challenge further exacerbated in Fed-
erated Learning (FL) by heterogeneous client data. Standard FL methods (e.g.,
FedAvg) converge poorly under such conditions, while existing personalized FL
approaches often assume uniform similarity or rely on overly simplistic strategies
that fail to capture nuanced feature shifts. We introduce FL-GAP, a framework
for Federated Learning with Graph-based Adaptive Personalization that system-
atically adapts to both client heterogeneity and generator shift. FL-GAP combines
three components: (1) Adaptive Layer Freezing, a validation-guided mechanism
that selectively updates and uploads high-utility layers, reducing drift and commu-
nication overhead; (2) Server-Side Probing, a privacy-preserving method that uses
zero-input embeddings to construct dynamic round-wise similarity graphs; and (3)
Neighbor-Union Layer Aggregation (NULA), a per-layer aggregation strategy that
leverages updates from similar neighbors while preserving personalization. We
evaluate FL-GAP on FDf-27, a federated benchmark derived from DF40 with 27
deepfake methods spanning face swapping, reenactment, synthesis, and editing.
FDf-27 defines five increasingly challenging scenarios, including cross-domain
and globally unseen methods. Experiments show that FL-GAP consistently out-
performs centralized, general FL, and personalized FL baselines, with particularly
strong gains in unseen-method and OOD settings, while cutting communication
by up to 75%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FL-GAP, a novel Graph-based Adaptive Personalization framework for Federated Deepfake Detection. It is designed to address the critical challenges of client data heterogeneity and distribution shifts, particularly from unseen deepfake generators. The core of FL-GAP lies in three synergistic mechanisms: 1) Adaptive Layer Freezing, which selectively updates and communicates only high-utility layers to reduce client drift and communication costs; 2) Server-Side Probing, a privacy-preserving technique that uses zero-input embeddings to construct a dynamic similarity graph of clients; and 3) Neighbor-Union Layer Aggregation (NULA), which performs fine-grained, layer-wise aggregation based on the similarity graph. The framework is rigorously evaluated on a newly curated benchmark, FDF-27, across five scenarios of increasing difficulty, including out-of-distribution settings with globally unseen generative methods.

### Strengths
1. It introduces graph structural similarity into federated deepfake detection, enabling fine-grained, layer-wise personalization that moves beyond simplistic client-independent or uniform aggregation strategies.

2. The framework demonstrates exceptional performance in challenging scenarios involving unseen generative methods and cross-domain data, proving its effectiveness in non-stationary environments.

3. The adaptive layer freezing mechanism achieves a significant reduction in communication volume (up to 75%) while maintaining or even enhancing model performance, making it suitable for bandwidth-constrained edge devices.

4. The introduction of the FDF-27 benchmark, with its five escalating evaluation scenarios, provides a valuable and reproducible testbed for future research in federated deepfake detection.

### Weaknesses
1. The framework assumes the server has access to a large-scale public dataset for initial model pretraining. This assumption may not hold in practical scenarios where such diverse and representative public data is unavailable.

2. While privacy-preserving, the use of an all-zero vector for server-side probing may not fully activate and differentiate model representations, particularly in deep networks. For certain architectures without biases or BatchNorm, the output could be degenerate, potentially affecting the quality of the similarity graph.

3. FL-GAP integrates three complex mechanisms. Although the overall performance gain is significant, the paper lacks a thorough ablation study to disentangle and quantify the individual contribution of each component, making it difficult to understand the source of its success.

4. Several key baselines used for comparison (e.g., FedProx, FedBN) are relatively older methods. It remains unclear how FL-GAP would perform against more recent state-of-the-art personalized FL methods from 2024-2025, especially since some older baselines are reported to be the best baselines in Table 1 and Table 2.

5. The paper's discussion and comparison within the broader field of personalized federated learning (PFL) seem limited. While it positions itself as solving challenges mentioned in the paper via personalization, it has not adequately compared it with the latest advances in the PFL methods.

### Questions
1. Given the integrated nature of the three core mechanisms, what is the relative contribution of adaptive layer freezing, graph construction via zero-probe, and NULA to the overall performance gain?

2. How does FL-GAP's performance compare against the latest state-of-the-art PFL methods (e.g., from 2024-2025)? Would the significant advantages held over older baselines like FedProx and FedBN hold against these newer approaches?

3. The zero-input probing is a clever privacy-preserving design, but what are its theoretical and practical limitations? Are there scenarios where it would fail to construct a meaningful similarity graph?

4. The communication savings are impressive, but what is the computational overhead introduced for dynamic graph construction and layer-wise NULA aggregation, especially as the number of clients scales up?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes to address the deepfake detection problem under the Federated Learning (FL) paradigm through a newly designed framework, FL-GAP. While the problem setting is relatively novel and timely, the paper lacks sufficient and convincing justification of its motivation, practical relevance, and significance. Moreover, the proposed components within the FL-GAP framework appear incremental and show limited distinction from existing techniques, which weakens the overall originality and contribution of the work.

### Strengths
-This work explores the challenge of deepfake detection under FL scenarios, a relatively novel and underexplored problem setting. The topic is timely and has the potential to advance the applicability of deepfake detection in privacy-sensitive and distributed environments, offering possible benefits to both the deepfake detection communities.

-The paper includes a theoretical analysis of communication efficiency, which strengthens the soundness and credibility of the proposed algorithm. 

-The paper introduces a new federated benchmark built upon the DF40 dataset to evaluate the proposed algorithm.

### Weaknesses
-Although deepfake detection under FL represents a new intersection of two research areas, the paper fails to provide a convincing motivation for why this combination is necessary or practically meaningful. The reasoning presented in the introduction is unpersuasive and conceptually weak. The claim that generative models are inaccessible and therefore require an FL-based approach is questionable. In practice, most generative models can still be used to produce synthetic data for training. Consequently, the argument that FL is essential due to data access limitations is not well-founded, especially given that large amounts of representative data can typically be obtained for training deepfake detection models.

-The proposed methods are incremental, as several components closely resemble existing approaches in the FL literature. For instance, the idea of adaptive layer freezing has been previously explored in PartialFed [1], while the use of a similarity graph based on weight matrices and the NULA strategy shows conceptual overlap with the similarity-based personalization mechanism in pFedLA [2]. These similarities reduce the perceived novelty and make it difficult to identify the unique contribution of the proposed framework.

[1] PartialFed: Cross-Domain Personalized Federated Learning via Partial Initialization

[2] Layer-wised Model Aggregation for Personalized Federated Learning

-The compared algorithms are not reasonable. All compared methods are designed primarily for FL, which makes the comparison less meaningful. To ensure fairness and demonstrate the true effectiveness of the proposed framework, the baselines should include combinations of FL algorithms with existing deepfake detection models. Without such comparisons, it is difficult to accurately assess the merit and practical advantage of the proposed method.

### Questions
It is unclear why the existing DF40 dataset is insufficient for the experiments. From my perspective, the proposed FDf-27 benchmark appears very similar to the original DF40 dataset, and the differences between them are not clearly articulated. In most prior works, such as those using CIFAR-10 or CIFAR-100, authors typically describe their data partitioning strategy rather than claiming the introduction of a new benchmark. Therefore, the justification for presenting FDf-27 as a distinct benchmark requires further clarification and stronger motivation.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes FL-GAP, a personalized federated learning framework tailored for deepfake detection under heterogeneous client data and unseen generative methods. The framework consists of three main components: (1) Adaptive Layer Freezing to selectively update and communicate only high-utility layers. (2) Server-Side Zero-Input Probing to construct round-wise similarity graphs between clients without accessing private data. (3) Neighbor-Union Layer Aggregation (NULA) to perform layer-wise aggregation only from similar neighbors. The authors introduce a new federated benchmark, FDf-27, derived from deepfake datasets, and evaluate FL-GAP under five increasingly challenging scenarios (seen/unseen methods, domain shifts, OOD). Experiment results show consistent improvements over both global FL and personalized FL baselines, while reducing communication by approximately 75%.

### Strengths
1. Timely and Relevant Application: Deepfake detection is a novel and interesting application of FL, and addressing generator shift in a privacy-preserving manner is meaningful.
2. Well-Motivated Personalization Approach: The integration of adaptive freezing and graph-based client similarity is conceptually coherent and addresses key limitations of static personalized FL.
3. Strong Empirical Results: Extensive experiments across multiple scenarios demonstrate the effectiveness of FL-GAP, with notable improvements in OOD robustness.
4. Benchmark Contribution: The introduction of FDf-27 provides a realistic and reproducible testbed that may benefit future research in the community.

### Weaknesses
1. Justification on the difference between public pretraining dataset and the private client datasets. FL-GAP requires a public pretraining dataset which is accessible to the FL server. It is important that the public dataset is indeed heterogeneous with respect to the clients' datasets. Authors could add experimental results to demonstrate their heterogeneity. 
2. Scalability and practical deployment. Experiments use models such as Xception. it remains unclear how the approach scales to larger or transformer-based architectures commonly used in modern AIGC detection.
3. Assumptions behind zero-input probing require more justification. While the use of zero-input stimulation is claimed to be privacy-preserving, it assumes that probe embeddings reliably capture model similarity. It is unclear whether these embeddings are stable across different architectures or under adaptive client updates.

### Questions
1. Could authors provide justification on the difference between public pretraining dataset and the private client datasets?
2. Could the method scale to transformer-based architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed a personalized FL algorithm for DeepFake (DF) detection problems, called FL-GAP, a graph-based adaptive algorithm composed of the following three components:
-Adaptive layer freezing (for layerwise selective local client updates),
-server-side probing to build a graph dynamically to group similar clients together
-neighbor-union layer aggregation to aggregate layer parameters along the graph similarity

Although there are lots of personalized FL algorithms out there, they focus on the particular application of DF detection, in which they said some existing methods fell short in terms of aggregation strategy, and dealing with data heterogeneity.

### Strengths
-Interesting design ideas of three components for the pFL for the DF application.

### Weaknesses
- Fig.1: It would be more interesting to compare their results with the most competitive baseline pFL algorithms, instead of just the pre-trained (non-personalized) model.

- The datasets for the public and client data (in Sec.2): They said the only difference between the public data D_pub and the client data {D_k} is that the latter has personalized content. But as they also said that D_pub was large-scale and diverse, it might be assumed that D_pub can represent any generic contents. Ie, it is not clear to what extent the two datasets are different?

- The setup described as three bullet points (right before Sec. 2.1): Isn't this the same as the conventional personalized FL setup? Ie, the input covariates are distributed differently across clients (domain shift)? What's the difference?

I find that in the main sections (Sec.2.1.1~2.1.3), each describing each of the three components, they provide some theoretical arguments/justifications, but they are mostly not directly related to the performance or convergence of the proposed algorithm. See below.

- Sec.2.1.1 (adaptive selective freezing layers): Proposition 2.1 merely says that the selection of the parameters by the gradient magnitude order will maximize the first-order decreases. Isn't this obvious? But why does it necessarily improve the generalization performance? This is not explained.

- Sec,2.1.2 (server-side k-nn graph construction): In Eq.(2.9) they use 0 as the probe input for client signature generation. Why does this give you informative and discriminative signatures for the client? Although in App.C.3 iii) they have some arguments about the informativeness and BatchNorms, but it is not intuitive nor convincing. Zero input would wipe out the first layer weights, and so on. Is your method tied to BN-based networks?

- Also Lemma 2.2 is out of the blue. Does it have anything to do with the similarity of clients? The proof in App.D.3 merely says that with some high prob, the k-th and (k+1)-th neighbors don't change after the server averaging. But it has nothing to do with the similarity among clients.

- Sec.2.1.3 (neighbor union layer aggregation): In Thm 2.3, they show that the NULA update is non-expansive, and it converges toward the weighted neighbor mean. But, it doesn't imply convergence to the global solution. And the theorem is far from standard FL convergence analysis.

- What about generalization error analysis? Do you have any theoretical bound on it?

- About experimental section: The results overall show that there are only some marginal improvements, which makes it hard to draw any conclusion statistically significant. Can you provide error bars with multiple runs? The experiments are not thorough in terms of FL settings either: there are FL parameters like local client number of epochs, client participation rates, network architectures, etc.; but they seemed to test only one fixed setting. 

- Another question is why the algorithm was tested only on the DF detection problem alone. Why not other standard FL benchmarks? Considering that the DF detection problem is not fundamentally different from other standard FL datasets/problems with domain shifts. Also, the proposed method seems to be tied to particular network architectures that rely on BN layers.

### Questions
See questions in the weakness section.

### Soundness
2

### Presentation
2

### Contribution
2

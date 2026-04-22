# Representation-Aligned Multi-Scale Personalization for Federated Learning

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 6, 2

## Abstract
In federated learning (FL), accommodating clients with diverse resource constraints remains a significant challenge. A widely adopted approach is to use a shared full-size model, from which each client extracts a submodel aligned with its computational budget. However, regardless of the specific scoring strategy, these methods rely on the same global backbone, limiting both structural diversity and representational adaptation across clients. This paper presents FRAMP, a unified framework for personalized and resource-adaptive federated learning. Instead of relying on a fixed global model, FRAMP generates client-specific models from compact client descriptors, enabling fine-grained adaptation to both data characteristics and computational budgets. Each client trains a tailored lightweight submodel and aligns its learned representation with others to maintain global semantic consistency. Extensive experiments on vision and graph benchmarks demonstrate that FRAMP enhances generalization and adaptivity across a wide range of client settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In federated learning, accommodating clients with diverse computational resources remains challenging. Existing approaches often derive client submodels from a shared global backbone, which restricts structural diversity and limits representational adaptation. FRAMP addresses this by generating client-specific models from compact descriptors, enabling personalization to both data and resource constraints. Each client trains a lightweight tailored submodel while aligning its representations to preserve global semantic consistency.

### Strengths
1.	**Elegant Solution to a Practical Problem**: The proposed approach demonstrates a clean architectural design: it integrates resource-awareness and personalization within a unified descriptor-driven framework, avoiding additional communication or coordination complexity. Resource heterogeneity is a critical bottleneck for real-world FL deployment, particularly in cross-device settings. The paper's motivation and methodological framing are both relevant to industry-scale scenarios (e.g., mobile or edge-device training).

2.	**Dynamic and Adaptive Submodel Extraction**: The paper’s adaptive submodel extraction mechanism (where parameter importance is dynamically encoded during training) is a notable contribution. It allows submodels to evolve organically based on training dynamics, rather than relying on static pruning or masking heuristics. This reduces manual tuning and ensures that model capacity allocation remains responsive to the data distribution and client conditions.

3. **Strong Writing and Technical Clarity**: The exposition is clear, with well-structured sections that balance algorithmic intuition and formal derivation. The authors successfully convey the relationship between descriptor generation, submodel extraction, and alignment.

### Weaknesses
1.	**Unclear Motivation Behind "Underutilized Parameters" Argument**:

The paper states:
"Although FIARSE supports dynamic submodel sampling in each round, parameters with low importance scores are seldom chosen, resulting in limited training opportunities for these parameters. Consequently, a significant portion of the model remains underutilized, leading to high structural similarity and reduced diversity among submodels."

First, it is unclear why low-importance parameters should be trained if they consistently exhibit marginal contribution to model performance.

Second, diversity among submodels is presented as an implicit goal, but the paper does not provide a theoretical or empirical argument explaining why greater structural diversity necessarily improves generalization or fairness.

2.	**Fixed Client Descriptors May Limit Adaptivity**: 

At initialization, each client transmits its descriptor to the server once, which remains fixed throughout training. This design choice could hinder adaptivity when client conditions evolve, for example, due to fluctuating compute availability, memory constraints, or battery limitations. A dynamic descriptor that updates periodically or conditionally could better reflect real-world device heterogeneity. The paper would benefit from either (a) empirical evidence that descriptors remain valid throughout training, or (b) discussion of mechanisms for re-synchronization.

### Questions
1.	How is a client's computational capacity (which indicates how large of a model can be assigned to it for training) computed in real-life?

2.	Does the client receive both the submodel parameters and the corresponding mask defining trainable weights? If so, does the mask transmission effectively double the communication cost? A quantitative comparison of FRAMP’s communication and computation overhead against baselines would help substantiate the efficiency claims.

3.	The paper mentions retaining only parameters above an adaptive importance threshold. Conceptually, what does this threshold represent; magnitude, gradient sensitivity, or representational salience? Why should retaining only “high-value” parameters always yield optimal performance? It is conceivable that a complementary mix of high- and low-importance parameters could yield better local minima or smoother gradient flow. Additionally, pruning aggressively might cause gradient instability or blow-up, especially under non-IID conditions.

4.	What is the difference between local prototype and vector used for the personalized model creation?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper aim to tackle the limitation of structural diversity and representational adaptation focusing on the challenge of integrating clients with diverse resource constraints. The paper proposes FRAMP (Federated Representation-Aligned Multi-scale Personalization), a framework for model-heterogeneous, personalized federated learning (FL). FRAMP (i) uses a server-side hypernetwork to generate a client-specific full model from a compact per-client descriptor; (ii) performs adaptive submodel extraction via global Top-K magnitude masking to satisfy each client’s sparsity/computation budget; and (iii) adds prototype-guided representation alignment, where clients upload class prototypes and align to server-aggregated global prototypes without any shared public dataset. Experiments of image classification and node classification on graph-structured data show the personalization, generalization, robustness to unseen budgets and clients compare to the submodel extraction approaches.

### Strengths
- FRAMP addresses the limitations of relying on a single shared global backbone by proposing a client-aware model generation mechanism. This mechanism leverages compact client descriptors to instantiate personalized full-size models, allowing for fine-grained adaptation to both the client's data characteristics and their computational budgets. Building upon this, FRAMP uses an adaptive submodel extraction strategy that dynamically selects tailored, sparse submodels based on evolving parameter importance, ensuring resource constraints are met without introducing additional overhead.

- Extensive experiments on various benchmarks demonstrate that FRAMP consistently outperforms most strong baselines on the reported setups that support model heterogeneity. FRAMP maintains strong adaptability and comparable accuracy when generalizing to unseen client scenarios or missing capacity profiles.

### Weaknesses
- The server must generate full client-specific weights, apply global Top-K masking, and backpropagate through the hypernetwork using updates. The paper reports server time per round (slightly higher in round 1, similar afterward), but a deeper accounting of compute/memory (e.g., peak RAM, scaling with client/model size, HN size) is missing. In large-scale vision backbones, full weight materialization per participating client can be non-trivial.

- FRAMP still exposes an attack surface for DLG/gradient-inversion–style attacks, since each round transmits masked model updates  and class prototypes to the server, and clients upload a descriptor at initialization. The paper argues that per-client personalized submodels and dynamic Top-K masking disrupt cross-round consistency and thus make such attacks harder, but this is a qualitative claim without formal guarantees. The presented “privacy analysis” mainly tests accuracy robustness to noisy/rotated prototypes rather than resistance to data reconstruction or label-distribution leakage. Without additional mechanisms such as secure aggregation and differential privacy, the risk of sensitive information leakage cannot be conclusively ruled out.

- I know that the space is limited due to page limitations, but the Conclusion section does not fully contain the contributions/limitation/future works of the paper.

### Questions
- When calculating TopK, the authors only keep parameters with large absolute values, but what about cases where a specific parameter value is small, yet it could play a significant role when considering its combination (interaction) with other parameters?

- Please include experiments under IID client partitions to disentangle the benefits of FRAMP from non-IID effects.

### Soundness
3

### Presentation
3

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
This paper proposes FRAMP (Federated Representation-Aligned Multi-Scale Personalization), a unified framework for personalized and resource-adaptive federated learning (FL). It targets the problem of client heterogeneity, both in computational capacity and data distribution, by avoiding reliance on a single global model. FRAMP uses a hypernetwork-based client-aware model generator to produce personalized full-size models from compact client descriptors, an adaptive submodel extraction mechanism based on parameter magnitudes, and a prototype-guided representation alignment module to maintain semantic consistency across clients. Experiments on CIFAR-10, CIFAR-100, and ogbn-arxiv show that FRAMP outperforms state-of-the-art baselines such as HeteroFL, FedRolex, ScaleFL, and FIARSE, especially under extreme heterogeneity, and generalizes well to unseen clients and model sizes.

### Strengths
++ Proposes a conceptually unified and technically elegant framework that integrates personalization, sparsity, and semantic alignment.

++ Addresses data and system heterogeneity simultaneously.

++ Provides comprehensive experiments with comparisons across multiple datasets and varying heterogeneity levels, including unseen clients and capacity profiles.

++ Clear writing, well-structured methodology, and solid empirical improvements demonstrate intense engineering rigor and reproducibility potential.

### Weaknesses
-- The contribution of the submission stems from providing a unified solution to both system and data-level heterogeneity. However, the techniques to achieve this are all from existing studies, i.e., model generation [1], sub-model extraction [2], and prototype-based alignment [3]. A lack of exploration on the unique challenges of combining these techniques makes the contribution incremental.



-- The submission claims that previous methods all use one fully shared global model as the backbone, and such a way would underutilize some parameters. However, this leads to several concerns:





Do high structural similarity and reduced diversity among submodels bring any crucial limitations? Any evidence?



Figure 2 only demonstrates that a smaller model would obtain lower performance. To support the claim that more activations on the early portion of model parameters, the submission should fix the total model size and only change the activation locations (e.g., early portion → mid portion → late portion) rather than change the model size and activation locations simultaneously.



-- The submission only provides the experimental results of the computation overhead of the hypernetwork in the server. Better to include more results of the overhead on additional data transmissions.



-- Though the Appendix provides the experimental results with Dirichlet-\alpha = 0.1, it is better to include more non-IID configurations.

 
[1] Personalized Federated Learning using Hypernetworks, ICML 2021.

[2] Fiarse: Model-heterogeneous federated learning via importance-aware submodel extraction, NIPS 2024.

[3] FedProto: Federated Prototype Learning across Heterogeneous Clients, AAAI 2022.

### Questions
Please see Weaknesses

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
3

### Summary
The paper asks two questions of the personalized Federated Learning (pFL) setup - a) for each client in a communication round, how to construct personalized submodel that adapts to both computational constraints and data distribution, and b) how to promote semantic consistency across clients in the absence of any shared reference dataset? The paper develops a new framework called FRAMP which is designed to incorporate an affirmative answer to both of these questions. FRAMP uses a hypernetwork on the server to instantiate personalized full-size model for each client, and thresholds model weights based on magnitude to achieve a target number of non-zero weights. Experimental evidence is provided for improved statistical performance compared to baselines.

### Strengths
(S1) The writing and presentation in the paper is clear. The problem is well motivated and FRAMP's design is explained well. Prior work is well cited although contextualization could be improved further.

### Weaknesses
(W1) Lines 284-287 state that each participating client updates all parameters $\omega_n$. This defeats the purpose of constructing personalized submodels that adapt to computational constraints (Line 50, Q1). Further, there isn't any theoretical/empirical study in the paper that shows FRAMP's advantage in terms of computational constraints.

(W2) It is difficult to justify originality and significance when contextualized against [FedDSE] (missing citation; relevant prior work) and [PeFLL]. FedDSE seems to address the two problems a) Limited Structural Diversity, and b) Client-Agnostic Importance Estimation, that are mentioned in section 3.2 as drawbacks of FIARSE.

[FedDSE] Haozhao Wang, Yabo Jia, Meng Zhang, Qinghao Hu, Hao Ren, Peng Sun, Yonggang Wen, and Tianwei Zhang. 2024. "FedDSE: Distribution-aware Sub-model Extraction for Federated Learning over Resource-constrained Devices". In Proceedings of the ACM Web Conference 2024 (WWW '24), 2902–2913. ACM. DOI: 10.1145/3589334.3645416.

(W3) Loose statements have been made at several places in the paper -
- Lines 252-254: "... jointly learn the mask and form the submodel, ..." This is somewhat misleading. As best I can tell, equation (3) is implemented as optimize weights and then threshold, which is different from jointly optimizing.
- Lines 363-365: "Fig. 5c demonstrates representation alignment, with local prototypes initially dispersed (left, early stage) converging toward global prototypes (right, later stage), leading to tighter clusters and improved consistency across clients." It is not clear to me how Figure 5c is conveying this statement.
- Line 140: "Some recent methods incorporate importance scores to guide mask selection." Which methods? Citation is missing.
- Line 176-177: "... clients learn inconsistent class prototypes due to model heterogeneity and non-IID data, ultimately compromising generalization performance." Where is this established? Citation? By itself, differences in steady state client accuracies under FIARSE (Figure 2) is not evidence that personalization is not working well. Optimal accuracies can indeed vary across clients.
- Line 439-440: "We attribute this to the difficulty of learning structured sparsity and controlling sparsity levels." This is a hypothesis, not an attribution.

(W4) Line 485: While generalization testing has been explained in sections 5.3 and 5.4, personalization testing is inadequately explained/referenced.

Things to improve the paper that did not impact the score:
- Figure 4: This is better represented as a cdf type plot. $F(a)$: num clients with test accuracy $>= a$. One plot for each size value, and each plot has all algorithms of interest.

### Questions
(Q1) Lines 155-159: While this is an interesting observation, there could be an alternate explanation for this, viz. the early layers are more about generalization and learning shared features than about personalization.

(Q2) Could the authors address (W1) and (W2)?

### Soundness
2

### Presentation
3

### Contribution
2

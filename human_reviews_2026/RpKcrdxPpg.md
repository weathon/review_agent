# Model Theft and Inversion Attacks Against Query-free Collaborative Inference Systems

- Decision: Reject
- Scores: 4, 2, 6, 6

## Abstract
Collaborative inference systems are designed to deploy high-performance models on resource-constrained edge devices by splitting the model into two parts, deployed separately on the client device and the server. However, server-side adversaries can still infer client's private information from the latter part of the model. Previous works rely on auxiliary data with matching labels and unlimited queries to reconstruct inference data or determine sample membership. In contrast, this paper introduces a novel threat called Model Theft and Inversion Attacks (MTIA), targeting a more realistic and challenging scenario where adversaries often lack access to label-consistent datasets. Moreover, adversaries cannot query the client device and have no knowledge of the client model’s architecture or parameters. To address these challenges, we leverage transfer learning and self-attention alignment to extract knowledge from the server model and align it with the target task. This enables model recovery with performance comparable to the original model while improving the reconstruction of high-fidelity private data. Additionally, we propose an enhancement that uses reconstructed images to further boost the recovered model’s performance. Extensive experiments across various datasets and settings validate the effectiveness, robustness, and generalizability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses privacy vulnerabilities in collaborative inference systems, where models are split between a client and a server. It introduces a novel threat, Model Theft and Inversion Attacks (MTIA), under a challenging and realistic scenario: the server-side adversary is query-free and only possesses a public dataset that is label-inconsistent with the client's private data. The proposed MTIA framework operates in two stages: (1) a transfer-learning-based model completion step to substitute the missing client-side model, and (2) a self-attention alignment step that uses the public dataset to align the substitute model's feature space with the server's model. Experimental results demonstrate that MTIA can successfully recover the client model's functionality with high fidelity, which subsequently enables high-fidelity model inversion attacks to reconstruct sensitive private training data.

### Strengths
1. The paper is well-written, presenting a clear and logical flow from problem definition and methodology to experimental validation.

2. The primary contribution is the introduction of a novel and practical threat model (query-free and label-inconsistent).

3. The paper proposes an effective two-stage framework with transfer learning and self-attention to solve this new problem scenario.

### Weaknesses
1. This paper assumes a collaborative inference scenario where only a single block is assigned to the client side. In other words, it is assumed that most of the model, excluding one block, is already known in the case of the ResNet-50 or MobileNetV2 used in the experiments. The additional experiments presented in Figure 4, which evaluate up to three blocks, also show similar results, where the server side holds the majority of the model. If the model is evenly partitioned or the client holds more blocks, failing to validate MTIA in these scenarios could suggest that the model is only applicable in situations where sufficient information is already available.

2. There is almost no performance difference between the pretrained model and the Transfer Learning (TL) only setting in the main paper, while there is a significant performance gap between TL and MTIA. This indicates that the alignment step is responsible for the majority of the performance gain. However, further explanation is needed as to why aligning attention maps using a public dataset helps in aligning the functionality of the model to its private dataset.

3. Apart from the problem formulation and framework design, the other components lack novelty. Model inversion attack and attention alignment are both well-established methods in the literature.

4. Despite model inversion being a central component of the proposed MTIA threat, the paper’s review of related work is notably incomplete. The review is narrowly focused on specific white-box, GAN-based methods (PPA and PLGMI) that are operationally required for the paper’s experiments. It fails to cite foundational works that established the model inversion threat (e.g., Fredrikson et al. [1]) as well as subsequent research from other attack settings, such as black-box attacks (RLB-MI [2]) and label-only attacks (BREP-MI [3]).

[1 ] Fredrikson, M., Jha, S., & Ristenpart, T. (2015, October). Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security (pp. 1322-1333).

[2] Han, G., Choi, J., Lee, H., & Kim, J. (2023). Reinforcement learning-based black-box model inversion attacks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 20504-20513).

[3] Kahla, M., Chen, S., Just, H. A., & Jia, R. (2022). Label-only model inversion attacks via boundary repulsion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 15045-15053).

### Questions
1. Your main experiments assume a server-heavy split where the client only has one block. Do you expect the MTIA attack to remain effective in more realistic scenarios, such as a 50:50 split or a client-heavy split where the server has less information?

2. The self-attention alignment step provides the most significant performance boost. Can you provide a more fundamental explanation for why aligning attention maps using a public, label-inconsistent dataset also works to align the model for the private task?

3. When you retrain the model on the reconstructed images, how do you account for the risk of error accumulation? Is it possible that training on these imperfect images could introduce noise and actually degrade the model's performance?

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
3

### Summary
This paper introduces Model Theft and Inversion Attacks (MTIA) for recovering a missing client model by accessing the server system in collaborative inference. MTIA first reconstructs the client model using a self-attention guided, transfer-based approach. The recovered model is then used to apply existing model inversion attacks and reconstruct the client's training data, which is subsequently used to further refine the recovered model. Authors demonstrate the effectiveness of MTIA across two datasets and two model architectures.

### Strengths
- The results demonstrate that MTIA is highly effective, able to recover the client model with a high test accuracy of 79-90\%.

- MTIA is able to recover the client model without using any query to client model.

### Weaknesses
- The proposed method focuses solely on recovering the client model and does not introduce any new approach for model inversion attacks. In fact, the authors employ PPA and PLGMI in Phase 2 to recover training data from the newly obtained client model. Therefore, using the term “Model Theft and Inversion Attacks” appears to overstate the actual contribution of the paper.

- The experimental results are questionable. The authors use the FaceScrub and CelebA datasets, one being public and the other private. Although the Adversary’s Knowledge section claims that the public and private datasets are non-overlapping, this assumption may not hold true for FaceScrub and CelebA. To the best of my knowledge, several classes overlap between these two datasets, which could artificially boost the performance of the client model.

- The server architecture is not clearly described. For instance, the authors mention using MobileNetV2 and ResNet50, but it is unclear which layers belong to the original client model and which are part of the server model.

- The paper lacks formal analysis or theoretical justification to explain why using self-guided attention should lead to improved performance in the new client model

### Questions
- Please clarify the potential overlap between the public and private datasets.

- Please provide a clearer description of the server architecture.

- Could the authors provide analysis or explaining why using self-guided attention should lead to improved performance in the new client model?

### Soundness
1

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
3

### Summary
This paper presents a comprehensive study on Model Theft and Inversion Attacks (MTIA) against query-free collaborative inference systems. The proposed MTIA framework introduces a two-step strategy that combines transfer learning and self-attention alignment to recover client-side models and subsequently reconstruct private data under highly restrictive conditions (label inconsistency, no query access). The paper demonstrates strong performance on multiple datasets (CelebA, FaceScrub, fingerprints, and object classification) and also evaluates robustness against several existing defenses (NoPeek, BiDO, DP, etc.). The topic is timely and relevant to privacy and security in distributed ML systems.

### Strengths
1. The paper studies a realistic yet underexplored attack setting, query-free collaborative inference with label inconsistency. This fills a meaningful gap compared to prior works assuming either full query access or label-consistent auxiliary datasets.

2. The two-stage design (transfer-based recovery + self-attention alignment) is clearly described, mathematically formalized, and experimentally validated. The theoretical explanation of the self-attention alignment loss (Eq. 3–6) is technically rigorous and intuitive.

3. The experiments are well-organized, spanning diverse datasets and architectures, and the paper conducts extensive ablation studies (e.g., effect of missing blocks, dataset size, fine-tuning epochs). Evaluation against multiple attack/defense baselines is thorough.

### Weaknesses
1. While several defenses are tested (Tables 3–4), it would be valuable to analyze how MTIA behaves under stronger modern defense paradigms, such as: 1)Adversarially-trained encoders for privacy-preserving inference (e.g., Noisy Adversarial Representation Learning, UAI 2023 [1]); 2)Defense approaches offering theoretical robustness guarantees (e.g., Theoretical Insights in Model Inversion Robustness and Conditional Entropy Maximization for Collaborative Inference Systems, CVPR 2024 [2]) Including these could strengthen the discussion on the boundary of MTIA’s capability and clarify how robust the proposed method is against more principled protection schemes.

2. The repeated MTIA attack (r-MTIA) involves multiple fine-tuning and GAN optimization cycles. A brief analysis of computational overhead and scalability (e.g., GPU hours, convergence time) would make the results more practical and comparable.

3. It would be helpful to discuss how sensitive the self-attention alignment loss is to layer selection (Eq. 3–6), and whether aligning only selected layers could reduce computational cost without hurting performance.

### Questions
Please refer to the listed weaknesses. The reviewer is interested in seeing whether the proposed attack methods remain effective against stronger defense mechanisms, such as adversarial representation learning and defenses offering theoretical robustness guarantees.

### Soundness
3

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
3

### Summary
The paper studies a model-inversion attack against a targeted classification model  M in a query-free setting. The attacker does not know the model’s early feature extractor  F_c  but knows the final layers F_s. The attacker also possesses a rough approximation of the training data (public dataset  D_pub). The attack is “query-free” in the sense that the attacker cannot probe  F_c.
The goals are 
- (G1) to recover a model F'  that approximates (F_c, F_s); and
- (G2) to recover representative instances for each class.

Specifically, the paper first trains a model F' with some layers frozen as F_s using the public dataset (goal G1).  Next employs known inversion techniques on F' to obtain (G2).

### Strengths
- The paper demonstrates that query-free model inversion is feasible when the attacker has a public dataset and access to the target model’s final layers. 

- The manuscript is generally well written and clear. I did have some difficulty following Step 2 in Section 3.2, but the overall presentation is accessible.

- There is a broad literature on model inversion under varied threat models; the specific setting studied here, query-free attacks where only the last few layers are known, is technically interesting and could be a meaningful addition.

### Weaknesses
- The claim that the query-free setting is the more realistic threat model is not fully justified. In the described threat scenario the attacker is hosting F_s  and receiving features from clients; it is unclear why the attacker would be constrained to a query-free approach rather than using client-provided instances directly. The paper should better motivate why an attacker in this position would prefer or be forced into the query-free strategy.

### Questions
- Strengthen the threat-model discussion: explicitly compare the practical capabilities and trade-offs between a query-free attacker and "non-query-free" methods.

### Soundness
3

### Presentation
3

### Contribution
2

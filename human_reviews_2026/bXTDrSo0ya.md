# TAP: Two-Stage Adaptive Personalization of Multi-task and Multi-Modal Foundation Models in Federated Learning

- Decision: Reject
- Scores: 2, 4, 4, 6, 4, 6, 6

## Abstract
Federated Learning (FL), despite demonstrating impressive capabilities in the training of multiple models in a distributed manner, has been shown to produce a final model not necessarily well-suited to the needs of each client. While extensive work has been conducted on how to create tailored personalized models, called Personalized Federated Learning (PFL), less attention has been given to personalization via fine-tuning of foundation models with multi-task and multi-modal properties. Moreover, there exists a lack of understanding in the literature on how to fine-tune and personalize such models in a setting that is heterogeneous across clients not only in data, but also in tasks and modalities. To address this gap in the literature, we propose TAP (Two-Stage Adaptive Personalization), which (i) leverages mismatched model architectures between the clients and server to selectively conduct replacement operations when it benefits a client's local tasks and (ii) engages in post-FL knowledge distillation for capturing beneficial general knowledge without compromising personalization. We also introduce the first convergence analysis of the server model under its modality-task pair architecture, and demonstrate that as the number of modality-task pairs increases, its ability to cater to all tasks suffers. Through extensive experiments, we demonstrate the effectiveness of our proposed algorithm across a variety of datasets and tasks in comparison to a multitude of baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
TAP is a two-step adaptive Personalized Federated Learning algorithm designed to personalize heterogeneous multi-modal and multi-task foundation models. TAP effectively balances global collaboration with client-specific adaptation, supported by theoretical convergence analysis highlighting the limits of a shared server model. By integrating margin-based optimization during federated training, TAP achieves personalization across diverse datasets, tasks, and architectures.

### Strengths
1.	The proposed two-step adaptive mechanism (margin-based optimization followed by post-training knowledge distillation) is intuitively motivated and straightforward to implement, potentially allowing integration into existing FL pipelines.

2. The inclusion of convergence analysis provides some theoretical context for why pure global aggregation may be insufficient.

3. The experiments cover multiple datasets and architectures, which helps demonstrate the potential generality of the approach, even though deeper ablations and baselines are needed for stronger validation.

### Weaknesses
1.	**Lack of motivating examples**: While the paper highlights the limitations of prior uni-modal and uni-task PFL approaches, it does not provide concrete examples of realistic multi-modal, multi-task federated scenarios (e.g., clients working on paired image–text data with distinct downstream tasks). This omission weakens the motivation for TAP's design. What are the scenarios where TAP can be practically used? 

2.	**Insufficient distinction from related work**: The related work section reads as a descriptive overview rather than a comparative positioning of TAP relative to existing PFL, multi-modal, or LoRA-based methods. It should explicitly articulate the novel contributions and technical differentiators that set TAP apart.

3. **Dependence on existing techniques**: The approach leverages LoRA for efficient fine-tuning but does not contribute novel adaptations or theoretical insights specific to LoRA in the PFL context. Hence, the claim of extending PFL to foundation models should be supported with more than existing parameter-efficient methods.

4. **Choice of baselines**: The paper does not include established PFL baselines, limiting the fairness and interpretability of the reported improvements. Comparisons with standard methods such as pFedMe [1], Per-FedAvg [2], or FedPer [3] would strengthen the evaluation.

5. **Unclear utility of knowledge distillation**: Tables 4 and 5 show marginal or no improvement when KD is applied, raising questions about its contribution to the method’s effectiveness. 

[1] Personalized Federated Learning with Moreau Envelopes (Canh et al., NeurIPS 2020)

[2] Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach (Fallah et al., NeurIPS 2020)

[3] Federated Learning with Personalization Layers (Arivazhagan et al., arXiv:1912.00818)

### Questions
1.	At initialization, do all clients receive the entire set of parameters (e.g., encoders, decoders, and mixture-of-experts modules) from the server, even for modalities or tasks they do not engage with? If so, this may introduce communication and storage inefficiencies.

2. When and how frequently is the personalized model trained? How long is each local training interval?

3. The paper defines historical terms $h_{i,o}(\ell)$​ and $h_{i,o}(p)$; what are their dimensions, update rules, or role in guiding the adaptive mechanism? Clarification on whether they represent moving averages, scalar accumulators, or vector-valued metrics would aid interpretability.

4. Why is historical loss tracking required to compute the indicator? Wouldn’t the current-round loss $\ell_t$ suffice for determining adaptation triggers?

5. When the indicator activates, are all personalized parameters replaced with global ones? If so, does this erase accumulated personalization, potentially discarding useful local adaptations that could generalize later?

6. Empirically, is the learning rate static or decayed over time? 

7. Are the reported results based on personalized accuracies (evaluated on the same clients used during training) or generalized accuracies (on unseen clients)? This distinction is critical for understanding TAP’s true generalization and personalization trade-offs.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes TAP for FL with heterogeneous multi-modal, multi-task foundation models. In Stage 1, each client maintains a local model alongside the global model. In Stage 2, the method performs knowledge distillation from the global model into the local one using a KL.

### Strengths
1. Addresses a challenging and underexplored problem in PFL where clients differ in tasks and modalities. 
2. A proof of convergence has been provided. The convergence bound for component-wise FedAvg over blocks supports the claim.

### Weaknesses
1. The convergence bound grows with the number of modality–task pairs $K^2$.
2. Experiments involve only a small number of clients (≈10) offering little evidence of scalability.

### Questions
1. Please emphasize the differences between the method with previous ones. My understanding is that the most significant innovation lies in the personalized approach in stage 2?
2. How does TAP perform when the number of clients increases, like to 100 clients?
3. How can the issues of modality–task pairs and scalability be addressed?
4. Please provide details of the computation overhead in Stage 2, such as training time and resource consumption.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes the TAP strategy, which integrates an adaptive parameter replacement mechanism during federated training with a knowledge distillation phase after training. This design enables clients to perform efficient personalized modeling in multimodal and multi-task heterogeneous environments. Furthermore, the convergence of the multimodal–multi-task server model is theoretically analyzed, and the method demonstrates promising results across multiple benchmarks.

### Strengths
1. The writing is clear and easy to follow;

2. The paper provides a well-articulated theoretical framework and convergence analysis;

3. It achieves state-of-the-art performance on eight common datasets for FLAVA and six for ViLT, distributed across 30 clients;

### Weaknesses
In fact, I believe this work mainly combines existing ideas and presents an incremental improvement rather than a fundamentally novel contribution.

Specifically, in multi-task learning, there already exist numerous studies that employ Mixture-of-Experts (MoE) for unified modeling (e.g., [1,2,3]). Similarly, for multimodal unification, related approaches have been explored (e.g., [4]), and even some works [5] have attempted to jointly unify both aspects. The contribution of this paper primarily lies in extending these ideas to the federated learning setting, which, in essence, appears relatively trivial.

Moreover, in the federated learning community, the use of knowledge distillation (KD) is far from new. Incorporating KD here further adds to the overall complexity of TAP without introducing substantial conceptual novelty. In other words, this work can be viewed largely as a combination of existing components and engineering-oriented refinements rather than a breakthrough innovation.

[1]. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts, KDD 2018

[2]. Mod-squad: Designing mixtures of experts as modular multi-task learners, CVPR 2023

[3]. M³vit: Mixture-of-experts vision transformer for efficient multi-task learning with model-accelerator co-design, NeurIPS 2022

[4]. Multi-modal gated mixture of local-to-global experts for dynamic image fusion, ICCV 2023

[5]. Dynamic modeling of patients, modalities and tasks via multi-modal multi-task mixture of experts, ICLR 2025

### Questions
Please see the Weaknesses.

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
3

### Summary
This paper proposes a new personalized federated learning method, called TAP, which enables the personalization of heterogeneous multi-modal and multi-task foundation models. Specifically, it designs a replacement operation to selectively update parameters across mismatched architectures, providing significant benefits to the personalization process. Moreover, a knowledge distillation (KD)-based post-training procedure is incorporated at the client side. In addition, the paper provides a convergence analysis of the server model under its modality–task pair architecture. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1. The proposed PFL algorithm supports heterogeneous multi-modal and multi-task personalization, and is the first to offer a convergence analysis specifically designed for the modality–task pair architecture.
2. In the federated training phase, each client adaptively determines whether to update portions of its local parameters with the global ones from the server according to its task performance, enabling selective absorption of knowledge that benefits its specific task. This is a particularly interesting and thoughtful design.
3. The authors conduct extensive experiments across multiple modalities, including image generation, text generation, image classification, and text classification, to comprehensively evaluate the proposed framework.

### Weaknesses
1. The font size used in Figures 1 and 2 is too small, which affects the overall readability and clarity of the illustrations.
2. Each task requires a manually specified margin hyperparameter, making the approach sensitive to hyperparameter selection.
3. The motivation behind employing knowledge distillation in the post-training stage is not well-grounded. Apart from empirical performance improvements, the framework lacks a clear theoretical justification or necessity for incorporating this additional step.

### Questions
1. As $\mathbf{X}_{[i]}$ is updated only on the tasks encountered in each local training round, how does the method address potential catastrophic forgetting when a client is associated with multiple tasks?
2. After the completion of FL training, does the additional knowledge distillation stage risk compromising the personalized adaptations of client models by imposing generalized representations from the global teacher? What is the technical motivation for introducing this step, considering the lack of a clear theoretical justification in the paper?
3. In Table 5, the improvement brought by knowledge distillation on image-based tasks appears marginal compared with that on text datasets. What factors contribute to this discrepancy? A more thorough analysis would help clarify this point.
4. In the experimental section, does each client correspond to a single task or multiple tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
In this paper, the authors propose TAP, a two-stage personalized federated learning framework for handling multi-task and multi-modal foundation models. Specifically, TAP introduces adaptive parameter replacement during federated training based on task-specific performance margins, and post-FL knowledge distillation to incorporate general knowledge without compromising personalization.

### Strengths
1. The proposed TAP addresses an underexplored but important setting: personalized federated learning (PFL) with both multi-task and multi-modal heterogeneity.

2. Provides the convergence analysis of a server model under a modality–task pair architecture.

3. The code is attached, making the method reproducible.

### Weaknesses
1. There is no analysis of how sensitive TAP is to margin hyperparameters. This is a hard threshold, which may cause optimization instability.

2. Figure 1-2 do not clearly illustrate which parameters are replaced or distilled. In particular, in Figure 2 stage(1) , the model appears visually identical before and after the replacement step, making it difficult to understand what changes occur or how the adaptive replacement actually operates. The figures are schematic rather than explanatory.

3. Communication efficiency is a key factor in personalized federated learning, yet the paper does not analyze or report the communication cost of TAP.

4. The proposed two-stage design (adaptive replacement and knowledge distillation) largely combines existing personalization and KD techniques in a straightforward way. Adaptive replacement is conceptually similar to selective parameter updates or client-specific gating, and post-FL KD is already common.

5.  The paper does not provide a clear discussion of its limitations or potential failure scenarios. While TAP shows consistent improvements across benchmarks, the authors do not analyze under what conditions the method might underperform.

6. The overall organization of the paper could be improved, especially in the Method section, which is dense and sometimes difficult to follow due to lengthy descriptions and intertwined explanations of architecture and training procedures. In addition, there are several presentation issues and minor errors that affect readability. For example, line 161 (“with it being being fine-tuned with LoRA...”) contains a duplicated word, and the section title 3.1 appears to include a typo (“Multi-Mask” should be “Multi-Task”).

### Questions
How does the proposed method handle aggregation across heterogeneous client architectures? Specifically, when clients possess different subsets of $\widetilde{W}$ due to varying tasks and modalities, how are these mismatched components aligned or mapped to enable FedAvg-style aggregation without a shared latent space?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 6

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an architecture for personalized federated learning dealing with multiple modalities, where different clients focus on potentially different tasks and different modalities, so only part of the full model is shared with a given client. Using a special architecture, building on the idea of Mixture of Experts, all client models get integrated in the global model, which is distributed again at the end of learning to post-train the local models via knowledge distillation.

### Strengths
The paper addresses the interesting yet challenging setting where different clients share only part of a global model, caused by different modalities and/or different tasks (mostly output modalities/formats).

### Weaknesses
1. The setup still seems to make quite a few assumptions about clients 'overlapping' and sharing a common goal, and it would be good to make those explicit. How similar or aligned should the tasks be to make sharing better than splitting ? 

2. The architecture of the global model is chosen specifically for this setup. All baselines and competing methods are compared against the proposed method using this same architecture, while they were not specifically designed for it (unlike the proposed method).
The results should be compared against more standard architectures, to be able to appreciate the results. 

3. There's little discussion in the paper about limitations of the proposed scheme.

4. The presentation of the paper could be improved. In particular the figures showing the overall architecture are not very clear.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 7

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies the problem of personalized fine-tuning of foundation models (with multi-task and multi-modal properties) in a Federated Learning (FL) setup. In this setting, the nature of heterogeneity in FL is expanded to include not just data heterogeneity, but also task and modality heterogeneity across clients. The paper develops a new learning algorithm called TAP for this scenario that is empirically shown to do well on accuracy metrics under several model architectures and datasets. TAP is further supported by a theoretical convergence analysis of the server model.

### Strengths
(S1) The problem at hand (pFL fine-tuning of foundation models with task, modality, data heterogeneity) is realistic and very important in practice. While the components used to build up TAP have appeared previously in published literature, putting them together to design a solution has originality.

(S2) The paper is well-written, and the presentation is clear. The problem is well motivated, and contributions of the paper are largely well contextualized w.r.t. cited prior work. TAP's design is well explained with adequate intuitions provided throughout.

(S3) Theoretical convergence analysis is presented for TAP under assumptions that are standard in FL literature. The theorem statement and proof look correct to me. The analysis also suggests a non-trivial insight that convergence rate could slow down as a larger number of modality-task pairs are catered to by the server model. The paper also presents good (could be improved upon) experimental support for the performance of TAP.

### Weaknesses
(W1) [Flow] has previously introduced the idea of dynamic routing that resembles the replacement mechanism discussed in section 3.2. Also, [FedHCA] presents several experiments which do cover multi-modal and multi-task FL setup. These should be cited and the differences briefly stated in the paper.

[Flow] Panchal, K., Choudhary, S., Parikh, N., Zhang, L., & Guan, H. (2023). Flow: Per-instance personalized federated learning through dynamic routing. 37th Annual Conference on Neural Information Processing Systems (NeurIPS 2023)

[FedHCA] Lu, Y., Huang, S., Yang, Y., Sirejiding, S., Ding, Y., & Lu, H. (2024). FedHCA$^2$: Towards Hetero-Client Federated Multi-Task Learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4106-4115.

(W2) Lines 345-347: While this is theoretically indicated by equation (7), does it actually show up in practice? I couldn't find which experiment in the paper would support/refute this possibility.

(W3) In the main body of the paper, there is no concise supporting statistics/discussion on the necessity of $R_i\[o\]$ based selection for the problem at hand. As best I could tell, supporting evidence for some tasks is only available in Appendix C.0.3 through Fig. 3 & 4.

Things to improve the paper that did not impact the score:
- Lines 151, 155 - $\mathcal M$ and $\mathcal O$ are undefined.

### Questions
(Q1) Based on details of DisentAFL and TAP, I don't understand why TAP should outperform DisentAFL/DisentAFL + Post-train. I couldn't find any targetted discussion around this in the paper. Could the authors explain?

(Q2) Lines 200-202: Does the server need to know the tasks and modalities at each client? Are there privacy/security implications?

(Q3) Section 5.2 Tables 1-3: Should there be different numbers to judge generalization and personalization separately?

### Soundness
3

### Presentation
3

### Contribution
3

# Toward Enhancing Representation Learning in Federated Multi-Task Settings

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Federated multi-task learning (FMTL) seeks to collaboratively train customized models for users with different tasks while preserving data privacy. Most existing approaches assume model congruity (i.e., the use of fully or partially homogeneous models) across users, which limits their applicability in realistic settings. To overcome this limitation, we aim to learn a shared representation space across tasks rather than shared model parameters. To this end, we propose *Muscle loss*, a novel contrastive learning objective that simultaneously aligns representations from all participating models. Unlike existing multi-view or multi-model contrastive methods, which typically align models pairwise, Muscle loss can effectively capture dependencies across tasks because its minimization is equivalent to the maximization of mutual information among all the models' representations. Building on this principle, we develop *FedMuscle*, a practical and communication-efficient FMTL algorithm that naturally handles both model and task heterogeneity. Experiments on diverse image and language tasks demonstrate that FedMuscle consistently outperforms state-of-the-art baselines, delivering substantial improvements and robust performance across heterogeneous settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses federated multi-task learning (FMTL) with heterogeneous models and tasks. The authors propose Muscle loss, a novel contrastive learning objective that aligns representations from multiple models simultaneously by capturing dependencies among all models' representations. Building on this, they develop FedMuscle, a practical FMTL algorithm where users transmit representation vectors of public data to a server, which computes aggregated matrices and weighting coefficients. The paper demonstrates that minimizing Muscle loss is equivalent to maximizing a lower bound on mutual information among models' representations. Experiments on image and language tasks show consistent improvements over baselines.

### Strengths
- The shift from parameter-sharing to representation-alignment is conceptually appealing and well-motivated. The paper addresses a genuine gap in FMTL: most existing methods assume model congruity, while this work handles arbitrary heterogeneous architectures. The focus on learning a shared representation space rather than shared parameters is a valuable perspective.
- The derivation of the Muscle loss from first principles through probability density ratios (Appendix A) demonstrates rigor. Theorem 1 establishing the MI lower bound is meaningful and provides principled justification for the approach. The analysis comparing Muscle loss to pairwise alignment from an MI perspective (Appendix F) clearly shows why capturing joint dependencies matters. 
- FedMuscle is well-engineered for federated settings. The communication cost reduction via random selection of M representation matrices (Algorithm 1) is pragmatic. Users don't reveal local model parameters (only representations on public data) which addresses privacy concerns better than some alternatives. The modularity of computation (server computes aggregations while users perform gradient updates) is efficient.
- The paper evaluates across diverse tasks (image classification, semantic segmentation, text classification) with multiple datasets. The comparison with numerous baselines (FedRCL, SAGE, CoFED, FedDF, FedHeNN) is thorough. Setup2's multi-modal evaluation strengthens claims about generality. Integration into CreamFL demonstrates broader applicability. Ablation studies on E, R, T, M, and d are helpful.

### Weaknesses
- A fundamental limitation inadequately addressed is how to decompose each heterogeneous model into representation model $w_n$ and task-specific head $\phi_n$. The paper assumes "the output features of all representation models have the same dimension d" and notes this "requirement does not preclude model heterogeneity" and "can be relaxed by appending a lightweight, learnable projection head." This is vague. For practitioners: What if optimal representation dimensions differ by task?, How do you choose which layers constitute the encoder vs. decoder for arbitrary architectures?, Is the projection head frozen or jointly trained? How is its initialization determined?
This choice fundamentally affects what "shared representation space" means and is never empirically investigated.
- FedMuscle critically relies on a shared public dataset D. The paper notes this dataset "may be uni-modal or multi-modal, depending on the users' tasks" and can be "easily obtained." However, in Setup1, using CIFAR-100 (lower performance) vs. COCO (higher performance) shows ~12% performance swing in $\Delta$. For multi-modal Setup2 with mixed CV/NLP tasks, using Flickr30K captions for both modalities seems contrived. Are image captions truly representative of what text classification models need? Real federated scenarios may not have access to high-quality public data relevant to all tasks. The paper doesn't analyze sensitivity to public data quality/task relevance rigorously.
- The multi-modal evaluation in Setup2 mixes semantically distant tasks. Users 1-6 use vision tasks on CV data, while users 9-10 do text classification. The alignment mechanism forces all to learn representations in the same space using Flickr30K captions as the public dataset. This seems artificial. How does aligning a semantic segmentation model with a text classifier via image captions facilitate knowledge transfer? The small improvements for some users (SS tasks in Table 2) may reflect this semantic distance. The paper claims the approach "handles both model and task heterogeneity," but doesn't sufficiently explore when heterogeneity becomes too extreme for representation alignment to be beneficial.
- Some recent SOTA works on FMTL are missing, for instance:

   - Yipan We, Yuchen Zou, Yapeng Li, Bo Du, “Towards Unified Modeling in Federated Multi-Task Learning via Subspace Decoupling”, arXiv preprint arXiv:2505.24185, 2025.
   - Yuxiang Lu, Suizhi Huang, Yuwen Yang, Shalayiding Sirejiding, Yue Ding , Hongtao Lu, “FEDHCA2 : Towards Hetero-Client Federated Multi-Task Learning”, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
   - Chaouki Ben Issaid, Praneeth Vepakomma, and Mehdi Bennis. "Tackling Feature and Sample Heterogeneity in Decentralized Multi-Task Learning: A Sheaf-Theoretic Approach." Transactions on Machine Learning Research, 2025.

### Questions
Refer to the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This study offers FedMuscle, a new federated multi-task learning (FMTL) approach that deals with model and task heterogeneity by aligning latent representations rather than exchanging model parameters.  The key concept is Muscle loss, a theoretically justified contrastive learning objective that aims to jointly align representations from several diverse clients rather than doing pairwise alignments like standard InfoNCE-based approaches.

### Strengths
The paper covers a wide range of model structures and task types.

The MI lower-bound perspective is nicely integrated with the contrastive framework.

Covers both unimodal and multimodal configurations; the results are consistent.

Clear improvement over strong baselines (FedHeNN, CoFED, CreamFL, and Muscle Loss).

### Weaknesses
The Muscle loss is simply a weighted multi-view InfoNCE variation; comparable theories exist, such as Gramian losses and multi-view MI maximization.

Dependence on a shared public dataset undermines the privacy argument and limits usefulness in confined contexts.

There are no theoretical assurances for FedMuscle's convergence or stability under customer heterogeneity.

Weak ablations include no sensitivity analysis on critical hyperparameters (τ, M, B) or comparison to adaptive optimizers (FedAdam, FedNova).

Lack of interpretability of "representation alignment" and it is unclear what semantic information is transmitted between tasks or how this affects per-task specialization.

### Questions
How sensitive is FedMuscle's performance to the number of clients (N) and sample parameter (M)?

Can muscle loss deal with missing modalities or imbalanced representation spaces across tasks?

How plausible is the premise of a common unlabeled dataset in medical or financial FL contexts?

How does the approach perform under highly skewed non-IID distributions or in asynchronous settings?

Could the authors compare the runtime (in hours or GPU-days) to FedHeNN or FedRCL?

Is there a theoretical bound on the communication cost-performance trade-off?

Is there a demonstrable association between the MI limit and Δ performance in experiments?

### Soundness
3

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
The authors propose an interesting idea of learning a shared representation space in federated multi-task learning settings. The proposed Muscle loss effectively captures dependencies across tasks, and the FedMuscle framework outperforms state-of-the-art baselines, demonstrating the effectiveness of the proposed method.

### Strengths
1. The authors provide a solid theoretical analysis of the proposed Muscle loss function.

2. The proposed FedMuscle algorithm can be applied to various tasks, including computer vision and natural language processing, demonstrating its capability to handle model and task heterogeneity in federated learning.

3. The contrastive Muscle loss can be seamlessly integrated into multimodal approaches.

### Weaknesses
From the perspective of federated learning theory analysis, the current work lacks proofs of convergence and generalization. Of course, this would be a substantial undertaking, and perhaps more in-depth theoretical analysis in this regard can be considered and refined in future research.

### Questions
Please see weaknesses.

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
4

### Summary
This paper proposes to learn a shared realistic representation space across tasks in Federated multi-task learning (FMTL) rather than shared model parameters, as most existing approaches assume the use of fully or partially homogeneous models across users, which limits their applicability in realistic settings.

### Strengths
The paper indicates a common limiting assumption among federated multitask
learning approaches (model congruity) and proposes a new method
(FedMuscle) to overcome this limitation. The paper justifies its method
via theoretical results showing that their approach maximizes the mutual
information among the models. The paper conducts experiments on both
image (ViT, SegFormer) and text (BERT, DistilBERT) domains to justify
the performance of FedMuscle compared to various baseline algorithms.

### Weaknesses
The requirement of a shared public dataset is very strong, particularly for
federated learning scenarios. On page 4, lines 163 to 165, this issue is addressed
suggesting using publicly available datasets or synthetic data samples but there are concerns of model collapse with synthetic data (though said concerns mostly focus on the recursively generated data by the model or
model family itself reinforcing its own biases, and said issue is resolved via
adding non-synthetic data which the local users’ private database would be
non-synthetic). The authors did not focus on any results on their method’s performance on synthetic data, so this is unclear.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

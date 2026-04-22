# Cross-View Lewis Weight Fusion Empowering Exemplar Replay for Federated Class-Incremental Learning

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Federated Class-Incremental Learning (FCIL) aims to continually expand a model’s recognition capacity in a distributed environment, enabling it to learn new classes while retaining knowledge of previously seen ones. Exemplar replay has emerged as a promising strategy owing to its simplicity and effectiveness. Existing methods either select exemplars based on local dynamics or construct global feature spaces to identify representative samples. However, they face inherent challenges in striking a balance between effectiveness and privacy. To address this issue, this paper proposes a Cross-views Lewis weIght Fusion method for exemplar replay in FCIL, termed CLIF, which fuses multi-view importance scores to guide representative sample selection under federated settings. Specifically, CLIF consists of two main modules: 1) the cross-view Lewis weight fusion module computes and integrates Lewis weights from multiple feature perspectives to achieve consistent importance estimation, ensuring that the selected samples better reflect the global data distribution and thus enhancing the representativeness of the replay subset. Building on this, 2) the frequency-based weighted training module adjusts the loss contribution of each sample according to its selection frequency across views, which emphasizes the contribution of critical samples. Moreover, we provide a theoretical analysis to guarantee the soundness and effectiveness of CLIF. Extensive experiments on three datasets demonstrate that our method consistently improves baselines by 1%–6%, supporting the above claims.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a method, CLIF, for balancing effectiveness and privacy in exemplar replay-based Federated Class-Incremental Learning (FCIL), which uses multi-view importance scores to select representative samples. In order to achieve that, they perform consistent importance estimation by computing and integrating the cross-view Lewis weights to ensure the representativeness of the replay subset consisting of selected samples. Based on this, they depend on the selection frequency of each sample to adjust its contribution to the optimization objective during the frequency-based weighted training. They present a theoretical analysis to guarantee the soundness and effectiveness of CLIF.

### Strengths
1. The paper presents a novel method of exemplar relay to mitigate the limitations of effectiveness and rigorous theoretical guarantees in prior works. The idea of Lewis weight fusion across views is original and contributes to the literature on Federated Class-Incremental Learning.
2. The paper demonstrates solid experimental quality, with well-designed comparisons against strong baselines such as Re-Fed+ and FedCBDR. The ablation studies and analysis under different hyperparameter settings are comprehensive.
3. The paper is generally well-written and logically structured. The related work identifies the limitations of current studies, clarifying the research gap and motivating the proposed solution. The motivation, methodology, and experiments are easy to follow. The figures and tables effectively illustrate the main findings.
4. The paper addresses an important problem in Federated Class-Incremental Learning by considering both local and global views, which is significant for real-world applications such as distributed model training and lifelong learning across devices in heterogeneous settings.

### Weaknesses
For the introduction and methodology, the following should be addressed.
1. In Line 72 of Section 1, the statement that “global-view methods address this limitation at the cost of privacy risks due to cross-device feature collection (Qi et al., 2025)” seems somewhat misleading. Actually, FedCBDR (Qi et al., 2025) applies inverse singular value decomposition (ISVD) to obtain pseudo features rather than direct feature sharing, which can be considered a privacy-preserving manner.
2. Although the motivation of employing Lewis weight sampling for selecting representative samples in FCIL is introduced in Section 3.2, it would be clearer if the authors could explicitly discuss why preserving the operator norm of the feature map is particularly beneficial for FCIL.
3. According to Algorithm 1, models of other clients are sampled randomly. It would be valuable if the authors could explore or discuss other potential model selection methods beyond random sampling, which might affect the overall performance.

For the experiments, the following should be addressed.
1. The paper mentions that three datasets CIFAR10, CIFAR100, and Tiny-ImageNet are partitioned into different numbers of tasks (3/5/5 and 5/10/10). However, the rationale for choosing these specific task splits is not discussed.
2. In Section 4, the authors mainly evaluate two variants of the proposed CLIF combined with Re-Fed+ and FedCBDR to compare with other approaches. The authors should clarify why the original CLIF method is not compared.
3. Section F.4 presents the large equivalence in effectiveness of proposed privacy-enhanced alternatives and original schemes. However, the authors do not provide an analysis to discuss whether these alternatives genuinely mitigate the risk of privacy leakage.

Minor comments:
1. Page 14: In Line 16 of Algorithm 1, Line 225 does not contain any relevant description of cross-view fusion with the max rule. The authors should check the reference.
2. In Line 119, Re-Fed should be Re-Fed+.

### Questions
If the issues raised in the introduction, methodology, and experiments in the author's response are well clarified, I would be willing to increase the score.

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
The paper introduces an exemplar selection method for Federated Class-Incremental Learning (FCIL). The core idea is the cross-view lewis weight fusion (CV-LWF), which leverages feature extractors from multiple clients to compute and fuse Lewis weight importance scores. This fusion aims to select a subset of exemplars that collectively preserve the feature subspace from a global perspective, thereby mitigating the local bias inherent in FL. The authors supplement this with a frequency-aware weighted training (FWT) strategy and show marginal performance gains over existing baselines.

### Strengths
- The proposed strategy offers a drop-in replacement for the exemplar selection and replay mechanism that improves the performance of existing baselines across different datasets and heterogeneity settings.

- The paper addresses the critical and highly practical challenge of FCIL，which is crucial for realizing practical and ethical AI in distributed systems. Solving the catastrophic forgetting problem within the Non-IID and privacy-preserving constraints of FL is a challenging goal.

### Weaknesses
- The technical contribution is severely limited. The paper merely combines an existing, well-known Lewis weights scores with a common aggregation technique. The core method is a straightforward application rather than a novel development. Furthermore, the demonstrated performance improvements are marginal across benchmarks.

- The calculation of $l_p$-Lewis Weights is computationally expensive, often involving iterative schemes or matrix inversion ($\mathcal{O}(d^2 n)$ or worse, where $d$ is feature dimension and $n$ is data size). CLIF multiplies this cost by requiring $M$ such computations per client per task, where $M$ is the number of views. This excessive computational burden on resource-constrained edge devices makes the entire approach practically infeasible and unscalable for real-world FL.
- The field of exemplar selection is widely known as Coreset Selection or Data Pruning. The paper critically fails to compare its Lewis Weight-based method against recently established coreset selection techniques. 
- The reliance on only small-scale datasets (CIFAR-10/100, TinyImageNet) is inadequate. Demonstrating efficacy on a large-scale benchmark like ImageNet is mandatory for a modern machine learning paper claiming general utility.
- The use of ResNet-18 as the architecture is highly outdated. Current state-of-the-art in both incremental and federated learning utilizes larger transformer-based architectures. The performance characteristics of the proposed method on modern, high-capacity models remains entirely unknown, severely limiting the applicability of the findings.
- The experiments are conducted entirely by training models from scratch. This does not reflect the current reality of deep learning, where models are almost always initialized with pre-trained weights, e.g., from ImageNet, or self-supervised methods like CLIP. The problem of FCIL applied to a high-quality feature space is different from training a small ResNet-18 from scratch. The lack of experiments in the pre-training/fine-tuning paradigm renders the reported results practically irrelevant.
- The theoretical guarantees provided are derived from the literature for fixed feature matrices. In FCIL, the feature backbone and thus the matrix are continuously updated through local training on the new task and global aggregation. The exemplars are selected based on the model state before the new task begins, but they are replayed with the model that is constantly shifting due to client drift and global updates. Therefore, the theoretical preservation of the subspace, and the derived error bounds for prediction, may be immediately invalidated or severely degraded after the first few local training steps of the new task. The stability of the selected exemplars under continual model evolution is not adequately addressed.

### Questions
- The theoretical guarantee of Lewis Weights assumes a fixed feature matrix $\mathbf{A}$. Given that the backbone $\phi$ continuously drifts during local training on the new task and during global aggregation, how stable are the selected exemplars? Have you quantified the loss of the subspace embedding property after $E$ local epochs on the new task data?
- Please provide a detailed breakdown and empirical wall-clock measurement of the total computational overhead on a typical client for running the entire CV-LWF exemplar selection process, compared to simple baseline selection methods, e.g., random or nearest-to-mean.

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
This paper proposes CLIF, a Cross-view Lewis weIght Fusion framework for Federated Class-Incremental Learning (FCIL). The method aims to balance effectiveness and privacy in exemplar replay by leveraging Lewis weights from multiple feature perspectives to guide exemplar selection, by introducing Cross-View Lewis Weight Fusion (CV-LWF) and Frequency-aware Weighted Training (FWT). Experiments on CIFAR-10/100 and Tiny-ImageNet (under varying heterogeneity and client settings) show consistent gains of 1–6% over strong baselines

### Strengths
1. The paper is clearly written and well-organized.
2. The authors provide a well-structured derivation and prove that their sampling procedure preserves subspace structure for all model views simultaneously.
3. The experimental evaluation is comprehensive, covering multiple datasets, heterogeneity levels, client counts, and includes ablation studies.

### Weaknesses
1. The experiments are conducted only on CIFAR and Tiny-ImageNet. While these are standard benchmarks, they are relatively small in scale. Evaluating the method on larger or more realistic non-IID datasets would strengthen the empirical validation and demonstrate better generalizability.
2. Computing Lewis weights and performing cross-view fusion across multiple backbones may introduce considerable memory and computational overhead, particularly when using 5–10 model views. The paper should provide more detailed analyses and quantitative results regarding these costs.
3. The paper states that all methods use ResNet-18 as the backbone for fair comparison, yet also mentions that the hyperparameter settings of baselines follow their original papers. This setup may not ensure true fairness, as the optimal hyperparameters can differ across backbone architectures and experimental configurations.

### Questions
1. It would be helpful to evaluate how well the proposed method scales to a much larger number of clients (e.g., 100 or more) beyond the 20- and 40-client settings reported in the paper.
2. The paper could also investigate performance under more extreme heterogeneity scenarios, such as using a smaller Dirichlet parameter (e.g., β = 0.1), to better assess the robustness of the method under highly non-IID data distributions.

### Soundness
3

### Presentation
3

### Contribution
3

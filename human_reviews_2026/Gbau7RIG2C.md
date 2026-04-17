# FedMAP: Meta-Driven Adaptive Differential Privacy for Federated Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 0, 4, 6

## Abstract
Federated learning (FL) enables multiple clients to train a shared model without sharing raw data, but gradients can still leak sensitive information through inversion and membership inference attacks. Differential privacy (DP) mitigates this risk by clipping gradients and adding calibrated noise, but most DP-FL methods rely on static noise and clipping schedules. Such rigid designs fail to account for client heterogeneity, changing convergence dynamics, and the growth of cumulative privacy loss. To address these challenges, we propose FedMAP, a closed-loop framework for adaptive differential privacy in FL. FedMAP integrates three components. First, a client-side MetaNet predicts clipping bounds and noise scales $(C_t,\sigma_t)$ from gradient statistics using a lightweight pretrained BERT-tiny backbone, enabling effective adaptation across communication rounds. Second, a server-side Rényi DP accountant tracks heterogeneous privacy costs, computes the global expenditure $\varepsilon_{\mathrm{global}}$, and broadcasts it as a budget signal that constrains cumulative loss and guides client adaptation. Third, a global feedback regularization mechanism combines local penalties on per-round privacy cost with global penalties from $\varepsilon_{\mathrm{global}}$, ensuring alignment between client adaptation and the overall budget. Experiments show that FedMAP improves privacy compliance, and offers stronger robustness against attacks compared with baselines.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposed a novel Federated Learning framework that protects against membership inference and reconstruction attacks under Differential Privacy. At a high level, the paper fine-tunes a BERT-based model to predict the hyperparameters (C and $\sigma$) for the DP-SGD algorithm to preserve the privacy of the client's data. The paper also focuses on a scenario in which each client has their own privacy budget. Thus, the paper proposed an updating mechanism and objectives to incorporate into this setting. The paper conducts extensive experiments to highlight the advantages of their proposed method.

### Strengths
- The proposed method provides flexibility and automation in tuning DP-SGD for different clients.
- Extensive Theoretical and Experimental results are provided to support the advantage of the proposed methods.

### Weaknesses
- The motivation of the work is not convincing. Specifically, it does not explain why different clients require different privacy budgets. For instance, if we consider FL for medical data, how can a client define a privacy budget to protect this sensitive data? Previous works have approached adaptive hyperparameters (C and $\sigma$) from a performance perspective, which is more convincing. Thus, I suggest that the author clarify this point.
- Secondly, the curation process for the label of C and $\sigma$ is unclear and not optimal. How do you determine that the curated labels are the best for the subsequent iterations? Isn't this process empirical, and does the curator need to tune different C and $\sigma$ at each iteration?
- Thirdly, although the paper considers different privacy budgets for different clients. This is not highlighted in the proposed method or in how it integrates this information. Furthermore, the loss function in Eq. 12 will encourage each client to have the same privacy budget, which is in contradiction with the paper's goal.
- Next, given a predicted C and $\sigma$ from the meta model, the proposed method cannot achieve the predefined privacy budget of the clients, resulting in weak protection. How do you ensure that the consumed privacy budget is lower than the budget predefined by the clients?
- Finally, the experimental results for Table 1, Figure 3, and 4 do not mention the privacy budget for each client, which reduces the validity of these experiments.

### Questions
- Please address all the points in the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
An overly complex method is proposed for private federated learning that purports to achieve a better utility/privacy tradeoff. The method is flawed because it does not account for the privacy loss of the MetaNet mechanism.

### Strengths
No notable strengths.

### Weaknesses
In exactly the same way as another paper I have just reviewed (evidently by the same authors, since much of the text is copied), this method is overly complex and the purported gains are not supported by the experiments, and in any case would not justify the implementation and pretraining costs.

To support a claim of improved privacy-utility trade-off, one must compare model utility (e.g., test accuracy) while holding $\varepsilon$ constant across all methods. The current experiments (e.g., Figure 3) compare utility against communication rounds, which is insufficient to demonstrate a superior trade-off. Nowhere in Section 4.1 does it state that all methods were calibrated to achieve the same total privacy budget $\epsilon$ for a fair comparison.

However the most problematic issue is that the mechanism is flawed: in fact it does not provide a formal DP guarantee for any level of $\epsilon$. It releases data-dependent parameters without accounting for their privacy cost. The server broadcasts $\varepsilon_\text{global}$ to all clients at each round. $\varepsilon_\text{global}$ is dependent on the data of clients from the previous round. Suppose an honest-but-curious client $A$ participating at rounds $t-1$ and $t$ observes a large increase in $\varepsilon_\text{global}$. Client $A$ could deduce that some other client $K$ at round $t-1$ likely had low-variance per-sample gradients, leading to a small noise scale $\sigma_K^{(t)}$. Since $\varepsilon_\text{global}$ is determined deterministically, $A$ can distinguish with certainty between two datasets $\mathcal{D}$ and $\mathcal{D'}$ that are identical except that in $\mathcal{D}$, $K$ has low variance per-sample gradients, while in $\mathcal{D'}$, $K$ has high variance per-sample gradients. This violates the definition of DP.

### Questions
Is anything I stated in the weaknesses section incorrect?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes a method called FedMAP, which enhances federated learning’s defense against gradient inversion and membership inference attacks. FedMAP equips each client with a fine-tuned MetaNet that predicts clipping bounds and noise scales based on gradient statistics. On the server side, a Rényi differential privacy accountant is employed to track each client’s privacy cost and compute the overall global expenditure, which is then broadcast to all clients to constrain cumulative loss and guide adaptive local updates. Empirical experiments on standard federated learning benchmarks demonstrate that FedMAP provides stronger protection against both gradient inversion and membership inference attacks compared to existing baselines.

### Strengths
1. The paper is easy to follow.

2. The authors address both gradient inversion and membership inference attacks in federated learning, which is a challenging and important problem.

3. The idea of using a neural network to predict clipping thresholds and noise scales for differential privacy mechanisms is promising.

4. The authors provide convergence results to support the theoretical soundness of the proposed FedMAP method.

5. Extensive experiments demonstrate the effectiveness of FedMAP against multiple attack methods. Moreover, the model accuracy achieved by FedMAP remains close to that of the non-private baseline.

### Weaknesses
1. The paper lacks a detailed description of the defense and attack models, which is crucial for helping readers understand the setup and assumptions of the considered DP-FL system.

2. The proofs of Theorems 1 and 2 are missing, preventing readers from verifying their details and correctness.

3. The rationale for selecting the four specific features as inputs to the MetaNet is not well justified, and further explanation or empirical evidence would strengthen this design choice.

### Questions
1. Regarding the fine-tuning of the MetaNet, does this process occur on the client side, performed independently by each client? Clarifying where and how this fine-tuning is conducted would help readers better understand the workflow.

2. If the above is true, and a client is currently training on the CIFAR dataset, is the MetaNet fine-tuned specifically on that client’s CIFAR data, or on a mixture of datasets such as CIFAR, FMNIST, and SVHN? The explanation in lines 174–175 of the paper is unclear and should be elaborated.

3. The rationale for constructing the labels of $C$ and $\sigma$ to be proportional to empirical observations is not well justified. This label design appears ad hoc and lacks theoretical or empirical support.

4. As shown in Inequality (3), the Gaussian mechanism requires the noise scale to exceed a certain lower bound to ensure differential privacy. How does the MetaNet guarantee that the predicted noise scale always satisfies this requirement? This concern is especially important given that the proposed system does not seem to employ secure aggregation. If the server is semi-honest, it may still attempt inference attacks despite added noise.

5. How do the authors derive Inequality (10)? A step-by-step derivation or reference to supporting materials would improve clarity.

6. What is the practical meaning or role of $q_{max}$ in the paper? Its definition and influence on the overall algorithm are not clearly explained.

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
This paper presents FEDMAP, a closed-loop adaptive differential privacy (DP) framework for federated learning (FL). The key idea is to dynamically predict clipping thresholds and noise scales using a lightweight MetaNet based on a frozen-layer BERT-tiny architecture. FEDMAP further introduces global privacy accounting via Rényi DP and global feedback regularization to align local DP spending with global privacy budgets. Experiments on CIFAR-10, SVHN, and Fashion-MNIST demonstrate improved privacy-utility trade-offs and stronger robustness to gradient inversion and membership inference attacks compared to DP-SGD, Soteria, and CENSOR. Theoretical convergence guarantees and DP analyses are provided, and extensive ablations show sensitivity to client participation and hyperparameters .

Overall, the work is timely, well-motivated, and empirically solid. The adaptive privacy calibration idea is intuitive and practical. However, some algorithmic and training details are ambiguous, experiments on larger models/datasets are missing, and several theoretical statements lack formal proofs.

### Strengths
1. **Adaptive DP calibration**. The framework introduces flexible and client-specific DP noise and clipping schedules, addressing client heterogeneity in FL.

2. **Meta-learning-based privacy control**. A lightweight BERT-tiny MetaNet effectively maps gradient statistics to DP parameters, demonstrating a novel use of meta-learning for privacy.

3. **Global privacy loss regularization**. The feedback mechanism aligns local DP spending with global budgets and prevents over-consumption of privacy.

4. **Theoretical grounding**. Convergence bounds and DP accounting provide theoretical credibility.

5. **Strong empirical validation across attacks**. Experiments evaluate multiple attacks and show competitive robustness and utility against baselines.

### Weaknesses
1. **Unclear MetaNet training and update procedure (critical).**
It is ambiguous whether MetaNet parameters are frozen during private training or continually updated. The algorithm suggests frozen transformer layers and trainable heads during pretraining, but does not clarify if they continue updating during FL, and how privacy and global penalties influence MetaNet outputs during training.

2. **Scalability to large-scale FL is uncertain.**
All experiments use small vision models (ResNet-18, LeNet) and datasets. It remains unclear if the method scales to transformers or large-scale NLP tasks, where computing gradient statistics and MetaNet inference might incur overhead.

3. **Missing proofs in Appendix.**
The main theorems are stated without formal proof details, which reduces theoretical rigor.

### Questions
1. **Difference between $D_t$ and $VarGrad_t$**.
In Eq. (4), both statistics quantify gradient variability. What unique information does each contribute? Is there a redundancy?

2. **Cost of computing gradient statistics.**
For large models, computing $\\|g\\|_2$ and covariance-based metrics could be expensive. Can the author provide a detailed comparison of the actual runtime overhead on different architectures?

3. **MetaNet training and DP interaction.**
- Are MetaNet parameters frozen during private training?
- If frozen, how can global penalty terms meaningfully influence privacy control beyond inference?
- If trainable, how are MetaNet updated? Do they follow the same FL training procedure as the main model?

Clarifying this point is crucial to judge correctness and privacy guarantees.

### Soundness
2

### Presentation
3

### Contribution
3

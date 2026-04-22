# Bridging Local Divergence in Federated Learning via Personalized Adjustment Matrix

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Data distribution divergence across clients often leads to misalignment between global federated models and local decision boundaries. While existing personalized federated learning approaches attempt to mitigate this through feature alignment or multi-head personalization, they typically introduce additional communication and local computation overhead, which in turn limits their effectiveness under severe heterogeneity. In this work, we introduce FedPAM, a complementary perspective that addresses this challenge through a client-specific Personalized Adjustment Matrix (PAM) combined with a contrastive alignment objective, achieving robust personalization with minimal additional cost, while keeping the standard FedAvg communication protocol unchanged. Experiments across diverse benchmarks confirm that FedPAM improves upon competitive personalized FL baselines, showing pronounced advantages in highly heterogeneous conditions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes FedPAM, a federated learning method that introduces a simple client-side linear adapter applied to the classifier head and a supervised contrastive alignment loss. The main objective of FedPAM is to reconcile the mismatch between global classifier boundaries and locally shifted feature distributions in non-IID settings. The paper claims that FedPam is advantageous compared to other works due to a minimal communication cost (as only the adjusted head is transmitted), and provides empirical evidence demonstrating its performance.

### Strengths
- The idea of adjusting the classifier with a small local matrix  $P_k$  is computationally light and integrates well with FedAvg. From a systems standpoint, a drop-in modification to FedAvg that empirically improves stability is appealing, especially without requiring an increased network load.
- The paper adopts thorough experiments, with extensive experiment grids and ablations.
- The presentation is clear, and the figures and proofs are easy to follow.

### Weaknesses
1. The convergence analysis is technically sound but seems a bit generic; it reads like a direct instantiation of the standard local-SGD/FedAvg proof on an augmented loss $\tilde F$. If I'm not mistaken, the personalized adjustment matrix only influences constants $(\tilde L,\tilde\sigma^2)$ and the stepsize range. Thus, in the current theoretical analysis, I don't see how PAM-specific structure is used to tighten the client-drift or variance terms. As FedPAM shows substantially better performance compared to other methods, it would be much better to quantify how PAM reduces heterogeneity/variance and reflect that inside the bound.
2. The proposed classifier head adapter $W_k^{adj} = WP_k$ is mathematically equivalent to learning per-client classifier heads or linear adapters, which is already extensively explored in previous works (e.g. as you mentioned in your related work, FedPer, FedRep, FedRoD, etc.). The paper shows that PAM works basically as a mapping between the optimized head and the original classifier, only by optimizing contrastive feature–classifier alignment. However, this is still not a formal demonstration that $WP_k$ *behaves differently* than just learning $W_k$. In terms of novelty, it would be much more informative if you showed *why* matrix factorization displays new behavior beyond simple local-head adaptation.
3. The reported accuracies on CIFAR and ImageNet approach centralized ResNet-18 performance (e.g. [CIFAR-10](https://github.com/kuangliu/pytorch-cifar)) despite training with non-iid data. Providing error bounds or intuitive reasons on how federated training can approach centralized training would make the results more believable.
4. The paper never inspects or visualizes $P_k$ despite claiming it “bridges divergence.” Quantifying and visualizing $P_k$ properties like inter-client variance would substantiate the mechanism and show whether it truly aligns classifier directions rather than simply over-parameterizing the head. On top of ablating $\lambda$, it would be nice to see an experiment that visualizes $P_k$ truly contracts divergence rather than adding capacity. For example, off the top of my head, recording the accuracy drop when swapping $P_k$ between similar vs. dissimilar clients could show whether if the matrix reduces gradient heterogeneity.

### Questions
1. Can you prove (or empirically verify) that PAM reduces inter-client variance or drift vs. FedAvg on $L_{\mathrm{ce}}$ (e.g., $\tilde\sigma^2$ or an effective smoothness decrease), and specify when this benefit outweighs the larger $\tilde L=L+\lambda L_P$ that shrinks the step size window?

2.  What inductive bias or constraints on $P_k$ (orthogonality/low-rank/regularization) make $W P_k$ fundamentally different from learning $W_k$? Please include matched-DOF head-only baselines to isolate the effect of factorization.

3. I am interested in a quantification or visualization of $P_k$ between similar vs. dissimilar clients, to the extent that shows that PAM measurably reduces heterogeneity metrics (e.g., gradient covariance, classifier direction dispersion, etc.).

### Soundness
3

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
4

### Summary
The paper proposed, FedPAM, addresses the data heterogeneity through a client-specific Personalized Adjustment Matrix (PAM) combined with a contrastive alignment objective. Experiments on 4 classification benchmarks show that FedPAM improves upon competitive personalized FL baselines.

### Strengths
1. The presentation makes the proposed method easy to understand and there are convergence analysis of the proposed method. (I don't have the bandwidth to check the proofs due to the review workload).

2. The ablation study looks comprehensive.

### Weaknesses
* The literature review is not comprehensive. The data heterogeneity, particularly as a form of data imbalance and/or long-tail, has been extensively studied by the community, yet the authors failed to discuss relevant works whoes designs share the similar principle, i.e., decoupling the feature extractors and classification and keeping a personalized the classification head. Similar works include but not limited to [1] and [2].

* Important details on the experiment are missing, making it hard to assess how the results justify the claims. Also the presentation of the ablation study section is confusing. Please see the `Questions` section for more details.

* The novelty seems to be limited. FedPAM's design seems to be a inspired by FedRod[3], FedNH[4], and FedProto[5]. On the architecture design side, it is the same as the FedRod in the sense that both have global and personalized classification heads. On the loss design side, although the actual loss functions for the personalized heads are different, FedPAM is similar to FedNH and FedProto in the sense that all try to align the extract features to the corresponding class anchor (one row of the classification head matrix). So it looks to me the  novelty is from the Personalized Contrastive loss (PCL). The experiments may justify the effectiveness of this design by showing FedPAM outperforms the FedROD. But its effectiveness  compared to FedNH and FedProto is unknown.


[1] Oh, Jaehoon, SangMook Kim, and Se-Young Yun. "FedBABU: Toward Enhanced Representation for Federated Image Classification." International Conference on Learning Representations.

[2] Shang, Xinyi, et al. "Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features."

[3] Chen, Hong-You, and Wei-Lun Chao. "On Bridging Generic and Personalized Federated Learning for Image Classification." International Conference on Learning Representations.

[4] Dai, Yutong, Zeyuan Chen, Junnan Li, Shelby Heinecke, Lichao Sun, and Ran Xu. "Tackling data heterogeneity in federated learning with class prototypes." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 6, pp. 7314-7322. 2023.

[5] Tan, Yue, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, and Chengqi Zhang. "Fedproto: Federated prototype learning across heterogeneous clients." In Proceedings of the AAAI conference on artificial intelligence, vol. 36, no. 8, pp. 8432-8440. 2022.

### Questions
* In line 51-53, the authors claim "One naive solution is to give each client an independent classifier head, but this sacrifices shared knowledge, increases communication and storage costs, and risks overfitting to limited local data.". Why having a separate classification head would increase the communication? As the author mentioned in Table 8, FedROD has the same communication cost as FedPAM. And FedBABU [1] would have $(1-\alpha)\Sigma$ communication costs.

* What does the `classifier feature interface` mean in line 93?

* My understanding is that the Personalized Adjusted Matrix $P_k$ is of size $C\times d$. But in the Ablation on PAM capacity section (line 431), the authors introduced an additional parameter $m$. I initially thought $m=d$, but authors also mentioned that "...an MLP after the feature extractor can project features from d to m". So I am not entirely sure what is $m$. And it looks like the ablation is on a MLP module, which is not mentioned in the Figure 1.

* What is the evaluation protocol? I am not sure how the test set is constructed for each client. Does all the clients share the same test set? If so, how is the test accuracy calculated? Does it follows the same protocol used in FedROD by re-weight the accuracy according to clients’ class distributions? These details are not disclosed, making it hard to understand how these results justify the contributions and effectiveness.

* In section 4 training protocol, "Each round, the server samples a fraction r of clients", what is the $r$ used? How does it connects the dropout ratio $p$? Is $p=1-r$? How should I understand $p=1$ and $p\in[0.1, 1]$?  
By definition, "We simulate random client dropout with probability p". For $p=1$, does this mean all clients dropped on all rounds? This does not make sense to me. 

* How does the under partial participation setting (Table 4) differ from the random dropout settings (table 3)?

* For table 2, how is the heterogeneity set for the scalability experiments? Methods like FedAvG, FedProx, FedRODare quite robust as $N$ increases, while there a double digits drop in FedPAM. Is it possible at some point when $N$ is large enough, FedPAM's performace converges to FedAvg?  This would weaken the claim on the scalability of FedPAM.

* In table 4, why increasing the number of participated clients hurt the performance? Most methods, except FedCP, show that higher participation ratio leads to higher accuracy, which is aligned with the intuition. 

**Minor Comments**
1. Failed to disclose the usage of large language model as required.

### Soundness
2

### Presentation
2

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
This paper proposes a personalized federated learning method that introduces a Personalized Adjustment Matrix (PAM) applied to the classifier head, along with a supervised contrastive alignment loss. The method aims to address classifier–feature misalignment under heterogeneous client distributions while maintaining zero additional communication cost, fully reusing the FedAvg protocol. The authors provide theoretical convergence guarantees and extensive empirical evaluation across multiple benchmarks.

### Strengths
1. FedPAM achieves state-of-the-art performance without extra communication cost and competitive local runtime.
2. Unlike baselines that degrade sharply under high heterogeneity, FedPAM maintains leading accuracy, showing great robustness.
3. The convergence analysis provides formal guarantees for FedPAM's stability.

### Weaknesses
1. The paper states PAM is "lightweight" but does not quantify its parameter count relative to the global model. For a ResNet-18 backbone and C=100 classes, a PAM of size 512×512 adds about 262k parameters per client—this is non-trivial for edge devices with limited memory. It will be helpful to provide analysis of how PAM size (m) impacts memory usage or inference latency.
2. It will be helpful to provide analysis of how λ generalizes across datasets and how to automate λ selection.
3. PCL mixes class-anchor negatives and in-batch negatives. The rationale behind this combination and sensitivity to batch size could be elaborated and empirically tested.

### Questions
1. Please see the weaknesses above.
2. Would FedPAM work similarly with transformer architectures and text? Does the classifier-feature mismatch arise similarly for LLM-fine-tuning scenarios?
3. If local datasets are extremely small (e.g., <100 samples per client), does PAM cause overfitting?

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
The paper proposes a personalized FL scheme that keeps a shared classifier $W$ but equips each client with a Personalized Adjustment Matrix $P_k$. Training uses standard cross-entropy on unadjusted logits $Wz$ plus a InfoNCE-style contrastive loss that draws features toward the adjusted anchors and pushes them away from other anchors/negatives. Experiments report gains over several PFL baselines.

### Strengths
- Personalization is localized to a small matrix that modulates the shared head—conceptually simple and computationally light.
- PAMs remain local, keeping communication identical to FedAvg. 
- Consistent empirical gains across several datasets and heterogeneity regimes, with sensible ablations on the PAM capacity and the contrastive-loss weight.

### Weaknesses
- Since $(WP_k) z = W(P_k z)$, the method is effectively a single client-local linear layer before the shared head. Beyond empirical results, the paper does not convincingly argue why this interface is superior to a simple baseline where each client keeps its own classifier head with a matched parameter budget
- The FedAvg-style convergence result is too generic to explain why the supervised contrastive term should improve performance in this setup (or under which conditions it would fail).
- Using classifier rows (after adjustment) as anchors is plausible, but the paper offers no theoretical or empirical evidence that this is better than feature prototypes (e.g., class means) or other prototype formulations.
- In the contrastive loss, anchors are learned classifier rows; with non-IID label imbalance, both the anchors and temperature can bias toward majority classes. The paper lacks per-class, worst-client, or percentile metrics to assess this.
- Experiments focus on small CNNs/ResNet-18; there are no larger backbones or non-vision tasks, leaving scalability and generality unclear.
- Evaluation focuses on global/mean accuracy; lacks per-client, worst-client, and percentile metrics that are standard for personalization and fairness.

### Questions
- Why is PAM preferable to a per-client linear head? Please provide motivation and experiments that justify this design choice over a matched-parameter per-client head.

- How do anchors behave under severe label skew? Since per-class anchors can be sensitive to imbalance, what mitigation (e.g., class--balanced sampling, reweighting, temperature/logit adjustment) do you use?

- Does FedPAM improve global accuracy, local personalization, or both? Given that you add a contrastive loss to CE, is there a trade-off between global and personalized accuracy? Please show this trade-off (e.g., as the contrastive weight varies).

### Soundness
2

### Presentation
3

### Contribution
2

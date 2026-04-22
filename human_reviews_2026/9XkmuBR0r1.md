# Deploying Models to Non-participating Clients in Federated Learning without Fine-tuning: A Hypernetwork-based Approach

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 4, 6

## Abstract
Federated Learning (FL) has emerged as a promising paradigm for privacy-preserving collaborative learning, yet data heterogeneity remains a critical challenge. While existing methods achieve progress in addressing data heterogeneity for participating clients, they fail to generalize to non-participating clients with in-domain distribution shifts and resource constraints. To mitigate this issue, we present HyperFedZero, a novel method that dynamically generates specialized models via a hypernetwork conditioned on distribution-aware embeddings. Our approach explicitly incorporates distribution-aware inductive biases into the model's forward pass, extracting robust distribution embeddings using a NoisyEmbed-enhanced extractor with a Balancing Penalty, effectively preventing feature collapse. The hypernetwork then leverages these embeddings to generate specialized models chunk-by-chunk for non-participating clients, ensuring adaptability to their unique data distributions. Extensive experiments on multiple datasets and models demonstrate HyperFedZero's remarkable performance, surpassing competing methods consistently with minimal computational, storage, and communication overhead. Moreover, ablation studies and visualizations further validate the necessity of each component, confirming meaningful adaptations and validating the effectiveness of HyperFedZero.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes HyperFedZero, a hypernetwork-based federated learning (FL) framework that aims to deploy models to non-participating clients without fine-tuning, addressing in-domain distribution shifts among clients. The method introduces two main components: A distribution extractor that generates distribution embeddings with NoisyEmbed and a Balancing Penalty to prevent feature collapse. A chunked hypernetwork that dynamically generates classifier parameters conditioned on these embeddings, enabling zero-shot adaptation. Experiments on seven datasets (MNIST, FMNIST, EMNIST, SVHN, CIFAR-10/100, Tiny-ImageNet) and multiple models demonstrate that HyperFedZero outperforms various baselines (FedAvg, FedProx, Ditto, pFedMe, FedJets, etc.) in zero-shot generalization to unseen clients, while maintaining comparable performance on seen clients and similar computational complexity.

### Strengths
+ The experimental evaluation is extensive. The empirical results are extensive across datasets and architectures. The ablation studies (Table 2) explore the sensitivity to embedding dimension, $alpha$, $\beta$, hypernetwork size, and chunk size. The visualization (Fig. 4) provides intuitive evidence of embedding separation and adaptation to unseen clients.

+ HyperFedZero avoids client-side fine-tuning, offering potential efficiency for resource-limited devices. This advantage can support practical usage.

### Weaknesses
- The novelty of this paper seems to be incremental. The combination of hypernetworks and FL personalization has been explored before. The main contribution (adding NoisyEmbed and Balancing Penalty) feels incremental, and the conceptual advancement beyond existing hypernetwork-based FL is modest.
- Theoretical foundation is limited. This paper lacks formal analysis on convergence, generalization bounds, or robustness. Equations (3)–(7) describe the mechanism, but there is no theoretical justification for why the method improves adaptation to unseen clients or prevents feature collapse. The “balancing penalty” is presented heuristically without rigorous connection to representation learning theory.
- The empirical advantage seems ambiguous, and the baselines are too old. While results in Table 1 show improvements, the margins are often small (1–2%) over FedJets or Scaffold on many datasets. No statistical significance tests or variance estimates are provided. The baselines are mostly before 2022. More recent or stronger baselines are required to compare to highlight the contribution.

### Questions
1. Can the authors provide any formal analysis or theoretical intuition for why the Balancing Penalty should mitigate feature collapse or why the overall system should converge? Have the authors considered adding empirical convergence plots or stability analyses (e.g., gradient variance over rounds) to support this claim? Would it be possible to quantify the benefit of the Balancing Penalty using representation metrics (e.g., intra-class variance, cosine similarity between embeddings)?
2. Could the authors justify the choice of baselines? Many compared methods (e.g., FedAMP) are relatively old. Why were more recent FL personalization methods not included? Could the authors add more recent baselines? How sensitive are the reported gains to the choice of datasets or model architectures? The reported gains are often within 1–2%. Could the authors provide statistical significance tests or error bars to support that these improvements are robust and not due to variance?
3. The paper’s main novelty seems to lie in introducing NoisyEmbed and the Balancing Penalty within a hypernetwork-based FL framework. Could the authors elaborate more concretely on how these components differ conceptually and mechanistically from prior hypernetwork-based FL methods? Is there any qualitative or theoretical evidence showing that NoisyEmbed or Balancing Penalty lead to non-trivial new behavior (e.g., improved feature geometry, stability, or generalization) compared to these earlier approaches?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a method to mitigate the data heterogeneity problem for non-participating clients in federated learning. It generates local model parameters through a hypernetwork conditioned on distribution-aware embeddings. These distribution-aware embeddings and the hypernetwork help the non-participating client to adapt to their data distribution without fine-tuning. The proposed method is compared against various baselines involving vanilla FL, in-domain with/without distributional shifts, and out of domain. Through complexity analysis, the paper argues that no additional computational overhead is incurred, and time, space complexity are the same as the baselines

### Strengths
1)The paper is generally well written. The problem is well defined, and the methodology is well presented. 
2)The ability of the proposed method to work without fine-tuning while still maintaining similar complexity as the baselines is practical and effective for the datasets and the settings presented. 
3)The paper compares their method against several baselines, showing better results consistently, and also provides ablations for various design choices.

### Weaknesses
1)Although with the evaluated datasets and model architectures, the overall model size of HyperFedZero is comparable to FedAvg, it may not scale well with more complex models or datasets while still maintaining the performance. 
2)The ablation study shows the method is sensitive to hyperparameters used, suggesting it may require extra careful tuning to get optimal results. 
3)The paper mentions their method maintains privacy, but if the hypernetwork and distribution extractor are shared among the clients any possible privacy issues that may arise could have been discussed.

### Questions
Is there any reasoning behind not using a larger number of total clients (e.g., 100) and a larger number of non-participating clients (e.g., 10,15) ? How would these changes affect the performance of the method?

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
4

### Summary
This paper focus on the in-distribution generalization problem to non-participating clients in FL under the constrained resource. Specifically, this paper propose HyperFedZero to generate specialized models. Extensive experiments demonstrate the efficacy of this framework.

### Strengths
1. Extensive experiments on a wide array of datasets

### Weaknesses
1. The setting of the article lacks authenticity. It seems that the author deliberately create a setting for this method. In the introduction part, the author mentions that such a scenario hinders further application in healthcare or edge computing. Can the author provide reference paper or a dataset to prove that such a problem does indeed exist among them? There is also no related work about this setting. 
2. The writing of the paper needs improvement. In the Introduction, the author did not explain why it is necessary to integrate distribution-aware inductive biases into the forward propagation process in order to improve the performance on unseen clients about the question “Can we directly encode distribution-aware inductive biases into the model’s forward pass in FL without fine-tuning?”
3. What is the distribution embedding? Is there any difference between the embedding extracted by the feature extractor? It is useful to clear this concept. It is also confused about the learning objective during phase 1. It seems no $e$ in $F_i$ for local training. 
4. Is there any further demonstration about the claim on Line 252-255 to choose Opt 2 from the theoretical perspective? There is also lack the detailed experimental of Opt 1 in the comparisons between condition options part.
5. The entropy regularization ($\mathbf{E}(-\mathbf{e}_i \log \mathbf{e}_i)$) does not specify the support or partitioning in Eq(4).
6. The chunked hypernetwork mechanism is described at a high level, but precise details—such as how chunks map to classifier parameter tensors, sampling strategies, and if/how temperature or batch normalization interact—are missing in the main paper. 
7. Based on the experimental results, there seems to be no significant performance difference between HyperFedZero and the traditional PFL method. However, as shown in Figure 1, left, there does appear to be a considerable gap. This could mislead others into thinking that the old method is completely unsuitable, causing the overall paper to overstate its contribution.
8. No experiments on real-world, large-scale, or out-of-domain datasets where client data distributions are highly skewed. Especially, the introduction highlights that the feature shifts on the unseen client were not tested in the experiment.
9. There is a light interpretability analysis explaining, for instance, how the distribution embeddings actually track, decompose, or separate salient client-side factors contributing to in-domain shift. More concrete case studies, e.g., visualizing misclassified groups or measuring embedding drift for new clients, would help.
10. Can the authors clarify precisely how the balancing penalty in Equation 4 is computed? Is the variance and mean calculated across clients, batches, or globally? How sensitive are the results to these aggregation choices and the entropy regularization?
11. There are some typos there, e.g, What is the meaning of 3SFC in line 315-316?
12. This problem can also be solved through test-time adaptation (TTA). Has the author compared this method with the TTA method in FL to further clarify its superiority, e.g., [R1-R3]?

References:

[R1] Jiang L, Lin T. Test-time robust personalization for federated learning[C]//Eleventh International Conference on Learning Representations, September 2022.

[R2] Bao W, Wei T, Wang H, et al. Adaptive test-time personalization for federated learning[J]. Advances in neural information processing systems, 2023, 36: 77882-77914.

[R3] Liang H, Zhang X, Cao S, et al. TTA-FedDG: Leveraging Test-Time Adaptation to Address Federated Domain Generalization[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2025, 39(18): 18658-18666.

### Questions
See above.

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
This paper presents HyperFedZero, a novel hypernetwork-based approach for zero-shot personalization in Federated Learning. The work addresses the critical but under-explored problem of generalizing to non-participating clients with in-domain distribution shifts. The methodology is technically sound and the experimental validation appears thorough.

### Strengths
1. The motivation is clearly declared:  the inability to handle non-participating clients with distribution shifts.
2. Extensive experiments across 7 datasets and 5 models are conducted.

### Weaknesses
1. The technical description lacks sufficient detail for reproduction.
2. The method lacks theoretical justification.
3. How does the method scale with:
a) Increasing number of clients?
b) Larger model architectures?
c) Higher-dimensional data?

### Questions
Please refer to weaknesses for details.

### Soundness
3

### Presentation
3

### Contribution
2

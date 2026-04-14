# A General Aggregation Federated Learning Intervention Algorithm based on $do$-Calculus

- Decision: Reject
- Scores: 5, 5, 3

## Abstract
This article explores federated long-tail learning (Fed-LT) tasks, which involve clients with private and heterogeneous data that exhibit a long-tail distribution. We propose two methods: (a) Client Re-weighted Prior Analyzer (CRePA), which balances the global model's performance on tail and non-tail categories and enhances performance on tail categories while maintaining it on non-tail categories. (b) Federated Long-Tail Causal Intervention Model (FedLT-CI) computes clients' causal effects on the global model's performance in the tail and enhances the interpretability of Fed-LT. CRePA achieves state-of-the-art performance, and FedLT-CI improves tail performance significantly without affecting non-tail performance. Extensive experiments indicate that CRePA achieved SOTA performance compared to other baselines on CIFAR-10-LT and CIFAR-100-LT. Applying the FedLT-CI to all baselines significantly improved tail performance without affecting non-tail performance.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes two methods, CRePA and FedLT-CI, to address the challenges posed by data heterogeneity and long-tail distributions in Federated Learning. CRePA improves the performance of the global model on tail categories by re-weighting client contributions, while FedLT-CI leverages causal inference to analyze each client's causal effect and optimizes model aggregation through intervention strategies. Experimental results show that CRePA and FedLT-CI enhance tail category performance while effectively reducing communication costs.

### Strengths
1.The overall quality of the writing is commendable.

2.The illustrations in the paper are highly clear and informative.

3.The proposed method is relatively innovative and is supporte1. In the introduction, the authors mention that some researchers' methods addressing Federated Learning heterogeneity and long-tail issues do not consider the impact of different clients on the aggregated model's performance on tail data from a causal perspective, which is "important". However, the paper does not clarify the importance of this perspective, and it appears to result in only a marginal improvement in the data.

### Weaknesses
1. In the introduction, the authors mention that some researchers' methods addressing Federated Learning heterogeneity and long-tail issues do not consider the impact of different clients on the aggregated model's performance on tail data from a causal perspective, which is "important". However, the paper does not clarify the importance of this perspective, and it appears to result in only a marginal improvement in the data.

2.The structure of the Introduction section in this paper is not well-organized. The authors begin by introducing the long-tail problem and data heterogeneity issues, followed immediately by an overview of the proposed CRePA and FedLT-CI methods. However, they subsequently discuss limitations in prior work, such as the failure of existing algorithms to address potential long-tail issues in FL, as well as concerns regarding communication costs and the causal perspective. Presenting these issues in previous work after introducing the proposed methods disrupts the logical flow. Clearly, it would be more coherent to first highlight the limitations of prior studies and then present the authors' methods.

Additionally, the authors claim to have "summarized" the contributions of the paper; however, the description of CRePA and FedLT-CI in the contributions section is even more detailed than in previous sections, introducing elements not mentioned earlier. For instance, the authors refer to an adaptive loss function proposed within CRePA, which was not mentioned previously. This makes the contributions section appear overly verbose, while the prior introduction is too brief.

3.The experimental setup section in this paper does not specify the metrics used, and accuracy is the only evaluation metric applied in the experiments. Consequently, the experimental results appear less convincing. Incorporating the AUC-ROC metric may provide further evidence of the model's performance.

### Questions
Q1: Is the causal effect important for addressing data heterogeneity and long-tail distribution?

Q2: Is it necessary to conduct further experiments to elaborate on the issue of communication costs?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript tackles the challenges of federated long-tail learning (Fed-LT), where clients have heterogeneous data that collectively exhibit a global long-tail distribution. The authors introduce two novel approaches: the Client Re-weighted Prior Analyzer (CRePA) and the Federated Long-Tail Causal Intervention Model (FedLTCI). CRePA improves tail performance while maintaining non-tail performance by learning client-specific weight distributions from gradient information. FedLTCI evaluates causal effects on the global model's tail performance, enhancing interpretability. Experiments on CIFAR-10-LT and CIFAR-100-LT show that CRePA achieves state-of-the-art results.

### Strengths
1. The issue of long-tail distribution in federated learning (Fed-LT) is significant and intriguing, highlighting the challenges faced in real-world applications.
2. The incorporation of causal inference in the FedLT-CI model is a novel approach in federated long-tailed learning, whichi enhances interpretability.
3. The methodologies employed demonstrate performance improvements in CIFAR-10/100.

### Weaknesses
**Motivation:**

1. The motivation behind utilizing aggregation to assist federated learning for long-tail distributions (Fed-LT) is not clearly articulated. In the Fed-LT context, each client typically possesses very few samples of tail classes, making it difficult to ensure that the aggregated model will achieve a more balanced performance across all classes.
2. The necessity of adopting a causal perspective in line 70 raises questions. In the context of Fed-LT, what are the implications of lacking interpretability? Understanding the consequences of this deficiency could strengthen the argument for the proposed approach.
3. The existing methods for Fed-LT present certain limitations that the proposed FedLT-CI aims to overcome. A clearer comparison of these limitations would enhance the reader's understanding of the contributions of this work.
4. The relationship between CREPA and FEDLT-CI is somewhat ambiguous. A more thorough explanation of how these two methodologies interact or complement each other would provide valuable insight into their combined effectiveness.

**Methods and Experiments:**

1. The citations related to data heterogeneity in federated learning are notably outdated. It is important to reference more recent studies in the Fed-LT to provide a comprehensive background and context.
2. The baseline models employed in the experiments seem to be somewhat dated, and the datasets utilized may lack diversity. A broader selection of baselines and more contemporary datasets would strengthen the experimental validation of the proposed methods.
3. The computational overhead associated with CREPA requires further investigation. Additionally, a discussion of the computational costs involved in implementing FEDLT-CI would be beneficial.
4. There is a concern regarding how to prevent clients with a significant representation of head information (e.g., clients c1, c19, c25, c31) but limited tail samples from being consistently excluded during aggregation. This could lead to a decline in the model's representational capacity for tail classes.

**Writing:**

1. In line 60, the transition marked by "therefore" lacks a clear rationale. A more explicit connection to the preceding content would enhance clarity.
2. The privacy issues referenced in line 67 are not clearly defined. Elaborating on the specific privacy concerns would provide a more comprehensive understanding.
3. The methodology for dataset partitioning needs clarification. Specifically, how do the long-tail and heterogeneous datasets generate?
4. In line 152, the statement that data is categorized into tail and non-tail classes seems inconsistent with the experimental design, which mentions "many middle few." This discrepancy should be addressed to avoid confusion.
5. The title "A General Aggregation Federated Learning Intervention Algorithm Based on do-Calculus" does not clearly indicate its relevance to the federated long-tail learning scenario, which may lead to confusion regarding the paper's specific focus.

### Questions
1. What prompted the exploration of aggregation as a means to assist federated long-tail learning (Fed-LT)?
2. Why is it necessary to adopt a causal perspective? In the context of FedLT, what are the consequences of lacking interpretability in Fed-LT scenarios?
3. What limitations do previous FedLT approaches have that necessitate the introduction of FedLT-CI?
4. What is the relationship between CREPA and FEDLT-CI?
5. How can we prevent clients with sufficient head information (e.g., c1, c19, c25, c31) but very few tail samples from being consistently excluded during aggregation? What impact could this have on the model's representational capability?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents two algorithms to address the challenges of data heterogeneity and long-tail distribution in Federated Learning (FL). The first method, Client Re-weighted Prior Analyzer (CRePA), balances the global model's performance on tail and non-tail categories by learning the prior distribution of weights for each client through gradient information. The second method, Federated Long-Tail Causal Intervention Model (FedLT-CI), computes the causal effects of clients on the global model's performance in the tail and enhances interpretability in FL. Extensive experiments on CIFAR10-LT and CIFAR-100-LT datasets demonstrate that CRePA outperforms other baselines.

### Strengths
1. Provides a robust theoretical foundation with detailed derivations for the causal intervention framework.
2. Conducts extensive experiments on CIFAR-10/100-LT across varying imbalance and heterogeneity settings.

### Weaknesses
1. The motivation of the paper lacks focus, as it seems to tackle data heterogeneity and the long-tail problem separately.
2. It's uncommon for this reviewer to assess a submission introducing two methods addressing distinct challenges. It gives the impression of combining two papers into one.
3. The connection between the two methods is unclear. CRePA appears unrelated to the do-calculus framework mentioned in the title.
4. The authors need to further explore CAUSAL EFFECT in the context of federated learning and compare with relevant baselines.
5. The experiments are limited to CIFAR datasets. Testing on more diverse datasets, especially real-world federated learning long-tail scenarios, would further validate the findings.

### Questions
see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

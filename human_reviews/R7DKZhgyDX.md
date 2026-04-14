# PerFIT: Personalized Federated Instruction Tuning via Neural Architecture Search

- Decision: Reject
- Scores: 8, 3, 3, 3, 6

## Abstract
Federated Instruction Tuning (FIT) has shown the ability to enable model instruction tuning among massive data owners without exposing privacy.  Yet, it still faces two key challenges, i.e., data and resource heterogeneity. Due to the varying data distribution and preferences among data owners, FIT cannot adapt to the personalized data of individual owners. Moreover, clients with superior computational abilities have to compromise to maintain the same fine-tuning architecture as the weaker clients. Such a constraint prevents the powerful clients from having more trainable parameters for better fine-tuning performances. To address these issues uniformly, we propose a novel Personalized Federated Instruction Tuning (PerFIT) framework based on architecture search. Specifically, PerFIT allows each client to search for a personalized architecture by expanding the trainable parameter space of the global model, pruning them, and obtaining personalized sparse patterns. We further propose personalized parameter-wise aggregation to facilitate flexible aggregation among clients with diverse sparse patterns. This procedure allows personalized instruction fine-tuning within the expanded parameter spaces, concurrently preserving the same number of trainable parameters as the vanilla state, thus introducing no extra resource burden. 
The evaluations with multiple LLMs on various instruction-following datasets demonstrate that our approach can achieve up to a 23% decrease in personalized perplexity compared to the state-of-the-art FIT methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents PerFIT, a framework that personalizes federated tuning of LLMs to tailor model architectures to clients' needs. It introduces a parameter-wise aggregation strategy to balance personalization and collaboration, ensuring effective model updates despite data and resource heterogeneity. PerFIT demonstrates improvements in perplexity and scalability across datasets, outperforming standard federated learning baselines. This contribution addresses key challenges in decentralized learning, such as model adaptation and communication efficiency, offering a scalable solution for diverse federated environments.

### Strengths
* The use of NAS helps design sparse and client-specific architectures. It ensures that even with limited client resources, models remain effective by focusing on personalized tuning. Models can achieve better convergence on client datasets without forcing a universal model structure.

* Instead of aggregating entire models, PerFIT uses parameter-wise strategies. This enables more efficient collaboration between clients while maintaining the benefits of local personalization, reducing the risk of model degradation across heterogeneous clients.

### Weaknesses
* The paper does not explore the scalability of PerFIT with models larger than 7B parameters or client populations beyond the tested settings. More experiments with larger-scale models and diverse client distributions could strengthen the claims, especially regarding the framework’s computational and time complexities. 

* A deeper analysis of communication costs and performance consistency across varying client heterogeneity could offer further insights into PerFIT’s applicability and robustness under real-world conditions.

### Questions
* How does PerFIT perform when client datasets are drastically imbalanced or highly diverse? Are there scenarios where parameter-wise aggregation fails to converge or leads to suboptimal results?

* How does PerFIT perform with larger models beyond 7B size, would the computational burden of PerFIT increase linearly? Are there any planned optimizations to prevent such overhead?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work primarily addresses the challenges of data distribution heterogeneity and client-side computational resource heterogeneity in federated instruction tuning. To tackle these issues, it proposes a neural architecture search-based approach to achieve personalized model parameters and structures. Data heterogeneity and resource heterogeneity are classic challenges in FL community, with substantial prior research dedicated to them. Revisiting these problems in the era of LLMs is meaningful. However, the work’s technical and theoretical contributions are not prominent, and there is also a lack of empirical analysis to support the stated research challenges. Additionally, the experimental evaluation has areas for improvement. Overall, I believe this paper is not suitable for acceptance, especially at a top-tier conference like ICLR.

### Strengths
1. This works is targeted to valuable issues in federated instruction tuning.
2. Figure 1 is well presented to make a clear introduction of the proposed approach.

### Weaknesses
1. The authors claimed that this work is motivated by "intrinsic connection between data heterogeneity and architecture heterogeneity", but did not present clear evidence for the intrinsic connection. If no intrinsic connection exists, this work actually solves two separated problems. Given the existence of corresponding work to address these issues, the novelty of this proposed work is diminished ([1] for resource heterogeneity, and [2] for benchmarking personalized approaches for data heterogeneity in federated LLM tuning).
2. I am curious whether this masking approach can truly address the issue of resource heterogeneity. Masking does not seem to reduce the computational load since current libraries lack satisfactory support for masked models in terms of computational efficiency. If the authors emphasize their method’s contribution to resource heterogeneity, supporting experimental results are needed.
3. A key goal of the proposed method is to address resource heterogeneity, claiming that previous FL fine-tuning work based on PEFT methods such as LoRA mainly involved homogeneous models. However, from Figure 1, it appears that only the LoRA components were subjected to NAS. Given that the parameters in LoRA only occupy a small portion of the entire LLM, I am curious about the extent to which this method actually contributes for solving heterogeneous computational resources on the client side.
4. In line 703, the authors claim that "The average perplexity in each round is reported. Please refer to Appendix C for details". However, I couldn’t find the corresponding information. Also, why was I directed from Appendix C to look for content in Appendix C?
5. Randomly assigning 200 data samples to each client represents a highly unrealistic scenario, where the data distribution is IID, and even the quantity of data is also IID (line 367). Experiments conducted under this scenario constitute the majority of the experimental evaluation, which somewhat undermines the persuasiveness of the method's effectiveness.
6. Since this work focuses on personalized FL, comparing only with the FIT method is insufficient. On one hand, more advanced personalized FL fine-tuning methods should be included for comparison, such as [3]. On the other hand, it is recommended to fine-tune the LLM obtained by FIT to adapt it as a personalized federated approach.
7. The authors demonstrate the convergence of their method. This type of analysis has already been extensively conducted in traditional FL studies. Considering that the theoretical modeling in this manuscript does not differ from traditional FL or masked-based FL, it is debatable whether dedicating substantial space to this well-established theoretical analysis is truly necessary. Moreover, whether LLMs genuinely satisfy the L-smoothness assumption remains a contentious issue, which makes the theoretical contribution of this paper less significant.


[1] Federated Fine-tuning of Large Language Models under Heterogeneous Tasks and Client Resources. NeurIPS 2024. 

[2] Federatedscope-LLM: A comprehensive package for fine-tuning large language models in federated learning. KDD 2024.

[3] FDLoRA: Personalized Federated Learning of Large Language Model via Dual LoRA Tuning. arXiv24.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes a NAS-based method to solve the data and resource heterogeneity faced by previous federated instruction methods which adopt a unified global model. To evaluate the effectiveness of the proposed approach, experiments are performed on four instruction datasets, with a native federated instruction tuning method as the baseline. These challenges are reasonable; however, the innovation of the method is limited, and the brought improvements are also minor.

### Strengths
1. This work is well-organized and well-presented, making it easy to follow.
2. Code is available. Although no documentation is provided to explain how to run the experiments, having the code is certainly better than having none at all.

### Weaknesses
1. Marginal improvements on performance. From Table 1, PerFIT exhibits a real small improvement on perplexity compared to FIT. Considering the wide range of values that perplexity can take, I am doubtful whether this slight improvement obtained by PerFIT actually contributes to enhancing the LLMs' performance. The authors could provide examples or analyses demonstrating how these small perplexity improvements translate to practical enhancements in LLM performance.
2. Lacks of novel contribution. Although the problem addressed in this work is meaningful, the proposed method does not show a significant distinction from traditional methods, i.e., it seems to merely change the application context from traditional FL for small models to LLMs fine-tuned with LoRA.
3. The first paragraph in the second section is entitled with "Federated Instruction Tuning of Large Language Models". However, the majority of this paragraph discusses matters unrelated to FL, making the inclusion of this paragraph perplexing.
4. This paper overclaims its contributions to the issue of resource heterogeneity. The method is based on LoRA, which typically accounts for only about 1% of the parameters of a full LLM. In this context, the gains from reducing the number of parameters through masking are minimal, regardless of whether it concerns computation, communication, or memory overhead. The authors should clearly quantify how much the resource heterogeneity could be enabled by the proposed approach.

### Questions
1. What is the essential difference between the proposed method and existing NAS-based FL methods?
2. Does adding a mask to LoRA adapters affect the consistency between the training objective and the FL objective?
3. From the benchmarking results in [1], the heterogeneity of data distribution seems to affects the fine-tuned results slightly. Does this affect the significance of addressing data distribution heterogeneity in personalized federated instruction tuning to some extent?

[1] FederatedScope-LLM: A Comprehensive Package for Fine-tuning Large Language Models in Federated Learning.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
The paper proposes a framework, Personalized Federated Instruction Tuning, which aims to enable personalized instruction tuning of large language models (LLMs) in federated settings. The approach incorporates Neural Architecture Search to allow each client to personalize their LoRA modules, thus addressing data and resource heterogeneity among clients. The framework includes personalized aggregation mechanisms to efficiently combine and redistribute updated parameters based on each client’s unique data and resource constraints.

### Strengths
- The paper addresses a meaningful problem of personalizing LoRA parameters for each client in a federated learning setup.
- Experimental results showcase PERFIT’s robustness across different LLMs, datasets, and client configurations
- By leveraging the pruning method to personalize and sparsely prune LoRA modules, the approach can minimize computational overhead and adapt to the computational capacities of different clients

### Weaknesses
- The description of the personalized aggregation module lacks clarity, particularly regarding the process of aggregating and redistributing mask and LoRA parameters across rounds. This complexity makes it challenging to fully grasp the module's function and purpose in the framework.
- The paper suffers from vague and inconsistent notation throughout, which makes it difficult for readers to follow the mathematical formulations and key concepts presented.
- The paper does not provide insights or analysis on adaptively setting the mask ratio for each client based on the data, which could be a significant parameter affecting performance based on individual client data distributions.
- Although the framework is positioned as utilizing NAS, the same base architecture is used across clients, with only varying degrees of unstructured pruning applied to LoRA modules, which may fall short of full architectural differentiation.
-  While a theorem is proposed, the paper does not provide detailed derivations, leaving gaps in the theoretical foundation and proof of the method’s performance. The insights behind the theorem should also be further explained. Why the $\kappa$ is negative? Please clearly explain the derivations of the theorem part.
- The evaluation relies solely on perplexity comparisons without examining time efficiency or computational costs, which are crucial for federated learning applications with resource constraints.
- The paper does not include baseline methods for comparison, which limits the ability to fully evaluate the effectiveness and rationale of the proposed approach. Including comparisons with simpler methods, such as fine-tuning the LoRA at each client as a personalization strategy, would provide valuable insight into its relative advantages and justify its complexity.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Instruction tuning has been shown to be crucial for large language models in generating responses aligned with human preferences. This paper explores a novel method of collaborative instruction tuning under data privacy constraints and proposes a personalized federated instruction tuning framework (PerFIT). To address the data and resource heterogeneity among clients and prevent resource-rich clients from being limited by the constraints of resource-poor clients, the authors introduce an architecture auto-search method, allowing each client to obtain a personalized instruction tuning architecture. Overall, this paper provides an optimally configured model architecture for clients with heterogeneous resources.

### Strengths
S1: The paper introduces a federated instruction tuning framework based on architecture auto-search, effectively addressing data and resource heterogeneity in federated learning.
S2: The paper’s structure is well-organized, and the logic is clear.
S3: Ample theoretical analysis is provided, supporting the effectiveness of the proposed method.

### Weaknesses
W1: The discussion of federated instruction tuning for LLMs in related work is insufficiently in-depth, as it only briefly mentions two LoRA-based FIT frameworks that address data heterogeneity.
W2: In Figure 1, there are discrepancies between the legend (②, ③, and ④) and the explanations in the text. For example, in the text, ② represents “Sparse Module Generation and Local Fine-tuning,” but it appears as “Sparse Module Generation” in the legend. The fine-tuning process should be indicated on the specific sparse modules in the figure, possibly by adding an icon (e.g., a flame) to represent fine-tuning.
W3: The paper claims this method as the first solution to address the issue of personalized federated instruction tuning; therefore, experiments only compare it with the global model. However, some existing methods already address data heterogeneity in federated instruction tuning. It is suggested that the paper include a comparative analysis with these methods regarding data heterogeneity.
W4: The paper does not provide open-source code.

### Questions
Q1: The notation for i and j in the personalized aggregation section is confusing. In m, i and j represent client IDs, while in A, B, and I, they denote positions. A clearer notation is recommended.
Q2: The third step in the framework is “personalized model aggregation.” Based on the description in Algorithm 3, this personalization is implemented in a grouped aggregation manner. It would be more precise to refer to this as “grouped model aggregation” as “personalized aggregation” is somewhat coarse.
Q3: Equation (6) lacks a label (line 265).
Q4: The last sentence on page 5 and the first sentence on page 6 lack continuity; please check for any missing information.

### Soundness
2

### Presentation
2

### Contribution
2

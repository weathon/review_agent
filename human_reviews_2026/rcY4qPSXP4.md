# CS-pFedTM: Communication-Efficient and Similarity-based Personalization with Tsetlin Machines

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Federated Learning has become a promising framework for preserving data privacy in collaborative training across decentralized data sources. However, the presence of data heterogeneity remains a significant challenge, impacting both the performance and efficiency of FL systems. To address this, we introduce CS-pFedTM (Communication-Efficient and Similarity-based Personalized Federated Learning with Tsetlin Machine), a method that addresses this challenge by jointly enforcing communication-aware resource allocation and heterogeneity-driven personalization. CS-pFedTM enforces communication budget feasibility through clause allocation and tailor personalization using clients' parameters similarity as a proxy for data heterogeneity. Experiments across multiple datasets show that CS-pFedTM consistently outperforms state-of-the-art personalized FL approaches, achieving at least $3.6\times$ lower upload cost, $5.58\times$ lower download cost, and $1.17\times$ higher runtime efficiency, while maintaining superior accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenges of data heterogeneity and communication efficiency in federated learning (FL). The authors propose CS-pFedTM, a personalized FL method based on Tsetlin Machines (TMs), which uses two independent models per client: a local TM for client-specific patterns and a global TM for shared knowledge. The core contribution is an adaptive clause allocation mechanism that dynamically balances local and global clauses based on client parameter similarity (as a proxy for data heterogeneity) and communication budgets. The method incorporates performance-based client selection and class-specific weight masking. Experiments on image and audio datasets demonstrate improved accuracy, reduced communication costs, and enhanced storage and runtime efficiency compared to state-of-the-art personalized FL baselines.

### Strengths
The use of Tsetlin Machines for FL personalization, with adaptive local-global clause allocation based on parameter similarity, is a unique contribution.  
The method significantly reduces upload/download costs, storage, and runtime memory while maintaining competitive accuracy  
The inverse relationship between parameter similarity (Jaccard index) and data heterogeneity (Wasserstein distance) is formally motivated and empirically validated.

### Weaknesses
Only FedTM is used as a TM-based baseline, leaving other potential TM variants or composites unexplored.  
Although results are averaged over three runs, no standard deviations or confidence intervals are provided.  
The individual contributions of components like weight masking, similarity-driven allocation, and performance-based selection are not isolated.  
The method relies on booleanized inputs, which may limit its use in domains where non-binary feature representations are critical.  
No discussion of scenarios where parameter similarity may fail to accurately reflect data heterogeneity, or how the method performs under extreme non-IID conditions.  
The method uses a reference round for similarity estimation, but dynamic changes in client data distributions are not addressed.  
While performance-based selection is used, its impact on fairness or long-term participation bias is not analyzed.

### Questions
Please respond to the Weaknesses.

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
3

### Summary
This paper addresses the data heterogeneity problem of the standard FL framework. The proposed method, CS-pFedTM, introduces Tsetlin Machines, a rule-based model based on finite-state automata and game theory, as an efficient alternative to DNNs, along with communication budget-aware resource allocation and heterogeneity-driven personalization method. Furthermore, the proposed method allocates clauses between local and global models based on client similarity computed from a reference round, and incorporates weight masking and performance-based client selection. Experiments under various models and datasets demonstrate that the proposed method outperforms SOTA personalized FL methods. While the application of TM to personalized FL is novel, the paper suffers from significant weaknesses in theoretical justification and experimental rigor that place it below the acceptance threshold.

### Strengths
This paper presents a novel integration of the Tsetlin Machine into a personalized federated learning framework to address data heterogeneity. In addition, complementary schemes are proposed to enhance personalization performance and communication efficiency.

### Weaknesses
(1) The proposed method largely depends on empirical extension of existing methods, lacking theoretical justification. To address this concern, please clarify and provide detailed explanations for the following:

(1-a) The paper claims that, to maintain fairness in client participation, only the top-performing clients’ states are uploaded and used in global aggregation (lines 295–298). However, this design may bias the model toward specific data distributions and exclude clients with more challenging datasets, potentially degrading generalization across the population. Moreover, the paper does not provide any theoretical justification or convergence analysis for this selective aggregation approach. Please clarify this concern.
Besides, please provide details of the client selection method: How are the top-performing clients selected? Is it Top-1 or Top-K? If it is Top-K, what is the default value of K, and how is it determined?

(1-b) The paper claims that the class-specific weights of TMs enable further personalization through weight masking: weights corresponding to classes not observed locally can be set to zero, allowing the model to quickly adapt to unseen classes (lines 201-203). However, the paper does not provide sufficient theoretical or empirical evidence to explain how this zero-masking mechanism contributes to fast adaptation. Please clarify the underlying rationale or mechanism — for example, whether zeroing out unobserved class weights reduces interference, facilitates gradient reallocation, or improves generalization.

(1-c) Due to the randomness of participating clients, the communication cost and similarity computed in the reference round may not reflect those in subsequent rounds. How can the authors ensure that the reference round is representative of the remaining rounds under random client participation?

(1-d) The explanation of how min_frac is computed (line 332) is provided in Algorithm 4 of the appendix. However, the paper does not offer any rationale or empirical justification for this computation scheme. Please provide a detailed description of how min_frac is determined and explain the reasoning behind this choice.

(2) The experimental claims do not fully support the contribution of the proposed method. To address this concern, please clarify and provide detailed explanations for the following:

(2-a) All results appear to be from single runs without confidence intervals. For rigorous evaluation, please provide the standard deviation over multiple random seeds.

(2-b) The 2-layer CNN model used in the experiments is not a commonly adopted architecture. At minimum, results on more standard models, such as a 4-layer CNN or MobileNet, are recommended as baselines for fair comparison.

(2-c) To assess the individual contribution of each component of the proposed method, please provide ablation studies where client selection, weight masking, and clause allocation are applied independently.

(2-d) The current presentation of Fig. 3 makes it difficult to visually compare algorithmic performance. It is recommended to present these results in a tabular format for easier comparison and clearer interpretation.

### Questions
Please refer to the Weakness section for detailed comments. In particular, I would appreciate clarification on the questions raised for each weakness. I will reconsider my evaluation after reviewing the authors’ rebuttal to these points.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel FL framework called CS-pFedTM. The method personalizes client models by adaptively allocating local and global clauses in a Tsetlin Machine based on inter-client parameter similarity, which serves as a proxy for distributional heterogeneity. The approach integrates class-specific weight masking and performance-based client selection to enhance scalability. Empirical studies on six benchmark datasets demonstrate the communication efficiency and overall performance.

### Strengths
This paper studies an interesting and important topic. The algorithm design is clear and novel, and the experimental results show that the proposed method significantly reduces communication overhead while maintaining accuracy.

### Weaknesses
* In contribution point 2, “TM parameters reflect overall system heterogeneity”, why is the system heterogeneity being reflected?

* Does the class-specific weight masking raise additional privacy issues or concerns?

* Corollary 1 seems more like an empirical hypothesis than a rigorous theorem in form. I think there are some issues: W $\to$ 0 then J $\to$ 1 only shows that if the distributions are similar, then the parameters are similar. But it does not mean that when the parameters are similar, the distributions are also similar. For example, when facing noisy and nonconvex models, the parameter similarity may still be high. I think only if the author wants to argue that low parameter similarity corresponds to higher distributional divergence would it make more sense. But why not just write that argument?

* Why the design is “but only the top-performing clients’ states (based on local performance) are uploaded and used in global aggregation”? In my view, this design may lose a lot of useful update information from other clients. Moreover, the design of similarity-driven personalization also leads to a trend where the global model is primarily driven by clients that share similar distributions and have high upload volumes, causing it to lose the long-tailed knowledge.

* No experiment isolates the contributions from performance-based client selection or weight masking (mentioned in Sec. 4.1 and Algorithm 1). Also, the ablation related to the masking parameter \tau is missing.

### Questions
See detailed questions in the weaknesses part.

### Soundness
2

### Presentation
3

### Contribution
3

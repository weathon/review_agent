# Training-free Counterfactual Explanation for Temporal Graph Model Inference

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 2

## Abstract
Temporal graph neural networks (TGNN) extend graph neural networks to dynamic networks and have demonstrated strong predictive power. However, interpreting TGNN remains far less explored than their static-graph counterparts. This paper introduces TEMporal Graph eXplainer (TemGX), a training-free,post-hoc framework that help users interpret and understand TGNN behavior by discovering temporal subgraphs and their evolution that are responsible for TGNN output of interests.We introduce a class of explainability measures that extends influence maximization in terms of structural influence and time decay to model temporal influence. We formulate the explanation task as a constrained optimization problem, and propose fast algorithms to discover explanations with guarantees on their temporal explainability. Our experimental study verifies the effectiveness and efficiency of TemGX for TGNN explanation, compared with state-of-the-art explainers. We also showcase how TemGX supports inference queries for dynamic network analysis.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes an efficient explanation method for temporal graph neural networks. They introduce several engineering techniques and approximation algorithms to find the set of important input information that explains the model's decision.

### Strengths
I think this is a solid paper. The problem is well analyzed and the reasoning and technicalities of the method make sense to me. I also think the experimental results are thorough and carefully designed.

### Weaknesses
I expect general readers will have a hard time digesting this paper since it involves a lot of technicalities (from TGNN, explanation methods for it and some background in approximation algorithms), I would prefer if the authors can present the work in more intuitive manners.

### Questions
Previously, the work "On the Limit of Explaining Black-box Temporal Graph Neural Networks" (https://arxiv.org/pdf/2212.00952) pointed out that there exists certain information of TGNN that can never be faithfully recovered if only perturbation are used to examine TGNN. I am wondering how this work addresses the problem pointed out in the above paper?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes TemGX, a training-free, model-agnostic, and counterfactual explanation framework for TGNNs that: (1) efficiently identifies temporal explanatory subgraphs responsible for a model prediction. (2) propose a class of temporal explainability measures to verify these explanations. Experimental results show that TemGX outperforms other TGNN explainer methods in AUFSC, Fidelity and running time.

### Strengths
- The paper is well-written, and the case analyses provide good insights into the real-world applicability of the proposed framework.
- Experimental results show that TemGX outperforms other TGNN explainers on AUFSC, fidelity, as well as efficiency. 
- The paper provides formal proofs that TemGX is a (1-1/e)-approximation, and it is in PTIME to verify the counterfactual properties for temporal subgraphs, which are meaningful contributions.
- The paper provides an analysis of the impact of different components (ICM, TRD, and temporal decay) of the temporal explainability measure on the fidelity score and shows that each component has a meaningful impact.

### Weaknesses
- It seems like the temporal explainability measure is a composition of established concepts, i.e. temporal effective resistance distance was previously proposed in (Zhu et al., 2024; Black et al., 2023), temporal decay was previously proposed in (Mei & Eisner, 2017), and temporal impact is an extension of the Independent Cascade Model. Can the authors clarify the novelty of this formulation?
- There is a lack of scalability analysis in the paper. It is unclear how the algorithm will scale to large temporal graphs. Even if the current algorithm is not scalable to large temporal graphs, the authors should still provide a theoretical or empirical scalability discussion.
- It seems like the algorithm is sensitive to the choice of the temporal decay rate λ. Does that mean we have to re-run hyperparameter sweeping for λ on each new dataset or task?

### Questions
- Please see Weaknesses section. Below are some extra questions:
- Can the author provide insights on how window size can impact TemGX performance?
- Temporal decay also seems to provide limited performance improvement compared to ICM and TRD. Can the authors provide insights on why this is the case.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a temporal graph explainer that identifies temporal subgraphs and their solutions based on counterfactual reasoning without any additional training over the base model. The explanation task is framed as a constrained optimization problem over temporal edges and nodes. The detailed experiments show the superiority of the proposed model.

### Strengths
- Clear, well-written introduction that makes the problem easy to understand.
- Extensive experiments across multiple datasets with strong baseline comparisons.
- Tackles a difficult problem with a careful, principled formulation.
- Supported by theoretical analysis.
- Open-source code is provided and well documented.

### Weaknesses
- The role and sources of randomness in the method are underexplained.
- The number of experimental trials (k = 5) is small for robust statistical conclusions.

### Questions
- You define temporal counterfactual edges, and then aggregate corresponding nodes/edges into a temporal subgraph as the explanation. Under what conditions does this subgraph remain a counterfactual? Please clarify your assumptions.
- Given the small number of trials, can you include mean ± standard deviation (and optionally statistical confidence) for all metrics? 
- Can you clarify where does stochasticity enter the pipeline?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces TemGX, a training-free, model-agnostic framework for explaining predictions from Temporal Graph Neural Networks (TGNNs). The proposed method discovers instance-level temporal subgraphs whose removal changes model predictions, aligning with counterfactual reasoning. TemGX quantifies temporal influence using information cascading and resistance distance metrics, and provides algorithms for efficient explanation generation with guarantees.

### Strengths
1 The paper introduces a training-free and queryable framework for generating counterfactual explanations of temporal graph models

2 The proposed method is straightforward and computationally efficient, and it could be adapted for different temporal GNN architectures and tasks.

3 The framework combines temporal influence modeling, resistance distance, and time-decay mechanisms in a coherent formulation supported by an approximation guarantee.

### Weaknesses
1 Baseline implementation details are insufficient. It is not mentioned which CoDy variant is used, nor its hyperparameters. Hyperparameters for the other baselines are also not mentioned. Code for the baseline implementations are not provided. 

2 Evaluation metrics are defined in an unconventional and problematic way. 
Higher sparsity (i.e., larger explanatory subgraphs) is implied to be better (explicitly stated in Appendix E), but this is confusing. Shouldn’t it be that smaller sparsity indicate more concise explanations, and therefore preferred? Instead, the current design seems to reward larger subgraphs, which naturally include more of the L-hop neighborhood and hence boost fidelity and AUFSC scores.
This effect is clearly visible in Figure 4, where the fidelity score of TemGX increases when the sparsity score increases. However, the fidelity score of CoDy basically stopped growing when sparsity reaches around 0.2. Doesn’t this mean that CoDy is superior, since it is able to identify the most decisive but also concise and stable counterfactual explanation?
Moreover, when sparsity equals to 1, it means the entire L-hop neighborhood is used for the explanation, which raises doubts about the usefulness of such explanations. If all available context is included, the explanation is questionable.
In addition, there appears to be inconsistency between Table 1 and Figure 4. While fidelity and AUFSC scores are reported in Table 1, the corresponding sparsity levels are not shown, making it difficult to interpret the trade-off. I would assume that the sparsity scores are fixed for all explanation methods, but this is never clarified. Based on Figure 4 and Table 1, it appears that in some cases, CoDy and TemGX are compared at different sparsity levels. For example, in the UCIM + TGN case, the fidelity scores for TemGX and CoDy are 0.468 and 0.394, respectively, as reported in Table 1. However, Figure 4 shows that TemGX achieves such fidelity score at a sparsity of around 0.4, while CoDy reaches its reported fidelity at a sparsity of around 0.2. This suggests that TemGX’s explanatory subgraph is roughly twice the size of CoDy’s in this setting. Without controlling for sparsity, it becomes unclear whether the fidelity improvement of TemGX truly reflects better explanation quality or simply results from including more of the neighborhood context.
In general, the evaluation is not convincing to me, and I believe this is the biggest weakness of the paper. If this issue can be sufficiently addressed, I would consider to increase the score.

3 Counterfactual analysis is under-specified. No results are provided on the flip-rate (how often the prediction actually changes, or how often counterfactual explanations can be found). Without this, the counterfactual nature of the explanations remains unproven.

4 Classification datasets (Multihost and Elliptic++) are used only in qualitative case studies and are excluded from quantitative results. The paper mentioned these 2 datasets are used for classification, but does not mention the model nor the quantitative explanation results, leaving this part of the evaluation incomplete.

5 Case studies: The case studies are poorly analyzed and not compared with the baselines, and their connection to the claimed strength of TemGX is superficial at most.

6 Poor writing. In general, the writing of the paper should be improved. The example in the introduction is difficult to follow. Figure 1, 2, and 5 all contain many subgraphs that are very small, and not sufficiently described, this is difficult for the readers to follow. The conclusion is overly brief. 

7 Missing citation and baseline. It seems that this paper is focusing on discrete time dynamic graphs, but CoDy is designed for continuous time dynamic graphs by its nature. The following paper presents a counterfactual explanation method for DTDGs, it should be included as a baseline, and their evaluation metrics validity and GED should also be taken into consideration. 

Prenkaj, B., Villaizán-Vallelado, M., Leemann, T., & Kasneci, G. (2024, August). Unifying Evolution, Explanation, and Discernment: A Generative Approach for Dynamic Graph Counterfactuals. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (pp. 2420-2431).

### Questions
1 What exact implementation of CoDy and other baselines were used, what are the hyperparameters?

2 Why were Multihost and Elliptic++ not quantitatively evaluated like the others?

3 How does TemGX behave when no counterfactual explanations exist for a prediction?

4 Please address the weaknesses mentioned above.

### Soundness
2

### Presentation
2

### Contribution
2

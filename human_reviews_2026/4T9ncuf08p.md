# Dataset Regeneration for Cross Domain Recommendation

- Decision: Reject
- Scores: 8, 4, 6

## Abstract
Cross-domain recommendation (CDR) has emerged as an effective strategy to mitigate data sparsity and cold-start challenges by transferring knowledge from a source domain to a target domain. Despite recent progress, two key issues remain: (i) Sparse overlap. In real-world datasets such as Amazon, the proportion of users active in both domains is extremely low, significantly limiting the effectiveness of many state-of-the-art CDR approaches.  (ii) Negative transfer. Existing methods primarily address this problem at the model level, often assuming that logged interactions are unbiased and noise-free. In practice, however, recommender data contain numerous spurious correlations, and this issue is exacerbated in CDR due to domain heterogeneity.
To address these challenges, we propose a dataset regeneration framework. First, we leverage a prediction model to generate a pool of high-confidence candidate interactions to link non-overlapping target-domain users and source-domain items. Second, inspired by causal inference, we introduce a filtering process designed to prune spurious interactions. This process identifies and removes not only noisy edges created during generation but also those from the original dataset, retaining only the interactions that have a positive causal effect on the target-domain performance. Through these two processes, we can regenerate a source-domain dataset that exhibits a tighter coupling and a more explicit causal connection with the target domain.
By integrating our method with three representative recommendation backbones—LightGCN, BiTGCF, and CUT—we show that it significantly boosts their predictive accuracy on the target domain, achieving substantial gains of up to 23.81\% in Recall@10 and 22.22\% in NDCG@10.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a data-centric framework, Generate-and-Filter (Gen/Del), for
cross-domain recommendation (CDR). Instead of focusing on model-level transfer, the
authors address data sparsity and negative transfer by regenerating a causal and
denoised source-domain dataset. The framework consists of two stages:
(1) Generation phase: A self-supervised model generates synthetic source-domain
interactions for users who exist only in the target domain, using masked-edge
reconstruction and BPR loss.
(2) Filtering phase: A counterfactual inference module assigns causal importance
weights to each generated or existing edge and filters out non-causal or spurious ones.
The resulting regenerated dataset can be plugged into any backbone recommender
(e.g., LightGCN, CUT, BiTGCF). Experiments on Douban and Amazon datasets show
consistent improvements across multiple backbones, with gains up to 23.8% in
Recall@10.

### Strengths
•	Originality: Presents a fresh, data-centric perspective on CDR, shifting focus from model-level transfer to dataset regeneration. The integration of causal counterfactual filtering with GNN-based representation learning is particularly innovative.
	•	Quality: Methodology is sound and well-formulated, with strong empirical results across multiple datasets and backbone models. Ablation studies effectively demonstrate the framework’s ability to mitigate negative transfer.
	•	Clarity: The paper is clearly written and well-structured, with intuitive explanations and informative figures.
	•	Significance: The framework is model-agnostic and has broad applicability, offering a principled foundation for future research on causal data manipulation and transfer learning.

Overall, the work is conceptually original, empirically convincing, and highly relevant to data-centric and causal learning in recommender systems.

### Weaknesses
(1) Limited Analysis of Computational Cost and Scalability
While the proposed Generate-and-Filter framework is conceptually appealing, the paper lacks a systematic evaluation of its computational overhead. The counterfactual filtering stage requires training an additional GNN and repeatedly assessing target-domain performance, which could be computationally intensive for large-scale datasets. However, the paper provides no quantitative analysis of runtime, memory consumption, or scaling behavior with respect to dataset size, leaving the practicality of the approach for industrial-scale recommender systems uncertain.

(2) Incomplete Symbol Definitions in the Counterfactual Interaction Filtering Section
Several key symbols in Section 2.3—such as F_t^s, y_i, E_t^s, and the mapping l(E_t)—are introduced without explicit definitions or consistent explanations. This lack of clarity makes the mathematical formulation difficult to follow and reproduce. A concise summary table of notations or explicit variable definitions would greatly enhance readability and reproducibility.

(3) Lack of Qualitative Analysis and Interpretability of Filtering Results
Although the paper presents quantitative improvements in metrics such as Recall@10 and NDCG@10, it lacks qualitative analysis of the filtering process. There are no examples or visualizations illustrating which user–item edges are pruned or retained by the counterfactual filtering stage. Without such interpretability analysis, it is difficult to understand what types of interactions the model identifies as causal versus spurious.

(4) Unclear Contribution of the Generation Phase
Ablation results suggest that most of the performance gains arise from the counterfactual filtering module rather than the data generation phase. However, the paper does not analyze the characteristics or quality of the generated interactions—such as their distribution, overlap with observed data, or effect on coverage. Consequently, the empirical contribution and necessity of the generation component remain ambiguous.

### Questions
(1) Address scalability and efficiency concerns
Providing details on the model’s runtime, computational cost, and resource usage
during experiments would help readers better understand the practical feasibility and
efficiency of the proposed framework.
(2) Deeper analysis of generation and filtering behavior
Analyzing how the generation phase adds synthetic edges and how the
counterfactual filtering module removes or retains interactions in practice would help
readers better understand the model’s decision behavior and its contribution to
performance improvements.

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
3

### Summary
This paper focuses on the cross-domain recommendation (CDR) task and addresses key challenges, including sparse user overlap across domains and negative transfer caused by spurious correlations in heterogeneous data. To tackle these issues, the authors propose a dataset regeneration framework that (1) generates high-confidence candidate interactions to link non-overlapping users and items, and (2) applies a causal-inference-inspired filtering process to remove spurious interactions from both the generated and original data. This approach enhances the causal connection between source and target domains. When integrated with recommendation models such as LightGCN, BiTGCF, and CUT, it substantially improves target-domain performance, achieving up to 23.81% gain in Recall@10 and 22.22% in NDCG@10.

### Strengths
1. This paper focuses on the cross-domain recommendation (CDR) task and addresses two major challenges: sparse user overlap across domains and negative transfer caused by spurious correlations in heterogeneous data.
2. To tackle these challenges, the authors propose a dataset regeneration framework. This approach strengthens the causal connection between the source and target domains.
3. The proposed framework, when integrated with recommendation models such as LightGCN, BiTGCF, and CUT, substantially improves target-domain performance.

### Weaknesses
1. The core argument of this paper is that prior work primarily addresses sparse overlap and negative transfer at the model level, whereas this work tackles these challenges from a data-centric perspective. In fact, in the cross-domain recommendation (CDR) field, several studies have already explored data-centric solutions, such as [1][2][3]. The authors also provide a comparative analysis between their approach and these existing data-centric methods.

[1]https://arxiv.org/pdf/2405.20710
[2]https://arxiv.org/abs/2307.13910
[3]https://dl.acm.org/doi/10.1145/3626772.3657902

2. There is an inconsistency between the paper title in the main text and the title on OpenReview. The authors should ensure that the titles are consistent before submission.

3. The proposed framework is divided into two stages: generation followed by filtering.
- For the generation stage, the authors employ self-supervised pretraining, which is a common practice in graph learning, and therefore this stage lacks significant novelty.
- For the filtering stage, the authors adopt counterfactual interaction filtering. It would be helpful to clarify the motivation for using this technique compared with existing filter-based methods. Are there unique challenges that the counterfactual approach specifically addresses?

### Questions
see weaknesses

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
This paper proposes a dataset enhancement strategy for cross-domain recommendation models. It aims to address the sparse cross-domain user overlap and noisy cross-domain signal (negative transfer) issues. The proposed strategy takes two stages. The first stage generates more user-item connections to address the sparse cross-domain user overlap issue, by learning a model that reconstructs edges in the source domain user-item graph. The second stage learns to identify spurious edges in the source domain user-item graph which should be removed to mitigate the negative transfer issue. Experimental results on two commonly used datasets, Amazon and Douban, showed the effectiveness of the proposed strategy.

### Strengths
S1. The paper is motivated well with a detailed example to illustrate issues of existing cross-domain recommendation solutions. 

S2. The proposed technique works on the dataset level and is orthogonal to cross-domain recommendation models, which has the potential to be applied to and strengthen different cross-domain recommendation models. 

S3. The proposed technique is shown to be effective on commonly used benchmark datasets.

S4. Source code is available.

### Weaknesses
W1. Technical details:

- The synthetic edge set contains edges between every non-overlapping user and their top-$k$ relevant items in the source set. Even the top-$k$ items might not be very relevant for some of the users, and hence there may be false positives. Using a fix $k$ for all users might not be the most effective. How about using a score threshold to filter the items instead (or a combination of both)? Also, how is the value of $k$ chosen in the experiments, and how does its value impact overall accuracy? 

- The NP-hardness of Problem $\overline{P}$ needs a proof. 

- How are the node embeddings in $\mathcal{F}_\theta^T$ initialized?

W2. Experiments:

- The performance gains obtained by using the proposed Gen/Del dataset preparation strategy is quite small as shown in Table 1 (noting the statistical significance test results). The second-best results in the two N columns of the Douban datasets didn't seem to be labeled correctly. 

- It would be interesting to see model running time results, model effectiveness results as $K$ (as in Recall/NDCG@$K$) varies, and model effectiveness results as the number of cross-domain overlapping users varies. 

W3. Presentation: 

- The preliminaries section should be moved to the main text to set up the context for the methodology section. Without it, the methodology section is difficult to follow.  

- Even with the preliminaries section, the paper needs a notation table to explain what the many symbols mean in the paper. 

- The final sentence in Appendix A, "The next section details the optimization techniques used to implement this filtering, integrating the pre-trained prediction model with edge weight adjustments to achieve the desired causal pruning.", seems to be disconnected from the subsequent section. 

- Typo: "”science fiction”" => "``science fiction”"; "in the Appendix B" => "in Appendix B"; "The single-domain baselines, trained exclusively on the target dataset" => "The single-domain baselines are trained exclusively on the target dataset"

### Questions
Please refer to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

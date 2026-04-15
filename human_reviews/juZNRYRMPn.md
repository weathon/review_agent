# Truth-Guided Negative Sampling in Self-supervised Graph Representation Learning

- Decision: Reject
- Scores: 6, 5, 3, 5

## Abstract
Negative sampling is an important yet challenging component in self-supervised graph representation learning, particularly for recommendation systems where user-item interactions are modeled as bipartite graphs. Existing methods often rely on heuristics or human-specified principles to design negative sampling distributions. This potentially overlooks the usage of an underlying ``true'' negative distribution, which we might be able to access as an oracle despite not knowing its exact form. 
In this work, we shift the focus from manually designing negative sampling distributions to a method that approximates and leverages the underlying true distribution.  We expand this idea in the analysis of two scenarios: (1) when the observed graph is an unbiased sample from the true distribution, and (2) when the observed graph is biased with partially observable positive edges. The analysis result is the derivation of a sampling strategy as the numerical approximation of a well-established learning objective. Our theoretical findings are also empirically validated, and our new sampling methods achieve state-of-the-art performance on real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a negative sampling approach from the ground up, offering a systematic analysis in contrast to other negative sampling studies. The paper also compares representative negative sampling methods and demonstrates significant improvements over previous approaches. However, there is room for improvement in the clearer definition of mathematical symbols (with some typos), the logical derivation of the sampling optimization objective, and the explanation of the sampling algorithm.

### Strengths
1. In contrast to other negative sampling studies, this paper reconsiders the formulation of negative sampling from the ground up, providing a systematic analysis. It begins with an ideal, unbiased estimator and then proposes a feasible optimization objective under biased conditions to design an effective sampling algorithm.
2. The paper compares representative negative sampling methods and shows a significant improvement in experimental results over previous approaches.

### Weaknesses
1. The paper contains numerous symbols, but some are either incorrect or poorly explained, increasing the reader's cognitive load. Details are elaborated in Question 1.
2. The implementation process of the sampling should be explained more clearly. While the current description aligns with the flow of the paper, it imposes a certain cognitive load on readers trying to understand the sampling algorithm. For instance, in Theorem 1, it is unclear how the class prior $\pi$  is defined, tuned, or calculated to advance the execution of the sampling algorithm.

### Questions
1. Typos and questions regarding symbols in the paper:    
(1) In line 3 of page 2, is "$p^−= P(e∣y = 1)$" correct?     
(2) In line 7 of section 3.2, "$\hat{p}^- = P(e∣s = 1)$", do you mean “$\hat{p}^+$” as the observable positive distribution?    
(3) In equation 5, which is the function g, is it $g^+$ or $g^-$?    
(4) What is the definition of $Z$ in equation 7? There seems to be an explanation for this symbol.    
2.  What is the relationship between $R_1(f)$ and $R_2(f)$? What is the indication of Proposition 2?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper studies the problem that how to approximate and leverage the true distribution for better model performance. To address this issue, they derived optimal sampling strategies for both unbiased and biased observed graphs, connecting these strategies to maximum likelihood estimation and incorporating graph topology. The experimental results are conducted to demonstrate the efficiency of the proposed method.

### Strengths
1. The research problem is relevant.
2. The authors present a learning-based method for approximating the propensity function.
3. The authors conducted a comprehensive theoretical analysis.

### Weaknesses
W1. The authors need to thoroughly review their paper, as there are some errors that require revision. (See D1)

W2. The authors need to strengthen the experimental section of the paper. (See D2-D6)

### Questions
D1. In the Introduction section, the formula for deriving the negative sample distribution using Bayes' rule should be presented asp^-=P(e|y=0)∝(1-P(y=1|e))P(e) .

D2. The authors state that the observed distribution in the LastFM is the same as the true distribution, while this is not the case for MovieLens and LastFM. However, it is unclear how the authors determined this. Further clarification is needed to explain the reasoning or evidence behind this assertion.

D3. Lacking the necessary baselines from the past two years, such as [1], [2],[3]

[1] Huang T, Dong Y, Ding M, et al. Mixgcf: An improved training method for graph neural network-based recommender systems[C]//Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021: 665-674.

[2] Shi W, Chen J, Feng F, et al. On the theories behind hard negative sampling for recommendation[C]//Proceedings of the ACM Web Conference 2023. 2023: 812-822. 

[3] Lai R, Chen R, Han Q, et al. Adaptive hardness negative sampling for collaborative filtering[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2024, 38(8): 8645-8652.

D4. The authors should include additional ablation experiments to investigate the impact of different features, such as node degree features and node embeddings, on model performance when estimating the propensity function.

D5. The convergence speed is a crucial aspect of negative sampling methods. The authors should provide a comparison of the convergence rates between their method and the baselines.

D6. The authors should provide a time complexity analysis of their proposed method.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This work studies on unbiased sampling for graph representation learning and proposes a new method that leverages distribution transformation and propensity scores. 

However, this work has some significant limitations in terms of novelty, presentation, I give a negative point of this work.

### Strengths
1. This work studies on an important problem.

2. Experiments on real-world datasets have been conducted to verify the efficacy of the proposed method.

### Weaknesses
1. My major concern lies on the novelty  Leveraging distribution transformation and propensity scores to overcome sampling bias is not a new concept and has been extensively studied in recent literature. Specifically, the relationship between $p^-$ and $p^+$ has been adopted by DCL [a5], and the same formula (line 1, page 4, in [a5]) is used to address sampling bias.  Furthermore, propensity scores have been widely used to address selection bias in the Recommendation Systems (RS) field [a6]. More critically, this work fails to review and explicitly cite these previous studies.

2. Another concern pertains to the clarity of the paper. Many important concepts are introduced without sufficient explanation, making the paper difficult to follow. Additionally, there are inconsistencies in definitions. For instance:

a) Eq.(4) is presented without much explanation. The motivation for introducing this formula and the reason for minimizing this objective remain unclear. The demonstration of unbiased estimation when using the distribution transformation of $p^-$ seems straightforward and does not require the introduction of these new concepts.

b) In section 3.2, $p^+$ is defined as $P(e|y=1)$, while in eq.(3), it is written as $p^+(.|u)$.  The interpretation of this notation is unclear. Meanwhile, eq.(7) introduces yet another notation, $p^+(u,v)$.

3. The experiments also have significant limitations:

a) Although this work claims to study samplers for graph representation learning, the experiments focus solely on recommendations. Notably, the selected baseline, MF, is not a graph-based method. Therefore, I suggest the authors conduct additional experiments on other graph learning scenarios and datasets.

b) The datasets used are relatively small. Larger datasets, such as Amazon, should be employed.

c) The baselines are outdated. Except for SENSEI, all the baselines predate 2020. More state-of-the-art sampling strategies could be employed, e.g., [a1][a2][a3][a4].

[a1]  IJCAI’24: Supervised contrastive learning with hard negative samples

[a2] WWW’23: Fairly adaptive negative sampling for recommendations

[a3] WWW’23: On the theories behind hard negative sampling for recommendation

[a4] CIKM’23: Batch-Mix Negative Sampling for Learning Recommendation Retrievers

[a5] NIPS’20: debiased contrastive learning

[a6] ICML’16: Recommendations as treatments: Debiasing learning and evaluation

### Questions
Please refer to weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper studies the problem of negative sampling. It proposes to approximating and utilizing the true distributions in sampling methods for graph representation learning. The authors theoretically derive and analyze the forms that the optimal sampling strategy should follow, under the assumption that the positive edges are sampled unbiasedly from the true distribution, or that the positive edges are only partially observable. Experiments are conducted on recommendations.

### Strengths
1.This paper introduce a novel approach to negative sampling by proposing to approximate the true negative distribution This method provides a more principled approach to understanding and executing negative sampling.
2: This paper outlines a strategy to adjust the empirical risk estimator to account for bias in recommendation systems, enhancing the model’s applicability and robustness.

### Weaknesses
1: The format does not follow the formatting instructions. There is no line numbers, the captions are putted below table, the title in pdf does not align with that in the system. A lot of typos exist, e.g., in page 8, “respective”, and in table 1, why the results of DNS are bolded? The presentation makes the paper a little bit hard to follow.

2: It is not clear why experiments are connected on recommendation. The most recent baseline in this paper is SENSI [1], it conducts experiments on normal node classification tasks, using datasets like Cora, CiteSeer and PubMed.

3: The experiments lack an analysis of time and space complexity. Excessive time consumption may render this method impractical for use in real-world recommendation systems. No time complexity or runtime comparison. No pseudo-code and no data statistics. There are too few experimental metrics. The dataset for the ablation study is insufficient.

[1] Reconciling competing sampling strategies of network embedding

### Questions
1: Can the author investigate in detail why the proposed method can improve the performance, for example from the perspective of false negative rate or hard negative samples? Currently we only have one metric, Recall@20, and no quantitative analysis.  Even if there is no quantitative analysis, it is better to have some other metrics. As far as I know, it is not common to use only recall@k and fix k.

2. Can negative samples represent the entire non-interaction space?

### Soundness
3

### Presentation
2

### Contribution
3

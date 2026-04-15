# Unified Interpretation of Smoothing Methods for Negative Sampling Loss Functions in Knowledge Graph Embedding

- Decision: Reject
- Scores: 6, 3, 5, 6

## Abstract
Knowledge Graphs (KGs) are fundamental resources in knowledge-intensive tasks in NLP. Due to the limitation of manually creating KGs, KG Completion (KGC) has an important role in automatically completing KGs by scoring their links with KG Embedding (KGE). To handle many entities in training, KGE relies on Negative Sampling (NS) loss that can reduce the computational cost by sampling. Since the appearance frequencies for each link are at most one in KGs, sparsity is an essential and inevitable problem. The NS loss is no exception. As a solution, the NS loss in KGE relies on smoothing methods like Self-Adversarial Negative Sampling (SANS) and subsampling. However, it is uncertain what kind of smoothing method is suitable for this purpose due to the lack of theoretical understanding. This paper provides theoretical interpretations of the smoothing methods for the NS loss in KGE and induces a new NS loss, Triplet-based SANS (T-SANS), that can cover the characteristics of the conventional smoothing methods. Experimental results on FB15k-237, WN18RR, and YAGO3-10 datasets showed the soundness of our interpretation and performance improvement by our T-SANS.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Knowledge Graphs (KGs) are vital in NLP, but creating them manually has limitations, leading to KG Completion (KGC) using KG Embedding (KGE) and Negative Sampling (NS) to handle many entities while reducing computational costs. The challenge of sparsity due to low link appearance frequencies in KGs is addressed through various smoothing methods like Self-Adversarial Negative Sampling (SANS), with this paper offering theoretical insights and introducing Triplet-based SANS (T-SANS), showing improved performance on multiple datasets.

### Strengths
The author illustrated clear motivation of this study and articulate the problem formulation mathematical-clearly. 

The authors provided sufficient information about the background and previous studies are well summarized.

I like the plots in the introduction, better demonstrating the existing challenges. 

The authors provided thorough experimentation to demonstrate the improved performance of the T-SANS, with quantitative comparison.

### Weaknesses
There are too many uncertainties in the contribution. The authors presented 
•  We theoretically show that T-SANS with subsampling can potentially cover the conventional usages of SANS and subsampling.
• We empirically verify that T-SANS improves KGC performance on sparse KGs in terms of
MRR.
• We empirically verify that T-SANS with subsampling can cover the conventional usages of
SANS and subsampling in terms of MRR.

Without confirmative mathematical approval, it's less convincing to argue that the conventional usages of SANS and subsampling can be covered by T-SANS. And the empirical study results also weaken this argument. 

Thus, I think the contribution is over claimed. 

I believe the negative sampling performance can vary significantly based on the distribution of KG in real-world data. It's worthy to mention that based on what kind of distribution of KG features (e.g., connectives) the proposed method can achieve compelling performance.

### Questions
Is the focus of this study to improve interpretation or model performance of KGE?

What is the distribution of the entities and edges of KG in the dataset look like? Does distribution of network feature influence the performance of the proposed NS algorithm?

What are the exact performance is in the experimentation. You only present the figures. Can you include table?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper provides a unified interpretation of smoothing methods, SANS and subsampling, for negative sampling loss function in KGE. Authors emphasize the importance of smoothing both p(x,y), p(y|x), and p(x) in the loss function to deal with the data sparsity of KG. Based on the analysis of SANS and subsampling negative sampling loss function, authors propose a new negative sampling function T-SANS, which integrate both subsampling and SANS in the loss function. Experiments show that with T-SANS as the negative sampling method, the KGE models generally performs better than SANS or existing subsampling negative sampling methods, especially on the extreme unbalanced and sparse KGs.

### Strengths
1. The topic is interesting and worth to investigate since negative sampling methods significantly affects the KGE performance for KGC tasks.
2. The paper tried to find a uniform loss function representation for SANS and subsampling negative sampling methods, which is good. 
3. T-SANS performs better than SANS and subsampling methods supports the importance of smoothing both p(x,y), p(y|x), and p(x) in the loss function.

### Weaknesses
1. My main concern of the paper is limited novelty contribution. T-SANS adds the subsampling based on SANS, referring to $p_{\theta}(x;\gamma)$ in Equation (12), which is the key difference between T-SANS and SANS. While as mentioned by the author in the footnote, Sun et al. (2019); Zhang et al. (2020b) use subsampling in their released implementation without referring to it in their paper. Thus, if I understood correctly,  I would like to say the actual implementation of methods of Sun et al. (2019); Zhang et al. (2020b) is very similar to T-SANS. Thus the novelty of this paper is limited. 
2. The work is motivated by that conventional works use SANS and subsampling with no theoretical background, and authors believed there is room for further performance improvement. It is unclear the why the lack of the theoretical background lead to potential performance improvement.
3. Some parts of the paper is not clearly explained or inaccurate and need further improvement, such as
* the $^{-\alpha}$ in Equation (4) is unexplained 
* statement in page 5 that "using Eq. (11) causes an imbalanced loss between the first and second terms since the sum of pθ (x, yi ) on ν number of negative samples is not always 1" is not accurate, since in the implementation of the model, there usually is a softmax function among over all the negative samples for a positive triple, which will make the sum of $p_{\theta} (x, y_i )$ to ν number of negative samples to 1. 
* the caption of Figure 4 is the same as Figure 3.

### Questions
1. What is the key/significant difference between T-SANS and the actually implementation of Sun et al. (2019); Zhang et al. (2020b) methods, i.e. SANS with subsampling? 
2. Should the caption of  Figure 4 to be updated?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper delves into the significance of Knowledge Graphs (KGs) in Natural Language Processing (NLP) tasks. The primary focus is on Knowledge Graph Completion (KGC), which aims to automatically complete KGs by scoring their links using Knowledge Graph Embedding (KGE). The paper discusses the challenges posed by the sparsity of KGs and the role of Negative Sampling (NS) loss in addressing these challenges. The paper introduces smoothing methods like Self-Adversarial Negative Sampling (SANS) and subsampling to tackle the sparsity issue. The main contribution is a theoretical interpretation of these smoothing methods and the introduction of a new NS loss called Triplet-based SANS (T-SANS). Experimental results on various datasets demonstrate the effectiveness of T-SANS.

### Strengths
S1 The paper provides a comprehensive theoretical understanding of smoothing methods for NS loss in KGE.
S2 The paper presents experimental results on multiple datasets, showcasing the effectiveness of T-SANS.

### Weaknesses
W1 While T-SANS aims to improve upon existing methods, the computational overhead, especially in terms of memory usage and processing time, might not be thoroughly addressed.

W2 While the paper provides a comprehensive theoretical understanding of smoothing methods for NS loss in KGE, it might be too dense for a broader audience. The depth of the theoretical content might make it less accessible to practitioners or researchers from adjacent fields.

W3 How does T-SANS handle extremely sparse datasets compared to other methods? Is there a threshold of sparsity beyond which T-SANS might not be as effective?

W4 How generalizable is T-SANS to other related tasks beyond KGC? Has it been tested on tasks other than KG embedding?

### Questions
Q1 How does T-SANS handle extremely sparse datasets compared to other methods? Is there a threshold of sparsity beyond which T-SANS might not be as effective?

Q2 How generalizable is T-SANS to other related tasks beyond KGC? Has it been tested on tasks other than KG embedding?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates how different smoothing methods affect the negative sampling losses for knowledge graph embedding. It introduces a new triplet-based self-adversarial negative sampling method that can adjust the frequencies of triplets, queries, and answers in the training data. It evaluates the proposed method on three benchmark datasets with five base models and demonstrates its effectiveness.

### Strengths
1. Negative sampling is a crucial technique for learning KG embeddings. This paper offers a valuable insight into the smoothing methods for learning loss in KGE. I think it is an interesting and relevant work for the KGE community.

2. Based on the comparison and analysis of existing smoothing methods, the paper proposes triplet-based SANS, which can outperform other baselines on three datasets.

### Weaknesses
1. In my view, the proposed method is incremental work based on previous studies. It is an extension of SANS.

2. Another weakness is that the selected KGE models in the experiments are old. Some popular or recent models, such as TuckER [1] and HousE [2], are not included, which, in my view, may weaken the soundness the work.

[1] Ivana Balazevic, Carl Allen, Timothy M. Hospedales: TuckER: Tensor Factorization for Knowledge Graph Completion. EMNLP/IJCNLP (1) 2019: 5184-5193

[2] Rui Li, Jianan Zhao, Chaozhuo Li, Di He, Yiqi Wang, Yuming Liu, Hao Sun, Senzhang Wang, Weiwei Deng, Yanming Shen, Xing Xie, Qi Zhang: HousE: Knowledge Graph Embedding with Householder Parameterization. ICML 2022: 13209-13224

### Questions
1. Why are some results on YGAO3-10 missing? I think it would be better to produce the results using open-source implementations.

2. Is it possible to provide any analysis or experimental results to assess the effect of negative sampling on the convergence rate?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

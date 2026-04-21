# Towards Better Evaluation of GNN Expressiveness with BREC Dataset

- Avg Score: 5.67
- Decision: Reject
- Scores: 5, 6, 6

## Abstract
Research on the theoretical expressiveness of Graph Neural Networks (GNNs) has developed rapidly, and many methods have been proposed to enhance the expressiveness. However, unifying all kinds of models into one framework is untractable, making it hard to measure and compare their expressiveness quantitatively. In contrast to theoretical analysis, another way to measure expressiveness is by evaluating model performance on certain datasets containing 1-WL-indistinguishable graphs. Previous datasets specifically designed for this purpose, however, face problems with difficulty (any model surpassing 1-WL has nearly 100\% accuracy), granularity (models tend to be either 100\% correct or near random guess), and scale (only several essentially different graphs in each dataset). To address these limitations, we propose a new expressiveness dataset, **BREC**, including 400 pairs of non-isomorphic graphs carefully selected from four primary categories (Basic, Regular, Extension, and CFI). These graphs have higher difficulty (up to 4-WL-indistinguishable), finer granularity (can compare models between 1-WL and 3-WL), and a larger scale (400 pairs or extend to 319600 pairs or even more). Further, we synthetically test 23 models with higher-than-1-WL expressiveness on our BREC dataset. Our experiment gives the first thorough measurement of the expressiveness of those state-of-the-art beyond-1-WL GNN models and reveals the gap between theoretical and practical expressiveness. We expect this dataset to serve as a benchmark for testing the expressiveness of future GNNs. Dataset and evaluation codes are released at: https://github.com/brec-iclr2024/brec-iclr2024.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper tackles the problem of benchmarking GNNs, especially being concerned with their expressive power. The authors propose the BREC dataset to tackle this problem. The dataset aims to fix some issues in existing datasets, in particular, the authors argue that existing benchmarks are not very granular and too easy with models either achieving perfect accuracy or random guessing. The authors further propose an evaluation technique that is more principled than existing techniques, that takes also into account numerical errors.

### Strengths
The authors are very thorough in their work and consider a number of interesting points when designing the dataset. I particularly enjoyed the section that takes into account numerical errors induced for instance by floating point arithmetic errors. I think that the authors propose a principled approach to tackle this issue that could be valuable in general, outside of the scope of this specific work as well.

### Weaknesses
While the work is valuable, I believe that it is not novel enough in its current state. The tasks are very synthetic and even though I agree with the authors that the BREC dataset seems more interesting than existing benchmarks such as CSL, in my opinion it seems to be a marginal improvement over such datasets in terms of the novelty factor. There is definitely a need for such datasets in the community and the new evaluation technique is interesting, but I am not convinced that in its current state the work is fit for ICLR. Overall, this feels more of an "extension" to current datasets than something very novel.

### Questions
Would the authors be able to clarify any interesting findings that come from Table 2? It is a bit challenging for me to spot any significant trends the way it is currently ordered. It might be useful to further group the models based on their type. 

Would it be possible to clarify the contributions of the work? I understand that the work is proposing a new dataset and a new evaluation technique for it (which I find to be interesting and valid), but are there further novel contributions?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper overcomes the limitations of previous expressiveness datasets in terms of difficulty, granularity, and scale by introducing 4 datasets. Each dataset covers different benchmarking purposes for comprehensive GNN expressiveness evaluations. The authors further introduce Reliable Paired Comparisons instead of applying traditional classification comparisons to eliminate possible spurious correlations that can lead to unfair comparisons.

### Strengths
1. The paper is the first benchmark that can cover different difficulties with fine granularity.
2. The authors are the first to implement CFI graphs that enable any k-WL tests.
3. Applying pair-wise comparisons instead of classifications to eliminate influence from other factors is reasonable and rigorous.
4. To overcome the dilemma between false negative and false positive, the authors propose RPC that includes similarity comparisons and internal fluctuation considerations. Moreover, the authors propose to adjust the similarity threshold adaptively.
5. The code is well-organized.

### Weaknesses
The overall benchmark is comprehensive and elaborated. I only have a few minor concerns/questions.
1. Why do the authors adopt cosine similarity instead of other contrastive loss? Will this loss be possible to introduce any spurious correlation leading to biased results?
2. Since PPGN was compared, I'm wondering why "INVARIANT AND EQUIVARIANT GRAPH NETWORKS" was not included in the comparisons.

### Questions
N/A

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Previous datasets designed for evaluating the expressiveness of GNNs had limitations in terms of difficulty, granularity, and scale. The authors propose a new expressiveness dataset called BREC, which includes 400 pairs of non-isomorphic graphs carefully selected from different categories. The dataset offers higher difficulty, finer granularity, and larger scale compared to previous datasets. The authors conduct experiments on 23 models with expressiveness beyond 1-WL (Weisfeiler-Lehman) on the BREC dataset, providing a thorough measurement of their expressiveness and highlighting the gap between theoretical and practical expressiveness.

### Strengths
* It is a valuable problem for evaluating the expressiveness of GNNs.
* The benchmark is extensive, including datasets with higher difficulty, finer granularity, larger scale and many models for comparisons.
* It is promising for the BREC dataset to serve as a benchmark for testing the expressiveness of future GNNs.

### Weaknesses
1. One concern is about the evaluation. It shows in Table 2 that only 400 samples are included in total. Including more samples in the datasets would make the experiments more convincing.

2. What about the relationship about the performance on the proposed synthetic datasets and real-world datasets? In other words, is the model that  shows to more expressive in the benchmark performing better in real-world tasks? It would be better to provide more fine-grained analyses.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

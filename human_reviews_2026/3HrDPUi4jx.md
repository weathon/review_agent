# Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6

## Abstract
The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for comprehensive understanding of scaling properties. This is challenged by: 1) the emergence phenomenon, where unpredictable capabilities appearing suddenly at critical model scales; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby constructing a more stable and predictable task subset that exhibits well-behaved scaling characteristics with the increase of compute budget. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.55\% average prediction error across eight key LLM benchmarks, thus providing actionable insights for scaling properties and training monitoring during LLM pre-training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a cluster-based method for task scaling law (i.e., predicting larger model performance based on smaller model performance). The method has four steps: (1) cluster test examples based on their difficulty features (i.e., how small language models perform on the example); (2) fitting scaling law for each cluster; (3) extrapolation for each cluster; (4) mapping back to the full evaluation set performance. On an internal model series, the proposed method outperforms baselines such as end-to-end prediction or loss-intermediate prediction. Further ablation is done on clustering methods and scaling law formula.

### Strengths
* The paper presents an effective and reasonable approach for downstream task scaling law prediction. The pilot study points out limitations about several questionable assumptions in task scaling law, which I found insightful. The proposed method is reasonable and intuitive.

### Weaknesses
* Some details of the proposed method is unclear or missing. See questions below.
* Results were limited to one model series. The paper could be more convincing if the method was shown to be effective on various model families.

### Questions
* Line 151: "three random clusters" - does this mean data points are randomly assigned to clusters, or three clusters randomly selected from your work in later chapters? Very confusing at this point.
* Line 214: "we generate 100 samples" - does this mean sampling 100 answers for every input query?
* Line 337: "we calibrate ... using evaluation results from existing models as anchors". Does this mean you fit the parameters in Eq. (5) using another model's performance? After reading appendix C.3, it seems this anchoring step is critical as removing it will results in deteriorated performance (worse than the end-to-end baseline). I have three follow-up questions regarding this: (1) Do baseline methods have access to this extra information? (2) Can you elaborate more on the in-house MoE model you used here? How large is it? (3) Can you report the performance of the calibration model and the target model (70B) on the 8 tasks in Table 1?

(I'm willing to adjust my rating if the questions above are resolved.)

Suggestions:
* I was very confused when reading the abstract. Terminologies like "difficulty scaling features" and "support subset" were not defined. "hindering prediction by smaller models" is confusing, do you mean "extrapolation from performance using smaller models"? "the exclusion of tasks" - do you exclude tasks or test examples? (My current understanding is that you exclude some test examples.)
* Suggested reference:
  * Establishing Task Scaling Laws via Compute-Efficient Model Ladders (https://arxiv.org/abs/2412.04403)
  * Observational Scaling Laws and the Predictability of Language Model Performance (https://arxiv.org/abs/2405.10938)
  * How Predictable Are Large Language Model Capabilities? A Case Study on BIG-bench (https://arxiv.org/abs/2305.14947)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Clustering-On-Difficulty (COD) to predict the downstream accuracy of a large model from smaller models. The pipeline is: (i) build a task‑wise pass‑rate vector across model scales and cluster tasks (improved MeanShift with radius & min‑size constraints) to reduce intra‑cluster variance; (ii) fit a cluster‑wise performance–compute scaling law derived from a loss scaling assumption; (iii) filter “non‑extrapolatable” clusters via parameter thresholds; (iv) map the predicted accuracy on the “predictable subset” to the full evaluation set with a monotone polynomial (quartic by default), optionally calibrated with anchor models. On eight eval sets (GSM8K, MATH, BBH, TriviaQA, MBPP, AGIEval, DROP, MMLU‑pro), COD reports 1.36% average prediction error across eight key LLM benchmarks, outperforming loss‑intermediate, end‑to‑end, pass‑rate, and BNSL baselines.

### Strengths
1. Directly predicting downstream performance (not just loss) is operationally valuable.
2. The author spends much effort on training nine models (122M→70B) under a consistent recipe and eight diverse benchmarks. Such results are valuable for the community.
3. The paper is well-written and easy to understand.

### Weaknesses
1. The generalizability of the model is questionable. All selected benchmarks are widely adopted in the community. How does the proposed generalize to unseen benchmarks?
2. The author only compares with their own baselines. Many performance prediction solutions on downstream benchmarks are missing in the comparison.
3. The COD experiments heavily rely on heuristic hyperparameters such as a, b & c. This limits its generalizability to other cases.
4. Typos: in the abstract, the mean error is 1.36%, in Table 1, it is 1.63%. Which one is correct?

### Questions
See cons

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
## Summary

This work  proposes a new framework for predicting downstream performance using smalller proxy models. It introduces Clustering-On-Difficulty (COD), which groups samples in eval tasks by estimated difficulty  to reduce variance and enable more stable predictions￼. The paper provides theoretical analysis to justify this approach, and shows, thorugh extensive experimental results, that their approach outperforms other competing methods. Overall it is a strong work, and should be, in my opinion, accepted to ICLR 2026.

## Typos, missing references
- line 61: " 1.36%ha"
- [Bhagia et al 2024](https://arxiv.org/abs/2412.04403)

### Strengths
- Method is effective, well justified, and works across large set of taks.
- The emprical results are very convincing.
- The paper includes extensive ablations in main text and appendix.
- Paper is important contribuition in the field of estimating downstream performance, as it significantly improve ability to predict downstream performance.

### Weaknesses
- The core for the clustering approach seems to hinge on the estimation of difficulty per each instance in a benchmark. To do so, the paper seems to imply that models that are also target of scaling laws are used. Unclear if the proposed approach would work if using a different model family (i.e., how would this approach work if one has to bootstrap from external open-weights LLMs?)
- To estimate task difficulty, it seems that one need a model already trained at largest target size to build clustering features? Is that the case?
- The paper is missing an ablation showing how prediction accuracy changes if fewer/smaller models are used.
- The main text is really difficult to follow due to missing details. The details are generally provided in the appendix, but the back-and-forth makes following the approach very hard. 
- The work is applied to MC tasks only. MC tasks have interesting discuitnuity properties [\(Wiegreffe et al, 2025\)](https://arxiv.org/abs/2407.15018), but are far from the only formulation of interest.

### Questions
I've made some educated guesses in my weaknesses section. Would be great if the authors could comment on them to ensure that I'm correct.

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tries to address the challenge of accurately predicting downstream performance of large language models (LLMs) before full-scale training. Existing scaling laws—either based on loss or direct performance–compute curves—fail to capture emergent behaviors and heterogeneous task difficulty.
The authors propose a Clustering-On-Difficulty (COD) framework that groups evaluation samples by their performance-scaling patterns using an improved MeanShift algorithm. Within each cluster, they fit a theoretically derived performance scaling law to extrapolate results from small models to large ones. Predictions from extrapolatable clusters are then mapped via a polynomial function to the full benchmark.
Experiments on eight benchmarks (e.g., MATH, GSM8K, BBH, MMLU-pro) show that COD achieves an average better prediction error of a 70B model than baselines.

### Strengths
1. The COD approach provides a clear decomposition of the performance-prediction problem into clustering, cluster-wise fitting, subset extrapolation, and subset-to-full mapping. This layered design could handle emergent and heterogeneous behaviors better.
2. The improved MeanShift with radius and minimum-cluster constraints balances automatic cluster discovery and intra-cluster homogeneity, outperforming K-Means, DBSCAN, and baseline MeanShift on both intra-cluster metrics and prediction error.
3. Eight diverse benchmarks and multiple baselines are covered, including detailed ablations (clustering, extrapolation formulas, mapping functions).
4. The paper provides pseudocode, theoretical proofs, and extensive appendices on hyperparameters, smoothing, and computational cost.

### Weaknesses
1. COD introduces multiple stages (feature extraction, clustering, extrapolation, mapping) with several hyperparameters (radius R, min cluster K, thresholds a/b/c, polynomial degree). This may hinder practical adoption and stability.
2. In what extent of error rate, could the preformance prediction be useful? Can this actually reflect the actual final benchmark performance?
3. In introduct the error rate is 1.36%, but in the table of experiment results, the error rate seems to be 1.63, which is inconsistent.

### Questions
Some typos: e.g. 1.36%ha in line 061, can them be fixed?

Any specific evidence why this method could handle better emergent and heterogeneous behaviors?

### Soundness
2

### Presentation
2

### Contribution
3

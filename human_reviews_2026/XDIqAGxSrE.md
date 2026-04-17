# Influence Guided Sampling for Domain Adaptation of Text Retrievers

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
General-purpose open-domain dense retrieval systems must usually be trained with a large, eclectic mix of corpora and search tasks. How should these diverse corpora and tasks be sampled for training? Conventional approaches are to sample them uniformly, or proportional to their instance population sizes, or depend on human-level expert supervision. It is well known that the training data sampling strategy can greatly impact model performance. However, how to find the optimal strategy has not been adequately studied in the context of embedding models. We propose Inf-DDS, a novel reinforcement learning–driven sampling framework that adaptively reweighs training datasets guided by influence‑based reward signals and is much more lightweight w.r.t. to GPU consumption. Our technique iteratively refines the sampling policy, prioritizing sampling from datasets that maximize the model performance on a target development set. We evaluate the efficacy of our sampling strategy on a wide range of text retrieval tasks, demonstrating strong improvements in retrieval performance and better adaptation compared to existing gradient-based sampling methods, while also being *1.5×–4×* cheaper than them in terms of GPU compute needed. Our sampling strategy achieves a **5.03** absolute *NDCG@10* improvement while training a multilingual *bge-m3-dense* model and an absolute *NDCG@10* improvement of **0.94** while training *sentence-transformers/all-MiniLM-L6-v2*, even when starting from an expert assigned weights on a large pool of training datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Inf-DDS, a novel method that uses influence, the metric gain from a small number of updates on the development set, as a reward signal for training a dynamic data sampling strategy. This method is designed to optimize the training data distribution for downstream tasks, particularly in dense retrieval settings. Empirical results on multiple benchmarks, including MLDR and BEIR, demonstrate significant improvements over existing baselines. The paper also introduces engineering solutions like weighted Reptile to reduce computational and memory overhead, making the method more efficient. However, the paper does not sufficiently address the risks of overfitting and data leakage.

### Strengths
1. **Clear and Direct Reward Design**: The use of influence as a reward based on actual metric improvements on the dev set makes the method straightforward and directly aligned with the downstream task's goals. 
2. **Effective Engineering Design**: The use of weighted Reptile for gradient sharing and memory efficiency shows thoughtful consideration for practical implementation, making the approach scalable to large datasets and models.
3. **Stability and Robustness**: The method demonstrates stable learning trajectories, with less sensitivity to noisy gradients compared to alternative methods like DoGE or MultiDDS. The authors provide a useful analysis of sampling stability.

### Weaknesses
1. **Potential Overfitting and Data Leakage**: The method uses a portion of the downstream test data as the development set to guide data sampling for the training set. This introduces the risk of **data leakage** and overfitting, as the model may learn to optimize for the specific dev set rather than generalizing to unseen data. This is a **key concern**, as shifting the dev set may lead to performance degradation when applied to a new, unseen dev split. The paper does not sufficiently investigate or mitigate this issue, nor does it perform experiments to confirm the stability of the approach across multiple dev splits.
2. **Insufficient Theoretical Analysis**: While the paper presents empirical results, it lacks a detailed theoretical analysis of the potential biases and variance in the influence estimates, especially when the number of inner steps is small. The relationship between influence and long-term generalization remains unexplored. A more rigorous theoretical grounding would enhance the robustness of the method’s claims.
3. **Unclear Generalization Across Domains**: The authors do not sufficiently explain or diagnose certain anomalies in their experiments, such as the strong upsampling of Swahili in the MLDR experiments. The paper should offer an analysis of why such samples provide improvements in performance and whether this is due to the model overfitting to particular domains or justifiable improvements in generalization.
4. **Lack of Comparison with Related Works**: The paper does not cite or compare with proxy-model based methods like DoReMi algorithms, especially methods do not involve downstream task data. In the realm of dense embedding data sampling optimization, there are also previous works are not cited or compared in the paper. For example, tDRO (Task-level Distributionally Robust Optimization for Dense Retrieval) addresses similar issues of dataset-level weighting for improving domain robustness in the realm of dense embedding fine-tuning. 
5. **Compute and Efficiency Trade-offs**: The computational cost of the proposed method is not adequately quantified, particularly when scaling to large datasets. The paper lacks a detailed comparison of the compute cost, GPU memory usage, and sample efficiency between Inf-DDS and other methods, making it difficult to assess the scalability of the method in practical scenarios.

### Questions
1. **Dev Split Sensitivity**: The method relies on a portion of the test data for the dev set. Could you provide experiments showing the sensitivity of the method to different dev splits and report the variance in performance across multiple splits?
2. **Overfitting Risk**: Have you considered any measures to regularize the learning of the sampler to prevent overfitting to noisy or biased dev sets? How do you plan to mitigate the risk of data leakage or the model becoming overly sensitive to the dev set?
3. **Computational Efficiency**: Can you provide a more detailed analysis of the computational trade-offs, such as GPU hour saving and sample efficiency, when compared to other methods?
4. **Comparison with proxy-model based methods**: Can you compare your approach with the recent proxy-model based methods in dense embedding training, e.g. tDRO, to highlight the differences and potential advantages of Inf-DDS, particularly in terms of generalization across domains and robustness to dev-split variations?
5. **Swahili Data Anomaly**: In your MLDR experiment, why does the Swahili language show a strong performance improvement when upsampled? Can you provide a more detailed analysis or an ablation study to explain this result?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Inf-DDS, a reinforcement learning–based sampling framework designed to optimize training data selection for general-purpose open-domain dense retrieval systems. Unlike conventional sampling methods that rely on uniform distribution, proportional instance counts, or expert supervision, Inf-DDS adaptively reweights datasets using influence-based reward signals to enhance model performance on a target development set. The approach requires less GPU usage than gradient-based alternatives. Empirical evaluations across various text retrieval tasks show that Inf-DDS improves retrieval performance, including gains in NDCG@10 scores for both multilingual and sentence-transformer models.

This is a well written paper making a meaningful contribution to training dense retrievers using a more principled way to sample datasets.

### Strengths
- This work addresses the important topic of sampling datasets and tasks for training general-purpose dense retrievers in a more principled manner.
- The proposed sampling method was shown to improve retrieval performance while requiring considerably less compute over the baselines across multiple datasets. 
- The presentation is clear overall, though Fig 2 (step 3) is not consistent with the pseudocode.

### Weaknesses
- The choice for the initial dataset sampling probability distribution is not justified. 
- The performance across the dev sets is considered to be equally important, while it may not be the case.

### Questions
- The paper stresses that larger dataset size doesn’t necessarily translate to more effective training, yet the dataset sampling probability distribution is initialized proportional to the dataset size. Why?
- The results shown in Fig 4b is unintuitive. For test domains for which training data is available (FEVER and HotpotQA), why aren’t the matching training domains given the highest weight? Is it just Inf-DDS being imperfect?

### Soundness
4

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
3

### Summary
This paper proposes Inf-DDS, a dynamic data sampling method for domain-adaptive text retriever training.

### Strengths
The problem of selecting informative samples during training is important, especially for reducing computation or improving convergence.
The core idea of prioritizing domains by observed dev improvement (rather than gradient alignment) is intuitive and close to the end goal.

### Weaknesses
1. The paper optimizes proxy losses (e.g., InfoNCE / KD loss deltas) but does not demonstrate that these correlate with ranking metrics such as NDCG@10.
2. Influence estimation requires extra forward/backward or Hessian-vector steps. The paper calls the method efficient but does not report GPU-hours / wall-clock / memory, so it is unclear whether gains outweigh the additional compute.
3. The paper alternates between linear-normalized influence weights and softmax Reptile updates, but does not clearly state which variant is used where, nor how τ affects training, hindering reproducibility.
4. No significance testing, so stability and generalizability remain unclear.

### Questions
1. For BEIR / Sent-Trans / MLDR, could the authors clearly specify:
(a) the training loss used for the retriever,
(b) the proxy loss used to estimate influence,
(c) the metric M whose change is used to define influence?

2. The paper alternates between describing:
linear-normalized influence weighting, and softmax weighted Reptile updates. So which update rule is actually used in the main reported results？

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Inf-DDS, a reinforcement-learning–based framework for domain adaptation in text retrievers. Instead of relying on static or gradient-based sampling, Inf-DDS uses influence scores—measuring how each training dataset affects performance on development sets—to update a sampling policy iteratively. It employs online proxy models and Reptile-style meta-updates to efficiently reuse gradients, thereby reducing GPU overhead. 

Experiments on BEIR, MLDR, and Sentence-Transformers training corpora show that Inf-DDS consistently improves NDCG@10 over MultiDDS, DoReMi, DoGE, and static sampling, with gains of up to +5.03 on MLDR and +0.94 over expert-curated weights on MiniLM. The method produces more stable sampling trajectories, generalizes across heterogeneous domains, and reduces compute by 1.5×–4× compared to gradient-based baselines.

### Strengths
* Novel influence-based reward mechanism offering more stable, interpretable sampling than gradient-based baselines.
* Computational efficiency via gradient reuse and partial subsampling.
* Clear motivation bridging influence functions with adaptive sampling.

### Weaknesses
* Reward estimation cost: computing per-domain influence still scales poorly for very large dataset pools; proxy reliance may not generalize.
* Influence score stability: although more stable than gradient-based methods, influence estimation still depends on the correctness of the proxy update steps.
* Initialization sensitivity: Inf-DDS performance depends heavily on the initial sampling distribution.
* Overfitting to dev sets: using dev-based rewards risks domain leakage.
* Influence effect on unexpected domains (e.g., Swahili in MLDR) remains unexplained and raises questions.

### Questions
Influence computation robustness: How does the method behave when proxy model updates and target metric diverge? Can influence computation amplify noise under distribution shifts?

Granularity question: Could instance-level or cluster-level influence scoring outperform dataset-level scoring? What prevents InfDDS from integrating finer-grained sampling within each dataset?

### Soundness
3

### Presentation
3

### Contribution
2

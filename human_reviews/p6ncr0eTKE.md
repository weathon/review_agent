# Task-Adaptive Pretrained Language Models via Clustered-Importance Sampling

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Specialist language models (LMs) focus on a specific task or domain on which they often outperform generalist LMs of the same size. However, the specialist data needed to pretrain these models is only available in limited amount for most tasks. In this work, we build specialist models from large generalist training sets instead. We adjust the training distribution of the generalist data with guidance from the limited domain-specific data. We explore several approaches, with clustered importance sampling standing out. This method clusters the generalist dataset and samples from these clusters based on their frequencies in the smaller specialist dataset. It is scalable, suitable for pretraining and continued pretraining, it works well in multi-task settings. Our findings demonstrate improvements across different domains in terms of language modeling perplexity and accuracy on multiple-choice question tasks. We also present ablation studies that examine the impact of dataset sizes, clustering configurations, and model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this work, the authors investigate the data selection problem during the pre-training and continued pre-training stages of language models. They propose CRISP, a method that clusters both the generalist and specialist datasets, and then selects data from the generalist dataset based on the cluster distribution of the specialist dataset. The authors conduct extensive experiments on six benchmarks to demonstrate the effectiveness of CRISP.

### Strengths
1. The authors conduct comprehensive experiments, utilizing significant computational resources, to validate the effectiveness of the proposed CRISP method.
2. Although the computational cost of CRISP is not explicitly discussed, based on its methodology, CRISP appears to be computationally efficient.

### Weaknesses
1. The paper is somewhat hard to follow. For instance, CRISP requires a K-means clustering model to select task-specific data, but the authors do not mention on which dataset the K-means model is trained. Furthermore, many technical details are placed in the appendix, making the paper hard to follow and not self-contained.
2. CRISP essentially selects data from the generalist dataset based on the distribution of the specialist dataset. There are several baseline approaches that should be considered, such as Cross-Entropy Difference [1], as mentioned in the related work section.
3. Although the author conduct extensive experiments with substantial amount of compute, the scientific contributions of this work appear to be marginal and the findings are not particularly surprising. Like previous works, CRISP aims to align the selected data more closely with the targeted domain. It is unclear what unique strengths CRISP offers compared to prior methods. For example, given the vast amount of training data, it would be beneficial if the authors can demonstrate CRISP is significantly more efficient than existing approaches. Furthermore, the data quality in the selection process might be another factor to consider.


### Additional Comments (do not affect my rating)

1. Recent works [2, 3] have demonstrated that results on RewardBench have a negative correlation with real downstream performance. I suggest the authors consider using other benchmarks for evaluation. Although these findings are very recent and were reported after the paper submission deadline—and it's not the authors' fault—I believe it would be beneficial if they could adjust their evaluation settings accordingly.
2. The authors utilize significant computational resources for the experiments in this work. I appreciate these efforts. It would be helpful if the authors could present a comparison regarding the compute cost to better understand the pros and cons of the approaches discussed.

Reference:

[1] Moore, Robert C., and William Lewis. "Intelligent selection of language model training data." In Proceedings of the ACL 2010 conference short papers, pp. 220-224. 2010.

[2] Zhou, Enyu, Guodong Zheng, Binghai Wang, Zhiheng Xi, Shihan Dou, Rong Bao, Wei Shen et al. "RMB: Comprehensively Benchmarking Reward Models in LLM Alignment." arXiv preprint arXiv:2410.09893 (2024).

[3] Frick, Evan, Tianle Li, Connor Chen, Wei-Lin Chiang, Anastasios N. Angelopoulos, Jiantao Jiao, Banghua Zhu, Joseph E. Gonzalez, and Ion Stoica. "How to Evaluate Reward Models for RLHF." arXiv preprint arXiv:2410.14872 (2024).

### Questions
1. If I understand the proposed approach CRISP correctly, should line 8 of Algorithm 1 be $x_i \sim \text{Uniform}(D^{g} \cap K(c_i))$?
2. Typically, we use accuracy to measure performance on MMLU, while the authors compute perplexity on MMLU. How are the inputs formatted? Are the question and the corresponding correct option concatenated?
3. As shown in Algorithm 1 and Figure 1, CRISP resamples the generalist data based on the specialist data histogram. Is the specialist dataset not used in the training process?

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
4

### Summary
This work focuses on training the specialist LM to avoid the limitation of scarce  specialist data.  To this end, this work explore three data selection method during pre-training in which the clustered importance sampling (named CRISP in this work) performs best. The CRISP aims to modifies the distribution of a generic training set guided by little specialist data. To be specific, the method resamples the generic set such that its cluster histogram matches the specialist data. Therefore, the model can always select the samples with the same or similar distribution with the targeted domains. Experimental results demonstrate that pre-training with this method provides better specialist models both for LM and question answering tasks. Besides, the additional analysis experiment benefits readers more, e.g., investigating the impact of model size, training set size, the number of clusters, clustering methods, continued pre-training and multi-task settings. All of the results show the advantage of the proposed method.

### Strengths
1. The authors extend the importance sampling (Grangier et al., 2024b) through adjusting the frequency of generalist clusters guided by specialist data. Besides, it scale the method to millions of clusters, show that it works with larger models, and extend it beyond language modeling tasks (i.e., adding the downstream tasks).
2. Extensive experiments show the importance of sampling data and the effectiveness of the cluster-based importance sampling method.
3. The cluster-based importance sampling method may benefit the researchers that devoted into the end-to-side small LLMs.

### Weaknesses
1. The cluster-based importance sampling method is not proposed by this work and the authors only show it works in larger models, and extend it to MQA tasks. Therefore, the contribution of technology is limited. 
2. From my perspective, the CRISP method frequently selects samples (tokens) that share the same distribution as the target domain. What if you were to iterate over generalist samples/tokens—without distinguishing whether they belong to the same domain—the same number of times as you would when training specialist LMs?
3. When a new domain or downstream task coming, it has to be trained a new specialist LMs from scratch. It maybe time-cost and laborious. 
4. If you apply the CRISP method based on a pre-trained generalist LMs, I wonder whether it can gain more benefit. If not, does it mean that you are still not using enough data in current pre-training setting?

### Questions
See the weaknesses.

### Soundness
2

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
2

### Summary
This paper addresses the challenge of building specialist language models (LMs) when domain-specific data is limited. The authors introduce Clustered Importance Sampling (CRISP), which adapts a generalist dataset for specialist pretraining by adjusting its distribution to align with limited domain-specific data. CRISP’s effectiveness is demonstrated across various domains, yielding improvements in language modeling perplexity and multiple-choice question accuracy. The approach is scalable, applicable to both small and large models, reduces the need for costly domain-specific data, and consistently outperforms other adaptation techniques, such as classifiers and gradient alignment, across a range of tasks and domains.

### Strengths
S1. The problem is clearly defined and well-motivated, addressing the need for effective specialist language models when domain-specific data is limited.

S2. CRISP demonstrates superior performance compared to both untrained and pre-trained models on the Redpj2 baseline, as well as classifier-based and DoGE approaches. It achieves lower perplexity on language modeling tasks and higher accuracy on multiple-choice question (MCQ) tasks across four different domains.

S3. The paper provides a comprehensive analysis, exploring the effects of cluster numbers, model sizes, varying amounts of training data, task transferability, and multitask training. CRISP has also been shown to be more efficient in terms of training cost and scales well with larger models.

### Weaknesses
W1. While the paper provides a comprehensive analysis of different cluster numbers, it is limited in terms of guidance on determining the optimal number of clusters beyond empirical trial and error, even though this is a crucial parameter for CRISP’s performance. Results indicate that the number of clusters influences the sampling distribution and domain adaptation effectiveness, concerning token repetition or overfitting in certain configurations.

W2. Although CRISP improves model performance in non-pretrained scenarios, it offers limited additional benefits for MCQ tasks after fine-tuning compared to task-specific pretraining.

W3. When the specialist dataset $\mathcal{D}^s$ is very small, CRISP appears to be less effective in enhancing end-task accuracy.

### Questions
I have the following additional questions and comments:

- As shown between SBERT and LSI experiments, the choice of clustering model impacts performance. Would using a more powerful embedding model like E5-large instead of the current MiniLM-L6-v2 improve performance, considering factors such as computational cost (as using E5-large would be more costly) and potential performance gains?
-  How long does this process typically take to build the clustering tree for different configurations? I think Figure 10 only accounts for the training cost and it would be nice to compare it with the overall training cost.

I have also identified some potential typos:
- Currently, the equation on page 4 after equation (2) gives $\mathcal{L}(\mathcal{D}^{s}; \theta) = \sum\limits_{c} \mathcal{L}(c; \theta) \frac{P(c | \mathcal{D}^{s})}{P(c | \mathcal{D}^{g})} P(c | \mathcal{D}^{g}) = \underset{c \sim (c | \mathcal{D}^{s})}{\mathbb{E}} [w(c) \mathcal{L}(c; \theta)]$. Shouldn't this be $\mathcal{L}(\mathcal{D}^{s}; \theta) = \sum\limits_{c} \mathcal{L}(c; \theta) \frac{P(c | \mathcal{D}^{s})}{P(c | \mathcal{D}^{g})} P(c | \mathcal{D}^{g}) = \underset{c \sim (c | \mathcal{D}^{g})}{\mathbb{E}} [w(c) \mathcal{L}(c; \theta)]$, that is $c \sim (c | \mathcal{D}^{g})$ instead of $c \sim (c | \mathcal{D}^{s})$? In addition, if correction is needed, would this change the interpretation of the Importance Sampling as the equation changes?
- In Algorithm 1, I think line 8 should be $x_i \sim \text{Uniform}(D^{g} \cap K(c_i))$ as it samples a generalist example over a samples cluster id in the large set as explained in previous paragraphs. In addition, in line 10, the set should start with $x_1$ instead of $x_0$ to be consistent with the indexing.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes a novel method to pre-train a specialist language model, where the amount of specialist data is scarce, and the amount of generalist data is rich. The high-level idea in this method is to sample from the generalist data and select the ones that are more similar to the specialist data. The authors target it with a simple cluster-then-importance-sampling, showing improvement in perplexity and accuracy on language model and multiple choice QA tasks.

### Strengths
* The proposed method is simple while effective, showing competitive performance compared to vanilla pre-training and other baselines.

* The ablation study in Section 5 is comprehensive and well-explored.

### Weaknesses
* Previous work [1] has proposed a similar method of adjusting generalist data according to clusters informed by specialist data. The major contribution of the method is that this work shows that that method will work when scaled up.

[1] Specialized Language Models with Cheap Inference from Limited Domain Data

### Questions
It would be helpful if the authors could provide a more detailed association with previous work.

### Soundness
3

### Presentation
3

### Contribution
2

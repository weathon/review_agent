# Don't Pay Attention, PLANT It: Pretraining Attention via Learning-to-Rank

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
State-of-the-art Extreme Multi-Label Text Classification models rely on multi-label attention to focus on key tokens in input text, but learning good attention weights is challenging.
We introduce PLANT — Pretrained and Leveraged Attention — a plug-and-play strategy for initializing attention.
PLANT works by planting label-specific attention using a pretrained Learning-to-Rank model guided by mutual information gain.
This architecture-agnostic approach integrates seamlessly with large language model backbones (e.g., Mistral, LLaMA, DeepSeek, and Phi-3).
PLANT outperforms state-of-the-art methods across tasks such as ICD coding, legal topic classification, and content recommendation.
Gains are especially pronounced in few-shot settings, with substantial improvements on rare labels. Ablation studies confirm that attention initialization is a key driver of these gains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper addresses the task of XMTC, where prior approaches often rely on multi-label attention mechanisms. However, such attention is typically hard to train effectively. The authors attribute this difficulty to the random initialization of attention parameters and propose PLANT, a novel attention initialization strategy. PLANT guides label-specific attention using a pretrained Learning-to-Rank model based on mutual information. The authors evaluate PLANT across multiple LLM backbones and demonstrate consistent improvements.

### Strengths
1. The method is conceptually clean and straightforward, making it easy to understand and reproduce. It is also model-agnostic, which suggests it can be readily adapted to a variety of LLM backbones.

2. The experimental section is comprehensive and well-structured. The authors validate PLANT's effectiveness across different model sizes, datasets, and evaluation metrics, which provides strong empirical support for the proposed approach.

### Weaknesses
1. From a broader deep learning perspective, it is generally accepted that parameter initialization plays a crucial role in model training. However, in the specific context of this paper, the connection between poor rare-label performance and random attention initialization is not sufficiently established. Around Line 99, the authors provide evidence of poor performance on rare codes, which, as I understand it, is intended to motivate PLANT by suggesting the following causal chain: “Rare label performance suffers <- attention is poorly learned <- due to random initialization.” However, poor rare-label performance could also stem from other factors, such as optimization conflicts during training, rather than the initialization issue alone. I recommend that the authors provide further analysis or empirical evidence to support this causal connection, which is central to the motivation of the proposed method.

2. While the method is well-designed and easy to follow, it would benefit from more justification or intuition behind its design choices. For example, some additional motivation or theoretical analysis to support the necessity of specific architectural decisions would help reinforce the method's novelty. Without this, readers may perceive the contribution as lacking in innovation.

### Questions
1. From a broader deep learning perspective, it is generally accepted that parameter initialization plays a crucial role in model training. However, in the specific context of this paper, the connection between poor rare-label performance and random attention initialization is not sufficiently established. Around Line 99, the authors provide evidence of poor performance on rare codes, which, as I understand it, is intended to motivate PLANT by suggesting the following causal chain: “Rare label performance suffers <- attention is poorly learned <- due to random initialization.” However, poor rare-label performance could also stem from other factors, such as optimization conflicts during training, rather than the initialization issue alone. I recommend that the authors provide further analysis or empirical evidence to support this causal connection, which is central to the motivation of the proposed method.

2. While the method is well-designed and easy to follow, it would benefit from more justification or intuition behind its design choices. For example, some additional motivation or theoretical analysis to support the necessity of specific architectural decisions would help reinforce the method's novelty. Without this, readers may perceive the contribution as lacking in innovation.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a smarter initialization approach for attention modules commonly used in Extreme Multi-Label Text Classification (XMTC) methods, introducing PLANT (Pretrained and Leveraged AtteNTion). PLANT is a two-stage framework: in the first stage, attention weights are pretrained using a learning-to-rank objective guided by mutual information gain (MIG) to rank token relevance per label; in the second stage, the pretrained weights are fine-tuned end-to-end. Extensive experiments demonstrate that the proposed method consistently outperforms strong baselines.

### Strengths
- S1: The proposed method introduces a simple yet effective task-specific pretraining objective for attention initialization.

- S2: The authors conduct comprehensive experiments showing consistent improvements across multiple datasets and various LLM backbones.

- S3: The method achieves substantial gains on rare labels and under few-shot settings, indicating better generalization to low-data regimes.

### Weaknesses
- W1: The learning-to-rank objective based on mutual information gain relies on sufficient co-occurrence statistics, which may not generalize well to low-resource domains or languages with limited training data.

- W2: The MIG computation and learning-to-rank pretraining may be computationally expensive for large label spaces; the paper does not discuss practical scalability or runtime considerations.

- W3: The paper lacks qualitative analyses or visualizations of the learned attention patterns, leaving unclear whether the pretrained attention captures meaningful token–label relationships or merely reflects dataset biases.

### Questions
- Q1:Have you considered using alternative ranking metrics instead of NDCG in Equation (2)?

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
4

### Summary
This paper proposes PLANT (Pretrained and Leveraged Attention), a novel method for initializing multi-label attention weights through a learning-to-rank pre-training mechanism, aiming to improve the performance of long-tailed labels in extreme multi-label text classification (XMTC).

### Strengths
This paper is clearly written and easy to follow.

### Weaknesses
1.	PLANT includes additional MIG pre-computation and L2R stages, but the paper does not provide a comparison of the time, memory, or parameter amounts for these stages, nor does it explain their incremental contribution to the total training cost.
2.	The attention matrix learned by PLANT in Stage 1 is directly used in Stage 2, but without any maintenance or freezing strategy. If Stage 2 training is too long, the Stage 1 signals may be overwritten, causing the "planted attention" to fail.
3.	Although the authors claim PLANT is “architecture-agnostic,” all experiments are limited to text domains (ICD, law, WIKI). To demonstrate the generality of “plantable attention,” its transferability should be verified in at least one non-text domain (such as image or table classification).
4.	By comparing the distribution of random attention and PLANT attention on key tokens, it can be verified whether they truly capture the semantics of the tags rather than word frequency preferences.

### Questions
1.	The quantitative relationship between MIG (Mutual Information Gain) and token-label relevance is not rigorously defined. The paper states it is calculated "through statistical co-occurrence," but fails to explain the normalization or bias correction methods, potentially leading to over-amplification of high-frequency words. 
2.	The differentiable approximation of nDCG@k is modeled solely through pairwise sigmoid, lacking global constraints on ranking consistency and theoretically unable to guarantee consistent ranking optimality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper considers extreme multi-label classification (XMTC) problem where the goal is to identify a relevant subset of labels from an extremely large set. In particular, the paper targets "long" or "technical" documents where label attention plays a crucial role. 
1. The paper identifies initialization of label attention to be a key bottleneck in current methods. It demonstrates that a good initialization can significantly boost performance especially in case of data scarce tail labels.
2. The proposed approach PLANT first learns the attention mechanism by trying to predict the tokens. Stage 2, learns the network in end-to-end fashion for the specific task.
3. Results are reported on MIMIC-III-full, MIMIC-IV-full, EURLex-4K and Wiki10-31K datasets. The proposed approach outperforms the baselines on the considered datasets.

### Strengths
1. The paper is well written and easy to understand. The goals, problem formulation etc. are well defined. 
2. Table 2 demonstrates that introduction of PLANT improves over the baselines without PLANT initialization. The results are consistents for multiple LLM backbones, i.e., Mistral-7B, LLaMA3-8B, DeepSeek-V3, and Phi-3.
3. Tables 3 compares PLANT against existing methods on the MIMIC-IV-full. It outperforms existing methods (GPT-4 Zero-Shot, GKI-ICD, PLM-CA, and CoRelation) on AUC, F1 and precision@k.
4. PLANT outperforms XMTC baselines on EURLex-4K dataset (legal document classification) and Wiki10-31K (tag prediction on Wikipedia). It considers recent methods such as DE, GANDALF. Although they are with a different backbone. 
5. Comprehensive ablation experiments are performed to study the impact of backbone, initialization, stage-wise training, negative sampling etc.,

### Weaknesses
1. Although the paper presents good results on public datasets till 31K labels. It would be good to see the results on larger datasets. For example, product-to-product recommendation (LF-AmazonTitles-131K or LF-AmazonTitles-1.3M [1]) or tag-prediction (LF-Wikipedia-500K). Please note that Wikipedia dataset considers full-text documents which aligns with the setup.
2. The paper misses out on discussion on scalability. It would be good to report training and inference times.
3. Which backbone is used for PLANT in Table 5? The model size of PLANT seems to be significantly higher than the XMTC baselines (DistilBERT encoder with 66M parameter is used by multiple methods) in Table 5. Is the difference because of attention initialization or just the virtue of a bigger model. It should be discussed in detail. 
4. Propensity scored metrics are used to adjudge the performance on tail labels. The paper should presents results on these metrics for extreme classification datasets. 
5. Label attention has been discussed by the AttentionXML [1] paper (pre LLM era). It should be discussed.

References:
[1] https://dl.acm.org/doi/10.5555/3454287.3454810
[2] https://aclanthology.org/2025.naacl-long.537/

### Questions
1. How does PLANT compare against the baselines in terms of training and more importantly inference time?
2. RQ4: Does these experiments only consider few-shot labels? The real world problem is to predict relevant labels even if they are rare. 
3. How does PLANT fair against the baselines in propernsity scored metrics? 
4. Which specific negative sampling strategy is used? 
5. How did you choose $k$ in Table 3? Any specific significance behind the number 8? 
6. Missing equation tag in 866-867
7. Small suggestion: Typesetting (within text) can be helpful but it can be distracting when overdone.

### Soundness
3

### Presentation
3

### Contribution
2

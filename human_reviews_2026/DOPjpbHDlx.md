# R&B: Domain Regrouping and Data Mixture Balancing for Efficient Foundation Model Training

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
While data mixing strategies have successfully reduced training costs, existing methods suffer from two critical flaws: they rely on predetermined data domains that may fail to capture semantic nuances, and they scale computationally with the number of domains in a prohibitive way. We address these challenges by paying a fixed one-time cost to repartition source data into semantically similar domains and reusing training gradients to estimate domain importance. We propose **R&B**, a two-stage framework that re-partitions training data based on semantic similarity (**Regroup**) to create finer-grained domains, then efficiently optimizes the data composition (**Balance**) by leveraging a Gram matrix induced by domain gradients obtained throughout training. Unlike prior works, **R&B** removes the need for additional compute to obtain evaluation information such as losses or gradients. We analyze this technique under standard regularity conditions and provide theoretical insights that justify **R&B**'s effectiveness compared to non-adaptive mixing approaches. Empirically, we demonstrate the effectiveness of **R&B** on five diverse datasets ranging from natural language to reasoning and multimodal tasks. With as little as 0.01\% additional compute overhead, **R&B** matches or exceeds the performance of state-of-the-art data mixing strategies.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposed a novel gradient-based method (R&B) a two-stage framework for efficient data mixture optimization. The paper repartition as Regroup training data into semantically coherent clusters using embedding similarity, and it optimize domain weights as Balance to get individual domain contributions and cross-domain relationships, leveraging domain gradients computed via training. The paper validate the superiority of the proposed method across five diverse data settings.

### Strengths
- The paper is well-organized and easy to read for the main contributions. The theoretical justification are also thoroughly presented in the supplementary material. 
- The proposed method makes use of more flexible and granular grouping method instead of the fixed and hand-crafted categorizations and it results in better performances. 
- The proposed method achieves efficiency by avoiding extra computational costs because it uses gradient information derived during training. 
- The proposed method is validated over multiple types of tasks and datasets from LLM to image classification.

### Weaknesses
- The regrouping and balancing framework yield new hyperparameters that must be tuned, which adds an extra hyperparameter optimization burden.
- According to Fig. 3, the performance can vary heavily depending on the number of clusters in certain datasets, indicating that this parameter has a highly sensitive impact on the overall performance.
- As the number of clusters (domains) increases, the computation and memory cost of constructing and using a Gram matrix may become burdensome. the scalability and generalization of this approach under multiple domains deserve further studies.
- For tasks with highly diverse domain distributions, semantic clustering might blur distinct domain characteristics, potentially causing overfitting or under-representation of certain domains.
- Is there any plan to release the official code?

### Questions
- In line 140, the capital delta is not defined. 
- Why is the capital delta's dimension (m-1)?
- Is it possible to use mutual information instead of gram matrix?

### Soundness
4

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
The paper proposes a novel framework to address the limitations of existing data mixture optimization methods, which often fail to capture semantic relationships and suffer from computational scalability issues. Specifically, the proposed approach consists of two stages: (1) it partitions the training dataset into fine-grained domains based on semantic similarity, a step named to as __Regroup__; and (2) it leverages a Gram matrix of domain gradients to optimize the overall data composition, a step named __Balance__. The method is supported by theoretical analysis and demonstrates consistent effectiveness across five diverse benchmark datasets, spanning natural language understanding, reasoning, and multimodal tasks.

### Strengths
1. The proposed method is intuitive, straightforward, and effective. The authors provide a clear and well-structured explanation of the conceptual framework, making the approach easy to follow.

2. The visual illustrations in the paper are highly informative and enhance the reader’s understanding. For instance, **Figure 1** clearly defines the mixing strategy and highlights the benefits of optimizing both the proportions and the number of domains, while **Figure 2** vividly visualizes the key steps of the training process.

3. The proposed method is comprehensively validated across multiple benchmark tasks—including natural language understanding, reasoning, and multimodal datasets

### Weaknesses
1. The novelty of the proposed method is  ambiguous, and its distinction from prior work—both conceptually and technically—is not clearly articulated. The third paragraph (Lines 45–70) attempts to highlight the limitations of existing approaches as motivation but does not cite any relevant literature. For example, in **Line 48**, the statement __“Optimizing the proportions … the general predefined domains”__ lacks a citation, making it unclear whether this is an original claim or one established in prior studies. Similarly, in **Line 53**, the phrase __“Existing approaches …”__ is not supported by references. Beyond the introduction, in **Section 3.2 (Line 186)**, the statement __“Intuitively, data points … in both clusters”__ raises the same concern—whether the idea of gradient alignment is novel to this work or derived from earlier research. Overall, the paper presents several claims without sufficient clarification of what constitutes its genuine contribution versus what is grounded in prior literature.

2. The paper’s discussion of the **data mixture optimization problem** lacks comprehensiveness and is not reader-friendly for those unfamiliar with the field. For instance, the first paragraph (Lines 32–35) is intended to introduce the background but provides only an abstract definition without concrete examples or an explanation of the practical benefits of data mixing. The motivation would be clearer if the authors illustrated how data mixing influences model performance—with or without optimization—and its real-world impact. Furthermore, in **Line 43**, the term __“the optimal groups of data mixing”__ is vague. The paper does not specify the evaluation metric used to assess a data mixing strategy—whether it is training loss, test loss, or another measure such as accuracy. The rationale for using loss as the optimization criterion, instead of accuracy-based metrics, should also be clarified. Finally, the paper does not discuss whether the proposed mixing strategy addresses in-distribution versus out-of-distribution generalization challenges.

3. The claimed computational advantage of the proposed method—with only a 0.01% additional compute overhead—is not clearly explained in the Method section. In **Line 80**, the description is confusing: the phrases __“only requires an additional overhead”__ and __“cuts more than 99% FLOPs”__ appear contradictory. The paper should explicitly clarify which component constitutes the computational bottleneck, how each computation cost is estimated, and what the reported overhead quantitatively represents. Additionally, the relationship between the overhead and FLOPs reduction should be rigorously defined and empirically supported. Given that computational efficiency is a key claimed advantage of this work, the authors are encouraged to dedicate at least one subsection in the Method section to describe the computational analysis in detail, and another subsection in the Experiment section to explain the experimental setup and the procedure used for measuring and validating the computation cost.

### Questions
- Overall, could the authors provide a more explicit comparison—both conceptual and technical—between their framework and existing data mixture optimization or domain partitioning methods?

- Are the claims in Lines 48–53 (e.g., “Optimizing the proportions … predefined domains” and “Existing approaches …”) based on prior literature or newly proposed in this paper? Please provide explicit citations where applicable.

- In Section 3.2 (Line 186), is the idea of gradient alignment between clusters an original contribution, or is it adapted from prior research?

-  What evaluation metric is used to assess the quality of a data mixing strategy—training loss, test loss, or accuracy—and why was this metric chosen?

- Does the proposed mixing strategy explicitly address in-distribution and/or out-of-distribution generalization challenges? If so, how?

- How can the claim of “only an additional overhead” be reconciled with “cuts more than 99% FLOPs”? What exactly is being reduced or added computationally?

- Could the authors provide a breakdown or formula showing how computation cost is estimated for each module?

- Would the authors consider adding a dedicated subsection in the Method or Experiment section explaining how computational efficiency was measured and validated?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces R&B, a two-stage framework to optimize data mixtures. The first stage, Regroup, aims to deal with the limitation of suboptimal human categorizations based on skills . It is a one-time offline process that re-partitions the training data into fine-grained, semantically coherent clusters using k-means on data embeddings from an existing embed model. The second stage, Balance, is an online algorithm that dynamically adjusts the sampling proportions of these new domains during training. It avoids expensive re-evaluations by reusing existing last-layer training gradients, computing a Gram matrix to align the sampling distribution with a target evaluation distribution. The authors claim R&B matches or outperforms state-of-the-art data-mixing strategies with negligible computational overhead.

### Strengths
1. The paper notices a key problem in practice: the domain predefined by human may be suboptimal. 
2. The paper empirically shows that design domain weights based on regrouped domains could improve performance. 
3. The paper proposes an efficient way to adjust domain weights during training through accumulating gradients.
4. The experiments cover multiple tasks.

### Weaknesses
1. Main weakness is that this paper is too similar to DGA [Fan et al., 2024a], although the authors claim that DoGE [Fan et al., 2024b] is the most relevant one. Specifically, DGA (1) uses embeddings from an existing embed model to represent data for self-clustering; then (2) uses the inner product of domains' gradients to compute alignment across domains for computing domain weights. These two steps are almost the same as the two key steps in this paper: regroup and reweight. The main differences are (1) R&B uses the last layer's gradient while DGA uses the full gradient, and (2) R&B uses accumulated gradients during training while DGA computes the gradient on the extra validation set. I think these two differences are minor.
2. This paper doesn't fully explain the benefit of using the last layer's gradient compared with the full gradient or other layers' gradients. I mean except memory overhead, why R&B using the last layer's gradient show better performance than DGA with full gradient?
3. The paper doesn't explain well why accumulated gradients can show better performance than gradients computed on extra evaluation set compared with DGA.
4. The main weakness of accumulated gradient during training is that some domains with low weights may have few samples leading to noisy/inaccurate gradients. How to avoid such a situation?
5. In the experiments, the paper only shows validation loss but has no downstream task performance, especially for the language model, which is what people care about more and is also a standard comparison way in data mixing papers [Xie et al., 2023a, Fan et al., 2024b, Liu et al., 2024].

### Questions
1. For the data regrouped into finer-grained domains, can you really get their new domain names like 'Historical Text', 'Music Theory' as shown in Figure 1 (1)?
2. Compared with the embedding-based data mixing method [1], what's the advantages and disadvantages of using an existing embed model instead?


[1] Wanyun Xie, Francesco Tonin, and Volkan Cevher. Chameleon: A flexible data-mixing framework for language model pretraining and finetuning. ICML, 2025.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents R&B (Regroup & Balance), a two-stage framework designed to optimize data mixtures efficiently during foundation model training. It tackles two main issues in existing approaches: their dependence on broad, human-defined data domains and the heavy computational cost of reweighting those domains. In the Regroup stage, R&B clusters training data into fine-grained, semantically coherent groups using embeddings. In the Balance stage, it dynamically adjusts data mixture proportions during training by using last-layer gradients that are already computed, eliminating the need for additional evaluation passes. A Gram matrix of domain gradient similarities guides a softmax-based update of sampling weights. Experiments across language, reasoning, and multimodal benchmarks show that R&B achieves equal or superior performance compared to state-of-the-art methods while cutting reweighting computation costs by over 99%.

### Strengths
1. Faster runtime
2. Well-motivated domain redefinition
3. Experimental setup

### Weaknesses
1. Dependence on embedding quality: the entire Regrouping stage depends on the quality of the text embeddings used for clustering. It remains unclear how sensitive the framework is to the choice of this model. If the embeddings fail to capture the salient semantic features for a given task, the resulting clusters may be suboptimal, leading to poor downstream performance.
2. The reasoning datasets should be evaluated on reasoning tasks, such as MMLU, GSM8K, MATH-500, etc., while the current evaluation uses validation loss.
3. The text can be clearer. For example, the caption of Algorithm 1 states "Data selection" but, as far as I understand, you are performing domain reweighing, not filtering any data.
4. The theoretical difference with DoGE and DGA is sometimes overstated, i.e., "The key innovation of R&B lies in its use of gradient information to dynamically adjust sampling priorities", which could also apply to DoGE/DGA. 
5. Some references are not discussed, for instance [1] use domain embeddings and Gram matrix computation and [2] also performs clustering.

[1] Xie, W., Tonin, F., & Cevher, V. (2025). Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning. ICML.
[2] Ling, Zhenqing, et al. "Diversity as a reward: Fine-tuning llms on a mixture of domain-undetermined data." arXiv preprint arXiv:2502.04380 (2025).

### Questions
1. What is the purpose of Lemma 3.1? The regrouping step to me seems quite straightforward (embed -> kmeans) and clear already.
2. See weaknesses

### Soundness
2

### Presentation
2

### Contribution
1

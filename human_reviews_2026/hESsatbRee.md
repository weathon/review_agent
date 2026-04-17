# Mitigating Catastrophic Forgetting in Target Language Adaptation of LLMs via Source-Shielded Updates

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 4, 2, 2

## Abstract
Expanding the linguistic diversity of instruct large language models (LLMs) is crucial for global accessibility but is often hindered by the reliance on costly specialized target language labeled data and catastrophic forgetting during adaptation. We tackle this challenge under a realistic, low-resource constraint: adapting instruct LLMs using only unlabeled target language data. We introduce **S**ource-**S**hielded **U**pdates (**SSU**), a selective parameter update strategy that proactively preserves source knowledge. Using a small set of source data and a parameter importance scoring method, SSU identifies parameters critical to maintaining source abilities. It then applies a column-wise freezing strategy to protect these parameters before adaptation. Experiments across five typologically diverse languages and 7B and 13B models demonstrate that SSU successfully mitigates catastrophic forgetting. It reduces performance degradation on monolingual source tasks to just 3.4% (7B) and 2.8% (13B) on average, a stark contrast to the 20.2% and 22.3% from full fine-tuning. SSU also achieves target-language performance highly competitive with full fine-tuning, outperforming it on all benchmarks for 7B models and the majority for 13B models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper focuses on adapting instruct-tuned LLMs to target languages using only unlabled target language text. Standard continual pretraining on target language data often results in catastophic forgetting where the new training data erases source knowledge and significnatly affects the core chat and instruction-following capabilities. To address this issue, this paper proposes Source-Shielded Updates, a simple and effective source-focused appraoch that shields source knowledge by freezing specific columns of the weight matrices while training on target language data. The columns to be frozen are selected by using a small set of source data and a parameter importance scoring method.

The proposed approach is verified by adapting 7B and 13B OLMo2 instruct models (trained on English-dominated Common Crawl corpus) to five different target languages. Experimental results demonstrate that SSU consistently outperforms relevant baselines in terms of target-language proficiency while preserving general source-language performance.

### Strengths
The proposed approach is simple and easy to use, which I consider as strengths given its effectiveness.

Different from existing approaches that use target data-driven signals to identify which parameters are to be trained, the proposed approach is source-focused and uses source data to select parameters that are kept frozen during training. This makes sense and also works better compared to target data-driven GMT method.

Several downstream evaluation datasets have been used to demonstrate source knowledge preservation.

Ablation studies were conducted demonstrating the effect of freezing ratio on the trade-off between source knowledge retention and target knowledge acquisition.

Effectiveness of proposed SSU strategy is demonstrated using multiple importance scoring methods.

### Weaknesses
The proposed approach is compared with alternative selection straggles at 50% freezing ratio. While 50% provides a good operating point for the proposed approach (in terms of trade-off between source and target metrics), it may not be the best operating point for the alternative approaches. So, the current results do not provide a full comparison between different selective update strategies.

### Questions
Getting plots like Fig.2 for the other selective update strategies will make the comparison between different approaches more thorough.

### Soundness
3

### Presentation
4

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
The paper introduces Source-Shielded Updates (SSU), a parameter update strategy to mitigate catastrophic forgetting during target language adaptation of instruction-tuned LLMs. SSU first identify critical parameters with Wanda technique, then freeze the corresponding parameters to prevent gradient update. Experiments conducted on five low-resource languages and two model scales (7B, 13B) demonstrate that SSU reduces catastrophic forgetting by a substantial margin compared to full fine-tuning and several strong baselines.

### Strengths
The question of adapting an instruction tuned LLM to support new languages without forgetting is critical. 
The proposed method is well motivated and mathematically sound.
The empirical results are strong. The proposed method successfully reduced forgetting.

### Weaknesses
1.	The novelty is largely overclaimed and many related works and baselines are not included as they should be. First, the idea of freezing critically parameters for learned knowledge has been widely adopted. A classical CL approaches, HAT, shares this idea. More recent works such as CAT and SPG. Therefore, the claim that `However, existing paradigms are ill-suited for the specific challenge of adapting instruct models with unlabeled target language text. They either rely on random selection, offering no principled way to preserve knowledge, or on signals from the new data to guide updates (target-focused).` is not true. Additionally, many CL baselines are missing. 

2.	The evaluation mostly considers ICL ability of LLMs. It would be better to include more generation/language modeling tasks to eval the new language ability.

3.	New language typically faces tokenizer issue. For example, new language may be OOV or be over tokenized. This requires updating the dictionary. The current approach do not support this.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Source-Shielded Updates (SSU), a source-driven selective-parameter update framework for adapting instruct-tuned LLMs to underrepresented languages using only unlabeled target-language text. SSU aims to mitigate catastrophic forgetting while retaining source-language instruction-following abilities.

### Strengths
1. The writing is clear, well-structured, and easy to follow, making the paper accessible and logically organized.


2. The column-wise masking design is an elegant and technically sound insight, offering a simple yet effective structural approach to preserve model representations.


3. The source-driven importance scoring provides a principled and data-grounded alternative to random or target-data-based freezing strategies.

### Weaknesses
1. The paper lacks a comprehensive hyperparameter search for baseline methods, which may make the reported comparisons less fair or less reproducible.


2. The proposed method benefits from additional source-language data for parameter importance estimation, while baseline methods do not, introducing a potential source of unfair advantage.


3. The novelty is limited, as similar importance-based freezing or selective update methods have been explored in prior works [1,2]. 


4. The paper does not include baselines that also leverage source data, which would provide a more balanced evaluation of the benefits of source-informed adaptation.


5. The comparison omits recent state-of-the-art methods addressing catastrophic forgetting in multilingual CPT, making it difficult to precisely quantify how much SSU advances the field. 

[1] Jung, S., Ahn, H., Cha, S., & Moon, T. (2020). Continual Learning with Node-Importance based Adaptive Group Sparse Regularization.

[2] Yao, K., Gao, P., Li, L., Zhao, Y., Wang, X., Wang, W., & Zhu, J. (2024). Layer-wise Importance Matters: Less Memory for Better Performance in Parameter-efficient Fine-tuning of Large Language Models.

### Questions
Would it be possible to modify existing baselines or introduce new ones that also incorporate source data to ensure a more fair comparison?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to mitigate catastrophic forgetting in LLM adaptation by identifying and freezing a subset of the parameters during adaptation. Empirically, the proposed approach has better performance than some existing adaptation methods.

### Strengths
- Catastrophic forgetting is an important problem. The proposed approach is tested on models up to 13B and works well empirically. However the comparisons are limited to a small subset of relevant baselines and miss many important and relevant methods in the literature.

- The paper is well written.

### Weaknesses
- The baselines used in empirical comparisons are not comprehensive. The paper misses many key baselines, both in related work summary and in empirical comparisons, such as: [1, 2, 3]. It's important to see how the proposed method's performance compares with these relevant baselines. 

[1] [Lottery Ticket Adaptation: Mitigating Destructive Interference in LLMs](https://arxiv.org/abs/2406.16797)

[2] [LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation](https://arxiv.org/html/2504.07448v1)

[3] [S2FT: Efficient, Scalable and Generalizable LLM Fine-tuning by Structured Sparsity](https://arxiv.org/abs/2412.06289)

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
2

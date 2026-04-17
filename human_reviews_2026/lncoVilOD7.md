# Demystifying Hybrid Thinking: Can LLMs Truly Switch Between Think and No-Think?

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 4

## Abstract
Hybrid thinking enables LLMs to switch between reasoning and direct answering, offering a balance between efficiency and reasoning capability. Yet our experiments reveal that current hybrid thinking LLMs only achieve partial mode separation: reasoning behaviors often leak into the no-think mode. To understand and mitigate this, we analyze the factors influencing controllability and identify four that matter most: (1) larger data scale, (2) using think and no-think answers from different questions rather than the same question, (3) a moderate increase in no-think data number, and (4) a two-phase strategy that first trains reasoning ability and then applies hybrid think training. Building on these findings, we propose a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from 1085 to 585 on MATH500) and occurrences of reasoning-supportive tokens such as "wait" (from 5917 to 522 on MATH500). Our findings highlight the limitations of current hybrid thinking and point the ways for enhancing its controllability.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a practical recipe that, compared to standard training, can maintain accuracy in both modes while significantly reducing no-think output length (from 1085 to 585 on MATH500) and occurrences of reasoning-supportive tokens such as “wait” (from 5917 to 522 on MATH500).

### Strengths
Many papers analyze how LRMs can shorten "think" processes or compare the switching between "think" and "no-think" modes. To some extent, this paper fills the gap in exploring how to further reduce the length in "no-think" scenarios. Although considering words like "wait" is a very common and trivial approach—with numerous existing works already addressing this—further reducing the length in "no-think" scenarios is practically necessary.

### Weaknesses
This paper seems too simple. Only two simple mathematical benchmarks are considered. The authors should add more complicated math tests and coding benchmarking tests, to show the generalization of this method on no-think mode.

### Questions
Generalization of this method on no-think mode.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the phenomenon that large reasoning models (LRMs) continue to produce reasoning-related tokens even in non-thinking modes.

The central insight is that hybrid thinking cannot fully disentangle reasoning and non-reasoning behaviors, as the no-think mode remains influenced by think-mode data.

Based on an analysis of contributing training factors, the authors propose a practical approach to improve controllability in hybrid thinking systems. Experimental results suggest that future hybrid thinking training should deliberately allocate a larger proportion of no-think data and adopt structured strategies such as two-phase training to achieve clearer mode separation and enhanced reasoning control.

### Strengths
1. The paper provides a novel perspective on hybrid reasoning by identifying that large language models tend to generate reasoning-related tokens even in non-thinking modes, offering new insights into the behavioral dynamics of reasoning control.
2. The proposed mitigation strategies—allocating a larger proportion of no-thinking data and adopting a two-phase training scheme—present practical and conceptually sound approaches to improving controllability in large reasoning model training.

### Weaknesses
1. The experimental evaluation lacks diversity in model size and architecture. The study is conducted primarily on Qwen2.5-7B and LLaMA3.1-8B, both relatively small and dense models, which limits the generalizability of the findings to larger or sparse (e.g., MoE) architectures.
2. The selected datasets—MATH500, AIME24, MMLU-STEM, and GPQA-Diamond—are all domain-specific and heavily focused on mathematical or scientific reasoning. The absence of simpler or general-purpose datasets, such as those involving everyday dialogue or commonsense reasoning, restricts the scope of the conclusions. It remains unclear whether reasoning leakage in the non-thinking mode is inherent to the model or primarily induced by the high difficulty level of the chosen tasks.

### Questions
1. The experiments are conducted exclusively on models in the 7B–8B scale. To what extent might the observed phenomena depend on model capacity or learning ability? Would larger models exhibit similar or different patterns of reasoning leakage?
2. Could the findings be extended to Mixture-of-Experts (MoE) architectures? If so, might the modular structure of MoE models help mitigate or amplify the observed reasoning leakage?
3. The selected datasets—MATH500, AIME24, MMLU-STEM, and GPQA-Diamond—are all math- and science-oriented. Would the same phenomenon persist on more general-purpose and less complex datasets (e.g., TruthfulQA)? How consistent is the reasoning leakage behavior in non-thinking modes across tasks of varying difficulty?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the hybrid thinking of LLMs from the perspectives of thinking controllability, post-training and data mixing strategies. At first, the authors highlight the fact that the thinking behaviors also happen a lot when the think mode is switched off by the designated prompt. Then, the impacts of data scale and mixing mechanism are examined on the basis of Qwen models. After that, the authors propose a two-phase training strategy that first trains reasoning ability and then switch into the hybrid think training. The results show that the proposed training strategy enhance the thinking control.

### Strengths
1. The paper is well written and easy to follow.
2. The topic of hybrid thinking control is interesting and has a solid impact towards the development of LLMs.
3. Their results well provide with some interesting observations. The proposed two-phase training strategy show good thinking control in terms of output length.

### Weaknesses
1. The authors only choose Qwen2.5-7B to evaluate the impact of different factors. It is unknown whether these findings hold consistently across different models sizes and manufacturers.
2. Another interesting concern is that the output lenght is negatively correlated with the inference accuracy under the non-think mode. That is to say, the proposed training strategy is not that effective to maintain the inference accuracy when a better control is favored. To somehow, it is a fair thing that we need to sacrifice something for the others.  But what is point of this paper instead?
3. The authors present the same set experiments with two styles, (i.e., figure and table). But these two styles carry the same results and same observation. I believe only choosing one style for result presentation is sufficient and more space can be saved for more experiments with other model sizes and manufacturers.

### Questions
All my questions are already discussed in Weaknesses Section.

### Soundness
2

### Presentation
2

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
This paper analyzes the key factors influencing the controllability of hybrid thinking training via Supervised Fine-Tuning (SFT), with a specific focus on the tendency of Large Reasoning Models (LRMs) to persist in a non-thinking mode during the final answer generation. The authors substantiate four key findings through extensive experiments on the Qwen2.5-7B-Instruct model. While numerous SFT data processing methods exist to enhance reasoning efficiency, the experimental validation of the proposed method is limited. These limitations include non-comparable data scales, a constrained selection of base models, and a lack of baseline comparisons. Consequently, more comprehensive experimentation is required to firmly establish the method's efficacy

### Strengths
1. The paper thoroughly analyzes the key factors affecting the controllability of hybrid thinking during SFT. Extensive experiments on the math datasets with Qwen2.5-7B-Instruct model demonstrate that the efficiency of the no-think mode can be improved with larger data scale, better data ratio, non-paired data, and improved two-phase training. These findings are helpful for SFT training for efficient LRMs.

2. The new recipe effectively improves the controllability of the no-think mode, where the number of thinking steps is dramatically reduced than the original recipe on two datasets.

### Weaknesses
1. The ratio of think and no-think data is more a trade-off between reasoning accuracy and generation length according to Table 4. In the two-phase training part and Section 4, the data mixture ratios of think and no-think are 1:1 and 5:4, respectively. Therefore, the data ratio finding is not practically used to validate its effectiveness.

2. The main advantage of the proposed training method is the more controlled and efficient no-think mode. There are no obvious improvements for the more time-consuming and important thinking mode. The controllability of no-think mode can be improved with better prompts. Therefore, this method should also be compared with other SFT and training-free methods in the perspective of efficiency and effectiveness trade-off.

3. In addition, in Table 6, the four findings are not used to improve hybrid thinking over existing training settings. Only Qwen2.5-7B-Base and LLaMA-3.1-8B-Base are fine-tuned in Section 4, and the recent and powerful Qwen3 or  other base models are helpful to validate the generalization of the proposed method.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

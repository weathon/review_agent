# Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons

- Decision: Reject
- Scores: 3, 6, 5

## Abstract
Large language models (LLMs) excel in various capabilities but pose safety risks such as generating harmful content and misinformation, even after safety alignment. In this paper, we explore the inner mechanisms of safety alignment through the lens of mechanistic interpretability, focusing on identifying and analyzing *safety neurons* within LLMs that are responsible for safety behaviors. We propose *inference-time activation contrasting* to locate these neurons and *dynamic activation patching* to evaluate their causal effects on model safety. Experiments on multiple prevalent LLMs demonstrate that we can consistently identify about $5$% safety neurons, and by only patching their activations we can restore over $90$% of the safety performance across various red-teaming benchmarks without influencing general ability. The finding of safety neurons also helps explain the ''alignment tax'' phenomenon by revealing that the key neurons for model safety and helpfulness significantly overlap, yet they require different activation patterns for the same neurons. Furthermore, we demonstrate an application of our findings in safeguarding LLMs by detecting unsafe outputs before generation.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Focusing on the safety mechanism of LLMs, this paper proposes (1) inference-time activation contrasting, to locate safety neurons, and (2) dynamic activation patching, to evaluate their causal effects on model safety. The key observation is that only a few (5%) neurons contribute to the safety of the model. This paper also proposes applications of the observations.

### Strengths
1. Understanding the safety mechanism of LLMs is a crucial research problem.
1. This paper focuses on various aspects of the proposed interpretability methods, including empirical observation on neurons, transferability, and potential application, making the contribution a comprehensive framework.

### Weaknesses
1. The presentation of this paper can be substantially improved. Many terms are not well explained in the paper, e.g. cost scores in Table 3, $(IA)^3$ in Section 4.1
1. The observation that a few safety neurons contribute to the safety of LLMs has already been spotted in some related work, but they are not cited and discussed.
  - On Prompt-Driven Safeguarding for Large Language Models. ICML 2024
  - Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications. ICML 2024
3. It seems that the 3 LLMs used are already aligned for safety (at least, to a certain degree) before they are released. What is the alignment in 4.1 here?
4. In my opinion, it would be necessary to include some advanced jailbreaking attacks for evaluation (both for the main observation and the application), since current LLMs can easily refuse to answer vanilla harmful questions.
5. Though evaluated 3 models, I still think the model scope is quite limited, e.g. all 3 models are in 7b size, but can the conclusion generalize to larger models?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
## Summary

The authors propose methods to identify "safety neurons" within large language models (LLMs) that are responsible for safety behaviors. They introduce "inference-time activation contrasting" to pinpoint neurons active in aligned models but inactive in unaligned ones, and "dynamic activation patching" to assess the causal impact of these neurons on safety. These findings suggest a pathway toward more controlled and robust alignment of LLMs with human values and safety requirements.

### Strengths
## Strengths
* The topic of LLM safety is highly relevant and timely.
* The paper makes solid contributions by:
    * Identifying safety neurons in three open-source LLMs.
    * Proposing an effective safeguard application.

### Weaknesses
## Weaknesses
* Novelty Concerns: The novelty of the proposed approach is unclear. Previous studies have investigated critical neurons within LLMs. The authors should clarify how their methods differ from or improve upon existing approaches.
* Limited Discussion: The paper lacks a sufficient discussion on how the proposed methods relate to existing representation engineering techniques (https://arxiv.org/pdf/2310.01405). A deeper comparison would help contextualize their contributions.

### Questions
## Questions:
* How does the proposed approach for identifying "safety neurons" differ from prior methods that target other types of critical neurons in LLMs?
* Can the "dynamic activation patching" method be generalized to other alignment applications, such as aligning models with values beyond safety (e.g., fairness)?
* Do you find any mechanistic insight? For example, did you observe specific patterns among the "safety neurons" related to particular types of safety risks, such as misinformation or toxicity?
* For safeguard applications, what is the overhead of your proposed approach?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a novel methodology for identifying specific MLP neurons that contribute to safety alignment in large language models. The authors present two complementary techniques: inference-time activation contrasting, which identifies neurons by comparing their activation patterns between pre- and post-safety-finetuned model checkpoints; and dynamic activation patching, which employs causal interventions to quantify the extent to which the identified neurons are responsible for the model's safety behaviors.

The authors show that inference-time activation contrasting can robustly identify neurons that are causally responsible for safety behavior (as measured by dynamic activation patching), on a wide range of benchmarks.

Through extensive experimentation, the authors demonstrate several key findings. When safety neurons are patched into instruction-trained models that were finetuned for helpfulness, it increases safety but reduces helpfulness. The reverse effect is also observed, suggesting that safety and helpfulness behaviors rely on similar neural mechanisms - providing mechanistic evidence for the alignment tax hypothesis. Additionally, the identified safety neurons can be used for harmful prompt classification to prevent unsafe model outputs.

### Strengths
- The authors tested their method on a variety of model families (LLaMa2, Mistral, and Gemma), and used a variety of different datasets and cost models to evaluate safety.  This helps increase confidence that the neurons are actually responsible for general safety behavior, and not just patterns present in a particular dataset/grading scheme.

- The authors show that the projections of their safety neurons onto the unembedding of the model, result in different tokens than toxicity neurons identified in previous work [1].  This distinction highlights that more complex instruction-tuned models have more nuanced mechanisms for dealing with safety than simply downweighting neurons that respond with toxic content.  

[1] Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, & Rada Mihalcea. (2024). A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity.

### Weaknesses
- The primary contribution of this work lacks sufficient novelty in the context of existing research. Prior work has already demonstrated successful localization of safety-relevant components in language models across multiple architectural levels, including neurons [1], parameters [2], residual activations [3], attention heads [4] [5], and layers [6]. While the authors occasionally reference some of these works throughout the paper, they fail to provide a comprehensive discussion of this existing research in either the related work section or the discussion.


- The authors fail to adequately justify their focus on MLP neurons as the optimal level of abstraction for localizing safety behavior in language models. While they concentrate exclusively on neurons, prior work has demonstrated that safety behaviors emerge across multiple architectural components, particularly in attention heads and residual stream activations. The decision to analyze only neurons, while excluding these other important components, requires stronger theoretical or empirical justification. This limitation is particularly notable given that existing research has specifically identified attention heads as crucial contributors to refusal behavior [4].

- The paper’s main contribution beyond identifying safety neurons is showing that helpfulness and safety training utilize similar mechanisms, which accounts for the “alignment tax” seen during safety training. However, the evidence provided in favor of this hypothesis is limited. The evidence can also be explained by dynamic activation patching not being a very good way of transferring specific mechanisms between different checkpoints.  The authors should also look at models finetuned on both helpful and harmful data at the same time (HHH trained model), and test whether safety and helpful neurons still conflict.

- The classification results in Section 6 are very misleading.  The authors suggest that safety neurons show promise in assisting with harmfulness classification.  However, the results in Appendix E suggest that safety neurons aren’t that much more useful for classifying harmfulness compared to random neurons (with random neurons being better when using 1500 neurons).  This suggests that the method does not actually localize safety neurons, or that localization is not very useful for probing for harmfulness.  Also, if the authors are going to claim that safety neurons are useful for building defenses that improve safety, they should compare it against similar setups such as in [3].

[1] Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, & Rada Mihalcea. (2024). A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity. 

[2]  Boyi Wei, Kaixuan Huang, Yangsibo Huang, Tinghao Xie, Xiangyu Qi, Mengzhou Xia, Prateek Mittal, Mengdi Wang, & Peter Henderson. (2024). Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications.

[3] Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, & Dan Hendrycks. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.

[4] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, & Neel Nanda. (2024). Refusal in Language Models Is Mediated by a Single Direction.

[5] Zhenhong Zhou, Haiyang Yu, Xinghua Zhang, Rongwu Xu, Fei Huang, Kun Wang, Yang Liu, Junfeng Fang, & Yongbin Li. (2024). On the Role of Attention Heads in Large Language Model Safety.

[6] Shen Li, Liuyi Yao, Lan Zhang, & Yaliang Li. (2024). Safety Layers in Aligned Large Language Models: The Key to LLM Security.

### Questions
- What motivated your decision to focus exclusively on MLP neurons, given that prior work has shown attention heads are crucial for refusal and safety behavior?

- Have you considered validating your hypothesis about helpfulness and safety mechanism overlap using models simultaneously trained on both helpful and harmful data?

- Are the probing results, primarily a negative result?  If so, the section should be edited to clarify that.

### Soundness
2

### Presentation
3

### Contribution
2

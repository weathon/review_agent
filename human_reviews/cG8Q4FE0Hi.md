# RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 6

## Abstract
Large language Models (LLMs) have achieved promising performance on arithmetic reasoning tasks by incorporating step-by-step chain-of-thought (CoT) prompting. However, LLMs face challenges in maintaining factual consistency during reasoning, exhibiting tendencies to condition overlooking, question misinterpretation, and condition hallucination over given problems. Existing methods use coarse-grained feedback (e.g., whether the answer is correct) to improve factual consistency. In this work, we propose RCoT (Reversing Chain-of-Thought), a novel method to improve LLMs’ reasoning abilities by automatically detecting and rectifying factual inconsistency in LLMs’ generated solutions. To detect factual inconsistency, RCoT first asks LLMs to reconstruct the problem based on generated solutions. Then fine-grained comparisons between the original problem and the reconstructed problem expose the factual inconsistency in the original solutions. To rectify the solution, RCoT formulates detected factual inconsistency into fine-grained feedback to guide LLMs in revising solutions. Experimental results demonstrate improvements of RCoT over standard CoT, Self-Consistency and Self-Refine across seven arithmetic datasets. Moreover, we find that manually written fine-grained feedback can dramatically improve LLMs’ reasoning abilities (e.g., ChatGPT reaches 94.6% accuracy on GSM8K), encouraging the community to further explore the fine-grained feedback generation methods.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper aims to improve LLM’s reasoning abilities and address the challenges of overlooking, question misinterpretation and condition hallucination in LLMs’ generated solutions. It proposes RCoT to detect and rectify such factual inconsistency through four steps, including reconstruction, decomposition, comparison, and revision. The experiments are conducted on randomly sampled sub-sets of seven arithmetic datasets.

### Strengths
- The motivation is clear and the analysis of challenges is reasonable.
- The performance of the proposed RCoT is demonstrated to be superior to the standard baselines.

### Weaknesses
- The experiments are only conducted on randomly sampled sub-sets of the test sets, which may raise concerns about the convincingness of the results. The experiment results do not allow for a direct apple-to-apple comparison with the reported results in other papers, such as those related to Self-consistency.
- The experiments on other reasoning tasks, such as commonsense reasoning and symbolic reasoning, are absent.
- The performance improvement is not significant compared to the self-consistency (84.5 v.s 83.5).  In addition, did the paper's testing of the Self-consistency algorithm use 30 paths? In Self-consistency, the typical number of paths used is (1, 5, 10, 20, 40). Why were 30 paths chosen? If the reason is to make comparable comparisons based on average tokens, it would be appropriate to report the performance and average tokens under different numbers of paths.
- There is a lack of in-depth analysis and evaluation beyond overall performance, such as the absence of assessment regarding improvements (Quantitative or user-study-based evaluations) in the three areas of overlooking, question misinterpretation, and condition hallucination. Table 5 only evaluates on 45 cases.
- The method is somewhat incremental. The decomposition, comparison, and revision components are not new in the context of CoT. While reconstruction is used in many fields, its application within CoT is new and appears to be the main technical contribution. However, the overall framework of RCoT is incremental and complex.
- Minor suggestions about the presentation:
    - On page 1, this paper introduces the condition of  "2 days away" in Figure 1 is mistakenly overlooked. However, there are no "2 days away" in Figure 1.
    - Can the three different examples in the Introduction be unified?
    - The font size in Table 1 is too small.
    - The order of citing figures and charts is mixed up. For example, Table 4 is cited before Tables 2 and 3, but it is located below them in the paper.

### Questions
See Weaknesses

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a novel approach RCoT to identify and rectify factual inconsistencies in the outputs of large language models (LLMs). The approach works by first prompting the LLM to reconstruct the question based on its answers, and then prompting the LLM to determine whether the reconstructed question is identical to the original question in terms of the conditions derived. If there are discrepancies between the conditions, the finer differences are used to rectify the LLM to provide a more accurate and consistent answer. Experiments show that the proposed RCoT outperform baselines in seven arithmetic datasets.

### Strengths
The general thoughts of the problem are interesting and novel -- proof by contradiction -- use LLM to prove things by contradiction. From my understanding, LLMs are used in the following different scenarios: (1) reconstructing the problem; (2) Listing the conditions of the original and reconstructed problems; (3) determining whether there are hallucinated and overlooked conditions; (4) Determining whether the reconstructed problem is identical to the original one. (5) rectifying the results based on the finer feedback summarized from (3). The above uses of the LLM are interesting and each worth individual study about the effects.

### Weaknesses
Although the general thoughts of the problem are interesting and novel, the paper itself has many obvious flaws.

First, the prompting method is overused. The paper does not establish causal connections between the different prompting stages. For example, to determine whether the reconstructed problem is identical to the original one (Figure 20, “question comparison”), the method does not consider the prompted results from “problem decomposition” and “condition comparison”. Additionally, to rectify the solution, the model does not take the results from “question comparison” into account.

Second, upon reviewing the examples  (Figure 20 "I apologize for my mistake...". ), I believe that the authors are exploiting the "dialog system" nature of the LLM interfaces. The dialog system introduces an extra conditioning on the chat history, which means that the actual prompts listed in the paper are all conditioned on previous prompts that have been used. As a result, I believe that the experiments are flawed. In contrast, the methods such as CoT (Wei et al., 2023), Active Prompt (Diao et al., 2023), and Self-Consistency (Wang et al., 2023) are stateless, meaning that they only involve a single interaction between a human and the LLM interface. I recommend that the authors learn from CoT which models P(answer|question, reasoning chain of other examples) to better formulate their condition dependences.

Third, the gain of providing the reason seems to be moderate. From table 2, "judgement" (Figure 20 "question comparison") should be the key factor while "reason" (Figure 20, “problem decomposition” and “condition comparison”) seems to be less important. From Table 4, the proposed complicated RCoT seems to be worse than Self-Consistency (Wang et al., 2023).

### Questions
I expect the authors to justify their choice of interactively prompting the large language models. Currently, I felt they only proved it kind of works but did not explain the reason. However, I felt the comparison are not fair and the results are not easy to reproduce.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper aims to tackle challenges like condition overlooking, question misinterpretation, and condition hallucination that LLMs meet in arithmetic reasoning benchmarks. On top of chain-of-thought (CoT) prompting, this paper proposes RCoT, which asks the LLM to rewrite the problem, compare it to the original one, and identify fine-grained differences in conditions and questions, thus finding mistakes and revising the original answer. Experiments show consistent improvements across benchmarks and LLMs, verifying effectiveness of the proposed method.

### Strengths
*Originality*: The core idea of this paper is novel and original.

*Quality*: The method is well-motivated and extensively evaluated.

*Clarity*: The delivery is very clear and easy to understand, I did not find issues in understanding.

*Reproducibility*: Code is provided to encourage reproducibility.

*Significance*: This work touches a major issue in LLMs, which is of much research significance.

### Weaknesses
- One drawback of this work could be its complexity, as illustrated in the diagram and verified by token counts. I understand it is comparable to some previous works, but optimizing its complexity is still an important aspect.
- (minor) The comparisons in tab. 1 is not very clear, eg, the results marked in green are not straightforward to understand except by reading the captions.
- (minor) Venues are missing from multiple references.

### Questions
Suggestion: it might be better to call "reconstruction" as "rewriting" or "paraphrasing".
Disclaimer: since I am not very familiar with related literature, my current rating is relatively conservative, and I'll reconsider it after reading opinions from other reviewers.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

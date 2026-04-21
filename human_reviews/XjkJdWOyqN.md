# Are Large Language Models Really Robust to Word-Level Perturbations?

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3, 3

## Abstract
The swift advancement in the scales and capabilities of Large Language Models (LLMs) positions them as promising tools for a variety of downstream tasks. In addition to the pursuit of better performance and the avoidance of violent feedback on a certain prompt, to ensure the responsibility of the LLM, much attention is drawn to the robustness of LLMs. However, existing evaluation methods mostly rely on traditional question answering datasets with predefined supervised labels, which do not align with the superior generation capabilities of contemporary LLMs. To address this issue, we propose a novel rational evaluation approach that leverages pre-trained reward models as diagnostic tools to evaluate the longer conversation generated from more challenging open questions by LLMs, which we refer to as the  $R$eward Model for $R$easonable  $R$obustness $Eval$uation ($TREvaL$).  Longer conversations manifest the comprehensive grasp of language models in terms of their proficiency in understanding questions, a capability not entirely encompassed by individual words or letters, which may exhibit oversimplification and inherent biases. Our extensive empirical experiments demonstrate that TREvaL provides an innovative method for evaluating the robustness of LLMs. Furthermore, our results demonstrate that LLMs frequently exhibit vulnerability to word-level perturbations that are commonplace in daily language usage. Notably, we are surprised to discover that robustness tends to decrease as fine-tuning (SFT and RLHF) is conducted.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes to evaluate LLM robustness in terms of word-level perturbation. Specifically, different types of pertubations (such as misspelling) at different levels are applied on a question answering dataset (NQ) and a reward model is used to measure the performance degeneration after perturbation.

### Strengths
- The paper is relatively well written and easy to follow

- The paper studies an interesting research question (robustness of LLMs regarding word-level perturbation) with analysis on the natural questions dataset

### Weaknesses
- [major] The paper only uses Natural Questions (NQ) as the benchmark for evaluation of word-level perturbation. Why is NQ a good benchmark for LLM robustness? More benchmarks will make the results and analysis more convincing.

- [major] The reward model is very sensitive and biased to the model that uses it for training (the Beaver model in the paper). In my experience it's often not suitable for scoring/evaluating other models. However, there's little discussion on justifying the RM choices and the reliability of using the RM for evaluation. Even for evaluating the Beaver model itself, there remains this question whether the Beaver RM is reliable for assessing performance drop in NQ. Is there any evidence of this? E.g., is the Beaver reward model fine-tuned on NQ?

- [minor] Level-3 perturbation seems too much to reflect real-world scenarios. Does it really make much sense to evaluate cases where the input is not human-readable?

### Questions
-  see weakness

### Soundness
1 poor

### Presentation
3 good

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
This work examines the robustness of Language Learning Models (LLMs) regarding their helpfulness and harmlessness, specifically focusing on the impact of input word perturbations. To this end, the authors leverage the existing Beaver-7B reward models and the question-answer dataset to create an evaluation benchmark. The experimental result exposes LLMs' deficiency in input perturbation, and llama2-7b performs the best among other LLMs.

### Strengths
1. This work proposed to use the reward model to measure the robustness gap of several llama-based models. Experimental result shows that llama2-7b is the most robust model in terms of helpfulness.

2. Further analysis reveals a decline in model helpfulness robustness throughout the fine-tuning process, which is an interesting finding for the community.

### Weaknesses
1. Evaluation is unfair. Experimental result shows that llama2-7b is the most robust model for the word perturbation. However, since the Reward Model Beaver-7B is induced from the llama2-7b model, it is unclear if it favors the system output of llama2-7b-chats. An OUT-OF-BOX reward model needs to verify this.

2. Some perturbation types are not practical in the actual use cases. For example, level 2 and level 3 of Misspelling, as well as level 3 of Swapping, are rare cases in practice, making this study unrealistic in a sense.

3. The evaluation scope is pretty limited and possibly biased to llama. All the models experimented with in this work are llama-based. Other open-source models, such as falcon-40b and dolly-12b, need to be tested to demonstrate the robustness of the proposed evaluation scheme.

### Questions
1. Why did you decide to focus on these three perturbation types? Some levels of perturbations such as "wuhitatf isop the cmemaningc komf veruonicla ipn english?" looks like messy code.
2. Did TREvaL work for other non-llama models?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposed Reward Model for Reasonable Robustness Evaluation (TREvaL) to evaluate the robustness of LLMs. In particular, the authors selected 1k open questions from Natural Questions datasets (Kwiatkowski et al., 2019), add three types of word-level perturbations to them and induce the language models to generate extensive responses. They then sent the clean and affected conversations to a reward model and calculate their distinguish drop rates as an identification of robustness. The empirical experiments demonstrated that TREvaL provides an innovative method for evaluating the robustness of LLMs. Notably, they found that robustness tends to decrease as fine-tuning
(SFT and RLHF) is conducted.

### Strengths
- The paper studied an important problem -- the perturbation robustness of LLMs.
- The paper is well written.

### Weaknesses
The experiments should be conducted in a more rigorous way:

- The authors used a reward model to measure the robustness of the LLMs in the paper. However, I have several concerns about the experimental settings: 1) only one kind of reward models are used to evaluate the performance, which makes the conclusions less convincing. 2) there is no rigorous human study shows that the drop in reward model score actually indicates a drop of the actual quality (although I think it is likely to be the case but it is also possible that the reward model relies on other correlations). A followup question is the degree to which the reward model's assessments align with human judgments. 3) the lack of an appropriate baseline is a significant omission: if a perturbation makes the original question incomprehensible, it is unreasonable to expect the model to perform correctly. A suitable baseline might involve a comparative task where humans conduct the same task as the model to observe the performance discrepancy. 

- The authors claimed that TREvaL is effective in measuring the robustness of the LLMs. To me, this is a over-claiming as only one task, open question answering, is included, which again weakens the main claim made in the paper. It would be more convincing to show the model performance across a wider spectrum of tasks.

### Questions
- I don't quite understand why harmfulness is also used as a metric for evaluation. I don't think the question in NQ will lead the model to output harmful answers. Can you give some examples of perturbation leading to harmful generations?

- Is Co the lower the better? In the appendix A.1, I saw Co increases but DR is computed as positive.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The issue of robustness in neural networks has gained paramount importance over recent years. With the surge in performance of Large Language Models (LLMs) across various domains, the question of their robustness becomes increasingly pertinent. In this work, existing datasets for robustness evaluation are classified into two categories: closed-question tasks such as GLUE and open-ended tasks like TriviaQA. However, with the evolving generative capabilities of LLMs, there is a growing concern that these tasks may no longer serve as appropriate benchmarks. To address this gap, this work introduces a GPT-based evaluation framework, named the "Reward Model for Reasonable Robustness Evaluation" or TREvaL. This framework is designed to assess the robustness of LLMs in the context of open questions. Instead of focusing purely on accuracy or similarity, TREvaL emphasizes the correlation between the content generated by the model and the selected open prompts. This is achieved by leveraging trained reward models to serve as evaluative judges. The experimental setup encompasses two foundational LLM families, Beaver and Llama2, spanning various model sizes. The primary adversarial attack methods utilized are word-level perturbations, encompassing strategies like word swapping, synonym substitution, and common misspellings. The findings underscore that the LLMs examined in this study exhibit a discernible gap in robustness. Also, the impact of parameters on harmlessness robustness seems to be slighter. Notably, the fine-tuning process appears to compromise the model's robustness. To further substantiate this observation, the paper presents loss landscape visualizations for different model stages, reinforcing the central findings.

### Strengths
(1) The paper is well-motivated and aims to rectify prevailing limitations in current benchmarks and evaluation protocols for LLM robustness. Considering the recent advancements in LLMs, research focusing on robustness evaluation stands to benefit both the academic community and practical applications greatly.

(2) The paper introduces a new robustness evaluation framework tailored specifically for LLMs. By emphasizing open questions and leveraging trained reward models, it intends to address and potentially overcome the limitations inherent in existing benchmarks.

(3) The experiments encompass three types of word-level perturbations and evaluate two series of open-source LLMs. By assessing robustness across different stages and model sizes and providing a visual representation through the loss landscape, the authors offer a comprehensive and holistic evaluation.

(4) The provided figures, tables, and mode details in the appendix effectively aid in clarifying the main ideas of this work, making them easily understandable.

### Weaknesses
(1) There seems to be a notable absence of a side-by-side comparison between TREvaL and existing benchmarks. Although some limitations in the existing benchmarks are mentioned briefly, it is not clear without such a comparison whether the proposed framework can address these limitations. Establishing a direct contrast would solidify the claims regarding the superiority or distinctiveness of TREvaL in the domain of LLM robustness evaluation.

(2) Following this point, the paper could benefit from a more comprehensive literature review, especially considering the plethora of recent studies centered on the robustness of LLMs [1]. The connections and distinctions between this work and prior research in the field are not thoroughly explored, which might leave readers without a complete contextual understanding of the contributions in the broader academic landscape.

(3) TREvaL incorporates trained reward models in the evaluation process, yet there is no explicit motivation or justification for this choice. A more detailed exposition of their role, significance, and advantages over other potential evaluative metrics would enhance the clarity and persuasiveness of TREvaL.

(4) The technical innovations appear somewhat incremental. For example, the reliance on just three basic types of word-level perturbations feels somewhat rudimentary, given the wide spectrum of possible textual adversarial attacks. While the focus on evaluating robustness in the context of open questions and across different fine-tuning stages is commendable, other aspects of the paper do not seem to introduce significantly new perspectives or methods to the field.

[1] Revisiting Out-of-distribution Robustness in NLP: Benchmark, Analysis, and LLMs Evaluations

### Questions
(1) Could you provide more insights comparing TREvaL directly with existing benchmarks? Otherwise, how can we confirm that the proposed framework addresses the limitations of current benchmarks more effectively?

(2) What motivated the choice to integrate a reward model within the evaluation framework? An immediate concern is whether the reward model itself is susceptible to the same perturbations, potentially compromising the integrity of the evaluation.

(3) Several definitions and statements appear to stand without clear foundational support. For instance, is robustness specifically defined as "the rate of performance decline under potential perturbations"? Is it accurate to categorize benchmarks into three distinct classes, as suggested? Furthermore, the definitions of terms like "helpfulness" and "harmlessness" robustness remain unclear. Also, how widespread is the acceptance of using a loss landscape as a tool for characterizing robustness?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

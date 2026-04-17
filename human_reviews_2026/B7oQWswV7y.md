# Automatic Instance Selection with Genetic Updating for Few-shot LLM Jailbreak

- Decision: Reject
- Scores: 8, 4, 8, 6

## Abstract
This paper studies the problem of few-shot large language model (LLM) jailbreak, which aims to trigger unsafe outputs of LLMs using only a handful of adversarial examples. However, the effectiveness of the current few-shot jailbreak attacks is limited by the challenge of systematically selecting the most potent instances, with existing methods often resorting to inefficient manual or random selection. In this paper, we propose a novel approach named Automatic Instance Selection with Genetic Updating (ACCEPT) for few-shot LLM jailbreak. The core of our ACCEPT is to utilize textual gradient and fitness scores to guide the optimization process automatically. In particular, our ACCEPT designs a loss objective prioritizing successful jailbreaks, which can further guide the selection of instances via textual gradient. Furthermore, we construct a pool with meaningless marks, and consider the injection operators as chromosomes following the genetic algorithm. A fitness function is then defined in jailbreak scenarios, which helps the iterations across generations for proper prompts. Extensive experiments across several benchmark datasets can validate the effectiveness of the proposed ACCEPT in comparison with extensive baselines.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a framework named ACCEPT for few-shot LLM jailbreaking. The method leverages textual gradients to automatically select the most effective instances and uses a genetic algorithm to enhance the harmful prompt through non-semantic perturbations. Experiments show that the proposed approach achieves superior results on both open-source and closed-source models.

### Strengths
1. The approach is highly innovative and logically sound; the concept of leveraging an LLM to generate "textual gradients" is particularly novel.
2. The experimental evaluation is thorough, and the accompanying analysis is detailed.
3. All figures and tables are aesthetically pleasing and easy to understand.

### Weaknesses
1. While the paper's primary focus is on offensive jailbreak techniques, the manuscript would be strengthened by briefly discussing the defensive implications of this work. For instance, adding a short elaboration in the conclusion on how these findings can inform the development of more robust defense mechanisms would be a valuable addition.
2. It would benefit from more specific details in the methodology sections. In particular, the descriptions of the TextGrad in Section 3.2 and the GA in Section 3.3 could be expanded with more concrete implementation details to improve clarity.

### Questions
1. In Section 2.1, the distinction between the two methods, "Adversarial Prompting and In-Context Learning," could be further detailed.
2. When the harmful request is combined with the selected instances for a jailbreak attack, how is it handled if the attack is not effective?
3. How are the emojis inserted into the harmful request, specifically the ones in Figure 7, selected? Inappropriate emojis could have a negative effect.
4. What does "fitness" mean in Figure 3?

### Soundness
4

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
5

### Summary
This paper addresses the in-context learning-based few-shot jailbreak problem and proposes a unified framework named ACCEPT. The framework introduces two complementary components. For instance selection, ACCEPT employs a TextGrad mechanism that reformulates the discrete selection of few-shot examples into an optimization process guided by textual feedback from LLMs. For prompt design, it integrates a genetic algorithm to search for effective injection strategies, enhancing the stealthiness and adaptability of jailbreak prompts. Comprehensive experiments conducted on multiple open-source and closed-source language models, across various datasets, demonstrate that ACCEPT achieves SOTA attack performance, significantly surpassing existing jailbreak baselines.

### Strengths
1. ACCEPT tackles the few-shot jailbreak problem, which is both practically significant and highly relevant to the security of current LLM systems.
2. The use of a genetic algorithm for prompt injection provides a heuristic yet interpretable optimization process. This design allows researchers to better understand how perturbations influence jailbreak success, improving both transparency and controllability.
3. The TextGrad mechanism leverages LLM feedback to guide instance selection, transforming a discrete combinatorial problem into a continuous optimization-like process.
4. Extensive experiments across multiple open-source and closed-source models show consistent and substantial performance improvements, validating the effectiveness and robustness of the proposed framework.

### Weaknesses
1. Although the TextGrad mechanism achieves promising optimization results through LLM-based feedback, the paper lacks a solid theoretical justification. Its convergence properties and robustness are not formally analyzed or discussed.
2. The study primarily relies on the ASR-GPT metric, using an LLM to judge attack success. This single-metric setup may introduce bias from the evaluator model itself.
3. The proposed framework integrates multiple heuristic search processes (e.g., the genetic algorithm and iterative LLM optimization), which are likely to incur significant computational overhead. However, the paper provides no discussion or analysis of efficiency or resource consumption.

### Questions
Please refer to my comments on weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces ACCEPT, an automated attack framework designed to address the challenge of instance selection in few-shot LLM jailbreak attacks. It achieves a superior jailbreak success rate through the synergistic cooperation of two components: a semantic-level module that selects optimal instances and a non-semantic-level module that enhances the attack prompt.

### Strengths
1. The overall framework is novel, enhancing both the selected instances and the malicious request from two synergistic perspectives: semantic and non-semantic.
2. The paper is well-written with a clear and coherent logical flow.
3. The experiments are extensive and sound.

### Weaknesses
1. The implementation of TextGrad relies on an LLM, so the quality of the LLM determines the final jailbreak success rate. It is necessary to elaborate on this part in the conclusion or appendix.
2. The paper should provide a clearer case study to illustrate how the "problem diagnosis" and "contribution assessment" parts of the textual gradient are generated and how they guide instance selection.

### Questions
1. In "Gradient-Guided Candidate Sampling", why select a candidate subset first instead of directly selecting from the entire sample pool C?
2. What does the single-point in the crossover operation mean?
3. In Algorithm 1, should the final output be E_{best}? I think that E_{T_{grad}} is not necessarily the best.
4. In the "Fitness Function," when calculating the probability for refusal terms, is it the sum of the refusal probabilities for each word, or is the refusal probability calculated for the entire sentence?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper devises a few-shot jailbreak method called ACCEPT. The goal is to provide the target LLM with some instances together with a harmful question and leverage the in-context learning to make it output a harmful response. The method starts with a set of candidate instances and utilizes an auxiliary LLM to provide "textual gradient" to narrow down the candidate pool and select new instances. To reduce the refusal rate, it also includes a genetic algorithm to inject special non-semantic tokens. The instance selection and the genetic algorithm run alternately. Experiments use two datasets, AdvBench and HarmBench, and six models against five baselines. Results show the proposed method can achieve best ASRs and is more robust against defenses. Ablation studies show that both the instance selection and the genetic algorithm are necessary.

### Strengths
1. It studies a critical issue of few-short jailbreaks.

2. The idea to combine the textual gradient and the genetic algorithm is interesting.

3. The experimental results show its effectiveness and robustness.

### Weaknesses
1. It's unclear how the fitness score is computed. Although the paper lists some refuse phrases in the appendix, it's not clear how they are used to compute the probability. For example, does it generate a set of samples and compute the ratio of the matched strings, or use some token distribution? If the token distribution is used, how can we attack the closed-source model or service that doesn't return the distribution?

2. It's unclear how the quality of the candidate pool will affect the performance. Because TextGrad only selects the instances from the pool without creating any new instances, and only one candidate pool was used. For example, if the algorithm starts with some random pool instead of the malicious ManyHarm, it may not succeed.

3. It's unclear what the time cost is or the queries needed to conduct this attack. Similarly, it's suggested to compare the efficiency with baselines.

4. Regarding the robustness, since the injected non-semantic tokens include some special tokens, paraphrasing or input filtering may easily remove the adversarial effect. Also, it's unclear what the baseline is in Figure 2.

### Questions
1. How is the fitness score computed?

2. Will a different candidate pool affect the performance significantly?

3. What is the time cost and the query budget?

### Soundness
2

### Presentation
3

### Contribution
2

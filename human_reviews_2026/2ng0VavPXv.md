# Training-free LLM Verification via Recycling Few-shot Examples

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges.  Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (ReFeri). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, ReFeri evaluates the generated outputs by combining two different scores, designed motivated by Bayes’ rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.5\%-through effective response selection, without additional training.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents ReFeri, a training-free framework for selecting the best output from multiple LLM generations by repurposing few-shot examples for verification, using a Bayes' rule-derived scoring function. While the idea of leveraging Bayes’ rule to combine forward and backward likelihoods is conceptually elegant, the claimed novelty is overstated.

### Strengths
1. The core idea of "recycling" few-shot examples for verification, not just generation, is  a novel perspective.

2. The experimental design is a major strength. The use of seven diverse benchmarks and three LLMs provides compelling evidence for the method's generalizability.

### Weaknesses
1. The proposed method is a re-interpretation of likelihood-based reranking and LLM-as-judge techniques, both of which already use internal model probabilities or few-shot conditioning. The paper fails to clearly delineate how ReFeri fundamentally differs from prior “Best-of-N” and likelihood-based methods (e.g., CoT-WP, PRM, ORM, or self-consistency with verification). The novelty is slightly diminished by the existence of methods like CoT-WP, which already use the forward score. The primary innovation then becomes the introduction of the backward score. 

2. The connection to Bayes' rule feels somewhat post-hoc and tenuous. Eq. (5) is a mathematical tautology, but the paper does not sufficiently justify why decomposing and estimating these two terms separately is more effective than directly modeling P(y_k), especially given the admitted "discrepancies" due to model limitations. The theoretical motivation lacks a solid, empirical or theoretical foundation for why it works better in practice.

3. The reported 1–2% improvement over the strongest baselines is statistically weak, especially given that the comparison baselines are limited to prompt-based methods. No evidence is provided that ReFeri generalizes beyond the selected reasoning datasets. The improvements reported (Table 1) lack standard deviation or confidence intervals. Given the stochasticity of LLM sampling, it’s unclear whether differences of 1–2% are meaningful.

4.The intuition behind the backward score is explained at a high level ("how well the candidate explains the few-shot examples"), but its mechanistic interpretation is vague. Does a high backward score imply the candidate response is a good "archetype" or "rule" that generalizes to the few-shot examples? 

5. The "lightweight approximation" is a major compromise. The decision to use only the single most relevant example for the backward score (Eq. 9-10) for computational reasons is understandable but undermines the core Bayesian formulation. The method is no longer a true application of Eq. (5) but a heuristic approximation. Why this approximation does not collapse the entire theoretical framework?

6. Although the method is claimed to be task-agnostic, all experiments are conducted on structured QA tasks. Open-ended tasks (e.g., summarization, dialogue) are not explored, where “verification” is more challenging.

7. The estimation model is ambiguity. The method relies heavily on a “likelihood estimation model,” yet it is never discussed how this model differs from the generation LLM, how temperature affects estimation, or how token-level log-probabilities are normalized across different lengths.

8. Although the method is claimed to be “training-free,” it involves multiple LLM calls per candidate and per example, which can be expensive. No clear runtime or scaling comparison with existing test-time reranking methods is given.

### Questions
1. Does a high backward score imply the candidate response is a good "archetype" or "rule" that generalizes to the few-shot examples? 

2. Why the lightweight approximation does not collapse the entire theoretical framework?

### Soundness
1

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
The paper presents a new training-free framework applicable to the verification of LLM outputs. The proposed approach called ReFeri re-uses the few shot examples given to the LLM to verify the output. It basically combines the score of the output sequence conditioned on the input examples with the score of the few shot examples conditioned on the output sequence. This is inspired by the classical Bayes rule. The experimental evaluation is carried out on several benchmark datasets including QA tasks as well as reasoning tasks, using GPT and Llama models. The results demonstrate improved performance compared with existing state-of-the-art approaches.

### Strengths
- Test-time scaling and verification of LLM outputs are crucial to improve LLM performance in complex tasks such as reasoning. Therefore, advances in LLM output verification are definitely called for.

- The proposed method is very efficient in the sense that it does not require additional training (as it is usually the case for RPMs).

### Weaknesses
- While the forward confidence score is fairly straightforward, I found the presentation of the backward confidence score a bit confusing. Perhaps an illustrative example would help to better convey the technical details. 

- The performance measures reported in Table 1 lack standard deviations.

- The novelty of the proposed approach is fairly limited, it is basically a simple combination of sequence probabilities that seem to work in practice on a selection of datasets. The paper does not provide any theoretical justification as to why the proposed scheme works.

### Questions
- Aș far as I understand the proposed scheme requires token probabilities. If that's not the case then how do you compute the forward and backward consistency scores?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces Referi (Recycle Few-Shot Examples to Verify LLM Outputs), a training-free framework for best-of-N selection in large language models. Current state-of-the-art approaches rely on output or process reward models, which require additional training for each target task. In contrast, Referi leverages few-shot examples and uncertainty estimation to perform selection without any task-specific training. It computes two complementary signals: (1) a forward confidence score, which measures the model’s confidence in its generated output given the few-shot examples, and (2) a backward confidence score, which assesses the confidence of the few-shot examples given the generated output. The framework is grounded in a Bayesian formulation, and its effectiveness is supported by experimental results.

### Strengths
- Paper is well-written and easy to follow.
- The authors motivate their proposal well and support the effectiveness of their method through various additional experiments. 
- Even though the confidence score calculation and usage already exist in the literature and do not provide much to the community, I liked the incorporation of the backward confidence score.

### Weaknesses
- A recent related work, which also uses uncertainty for best of N selection,  is missing in the evaluations and related works:

 Zhewei Kang et al, Scalable Best-of-N Selection for Large Language Models via Self-Certainty, 2025

- I think one simple and crucial baseline is missing: self-consistency (majority voting). Selecting the most common answer across sampled generations. This is a stronger baseline than random selection.

- Seeing the golden performance would be helpful for better analysis. I.e, the performance of a golden selection mechanism (always selecting the correct option if it exists). 

- There is an implicit assumption that the probabilities of the estimation model are well-calibrated. However, this is not the case in the real world. How would the calibration level of the estimation model affect Referi? I think this situation might amplify bias in certain cases, which should be added as a discussion in the paper.

### Questions
See weaknesses.

### Soundness
4

### Presentation
4

### Contribution
2

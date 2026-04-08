## Human Reviewer 1

### Summary
This paper presents RLIE (Rule Generation with Logistic Regression, Iterative Refinement, and Evaluation), a neuro-symbolic framework that combines large language models with classical probabilistic modeling for interpretable rule learning. The proposed pipeline consists of four stages: rule generation via LLMs, weight estimation using logistic regression, iterative refinement on hard examples, and evaluation under four inference strategies (E1–E4). Experiments on six binary classification tasks demonstrate that the linear-only logistic model (E1) achieves the best performance, suggesting that LLMs are effective in generating rule candidates but less reliable at probabilistic integration.

### Strengths
The idea of combining LLM-based semantic rule generation with a probabilistic model for global reasoning is conceptually appealing.

### Weaknesses
1. Although the paper presents a well-organized framework that integrates LLM-based rule generation with probabilistic weighting via logistic regression and iterative refinement, the overall idea is conceptually incremental. The notion of combining LLM-generated symbolic rules with classical probabilistic or statistical models has already appeared in several recent neuro-symbolic or rule-learning works.

2. The experiments rely solely on GPT-4o-mini with a near-deterministic decoding setting. It remains unclear whether the observed results, especially the relative advantages of the linear combiner over LLM-augmented inference, would hold for other LLMs.

3. The paper contains several noticeable formatting problems that affect readability. For example, Tables 1 and 2 exceed the page margins, and some layout elements are misaligned. In addition, there are minor typographical errors—most notably, “Lange Models” in the abstract should be “Language Models.” Careful proofreading and layout adjustments are recommended before publication.

4. No error analysis is provided to explain where and why the proposed method succeeds or fails. Moreover, although the authors claim that the learned rules are interpretable, there are no visualizations or case studies demonstrating the semantic plausibility or human-understandability of these rules. Adding such analyses would significantly enhance the paper’s insightfulness and credibility.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper introduces RLIE, a framework designed to integrate Large Language Models (LLMs) with probabilistic rule learning. The primary contribution is a four-stage process that aims to overcome the limitations of traditional rule learning by leveraging the natural language capabilities of LLMs. The stages are: (1) Rule generation, where an LLM proposes and filters candidate rules in natural language; (2) Logistic regression, which learns probabilistic weights for the rules to enable global selection and calibration; (3) Iterative refinement, where the rule set is continuously optimized based on prediction errors; and (4) Evaluation, which assesses the performance of the learned rule set. The goal is to create a more robust neuro-symbolic reasoning system by combining the generative power of LLMs with classical probabilistic methods.

### Strengths
- The paper addresses an important and challenging problem: integrating the semantic capabilities of LLMs with more structured, probabilistic reasoning frameworks.
- The proposed iterative refinement loop, where the LLM is prompted to revise rules based on model errors, is an interesting idea for automated feature engineering.
- The work explores different ways of combining learned rules with LLMs for inference, leading to an interesting (though negative) result about the difficulty of fine-grained probabilistic control in LLMs.

### Weaknesses
- The central concept of a "rule" is ill-defined and misleading. What the paper calls "rules in natural language" are effectively just natural language prompts or questions posed to an LLM to generate ternary features (+1, 0, -1). These "rules" lack the formal structure, interpretability, and compositionality of rules in traditional symbolic systems.
- The experimental comparison is flawed. The paper compares the performance of a trained logistic regression model against a prompted LLM that is given the rules and weights. This is an unfair comparison, as one is a trained system while the other is not. The potential of the LLM-based classifier has not been fully explored.
- The empirical evaluation is weak and lacks rigor. The experiments are conducted using only a single, small model ("gpt-4o-mini"). The paper's conclusions about LLM capabilities are therefore based on very limited evidence and may not generalize to other, more capable models.
- The system demonstrates no compositionality between rules, which is a key feature of traditional rule-based systems. The "rules" are treated as independent features for a linear model.

### Questions
1. The paper's claims revolve around "rules," but the learned artifacts appear to be non-compositional natural language prompts for feature extraction. Can you justify the use of the term "rule" and explain how these differ from simple learned features, given their lack of formal structure or compositionality?
2. The paper's primary conclusion relies on an experimental setup that compares a trained model (Logistic Regression) with an untrained one (a zero-shot LLM), using only a single, non-frontier model. How can the general claims about LLM limitations be supported by this specific and seemingly flawed comparison?

### Soundness
1

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper proposes a large language model (LLM)-based rule learning framework, RLIE. The framework leverages LLMs to generate natural language rules, which are assigned probabilistic weights via a logistic regression model. In the iterative optimization process, difficult samples from task predictions are fed back to the LLM along with the current rule set to generate new rules. The authors evaluate the method on six text classification datasets under different reasoning strategies. The results demonstrate that directly using the rules and their learned weights for prediction achieves better performance.

Although the idea of using logistic regression to learn rule weights proposed in this paper is reasonable, the method still has several important flaws in its experimental design that need to be addressed.

### Strengths
1.	The paper is well-structured and readable.
2.	The methodology on using logistic regression to learn rule weights is sound and shows a certain degree of novelty, compared with the traditional top-K methods.

### Weaknesses
1.	Insufficient baselines for experimental comparisons.   In the experiments, the authors use approximately 400 labeled samples but do not compare with the methods such as few-shot in-context learning (ICL) or fine-tuning neural networks, as discussed in the studies like: “What Makes Good In-Context Examples for GPT 3?” and “In-Context Learning Learns Label Relationships but Is Not Conventional Learning.”
Moreover, the authors do not assess how the proposed method scales with varying model capacities. It only uses a single backbone model (GPT 4o mini). It is suggested to compare with the models of different sizes or different architectures. 
2.	Limited experimental tasks. The experiments are confined to binary text classification task, which are relatively simple. For example, in Table 1, the single-rule baseline (IO Refinement) achieves the best performance on two datasets, suggesting that the reasoning complexity is limited. Additionally, the test sets are small (about 300 samples), which further restricts the generalizability of the method.
3.	Insufficient analysis of method effectiveness. The paper does not evaluate the quality of the generated rules, such as how the diversity of rules influences the effectiveness of method, what is the impact of the key hyperparameters. Although the coverage threshold γ is fixed at 0.2, it is still required to conduct experiments to examine its influence.
4.	Lack of clarity in methodological details. It is unclear whether the optimization of rule weights adequately covers all rules. During incremental rule generation, the process for assigning weights to new rules is not explained. Furthermore, in scenarios with frequent rule updates, it remains uncertain whether these weights receive sufficient training.

### Questions
1.	The single-rule baseline (IO Refinement) outperforms the proposed method RLIE. Does this imply that combining multiple rules might be unnecessary in such cases? How do the authors explain the inferior performance of RLIE compared to a simpler approach on these specific tasks?
2.	It is unclear whether each rule’s weight is sufficiently trained during the iterative optimization process. While stopping criteria for the iterations are provided, there is a lack of statistical analysis or visualizations illustrating the frequency of weight updates or the convergence behavior of individual rules.


Finally, it is recommended that the authors conduct further analysis of the generated rules and reasoning outcomes, such as examining the distribution of error types in incorrectly reasoned samples and evaluating whether the new rules have effectively corrected these errors. This is particularly important given that the rules are expressed in natural language, where interpretability is a crucial factor.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
5

---

## Human Reviewer 4

### Summary
The paper introduces RLIE, a framework that combines LLMs with probabilistic rule learning to generate and refine rules for better decision-making. The RLIE process involves four stages: generating candidate rules using an LLM, applying logistic regression to learn the weights of these rules, refining the rule set iteratively based on prediction errors, and evaluating the model by comparing its performance to other methods of rule integration. The study shows that applying weighted rules directly results in superior performance, but injecting rules into an LLM using prompt-based methods can lead to degraded performance. This suggests that while LLMs are strong in semantic generation, they struggle with fine-tuned, controlled probabilistic integration. The main contribution of the paper is the development of a unified framework that combines LLMs with traditional probabilistic rule combination techniques, advancing the field of neuro-symbolic reasoning systems.

### Strengths
1. The paper is well-written and methodologically sound. It provides a clear and detailed explanation of the RLIE framework, including the rule generation, logistic regression, iterative refinement, and evaluation stages. 
2. The work has significant potential for advancing the field of neuro-symbolic AI. By successfully combining LLMs with probabilistic rule learning, it addresses an important challenge in AI: integrating the flexibility of generative models with the precision of rule-based reasoning.

### Weaknesses
1. The work lacks experimentation on more models to ensure that its effectiveness is broad and not limited to specific ones.
2. There are significant formatting issues with Tables 1 and 2, as they exceed the page width.
3. The introduction to the task in the work is not clear enough. More concrete examples should be introduced to describe the entire process, in order to improve the readability of the paper.
4. The method is not broadly effective across all tasks, and its average performance improvement over previous work is limited.

### Questions
1. Can you reproduce some comparative experiments on weaker open-source models (such as Qwen3) and stronger closed-source models (such as GPT-5) to demonstrate the generalizability of the method?
2. Why does the method perform significantly worse than the IO Refinement method on the Dreaddit and LLM Detect datasets?
3. The generalizability of the conclusions found in Section 5.2 is questionable. Is the performance drop caused by the introduction of LLM due to the selection of relatively weaker models? Please add experiments on more models to further support your findings.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
2

### Confidence
3
# iART: Imitation guided Automated Red Teaming

- Decision: Reject
- Scores: 2, 6, 6

## Abstract
The potential of large language models (LLMs) is substantial, yet they also carry the risk of generating harmful responses. An automatic "red teaming" process constructs test cases designed to elicit unfavorable responses from these models. A successful generator must provoke undesirable responses from the target LLMs with test cases that exemplify diversity. Current methods often struggle to balance quality (i.e., the harmfulness of responses) and diversity (i.e., the range of scenarios) in testing, typically sacrificing one to enhance the other, and relying on non-optimal exhaustive comparison approaches. To address these challenges, we introduce an imitation-guided reinforcement learning approach to learn optimal red teaming strategies that generate both diverse and high-quality test cases without exhaustive searching. Our proposed method, Imitation-guided Automated Red Teaming (iART), is evaluated across various LLMs fine-tuned for different tasks. We demonstrate that iART achieves not only diverse test sets but also elicits undesirable responses from the target LLM in a computationally efficient manner.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes iART (Imitation-guided Automated Red Teaming), a reinforcement learning-based approach for automatically generating diverse and high-quality test cases to elicit harmful responses from large language models. The method introduces two key components: (1) imitation guidance using a "harm model" trained on examples of undesirable outputs to guide the attack LLM, and (2) a diversity module that models the distribution of previously generated outputs to penalize repetition. The approach is evaluated on text continuation and instruction-following tasks across multiple target LLMs (Mistral-7B, GPT-2-Alpaca, Dolly-3B), demonstrating improvements over baselines (RL, RL+TDiv, RL+Curiosity) in both quality and diversity while being computationally more efficient.

### Strengths
The paper clearly articulates the fundamental tension between quality and diversity in automated red teaming, and identifies computational inefficiency as a key limitation of existing methods.

### Weaknesses
1. The paper fails to cite and compare with several closely related and recently published works that address similar challenges in automated red teaming. For instance, DiveR-CT (Zhao et al., 2025) proposes relaxing conventional constraints on the objective and semantic reward, granting greater freedom for the policy to enhance diversity—an approach that directly addresses the quality-diversity trade-off that iART claims to solve. Similarly, HARM (Zhang et al., 2024) introduces a top-down hierarchical approach to generating test cases that ensures diversity through structured decomposition of harm taxonomies. Without empirical comparisons showing how iART performs relative to these methods, the paper's claimed contributions appear limited and its positioning within the current state-of-the-art remains unclear.
  reference:
 - (Zhao et al., 2025) DiveR-CT: Diversity-enhanced Red Teaming Large Language Model Assistants with Relaxing Constraints
 - (Zhang et al., 2024) Holistic Automated Red Teaming for Large Language Models through Top-Down Test Case Generation and Multi-turn Interaction
2. All main experiments in the paper use GPT-2-137M as the attack model, which raises significant questions about whether the approach scales to more capable and practical attack models. While Appendix E presents results using Mistral-7B as the attacker, these experiments reveal concerning discontinuities in the diversity plots that are only briefly mentioned and inadequately explained. The authors attribute these gaps to "no examples available at specific toxicity thresholds", but this explanation is unsatisfying and suggests potential instabilities when scaling up the attack model. Furthermore, the paper provides no analysis of how computational costs scale with attack model size—a critical consideration given that the diversity module requires online training and the harm model requires inference at each step. Without understanding these scaling properties, it remains unclear whether iART's computational advantages over RL+Curiosity would persist with larger, more powerful attack models that might be needed to successfully red team state-of-the-art LLMs.
3. The paper's evaluation framework has several limitations that undermine confidence in the results. First, using cosine similarity for measuring diversity is relatively shallow and may fail to capture semantic diversity—two test cases could have different embeddings but explore the same type of harm, or conversely, have similar embeddings while targeting different vulnerabilities. Second, the F1DQ metric that combines quality and diversity using harmonic mean lacks theoretical or empirical justification; the paper does not explain why harmonic mean is the appropriate aggregation function or whether other combinations (e.g., weighted averages, geometric mean) were explored. Third, relying on a single RoBERTa toxicity classifier as the sole evaluator may miss nuanced forms of harm and creates a narrow definition of "quality" that could be gamed by the attack model. Most critically, the paper includes no human evaluation to validate that the discovered "attacks" are genuinely problematic or that the diversity improvements represent meaningfully different failure modes rather than superficial lexical variations. Without human assessment, it is difficult to determine whether iART truly advances the practical goal of comprehensive LLM safety testing.

### Questions
1. Can you provide ablation studies showing how performance varies with different sizes and types of harm datasets? What is the minimum viable D_harm?
2. Can iART discover types of harmful content not present in D_harm? Or does it primarily find variations of known harms?
3. How does iART compare with gradient-based attacks like GCG in terms of attack success rate and diversity? These methods have shown high success rates on recent models.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on red-teaming for LLMs, following the settings of “RL” (Perez et al., 2022)  and “RL+Curiosity” (Hong et al., 2024). The authors added another input to the RL+Curiosity original pipeline, which is a harmful LM to provide standard harmful outputs. It’s trained on a harmful dataset. With this addition the RL process can get additional teacher guidance on what a “harmful” response would be under the current input prompt, and there is one more term in the objective to encourage the attacker to elicit similar responses from target LLM (measuring cosine similarity as reward). The experiment results show that this approach converged faster with better quality and significantly lowered the execution time to find a harmful prompt compared to “RL+Curiosity”.

### Strengths
1. The imitation-guidance clearly stabilizes RL training: it acts like a teacher signal that speeds up convergence and makes the attacker’s learning more stable.
2. Experiments show the attacker reaches effective harmful prompts more quickly and with higher success than RL+Curiosity.
3. Effective across different language models and observe similar gains, suggesting the method generalizes across model architectures and sizes.

### Weaknesses
1. There is a potential issue that the imitation may limit exploration (mode bias). The attacker is explicitly rewarded to produce prompts that make the target model output responses similar to the harm (teacher) model. That can bias the attacker to explore only regions of the target’s output space that match the teacher’s harmful patterns. Even if the teacher itself produces diverse harmful outputs, the attacker’s exploration is effectively constrained to that teacher-driven subspace. As a result, the attacker may miss other kinds of harmful behaviors that the target can produce but the teacher is not able to produce.

2. The above issue may not be reflected on the current evaluation benchmarks. Because the harm model’s outputs are themselves diverse, standard diversity measures (Self-BLEU, embedding cosine diversity) will still report high diversity even when the attacker is merely mimicking the teacher. Those metrics do not reveal whether the attacker is covering teacher-like harmful modes only or genuinely discovering teacher-independent harmful modes.

3. Also, there’s potential for overfitting to the harm dataset. Because imitation guidance depends on a harmful dataset to train the teacher, any dataset bias transfers into what the attacker searches for and limits the coverage of other real-world harmful cases.

### Questions
I would appreciate if the authors responded to some of the potentially weaknesses outlined above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies a red teaming LLM problem. This paper proposes to use a pre-collected dataset of harmful outputs to help the attack LLM generate prompts triggering harmful outputs, and a model that measures if a certain response has been generated before to encourage the attack LLM to produce diverse outputs. The results show that the performance in quality and diversity is similar to the prior work, while the proposed method significantly reduces the execution time.

### Strengths
The execution time seems to be largely reduced compared with the prior works, which is a great contribution. However, I would love to see more explanation of how the proposed method made it into the paper rather than the appendix.

### Weaknesses
- The writing was not clear. See my questions.
- Presentation is messy. The layout of the figures needs to be largely improved.

### Questions
- Line 170: I didn't understand the need to train the harm LLM. It seems to me that encouraging the attack LLM to generate prompts that make the target LLM produce outputs similar to the outputs from the harm LLM is the same as eliciting outputs with harmful outputs measured by an evaluator.
- Section 4.2 & Section 4.1: The details of the diversity module and harm LLM shouldn't be put in the Appendix. The recipe for training the diversity module and harm LLM seems to be the main contribution of this paper.
- Fig 3: There are a lot of glitches on the borders of the figure. It looks unprofessional. Also, there are no legends.

### Soundness
2

### Presentation
2

### Contribution
2

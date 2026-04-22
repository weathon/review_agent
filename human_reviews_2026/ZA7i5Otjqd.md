# Aligning Deep Implicit Preferences by Learning to Reason Defensively

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 2, 6

## Abstract
Personalized alignment is crucial for enabling Large Language Models (LLMs) to engage effectively in user-centric interactions. However, current methods face a dual challenge: they fail to infer users' deep implicit preferences (including unstated goals, semantic context and risk tolerances), and they lack the defensive reasoning required to navigate real-world ambiguity. This cognitive gap leads to responses that are superficial, brittle and short-sighted. To address this, we propose Critique-Driven Reasoning Alignment (CDRA), which reframes alignment from a scalar reward-matching task into a structured reasoning process. First, to bridge the preference inference gap, we introduce the DeepPref benchmark. This dataset, comprising 3000 preference-query pairs across 20 topics, is curated by simulating a multi-faceted cognitive council that produces critique-annotated reasoning chains to deconstruct query semantics and reveal latent risks. Second, to instill defensive reasoning, we introduce the Personalized Generative Process Reward Model (Pers-GenPRM), which frames reward modeling as a personalized reasoning task. It generates a critique chain to evaluate a response's alignment with user preferences before outputting a final score based on this rationale. Ultimately, this interpretable, structured reward signal guides policy model through Critique-Driven Policy Alignment, a process-level online reinforcement learning algorithm integrating both numerical and natural language feedback. Experiments demonstrate that CDRA excels at discovering and aligning with users' true preferences while executing robust reasoning. Our dataset is available at \url{https://DeepPref.github.io/}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces **Critique-Driven Reasoning Alignment (CDRA)**, a framework designed to bridge the gap between users’ implicit preferences and their explicit, surface-level feedback in personalized alignment. The authors first construct a synthetic dataset, DeepPref, which captures the reasoning process behind inferring users’ latent intents and corresponding rewards. Using this dataset, they train the **Personalized Generative Process Reward Model (Pers-GenPRM)**, which models personalized reward inference as a reasoning task. Finally, they propose **Critique-Driven Policy Alignment (CDPA)**, which leverages the learned rewards to compute token-level advantages and mitigate the zero-gradient issue. Experimental results demonstrate that CDRA consistently outperforms all baselines on both the DeepPref and PrefEval benchmarks.

### Strengths
- The motivation of bridging the gap between users’ implicit preferences and surface-level ones is clear and reasonable, effectively framing personalized alignment as a reasoning process.
- The idea of using the reward model to provide token-level advantages is solid, as it offers denser feedback signals when optimizing the policy.

### Weaknesses
- The experiments are conducted on only one model; the results would be more convincing if additional models were included.
- The paper lacks a comprehensive human study; directly assessing users’ satisfaction with the responses would strengthen the evaluation.

**Minor issues:**

- It is somewhat confusing and potentially misleading to place the *Evaluation Protocol* paragraph before the training section. I suggest moving it afterward for better readability.
- Some notations are undefined, including $N$ in Equation 1 and $r_{i,t}$ in Equation 5.

### Questions
- How does the model identify the $T_i$ reasoning steps within each response?
- Regarding the proposed metric, *Deep Preference Understanding*, does the prompt for the LLM evaluator of step-wise rewards and critiques include similar criteria? If so, could this lead to reward hacking?
- The dataset appears to contain only surface-level preferences, not implicit ones. How can we be sure that the LLM evaluator truly captures users’ latent intents rather than hallucinating them?
- Since the reward model (Pers-GenPRM) is trained in a supervised manner, is it possible to directly use the LLM evaluator without fine-tuning a separate reward model?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces "Critique-Driven Reasoning Alignment" (CDRA), which uses structured reasoning to judge user-specific preferences, and they also include a dataset: DeepPref, a benchmark enabling the measurement of how well a given alignment setup aligns a model with a user's deeper (e.g. unstated or implied) preferences, and a reward modeling framework: Pers-GenPRM, a personalized generative reward model. They show that their method can improve deeper alignment while still maintaining performance on explicit preference ranking tasks.

### Strengths
This paper rigorously explores how to better align models with implicit or implied preferences from users, and their experiments are quite thorough and well-grounded. They use reasonable datasets on up to date models, and compare with strong baselines, so their results seem quite reliable. Their methodology is straightforward and does not require too much compute to explore, and I'd be very curious to see how well it scales up to larger model sizes or more difficult datasets.

### Weaknesses
The paper's presentation is at times slightly confusing; the introductory paragraph is quite dense, and it's difficult at first read to keep all of the acronyms straight. Figure 4 (the radar plot) is also hard to read with as many lines as it has, and I'd recommend maybe a collection of bar charts, or a set of averages with detailed evaluations in the appendix. Figure 1/the introductory example about location data is also confusing because a "correct" example response is not provided; the first person shooter/mass effect example is more clear in my opinion.

I'm also not totally convinced that the personalized reward model is necessary; it also feels like a setup where a rubric, and/or test-time scaling for a generative reward model, could achieve the same goal? That's not necessarily a mark against the paper, but I'd be curious to see how different judgement strategies work with your methodology, if time/space/compute allows.

### Questions
For chain of thought, can you tell me more about your experimental setup? Did you also use Qwen 2.5 7B Instruct to generate the reasoning chains, and if so, did you explore using any of the more recent open weight reasoning models that do this natively? I believe a few have been trained on top of Qwen 2.5 7B Instruct as well. I'm curious if improved reasoning in the base model noticeably improves performance on your benchmark, or if more complex prompting is necessary.

> This approach resolves the “zero advantage” problem that arises in standard RL when multiple responses are functionally correct but differ in their underlying reasoning quality (likes 104 - 106)
I'm a bit confused by this statement. To me this sounds instead like you're describing a scenario where the judgement process is not correct, and your method is *a* way to address it, but theoretically another way would be to maybe use a rubric instead of a single scalar reward, or some other improved evaluation metric? Am I understanding this correctly?

### Soundness
3

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
4

### Summary
This paper tries to tackle two challenges in current Large Language Model (LLM) alignment: the preference gap (failing to infer users' deep, unstated goals and risks) and the process gap (failing to execute defensive reasoning). The authors first introduces a new dataset called DeepPref, which consists of 3000 preference-query pairs with "critique-annotated reasoning chains" generated by GPT-4.1 to provide process-level supervision. A Personalized Generative Process Reward Model (Pers-GenPRM) is then introduced, which evaluates a response by first generating an explicit critique chain and then deriving a score from that rationale. Finally, the authors then propose Critique-Driven Reasoning Alignment (CDRA), an online reinforcement learning paradigm that uses numerical scores and the natural language feedback from the Pers-GenPRM to train a final policy model. Experiments show that CDRA outperforms existing training/test-time methods in both deep preference understanding and defensive reasoning.

### Strengths
**S1.** The paper tries to tackle important problems in LLM alignment, which are introduced by the authors as implicit preference and process gaps.

**S2.** The overall CDRA framework outperforms existing training/test-time methods on DeepPref and PrefEval across multiple metrics that measures preference understanding and defensive reasoning.

### Weaknesses
**W1.** The DeepPref dataset, which is supposed to be the foundation of the paper, is entirely synthesized by GPT-4.1 and has no human validation. Therefore, in my opinion, it remains unclear whether the CDRA framework is actually learning human-aligned reasoning or merely imitating another model’s critique style. This limitation is particularly quite important given that one of the paper’s stated goals is to close the preference gap (the inability to infer deep, implicit human intent), so if current LLMs exhibit the preference gap that CDRA aims to solve, then wouldn't the synthetic critiques and reasoning chains produced by GPT-4.1 may also encode that same deficiency (along with bias towards certain culture/demographics)?

**W2.** The evaluation also relies on GPT-4.1 as the judge, which may share biases with the model that generated the training data. This setup risks circularity and could inflate performance by rewarding outputs that mimic GPT-4.1’s reasoning patterns rather than genuinely improving alignment quality.

**W3.** While CDRA is compared with various different train/test-time paradigms, the paper’s empirical scope is still narrow in terms of variation. The paper evaluates only one model only, no comparison against other existing reward models, and no comparison against other process-based reward modeling approaches that tackle similar problem.

### Questions
Other than states in the weaknesses, I have some further questions:

**Q1.** Seems like no files in the anonymous link can be viewed.

**Q2.** Since the base model used is an Instruct model, I am curious whether existing reasoning models or reasoning-based reward models [1, 2] might already exhibit some degree of implicit reasoning. Furthermore, could the observed improvements stem mainly from the model spending more tokens “thinking aloud” during reward-model training, rather than from genuine preference inference?

**Q3.** How does this approach compare to methods where models explicitly ask users clarification questions to resolve ambiguity? [3, 4] Perhaps explicit intent-elicitation might reduce the risk of inferring incorrect or culturally biased “preferences,” which would give a better "personalization". In contrast, the proposed method infers implicit intent without user interaction, which may not generalize well across diverse populations.

**Q4.** Perhaps some human-evaluations under multi-turn conversations?

### References

[1] Anugraha, D., Tang, Z., Miranda, L. J. V., Zhao, H., Farhansyah, M. R., Kuwanto, G., ... & Winata, G. I. (2025). R3: Robust rubric-agnostic reward models. arXiv preprint arXiv:2505.13388.

[2] Chen, X., Li, G., Wang, Z., Jin, B., Qian, C., Wang, Y., ... & Ji, H. (2025). Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387.

[3] Wu, S., Galley, M., Peng, B., Cheng, H., Li, G., Dou, Y., ... & Gao, J. (2025). Collabllm: From passive responders to active collaborators. arXiv preprint arXiv:2502.00640.

[4] Wang, A., Lin, Y., Liu, J., Wu, S., Liu, H., Xiao, X., & Su, J. (2025). Beyond Passive Critical Thinking: Fostering Proactive Questioning to Enhance Human-AI Collaboration. arXiv preprint arXiv:2507.23407.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Critique-Driven Reasoning Alignment (CDRA), a process-driven approach to personalize LLMs by (1) constructing a critique-annotated dataset DeepPref, (2) training a generative process reward model Pers-GenPRM that emits stepwise critiques + scores, and (3) optimizing the policy with a token/step-wise RL algorithm (CDPA). The authors report strong improvements on their benchmark and an external PrefEval set using an LLM-as-judge evaluation.

### Strengths
- Novel framing. The distinction between a preference gap and a process gap, and explicitly supervising the reasoning process via critiques, is a clear, useful conceptual contribution.

- Ablations and multi-metric evaluation. The ablation removing Pers-GenPRM and RFT provides insight on component contributions; the multi-dimensional metrics (deep mining, innovative expansion, defensive reasoning, preference following) are appropriate for the task.

### Weaknesses
● Heavy reliance on LLM-as-judge
The evaluation protocol uses GPT-4.1 as the primary judge (LLM-as-a-judge) for metrics like AccDA and AccPF. This risks circularity: a model trained on LLM feedback may be favored by an LLM judge, and shared model/systemic biases can inflate scores. The paper reports a 96% agreement on 300 samples between the LLM judge and humans (Appendix), but that’s a small check relative to the breadth of claims. More human evaluation is required to validate the claimed gains.

● Generalization beyond DeepPref.
Reported gains are strong on DeepPref (and PrefEval), but it’s unclear how CDRA generalizes to genuine downstream tasks or real user interactions. Add evaluations on external, independently curated personalization datasets or task-based scenarios to demonstrate robustness.

● About Baseline.
No comparisons to popular on-policy preference-alignment methods (PPO/GRPO).

### Questions
● Exactly how was DeepPref constructed? For each item, who authored the persona, queries, “true” preferences, critiques, and scores — humans, LLMs, or a hybrid?

● How does CDRA conceptually differ from existing process-supervision or critique-guided alignment methods — please point to the closest prior works and clarify novelty.

● Why should stepwise generative critiques provide strictly more useful training signal than scalar reward models or pairwise preference labels?

### Soundness
3

### Presentation
3

### Contribution
3

# GSPR: Aligning LLM Safeguards as Generalizable Safety Policy Reasoners

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
As large language models (LLMs) are increasingly integrated into numerous applications across various domains, LLMs’ safety becomes a critical concern for both application developers and intended users. Currently, great efforts have been made to develop safety benchmarks with fine-grained taxonomies. However, these benchmarks’ taxonomies are disparate with different safety policies.
Thus, existing safeguards trained on these benchmarks are either coarse-grained to only distinguish between “safe” and “unsafe,” or constrained by the narrow risk taxonomies of a single benchmark. To leverage these fine-grained safety taxonomies across multiple safety benchmarks, in this paper, we propose GSPR, a Generalizable Safety Policy Reasoner to identify unsafe input prompts and LLMs’
outputs with violated safety taxonomies through Group Relative Policy Optimization (GRPO). 
Unlike prior safeguards which only cover a fixed set of risk factors, our GSPR incentivizes its reasoning capability with varied safety taxonomies through our careful cold-start strategy and reward design. 
Consequently, our GSPR can be trained across multiple safety benchmarks with distinct taxonomies and naturally exhibits powerful generalization ability. We conduct extensive experiments to show that our GSPR significantly improves existing safety guardrails’
reasoning capabilities for both safety and category prediction tasks. 
Moreover, our GSPR not only demonstrates powerful safety generalization abilities but also achieves the least inference token costs with explanations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GSPR (Generalizable Safety Policy Reasoner), a reinforcement-learning–aligned framework for fine-grained and generalizable LLM safety reasoning. The paper argues and proposes dynamically incorporating different safety policies from different providers to reason over various safety domains. GRPO is used in combination with SFT to improve reasoning quality and enforce structured otuput formatting. 
Experimental results on eight benchmarks show that GSPR achieves state-of-the-art performance in both safety classification and category prediction, surpassing previous RL-aligned guardrails (e.g., RSafe) by up to 45% in category accuracy.

### Strengths
- Clear, reproducible training pipeline; ablations show each piece helps.
- Consistent S-Acc gains and meaningful C-Acc improvements in several settings.
- Practical attention to token cost and generation hygiene (repetition/language mixing).
- Useful qualitative cases; explanations improve transparency relative to pure classifiers.

### Weaknesses
* Novelty vs. Prior Multi-Policy Guardrails - There is a body of work training guardrails over multiple benchmarks/taxonomies and/or policy-conditioned prompts, and RL-aligned safety models. The paper needs a crisper positioning: what exactly is new beyond combining these ingredients? Ask: Add a thorough related-work section covering multi-policy/taxonomy conditioning and RL-aligned safety reasoners; run head-to-head with the strongest of those under identical taxonomies and format constraints.
* Stronger Baselines / Controls - Train baselines with the same flexible prompt (policy list + \safety/\category) but without GRPO (SFT only) and with PPO (no GRPO), matched for steps/compute, to isolate GRPO’s contribution.
* Include frontier API judge baselines with structured prompting for categories (where allowed), not just default settings.
* Robustness to Reward Hacking & Formatting - Add semantic judges (weak or learned reward models) to verify that correct categories are semantically justified, not just formatted.

### Questions
* There has been prior work on learning from various safety policies before, why not include in related work ? e.g GuardFormer : Guardrail Instruction
Pretraining for Efficient SafeGuarding - https://openreview.net/pdf?id=vr31i9pzQk
* How sensitive are results to α₁/α₂? Provide a sweep and show S-Acc/C-Acc/length trade-offs.
* What happens if you remove the format penalty but keep safety/category rewards?
* How does GSPR handle entirely novel policy text that overlaps semantically but not lexically with training categories?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work addresses the challenge that existing reasoning-based guard models struggle with fine-grained safety category prediction. To improve category-wise safety moderation, the authors propose GSPR, which introduces variable and flexible safety taxonomies into the prompt rather than using a fixed set of policies. The model is trained using GRPO similarly to RSafe. Experiments demonstrate meaningful improvements in category classification accuracy compared to prior methods.

### Strengths
- The paper clearly introduces the importance of accurate category classification beyond simple safe/unsafe decisions in real-world LLM safety applications. This focus on explainability and policy grounding is well motivated.
- The proposed GSPR shows notable empirical gains in category prediction performance over existing guard models and training strategies. This suggests that prompting with flexible safety categories is an effective design choice for improving guardrail reasoning.

### Weaknesses
## 

- While the proposed approach is practical and performs well, the methodological novelty may feel somewhat incremental. Reinforcement learning for guardrail alignment has been previously explored (e.g., RSafe [1]), and introducing dynamic categories in prompts could be seen as a natural step when optimizing for fine-grained categorization. Clarifying the conceptual novelty and broader implications could further strengthen the contribution.
- In Table 2, the improvement in safety accuracy (S-Acc) with Cold-Start is promising. However, since the Cold-Start stage uses carefully curated data from Gemini-2.5-Flash and GSPR w/o Cold-Start performs similarly to RSafe, one could reasonably expect RSafe to benefit similarly from a comparable warm-start strategy. Although the Cold-Start itself is useful, its originality may be limited given its prevalence in RL-based alignment pipelines [2]. Providing more justification for its role or demonstrating alternatives would help highlight the contribution.

[1] Zheng et al., RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards

[2] DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Questions
- For baselines that rely on fixed safety policies, how are these policies incorporated into prompts? Since different benchmarks define different taxonomies, does each dataset use a separate prompt design, or is a unified prompt format used (e.g., all policies)? It would be helpful to clarify whether an ablation exists.
- At inference time, how are the safety categories selected for the prompt? Are benchmark-specific categories provided, or are all categories included to ensure full coverage?
- In practical deployment, the model deployers may not know in advance which taxonomy applies. How would the system operate in real-world scenarios? Would category retrieval or selection be necessary to determine which policies to present to the model?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GSPR, a training pipeline aimed at enabling reasoning models to detect unsafe input prompts and outputs. A central claimed contribution is that the algorithm facilitates adaptation to diverse safety taxonomies through a deliberate cold-start strategy and targeted reward engineering. The authors also conduct experiments to evaluate the GSPR's efficacy.

### Strengths
1. The motivation is reasonable, and the field is important. Safeguards for large language models (LLMs) must indeed support the generalization of flexible safety policies.
2. The authors address unseen safety policies and evaluate the conditions under which adaptation occurs.

### Weaknesses
1. **Limited Novelty**. From my perspective, this paper does not introduce substantial innovations. It primarily applies GRPO—a well-established algorithm—to safety evaluation tasks and reasoning models. The highlighted motivation and contributions strike me as somewhat trivial. A capable LLM can already adapt to varying safety policies embedded in prompts, leveraging its strong instruction-following capabilities.
2. The paper's use of distillation from Gemini Flash for cold-start initialization only reinforces this view. I suspect that a sufficiently powerful model could handle this task effectively without such interventions. Moreover, the task design itself raises concerns: In an era of highly advanced models, why constrain the candidate policies to a mere 167 examples? Why not generate rubrics dynamically or retrieve them from a much larger pool of thousands?
3. Building on the above, the scale of the 167 policies feels underwhelming and less meaningful than it could be. In my view, the true crux of success here lies in the LLM's instruction-following prowess. I would wager that these policies are fairly commonplace and already ingrained in modern LLMs such as Qwen, DeepSeek, and LLaMA—even in your so-called "out-of-domain" safety evaluations. Have the authors considered testing with truly uncommon or even counterfactual policies? Would the model degrade under such conditions, or maintain robustness?
4. The authors overlook several critical baselines, especially those from GenRM literature. Notably, the safety tasks targeted here represent only a narrow subset of broader reward modeling (RM) challenges. Many RM datasets (e.g., Skywork-Reward-Preference and your used wildguard) already encompass safety classification. Furthermore, some reward models can autonomously generate flexible policies (or rubrics) before applying judgments. In contrast, this work's format remains constrained: most prompts are static, with only minor flexibility introduced. This limits its scope compared to more dynamic approaches. Numerous prior studies have tackled analogous challenges, yet this paper fails to discuss them meaningfully. Relevant examples include:

[1] [ShieldLM: Empowering LLMs as Aligned, Customizable and Explainable Safety Detectors](https://arxiv.org/pdf/2402.16444)

[2] [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/pdf/2504.02495)

[3] [J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning](https://arxiv.org/pdf/2505.10320)

[4] [RM-R1: Reward Modeling as Reasoning](https://arxiv.org/pdf/2505.02387)

[5] [Qwen3Guard Technical Report](https://arxiv.org/pdf/2510.14276v1) (very recently published; could be excused if overlooked))

In summary, the use of dynamic rubrics or principles is now commonplace in this community. Against this backdrop, I find the paper's contributions insufficiently distinctive.

### Questions
See above.

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
The paper proposes GSPR, a Generalizable Safety Policy Reasoner to identify unsafe input prompts and LLMs’ outputs with violated safety taxonomies through Group Relative Policy Optimization (GRPO). The results suggest that it can achieve remarkable detection performance across various safety benchmarks.

### Strengths
- Comprehensive policy coverage. It includes 19 fine-grained taxonomies with 167 safety policies.
- Strong performances. The method demonstrates strong performance over the baselines.

### Weaknesses
- Missing related works. Some related works also consider fine-grained safety classification beyond binary classification, such as ShieldLM and Qwen3Guard. 
- Lack of in-depth analysis. We can only see that the method performs well across different benchmarks. But how safety policies are different across these benchmarks are unclear. For example, will the same sample be safe under some policy but be unsafe under another policy? It is necessary to conduct in-depth analysis regarding the concrete role of safety policies.
- Lack of ablation study on the training data. The paper mixes data from various sources for training, but the impact of these training data from different sources is unclear.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

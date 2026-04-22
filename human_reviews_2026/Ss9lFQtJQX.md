# Think Socially via Cognitive Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
LLMs trained for logical reasoning excel at step-by-step deduction to reach verifiable answers. However, this paradigm is ill-suited for navigating social situations, which induce an interpretive process of analyzing ambiguous cues that rarely yield a definitive outcome. To bridge this gap, we introduce Cognitive Reasoning, a paradigm modeled on human social cognition. It formulates the interpretive process into a structured cognitive flow of interconnected cognitive units (e.g., observation or attribution), which combine adaptively to enable effective social thinking and responses. We then propose CogFlow, a complete framework that instills this capability in LLMs. CogFlow first curates a dataset of cognitive flows by simulating the associative and progressive nature of human thought via tree-structured planning. After instilling the basic cognitive reasoning capability via supervised fine-tuning, CogFlow adopts reinforcement learning to enable the model to improve itself via trial and error, guided by a multi-objective reward that optimizes both cognitive flow and response quality. Extensive experiments show that CogFlow effectively enhances the social cognitive capabilities of LLMs, and even humans, leading to more effective social decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a training framework to improve the social reasoning capability of small LMs via model distillation. A social cognitive reasoning dataset is constructed based on initial social situation descriptions collected online and a set of heuristics using LLMs with reasoning capability. Those data are then used to improve small LMs via supervised fine-tuning and reinforcement learning with auxiliary rewards. The evaluation results indicate the proposed framework leads to better social reasoning performance compared to directly using the same technique individually (e.g. SFT, RL, distillation).

### Strengths
The paper is well written with a clear presentation of motivation, method, and evaluation.

The proposed training framework is grounded in social cognition theory and potentially applies to general social reasoning situations.

The evaluations are thorough with both LLM-as-judge automatic judgment and human judgment.

### Weaknesses
The proposed framework heavily relies on the reasoning capability of LLMs since the data is generated and augmented with LLMs based on a structured workflow with handcrafted heuristics, then evaluated all by LLMs as well. As mentioned in the paper, the robustness of this process may be influenced by the internal bias within the specific LLM. Additionally, to what degree the cognitive flow simulation applies to an out-of-distribution social situation that is not generated via the proposed pipeline is not clear. For example, situations with partial observation and long-horizon interaction where the heuristics may not generalize well.

From the AL/ML perspective, the main contribution of this work is a structured social reasoning framework to generate supervised data for model distillation and corresponding auxiliary rewards used in RL training. While the results on human decision-making are interesting, they are somewhat disconnected from the main technical contribution. Because It is less clear if the same heuristics helping humans can explain why they work for LLMs.

Table 1 needs more context to make sense. In pair-wise comparisons, an agreement ratio kappa around 0.6 seems to be relatively low. This means only ~70% of the time the LLM evaluators rank the pair consistently with humans. This leads to the concern of to what degree the LLM evaluator represents human preference in social reasoning situations.

In section 4.4, the concept of reasoning depth is used before being properly defined.

The paper appears to involve an intense human subject study but does not disclose sufficient information regarding the details and ethical approvals (e.g., IRB). For example, How were the annotators and volunteers recruited? How much were they paid and for how many hours of participation? What were their demographic backgrounds and knowledge of LLMs? Did participants know the experiment design (e.g., the source of different example instances)? Also given that the study uses online-collected content modified by LLMs, I am concerned about the safety filtering process mentioned. Could the authors please elaborate on this process?

The design decision of using majority voting in human annotation lacks rationale. As motivated in the introduction, social reasoning in many situations is by nature ambiguous and may be impacted by personal background and preference. It makes more sense to have independent judges and take the individual difference among the human population into consideration.

Presentation format may be a confounded variable in both the benchmark and two human decision-making experiments. Because the cognitive flow is generally shorter and structured with bullet points while CoT is usually a long chunk of plain text. It is difficult to distinguish whether the effect observed between conditions is due to format or content. An ablation study may be helpful.

### Questions
Although the introduction motivates the work to extend model training to social scenarios with ambiguous cues and no definitive outcomes, the proposed framework actually clearly defines those cues and outcomes via heuristics and powerful LLMs, then distills data to train the small model. This seems to be contradictory. Please clarify.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces CogFlow, a pipeline that distills social situations from Reddit using R1, simulates the flow of thoughts through six tagged “cognitive units,” ranks and filters these units with LLM judges, and performs SFT warm-up followed by GRPO on a smaller-scale 8B Qwen model. LLM-as-judge evaluation results show that the CogFlow pipeline yields better reasoning quality and more concise reasoning traces.

### Strengths
The paper is well written and easy to read. The targeted domain of social reasoning is interesting and deserves greater attention from the research community.

### Weaknesses
- Lack of technical contribution. This paper is one among many recent works that simply adapt SFT + GRPO to a specific domain — in this case, a narrow “social reasoning” setting. Although the authors repeatedly use the term “cognitive,” the work has little to do with actual cognitive science. It is unclear what novel insight or methodological advance this paper offers. Even if the approach generalizes to other tasks (considering broader impact), it seems that the gains would primarily stem from SFT + GRPO, rather than from the proposed “cognitive units.”

- Circluated evaluations. All the questions were generated by a language model (DeepSeek R1). The teacher traces were also generated by a language model, and the student is a language model as well. The performance is evaluated using a language-model-based reward, yet the human agreement (κ) is not outstanding. It’s hard to say whether this setup captures human social cognition or merely reflects the reasoning patterns favored by DeepSeek R1.

- Flawed baselines. Some prompt-engineering baselines are missing — for instance, how would a tuning-free prompt template that enforces a compact, fixed section structure perform? Even simple few-shot prompting using the teacher traces could serve as a meaningful comparison.

- Narrow domain. There are no out-of-domain or cross-task generalization tests. All the distilled traces are also on the same task. It’s hard to say how this would generalize as a universal tool for language model social reasoning.

### Questions
Can you think of other tasks your framework can generalize to?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
CogFlow introduces a framework for aligning large language models with human social reasoning by grounding their thinking process in cognitive theory. Instead of optimizing only for correct answers, CogFlow models the reasoning path as a structured sequence of six cognitive units—Observation, Attribution, Motivation, Regulation, Efficacy, and Behavior—adapted from Bandura’s social cognitive framework.

The system first uses DeepSeek-R1 to generate raw reasoning traces and then filters and ranks them using comparative preference ranking (CPRank²). Each reasoning flow is scored on three theory-based metrics—coherence, interpretability, and predictability—by LLMs acting as judges. The top-ranked reasoning samples become the reference set for training a process-level reward model (RMϕ) that approximates DeepSeek-R1’s and human-like preference judgments.

During reinforcement learning, CogFlow is fine-tuned with a multi-objective reward combining response quality, cognitive-unit diversity, and reasoning-length regularization. The length reward (RLen) enforces concise but complete reasoning, while diversity (RDiv) promotes balanced use of different cognitive units. The final reward encourages structured, clear, and human-like social reasoning processes.

Compared with baselines such as Direct and Distilled-R1 models, CogFlow achieves higher automatic and human evaluation scores. Human judges moderately agree with the LLM-based rankings (κ ≈ 0.47) and consistently prefer CogFlow’s reasoning for coherence and clarity. CogFlow reasoning also improves human decision accuracy in behavioral tasks, showing that process-level training yields reasoning that is more efficient, interpretable, and aligned with human cognition.

### Strengths
- This paper introduces a novel framework inspired by cognitive science to enhance LLMs’ reasoning ability in social situations. By decomposing reasoning into distinct cognitive units, the framework enables fine-grained tracking and optimization of process-level rewards.

- The study systematically compares not only different model architectures but also multiple training paradigms—such as Supervised Fine-Tuning (SFT), Generalized Reinforcement Preference Optimization (GRPO), and distilled reasoning models—demonstrating that CogFlow achieves consistently superior performance under full reward configurations.

- The proposed model defines a comprehensive reward structure that integrates both response-level and process-level rewards. Through a trained reward model and auxiliary cognitive metrics, CogFlow provides more structured and interpretable supervision of the reasoning process.

- Human evaluations are incorporated to validate the alignment between the trained reward model and LLM-as-a-judge assessments, offering partial evidence of cross-validation between machine and human preference signals.

- The paper includes detailed ablation studies on key reward components, such as reasoning-length and diversity rewards, confirming their individual a

### Weaknesses
- The paper does not include a direct RLHF baseline where the model is trained on **human preference data**. Although human evaluations are used post hoc to validate the **LLM-as-a-judge** results, all reward models are trained exclusively on **LLM-generated preference labels** (from DeepSeek-R1 and Qwen-3-32B). This design raises concerns about **self-referential training–evaluation circularity**, since both the reward signal and the evaluation depend on similar model families. A more robust comparison would be to train a model directly using human preference data via RLHF and evaluate it using the same CogFlow reward models. This would clarify whether CogFlow’s improvement truly comes from the **cognitive-unit decomposition and process-level reward design**, or merely from alignment with its LLM evaluators.

- The evaluation metrics, while diverse, remain largely **subjective**. Although the framework defines cognitive units such as *Observation* and *Attribution*, their assessment still relies on qualitative judgments (e.g., “coherence” or “interpretability”) scored by humans or LLMs without explicit operational definitions. To strengthen the theoretical contribution, the authors could further formalize what constitutes high-quality reasoning within each cognitive unit. For example, the “Observation” unit could be evaluated by checking whether key situational elements are accurately identified and semantically consistent. While such grounding would require substantial additional work, it would make the framework more **reproducible, interpretable, and empirically falsifiable**, turning subjective evaluations into observable reasoning diagnostics.

### Questions
I currently do not have any questions.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a novel paradigm called Cognitive Reasoning (CR), which aims to address the inherent limitations of Large Language Models (LLMs) when dealing with complex social situations. The authors argue that traditional logical reasoning paradigms (e.g., Chain-of-Thought, CoT) are suitable for verifiable questions with clear-cut answers, but fail to cope with the ambiguity, interpretive processes, and non-deterministic outcomes common in social contexts. The CR paradigm mimics human social cognition by structuring the interpretive process into a flow of interconnected cognitive units (e.g., observation, attribution). These units combine adaptively to facilitate effective social thinking and responses.

### Strengths
1. The topic is interesting and important. Uncertainty exists in social reasoning scenarios a lot. 
2. The authors propose their own theory, cognitive flow, which consists of six core cognitive units. And the model can adaptively select a better cognitive reasoning path in a tree structure. Classic but cool idea!
3. The authors also contribute a generated dataset.
4. They conducted comprehensive experiments to show the superiority of their model.
5. The figures to show transition patterns are nice. We get some helpful insights into what's going on inside the model.

### Weaknesses
1. The six core cognitive units do not directly come from the reference Bandura 1986. So the validity of the proposed theory needs to be further examined and clarified. You use nouns to name the six core cognitive units -- why not using verbs? And, "Efficacy" seems to be a principle, not a cognitive process. A rational agent plans effectively and efficiently. But in your definition, "efficacy" is not about effectiveness, but rather the belief in executing a social behavior, which confused me a lot. Same problem when you define "motivation". I think the way you define such cognitive terms is very inaccurate.  This substantially diminishes the paper's significance, since it's a problem with the base.
2. When you craft your dataset, you use R1 all the time, with no human verification. 
3. How is the next unit selected? LLM again? 
4. I believe there are many papers about social reasoning with LLM nowadays. Some use only textual input, some even use videos. I didn't see any discussion or reference to these papers, nor comparisons with their LLM architectures. 
5. "Cognitive reasoning" paradigm is not new and not invented by you. Overclaimed. Please study the literature in depth. What is the core difference between your method and these existing methods?
6. Better to show some qualitative results. 
7. Why do you choose to use data from Reddit? Will this make the dataset biased?
8. [L180-181] next cognitive units should be k+1.
9. Why do you choose to use R1, not GPT?
10. What are the "5 major categories and 16 subcategories" in detail?
11. [Table 1] kappa values are not high - which shows the problem of the R1-generated dataset.
12. [Figure 6] what does the red color mean?

### Questions
See the above.

### Soundness
3

### Presentation
2

### Contribution
3

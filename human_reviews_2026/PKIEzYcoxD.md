# ToMPO: Training LLM Strategic Decision Making from a Multi-Agent Perspective

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Large Language Models (LLMs) have been used to make decisions in complex scenarios, where they need models to think deeply, reason logically, and decide wisely. Many existing studies focus solely on multi-round conversations in social tasks or simulated environments, neglecting the various types of decisions and their interdependence. Current reinforcement learning methods struggle to consider the strategies of others during training. To address these issues, we first define a strategic decision-making problem that includes two types of decisions and their temporal dependencies. Furthermore, we propose **T**heory **o**f **M**ind **P**olicy **O**ptimization **(ToMPO)** algorithm to optimize the perception of other individual strategies and the game situation trends. Compared to the Group Relative Policy Optimization (GRPO) algorithm, ToMPO enhances the LLM's strategic decision-making mainly by: 1) generating rollouts based on reasoning the strategies of other individuals, 2) estimating advantages at both the graph-level and sample-level, and 3) balancing global and partial rewards. The ToMPO algorithm outperforms the GRPO method by 35% in terms of model output compliance and cooperative outcomes. Additionally, when compared to models with parameter sizes 100 times larger, it shows an 18% improvement. This demonstrates the effectiveness of the ToMPO algorithm in enhancing the model's strategic decision-making capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces Theory of Mind Policy Optimisation (ToMPO), a reinforcement learning framework for improving LLMs’ strategic decision-making in multi-agent environments. Agents make two interdependent decisions per round, graph-level (who to connect with) and effort-level (how much to invest). The authors extend two games, the BCZ network-effort game and the Public Goods Game (PGG), into sequential multi-agent settings. ToMPO combines:

1. Supervised fine-tuning on expert “Program-of-Thought” data from GPT-o3, and
2. Reinforcement fine-tuning against GPT-o3 agents, optimising both sample- and graph-level advantages.

### Strengths
1. Novel framing of multi-agent strategic reasoning: The paper identifies an under-explored LLM capability, reasoning about sequential, interdependent decisions where individual and group incentives co-evolve.
2. Conceptually interesting algorithm: ToMPO attempts to extend PPO/GRPO by introducing Theory-of-Mind style credit assignment, rewarding decisions that align both with self and others’ inferred strategies.
3. Clear hierarchical decomposition: Splitting decision optimisation into forward (effort) and inverse (graph) processes provides an interpretable mapping between short-term utility and long-term social structure.
4. Relevance and timeliness: The work fits within the community’s growing interest in LLMs as agents, multi-agent RL, and emergent social behaviour.

### Weaknesses
1. Unclear empirical gain: It is difficult to see where the claimed improvement comes from. In several tables, the ToMPO-trained Qwen model performs similarly or worse than the backbone baseline on key metrics, except for U1 (compliance). The observed increases in U2 or U3 could plausibly result from better formatting or rule adherence rather than genuine strategic improvement. Further ablations on this are required. 
2. Ambiguous source of improvement: There are no ablations isolating which components, graph-level advantage, Theory-of-Mind rollouts, or memory rewards, drive the results. Without this, the causal mechanism behind ToMPO’s reported gain is speculative.
3. Lack of clarity and transparency: Important definitions and settings (e.g., BCZ-1/2, “heterogeneous” vs “homogeneous” agents, ToMPO weighting) appear late or only in appendices, making it hard to reconstruct the experiment pipeline.
4. Dependence on GPT-o3: GPT-o3 both supplies the expert data and plays every opponent during reinforcement training. As a result, the learning signal effectively trains Agent-0 (Qwen) to imitate GPT-o3, not to develop independent strategic reasoning.
5. Metrics emphasise compliance, not reasoning: U1–U3 measure formatting validity and numerical optimality, not whether the model actually reasons about others’ incentives or adapts over time. A model could score well without genuine Theory-of-Mind capabilities.
6. Writing and presentation.

### Questions
1. BCZ-1 and BCZ-2 are not formally introduced in the main body of text.
2. The difference between BCZ and PGG and what they uniquely offer required further detail.
3. Only upon looking at the prompt did it become clear to what each agent knew and didn't know during the game play. The key information should be included in the main body of text. 
4. Were preliminary tests run in self-play or versus GPT-o3 aswell?
5. Did you attempt few-shot prompting before SFT to improve compliance in the weaker models?
6. Are results statistically significant?
7. Its not clear to me exactly how ToMPO captures others’ *mental states* beyond observing their actions. Could you please provide further clarification on this?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ToMPO, a novel reinforcement learning algorithm for training LLMs in strategic multi-agent decision-making settings. The authors formalize a graph-effort decision process where an agent must first decide on social connections (graph decision) and then make resource allocation or strategy decisions (effort decision). ToMPO leverages a multi-agent "theory of mind" perspective via agent rollouts, dual-level credit assignment (graph- and sample-level advantages), and a custom reward design that balances global and individual outcomes.

### Strengths
Introduces a novel class of multi-agent LLM decision problems involving dynamic social graphs and sequential choices.
ToMPO creatively incorporates multi-agent reasoning into policy gradients via dual-level advantage estimation.
A well-designed reward structure that balances local and global goals effectively stabilizes RL fine-tuning.

### Weaknesses
The paper lacks confidence intervals or variance estimates. Please report 95% confidence intervals over multiple seeds. Statistical significance is critical for validating the results. I may raise my score substantially if the reported gains are statistically robust across diverse setups.
No ablation studies to disentangle effects of key components (e.g., graph-level advantage, ToM rollouts).
Depends on expert data for SFT; unclear how performance degrades with weaker or fewer demonstrations.
Baseline comparisons are limited.
Evaluation remains in custom environments; unclear generalization to more realistic or diverse strategic tasks.
Dense notation and undefined evaluation metrics slightly hinder accessibility.

### Questions
How sensitive is ToMPO’s performance to the quality or quantity of expert demonstrations? Could it work with noisy or imperfect experts?
Can you report ablations for ToMPO’s components (graph-level advantage, rollout reasoning)? Which parts are most critical?
Would your method generalize to agents trained symmetrically (not just Agent 0)? Could one model handle all agents?
How was the 0.8/0.2 reward weighting chosen? Were other hyperparameter combinations tested?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides a variation of GRPO that factors in ToM for more effective strategic decision-making. It is well-motivated, rigorous, and provides significant improvements in the areas where it is applied (namely, cooperative settings).

### Strengths
1. Original contribution with strong numbers and improvements over baselines.
2. Well-founded, rigorous work.
3. Evaluation includes a mixture of open-source and closed-source models.

While these are short sentences, I need to emphasise that this is an original, exciting contribution. These strengths make the paper very robust--provided that the two major weaknesses below are addressed. 
In my opinion, employing ToM for reinforcement learning is a good idea that can make good contribution to MAS. What matters from this paper is that there are multiple experimental settings in which ToMPO is shown to work.

### Weaknesses
In terms of contributions, this paper is strong. However, it is very poorly written, and the reproducibility of the work is only partial. Both are major--but not impossible to fix--weaknesses.

Writing:
- The flow of the paper needs to be improved: there are a lot of places where things are introduced before/after/referred to without a logical succession flow. For example, the paper starts by framing the problem as (L84): 'In contrast to the scenarios discussed in Theory of Mind': which scenarios? If you have a fully-fledged formal definition of your scenario (L91+) it would make sense to compare both the scenarios and your reframing.
- Overall the paper can be made far more readable by moving the notation to the appendix, and the related works section back to the main body. It defines names that are used but not defined in the main body of the paper, and provides framing for your entire research (see, e.g., my questions from above).
- The main contribution (L72) indicates the introduction/definition of a (new) problem for decision-making. However, the entire work relies on an abstract notion of the data. Given that LLMs are sensitive to natural language, explaining _what_ this problem is (or adding examples of the datapoints) in the main body of the paper would be helpful; and likely will require rephrasing L72.
- The framing takes about eight pages, and the results about one page. There is no discussion of the work, or conclusions drawn. As it stands, it reads more like a technical report, rather than work following the scientific method.
- Do you need the definition of LoRA (L273)?

Minor:
- There are plenty of sentences that may be simplified to improve the flow of the work (e.g., L54, repeat 'cooperation', L63, repeat 'performance').
- Abbreviations are introduced and re-introduced often, including obvious ones like 'Large Language Models'. Sometimes they are introduced backwards (L62; 'State-of-the-art (SOTA)' should be backwards and 'State' is not capitalised)
- Dangling parenthesis in L43
- L110: equation -> equations
- L116: 'these two processes align with the credit alignment principle' (and a referral to an equation). Since CAP refers to an equation, it is better to emphasise it ('the _credit alignment principle_ (eq. X)' or 'the credit alignment principle from eq. X'). This will vastly improve readability since there are plenty of definitions thrown around.
- C.2 tables: maintain consistent capitalisation. Variables need to be explained.
- B: the algorithm needs an explanation in natural language
- Significant digits are inconsistent (see, e.g., Table 2)
These are just examples: the paper needs further, thorough proofreading

_Reproducibility_: while there is pseudocode and prompts, there is no code released even though this is an algorithm that relies on stochasticity. Code release is important in this case, because--without formal proofs of correctness--the only other way to ascertain that this paper does succeed at what it claims is by examining the code, including whether any random seeds were set.
Since this relies on synthetic data generation (which was admittedly skewed, as per the authors), and it is claimed as part of the contributions (L74), this should be released as well. I'm fairly certain that, since the authors make references to the 'associated code implementation', this was solely an oversight on their part.

As I said, the paper's technical contributions are solid. In its current state, it is not of sufficient quality and the lack of reproducibility does put some doubts in this work's trustworthiness, which limits the reviewer's assessment of the work. However, I also believe that the drawbacks could be very easily addressed.

### Questions
L468 -- 'compared to a model with 100 times the number of parameters' -- does this mean not the comparison between backbone and finetuned Qwen? Which model is this?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper aims at improving LLMs' strategic decision making capability from a multi-agent perspective, which also consider decisions made by other agents. They first define a problem for real-world strategic decision-making and provide an environment as test bed. Then they evaluate the performance of current SOTA models with the environment, which also yields a dataset of strategic decisions made by the SOTA models, enabling further imitation learning for a smaller model. Then they propose a reinforcement learning algorithm called ToMPO (Theory of Mind Policy Optimization) to train a Qwen-2.5-7B-instruct. Results show that, with ToMPO, their trained agent is particularly good at achieving global welfare.

### Strengths
1. This paper addresses an important research question that how do we train an agent to make better decisions in a multi-agent environment. More specifically, better cooperative decision making. The paper tackles this problem from the perspective of considering strategies of others
2. Training with the method proposed by the paper achieves a good performance, even surpassing a much larger baseline.

### Weaknesses
1. Lack of case study. For example, presenting a decision-making trajectory could provide richer context and help readers better understand the problem.
2. Lack of in-depth analysis. For example, how are the hyper-parameters (w_{local} and w_{sample}) chosen? The reward design appears to be complex; providing further analysis could offer deeper insights into the design.
3. Lack of experiments. Applying the method to a single model seems to be insufficient. Does the current conclusion also hold for other models?

### Questions
1. Is the design specific to the two tasks evaluated in the paper, or can it be applied to a broader range of tasks? Providing an example that links it to other realistic cooperative multi-agent scenarios could help clarify the context.
2. Is it possible to include other MARL baselines, other than simple IO prompting?

### Soundness
4

### Presentation
3

### Contribution
3

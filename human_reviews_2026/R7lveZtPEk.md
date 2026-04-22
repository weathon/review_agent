# LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling

- Avg Score: 5.33
- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences.
As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model’s responses are not only high-quality but also grounded in and consistent with the provided context.
Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context–response consistency.
In this work, we introduce \texttt{Long-RewardBench}, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks.
Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments.
Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs).
Experiments show that our approach not only substantially improves performance on long-context evaluation, but also preserves strong short-context capability
Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission introduces Long-RewardBench, a new benchmark for evaluating RMs under long-context scenarios (up to 128K tokens) — a setting not covered by prior work such as RewardBench, to the reviewer's knowledge. It also proposes a multi-stage training strategy to construct “LongRMs” that retain short-context performance while improving long-context consistency.

The benchmark covers Pairwise and Best-of-N tasks and synthesizes ground-truth preferences automatically using task-specific metrics and reasoning explanations from LLMs. The training pipeline consists of SFT and RL Alignment, with data synthesized by Consistency Majority Voting

Empirical results show that existing GenRMs drop to near-random performance as context grows beyond 4K tokens (with potential caveats). LongRMs trained by the proposed method recover substantial accuracy, sometimes surpassing 70B models and approaching proprietary Gemini 2.5 Pro performance.

### Strengths
Overall the reviewer feels the paper might be a timely contribution with the following strengths:

1. A timely contribution. To the reviewer's knowledge, the paper identifies a meaningful gap in RM evals - long-context reasoning and alignment (e.g., for agents) indeed require new evaluation setups.

2. The benchmark pipeline looks sensible, with considerations such as explicit sampling, task balance, and automatic metric-based preference generation. The use of both Pairwise and Best-of-N settings are reasonable.

3. The Short-to-Long Data Synthesis and Consistency Majority Voting pipelines look interesting, combining ideas from self-distillation and DPO-style preference optimization.

4. The final results look sensible - small (8B) models can match much larger ones in long-context RM tasks.

### Weaknesses
The paper has the following weaknesses. Overall several technical details do not seem clear or convincing, and given the nature of the work (RM benchmark and RM model contribution), the contribution is uncertain without a clear access to artifacts.

1. The reviewer is not fully convinced by the "ground-truth", which is critical for evaluations. The “ground-truth” judgments are derived from automatic metrics (e.g., ROUGE-L) or LLM-based explanations. This introduces potential label bias and leakage: since strong LLMs are used both for generation and labeling, it is unclear how much of the benchmark reflects true human-aligned preference rather than model self-agreement. 

2. Uncertainty of broader impact. The paper claims public release “upon publication” but does not yet provide leaderboard or dataset URLs. Compared with RewardBench, Long-RewardBench’s openness and ease of reuse remain uncertain. In fact, the evaluation of artifacts is required for relevant tracks, e.g., in NeuraIPS. ICLR does not have such a specific track, but the reviewer believes the reviewing criterion still applies.

3. Some numbers and analysis look suspicious.The claim that Llama-3.3-70B-Instruct achieves 0 % accuracy (random = 50 %) suggests possible implementation or labeling issues. If it is a matter of setup issues, or applying models in scenarios that do not fit, the analysis is not convincing or very solid.

4. Several technical details and arguments look too vague. For example, the conceptual vagueness of “faithfulness” in SFT objective.
The loss function is the standard SFT objective; no explicit term enforces context grounding beyond data synthesis. Clarifying how “faithfulness” is computed or measured would strengthen the technical rigor.

5. Overall the technical depth of the paper is not deep. The method is primarily procedural and empirical, there is minimal analysis of why long-context alignment fails theoretically (e.g., attention decay, token position bias, gradient vanishing).

### Questions
Please refer to detailed reviews above.

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
This paper investigates the limitations of existing short-context reward models (RMs) in long-context scenarios and proposes a multi-stage training framework to address this issue. The authors begin by introducing Long-RewardBench, which contains responses exceeding 4k tokens and covers a wide range of tasks. Using this benchmark, they demonstrate that existing RMs struggle to score preferences accurately when the context becomes long. To mitigate this, they propose a multi-stage training framework, which consists of a Cold Start SFT stage followed by an alignment stage via DPO. Experimental results show that their method is effective in improving reward modeling capabilities in long-context scenarios and is applicable to both generative and discriminative RMs.

### Strengths
1.  The paper addresses a timely and significant challenge: improving the capability of long-context reward modeling, which remains a key frontier in current research.
2.  The proposed training method demonstrates clear effectiveness, particularly when applied to generative reward models.
3.  The method is technically sound and supported by a well-articulated motivation.
4.  The experiments are comprehensive, and the results show significant improvements over the established baselines.

### Weaknesses
1. The method is less effective when applied to the discriminative RMs, where the performance of several tasks (e.g., Math) is degraded after the alignment stage.
2. The method may introduce length bias; for instance, the performance is less stable on the RewardBench.
3. It is suggested to make the description of the method clearer. For example, why do you need two task formats during training, and why is the pairwise comparison not sufficient?

### Questions
1. Consider the following scenario, where the chosen response is short, and the rejected response is lengthy and verbose. What is the performance of the RMs after alignment?
2. In Table 1, why is the Rank 2 performance less than the random guess performance, while Rank 3 and 4 are higher?
3. The paper has mentioned that the inconsistency between the judgment and explanation might affect the RM's performance. Then what if the RM is enforced with a short explanation or simply outputs the verbal label without explanation?
4. What if we first remove redundant information (like the data curation process in the Cold-Start SFT stage) from the long context and then let the short-context reward model judge the preference?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a benchmark dataset that measure the performance of reward models in the long context setting. Using this benchmark, the authors measured the long-context performance of existing reward models. They find that existing reward models do not perform well in a long context setting. 

The paper then tried to use standard context extension approaches to extend the context length of existing reward models, but find that the performance of such extended models are not well either on their benchmark datasets. The authors further investigate why these models perform poorly and identify two major reasons. 1. failure of following instructions and understanding long-context content; 2. inconsistency between judgement and explanation.

Motivated by these error patterns, the paper propose a training recipe to improve long-context reward model performance. The recipe targets the error patterns identified. Experiments show improvement.

### Strengths
originality: in terms of the training recipe proposed by the paper, the use of supervised fine-tuning and reinforcement learning is non-surprising. nonetheless, the recipe did seem to tailor to the error patterns identified previously. The design of recipe that tailored to the error patterns can be viewed as something original.

quality: the authors supports the points raised in the paper via various empirical evidence. This provide a certain level of assurance to the correctness of the work.

clarity: presentation of the paper is clear.

significance: the paper provides an approach to scale existing reward models to long-context ones.

### Weaknesses
originality: is there any justification/intuition why the training recipe needs to follow a first SFT then RL type of approach?

quality: I think I am not fully convinced with some of the choices made in the paper.
1. when deriving a dataset, why scoring model response with a task-specific automatic metric provides an accurate enough ground truth?
2. line 318: how do we know the predicted scalar value is more consistent with model explanation compared to the inconsistency in a pairwise setting?

significance: is there any justification/intuition why it is important to improve the long-context capabilities of reward models? With the continual development of LLMs, one may expect that models with longer context  will be (or have already been) used as a reward model.

### Questions
1. what are the task-specific automatic metrics that you used?
2. how do you conduct error analysis in line 200?

### Soundness
2

### Presentation
3

### Contribution
2

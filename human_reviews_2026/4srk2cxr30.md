# Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 6, 4

## Abstract
Recent advances in Large Language Models (LLMs) have enabled strong performance in long-form writing, but current training paradigms remain limited: Supervised Fine-Tuning (SFT) remains constrained by data saturation and performance ceilings, while Reinforcement Learning with Verifiable Reward (RLVR), though successful in verifiable domains like math and code, cannot be directly migrated to open-ended long-form writing due to a lack of ground-truths. To further advance long-form writing, we present Writing-RL: an Adaptive Curriculum Reinforcement Learning framework to advance long-form writing capabilities beyond SFT. 
The framework consists of three key components: Margin-aware Data Selection strategy that prioritizes samples with high learning potential, Pairwise Comparison Reward mechanism that provides discriminative learning signals in the absence of verifiable rewards, and Dynamic Reference Scheduling approach, which plays a critical role by adaptively adjusting task difficulty based on evolving model performance. Experiments on 7B-scale writer models show that Writing-RL effectively improves long-form writing performance over strong SFT baselines. Furthermore, we observe that models trained with long-output RL generalize surprisingly well to long-input reasoning tasks, potentially offering a promising perspective for rethinking long-context training.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work focuses on the long-form writing task. To overcome the limitations of supervised fine-tuning (SFT), the authors propose Writing-RL, an adaptive curriculum reinforcement learning framework. The framework incorporates three main components: (1) a margin-aware data selection strategy that prioritizes samples with high learning potential; (2) a comparison-based reward function that evaluates generated responses against reference texts; and (3) a dynamic reference scheduling mechanism that adjusts task difficulty over the course of training. Experimental results demonstrate the effectiveness of the framework and provide interesting insights regarding long-context input tasks.

### Strengths
1. Long-form text generation is an important yet relatively under-explored research problem.

2. Leveraging reinforcement learning to surpass the capacity limits of SFT is well-motivated and methodologically sound.

3. The curriculum design is intuitive and reflects principles similar to human learning behavior.

### Weaknesses
1. The paper claims that reference texts used during margin-aware data selection are obtained from “competitive models.” However, in the experiments, the authors employ several “competent larger models” for this role. This raises concerns about whether the observed improvements primarily stem from stronger teacher models rather than the proposed framework itself. In other words, the performance gains appear to depend heavily on the strength of the teacher model.

2. The framework relies on LLM-as-a-judge both (1) to perform pointwise evaluation for reference quality during data selection and (2) to provide pairwise comparison rewards during RL training. However, evaluating long-form text quality is inherently difficult, and LLM-based judgments are known to be unstable and sometimes inconsistent. Given that the reward signal may be noisy or unreliable, it is unclear how training stability is maintained or guaranteed.

### Questions
1. How sensitive is the final model performance to the choice of “competitive” (teacher) models used during margin-aware data selection? Would weaker teacher models lead to significantly worse results?

2. How do the authors ensure the reliability and consistency of multi-dimensional pointwise evaluation via LLM-as-a-judge, particularly for long-form outputs?

3. Given the known stochasticity and instability of LLM-based reward evaluations, how is reward noise mitigated to ensure stable and effective RL training?

4. Can the author explain more how to avoid the instability and randomness of using LLM-as-a-judge for generating the reward for long-form writing?

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
4

### Summary
The paper presents a suite of techniques to improve LLM performance on a suite of writing benchmarks. The main methodological contributions are (a) selecting training examples by prioritizing examples where frontier LLMs strongly outperform the model being tuned (margin-aware selection strategy), (b) creating a pairwise reward that compares the model-generated response to the reference in deciding a score of {0, 0.5, 1}. (c) dynamically updating the references used for training examples based on the quality of model-generated outputs. Overall, as an ML problem, the authors provide sufficient evidence for the benefit of their modifications in improving benchmark performance. My main concerns are around qualitative evaluation (does the text appreciably change in style or content due to RL training, given that the improvement is only a few percentage points) and details about the human evaluation conducted - I think these are particularly important due to the nature of writing as a human-centered task.

### Strengths
1. Each individual method to improve performance is well motivated and intuitive, and their proposed RL training recipe leads to higher performance on both in-domain and (slightly) OOD benchmarks. 

2. The analysis that RL generalizes but SFT degrades in the OOD setting is valuable and adds to the growing body of similar findings in the literature. 

3. The authors perform careful ablation of many of their design choices in data selection, reward design and reference ordering (Tables 4, 5, 6), and it is expected/natural that the model faces a slight degradation in general capabilities due to the targeted fine-tuning (Table 7).

### Weaknesses
1. The verification of the pairwise LLM-as-judge reward is not sufficiently well documented. How many judges score each example? What is the inter-annotator agreement? Do models agree more highly on any particular label (i.e. are they less reliable on marginal 0.5 decisions)? Are models sensitive to the length of the responses being compared?

2. In Table 2, the SFT and RL models strongly outperform the baseline (unfinetuned) Qwen and Llama models in the in-domain setting. But the drop-off in generalization in Table 3 is informative. SFT is _worse_ than the base model, and RL is slightly better. Was the human eval in Section 4.5 on examples from ID or OOD benchmarks? I think we need both to get a clear picture, as well as a comparison to the base model. Also, what were the reasons behind human decisions? Were the outputs qualitatively different, or was it an improvement in faithfulness of content/coherence?

3. For a subjective task like writing, I think an open-ended case study is needed for an ecological evaluation of performance. For instance, while a benchmark score might improve by a few percentage points, when presented with suggestions from the different models, do writers actively prefer the RL-tuned model? This is different from the intrinsic human-eval in Section 4.5, which tests pairwise preference on a fixed set of examples. What I'd want to see is how much users prefer outputs from the RL-tuned model in the case where they are writing on their own and not 'judging' outputs. This could partially be mitigated by some qualitative analysis on the actual text produced - I might have missed it, but I didn't see any detailed examples of outputs from the different methods showing this. 

These all seem largely fixable so I'm happy to engage during author response.

### Questions
1. To make the paper more self-contained, it would be helpful to include descriptions of the particular writing tasks. Also, since the tasks are created with model generation + human refinement, is it representative of the true distribution of writing tasks in terms of diversity? How is LongBench different from WriterBench? Should we expect the drop-off in performance in SFT in Table 2?

2. L.322 - based -> base, though based is unintentionally funny here

3. It would be helpful to respond to recent work that shows that LLM-as-judge is unreliable as a reward for writing [1] and that these LLM-rewards can lead to reward hacking in writing [2]. 

[1] Chakrabarty, Tuhin, Philippe Laban, and Chien-Sheng Wu. "Ai-slop to ai-polish? aligning language models through edit-based writing rewards and test-time computation." arXiv preprint arXiv:2504.07532 (2025).

[2] Pan, Jane, et al. "Spontaneous reward hacking in iterative self-refinement." arXiv preprint arXiv:2407.04549 (2024).

### Soundness
2

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
This paper proposes an adaptive curriculum RL framework to improve long-form writing beyond SFT. The method has three components: Margin-aware data selection: select training prompts where the policy’s response lags the best competing reference by the largest margin (learning potential), estimated with multi-model generation and LLM-as-judge scoring. Pairwise comparison reward: reward the policy when its response beats a reference according to an LLM judge (win=1, tie=0.5, loss=0), with the model’s output deliberately placed in the second position to counter position bias. Dynamic reference scheduling: for each prompt, maintain an ordered list of references (increasing quality) and “level up” to a stronger reference whenever the policy beats the current one. The empirical experiment results demonstrate better performance than the baselines.

### Strengths
- Empirical results are good, consistent gains on three writing benchmarks across two bases, competitive with proprietary systems.


- Human evaluation corroborates automatic metrics (100 pairwise comparisons).


- Interesting generalization from long-output RL to long-input reasoning on LongBench v2.

### Weaknesses
- Training rewards hinge on one judge (Qwen-Plus). Although different judges are used at eval time, the training-time reward could encode judge-specific preferences. Agreement with humans is reported but limited in scope.

- Only ~1.5k prompts per source are used for RL after filtering; more detail on topic coverage and potential overlap with evaluation prompts would help assess overfitting risks.

- Tables lack variance or significance tests; it is unclear whether deltas (often 1–2 points) are statistically reliable across seeds.

- Compute and cost. The method involves multi-model reference generation and LLM judges during training; however, the training-time cost/latency is not quantified, making reproducibility/industrial adoption harder to gauge. (Setup described, but cost analysis missing.)

### Questions
- How does performance change if you (a) ensemble multiple judges for rewards, (b) randomize judges per batch, or (c) periodically swap the judges? Does this reduce bias and further improve generalization?

- What are the reference generation and reward-computation costs (GPU hours, $) for your typical run? A breakdown would clarify practicality.

- Ablate curriculum vs RL: If you keep SFT optimization but use dynamic reference scheduling as hard-negative distillation (no RL), how much of the gain remains? Conversely, PPO with pointwise rewards but no scheduling?

- Generalization controls: Can you report: (i) length-matched SFT (same output lengths as RL rollouts), (ii) pairwise-reward RL without margin-aware selection, and (iii) dynamic scheduling with static pointwise rewards? 
- What does the color represent in Figure 2?

- In line 87, the policy model response and the highest-quality reference, how to define the high-quality references? If we use some low-quality references from some models, how about those results?

### Soundness
2

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
3

### Summary
This paper proposes Writing-RL, an Adaptive Curriculum Reinforcement Learning framework to solve the open-ended long-form writing problem. The framework is built on three key ingredients: (1) A data selection strategy that prioritizes samples with the largest learning potential, which is the score difference between the current policy and the highest-quality reference, (2) A reward function that give rewards via pairwise LLM-as-judge comparison against a high-quality reference, and (3) A dynamic scheduling strategy that assigns reward references of increasing difficulty during training. Writing-RL outperforms a range of strong SFT baselines across all three benchmarks. More interestingly, an “output-to-input” generalization is observed, where models trained with long-output RL also generalize surprisingly well to long-input reasoning tasks.

### Strengths
The paper is well motivated and clearly scoped. The writing is well-organized and easy to follow.

The experimental section is comprehensive, solid, and convincing. The authors evaluate against a wide range of competitive baselines across three writing-focused benchmarks, and further include human evaluations to validate the model’s performance. They also report their method’s impact on the general capabilities of LLMs. Finally, the paper contains extensive ablations on curriculum scheduling variants, reward design, and data selection strategies.

The paper also handles experimental details carefully and fairly. For example, it explicitly controls for position bias and justifies the choice of the judge LLM, which increases the credibility of the reported gains.

### Weaknesses
A potential weakness is the cost of the overall pipeline. All three stages, data selection, reward assignment, and dynamic scheduling, rely on repeated LLM-as-judge calls, which could make training expensive.

### Questions
Algorithm 1 replaces a reference with a stronger one as soon as the model wins once. Did you try requiring k wins before promotion? Given LLM-judge variance, some frontier references may be replaced prematurely (i.e., the model only won once by chance), and the next reference may be too hard, making it impossible for the model to beat it afterward.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Writing-RL, an adaptive-curriculum reinforcement-learning framework aimed at improving long-form writing capabilities in large language models. The method is composed of three key components: (i) margin-aware data selection, which prioritizes training examples where the policy model lags significantly behind stronger references; (iii) pairwise comparison reward, which provides more discriminative feedback without verifiable ground truths; (iii) dynamic reference scheduling, which gradually increases task difficulty by upgrading reference quality once the policy surpasses the current baseline. 
Empirical results on two backbone models(Qwen2.5‑7B-Instruct and Llama3.1‑8B-Instruct) demonstrate meaningful gains across several long-form writing benchmarks (WritingBench, EQ-Bench, LongBench-Write). The authors additionally observe an unexpected cross-domain effect: models fine-tuned via Writing-RL on long-output generation also show improved performance on long-input reasoning tasks (LongBench-v2), suggesting that RL for long-form generation may also enhance long-context comprehension.

### Strengths
1.	The paper tackles long-form writing, a task space where reinforcement learning has struggled due to unverifiable rewards and instability. Extending RL beyond verifiable domains like math or code to open-ended text generation is both challenging and impactful, positioning this work as one of the first systematic studies of RL in long-form writing.
2.	The proposed Writing-RL framework unifies Margin-aware Data Selection, Pairwise Comparison Reward, and Dynamic Reference Scheduling into a cohesive adaptive curriculum. Each addresses a different pain point, data utility estimation, reward discriminability, and difficulty calibration, creating a more stable RL loop for non-verifiable tasks.
3.	The observation that long-output RL improves long-input reasoning on LongBench v2 (Table 3) is insightful. It hints at a potentially deeper connection between generation-length and context-length generalization, offering a new perspective for future long-context training research.

### Weaknesses
While the paper presents an interesting direction, several aspects limit its overall contribution and clarity:

1.	Incomplete technical specification.
Many important implementation details are missing, especially in the PPO setup, such as how the value function is parameterized (whether a model like AQuarterMile/WritingBench-Critic-Model-Qwen-7B is used?), whether reward normalization is applied, and what decoding temperature settings are used during training and evaluation. Also mentioned that the paper claims Writing-RL out perform strong SFT baselines but provides no hyperparameter details of SFT experiment(e.g., learning rate, number of epochs, optimizer, or scheduler), making it difficult to verify fairness. Without these, it is difficult to assess reproducibility or confirm whether the reported gains arise from the proposed curriculum design or specific training heuristics.

2.	Limited novelty in core components.
Although the integration of margin-aware data selection and dynamic reference scheduling is well-motivated, the reward design resemble ideas from existing llm-as-a-judge works (e.g., RLAIF by Lee et al. https://openreview.net/forum?id=AAxIs3D2ZZ , JudgeLRM by Chen et al https://arxiv.org/pdf/2504.00050 ). The paper would benefit from a clearer justification or ablation isolating their unique contribution in the writing domain. 
3.	Lack of statistical evidence for claimed improvements.
Across multiple ablations (Table 4 for data-selection and Table 6 for curriculum scheduling), the performance gains of the proposed methods are small, often within ~0.5–1.5 point. However, the paper does not report standard deviations. It is therefore unclear whether the improvements of Margin-aware Data Selection over Difficulty-prioritized (86.40 → 87.02) and Dynamic Curriculum over Static/None (84.49 → 83.87/83.15) are statistically significant. The authors should report variance or confidence intervals to substantiate the significance of these differences.
4.	Insufficient discussion of generalization mechanisms.
The “output-to-input generalization” finding is intriguing but currently speculative. The paper lacks controlled experiments to disentangle whether this transfer effect arises from the adaptive curriculum, the LLM-judge preference distribution, or incidental overlap in data distribution.

### Questions
1.	Clarification on PPO Implementation and Value Function: The paper states that Writing-RL adopts PPO for training (Section 4.2, Algorithm 1), but the formulation of the value function or advantage estimation is not specified. Could the authors clarify how the critic or value head is parameterized and trained, and whether any reward normalization or bootstrapping strategy is applied given the discrete {0, 0.5, 1} reward signal?
2.	Generalization to Long-Input Reasoning: The “output-to-input generalization” finding is intriguing. Could the authors discuss whether the improvement on long-input reasoning is purely due to the curriculum RL process, or whether the LLM judge’s preference distribution indirectly regularizes long-context coherence? Quantitative ablations on a smaller scale (e.g., replacing the LLM judge to isolate reward-model bias) would strengthen the claim.
3.	Reward Design and Learning Stability: The paper defines a ternary pairwise reward in {0, 0.5, 1}, where 0.5 corresponds to a tie. Could the authors discuss what would happen if the reward were binary only ({0, 1}), without the tie case (1 if Judge(ref, x) = x ≻ ref, and 0 otherwise)? In particular, how would removing the neutral signal affect training stability, variance of policy gradients, or convergence speed in PPO? Some empirical or theoretical justification for including the tie outcome would help clarify its necessity.
4.	Missing hyperparameter details: The paper omits rollout hyperparameters and learning rate scheduler type. Providing these details would improve clarity and reproducibility.
5.	Ablation without value function (GRPO-style baseline): Since the paper attributes Writing-RL’s improvement partly to its adaptive curriculum design rather than the underlying PPO framework, would the authors consider running an ablation that removes the value function entirely, like GRPO-style optimization, while keeping the same LLM-as-a-judge pairwise reward? This experiment could help isolate whether the gains primarily stem from the curriculum mechanism or from the critic-based advantage estimation.

### Soundness
2

### Presentation
3

### Contribution
2

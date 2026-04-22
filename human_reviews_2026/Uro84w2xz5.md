# Supervised Reinforcement Learning: From Expert Trajectories to Step-wise Reasoning

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4, 6

## Abstract
Large Language Models (LLMs) often struggle with problems that require multi-step reasoning. For small-scale open-source models, Reinforcement Learning with Verifiable Rewards (RLVR) fails when correct solutions are rarely sampled even after many attempts, while Supervised Fine-Tuning (SFT) tends to overfit long demonstrations through rigid token-by-token imitation. To address this gap, we propose Supervised Reinforcement Learning (SRL), a framework that reformulates problem solving as generating a sequence of logical ``actions''. SRL trains the model to generate an internal reasoning monologue before committing to each action. It provides smoother rewards based on the similarity between the model's actions and expert actions extracted from the SFT dataset in a step-wise manner. This supervision offers richer learning signals even when all rollouts are incorrect, while encouraging flexible reasoning guided by expert demonstrations. As a result, SRL enables small models to learn challenging problems previously unlearnable by SFT or RLVR. Moreover, initializing training with SRL before refining with RLVR yields the strongest overall performance. Beyond reasoning benchmarks, SRL generalizes effectively to agentic software engineering tasks, establishing it as a robust and versatile training framework for reasoning-oriented LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors present a method for supervised or assisted RL for LLMs.
The motivation given for the approach is that for difficult tasks, where the LLM rarely reaches the right answer, naive use of RLVR does not help.
Thus, the authors propose to break the process of generating an answer using the LLM into multiple steps.
At each step the LLM is given the problem and a partial answer, and is required to output the only the next step (action) to solve the problem after thinking.
The reward for the computed next step is computed as the normalized edit distance between the generated action and the expert predicted step.
This assistance or supervision using the partially completed solutions and to only focus on generating the next step helps the LLM learn better.

The method requires a dataset or expert generations with an action-based formatting, where each example can be broken into a sequence of actions or steps.
The definition of action can vary with domain and datasets.
For this paper, the authors use step-wise structure of math solutions and environment commands for SWE tasks to distinguish individual actions.
The authors also use dynamic sampling to remove training examples with low variance in scores in all cases including baselines.

The authors show that the method performs well and provides a boost of 3% over RLVR on math tasks (3.7% for SRL+RLVR) on average. While naive RLVR does not yield any improvement, SRL does. SRL also results in 6.4% improvement in the oracle edit task and 4.4% in end-to-end task.

For qualitative analysis, the authors state that the method allows the agent multi-step thinking which manifests as, outlining all steps in a comprehensive initial plan, on-the fly adjustments, and reflective verification. However, only one representative example has been shown for this behavior. The overall answer lengths are also similar to the base-model.

### Strengths
- The method is simple and easy to follow.
- Based on the motivation, the approach of providing the supervision of first few steps and a dense reward makes sense.
- The method provides a boost in performance for math and SWE tasks.

### Weaknesses
- Adding comparisons against other methods that use intermediate rewards can be helpful (maybe process reward model (PRM) methods, e.g. [1]).
- Since the thinking patterns are listed as a contribution, the representative examples showing the interleaved thinking patterns and others can be expanded and analyzed.
- Since the method heavily relies on the data being in a specific format, there should be some discussion of limitations and expansion to other domains.


[1] Lu, J., Dou, Z., Wang, H., Cao, Z., Dai, J., Feng, Y. and Guo, Z., 2024. Autopsv: Automated process-supervised verifier. Advances in Neural Information Processing Systems, 37, pp.79935-79962.

### Questions
- Can the authors provide a comparison to some other self or expert supervised method that rewards intermediate steps?
- Did the authors try other reward schemes than string edit distance (like LLMs)? The same information can be presented in different sentences, and sometimes a slight difference in the text can drastically change the meaning; therefore, the edit-distance heuristic may not always hold.
- The temperature used (1.0) does not seem to be the default to my limited knowledge. If so, how do the average@32 scores change with temperature like 0.6/0.7?
- What was the final prompt used for inference, because the one mentioned in Appendix B states only one thinking step and outputs only one reasoning step (seems to be for training)?

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
3

### Summary
The SRL framework breaks down problems into a sequence of "actions". In each step, the model first generates an "internal monologue" and then executes an "action".

SRL's reward mechanism abandons sparse final-answer rewards and rigid imitation, instead providing dense signals at every step based on the similarity between the model's "action" and the expert's "action".

Crucially, SRL only rewards the "action," maintaining flexibility for the "internal monologue". This design combines dense signals (addressing the RLVR problem) with flexible reasoning (avoiding SFT rigidity) to promote stronger reasoning abilities.

Experiments show that SRL significantly outperforms SFT and RLVR on both mathematical reasoning and software engineering benchmarks. A curriculum learning strategy of "cold-starting with SRL then fine-tuning with RLVR" achieves the best results.

### Strengths
1. It finds a clever balance between SFT and RL. The core concept of "rewarding actions, liberating reasoning" is an important innovation for LLM reasoning training.
2. Comprehensive experiments were conducted on difficult benchmarks in mathematics and software engineering. The superior performance of SRL → RLVR is highly persuasive.
3. In-depth analysis (Section 5.2): High-quality ablation studies demonstrate that fine-grained guidance is key (Table 3). Behavioral analysis shows that the model exhibits more flexible, advanced reasoning patterns, rather than simply increasing output length (Figure 4).
4. Strong generalization ability: Successfully transferred from mathematics to software engineering tasks (Table 4), demonstrating its potential as a general-purpose reasoning framework.

### Weaknesses
1. Dependence on Expert Data and "Action" Definition: SRL is highly dependent on high-quality, easily decomposable expert trajectories and the definition of "actions."
2. Rationality of the Reward Function: The reward function is based on syntax (e.g., difflib) rather than common measures like KL divergence. How can the rationality of this approach be proven?
3. The primary weakness of this paper's methodology is its lack of novelty, as its core idea significantly overlaps with existing work: (1) Unoriginal Theoretical Foundation: Theoretically, the approach of using expert demonstrations to solve the sparse rewards problem is a classic paradigm in traditional reinforcement learning. Techniques such as Reverse Curriculum Learning and Learning from Demonstration have long been explored and validated in related fields. (2) Significant Overlap with Recent LLM Work: In the specific application of LLM reasoning, the proposed SRL framework is highly similar to the recent $R^3$ paper [1]. $R^3$ explicitly explored using expert demonstrations to train the model starting from intermediate states of the demonstration. Through a reverse curriculum, $R^3$ effectively converts sparse outcome rewards into "approximately step-by-step supervisory signals". The SRL paper similarly relies on decomposing expert trajectories into step-wise "actions" (what $R^3$ calls "intermediate states") and constructs partial trajectories for training. The core mechanisms of both methods are fundamentally alike. So, more discussion and comparison should be included.


[1] Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

### Questions
1. Can you clarify the difference with works like $R^3$ in the paper?
2. Can you include more backbone models to make the paper more solid?

### Soundness
3

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
4

### Summary
This paper proposes Supervised Reinforcement Learning (SRL), a new framework that bridges the gap between Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR) for complex reasoning tasks. 
This approach provides richer learning signals than RLVR and avoids the rigid imitation of SFT. 
Experiments on mathematical reasoning and software engineering benchmarks show that SRL significantly outperforms both baselines, and a curriculum of SRL followed by RLVR achieves the best overall results.

### Strengths
1. The novel SRL framework addresses RLVR's sparse rewards and SFT's rigid mimicry by decomposing expert trajectories into step-wise logical actions with dense sequence-similarity rewards, uniquely combining SFT’s supervision and RL’s strengths and showing superior performance on math reasoning benchmarks.
2. SRL demonstrates robust cross-domain efficacy across different multi-step reasoning tasks, proving its versatility.
3. SRL excels in data efficiency for hard problems, using limited expert data to avoid SFT's performance degradation and enabling a high-performing SRL→RLVR curriculum.

### Weaknesses
1. SRL relies on the decomposition of expert trajectories into logical actions without providing details on automation for unstructured trajectories. Therefore, I'm not sure if it is scalable for large-scale or unstructured tasks and whether it introduces subjective bias.
2. The reward only considers action similarity, potentially leading to false positive rewards from flawed reasoning. This impact should be considered.
3. The experiments are conducted solely on small to mid-sized models. It would strengthen the paper to include results or discussion on whether SRL’s advantages extend to larger model sizes, as this would provide a clearer picture of its scalability and general applicability.

### Questions
1. I am curious how robust your difflib-based action-similarity reward is to “near-miss” actions
2. I think the framework genuinely novel and potentially a new training paradigm for leveraging reward signals. I encourage you to evaluate SRL beyond structured domains in your paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Supervised Reinforcement Learning, a framework designed to enhance complex reasoning in LLMs by decomposing problems into step-wise actions. SRL combines dense, step-level rewards with flexibility in internal reasoning generation. The method is evaluated on mathematical reasoning and software engineering tasks, showing improvements over SFT and RL baselines. Key contributions include: (1) a step-wise reward signal to mitigate sparse rewards, (2) empirical gains on challenging benchmarks, and (3) a curriculum strategy (SRL→RLVR) for further refinement.

### Strengths
The problem formulation (e.g., action-based decomposition) is well-motivated and intuitive.
Results on math benchmarks show consistent improvements, with ablations justifying key components.
The curriculum strategy (SRL→RLVR) and flexibility in internal monologue generation are valuable insights.

### Weaknesses
The methodology bears significant resemblance to R³ (arXiv:2402.05808), which also uses a reverse curriculum with intermediate state sampling and outcome supervision for step-wise learning. Though SRL uses action similarity rewards instead of final-answer rewards, the high-level strategy of "starting from expert states and moving backward" is not sufficiently differentiated.
Comparisons focus on SFT and RLVR but omit advanced RL methods (e.g., DAPO, Dr. GRPO) and direct comparisons to R³.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Supervised Reinforcement Learning, a framework that bridges the gap between Reinforcement Learning with Verifiable Rewards, which suffers from sparse rewards on difficult problems, and Supervised Fine-Tuning, which often leads to overfitting through rigid token mimicry. SRL reformulates problem-solving as a sequence of logical actions, where the model generates an internal reasoning monologue followed by an action, receiving dense rewards based on action similarity to expert demonstrations. This allows flexibility in reasoning while providing granular feedback.

### Strengths
1. The paper presents a well-motivated and innovative approach to addressing limitations in LLM training for complex reasoning, with clear writing.

2. The introduction of SRL creatively combines elements of imitation learning and RL by decomposing expert trajectories into step-wise actions and rewarding only the actions, allowing the model to develop its own internal monologues. This novel formulation addresses the sparse reward issue in RLVR and the rigid mimicry in SFT, drawing from prior work like GRPO but extending it to dense, sequence-similarity-based rewards for hard problems.

3. By enabling effective learning on difficult datasets like s1K, SRL has broad implications for enhancing LLMs in domains requiring multi-step reasoning, such as math and software engineering. Its generalization to agentic tasks and potential as a cold-start for RLVR pipelines could influence future distillation and RL methods for smaller models.

### Weaknesses
1. The method lacks sufficient implementation details, raising concerns about reproducibility. For instance, the prompts or heuristics used to decompose expert reasoning traces into individual steps (e.g., parsing numbered sections from DeepSeek R1 outputs) are not provided, making it challenging to replicate the step-wise data construction process.

2. Although SRL rewards only the action sequence to avoid token-by-token memorization, this still imposes constraints on the model's output. For example, it may force the student to mimic potentially suboptimal or erroneous intermediate actions from the teacher (e.g., steps involving reflection to correct mistakes), even if the student could compute them correctly without such detours. This could limit flexibility in cases where the teacher's reasoning includes unnecessary self-corrections.

3. Experiments are limited to a single 7B model (Qwen2.5-7B-Instruct), without exploring scalability across different model sizes (e.g., 1B, 13B, or larger). This leaves uncertainty about whether SRL's benefits hold for smaller or larger models, potentially restricting insights into its robustness.

### Questions
1. For extremely difficult problems where the model fails to rollout any reasonable intermediate step actions (e.g., pass@k near zero even for steps), how does SRL enable effective learning? Could the dense rewards still provide guidance, or would additional techniques like bootstrapping be needed?

2. The method relies on a strong teacher model (e.g., DeepSeek R1) to generate reliable expert trajectories. If no such trustworthy teacher is available (e.g., for novel domains), how could SRL be adapted—perhaps through self-generated or synthetic data, or iterative refinement?

### Soundness
4

### Presentation
3

### Contribution
3

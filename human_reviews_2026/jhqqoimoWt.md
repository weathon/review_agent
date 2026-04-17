# FAPO: Flawed-Aware Policy Optimization for Efficient and Reliable Reasoning

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Reinforcement learning with verifiable rewards (RLVR) has emerged as a promising paradigm for enhancing the reasoning capabilities of 
large language models (LLMs).
In this context, models explore reasoning trajectories and exploit rollouts with correct answers as positive signals for policy optimization.
However, these rollouts might involve flawed patterns such as answer-guessing and jump-in-reasoning.
Such flawed-positive rollouts are rewarded identically to fully correct ones, causing policy models to internalize these unreliable reasoning patterns.
In this work, we first conduct a systematic study of flawed-positive rollouts in RL and find that they enable rapid capability gains during the early optimization stage, while constraining reasoning capability later by reinforcing unreliable patterns.
Building on these insights, we propose **F**lawed-**A**ware **P**olicy **O**ptimization (**FAPO**), which presents a parameter-free reward penalty for flawed-positive rollouts, enabling the policy to leverage them as useful shortcuts in the warm-up stage, securing stable early gains, while gradually shifting optimization toward reliable reasoning in the later refinement stage.
To accurately and comprehensively detect flawed-positive rollouts, we introduce a generative reward model (GenRM) with a process-level reward that precisely localizes reasoning errors.
Experiments show that FAPO is effective in broad domains, improving outcome correctness, process reliability, and training stability without increasing the token budget.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FAPO, a novel approach to improving RL for LLMs in reasoning tasks. It identifies that flawed-positive rollouts persist throughout RL training and hinder model performance, then proposes FAPO which combines a compact generative reward model (FAPO-GenRM-4B) to detect reasoning errors with process-level rewards and a parameter-free algorithm that applies adaptive penalties to flawed positives. FAPO demonstrates consistent improvements across 7B and 32B models on mathematical (AIME24: +4.7%, AIME25: +3.1%) and general reasoning tasks (GPQA-Diamond: +1.5%), while substantially reducing flawed-positive ratios.

### Strengths
1. The paper is well-written and easy to follow. In particular, figures in the paper effectively illustrate key concepts
2. The systematic study in Section 2.2 provides valuable insights into the prevalence and evolution of flawed-positive rollouts during RL training, supported by both automatic evaluation and human verification
3. The effectiveness of FAPO is demonstrated both theoretically and empricially

### Weaknesses
1. The hybrid process + outcome reward formulation is not novel. Prior works have extensively explored combining step-level and outcome-level rewards (e.g., arXiv:2312.08935, arXiv:2504.13958)
2. The paper only evaluates on mathematical reasoning tasks (AIME24, 25) and one general domain benchmark (GPQA-Diamond), which is insufficient to claim broad applicability. The paper would benefit significantly from including additional widely-used benchmarks such as AMC and MATH
3. Only two model families (Qwen2.5-Math-7B and Qwen2.5-32B) are tested, both starting from pre-trained models. It would be valuable to stress-test the proposed GenRM on models that already have some code-starts with long-CoT trajectories
4. While the improvements are consistent, the absolute performance gains come at considerable cost: (1) requires training an additional 4B reward model, (2) needs complex asynchronous infrastructure

### Questions
1. Why does the student model outperform the teacher model? Table 2 and Figure 3 show that FAPO-GenRM-4B outperforms Qwen3-32B (teacher) on FlawedPositiveBench
2. What is the overall computational cost overhead? The paper mentions training time increases by "less than 20%" but lacks comprehensive cost analysis (e.g., the cost of data synthesis, training GenRM, and detailed inference costs during RL). In particular, I am curious whether the inference cost would increase for GenRM if performing FAPO on models that have some code-starts with long-CoT

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper begins by identifying a common failure mode in outcome-based RLVR: flawed-positive rollouts, which are trajectories that produce correct final answers through flawed reasoning. It first quantifies their prevalence and twofold effects during training, then proposes FAPO (Flawed-Aware Policy Optimization) as an enhancement to DAPO/GRPO based on this insight. FAPO trains a generative reward model (GenRM) with both outcome- and process-level supervision to detect flawed positives, and integrates this signal into the policy optimization process via an additional penalization term to the outcome reward. Experimentally, the trained FAPO-GenRM achieves higher F1 scores than baselines on ProcessBench and the curated FlawedPositiveBench. When applied to train Qwen-7B and Qwen-32B reasoning models, FAPO delivers improved performance on AIME24/25 and GPQA-Diamond, reducing flawed-positive ratios compared to GRPO baselines without increasing the inference token budget. Additional ablations analyze GenRM effectiveness, examine potential reward-hacking risks, and show that asynchronous deployment keeps computational overhead modest.

### Strengths
- **Well-motivated.** The paper clearly identifies *flawed-positive rollouts* as a pervasive yet unresolved failure mode in RLVR, illustrating their dynamics through quantitative trends (Figure 2) and motivating the need for process-level awareness beyond outcome-based rewards. This diagnosis provides a solid conceptual foundation for introducing FAPO.
- **Sound theoretical formulation.** The theoretical sections (Sec. 3.2, Appendix A) formalize the reward-penalization mechanism using group-relative advantage estimation. The analysis further justifies adopting a fixed penalty coefficient (λ = 1) based on the majority-guided condition, making FAPO both principled and free of hyperparameters.
- **Comprehensive experimental validation.** The experiments cover multiple reward benchmarks (FlawedPositiveBench, ProcessBench) and reasoning benchmarks (AIME24/25, GPQA-Diamond). The study includes one 4B model for the reward model and two models (7B and 32B) for the reasoner model training, supported by detailed analysis on FAPO-GenRM's detection ability, training effectiveness, and the reward-hacking risks. Together, these components form a coherent and comprehensive empirical evaluation.

### Weaknesses
- **[Soundness]** While the observations in Section 3 are insightful, the claimed causal relationship between flawed positives and performance gains (lines 190–192) requires further evidence. The presented figure only demonstrates a correlation, not causation, and additional controlled ablations would be needed to validate this conclusion.
- **[Soundness]** The evaluation datasets are relatively small. AIME (30 samples) can be insufficient for robust reasoning assessment, and on the larger GPQA-Diamond (198 samples), the observed improvement is modest (Figure 4, 1.5 points). Moreover, in Figure 4 (bottom right), the flawed-positive ratio remains high even after FAPO training, suggesting potential limits in its corrective capability.
- **[Significance]** Although the theoretical formulation is novel and well-structured, the method essentially extends RLVR by integrating outcome and process rewards into a unified training signal. Similar ideas have been explored in prior works such as [1] and [2], which slightly reduces the originality of the contribution. Furthermore, the algorithm focuses solely on mitigating flawed positives, leaving other issues in RLVR, such as false negatives, unaddressed.

[1] Process Reinforcement Through Implicit Rewards, Arxiv Feb 2025

[2] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains, Arxiv July 2025

### Questions
1. How sensitive is FAPO to the choice of λ beyond the majority-guided setting (λ = 1)? Could a curriculum that gradually increases λ during training yield better performance than the fixed choice?
2. Do the authors have the performance results on other benchmarks (i.e., AIME25, GQPA) corresponding to the setup shown in Figure 5?
3. During RL training, the GenRM remains fixed while the policy model continuously updates, which may lead to reward-hacking behavior. In Figure 7, FAPO-GenRM appears more robust than PRM, but to what extent? Has its robustness been quantitatively evaluated, and could it eventually suffer from reward hacking as well?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses the issue of "flawed-positive" rollouts in reinforcement learning with verifiable rewards (RLVR) for LLMs, where models arrive at correct answers via unreliable reasoning (e.g., guessing). The authors first analyze this phenomenon, finding that while flawed positives provide rapid early gains, they ultimately constrain capability by reinforcing these unreliable patterns. To mitigate this, they propose Flawed-Aware Policy Optimization (FAPO), a method that applies a parameter-free reward penalty to flawed-positive rollouts. This approach aims to leverage flawed positives as shortcuts during the initial warm-up phase while gradually shifting the optimization objective toward reliable reasoning in the later refinement stage. To enable this, the work also introduces a generative reward model (GenRM) trained with a process-level reward to accurately detect and localize reasoning errors within rollouts. The authors claim FAPO improves correctness, reliability, and stability without increasing the token budget.

### Strengths
1. The paper clearly identifies and analyzes a critical problem in RLVR. The preliminary study (Section 2.2) effectively demonstrates the "twofold effect" of flawed positives—acting as "stepping stones" early in training but hindering optimization later —providing a solid empirical foundation for the proposed solution.

2. The FAPO algorithm's adaptive reward penalty is simple yet theoretically grounded. The analysis in Appendix A demonstrates how this mechanism creates an automatic, parameter-free "optimization shift".

3. The development of a compact (4B) generative reward model (GenRM) is a key strength. The step-wise, distance-sensitive reward formulation used to train it is well-designed to encourage precise error localization rather than simple binary guessing. This trained GenRM is shown to be highly effective, outperforming its 32B teacher model and a 72B SOTA discriminative model.

4. The experiments are thorough, testing on multiple models (7B, 32B) and benchmarks (AIME24, AIME25, GPQA). Crucially, the authors present full learning curves rather than just final checkpoints. This transparently supports claims of improved training stability , outcome correctness , and process reliability (a reduced flawed positive ratio).

### Weaknesses
1. The entire framework's effectiveness is contingent on the quality of the FAPO-Critic-85K dataset, which was labeled by a "teacher model" (Qwen3-32B). The GenRM can only learn to detect flaws that the teacher model can identify. This creates a fundamental performance ceiling; any subtle errors missed by the teacher will be propagated, and FAPO will fail to penalize them.

2. The GenRM is trained primarily on mathematical reasoning tasks. While it shows good performance on GPQA-Diamond, it is unclear how well this math-trained critic can generalize to detecting a wider array of non-mathematical flawed reasoning (e.g., logical fallacies, factual inconsistencies) in open-domain tasks. The paper acknowledges this as a limitation.

3. The proposed asynchronous architecture, while necessary for performance, adds significant infrastructure complexity compared to standard RLVR, which uses a simple, synchronous rule-based verifier. The authors note this adds a non-trivial training time increase of "less than 20%", which could be a barrier to adoption.

### Questions
none

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
3

### Summary
This paper addresses a critical issue in RLVR for LLMs: the reinforcement of flawed reasoning patterns. In RLVR, models are rewarded for generating trajectories that lead to a correct final answer. However, many such positive rollouts contain unreliable reasoning steps, which are rewarded identically to fully correct solutions. This flawed-positive problem can lead to models that are correct on a specific metric but ultimately unreliable.

The authors first conduct a systematic analysis, revealing that flawed positives act as valuable stepping stones for rapid early learning but later constrain performance by reinforcing bad habits. To address this, they propose FAPO, a two-stage solution: They train a efficient generative reward model using a novel step-wise RL objective that localizes the first error in a reasoning chain. This model achieves state-of-the-art performance on error detection benchmarks. FAPO applies a parameter-free reward penalty to rollouts flagged as flawed-positive by the GenRM. The penalty is designed to dynamically shift the learning focus, initially allowing flawed positives to aid learning before gradually steering the policy toward fully reliable reasoning as its capability improves.

### Strengths
The problem tackled is of great significance to the LLM reasoning community. As RLVR becomes a dominant paradigm for advancing LLM capabilities, ensuring that the learned reasoning is not just correct but also reliable and transparent is crucial for safety and trustworthiness. FAPO provides a practical, efficient, and theoretically grounded solution that improves both the efficiency and the final quality of RL training. The release of code and benchmarks further enhances its impact. Experiments on mathematical and GPQA reasoning tasks demonstrate that FAPO improves outcome correctness, reduces the rate of flawed positives, and enhances training stability without increasing response length.

### Weaknesses
While the results on mathematical reasoning and GPQA are strong, the paper's claims of broad domains would be more convincing with validation on a wider range of tasks. A key domain of interest is code generation, where verifiable rewards are common and flawed reasoning (e.g., code that passes specific tests but is buggy or inefficient) is a major concern. Demonstrating coding effectiveness would significantly broaden the method's impact and generalizability.

The 20% training-time overhead, while reasonable, could be a barrier to truly massive-scale training. A more detailed discussion of the bottlenecks and potential optimizations for a synchronous system (even as future work) would be valuable for practitioners looking to adopt it at a larger scale.

The paper focuses on tasks with easily verifiable final answers. A natural question is how FAPO would be adapted to more subjective or open-ended tasks where a "correct" final answer is not binarily verifiable. In such scenarios, would the process-level reward become the primary signal?

### Questions
Please refer to "Weaknesses".

### Soundness
3

### Presentation
2

### Contribution
3

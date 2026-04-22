# Wiki-R1: Incentivizing Multimodal Reasoning for Knowledge-based VQA via Data and Sampling Curriculum

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 4, 6

## Abstract
Knowledge-Based Visual Question Answering (KB-VQA) requires models to answer questions about an image by integrating external knowledge, posing significant challenges due to noisy retrieval and the structured, encyclopedic nature of the knowledge base. These characteristics create a distributional gap from pretrained multimodal large language models (MLLMs), making effective reasoning and domain adaptation difficult in the post-training stage. In this work, we propose \textit{Wiki-R1}, a data-generation-based curriculum reinforcement learning framework that systematically incentivizes reasoning in MLLMs for KB-VQA. Wiki-R1 constructs a sequence of training distributions aligned with the model’s evolving capability, bridging the gap from pretraining to the KB-VQA target distribution. We introduce \textit{controllable curriculum data generation}, which manipulates the retriever to produce samples at desired difficulty levels, and a \textit{curriculum sampling strategy} that selects informative samples likely to yield non-zero advantages during RL updates. Sample difficulty is estimated using observed rewards and propagated to unobserved samples to guide learning. Experiments on two KB-VQA benchmarks, Encyclopedic VQA and InfoSeek, demonstrate that Wiki-R1 achieves new state-of-the-art results, improving accuracy from 35.5\% to 37.1\% on Encyclopedic VQA and from 40.1\% to 44.1\% on InfoSeek. The project page is available at https://artanic30.github.io/project_pages/WikiR1/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes Wiki-R1, a curriculum reinforcement learning framework based on data generation, which specifically includes three methods: (1) controlling the difficulty of samples during training, and (2) a novel sampling strategy that selects samples likely to yield significant advantages in reinforcement learning updates based on their reward values.

### Strengths
1) The paper’s motivation is very clear.
2) The main figure in the paper is drawn clearly.

### Weaknesses
1) The paper provides limited explanation of the proposed method, making it somewhat unclear.
2) In Table 2, Qwen2.5 is used, while in Table 3, Qwen2.5-VL is employed; however, the main table does not explicitly specify which LLM is used, leading to some confusion regarding the model configurations.
3) In Table 3, the inclusion of the sampling curriculum and observation propagation strategies yields inferior performance compared to using data curriculum generation alone, which undermines the effectiveness of the sampling curriculum and observation propagation strategies.
4) Table 3 does not clarify whether the SFT (Supervised Fine-Tuning) data includes ground-truth (GT) annotations, raising the possibility of an unfair comparison.
5) The method needs to be evaluated on more datasets and base models to better validate its effectiveness.

### Questions
See Weakness

### Soundness
1

### Presentation
2

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
This paper proposes Wiki-R1, a data-generation-based curriculum reinforcement learning framework designed to enhance multimodal large language models (MLLMs) on Knowledge-Based Visual Question Answering (KB-VQA) tasks.
The key idea is to systematically bridge the distribution gap between pretraining data and the KB-VQA domain by Controllable Curriculum Data Generation and Curriculum Sampling with Observation Propagation.
Through these mechanisms, the model learns progressively harder reasoning examples, mitigating sparse rewards during RL fine-tuning.
Experiments on Encyclopedic-VQA and InfoSeek show new state-of-the-art performance with strong generalization to unseen questions.

### Strengths
1. The framework introduces an elegant combination of data-level and sampling-level curricula. The idea of controlling retrieval difficulty rather than merely selecting data is innovative and well-motivated by the sparse reward challenge in KB-VQA.
2. The approach yields notable accuracy gains with only ~40k training samples — far less than prior methods requiring millions — highlighting efficiency and scalability.

### Weaknesses
1. While the combination of controllable retrieval and curriculum sampling is well-engineered, the theoretical novelty may be seen as incremental over prior curriculum RL works. The core mechanism (progressively harder data and adaptive sampling) is conceptually similar.
2. The performance gains are mostly demonstrated on EVQA and InfoSeek. It’s unclear whether the framework generalizes to other KB-VQA settings (e.g., OK-VQA) or to different retrieval model architectures.
3. The literature review seems to focus on works on EVQA/InfoSeek. A large collection of works in other KB-VQA datasets (e.g. OK-VQA) are missed in the discussion.
4. The proposed model outperforms many existing works, however, it remains unclear to me whether the gain is from RL training since many existing works are training-free. The authors should make it clear in the table (whether the approach is training-free) and compare the approach with training methods.

### Questions
The paper defines an upgrade threshold τ = 0.55 for moving to the next difficulty level. How sensitive is performance to this threshold? Would too-rapid or too-slow progression harm training stability?

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
4

### Summary
This paper identifies the distribution gap with KB-VQA data which leads to sparse rewards in RL approaches and proposes Wiki-R1, a curriculum RL framework with two core components: Controllable Curriculum Data Generation and Curriculum Sampling with Observation Propagation. Experiments on Encyclopedic VQA (EVQA) and InfoSeek benchmarks show state-of-the-art (SOTA) results.

### Strengths
1. Targeted Problem: Addresses a critical pain point of distribution gap and sparse rewards in RL-based KB-VQA.
2. Generalization: Excels on unseen question splits e.g., InfoSeek Unseen-Q:47.8% vs. prior 40.4, indicating robust reasoning.
3. Component Validity: Ablations clearly show that data curriculum improves DAPO performance, and propagation is necessary for sampling curriculum to work).

### Weaknesses
1. Limited Retrieval Control: The data generation relies on adjusting retrieval noise (number of candidates, ground-truth inclusion) but does not fully control the type of noise (e.g., irrelevant vs. slightly relevant candidates).
2. Hyperparameter Transparency: No sensitivity analysis for key hyperparameters (e.g., curriculum gap threshold τ, observation propagation smoothing factor α).
3. RL Algorithm Scope: Only uses DAPO as the base RL algorithm—no comparison with other RL methods (e.g., PPO, GRPO) to validate generalizability.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the task of enhancing multimodal large language models (MLLMs) for Knowledge-Based Visual Question Answering (KB-VQA), where models need to integrate visual and textual information to answer questions. The authors propose Wiki-R1, a reinforcement learning framework with two main contributions. First, Curriculum Data Generation manipulates the retriever to create training samples at different difficulty levels, enabling the model to learn progressively from easy (accurate retrieval) to hard (noisy retrieval) examples. Second, Curriculum Sampling with Observation Propagation implements a sampling strategy that prioritizes the most informative samples (those with approximately 50% accuracy) and propagates difficulty estimates to unseen samples. These components work together with RL optimization to provide denser reward signals and improve reasoning capabilities in KB-VQA scenarios. The method is evaluated on two KB-VQA benchmarks.

### Strengths
- The paper clearly identifies why RL fails in KB-VQA (retrieval noise leads to sparse rewards) and proposes a reasonable data-centric solution to increase reward density and improve downstream reasoning performance.
- The core idea is elegant: generating progressive difficulty data through controllable parameters and selecting the most valuable samples using accuracy-based Gaussian sampling to balance between "already learned" and "not yet learned" examples. I particularly appreciate the Observation Propagation mechanism for finding similar samples to stabilize training.
- The method shows consistent performance gains and potential better data efficiency compared to baselines.

### Weaknesses
- Some key parameters are not specified, including the exact definition of the reward function in the RL objective, and the observation propagation parameters.
- The reliance on TF-IDF similarity may restrict the method to surface-level lexical matching, potentially missing semantically similar samples that use different vocabulary.
- The paper lacks details needed to assess the true benefit of RL. Specifically: What retrieval configuration was used for the SFT baseline? Does it use the same samples as the RL experiments? Was the SFT training data also curriculum-generated? Without a fair comparison where both SFT and RL use identical curriculum data, it's difficult to isolate whether the gains come from the curriculum design itself or from the RL optimization.
- Results are reported as single values without standard deviations or significance tests, making it difficult to assess the reliability of the comparisons.
- Table 5's comparison may be misleading because it only lists training sample counts without reporting actual training time, GPU costs, or convergence speed. While Wiki-R1 achieves better results with less data, it employs computationally intensive RL training (multiple rollouts per sample, dynamic sampling, observation propagation), whereas baselines use simpler SFT. The paper reports Wiki-R1's training time but provides no training time data for baselines, making it hard to determine which method has lower total computational cost.

### Questions
What exact data configuration was used for the SFT baseline in Table 3? Does it use the same entity-balanced samples as the RL experiments?

### Soundness
3

### Presentation
3

### Contribution
3

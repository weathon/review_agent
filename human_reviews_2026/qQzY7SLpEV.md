# EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning for LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
Reinforcement learning with verifiable reward (RLVR) has become a promising paradigm for post-training large language models (LLMs) to improve their reasoning capability. However, when the rollout accuracy is low on hard problems, the reward becomes sparse, limiting learning efficiency and causing exploration bottlenecks. Existing approaches either rely on teacher models for distillation or filter out difficult problems, which limits scalability or restricts reasoning improvement through exploration.

We propose EvoCoT, a self-evolving curriculum learning framework based on two-stage chain-of-thought (CoT) reasoning optimization. EvoCoT constrains the exploration space by self-generating and verifying CoT trajectories, then gradually shortens CoT steps to expand the space in a controlled way. The framework enables LLMs to stably learn from initially unsolved hard problems under sparse rewards. We apply EvoCoT to multiple LLM families, including Qwen, DeepSeek, and Llama. Experiments show that EvoCoT enables LLMs to solve previously unsolved problems, improves reasoning capability without external CoT supervision, and is compatible with various RL fine-tuning methods. We release the source code to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes the EvoCoT framework, which overcomes the exploration bottleneck in reinforcement learning with RLVR for LLMs through a two-stage self-evolving curriculum learning mechanism. The method improves reasoning ability effectively without external supervision and demonstrates its effectiveness across various models and benchmark tasks.

### Strengths
- EvoCoT enables stable learning under sparse reward conditions, effectively alleviating the reward sparsity issue commonly seen in RLVR tasks.
- By progressively truncating self-generated CoTs, EvoCoT automatically constructs training samples with gradually increasing difficulty, achieving a natural "easy-to-hard" curriculum. This process eliminates the need for manual difficulty annotation or ordering of training samples, significantly simplifying the training pipeline.

### Weaknesses
- Figure 2 in RQ1 may not provide convincing evidence, as EvoCoT is conditioned on partially correct CoTs during rollouts, which naturally leads to more correct outputs. Direct evaluation on a held-out validation set would more effectively demonstrate the method’s impact.
- The generalization experiments in RQ2 are still limited to the mathematical domain. Since reasoning patterns may be highly similar across different math datasets, this setup may not sufficiently demonstrate generalization. Testing on heterogeneous tasks—such as code generation or logical reasoning—would provide more convincing evidence of cross-domain transferability.

### Questions
- The two-stage design may significantly increase sampling overhead. In standard GRPO training, one step samples `batch_size × n_sample` trajectories. Does one EvoCoT step consist of the full Stage 1 and Stage 2? Or does it only sample `batch_size` examples from the filtered (Q, C) pairs in Stage 1, then apply `n_sample` rollouts? How does the training time of EvoCoT compare to GRPO when using the same number of training steps? Does EvoCoT still maintain an advantage under equal wall-clock time constraints?
- When the provided CoT is very long, the problem becomes too easy, and the model may achieve perfect accuracy; however, the reward remains sparse. Would this lead to inefficiencies in training?
- Was GRPO trained only on the subset of *hard* problems? If so, the initial gradients may be extremely weak, making the comparison potentially unfair.

### Soundness
2

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
This work propose a curriculum learning framework for RLVR reasoning process. In stage 1, LLM generates CoT trajectories conditioned on questions and final answers. The stage 2 progressively shortens the reasoning paths constructed in stage 1.

### Strengths
1. The sparse-reward problem in RLVR that this work focuses on is important and worthy of investigation.
2. The proposed curriculum learning method is simple and extensible to prior methods, without the need of human annotations or expert models.

### Weaknesses
1. The performance gains over method-level baselines appear modest and, in one case, are lower than PRIME with Qwen2.5-7B (53.5 vs. 55.3). 
2. The step-wise procedure in Step 2 seems potentially time-consuming. For a reasoning trajectory of n steps $(Q, c_1, c_2, \ldots, c_n)$, the LLM needs to produce n answers for curriculum learning. How is the value of $n$ determined across different trajectories?

### Questions
1. The setup for Figure 2 is a bit unclear. Could you clarify the conditions under which the GRPO curve remains at 0?
2. Could you please report the training time of EvoCoT compared with vanilla GRPO?
3. As illustrated in Figure 1 (Stage 2), shorter CoTs intuitively expand the exploration space. Could you provide an empirical analysis quantifying this exploration improvement? For example, metrics indicating exploration under different CoT lengths would be helpful.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
EvoCoT is a self-evolving framework which utilizes chain of thought trajectories and improve model performance on hard mathematical problems. The first stage extracts logically consistent CoT trajectories and the second stage involves training the model on these trajectories, gradually removing later CoT steps as training progresses. As partial CoT trajectories are harder to learn from than complete ones, this creates a curriculum for more guided learning. 

Stage 1: Answer-Guided Reasoning Path Self-Generation: In this stage, the model is shown a prompt and answer and asked to derive the answer. To verify that the generated reasoning chain is correct, the model is then given the prompt and reasoning chain and asked to compute the answer. If the 2 answers don't match, the example is discarded. 

Stage 2: Step-Wise Curriculum Learning: We use the chains extracted in the first step to improve the model's reasoning capabilites. To start, the model is shown the complete reasoning trajectory. With every step, one reasoning step is removed in reverse order. As training progresses, the the chains get shorter and the problem gets harder. 

The authors demonstrate the effectiveness of EvoCoT on a variety of model families, baselines and benchmarks.

### Strengths
- The paper is well-written and easy to understand. 
 - It showcases the effectiveness of EvoCoT across a number of model families, benchmarks and baselines.

### Weaknesses
EvoCoT has limited novelty. [[1]](https://arxiv.org/pdf/2203.14465) proposes the idea of generating and training on CoT paths with the main difference being that STaR doesn't provide the model with the answer during generation. Similarly, the idea of using truncated versions of the traces to enable harder problem solving has been explored in other works [[2]](https://arxiv.org/abs/2402.05808)[[3]](https://arxiv.org/abs/2506.18110).

The effectiveness of EvoCoT is also not clear. It seems to generalize better to certain tasks over others (AIME 24 and AMC 23 generally improve while others remain the same or regress). It's not obvious why that is the case. The paper would benefit from a deeper analysis into the conditions under which EvoCoT can enable the most gains. One part of this has been explored as we see that a more powerful model generates higher quality trajectories leading to stronger improvements. However, this hypothesis should be tested against larger models as well. 

The generation of CoT trajectories and stepwise learning introduces overhead during training. It would be good to include some numbers to quantify this. 

EvoCoT seems to suffer from label leakage related training accuracy inflation issues as seen by the strong improvements on the GSM8K and MATH training set in Table 2 but neutral results on the test sets in Table 3.

### Questions
Please see weaknesses

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a new two-stage post-training method to LLMs, where the reasoning chains in training are ordered in a way that is supposedly making the learning from easy to hard.

### Strengths
- Significance: curriculum learning is an important direction to study, and current practitioners not sufficiently understand how to design effective curriculum. This paper provides a fresh perspective from CoT lengths.

### Weaknesses
- Method: The training method looks to be problematic.
  - First, please provide a clear algorithm on your method illustrating how the LLM is being trained in detail. Example: [[Phuong and Hutter](https://arxiv.org/pdf/2207.09238)]. Eq. 4 and Eq.5 are too toy and hard to understand.
  - Second, the method appears to be **off-policy**? Please provide careful analysis on the potential off-policy effect for the practical algorithm used. 
  - Third, the term curriculum learning has traditionally been used to describe the easy-to-hard scheduling/sampling of the marginal distribution of envs/tasks/prompts, e.g., [[Bengio et al., 2009](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf)], [[Parker-Holder et al., 2022](https://arxiv.org/pdf/2203.01302)], [[Ye et al., 2024](https://arxiv.org/pdf/2411.00062)], instead of the conditional distribution as in this work; it would be helpful to add comparisons to help readers to gain a deeper understanding. Also, training loss or reward discrepancies are often used as the proxy for difficulty in prior works, while this work uses the length of reasoning trajectories -- the authors should provide more ablations on this proxy.
- Format: There are some issues with the format. For example, \citep should be used to add parentheses to the citations; the star notation (*) is usually used to denote optimality, and its usage in Eq.4 is confusing.
- Experiments: The experimental results as shown in Table 3 looks to be relatively neutral in general. It would be also helpful to add comparisons to other curriculum-based methods.

References:
[1] Bengio, Yoshua, Jérôme Louradour, Ronan Collobert, and Jason Weston. "Curriculum learning." In Proceedings of the 26th annual international conference on machine learning, pp. 41-48. 2009.
[2] Parker-Holder, Jack, Minqi Jiang, Michael Dennis, Mikayel Samvelyan, Jakob Foerster, Edward Grefenstette, and Tim Rocktäschel. "Evolving curricula with regret-based environment design." In International Conference on Machine Learning, pp. 17473-17498. PMLR, 2022.
[3] Ye, Ziyu, Rishabh Agarwal, Tianqi Liu, Rishabh Joshi, Sarmishta Velury, Quoc V. Le, Qijun Tan, and Yuan Liu. "Scalable Reinforcement Post-Training Beyond Static Human Prompts: Evolving Alignment via Asymmetric Self-Play." arXiv preprint arXiv:2411.00062 (2024).
[4] Pattnaik, Pulkit, Rishabh Maheshwary, Kelechi Ogueji, Vikas Yadav, and Sathwik Tejaswi Madhusudhan. "Curry-dpo: Enhancing alignment using curriculum learning & ranked preferences." arXiv preprint arXiv:2403.07230 (2024).

### Questions
Please see the weakness section. I am happy to raise my score if the authors could sufficiently address the concerns raised.

### Soundness
2

### Presentation
2

### Contribution
2

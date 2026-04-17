# APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation

- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Reinforcement learning (RL) has become a cornerstone in advancing large-scale pre-trained language models (LLMs). Successive generations—including GPT-o series, DeepSeek-R1, Kimi-K1.5, Grok 4, and GLM-4.5—have relied on large-scale RL training to enhance reasoning and coding capabilities. To meet the community’s growing RL needs, numerous RL frameworks have been proposed. Most of these frameworks primarily rely on inference engines for rollout generation and training engines for policy updates. However, RL training remains computationally expensive, with rollout generation accounting for more than 90\% of total runtime. Besides, its efficiency is often constrained by the long-tail distribution of rollout response lengths, where a few lengthy responses stall entire batches, leaving GPUs idle and underutilized. As model and rollout sizes continue to grow, this bottleneck increasingly limits scalability. To address this challenge, we propose Active Partial Rollouts in Reinforcement Learning (APRIL), which mitigates long-tail inefficiency. In the rollout phase, APRIL over-provisions rollout requests, terminates once the target number of responses is reached, and recycles incomplete responses for continuation in future steps. This strategy ensures that no rollouts are discarded while substantially reducing GPU idle time. Experiments show that APRIL improves rollout throughput by at most \textbf{49.5}\% across commonly used RL algorithms (GRPO, DAPO, GSPO), accelerates convergence, and achieves at most \textbf{12.8}\% higher final accuracy across tasks. Moreover, APRIL is both framework- and hardware-agnostic, already integrated into the slime RL framework, and deployable on NVIDIA and AMD GPUs. Taken together, this work unifies system-level and algorithmic considerations in proposing APRIL, with the aim of advancing RL training efficiency and inspiring further optimizations in RL systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a framework called Active Partial Rollouts in Reinforcement Learning (APRIL), designed to accelerate training for critic-free, policy-gradient RL algorithms. While the framework demonstrates notable speedups, the experimental evaluation is limited: the authors test only small models, a mathematics-focused dataset, and a setting that is effectively near on-policy.

### Strengths
1. The paper is well organized and easy to follow; the figures effectively aid comprehension.
2. The reported acceleration is strong and is demonstrated across different GPUs and algorithms.
3. The background section is thorough, providing a comprehensive review of relevant prior work.

### Weaknesses
1. The partial-rollout design appears problematic. As the authors acknowledge, off-policy samples can introduce instability. In my view, APRIL’s observed stability stems from updating sampled trajectories jointly within a single batch, effectively avoiding the need for importance sampling. If the authors adopted the DAPO setting—where each RL step uses a batch of 4,096 samples that are updated separately over 16 iterations—the policy would drift substantially after each update, and APRIL would likely become unstable. Nevertheless, such off-policy configurations are often necessary for larger models due to I/O constraints.
2. The evaluation is restricted to relatively small models (<8B) and math-related tasks; scalability and generality remain unverified.
3. Although minor, figure text should be at least as large as the main text. Moreover, screenshots from W&B or TensorBoard are not suitable as formal figures in an academic paper.
4. While I understand the motivation to highlight practical impact (e.g., merges into well-known GitHub repositories), this disclosure compromises the anonymity required by ICLR and should be removed.

### Questions
1. Can the authors report results under highly off-policy regimes, matching the DAPO configuration?
2. Can the authors evaluate the method on larger models and additional domains (e.g., coding and reasoning)?
3. Do the authors employ importance sampling to correct for off-policy data? The current setup appears mathematically inconsistent without it. If importance sampling is applied, how does it affect stability and performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes APRIL, a scheduling mechanism that aims to strike a balance between synchronous and asynchronous, on-policy and off-policy reinforcement-learning training for LLMs. It halts over-provisioned rollouts and buffers partial completions for continuation in later steps. The goal is to mitigate long-tail response-length stalls that underutilise GPUs. Reported results show a 22.5% gain in rollout throughput and even slightly better performance, despite some data becoming stale due to the scheduling.

### Strengths
The paper addresses an important bottleneck in post-training LLMs with RL. The authors aim to strike a balance between highly asynchronous scheduling—which increases data staleness—and fully synchronous RL approaches such as GRPO, which can be slow. The paper’s structure and exposition are clear, and the mechanism is well presented, though a few sections could be clarified further.

### Weaknesses
I have the following major concerns:
1. The “partial rollout” solution is not new. Both the Kimi K1.5 technical report and SortedRL have a similar architectural design: early termination of long-tail completions and buffering for continuation. Although the authors mention the two papers in the related work, it is descriptive but non-comparative.
2. Off-policy staleness is only empirically waved away. The throughput, compared to other async RL scheduling methods like AREAL and AsyncFlow, underperforms. So the soundness of this mechanism depends on its robustness to stale data with standard on-policy RL. For the empirical results, I can’t find explicit reports of multiple random seeds, CIs, or other statistical tests to verify empirical robustness. I think adding those experiments will strengthen the claim. But it still does not allay my concern regarding the unaddressed stale data, as the paper “No Representation, No Trust: Connecting Representation, Collapse, and Trust Issues in PPO” has stated that stale data can increase non-stationarity, ultimately leading to performance collapse. The phenomenon is also discussed in the recent paper “Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs?”. The actionable item is to add some ablation experiments to explain to what degree staleness starts to hurt, and whether the model will collapse if trained longer.

Minor places:
1. At line 161, the phrase “rollout rollouts …” is confusing.
2. How the reward is handled for the incomplete rollouts is only implicitly mentioned at lines 88–89. My understanding is that the reward is not computed until the rollout is finished. I think this is worth stating more clearly.
3. In Figures 4–9, the plot labels are not readable unless you zoom in very closely.
4. The quotation marks are wrong, e.g., line 61 and line 1177.

### Questions
1. For the cached incomplete rollouts, how are the KV cache managed with the inference engine.
2. How is your method different from Kimi K1.5 and SortedRL?
3. Have you tried to train APRIL for longer, any policy collapsing observed?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces APRIL, a novel off-policy method for RL training, designed to mitigate the inefficiencies caused by lengthy rollouts. APRIL employs over-provisions on rollout requests, initiates the training phase when a training batch is ready, and allows incompleted rollouts to resume in the later RL training steps. APRIL provides comprehensive evaluations to validate the effectiveness of its design.

### Strengths
1. The optimization techniques are well-reasoned, effectively addressing the important and timely problem of on-policy RL training, which may be inefficient when the data distribution is long-tailed.
2. The evaluation results of this paper are comprehensive, covering both the system side and algorithm side evaluations. The evaluation results demonstrate significant improvement in system throughput or algorithmic convergence over the chosen baseline.
3. The writing of this paper is easy to follow as well as rigorous.
4. The implementation of this paper is well-optimized; its open-sourced code will significantly contribute to the machine learning community.

### Weaknesses
1. The evaluation section only picks one baseline, which is strictly on-policy RL training. However, there are already some well-known frameworks (e.g., AReaL) that potentially employ a similar idea to this paper. The additional performance gains over these state-of-the-art frameworks are not explored.
2. Some of the experimental results might be contradictory to the previous approaches. For example, this paper claims that, with their off-policy method, the algorithm accuracy on the math reasoning task is superior to the strict on-policy method. While in the previous paper, e.g., AReaL, shows that without a specific algorithmic design (e.g., on the loss terms), the RL training performance can degrade significantly when incorporating stale data (i.e., rollouts that are generated with older versions of model parameters). 
3. The optimizations introduced in this paper rely on an assumption that the rollout generation phase dominates the RL training. However, with larger inference batch sizes and more advanced GPUs (i.e., GPUs with larger GPU memory and GPU HBM memory bandwidth), will the rollout phase still dominate the RL training? For example, for an 8B model, one NVIDIA H200 GPU can accommodate a large batch size; with the strong HBM memory bandwidth capacity (4.8 TB/s on H200 GPUs), H200 GPUs can process rollouts very rapidly.

### Questions
Q1: What is the system throughput, as well as the algorithm convergence comparison of APRIL and previous state-of-the-art off-policy frameworks? If available, could the authors provide at least one additional baseline on one model?

Q2: Why is the algorithm convergence of APRIL superior to the on-policy method?  Some of the previous work claims that only with a careful algorithmic design could the off-policy method achieve a similar algorithm convergence. For example, in the AReaL paper, we can see that with more stale data, the algorithm's convergence drops significantly. The stale data, which may behave like a regularizer as stated by the APRIL paper, seems purely a drag down on algorithm convergence in AReaL. Even with AReaL's careful design (Sec 5 of their paper), the algorithm's convergence still cannot surpass synchronous RL training. As admitted by the APRIL authors, on average, there are 40% of stale data during training, while it's confusing that without a careful algorithm design, the algorithmic convergence can improve. Could the authors provide more insights? 

Q3: What is the key difference between APRIL and AReaL? As stated in the APRIL paper, the maximum stale data used model parameters within 5 steps. What if AReaL adopts a small data staleness degree, e.g., 2 or 4 (Sec 5 of their paper). The two systems are operating similarly. Could the authors provide more analysis?

Q4: What is the percentage of time consumed on rollout generation when using more advanced hardware (and tuning to a new optimal inference batch size)? Does the APRIL's assumption still hold on NVIDIA H200 GPUs (if available, NVIDIA B200 GPUs or AMD MI350 GPUs)? This paper would benefit from more theoretical or empirical analysis of their assumption on more advanced GPUs.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work presents an asynchronous RL training framework for language models. To explain, in classical RL framework, we sample multiple responses for each prompt in each batch, and wait until the responses have been fully generated; in this work, the author proposes to over provision prompts for each batch, and stop rolling out once the number of rollouts reaches a threshold, and the incomplete rollouts are stored in the data buffer and are resumed in later steps. The authors run experiments with the proposed framework using qwen-3b/8b with mathematical datasets, and show improved efficiency and performance.

### Strengths
- **Significance**: Improving training efficiency and leveraging partial rollouts are important directions that many practitioners care about. Also, over-sampling prompts/states and using only a subset of them for gradient update in a batch has been used a lot in prior works, but mostly focusing on creating a learning curriculum as far as I know. This paper provides a different perspective by using the technique to accumulate more valid/complete rollouts within a fixed time window.

### Weaknesses
- **Quality**: Overall, this work feels like a technical report/blog, rather than a rigorous research paper with sufficient depth.
    - **Lack of experiments and formal analysis**: Off-policy correction is an important topic in RL, however there are no convincing analysis in the paper discussing the potential issue from the proposed asynchronous framework, and the current experiments provided does not provide sufficient evidence. Would the performance gain remains under different hyperparameters and benchmarks? Also, it may not be true that "slightly off-policy rollouts may act as a regularizer, preventing the policy from diverging into pathologically long generation modes"; i suggest the authors to consider quantifying the diversity/divergence and present more evidence to readers. It would be helpful to include a *technical* writeup (instead of qualitative description) on how prior works analyzing and addressing the off-policy issues, and discuss if this work could provide any formal guarantee on the convergence of the algorithm.
    - **Presentation**: The paper looks not to be ready for publishing. For example, Figure 4 and 5 looks to be directly coming from weight & bias platform. Please revise them for better readability.

- **Originality**: The proposed method is very much similar to an existing work [[Zhang et al., 2025](https://openreview.net/forum?id=YoV9lIZ827)]. Although the authors have discussed briefly in the related work, it is hard to see if there is any fundamental difference between the two.

### Questions
1. Algorithm: Would you please provide a formal algorithm block in the paper? This could enhance the readability of the work. Example: [[Phuong and Hutter, 2024](https://arxiv.org/pdf/2207.09238)].
2. Method: Would you please be specific on how the advantage term are estimated? I assume that it is done after the full completion, but it would be nice to formally write out the equations.
3. Figures: In addition to Figure 3, would it be possible to also plot rewards (or accuracy) v.s. length, and their changing trend as the training progresses. 
5. Experiments: It would be meaningful to evaluate the scalability of the proposed method. I doubt if the performance would still be on par if the authors train longer with many different hyperparam combination / evaluate on more challenging benchmarks.
6. Analysis: Could you please provide formal analysis about the off-policy effect on the convergence of the algorithm?
7. Could you discuss in detail that how your work differs from prior works like sortedRL? The claim that "however, to the best of our knowledge, the specific mechanism ... have not been detailed publicly. Our work implements a specific form of this concept ..." is not convincing and cannot justify if this work has any original methodological contribution.

### Soundness
2

### Presentation
2

### Contribution
2

# OPPO: Accelerating PPO-based RLHF via Pipeline Overlap

- Decision: Accept (Poster)
- Scores: 4, 6, 8, 8

## Abstract
Proximal Policy Optimization (PPO)-based reinforcement learning from human feedback (RLHF) is a widely adopted paradigm for aligning large language models (LLMs) with human preferences. However, its training pipeline suffers from substantial inefficiencies due to sequential multi-model dependencies (e.g., reward model depends on actor outputs) and long-tail response lengths, where a few long responses straggle the stage completion. We present OPPO, a novel, lightweight, and model-agnostic PPO-based RLHF framework that improves training efficiency by overlapping pipeline execution. OPPO introduces two novel techniques: (1) Intra-step overlap, which streams upstream model outputs (e.g., actor model) in right-sized chunks, enabling the downstream model (e.g., reward) to begin prefill while the upstream continues decoding; and (2) Inter-step overlap, which adaptively overcommits a few prompts and defers long generations to future steps, mitigating tail latency without discarding partial work. OPPO integrates easily with existing PPO implementations with a lightweight wrapper. Extensive evaluations show that OPPO accelerates PPO-based RLHF training by $1.8\times$--$2.8\times$ and improves GPU utilization by $1.4\times$--$2.1\times$ without compromising training convergence.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines training system efficiency in PPO-based RLHF algorithms, revealing significant GPU underutilization and latency bottlenecks in current synchronous approaches. To overcome these limitations, the authors introduce a dual-strategy solution: First, intra-step overlap enables pipelined execution by chunking upstream model outputs (e.g., actor model) and streaming them to downstream models (e.g., reward model), allowing prefill operations to commence before upstream decoding completes. Second, inter-step overlap employs adaptive prompt overcommitment, strategically postponing lengthy generations to subsequent steps to reduce tail latency while preserving computational progress. Experimental results on the TRL framework demonstrate 1.8×–2.8× efficiency gains over baseline implementations.

### Strengths
- Provides valuable empirical analysis of inefficiencies in practical PPO-based training systems.
- Introduces effective actor generation and reward scoring overlap strategies to improve pipeline efficiency.
- Demonstrates promising empirical results on small models, achieving reduced running time without performance degradation.

### Weaknesses
- The paper lacks systematic analysis of system efficiency. A more principled approach to analyzing the dimensions for reducing time complexity and examining related hyperparameters would strengthen the contribution.

- The improvements are demonstrated on small-scale models with limited training data. It remains unclear whether these advantages extend to large-scale models or architectures beyond those tested in the paper. A theoretical framework and analysis would make the claims more convincing.

- The paper lacks comparison with existing systems such as Verl, OpenRLHF, AReaL, and RoLL. These systems have explored numerous resource management and load-balancing strategies for addressing challenges like long-tail generation.

- The improvements are restricted to PPO algorithms in the RLHF setting, which limits the paper's impact. A significant portion of modern alignment training uses GRPO and RLVR approaches, where no reward model is involved, and the proposed overlap strategies may not be applicable.

### Questions
- Regarding the intra-step overlap strategy, could this be implemented entirely at the sequence level? Specifically, would it be feasible to use an asynchronous approach where the generation batch is split into mini-batches, with each completed sequence immediately forwarded to the reward model for scoring upon generation completion?
- Could the authors provide a detailed comparison with recent advances in Verl, AReaL, RoLL, and OpenRLHF? These represent more modern, industry-grade systems with their own optimization strategies. A comparison would help clarify the unique contributions of this work relative to the current state-of-the-art, as TRL may not reflect the most competitive baseline.

### Soundness
2

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
3

### Summary
The paper focuses on improving the training efficiency of PPO algorithms. The authors propose two overlapping techniques:
- Intra-step overlap: allows the downstream model (e.g., reward model) to begin prefill while the upstream model continues decoding.
- Inter-step overlap: adaptively overcommits a few prompts and defers longer generations to future steps.
Experiments demonstrate that the proposed methods significantly accelerate PPO-based RLHF training.

### Strengths
- The paper is well written and easy to follow.

- Training efficiency of PPO algorithms is an important problem. The proposed two overlapping techniques are intuitive and well motivated.

- Experiments across different tasks are conducted to demonstrate the acceleration achieved by the proposed algorithm.

### Weaknesses
- For intra-step overlap, in reasoning scenarios, difficult problems often require generating long responses. With this technique, is there a risk that updates for difficult problems will always be deferred, while the algorithm converges quickly on easier problems?

- Could intra-step overlap affect the distribution being learned, as discussed in the first question?

- There are other techniques, such as fine-grained parallelism adopted in VERL for accelerating PPO, but these are not included in the experiments. Could the authors elaborate on the reason for this omission?

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper identifies and addresses two major sources of inefficiency in standard PPO-based RLHF training pipelines: (1) sequential dependencies between the actor, critic, and reward models, where downstream models must wait for the upstream actor to fully complete generation , and (2) long-tail latency, where a few long "straggler" responses in a batch delay the completion of the entire step .

The authors propose OPPO, a lightweight systems optimization framework that introduces two novel overlap techniques:

Intra-step Overlap: The actor model streams its generated tokens in adaptive chunks to the downstream critic and reward models. This allows the downstream models to begin their computationally expensive "prefill" stage concurrently while the actor is still decoding, effectively hiding the scoring latency .

Inter-step Overlap: The pipeline "overcommits" by starting with a batch of B + Δ prompts. It sends the first B responses that complete to the training stage, while the Δ unfinished straggler responses are deferred and resumed in the next iteration. This mitigates tail latency without discarding partial work .

Empirically, OPPO is shown to accelerate end-to-end PPO training by 1.8×–2.8× and improve GPU utilization by 1.4×–2.1×, all while achieving identical step-to-reward convergence and final model quality

### Strengths
Massive Efficiency Gains: The headline result of a 1.8×–2.8× speedup on a notoriously slow and expensive pipeline is a major strength. This is a huge practical win.

Preservation of Convergence: The paper's strongest evidence is that this speedup comes "for free." Figure 4 and Table 1 show that the step-to-reward convergence and final model quality are nearly identical to the baseline. This proves it's a true systems optimization, not an algorithmic trade-off.

Orthogonal Solution: The method attacks two different, orthogonal bottlenecks: pipeline bubbles (via intra-step overlap) and straggler latency (via inter-step overlap). The ablation in Figure 6 clearly shows that both components are necessary for the full speedup.

Smart Adaptive Control: The heuristic for dynamically controlling the overcommitment level Δ is very clever. It adaptively becomes more aggressive when training is improving and more conservative (decaying Δ) as training converges, balancing speed with stability.

### Weaknesses
"Lightweight" Claim: The paper claims the method is "lightweight" and requires "a few lines of code change". This seems like a significant overstatement. Implementing a streaming data buffer, inter-model communication for chunks, and the dynamic Δ scheduler (Algorithm 1)  is a non-trivial systems engineering task that deeply modifies the training loop's data flow.

Heuristic-Driven Controls: The method's dynamic controls are entirely heuristic. The chunk size is tuned by "periodically... appl[ying] a few candidate chunk sizes" , and the Δ level is based on the reward slope. While these heuristics are smart and work well, they are not theoretically derived and may require careful tuning (e.g., the window size W, bounds Δmin/Δmax).

Unanalyzed Staleness: The inter-step overlap introduces a form of staleness. While the policy used for generation is consistent (the work is just deferred), the prompts that are deferred are from an older data distribution. The paper shows that high staleness is bad (Fig 2c)  but doesn't analyze if this prompt-distribution staleness has any subtle, negative effects on the converged model, especially if Δ is large.

### Questions
Could you elaborate on the "few lines of code change" claim? This appears to require a complex streaming and scheduling data structure (Algorithm 1) . What is the practical engineering lift to integrate OPPO into an existing TRL-based pipeline?

The dynamic control for Δ is based on the reward slope st. How sensitive is this heuristic to a noisy reward signal? If the reward oscillates (common in RL), would this cause Δ to become unstable and harm performance?

The paper mentions OPPO "generalizes to other paradigms such as DPO". How would this work? The DPO pipeline does not have the same multi-stage, multi-model dependency as PPO. What exactly would be overlapped?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces a new pipeline methodology for PPO. The methodology is illustrated for single-node training and results in significantly faster training times.

### Strengths
The impacts on the speed of post training are quite significant, and the implementation appears to be easy to add to existing code bases. The paper is well-written with detailed experiments and clear plots. I also really appreciate the annotated algorithm.

### Weaknesses
> Among RLHF methods, Proximal Policy Optimization (PPO) (Schulman et al., 2017) has been the de facto standard due to its training stability and flexibility across diverse reward models and objectives. PPO underpins the training of state-of-the art LLMs such as GPT-4 (OpenAI et al., 2024), Gemma (Team et al., 2024), and LLaMA (Touvron et al., 2023; Grattafiori et al., 2024). 

The quoted passage is wrong. As far as I am aware there is no public information on the RLHF strategies used in GPT-4 or Gemma 3, and while LLaMA 2 (Touvron et al., 2023) used PPO, LLaMA 3 and 4  (Grattafiori et al., 2024; Meta, 2025) use DPO. I do not believe that PPO is the current standard anymore, with both DPO and GRPO being significantly more popular. Qwen 2 used DPO, Qwen 2.5 used both DPO and GRPO, and Qwen 3 used just GRPO. GRPO is used by DeepSeek in all their models. DPO is used by Tulu, Hermes3, and several other models as well. I am unaware of any major model trained in 2024 or 2025 that discloses that it uses PPO. None of this means that improving PPO's efficiency isn't useful, but this paper heavily oversells how widely used PPO is and uses that as a major motivator.

More importantly, this paper appears to exclusively examine single-node training. While that's not inherently problematic it is a strong limitation to the impact of the methodology since it would be inapplicable for large models. In the absense of any discussion or results of multinode training, I'm going to assume that the method doesn't work in a multinode settting. I strongly recommend including this as a limitation in the discussion of the paper.

### Questions
Do you have any results in a multinode setting?

### Soundness
3

### Presentation
3

### Contribution
3

# Scaling Agent Learning via Experience Synthesis

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
While reinforcement learning (RL) can empower autonomous agents by enabling self-improvement through interaction, its practical adoption remains challenging due to costly rollouts, limited task diversity, unreliable reward signals, and infrastructure complexity, all of which obstruct the collection of scalable experience data. To address these challenges, we introduce DreamGym, the first unified framework designed to synthesize diverse experiences with scalability in mind to enable effective online RL training for autonomous agents. Rather than relying on expensive real-environment rollouts, DreamGym distills environment dynamics into a reasoning-based experience model that derives consistent state transitions and feedback signals through step-by-step reasoning, enabling scalable agent rollout collection for RL. To improve the stability and quality of transitions, DreamGym leverages an experience replay buffer initialized with offline real-world data and continuously enriched with fresh interactions to actively support agent training. To improve knowledge acquisition, DreamGym adaptively generates new tasks that challenge the current agent policy, enabling more effective online curriculum learning. Experiments across diverse environments and agent backbones demonstrate that DreamGym substantially improves RL training, both in fully synthetic settings and in sim-to-real transfer scenarios. On non-RL-ready tasks like WebArena, DreamGym outperforms all baselines by over 30%. And in RL-ready but costly settings, it matches GRPO and PPO performance using only synthetic interactions. When transferring a policy trained purely on synthetic experiences to real-environment RL, DreamGym yields significant additional performance gains while requiring far fewer real-world interactions, providing a scalable warm-start strategy for general-purpose RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces DreamGym, a framework designed to synthesize diverse experiences with scalability in mind to enable effective online RL training for autonomous agents. This method embeds transition knowledge into the a experience model and leverage it to generate new challenging task. To verify their method, the author conduct some experiments on several agentic tasks like Webshop.

### Strengths
1. The idea of experience model is novel. 
2. The paper is well written.
3. The idea of embedding transition dynamic is good.

### Weaknesses
1. The baselines are too simple. There are many intrinsic rewards designed in reinforcement learning to encourage exploration, we recommend you include at least one.
2. Moreover, depending success label to select task group is not feasible for most situations. There can be a lot of circumstances in RL, The agent performs correctly in the first half of the task, but it is still making mistakes in the second half, in which case the task will not selected because the variation is zero. 
3. No task generation example is provided. I expect to see some instances of the selected task groups and the generated new task from task model.

### Questions
See weaknesses. If none of the concerns is addressed，I will consider to downgrade the score.

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
3

### Summary
This paper introduces DREAMGYM, a scalable framework for training LLM-based agents without heavy environment interaction. Instead of collecting real rollouts, the framework employs a reasoning-based experience model that generates synthetic trajectories through reasoning, simulating next states and rewards. It further incorporates curriculum-based task generation, selecting tasks with high reward entropy to maximize information gain and progressively increase difficulty. Experiments on WebShop, ALFWorld, and WebArena show that DREAMGYM achieves comparable or superior performance to PPO and GRPO while using less than 10% of real environment interactions.

### Strengths
- Targets a practical yet challenging problem in LLM agent RL — reducing training cost, wall-clock time, and instability arising from expensive environment interactions.
- The paper is well-structured and clearly written: the problem is precisely defined, and each subproblem is addressed through three clearly delineated components — experience model inference, experience model training, and curriculum-based task generation. It makes the overall paper easy to follow, even for non-expert readers.
- The experimental design is extensive and well-organized, effectively addressing questions that naturally arise during reading.

### Weaknesses
- Potential error accumulation in self-training loop: Since the experience model continuously generates synthetic rollouts and refines itself using those same transitions, any early bias or inaccuracy in its learned dynamics may compound over iterations. While this may not be severe in relatively simple domains (as Figure 4 suggests small gaps between real and synthetic transitions), it could become problematic in more complex or long-horizon environments where early modeling errors propagate and amplify.

- Unclear buffer update strategy: Are all synthetic transitions stored, or only feasible, high-quality ones are retained? Without a clear filtering or prioritization mechanism, low-quality or hallucinated transitions could accumulate in the buffer, potentially degrading both policy learning and experience model stability over time.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes DREAMGYM, a unified and scalable RL training framework for LLM agents. The core idea is to compress environment dynamics into a reasoning-based experience model that operates in an abstract textual state space, producing consistent state transitions and feedback. Coupled with an experience replay buffer (seeded with offline data and continually enriched with online trajectories) and reward-entropy–based curriculum task generation, DREAMGYM enables low-cost, efficient RL training entirely in a “synthetic environment,” while supporting sim-to-real transfer. Experiments show that on the non-RL-ready WebArena, DREAMGYM surpasses baselines by >150%; in RL-ready settings (WebShop/ALFWorld), purely synthetic interactions can match GRPO/PPO; and with sim-to-real, using ≤10% real interactions yields a further 64.5% gain. The paper attributes practical difficulties to costly real-world interactions, insufficient task diversity, unstable reward signals, and heavy infrastructure burdens; DREAMGYM addresses these via a unified abstract state space, vectorized/unified rewards, and an expandable task set.

### Strengths
1. By training in synthetic environments and combining a model capable of distilling experience with experience replay and an automatic problem-generation (curriculum) mechanism, the authors reduce dependence on the real system and improve training stability and efficiency. On benchmarks such as WebArena, WebShop, and ALFWorld, the method achieves strong results, which further improve after incorporating a small amount of real data.

2. The exposition is clear: the paper’s narrative is well structured and, overall, easy to read.

### Weaknesses
1. The work lacks experimental comparisons with representative methods in the “LLM Agents Reinforcement Learning” line of research; existing comparisons focus mainly on SFT/DPO and PPO/GRPO, making it difficult to accurately define this method’s relative advantages and applicability boundaries among peer approaches. Meanwhile, the related-work survey is neither systematic nor sufficiently comprehensive, with inadequate coverage of recent developments.

2. Although the paper claims reduced real-world interaction and engineering cost, it does not report verifiable metrics (e.g., wall-clock time, GPU hours, throughput/token cost). This prevents a reliable assessment of the return on investment across different compute budgets and scales.

3. Missing pseudocode; key details—such as data filtering/cleaning procedures, prompt templates, random seeds, critical hyperparameter tables and stopping criteria, and the order/frequency of alternating updates between the experience model and the policy—are insufficiently specified, and no accompanying scripts are provided, resulting in a high overall barrier to reproducibility.

### Questions
Please refer to the “Weakness” section for related questions.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper tackles the significant challenges of applying reinforcement learning RL to LLM agents, namely the high cost of environment interaction, unreliable reward signals, and the difficulty of scaling task diversity . The authors propose DreamGYM, a framework that synthesizes agent experiences to enable effective and scalable online RL. Instead of interacting with a costly real environment, the agent interacts with a "reasoning-based experience model". This model, an LLM trained on offline trajectories annotated with CoT reasoning , learns to predict state transitions and feedback signals in an abstract textual space. The framework has two other key components: an experience replay buffer (seeded with offline data and enriched with new interactions) to stabilize training, and an adaptive curriculum task generator that uses a "reward entropy" heuristic to create progressively more challenging tasks for the agent. The experiment show that the purely synthetic approach is competitive with baselines trained on 80K real interactions. On the "non-RL-ready" WebArena benchmark, DreamGYM enables RL training and achieves a relative improvement of over 150% compared to baselines. The sim-to-real approach demonstrates significant performance gains and sample efficiency.

### Strengths
The paper targets an important bottleneck in the field of autonomous agents. The prohibitive cost and low sample efficiency of online RL in complex, real-world environments (like web browsers) is a major barrier to progress. The paper's goal of creating a scalable, synthetic training environment is highly relevant and impactful. 

The framework's ability to enable effective RL on WebArena, an environment considered "not RL-ready", is the most compelling result. Achieving success rates over 30-45% where previous methods (including traditional RL) struggle to get past 10-19%  is a significant practical achievement.

### Weaknesses
The central idea is to use an LLM as a learned world model for model-based RL. The paper cites related work like Dreamer and other LLM-based environment models but claims to be different by focusing on "policy improvement" rather than "fidelity-first". This is a weak distinction, as policy improvement is the goal of all MBRL systems, including Dreamer. The main novelties are the (effective) use of CoT reasoning in the model's SFT training and the entropy-based curriculum generator. These are good contributions, but the overall framework is a (well-executed) application of established MBRL concepts to the LLM agent domain, not a fundamental new paradigm.

Besides, the paper emphasizes that it avoids costly real-world rollouts. However, it includes two important, undiscussed costs:
- Annotation: The experience model is trained on thousands of trajectories (e.g., 3.6K for WebShop, 5.2K for ALFWorld ), each annotated with reasoning traces by a "powerful LLM". This is a substantial one-time computational cost, relying on a (presumably) proprietary, high-capacity model (like GPT-4) that is not part of the open-source setup.
- Inference: The "fast" synthetic environment is itself a Llama-3.1-8B model. Generating each step of a trajectory requires a full inference pass from this 8B model . While this avoids infrastructure issues like Docker, it is still computationally intensive. Figure 3 (Left)  shows a reduction in total training time, but a clearer analysis of the wall-clock time per step (Real Env vs. DreamGYM) is needed to fully assess the *efficiency* claim.

### Questions
Please refer to the weakness part above.

### Soundness
4

### Presentation
3

### Contribution
3

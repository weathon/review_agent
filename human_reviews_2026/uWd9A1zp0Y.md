# Learning to Reason as Action Abstractions with Scalable Mid-Training RL

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Large language models excel with reinforcement learning (RL), but fully unlocking this potential requires a mid-training stage. Intuitively, an effective mid-training stage should both learn a strong policy prior and enable fast learning through online interactions. We formalize this intuition by presenting the first theoretical result on how mid-training shapes post-training: it acquires strong policy priors by efficiently pruning the action space and accelerates RL convergence by shortening the effective planning horizon. Moreover, we prove that temporal abstractions simultaneously compress the size of the action set and reduce the decision horizon, thereby improving regret minimization after training. Building on these insights, we introduce Reasoning as Action Abstractions (RA3), a scalable mid-training algorithm. Specifically, we derive a temporal variational bound and optimize it by iteratively discovering temporally-consistent latent structures via RL, then fine-tuning on the bootstrapped data. Experiments on code generation tasks demonstrate the effectiveness of our approach. Across multiple base models, RA3 improves the average performance on HumanEval and MBPP by 8 and 4 points over the base model and the next-token prediction baseline. Furthermore, RA3 achieves faster convergence and higher asymptotic performance in RLVR on HumanEval+, MBPP+, LiveCodeBench, and Codeforces.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents RA3 (Reasoning as Action Abstractions), a mid-training algorithm for large language models that learns temporally consistent latent actions. The authors provide the theoretical analysis explaining how mid-training shapes post-training reinforcement learning (RL), identifying two key mechanisms: (1) efficient pruning of the decision space and (2) shortened effective planning horizons. Empirically, RA3 improves code generation accuracy across multiple models and datasets.

### Strengths
1. This work provides the formal framework linking mid-training design to post-training RL performance, supported by clear theorems and lemmas.
2. The proposed method achieves consistent improvements on HumanEval, MBPP, LiveCodeBench, and Codeforces benchmarks, using both Qwen and Llama models.
3. The whole paper is clear with good experimental presentation.

### Weaknesses
1. The theoretical assumptions (finite action subsets, independent expert demonstrations) could be idealized and may not hold in practical large-scale LLM training.
2. Experimental comparisons are limited mainly to NTP, more baselines like BRiTE are recommended to be included.
3. The authors are recommended to include code or dataset for reproduction.

### Questions
My main concern is the theoretical assumption and experimental comparison. Please refer to the Weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies mid-training for LLMs and argues that effective mid-training should (i) prune the decision space to learn a strong policy prior and (ii) shorten the effective planning horizon so post-training RL converges faster. The authors formalize both effects: they show sample-complexity bounds for pruning “bad” action subsets and prove that temporally-extended actions reduce the RL horizon, improving convergence rates. Building on this, they propose RA3 (Reasoning as Action Abstractions)—a scalable mid-training algorithm that introduces a temporal ELBO for next-token prediction, optimized via EM: an E-step uses self-supervised RL to discover a sequence of temporally consistent latent actions (the “reasoning”/intent), and an M-step fine-tunes on the bootstrapped data.

### Strengths
1. The paper gives a regret decomposition and connects mid-training to post-training RL via pruning error and RL error, with a sample-complexity result (bad-subset pruning) and a convergence-rate bound favoring temporally extended actions—a useful, general framing beyond the specific algorithm.

2. The latent persistence reduces rollout frequency; the penalty turns RA3 into NTP in the limit, giving a knob for cost control. The paper also provides a concrete training recipe (group baselines, asynchronous rollouts).

3. With GRPO RLVR, RA3 starts from a better prior and learns faster, achieving higher asymptotic accuracy on HumanEval+, MBPP+, LiveCodeBench, and Codeforces; ablations on the penalty c illustrate the compute/quality trade-off.

### Weaknesses
1. All experiments are in Python code generation; claims about “reasoning” and “agents” are theoretically motivated but empirically untested outside code. Extension to math, tool-use, or GUI agents would strengthen generality.

2. RA3 introduces several knobs (penalty c, latent prior α/format reward, rollout group size, max latent length). The ablation on c is helpful, but broader robustness/stability sweeps (and wall-clock cost vs. NTP/RLVR) are limited.

3. Baselines emphasize NTP and post-training RLVR; fewer head-to-head comparisons against other mid-training methods that distill reasoning (e.g., synthetic CoT mid-training) leave some ambiguity about when RA3’s RL-discovered latents outperform distilled traces at similar cost.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explores the concept of mid-training for large language models (LLMs) in reinforcement learning (RL), proposing that an effective mid-training stage can both learn a strong policy prior and enable faster adaptation through online interactions. The authors provide theoretical insights into how mid-training shapes post-training by pruning the action space and shortening the planning horizon, which accelerates RL convergence. They also investigate the role of temporal abstractions in compressing the action set and reducing decision horizons, improving regret minimization. Based on these insights, the paper introduces Reasoning as Action Abstractions (RA3), a scalable mid-training algorithm that iteratively discovers temporally-consistent latent structures via RL and fine-tunes on the bootstrapped data. Experiments on code generation tasks demonstrate that RA3 improves performance across multiple benchmarks and base models, showing faster convergence and higher asymptotic performance.

### Strengths
1. The idea of mid-training is novel, and the theoretical insights about pruning the action space and accelerating RL convergence are convincing.
2. Introducing temporal abstractions into RL training for LLMs is a creative and well-motivated approach.
3. The RA3 algorithm is well-designed, conceptually simple, and appears easy to reproduce, making it practically useful for further research or application.
4. The paper presents extensive experiments and ablation studies, which provide strong empirical evidence supporting the effectiveness of RA3.

### Weaknesses
1. The paper lacks experiments on larger-scale models, which could provide more insights into the scalability of the proposed approach.
2. There is insufficient discussion on the training efficiency of RA3, particularly in terms of computational cost or resource requirements compared to other methods.
3. The presentation of the paper could be improved. Some sections, especially the theoretical parts, are difficult to follow due to unclear explanations or insufficient detail.

### Questions
1. Could you elaborate on the potential challenges or limitations of applying RA3 to larger-scale models? For example, does the method scale efficiently with model size, or are there bottlenecks to address?
2. Some of the theoretical results (e.g., regarding temporal abstractions and regret minimization) are compelling but not fully intuitive. Could you provide additional clarification or examples to make these results more accessible?

### Soundness
2

### Presentation
2

### Contribution
3

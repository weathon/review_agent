# How reinforcement learning after next-token prediction facilitates learning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Recent advances in reasoning domains with neural networks have primarily been enabled by a training recipe that optimizes Large Language Models, previously trained to predict the next-token in a sequence, with reinforcement learning algorithms. We introduce a framework to study the success of this paradigm, and we theoretically expose the optimization mechanisms by which reinforcement learning improves over next-token prediction in this setting. We study learning from mixture distributions of short and long “chain-of-thought” sequences encoding a single task. In particular, when the task consists of predicting the parity of $d$ bits and long sequences are rare, we show how reinforcement learning after next-token prediction enables autoregressive transformers to generalize, whereas mere next-token prediction requires extreme statistical or computational resources to do so. We further explain how reinforcement learning leverages increased test-time computation, manifested in longer responses, to facilitate this learning process. In a simplified setting, we theoretically prove that autoregressive linear models following this training recipe can efficiently learn to predict the parity of $d$ bits as long as the proportion of long demonstrations in the data mix is not exponentially small in the input dimension $d$.
Finally, we demonstrate these same phenomena in other settings, including the post-training of Llama-series models on mixture variations of common mathematical reasoning benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explains why large language models improve dramatically when reinforcement learning (RL) follows next-token prediction training. The authors prove that next-token prediction alone cannot generalize on hard tasks if long reasoning chains are rare, whereas RL amplifies these rare sequences and enables efficient learning. Experiments on both synthetic and real-world reasoning tasks confirm that RL boosts accuracy and increases response length. The results highlight RL’s role in unlocking generalization from limited but valuable reasoning data.

### Strengths
* The paper is the first to provide a formal separation between next-token prediction and next-token prediction plus RL in autoregressive models, giving a novel theoretical account of why the widely-used “pre-train then RL” recipe succeeds.  
* By showing that RL can turn a sample-hard, exponentially data-hungry problem into a polynomial-time solvable one whenever long demonstrations are merely polynomially rare, the work directly informs how scarce reasoning data should be leveraged in large-scale model development.

### Weaknesses
* All conclusions of the paper are drawn around the task of predicting the parity of d bits given access to a source of sequences, and its generalization to more common reasoning tasks (science or open-ended) is questionable.
* There is a lack of experiments on a broader range of and more recent LLMs, e.g., qwen3.

### Questions
* Theorem 1 gives $p_{cot} < 1/3$ as the exact point where greedy decoding stays short. Is this an artifact of the two-step linear decision model, or does it survive richer embeddings (e.g., transformers with non-linear MLPs)? An ablation that keeps the data distribution but increases model expressivity would clarify whether the threshold is distribution- or architecture-specific.

* The post-training bound hides the per-round sample size inside Õ(·) and assumes fresh data every round. How many total unique prompts does the algorithm really need? 

* Parity has a single deterministic “long path”. Real reasoning data often contain many valid chains of varying length and quality. Does the two-component mixture still capture the dynamics, or does the presence of noisy/partial chains shift the critical $p_{cot}$ or require a different RL objective?

* The theory 2 requires $p_{cot} \in \Omega (d^{-\kappa})$. For $d\approx 1,000$ (typical for LLM prompts) this seems prohibitive. Is the polynomial dependence tight, or do transformers empirically succeed with much smaller constants?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a framework to theoretically understand the success the popular LLM training paradigm that RL post-training after SFT. This paper also uses the parity check experiments to show that RL enables the model to generalize to difficult tasks while SFT cannot. The paper also demonstrated this phenomenon on math problems.

### Strengths
- the paper is well-motivated - trying to understand the reason behind the current successful LLM training paradigm.
- the paper proposed a relatively comprehensive theoretical analysis framework.
- the paper empirically validated its perspective through a cleverly designed and controlled parity check experiment.

### Weaknesses
1. The theoretical proof relies on a simplified linear autoregressive model.
2. Some of the ideas of this paper are already pointed out in papers like DeepSeek-R1.
2. This paer lacks of practical guidance for future LLM training and new algorithm.

### Questions
- Could similar phenomena be observed when in a worded version of a parity task? Such as giving the task description as language input of a LLM.
- The theoretical analysis relies relies on a simple linear framework, how is it extended to non-linear structures like the Transformers?

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
This paper studies why RL after next-token prediction helps large language models learn better reasoning. The authors build a simple setting where training data mixes short and long “chain-of-thought” examples. They theoretically and experimentally prove that next-token prediction alone often fails when long samples are less than 1/3 in pretraining dataset. And RL can quickly improve learning by focusing on longer samples, which leads to longer and more correct responses.

### Strengths
1. Strong theoretical proof combined with solid experiments, showing when and why RL succeeds.
2. The paper gives a simple and convincing explanation of why RL helps reasoning. Providing clear insights.
3. The answer of  two core questions—"why RL works" and "why length increases" could offer a new design direction for LLM reasoning optimization.

### Weaknesses
1. The theory assumes fully correct CoT examples. However, the real pretrain data usually contains noisy or wrong data. And even positive RL trajectories could contain noise or false-positive ones. The paper lacks discussion of these factors.
2. The theoretical proof could be written in a more organized manner, including formulas. This would make that part easier to understand.

### Questions
1. How robust are the theoretical conclusions if long CoT samples contain noise?
2. Can you still observe “length-driven generalization” when RL rewards contain a certain proportion of length penalty?

### Soundness
4

### Presentation
3

### Contribution
4

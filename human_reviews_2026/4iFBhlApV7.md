# Beyond Memorization: Extending Reasoning Depth with Recurrence, Memory, and Test Time Compute Scaling

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 0, 6, 4, 4

## Abstract
Reasoning is a core capability of large language models, yet how multi-step reasoning is learned and executed remains unclear. We study this in a controlled cellular-automata (1dCA) framework that excludes memorisation by using disjoint train/test rules. Models are trained on short state sequences, required to _infer_ the hidden local rule, and then _chain_ it for multiple future steps. We find that most neural architectures learn the rule and achieve high next-step accuracy, but performance drops sharply as the required number of steps increases. Increasing model depth is crucial, and extending _effective_ depth via recurrence, memory, or test-time compute improves results but remains bounded. Complementing these controlled experiments, a natural-language proxy game shows that contemporary LLMs largely fail on the complex setting. Together, these results separate genuine rule induction from memorisation, quantify how difficulty scales with reasoning depth, and highlight the joint roles of architecture and training/inference procedures.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper studies multi-step reasoning in neural networks using a controlled one-dimensional cellular automata ($1$dCA) benchmark that prevents memorization by using **disjoint train/test rule sets**. Models must (i) infer the hidden local rule and (ii) **chain** it for $k$ future steps. The authors (a) introduce four task variants (O-S, O-O, O-RS, RO-S) to tease apart rule induction vs. state propagation, (b) compare small Transformers, LSTMs, SSMs (Mamba-style), and a recurrent-memory Transformer (ARMT), and (c) test **depth-extension** strategies: Adaptive Computation Time (ACT), GRPO-style RL that induces "thinking tokens'', and token-level Chain-of-Thought (CoT) supervision. Main findings are that single-step prediction and rule recovery are easy, but accuracy **collapses with look-ahead ($k$)**; **depth helps more than width**; ACT gives about "$+1$ effective step''; ARMT reaches $k \approx 2$; GRPO reaches $k=3$ without intermediate labels; supervised token-level CoT attains near-perfect accuracy up to $k=4$. A natural-language proxy ("Handsup'' task) mirrors these trends for today's LLMs.

### Strengths
* **Clear problem decomposition.** The O-S / O-O / O-RS / RO-S variants cleanly separate rule induction from state propagation/chaining.
* **Memorization control.** Disjoint train/test rule sets are an effective way to force generalization beyond lookup.
* **Systematic comparisons.** Side-by-side evaluation of depth vs width, ACT vs GRPO vs CoT, and inclusion of a memory-augmented transformer (ARMT).

### Weaknesses
* **Novelty.** The main conclusions substantially overlap with prior works:

  * **Error compounding / exposure bias** and the failure of one-step training under rollout (Scheduled Sampling from Bengio et al. 2015; DAgger from Ross et al. 2011).
  * **Depth/recurrence/memory** as remedies for multi-step computation (End-to-End Memory Networks; Neural Turing Machine / DNC; Universal Transformer iteration).
  * **Test-time compute via reasoning tokens** (CoT from Wei et al. 2022; self-consistency) and **RL-induced scratchpads**.
  * **Neural CA literature**: learning CA rules and iterative local updates (Wulff & Hertz 1992; Mordvintsev et al. 2020 on Neural CA; follow-ups applying NCA to ARC and other CA/GCA tasks).
    The paper does not adequately acknowledge or distinguish itself from these lines.
* **Compute/accounting confounds.** Comparisons between **deeper nets**, **ACT micro-steps**, **GRPO**, and **token-level CoT** do not rigorously normalize for **effective compute** (parameters × steps × tokens). Some gains may stem from simply doing more computation rather than better mechanisms.
* **Limited ablations on rollout stability.** Known fixes for exposure bias (scheduled sampling, professor forcing, data aggregation) are not evaluated. These could narrow the gap between one-step training and multi-step rollout without the need for CoT or RL.
* **Architectural scope.** Depth vs width conclusions are drawn for a specific small Transformer. Missing baselines include **Universal Transformers** (iterative layers), **RWKV/parallel RNNs**, or **explicit algorithmic modules** (e.g., learned rule tables), which are directly relevant in CA settings.
* **Theory gap.** The compounding-error narrative is intuitive but not formalized (e.g., Lipschitz constants or light-cone arguments for CA). A lightweight analysis could sharpen claims and predict the observed breakpoints.

### Questions
1. **Compute normalization:** How do you normalize **effective compute** across methods (deeper layers vs ACT halting steps vs RL-generated thinking tokens vs CoT length)? Can you report accuracy as a function of *total FLOPs* and *latency*?
2. **Exposure-bias baselines:** What happens if you add **scheduled sampling** or **DAgger-style** data aggregation to the one-step training? Does this close some of the multi-step gap without CoT or RL?
3. **Universal Transformer / iterative layers:** Have you tried an **iterated (weight-tied) transformer** to separate “depth” from parameter count? This seems directly targeted at iterative computation without blowing up parameters.
4. **Why ARMT underperforms on rule inference?** You report ARMT trailing others on O-RS bit accuracy. Is this due to segment design, memory contention, or attention pattern? Any ablation on segment length or memory size?
5. **Rollout training objectives:** Your O-O supervision did not help much. Did you try **teacher-forcing vs free-run** mixtures, **curricula** on k, or **intermediate loss shaping** (e.g., consistency losses) to stabilize multi-step training?
6. **Theoretical framing:** Can you bound the expected **Hamming error growth** under your models’ Lipschitz constants, or use CA light-cone arguments to justify the linear/exponential regimes you observe?
7. **LLM proxy:** In the Handsup task, could **self-consistency** or **verification-guided decoding** close the gap for open-weight models, or is the failure robust even with these stronger test-time procedures?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the multi-step reasoning ability of LLMs in a controlled cellular-automata (1dCA) framework. The objective is to let models learn hidden rules from disjoint train and test sets. The experiments reveal that increasing model depth is crucial, and extending effective depth via recurrence, memory, or test-time compute improves results but remains bounded. In particular, experiments show that 4-layer models struggle with 2-step reasoning tasks, and test-time computing methods such as Adaptive Computation Time (ACT) and Chain-of-Thought (CoT), and achieve 3 or 4 steps. In the discussions, authors raise the claim that their work adds to the evidence that reasoning failures often stem from insufficient depth allocation and sparse optimisation signals.

### Strengths
- Experiments are well-designed, novel (to my understanding).
- Model choices and ablation studies support the central claim of the paper.
- Easy to understand and well-organized writing flow.

### Weaknesses
- The work seems largely empirical, and the novelty of the claim is not much.

The relationship between model depth and reasoning ability has been studied extensively by previous works. The authors provided a very good related works section, but I don't see the central claim to be much different from the established belief in the community. To be exact, this work proposes a new method (1dCA) to study reasoning and model depth and achieves the same conclusion as previous works. The authors provide a contribution list at the end of the introduction, but it only highlights the new 1dCA method. While this is definitely helpful, especially with the comprehensive experiments on different test-time compute methods, I still struggle to see this work's broader significance in the community.

- Some results are a little bit concerning (see questions)

### Questions
1. Besides empirical methods, please try to also clarify the novelties in the understanding of test-time compute and reasoning models (if there are any).
2. How is CoT achieved in the model? I might have missed it, but I can't find details of CoT-related experiments. 
3. k=4's result in Figure 4A is a little concerning. Under the current understanding that model depth has a strong relationship with reasoning ability, we would expect a quasi-linear line for each k until saturation, and this is indeed true for k<=3. However, for k=4, we see an increase until 4 layers, then a flat curve. Please justify why that is the case.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper uses a dataset constructed from one-dimensional cellular automata with controllable complexity to study the effects of model depth on reasoning ability. The authors discuss various architectures, including those incorporating recurrence and memory, and explore their reasoning capabilities within this stylized setting.

### Strengths
1. Clear writing with a good explanation of the setup.
2. A dataset with clearly controllable difficulty is nice for rigorous study.
3. It covers a diverse set of network architectures, which makes the study more convencing

### Weaknesses
There are two major weaknesses:
1. Lack of novelty. My main concern is the lack of novelty, as many similar stylized settings have been explored previously. I do not see any substantial new insights or conclusions beyond showing that the same qualitative story holds for additional architectures.
- Prior work [1] has already examined reasoning capability as a function of model depth while maintaining a comparable FLOPs budget, using a dataset with well-controlled difficulty and stronger resemblance to natural language.
- Work [2] also employed cellular automata datasets to study reasoning ability, showing a clear correlation between model capability and the complexity of the automata.
- Work [3] explicitly analyzed model depth in relation to data complexity under Wolfram’s classification, including skip step predictions.

2. Uncontrolled computational budget. Another major weakness lies in comparing different architectures without controlling for FLOPs or other measures of computational budget. This omission undermines the validity of the conclusions, since model width can also strongly affect performance. This issue was carefully discussed in [1].

[1] Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process, https://arxiv.org/pdf/2407.20311

[2] Intelligence at the Edge of Chaos, https://arxiv.org/pdf/2410.02536v1

[3] Exploring Model Depth and Data Complexity Through the Lens of Cellular Automata, https://openreview.net/forum?id=SGoI97b5KK#discussion

### Questions
1. Could the authors address my concerns above?

2. From [2, 3, 4] it is clear that the complexity varies drastically in the cellular automata dataset, and predicting $t+n$ with $n>1$ is not always harder than $n=1$ due to the complex renormalization nature of the cellular automata. Could the authors discuss this further?


[4] Coarse-graining of cellular automata, emergence, and the predictability of complex systems, https://journals.aps.org/pre/abstract/10.1103/PhysRevE.73.026203

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how large language models acquire and execute multi-step reasoning — specifically, whether their capability stems from memorization or true generalization. To study this, the authors design a new benchmark, 1dCA, which evaluates models’ ability to generalize beyond training distributions. Experimental results show that smaller models, such as standard Transformers, do exhibit non-trivial generalization, suggesting that their reasoning performance is not solely the result of memorized patterns.

### Strengths
1. The study tackles an important and timely question in reasoning research—whether LLMs’ multi-step reasoning ability reflects true generalization rather than mere memorization
2. Well-designed synthetic benchmarks: Allows controlled measurement and ablations not feasible on real datasets.
3. The analysis provides actionable guidance on how architectural design and training choices

### Weaknesses
In the Handsup experiments, simple CoT prompting is insufficient to reveal the model’s true reasoning capability. The experiments conducted on large models are not very convincing. For this type of task, the performance after SFT or in-context learning can differ substantially from zero-shot results. Moreover, Transformers achieve higher accuracy largely because they are trained on similar datasets, which does not demonstrate any fundamental difference between Transformers and LLMs, nor does it imply inherent limitations of LLMs. As a result, the experimental setup does not adequately support the subsequent interpretations regarding LLM behavior.

### Questions
It would be helpful to provide in-context learning results for LLMs on the Handsup benchmark. I suggest explicitly including the rule in the few-shot demonstrations to examine whether the model’s behavior changes when the underlying procedure is made more explicit.

### Soundness
3

### Presentation
3

### Contribution
3

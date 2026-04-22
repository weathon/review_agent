# Scaling Short-Term Memory of Visuomotor Policies for Long-Horizon Tasks

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
Many robotic tasks demand short-term memory, whether it’s retrieving objects that are no longer visible or turning off an appliance after a certain amount of time. Yet, most visuomotor policies remain myopic, relying only on immediate sensory input without leveraging past experiences to guide decisions. We present PRISM, a transformer-based architecture for visuomotor policies to effectively use short-term memory via two key components: (i) gated attention, which selectively filters retrieved information to suppress irrelevant details, and (ii) a hierarchical architecture that first compresses local interactions into compact tokens and then integrates them to capture temporally extended dependencies. Together, these mechanisms enable us to scale short-term memory in visuomotor policies for up to two minutes at five frames per second, an order of magnitude longer than previous approaches. To systematically evaluate memory in visuomotor control, we introduce ReMemBench—a benchmark of eight diverse household manipulation tasks spanning four categories of short-term memory—designed to foster general memory mechanisms rather than siloed, task-specific solutions. PRISM consistently outperforms prior works, including transformer-based visuomotor policies with short-term memory, recurrent architectures, and other short-term memory-management strategies. In ReMemBench and real-world evaluation, PRISM achieves an absolute improvement of 11–15 points over the strongest baseline. On RoboCasa and LIBERO, it further yields 11–14-point gains over its no-memory variant and outperforms strong fine-tuned VLA baselines such as GR00T-N1-3B and OpenVLA, without any pretraining. Together, PRISM and ReMemBench establish a foundation for developing and evaluating short-term memory–augmented visuomotor policies that scale to long-horizon tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper addresses the limitation that current visuomotor policies for robotic manipulation are myopic and cannot effectively use short-term memory for long-horizon tasks. The authors introduce PRISM, a transformer-based policy architecture that scales short-term memory through two key innovations: (1) gated attention that selectively filters irrelevant information from memory to prevent spurious correlations, and (2) a hierarchical architecture that compresses local interactions before global attention to reduce computational cost. To systematically evaluate memory capabilities, they also propose REMEMBENCH, a benchmark with 8 household manipulation tasks spanning 4 cognitive memory categories (spatial, prospective, object-associative, and object-set). PRISM achieves 2× the success rate of state-of-the-art baselines on REMEMBENCH and real-world tasks, demonstrating that effective short-term memory mechanisms can significantly improve performance on tasks requiring temporally extended reasoning.

### Strengths
- The paper provides an excellent explanation and decomposition of short-term memory into four cognitive categories (spatial, prospective, object-associative, and object-set) in Sec.4, grounded in cognitive science literature. This systematic approach to REMEMBENCH design encourages development of general short-term memory mechanisms rather than task-specific solutions.

- PRISM demonstrates substantial and consistent improvements across multiple evaluation settings. The real-world validation is particularly valuable for demonstrating practical applicability.

### Weaknesses
- Insufficient ablation of key hyperparameters. Specifically, no ablation on the compression factor $m$.

- Presentation and formatting issues. The following issues raise concerns about the overall polish and readiness of the submission:
  - Figure-text overlap around lines 422-423 suggests rushed preparation
  - Broken citation for CrossMAE (line 675-676: "*CrossMAE* (?)")
  

- Unclear future direction on long-term memory: The paper defines long-term memory as "cross-episode retrieval" (line 479-482) but provides insufficient explanation of how this would function in their framework. The distinction between short-term (within-episode) and long-term (cross-episode) memory deserves clearer articulation, particularly regarding whether the authors envision something like retrieval-augmented generation for robotics or continual learning scenarios with replay buffers.

### Questions
- The paper categorizes short-term memory into four types (spatial, prospective, object-associative, object-set) based on cognitive science literature. Are these four categories exhaustive, or are they a representative subset of a broader taxonomy? Could you clarify whether REMEMBENCH aims to cover all major categories of short-term memory relevant to robotics, or if additional categories might be added in future work? What criteria from cognitive science guided the selection of these specific four categories?

- Several tasks might potentially be solved without explicit memory through richer state representations. For example, in "Wash and Return to Container" (Fig.4), an observation function that detects wetness could disambiguate "just washed" vs. "needs washing" states, replacing the need to remember the washing action. Similarly, in "Cook Meat," a visual cue indicating stove temperature or cooking duration might substitute for remembering when the stove was turned on. This raises a fundamental question: what is the principled distinction between tasks that inherently require memory vs. tasks solvable through better perception/state engineering or other alternative techniques?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PRISM, a transformer-style visuomotor policy that scales short-term memory for long-horizon manipulation via two components: (i) feature-aware gating that selectively filters history tokens based on their relevance, and (ii) a hierarchical memory that locally compresses tokens and performs global attention over compact summaries, reducing cost for long contexts. The authors introduce REMEMBENCH, eight kitchen-style manipulation tasks spanning four short-term memory categories—spatial, prospective, object-associative, and object-set—constructed so that success depends on within-episode recall rather than single-frame observability. PRISM is trained with imitation learning (IL) under partial observability from closed-loop observations (RGB + proprioception). Empirically, PRISM outperforms LSTM, Mamba, and a PTP-trained baseline on REMEMBENCH, shows some gains on unrelated atomic tasks, and demonstrates a modest but meaningful improvement on a real-robot task. Efficiency plots indicate substantially lower VRAM usage and latency at long context lengths versus non-hierarchical models.

### Strengths
* **Problem framing.** Treats IL under partial observability explicitly, with tasks that isolate different short-term memory demands.
* **Design.** Pragmatic combination of feature-aware gating and hierarchical summaries that targets the compute/latency pain of long contexts.
* **Empirics.** Consistent improvements on memory-dependent tasks; efficiency gains at inference; some real-robot evidence.

### Weaknesses
* **Missing transformer IL baselines.** The paper compares PRISM only to LSTM, Mamba, and a PTP-trained baseline, but omits transformer-based IL policies tailored to partial observability (e.g., windowed causal Transformers, Transformer-XL/GTrXL, linear-attention Transformers). Without these, it is unclear whether PRISM outperforms standard transformer sequence models for IL-POMDPs.
* **Memory architecture vs action space.** The authors note that many prior embodied works used discrete actions (true, e.g., Scene Memory Transformer). However, the key issue here is memory under partial observability, not the action parameterization. Transformer memory mechanisms—windowed, segment-recurrent (Transformer-XL/GTrXL), global/episodic (scene/episodic memory), and linear-attention—are action-agnostic and directly apply to continuous-action IL by swapping only the output head (Gaussian or diffusion). Omitting these transformer memory baselines—even with continuous heads—leaves the necessity of PRISM’s gate+hierarchy untested.
* **Prior-work coverage.** Related work should explicitly cover transformer memory for embodied agents (e.g., scene/episodic memory transformers) and segment-recurrent transformers for partial observability. In vision-language navigation (VLN), Episodic Transformer encodes the entire episode history (observations/actions + language), showing that long-horizon transformer memory is crucial for compositional tasks. Although the present method does not use language and outputs continuous actions, these architectures are action- and modality-agnostic and should be acknowledged and positioned.

### Questions
1. **Transformer baselines (continuous actions).** Please add:
   **(a)** Windowed Transformer BC; **(b)** Transformer-XL/GTrXL BC (segment length S, memory M); **(c)** a linear-attention Transformer BC.
   Use continuous outputs (Gaussian or diffusion), identical inputs/loss to PRISM, and match parameters, history length, and training budgets.
2. **Scene-memory variant.** On a small subset (e.g., one task per memory category), include an SMT-style global-memory baseline (global attention over stored observation embeddings) to bound accuracy/compute trade-offs relative to PRISM’s hierarchy.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
A new transformer architecture, PRISM, and a robotics benchmark for memory-dependent tasks, ReMemBench, are proposed.

### Strengths
**Strengths:**
1. Memory-augmented architectures for robotics are an important and rapidly developing direction.
2. The paper is well-written and easy to read.
3. A new architecture and benchmark for memory-dependent robotic tasks are presented.
4. Experiments were conducted on a real robot.

### Weaknesses
**Weaknesses:**
1. There is no comparison of ReMemBench with the Mikasa-Robo benchmark [1], which is specifically designed to test memory mechanisms in robotics.
2. Only simple baselines, not designed for memory-dependent tasks, are used for comparison. Comparisons with specialized baselines, such as RATE [2] and GTrXL [3], are necessary.
3. The main experiments are conducted only on the benchmark proposed in the paper. Comparisons on existing benchmarks, such as Mikasa-Robo [1] and MemoryBench [4], are needed.
4. To ensure that the memory mechanism does not impair the model on simple tasks, evaluation on existing benchmarks, such as RLBench [5], LIBERO [6], and SimplerEnv [7], is critically important. Current results on RoboCasa are insufficient, especially since they are presented without the context of other approaches’ performance.
5. Detailed information on data collection and model training, which are necessary for reproducibility, is missing.

**Minor issues:**
1. Figure 5 overlaps with the text – line 422
2. Broken link to CrossMAE – line 676

**References:**
1. Cherepanov, Egor, et al. "Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning." arXiv preprint arXiv:2502.10550 (2025).
2. Cherepanov, Egor, et al. "Recurrent action transformer with memory." arXiv preprint arXiv:2306.09459 (2023).
3. Parisotto, Emilio, et al. "Stabilizing transformers for reinforcement learning." International conference on machine learning. PMLR, 2020.
4. Fang, Haoquan, et al. "Sam2act: Integrating visual foundation model with a memory architecture for robotic manipulation." arXiv preprint arXiv:2501.18564 (2025).
5. James, Stephen, et al. "Rlbench: The robot learning benchmark & learning environment." IEEE Robotics and Automation Letters 5.2 (2020): 3019-3026.
6. Liu, Bo, et al. "Libero: Benchmarking knowledge transfer for lifelong robot learning." Advances in Neural Information Processing Systems 36 (2023): 44776-44791.
7. Li, Xuanlin, et al. "Evaluating real-world robot manipulation policies in simulation." arXiv preprint arXiv:2405.05941 (2024).

### Questions
1. How does the task-type categorization in ReMemBench correspond to the memory types in Mikasa-Robo [1]?
2. How does the approach perform compared to strong baselines such as RATE [2] and GTrXL [3]?
3. How and in what quantity was the data collected for training?
4. Do the training details in Section B of the appendix apply only to PRISM, or to all baselines?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This manuscript brings a new benchmark `ReMemBench` and also solves the long context limitation of the transformer based policies by introducing an attention mechanism.

### Strengths
New Benchmark is provided.

Manuscript is easy to read and all the steps are sound and intuitive.

Real-robot deployment.

State of the art results.

### Weaknesses
Methodological novelty is minimal. For example, using attention mechanism to remove distractions in test time in imitation learning for visuomotor policies has been demonstrated before. For example: `“Pay attention!-
robustifying a deep visuomotor policy through task-focused visual
attention,” CVPR, 2019`.

### Questions
1- How the model can decide if everything from a timestamp is relevant and needs to be tokenized or part of it. And if this is configurable or automated (learned) by the model. What is the impact of it? Meaning what if we have stricter attention mechanism that is very conservative in selecting information, vs having a more free flow of information into the memory? How does it impact the performance on the benchmarks?

2- I am wondering if the authors tried applying noise to the attention mechanism during training to make it more resilient during test time.

3- [Out of curiosity, authors can skip this question] Can the proposed method be extended to novel objects during test time? How much drop we expect if the task requires manipulating a novel object, not seen in training.

4- Line 249: I am wondering if the design of g(x) like number of parameters or other designs rather than simple MLP can impact the final results.

### Soundness
3

### Presentation
3

### Contribution
2

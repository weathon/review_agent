# To Infinity and Beyond: Tool-Use Unlocks Length Generalization in State Space Models

- Decision: Accept (Oral)
- Scores: 8, 4, 8, 8

## Abstract
State Space Models (SSMs) have become the leading alternative to Transformers for sequence modeling tasks. Their primary advantage is efficiency in long-context and long-form generation, enabled by fixed-size memory and linear scaling of computational complexity. We begin this work by showing a simple theoretical result stating that SSMs cannot accurately solve any "truly long-form" generation problem (in a sense we formally define), undermining their main competitive advantage. However, we show that this limitation can be mitigated by allowing SSMs interactive access to external tools. In fact, we show that given the right choice of tool access and problem-dependent training data, SSMs can learn to solve any tractable problem and generalize to arbitrary problem length/complexity (i.e., achieve length generalization). Following our theoretical finding, we demonstrate that tool-augmented SSMs achieve remarkable length generalization on a variety of arithmetic, reasoning, and coding tasks. These findings highlight SSMs as a potential efficient alternative to Transformers in interactive tool-based and agentic settings.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies **length generalization** of **State Space Models (SSMs)** and recurrent controllers on long-form generation tasks. It formalizes a class of **generalized SSMs (GSSMs)**, that are a finite-state controllers whose memory does **not** grow with sequence length, and a family of **long-form generation** problems where the support size of valid outputs increases with task complexity. Two central theoretical results:

* **Impossibility (Thm 2.1):** Any CoT-only or **single-turn** tool-use GSSM must fail on long-form generation beyond some size with a specific error.
* **Possibility (Thm 2.2):** With **interactive** tool-use (a simple read/write pointer-tape oracle), there exist training distributions under which a GSSM **length-generalizes** on any *computationally tractable* long-form task, i.e., learns a constant-state controller that operates unbounded external memory.

Empirically, the paper shows that SSMs/RNNs trained on **interactive, ReAct-style** trajectories strongly extrapolate on:

* synthetic arithmetic (multi-digit addition/multiplication up to 1K digits),
* reasoning (logical DAG evaluation to 1K nodes; some generalization on Tower of Hanoi),
* a **coding** bug-fix task with bash/file tools (run→inspect→patch→re-run), where Mamba trained on **interactive/distilled** traces maintains high pass rates well beyond training repository sizes, while **single-turn** tool-use collapses and local-attention Transformers degrade.

**Contributions:**

1. A clean **formalization** of long-form generation and GSSMs.
2. A **negative result** showing fixed-memory controllers (incl. CoT or single-turn tool calls) cannot solve such tasks at scale.
3. A **positive result**: interactive tool-use + external memory is sufficient for “infinite” length generalization.
4. Broad **experimental evidence** across arithmetic, reasoning, and coding, with a realistic agentic setup and trajectory distillation.

### Strengths
* **Originality:** Elegant formalization of long-form generation via support-growth; clean GSSM abstraction; sharp **necessary** (no interaction) vs **sufficient** (interactive tools) results.
* **Quality:** Both **theory and practice** reinforce each other; experiments span arithmetic, graph reasoning, and a realistic coding benchmark with real tool commands and agent distillation.
* **Clarity:** Clear separation of CoT-only, single-turn, and interactive regimes; easy-to-follow trajectory visualizations.
* **Significance:** Offers a compelling recipe for **length-generalizing agents**: small SSM controller + interactive memory/tooling + trajectory imitation. Practical implications for efficient, long-horizon tasks.

### Weaknesses
* **Baselines & fairness:** Transformer comparisons are mainly **local/sliding-window**. Missing head-to-head with **memory-augmented** or **recurrent** Transformer variants (e.g., retrieval-augmented, ring/segment memory, recurrent attention, kNN-LM, RMT) that also enable externalized state.
* **Policy-execution gap:** Training uses teacher forcing on trajectories; it would help to quantify compounding-error effects when **executing** the learned policy end-to-end under stochastic observations.
* **Scaling breadth:** Coding experiments use 1.4B-scale models; it would be informative to include **larger SSMs** and show whether gains persist or amplify vs similarly scaled Transformer agents with function-calling tool APIs.

### Questions
1. **Scope of Theorem 2.2:** The result asserts existence of training distributions {P_n}. How constructive is the procedure? Can you characterize **sample complexity** (number of transitions/examples) needed to learn a given algorithm’s controller?
2. **Transformers + external memory:** How do results change when comparing against **Transformers with the same pointer-tape tool** and **interactive** protocol? Are failures due to representation (attention) or optimization/data?
3. **Tool noise & latency:** Real tools return noisy, delayed, or partially structured outputs. How sensitive is the controller to **observation noise**, **command failures**, or **latency**?
4. **Generalization beyond seen transitions:** For multiplication/addition, do models truly learn **state transition rules**, or are there subtle failure modes (e.g., rare carry chains)? Any adversarial or **worst-case** tests?
5. **Safety & security:** In coding, could the learned policy execute unsafe commands? Any recommended **sandboxing** or **guardrails** for deploying SSM agents with system tools?
6. **When is single-turn enough?** Can you give some task classes where single-turn (or CoT-only) suffices, perhaps when supp_alpha grows slowly or when a single observation is maximally informative?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proves that SSMs cannot solve long-form generation tasks due to their fixed-size memory, but that interactive tool-use enables them to achieve length generalization—allowing SSMs to handle arbitrarily long and complex problems by leveraging external memory.
Empirically, the authors show that tool-augmented SSMs (e.g., Mamba) outperform Transformers in arithmetic, reasoning, and coding tasks, demonstrating scalability and efficiency advantages when used as interactive agents rather than standalone models.

### Strengths
- The paper proves that SSMs, despite their efficiency, cannot solve long-form generation tasks under standard or single-turn settings.
- The paper validates the theory on diverse task categories, including arithmetic, algorithmic reasoning, and real-world coding.

### Weaknesses
- The experiments primarily use small or mid-sized models. These experiments cannot prove if the proposed methods could generalize to larger scales.
- The paper evaluates on synthetic arithmetic, algorithmic, and coding tasks only. There is no testing on natural-language, multimodal, or real reasoning benchmarks.
- Although tool-use extends memory, it introduces I/O and latency overhead, which may offset the computational gains of SSMs in real-time systems.
- Typos, such as "node node" in line 398

### Questions
- How this proposed methods could generalize to real-world reasoning or language understanding tasks?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper argues theoretically and empirically for augmenting SSM architectures with tools that can function as an external memory. In principle, this provides a simple way to overcome the memory limitations of recurrent architectures (as opposed to the more conservative approach of hybridizing with attention layers). The authors first formalize the value of memory tools with some basic theory and then show empirically that SSMs using external memory tools can achieve good performance and length generalization on synthetic tasks. They also show an encouraging proof-of-concept result where (distilled) SSM coding agents with tool use better generalize to large codebases than transformer coding agents.

### Strengths
1. The overall argument of the paper is well motivated and presented. I appreciate the principled framing even if the theory is a bit surface-level
2. The question of how to extend SSMs with additional memory is very timely, and the community has not converged on a solution. One approach (e.g., used by Qwen-3-Next) is to hybridize by mixing transformer and SSM layers. But this paper proposes and evaluates a more radical solution without attention, which is valuable.
3. The empirical results in the paper are quite interesting. In particular, the empirical results showing better length generalization for distilled SSM coding agents with tool use (vs. transformer agents) is a promising proof of concept that the synthetic-task results are practically relevant

### Weaknesses
### Long-Form Task Definition

A task is defined to be long-form generation if the output size (number of possible outputs with probability mass) grows with input size n. I wouldn't necessarily agree with this definition. What if there is only one choice for y given x, but that y has to be large, and it's length scales with y - why don't we consider this long-form generation?

### Theory Reinvents the (Automata-Theoretic) Wheel

The theoretical results are somewhat superficial. While I appreciate the principled framing, they essentially boil down to an argument that, if a model has fixed memory, tools that serve as interface to additional memory are helpful for memory-intensive tasks.

> Note that any model that has fixed memory as a function of the sequence length satisfies the definition of a GSSM.

Your definition of a generalized state-space model with constant |S| is just reinventing the notion of a probabilistic finite automaton (with a potential distinction based on the long-form task definition thing mentioned above, though I'm not sure it's a meaningful one). See, for example, [Merrill et al., Section 3](https://arxiv.org/abs/2404.08819) for a discussion of the connection between SSMs, state tracking tasks, and automata/regular languages.

In formal language theory terms, your notion of GSSMs with interactive tool use has a natural interpretation as finite automata augmented with some tool data structure, similar to how pushdown automata are automata augmented with a stack data structure (and this lets pushdown automata solve additional languages that require more memory). Interaction is essentially giving the model more memory through interaction with an external data structure. I think making the connection between GSSMs and automata explicit and discussing this interpretation of tools as data structures would help clarify your argument.

Theorem 2.1 is just the Myhill-Nerode theorem (a probabilistic extension of the basic deterministic one)

Theorem 2.2: it's a bit overly verbose to talk about learning algorithms when this is essentially an expressivity construction. i.e., the learning algorithm and training data you have here is basically just hard-constructing an expressivity construction?

### Questions
Table 1: clarify this is with tool use. To really make your point that tools are necessary, you could show performance without tools.

what does it mean to move a pointer left or right? Increment/decrement it?

It seems your "logical graph problem" is really the circuit evaluation problem, which is known to be P-complete and thus outside the representational power of transformers unless the complexity classes TC0 and P collapse. Furthermore, even with CoT, it would be very surprising if transformers could solve this problem with a small number of steps (e.g., even with log^k n steps, CoT transformers cannot solve this problem unless TC0 and P collapse).

Do you have opinions on whether SFT alone would suffice to encourage tool use with SSM models? Or would RL be helpful for teaching a model to acquire tool use?

The following statements at the abstract level could be made more clear, since it's not sure what long-form generation or tractable mean at this point:

> We begin this work by showing a simple theoretical result stating that SSMs cannot accurately solve any long-form generation problem, undermining their main competitive advantage.

> SSMs can learn to solve any tractable problem and generalize to arbitrary problem length/complexity

Regarding length generalization with SSMs:
> SSMs have been shown to display robust length generalization capabilities in certain cases.

IMO you could make a stronger claim. It seems quite robust at this point that SSMs show strong length generalization relative to transformers. e.g., the gated delta net, path attention, and samba papers show zero-shot results consistently outperforming transformers on perplexity evals and RULER.

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
This paper investigates the length generalization capabilities of State Space Models (SSMs) in long-form generation tasks. The authors prove that SSMs cannot solve such tasks with bounded memory alone, but can achieve length generalization when augmented with interactive tool use. Experiments on arithmetic, reasoning, and coding tasks demonstrate that tool-augmented SSMs can generalize from short training sequences to much longer test sequences, outperforming Transformers in certain settings.

### Strengths
1. The paper provides formal definitions (GSSM, long-form generation tasks) and proves a crisp dichotomy: SSMs fail without interaction but succeed with interactive tool access.

2. The arithmetic experiments show remarkable generalization.

3. The experimental coverage is quite comprehensive. The paper evaluates multiple architectures (Mamba, LSTM, GRU, Pythia, Mistral) across diverse tasks (arithmetic, logic, coding), with both synthetic and agent-generated trajectories.

4. Connecting SSM limitations to tool-augmented agents is timely and relevant, given the growing interest in agentic AI systems.

5. The progression from theory to synthetic experiments to realistic coding tasks is logical and easy to follow.

### Weaknesses
1. The theory relies on training with complete Turing machine traces, but the experiments use hand-crafted or distilled trajectories with no analysis of this gap.

2. The key coding experiment uses a synthetic benchmark with a single bug pattern, not a complex real-world benchmark (e.g., SWE-bench).

3. Missing Sample Complexity Analysis: The theory proves learnability (Lemma B.1) but gives no practical bounds on the sample size (m) or task complexity (n_0) required to achieve it.

4. Incomplete baselines: The paper omits comparisons to SOTA hybrid SSM-Attention models (e.g., Jamba), which are a key alternative for efficient long-context modeling.

### Questions
1. How do the hand-crafted trajectories in arithmetic experiments satisfy (or approximate) the Turing machine trace assumption in Theorem 2.2?

2. In the coding experiments, you filter for successful trajectories. How does this affect the model's ability to handle errors or suboptimal tool use at test time?

3. Have you tried evaluation on actual software engineering tasks (e.g., SWE-bench lite)? What performance would you expect?

### Soundness
3

### Presentation
3

### Contribution
3

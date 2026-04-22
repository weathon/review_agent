# Ludax: A GPU-Accelerated Domain Specific Language for Board Games

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Games have long been used as benchmarks and testing environments for research in artificial intelligence. A key step in supporting this research was the development of game description languages: frameworks that compile domain-specific code into playable and simulatable game environments, allowing researchers to generalize their algorithms and approaches across multiple games without having to manually implement each one. More recently, progress in reinforcement learning (RL) has been largely driven by advances in hardware acceleration. Libraries like JAX allow practitioners to take full advantage of cutting-edge computing hardware, often speeding up training and testing by orders of magnitude. Here, we present a synthesis of these strands of research: a domain-specific language for board games which automatically compiles into hardware-accelerated code. Our framework, Ludax, combines the generality of game description languages with the speed of modern parallel processing hardware and is designed to fit neatly into existing deep learning pipelines. We envision Ludax as a tool to help accelerate games research generally, from RL to cognitive science, by enabling rapid simulation and providing a flexible representation scheme. We present technical notes on the description language and compilation process, along with speed benchmarking.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Ludax, a framework that automatically generates vectorizable JAX environments for board games.
It introduces a dedicated game description language (DSL) and a pipeline that converts this DSL into executable JAX code.
The authors also highlight several design techniques for maintaining JAX jittability—ensuring static shapes and pure functional style.
In experiments with selected board games, Ludax-generated environments achieved comparable speed to hand-crafted JAX baselines and significantly outperformed the non-vectorized Java-based framework Ludii, from which Ludax draws inspiration.

### Strengths
While neither DSL-based game generation nor vectorized environments are entirely new concepts, their combination is valuable and timely.
Conceptually, it matters because recent advances in reinforcement learning algorithmsrely heavily on vectorizable simulators such as PQN, and technically, because implementing diverse games in JAX is non-trivial due to strict requirements for jittability (static array shapes, pure functional programming).

Ludax effectively addresses these challenges by offering a DSL and a code generation pipeline tailored for JAX.
The paper clearly explains how to define atomic functions, map them to DSL primitives, and generate executable JAX code.
The experiments convincingly show that the generated code is both fast and compatible with RL training pipelines, as evidenced by the consistent learning performance compared to manually written environments.

### Weaknesses
As the authors themselves note, limited generalizability is the main bottleneck.
To define a new game, one must manually implement new atomic functions that are both compositional and general enough to be reused for other games.
However, many game rules are highly domain-specific and cannot be easily decomposed into such atomic components.

Thus, Ludax excels at interpolating within the family of games expressible by existing atomic functions (e.g., two-player, rectangular board, placing or sliding pieces, line-based victory conditions such as Hex, Gomoku, Connect Four, or Yavalath).
But it struggles to extrapolate to more complex games requiring novel logic.
Although the authors mention that adding new atomic functions can extend Ludax, designing such composable primitives remains a core research challenge rather than an engineering task.
Hence, this limitation should not be overlooked as a minor or easily solvable issue.

### Questions
- Do the authors have any insight into how to incorporate complex, non-compositional game logic into Ludax’s atomic framework? Designing such reusable primitives seems to require substantial domain intelligence.

- As a concrete example, could the authors show how Ludax could be extended to implement a more complex game like Go, which lies close to its current scope but introduces challenging mechanics such as group connectivity, liberties, and ko rules?

### Soundness
2

### Presentation
2

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
In this paper, the authors present a DSL, Ludax, for board games that combines the expressiveness of Ludii with the GPU performance of PGX. Ludax's design draws heavily on the Ludii game description DSL, but changes are introduced to improve performance and simplify the description of certain meaningful game states. Ludax game descriptions are compiled using the Lark Python library, and operators are processed from the bottom up into atomic JAX functions. Some of these JAX functions are converted into maps from the current game state into a Boolean truth value and passed up the parse tree. A major improvement of Ludax is the transformation of iterative procedures related to game states into matrix-value functions amenable to fast precomputation using GPUs, similar to PGX. The authors support the efficiency of Ludax in the evaluation section by comparing the performance with PGX and achieving comparable results while maintaining similar convergence trajectories during training.

### Strengths
- Ludax combines the representation power of Ludii for board games with the performance improvements of a bespoke implementation written by hand in PGX. This strategy will allow researchers to investigate a wide range of existing board game dynamics quickly and implement new games amenable to accelerated training.
- The performance profiles comparing Ludii, Ludax, and PGX illustrate nicely the level of training throughput achievable using Ludax, as shown in Figure 2.
- The fidelity of the game dynamics after lowering to JAX using Ludax is illustrated in Figure 3 with very similar training dynamics when using Ludax and PGX.

### Weaknesses
- The biggest weakness is the lack of breadth of the current features of Ludax compared to Ludii. Although Ludax does provide support several games, the contribution would be significantly higher if more environments were supported, given the number of games available in Ludii.
- The presentation of Ludii in Section 3 is extensive, but an extensive portion is dedicated to describing existing components instead of focusing on the contributions of Ludax.
- This paper seems like an interesting direction, but the current limitations make it fall short of the contribution threshold for the current venue, in my opinion.

### Questions
- What are the primary hurdles impeding the support for more Ludii environments?

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
This paper provides for a jax accelerated RL framework for games. Ludax is a domain specific language for games, that allows for the description of  hundreds of games, and then is automatically optimized for Jax.

### Strengths
The paper is well written, and the motivations are clear. Games are certainly important in the history of AI, and have led to many breakthroughs. Further, a language like Ludax has uses beyond games, and can express a variety of problems, and also be useful in analyzing RL generation. or rapid testing on procedurally generated game environments. Impressively, the speed of Ludax games is comparable with specific game jax implementations. The supplemental materials were clear, and the code was easy to review.

### Weaknesses
My main issue with the paper is its limited novelty. While section 3.4 does discuss some non trivial differences from Ludii, fundamentally Ludax is simply a Jax port of Ludii. While this is certainly a very useful achievement, and I am sure it will be used, I do not believe that slightly modifying an existing tool to work with Jax merits a paper at ICLR. I commend the authors for their quality work. and recommend submitting to a more appropriate venue.

### Questions
I would ask the authors to elaborate on the novelty of this work, beyond reworking some parts of Ludii to work with Jax (section 3.4).

### Soundness
4

### Presentation
4

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
The paper presents Ludax, a domain-specific language (DSL) for describing two-player, perfect-information board games that compiles into GPU-accelerated JAX environments. It combines the human-readable syntax inspired from Ludii with PGX, enabling rapid simulation and model training across a class of placement/capture/connection games for LLMs. The compilation process transforms high-level game rules into composable, JIT-compilable JAX functions with optimizations such as precomputed line indices and dynamic state tracking. The conducted experiments show significant throughput improvement compared to Ludii and demonstrate the capability of RL agent training.

### Strengths
1. Novelty: Ludax successfully combine PGX with Ludii to allow them compliment to each other and achieve both generality and acceleration via principled code generation.

2. Usability: The environment is able to directly fit into JAX-based RL pipelines and the game descriptions closely mirror natural language lowering the technical requirements for domain experts and researchers to prototype variants. The scalability and structured description make it possible to train LLM for game generation, reasoning and potentially world modeling.

### Weaknesses
1. Hard Limitation: Limited to single-piece, placement/capture games on regular boards. No support for multi-piece types, stacking, promotion, or irregular geometry.

2. Benchmark Analysis: No memory profiling or compile-time analysis. No ablation of optimizations (precompute vs. naive). No large-board stress test (e.g., 19×19 Pente).

2. LLM integration: No demonstration of LLM-guided search over Ludax space (e.g., evolving win conditions) or RL generalization across generated variants.

### Questions
Address weakness above.

### Soundness
3

### Presentation
2

### Contribution
3

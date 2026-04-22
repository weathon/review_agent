# Investigating Memory in RL with POPGym Arcade

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 6, 0

## Abstract
How should we analyze memory in deep RL? We introduce mathematical tools for fairly analyzing policies under partial observability and revealing how agents use memory to make decisions. To utilize these tools, we present POPGym Arcade, a collection of Atari-inspired, hardware-accelerated, pixel-based environments sharing a single observation and action space. Each environment provides fully and partially observable variants, enabling counterfactual studies on observability. We find that controlled studies are necessary for fair comparisons, and identify a pathology where value functions smear credit over irrelevant history. With this pathology, we demonstrate how out-of-distribution scenarios can contaminate memory, perturbing the policy far into the future, with implications for sim-to-real transfer and offline RL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents a POPGym Arcade benchmark and memory evaluation tools in RL. The empirical studies demonstrate that out-of-distribution scenarios can contaminate memory.

### Strengths
- This paper introduces POPGym Arcade, an Atari-inspired benchmark, providing fully and partially observable environments.
- This paper proposes memory analysis tools to isolate and study memory capabilities and understand what memory models learn.

### Weaknesses
- The overall readability of the paper is poor, and the motivation is not clearly articulated.
- Only one algorithm (PQN) is used to evaluate the proposed benchmark. It would be convincing to include more memory-focused methods, such as R2I [1],
- The current tasks are relatively simple, lacking more challenging ones that could more thoroughly evaluate the model's memory capabilities, such as MemoryMaze [2].
- As shown in Figure 5, the added noise is somewhat too large. It would be better to show results for OOD scenarios with less severe noise to provide a more convincing evaluation.

References:

[1] Samsami et al. "Mastering Memory Tasks with World Models", ICLR, 2024.

[2] Pasukonis et al. "Evaluating Long-Term Memory in 3D Mazes", arXiv preprint arXiv:2210.13383, 2022.

### Questions
- Have you tried applying memory models such as Mamba [1] or Transformer [2] to the proposed benchmark?
- Have you tried adding background distractions, similar to those in the Distracting Control Suite [3], to assess the robustness of memory-based agents to perturbations?

References:

[1] Dao et al. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML, 2024.

[2] Ashish et al. "Attention Is All You Need", NIPS, 2017.

[3] Stone et al. "The Distracting Control Suite -- A Challenging Benchmark for Reinforcement Learning from Pixels", arXiv preprint arXiv:2101.02722, 2021.

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
Authors conduct controlled analysis on the role of memory in RL environments.
They define evaluation tools like the observability gap, the memory bias, and the recall density.
They design POPGym Arcade of 10 base environments of MDP/POMDP twins to conduct controlled experiments to showcase the utility of the framework.

### Strengths
Originality and significance: The motivation of this work stands. It bothers me too that existing literature often merely touch over the role of memory with guesswork, so to me this is a really nice attempt to formally diagnose and scrutinize the learning process empirically in RL with strongly controlled environments.

Quality and clarity: The theoretical framing in Sec 4 and the design in Sec 5 are both solid, reasonable and thoughtful.

### Weaknesses
The main complaint I have is the algorithm coverage and depth of analysis: evaluating only on PQN (and its variation) seems lacking to comment in general about RL and memory. Could you try include at least one model-based RL algorithm? What about one that requires replay buffer (and what about the role of different sampling strategies?) There are lots of interesting investigations one can perform with your excellent environment setup.

### Questions
1. How do you decide which part of MDP is hidden in their twin POMDP? For example, one could hide the cartpole all together, or one could hide only the pole but not the cart. To me, partial observability is a continuous spectrum and I imagine some hidden states are easier/harder to learn from. Or did you have a general protocol when you convert *any* MDP task to their POMDP compartment?
2. I understand 10 environments are not a lot to work with, but do you see consistent patterns when slicing across dimensions like dense vs. sparse rewards wrt. value smearing?

Suggestions on the writing (not weaknesses and not reasons to reject)
1. "POPGym Arcade is Fast" is a nice-to-have (who doesn't want their eval to run fast?), but is tangential to your analytical contribution. I recommend downplaying the performance, and instead use that space for more memory related experiments and analysis.
2. I found Sec 6 (Experiments) and Sec 7 (Results) hard to read, as the research question, the analysis, and the plot are scattered so I had to jump back and forth multiple times (for each RQ). If somehow one could reserve the locality it would be clearer.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper provides a new benchmark for research in partially observable reinforcement learning tasks. The benchmark contains re-implementation of 10 common environments in jax, making them hardware acceleratable. The paper also provides implementations of various sequence models combined with the ''PQN" algorithm. Finally, the paper curates 4 metrics (memory gap, observation gap, and input sensitivities) to evaluate different facets of memory-based RL agents.

### Strengths
The main strengths of the paper are:
1) Simplicity: The benchmarks are simple and easy to reconfigure. 
2) Speed: Because these POMDPs are implemented using jax, they allow researchers to quickly iterate on ideas.
3) Implementation: Multiple memory based models have already been implemented, and the code seems to be readable and clean.

### Weaknesses
The papers does not have any major weakness as it does a good job in a niche topic - providing fast benchmark for POMDP research.

### Questions
The minor weaknesses of the paper are:
1) I don't understand what Figure 3 is trying to portray. Could you modify the plot to better visualize the data or add a detailed caption?
2) Why didn't the authors implemented the transformer architecture?
3) Figure 4 is also pretty unclear. Is it a POMDP or an MDP? If it is an MDP, then what is the point of using a memory model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper introduces a new benchmark, POPGym Arcade, and a corresponding set of mathematical tools designed to analyze the role of memory in deep reinforcement learning agents. The core contribution is POPGym Arcade: a collection of 10 hardware-accelerated, Atari-inspired pixel-based environments (e.g., Tetris, Battleship, CartPole). Its key feature is that each environment provides paired MDP and POMDP variants, which share identical underlying dynamics and state/action spaces, thereby enabling controlled studies using the proposed tools.

### Strengths
The paper is well-written and clearly structured. The tools are defined mathematically, and the experiments (e.g., Fig. 2) effectively illustrate their utility. The core finding that return-only comparisons are confounded because the Bias and Gap can be on similar scales is clearly articulated and well-supported.

### Weaknesses
The contribution of the benchmark itself (POPGym Arcade) is my main source of contention with this work. The justification for this new suite of "toy" environments is thin, especially given the existing landscape of RL benchmarks.

The paper argues that this new benchmark is necessary to utilize the proposed analysis tools. However, the field is already saturated with Atari-style and grid-world environments, a fact acknowledged by the paper's own related work table (Appendix I). Moreover, these existing benchmarks are sufficient; complex memory challenges are not new. The original Arcade Learning Environment (ALE) contains well-established POMDPs like Montezuma's Revenge and Private Eye that require deep, long-term memory. Other benchmarks like DMLab , MemoryGym , and the original POPGym (which this work builds on) were all created to explicitly test memory. It is unclear why an entirely new suite of 10 games was needed to show these effects. The environments created, such as CountRecall or AutoEncode, feel overly simplistic and purpose-built for the metrics. The RL community is actively trying to move beyond 2D, Atari-style tasks toward more complex, 3D, and physics-based challenges. Proposing a new benchmark in this paradigm, even if it is hardware-accelerated, feels like a step backward and of limited long-term utility to the community.

The paper's main defense is the necessity of paired MDP/POMDP twins. While this enables a very clean measurement of the observability gap, it is not the only way to conduct controlled studies. This design constraint leads to the creation of artificial tasks rather than motivating the tools on problems the community already struggles with. A more compelling paper would have demonstrated these tools on existing, difficult benchmarks (e.g., by creating POMDP-variants of existing MDPs, as done in POBAX Tao et al. (2025)).

### Questions
1. The primary critique is the necessity of the POPGym Arcade benchmark. Why were existing, challenging benchmarks (e.g., ALE, DMLab, MemoryGym ) insufficient for demonstrating your analysis tools? What critical aspect do these 10 new games provide that could not be achieved by applying observational masking to existing hardware-accelerated MDPs (e.g., from Gymnax or Jumanji )?

2. The experiments on OOD contamination (Figs 5, 6) are interesting. You note that transformers might mitigate this by discarding OOD inputs once they fall out of the context window. This seems like a critical comparison. Why were transformer-based architectures not included in the main comparisons (Fig 2) alongside RNNs (GRU, MinGRU) and SSMs (LRU, FART)?

3. The related work table (Appendix I) claims POPGym Arcade is unique in having both GPU acceleration and true MDP/POMDP twins. However, the table also lists POBAX (Tao et al., 2025) as having both. Could you clarify the key differentiator from POBAX, which also seems to focus on "benchmarking partial observability"?

### Soundness
2

### Presentation
3

### Contribution
1

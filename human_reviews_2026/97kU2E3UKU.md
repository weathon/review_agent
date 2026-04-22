# LLM-First Search: Self-Guided Exploration of the Solution Space

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 2, 2

## Abstract
Large Language Models (LLMs) have demonstrated remarkable improvements in reasoning and planning through increased test-time compute, often by framing problem-solving as a search process. While methods like Monte Carlo Tree Search (MCTS) have proven effective in some domains, their reliance on fixed exploration hyperparameters limits their adaptability across tasks of varying difficulty, rendering them impractical or expensive in certain settings. In this paper, we propose \textbf{LLM-First Search (LFS)}, a novel \textit{LLM Self-Guided Search} method that removes the need for pre-defined search strategies by empowering the LLM to autonomously control the search process via self-guided exploration. Rather than relying on external heuristics or hardcoded policies, the LLM evaluates whether to pursue the current search path or explore alternative branches based on its internal scoring mechanisms. This enables more flexible and context-sensitive reasoning without requiring manual tuning or task-specific adaptation. We evaluate LFS on Countdown and Sudoku against three classic widely-used search algorithms, Tree-of-Thoughts' Breadth First Search (ToT-BFS), Best First Search (BestFS), and MCTS, each of which have been used to achieve SotA results on a range of challenging reasoning tasks. We found that LFS (1) performs better on more challenging tasks without additional tuning, (2) is more computationally efficient compared to the other methods, especially when powered by a stronger model, (3) scales better with stronger models, due to its LLM-First design, and (4) scales better with increased compute budget. Our code will become publicly available upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes LLM-First Search (LFS), a test-time LLM agentic framework where the LLM explore different decision nodes in a tree-based manner similar to prior works, but prompts LLM itself to give scores to every expandable node in the tree and decide whether to explore or exploit. If the LLM decides to explore and jump from the current reasoning trace to another trace, the LLM will switch to the most promising alternative node retrieved by a priority queue; otherwise, the LLM will continue diving into the current node and update the priority queue. The proposed method outperforms several baselines on two tasks, which are Sudoku and Countdown.

### Strengths
1. The idea is simple, intuitive and clearly conveyed: to remove handcrafted rule for exploration-exploitation and let LLM take charge of which node to expand during tree search. 

2. The related work seems to be rather complete, which discusses different LLM test-time reasoning framework in details and clearly stated the difference between the proposed method and prior works.

3. The paper has a very detailed appendix, which greatly increases the reproducibility of the paper by providing all the prompts and implementation details. The appendix also provides detailed experiment results which clearly show the advantage of LFS.

### Weaknesses
1. As mentioned in the limitation section, the proposed method highly relies on the LLM's base ability as the exploration-exploitation tradeoff is made by the LLM itself. However, if future long-context, thinking state-of-the-art models are strong enough, they may inherently possess the ability to jump between different branches of thoughts without adopting LFS (e.g. many papers [1] mention crucial tokens such as "wait" or "however" in solving complicated math problems, which can be seen as a variant of LFS). 

2. Following 1, the motivation of getting rid of "fixed handcrafted exploration" becomes less convincing as LFS also adopts somewhat handcrafted way of expanding nodes (priority queue and the explicit, explore-exploit judgment), and MCTS-based methods also involves LLM-guided evaluation of scores as mentioned in Alg. 4. The question is: since every method contains more or less degree of "fixed handcrafted exploration" compared to expecting the "wait" or "however" token from LLM itself, for state-of-the-art models, where should we stop and say "this level of exploration-exploitation strategy contains neither too much nor too little handcrafted prior?" 

**Minor Weakness**

1. In line 652, "LLM-First- Search" -> "LLM-First-Search"

2. In line 222, "Three-of-Thoughts" -> "Tree-of-Thoughts"

3. The formatting in page 27 is strange; it only has one line saying "Exploration Decision System Instruction and User Request".


**References**

[1] S. Wang et al. Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning, 2025.

### Questions
I have several questions: 

1. The authors mention that the MCTS tree is noticably wider and over-explores. This is somewhat counterintuitive, as MCTS's rollout always goes on until the leaf / terminal (see Alg. 4 line 6-9), while in LFS the LLM can jump to another branch in the middle of a rollout (see Alg. 1); i.e, MCTS's rollouts are on average deeper. Can the author explain this?  

2. Following 1, the authors also mention that the result shows that MCTS with a value of $C=0.5$ (i.e. fewest exploration) works better than higher $C$. The question is: will MCTS's performance grow further if $C$ is further decreased to, e.g., 0.1 or 0.2? Is 0.5 still "too high an exploration constant" as suggested in line 85?

3. As there are many possible future states, the priority queue could save many unexplored nodes. However, the LLM ususally gives a discrete scale (e.g. 0.05, 0.1, 0.15, ..., 1) instead of a continuous value between 0 and 1. Does LFS often encounter nodes with the same weights in the priority queue? If so, how does LFS deal with this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LLM-First Search (LFS), a novel method for reasoning and planning in Large Language Models (LLMs). The core premise of LFS is to replace the fixed search strategies and pre-defined hyperparameters (such as the exploration constant $C$ in MCTS) with an autonomous control mechanism managed directly by the LLM itself.

The LFS method employs the LLM in two key operations: (1) 'Evaluate', where the LLM assesses the value of all available actions from a given state, and (2) 'Explore', where the LLM dynamically decides whether to continue exploiting the current path or to "jump" to an alternative, high-potential path. To facilitate this, LFS maintains a priority queue of all unelected, promising actions, which serves as a mechanism for backtracking.

The authors evaluate LFS on two benchmarks, Countdown and Sudoku. The results reportedly show that LFS outperforms baseline methods, including ToT-BFS, BestFS, and MCTS, particularly on more difficult tasks, while also demonstrating superior computational efficiency.

### Strengths
1.  Well-Motivated Problem: The paper clearly identifies a significant limitation in existing search-augmented LLMs. It highlights the impracticality and sub-optimality of relying on fixed hyperparameters (like the MCTS exploration constant $C$), which require costly re-tuning for different tasks or models. The finding that MCTS performance can degrade when using a stronger model provides a compelling motivation for a more adaptive approach.

2.  Strong Empirical Performance: Within the confines of the two evaluated benchmarks, LFS demonstrates superior performance. The method achieves a higher WinRate than all baselines, with the performance gap widening on the most challenging tasks (e.G., Countdown Diff=7).

3.  Impressive Scalability and Efficiency: LFS shows strong evidence of scalability. It not only performs well but shows a greater performance improvement when paired with a stronger model (O3-mini) compared to baselines. Furthermore, the method is shown to be more computationally efficient, achieving a better Area Under Profile (AUP) for efficiency and generating smaller, more focused search trees than MCTS.

### Weaknesses
1.  Concerns Regarding Novelty: The proposed method appears to have significant overlap with the Tree-of-Thoughts (ToT) framework. The core mechanism—using an 'Evaluate' prompt for scoring and an 'Explore' prompt for decision-making, coupled with a priority queue—could be interpreted as a sophisticated form of prompt engineering built upon the ToT concept, rather than a fundamentally new search paradigm. The novelty beyond this implementation is not made sufficiently clear.


2.  Insufficient Analysis of Mechanism: The paper provides extensive data on *what* LFS achieves (i.e., higher WinRates) but lacks a deep analysis of *how* it achieves this performance. The claim that "self-guided" exploration works better is a high-level description of the phenomenon, not an explanation. There is no analysis of the LLM's decision-making process during the 'Explore' step. What cues does the LLM use to decide to backtrack? How do these emergent heuristics qualitatively differ from, and improve upon, the MCTS formula? Without this analysis, the paper is reporting numbers, not mechanisms.

3.  Limited Generalization: The validation is confined to two highly structured "toy examples" (Countdown and Sudoku), where states are clearly defined and rules are deterministic. No generalization is guaranteed beyond these benchmarks. The authors themselves acknowledge in the appendix that these tasks "lack some complexities of real-world problems". It is unclear if this "LLM-First" approach would be effective, or even feasible, in more open-ended, complex reasoning tasks where state representation is ambiguous.

### Questions
4.  Clarity and Presentation: The manuscript's structure is not always well-organized, making the central arguments difficult to follow at times. In particular, Figure 1 is hard to understand.

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes LLM-First Search (LFS), a self-guided test-time search method where the language model itself decides when to exploit the current path or explore alternatives, scoring actions and dynamically backtracking via a priority queue and it removes hand-tuned exploration schedules and heuristics common in ToT-BFS, BestFS, and MCTS. Evaluated on Countdown and Sudoku with GPT-4o and o3-mini, LFS achieves competitive or superior WinRate, better efficiency (wins per token), and stronger scaling with harder tasks, larger models, and higher compute budgets.

### Strengths
1. Choosing win rate to accommodate generation stochasticity is sensible. The paper also reports Wilson 95% CIs, efficiency (wins per token), and performance profiles (AUP), which improves statistical transparency.

### Weaknesses
1. Countdown and Sudoku are rigorous but synthetic; standing alone (without math/coding or other reasoning tasks) they have relatively limited action spaces, which narrows external validity. The authors themselves note the restricted scope. However, given that it is a paper that proposes a methodology, the limitation is substantial.   


2. Limited ablations: there is no analysis of LFS internals to understand the effectiveness of the proposed pipeline. Component-wise ablations would help isolate what drives gains.    


3. Baselines seem incomplete: ToT-BFS baseline only appear in GPT-4o but not in o3-mini. MCTS for o3-mini only uses c=0.5 and as authors pointed out this particular choice was seen only optimal for CountDown. It might lead to unintentionally choosing weak baseline due the incompleteness of experiments. Furthermore, authors only evaluated on two models, which might be limited to validate the methodology.

### Questions
1. In Table 1/Figs., why is 6×6 Sudoku so low (e.g., with o3-mini: 0–25% WinRate across methods)? Do you suspect artifacts of the setup, and why not 9×9 to better reflect standard Sudoku difficulty?

### Soundness
1

### Presentation
2

### Contribution
1

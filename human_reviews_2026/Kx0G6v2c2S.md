# DeepSearch: Overcome the Bottleneck of Reinforcement Learning with Verifiable Rewards via Tree-based Search

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Although Reinforcement Learning with Verifiable Rewards (RLVR) has become an essential component for developing advanced reasoning skills in language models, contemporary studies have documented training plateaus after thousands of optimization steps, i.e., notable decreases in performance gains despite increased computational investment. This limitation stems from the sparse exploration patterns inherent in current RLVR practices, where models rely on limited rollouts that often miss critical reasoning paths and fail to provide systematic coverage of the solution space. We present DeepSearch, a framework that integrates Monte Carlo Tree Search (MCTS) directly into RLVR training. In contrast to existing methods that rely on tree search only at inference, DeepSearch embeds structured search into the training loop, enabling systematic exploration and fine-grained credit assignment across reasoning steps. Through training-time exploration, DeepSearch addresses the fundamental bottleneck of insufficient exploration, which leads to diminishing performance gains over prolonged training. Our contributions include: (1) a global frontier selection strategy that prioritizes promising nodes across the search tree, (2) selection with entropy-based guidance that identifies confident paths for supervision, and (3) adaptive replay buffer training with solution caching for efficiency. Experiments on mathematical reasoning benchmarks show that DeepSearch achieves an average accuracy of 62.95% and establishes a new state-of-the-art reasoning model, while using 5.7x fewer GPU hours than extended training approaches. These results highlight the importance of strategic exploration over brute-force scaling and demonstrate the promise of algorithmic innovation for advancing RLVR methodologies. DeepSearch establishes a new direction for scaling reasoning capabilities through systematic search rather than prolonged computation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper integrates MCTS directly into RLVR training to scale breadth, not just depth. By doing so, it broadens exploration and enables finer-grained credit assignment by propagating verifier signals back through the search tree. Practical search and data-filtering heuristics improve trajectory quality, and training employs a tree-aware GRPO in which per-node Q-values from MCTS backups serve as advantages after sequence-level normalization. The approach delivers higher Pass@1 with substantially better compute efficiency on math benchmarks for 1.5B-scale models.

### Strengths
The main strength of the paper is that it targets a very strong baseline. The paper improves an already competitive 1.5B model (Nemotron-Research-Reasoning-Qwen-1.5B v2) and contrasts against a solid RLVR baseline (DAPO), making the incremental gains meaningful at this scale. Results suggest simple RLVR fine-tuning provides limited uplift, while the proposed training-time MCTS adds ~1–2pp Pass@1. Design choices are justified. Most heuristics are motivated, and the ablation is informative and clearly shows which components matter.

### Weaknesses
1- there is not much related work! for targeting better credit assignment there are many paper addressing brining test time compute for more fine grains credit assignment.
2- the biggest gain in table 4 shows for "Frontier Selection" and rest of the justifications, such as better credit assignment, remain very substantially small.
3-  2% improvement is not a significant improvement. It is unclear what the main conclusion of the paper is.Additionally, how can we translate this paper to a novel RLVR algorithm for general application? The paper claims a paradigm shift, but it is unclear why this is a new paradigm.

### Questions
The paper has many moving parts. What is the main conclusion of the paper? Apart from the methodology itself, what is the exact shifted paradigm? If I understand correctly, it should be that brute-force exploration is not enough, and we need a tree-based exploration? Is there an intuition on why tree-search based explorations are better?

### Soundness
3

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
This paper addresses performance plateaus in Reinforcement Learning with Verifiable Rewards (RLVR) by integrating Monte Carlo Tree Search directly into the training process rather than limiting it to inference. The authors propose DeepSearch, which features: (1) a global frontier selection strategy that prioritizes promising nodes across the entire search tree rather than following traditional root-to-leaf UCT traversals, (2) entropy-based guidance to identify confident incorrect reasoning paths for supervision, and (3) an adaptive replay buffer with solution caching to focus computation on challenging problems. Evaluated on mathematical reasoning benchmarks (AIME 2024/2025, AMC 2023, MATH500, Minerva, Olympiad), DeepSearch achieves 62.95% average accuracy on a 1.5B model, representing a 1.25 percentage point improvement over the previous best while using 72× fewer GPU hours than extended training approaches. The results suggest that systematic exploration during training is more effective than simply scaling training depth.

### Strengths
Originality:
- The core insight of embedding structured search into the training loop rather than limiting it to inference is compelling and represents a meaningful shift in perspective
- The global frontier selection strategy is an interesting departure from traditional UCT traversals, addressing computational efficiency while maintaining search quality
- The combination of entropy-based negative example selection with adaptive replay buffers creates a thoughtful approach to computational resource allocation

Quality:
- The experimental evaluation is thorough, with strong baselines including very recent work (Nemotron v2, DeepScaleR)
- The efficiency analysis is particularly valuable—demonstrating 5.7× fewer GPU hours while achieving better performance makes a strong case for the approach
- Table 4's evolutionary ablation study effectively illustrates how each component contributes to the final performance
The search strategy ablation (Table 3) provides useful insights into design choices

Clarity:
- Figure 1 provides an excellent high-level overview of the framework, making the approach accessible
- The motivation is well-articulated—the observation about training plateaus in extended RL training resonates
- The method description flows logically from MCTS components through training strategy
- Algorithm 1 in the appendix offers comprehensive implementation details

Significance:
- Addresses a genuine bottleneck in RLVR that has been observed across multiple recent works
- The efficiency gains are substantial and practically important given the computational costs of modern reasoning model training
- The framework appears general enough to potentially benefit other RLVR applications beyond mathematics

### Weaknesses
Experimental Analysis:
- While the average improvement is clear, it would be illuminating to characterize which types of problems benefit most from the approach. Are there problem characteristics that predict when MCTS exploration is particularly valuable?
- The comparison with extended training is compelling, but I wonder how performance would compare if the extended training baseline also incorporated some of DeepSearch's innovations (like adaptive filtering) without full MCTS
- Table 2 shows extended training plateauing, but it would be interesting to see if combining extended training with DeepSearch yields further gains, or if there are diminishing returns

Method Design Choices:
- The entropy-based selection for negative examples (Eq. 2) is intuitive, but I wonder if there are alternative criteria that might be equally or more effective. Have you explored selecting based on value function disagreement or other uncertainty measures?
- The β parameter for DirectRollouts when cached solutions exist seems important but isn't thoroughly analyzed. How sensitive are results to this choice?
- The filtering threshold δ is set to 25%—could this be adaptive based on training progress or problem difficulty distribution?

### Questions
- Computational breakdown: Could you provide a breakdown of where computational time is spent (tree expansion, policy inference, Q-value updates, etc.)? This would help identify further optimization opportunities.
- Parallelization: How does MCTS tree construction parallelize compared to standard rollout generation? Are there architectural choices that could improve throughput?
- Comparison fairness: The extended training baselines use standard DAPO. Would extended training with better exploration strategies (but without full MCTS) close some of the gap? This would help isolate MCTS's specific contribution.
- Learned components: Have you considered learning any of the MCTS components (e.g., the frontier priority function or expansion width) rather than hand-designing them?
- Multi-task learning: Could a single DeepSearch model be trained across multiple reasoning domains simultaneously, with shared MCTS infrastructure?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper examines an assortment of correlated methods to better scale RLVR for use in reasoning problems.  The paper forwards three interventions, as shown in the overview Figure 1: (1) DeepSearch-MCTS, a means of deciding how to appropriately expand nodes in MCTS; (2) Adaptive training to pair with (1) by applying a tree version of GRPO; and (3) a replay buffer to enhance the study of difficult and successful trajectories.  Experiments by the authors show that the interventions taken together are synergistic and lead to performance improvements beyond performance, including a half-magnitude savings in rollout inference costs.

### Strengths
* The three interventions are well-motivated and illustrated conceptually well in Fig 1.
* The authors explain the rationale of some parts of their method (L209-215), to bring together the motivation and explain design choices.

### Weaknesses
Overall, while the methods disclosed seem convincingly effective, the vanilla UCT baseline also does quite well, and the complexities of the methods proposed give many dimensions of freedom for possible overfitting (as confidence intervals overlap considerably).  While the authors do test with SOTA baselines, I'm not convinced of the introduced mechanism's impact on the results for these reasons.
  
* The mechanisms used seem ad hoc and not particularly well-justified (e.g., S2.1 could max depth $n$ be defined conditioned on or calculated from some other means?)
* Why do you select the most confident negative example?  Can you justify it (L141-143 seems insufficient)?
* Using S2.2 to back-propagate up the best negative and positive cases seem ok but divergent.  Why would you want to handle both in the same way?
* It would be instructive for you to (briefly) explain UCT _before_ L181.

### Questions
* Is  _temporal decay_ (L152) better termed as _depth decay_ (or just make an association to a _discount factor_ in RL rewards)?
* Your Eq. (7) is a bit hard to follow with three cases; can you just help narrate this prose text instead?
* How much does $N(s)$ influence the computation and search?  It is implied that it is helpful but it's not clear whether it plays much of a role (repeated visits).
* Keeping track of a frontier could be expensive.  How does this figure into the (asymtotic and real wall-clock) cost?  What manner do you operationalise this for (assumed GPU bound) processing?
* L241: are there borderline cases where no good success traces emerge, and as such the adaptive training fails?  This step seems to assume there are always successes that can be judged.
* L269: How important (e.g., what %age was thrown out) by the condition "remove data containing garbled text or infinite repetitions"?  Given evidence that such filtering does not invalidate your narrative.
* L274: Can you show us how the ratio between cached problems and unsolved problems evolves during your adaptive training (and without, as an ablation)?
* L289: How many of q-values are (severely) clipped?  What does the distribution of q-values look like, post-clipping?  This seems needed to really support that such an intervention is needed.

### Soundness
2

### Presentation
2

### Contribution
2

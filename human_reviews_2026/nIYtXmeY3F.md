# Discovery of Adversarial Endgame Chess Positions

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Chess engines have become an essential component of today's lucrative online chess market, and many players treat their recommendations as the ground truth.
However, these engines are not perfect and can make mistakes when faced with certain endgame positions. The occurrence of such positions within an engine's search could lead to errors cascading to the root. Despite this, the systematic generation and analysis of positions that expose such weaknesses remains an underexplored area of research.
To fill this gap, we develop AdvChess, a novel framework to automatically generate adversarial chess positions. These are positions where state-of-the-art engines deviate from theoretically optimal play.
Our approach focuses on identifying fair and legal positions where engine failures result in significant outcome changes, particularly in the context of endgame play, where ground-truth labels can be extracted from specialized endgame tablebases.
We design state and action encodings as well as a reward function for the foundation of the generative modeling problem. 
We find that adversarial positions generated for Stockfish are least transferable across different computational settings
and that transferability does not correlate directly with engine strength.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ADVCHESS, a novel framework to automatically generate adversarial chess positions. The authors focus on identifying fair and legal positions where engine failures result in significant outcome changes. In addition, the authors design a reward function AS-LE to simultaneously encode three desired aspects: (1) adversarial score, (2) position legality, and (3) material balance of the position, for guiding the generative process toward the final objective.

Main contributions of the authors include:
1. Benchmarking several advanced sampling methods (MCMC, PPO, and GFlowNets) against
the uniform baseline, and analyze their performance and limitations in adversarial position discovery.
2. Designing and implementing a novel search algorithm, AS-LE, as the core sampling component within the ADVCHESS framework, which combines local and uniform search and significantly outperforms all tested baselines.

### Strengths
In terms of significance, the paper addresses the systematic generation and analysis of adversarial positions: specialized board configurations that reveal weaknesses in engine decision-making, leading to suboptimal play and altering theoretical game outcomes from wins to draws or losses. The discussion on sampling methods is mostly well done. 
In terms of originality, their main contribution is designing a novel search algorithm, Adversarial Search via Local Exploration (AS-LE), which leverages a combination of local and uniform search to effectively explore the vast space of chess positions and integrates it with the ADVCHESS framework to find rare, failure-inducing endgame positions by navigating the vast and sparse search space. The authors present and perform a clear empirical study to evaluate their framework by benchmarking it against three diverse chess engines (Stockfish, Winter, and Floyd) with three different metrics. They also compare their proposed sampling algorithm, AS-LE, against four other methods to establish its superior efficiency.
Lastly, the quality of this work is analyzing the transferability of the discovered positions across various engine configurations, as shown in Figure 6.

### Weaknesses
Overall, this paper could be a significant algorithmic contribution, with the caveat for some clarifications on the theory and experiments. Given these clarifications in an author response, I would be willing to increase the score. 

For the theory–sampling algorithm adaptation section 4.1, there are a few steps that need clarification and further clarification on novelty. Did you use MCMC, PPO, and GFlownet as base algorithms and integrate those with the new reward function? So your reward seems just a component and an integration to these algorithms, rather than designing a new sampling algorithm? In other words, it is not clear if AS-LE is a new sampling algorithm, or it is just an integration of already existing sampling algorithms? How did you manage to overcome challenge 2 (Complexity of Strategies (C2))? Can your reward function alone circumvent all 4 challenges?

For the experiments, on page 7, you mentioned that “Further implementation details and algorithm
hyperparameters are given in the Appendix B.”, but no appendix is provided. More about implementation details would be illuminating. There is also another missing appendix. In Figure 4, “(Other search-node budgets are in Appendix C)”. It would be absolutely good to see more details regarding these experiments.

### Questions
For the experiments, the following should be addressed:
1. It would have been better to also show the performance graphs with and without the improvements to the reward function.
2. The central contribution is designing a reward function. It would be beneficial to see, empirically, how the performance differs with and without the reward.
3. Implementation details, hyperparameters, and parameters are missing. I would like to know how changing the parameters affects the performance?

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
The paper aims to discover the adversarial chess positions that will let chess engines make mistakes, specifically in the endgame. To discover these positions, the paper developed the ADVCHESS framework, which uses the sample methods to generate adversarial chess positions. The authors first compare several sample methods, including uniform sampling, MCMC, GFlowNets, and PPO, against a custom reward function. Then, they observe that adversarial chess positions tend to cluster densely in the state space, leading to the main method, Adversarial Search via Local Exploration (AS-LE). AS-LE operates in two phases: a Discovery phase that performs via uniform random search until an adversarial position is found, and an Exploration phase that performs via DFS. The results show that AS-LE outperforms other sample methods, collecting the maximum number of unique adversarial chess positions.

### Strengths
1. The paper is well-structured and well-written. The definition of adversarial positions and the reward function are clearly addressed. Figures (e.g., Figures 2-3) and their explanations improve readability, allowing the readers to understand easily.

2. The research topic of finding adversarial chess positions is underexplored. The paper proposes a novel framework, ADVCHESS, which includes all the sample methods mentioned above, to generate adversarial chess positions. Especially the outperform sample method AS-LE, which is easy and powerful.

3. The success of the ADVCHESS framework pioneers the field of systematically generating adversarial chess positions, enabling developers to identify and mitigate engine weaknesses more effectively.

### Weaknesses
Overall, the results are interesting, but the motivation of this paper is not clear. I think the authors should carefully address each of the following weaknesses.

1. For example, this paper focuses on using endgame positions to identify adversarial positions. However, it is not clear why it only investigates endgame positions, which are relatively easy positions. Furthermore, it is also not clear how these discovered endgame positions help.

2. Although chess has an extensive endgame database, the proposed method should not be limited to endgame scenarios and the chess environment. It would be convincing to incorporate other games (e.g., Go, Hex, Othello) to demonstrate the generalizability.

3. Studies on adversarial in board games have been widely investigated, and some of them can also generate the adversarial position or policy [1-2]. Their approach can also automatically discover adversarial positions without the limitation to endgame positions. The paper should carefully review these backgrounds, explicitly describe the differences between the proposed methods and previous methods, and even compare the approaches in the experiments.

[1] Lan, Li-Cheng, et al. "Are alphazero-like agents robust to adversarial perturbations?." Advances in Neural Information Processing Systems 35 (2022): 11229-11240.

[2] Wang, Tony Tong, et al. "Adversarial policies beat superhuman Go AIs." International Conference on Machine Learning. PMLR, 2023.

4. The paper lacks an analysis of the number of different clusters identified by each sampling method. The experiment results indicate that the Uniform Sampling method is underperforming, while AS-LE is outperforming. This suggests that the improvement of AS-LE is limited to similar positions in the same cluster, rather than increasing the number of clusters. A comparison of cluster diversity across sampling methods would be more convincing.

### Questions
1. Please address each concern raised in the weaknesses.

2. What's the motivation of this work? Why is this work mainly focusing on endgames? Can it be applied to other games without the endgame database? How can these discovered endgame positions benefit future research?

3. The part of "Intra-Engine Transferability" and results show that transferability decreases as the difference between search-node budgets grows. Intuitively, the adversarial chess positions that fool a stronger engine (high simulation) should also mislead a weaker one (low simulation), implying high transfer from weak to strong. However, the results show the opposite trend. Providing a deeper analysis would be helpful.

4.  Section 6.2 refers to Figure 7, but no Figure 7 appears in the paper. Does it mean Figure 5, or is a figure missing?

5. Figure 6 lacks of caption to explain how to read the experiment results, which makes readers hard to understand.

### Soundness
2

### Presentation
4

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
This paper introduces an algorithm for constructing endgame Chess positions for which engines deviate from optimal play. While previous studies rely on uniform sampling methods for position generation, AdvChess generates positions systematically through heuristics, endgame tablebases, and engine outputs. The authors compare against several baselines and determine that their method discovers many more adversarial samples.

### Strengths
Systematic methods for finding failure cases could be important for driving further engine improvements. Given the authors' claim that prior work generates these positions uniformly, it seems important to consider new methods.

The presentation of the paper is strong, and it is generally well-written. Empirical and algorithmic details are explained precisely.  The provided code and hyperparameter settings are much appreciated and contribute positively to the quality of the work.

### Weaknesses
**Baselines.** The baselines provided in this paper seem somewhat arbitrary rather than from prior work, and their choices are under-justified. Why should I expect the particular instance of MCMC to be effective at generating such positions? Was PPO tuned effectively for this particular task? Choices for those two algorithms could make a significant difference in terms of the number of adversarial positions generated.

**Significance of the contribution** The paper's main contribution is AdvChess, which performs well compared to the tested baselines, but I'm unconvinced about AdvChess' relevance outside of that and the transferability hypotheses. Could AS-LE be extended to do the same thing in other perfect information games? In general, more discussion about the limitations of the work would benefit the paper significantly.

**Relevance to the ICLR community** It is not clear to me that this is an appropriate venue for this kind of work. The choice of "primary area" for the submission is perhaps evidence of this, as it seems like a stretch to classify this as a contribution to *probabilistic methods*.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces ADVCHESS, a novel framework for the systematic discovery of adversarial endgame positions in chess—board configurations where state-of-the-art engines (e.g., Stockfish) deviate from theoretically optimal play as verified by endgame tablebases (EGTBs).

### Strengths
**Novel and Impactful Problem Formulation**:
The paper addresses a timely and underexplored problem: the systematic discovery of adversarial endgame positions that expose failures in state-of-the-art chess engines. Despite the widespread trust in engines like Stockfish, their vulnerabilities in theoretically solvable endgames have not been rigorously studied through automated generation—this work fills that gap.

### Weaknesses
**Lack of Theoretical Analysis**: The paper observes that adversarial positions are “densely clustered” and leverages this empirically to design AS-LE, but it does not provide a theoretical explanation for why this clustering occurs or how it relates to engine internals (e.g., evaluation heuristics or search pruning).

**Limited Generalizability of AS-LE**: The proposed AS-LE algorithm relies on local perturbations (e.g., moving or transforming a single piece), which is effective in sparse endgame settings but may not scale to midgame or more complex positions with higher-dimensional state spaces.

**Insufficient Connection to Prior Work**: While the paper cites related studies, it does not deeply analyze whether the discovered adversarial positions represent new types of engine failures or merely reproduce known weaknesses.

**No Ablation on Reward Components**: The reward function combines adversarial score, legality, and material balance, but the paper does not include ablation studies to assess the contribution of each component to sampling efficiency or diversity.

### Questions
**On the underlying mechanism of clustering**:  
   The paper notes that adversarial endgame positions are “densely clustered” in the state space and leverages this observation to design the AS-LE algorithm. However, the **root cause of this clustering remains unclear**. Have the authors investigated whether this phenomenon is linked to specific weaknesses in engine evaluation functions (e.g., misjudgment of certain tactical motifs) or search-pruning strategies (e.g., premature pruning of critical branches in alpha-beta search)? Could the authors provide deeper attribution analysis or visual evidence to support this insight?

**On scalability and generalizability of the method**:  
   AS-LE relies on local perturbations—such as moving or transforming a single piece—which works well in 5–6 piece endgames where ground-truth labels are available via endgame tablebases. Have the authors considered—or do they plan to extend—the ADVCHESS framework to **middlegame scenarios** (e.g., positions with 8–10 pieces)? In settings without perfect ground truth, how would “adversariality” be defined or approximated? Would the local exploration strategy of AS-LE still be effective in such higher-dimensional and less-structured state spaces?

### Soundness
2

### Presentation
3

### Contribution
1

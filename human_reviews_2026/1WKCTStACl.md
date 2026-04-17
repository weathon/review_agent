# AlphaZeroES: Direct Score Maximization Can Outperform Planning Loss Minimization in Single-Agent Settings

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Planning at execution time has been shown to dramatically improve performance for AI agents.
A well-known family of approaches to planning at execution time in single-agent settings and two-player zero-sum games are AlphaZero and its variants, which use Monte Carlo tree search together with a neural network that guides the search by predicting state values and action probabilities.
AlphaZero trains these networks by minimizing a planning loss that makes the value prediction match the episode return, and the policy prediction at the root of the search tree match the output of the full tree expansion.
AlphaZero has been applied to various single-agent environments that require careful planning, with great success.
In this paper, we explore an intriguing question:
in single-agent settings, can we outperform AlphaZero by directly maximizing the episode score instead of minimizing this planning loss, while leaving the MCTS algorithm and neural architecture unchanged?
To directly maximize the episode score, we use evolution strategies, a family of algorithms for zeroth-order blackbox optimization.
We compare both approaches across multiple single-agent environments.
Our experiments suggest that directly maximizing the episode score tends to outperform minimizing the planning loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AlphaZeroES, a modification to the AlphaZero algorithm for single-agent settings. The core idea is to replace AlphaZero's standard planning loss—which minimizes the difference between network predictions and MCTS search results (for policy) and episode returns (for value) —with a new objective that *directly* maximizes the total episode score. Because the MCTS component is non-differentiable, the authors employ Evolution Strategies (ES) as a zeroth-order, black-box optimizer to train the neural network parameters. The method is evaluated on five single-agent environments (Navigation, Sokoban, TSP, VKCP, and MDP), where the authors claim that AlphaZeroES "dramatically outperforms" the standard AlphaZero baseline while using the same MCTS algorithm and network architecture.

### Strengths
The paper's primary strength is the question it poses: challenging the optimality of the standard AlphaZero planning loss. This is a fundamental and worthwhile question to investigate.

The paper's observation that maximizing the score via ES does not correlate with minimizing the standard planning losses (and can even be anti-correlated) is an interesting finding.

The ablation study in Appendix D, which attempts to separate the contributions of the policy and value network optimization, is a good addition and provides some insight, showing the source of improvement is environment-dependent.

### Weaknesses
My main concerns are as follows:

1.  **Lack of Intuition and Analysis:** The paper provides no satisfying explanation for *why* its method works. In fact, the results in Figures 2, 3, 5, etc., show that the value and policy losses for AlphaZeroES often increase or stagnate while the score improves. The paper even states "a definitive explanation... is beyond the scope of this paper". This is not acceptable. If the network's value/policy heads are producing "worse" predictions (according to the standard loss), how is the MCTS using these heuristics to produce a *better* overall policy? What is being learned? Without this analysis, the paper is just a collection of puzzling empirical results.

2.  **Questionable Sample Efficiency:** Evolution Strategies are zeroth-order methods and are notoriously sample-inefficient, especially in high-dimensional parameter spaces (like neural networks). Yet, this paper claims comparable or better performance within the same training time and number of episodes as gradient-based AlphaZero. This is an extraordinary claim. It suggests that ES is *more* sample-efficient than backpropagation for this complex planning problem, which contradicts a large body of literature. This result is highly questionable and may be an artifact of poorly tuned baselines or very simple environments.

3.  **Vague Methodology & Reproducibility:** As mentioned in the Presentation section, the paper lacks a clear, high-level pseudocode for the AlphaZeroES training loop. Section 4.3 just describes ES, not its integration. This, combined with the lack of a complete, runnable code repository, makes the results difficult to trust or replicate.

4.  **Poor Scalability and Simple Environments:** The experiments are conducted on "toy" problems. A 10x10 grid or a 20-city TSP is not a convincing demonstration of a method intended to improve *AlphaZero*. AlphaZero's fame comes from its ability to master extremely complex domains like Go or chess. The scalability experiments in Appendix C only test up to 36 nodes and 128 hidden dimensions. This is insufficient. It is highly likely that a black-box ES approach will fail to scale to the millions of parameters and vast search spaces where gradient-based AlphaZero excels.

### Questions
1.  Could the authors provide a more concrete analysis of the learning dynamics? If the value loss is high (e.g., Fig 2), does this mean the MCTS effectively learns to ignore the value head's output and relies purely on rollouts? What do the learned policy/value predictions actually look like? How can the search be effective if its guiding heuristics are, by the paper's own metrics, not improving?

2.  Can you please comment on the surprising sample efficiency? Why would a zeroth-order method (ES) be more efficient than backpropagation here? Were you able to run the baseline AlphaZero with a fully optimized set of hyperparameters? The results are very counterintuitive.

3.  To properly test the claims of scalability, would it be possible to test this on a much larger, more standard benchmark? For example, a larger combinatorial problem (e.g., TSP with $n=100$) or a different domain entirely (like a small board game)? The current environments are too simple to support the paper's strong claims.

4.  Section 5 states that "AlphaZero and AlphaZeroES took about the same amount of time per iteration." This is a very surprising claim. A standard ES update (like OpenAI-ES) requires $N$ full episode rollouts (one for each worker/perturbation) just to compute a *single* pseudogradient $g$. In contrast, a standard AlphaZero update (or batch update) can learn from the data of a single episode (or batch of episodes) via backpropagation. Could you please clarify what "iteration" refers to in this context? Does it mean one full parameter update, or one generated episode? This claim is central to the method's viability but seems to misrepresent the computational cost of ES.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces AlphaZeroES, which replaces AlphaZero's planning loss minimization with direct episode score maximization using evolution strategies (ES). The authors test this on single-agent environments including Navigation, Sokoban, TSP, VKCP, and MDP, and report that AlphaZeroES consistently outperforms standard AlphaZero while maintaining the same MCTS algorithm and neural architecture.

### Strengths
1. The paper poses a focused, well-motivated question about whether direct score optimization can outperform indirect planning loss minimization.
2. Comprehensive demonstration of related work is provided.
3. Limitation is honestly discussed.
4. Statistical tests are provided, including Wilcoxon, signed-rank, and paired t-tests, which show statistical significance.

### Weaknesses
1. The organization and presentation of the paper can be improved. It would be better put mathematical expressions separately for clear elaboration instead of using in-line mode. For Section 5 (Experiments), the authors spend a great deal of space to introduce the environments instead of discussing the quantitative results demonstrated in the figures. Moreover, for each environment, all contents are put in huge bulk of single paragraph, making it difficult to follow.
2. The paper lacks theoretical justification. In Section 6 (Discussion), the claim about "simple optimal policy but complex value function" lacks rigor and doesn't explain why this would systematically favor ES.
3. The paper claims to test "across a wide range of learning rates for fair comparison" between AlphaZero and AlphaZeroES. However, this doesn't ensure a fair comparison. Different learning rates are optimal for different objectives, making it impossible to isolate whether improvements come from the objective change or better hyperparameter selection.
4. The loss report is inconsistent. Figures show AlphaZeroES doesn't minimize value/policy losses, yet performance improves. This disconnect isn't adequately explained.
5. Sensitivity analysis is not provided regarding perturbation scale, which is set to be 0.1. The ES-specific hyperparameter selection is not explained.

### Questions
1. Did the authors perform a grid search or other systematic hyperparameter optimization for both methods? If so, what was the protocol?
2. How did the authors ensure the learning rate ranges tested were appropriate for each method's objective? Did the authors verify convergence for both methods at their optimal learning rates?
3. How did the authors select the perturbation scale of 0.1 for ES? What happens with different scales, such as 0.01, 0.05, 0.2, or adaptive schedules?
4. Regarding loss report, if AlphaZeroES achieves high scores without minimizing value/policy losses, what exactly has it learned? Could the authors analyze the learned representations? Could the authors plot the correlation between planning loss and episode score throughout training for both methods?
5. For theoretical justification of "self-consistency is not necessarily aligned as an objective with performing better", could the author explain why would policy-value inconsistency not hurt MCTS performance, given that MCTS explicitly uses both value estimates and visit counts for action selection? Could the authors provide a formal characterization of when "simple policy but complex value function" occurs? Under what conditions would this favor ES over gradient-based methods?

### Soundness
2

### Presentation
2

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
The authors demonstrate that one can use evolution strategies with an AlphaZero setup to directly optimize maximizing the reward. This appears to work well--or better than the base version in the tested games.

### Strengths
* It is always interesting to see ES
* The authors test their method on a number of games
* The authors provide and extensive survey of work in the area in the appendix
* The authors highlighted a few contexts where the ES strategy also did a good job of optimizing the auxiliary tasks that AZ targets.

### Weaknesses
* Are there any other baselines or methods (or families of methods AC) you could provide? Here, you only show AZ.
* Overall, (1) it is unclear what the score lines mean, what is a good score / success, (2) relatedly, if AZ is not learning at all or having any success.

### Questions
The main question I still have is a want for more context: What type of baselines could you provide, what kinds of other RL agents are typically used for these tasks and how would they compare? Were the AZ agents fairly tested?

* Are the results good? You outperform AZ but it is not clear from the paper would a good performance is. Do other models do much better on these tasks? If AZ models do very poorly and perhaps are under-trained, then doing better is not necessarily super meaningful. Adding context of what success looks like for each task would help.
* Could you try other ways of organizing the plots? I don't think you need to show the loss/value for everything. Make the primary score plots bigger and more readable?
* How much compute is done per model/run ("4 hours of training time per trial") vs what is needed? Did the model training converge? 
* What are the standard approaches for these tasks? Is this within the scope under which AZ works? I do see the appendix figures where ES still always did better even under increased compute.

# Minor notes
* L289 - Seem to "increase"

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper focuses on enhancing AlphaZero algorithm with evolution-based optimization algorithms, which allows to directly optimize cumulative reward function over an episode. The core idea is to train actor model with ES algorithm rather then gradient descent.

### Strengths
- One of the huge advantages of this method is it ability to scale to a large number of workers, which allows to train agent in parallel setting.

### Weaknesses
- No direct comparison to AlphaZero and other methods. Experiments only show an effect of different hyperparameters on AlphaZeroES performance.
- The description of resulting algorithm is hard to understand from pure text description. There is no outline (i.e. step-by-step description or pseudocode) provided in the main part of the paper.

### Questions
- Is there any direct comparison with AlphaZero and other RL methods in terms of performance?
- The main motivation of applying ES is direct black-box optimization of cumulative reward. However, RL algorithms such as REINFORCE can do that to. Even more, the agent learning algorithm of original AlphaZero can do that too via n-step returns or lambda-returns. Is there any other motivation for applying ES specifically?

### Soundness
1

### Presentation
2

### Contribution
2

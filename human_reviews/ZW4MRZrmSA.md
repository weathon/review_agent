# A Policy-Gradient Approach to Solving Imperfect-Information Games with Best-Iterate Convergence

- Avg Score: 6.25
- Decision: Accept (Poster)
- Scores: 8, 6, 5, 6

## Abstract
Policy gradient methods have become a staple of any single-agent reinforcement learning toolbox, due to their combination of desirable properties: iterate convergence, efficient use of stochastic trajectory feedback, and theoretically-sound avoidance of importance sampling corrections. In multi-agent imperfect-information settings (extensive-form games), however, it is still unknown whether the same desiderata can be guaranteed while retaining theoretical guarantees. Instead, sound methods for extensive-form games rely on approximating \emph{counterfactual} values (as opposed to Q values), which are incompatible with policy gradient methodologies. In this paper, we investigate whether policy gradient can be safely used in two-player zero-sum imperfect-information extensive-form games (EFGs). We establish positive results, showing for the first time that a policy gradient method leads to provable best-iterate convergence to a regularized Nash equilibrium in self-play.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a policy gradient method, QFR, for two-player (imperfect-information) zero-sum extensive-form games. The algorithm can approximate best-iterate convergence to a regularized Nash equilibrium. Policy gradient updates are preferred over CFR’s counterfactual updates because they can be computed from rollouts and do not require importance sampling. The paper proves its findings and runs some benchmarks on toy environments.

### Strengths
The paper proposes a compelling algorithm and goes to great lengths to prove its convergence properties. Not all competing algorithms in the class have proven convergence properties. Some strengths over CFR:

* Best iterate convergence is more convenient than time-average convergence
* No importance sampling

The paper seems technically sound – I haven’t found a flaw yet – however there are parts that I did not understand (see weakness/questions). I am very open minded to increase the soundness score and overall rating of this paper after discussions with authors and other reviewers.

### Weaknesses
Some terms seem to be undefined which compounded confusion on an already complex paper. Please fix the undefined symbols (see questions for concrete examples).

There are many step-jumps in the main text that are not immediately clear to me. Some of these are explained in the appendices, but they are not always well sign-posted. Improved sign-posting may make reading through the paper easier (see questions for concrete examples).

Focuses on two-player zero-sum (a simple game-theoretic setting).

### Questions
I am really struggling to understand how Equation 4.3 is derived. Is there somewhere in the appendix that describes it? This confusion snowballed throughout the rest of Section 4.2.

“We can see QFR outperforms BOMD in all games”. Does it in Liar’s Dice?

Minor:

* Line 208: I find the definition of d(h|s) hard to understand. Can another sentence be added, or some more steps shown? Footnote 2 did not seem to enlighten it for me.
* Line 240: The notation of the policies is confusing. And what is pi bar? Has it been defined?
* Line 245: “ad ditional”
* Line 245: What does subscript “bi” mean on the regularizer? Bilinear? (edit: later on line 269, bidilated)
* Line 256: mu should be a psi?
* Line 296: Dodgy line formatting of “else”
* Line 303: Algo line 10. Where did this come from? Add a couple more words explaining the policy gradient update.
* Line 323: “...learning *rate* of inforset…”
* Line 411: +“the”, “and” -> “or”
* Line 417: I am really struggling to see where this inequality comes from.
* Line 469: “... with under …”
* Line 716: “a NE” -> “an NE”
* Line 796: What is alpha_s’? Did it get defined?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The author(s) propose a Q-function based RL approach to solve zero-sum, extensive-form games under imperfect information. The proposed approach only requires sampling randomly generated trajectories and has iterate convergence guarantees.

### Strengths
- The contribution of the paper is strong, where the proposed approach only requires sampling of randomly generated trajectories (as opposed to using importance sampling) to estimate value function and compute policy. The approach is proved to have best-iterate convergence guarantees to the Nash equilibria of the regularized game under both full and imperfect information. 
- Clear introduction of related works and main obstacles in the field as well as contribution statement. I really enjoyed reading this part.

### Weaknesses
1. Line 280: The introduction of the algorithm can be more detailed. Right now, it is only a few lines and assumes the readers are already familiar with the references. For example, the author(s) ca n do a better job walking the readers through their update of the regularizer at Line 8.
2. The exploitability metric used in the experiment section is undefined. The experiment section can be better presented: what makes the proposed approach outperform a certain baseline in a certain setting? Why in certain setting, the proposed approach is inferior? Could the author(s) comment on how many samples are required by their approach vs. the baselines?
3. Completeness of the main text: In my opinion, discussion and conclusion should definitely be in the main text. Right now, the main text is not self-contained. That being said, while I appreciate the completeness of the first two sections, I also feel that they could probably be tightened to make room for other important content without losing much information. If the author(s) finds it impossible to fit the important content all in the main text, this paper might fit better for a journal. 
4. Other minor points: 
- Introduction (line 35): The statements (I) and (III) overlap with each other; they might be merged into one
- Line 91 “Techniques based on rollouts of …” overlaps the previous paragraph. Author(s) could move this point to above
- line 245: “additional”
- Algorithm 1 line 6: indentation of the if-else statement is broken
- Line 282: “Lastly, all infosets along the trajectory…” - I think the author(s) meant to point to Line 12 rather than Line 11

### Questions
1. Eq (2.3): should the direction of the inequality be flipped?
2. Have the author(s) tried to anneal the regularization? Or the results given in the experiments are all subject to regularized objective functions?
3. The proposed approach samples one trajectory using the current policy to compute estimates of value function at each iteration, I feel like this approach might have a high variance for larger games. Did author(s) have any observation and/or explanation regarding this?
4. The method seems to be a value function-based RL approach, as it estimates value functions and computes policies by maximizing the value functions. Why call it “policy gradient” method? Am I missing something?
5. The approach integrates existing approaches, e.g., convex regularizers and mirror descent. Could the author (s) clarify in the text clearly, what are the algorithmic modifications of the proposed approach beyond existing methods that enable the main features? My sense is that weighting of the regularizers considering both players’ paths along the tree is the main stretch, but I would like to know more specifically.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper proposes a policy gradient method, Q-Function based Regret Minimization (QFR) for solving two-player zero-sum extensive-form games, which uses rollouts to estimate values without importance sampling and achieves best-iterate convergence. Experiments on poker and Hex support the proposed method.

### Strengths
The idea of using Q-value in regret minimization is reasonable.

The writing is clear and and the paper is easy to follow.

### Weaknesses
The proposed algorithm combines optimistic mirror descent updates and estimating Q-values from rollouts, both of which seem to be well known techniques. The technical novelty might be limited. 

Motivations seem to be disconnected with later sections, such as experiments.

### Questions
The three desirable properties were mentioned at the beginning of the paper and seem to be important. But they seem to be not mentioned in later sections. For example, how can be observe from experiments the advantage of not using importance sampling in QFR? Does it make the proposed method have smaller variance than other methods? And also, how is the using rollout property helpful? Does it make the proposed method more sample efficient than other methods in the experiments?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors propose an algorithm named QFR, which performs policy gradient updates to solve for regularized Nash equilibrium of two-player zero-sum EFGs. Convergence analysis is presented with sublinear rates. Experiment results also show the compatibility of Q-value based policy gradient updates to two-player zero-sum EFGs.

### Strengths
The analysis is sound with sublinear iterate average convergence. Proposed idea is novel and easy to follow.

### Weaknesses
Analysis section:

- Although it makes sense to enforce strong convexity to the bilinear objective via regularization for easier analysis and stronger convergence guarantee, it also brings a bias to the equilibrium. As the authors are introducing a new regularization as their part of the novelty, it is also expected that the authors show how large this bias is.

Experiment section:

- As the authors mentioned in intro, the motivation behind proposing and proving the convergence of policy gradient based algorithm is to fit better under today's DRL trend. However, the experiment section does not show the advantage of QFR in this regard. As such, I suggest the authors to perform experiment on larger size and compare the solution performance (sample efficiency, computational efficiency etc) across all the different baseline algorithms.

- In figure 1, the convergence seems not reached for Liar's Dice and Leduc Poker. What is the reason behind truncation at the iteration 1000?

### Questions
Please see weakness section.

### Soundness
3

### Presentation
3

### Contribution
3

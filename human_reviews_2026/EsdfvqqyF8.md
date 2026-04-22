# Learning to Incentivize on the Fly: Leader-Follower Games with Policy Recommendation

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
In dynamic and strategic interactions, ensuring incentive compatibility (IC) is essential for achieving stable and predictable outcomes. This paper explores online learning under a novel IC constraint in the context of leader-follower Stackelberg games with policy recommendation. In these games, the leader announces a policy to commit to while recommending a policy for the follower to adopt. The leader's optimal strategy is captured by the Stackelberg equilibrium, where the leader's announced policy maximizes her rewards, and the recommended policy is incentive-compatible—serving as the follower's optimal response given the leader's commitment. We study the online learning problem from the leader's perspective, where she has no prior knowledge of the follower's reward function and hence must infer it solely through observed follower actions. To address this problem, we develop a theory for such games, and propose a provably efficient algorithm that minimizes regrets with respect to both the leader's rewards and violations of the IC constraint. The algorithm integrates the maximum likelihood estimation of the follower’s response model with optimistic planning over an estimated IC constraint. Crucially, we establish that our algorithm is robust to misspecification of the follower's behavioral parameters, exhibiting graceful degradation where the performance loss scales linearly with the estimation error. To the best of our knowledge, this is the first provably efficient online learning algorithm for incentive-compatible decision-making, highlighting the potential of online learning in addressing challenges in misaligned multi-agent systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies online learning for leader–follower matrix games when the leader commits to a policy and also recommends a policy to the follower, under an "IC" (credibility) constraint. The follower is modeled as boundedly rational and computes a one-step mirror-descent update that trades off reward, entropy, and a KL penalty from the leader’s recommendation. The authors formalize ICV via a TV between the recommendation and the follower’s response, and show that QRE is the unique fixed point of the recommendation–response map. This paper proposes PROM, and under linear rewards and finite follower actions they prove \(\tilde O(d^2\sqrt{T})\) bounds on standard regret, IC-violation regret and absolute regret.

### Strengths
1. The paper introduces a new formulation of the objective, where the leader is constrained to a credibility constraint and the follower updates according to the leader's recommendation policy . The ICV metric and its fixed-point connection to the quantal response are interesting and provide a well-defined policy set.
2. The proposed PROM algorithm achieves $\tilde{\mathcal{O}}(d^2 \sqrt{T})$ standard regret and ICV regret, providing a sample-efficient learning method.

### Weaknesses
1. The paper assumes the leader knows\(\eta,\tau\);This can be unrealistic and learning them is nontrivial. It’s unclear how sensitive and robust the results are under noisy or approximate $\alpha, \beta$. 
2. The paper’s “IC” is not the incentive compatibility in the mechanism-design sense. What the paper calls “IC” is closer to a credibility requirement than the truth-telling incentive compatibility in classical mechanism design. And generally the leader has no incentive to add this “IC” constraint and the follower has no clear incentive to add a KL term according to the leader’s policy; these settings and assumptions lack motivation.
3. The technical contribution is incremental to prior work on Stackelberg/QSE learning and entropy-regularized responses.

### Questions
1. How sensitive are the results to the choices of $\eta$ and $\tau$? In particular, how do estimation errors in these parameters influence the results?
2. Suggestion: It is more accurate to use “online learning” rather than “online RL” throughout.  “online RL” typically implies results in Markov decision processes/Markov games, which your current setting (matrix Stackelberg games) does not cover and readers may anticipate analyses beyond the scope of this work.
3. In your analysis and proofs, which aspects of the analysis differ from the $H=1$ case in Chen et al.? In particular, which techniques are newly introduced due to the need to control \textbf{ICV} and distribution shift compared to previous work?

### Soundness
3

### Presentation
3

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
The paper proposes an algorithm to achieve sublinear regret in a variant of the Stackelberg game, where the leader can recommend to the follower, and the follower will take one step of mirror descent from the recommendation.

### Strengths
- The writing of the paper is clear and easy to follow.
- The paper discussed its relevance to prior work clearly.

### Weaknesses
My primary concern with this paper is the lack of motivation. The paper introduces a new setting where players’ objective functions include an additional entropy regularizer. However, this assumption seems questionable, as such regularizers are typically used as a tool to accelerate convergence (see last-iterate convergence) / encouraging exploration (see learning in MDP) rather than being inherent components of the game itself.

Moreover, the paper assumes that the follower performs mirror descent updates from the recommendation due to limited computational power. While this may seem reasonable, the choice of mirror descent is not well justified. More critically, the proposed algorithm requires the leader to know the follower’s learning rate, which is an unrealistic assumption.

In my view, a more natural assumption would be that the follower computes a best response within a neighborhood of the recommendation, constrained by a radius $r$.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
The paper introduces a model of leader-follower games with actions and policy recommendations, in the spirit of models with adverse selection and moral hazard, compare [Myerson 1983](https://www.jstor.org/stable/1912116?seq=4). They consider one-step mirror descent and quantal responses as equilibrium concepts, and provide regret bounds for the computation of optimal policies. The main claimed contributions are a novel incentive compatibility notion and a first efficient algorithm for incentive-compatible decision-making.

### Strengths
The paper is introducing their model carefully, and proposes regret bounds. I could follow the complete specification of the model and their regret statements.

### Weaknesses
I take issue with both of their claimed contributions (the rest of the paper seems to be rather standard in terms of techniques):
1. The claimed incentive condition relating to policy recommendation is well-known in the literature of mechanism design [Myerson 1983](https://www.jstor.org/stable/1912116?seq=4) (original formalism) [Bergemann 2019](https://www.aeaweb.org/articles?id=10.1257/jel.20181489) (survey of information design) [Kamenica, Gentzkow 2011](https://web.stanford.edu/~gentzkow/research/BayesianPersuasion.pdf) (a main application of recommendation-based incentive constraints) and called the *obedience* constraint. There are several methods for its computation. The paper does not relate to this literature. Engaging with it would also help the paper by not requiring to impose obedience in Section 4.1 but by saying that it is a consequence of a revelation principle.
2. The authors claim that their algorithm is the first efficient algorithm for incentive-compatible decision-making seems too broad to me. Would [Zhang et al. '24](https://dl.acm.org/doi/abs/10.1145/3670865.3673536) (which also recommends policies) be an efficient algorithm? Is an efficient algorithm in terms of the description size of the game or purely the time horizon?

If the novelty for both of these contributions can be clarified/demonstrated, I am willing to raise my score.

### Questions
- Please engage with the literature on information design, mechanism design, and Bayesian persuasion.
- Qualify the statement on the novelty of the efficient algorithm.
- Would [Zhang et al. '24](https://dl.acm.org/doi/abs/10.1145/3670865.3673536) (which also recommends policies) be an efficient algorithm? Is an efficient algorithm in terms of the description size of the game or purely the time horizon?

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
This paper studies learning in leader–follower Stackelberg games with partial feedback, where the leader’s reward function is unknown to both agents, while the follower’s reward function is known only to herself. In each round, the leader announces a policy together with a recommended policy for the follower. The follower then updates her policy via a single-step mirror descent. The paper introduces a dynamic incentive-compatibility (IC) constraint on the sequence of recommended policies over time, and proposes an efficient algorithm that achieves $\tilde{O}(\sqrt{T})$ regret with respect to both the leader’s objective and the incentive-compatibility violation (ICV) regret.

### Strengths
1.The paper introduces an interesting setting of learning in Stackelberg games with policy recommendations. The idea of incorporating a recommended policy and incentive-compatibility (IC) constraints is novel in this learning framework.

2.The proposed algorithm is well-designed, and the theoretical results and proofs are clearly presented.

3. The paper is well written and easy to follow.

### Weaknesses
In general, I think this is a fairly interesting paper for ICLR. The potential limitations such as the assumptions of finite action spaces, known parameters $\eta$ and $\tau$, and the restriction to matrix games are clearly discussed in the appendix and can reasonably be left for future work.

My main concern lies in the modeling choice: it is unclear whether modeling the follower with a single-step OMD is the most appropriate approach. In the current finite-action setting, this step does not appear to offer computational advantages over directly computing the quantal best response. I also wonder whether it would make more sense to consider a simpler setting where the follower exactly follows the recommended policy, and what the results would look like in that case.

Minor comments: I suggest that the authors include a summary table listing all key assumptions (e.g., what is known to the leader and follower, whether actions are finite, linearity assumptions, etc.), which would improve readability.

Relevant citation:
Haghtalab, Nika, Chara Podimata, and Kunhe Yang. Calibrated Stackelberg Games: Learning Optimal Commitments against Calibrated Agents. NeurIPS 2023.

### Questions
1. Is the computational complexity of a single-step OMD lower than directly computing the quantal best response? If not, what is the benefit for the follower to decide her policy based on y rather than computing a quantal best response?

2. Can the results be generalized to the setting where the follower directly follows the leader’s recommended policy (with similar IC constraints)?

3. Is the $d^2$ dependence in the bound tight?

### Soundness
3

### Presentation
3

### Contribution
3

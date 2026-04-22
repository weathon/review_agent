# Learning to Play Multi-Follower Bayesian Stackelberg Games

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 10

## Abstract
In a multi-follower Bayesian Stackelberg game, a leader plays a mixed strategy over $L$ actions to which $n\ge 1$ followers, each having one of $K$ possible private types, best respond.
The leader's optimal strategy depends on the distribution of the followers' private types.
We study an online learning version of this problem: a leader interacts for $T$ rounds with $n$ followers with types sampled from an unknown distribution every round. The leader's goal is to minimize regret, defined as the difference between the cumulative utility of the optimal strategy and that of the actually chosen strategies. We design learning algorithms for the leader under different feedback settings. Under type feedback, where the leader observes the followers' types after each round, we design algorithms that achieve $O\big(\sqrt{\min(L\log(nKA T), ~ nK ) \cdot T} \big)$ regret for independent type distributions and $O\big(\sqrt{\min(L\log(nKA T), ~ K^n ) \cdot T} \big)$ regret for general type distributions.
Interestingly, those bounds do not grow with $n$ at a polynomial rate. 
Under action feedback, where the leader only observes the followers' actions, we design algorithms with $O( \min(\sqrt{ n^L K^L A^{2L} L T \log T}, ~ K^n\sqrt{ T } \log T ) )$ regret.
We also provide a lower bound of $\Omega(\sqrt{\min(L, ~ nK)T})$, almost matching the type-feedback upper bounds.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies online learning for multi-follower Bayesian Stackelberg games where a leader commits to a mixed strategy and $n$ followers (each with one of $K$ private types) best respond. The authors analyze two feedback models: type feedback and action feedback. The paper proposes a geometric decomposition of the leader’s simplex into best-response regions and algorithms with sublinear regret. For type feedback they give $\tilde{O}(\sqrt{\min\{L,nK\}}\,T)$ for independent types, $\tilde{O}(\sqrt{\min\{L,Kn\}}\,T)$ for general types, and a matching lower bound $\Omega(\sqrt{\min\{L,nK\}}\,T)$. For action feedback they propose a reduction to stochastic linear bandits with $O(Kn\sqrt{T}\log T)$ regret and a UCB  approach with $O(\sqrt{n^{L}K^{L}A^{2L}}\,L\,T\log T)$ regret.

### Strengths
The paper introduces a clean geometric decomposition of the leader’s strategy space into follower best-response regions and leverages this structure to derive regret guarantees for multi-follower Bayesian Stackelberg games. The resulting bounds avoid polynomial growth in the number of followers $n$, and the authors provide a lower bound that nearly matches the type-feedback upper bounds.

### Weaknesses
1. In the general single-leader-multi-follower setting, a follower’s utility typically depends on both the leader’s action and other followers’ actions (i.e., there are cross-follower externalities). The paper assumes each follower’s payoff depends only on their own action and type plus the leader’s action—which substantially simplifies the model and makes the extension from one to many followers more direct. While this assumption enables clean analysis, it limits the applicability of the results to settings without strategic interdependence among followers.
2. The analysis assumes followers play the exact best response in every round. This is a strong assumption for a learning setting: in many practical environments followers also learn, face estimation error, or act under noisy rewards, leading to approximate or stochastic best responses. The results would be more interesting with guarantees to $\varepsilon$-best responses or noise in followers’ payoff observations.
3. The reduction to stochastic linear bandits is not novel. Similar reductions have appeared in prior work [1].


[1] Nearly-optimal bandit learning in stackelberg games with side information. Balcan, Maria-Florina, et al.

### Questions
Could you elaborate more the statement: “In comparison, Conitzer & Sandholm (2006) prove that the optimal strategy is NP-hard to compute in BSGs with an asymptotically increasing $L$. We show this is polynomial-time solvable for a constant $L$.” (Line 274)
Could you cite the exact theorem/lemma in Conitzer & Sandholm, specify the precise problem and briefly contrast your assumptions with theirs to explain why constant $L$ yields polynomial time here.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies multi-follower online Bayesian Stackelberg games. The authors study 2 types of feedback: type feedback, where at each turn the leader observes the tuple of realized types, and action feedback, in which the leader observes the individual actions. Clearly knowing the types implies knowing the actions.
Types are either drawn from a joint or product distribution.
Under type feedback the authors gives an algorithm which attains regret of $\sqrt{T\min(L,K^n)}$  under joint distribution, or, $\sqrt{T\min(L,Kn)}$ under product distribution, where $L$ is the number of actions of the followers, $K$ the number of possible types and $n$ the number of players. The algorithm revolves around 
Under type feedback (under joint distribution) the algorithm is simply follow the leader. Interestingly, the analysis shows concentration around the empirical utility rather than under the correct type distribution. This aligns with the following section on action feedback, where working in the utility space is the correct trick to get around exponential dependencies.
Under type feedback and independent distribution, the algorithm simply estimated each distribution separately.
Under action feedback, the authors rely on a known reduction for a related problem of online Bayesian persuasion. The technical challenge here is to show that there exists a single LP that solves the problem.

### Strengths
This is a good paper that has non trivial contributions, even if maybe a bit lackluster from a technical side.

### Weaknesses
I see two main weaknesses:
1. Even if I find the overall presentation satisfactory, its quality degrades in some specific points. It is not clear where do you use the best response partitions. For example, the discussion under Lemma 4.1 seems to be very important, but right now it feels somewhat obscure and does not do a good job at explaining the crucial points. Another point in which the presentation is a bit confused is around line 409. It is not even clear if this part relies at all on the reduction of Bernasconi et al. or not. I think not, but better clarity would be nice. Overall, I think the paper suffers from having too many results compressed. I would suggest to the authors that maybe they defer some of the results to the appendix and devote more space to points that are now a bit compressed.
2. Why do you not consider adversarial types? I think your FTL approach would still be a good candidate for adversarial types (obviously by using FTPL instead). The reduction of Bernasconi for sure also works for adversarial types. So it is a bit strange that you do not consider this problem, which is the most studied in Online learning in Econ settings.

* Typo in statement of Lemma C.1

### Questions
Is the improvement from K^{3n/2} to K^n only due to the fact that you consider stochastic types rather than adversarial ones?

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
3

### Summary
The paper studies the online learning version of the multi-follower Bayesian Stackelberg game. The authors consider the setting where the leader knows each follower's utility function but not their private types. The paper designs learning algorithms for the leader under both type-feedback and action-feedback settings, and provides detailed regret analysis.

### Strengths
The paper is well-written and easy to follow. The authors conduct an in-depth analysis of the geometric properties of the game, which provides insights for the learning algorithms. I find the proposed algorithm interesting as it significantly reduce regret. The theoretic results look solid and strong to me (did not check all the proof details though). The multi-follower setting is much more general than the standard version, and the propose method can potentially improve the applicability of the Bayesian Stackelberg game.

### Weaknesses
There are some minor issues with the paper. The definition of $W$  in Subsection 3.2 seems inconsistent. If $W$ is a matrix, then the $i$-th element of $W$ should be $w_i: \Theta^K \mapsto A^K$, not $w_i: \Theta \mapsto A$. Am I missing something here?

### Questions
Please see my comments above.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper studies the problem of learning an optimal commitment against multiple followers, each of whose type is drawn from a distribution. In doing so, this paper generalizes the usual Stackelberg paradigm of optimal commitment against a single follower and even the Bayesian Stackelberg setting where the single follower’s type is drawn from a distribution.

The latter problem is known to be NP-hard in general, implying hardness for this more general problem. A natural algorithmic approach for this problem is to partition the leader’s action space into regions that correspond to best-response regions for the different follower profiles. However, the bottleneck to such an approach is that the number of partitions grows exponentially in the number of types for the follower — this problem doubly applies with multiple followers, since the number of best-response (BR) profiles has exponential dependence on the number of followers as well.

This paper surmounts this technical difficulty by using results from computational geometry to bound the number of non-empty regions with exponential dependence only on the number of “pure” actions of the learner. The paper also provides an explicit enumeration of these non-empty regions with the same asymptotic parameters, thus providing an offline algorithm for the problem. Then, they strengthen this to an online learning algorithm, which is able to learn the optimal policy at a rate faster than just learning the distributions based on intricate technical analysis. Their results cover both the settings where the type of the followers is seen at the end of each round versus just their action, and provide lower bounds with the same exponential dependence.

### Strengths
The paper provides a novel technique that breaks through the barrier of exponential best response profiles with multiple followers/ follower types, by focusing on the dimension of the leader's action space. This allows for a fine-grained view of the complexity of finding an optimal commitment and opens the door for the upper bound results in the paper which as noted, from the first positive results for multiple followers. The paper covers a variety of settings and is overall a very strong submission with novel technical contributions.

### Weaknesses
NA

### Questions
Do the results extend to settings where the action sets are arbitrary convex sets of a specified dimension, especially for the leader?

### Soundness
4

### Presentation
4

### Contribution
4

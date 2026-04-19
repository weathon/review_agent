# Minimax-optimal trust-aware multi-armed bandits

- Decision: Reject
- Scores: 8, 6, 8, 8

## Abstract
Multi-armed bandit (MAB) algorithms have achieved significant success in sequential decision-making applications, under the premise that humans perfectly implement the recommended policy. However, existing methods often overlook the crucial factor of human trust in learning algorithms. When trust is lacking, humans may deviate from the recommended policy, leading to undesired learning performance. Motivated by this gap, we study the trust-aware MAB problem by integrating a dynamic trust model into the standard MAB framework. Specifically, it assumes that the recommended and actually implemented policy differs depending on human trust, which in turn evolves with the quality of the recommended policy. We establish the minimax regret in the presence of the trust issue and demonstrate the suboptimality of vanilla MAB algorithms such as the upper confidence bound (UCB) algorithm. To overcome this limitation, we introduce a novel two-stage trust-aware procedure that provably attains near-optimal statistical guarantees. A simulation study is conducted to illustrate the benefits of our proposed algorithm when dealing with the trust issue.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper integrates human trust dynamics into the multi-armed bandit (MAB) framework to address the influence of trust on sequential decision-making. Recognizing that a lack of trust can cause users to deviate from recommended policies, potentially leading to suboptimal outcomes, the authors critique existing approaches for focusing primarily on learning in uncertain environments without accounting for human willingness. To address this gap, they propose a two-stage trust-aware MAB algorithm that adapts to the decision-maker’s trust level, aiming to build and maintain trust while efficiently identifying optimal actions. The authors provide a minimax regret bound for this trust-aware MAB model and highlight that common algorithms, such as Upper Confidence Bound (UCB), can incur near-linear regret when trust is compromised. Theoretical results show that the proposed algorithm achieves a regret bound of order $\sqrt{KH \log H}$ without requiring prior knowledge of the decision-maker’s trust level or policy.

### Strengths
1. This paper frames a new and interesting problem setting in MAB, which considers the decision-making problem with humans in the loop and accounts for more robust solutions when people do not trust the recommended actions.
2. To tackle this new problem, this paper proposes a novel UCB-based algorithm to solve the issue of lack of trust and provides rigorous theoretical analysis for the lower bound and the upper bound, as well as why the classical UCB algorithm fails.
3. To the best of my knowledge, this paper is the first work to develop a trust-aware algorithm that provably achieves near-minimax optimal statistical efficiency. The theoretical results are validated by empirical studies.
4. The presentation of the paper is clear and easy to follow.

### Weaknesses
I do not have major concerns about the novelty and contributions of the paper. However, some assumptions could be justified/relaxed to enhance the practical value of the work 

1. From my perspective, Assumption 1 is quite strong in the sense that the implementor’s trust set $\mathcal{T}$ must contain one optimal arm with reward $r*$. In the recommender system, for example, there are thousands of products to recommend and it is not very realistic that the implementer can have access to such  $\mathcal{T}$. Can this assumption be relaxed if $\mathcal{T}$ only contains the optimal arm with some probability or suboptimal arm whose reward is bounded by $\epsilon$. This is more realistic since we may have some offline data before the learning game which can output the optimal arm with a certain probability or an arm with bounded suboptimal gap.
2. The current algorithm assumes that whether the user adopts the recommended policy $\mathcal{X}_h$ is a Bernoulli random variable and can be observed. However, in real applications, such feedback is not directly accessible as the implementor may not willing to reveal whether they take the recommended policy or not (suppose he does not trust others).  As such, I think the current algorithm may not work without this feedback.

### Questions
1. Can the current algorithm handle the case when the implementer's trust set only contains the optimal arm with some probability or the suboptimal arm whose suboptimality gap is bounded?
2. Can the authors give some practical applications where the implementer would like to reveal their choice on whether he/she follows the recommended policy? If the implementer chooses not to reveal this feedback, is there any way the learner can infer whether the implementer follows the recommended policy from other types of feedback?

### Soundness
3

### Presentation
3

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
This paper formulates the problem of trust-aware stochastic bandits in which the algorithm cannot interact with the environment directly, but instead recommends an action to the implementer that then either plays the recommended action with probability $t_h$ or plays a different action with probability $1 - t_h$. The paper assumes that $t_h$ follows a dynamic model, where (roughly-speaking) $t_h$ increases if the algorithm plays an action in a (unknown) trusted set of actions, and decreases if the algorithm plays an action not in this set. They give an algorithm that first explores the actions to identify the set of trusted actions and then plays UCB. They show that this algorithm enjoys $\tilde{O}(\sqrt{KD} + K^4)$ regret and give a regret lower bound of $\Omega(\sqrt{KT})$. They also show that UCB suffers linear regret in this setting.

### Strengths
- The proposed algorithm provides a way to estimate the dynamical model with stochastic feedback in a small number of rounds. This appears to be a relatively difficult problem that is certainly more difficult than estimating the mean of a fixed distribution (as is typically done in explore-exploit problems for stochastic bandits).
- The paper shows (nearly)-minimax optimal regret.
- The presentation is good.

### Weaknesses
- I didn't see where the paper states the information available to the algorithm. It looks like the algorithm (in Algorithm 1) uses $\chi_h$. I don't think this is very realistic because the algorithm is observing the implementer's internal randomness. It would be more realistic if the algorithm observed the implementer's action $a_h^{ac}$. Note that this is not the same as observing $\chi_h$ because the implementer's policy could be the same as the algorithm's in some rounds and therefore $a_h^{pm} = a_h^{ac}$ even though $\chi_h$ might be $0$.
- I think the claims of $\tilde{O}(\sqrt{K T})$ are stated incorrectly, as (to my knowledge) it is standard to show all $K$ terms when one is shown, i.e. need to state it as $\tilde{O}(\sqrt{KT} + K^4)$.
- Although it is often stated as *nearly* minimax optimal regret, there are places that claim minimax optimal regret (e.g. the title) which I don't believe is a correct characterization given that the regret bounds are $\tilde{O}(\sqrt{KT} + K^4)$ and $\Omega (\sqrt{KT})$.

### Questions
- Can you clarify the reason for providing the algorithm with the implementer's internal random variable $\chi_h$?
- I suggest stating the regret upper bounds more precisely in the introduction.

### Soundness
2

### Presentation
3

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
This paper investigates the trust-aware multi-armed bandit (MAB) problem by incorporating a dynamic trust model into the conventional MAB framework. The authors address the discrepancy that arises when humans may not fully adhere to recommended policies due to varying levels of trust. To mitigate the limitations of standard MAB algorithms in these trust-dependent scenarios, they propose a two-stage trust-aware UCB algorithm and theoretically derive minimax regret bounds. A simulation study illustrates the practical benefits of their approach.

### Strengths
1) The paper addresses a critical gap by incorporating human trust into MAB frameworks, which is relevant for real-world applications like human-robot interactions.

2) The authors establish the minimax lower bound for their trust-aware model and demonstrate that standard UCB algorithms can incur near-linear regret under trust-related deviations.

3) The authors develop a novel trust-aware UCB algorithm, supported by rigorous theoretical results, including a near-minimax optimal regret bound.

### Weaknesses
1) The trust model used in this paper (the disuse trust behavior model) is relatively simple, and its applicability to more complex trust frameworks may be limited.

2) While the regret bound of Algorithm 1 in Theorem 2 is nearly optimal with respect to the time horizon $H$, the dependence on $K$ could be improved. As the authors note in Remark 4, optimizing the trust set identification stage could help reduce the burn-in cost. Additionally, Assumption 1 might further improve the regret in the trust-aware exploration-exploitation stage: since in Stage 2, the algorithm only needs to explore the arms within the trust set, it’s possible that the regret could scale as $O(\sqrt{|\mathcal{T}| H \log H})$.

3) The numerical experiments, which consider $ K = 10 $ arms with a uniform own policy over two arms, seem limited. The authors may want to explore more complex own policies involving more arms. Furthermore, it is unclear why the trust-aware UCB (blue) outperforms the UCB without trust issue (green) in Figure 1.

### Questions
1) It would be great to include a formal proof demonstrating that, without Assumption 1 (possibly in combination with certain assumptions about the own policy), no algorithm can achieve sublinear regret.

1) Exploring specific types of implementers with defined own policies could be interesting; prior knowledge of these policies might further enhance the regret results.

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
4

### Summary
This paper considers a stochastic multi-armed bandit setting where the actions recommended by the bandit policy is not perfectly executed by the human actor. Specifically, they consider a setting with $K$ arms played over $H$ timesteps, where the human has some set $\mathcal{T} \subset [K]$ of arms that they trust. This trust set is unknown to the policymaker, leading to information asymmetry. 

The likelihood that the human follows the recommendation made by the policymaker depends on the "trust" that the policymaker has built so far. The trust level $t_h$ is defined in terms of the percentage of the time that the policymaker has recommended an arm that belongs in the trust set $\mathcal{T}$. 

The authors show that UCB can lead to supralinear regret above $H / \sqrt{\log (T)}$ for some combination of bandit, trust set, and human policy. Thus, the authors propose a two-stage "trust aware" UCB-based algorithm where the first stage involves uniform arm selection, to eliminate arms outside the trust set, then a second stage that performs exploration/exploitation similar to standard UCB. The authors show that the expected regret is on the order $O(\sqrt{KH})$ and run an experiment showing improved regret in a single setting with $K=10$ arms and a horizon length of $H=14,000$.

### Strengths
1. The authors consider an important problem, of imperfect execution by downstream actors during the implementation of a sequential decision-making algorithm. The model is reasonably formulated, with a changing "trust level" by the human depending on the policy recommendations so far. The notion of private information (i.e., the trust set) that is not visible to the policymaker is also reasonable and well-formulated. 

2. The theoretical results are rigorous, first showing how the standard UCB algorithm fails in this setting, then showing $\sqrt{KH}$ minimax optimal regret for their proposed algorithm. Additionally it is useful to observe that when the time horizon $H$ increases, the larger polynomial regret $O(K^4 \log^2(H))$ for the trust-set identification stage is negligible; this is reasonable as often algorithms assume that $H$ is significantly larger than $K$.

3. The paper is well-written and generally well presented.

### Weaknesses
1. Empirical results are limited and fail to show robustness. There is only a single setting, where the number of arms is rather small ($K=10$), the trust set is extremely small (just 2 arms, $\mathcal{T} = \{9,10\}$), and the human has a very naive policy (uniform exploration, with no learning). The arms are also selected so that the trust set is the two highest-reward arms. How does performance compare when the trust set is 50\% of arms? What about a larger number of arms? What about when the trust set is the optimal arm and many very poor-performing arms?

Or what about the setting that you describe, where there may be multiple equally optimal arms, and the trust set only includes one of the optimal arms?

2. The assumption that the trust set includes the optimal arm is very strong, but I understand that is necessary for the analysis.

3. A more reasonable "trust level" would be based on performance, not just blindly whether or not the arms selected follow the humans' intuition of the a priori trust set. For example, this would include the case where there are multiple optimal arms $A,B$, the algorithm repeatedly recommends arm $A$, but only arm $B$ is in the humans' trust set --- the trust level should not monotonically go down in this case.

Another way of formulating this would be to have the algorithm designer be able to influence the trust set $\mathcal{T}$ to add more arms, which are shown to perform well.



Smaller comments

1. The plot is hard to read (especially in a printout); it would help to increase the font size.

### Questions
1. The trust level as formulated is very hard to improve. Would it make sense to add some decay, so that the trust level is only based on the past $L$ number of observations? (so that trust can be better improved over time)

### Soundness
4

### Presentation
4

### Contribution
3

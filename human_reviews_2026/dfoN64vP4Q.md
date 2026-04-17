# Lipschitz Bandits with Stochastic Delayed Feedback

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
The Lipschitz bandit problem extends stochastic bandits to a continuous action set defined over a metric space, where the expected reward function satisfies a Lipschitz condition. In this work, we introduce a new problem of Lipschitz bandit in the presence of stochastic delayed feedback, where the rewards are not observed immediately but after a random delay. We consider both bounded and unbounded stochastic delays, and design algorithms that attain sublinear regret guarantees in each setting. For bounded delays, we propose a delay-aware zooming algorithm that retains the optimal performance of the delay-free setting up to an additional term that scales with the maximum delay $\tau_{\max}$. For unbounded delays, we propose a novel phased learning strategy that accumulates reliable feedback over carefully scheduled intervals, and establish a regret lower bound showing that our method is nearly optimal up to logarithmic factors.
Finally, we present experimental results to demonstrate the efficiency of our algorithms under various delay scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies Lipschitz bandits (that is, the action set is continuous defined over a metric space) with stochastic delayed feedback. The authors give algorithms for both bounded and unbounde delay setting with theoretical bounds.

There's also a lower bound given to demonstrate the nearly tightness of upper bounds. There are simulation results as well but given the literature there's no baseline algs. to compare with.

### Strengths
1. A new setup that hasn't been studied and requires new design in the algorithm.
2. Lower bound compliments the results

### Weaknesses
n/a.

### Questions
1. My understanding from Sec. 4 is that, the challenge comes from the correlation between arms (observing one arm also helps learn its neighbors), which indeed prevents us from showing some certain ``suboptimality gap inequality.'' However, intuitively, wouldn't the shrinking confidence level a good thing, which means the uncertainty is reduced faster? Why does it turn out to be an ``issue'' rather than something we can/should leverage?
2. Could the authors elaborate more on the computational complexity of Alg. 2?
3. Even in the bounded delay setting, ``This implies that feedback is always eventually observed and never missing,'' is not precise, because feedback can still come after $t=T$?
4. Some minor issues: ``BOLD and QPM-D'' appears in Line 094, but it's hard to understand what they are; $d_z$ appears in Line 063 without even an informal definition.

### Soundness
4

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
4

### Summary
The work studies the problem of Lipschitz bandits problem where the reward is not observed immediately rather after a random delay. The authors develop two algorithms, a zooming algorithm called as Delayed Zooming Algorithm for bounded delays and DLPP for unbounded delay. They also showcase the optimal performance of these algorithm with theoretical guarantees in regret bound. To support this regret guarantee, they also provide lower bound guarantee. The theoretical work is complemented with the experimentation to show the effectiveness of the algorithm.

### Strengths
The problem formulation is new and interesting with Lipschitz bandits having stochastic delayed feedback.

The work proposed two algorithms covering two regimes wrt delay, Delayed Zooming Algorithm for bounded delay and DLPP for unbounded delay. 

The lazy update trick is neat wrt confidence radius on unpulled arm.

The work also includes matching lower bound that aligns with the upper bound up to log factor and is important to showcase the importance of the upper bound

### Weaknesses
Alg 1 and Alg 2 both assume that they have access to a oracle for covering but it is not the case in many practical scenarios and can be expensive. 

Both the delay and reward distribution are assumed to be independent and it is not the case in many real world system. 

The comparisons are not established with respect to other censored bandit setting or closer setting that could be made to work with some relaxation for considering them for baseline. This could have improved the experimentation setting to help understand the performance gains. 

Also, the experimental setting is minimal with only simulations having different delay distributions. There is no real world dataset used to evaluate the experiments.

### Questions
Since Algorithm 1 requires bounded delay \tau_{max}, can delay from the distribution be scaled down to account of practical situation (in case a wide delay Random variable has to be accommodated) so the regret guarantees still holds ?

The problem setting assumes that the delay distribution and reward distribution are independent. However, in many situation like online advertisement where the delay (conversion) in getting a reward is often associated with reward. In this case, how would the analysis break if the delay depends on the reward?

In DLPP, phases end after every ball gets $v_{m}$ samples to be observed and since $\tau$ also includes $\inf$, and if the prob. distribution is supported more on the $\inf$ (so reward is never observed) so how does the algorithm overcome this scenario.

You measure regret on generated rewards not only on the observed reward. In real systems we only evaluate on observed conversions. How different would the bounds look under only observed reward and would the algorithm need to change to accommodate this ?

Also, if $P(\tau = \inf) > 0 on delay distribution, Does that not reduce to the standard Censored Bandits ? If it does reduce, how does the regret of proposed algorithm compare against the Censored Bandits.

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
This paper studies the stochastic Lipschitz bandit problem with stochastic delayed feedback. 
For the case of bounded delays, the authors extend the zooming algorithm to the delayed setting and obtain a regret bound of 
$
\tilde{O}\left(T^{\frac{d_z+1}{d_z+2}} + \tau_{\max} T^{\frac{d_z}{d_z+2}}\right),
$
where $\tau_{\max}$ denotes the maximum delay. 
For the unbounded-delay case, they propose a new algorithm that achieves nearly optimal regret bounds. 
The paper further establishes instance-dependent lower bounds in the general unbounded-delay setting and presents empirical results that validate the theoretical findings.

### Strengths
1. The paper is well organised and  is easy to follow
2. The quantile-based upper bound and the matching lower bound (up to logs) are compelling and intuitive for unbounded delays.

### Weaknesses
1. Zooming algorithm is only proven for bounded delays, yet experiments show it works for unbounded delays. This gap is acknowledged but not resolved, leaving a significant theoretical question unanswered

### Questions
1. What are the main technical challenges in analysing the zooming algorithm under unbounded delays? 
Empirically, the zooming algorithms seem to perform better than DLPP in both bounded and unbounded delays.

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
The paper studies the standard Lipschitz bandit with stochastic delayed feedback. The paper gives theoretical results in bounded delay and unbounded delay scenarios. The paper conducts experiments to show the effectiveness of the proposed algorithms.

### Strengths
Originality: the exact problem setup has not been studied before to my knowledge. 

Quality/clarity: I like the completeness of the paper. The setup is clear. The algorithm description is clear. The reason why the proof is difficult is clear and the experiments are quite comprehensive for a theoretical paper. 

Significance: This paper paves a way for more studies in delayed feedback of Lipschitz bandit.

### Weaknesses
A minor thing: In related work there is a lot of relevant literature missing on bandits with delayed feedback. For example: 

Thompson Sampling with Unrestricted Delays; Delay as Payoff in MAB; Impatient Bandits: Optimizing for the Long-Term Without Delay; Adaptivity and confounding in multi-armed bandit experiments

### Questions
1. For line 154 there are two $c$'s, one is $N_c(r)$ and one is the constant, maybe change this to make it less confusing?

### Soundness
3

### Presentation
3

### Contribution
3

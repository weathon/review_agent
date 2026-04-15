# Avoiding Catastrophe in Online Learning by Asking for Help

- Decision: Reject
- Scores: 6, 5, 6, 6

## Abstract
Most learning algorithms with formal regret guarantees assume that no mistake is irreparable and essentially rely on trying all possible behaviors. This approach is problematic when some mistakes are _catastrophic_, i.e., irreparable. We propose an online learning problem where the goal is to minimize the chance of catastrophe. Specifically, we assume that the payoff in each round represents the chance of avoiding catastrophe that round and try to maximize the product of payoffs (the overall chance of avoiding catastrophe) while allowing a limited number of queries to a mentor. We first show that in general, any algorithm either constantly queries the mentor or is nearly guaranteed to cause catastrophe. However, in settings where the mentor policy class is learnable in the standard online model, we provide an algorithm whose regret and rate of querying the mentor both approach 0 as the time horizon grows. Conceptually, if a policy class is learnable in the absence of catastrophic risk, it is learnable in the presence of catastrophic risk if the agent can ask for help.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel online learning framework in which the learner aims to minimize the catastrophe probability, that is, avoid choosing catastrophic actions. In order to do that, it is assumed the possibility to query a mentor (the baseline) at time $t\in[T]$, which returns the mentor's action associated with the current state $s_t$. Moreover, local generalization of the mentor policy is assumed (namely, a sort of continuity assumption between states given the mentor policy). The performance metrics used to evaluate the online algorithms are the regret between the product of the avoiding catastrophe probabilities attained by the learner and the mentor and the number of queries performed. First, the authors show the impossibility of learning in the general setting. Thus, the authors focus on the case of finite VC or Littlestone dimension. In such a setting, the authors provide a simple hedge-kind of algorithm that attains sub constant regret and a sublinear number of queries to the mentor during the learning dynamic.

### Strengths
I believe that the setting proposed in this work is of interest from a theoretical perspective.
Indeed, to the best of my knowledge, the framework is novel, and as pointed out in the work, it presents some peculiar theoretical challenges. The results seem correct and reasonable to me. Finally, the authors put much effort into explaining the main ideas behind the theoretical results.

### Weaknesses
To me, there are two main weaknesses in this work. The first one is the lack of particular algorithmic novelty. The algorithm proposed is a clear, simple adaptation of hedge. Moreover, this work strongly relies on many different results from statistical learning theory and online learning (see Section 5.2, where all the results belong to existing works). This is indeed not a sufficient reason for rejection, but somehow, it worsens the contribution of the work. 

The second weakness concerns the practical relevance of the setting. While I agree that the setting is interesting from a theoretical perspective, it is not clear to me why it should be of interest in real-world scenarios. Specifically, the authors clearly state that this setting is of practical relevance since, unlike standard online learning, it allows for the avoidance of catastrophic actions. In contrast, the standard exploration-exploitation trade-off requires, in general, to try all the possible actions, even really dangerous ones. Nevertheless, there exist many works on learning in the presence of **unknown** constraints which exactly tackle the problem specified above. While without further assumptions, it is not possible to satisfy unknown constraints in every round, it is sufficient to require a feasible solution in input, to be simultaneously no-regret w.r.t. the rewards and avoiding possible dangerous actions with high probability (see, e.g., [Liu et al. 2021], [Stradi et al. 2024] for online learning in MDPs). Moreover, notice that those papers do not need any query to the mentor, and as previously specified, do not focus only on avoiding bad actions, but they maximize specific unknown rewards functions. Finally, while their regret is sublinear in $T$, their regret definition seems to be much stronger since the baseline is the optimal solution and not a general mentor policy.


[Liu et al., 2021] "Learning policies with zero or bounded constraint violation for constrained mdps."

[Stradi et al. 2024] "Learning Adversarial MDPs with Stochastic Hard Constraints"

### Questions
Could you please discuss on the second weakness? Which are the main advantages of your setting w.r.t. the one described above?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper considers a new variant of the classical online learning setting in which the aim is to avoid "catastrophe" relative to a mentor policy \pi^m. Concretely, letting \mu denote the reward, the aim is to choose a sequence of actions a_1, ..., a_T based on a sequence of covariates s_1, ..., s_T such that the following notion of regret is small:

\prod_{t=1}^{T}\mu(s_t, pi^m(s_t)) - \prod_{t=1}^{T}\mu(s_t, a_t).

That is, compared to the standard online learning formulation (e.g., Cesa-Bianchi and Lugosi), which considers regret of the form \sum_{t=1}^{T}\mu(s_t, pi^m(s_t)) - \sum_{t=1}^{T}\mu(s_t, a_t), we care about the product of rewards instead of the sum of rewards. This reflects the fact that we are averse to catastrophe, since a very small reward at any round $t$ can completely ruin our prospects of achieving low regret.

Beyond the differences above, the setting the authors consider also has the feature that the reward is never observed. Rather, the learner can choose to query the mentor at each round $t$ and observe pi^m(s_t). The aim is to ensure the regret and the number of mentor queries is sublinear in $T$.

The authors provide the following results:
* When (1) the mentor satisfies a "local generalization" property related to Lipschitzness and (2) their policy belongs to a class \Pi which is either a Littlestone class or a VC class (with the additional assumption that outcomes are "smooth") it is possible to achieve sublinear regret and expert queries through a variant of the exponential weights algorithm.

* Without the Littlestone or VC assumptions, sublinear regret is impossible, even under the local generalization assumption.

### Strengths
This paper makes a reasonable contribution to the online learning literature by proposing what is, to my knowledge, a novel setting and problem formulation. I believe the upper and lower bounds the authors provide are novel and do not necessarily follow from existing work. I also found the paper to be generally well written and easy to follow.

### Weaknesses
The main drawbacks preventing me from giving a higher score are as follows:

* While the problem formulation is loosely inspired by literature on AI risk, it is unclear whether the new problem formulation the authors provide actually has implications for this literature. Spelling out potential connections and implications of the results seems important, as otherwise I am worried the impact might be rather limited (e.g., a niche setting of interest to online learning experts, but not necessarily of interesting outside the core theory community). Overall, if there are interesting consequences of algorithic results here it would be great to highlight this.

* Given that the main contribution of the paper is to introduce a new problem formulation, it seems important to justify all of the assumptions and argue that they are fundamental. Here, I am somewhat concerned about the local generalization assumption, which seems a bit arbitrary and inelegant. While I can believe that some assumption of this type might be required, it is not clear that this specific one is fundamental. In particular, I do not like that the assumption requires that $s$ lies in Euclidean space, which feels quite arbitrary and not really in the spirit of online learning (which is generally agnostic to the space of covariates, and depends on it only through assumptions on the class \Pi itself). Nailing down what is the right notion of local generalization would definitely strengthen the paper.

-----

Some minor comments:
* The authors may want to compare with the literature on online learning with the logarithmic loss. E.g., if we do standard online learning with additive regret, but choose \log(\mu(s_t, a)) as the reward, then a standard additive regret bound of the form \sum_t reward(s_t, \pi^m) - reward(s_t, a_t) \leq Reg implies that \log(\prod \mu(s_t, pi^m)/\mu(s_t, a_t)) \leq Reg. This is a different guarantee from what the authors provide, but has a similar flavor, so it might be interesting to compare.

* This setting also has an imitation learning flavor, so it would be good to add some discussion on how it connects to the IL literature.

* The term "state" is a bit confusing/misleading since this is a standard online learning setting as opposed to RL (i.e., we just take the states as given, and are not interesting in counterfactuals wrt how they might have evolved if we had acted differently). Calling them "contexts" or "covariates" would be more clear.

### Questions
Is the local generalization assumption fundamental? See comments above

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new model of online learning, with the performance measured in terms of the product of the per-step value (probabilities). Under this new setting, the authors provide both difficulty results and an algorithm for VC/Littlestone classes.

### Strengths
I believe it makes much sense to formulate "asking for help" in terms of query complexity. It is clearly of importance to understand how access to expert advice helps online learners. The proposed algorithm, though looks standard, may also be of techical interest,

### Weaknesses
I am a little bit skeptical of the formulation of regret. 

1. The probabilistic interpretation of the product $\prod_{t=1}^T \mu(s_t,a_t)$ is unclear. Given the observed sequence of $(s_1,a_1,...,s_T,a_T)$, the product is the probability of avoiding catastrophe only if the catastrophe at step t does not affect the s_{t+1}.

2. For the regret to be non-trivial, it is necessary that $\sum_{t=1}^T -\log \mu^m(s_t)$ to be bounded by a constant as T tends to infinity. I believe this is a very strong assumption on the mentor, and the hence the upper bound can be trivial for most scenarios. It is possible to consider a less restrictive setting? The hardness result may not be an excuse here.

3. It looks possible to take a logarithm and convert this problem to the standard no-regret learning setting, with the goal of achieving sub-constant regret. Given that the mentor needs to achieve constant loss, this goal seems to be achievable. It would be beneficial to include a discussion on why it is infeasible.

### Questions
In Sec 3, Q_T is defined to be a deterministic quantity of the algorithm. But all the bounds are in terms of E[Q_T].

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The submission studies an online disaster avoidance problem. The learning agent is allowed to query the mentor to acquire information to avoid disaster. To circumvent the tractability issue of Bayesian methods, a hedge-based algorithm is proposed under an additional assumption called “local generalization.” Also, unlike conventional online learning, the regret is defined in a multiplicative way, and the proven regret bound (Theorem 5.2) is subconstant. Besides the positive result, the submission also shows an impossibility result (Theorem 4.1) in a general case.

### Strengths
- A non-Bayesian approach to address the online disaster avoidance problem.
- The analysis simultaneously controls the regret and the query frequency.
- A novel packing argument is developed to bound the query complexity.

### Weaknesses
- (1) It seems to be too advantageous for the learning agent to be able to query the optimal policy (i.e., the mentor), given that a local generalization assumption has been made for exploitation. Practically speaking, if querying is allowed, the learning agent should also have to find out who the optimal target to query in the policy class is.
- (2) The argument (lines 508–514) for the novel packing (line 460) is not clear enough. An explanation of the difficulty and the technical contribution is needed.
- (3) There are confusing sentences hindering the readability. These are the sentences in line 126 (As a corollary, …) and line 134 (We initially …).

### Questions
- (1) Are there other impossibility results besides Theorem 4.1, especially in the Bayesian regime? If the submission provides the first impossibility result, that is a contribution, too. If there are other impossibility results, we should list them and compare them with Theorem 4.1. 
- (2) Is the multiplicative objective really a good choice? For example, consider the following two payoff sequences: (0, 0.999, 0.999, 0.999) and (0.1, 0.1, 0.1, 0.1). The total payoff (multiplicative) of the first sequence is 0 and is smaller than that of the second one. But intuitively, the second sequence may have a better chance of avoiding a disaster.

### Soundness
3

### Presentation
2

### Contribution
2

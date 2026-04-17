# How to Lose Inherent Counterfactuality in Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Learning in high-dimensional MDPs with complex state dynamics became possible with the progress achieved in reinforcement learning research. 
At the same time, deep neural policies have been observed to be highly unstable with respect to the minor variations in their state space, causing volatile and unpredictable behaviour. 
To alleviate these volatilities, a line of work suggested techniques to cope with this problem via explicitly regularizing the temporal difference loss to ensure local $\epsilon$-invariance in the state space.
In this paper,  we provide theoretical foundations on the impact of 
robust, i.e. adversarial,
training on reinforcement learning.
Our comprehensive theoretical and experimental analysis reveals that standard reinforcement learning inherently learns counterfactual values while recent training techniques that focus on explicitly enforcing $\epsilon$-local invariance cause policies to lose counterfactuality, and further result in learning misaligned and inconsistent values. 
In connection to this analysis, we further highlight that this line of training methods breaks the core intuition and the true biological inspiration of reinforcement learning, sacrifices essential inherent skills that enable reasoning and generalization, and introduces an intrinsic gap between how natural intelligence understands and interacts with an environment in contrast to AI agents trained via $\epsilon$-local invariance methods.
The misalignment, inaccuracy and the loss of counterfactuality revealed in our paper further demonstrate the need to rethink the approach in establishing truly reliable and generalizable reinforcement learning policies.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper provides empirical and theoretical evidence challenging the prevalence of $\epsilon$-invariant adversarial retraining in robust reinforcement learning (RL), and provides a new dimension of analysis to the performance-robustness tradeoff in adversarial RL. Specifically, the paper provides proofs that policies can either be $\epsilon$-invariant or optimally estimate state-action values, but not both. The empirical results follow this conclusion, showing that when $\epsilon$-invariant policies are forced to choose actions based on counterfactual observations, performance degradation is worse than would otherwise be seen with nominal policies.

### Strengths
- The paper provides strong theoretical and intuitive results showcasing the downsides of commonly used robust RL frameworks.
- The work is very insightful and gives strong justifications for where the field of adversarial robust RL should be looking.

### Weaknesses
#### Preliminaries
By reading the paper, the term _counterfactuality_ can be implicitly understood as the accuracy of the value function for non-optimal actions. The paper would benefit from an explicit definition, as it would help understand the implications of losing counterfactuality.

#### Minor formatting:
- Figure 6 caption, "Lipschitz" has a "c".
- In section 2, concerns on reliability, the citation for [1] has the author's first and last name switched, should be Zhang et. al. (dblp: https://dblp.org/rec/conf/nips/0001CX0LBH20.html)

[1] Huan Zhang, Hongge Chen, Chaowei Xiao, Bo Li, Mingyan Liu, Duane S. Boning, Cho-Jui Hsieh: Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations. NeurIPS 2020

### Questions
**Questions**
- Is counterfactuality well-defined?
- Why is counterfactuality important in Q functions, beyond conceptual links to neuroscience? Policy gradient methods, such as PPO, do not use the Q/Value function after training, so it would seem that counterfactuality is not meaningfully leveraged for its generalizability.
- Some more recent robust RL methods optimize other objectives, such as maximin value [2] and minimum regret [3], instead of value-optimal invariant actions. Can the conclusions in this paper be extended to have implications for those methods as well? 
- Theorem 3.3 proves the inherent tradeoff between accurate Q functions and robustness. If $\epsilon$-invariance is enforced in the policy objective only, is this a potential solution to the described problem?
- Also in Theorem 3.3, it is stated that the two functions $Q^*$ and $Q_\theta$ 'agree' on certain states. Is the agreement an equivalence of value, or is it referring to the ordinal rank of actions?

Lastly, a comment: Section 4.1 describes that the performance drop of vanilla policies is larger than that of robust policies for the worst-case action. It could be argued that the robust methods are succeeding in their objective here.

[2] Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang: Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning. NeurIPS 2022

[3] Roman Belaire, Arunesh Sinha, Pradeep Varakantham: On Minimizing Adversarial Counterfactual Error in Adversarial Reinforcement Learning. ICLR 2025

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper claims that $\varepsilon$-locally invariant ($\varepsilon$-LI) training—defined as keeping the optimal action unchanged for any perturbed state $\bar{s}$ inside an $\varepsilon$-ball—breaks RL’s inherent counterfactuality and yields misaligned or over-smoothed $Q$-functions.  
However, its reading of prior “observation-robust” RL (notably Huan et al., NeurIPS 2020) appears stronger than what that work actually proves.  
Huan formalizes a state-adversarial MDP with a perturbation set $B(s)$, analyzes value functions under an optimal adversary, and introduces regularizers or certificates that bound changes, not strict invariance of $\arg\max_a Q(s,a)$ for all $\bar{s}\in B(s)$.  
By building its theory on strict $\varepsilon$-invariance (argmax stability over the entire $\varepsilon$-ball), the paper’s counterfactuality-loss claims do not directly follow from Huan’s bounded-difference / certification framework.

### Strengths
- Raises a timely and interesting question: Can robustness training erase counterfactual structure?
- Clear toy-MDP reasoning (e.g., linear function arguments around $\varepsilon$-balls) clarifies a plausible accuracy–invariance tension.
- Provides a useful critique that encourages more precise definitions of “robustness” targets in RL (invariance vs bounded sensitivity).

### Weaknesses
- Over-interpreting prior work as $\varepsilon$-invariance.

The paper states that $\varepsilon$-LI methods “form the foundations of robust RL” and implies that Huan et al. (2020) learn $\varepsilon$-invariant policies.
However, Huan formalizes $B(s)$, analyzes SA-MDP values under an optimal adversary, and proposes regularizers/certificates that bound changes —not a guarantee that $\arg\max_a Q(s,a)$ is identical for all $\bar{s}\in B(s)$.

- Assumption-driven logical dependency.

The paper’s core theoretical framework (Def. 3.2—$\varepsilon$-LI and subsequent theorems) assumes strong argmax invariance.
Huan’s mathematics, however, deals with continuity/boundedness and partial certification (fractional guarantees).
Therefore, the paper’s counterfactuality-loss conclusion, which depends on full invariance, cannot be directly derived from the premises.

- Ambiguity in generalization scope.

The paper mainly targets the observation-robust line of research but does not rigorously distinguish whether the same “counterfactuality loss” applies to action-robust, transition-robust, or reward-robust settings.
Since each robustness category defines perturbation over different MDP components, it is unclear whether the claimed phenomenon generalizes.

### Questions
- Definition consistency (core issue).

The paper’s Definition ($\varepsilon$-LI: argmax invariance within the $\varepsilon$-ball) and Huan et al.’s bounded-change/certification results differ in strength.
Which prior methods actually satisfy your definition of $\varepsilon$-LI?
If none, wouldn’t it be more appropriate to describe your work as a limit-case analysis—a theoretical exploration of what would happen under strong invariance—rather than as a critique of existing algorithms?

- Replacement with bounded-difference assumption.

If the invariance condition were replaced by Huan’s bounded-difference assumption—e.g.,
$|Q(s,a)-Q(\bar{s},a)|$ or $D_{TV}(\pi(\cdot|s),\pi(\cdot|\bar{s}))$ being bounded—
how would your counterfactuality-loss theorem change?
Would the same qualitative result hold, or would the strength of the conclusion weaken?

- Applicability across robustness categories.

Beyond observation-robust settings, does the proposed “loss of counterfactuality” also occur in action-robust, transition probability-robust, or reward-robust RL algorithms?
Since the definition of robustness (what is being regularized or perturbed) varies across these classes, the effect on $Q$-ordering may also differ.
Could you clarify this distinction theoretically or empirically?

- Supplementary/Appendix clarification.

Supplementary material(L229) references Appendix, which is not included in the submission.
Instead of adding it to the supplementary, it may be clearer and more appropriate to include those contents directly as a Main Paper Appendix, so that key theoretical or experimental details are accessible without external files.

### Soundness
2

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
1

### Summary
This paper shows that enforcing ϵ-local invariance—a common technique to stabilize deep reinforcement learning policies—undermines the natural counterfactuality of value estimation, leading to inconsistent and misaligned policies. The authors argue this approach contradicts the biological principles underlying reinforcement learning, creating a gap between artificial and natural intelligence. Their findings urge a rethinking of regularization methods to preserve counterfactual reasoning while ensuring robustness.

### Strengths
I'm not an expert in this area and don't understand much about this work.

### Weaknesses
See Questions.

### Questions
- What is an action modification exactly?
- What is the definition of $\mathcal{P}_{w}$? What is $w$ in it?
- I don't understand Figure 4. What is the x-axis? What does the color represent exactly?
- What is the definition of counterfactuality under RL settings? How do you measure it exactly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors study recently developed methods for RL training that enforce ϵ-local invariance using theoretical and empirical analysis. They show that such methods have significant problems when compared to more traditional methods.

### Strengths
**Clear organization and writing**: While there are some improvements to be made (see below), the overall organization and writing of the paper is clear.

**Important topic**: This topic is clearly important. Safety, reliability, trustworthiness, and robustness have been a major theme of research in RL over the past several years, and this paper is an important re-examination of some of the consequences of current solutions.

**Both theoretical and experimental evidence**: The authors provide both theoretical and more concrete and realistic empirical evidence. This should be more strongly leveraged in the paper — for example, by more clearly identifying intuitions and mechanisms (see below) — but the theory and experiments are there to do that.

**Strong results**: The results, particularly in the experimental sections, are fairly strong. It’s clear that there are major differences between the training methods that enforce ϵ-local invariance and more traditional methods. Ultimately, this is what matters.

### Weaknesses
**Insufficiently explained impacts**: Many of the effects of enforcing ϵ-local invariance that are noted by the authors are sufficiently tied to impacts in both the abstract and introduction. For example, in the paper’s abstract, the authors say that such training “cause policies to lose counterfactuality”, “result in learning misaligned and inconsistent values”, “break the core intuition and the true biological inspiration of reinforcement learning”, and “introduce an intrinsic gap between how natural intelligence understands and interacts with an environment in contrast to AI agents”. These sound bad, but what are the impacts on performance? That is, under what circumstances and by how much does this training approach reduce concrete performance measures such as long-term reward across some set of environments? Note that the performance impacts are made clear in the experimental section, but that should be carried through into the abstract and introduction.

**Hyperbolic language**: The authors use hyperbolic language throughout the paper. For example, in the paragraph on contributions, the authors refer to “our comprehensive study” (when “our study” would be adequate), refer to “a theoretically well-founded rigorous analysis” (when “a theoretical analysis” would be adequate), and note that “ϵ-local invariance training shatters this elegant relationship” (when “ϵ-local invariance training disrupts this relationship” would be adequate). The authors should use more neutral language throughout the paper. 

**Lack of clear intuition and mechanisms**: The authors state high-level ideas in the introduction and elsewhere (e.g., “Our extensive analysis and results discover that ϵ-invariance training methods break the core intuitive principles of reinforcement learning…”), and they provide low-level theorems and experiments. However, the intuitions behind those theorems and the mechanisms to that lead to the experimental results are not made clear. The paper would be greatly improved by more mid-level intutions and mechanisms for how ϵ-local invariance affects policies (and, thus, performance).

**Overly broad conclusions**: The paper includes statements such as “Reinforcement learning has inherent counterfactual ability…” However, the paper clearly cannot provide evidence for *all* RL methods, so statements such as this should be limited to some set of methods. Additionally, the paper does not define “inherent counterfacual ability.” The conclusions should be stated more narrowly.

### Questions
The abstract notes that “…this line of [ϵ-local invariance] training methods break the core intuition and the true biological inspiration of reinforcement learning...” Why is this a problem? You outline many other problems, but I don’t see why these are a particular problem.

The introduction notes that “Our analysis reveals that reinforcement learning possesses an inherent ability for counterfactual reasoning and is naturally aligned with human decision-making processes…” Concretely, what does this mean?

### Soundness
3

### Presentation
3

### Contribution
4

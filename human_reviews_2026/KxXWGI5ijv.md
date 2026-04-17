# Learning To Acquire Resources in Competition

- Decision: Reject
- Scores: 6, 2, 6, 4

## Abstract
We consider multiple agents competing to acquire stakes in some costly divisible resource (e.g. shares of a financial asset, compute resources, or commodities) over time. We propose a novel game-theoretic model for this problem that generalizes settings studied in diverse literatures, and analyze it under different assumptions on agent information. Given complete-information, we establish the existence and uniqueness of a pure Nash equilibrium (NE) in this generalized setting. This is shown to be efficiently computable but has worst-case unbounded price of anarchy. Alternatively, under partial-information with a common prior, we establish the existence and uniqueness of a Bayesian Nash equilibrium (BNE), which is also efficiently computable. Finally, we propose a more realistic learning setting for the game, where agents have partial information but no common prior. Instead, they must learn how to act given online contextual feedback from interactions in stochastically sampled game instances. We provide sufficient conditions on agents doing simultaneous no-regret learning for convergence to Bayesian coarse-correlated equilibrium (BCCE) or last-iterate convergence to the BNE. In each setting, we provide detailed simulations, which empirically validates our theory and provides new insights into strategic behavior of resource acquisition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how multiple strategic agents acquire a costly divisible resource over time. It proposes a general discrete-time model with convex action set and concave idiosyncratic utilities, and a simple linear price dynamics with permanent and temporary impact. In the complete-information game there exists a unique pure NE, computable via a strongly-monotone VI with extragradient convergence; however the PoA is unbounded. In a Bayesian setting with finite type spaces and common prior, there is a unique pure BNE and can be obtained by extra-gradient algorithm. In a repeated game setting without a common prior, no-regret learning leads to an $epsilon$-BCCE and using the doubly-optimal OGD schedule gives last-iterate convergence to the approximate BNE under strong monotonicity.

### Strengths
1. This paper forms a general model with convex constraints, concave utilities and linear price dynamics with permanent and temporary impact. 
2. Prove the unique, efficiently computable NE and BNE via strongly monotone VIs and extra-gradient in the first two settings.
3. For the learning without priors case, show that the average convergence to an approximate BCCE under generic no-regret, and last-iterate convergence to an approximate BNE under AdaOGD.

### Weaknesses
1. Assuming access to full counterfactual cost or unbiased gradients is somehow unrealistic in markets. The last-iterate result requires all agents to use the specified OGD schedule and assumes strong monotonicity and access to unbiased gradient/cost feedback. It’s unclear how robust this is under misspecified $\alpha, \beta$.
2. Given PoA is unbounded, the paper would benefit from identifying conditions leading to a bounded PoA.
3. The linear price-dynamics model is fairly simple, and the main proofs rely on standard VI arguments for strong monotonicity and well-known no-regret learning results. The learning component largely instantiates existing theory in this setting.

### Questions
1. How sensitive are the results to the choices of $\alpha$ and $\beta$? In particular, how do estimation errors in these parameters influence the convergence rate and the validity of the uniqueness/monotonicity assumptions?
2. In realistic settings, agents observe realized prices, not full gradients. Can your results extend to bandit feedback setting?

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
4

### Summary
The paper introduces a model of resource acquisition games and analyse it under various information conditions. Under complete information, the paper shows existence and uniqueness of a pure Nash equilibrium which can be efficiently computed but has worst-case unbounded price of anarchy. When information is partial but agents share a common prior, there exists a unique Bayesian Nash equilibrium, which can also be compute efficiently. When no common prior is available the paper studies the case in which agents gather information while playing by following an online learning algorithm. The paper provides sufficient conditions for convergence to a Bayesian CCE and last-iterate convergence to the BNE.

### Strengths
The model studied in the paper is interesting and well motivated by practical applications in finance and other markets. The paper is generally well written and clear. Technical results appear to be correct. Experiments, despite being on very simple instances, hint at interesting behaviors that may open up new directions of research on this model.

### Weaknesses
My main concern is with the strength of the technical contributions. Once equipped with Lemma 1, Section 3 and 4 rely on fairly standard tools for these kind of problems. In particular, the positive results are kind of expected given the specific structure of the problem being considered. 

Section 5 is the one I find most interesting. However, Theorem 4 follows fairly standard ideas and is largely expected given similar results on convergence to BCCE in other games. Theorem 5  is largely based on proving that the problem being considered meets the requirements to apply the theorem by Jordan ed al. (2024).

Moreover, given the nice connection to practical applications, I would have appreciated a longer discussion about the feedback available to the online algorithm, and whether it makes sense in practice. From the discussion in paragraph starting at 359 this is not entirely clear. For instance, is assuming access to stochastic gradient feedback reasonable in practice? I would probably expect to have something closer to bandit feedback. Some discussion on this would be a useful addition.

Finally, simulations display some interesting behaviours (eg phase transitions) but they are all carried out on extremely simplified settings and it is difficult to extrapolate general insights from them. Extending the experimental analysis to richer synthetic or real-world settings would also be a nice addition to the current set of results. 

Typos: ““proprety” line 239; “stochstic” line 420

### Questions
See question on feedback above.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper analyzes a class of games involving agents attempting to acquire resources. The class of games is shown to be strongly monotone, which leads to a standard family of results including fast convergence of extra-gradient descent to Nash equilibria, among others.

### Strengths
The paper is well written, and the results are rather strong, basically completely answering the question of what happens with learning dynamics in this class of games.

### Weaknesses
This paper's main argument is essentially: "Here is an interesting class of games. They are strongly monotone (Theorems 1 and 3). Thus, applying known results for strongly monotone games verbatim, good things happen". The strong monotonicity results seem pretty straightforward from the definitions, so the strength of the paper boils down completely to understanding how important this class of games is. I do not feel well placed to evaluate that---I come from a computer science background, and I am admittedly relatively unfamiliar with the more economic side of this paper. Thus, I will not attempt to do so, in the hope that at least one other reviewer has a better understanding here. 

Minor note on the comment at L829 about CCE collapsing to NE in this class of games: I do not think that this follows from the preceding argument. It is not true that strict concavity of utilities implies that CCEs collapse to NEs. Instead, Theorem 1 (monotonicity of the associated VI operator) is what shows that CCE collapses to Nash---or, at least, that the marginals of any CCE form a Nash. (I did not carefully check the proof(s) of monotonicity, though it seems believable.) Perhaps the comment should be moved to after Theorem 1 instead.

### Questions
I have no specific questions, though I invite any comments on anything I said above.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper extends a model studied by previous work to study resource acquisition under competition. The paper extends the model of previous work by allowing utility functions of agents to be concave functions of the acquired resource quantity. The model also extends to include a non-strategic or exogenous agent.

With these extensions, the paper shows that the equilibrium of the game defined remains unique and the paper demonstrates how to compute this equilibrium. They also study a version of the game with incomplete information, where agents have uncertainty about types and game parameters, holding beliefs over them rather than knowing exact values. The paper shows that there is a unique Bayesian Nash equilibrium. Finally, the paper provides classes of algorithms that allow agents to converge to a Bayesian CCE or the Bayesian Nash equilibrium.

### Strengths
The paper studies various extensions of previous work and shows that there remains a unique equilibrium. Given uniqueness, the equilibrium seems like a plausible solution concept for this setting.

### Weaknesses
The extensions are not well-motivated. Given that the differences between this work and previous work are the extensions, it would be useful to provide more information about how the extensions are important for the motivating applications to understand why we should find the extensions important. Additionally, if the main contribution of the work is extending the setting of the game over theoretical contributions, I think there should be more justification on why the extensions are important. 

The novel technical contributions are not clear. At first glance, it appears like going from linear utilities to concave utilities are standard extensions that work out in many standard settings . It is unclear how much technical novelty there is in implementing these extensions. Likewise, in the section on learning dynamics for convergence to equilibrium, it seems like the results are applications of standard results on equilibrium convergence without any new technical contributions.

### Questions
Can you describe some of the technical challenges for your results compared to results of previous work? Can you describe the novel technical contributions your work makes?

### Soundness
3

### Presentation
2

### Contribution
2

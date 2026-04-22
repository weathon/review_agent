# Statistical Guarantees for Offline Domain Randomization

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Reinforcement-learning agents often struggle when deployed from simulation to the real-world. A dominant strategy for reducing the sim-to-real gap is domain randomization (DR) which trains the policy across many simulators produced by sampling dynamics parameters, but standard DR ignores offline data already available from the real system. We study offline domain randomization (ODR), which first fits a distribution over simulator parameters to an offline dataset. While a growing body of empirical work reports substantial gains with algorithms such as DROPO, the theoretical foundations of ODR remain largely unexplored. In this work, we cast ODR as a maximum-likelihood estimation over a parametric simulator family and provide statistical guarantees: under mild regularity and identifiability conditions, the estimator is weakly consistent (it converges in probability to the true dynamics as data grows), and it becomes strongly consistent (i.e., it converges almost surely to the true dynamics) when an additional uniform Lipschitz continuity assumption holds. We examine the practicality of these assumptions and outline relaxations that justify ODR’s applicability across a broader range of settings. Taken together, our results place ODR on a principled footing and clarify when offline data can soundly guide the choice of a randomization distribution for downstream offline RL.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper provides the first theoretical analysis of offline domain randomization (ODR), formulating it as maximum-likelihood estimation over simulator parameters and proving weak consistency and strong consistency under various regularity conditions. The work bridges the gap between empirical ODR methods like DROPO and theoretical understanding.

### Strengths
1. Novel theoretical contribution: First formal consistency guarantees for ODR, addressing an important gap between practice and theory

2. Practical considerations: Section 6 thoroughly examines when assumptions hold and offers relaxations (e.g., ergodic processes instead of i.i.d., weaker tail conditions)

3. Clear presentation: Problem setup is well-motivated and mathematical framework is clearly articulated

### Weaknesses
1. No empirical validation: The paper is purely theoretical with no experiments demonstrating when assumptions hold in practice or how consistency rates scale with data

### Questions
1. Do the assumptions (particularly 3 and 5) hold in standard benchmark environments (e.g., MuJoCo with mass/friction randomization)?

2. Can the framework extend beyond Gaussian to mixtures or other distribution families?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper provides a rigorous theoretical foundation for offline domain randomization (ODR), an increasingly important topic in bridging the sim-to-real gap in reinforcement learning. The authors frame ODR as a maximum-likelihood estimation problem over a parametric family of simulators and derive formal convergence guarantees. Specifically, they show that under mild regularity and identifiability assumptions, the estimator is weakly consistent, and under additional uniform Lipschitz continuity, it becomes strongly consistent, converging almost surely to the true dynamics as the amount of offline data increases. The proofs are technically solid and clearly structured, and the paper does a commendable job explaining not only the mathematical results but also their practical meaning.

### Strengths
The paper represents one of the first attempts to formally establish statistical guarantees for ODR, an area that has previously relied primarily on empirical evidence (e.g., algorithms such as DROPO). The theoretical treatment is rigorous yet well-motivated, and the authors are careful to analyze the realism of their assumptions, providing insightful discussions on how they could be relaxed to cover broader scenarios. This combination of solid mathematical grounding and practical reflection significantly strengthens the paper’s contribution. In particular, the convergence results provide valuable theoretical reassurance that incorporating real offline data into domain randomization is not only empirically beneficial but also statistically sound.

### Weaknesses
The analysis assumes that environment parameters are predefined, but in practice, it may be more realistic to start with a broader set of perturbable parameters and iteratively remove those with small variance as data accumulates. It would be helpful to discuss whether the current proofs would still hold, or require modification, under such an adaptive parameter-selection procedure. 

While the theoretical contribution stands well on its own, the paper could be strengthened by adding a few illustrative experiments (perhaps simple synthetic tests) that empirically confirm the convergence behavior or highlight the limits of the stated assumptions. Such additions would make the results more accessible to a wider audience beyond theory-focused researchers.

### Questions
Could the authors illustrate, even with a toy example, how the weak and strong consistency results manifest as offline data increases?

Given that many of the primary applications for ODR, such as robotics, are dominated by non-smooth dynamics (e.g., hard contact, stiction-friction) where Assumption 5 may not hold, could the authors elaborate on the theoretical implications of this gap?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper “Statistical Guarantees for Offline Domain Randomization” establishes a theoretical foundation for Offline Domain Randomization (ODR), a variant of Domain Randomization that leverages offline data to better align simulated and real-world dynamics. The authors formulate ODR as a maximum-likelihood estimation problem and derive statistical guarantees on its consistency, with the goal of improving sim-to-real transfer in Reinforcement Learning (RL).

### Strengths
1. By framing ODR as a maximum-likelihood estimation (MLE) problem, the paper elevates it from a purely empirical heuristic to a method with formal statistical grounding, establishing properties such as consistency.

2. The paper provides a clear exposition of its underlying assumptions, such as i.i.d. sampling, mixture positivity, and Lipschitz continuity, and thoughtfully discusses possible relaxations, which helps clarify the scope and applicability of the theoretical results.

### Weaknesses
1. The theoretical framework assumes that the true environment dynamics $𝑀^∗$ lie within a known parameterized simulator family 
{$𝑀_𝜉$}, and that a representative dataset of real-world transitions is available. In practice, however, the true parameterization is unknown, and it is rarely possible to guarantee that the simulator family adequately captures real-world behavior. This makes the theory elegant but largely non-operational in realistic settings.

2. The proposed ODR framework relies on access to real-world data to fit the randomization distribution, which partly contradicts the original motivation for offline RL, to minimize or avoid costly real-world data collection.

3. The paper does not include empirical results or analyses addressing robustness under model misspecification, i.e., when the true environment lies outside the assumed simulator family.

4. The work sidesteps the central open challenge in domain randomization: how to design or learn an appropriate simulator distribution when the true real-world distribution is unknown.

### Questions
1. How much offline data is needed to achieve meaningful consistency in practice? Are there any finite-sample bounds?
2. Can your theoretical framework explain why existing DR approaches (like DROPO or DROID) sometimes work even without theoretical consistency?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors present a rigorous theoretical analysis on the offline domain randomization (ODR) framework, which enables offline datasets of real-world data to be used for inference of simulation parameters and ultimately for training effective RL policies in simulation.
As current ODR methods are vastly empirical, the current manuscript studies the theoretical implications and assumptions of such setting, including assumptions on identifiability of dynamics parameters, and implications on when posterior distributions are guaranteed to converge to degenerate zero-variance point-estimates.

### Strengths
- The paper tackles a relevant framework in the field of sim-to-real transfer (e.g. ODR) that gained traction in recent years but lacked a thorough theoretical understanding. Such setting opens applications of domain randomization that are arguably safer and more sample efficient than uniform domain randomization methods.

### Weaknesses
- Restricting Gaussian assumption: recent empirical works further extend the ODR framework by considering normalizing flows or neural density estimators over dynamics parameters [1]. It's unclear how much the presented analysis is restricted to (1) simulators that follow a Gaussian parameter distribution and to (2) transition functions that are also assumed to be Gaussian.

[1] Muratore, Fabio, et al. "Neural posterior domain randomization." Conference on robot learning. PMLR, 2022.

### Questions
- How much of the derivations in this work are restricted by assumptions on simulator dynamics behaving as a parametric family of distributions? And what about assumptions on Gaussian distributions in particular?
- The original empirical works (e.g. DROPO) consider the case of unmodeled phenomena, i.e. where the parametric family of simulators may not perfectly match the entire dataset of offline transitions, hence maximizing the log-likelihood may not converge to such degenerate zero-variance distributions. Can the results on weak and strong consistency extend to such cases, or do the authors consider this setting?

### Soundness
2

### Presentation
3

### Contribution
2

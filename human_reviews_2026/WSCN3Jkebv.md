# Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 2

## Abstract
We propose a general framework for causal Imitation Learning (IL) with hidden confounders, which subsumes several existing settings. Our framework accounts for two types of hidden confounders: (a) variables observed by the expert but not by the imitator, and (b) confounding noise hidden from both. By leveraging trajectory histories as instruments, we reformulate causal IL in our framework into a Conditional Moment Restriction (CMR) problem. We propose DML-IL, an algorithm that solves this CMR problem via instrumental variable regression, and upper bound its imitation gap. Empirical evaluation on continuous state-action environments, including Mujoco tasks, demonstrates that DML-IL outperforms existing causal IL baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a causal imitation learning (IL) framework that explicitly models two forms of hidden confounding. It leverages trajectory histories as instruments to convert confounded IL into a conditional moment restriction (CMR) problem and introduces DML-IL, an instrumental-variables–based algorithm to solve it.  The authors establish an upper bound on the imitation gap that recovers earlier results as special cases, and demonstrate in continuous-control benchmarks (including MuJoCo) that DML-IL consistently outperforms existing causal IL baselines.

### Strengths
- some important assumptions are clearly stated
- some motivating examples (e.g., Example 3.1) are provided
- an expansion of [3], with a longer confounding horizon $k > 1$

### Weaknesses
- Some important relevant papers are not included (e.g., [1]).
- Claims outpace evidence. The "unifying" scope is asserted more strongly than it is demonstrated.
- Some assumptions are hard to be verified in practice

Specific questions below.

### Questions
## Identifiability under missing expert covariates

- Your formulation assumes that the expert’s additional covariates $u_{t}^{o}$ are never directly observed in demonstrations (only indirectly via histories). Can you clarify whether your identification or bounds require any minimal observability or recoverability conditions on $u_{t}^{o}$?
- [1] studies the scenario where exact imitation is impossible. How does your analysis relate to partial-identification results showing impossibility from demonstrations when state/action/reward are jointly confounded [1]? You note that without assumptions the expert policy is unidentifiable; please delineate precisely which of your assumptions (e.g., Additive Noise, Confounding Noise Horizon) are sufficient to escape [1]’s impossibility and where residual partial identification remains. It might be better if the authors could add such discussions comparing with [1] in their paper.

## Confounding noise horizon

- Algorithm 1 requires k as input. However, in practice, how do we know k beforehand? Do we need to run causal discovery algorithms to obtain k? Additionally, can you provide a data-driven procedure used in your experiments, report sensitivity to $k$ or $\bar{k}$, and include overidentification or CI-based diagnostics per environment?
- In the worst case k=T−1, the instrument is the entire past. With continuous, high-dimensional states, both statistical and computational complexity can scale poorly. In that case, the scenario is similar to POMDP. Please characterize how sample complexity and runtime of DML-IL scale with k, T and state/action dimensionality.
- Since the validity $h_{t-k} \perp u_{t}^{\varepsilon}$ is very hard to be verified, how do the proof steps change if the instrument is slightly invalid (e.g., $\delta_{\mathrm{IV}}$ violation)? Robustness version of Theorem 4.5?
- Is $\delta$ treated as a structural constant, or can it be bounded/estimated from data (e.g., via proxy tests) ? Can you provide an example showing how large $\delta$ can be in typical MuJoCo settings?

## Additive-noise assumption

> Actions are generated via $a_{t}=\pi_{E}\left(s_{t}, u_{t}^{o}\right)+u_{t}^{\varepsilon}$.

-  What if in general like $a_{t}=\pi_{E}\left(s_{t}, u_{t}^{o}, u_{t}^{\varepsilon}\right)$ non-additive cases [1][2], how could you deal with them?
- Could you bound the bias of DML-IL under structured departures from additivity (e.g., small multiplicative terms) or provide a sensitivity analysis (e.g., "plausibly exogenous") approaches?

## Experiments

> we generate 40 expert trajectories, each of 500 steps, following our previously defined environments. 

- Why 40×500 specifically? Please report sensitivity to the number/length of demos.

- K-fold details. You mention K-fold cross-fitting and give Algorithm 2. What K did you use in each experiment? Were folds stratified by environment/noise setting? How sensitive are results to K (1, 2, 5, 10)? Please include means ± CIs across seeds.

- Code and seeds. Please release code, random seeds, and environment configs (including the exact noise processes) so others can reproduce the roll-out model training and cross-fitting setup described in Appendix

## Scope

Finally, I strongly recommend the authors to change their claim about "**Unifying Framework**" for Causal Imitation Learning. There are lots of cases that cannot be covered by your framework (i.e., the current scope excludes several practically common regimes), e.g., violations of the additive-noise assumption (which is quite common), stochastic experts, sub-optimal experts, invalid history instruments, unknown or unbounded confounding horizons, partial observability, different causal diagrams and so on. These are important and active research fronts; acknowledging them explicitly—ideally with a short taxonomy of covered vs. uncovered regimes and a softened headline claim —would make the paper’s contributions more precise and credible while pointing concretely to valuable directions for future work.

I would be _inclined to raise my score_ if the revision clarifies scope as suggested and properly answers my questions above.

-----


[1] "Causal imitation for markov decision processes: A partial identification approach." Advances in neural information processing systems 37 (2024): 87592-87620.

[2] "Causal imitation learning via inverse reinforcement learning." The Eleventh International Conference on Learning Representations. 2023.

[3] "Causal imitation learning under temporally correlated noise." International Conference on Machine Learning. PMLR, 2022.

[4] "Invariant causal imitation learning for generalizable policies." Advances in Neural Information Processing Systems 34 (2021): 3952-3964.

### Soundness
2

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
This paper proposes a unifying framework for causal imitation learning in the presence of hidden confounders, distinguishing between two types: variables observed by the expert but not the imitator (expert-observable), and noise hidden from both parties (expert-unobservable). By leveraging trajectory histories as instrumental variables and assuming a bounded confounding noise horizon along with additive noise structure, the authors reformulate the imitation problem as a Conditional Moment Restriction (CMR). They introduce DML-IL, a novel algorithm based on double machine learning to solve this CMR, and provide a theoretical upper bound on the imitation gap that generalizes prior results. Empirical evaluations on both a custom airline pricing task and modified MuJoCo environments demonstrate that DML-IL outperforms existing causal imitation baselines when both types of confounders are present.

### Strengths
The paper is clearly structured and presents a compelling case for a more holistic approach to addressing hidden confounding in imitation learning. The proposed framework elegantly subsumes several prior settings, and the experimental results convincingly validate the method’s effectiveness across diverse environments. The theoretical analysis is rigorous and meaningfully connects to established results in causal inference and econometrics.

### Weaknesses
- The framework relies on two key assumptions that may limit its applicability in certain domains. First, the additive noise assumption (i.e., that unobservable confounders affect actions additively) is essential for identifiability but may be overly restrictive in complex systems where interactions are nonlinear or multiplicative (e.g., biological or financial systems). While this assumption is standard in instrumental variable literature, its validity must be carefully assessed per application.

- The method requires knowledge (or a valid upper bound) of the confounding noise horizon k —the temporal range over which unobservable noises remain correlated. Although the paper discusses practical heuristics and conditional independence tests to estimate k, this parameter is not identifiable from data alone, and misspecification (especially severe underestimation) can degrade performance, as shown in the ablation studies.

### Questions
- How robust is the proposed method against model misspecification, particularly with respect to the noise horizon at Assumption 3.2?

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
This paper proposes a unifying framework for causal imitation learning (IL) with hidden confounders, distinguishing between expert-observable confounders (variables observed by the expert but not the imitator) and expert-unobservable confounders (confounding noise hidden from both). By leveraging trajectory histories as instrumental variables, the authors reformulate causal IL as a Conditional Moment Restriction (CMR) problem and propose DML-IL, an algorithm that solves this CMR via instrumental variable regression. They provide an upper bound on the imitation gap and demonstrate empirically on continuous state-action environments that DML-IL outperforms existing causal IL baselines.

### Strengths
- Originality: The distinction between expert-observable and expert-unobservable confounders provides a novel and more realistic problem formulation that generalizes several prior works. The neat reframing of causal IL into instrumental variables and CMR problems is elegant and theoretically well-motivated.
- Quality: The theoretical analysis is solid, with the neat theory on the imitation gap bound (Theorem 4.5) that recovers prior results as special cases (Corollaries 4.6 and 4.7).
- Clarity: The paper is generally well-written with clear motivation through Example 3.1 and good use of figures (especially Figure 1). The connection between hidden confounders and the CMR formulation is explained systematically.
- Significance: The unifying framework addresses a realistic scenario where multiple confounding factors coexist, which is practically important and has been overlooked in prior work that typically addresses confounders in isolation.

### Weaknesses
- Limited scope of unification: The framework doesn't truly subsume Vuorio et al. (2022), which solves the problem through environment interaction. The experimental results don't include comparisons to interactive scenarios, limiting the claim of being a "unifying" framework. The paper focuses exclusively on offline IL from fixed demonstrations.
- Gap between theory and experiments: No analysis is provided comparing experimental results to the theoretical bounds from Theorem 4.5, making it unclear how tight these bounds are in practice.
- Algorithm derivation unclear: The connection between the CMR error minimization and Algorithm 1's two-stage procedure needs better motivation. The paragraph below Algorithm 1 explaining why training on expert trajectories does not work, is difficult to follow.
- Restrictive assumptions: The framework requires knowledge of (or an upper bound on) the confounding noise horizon k, which cannot be verified empirically. The additive noise assumption (Assumption 3.3) is quite strong and limits applicability. The authors don't fully discuss the effect of these limitations.

Minor:
- In Definition 4.2, the text describes "the ratio between the learning error... and its L2 error" but the equation shows the reciprocal of this ratio (L2 error in numerator, learning error in denominator).

### Questions
- How do the experimental results quantitatively compare to the theoretical bounds from Theorem 4.5? Can the authors provide concrete values for the variables in the theorem in the experimental settings to assess bound tightness?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for causal imitation learning with hidden confounders that distinguishes two types of latent confounders: (i) expert-observable confounders (visible to the expert but not to the imitator) and (ii) expert-unobservable confounders (hidden to both). Using trajectory histories as instrumental variables, the approach reduces imitation learning to a conditional moment restriction problem. They prove an imitation-gap upper bound that recovers prior bounds as special cases, and provide empirical validation on a toy pricing environment and modified MuJoCo tasks where their algorithm outperforms some baselines.

### Strengths
- Casting imitation under confounding as a CMR with trajectories as instruments is conceptually simple and lets the authors leverage existing IV/CMR machinery.

- The paper shows corollaries recovering Swamy et al. results as special cases.

### Weaknesses
A confusing point of this paper is the arrow from $u_\epsilon$ to $a$, which means that even the expert does not have full control over its actions--the expert might choose to take one action, but another action take place in the end because of $u_\epsilon$. I did not get how this is justified.

---
>"We assume the confounding noise is additive to the action, which is standard in causal inference. Without this assumption, the causal effect becomes unidentifiable."

I do not think that this is really 'standard'. The claim of unidenitifiability in case it is not assumed also requires more elaboration, as the setting of this paper is not exactly that of Balke and Pearl 1994. Most importantly, this assumption, as well as the first equation on page 5 at the beginning of Section 4 only make sense if the action $a_t$ is continuous, which is a point that needs be clarified.

---
It is not clear how Assumption 3.2 implies what is being used in the derivations afterwards. There is a missing step in the proofs which doesn't sound straightforward to me. Unless I'm missing something, there is an implicit assumption made that is not mentioned in the paper. See my question 2 below.

---
Algorithm 1 is adapted from Shao et al, 2024, but with the current form of adaptation, it is not clear whether or not convergence guarantees transfer from that paper. See my question 1.

---
There are a few symbols and notions that are used on the go without being defined at all, which makes following the paper difficult. For example, policy $\pi_h$ is defined as a function from $\mathcal{H}$ to $\Delta(A)$, but $\Delta()$ is never defined. So it is not even clear what we are looking for.
Same goes for dim(A) in Corollary 4.7 which is never defined, although it is guessable unlike $\Delta(A)$.

---
The plots of Figures 2 and 3 are too small, and the font size is so small that the text is almost not readable. 

---
Minor:

Figure 1: dotted -> dashed, last paragraph of page 9: one limitation are -> one limitation is

### Questions
1. Could you please clarify why algorithm 1 is a double machine learning algorithm? What are the nuisance parameters? How is the Neyman orthogonality guaranteed? It is simply mentioned that under "regularity assumptions" the estimator converges at root-n rate. I do not think achieving root-n convergence (especially without cross-fitting) is as simple as that, perhaps Donsker-type conditions are meant as "regularity conditions?" I believe the authors should at least name the assumptions for convergence, and with the Gaussian mixture roll-out, a more careful treatment is required.

---
2. When proving Eq. (4), under the derivation of CMR problem on page 5, the authors use $u^\epsilon_t \perp\\!\\!\\!\perp h_{t-k}$, whereas the assumption they have made is $u^\epsilon_t \perp\\!\\!\\!\perp u^\epsilon_{t-k}$. Could you elaborate on how this assumption implies $u^\epsilon_t \perp\\!\\!\\!\perp h_{t-k}$? It does not sound trivial to me.

---
3. Example 3.1 is meant to make the paper clearer, but it sort of adds to the confusion. What is the action set here? Does the agent set the price or does it set the profit margin? If it sets the price, then the example is confusing and does not fit into the paper. If it sets the profit margin, then I find the example unrealistic as this would mean that the airline can set the profit margin but not the price?

### Soundness
2

### Presentation
3

### Contribution
2

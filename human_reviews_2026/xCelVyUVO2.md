# The Sample Complexity of Online Reinforcement Learning: A Multi-model Perspective

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
We study the sample complexity of online reinforcement learning in the general non-episodic setting of nonlinear dynamical systems with continuous state and action spaces. Our analysis accommodates a large class of dynamical systems ranging from a finite set of nonlinear candidate models to models with bounded and Lipschitz continuous dynamics, to systems that are parametrized by a compact and real-valued set of parameters. In the most general setting, our algorithm achieves a policy regret of $\mathcal{O}(N \epsilon^2 + d_\mathrm{u}\mathrm{ln}(m(\epsilon))/\epsilon^2)$, where $N$ is the time horizon, $\epsilon$ is a user-specified discretization width, $d_\mathrm{u}$ the input dimension, and $m(\epsilon)$ measures the complexity of the function class under consideration via its packing number. In the special case where the dynamics are parametrized by a compact and real-valued set of parameters (such as neural networks, transformers, etc.), we prove a policy regret of $\mathcal{O}(\sqrt{d_\mathrm{u}N p})$, where $p$ denotes the number of parameters, recovering earlier sample-complexity results that were derived for *linear* *time-invariant* dynamical systems. While this article focuses on characterizing sample complexity, the proposed algorithms are likely to be useful in practice, due to their simplicity, their ability to incorporate prior knowledge, and their benign transient behaviors.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a theoretical study of the problem of online reinforcement learning in general (non-linear) continuous systems from the perspective of sample complexity. For context, this version of RL is the non-episodic setting which arose from generalising multi-armed bandits: one competes via regret with the best policy according to the ergodic/infinite horizon undiscounted cost functional (see Thm. 2.1-3). 

The algorithmic solution here is a model-based exponential weights method, which applies to both finite and compact sets of models, which is appreciable. In terms of analysis, classical covering arguments on concentration inequalities of least-squares estimation are replaced by packing arguments for model classes in an expected regret analysis. The results consist in three bounds, in classical manner: the first is a gap-dependent bound, the second is scale dependent bound which depends on $\epsilon>0$ and the packing number of the model class at scale $\epsilon$, and finally a worst-case bound. 

In the interest of full disclosure, I had the pleasure of reviewing this article before at ICML and EWRL, which allowed the authors to respond directly to some of my comments and questions and address the weaknesses I identified.

### Strengths
The theorems are clear and well-stated, the proofs are clear, and I haven’t found any issues with them in a superficial inspection. Technical claims are made and proven in a sound manner. The discussion of related works has grown and grown over my several encounters with the paper, and its depth and clarity really deserve commendation now!

I think the contributions of the separation principle approach are deserving of dissemination, and while it is still difficult to situate the assumptions of different lines of work in online RL relative to each other,  the new set of assumptions should be of interest to others in the field.

### Weaknesses
As said above, my main criticisms of the paper have already been addressed by the authors in previous revisions. 

One minor weakness which remains is that the writing has some strange repetitions that, I think, could be avoided. For instance, there are 3 persistent excitation conditions that vary slightly (and similarly for the Bellman super-solution assumption). Combining all three into a single assumption, e.g. by letting the constant take one of three values depending on the setting, would probably simplify the exposition of the appendices.

### Questions
N/A

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
3

### Summary
The paper studies online reinforcement learning for non-episodic, continuous-state, continuous-action dynamical systems when the true dynamics are only known to lie in a rich model class. It proposes a family of posterior-sampling / Hedge-style algorithms that (i) keep a running, normalized one-step prediction loss for each candidate model, (ii) periodically sample a model according to a softmax over these losses, (iii) apply the corresponding certainty-equivalent controller, and (iv) inject excitation to guarantee persistence of excitation. The analysis is given for three settings: (S1) a finite set of nonlinear candidate models, where the frequentist policy regret scales as $O((\ln N + \ln m)/\Delta)$; (S2) a bounded (possibly infinite) class of Lipschitz dynamical systems, controlled via an $\varepsilon$-packing and yielding regret $O(N\varepsilon^{2} + (\ln N + \ln m(\varepsilon))/\varepsilon^{2})$; and (S3) a compact $p$-dimensional parametric family (e.g. neural networks), where the regret becomes $O(\sqrt{Np})$, recovering LQR-type rates as a special case. Conceptually, the work cleanly separates model identification from certainty-equivalent control, shows that simple excitation suffices to obtain nonasymptotic, frequentist guarantees, and unifies several strands of RL, online learning, and adaptive/multi-model control within a single sample-complexity framework.

### Strengths
1. The paper unifies three increasingly general control/learning regimes with one posterior-sampling–plus–Hedge template. It starts from a finite candidate set $\{f_1,\dots,f_m\}$, where the frequentist policy regret is of order $O((\ln N + \ln m)/\Delta)$, so the dependence on $m$ is logarithmic as in online learning. It then lifts this to an infinite/bounded function class by constructing an $\varepsilon$-packing and obtains regret of the form $N\varepsilon^2 + (\ln N + \ln m(\varepsilon))/\varepsilon^2$, which is the standard approximation vs. estimation tradeoff. Finally, for a compact $p$-dimensional parametric family it shows regret $(c_{r1}\ln N + c_{r2}p)\sqrt{N}$, recovering $O(\sqrt{Np})$ for linear/LQR while still covering nonlinear dynamics. This gives a nonasymptotic, frequentist guarantee that is stronger than Bayesian-average PSRL bounds in related work. 
2. The algorithm is elegant: each round draws a model using a Hedge/posterior update, runs the corresponding certainty-equivalent policy, and injects excitation to ensure persistence-of-excitation and fast posterior concentration. This realizes a practical separation between model identification and control, avoids heavy OFU-style planning in continuous spaces, and naturally incorporates prior knowledge through the candidate set or parameter prior. 
3. The algorithm comes with solid theory. The paper gives three clear regimes of guarantees. For a finite set of models, the policy regret is $O((\ln N+\ln m)/\Delta)$. For a Lipschitz class controlled by an $\varepsilon$-packing, it becomes $O(N\varepsilon^2+\ln m(\varepsilon)/\varepsilon^2)$. For a $p$-dimensional parametric family, it is $O(\sqrt{Np})$, matching LQR/adaptive-control rates. The bounds can be compared and tuned via $\varepsilon$.

### Weaknesses
1. The analysis is essentially realizability-based: in all three settings (S1 with a finite set of candidates, S2 with an $\varepsilon$-packed class, and S3 with a parametrized family) the true dynamics $f$ is assumed to lie in the modeling class $F$. In S1, Theorem 2.1 yields a policy-regret bound of order $O((\ln N + \ln m)/\Delta)$, but this relies on a separation margin $\Delta>0$ between the candidate models so that suboptimal ones can be eliminated; when the models are nearly indistinguishable the paper itself points out that one has to revert to an $O(\sqrt{N \ln m})$-type rate. For S2 and S3, the proposed procedures (Alg. 2 and Alg. 3) inject Gaussian excitation $n_{u,k} \sim \mathcal N(0,\sigma_u^2 I)$ every $M$ steps to enforce the persistence-of-excitation requirement (Assumption 6) along the closed-loop trajectories. This condition is stated to hold uniformly over all misspecified models in the class, which makes the theory clean but may be nontrivial to verify or enforce on an actual control system.
2. For the general setting (S2), the paper first reduces the infinite dynamics class to a finite one by taking an $\varepsilon$-packing and then applies the same multi-model scheme as in S1. This leads to the regret bound $O(N\varepsilon^2 + \ln m(\varepsilon)/\varepsilon^2)$ in Theorem 2.2. However, using this result in practice implicitly requires (i) access to or construction of such an $\varepsilon$-packing of the dynamics class, and (ii) the ability to run a controller for every element in the resulting finite cover. The paper also acknowledges that for Lipschitz-bounded dynamics the packing number $m(\varepsilon)$ can grow very quickly in high dimension, so this part should be read more as an information-theoretic learnability guarantee than as a directly deployable method for high-dimensional continuous systems.

### Questions
My concerns are already detailed in the cons section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the sample complexity of online RL in non-episodic settings with continuous state and action spaces. The authors develop a unified algorithmic framework that combines posterior sampling, Hedge-style model weighting, and certainty-equivalent control under persistent excitation. They provide theoretical policy-regret guarantees across three regimes:
- finite model classes $O((\ln N + \ln m)/\Delta)$, 
- general function classes via $\varepsilon$-packing ($O(N\varepsilon^2 + \ln m(\varepsilon)/\varepsilon^2)$),
- parameterized model families $O(\sqrt{Np})$. 

The analysis explicitly incorporates model identifiability ($\Delta$), information complexity (packing number), and structural dimension ($p$), and demonstrates that the bounds are tight up to logarithmic factors. The work also relaxes the classical PE condition by connecting it to controllability Gramians and sub-Gaussian stability assumptions.

### Strengths
1. The discussion of related work is exceptionally clear, and the citations appear comprehensive.
2. The theoretical analysis is rigorous.
3. The paper is well organized, and the narrative progresses with a coherent, reader-friendly logic.

### Weaknesses
While I do not see any glaring flaws, the following points prevent a stronger recommendation:
1. Under the stated assumptions, the theoretical guarantees are not particularly surprising. Despite the authors’ thorough comparison with prior work, the contribution seems incremental relative to the papers referenced around line 67 of the manuscript.
2. The problem setting is rather restricted, and its practical value is uncertain. The paper provides only simple numerical examples in the appendix, leaving real-world applicability unclear. In many practical scenarios, estimating $\mu_\theta$ is nontrivial, and the cardinality $|F|$ may grow exponentially with the size of the state space, which is not encouraging.

### Questions
1. In the weakly identifiable limit ($\Delta \to 0$), can one obtain a smoother adaptive transition between the $(\ln N + \ln m)/\Delta$ and $\sqrt{N\ln m}$ regimes, possibly through hierarchical discretization or adaptive model aggregation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses theoretical guarantees for non-episodic reinforcement learning in Euclidean state and action spaces with general nonlinear dynamics.
The true dynamics are the sum of a deterministic function and sub-Gaussian noise.
The learner is given a class $F$ of dynamics models and with each dynamics model $f^i \in F$, a corresponding deterministic policy $\mu^i$, that may or may not be optimal.
Realizability is assumed: the true dynamics $f$ belong to $F$.
Three types of model-class are studied:

- S1: A finite set of Lipschitz functions.
- S2: A bounded set in a normed function space.
- S3: A parameterized family with a compact Euclidean parameter space.

These are general enough to subsume lots of prior work.
The performance criterion is policy regret against the $\mu$ associated with the true $f$.

At a high level, the algorithms look like Hedge over the space of candidate models (i.e. possibly a continuum),
where the "loss sequence" for each candidate model is given by a sum of normalized one-step squared prediction errors.
However, one cannot use the standard Hedge analysis for this problem, because the prediction errors depend on the visited state sequence, which depends on the policies that have been deployed by the learner since the start.

To deal with the latter issue, the algorithm imposes its own episodes and holds the policy constant within each episode.
The algorithm adds white Gaussian noise to the policy's deterministic actions.
THe main challenge is to control the state magnitude while suboptimal policies are in use, and to ensure that the noise and episode length are exciting/exploratory enough to correctly evaluate the predictive accuracy of each candidate.

The central analysis is for the finite case S1.
The assumptions include:

- a technical Assumption 1 about a particular variant of cost-to-go function.
- typical Lipschitz/smoothness on the policies and stage costs.
- the cost-to-go has a quadratic lower bound.
- a "gap" lower bound on the (normalized) difference in predictions between the true $f$ and all other candidates $f_i \in F$.
  This bound appears to depend on both the properties of the class $F$ (no two candidates are too similar)
  and the ability of the additive noise in the action space to sufficiently excite those differences.

The authors prove sublinear policy regret for each of the three settings, although in S2 the regret exponent gets very close to $1$ for high-dimensional states and/or actions.
In particular, the results from S3 recover some regret bounds from the literature for linear dynamical systems, i.e. much more restrictive cases.

The analysis shows that the algorithm identifies the correct candidate model in finite time almost surely.
This, in turn, is used for the regret bound.

The extension to S2 is not computationally tractable, but gives statistical results.
It is assumed we can identify the function $\bar f$ that minimizes the "Hedge loss sequence" over the entire $F$.
Then, we construct finite cover of $F$ around $\bar f$; select from the cover using the Hedge-like rule; and synthesize the corresponding policy.

The extension to S3 assumes that we can somehow sample from the Hedge-like probability measure over the parameter space directly.
The authors claim a motivation from neural networks, where this is (to my knowledge) wishful thinking.
However, they note that it is tractable for feature-space models of the form $f(x,u;\theta) = \phi(x, u)^\top \theta$ which are widely studied in RL theory.
The synthesis of the corresponding policy is still computationally difficult except in special cases.

### Strengths
The paper studies non-episodic RL for a class of nonlinear dynamical systems that is more general than lots of related work.
This problem is clearly on the frontier of RL theory.
The proposed techniques use a nice mixture of online learning theory, Bayesian methods, and nonlinear control theory.
The paper should be of interest to researchers with both RL and control backgrounds.

The main assumptions (besides ignoring computational cost) are Assumption 1, which is related to Bellman optimality and dissipativity, and Assumption 3, which is related to exploration/persistent excitation. To be honest, it is hard for me to confidently say whether or not these assumptions are restrictive. It seems that we have somehow ruled out hard exploration problems, since we are able to get the regret bound with an exploration strategy that is naive compared to those required even for tabular MDPs in the worst case. However, since the setting is single-trajectory, it is clear that *some* kind of assumptions to limit the negative impact of disturbances and control the difficulty of exploration are necessary.

The framework mostly ignores computational issues and focuses on statistical guarantees, but this is standard in learning theory. The settings S2 and S3 (except for the special linear-in-features case) are possibly too general to admit efficient algorithms. It will be interesting to see if any follow-up work can instantiate those algorithms for other special cases.

The paper supports, along with other recent work, the overall idea that "certainty equivalence" is a good approach for RL. This is a positive result that simplifies our analysis of RL problems.

I did not have time to check the proofs, but the overall proof structure is logical, and the techniques used seem appropriate.

Overall, I think this is a strong contribution to the RL and learning-based control research communities.

### Weaknesses
In the Theorem 1 statement, it is a bit confusing to see the equation (1) suddenly called "$\mathcal{H}_2$ gain", maybe it is equivalent to the classic $\mathcal{H}_2$ gain for linear dynamics and $l(\cdot,\cdot)$ quadratic, but this version is still unfamiliar and RL audiences definitely won't know it. In general, the paper seems to assume a level of familiarity with classic control theory that the ICLR audience may not possess; it would improve the paper to do a bit more hand-holding.

Recovering near-optimal regret bounds for linear systems is very nice, but there is a big gap of generality between the proposed work and linear systems.
It would be interesting to know if the framework is also capable of recovering regret bounds for more general frameworks like bilinear classes, Bellman eluder dimension, etc.

Deferring detailed related work discussion to the appendix is unusual. I suggest to move more of the most closely related work on RL theory for continuous state+action spaces with nonlinear dynamics to the main body.

I suggest to reallocate space in the main body -- the S2 case already has its algorithm pushed to the appendix, so one must flip back and forth while reading about the packing strategy in Section 3.2 -- I suggest to shorten the discussion of S2 in the main body and use that space to give a bit more detail/intuition on S1 and the related work in the main body.

The authors discuss the limitations candidly throughout the paper, so it was surprising to see the conclusion without any thoughts on how they might be improved in future work.

### Questions
In Equation (1), must $\gamma^i$ be finite for all $i$?

The discussion of Assumption 1 could use more intuition. The authors discuss technical details on how the $-\gamma$ and $-d_u L_u \sigma_u^2$ terms help adapt the Bellman-like condition to deal with the infinite-horizon average cost objective and the extra price of excitation. However, I am still left unsure about: **what kind of systems/policies have we ruled out by making this assumption**?

Theorem 3.2 is very similar to Theorem 2.1, but not exactly the same. Why can't we have a proof sketch of Theorem 2.1 in this section instead?

One notable related line of work is the Decision-Estimation Coefficient [1, for example]. It also provides a highly general RL theory framework, uses the model-based RL paradigm, and focuses exclusively on the statistical aspect (not computational). Although (to my knowledge) that work is more abstract and further from practical, and it is an episodic setting, but it seems too closely related to skip. How does this work compare?

[1] Dylan J. Foster, Noah Golowich, Yanjun Han. Tight Guarantees for Interactive Decision Making with the Decision-Estimation Coefficient. COLT 2023.

### Soundness
4

### Presentation
3

### Contribution
3

# Optimal Stopping for Sequential Bayesian Experimental Design

- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
In sequential Bayesian experimental design, the number of experiments is usually fixed in advance. In practice, however, campaigns may terminate early, raising the fundamental question: when should one stop? Threshold-based rules are simple to implement but inherently myopic, as they trigger termination based on a fixed criterion while ignoring the expected future information gain that additional experiments might provide. We develop a principled Bayesian framework for optimal stopping in sequential experimental design, formulated as a Markov decision process where stopping and design policies are jointly optimized. We prove that the optimal rule is to stop precisely when the immediate terminal reward outweighs the expected continuation value. To learn such policies, we introduce a policy gradient method, but show that naïve joint optimization suffers from circular dependencies that destabilize training. We resolve this with a curriculum learning strategy that gradually transitions from forced continuation to adaptive stopping. Numerical studies on a linear-Gaussian benchmark and a contaminant source detection problem demonstrate that curriculum learning achieves stable convergence and outperforms vanilla methods, particularly in settings with strong sequential dependencies.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new approach to an understudied problem in Bayesian optimal experimental design (BOED) in the explicit likelihood setting that utilizes reinforcement learning (RL) policy-based BOED as opposed to variational inference (VI), non-RL policy-based, or maximum likelihood estimation of experimental designs and a critic (posterior) of interest. The new approach to early-stopping takes a simple rule of adding a constant loss to the "reward" in addition to the standard Expected Information Gain (EIG) in BOED. Results in two toy models of BOED are promising but lack evaluation in settings where the design isn't a constant cost.

### Strengths
The paper takes a good first approach to the optimal stopping problem in BOED through the lens of a MDP. I enjoyed reading about the connection between optimal stopping in the RL Bellman sense and optimal stopping in the setting of a constant cost. Allusion to some works in stochastic processes is mentioned but it seems RL is the most influential method, whereas I think more discussion of relation to stochastic processes would be beneficial. The authors also clearly state the intuition about how their framework works. They evaluate some of the most-critical hyperparameters of their method, demonstrating convergence to a known optimal stopping point for an analytic linear example.

### Weaknesses
The paper feels rushed. While the authors are addressing an interesting question in a narrow aspect of BOED, they drift into a pure RL formulation that sweeps more holistic evaluation of their method under a rug. BOED is based on the Bayesian principle of updating the current belief state of the posterior to prior. Their reward shares a similar achilles heel as other BOED methods in lacking evaluation of either the calibration of the resulting posterior or some form of validation metric outside of the EIG, which can be gamed. Omitting this type of evaluation makes me less confident in the results of the paper.

With respect to the method, I find it odd and arbitrary that the cost evaluated in the main paper is just a constant rather than a design-dependent value. For example, many real-world design depend on the experiment, such as the dollar amount of investing at a certain time. A constant seems to only penalize the time at which the experiment was performed. The paper partially addresses this in Figure 8 in the appendix but further analysis is omitted.

Additionally, the paper evaluates on only the source-finding problem in BOED and a trivial linear gaussian benchmark. Related to my previous critique, applying optimal stopping to a non-trivial problem that is design-dependent would demonstrate the utility of the paper's contribution.

Some of the exposition could be more clear, such as defining what the incremental utility $U_I$ are in Theorem 2 and tradeoffs between choosing it and $U_T$. Also, it'd help reading clarity to include those standalone symbols instead of mentioning them by name earlier then by symbol in the theorem.

I also have the following stand-alone critiques.
- Need to cite Kleinegesse et al. 2021 paper "Gradient-based Bayesian Experimental Design for Implicit Models using Mutual Information Lower Bounds" on value of future data points
- addictive -> additive at line 125, as well as other grammatical issues and typos throughout the paper
- It's not clear how the MDP formulation is unified in line 200. This seems like a broad and out-of-context statement.
- Simulation efficiency isn't discussed. According to Algorithm 1, we have L policy updates, each requiring M episodes, which uses N designs, so $L \times M \times N$ simulations, which, for real-world simulations, is quite expensive. I'd appreciate explicitly saying how many simulations were used in experiments.
- There doesn't seem to be a good analysis of the stopping probability choice, which would help practitioners implement this method.

### Questions
- How do the theorems change with design-dependent costs?
- Which form of mutual information are they defining the gradients w.r.t. to derive their gradient? It's not stated and makes the process of finding the gradient opaque, which concerns me.
- How does the exploration scale $\sigma_\text{explore}$ effect training? Why is it called $\epsilon_\text{explore}$ some places but not others? Assuming it's just the normal distribution parameterization making the designs follow a distribution but it'd be nice to make this more explicit.
- Is the design policy network parameterized by unique parameters at each step? i.e. $w_k$ that they use in their derivation in appendix A.3. Weird they use that as the gradient for deriving their optimal stopping point.
- How was the stopping probability schedule chosen? What are the implications of that choice? 
- For the convection-diffusion model, Vanilla PG doesn't seem to underperform that dramatically. It does seem more conservative and maybe it has utility in other domains where design policies need to be more conservative? e.g. evaluating the EIG using CVAR.
- In line 105, what do the authors mean by RL can learn stopping rules but assume a fixed data acquisition process? Isn't this paper assuming a fixed data acquisition process (likelihood)?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formulates Bayesian experimental design as a joint design-and-stopping problem. The authors model sequential experiment selection as a belief-state MDP, where at each step the agent decides both which design (experiment) to perform and whether to stop collecting data. The paper proposes an actor-critic policy-gradient algorithm augmented with curriculum learning that jointly learns a design policy for experiment selection and a stopping policy for termination decisions.

### Strengths
The paper presents a clear and technically sound formulation of Bayesian experimental design as a joint design-and-stopping problem, explicitly modeling both experiment selection and termination decisions within a unified belief-state MDP.

While the general idea of comparing information gain to sampling cost has been explored in past work (e.g., knowledge gradient), this work is original in casting the problem as an explicit control process and learning both design and stopping policies jointly via a policy-gradient algorithm.

### Weaknesses
1. The paper can provide more discussions on related literature: 
- The Knowledge Gradient (KG) policy [1,2] already incorporates a form of stopping rule for Bayesian experimental design, where sampling continues only while the expected value of information exceeds its cost. Although this stopping criterion is implicit rather than derived from a joint framework, it addresses a similar decision problem. 
- The Pandora’s Box problem [3] is also a problem of joint selection and optimal stopping, where an agent sequentially explores costly options and decides when to stop — conceptually parallel to the structure of this paper’s MDP formulation. 
- There is a recent work [4] also considers a joint selection and stopping framework, explicitly proposing a joint cost-aware policy for Bayesian optimization — a problem closely related to Bayesian experimental design.

2. The paper can provide brief explanations on terminologies such as curriculum learning, actor-critic, policy gradient for broader audiences and more reasons of these choices.

3. The paper can have more discussions on computational tractability and efficiency of their algorithms, particularly in high-dimensional continuous design space (there is one sentence in the conclusion but I'd like to see some explanations).

4. The paper can summarize all hyperparameters to be determined and sensitivity to choices.

References: \
[1] Frazier, Peter I., Warren B. Powell, and Savas Dayanik. "A knowledge-gradient policy for sequential information collection." SIAM Journal on Control and Optimization 47, no. 5 (2008): 2410-2439. \
[2] Ryzhov, Ilya O., Warren B. Powell, and Peter I. Frazier. "The knowledge gradient algorithm for a general class of online learning problems." Operations Research 60, no. 1 (2012): 180-195. \
[3] Weitzman, Martin. Optimal search for the best alternative. Vol. 78, no. 8. Department of Energy, 1978. \
[4] Xie, Qian, Linda Cai, Alexander Terenin, Peter I. Frazier, and Ziv Scully. "Cost-aware stopping for bayesian optimization." arXiv preprint arXiv:2507.12453 (2025).

### Questions
1. The paper’s formulation and stopping condition appear conceptually close to the Knowledge Gradient, which also incorporates a stopping rule in Bayesian experimental design. Could the authors clarify how their proposed policy differs from the KG approach? Could the authors elaborate on why policy-gradient policy can be more effective in this joint Bayesian experiment design and stopping problem?

2. The algorithm involves Monte Carlo simulation, is it time-consuming?

3. Could the authors clarify in what sense the learned policy is empirically optimal? For instance, is it approximately optimal with respect to the simulated belief dynamics, or are there any conditions under which convergence to the Bayesian-optimal solution can be expected? Can there be any potential model misspecification? How the approximations in Monte Carlo and policy-gradient optimization affect the optimality?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper frames sequential Bayesian experimental design with an explicit stop action. It proves that the optimal rule is to stop when the terminal reward exceeds the expected continuation value, and shows that terminal and incremental reward formulations are equivalent. It then trains a policy-gradient actor–critic: the actor proposes designs, the critic estimates continuation value (Q), and stopping is decided by comparing Q to the terminal reward. To avoid early collapse, it introduces a curriculum schedule that gradually unlocks stopping during training. Experiments on a linear–Gaussian benchmark and a convection–diffusion (PDE) contaminant-source task show that the curriculum stabilizes training and reduces premature stopping compared to a vanilla PG baseline.

### Strengths
1. The paper makes stopping explicit and principled: decisions follow a simple rule—stop when the terminal reward meets or exceeds the continuation value—so it’s easy to implement and explain.

2. The theory reduces stopping to a concrete test (terminal reward vs. the critic’s Q), making the decision operational rather than heuristic.

3. Training is simple and effective: an actor–critic with a lightweight curriculum that prevents early collapse and stabilizes joint learning.

4. Empirically, the curriculum improves stability and cuts premature stopping on both the linear–Gaussian benchmark and the PDE contaminant-source task compared with a vanilla PG baseline.

### Weaknesses
1. The terminal–incremental equivalence (Theorem 2) is already established in DAD; this paper mainly adds an experimental-cost term. The authors should acknowledge this more clearly and tighten the novelty claim.

2. When N = 4, directly approximating the posterior (e.g., variational inference, Gaussian approximations, or diffusion-based posteriors) and then computing the reward may be cheaper and clearer than training an actor–critic. The paper should justify when learning actually outperforms these exact/analytic or grid-based methods.

3. DAD/iDAD/Step-DAD [1] handle non-Gaussian and implicit models, while this paper only shows Gaussian cases. If everything is Gaussian, closed-form posteriors/EIG are available—so the paper should exploit those baselines and/or include at least one non-Gaussian setting to justify the scope.

4. The stopping schedule is hand-designed, not learned; results may be schedule-dependent, and there’s little ablation on the shape, duration, or final gating level.



[1] Hedman, Marcel, et al. "Step-DAD: Semi-Amortized Policy-Based Bayesian Experimental Design." arXiv preprint arXiv:2507.14057 (2025).

### Questions
1. Have you applied your approach with alternative posterior approximations (e.g., variational inference, Gaussian, diffusion-based posteriors) and compared the results within the same framework? If not, what adaptations would be needed?

2. Did you try alternatives to stop-gating (e.g., a small per-step delay penalty, scheduled sampling, or entropy regularization on "continue")? If not, what was the intuition for choosing curriculum?

3. The current experiments assume Gaussian priors and Gaussian noise. Have you evaluated a Laplace approximation to the posterior as a baseline?

4. Please report wall-clock and sample-efficiency numbers (training time, number of episodes) to assess practical performance.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors suggest that choosing when to stop running new experiments is a neglected problem within Bayesian experimental design. They propose solving this using a policy-based approach in which the action space of the experimental-design policy includes “stop”. Their training method is an actor-critic algorithm in which stochasticity in the stopping decision is reduced over training steps (they refer to this as “curriculum learning”). They report training results for their method and some alternative methods.

### Strengths
Originality: popular policy-based approaches to BED (eg, Foster et al, 2021; Huan & Marzouk, 2016) tend to assume a sequence of T experiments; this work considers stopping after fewer than T experiments.

Quality: the writing is generally good, and I’m not aware of any major technical errors.

Clarity: aside from an overload on equations in Sections 3-4, the paper is reasonably easy to follow.

Significance: incorporating stopping into policy-based BED is a good idea for the future work to incorporate.

---

Foster et al (2021). Deep adaptive design: amortizing sequential Bayesian experimental design. ICML.

Huan & Marzouk (2016). Sequential Bayesian optimal experimental design via approximate dynamic programming. arXiv.

### Weaknesses
I’m currently unconvinced of how substantive and necessary the contribution is, and I worry that the work is presented in a way that obfuscates the nature of the contribution.

The authors tell us the BED literature has neglected stopping. Yet there is actually a good amount of coverage, including in work by Berger (1985), Garnett (2023) and Ryan et al (2016).

The authors also present three equation-heavy pages of methodological description, with the implication is that working out a solution requires this level of complexity. But it is unclear to me that the method doesn’t just amount to including “stop” in the action space of the experimental-design policy, with which we can then just use an existing algorithm, such as that of Huan & Marzouk (2016) without any modifications, or that of Foster et al (2021) with minor modifications to account for experiment costs.

On top of this, the work does not match the machine-learning community’s typical standards for evaluating new methods. The authors present training results, not test results. They also do not compare against notable methods from past work, such as those mentioned above.

---

Berger (1985). Statistical Decision Theory and Bayesian Analysis. Springer.

Garnett (2023). Bayesian Optimization. Cambridge University Press.

Ryan et al (2016). A review of modern computational algorithms for Bayesian optimal design. International Statistical Review.

### Questions
Can you explain why you can’t just augment existing policy-based BED, such as the method proposed by Huan & Marzouk (2016), to include “stop” in the action space of the experimental-design policy?

How well does your method perform in testing rather than training?

How well do the abovementioned baseline methods work?

### Soundness
1

### Presentation
2

### Contribution
1

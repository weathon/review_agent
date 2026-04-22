# Stability and Generalization for Bellman Residuals

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Offline reinforcement learning and offline inverse reinforcement learning aim to recover near–optimal value functions or reward models from a fixed batch of logged trajectories, yet current practice still struggles to enforce Bellman consistency. Bellman residual minimization (BRM) has emerged as an attractive remedy, as a globally convergent stochastic gradient descent–ascent based method for BRM has been recently discovered. However, its statistical behavior in the offline setting remains largely unexplored. In this paper, we close this statistical gap. Our analysis introduces a single Lyapunov potential that couples SGDA runs on neighbouring datasets and yields an $\mathcal{O}(1/n)$ on-average argument-stability bound—doubling the best known sample-complexity exponent for convex–concave saddle problems.  The same stability constant translates into the $\mathcal{O}(1/n)$ excess risk bound for BRM, without variance reduction, extra regularization, or restrictive independence assumptions on minibatch sampling. The results hold for standard neural-network parameterizations and minibatch SGD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper analyzes the statistical behavior of Bellman Residual Minimization (BRM) for offline RL/IRL. Building on the recent optimization view that the bi-conjugate BRM objective induces a PL–strongly-concave minimax structure, the authors couple two SGDA runs on neighboring datasets via (i) a single Lyapunov potential that mixes primal suboptimality and primal–dual mismatch, and (ii) a “ghost-index” device to decouple sampling noise. They prove on-average argument stability of SGDA with an O(1/n) rate (under Robbins–Monro stepsizes), and transfer this to O(1/n) generalization and an excess-risk bound that cleanly decomposes optimization and estimation errors. The setup, assumptions (A1–A9), and the transfer to weak PD-gap follow the minimax stability framework.

### Strengths
Originality
- Closes a real theoretical gap: Prior minimax stability analyses (e.g., Wang–Lei–Ying–Zhou, NeurIPS 2022) deliver O(n^{-1/2}) rates under convex–concave assumptions. This paper’s O(1/n) stability and generalization results for SGDA in a PL–strongly-concave regime appear novel.
- Combines multiple theoretical tools—bi-conjugate BRM formulation, PL geometry, a Lyapunov potential, and ghost-index coupling—into a coherent analysis without variance reduction or independence assumptions.
- The unification of optimization and generalization analysis through a single Lyapunov potential is an elegant methodological contribution.

Quality
- The proofs are internally consistent and technically sound under the stated assumptions (A1–A9). The Lyapunov-based stability recursion is clearly constructed and all major theorems are proven in full.
- The paper avoids dependence on variance-reduction or mixing assumptions, deriving O(1/n) bounds via standard SGDA under Robbins–Monro step sizes.
- The key limitations lie in the strong assumptions—bounded per-sample gradients, uniform constants across neighboring datasets, and uniqueness of the saddle—that may not strictly hold for deep neural networks.

Clarity
- The exposition is clear, particularly in articulating the problem gap (“optimization picture is clear; statistical picture remains open”).
- The algorithmic setup, potential function, and contraction argument are well explained with intuitive justification for summability of noise terms.
- Proof dependencies and structure are explicitly cross-referenced in the reproducibility statement, ensuring transparency.

Significance
- The results provide the first O(1/n) generalization bound for Bellman Residual Minimization in offline reinforcement learning, doubling the exponent achieved in prior convex–concave analyses.
- The theoretical framework may generalize to other PL-minimax problems beyond BRM, influencing theoretical and algorithmic directions in RL and IRL.
- While the assumptions restrict direct practical application, the analysis sets a higher theoretical standard for understanding statistical generalization in nonconvex–concave RL objectives.

### Weaknesses
1) Assumptions feel strong and under-motivated for neural BRM
Issue: The analysis depends on assumptions such as bounded per-sample gradients, uniform constants across neighboring datasets, and uniqueness of the saddle. These are not linked to concrete architectural or data-level conditions.
Actionable Fixes:
- Provide sufficient conditions (e.g., Lipschitz activations, spectral normalization, weight decay) ensuring these assumptions hold.
- Add perturbation lemmas for small constant drift across neighboring datasets.
- Explain how regularization ensures uniqueness of the saddle.

2) Positioning vs. existing stability literature could be sharper
Issue: The claimed novelty (O(1/n) vs O(1/√n)) relative to convex–concave minimax works (e.g., Wang et al., NeurIPS 2022) lacks a clear side-by-side comparison.
Actionable Fixes:
- Include a comparison table contrasting assumptions, settings, and rates.
- Explicitly highlight which steps rely on PL–strong concavity and would fail otherwise.

3) Minibatch dependence not clearly quantified
Issue: Theorems mention minibatch adaptation “verbatim” without giving explicit batch-size-dependent constants.
Actionable Fixes:
- Add a corollary deriving ε_T(B) with explicit 1/B scaling and its impact on generalization and excess-risk bounds.
- Provide practical guidance on choosing batch size B.

4) Lack of empirical sanity checks
Issue: The paper claims parametric O(1/n) scaling but shows no supporting experiment.
Actionable Fixes:
- Include a toy experiment using linear BRM satisfying all assumptions to empirically verify slope ≈ –1 in log–log plots.
- Compare against convex–concave baselines to show contrast.

5) Clarity gaps in bi-conjugate BRM formulation
Issue: The connection from the bi-conjugate Bellman residual to the minimax form is hard to follow for non-experts.
Actionable Fixes:
- Add a concise boxed derivation linking the BRM objective to the dual variable.
- Include a diagram illustrating shared-index coupling and “hit” events.

6) Excess-risk decomposition underemphasized
Issue: The clean decomposition between stability and optimization error appears late and without clear interpretation.
Actionable Fixes:
- Promote the decomposition as a boxed equation in the main text.
- Explain how tuning T and η_t balances the two error terms.

7) Limited discussion beyond entropy-regularized BRM
Issue: It is unclear whether the results extend to non-entropy (hard-max) BRM formulations.
Actionable Fixes:
- Add remarks outlining when PL–strongly-concave structure persists under different smoothings (e.g., Moreau envelopes).

8) Ambiguity in “one pass over n samples” phrasing
Issue: The notion of “one pass” may be misread without clarifying total gradient calls or sampling scheme.
Actionable Fixes:
- Specify whether T ≈ n steps correspond to one epoch and whether sampling is with or without replacement.

Overall, the paper would improve by making its assumptions verifiable in practice, providing explicit batch-size scaling, and including minimal empirical verification. These additions would make the theory more credible, checkable, and actionable for the ICLR audience.

### Questions
1. On Assumptions and Applicability
- Could you provide explicit sufficient conditions on the neural-network architecture or data distribution that ensure assumptions (A5) and (A8) hold? For example, do ReLU or tanh activations satisfy the Lipschitz and gradient-boundedness assumptions under spectral normalization or weight clipping?
- The analysis assumes a unique saddle point, yet neural networks are often overparameterized. Is uniqueness strictly necessary, or could the analysis extend to a set of equivalent saddles?

2. On Novelty and Positioning
- The claimed improvement from O(n^{-1/2}) to O(1/n) hinges on the PL–strongly-concave structure. Could you explicitly summarize which elements of your proof break down in purely convex–concave settings?
- To what extent could your Lyapunov and ghost-index coupling analysis extend to other PL-minimax settings (e.g., actor–critic or distributional RL formulations)?

3. On Practical Interpretability
- You mention that the minibatch setting follows “verbatim” with rescaled constants. Could you please provide the explicit scaling law of ε_T(B) in terms of B and n?
- When stating that you achieve the O(1/n) rate “after one pass over n samples,” do you mean T ≈ n SGDA steps, one epoch with sampling with or without replacement?

4. On Theoretical Sharpness
- Your current bounds are in expectation. Do you think similar rates could hold with high probability using martingale inequalities (e.g., Azuma or Freedman)? If so, how would the constants or rates degrade?
- Could you comment on how sensitive your results are to the condition numbers L/μ_PL and L/ρ?

5. On Empirical Verification
- Would you be open to adding a toy experiment (e.g., linear-quadratic BRM under the assumptions you make) to confirm the slope of the generalization error versus sample size?
- Even a small-scale plot could visually substantiate the theoretical rate and convince a broader ICLR audience.

6. On Extensions and Generality
- Your analysis focuses on the softmax (entropy-regularized) case. Could you clarify whether the PL–strongly-concave geometry and stability proof extend to hard-max or Moreau-smooth Bellman operators?
- Would your argument still hold under Markovian dependence rather than i.i.d. samples? If not directly, what modifications would be necessary to handle the mixing-time dependence?

7. On Presentation and Readability
- Could you include a short boxed derivation showing how the Bellman residual minimization problem transforms into the minimax form involving the dual variable?
- The final decomposition separating optimization and generalization errors is one of your most interpretable results. Consider moving it earlier into the main body with a brief intuitive discussion.

8. On Possible Future Directions
- How do you envision extending your analysis to policy-based or actor–critic settings, where the loss is not strictly bi-convex/bi-concave?

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
2

### Summary
This paper studies Bellman Residual Minimization (BRM) for offline RL. Using a bi-conjugate reformulation, minimizing MSBE is turned into a Polyak--Łojasiewicz (PL)–strongly-concave minimax problem that can be solved by SGDA, thereby avoiding the double sampling problem. The analysis couples two SGDA runs on neighboring datasets and proves on-average algorithmic stability with an $O(1/n)$ rate, without requiring variance reduction or independence assumptions. By stability-to-generalization transfer, the work bounds (i) the gap between population and empirical Bellman-residual risks and (ii) the population Bellman-residual risk of the SGDA output.

### Strengths
- Without requiring independence assumptions on the sample indices nor variance reduction, the paper establishes an $O(1/n)$ on-average stability and, via stability-to-generalization transfer, an $O(1/n)$ generalization bound for BRM, doubling the exponent from $1/2$ to $1$ over prior work.
    
- The population excess risk is cleanly decomposed into an optimization term that decays with training and a sample-size–dominated statistical term, naturally aligning with standard minibatch SGDA.
    
- All assumptions are stated explicitly and clearly, making the analysis easy to follow.

### Weaknesses
- It would be helpful to add illustrative examples and comparisons to aid understanding (see Q 1 and 2).


- Sections~2 and 3 include substantial repetition of well-known material, and the exposition feels overly long. For example, the standard SGDA routine could be moved to the appendix for brevity.

### Questions
- How strong is Assumption A8? Do the constants remain unchanged under a single-sample replacement in general setting, and could the authors provide a concrete example illustrating when A8 holds or fails?

- In Corollary~4, could you quantify the iteration threshold $T^\star$ at which the optimization term is below the statistical term formally? Additionally, for the small-$T$, could you provide a comparison with prior methods? 
  
- Would it be possible to use one of $(w,v)$ or $(\theta_1,\theta_2)$ to unify the notation since these seem to denote the same primal/dual variables?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper analyzes the excess risk bound for offline reinforcement learning in view of Bellman residual minimization.

### Strengths
The problem of analyzing the excess risk bound for Bellman residual minimization does seem open so far.

### Weaknesses
- The comparison to existing works in approximate dynamic programming methods e.g. projected Bellman equation-based approaches seems inadequate. Is Bellman residual minimization the only way to accommodate the difficulty of enforcing Bellman consistency? What are the other existing risk bounds when incorporating function approximations and how do these results compare?
- The techniques used seem to be standard, e.g. PL for analyzing SGDA etc. It seems unclear from the manuscript what are the technical challenges and the techniques developed in this paper that are independent of the developments from combining Kang et al. 2025 and Wang et al 2022. What is the motivation when defining the Lyapunov potential? Some discussions around lines 369-375 when introducing this object would greatly help the reader.
- The presentation of Theorem 6 and in general Section 3 can be improved. As far as I understand, this paper is considering the specific problem of learning the (action)-value function, and thus introducing 9 assumptions for a general function F and auxiliary results about general risks introduces additional notation while not clear to what extent they are helpful in elucidating the final result (Theorem 6). I would think a clearer explanation why value functions and lyapunov potential satisfy the assumptions needed to establish Theorem 6 and intuition of the result would be more helpful than the results about general F along with 9 additional assumptions (that will automatically be satisfied).


Minor points:
- There are superfluous "equation" when referring to equations throughout the paper, e.g., Equation 4, etc. Please remove those.
- Line 80: "Throughout, focus on single-agent decision making problem interacting with a discounted Markov Decision Process (MDP) described by the tuple ( S, A, P, r, β , ν 0)" is lacking a subject.
- Bellman consistency in line 38 comes out directly without motivation or explanation. Why do we want consistency and what does it mean? In the last sentence you said "satisfies the Bellman optimality equations even though no new state–action pairs can be queried." but Bellman consistency means fixed point of Bellman equations, which is not shown here.

### Questions
See previous section

### Soundness
3

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
The paper considers the problem of minimizing the Bellman error in a TD (Temporal Difference) update and cast this as a minimax problem of optimizing an objective that involves two parameterized functions : 1) Q function as a function of state and action and 2) The other is a parameterized neural net that given current state and action approximates the value function of the future sampled state. This optimization objective is derived (from prior work) by characterizing the bias between the squared Bellman error with respected to the expected TD operator and sampled TD Bellman error.  Further the surprising fact about this parameterization is that the problem is concave with respect the second function and the objective after inner optimization satisfies the PL condition with respect to the first Q function when you consider the stochastically approximated variant under general parameterizations (specifically linear function approximation).

Motivated by this, the authors propose to perform a stability analysis that would bound the generalization error (in terms of the duality gap) between the mini max problems which sees the population version and the the sample version. Authors adopt the stability analysis (that is known to imply generalization in the sense of duality gap from prior work) where the mini max problem see two sets of sequence of samples (state transitions) where one of the samples is different and authors seek to bound the distance of between the primal and dual iterates of these two coupled minimax problems. 

 Authors introduce two interesting ideas: 1) Ghost index which is an index independently sampled from the dataset which is independent of the Filtration and gradient with respect to this sample in expectation can approximate the population gradient 2) PL condition implies for the outer problem and strong concavity for the inner problem imply contraction for a Lyapunov function that is a combination of the primal gap and the dual gap in the expected function value.

Authors use this and existing results about stability to prove generalization of the primal and dual gap from sample to the population version.

### Strengths
The paper (to my knowledge) is the first to consider stability analysis exploiting the PL condition and strong concavity of the respective problem to show generalization errors in primal and dual gaps.  There are a lot of algebraic manipulations that deftly use the ghost index, contraction properties of the outer and inner problem to establish bounds on generalization error. The application to Bellman residual optimization is noteworthy although it borrows heavily from prior work.

### Weaknesses
1) My first concern is inadequate quoting of results from Kang et al 2025 that misleads reading this paper. Line 230 and 231 says that Kang et al. 2025 proved that PL condition is satisfied with respect to the parameters of the Q function (primal variables) when parameterized by a Neural Network. I read the prior paper. There are lots of caveats to the Neural Network result - it traces back to the result in https://arxiv.org/pdf/2003.00307 - where authors show that - wide and deep neural nets satisfy the PL condition over a radius around a random initialization if the width scales as radius^depth.  Further, the theorem is easily proven only for linear function approximation in Kang et.al. 2025.

2) Second concern is that ghost index trick works because, say for the inner problem, gradient is assumed to be uniformly bounded. This is rather a very strong assumption. However, the inner problem is strongly concave and *Page 2 of this ICML paper https://proceedings.mlr.press/v80/nguyen18c.html  shows that unless the ball of iterates is bounded explicitly, uniform gradient norm bound contradicts strong convexity (or concavity) !*

Authors can have uniform bound G on gradient norm only if the iterates stay within a ball of certain radius from where it starts at least for the inner concave problem. The algorithm described is unprojected SGDA and the problem needs to project itself on every update to some ball. In the RL context that would mean projecting the iterates of the parameters of the Q function to a ball that would encapsulate the optima - rather a very strong assumption. Even the Neural net satisfying bounds of gradient, Hessian and Jacobian operator (assumption 5 in Kang et al 2025  paper) is possibly within some small ball around the initialization for a network of given width.

### Questions
1) Can you answer the above 2 weakness points ? Question about the need for projected steps if gradient bound is assumed is rather concerning and could be a serious weakness as written

2) Paper quotes the deadly triad relating to convergence of Q learning. There is a recent paper on resolving it for linear function approximation (https://arxiv.org/abs/2203.02628) using truncation and target network. Discussing these alternative works is very important.

I think the gradient bound issue is more serious. Therefore, I have given rating of 2. I would wait for authors to respond to that and I can raise my score.

### Soundness
3

### Presentation
3

### Contribution
2

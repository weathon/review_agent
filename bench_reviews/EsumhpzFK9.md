## Summary
This paper proposes KARMA, a reinforcement learning framework that augments the reward signal using two additional ingredients: structured domain knowledge encoded as a knowledge graph and a learned causal model used for counterfactual reward adjustment. The intended contribution is a modular reward-shaping framework that dynamically shifts from knowledge-guided shaping early in training to causally informed shaping later, with claimed benefits in sample efficiency, robustness, and generalization across grid navigation, robotics, and traffic control tasks.

## Strengths
- **The paper targets a real and important failure mode—spurious or misleading reward signals—and addresses it at the reward-design level rather than only through better policies or representations.** This is more specific than generic “causal RL”: KARMA explicitly applies causality to *reward adjustment* via Eq. (1), which is a meaningful conceptual angle.
- **The framework is modular in a way that is easy to inspect experimentally.** The decomposition into knowledge integration, causal discovery, and reward adjustment modules is reflected in the ablation table (Table 2), which at least attempts to separate where the gains come from rather than only presenting an end-to-end black box.
- **The empirical scope is broader than a single toy benchmark.** The paper evaluates on three qualitatively different settings—GridWorld with causal interference, robot skill acquisition, and traffic signal control—and reports not only final return/sample efficiency (Table 1) but also robustness and distribution-shift results (Figure 6), which is aligned with the paper’s stated goals.
- **The ablations support that the full combination matters more than any single ingredient alone.** In Table 2, removing reward adjustment hurts most, while removing knowledge or causal learning also degrades results, which is consistent with the paper’s claim that the synergy of the components is important rather than incidental.
- **The paper is unusually explicit about computational cost for this kind of systems-style method.** Table 3 reports training time, peak memory, and inference latency, making clear that the gains are not free and allowing readers to judge the tradeoff.

## Weaknesses

### Fatal
- **The core method is underspecified to the point that the main contribution is not scientifically verifiable from the paper text.**  
  The central mechanism is Eq. (1),
  \[
  R'(s,a,r,s') = r + w_K(t)R_{\text{knowledge}}(s,a,s') + w_C(t)R_{\text{causal}}(s,a,s')
  \]
  but neither \(R_{\text{knowledge}}\) nor \(R_{\text{causal}}\) is defined operationally. Section 3.3 only states that “\(R_{\text{knowledge}}\) promotes trajectories consistent with KG constraints” and that “\(R_{\text{causal}}\) is obtained through counterfactual queries on \(C\), using Pearl’s do-calculus to estimate causal effects of actions, disentangled from confounders.” This is too high-level for the paper’s core algorithm: there is no mathematical form, no estimator, no pseudocode, no description of how the counterfactual quantity is computed per transition, and no explanation of how these terms are scaled or normalized so that PPO can be trained stably.  
  Section 4.4 adds some implementation details (“linear annealing,” “counterfactuals computed via structural equation modeling”), but still does not specify the actual structural equations, fitting objective, or how online estimates enter the reward at each step. Since the central claim is precisely that this reward adjustment mechanism drives the gains, the lack of formal algorithmic definition materially undermines technical soundness.

### Major:
- **The claimed theoretical guarantees are not substantiated in the main paper at a level commensurate with the strength of the claims.**  
  Section 3.4 lists four guarantees—convergence of causal discovery, policy invariance, improved sample efficiency, and convergence of KARMA-RL to an \(\epsilon\)-optimal policy—but presents them only as bullet points under “mild assumptions.” The assumptions are not spelled out in the main text, nor are theorem statements or proof sketches given. Some of these claims are individually plausible in restricted settings (e.g., policy invariance if the shaping term is potential-based), but the paper does not establish that the *full* KARMA reward in Eq. (1) satisfies those conditions. In particular, the policy-invariance statement is explicitly conditional: “**If** \(R_{\text{knowledge}}\) is designed as a potential-based shaping function, the optimal policy is preserved.” The paper never shows that the implemented reward actually has that form, and no analogous condition is given for \(R_{\text{causal}}\). Because the paper foregrounds theory in the abstract and contributions list, this lack of formal support is a substantial weakness.
- **The causal discovery component is insufficiently justified for the online RL setting used here.**  
  The paper states in Section 3.2 that it uses “constraint-based methods (e.g., PC, FCI) with score-based refinements,” and Section 4.4 says causal discovery is updated “every 1000 interactions.” However, the paper does not explain how these discovery procedures are adapted to temporally correlated, policy-dependent RL data, nor how conditional independence testing is handled in continuous/high-dimensional environments like the 7-DOF robot and traffic control tasks. This is not merely a request for extra detail: the credibility of \(R_{\text{causal}}\) depends on whether the learned graph is meaningful under the actual data-generation process. The text mentions “MDP-informed temporal constraints,” which partially addresses obvious temporal ordering, but it does not resolve the broader issue that the discovery pipeline’s assumptions and estimation quality are left vague precisely where the method depends on them.
- **The empirical evidence does not adequately isolate whether the gains come from causal reasoning specifically, versus from adding reward density or extra task-specific information.**  
  The paper compares against standard RL, knowledge-based RL, and causal RL baselines, and provides internal ablations. However, it does not include a matched reward-shaping baseline that would control for the possibility that much of the gain comes simply from providing a denser auxiliary reward rather than from correct causal/counterfactual adjustment. This matters because Eq. (1) adds two extra reward terms on top of the environment reward; without a stronger control, the causal interpretation of the gains remains weaker than the headline framing suggests.
- **The paper’s robustness claims are incomplete because robustness to *incorrect knowledge* or *misspecified causal priors* is not evaluated.**  
  Section 6 explicitly acknowledges that “large errors can harm learning” and that performance depends on “the quality of its knowledge graph and causal model.” But the experiments do not probe this dependence: there is no stress test with corrupted knowledge graphs, contradictory rules, or degraded causal structure estimates. Since a core premise of the framework is that prior knowledge helps disambiguate spurious reward signals, the absence of sensitivity analysis leaves a significant gap between the claimed practical robustness and what is demonstrated.

### Minor
- **The RLVR/LLM-alignment motivation is much broader than the evaluation actually supports.**  
  The introduction motivates the work partly through spurious rewards in RLVR for language models, but all experiments are on classical control/simulation tasks. This does not invalidate the paper, but the significance claims around alignment and RLVR are aspirational rather than empirically supported by the presented results.
- **The dynamic weighting mechanism is plausible but not convincingly analyzed.**  
  The paper claims that \(w_K(t)\) should matter early and \(w_C(t)\) later, but beyond the statement of “linear annealing” and the static-weight ablation, there is no analysis of the schedule itself, no plot of the weights over training, and no evidence that this transition corresponds to a meaningful change in what the agent learns. As written, it is hard to tell whether the schedule is a key idea or simply a generic reward-bonus decay heuristic.
- **The computational overhead is reported but not decomposed.**  
  Table 3 is useful, but it would be more convincing to separate the cost of KG processing, causal graph updates, and counterfactual reward computation. That is especially relevant because scalability is already acknowledged as a limitation in Section 6.
- **Only five runs are used, which is on the light side for claims about consistent gains across multiple components and settings.**  
  The paper does report mean and standard deviation and mentions t-tests, which is better than many submissions, so this is not a major methodological flaw. Still, for a method with several moving parts and potentially high variance, the evidence base is somewhat limited.

### Trivial
- None.

## Nice-to-Haves
- Add a controlled baseline with comparable dense reward shaping but without causal machinery, to better isolate the value of the causal component.
- Include explicit pseudocode and formulas for \(R_{\text{knowledge}}\), \(R_{\text{causal}}\), the SEM fitting procedure, and the weight schedules \(w_K(t), w_C(t)\).
- Add sensitivity experiments with noisy/incomplete/contradictory knowledge graphs and imperfect causal priors.
- Visualize the learned causal graph against known structure in the GridWorld setting, where ground-truth causal dependencies appear available.
- Break down runtime by module and discuss whether parts of the pipeline can be precomputed or amortized.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“The paper misses critical related work.”**  
  Removed per instruction: I cannot verify external omissions beyond the cited references, and the criticism was framed broadly rather than against a concrete mischaracterization in the paper.
- **“The comparison is unfair because baselines do not get the same extra information.”**  
  Removed. The authors’ method is intentionally designed to use additional knowledge/causal structure; this asymmetry does not by itself make the comparison invalid, and in fact often strengthens the claim if the baseline is disadvantaged less than the proposed method.
- **“Code is not public during review / supplementary material is inaccessible / post-publication release harms reproducibility.”**  
  Removed as a core criticism. Lack of immediate release should not be treated as decisive here, and the paper does provide some implementation details. The real issue is not release status but that the main algorithm itself is underspecified.
- **Generic praise such as “the paper is well-written” or “the topic is important.”**  
  Removed because these strengths are too generic under the reviewing instructions.
- **Purely stylistic complaints about the related work being descriptive or the prose using broad motivation language.**  
  Removed unless tied to a substantive technical issue.

## Novel Insights
The most important synthesis across the reviews is that the paper’s central risk is not simply “more details would help,” but that the claimed contribution sits at an awkward boundary between conceptual framework and complete algorithm. The experiments and modular ablations suggest there may be a real underlying idea here—using knowledge to regularize early learning and causal estimates to refine rewards later—but the paper never closes the loop from concept to a technically precise, auditable method. Put differently: the strongest positive signal is that the authors have identified an interesting *design space* for reward shaping, while the strongest negative signal is that the current submission does not yet establish KARMA as a sufficiently specified method within that design space.

## Suggestions
- **Define the core reward terms formally.** Give exact formulas or algorithms for \(R_{\text{knowledge}}\) and \(R_{\text{causal}}\), including normalization/scaling and how they are injected into PPO advantage estimation.
- **State theorems properly in the main paper.** At minimum, include theorem statements, assumptions, and proof sketches for the convergence/invariance claims, and be explicit about which claims apply only to restricted variants of KARMA.
- **Clarify causal estimation in RL data.** Explain how the SCM is fit from sequential interaction data, how often it is updated, what variables are included, and what assumptions justify using the chosen causal discovery method online.
- **Add stronger controls.** Compare against a matched non-causal dense reward shaping baseline to test whether causality contributes beyond reward densification.
- **Stress-test prior misspecification.** Corrupt the knowledge graph and/or causal priors in controlled ways and quantify degradation.
- **Show one concrete case study.** For example, in GridWorld, visualize a spurious feature, the learned causal graph, and an instance where KARMA adjusts the raw reward in the intended direction.
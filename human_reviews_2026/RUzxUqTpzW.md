# Does “Do Differentiable Simulators Give Better Policy Gradients?” Give Better Policy Gradients?

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
In policy gradient reinforcement learning, access to a differentiable model enables 1st-order gradient estimation that accelerates learning compared to relying solely on derivative-free 0th-order estimators. However, discontinuous dynamics cause bias and undermine the effectiveness of 1st-order estimators. Prior work addressed this bias by constructing a confidence interval around the REINFORCE 0th-order gradient estimator and using these bounds to detect discontinuities. However, the REINFORCE estimator is notoriously noisy, and we find that this method requires task-specific hyperparameter tuning and has low sample efficiency. This paper asks whether such bias is the primary obstacle and what minimal fixes suffice. First, we re-examine standard discontinuous settings from prior work and introduce DDCG, a lightweight test that switches estimators in nonsmooth regions; with a single hyperparameter, DDCG achieves robust performance and remains reliable with small samples. Second, on differentiable robotics control tasks, we present IVW-H, a per-step inverse-variance implementation that stabilizes variance without explicit discontinuity detection and yields strong results. Together, these findings indicate that while estimator switching improves robustness in controlled studies, careful variance control often dominates in practical deployments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper studies the problem of finding optimal trajectories using differentiable simulators. 

Differentiable simulators allow us to differentiate dynamics (and rewards) w.r.t. actions, allowing efficient policy optimization. However, gradients are inaccurate in the presence of dynamic discontinuities (e.g., contact), posing a significant challenge for the use of gradient information in trajectory (or policy) optimization.

Previous work proposed mitigating the issue by combining zero-order (e.g., REINFORCE) and first-order methods (e.g., reparametrization gradients that differentiate the simulator). Zero-order gradients are unbiased, but typically suffer from high variance. First-order gradients, on the contrary, are subject to low variance, but can sporadically be subject to high variance in steep or non-differentiable regions. 

In particular, the authors consider two algorithms: Inverse Variance Weighting (IVW) and Interpolation Protocol (AoBG).

IVW uses the empirical variance of the estimator to reduce variance, allocating a higher weight to the estimator with the smaller variance. This method usually allocates a higher weight to the first-order method, but still fails to remove the bias. 

AoBG introduces an additional safeguard mechanism that aims to check the presence of bias by allocating more weight to the zeroth order in case the first order and the zeroth order estimates largely disagree. However, according to the authors, the hyperparameters introduced by AoBG are subject to high sensitivity and should be tuned for different environments/tasks. 

Building on these intuitions, the authors propose an algorithm that also builds on IVW and, similarly to AoBG, a "filtering" that allows one to revert to the zeroth-order information in the presence of bias. The authors notice that steep dynamics not only introduce a bias but might also harm the variance estimation of the estimator (i.e., the empirical variance might not be accurate). They devise a test that checks at once that the dynamics are not discontinuous, while the empirical variance is trustworthy (eq. 14). If that is the case, they rely on IVW weighting to perform the gradient estimation; otherwise, they revert entirely to relying on the zeroth-order estimator.

The authors show that a) their algorithm is less sensitive to estimation errors due to discontinuities (Figures 2 and 3); b) Figure 4 clearly highlights the ability of the developed algorithm to switch to the zeroth order derivative when needed (unlike IVW), while AoBG can be either too conservative or too lenient. Although the error difference in the ablation studies does not seem so significant, the policy optimization experiment (Figure 5) really shows the advantage of relying on good gradient estimation. This advantage is greater in high-dimensional, contact-rich tasks like Ant and Hopper, whereas it is negligible in the smooth, low-dimensional dynamics of the pendulum.

### Strengths
The paper analyses an important and relevant problem in model-based policy optimization arising from inaccuracies in differentiable simulators.

The authors present the state of the art well, explain the problems, introduce their solution, and build upon it. The reading is clear and interesting.

The presented method is sound. 

The empirical analysis is very well done. The authors present three studies: 1) they check how a discontinuity affects the gradient estimators; 2) they check both pure trajectory optimization and policy optimization, showing gradient estimation errors, the weights of the estimators, and the total cost; and 3) the learning curve in a pure policy optimization scenario in three MuJoCo tasks. A hyperparameter sensitivity analysis is provided in the Appendix. 

The results agree with the intuition built into the method.

### Weaknesses
This paper has a few weaknesses:

1) Equation 14, which is the core of the method, is not really explained in the main paper. The authors should have devoted some effort in building an intuition for it, rather than relegating the proof to the appendix. I am aware that, in estimating the gradient of a function, the concavity (or convexity) has an impact on its variance, thus I think that building an intuition on eq. 14 should not be too hard.

2) The acronym AoBG has never been introduced. 

3) Equations 4 and 6 seem to be one-sample estimates, but later in the paper, it becomes clear that they are utilizing n-samples. 

4) The notation in equation 3--6 is a bit too compact, hiding details and dependencies between variables. 

5) the number of seeds used is not explicitly stated in the main text (unless I have missed it). The tales in Appendix K report number of trials (which is typically high). I am unsure whether the trial correspond to the number of seeds.

### Questions
Questions
--------------

I am unsure why the proposed method checks only the empirical variance of the first-order estimate. Typically, zeroth-order estimators also suffer from high variance, which can make the empirical variance less precise. Could the authors clarify that?

Suggestions
----------------

A technical note: the baseline subtraction can make REINFORCE biased if it is estimated with the same samples. Look at this example, where b is the average reward (typical baseline subtraction):

$$\hat{g}^{(n)} = n^{-1}\sum_{i=1}^n \nabla \log p(\tau_i) \left(r(\tau_i) - \hat{b}^{(n)}\right)$$ with 
$$\hat{b}^{(n)} = n^{-1} \sum_{i=1}^n r(\tau_i) $$

For $n=1$, $\hat{g}^{(n)} =0$, clearly show that, unless the ground truth is always zero, this gradient estimator is biased. Of course, this issue is avoided when one uses two independent sets of samples to estimate the gradient and the baseline.

I recommend adding this useful information to the text and explicitly writing how you compute the baseline subtraction for the 0th order.

Section 4.2 could have a more meaningful title.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper examines the limitations of differentiable physics engines when used to compute policy gradients in reinforcement learning.  In contrast with zero-order methods, such as REINFORCE and its descendants, differentiable physics engines allow direct first-order computation of the policy gradient.  This can reduce the variance of the gradient estimate, since it is less reliant on random sampling.  However, the bias of the gradient estimate may increase when the physical dynamics are not strictly differentiable everywhere - for example, at impulse events. Recent works employ a weighted average of zero- and first-order estimates to get the best of both worlds, but selecting the weights for the averaging is a non-trivial problem.

The present paper explores two methods to set those weights.  The first method (DDCG) uses statistical hypothesis testing to detect when the dynamics are near a discontinuity, in which case the zero-order estimate is used.  The second method (IVW-H) is a variant of a prior technique, "inverse variance weighting," which computes the weights on a per-time-step basis.  The paper includes some theoretical considerations motivating these methods, as well as empirical comparisons with several prior works in several environments - including "toy" environments to aid the analysis as well as some Mujoco continuous control tasks.  On the one hand, the results show that explicit discontinuity detection with DDCG is competitive with prior methods, while being less sensitive to hyper-parameter tuning.  On the other hand, the Mujoco task results show that IVW-H, which does not explicitly detect discontinuities, performs comparably to or better than prior methods.  The paper concludes from the latter results that variance control may be more important than bias control, at least in some Mujoco RL benchmarks.

### Strengths
- While I am not fully up to date on differentiable physics simulation, as far as I know the methods introduced in this paper are original.
- The empirical results are rather comprehensive, involving multiple environments of varying complexity and comparison with multiple baselines.
- The empirical results suggest that the new methods are more effective than past work, and likely to be a significant contribution in the sub-field of differentiable physics and first-order policy gradients.
- Aside from certain issues described below, the paper is mostly clear and relatively easy to follow.

### Weaknesses
- I have a doubt concerning theoretical soundness.  The DDCG mixing weights are determined via Eq (14) which, according to the paper's notation, appears to use a true variance, not a sample estimate.  But then how is this equation used in practice, if the true variance is not known?
- I also have a doubt about the experimental results and conclusions.  The paper makes a conclusion about the relative importance of variance vs bias control.  However, it does not seem there is any experiment that directly compares the two proposed methods on the same benchmark - DDCG for bias control, IVW-H for variance control.  So, I am not sure the empirical results support the overall conclusion of the paper.  I also wonder why there is no experiment evaluating both methods on the same benchmark.
- In terms of presentation, much important information is pushed to the appendix.  I understand the page limit is tight, but it would be better if the main text were more self-contained.
- Related to the previous two points, this paper appears to explore two orthogonal methods (DDCG, IVW-H), which dilutes the focus of the paper.  If each method were published in a separate paper, each paper would be more focused, and there would more room in the main text for a detailed analysis of each method.
- There are some terms/symbols that are never defined or defined much later in the paper.  In particular:
    - The acronym "AoBG" is never defined, I had to find it in one of the citations.  It should be defined at its first occurrence.
    - In section 4.1, the function $f$ is mentioned abstractly without any specific example of what this function is.  A concrete example is given much later in 5.2.3, but the paper's clarity would benefit if one or more examples are mentioned when $f$ is first defined.

### Questions
- Can the authors explain more about Eq 14 - whether the true variance is/must be known for this method to work, why or why not?
- Can the authors justify why they did not evaluate DDCG and IVW-H on the same benchmark?

### Soundness
2

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
The paper re-examines when analytic policy gradients from differentiable simulators truly help, proposing a lightweight statistical check to safely integrate analytic gradient with model-free gradient estimates via inverse-variance mixing

### Strengths
- The proposed check is theoretically sound, clean, simple to verify, and provably more efficient than the existing AoBG approach; consequently, the estimates given by the proposed approach are provably less noisy than AoBG
- Both adequate empirical results and theoretical justifications (except for one point I listed below in weaknesses)
- I enjoyed the presentation, which walks the readers through the problem setup, existing challenges, and their solution smoothly
- I like the experimental setups, especially that the authors design very focused experiments to test the proposed approach

### Weaknesses
- Line 147 “This estimator remains unbiased if R is continuous,” — I think this statement is misleading. The first order gradient is only unbiased if the dynamics model, reward function employed are perfect. It is just that in continuous regimes, the dynamics model normally tends to be more accurate. Similarly, “However, when R is discontinuous, the 1st-order estimator can be biased.” — the first-order gradient is almost always biased since there is barely a perfect model. 
- It seems that choosing a proper c is task-dependent. Also, assuming a near-quadratic model is restrictive 
- Why not also run AoBG in Section 5.3? I think it would be valuable to also see how AoBG performs in these continuous control tasks, especially given that performance gap between AoBG and DDCG is mostly small in previous tasks and that the paper is titled “Does “do differentiable simulators give better policy gradients?” give better policy gradients?”

### Questions
- Equation 14 — The proposed check computes variance for the function values at the RHS. How to make sure that the quantity is accurately estimated in the first place? If it needs to be estimated from a sampled batch, then isn’t it running into the same issue and the argument circular?
- Figure 4 (a-b): The estimated alpha values are quite different between AoBG and DDCG in these two cases, but why are the resulting costs almost the same? 
- Figure 4 (c): Why the 0th order gradient performs as well as DDCG and AoBG in this case? Does that suggest the evaluation is using too many samples such that the 0th-order gradient estimate is already accurate?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates a key challenge in reinforcement learning: how to best calculate policy gradients when using a differentiable simulator. It re-examines the trade-offs between two types of gradient estimators and proposes new methods to improve performance and robustness.

### Strengths
- The paper tackles a significant and practical issue in reinforcement learning: how to effectively use differentiable simulators when their gradients are biased by non-smooth dynamics.
- It provides a clear and constructive critique of the AoBG method. It compellingly demonstrates AoBG's key weakness: extreme sensitivity to its hyperparameter which requires extensive, task-specific tuning.
- The paper proposes DDCG, a novel composite estimator. This method is a conceptual advance because its statistical test does not rely on the "notoriously noisy" 0th-order gradient estimator, which improves scalability and sample efficiency compared to AoBG.

### Weaknesses
- The paper's experimental validation is split. The proposed DDCG is tested on the "empirical-bias" tasks, while the simpler IVW-H is tested on MuJoCo.
- The conclusion that variance, not bias, is the dominant issue in practice is based on only three MuJoCo tasks.
- The DDCG method largely follows the "test-and-fallback" template of AoBG, with the main contribution being a replacement statistical test. This limits the paper's methodological novelty.

### Questions
I have no more questions beyond those mentioned in weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

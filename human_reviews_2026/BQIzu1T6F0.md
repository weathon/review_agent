# A New Approach to Controlling Linear Dynamical Systems

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
We propose a new method for controlling linear dynamical systems under adversarial disturbances and cost functions. Our algorithm achieves a running time that scales polylogarithmically with the inverse of the stability margin, improving upon prior methods with polynomial dependence maintaining the same regret guarantees. The technique, which may be of independent interest, is based on a novel convex relaxation that approximates linear control policies using spectral filters constructed from the eigenvectors of a specific Hankel matrix.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a new algorithm for online control of linear dynamical systems (LDS) under adversarial disturbances and convex cost functions. The central idea is to use a spectral representation of control policies, transforming the non-convex control problem into a convex online learning problem using Hankel-based spectral filters. This allows efficient computation while preserving strong theoretical guarantees.

### Strengths
1. The spectral filters for marginally stable LDS can effectively compress the information of external disturbance. Thus, it allows efficient computation while preserving strong theoretical guarantees.

2. The work sounds solid, although I didn't check the detail proofs.

### Weaknesses
1. **The criticism on LQR is not accurate**

For modern industrial applications, the top three control algorithms are likely to be PID, LQR and MPC, despite that those theoretical assumptions are not satisfied. Moreover, the example of drone flying control system is not a good motivate example for the proposed approach. A simple reason is that those scenarios mentioned in this paper are currently handled by the above three algorithms in many commercial products.  

Also, the statement about the benefits of marginal stability is misleading: "...yields smoother, more energy-efficient control, useful in settings like robotics, thermal systems, and satellite dynamics". It can lead to small control effort if the open-loop system is unstable or marginally stable. If open-loop is very stable, then one needs extra control effort to reduce the stability margin. Due to its small robustness margin, marginally stability is often not desired in applications related to robotics and satellites. A common setup in those applications is a strong stable tracking controller together with a smooth reference generator. It can yield smooth control input while remaining strong robustness against disturbances.

2. **Some highly related works  are missing**

In Section 1.2, the first two paragraph dates back to the beginning of control theory, which is not necessary. On the other side, it missed some more recent and highly related works called **system level synthesis (SLS)** [1,2] in the control literature. From my understanding, the proposed approach is a special case of SLS, which belongs to a more general framework called **Youla parameterization**. Compared to SLS, the proposed approach has two new features: 1) in SLS setup, the cost function is often known while this work considers the online convex optimization setup; 2) SLS is for general stability while this paper has a tailed algorithm for marginally stable LDS. 

3. **Does the proposed approach guarantee stability?**

In this work, it only assumes that the cost function is convex and Lipschitz. Let us consider a linear cost on $x$ and $u$ and there is not constraint $\mathcal{K}$. Then, Algorithm 1 will drive the state to infinity.  Maybe it needs some additional assumptions, e.g., $c(x,u)$ is bounded from below.

4. **Lack of experiments on adversarial cost and disturbance**

This paper claims those benefits in the theory but no experimental illustration is provided. It only investigates simple disturbances including Gaussian,  Rademacher noise and deterministic sinusoidal. And the cost is a fixed quadratic.

[1] Anderson et al. System level synthesis. Annu. Rev. Control, 2019.

[2] Wang et al. A system level approach to controller synthesis. IEEE-TAC, 2019.

### Questions
1. Let us consider an extreme case where System (1.1) does not have any external disturbance (i.e. $w_t \equiv 0$). Then, under this setup, Line 7 of Algo. 1 will produce $\tilde{W}=0$. Moreover, whatever $M^t$ is optimized, there is no effective control input, i.e. $u_t=0$. Do those theoretic claims still hold?

2. Follow the previous question. If $w_t$ is small, how will the proposed algorithm behave? Will the loss decreases slower? Or the decay rate does not change?

3. The examples consider simple diagonal $A$ and dense $B$. How about the controllable canonical form where both $A$ and $B$ has certain sparse property? If there is one control input $u\in \mathbb{R}^{1}$ but many states $x\in \mathbb{R}^n$ with large $n$, then the effect on $x(n)$ is delayed by $n$ steps. Will it causing some large oscillations in this loss curve?

### Soundness
3

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
The paper proposes a new \emph{spectral convex relaxation} for online (adversarial) control of linear dynamical systems. 
It replaces explicit system-dependent features with a fixed, universal set of \emph{Hankel eigenvector filters}, 
thereby transforming control into an online convex optimization problem. 
The resulting algorithm achieves a regret bound of $\widetilde{\mathcal{O}}(\gamma^{-4}\sqrt{T})$, 
where $\gamma$ denotes the \emph{stability margin} of the comparator policies (i.e., the spectral radius bound of the closed-loop matrix). 
This dependency in $\gamma$ is better compared to existing approaches.
The method runs in polylogarithmic time per step and is supported by a detailed approximation analysis 
and experiments demonstrating a competitive runtime efficiency compared to existing baselines.

### Strengths
The paper is very well written and the overall presentation is clear and easy to follow. 
The structure and flow of ideas make it straightforward to understand the motivation, 
technical setup, and implications of the proposed approach. 
Both the algorithmic framework and the main theoretical result Theorem 2.1) are presented in a transparent way, 
with all relevant quantities and assumptions clearly defined. 
In particular, the paper explicitly spells out the concrete hyperparameter choices 
$m$, $h$, and $\eta$ that lead to the stated regret bound, 
which greatly improves the reproducibility and interpretability of the theoretical results. 
A further strength is the improved dependence on the stability margin $\gamma$ in the regret bound, 
which represents a significant advance over previous approaches whose computational or regret guarantees scale more unfavorably with $\gamma$. 
Finally, the discussion of related work is comprehensive and well organized, 
situating the contribution within the broader literature on online control and learning-based methods.

### Weaknesses
**Weaknesses and Suggestions.**

 1) *Reliance on known $(A,B)$ is unrealistic in many applications.*
  The entire pipeline assumes exact knowledge of the system matrices to reconstruct disturbances and build spectral features.
  This limits applicability to model-mismatch or unknown-dynamics settings.
  Suggestion: Add a discussion (or short appendix) on robustness to misspecified $(A,B)$ and outline a data-driven variant where
  $(\widehat A,\widehat B)$ are estimated online (or pre-estimated) with explicit stability conditions under identification error,
  e.g., bounds of the form $\|A-\widehat A\|,\|B-\widehat B\|\le\varepsilon$ and their impact on regret constants.

  2) *Placement of Theorem 2.1 (many forward references).*
  As written, Theorem 2.1 appears before the technical setup of Section 3, causing heavy forward referencing and making the statement hard to parse on first reading.
  Suggestion: Move Theorem 2.1 to immediately \emph{after} Section 3, once the comparator class, spectral filters, and surrogate loss are fully defined.
  This will make the result self-contained at its first occurrence.

  3) *Reorder Section 3 before the main results.*
  Closely related to the previous point: placing the entire Section~3 (setup, assumptions, and construction of the spectral controller)
  before the main theorem would improve readability and reduce back-and-forth navigation.
  Suggestion: New order: (i) problem setup and assumptions, (ii) spectral feature construction \& surrogate loss (current Section 3),
  (iii) then algorithm and main regret theorem.

  4) *No empirical verification of the $\sqrt{T}$ regret scaling in Theorem 2.1.*
  The current experiments do not test the $\sqrt{T}$ dependence, so they cannot confirm the rate empirically.
  Suggestion (add a minimal experiment):

   4a) Fix a stable LTI instance and cost sequence within the paper's assumptions (convex Lipschitz costs), and fix a stability margin $\gamma$ (e.g., by choosing a comparator with spectral radius $1-\gamma$).
    
4b) Run the proposed method for horizons $T\in\{2^{10},2^{11},\dots,2^{18}\}$, keeping all other hyperparameters at the prescribed values in Theorem 2.1 (the stated $m,h,\eta$ schedule).
   
4c) For each $T$, compute regret with respect to the comparator class used in the theory (or as close as feasible), average over multiple seeds, and report the mean $\pm$ std.
   
4d) Plot $\log(\text{regret})$ vs.\ $\log T$ and report the fitted slope (expectation: slope $\approx\frac{1}{2}$ within confidence bands).
    \item (Optional) Repeat for a few $\gamma$ values to visualize the $\gamma$-dependence of constants in the regret (even if asymptotically the slope stays $\approx\frac{1}{2}$).

### Questions
1) What can be said when $A,B$ are unknown (cf. weakness section above)? 
    Do the results extend to estimated $(\widehat A,\widehat B)$, and how do identification errors 
    $\|A-\widehat A\|$, $\|B-\widehat B\|$ affect the feature construction, stability, and regret guarantees?

2) When referring to a ``convex cost,'' do you mean convex jointly in $(x,u)$ for each $t$? 
    Please clarify the precise convexity and Lipschitz assumptions, including in which variables and norms they hold.

3) In the first displayed equation of Section 1.1, the policy class in the regret definition is missing. 
    Please specify explicitly over which set of policies or feedback gains the infimum is taken.

4) In Algorithm 1, the loss function $\ell_t$ is referenced but not defined at that point. 
    It only appears later in Equation (4.1), which interrupts the reading flow. 
    Please either define $\ell_t$ where Algorithm~1 appears or add a forward reference in the algorithm caption.

5) Definition 3.1: can the stated condition occur at any time $t$? 
    I assume yes, please make the time quantification explicit.

6) Definition 3.4: please be precise that $K$ denotes a feedback gain that induces a linear policy; 
    $K$ itself is not a policy. Consider rephrasing accordingly to avoid ambiguity.

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
2

### Summary
This paper proposes an online control algorithm for linear stochastic systems with adversarial disturbance. The proposed algorithm approximates a policy using spectral features derived from Hankel matrix eigenvectors, allowing efficient updates with regret comparable to existing approaches while significantly improving runtime. The author demonstrates that the proposed algorithm can match or surpass existing baselines empirically while using far fewer parameters.

### Strengths
- The use of spectral filtering in online control appears to be a novel idea that could have a broader impact on the control domain.
- The theoretical analysis is thorough and clearly demonstrates improvement over prior approaches.

### Weaknesses
- The empirical evaluation remains relatively narrow in scope. Including results on higher-dimensional systems would strengthen the paper.
- While the main advantage of the proposed algorithm lies in its runtime efficiency, this aspect is not evaluated in the experiments.

### Questions
- Could the authors provide more intuition behind the motivation for using spectral filters? What advantages do spectral features offer compared to other feature representations in online control?
- Could the authors comment on the actual runtime improvement over baseline methods in the experiments?

### Soundness
3

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
4

### Summary
This paper considers the problem of controlling a known linear dynamical system under possibly adversarial disturbances and time-varying costs. The authors propose an approach motivated by "spectral filters", wherein the eigenvalues/eigenvectors of a particular Hankel matrix determined by a user-set stability margin (over)estimate are used to parameterize a control law that maps past disturbances to controls. Compared to prior work in "non-stochastic control", by using this particular parameterization, the proposed method attains regret bounds that improve the dependence on the stability margin, as well as improve the run-time dependence on the stability margin from polynomial to polylogarithmic.

### Strengths
Overall, this paper is well-written and the message is rather clear. The proposed method is a novel adaptation of prior literature in both non-stochastic control and spectral filtering that seems to improve existing regret and runtime guarantees for this problem under relatively relaxed theoretical conditions. The approximation techniques involving the Hankel matrix are interesting, and as far as I know rather novel in the space of (linear) control. Thus, I think this paper contains worthy and interesting content for the niche of learning on dynamical systems researchers.

### Weaknesses
Some immediate points that deserve some discussion are as follows:

1. The results seem to rely on the dynamics matrices $A,B$ being known. How well do the proposed results extend to the unknown dynamics case?

2. The proposed method seems rather tied to linear dynamics parameterization. How does one naturally apply this method to non-linear dynamics/policy parameterization? On that note, the discussion of applying GPC and OSC to nonlinear systems (with potentially nonlinear parameterization) in experiment Section 5.1 is rather terse, and it is not immediately clear how the proposed methods are being applied---the appendix doesn't seem to have further information on this front.

3. The numerical experiments lack comparisons to even simple baselines. For example, considering some of the experiments concern control with Gaussian or zero-mean stochastic disturbances, it would be instructive to see the loss dynamics of, e.g. online LQR. Furthermore, in the known-linear-dynamics (potentially quadratic-loss case), there is quite a lot of literature on robust control, see e.g. classical mixed H2/H$\infty$ control, competitive control [1], and adversarially robust synthesis [2]. I think it is both worth including some discussion about robust control perspectives, and a couple more simple baselines beyond disturbance-action filter approaches to get a better sense of the method's relative performance.

4. As a minor comment, it should be noted the margins of the submission seem to have been different than the template.

[1] Goel and Hassibi, "Competitive Control"

[2] Lee et al. "Performance-Robustness Tradeoffs in Adversarially Robust Linear-Quadratic Control"

### Questions
Please see under Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

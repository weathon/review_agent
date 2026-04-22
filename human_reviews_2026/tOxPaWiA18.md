# The Effect of Temporal Resolution in Offline Temporal Difference Estimation

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 8, 4

## Abstract
Temporal Difference (TD) algorithms are the most widely employed methods in Reinforcement Learning. Notably, previous theoretical analysis on these algorithms consider the sampling time as fixed a priori, while it has been shown that the temporal resolution can impact data efficiency (Burns et al., 2023). In this work, we analyze the performance of mean-path semi-gradient TD(0) for offline value estimation, emphasizing the dependence on the temporal resolution, a factor that indeed proves to be of crucial importance. For continuous-time stochastic linear quadratic dynamical systems with a fixed data-budget, the Mean Squared Error in value estimation shows an optimal non-trivial value for the time discretization, and this choice affects the reliability of the algorithm. We also show that this behavior differs from that of the Monte Carlo algorithm (Zhang et al., 2023). We verify the theoretical characterization in numerical experiments in linear quadratic system instances and further demonstrate, in a stochastic control setting, that the step-size trade-off persists in policy iteration.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies the impact of temporal resolution on Temporal Difference (TD) learning algorithms. The contributed theoretical analysis, focused on the continuous-time linear quadratic setting, is rigorous and provides a clear, non-trivial result: the existence of an optimal discretization step h* that minimizes the mean-squared error and its insensitivity to the data budget B, different from prior results for Monte Carlo methods. The paper is well-structured and the numerical experiments adequately support the theoretical claims.

### Strengths
+this paper tackles an interesting and underexplored problem-understanding the impact of temporal discretization parameter h on TD learning
+this paper derives the MSE for offline mean-path TD(0) as a function of h and characterizes the optimal h* 
+comparison with the Monte Carlo results from Zhang et al. (2023) demonstrates the difference between the two estimation algorithms
+solid numerical experiments validate the theoretical findings

### Weaknesses
-this paper analyzes the offline mean-path semi-gradient TD(0) algorithm. This algorithm computes the exact expected update over the entire offline dataset at each step. Although this is a standard theoretical simplification (as in Bhandari et al., 2018), it sidesteps the variance introduced by stochastic sampling in practical TD(0)
-the analysis in Section 5 numerically shows that the algorithm converges to the LSTD solution (Eq. 10). However, the paper would be strengthened by a formal discussion of the assumptions required for this.
-the suggestion in Section 4.2 for a burn-in phase to find h* is viable. However, it relies on having a Monte Carlo estimate of the true value V to compute the empirical MSE. In most real-world problems, V is unknown, which is exactly why we use TD algorithms in the first place. Put differently, if one already has a good enough estimate of V to perform this tuning, why need TD? 
-the constants invovled in the optimal h* are complicated functions of the underlying unknown system parameters, and it is inviable to get h* without knowing the dynamics. 
-The analysis relies on a specific, well-chosen feature \phi(x), and it is not clear how the findings about the temporal resolution and the optimal h* depend on the given perfect representation and especially in the presence of learning approximation error?

### Questions
See above

### Soundness
3

### Presentation
2

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
The paper analyzes how temporal discretization affects TD(0) value estimation for continuous-time linear quadratic systems. Main claim is there's an optimal discretization step h*. Specific special cases are analyzed.

### Strengths
The question is interesting - temporal resolution is overlooked in RL so studying it systematically makes sense.

The empirical results clearly show the U-shaped curve for optimal h, which is the main point.

The comparison showing TD has constant h* while MC has h* ~ B^1/2 is a neat observation.

### Weaknesses
The scope is way too narrow. This is literally just:

1D Langevin dynamics 
One specific algorithm (mean-path TD, not even standard TD)
Quadratic features with zero approximation error
Offline only

How is anyone supposed to know if this generalizes? Linear quadratic systems are the simplest possible case. The authors admit this in Section 6 but it's a dealbreaker. Why not test on at least a 2D system or some standard RL benchmark before submitting?
Mean-path TD is non-standard. It updates using the mean gradient over the whole dataset (eq 4), not stochastic updates like regular TD. This is basically a batch method. So you're analyzing convergence to LSTD, not how good LSTD is, and this isn't how people actually use TD in practice.

The practical advice doesn't work. The optimal h* in Corollary 4.2 depends on constants C11, C12, etc that require knowing system parameters. But in real RL you don't know these! They suggest doing grid search over h with a small budget, but that means collecting data at multiple frequencies which is wasteful. And if you don't have a model, how do you know your estimated h* is any good?

The MC comparison is weird. Figure 6 claims TD "outperforms" MC but they use different horizons (T=8 for TD vs T≈52 for MC). That's not a fair comparison - you're giving the methods different amounts of information. Also MC is known to have high variance for infinite horizon so this isn't surprising.

Experiments only test the same 1D Langevin system with different parameters. No other dynamical systems, no higher dimensions, no standard benchmarks. The shaded regions in Figures 4-6 show pretty large variance too.

### Questions
For standard stochastic TD(0) (not mean-path), would you see the same behavior?

How sensitive is this to the feature choice? What if φ(x) was different?

Can you quantify *when* the B terms in eq 8 become negligible?

Why not test even just 2D Langevin to show some generalization?

This reads more like preliminary work that needs extension before publication. Can you could show similar tradeoffs exist in more realistic settings, or give better practical guidance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on continuous time decision making problems and tries to find the optimal temporal resolution to sample such a system to minimize the mean squared error (MSE) of a temporal difference (TD) based value estimator in the offline setting (where the data has been sampled upfront before value estimation begins). To do so it focuses on a class of problems that can be analyzed theoretically, namely continuous-time stochastic linear quadratic systems. The paper analyzes how the MSE of this offline semi-gradient TD algorithm with respect to the true value function changes as a factor of all the different variables in this system, and then characterizes the optimal temporal resolution for a given system. While the optimal temporal resolution would require knowing system dynamics upfront or evaluating multiple different resolutions, the paper proposes a practical approach to find the right temporal resolution by using a smaller batch based on the theoretical finding that the MSE is independent of batch size if it is large enough.

Finally, the paper validates its results and additional questions that the analysis brings up by evaluating a particular linear quadratic system with different parameters.

### Strengths
* The paper sets up a clear problem statement and a valid simplification that can be analyzed
* The theorems appear to be correct, although this reviewer mostly skimmed the proofs
* The paper presents the results of the analysis in a manner that is mostly easy to parse, specifically equation 7 and the subsequent explanation. These efforts will make it easy for an intelligent reader who is not immersed in the literature this paper builds upon to follow the paper and understand it.
* Subsection "How to choose temporal resolution for TD" is a good example of practical takeaways that this paper seeks to communicate.
* Deeper comparison to closest comparable work that this paper seems to build upon in Section 4.3 is also appreciated.
* The empirical evaluation seems to bear out the theoretical analysis presented in the paper.
* The presentation of the empirical evaluation is also very clear and well communicated.

### Weaknesses
* While it is understandable that the theoretical analysis requires a system like a linear quadratic one, it would be good to evaluate empirically if the findings hold for others.
* A practical example of whether the strategy suggested at the end of section 4.2, perhaps paired with an unknown dynamical system as mentioned in the previous point, would be useful.
* A little more detail in the captions for the Figures 2-6 would be helpful for a reader.
* Minor: figure 3 is placed after figures 4 and 5 in the paper

### Questions
For the result in Figure 6, since MC MSE is dependent on batch size B, if we increased B would we see it get closer to TD? Or would the improved estimation of TD outpace improvements to MC?

### Soundness
3

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
The paper analyzes how temporal resolution (step size $h$) impacts offline value estimation with mean-path semi-gradient TD(0) on continuous-time linear systems with quadratic rewards. For fixed data budget $B$, they derive a small-$h$ MSE expansion, $\mathrm{MSE}_t \approx C_0 + C_1 h + C_2 h^2$, implying an interior optimum $h^{\star}>0$; unlike Monte Carlo, where $h^{\star}$ scales with $B$. Experiments show TD iterates reach the LSTD fixed point, MSE is minimized at intermediate $h$, and the empirical $h^{\star}$ is largely insensitive to $B$.

### Strengths
- Focus on a concrete and important knob—temporal resolution—for offline TD value estimation.
- Tractable MSE expansion in $h$ with identifiable coefficients, yielding a closed-form $h^{\star}$.

### Weaknesses
- Results are for 1D Langevin/LQ with linear features that exactly span the value; unclear whether the same $h^{\star}$ behavior holds with function-approximation error, higher-dimensional or nonlinear dynamics, or non-quadratic rewards.  
- Mean-path semi-gradient TD(0) is analytically friendly but less standard than stochastic TD, TD($\lambda$), or GTD

### Questions
- Do you expect similar $h^{\star}$–vs–$B$ insensitivity for stochastic TD with constant stepsizes and iterate averaging, or for TD$(\lambda)$?

### Soundness
3

### Presentation
3

### Contribution
2

# Scalable Decision Focused Learning via Online Trainable Surrogates

- Decision: Reject
- Scores: 4, 2, 4, 2

## Abstract
Decision support systems often rely on solving complex optimization problems that may require to estimate uncertain parameters beforehand.
Recent studies have shown how using traditionally trained estimators for this task can lead to suboptimal solutions.
Using the actual decision cost as a loss function (called Decision Focused Learning) can address this issue, but with a severe loss of scalability at training time.
To address this issue, we propose an acceleration method based on replacing costly loss function evaluations with an efficient surrogate.
Unlike previously defined surrogates, our approach relies on unbiased estimators -- avoiding the introduction of spurious local optima -- and can provide information on its local confidence -- allowing to switch to a fallback method when needed.
Furthermore, the surrogate is designed for a black-box setting, which enables compensating for simplifications in the optimization model and accounting for recourse actions during cost computation.
In our results, the method reduces extremely costly inner solver calls, while keeping a solution quality comparable to other state-of-the-art techniques.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a scalable framework for decision-focused learning (DFL) in black-box settings. The core idea is to replace repeated solver calls with a Gaussian-process-based surrogate that provides differentiable regret estimates. The method combines random smoothing, importance sampling, and confidence-based fallback to balance efficiency and accuracy

### Strengths
- The paper addresses a clear and practical bottleneck in DFL: the high cost of repeated solver calls during training. The motivation is concrete and connects well to real-world applications where the optimization process is computationally expensive.
- The surrogate approach is conceptually consistent. Using a Gaussian Process (GP) with confidence estimation provides a unified way to obtain differentiable regret approximations and to decide when to rely on the black-box solver.
- The combination of random smoothing and importance sampling mitigates the zero-gradient issue and reuses samples efficiently. This makes the approach easy to integrate into existing DFL pipelines.

### Weaknesses
- The conceptual novelty is limited. The proposed approach combines several known techniques, the stochastic smoothing, the confidence-based fallback, into a single framework. Each component is well understood, and the paper does not demonstrate a fundamentally new principle beyond combining them coherently.
- The claim of “unbiasedness” is not well-justified. Theoretical support for unbiased gradient estimation through a GP surrogate under random smoothing and importance sampling is only stated qualitatively. Without formal proofs or convergence analysis, the reliability of the surrogate’s gradient remains uncertain.
- The hidden computational cost is understated. Sample sharing and similarity computation require a large set of solver evaluations during initialization. This preprocessing step might offset part of the claimed efficiency gains, especially in high-dimensional decision spaces.
- The framework heavily relies on the fallback to the SFGE baseline. The performance improvements may come more from the gating mechanism than from the surrogate itself. There is no comparison with alternative fallback strategies, so it is unclear whether the method’s advantage is robust.
- Some results are not well explained. The sensitivity experiments show that increasing the aggressiveness of the surrogate (larger β) leads to lower regret, which contradicts intuition. The authors only speculate on possible causes but provide no deeper analysis.

### Questions
- What exactly does “unbiasedness” refer to in this context? Is the surrogate’s estimate of the smoothed regret unbiased, or is the gradient of the regret unbiased with respect to the true loss? Under what assumptions does this hold for a GP trained with random smoothing and importance sampling?
- How is the confidence threshold β chosen in practice? Could it be learned adaptively rather than fixed, perhaps using an acquisition-function-like criterion? Would such an adaptation explain the counterintuitive result that larger β improves regret performance?
- How large is the preprocessing cost of generating shared samples and computing similarity matrices? How does the total computational budget compare with that of a plain black-box DFL under the same number of solver calls?
- What would happen if the fallback solver were replaced with another baseline (e.g., SPO+ or a directional gradient method)? Would the surrogate still yield consistent improvements, or is the benefit tied to SFGE’s specific behavior?

### Soundness
2

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
3

### Summary
This paper proposes a method to make Decision-Focused Learning scalable by avoiding expensive inner solver calls during training. The authors introduce an online-trainable surrogate model—specifically, a Gaussian Process—that approximates the regret function for each training instance.
Each surrogate is updated as new predictions are evaluated, and used to provide approximate gradients when confident; otherwise, a fallback black-box DFL method is called. To handle zero-gradient regions, they apply stochastic smoothing and importance sampling over previously computed regrets, and allow sample sharing across similar surrogates.
Empirically, the proposed method reduces solver calls and runtime on synthetic DFL benchmarks, such as Knapsack, Weighted Set Multi-Cover and a fully synthetic optimization task, while roughly matching or sometimes improving decision quality relative to existing black-box DFL approaches (LODL, EGL, LANCER).

### Strengths
- The paper tackles an important and practical challenge: DFL’s poor training scalability due to repeated solver invocations, and try to build on a previous approach which also tackle the DFL problems by learning surrogate regret losses.
- Experiments are thorough, and the code has been released in the supplemental material.

### Weaknesses
- Core Idea. 
The method replaces the original and computational expensive regret loss for each training instance with its own trainable GP surrogate, which is repeatedly refined by evaluating that same expensive loss whenever confidence is low. This design essentially amounts to a brute-force caching or regression scheme over previously computed losses
- Lack of generalization.
Training an independent surrogate per sample prevents information sharing and limits the method’s ability to generalize across contexts; the proposed “sample sharing” heuristic partially mitigates this but at the cost of bias and added complexity.
- Scalability ceiling.
GPs have O(n^3) training cost, and maintaining one per instance (plus KDE samples) is unlikely to scale beyond small datasets; no sparse or approximate GP strategy is provided.
- Limited and artificial evaluation.
Benchmarks are small synthetic combinatorial tasks; no demonstration on real industrial or continuous problems where scalability truly matters.

### Questions
How does this approach fundamentally differ from memoization or regression over cached solver outputs? What prevents it from degenerating into expensive per-sample bookkeeping? 

The method maintains a separate GP surrogate per training instance. How does this scale to larger datasets or higher-dimensional settings? Would a shared conditional surrogate $r_\phi(x, \hat y)$ that generalizes across instances be more efficient or principled?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper advances the surrogates literature related to decision focused learning and differentiable optimization. Specifically, it proposes to learn a GP as the surrogate model for regret and then draw more samples to update the GP if training dynamics hit a region where the posterior uncertainty is too high.

### Strengths
Surrogate methods are attractive because they are agnostic to the specific kind of optimization problem at hand, and this class of methods has attracted several attempts in recent years. This paper appears to improve fairly uniformly over them. I am particularly convinced by the results in the appendix that EGL (the best baseline) has inconsistent performance across its different variants, while the proposed method lacks this hyperparameter (the loss function family) and performs consistently well. This is a significant advantage in practice since robustness is a big issue for decision focused learning methods.

### Weaknesses
Arguably, this paper is mostly a direct modification of previous ideas: make the family for the surrogate a GP instead of previously proposed parametric families. This is not necessarily a bad thing -- simple ideas that are easily implementable and improve performance can be valuable -- but it places more emphasis on the empirical results being robust. I think that the results right now are short of a fully convincing picture but that this is very fixable with some new experiments.

First, the results are on a limited set of datasets, several synthetic or "toy" in some way. I think this is because the authors focus on a problem variant with uncertainty in the constraints, but I don't see why the proposed methods are specific to this setting. The paper would be greatly strengthened by running on a larger set of benchmarks including more real datasets, like those used in previous work. 

I would also appreciate an ablation that tested the impact of adding new samples during training, vs committing the entire sampling budget to the pretraining stage, and some analysis of how the performance of the method varies as a function of the parameter that determines when to draw a new sample. The idea of being able to dynamically decide when to draw new samples is intriguing, and a potential advantage of the GP, but I don't currently see an assessment of whether that is driving any improvement in performance.

### Questions
Did I miss the ablation somewhere?

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
2

### Summary
This paper deals with methods of decision focused learning (DFL) which can suffer from computational drawbacks, namely issues in scalability during training. This method offers improvements in training time by introducing a surrogate loss function using Gaussian Processes to approximate the regret for each point. This reduces the computational overhead and is black-box to the problem, allowing for improved training performance.

### Strengths
This paper combines multiple statistical tools in interesting ways to approximate the regret landscapes. It appears that the method can outperform most of the baselines in terms of regret/function calls tradeoff.

### Weaknesses
There is some convincing that needs to be done to justify this method. Importantly, while it performs well empirically on the experiments, there is a lack of real-world data experiments as they all seem to be synthetically generated. For a practical method that is meant to improve computational performance, it ought to be demonstrated in practice. Secondly, there are no theoretical properties of the surrogate loss function. It would be desirable to have consistency properties of the surrogate loss. Lastly, although speedups are important, it would be good to justify when they are necessary, as once decision-focused methods are trained, they are lightweight in usage, thus training overhead might be permissible.

### Questions
All the methods lose significantly in regret to SPO+ for the experiment it is included; is there any explanation for why? In settings where SPO+ is applicable, should this be expected?

### Soundness
2

### Presentation
2

### Contribution
2

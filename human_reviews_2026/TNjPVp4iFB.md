# Sharpness-Aware Minimization Can Hallucinate Minimizers

- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
Sharpness-Aware Minimization (SAM) is a widely used method that steers training toward flatter minimizers, which typically generalize better. In this work, however, we show that SAM can converge to hallucinated minimizers---points that are not minimizers of the original objective. We theoretically prove the existence of such hallucinated minimizers and establish conditions for local convergence to them. We further provide empirical evidence demonstrating that SAM can indeed converge to these points in practice. Finally, we propose a simple yet effective remedy for avoiding hallucinated minimizers.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper focuses on a critical failure mode of Sharpness-Aware Minimization — hallucinated minimizers. Through theoretical analysis and experimental validation, it reveals the convergence defect of SAM in non-convex scenarios and uses a simple switching strategy to address this issue.

### Strengths
1. This paper completes strict proofs in non-convex and high-dimensional scenarios, covering isolated/non-isolated local maximizer sets.

2. It verifies the effectiveness of the switching strategy by sufficient experiments.

### Weaknesses
1. The primary advantage of SAM lies in its generalization ability rather than convergence. Thus, whether SAM converges to the true minimizers of the training loss is not a critical issue for understanding its behavior.

2. Even considering convergence, this issue should be addressed straightforwardly by decaying $\rho$ to zero during training.

3. The theoretical result is unsurprising, as the loss landscape is highly non-convex.

4. The switching strategy (SGD → SAM) is not novel; it has been extensively studied both theoretically and experimentally in prior works (e.g., Andriushchenko & Flammarion, ICML 2022; Zhou et al., ICLR 2025).

### Questions
See Weaknesses.

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
4

### Summary
This paper shows that SAM may converge to hallucinated minimizers, which can affect convergence and degrade performance. The authors prove the existence of such hallucinated minimizers and demonstrate that they can act as attractors. They conduct experiments to support their claims and suggest avoiding these hallucinated minimizers by switching from SGD to SAM.

### Strengths
This paper presents a novel perspective by identifying the existence of points where the gradient of the SAM objective vanishes, yet which are not stationary points of the original objective. The authors refer to these as hallucinated minimizers. Such points may be far from true minima and even close to maxima, thereby degrading performance. The paper provides a theoretical proof of their existence and discusses the conditions under which they can become attractors. Inspired by this theoretical insight, the authors suggest switching from SGD to SAM to address this issue, and the experimental results indicate that this can effectively alleviate SAM’s sensitivity to ρ. Overall, the exposition is reasonable and may contribute to a deeper understanding of SAM.

### Weaknesses
There exist points in neural network functions that are stationary for the SAM objective but not for the original objective, which seems rather intuitive. Although the authors make some effort to formalize the notion of hallucinated minimizers, from a technical standpoint I do not see a particularly significant contribution. Merely proving their existence does not provide much insight. Another main result of the paper is to show that hallucinated minimizers can become attractors. However, this requires quite strong regularity conditions and only demonstrates that the algorithm may converge to hallucinated minimizers; whether and when such convergence actually occurs in general case remain unclear.

The authors conduct several experiments to support their claims, but I find them unconvincing. Specifically, they argue that convergence to hallucinated minimizers is common for SAM in deep learning, yet I only observe divergence when the parameter ρ is relatively large. Unlike the toy examples, the authors do not report gradient norms or visualize the loss landscape, making it difficult to attribute the convergence failure to hallucinated minimizers. Moreover, I have not seen reports of SAM failing to converge in the literature, provided that ρ is chosen appropriately in practice.

The authors propose switching from SGD to SAM to address convergence failures, but this argument does not seem entirely coherent. They claim that hallucinated minimizers often occur near local maxima, and hence
are more likely to occur in high-loss regions.; however, I do not understand why a higher loss necessarily implies proximity to a local maximum. This seems to depend on local curvature rather than loss magnitude, suggesting that the authors may have conflated these concepts.

### Questions
I did not check the proofs line by line, but the authors state that “Theorem 2.1, whose proof requires the perturbation radius ρ to exceed the distance between a minimizer and a maximizer.” It should be noted that in high-dimensional spaces, distances typically grow with dimensionality. Does this imply that the theorem would require a larger ρ in high-dimensional settings, and thus is less likely to hold in deep learning scenarios?

In the experiments, the authors show that switching from SGD to SAM improves performance. Why does this differ from Figure 8 in Andriushchenko & Flammarion (2022), where they found that such a switch had an insignificant or even negative effect? Have the authors tried performing such a switch at different percentages?

### Soundness
2

### Presentation
2

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
This paper offers a highly original and rigorous analysis of SAM’s training dynamics near local maxima, a regime largely overlooked in prior work that has centered on saddle points. Under a clear and well-scoped setting, the authors prove that when local maxima exist, the algorithm can converge to hallucinated attractors; they precisely characterize the structure of these hallucinated local minima and establish their attractive dynamical properties. The proofs are cleanly organized, with the chosen mathematical tools applied appropriately. The exposition is exemplary—definitions and assumptions are explicit, intermediate lemmas are well-motivated, and the narrative flow makes technically demanding arguments easy to follow.

### Strengths
- The paper presents a theoretically complete and logically consistent treatment of the chosen problem, offering clear insights and rigorous analysis.
- Unlike prior studies that mainly focus on saddle points, this paper analyzes the failure mechanisms of SAM around local maxima, thereby broadening the scope of understanding.
- The authors establish the existence of hallucinated minima and provide a systematic characterization of their dynamics and the mechanisms that induce them
- The flow from definitions to assumptions, theorems, and lemmas is logically coherent, and the core ideas are clearly articulated, allowing readers to follow even complex arguments with ease.

### Weaknesses
- The significance of the problem itself is not convincingly justified, raising doubts about its broader relevance and practical impact.
- A key limitation of the theoretical analysis is its reliance on strong, idealized assumptions. The framework presumes the existence of isolated local maximizers and real-analytic loss functions, and requires a perturbation radius that aligns with the specific geometry of the landscape. These conditions are rarely met in the high-dimensional and non-analytic nature of modern neural networks trained with stochastic optimization.
- The most detailed analyses and visualizations are confined to a simple 2-layer MLP on MNIST. The ResNet-18/CIFAR-100 results, while suggestive, lack deeper probes of the loss landscape or properties of the converged points. Broader coverage across architectures, datasets, and realistic ρ ranges would strengthen the empirical case
- The “hallucination” behavior, though interesting, appears modest in commonly used SAM regimes and is not yet shown to materially affect performance across realistic workloads. Clearer quantification of effect sizes and frequency under standard training settings would help establish practical impact.
- Although the phenomenon is intriguing, its real-world impact remains under-substantiated; stronger evidence across architectures and tasks is needed to establish practical relevance.

### Questions
1. One advantage for using SAM is its robustness to label noise. Your paper demonstrates a significant failure mode of SAM in a clean-label setting. Does this failure mode—converging to hallucinated minimizers—persist under noisy label conditions, or is it mitigated in the very regime where SAM is considered most beneficial?
2. Your central hypothesis is that the SGD→SAM switching strategy is effective because SGD avoids high-loss regions near local maximizers, which are the source of hallucinated minimizers. However, the paper lacks direct empirical evidence that local maximizers are a significant feature of the initial loss landscape. How can you empirically verify the existence and prevalence of local maximizers near random initialization for your experimental setups

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
The paper shows that Sharpness-Aware Minimization can converge to “hallucinated minimizers” — points that minimize the SAM surrogate but are not stationary, let alone minimizers, of the original loss.
The authors prove that such spurious attractors exist in very general non-convex, real-analytic settings, describe their manifold structure, and confirm their appearance in deep-net experiments with both full-batch and stochastic gradients.
To avoid them they propose a trivial “GD-first, SAM-later” switching rule that keeps large perturbation radii and improves both training stability and test accuracy.

### Strengths
1. The use of a constructed smooth non-convex function (Figure 1) effectively demonstrates how the SAM objective can create spurious minima not present in the original function, making the phenomenon intuitive.

2. Beyond identifying the problem, the authors propose a practical fix—using a damped version of the gradient—to prevent convergence to hallucinated minimizers, enhancing the reliability of SAM in real applications.

3. The paper provides a rigorous theoretical proof that SAM can converge to non-minimizing points (hallucinated minimizers), which challenges the common assumption that SAM always guides optimization toward meaningful solutions.

### Weaknesses
1. While the paper includes empirical results, they are primarily based on simple synthetic examples or small-scale models; more extensive validation on large-scale deep learning tasks would strengthen the practical relevance.

2. The proposed remedy is simple and effective in theory, but the paper does not thoroughly compare it against other SAM variants or alternative flatness-promoting methods, limiting the assessment of its relative advantages.

3. In the last part of this paper, authors choose s to use plain gradient descent for the first 10% of training iterations. There is a lack of quantitative metrics to determine when to switch to the SAM method.

4. In Theorem 3.2, the authors claim that a poor initialization may lead to hallucinated minimizers. However, there is no theoretical analysis indicating that plain gradient descent is more likely to yield a good initialization than SAM.

### Questions
1. If the  perturbation radius pho is small enough, will the hallucinated minimizers vanish?

### Soundness
3

### Presentation
3

### Contribution
3

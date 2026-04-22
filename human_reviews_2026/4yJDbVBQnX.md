# SHANG++: Robust Stochastic Acceleration under Multiplicative Noise

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Training with multiplicative noise scaling (MNS) is often destabilized by momentum methods such as Nesterov's acceleration, as gradient noise can overwhelm the signal. A new method, SHANG++, is introduced to achieve fast convergence while remaining robust under MNS. With only one-shot hyper-parameter tuning, SHANG++ consistently reaches accuracy within 1% of the noise-free setting across convex problems and deep networks. In experiments, it outperforms existing accelerated methods in both robustness and efficiency, demonstrating strong performance with minimal parameter sensitivity.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose SHANG++, a novel optimizer for empirical risk minimization, particularly in the context of ANN training.
The method is claimed to offer increased robustness to multiplicative gradient noise.
SHANG++ builds upon the SHANG algorithm, providing faster theoretical convergence and enhanced noise robustness.
The authors present a detailed convergence proof and evaluate their optimizer on the computer vision datasets MNIST and CIFAR-10/100.

### Strengths
- Developing more noise robust optimizers with low HP sensitivity is a problem of great interest.

### Weaknesses
- The paper's core idea appears to rely heavily on HNAG by Chen & Luo (2021), however, their work is not adequately discussed. For readers unfamiliar with HNAG (such as myself), Equation 2.1 is therefore difficult to interpret. As a result, both the motivation and the novel contribution of the paper remain unclear.
- The theoretical section is very difficult to follow. Rather than devoting multiple pages to dense technical proofs and equations, the paper should first clearly state its assumptions, theoretical setting, and main conclusions in the body text. The detailed derivations and proofs can be moved to an appendix. While the proofs themselves might be sound (which I did not check in detail), the overall theoretical context is insufficiently explained.
- The empirical evaluation is weak. Using LeNet-5 on MNIST is outdated and does not meaningfully demonstrate the optimizer's potential for modern applications. Although experiments on CIFAR are somewhat more relevant, additional experiments on larger models and datasets (e.g., a small ViT on ImageNet) would make the evaluation more convincing. Moreover, the baseline results are unreasonably poor. Even though I am not sure about the impact of the low batch size, LeNet-5 trained with SGD should reach around 99% accuracy on MNIST, whereas your reported 91% is close to the performance of a linear classifier. Similarly, achieving only 68% accuracy for ResNet-34 on CIFAR-10 suggests serious implementation or training issues. A strong and competitive baseline must be established before claiming improvements from SHANG++.
- The presentation of results is very bad. Many figures lack axis labels (e.g., it is unclear what is plotted on the x-axis of Fig. 3.1). The titles read like folder names. The figures and especially the legends are very small. Figures 3.1 and 3.2 should use the full page width for better visibility. Table 3.1 is overly crowded and unnecessarily precise. Showing three decimal places adds no value, as the first decimal already lies within the reported standard deviation.
- Table 3.2 and Figure 3.3 appear redundant. The baseline optimizers (SGD, NAG, Adam) should be included in these comparisons as well. Testing the main claim of paper, the robustness of SHANG++ to multiplicative gradient noise, only on CIFAR10 is insufficient.
Testing the claimed robustness of SHANG++ to multiplicative gradient noise solely on CIFAR-10 is insufficient. This experiment should be repeated on more and ideally larger datasets, especially given the paper's conclusion that SHANG++ is a "practical optimizer for large-scale noisy training."
- There is no "Related Work" section.

### Questions
1. What is the methodical difference to SHANG?
2. What is shown on the x-axis of Fig 3.1?
3. It looks like SGD reaches to lowest test loss for ResNet-50 on CIFAR100 (Fig. 3.2 bottom right). This is not discussed and not reflected in the results in Tab 3.1.
4. Why did you choose $\mu = 0$ and $\beta = \alpha / \gamma$ for your experiments?
5. How did you optimize your HPs? On a validation set or test set? You mention that you choose $\gamma$ From different intervals for LeNet/ResNet. That looks like a preselection of the HP, which should be part of the HP optimization in first place. How did you come up with these intervals?
6. Why do you increase $\gamma$ when you reduce the learning rate?
7. In your "one-shot" protocol, how did you choose the HPs? Even if you fix them, you have to chose them somehow.
8. What happened with AGNES at $\sigma = 0.05$ in Tab 3.2/Fig 3.3?
9. Does the faster convergence of SHANG++ compared to SHANG also hold in experimental settings?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The submission has several presentation and formatting issues that make it difficult to read. The paper clearly does not follow the official ICLR template: font style and layout are inconsistent with other submissions, section headings are poorly rendered (some almost unreadable), and the title is not in uppercase as required. The structure of the paper is disorganized, with unclear section hierarchy. Symbol usage is also confusing—for example, weights are denoted by *x* while samples are denoted by *XY*, without proper definition. Figures are overly compact and hard to interpret; adding zoomed-in views would help. Some references appear mismatched with the text (e.g., line 424 refers to Appendix A.4, which does not match its description).

On the theoretical side, the claimed innovation is limited. While SHANG++ improves stability and simplifies tuning compared to SHANG, the methodological advance over AGNES and SNAG is incremental. The main technique—stochastic HNAG discretization with noise suppression—appears to be a modest extension of Chen & Luo (2021) and largely builds on existing ideas. The paper also lacks discussion of heavy-tailed or multiplicative noise settings as analyzed by Hodgkinson & Mahoney (2021), which constrains the scope of its theoretical contribution. The authors should more clearly highlight the key technical challenges and their unique insights.

Theoretical coverage is narrow, restricted to convex and quadratic objectives, whereas most experiments are conducted on nonconvex networks such as CNNs and U-Nets. Without a formal extension to nonconvex landscapes, the theoretical results do not fully support the empirical claims regarding robustness and stability in deep learning.

Empirically, the experiments are limited to small-scale image datasets. There are no evaluations on larger models (e.g., ViT) or datasets (e.g., ImageNet), nor on NLP tasks, which limits the generality of the conclusions. Although the paper claims improved computational efficiency, no quantitative results (training time, memory footprint, or iteration latency) are provided. It is also unclear whether Table B.1 correctly labels SHANG vs. SHANG++. Additional experiments with very small batch sizes (e.g., 16 or 32) would strengthen the argument.

Hodgkinson L, Mahoney M. Multiplicative noise and heavy tails in stochastic optimization[C]//International Conference on Machine Learning. PMLR, 2021: 4262-4274.

### Strengths
see Summary

### Weaknesses
see Summary

### Questions
see Summary

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a (Gauss-Seidel-type) discretization, named SHANG++, of a Hessian-driven momentum ODE (HNAG), where the gradient is replaced by a biased estimator under multiplicative noise. The two theoretical results aim to recover similar rates as for the classical momentum algorithms in the (quadratic)-strongly convex case (Theorem 2.1) and in the convex case (Theorem 2.2). Several experiments in Section 3 aim at showing that SHANG++ is more robust to noise and converges faster than former algorithms used or introduced in the context of multiplicative noise.

### Strengths
The idea of finding new discretizations of Heavy ball/Nesterov-like ODE, that combine theoretical guarantees in convex optimization settings while improving empirical effectiveness on learning task is promising. This paper could be a step in this direction. The figures presented in the paper look encouraging about the benefit of the method.

### Weaknesses
If the paper shows potential, I believe it is not yet mature enough for acceptance in its current form.

About the theoretical part:

- My main concern is about Theorem 2.1. In its statement, the quantity over which a control is given involves a non positive term, namely $f_{\mu}(x) := f(x) - \frac{\mu}{2}||x-x^\ast ||^2$. Using properties of strong convexity, the best lower bound we can get is $f_{\mu}(x)-f_\mu(x^\ast) \ge 0$, such that we can not deduce a rate for $f(x)-f(x^\ast)$. In its current state, Theorem 2.1 does not provide a convergence of the quantity $f(x)-f(x^\ast)$, such that is seems hard to state its significance.

- The Bregman divergence is defined on page 2. It appears at some point in the proofs, but surprisingly it seems to be useless as it does not appear in the algorithm nor in the result, all Lyapunov making use of the classical Euclidian norm instead. I think it should be clarified.

- Corollary C.3 (page 27) is given without proof.


About the numerical experiment part:

If the figures show a benefit of SHANG++, it seems that the code given in the supplementary material does not match the figure presented in the paper. I checked the code, and I could not find the implementation of "AGNES" and "SNAG", which serve as baselines in many figures of the paper. I have run the file "convex\_example.py", which very likely should generate Figure 3.1. It generated only 2 curves (versus 6 for Figure 3.1), labeled "SHNAG" and "ISHNAG", which do not correspond to anything in the paper, except appearing in the label of Figure A.6. For now, the experiments of the paper are not reproducible, and the provided code seems unfinished. I believe the exact code that generated the figures should be provided.

Remarks on presentation

- I think the different paragraphs/statements should be more clearly separated. It could simply be done by the use of bold font for names of paragraphs, sections and things as "Theorem" etc.

- The notation $\kappa$ is used on page 1 but is not defined. 

- I think (Hermant et al. (2025)) should not be presented as the work that introduces SNAG. As they mention themselves, it is a stochastic version of a classical Nesterov algorithm. Algorithms of this kind at least exist since (Nesterov, 2012). 

- On pages 8-9, there is an enumeration of 3 points to comment on Table 3.2 and Figure 3.3. Point 1 is commented using percentage, point 2 using "pt", and point 3 using percentage again. I believe it can be confusing.

Reference:
Nesterov, Efficiency of coordinate descent methods on huge-scale optimization problems, Journal on Optimization, 2012

### Questions
- On page 8, what is the "mean Top-1 error" ? It seems to be not defined.

- Could you be more explicit about your notation $z_k^+$ ?

### Soundness
2

### Presentation
1

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
This paper introduces SHANG++, a first-order stochastic optimization method that achieves accelerated convergence while being robust to multiplicative noise. Motivated by the instability of standard momentum methods like Nesterov's Accelerated Gradient (NAG) under the Multiplicative Noise Scaling (MNS) condition, the authors propose an algorithm derived from the discretization of the Hessian-driven Nesterov Accelerated Gradient (HNAG) dynamical system. The theoretical analysis is sound and delivers state-of-the-art convergence rates under the MNS condition. And the theoretical claims are backed by a convincing set of experiments.

### Strengths
1. The paper does an excellent job of positioning itself within the recent literature on accelerated methods under MNS. It identifies a clear and important gap and presents its historical path in the introduction.
2. The derivation of SHANG++ from the continuous-time HNAG flow is elegant and provides a principled foundation for the algorithm's design. The connection to a dynamical system offers clear intuition for its components. 
3. The state-of-the-art theoretical guarantees in analysis is a major strength.
4. The experiments are thorough and directly support the paper's claims.
5. The paper is very well-written and easy to follow. The notation is standard, and the core ideas are communicated effectively.

### Weaknesses
While the paper is very strong, there are a few minor points that could be clarified or strengthened:

1. As the authors correctly state in the limitations, the convergence guarantees are for convex objectives. While this is standard for this line of work, the empirical success on highly non-convex deep learning tasks is not fully explained by the theory. A brief discussion on potential avenues for non-convex analysis or intuitions for why the convex-case stability properties might transfer (e.g., behavior in locally convex basins) would be welcome, though not essential.
2. Connection to SNAG: The appendix shows that SNAG can also be viewed as a discretization of the same HNAG flow. This is a very interesting connection. It would be beneficial to bring a summary of this insight into the main paper.
3. The paper does not explicitly discuss the computational cost of SHANG++ per iteration. It appears to be identical to standard momentum methods (one gradient evaluation, a few vector additions/scalings per step), but it would be good to state this explicitly for completeness.
4. The paper does not comment on HNAG (Chen & Luo, 2021) in the literature review, until line 075 introduces SHANG++ based on HNAG. It would be interesting to see how HNAG positions itself in the history to the SGD (with MNS condition). What motivates the authors to derive the SHANG++ from HNAG, which seems not directly (at least not presented in the Introduction) to the MNS studies.

### Questions
1. The noise-damping parameter m is central to SHANG++. The experiments show m=1.5 is a good default. Did you observe a trade-off where a very large m might over-damp the updates and slow down convergence, even if it enhances stability?
2. Regarding the implementation in Algorithm 1, the parameter β from the theoretical section is implicitly set to α/γ. This seems to be a practical heuristic. How does this choice relate to the optimal β derived in the theoretical analysis (e.g., β = (1+σ²)α/μ in the strongly convex case)? Is it possible that learning or scheduling β could lead to further improvements?

### Soundness
4

### Presentation
4

### Contribution
3

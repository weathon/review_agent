# Scalable Element-wise Finite-Time Optimization for Deep Neural Networks

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Optimization algorithms are fundamental to deep neural network training, where exponential growth from millions to hundreds of billions of parameters has made training acceleration a critical necessity. While adaptive methods like Adam achieve remarkable success through element-wise learning rates, understanding their continuous-time counterparts can provide valuable theoretical insights into convergence guarantees beyond asymptotic rates.
Recent advances in continuous-time optimization have introduced fixed-time stable methods that promise finite-time convergence independent of initial conditions. However, existing approaches like FxTS-GF suffer from dimensional coupling, where coordinate updates depend on global gradient norms, creating suboptimal scaling in high-dimensional problems typical of deep learning.
To address this issue, we introduce an element-wise finite-time optimization framework that eliminates dimensional coupling through coordinate-independent dual-power dynamics. Furthermore, we extend the framework to momentum-enhanced variants for deep model training while preserving convergence properties through continuous-time analysis. Under mild assumptions, we establish rigorous finite-time and fixed-time convergence guarantees. Notably, our framework reveals that widely-used sign-based optimizers like SignSGD and Signum emerge as limiting cases, providing theoretical grounding for their empirical effectiveness. Experiments on CIFAR-10/100 and C4 language modeling demonstrate consistent improvements over existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper builds upon the work of Budhraja et al. by introducing an ODE that exhibits finite-time convergence guarantees. Compared to previously studied finite-time gradient flows, the proposed methods enjoys coordinatewise adaptivity and can be seen as a generalization of signSGD. A proof of finite-time and fixed-time convergence is provided in the smooth and PL regime, and the results are shown to hold for variants incorporating exponential moving averages and momentum. In addition, the empirical effectiveness of the proposed optimizer is explored on vision and pretraining tasks.

### Strengths
1. The paper is clearly written and easy to follow.
2. The idea of extending existing finite-time gradient flows to be coordinatewise adaptable is well-motivated.
3. The hyperparameter conditions for finite-time and fixed-time convergence are clearly stated in the theorems.

### Weaknesses
1. The method itself, equation (2), is not intuitively motivated or explained. For instance, it is not immediately obvious what $p_1$ and $p_2$ are supposed to represent, even though they are of critical importance to the effectiveness of the method.
2. The proposed optimizer introduces a slew of hyperparameters, for which there is little discussion of practical recommendations or tuning suggestions.
3. The theoretical results are given for the continuous-time dynamics, and no discussion is provided on how these results transfer to the discretized versions. The paper lacks the results that are expected of a typical optimizer paper, e.g. a convergence rate on the gradient norm or objective value.
4. No empirical evidence is provided to support the claim that the method achieves finite-time convergence. If the continuous-time results do indeed transfer to the discrete-time setting, then such an empirical result would greatly support the effectiveness of the proposed method.
5. A central claim of the paper is that the proposed method is scalable, but there is little evidence to support this with the provided experiment results. I would suggest providing results on models with 7B+ parameters and evaluating on tasks that are significantly harder than CIFAR.

I did not carefully review the proofs, but unfortunately, the mentioned issues are already significant enough for me to recommend rejection.

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces an Element-wise Finite-Time (EFT) optimization framework and its momentum variants for deep neural network training. The core motivation is to adapt control-theoretic finite-time and fixed-time stability concepts of ODEs to large-scale deep learning. The proposed EFT framework replaces this global norm dependency with element-wise operations based on the sign and fractional powers of individual gradient components.

### Strengths
Problem Relevance: The paper addresses a practical and important issue: how to apply optimization theories with stronger convergence guarantees (finite/fixed-time) to large-scale, high-dimensional deep learning tasks, overcoming the scalability bottlenecks of existing methods.

Experimental Results: The paper presents preliminary evidence showing that the proposed methods (especially the momentum variants) outperform some baseline optimizers (including SGD, AdamW, and FxTS-GF) on benchmarks like CIFAR image classification and C4 language modeling, indicating its potential.

### Weaknesses
Major ones: 

1. Borrowed Core Idea: The core theoretical framework using dual-power dynamics (combining terms with exponents less than 1 and greater than 1) to achieve finite/fixed-time convergence is not original to this paper. This concept has already been developed such as Powerball method (https://arxiv.org/pdf/1603.07421), FxTS-GF method (https://arxiv.org/pdf/1808.10474) and others.

2. Discretization Gap: The paper provides continuous-time analysis but implements discrete updates via Euler discretization. The authors acknowledge a "discretization gap" in the appendix, noting that the fixed-time regime ($p_1 < 1$) is sensitive to discretization effects and step size choices in practice, potentially leading to instability. This is a critical issue. Properties relied upon in continuous-time finite-time stability proofs often do not hold under fixed step-size discretization $\eta$. The practical relevance of the continuous-time guarantees is therefore doubtful without a discrete-time analysis or a more thorough empirical investigation of how step size $\eta$ affects stability and convergence. Please refer to https://www.ijcai.org/proceedings/2020/451 for further details, could these results be helpful for analyzing (1)? 

(Minor ones)
3. Contribution as Adaptation: The main claimed innovation is making the dynamics element-wise to eliminate dimensional coupling. The principle of element-wise scaling (based on local gradient info rather than global norms) is a key feature of successful adaptive optimizers in deep learning (like AdaGrad, RMSprop, Adam) and is a common technique. Therefore, the core contribution appears to be applying a standard deep learning heuristic (element-wise adaptation) to an existing theoretical framework. The work should be more accurately positioned as an element-wise adaptation of existing fixed-time gradient flows, not an entirely new framework or paradigm.

4. Interpretation of Results: The results presented in the paper that frames SignSGD and Signum as limiting cases of their framework were already known in Zhou et al. IJCAI, 2020  (https://www.ijcai.org/Proceedings/2020/0451.pdf). The connection drawn to SignSGD/Signum as limiting cases ($p_1 \to 2, p_2 \to 0$) 13 is mathematically interesting but its practical significance is unclear. As parameters approach these limits, the theoretical convergence bounds might degrade or become infinite. Does this connection offer new insights into why SignSGD works, or is it just a boundary condition of the mathematical form? The paper claims it provides "theoretical grounding" but doesn't elaborate on the specific insights gained.

### Questions
please refer to my specific comments,

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper motivates from the continuous-time counterpart of optimization algorithms, and proposes novel optimizers EFTOM and PEFTOM based on this perspective. Theoretical analysis and empirical justification of the algorithms are provided in the together for the algorithms.

### Strengths
1. The idea of deriving optimizers from a continuous perspective looks interesting, and the derived algorithms are reasonable since it matches the optimzier derived from the standard steepest descent framework (e.g., SignSGD).
2. The theoretical convergence results seem to be valid, showing finite-time convergence of the algorithm incorporated with momentum.

### Weaknesses
1. For the theoretical part, the convergence rates all have heavy dependence on dimensionality, which can be extremely large in practice. This drawback can be avoided by analysis of similar algorithms like SignSGD or Signum.
2. For the experiment part, could the authors give how the hyperparameters are tuned for EFTOM and PEFTOM? It also looks strange that Table 5 indicates the best SGD learning rate is $ 0.2 $ , which is not listed in the listed SGD learning rate search grid. This unclear setting raises questions in fair comparisons with other optimizers.

### Questions
1. Could we fix the explicit dependence on dimensionality by employing similar assumptions as the SignSGD paper, i.e., we consider smoothness in a different norm?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces an element-wise optimization framework that overcomes the scalability issues of control-theoretic methods in high-dimensional deep learning. By using coordinate-independent dynamics, it ensures rigorous finite-time convergence while preserving the adaptivity essential for neural network training. This theory unifies various optimizers under one principled foundation, rigorously justifying the empirical success of SignSGD and Signum as special cases.

### Strengths
- The work provides valuable new insights by bridging control-theoretic stability principles and large-scale non-convex optimization.

- It is beneficial for understanding optimizer convergence behavior, moving beyond traditional asymptotic rates to finite-time guarantees.

- The proposed PEFTom optimizer demonstrates competitive performance compared to existing state-of-the-art methods.

### Weaknesses
- The experimental results do not fully substantiate the central theoretical claim that PEFTom achieves finite-time convergence. More targeted experiments are needed to directly illustrate this property.

- The experimental setup relies on CNN architectures (e.g., ResNet, DenseNet) and datasets (CIFAR-10, CIFAR-100) that are now considered somewhat stale. To convincingly demonstrate the optimizer's effectiveness and scalability, experiments on more modern architectures (e.g., Vision Transformers) and larger, more complex datasets (e.g., ImageNet) are recommended.

### Questions
The PEFTom optimizer introduces several new hyperparameters (e.g., $c_1, c_2, p_1, p_2$). Could the authors provide:

* A sensitivity analysis to show how the algorithm's performance is affected by variations in these parameters?

* Practical guidance or heuristics for tuning these hyperparameters effectively?

### Soundness
3

### Presentation
3

### Contribution
3

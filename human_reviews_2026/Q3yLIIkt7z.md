# Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime

- Avg Score: 7.00
- Decision: Accept (Oral)
- Scores: 8, 8, 6, 6

## Abstract
Neural scaling laws underlie many of the recent advances in deep learning, yet their theoretical understanding remains largely confined to linear models. In this work, we present a systematic analysis of scaling laws for quadratic and diagonal neural networks in the feature learning regime. Leveraging connections with matrix compressed sensing and LASSO, we derive a detailed phase diagram for the scaling exponents of the excess risk as a function of sample complexity and weight decay. This analysis uncovers crossovers between distinct scaling regimes and plateau behaviors, mirroring phenomena widely reported in the empirical neural scaling literature. Furthermore, we establish a precise link between these regimes and the spectral properties of the trained network weights, which we characterize in detail. As a consequence, we provide a theoretical validation of recent empirical observations connecting the emergence of power-law tails in the weight spectrum with network generalization performance, yielding an interpretation from first principles.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper obtains precise limits of the test loss and weight spectrum for two models of shallow (linear/quadratic) networks. A complete phase diagram is given as a function of the regularization intensity and effective sample size.

### Strengths
The paper gives a very complete answer for the scaling laws of the regularized global minimum in these two simplified models. These networks implicitely add a L1-type of bias, comparison to previous work on random feature models/kernel methods which are in a L2 setting. The tools for the L2 setting are now well established (Random Matrix Theory etc), but the L1 case was to my knowledge less studied. 

The paper is very clear, and the authors have done a great job of explaining the different phases and relate them to behavior of the weight spectrum, which helps a lot to not be lost in the many big formulas.

### Weaknesses
The paper only studies diagonal linear networks and quadratic networks with fixed second layer weights, which is basically a shallow fully-connected linear network with symmetric weights. These are very simple models, far from anything used in practice, but one has to start somewhere.

### Questions
- Do the L1 models that you consider outperform their L2 counterparts in general or in specific settings? And if so by how much? Since you seem to consider true functions that are not exactly sparse but only have a certain decay, I would imagine that the advantage of the L1 approach would appear for sufficiently steep decay of the true function spectrum?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper provides a theoretical framework for neural scaling laws in shallow neural networks that go beyond the random features (lazy training) regime by explicitly modeling feature learning. It focuses on two analytically tractable architectures:

1) Diagonal networks — equivalent to LASSO (l1-regularized linear regression).

2) Quadratic networks — equivalent to matrix compressed sensing (low-rank estimation with nuclear norm regularization).

Through these mappings, the authors derive:

A complete phase diagram (Fig. 1, p. 6) of generalization excess risk as a function of sample size, model dimension and weight decay.

A universal scaling law showing crossovers between plateaus, fast- and slow-decay regimes, and double-descent–like interpolation peaks.

A connection between scaling phases and spectral properties of the learned weights, yielding a theoretical explanation for the empirically observed power-law weight spectra in modern deep networks.

The analysis combines approximate message passing (AMP), state evolution, and random matrix theory, yielding results that match both Bayes-optimal rates and extensive numerical simulations (Figs. 2–6, pp. 7–25).

### Strengths
Novel analytical unification: 
The paper builds an elegant bridge between feature-learning neural networks, LASSO, and matrix compressed sensing, allowing exact asymptotic characterization of generalization in previously intractable regimes.

Comprehensive phase diagram: Figure 1 neatly summarizes all scaling regimes and their transitions. It captures empirical phenomena (plateaus, overfitting peaks, fast decays) reported in large-scale scaling-law studies (Kaplan et al. 2020; Paquette et al. 2024).

Spectral–generalization link: The “universal error decomposition” (Result 3) expresses the generalization error as an explicit functional of the weight eigenvalue spectrum, providing the first first-principles explanation of heavy-tailed spectra linked to generalization quality.

### Weaknesses
Restricted architecture scope: Results hold for shallow (two-layer) diagonal or quadratic networks; extensions to multilayer or non-polynomial activations remain speculative. It’s unclear how much insight transfers to modern deep transformers or CNNs.

Dependence on heuristic AMP validity: The central results (e.g., Eq. 19) assume non-asymptotic validity of state evolution, supported empirically but not proven. The authors acknowledge this (Sec. 2.4) as a conjecture, so the paper’s main theorems are partially non-rigorous.

Target-model assumptions: The teacher–student setup with Gaussian inputs and power-law eigenvalues is mathematically convenient but idealized. Real data distributions or nonlinear activations could alter scaling transitions.

### Questions
1) Can the spectral–generalization decomposition (Eq. 17) be extended to multi-layer or nonlinear activations?

2) How sensitive are your phase boundaries to non-Gaussian inputs or correlated features?

3) Is there a principled path toward proving the non-asymptotic AMP conjecture (Eq. 19)?

4) Does the observed \lambda^{-2/3} interpolation scaling have a connection to known double-descent exponents in overparameterized deep models?

### Soundness
4

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
This work extends neural scaling law theory  by analyzing quadratic and diagonal neural networks in the feature learning regime, using tools from matrix compressed sensing and LASSO to map out how excess risk scales with data, model size, and weight decay. The authors identify distinct scaling regimes, plateaus, and spectral signatures (like power-law tails in the weight distribution) and show how these phenomena predict generalization, providing a first-principles explanation for trends observed empirically in large neural networks and the correlation between power-law tails in the weight distribution and generalization in neural networks.

### Strengths
1. The paper gives a unified phase diagram of generalization error across sample size, width, and regularization for diagonal and quadratic networks (true feature learning, not just kernels).

2. Exploring the theoretical relationship between the weight spectrum and generalization is an important open problem, and this paper provides an explanation for it in certain special neural network architectures. It analytically characterizes the trained weight spectra (bulk, spikes, heavy tails) in every regime and links these spectral features directly to underfitting, overfitting, and approximation error. 

3. It explains benign vs harmful overfitting and double-descent–like peaks from first principles, and shows how optimal regularization suppresses noisy bulk without destroying learned signal.

### Weaknesses
**If you address the issues in the Weaknesses and Question sections, I will increase the score.**

1. The writing of the paper could be improved. Some of the notation is not explained very clearly (see  Questions sections), and also it seems the paper does not include a conclusion section.
2. The current setting is limited to special two-layer neural networks, which allows the study of their excess risk minimizer to be reduced to matrix compressed sensing and Lasso problem. I would like to know whether, for more general two-layer (or even multi-layer) neural networks, there is a similar way to reduce the problem. If this is difficult, do you think it would be possible instead to study the training dynamics of the weight spectrum under (stochastic) optimizers in order to explore the relationship between heavy-tailed weight spectra and training dynamics?
3. In the paper, you assume that the spectrum of the target network weights follows a power-law distribution (which is  heavy-tailed ). Is this assumption necessary[1]? and how much does this heavy-tailed structure affect the final weight spectrum of the student network?
4. Minor: It looks like in Figure 3, the third plot in the first row should be labeled with the exponent $-2 + \frac{1}{\gamma}$.


[1] Gurbuzbalaban, Mert, Umut Simsekli, and Lingjiong Zhu. "The heavy-tail phenomenon in SGD." International Conference on Machine Learning. PMLR, 2021.

### Questions
In addition to the issues mentioned in the Weaknesses section, I may also have the following questions.

1. In line 473, what does the first  $R$ represent? In addition, what is the relationship between the $R_{n,d}$ of ERM mentioned here (line 479) and the $R$ that appears in equations (11), (14), and (17)?

2. There are also some papers [2][3][4][5][6][7][8][9][10] that discuss the relationship between the spectrum of the weights / features / data and generalization or neural scaling law. Could you provide some comments and discussion on these papers?




[2] Bartlett, Peter L., et al. "Benign overfitting in linear regression." Proceedings of the National Academy of Sciences 117.48 (2020): 30063-30070.

[3] Simsekli, Umut, et al. "Hausdorff dimension, heavy tails, and generalization in neural networks." Advances in Neural Information Processing Systems 33 (2020): 5138-5151.

[4] Hodgkinson, Liam, et al. "Generalization bounds using lower tail exponents in stochastic optimizers." International Conference on Machine Learning. PMLR, 2022.

[5] Wang, Yutong, Rishi Sonthalia, and Wei Hu. "Near-interpolators: Rapid norm growth and the trade-off between interpolation and generalization." International Conference on Artificial Intelligence and Statistics. PMLR, 2024.

[6] Wang, Zhichao, et al. "Spectral evolution and invariance in linear-width neural networks." Advances in neural information processing systems 36 (2023): 20695-20728.

[7] Dandi, Yatin, et al. "A random matrix theory perspective on the spectrum of learned features and asymptotic generalization capabilities." arXiv preprint arXiv:2410.18938 (2024).

[8] Worschech, Roman, and Bernd Rosenow. "Analyzing Neural Scaling Laws in Two-Layer Networks with Power-Law Data Spectra." arXiv preprint arXiv:2410.09005 (2024).

[9]  Kothapalli, Vignesh, et al. "From spikes to heavy tails: Unveiling the spectral evolution of neural networks." Transactions on Machine Learning Research (2025).

[10] Arous, Gérard Ben, et al. "Learning quadratic neural networks in high dimensions: SGD dynamics and scaling laws." arXiv preprint arXiv:2508.03688 (2025).

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studied the neural scaling law and analyzed the spectra of the trained weight matrix of two-layer diagonal networks and quadratic neural networks. Their teacher network has the same architecture as the student model, but with certain noise. By transferring diagonal networks to the LASSO problem and transferring quadratic neural networks to matrix compressed sensing, the authors can utilize  Generalized Approximate Message Passing (GAMP)  and its state evolution to compute the final excess risks and the spectral distributions of trained weight matrices. This heuristic computation works well for a power law target, a general ridge parameter, input dimension, and effective sample size. As a result, this paper showed how the scaling of the sample size and regularization affect the generalization, and how different scalings lead to different spectral behavior of the trained weight matrix (e.g., rank collapse, bulk+spikes, heavy-tailed distribution).

### Strengths
The paper is well-written, and its results are of interest to the deep learning theory community. To the best of my knowledge, this is the first paper that theoretically explains the relationship between the spectral behavior of learned weights and generalization errors, as well as the 5+1 phases for trained weights observed by Mahoney and Martin in 2019.

### Weaknesses
The neural network architectures are too specific. I understand that for general two-layer neural networks, the AMP is too complicated to analyze. However, it would be better to explain the difficulty, if there is any other simple architecture that can still work beyond quadratic or diagonal networks, and what the expected results are for general cases. It would be insightful to provide such information or discussion.
Additionally, there is a lack of training in dynamic analysis. This paper only used the state evolution to analyze the solution of the LASSO and matrix compressed sensing, but did not study the scaling of the training time or computational scaling.

### Questions
1. What is $\varepsilon$ in Figure 1? Figure 1 is nice and very important for the presentation. However, it is a little bit messy in the x and y axes. There is a line $\lambda = \sqrt{n_{\text{eff}}/d}$, but when $n_{\text{eff}} = 0$, why do you have $\lambda = 1/d$? You should explain the red lines in the caption of the figure.

2. For VIa and VIb, both risks decay fast, but the spectra of weights behave differently. One is bulk+spikes, and the other is heavy-tailed. Does this mean that we do not need heavy-tailed spectra to get a good generalization error? Martin & Mahoney, 2021a, and Martin et al., 2021, argued that networks with heavy-tailed spectra exhibit optimal generalization capabilities.

3. In results 2, does the trained weight still have independent entries? $\hat{\theta}_i$ are independent for different $i$?

### Soundness
3

### Presentation
3

### Contribution
3

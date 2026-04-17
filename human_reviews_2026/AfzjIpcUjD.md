# $\alpha$-PFN: Fast Entropy Search via In-Context Learning

- Decision: Reject
- Scores: 6, 4, 4, 6, 6

## Abstract
Information‐theoretic acquisition functions such as Entropy Search (ES) offer a principled exploration–exploitation framework for Bayesian optimization (BO). However, their practical implementation relies on complicated and slow approximations, i.e., a Monte Carlo estimation of the information gain. This complexity can introduce numerical errors and requires specialized, hand-crafted implementations.
We propose a two‐stage amortization strategy that learns to approximate entropy search-based acquisition functions using Prior‐data Fitted Networks (PFNs) in a single forward pass. A first PFN is trained to be conditioned on information about the optima; second, the $\alpha$‐PFN is trained to predict the expected information gain by training on information gains measured with the first PFN. The $\alpha$-PFN offers a scalable and learnable approximation, which replaces the complex approximations with a single forward pass per candidate, enabling rapid and extensible acquisition evaluation. Empirically, our approach is competitive with state‐of‐the‐art entropy search implementations on synthetic and real‐world benchmarks while accelerating the different entropy search variants by over at least a factor of 12x, with the largest speed ups around 30x for the highest 8 dimensional problems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
- This paper addresses Bayesian optimization (BO) with information-theoretic acquisition functions (ACQFs) such as Entropy Search (ES), Predictive Entropy Search (PES), Max-value Entropy Search (MES), and Joint Entropy Search (JES).
- These ACQFs lack analytical closed forms and require costly Monte Carlo–based approximations, which make them computationally expensive. This motivates the use of approximations for these ACQFs.
- The authors propose a two-stage amortization framework using Prior-data Fitted Networks (PFNs) to approximate these ACQFs.
- The first-stage PFN (base PFN) is trained on millions of simulated Gaussian Process (GP) sample paths and their optima. It predicts the posterior predictive distribution $q(y_{tst}|D_{trn},x_{tst},I)$, conditioned on information about the true optimum ($I=x^∗,f^∗,(x^∗,f^∗)$ depending on the ACQF variant). The entropy difference between the conditioned and unconditioned predictions provides the information gain, which serves as training data for the next stage PFN.
- The second-stage PFN ($\alpha$-PFN) is then trained to directly predict the expected information gain (the ACQF value), effectively learning to approximate PES, MES, and JES in a single forward pass
- During actual BO loop, only the $\alpha$-PFN is used to evaluate the acquisition function and select the next point.
- Experiments on synthetic and real-world benchmarks show that $\alpha$-PFN achieves performance competitive with standard GP-based PES/MES/JES, while providing substantial computational speedups up to 30x faster in some of the test problems.

### Strengths
- The paper proposes an original two-stage PFN framework to efficiently approximate expensive information-theoretic ACQFs in BO. The idea of amortizing PES/MES/JES computation via neural networks is creative and addresses a key bottleneck in practical BO.
- Theoretical analysis is rigorous and clearly connects the two PFN stages, demonstrating that the $\alpha$-PFN can recover the expected information gain under mild assumptions. 
- Moreover, the graphical illustrations and diagrams are clear and informative, effectively conveying the high-level ideas and overall workflow of the proposed framework.

### Weaknesses
- The paper motivates its contribution by the high computational cost of information-theoretic ACQFs, yet it does not clearly articulate why such ACQFs are preferable to simpler and cheaper alternatives like Expected Improvement (EI) or Upper Confidence Bound (UCB). Without this justification, it is unclear when the proposed acceleration would be practically valuable. A stronger discussion or empirical comparison highlighting scenarios where PES/MES/JES outperform EI would improve the motivation and significance of the work.
- Although inference is much faster, training remains highly resource-intensive, requiring millions of GP simulations and two large PFN models. This appears to be a major limitation of the approach and should be emphasized more clearly in the main text.
- Furthermore, the reported performance–runtime comparison could be presented in a more informative manner, for example by including a Pareto front that visualizes the trade-off between optimization performance and computational cost.
- This analysis should also account for the overall runtime, including the upfront cost of training the α-PFN.

### Questions
- Line 168: Typo $D_{trn}$
- What is the dimension of each LC-Bench problem? Having parenthesis in the plot title as in synthetic test functions would be helpful.
- How $\alpha$-PFN based acquisition function is optimized in each iteration?
- It would be great to see numerical experiments in higher dimension to see how the performance of the proposed method behaves.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper challenges the speed up of the information-theoretic Bayesian optimization (BO) algorithms leveraging prior-data fitted networks (PFNs). Different from the existing studies that leverage PFNs for Bayesian predictions, this paper leverages PFNs to directly meta-learn the BO optimization trace. As a result, the authors report a consistent 12x speedup compared with the usual Gaussian process-based entropy search algorithms.

### Strengths
This paper is clearly well-written and interesting as a practical verification, leveraging the transformer for BO. 
The speed up can be a solid benefit, particularly when the objective function evaluation cost is moderate.

### Weaknesses
- Since this paper concentrates on the experimental validation of the transformer-based network on the black-box optimization problem, I would expect more extensive experiments. In particular, when the evaluation cost of the objective function is moderate, I think that other black-box optimization methods, such as genetic algorithms, can be an option. Thus, the experimental result comparing such wider black-box optimization methods on the wall-clock time can be of interest.


- This paper concentrates on entropy search algorithms. On the other hand, in the widely studied optimization problems, such as vanilla, multi-objective, and multi-fidelity optimization, there are already cheap and practically strong BO methods. The benefit or future potential of the proposed method compared with such known cheap baselines was unclear to me.


- I think that one of the natural directions to leverage the neural network for bandit or optimization is employing neural network-based surrogate models. Indeed, there are several studies, such as neural bandit [1]. However, there is no discussion and comparison of studies that leverage neural networks in other ways. 

[1] Dongruo Zhou, Lihong Li, Quanquan Gu, Neural Contextual Bandits with UCB-based Exploration, ICML2020.


Minor:
- The standard errors in the regret plot are heavily overlapped, which degrades the visibility. Please consider using an error bar or other change to improve the visibility.

### Questions
Please answer the above comments.

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
The authors address the inefficiency of complex Monte Carlo approximations for Entropy Search (ES)-based acquisition functions in Bayesian Optimization (BO), which cause numerical errors and require cumbersome hand-crafted implementations. It proposes a two-stage amortization strategy using Prior-data Fitted Networks (PFNs): a base PFN is first trained to condition on information about the optima, and then the α-PFN is trained to predict the expected information gain using the information gains computed by the base PFN, enabling approximation in a single forward pass.

Key contributions:
1) It innovatively applies PFNs to amortize the approximation of ES variants (Predictive Entropy Search, Max-value Entropy Search, Joint Entropy Search), replacing costly sampling-based methods with a fast single forward pass and achieving speedups of at least 12 times (up to 41.2 times for 8-dimensional problems); 
2) It supports fully Bayesian Gaussian Process models by integrating hyperparameter uncertainty, a capability rarely seen in existing ES implementations;
 3) Empirically, it matches the performance of state-of-the-art ES methods on synthetic Gaussian Process benchmarks and real-world hyperparameter optimization tasks from LC-Bench while maintaining high computational efficiency.

### Strengths
This paper proposes a two-stageα-PFN framework, which uses PFN amortized approximation to break through the limitation that traditional Entropy Search (ES)-based acquisition functions rely on complex Monte Carlo sampling. It is compatible with PES/MES/JES variants and supports fully Bayesian Gaussian Processes. Experiments on synthetic data and LC-Bench tasks verify that its performance is comparable to that of mainstream ES methods, with a speedup of at least 12× (reaching 41.2× for 8-dimensional tasks). The paper has a clear logical structure and explicit details, providing a new paradigm for the research on efficient acquisition functions in Bayesian Optimization and reducing the computational cost of practical applications.

### Weaknesses
1. Insufficient generalization in high-dimensional scenarios: The model is only trained up to 6 dimensions and validated on 8-dimensional tasks, failing to cover higher dimensions (e.g., 50+ dimensions), and the reasons for performance degradation in high dimensions are not analyzed. It is recommended to supplement synthetic experiments for 10-50 dimensions, analyze the feature utilization efficiency of PFN, and conduct corresponding optimizations.

2. Limited coverage of real-world scenarios: The validation is only conducted on LC-Bench hyperparameter optimization tasks, without involving complex real-world scenarios such as multi-objective optimization and noisy data. It is recommended to expand to multi-objective Bayesian Optimization (BO) tasks, validate on real noisy datasets, and supplement noise-resistant strategies.

3. Unoptimized PFN architecture: A fixed 6-layer Transformer architecture is adopted, with no exploration of the impact of different numbers of layers or attention heads, nor any attempts at model lightweighting. It is recommended to conduct architecture ablation experiments to select the optimal configuration and attempt model distillation.

4. Incomplete baseline comparisons: Only traditional ES variants and Log EI are used for comparison, with no inclusion of emerging methods (e.g., BORE). It is recommended to supplement such baselines to clarify the technical positioning of α-PFN.

### Questions
1. Supplement high-dimensional performance decomposition experiments: Count α-PFN’s time in "acquisition function evaluation", "data preprocessing," and "model inference" for 10-50D synthetic tasks, compare with traditional ES. Analyze high-dimensional feature importance via attention visualization; add mutual information-based feature selection if redundant features exist.

2. Optimize synthetic trajectory parameters: Use grid search (instead of random sampling) to find optimal ranges based on real-task (e.g., LC-Bench) historical queries. Introduce DTW distance to verify trajectory consistency, improving PFN training data authenticity and model generalization.

3. Explore hyperparameter sampling optimization: Test 5/10/20/50 sampling times’ impact on α-PFN. Fix 10 times if performance is comparable to more samplings. Try gradient-based methods (e.g., HMC) to enhance hyperparameter coverage and fully Bayesian model accuracy.

4. Improve out-of-distribution domain shift: Add domain adversarial training or augment GP samples like Levy function for tasks (e.g., Levy 4D). Supplement "domain shift degree-model performance" curves to clarify α-PFN’s adaptation boundary.

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
3

### Summary
The paper proposes α-PFN, a two-stage amortization approach that learns information-theoretic acquisition functions (PES, MES, JES) for Bayesian optimization (BO) via Prior-data Fitted Networks (PFNs). A base PFN is first trained to approximate the posterior predictive distribution conditioned on oracle information about the maximizer ((x^*)) and/or max value ((f^*)). A second network, α-PFN, is then trained to predict the expected information gain in a single forward pass, replacing Monte-Carlo–based ES approximations at inference time. The method targets fast, learnable ES-style acquisitions with **12×–41×** speed-ups while remaining competitive in regret on synthetic functions and LC-Bench HPO tasks; it further includes a fully Bayesian variant by training on GPs with hyperpriors. The paper provides a derivation showing that α-PFN’s output distribution matches the ES objective in mean under an exact-base-PFN assumption. Code and checkpoints are released via an anonymized repository.

### Strengths
1. **Clear conceptual advance (amortized ES):** Recasts PES/MES/JES evaluation as a learned acquisition problem; α-PFN directly outputs the information gain, eliminating test-time sampling required by standard ES implementations. This is cleanly described and visualized (Figs. 1–2).  
2. **Theoretical alignment to ES:** The training objective for α-PFN minimizes the KL between the true information-gain distribution and the model; the mean of α-PFN’s output equals the ES acquisition (Eq. (6)–(8), “Insight 1”), assuming the base PFN is accurate. This connects the method to ES formally.  
3. **Runtime gains:** Reported **12×–41×** speed-ups across 2D/4D/8D while maintaining competitive regret; substantial for high-throughput BO. 
4. **Handling of fully Bayesian GPs:** Training on GP hyperpriors makes fully Bayesian ES tractable in this learned framework (single acquisition rather than hyperparameter-averaged acquisitions). 
5. **Domain-shift aware training:** Introduces a synthetic trace generator to mimic the global→local search transition in practical BO, addressing the distributional mismatch seen when training on uniformly sampled inputs.  
6. **Empirical scope & transparency:** Evaluations on GP priors (2D/6D), synthetic test functions up to 8D, and LC-Bench HPO; results include 100-run averages and error bands; code + checkpoints are provided; compute budget documented.

### Weaknesses
1. **Key assumption left unquantified:** The formal link requires that the base PFN accurately approximates (p(y\mid x,D,I)), yet no error bounds or calibration diagnostics are provided for this approximation. This weakens the theoretical guarantee from a practical standpoint. 
2. **Out-of-distribution degradation:** The method can underperform on OOD settings (e.g., Levy 4D; late iterations on Ackley 8D), highlighting prior-mismatch sensitivity and challenges generalizing beyond the training envelope. 
3. **Training compute and engineering overhead:** Precomputing RFF GP samples and maximizers is expensive (~**40k CPU hours** per setting), and α-PFN must be retrained per prior, limiting plug-and-play applicability.  
4. **Comparative fairness of timing:** Speed-ups compare GP-CPU vs PFN-GPU. A more stringent comparison would include GPU-accelerated GPs (e.g., GPyTorch/BoTorch on GPU) and report per-candidate acquisition latency, not only per-BO-loop runtime.

### Questions
1. **Calibration/accuracy:** How close is the base PFN’s conditioned PPD to GP posteriors? Please provide calibration plots and KL/JS to a GP baseline on small problems. 
2. **Target fidelity:** For select instances where GP entropies are computable, how closely do PFN-based entropies match?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the problem that information-theoretic acquisition functions such as Entropy Search (ES) in Bayesian Optimization (BO) rely on complex Monte Carlo approximations, leading to slow computation and susceptibility to numerical errors. It aims to significantly improve the evaluation efficiency of acquisition functions and reduce the computational overhead of BO while ensuring optimization performance. A two-stage amortization strategy (α-PFN) based on Prior-data Fitted Networks (PFNs) is proposed: first train a base PFN to approximate the posterior predictive distribution, then train the α-PFN to directly predict the value of the acquisition function, replacing complex approximations with a single forward pass. Experiments verify that the α-PFN achieves performance comparable to mainstream methods and achieves a substantial speedup

### Strengths
1. It is highly meaningful that the paper adopts a two-stage amortization strategy (α-PFN) based on Prior-data Fitted Networks (PFNs) to address issues such as slow computational speed and susceptibility to numerical errors in traditional entropy search methods.
2. The paper elaborates on the proposed method in detail.
3. The paper has a reasonable organizational structure.
4. The paper conducts extensive experiments to verify the proposed method.
5. Code for reproducing the experiments is provided.

### Weaknesses
1. The method proposed in the paper is very similar to InfoNet, and such methods should be discussed in detail.

2. Figure 1 could be more visually appealing.

3. The paper seems to have used a hyperparameter search method, but relevant detailed data are not presented.

[1] Hu, Z., Kang, S., Zeng, Q., Huang, K., & Yang, Y. (2024). Infonet: neural estimation of mutual information without test-time optimization. arXiv preprint arXiv:2402.10158.

### Questions
no

### Soundness
3

### Presentation
3

### Contribution
3

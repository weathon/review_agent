# Rectifying Adaptive Learning Rate Variance via Confidence Estimation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Recent advances in training physics-informed neural networks (PINNs) highlight the effectiveness of second-order optimization methods. Adaptive variants such as AdaHessian, Sophia, and SOAP leverage approximate curvature information to achieve strong performance on challenging benchmarks. However, adaptive optimizers are prone to instability during the early stages of training—a limitation addressed in part by RAdam through rectification of the adaptive learning rate.
We introduce Adaptive Confidence Rectification (ACR), a novel uncertainty-aware rescaling mechanism that enhances RAdam’s rectification strategy by dynamically adjusting the learning-rate correction based on an empirical measure of confidence. Our method integrates seamlessly with diverse optimizers and training regimes, consistently improving convergence stability and optimization accuracy. Extensive experiments on large-scale PINN tasks demonstrate reliable performance gains over both rectified and non-rectified baselines, establishing ACR as a robust and broadly applicable optimization framework.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a novel method to rectify the adaptive learning rate of ANN optimizers such as Adam or SOAP.
The method, called Adaptive Confidence Rectification (ACR), is motivated by shortcomings of the related rectification strategy RAdam.
While RAdam assumes initial gradients to follow a zero-mean Gaussian and follows a time-dependent but non-adaptive rectification approach, ACR consists of two components:
A confidence estimator is used to adapt the learning rate rectification to the variability of the second moment of the gradient.
Large changes in the second moment result in lower learning rates and thus stop noisy update steps.
Second, gradient-aware scaling is used to adapt the learning rate to different network layers.
The authors apply their proposed method to PINNs trained with different network architectures on different PDEs and based on different optimizers.

### Strengths
- The paper is well written, easy to follow, and the adaptive rectification is well motivated. Related concepts are briefly introduced where needed, and the methods are comprehensible and well structured.
- The proposed method is intuitive.
- While I have to admit that I am not very familiar with popular benchmarks for PINNs and thus cannot assess the validity of the chosen benchmarks or the baseline results, the large performance improvements on the shown datasets are great.

### Weaknesses
- The paper proposes two more or less independent concepts and does not discuss the relation between them. The paper basically only motivates the adaptive confidence mechanism (ACM) but does not give much justification for the scale awareness (SA). It is not clear which of the two effects causes the performance improvements shown. The authors thus should disentangle these effects by performing experiments with ACM only and SA only and compare these against their baseline (RAdam with SOAP) and their proposed method (ACM+SA).
- The choice of hyperparameters is not clear. The authors claim not to have optimized their HPs; however, they chose the values somehow, and this should be made clearer. Furthermore, the dependence of their method on additionally introduced HPs is crucial. Thus, a study on the hyperparameter dependence should be performed. This includes $\rho_{safe}$, $\tau$, and $\lambda$ (Eq. 10), and even though probably less relevant, also $\beta_s$.
- The definition of $\lambda$ is obscure. It is used as weight decay factor in the given algorithms but also occurs as the weighting factor for the gradient-aware scaling in Eq. 10. You are not using the same value as for WD, right? This would not make any sense to me. The value of $\lambda$ is neither given nor discussed in the HP section at the end of Sec. 4.1. No code is provided to check these ambiguities.
- I would imagine $\tau$ and $\lambda$ to be correlated with the learning rate. A study of the correlation between these hyperparameters as well as a study of the impact of the learning rate on the baseline performance could validate that the shown performance improvements are not simply caused by a badly chosen learning rate for the baseline, which is rectified by the implicit dependence of the learning rate on $\tau$ and $\lambda$.
- It would be very interesting to actually study the confidence/variation of the second-moment estimates. You could plot $CV_t$ and $\text{conf}_t$ against the epochs during training, especially in the early stages of training. This could show if the adaptivity of the rectification also has an impact in later epochs where, e.g., the rectification of RAdam has already saturated.
- This similarly holds for $\rho_{\text{target}}$. Does it really deviate that much from $\rho_t$? How does the weighting between $\rho_t$ and $\rho_{\text{safe}}$ evolve during training? This also relates to the dependence of your method on $\rho_{safe}$. Does your method result in an increase or a reduction of $\rho$/$r$?
- Compared to RAdam, your approach is very heuristic and not much theoretically justified. I think that does not have to be an issue if your method works empirically, is well motivated, and/or can be explained empirically. However, in your case, your method seems to be very bloated. I will slightly elaborate on this. You base your approach heavily on RAdam. RAdam is derived from (more or less) solid theoretical assumptions, resulting, e.g., in the kind of messy formula for $r_t$ based on the degrees of freedom $\rho$, and also brings some problems for $\rho < 4$, demanding the differentiation also adopted by you in Eq. 11. Practically speaking, the effect of RAdam is a simple learning rate warm-up, which is, as you stated, non-adaptive. It could also have been implemented by, e.g., $r_t = 1 - \beta^{t-1}$. However, this does not align well with their theoretical justification. This justification is not given in your case. $\rho_{\text{target}}$ might empirically work well; however, it is not related to "degrees of freedom" anymore. Thus, since it is heuristic anyway, you could simplify your adaptive confidence mechanism a lot. E.g., simply using $r_t = \text{conf}_t$ might already work. An interesting way to check this would be to plot $r_t$ next to $\text{conf}_t$ during training and observe similarities.

Minor:
- All the L2 error vs. epoch plots are not needed. All of these basically only show that your method reaches lower values than RAdam. But this is already reflected in the tables. Showing one of these plots and shifting the rest to the appendix is fully sufficient and will give you more space for more relevant analysis.
- There is a $g^2$ missing in the inline equation in line 167.
- It would be nice to indicate that Sections D and E are part of the appendix when referencing them at the beginning of Sec. 4.2.
- Even though it probably will not change the picture, some of your trainings are not really converged and could have been longer, in particular, Allen in the top left of Fig. 3.

### Questions
1. What is $\rho_{\text{max}}$? You mention $\rho_{\text{max}} = 5$ in the appendix. Do you mean the "4" in Eq. 11? If yes, that’s inconsistent.
2. Why did you introduce $\sqrt{d}$ in Eq. 8? No motivation and no theoretical or empirical justification is given. Again, two entangled effects are introduced which are not disentangled sufficiently. Scaling the learning rate by the momentum norm or by the "parameter dimension" are two independent effects that have to be discussed. And what is the "parameter dimension"? You should state more clearly how $d$ is calculated, e.g., for convolutional layers, biases, or linear matrices.
3. Why is the SD so big for KdV AdaHessian RAdam (Tab. 3)?
4. Is the gradient-aware scaling your genuine idea, or did other people do this before? If yes, you have to provide references.
5. While you motivated why ACR is especially useful when training PINNs, you could also apply it to other datasets and also simply combine it with Adam similar to the experiments of RAdam. Did you also perform experiments e.g. on NLP or CV benchmarks?

### Soundness
3

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
2

### Summary
This paper proposes Adaptive Confidence Rectification (ACR), a novel method to stabilize second-order optimization for Physics-Informed Neural Networks (PINNs). The authors identify that existing adaptive methods (e.g., AdaHessian, Sophia) and even rectification-based methods like RAdam suffer from instability due to variance inflation in second-moment estimates and flawed assumptions about gradient distributions.

ACR aims to solve this by introducing a confidence-driven mechanism. Key components include:

a. Using the coefficient of variation (CV) to quantify uncertainty.

b. Deriving a confidence factor from the CV to compute an adaptive target that interpolates between standard and conservative updates.

c. Employing gradient-aware scaling to modulate the rectification.

The experimental results demonstrate that ACR achieves competitive or superior performance compared to baselines on several PDE-driven tasks. The authors show its robustness across different architectures and its successful integration with other second-order optimizers like SOAP.

### Strengths
\textbf{Significant Problem}: The paper addresses a clear and important problem in the optimization of deep neural networks, particularly for scientific machine learning applications like PINNs, where the instability of adaptive second-order methods is a known barrier.

\textbf{Novel Mechanism}: The core idea of using a statistical confidence measure (derived from the CV) to dynamically rectify the second-moment estimate is intuitive and novel. It presents a principled approach to stabilize the optimizer based on its own internal state uncertainty.

\textbf{Comprehensive Experiments}: The empirical evaluation is thorough, covering multiple PDE benchmarks, different network architectures, and combinations with other optimizers. The results consistently show that ACR matches or outperforms key baselines, validating its effectiveness in practice.

### Weaknesses
The paper is promising, but its central claims rest on methodological and empirical points that require further justification.

Lack of a Direct Stability Metric: The paper's primary claim is improved stability, yet the evaluation relies almost exclusively on final task performance (e.g., L2 error). While lower error is a positive outcome, it is an indirect measure of stability.

Justification for the Choice of Confidence Metric: The paper's central contribution is the use of "confidence estimation," but it relies specifically on the coefficient of variation (CV) in a particular form. The manuscript lacks a clear justification for this design choice.

Introduction of New Hyperparameters: The ACR method introduces new parameters, notably the threshold for conditionally applying the coefficient. The paper does not provide a clear sensitivity analysis or tuning strategy for these parameters.

Generalizability of the Rectification Method: The paper positions ACR as an alternative to RAdam. However, the core idea of "confidence rectification" seems general.

### Questions
How can the authors substantiate the claim of stability beyond just achieving a lower final L2 error? True stability could be demonstrated more directly, for example, by plotting the variance of the learning rate or the second-moment estimate itself over time, or by measuring sensitivity to initialization. As it stands, it is difficult to distinguish if ACR is truly more "stable" or simply a more effective optimizer for these specific tasks.

Why is the CV the most appropriate metric for this task compared to other established uncertainty quantification techniques (e.g., bootstrapping, Monte Carlo simulation, first-order Taylor expansion)? A more rigorous discussion, or ideally an ablation study, on the advantages of CV (e.g., computational cost, accuracy) over these alternatives would significantly strengthen the paper's methodological claim.

Does this threshold require dataset-specific tuning? A key drawback of many adaptive optimizers is the replacement of one difficult parameter (learning rate) with several others. The authors should provide a detailed analysis of this threshold's impact and a fair comparison of the total tuning effort required for ACR versus the baseline methods like RAdam.

Could this same confidence mechanism be applied to rectify RAdam itself? RAdam's rectification is based on the length of the SMA. It would be insightful to see if replacing or-augmenting RAdam's heuristic with ACR's confidence factor yields further improvements. This would help isolate the contribution of confidence-based rectification from the other components of the ACR optimizer.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces ACR (Adaptive Confidence Rectification), a refinement of RAdam designed to stabilize adaptive optimizers by estimating the reliability of variance during training. It measures the stability of moment estimates to compute a confidence score, which adjusts RAdam’s rectification term toward a safer baseline when uncertainty is high. A separate scaling factor further modulates updates based on gradient magnitude. ACR can be plugged into optimizers such as SOAP, AdaHessian, and Sophia, improving convergence speed and reducing $L_2$ error on PDE benchmarks

### Strengths
Cleare motivation: The intro convincingly argues that rectification matters in early iterations and that RAdam’s Gaussian assumption can misfit heavy-tailed/multimodal gradients often seen in deep training

ACR’s confidence-weighted target $ \rho_{\text{target}} $ and layer-wise scaling $ \gamma_t $ are straightforward to bolt onto SOAP, AdaHessian, and Sophia, with explicit pseudocode

On multiple PDEs, model sizes, architectures (MLP, ResNet, PirateNet), and optimizers, ACR improves $L_2$ error and often convergence speed.

### Weaknesses
The paper frames ACR as “principled,” but no formal convergence or stability theorems for ACR are given.  Consider adding at least a local-stability or bias/variance bound for the rectified update, or tone down the language

Core ACR contributions (confidence via $s_t$, CV normalization, and mixing $\rho_t$ with $\rho_{\text{safe}}$) are reasonable but conceptually close to RAdam’s rectification, which adds a heuristic stability gauge and per-layer scaling. A sharper theoretical argument (beyond motivation) for why CV-based mixing improves the bias-variance behavior of the EMA normalization would help support novelty

The paper asserts gains but does not give forward/backward counts, wall-clock, or memory vs. SOAP+RAdam under matched budgets. Provide a fair compute table

Results are on synthetic PDE suites generated via Chebfun and PINN architectures. That’s appropriate for the target domain, but the paper argues general optimizer value; a small non-PINN check (even a toy image or language task) would help generality claims.

### Questions
Try to sweep some paramets and report stability plots (divergence rate / variance spikes) along with L_2 outcomes. This will justify the “no fine-tune” claim

Consider isolating contributions: (a) confidence-mixing only (no $ \gamma_t $), (b) $ \gamma_t $ only, (c) both. Current ablations don’t disentangle these effects.
	​

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes ACR to address the instability issues that arise when training PINNs with second-order optimizers. The method is based on an empirical confidence-based measure and does not assume that the gradients follow a Gaussian distribution. The authors validate the effectiveness of their approach through experiments.

### Strengths
Overall, the paper is convincing. The authors identify a common issue when training PINNs with second-order optimizers, namely the instability that occurs in the early stages of training due to second-order estimation. The proposed method, ACR, unlike prior work such as RAdam, does not rely on the Gaussian distribution assumption. Instead, it leverages the observed variability of second-order moment statistics to dynamically adjust the rectification strength. The authors integrate ACR into several second-order optimizers, including SOAP, AdaHessian, and Sophia, and demonstrate consistent improvements.

### Weaknesses
This paper relies heavily on empirical design and lacks corresponding theoretical support, such as convergence guarantees or an analysis of how ACR affects the bias and variance of the second-order moment estimates.

Although the definition of confidence appears reasonable, it introduces many additional hyperparameters. The authors do not specify how the default values of these parameters were chosen, nor do they report any ablation or sensitivity analysis regarding them, which may limit the practical applicability of the proposed method.

The paper also lacks an ablation study on the internal components of ACR.

### Questions
Could the authors explain how the hyperparameters were selected? Could they provide a sensitivity analysis and present more detailed ablation experiments? Does the proposed method introduce additional computational overhead?

### Soundness
3

### Presentation
3

### Contribution
3

# Robust Spiking Neural Networks Against Adversarial Attacks

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Spiking Neural Networks (SNNs) represent a promising paradigm for energy-efficient neuromorphic computing due to their bio-plausible and spike-driven characteristics. 
However, the robustness of SNNs in complex adversarial environments remains significantly constrained. In this study, we theoretically demonstrate that those threshold-neighboring spiking neurons are the key factors limiting the robustness of directly trained SNNs.
We find that these neurons set the upper limits for the maximum potential strength of adversarial attacks and are prone to state-flipping under minor disturbances. To address this challenge, we propose a Threshold Guarding Optimization (TGO) method, which comprises two key aspects. First, we incorporate additional constraints into the loss function to move neurons' membrane potentials away from their thresholds. It increases SNNs' gradient sparsity, thereby reducing the theoretical upper bound of adversarial attacks. Second, we introduce noisy spiking neurons to transition the neuronal firing mechanism from deterministic to probabilistic, decreasing their state-flipping probability due to minor disturbances. Extensive experiments conducted in standard adversarial scenarios prove that our method significantly enhances the robustness of directly trained SNNs. These findings pave the way for advancing more reliable and secure neuromorphic computing in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a Threshold Guarding Optimization (TGO) approach against the adversarial robustness problem of directly-trained spiking neural network (SNNs) architectures. The method relies on defending neurons that have membrane potentials very close to the firing threshold, which SNN adversaries essentially exploit. The defense mechanism uses (1) layerwise loss regularizers that move neuron membrane potentials away from the firing thresholds, thus effectively creates sparsity in the surrogate gradients, and (2) noisy LIF neurons to reduce the likelihood of state-flipping under minimal adversarial noise disturbances. Experiments on CIFAR-10/100 with VGG-11 and WRN-16 architectures demonstrate that TGO is effective when combined with adversarial training based methods.

### Strengths
- The paper analyzes a clear cause of adversarial robustness of SNNs, i.e., neurons having threshold-neighboring membrane potentials for test samples. This is also the case for ANN neuron activations, which naturally aligns well in this paper.

- The narrative and descriptions of the TGO methodology is clear.

### Weaknesses
- Experimental evaluations are rather ambigious to draw any conclusions (e.g., noisy inference processes, potentially missing random restarts and EoT, weak attack strengths). It also appears like with simple BPTT, TGO is not highly effective as a standalone defense. Furthermore, surrogate gradient ensemble evaluations are missing, which should have been the rigorous attack baseline for SNN adversarial robustness.

- There are several missing details in terms of hyperparameters, evaluation settings and consistency of the results from the main text to the appendix (where there are really informative results existing).

### Questions
- One of the most critical components of the defense is the use of randomness and injecting noise during inference. This is well-known to significantly prohibit accurate adversarial robustness evaluations. However, authors state that they only employ EoT to investigate this in Appendix B, although EoT and reliably evaluated robust accuracies under random restarts should have been present in all evaluations of the main manuscript.

- The paper talks about directly-trained SNNs and surrogate gradient choices, but never really states the exact surrogate gradient function used in training and evaluation of their models?

- Following the above question, there is also already an established surrogate gradient ensemble based SNN attack, which the paper does not consider: https://openreview.net/pdf?id=I8FMYa2BdP . It essentially aims to evaluate directly trained SNNs more reliably, by allowing the white-box adversaries to try out different surrogate gradient functions for more stable attacks. This aligns with basic security principles, where white-box adversaries should have complete access and capabilities in evaluating models. I would expect the authors to evaluate their "adversarially robust SNNs" under such ensemble attacks, by reporting robust accuracies under surrogate gradient adaptive adversaries.

- The notion of imposing gradient sparsity was already the main idea in the SR approach. How is the present paper different?

- Why are the naive attacks in Table 1 for CIFAR100/WRN-16 with AT+TGO(Ours) more effective than the results in Table 5 when EoT is added? The main idea in EoT is to obtain more rigorous adversaries, without making the attack weaker?

- In Table 2 APGD_CE row, increasing the steps from 80 to 100 makes the attacks slightly weaker, given the numbers in the table. This should not happen in any case. There is some ambiguity regarding the convergence of attacks. How is the attack success calculated with increasing number of iterations?

- Are there any other datasets/architectures that this approach would scale and advance the SoTA? Can we use this method besides VGG and WRN type of spiking networks, or on images larger than 32x32 resolution?

- In general, evaluations are also demonstrated at a stronger perturbation radius than in the training phase. Also, is there the usual "random restarts" idea implemented in these multi-step PGD attacks?

- No hyperparameter details are present. What is the lambda hyperparameter value in the main results? It is not clearly described anywhere, and none of the results for lambdas match consistently between Table 1 and Table 7 either, there appears to be some rows shifted or something. Overall, it is not possible to extract accurate information from the current presentation of results either.

- The results on DVS datasets are only present in Appendix A, very briefly without details. Can the authors elaborate further, how they actually implemented this? These attacks should be fundamentally different to implement, since the inputs are binary. How does the perturbation strength work here for instance? Also, there is a typo there: DVS-CRIAF10 -> DVS-CIFAR10.

- Regarding Figure 4 right side loss landscapes - It is not described how two-dimensional loss landscape visualizations are generated?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a training strategy, termed TGO, to enhance the robustness of spiking neural networks (SNNs) against adversarial attacks such as FGSM, PGD, and their variants. The main idea is to constrain the membrane potential to remain sufficiently distant from the firing threshold, thereby reducing sensitivity to perturbations. In addition, the strategy introduces neuron-level perturbations (NLIF) and regularizes the probability of spike flipping. Experiments on the CIFAR-100 dataset demonstrate the effectiveness of the proposed approach.

### Strengths
**S1.** The idea is straightforward and easy to understand.

**S2.** The authors conduct extensive experiments, including Expectation over Transformation (EoT) and loss landscape analysis.

**S3.** As shown in the tables, TGO achieves the best robustness performance compared with state-of-the-art (SOTA) training strategies.

### Weaknesses
**W1.** The explanation of the idea is unnecessarily complicated. In particular, Theorem 2 seems redundant — it is difficult to follow due to the heavy notation, and after reading the proof in the appendix, it appears to be a straightforward extension of Theorem 1.

**W2.** I believe the proof of Theorem 3 may be incorrect. According to Appendix E, the flipping probability from 1 to 0 should be expressed as the conditional probability and the same applies to the flipping probability from 0 to 1.
$$ 
P_{1\rightarrow 0} = P(\eta[t] < V_{th} - V[t] | V[t] \geq V_{th}).
$$ 

**W3.** Theorem 1 is proved under an $\ell_2$ constraint, whereas the experiments are conducted with $\ell_\infty$ perturbations. This inconsistency raises questions about the necessity and practical relevance of the theorem.

**W4.** According to Eq. (7), the constraint loss needs to be computed in a layer-wise manner. Meanwhile, the NLIF module introduces perturbations at every layer. How efficient is TGO in terms of computation compared with other training strategies? Does the training time increase significantly as the network depth grows?

**W5.** The paper does not specify the value of $\delta$ in Eq. (7) or explain how it was chosen. Please clarify it.

**W6.** The paper contains some typos, though they do not affect my overall rating. A few examples are listed below:

1. Line 182: $|J_f(x)|_2^2$.
2. In Theorem 1, $\eta$ has a mean of 0, whereas in Lines 316–317 and Figure 2(d), another mean $\mu$ appears.
3. In Eq. (11), it seems that $P$ should be used instead of $\Delta P$.
4. The clean accuracy is not highlighted in Table 1.
5. In Appendix C, the proof is written under the condition $||\delta||_2 \leq 1$ rather than $||\delta||_p \leq 1$.

### Questions
Please address W2–W5 in the Weakness section. I will consider increasing my score if these concerns are fully addressed.

### Soundness
3

### Presentation
2

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
This paper proposed a Threshold Guarding Optimization (TGO) method for enhancing the robustness of SNNs. TGO aims to reduce the number of threshold-neighboring spiking neurons, thereby decreasing the state-flipping probability. Noisy-LIF neurons are also adopted to eliminate the influence of adversarial perturbations. Experiments results show that TGO surpasses SOTA SNN-based adversarial defense methods.

### Strengths
1.The reasoning of the paper is clear and coherent. Reducing the number of threshold-neighboring spiking neurons provides a new insight in enhancing the robustness of SNNs.

2.The theoretical analysis of the paper is reasonable.

3.The improvement of TGO is significant, improving the robustness effectively. (Only if the experimental results are convincing, see weaknesses below)

### Weaknesses
1. In Line 021 in abstract and Line 061 in introduction, the authors mentioned their method can enhance ‘gradient sparsity’. Normally the sparsity corresponds to L0-norm [1]. However, in Theorem 1, the author aims to optimize L2-norm of the Jacobian matrix, which is inconsistent to optimizing the gradient sparsity. The term ‘sparsity’ seems inappropriate.

2. What is Figure 2 used for? The main text does not mention or introduce Figure 2, leaving Figure 2 alone. What is $H[t]$ in Figure 2(a) and Figure 2(d)? Why does the state-flipping probability correspond to $H[t]$ (or $U[t]$) instead of $V[t]$? In Figure 2(c), it seems that your method only penalizes membrane potential under threshold, and membrane potential beyond threshold remains unchanged.

3. The experimental results are unconvincing. 
- 3.1 The authors only conducted experiments in CIFAR100 (and a small experiment in DVS-CIFAR10). In Line 335, the authors mentioned CIFAR10, but I cannot see any experiment of CIFAR10 even in Appendix. 
- 3.2 The authors only adopted ANN-based attacks. As the paper focuses on SNNs, SNN-based attacks such as [2][3] must be included for comprehensive evaluation. 
- 3.3 In Figure 3, your method TGO+AT (about 59%) is lower than SR+AT (about 62%) in clean accuracy. However, in Table 1, TGO+AT (64.49%) is higher than SR+AT (60.37%) in clean accuracy. 
- 3.4 Moreover, the adversarial accuracy in APGD and MTPGD in Table 2 is significantly higher than PGD in Table 1. As APGD is much stronger than PGD, why does this situation occur? For instance, in Line 367, PGD10 obtained 22.75% accuracy, while in Line 399, APGD10 only obtained 29.19% accuracy.

References:

[1] Liu, Yujia, et al. "Enhancing Adversarial Robustness in SNNs with Sparse Gradients." International Conference on Machine Learning. PMLR, 2024.

[2] Lun, Li, et al. "Towards Effective and Sparse Adversarial Attack on Spiking Neural Networks via Breaking Invisible Surrogate Gradients." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

[3] Hao, Zecheng, et al. "Threaten spiking neural networks through combining rate and temporal information." The Twelfth International Conference on Learning Representations. 2024.

### Questions
1. In Line 079, ‘TGO combined with vanilla SNNs surpasses those adversarial training strategies for the first time’. Which experiment supports this contribution?

2. Typos in formulas. Like Theorem 1, Line 272 Vth. Please check the whole manuscript.

### Soundness
2

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
5

### Summary
This paper introduces a Threshold Guarding Optimization (TGO) method to enhance the adversarial robustness of SNNs. By regulating neuron membrane potentials and employing probabilistic firing via noisy neurons, TGO significantly reduces vulnerability to adversarial perturbations, outperforming existing methods in various adversarial scenarios.

### Strengths
- This paper is well-written and logically structured, making complex concepts accessible and easy to follow.
- This paper provides a mathematical analysis linking “threshold-neighboring neurons” to adversarial vulnerability, which is a novel and interesting framework for SNN robustness research.
- The authors demonstrate the effectiveness of the proposed TGO method across a wide range of adversarial attack scenarios. The experiments span multiple datasets, network architectures, and adversarial settings, showing strong and consistent results, including outperforming SOTA baselines.

### Weaknesses
- The method introduces additional hyper-parameters such as coefficient parameter $\lambda$ and noise level $\sigma$. However, the effectiveness of the $\lambda$ scheduling and sensitivity of the noise level $\sigma$ is missing.
- The paper does not report the additional training cost introduced by the TGO compared to baselines such as adversarial trainings (AT, RAT).

**Limitation**

According to the reported results, the proposed method appears to reduce clean accuracy, indicating a potential trade-off between robustness and standard performance.

### Questions
Can the authors provide a more detailed derivation or intuition for Eq. (7)?

### Soundness
3

### Presentation
3

### Contribution
2

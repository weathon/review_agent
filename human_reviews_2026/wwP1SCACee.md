# A Convergence Analysis of Adaptive Optimizers under Floating-point Quantization

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6, 2

## Abstract
The rapid scaling of large language models (LLMs) has made low-precision training essential for reducing memory, improving efficiency, and enabling larger models and datasets. Existing convergence theories for adaptive optimizers, however, assume all components are exact and neglect hardware-aware quantization, leaving open the question of why low-precision training remains effective. We introduce the first theoretical framework for analyzing the convergence of adaptive optimizers, including Adam and Muon, under floating-point quantization of gradients, weights, and optimizer states (e.g., moment estimates). Within this framework, we derive convergence rates on smooth non-convex objectives under standard stochastic gradient assumptions, explicitly characterizing how quantization errors from different components affect convergence. We show that both algorithms retain rates close to their full-precision counterparts provided mantissa length scales only logarithmically with the number of iterations. Our analysis further reveals that Adam is highly sensitive to weights and second-moment quantization due to its reliance on $\beta_2 \to 1$, while Muon requires weaker error control and is thus potentially more robust. These results narrow the gap between empirical success and theoretical understanding of low-precision training methods. Numerical experiments on synthetic and real-world data corroborate our theory.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper provides the theoretical convergence results for Adam and Muon, under floating-point quantization of gradients, weights, and optimizer states (e.g., the EMA moment estimation in Adam). They show that both Adam and Muon can derive convergence rates of $O(1/T^{1/4})$ on smooth non-convex objectives, even under the quantization.

### Strengths
(a). To my best knowledge, this seems to be the first theoretical convergence result for Adam under the quantization of two EMA moment estimations, and for Muon.  

(b). The results somehow reflect the effects of quantization on the convergence rate. In addition, the results seem to provide the insight that Adam is sensitive to weights and second-moment quantization, while Muon is potentially more robust, as Muon allows for a weaker quantization error control than Adam.

### Weaknesses
My major concerns lie in the theoretical part.

- The term $\tilde{Q}(T)$ in the convergence bound of Theorem 4.5 is not very clear. Although the authors provide a detailed expression in Eq. A.43, it's still very complicated and lacks of detailed discussion with regard to the dependency on $T$. The dependency on $T$ is crucial since it is the dominating order in the convergence rate. I suggest providing a detailed calculation on the order of $\tilde{Q}(T)$, particularly when $\eta,\beta_2$ and terms like $q_G,q_W$ are set as in Line 372.

- Assumptions 3.1 requires the compressed coefficient to be $2^{-M}$, where $M$ is the mantissa length of the target floating-point format. It seems that $M$ is an important parameter to quantify the accuracy of quantization, which, however, does not appear in the convergence bound. 

- The convergence results are heavily relied on sufficiently small $q_G,q_W,q_M,q_V$ (relative quantization errors), which assembles the case of non-quantization. Based on the existing convergence results for non-quantization of Adam and Muon, it seems that the results in this paper are not very novel. In addition, it lacks very clear definitions of these terms in the main body of the paper. 

- The convergence results require relative quantization errors to be $O(1/T)$ or $O(1/T^2)$ order. However, under the quantization error made in Assumption 3.1, is it possible to achieve a sufficiently small relative quantization error, such as $O(1/T)$ order?

### Questions
- Literatures such as [1] and [2] usually consider the constant compressed coefficient. Could the convergence results be extended to a broader range of compressed coefficients, such as any constant within $(0,1)$?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a theoretical analysis of the convergence properties of Adam-type optimization algorithms. The authors propose a unified framework to analyze convergence rates and provide theoretical guarantees for both convex and non-convex settings. The work includes detailed proofs in the appendix and experiments on synthetic datasets and CIFAR-10 to validate the theoretical claims. The paper also discusses practical implications for hyperparameter selection and algorithm design.

### Strengths
S1: The paper provides the first convergence guarantee for Adam under a practical floating-point quantization model, addressing a significant gap in the literature.

S2: The theoretical analysis is rigorous and well-structured, with careful handling of quantization errors and their impact on convergence.

S3: The paper establishes concrete hyperparameter settings that ensure convergence at the same rate as full-precision Adam, making the theoretical results practically applicable.

S4: The empirical results (Figure 4) provide strong validation of the theoretical findings, showing that increased precision (larger mantissa bit-lengths) leads to smaller converged gradient norms.

S5: The paper carefully justifies the theoretical framework by establishing two foundational equivalences, demonstrating that the analysis of quantizing weighted-sum states is directly applicable to practical quantization scenarios.

### Weaknesses
W1: The paper does not sufficiently discuss the practical implications of the theoretical results for real-world applications, particularly regarding the trade-off between precision (bit-length) and computational efficiency.

W2: The convergence analysis assumes certain conditions that might be restrictive in practice, but the paper doesn't fully explore how these conditions affect real-world implementation.

W3: The empirical evaluation appears limited to a single dataset (Rosenbrock) in Figure 4, which may not be sufficient to generalize the findings across different optimization problems and model architectures.

W4: The paper doesn't provide a comprehensive comparison with other quantization techniques for optimization algorithms, making it difficult to assess how Quantized Adam compares to alternative approaches.

W5: The connection between the theoretical convergence rate and practical training performance (e.g., final model accuracy) is not explicitly established, which would strengthen the practical relevance of the results.

### Questions
Q1: Could you provide more detailed analysis of the practical trade-offs between precision (bit-length) and computational efficiency in Quantized Adam? Specifically, how does the convergence rate $O(T^{-1/4})$ translate to practical training time and memory usage?

Q2: The empirical evaluation appears limited to a single dataset (Rosenbrock). Could you expand the experiments to include more diverse optimization problems and model architectures to better validate the theoretical results?

Q3: How does Quantized Adam compare to other quantization techniques for optimization algorithms (e.g., error-feedback mechanisms, stochastic rounding) in terms of convergence rate and practical performance? A comparative study would strengthen the paper's contribution.

Q4: Could you elaborate on how the theoretical convergence rate $O(T^{-1/4})$ translates to practical training performance (e.g., final model accuracy) for different quantization levels? This would help bridge the gap between theory and practice.

Q6: The paper mentions "we use a small constant $\epsilon>0$ for numerical stability" but doesn't discuss the impact of  $\epsilon$ on convergence. Could you analyze how the choice of $\epsilon$ affects the convergence rate and practical performance?

### Soundness
3

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
3

### Summary
The paper presents a theoretical framework analyzing the convergence of adaptive optimizers (Adam and Muon) under quantization of gradients, weights, and optimizer states, extending prior work that studied only partial components. It quantifies how these errors influence convergence and when performance remains close to full precision. Adam is shown to be more sensitive to quantization, while Muon is more robust. Results are mainly supported by synthetic experiments.

### Strengths
- The paper extends prior work with a rigorous convergence analysis of adaptive optimizers under quantization of gradients, weights, and momentum terms.

- It proposes a quantization schedule that aligns the behavior of quantized optimizers with their full-precision counterparts, offering insights into the sensitivity of different components to quantization error.

- The inclusion of the Muon optimizer broadens the analysis and enhances the paper’s practical relevance.

### Weaknesses
The paper omits key references and makes inaccurate claims about prior work. For instance, 

- [Hou et al. 2019] analyzed not only SGD but also adaptive optimizers such as Adam under weight and gradient quantization (without the first-order momentum term, i.e., $\beta_1=0$). As shown by [D´efossez et al. 2022], omitting momentum only introduces a multiplicative slowdown term, which should be acknowledged unless the new quantization error model changes this relationship.

- Another closely related but uncited work is [Ozkara et al., 2025], which studies the convergence rate of Adam under weight quantization (again without first-order momentum) and analyzes the effect of stochastic rounding, an increasingly important direction. It remains unclear whether the proposed framework can naturally incorporate stochastic rounding into analysis.

Some theoretical setups are impractical. For example, 
- assuming $\beta_2 \to 1$ is unrealistic, in practice a fixed $\beta_2 < 1$ is used, when the term $T \log(\beta_2)$ would diverge. Besides, [Ozkara et al., 2025] already emphasizes the reliance of convergence on $\beta_2 \to 1$. 

- even accepting the assumption, the proposed quantization error schedule that makes the error term vanish is infeasible in practice, as precision cannot be controlled dynamically without effectively reverting to full-precision training. In reality, quantization error is bounded by the machine epsilon of the floating-point format unless altering the format during training

The experimental validation is limited and basic. There are no studies examining the influence of (\beta_2 \to 1) or verifying how the proposed quantization error schedule aligns with theoretical predictions.

Lu Hou et al., Analysis of quantized models. In International Conference on Learning Representations, 2019.

Ozkara et al., Stochastic rounding for LLM training: Theory and practice. In The 28th International Conference on Artificial Intelligence and Statistics.

### Questions
- Around assumption 3.1, isn’t the definition of relative error and quantization error (via mantissa length) essentially the definition of machine epsilon for the underlying floating-point format? If not, what distinguishes it?

- [Ozkara et al., 2025] derives a bound involving error term $T\sqrt{\log(1/\beta_2)}$, while this paper presents one with $-T\log(\beta_2)$. what causes this discrepancy, and could the authors provide a direct comparison? The difference implies distinct requirements for the quantization-error schedule.

- In real training, the actual gradients come with end-to-end training , a compound effect of quantization across multiple layers, how does that correlate or concludes to the quantization error or machine epsilon $q_W$, they are different. 
In real mixed-precision training, gradients are perturbed by end-to-end quantization and matmul across multiple layers, leading to compounded effects. How does this aggregate behavior appear in the end to only the relative error (machine epsilon) $q_G$ term from single quantization function?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In summary, the paper fills an important gap: it builds on existing convergence results for adaptive optimizers but in a new setting --- floating-point errors. The authors establish new converge guarantees for Muon and Adam, showing that both methods retain rates close to their full-precision counterparts. The key assumption --- the boundedness of the relative quantization error is quite realistic and achievable in low-precision-aware architectural designs. To the best of my knowledge, no previous paper has offered similar full-precision-vs-quantization convergence rates for adaptive methods.

### Strengths
**Rigorous analysis and theoretical insights that align with recent practice.** Providing clear statements (Th. 4.5 and Th. 4.6), the work explains why Muon tolerates quantization better than Adam --- mostly due to an important assumption of $\beta_2\to1$ in the Adam analysis. This theoretical insight matches practitioners’ observations [1], narrowing the theory–empirical gap.

**Empiritical validation confirms theory.** Experiments on the Rosenbrock function and small  fully connected models confirm theory. For instance, Figs. 3–4 show that increasing mantissa bits reduces final gradient norms, consistent with the $\Omega(\log T)$-bit requirement 

**Framework.** The idea of analyzing jointly quantized gradients, weights, and optimizer states, whereas prior work focused mainly on gradient-only quantization sounds promising for future research.

[1] "Beyond Outliers: A Study of Optimizers Under Quantization", 2025

### Weaknesses
**Missing research on convergence of matrix-based optimizers, leaving a room for improvement.**  Unlike Kovalev et al. [2] who handles constrained/composite and star-convex settings, or Shen et al. [3], who exploits Hessian structure in several assumptions, the presented theory is only for unconstrained smooth non-convex functions. Also discussions with the results on constrained / unconstrained LMO optimization [4] --- resulting in the Scion optimizer --- would benefit the theoretical flavor of the work. Please, offer any ideas on how to extend your findings to assumptions in mentioned works.
Additionally, can you provide an idea of how to extend your results to another, promising setting regarding --- non-smooth, convex functions? As it is demonstrated to be a setup which explains LLMs fairly well [5,6].

**Mantissa growth requirement.** For Muon, to retain the rate of the full-precision training, you assume the logarithmic grows of mantissa $M = \Omega(\log{T})$. However, in practice, the bit-width is typically fixed (e.g., 8-bit). This means convergence is to a neighborhood in fixed precision. The paper does show empirically that moderate precision suffices, but the theory only covers the increasing-bit regime. So there is a mismatch that can be left for future research. If this issue is not solved, I recommend to live it as a limitation. Otherwise, it would be nice to offer some discussion regarding this topic in the paper.

**Missing research on optimizers trained in low precision formats.** Naturally, the state-of-the-art LLM training is held in the mixed-precision format --- when optimizer states, softmax, normalization layers are in float32 and other parameters are in bfloat16. Recent work [1] studies optimizer behavior in quantization-aware training paradigms, running models in precisions of up to 4 bits. A notable takeaway --- Shampoo consistently yields the lowest accuracy drop. I believe this can be helpful because studying convergence (in the low-precision setup) of other matrix-based optimizers that emerge as a steepest descent under the spectral norm can be a direct consequence of your research. Moreover, two concurrent works [7,8] have benchmarked a zoo of optimizers at scale, showing that matrix-whitening methods are highly performant.
Naturally, extending your theoretical findings to other "matrix" optimizers would be very useful for the community. Can you give a couple of comments regarding how it is possible to extend you framework to optimizers validated in the works above?

[1] "Beyond Outliers: A Study of Optimizers Under Quantization", 2025

[2] "Understanding Gradient Orthogonalization for Deep Learning via Non-Euclidean Trust-Region Optimization", 2025

[3] "On the Convergence Analysis of Muon", 2025

[4] "Training Deep Learning Models with Norm-Constrained LMOs", 2025

[5] "The Road Less Scheduled", 2024

[6] "Prodigy: An Expeditiously Adaptive Parameter-Free Learner", 2023

[7] "Benchmarking Optimizers for Large Language Model Pretraining", 2025

[8] "Fantastic Pretraining Optimizers and Where to Find Them", 2025

### Questions
Se the **Weaknesses** part

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents the first theoretical framework analyzing the convergence of adaptive optimizers like Adam and Muon under floating-point quantization of gradients, weights, and optimizer states. It shows that both can maintain near full-precision convergence rates if mantissa precision scales logarithmically with iterations, with Muon proving more robust to quantization errors than Adam.

### Strengths
complete quantization error analysis under certain settings, both for Adam and Muon

### Weaknesses
Basically experiments are limited, theory is not that informative either in proving practicality. The novelty and contribution is limited.

- line 402 "the second moment (qV) is stricter than for the first moment (qM)" -> there are well known fact existing works e.g., https://arxiv.org/abs/2405.03637 and https://arxiv.org/abs/2405.03637
- many missing connection with stochastic rounding work where it give unbiased estimation, but brining higher variance. In modern low bit training like 4 bits, SR is widely adopted. for 8 bit, MX and NV leads to minimal quantization errors in computing gradient. Moreover  thm3 https://arxiv.org/pdf/2502.20566 has some simpler Adam analysis under quantization error of q_V . better compare.
- experiments are very limited, not covering practical scenarios like LLM, missing of which does not affect any practicality
- more through theoretical analysis is needed, otherwise, it is just naive extension. For example, for β_12(1 + qM)2 <β2(1−qV),the effect of qM and qV is real? any toy example to test? 
- for "ensuring the relative quantization errors satisfy qG,qM = O(1/T)", need more explanation as, although theretical understandable, qG is not controllable as a function of T in practice, also dependency on W, W_Q. 
- which quantization is most important between weight, gradient, M1, M2? are these supported from theory and aligned with empirical studies?

### Questions
See weakness

### Soundness
2

### Presentation
3

### Contribution
2

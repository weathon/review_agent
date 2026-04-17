# A Physics-Inspired Optimizer: Velocity Regularized Adam

- Decision: Accept (Poster)
- Scores: 8, 6, 0, 4

## Abstract
We introduce Velocity-Regularized Adam (VRAdam), a physics-inspired optimizer for training deep neural networks that draws on ideas from quartic terms for kinetic energy with its stabilizing effects on various system dynamics. 
Previous algorithms, including the ubiquitous Adam, operate at the so-called adaptive edge of stability regime during training, leading to rapid oscillations and slowed convergence of loss.
However, VRAdam adds a higher order penalty on the learning rate based on the velocity such that the algorithm automatically slows down whenever weight updates become large. In practice, we observe that the effective dynamic learning rate shrinks in high-velocity regimes, and damping oscillations. By combining this velocity‑based regularizer for global damping with Adam’s per‑parameter scaling, we create a powerful hybrid optimizer. For this optimizer, we provide rigorous theoretical analysis of operation at the edge of stability from a physical and control perspective for the momentum. Furthermore, we derive convergence bounds with the rate $\mathcal{O}(\ln(N)/\sqrt{N})$ for a stochastic non‑convex objective under mild assumptions. We demonstrate that VRAdam exceeds the performance against standard optimizers including AdamW. We benchmark various tasks such as image classification, language modeling, and generative modeling using diverse architectures and training methodologies including Convolutional Neural Networks (CNNs), Transformers, and GFlowNets.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Velocity-Regularized Adam (VRAdam), a new optimizer for deep neural network training that introduces a physics-inspired regularization mechanism. Based on the classical Adam optimizer, VRAdam incorporates a quartic kinetic energy term to dynamically regulate the effective learning rate based on the velocity (momentum) of parameter updates. The resulting learning rate shrinks automatically in high-velocity regimes, reducing oscillations and improving convergence stability. The paper provides a theoretical framework, proving uniform exponential stability via Lyapunov analysis for stochastic non-convex objectives. Extensive experiments on CIFAR-10, WikiText-2, GridWorld, and GPT-2 fine-tuning show that VRAdam achieves faster convergence, smoother training curves, and better generalization compared to AdamW and other optimizers.

### Strengths
1. The introduction of a quartic kinetic energy term as a stabilizing mechanism is a fresh and interesting physics-based perspective on optimizer design.

2. The analogy between optimization trajectories and particle dynamics adds an intuitive understanding of how the method moves away from instability near the edge of stability.

3. The paper rigorously proves global uniform exponential stability and convergence under mild conditions, supported by clear mathematical derivations.

4. VRAdam consistently outperforms AdamW, RAdam, RMSProp, and SGD across diverse tasks (CNNs, Transformers, GFlowNets, and LLMs).

### Weaknesses
1. The presentation is not friendly to readers not familiar with optimization techniques using Langevin dynamics.

2. Lots of hyperparameters are used. Although $\beta_3$ is claimed to be robust, the practical sensitivity of VRAdam to its hyperparameters $like (\alpha_0,\alpha_1,\beta_3)$ is not fully explored. Are they same with Adam or will be influenced by $\beta_3$ ?

3. While AdamW(2017) is a strong baseline, the study omits comparisons with newer optimizers (e.g., LION, AdaHessian), which are relevant for modern deep learning tasks.

4. The heavy use of physical analogies (NRQCD, Lagrangians) brings difficulties to understand the motivation and improvement of the algorithm for readers unfamiliar with physics.

5. The experiments only report the validation and test loss, the improvement against AdamW seems modest, whether VRAdam improves the task’s accuracy is not reported. Besides, no clear ablation on the contribution of the quartic term versus standard momentum damping—this would help isolate the true effect of velocity regularization.

### Questions
1. See weakness.

2. The author choose the quartic kinetic term NRQCD system as T(v), is there any other choice? Can you report the ablation study since it deserves to be the key insights of the improvement against AdamW.

3. How sensitive is VRAdam to the choice of the velocity penalizer β₃? Does it generalize well across tasks without tuning?

4. Does the quartic kinetic term introduce any bias that could affect convergence to flatter minima or generalization in practice? Can it be integrated with other techiques to improve generalization like in ‘Improving Generalization of Deep Neural Networks by Optimum Shifting , AAAI25’?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a physics-inspired optimizer, VRAdam, designed to address the oscillation and convergence slowdown problems that occur when adaptive optimizers (such as Adam) train near the stability boundary. Its core idea originates from kinetic models in classical physical systems. By introducing a velocity regularization term, the method imposes a higher-order penalty on the momentum (velocity) during optimization, thereby dynamically adjusting the learning rate and suppressing instability.

Contributions
1. Proposes a physics-inspired optimizer, VRAdam, which introduces a velocity regularization term into Adam to dynamically adjust the learning rate and suppress training oscillations.

2. Theoretically proves the stability of a simplified model and establishes a convergence bound under non-convex settings.

3. Demonstrates the effectiveness of the method on image classification, language modeling, generative flow networks, and GPT-2 training tasks.

### Strengths
1. The paper is not a minor modification of Adam but introduces an intuition derived from a non-standard Lagrangian containing a $v^4$ term. This physics-to-algorithm mapping is uncommon in existing literature and demonstrates novelty.

2. Theoretically, the paper provides a proof of uniform exponential stability for VRMomentum under quadratic objectives (Theorem 4.1) and establishes a convergence rate bound under stochastic non-convex settings comparable to that of Adam (Theorem 4.2).

3. The experimental tasks cover a wide range of applications.

4. If VRAdam proves to be stable and effective under broader settings (including larger-scale models and more repeated runs), it could be a strong improvement over existing optimizers.

### Weaknesses
1. The theoretical analysis is based on a simplified model, while the actual implementation includes all modules. This may lead to inconsistencies between theory and practice.

2. The tables lack explanations regarding reproducibility and do not include error bars, raising questions about the reliability of the results.

3. The authors are advised to discuss the robustness of the results to hyperparameters, as all experiments use a single configuration. In all experiments, β₃ is fixed to 1, but the rationale behind this choice is not provided.

4. The baseline methods are relatively outdated and insufficient in number. Although the related work section mentions more recent optimization techniques, these newer methods are not included in the experiments.

5. The authors only compared the training time between AdamW and VRAdam on GPT-2, and the results did not show a clear time advantage.

### Questions
1. The visualization of results could be improved. Why is Figure 5 in the appendix blank, and why are some figure fonts too small?

2. The theoretical proofs are based on a simplified model, while the actual implementation includes all modules. The authors are encouraged to explicitly discuss how this simplification affects the stability and convergence conclusions.

3. The reproducibility and statistical significance of the results are difficult to assess.

4. Did the authors conduct any study or experiments regarding the choice of $\beta_3$?

5. Have the authors attempted experiments on more complex tasks?

6. Does VRAdam provide any advantage in training time, and does it introduce additional computational overhead?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The authors propose a physics-inspired variant of AdamW, where the effective learning rate is regularized via the magnitude of the momentum. Some theoretical and empirical results are provided, including a convergence rate and evaluation on CIFAR.

### Strengths
1. The proposed method is novel, and the inspiration from physics is motivated.
2. The theoretical results appear to be solid, and the experimental results seem promising.

### Weaknesses
1. A major part of the theoretical analysis assumptions a convex quadratic objective, which the authors claim to be "[traditional] for stability analysis". However, this is far too strong of an assumption to be practically useful in any deep learning context. Furthermore, the convergence rate is a trivial corollary of the work of Defossez et al., providing little theoretical insight or novelty.
2. The provided experimental results are conducted with small (124M) models and evaluated on simple tasks such as CIFAR. This makes it difficult to establish the scalability of the proposed method for practical deep learning.
3. The proposed method introduces a hyperparameter $\beta_3$, with no discussion on tuning or practical recommendations.
4. The writing is sometimes very difficult to read. While inspiration from other fields is quite common in machine learning, the extended use of physical analogies may not be easily understood by the ICLR community (it was certainly lost on me).
5. Minor nits: ensure the displays on page 3 are normal sized font. Use `\citep` for citations that should be parenthetical.

As there are serious limitations with both the theoretical and empirical contributions of this work, I recommend rejection.

### Questions
1. Could the authors elaborate on how the kinetic energy on line 129 was chosen? What happens if we choose something else? Could this lead to a better optimizer than the proposed method?
2. See Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This manuscript challenges gradient descent optimization techniques. Motivated by physics, the authors present VRAdam, which automatically controls the learning rate based on the momentum. Experiments on CIFAR, Wikitext, GFlowNet, and GPT-2 demonstrate improvements.

### Strengths
- The motivation, especially the physics-inspired optimizer, is interesting. Indeed, the momentum optimizer itself is derived from Newtonian dynamics.
- The theoretical analysis looks solid and promising for the improved convergence.
- Source code is available, which eases deployment in practice.

### Weaknesses
- The norm of $v_t$ of the VRAdam looks to compute the global norm, but it may be appropriate to use the parameter-wise norm to allow parameter-wise learning rate control. This choice is not sufficiently discussed.
- VRAdam brings three hyperparameters of \alpha_0, \alpha_1, and \beta_3. I think these additional hyperparameters make it difficult to adopt the VRAdam in practice.
- Accuracy of 80% for ResNet-32 with CIFAR-10 is a weak baseline.
- Table 1 is not convincing enough. These results should be supplemented with more quantitative and qualitative analysis.
- The experimental results, such as Table 2, are focused on the final validation loss. Is it possible to demonstrate other indices, such as practical ones? I think certain practitioners may want to capture the performance more practically, but the value of loss is difficult to understand on an absolute scale. It is also difficult to understand whether it corresponds to sufficient convergence or is still far from convergence.
- LLM results were only trained for 2 epochs, which I think is insufficient for convergence.
- I think Eq. 36 would be -m/4 + O(\lambda), not -3m \lambda/4 + O(\lambda^2). Is it possible to provide an exact derivation?
- Writing should be improved.
    - Kingma & Ba (2017) → Kingma & Ba (2015)
    - “the, global” → “the global”
    - “d” → “(d)” for the caption of Figure 2.
    - “physical inspired” → “physics-inspired” at Line 70.
    - “Note, that” → “Note that” at Line 784.
- The manuscript writes to compute velocity norm, whereas the source code computes gradient norm by default (normgrad=True). To be compatible with the description in this manuscript, normgrad=False may be correct for default.

### Questions
Please see the weaknesses above. My score is based on the assumption that all typos are corrected in the revised manuscript.

### Soundness
3

### Presentation
2

### Contribution
3

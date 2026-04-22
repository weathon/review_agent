# Binary Oscillation-Regulated Network (BORN): Approach for Binary Neural Networks Training

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4

## Abstract
Binary Neural Networks (BNNs) are gaining attention for making energy-intensive deep learning more accessible to resource-constrained edge devices. Traditionally, training methods for such models focus on minimizing quantization error in forward propagation and approximating the $sign$ function for gradient computation. However, these methods overlook the BNN-specific phenomenon of oscillations and do not adapt the learning rate to the training dynamics. To address this, we propose BORN, which adapts learning to oscillatory behavior. The proposal is based on two key innovations: an Oscillations-aware Sign Approximation, which reduces gradient information loss by gradually approximating the $sign$ function in backward propagation, and an adaptive learning rate adjusted according to progress and oscillations. Beyond that, we are the first to combine a static sign approximation for activations with a dynamic one for weights, which stabilizes optimization while maintaining flexibility. Moreover, unlike prior work that chose approximations solely to replicate the sign function as closely as possible, we are the first to motivate the choice of approximating functions through the properties they induce for binary optimization, providing a principled foundation for their design. The novelty of BORN lies not only in its BNN-specific adaptive mechanisms but also in its ease of integration into existing architectures such as convolutional and transformer networks. To validate this, we integrate the method into several state-of-the-art (SoTA) BNN training frameworks across tasks like super-resolution (SR) and large language modeling (LLM). Experimental results show that the proposed method can cover an average of 34.81% of the proximity interval from SoTA methods to full-precision models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Binary Oscillation-Regulated Network (BORN), a new training algorithm designed to improve the stability and performance of Binary Neural Networks (BNNs). The authors identify that BNN training is plagued by unstable "weight oscillations," where parameters repeatedly flip signs ($\pm1$), preventing convergence.

### Strengths
1. The core concept of using weight oscillations as an explicit feedback signal to control both the gradient approximation and the learning rate is a novel and creative approach to BNN training
2. The method is tested across three distinct and challenging domains (image super-resolution, classification, and language modeling), demonstrating its general applicability
3. The paper includes a formal theoretical analysis of convergence, which adds rigor and provides intuition for why the method works, particularly its advantage over STE

### Weaknesses
1. The experimental tables (Table 3, Table 4) are confusingly labeled with $BORN$ and $BORN^1$ (and implied BORN1-5 variants), with no explanation of what these different versions are. This makes the main results very difficult to interpret. Also there are references to table ??
2. No code is provided. 6 Hyperparameters are introduced without explaining how they were tuned. Key practical details are vague such as the exact method for measuring oscillations
3. While BORN shows consistent improvements, the gains are often moderate and do not always surpass other SOTA methods listed in the tables

### Questions
1. What are $BORN^1$ - $BORN^5$?
2. BORN introduces many new hyperparameters ($\lambda_1, \lambda_2, T_{min}, T_{max}, F, b, \eta$). What was your tuning strategy?
3. Will the code be made available to ensure the results can be replicated?
4. What is the computational overhead (in wall-clock time or FLOPS) of calculating the oscillation metric and updating the OSA and ALR parameters at each step?

### Soundness
2

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
4

### Summary
The paper proposes Binary Oscillation-Regulated Network (BORN), a novel algorithm for training Binary Neural Networks (BNNs) that mitigates oscillation-related instability and accelerates convergence. BORN introduces two key components: an Oscillations-aware Sign Approximation (OSA), which gradually approximates the sign function based on training progress and oscillation rate, and a Static Sign Approximation (SSA) for activations to preserve non-vanishing gradients. Additionally, a dynamic learning rate extension adapts to the state of the OSA. Theoretical analysis demonstrates convergence and shows that BORN achieves a lower final loss compared to the Straight-Through Estimator (STE). Experiments on super-resolution, classification, and language modeling tasks show that BORN covers, on average, 34.81% of the proximity interval between state-of-the-art BNNs and full-precision models.

### Strengths
- The paper is generally well written and clearly structured, making the main ideas easy to follow.
- The mathematical formalization is appropriate and rigorous, with detailed derivations that support the proposed approach.
- The authors provide a theoretical convergence analysis, including proofs that connect the continuous and discrete training dynamics and show that BORN achieves a lower final loss than STE.
- The theoretical formulation is well supported by experiments, which include multiple tasks (super-resolution, classification, and language modeling) and ablation studies validating the contribution of each component.
- Overall, the paper presents a principled and well-validated approach to addressing oscillation-related instability in BNN training.

### Weaknesses
- In the classification experiments, the method is evaluated only on ResNet-style architectures. It would strengthen the work to include results on transformer-based models (e.g., ViT or Swin), especially since the paper claims easy integration of BORN into such architectures.

- The paper would benefit from a broader set of comparison baselines on both ImageNet and CIFAR-10 using ResNet-18. For instance, BiPer (Vargas et al., 2024) achieves higher accuracy than the proposed method on CIFAR-10 (93.75% (BiPer) vs. 93.41% (BORN$^5$)), although BORN performs better on ImageNet. Since BiPer is included only in the appendix for ImageNet, extending the comparison to CIFAR-10 and possibly to other strong recent BNN baselines would provide a more comprehensive evaluation and strengthen the empirical claims.

- The notation of the BORN variants in Table 3 (e.g., BORN¹, BORN², etc.) is unclear. I couldn't find what these superscripts represent, whether they indicate different ablations, training setups, or architectures. Clarifying this notation in the table caption or main text is important for reproducibility and understanding.

Minor comments:

- There is a missing Table reference on page 8. Please ensure that all tables are properly referenced.

### Questions
- Could the authors provide results or discussion on how BORN performs with vision transformer-based architectures (e.g., ViT, Swin)? Is this integration actually feasible within the current BORN framework?
- Could the authors include additional comparisons and analysis with recent BNN baselines in the main document, for example, including the results for BiPer on CIFAR-10 to complement the ImageNet comparison?
- Could the authors clarify what the superscripts in Table 3 (e.g., BORN¹, BORN², etc.) represent? For instance, whether they correspond to different architectures, datasets, or ablation variants?

### Soundness
4

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
4

### Summary
The paper proposes a training scheme for binary neural networks (BNNs) that couples an adaptive sign operator with a time-varying learning rate (LR) schedule. The operator gradually shrinks gradient updates as parameters near their ±1 targets, while the LR is tuned in step over the course of training. The authors also analyze convergence theoretically. Departing from BNN work that mostly targets classification, they evaluate on broader tasks such as image super-resolution and large-scale language modeling. The method slots into several state-of-the-art BNN backbones and generally attains competitive results. The appendix provides full proofs and implementation details. Overall, the work tackles a key challenge in BNN training with a method that has both solid theory and promising empirical performance.

### Strengths
The paper is well-written and presents a BNN-specific training scheme with oscillation-aware sign approximation and adaptive learning rate. That is well motivated to stabilize gradients and aid convergence.

The method appears modular and easy to integrate into existing pipelines with minimal additional implementation burden.

Empirical results indicate consistent gains over common BNN and QAT baselines across tasks beyond classification, suggesting broader applicability.

Theoretical analysis of convergence, while compact, provides useful intuition linking the proposed mechanisms to training stability.

### Weaknesses
The manuscript presents several demonstrations aimed at establishing the convergence of the proposed method (Section 4.1) and estimating its convergence rate (Section 4.2). While Section 4.1 appears relatively straightforward, since under the stated assumptions the optimization naturally leads to an optimal solution, the main concern lies in clarifying how Theorem 2 (Section 4.2) concretely relates to the proposed methodology. At present, the theoretical results seem somewhat detached from the methodological framework, giving the impression that the theory is developed in a more general context rather than directly supporting the proposed approach. Strengthening the connection between the theoretical analysis and the practical implementation would significantly enhance the coherence of the paper and better highlight the originality and scope of the authors’ contribution.

There is no clear explanation for why $h(x)$ and $g(x)$ are chosen as tanh functions. In particular, theoretical analyses do not justify this choice. Therefore, it seems reasonable to consider any functions that exhibit suitable behavior for binarization, as long as they serve the same purpose as the selected ones. The authors should clarify the rationale behind their choice and relate it explicitly to the theoretical analysis. In particular, they should conduct a toy experiment to demonstrate how the theoretical findings align with the chosen functions.

The adaptive learning rate resembles standard schedulers such as StepLR or Cosine, and its design choices are not sufficiently justified. The authors should compare LR trajectories and time to accuracy curves under matched initial and final LRs, and include an ablation of the adaptation trigger, smoothing window, and thresholds.

### Questions
1. What distinct roles do OSA and SSA play, and why are both required rather than one?

2. What are the exact backward derivatives for the approximation functions $g(x)$ and $h(x)$, and are they clipped or scheduled during training?

3. How does the learned LR trajectory differ from StepLR, Cosine, and OneCycle when initial and final learning rates are matched?

4. Can you provide component ablations OSA only, SSA only, LR only, OSA plus SSA, and the full method on at least one CIFAR and one ImageNet setting?

### Soundness
3

### Presentation
3

### Contribution
3

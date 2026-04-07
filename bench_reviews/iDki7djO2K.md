## Summary
This paper proposes a novel, general definition of forgetting in machine learning as a violation of self-consistency in a learner's predictive distribution over future experiences. This yields a measure, the propensity to forget, which the authors empirically validate across regression, classification, generative modeling, continual learning, and reinforcement learning, demonstrating that forgetting is pervasive and influences learning efficiency.

## Strengths
- **Novel and unifying theoretical framework:** The paper provides the first algorithm- and task-agnostic definition of forgetting based on predictive self-consistency, moving beyond paradigm-specific definitions and offering a principled foundation for analyzing information retention. The formalism is carefully constructed (Sections 3–4) and motivated by insightful thought experiments (Appendix C).
- **Comprehensive empirical validation:** The paper supports its theory with experiments across diverse learning settings (supervised, generative, RL, CL), consistently showing non-zero forgetting and revealing dynamics such as trade-offs with efficiency and spikes at task boundaries. This breadth strongly supports the claim that forgetting is a fundamental property of learning.
- **Actionable insights and measure:** The derived propensity to forget measure varies meaningfully with hyperparameters (e.g., momentum, batch size, architecture) and correlates with learning dynamics, providing a new tool for analysis. The empirical demonstration that optimal training efficiency often occurs at non-zero forgetting is a particularly intriguing finding.

## Weaknesses
- **Limited scalability and practical utility of the measure:** The propensity to forget requires computationally expensive particle-based rollouts (e.g., 1000 particles, k=40 steps) as described in Algorithm 1 and Appendix D.1. The paper does not address the feasibility of applying this measure to large-scale models (e.g., modern LLMs or vision transformers) or discuss approximations that would make it practical for real-time analysis.
- **Insufficient empirical scale to fully support claims:** While the experiments cover multiple paradigms, they are largely on simple tasks (sinusoid regression, two-moons classification, cartpole RL). The single CIFAR-10 experiment (Figure 11) is a step toward larger-scale validation but is not enough to substantiate the claim that "forgetting is everywhere" in deep learning. More challenging benchmarks are needed to generalize the findings.
- **Lack of comparison with existing forgetting metrics:** The paper does not quantitatively compare its measure against standard forgetting metrics (e.g., average forgetting, backward transfer) from continual learning literature. Such a comparison is necessary to demonstrate that the proposed measure better disentangles forgetting from backward transfer or provides unique insights beyond established metrics.
- **Theoretical scope and interpretation in non-stationary environments:** The definition treats any violation of self-consistency as forgetting, which may include rational adaptation in non-stationary or model-misspecified settings. While thought experiments (Appendix C) address some edge cases, the interpretation under realistic distribution shifts or model misspecification remains unclear, and the hybrid distribution \(q_e\) is not fully specified for complex environments.

## Nice-to-Haves
- Ablation study on the sensitivity of the measure to the number of particles and horizon \(k\) to ensure robustness.
- Visualizations of predictive distribution evolution for a simple regression task to concretely illustrate what specific capabilities are forgotten.
- Discussion of how the measure could be approximated more efficiently (e.g., via fewer particles, shorter horizons) for practical use as a diagnostic tool or regularizer.

## Removed Points
*These points are flagged to be removed, treat them with caution.*
- **Overstatement of novelty:** The claim that "no unified definition has emerged" is strong but supported by the paper's review of existing definitions and the genuinely new perspective offered.
- **Complexity of the two-update formalism:** The distinction between learning-mode \(u\) and inference-mode \(u'\) updates is justified to separate belief updates from auxiliary state evolution and is necessary for the theory.
- **Missing broader impact discussion:** While a discussion of societal implications is often valuable, this foundational paper focuses on theoretical and empirical contributions; however, a brief consideration of potential impacts would be beneficial.
- **Demands for user studies or extensive theoretical proofs beyond scope:** The paper appropriately combines theoretical formalism with empirical validation; requiring additional theoretical derivations or user studies is not standard for this type of contribution.

## Novel Insights
The paper's core insight is that forgetting can be characterized as a lack of self-consistency in a learner's predictive distribution, which naturally yields a general measure. This reframes forgetting from a failure mode limited to specific settings to a fundamental property of learning dynamics. The empirical finding that optimal training efficiency often occurs at non-zero forgetting suggests that forgetting is not merely a negative phenomenon but a regulated aspect of learning that can be beneficial, providing a new perspective for analyzing and designing learning algorithms.

## Suggestions
1. Conduct experiments comparing the propensity to forget with existing forgetting metrics on standard continual learning benchmarks (e.g., Split MNIST/CIFAR) to validate its advantages in disentangling forgetting from backward transfer.
2. Demonstrate the measure on at least one larger-scale dataset and architecture (e.g., ResNet on CIFAR-100) to show scalability and provide more convincing evidence for the pervasiveness of forgetting in deep learning.
3. Include a discussion on how to approximate the measure more efficiently (e.g., via variance reduction techniques) and its potential use as a diagnostic tool or regularizer in algorithm design.
4. Clarify the interpretation of the definition under model misspecification and non-stationary environments, possibly via additional thought experiments or analysis in Section 4.2.
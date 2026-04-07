## Summary
This paper bridges classical information theory with modern representation learning by proposing a learnable three-channel codec based on the Gray-Wyner network. It theoretically bounds lossy common information, derives an optimization objective for the transmit-receive rate tradeoff, and validates the approach on synthetic data and vision benchmarks, demonstrating reduced redundancy compared to independent coding.

## Strengths
- **Theoretical grounding and novel extension:** The paper provides a principled, information-theoretic foundation by deriving new bounds for lossy common information (Theorem 1) and an optimizable objective for the transmit-receive tradeoff (Theorem 2). This is a significant extension of classical Gray-Wyner theory to the learned representation setting.
- **Comprehensive empirical validation:** The method is rigorously evaluated, first on a synthetic dataset to validate control over the tradeoff and architectural ablations, then on controlled edge cases (colored MNIST), and finally on two challenging real-world vision task pairs (Cityscapes and COCO). The results consistently show the method's ability to reduce redundancy and navigate the tradeoff.

## Weaknesses
- **Empirical gap to theoretical bounds:** The empirical rates achieved on the synthetic dataset are notably higher than the theoretical rate-distortion limits (Figures 3, 9, Tables 2-5). While the paper acknowledges this common issue in learned compression, a deeper analysis of the sources of this gap (e.g., entropy model suboptimality, quantization, architecture capacity) is missing and would strengthen the work.
- **Limited quantitative analysis of disentanglement:** For the real-world vision experiments, the paper lacks a quantitative analysis of what information is captured in the common versus private channels (e.g., via mutual information estimates or auxiliary task probes). While qualitative MNIST reconstructions (Fig. 10) are provided, quantitative evidence for the claimed disentanglement on complex tasks would solidify the core contribution.

## Nice-to-Haves
- **Extension to more than two tasks:** The conclusion mentions the exponential scaling of channels for more tasks as a limitation. A preliminary experiment or concrete architectural sketch for a three-task scenario would help readers assess the framework's scalability.
- **Hyperparameter sensitivity analysis:** A more systematic ablation of the tradeoff parameter β and the auxiliary loss weight γ, especially on the vision benchmarks, would provide clearer guidance for practitioners.
- **Comparison to broader multi-task learning baselines:** While the paper appropriately compares to its own derived baselines (Joint, Independent), a comparison to a modern multi-task learning method that learns shared representations (without explicit rate constraints) could better contextualize the practical value of the rate-distortion efficiency gained.

## Removed Points
*These points are flagged to be removed; treat them with caution.*
- **Criticism of restrictive Markov assumptions:** The paper initially states Markov conditions (Eq. 1) but explicitly removes this requirement in Section 3.3, stating the architecture provides access to both sources for all branches. Therefore, this is not a valid weakness.
- **Demand for experiments on task conflicts or negative transfer:** The paper's scope is tasks with some common information. Testing on antagonistic tasks is outside its stated focus.
- **Request for theoretical proofs of representation compatibility:** The paper's core contribution is empirical and algorithmic; Appendix C provides a theoretical discussion, but demanding formal proofs for this analysis imposes an arbitrary rigor requirement not standard for this type of work.
- **Criticism of formatting/style nitpicks:** Minor typographical errors (e.g., in the rate-distortion function definition) do not constitute a substantive weakness.

## Novel Insights
The paper's key novel insight is the operationalization of the Gray-Wyner rate region's transmit-receive tradeoff via a tunable parameter (β) in a neural network optimization objective. This provides a direct, learnable mechanism to navigate the fundamental information-theoretic tradeoff between total bitrate and the bitrate required when tasks are decoded separately. The synthetic experiments (Fig. 3a) clearly demonstrate this control, showing the common channel rate moving from above to below the empirical mutual information as β shifts from 1 to 2.

## Suggestions
- **Analyze the rate gap:** In the discussion or appendix, provide a focused analysis hypothesizing why the empirical rates diverge from theoretical bounds (e.g., limitations of the entropy model, quantization, or function family capacity). This would turn a noted limitation into a constructive direction.
- **Quantify channel information for vision tasks:** Perform an additional analysis, perhaps using a simple proxy, to estimate the task-relevant information contained in each channel for the Cityscapes/COCO experiments. This would provide concrete evidence for the learned disentanglement.
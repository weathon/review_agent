# SoCo: Progressive Spectrum Optimization for Large Language Model Compression

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Large language models (LLMs) have demonstrated remarkable capabilities, yet prohibitive parameter complexity often hinders their deployment. Existing singular value decomposition (SVD) based compression methods equate singular values with component importance, an assumption that often fails to correlate with downstream task performance. In this work, we introduce SoCo (Singular spectrum optimization for large language model compression), a novel framework that learns to rescale SVD components. Concretely, we employ a learnable diagonal matrix to assign importance scores and introduce Progressive Spectrum Optimization, a principled strategy that operates in a single, continuous training run. Inspired by heuristic optimization, this process guides the learnable scores through distinct functional phases—from an initial exploration of the solution space, through an oscillation refinement, to a final, decisive sparsification—thereby navigating the complex optimization landscape to balance compression and performance. Thanks to this adaptive process, SoCo prunes components based on their learned importance, rather than a fixed order. More importantly, amplified scores on preserved components allow them to compensate for the information loss from pruning. Experimental evaluations across multiple LLMs and benchmarks demonstrate that SoCo surpasses state-of-the-art methods in large language model compression.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SoCo (Singular spectrum optimization for large language model Compression), a novel compression framework for large language models. Unlike traditional SVD-based methods that simply truncate components by singular value magnitude order, SoCo employs a learnable diagonal matrix to reassess the importance of SVD components, drawing inspiration from heuristic optimization algorithms. The method optimizes importance scores through three phases within a single training run: compression-driven exploration, oscillatory refinement, and decisive sparsification, effectively avoiding local optima that direct optimization can easily fall into. Unlike existing methods, SoCo not only prunes components based on learned importance but also amplifies the scores of preserved components to compensate for pruning loss.

### Strengths
- SoCo shows substantial improvements over state-of-the-art methods, with particularly notable gains at higher compression ratios.
- SoCo is evaluated on multiple LLMs across diverse architectural families showing consistent effectiveness across different model architectures.

### Weaknesses
- The paper lacks comparison with quantization-based compression methods and doesn't explore how SoCo could be combined with quantization in a comprehensive compression pipeline. Since quantization is a common complementary approach to model compression, this represents a significant gap in the evaluation.
- The main results compare SoCo (which involves training) against baselines without any fine-tuning, creating potentially unfair comparisons. While the authors try to address this concern in Section 4.3 by comparing against LoRA-enhanced baselines, this analysis is insufficient: it only evaluates a single model on one dataset's perplexity, lacks details about LoRA fine-tuning settings, and doesn't discuss the computational overhead comparison between different methods.
- As a model compression method, the paper doesn't report inference speedup or throughput (tokens/sec) on real hardware under different settings (batch sizes, sequence lengths, etc.), which is crucial for evaluating the practical benefits of compression in deployment scenarios.

### Questions
1. How does SoCo interact with quantization techniques, and can these compression methods be effectively combined in practice?
2. Could the authors provide more comprehensive comparisons with fine-tuned baselines beyond the limited analysis in Section 4.3? Specifically, what are the LoRA fine-tuning settings, computational overhead comparisons, and results across multiple models and datasets rather than just single model perplexity on C4?
3. What inference speedup and throughput improvements does SoCo achieve on real hardware?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SoCo, a novel framework that learns to adaptively rescale and prune the singular spectrum of weight matrices in LLMs. Instead of directly truncating singular values in fix order, SoCo employs a learnable diagonal matrix of importance scores that are optimized end-to-end through a Progressive Spectrum Optimization process. This optimization proceeds through three phases: exploration, oscillatory refinement, and decisive sparsification, guided by differentiable objectives combining compression ratio, performance preservation (KL divergence), and sparsity regularization. Experiments show that SoCo consistently outperforms SOTA SVD-based methods and remains efficient in both trainable params and runtime.

### Strengths
1. The proposed SoCo framework makes the SVD-based pruning process learnable, making SVD pruning more versatile and accurate on different models and tasks, and the insights proposed are of high value to subsequent related work.
2. The Progressive Spectrum Optimization (PSO) is conceptually elegant, breaking training into distinct phases that progressively stabilize and polarize importance scores. The combination of differentiable compression and KL-based alignment is effectively balances compression with accuracy. The PSO also jointly optimizes decomposition and rescaling in an end-to-end way, improving global convergence.
3. The authors conduct comprehensive experiments, including comprehensive comparisons with similar SVD-based methods and other pruning methods on various models and tasks, and design corresponding ablation experiments for almost every component (1-2-3 phases, deviation term, functions). These experiments strongly validated the advanced nature of SoCo and the effectiveness of its design.
4. The authors also provide detailed hyperparameters and experimental settings, and deploy SoCo on real devices to test the actual acceleration performance, which improves the reproducibility and application value of the work.

### Weaknesses
1. It’s not fully explicit how the trainable deviation $d$ is applied to $W'$, more implementation specifics (for example, are there similar constraints on $d$ during PSO) would aid reproducibility for diverse codebases.
2. The proposal of PSO mentioned in 3.3 seems intuitive. Is there any further formal theoretical guarantee, such as the existence or convergence of the final performance?
3. Comparison with non-SVD pruning methods is limited. Table 4 only shows the performance on PPLs, without comparing performance on multiple downstream tasks like other results. It would be helpful if the authors could provide more detailed results and analysis.

### Questions
1. Training largely relies on WikiText-2 as the calibration corpus; it is small and stylistically narrow relative to real deployment. How sensitive are results to calibration data scale/domain? 
2. Have you tried to design different thresholds for different layers (the activation distributions of network layers with different depths are significantly different)? Any benefit to adaptive thresholds vs preset?
3. Is SoCo compatible with commonly used model quantization methods, such as GPTQ? Can subsequent LoRA fine-tuning further restore performance after SoCo pruning (especially at high compression ratio)?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SoCo, a framework for compressing large language models (LLMs) by learning to rescale and prune SVD components in the model's weight matrices. Rather than solely truncating the smallest singular values as in standard SVD-based compression, SoCo introduces learnable importance scores (via a diagonal matrix) and proposes a progressive spectrum optimization procedure inspired by heuristic optimization algorithms like simulated annealing. The process iteratively explores, refines, and sparsifies the importance assignments, aiming to achieve a better trade-off between model size and downstream task performance. Empirical results across a range of LLMs and benchmarks consistently show improved perplexity and task accuracy over several SVD-based and pruning baselines, especially at high compression ratios.

### Strengths
1. Sufficient Experiments: The experiments include a wide variety of settings, multiple models, and strong SVD/Pruning baselines. Figure 1 and Table 2 clearly show that SoCo outperforms all tested baselines, especially at aggressive compression rates.
2. Principled Approach and Insightful Design: The paper identifies and addresses a core limitation of classical SVD-based compression, the disconnect between singular value ordering and real downstream task importance. The proposed introduction of a learnable diagonal score matrix is a conceptually sound way to endow SVD compression with task-awareness.

### Weaknesses
1. A more comprehensive analysis is required to elucidate why SoCo attains such outstanding performance. Although the paper suggests that the ordering of singular values may not necessarily correlate with downstream task performance, Figures 3, 4, and 9 reveal that SoCo not only learns to prune the SVD components within the model’s weight matrices but also to rescale these components and reallocate pruning ratios across layers. Additional analytical experiments are necessary to quantify the contribution of each component to SoCo’s overall performance and to empirically validate the claim that the ordering of singular values may not directly correspond to downstream task performance.
2. The paper uses specified training steps rather than any criteria to determine when to transition from Phase 1 and Phase 2 to Phase 3, and whether to end Phase 3 training. As these standards can significantly influence model performance, where overtraining may cause overfitting and insufficient training may lead to suboptimal convergence, the authors should provide explicit guidelines or empirical criteria for deciding when to switch training phases and when to stop training.
3. Lines 202-204 mention that SoCo introduces a trainable deviation term $d$ after transforming $W^’$. However, the paper does not explain why this term is necessary. What motivates the inclusion of $d$. Does SoCo encounter any critical issues that require introducing additional trainable parameters to mitigate them?

### Questions
See Weaknesses

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
4

### Summary
This paper introduces SoCo, a framework for compressing large language models (LLMs) by learning to rescale and prune SVD components in the model's weight matrices. Rather than solely truncating the smallest singular values as in standard SVD-based compression, SoCo introduces learnable importance scores (via a diagonal matrix) and proposes a progressive spectrum optimization procedure inspired by heuristic optimization algorithms like simulated annealing. The process iteratively explores, refines, and sparsifies the importance assignments, aiming to achieve a better trade-off between model size and downstream task performance. Empirical results across a range of LLMs and benchmarks consistently show improved perplexity and task accuracy over several SVD-based and pruning baselines, especially at high compression ratios.

### Strengths
1. Sufficient Experiments: The experiments include a wide variety of settings, multiple models, and strong SVD/Pruning baselines. Figure 1 and Table 2 clearly show that SoCo outperforms all tested baselines, especially at aggressive compression rates.
2. Principled Approach and Insightful Design: The paper identifies and addresses a core limitation of classical SVD-based compression, the disconnect between singular value ordering and real downstream task importance. The proposed introduction of a learnable diagonal score matrix is a conceptually sound way to endow SVD compression with task-awareness.

### Weaknesses
The paper does not specify any criteria for determining when to transition from Phase 1 and Phase 2 to Phase 3 training, or when to terminate Phase 3. As these standards can significantly influence model performance, where overtraining may cause overfitting and insufficient training may lead to suboptimal convergence, the authors should provide explicit guidelines or empirical criteria for deciding when to switch training phases and when to stop training.

### Questions
1. A more comprehensive analysis is required to elucidate why SoCo attains such outstanding performance. Although the paper suggests that the ordering of singular values may not necessarily correlate with downstream task performance, Figures 3, 4, and 9 reveal that SoCo not only learns to prune the SVD components within the model’s weight matrices but also to rescale these components and reallocate pruning ratios across layers. Additional analytical experiments are necessary to quantify the contribution of each component to SoCo’s overall performance and to empirically validate the claim that the ordering of singular values may not directly correspond to downstream task performance.
2. Lines 202-204 mention that SoCo introduces a trainable deviation term $d$ after transforming $W^\prime$. However, the paper does not explain why this term is necessary. What motivates the inclusion of $d$. Does SoCo encounter any critical issues that require introducing additional trainable parameters to mitigate them?

### Soundness
2

### Presentation
2

### Contribution
2

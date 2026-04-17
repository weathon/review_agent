# OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generation tasks. However, their massive parameter scale leads to significant resource consumption and latency during inference. Post-training weight-only quantization offers a promising solution by reducing model size and accelerating token generation through alleviating the memory-bound issue. Nevertheless, there are inherent systematic outliers in weights, and although some efforts have attempted to address them, such as scaling and rotation, the performance of low-bit quantization remains far from satisfactory. In this paper, we propose Outlier Self-Absorption Quantization (OSAQ), which performs second-order low-rank derived additive weight suppression for low-bit weight-only LLM quantization. Specifically, we observe that Hessian exhibits low-rank consistency across different inputs, with certain directions persistently lacking strength. Leveraging this property, we construct an additive weight transformation based on the Hessian’s null space, thereby suppressing weight outliers without affecting the task loss. This additive transformation can be absorbed into the weights offline, requiring no inter-layer transformations and introducing no inference overhead. Moreover, the construction is efficiently achieved by a closed-form solution, without resource-intensive training or iterative procedures. Extensive experiments across models of varying scales and tasks are conducted, and the results show that OSAQ effectively suppresses outliers and improves low-bit quantization performance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose a method to mitigate outliers in the weights of LLMs. They first observe that the null space of the loss Hessian is relatively stable. Building on this, they add $\Delta W$ within that null space to minimize $l-\infty$ of $W+\Delta W$. They further approximate the objective utilizing the Softmax-$\infty$ Objective Appoximation and derive a closed-form solution. Experiments verify its effectiveness on the LLaMA-series models in terms of PPL, QA tasks.

### Strengths
1. The motivation is clear and strong, and the proposed method is established based on the observation.
2. The paper is well-organized overall, making it easy to follow.
3. The experiment and visualization verify the effectiveness of the proposed method.

### Weaknesses
1. The settings and detailed visualization procedure used in Figure 1(b) remain unclear. The authors should elaborate to clarify the observation. The norm length appears odd in the 2D representation.
2. It seems somewhat contradictory that the null space of  $H$ is obtained via the null space $X^TX$, yet the authors claim that the null space of $X$ is not consistent in Figure 1(b). 
3. Sensitivity/robustness of approximation to obtain the null space with using approximation method or not ($X^TX$ vs $H$), the threshold, the size of the calibration set, and/or the chosen subspace dimension should be reported.
4. The gains appear limited. Across models and bit widths, the improvements are not consistently stable or significant.
5. The authors mentioned the assessment on MTBench in Line 296, but I cannot find any results regarding it.

### Questions
1. Can the proposed method be applied to weight–activation quantization settings? Rotations may change inter-layer activation inputs, which could make implementing $W+\Delta W$ harder.
2. What is the performance of solely use OSAQ (i.e. OSAQ+RTN) to quantize LLMs? To what extent does the method rely on other quantization methods?

### Soundness
2

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
4

### Summary
This paper addresses LLMs’ high inference resource use and latency via post-training weight-only quantization, tackling the key obstacle of weight outliers (poorly handled by existing scaling/rotation methods). It proposes OSAQ, which uses Hessian’s low-rank consistency across inputs to identify a stable null space. By linearly combining null-space vectors, OSAQ builds an additive weight transformation that suppresses outliers without task loss, absorbs offline (no inference overhead), and uses a closed-form solution (no heavy training). Experiments show OSAQ boosts low-bit quantization—e.g., 2-bit OSAQ+GPTQ cuts perplexity over 40% vs. vanilla GPTQ.

### Strengths
1. Starting from the perspective of weights, the authors perturb the weights using addition while maintaining an approximate invariance of the output, thereby smoothing the weight distribution. Their method is compatible with weight-calibrating approaches such as GPTQ, and the authors have verified the effectiveness of the method through extensive experiments.
2. The authors' method does not introduce any additional inference overhead, and only incurs 10% to 20% overhead during the calibration process.

### Weaknesses
I believe the authors' work is very rigorous and has no obvious weaknesses.

### Questions
1. Rotation smooths the distribution by transferring outliers from one channel to other channels, so I am confused why the authors claim that rotation cannot effectively eliminate outliers？
2. How does the authors' method perform in scenarios where activation values are also quantized?
3. Can the authors' method be combined with methods based on scale and rotation?

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
3

### Summary
The paper proposes OSAQ, an additive outlier-suppression scheme guided by the Hessian null space, with a closed-form coefficient solver via a Softmax-∞ surrogate. The update is absorbed into weights and adds no inference cost, complementing scaling/rotation methods. Broad experiments on LLaMA-2/3 and instruction-tuned 123B/405B show notable low-bit gains, especially at 3-bit and 2-bit.

### Strengths
1. Novel additive paradigm grounded in low-rank consistent Hessian; clear second-order loss-invariance rationale.
2. Closed-form per-channel solution; plug-and-play with GPTQ/AWQ/QuIP; zero inference overhead.

### Weaknesses
1. The improvement on the existing methods is incremental.

2. How robust is the “stable Hessian null space” assumption across layers, models, and domains? Any quantitative measures (e.g., principal angles) across batches?

3. The article should include more thorough comparisons/combination with recent rotation families (e.g., DuQuant/QuaRot/SpinQuant) under matched settings.

### Questions
Please see the weakness.

1. How sensitive is performance to the temperature 𝜏? Any heuristic for layer-wise adaptive 𝜏?

2. What guided the Softmax-∞ surrogate choice for ℓ∞? Did you try p-norm annealing (e.g., p→∞) or direct max-margin formulations?

3. Ablations for stability of the null space vs. calibration set size (e.g., 64/256/1k samples).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes OSAQ, a Hessian-based additive transformation method for mitigating systematic outliers in low-bit weight quantization of LLMs. The authors claim that by leveraging the low-rank consistency of the Hessian and operating within its null space, one can add a closed-form, inference-free correction term that absorbs outlier effects without retraining or extra computational cost.

### Strengths
- Extending Hessian-based quantization ideas toward a closed-form additive formulation is conceptually interesting and novel.
- If the proposed null-space additive transformation indeed preserves task loss and can be absorbed into model weights, it would be a practically appealing solution.

### Weaknesses
-  The experiments do not include key state-of-the-art methods addressing outliers, such as QuaRot, DuQuant, and SpinQuant, under a unified evaluation setup.
-  The numerical stability of the closed-form solution, especially for large layers or block-wise quantization, is not discussed.

### Questions
- Under what conditions does the transformation guarantee that the first-order term is zero, ensuring loss invariance? Does this assume the model is at or near a local optimum?
- If the calibration and inference distributions differ, does the null-space property still hold? Could you bound or quantify the resulting error?
- What is its relationship with the trace-based selection strategy used in HAWQ-V2, and what are the corresponding advantages or disadvantages?

### Soundness
2

### Presentation
3

### Contribution
2

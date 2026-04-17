# StatQAT: Statistical Quantizer Optimization for Deep Networks

- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
Quantization is essential for reducing the computational cost and memory usage of deep neural networks, enabling efficient inference on low-precision hardware. Despite the growing adoption of uniform and floating-point quantization schemes, selecting optimal quantization parameters remains a key challenge, particularly for diverse data distributions encountered during training and inference. This work presents a novel statistical error analysis framework for uniform and floating-point quantization, providing theoretical insight into error behavior across quantization configurations. Building on this analysis, we propose iterative quantizers designed for arbitrary data distributions and analytic quantizers tailored for Gaussian-like weight distributions. These methods enable efficient, low-error quantization suitable for both activations and weights. We incorporate our quantizers into quantization-aware training and evaluate them across integer and floating-point formats. Experiments demonstrate improved accuracy and stability, highlighting the effectiveness of our approach for training low-precision neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes utilizing analytic quantizers (for weights) and iterative one (for activation) in quantization aware training.
Given the statistics (of weights during QAT), the analytic quantizer for the target data format like FP4 chooses the best quantization parameters (clipping threshold and zero) from pre-computed ones assuming Gaussian distributions.
The experiments show SOTA results.

### Strengths
The experimental results are quite impressive.
Especially, it looks nice that the proposed method is effective in tensor-wise cases where quantization gets simpler than in channel-wise ones.

### Weaknesses
Pre-computing quantization parameters based on known distributions and utilizing them (based on the statistics obtained) during QAT is not new.
For instance, in [1], the authors propose an idea called statistics-aware weight binning (SAWB) which quantizes weights based on Gaussian assumption during QAT.

[1] Bridging the Accuracy Gap for 2-bit Quantized Neural Networks (QNN), https://arxiv.org/pdf/1807.06964

Compared with [1], one key difference will be FP4 since it was not available when [1] was published.
It would be nice to discuss what are 'fundamentally' new problems, in case of FP4, for statistics-aware quantization.
Proposing a statistics-aware method for FP4 based on a floating point-aware analytical analysis (which looks similar to the one in [1] in principle) looks rather weak to me as novelty 
since only number representations are different between Int [1] and Float while the main idea of utilizing the pre-computed results (optimal quantization parameters for a given distribution) looks the same.

### Questions
It would be nice to explain how the proposed work advances QAT in terms of analytic analyzer 
in comparison with existing QAT works based on analytic quantizers and Gaussian (or pre-determined) distribution assumption.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors discuss weight and activation quantization in NNs. They analyze the errors of different quantization choices (uniform, FP) and derive a k-means style approach to find appropriately-spaced cluster centers given observed data (weights & activations). They derive iterative and - if data (weights) can be assumed normally distributed - analytical update rules for those cluster centers. 
The authors perform QAT experiments with their proposed methods and compare against baselines.

### Strengths
This work is well-motivated, the problem of QAT is highly relevant to deploy large models efficiently and the exposition is well-done.
Considering the latency of the quantizer is an important contribution, especially since interative and analytic perform comparably.

### Weaknesses
The only weakness I can identify is the limited empirical evaluation. Looking at e.g. ParetoQ, there is an opportunity to study <4bit quantization - as well as the opportunity to include more models & datasets to improve the robustness of this work.

### Questions
I believe in EQ 7 the lower limit of the second integral should be $s(N-5/2) + z$
substituting $t_{N-2}=s(N-2-1/2+z)=s(N-4/2-1/2+z) = s(N-5/2+z)$

### Soundness
3

### Presentation
4

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
The paper proposes StatQAT, a statistical framework for choosing quantization parameters for both uniform and FP schemes.
The core is an analytical decomposition of quantization error into clipping + stepping terms, iterative updates for scale/shift and updates that precompute an optimal clipping constant via SNR maximization, then set per-tensor scales in one shot.
Varius experiments on llms as well as resnet18.

### Strengths
### Clear error model
for both uniform and floating-point quantizers, including a tractable stepping-error mixture for FP formats (Sec. 3.2, Eqs. 15–16), which is rarely derived cleanly in QAT papers. The SNR-based perspective is useful (Fig. 1, p. 7). 

### Simple, practical update rules
Also shown in appendix how faster the analitical compared to iterative

## Broad Evaluation
Two models with two sizes make the impact broad across models and datasets.

### Weaknesses
1. The iterative uniform update is essentially a **constrained 1-D k-means** (Lloyd–Max) with fixed spacing , and the FP version mirrors that idea on a scaled FP grid. This feels like a straightforward specialization of classic quantizer optimization to the uniform/FP grids rather than a new algorithmic principle.

2. The “first closed-form analytic solution for FP quantizers in QAT” (Abstract, Intro) seems overstated. optimal **C** is chosen by numerical search on a derived SNR expression (eq 22). Please tone down the claim or clarify precisely what is closed-form.

3. **Positioning vs. prior work** - Prior art already optimizes clipping/scale during QAT (e.g., LSQ/PACT) and analyzes error (Lloyd–Max, optimal clipping). you list  them in the intro, but the manuscript should articulate what is new beyond applying an SNR-driven selection and  demonstrait that a single-step analytic update matches iterative methods under QAT constraints.

4. **Clarity & presentation** - Section 2 is long background, Section 3 sometimes re-derives known pieces (for example, uniform steping error). While i understand that it's improtant for deriving your decomposed error, you can simply reffer to literuture or move to the appendix (similar to some of the solved known integrals you've put there), as it take major part of the paper lenghts.
**Figures** should be PDF and not images, so the quality remain high when zooming, as currently the quality of the figures are poor.

5. Reproducability - I failed to find detalis about the datased you used for fine-tuning the LLMs, with details about it (some QAT uses only the data without labels, or with artificial labels, the size matter a lot since it's much heavier on the saving side of your non iterative suggestion).

6. Where does the gain come from? I would be happy to see more ablations that explain that.. the SNR curves (Fig. 1) suggest robustness/benefit from better clipping. It would help to add per-layer scale histograms, SNR per layer, or an analysis of failure cases vs min-max/LSQ to better understand the effects.

### Questions
I incorporated my questions in the Weaknesses section.

But regardless, can you provide even a very small scale correlation study why ablation over resnet and cifar translate into LLMs?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper demonstrates a statistical framework for optimizing quantization parameters for uniform and floating-point quantization and proposes two quantization procedures, namely, iterative quantizers for arbitrary data distributions (activations) via alternating updates of scale and shift for each time step, and analytic quantizers using a closed form solution for Gaussian-like weight distributions. These methods can be integrated into Quantization-Aware Training (QAT). The method is validated using ResNet, MobileLLM, and Llama 3.2 and the quantizers are compared to minmax and normal quanitzers.

### Strengths
1. Nice to see inclusion of both integer and floating-point formats, and both activations and weights.
2. The writing is clear and easy to follow.
3. The statistical foundation for quantizer design is interesting.

### Weaknesses
1. Limited evaluation - the models evaluated are very limited, not clear if the technique will scale to larger models used today beyond ResNet-18 and larger than 3B parameter LLMs. Broader domain coverage could strengthen claims.
2. Limited comparison - I don't see comparisons with more recent SOTA techniques like SpinQuant for PTQ, LSQ for QAT, etc. 
3. Runs on CIFAR-10 can be noisy, I don't see any characterization of noise.

### Questions
1. The labels used for Fig 2 (left) seem to be incorrect.
2. Can you comment on how this can be combined with existing QAT techniques like LSQ, and include comparisons with some of the techniques mentioned above?

### Soundness
3

### Presentation
3

### Contribution
2

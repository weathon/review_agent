# Metis: Training LLMs with FP4 Quantization

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
This work identifies anisotropy in the singular value spectra of parameters, activations, and gradients as the fundamental barrier to low-bit training of large language models (LLMs). These spectra are dominated by a small fraction of large singular values, inducing wide numerical ranges that cause quantization bias and severe spectral distortion, ultimately degrading training performance. This work presents \emph{Metis}, a spectral-domain quantization framework that partitions anisotropic spectra into narrower sub-distributions for independent quantization, thereby reducing errors and preserving spectral structure. To minimize overhead, 
Metis leverages two key properties of the dominant spectral subspace: preservation via sparsely random sampling and preservation via random projection, reducing decomposition cost to a negligible level. On LLaMA-3 8B trained with 100B tokens, Metis enables robust W4A4G4 training with FP4 quantization of weights, activations, and gradients, yielding only a 0.4\% training loss gap and a 0.1\% degradation in downstream accuracy relative to BF16. Beyond matching BF16 fidelity, Metis also surpasses Nvidia’s FP4 recipe, consistently achieving lower loss and higher downstream accuracy while incurring significantly lower computational overhead. 
The code implementation for Metis is available at: \url{https://github.com/sii-research/Metis}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Metis, a spectral-domain quantization framework for FP4 training of large language models. It identifies anisotropy in the singular value spectra of weights, activations, and gradients as a key barrier to low-bit training. Metis partitions these anisotropic spectra into narrower sub-distributions for independent quantization, greatly reducing distortion and preserving spectral structure. It introduces efficient decomposition techniques—sparse random sampling and random projection—to minimize overhead. Experiments on GPT-2 (130M, 1.1B) and LLaMA-3 (8B) show that Metis achieves only 0.4% higher loss and 0.1% accuracy degradation relative to BF16, outperforming NVIDIA’s FP4 recipe.

### Strengths
Addresses a fundamental bottleneck (anisotropy) in ultra-low-bit LLM training.

The spectral-domain perspective is novel and well-motivated.

Theoretical analysis and empirical evidence are solid and clearly explained.

The proposed techniques (sparse sampling, random projection) make FP4 training computationally feasible.

Demonstrates strong results on multiple models, surpassing existing FP4 baselines.

### Weaknesses
Evaluation is limited to up to 8B models. Since anisotropy worsens with scale, validating on 70B+ models would better demonstrate scalability.

End-to-end efficiency (e.g., wall-clock speedup, memory usage) is not quantitatively reported; the paper mainly provides theoretical overhead analysis.

More comparison with other recent FP4/low-bit training frameworks (e.g., HALO, SVDQuant) would strengthen the positioning.

Experiments rely on FP4 simulation in BF16, not native FP4 hardware, which limits the practical verification of speed and stability claims.

### Questions
Questions

How well does Metis scale to larger models (e.g., 70B or 100B)?

What are the measured training throughput and memory savings compared with NVFP4 or FP8 baselines?

What is the difference bitween the training method of Metis and BitNet-family (binary of tenary training)?

### Soundness
3

### Presentation
3

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
This paper introduces Metis, a spectral-domain quantization framework designed to enable robust 4-bit (FP4) training for Large Language Models (LLMs). The authors identify anisotropy—the domination of parameter, activation, and gradient spectra by a few large singular values—as the fundamental barrier to low-bit training. Metis solves this by partitioning the anisotropic spectra into narrower sub-distributions, which are then quantized independently to reduce errors and preserve spectral structure. To make this computationally feasible, Metis uses sparse random sampling and random projection to reduce the cost of spectral decomposition.

### Strengths
1. This paper provide comprehensive analysis to gradient, weight, activation based on SVD, reveaing that a few singular values dominating the spectrum.
2. This paper demonstrate that absmax quantization favors larger values, and bring significantly bias to small values.
3. The performance is impressive, achieving nearly lossless performance in NVFP4 training.

### Weaknesses
1. This paper focus on FP quantization, but the quantization illustrated in Lines 181 are INT quantization formulation.
2. Two different "X" symbol in line 269 and line 270.
3. The writing in section 3.2 is unclear, it should be emphasize which GeMM are execute in 4-bit and which GEMM are execute in BF16.
4. Metis introduces too much additional operation, though the computational FLOPs is relatively small, the additional memory overhead or IO overhead are not negligible. This should be discussed.

### Questions
1. what is the meaning of   $\bar{X} in Eq.(2) ,as well as other symbol with bar? This paper do not give definition for these symbols.
2. Why NVFP4 training can achieve lower loss compared to BF16 training in GPT models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper begins by studying the spectral space of weights, activations, and gradients. The findings show a clear "anisotropy" in the singular distribution, which causes wide numerical ranges, hence high low-bit quantization errors. METIS is then designed to 1) decompose the matrices into a low-rank component with large singular values, and a residual component with smaller singular values, 2) do so efficiently using sparse random sampling and random projection, and 3) quantize each component separately. The results show that NVFP4 METIS is able to closely recover BF16 baseline across GPT/Llama models, and outperform naive NVFP4 baseline.

### Strengths
1. I believe METIS is generally a very strong paper on the analysis side, and some of their findings are novel and interesting.

2. Very clear presentation.

3. Clean and clear ablation studies, including showing that the residual term in the weights does not "develop" large singular values.

### Weaknesses
1. The asymptotic runtime analysis, in the era of Blackwell GPUs, is not convincing at all, especially given all the memory-bound operations (e.g., multiple quantization calls, etc). On the other hand, the authors claim that NVIDIA's FP4 stack was not released at the time of submission, but cutlass already supported NVFP4 matrix multiplication in April 2025 [1]. I think speedup numbers, including layer-wise and end-to-end, are crucial for this paper.

2. Some minor concerns exists, see questions.

[1] https://github.com/NVIDIA/cutlass/releases/tag/v3.9.0

### Questions
1. Could the authors provide real-world speedup number? I understand the rebuttal timeframe is too short for writing kernels, but from my point of view, this is critical. The authors can maybe take advantage of the kernels in [2].

2. Several times in the paper it was mentioned that gradients are being quantized. I think this is ambiguous, as it could mean both activation gradients and weight gradients (I believe the authors mean the former). Can you please make this clear?

3. The authors mention that l >> k (around line 222). What will happen if that's not the case? Note that this actually happens quite often, especially in research (with gradient accumulation).

[2] https://github.com/IST-DASLab/qutlass 

In general, my biggest concern regarding this paper is about the runtime analysis. If my concern is addressed, I will consider increasing my score, possibly significantly. But I don't think the paper is complete without it.

### Soundness
2

### Presentation
4

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
The paper tackles the core problem of why end-to-end quantized (specifically FP4) training of LLMs underperforms. Quantization plus strongly anisotropic tensors (weights, activations, and gradients) leads to disproportionate distortion of the small singular directions. They propose Metis, which does a cheap, sampled spectral decomposition per step, splits tensors into a small low-rank branch and a quantization-friendly residual, and quantizes the resulting narrower parts separately. This gives an almost fully quantized setup (W4A4G4) and, on LLaMA-3 8B pretrained for 100B tokens, the method stays close to BF16 with modest benchmark drops. Experiments are quite strong (8B training, meaningful ablations), and the intuition is clear and well-argued.

### Strengths
1. The paper correctly identifies spectral anisotropy (few large singular values + many small ones) as the real reason FP4 fails, and the “separate big directions, then quantize” story is both simple and compelling.

2. Using sequence subsampling and randomized projection to estimate the dominant subspace is a sensible way to make SVD affordable; even if the theoretical bounds are loose, the justification is acceptable for practical use.

3. 8B pretraining at 100B tokens, even if not at full budget, is a significant scale for a quantization paper, and much stronger than GPT-2-only demonstrations.

4. Doing weights, activations, and gradients almost FP4 is a decent advance over weight-only or weight+activation.

5. The ablation that removes spectral decomposition from W/A/G and shows that gradients matter the most is especially convincing and supports their main diagnostic.

6. Modest but consistent benchmark results. The reported downstream gaps are small but believable given the setup.

### Weaknesses
1. There is no actual wall-clock or kernel-level overhead comparison, despite the fact that FP4 kernels (NVFP4-style) are already available in systems like QuTLASS and vLLM. Even an FP4 (not NVFP4) timing would have been possible, as it has similar FLOPS properties to NVFP4.


2. At inference time, you still need the fancy activation-side spectral quantization. That limits how much speedup and how much deployment simplicity you actually gain, and it’s unclear how this scales.


3. The experiments are single-seeded, and no CI is provided. Especially for smaller models (GPT-2 scale), results should be multi-seeded; the reported downstream gaps are within typical variance.


4. Benchmarks are modest. Stronger discriminative tasks, such as MMLU, GSM8K, or other more challenging evaluations, would better reveal the gains from this method.


5. No comparison to recent strong PTQ like QTIP, Quip#, AQLM, OmniQuant. 

6. No comparison to recent FQT methods, such as “FP4 All the Way” or rotation-based FP4 training (like “HALO” in NVFP4). So it’s hard to situate Metis among existing efforts.


7. The paper says Hadamard incurs notable overhead and doesn’t really fix the distribution, but fast Hadamard is $ O(n \log n) $, and HALO/HALO-like methods do notably reduce kurtosis to a Gaussian level (see HALO §A.4).


8. The method is algebraically described but not presented as a concrete algorithm.


9. It would have been more informative to include validation loss curves instead of or along the training loss curve.


10. The decomposition bounds presented are quite loose; that’s acceptable as motivation, but it weakens the theory-backed framing.


11. In Fig. 1, studying relative error per singular value is not the best target, since what matters is the matrix-level reconstruction. Although Fig. 3 partially fixes this.


12. Fig. 4 “sample ratio” is underspecified. Using a percentage of sequences is a vague measure because it depends on sequence length; an $ l_k/k $ or $ l_k/m $ would be clearer.


13. Looks close to LQ-LoRA + QAT but generalized. Conceptually, the method identifies the important subspace and quantizes the rest, similar to low-rank+quantization approaches like LQ-LoRA but with SVD approximation and STE-QAT.

### Questions
1. You mention that Metis adds only modest complexity due to low-rank factors and sampled decomposition, but there is no end-to-end timing. Can you provide actual wall-clock comparisons (forward, backward, and full step) against a plain NVFP4 training baseline on the same hardware/software stack? Given the available kernels.

2. For the quantized GEMMs, how much of the extra time comes from (i) subspace estimation (sampling + projection), (ii) extra GEMM terms from the expanded $ (X_k + X_R)(W_k + W_R) $, and (iii) quantization/dequantization?

3. Can you provide a comparison against strong recent PTQ/few-shot quantization methods (QTIP, Quip#, AQLM, OmniQuant) and to fully-quantized training lines such as rotation-based FP4 (“FP4 All the Way”, HALO-like)

4. For inference, do we still have to run the spectral decomposition / special activation quantization, or can the model be exported to a simpler FP4/PTQ format? How much would the model lose if done so?

### Soundness
4

### Presentation
4

### Contribution
3

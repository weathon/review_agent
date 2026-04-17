# Bit-by-Bit: Progressive QAT with Outlier Channel Splitting for Stable Low-Bit LLMs

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
Training large language models (LLMs) at ultra–low precision remains challenging: direct low-bit quantization-aware training (QAT) often suffers from slow convergence that demands substantial training budgets, as well as quantization errors arising from heavy-tailed outlier channels and the accumulation of errors across layers. To address these issues, we present \textsc{Bit-by-Bit}, a progressive QAT framework with outlier channel splitting. Our approach integrates three key components: (1) block-wise progressive training that reduces precision stage by stage, ensuring stable initialization for low-bit optimization; (2) rounding-aware outlier channel splitting, which mitigates quantization error while acting as an identity transform that preserves the quantized outputs; and (3) microscaling groups with E4M3 scales to capture dynamic activation ranges aligned with OCP/NVIDIA practices. Furthermore, we exploit the nested structure of integer quantization grids to enable a single-run, once-for-any-precision model that can be directly deployed at multiple bit-widths without retraining.
We conduct comprehensive evaluations under both weight-only and weight–activation quantization settings. Under W2A2 quantization, Bit-by-Bit narrows the perplexity gap with full-precision models on WikiText2 to just 2.25, consistently outperforming BitDistiller by 24.19 and EfficientQAT by 20.59 on Llama2-7b. Moreover, on the Llama3 family—known for its quantization difficulty, Bit-by-Bit surpasses other QAT baselines.
Code is available in the Appendix.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Training Large Language Models (LLMs) at ultra-low precision is unstable and error-prone. This paper introduces BIT-BY-BIT, a Quantization-Aware Training (QAT) framework that stabilizes this process by progressively reducing precision in stages . The method uses rounding-aware outlier channel splitting to manage extreme values and microscaling groups to capture dynamic ranges. This approach allows the model to be deployed at multiple bit-widths without retraining and significantly outperforms existing QAT baselines.

### Strengths
1. The introduction about background and related works are comprehensive.
2. The experiments statement is clear and easy to follow.

### Weaknesses
1. This paper claims that the low-bit training is unstable. However, previous works [1,2] have shown that though low-bit trianing achieve inferior loss, the trianing is stable and loss decreases smooth.
2. Accumulation of quantization error is challenge. However, Figure 3 can demonstrate this point becuase that the magnitude of every block also increases with block index. Therefore, relative error rather than absolute error may be more appropriate for understanding error accumulation.
3. Though Bit-by-Bit achieve better performance as shown in Table 1, the comparisons are unfair. For fair comparisons, different quantization methods should use same  quantization group size to ensure same inference efficiency.

[1] Compression Scaling Laws: Unifying Sparsity an Quantization
[2] Scaling Law for Quantization-Aware Training

### Questions
1. W4A4 LLM also suffer from peformence degeneration, and it is also more practical because of hardward suppoting. Why this paper focus on W2A2 instead of W4A4? 
2. Why proposed method significantly outperforms EfficientQAT in WikiText2 perplexity, while lag behind EfficientQAT in C4 perplexity. Is this caused by over-fitting problem?
3. In my understanding, there is not hardward supporting for 2-bit GEMM. How can this paper achieve speedup in W2A2 compared to W2A16.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors present a framework to mitigate the impact of outliers during low-bit quantization-aware training (QAT). The method integrates three key components: 1) block-wise progressive training, 2) rounding-aware outlier channel splitting, and 3) a grouping strategy aligned with OCP MX and NVIDIA NVFP4 formats. Evaluation on models such as Mistral-7B and LLaMA-3.2-3B demonstrates benefits across several downstream tasks, particularly under extreme low-bit settings like W2A2.

### Strengths
- The paper is well-written, with clear logic explaining the core insights, methodology, and how each component addresses specific challenges.
- The once-for-any-precision training pipeline is a notable strength, offering a promising direction to reduce post-training costs as model sizes increase.

### Weaknesses
- Experiments are limited to several smaller LLMs, and evaluation focuses primarily on next-token prediction tasks, lacking assessment on complex reasoning benchmarks (e.g., GSM8K).
- Inference performance experiments are conducted only on Llama-3.2-1B, without details on latency analysis.
- The once-for-any-precision approach introduces a combined loss for multiple bit-widths, but no ablation study or analysis compares its performance against single bit-width training.

### Questions
- For the speed measurement in Section 4.5, could you elaborate on the experimental configuration, such as batch size and hardware parameters?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Bit-by-Bit, a progressive quantization-aware training (QAT) framework for low-bit LLMs. The method combines: (1) block-wise progressive precision scheduling to lower bit-width stage by stage; (2) a rounding-aware outlier channel splitting (OCS) that aims to preserve quantized outputs while reducing heavy-tailed errors; and (3) microscaling groups with FP8 E4M3 (MXFP) scales aligned with common hardware practice. The authors further claim a “once-for-any-precision” model via nested integer grids, enabling deployment at multiple bit-widths without retraining. Experiments on Llama-2/3 under weight-only and weight–activation settings show improvements over several QAT baselines; under W2A2, the WikiText-2 perplexity gap to full precision is reported as 2.25 and exceeds BitDistiller and EfficientQAT.

### Strengths
1. Clear writing and organization. The method and engineering choices are presented cleanly.

2. Empirical gains. The paper reports improvements on multiple LLM families.

### Weaknesses
1. Limited novelty. The main components—stage-wise progressive precision, outlier channel splitting (OCS), and microscaling—are established techniques that the paper largely combines rather than clearly re-inventing. 

2. Ablations centered on W2A16. The only detailed ablation table is explicitly for the w2a16 (weight-only) setting, whereas the headline claims emphasize ultra-low w2a2 as well; comparable w2a2 ablations are not reported in the current draft. 

3. Metric choice vs. quantization regime not analyzed. The paper compares outlier metrics (e.g., |x|²·|w| vs. max/kurtosis) within the w2a16 ablation table, but does not analyze how metric/calibration choices differ between weight-only quantization and weight-activation quantization regimes. 

4. Positioning vs prior art needs clarification. OCS is presented with prior attribution (Zhao et al., 2019), and the microscaling format is aligned with existing MX/NVIDIA conventions (E4M3); the manuscript could articulate more precisely what is theoretically new beyond these adoptions.

### Questions
As aforementioned, it is necessary to analyze how the metric/calibration objective interacts with weight-only quantization vs. weight-activation quantization. Which objective is optimal under each regime, and how sensitive are results?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper identifies the core instability issue in ultra–low-bit QAT, supported by loss-landscape visualizations (Figure 1) and block-wise error accumulation (Figure 3).

### Strengths
Provides strong experimental improvements across LLaMA and Mistral families for both w2a16 and w2a2, outperforming EfficientQAT, ParetoQ, and BitDistiller by significant margins (e.g., PPL 7.72 vs 26.06 for LLaMA2-7B w2a2)

### Weaknesses
The paper states a “small increase,” but:No FLOPs overhead numbers; No latency overhead due to channel duplication; Real deployment cost unclear, especially for dense GEMMs.

SpinQuant and QuaRot (rotation-based) work only in PTQ mode here. Would benefit from comparison to stronger recent 2-bit QAT or distillation-heavy methods.

Multi-precision loss terms are combined with weights. But interference between different quantization grids can lead to: 
competing gradient signals; smoothing of high-bit representations; More discussion needed on mitigation (e.g., curriculum or orthogonal losses).

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
2

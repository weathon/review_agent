# TurboBoA: Faster and Exact Attention-aware Quantization without Backpropagation

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4, 4

## Abstract
The rapid growth of large language models (LLMs) has heightened the importance of post-training quantization (PTQ) for reducing memory and computation costs. Among PTQ methods, GPTQ has gained significant attention for its efficiency, enabling billion-scale LLMs to be quantized within a few GPU hours. However, GPTQ's assumption of layer-wise independence leads to severe accuracy drops in low-bit regimes. Recently, BoA improved upon GPTQ by incorporating inter-layer dependencies within attention modules, but its reliance on sequential quantization across all out-channels makes it substantially less efficient. In this paper, we propose TurboBoA, a new backpropagation-free PTQ algorithm that preserves the accuracy benefits of BoA while significantly accelerating the process. The proposed TurboBoA introduces three key innovations: (i) joint quantization of multiple out-channels with a closed-form error compensation rule, which reduces sequential bottlenecks and yields more than a three-fold speedup; (ii) a correction mechanism for errors propagated from preceding quantized layers; and (iii) adaptive grid computation with coordinate descent refinement to maintain alignment during iterative updates. Extensive experiments demonstrate that TurboBoA delivers substantial acceleration over BoA while consistently improving accuracy. When combined with outlier suppression techniques, it achieves state-of-the-art results in both weight-only and weight-activation quantization. The code will be available at https://github.com/SamsungLabs/TurboBoA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenge of efficiently performing post-training quantization for Transformers, especially attention layers, without backpropagation, while maintaining high accuracy at extremely low bit-widths. It proposes TURBOBOA, which jointly quantizes multiple output channels with closed-form error compensation, incorporates cross-layer error correction, and adaptively refines quantization grids. Experiments on LLaMA models show that TURBOBOA achieves 4–6× speedup over BOA while improving perplexity and zero-shot accuracy, particularly in 2-bit and 3-bit settings. The results demonstrate that TURBOBOA effectively balances efficiency and precision, making low-bit quantization more practical for large-scale Transformers.

### Strengths
- The paper clearly identifies the limitations of existing post-training quantization methods, particularly the trade-off between accuracy and efficiency in attention-aware quantization.

- TURBOBOA introduces joint channel quantization with closed-form error compensation, cross-layer error correction, and adaptive grid refinement, effectively addressing accuracy loss in low-bit quantization.

- The method is evaluated on multiple LLaMA models with various bit-widths and settings, showing significant speedup and improved perplexity and zero-shot accuracy.

- The paper successfully demonstrates that TURBOBOA achieves a practical trade-off between computational efficiency and quantization accuracy for large-scale Transformer models.

### Weaknesses
- The method involves multiple closed-form derivations and joint updates, which may increase implementation and debugging difficulty.

- Cross-layer error compensation requires extra intermediate computations, which could be non-negligible on very large models or resource-constrained hardware.

- The choice of the number of jointly quantized channels (N) can affect both speed and accuracy, and may require tuning for different models or layers.

- This paper emphasizes that backpropagation is not required, but in practice it still involves a considerable amount of computation. In contrast, EfficientQAT[1] achieves significant performance improvements with minimal cost. Could you elaborate again on the advantages of being backpropagation-free? Generally, we still aim to obtain models with higher accuracy benefits.

[1] Chen, Mengzhao, et al. "Efficientqat: Efficient quantization-aware training for large language models." arXiv preprint arXiv:2407.11062 (2024).

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

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
This paper presents TurboBoA, a post-training quantization (PTQ) framework that enhances the BOA algorithm by introducing attention-aware optimizations that accelerate quantization while preserving accuracy. The key idea is to eliminate sequential dependencies in channel-wise quantization through a multi-channel parallelization strategy, extend error compensation across transformer blocks, and introduce adaptive grid selection for attention-wise refinement. The method is designed to operate without backpropagation which allows faster quantization on large-scale language models such as Llama3 and Llama2 variants. Experiments on WikiText2 and C4 datasets under different precision settings (INT2, INT3, W4A4KV4, and W4A4KV16) demonstrate that TurboBoA consistently achieves lower perplexity and higher zero-shot accuracy compared to GPTQ, SpinQuant, DuQuant, and BOA.

### Strengths
1.This paper addresses a practically relevant problem in LLM quantization, focusing on bridging the accuracy-efficiency trade-off.  

2.The algorithmic improvements are well-motivated, clearly described, and supported by mathematical derivations and pseudocode for reproducibility. 

3.The evaluation is extensive. It covers multiple model scales and both weight-only and weight-activation quantization. 

4.TURBOBOA achieves notable runtime improvements while maintaining accuracy. 

5.The combination with outlier suppression methods (QuaRot, SpinQuant, OSTQuant) highlights good modularity and compatibility with transformation-based approaches.

### Weaknesses
1.The paper lacks a rigorous theoretical justification for why the proposed joint quantization maintains accuracy despite reduced error-compensation flexibility. The reasoning remains heuristic without curvature analysis or theoretical bounds on error propagation. 

2.The novelty is limited. The method largely builds upon established concepts in Hessian-guided PTQ, extending BOA with more efficient update rules and adaptive scaling. The work feels more like a refined engineering optimization than a fundamentally new quantization paradigm. 

3.The experimental section is comprehensive but remains narrow in scope. The work focused almost entirely on the LLaMA family. There is no evidence the approach generalizes to architectures with different attention patterns. 

4.The discussion on computational overhead is insufficient. The algorithm introduces multiple Cholesky decompositions and iterative grid refinements, but the additional cost versus BOA or GPTQ is not systematically quantified. 

5.Although results show improvement in perplexity and zero-shot accuracy, the absolute performance gains are modest in some configurations. That could be due to diminishing returns relative to complexity. 

6.The broader claims of “state-of-the-art quantization” (p9, l470-471) could be better substantiated through comparisons with more recent PTQ or hybrid fine-tuning approaches.

### Questions
Please refer to the weaknesses.

### Soundness
3

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
3

### Summary
This paper proposes TurboBoA, a new backpropagation-free, post-training quantization (PTQ) algorithm for large language models (LLMs). The work aims to resolve the critical trade-off between fast but less-accurate PTQ methods like GPTQ, and more accurate but computationally slow methods like BoA. TurboBoA introduces three main innovations: (i) a joint quantization scheme that processes $N$ out-channels simultaneously, enabled by a closed-form error compensation rule (Proposition 3.1), to accelerate the sequential process of BoA; (ii) a correction mechanism to compensate for quantization errors propagated from preceding Transformer blocks (Proposition 3.2), mitigating error accumulation across the model's depth; and (iii) an adaptive grid selection strategy combined with an attention-wise scale refinement step to better align quantization grids with the iteratively updated weights (Proposition 3.3). Experimental results demonstrate that TurboBoA achieves a 4-6x speedup over the original BoA (Table 2) while also consistently improving accuracy. When combined with outlier suppression techniques, the method achieves state-of-the-art results on Llama models in both weight-only (Table 4) and weight-activation quantization (Table 5) settings.

### Strengths
* **Clear Motivation and Strong Problem Definition.**
    * The paper clearly articulates the limitations of existing backpropagation-free PTQ methods, situating the work in a well-understood context.
    * It correctly identifies a critical trade-off: GPTQ's layer-wise independence assumption leads to high speed but poor accuracy in low-bit regimes, whereas BoA's attention-aware dependency modeling is accurate but suffers from a severe bottleneck due to its sequential processing of out-channels.
    * TurboBoA is well-positioned as a direct solution to this trade-off, explicitly designed to retain the accuracy benefits of BoA while recovering the efficiency of methods like GPTQ.

* **Novel and Technically Sound Methodological Contributions.**
    * The primary idea of jointly quantizing $N$ out-channels simultaneously (Section 3.1) is a practical and effective method for acceleration, which is well-visualized in Figure 1.
    * This acceleration is supported by a non-trivial, closed-form error compensation rule (Proposition 3.1, Appendix C) that correctly incorporates dependencies for the jointly quantized block, which is critical for preserving accuracy.
    * The introduction of error compensation for *preceding* quantized Transformer blocks (Proposition 3.2, Section 3.2) is a key novelty. It directly addresses the problem of error accumulation across model depth by accounting for the input deviation $\Delta X = X - \tilde{X}$.
    * The two-part adaptive grid selection mechanism (Section 3.3) is also a strong contribution. It first aligns grids to updated weights *before* quantization (Line 9, Algorithm 1) and then refines scales *after* quantization using coordinate descent to minimize the true attention-wise loss (Proposition 3.3, Line 13, Algorithm 1).

* **Comprehensive and Rigorous Experimental Evaluation.**
    * The paper features an excellent ablation study (Section 4.2) that validates each of the three contributions (termed F1, F2, F3).
    * Table 2 provides clear evidence for the 4-6x speedup from joint quantization (F1) and confirms that the accuracy degradation from this step alone is negligible.
    * Table 3 successfully isolates the accuracy gains from F2 (pre-block error) and F3 (adaptive grid), demonstrating their complementary and significant benefits over the baseline.
    * State-of-the-art comparisons are extensive, covering both weight-only (Table 4) and weight-activation (Table 5) settings across a wide range of Llama models and bit-widths (INT2, INT3, W2A4, W4A4).
    * The inclusion of results *without* any transformation-based methods (Table 6, Appendix F.1) is a valuable addition, as it isolates the performance of the core quantizer itself, showing TurboBoA still significantly outperforms BoA and GPTQ.
    * The evaluation metrics are comprehensive, including PPL on two datasets (Wiki2, C4) and average accuracy on eight zero-shot commonsense reasoning tasks.

---

### Weaknesses
* **Analysis of Hyperparameter $N$.**
    * The paper introduces $N$, the number of jointly quantized out-channels, as a new and important hyperparameter governing the speed/accuracy trade-off.
    * While Table 2 ablates $N$ for $N \in \{1, 4, 8, 16\}$, all subsequent experiments (Tables 3, 4, 5, 6, 7) appear to use a fixed $N=16$.
    * The manuscript does not provide a discussion on how $N=16$ was chosen as the default, or how sensitive the final state-of-the-art results are to this specific value.
    * The trade-off for $N > 16$ (e.g., $N=32, 64$) is not explored. This leaves the full behavior of this parameter uncharacterized, and it is unclear if $N=16$ is optimal across all model sizes and bit-widths.

* **Clarity of Computational Overhead.**
    * The paper states that the overhead of F2 (error compensation for pre-quantized blocks) is "not negligible" because it requires computing the FP representation $\tilde{X}$ to find the input deviation $\Delta X$.
    * While the data in Table 3 supports this (e.g., for Llama3-8B, adding F2 increases runtime from 24.59 min to 37.83 min, an increase of ~13.2 min), the text does not explicitly analyze *why* this cost is so high.
    * It is unclear if this computation is a one-time cost per model or must be performed per-block, and how this cost scales with model size or sequence length. A more direct breakdown of the runtime cost of each component (F1, F2, F3) would be beneficial.

* **Mathematical Presentation and Notation.**
    * The notation in the core propositions is dense. For instance, in Proposition 3.1 (Eq 5) and 3.2 (Eq 8), block matrix notations like $([U_{out}^{T}]_{B,B})^{-1}$ are used. While a general notation guide is provided (Section 1), these specific expressions are complex and could benefit from an explicit definition in the context of the proposition for easier readability.
    * The scale-setting step is given in Algorithm 1, Line 9. This is part of the "adaptive grid selection" (F3), but the manuscript provides no derivation or explanation for how this $\min_{s}$ operation is performed, in contrast to the detailed derivation for the "attention-wise refinement" part of F3 (Proposition 3.3, Appendix E).
    * The proof for Proposition 3.2 (Appendix D) involves a significant leap from (Eq 21) to (Eq 22). This step relies on replacing a term with $H_{in}^{-1} - H_{in,-j}^{-1}$ and "exploiting the properties of Cholesky decomposition". This derivation is non-trivial and would be much easier to verify if the intermediate steps or the specific matrix identities (e.g., Sherman-Morrison) were provided.

### Questions
* **Expanding the Analysis of $N$.**
    * Could the authors please provide a justification for the choice of $N=16$ in the main experiments?
    * It would significantly strengthen the paper to include a brief sensitivity analysis of $N$ on a representative model (e.g., Llama3-8B) for the final TurboBoA method (F1+F2+F3), not just for F1 as in Table 2.
    * A discussion on the performance/speed trade-off for larger values, such as $N=32$ or $N=64$, would be valuable to understand the practical upper bound of the joint quantization speedup.

* **Clarifying the Runtime of Feature F2.**
    * To improve clarity, please provide a more direct breakdown of the runtime cost of each component. A small table or paragraph detailing [Time(BoA), Time(BoA+F1), Time(BoA+F1+F2), Time(BoA+F1+F3), Time(TurboBoA)] for one model would make the individual overheads of F2 and F3 explicit.
    * Please also clarify *why* F2 is computationally expensive. Does computing $\tilde{X}$ require a full forward pass of the FP model on the calibration data for each Transformer block being quantized?

* **Improving Mathematical Readability.**
    * For Proposition 3.1 and 3.2, please consider adding a brief note to explicitly define the block matrix notations (e.g., $[U]_{:,B}$ as the submatrix of $U$ taking columns indexed by $B$) directly within the proposition statement to aid the reader.
    * Please elaborate on the grid initialization step in Algorithm 1, Line 9. How is this $\min_{s}$ operation performed? Is it a standard min-max clipping based on $W^{(i)}_{update}$, and if so, how is the trace objective minimized?
    * In Appendix D, please expand the derivation from (Eq 21) to (Eq 22). Showing the specific matrix identity being used (e.g., related to the inverse of a matrix after a rank-1 update, or how $H_{in,-j}^{-1}$ is derived) would make the proof much more self-contained and easier to follow.

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
4

### Summary
This paper proposes TurboBOA, a faster variant of BOA for LLM PTQ.

The main contributions are:

1. Quantizing N rows jointly in closed-form instead of row-by-row (BOA)

2. a ΔX-aware error accumulation formulation that incorporates propagated quantization residuals into the next step

3. “adaptive grid selection” = re-selecting quantization scale (hence grid points) conditioned on the updated (error-polluted) input, instead of using the original FP activations. a small closed-form refinement CD update for attention blocks

4. Experiments are on LLaMA variants, reporting 4–6× speedup vs BOA with similar or better PPL.

### Strengths
the motivation (reduce BOA’s sequential dependency bottleneck, accumulation error, fixed grid) is legitimate and practically important


The methods has good empirical improvement over BOA.

writing quality is mostly clean

### Weaknesses
Motivation: the method does not completely consider cross-layer dependency, as the objective follows BOA in Table 1. As I understand, instead of cross layer dependency, it seems like sequential optimisation taking into account the interaction of attention mask, especially for the objective of W_q and W_k

Technical novelty:
1. the ΔX-aware accumulation term and how the author solve this problem is conceptually very close to GPTAQ-style accumulated loss [1]. Without explicit comparison, this make the contribution less strong

2. the multi-row closed-form step is the mathematically natural extension of the BOA 1-row closed-form, while the adaptive grid selection is very similar to re-estimating scale in GPTQ+finetune regimes so while it is naturally, the technical part is not  too strong for me; is the only purpose of multi-row closed-form to reduce the overhead?

Empirical evaluation: how is the results and efficiency compared to method like GPTAQ [1]?


[1] Li et al. (2025). **GPTAQ: Efficient Finetuning-Free Quantization for Asymmetric Calibration**. arXiv:2504.02692.

### Questions
please see weakness

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes TurboBoA, an improved PTQ method that accelerates the BoA algorithm while enhancing accuracy. The key innovation is jointly quantizing multiple out-channels simultaneously using a closed-form error compensation rule, achieving 4-6× speedup. The method demonstrates state-of-the-art results on Llama models in both weight-only and weight-activation quantization.

### Strengths
The paper provides three propositions with closed-form solutions for joint quantization, cross-block error compensation, and adaptive grid selection. The mathematical derivations are rigorous and elegant, enabling efficient backpropagation-free optimization. (Note that I have not checked the math very carefully.)

TurboBoA achieves 4-6× speedup over BoA while improving accuracy. For INT2 quantization on Llama3.2-1B, it reduces Wiki2 PPL from 40.86 to 33.33 while cutting processing time from 13.33 to 5.33 minutes.

### Weaknesses
There's no theoretical bound on accuracy degradation as a function of $N$ (number of jointly quantized channels) and model properties.

The improvement compared with BoA is not practical. Though it reduces Wiki2 PPL from 40.86 to 33.33 while cutting processing time from 13.33 to 5.33 minutes, but in my opinion, for a LLM PTQ method, the calibration time reduced from 13 to 5 mins, is not a practical improvement. 

Important design choices lack ablation studies, including the number of coordinate descent iterations, the optimal choice of N across different models/bit-widths, and evaluation is limited to only the Llama family of models.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

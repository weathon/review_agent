# Elucidating the Design Space of FP4 training

- Decision: Reject
- Scores: 2, 2, 2, 2

## Abstract
The increasing computational demands of foundation models have spurred research into low-precision training, with 4-bit floating-point (\texttt{FP4}) formats emerging as a frontier for maximizing hardware throughput. While numerous techniques have been proposed to stabilize \texttt{FP4} training, they often present isolated solutions with varying, and not always clear, computational overheads. This paper aims to provide a unified view of the design space of \texttt{FP4} training. We introduce a comprehensive, quantisation gradient-based framework for microscaling quantization that allows for a theoretical analysis of the computational costs associated with different stabilization methods on both the forward and backward passes. Using a simulator built on this framework, we conduct an extensive empirical study across a wide range of machine learning tasks, including regression, image classification, diffusion models, and language models. By systematically evaluating thousands of combinations of techniques—such as novel gradient approximations, rounding strategies, and scaling methods, we identify which configurations offer the most favourable performance-to-overhead trade-off. We find that the techniques enabling the best trade-off involve carefully combining Hadamard transformations, tensor scaling and stochastic rounding. We further find that using \texttt{UE5M3} as a scaling factor potentially offers a good compromise between range and precision with manageable computational overhead.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper conducts an empirical study of existing proposed FP4 training algorithms to see which techniques are worth their (possibly 0) overhead over a baseline naive FP4 training setup. The paper concludes that (random) Hadamard transforms, stochastic rounding, and tensor scaling prior to MX scaling are generally worth doing.

### Strengths
The paper trains a lot of models and conducts a somewhat comprehensive literature review.

### Weaknesses
My main issue with this work is that it doesn't really add anything to the growing literature of FP4 training papers. Furthermore, it also gets some details about prior works wrong. For example:
- The main point of Tseng et al. (2025)'s FP4 paper is that stochastic rounding is needed for unbiased gradient estimates, and that Hadamard transforms are a beneficial but secondary step over stochastic rounding. This paper does not mention this at all during its discussion of stochastic rounding, and only attributes the Hadamard transform to the Tseng paper.
- This paper discusses SR in section 3.1 but doesn't actually analyze what the point of SR is or how the Chmiel et al. and Fitzgibbon & Felix perform SR differently from each other and Tseng (or if they're all doing the same thing, which seems unlikely to me since the Fitzgibbon paper is on various algorithms to perform SR). There is also no discussion on the variance of different SR implementations in this paper.
- NVIDIA's NVFP4 training blog came out before the ICLR deadline (https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) and says that they used the RHT + SR, but I didn't see any discussion of it in this paper. I understand this could be considered concurrent work, but since FP4 training is an extremely fast moving field, I would still expect the authors to say something about it.
- The definitions and propositions in this paper are mainly used to define what each of the analyzed methods is doing, which is fine, but the equations take up a lot of space and don't add any rigor to the paper. There is little formal analysis (for ex. Chmiel et al.'s paper contains some analysis on noise levels and convergence) in this work that lets the reader understand why these methods are useful for FP4 training.

### Questions
See above. In its current state, this paper seems incomplete.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies FP4 training by deriving gradient flow under quantization (to block values and scaling factors) and exploring combinations (design space) of scaling, rounding, and transformation techniques such as RHT. It analyzes overhead to some extent and evaluates stability and performance through ablations across multiple tasks.

Overall, the work resembles a quantization-aware training regime rather than a practical FP4 training recipe, as the gradient flow considerations introduce inefficiencies. The contributions are incremental relative to existing literature:

Principle 1: STE is stable and preferred. This is unsurprising, as STE is already standard practice (even with SR, RHT etc.) and also brings clear efficiency benefits.

Principle 2: A precise scaling factor is preferred, with UE5M3 shown to perform well on selected tasks (no sound analysis on does this suffice?). In practice, this aligns with what one would expect—even FP32 scaling factors are desirable when efficiency is not a concern.

Principle 3: RHT, stochastic rounding, tensor scaling, and optimizer modifications yield consistent positive returns. These outcomes largely confirm prior findings, particularly those of Tseng et al. (2025) and Chmiel et al. (2025).

### Strengths
The theoretical derivation (Proposition 1 and Corollary 1) provides a formal treatment of gradient flow with quantization. 

The paper introduces an efficient approximation & adjustment method for gradients of the quantization function and investigates scaling factors with error analysis. 

The systematic sweeps over scaling and stabilization methods are potentially insightful, if presented better, especially regarding tensor scaling. 

The work also integrates with StableSPAM and loss scaling, grounding the ideas in practical training stability.

### Weaknesses
**Reference accuracy issues: Several claims about prior work are inaccurate/misleading:**

- line 53, Wang et al. (2025): High-precision scales affecting stability is questionable; efficiency wise, it depends on hardware & kernel fusions (e.g., DeepSeek-V3 blockwise with FP32 scales).

- line 57, Tseng et al. (2025): They quantize backward pass including weights/activations/gradients, not just gradients. also to note, it's one of early works that introduce stochastic rounding to fp4 training

**Writing/presentation quality: Poor figure readability (tiny legends, labels, line width, overlapping axes figure 4d), inconsistent paragraph formatting (spacing, bold text), unclear symbols meaning (e.g., M in line 377).**

- Underspecified results: Principle 2 is an important finding but with nearly all results relegated to appendix, lack of clear setup/interpretation. Currently it’s hard to draw meaningful conclusion from line 416 to 427, UE5M3 seems better than E8M0, is it optimal? or it could also diverge as model/training length scales up?

Paper structure is scattered; key insights are buried. The presentation can surely be made better to add values to the paper. 


**Efficiency concerns:**

- I think this will be the major efficiency concern with ANY gradient computations to the quantization function, as it appear in high-precision; (likely fp32/bf16) elementwise multiplications (L123–126) will happen on CUDA cores (not tensor cores) and bottleneck training, noting the tensor cores FLOPs is tens/hundreds times of cuda cores FLOPs. It would yet be interesting to study the quantization gradients from a quantization-aware training perspective.

- The proposed gradient flow of quantization requires storing high-precision tensors, expensive in memory/communication under TP/SP.

- Table 1 methods (softmax-based scaling) add large overhead and complicate pipelines, even for absmax (with indexing saving).

- The proposed hybrid scaling (L294–297) introduces further forward–backward weight/activations mismatch and quantization error, risking instability. As double quantization error between forward and backward already hurts performances in some applications.

### Questions
1. Line 176, this implicitly assumes $g>1$, why? isn’t $\tilde{s}_p = 2g / E4M3_{max}$? Reference to Chmiel et al. (2025) seems mismatched—did they actually discuss this scaling technique? and did they actually discuss zero scale? It should also be clarified that the zero-value scale issue only happens for NVFP4 with E4M3FN scale, not for MXFP4 with 2^UINT8 scale. 

2. Line 194, why apply SR to scaling factors? Example: absmax=8 → scale=0.75; rounding up to 1 causes huge bias (values 6–8 collapse to 6), while rounding to 0.5 only gives nearest-rounding level quantization error. Why expect SR to help here?

3. How does Fig. 4 reflect performances of different configs (SR, RHT, etc.)? The correspondence is unclear.

4. from line 214 and figure 1a, how is a_i, b_i decided and what’s the value? also fig. 1b shows clipping below 1.0? if this is the case, should there also be clipping above? How is threshold determined?

5. Are FP4 results simulated with e.g. BF16 or run on actual FP4 hardware (e.g., Blackwell)?

6. NaN handling: what specific mechanism is applied, and where in the pipeline?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper explores FP4 training of deep neural networks using the MxFP4 and NvFP4 data formats. The paper experiment with various combinations of techniques and hyperparameters commonly used in FP4 training and recommend configurations that provide a reasonable balance between performance and overhead.

### Strengths
1). The paper addresses an important problem by investigating the latest FP4 datatype for efficient training of deep learning models.

2). The paper conducts extensive experiments to evaluate different parameters and techniques commonly used in FP4 training.

### Weaknesses
1). The paper feels more like an experimental report than a research paper. There is very little technical novelty. Several sections (e.g., Sections 2 and 3) mostly reintroduce known materials, such as MxFP4 and NvFP4 formats, standard gradient computation, known techniques, and commonly used heuristics without offering new insights.

2). Most of the experiments are done on relatively small models. However, larger models typically have stronger outliers and are more sensitive to low-precision training. The paper also mixes different model sizes and types without explaining why these choices were made.

3). A large part of the paper discusses various gradient approximation methods, but these methods are highly heuristic, lack theoretical justification, and introduce techniques (such as clipping) in an ad hoc manner. The results also show that these gradient methods perform no difference from the standard STE.

4). The experimental results include many crowded plots in Figures 3 and 4, but the paper does not explain why these plots are presented or what specific observations the reader should draw from them.

5). The discussion of results is not very systematic and offers little insight. Most comments simply state whether something works or not; or explain with speculation without evidence. There is no deeper analysis on model type, dataset, or task.

### Questions
1). In the definition of performance gain, what are M_"ref" and M_c? How is the complexity penalty defined?

2). Principle 2 in the results states that the E4M3 scale "did not converge despite applying tensor scaling due to its range limitation." If limited dynamic range is the issue, would E8M0 work better given its larger range?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a unified quantization-gradient framework for analyzing and stabilizing FP4 training in low-precision deep learning. Through extensive simulations, it identifies optimal configurations—combining Hadamard transforms, tensor scaling, and stochastic rounding—that achieve the best balance between performance and computational cost.

### Strengths
The paper goes through details of the math, existing corresponding methods, and examine on various tasks.

### Weaknesses
The novelty is limited and difficult to idenfity. The author claimed E5M3 outperforms than E8M0, but cannot find explicit comparison on many body or appendix. They found good consistency performance of some technique, but does not lead to significant finding, compared with SOTA, e.g., does it outperform SOTA MXFP4/NVFP4 training significantly?

Another diff betwen NV and MX is the blocksize, which is missing to ablate.

### Questions
Although the claim E5M3 outperforms than E8M0 is true, 4 bits is known to apply advanced technique like SR, RHT, QAT, author could not clearly present results under these techniques clearly, in terms of they are loselss. and very difficult to find, i.e., poor presentation. so which 4 bit recipe is the best, and second and how much of gain in terms of accuracy and speed? within standard deviation or not?

### Soundness
3

### Presentation
2

### Contribution
2

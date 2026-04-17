# Disentangling Instruction Influence in Diffusion Transformers for Parallel Multi-Instruction-Guided Image Editing

- Decision: Reject
- Scores: 6, 6, 4

## Abstract
Instruction-guided image editing enables users to specify modifications by natural language, offering more flexibility and control. Among existing frameworks, Diffusion Transformers (DiTs) outperform U-Net-based diffusion models in scalability and performance. However, while real-world scenarios often require concurrent execution of multiple instructions, step-by-step editing suffers from accumulated errors and degraded quality, and integrating various instructions with a single prompt usually results in incomplete edits. We propose an Instruction Influence Disentanglement (IID) framework that enables the parallel execution of multiple instructions within a single denoising process for DiT-based models. By analyzing self-attention mechanisms in DiTs, we identify distinct attention patterns in multi-instruction settings and derive instruction-specific masks to disentangle the influence of each instruction. These masks then guide the editing process to ensure localized modifications while preserving consistency in non-edited regions. Extensive experiments demonstrate that IID can enhance fidelity and instruction completion, while also reducing computation overhead compared to existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces VAREdit, an instruction-guided image editing framework based on a visual autoregressive (VAR) paradigm. Unlike diffusion-based methods, VAREdit casts image editing as a next-scale prediction problem, enabling fine-grained, conditional edits by sequentially generating multi-scale visual tokens conditioned on source image features and text instructions. The authors identify a critical scale-mismatch problem in conditioning and propose the Scale-Aligned Reference (SAR) module, which injects scale-matched features in the first self-attention layer to bridge this gap. Experiments on EMU-Edit and PIE-Bench benchmarks show that VAREdit outperforms leading diffusion-based and autoregressive baselines, achieving better editing adherence and higher efficiency.

### Strengths
- The paper provides a strong motivation for exploring autoregressive modeling in instruction-guided image editing, departing from the dominant diffusion-based paradigm.
- The approach is rigorously detailed, including mathematical formulations and innovative SAR module design for resolving scale-mismatch within Transformer conditioning.
- VAREdit achieves significantly better results than state-of-the-art methods on multiple benchmarks. Table 1 demonstrates consistent improvement in both CLIP-based and GPT-based metrics.
- The impact of different conditioning strategies is deeply analyzed both quantitatively and qualitatively, directly reinforcing the advocated technical claims. Self-attention heatmaps offer insightful evidence supporting the architectural decisions and contribute to the interpretability of the SAR module’s necessity.
- VAREdit demonstrates a substantial speedup relative to comparable diffusion-based models even at high resolutions and shows strong performance consistency across diverse editing scenarios.

### Weaknesses
- The paper does not discuss several directly related and highly relevant studies. Although the paper situates VAREdit with respect to immediate baselines, the discussion of recent work at the intersection of visual instruction tuning and large-scale multimodal transformers is lacking. Important directly related studies, such as LLaVA, HQ-Edit and FireEdit, are missing and should be cited/discussed in Sections 2 and 4 to better contextualize the model’s advancements over recent instruction-based editing and evaluation paradigms.
- There is little qualitative or quantitative analysis of VAREdit’s failure patterns, such as out-of-distribution instructions, ambiguous edits, or adversarial prompts. For example, in Figure 6 and 10, examples are only shown for successes, not for challenging or negatively scored cases. This omission limits understanding of the model's generalizability and robustness, especially given its compositional assumptions.
- Although the math is generally sound, certain derivations (e.g., the downsampling/aggregation in SAR) skip implementation specifics. The construction and normalization of the joint key/value space in the attention module, specifically $\hat{\mathbf{O}}_k^{(\text{tgt})}$, would benefit from clarification: How precisely are scale-aligned downsampled features interpolated for non-integer spatial alignment? Are there normalization issues or class-imbalance handling in the bitwise classifier loss? It is not sufficiently detailed how the cross-scale references are incorporated at the token-level, especially in large-scale settings.
- While the paper covers two strong benchmarks (EMU-Edit, PIE-Bench), it does not explore open-domain or “wild” images, nor does it evaluate on real-world, high-resolution, or in-the-wild compositionally complex scenes. It is unclear if the efficiency and fidelity gains carry over outside of benchmark settings.
- The reliance on GPT-4o automated judging may introduce bias toward generating edits that score well under specific metrics, potentially at the expense of genuine semantic alignment. There is limited discussion of how well these metrics correlate with human evaluations, or of the possible metric gaming by autoregressive systems.
- It is unclear if the SAR advantage is preserved at $512 \times 512$ and beyond. Additional ablations at higher resolutions would add confidence.
- The diffusion models selected for comparison are strong, yet results would be further bolstered by comparison with the latest models from adjacent domains (e.g., SemanticDraw, TKG-DM, or ByTheWay for video/frame consistency or interactive content creation).
- Although the paper promises code release, reproducibility would be improved with a more detailed pseudo-code or algorithm outline for the SAR module, and more comprehensive experimental pipelines in the appendix.
- Some visualizations and tables (e.g., Figure 4 and radar charts in Figure 5) could benefit from more explicit axis labels, scales, and baseline explanations for readers less familiar with the specific benchmarks.

### Questions
- Could the authors provide more examples of failure cases or edge cases where VAREdit struggles (e.g., ambiguous or out-of-domain edits)? Quantitative breakdowns of error types or failure rates would greatly help clarify robustness.
- How does the SAR module scale with higher spatial resolutions (e.g., $1024 \times 1024$) or longer token sequences? Are there inefficiencies or quality drops, and could the authors share any preliminary results or analysis?
- Is there a risk of overfitting to the GPT-based metrics or prompt templates? Has any human evaluation (crowdsourced or expert) been performed to validate the GPT judge? If so, how do human and automated scores correlate?
- In the mathematical exposition of SAR, could the authors clarify how spatial downsampling is performed in practice, and if any normalization or re-weighting is involved during the key/value aggregation process? Are there caveats with very large codebooks or unusual grid sizes?
- Could the authors discuss the potential for extending VAREdit to other modalities, such as video (for temporal consistency) or 3D editing? What would be the main technical obstacles or adaptations required?

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
The paper proposes Instruction Influence Disentanglement (IID), a training-free framework for parallel multi-instruction image editing within a single denoising pass. It targets Diffusion Transformer (DiT) models like FluxEdit, Omnigen, and FluxKontext, aiming to solve two main problems: cumulative errors from step-by-step editing and conflicts from naive instruction concatenation. IID analyzes self-attention patterns in DiTs and introduces a head-wise mask generation strategy. For each instruction, it subtracts the mean attention map of other instructions to isolate the editing region, then aggregates and binarizes these masks. Next, an adaptive blender composes instruction tokens and latent images at a predefined timestep S. It uses a re-ranking process with an influence score to mitigate any dominance effects. Finally, an attention mask constrains instruction tokens to their respective regions, enabling disentangled parallel edits. Experiments on MagicBrush and a custom dataset show consistent improvements over baselines across metrics like L1/L2, CLIP-I/T, and DINO, as well as in human preferences.

### Strengths
1. Proposes a novel, training-free parallel editing framework. It utilizes "instruction-wise attention subtraction" to leverage the model's own attention for disentangling multiple instructions, eliminating the need for external segmenters.
2. The evaluation is comprehensive, demonstrating consistent performance improvements across multiple mainstream DiT backbones (e.g., FluxEdit, Omnigen) and datasets.
3. Addresses a critical, real-world need in multi-instruction DiT editing, solving a key problem where existing methods fail or are ineffective.

### Weaknesses
1. While the paper provides a theoretical discussion of error accumulation and instruction conflicts in the appendix, the main text adopts a simple averaging scheme for overlapping regions during blending, without analyzing its potential to reintroduce conflicts or degrade boundary consistency.
2. The evaluation is primarily conducted on relatively simple editing tasks. The custom dataset construction using GPT-4o may introduce biases, and the human evaluation involves only 5 participants, which may not be sufficient for robust conclusions. The paper lacks comparison with more recent multi-instruction editing methods or adaptation of existing approaches to DiT architectures.
3. The paper does not provide sufficient analysis of scenarios where the method fails or performs poorly.

### Questions
1. The IS score used to guide instruction position re-ranking is an innovative idea. How stable is IS under paraphrasing, i.e., different phrasings of the same instruction?
2. In applying masks within the DiT denoising process, how does IID compare against using minimally adapted external segmenters (e.g., SAM) to derive instruction-specific masks for DiTs? Does an external mask improve completion or fidelity, or introduce new artifacts and complexity?

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
The paper addresses multi-instruction-guided image editing in DiTs, where users specify several natural-language instructions simultaneously. Existing strategies tend to degrade image quality or yield incomplete edits. The authors overcome these limitations via introducing IID, a training-free framework enabling multiple edits within a single denoising process. It constructs head-wise attention masks to blend instruction-specific latents, and builds a compositional attention mask that ensures local edits while preserving non-edited regions. Experiments demonstrate that IID improves image fidelity and efficiency across Omnigen, FluxEdit, and FluxKontext models.

### Strengths
1.The introduction precisely identifies two failure cases in multi-instruction editing—error accumulation in sequential steps and instruction conflicts in prompt concatenation. This evidence grounds the paper’s motivation in observable limitations of prior methods.

2.The proposed mask derivation introduces a simple yet effective subtractive strategy across attention heads to isolate editing regions.

3.The adaptive blender and re-ranking strategy mitigate instruction dominance, ensuring balanced edits. 

4.Good performance achieved under both MagicBrush and a custom dataset extending to six-instruction cases.

### Weaknesses
1. The adaptation to the proposed method is limited to some extent. According to the experiments from the paper, the method only works in DiT-based architectures.

2.One of my concerns is the actual application. Compared with the parallel instructions paradigm, sequential instruction editing is usually used in real-world application as the user can adjust the requirement at the next time. However, the parallel instruction is equivalent to the T2I editing that the text is pointwise and concise, and has differences compared with sequential image editing. 

3. Based on the stated concern, it is better to show the generation performance with the traditional generated model with the same type of instructions to demonstrate the effectiveness.

4. Human evaluation involves only five participants, lacking statistical analysis or inter-rater reliability, which undermines the reproducibility of subjective results.

5.The subsequent thresholding with Gaussian + Otsu filters introduces multiple heuristic stages. How about the performance relative to other mask computation settings?

### Questions
Please see the weakness above.

### Soundness
3

### Presentation
3

### Contribution
2

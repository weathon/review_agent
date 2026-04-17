# Visual Autoregressive Modeling for Instruction-Guided Image Editing

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Recent advances in diffusion models have brought remarkable visual fidelity to instruction-guided image editing. However, their global denoising process inherently entangles the edited region with the entire image context, leading to unintended spurious modifications and compromised adherence to editing instructions. In contrast, autoregressive models offer a distinct paradigm by formulating image synthesis as a sequential process over discrete visual tokens. Their causal and compositional mechanism naturally circumvents the adherence challenges of diffusion-based methods. In this paper, we present VAREdit, a visual autoregressive (VAR) framework that reframes image editing as a next-scale prediction problem. Conditioned on source image features and text instructions, VAREdit generates multi-scale target features to achieve precise edits. A core challenge in this paradigm is how to effectively condition the source image tokens. We observe that finest-scale source features cannot effectively guide the prediction of coarser target features. To bridge this gap, we introduce a Scale-Aligned Reference (SAR) module, which injects scale-matched conditioning information into the first self-attention layer. VAREdit demonstrates significant advancements in both editing adherence and efficiency. On EMU-Edit and PIE-Bench benchmarks, VAREdit outperforms leading diffusion-based methods by a substantial margin in terms of both CLIP and GPT scores. Moreover, VAREdit completes a 512$\times$512 editing in 1.2 seconds, making it 2.2$\times$ faster than the similarly sized UltraEdit. 
Code is available at: https://github.com/HiDream-ai/VAREdit.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VAREdit, the first tuning-based visual autoregressive model for instruction-guided image editing. It proposes the novel SAR module to mitigate the scale mismatch problem in the VAR tuning process. VAREdit achieves SOTA across different image editing benchmarks. The paper is well written and organized, despite the lack of reasoning-based and multi-step image editing analysis.

### Strengths
1. Novel Paradigm: The paper introduces VAREdit, the first tuning-based visual autoregressive (VAR) framework for instruction-guided image editing, which reframes editing as a next-scale prediction task. The proposed Scale-Aligned Reference (SAR) module effectively mitigates the scale mismatch problem by injecting scale-matched conditioning only into the first self-attention layer. The motivation and design are well-analyzed through self-attention heatmap studies.

2. Efficiency and Scalability: VAREdit achieves 2.2× faster inference than comparable diffusion models while maintaining superior quality. The scalability from 2.2B to 8.4B models shows consistent improvement across all editing types.

3. Well Writing and Structure: The paper is clearly structured, with detailed methodological explanations, visualizations, and ablations. It maintains excellent reproducibility and clarity in presentation

### Weaknesses
Because of its AR-based nature, I wonder if such a VAR-based image editing method can show strong ability in reasoning-based image editing and multi-step image editing tasks. If the author can include some relevant experiments, such as WISE or GEdit, I would consider raise the score.

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes VAREdit, a visual autoregressive (VAR) framework that reframes instruction-guided image editing as a next-scale prediction problem. Given an instruction and a source image, a multi-scale tokenizer plus a VAR Transformer predicts target residuals from coarse to fine. A key observation is that using only finest-scale source features to condition coarse target predictions causes a scale mismatch. The authors introduce a Scale-Aligned Reference (SAR) module that injects scale-matched conditioning only in the first self-attention layer, while subsequent layers keep using finest-scale conditioning. On EMU-Edit and PIE-Bench, VAREdit reports higher CLIP and GPT-based metrics than diffusion and AR baselines, and achieves ~1.2 s latency for 512*512 edits with an 8.4B model.

### Strengths
- The paper introduces a novel method called Scale-Aligned Reference (SAR), which effectively balances computational efficiency and editing performance. 
- The proposed method achieves strong performance on EMU-Edit and PIE-Bench, with moderate latency, outperforming several diffusion-based and autoregressive (AR) baselines.
- The paper includes comprehensive experimental analysis; it conducts attention-level investigations to motivate the design of SAR, provides both qualitative and quantitative results, and performs ablation studies to demonstrate the effectiveness of SAR.

### Weaknesses
- the paper provides self-attention heatmaps based on the full-scale setting to motivate the design of SAR, but a similar analysis of the tuned model is missing; including this will cross-validate the modeling choice and strengthen the motivation.
- the paper relies on GPT score to evaluate editing models; however, GPT judges may introduce hallucinations and prompt-induced biases. A better way would be to use a controlled human study to cross-validate the reliability of using GPT as the evaluator.
- while the method demonstrates strong performance, it does not compare against frontier editing models such as GPT-4o-Image, FLUX.1 Kontext, or Qwen-Image.
- the authors claim their model as "the first tuning-based visual autoregressive model for instruction-guided image editing.", which is false, see [NEP: Autoregressive Image Editing via Next Editing Token Prediction](https://arxiv.org/pdf/2508.06044). VAR broadly means autoregressive modeling of visuals. The authors might be using it to indicate "image generation via next scale prediction," but it is very misleading. It would be better to spell it out.

### Questions
Following the Weakness section, my questions are as follows:

Q1: Could you provide the self-attention heatmaps under the SAR setting? I would like to visually confirm how SAR changes the attention distribution compared to the full-scale setting.

Q2: Can you provide references to support the claim in lines 131-132: *Compared to diffusion
models, AR approaches generally offer superior prompt adherence and faster inference.*

Q3: Please can you include performance comparisons with existing frontier editing models (e.g., GPT-4o-Image, Qwen-Image, etc.) in your experiments?

### Soundness
3

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
4

### Summary
This paper introduces VAREdit, an instruction-guided image editing framework that leverages a visual autoregressive modeling paradigm. By formulating editing as a next-scale prediction problem, VAREdit conditions on both source images and textual instructions to autoregressively generate multi-scale tokenized image features. The paper identifies a scale-mismatch issue in using finest-scale-only conditioning and proposes the Scale-Aligned Reference module, which bridges this gap by injecting scale-matched information only in the Transformer’s first self-attention layer. Experiments on EMU-Edit and PIE-Bench benchmarks demonstrate that VAREdit outperforms several competitive diffusion and AR-based baselines in both instruction adherence (as measured by GPT and CLIP scores) and inference speed. Additional ablations and qualitative studies illuminate the advantages of the SAR module.

### Strengths
1. The paper analyzes and highlights the limitations of both full-scale and finest-scale-only image conditioning for autoregressive editing tasks, a nuanced challenge not prominently addressed in prior work.
2. The proposed Scale-Aligned Reference module is conceptually well-motivated, solving the scale-mismatch issue in a resource-efficient way by using scale-aligned source features solely in the first attention layer. The supporting evidence includes explicit self-attention heatmaps and precise mathematical descriptions for how SAR modifies qkv calculation.
3. VAREdit sets a new state-of-the-art on the EMU-Edit and PIE-Bench datasets across a variety of metrics. Table 1 illustrates consistent improvements in both instruction adherence (GPT-Success, GPT-Balance) and region preservation (GPT-Over) compared to a broad selection of recent diffusion and AR-based methods.

### Weaknesses
1. The Related Work underappreciates or omits explicit discussion of closely related contemporaneous models for instruction-based or in-context image editing. [1,2,3] are especially relevant and directly relate to the manuscript’s focus and should be discussed in both the Related Work section and empirical comparisons if possible.

2. The approach is mostly evaluated in the standard fine-tuned regime and does not examine how VAREdit generalizes to unseen editing types, tasks, or user instructions without retraining—settings increasingly emphasized in recent autoregressive editing work[1]. The absence of in-context or few-shot adaptation benchmarks leaves a gap in understanding VAREdit’s flexibility beyond its tuned task distribution.

3. While qualitative figures suggest strong performance, a more honest (or even quantitative) exploration of limitations or failure modes (especially for global/text edits where 2.2B model underperforms) would strengthen the impact. For example, are there common instruction types, object classes, or image domains where even the SAR variant consistently underdelivers, and why? Are these failures more about model scale, data coverage, or something intrinsic to next-scale AR modeling?

4. All quantitative evaluation relies on CLIP and GPT-based metrics, which, while valuable, do not wholly replace exhaustive user studies, especially regarding over-editing artifacts or semantic misalignments. Some visual results imply subjective improvement, but explicit human preferences or error typologies are missing.

5. The experiments are mostly confined to two academic datasets. There is little examination of generalization to diverse, out-of-distribution, or real-world photographic content, which is critical for practical adoption. Even a small cross-dataset or wild-image ablation would be illuminating.

6. While the SAR module is intuitively described, the formulation around how scale-aligned features are constructed and integrated into the attention mechanism could be mathematically tighter. For example, clarification and explicit definition of how $\operatorname{Down}(\cdot)$ and its learned parameters are shared or adapted across scales would improve reproducibility. The transition from Equation 2 (cumulative features) to the SAR-enhanced self-attention mechanism could also use more technical rigor regarding masking, normalization, and gradient flow. Moreover, practical details such as how SAR interacts with positional encoding or codebook lookups for each scale should be more explicitly stated, especially since Figure 2 elides some of these corners.

7. While Figure 2 effectively visualizes the end-to-end pipeline, it omits some low-level components, leading to slight ambiguity (e.g., are the multi-scale VQ-Encoders shared across source and target? Are all tokens attended equally pre- and post-SAR? How are target and source regions aligned spatially or by edit mask, if at all?). Likewise, the heatmaps are highly informative, but a more interpretable color scale or clear highlighting of the causal directionality would help non-specialist readers.

8. The training dataset is aggregated and post-processed using a vision-language model filter, but only high-level processing details and a prompt are provided. Concrete numbers or public release status for the filtered data subsets would enhance reproducibility. Similarly, claims that code/models will be released “after acceptance” limit the immediate reproducibility of results.

9. Although the cited baselines are comprehensive for diffusion and AR models, the omission of multimodal LLM-based frameworks as competitors [2] weakens the claim of state-of-the-art. Even a partial or qualitative comparison would be helpful, as these systems are becoming increasingly prominent.

10. While the SAR mechanism is well-motivated, the extent to which it represents a substantial departure from architectural adjustments (as opposed to an incremental fix) is debatable. The key technical novelty (selective multi-scale reference injection in the first layer) may be seen as a relatively modest extension rather than a fundamentally new paradigm.

[1] Unleashing In-context Learning of Autoregressive Models for Few-shot Image Manipulation.

[2] Guiding Instruction-Based Image Editing via Multimodal Large Language Models

[3] Fireedit: Fine-grained instruction-based image editing via region-aware vision language model

### Questions
1. How does VAREdit perform in genuinely few-shot/in-context adaptation settings? Can the model apply novel editing instructions (or new types of edits) without SGD-based fine-tuning, or does it require full retraining? Comparative numbers or case studies against recent AR-based in-context editing models would be instructive.
2. What are the observed failure modes for VAREdit, especially for large/semantic/global edits or high-text-content regions? Can the authors provide a quantitative or qualitative assessment of these, and whether they are due to model scale, dataset limitations, or issues intrinsic to the SAR module?
3. Details on the SAR module’s construction: Are the downsampling mappings for scale-aligned features jointly learned, fixed, or derived from the tokenizer? How does SAR interact with residual connection and positional encoding in very deep Transformers?
4. How robust is VAREdit to highly ambiguous, underspecified, or conflicting instructions? Does it have mechanisms (e.g., uncertainty estimation, output confidence) to flag instructions where adherence is questionable?
5. Is there any plan for releasing the filtered and processed dataset used in training, and can more precise statistics be included in the public release for reproducibility?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VAREdit, the first training-based Visual Autoregressive (VAR) model for instruction-guided image editing. The key contribution is reframing image editing as a next-scale prediction problem, where the model autoregressively generates multi-scale target features conditioned on source image features and textual instructions. The authors identify a critical scale-mismatch challenge when conditioning on finest-scale source features alone: high-frequency details disrupt the prediction of coarse-grained target structures. To address this, they propose the Scale-Aligned Reference (SAR) module, which dynamically generates and injects scale-matched source information specifically into the first self-attention layer, while deeper layers continue to use finest-scale conditioning. Through systematic analysis of self-attention patterns, the authors demonstrate that this design aligns with the model's functional differentiation across layers—where the first layer establishes global structure and deeper layers perform local refinement. Extensive experiments on EMU-Edit and PIE-Bench benchmarks show that VAREdit achieves substantial improvements over state-of-the-art diffusion-based methods

### Strengths
1. This paper explores how to apply VAR architecture to image editing, moving away from dominant diffusion approaches. The SOTA results on balanced metrics and 2.2x speedup are significant, suggesting AR models are a highly promising direction for image editing.
2. The core contribution is the deep analysis of the scale mismatch problem in VAR conditioning. The attention map analysis (Figure 3) is strong evidence. The resulting SAR module is an elegant and efficient solution that precisely targets the identified bottleneck (the first layer) without incurring extra inference cost.
3. The paper rightly identifies flaws in standard metrics. Relying on GPT-Balance and especially the analysis in Figure 4 (evaluating preservation only on successful edits) provides a much more rigorous and convincing evaluation than prior work.

### Weaknesses
1. The paper's primary metrics (GPT-Suc., GPT-Over., GPT-Bal.) rely on GPT-4o as a judge. While arguably better than CLIP, this is costly, slow, and dependent on a proprietary API, making the evaluation results difficult and expensive to reproduce. The authors should provide an alternative using an open-source VLM (such as Qwen3-VL) in the rebuttal.
2. The authors used CLIP and GPT scores as evaluation metrics, but lacked test results on benchmarks commonly used in the editing field, such as ImgEdit and GEdit-Bench. Considering the authors' statement that CLIP scores do not accurately reflect editing performance, and the overhead and uncertainty of GPT scores, using ImgEdit and GEdit-Bench can provide reproducible and stable comparative results.
3. The training data was filtered using an external VLM (Kimi-VL), which discarded ~1M samples. This is a significant, non-reproducible step, and the final model quality may be heavily dependent on this external filter.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

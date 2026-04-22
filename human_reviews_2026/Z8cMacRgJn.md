# Mitigating Object Hallucinations in Large Vision-Language Models via multi-scale visual integration

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Large Vision-Language Models (LVLMs) face the persistent challenge of object hallucination, where models generate descriptions of objects that do not exist in the input image. In this work, we revisit hallucination arising from visual encoding and show that, while moderate resolution scaling alleviates hallucination, a similar context-induced effect, previously observed in text generation, also emerges in the visual domain: excessive visual tokens diffuse attention and reintroduce hallucination. To address these issues, we propose a multi-scale visual alignment decoding framework, which supplements visual knowledge at multiple granularities while ensuring attention remains focused on the correct regions, thereby mitigating context-driven hallucination. Extensive experiments demonstrate that our approach substantially reduces object hallucination and achieves stronger image-text alignment than state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper analyzes object hallucination in LVLMs through a resolution-scaling ablation study. It reports a boundary effect: moderately higher visual resolution reduces hallucinations, but pushing resolution too far increases visual tokens, diffuses attention, and brings hallucinations. Building on this, the authors propose a multi-scale visual decoding framework that mixes (i) high-res object-level grounding and (ii) lower-res global attention constraints, with a fusion-based score that combines these signals with model likelihood to steer decoding. Experiments on CHAIR/POPE and a GPT-4V-assisted evaluation show improvements over several contrastive decoding baselines.

### Strengths
1. To the best of my knowledge, this is the first paper to analyze hallucination as a function of image resolution. I agree with the central claim: for models relying on discrete tokenizers, overly high resolution can harm input consistency and, in turn, confuse the VLM.

2. As a remedy, using Grounding DINO on high-resolution inputs for fine-grained consistency, while using CLIP on low-resolution inputs for global alignment, is a reasonable high-level design.

### Weaknesses
1. (Major) The paper’s motivation is under-analyzed. It only shows performance curves across resolutions without digging into root causes that can suggest logical reasons for the proposed solution. Because this logical chain from motivation to proposed methods is very thin, I remain unconvinced that these two specific components (Grounding DINO + CLIP) are the right answer. Although Table 5 provides an ablation study, it shows only modest quantitative changes and does not substantiate the necessity of the method.

2. (Major) The approach ultimately depends on external models, which makes direct comparison to purely internal (decoding-only) baselines difficult. (At least, the same computational costs, wall-clock latency or FLOPs, etc..)

3. (Moderate) The text claims the overhead is “close to greedy,” yet the timing table reports 27.7s vs 3.9s for greedy (and ~19.1s for CGD), which is a substantial slowdown for captioning-length outputs. This gap matters for interactive LVLM use. A tighter runtime analysis—or an anytime / early-exit variant—would help.

### Questions
1. Why didn’t you analyze the root causes of resolution-induced hallucination more deeply? Personally, I suspect a primary factor is that a discrete tokenizer receives different semantic units as resolution changes, but there could be multiple contributing reasons. The solution should be derived from such a causal analysis, yet the current connection feels too weak.

2. Why specifically choose Grounding DINO and CLIP? I don’t see a compelling rationale for why these particular methods are necessary.

### Soundness
2

### Presentation
2

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
This study proposes a method that reduces the hallucination problem of LVLMs by considering both low and high resolution visual input tokens.

The authors are first inspired by the fact that the sampling distribution gap between LVLM with and without image input peaks at intermediate resolution. Based on that , they discovered the phenomenon that the attention distribution varies significantly with different resolution setup, while different setup actually contributes to answering different level of questions.

To utilize this observation, this study proposed a method that constrains the attention mechanism for both high-resolution and low-resolution settings. Experiments show that the proposed method can significantly reduce the hallucination problem of LVLMs on different tasks.

### Strengths
see summary

### Weaknesses
The weakness is mainly in presentation. I think Figure 2 taking half page does not make sense and Figure 4 is not expressive enough.

### Questions
n/a

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
This paper addresses the issue of object hallucination in large vision-language models (LVLMs), where models generate descriptions of non-existent objects or details. The authors identify a boundary effect in visual resolution scaling: while moderate increases in resolution reduce hallucination, excessively high resolutions lead to attention diffusion, causing hallucinations to reemerge.

To overcome this, the authors propose a multi-scale visual decoding framework that integrates: fine-grained grounding via GroundingDINO to recover object-level details, and global attention constraints via CLIP to maintain semantic alignment. These are combined with the LVLM’s internal likelihood in a fusion module to guide decoding. The method is training-free and does not require external data.

Extensive experiments on CHAIR, POPE, and GPT-4V-assisted evaluations demonstrate state-of-the-art performance in reducing hallucinations while maintaining or improving caption quality across multiple LVLM backbones (e.g., LLaVA-1.5, InstructBLIP, mPLUG-Owl2). Ablation studies confirm the contribution of each component, and the method shows promising generalizability to safety-critical domains.

### Strengths
While many researchers advocate for simply increasing image resolution to mitigate hallucinations, this work demonstrates the existence of a "sweet spot." It shows that beyond this point, further increasing resolution can paradoxically exacerbate hallucinations due to "attention diffusion." This aligns with the intuition that an excessive number of image tokens, coupled with the VLM's inherent tendency to attend less to visual tokens as text generation progresses, can degrade performance. The paper's strength lies in directly addressing this specific, well-motivated problem.

The method's strength lies in its design, presenting an innovative "plug-and-play" solution. The core of the method is a training-free, multi-scale visual decoding framework. Its ingenuity is demonstrated through the creative and purposeful combination of existing powerful tools (GroundingDINO and CLIP), rather than the introduction of complex new modules. The fine-grained grounding module accurately captures object-level evidence at high resolution, while the global attention constraint module maintains semantic consistency at a lower resolution; together, they synergistically address the "attention diffusion" paradox caused by simply increasing resolution. Furthermore, the method employs a carefully designed fusion scoring function that organically integrates visual grounding, semantic alignment, and the model's internal likelihood, supplemented by an iterative decoding strategy. This ensures that the output remains faithful to the image content while respecting the generative capabilities of the underlying model. This design allows the framework to be directly applied to a variety of existing LVLMs demonstrating exceptional practicality and generality.

### Weaknesses
The computational efficiency of the proposed method is a significant concern, potentially more central and severe than presented in the limitations section. The core approach, which relies on two external models (GroundingDINO and CLIP) and employs iterative decoding, inherently introduces substantial computational overhead. As shown in Appendix Table 15, the decoding time of the proposed method (27.68s) is substantially higher than standard greedy decoding (3.90s) and is even slower than some existing methods (e.g., CGD at 19.13s). This severely limits its practicality for real-time or large-scale applications.

While the experiments are comprehensive on standard benchmarks like MS-COCO, these datasets are relatively "clean." The paper's limitation regarding performance in "densely packed objects" or "specialized domains," while valid, could be more concretely addressed. The generalizability and robustness of the method would be significantly strengthened by evaluating it on more challenging benchmarks, such as those involving multi-turn VQA, which require sustained visual grounding over a conversational context. 

For experiments such as on CHAIR, there is no sensitivity/statistical significance analysis showing the randomness of the results, as we know the scores could vary with a large variance. The paper could benefit from testing on more recent benchmarks and models, such as InternVL, PaliGemma, Phi vision, Meta PLM, AMBER, MME, THRONE.

Paper could benefit from better writing and cleaner presentation.

### Questions
The core analysis in Section 2, which establishes the "boundary effect" of resolution scaling, is conducted using Qwen2.5-VL.  I remember that Qwen2.5-VL's visual encoder has a maximum input resolution (e.g., 448x448 pixels, can you check the specific number ).  For images larger than this limit, the standard practice is for the visual processor to downsample them to the model's acceptable size.

Could you please clarify in detail how you handled the input images for the high-resolution settings reported in your experiments (e.g., 896 as shown in Figure 1a)?

Specifically, were the images at these high resolutions (e.g., 896x896) fed directly to the Qwen2.5-VL model, which would then internally downsample them to its maximum capacity?

Or, did you employ a different preprocessing strategy (e.g., patch-based processing) that truly allows the model to ingest and tokenize the full high-resolution image?
This clarification is crucial because: If the images were downsampled, the observed performance degradation at high resolutions might not be due to "attention diffusion" from an excessive number of visual tokens, but rather an artifact of information loss or distortion caused by aggressive resizing.  This would fundamentally challenge the proposed hypothesis. A clear explanation of the input pipeline is necessary to validate that the model indeed received and processed the intended long visual token sequences, which is the cornerstone of your analysis.

### Soundness
2

### Presentation
1

### Contribution
2

# HieraQuery: Bridging Multimodal Understanding and High-Quality Generation through Multi-Scale Query Learning

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Unified multi-modal LLMs enable the integration of visual understanding and generation in a single framework. Recent study shows that a set of learnable queries can serve as an effective interface between autoregressive multimodal LLMs and diffusion models, though the visual quality of generated images still lag behind dedicated generation models. The major bottleneck lies in the difficulty of a single set of learnable queries to generate accurate visual representations in a single round of inference. Hence, we introduce HieraQuery, which leverages a hierarchy of learnable visual queries to generate high-quality visual contents in a coarse-to-fine manner. Specifically, several sets of learnable queries are provided to the language model, where preceding ones are used to generate images of lower resolution, focusing on the global structures of the generated content, while the subsequent ones serve as the condition for generating higher resolution images, concentrating on the fine-grained details. In addition, a multi-scale representation alignment strategy is proposed to enforce cross-scale consistency and accelerate convergence. Ablation analyses demonstrate that using the hierarchical visual queries can effectively improve the visual generation capability of unified multi-modal LLMs, and scaling up the number of scales proves an effective way for further improving the generation quality.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a unified autoregressive multimodal LLM framework. It introduces hierarchical learnable visual queries for coarse-to-fine image generation: preceding queries handle low-resolution global structures, while subsequent ones refine high-resolution details. A multi-scale representation alignment strategy ensures cross-scale consistency and accelerates training.

### Strengths
1. The work shows strong technical quality through comprehensive experiments on benchmarks like GenEval, MJHQ-FID, and GEdit-Bench, outperforming SOTA unified models. Ablations validate the multi-scale query and alignment strategies.
2. HieraQuery advances unified multimodal LLMs, and its scalable multi-scale approach addresses bottlenecks in error accumulation and resolution handling, benefiting the AI community in developing more versatile vision-language models.

### Weaknesses
1. Table 1 shows noticeable drops in understanding benchmarks compared to MetaQuery (which uses a similar base): MMB decreases from 83.5 to 80.7, MMMU from 58.6 to 54.3. MM-Vet improves slightly (66.6 to 74.0), but the overall trend suggests the multi-scale query integration may interfere with the LLM's text-visual alignment for understanding tasks.
2. The core contribution—multi-scale queries for coarse-to-fine generation—builds directly on MetaQuery by hierarchizing queries. While this addresses MetaQuery's plateauing with token count, it resembles existing multi-scale generation techniques, such as  *Chain-of-Sight* mentioned in the paper. This makes the novelty appear incremental rather than transformative for unified MLLMs.
3. Further detailed ablation studies are missing. For example, no quantitative analysis is provided on how performance and computational cost trade off as $K$ in $S= \\{s_1, s_2, . . . , s_K\\}$ in increases. Without this analysis, it is unclear whether increasing $K$ yields diminishing returns, introduces redundancy, or disproportionately raises inference latency.

### Questions
1. Could the authors elaborate on how HieraQuery's hierarchical queries differ mechanistically from multi-resolution approaches in diffusion models? A detailed comparison, perhaps with pseudo-code or diagrams, could clarify if this is a truly novel integration or an incremental extension, potentially strengthening my view on the contribution's originality.
2. In Table 1, HieraQuery shows slightly lower scores on understanding benchmarks. Since the MLLM (Ming-Lite-Omni) is frozen during training, could this be causing a degradation?
3. Minor issue: There is a redundant writing issue in L267-L269.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper HieraQuery introduces a hierarchical query modeling framework to improve unified multimodal generation.
Instead of relying on a single query token set, it designs multi-scale queries—from coarse to fine—to progressively build images, where coarse queries capture global semantics and fine queries refine local details.
It further aligns features across scales through multi-scale representation alignment, ensuring semantic consistency between the diffusion backbone and the visual encoder.
Built on a frozen MLLM (Ming-Lite-Omni) with diffusion backbones like SD3-Medium and SANA-1.6B, the system achieves better image fidelity and structure, improving FID, GenEval, and DPG performance while maintaining strong multimodal understanding.

### Strengths
1.The hierarchical query framework provides a clear and modular architecture that systematically connects multimodal understanding with coarse-to-fine generation.
2.The multi-scale query learning and cross-scale alignment improve the balance between global semantic coherence and fine-grained visual fidelity.
3.The model demonstrates strong results on both text-to-image and editing tasks, showing good generality across benchmarks.
4.The paper presents extensive experiments as well as thorough ablation studies.

### Weaknesses
1. There is a typo in Figure 3: it should be qualitative and not quantitative.
2. The qualitative examples are too simple. For compositional task, please include more complex prompts.
3. The paper does not provide a quantitative analysis of additional computational cost, such as FLOPs, memory, or end-to-end inference time, so the efficiency trade-offs of the hierarchical design remain unclear.
4. The paper mainly reports benchmark scores (FID, GenEval, DPG) without perceptual or human evaluation on image coherence, consistency, or failure modes.
5. The understanding–generation balance is claimed but not verified with detailed breakdowns; it remains unclear whether improvements in generation come at the cost of understanding accuracy.

### Questions
In the ablation study, several configurations yield very similar quantitative results. How did the authors determine which setting is optimal in the first place? Was the choice based on efficiency (e.g., computational cost, FLOPs, or latency), stability, or qualitative evaluation? Providing a clearer selection criterion would help assess the robustness of the claimed improvements.

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
2

### Summary
This paper propose HieraQuery, a multi-scale query learning method for high-quality visual generation, which leverages a hierarchy of learnable visual queries for generation in a coarse-to-fine manner and includes a multi-scale representation alignment strategy for cross-scale consistency and convergence acceleration. Extensive experiments is conducted and the proposed method demonstrate the performance improvement on the visual generation capability.

### Strengths
1. Query learning is worth studying and have certain commonality for multimodal tasks, which is brave and novel.
2. The experiment result is solid under text-to-image generation, image style transfer and fine-grained editing.
3. Convincing visualization examples are provided to prove the effectiveness of the proposed method.

### Weaknesses
1. Lack of concept comparison framework between the baseline methods and proposed HieraQuery method. Providing this may help better presentation.
2. Comparison on understanding benchmarks should includes used MLLM backbone.
3. Several writing typos: “benchmarsk” at line 265 and MJHQ FID) at Table 4.

### Questions
1. See weakness.
2. The training cost and inference speed between proposed method and baselines may require to provide.
Overall, I think this work is novel, but with several weaknesses on writing and experiment. So I tend to adjust my score and confidence based on the author's response and the opinions of other reviewers.

### Soundness
3

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
5

### Summary
This paper proposed HieraQuery, which extends the method of MetaQuery to a multiscale paradigm by generating images of multiple resolutions on different sets of conditioning queries. To enforce cross-scale consistency, REPA is applied to different levels, so the DiT features from all these scales are aligned to the same semantic representation. Beyond image generation, the framework is also adapted to image editing, with a progressive reconstruction-and-then-editing paradigm. In the experiment, performance gain by using the multiscale queries is observed on GenEval.

### Strengths
1. The motivation is clear, and the paper is easy to follow.

2. The proposed framework, HieraQuery,  comprehensively supports image understaning, generation, and editing.

3. HieraQuery achieves competitive performances on all three tasks studied.

### Weaknesses
1. Technical contribution is limited, since the ideas of multiscale generation and representation alignment are already proposed in existing works, i.e, VAR and REPA.

2. Although this paper studies unified models that cover understanding, generation, and editing. The advantage of multiscale queries is only experimentally supported by results on the image generation benchmark---GenEval.

3. Some results are expected but missed in the ablation study:

    **a.** In Table 3, the result of applying REPA to the single-scale setting is expected.
    **b.** DiT is shared across different scales. Will some scale-specific parameters benefit the final performance?

4. In the 2nd and 3rd training stages, only image-to-image losses are applied, which seems to damage the text-to-image generation capability of the model.

### Questions
Some details are missing:
1. SD3-Medium and SANA-1.6B are both studied. But it is unclear which model is used in Table 1&2.
2.  Can the authors provide more details on editing? For example, are the same model parameters used for both generation and editing? Are VAE latents fed to the DiT to ensure consistency between source and edited images?

### Soundness
2

### Presentation
2

### Contribution
2

---
job_id: 4290e294-c13b-4c48-9489-b5f2125a4f85
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 1EdAn5gMVv.pdf
paper: SpatialBoost: Enhancing Visual Representation Through Language-Guided Reasoning
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is squarely about representation learning for vision encoders, multimodal (vision–language) training, and applications to 3D perception and robotics, which fits ICLR’s core topics.

## Minimum Quality
Pass ✅.  
The paper is complete and well structured: it contains Abstract, Introduction, Related Work, Method (with math), Experiments, Results, and Conclusion. The method is technically reasonable, empirically evaluated on many benchmarks, and the exposition is clear enough to understand and critique. No fatal methodological or theoretical flaw is apparent.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I see no indications of embedded instructions to reviewers, hidden prompts, or other manipulative content.

---

# Expected Review Outcome:

## Summary

The paper proposes **SpatialBoost**, a framework to enhance pre-trained vision encoders with 3D spatial understanding by leveraging language-guided reasoning. The core idea is to extract dense 3D cues from 2D images (depth, segmentation, 3D reconstruction), convert them into hierarchical multi-turn spatial Q&A chains (pixel / object / scene level), and use an LLM decoder plus a dual-channel attention mechanism to fine-tune vision encoders while largely preserving their original capabilities. SpatialBoost is applied to several strong encoders (OpenCLIP, SigLIPv2, DINOv2, DINOv3) and evaluated on dense prediction, 3D scene understanding, robot control, classification, retrieval, and VQA benchmarks, showing consistent gains, particularly on 3D-centric tasks.

## Strengths

1. **Clear, well-motivated problem and high relevance.**  
   The paper targets an increasingly important gap: strong 2D vision encoders often lack genuine 3D spatial awareness, limiting applications in robotics and spatial reasoning. The motivation is articulated clearly in the Introduction (Pages 1–2) and supported with prior work on multi-view pretraining and robot control. This is squarely in the sweet spot of representation learning for ICLR.

2. **Conceptually coherent framework that ties several components together.**  
   SpatialBoost combines (i) 3D cue extraction from images, (ii) conversion of these cues into hierarchical multi-turn spatial Q&A chains, and (iii) LLM-based supervision for vision encoder fine-tuning using a dual-channel attention mechanism. **Figure 1** and **Figure 2** nicely visualize this pipeline: Figure 1(a) shows the progression from single/multi-view images to depth, reconstruction, segmentation, and region captions; Figure 2 illustrates pixel/object/scene-level QAs and how they build up a chain-of-thought. This makes an otherwise complex system reasonably digestible.

3. **Comprehensive empirical evaluation across many task families.**  
   The experimental coverage is unusually broad:
   - Dense prediction (Table 1 & 2: NYUd/KITTI depth, ADE20K/VOC segmentation),
   - 3D scene understanding with Lexicon3D (Table 3),
   - Robot control with CortexBench (Table 4 and **Figure 4**),
   - Classification & retrieval (Table 5),
   - 2D and 3D VQA (Table 9),
   - Additional analyses and ablations (Tables 6–8, 15–18, and **Figures 5, 6, 7, 9**).

   SpatialBoost consistently improves over strong baselines across all encoders. For example, in **Table 1**, DINOv3’s NYUd RMSE improves from 0.31 to 0.25 (linear) and from 0.25 to 0.21 (DPT), while in **Table 3** DINOv3’s SQA3D BLEU-1 jumps from 51.4 to 54.9 and registration recall from 86.9% to 97.5%. These multi-domain gains are hard to attribute to chance or bad baselines.

4. **Interesting use of LLM-based supervision for spatial representation learning.**  
   Rather than supervising the vision encoder directly with pixel-level losses, the authors use an aligned LLM to decode spatial QAs, and the gradients flow back through a projector into the encoder (Section 3.1). **Table 6** is particularly informative: compared to fine-tuning with linear heads, SAM, or VGGT decoders, LLM-based fine-tuning yields the best combination of ImageNet classification (+2.0 accuracy points over DINOv2), ADE20K mIoU (+7.8 absolute) and NYUd depth RMSE (0.38 → 0.32) while also improving ScanQA BLEU-1. This provides at least some evidence that language supervision is not just a cosmetic choice but a competitive modality for dense knowledge transfer.

5. **Dual-channel attention as a pragmatic fine-tuning mechanism.**  
   Equation (1) formalizes the mixing of original and “plus” attention channels with an elementwise trainable gate α. During Stage 3, only the plus channel and α are updated, supposedly reducing catastrophic forgetting. **Figure 3** illustrates the architectural pattern clearly, and **Figure 6** together with **Table 17** show that naive full fine-tuning and LoRA both degrade ImageNet classification substantially (e.g., 86.3 → 79.5 / 81.7), whereas dual-channel attention boosts it to 87.6 while also improving depth. This is a neat, simple mechanism for “spatially specialized” adaptation.

6. **Evidence that the multi-turn hierarchical design and multi-view data matter.**  
   The ablations in **Table 7** and **Table 15** examine the effect of reasoning order and hierarchy: forward pixel→object→scene ordering systematically beats reverse or random ordering, and using all three levels (“Pix+Obj+Scene”) yields the best performance across depth, segmentation, and classification. **Figure 5** and **Table 18** further show monotonic improvements with more reasoning data. This addresses a common worry that “we added some synthetic data and it magically got better” by at least partially localizing what aspects of the data design contribute.

7. **Strong improvements on genuinely 3D-heavy tasks, not just 2D benchmarks.**  
   On Lexicon3D (Table 3), SpatialBoost dramatically boosts 3D semantic segmentation for OpenCLIP (mIoU 6.9 → 54.9) and SigLIPv2 (9.2 → 55.5), plus large gains in geometric registration (RR@0.05m from 22.6 to 78.8 for OpenCLIP). The robot control results in **Table 4** and **Figure 4** show sizable increases in average success rates (e.g., DINOv3 72.8 → 80.8), which is a strong signal that the representations capture something useful about 3D structure beyond dataset-specific quirks.

8. **Careful experimental protocol with frozen backbones in most probes.**  
   For the majority of downstream evals (dense prediction, Lexicon3D, classification, retrieval), the vision backbone is frozen and only a small task-specific head is trained, as described in Section A. This makes it more plausible that the observed gains truly come from improved representations, rather than from re-training large parts of the model per task.

## Weaknesses

1. **Very heavy reliance on external models and synthetic labels, with limited analysis of noise and bias propagation.**  
   The spatial Q&A data relies on DepthPro, SAM2, VGGT, CLIP-based filters, and GPT-4o for question generation and validation (Section 3.2, Appendix C/D). These models introduce their own biases and failure modes, yet the paper offers only a shallow analysis of this issue. The only systematic check is **Table 19**, which compares “VFM-based” vs “GT-based” data on ScanNet and concludes that performance differences are negligible. However:
   - This comparison is quite narrow (one dataset, 100K samples).
   - It uses ScanNet’s own GT, which may not be representative of the broader image domains (SA1B, Ego4D, etc.).
   - It does not address possible **systematic** biases (e.g., depth inaccuracies near object boundaries, indoor-centric geometry) versus simple noise.
   Given that SpatialBoost’s main claim is to inject accurate 3D spatial knowledge, the paper should dig deeper into how noisy 3D estimators and LLM captioners impact the learned representations.

2. **Attribution of gains to the proposed components vs just “more data and compute” is still somewhat murky.**  
   Many improvements could plausibly come from simply seeing 300K additional images plus GPT-generated captions / QAs. Although the authors include some informative ablations:
   - **Table 8** contrasts “Simple FT” with SpatialBoost under matched sample counts, showing that pre-training longer on the original objective yields much smaller gains.
   - **Table 6** compares LLM supervision against several pixel-level decoders.
   - **Tables 7, 15, 16** explore reasoning hierarchy and SV/MV mixtures.

   These are useful, but still leave important questions open:
   - There is no direct comparison to a **language-only** CoT dataset without explicit 3D reasoning (e.g., GPT captions plus random non-spatial QAs) under exactly the same data scale.
   - There is no comparison to a “simpler” language supervision where the LLM just predicts depth or 3D coordinates numerically, without the multi-turn narrative structure, to isolate the value of CoT.
   - It is also unclear whether a similar dual-channel fine-tuning using **pure 2D discriminative objectives** (e.g., more contrastive pretraining or image-only MIM) could unlock comparable gains at similar compute.

   As a result, the causal role of “language-guided spatial reasoning” vs “more synthetic multimodal supervision with extra compute” is not fully disentangled.

3. **Methodological clarity and analysis of the dual-channel attention mechanism are limited.**  
   The only formal definition is Equation (1), which defines  
   \[
   \mathrm{Attn}^{\text{final}}(\mathbf{x}) = \boldsymbol{\alpha} \cdot \mathrm{Attn}(\mathbf{x}) + (1-\boldsymbol{\alpha})\cdot \mathrm{Attn}^{+}(\mathbf{x})
   \]
   with elementwise α ∈ (0,1)^d. Several issues here:
   - There is no discussion of the **capacity and regularization** implications of doubling every attention module (+25–30% parameters per Section A.2). Why is mixing outputs linearly with a per-dimension α preferable to, say, LoRA on Q/K/V or a gated residual MLP?  
   - The notation suggests α is a vector applied elementwise, but the implementation details (which dimension is d: head dim? model dim? summed over sequence?) are not clarified; this matters for understanding how “selective” the gating really is.
   - The only empirical evidence comes from **Figure 6** and **Table 17**, both of which evaluate only DINOv2 on two tasks. The degradation of full fine-tuning and LoRA could be due to suboptimal hyperparameters or learning-rate schedules rather than an inherent property of those methods. The paper does not explore sensitivity to learning rate, number of trainable layers, or rank for LoRA.

   Overall, dual-channel attention is used as a black-box fine-tuning trick, with minimal mathematical or empirical justification beyond a single comparison.

4. **Some quantitative results are so dramatic that they raise questions about evaluation consistency.**  
   The jump in 3D semantic understanding for OpenCLIP and SigLIPv2 in **Table 3** is enormous; for example, OpenCLIP’s 3D SU mIoU goes from 6.9 to 54.9, and SigLIPv2 from 9.2 to 55.5. For such a massive gains, more detail is needed:
   - Are the downstream segmentation heads identical across all models, trained with the same hyperparameters and training schedule?  
   - Did pre-processing, point sampling, or class balancing change when using SpatialBoost encoders (since features now come from different layers or distributions)?  
   - Lexicon3D evaluates on ScanNet, which is also part of the multi-view reasoning data used in Stage 3 (Section C/D). While the paper claims only visual backbones are frozen during probing, reuse of the same scenes for both pretraining and evaluation could inflate apparent generalization.
   Without a more meticulous breakdown, it is hard to be fully confident that these large numbers solely reflect “better 3D understanding” rather than evaluation pipeline shifts or mild data leakage.

5. **Limited discussion of computational cost and practicality.**  
   SpatialBoost requires:
   - Running depth estimation, segmentation, and 3D reconstruction on 100K–300K images or multi-view sequences (Sections 3.2, D),
   - Using GPT-4o repeatedly to generate multi-turn QAs and validate multi-view rationales,
   - Training a Qwen-2.0-7B LLM in multi-stage fashion plus fine-tuning large ViT-g/ViT-7B encoders with 25–30% more parameters.

   The paper briefly notes training on 4×A100 in Section A.1 but does not quantify wall-clock cost, GPT usage, or memory requirements. For practitioners, this is non-trivial: generating this dataset and running Stage 1–3 fine-tuning is likely expensive, and the dual-channel models are heavier at inference too. A more explicit cost–benefit analysis or at least rough FLOP/token count would make it easier to judge whether the approach is a realistic recipe or a proof-of-concept.

6. **Spatial reasoning dataset construction is intricate but still somewhat opaque and potentially brittle.**  
   While Appendices C and D give many templates, some core design choices remain underexplained:
   - The exact algorithms for choosing coordinates and bounding cubes in the canonical space (Section D, “Point Cloud Processing”) are high-level; e.g., how many objects per image, how are small or overlapping objects handled, what is the distribution of distances and depth scales?
   - The multi-turn chain is always in the same order (pixel → object → scene → captions), but **Table 7** only investigates ordering qualitatively; there is no quantitative measure of question difficulty or reasoning depth.  
   - **Figure 2** shows a single illustrative chain, but there is no analysis of how often later turns truly require earlier rationales; it may often degenerate into independent questions coexisting in one conversation.

   Given that the multi-turn CoT design is central to the paper’s conceptual pitch, the lack of a more principled analysis or visualization (beyond templates) weakens the story.

7. **Related work is extensive but omits several directly relevant recent papers.**  
   The Related Work section focuses on self-supervised, multi-modal, and multi-view representation learning, but does not cover some very close recent efforts that also use language to inject spatial / depth reasoning into VLMs:
   - SSR: rationale-guided spatial reasoning for depth in VLMs.  
   - Works on enhancing visual reasoning abilities of LLMs and multi-hop reasoning via self-distillation and multi-prompt ensembling.  
   - Recent surveys on vision encoders in VLMs discussing fine-tuning strategies and spatial capabilities.  
   - VQA systems that use LLM-generated panoramic or multi-view captions for KB-VQA.  
   These omissions make it harder to clearly position SpatialBoost relative to other LLM-guided spatial reasoning pipelines and to argue for its unique contribution.

8. **Evaluation uses LLM-as-a-judge in key spatial benchmarks, with minimal calibration.**  
   In **Section B.1** (Table 9), SpatialRGPT and BLINK-D accuracy are measured via GPT-4 judgments due to multiple valid answers. While this is increasingly common, the paper does not:
   - Report inter-judge reliability or sensitivity to prompt phrasing,  
   - Compare GPT-4 judgments to standard automatic metrics on a subset (where they exist).  
   Since these scores are used to claim that SpatialBoost + Vicuna-7B surpasses closed-source LMMs like GPT-4o and Gemini-2.5 Flash, more rigor is needed before treating these as solid quantitative comparisons.

9. **Minor mathematical and notation issues.**  
   A few details are either confusing or sloppy:
   - In the formal description of multi-turn conversations (Section 3.1, Page 4), the notation \((\mathbf{x}_{\mathbf{q}}^{1},\mathbf{x}_{\mathbf{x}}^{1},\dots,\mathbf{x}_{\mathbf{q}}^{T},\mathbf{x}_{\mathbf{x}}^{T})\) is unclear: it appears that \(\mathbf{x}_{\mathbf{q}}^t\) and \(\mathbf{x}_{\mathbf{x}}^t\) denote question and answer token sequences, but this is not defined, and “x” is overloaded with image and text tokens.  
   - Equation (1) introduces α as a vector parameter per hidden dimension, but the paper does not explicitly state whether the same α is shared across all heads and tokens, or whether there is one α per head or per layer; this matters for understanding the parameter count and behavior.  
   These are not fatal issues but they hinder precise understanding of the architecture.

## Potentially Missing Related Work

1. **Liu et al., “SSR: Enhancing Depth Perception in Vision-Language Models via Rationale-Guided Spatial Reasoning”, 2025.**  
   This work explicitly transforms raw depth data into structured textual rationales to improve spatial reasoning in VLMs, which is conceptually very close to SpatialBoost’s idea of converting 3D cues into linguistic chains. It should be discussed in Section 2 (Multi-modal Learning or 3D-related methods), and it would be natural to contrast their rationale design and training pipeline with the proposed multi-turn CoT dataset. If feasible, SSR-style supervision could also serve as a baseline in some of the depth-related experiments (Table 1 or Table 6 comparisons).

2. **Li et al., “Enhancing Advanced Visual Reasoning Ability of Large Language Models”, 2024.**  
   This paper focuses on improving visual reasoning in LLMs, often using structured prompts and CoT. It is related to the Stage 2 visual instruction tuning and the spatial reasoning QAs in Section 3.2. It should be mentioned in Related Work when discussing CoT-based visual reasoning, and possibly compared qualitatively regarding how much reasoning is delegated to the LLM vs encoded back into the visual backbone.

3. **Wu et al., “Enhancing Multi-hop Reasoning in Vision-Language Models via Self-Distillation with Multi-Prompt Ensembling”, 2025.**  
   This work addresses multi-hop reasoning in VLMs via distilled supervision from multiple prompts. While the focus is different, it is relevant to the chain-of-thought, multi-turn design in SpatialBoost. Citing it in Section 2 or Section 3.2 and clarifying how SpatialBoost’s hierarchical QAs differ from multi-prompt ensembling would help situate the contribution.

4. **Xiao, “Vision Encoders in Vision-Language Models: A Survey”, 2025.**  
   Given that this paper’s stated goal is to systematically enhance pre-trained vision encoders used in VLMs, a recent survey on vision encoders would be a useful anchor. It should be cited in Related Work (end of Section 2) and possibly referenced in the Introduction when motivating why upgrading encoders’ spatial ability matters for downstream MLLMs.

5. **Tan et al., “Enhancing Few-Shot KB-VQA with Panoramic Image Captions Guided by Large Language Models”, 2025.**  
   This work uses LLMs to generate panoramic captions for VQA tasks, leveraging broader spatial context in language to aid visual reasoning. It is relevant to SpatialBoost’s scene-level QAs and scene caption turns. It could be cited in Section 2 and briefly contrasted with the proposed use of LLMs for spatial QAs rather than just for panoramic captions.

## Questions

1. **Data / model leakage and evaluation rigor for Lexicon3D.**  
   - Can you clarify exactly which splits of ScanNet (and other Lexicon3D-related datasets) are used for multi-view reasoning data vs evaluation? Are the same scenes used for both pretraining (Stage 3) and probing?  
   - If there is overlap, can you provide results where scenes from a dataset are *excluded* from the spatial reasoning data when that dataset is later used for evaluation, especially for the dramatic 3D SU gains in Table 3?

2. **Cost and practicality.**  
   - Could you provide approximate compute and cost numbers for the full pipeline: number of GPT-4o calls (for QA generation and validation), GPU-days for Stage 1–3, and time to generate the spatial QAs (including depth/segmentation/3D reconstruction)?  
   - How does inference cost compare between the original encoder and the dual-channel version (memory, latency), especially for ViT-7B/16?

3. **Role of language vs spatial content.**  
   - Have you tried a control experiment where you keep the same multi-turn format but replace spatial QAs with generic semantic QAs or captions (still generated by GPT-4o) to test whether spatial content per se matters, beyond “more linguistic supervision”?  
   - Alternatively, could you train a variant where the LLM predicts only numerical depth values or 3D coordinates (no narrative rationales), to isolate the benefit of CoT-style explanations?

4. **Dual-channel attention design choices.**  
   - Is α shared across tokens and heads within a layer, or is it head-specific / token-specific? Please clarify the exact tensor shape and parameterization.  
   - Have you explored variants where α is initialized to favor the new channel (e.g., bias toward 0) or where the plus channel is regularized to stay close to the original Attn weights (e.g., L2 penalty)? Any insights on stability or performance?

5. **Extremely large gains on 3D SU.**  
   - For Table 3, can you provide per-class mIoU breakdowns or qualitative visualizations of 3D segmentations before and after SpatialBoost to help verify that the gains are not concentrated in a few easy classes or artifacts of the evaluation?  
   - Are the feature dimensions, normalization, and projection heads identical across the compared models?

6. **LLM-as-a-judge calibration.**  
   - For SpatialRGPT and BLINK-D (Table 9), have you compared GPT-4 judgments against any automatic or human labels on a subset to confirm that the evaluation is not systematically biased in favor of one model family?  
   - Could you report some sensitivity experiments on the judging prompt (e.g., alternate formulations), or at least provide the exact prompt and a small annotation study?

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The method is technically reasonable and the empirical evidence overall supports the central claims, though there are open questions about attribution of gains, dual-channel analysis depth, and potential data leakage on some 3D benchmarks.

## Presentation Rating

3: good.  
The paper is generally clear, well-structured, and richly illustrated (Figures 1–7), with extensive tables (1–8, 15–21). Some notation and architectural details (e.g., α in Eq. (1), conversation tokenization) could be tightened, and the related work could better cover closely related spatial-reasoning LLM works.

## Contribution Rating

3: good.  
The combination of LLM-based multi-turn spatial reasoning with a dual-channel fine-tuning mechanism for vision encoders is a meaningful and timely contribution. It is not fundamentally new in every component, but the integration, scale, and breadth of evaluation make it a valuable addition to the literature.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

SpatialBoost presents a well-motivated and fairly substantial contribution: a concrete recipe to imbue strong 2D encoders with much better 3D spatial understanding using language-guided reasoning, and demonstrates consistent gains across diverse benchmarks, including robotics. At the same time, the framework is complex and resource-heavy, some ablations leave ambiguity about the precise source of improvements, and certain headline results (particularly on Lexicon3D) would benefit from deeper validation and transparency. I lean to a positive recommendation, conditional on the authors clarifying these points in rebuttal.

## Reviewer Confidence

4: confident.  
I am familiar with self-supervised vision, VLMs, and 3D representation learning, and I carefully checked the core math (Eq. (1)), the architectural descriptions, and the main experimental tables, though I did not attempt to reproduce the experiments.
# Automatic Image-Level Morphological Trait Annotation for Organismal Images

- Decision: Accept (Poster)
- Scores: 6, 8, 6

## Abstract
Morphological traits are physical characteristics of biological organisms that provide vital clues on how organisms interact with their environment. Yet extracting these traits remains a slow, expert-driven process, limiting their use in large-scale ecological studies. A major bottleneck is the absence of high-quality datasets linking biological images to trait-level annotations. In this work, we demonstrate that sparse autoencoders trained on foundation-model features yield monosemantic, spatially grounded neurons that consistently activate on meaningful morphological parts. Leveraging this property, we introduce a trait annotation pipeline that localizes salient regions and uses vision-language prompting to generate interpretable trait descriptions. Using this approach, we construct Bioscan-Traits, a dataset of 80K trait annotations spanning 19K insect images from BIOSCAN-5M. Human evaluation confirms the biological plausibility of the generated morphological descriptions. We assess design sensitivity through a comprehensive ablation study, systematically varying key design choices and measuring their impact on the quality of the resulting trait descriptions. By annotating traits with a modular pipeline rather than prohibitively expensive manual efforts, we offer a scalable way to inject biologically meaningful supervision into foundation models, enable large-scale morphological analyses, and bridge the gap between ecological relevance and machine-learning practicality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Morphological traits (the measurable physical characteristics of organisms) accurately predict how species interact with their environments, but extracting these traits remains a slow, expert-driven process. This paper contends that sparse autoencoders trained on foundation-model features yield monosemantic, spatially grounded neurons that consistently activate on meaningful morphological parts. Leveraging this property, we introduce a trait annotation pipeline that localizes salient regions and uses vision-language prompting to generate interpretable trait descriptions. Using this approach, we construct BIOSCAN-TRAITS, a dataset of 80K trait annotations spanning 19K insect images from BIOSCAN-5M. Human evaluation confirms the biological plausibility of the generated morphological descriptions. When used to fine-tune BioCLIP, a biologically grounded vision-language model, BIOSCAN-TRAITS improves zero-shot species classification on the in-the-wild Insects benchmark.

Authors' contributions: The recognition that sparse autoencoders (SAEs) can be used as interpretable part-detectors for trait extraction. In practice, training an SAE over pre-trained image features produces units whose activations map back onto tight, spatially coherent regions, and a method taking advantage of SAEs as trait detectors. A new dataset, BIOSCAN-TRAITS, and some experiments showing that it is useful for downstream classification tasks.

### Strengths
* This is an unusually well-written paper. I very much appreciate the clear dual expertise of the authors, which allows them to efficiently and accurately describe the mechanism by which SAEs operate as well as the challenges of Morphological Trait Extraction. The Related Works section of this paper, while not long, is good.
* The topic is important, and understudied; I'm very happy to see more work in this area.
* Table 4 is a nice contribution; it's a nice finding that BIOSCAN-TRAITS works better than species-only fine-tuning.

### Weaknesses
I do have some concerns about the authors overclaiming particularly in the introduction, and the results not being as comprehensive as I would like in certain places.

* The paper spends a long time on introduction and related work; we only get to the actual experiments on page 4. This is not a stylistic problem, as I recognize this paper probably needs more introduction than most, but it does mean there isn't as much room for actual description of the method and experiments. I think the second half of the introduction could be compacted, especially considering it contains many bold statements which aren't all that well supported.
* The method ablations are unusually weak for an empirical paper; only two backbones considered, DINOV2 and ViT-B-16 CLIP, and no ablations on the choice of SAE. A wider range of self-supervised and contrastive backbones would have been nice to see here, e.g. DINOV1, DINOV3, SigLIP, etc, and the decision to use an SAE is absolutely central to the method and should have been ablated.
* Personally I think the Algorithm 1 is not that important for understanding the work and should go to the appendix; I would rather see more core experiments (or ablations) here and the algorithm in the appendix.
* GradCAM is fine as a (weak) initial baseline; a more realistic baseline would be semi-supervised object detection based on target traits similar to (https://acsess.onlinelibrary.wiley.com/doi/10.1002/ppj2.20107), would also be nice to see how PCA, L1-regularized linear models, and Random projections do. These are extremely cheap to run, you don't even need a GPU.
* The authors write that the procedure is fully unsupervised; this is true only if you don't count the backbone -- DINO uses various kinds of weak and self supervision during pretraining, I think it would be clearer to include a mention of this. Furthermore, species-contrastive ranking sounds very much like it would require supervised / labeled data.
* BIOSCAN-5M is an unusually friendly dataset for this type of mapping because the insects are dead and presented mostly centered in the image, and well lit on plates, and there is always exactly 1 insect per image. It would have been nice to have seen an ablation on how well SAEs work on in-the-wild insect data, but failing that, I think the claim should be softened accordingly.
* SAEs are not nearly as robust as the paper's claims would indicate. Sometimes they work well, but often they're no better than much simpler, older approaches. https://arxiv.org/html/2501.16615v1, https://arxiv.org/html/2505.11756, https://arxiv.org/abs/2501.17148, https://arxiv.org/abs/2502.16681v1. Max Tegmark, Neel Nanda, Christopher Manning, Christopher Potts are all doubters of SAEs at this point. I think this should be mentioned as a significant limitation.

If these concerns can be addressed during the rebuttal, I think the paper is worthy of acceptance.

### Questions
* Are the images for the main figures cherry-picked? If so, the authors should say so.
* What are the limitations and drawbacks of the authors' approach? Under what circumstances will it be most effective?

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a pipeline for automatically annotating morphological traits at the image level from organismal images. The method (i) extracts dense visual features (DINOv2) and passes them through a sparse autoencoder (SAE) to obtain monosemantic, spatially grounded latent units; (ii) converts high-activation latents into localized boxes/masks; and (iii) prompts a multimodal LLM (here, Qwen2.5-VL-72B) to produce natural-language trait descriptions for those regions. Applying this to BIOSCAN-5M, the authors release BIOSCAN-TRAITS: ~80k trait annotations over ~19k insect images. Human evaluation indicates these SAE-guided descriptions are more precise than MLLM-only and Grad-CAM baselines. Fine-tuning BioCLIP on BIOSCAN-TRAITS improves zero-shot species classification on the Insects benchmark (+2.9% absolute), suggesting trait-level supervision boosts downstream generalization.

### Strengths
This is a well-written paper on a topic of significant importance. 

*Clear, modular pipeline with interpretability baked in*: SAEs yield localized, monosemantic units (e.g., “wing”, “antenna”) that ground the text prompts and reduce hallucination. The pipeline is easy to reason about and replicate. 

*Substantive dataset contribution*: BIOSCAN-TRAITS (80k traits / 19k images) fills a gap in large-scale, trait-level supervision for biodiversity ML.

*Downstream utility*: Fine-tuning BioCLIP with trait-level labels improves zero-shot accuracy on an in-the-wild benchmark, underscoring practical value.

### Weaknesses
*Model coverage of VLMs/LMMs is narrow*: Most experiments rely on Qwen2.5-VL (7B vs. 72B); broader open/closed LMM comparisons (and robustness under domain shift) would strengthen the paper.

*Cost/latency unquantified beyond tokens*: While the paper reports token counts per setting, it lacks a wall-clock + $-cost analysis per annotated trait/image across model sizes and batching regimes. This will be very useful for labs and museums planning large runs.

*BioCLIP only (no BioCLIP2):* Downstream tests fine-tune BioCLIP; given rapid progress in biologically grounded VLMs, results may be stronger (and more current) with BioCLIP2 if available. 

*End-user pathways are implicit*. The paper could more concretely describe how curators or ecologists operate the system (inputs, thresholds, QA loop, export formats) and how to integrate trait text with existing collection databases.

### Questions
**Why not BioCLIP2?** You fine-tune BioCLIP and show +2.9% on Insects. If BioCLIP2 is available, can you (a) replicate the gain, and (b) test whether trait-level supervision compounds improvements already present in BioCLIP2? Reporting both would isolate the incremental value of trait supervision from backbone/dataset size advances.

**Other LMMs (open vs. closed) and robustness**: Beyond Qwen2.5-VL (7B/72B), have you tried LLaVA-Next, Gemini-Vision, or GPT-4o-mini for trait generation? A compact matrix (quality, hallucination rate, token usage, throughput) would reveal portability, licensing trade-offs, and whether SAE-guided prompting narrows the gap between open and closed models. You already compared Qwen 7B vs. 72B; extending that table to 2–3 additional LMMs would be very informative.

**Cost-of-use analysis:**  Could you add: Per-image cost & time: preprocessing (feature extraction + SAE forward), #patches, tokens, MLLM seconds, and total $ at typical API/on-prem pricing; Throughput under batching and GPU memory constraints for open-weights models. This would let institutions forecast the budget for, say, 100k images.

**How do you envision end-users using it?**  It would help to spell out a reference workflow for curators/biologists
Even a short “Operations” section or a demo video link would greatly increase practical adoption.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work uses sparse autoencoders to generate trait descriptions of insect images from BIOSCAN-5M. The authors select a 19k image subset of the dataset labeled to the species level and generate trait captions. They perform a series of ablations to explore how the train description vary as a function of workflow components. Three domain experts provided ratings to assess quality. The authors investigate improvements in a zero-shot setting when using their dataset for fine tuning BioCLIP.

### Strengths
- The combination of SAE and MLLM is a new approach to auto-generating trait captions for biological images. 
- The authors provide a number of ablations, as reviewed by domain experts, to assess the quality of the resulting annotations as a function of different elements of their pipeline. 
- Overall writing clarity is ok, with a few confusing sections as noted below. 
- The annotation pipeline is the most significant part of the work thought it will need lots more testing in different domains, even of biodiversity imagery.

### Weaknesses
- The ablation study appears quite robust, but is missing an exploration of what $t_{activation}$ does. 
- The explanation of the zero-shot experiments need some clarification.
- The authors could spend some time discussing how their labeling strategy might impact a hierarchical framing of the classification problem. It seems they significantly reduce the amount of data available by requiring images to be drawn from the same taxonomic level.

### Questions
- Few of the terms in the equations in section 3.1 are defined. What do they represent?
- Line 200: what proportion of BIOSCAN-5M is annotated to the species level? 
- How is $t_{activation}$ set? How does varying the threshold impact trait selection? 
- Line 318: What is the rubric used for human quality rating? The scores are stated without a range so it is difficult to contextualize improvements. 
- In section 4.5, is 'in-the-wild' meant as natural images vs. the preserved samples in BIOSCAN-5m? Or something else? There are number of other distribution shifts inherent in the case of the former that worth expanding upon. 
- How were the traits encoded in when fine-tuning BioCLIP? 
- What is the species overlap with between BIOSCAN-5m and the BioCLIP training sets?

### Soundness
3

### Presentation
2

### Contribution
3

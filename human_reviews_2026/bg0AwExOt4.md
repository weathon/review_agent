# Aligning Forest and Trees in Images and Long Captions for Cross-Domain Grounding

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large vision-language models such as CLIP align images and captions as wholes but falter on long, detailed descriptions. Fine-grained understanding demands capture of hierarchical semantics, seeing both forest and trees, 
within and across domains. Yet syntactic and semantic structures seldom mirror visual organization, and vision alone tends to create spurious fragments unless text anchors and unifies.
We propose F-CAST, a hierarchical image-text representation learning framework that discovers aligned spatially oriented text and visual hierarchies directly from image and long-caption corpora, without region-sentence labels.  
It uses a CAST visual encoder for fine-to-coarse scene parsing and a hierarchical transformer text encoder that first encodes each sentence then fuses them into a whole-caption representation.  
A two-level alignment loss, extending FLAIR, aligns whole images with whole texts while biasing image-sentence matches so coarse concepts emerge from fine-grained evidence rather than ignoring it.
Trained on 30M image--text pairs, F-CAST delivers strong scaling and sets state-of-the-art performance on six long-text benchmarks.  
Experiments show that hierarchical alignment of vision and language enables F-CAST to discover fine-grained, visually grounded text understanding without supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes F-CAST, a hierarchical vision–language pretraining framework for long captions. On the vision side, it replaces a flat ViT with CAST to produce fine-to-coarse **segment tokens.** On the text side, it uses a two-stage hierarchical transformer: **Stage-1** encodes sub-captions (sentences/chunks), **Stage-2** composes them (with an adapter) into a whole-caption embedding. Cross-modal training uses two sigmoid losses: (i) a part-level text-grounded loss that aligns attention-pooled visual segments with sub-captions, and (ii) a whole-level loss aligning a global image token with the composed whole-caption embedding.

### Strengths
* Unified hierarchical alignment with simple losses; easy to slot into CLIP-style training while improving long-text robustness. 

* Strong empirical gains across six benchmarks; scaling plot indicates gains persist with more data. 

* Ablations indicate each piece (CAST, hierarchical text, two-level loss) contributes materially.

### Weaknesses
* **Limited novelty— Most ingredients are adaptations of prior work:** CAST for the visual hierarchy, and FLAIR-style text-conditioned image representations extended to a two-level (part/whole) loss. The paper itself positions F-CAST as “adopt CAST” + “extend FLAIR” with hierarchical text and a two-level alignment objective, which reads more like a careful systemization than a fundamentally new algorithmic idea.



* **Grounding validation gap:** relies on attention maps; lacks quantitative phrase↔region grounding benchmarks, making the “discovered hierarchy” claim hard to verify scientifically. 


* **Auxiliary-only parts at test time:** The part-level pathway is unused during inference, so is it necessary for the reported retrieval gains, or is it merely a training prior? A controlled experiment removes part-level loss at train time, but a matching compute is needed. 



* **Synthetic caption dependence:** trained on long synthetic recaps (DreamLIP); no robustness audits vs. noise/hallucination or transfer to human-written dense captions beyond the ones reported.

### Questions
* Quantify grounding: Can you report standard grounding metrics (e.g., phrase localization/pointing accuracy or region retrieval) on datasets with region annotations (RefCOCO/RefCOCO+/Flickr30k Entities) to substantiate the part-level alignment claim? How does F-CAST compare to FLAIR under identical protocols? 



* Ablate test-time dependence: What happens if you retain the part-level branch at inference (e.g., late fusion or re-ranking) vs. your current whole-only inference? Conversely, if you remove L_part during training, holding compute constant, how much of Table 1 survives? (Your Table 5 variant still shows a gap, but more controls are needed.) 



* Data realism & cost: Provide robustness studies to caption noise (shuffle/perturb chunks), and report FLOPs/throughput/memory vs. FLAIR/Long-CLIP to justify practicality. Also, clarify exact GPU-days per dataset and training schedule details beyond “5 days on 8×A100 for 30M”.


* Where does part-level alignment help? Your inference uses only the global embedding. What specific cases benefit from the part-level loss?


**I am open to changing my score based on the author's responses.**

### Soundness
3

### Presentation
2

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
The research addresses a critical limitation in current AI vision-language models: their inability to comprehend detailed, lengthy image descriptions containing specific details about objects, their attributes, and spatial relationships. The authors propose that effective understanding requires a hierarchical approach that mirrors human perception—simultaneously grasping both the overall scene and specific details.

Their solution, F-CAST, implements a three-stage hierarchical system that processes visual and textual information in parallel levels of granularity. The model progressively groups image patches into objects and then complete scenes while simultaneously analyzing individual sentences before combining them into full descriptions, ultimately aligning sentence-level details with corresponding image regions and matching complete captions with whole images. F-CAST is claimed to have achieved state-of-the-art results across six benchmarks and learned to associate specific phrases with image regions without explicit supervision.

### Strengths
Comprehension and matching long-text to images is an important and interesting problem. Also, and in general, the paper is well-written and technically sound.

### Weaknesses
As noted above, the challenge itself is not novel, nor is the solution strategy of joint learning. More detail on what is special about this specific manifestation of the problem and on the solution would have increase the novelty of this work.

### Questions
N/A

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
This paper proposes F-CAST, a hierarchical vision-language model that aligns fine-to-coarse visual structures with spatially oriented textual hierarchies for improved grounding in long-image caption understanding. By combining a CAST-based visual encoder and a two-stage hierarchical text transformer with a two-level alignment loss, F-CAST achieves state-of-the-art performance on six long-text image retrieval benchmarks without requiring region-sentence annotations.

### Strengths
1. The paper is well-written and easy to follow.
2. The proposed token reconstruction alignment and subcaption-aggregated patch alignment strategies are interesting and insightful.
3. Experimental results reflect the effectiveness of the proposed method to some extent.

### Weaknesses
I appreciate the contributions of this paper, but I have two main concerns—particularly the second one—that prevent me from giving a positive recommendation at this stage:

Although the F-CAST framework is interesting, its individual components appear to be borrowed directly from prior methods, essentially forming a combination of existing approaches.
F-CAST only reports quantitative results on long-text image retrieval benchmarks; its part-level capabilities are not sufficiently validated. I would expect the authors to align their evaluation with FG-CLIP or [1] and provide additional fine-grained results.

[1] UMG-CLIP: A Unified Multi-Granularity Vision Generalist for Open-World Understanding. ECCV, 2024.

### Questions
Please refer to the 'weakness' part.

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
This paper proposes a hierarchical image-text representation learning framework, F-CAST, which enables vision-language models to learn hierarchically aligned information from images and long captions. The proposed method exceeds the SOTA methods on six long-text benchmarks.

### Strengths
- F-CAST learns fine-grained, visually grounded text understanding without supervision.
- The proposed methods exceeds the SOTA methods on six long-text benchmarks.

### Weaknesses
- F-CAST aims to improve the alignment between the visual hierarchy of images and the spatial hierarchy of long captions. However, the method is primarily evaluated on long text-image cross-modal retrieval tasks, which mainly assess global-to-global alignment in vision-language models. The hierarchical alignment capability is only illustrated through visualization rather than quantitative evaluation. 
- The proposed method mainly focuses on local information at the sub-caption level. However, sub-captions often contain hierarchical information that may align with different visual regions. These multi-level relationships within sub-captions may not be fully captured by the proposed method. For example, in Figure 3, the first sub-caption also mentions “man” in addition to "horses."

### Questions
The paper mainly visualizes F-CAST’s ability to perform local alignment between sub-captions and image regions. What about the global alignment between the long text and the corresponding image?

### Soundness
3

### Presentation
2

### Contribution
3

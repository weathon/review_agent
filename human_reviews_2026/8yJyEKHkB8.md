# TextAtlas5M: A Large-Scale Dataset for Long and Structured Text Image Generation

- Decision: Reject
- Scores: 6, 0, 8, 4

## Abstract
Text-conditioned image generation has gained significant attention in recent years and is processing increasingly longer and comprehensive text prompts.  In everyday life, dense and intricate text appears in contexts like advertisements, infographics, and signage, where the integration of both text and visuals is essential for conveying complex information.  However, despite these advances, the rendering of images containing long-form text remains a persistent challenge, largely due to the limitations of existing datasets, which often focus on shorter and simpler text.  To address this gap, we introduce TextAtlas5M, a novel dataset specifically designed to evaluate long-text rendering, where ``long text'' refers not only to textual length but also to layout complexity and semantic richness.  In our context, long text involves dense visual content, hierarchical structures, and interleaved text-image layouts, as exemplified by subsets like TextVisionBlend, PPT2Structured, CoverBook, and TextScenesHQ.  Our dataset consists of 5 million generated and collected images across diverse data types, enabling comprehensive evaluation of large-scale generative models on long-text image generation.  We further curate 4,000 human-improved test cases (TextAtlasEval) across 4 domains, establishing one of the most extensive benchmarks for text rendering.  Evaluations suggest that TextAtlasEval presents significant challenges even for the most advanced proprietary models (e.g., GPT4o), while open-source counterparts show an even larger performance gap.  Notably, diffusion and autoregressive models with weak text rendering improve substantially after training on our dataset. These findings position TextAtlas5M as a valuable resource for training and evaluating next-generation text-conditioned image generation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents the construction of a text-rich image dataset and demonstrates its empirical effect for training and evaluation.

### Strengths
- The task of text rendering is highly important and has received significant attention recently; the paper’s focus is therefore timely and relevant.
- The dataset exhibits broad coverage across domains, and the generation/collection/annotation pipeline appears well-designed and reasonable.
- If the actual data quality is high, this resource is likely to be widely adopted by the community and could have a substantial impact.

### Weaknesses
Regarding the evaluation setting, it is unclear why existing benchmarks such as LongText-Bench and CVTG-2K are neither referenced nor compared.

### Questions
Overall, I am not deeply familiar with current works on datasets for text-rich image generation. I would appreciate it if the AC could place additional weight on the assessments of reviewers with stronger domain expertise when making the final decision.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper “TextAtlas5M: A Large-Scale Dataset for Long and Structured Text Image Generation” introduces TextAtlas5M, a 5-million-sample dataset designed to train and evaluate text-to-image generation models capable of rendering long, dense, and semantically rich text. The authors note that existing datasets (e.g., Marion10M, AnyWords3M, RenderedText) primarily feature short, simple text, which limits progress in generating complex, layout-rich visual text scenes such as posters, infographics, or presentation slides. TextAtlas5M comprises both synthetic and real-world data across multiple subcomponents, including CleanTextSynth, TextVisionBlend, StyledTextSynth, PPT2Structured, Paper2Text, CoverBook, LongWordsSubset, and TextScenesHQ. Each subset targets a specific aspect of text-image complexity, from clean typographic rendering to natural interleaving of text and visuals. The paper also introduces a 4,000-sample evaluation benchmark, TextAtlasEval, covering diverse text density and layout conditions. Quantitative evaluations of major open-source (AnyText, TextDiffuser2, PixArt, Infinity) and proprietary (GPT-4o, Grok3, DALL·E 3, Nano-Banana) models reveal that even the strongest models struggle with long-text generation, particularly in layout consistency. Fine-tuning autoregressive (Janus-Pro, Lumina-mGPT) and diffusion (PixArt-α) models on TextAtlas5M leads to substantial improvements across OCR accuracy, F1, and character error rate metrics, demonstrating the dataset’s effectiveness.

### Strengths
The dataset’s design is impressively comprehensive, incorporating multiple synthetic and real-world sources with detailed annotations and diverse domains.

### Weaknesses
While the dataset is highly valuable, the paper has several conceptual and methodological weaknesses. First, the notion of “long text” is operationalized mainly by token count and layout density, but not by semantic or structural complexity (e.g., multi-column alignment, cross-references, or multimodal discourse). Thus, while token length averages (≈149 per image) are impressive, the dataset may still not capture the true difficulty of coherent document-level generation. Second, the reliance on synthetic and LLM-generated annotations (via GPT-4o, Qwen2-VL, Intern-VL2) introduces potential linguistic artifacts, bias, or visual incoherence that are not systematically quantified or corrected. The authors claim to perform human refinement but do not specify the scale or inter-annotator agreement. Moreover, no standard metric is proposed for long-text fidelity beyond OCR and CLIP, which are known to be poor correlates of perceived quality in text rendering. The layout-aware IOU metric, while novel, is simplistic—it only counts bounding boxes, ignoring alignment, readability, or typographic aesthetics. From an experimental standpoint, while fine-tuning gains are striking, the absence of ablation studies (e.g., per-subset contribution or token length sensitivity) limits understanding of which data subsets drive improvement. Furthermore, evaluation against GPT-4o or DALL·E 3 uses models trained on proprietary data that might already include overlapping content, making the comparison somewhat ambiguous. Finally, the discussion section lacks deeper reflection on potential risks such as LLM-based synthetic data contamination, copyright issues in CommonCrawl-derived subsets, or bias propagation in real-world domains.

### Questions
The authors should better define “long text” beyond raw token counts, perhaps by introducing text layout complexity metrics (e.g., hierarchical density, information entropy of layout). Including inter-rater reliability for human-refined annotations would bolster credibility.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors introduce TextAtlas5M, a large-scale dataset of 5 million images designed to address the challenges of long-text image generation, where “long text” encompasses not only textual length but also layout complexity and semantic richness. 

The dataset includes diverse subsets featuring dense visual content and interleaved text-image layouts. In addition, the authors curate TextAtlasEval, a benchmark of 4,000 human-refined cases across four domains, providing a comprehensive evaluation of text rendering performance. Experiments reveal that even advanced models like GPT-4o face significant difficulties, while models trained on TextAtlas5M show marked improvement, establishing it as a valuable resource for advancing text-conditioned image generation.

### Strengths
*  The paper tackles a real gap in text-to-image research by building a large-scale dataset focused on long and structured text rendering.

* The proposed dataset covers both synthetic and real-world images (slides, posters, book covers, infographics, etc.), giving strong variety and realism.

* The proposed benchmark is practical. The TextAtlasEval test set is carefully curated and human-refined, with solid evaluation metrics (OCR, F1, CLIP, etc.).

### Weaknesses
Overall I dont have much concerns for this paper. I am curious about how to measure the aesthetic score for samples in the collected dataset? In real application usage, it is necessary to do further filtering. Besides, did authors conduct de-duplication for the dataset? And are all document images legal for release?

### Questions
N/A

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
3

### Summary
This paper focused on the well-known challenge in long-form visual text generation for image generative models. The authors curated a large-scale high-quality dataset of 5M images which contain dense long visual text by either synthesizing images or crawling real-world images. The dataset spans multiple diverse real-world scenarios, such as PowerPoint, arxiv papers, and posters. They also built an evaluation benchmark to assess the long-form text generation performance. The efficacy of the collected dataset was proved through the fine-tuning experiments on three open-source models.

### Strengths
1. The paper attempts to advance the research on long text generation, which is a useful real-life application scenario but still remains challenging for current SOTA image generation models. The future release of the curated dataset could benefit the whole image generation research community.
2. The overall data curation pipeline is systematic and deliberate.
3. Most details are described are listed in the main context or the appendix.
4. Some SOTA open-source or commercial image generation models are evaluated on the benchmark created by the authors.

### Weaknesses
1. The definition of long text is a little bit vague. Did you set a specific threshold or a range for the number of characters or words during data curation and filtering? Could you show the distribution plot of character/word counts across all the samples in the dataset or its subsets? 
2. The delivery of the paper needs improvement. Some parts are not well explained.
- In Figure 2, it would be better if you could explicitly denote which subsets are from the synthetic split, and arrange all blocks in order, i.e., putting relevant blocks close to each other. Currently it is hard to follow and not well aligned with Sec. 3.1 and 3.2
- It is better to explain the meaning of the phrase"interleaved documents" at its first occurrence.
- It could be clearer if there was also a pie chart about the proportions of each subset in the whole dataset like Figure 5.
- For table 2, please also highlight the best performance among all the commercial models so that the gap between open-source models and closed-source models would be clearer.  
- I think section 3.4 is not that informative and it could be merged with the sections about different dataset subsets.
- Some grammatical errors.
 (a) Line 92-93 "TextAtlasEval fill" should be "fills"
 (b) Line 354 should be "....remain a challenge...." or "...remain challenging..."
3. Have you tried to fine-tune the SOTA open-source models like Qwen-Image on TextAtlas5M? I find that it would be more convincing and promising if you could show that these SOTA open-source models could approach the commercial models like GPT-4 or nano banana after further fine-tuning. I understand that choosing the autoregressive baselines is for potential long-context reasoning. But it seems that the chose autoregressive baselines are much weaker than SOTA diffusion models. It would be better to start with stronger baselines. Also, could you please clarify why you chose Janus-Pro-1B and PixArt-α rather than Janus-Pro-7B and PixArt-Σ?

### Questions
1. Would it be possible to see the performance of fine-tuned models on existing public text generation benchmarks like the text rendering subset of OneIG-EN? Although it is good to see that finetuned models are excellent in generating short text according to Table 6, it would be better if you could also test on some well-recognized text generation benchmarks.
2. Did you use some text rendering tools to overlay text on to the empty space to synthesize images for StyledTextSynth?
3. What are the critical differences between PPT2Details and PPT2Structured?
4. How did you get the ground-truth for the number of text boxes in Table 3?

### Soundness
2

### Presentation
2

### Contribution
3

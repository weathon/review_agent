# Seeing Through Deception: Uncovering Misleading Creator Intent in Multimodal News with Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
The impact of multimodal misinformation arises not only from factual inaccuracies but also from the misleading narratives that creators deliberately embed. Interpreting such creator intent is therefore essential for multimodal misinformation detection (MMD) and effective information governance. To this end, we introduce DeceptionDecoded, a large-scale benchmark of 12,000 image-caption pairs grounded in trustworthy reference articles, created using an intent-guided simulation framework that models both the desired influence and the execution plan of news creators. The dataset captures both misleading and non-misleading cases, spanning manipulations across visual and textual modalities, and supports three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. We evaluate 14 state-of-the-art vision-language models (VLMs) and find that they struggle with intent reasoning, often relying on shallow cues such as surface-level alignment, stylistic polish, or heuristic authenticity signals. To bridge this, our framework systematically synthesizes data that enables models to learn implication-level intent reasoning. Models trained on DeceptionDecoded demonstrate strong transferability to real-world MMD, validating our framework as both a benchmark to diagnose VLM fragility and a data synthesis engine that provides high-quality, intent-focused resources for enhancing robustness in real-world multimodal misinformation governance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DECEPTIONDECODED, a multimodal news benchmark with explicitly defined creator intent to support misleading intent detection, source attribution, and desire inference. It reveals that current VLMs fail to reason about intent beyond surface alignment and stylistic cues.

### Strengths
1. The definition of creator intent as “desired influence + execution plan” is theoretically motivated by communication strategy literature and operationalized explicitly during content generation, which avoids the usual post-hoc inference ambiguity

2. Experiments across 14 diverse VLMs and multiple reasoning paradigms systematically show that even leading models like Claude-3.7 and GPT-4o struggle

3. Transfer experiments demonstrate that adding this intent-centric data improves Macro-F1 by up to +30.7 for Qwen-VL-7B, suggesting this dataset captures knowledge that generalizes beyond its own controlled setting and is beneficial for real-world scenarios

### Weaknesses
1. The dataset’s misleading intent is fundamentally pre-scripted by prompts and therefore risks oversimplifying real-world deception in which intent is shaped by dynamic sociopolitical incentives, audience feedback, and long-term agenda setting, none of which are captured by one-shot synthetic manipulations

2. The manipulations rely mainly on GPT-4o (text) and FLUX or GPT-image-1 (image) generation, so the dataset may inherit generator-specific patterns or stylistic artifacts

3. Evaluation tasks are closed-set classifications, whereas real intent is continuous, multi-layered, and often only partially present, so current task design may underestimate the ambiguity in real world cases

### Questions
1. In Table 2, GPT-4o-mini and several open-source models achieve almost 100 percent accuracy on non-misleading (NM) cases simply by predicting “NM”, so could the authors explain why such trivial behavior is not prevented by the dataset design and how this may inflate average accuracy.

2. Figure 3 shows models fail much more often when misleading text is written in a professional tone, so could the authors clarify how they ensured that “professional misleading style” does not accidentally include implicit emotional or sensational phrasing, which might cause style-based bias.

3. Since the article context is information-rich, and Table 4 shows that adding the article dramatically boosts performance for most of the models, could the authors analyze whether article–caption lexical overlap is causing shortcut exploitation instead of true intent reasoning.

4. In human evaluation, image-modified instances show noticeably lower agreement than text-modified ones (89.2 percent vs 99.2 percent), so could the authors show concrete examples where annotators disagreed to illustrate what types of visual manipulations fail to convey misleading intent clearly.

5. Table 5 shows that adding only one line of external framing hint leads to accuracy changes up to +54.5 or −41.8, so could the authors analyze which components of their prompt template are causing models to overly rely on such hints rather than visual–textual evidence.

### Soundness
2

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
4

### Summary
This paper introduces DECEPTIONDECODED, a benchmark dataset for analyzing misleading creator intent in multimodal news. The dataset contains 12,000 image–caption–article triplets, each grounded in verified VisualNews articles, with both misleading and non-misleading variants generated under predefined “creator intents.”  They evaluate 14 vision–language models, including GPT-4o, Claude-3.7, Gemini-2.5-Pro, and Qwen2.5-VL. The results indicate that even state-of-the-art models perform poorly on intent reasoning, tending to rely on surface-level cues such as image-text consistency or stylistic polish.

### Strengths
1. The paper moves beyond conventional multimodal misinformation benchmarks that emphasize factual misalignment, by explicitly modeling creator intention—a rarely addressed but crucial dimension in understanding real-world deception.
2. The cross-model comparison offers valuable insights into the current limitations of multimodal LLMs, especially regarding consistency-based reasoning versus implication-based reasoning.

### Weaknesses
1. Diversity of generated intents is insufficiently substantiated. The paper relies heavily on GPT-4o to simulate creator intents but does not provide a systematic method to ensure semantic diversity beyond domain coverage. Merely prompting GPT-4o may not guarantee varied “intent expressions,” leading to possible homogeneity in narrative framing.
2. Exclusion of deepfake/manipulated-identity content limits ecological validity. Identity-based manipulations (e.g., political figures’ face swaps) represent a significant real-world threat vector. The absence of such cases restricts the benchmark’s relevance to the broader multimodal deception landscape.
3. Lack of evaluation for misleading vs. non-misleading cases. Since the distribution is potentially skewed, reporting only accuracy can be misleading. The paper should provide F1 or macro-F1 scores to account for imbalance, and compare with related datasets such as NewsCLIPpings and MMFakeBench to show whether DECEPTIONDECODED introduces greater realism or challenge.
4. Ambiguity in controlling the boundary between subtle and significant manipulations. The paper distinguishes between “subtle” and “significant” misleading cases in both image and text manipulations, but it is unclear how the generation process enforces or verifies this distinction quantitatively. Moreover, the reported 2% human evaluation is used only to assess label consistency and plausibility, which is insufficient to guarantee that the subtle–significant boundary is consistently maintained across 12,000 samples.

### Questions
5. Ambiguity between the “implication-oriented (I)” and “consistency-oriented (C)” paradigms.
While both evaluation paradigms are central to the analysis, their conceptual distinction and implications for model reasoning are not clearly defined.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DECEPTIONDECODED, a novel benchmark designed to evaluate Vision-Language Models (VLMs) in detecting creator intent behind misleading multimodal news content. The dataset is constructed using a synthetic, intent-guided framework that generates manipulations grounded in real news, ensuring relevance and control over deception intent. The study evaluates state-of-the-art VLMs under various input conditions (e.g., image+text, text+article) and with authenticity cues (helpful or adversarial hints).

### Strengths
1 The paper shifts the focus from simple fact-checking to intent detection in multimodal misinformation, which is a more nuanced and realistic challenge. This addresses a critical gap in current VLM evaluation.

2 DECEPTIONDECODED is well-constructed, grounded in real news, and systematically manipulates intent. The use of human evaluation to validate realism, intent alignment, and label correctness strengthens the dataset’s credibility.

3 The experimental design is thorough, testing models under multiple input modalities and with adversarial authenticity cues. This provides deep insights into model behavior and failure modes.

4 The findings reveal specific weaknesses in current VLMs—such as susceptibility to misleading images, reliance on style over substance, and vulnerability to spurious hints—offering clear directions for future research.

### Weaknesses
1 The paper is purely diagnostic. It identifies problems but does not propose or evaluate any methods to improve VLMs’ robustness against the identified deception strategies, limiting its impact on advancing model capabilities. 

2 Although inter-annotator agreement is measured, judgments about “misleading intent” can be subjective. The paper could better discuss edge cases or ambiguous examples where annotators disagreed.

3 The study focuses primarily on caption manipulation within a fixed image-news context. Real-world misinformation often involves deeper fabrications, manipulated images, or entirely synthetic content, which are not addressed.

### Questions
1 The paper is purely diagnostic. How can the insights from this benchmark be used to build more resilient models? What concrete methods do the authors suggest for improving model robustness using these findings?

2 While inter-annotator agreement is reported, what types of manipulations led to disagreement on “misleading intent”? Could you discuss a few ambiguous cases to better illustrate the subjectivity in labeling?

3 The study focuses on caption manipulation with real images. How might the findings change if the images themselves were manipulated or synthetic, as is common in real-world disinformation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces DECEPTIONDECODED, a large-scale benchmark for understanding and detecting misleading creator intent in multimodal news. This work centers on modeling the combination of desired influence and execution plan behind deceptive news creation. The benchmark comprises 12,000 image–caption–article triplets, each grounded in trustworthy news contexts from VisualNews and simulated through intent-guided generation using GPT-4o and FLUX.1. It supports three intent-centric tasks: (1) misleading intent detection, (2) misleading source attribution, and (3) creator desire inference. Comprehensive evaluations of 14 VLMs reveal that even leading models struggle to reason about creator intent. Fine-tuning on DECEPTIONDECODED improves performance on external MMD benchmarks (e.g., MMFakeBench), underscoring its transferability.

### Strengths
1.The shift from content-level to intent-level misinformation detection is a underexplored direction that extends beyond factual correctness to communicative objectives.
2.The benchmark construction pipeline is well thought out: grounded in verified contexts, controlled through creator-intent configurations, and validated via human evaluation with high inter-annotator agreement
3.Evaluating 14 diverse VLMs across multiple reasoning paradigms (implication vs. consistency-oriented) offers valuable cross-model insights into weaknesses in current multimodal reasoning.

### Weaknesses
1.Absence of joint multimodal deception synthesis. Misleading instances are generated by modifying either text or image, but never both simultaneously. This simplification limits the ecological realism and potential difficulty of the dataset, as real-world misinformation often involves coordinated visual–textual deception.
2.Conceptual ambiguity of AI synthesis traces in the news context. The authors explicitly exclude AI-generated artifacts from being considered evidence of deception. However, this assumption may be problematic in the journalism setting, since the very use of AI-generated visuals in news reporting could itself indicate misleading or inauthentic intent. 
3.Over-reliance on generative synthesis instead of precise editing. The misleading visuals are produced entirely through image generation (e.g., FLUX.1), without exploring fine-grained editing operations such as subtle facial expression shifts, contextual object replacements, or localized semantic changes. This may reduce the dataset’s ability to capture nuanced real-world manipulations.

### Questions
1.Have the authors compared DECEPTIONDECODED against existing unimodal intent detection models? Can the dataset still challenge models when only one modality is provided? This would clarify the necessity and distinct value of multimodal intent reasoning.
2.Could the framework be extended to include dual-modality misleading constructions. For instance, simultaneously altering both caption and image? Would such jointly deceptive instances better reflect real misinformation and raise the benchmark’s difficulty?
3.Have the authors quantified the number of samples for each misleading-intent category or verified that the dataset is balanced across intent types and news domains? If certain intents (e.g., political or health-related deception) dominate, how might this imbalance affect model evaluation and the generality of conclusions about “intent reasoning” performance?

### Soundness
3

### Presentation
3

### Contribution
3

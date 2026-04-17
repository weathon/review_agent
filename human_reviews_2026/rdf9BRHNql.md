# TowerVision : Understanding and Improving Multilinguality in Vision-Language Models

- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Despite significant advances in vision-language models (VLMs), most existing work follows an English-centric design process, limiting their effectiveness in multilingual settings. In this work, we provide a comprehensive empirical study analyzing the impact of several multilingual design choices, such as training data composition, encoder selection, and text backbones. The result is  TowerVision, a family of open multilingual VLMs for both image-text and video-text tasks, built upon the multilingual text-only model Tower+. TowerVision achieves competitive performance on multiple multilingual benchmarks and shows particular strength in culturally grounded tasks and multimodal translation. By incorporating visual and cultural context during fine-tuning, our models surpass existing approaches trained on substantially larger datasets, as demonstrated on ALM-Bench and Multi30K (image tasks) and ViMUL-Bench (video tasks). 
Alongside the models, we release VisionBlocks, a high-quality, curated vision-language dataset. 
Our findings highlight that multilingual vision-language training data substantially improves cross-lingual generalization---both from high-resource to underrepresented languages and vice versa---and that instruction-tuned LLMs are not always the optimal initialization point. 
To support further research, we publicly release all models, data, and training recipes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents TowerVision, a suite of multimodal multilingual large language models (LLMs) for both image-text and video-text tasks, built upon the Tower+ models and SigLIP2. TowerVision leverages visual and cultural context during fine-tuning through a three-stage training process consisting of: (1) projector pretraining, (2) vision fine-tuning, and (3) video fine-tuning. The final TowerVision models outperform existing systems trained on much larger datasets across benchmarks such as ALM-Bench, Multi30K, and ViMUL-Bench.

Key contributions of the paper include the release of VISIONBLOCKS, a curated multilingual vision-language dataset; the open-sourcing of the models; and empirical insights into model backbones and training data composition for building more effective multilingual multimodal LLMs.

### Strengths
S1: The paper provides a suite of open-sourced multimodal multilingual LLMs for both image-text tasks, as well as video text tasks. The paper also provides a curated training dataset including both human and synthetically generated resources of 6M instances for enabling future research in this area.

S2: The evaluation of Towervision is extensive. They span across multiple modalities, languages (20+), and task types (VQA, OCR, translation, cultural reasoning). The resulting models are competitive against strong baselines (Qwen2.5-VL, Gemma3, Aya-Vision, CulturalPangea) and demonstrate robust performance.

S3: The analysis of where multilingual training matters is insightful, where multilingual backbones matter more than multilingual captions during alignment.

### Weaknesses
W1: While the paper provides an analysis of when multilingual training can improve the final LLM’s performance, a deeper examination of the data mixture would offer additional insights. For instance, the authors could analyze the benefits of training on different types of datasets (e.g., mathematical versus chart-based datasets).


W2: Although the paper addresses an important problem and contributes new models to multilingual multimodal research, it offers limited architectural and methodological novelty.

W3: The clarity of writing could be improved, as some important details are missing. For example:
 - Table 4: The Gemma2-2B/9B-IT models are not included in the comparison. How do the results compare to fine-tuning the IT model?
 - Table 5: Details about the dataset used for evaluation are missing. What data was used for these evaluations?
 - Line 398: The term “low-data regimes” should be clarified. What exactly does it refer to in this context?
 - Table 7: How does multilingual data affect video fine-tuning for the 9B model?

### Questions
- See W3
- Do you have any insights regarding to the data mixture? What type of training data is effective in improving the ViMUL-Bench?

### Soundness
3

### Presentation
3

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
Most Vision-Language Models (VLMs) are English-centric and struggle to generalize across languages and cultures. This paper introduces TowerVision and TowerVideo as two variants for building and analyzing multilingual vision-language models based on the Tower+ text backbone. The approach is supported by a new multilingual dataset, VisionBlocks, that combines translated, synthetic, and culturally grounded image-text pairs. The resulting models show improved cross-lingual understanding and cultural grounding on several multilingual benchmarks. The study provides a systematic investigation of multilinguality in VLMs and explores when and how to introduce multilinguality into them.

### Strengths
1. The paper is well motivated by the imbalance in current VLMs, which are predominantly English-centric. Addressing multilingual and cultural inclusivity is both timely and impactful, supporting fairer and broader use of VLMs across language communities. 
2. The introduction of VisionBlocks, a curated multilingual vision-language dataset, is a meaningful contribution. It provides a useful resource that directly improves multilingual performance and could benefit future research in this area. 
3. Beyond presenting a multilingual model, the paper offers several valuable observations, such as the effects of multilingual backbones, alignment stages, and fine-tuning strategies, that are informative not only for multilinguality but also for the general design of VLMs.

### Weaknesses
1. Section 2.2 primarily describes implementation details and training configurations rather than introducing a distinct modeling contribution. It remains unclear what is newly proposed in this work versus what follows established multilingual VLM pipelines from prior studies. Thus, from a technical standpoint, the work mainly extends existing pipelines with additional multilingual data. While the dataset itself is valuable, the overall improvement, training on more data with translated text in target languages, is expected rather than conceptually new. The contribution thus lies more in resource aggregation than in methodological innovation.
2. The proposed multilingual dataset is largely an LLM-augmented collection of existing image-text and video-text resources. Several design choices, such as using the Tower model for translation, evaluating with COMETKIWI, and adopting an arbitrary threshold of 0.8 are not well justified or empirically supported. These decisions appear somewhat ad hoc and lack evidence beyond internal preference or prior use by the same research group.
3. As the authors reported, the proposed models do not outperform baselines across all tasks. Results in Table 1 and Table 3 show mixed or marginal improvements, including multilingual video-to-text tasks, suggesting that the benefits are uneven across modalities and benchmarks.
4. The analysis remains largely quantitative, focusing on benchmark scores and metrics. Given the mixed quantitative results, a qualitative examination, such as cross-lingual error cases or culturally specific examples, would provide stronger insights into the actual behavior and limitations.

### Questions
The main concerns stand on the technical contribution, which appears limited beyond the introduction of new multilingual data. The results are also mixed. Additional qualitative or error analyses could help explain this. While minor, please double-check citation formatting (e.g., line 148).

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces TOWERVISION, a family of open multilingual CLMs designed to address the limitations of current English-centric VLMs . The authors also release TOWERVIDEO, a video-capable variant, and VISIONBLOCKS, a curated multilingual multimodal dataset. TOWERVISION utilizes a multi-stage training process and specific architectural choices to enhance multilingual capabilities. To overcome the scarcity of high-quality multilingual vision-text data, the authors created VISIONBLOCKS, a dataset containing approximately 6 million samples. It aggregates filtered public data, newly translated captions (using TOWER), and synthetic data generated via Gemini 2.5 to improve fine-grained visual details. TOWERVISION achieves competitive or superior performance on multilingual benchmarks like ALM-Bench (cultural understanding) and Multi30K (multimodal translation). Using a multilingual text backbone (TOWER+) consistently outperforms standard backbones (GEMMA2) for cross-modal tasks.

### Strengths
1. TOWERVISION achieves competitive or superior performance on various multilingual benchmarks, showing particular strength in culturally grounded tasks (like ALM-Bench) and multimodal translation (like Multi30K).

2. The paper provides a systematic study of multilingual design choices, analyzing the impact of different components such as training data composition, encoder selection, and text backbones.

3. The paper demonstrates that using a multilingual text backbone (TOWER+) and a multilingual vision encoder (SigLIP2) significantly improves cross-modal and cross-lingual performance compared to English-centric baselines .

4. The training approach substantially improves cross-lingual generalization, benefitting both high-resource and underrepresented languages, and even showing zero-shot improvements for unsupported languages.

5. The paper contributes open resources to the community, including the TOWERVISION and TOWERVIDEO models, training recipes, and VISIONBLOCKS, a curated high-quality multilingual vision-language dataset of approximately 6 million samples.

6. The work extends beyond static images to video with TOWERVIDEO, which achieves competitive performance on culturally diverse video benchmarks.

### Weaknesses
1. The finding that adding high-quality multilingual captions during the projector alignment stage yields "little to no positive effect" and sometimes "slightly decreases performance" on multilingual subsets is counter-intuitive to the paper's central claim that multilingual components improve performance. The paper currently relies on a surface-level hypothesis that focusing on "diverse, high-quality English captions" is simply more effective for standardizing visual and textual representations at this stage. This leaves a critical question unanswered: why does multilingual data fail at this specific stage when it succeeds elsewhere?

2. While TOWERVIDEO is presented as a major contribution, its implementation appears to be a straightforward extension of standard practices rather than a novel multilingual video approach. There is no ablation study for video-specific parameters equivalent to the depth provided for image resolution and tiling, leaving it unclear if 32 frames is optimal for multilingual video understanding, which often relies on reading temporally disparate text or recognizing subtle cultural cues.

3. The authors explicitly acknowledge that TOWERVISION is "less competitive on OCR-related tasks" due to a lack of OCR-focused data in VISIONBLOCKS. For a model aiming at robust multilingual capability, OCR is a fundamental requirement (e.g., reading menus, signs, or documents in various languages). Failing to address this known limitation weakens the claim of a truly comprehensive multilingual VLM.

4. The claim that VISIONBLOCKS is "high-quality" heavily relies on automated filtering, specifically using a COMETKIWI threshold of 0.85 for translations and Gemini 2.5 for synthetic data generation. While standard, these methods can miss subtle artifacts or cultural inaccuracies that automated metrics cannot detect, potentially undermining the "culturally grounded" goal of the model.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

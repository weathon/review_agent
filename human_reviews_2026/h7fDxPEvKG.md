# Immersive Multimodal Translation: A Proxy Task for Cross-modal and Objective Evaluation of Unified Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Unified multimodal models that jointly perform image understanding and generation have achieved substantial progress. However, a critical challenge persists in establishing rigorous evaluation protocols. Existing benchmarks typically assess generation and understanding tasks independently and rely on large multi-modal language models (MLLMs) for scoring. Such approaches introduce language-centric biases and lack objective ground truth, thereby limiting the reliability and fairness of model assessment. To address this, we propose Immersive Multi-modal Translation (IMT), a novel proxy task that requires models to translate textual content within images while preserving visual context. IMT naturally captures cross-modal synergy between understanding and generation, while enabling transparent, objective evaluation through established metrics from natural language processing and computer vision. To support systematic study, we construct IMTBench, a benchmark spanning three scenarios, including document, webpage, and scene image, with nine languages, and 2,000 carefully curated samples. IMTBench incorporates a three-dimensional evaluation framework measuring translation quality, background fidelity, and visual text rendering accuracy. Extensive experiments across diverse unified multi-modal architectures, supported by a companion dataset IMT-1M, reveal that current open-source models still fall significantly short of commercial expert systems. By providing objective, cross-modal evaluation protocols, we believe that IMT and IMTBench can offer actionable guidance for future research in unified multi-modal intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Immersive Multimodal Translation (IMT), which translates text in images while keeping visual context . The authors further propose IMTBench, a benchmark with 3 scenarios, 9 languages, and 2k samples. The authors use three metrics (translation quality, background fidelity, text rendering) to evaluate models. Experiments show open-source/closed-source models still lag behind commercial systems, but fine-tuning helps.

### Strengths
1. The paper is generally well-written and easy to follow.
2. The three-dimensional evaluation metrics (text, alignment, vision) are clear and comprehensive, avoiding one-sided assessments.
3. Fine-tuning experiments on IMT-1M provide practical insights (e.g., model convergence differences), which help improve unified models.

### Weaknesses
1. My biggest concern about this paper is that The IMT task may not truly assess the core multi-modal synergy of unified understanding-generation models. It mainly requires translating textual content in images and rendering the translated text, rather than demanding deep semantic reasoning across modalities. This means the task only tests surface-level multi-modal alignment, not the complex reasoning ability that unified models should possess .
2. The generation-side evaluation of the IMT task is too limited. It only focuses on whether the translated text is correctly rendered into the image (preserving background and layout) and does not require the model to generate semantically coherent content based on multi-modal understanding. Thus, it fails to test the core generation capability of unified models, which should involve semantic-based creative generation .
3. The task over-reliance on textual translation may bias the evaluation toward models with strong NLP capabilities, while ignoring their visual understanding and generation strengths/weaknesses. For example, a model with poor image context understanding but excellent translation performance could still score well on IMT, misleading the assessment of its overall multi-modal ability .

### Questions
Please see weaknesses.

### Soundness
2

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
3

### Summary
The paper introduces Immersive Multimodal Translation (IMT) as a proxy task to evaluate the synergy between image understanding and joint image/text generation. It presents IMTBench (2k items spanning Document/Web/Scene across 9 languages) and IMT-1M for training, and scores systems via three metrics—text quality, edit correctness, and background fidelity—aggregated into a single score. Results show commercial cascaded pipelines lead on text and alignment, while unified models better preserve visual naturalness. Fine-tuning on IMT-1M improves open-source models but they still trail the pipelines.

### Strengths
1.Clear task formalization and decomposition; the explicit prohibition on using T_src is consistent with end-to-end evaluation goals.

2.Three non-MLLM metrics with formal definitions; the scoring pipeline is reproducible.

3.Broad scenario and language coverage with transparent dataset statistics and visualizations.

4.Practical fine-tuning setups with hardware disclosure.

### Weaknesses
1.Limited methodological novelty. IMT primarily integrates existing TIT/IIT lines and known metrics; no new learning paradigm or loss is introduced.

2.Evaluation fairness. Proprietary systems are evaluated via paired understanding+generation APIs (e.g., GPT-4o + GPT-Image-1), which conflicts with the end-to-end restriction and may bias cross-paradigm comparisons.

3.Metric specification & robustness gaps. COMET version/normalization, OCR engine, and mask-generation details are missing; no human correlation or sensitivity analysis is provided.

4.Scene reference bias. Reference images are created by GPT-Image/SeedEdit, which may anchor S_vision toward those editing styles.

### Questions
1.Was train–test de-duplication performed between IMT-1M and IMTBench？

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes In-Image Machine Translation (IIMT) as a way to evaluate unified multimodal models (text and image modalities).
The authors introduce IMTBench, a novel benchmark covering 2,000 samples across 3 splits (scene, web, document) and 9 languages (Arabic, Russian, Chinese, English, Japanese, Italian, French, German, Spanish).
Furthermore, the authors generate 1 million training samples following the same pipeline and show that fine-tuning UniWorld can improve the models’ performance on IMTBench.

### Strengths
IIMT is a quite relevant task for the field with multiple widely adopted commercial solutions (such as e.g. Google Translate’s visual translation and equivalents). Yet, only limited image-to-image translation data is available for research.
The introduced IMTBench and IMT-1M training data seem a meaningful contribution to this area of research and improve over existing sets in diversity, scale, and realism.

### Weaknesses
* The paper’s opening claims that the IMT task is a more objective evaluation over other interleaved image-text generation evaluations such as InterleavedBench. However, this claim of higher “objectiveness” seems not substantially explored in the paper:
    * First, some relevant prior work such as ISG-Bench [1] or MMIE [2] are not discussed. The claim of the introduction that other benchmarks “can only evaluate performance in a single modality” seems not correct given this prior work (MMIE, ISG-Bench).
    * Secondly, the argument that since those evaluations include a model-as-a-judge would mean that they “cannot be regarded as truly objective metrics” seems to imply that using a model for judging is what makes them not “objective”, yet the proposed IMTBench also uses model based evaluation (COMET, OCR, Mask LPIPS). Moreover, the “Scene” split is generated using GPT-Image and SeedEdit which are then filtered manually for correctness. One may argue that such method will also introduce non-zero bias based on those generation model’s capabilities in the resulting dataset. Such issues are arguably minor but then the claim that the this evaluation will be clearly more objective than judgement with a strong multimodal LLM without further study is perhaps not clearly true.
* In section 4.1, the manuscript mentions that proprietary unified multi-models in the study cannot generate text and images simultaneously but GPT-4o’s native image generation had already been released back in March 2025. Earlier sections also mention the more recent Nano Banana model, which also offers such capabilities. Adding these or other frontier models would make the presented results more relevant.
* In section 5, fine-tuning based on the introduced IMT-1M sample is explored. Section 5.1 discusses training loss curves and attributes the shape and absolute values to differences in the model’s pre-training and architectures. However, it seems that this is based training runs following the models’ original training recipe which are naturally not consistent between these models. I am not sure if such limited study allows a conclusion such as “purely autoregressive architectures are limited by slower convergence, whereas strong diffusion-based pre-training enables faster generalization on the IMT task”. Furthermore, while loss curves are shown for Bagel, Uniworld, and Janus, table 2 only shows performance impact on UniWorld. Why are the results on the other models not reported? The qualitative result in Figure 6 is peculiar since as far as I can assess the French translation, the result after fine-tuning is incorrect and has no meaning in French. This seems to contradict the statement that the model “acquires correct semantic knowledge to produce accurate translations” and may imply that the fine-tuning may have impacted instruction following more than task quality (section 5.2). As such, evidence of the efficacy of fine-tuning on IMT-1M appears limited.
* There is limited study as to where and how models fall short on the tasks introduced in this work.


Minor: the paper has some typos:
* In figure 1, inconsistent spelling: “Single-modal”, “Single Modal”
* In section 1: “Nano Banana” is written as “Banana-nano”
* In section 4.1: “Findings” (upper case F when it should be lower case)


Overall, I feel this work’s main contributions are a novel dataset for the somewhat established IIMT task, which additionally includes a text ground truth. The proposed evaluation protocol seems defined based on a desire to reduce the reliance on model-as-a-judge evaluations. While the advantage of the proposed protocol is not directly studied, the protocol itself appears reasonable.
The writing appears to somewhat over-emphasis the novelty beyond these (already meaningful) contributions.


[1] Chen, Dongping, et al. "Interleaved scene graphs for interleaved text-and-image generation assessment." arXiv preprint arXiv:2411.17188 (2024).
[2] Xia, Peng, et al. "Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models." arXiv preprint arXiv:2410.10139 (2024).
[3] Gao, Yu, et al. "Seedream 3.0 technical report." arXiv preprint arXiv:2504.11346 (2025).

### Questions
* In section 2.2, the authors describe the task they call “IIT”. However, in the cited literature, this task seems to be called “TATI” (Qian et al., 2024) and “IIMT” (Lan et al, 2024, Tian et al., 2025a, Tian et al., 2025b). Is there a reason for the difference in naming?
* The introduced “IMT” task seems a combination of IIMT with outputting the translated text (as text) as described in 3.1. However, this is not clearly marked as the definition of “IMT”. Is my understanding correct? Can this be clarified in the text? (Additionally, in section 2.2 IIT is described as “directly replace the source language text within the image with the corresponding target language text, without generating textual output as an intermediate result”; Is this strictly true? While IIT, at least IIMT, may not test any intermediate text being generated I am not sure if generating such text would strictly speaking change the task? This may even happen transparently with e.g. prompt rewriting/expansion in some systems.)
* What OCR engine is used for S_align? Can this be documented in the paper to ensure reproducibility?
* For S_vision, can you elaborate how the binary mask works, exactly? The annotations shown in Figure 12 do not seem to show mask annotations? Also, wouldn’t this mean the method would not reward preserving text style / font? Has an unmasked version of the metric been considered?
* From [3], Seedream 3.0 supports only Chinese and English, yet it seems it was tested against IMTBench which covers 9 languages. Would it make sense to report performance on officially supported languages separately, perhaps in the appendix? Are the trends the same as when averaged across all locales? While Figure 3 provides some radar plots in this direction, it is my understanding that this still represents 9-1 and 1-9 averages, which do not allow this analysis?
* Nit / suggestion: In Figure 4, the target ground truth images are not provided, adding them may make the figure a bit more comprehensive?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper identifies a key challenge in evaluating unified multimodal models: existing benchmarks either assess generation and understanding tasks in isolation or rely on subjective, potentially biased scoring from other large models (e.g., GPT-4). To address this, the authors propose a new proxy task called Immersive Multimodal Translation (IMT). The task requires a model to take an image containing text, translate that text into a target language, and render the translated text back into the image while preserving the background.
To support this task, they introduce IMTBench, a new benchmark with 2,000 samples across documents, webpages, and scenes, spanning nine languages. They also propose a three-part evaluation framework based on objective, established metrics: (1) translation quality using COMET (Stext), (2) text rendering fidelity using OCR accuracy (Salign), and (3) background preservation using masked LPIPS (Svision). Their experiments show that specialized commercial pipelines significantly outperform current open-source and proprietary unified models on this task, suggesting a large gap for future work.

### Strengths
1. Well-Motivated Problem: The paper correctly identifies a significant issue in the field: the over-reliance on subjective, expensive, or biased MLLM-based scorers for evaluating generative multimodal models. The goal of creating an objective, automated evaluation is highly laudable.

2. Significant Data Collection Effort: The creation of IMTBench and the associated 1M-sample training set is a substantial engineering effort. Curating a multilingual, multi-domain dataset for this task is non-trivial and provides a new resource for the community.

### Weaknesses
1. The core of the proposed evaluation is insufficient. The $S_{align}$ score (OCR accuracy) only measures if the translated text is legible, not if it is rendered believably. Key visual aspects like font style, color, lighting, perspective, and blending—all crucial for an "immersive" result—are completely ignored. This is a major gap that undermines the claim of a comprehensive evaluation framework.

2. The central result from the experiments is that specialized commercial APIs (which are likely multi-stage pipelines of OCR, machine translation, and inpainting models) outperform end-to-end unified models. This is largely expected. Specialized tools are almost always better than general-purpose ones on their specific task. This finding does little to advance our fundamental understanding of unified model architectures.

3. A large portion of the benchmark (Document, Web) and even the ground truth for the "real-world" Scene images are generated synthetically. This limits the benchmark's ability to evaluate performance on the vast and unpredictable diversity of real-world typography and image conditions.

4. Could you elaborate on the novelty of the IMT task compared to the existing body of work on scene text translation and editing? What makes IMT a better "proxy task" for general multimodal intelligence than these related problems?

### Questions
While the paper is well-written and addresses an important problem, its core contributions are minor. The proposed evaluation protocol is insufficient for the very task it defines, as it might ignore the visual quality of the text generation. Please find the questions in the weakness part above, and hopefully the authors could address them.

### Soundness
2

### Presentation
3

### Contribution
2

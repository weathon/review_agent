# MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 6

## Abstract
Spatial intelligence is essential for multimodal large language models (MLLMs) operating in the complex physical world. Existing benchmarks, however, probe only single-image relations and thus fail to assess the multi-image spatial reasoning that real-world deployments demand. We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours meticulously crafting 1,000 challenging, unambiguous multiple-choice questions from over 120,000 images, each paired with carefully designed distractors and a stepwise reasoning process. We conduct extensive experiments and evaluate 37 open-source and proprietary MLLMs, observing a wide gap: the strongest open-source model attains roughly 30\% accuracy and OpenAI's GPT-5 reasoning model reaches 40\%, while humans score 97\%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research. Leveraging the annotated reasoning processes, we also provide an automated error analysis pipeline that diagnoses four dominant failure modes, including (1) grounding errors, (2) overlap-matching and scene-reconstruction errors, (3) situation-transformation reasoning errors, and (4) spatial-logic errors, offering insights for advancing spatial intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MMSI-Bench, a new VQA benchmark for evaluating the multi-image spatial intelligence of MLLMs. The dataset consists of 1,000 challenging, multiple-choice questions meticulously crafted by 3D-vision researchers, which are designed to be unanswerable from any single image. Each question is paired with a human-annotated, step-by-step reasoning process and is categorized into one of eleven spatial reasoning tasks. An extensive evaluation of 37 MLLMs reveals a substantial 55-point performance gap between human (97.2%) and SOTA model (41.9%) accuracy. The authors also provide an automated error analysis pipeline, identifying "overlap-matching and scene-reconstruction" as the dominant failure mode for current models.

### Strengths
1. The paper addresses a clear and important gap in MLLM evaluation, moving beyond existing benchmarks that often focus on single-image reasoning or use automated templates.
2. The dataset's fully human-centric curation process, involving 3D-vision experts and 8 diverse, real-world data sources, produces challenging and linguistically varied questions.
3. The authors provide an extensive evaluation of 37 MLLMs, establishing a robust baseline and highlighting a massive performance gap between SOTA models and humans.
4. The inclusion of an automated error analysis pipeline, which categorizes failures into intuitive types, provides concrete and actionable directions for future research.
5. The investigation into prompting techniques, including a novel visual prompting method using feature matching, provides strong evidence that current models have fundamental limitations in this domain.

### Weaknesses
1. The dataset size of 1,000 samples, while acknowledged as a result of costly human curation, is small. This raises questions about the in-depth diversity within each of the 10 sub-categories, which have ~100 samples or fewer on average.
2. The "multi-image" claim feels overstated, as all categories except for "Multi-Step Reasoning" are explicitly constrained to using exactly two images.
3. Key results that strongly support the benchmark's novelty, such as the poor performance of models specifically finetuned on other spatial datasets, are relegated to the appendix.
4. The methodology for constructing "Multi-Step Reasoning" questions is underspecified, lacking detail on how annotators were guided to combine the basic task types.

### Questions
1. You demonstrate that zero-shot linguistic and visual prompting fails to provide significant gains. Did you experiment with few-shot in-context learning by providing the full human-annotated reasoning chains as exemplars in the prompt?
2. The error analysis in Figure 7 is aggregated across the entire dataset. Could you provide a breakdown of the error type distributions for the most challenging categories, specifically "Multi-Step Reasoning" and "Motion (Camera)"?
3. The "Positional Relationship" category is very broad, covering six distinct sub-types. Do models show significant performance variation across these sub-categories (e.g., is "Cam-Cam" more difficult than "Obj-Obj")?
4. What specific guidelines were given to the 3D-vision researchers for creating the "Multi-Step Reasoning" questions, and how did you ensure these tasks truly required a sequence of the basic spatial skills?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents MMSI-Bench, a benchmark designed to evaluate multi-image spatial intelligence in multimodal large language models (MLLMs). The dataset contains 1,000 human-curated, multiple-choice questions requiring reasoning across multiple real-world images, with detailed reasoning annotations. The authors categorize 11 spatial reasoning tasks covering position, motion, and attributes, and benchmark 37 models. Results show a large gap between current MLLMs (best at \~42%) and human performance (\~97%), revealing major weaknesses in spatial reasoning across viewpoints. They also propose an automated error-analysis pipeline that identifies key failure types such as grounding and scene-reconstruction errors.

### Strengths
Originality: The focus on multi-image spatial reasoning fills a clear gap between single-image VQA and real-world embodied perception. The fully human-curated design adds credibility compared to prior template-based datasets.

Quality: The taxonomy of spatial relations (camera, object, region) is systematic, and the annotation process with reasoning traces and multi-reviewer verification shows rigor. The large-scale evaluation across 37 models is comprehensive and carefully controlled.

Clarity: The paper is well organized with clear figures and strong examples of question categories. The error typology (grounding, scene reconstruction, situation transformation, spatial logic) provides insight beyond raw accuracy.

Significance: MMSI-Bench exposes a real bottleneck in MLLMs’ ability to perform grounded spatial reasoning. The benchmark can drive future work on embodied AI, robotics, and multi-view understanding.

### Weaknesses
The dataset is still modest in scale (1k QA pairs), which limits generalization analysis. It would help to report variability or cross-split reliability. Many questions rely on human interpretation of viewpoint or direction. Some ambiguity might remain even with expert curation, which could affect reproducibility.

The evaluation metric focuses only on answer accuracy; assessing reasoning trace similarity (e.g., using annotated rationales) could reveal finer-grained improvements. While the benchmark is thorough, the paper lacks concrete recommendations or model design principles derived from the findings.

### Questions
1. How was question difficulty calibrated beyond human answer time? Did annotators estimate complexity or confidence?

2. Could reasoning annotations be used for training models (not just analysis)? If so, how does that affect overfitting?

3. How consistent are human annotators across the four error types? Any quantitative measure?

4. Did you observe differences between models trained with ego-centric data vs general web-image pretraining?

5. How might MMSI-Bench interact with embodied datasets like Habitat or RoboBrain for active perception tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces MMSI-Bench, a benchmark designed to evaluate the multi-image spatial reasoning capabilities of multimodal large language models (MLLMs). The dataset was created manually by six 3D-vision researchers. It consists of 1,000 multiple-choice question-answer pairs and 1,990 unique images sourced from eight real-world datasets (Matterport3D, ScanNet, DTU, nuScenes, Waymo, AgiBot-World, DAVIS 2017, and Ego4D). The questions are categorized into 11 tasks, under 4 main categories - positional relationships, attribute, motion and multi-step reasoning. The questions revolve around 3 spatial elements - camera, objects, and region. Most questions, other than the multi-step reasoning questions, involve 2 input images. Another key contribution is that each question is accompanied by a human-authored, step-by-step reasoning chain.

The paper conducts a comprehensive evaluation of 37 MLLMs. The primary result is a massive performance gap: the best-performing model (GPT-5) achieves only 41.9% accuracy, while human-level performance is 97.2%. The results also show that multi-step reasoning is particularly challenging for models. The authors also report fine-tuning and prompting ablations both of which provide minimal to no benefit. Language prompting involved zero-shot chain-of-thought reasoning and visual prompting involves highlighting PATS correspondences between image pairs. Finally, the authors provide insights into the failure modes of the evaluated models.

### Strengths
1. The benchmark's core strength is its manual, expert-driven annotation process. The questions are linguistically diverse, non-trivial, and require spatial understanding. The problem of multi-image spatial reasoning is highly relevant and timely for advancing embodied AI and robotics, and this paper clearly demonstrates a critical capability gap.
2. The paper evaluates an extensive suite of 37 models, providing a valuable and comprehensive snapshot of the entire SOTA. The inclusion of "Human Performance" (97.2%) is a strong baseline that effectively contextualizes the low model scores.
3. The findings that both advanced prompting (CoT, visual prompting) and fine-tuning fail to provide significant gains are valuable. These negative results strongly suggest that the models' failure is not a simple problem but a more fundamental capability deficit.
4. The paper is well-written making it easy to read and follow.

### Weaknesses
1. 1,000 samples is small size, especially when divided across 11 tasks. This limited scale is a direct trade-off for the high-quality manual annotation (300+ hours), but it makes the benchmark difficult to scale and creates a risk of models eventually overfitting to this specific test set.
2. Blind GPT-4o is not a suitable baseline since the questions depend heavily on the images. Language priors are unable to capture the context of the problem unless the images are described in words and the accuracy is expected to be similar to random baseline. The authors should update the baseline to be something more suitable.
3. The paper's analysis of the core problem is unclear. In model size ablations, the authors suggest the bottleneck lies in data quality and diversity. However, the paper's own experiments fine-tuning and prompting show that existing methods to bridge data gaps do not work. This suggests the bottleneck is more likely architectural or requires in-domain fine-tuning, which the paper does not explore.

### Questions
1. How are the multiple images fed into the models? Are they concatenated into a single image, as not all models natively support multiple image inputs?
2. The order in which the images are presented to the model seems critical, especially for tasks involving motion. How is this temporal order preserved and communicated to the model during evaluation? Can the authors provide an ablation study on the effect of image ordering?
3. Is the visual prompt (with correspondence lines) provided along with the normal image inputs? There is a concern that the lines themselves might occlude important details in the images, making them harder to see. That could be the reason behind marginal prompting gains.
4. The authors should provide details about the distribution of data across the 11 tasks.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces MMSI-Bench, a large-scale benchmark for evaluating multi-image spatial intelligence in multimodal large language models (MLLMs).
Unlike prior benchmarks focusing on single-image reasoning, MMSI-Bench evaluates an MLLM’s ability to reason across multiple images to infer spatial relationships, motion, and object-camera-region dynamics.
The dataset contains 1,000 multiple-choice questions (covering 10 atomic spatial reasoning categories and one multi-step reasoning task) curated by experts from over 120,000 real-world images drawn from datasets such as ScanNet, nuScenes, Matterport3D, and Ego4D. Each question includes human-written reasoning chains and carefully designed distractors.
Extensive evaluations of 37 MLLMs (including GPT-5, Gemini-2.5, Claude-3.7, Qwen2.5-VL, and InternVL-3) reveal that even the best proprietary model achieves only ~42% accuracy, while humans reach 97%, showing a large performance gap. The paper also presents an automated error analysis framework identifying four dominant failure modes: grounding, overlap-matching, situation-transformation, and spatial-logic errors.

### Strengths
Novel Benchmark Scope: MMSI-Bench uniquely targets multi-image spatial reasoning — a critical yet underexplored capability for MLLMs and embodied AI systems. Prior works (e.g., BLINK, ReMI, MuirBench) only contain limited spatial sub-splits, while this benchmark provides systematic coverage.

High-Quality, Human-Curated Data: Each question is manually designed and audited by multiple experts with reasoning explanations, ensuring clarity, difficulty, and lack of ambiguity. The benchmark’s construction pipeline (Fig. 4) and taxonomy (Table 1) are well-documented and rigorous.

Comprehensive Evaluation: The authors benchmark 37 models, analyze scaling trends, compare open-source and proprietary systems, and examine effects of CoT and visual prompting. This breadth enhances credibility.

Insightful Error Taxonomy: The four-type categorization of reasoning errors (Fig. 6) — grounding, scene reconstruction, situation transformation, and spatial logic — provides clear direction for future model development.

Impactful Findings: The results demonstrate that current MLLMs lack robust spatial reasoning and that scaling model size or prompt engineering yields marginal gains, implying fundamental architectural and data limitations.

Strong Writing and Presentation: Figures and examples (e.g., Fig. 2’s diverse question types) are clear, and the organization is consistent with ICLR standards.

### Weaknesses
Manual Effort vs. Scalability: Although the manual curation ensures quality, it also limits scalability — future expansions may face bottlenecks unless semi-automatic generation or verification methods are introduced.

Metric Simplicity: The benchmark reports only accuracy on multiple-choice tasks. Incorporating richer evaluation metrics (e.g., reasoning correctness or step alignment) could offer more granular insight.

Potential Dataset Bias: While data diversity is claimed, the benchmark draws primarily from common 3D and driving datasets, possibly biasing toward indoor and urban scenes rather than outdoor natural environments.

Limited Generalization Discussion: The paper does not test transfer to downstream embodied tasks (e.g., navigation, manipulation) where multi-view reasoning is crucial.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

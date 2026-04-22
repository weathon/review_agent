# TimeSpot: Benchmarking Geo-Temporal Understanding in Vision–Language Models in Real-World Settings

- Avg Score: 4.67
- Decision: Reject
- Scores: 6, 6, 2

## Abstract
Geo-temporal understanding, the ability to identify the location, time, and contextual features of an image from visual cues alone, is a fundamental aspect of human intelligence with wide-ranging applications, from disaster response to navigation and geography education. While recent vision–language models (VLMs) have shown progress in image geo-localization using conspicuous cues like landmarks or road signs, their ability to understand temporal signals and related spatial reasoning cues remains underexplored. To address this gap, we introduce **TimeSpot**, a comprehensive benchmark for evaluating real-world geo-temporal reasoning in VLMs. **TimeSpot** contains 1,455 images from 80 countries, where models must infer temporal attributes (season, month, time of day, daylight phase) and geolocation attributes (continent, country, climate zone, environment type, latitude–longitude coordinates) directly from the image. In addition, it includes spatial reasoning tasks that require integrating geographical, spatial, and temporal cues to solve complex understanding problems. Unlike prior benchmarks that emphasize obvious cues or iconic imagery, **TimeSpot** prioritizes diverse and subtle settings, reflecting the difficulty of reasoning under real-world uncertainty. Our evaluation of state-of-the-art VLMs, including both open- and closed-source models, shows consistently low performance across tasks, underscoring the substantial challenges that remain for robust temporal and geographic reasoning and the need for improved methods to achieve reliable and trustworthy geo-temporal understanding in VLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
As geo-temporal reasoning remains a major challenge for vision-language models (VLMs) in unconstrained real-world imagery, this paper introduces TIMESPOT, a benchmark dataset comprising 1,455 images collected from 80 countries. Each image is annotated with rich temporal attributes (season, month, time, daylight) and geographic metadata (continent, country, climate, environment, coordinates).
Comprehensive evaluations reveal persistent weaknesses in current VLMs: even the strongest models reach only 77.6% country-level accuracy, 33.7% time-of-day accuracy, and exhibit median geodesic errors exceeding 890 km. Weaker models perform far worse, with accuracies below 50% and extreme localization errors beyond 4,700 km. Temporal predictions often violate solar or hemispheric constraints, while spatial predictions overly depend on coarse priors—leading to systematic country swaps and climate misclassifications. Challenging conditions such as low light, twilight, or urban-canyon settings further exacerbate performance drops, suggesting that models fail to leverage shadows, illumination gradients, and fine-grained geographic cues. Overall, TIMESPOT exposes the fragility of current VLMs in joint geo-temporal reasoning, indicating that simple scaling or instruction tuning is insufficient for robust generalization.

### Strengths
1. This is the first geo-localization benchmark that explicitly integrates temporal information, introducing new dimensions of difficulty and realism for evaluating VLMs.

2. The paper conducts extensive evaluations across multiple models and tasks, providing a broad empirical picture.

3. The analysis of failure cases is detailed and insightful, offering concrete implications for future VLM development.

### Weaknesses
1. The dataset size (only ~1.5K samples) is relatively small, which limits its usability as a training resource; it can primarily serve as a test or diagnostic benchmark rather than a foundation for model training.

2. While the appendix is lengthy, the main paper body feels underdeveloped, giving an impression of imbalance and somewhat unpolished presentation.

3. The authors did not provide a public code or data release, which is critical for a benchmark paper to ensure reproducibility and community adoption.

### Questions
Could the authors elaborate on the limitations mentioned in the Weaknesses section?

### Soundness
3

### Presentation
3

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
The paper introduces TimeSpot, a new benchmark for evaluating geo-temporal reasoning in vision-language models, that is, the ability to infer both where and when an image was captured based solely on visual cues. TimeSpot consists of 1455 real-world images spanning 80 countries, annotated with nine structured fields: four temporal (season, month, time, daylight phase) and five geographic (continent, country, climate zone, environment type, latitude–longitude). The dataset emphasizes subtle and salient cues such as illumination, vegetation phenology, shadow geometry, architecture, etc. Evaluation is conducted on a range of open and closed source VLMs, assessing both categorical accuracy and geodesic/time errors. The results show that while some proprietary models achieve moderate country accuracy and coordinate precision, temporal reasoning remains very weak (~30% time accuracy, ~4h MAE).

### Strengths
The paper's primary strength lies in its well-defined motivation and problem formulation, directly tackling the missing evaluation of how vision-language models reason about when and where an image was captured. It introduces a benchmark with structured geo-temporal fields and validation grounded in physical realism. The dataset appears carefully curated, using a combination of deterministic label derivation and human verification to ensure accuracy and consistency. Altogether, these aspects make TimeSpot a robust, and valuable benchmark for advancing real-world spatiotemporal reasoning in modern VLMs.

### Weaknesses
**Scale & balance.** The dataset is moderate in size (1455 images across 80 countries) and shows clear skews, daytime (1182) vs. night (273), and urban (648) scenes dominate, while night and non-urban contexts remain underrepresented. This imbalance may limit generalization and calls for broader coverage or rebalancing.

**Annotation reliability not quantified.** The pipeline is described in detail, but no inter-annotator agreement, disagreement rates, or human-hours are reported, leaving label reliability and effort transparency unclear. Such statistics should be included to validate annotation quality and resource requirements.

**Calibration claims need numbers.** The paper discusses calibration, but explicit ECE or risk coverage results are not presented alongside the main tables. The authors should include these quantitative metrics for calibration-related claims.

**Limited temporal scope.** Although the benchmark aims to assess geo-temporal reasoning, it currently relies on single static RGB images, making it more about temporal inference than temporal reasoning. Incorporating multi-temporal sequences (e.g., same site across seasons or day phases) would better capture real temporal dynamics and improve robustness evaluation.

**Geographic imbalance.** The country distribution is uneven (e.g., 196 images from the USA but only a few from many others), which may cause models to rely on broad regional patterns. The authors can include frequency-weighted summaries to clarify how sample size affects country-level accuracy.

### Questions
Authors are encouraged to provide additional clarifications and supporting analyses addressing these points. Specifically, they should (1) report inter-annotator agreement, disagreement rates, and annotation effort to strengthen reliability claims; (2) include explicit calibration metrics such as ECE or risk-coverage tables alongside accuracy results; (3) examine how dataset imbalance affects performance through frequency-weighted or distribution-aware country analyses; and (4) discuss possible dataset extensions, such as multi-temporal and multi-modal variants (interesting to see), to better capture real temporal dynamics and reduce bias toward daytime, urban imagery.

### Soundness
3

### Presentation
3

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
This work introduces TIMESPOT, which contains 1,455 images across 80 countries with structured temporal (e.g., season, month, time, daylight) and geographic (e.g., continent, country, climate, environment, coordinates) attributes. This benchmark highlights the importance of evaluating real-world geo-temporal reasoning in vision-language models, which require integrating geographical, spatial, and temporal cues to solve complex understanding problems. Extensive evaluations show that substantial challenges in achieving robust temporal and geographic reasoning exist in both open- and closed-source models.

### Strengths
1. The paper is well-motivated, clearly articulating the need for evaluating geo-temporal reasoning in vision-language models. The proposed TIMESPOT benchmark addresses a meaningful and underexplored gap in the field, offering a novel and practical direction for advancing real-world multimodal understanding.
2. The error analysis section provides thoughtful and revealing insights into model failures, effectively highlighting the specific challenges models face in integrating temporal and geographic cues. These observations not only explain current limitations but also offer valuable guidance for future model design and dataset curation.

### Weaknesses
1. The writing of this paper is very hard to follow in many places; it feels like something went wrong during editing (maybe a find-and-replace gone wrong? e.g., hyphens (-) turned into commas (,) (see lines 049, 088-092, 100, 106-107, 136-142, 149-150, 159, 188-196, 201-206, 214-215, 223-226, 291-296…). As a result, the text often reads awkwardly or even confusingly, making it tough to grasp what the authors actually mean. 
2. Some figures and tables (e.g., Figure 1, Figure 2, and Table 1) aren’t referenced in the main text, so it’s unclear when or even whether you’re supposed to look at them. This makes the visuals feel disconnected from the narrative. Also, as a small but common formatting note: table captions should go above the table, not below. 
3. The paper mentions a “versioned JSON record” that includes ground truth, validation outcomes, human notes, adjudications, and constraint sets (line 297-299), but nowhere (not in the main text or the appendix) is there an actual example of what this JSON looks like. Without seeing a concrete instance, it’s hard to fully grasp how the data was structured or how the annotation and validation pipeline actually worked. 
4. Table 1 includes several benchmarks that aren’t really comparable to TIMESPOT, especially cross-view geolocalization datasets, which solve a fundamentally different problem. These would be better discussed briefly in the related work rather than listed side-by-side. Meanwhile, the table misses key existing datasets that focus on ground-level image understanding (e.g., [1-5]), making it harder to see how TIMESPOT advances the state of the art.

[1] LLMGeo: Benchmarking Large Language Models on Image Geolocation In-the-wild, CVPR 2024 Workshop    
[2] Image-Based Geolocation Using Large Vision-Language Models, Arxiv 2024          
[3] Evaluating Precise Geolocation Inference Capabilities of Vision Language Models, AAAI 2025 Workshop         
[4] AI Sees Your Location—But With A Bias Toward The Wealthy World, Arxiv 2025             
[5] From Pixels to Places: A Systematic Benchmark for Evaluating Image Geolocalization Ability in Large Language Models, Arxiv 2025

### Questions
Please see the Weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

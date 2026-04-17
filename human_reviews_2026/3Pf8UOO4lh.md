# TDBench: Benchmarking Vision Language Models on Top-Down Image Understanding

- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Top-down images play an important role in safety-critical settings such as autonomous navigation and aerial surveillance, where they provide holistic spatial information that front-view images cannot capture. Despite this, Vision Language Models (VLMs) are almost trained and evaluated on front-view benchmarks, leaving their performance in the top-down setting poorly understood. Existing evaluations also overlook a unique property of top-down images: their physical meaning is preserved under rotation. In addition, conventional accuracy metrics can be misleading, since they are often inflated by hallucinations or "lucky guesses", which obscures a model’s true reliability and its grounding in visual evidence. To address these issues, we introduce TDBench, a benchmark for top-down image understanding that includes 2000 curated questions for each rotation. We further propose RotationalEval (RE), which measures whether models provide consistent answers across four rotated views of the same scene, and we develop a reliability framework that separates genuine knowledge from chance. Finally, we conduct four case studies targeting underexplored real-world challenges. By combining rigorous evaluation with reliability metrics, TDBench not only benchmarks VLMs in top-down perception but also provides a new perspective on trustworthiness, guiding the development of more robust and grounded AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces TDBench, a new Q&A benchmark for top-down imagery. The dataset is constructed from existing image sources and includes around 2,000 curated questions. The authors also propose a new metric, RotationalEval (RE), designed to evaluate rotation-invariant reasoning by requiring correct answers across four orientations (0°, 90°, 180°, 270°). Using this benchmark, the paper evaluates multiple models and analyzes performance differences through equations relating RE, VE (Vanilla Evaluation), and MA (wrong answers). In addition, the authors conduct four case studies, including digital zoom, physical zoom, occlusion, and z-axis depth understanding, to further explore model behavior under different visual conditions.

### Strengths
- The proposed metric effectively leverages the rotation-invariant property of top-down imagery, which is an interesting and meaningful perspective. However, it is somewhat limiting that the metric only considers four discrete rotation angles (0°, 90°, 180°, 270°). Since the paper uses drones as an example, one might expect continuous viewing angles in real-world scenarios, where improving RE may not necessarily correlate with real-world performance.
- The comprehensive performance analysis across various models is impressive

### Weaknesses
- As mentioned in the paper, a satellite image can be considered a representative example of top-down images. However, it remains unclear whether the analyses obtained through the proposed TDBench provide new insights compared to existing studies that analyze similar domains with satellite images using vision-language models (VLMs).
- The statistics of the proposed benchmark are not clearly presented. It is difficult to determine how many image–QA pairs exist for each of the ten tasks, and how real versus synthetic data are distributed across them. For example, Sec. B.2 mentions 2,200 images, while Sec. 3.4 refers to 2,000 problems. If space limitations in the main paper prevented including these details, it would be helpful to at least provide a summary table in the appendix.
- The dataset consists of images (I), questions (Q), choices (C), and labels (L). While the images are sourced from existing datasets, the paper lacks a clear explanation of how Q, C, and L were generated.
- In Figure 9 (Sec. B.1), there appear to be four task types (object localization, attribute recognition, visual grounding, spatial relationship) that are not rotation-invariant, as the correct answer changes upon rotation. It is unclear whether new choices and labels were constructed for each rotation case in such situations.
- The analysis in Sec. 4.3 is interesting but seems to rely on strong assumptions when formulating relationships among RE, VE, and MA. The notion of distinguishing questions as known or unknown is ambiguous. Does it refer to whether the image–question–answer pair appeared in training, or something else? Moreover, it is questionable whether rotated images can truly be treated as independent samples, and the meaning of conditionality in this context is unclear. While the equations may reveal some trends, it is debatable whether they support the claim of a “deeper analysis.”
- Although four case studies are presented, only the fourth one appears to be specific to top-down imagery. The others (digital zoom, physical zoom, occlusion) could apply equally to general spatial imagery, and therefore do not provide novel or top-down–specific insights.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a new benchmark for top-down image understanding. Current LVLMs are primarily trained and evaluated on frontal views; consequently, their performance on top-down views is worse than on frontal views. To evaluate this gap, the TDB benchmark is introduced, along with additional metrics such as rotation-invariant evaluation and probability-based knowledge reliability analysis.

### Strengths
1. Evaluating LVLMs from a top-down view is novel, and there are many critical applications in this domain.

2. The dataset covers a wide range of tasks from a top-down perspective.

3. The motivation is clear, and the paper is easy to follow.

### Weaknesses
1. The contribution seems limited, as the paper primarily focuses on evaluating different LVLMs on the proposed TDB benchmark.

2. Providing more insights into why model performance degrades under the top-down view would be beneficial. An initial attempt to address this issue could further strengthen the paper.

3. The paper lacks comparisons with domain-specific models. For example, reporting the performance of state-of-the-art models on selected TDB tasks could help better illustrate the performance gap.

### Questions
What is the performance of state-of-the-art models on selected TDB benchmarks?

If domain-specific models perform well on this data, would it be possible to use them to process the images and then feed the results to LVLMs or LLMs for further analysis?

### Soundness
2

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
This paper presents TDBench, a new benchmark designed to evaluate Vision–Language Models (VLMs) in top-down image understanding scenarios, such as aerial and autonomous navigation views. The benchmark includes 2,000 curated QA pairs per rotation angle, and introduces RotationalEval (RE), an evaluation metric that measures answer consistency across four rotated views of the same scene. The authors further propose a reliability analysis framework to separate genuine understanding from chance-level responses. Extensive experiments across major VLMs (GPT-4V, Gemini, Claude, InternVL, Qwen-VL, etc.) show that existing models perform poorly in rotational robustness and spatial consistency. Overall, the paper highlights an underexplored but practically important perspective in multimodal evaluation: understanding and reliability in top-down perception.

### Strengths
1. The benchmark addresses an overlooked setting (top-down perception), which is practically relevant for safety-critical applications such as aerial surveillance and navigation.
2. The dataset and evaluation framework are clearly described and carefully curated, with attention to rotation invariance and visual grounding reliability.
3. The authors evaluate a broad range of state-of-the-art VLMs and provide detailed error and reliability analyses.
4. The paper is well organized, with strong motivation, readable methodology, and informative figures.

### Weaknesses
1. The core contributions are primarily in dataset and evaluation design rather than algorithmic or theoretical advances. RotationalEval is intuitive and could be viewed as a straightforward extension of consistency testing.
2. The reliability framework is described conceptually but lacks a formal quantitative or probabilistic definition of “genuine knowledge vs. chance.”
3. While rotation is important, other forms of geometric variation (scale, translation, occlusion) are not addressed. The study could be more comprehensive in analyzing general spatial invariance.
4. Some question templates and image sources are limited in semantic diversity, which might restrict the generalizability of conclusions.

### Questions
1. How does RotationalEval handle questions that are not rotation-invariant (e.g., “What is on the left/right side of the image?”)?
2. Can the proposed reliability framework be formalized mathematically, rather than qualitatively distinguishing “genuine” vs. “lucky” answers?
3. Have the authors considered evaluating compositional reasoning or anomaly reasoning tasks in aerial views to complement spatial consistency?
4. How well do current reasoning-based models perform on this task? Could the paper include a more detailed comparison or discussion of reasoning-centric approaches and their limitations in contextual anomaly understanding within top-down scenes?
5. How scalable is TDBench？Can it be extended to larger multimodal corpora or fine-tuned models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Motivated by a lack of benchmarks dedicated to top-down image understanding for outdoor images, the authors propose TDBench. TDBench is a benchmark of 2000 questions with images drawn from public datasets of drone images and simulated environments. It consists of two question types, one involving multiple-choice questions and another involving bounding box prediction. The dataset features 10 question categories with 200 questions each. A key design feature is "RotationalEval," which evaluates model responses to questions for 4 separate rotations of the image and treats a question as known if answers to all rotations are correct. The authors evaluate a total of 60 vision-language models on the dataset and conduct additional case studies using a held-out dataset of 2100 questions.

### Strengths
- The motivation behind the dataset is valid, as there is a lack of benchmarks for top-down image understanding that do not place object detection as their focus.
- The RotationEval setting is a clever way to mitigate the impact of random guessing on a dataset that relies on multiple choice questions for evaluation.
- The authors evaluate a broad range of models, both proprietary and open, of various sizes (60 total). Figure 3 is acts as a very clear way of demonstrating results and scaling trends, particularly when there are this many models.
- The data dedicated to case studies help assess the impact of various features such as drone altitude or the proportion of the target object in the image.

### Weaknesses
- While the authors note in the appendix that the dataset was filtered for correctness, there are two analyses that I think would help verify the quality of the dataset. Firstly, there should be a text-only evaluation without any images to ensure that the dataset (or particular question categories) are not solvable using guesswork or spurious correlations. Secondly, human accuracy should be provided, at the very least for a subset of questions, to have an oracle estimate of accuracy.
- It was somewhat difficult to parse the definitions in Section 4.3. Particularly, I'm not sure what the condition is for a question to be "known" or "guessed." In the framing of the RotationEval section, guessing implies inconsistent answers for the rotations. If so, how do "known" questions not have an accuracy of 100%?
- The key contribution of TDBench is providing a benchmark for top-down image understanding that does not rely purely object detection. At the same time, however, questions categories such as Object Presence, Counting, Localization, Visual Grounding and Spatial Relationship effectively rely on the exact same set of skills as object detection but render them in QA form.

### Questions
- How was the preliminary audit for VisDrone conducted?

### Soundness
3

### Presentation
3

### Contribution
3

# GenExam: A Multidisciplinary Text-to-Image Exam

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Exams are a fundamental test of expert-level intelligence and require integrated understanding, reasoning, and generation. Existing exam-style benchmarks mainly focus on understanding and reasoning tasks, and current generation benchmarks emphasize the illustration of world knowledge and visual concepts, neglecting the evaluation of rigorous drawing exams. We introduce GenExam, the first benchmark for multidisciplinary text-to-image exams, featuring 1,000 samples across 10 subjects with exam-style prompts organized under a four-level taxonomy. Each problem is equipped with ground-truth images and fine-grained scoring points to enable a precise evaluation of semantic correctness and visual plausibility. Experiments show that even state-of-the-art models such as GPT-Image-1 and Gemini-2.5-Flash-Image achieve less than 15% strict scores, and most models yield almost 0%, suggesting the great challenge of GenExam. By framing image generation as an exam, GenExam offers a rigorous assessment of models' ability to integrate understanding, reasoning, and generation, providing insights on the path to general AGI. Our benchmark and evaluation code will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces GenExam, a benchmark of 1,000 exam-style text-to-image tasks spanning 10 subjects (math, physics, chemistry, biology, computer science, geography, economics, music, history, engineering). Each item includes a ground-truth image, a four-level taxonomy, and fine-grained scoring points to evaluate semantic correctness and visual plausibility (spelling, logical consistency, readability). Evaluation is performed with MLLM-as-a-judge (GPT-5) using per-item scoring questions; both strict (all points correct + perfect plausibility) and relaxed scores are reported. Experiments across closed-source and open-source models show very low strict accuracy (<15% for top systems), underscoring the task’s difficulty. The paper also conducts a human-alignment study on a subset and reports high correlations between the relaxed score and human ratings.

### Strengths
- Well-posed benchmark design with executable scoring criteria: The benchmark formalizes exam-style prompts with per-item scoring points and ground-truth images, enabling precise judgments of semantic correctness beyond aesthetic similarity. The two-axis evaluation (semantic correctness; plausibility via spelling/logic/readability) and strict vs. relaxed scoring are clearly defined and motivated. 
- Breadth and hierarchy of subject coverage: GenExam spans 10 subjects and provides a four-level taxonomy grounded in standards (e.g., ISCED-F), with examples covering rigorous, diagram-like prompts. This supports systematic analysis by domain and difficulty. 
- Clear data-curation pipeline and statistics: The paper documents a multi-stage pipeline (web & existing datasets → GPT-assisted generation of prompts & scoring points → manual PhD triage and correction) and provides statistics on prompt length and scoring-point counts.

### Weaknesses
- Evaluator dependence and potential circularity: The benchmark’s scoring relies heavily on GPT-5 as the MLLM-judge with crafted prompts. While some robustness checks are in the appendix, the main text still leaves open concerns about model-judge coupling, evaluator drift, and template sensitivity. Broader cross-evaluator comparisons (e.g., non-GPT evaluators) and more extensive human studies across multiple model outputs (not only GPT-Image-1) would strengthen claims. 
- Baseline fairness and statistical rigor: The paper compares many systems but states “default T2I configuration” for inference; strict/relaxed scores are presented without variance or significance tests. Given the exam-style nature and the low absolute accuracies, multiple runs and budget-matched settings (e.g., seeds, sampling steps) are important to avoid confounds.
- Data provenance & annotation details remain under-specified: Prompts and scoring points are GPT-assisted and then manually corrected. However, the paper does not quantify inter-annotator agreement, error rates in the auto-drafts, or detailed quality-control outcomes post-triage. More transparency on annotation reliability would improve reproducibility.
- Strict scoring may conflate layout micro-errors with semantic failure: The strict score requires all scoring points correct and 2/2 on each plausibility dimension; a single mislabeled arrow or minor typographic slip yields a zero. While intentionally stringent, the metric may obscure partial semantic competence. Reporting per-point precision/recall, or a “nearly-correct” band, could help diagnose capability gaps without over-penalizing minor layout issues.
- Diversity & sensitivity of the dataset: Although the paper claims broad subject coverage and shows type distributions, it does not quantify demographic, linguistic, or regional diversity (e.g., English-only prompts? region-specific map conventions? notation variants across curricula). Subjects such as history/geography and even biology/chemistry can encode cultural or geopolitical sensitivities (e.g., territorial maps, naming conventions, symbol sets). The Ethics section asserts that potential biases were “checked,” but provides no measurement or sensitivity analyses (e.g., subgroup performance by subject region, notation variants, left-to-right vs right-to-left scripts). A structured bias audit (coverage metrics, subgroup error rates, adversarial examples touching sensitive content) would substantiate the claim of “balanced knowledge coverage.”
- Limited release specifics for evaluator prompts & assets: The paper references evaluator prompts (Appendix F.3) and plans to release code, but the main text does not specify exact release artifacts (e.g., frozen judge prompts, scoring-point JSONs, splits, evaluation harness). Locking these down is critical for reproducibility and longitudinal tracking.

### Questions
- Will you release all per-item scoring points, judge prompts, and a frozen evaluator config so others can replicate strict/relaxed scores exactly?
- Can you report variance/significance across repeated generations with matched budgets for a subset of models?

### Soundness
3

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
3

### Summary
GenExam is a benchmark for complex open-ended QA across 16 disciplines, featuring long-form questions requiring multi-step reasoning. The authors test several top models using an LLM-based rubric and human graders, analyze alignment between automated and human scores, and categorize tasks by cognitive skill to pinpoint reasoning challenges. Results show even the best models perform far below human experts, illustrating the benchmark’s difficulty.

### Strengths
1. Spans 16 disciplines with long-form answers, offering a broader and more realistic evaluation than prior multiple-choice benchmarks and probing high-level reasoning.

2. Uses several SOTA models and dual scoring (LLM rubric + human); checking LLM scores against human grades adds credibility and ensures automated grading aligns with human standards.

3. Categorizing tasks by cognitive skill yields detailed insights into model performance; rubric-scored long answers highlight specific reasoning failures that an accuracy metric would miss.

### Weaknesses
1. Qualitative analysis needs more depth in terms of the pattern in the errors in open-source vs closed-source models.
2. The cognitive skill categories are underdefined – without clear definitions or examples, conclusions about specific reasoning weaknesses lack support.
3. To check semantic correctness, the paper adds the ground truth image as input, so that the MLLM judge can use it as a reference. Why did they not provide the textual description of the ground truth image which might be more reliable?

### Questions
1. To check semantic correctness, the paper adds the ground truth image as input, so that the MLLM judge can use it as a reference. Why did they not provide the textual description of the ground truth image, which might be more reliable?
2. Could you please share the model accuracy across image types? Which type of image is hard to generate for the MLLMs?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents GenExam, a benchmark that evaluates text-to-image and multimodal models through exam-style drawing tasks across ten academic subjects. Each problem includes a ground-truth image and fine-grained scoring points for assessing semantic correctness and visual plausibility. The benchmark reveals that even top models perform poorly on such reasoning-intensive generation tasks, highlighting the gap between aesthetic generation and structured understanding.

### Strengths
1. Novel evaluation perspective. Framing text-to-image generation as exam-style reasoning is original and thought-provoking.
2. Technically rigorous construction. Prompts, reference images, and scoring points are carefully curated and validated.
3. Broad multimodal coverage. The benchmark spans multiple subjects and provides detailed, multi-dimensional evaluation.

### Weaknesses
1. Primarily a benchmark contribution with limited analytical depth regarding model reasoning behavior.
2. Certain evaluation metrics (e.g., strict vs. relaxed scores) lack theoretical grounding or comparative justification.
3. Reliance on GPT-5 as the sole evaluator raises reproducibility and bias concerns.

### Questions
1. How is scoring consistency ensured when using GPT-5 as an evaluator?
2. What design principles guided the selection of scoring points across subjects?
3. Have the authors considered human validation to support the reliability of automated scoring?

### Soundness
2

### Presentation
3

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
The paper introduces GenExam, a new benchmark for discipline-style drawing exams in text-to-image generation. It comprises 1,000 prompts across 10 subjects, each paired with a reference image and fine-grained “scoring points”. Evaluation uses an MLLM-as-a-judge to grade semantic correctness via per-item scoring points and visual plausibility, producing strict and relaxed scores. SOTA model only achieve ~12% strict score, while open-source models are near zero, indicating the task’s difficulty.

### Strengths
- The problem is clearly framed and well-motivated. 

- The strict/relaxed scheme formalizes both exactness and presentational quality; the rubric for spelling, readability, and logic is explicitly specified.

- The authors measure correlations between human judgments and their relaxed score for a human alignment check.

### Weaknesses
- The evaluation relies heavily on an LLM-as-judge framework, yet large language models themselves often struggle with the same skills being assessed here (fine-grained visual reasoning, spatial accuracy, and multi-attribute grounding). This raises concerns about the reliability and objectivity of the scoring process. Without evidence of cross-model agreement, calibration against human experts, or robustness to judging prompts, it’s unclear whether the reported results truly reflect model performance or simply the biases and limitations of the chosen LLM evaluator.

- The human study appears limited (e.g., 250 samples from one model). Broader sampling (across subjects/models) and inter-rater reliability details would strengthen the claim that relaxed scores align with humans.

- The error analysis is mainly qualitative without a structured taxonomy or statistical breakdown of error types across the dataset.

### Questions
- The rationale behind the chosen relaxed scoring weights (0.7, 0.1, 0.1, 0.1) is unclear. It would be helpful to justify these specific proportions and to examine whether alternative weighting schemes or aggregation methods (e.g., majority voting or learned weighting) might alter the relative model rankings or the study’s main conclusions.

- Can you provide detailed error breakdown (quantitative, not only qualitative) across common error types (e.g., semantic, spelling, logic, layout) for all models and content types?

### Soundness
3

### Presentation
3

### Contribution
3

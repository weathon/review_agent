# ImagenWorld: Stress-Testing Image Generation Models with Explainable Human Evaluation on Open-ended Real-World Tasks

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Advances in diffusion, autoregressive, and hybrid models have enabled high-quality image synthesis for tasks such as text-to-image, editing, and reference-guided composition. Yet, existing benchmarks remain limited, either focus on isolated tasks, cover only narrow domains, or provide opaque scores without explaining failure modes. We introduce \textbf{ImagenWorld}, a benchmark of 3.6K condition sets spanning six core tasks (generation and editing, with single or multiple references) and six topical domains (artworks, photorealistic images, information graphics, textual graphics, computer graphics, and screenshots). The benchmark is supported by 20K fine-grained human annotations and an explainable evaluation schema that tags localized object-level and segment-level errors, complementing automated VLM-based metrics. Our large-scale evaluation of 14 models yields several insights: (1) models typically struggle more in editing tasks than in generation tasks, especially in local edits. (2) models excel in artistic and photorealistic settings but struggle with symbolic and text-heavy domains such as screenshots and information graphics. (3) closed-source systems lead overall, while targeted data curation (e.g., Qwen-Image) narrows the gap in text-heavy cases. (4) modern VLM-based metrics achieve Kendall accuracies up to 0.79, approximating human ranking, but fall short of fine-grained, explainable error attribution. ImagenWorld provides both a rigorous benchmark and a diagnostic tool to advance robust image generation.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces ImagenWorld, a new large-scale benchmark designed to "stress-test" modern image generation and editing models.  ImagenWorld is built on a comprehensive foundation comprising:
- Six Core Tasks: It covers a wide range of functionalities, including text-guided image generation, single and multiple reference-guided generation, and the corresponding editing variations of these tasks.
- Six Topical Domains: The benchmark includes a diverse set of real-world use cases: artworks, photorealistic images, information graphics, textual graphics, computer graphics, and screenshots.
- Explainable Human Evaluation: At its core, ImagenWorld relies on a detailed human evaluation schema. Instead of just providing a single score, annotators identify and tag specific, localized errors at both the object level (e.g., a "missing chandelier") and the segment level (e.g., visual inconsistencies in a specific region of the image). This is supported by 3.6K condition sets and 20K fine-grained human annotations.

The authors conduct a large-scale study evaluating 14 different models, including closed-source systems like GPT-Image-1 and Gemini 2.0 Flash, as well as various open-source models. The key findings from this evaluation are:
- Models generally struggle more with editing tasks than generation, often failing by either completely regenerating the image or ignoring the edit instruction entirely.
- All models excel in artistic and photorealistic domains but perform poorly on text-heavy and symbolic domains like screenshots and information graphics. An exception is Qwen-Image, whose performance suggests that targeted data curation can mitigate this weakness.
- While closed-source models lead in overall performance, open-source models are competitive in standard text-to-image generation, indicating that progress in more complex editing and multimodal tasks requires more than just scaling.
- Modern Vision-Language Models (VLMs) used as automated judges achieve high correlation with human rankings but fail to provide the fine-grained, explainable error analysis that human annotators can.

### Strengths
1. The paper presents a valuable benchmark and evaluation protocol for image generation models. As the capabilities of the models are improving and we are moving to models that can perform several different tasks from text-to-image generation to multi-image reference image editing unifying evaluation is important.
2. The authors systematically evaluate several closed- and open-source models and draw conclusions on the current weaknesses of image generation models. They make sure to include different model families and measure performance for different tasks and with respect to different generation aspects (prompt relevance, content coherence, etc)
3. The authors assess the reliability of automatic judges by measuring correlation with human ratings and find that there is moderate to strong correlation with coarse-grained ratings per generation aspect.

### Weaknesses
1. One important contribution of the paper is the collection of fine-grained feedback on the image generation errors by human raters by annotating the object and segment errors in the images. These annotations are not being sufficiently utilized in the paper for either analysis or automatic judge improvement. The authors mention is Section 5.3 that "Taken together, the results suggest that VLM-based evaluation provides a reliable proxy for human annotation, particularly for assessing semantic alignment and visual appeal, while human judgment remains essential for identifying fine-grained details such as visual defects." without providing any analysis on the fine-grained collected data (they only present results and correlation on the coarse-grained ratings). Moreover, it would be great to show whether a subset of these data can be used as a few shot prompt or for fine-tuning in order to improve the capabilities of VLMs on evaluating fine-grained errors in image generation.

### Questions
1. Regarding to the weakness point mentioned above, how do you conclude that VLMs cannot provide more fine-grained details? Can this be improved by simply giving few-shot examples or fine-tuning the evaluator to create fine-grained criteria lists?

### Soundness
3

### Presentation
4

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
The paper introduces Imagen World, a large-scale benchmark designed to stress-test generative image models across a broad spectrum of 3,600 condition sets, covering six core tasks (generation and editing with varying reference guidance) and six topical domains. To address the limitations of opaque scalar metrics, the authors propose an explainable human evaluation schema, collecting 20,000 annotations that include both standard ratings and fine-grained object/segment-level error tags. The study evaluates 14 models, revealing that closed-source systems generally lead, particularly in complex editing tasks and text-heavy domains, while also benchmarking VLM-based automated metrics against these human judgments.

### Strengths
- The benchmark addresses a clear gap in current evaluation by unifying a diverse set of tasks (generation and editing) and real-world domains (e.g., Information Graphics, Screenshots) under a single standardized protocol.

- The proposed explainable evaluation schema is a logical attempt to move beyond simple scalar scores. By asking annotators to explicitly tag object and segment-level failures, the benchmark provides richer diagnostic information on why models fail.

- The authors provide a large-scale study of 14 varied models, offering a comprehensive view of the current capabilities of both unified state-of-the-art systems and task-specific baselines.

### Weaknesses
- The automated evaluation hinges entirely on a proprietary model (Gemini-2.5-Flash) accurately assessing four distinct criteria (Relevance, Aesthetics, Coherence, Artifacts) and performing object listing. This reliance on a specific, closed-source model acts as a severe reproducibility bottleneck. If the model version changes or becomes unavailable, the automated portion of the benchmark is fundamentally compromised.

- The paper lacks necessary statistical rigor for a benchmark standard. It presents mean scores without conducting statistical significance tests. Without these, it is hard to determine if the performance gaps between top-performing models are meaningful or simply noise.

- While the paper reports inter-rater agreement (Krippendorff’s alpha), it fails to adequately address the high variability observed (ranging from a low 0.52 to 0.80 in Table 3). With such high disagreement for certain models/tasks, it is difficult to gauge the actual reliability of the human ground truth in those areas.

- The paper fails to cite or compare against established "decompose-and-verify" metrics that are highly relevant to fine-grained evaluation. A key missing reference is Gecko [1]. 


[1] Wiles et al., Revisiting text-to-image evaluation with gecko: On metrics, prompts, and human ratings, ICLR 2025

### Questions
- Can you clarify exactly how the VLM judge was implemented? Did it evaluate all four dimensions in a single pass? Have you considered that this might introduce a bottleneck, and have you experimented with specialized VLM judges for each distinct criterion to improve correlation?

- In Table 4, the VLM significantly under-penalizes artifacts compared to humans (positive bias). Can you provide some empirical evidence or analysis explaining this discrepancy? Is it a fundamental limitation of using current VLMs that may have been trained on noisy data?

- How was the prompt template for the VLM judge (Appendix A.7) derived? Did you utilize a "gold set" of human-annotated examples to validate and optimize this prompt before deployment?

- Regarding the reported Krippendorff’s alpha in Table 3, why was the least inter-annotator agreement found for the overall best-performing model (GPT-Image-1 in TIG)? What does this imply about the subjectivity of evaluating higher-quality outputs?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ImagenWorld, a unified benchmark for image generation and editing that spans six task types (generation/editing with 0/1/multi references) and six topical domains, totaling 3.6K condition sets. It is supported by ~20K fine-grained human annotations and an explainable evaluation schema that tags object-level and segment-level issues, alongside automated VLM-based metrics. A large-scale study across 14 models reveals consistent trends: models struggle more with editing than generation, and with text-heavy/symbolic domains (e.g., screenshots, infographics), while targeted data curation can narrow gaps; VLM-as-judge correlates well with humans (Kendall up to 0.79) but lacks fine-grained error attribution. Overall, the work positions ImagenWorld as both a rigorous benchmark and a diagnostic tool for real-world image synthesis.

### Strengths
1. Substantially more “solid” than typical T2I-only benchmarks. Unlike prior works that center on narrow, compositional prompts, ImagenWorld covers six tasks × six domains with 3.6K condition sets and ~20K human annotations, and adds explainable labels (object-level and segment-level issue tags) that go beyond opaque scalar scores—this breadth and diagnostic depth should be highly valuable to the community.

2. Well-designed evaluation pipeline that bridges rigor and scalability. The study evaluates 14 diverse models under a single protocol and reports aligned human/VLM results (Kendall up to 0.79) while retaining human-annotated, localized error tags—striking a practical balance between scalable automatic scoring and explainable human diagnostics.

3. Clear, readable writing. The paper consistently defines evaluation criteria (Prompt Relevance, Aesthetic Quality, Content Coherence, Artifacts) and motivates them with concrete examples, which makes the methodology easy to follow.

### Weaknesses
The paper lacks a solution method showing how to leverage the benchmark’s diagnostics (e.g., a targeted data-curation or instruction-parsing baseline). Adding such a solution would make the benchmark more actionable with minimal extra work.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

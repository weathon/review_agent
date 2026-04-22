# WildSVG: Towards reliable SVG generation under Real-Word conditions

- Avg Score: 4.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 6

## Abstract
We introduce SVG extraction, the task of translating specific visual inputs into scalable vector graphics. Existing multimodal models such as StarVector achieve strong results when generating SVGs from clean renderings or textual descriptions, but they fall short in real-world scenarios where natural images introduce noise, clutter, and domain shifts. To address this gap, we extend StarVector’s capabilities toward robust vision-to-SVG translation in the wild. A central challenge in this direction is the lack of suitable benchmarks. To fill this need, we develop two complementary datasets: Natural WildSVG, consisting of real-world images paired with SVG annotations, and Synthetic WildSVG, which integrates complex and elaborate SVG designs into real-life scenarios to simulate challenging conditions. Together, these resources provide the first foundation for systematic benchmarking SVG extraction. Building on them, we benchmark StarVector and related models. Our study establishes SVG extraction as a new problem domain, introduces datasets and evaluation protocols for its study, taking initial steps toward extending multimodal LLMs to handle reliable SVG generation in complex, natural scenes.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces SVG extraction, the novel task of translating specific visual inputs (such as real-world images) into Scalable Vector Graphics (SVG). Existing multimodal LLMs (e.g., StarVector) perform well when generating SVGs from clean renderings or text but falter under noisy, cluttered, and domain-shifted natural images. To address this, the authors define and formalize SVG extraction as a research problem. Besides, they present WildSVG, the first benchmark dedicated to SVG extraction. They also devise new evaluation protocols and multi-metric analysis (L2, SSIM, LPIPS, DINO). Finally, this paper benchmark a range of state-of-the-art VLLMs on the task, including StarVector, GPT-4/5, Claude Opus, Gemini, and Qwen.
Their results highlight the performance ceiling of current models, the semantic vs. fidelity trade-off, and the increased difficulty of SVG extraction in natural settings. The work establishes a foundation for systematic research and development on reliable SVG generation from complex images.

### Strengths
1. This paper present the first benchmark dedicated to SVG extraction, comprising Natural WildSVG, focusing on real-world images paired with verified SVG annotations, and Synthetic WildSVG, focusing on natural images with synthetically embedded, complex SVGs.
2. They devise new evaluation protocols and multi-metric analysis. Besides, they benchmark several VLM on this task.
3. The paper outlines clear future directions and potential integration with multimodal LLM pipelines, encouraging further exploration and dataset expansion.

### Weaknesses
1. From my perspective, the proposed task lacks sufficient innovation.
2. Limited benchmark diversity: Expansion to more diverse SVG types (beyond logos, include pictograms, diagrams, UI elements) could clarify task boundaries and model strengths.
3. Lack of Editing-based SVG extraction: It is possible to first use image editing to extract the target object into raster image, them vectorize the raster image into SVG.
4. More Robust Metrics: this paper only consider visual similarity as the metric. However, code compactness and editablity are also important.

### Questions
1. Have you or do you plan to fine-tune VLLMs specifically on WildSVG?
2. How might you further extend the dataset into more SVG types, e.g. diagrams, UI elements?
3. Beyond the current metrics, have you considered human evaluations on SVG usefulness/quality, or downstream applicability (e.g., re-editing for designers)?

### Soundness
3

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
4

### Summary
This work introduces SVG extraction, a task for generating vector graphics from specific elements within real-world images. The authors contribute the WildSVG benchmark, comprising new natural and synthetic datasets, to facilitate research on this challenging problem. They benchmark a suite of modern VLLMs, revealing a significant performance gap and an interesting trade-off between the models' semantic understanding and aesthetic fidelity.

### Strengths
1. The claims are well backed by experiments. Using both one-step and two-step evaluation settings helps disentangle localization from vectorization capabilities. The choice of metrics, covering both pixel-level and semantic similarity, is comprehensive.

### Weaknesses
1. The presentation of the paper is extremely poor and looks like it is written in hurry.
2. The test sets are worryingly small. This raises serious questions about the statistical significance of the reported results and the reliability of the benchmark for distinguishing between top-performing models where score differences are marginal.
3. The authors astutely identify that most VLLMs cheat by using SVG text primitives instead of drawing shapes. However, the chosen raster-based metrics (DINO, LPIPS, etc.) fail to penalize this behavior and may even reward it, meaning the quantitative results do not fully reflect this important qualitative failure mode.

### Questions
1. Given the small test sets, can you provide confidence intervals or a statistical significance analysis for your results? It is difficult to assess whether the reported performance differences between models in Tables 4 and 5 are meaningful.
2. You suggest that StarVector's failure in the one-step setting is due to being overwhelmed by noise. Could this failure not also be a more fundamental architectural flaw stemming from its training regime, where text and image conditioning were learned separately, leading to weak prompt alignment?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses a novel task, SVG extraction that combines vision and structured graphics, which is timely given the rise of multimodal LLMs. The authors introduce WildSVG, the first benchmark for extracting vector graphics from real images. They construct two complementary datasets: Natural WildSVG and Synthetic WildSVG. The evaluation protocol is carefully designed with multiple metrics to capture different aspects of output quality. Overall, framing SVG extraction is a well-motivated problem with clear definitions.

### Strengths
1. The authors evaluate a wide range of state-of-the-art vision-language models (Qwen, Gemini, Claude, GPT, StarVector, GLM) on both Natural and Synthetic WildSVG test sets.

2. They use four complementary metrics (L2, SSIM for pixel fidelity; LPIPS, DINO for perceptual/semantic similarity), which is an appropriate choice to capture different aspects of the generated SVG.

### Weaknesses
1. How often did the VLLM-based SVG matching fail or produce incorrect logo–SVG pairs in Natural WildSVG? Were any manual checks done, and how sensitive are the results to these mismatches?

2. Can you clarify how the “focus prompt” is formulated and used? If the prompt is ambiguous or generic, how does it affect the model’s output?

3. Can you provide more details on the synthetic data creation? How diverse are the embedded SVG contexts (lighting, occlusion, styles)?

4. How do you ensure that the chosen pixel/semantic metrics correlate with true SVG fidelity?

### Questions
See weakness section.

### Soundness
2

### Presentation
3

### Contribution
2

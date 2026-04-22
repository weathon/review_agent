# VectorGym: A Multi-Task Benchmark for SVG Code Generation and Manipulation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 6, 2

## Abstract
We introduce VectorGym, a multi-task benchmark for evaluating Vision-Language Models (VLMs) on Scalable Vector Graphics (SVG) code generation and manipulation. VectorGym addresses the critical lack of challenging benchmarks aligned with real-world design workflows, specifically requiring mastery of complex primitives and multi-step edits. Our benchmark comprises four complementary tasks: the novel Sketch2SVG (VG-Sketch) conversion; a new SVG editing dataset (VG-Edit) involving higher-order primitives and semantic reasoning; and rigorous benchmarks for Text2SVG (VG-Text) and SVG captioning (VG-Cap). VectorGym derives particular value from expert human-authored SVG annotations across all tasks, ensuring a rigorous challenge. VectorGym also introduces a VLM-as-judge metric tailored for SVG generation, validated against human judgment. Our comprehensive evaluation of leading VLMs and our own GRPO-trained models reveals significant performance gaps, establishing VectorGym as a robust framework for advancing visual code generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces **VectorGym**, a comprehensive benchmark dataset for evaluating SVG-related tasks.  
VectorGym includes multiple tasks such as **sketch-to-SVG conversion, SVG editing, text-to-SVG generation, and SVG captioning**.  
The authors evaluate the performance of various **Vision-Language Models (VLMs)** on these datasets.  
Furthermore, they assess the results using multiple evaluation metrics, including **VLM-as-a-judge** approaches.

### Strengths
- 1. The paper proposes a human-annotated dataset for several **SVG generation tasks**.
- 2. The paper provides a detailed evaluation of the **VLM-as-a-judge** framework.
- 3. Multiple models are evaluated, and the characteristics of each task are analyzed.

### Weaknesses
- 1. Although the created dataset is divided into train / validation / test splits for evaluation, only zero-shot evaluation is performed, so the effect of using the data for fine-tuning remains unclear.
- 2. The paper mentions that an LLM was utilized to assist in drafting the Related Work section; however, the citation link for SVGEditBench directs to an entirely unrelated publication, indicating a potential instance of hallucination in the generated text. Moreover, it appears that the description in the paper actually refers to SVGEditBench2, yet in Table 1 it is simply cited as SVGEditBench, which reduces the accuracy and credibility.
- 3. The procedure for constructing the SVG Edit dataset is unclear.
    - 3.1. When humans performed the edits, did they use a drawing tool to modify the images, or did they manually rewrite the SVG code directly?
   - 3.2. Section A.1.4 presents examples of “Required complex edits”, but how were these criteria defined? What other perspectives were considered?
    - 3.3. In SVGEditBench2, edit instructions are generated based on differences between similar images, rather than simple rule-based edits, resulting in more complex instructions. Compared to such existing approaches, does the proposed dataset contain more complex and diverse edit instructions?

### Questions
- 1. Regarding the correlation with human evaluation In Table 2.
    - 1.1 why are the results separated between generated samples and ground truth?
    - 1.2. For the SVG edit task, why does the correlation become high for generated samples but drop significantly for ground truth? Does this mean that VLMs assign high scores even when the outputs are correctly generated? If that is the case, wouldn’t it imply that the metric fails to properly distinguish correct generations, thus questioning its reliability as a trustworthy evaluation measure?
    - 1.3 In the Sketch2SVG task, why does the correlation appear lower compared to the other tasks?
- 2. Benchmarks such as SVGenius also provide comprehensive evaluations across multiple tasks, including editing, understanding, and generation. In comparison to such prior benchmarks, what are the main differences or novel contributions of this paper? Are there distinctions beyond the inclusion of sketch-based data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a new human annotated multi-task benchmark for SVG code generation and manipulation, covering Sketch2SVG, instruction-guided SVG editing, Text2SVG, and SVG captioning. The authors curate 7,000 real-world SVGs, collect human sketches, and detailed captions, and evaluate a wide range of proprietary and open-source VLMs.

### Strengths
1. This paper seems to be the first to combine Sketch, Edit, Text and Captioning tasks, marking for its novelty.
2. This paper provides extensive evaluations of different tasks using multiple open source models covering major evaluation metrics.

### Weaknesses
1. The evaluation selects the best of 5 sampled outputs using the same VLM-judge metric. That creates an evaluation bias and may overstate real single-sample performance.
2. The VLM-judge validation uses a validation set of only 50 samples per task to compute Pearson correlations with human annotators, which seems to be too small for robust judge selection, given the diversity of SVGs and edit types.

### Questions
1. How sensitive is judge selection to the 50-sample validation set?
2. How do rankings change if you (a) use a single deterministic sample, (b) use median/mean over n samples, or (c) use oracle selection based on human scores?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces VectorGym, a multi-task benchmark designed to evaluate Vision-Language Models (VLMs) on Scalable Vector Graphics (SVG) generation and manipulation. It spans four tasks: Sketch2SVG, SVG Editing, Text2SVG, and SVG Captioning, supported by a 7k-sample human-annotated dataset. The authors propose a VLM-as-judge metric validated via human correlation and benchmark both proprietary and open-source models, finding GPT-5 and Claude-4 to perform best.

### Strengths
- Addresses a real evaluation gap in SVG generation and editing with a well-motivated multi-task design.

- Uses human-authored, complex edits and sketches rather than synthetic data.

- Extensive model coverage with consistent zero-shot evaluation across tasks.

### Weaknesses
- VLM-as-judge is not novel; prior works in vision-language evaluation (e.g., LLaVA-Bench, EvalAlign) already employ this approach. Framing it as a key contribution is overstated.
- The term “complex human annotations” is vague and not operationalized--no quantification of complexity, annotator agreement, or examples showing what differentiates them from existing datasets.
- Table 1 is misleading--it should explicitly mark whether prior datasets included any human annotation. The current comparison may overstate novelty.
- All four tasks (Text2SVG, Sketch2SVG, Editing, Captioning) have been studied individually in previous benchmarks; the main novelty is unification, not new task design.
- Sketch2SVG evaluation is questionable: sketches lack full color or geometric precision, but evaluation is done against SVGs which are visually richer. This biases visual-similarity metrics that penalize missing colors or fine details absent in the input.
- Circular evaluation flaw: best-of-n generation and selection uses the same VLM-as-judge for scoring, biasing results.
- Reproducibility concerns: heavy reliance on proprietary APIs (GPT-4o, GPT-5) without open-source judge substitute.
- No statistical testing or confidence intervals on leaderboard results.
- VLLM as a Judge prompt may not be specific enough since score thresholds are defined in ranges: while interpreting results, its hard to make a distinction between a score 7 or 8 since both are supposed to represent "Mostyly accurate and complete, minor issues in detail or quality, clear and visually appealing". Self consistency of these 
- Potential dataset contamination: LLMs (Qwen2-VL) used for caption validation may overlap with evaluated models. Human validation of the captions should have been supported.

### Questions
- How are “complex human annotations” defined and measured?
- In Table 2: Text2SVG and SVG Edit: Claude 4 Sonnet and Gpt4o have the exact same values upto 3 decimal places, as reported. Are you sure these are valid and not a mistake?
- Following up from the previous question, can the authors provide variance or statistical significance for differences <0.2 in judge scores?
-What exact filtering thresholds (token count, color entropy) were applied during curation?
- Can you include examples of human validation vs LLMaaJ in the appendix? Also attach metrics with all qualitative examples to give an idea of how the metrics are significant.

Minor concerns:
Stary characters like upside down ? in line: 385

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
This paper constructs various benchmarks to evaluate the ability of VLMs to understand SVG data and reports their performance across different models. Specifically, the benchmarks are organized into four tasks: Sketch (creating SVGs from drawings), Edit (editing SVGs), Text (generating SVGs from text), and Cap (captioning SVGs). Evaluations are reported for both open-source and proprietary models.

### Strengths
The purpose of this paper is clear. It is meaningful to investigate how well LLMs/VLMs can understand the SVG data format. The authors specifically constructed a large-scale dataset and conducted this investigation.

It's also worth noting that various experiments were conducted and evaluations performed for both open-source and proprietary models.

### Weaknesses
This paper has the following weaknesses.

## Comparison with Previous Benchmarks

The paper lacks discussion on how it differs from the various benchmarks shown in Table 1. As indicated in Table 1, several benchmarks already exist for evaluating SVG understanding. However, the paper does not discuss what distinguishes its benchmarks from the existing ones, nor does it compare its conclusions and qualitative discussions with those of prior benchmark studies. The absence of such comparisons is problematic.

In the rightmost column of Table 1, previous studies are marked as lacking "Complex Human Annotations," but the meaning of "Complex" is not explained. It is unclear in what sense the annotations in this paper are considered "complex" compared to those in previous works.

## Citation of Nonexistent Papers

Among the papers listed in Table 1, there are significant bibliographic errors in two of them:

- VGBench is cited in this paper as the Findings of EMNLP 2024 by Xia+, which is incorrect. The correct citation is the main conference paper at EMNLP 2024 by Zou+, with a different title. https://aclanthology.org/2024.emnlp-main.213/
- SVGEditBench is attributed to Shu+ in this paper, which is also incorrect. The correct authors are Nishina+, with a different title and arXiv link. https://arxiv.org/abs/2502.19453

In particular, for SVGEditBench, "Changyue Shu" is listed as the author. However, no such person exists, according to a Google search. Therefore, this error is more than a simple BibTeX mistake.

Misidentifying the most relevant comparative works is a serious flaw. It suggests that the authors may not have properly reviewed prior research, despite claiming that earlier works lacked "Complex Human Annotations."

## Insufficient Description of the Proposed Method

The section describing the proposed method (Sec. 3) is only one page long, and everything from page 4 onward is devoted to experiments. There are no details about how the task definitions were established or how the dataset was constructed. While the length of a section does not determine a paper's value, the lack of methodological detail makes the paper feel more like a technical report than a scientific research paper.

## Legal Issues Regarding SVG Data

The authors do not address the legal issues surrounding the SVG data. In Section 3.2, it is stated that SVG data were obtained from the SVG-Stack dataset, and in Section 6 ("Ethics Statement"), it is claimed that licensing was handled appropriately. However, there is no explanation of how this was done. Are the SVGs used distributed under licenses such as MIT or CC that allow redistribution? Moreover, do the authors intend to release the dataset? These points are insufficiently discussed.

### Questions
Any comments on legal issues?

### Soundness
2

### Presentation
1

### Contribution
2

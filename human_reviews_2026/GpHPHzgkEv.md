# SoM-1K: A Thousand-Problem Benchmark Dataset for Strength of Materials

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Foundation models have shown remarkable capabilities in various domains, but their performance on complex, multimodal engineering problems remains largely unexplored. We introduce SoM-1K, the first large-scale multimodal benchmark dataset dedicated to evaluating foundation models on problems in the strength of materials (SoM). The dataset, which contains 1,065 annotated SoM problems, mirrors real-world engineering tasks by including both textual problem statements and schematic diagrams. Due to the limited capabilities of current foundation models in understanding complicated visual information, we propose a novel prompting strategy called Descriptions of Images (DoI), which provides rigorous expert-generated text descriptions of the visual diagrams as the context. We evaluate eight representative foundation models, including both large language models (LLMs) and vision language models (VLMs). Our results show that current foundation models struggle significantly with these engineering problems, with the best-performing model achieving only 56.6\% accuracy. Interestingly, we found that LLMs, when provided with DoI, often outperform VLMs provided with visual diagrams. A detailed error analysis reveals that DoI plays a crucial role in mitigating visual misinterpretation errors, suggesting that accurate text-based descriptions can be more effective than direct image input for current foundation models. This work establishes a rigorous benchmark for engineering AI and highlights a critical need for developing more robust multimodal reasoning capabilities in foundation models, particularly in scientific and engineering contexts.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SoM-1K, the first large-scale multimodal benchmark designed to evaluate foundation models on Strength of Materials (SoM) problems that combine textual and schematic information. Experimental results show that even top-performing models achieve only 56.6% accuracy, with LLMs using Descriptions of Images (DoI) outperforming vision-language models—highlighting the need for more advanced multimodal reasoning capabilities in engineering AI.

### Strengths
1. Evaluating modern MLLMs on diverse engineering problems is an important and timely contribution.

2. Novelty: SoM-1K is the first domain-specific multimodal benchmark for Strength of Materials.

3. The paper is clearly written and easy to follow.

### Weaknesses
1. Human-based evaluation: The evaluation is conducted manually by human annotators rather than automated tools, which may lead to inconsistencies across models or researchers. Furthermore, the need for human evaluators limits the reproducibility and scalability of the benchmark for future studies.

2. Limited model coverage: The paper only evaluates relatively outdated models. More recent models such as o3, Gemini 2.5 Pro, Qwen 2.5-VL series, and InternVL 3 series should be included to provide a stronger and more relevant comparison.

3. Poor figure readability: The figures are difficult to interpret and compare across models and prompts. Presenting the main results in a tabular format would improve readability and facilitate comparison.

4. Weak model baseline in Section 4.3: The analysis and findings in Section 4.3 are based on less capable models. Including results from stronger, state-of-the-art models (e.g., o3, Gemini 2.5 Pro) would make the conclusions more convincing.

5. Questionable novelty of DoI: The proposed Description of Images (DoI) approach appears conceptually similar to prior “describe-then-reason” prompting methods. The key difference—using expert-generated descriptions instead of model-generated ones—limits real-world applicability, since such expert annotations are not available during inference.

6. Unclear reasoning claim: The statement “we propose and validate the use of text-based diagram descriptions (DoI) as an effective prompting strategy to reduce reasoning errors” seems inconsistent. Since DoI is explicitly defined as containing no reasoning (only perceptual content), it likely improves perception accuracy rather than reasoning ability. The paper should clarify this distinction.

### Questions
Refer to weaknesses

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
This paper introduces SoM-1K, a multimodal benchmark for evaluating foundation models on Strength of Materials (SoM) problems that combine text, equations, and engineering diagrams. The dataset contains 1,065 annotated problems covering axial loading, torsion, and bending, each paired with expert-generated textual diagram descriptions (DoI). The authors benchmark eight large models (LLMs and VLMs) and find that text-based DoI inputs significantly outperform direct visual inputs, highlighting models’ limitations in diagram interpretation. The dataset is well-constructed and addresses a gap in engineering-focused evaluation, though the contribution is primarily dataset-centric with limited methodological novelty.

### Strengths
Well-curated and domain-authentic dataset; clear evaluation design isolating visual vs textual reasoning; provides an insightful observation that current VLMs fail on symbolic, structured diagrams while LLMs handle their textualized forms better.

### Weaknesses
1. Lack of benchmark validation and answerability calibration.
The paper does not report item difficulty, discrimination, or inter-annotator consistency, making it unclear whether low model scores stem from reasoning limitations or inherently ambiguous or overly difficult problems.
2. Questionable equivalence between visual and textual conditions.
The DoI inputs often encode explicit geometric and boundary relations that are not directly visible to models, making PS+DoI semantically richer than PS+I. This weakens the claim that DoI merely textualizes visual information.
3. Absence of human baselines.
The study relies entirely on expert grading without reporting any human performance or consistency metrics, preventing interpretation of model accuracy (e.g., 56%) relative to realistic human-level baselines.

### Questions
1. Have you validated per-item difficulty or answerability (e.g., via item–response analysis or human–model agreement) to ensure benchmark reliability?
2. How do you guarantee that DoI does not introduce extra semantic information beyond the schematic itself? Could you quantify or control for information density differences?
3. What is the human performance level (e.g., engineering students, instructors) on SoM-1K, and how does it compare to model results?

### Soundness
2

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
2

### Summary
The paper SoM-1K: A Thousand-Problem Benchmark for Material Strength presents the first large-scale multimodal dataset designed to evaluate large language and vision-language models for authentic engineering problem solving.  The benchmark includes 1065 textbook-sourced problems covering axial loading, torsion, bending, and composite scenarios, each accompanied by a problem statement, schematic diagram, description of image (DoI), and expert ground truth solution.  Eight cutting-edge models (such as GPT-4o, Claude 3.7, and Qwen-Plus) are tested using various prompt modalities (text + image + DoI).  The results show that even the best model achieves only about 56% accuracy and that textual DoI inputs consistently outperform direct visual inputs, revealing fundamental flaws in current MLLMs' ability to extract and reason from engineering schematics.  The paper presents a rigorously curated dataset, a novel DoI prompting paradigm, and detailed error analysis, establishing SoM-1K as a foundational benchmark for AI-for-mechanics and multimodal scientific reasoning research.

### Strengths
- Authentic and Domain-Grounded Benchmarks
The dataset was created directly from real Strength of Materials (SoM) textbooks and competition problems, making it significantly more realistic than synthetic physics or symbolic datasets.  The inclusion of various problem types - axial, torsion, bending, and combined load cases, addresses the key conceptual foundations taught in mechanics courses.

-Well-designed Multimodal Structure
Each item has four components: Problem Statement (PS), Image (I), Description of Image (DoI), and Ground Truth (GT), all of which are tightly aligned and manually verified.  This organization enables the precise identification of where multimodal reasoning fails (visual extraction versus symbolic reasoning).

-Use "Description of Image" (DoI) Input
Introducing DoI as a textual connection between the schematic and reasoning steps is a valuable innovation.  Empirically, DoI prompts significantly improve accuracy when compared to direct visual input, indicating that translating visual information into structured text is still the most effective pathway for current models.

-Comprehensive Model Evaluation
The benchmark compares several leading models (GPT-4o, Claude 3.7, Gemini 2.5, Qwen-VL 2.5, and so on) using three consistent prompt settings (PS + I, PS + DoI, and PS + I + DoI).  This breadth allows for a balanced comparison of text-only, vision-language, and hybrid reasoning approaches.

### Weaknesses
-No human or expert baseline.
Without reporting human accuracy, it is difficult to determine whether 56% model performance is near-human or significantly below it.  A reference from engineering students or instructors would help to contextualize the difficulty.

- Evaluation Metric
The binary correctness score does not account for partial reasoning quality, such as correct free-body setup versus numerical slip. A step-by-step or partial-credit evaluation could be more effective in measuring reasoning depth.

- Scalability of DOI Annotation
The DoI layer currently relies on manual verification, which may limit its scalability to larger or cross-domain datasets. The paper could have gone into greater detail about how to automate or standardize this process.

### Questions
Annotation Quality - How many annotators verified the DoI descriptions, and how was inter-annotator consistency assessed?

Human Baseline - Have you tested the SoM-1K with experts to determine typical human accuracy?

DoI Generation - Were the DoIs generated by a machine and then edited, or were they entirely written by humans?  Could future datasets rely on automated generation?

Practical Impact - Beyond benchmarking, how do you see SoM-1K being used? For model fine-tuning, educational feedback, or AI-aided design?

Prompt Fairness - Were all models evaluated using the same context length, temperature, and image resolution to ensure an accurate comparison?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces SoM-1K, a 1,065-problem multimodal benchmark for strength of materials that pairs textual statements with schematic diagrams and descriptions of images (DoI) to textualize visuals. It evaluates multiple LLM/VLM models under three prompting regimes with majority voting over five generations and expert binary grading.

### Strengths
- Evaluation uses multiple prompts per item, majority vote, and human grading
- Results show that DoI reduces visual misinterpretation (extraction) errors across VLMs and current models are more reliable with precise text than raw schematics.

### Weaknesses
- The paper fails to cite or compare against existing multimodal benchmarks such as MMMU [1], which is broader and more comprehensive, covering multiple academic subjects and modalities. Without such comparison, the novelty and relative contribution of SoM-1K remain unclear. The authors should explicitly position their work within the broader landscape of multimodal reasoning benchmarks, clarifying what unique capabilities or insights it offers beyond existing large-scale datasets.

- The paper does not include a human performance baseline (e.g., results from students or experts solving the same problems). The absence of such a reference makes it difficult to assess how large the gap remains between current models and expert-level reasoning.

- Figure 4 is hard to interpret as all models, prompting strategies, and open/closed-source systems are plotted together, making trends difficult to distinguish. Separating results by model type (open vs. closed) and prompting strategy would improve clarity and make performance comparisons more meaningful.

- Relying solely on accuracy may mask variation in partial-credit reasoning quality. The paper would benefit from per-subcategory breakdowns (e.g., determinate vs. indeterminate, frames vs. beams).

- The evaluation of VLMs lacks sufficient details regarding image preprocessing and resolution settings, which can critically affect visual reasoning performance. The authors should specify how diagrams were rendered, scaled, and fed into each model. Moreover, it would strengthen the study to explore whether structured diagram parsing (e.g., converting schematic images into vectorized primitives or symbolic graph representations) can reduce the performance gap between the PS+I (image-based) and PS+DoI (textualized diagram) conditions. Such analysis would help clarify whether the observed advantage of DoI stems from richer semantics or simply from preprocessing limitations in visual input.


[1] Yue, Xiang, et al. "Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
see weaknesses above

### Soundness
2

### Presentation
2

### Contribution
2

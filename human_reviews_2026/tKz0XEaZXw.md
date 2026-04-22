# Co-EditBench: Human-Aligned Benchmark for Instruction-Based Image Editing with Multi-Dimensional Assessment

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Multimodal large language models (MLLMs) have made significant progress in instruction-guided image editing; however, comprehensively evaluating them in a way that aligns with human judgment remains a considerable challenge. Existing benchmarks often exhibit obvious limitations, including restricted editing types, limited evaluation dimensions, coarse perception of image details, and systematic deviation from subjective aesthetics. To overcome these issues, we proposed a more comprehensive evaluation benchmark, Co-EditBench, for human-aligned evaluation. First, we constructed a diagnostic dataset by crowd-sourcing, to obtain high-resolution, real-world image-instruction pairs covering 16 editing types. Then, to enable a fine-grained and consistent assessment, we define 11 novel evaluation dimensions that dissect “AI artifacts” into traceable visual pathologies. Additionally, we propose a comprehensive automated evaluation pipeline Co-EditEval that leverages multi-dimensional evaluators and a meticulously designed Chain of Thought for contextualized visual reasoning. Extensive experiments demonstrate that Co-EditBench provides a more reliable and nuanced evaluation than existing benchmarks, achieving a significant correlation with human judgments.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents **Co-EditBench**, a benchmark for evaluating multimodal large language models (MLLMs) in instruction-guided image editing. It addresses limitations in existing benchmarks by providing a diagnostic dataset with 16 editing types, 11 fine-grained evaluation dimensions, and an automated pipeline (**Co-EditEval**) leveraging contextualized visual reasoning. Experiments show Co-EditBench aligns strongly with human judgment, offering more reliable and nuanced evaluations.

### Strengths
1. **Comprehensive Evaluation Benchmark**: Co-EditBench introduces a diagnostic dataset covering 16 real-world editing types and defines 11 fine-grained evaluation dimensions, addressing gaps in existing benchmarks and enabling nuanced assessments.  

2. **Human-Aligned Evaluation**: The benchmark strongly correlates with human judgments, ensuring evaluations reflect subjective aesthetics and detailed visual perception.

### Weaknesses
1. **Incremental Evaluation Approach**: The evaluation method relies on weighted scoring based on existing feature similarity metrics and interpretable Chain-of-Thought reasoning. However, this approach feels incremental, as similar methodologies have been adopted by prior benchmarks like ImageEdit, Wise, and GenEval. The contribution in evaluation design is overstated and lacks substantial novelty.  

2. **Regression to Outdated Metrics**: The inclusion of CLIP-based feature similarity metrics in the evaluation pipeline is concerning, as these metrics have been increasingly abandoned by recent works due to their lack of accuracy and reliability. While interpretable CoT-based evaluation is promising, relying on CLIP undermines the consistency and precision of the evaluation method.  

3. **Missed Opportunity for RL-based Reward Models**: Instead of using feature-based similarity metrics, constructing RL-based reward models to validate multiple evaluation dimensions would provide a more scalable and interpretable solution. Such an approach would allow for checklist-based evaluations that are both flexible and robust, better aligned with current advancements in evaluation methodologies.

### Questions
The absence of RL-based reward models for explainability and verification in this work raises concerns about the contribution of its evaluation methodology. RL-based reward models provide a modern and promising approach for validating image edits by offering both scalability and interpretability. Their ability to dynamically adapt to evaluation criteria ensures alignment with human reasoning and increases robustness across diverse editing tasks.

### Soundness
2

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
3

### Summary
This paper introduces Co-EditBench, a new benchmark for instruction-based image editing that features high-quality, manually verified masks distinguishing edited and non-edited regions. The central contribution is Co-EditEval, an automated evaluation pipeline designed to overcome the poor human alignment of existing metrics. Its key innovation lies in a Tailored Multi-dimensional Evaluation (TME) framework, which adopts a hybrid strategy: employing a CoT-guided MLLM for semantic evaluation and leveraging specialized perceptual metrics for fidelity assessment. Importantly, Co-EditEval incorporates region-aware fidelity computation based on the provided masks and aggregates 11-dimensional evaluation scores under a Completion-Guided principle. Empirical results demonstrate that the pipeline achieves a high Spearman correlation with human judgments, substantially outperforming previous benchmarks.

### Strengths
1. This paper integrates semantic reasoning and fidelity evaluation, with mask-based region awareness improving over global metrics.
2. The strong 0.889 SROC correlation and clear ablation results validate the necessity of both CoT and TME components.

### Weaknesses
The main concern lies in the CoT-based evaluation strategy, which raises several issues:
1. The evaluation relies on a closed-source model, i.e., Gemini 2.5-Pro, with CoT, which may be cost-prohibitive for other researchers, as it often requires extensive API usage. This could limit the benchmark’s accessibility and usage for other researchers.
2. The computational and time costs of CoT-based evaluation are unclear. For example, how long would it take to evaluate 1,000 images?
3. How does the pipeline handle hallucinations, given that CoT outputs are not always accurate, especially for fine-grained edits?

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
3

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
This paper introduces Co-EditBench, a new benchmark for evaluating instruction-based image editing models. It addresses key limitations of existing benchmarks, such as limited editing types, few evaluation dimensions, and weak alignment with human perception. The authors build a dataset of over 1,100 high-resolution image–instruction pairs covering 16 editing types, each with high-quality masks to separate edited and non-edited regions. They define 11 evaluation dimensions grouped into four areas: edit completeness, image quality, non-edited preservation, and identity preservation. The paper also proposes Co-EditEval, an automated evaluation pipeline that uses multiple evaluators (including MLLMs and similarity models) with a Chain-of-Thought (CoT) prompting strategy. Experiments on 25 recent image editing models show that Co-EditBench correlates better with human judgment than previous benchmarks like ImgEdit and GEdit.

### Strengths
1. Comprehensive benchmark with rich diversity across editing types and evaluation dimensions.

2. Strong motivation and clear identification of limitations in previous benchmarks.

3. Methodologically sound data collection and annotation process, including human verification.

4. Multi-dimensional evaluation design reflects real-world human judgment.

5. Detailed experimental validation, including large-scale comparison and ablation studies.

6. Good correlation between Co-EditEval scores and human ratings, demonstrating practical relevance.

### Weaknesses
1. The paper is somewhat heavy in detail; simplifying explanations in the methods section could improve readability.

2. Although the benchmark is extensive, the dataset’s accessibility and licensing terms are not fully clear (e.g., how others can use the crowd-sourced images).

3. The evaluation pipeline relies on commercial MLLMs like Gemini 2.5-Pro and GPT-4o, which could raise reproducibility concerns.

4. More discussion on potential biases in the crowd-sourced data and MLLM-based evaluators would strengthen the ethical transparency.

### Questions
1. Will Co-EditBench and Co-EditEval be released publicly, and under what license?

2. How consistent are the results when using different MLLM evaluators (e.g., GPT-4o vs. Gemini vs. Claude)? Can the evaluation benefit by using an ensemble and averaging the results? 

3. Did you consider the computational cost of the full evaluation pipeline? Could a lightweight version be used for faster benchmarking?

4. Are there any plans to expand the benchmark to video editing or 3D scenes?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper targets limitations of existing benchmarks for instruction-based image editing, including restricted editing coverage, limited evaluation dimensions, coarse perceptual alignment, and deviation from human preference. 
 
The authors propose Co-EditBench, a human-aligned benchmark, and introduce an automated evaluation pipeline Co-EditEval.
 
Key contributions include:
1. A diagnostic dataset built via crowdsourcing, covering 16 editing types across images and text instructions.
2. Definition of 11 fine-grained evaluation dimensions to diagnose “AI artifacts” based on edit completeness, image quality, non-edit preservation, and identity preservation.
3. A multi-dimensional automated evaluation framework leveraging a completion-guided principle based curated aggregation, ensuring secondary metrics cannot exceed edit-completeness performance.
4. Incorporation of chain-of-thought-based reasoning to improve contextual evaluation.

### Strengths
1. Fine-grained evaluation dimensions: 

    Defines 11 sub-metrics grouped into key categories (edit completeness, image quality, non-edit preservation, identity).

2. Completion-Guided Principle: 

    Caps secondary scores based on the primary edit-completeness score to prevent superficial improvements from inflating scores.

3. Human alignment verified: 

    Human evaluation by 34 annotators demonstrates a correlation between Co-EditEval and human judgment.

4. Comprehensive benchmark results: 

    Evaluates 25 SOTA models, revealing widespread weaknesses under a rigorous evaluation setup.

### Weaknesses
1. Metric weighting is partly subjective: 

    Although weights are defined, the rationale is neither well supported nor ablated.

2. Generality concerns: 

    How Co-EditEval performs under unseen editing types is not fully explored. See question 2.

### Questions
1. Customized generation:

    Customized generation often involves significant redrawing. In this case, how should the mask be selected? Should the entire image be selected as a mask or...?

2. In-context editing can involve reference images, implicit style transfer, or semantic relation changes between entities (e.g., “Match lighting with the reference image”).
 
    Can Co-EditEval evaluate in-context editing scenarios where the notion of an edit region is not spatially well-defined?
 
    Since IC-Edit [1] and RelationAdapter [2] target this type of in-context image editing, including them would help contextualize your benchmark’s applicability.

    [1] Enabling Instructional Image Editing with In-Context Generation in Large Scale Diffusion Transformer

    [2] Learning and Transferring Visual Relation with Diffusion Transformers

### Soundness
3

### Presentation
3

### Contribution
2

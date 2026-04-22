# Factuality Matters: When Image Generation and Editing Meet Structured Visuals

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
While modern visual generation models excel at creating aesthetically pleasing natural images, they struggle with producing or editing structured visuals like charts, diagrams, and mathematical figures, which demand composition planning, text rendering, and multimodal reasoning for factual fidelity. To address this, we present the first comprehensive, systematic investigation of this domain, encompassing data construction, model training, and an evaluation benchmark. First, we construct a large-scale dataset of 1.3 million high-quality structured image pairs derived from executable drawing programs and augmented with chain-of-thought reasoning annotations. Building on it, we train a unified model that integrates a VLM with FLUX.1 Kontext via a lightweight connector for enhanced multimodal understanding. A three-stage training curriculum enables progressive feature alignment, knowledge infusion, and reasoning-augmented generation, further boosted by an external reasoner at inference time. Finally, we introduce StructBench, a novel benchmark for generation and editing with over 1,700 challenging instances, and an accompanying evaluation metric, StructScore, which employs a multi-round Q\&A protocol to assess fine-grained factual accuracy. Evaluations of 15 models reveal that even leading closed-source systems remain far from satisfactory. Our model attains strong editing performance, and inference-time reasoning yields consistent gains across diverse architectures. By releasing the dataset, model, and benchmark, we aim to advance unified multimodal foundations for structured visuals.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper first introduces a dataset for structured image editing, then propose a benchmark built from this dataset. It also trained a model using the dataset and evaluate on the benchmark.

### Strengths
1. Used programs to generate image, which makes it more robust and controlled.  
2. The attempt to show alignment between human scores and benchmark metrics helps validate whether automated evaluation aligns with human judgment.

Overall, the data generation process seems reasonable for producing a large-scale benchmark, as the authors demonstrate that models trained from this pipeline show good performance both on this benchmark and on another relevant benchmarks.

### Weaknesses
1. The training pipeline can benefit from more ablation analysis. For example, what is the improvement when training with and without Stage 3 ? The choice of three stages is intuitive, but it requires more empirical evidence to justify the curriculum design. In addition, since the authors incorporate another dataset outside of their own dataset in stage 3, an ablation should be conducted to clearly show the effect of this addition, i.e. whether the improvement comes from the ChatGPT generated CoT or from the dataset itself.

2. The variation in evaluation results remains quite large with a different LLM(when comparing tables in the appendix produced by Qwen versus those in the main paragraph), even though the trends are similar. This is my main concern for the paper. Especially for a benchmark, if the score or model ranking can change depending on which VLM/LLM backend is used for scoring, it becomes difficult to trust this measure of progress. This subjectivity potentially suggests that the evaluation protocol may need further stabilization.

### Questions
Please see weakness

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a large-scale dataset tackling the structured visuals editing problem, along with a novel benchmark, StructBench, and metric StructScore to evaluate the quality of edits. The proposed dataset, with the three-stage training paradigm introduced, helps achieve state-of-the-art performance in the StructBench benchmark.

### Strengths
1. The proposed large-scale dataset contains image pairs and respective drawing programs and editing instructions, which are useful for future advancements in text-to-structured visual generation or editing. With the drawing program info, it can also be beneficial to text-to-vector-based (such as SVG) diagram generation by simply changing the output format for image rendering.

2. The proposed StructBench, along with the novel metric, StructScore, is superior to traditional editing benchmarks with metrics such as PSNR, as demonstrated by the high correlation with human preference. The addition of Qwen-based evaluator also rules out the reproducibility concerns of closed-source models such as GPT-5.

3. The three-stage training approach in the paper effectively bridges Qwen2.5-VL with FLUX.1, achieving state-of-the-art performance on the proposed benchmark, while maintaining competitive general editing capabilities.

### Weaknesses
- Some minor unclear points in the paper structure. In L365, there are mentions of StructEditBench and StructT2IBench. In section 3, only StructBench was introduced. I assume the Q&A construction and metrics around L244 are applied to both editing and T2I settings, but it should be clearly explained in this section.

### Questions
I am happy with the paper overall. Some technical details I would like to ask

- In Appendix A.2, only part of the training hyperparameters were provided. It would be helpful to include additional info such as GPU hours, number of iterations, LR scheduling, etc., given that the three-stage training involves fairly large datasets such as FLUX-Reason-6M and the one proposed in this paper.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper targets structured image generation/editing (charts, math/graph diagrams, tables, scientific schematics), where current T2I/VL models often produce plausible but factually wrong images. The authors build a 1.3M code-aligned dataset with paired instructions and CoT, plug Qwen-VL into FLUX.1-Kontext via a lightweight connector, train in three stages, and propose StructBench + StructScore to evaluate fine-grained factuality. They report strong gains over open models and show that adding explicit reasoning helps other models too

### Strengths
- Well-motivated task: clearly shows why structured visuals are different from “pretty pictures” and why factuality/layout need a dedicated treatment.
- Clean data pipeline: executable graphics → instruction + code edit - re-render - filtered, so supervision is tight and verifiable.
- Useful benchmark: StructBench/StructScore gives a more faithful measure than CLIP-style metrics and correlates with human judgment.
- Clarity and Comprehensiveness: The paper is exceptionally well-written and structured. The problem, contributions, and methodology are articulated with clarity.

### Weaknesses
- Dataset Domain and Generalizability: The reliance on executable programs for data generation may introduce a significant domain gap. It is questionable whether this dataset fully captures the diversity, noise, and "messiness" of structured visuals found in real-world sources

### Questions
- Have you considered evaluator bias? In StructScore, if the VLM used to ask questions and the VLM used to answer them belong to the same model family as the model being evaluated, how do you avoid evaluation bias or overfitting to the evaluator’s visual priors?

- Have you considered how stable the automatic metric is across different model versions? Could small changes in the judge cause the model rankings to change?

### Soundness
3

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
3

### Summary
This paper tackles a point in current image generation: models are great at making artistic "vibes" but absolutely terrible when you need them to be factually accurate, like drawing a specific chart, a math diagram, or a table.
To fix this, the authors didn't scrape the web; they generated a huge dataset (1.3M pairs) by running actual code (like Python/LaTeX), ensuring perfect ground truth. They then trained a FLUX-based model using a three-stage curriculum designed to make the model "think" more about structure before generating. They also had StructBench, a benchmark for these tasks. Seems like this benchmark is difficult.

### Strengths
Tackles an under-researched area where standard aesthetic-focused models fail precisely because high factual accuracy (text rendering, exact layout) is required.


Good idea: Utilizing programmatically generated images (via code) creates a verifiable and noise-free ground truth, superior to scraping diverse but unreliable web data for this specific task.

StructScore improves upon naive "VLM-as-a-judge" approaches by using fine-grained, verified Q&A pairs, showing strong alignment with human evaluators.

Integration of Reasoning in the paper

### Weaknesses
1. Do we really need image editing to do this task? People can just ask LLM to write code to generate the new image, which is more accurate. With pixel level image generation model, there will be inevitable artifacts. 

2. How to make sure this data curation pipeline is accurate? Any error rate statistics?

3. No evaluation in addition to infographics. You should evaluate on the natural image editing task.

### Questions
See above comments.

### Soundness
3

### Presentation
3

### Contribution
3

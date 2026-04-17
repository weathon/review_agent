# Evian: Towards Explainable Visual Instruction-tuning Data Auditing

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 4

## Abstract
The efficacy of Large Vision-Language Models (LVLMs) is critically dependent on the quality of their training data, requiring a precise balance between visual fidelity and instruction-following capability. Existing datasets, however, are plagued by inconsistent quality, and current data filtering methods rely on coarse-grained scores that lack the granularity to identify nuanced semantic flaws like logical fallacies or factual errors. This creates a fundamental bottleneck in developing more reliable models.
To address this, we make three core contributions. First, we construct a large-scale, 300K-sample benchmark by systematically injecting diverse, subtle defects to provide a challenging testbed for data auditing. Second, we introduce a novel "Decomposition-then-Evaluation" paradigm that breaks model responses into constituent cognitive components: visual description, subjective inference, and factual claim, enabling targeted analysis. Third, we instantiate this paradigm via EVIAN (Explainable Visual Instruction-tuning Data AuditiNg), an automated framework that evaluates these components along the orthogonal axes of Image-Text Consistency, Logical Coherence, and Factual Accuracy. Our empirical findings challenge the prevailing scale-centric paradigm: a model fine-tuned on a compact, high-quality subset curated by EVIAN consistently surpassed models trained on orders-of-magnitude larger datasets. We also reveal that dividing complex auditing into verifiable subtasks enables robust curation, and that Logical Coherence is the
most critical factor in data quality evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the data quality bottleneck in training Large Vision-Language Models by proposing fine-grained auditing instead of coarse filtering. It builds a 300K-sample benchmark with systematically injected, subtle defects, and introduces a Decomposition-then-Evaluation paradigm that separates responses into visual description, subjective inference, and factual claim. The EVIAN pipeline evaluates these components along Image-Text Consistency, Logical Coherence, and Factual Accuracy. Experiments show that models fine-tuned on EVIAN-curated, compact, high-quality data outperform those trained on much larger datasets, with Logical Coherence emerging as the most impactful quality dimension.

### Strengths
1. The problem this paper studies is a critical problem in visual instruction tuning.
2. The proposed auditing method is technically sound and performs well as demonstrated by experiments.
3. The formulated multi-faceted quality assessment is also practical and reasonable.

### Weaknesses
1. Limited baselines for comparisons. Since a well-trained MLLM (Qwen2.5-VL) is involved in the proposed method, some important baselines are available. For example, what about distilling from this MLLM to clean the dataset. In addition, can we audit the datasets by directly prompting the MLLM?)
2. Lack of **direct** comparisons on the performance of different auditing methods. Although Table 3 shows the proposed method is superior to existing ones in terms of MLLM benchmarks scores, this is **indirect**. Can the authors compare the auditing performance of the baselines (existing ones and the direct prompting method mentioned in weakness 2) in a **direct** manner? The authors could plot a precision-recall curve for all the compared methods. 
3. Lack of results on per-category dataset audition. Can the authors report the auditing performance (direct evaluation) for each category studied (e.g., categories in Figure 4)?

### Questions
1. What models are involved in the SCALE baseline?
2. The authors use Qwen2.5-VL-7B to perform audition. However, this model might perform poorly for tasks like fact checking or logical reasoning. The authors are suggested to use its larger version, e.g., Qwen2.5-VL-72B.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces EVIAN, a framework for auditing visual instruction-tuning data. The framework decomposes model responses into visual, inferential, and factual components, which are subsequently evaluated for consistency, logical soundness, and factual accuracy. Furthermore, the paper presents a 300K-sample benchmark containing systematically injected defects to assess the data auditing capability of the proposed pipeline. Experimental results on this benchmark demonstrate that models trained on EVIAN-curated data consistently outperform several baselines.

### Strengths
This paper investigates the evaluation, cleaning, and construction of high-quality visual instruction tuning datasets—a critical problem for advancing VLMs.

### Weaknesses
1. The core approach relies on using LLMs to decompose and re-annotate existing annotations. However, compared to prior LLM/VLM-based data-cleaning works [1–4], this paper lacks significant technical novelty. The primary distinction lies in the specific text prompts employed. Additionally, the paper doesn’t properly discuss related works on LLM- or VLM-assisted data refinement, especially in the context of instruction-tuning [1–4]
    
    [1] Chen et al. Alpagasus: Training a Better Alpaca with Fewer Data.
    
    [2] Wei et al. InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4.
    
    [3] Wang et al. Finetuned Multimodal Language Models Are High-Quality Image-Text Data Filters.
    
    [4] Zhang et al. Reflective Instruction Tuning: Mitigating Hallucinations in Large Vision-Language Models.

    
2. The experiments are all conducted on a custom benchmark that uses synthetic noise. This is problematic because it’s not representative of real-world data. Since the noise is artificially added, models might just learn to pick up superficial patterns rather than solving noise-related issues. As a result, the reported results don’t give us a reliable picture of how the method would work in real-world scenarios.
3. The evaluation omits several critical baselines. Notably, the performance of Qwen2-VL and the results of finetuning baselines on clean data are not reported. In addition, the proposed method uses a 235B-parameter LLM, while most of the compared models are much smaller. This huge gap in model size makes the comparisons unfair.
4. The proposed pipeline is overly complex and computationally expensive. It remains unclear how it would perform relative to a simpler yet strong alternative, e.g., re-annotating data using a state-of-the-art VLM of comparable size.
5. One big issue with noisy instruction-tuning data is that it can make hallucination problems. However, the paper doesn’t include any evaluation on commonly used hallucination benchmarks, such as POPE.
6. The paper have several readability issues:
    - The notations *I*, *P*, and *R* (line 161) are not introduced.
    - Figure 3’s color scheme is confusing and poorly explained—it’s not introduced properly in the figure, caption, or text.

### Questions
Please see weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the critical problem of data quality in the training of LVLMs, particularly for visual instruction tuning. The authors argue that existing data filtering methods, which often rely on coarse-grained similarity scores, fail to detect subtle but crucial semantic flaws like logical fallacies, factual inaccuracies, or hallucinations.
To tackle this, the paper introduces three core contributions:

1. A new paradigm called "Decomposition-then-Evaluation," which breaks down complex model responses into three verifiable components: pure visual descriptions, subjective inferences, and external factual claims.

2. An instantiation of this paradigm in a pipeline named EVIAN. EVIAN first decomposes the responses and then evaluates them along three distinct axes: Image-Text Consistency, Logical Coherence, and Factual Accuracy.

3. A large-scale, 300K-sample benchmark specifically designed for auditing, which includes a "gold standard" set and a "challenge" set created by systematically injecting a diverse taxonomy of defects.

The authors' key empirical finding is that a model fine-tuned on a small, 10K-sample subset curated by EVIAN significantly outperforms models trained on the entire 300K unfiltered dataset and on subsets selected by other filtering methods. Furthermore, an ablation study reveals that Logical Coherence is the most critical dimension for data quality, as its exclusion leads to a drastic degradation in downstream performance.

### Strengths
1. The quality of training data is arguably one of the most significant bottlenecks for advancing LVLMs. This paper moves beyond simplistic "more data is better" or coarse filtering approaches to tackle the nuanced and critical issue of semantic and logical data integrity

2. The "Decomposition-then-Evaluation" paradigm is a strong conceptual contribution. It is an elegant way to reframe the intractable problem of evaluating a complex, open-ended response into a set of more manageable, verifiable sub-problems. The three evaluation axes are well-chosen and cover the primary failure modes of current LVLMs.

### Weaknesses
1.  **Heavy Reliance on a Superior Model as Auditor:** The framework's core logic depends heavily on using large, powerful models as auditors, which raises fundamental questions about the source of the observed improvements. This issue is particularly pronounced in the paper's experimental setup:

    *   **A Form of Knowledge Distillation:** The primary experiments involve fine-tuning a `Qwen2-VL-2B` model on data curated by a more advanced `Qwen2.5-VL-7B` model from the same family. This setup can be interpreted as a form of knowledge distillation, where a stronger "teacher" model provides high-quality signals (the EVIAN scores) to guide the training of a smaller "student" model. Consequently, it becomes difficult to disentangle the benefits of the EVIAN methodology itself from the benefits of simply having a superior model guide a weaker one. The gains might stem more from the raw capability of the teacher model than from the intrinsic value of the decomposition framework.

    *   **The "Sufficiently Powerful Auditor" Paradox:** This leads to a more profound question: if the goal is to improve an already SOTA base model, can we always find a significantly more capable model to serve as a reliable auditor? This approach may work for improving smaller models, but its utility for pushing the frontier of the most advanced models is questionable, as it presupposes the existence of a "super-SOTA" judge.

2.  **Scalability Concerns:** Recent work, such as the analysis from the FineVision paper, has explored data curation at a much larger scale. One of their key findings suggests that once the dataset size reaches a certain critical mass, the negative impact of removing even "low-quality" samples may outweigh the benefits of training on a smaller, cleaner set. The study notes:
    > "This could indicate that with a sufficiently large dataset that you train on long enough, it hurts more to remove samples, even if they were judged to be of low quality, than to train on them."

### Questions
Would your conclusion still hold when scaling to models and datasets that are orders of magnitude larger?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents EVIAN, a novel explainable data auditing framework for Visual Instruction Tuning in VLMs. The method follows a Decomposition-then-Evaluation paradigm that decomposes model responses into visual, inferential, and factual components and assesses them along three interpretable dimensions—Image-Text Consistency, Logical Coherence, and Factual Accuracy. The authors further construct a 300K-sample benchmark with systematically injected semantic defects to facilitate fine-grained auditing. Empirical results show that models fine-tuned on EVIAN-curated data outperform those trained on much larger unfiltered datasets, suggesting that data quality outweighs data scale in VLMs training.

### Strengths
- The Decomposition-then-Evaluation approach offers an interpretable and modular framework for assessing data quality, moving beyond opaque single-score filtering metrics like CLIPScore.
- The authors introduce a 300K-sample benchmark with controlled semantic defect injection—providing a rare, well-structured dataset for studying fine-grained multimodal data auditing.
- EVIAN consistently outperforms previous baselines such as SCALE and CLIP-based methods across multiple downstream benchmarks (e.g., MME, ScienceQA), demonstrating robust generalizability.
-  The paper includes detailed analyses that isolate the contribution of each dimension (SL, SK, SV), clearly showing Logical Coherence as the most critical factor.

### Weaknesses
- The auditing pipeline relies heavily on Qwen-series models (e.g., Qwen3-235B and Qwen2.5-VL-7B). This raises concerns about model bias transfer, as the auditor and the evaluated data share similar architectures and training distributions.
- The benchmark’s defect injection procedure—though systematic—depends on LLM-generated corruptions. These artificial errors may not fully capture the complexity or distribution of real-world data flaws. The paper could improve by including evaluations on naturally noisy datasets or human-annotated errors.
- While EVIAN improves data quality, the computational cost of multi-step decomposition and evaluation (each requiring multiple large model calls) is not quantified. 
- The final selection score uses an equal-weight averaging scheme (Eq. 2). Although the authors note tunability, they do not show how varying weights across tasks affects performance. A sensitivity analysis would better justify the robustness of this design choice.

### Questions
Are there any human evaluation results confirming that high EVIAN scores correspond to perceived high-quality samples?

### Soundness
3

### Presentation
3

### Contribution
2

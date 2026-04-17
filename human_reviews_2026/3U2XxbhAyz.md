# Q-Mirror: Unlocking the Multi-Modal Potential of Scientific Text-Only QA Pairs

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
High-quality, multi-modal benchmarks are crucial for advancing scientific reasoning in large models yet their manual creation is costly and unscalable. To address this bottleneck, we explore the potential for transforming Text-Only QA Pairs (TQAs) into high-quality Multi-Modal QA Pairs (MMQAs), which include three parts: 1) Task Definition & Evaluation Rubric: We develop a TQA-to-MMQA framework and establish a comprehensive, multi-dimensional MMQA quality rubric that provides principles for the transformation. 2) Benchmark Construction: Then we construct two extensive benchmarks to rigorously evaluate state-of-the-art generation & understanding models on the distinct tasks of MMQA generation & MMQA quality evaluation. 3) Preliminary Solution: We develop an agentic system Q-Mirror, which operationalizes our framework by integrating MMQA generation and evaluation into a closed loop for iterative refinement. Our experiments show that while state-of-the-art models can generate MMQAs, their outputs still leave substantial gaps, underscoring the need for reliable evaluation. We further demonstrate that top-tier understanding models align closely with human judgment in MMQA quality assessment. Leveraging both insights, the Q-Mirror agent raises average scores from 78.90 to 85.22 and pass rates from 72% to 95%, offering a practical path to large-scale scientific benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Q-Mirror, a systematic framework that converts text-only scientific QA pairs (TQAs) into multi-modal QA pairs (MMQAs) by integrating visual elements such as diagrams and formulas.

The contributions include:

A TQA-to-MMQA transformation framework guided by a multi-dimensional quality rubric (Information Consistency, Cross-Modal Integration, and Standalone Quality).

Construction of two benchmarks — one for MMQA generation and another for MMQA evaluation.

Development of an agentic system (Q-Mirror) that combines generation and evaluation in a closed loop to iteratively refine MMQAs.

Experiments benchmark multiple LMM families (GPT, Qwen, Doubao, Grok, etc.) and automated judge models against human annotations. The Q-Mirror agent improves the average quality score from 78.9 → 85.22 and pass rate from 72% → 95%, showing strong empirical benefits.

### Strengths
## Timely and Original Problem

Addresses a genuine and pressing issue: the scarcity of high-quality multi-modal scientific data.

The idea of “unlocking latent multi-modal potential” in text-only data is conceptually novel and impactful.


## Systematic and Reproducible Framework

Provides a principled rubric (IC, CM, QT) that is clear, quantifiable, and theoretically motivated.

Includes algorithmic details (Algorithm 1), explicit thresholds, and controlled experimental parameters (τ=80, N=5, etc.).


## Comprehensive Evaluation

Benchmarks across five model families for generation and ten LMMs for evaluation.

Includes both human expert annotations and automatic scoring, plus inter-model agreement analysis (Spearman correlations, ensemble reliability).

Promises public release of datasets and code — excellent for reproducibility.

### Weaknesses
W1: Q-Mirror both uses and evaluates LMMs as judges, which risks self-confirmation bias.

W2: Although an ensemble mitigates this, it’s unclear whether the “judge alignment” truly reflects objective human-level quality or just model-to-model agreement.

W3: It’s unclear how much improvement comes from the framework itself vs. the underlying GPT models.

W4: The paper briefly mentions poor performance on “scientific plausibility (SC)” but does not analyze why or provide visual examples of failure.

W5: The quality rubric (IC, CM, QT) is only heuristically defined. There is no theoretical or statistical validation that these dimensions are orthogonal or collectively sufficient to represent MMQA quality.

W6: Only 440 total QA pairs (310 + 130) are used — too small to justify the claim of “large-scale benchmark” or to establish statistical reliability.

W7: The sampled tasks are largely STEM textbook-level, not real-world multimodal scientific reasoning (e.g., data plots)

W8: The choice of weighting (0.3, 0.3, 0.4) for the final composite score (Equation 4) is arbitrary, with no ablation or justification.

W9: The scalability claim (“cost-effective at scale”) is unproven — no time, cost, or computational analysis is presented.

### Questions
same as weakness

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
4

### Summary
This paper addresses the lack of scalable methods for generating high-quality multimodal question-answer (MMQA) pairs. It proposes a framework that converts text-only QA pairs into MMQAs. The authors formally define the subtasks involved in MMQA generation and introduce an evaluation rubric to assess model performance. Using this rubric, they construct a benchmark to evaluate state-of-the-art large language models (LLMs) on MMQA tasks. Furthermore, they develop an agentic system, Q-Mirror, which iteratively generates and evaluates MMQAs to improve quality and consistency.

### Strengths
The paper demonstrates strong originality by introducing a novel rubric for evaluating generated multimodal question–answering (MMQA) systems. Its evaluation framework is both detailed and comprehensive, covering a wide range of models. The writing is clear and well-structured, with informative tables and figures that enhance understanding. Overall, the work addresses an important gap in the field by tackling the lack of standardized and high-quality MMQA evaluation methods.

### Weaknesses
The paper proposes a method to address the lack of scalable tools for transforming text-only QA pairs into multimodal ones. However, the reliance on both expert and LLM judgments limits the claimed scalability, as human input remains a core requirement for maintaining quality.
The results reveal strong performance in the SI dimension but poor performance in the SC dimension. This suggests that the tested LLMs struggle to generate or select images that meaningfully represent the underlying problem, reducing the utility of the resulting MMQAs. The paper would benefit from exploring alternative image generation or retrieval strategies that ensure visual components are contextually relevant and enhance the QA task, rather than serving as loosely associated visuals.
Moreover, the reported 62.8% agreement between LLM judges and human annotations indicates a relatively weak alignment, which undermines confidence in the Q-Mirror agent’s closed-loop evaluation process. Since the evaluation stage is central to the agent’s effectiveness, future work should focus on improving alignment with human judgment

### Questions
1. How was the single-model solvability computed?
2. How were the values for beta set and how does changing these values affect the outcome of the CM score?
3. There is mention of using a judge ensemble to guarantee closer alignment with human judgement. Could you provide some empirical evidence for this?
4. As each model produces varying diagrams, how is the output scored for the benchmark?

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
3

### Summary
This paper proposes a framework and an agentic system, Q-Mirror, to scalably generate high-quality multimodal benchmarks by transforming text-only QA pairs through a closed-loop process of iterative generation and evaluation.

### Strengths
- This paper attempts to address an interesting and important problem: how to construct a pipeline for transforming Text QA into Multimodal QA, and how to evaluate the quality of this transformation.

- The proposed pipeline and quality rubric are intuitively sensible.

- The authors have evaluated the transformation quality across multiple model families.

### Weaknesses
- The paper mentions establishing "a transformation pipeline for effective SMQ-to-MMQA transformation," yet it does not actually generate a large-scale MMQA dataset to train VLMs, which would be the definitive test of the data's quality. In other words, does an improved score on the MMQA generation benchmark actually correlate with higher data quality?
    - I believe the most reliable method for assessing data quality is to train a VLM on the synthetic data. The model's performance on downstream benchmarks would then reflect the true data quality, and one could verify if these results align with the rankings presented in Table 4.

- The paper should include a cost analysis of the Q-Mirror Agent. The Q-Mirror process involves multiple inference and refinement steps. If the resulting performance gains come at a computational cost that far exceeds the GPT family, it will likely be difficult to scale in practical applications.

### Questions
Why are both Line 382 and Line 389 labeled "Judge Alignment on Q-Mirror-Expert"?

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
4

### Summary
This paper presents a well-executed and clearly written framework for converting text-only scientific QA pairs into multimodal QA tasks. The methodology is systematic and reproducible, with clearly defined rubrics (IC, CM, QT) and thorough experimental validation. The proposed evaluation loop combining generation and judgment is well motivated and demonstrates consistent gains.

### Strengths
1. The paper is clearly written and well-structured, presenting a systematic framework for converting text-only scientific QA pairs into multimodal QA tasks.
2. The proposed evaluation rubric (IC, CM, QT) and the closed-loop generation-judgment process are well motivated and executed with solid experimental validation.
3. The methodology is practical, showing consistent gains and good alignment between automated and human evaluations.
4. Overall presentation quality is high, with organized sections, clear figures, and sound analysis.
5. The experiments are pretty complete and include a lot of open-sourced and closed-sourced models.

### Weaknesses
I would like to first say that I actually like the idea and the paper overall; it is a creative attempt to bridge text-only QA with multimodal reasoning. However, I do have one strong concern regarding the quality of the generated images.

In Figure 7, which the authors describe as a "successful example," the generated table headers are nearly illegible. Most characters are incomplete or incorrectly rendered. This kind of error is, to some extent, expected, as text-to-image models frequently struggle with accurate text transcription.

Similarly, in Figure 10, another claimed "successful case," the original question explicitly mentions that "An electron is moving along the horizontal axis from the left with a Lorentz factor of 4." This crucial information is completely missing in the corresponding figure. It is unclear how such a model could truly support reasoning if the generated visualization is missing essential quantitative context. Interestingly, this information appeared in an earlier version of the figure (see figure 11), but was removed in the later ones, which raises questions about the consistency and reliability of the generation process.

### Questions
As mentioned in the weaknesses, could the authors provide a few factually correct examples of generated images? Based on the current samples shown in the paper, it is difficult for readers to believe that the method truly works as intended, since even the "successful" figures contain clear factual or textual errors.

I would be happy to raise my score if the authors can demonstrate that the image generation process can reliably produce scientifically accurate and semantically consistent visualizations. Overall, my main concern lies in the factual correctness of the generated images rather than the overall idea, which I find promising.

### Soundness
3

### Presentation
3

### Contribution
2

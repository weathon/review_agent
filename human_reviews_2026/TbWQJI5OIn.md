# The Telephone Game: Evaluating Semantic Drift in Unified Models

- Decision: Reject
- Scores: 4, 4, 8

## Abstract
Employing a single, unified model (UM) for both visual understanding (image-to-text: I2T) and visual generation (text-to-image: T2I) has opened a new direction in Visual Language Model (VLM) research. While UMs can also support broader unimodal tasks (e.g., text-to-text, image-to-image), we focus on the core cross-modal pair T2I and I2T. Existing evaluation benchmarks consider these capabilities in isolation: FID and GenEval for T2I, and benchmarks such as MME, MMBench for I2T. These isolated single-pass metrics do not reveal  cross-consistency: whether a model that “understands” a concept can also “render” it, nor whether semantic meaning is preserved when cycling between image and text modalities. To address this, we introduce the Semantic Drift Protocol (SDP) for Unified Models, a cyclic evaluation protocol that alternates I2T and T2I over multiple generations to quantify semantic drift. We propose two metrics: (i) Mean Cumulative Drift (MCD), an embedding-based measure of overall semantic loss; and (ii) Multi-Generation GenEval (MGG), an object-level compliance score extending GenEval. To assess generalization beyond COCO dataset, which is widely used in training; we create a new benchmark Nocaps+Docci400, sampled from NoCaps and DOCCI and evaluate on seven recent models. SDP reveals substantial variation in cross-modal stability: some models like BAGEL maintain semantics over many alternations, whereas others like Vila-u drift quickly despite strong single-pass scores. Our results highlight SDP as a necessary complement to standard I2T and T2I evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes an evaluation benchmark for unified models, termed the Semantic Drift Protocol (SDP), which focuses on measuring semantic drift during the image-to-text (I2T) and text-to-image (T2I) generation cycle. Specifically, the authors introduce two metrics:  (i) Mean Cumulative Drift (MCD), an embedding-based measure of overall semantic loss; and  (ii) Multi-Generation GenEval (MGG), an object-level compliance score that extends GenEval.  Extensive experiments demonstrate the effectiveness of the proposed evaluation framework.

### Strengths
* SDP takes multi-turn generation tasks into account, addressing a gap in previous evaluation frameworks.  
* The experimental design is thorough, and the results yield insights with meaningful implications for the research community.

### Weaknesses
* The proposed framework lacks novelty. The main contributions, MCD and MGG, are relatively straightforward: MCD is essentially the mean of multi-turn embedding similarity scores, while MGG is a direct extension of GenEval.
* The motivation is somewhat unclear. Although multi-turn dialogue tasks for unified models represent an important research direction, the proposed evaluation framework does not appear to align well with realistic application scenarios. In most real-world settings, after obtaining an initial dialogue output, users would typically modify their prompts to refine the result rather than repeatedly generating the same (I, T) pair as described in this paper.
* While the paper introduces an evaluation framework, it would be more compelling if the authors demonstrated its practical utility, for instance, by performing post-training or fine-tuning based on the proposed scores and showing measurable improvements in model performance.
* The paper includes human evaluation results but provides insufficient details about the evaluation procedure. A more comprehensive description of the human evaluation protocol is necessary to ensure the fairness and reliability of the results.
* The writing lacks clarity and coherence, and the paper contains numerous typographical errors (e.g., line 25: “image-totext” should be “image-to-text”).

### Questions
See 'Weakness'.

### Soundness
2

### Presentation
2

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
This paper proposes the Semantic Drift Protocol, an evaluation framework for unified multimodal models that perform both text-to-image and image-to-text tasks. It introduces two key metrics — Mean Cumulative Drift and Multi-Generation GenEval — to measure how well models preserve meaning across multiple modality conversions, addressing a gap in existing benchmarks that only evaluate single-pass performance. The authors also present a new benchmark dataset, NoCaps + DOCCI400, and evaluate seven recent models, finding large differences in their cross-modal semantic stability.

### Strengths
Most existing multimodal benchmarks assess text-to-image and image-to-text performance separately, giving little attention to semantic consistency across iterative modality exchanges.  Such consistency is especially important for tasks like multi-turn visual question answering, vision–language dialogue, video captioning, and related multimodal tasks, where meaning must remain stable across modalities.

The proposed evaluation protocol fills this gap by simulating successive T2I–I2T transformations, capturing how meaning shifts across repeated modality exchanges — a property overlooked by single-pass metrics.

By comparing shared-weight and decoupled architectures, the paper further shows that shared models tend to preserve meaning more effectively across rounds.

### Weaknesses
The paper presents an interesting idea and addresses a gap in multimodal evaluation, but its conceptual and empirical grounding are weak.
The definition of *semantic drift* lacks rigor, the metrics suffer from representational bias, and the experiments fail to validate the stated motivation — that higher semantic consistency benefits downstream multimodal reasoning tasks.
Overall, the work remains exploratory rather than a robust or generalizable benchmark.

1. Conceptual Ambiguity in the Definition of Semantic Drift
The central concept is poorly defined. Treating semantic drift as embedding similarity decay is conceptually weak and conflates representation change with genuine semantic degradation. This assumption is unverified—embedding distance measures variation in representation, not meaning. As a result, benign paraphrasing or stylistic variation can be incorrectly penalized as drift, while the metric itself merges object-, attribute-, and relation-level shifts into a single opaque score.

 2. Bias from Embedding-Space Alignment
Both metrics (MCD and MGG) rely on pretrained embeddings and detectors, introducing representational bias.  Models trained in similar embedding spaces may appear more consistent—not due to true semantic stability, but because their representations align with the evaluation backbone.
Without checks using alternative embeddings or detectors, it is unclear whether the reported rankings reflect genuine stability or embedding overlap, undermining the claim that SDP is architecture-agnostic.

3. Lack of Downstream or Practical Validation
All experiments are limited to synthetic T2I–I2T cycling and a human-correlation study, with no evidence that higher semantic consistency improves real multimodal tasks such as VQA, captioning, or visual dialogue. Without quantitative or qualitative downstream validation, the claimed importance of the proposed metrics remains speculative, and it is unclear whether the evaluation protocol reflects meaningful model behavior beyond the test loop.

### Questions
Definition of Semantic Drift
The definition of “semantic drift” as embedding similarity decay is ambiguous. It does not distinguish genuine semantic loss (e.g., object or relation errors) from benign changes like paraphrasing or stylistic variation. How can the authors ensure that their metric reflects semantic degradation rather than surface variation?

Embedding Dependence
Since both MCD and MGG rely on pretrained embeddings (CLIP, DINO, MPNet), results may favor models trained in similar representation spaces. Have the authors tested whether model rankings remain consistent under alternative embeddings or random projections?

Downstream Relevance
The paper claims that higher semantic consistency benefits multimodal task, but provides no downstream validation. Can the authors show whether SDP scores correlate with task performance on VQA, captioning, or dialogue?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Evaluation of unified multimodal models remains limited, as image understanding and generation are evaluated in isolation. This paper formulates the semantic drift and cross-consistency problems, and shows that existing single-pass metrics cannot thoroughly measure the gap between understanding and generation capabilities. To address this, the paper proposes the Semantic Drift Protocol (SDP), a cyclic evaluation framework that measures how well a unified model preserves semantic fidelity by alternating between image-to-text (I2T) and text-to-image (T2I) generation. They also introduce a new benchmark (Nocaps+Docci400) evaluating seven recent models.

### Strengths
* The proposed evaluation protocol effectively measures a unified model's ability to preserve semantic fidelity across modalities, which existing metrics or benchmarks fail to capture. The results in Figure 5 clearly demonstrate this point, providing intriguing evidence of how semantic drift can occur across different categories.
* The main figures (Figures 1 and 2) clearly explain the core problem the paper is addressing.
* The categorization of unified models in Section 2 is clear and well-motivated, and the experiments thoughtfully include models from different categories.
* They provide a human study that confirms a strong correlation between the proposed metric and human-perceived quality.

### Weaknesses
* Including a guideline on the recommended number of evaluation cycles would improve reproducibility and facilitate adoption of this protocol in future research.
* It would be helpful to improve the clarity of the x-axis and y-axis labels in Figure 6 (e.g., Sδ(g) distance score vs. Number of generations). Also, the figure should be referenced accurately in the main text (line 398: change "Plot" to "Figure" 6).
* Typos in line 252.

### Questions
* It is very interesting that BLIP-3o, a partially shared model, performs exceptionally well on this metric, achieving the second-best result. Do the authors have any insights into why this might be the case?
* Overall, the paper suggests a novel evaluation protocol that can effectively measure a qualitatively different aspect of unified models. Minor presentation quality (e.g., titles, axis labels, and font size of plots, and typos) could be improved for greater clarity and impact.

### Soundness
3

### Presentation
3

### Contribution
3

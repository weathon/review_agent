# MPS: A Multi-Perspective Benchmark For Assessing Spurious Correlations in Text Classification

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Text classification is especially susceptible to diverse spurious correlations, such as those related to word-frequency and concept-level patterns. Nevertheless, there is a lack of a comprehensive and standardized benchmark for evaluating the robustness of models against these spurious correlations. To address this crucial issue, we present MPS (Multi - Perspective Benchmark For Assessing Spurious Correlations in Text Classification). To construct this benchmark, we collect eight widely used text classification datasets and introduce five categories of spurious correlations for each of them, producing 40 variants of datasets for comprehensively evaluating spurious correlations in diverse settings.We then extensively evaluate various text classification models and state-of-the-art anti-spurious correlation methods on this benchmark, which uncovers the vulnerabilities of these models and methods to diverse spurious correlations. A follow-up comparative analysis on this benchmark is performed to assess the performance of these anti-spurious correlation methods and humans in diverse settings.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a multi-perspective benchmark for systematically evaluating how text classifiers rely on spurious correlations, covering five types of spurious cues -- sentence-level, core-word, negation-based, question-based, and word-frequency biases -- across eight popular NLP datasets. The authors propose two new evaluation metrics derived from worst-group accuracy, to measure robustness degradation and potential improvement. Through extensive experiments on eight widely used datasets and multiple model families, they find that existing models and mitigation methods struggle to handle all types of spurious correlations, with question-based biases emerging as particularly challenging.

### Strengths
(1) The introduction of the Question-Based Spurious correlation (QBS) category highlights a previously underexplored yet impactful source of model bias.

(2) The proposed δ and Δ metrics, based on worst-group accuracy, allow fine-grained quantification of robustness degradation and potential improvement.

(3) The paper evaluates a wide range of models (traditional, MLMs, LLMs) and robustness methods, revealing consistent vulnerabilities across approaches.

### Weaknesses
(1) The idea of benchmarking spurious correlation learning in text classification is not new -- it builds directly on prior frameworks (e.g., Shortcut Maze) without introducing fundamentally new insights. The paper frames itself as the “first comprehensive benchmark”, which is inaccurate given prior large-scale studies.

(2) Unlike earlier work that manipulates correlation intensity (e.g., λ in Shortcut Maze), MPS uses only binary Balanced vs. Imbalanced splits, reducing granularity in robustness measurement.

(3) Although claiming to be comprehensive, the paper omits four spurious types -- synonym, register, author style, and concept correlation -- that were defined in Shortcut Maze.

(4) The new question-based spurious type lacks clear operational boundaries. some questions may genuinely reflect task semantics rather than bias.

(5) The main paper would benefit from a summary figure or table defining, explaining, and exemplifying each spurious type.

(6) Because MPS constructs artificially balanced and imbalanced splits, it may not fully capture naturally occurring spurious correlations present in real-world data.

(7) The redistribution used to create these splits may unintentionally alter label distributions and subgroup sizes, introducing confounding factors beyond the intended spurious correlations. Consequently, performance gaps between splits might reflect class imbalance or distributional shifts, rather than true model sensitivity to spurious features.

### Questions
(1) Could the authors explicitly articulate how MPS advances beyond prior benchmarks such as Shortcut Maze? In particular, how do the proposed five spurious types and eight datasets provide new analytical insights rather than simply broader coverage? A clearer statement of conceptual novelty would help calibrate expectations.

(2) In constructing Balanced and Imbalanced splits, how do the authors control for changes in label distributions, subgroup sizes, or topic diversity that could introduce confounding effects?

(3) The human study is an interesting addition—could the authors elaborate on participant demographics, task instructions, and agreement rates?

(4) Since MPS focuses solely on text classification, do the authors foresee extending the benchmark to generation or reasoning tasks?

### Soundness
3

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
2

### Summary
The paper introduces MPS, a benchmark built from 8 text classification datasets to evaluate model robustness under five spurious correlation types, namely SCS, CCS, NBS, QBS, and WFS. It defines two analysis metrics, which are δ (change in worst-group accuracy when balancing a given shortcut type) and Δ (headroom from worst-group under shortcut to overall accuracy after balancing), to quantify each shortcut’s impact. Across models and mitigation methods, results show no single method is robust across all five types, and the newly defined QBS is particularly challenging.

### Strengths
The paper’s core strength lies in its scope and standardization. A unified benchmark across eight datasets and five shortcut types), with paired balanced/imbalanced splits, and clear robustness metrics (W-ACC, δ/Δ) that isolate and quantify shortcut effects. Empirically, it’s broad and careful, covering classic baselines, pretrained encoders, multiple mitigation families, and LLM backbones, plus a human reference, yielding actionable findings.

### Weaknesses
1. The distinction between sentence-level concepts (SCS) and core-word concepts (CCS) is not fully transparent from the main text.
2. It is not clear how static attributes of SCS are selected and the relationship with their corresponding labels.
3. Some ambiguous parts, like the usage of W-ACC and δ, are listed in the following Questions section.

### Questions
1. Table 7’s descriptions do not make the distinction between SCS and CCS sufficiently clear. Could you provide precise definitions and concrete, dataset-specific examples for each to clarify the difference?
2. When δ < 0 (e.g., strong results on an imbalanced split), how do you conclude that the model exploits spurious correlations rather than demonstrating genuine understanding?
3. How to explain the large negative δ of TCM on Ag News in Table 1?
4. Could the low W-ACC simply because of small sample sizes in certain (𝑦, 𝑎) groups, rather than true vulnerability to the spurious attribute.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces MPS (Multi-Perspective Benchmark For Assessing Spurious Correlations in Text Classification), a comprehensive benchmark for evaluating model robustness against spurious correlations in text classification. The authors systematically categorize spurious correlations into five types: SCS (Sentence-level Concept Spurious), CCS (Core-word Concept Spurious), NBS (Negation-Based Spurious), QBS (Question-Based Spurious), and WFS (Word-Frequency-based Spurious). Using 8 widely-used datasets, they create 40 dataset variants and conduct extensive evaluations of various models (MLMs, LLMs, traditional ML) and anti-spurious correlation methods. The work includes human performance comparisons and introduces novel metrics ($\delta$ and $\Delta$) to quantify spurious correlation effects.

### Strengths
1. The five-type taxonomy of spurious correlations (SCS, CCS, NBS, QBS, WFS) provides a clear framework for analysis. QBS (Question-Based Spurious correlations) appears to be a genuinely new contribution that existing methods struggle with.
2. Testing 12 models and 5 anti-spurious correlation methods across 40 dataset variants represents substantial empirical work. The inclusion of human performance baselines (Table 4) provides valuable context and reveals interesting patterns (e.g., humans outperform models on emotional tasks but under-perform on contextual classification).

### Weaknesses
1. The 10% manual verification rate is too low for a benchmark paper. The paper should provide: (a) inter-annotator agreement scores, (b) error analysis of LLM mistakes, (c) validation on a larger sample or full validation on at least one dataset. Sample sizes, annotator selection criteria, training procedures, and inter-annotator agreement are not provided. This makes it difficult to assess the reliability of human performance claims.
2. The Imbalanced/Balanced construction procedure needs algorithmic detail. What constitutes "overwhelmingly dominant"? What are the exact class distribution targets? Without this, reproduction is difficult.
3. The paper doesn't ablate key design choices. For example: How sensitive are results to the choice of 6 spurious attributes for SCS/CCS? What about the training epoch selection strategy?
4. The paper evaluates existing methods but doesn't propose new solutions. While benchmark papers need not introduce new methods, some guidance on promising directions would strengthen the contribution.

### Questions
1. Can you provide pseudocode or precise algorithmic descriptions for constructing Imbalanced and Balanced subsets? What specific thresholds define "overwhelmingly dominant"?
2. Beyond the 10% validation, can you provide error analysis? What types of mistakes does Llama 3.1 make in concept annotation? How do error rates vary across datasets?
3. What are the sample sizes for human evaluation? How many annotators? What was their expertise level? What was inter-annotator agreement?
4. Have you tested whether models trained to be robust on one spurious correlation type show improved robustness on other types?

### Soundness
2

### Presentation
3

### Contribution
2

# AXIS: Explainable Time Series Anomaly Detection with Large Language Models

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Time-series anomaly detection (TSAD) increasingly demands explanations that articulate not only if an anomaly occurred, but also what pattern it exhibits and why it is anomalous. Leveraging the impressive explanatory capabilities of Large Language Models (LLMs), recent works have attempted to treat time series as text for explainable TSAD. However, this approach faces a fundamental challenge: LLMs operate on discrete tokens and struggle to directly process long, continuous signals. Consequently, naïve time-to-text serialization suffers from a lack of contextual grounding and representation alignment between the two modalities. To address this gap, we introduce  AXIS, a framework that conditions a frozen LLM for nuanced time-series understanding. Instead of direct serialization,  AXIS enriches the LLM's input with three complementary hints derived from the series: (i) a symbolic numeric hint for numerical grounding, (ii) a context-integrated, step-aligned hint distilled from a pretrained time-series encoder to capture fine-grained dynamics, and (iii) a task-prior hint that encodes global anomaly characteristics. Furthermore, to facilitate robust evaluation of explainability, we introduce a new benchmark featuring multi-format questions and rationales that supervise contextual grounding and pattern-level semantics. Extensive experiments, including both LLM-based and human evaluations, demonstrate that AXIS yields explanations of significantly higher quality and achieves competitive detection accuracy compared to general-purpose LLMs, specialized time-series LLMs, and time-series Vision Language Models. The code is available in \url{https://anonymous.4open.science/r/TimeSemantic-1742/main.py}

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel approach to the time-series anomaly detection problem. The authors reformulate the task by focusing on interval-based anomalies rather than point-based ones, motivated by the observation that anomalies in real-world time series often occur over contiguous intervals instead of isolated timestamps. The problem is defined as follows: given a univariate time series and a natural-language query specifying an interval, the goal is to determine whether the interval is anomalous and to provide an explanation for this decision. To address this problem, the authors propose AXIS, an LLM-based framework designed to handle continuous time-series data. Since standard LLMs are not well-suited for processing continuous signals, AXIS introduces three complementary representation pathways, referred to as hints: the symbolic numerical hint, the context-integrated step-aligned hint, and the task-prior hint. These pathways aim to bridge the gap between time-series representations and language-based reasoning. The training process consists of two phases. In the first phase, the time-series encoder used in the context-integrated step-aligned hint is pre-trained to capture temporal dependencies. In the second phase, both the encoder and the LLM are kept frozen while the remaining parameters are optimized to align the representations with the anomaly explanation task. The paper also introduces a new benchmark for semantic time-series anomaly explanation, designed to evaluate both detection and interpretability. Experimental results show that AXIS outperforms baseline methods on this benchmark. Additionally, an ablation study is conducted to assess the contributions of individual components, providing evidence for the effectiveness of the proposed design choices.

### Strengths
- The proposed problem definition is novel and conceptually appealing. In particular, the formulation based on intervals rather than single points is well motivated, as anomalies in real-world time series typically extend over contiguous intervals instead of isolated timestamps. Furthermore, framing the task as generating explanations adds interpretability and makes the objective of the model more aligned with practical anomaly analysis needs.

- The paper clearly articulates one of the key challenges in applying LLMs to time-series data: LLMs operate on discrete tokens and therefore struggle to directly process long, continuous temporal signals. To address this issue, the authors propose a novel prompt construction strategy within the AXIS framework, which integrates three complementary representation pathways:
    - a symbolic numeric hint, providing numerical grounding for continuous values;
    - a context-integrated, step-aligned hint, distilled from a pre-trained time-series encoder to capture fine-grained temporal dynamics; and
    - a task-prior hint, encoding global anomaly characteristics and prior knowledge relevant to the task.

- The experimental results presented in Table 1 demonstrate that the proposed AXIS framework outperforms baseline methods on the newly introduced benchmark.

- The ablation study reported in Table 2 provides evidence that the proposed prompt design contributes significantly to performance improvements, confirming the effectiveness of the multi-hint representation strategy.

- The model architecture analysis in Table 3 indicates that the proposed approach is relatively robust across different LLM architectures, although performance variations are observed among specific models.

- Overall, the paper is clearly written and well structured. It provides sufficient background and motivation to understand the problem setting and the contributions. Figure 2 offers a concise and informative summary of the differences between the proposed approach and existing LLM-based methods for time-series anomaly detection.

### Weaknesses
- The main concern lies in the generation process and overall quality of the proposed benchmark. Although the paper introduces a new semantic anomaly explanation benchmark, its characteristics are not thoroughly analyzed. For instance, the paper could include descriptive statistics of the datasets, such as the number of samples, anomaly distributions, and interval lengths. It would also be useful to report how simple baseline heuristics perform on these datasets to better contextualize the difficulty and validity of the benchmark. Since all experimental results are evaluated exclusively on this benchmark, a more detailed analysis is crucial to assess the reliability and generalizability of the reported findings.

- The justification for the two-phase training strategy remains unclear. The paper would benefit from a more detailed explanation, supported by either theoretical reasoning or empirical evidence, to demonstrate why this specific training scheme is necessary or advantageous compared to alternative training strategies.

- The paper does not discuss how robust the proposed model is to input noise or data corruption. It would be valuable to analyze the model’s sensitivity to noise in the time series, such as testing whether performance remains stable when individual values or small segments are perturbed or corrupted. Such an analysis would provide additional insight into the robustness and practical applicability of the proposed approach.

### Questions
- My primary concern relates to the generation process and overall quality of the proposed benchmark, as discussed in the weaknesses section. I would be inclined to increase my evaluation score if the paper provided additional analyses or evidence validating the reliability and soundness of the benchmark.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents AXIS, a framework for explainable time-series anomaly detection (TSAD) that conditions a frozen large language model (LLM) to tackle two key challenges (1) contextual grounding and representation alignment (2) and contrasts its design with naive time-to-text serialization approaches. AXIS leverages three complementary types of hints: (i) a symbolic numeric hint that exposes the raw values within the queried time window, (ii) a context-integrated, step-aligned hint distilled from a pretrained time-series encoder to capture local and fine-grained temporal dynamics, and (iii) a task-prior hint that encodes global anomaly characteristics.

For evaluation, the authors (a) construct a multi-format QA benchmark of anomaly-centric questions paired with human-written rationales, (b) assess explanation quality using a G-Eval-style LLM-as-a-judge pipeline, and (c) conduct a human study comparing model outputs. Overall, AXIS achieves improved semantic anomaly explanation performance, outperforming LLMAD and several other baselines. Ablation studies further verify that all three types of hints contribute substantially to the model’s effectiveness.

### Strengths
S1: The paper clearly articulates why existing anomaly detectors and post-hoc attribution methods fall short in answering the question “why was this flagged?”, and proposes a principled remedy centered on contextual grounding and representation alignment.

S2: The combination of the step-aligned hint, task-prior hint, and symbolic numeric hint forms a well-motivated triad; employing a Perceiver-style cross-attention mechanism to map time-series features into the LLM token space is technically sound and aligns with prior work on prefix and prompt tuning, here innovatively adapted for TSAD explanation.

S3: The experimental comparisons are comprehensive, covering classical and deep baselines, time-series foundation models (e.g., Chronos, TimesFM, MOMENT), specialized TS-LLMs (e.g., ChatTS, AnomLLM, ChatTime), and a plot-based VLM baseline, demonstrating broad empirical coverage.

### Weaknesses
W1: The main explanation results rely on a single LLM judge (Gemini-2.5) using a G-Eval-style rubric. Prior studies have documented systematic biases in such setups—e.g., position and authority effects, prompt sensitivity, and reproducibility drift as models evolve. This raises concerns about the robustness and fairness of evaluation, suggesting that reliance on a single evaluator may overstate performance consistency.

W2: The proposed benchmark is procedurally generated. While the design encourages morphology-level reasoning, it risks inheriting the authors’ priors or artifacts from the data generation pipeline, particularly since similar classes of LLMs appear both in benchmark creation and evaluation. Although the paper mentions “benchmark integrity” checks, the description is brief and lacks sufficient transparency for reproducibility.

W3: The benchmark and Phase-I experiments appear limited to univariate settings, whereas many real-world TSAD applications involve multivariate signals with exogenous features. Recent TS-LLM and TS-VLM efforts (e.g., ChatTS, ChatTime, VLM-based TSAD) have begun addressing such multivariate reasoning.

### Questions
Q1: Did you evaluate AXIS’s explanations using multiple LLM judges and report inter-judge agreement? Were any controls applied to mitigate position, verbosity, or stylistic biases—such as randomized output order or style-normalized prompts?

Q2: What are the inter-rater agreement statistics (e.g., Fleiss’ κ or Kendall’s W) for the 280 annotated questionnaires? How were the human experts recruited and compensated, and what were their domain backgrounds? (Appendix F could provide more transparency here.)

Q3: How does the step-aligned hint extend to multivariate time-series inputs, considering dimensionality and cross-channel dependencies? Are there any preliminary results or ablations on multivariate benchmarks to demonstrate scalability?

Q4: Could you report the average token counts and generation latencies per question type and time-window size? This would help clarify the computational overhead of AXIS relative to simpler baselines.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents AXIS, a framework for semantic time-series anomaly explanation that conditions a frozen LLM via three complementary hint pathways:
1. Symbolic Numeric hint.    
2. Context-Integrated, Step-Aligned hint  
3. Task-Prior hint. 

In the final prompt, $\tilde F$ and $\tilde H_{s:e}$ replace the embeddings of $K+(e-s)$ placeholder tokens, while the numeric string is injected verbatim into the text; the LLM then generates an explanation $y$. Training is two-stage: (i) pretrain $f_\theta$ with a joint objective (masked reconstruction + anomaly classification) and freeze it; (ii) optimize only the hint tuner (including $M$ and $P_{\text{fix}}$) with a next-token prediction loss, keeping the LLM and encoder frozen. The paper also builds a paired-window, explanation-oriented benchmark with normal/abnormal pairs and multi-format questions (MC/True-False/Open-Ended) plus human/LLM judgments.

### Strengths
- Clear problem reformulation. Recasts TSAD from point-wise scoring to a window-level explanation problem, formalizing $G:(x_{1:T},q,s,e)\mapsto y$, which aligns with practical interpretability needs.
- Frozen-LLM practicality. All hints operate with a frozen backbone LLM, lowering deployment cost and risk.
- Clean unification interface. $\tilde H_{s:e}$, $\tilde F$, and the numeric string are fused in a single prompt, keeping the injection interface modular and reusable.
- Paired controls and multi-format evaluation. The benchmark includes normal/abnormal pairs and multiple question formats, plus human/CD-style summaries—useful for qualitative diagnosis.

### Weaknesses
1. Math/training under-specification (reproducibility risk).  
    The main text omits explicit forms and weights for the pretraining joint loss; Stage II is only described as NTP. The paper should state, for example,  
$$  
    L_{\text{pretrain}}=L_{\text{recon}}+\lambda,L_{\text{cls}},  
    $$  with $\lambda$ values, and define $L_{\text{NTP}}$ precisely. Dimensional alignment for prototype attention is incomplete (exact shape of $W_q$; whether multi-head is used; scaling/normalization). Currently $\operatorname{Attn}(Q,K,V)$ inputs are given without implementation details.
    
2. Evidence chain leans on the self-built benchmark.  
    Core results are on synthetic/self-constructed data; statements like “comparable on real TSAD datasets” are relegated to the appendix without prominent tables and significance/variance reporting. Public datasets with explanation-quality metrics should appear in the main paper.
    
3. LLM-as-a-Judge dependence and reporting.  
    Although CD plots are reported, there is no systematic inter-rater reliability across judges/models. The text mentions removing a baseline due to very poor performance, which risks selection bias; either keep it with a discussion of why it fails or re-run under controlled, equal-resource settings.
    
4. Uneven ablation interpretation.  
    In the ablation, the context hint yields minimal gain on Multiple Choice (e.g., $4.19\to4.09$, $-0.10$), yet has a large impact on True/False (e.g., $3.65\to2.44$). The paper does not analyze why different formats exhibit such divergent sensitivities (e.g., distractor design, window length, prototype size).
    
5. Missing equal-resource comparisons to strong baselines.  
    Direct, matched-budget comparisons to alternative routes—image-VLM, ChatTS-style multimodal alignment, Time-LLM-style reprogramming—are limited or absent in the main text, hindering external validity claims.

### Questions
1. Training objectives and hyperparameters.  
    Please include explicit formulas/weights for the two stages (masking strategy, classification head details), and list key hyperparameters for Stage-II NTP (LR/steps/batch sizes) plus stabilization tricks.
    
2. Prototype bank and attention implementation.  
    What are $P$ (prototype count), initialization/regularization for $M$, and the exact shape of $W_q$? Do you use multi-head attention with scaled dot-product? How are head count and $d_h$ matched?
    
3. Question-type sensitivity.  
    Why is the context hint nearly useless for MC but critical for TF? Can you ablate distractor quality, window length, and prototype size to produce sensitivity curves per question type?
    
4. External benchmarks and real-world cases.  
    Can you report explanation quality (blind human evaluation + reliability coefficients) on 2–3 public TSAD datasets and provide at least one real operational log case study (with cost/latency)?
    
5. Judge robustness.  
    Beyond the current judge, can you add a second LLM-judge and a human subset, reporting Kendall/Spearman correlations and statistical significance?

### Soundness
2

### Presentation
2

### Contribution
2

# TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems

- Avg Score: 5.33
- Decision: Accept (Oral)
- Scores: 2, 8, 6

## Abstract
Evaluation of Text to Speech (TTS) systems is challenging and resource-intensive. Subjective metrics such as Mean Opinion Score (MOS) are not easily comparable between works. Objective metrics are frequently used, but rarely validated against subjective ones. Both kinds of metrics are challenged by recent TTS systems capable of producing synthetic speech indistinguishable from real speech. In this work, we introduce Text to Speech Distribution Score 2 (TTSDS2), a more robust and improved version of TTSDS. Across a range of domains and languages, it is the only one out of 16 compared metrics to correlate with a Spearman correlation above 0.50 for every domain and subjective score evaluated. We also release a range of resources for evaluating synthetic speech close to real speech: A dataset with over 11,000 subjective opinion score ratings; a pipeline for recreating a multilingual test dataset to avoid data leakage; and a benchmark for TTS in 14 languages.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces TTSDS2, a comprehensive distributional metric for evaluating text-to-speech (TTS) quality. TTSDS2 integrates four key quality dimensions—generic quality, speaker similarity, prosody, and intelligibility—and, for each dimension, leverages multiple embedding models to compute the Wasserstein distance between embeddings of synthesized and reference speech. To assess the robustness of the metric, the authors compile a diverse evaluation dataset encompassing read speech, noisy speech, YouTube speech, and children’s speech, and benchmark 20 TTS systems using both human (MOS) and automatic metrics. Experimental results demonstrate that TTSDS2 consistently correlates strongly with human judgments across domains. The authors also provide a continually updatable evaluation pipeline designed to mitigate data leakage and maintain a reliable benchmark over time.

### Strengths
- **Comprehensive and Diverse Evaluation:**
    
    The paper introduces a test set with human annotations covering four diverse domains—read speech, noisy speech, YouTube speech, and children’s speech. Both the dataset and the evaluation are publicly released, enabling future reproducibility and comparison. The study also benchmarks a wide range of baseline TTS models and compares against multiple existing quality metrics, ensuring a thorough evaluation.
    
- **Strong Correlation with Human Judgments:**
    
    TTSDS2 demonstrates consistently high correlation with human perceptual ratings across different speech domains, confirming its robustness and reliability as an automatic TTS quality metric.
    
- **Continually Updatable Benchmark:**
    
    The proposed benchmark pipeline supports continual updates while preventing data leakage, which makes it a valuable and sustainable resource for future TTS research and model evaluation.

### Weaknesses
- **Poor Writing Quality and Structure:**
    
    While the main contribution of the paper is clear, the overall writing is repetitive and poorly structured, particularly in Section 1. 
    
- **Questionable Gaussian Assumption in Metric Design:**
    
    TTSDS2 assumes that the speech embeddings follow a multivariate Gaussian distribution when computing the Wasserstein distance. However, this assumption may not hold in practice, and the authors provide no empirical justification for it. A more principled alternative would be to employ an optimal transport algorithm to estimate the Wasserstein distance directly from the embedding distributions.
    
- **Unconvincing Multilingual Evaluation:**
    
    The paper’s claim of multilingual validity for TTSDS2 is not sufficiently supported. To substantiate the multilingual generalization of the proposed metric, additional human evaluations across languages are necessary.
    
- **Potential Annotation Bias:**
    
    Each annotator is assigned to only one dataset, which could introduce domain-specific bias and limit cross-domain consistency in the human evaluation results.

### Questions
- **Motivation for Distributional Metric:**
    
    What is the underlying intuition or theoretical justification for why a *distributional* metric should better reflect human perception compared to point-wise measures?
    
- **Averaging Strategy Across Dimensions:**
    
    The paper averages scores across different embedding models within each dimension, and then across the four dimensions. Have the authors explored using weighted combinations or learned weights to better capture the relative importance of each dimension?
    
- **Speaker Similarity Performance:**
    
    The speaker similarity metrics consistently rank as the second-best in correlation with human judgments. Can the authors provide an explanation or analysis for why this particular dimension performs slightly worse than the others?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces TTSDS2, a factorized, distributional objective metric for evaluating modern TTS systems. It measures similarity to real speech across multiple perceptual dimensions (generic, speaker, prosody, intelligibility) using Wasserstein distance over model-derived embeddings. The authors evaluate 20 recent open-weight TTS models across four domains and show that TTSDS2 is the only metric achieving Spearman > 0.5 consistently against MOS/CMOS/SMOS. The paper also provides a multilingual benchmark (14 languages) and an automated pipeline for recurring evaluation. Overall, I believe the work is timely, well-executed, and useful.

### Strengths
* The problem is well-motivated: subjective evaluation is expensive and rapidly becoming insufficient as systems approach human quality. The factorized distributional approach is intuitive and gives interpretable sub-scores.

* The empirical evaluation is broad, covering 20 recent models and multiple domains (clean, noisy, wild, children).

* Demonstrating consistent correlation >0.5 with human ratings across all domains is compelling.

* Benchmark and pipeline release are valuable to the community and likely to be widely used.

* Writing is clear and situates the work well within the literature.

### Weaknesses
* A more principled justification or ablation is warranted as the choice of feature set can appears somewhat tuned for correlation.
* Figure 3 is not referenced anywhere in text and the caption is vague.
* The multilingual evaluation lacks validation with human preference tests (e.g., CMOS / MUSHRA), making it difficult to verify whether TTSDS2 maintains reliability beyond English.

### Questions
Are there clear failure cases where TTSDS2 disagrees with MOS/CMOS? If so, what characteristics do these samples/systems share?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces TTSDS2, a framework for evaluating text-to-speech (TTS) systems. While TTS has advanced rapidly, existing evaluation metrics often fail to capture the human-level quality of recent systems (particularly in multilingual contexts). Moreover, subjective evaluations such as MOS are difficult to compare across systems due to differing evaluators and datasets. TTSDS2 addresses these issues by assessing how closely the distributions of various perceptual factors (e.g., speaker identity, prosody, intelligibility) in synthetic speech resemble those in real speech, relative to noise. Experimental results demonstrate that TTSDS2 correlates strongly with human listening tests and consistently outperforms other open-source objective metrics.

### Strengths
- Comprehensive comparison against numerous evaluation metrics.
- Evaluation across 20 recent TTS systems and 14 languages.
- Diverse and well-structured test sets covering clean, noisy, wild, and children’s speech domains.
- Offers a promising, scalable solution for objective TTS evaluation.
- Shows clear and consistent alignment with subjective (human) evaluation results.

### Weaknesses
- The overall methodology closely resembles the original TTSDS, with limited conceptual novelty.
- The paper averages multiple perceptual factor scores into a single TTSDS2 score but provides no justification for using equal weighting.
- The selection criteria for the feature sets within each factor are not clearly explained or motivated.

### Questions
- Why do the authors use *noise* samples as the negative anchor? Wouldn’t failed or unnatural TTS outputs serve as a more realistic negative baseline?
- How does the inference or computation time of TTSDS2 compare to other objective evaluation metrics?
- What is the rationale behind averaging all factor scores equally to form the final TTSDS2 score? Are all perceptual dimensions equally important for perceived quality?
- Could the authors report the correlation of each individual factor (e.g., prosody, intelligibility) with mean opinion scores to clarify which aspects contribute most to perceptual alignment?

### Soundness
3

### Presentation
3

### Contribution
3

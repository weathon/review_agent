# LAMDA: A Longitudinal Android Malware Benchmark for Concept Drift Analysis

- Decision: Accept (Poster)
- Scores: 8, 6, 6

## Abstract
Machine learning (ML)-based malware detection systems often fail to account for the dynamic nature of real-world training and test data distributions. In practice, these distributions evolve due to frequent changes in the Android ecosystem, adversarial development of new malware families, and the continuous emergence of both benign and malicious applications. Prior studies have shown that such concept drift—distributional shifts in benign and malicious samples, leads to significant degradation in detection performance over time. Despite the practical importance of this issue, existing datasets are often outdated and limited in temporal scope, diversity of malware families, and sample scale, making them insufficient for the systematic evaluation of concept drift in malware detection.

To address this gap, we present LAMDA, the largest and most temporally diverse Android malware benchmark to date, designed specifically for concept drift analysis. LAMDA spans 12 years (2013–2025, excluding 2015), includes over 1 million samples (approximately 37\% labeled as malware), and covers 1,380 malware families and 150,000 singleton samples, reflecting the natural distribution and evolution of real-world Android applications. We empirically demonstrate LAMDA's utility by quantifying the performance degradation of standard ML models over time and analyzing feature stability across years. As the most comprehensive Android malware dataset to date, LAMDA enables in-depth research into temporal drift, generalization, explainability, and evolving detection challenges.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper introduces LAMDA, a large-scale Android malware dataset spanning 12 years (2013–2025, excluding 2015) with over 1 million APK samples, designed specifically to study concept drift in malware detection. The dataset comprises approximately 37% malware samples across 1,380 families and includes static Drebin features. The authors empirically demonstrate performance degradation of standard ML models over time and analyze feature stability, providing a temporal benchmark substantially larger and more diverse than existing datasets. The paper includes comprehensive drift analysis using multiple techniques including Jeffreys divergence, t-SNE visualization, SHAP-based explanations, and label drift analysis.

### Strengths
	Scale and Temporal Scope: Over 1 million samples across 12 years with 1,380 families and 150K singleton samples provide unprecedented temporal coverage and diversity for Android malware research. This addresses a genuine gap in existing datasets. 
	Comprehensive Drift Analysis: The multi-faceted approach (supervised learning degradation, feature distribution shifts via Jeffreys divergence, feature stability scores, SHAP-based explanation drift, label drift) provides rich evidence for concept drift. The integration of multiple complementary methods strengthens the analysis. 
	Reproducibility and Scalability: Publication of feature matrices, variance threshold objects, and code supports reproducibility. The design enables extensibility to new samples, which is valuable for long-term use. 
	Rigorous Experimental Validation: The comparison between LAMDA and APIGraph on identical evaluation protocols effectively demonstrates that LAMDA exhibits stronger, more realistic drift. High standard deviations in LAMDA results versus APIGraph's stability support the claim of pronounced drift.

### Weaknesses
Unclear Scan Consistency： The paper does not specify whether VirusTotal labels were obtained from single-pass or repeated scans. Since detection outcomes can vary across rescans, this ambiguity may introduce label inconsistency. 
 
Lack of Intra-Sample Drift Analysis： The study analyzes global and family-level drift but does not consider intra-sample temporal variation—how the same APK’s features might change across time. Such analysis could better capture longitudinal behavior shifts. 
 
Static Feature Limitation： LAMDA focuses exclusively on Drebin-style static features. While this ensures comparability, it may overlook runtime or dynamic behaviors that evolve differently, slightly limiting ecological completeness.

### Questions
Collection Procedure and Label Stability: During dataset construction, were APKs scanned once or multiple times on VirusTotal? If repeated scans occurred, how were temporal discrepancies in detection counts handled—by selecting the earliest, latest, or majority label? Clarifying this would help assess label stability across the 12-year span. 
 
Intra-Sample Temporal Drift: Has the team examined how features or VirusTotal labels for the same APK hash change across years? This could quantify intra-sample drift and distinguish it from population-level concept drift. 
 
Dynamic and Hybrid Features: Given the exclusive use of static Drebin features, do the authors plan to include dynamic runtime features (e.g., API invocation traces, network behaviors) or hybrid representations in future LAMDA versions? This would enrich longitudinal analysis and reflect real-world adaptive threats. 
 
Temporal Label Validation: Considering that VirusTotal engines evolve over time, did the authors fix specific engine versions or cross-engine consensus thresholds to mitigate version-induced label drift?

### Soundness
3

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
2

### Summary
The authors present LAMDA, a malware dataset that spans 12 years and therefore is aimed at capturing classifier drop in performance due to representational drift.

### Strengths
This work is in an area that is now not near my current area of research, thus my lower confidence score.

The dataset is large and to the best of my knowledge the longest longitudinal malware dataset collected to date. The analysis is very thorough.

### Weaknesses
See questions

### Questions
In Figure (2) for LAMDA, could the authors explain why there is a large performance drop in 2017 and 2018?

Drebin-style features are rather old (from 2014), could the authors support the choice for this feature set? 

In Table 4 in the appendix, it seems like there are extremely low malware samples from 2023-2025 compared to the train set. This seems to coincide in Figure (2) LAMDA with a very big performance drop. I wonder if the authors have any comments on this? It seems strange to keep especially the years of 2024 and 2025 given the huge class imbalance.

### Soundness
2

### Presentation
2

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
The authors introduce LAMDA, a temporally diverse dataset meticulously crafted to tackle the challenges of concept drift in Android malware detection. This benchmark spans 12 years and includes over 1 million samples. Additionally, the authors assess state-of-the-art concept drift adaptation methods, revealing their limitations when applied to LAMDA. This highlights the urgent need for more robust approaches in the field.

### Strengths
The paper is well-organized.
The research topic is significant.
The experiments are sufficient.

### Weaknesses
It lacks a clear comparison with relevant datasets.
It lacks specific guidance for future work.

### Questions
1. It is suggested  to conduct a tabular comparison of the existing data to display information such as the number of malware samples, family types, and distribution over the years. This will clarify the innovative aspects of this study.

2. Has the data been deduplicated? Given that many malware samples exhibit identical features at the characteristic level (even if their hash codes differ), it is crucial to know if any deduplication efforts have been made to ensure diversity in software feature.

3. Please provide feasible research directions for future concept drift adaptation work based on the results of your dataset collection. For instance, does the dataset exhibit characteristics that differ from other datasets, making concept drift adaptation more challenging? While it is still recommended to continuously expand the dataset, simply increasing the amount of data is not sufficient for innovation. It is also necessary to introduce fresh perspectives and methodologies.

4. I am particularly curious about the decision to submit this work to ICLR instead of a conference or journal focused on software engineering or security.

### Soundness
3

### Presentation
3

### Contribution
3

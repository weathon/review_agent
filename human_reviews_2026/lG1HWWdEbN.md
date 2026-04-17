# VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations

- Decision: Accept (Poster)
- Scores: 2, 6, 4, 8

## Abstract
Visualization, a domain-specific yet widely used form of imagery, is an effective way to turn complex datasets into intuitive insights, and its value depends on whether data are faithfully represented, clearly communicated, and aesthetically designed. However, evaluating visualization quality is challenging: unlike natural images, it requires simultaneous judgment across data encoding accuracy, information expressiveness, and visual aesthetics. Although multimodal large language models (MLLMs) have shown promising performance in aesthetic assessment of natural images, no systematic benchmark exists for measuring their capabilities in evaluating visualizations. To address this, we propose VisJudge-Bench, the first comprehensive benchmark for evaluating MLLMs' performance in assessing visualization aesthetics and quality. It contains 3,090 expert-annotated samples from real-world scenarios, covering single visualizations, multiple visualizations, and dashboards across 32 chart types. Systematic testing on this benchmark reveals that even the most advanced MLLMs (such as GPT-5) still exhibit significant gaps compared to human experts in judgment, with a Mean Absolute Error (MAE) of 0.553 and a correlation with human ratings of only 0.428. To address this issue, we propose VisJudge, a model specifically designed for visualization aesthetics and quality assessment. Experimental results demonstrate that VisJudge significantly narrows the gap with human judgment, reducing the MAE to 0.421 (a 23.9% reduction) and increasing the consistency with human experts to 0.687 (a 60.5% improvement) compared to GPT-5. The benchmark is available at https://github.com/HKUSTDial/VisJudgeBench.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents VISJUDGE-BENCH, a large-scale, expert-annotated benchmark for evaluating the aesthetics and quality of data visualizations using multimodal large language models (MLLMs). The dataset covers 3,090 samples across a diverse set of visualization types and is accompanied by a multi-dimensional evaluation framework (“Fidelity-Expressiveness-Aesthetics”). The authors systematically benchmark several MLLMs and introduce VISJUDGE, a fine-tuned model that shows improved alignment with human expert ratings.

### Strengths
Addresses an Important Gap: The paper tackles the underexplored problem of automated visualization quality assessment, providing a resource that could catalyze further research in this area.
Comprehensive Dataset: VISJUDGE-BENCH is notable for its scale, diversity (single/multi/dashboard visualizations), and rigorous annotation pipeline.
Systematic Evaluation: The authors benchmark multiple MLLMs and provide detailed analysis of their strengths and weaknesses across different evaluation dimensions.
Open Resources: The dataset and benchmark are made available to the community, supporting reproducibility and future work.

### Weaknesses
Data Fidelity Unclear: The paper emphasizes “fidelity” as a core evaluation dimension, but it is not clear how this is assessed given that the source data for the visualizations is not available. Without access to the underlying data, it is difficult to objectively judge whether a visualization is faithful or misleading.
Limited Dataset Benchmarking: As a dataset paper, there is insufficient analysis of the final dataset’s strengths and limitations. There is little discussion of what types of visualizations, design flaws, or domains are well-represented or underrepresented, and how annotation quality varies across the dataset.
Scope of Contributions: Beyond dataset curation and MLLM fine-tuning, the paper’s contributions are somewhat incremental. The model fine-tuning, while useful, is marginal given the rapid progress in adapting MLLMs to new domains.
Generalization and Utility: The paper does not sufficiently characterize what the dataset enables (and what it does not), nor does it discuss how VISJUDGE or the benchmark would perform in real-world or out-of-distribution scenarios.

### Questions
How is data fidelity assessed without access to the source data? Are there proxies or heuristics used, and what are their limitations?
Can you provide a more systematic benchmarking of the dataset’s coverage, strengths, and weaknesses? For example, are certain chart types, styles, or domains underrepresented?
Did you analyze annotation quality and agreement across different annotator backgrounds or regions?
What are the practical limitations of using VISJUDGE-BENCH for training or evaluation in real-world visualization systems?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a MLLM benchmarking framework called VISJUDGE-BENCH for visualization quality assessment, aiming to address the notable shortcomings of existing models in evaluating data fidelity, information expressiveness, and aesthetic quality. The research team constructed a benchmark dataset containing 3,090 expert-annotated samples, covering three major categories (single visualizations, multiple visualizations, and dashboards) across 32 chart types. Through systematic testing, it was found that the current state-of-the-art MLLMs exhibit significant biases when assessing visualization quality. To bridge this gap, the research team developed the VISJUDGE model, which reduces the Mean Absolute Error (MAE) to 0.442 (a 19.8% reduction) and increases the correlation with human expert ratings to 0.681 (a 58.7% improvement). This study provides a standardized framework and optimization pathway for the development of automated visualization evaluation technology.

### Strengths
Originality: It proposes VISJUDGE-BENCH, the first MLLM visualization evaluation benchmark covering the three dimensions of "Data Fidelity-Expressiveness-Aesthetics", filling the gap where existing benchmarks only focus on a single dimension. Through GRPO reinforcement learning and LoRA fine-tuning, it achieves significant alignment between MLLMs and human experts in visualization evaluation for the first time.
Quality: The benchmark construction undergoes rigorous multi-stage screening and multi-level quality control; the experimental design adopts a unified environment, uses multiple metrics, and covers scenarios of different complexities, ensuring data reliability.
Clarity: The paper uses figures and diagrams to support its arguments, with clear logic and presentatio.
Significance: It addresses the issue of the lack of standardized benchmarks for MLLM visualization evaluation, providing a benchmark for subsequent model optimization.

### Weaknesses
The paper only uses 3 annotators, which may pose a risk of concentrated subjective bias, and the elimination of abnormal scores may lead to insufficient valid data volume.
The 7 MLLMs tested in the experiment do not cover gradient comparison of models with different parameter scales, and the only open-source model included is Qwen2.5-VL-7B, making it difficult to verify the adaptability of VISJUDGE's fine-tuning strategy and the universality of the benchmark in the open-source ecosystem.

### Questions
The specific scoring criteria are only presented in the appendix, and some representative scoring criteria could be selected and incorporated into the main text.
All 3,090 samples of VISJUDGE-BENCH are static visualizations, and it is suggested to include dynamic samples to support the measurement of MLLM evaluation capabilities in dynamic scenarios.

### Soundness
2

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
3

### Summary
VisJudge-Bench introduces a large-scale benchmark for evaluating multimodal large language models (MLLMs) on visualization aesthetics and quality assessment. It contains expert-annotated samples across diverse chart types and visualization settings. Experimental results show that existing MLLMs, including GPT-5, still fall short of human performance. To mitigate this, the authors propose VisJudge, a model trained via the GRPO method, which achieves substantial improvements in accuracy and alignment with human judgments.

### Strengths
1. The topic is interesting and the task is practical in real-world applications.
2. The authors constructed VISJUDGE-BENCH, a comprehensive benchmark based on the principles of Fidelity, Expressiveness, and Aesthetics to evaluate MLLMs’ visualization assessment capabilities.
3. They systematically evaluated representative MLLMs and found significant gaps compared with human expert standards.

### Weaknesses
1. Annotators are mainly presented with specific questions and asked to do scoring tasks. Are they asked to do enough free-form critique? It may be beneficial to diversity and generalizability.

2. While the construction of the benchmark dataset is well illustrated, I am very confused about the use of it. In Section 4, authors claimed: "To validate VisJudge-Bench as an effective training resource...". I am confused with the authors' purpose.

(1) Given that its called 'VisJudge-Bench' and referred as a benchmark in prior sections, as well as its tiny data size (only ~3k images), it is naturally a benchmark. However, authors further split it into train/test set and fintuned a MLLM on it. What is the purpose of the experiment?

(2) Training a MLLM with such a tiny dataset and testing it on the dataset from the same distribution does not make much sense. The model can easily overfit and achieve good results with the certain data distribution, therefore the results are not convincing. Generalizability is a big concern. 

In summary, the experiment part is confusing. As a benchmark paper, we may expect a comprehensive evaluation of MLLMs on all of the
collected data, instead of an experiment where the benchmark data is split into two parts for training and testing, which does not convey much useful information.

Overall the topic is interesting and the construction of dataset can contribute to related studies. However, the organization of the paper is confusing. The whole experiment section is a bit off. I may accept it as a benchmark paper but not as a dataset one. The current version is not a solid ICLR submission.

### Questions
1. Annotators are mainly presented with specific questions and asked to do scoring tasks. Are they asked to do enough free-form critique? It may be beneficial to diversity and generalizability.

2. What is the purpose of the experiment part?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces VISJUDGE-BENCH, a comprehensive benchmark for evaluating the capabilities of Multimodal Large Language Models (MLLMs) in assessing the quality and aesthetics of data visualizations. The authors propose a "Fidelity-Expressiveness-Aesthetics" framework, operationalized into six measurable dimensions, to create a dataset of 3,090 expert-annotated samples. The work systematically benchmarks several state-of-the-art MLLMs, revealing significant gaps compared to human expert judgment, particularly in subjective aesthetic evaluations. To address this, the paper also presents VISJUDGE, a fine-tuned model that demonstrates substantially improved alignment with human ratings. The paper is well-written, the experiments are thorough, and the dataset is a valuable contribution to both the machine learning and visualization communities. However, the work rests on the critical assumption that a singular ground truth for visualization quality can be established, a premise that may oversimplify the inherent subjectivity and context-dependency of human perception, which is the main point of concern.

### Strengths
1. **Timely problem:**  
   The paper addresses a relevant and underexplored challenge—evaluating MLLMs on data visualizations, which differ significantly from general images and remain difficult for current models.

2. **Well-motivated framework:**  
   The Fidelity–Expressiveness–Aesthetics framework is grounded in visualization theory and provides a clearer structure than using a single quality score.

3. **Solid experiments:**  
   The benchmarking results across multiple MLLMs offer useful insights into current model strengths, weaknesses, and issues such as score inflation.

4. **Reproducibility efforts:**  
   The release of data, code, and detailed documentation supports replicability and future work.

5. **Clear presentation:**  
   The paper is well-written, logically structured, and supported by clear visuals and appendices.

### Weaknesses
1. Fundamental Assumption of a Singular "Ground Truth": The primary and most significant weakness of this work stems from a fundamental assumption that a singular, objective "ground truth" for visualization quality exists and can be established. This is particularly contentious for subjective dimensions like Aesthetics and even Expressiveness, where human preferences are known to be diverse, context-dependent, and often lack a single consensus.
The paper's methodology—collecting scores from three crowdworkers and then using expert review to resolve disagreements—effectively enforces a consensus rather than acknowledging or modeling the inherent diversity in human perception. The authors do not sufficiently discuss the possibility that disagreement among annotators may not be a sign of low-quality annotation but rather a reflection of genuine, valid differences in interpretation and preference within the human population. This core assumption is critical, as its potential invalidity could undermine the significance of the benchmark's "ground truth" and, consequently, the meaning of the entire evaluation framework. Regrettably, because the potential invalidity of this core assumption would render all subsequent experiments meaningless, this is the main reason I cannot give a higher score, despite the authors' thorough and professional work.
2. Domain-Specificity and Audience Dependence: A compelling, concrete example illustrates the issue above. In Figure 2 and Figure 21, the authors present a "layered heatmap with candlesticks" as an example of a poor visualization that MLLMs incorrectly rate highly. The authors claim it "does not accurately present the data," is "chaotic," and "impossible to decipher." However, this visualization is a standard and highly effective tool in the cryptocurrency trading domain, known as a "liquidation heatmap," which displays price action (candlesticks) alongside market liquidation levels (heatmap) (see, for example, https://www.coinglass.com/pro/futures/LiquidationHeatMap). To a domain expert (e.g., a trader), the chart is professional, intuitive, and information-dense. To a generalist audience, it is indeed confusing.
This example powerfully illustrates the central weakness: visualization quality is often domain-specific and audience-dependent. A "ground truth" established by generalist annotators may not be valid for specialized audiences. I do not criticize the authors for their lack of specific investment knowledge; rather, I use this to emphasize that finding a universal consensus on visualization quality is difficult, if not impossible. A deeper discussion of this challenge is needed.
3. Insufficient Analysis of Complex Visualizations: The results clearly show that model performance degrades significantly on more complex visualizations like multi-view charts and dashboards compared to single-view charts. While the paper notes this drop, it lacks a deep, quantitative analysis of why this occurs. A more detailed investigation into the specific failure modes (e.g., assessing inter-chart relationships, handling information density) would have been highly valuable, especially since evaluating such complex artifacts is a major research gap that this work laudably aims to address. This feels like a missed opportunity to generate deeper insights.

### Questions
- **Modeling Disagreement vs. Consensus:**  
  The benchmark uses a single consolidated ground truth score. Given the known subjectivity in aesthetic and design judgments, could the authors share whether inter-annotator disagreement was analyzed before expert consolidation? It may be helpful to discuss whether such disagreement could carry meaningful signal rather than being treated purely as noise.

- **Potential for Distribution-Based Evaluation:**  
  Related to the above, have the authors considered representing human ratings as a distribution instead of a single averaged score? Capturing variance or multimodal preferences might provide a more realistic target for model alignment with diverse human judgments.

- **Role of Domain Expertise:**  
  Some visualizations (e.g., financial heatmaps with candlesticks) may be more interpretable to domain experts than to general audiences. Were annotators screened for domain familiarity, and do the authors foresee value in domain-specific subsets of the benchmark? It would be interesting to know how “ground truth” ratings might vary under domain-aware annotation.

- **Deeper Analysis of Multi-View Challenges:**  
  The performance drop on multi-view and dashboard visualizations is an important observation. A brief breakdown of typical failure modes—e.g., difficulty assessing inter-chart relationships vs. cumulative single-chart errors—would provide useful insight into where models struggle most and help guide future work.

### Soundness
3

### Presentation
4

### Contribution
3

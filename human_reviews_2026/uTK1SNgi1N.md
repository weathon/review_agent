# Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 2

## Abstract
Large time series models (LTMs) have emerged as powerful tools for universal forecasting, yet they often struggle with the inherent diversity and nonstationarity of real-world time series data, leading to an unsatisfactory trade-off between forecasting accuracy and generalization. Rather than continually finetuning new LTM instances for each domain, we propose a data-centric framework, time-series adaptive transformation optimization (TATO), that enables a single frozen pre-trained LTM to adapt to diverse downstream domains through an optimally configured transformation pipeline. Specifically, TATO constructs three representative types of transformations, including context slicing, scale normalization, and outlier correction, to help LTMs better align with target domain characteristics. To ensure robustness, we incorporate carefully selected time series augmentations and a two-stage ranking mechanism that filters out pipelines underperforming on specific metrics. Extensive experiments on state-of-the-art LTMs and widely used datasets demonstrate that TATO consistently and significantly improves domain-adaptive forecasting performance, achieving a maximum reduction in MSE of 65.4\% and an average reduction of 13.6\%. Moreover, TATO is highly efficient, typically completing optimization in under 2 minutes, making it practical for real-world deployment. The source code is available at https://github.com/thulab/TATO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Time-Series Adaptive Transformation Optimization (TATO), a framework designed to enhance the generalizability of frozen large time series models (LTMs) across diverse domains. Instead of retraining or creating new models, TATO automatically searches for empirically optimal transformation pipelines—comprising context slicing, scale normalization, and outlier correction—to adapt LTMs to new data distributions. Through a robust two-stage ranking mechanism, TATO ensures consistent performance across metrics and achieves substantial forecasting improvements, reducing MSE by up to 68.4% on standard benchmarks, all within a practical optimization time of under two minutes.

### Strengths
1.	This paper stands on a data manipulation perspective to improve the performance of time series forecasting, which is kind of novel. 
2.	TATO is evaluated on both time series foundation models and LLM-based models to demonstrate its universal applicability across various model architectures. 
3.	Performance gain brought by TATO is significant, reaching up to 68.4%.

### Weaknesses
1.	It is hard to understand what transformations are applied to the time series in Figure 1.
2.	TATO’s effectiveness heavily relies on a fixed set of nine transformation operators. The search space is manually defined, which may limit adaptability to unseen or highly domain-specific data types.
3.	The framework is presented primarily as an empirical optimization procedure, with little theoretical grounding on why certain transformations generalize well across domains or how the two-stage Pareto ranking ensures robustness.

### Questions
1.	Can TATO be applied to other time series tasks, such as imputation?
2.	What is the relationship between FrozenForecasting and Data2Model? It appears that the authors use them interchangeably.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an interesting paradigm that aligns time series data with a pretrained large time series model through transformation pipelines. By automatically selecting feasible pipelines via a two-stage ranking process, the proposed framework aims to enhance forecasting performance on target domains without requiring fine-tuning of the backbone models.

### Strengths
- The idea of leveraging transformation pipelines to align time series data with forecasting models is novel and promising, as it improves domain-adaptive forecasting performance while keeping the backbone models fixed.
- The proposed pipeline optimization process is efficient, typically requiring less than two minutes in most cases.
- Experimental results demonstrate that the proposed framework effectively boosts the performance of various backbone models on widely used benchmark datasets.

### Weaknesses
- Clarity and presentation:
  - The framework description lacks sufficient detail. For instance, how are individual trials sampled from the pipeline space? Providing pseudocode or an algorithmic overview would greatly improve readability and reproducibility.
  - It is mentioned that only a portion of the training samples are used in the TATO framework, but the selection strategy for these samples is unclear. A detailed explanation of how these samples are chosen from the full training set would strengthen the paper.
- Experimental analysis:
  - Table 2 reports runtime comparisons across different configurations, but it would be helpful to also include forecasting performance metrics in the same table to illustrate the efficiency–effectiveness tradeoff of TATO.
  - Although TATO is primarily proposed for large time series models, it would be interesting to investigate whether conventional models such as PatchTST can also benefit from TATO under cross-domain forecasting scenarios.
  - Section 4.5 compares LTMs tuned over eight datasets against TATO. However, it would be insightful to include results where the model is tuned on a *single* dataset (e.g., tuning Timer on ETTh1 with a comparable computational budget in terms of training time or samples) to better contextualize TATO’s performance.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
2

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
The paper proposes TATO, a framework to adapt data rather than models for time series forecasting. Instead of fine tuning large time series models (LTMs), TATO keeps them frozen and learns optimal preprocessing transformations to improve performance across domains. The method searches transformation pipelines including context slicing, scale normalization, and outlier correction using hyperparameter optimization and a two stage Pareto based ranking for robustness. Experiments on LTMs (e.g., Timer, Moirai, Chronos) and datasets (ETT, Electricity, Exchange, Traffic, Weather) show consistent improvements. This work introduces a practical FrozenForecasting paradigm for universal, efficient forecasting.

### Strengths
The paper presents an original paradigm that shifts the focus from model adaptation to data adaptation, introducing the concept of FrozenForecasting for large time series models. The proposed TATO framework is well designed, combining hyperparameter optimization and Pareto based ranking to ensure both robustness and efficiency. The methodology is clearly described with experiments across several state of the art LTMs and diverse datasets. The work provides some evidence of performance improvement with minimal computational overhead, highlighting its practicality for real world applications.

### Weaknesses
While the idea of adapting data instead of models is novel, the paper could benefit from a deeper theoretical justification of why certain transformations consistently enhance generalization across domains. The search space for transformation pipelines, though compact, may still be computationally demanding for larger datasets or longer horizons, and scalability analyses beyond 500 samples are limited. Some test-time adaptation methods for time-series forecasting were completely ignored by the authors, such as TAFAS (Kim et al. (2025).) [1], PETSA (Medeiros et al. (2025)) [2], DynaTTA (Grover & Etemad. (2025)) [3], I would suggest the authors to review these additional methods for the related works to strong contribution of the proposed TATO, and also compare with some of the methods if possible, because in the main comparisson the paper is missing comparisson with other methods. Additionally, the ablation results could include more quantitative discussion on how each operator contributes to generalization rather than overall MSE reduction.



[1] HyunGi Kim, Siwon Kim, Jisoo Mok, and Sungroh Yoon. Battling the non-stationarity in time
series forecasting via test-time adaptation. In AAAI, pp. 17868–17876, 2025. URL https:
//doi.org/10.1609/aaai.v39i17.33965.

[2] Heitor Medeiros, Hossein Sharifi-Noghabi, Gabriel Oliveira, and Saghar Irandoust. Accurate
parameter-efficient test-time adaptation for time series forecasting. In Second Workshop on Test-
Time Adaptation: Putting Updates to the Test!, 06 2025. doi: 10.48550/arXiv.2506.23424.

[3] Shivam Grover and Ali Etemad. Shift-aware test time adaptation and benchmarking for time-series
forecasting. In Second Workshop on Test-Time Adaptation: Putting Updates to the Test! at ICML
2025, 2025.

### Questions
1. How sensitive is the optimization process to the size and diversity of the sampled data? Would using fewer or noisier samples significantly affect the quality of the selected transformation pipeline?

2. Could the authors clarify how TATO generalizes to multivariate or irregularly sampled time series, where contextual or scale transformations may interact differently?

3. How does TATO compare empirically or conceptually with recent test time adaptation methods TAFAS, PETSA, DynaTTA?

4. Are there any insights into which specific transformations contribute most to improving generalization across domains?

5. Could TATO be combined with lightweight fine tuning or LoRA style adaptation to further improve performance while maintaining efficiency, and if so, what trade offs might emerge?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper is flawed due to deficient writing, a lack of crucial experimental comparisons against a fine-tuned LTM and other established time-series methods, and the absence of a required quantitative analysis on computational overhead.

### Strengths
The authors provide sufficient ablation studies and comparisons, which enhance the credibility of the findings.

### Weaknesses
1. The writing is deficient, like the transition from paragraph in line 55 to 57 feels abrupt; Section 2.3 is a seemingly extraneous point, confusing the reader and disrupting the narrative flow; Section 3.1 is repetitive because the same content has been discussed before; In figure 2, it seems like the order of 3 types of operators is reorderable, but it's said in the text that they are fixed, it's confusing which ones are reorderable; In Section 3.2.2, post-process operators are mentioned, but lacks a explaination of what exactly they are; Section 4.2 is repetitive too.
2. You have verified that your method yields better results than the baseline (without TATO) on a specific dataset. However, how does it compare to LTM that has been fine-tuned on the same dataset? A sole comparison between having TATO and not having TATO is insufficient to prove the efficacy of your method, as direct fine-tuning on the LTM might achieve superior results.
Furthermore, regarding the SOTA method, how much computational resource saving does it actually provide? You need to compare the overhead and computational resources required by your method against those needed for fine-tuning the LTM, rather than merely stating how little time your method requires.
3. As I understand it, TATO is merely a combination of several common transformation methods. Given this, there are many other established methods for processing time-series data—is your approach superior to them?
If your claimed advantage is superior generalization, then you must also provide evidence that other methods perform worse than yours in terms of generalization. It is not enough to simply state that these other methods are designed for "specific problems"; this claim needs to be supported by experiments.
4. I don't understand the point of section 4.5. What specific benefit of your method are you trying to demonstrate with this experiment? Furthermore, why are only two models used in this experiment? What about the others? 
"First, we finetune Timer-UTSD over all eight datasets above to get a single LTM that works well on all of them." How well does it work? Lacks experimental data.

### Questions
No

### Soundness
2

### Presentation
2

### Contribution
2

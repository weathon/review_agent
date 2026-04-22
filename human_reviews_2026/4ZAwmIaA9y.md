# GARLIC: Graph Attention-based Relational Learning of Multivariate Time Series in Intensive Care

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Healthcare data, such as Intensive Care Unit (ICU) records, comprise heterogeneous multivariate time series sampled at irregular intervals with pervasive missingness. However, clinical applications demand predictive models that are both accurate and interpretable. We present our Graph Attention-based Relational Learning for Intensive Care (GARLIC) model, a novel neural network architecture that imputes missing data through a learnable exponential-decay encoder, captures inter-sensor dependencies via time-lagged summary graphs, and fuses global patterns with cross-dimensional sequential attention. All attention weights and graph edges are learned end-to-end to serve as built-in observation-, signal-, and edge-level explanations. To reconcile auxiliary reconstruction and primary classification objectives, we developed an alternating decoupled optimization scheme that stabilizes training. On three ICU benchmarks (PhysioNet 2012 \& 2019, MIMIC-III), GARLIC sets the new state of the art in outcome prediction, significantly improving AUROC and AUPRC over best-performing baselines at comparable computational cost. Ablation studies confirm the contribution of each module, and feature-removal trials validate the fidelity of importance attribution through a monotonic performance drop (full > top 50\% > random 50\% > bottom 50\%). Real-time case studies demonstrate actionable risk warnings with transparent explanations, marking a significant advance toward accurate, explainable deep learning for irregularly sampled ICU time series data. Moreover, we demonstrated GARLIC's superiority in data imputation and classification on various time-series datasets beyond the ICU domain, showing its generalizability and applicability to broader tasks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a graph attention-based relational learning framework, termed GARLIC, designed for the classification of irregular multivariate time series in intensive care. The GARLIC model incorporates decay-based imputation, time-lagged graph learning, and cross-dimensional attention mechanisms to effectively address irregular missingness and capture complex inter-variable dependencies. Experimental evaluations on the P12, P19, and MIMIC-III medical datasets demonstrate that GARLIC outperforms existing baseline methods in classification performance.

### Strengths
1.	The paper is clearly written, with a well-structured and persuasive review of related work. The experimental analysis and discussion are comprehensive and insightful.

2.	The study focuses on the challenging task of irregular multivariate time series classification, effectively addressing both the modeling of irregular temporal patterns and the dependencies among variables. This contributes valuable insights to the development of methodologies in irregular time series classification.

3.	The integration of attention mechanisms to enhance interpretability in irregular time series classification represents a notable strength of the work.

### Weaknesses
1.	The study focuses solely on the intensive care medical domain, despite irregular multivariate time series also being prevalent in areas such as human activity recognition, mobility, and sensor data [1]. The model’s motivation and design appear not strongly tied to the medical scenario.

2.	The proposed GARLIC model largely follows the GRU-D approach for irregular pattern learning and employs common graph- or attention-based structures for variable dependencies modeling, limiting its novelty.

3.	While interpretability is emphasized, the analysis only identifies important features without clearly linking them to irregular patterns or inter-variable relationships. The rationale for selecting a 50% masking ratio is also insufficiently justified.


4.	The case study in Figure 3 is difficult to understand, relying heavily on medical terminology without citations, which may hinder comprehension for readers unfamiliar with the P12 and P19 datasets.

5.	The strategy of imputing irregular time series into regular ones is straightforward, but the baseline methods used for comparison do not include models designed for regular time series classification.



6.	The absence of released source code limits the reproducibility of experimental results.

[1] PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks. arXiv, 2025.

### Questions
1.	According to the experimental analysis in Reference [1], the Rocket algorithm for regular time series classification significantly outperforms several irregular time series classification baselines. Could the authors provide a comparative analysis of Rocket versus GARLIC under similar experimental settings as in Reference [1]?

2.	Reference [1] adopts the macro-averaged F1 score as an evaluation metric for imbalanced classification tasks. How do the strongest baseline methods in this paper, as well as Rocket, perform compared with GARLIC in terms of the F1 metric?

3.	Based on the dataset organization for irregular time series classification in Reference [1], could the authors clarify why the study focuses exclusively on the medical domain? Alternatively, how does GARLIC perform compared to Rocket across all benchmark datasets summarized in Reference [1]?

[1] PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks. arXiv, 2025.

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
4

### Summary
This paper investigates multivariate time series data from Intensive Care Units (ICUs) and addresses key challenges identified in prior research, including the inherent irregularity and heterogeneity of clinical measurements, as well as the limited interpretability of existing approaches. To tackle these challenges, the authors propose the Graph Attention-based Relational Learning for Intensive Care (GARLIC) model, which incorporates dedicated mechanisms to enhance analytic performance and provide integrated interpretability. The proposed model is empirically validated on three real-world ICU datasets, demonstrating its efficacy in improving predictive performance and providing meaningful interpretations.

### Strengths
**S1.** The paper identifies representative challenges associated with multivariate time series data from ICUs. The proposed GARLIC model is well designed to address these issues through (i) a decay-based imputation mechanism to handle irregular missingness, (ii) a self-attentive, time-lagged graph learning mechanism to capture spatial dependencies, and (iii) a cross-dimensional attention mechanism to integrate temporal signal embeddings. These design choices appear technically sound and well aligned with the stated objectives.

**S2.** The experimental evaluation is comprehensive, assessing the efficacy of GARLIC from multiple perspectives, including comparative performance against strong baselines, computational efficiency, ablation studies, sensitivity analyses, and evaluation on broader downstream tasks.

**S3.** The interpretability aspect of GARLIC is examined both quantitatively, through a perturbation-based evaluation strategy, and qualitatively, via representative case studies. These analyses effectively demonstrate the model’s potential to provide clinically meaningful insights alongside strong predictive performance.

### Weaknesses
**W1.** The identified challenges—irregular sampling, heterogeneous measurements, and limited interpretability—are not unique to ICU data but are generally characteristic of electronic health record (EHR) data. It would therefore be beneficial for the authors to clarify the rationale for restricting the study’s scope to ICU data. Specifically, it would strengthen the paper to discuss why the broader context of EHR analytics was not adopted as the primary focus, and how the proposed methods and findings relate to, or differ from, existing studies on general EHR data.

**W2.** The key architectural components integrated into GARLIC, including the Time-Lagged Graph Message Passing and Cross-Dimensional Sequential Attention mechanisms, appear to be largely drawn from prior research in this domain. As a result, the methodological novelty of the model seems limited. The paper would benefit from a more explicit articulation of how GARLIC advances beyond existing approaches and what its tangible contributions are beyond the combination of established techniques.

**W3.** Given that this work concerns ICU data analytics, a high-stakes and clinically sensitive application domain, it would be valuable for the authors to elaborate on the current stage or planned trajectory toward real-world deployment. A clearer description of practical considerations, such as model validation in live settings, potential integration with clinical workflows, and collaboration with medical practitioners, would significantly enhance the work’s translational relevance and credibility.

### Questions
Beyond W1-W3, I have the following questions:

**Q1.** In the performance–efficiency trade-off analysis presented in Appendix A.2.3, it is observed that the P19 dataset is used in Table 4 for the evaluation under varying time lag settings, whereas the P12 dataset is adopted for the other three varying configurations. It would be helpful to clarify the rationale behind these dataset-specific choices.

**Q2.** The main ablation study in Appendix D.1 is conducted on smaller datasets rather than the largest one, MIMIC-III. What is the underlying rationale for this choice? Performing ablation on the most representative or largest dataset would provide stronger evidence for the robustness and generalizability of the proposed model.

**Q3.** Regarding the imputation results, it is shown that the baseline mTAND outperforms GARLIC under high missing rates and performs comparably under low missing rates. This observation may suggest potential limitations of GARLIC in handling severe missingness. Further analysis or discussion explaining this behavior would be helpful for clarification.

### Soundness
3

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
This paper proposes a multivariate time series learning method for intensive care. The proposed method first impute the missing value in the time series by an exponential-decay mechanism, and then apply the attention mechanism across the feature and time dimensions to learn the final representations. A data attribution method is also proposed to estimate the contribution of features at different timestamps by propagating the attention weight back to the input multivariate time series.

### Strengths
- The proposed method not only achieves strong empirical performance but also demonstrates integrated interpretability, effectively addressing the needs of the intensive care problem under study.
- Comprehensive experiments, including baseline comparisons, ablation studies, sensitivity analysis, and case studies, are conducted to validate the effectiveness of the proposed method and substantiate its claims.

### Weaknesses
- The contribution of this paper is somewhat incremental. The exponential-decay mechanism , interleaved attention across time and feature dimensions, and estimate the importance of data by attention weights are common techniques in multivariate time series learning. The proposed method seems to be a combination of these existing techniques, lacking unique adaptations tailored to the specific problem being addressed.
- For Table 9, further clarification is needed regarding the dramatic performance drop when the GRU module is removed.
- The notations in the method section can be improved: 1) All the attention module is denoted by a “xxAtten” text, the author should more clear introduce the computation procedure. For example, what is the q,k,v of the SignalAttn, and how  $u_{k,t}$ is computed in Equation (6). 2) Equation (9) seems problematic, where the subscript t only appear in the right side.

### Questions
Please see the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces GARLIC, a model designed for ICU time-series data characterized by irregular sampling and substantial missing values. It consists of four main components:

•A decay-based imputation module for handling missing data.

•A time-lagged graph network that enables information exchange among signals.

•Signal and temporal attention layers, separated by a GRU, to highlight the most relevant signals and time points.

•An alternating training strategy that switches between reconstruction and prediction objectives.

The model is evaluated on the P12, P19, and MIMIC-III datasets, where it demonstrates superior AUROC and AUPRC performance compared to baseline methods. Additionally, the authors include an interpretability test, similar to ROAR, to assess whether the attention mechanisms emphasize clinically important features.

### Strengths
•Addresses a challenging and important issue in clinical data: irregular and missing time-series signals.

•The model design is clear and well-motivated, effectively combining graph reasoning with atteq1Qntion mechanisms.

•Evaluation is conducted across multiple datasets, demonstrating consistent performance.

•The inclusion of an interpretability test adds value by assessing whether the attention weights align with clinically meaningful features.

### Weaknesses
•The novelty feels limited; it’s mostly a combination of existing parts (decay imputation + graph + attention + alternating training). It’s more of an engineering improvement than a new concept.

•The data split setup might cause data leakage. For example, in MIMIC-III, they say they use “all available ICU data,” which can include info after the event, making prediction easier but unrealistic.

•There’s no patient-level or hospital-level split, which means the model might just memorize patterns from the same patients or hospitals instead of generalizing.

•The interpretability approach is weakly justified. You say “we redistribute the imputed contributions uniformly” across observed values — but this is just a heuristic and might distort the results.

•The benefit of alternating training seems small compared to its added complexity.

•They claim “energy-efficient communication and comparable runtime,” but there’s no detailed info on hardware, fairness, or batch sizes, so it’s hard to trust the runtime comparison.

•Calibration and clinical usefulness (like early prediction or alert timing) are not discussed, which is important for ICU tasks.

### Questions
•Did you make sure to avoid patient or hospital overlap between training and test sets?

•For MIMIC-III, since you use “all ICU data,” how do you ensure the model doesn’t use information collected after the outcome?

•How sensitive are your results to the uniform redistribution rule used in your interpretability part (Eq. 11)?

•Did you try a simpler baseline where you just combine decay imputation + GRU without the graph to see how much the graph actually helps?

•How does the model perform when using only the first 24 or 48 hours of data? This would test if it’s useful for early prediction.

•Are the edges in your learned graph constrained or reviewed for clinical meaning?

•Did you test robustness for different missingness rates or lag values (τ)?

### Soundness
3

### Presentation
3

### Contribution
2

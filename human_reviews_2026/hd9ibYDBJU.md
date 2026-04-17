# FACTS: A Future-Aided Causal Teacher-Student Framework for Multimodal Time Series Forecasting

- Decision: Reject
- Scores: 6, 4, 6

## Abstract
Traditional \emph{unimodal} time series forecasting models often perform unreliably in real-world applications because they fail to capture the underlying causal drivers of temporal change. Fortunately, auxiliary modalities can unveil these drivers, \textit{e.g.}, sky images capture the illumination conditions that govern solar power generation. However, the most informative \emph{future} auxiliary signals directly tied to the target time series are unavailable at inference, while integrating such data is further hindered by cross-modal heterogeneity and structural mismatch. To address these challenges, we propose FACTS, a Future-Aided Causal Teacher-Student framework for \emph{multimodal} time series forecasting. The teacher network, used only during training, leverages future auxiliary data to disentangle the causal responses underlying temporal dynamics, while the student network, trained solely on historical data, learns such causal knowledge via our proposed causal-perturbation contrastive distillation. To accommodate heterogeneous inputs, we design a bilinear orthogonal projector that efficiently converts high-dimensional auxiliary data into a compact series over time, allowing us to model both auxiliary data and time series via a unified bidirectional attention backbone. Furthermore, we devise a lag-aware fusion to align cross-modal signals within a tolerance window and apply random modality dropout to enhance student's robustness to modality missingness. Extensive experiments on benchmark datasets demonstrate that FACTS significantly outperforms state-of-the-art methods, achieving average improvements of 32.98\% in MSE and 22.25\% in MAE. Code is available at \url{https://anonymous.4open.science/r/FACTS-7F94}.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes FACTS, a Future-Aided Causal Teacher-Student framework for multimodal time series forecasting. The key idea is to leverage future auxiliary modalities (such as images and weather data) during training to uncover causal factors that drive temporal dynamics, and distill this knowledge into a student network that uses only historical data at inference.

The method introduces three main components: (1) a Bilinear Orthogonal Projector (BOP) that converts high-dimensional auxiliary data into compact serialized time series; (2) a lag-aware multimodal fusion mechanism with random modality dropout to handle temporal misalignment and missing modalities; and (3) a Causal-Perturbation Contrastive Distillation (CPCD) objective that transfers causal knowledge from the teacher to the student. Experiments on four multimodal forecasting datasets show significant performance gains (around 33% lower MSE on average) over state-of-the-art baselines.

### Strengths
1. The proposed teacher-student design, which leverages future multimodal signals to facilitate causal representation learning while maintaining a purely historical input during inference, is both conceptually elegant and practically impactful. It effectively balances realism (no future data at test time) with enhanced learning through privileged information.

2. The model is comprehensively evaluated on several diverse and representative datasets, demonstrating the framework’s robustness and adaptability across various multimodal forecasting scenarios.

3. The experiments include thorough comparisons against a wide range of baselines, covering both unimodal and multimodal forecasting methods, thereby providing convincing evidence of the model’s superior performance and general applicability.

4. The paper conducts detailed ablation studies that isolate and analyze the contribution of each component—such as BOP, CPCD, and lag-aware fusion—clearly demonstrating their necessity and complementary effects within the overall framework.

### Weaknesses
see in questions.

### Questions
1. The paper does not clearly specify the training schedule of the teacher–student framework. Is the teacher network fully trained to convergence before the student distillation begins, or are the two networks optimized jointly in an alternating or end-to-end manner?

2. The Bilinear Orthogonal Projector (BOP) is primarily demonstrated on image data. Could the authors elaborate on its generality—specifically, whether BOP can be effectively adapted to other high-dimensional modalities such as text embeddings, audio spectrograms, or video streams? If not, what limitations might arise when extending it beyond visual inputs?

3. How significant is the computational overhead incurred by using both teacher and student networks, especially when compared to alternative approaches? Could the authors provide quantitative estimates or benchmarks to clarify this aspect?

4. The paper claims that random modality dropout enhances robustness when certain modalities are missing at inference time. Have the authors conducted experiments under more extreme or persistent missing-modality scenarios (e.g., complete loss of image data throughout inference)? Such evaluation would provide stronger evidence for the model’s robustness and practical deployability in real-world settings with unreliable sensors.

5. Is the ratio 𝜆 between the MSE loss and the CPCD loss a manually set hyperparameter or a learnable parameter? What is the value of this parameter, and how does varying 𝜆 affect the final model performance?

### Soundness
3

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
2

### Summary
The authors describe a method for inferring causal information related to time series by distilling a larger teacher model that does not require direct observations at test times. The main design choices of the architecture are to allow heterogeneous multimodal data to influence the time series prediction. The authors illustrate the performance of the method on a series of data sets and indicate that the proposed method outperforms the baselines.

### Strengths
The model empirically works very well and was tested on a variety of challenging datasets. 

The authors consider a challenging problem and use a variety of techniques to propose a solution.

### Weaknesses
There does not seem to be a cohesive underlying reasoning for the improvement in performance. The paper introduces a number of known techniques in combination with their particular problem to indicate improvements in performance. However, it’s not a unifying theme underlying all of the changes the authors include to the model.  

There do not seem to be any guarantees to ensure that the method is able to accurately disentangle the causal relationships, which is the main motivator of the paper.

### Questions
Did the authors consider any tests to see how well orthogonality is preserved during training? 

Are there any specific guarantees one can make on the performance under perturbations such as spurious correlations? It would be interesting to see if the loss can be analyzed such that one can understand the conditions under which the model will be robust to such artifacts. 

Under what conditions on the data can the model disentangle the underlying causal factors? Maybe analyzing this could be helpful to making stronger claims regarding the expected model performance.

### Soundness
3

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
3

### Summary
The paper proposes FACTS, a teacher-student framework designed to address multimodal time series forecasting by leveraging auxiliary modalities to capture causal drivers of temporal dynamics. The core innovation lies in using a teacher network with access to future auxiliary data during training to distill causal knowledge into a deployable student network that uses only historical data. The authors test FACTS on four datasets and report performance improvements over baselines.

### Strengths
1. Articulates why unimodal forecasting fails and why auxiliary modalities capture critical causal drivers.
2. Using perturbed future auxiliary data as negative samples in contrastive distillation is intuitive and aligns with causal learning principles.
3. Compares against methods spanning unimodal, LLM-based, and multimodal approaches with systematic ablations.
4. Outperforms baselines on various datasets with reduced standard deviations, suggesting robustness.

### Weaknesses
1. Teacher-student distillation, contrastive learning, and bilinear factorization are well-established; the contribution is primarily architectural integration rather than methodological innovation.
2. The author emphasized the multimodal, but only compared three multimodal datasets.
3. No justification for why δ_max = {0,1,...,5} works across datasets.
4. Many hyperparameters are introduced; how to keep the fairness of these hyperparameters across different datasets?

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

# Unified Generative Modeling for Multimodal Time Series Analysis

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 2, 6

## Abstract
Modeling multimodal time series has become an emerging research focus, aiming to incorporate auxiliary modalities, such as textual descriptions, into time series analysis. This integration enables a deeper understanding of temporal patterns by leveraging diverse sources of information. However, existing approaches often treat external modalities merely as supplementary domain features, neglecting the joint distribution between time series and auxiliary modalities. Moreover, most prior methods are tailored to specific tasks, limiting their generality and the effective utilization of multimodal data. In this paper, we propose GenTS, a unified generative model for multimodal time series analysis, integrating a variety of downstream tasks within a unified modeling framework. GenTS is trained to generate time series from textual descriptions and to forecast future values conditioned on historical multimodal data, simultaneously. This approach enables the model to capture the joint distribution between time series and external modalities, supporting a broad range of applications such as conditional generation, forecasting, and time series editing. Furthermore, by incorporating time series captioning as an integral component, GenTS has largely alleviated the common challenge of multimodal data scarcity. Extensive experiments on diverse real-world datasets demonstrate the effectiveness and generality of our approach across multiple tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Proposes GENTS, a unified generative framework for multimodal time series analysis, tackling data scarcity with domain-agnostic captioning and enabling multi-task (forecasting, generation, editing) via diffusion models.

### Strengths
- Introduces a flexible unified framework generalizing across multiple time series tasks (forecasting, generation, and editing).
- Domain-agnostic captioning effectively mitigates multimodal data scarcity.
- Comprehensive experiments across diverse datasets/tasks show strong performance.

### Weaknesses
- The ablation studies lack granularity regarding critical components. The paper needs detailed analysis of: (a) relative contributions of condition adapters versus self-attention mechanisms, (b) impact of different attribute categories in captioning, and (c) sensitivity to diffusion step embeddings.
- While Table 9 mentions inference times, the paper lacks analysis of training complexity, memory requirements, and scalability with sequence length.

### Questions
- How does GENTS scale computationally with increasing sequence length and variable dimensions? What is the theoretical complexity compared to task-specific baselines?
- What are the failure modes of the domain-agnostic captioning approach? Can it capture complex patterns like multi-scale anomalies or non-stationary transitions that lack clear statistical signatures?
- How does the unified training objective balance potentially conflicting optimization requirements between forecasting and generation tasks? The paper mentions λF:λG=2:1 works well, but provides no theoretical justification.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GENTS, a novel unified generative model specifically designed for multimodal time series analysis, which seeks to overcome the limitations of existing approaches that often only treat external modalities, such as textual descriptions, as supplementary features. The core contribution is a cohesive framework that explicitly models the joint distribution between the time series and auxiliary data to enhance generalization and performance across various prediction and editing tasks. The authors demonstrate that GENTS achieves significant performance improvements over strong foundation models and dedicated diffusion baselines on several distinct time series datasets.

### Strengths
1. The motivation behind this work is both novel and valuable, advocating for a shift from treating auxiliary modalities merely as supplementary features to explicitly modeling the joint distribution between the time series and the external data for richer analysis.

2. The proposed GENTS methodology appears sound and offers a reasonable, unified generative approach to incorporate diverse multimodal data, which is essential for advancing the generality of time series models.

3. The experimental evaluation is relatively comprehensive, showcasing the model's superior performance across multiple disparate time series tasks against several competitive and recently proposed foundation and diffusion baselines.

### Weaknesses
The primary confusion lies in the clarity of the experimental setup, which fundamentally impacts the assessment of novelty and fairness. Specifically, it remains ambiguous whether GENTS is implemented as a multi-task framework (trained and evaluated independently on each task's dataset) or if it functions as a single, large pre-trained foundation model evaluated via zero-shot transfer on downstream tasks. If this work falls into the former category (separate training per task), its technical novelty is somewhat limited; however, if the authors pursued the latter (unified pre-training followed by zero-shot evaluation), the comparison to task-specific baselines is likely unfair as the baselines were not optimized under equivalent pre-training data conditions.

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GenTS, a unified generative framework based on diffusion models for multimodal time series analysis. The core idea is to model time series data and textual descriptions together, with the ability to diverse tasks— forecasting, unconditional generation, and editing—within a single masked-conditional generation paradigm. 

Another contribution is a domain-agnostic time series captioning method that automatically generates textual descriptions from statistical and morphological attributes because of the lack of real data.

### Strengths
In general, this paper is comprehensive and try to bring traditional time series tasks to next level. Also, this paper did multiple experiments to prove the model's capabilities in forecasting, editing, and generation.

### Weaknesses
1.The paper should clarify what is unified modeling in the time series case. Because the general concept of unified modelling should be training with multiple tasks. But in this paper, although there are 3 tasks, the editing task is training-free. So in essense, this paper only targets in conditional generation and unconditional generation, which is very common in controllable generation, and ** should not be named unified modelling.**  Some real unified time series modelling papers include TimeDiT, UrbanDiT.

2. Unclear dataset description in editing task. This paper used real-world dataset with text to test editing tasks, but the experimental details are unclear, for example, for TimeMMD-Env, how to define the orignal time series?

3. Also, there should be some case study to visualize the capability of edting.

4. In the model architechture, the text is incorporated by adaLN, however, adaLN only suitable for global text instead of detailed text, for the tasks in this paper, cross-attention to incorporated the text should be more appropriate. You can see the GenTron and HunyuanDiT paper to know why cross-attention should be used. Also, cross-attention experiments should be added.

5. Why not use the flow-matching, because in generation area, flow matching has supassed diffusion.

### Questions
1. On the unified modeling: Can clarify the defination or change the name?

2. On the experimental setup for editing: Can you detail the precise process for creating the 'new textual guidance' used for editing the real-world datasets like TimeMMD-Env?

3. On the model architecture choice: What was the specific reasoning behind using an AdaLN mechanism for text conditioning instead of a cross-attention layer, which is commonly used for detailed textual control?

4. On the choice of generative backbone: Was there a specific reason for choosing a diffusion model as the backbone over more recent alternatives like Flow Matching?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GENTS, a diffusion-based framework that learns the joint distribution of time series and text, unifying forecasting, generation, and editing under a masked conditional generation paradigm.

### Strengths
Integrates forecasting, generation, and editing into a single framework.

Shows significant gains over TimeWeaver and TEdit on generation and editing; for forecasting across multiple datasets, performance is competitive with or better than strong baselines.

Clear and easy to follow.

### Weaknesses
Conceptually similar to conditional time-series generation (e.g., TimeWeaver) and text-guided editing (e.g., TEdit); those works already introduced J-FTSD/RaTS. The main novelty here is the unified packaging and the captioner.


Baselines include Chronos and Moirai, but the area is moving fast; consider adding newer models on GIFT-Eval, such as Chronos-2, Moirai-2, and TimesFM-2.5.


The caption pipeline uses tsfresh attributes templated into text; this ensures cross-domain applicability but limits expressiveness and interpretability.


GENTS is noticeably slower than non-diffusion baselines.


A uniform setup is used across datasets (700 epochs, bs=1024, lr=1e-3) with 3 seeds on an A40; fairness to all baselines is unclear.


In Table 2 (Weather), J-FTSD shows a large scale gap (GENTS = 1.320 vs. ~290 for two strong baselines); please further explain this anomalous advantage.


Beyond automatic metrics, consider adding a small human preference study (fidelity and instruction alignment) to support the conclusions.

### Questions
Please check weakness

### Soundness
2

### Presentation
3

### Contribution
2

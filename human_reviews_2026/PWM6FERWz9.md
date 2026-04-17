# GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series data

- Decision: Accept (Poster)
- Scores: 6, 8, 4, 4

## Abstract
Despite recent progress in time-series foundation models, challenges persist in improving representation learning and adapting to diverse downstream tasks. We introduce a General Time-series Model (GTM), which advances representation learning via a novel frequency-domain attention mechanism that captures time-granularity-aware features, an aspect underexplored in prior research. We further propose a novel pre-training strategy that unifies reconstruction and autoregressive objectives through a hybrid masking mechanism. Our pre-training strategy, combined with 2D positional encoding and span shuffling, enhances the robustness and generalization of representations. GTM is established as the first generative-task-agnostic model for time-series analysis, enabling seamless adaptation to various generative tasks without any task-specific modifications. Extensive experiments demonstrate that GTM consistently outperforms SOTA models on various generative tasks and achieves strong classification results with minimal adaptation. Furthermore, GTM exhibits clear scaling behavior, with accuracy improving as model size and pre-training data increase.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a time-series model that incorporates frequency-domain information in self-attention, a pre-training strategy that combines reconstruction and autoregressive objectives, and it shows that this model (after being trained on large dataset) it can be considered as a foundation model. Experimental results on a large number of time series datasets for different tasks shows that the model is competitive to existing ones, and in many cases leads to higher performance.

The main argument for the improved performance of the proposed model is the use of frequency domain features, something that the existing models are not exploiting. This argument has been used in many cases in the past, and it does not seem to be well-justified: no theoretical explanation is provided to prove the advantage of using a specific spate-time domain transformation (the Fourier transform) for internal neural network representations, while the experimental comparisons do not show a clear boost in performance that would be attributed to the adoption of a clearly better type of information.

Overall, the experimental analysis is comprehensive and it shows advances for the proposed model.

### Strengths
- The proposed model shows to achieve comparable or even better (on average) performance compared to existing models in a variety of tasks and datasets.
 - A comprehensive experimental evaluation is provided.

### Weaknesses
- No computational and memory cost analyses are provided for the proposed model.
 - No comparisons in terms of inference time are provided for the proposed model in relation to the competition.
 - Comparisons with recent related models, e.g., UniTS, Moirai, ST-LLM, are not provided.

### Questions
- What is the computational and memory costs of the proposed model?
 - How does the model compare with existing ones in inference speed?
 - Comparisons with more recent related models.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces the General Time-series Model , a novel foundation model based on the empirical finding that time-series data exhibits distinct frequency-domain distributions  at different temporal granularities . To exploit this, GTM incorporates a novel Fourier attention mechanism that learns to capture these granularity-aware representations. A key contribution is its unified pre-training strategy, which uses a hybrid masking approach to jointly optimize reconstruction and autoregressive objectives. This makes GTM the first generative-task-agnostic model, enabling it to handle diverse tasks like forecasting, imputation, and anomaly detection without any task-specific modifications. Extensive experiments confirm that GTM consistently outperforms or matches state-of-the-art models on these generative tasks, achieves strong results on classification, and demonstrates effective scaling with increased model size and data.

### Strengths
1. The paper is exceptionally well-motivated. It begins with a clear, empirical analysis using FFT and 2D KDE to demonstrate that the joint probability distributions of amplitude-frequency and phase-frequency vary significantly across different temporal granularities . This finding provides a strong, intuitive justification for the core architectural novelty of the proposed model.

2. The paper introduces a novel Fourier attention mechanism designed to explicitly capture the granularity-aware features identified in the initial analysis. Instead of just applying a standard FFT, the module learns to weigh different frequency-domain transformations based on the input's time granularity. This is a clever method to inject domain-specific knowledge about time-scale properties directly into the learning process.

3. A key contribution is the novel pre-training strategy that unifies reconstruction and autoregressive objectives. By adopting a GLM-style hybrid masking approach—which combines random span masking  with consecutive tail masking —the model is trained to handle a variety of generative tasks from the outset.

4. This unified pre-training strategy successfully establishes GTM as a generative-task-agnostic model. The authors demonstrate that the same pre-trained model can be applied to forecasting, imputation, and anomaly detection without any task-specific architectural modifications, which is a significant step toward a true "foundation model" for time series.

### Weaknesses
1. The hybrid masking strategy is a core contribution, but a critical detail is underspecified. The paper mentions applying a "controlled proportion of consecutive [MASK] tokens to at the tail". The value or control mechanism for this proportion, which presumably balances the reconstruction and autoregressive objectives, is not defined in the main paper or the appendix's implementation details .

2. The proposed Fourier attention module adds a significant number of operations within each decoder block: a standard temporal attention, an FFT, a granularity-based attention, five low-rank matrix multiplications, one full-matrix multiplication, and an inverse FFT . While the authors commendably provide an efficiency analysis in Table 24, which shows GTM is competitive, this added complexity is a non-trivial tradeoff for the performance gains.

### Questions
Listed in Weakness

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
3

### Summary
This work introduces GTM, a general model for generative time series learning. The authors claim that this model is one of the first to be task-agnostic for generative tasks, allowing the model to be applied in a variety of settings without task-specific modifications. The core contribution comes down to the featurization of the time series, where the authors combine many previous methods and use multiple decompositions of the time series within the model. There are also some interesting architectural choices that enable the model's performance. The model performs well on downstream benchmarks, with the authors undergoing significant downstream testing.

### Strengths
- The writing is clear and straightforward, laying out the reasoning for the model components and describing them in sufficient detail. The architecture seems to have a lot of backing from literature, and the authors build nicely on previous work.
- It is encouraging to see that the combination of lots of disparate components of time series analysis results in a performative model. I think this is an important contribution to demonstrate, and the authors motivate the use of these features very well. 
- The experiments in the paper are extensive and thorough, demonstrating the model’s applicability on a variety of downstream tasks. In addition, the appendix has a detailed number of experiments, which is helpful in observing model performance.

### Weaknesses
- One of the biggest weaknesses of the model is the resources required to run this architecture on practical time series data. A number of components of the model are very computationally expensive, including the full temporal attention and the internal FFT calls in the model. Some analysis of computational efficiency would be helpful.
- The performance, while better than baselines across most tasks, is only marginally better than other models. In the field of time series foundation models, where a plethora of models exist that all perform better or worse in a specific setting, this is not very impressive. 
- There are no error bars anywhere in the results, making it difficult to determine significance of the observed gains in performance.
- There are a number of suspicious results from the ablations, including that the pretraining is not much better than training on specific datasets, indicating that the “generality” of the model might not be optimal. In the case where the pretraining does not help, the authors would need to test against task-specific models as the regime becomes different. In addition, the boost in model performance is marginal in the scaling law study, raising questions as to the generality of the model.

### Questions
- Have the authors tested the model on long-range datasets? What is the maximum length of the datasets used in this paper?
- Are the results over baselines statistically significant?

### Soundness
2

### Presentation
4

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
The paper proposes **GTM (General Time-series Model)**, a generative-task-agnostic time-series foundation model that enhances representation learning by integrating temporal and frequency-domain features. It introduces a Fourier attention mechanism to capture time-granularity-aware patterns and a unified pre-training framework (hybrid masking, 2D positional encoding, span shuffling) that unifies reconstruction and autoregressive objectives. GTM seamlessly adapts to diverse generative tasks (forecasting, imputation, anomaly detection) without task-specific modifications, outperforms SOTA models across benchmarks, and exhibits clear scaling behavior with model size and pre-training data.

### Strengths
1. This paper addresses a critical gap in existing time series foundation models by introducing a Fourier attention mechanism that explicitly captures time-granularity-aware patterns from the frequency domain.
2. As far as I am concerned, GTM is the first TSFMs that supports seamless adaptation to all generative tasks (forecasting, imputation, anomaly detection) without task-specific modifications (e.g., tokenization adjustments, projection header changes). 
3. The paper conducts rigorous experiments across diverse benchmarks with fair and consistent enhancements, including 5 datasets for forecasting/imputation, 5 for anomaly detection, and 10 for classification.

### Weaknesses
1. The paper does not deeply explore the computational overhead of the Fourier attention mechanism—specifically, how FFT/iFFT operations (integrated into each decoder block) impact inference latency for real-time applications (e.g., streaming sensor monitoring). 
2. The model analysis of this paper is insufficient. For example, no trade-off analysis (e.g., simplifying frequency modules to reduce latency) is provided, which is critical for practical deployment.
3. The TSFM baselines are outdated. Please include more additional baselines such as TabPFN, Sundial, TimeMOE.

### Questions
1. Comparing to the models listed in the baseline, how much more/less computational resource does GTM consume? Can GTM still remain high performance if the fourier transform calculation time is limited?
2. In the Fourier attention mechanism, different frequency patterns are taken into attention calculation by forming a vectorized feature. Does this strategy outperform mixing (by MLPs for instance) these frequency patterns into a representation embedding?

### Soundness
3

### Presentation
3

### Contribution
2

# Continuous Evolution Pool: Taming Recurring Concept Drift in Online Time Series Forecasting

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
Recurring concept drift, a type of concept drift in which previously observed data patterns reappear after some time, is one of the most prevalent types of concept drift in time series. As time progresses, concept drift occurs, and previously encountered concepts are forgotten, thereby leading to a decline in the accuracy of online predictions. Existing solutions mainly employ parameter updating techniques to delay forgetting; however, this may result in the loss of some previously learned knowledge while neglecting the exploration of knowledge retention mechanisms. To retain all knowledge and fully utilize it when the concepts recur, we propose the **C**ontinuous **E**volution **P**ool (CEP), a pooling mechanism that stores different instances of forecasters for different concepts. Our method first selects the forecaster nearest to the test sample and then learns the features from its neighboring samples—a process we refer to as **retrieval**. If there are insufficient neighboring samples, it indicates that a new concept has emerged, and a new model will evolve from the current nearest sample to the pool to **store** the knowledge of the concept. Simultaneously, the **elimination** mechanism will enable outdated knowledge to be cleared to ensure the prediction effect of the forecasters. Experiments on real-world datasets demonstrate that by retaining distinct conceptual knowledge, CEP significantly enhances online forecasting accuracy, reducing the error by over 20%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The aim of the paper is to address the possible time evolution (i.e., concept drift) in time series forecasting. This is a crucial issue since (fairly) all the time series are directly or indirectly time-dependent. In more detail, the paper introduces a framework, called Continuous Evolution Pool (CEP), for the time series forecasting under recurrent concept drift. The framework introduces a feature vector called gene (being the representative of the time series) and mechanisms to add, reactive and remove concept drift. When a new concept arrives, it is inserted in the pool of concepts and it reactivated when the same gene is identified, so as to reduce the catastrophic forgetting effects.

### Strengths
- The paper addresses the important challenge of recurring concept drift in non-stationary time series
- The framework introduces mechanisms for addition, deletion, activation of recurrent

### Weaknesses
- Limited feature representation: The gene vector relies only on mean and variance, which may be insufficient for distinguishing complex temporal concepts. What about considering moments or frequency-domain statistics to provide richer representations?

- Choice of distance metric: The model compares genes via Euclidean distance. However, for time series similarity, techniques such as DTW (Dynamic Time Warping) or Mahalanobis distance might better capture temporal and statistical dependencies. The authors should discuss this limitation.

- What about the splitting threshold? I guess it is a crucial parameter to set

- What exactly is nwait? Is it a time count or number of missed updates? Its calibration seems critical to stability.

- The value of L is fixed across all datasets. Is this empirically optimal or arbitrary? It might influence both performance and drift detection frequency.

- What about considering not just threshold for detecting changes? I would suggest to consider change detection mechanisms such as CUSUM, etc..

- What if the detection of a new concept is missed? This is a False Negative detection that could severely induce issues in the characterization of the error. I guess thresholds are not robust enough for detection. 

- I would have expected to see zero-shot learning foundation models for time series forecasting (e.g., Chronos, Tirex) in the experimental section.

### Questions
See previous section

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the challenging problem of recurring concept drift in online time series forecasting. The authors identify a key limitation of existing methods—catastrophic forgetting of past concepts due to continuous parameter updates—and propose a novel framework called the Continuous Evolution Pool (CEP).

The core idea is to maintain a dynamic pool of forecasters, each specialized for a distinct data concept. CEP operates through three main mechanisms:

Retrieval: For a new input instance $\mathbf{x}_t$, its "gene" $\mathbf{g}_x = (\mu(\mathbf{x}_t), \sigma(\mathbf{x}_t))$ is computed, and the closest forecaster in the gene space is selected for prediction.

Evolution: If the instance's gene deviates significantly from the nearest forecaster's gene (based on a statistically motivated threshold, e.g., $|\mathbf{g}{x,\mu} - \mathbf{g}{N,\mu}| > \tau_{\mu} \cdot \mathbf{g}_{N,\sigma}$), a new forecaster is "evolved" by splitting from the nearest one, effectively capturing a new or recurring concept.

Elimination: Inactive forecasters are pruned to manage pool size and remove outdated knowledge.

The authors demonstrate through extensive experiments that CEP significantly boosts the performance of various base forecasters (e.g., TCN, PatchTST) and substantially outperforms strong online learning baselines, especially under delayed feedback scenarios.

### Strengths
**Originality:** The formulation of the "Continuous Evolution Pool" is a fresh and compelling idea. The gene-based, proactive concept identification is a significant shift from reactive, error-based adaptation strategies.

**Quality & Significance:** The empirical validation is thorough and convincing. The method's ability to significantly improve performance across diverse model architectures and under challenging delay scenarios is a major strength. 

**Clarity:** The core algorithm is presented clearly, and the extensive appendix provides valuable theoretical justifications (regret analysis, statistical grounding) and practical guides.

**Practicality:** The method is model-agnostic and comes with a practical hyperparameter tuning guide, enhancing its potential for real-world adoption.

### Weaknesses
**Theoretical vs. Empirical Grounding:** While the appendix provides a regret analysis, it remains somewhat high-level. A more formal bound on the identification regret, perhaps under specific assumptions about the separation between concepts in the gene space, would strengthen the theoretical analysis.

**Limitations of the Gene Representation:** The gene, while effective, is a relatively simple statistical summary. The paper acknowledges its limitation on the Traffic dataset (with frequent, low-magnitude fluctuations) but could discuss this more deeply. Are there types of concept drift (e.g., changes in auto-correlation structure, frequency content) that this gene would be inherently blind to? A brief discussion or experiment hinting at this would be valuable.

**Multivariate extension:** The method is evaluated only on univariate settings. Many real-world time series are multivariate, and concept drift may affect channels asynchronously. How CEP scales to this setting is unclear.

**Experiment Analysis:** The paper could provide more insight into why certain strong baselines like FSNet and OneNet fail so dramatically in the delayed setting (Table 2). A brief qualitative analysis or hypothesis in the main text would add depth beyond just reporting the performance gap.

### Questions
1. The elimination criterion (Eq. 12) uses a fixed ratio . Could this accidentally remove a forecaster for a rare-but-important concept (e.g., annual seasonality in a short evaluation window)? Have you considered a more sophisticated retention policy (e.g., based on historical recurrence intervals)?

2. You justify the gene $\mathbf{g}_x = (\mu, \sigma)$ well from a statistical moment perspective. Did you experiment with incorporating other simple, efficient-to-compute features into the gene (e.g., a crude measure of trend or a single autocorrelation coefficient)? If so, what were the results? If not, could such an extension be a straightforward way to handle a wider class of concept drifts?

3. The performance on the Traffic dataset suggests a limitation with your fixed $\tau_{\mu}=3$ threshold for datasets with subtle, non-recurring drifts. Your practical guide suggests tuning $\tau_{\mu}$ lower for such cases. Did you run experiments on Traffic with a lower $\tau_{\mu}$, and if so, did performance recover? This would be a powerful point to demonstrate the method's adaptability.

4. While you state the inference time is identical to the base forecaster, the evolution, retrieval, and elimination operations do introduce overhead. Could you provide quantitative data (e.g., average percentage increase in total runtime compared to a naive model) to give readers a concrete sense of the practical computational cost?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
The paper proposes an innovative framework named the "Continuous Evolution Pool" (CEP) to address recurring concept drift in online time series forecasting. At its core, the framework introduces "genes" to proactively identify data concepts and dynamically maintains a specialized pool of forecasters. The strengths of the work lie in its clear structure, an innovative proactive drift detection paradigm, and comprehensive experimental validation across multiple datasets and foundational models; rich visualizations and ablation studies also provide strong support for its conclusions. However, defining a "gene" solely by its mean and standard deviation may fail to capture more complex data distributions, while the hard assignment strategy for forecasters may perform poorly in transitional concept regions. The fixed elimination period for forecasters could also struggle to mitigate "catastrophic forgetting" for concepts with long recurrence periods. Furthermore, the method is primarily designed for periodic tasks, limiting its effectiveness in non-recurring drift scenarios, and the theoretical derivation on the effectiveness of Euclidean distance lacks direct experimental verification. In summary, while the paper is well-structured and the method is novel, the robustness and theoretical validation of its core mechanisms require further investigation and experimental support.

### Strengths
1.The paper has a clear structure and logical flow. It first identifies the challenge of recurring concept drift and systematically analyzes the limitations of existing methods. Subsequently, the paper proposes the core solution, the CEP framework, and progressively elaborates on its key mechanisms.

2.The method is highly innovative. Instead of relying on the traditional passive approach of adapting to drift based on prediction error, the paper innovatively proposes a proactive detection mechanism based on "genes".

3. The paper not only details the algorithmic aspects of CEP but also provides an in-depth discussion of its key advantages for practical deployment, such as avoiding "catastrophic forgetting" by retaining specialized forecasters and managing computational resources through its elimination mechanism.

4.The experimental design is thorough and provides comprehensive validation. The paper not only tests its method on multiple real-world datasets with diverse characteristics but also applies it as a universal framework to several mainstream forecasting models, proving its general applicability and effectiveness.

5.The rich visualization analysis and detailed ablation studies clearly reveal the internal working mechanisms of CEP and the necessity of its components, thereby strengthening the persuasiveness of the conclusions.

### Weaknesses
1.The paper defines a "gene" as the mean and standard deviation of a data window. Although this method is simple and computationally efficient, it may not be sufficient to capture the distribution within the window, making it unable to detect all types of concept drift.

2.When a data window is in a transitional stage between two concepts, it may be difficult to select an appropriate forecaster for prediction.

3.The proposed method performs exceptionally well on data with recurring patterns, but its efficiency on non-recurring tasks is limited.

4.The paper derives that if the concepts are normally distributed, maximizing the log-likelihood is approximately equivalent to minimizing the Euclidean distance. However, this derivation is not supported by experimental evidence, thus the effectiveness of using Euclidean distance is not verified.

5.The paper removes forecasters that have been idle for a certain period. However, this fixed elimination period means that it cannot mitigate catastrophic forgetting for concepts whose recurrence period exceeds the set threshold.

### Questions
1. The paper defines a "gene" using the mean and standard deviation of a data window for proactive concept identification. While computationally efficient, this might not capture complex temporal patterns or shapes. The lightweight alternative or additional features (e.g., trend, periodicity, or simple shape descriptors) could be explored in future work to enhance the robustness.

2. The paper employs a hard assignment strategy, matching a data window to a single forecaster. A concern is that this might perform poorly during transitional phases between concepts. Please explain it in detail.

3. The mechanism for removing idle forecasters uses a fixed period, which may forget concepts with long recurrence intervals. Have you considered implementing a more adaptive elimination strategy to maintain the performance?

4. The method is explicitly designed for recurring concept drift, which may inherently limit its efficiency on tasks with non-recurring patterns. It is needed to consider other types of concept drift.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper deals with concept drift problem in time series forecasting. It proposes a pooling mechanism to associate each forecaster to a concept. This mechanism includes creation of new forecaster for new concept, elimination of forecasters corresponding to outdated concepts. The proposed mechanism can be associated with any forecaster models. Experimental results suggest that it allows to improve the forecasting  performance in the presence of concept drifts.

### Strengths
The proposed mechanism is relatively simple and is general which allows it to be easily coupled with any forecaster, 

The experimentation on forecasting performance improvement is extensive both in terms of the forecasting models compared and the dataset used.

### Weaknesses
The simplicity of the concepts managed by the proposed mechanism is also a major limitation of this work, In fact, a concept in this work correspond basically to the means and standard deviation of the time series in a look-back window. Concepts with very different patterns may have the same statistical properties of means and standard deviation.

Anohter limitation is that the proposed mechanism is for single time series. Multiple time series exhibitete even more complex pattens for concepts.

Some key parameters such as the splliting parameter $tau$ is difficult to fix. There is a lack of discussion about how to chose the values for these parameters.

Tables and figures are not always well explaned. For instance, it it not clear what the percentage means in Table 2. The three settings in Figure 2 are not well explained neither. The quality of writing of this paper is between fair and poor.

### Questions
1- It is not clear why "recurring" concept drift is any of particularity as compared to normal concept drift. It is just some concepts that appear and reappear. I don't see any special traitment in the proposed mechanism that deals with "recurring".

2- I don't understand why there is a warm-up stage. Yes, I understand that you start with one forecaster and there is a process tu tune it to fit to some initial samples. This is not fundamentally different from creation of any new forecaster in the "Online stage".

### Soundness
2

### Presentation
2

### Contribution
2

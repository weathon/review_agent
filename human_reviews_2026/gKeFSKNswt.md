# PLanTS: Periodicity-aware Latent-state Representation Learning for Multivariate Time Series

- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Multivariate time series (MTS) data are ubiquitous in domains such as healthcare, climate science, and industrial monitoring, but their high dimensionality, scarce labels, and non-stationary nature pose significant challenges for conventional machine learning methods. While recent self-supervised learning (SSL) approaches mitigate label scarcity by data augmentations or time point-based contrastive strategy, they overlook the intrinsic periodic structure of MTS and fail to capture the dynamic evolution of latent states. We propose PLanTS, a periodicity-aware self-supervised learning framework that explicitly models irregular latent states and their transitions. We first designed a periodicity-aware multi-granularity patching mechanism and a generalized contrastive loss to preserve both instance-level and state-level similarities across multiple temporal resolutions. To further capture temporal dynamics, we design a next-transition prediction pretext task that encourages representations to encode predictive information about future state evolution.  We evaluate PLanTS across a wide range of downstream tasks—including classification, forecasting, trajectory tracking, and anomaly detection. PLanTS consistently improves the representation quality over existing SSL methods and demonstrates superior computational efficiency compared to baseline methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PLanTS, a periodicity-aware self-supervised learning framework for multivariate time series. It models latent states and their transitions using a periodicity-aware multi-granularity patching mechanism and a contrastive loss combining instance and state-level samples. Additionally, it includes a next-transition prediction task to capture temporal dynamics.

### Strengths
The next transition task is novel in time series. The paper is also easy to read and follow. Authors also performed several experiments including several tasks, such as classification and forecasting.

### Weaknesses
The idea of multi-granularity patching is not new, and using FFT to extract dominant periodicities for patching has already been explored, for example in TimesNet. My main concern is the next-transition prediction task.
Since the PTB-XL dataset consists of 10-second, 12-lead ECG samples, it is unclear how the model can meaningfully learn state transitions within such short segments. This approach seems more suitable for continuous recordings with well-defined transitions. Also, most of the datasets are collected in a controlled environments where the next state transition is already set. How does the model can deal with that situation?

### Questions
1) How does the proposed multi-granularity patching differ from prior FFT-based methods like TimesNet?

2) With only 10-second PTB-XL samples, what does “next-state transition” mean, and how can the model learn it without continuous data?

3) Has the next-transition task been tested on datasets with longer recordings to verify its validity?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focus on the self-supervised learning for multivariate time series modeling. A model named PLanTS is proposed, which is a periodicity-aware self-supervised learning framework that explicitly models irregular latent states and their transitions. Key techniques include multi-granularity patching and contrastive. Experiments are carried out based on classification, forecasting, trajectory tracking, and anomaly detection tasks.

### Strengths
1. SSL for MTS modeling is a critical problem

2. Source codes are provided to ensure reproducibility.

### Weaknesses
1. More discussions on the motivation are needed. Why do we need periodicity-aware and multi-granularity self-supervised learning for MTS modeling? Under what challenges and problems?

2. Does multi-granularity patching relate to multi-scale modeling? There are many related works that need discussion, e.g., Timemixer.

3. Experiments need improvement; more baseline methods could be included, e.g., methods that are based on masked-reconstruction (SimMTM), and TimeSiam.

4. To evaluate the time series forecasting performance. More datasets except for ETTs are needed.

### Questions
Please refer to the weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors propose a periodicity-aware, multi-granularity self-supervised learning framework for non-stationary multivariate time series representation learning based on contrastive learning. This framework can be applied to downstream classification, forecasting, trajectory tracking, and anomaly detection tasks.

### Strengths
1. SSL for MTS modeling is a critical task.

2. Source codes are provided.

3. The paper is well-structured and easy-to-follow

### Weaknesses
1. Some critical related works are missing. For example, multi-scale modeling methods for time-series like Timemixer++. Also, there are numerous time series foundation models that require pre-training based on large-scale datasets, which can also be regarded as representation learning for MTS.

2. It's unclear why multi-scale modeling can address nonstationary challenges in MTS.


3. More large-scale forecasting benchmarks could be included, e.g., weather and traffic datasets like PEMS.

### Questions
1. Since the proposed method adopts multi-scale modeling techniques. Can itbe  applied to multi-rate time series modeling? More discussions are needed.

### Soundness
2

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
5

### Summary
This paper proposed a representation learning methods for multi-variate time series data. The method first uses Fourier transformation to find frequency components with top-k amplitudes to select window sizes to perform multi-scale time series patching. Then the representations for the similar subsequence patched among patches in one times series of different start time stamp or across different time series with the same time stamp are aligned according to “Maximum Cross Correlation” scores using InfoNCE loss. The two parts of loss are used to optimize two different encoders and the representations learned is concatenated into a whole.

### Strengths
1. This paper proposes simple methods that is easy to follow.
2. The multiple types of downstream tasks enhance validity of proposed method.
3. The proposed method shows superiority over the selected baselines.

### Weaknesses
1. The methods are introduced in a straight way, which lack of salient analysis, insights ,and takeaways. Such as insights about the use of “Maximum Cross Correlation”, or how to feed the multi-scale patches into the encoders.
2. The baselines are pretty old, new methods should be presented, such as CSL[1].
3. Figure 2 introduces pretext tasks for TNC and TS2Vec. However the purpose for demonstrating them, e.g., for contrastive usage, is not introduced explicitly enough.
4. Taking an average over performances on 30 UEA dataset is not a reasonable operation, because there is no statistical meaning.
[1] Liang, Z., Zhang, J., Liang, C., Wang, H., Liang, Z., & Pan, L. (2023). A Shapelet-based Framework for Unsupervised Multivariate Time Series Representation Learning. Proc. VLDB Endow., 17, 386-399.

### Questions
As the weaknesses shows.

### Soundness
2

### Presentation
2

### Contribution
2

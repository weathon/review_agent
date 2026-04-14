# Learn hybrid prototypes for multivariate time series anomaly detection

- Decision: Accept (Poster)
- Scores: 6, 5, 5, 6, 6

## Abstract
In multivariate time series anomaly detection (MTSAD), reconstruction-based models reconstruct testing series with learned knowledge of only normal series and identify anomalies with higher reconstruction errors. In practice, over-generalization often occurs with unexpectedly well reconstruction of anomalies. Although memory banks are employed by reconstruction-based models to fight against over-generalization, these models are only efficient to detect point anomalies since they learn normal prototypes from time points, leaving interval anomalies and periodical anomalies to be discovered. To settle this problem, this paper propose a hybrid prototypes learning model for MTSAD based on reconstruction, named as H-PAD. First, normal prototypes are learned from different sizes of the patches for time series to discover interval anomalies. These prototypes in different sizes are integrated together to reconstruct query series so that any anomalies would be smoothed off and high reconstruction errors are produced. Furthermore, period prototypes are learned to discover periodical anomalies. One period prototype is memorized for one variable of the query series. Finally, extensive experiments on five benchmark datasets show the effectiveness of H-PAD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposed a reconstruction-based model called H-PAD for multivariate time series anomaly detection to address the issue of over-generalization.

### Strengths
1. Clear motivation
2. Well structured

### Weaknesses
1. The font size of the figures is too small.
2. There is a lack of related work, such as "Joint Selective State Space Model and Detrending for Robust Time Series Anomaly Detection".
3. The principle of the proposed method is not clear enough. For example, please explain how the proposed method benefits from mapping the original features to a higher dimensional space (D > C). If C is already very large, will the proposed method still be effective?
4. As shown in Table 1, the performance gain of the proposed method is marginal. Please test it on more datasets say 3 more datasets.
5. Where is your code? the reproducibility is an issue.
6. How did you set the parameters of your proposed method and all compared baselines?
7. How should we choose the parameters of your proposed method?
8. The cases in Figure 5 are overly simple/easy which cannot reflect the advantage of the proposed method.


**I am willing to increase my scores if these issues are well addressed.**

### Questions
Please see above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a method to address the issue of overfitting to anomalies in existing time series anomaly detection algorithms. The approach involves learning different patches and periodic prototypes, and detecting anomalies through reconstruction. Experiments demonstrate that the proposed method outperforms existing algorithms.

### Strengths
Comprehensive experiments were conducted to validate the proposed method.

### Weaknesses
1. The paper provides the formulas for the algorithm but lacks an explanation of the rationale and thought process behind their design. This omission may hinder readers' understanding of why the proposed method is effective.
2. The paper primarily claims that current algorithms suffer from overfitting to anomalies. However, subsequent sections on method design do not explain how the proposed method addresses this issue.
3. The text in Figures 1, 3, and 4 is too small.

### Questions
1. The paper claims that the proposed method can learn contextual information, and occasional point anomalies cannot utilize this context, thus avoiding overfitting. However, in reality, current time series analysis algorithms can also leverage contextual information. Could the paper provide a clearer explanation of why using multiple patches can mitigate the issue of overfitting to anomalies?
2. In the contributions section, what does “reconstruct abnormal series to be normal ones” mean?
3. Decomposing data using FFT and analyzing time series data from both the time and frequency domains is a common approach in many methods. What are the advantages of the proposed method compared to these existing techniques?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The main contribution of this paper lies in proposing a multiscale time series anomaly detection method H-PAD that combines local and periodic information. By designing local and periodic prototypes, introducing sparsity and periodic constraints, and integrating anomaly scoring mechanisms that consider both reconstruction errors and feature space deviations, the method effectively enhances the accuracy and robustness of anomaly detection.

### Strengths
1. The paper introduces a framework, H-PAD, for multivariate time series anomaly detection by combining patch-based and period-based prototypes to capture both local and global patterns. Combining local and periodic prototypes offers rich contextual information for anomaly detection.

2. The methodology uses both time-domain and frequency-domain features to enhance detection accuracy. The dual-prototype mechanism, along with tailored anomaly scoring, demonstrates a robust approach to avoiding over-generalization.

3. The reconstruction approach allows the model to effectively replicate normal patterns, aiding in more accurate anomaly identification.

### Weaknesses
1. The simple weighted average fusion of local and periodic reconstruction results may lead to information loss or conflict, lacking flexibility. 
2. The lack of detailed explanation regarding the implementation mechanism and role of the sparsity constraint may affect understanding and application effectiveness.
3.	The lack of explanation regarding the basis for weight parameters in the loss function may lead to unstable model performance across different tasks.
4.	The setup of the experimental section is not sufficient. Some parameter sensitivity experiments could be conducted to make the theoretical part of the article more convincing.
5.	While the paper is mostly clear, certain aspects, such as some definitions and mechanisms, could benefit from additional clarification to improve replicability and reader comprehension.

### Questions
1.	In the INTRODUCTION section of the article, line 65 contains a typographical error: "this paper proposes an MSTAD..." should be "MTSAD."
2.	In line 81, the description of Contribution 2, "but also can reconstruct abnormal series to be normal ones," is not accurately described. Providing a more detailed explanation might be better.
3.	In line 180, it should specify that "z1=1” corresponds to the original sequence X. Adding this detail would be more informative. Additionally, it would be helpful to clarify how the encoder part works—whether it directly uses the encoder block from the Transformer. Providing a more specific structural introduction would improve clarity.
4.	In line 201, the introduction of the update gate is abrupt, and its function is unclear. Additionally, the introduction of the linear transformation matrices U_z and W_p is not well defined—what is their relationship to the context? It would be helpful to explain why linear transformations are applied to b and q.
5.	In section 3.3, line 276, the calculation of the reconstructed sequence involves directly averaging the temporal and frequency domain reconstruction information. Since the sources and characteristics of these two types of reconstruction information are different, is this setting reasonable? It would be advisable to provide some explanation.
6.	In line 296, are alpha_1, alpha_2, and alpha_3 manually adjusted hyperparameters or dynamically learnable parameters using an adaptive method? If they are manually adjusted, how can their approximate ranges be determined? It would be helpful to provide some analyses regarding the parameter settings.

### Soundness
2

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
This manuscript proposes a hybrid prototypes learning model, H-PAD, which addresses the problem that existing models can only detect point anomalies. Specifically, normal prototypes are learned from different sizes of patches for time series to discover short-term anomalies. These prototypes in different sizes are integrated together to reconstruct query series so that any anomalies would be smoothed off and high reconstruction errors are produced.  Furthermore, period prototypes are learned to discover periodical anomalies. One period prototype is memorized for one variable of query series.

### Strengths
1. The paper is clearly organized.
2. The authors propose H-PAD for multivariate timing anomaly detection, which addresses the problem that existing models can only detect point anomalies.

### Weaknesses
1. It is recommended that the authors optimize Fig. 1 to better describe the motivation for this paper.
2. Since anomaly detection is inherently class unbalanced, it is recommended that the authors add AUC to Table 1 to fully analyze the effectiveness of the model.
3. In experiments, whether or not these datasets chosen by the authors contain types of anomalies other than point anomalies seems to be important for the performance of the model. If we only look at Fig. 5, it seems that they are all point anomalies. It is recommended that the authors further add more types of anomalies to the visualization analysis to demonstrate the benefits of H-PAD.

### Questions
See Weaknesses please.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes H-PAD, a method to learn hybrid prototypes for multivariate time series anomaly detection. Hybrid prototypes contain both local and global information to help discover both shot-term (point) and long-term (period) anomalies. The authors evaluate their proposed method against various baseline models on 5 datasets and perform ablation studies to understand the importance of each component in the model architecture.

[Update] Adjusted original score after reviewing the authors' rebuttal and revised manuscript

### Strengths
1. The authors are familiar with the current literature on time series anomaly detection and evaluate their proposed method against SOTA baselines. 
2. Useful ablation studies are performed to understand the importance of each component (patch vs. period prototypes) in the model architecture.
3. The model architecture design using query-based reconstruction (in both temporal and frequency domains) is motivated and explained with clear technical details.

### Weaknesses
1. The writing quality should be improved for better clarity. The authors use several non-standard terms such as “different local sizes” which should be corrected. 
2. The authors should provide additional implementation details on data processing and model training to help other researchers reproduce and extend their results. For example, how are the patch sizes {z1, z2, …, zm} and k (as in top-k amplitudes of FFT) selected?
3. The authors should discuss the limitations of their work and outline the directions for future research.
4. There are numerous typos and grammatical errors that need to be proofread and corrected. For example, “reference phase” should be “inference phase” (page 1) and “learn and memory prototypes” should be “learn memory prototypes” (page 2).
5. The authors should provide rigorous mathematical definitions of affiliation precision/recall and RAP/RAR since they may not be familiar to most readers. The authors should also clearly explain why these metrics are used instead of the ordinary precision/recall/AUC.
6. It’d be helpful to have more detailed review of the mechanism of memory networks and memory prototypes (either in Related Work or in Supplementary Materials) since these concepts may not be very familiar to most readers.  
7. In addition to real-world datasets, it’d be ideal to evaluate the model on simulated time series data to verify that the patch and period prototypes indeed capture multi-scale and multi-period information and effectively detect the corresponding anomalies.

### Questions
1. How are the time series data preprocessed? What are the sizes of the datasets? Did the authors apply any filters or normalization to the datasets prior to training the model?
2. How are different types of anomalies (point vs. period) defined in these datasets? 
3. What are the computing resources used to train the model? How is the model training efficiency?
4. What are the raw precision, recall and AUC metrics of anomaly detection?
5. How does model performance change with the dimensionality of the time series?
6. What does it mean that “Generally speaking, the series of scale z1 is actually the original series X.”? Does this mean z1 is always set to 1? If so, the authors should clearly state this to avoid confusion. 
7. Why is Figure 5 (c) an example of period anomaly instead of point anomaly? It seems that the period is the same but the amplitude is anomalous. 
8. What distance metric is used to calculate affiliation precision/recall?

### Soundness
3

### Presentation
2

### Contribution
2

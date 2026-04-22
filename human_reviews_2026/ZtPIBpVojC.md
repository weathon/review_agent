# Low Rank Transformer for Multivariate Time Series Anomaly Detection and Localization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Multivariate time series (MTS) anomaly diagnosis, which encompasses both anomaly detection and localization, is critical for the safety and reliability of complex, large-scale real-world systems. The vast majority of existing anomaly diagnosis methods offer limited theoretical insights, especially for anomaly localization, which is a vital but largely unexplored area. The aim of this contribution is to study the learning process of a Transformer when applied to MTS by revealing connections to statistical time series methods. Based on these theoretical insights, we propose the Attention Low-Rank Transformer (ALoRa-T) model, which applies low-rank regularization to self-attention, and we introduce the Attention Low-Rank score, effectively capturing the temporal characteristics of anomalies. Finally, to enable anomaly localization, we propose the ALoRa-Loc method, a novel approach that associates anomalies to specific variables by quantifying interrelationships among time series. Extensive experiments and real data analysis show that the proposed methodology significantly outperforms state-of-the-art methods in both detection and localization tasks. Code is available at: https://github.com/CharisShimillas/ALoRa.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper tackles multivariate time series anomaly diagnosis, covering both detection and localization. It analyzes the learning behavior of Transformers from a theoretical perspective and connects it to classical statistical time-series analysis. Based on these insights, the authors propose the Attention Low-Rank Transformer (ALoRa-T) with low-rank regularization to better capture temporal anomaly patterns, and introduce ALoRa-Loc for variable-level anomaly localization. Experiments on real and synthetic datasets show that the proposed approach outperforms existing methods in both detection and localization tasks.

### Strengths
1. The paper offers valuable theoretical insights by linking the Transformer’s self-attention mechanism to established statistical time-series principles, providing a more interpretable foundation for deep anomaly detection models.
2. Unlike many prior works focusing only on detection, the introduction of ALoRa-Loc enables variable-level anomaly attribution, advancing the underexplored area of multivariate anomaly localization.

### Weaknesses
1. The distinction between “time series” and “variable” is not consistently maintained throughout the paper. Since each variable corresponds to a univariate time series, the terminology should be clarified to avoid conceptual confusion.
2. The paper states that each kernel learns representations from only two time series, but the motivation for selecting exactly two is not explained. 
3. The metrics used to evaluate anomaly localization ability — Hit Rate, Normalized Discounted Cumulative Gain (NDCG), and Interpretation Score — are not well-suited for this task. Hit Rate and NDCG are designed for ranking or recommendation settings, while Interpretation Score lacks a clear definition in the context of anomaly localization.
4. The paper does not compare with established approaches for anomaly localization or root cause identification, such as “Root Cause Analysis of Anomalies in Multivariate Time Series through Granger Causal Discovery.”
5. In Table 2, several numeric values use commas instead of decimal points.

### Questions
Please see the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a transformer-based framework for time-series anomaly detection that leverages attention rank analysis to interpret and localize anomalies. The key idea is that the rank of self-attention matrices increases when anomalies occur, providing a new signal for both detection and localization.

### Strengths
1. The idea of detecting anomalies by analyzing the transformer’s learning behavior is original and insightful. It opens a new direction for understanding model-internal representations in time-series anomaly detection.

2. The focus on anomaly localization is meaningful and practically valuable.

### Weaknesses
1. The paper uses Spearman correlation to estimate dependencies among sequence pairs but does not justify why this choice is preferred over Pearson correlation or Cosine Similarity. Furthermore, the paper states that only the top-K correlated pairs are retained, yet the criterion for determining K is not specified or experimentally analyzed.

2. The central claim that “the rank of SA-matrices increases in the presence of anomalies” is only supported by empirical observation on a few datasets. The paper does not provide a theoretical explanation or evidence that this phenomenon holds consistently across diverse anomaly types and domains.

3. The definitions of variables are inconsistent—sometimes the input sequence is denoted as x, other times as y, making the mathematical expressions difficult to follow.

4. The inference process depends critically on the threshold h_2. Although the paper mentions that Appendix A describes its selection, the appendix does not include such details yet.

5. Localization evaluation requires ground-truth information about the precise anomalous series. However, the datasets used in the experiments typically provide only record-level anomaly labels (anomalous or normal per timestamp) without explicit localization annotations. Could the authors clarify how the localization ground truth is obtained?

6. The main text experiments are overly concise and lack detailed analysis. Although the appendix includes an ablation study, it only evaluates the embedding module. A more critical ablation, particularly on the ALoRa loss function, is missing and should be included to support the claimed effectiveness of the proposed loss.

7. The paper does not provide the source code, and the methodological descriptions are not detailed enough to reproduce the reported results reliably.

### Questions
See the weaknesses section

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
The paper proposes ALoRa, a Transformer-based framework for multivariate time-series (MTS) anomaly detection and localization grounded in a theoretical analysis of Transformer encoders on MTS. The authors show that the encoder’s latent representations can be expressed as linear combinations of Space-Time Autoregressive (STAR) processes, which motivates (i) ALoRa-T—a Transformer with low-rank regularization on self-attention—and (ii) a detection score that counts significant singular values of the final attention matrix. They further derive contribution weights from inputs → latent → outputs to trace anomaly propagation and attribute anomalies to variables (ALoRa-Loc).

### Strengths
(1) The paper provides a coherent spectral perspective on attention that is simple to compute conceptually and ties to an interpretable diagnostic. 

(2) The authors diagnose that point-adjustment inflates results—sometimes making them indistinguishable from random scoring—and therefore pivot to range-aware/affiliation-based metrics, improving evaluation validity.

(3) The localization section explicitly models propagation via contribution weights (E, C), which is more principled than per-dimension reconstruction heuristics. 

(4) The training objective is compact and implementable; the regularizer integrates cleanly with standard reconstruction losses.

### Weaknesses
(1) While the detection pipeline uses two thresholds ($h_1, h_2$), Appendix A provides data-driven approach of choosing threshold $h_1$, but this is still a per-dataset manual step, introducing hyperparameter sensitivity. Also, neither ablation on $h_2$ selection nor heuristics on choosing it was provided.

(2) The paper’s central intuition—\textit{anomalous windows yield higher attention rank}—is supported empirically (plots/observations) but lacks a formal guarantee. No theoretical background specifies conditions under which anomalies must raise rank (or non-anomalies must not).

(3) ALoRa-Loc traces propagated influence, but ranking metrics like HR/NDCG/IPS do not distinguish origin variable from downstream affected variables; without per-segment confusion analyses, it’s unclear whether the method finds causes or merely effects.

### Questions
(1) How sensitive is the final detection F1-score (which relies on the combined $AS(x_t)$) to this choice? For instance, what is the performance impact if $h_1$ is set 10x larger or 10x smaller than the value chosen via the eigenvalue distribution analysis?

(2) Please provide explanation on how $h_2$ value was selected and why.

(3) Do different anomaly types (point vs collective vs contextual) induce distinct singular-value patterns? Any class-wise analysis of detection latencies?

(4) For segments where the anomaly propagates widely, how often does top-k ALoRa-Loc identify the true origin vs “most affected” variables? Could you report per-segment confusion analyses?

(5) Some important ablations are missing: rank-only score vs error-only vs multiplicative combo; head-wise vs averaged penalty; all-pair vs top-K embeddings; FFN on/off at matched params. Could you please provide ablations on these?

* A minor typo in line 228; "throught" $\rightarrow$ "thought"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper advances multivariate time series (MTS) anomaly detection in three distinct aspects.  First, it provides theoretical insights into how the Transformer encoder represents and learns from the MTS data, revealing how its representations relate to classical time series. For instance, the authors equate the embedding process to Vector Moving Average (VMA) filtering, and the self-attention mechanism to the Space-Time Autoregressive (STAR) structure. Second, the authors propose Attention Low-Rank Transformer (ALoRa-T), which consists of the LightMTS-Embed module and Attention Low-Rank (ALoRa) layers, and a decoder. Lastly, given this new architecture, the authors propose a novel detection score and localization method: ALoRa-T score and ALoRa-Loc method.

### Strengths
- The authors theoretically relate the Transformer architecture back to the techniques from classical time series modeling. Based on this insight, they propose technically sound and well-motivated modifications to the Transformer architecture, further specializing it for the task of MTS anomaly detection.

- The authors propose novel detection and localization frameworks that are more reliable than previously used metrics.

- Together, the proposed method and detection/localization methods successfully outperform other baselines. The experimental results are quite comprehensive, and the authors have included code and sufficient experimental details to reproduce the results.

### Weaknesses
- According to Table 1, it appears that AloRa-Det is more effective on some datasets (ex) HAI or SMD) than other (SwAT, MSL). What causes such a discrepancy in the results? Is ALoRa-Det more effective at detecting certain anomaly types than others?

- The majority of the baselines are drawn from Transformer-backed anomaly detection methods (for a good reason). Yet, it would be helpful to add some baselines from other families of MTS anomaly detection methods, such as reconstruction or contrastive learning-based methods.

- Do the authors expect their method to stay functional in application scenarios where anomalies and distributional shifts (concept drifts) appear mixed together? If so, how could ALoRa-T be extended or modified to such cases?

- Although the authors present ablation studies in Section D, I believe a more thorough ablative study that investigates the effectiveness of each proposed technical component separately to assess its contribution is necessary.

- Just a minor comment on paper formatting: I understand that the authors have chosen to move many of the experimental results due to the page constraint, but I personally think key results and analyses should still remain as a part of the main manuscript. I suggest that the authors truncate some of the materials in the introduction/related works to make room for the results section.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
3

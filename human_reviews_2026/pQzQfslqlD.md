# Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Although contrastive and other representation-learning methods have long been explored in vision and NLP, their adoption in modern time series forecasters remains limited. We believe they hold strong promise for this domain. To unlock this potential, we explicitly align past and future representations, thereby bridging the distributional gap between input histories and future targets. To this end, we introduce TimaAlign, a lightweight, plug-and-play framework that establishes a new representation paradigm, distinct from contrastive learning, by aligning auxiliary features via a simple reconstruction task and feeding them back into any base forecaster. Extensive experiments across eight benchmarks verify its superior performance. Further studies indicate that the gains arise primarily from correcting frequency mismatches between historical inputs and future outputs. Additionally, we provide two theoretical justifications for how reconstruction improves forecasting generalization and how alignment increases the mutual information between learned representations and predicted targets. Code is in available at https://github.com/TROUBADOUR000/TimeAlign.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes TimeAlign, a lightweight framework that enhances time series forecasting by introducing a reconstruction branch and aligning its representations with those of the prediction branch at multiple layers. Theoretically, the authors justify that (1) the reconstruction branch improves generalization, and (2) the alignment loss strengthens the mutual information between historical and future representations. Extensive experiments on eight benchmark datasets and multiple backbones demonstrate consistent performance gains.

### Strengths
1. The motivation is clear and intuitive. The paper explicitly identifies three limitations of current forecasting paradigms and addresses them directly through reconstruction and alignment. 

2. The method requires only a lightweight reconstruction branch during training, and introduces no additional parameters at inference.

3. The paper presents two theoretical analyses: reconstruction-guided generalization based on the assumption of a linear model, and an approach based on mutual information for alignment, which aligns well with empirical observations.

4. The effectiveness of TimeAlign has been validated on eight widely used benchmarks.

### Weaknesses
1. It seems important to stop the gradients from the alignment loss to the reconstruction branch, but the justification for this is insufficient, as it lacks a comparison with symmetric gradient flow.

2. The overall loss function combines three terms, but the paper does not analyze sensitivity to the hyperparameter margins $\delta_{loc}, \delta_{glo}$.

3. It is unclear whether all the baselines were retuned under identical conditions with regard to the look-back window and learning rate. As TimeAlign is a plug-in, consistent hyperparameter settings across backbones are required for a fair comparison.

### Questions
1. The authors conducted plugin experiments on iTransformer and DLinear. So, which types of backbones (e.g., Transformers, MLPs, CNNs, etc.) benefit the most from being combined with TimeAlign?

2. What is the computational and memory scaling of the global alignment loss as sequence length and channel dimension grow?

3. In the NIPS 2024 workshop[1], some researchers pointed out that current methods sometimes use the "drop-last" trick [2] to improve performance. Therefore, It is recommended that you clarify whether the "drop - last" operation was used in your paper in the implementation details section of your paper for transparency.

[1] Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation

[2] TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents TimeAlign, a framework for time series forecasting that bridges the gap between past and future data. It uses a dual-branch architecture, with a Predict Branch for forecasting and a Reconstruct Branch for target reconstruction. TimeAlign employs distribution-aware alignment, combining local and global objectives to preserve fine-grained details and enhance forecasting accuracy. Experiments show it outperforms existing methods, with theoretical analysis highlighting its ability to improve generalization and increase mutual information between learned representations and future targets.

### Strengths
1. The paper is easy to follow, with clear and thorough writing. The motivation highlights three key limitations in current time series forecasting, which are well-justified.

2. The experiments are solid. The main experiments demonstrate that TimeAlign outperforms other SOTA methods, and the ablation studies validate the effectiveness of its components.

3. I appreciate the theoretical analysis provided in Section 4, Appendix A, and Appendix B, which enhances the theoretical contribution of the paper and explains the underlying reasons for the method's effectiveness.

4. TimeAlign can also serve as a plug-and-play module, making it easy to integrate with other mainstream methods.

### Weaknesses
1. Equation 8 in Line 292 is slightly inconsistent with Figure 3. One version includes both $\lambda_1$ and $\lambda_2$, while the other has only $\lambda$. It seems the version you're using is the one with a single $\lambda$, so the formula in the figure should be updated. Also, Figure 6 analyzes the impact of different $\lambda$ on performance. If following the formula in the figure with both $\lambda_1$ and $\lambda_2$, what would be their effect? I would appreciate a more detailed analysis of the loss function's behavior.

2. In the Weather/Solar plug-in experiments, author attributes performance differences to outliers and zero-padding issues. However, more concrete quantitative evidence is needed to support this claim.

3. The knee-point detection used to separate high/low frequencies (Figure 2) may be sensitive to noise and window selection, which could affect the reliability of the high-frequency energy/similarity analysis.

4. I find Figure 4 (right) quite interesting, but in Lines 453-457, it seems you are merely describing an experimental result. What is the underlying reason for iTransformer+Align converging faster than iTransformer? Also, do other architectures, such as CNN-based or Linear-based models, show similar behavior?

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a representation learning framework for time series data based on reconstruction objectives.

### Strengths
1. The empirical results are generally strong; the reported evaluation metrics outperform or are on par with existing baselines, although some results appear potentially inconsistent and merit further verification.

### Weaknesses
Major Weaknesses

1. **Lack of rationale for reconstruction:**
The motivation for using reconstruction as the core learning principle is not clearly articulated. The paper claims that reconstruction aligns representations with the target distribution because it “recovers inputs from themselves,” but this statement simply restates the definition of reconstruction. It does not provide a conceptual or theoretical justification for why reconstruction should yield better representations. Similarly, the claim that reconstruction emphasizes high-frequency details is asserted but never explained or supported by analysis.

2. **Unclear methodological details:**
Several methodological choices are insufficiently justified or analyzed.
- Why are the margin terms ($\delta_{loc}$, $\delta_{glo}$) necessary?
- What is the rationale for using the GELU activation function specifically?
- What motivates the introduction of a weight-based dynamic loss?

3. **Abnormal assumptions in Eq. 9 and Appendix B:**
The theoretical analysis assumes that patterns are drawn from a Gaussian distribution and that outputs follow a linear transformation only at the first time step, with Gaussian noise thereafter. These assumptions are highly unrealistic for time series data. Consequently, the claim that an optimal estimator exists in a linear form is not meaningful in practice. The proof and theorem therefore have limited - if any - practical relevance.

4. **Weak justification of claims in Section 4.2:**
The discussion on mutual information maximization (MIM) lacks substance. Prior works such as MINE (Belghazi et al., ICML 2018) and other MIM-based methods have already shown that deep models implicitly maximize mutual information through their layers. The paper does not provide convincing evidence that reconstruction is strictly better than other MIM approaches.

5. **Inconsistent ablation results:**
In the ablation study, the variant using only the “Predict Branch” (i.e., without reconstruction) outperforms most state-of-the-art baselines. This raises concerns about the fairness of the experimental setup and the actual contribution of the proposed reconstruction mechanism. Further clarification or additional controlled experiments are needed.

Minor Weaknesses

6. The Introduction lacks clear structure. The first and third observations are redundant and should be merged or rephrased.

7. The figures are visually cluttered; they attempt to convey too much information at once, which makes them hard to interpret.

### Questions
I wrote these questions based on the priority.

1. Provide a clear and theoretically grounded explanation of why reconstruction is beneficial for time series forecasting. (Related to W1)

2. Re-examine the performance of the Predict Branch in the ablation studies and explain why it outperforms many baselines. Ensure that the experimental setup and baseline implementations are described in sufficient detail to guarantee fair comparison. (Related to W5)

3. Offer a solid rationale for the Gaussian and linearity assumptions introduced in Eq. 9 and Appendix B, and discuss their practical relevance to real-world time series data. (Related to W3)

4. Clearly articulate how and why reconstruction provides explicit advantages within the Mutual Information Maximization (MIM) framework, compared to other existing MIM-based approaches. (Related to W4)

5. Provide reasoning or empirical evidence supporting key design decisions - such as the use of margin terms, GELU activation, and dynamic loss weighting. (Related to W2)

### Soundness
2

### Presentation
2

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
This paper proposes TimeAlign, a lightweight and plug-and-play framework for time-series forecasting that aims to mitigate the distributional bias between historical inputs (Past) and future targets (Future). The authors observe that existing models tend to over-rely on low-frequency periodic patterns, lose high-frequency details, and suffer from representational misalignment between past and future features. To address this, TimeAlign introduces an auxiliary Reconstruction Branch and employs both global and local distribution-aware alignment mechanisms to constrain the main Prediction Branch, enabling representations that are simultaneously distribution-aware and detail-preserving. The method is theoretically analyzed through mutual information and empirically improves forecasting performance across multiple benchmarks and backbones.

### Strengths
S1: The paper explicitly attributes forecasting errors in time series forecasting models to the distributional misalignment between past and future representations, and quantifies this mismatch through cosine-similarity analysis and visualization.

S2: The method is simple and modular. A reconstruction branch could be added to any backbone model, and the prediction branch is constrained through dual alignment. Both alignment formulations, relaxation terms, and the adaptive weighting factors (α/β) are well defined, and the overall objective adds only one alignment loss term to the original training loss, making the module fully plug-and-play.

S3: The paper provides formal analysis showing that reconstruction-based alignment can tighten generalization bounds and implicitly enhance the mutual information between model representations and targets. This gives the method a stronger theoretical foundation than most empirical time series forecasting approaches.

S4: The proposed module is tested on several representative time series forecasting architectures (e.g., iTransformer, DLinear) and achieves consistent improvements across benchmarks. It also demonstrates faster convergence during training, with negligible additional overhead.

### Weaknesses
W1. Although the paper motivates TimeAlign through the problem of frequency smoothing, the proposed method does not explicitly model spectral information. Its improvement on high-frequency dynamics is achieved only implicitly via reconstruction and alignment. The frequency analysis is limited to the high-frequency energy ratio and spectrogram visualization, which measure the amount of high-frequency content but not its correctness or temporal alignment with the ground truth. Without band-wise or phase-level validation, it remains unclear whether the model truly learns meaningful high-frequency dynamics or merely amplifies spectral noise.

W2: The reconstruction branch provides strong supervision from the future target distribution during training, but is discarded entirely at inference. This raises a potential train-test inconsistency: predictive representations are optimized under guidance unavailable during deployment. The paper does not evaluate whether removing the reconstruction branch alters representation stability or performance, leaving uncertainty about the robustness of TimeAlign under distribution shift or domain generalization.

W3: The alignment mechanism may implicitly push the prediction branch to form a collapsed representation resembling a static “future prior.” Since the reconstruction branch (which observes future data) is only available during training, the learned “future-like” latent space might represent a statistical prior distilled from past futures rather than the true evolving future distribution. This is not necessarily harmful, but it remains unclear whether such a prior actually helps mitigate distribution mismatch at inference, especially when future dynamics deviate from the training regime.

### Questions
1. Could the authors provide more evidence that the model learns meaningful high-frequency dynamics rather than merely amplifying spectral noise? For instance, have the authors considered phase-level or band-wise validation (e.g., coherence with ground-truth signals) to confirm that the reconstructed high-frequency components are temporally aligned?

2. How does this prior behave under distribution shifts or evolving dynamics where future statistics differ from training? It would be informative to test this explicitly. For example, simulate mismatched future distributions or evaluate on real-world datasets with non-stationary temporal patterns to verify whether the learned alignment remains adaptive.

### Soundness
4

### Presentation
2

### Contribution
4

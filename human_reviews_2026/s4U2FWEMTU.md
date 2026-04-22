# Online time series prediction using feature adjustment

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Time series forecasting is of significant importance across various domains. However, it faces significant challenges due to distribution shift. This issue becomes particularly pronounced in online deployment scenarios where data arrives sequentially, requiring models to adapt continually to evolving patterns. Current time series online learning methods focus on two main aspects: selecting suitable parameters to update (e.g., final layer weights or adapter modules) and devising suitable update strategies (e.g., using recent batches, replay buffers, or averaged gradients). We challenge the conventional parameter selection approach, proposing that distribution shifts stem from changes in underlying latent factors influencing the data. Consequently, updating the feature representations of these latent factors may be more effective. To address the critical problem of delayed feedback in multi-step forecasting (where true values arrive much later than predictions), we introduce ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space). ADAPT-Z utilizes an adapter module that leverages current feature representations combined with historical gradient information to enable robust parameter updates despite the delay. Extensive experiments demonstrate that our method consistently outperforms standard base models without adaptation and surpasses state-of-the-art online learning approaches across multiple datasets.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles the challenge of distribution shift in online time series forecasting. It introduces a paradigm that deviates from conventional methods of updating model parameters. The authors posit that distribution shifts are fundamentally caused by changes in underlying latent factors, and therefore, it is more effective to directly adjust the feature representations produced by a model's encoder.
To this end, they propose ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space), a method that uses a lightweight adapter module to predict a corrective "delta" for the feature vector.

### Strengths
The paper is built upon a clear and intuitive premise. And the proposed method is tested on multiple public datasets using different base forecasting models, and its performance is compared against several baseline approaches. The authors also include supplementary analyses, such as ablation studies and a discussion of hyperparameters, which offer some insight into the behavior of the proposed method.

### Weaknesses
1. **"Historical Gradient"**: The method relies on a "historical gradient" computed from a batch of data ending at timestep t-k to inform the feature correction at the current timestep t. While this is a pragmatic approach to handling delayed feedback, this term is not a true gradient for the current step but rather a delayed and averaged approximation. The paper could provide a clearer discussion on the implications of this temporal mismatch, especially in scenarios with abrupt, non-gradual shifts where this historical information might be stale or misleading.

2. **Insufficient Engagement with Test-Time Adaptation (TTA) Literature:** The task of adapting a pre-trained model to a stream of incoming data under distribution shift is the central problem in the Test-Time Adaptation (TTA) field. The paper does not adequately position its work within this highly relevant body of literature in the main text. A more thorough comparison and discussion of the similarities and differences with TTA methods would provide better context for the paper's contributions.

3. **Practical Concerns Regarding Computational Efficiency:** The efficiency analysis in the appendix reveals that ADAPT-Z, while often more memory-efficient than full-model fine-tuning, is slower in terms of runtime than most baseline methods. For real-world online applications where prediction latency is a critical constraint, this added computational cost could be a significant drawback. This trade-off between accuracy and speed deserves a more prominent discussion in the main paper.

4. **Potential Ambiguity in Dataset Split Justification:** The authors justify their use of a 60/10/30 train/val/test split by arguing it is more "realistic" than the 25/5/70 splits used in some prior work. While their reasoning is plausible, this choice results in a significantly shorter online deployment period for evaluation. This could make the adaptation challenge less severe compared to enduring a distribution shift over 70% of the data, thereby complicating direct comparisons with results from papers that used the longer test split.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces ADAPT-Z (Automatic Delta Adjustment via Persistent Tracking in Z-space), a novel online time-series forecasting framework that adapts to distribution shifts by updating feature representations instead of model parameters. It employs a lightweight adapter network that fuses current latent features and historical gradients to generate correction terms ($\delta_t$) in the feature space, effectively mitigating delayed feedback in multi-step forecasting. Extensive experiments on 13 datasets and three base models (iTransformer, SOFTS, TimesNet) demonstrate consistent gains over state-of-the-art online learning methods such as DSOF, SOLID, Proceed, and ADCSD. The paper highlights that feature-space adaptation offers a more stable and interpretable alternative to conventional parameter updates under non-stationary environments.

### Strengths
The paper offers an original perspective by focusing on feature-space adaptation rather than parameter updates, addressing distribution shifts through latent factor correction. It provides strong technical quality, supported by extensive experiments across 13 datasets and several architectures with consistent improvements. The methodology is clearly presented, including detailed pseudo-code, ablation studies, and reproducibility information. Its significance lies in showing that updating latent feature representations is a stable and general strategy for online time-series forecasting under non-stationary conditions.

### Weaknesses
While the experiments are extensive and demonstrate consistent gains, they lack statistical robustness, as no variance or confidence intervals are reported across different random seeds, making it difficult to assess the reliability of the improvements. The evaluation could be broadened to include a wider range of model architectures, such as MLP-based and linear forecasting models, which are common baselines in the time-series literature. The computational efficiency of the approach is also unclear, particularly the memory and runtime overhead introduced by storing and updating historical gradients, which may limit scalability in long-horizon or high-frequency settings. Furthermore, assessing performance under stronger distribution shifts or longer forecasting horizons would provide stronger evidence of generalizability and robustness across real-world deployment scenarios.

### Questions
1. Could the authors provide statistical analysis (e.g., variance or confidence intervals across multiple random seeds) to assess the consistency and significance of the reported improvements? P.S.: This can be for a small scenario just to show that we can improve with the proposed method. I am not asking for anything new or extensive here. It is just to make the paper stronger.

2. How well does ADAPT-Z generalize across different backbone families, such as linear or MLP-based forecasting models, which are widely used in the literature?

3. What is the computational and memory overhead associated with maintaining and updating historical gradients during online adaptation, and how does this scale with longer horizons or larger datasets?

4. Have the authors evaluated the method under stronger or abrupt distribution shifts, such as sudden regime changes, to test its robustness in more realistic online environments?

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
4

### Summary
This paper proposes ADAPT-Z for online time series forecasting tasks to address distribution shift issues, particularly the critical problem of delayed feedback in multi-step forecasting. The paper argues that distribution shifts stem from changes in the underlying factors influencing the data, and ADAPT-Z introduces an adapter module that leverages current feature combined with historical gradient information to achieve high-quality parameter updates even in the presence of delays.

### Strengths
1. ADAPT-Z focuses on features to address shift, dynamically estimating bias using features combined with historical gradient information, while also exhibiting strong compatibility and can be used alongside normalization methods.
2. The experiments effectively corroborate most of the claims made in the paper.
3. The experiments are comprehensive, having investigated most aspects of the research that warrant further exploration.
4. The workflow of this paper is very clear.

### Weaknesses
1. The paper identifies concept drift as a key limitation of existing research schemes, but the method does not explain how ADAPT-Z addresses it, making it unclear why concept drift is mentioned.
2. For the main results, the metric used for evaluation is MSE. Could the performance of various methods also be examined using MAE?
3. In section 4.3.3, the paper explores the performance impact of extracting feature from different regions of iTransformer. Could this exploration be extended to feature locations in other models? For instance, the models used in this paper, such as SOFTS and TimesNet. Based on the best-performing feature locations across several models, could any commonalities be identified?
4. In section 4.3.6, the paper states: "our shorter prediction horizons (1-48 steps) experience more volatile distribution shifts. This instability makes it challenging for normalization method to consistently align input-output distributions across diverse time segments." Could the stable improvement of combining ADAPT-Z with normalization methods be explored under longer prediction horizons? Simultaneously, could ADAPT-Z's leading performance in forecasting tasks be examined under these longer prediction horizons as well?
5. The idea is quite simple. Can you clarify the novelty lies in the proposed method?

### Questions
As in Weaknesses

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
3

### Summary
This paper proposes a novel online adaptation method with the integration of current features and historical gradients for online time series forecasting, which can address delayed feedback and distribution shift in online multi-step forecasting. Experiments demonstrate the effectiveness of ADAPT-Z.

### Strengths
1. Integrating the current contextual features and historical gradient information to resolve the challenge of delayed feedback and ensure update stability is interesting.
2. The experiments are presented in considerable detail.

### Weaknesses
1. The novelty of ADAPT-Z seems limited as its design combines adapters and historical gradients, which are extensively studied in Parameter-Efficient Fine-Tuning (PEFT) and continual learning, respectively. The contribution is more of an engineering integration rather than a novel contribution. The authors should further emphasize their main contribution.
2. In Table 2, the reported performance for baselines FSNet and OneNet is highly questionable. Their MSEs are orders of magnitude worse than other methods and drastically inconsistent with their original papers. Such a stark discrepancy suggests potential issues with their implementation or the experimental setup, making a fair comparison questionable. The authors should clarify this significant discrepancy.
3. In Section 4.3.3, the authors claim that the performance of ADAPT-Z is correlated to the choice of feature layer and the optimal layers varying across datasets. However, the paper lacks a deeper analysis of the correlation between dataset characteristics and the best feature layer, leaving this critical design choice largely unexplained.
4. More recent prediction models should be compared to further validate the effectiveness of ADAPT-Z, e.g., TimeFilter [1] and TimeKAN [2].

[1] Hu Y, Zhang G, Liu P, et al. TimeFilter: Patch-specific spatial-temporal graph filtration for time series forecasting. ICLR 2025.

[2] Huang S, Zhao Z, Li C, et al. TimeKAN: KAN-based frequency decomposition learning architecture for long-term time series forecasting. ICLR 2025.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

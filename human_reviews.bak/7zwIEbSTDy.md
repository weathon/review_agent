# PPT: Patch Order Do Matters In Time Series Pretext Task

- Decision: Accept (Poster)
- Scores: 3, 8, 6, 6

## Abstract
Recently, patch-based models have been widely discussed in time series analysis. However, existing pretext tasks for patch-based learning, such as masking, may not capture essential time and channel-wise patch interdependencies in time series data, presumed to result in subpar model performance. In this work, we introduce *Patch order-aware Pretext Task (PPT)*, a new self-supervised patch order learning pretext task for time series classification. PPT exploits the intrinsic sequential order information among patches across time and channel dimensions of time series data, where model training is aided by channel-wise patch permutations. The permutation disrupts patch order consistency across time and channel dimensions with controlled intensity to provide supervisory signals for learning time series order characteristics. To this end, we propose two patch order-aware learning methods: patch order consistency learning, which quantifies patch order correctness, and contrastive learning, which distinguishes weakly permuted patch sequences from strongly permuted ones. With patch order learning, we observe enhanced model performance, e.g., improving up to 7% accuracy for the supervised cardiogram task and outperforming mask-based learning by 5% in the self-supervised human activity recognition task. We also propose ACF-CoS, an evaluation metric that measures the *importance of orderness* for time series datasets, which enables pre-examination of the efficacy of PPT in model training.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a self supervised patch order learning pretext tasks for time series classification. Experiments are shown to quantify the benefits of path order learning. An evaluation metric to measure importance of orderness is also proposed.

### Strengths
The paper offers a perspective on ordering of patches in self supervised TS learning, and presents experiments on some datasets to show benefits and drawbacks of the approach.

### Weaknesses
1. Lack of SoTA baselines, compare against and cite:

- S. Gao et al, UNITS: A Unified Multi-Task Time Series Model, https://arxiv.org/abs/2403.00131, NeurIPS 2024
- Yong Liu et al, itransformer: Inverted transformers are effective for time series forecasting, ICLR 2024
- Shizhan Liu et al. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting, ICLR 2021
- Haixu Wu et al. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting, NeurIPS 2021

In particular, UniTS has been shown to outperform PatchTST and other baselines in many setups.

2. The authors do not consider tasks beyond classification. I'm not convinced that the method is truly better than others when considering the representations generated, for example no comparisons with SoTA is shown on anomaly detection or clustering tasks. See for example, TF-C Appendix G, which is a baseline method authors compare against.

- Zhang et al, Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency, NeurIPS 2022

3. Limited evaluations. Experiments are all one-to-one dataset. The authors do not consider one-to-many, or other settings to test whether the representations are generalizable to cross-dataset settings etc. 

4. The gains are not convincing since many other baselines outperform the proposed method in several metrics.The improvement over GPT4TS is marginal.

5. It's not clear whether this method even works for diverse dataset pretraining settings as in UniTS (https://arxiv.org/abs/2403.00131). Since this method is mainly for pretext tasks, this is an important point to raise given we're in the era of foundation models.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a new pretext task designed for patch-based time series models, focusing on the order of time series patches. The Patch order-aware Pretext Task (PPT) is aimed at improving time series classification performance by considering the sequential and channel-wise order of data patches. This task leverages controlled patch permutations to produce supervisory signals for training models to learn inherent order dependencies. PPT incorporates both patch order consistency learning and contrastive learning to handle weakly and strongly permuted patch sequences, achieving better model performance in various time series tasks.

### Strengths
- Novelty in time series pretext task: PPT introduces a novel pretext task that leverages patch order consistency learning and contrastive learning to improve time series classification performance. This is a significant contribution to the field of time series representation learning.
- Flexibility and adaptability: By introducing both consistency and contrastive learning, PPT can be incorporated into various task settings, either self-supervised or supervised, making it a versatile method for time series classification.
- New metric for evaluating the importance of orderness: The paper also introduces ACF-CoS, which measures the importance of orderness in time series data, providing a new metric for evaluating the effectiveness of PPT and potentially other time series models.
- Strong empirical results: The paper demonstrates the effectiveness of PPT in improving time series classification performance across various datasets, showing significant improvements over existing methods.
- Clear and well-structured presentation: The paper is well-written and structured, especially the explanation of the PPT task and its illustration with examples. The clarity of the presentation enhances the understanding of the proposed method.

### Weaknesses
- Unclear details on consistency loss: The paper do not provide more detail into how the context vectors are generated other than stating that they are generated by an auto-regressive model. More details into the architecture and training of this model would be helpful for better understanding the consistency loss.
- Lack of discussion on hyperparameter $\lambda$: The paper does not provide a detailed study or discussion on the hyperparameters $\lambda$, which are to be "learnable parameters". In the paper they cited that these hyperparameters are calculated from the prediction of a separate model, but more details on how this is done would be beneficial. It would also be helpful to provide ablation studies on the effect of such dynamic $\lambda$ on the performance of the model compared to fixed $\lambda$.

### Questions
1. Could you provide more details on how the context vectors are generated by the auto-regressive model for the consistency loss? How is the architecture of this model designed, and how is it trained?
2. Could you provide more details on how the hyperparameters $\lambda$ are adjusted during training? What are the final results of the learned $\lambda$ values at the end of training, and how do they affect the performance of the model?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a novel Patch Order-aware Pretext Task (PPT) for time series classification, focusing on the importance of patch order in time series data. Traditional patch-based models often overlook the critical order dependencies in time and channel dimensions, which can lead to suboptimal performance. PPT addresses this gap by generating supervisory signals through controlled channel-wise patch permutations, allowing models to learn time series' intrinsic sequential characteristics. The main contributions of the paper are as follows: (1) Patch Order-aware Pretext Task (PPT): This is the first pretext task specifically designed to enhance order-awareness in patch-based time series models. PPT leverages controlled channel-wise permutations to provide structural supervisory signals in both time and channel dimensions. (2) Two Learning Methods: Patch Order Consistency Learning evaluates the correctness of patch order by distinguishing between original and strongly permuted sequences; Contrastive Learning differentiates between weakly and strongly permuted sequences, helping the model capture finer order distinctions. (3) ACF-COS Metric: The paper proposes ACF-COS (Autocorrelation Function with Cosine Similarity), a metric to quantify the "orderness" in time series data, which serves as a pre-evaluation tool to determine if PPT would be beneficial for a particular dataset. (4) Performance Gains: PPT demonstrates significant performance improvements on various time series tasks, especially in scenarios where order-awareness is crucial, such as ECG and human activity recognition (HAR) tasks, with performance boosts of up to 7% and 5% respectively. By integrating order-awareness through PPT, the approach enhances the ability of time series models to utilize structural dependencies across patches, setting a new direction for patch-based time series analysis.

### Strengths
**Introduction of the Patch Order-aware Pretext Task (PPT)**: This paper presents the first order-aware pretext task specifically designed for patch-based time series models. Unlike traditional tasks such as masking and reconstruction, PPT captures essential order dependencies across time and channel dimensions, a distinctive feature of time series data. By leveraging this natural inductive bias of sequence ordering, PPT provides effective self-supervision signals that help the model better understand structural characteristics in time series classification tasks.

**Implementation of Order Consistency and Contrastive Learning for Pretraining**: To achieve order-aware learning within PPT, the authors designed two learning strategies: order consistency learning and contrastive learning. Order consistency learning distinguishes between the original sequence and strongly perturbed sequences, validating the model's understanding of patch order correctness. Contrastive learning differentiates between weakly and strongly perturbed sequences, allowing the model to capture finer distinctions in order dependencies. Together, these learning approaches enhance the model's sensitivity to sequence order, helping it capture the critical ordering information embedded within time series data and improving performance on order-dependent tasks.

**Significant Performance Improvement**: PPT demonstrates notable improvements across multiple time series tasks, particularly in scenarios where order information is critical. Experimental results show that PPT enhances accuracy by up to 7% in ECG classification and by 5% in human activity recognition (HAR), underscoring its effectiveness. By integrating PPT in both self-supervised and supervised settings, models are able to better leverage the structural dependencies in time series data, achieving superior classification performance. This advancement not only showcases PPT's value in practical applications but also highlights the potential of order-aware tasks in time series analysis.

### Weaknesses
**Limited Applicability to Non-Order-Dependent Tasks**: PPT is specifically designed to exploit order dependencies in time series data, which limits its applicability to tasks where sequence order is critical. For datasets or tasks that do not rely heavily on the sequential order of data (e.g., tasks with significant noise or random sequences), PPT may add unnecessary complexity without meaningful performance gains. The paper itself notes that PPT does not perform well in such scenarios, and a robust evaluation metric is required to assess its suitability beforehand.

**Dependence on Hyperparameter Tuning for ACF-COS and Permutation Strength**: The effectiveness of PPT depends on the selection of appropriate hyperparameters, especially for the ACF-COS metric and permutation intensity (weak vs. strong permutations). However, the paper provides limited guidance on systematically tuning these parameters for different datasets, which could hinder reproducibility and practical implementation. Users may need to perform extensive trial-and-error testing to optimize PPT for specific datasets, which is resource-intensive and may not always yield consistent results.

**Limited Success in Time Series Forecasting**: Although PPT shows promise in classification tasks, it is less effective for time series forecasting applications, where trend continuity is crucial. The patch permutation strategy disrupts these trends, making it challenging for models to capture long-term dependencies needed for accurate forecasting. The paper does not provide a modified approach for forecasting tasks, which restricts PPT’s utility and impact in a broader range of time series applications.

### Questions
1. **Is there a method that can be adapted for both classification and forecasting tasks in time series?** Specifically, is it possible to design a pretext task that effectively utilizes order information for classification while preserving trend continuity to support forecasting tasks? If so, can this approach balance prediction accuracy with the advantages of sequence dependency?

2. **Does the ACF-COS metric demonstrate robust and broad applicability in measuring the orderness of time series data?** Specifically, can ACF-COS accurately differentiate between various types of time series tasks, such as high-noise sequences, non-order-dependent tasks, and strongly order-dependent tasks? Additionally, how can we ensure the stability of ACF-COS across different patch sizes, sampling frequencies, or data distributions to reliably evaluate the suitability of PPT for diverse time series datasets?

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
4

### Summary
The authors believe that patch order awareness is very important in time series modeling and  introduce PPT (Patch order-aware Pretext Task) for time series classification. The authors also develop ACF-CoS metric for quantifying the importance of patch orderness and interpreting applicable to PPT scenarios. Experimental results demonstrate that PPT shows advantages in self-supervised, fully-supervised, and semi-supervised learning with sparse labels.

### Strengths
- Work is novel and motivation regarding patch order is reasonable for time series analysis.
- Well written, clear structure, easy to understand
- Experiments are sufficient, and the analysis based on ACF-CoS is interesting

### Weaknesses
Rocket[1] and MiniRocket[2] are important baselines for time series classification tasks. I think considering them can make the experiment more comprehensive (including classification performance and algorithm efficiency). 

[1] Angus Dempster, et al. ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. arxiv preprint arxiv: 1910:13051 

[2] Angus Dempster, et al. MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification. arxiv preprint arxiv: 2012:08791

### Questions
- Time dimension patch order is intuitive, but channel dimension order is not easy to understand and it needs further discussion.
- Although the author provides the consumed time of permuting a batch of instances, considering the need to store additional augmented samples and the need for additional calculation, it would be better if the author can provide the additional speed and memory consumption required by PPT.
- Confusing content: In Tab. 4 caption: “a decrease in ACF-CoS leads to an improved likelihood of PPT being beneficial”. Should “a decrease” be modified to “an increase”? In my understanding, there is a positive correlation between the ACF-COS score and PPT's performance.

### Soundness
3

### Presentation
4

### Contribution
3

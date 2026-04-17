# Stochastic Order Learning: An Approach to Rank Estimation Using Noisy Data

- Decision: Reject
- Scores: 6, 4, 8, 6, 2

## Abstract
A novel algorithm, called stochastic order learning (SOL), for reliable rank estimation in the presence of label noise is proposed in this paper. For noise-robust rank estimation, we first represent label errors as random variables. We then formulate a desideratum that encourages reducing the dissimilarity of an instance from its stochastically related centroids. Based on this desideratum, we develop two loss functions: discriminative loss and stochastic order loss. Employing these two losses, we train a network to construct an embedding space in which instances are arranged according to their ranks. Also, after teaching the network, we identify outliers likely to have extreme label errors and relabel them for data refinement. Extensive experiments on various datasets show that the proposed SOL algorithm yields decent rank estimation results even when labels are corrupted by noise.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Stochastic Order Learning (SOL), a novel algorithm for robust rank estimation when data labels are corrupted by noise. The key contributions include modeling label errors as random variables, formulating a desideratum based on minimizing stochastic dissimilarity from centroids, and introducing two new loss functions: discriminative loss and stochastic order loss. The method also incorporates an outlier detection and relabeling scheme to refine noisy training data. Extensive experiments on various datasets (facial age estimation, aesthetic score regression, medical assessment, and textual regression) demonstrate that SOL achieves state-of-the-art performance and exhibits strong robustness to label noise.

### Strengths
1. The core idea of treating label errors as random variables and formulating the objective based on minimizing stochastic dissimilarity is a theoretically sound approach to handling noise in ordinal data.
2. The paper introduces two complementary loss functions: the discriminative loss and the stochastic order loss. The former is for embedding space construction by attracting to neighboring centroids and repelling from distant ones, and the latter enforces pairwise ordering relationships in a stochastic manner.
3. The outlier detection and relabeling scheme provides a practical way to refine noisy ranks, improving the overall reliability of the training data.
4. The algorithm is tested extensively across a variety of rank estimation tasks, including facial age estimation (MORPH II, CLAP2015), aesthetic score regression (AADB), medical assessment (RSNA), and textual regression (WMT2020), under synthetic (Gaussian, Laplacian, Uniform) and real-world noise settings.

### Weaknesses
1. While experiments demonstrate that the label refinement generally reduces MAE and standard deviation of noise, the relabeling scheme (Equation 20) uses a heuristic approach for the magnitude of label error correction (half of the mean absolute difference over all training instances). A stronger theoretical justification or a more adaptive mechanism for this step could enhance its reliability.
2. Some Missing related works[1,2,3]. Expecially, [1] used the normal distribution and the Gaussian kernel to model label ambiguity, which is similar to the noise modeling in this work.

[1] Gao, Bin-Bin, et al. "Deep label distribution learning with label ambiguity." TIP 2017

[2] Li, Shikun, et al. "Selective-supervised contrastive learning with noisy labels." CVPR 2022

[3] Liu, Yang, and Hongyi Guo. "Peer loss functions: Learning from noisy labels without knowing noise rates."  ICML 2020

### Questions
See above.

### Soundness
4

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
3

### Summary
The paper proposes an algorithm called Stochastic Order Learning (SOL) for ordinal data rank estimation under label noise. The authors model label errors as random variables and, based on this idea, introduce two loss functions — discriminative loss and stochastic order loss — to learn an embedding space that preserves rank order despite noisy annotations. Additionally, they design an outlier detection and relabeling mechanism based on the learned embeddings to reduce the effect of noisy labels.

### Strengths
- The authors correctly identify the limitation of existing ordinal regression and order-learning methods in handling label noise. The stochastic reformulation is an interesting conceptual step forward.
- Combining stochastic modeling of label errors with metric learning objectives is novel within the ordinal regression field.
- The experiments span diverse domains — facial age estimation, aesthetic score prediction, medical image assessment, and text regression — providing broad empirical context.

### Weaknesses
- The algorithm repeatedly relies on the exact probability values $𝑝_𝑠$ of the assumed noise distribution in equations (3), (9), and (12)–(17). However, in real-world tasks the noise variance σ is unknown. The authors only fix a constant “test value” 𝜎 test
during inference (see Eq. (22) on page 7 and Appendix C.3). This assumption undermines the theoretical validity of the method: since the noise distribution cannot be known or verified, the resulting stochastic weighting is arbitrary and not grounded in data.
- In equations (8)–(10), the discriminative loss aggregates weighted squared distances between each sample and multiple centroids.
However, this formulation does not ensure that the monotonicity constraint (Eq. (5)) holds.
Appendix A only shows that monotonicity is a sufficient condition, not a necessary one. The authors incorrectly reverse the logic — assuming that minimizing the proposed loss would imply monotonicity.
This is a clear logical inversion error, meaning the optimization process does not guarantee the intended ordered embedding structure.
- All experiments are conducted using synthetic noise (Gaussian, Laplacian, Uniform), despite the paper claiming to handle real-world noisy labels.
The only real-noise experiment (on WMT2020) shows merely about a 2% improvement, which is negligible given the additional model complexity. Although the paper claims the training cost is “acceptable,” Tables 19–21 show that SOL’s training time roughly doubles (87–100% slower) compared to the baseline GOL.
On larger datasets such as RSNA, a single epoch exceeds 1000 seconds — an impractical runtime for real-world use.

### Questions
see the Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes a novel method, Stochastic Order Learning (SOL), for robust rank estimation on label-noisy data. The method models label errors as random variables and learns an embedding space where each instance is encouraged to approach its stochastically related rank centroids. To achieve this, the authors design two loss functions, the discriminative loss and the stochastic order loss. After training, the method further improves data quality by detecting and relabeling outliers (instances with extreme label errors). Experiments on various datasets (facial age estimation, aesthetic score regression, medical imaging, and textual regression) demonstrate high accuracy and strong robustness to label noise.

### Strengths
his paper's strengths are as follows.

(1) This paper is the first study to address rank estimation with label noise, a setting that is pervasive in real-world scenarios, as noted in the paper. The contribution is thus highly significant for practical applications.

(2) The paper introduces a natural probabilistic model of label errors for rank estimation and proposes an appropriate learning framework based on this probabilistic formulation.

(3) The method demonstrates robust performance under both artificially generated and naturally occurring label noise.

### Weaknesses
This paper's weakness is as a follow.

(1)  The paper assumes label noise as formulated in Equation (2), but it remains unclear how the noise parameter σ is determined in real-world problems. During training, σ controls the amount of label corruption and is therefore crucial, yet in practice, this parameter is typically unknown. How is σ selected or estimated in real-world settings?

### Questions
(1) In real-world scenarios, does label noise actually follow the distribution assumed in Equation (2)? Moreover, how do the authors verify that such a type of label noise occurs in real data?

(2) From the quantitative results in Appendix D.3, the impact of outlier detection and relabeling appears very small. Why does removing outliers not lead to a more noticeable quantitative improvement?

(3) It is recommended to include visualizations of outlier detection on real-noise datasets (e.g., WMT2020).  Since detecting real noisy labels would be highly beneficial in practice, this would better showcase the potential of the proposed approach.

### Soundness
4

### Presentation
4

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
This paper proposes an algorithm called Stochastic Order Learning (SOL) for robust and reliable rank estimation in the presence of label noise. The key idea is to model label errors as random variables, following a discrete Gaussian distribution, and to learn an embedding space where instances are arranged according to their true ranks despite noisy labels. The method introduces two loss functions—discriminative loss and stochastic order loss, which are to enforce geometric constraints in the embedding space. Additionally, SOL includes an outlier detection and relabeling mechanism to refine the training data. Extensive experiments on facial age estimation, aesthetic score regression, medical assessment, and textual regression datasets demonstrate that SOL outperforms existing noise-robust classification, regression, and rank estimation methods under various synthetic and real-world noise settings.

### Strengths
1.	The paper tackles an important problem - label noise in ordinal regression.
2.	Extensive experiments across multiple domains (computer vision and natural language processing) and various noise types (Gaussian, Laplacian, and Uniform) demonstrate the effectiveness of the proposed approach.

### Weaknesses
1.	The method's performance is tied to the assumption of a symmetric, unimodal noise distribution, which is a key limitation in practical applications.
2.	Ablation studies and hyperparameter analysis are heavily focused on CLAP2015. It is unclear if the same settings and component importance hold for datasets with the largest gains (e.g., GDELT is not mentioned in this context, but the principle applies to datasets where SOL shines).
3.	The method introduces non-trivial computational cost compared to non-stochastic baselines, which could be a constraint.

### Questions
1.	The method assumes a symmetric noise model (Eq. 2). How would SOL perform if the real-world label noise is asymmetric (e.g., annotators consistently over-estimate ages)?
2.	The hyperparameter σtest is fixed during inference. Could the performance be further improved by making it adaptive or by estimating it from the data?
3.	The outlier relabeling uses a global average correction (Eq. 20). Have you explored instance-specific relabeling strategies, and why was a uniform correction chosen?
4.	The computational cost is higher than GOL. Are there strategies to improve the efficiency of the stochastic distance computations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The proposed stochastic order learning method frames orders as random variables, and develops discriminative loss and stochastic order loss to optimize network parameters. Experiments are conducted on benchmark facial age estimation datasets. Results show its superiority over baselines under different noise distributions.

### Strengths
-The proposed method models label errors as random variables and provides a solid theoretical basis.

-Stochastic order learning method is not sensitive to the prior noise distribution, shown in Table 1~4. Different noise distributions, such as Gaussian, Laplacian and uniform distribution, lead to similar performance.

-Extensive experiments are conducted on various datasets and results show its effectiveness for the age estimation task.

### Weaknesses
-Baseline methods are not comprehensive. A naïve method is to utilize these mature ranking loss functions in Learning to Rank methods, like RankNet and SoftRank. Similar idea has been implemented in SoftRank. These kinds of methods should be compared in the experiments.

-Compared with GOL in Table 1~4, the performance improvement of stochastic order learning method is marginal. 

-Compared with those benchmark loss functions, the computation complexity of the proposed stochastic order loss is higher. Moreover, the time complexity of the proposed loss function should be provided.

### Questions
Ranking loss functions, like RankNet and SoftRank, are not compared in the experiments.

### Soundness
2

### Presentation
3

### Contribution
2

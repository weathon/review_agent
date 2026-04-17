# Learning Generalized Label Distributions

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Label ambiguity is pervasive in supervised learning, motivating a variety of representations beyond the traditional single-label setting. While label distribution (LD) provides a probabilistic description and has attracted increasing attention, we reveal its inherent limitations, including inconsistency with raw data, distortion of inter-sample order, and limited applicability. To address these issues, we introduce generalized label distribution (GLD), a unified representation that can perfectly recover raw data while preserving inter-sample order consistency, transform into existing forms of label representations without information loss, and capture out-of-distribution samples as well as negative label correlations. We further develop GLD learning algorithms and demonstrate their effectiveness through both theoretical analysis and extensive experiments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript proposed generalized label distribution to solve the issues faced by current label distribution methods, such as inconsistency with raw data, distortion of inter-sample order, and limited applicability. And they demonstrate the effectiveness through
both theoretical analysis and extensive experiments.

### Strengths
The math seems solid. The writing is clear and easy to follow. But the significance is very limited.

### Weaknesses
The motivation is not sound or meaningless. What is the point to propose a generalized label distribution? At the beginning, the authors claim the label ambiguity exists, which means the label ambiguity is a problem while label distribution is a kind of method to solve the label ambiguity. But the authors take the label distribution as a research problem which is misleading. Just as written in the limitations by the authors, "applications of GLD are yet to be explored". It is meaningless to propose a generalized label distribution as a solution to solve the drawbacks of label distribution while no application setting need it; thus, the contribution is very limited.

### Questions
What is the point to propose a generalized label distribution?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the concept of a generalized label distribution (GLD) as a new representation of label ambiguity to address the inherent limitations of Label Distribution (LD) in supervised learning, which include inconsistency with raw data, disruption of inter-sample order, and inability to handle Out-of-Distribution (OOD) samples and negative label correlations. Moreover, inspired by the three types of algorithms for learning LDs (e.g., problem transformation, algorithm adaptation and specialized algorithm), this paper designs three Inspired by the distribution of learning labels, this paper designs three algorithms for learning GLDs: GLD-SVR corresponds to problem transformation, GLD-kNN corresponds to algorithm adaptation, and GLD-BFGS corresponds to specialized algorithm. They also adapt existing LDL methods (e.g., GLD-DF, GLD-LRR) to extend their applicability to GLD.

### Strengths
1. Through theoretical proofs and analyses in this paper, the author's definitions and proposals regarding GLD breaks through the limitations of LD, which were lacking in previous papers.
2. Theoretical analyses (mutual information, generalization bounds) are comprehensive.
3. The related work in this paper is comprehensive, citing classic LD literature and cross-domain studies like multi-label learning and OOD detection clearly distinguishing GLD from existing work. The structure of the paper is clear.

### Weaknesses
1. The experiments in the paper are thorough, but ablation experiments for core components of GLD-BFGS are absent, such as the Mahalanobis distance loss, so it is unclear whether the performance improvement comes from the GLD representation or algorithmic details.
2. All experiments are limited to low-dimensional labels and does not validate on high-dimensional tasks. GLD’s advantage persistence under label dimensionality scaling remains unconfirmed.
3. The LD OOD threshold is an empirical setting, and without cross-validation or comparison with other LD OOD methods, it can lead to unfair comparisons that exaggerate GLD's OOD advantages.
4. Key scales of the real datasets (e.g., the sample sizes, feature dimension for jaf/wq/enb) are not reported, and large-scale validation is not conducted, raising concerns about the generalizability of the results.
5. Robustness testing lacks quantitative descriptions of noise intensity, such as the standard deviation of Gaussian noise or the label zeroing rate, which reduces the reproducibility GLD's anti-interference results and weakens robustness claims.

### Questions
1. Could you add ablation experiments for GLD-BFGS’s core components? E.g.: Contrast GLD-BFGS with Mahalanobis loss vs. MSE loss to isolate loss function impact.
2. Could you validate GLD on high-dimensional label datasets?
3. Could you optimize the LD OOD threshold via cross-validation, or compare it with alternative LD OOD methods? This verifies if GLD’s OOD advantage is not due to LD metric bias.
4. Could you provide full scales of real datasets (sample size, feature dimension, label count) and validate GLD on large-scale datasets? This strengthens generalization assessments.
5. Could you supplement quantitative noise intensities in robustness tests?

### Soundness
2

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
4

### Summary
This paper revisits the problem of representing label ambiguity and theoretically demonstrates that the widely adopted label distribution (LD) representation exhibits several intrinsic limitations. To address these issues, the authors propose a novel representation, termed generalized label distribution (GLD), which unifies various existing forms of label representations under a single framework. GLD is theoretically analyzed and supported by newly designed learning algorithms, whose effectiveness is demonstrated through both analytical and experimental evaluations.

### Strengths
1. This paper provides a creative and interesting perspective: its novelty lies in the theoretical deconstruction of the conventional LD paradigm, which reveals inherent inconsistencies and motivates the formulation of a more expressive and principled GLD framework with broad potential impact.
2. This paper provides rigorous and convincing theoretical analyses that comprehensively expose the limitations of LD and justify the superiority of GLD and its associated learning algorithms.
3. The proposed GLD exhibits a strong unification property, elegantly and generally recovering raw data without information loss, and deriving other label forms.
4. The empirical evaluation of GLD algorithms reinforces the theoretical claims and enhances the overall credibility of the proposed framework.

### Weaknesses
1. Some mathematical descriptions are redundant and contain inaccuracies or lack sufficient elaboration. Please refer to the specific questions below.
2. The limitations of GLD are not thoroughly discussed. A more explicit elaboration on these aspects would better clarify the respective scenarios where LD and GLD are most applicable.

### Questions
1. Theorems 3.6-3.9 lack adequate explanation in the manuscript. What key insights or conclusions do each of these theorems reveal?
2. There appear to be potential issues with some mathematical notations: Is $i$ in Eqn. (2) redundant? Should $L_j$ in Lines 104-105 be denoted as $L_j^{*}$ instead?
3. While GLD is presented as a powerful framework, it may be challenging to apply in scenarios with limited raw data due to human or time constraints. What are the specific requirements for constructing GLD, and are there strategies to handle data scarcity?
4. The limitations of GLD are not thoroughly discussed. Are there specific aspects or scenarios where traditional LD remain irreplaceable? Could the authors provide a more in-depth discussion of its potential limitations?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper argues that conventional Label Distribution (LD) is intrinsically limited-it can be misaligned with raw targets, distort inter-sample order, and struggles with OOD detection and negative label correlations. 

It proposes a Generalized Label Distribution (GLD): a per-label bijective affine map of raw targets into $[-1,1]^c$ that is invertible and can be losslessly converted into other label forms (LL, TL, LR, SL). 

Experiments on synthetic and real datasets evaluate LL/LD prediction, ranking, and an OOD metric; GLD variants often outperform LD counterparts and show robustness under several noise processes.

### Strengths
1. GLD is bijective to raw targets, preserves inter-sample order, and permits lossless conversions among multiple label forms.

2. Clear information comparison (mutual-information inequality) and generalization bounds under a Mahalanobis loss.

3. Broad metrics (LL/LD prediction, rank correlations, an OOD metric) across synthetic and real datasets; GLD methods frequently top LD baselines.

### Weaknesses
1. Many LDL settings begin with label distributions collected from annotators; no continuous $q$ exists. The paper should clarify which problem families naturally expose $q$, and when GLD is more than renormalized multi-target regression.

2. The Mahalanobis loss relies on $\Sigma$; details on estimating/regularizing $\Sigma$ and sensitivity are light.

3. Several benchmarks are classic multi-output regression. To convince the LDL community, include canonical LDL datasets with crowd distributions and compare to Dirichlet/soft-label baselines and modern LDL methods.

### Questions
1. For OOD, please compare against standard embedding-space OOD baselines and report AUROC/FPR95, not only a bespoke OOD-Err.

2. Can you add ablations indicating that gains are not from the Mahalanobis loss alone (e.g., swap it into LD models)?

### Soundness
3

### Presentation
3

### Contribution
2

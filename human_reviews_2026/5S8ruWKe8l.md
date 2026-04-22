# Fast and Stable Riemannian Metrics on SPD Manifolds via Cholesky Product Geometry

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 4

## Abstract
Recent advances in Symmetric Positive Definite (SPD) matrix learning show that Riemannian metrics are fundamental to effective SPD neural networks. Motivated by this, we revisit the geometry of the Cholesky factors and uncover a simple product structure that enables convenient metric design. Building on this insight, we propose two fast and stable SPD metrics, Power--Cholesky Metric (PCM) and Bures--Wasserstein--Cholesky Metric (BWCM), derived via Cholesky decomposition. Compared with existing SPD metrics, the proposed metrics provide closed-form operators, computational efficiency, and improved numerical stability. We further apply our metrics to construct Riemannian Multinomial Logistic Regression (MLR) classifiers and residual blocks for SPD neural networks. Experiments on SPD deep learning, numerical stability analyses, and tensor interpolation demonstrate the effectiveness, efficiency, and robustness of our metrics. The code is available at https://github.com/GitZH-Chen/PCM_BWCM.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper revisits the Cholesky geometry of SPD matrices and exposes a product structure: a Euclidean metric on the strictly lower-triangular part plus (n) copies of a 1-D positive-reals metric on the diagonal. Leveraging this, the authors define two new Cholesky-side metrics -- Diagonal Power Metric (θ-DPM) and Diagonal Bures-Wasserstein Metric ((θ,M)-DBWM)--and pull them back through Cholesky to SPD metrics: Power-Cholesky Metric ($\theta$-PCM) and Bures-Wasserstein-Cholesky Metric (($\theta$,M)-BWCM). These metrics admit closed-form operators (geodesic, log/exp, parallel transport, Frechet mean) and gyrovector operations, while replacing diagonal log/exp by diagonal powers for improved numerical stability. Empirically, they integrate the metrics into Riemannian MLR classifiers and Riemannian residual blocks, showing accuracy/runtime advantages over AIM/LEM/LCM on Radar, HDM05, and FPHA; and they present large-scale stability tests (failure rates under tiny diagonal entries) favoring the proposed metrics.

### Strengths
* Originality: Product-geometry perspective on Cholesky leading to $\theta$-PCM/($\theta$,M)-BWCM with closed-form operators and gyro-structures. 
* Quality: Mathematical development is careful; operator summaries are comprehensive; empirical evaluations show consistent accuracy/runtime gains and robustness under tiny eigenvalues. 
* Clarity: Pipeline overview and tabulated operators (Table 1) facilitate adoption. 
* Significance: Practical drop-ins for SPD MLR and residual blocks; mitigates numerical brittleness of diagonal log/exp while retaining LCM-like utility.

### Weaknesses
1. Baseline breadth. Experiments primarily compare to AIM/LEM/LCM; PEM and (G)BWM baselines as end-to-end learners are not presented, leaving open how much improvement stems from closed-form convenience vs. intrinsic geometry. 
2. Assumption transparency. Gyro-structures and some exp/log formulas are locally defined and require ($L^\beta$) feasibility (e.g., ($L^\beta+K^\beta-I\in D_{++}$)); these constraints aren’t prominently quantified for training dynamics. 
3. Reproducibility details. While mean$\pm$std are reported, seed counts, episode splits (for cross-dataset parity), and tuning parity across metrics could be clearer; code is not yet available (this is highly preferred). 
4. Invariance discussion. The paper could explicitly contrast affine-invariance and other desirable properties of existing metrics vs. the proposed ones, and discuss trade-offs.

### Questions
1. Invariance & trade-offs. How do $\theta$-PCM/($\theta$,M)-BWCM compare to AIM’s affine-invariance in practice? Are there scenarios where losing/altering invariance harms performance, and can deformation ($\theta$) mitigate this? 
2. Baseline expansion. Can you include PEM and BWM/GBWM (even via numerical solvers) as end-to-end baselines on at least one dataset to calibrate geometry choice vs. closed-form convenience? 
3. Domain of validity. Please quantify the practical parameter/step-size ranges ensuring (L^\beta) feasibility during training; do you observe violations, and how are they handled (clamping, retraction)? 
4. $\theta$ sensitivity. Provide systematic sweeps of ($\theta$) (and (M)) across tasks; when does the limit ($\theta\rightarrow 0$) (LCM-like) help/hurt in practice? 
5. Scaling & large-scale tasks. Any results with high-dimensional SPD in modern deep nets (e.g., covariance pooling on ImageNet-scale features) to validate computational/storage gains? 
6. Code & seeds. Please clarify #seeds, hyperparameter parity, and release a minimal reference implementation for log/exp/transport under your metrics to facilitate adoption.

### Soundness
3

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
This paper revisits the geometry of the Cholesky manifold and identifies a product structure that decomposes the Cholesky factor into a Euclidean component. Leveraging this insight, the authors introduce two new Riemannian metrics on SPD manifold: PCM and BWCM. The authors apply the proposed metrics to build Riemannian MLR classifiers and residual blocks. The authors evaluate the two metrics and applications with 3 datasets: Radar, HDM05 and FPHA.

### Strengths
Replacing logarithms with power transforms to address numerical instability in LCM is well-motivated and empirically validated. The proposed metrics maintain closed-form. This paper is in general sound, the usability of the two proposed metrics is critical for deep learning integration.

### Weaknesses
This paper is in general well-wriiten, but there are still some concerns:
1. Several Riemannian and gyro-operators used in this work are only local defined. In particular, certain expressions are well-defined only under positivity constraints such as $L^{\beta} + K^{\beta} - I \in D_{n}^{++}$
2. 𝜃 power is a main contribution, however, the authors don't present how to choose 𝜃
3. Recent SPD learning works with GBWM-based classifiers are not experimentally compared
4. The stability of the DPM and DBWM are not empirically validated

### Questions
1. Can the authors provide guidance on how to choose the $\theta$ values
2. The authors presents the stability of the proposed metrics, but I didn't see how it is validated with experiments
3. As this paper is on Bures–Wasserstein–Cholesky metric, I think the authors should compare with othe works Bures–Wasserstein geometry, e.g.,
         1. Learning to Normalize on the SPD Manifold under Bures‑Wasserstein Geometry (Wang 2025)
         2. Learning with Symmetric Positive Definite Matrices via Generalized Bures‑Wasserstein Geometry (Han 2021)

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
This paper proposes two new SPD metrics called Power–Cholesky Metric (PCM) and Bures–Wasserstein–Cholesky Metric (BWCM). The proposed metrics are used to develop Riemannian Multinomial Logistic Regression (MLR) classifiers and residual blocks for SPD neural networks. The authors conducted experiments on radar signal classification, action recognition, and tensor interpolation to show the effectiveness of the proposed metrics.

### Strengths
- This paper aims at improving SPD neural networks which have applications in many fields. 
- In general, the exposition is clear and the paper is easy to follow.

### Weaknesses
- Contribution is marginal.

### Questions
This paper is heavily based on the works in [Nguyen and Yang, ICML 2023; Nguyen et al., ICLR 2024; Chen et al., NeurIPS 2024]. The proposed Riemannian metrics are formed by introducing small changes to existing ones.

For the experiments, while some improvements are shown on the chosen datasets, I do not think it is due to a better design of the proposed metrics compared to the well-known ones such as AIM, LEM, and LCM. This is because the performance of the proposed metrics depends on additional parameters (e.g., $\theta$) and one can easily figure out good settings for these parameters to get enhanced performance on some specific datasets. The main advantage of the proposed Riemannian metrics w.r.t. AIM and LEM is computation time, but as can be seen from the experiments, they have similar computation times as LCM. 

Overall, I think the paper presents marginal contributions to the literature of SPD neural networks which are not good enough for publication in ICLR 2026.

Question:

1. In terms of technical contributions, what is the novelty of this work w.r.t. those in [Nguyen and Yang, ICML 2023; Nguyen et al., ICLR 2024; Chen et al., NeurIPS 2024] ?

### Soundness
3

### Presentation
3

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
The authors proposed two new metrics for the Cholesky manifold inspired by the 
Log-Cholesky Metric (LCM) [Lin, 2019] and Bures–Wasserstein Metric (BWM) [Bhatia et al., 2019]. They show that the Cholesky manifold admits a simple decomposition into a linear space of strictly lower triangular matrices (SL) with the Euclidean metric and the product of positive numbers for the diagonal part. 
The specific choices of metrics for the later results in a novel 
$\theta$-Diagonal Power Metric ($\theta$-DPM) and $\mathbb{M}$-Diagonal
Bures-Wasserstein Metric ($\mathbb{M}$-BWM). The authors  develop closed formulas for geodesics, Riemannian logarithm, vector transport and weighted Frechet mean, which allows for 
deriving gyro-structures on the Cholesky manifold. Finally, the authors suggest 
using their metrics for SPD manifold as pullback metrics via Cholesky decomposition, which was proved to be a diffeomorphism in [Lin, 2019].

The authors claim that new metrics (both on the Cholesky manifold and SPD as a pullback) are demonstrating better numerical properties since 
they are based on linear operations and inversions instead of exponential functions. The authors have conducted two experiments for the Multinomial Logistics Regression (MLR) classifiers on Radar dataset [Brooks et al., 2019] (signal classification), 
HDM05 [Muller et al., 2007] and FPHA [Garcia-Hernando et al., 2018] datasets (human actions classification). The proposed approach is appears to be more reliable in the SPD interpolation task.

### Strengths
- Rigorous analysis of proposed Riemannian metrics for the Cholesky manifold: from geodesics to pullback metrics for SPD manifold.
 - Stability experiments: empirical demonstration of failure probability for the derived metrics.

### Weaknesses
- I am not sure if the composition of logarithm and exponentiation from LCE is not possible to be done using some standard numerical tricks like logsumexp or smth in this direction. Could you give the exact formula for the unstable part of LCE and reason why there is no any standard workaround? I will be ok with raising the score if this part is properly addressed, because it seems to be one of the key advantages of choosing your method.
- Given that $\theta$ is a central contribution of this work, an accurate ablation study is important to isolate its effect. Similarly regarding the matrix $\mathbb{M}$ across the $\theta$-DPM, $\mathbb{M}$-BWM, and $(\theta, \mathbb{M})$-BWM models.

### Questions
- Have you conducted experiments for failure probabilities estimation for pullback metrics on the SPD manifold? 

 - How can I estimate $\theta$ parameter for a dataset in advance? Why is it negative in Tab. $8$ (lines $918$-$924$)? Is there any analysis for negative $\theta$?
 
 - What does the resulting asymptotical complexity of the proposed metrics equal to?

### Soundness
3

### Presentation
2

### Contribution
2

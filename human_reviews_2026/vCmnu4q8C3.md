# Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 4, 4, 4

## Abstract
Prediction intervals are essential for applying machine learning models in real applications, yet most conformal prediction (CP) methods provide coverage guarantees that overlook the heterogeneity and domain knowledge that characterize modern multimodal datasets. We introduce Mixture-of-Experts Conformal Prediction (MoE-CP), a flexible and scalable framework that uses the gating probability vectors of Mixture-of-Experts (MoE) models as soft domain assignments to guide similarity-weighted conformal calibration. MoE-CP weights calibration residuals according to the similarity between gating vectors of calibration and test points, producing prediction intervals that adapt to latent subpopulations without requiring explicit domain labels. We provide theoretical justification showing that MoE-CP preserves nominal marginal validity under common similarity measures and improves conditional adaptivity when the gating captures domain structure. Empirical results on synthetic and real-world datasets demonstrate that MoE-CP yields more domain-aware, interpretable, and often tighter intervals than existing conformal baselines while maintaining target coverage. MoE-CP offers a practical route to reliable uncertainty quantification in latent heterogeneous, multi-domain environments.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a variant of the randomly localized conformal prediction (RLCP) introduced by Hore and Barber. The approach evaluates similarities between inputs in the latent space defined by the vector of probabilities output by the gating mechanism of a MoE model. The key idea is that inputs with similar statistics are also likely to be routed to the same experts if the gating mechanism is well trained.

### Strengths
The proposed approach is sound, retains statistical validity, and leverages advances in MoE architectures. 

The experimental results, while limited, are sufficient to support the main claims of the paper.

### Weaknesses
The submission may not be sufficiently clear in the earlier sections about the relationship of this work with Hore-Barber. I think that this should be made clearer when presenting the contribution. As is, this is only discussed in Remark 5 on p. 5.

It is unclear a priori why the same temperature parameter tau is used in both (3) and (4). 

Assumption 1 is not clear. The authors claim that the assumption holds for a variety of divergences, but they do not provide any supporting evidence for this.

There are some typos, such as "conformapl" on p. 1 and the missing space in "by(3)" in Algorithm 1.

### Questions
1) Why is the same temperature parameter tau used in both (3) and (4)?

2) Under what conditions is Assumption 1 verified?

### Soundness
3

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
4

### Summary
This paper proposes MoE-CP, which uses MoE gating vectors as soft domain assignments to weight calibration residuals in conformal prediction. The method aims to produce adaptive prediction intervals for heterogeneous data without explicit domain labels, with theoretical guarantees for marginal validity and empirical validation showing tighter intervals than standard conformal methods.

### Strengths
Creative use of MoE gating vectors for conformal prediction weighting addresses heterogeneous data limitations.

Comprehensive analysis including marginal validity (Theorem 1), robustness to divergence choice (Theorem 2), and conditional coverage under mixture representation (Theorem 3).

### Weaknesses
Requires training an MoE model before conformal calibration, adding significant computational cost compared to standard conformal methods.

Multiple hyperparameters require careful tuning, and there is no detailed computational cost analysis compared to baseline methods.

Core assumption that MoE gating vectors meaningfully capture domain structure may not hold in practice, especially when true domains don't align with MoE's learned partitioning.

Limited discussion of how to validate whether MoE gating provides good domain separation.

MoE models are trained to minimize MSE loss, not to learn domain boundaries, creating fundamental misalignment between training objective and intended use.

Real datasets are low-dimensional, limiting generalizability to modern large-scale applications. Synthetic experiments only consider simple polynomial relationships with clear domain boundaries.

Randomization step in equation (3) appears somewhat ad-hoc without strong theoretical justification.

Exchangeability assumption may be violated in truly multi-domain data where domains have different distributions.

### Questions
The authors should address the concerns raised in the weakness section above.

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
3

### Summary
The paper proposes Mixture-of-Experts Conformal Prediction, which integrates conformal prediction with mixture-of-experts models. It assigns higher weights to calibration samples whose gating vectors are more similar to the test point’s, thereby adapting prediction intervals to latent domain structures.

### Strengths
1- The paper introduces an interesting and well-motivated idea that combines Mixture-of-Experts (MoE) models with conformal prediction

2- The paper provides a theoretical analysis, proving marginal validity and approximate conditional coverage

### Weaknesses
Please check the questions!

### Questions
1- The authors should clarify the precise meaning of $\mu(x)$ and $\pi(x)$. While both are formally defined, their conceptual interpretation and how they are learned from data remain vague.

2- It is unclear how the number of experts and the temperature parameter are selected in practice. Please explain how you chose these parameters.

3- Since the approach involves computing similarity-based weights for each test point, the computational complexity may be substantial for large calibration sets. It would be helpful if the authors could provide a computational complexity analysis and discuss practical runtime implications.

4- There are existing conformal prediction approaches that incorporate expert advice [1, 2]; the authors are encouraged to provide a comparison with these methods, either conceptually or empirically

5-The impact of the randomization step (via multinomial sampling in Eq. 3) on interval variability and stability is not analyzed. Please provide an analysis to show its impact

[1] Hajihashemi, E. and Shen, Y., 2024. Multi-model ensemble conformal prediction in dynamic environments. Advances in Neural Information Processing Systems, 37, pp.118678-118700.
[2] Gibbs, I. and Candès, E.J., 2024. Conformal inference for online prediction with arbitrary distribution shifts. Journal of Machine Learning Research, 25(162), pp.1-36.

### Soundness
2

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
4

### Summary
The authors introduce Mixture-of-Experts Conformal Prediction (MoE-CP), a flexible method that uses the gating probability vectors of Mixture-of-Experts (MoE) models to estimate the similarity between test sample and calibration samples, thereby achieve adaptive conformal prediction.

### Strengths
1. Sufficient discussion about background and clear introductive figure.

2. Extensive ablation study on hyperparameter selection of the proposed method.

### Weaknesses
1. This work basically follows the logics of prior research: compute the similarity between calibration samples and a give test sample to estimate local conformal score density, which is used to output a local 1-alpha threshold for prediction sets. Thereby, the overall contribution is incremental.

2. The gate model typically plays a role of PCA to facilitate the density estimation. In other words, the number of expert models K must be sufficiently smaller than feature dimensions to be functional, which limits the application of the proposed method.

3. Density-estimation-based localized CP , such as [1,2], are sensitive to hyperparameter selection. The authors only mention which base predictive models are used, without discussing if their hyperparameters are tuned carefully. Hence, the experiement results may be unfair. 4. Recently, generative models for adaptiveness progressed, such as [3,4], which are missed in the work. 

[1] Leying Guan. Localized conformal prediction: A generalized inference framework for conformal prediction. Biometrika, 110(1):33–50, 2023.
[2] Rohan Hore and Rina Foygel Barber. Conformal prediction with local weights: randomization enables robust guarantees. Journal of the Royal Statistical Society Series B: Statistical Methodology, 87(2):549–578, 2025.
[3] Colombo, Nicolo. "Normalizing flows for conformal regression." arXiv preprint arXiv:2406.03346 (2024).
[4] Fang, Zhenhan, Aixin Tan, and Jian Huang. "CONTRA: Conformal prediction region via normalizing flow transformation." The Thirteenth International Conference on Learning Representations. 2025.

### Questions
1. What about extending your idea to classification? What challenge will you face?

2. What is the difference between the similarity weights from gate model and a density estimator?

### Soundness
2

### Presentation
2

### Contribution
2

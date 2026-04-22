# Fair Conformal Classification via Learning Representation-Based Groups

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Conformal prediction methods provide statistically rigorous marginal coverage guarantees for machine learning models, but such guarantees fail to account for algorithmic biases, thereby undermining fairness and trust. This paper introduces a fair conformal inference framework for classification tasks. The proposed method constructs prediction sets that guarantee conditional coverage on adaptively identified subgroups, which can be implicitly defined through nonlinear feature combinations. By balancing effectiveness and efficiency in producing compact, informative prediction sets and ensuring adaptive equalized coverage across unfairly treated subgroups, our approach paves a practical pathway toward trustworthy machine learning. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of the framework.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for fair conformal inference in classification tasks. The method adaptively identifies subgroups and enforces conditional coverage guarantees on these learned groups. To evaluate subgroup fairness, the authors introduce a nonlinear variant of the Worst Slab Coverage (WSC) metric, designed to better capture coverage deficiencies in complex or non-linearly defined groups. Experiments are conducted on synthetic data and one real-world dataset.

### Strengths
- The approach is **novel**: instead of enumerating or greedily constructing subgroups, the authors learn subgroup structure via representation learning—allowing for more complex group definitions (for instance, XOR-type interactions) that are hard to capture with feature-based heuristics.
- It **improves over the baseline AFCP** in terms of achieving fairer conditional coverage, with only a modest reduction in efficiency (prediction set size). The substantial runtime improvement makes it a much more practical alternative.
- The **methodology is theoretically sound**, supported by clear formulations and proofs that establish its coverage guarantees.

### Weaknesses
- The **empirical evaluation is limited**, with most experiments conducted on synthetic data and only a single real-world dataset (Nursery). Including larger and more commonly used datasets—such as the *Folktables* benchmarks used in prior fairness/conditional conformal prediction work ([1]–[3])—would strengthen the experimental evidence.
- The paper does not discuss the **sensitivity of the hyperparameter β** in the final loss function, which may affect the fairness–efficiency trade-off.
- A few parts of the **writing could be polished for clarity**, including the caption for Figure 2/Table 1 and the paragraph starting at line 405.

References

[1] O. Bastani et al: Practical adversarial multivalid conformal prediction [NeurIPS 2022]

[2] C. Jung et al: Batch Multivalid Conformal Prediction [ICLR 2023]

[3] AT. Vadlamani et al: A Generic Framework for Conformal Fairness [ICLR 2025]

### Questions
See weaknesses.

- The paper focuses on **equalized coverage** as the fairness notion for conformal prediction. However, other fairness notions—such as those derived from popular ML fairness metrics—have been considered in related work ([3]). Does your method naturally extend to these alternative definitions, and if so, how?

References

[3] AT. Vadlamani et al: A Generic Framework for Conformal Fairness [ICLR 2025]

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes FAREG, a conformal prediction method that ensures fairness by adaptively identifying “unfair” subgroups in a learned latent representation space. Unlike prior conformal predictors with equalized coverage that only consider simple or pre-defined groups (e.g. single sensitive features), FAREG learns a representation $Z = f(X)$ via a variational encoder–decoder and discovers complex subgroups (even defined by nonlinear feature combinations like XOR) associated with low coverage. The method then adjusts prediction sets to guarantee adaptive equalized coverage for these discovered subgroups. Additionally, the paper introduces WSC+, a “nonlinear” worst-case conditional coverage metric extending the worst-slab coverage (WSC) metric of Cauchois et al. (2021)

### Strengths
The paper tackles the important intersection of uncertainty quantification and algorithmic fairness. Ensuring no subgroup is underserved by predictive uncertainty is crucial for trustworthy AI. The work is timely and relevant to high-stakes domains. FAREG’s combination of representation learning with conformal prediction is novel. It significantly extends prior fair conformal methods (from single-feature groups to rich latent groups).

### Weaknesses
- The proposed algorithm is complex, involving a custom VAE training and Monte Carlo sampling of groups. Some steps (the PGD projection and the final aggregation of prediction sets) are not explained in depth in the main text. 

- FAREG focuses on one subgroup (or a mixture of one) to protect. If there are multiple distinct biased subgroups, it’s unclear if FAREG can handle them simultaneously.

- some relevant baselines and related works are not mentioned. for instance  (https://arxiv.org/pdf/2505.16115) or (https://arxiv.org/abs/2305.12616)

### Questions
- Could you elaborate on how exactly the $T$ samples of $S$ (Algorithm 1, lines 12–16) are used to form the final prediction set $C(X_{N+1})$?


- Under what assumptions does FAREG guarantee $P(Y \in C(X) \mid X \in \hat G) \ge 1-\alpha$ for the discovered group $\hat G$? Is this guarantee exact finite-sample (with sample splitting) or only asymptotic/high-probability?

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
The paper introduces FAREG, a fair conformal classification method that finds and fixes under-covered subgroups so that they reach the target coverage, without blowing up prediction-set size. Here are the mains steps of the method.
1. Learn a compact representation Z of the features X.
2. On this Z, train a small classifier that scores how likely each sample belongs to an “unfair subgroup” (i.e., tends to be under-covered).
3. Encourage this classifier to pick the samples with the lowest conditional coverage while keeping the subgroup at least a fraction δ of the data (for statistical reliability).
4. Build a standard conformal prediction set for everyone, then build extra subgroup-specific sets for the selected groups, and take the union. This guarantees adaptive equalized coverage for those groups.

### Strengths
1. The problem is meaningful and important. 
2. The method appears technically sound. 
3. Although the full pipeline is more involved than the summary above, the paper is mostly readable.

### Weaknesses
First, the goal of this work is a subset of existing objectives. Prior work has focused on conditional coverage
$$
\mathbb{P}\!\left(Y \in C(X_{n+1}) \mid X_{n+1}=x\right)=1-\alpha,
$$
which is stronger than
$$
\mathbb{P}\!\left(Y \in C(X_{n+1}) \mid X_{n+1}\in \widehat{\mathcal{G}}\right)=1-\alpha.
$$
If a conformal method attains (approximate) conditional validity, the proposed method may be less useful. It would be helpful to show that results still improve when FAREG is built on top of conditionally valid conformal methods such as~[1]. Second, based on Figures 2 and 5, gains in coverage appear to come at the cost of efficiency: the proposed method achieves higher coverage but also larger prediction sets.

[1] Isaac Gibbs, John J Cherian, and Emmanuel J Cand`es. Conformal prediction with conditional guarantees. Journal of the Royal Statistical Society Series B: Statistical Methodology, pp. qkaf008, 2025.

### Questions
What is the hypothesis class $\mathcal{H}$ in the proposed algorithm? What is the VC dimension of $\mathcal{H}$ in this setting?

### Soundness
3

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
The authors introduce FAREG, a method to employ variational information bottleneck to achieve fair conformal prediction by identifying worst performing subgroups. They also propose an improvement over the WSC metric to capture non-linear slabs. They perform experiments on a self-constructed synthetic dataset and the Nursery dataset, comparing against standard baselines. The authors also provide code supporting the reproducibility of their method.

### Strengths
- **[S1]** The authors do a great job of motivating the paper well. Figure 1 is a great example of how the current set of conformal prediction methods struggle in certain data settings. There is a clear gap in the literature that needs to be addressed and the manuscript is an attempt at that.
- **[S2]** The core idea of FAREG is quite novel, I haven’t come across works using variational information bottlenecks in this context and it feels like a natural fit for potentially non-linear subgroups.
- **[S3]** The synthetic experiment in Section 4.2 is designed perfectly to highlight the proposed method's strength compared to the baselines and is a great way to show the applicability of the approach.
- **[S4]** The code seems succinct and sufficient, supporting the neat empirical claims in the paper.

### Weaknesses
- **[W1]** Figure 4(a) and the claims in Line 374 that the time complexity of FAREG is linear in the number of data instances seems misleading, given that this does not account for the training of the encoder-decoder network which is a non-trivial computation. ACP is a post-processing algorithm which does not involve training of any such network and therefore a fairer comparison would be to compute the wallclock times of the entire algorithms from start to end.
- **[W2]** While the synthetic experiment is a great demonstration of the algorithm's strength, it feels like the perfect application in terms of the XNOR-like bias but the authors do not include any other synthetic constructions that could help understand the algorithm's utility in general situations. 
- **[W3]** It’s unclear to me how the following statements all hold true together: Line 87: The interpretability is enhanced; Line 483: Expressivity may sacrifice interpretability; Line 941: This result strengthens the interpretability. Intuitively, it seems that the 2nd statement in the limitations is true, so what do the other two statements mean?
- **[W4]** The proof for Theorem 1 in Section A.4 is extremely handwavy. It uses the Theorem 1 from (Zhou & Sesia, 2024) but never establishes how it is better than it formally. There’s a strange argument about the VC Dimension and higher expressivity written in words, but no formal proof of the statistical consequences on learning the groups on the same data used for calibration.

### Questions
- **[Q1] ** See [W1]. Can you provide a more detailed and fair comparison of the wallclock time of FAREG and AFCP, showing the entire training, processing and postprocessing steps.
- **[Q2]** See [W2]. Would be curious to see applications of FAREG on other synthetic setups, which are slightly more complicated than XNOR for example.
- **[Q3]** See [W3]. Could you clarify the confusion about the interpretability of the algorithm and state what the effect of the encoder-decoder architecture is?
- **[Q4]** See [W4]. Could you provide a more formal proof for Theorem 1, clearly highlighting the superiority over AFCP and the effects of that?

### Soundness
2

### Presentation
3

### Contribution
3

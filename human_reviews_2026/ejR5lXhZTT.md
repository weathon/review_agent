# Using Noise to Help Reach Global Minima: Turning Matrix Completion into Noisy Matrix Sensing

- Avg Score: 6.00
- Decision: Reject
- Scores: 8, 6, 4

## Abstract
Matrix completion (MC) is an important yet challenging non-convex problem. In realistic settings, exact recovery of $M^*$ typically requires strong incoherence and an impractically large number of observed entries. Instead of enforcing exact recovery, we inject noise perturbation to construct a closely related surrogate that turns MC into a noisy matrix sensing problem with a more benign landscape. Although this surrogate permits a slight, controllable loss in accuracy, it can be solved effectively via over-parameterization (increasing model size), echoing modern machine learning practices where large models and stochasticity (e.g., SGD, dropout) make hard objectives tractable. Under the assumption that each entry of the matrix is observed independently and uniformly, we establish explicit accuracy–probability trade-offs as functions of the sampling rate $p$ and a user-chosen noise level. Empirically, our approach succeeds in low-observation regimes where classical exact-recovery pipelines are brittle. More broadly, our approach underscores a general paradigm in which noise perturbations are combined with large models to tackle modern ML tasks, and we use MC as a clean benchmark to formalize this perspective and unify noise, over-parameterization, and recoverability within a single framework.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies a surrogate formulation of the classical matrix completion problem, which is inherently non-convex and typically requires strong incoherence and high sensing rates for exact recovery. The authors propose converting the problem into a noisy matrix sensing formulation by injecting controlled noise perturbations, thereby ensuring valid Restricted Strong Smoothness (RSS) and Restricted Strong Convexity (RSC) conditions.

The paper first establishes conditions under which this surrogate problem, termed $\epsilon$-MC, admits a global solution close to the ground-truth matrix (Theorem 2.2). A key contribution is the derivation of an explicit accuracy–sampling probability trade-off, parameterized by the sampling rate $p \in [0,1]$ and the perturbation level $\epsilon$.

While this formulation guarantees valid RSS and RSC constants, it still suffers from large RIP constants, making optimization challenging.
To address this, the authors adopt and extend the lifted tensor framework of Ma et al. (2024) to the noisy setting, leveraging overparameterization to guarantee favorable optimization geometry—specifically, that all non-global critical points are strict saddles. Numerical experiments comparing the proposed tensor PCA solver with several baselines (e.g., BM factorization, SDP relaxation, spectral reweighting) demonstrate improved recovery success rates across both independent and structured sampling regimes.

### Strengths
* The paper introduces a perturbed matrix completion formulation that ensures valid RSS and RSC constants and provides an explicit quantitative relationship between recovery accuracy, sampling rate, and noise level.

* It extends the lifted tensor framework to the perturbed setting, proving guarantees for local minima and establishing that spurious solutions become strict saddles under the proposed formulation.

* The inclusion of numerical experiments—though the paper is largely theoretical—adds credibility and demonstrates the practical advantage of the proposed approach over multiple baselines.

### Weaknesses
* Scalability:
The lifted-tensor solver is memory-intensive and currently applicable only to small-scale matrices. The paper does not discuss possible algorithmic strategies, approximations, or structural assumptions that could make the proposed method practical for large-scale problems.

* Information-theoretic limits:
While the paper derives explicit performance–sampling trade-offs, it does not discuss how close these results are to information-theoretic recovery limits for matrix completion. A comparison or partial converse analysis could provide valuable insight into the tightness and fundamental optimality of the proposed bounds.

### Questions
* Are there potential directions that could make the proposed approach more scalable?

* Can the authors provide any converse-style analysis or information-theoretic lower bounds on recovery accuracy in terms of the sampling rate $p$?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors address the matrix completion problem and propose to reformulate it as a noisy matrix sensing problem. They establish theoretical conditions showing that the solution to this reformulated problem remains close to that of the original matrix completion formulation, and they further propose handling the problem through a lifted tensor framework.

### Strengths
Compared with existing approaches, the proposed method enables successful completion even when the sampling rate p is very small, though this comes at the cost of reduced accuracy. The idea is interesting, and the paper is well written, well structured, and clearly presented.

### Weaknesses
I have a concern regarding the experimental validation. The authors evaluate the success rate over 20 trials using a fixed estimation error threshold and a fixed noise level. Although an explanation is provided for this choice, the experimental setup, in my view, does not convincingly support the paper’s main claim—that the proposed method enables completion at low p values but sacrifices accuracy, especially with a user-chosen noise level. To substantiate this point, the experiments should be improved, either by including additional results (e.g., varying the noise level or error threshold) or by providing a more detailed justification and interpretation of the current setup.

### Questions
The analysis focuses solely on the overall estimation error. It would be interesting to explore where the reconstruction errors occur relative to the ground truth. Are these errors concentrated in specific regions of the estimated matrix, or are they more uniformly distributed?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The author studies how to inject noise perturbation to turn matrix completion problem into a noisy matrix sensing problem, which can be easier to solve. They show that the induced matrix sensing problem has a  RIP property and establish the accuracy-probability trade-offs for the noisy matrix sensing problem. Then, they employ the lifted tensor framework to further deal with the matrix sensing problem, to overcome the small RSC constant issue.

### Strengths
1. The paper is well-structured and easy to follow. The high-level motivation and the intuitive explanation about the main theoretical results are provided in a clear way. 

2. The theoretical contributions are solid. The idea of introducing perturbation noise into a matrix completion problem and reformulating it as a noisy matrix sensing problem is novel. The paper also presents interesting theoretical results, such as the probability–accuracy trade-off, which is a new contribution in this area based on my knowledge.

### Weaknesses
The main concern lies in the interpretation of the derived theoretical results. In Theorem 3.1, the upper bound includes the term $1/\alpha_s$. However, since the transformed matrix sensing problem only has a small RSC constant, $\alpha_s=\varepsilon^2$ could be very small. In fact, in your experiment, the $\varepsilon$ is set to be $1e-5$, which would render the theoretical bound practically meaningless. Similar concerns apply to Theorem 4.1, where the upper bound also contains a $1/\varepsilon$ term, which is also very large when $\varepsilon$ is set to be 1e-5.

Then, if the perturbed noise matrix sensing problem has a small RSC constant and will finally lead to a practically meaningless result, the advantage of using this perturbation becomes unclear.

### Questions
1. Could the authors clarify that why  the proposed approach still outperforms other baselines even if $\varepsilon$ is set to be 1e-5 in the experiment? Does this suggest that the theoretical results may leave substantial room for improvement?

### Soundness
3

### Presentation
3

### Contribution
2

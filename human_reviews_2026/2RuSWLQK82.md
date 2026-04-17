# Towards a Certificate of Trust: Task-Aware OOD Detection for Scientific AI

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Data-driven models are increasingly adopted in critical scientific fields like weather forecasting and fluid dynamics. These methods can fail on out-of-distribution (OOD) data, but detecting such failures in regression tasks is an open challenge. We propose a new OOD detection method based on estimating joint likelihoods using a score-based diffusion model. This approach considers not just the input but also the regression model's prediction, providing a task-aware reliability score. Across numerous scientific datasets, including PDE datasets, satellite imagery and brain tumor segmentation, we show that this likelihood strongly correlates with prediction error. Our work provides a foundational step towards building a verifiable 'certificate of trust', thereby offering a practical tool for assessing the trustworthiness of AI-based scientific predictions.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a task‑aware (mostly regression-based tasks) out‑of‑distribution (OOD) detector for scientific ML that estimates a joint likelihood $\log p(x, y_{\text{pred}})$ with a score‑based diffusion model and uses it as a reliability certificate for an arbitrary downstream predictor. The key intuition is a heuristic link between prediction error and joint likelihood (Eq. (2)), and the practical pipeline solves the probability‑flow ODE to score new data points.

Decision thresholds are set from a small set of training samples to separate ID vs OOD (Fig. 1C) and the same score can be used to fit an a‑posteriori error curve when a few labeled test samples are available (Fig. 1D and Fig. 24). Valid experiments are run on canonical datasets: regression on PDE operators (wave and 2D Navier–Stokes), satellite humidity forecasting (MERRA‑2), image classification (MNIST, CIFAR‑10), and brain‑tumor segmentation The proposed method is compared with several diffusion-based baselines on these setups with metrics like ACC, FPR, FDR to demonstrate the effectiveness of this joint likelihood as compared to input-only likelyhood $p(x)$ in correlation to prediction error. Ablation studies are conducted as well to further isolate the impact of number of calibration samples and different task models.

### Strengths
1. Empirical result appears to be strong. Fig.2 and Table 3 are good illustration indicative of why $p(x)$ could fail for scientific regression and why joint likelihood is a better proxy.
2. Experiment setups are carefully crafted and elaborated. Full details are presented, and a wide spectrum of data sources are provided with proper justifications.
3. The proposed method is intuitively sound within the context of score-based diffusion, and appears to be robust to various scientific regression tasks. 
4. Simple exponential fit provides actionable error estimates once a few test labels are available.
5. Related studies is well discussed, demonstrating the gaps in this field, and strengthening the originality of the work
6. The method shows a certain degree of significance by filling the gaps in good predictors for regressor tasks. However, given the lack of discussion on its computational costs and complexity in real scenarios, it is hard to gauge whether it will be meaningful/provide additional insights for more practical methods in this area.
7. In general, given the depth of the discussion and the breath of the experiments conducted, it is a high quality presentation.

### Weaknesses
1. Lack of theoretical supports and further discussions: Equation (2) assumes approximately uniform training error over $p(x,y)$. This key step is not formalized; conditions under which the correlation degrades are not identified (e.g., heavy label noise, multi‑modal conditionals $p(y|x)$ with identical likelihood but disparate errors). Strengthening or bounding the approximation would help.
2. The choice of metrics can be better justified. The metrics used (ACC, FPR, FDR) can be swapped with other common OOD metrics (AUROC, TNR@TPR95) for stronger comparisons to prior works
3. (optional) To strengthen the significance of the work, comparison other baselines other than diffusion-based scores should be considered. For example, (i) likelihood‑ratio methods (e.g., Ren et al., 2019) applied to joint or representation spaces; (ii) ensembles/MC‑dropout as task‑aware uncertainty proxies; (iii) conformal risk‑coverage curves for selective prediction; (iv) energy‑based scores on joint representations.
4. The “median − 1.5σ” rule (Sec. 3; App. A.4) is somewhat ad‑hoc despite ablations (Table 2 & Table 4). More principled calibration (e.g., quantile‑conformal bands or controlling FPR at target levels) would increase reliability.
5. Computing joint log-likelihood requires solving the probability‑flow ODE and estimating divergences with 32 MC samples (App. A.3). Runtime, memory, and scalability versus input/output resolution and time‑steps are not quantified. This matters for high‑res scientific workloads -> Potentially limited practicality and thus significance.
6. Code availability are not provided.

Reference:
Jie Ren, Peter J. Liu, Emily Fertig, Jasper Snoek, Ryan Poplin, Mark DePristo, Joshua Dillon, Balaji Lakshminarayanan. “Likelihood Ratios for Out‑of‑Distribution Detection.” NeurIPS 2019.

### Questions
1. On Eq. (2):  Can you formalize conditions under which the error–likelihood correlation provably holds (e.g., bounded density ratio between train/test, Lipschitz loss, coverage of $p(y|x)$? Conversely, can you build a counterexample where $\log p(x, y_{\text{pred}})$ is high but error is high (beyond the anecdotal segmentation failure without noise)?
2. In NS experiments, you average direct and autoregressive likelihoods (Sec. B.2.2, Figs. 18–20). Can you quantify when each dominates and whether the mixture is necessary if the regressor is trained with scheduled rollouts?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose Joint Likelihood-Based Certificate (JLBC), which uses a diffusion model to perform out-of-distribution (OOD) detection. The core idea of this method is to train a diffusion model to approximate the joint distribution $p(x, y_\text{pred})$ where $y_\text{pred}$ is the output of the base regression model. The log-likelihood estimated from the diffusion model serves as a certificate of trust, which correlates with the base model's prediction error. Experimental results over a number of domains (e.g., medical, wave, weather) support the validity of this correlation.

### Strengths
1. The method doesn't require any retraining of the base regression model, reducing computational overhead. Moreover, the method operates zero-shot (aka it doesn't require any ground truth OOD vs. ID labels).
2. The use of diffusion models to represent / estimate joint likelihoods is well studied / supported.
3. Empirical studies. The experiments are performed over a number of different datasets and downstream tasks, giving a good idea of the expected performance of the method in real-world scenarios.

### Weaknesses
1. Novelty and distinction from prior works is somewhat unclear. Section 2 references a number of OOD detection methods and yet concludes with "[there] is [a] scarcity of OOD detection...methods for...scientific machine learning applications," which seems contrary. Additional discussion specifically on the shortcomings of previous methods---which are addressed by the proposed method---should be included. Moreover, further discussion on diffusion / learning-based methods for uncertainty estimation should be in mentioned, since many of these methods seem relevant (e.g., estimation of epistemic uncertainty is often used for OOD detection).
2. Computational overhead. Diffusion model training and inference can be expensive, perhaps even more so than the underlying regression model. 
3. Conciseness. The writing in the manuscript is quite dense (particular the results section) and could benefit from more clear / concise organization.
4. Lack of theoretical guarantees. The method proposes the joint likelihood estimate as a certificate of trust, yet there are no concrete guarantees on the correctness of the estimate (i.e., OOD prediction's) accuracy.

### Questions
1. How does the proposed approach compare to uncertainty-based methods (e.g., IGRUE, HyperDM, Rate-in, Bayesian NNs, MC dropout, etc.) on the same tasks?

### Soundness
4

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
2

### Summary
This work proposes a method to classify test examples as in-distribution or out-of-distribution. To this end, the authors propose a diffusion model to compute the certificate (joint likelihood of test example and the model prediction). The experiments include two PDE datasets, a satellite-data based forecasting, image classification and brain tumor segmentation. Across all datasets, the estimated certificate strongly correlates with prediction errors on ID vs OOD data. 

While the results seem to be strong, the results section is really hard to follow with most tables and figures in the Appendix. I'd encourage the authors to move some of the key results to the main text.

### Strengths
- The paper studies an important research question and presents strong results across a wide variety of domains and tasks (regression and classification)
- The proposed method works both in a zero-shot setup as well as with a setting with ground-truth examples.
- When compared with other diffusion-based baselines (Table 1), the proposed method outperforms across all datasets.

### Weaknesses
- The paper currently misses out on connecting the baselines (lines 425-431) with previously published work. Its unclear if any of these are directly related to the works cited in section 2.
- Its unclear if the gains reported in Table 1 are statistically significant. It would be helpful to include some tests for this.
- The main text (section 4, pages refers extensively to the results in the Appendix, this makes it really hard to follow the arguments. I'd encourage editing the paper to move key results to the main text.

### Questions
- Whenever applicable, provide citations for the baselines.
- There is only a short discussion in the paper regarding posterior estimates. I'd encourage expanding this subsection.

### Soundness
4

### Presentation
2

### Contribution
4

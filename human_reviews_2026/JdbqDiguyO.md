# Noise-Robust Density Estimation for Tabular Data Anomaly Detection

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Density-based anomaly detection methods often provide accurate and interpretable predictions but their performance can be severely affected by the inherent noise of data. In this paper, we present a noise-robust density estimation (NRDE) method for tabular data anomaly detection. We aim to estimate density of pure data with influence of noise isolated, which is a non-trivial task since data-generating process is completely unknown. NRDE learns a Jacobian-regularized normalizing flow to estimate the sources of data and categorizes sources into two groups, where one group generates  pure data and the other generates noise. Then we can estimate the density of pure data and use it to detect anomalies caused by the sources of pure data rather than the changes caused by the sources of noise. Therefore, compared with other density based methods, our NRDE is much more robust to noise. Besides the new algorithm, we also provide theoretical results to support the effectiveness of NRDE. We compare NRDE with $15$ baselines on $47$ benchmark datasets under different settings, including vanilla anomaly detection, anomaly detection with anomaly contamination, anomaly detection on noisy data and transductive outlier detection. The results demonstrate effectiveness and superiority of NRDE.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
a normalizing flow method for anomaly detection that separates the signal part from the noise part.

### Strengths
The idea to separate noise from "pure data" in the modeling process seems as a good basis for research.

### Weaknesses
The assumption in Eq.6 drives this work. However, this is not validated nor justified. For me, it looks more like a matter of defining what is considered data and what is considered noise. Also, what determines the scale of the components in the vector s. They can be arbitrary scaled and still independent. On the other hand, if the scale is influenced by their influence on the data, then low variance implies little effect, so the difference they made can be negligible. 

The writing in general is very poor, making the paper hard to understand. This is true for all sections, starting from the abstract. I had to stop reading the methodology section early since it was becoming frustrating to keep up. For example, p_X is not defined, and then it is used in Eq.9. Also, the denominator of Eq.9 is clearly zero. 

I have little trust in the results of Figure 4. Although ADBench is a public benchmark, the authors run on it using their own code. Also, the uploaded code is partial and does not reproduce the results.

### Questions
see above

### Soundness
2

### Presentation
1

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
The paper proposes a noise-robust density estimator for tabular anomaly detection. A normalizing flow is trained with a Jacobian-based regularizer so that latent coordinates separate into signal and nuisance groups. After training, test points are scored by a weighted log density that emphasizes high-variance latent directions, which is meant to approximate the density of the noise-free component and reduce false alarms on normal but noisy data. The method is clearly motivated and the pipeline from the generative view to the practical scoring rule is coherent. Experiments on forty-seven ADBench datasets and several stress settings report strong average ranks against classical and modern baselines, with ablations that attribute gains to both the regularizer and the weighted density. Figures and tables make the intuition tangible. 

I like the idea and the careful framing, but some choices limit the strength of the claims. The approach assumes that signal sources have much larger variance than noise sources and that certain smoothness conditions hold. These are plausible but not verified on real datasets. Weight computation from average Jacobian norms can be sensitive to feature scaling and to architectural choices. Several baselines look under-tuned, and search spaces differ; some figures lack uncertainty intervals. The noisy data protocol perturbs normal points more strongly than anomalies, which may advantage methods that explicitly discount noise. The method is also costlier than a plain flow during training, and the work would benefit from an end-to-end latency profile. With additional diagnostics for the assumptions, more uniform tuning, and fuller statistics, the contribution would be quite solid in practice.

### Strengths
The paper addresses a real pain point of density-based anomaly detection on tabular data, namely false alarms caused by nuisance variation that is not semantically relevant for the task. The proposed Jacobian regularized flow with a weighted log density is conceptually simple, implementable with common flow libraries, and well-documented. The modeling story is coherent from the generative view through the practical scoring rule. The empirical section spans a broad range of datasets and settings, and the authors provide extensive tables and visualizations. I especially liked the clear diagrams, as well as the Jacobian row norm plots, which make the mechanism tangible. The ablations show that each component adds value, and the paper includes a time complexity discussion and implementation details that will help reproduction.

### Weaknesses
The core assumption that signal sources have markedly larger variance than noise sources is strong and domain-dependent, and the paper does not provide a quantitative diagnostic to verify it on real data before applying the method. The weighting relies on average Jacobian norms, which are known to be sensitive to scaling and to the architecture of the flow. The study hints at robustness, but a more systematic sensitivity analysis would be welcome. Several baselines appear under-tuned or constrained, and the tuning budgets differ, which may inflate gains. Statistical reporting is thin for some figures, and the noisy data protocol perturbs normal test points more strongly than anomalies, which may bias the comparison. The theoretical section assumes reasonable conditions for analysis, but these conditions are not verified for all datasets. Finally, practicality at scale is mixed. The method computes Jacobian norms and adds a regularizer during training, which is more costly than a plain flow. The paper compares testing complexity in Appendix B, but an end-to-end latency and memory profile for large datasets would provide a clearer picture of the practical implications.

### Questions
1) Could you provide a simple diagnostic that estimates the strength of variance separation between signal and noise sources on a given dataset, for example, a gap statistic over Jacobian row norms or a permutation-based calibration, and advise a user when NRDE is appropriate?

2) How sensitive are the results to feature scaling and to the choice of flow architecture, for instance, the number of coupling layers and the width?

3) In the noisy data experiment, normal test points receive stronger perturbation than anomalies. Would you consider a symmetric design and report the same table. 

4) Can you add confidence intervals for AUROC and AUPRC in the main figures and expand the transductive study to more datasets? 

5) Could you include a small empirical audit of the modeling assumptions, such as independence proxies or partial correlation summaries, and report how often the assumptions seem to hold across the forty-seven datasets?

### Soundness
3

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
4

### Summary
The authors introduce Noise-Robust Density Estimation (NRDE), an unsupervised anomaly-detection method for tabular data that estimates the density of clean samples while reducing the influence of noise. The method builds on a normalizing flow architecture (similar to RealNVP) and uses a Jacobian-based regularization that separates signal from noise in the latent space (in an ICA fashion). It is evaluated on 47 AD-BENCH datasets, showing that it outperforms multiple baselines. They also provide an ablation and theoretical analysis to support its ability to isolate the true data density. Finally, the method is demonstrated to be robust to noise.

### Strengths
The authors address an important well well-studied problem.

Overall, the paper is well written and mostly easy to follow. 

I like the idea presented in the paper, and 
The method demonstrates good results on a wide range of datasets. 

The method is more robust to noise compared with baselines.

### Weaknesses
The main weakness of the paper is that some technical details are obscure and written in a vague way in the appendix. For example, the discussion about \lambda and the learning rate does not provide a complete picture on how these hyperparameters are selected.

For instance:
“Fixing λ = 0, decrease learning rate from 0.01 to 0.001 until training becomes stable (i.e., no loss explosion); (ii) Then, based on (15), viewed as minW L(λ, W), select λ ∈ 1, 0.1, 0.01 such that the regularization term λR(·) is on a comparable scale with 0.1 · L(0, W).”

-If lambda is first initialized as 0, how much effect does it have? I see an ablation and sensitivity analysis, but only for 5  datasets.

-What are the final values of all these parameters? How exactly were they selected? The explanation above does not provide a procedure based on which I can reproduce this selection (“comparable scale with”,” decrease the learning rate”, how?)
It would be valuable to show the dynamics of the training loss and performance across the training procedure.

*I would be happy to raise my score if these issues could be clarified in the rebuttal.

### Questions
Why were these five datasets selected for all the extended evaluations (contamination, ablation…)
For the sensitivity analysis, suddenly one more dataset is removed, why?

The number of baselines, as stated in the paper, varies (13, 14, 15). You actually compare to 15.
In 4.4 why is the noise added to normal test samples stronger?

Citation format doesn’t fit the writing. Namely:
Anomaly detection Chandola et al. (2009); Pang et al. (2021); Ruff et al. (2021)-> should be
Anomaly detection (Chandola et al. (2009); Pang et al. (2021); Ruff et al. (2021))
Only if the name is part of the sentence, it should be without brackets.

Commas are missing after some equations, for example, 15.

Typos:
rangeing- ranging

### Soundness
3

### Presentation
3

### Contribution
3

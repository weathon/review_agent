# Latent g-Computation for Potential Outcomes Distributional Estimation under Time-Varying Treatments

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Estimating individualized potential outcomes (POs) under time-varying treatments is central to fields like medicine, marketing, or public policy. However, most methods yield only point estimates, offering limited guidance for risk-sensitive decisions. In this work, we propose G-Latent, a novel method to estimate full PO distributions via the g-formula. Unlike prior work such as G-Net  (Li et al., 2021), we learn per-step outcome distributions directly through variational autoencoders (VAEs) rather than relying on global residual pools. Our core contribution is the introduction of a latent-space rollout in which each time-step embedding is updated from latent representations instead of observed samples. At inference time, this enables Monte Carlo (MC) sampling without data-space autoregression, reducing error accumulation and increasing sampling speed. To enhance expressivity, we adapt a VAE parameterization based on infinite mixtures of asymmetric Laplace distributions from  An & Jeon (2023) to our time-series setting. We also decouple sequence processing: a transformer encodes the history up to a given time, while a lightweight GRU advances the forecast horizon, avoiding repeated transformer passes across steps and MC samples at inference. We validate our approach on synthetic, semi-synthetic, and real-world datasets, and provide theoretical guarantees.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper introduces G-Latent, an ML framework for time-varying causal inference. The main architectural novelty lies in the latent rollout technique used, which enables more computationally efficient and (potentially) accurate inference. Theoretical and empirical results were provided.

### Strengths
The topic of the paper addresses the timely topic of integrating deep machine learning models into time-varying causal inference. 

The paper is technically sound, and the modeling approach was analyzed in depth.

The experiment results are clean and fair. It shows that the proposed framework can make accurate counterfactual distribution estimation and enjoys good computational efficiency.

I appreciate the efforts of the authors in identifying potential limitations of their proposed approach.

### Weaknesses
Presentation: The paper is heavily loaded with technical details but lacks explanatory text. This decreases the readability of the paper. Theoretical results lack intuitive interpretations. Some notations (e.g., scope) appear without a formal definition.

Limited architecture novelty: the core novelty of the framework seems to lie in replacing the original data-space autoregression with latent GRU updates. This seems to be a variant of existing VAE-based / representation learning-based causal inference methods, rather than an architecture innovation.

Assumption restrictiveness: The paper's theoretical contribution is somewhat limited, as the analysis relies on the "latent factorization and context sufficiency" assumption, which postulates that the learned embeddings provide good approximations. This assumption is rather heuristic and was not validated via theoretical derivation or numerical experiments. Its informal presentation (without being stated as a numbered assumption) makes the exposition appear somewhat evasive.

Limited experiment scope: The two datasets used in the experiments are both ICU datasets. The method's application potential to other settings and other data types (e.g., images) remains unknown. Additionally, the code is not provided.

Jargon inconsistency: Some jargon used in the paper is not consistent. For example, "A" was referred to as treatment, decision, and plan.

### Questions
What is the definition of scope?

Why is the latent rollout idea novel? 

Why was the ICU dataset chosen in the experiments? How would the framework generalize to other types of datasets? Can the modeling assumption be empirically validated?

### Soundness
4

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a method for estimating distributions of potential outcomes over time based on variational autoencoders. For this purpose, the authors extend recently proposed methods based on G-computation (G-Net) and move the G-computation steps into the latent space. The method is validated on synthetic and real-world datasets.

### Strengths
- Uncertainty quantification for treatment effects is a relevant and important topic, particularly in the time series setting
- The proposed architecture is compelling and yields several practical advantages
- Experimental results look promising

### Weaknesses
- Novelty: The key ingredients (e.g., G-computation, VAEs) are known. The contribution is mainly a novel backbone/ model architecture tailored to distributional G-computation
- In the Appendix, the authors describe that they used the same shared and fixed hyperparameters for all transformer architectures (reasonable). However, for G-Latent, additional hyperparameters are tuned in a data-driven manner. I think this might give an unfair advantage to the G-Latent method as compared to the baselines
- The authors evaluate using a point error metric (RMSE) and two distributional metrics (energy and KL). However, the distributional metrics appear to be based on assumptions about the data (e.g., KL is compared to a Gaussian kernel density estimator). I think it may be valuable to use an additional assumption-agostic distributional distance (e.g., Wasserstein distance to the empirical data distribution).
- Missing baselines: There are existing baselines not compared with, that can learn the aleatoric uncertainty of potential outcomes over time: E.g., "Counterfactual Generative Models for Time-Varying Treatments" (Uses IPW instead of G-computation). Additionally, the authors claim that "Bayesian Neural Controlled Differential Equations for Treatment Effect Estimation" only models epistemic aleatoric uncertainty, which seems incorrect, as the paper models the posterior predictive distribution

### Questions
- Why are the results reported only for selected time steps t? How was this selection made?
- How does the method perform when the strength of time-varying confounding is varied?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces G-Latent, a neural method for estimating distributions of potential outcomes in discrete time. 
The authors leverage a transformer-VAE based architecture with a GRU decoder component to approximate the distributional version of G-computation. 
Further, the advantages of latent representations over standard autoregressive rollouts are discussed.
Finally, the method is benchmarked on a semi-synthetic dataset and a real world dataset.

### Strengths
- Complete piece of work: The paper provides (almost) all necessary notation, assumptions and proofs, and is therefore self-contained. 

- Sensible approach: The approach for G-computation chosen by the authors seems to be a sensible choice. Performing distributional G-computation with a VAE approach rather than with residual hold-outs as in G-Net is certainly a step forward regarding implementation.

- Important topic: Estimating potential outcomes over time has recently received more attention, and incorporating uncertainty quantification is an important part of it.

### Weaknesses
**Major:**

- **Very limited novelty:** Overall, the novelty of this work is very limited, which is my main concern: From a theoretical perspective, distributional G-computation with neural networks **has been established by G-Net** [1]. 
From an implementation perspective, while using transformers and VAEs is an improvement over LSTMs with residual hold-outs, the architecture presented is just a **simple combination of the multi-input transformer by [2] with the asymmetric VAE in [3]**.

- **Training objective:** The training objective looks a bit off to me. Specifically, I cannot make sense of the auxilliary one-step prediction loss. There is **no technical reason** why the history representation has to be (necessarily) predictive of a one-step ahead factual outcome.
This component is not grounded theoretically; it is further striking to me how a balancing parameter $\lambda$ should be tuned in practice.

- **Limited evaluation:** there is only a **single dataset** (semi-synthetic MIMIC III) on which the authors actually benchmark their method for potential outcome estimation, which is not think this is sufficient. Further, I do not like the reasons for which [4] and [5] were not chosen as baselines.

- **Point estimation:** Proposition 5.3 argues that computing latent representations is superior to autoregressive rollouts. 
This is an interesting insight, but the theorem only covers distributional estimation, and oversees that baselines that are designed for computing point estimates of the potential outcomes should be analyzed with different metrics other than the total variation.
Specifically, while computing distributions of latent representations may be superior to computing distributions of autoregressive rollouts, it may not be superior to computing point estimates of autoregressive rollouts when the goal is to estimate a point estimate of the potential outcome. Computing distributions is statistically more difficult than computing point estimates, and if the goal is to compute a point estimate of the potential outcome, there is no reason to use G-Latent (which is even validated by the experiments).
There is nothing wrong with the Proposition itself: instead, what is want to point out is that the lemma does not apply to atuoregressive point estimation methods such as CT and, therefore, the **insight is rather limited.**

**Minor:**

- Many of the cited works are already published by conferences and not preprints on arxiv any longer. I would advise the authors to double-check their references.

- The total variation distance should be introduced somewhere in the paper.

- I think the paper could be presented a little better. There is a lot of text that could, perhaps, be shortened and summarized in an additional figure. But this is, of course, a matter of personal taste.

- In Proposition 5.3, the assumption made in the implied Lemma from the appendix should be clearly stated.



____
[1] Rui Li, Stephanie Hu, Mingyu Lu, Yuria Utsumi, Prithwish Chakraborty, Daby M Sow, Piyush Madan, Jun Li, Mohamed Ghalwash, Zach Shahn, et al. G-net: a recurrent network approach to g-computation for counterfactual prediction under a dynamic treatment regime. In Machine Learning for Health, pp. 282–299. PMLR, 2021.

[2] Valentyn Melnychuk, Dennis Frauen, and Stefan Feuerriegel. Causal transformer for estimating counterfactual outcomes. In International conference on machine learning, pp. 15293–15329. PMLR, 2022.

[3] Seunghwan An and Jong-June Jeon. Distributional learning of variational autoencoder: Application to synthetic data generation. Advances in Neural Information Processing Systems, 36:57825–57851, 2023.

[4] Konstantin Hess, Valentyn Melnychuk, Dennis Frauen, and Stefan Feuerriegel. Bayesian neural controlled differential equations for treatment effect estimation. International Conference on Learning Representations, 2024.

[5] Wenhao Mu, Zhi Cao, Mehmed Uludag, and Alexander Rodr´ıguez. Counterfactual probabilistic diffusion with expert models. arXiv preprint arXiv:2508.13355, 2025.

### Questions
- Can the authors clearly point their technical novelty?

- What is the reason for the auxilliary one-step prediction loss?

- What is the authors' take on point estimation of potential outcomes with their method?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new method for potential outcomes prediction in a time-varying potential outcomes framework, namely, G-Latent. Specifically, the proposed G-Latent estimates the distribution of multi-step-ahead potential outcomes with a tailored time-varying variational auto-encoder (VAE). The model is based on (i) transformer-encoded representations, (ii) a light-weight GRU for multi-step-ahead prediction, and (iii) infinite mixtures VAE parametrization to enhance expressivity.  Importantly, the authors suggested using a roll-out in the latent space so that the time-consuming auto-regressive prediction can be omitted. Finally, the  G-Latent is evaluated on several (semi-)synthetic benchmarks.

### Strengths
The paper studies a relatively understudied area of the causal ML, time-varying generative modelling of potential outcomes. The proposed method is original.

### Weaknesses
I have several major concerns regarding the core idea of the method: Replacing the full g-computation with the roll-out in time in the latent space. 
- I agree that under certain conditions, we don’t need to auto-regressively feed the generated original covariates and can rather operate in the latent space. However, I don’t see how we can circumvent the exponential (in special cases polynomial) complexity of g-compution wrt. to the prediction horizon $\tau$ (i.e., it can be seen as an instance of the nested MC-estimator [1]). Therefore, in my opinion, the latent rollout (Algorithm 1) does not achieve the full **consistency** of the method: I would expect some sort of marginalization in the latent space $z_t$ over time steps $t, \dots, t + \tau -1$.  This would then have the same sampling speed wrt. $\tau$ as the original G-Net (as we would require $M^\tau$ samples to sample $M$ instances of $Y_{t+\tau}[a_{t}:a_{t+\tau-1}] for consistency [1]).  
- I carefully checked the theoretical results Sec 5.3 (regarding the issue of consistency): Even under the assumed factorization from Eq. 8, I don’t think we can simply skip the marginalization over time-steps $t, \dots, t + \tau -1$. Equivalently, it is not clear to me how “The inner integrals over $l_{t+t’}$ evaluate to 1 for $t′ = 1, \dots, \tau − 1$” (line 1016). This would be true for the conditional distributions of outcomes but not for the distributions of potential outcomes (otherwise, our estimator yields runtime confounding [2]).   

If the authors can clarify this crucial detail of their method (and all the other questions), I am open to raising my score. I understand that many existing methods in the related work (e.g., CT or G-Net) do not properly discuss the consistency of the multi-step-ahead prediction. However, the issue of consistency is crucial for the reliable application of the method.  

Also, I found some small mistakes:
- Lines 138-139. “epistemic … [? and aleatoric ?]“, “bayesian” (= wrong capitalization). 
- Lines 196-197. “(Variants such as βVAE scale the KL term by a factor β.)” This is not a proper sentence. 
- Line 258. “ In this work, we extend DistVAE, introduced”: Reference to DistVAE is missing.
- I found the notation (upper and lower indices) inconsistent in App. E. 2.
- I think Assumption A.3 is not correct (see the correct version here [3]).

References:
- [1] Rainforth, Tom, et al. "On nesting Monte Carlo estimators." International Conference on Machine Learning. PMLR, 2018.
- [2] Coston, Amanda, Edward Kennedy, and Alexandra Chouldechova. "Counterfactual predictions under runtime confounding." Advances in neural information processing systems 33 (2020): 4150-4162.
- [3] Frauen, Dennis, Konstantin Hess, and Stefan Feuerriegel. "Model-agnostic meta-learners for estimating heterogeneous treatment effects over time." arXiv preprint arXiv:2407.05287 (2024).

### Questions
- Are you sure a method from [1] cannot be adapted to the current setting? The experiments in [1] also include discrete-time benchmarks.
- Why are negative log-likelihoods for $y$ and $x$ different in the reconstruction loss (Eq. 6)?
- I wonder how the proposed method compares to the full auto-regressive version (where we reconstruct all the time-varying covariates). I encourage the authors to provide such an ablation.

References:
- [1] Mu, Wenhao, et al. "Counterfactual probabilistic diffusion with expert models." arXiv preprint arXiv:2508.13355 (2025).

### Soundness
1

### Presentation
2

### Contribution
2

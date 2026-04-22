# Scalable Bayesian Monte Carlo: fast uncertainty estimation beyond deep ensembles

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
This work introduces a new method designed for Bayesian deep learning called scalable Bayesian Monte Carlo (SBMC). The method is comprised of a model and an algorithm. The model interpolates between a point estimator and the posterior. The algorithm is a parallel implementation of sequential Monte Carlo sampler ($SMC_\parallel$) or Markov chain Monte Carlo ($MCMC_\parallel$). We collectively refer to these consistent (asymptotically unbiased) algorithms as Bayesian Monte Carlo (BMC), and any such algorithm can be used in our SBMC method. The utility of the method is demonstrated on practical examples: MNIST, CIFAR, IMDb. A systematic numerical study reveals that for a comparable wall-clock time to state-of-the-art (SOTA) methods like deep ensembles (DE), SBMC achieves comparable or better accuracy and substantially improved uncertainty quantification (UQ)--in particular, epistemic UQ. The benefit is demonstrated on the downstream task of estimating the confidence in predictions, which can be used for reliability assessment or abstention decisions. Code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes Scalable Bayesian Monte Carlo (SBMC) as a new framework aiming to combine point estimation with posterior sampling for uncertainty quantification in deep learning. SBMC conceptually interpolates between deterministic estimators and full Bayesian inference through parallelized SMC or MCMC procedures. The authors assert that this yields superior epistemic uncertainty estimates at a computational cost similar to deep ensembles. Nevertheless, much of the approach appears to repackage established concepts rather than introducing fundamentally new methodology. Further, the authors devote substantial effort to extensive benchmarking.

### Strengths
* **Simple concept of prior anchoring:** The core and simple modeling idea of interpolating between a point estimate and a posterior approximation is conceptually interesting. The fact that this framework is sampler-agnostic is a definite plus, allowing for flexibility in its implementation. I also liked the differentiation from the tempering-induced interpolation in the literature concerned with Cold Posteriors.
* **Extensive Empirical Evaluation:** A significant amount of work has clearly gone into the experimental validation. The large number of experiments/ablations in the appendix demonstrates a thorough empirical effort, which is commendable.

### Weaknesses
* **Concerns Regarding Novelty:** The paper's primary weakness is its limited novelty, particularly on the algorithmic side. As far as I understand it, the proposed algorithm is essentially a parallel MCMC or SMC sampler initialized near a MAP estimate and with slightly altered prior. This is not a new concept; many old (e.g. [1]) and also recent works, including some cited by the authors in Section 4 (e.g., SMS-UBU, Bayesian Deep Ensembles), have explored similar ideas. The paper fails to provide a clear differentiation from or comparison to these highly related methods, making the contribution seem incremental. Furthermore, the authors claim that SMS-UBU "did not perform as well" in their experiments without providing any empirical evidence to substantiate this claim, which is a serious omission. Large portions of the paper (e.g., the discussion of SMC and MCMC on page 4 & algorithms 2 and 3) reiterate well-known results/methods. 

* **Imprecise Framing:** The framing of the work contains several imprecise and potentially misleading statements. Referring to the $s=1$ case as the "full Bayesian posterior" is imprecise; any MCMC/SMC method with a finite compute budget can only ever produce an *approximation* of the true posterior. The introduction also makes claims about the method without formal definitions, making the initial exposition difficult to follow. The framing of Deep Ensembles as Bayesian is also problematic, as it's widely understood that DEs are not a consistent approximation of the Bayesian posterior (see [2]). 

* **Choice of benchmarks:** Figure 1 is unclear, difficult to interpret, and uses the term "mixing" in a non-standard way for a multimodal toy target, where a low ACF is not a reliable indicator of global mixing. This highlights the fact that it is unclear how the target (if it exists), to which SBMC converges, is characterized.  Furthermore, the paper uses out-of-distribution (OOD) detection benchmarks to validate its claims on epistemic UQ. However, recent work argues convincingly that these two objectives are fundamentally misaligned [3], making the OOD benchmark results a questionable proxy for the quality of epistemic uncertainty estimation. Overall, the experiments feel scattered and do not follow a clear or consistent line of reasoning.

* **Structure and Presentation:** The paper's structure and overall presentation need improvement. The writing is often unclear (many concepts are not properly defined/introduced early on), and crucial components like the "anchoring" step are not explained with sufficient clarity. The decision to include preliminary results on a new experiment in Section 5.1 is an odd choice that makes the work feel unfinished. Finally, the manuscript suffers from numerous formatting errors and incomplete references (e.g., missing venues for Papamarkou et al., 2024; Paulin et al., 2024, the latter should be 2025), which detracts from its overall quality and polish. 


## References 


[1] Vehtari, A., Sarkka, S., & Lampinen, J. (2000, July). On MCMC sampling in Bayesian MLP neural networks. In Proceedings of the IEEE-INNS-ENNS International Joint Conference on Neural Networks. IJCNN 2000. Neural Computing: New Challenges and Perspectives for the New Millennium (Vol. 1, pp. 317-322). IEEE.

[2] Wild, V. D., Ghalebikesabi, S., Sejdinovic, D., & Knoblauch, J. (2023). A rigorous link between deep ensembles and (variational) bayesian methods. Advances in Neural Information Processing Systems, 36, 39782-39811.

[3] Li, Y. L., Lu, D., Kirichenko, P., Qiu, S., Rudner, T. G., Bruss, C. B., & Wilson, A. G. (2025). Out-of-Distribution Detection Methods Answer the Wrong Questions. ICML.

### Questions
Please address my following questions. Satisfactory responses, in particular to (1-2), would in my opinion significantly strengthen the paper. 

**(1)** The primary benefit of the proposed method could simply stem from initializing sampling chains at or near the MAP solution, a strategy explored in prior work. To disentangle the benefit of your novel prior anchoring from this initialization effect, could you provide an ablation study? Did you investigate this question already? Specifically, the ablation I have in mind is to compare SBMC to a standard MCMC/SMC sampler that is simply initialized at the MAP but uses a standard, non-interpolated prior (i.e., fix $s=1$ and start at the MAP). 

**(2)** The paper mentions several state-of-the-art methods in Section 4 but omits them from direct comparison in Table 1 (e.g., SMS-UBU, Bayesian Deep Ensembles). Can you either include these highly relevant baselines or provide a convincing justification for their exclusion? Furthermore, could you please provide the empirical evidence to substantiate your claim that SMS-UBU "did not perform as well in practice"? 

**(3)** Could you provide practical guidance or heuristics on how to set the sampling duration for the MCMC/SMC component of SBMC? Additionally, the results in Figure 3 show that most sampling-based methods perform poorly and with extreme variance on the abstention recall metric. Do you have an explanation for this behavior? 

**(4)** You explore both (SG)MCMC and SMC samplers for the BMC component in your model. There seem to be considerable differences in performance on a per-task basis between the two approaches. In the conclusion, you state that both algorithms are “attractive options”. Could you elaborate on which approach you would generally suggest a practitioner to use?

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
The paper introduces a new Bayesian deep learning framework that bridges the gap between computationally expensive Bayesian Monte Carlo (BMC) samplers and fast but heuristic methods like deep ensembles. The core idea is an approximate model that interpolates between a maximum a posteriori (MAP) point estimator and the full Bayesian posterior via a scalar parameter $s$, allowing users to trade off between computational cost and uncertainty accuracy. Combined with parallel implementations of MCMC and SMC algorithms, SBMC delivers near-linear scalability and strong uncertainty quantification (UQ), particularly epistemic uncertainty, at a cost comparable to deep ensembles. Experiments on datasets demonstrate that SBMC achieves comparable accuracy but significantly better UQ and reliability, with applications such as confidence-based abstention.

### Strengths
1.	**Practical Motivation and Design:** The paper addresses a clear problem: scaling Bayesian inference to deep nets to obtain well-calibrated uncertainty. The SBMC model is intuitively motivated as interpolating between a cheap point estimate and the full posterior. This interpolation idea is simple and flexible: by tuning $s$ one can trade off bias vs. sampling difficulty. The algorithmic idea of running multiple short MCMC/SMC chains in parallel is straightforward and leverages modern parallel hardware.

2.	**Comprehensive Experiments:** The evaluation is extensive. The authors compare SBMC against several relevant baselines (MAP, SWA, MC Dropout, Laplace, Deep Ensembles) under a fixed compute budget (measured in epochs) on MNIST7, IMDb, and CIFAR10. They report accuracy, NLL, and epistemic entropy metrics. They also include “gold-standard” HMC runs (long chains) to estimate ground-truth uncertainty. The results consistently show that anchor+parallel BMC achieves accuracy comparable to strong baselines while yielding higher epistemic uncertainty on errors and out-of-domain inputs. 

3.	**Clarity of Presentation:** The paper clearly lays out the method, with Algorithm boxes for SBMC (and underlying SMC, MCMC routines). The interpolation model (Gaussian prior mean = $\theta_{MAP}$, covariance ∝ $s$) is explicitly defined.

### Weaknesses
1.	**Limited Novelty / Relation to Prior Methods:** The core idea (anchoring the posterior at a point estimate and running multiple short MCMC chains) is conceptually similar to known techniques. The paper cites Randomized Maximum Likelihood (RML) and ensemble anchoring methods (e.g. Gu & Oliver 2007; Bardsley et al. 2014). While SBMC’s scalar interpolation $s$ is a convenient formalism, the idea of tempering/anchoring (even mentioning “cold posteriors”) is not fundamentally new. Indeed, Paulin et al. (2024) previously anchored to SWA and ran multiple Langevin chains; this work’s difference (anchoring to MAP instead of SWA mean, using HMC/SMC) seems incremental. The contributions list claims a “new SBMC method” and “thorough evaluation”, but the method is essentially a heuristic interpolation rather than a novel algorithmic insight. 

2.	**Approximation Bias / Lack of Guarantees:** By construction, SBMC does not target the true Bayesian posterior except at $s=1$. For $0<s<1$ the estimator is biased. The authors acknowledge this (“since the method no longer targets the posterior for any $s<1$, we adopt a heuristic approach”), but they offer no theoretical analysis of the bias or uncertainty error introduced by $s<1$. The only justification is empirical: they show SBMC’s UQ looks sensible on their tasks. However, without any formal bounds or diagnostics, it is hard to trust SBMC’s uncertainty calibration in general. In fact, SBMC requires careful tuning of both $s$ and the prior variance $v$, as noted: small $s$ makes mixing easier but “introduces bias because $\pi_s \neq \pi$”. The paper’s advice to set $s=0.1$ as a default (with cross-validation for $v$) feels heuristic.


3.	**SWAG / More Baselines:** The authors reference SWAG (SWA-Gaussian) as a popular scalable Bayesian method, but they do not include SWAG results. Without comparing to SWAG or other modern UQ methods (e.g. KFAC-Laplace, SGD-based posteriors, deep Gaussian processes), it is unclear how SBMC stacks up.

4.	**Scale of Models:** The neural networks used (e.g. “architecture in Appendix E”) seem relatively small. There is no large-scale experiment (e.g. a full ResNet on CIFAR, or an NLP model beyond small GPT-2 finetuning). Claims of scalability are not validated on truly large networks where MCMC is most challenging. 

5.	**Compute Budget Accounting:** The comparisons assume “equal wall-clock time” by measuring epochs. However, SBMC needs to compute the MAP (another pass of SGD) plus run parallel chains. In Table 1 the “Total Cost” for SBMC is roughly double an SMC baseline (e.g. SBMC (1 chain) uses 330 epochs vs. MAP’s 160). The authors note SBMC’s total time is roughly double due to MAP computation (Sec. 3.1), but in claiming “same wall-clock as DE” this overhead must be justified. It seems SBMC is only comparable if parallel resources make the MAP step negligible. 

6.	**Evaluation Metrics:** The focus is on epistemic entropy as the UQ metric. While reasonable, this is somewhat non-standard. Calibration error (ECE/Brier) or negative log-likelihood on a held-out test set (beyond MNIST NLL) are not extensively reported.

### Questions
1.	**Comparison to SWAG and other baselines:** Why were methods like SWAG or deep Laplace (e.g. KFAC-Laplace) not included in the experiments? SWAG, in particular, is known to improve uncertainty on CIFAR with modest cost.

2.	**Hyperparameter Sensitivity:** You recommend $s=0.1$ by default and note the prior variance $v$ must be tuned. How sensitive are results to these choices across tasks? 

3.	**Scale and Efficiency:** Can SBMC handle larger-scale networks and datasets? For example, have you tried SBMC on a full CNN (e.g. ResNet) for CIFAR10, or on language models beyond the tiny GPT-2 setting? Also, how does SBMC’s actual GPU/runtime compare to DE in wall-clock seconds, given the extra MAP pass?

4.	**Multi-modal Posteriors:** In problems where the posterior is highly multi-modal, anchoring to a single MAP might be limiting. Have you considered initializing parallel chains at different modes (e.g. using multiple MAPs) to better explore multiple modes? 

5.	**Theoretical Guarantees:** Do you have any theoretical insight into how the bias $E_{\pi_s}[φ] - E_\pi[φ]$ depends on $s$? For example, can you bound the error in expectations as a function of $s$ or relate it to tempered posteriors? Without such guarantees, how should a practitioner choose $s$ to ensure reliability?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents Scalable Bayesian Monte Carlo (SBMC) method to balance computational efficiency with high-quality uncertainty quantification. The approach combines an interpolated model by combining a point estimate and a full Bayesian treatment with parallelized Monte Carlo algorithms, such as sequential Monte Carlo and Markov chain Monte Carlo. The authors showed that SBMC delivers competitive or better accuracy while significantly improving epistemic uncertainty compared with baselines such as deep ensembles across multiple benchmarks.

### Strengths
The paper is well-written, and the empirical results are presented clearly. Below are the strengths of the presented method:

1. SBMC introduces a model approximation which is anchored posterior that uses a scalar interpolation parameter to tune the trade-off between the fast Maximum A Posteriori estimator and the full Bayesian posterior.

2. The method is highly scalable due to the parallel implementation of consistent Bayesian Monte Carlo algorithms and achieved near linear speed-up.

3. Empirical studies show that SBMC achieves comparable or better accuracy and substantially improved epistemic uncertainty quantification versus Deep Ensembles at similar wall-clock time.

4. SBMC also showed good performance in estimating prediction confidence where a meta-classifier based on SBMC posterior improves detection of incorrect/out-of-domain data.

### Weaknesses
I think the paper can be improved by addressing the following weak points:

1. The paper relies on empirical tuning for the scalar interpolation parameter s and provides no theoretical analysis linking s to the difference anchored posterior and true posteriors.

2. SBMC requires computing a good Maximum A Posteriori to act as the anchor before parallel Bayesian monte Carlo sampling and it is seen in the MNIST7 experiment that the total cost is roughly doubled compared deep ensembles method due to this prerequisite step. 

3. The author used a discontinuous step for the parameter $\alpha(s)$ in equation 2 without providing good justification for these choices over continuous alternatives. 

4. A comprehensive hyperparameter table is missing to provide detailed implementation details and a graphical schematic for meta-classifier pipeline is missing.

### Questions
1. Why is $\alpha(s)$ defined as a step function with $\alpha(s)=1$ for $s<1/2$ and 0 otherwise? Would a smooth interpolation such as linear $\alpha(s)=1-s$ can produce different results, and how sensitive are the results to this choice?

2. For the meta-classifier, how many posterior samples were used to compute expectations and variances of $p_{max}(x, \theta)$ p_max (x,θ) and $\Delta_{max}(x, \theta)$? How is total entropy estimated for MAP and deep ensemble models that produce deterministic predictions?

3. Appendix 5.1 mentions preliminary results on GPT‑2; can you provide more details on applying SBMC to language models? What challenges may arise when scaling to tens of millions of parameters?

### Soundness
3

### Presentation
2

### Contribution
3

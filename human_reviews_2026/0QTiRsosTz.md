# Neural SDEs as a Unified Approach to Continuous-Domain Sequence Modeling

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Inspired by the ubiquitous use of differential equations to model continuous dynamics across diverse scientific and engineering domains, we propose a novel and intuitive approach to continuous sequence modeling. Our method interprets timeseries data as discrete samples from an underlying continuous dynamical system, and models its time evolution using Neural Stochastic Differential Equation (Neural SDE), where both the flow (drift) and diffusion terms are parameterized by neural networks. We derive a principled maximum likelihood objective and a simulationfree scheme for efficient training of our Neural SDE model. We demonstrate the versatility of our approach through experiments on sequence modeling tasks across both embodied and generative AI. Notably, to the best of our knowledge, this is the first work to show that SDEbased continuous-time modeling also excels in such complex scenarios, and we hope that our work opens up new avenues for research of SDE models in high-dimensional and temporally intricate domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel Neural Stochastic Differential Equation (Neural SDE) framework for continuous-domain sequence modeling. It treats time-series data as samples from an underlying continuous dynamical system and models its evolution using an SDE with neural networks for both the drift (deterministic trend) and diffusion (stochastic fluctuation) terms. Key contributions include a simulation-free maximum likelihood training objective and a decoupled two-stage optimizer, which eliminate the need for costly forward simulations during training.

### Strengths
1. The proposed Neural SDE framework introduces a simulation-free maximum likelihood approach, enabling more efficient modeling of continuous-time sequences.

2. Its effectiveness has been validated across multiple experiments, including imitation learning and video prediction tasks.

3. The manuscript is well-written, with clear and concise presentation, ensuring strong readability.

### Weaknesses
1. Generative models do not focus on intermediate processes; consequently, their efficacy in sequence modeling tasks remains somewhat limited.

2. To better verify the effectiveness of the proposed Neural SDE approach, it is necessary to design ablative experiments concerning its innovations.

### Questions
1. In Equation 7, $𝑋_t$ is constructed using stochastic interpolation (Equation 5). However, since stochastic interpolation lacks physical realism, how can the consistency between its gradients and the vector field be guaranteed?

2. The paper proposes two simplifying assumptions in Section 5. Could the authors comment on the framework's potential for future extensions regarding these two aspects?

3. When the Δt in the first term of Equation 11 varies, the effect of the loss function is expected to differ. For instance, when Δt is relatively large, Δx/Δt may not adequately approximate the gradient, potentially leading to a significant bias in the first term.

4. See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a maximum-likelihood (MLE)–based unified Neural SDE framework for sequence modeling. From the single-step transition NLL, it derives a two-stage decoupled training objective: a log-squared residual regression for the drift L_f (Eq. 14) and a “residual matching” loss for the diffusion L_g (Eq. 15), which avoids forward/backward SDE unrolling (Eqs. 11, 13, 14, 15; §3.2). The authors introduce a desingularization constant \delta and prove a temporal scale invariance theorem, enabling training on datasets without explicit timestamps (Appx. D, Thm. 1). Experiments span 2D bifurcation, Push-T imitation learning, and KTH/CLEVRER video prediction, with NFE–quality tradeoffs, a unified FVD protocol, and ablations (Fig. 2; Tab. 1–3).

### Strengths
* Decoupled optimization from the NLL: analytic diffusion optimum leads to a log-squared drift loss and diffusion residual matching; no SDE unrolling needed (Eqs. 11, 13–15).
* Scale-invariance and large-error robustness of the log-squared loss, plus desingularization \delta (Appx. D, Eq. 23).
* Temporal scale invariance theorem supports training with an arbitrary uniform \Delta t when timestamps are missing.
* Unified FVD protocol with repeated evaluation mitigates sample sensitivity.
* Strong sampling efficiency: about 17 steps achieve quality comparable to 100-step baselines (Fig. 2 and notes).
* Scaling behavior and “implicit interpolation” highlight advantages of continuous-time modeling (Fig. 3; Appx. F).

### Weaknesses
* On two video datasets, FVD is often worse than Flow Matching or PFI; results are competitive rather than clearly superior (Tab. 2).
* Push-T lacks variance and statistical tests, limiting evidential strength (Tab. 1).
* No systematic ablation or convergence curves comparing decoupled vs. joint training.
* Diagonal diffusion and Markov assumptions limit cross-dimensional noise correlations and long-range dependencies (§6).
* The denoiser weight \alpha is hand-tuned; the generalization cost is unclear (§6).

### Questions
1. How does decoupled versus joint training perform on video tasks? Please provide side-by-side training curves and final metrics under matched budgets.
2. Can \alpha be learned or made adaptive? Can it be linked to g(x) or state uncertainty to reduce tuning overhead (cf. Appx. I)?
3. For “free interpolation,” can you report quantitative metrics and a visualization set, compared to interpolation or retraining baselines?
4. Beyond NFE–performance, can you report end-to-end wall-clock latency and throughput, including different batch sizes?
5. For the 20 repeats in Tab. 2, do they span seeds and data resampling? Did you run significance tests (e.g., paired tests on FVD)?
6. For Push-T, can you add more metrics (e.g., success rate, path length minimization) and report confidence intervals over multiple runs?
7. Possible typos in Eqs. 11/13/14/15: should \Delta t_i be \Delta t instead?
8. Line 232: remove the trailing period.
9. The derivation mixes Hadamard products with g(x_t) treated as a diagonal matrix, which is inconsistent. For clarity, replace the Hadamard product with standard matrix multiplication.
10. In Eq. (15), consider removing the factor 1/2.

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
The paper presents a novel approach to train neural SDEs to model continuous sequential data. Unlike existing approaches for training neural SDEs which require simulating the SDE at training time, the authors propose a two-stage simulation-free optimization scheme derived from negative log-likelihood based on Euler discretization. This is done via cleverly exploiting the Markovian assumption in observations, and using a time-invariant SDE with diagonal diffusion.

### Strengths
- I find the proposed two-stage optimization framework for fitting neural SDEs quite elegant and intuitive. The idea of decoupling the training of the drift and diffusion terms can potentially solve the current optimization, stability, and diffusion term interpretability challenges with current approaches.

- The claimed empirical benefits of the approach (e.g., fewer NFEs, modeling multi-modal distributions, scale invariance) are well supported by experiments.

- The authors clearly list the limitations of the proposed approach.

### Weaknesses
The main challenge I have with this paper is that the motivation/the problem that it is trying to solve is not very clear to me.
First, the authors generally argue for adopting neural SDEs as a unified approach to sequence modeling connecting both dynamical systems in science and engineering described as differential equations as well as modern generative models. It is not quite clear to me how this is achieved with the proposed approach and conducted experiments. Prior approaches which argued for this [1,2] argue for viewing SDEs as differentiable mathematical objects that can incorporate neural networks. This enables developing models that are domain-informed or comparing white, grey, and black box models. The proposed approach and experiments focus on black box neural SDEs, and tasks that are not based on science/engineering applications.

Second, the authors argue that the proposed approach enables simulation-free training for generative modeling of sequence data. They argue that existing approaches either require expensive SDE simulation during training or modern bridge-based approaches applied on sequential data can be expensive due to uninformative priors or don't respect true temporal progression. While these statements are fundamentally true, there are several methods over the last 2 years that aim to address these problems and show superior performance on modeling sequential data[3,4,5,6], while already incorporating partial observability, external covariates, and long-range dependencies. These approaches are never discussed in the paper.

References

[1] Rackauckas, Christopher, et al. "Universal differential equations for scientific machine learning." arXiv preprint arXiv:2001.04385 (2020).

[2] ElGazzar, A., & van Gerven, M. (2024). Universal differential equations as a common modeling language for neuroscience. arXiv preprint arXiv:2403.14510.

[3] Bartosh, G., Vetrov, D., & Naesseth, C. A. (2025). SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations. arXiv preprint arXiv:2502.02472.

[4] Zhang, X. N., Pu, Y., Kawamura, Y., Loza, A., Bengio, Y., Shung, D., & Tong, A. (2024). Trajectory flow matching with applications to clinical time series modelling. Advances in Neural Information Processing Systems, 37, 107198-107224.

[5] El-Gazzar, A., & van Gerven, M. (2025). Probabilistic Forecasting via Autoregressive Flow Matching. arXiv preprint arXiv:2503.10375.

[6] Kollovieh, M., Lienen, M., Lüdke, D., Schwinn, L., & Günnemann, S. (2024). Flow matching with gaussian process priors for probabilistic time series forecasting. arXiv preprint arXiv:2410.03024.

### Questions
- Can the authors elaborate further on how their approach unifies sequence modeling? 
- Can the dervied optimization framework work for non-neural sdes or hybrids?
- Following the unfied approach argument, can the author comment on how the proposed optimization framework can be extended for partial obsevations?
- I found the use of the term flow coefceint to desribe the drift function rather confusing and dont belive its standard in  the literature. Can the authors elaboarte on that.

### Soundness
2

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
The paper proposes a Neural SDE framework for sequence modeling that treats data as samples from an underlying continuous-time dynamical system and trains without simulating the SDE. Using Euler–Maruyama, it models one-step Gaussian transitions to form a maximum-likelihood (NLL) objective, then decouples learning into two stages: fit the drift via a log-squared residual loss and estimate the diffusion via residual matching. Under time-invariant dynamics with diagonal diffusion for efficiency, the method enables fast, parallelizable sampling and low inference cost. Evaluated on trajectory prediction, imitation learning, and video prediction, it delivers performance comparable to diffusion/flow-matching approaches while exhibiting favorable (power-law) scaling behavior.

### Strengths
- The paper is well written and easy to follow. The authors also provide an anonymized code base.
- The proposed method appears simple and efficient. In particular, the idea of decoupled two-stage optimization simplifies the approach.
- The authors did a good job on the practical part of the work, presenting diverse experiments on different benchmarks and providing both qualitative and quantitative analyses.

### Weaknesses
The main issue with this work, in my opinion, is the lack of novelty. I believe that training an SDE by treating consecutive samples from an Euler–Maruyama (EM) discretization and performing (quasi-)maximum likelihood estimation is a long-established idea. For instance, a very similar approach was described as early as 1995 [1].

Moreover, the authors do not discuss more recent literature such as Trajectory Flow Matching (TFM) [2], which proposes a very similar simulation-free training method for Neural SDEs (they even have a similar separate optimisation of the diffusion coefficient), or the more general SDE Matching [3] and ARCTA [4], which enable simulation-free training of non-Markovian processes.

The second major issue is the limited applicability. As the authors admit, their method only works with Markov processes. However, they also rely on EM discretization, which means it only allows training with relatively dense observations. This limitation is particularly unfortunate, since it seems that it could be relatively easily mitigated by conditioning on the last K observations and constructing interpolations between consecutive observations (as done in TFM).

Minor:
- Figure 1 is never cited in the text.

[1] Pedersen, Asger Roer. "Consistency and asymptotic normality of an approximate maximum likelihood estimator for discretely observed diffusion processes." Bernoulli (1995): 257–279.

[2] Zhang, Xi Nicole, et al. "Trajectory flow matching with applications to clinical time series modelling." Advances in Neural Information Processing Systems 37 (2024): 107198–107224.

[3] Bartosh et al. "SDE Matching: Scalable and Simulation-Free Training of Latent Stochastic Differential Equations." The 43rd International Conference on Machine Learning (2025).

[4] Course, K. and Nair, P. "Amortized reparametrization: efficient and scalable variational inference for latent SDEs." Advances in Neural Information Processing Systems, 36 (2023).

### Questions
- Please address the Weaknesses section above.
- Could you comment on Equation 14? The text describes it as a mathematically correct approach. However, it seems to be a mathematically motivated heuristic, since the optimal diffusion coefficient depends not only on the current state but also on Δ. Therefore, we cannot simply substitute an optimal diffusion coefficient into Equation 11 and derive Equation 14. If my understanding is correct, I believe this point should be discussed more accurately in the text.
- Could you provide more details about the setup for DDIM and Rectified Flow in your 2D branching trajectories experiment? I got the impression that you trained these models using samples from your branching trajectories as noisy samples. However, these models were not designed to operate in this regime. Since your goal is to compare your time-series modeling approach with other time-series modeling methods, it would be more appropriate to train DDIM and Rectified Flow to generate the entire sequence as a single high-dimensional object. While this would be computationally more expensive than your approach, I would not expect it to have any issues with sample quality.

### Soundness
3

### Presentation
3

### Contribution
1

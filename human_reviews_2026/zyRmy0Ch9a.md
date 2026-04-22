# Diffusion Bridge Variational Inference for Deep Gaussian Processes

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 8

## Abstract
Deep Gaussian processes (DGPs) enable expressive hierarchical Bayesian modeling but pose substantial challenges for posterior inference, especially over inducing variables. Denoising diffusion variational inference (DDVI) addresses this by modeling the posterior as a time-reversed diffusion from a simple Gaussian prior. However, DDVI’s fixed unconditional starting distribution remains far from the complex true posterior, resulting in inefficient inference trajectories and slow convergence. In this work, we propose Diffusion Bridge Variational Inference (DBVI), a principled extension of DDVI that initiates the reverse diffusion from a learnable, data-dependent initial distribution. This initialization is parameterized via an amortized neural network and progressively adapted using gradients from the ELBO objective, reducing the posterior gap and improving sample efficiency. To enable scalable amortization, we design the network to operate on the inducing inputs $\mathbf{Z}^{(l)}$, which serve as structured, low-dimensional summaries of the dataset and naturally align with the inducing variables' shape. DBVI retains the mathematical elegance of DDVI—including Girsanov-based ELBOs and reverse-time SDEs—while reinterpreting the prior via a Doob-bridged diffusion process. We derive a tractable training objective under this formulation and implement DBVI for scalable inference in large-scale DGPs. Across regression, classification, and image reconstruction tasks, DBVI consistently outperforms DDVI and other variational baselines in predictive accuracy, convergence speed, and posterior quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Diffusion Bridge Variational Inference (DBVI), a new approach for performing inference in Deep Gaussian Processes (DGPs). DBVI builds on Denoising Diffusion Variational Inference (DDVI) but improves it by learning how to start the reverse diffusion process from a more informed, data-dependent initialization instead of a random prior. This helps the model start closer to the true posterior and improves both accuracy and efficiency. The authors interpret this modification through Doob’s h-transform, giving a bridge-based view of the diffusion process that keeps the method theoretically consistent while making it more flexible. They also describe a practical inference scheme based on inducing points to make training scalable.
In experiments on regression, classification, and unsupervised learning tasks, DBVI shows consistent improvements over DDVI and other standard inference methods for DGPs such as DSVI, IPVI, and SGHMC.

### Strengths
Originality:
The paper proposes the novel idea of reinterpreting DDVI as a kind of diffusion bridge using Doob’s h-transform. This leads to a principled way of conditioning the diffusion process on input data. The use of an input-dependent initialization for diffusion-based inference is conceptually elegant and gives a new perspective on how diffusion models can be adapted for Bayesian inference.


Quality:
The technical development is solid and well thought out. The authors clearly connect their bridge formulation to the underlying variational objective. The experiments are thorough, covering regression, classification, and unsupervised learning, and DBVI consistently outperforms DDVI and other inference methods like DSVI, IPVI, and SGHMC. 


Clarity:
Overall, the paper is well written and easy to follow. The authors do a good job of explaining the motivation behind their changes to DDVI. The main ideas are presented in a logical order, and the appendix includes helpful details for implementation. 


Significance:
This work makes a meaningful contribution to improving inference in deep Gaussian process models. By addressing a key limitation of DDVI and showing consistent gains across several tasks, the paper offers a practical improvement that should be useful to researchers.

### Weaknesses
Primary Weakness: 

Magnitude of improvements: While DBVI consistently outperforms DDVI and other methods, the numerical improvements are sometimes modest (e.g., some overlapping error bars in Figure 3). A discussion of whether these gains translate to meaningful practical differences would strengthen the empirical section.


Minor Weaknesses: 

Figure readability: The font size in Figures 1, 2, and 3 is quite small, making some labels difficult to read without zooming in a lot. This is a minor visual issue that can easily be fixed for the camera-ready version.


Formatting issue: The arrow in Figure 2 partially obscures the word “likelihood.”

### Questions
1: Could the authors comment on the computational cost of the additional amortization network used to initialize the diffusion bridge?
In particular, does this learned initialization introduce noticeable overhead compared to the standard DDVI setup?


2: In Figure 3, the improvements over DDVI are consistent but often modest. Could the authors comment on whether larger benefits might appear in other settings and why? 




3: Do the authors plan to release code and trained models for reproducibility and to facilitate adoption by the community?

### Soundness
3

### Presentation
3

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
This paper proposes Diffusion Bridge Variational Inference (DBVI), a novel method for posterior inference in Deep Gaussian Processes (DGPs). It aims to solve a key limitation of its predecessor, Denoising Diffusion Variational Inference (DDVI), namely the inefficiency of starting the reverse diffusion from a fixed, unconditional Gaussian prior (a "cold start").The core idea of DBVI is to replace this fixed start with a learnable, data-dependent "warm start" distribution, $p_0^\theta(U_0|x) = \mathcal{N}(U_0; \mu_\theta(x), \sigma^2 I)$. The paper introduces two key technical innovations to make this work: It formally re-interprets the diffusion process as a Doob-bridged diffusion, which is grounded in Doob's h-transform. This allows for the derivation of a new, tractable ELBO objective.To make the initial distribution $p_0^\theta$ scalable and avoid conditioning on the full dataset, it proposes a structured amortization network $\mu_\theta$ that cleverly conditions on the layer-wise inducing inputs $Z^{(l)}$ as proxies for the data.

### Strengths
The paper's core strength is its theoretical rigor. It correctly identifies a clear weakness in DDVI (the "cold start") and proposes a non-trivial, principled solution. Grounding the "warm start" in the mathematics of Doob's h-transform allows for a clean derivation of the ELBO and proves that DBVI is a strict generalization of DDVI.

The most innovative practical contribution is the structured amortization design. Naively conditioning $\mu_\theta$ on the raw data $x$ would be infeasible. The proposal to use the learnable inducing inputs $Z^{(l)}$ as a data-dependent, low-dimensional proxy is an elegant and effective solution that neatly sidesteps issues of dimensionality and data dependency.

The paper's central hypothesis a "warm start" improves inference efficiency, is directly and convincingly validated by the case study in Figure 4. This plot clearly shows DBVI converging significantly faster and to a better final RMSE than DDVI, confirming the mechanism works as intended.

### Weaknesses
The experimental evaluation is incomplete for a paper claiming state-of-the-art posterior approximation. A major, competing line of work for expressive DGP posteriors, Normalizing Flows (NFs), is entirely absent from the related work and experimental comparisons. Without benchmarking against NF-based VI methods, the claims of superior posterior quality and accuracy are unsubstantiated.

Missing Practical Baseline: The paper fails to establish a practical "sanity-check" baseline. For the image classification tasks, the DGP is applied to features from a ResNet-20. The performance of this feature extractor backbone alone must be reported. If the final, highly complex 4-layer DBVI model (which achieves 95.68% accuracy on CIFAR-10) does not substantially outperform the ResNet-20, it implies the entire DGP/DBVI machinery adds significant complexity for little to no practical gain.

Unfavorable Complexity-Performance Trade-off. This is the most significant weakness. The paper advocates for a method that is substantially more complex than its predecessor. It requires an SDE solver for $s_\phi$, a new NN for $\mu_\theta$, and an ODE solver for $(m_t, \kappa_t)$. The justification for this complexity rests on predictive gains that are empirically marginal (e.g., 95.68% vs 95.56% on CIFAR-10; 0.859 vs 0.857 AUC on HIGGS). This trade-off makes the practical utility of DBVI highly questionable.

While Table 1 provides per-iteration timings, the paper lacks a formal analysis of the additional computational overhead. It should provide a breakdown of the cost of the $\mu_\theta$ forward pass and the $(m_t, \kappa_t)$ ODE solver, and discuss how these new costs scale with the number of inducing points ($M$) and layers ($L$).

### Questions
Baselines: Could the authors provide results for two critical missing baselines?a. A competing SOTA expressive VI method, such as a Normalizing Flow-based DGP.b. The "backbone-only" baseline for the CIFAR-10 experiment (i.e., the accuracy of the ResNet-20 feature extractor).

Given that the strongest empirical result is the convergence speedup (Fig. 4), while the final accuracy gains are marginal, would the authors agree that the primary contribution of DBVI is in inference efficiency rather than predictive accuracy? If so, I would recommend reframing the paper's narrative to emphasize this.

 Could the authors provide a more detailed breakdown of the additional computational cost of DBVI over DDVI? Specifically, what is the wall-clock cost and scaling behavior of the $\mu_\theta$ forward pass and the $(m_t, \kappa_t)$ ODE solver?

The $Z^{(l)}$-amortization is a clever feedback loop. How sensitive is this mechanism to the initialization of the inducing inputs $Z^{(l)}$? Does a poor $Z^{(l)}$ initialization lead to a poor $U_0$ start, which in turn hinders the model's ability to find good $Z^{(l)}$ locations?

### Soundness
3

### Presentation
3

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
This paper generalizes denoising diffusion variational inference (DDVI) for deep Gaussian processes (DGP) by replacing the unconditional starting distribution with a learnable, data-dependent initial distribution and reinterpreting the DDVI framework with incorporation of Doob's h-transformation as a diffusion bridge model (DBVI). The proposed method can reduce the gap between the prior and the posterior in the diffusion process and hence speed up the training. There are a few benchmarks included to demonstrate the improvements in accuracy, convergence speeds and posterior quality.

### Strengths
It elegantly extends DDVI with Doob-bridge modification. The theory behind such extension is sound and neat. The derived loss clearly connects to that of DDVI and it is straightforward to spot the innovation.

### Weaknesses
However, the numerical experiments could be improved. The benchmark results mainly focus on estimation/prediction accuracy, not much on the posterior analysis. See more comments in the questions below.

I would rate 5 if allowed and will raise my score if the numerical evidence could be strengthened.

### Questions
1. There is only Figure 4 illustrating arguably after convergence on the small dataset. It appears DDVI reduces the error faster initially before 50 iterations and catches up around 200 iterations. How about after 200 iterations? Can you include more of such plots to demonstrate "faster convergence"?

2. Claiming better posterior quality, the authors only show reconstruction RMSE and test log-likelihood, which may not well evaluate quality of the learned posterior. How about empirical KL if you can test in some simulation? Or the rate of credible intervals covering true values?

3. Is the computational cost proportional to the layers of deep GP? How does the diffusion time step K interact with the number of layers L interact?

4. Minor suggestion, are $d_{in}$ and $d_{out}$ layer specific? Should then carry layer index $l$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Diffusion Bridge Variational Inference (DBVI), an extension of the recently introduced Denoising Diffusion Variational Inference (DDVI) method for Deep Gaussian Processes. While DDVI models the variational posterior as the terminal distribution of a reverse-time diffusion SDE parameterized by a neural score network, it suffers from an unconditional Gaussian initialization that is typically far from the true posterior, resulting in long, inefficient inference trajectories. DBVI addresses this limitation by making the diffusion start data-dependent...using an amortized initialization. Using Doob’s h-transform, they reinterpret the reverse SDE as a bridge process that explicitly “bends” the diffusion between a start and an end distribution. Empirical results across regression, classification, and image-reconstruction benchmarks show that DBVI consistently improves predictive accuracy, posterior quality, and convergence speed over DDVI, and other benchmarks.

### Strengths
- The paper identifies the unconditional start distribution in DDVI as the source of slow convergence and inaccurate posteriors, and replaces it with a data-conditioned start via a Doob bridge.
- The use of a linear reference drift ensures that even after introducing endpoint conditioning, the bridge process has closed-form Gaussian marginals at all intermediate times.
- Like DDVI, the variational posterior is defined implicitly by a reverse diffusion, which is significantly more flexible than mean-field or low-rank Gaussian approximations typically used in DGP inference.
- Improvements are observed not only on small UCI datasets but also on large datasets (like HIGGS/SUSY), image classification (more subtle), and the frey faces task.

### Weaknesses
- The paper explains the Doob correction, but the experimental section gives limited direct visualization of how the bridge shortens the reverse diffusion trajectory (e.g path length, KL rate, or score norm decay). This makes it harder to see the effect that drives the performance gains.
- The method (in my understanding) depends crucially on using a linear reference diffusion so that intermediate marginals remain Gaussian and h is tractable. If the model or dataset induces posteriors that are far from those implied by linear reference dynamics, the reference score may become a poor baseline and learning may slow or plateau.

### Questions
- The model fixes the covariance and learns only the mean. How sensitive is DBVI to this choice? Do performance or convergence degrade meaningfully if the variance is misspecified, and can a learned covariance or low-rank structure further improve inference? 
- The Doob bridge construction relies on the fact that the forward reference diffusion has closed form Gaussian marginals at all t, does learning the covariance disrupt this?
- More generally, what are the limitations of tying amortization exclusively to the inducing input grid?
- Could there be a diagnostic metric which can quantify how much the data conditioning shortens the reverse path?

### Soundness
3

### Presentation
3

### Contribution
3

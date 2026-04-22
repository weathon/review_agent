# DiffBED: Scaling Bayesian Experimental Design to High-Dimensions

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 4, 4

## Abstract
Bayesian experimental design (BED) is a principled framework for intelligent data acquisition. However, current approaches do not scale to problems with high–dimensional designs, impeding its uptake. We show that this limitation arises predominantly from the difficulty in specifying a likelihood model that remains accurate throughout the design space, and that without this, standard design optimisation procedures lead to a reward-hacking-like behaviour that exploits deficiencies in the likelihood, producing implausible or unrealistic designs.
To overcome this, we introduce DiffBED, an approach based on a novel BED objective that explicitly rewards realistic designs. Realism is captured by a diffusion model, which we guide using information-theoretic experimental design criteria to generate highly informative yet realistic designs. This enables BED at an unprecedented scale: while existing applications of BED have been restricted to design spaces with a handful of dimensions, we show that DiffBED can successfully scale to designing high–resolution images.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DiffBED, a Bayesian experimental design method for high-dimensional design spaces that uses a pretrained diffusion model as a prior over feasible designs. The approach guides the reverse-diffusion process with gradients of expected information gain (EIG), producing proposals that stay on the data manifold (realistic) while remaining highly informative for learning the latent target. The authors argue that naïvely maximizing EIG in high dimensions can “reward-hack” likelihood misspecification; DiffBED mitigates this by sampling from an EIG-tilted prior rather than optimizing EIG directly. On image-level tasks—including 512×512 shoe designs—DiffBED outperforms standard BED and several strong baselines.

### Strengths
1. The paper clearly shows how naive EIG in high-dimensional settings can suffer from likelihood misspecification, which makes staying on the data manifold meaningful.

2. Exponential tilting of a diffusion prior is a clean way to sample realistic, informative designs—no extra training required.

3. This method uses a pretrained diffusion model, so there’s no retraining at every step which is time consuming for sequential BED.

4. This method works on large images (512×512) and outperforms standard BED.

5. This method improves the similarty to $\theta_{true}$ , which indicates true target recovery.

### Weaknesses
1. The paper doesn’t compare against a simple constrained EIG optimizer that stays on the data manifold (e.g., trust-region steps or directly maximizing EIG with a prior penalty). Without that apples-to-apples baseline, it’s hard to tell whether the gains come mainly from enforcing feasibility or from sampling the EIG-tilted prior itself.

2. Exponential tilting can collapse to a single mode on multimodal landscapes, and picking a good $\alpha$ is non-trivial because it trades diversity against feasibility.

3. The method leans on a strong pretrained diffusion prior. If an appropriate prior doesn’t exist or is domain-mismatched, it may be inapplicable, and tilting can’t add mass where the prior has zero mass.

4. Results focus on discrete outcomes; if y is continuous, EIG estimation and gradients are harder.

### Questions
1. Did you try a direct EIG optimizer that stays on the data manifold (e.g., trust-region steps or EIG + prior penalty)? What’s your intuition for how it would perform vs. DiffBED?

2. How sensitive are performance and diversity to $\alpha$?

3. On multi-peak landscapes for $p_{ref}(\theta)$, do you see mode collapse when $\alpha$ is small?

4. What happens when the prior is biased or domain-mismatched—does it still produce meaningful designs?

5. For continuous y, how would you estimate EIG, and which gradient estimator would you use (just a brief idea is fine)?

*** Duplicate word *** 

line 222, by solving time-reversal of Equation (9)

### Soundness
3

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
2

### Summary
This paper addresses the problem of scaling Bayesian Experimental Design (BED) to high-dimensional design spaces by first identifying a critical issue: directly optimizing Expected Information Gain (EIG) in high dimensions exploits model misspecification, producing unrealistic designs (e.g., noise images) where the likelihood is overconfident. The authors propose DiffBED, which constrains design optimization using a diffusion model as a reference prior over feasible designs, sampling from a distribution proportional to $p_{ref}(\epsilon) \cdot \exp(EIG(\epsilon)/\alpha). Designs are generated via information-guided diffusion, combining the diffusion model's score function with EIG gradients in the reverse SDE. The authors conducted experiments on image-based tasks, and DiffBED generates realistic and informative designs in high-dimension space, while standard gradient-based BED fails by producing noise despite high model EIG.

### Strengths
1. The problem studied in this paper is well-motivated, and the authors clearly justified that EIG optimization inherently seeks regions where the likelihood is overconfident, identifying model misspecification as the fundamental barrier to scaling BED.

2. The reward hacking analogy well motivates the approach, and by incorporating pre-trained diffusion models as reference priors, the proposed method prevents reward hacking while maintaining high informativeness.

3. Experimental results demonstrate that DiffBED is able to scale on high-dimensional settings.

### Weaknesses
1. The proposed method is highly dependent on having a high-quality, pre-trained diffusion model for the design space. This does not fully solve the high-dimensional problem as shift it from needing a perfect likelihood to needing a perfect generative prior. It would be great if the authors could elaborate more on this part.

2. The proposed method is computationally expensive as it requires running a full, guided reverse-diffusion process for each experimental iteration.

3. I think the authors should position the method as a hybrid, not a purely Bayesian one. It combines principled Bayesian inference (for $\theta$) with a pragmatic, regularized optimization (for $\xi$). The design objective $p^{*}(\xi)$ introduces $p^{ref}(\xi)$ as an external regularizer not derived from the original generative model.

### Questions
1. Please see the comments in Weakness part.

2. Could the authors provide more details on the BED baseline used in the experiments? I would suspect that the main computational bottleneck is the EIG gradients calculation, so how much does the proposed method cost more than the BED baseline?

3. It seems that the Rank baseline is highly competitive in the conducted experiments, can the authors discuss in more detail?

4. The proposed method depends on high-quality pre-trained diffusion models and encoders, which are feasible for vision but limited in other domains. Could the authors discuss the feasibility of applying DiffBED beyond vision (e.g., molecules, audio, text, tabular data), what happens when good diffusion models might not exist, and discuss when DiffBED is preferable over simpler pool-based approaches like Rank?

### Soundness
2

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
3

### Summary
This paper presents an approach to apply diffusion models to propose informative and plausible designs for Bayesian adaptive experimental design in high-dimensional design spaces. The paper shows that, in such settings, likelihood model misspecification can severely misguide BED algorithms towards designs which have an apparent high expected information gain (EIG), though due to errors in the likelihood model, leading traditional methods to produce no longer meaningful designs in high dimensions. To mitigate that, the proposed method employs pre-trained diffusion models whose samples are guided by an EIG-based gradient estimator towards designs which are informative for the given task without the need to retrain the diffusion model. Experiments are presented on high-dimensional experimental design problems involving images.

### Strengths
* The paper is mostly well written and follows a clear structure which is relatively easy to follow.
* The proposed method seems relatively simple to implement within modern generative modelling frameworks.
* Some of the principles and insights in this paper may be applicable beyond its setting, such as the presentation on the issues with misspecification in BED, how a prior over feasible designs can help mitigate these issues, and how to apply pre-trained diffusion models to design tasks without a need for retraining.

### Weaknesses
* In Sec. 3, it is not clear how the decomposition of the EIG from Eq. 5 to 6 was derived. Eq. 5 was derived using the alternative formulation of the TEIG in terms of $\theta$, instead of $y$, which was the one presented in Eq. 4, making things confusing. Eq. 6 seems to have been derived by adding and subtracting TEIG from the model's EIG, but why the step in Eq. 5 was necessary is unclear.
* It is not explained how Eq. 11 was derived. I believe that might require some background in diffusion models, though such a derivation should at least be found somewhere in the appendix, given that the EIG guidance term is a crucial contribution from this paper.
* Experiments are presented on image design tasks, which escape traditional problems in experimental design, making it difficult to assess the impact of the paper outside this context. Despite the complexity of optimisation and sampling in high-dimensional design spaces, the probabilistic models in these experiments seem reasonably simpler than traditional models found in science and engineering applications where BED typically finds its applications. Hence, I'm unsure of this paper's impact.
* The presented performance plots show that, besides the gradient-descent BED baseline, the proposed DiffBED's performance is mostly very close to the performance of simpler baselines. Yet, I reckon that std. deviations, which I interpret as the shaded areas around the curves, are quite small as well. Therefore, the practical significance of this paper's contribution remains unclear.
* I'm also unsure about the claim in Sec. 5 that "the first to identify that model misalignment is not constant across design space and show the potential for reward hacking". Prior work in BED has discussed issues related to model misspecification and their effect on design optimisation, such as Foster et al. (2025), though to a different extent. In addition, the issue of model mismatch as a function of the designs is explicitly modelled in, e.g., Bayesian calibration frameworks by the model discrepancy/error term (Kennedy & O'Hagan, 2001), which have been recently applied to BED (Oliveira et al., 2024; Sürer et al., 2024). Additional references are listed below.

Minor:
* A few typos are present throughout the text, which require careful revision.
* Eq. 10 is missing a time differential $\mathrm{d}t$ next to the term in brackets.
* I believe it should be $\xi_t$, instead of $x_t$, in the inline equation on line 244.

References:
* Kennedy, M. C., & O’Hagan, A. (2001). Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 63(3), 425–464.
* Oliveira, R., Sejdinovic, D., Howard, D., & Bonilla, E. V. (2024). Bayesian Adaptive Calibration and Optimal Design. 38th Conference on Neural Information Processing Systems (NeurIPS 2024).
* Sürer, Ö., Plumlee, M., & Wild, S. M. (2024). Sequential Bayesian experimental design for calibration of expensive simulation models. Technometrics, 66(2), 157-171.

### Questions
Please, see weakness points above. In addition, I have the following more specific questions.

* Have other high-dimensional experiment settings been considered, beyond images? As far as I understand, diffusion models could also be applied in lower-dimensional settings and compared against other BED baselines in more traditional problems. Some synthetic problems can have their dimensionality adjusted in ablation problems to show how the performance of standard methods potentially degrade as the dimensionality increases, while hopefully DiffBED maintains reasonable performance levels.

* Did each experimental trial have a different target image in Sec. 6.1? Or was the same image used across each experiment, with the different random seed only affecting optimisation behaviour?

### Soundness
3

### Presentation
3

### Contribution
2

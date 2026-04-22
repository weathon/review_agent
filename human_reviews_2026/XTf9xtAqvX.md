# Reframing Generative Models for Physical Systems using Stochastic Interpolants

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Generative models have recently emerged as powerful surrogates for physical systems, demonstrating increased accuracy, stability, and/or statistical fidelity. However, most approaches rely on iteratively denoising a Gaussian, a choice that may not be the most effective for autoregressive prediction tasks in PDEs and dynamical systems such as climate. Given the proximity of current and future distributions in these tasks, we consider generative models based on stochastic interpolants, which can directly learn maps between arbitrary densities. To evaluate this framework, we benchmark a variety of generative models across fluid and climate systems. Our experiments suggest that stochastic interpolants can use fewer sampling steps and produce more accurate predictions than models relying on transporting Gaussian noise. We also observe broad trends across generative models, such as the ability to exchange deterministic accuracy, spectral consistency, and probabilistic calibration through sampling. Given these results, this study establishes stochastic interpolants as a competitive baseline for physical emulation and gives insight into the abilities of different generative modeling frameworks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper benchmarks generative surrogates for physical systems and argues for stochastic interpolants(SI) that learn a process directly between current and next states, enabling fewer sampling steps and improved accuracy/calibration for autoregressive forecasting. The authors conduct a comprehensive benchmark of SI against other leading generative models (DDPM, DDIM, EDM, Flow Matching) and deterministic baselines (FNO/SFNO). The evaluation is performed on three diverse and challenging physical datasets: Kolmogorov Flow (PDE), Rayleigh-Bénard Convection (chaotic PDE), and PlaSim (global climate model). SI is positioned as a competitive baseline for physical emulation, with tunable ODE/SDE sampling to balance pointwise accuracy v.s. spectral fidelity.

### Strengths
- Originality. This paper performs autoregressive physical emulation using stochastic interpolants (SI) that evolve samples directly from current to future states, unifying ODE/SDE sampling and avoiding Gaussian-noise conditioning when source/target are close. This is well-motivated for PDE/weather.
- Quality. This paper uses matched backbones/latents across methods, reports NFEs, and evaluates on three representative datasets with pointwise, spectral, and probabilistic metrics; includes ablations (pixel vs latent, compression ratio, deterministic vs probabilistic).
- Clarity. This paper provides a concrete SI definition with boundary conditions and a clear training loss, plus explicit ODE/SDE sampling equations. The metrics are defined in the paper/appendix for reproduce evaluation.

### Weaknesses
## Contradiction and misinterpretation in the paper's motivational evidence
Theoretically, after the initial transition phase/ turbulent mixing (around t=20 in the Rayleigh-Bernard convection case), the system is expected to enter a statistically stationary regime. In this state, while the instantaneous $u(t)$ is chaotic, the overall distribution $p(u(t))$ should remain basically unchanged.
- The paper's own SWD and MMD scores in Figure 2 are in direct contradiction with physical principles and C2ST score. Base on SWD and MMD curves, the paper states, *"subsequent states seem to be farther during and after this transition".* This suggests the distributions are "far" apart, which is unphysical and should not have happened. 
- At the statistically stationary stage, the C2ST score stays at ~50%. The paper sees the 50% score and provides this explanation: *"Interestingly, classifiers struggle to take advantage of this; subsequent timesteps have ~50% classification accuracy in turbulent regimes, as consistent changes between states become harder to learn"*.  As discussed, the observation that the classifier "struggle to take advantage of" is unphysical and should not have happened. Also, A 50% score does not mean that changes between turbulence states are hard to learn. It means the classifier has failed because the two distributions have become indistinguishable. This is not a failure of the metric, it is a strong evidence for the paper's hypothesis. It shows that in the statistically steady turbulent regime, the distribution at time t is so similar to the distribution at t+1 that a powerful classifier cannot find any signal to tell them apart. 

**The authors  fail to discuss this glaring contradiction, and also misinterpret the one metric (C2ST) that actually supports their physical intuition.**

## Insufficient contribution beyond hyperparameter tuning and the selective application of known techniques.
The paper's core contribution reduces to: "We applied stochastic interpolants (an existing method) to physical emulation (where similar generative models already exist) and found they work competitively with appropriate hyperparameter tuning."

-  **What is already known** The complete mathematical framework, training objectives, sampling procedures, ability to adjust during sampling without retraining, data-dependent couplings, are already fully established. (e.g. Albergo & Vanden-Eijnden 2023, Albergo et al. 2023, Chen et al. 2024, Albergo et al. 2024). The trade-offs between deterministic/stochastic sampling and the success of interpolants/bridges in related domains are already known. The observation that transporting related distributions is easier than transporting from Gaussian noise is the on of the foundational motivations of the the entire bridge/interpolant literature (e.g. Zhou et al. 2023, Su et al. 2023, Albergo et al. 2024, Zheng et al. 2025, Bortoli et al. 2023, Tong et al. 2024). This is not a new insight. The applications of SI to physics (especially, forecasting and fluids) are not new. 

- **What the paper contributes** Empirical results showing SI is competitive (sometimes better, often comparable, but **without error bar**) on three physical systems; Ablations on hyperparameters; Distance heuristics visualizations which demonstrate a known issue.

- **The  "Win" for SI is attributable to tuning** The paper only demonstrates that a hyper-tuned SI can beat default-tuned baselines.
    - The paper's own ablation study (Table 9) is the key piece of evidence. For example, it shows that SI's performance on Kolmogorov flow is critically dependent on the noise coefficient $\sigma$. At $\sigma=3$, its NRMSE is the worst of any generative model in Table 1. At $\sigma=1$, its NRMSE is the best score reported in Table 1. While a valid empirical study would apply this same level of tuning to all competitors. The paper does not. It explicitly states it uses the default, non-optimized schedules for its baselines, such as "the noise schedule for EDM is reproduced from Karras et al. (2022)" and "For FM models, we use the rectified flow schedule". **This unequal tuning invalidates the paper as a fair empirical study**. 

- **The paper's "Solutions" are selective applications of known tricks**
    - The paper's solution for the accuracy-vs-statics trade-off is to adjust the sampling (i.e. switch between an ODE and SDE sampler). But this flexibility is a known, fundamental property of all continuous-time generative models, including the EDM baseline. A true empirical study would have applied this "trick" to all capable models.

To conclude, the paper suffers from a positioning crisis, as it is not a new method, not a novel application, and fails as a fair benchmark/empirical study. 

## Statistical robustness. 
The paper's conclusions are not statistically robust. Kolmorogov flow experiment is under-sampled. The results are reported over 3 seeds, which is insufficient to robustly capture the variance inherent in generative models. For Rayleigh-Benard convection and PlaSim, only one run is performed. Main tables except table 1 lack error, probabilistic metrics are averaged but no bands are shown. Therefore, the conclusions about method ranking are not statistically supported. This could be a material limitation for more chaotic/turbulent tasks.

### Questions
Some minor concerns:
- Scope narrower than the abstract's claim of "diverse physical domains." The work benchmarks three close related fluid/climate settings - Kolmogorov flow, Rayleigh-Benard convection, and the challenging PlaSim climate emulation. That's diverse within fluids, but not across broader physical domains. For the simplest dataset, Kolmogorov Flow, the authors report error and standard deviation over only three seeds, and then reduce that to a single run for their more complex, chaotic datasets. For Kolmogorov flow, this is insufficient to capture variances 
- Spectral RMSE definition is inconsistent between main tex and appendix. The definition in Appendix D.2 misses the square.
- The paper cites Lancelin et al. (in prep) and Wikner et al. (in prep). There are two issues. The first one is, these works are not publicly available and hence unverifiable. The second one is, **how would the authors of this submission be among the few aware of these unavailable "in prep" works and their title/contents?**

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper describes its contribution as reframing the generative modeling of physical systems through the modeling approach of Stochastic interpolants.  Specifically the authors focus on the task of conditionally modeling the next time step solution u(t+1) of systems described by PDEs, given the current timestep u(t). The authors compare various generative modeling methods, including recent variants of stochastic interpolants, for a number of PDEs describing Kolmogorov flows, RB convection, and climate/weather. The authors emphasize the utility of SDE generative models whose initial condition starts at the current solution of the PDE, rather than from a Gaussian.

### Strengths
Thorough benchmarking of a wide array of generative modeling methods with a lot of reproducibility details mentioned, on several challenging PDE simulation problems.

### Weaknesses
I believe the paper exhibits great work on benchmarking existing generative modeling methods on a number of challenging tasks. However, I believe the contributions of the work could use some reframing. After reading, I’m not left with the impression that I have seen a reframing, or a new method proposed. In particular, while section 2 is a review of stochastic interpolants, section 3 “Methods”  mostly describes datasets, baselines, autoencoders, architectural choices, and metrics. 

What I don’t see is much of a discussion of “reframing” in the paper in the form of something like
- it’s a long standing trend that people do XYZ, and we show evidence here that we should instead look at the problem as XYZ’ 
- or solve it in a new way with method XYZ’ that we propose 
- or solve it with method XYZ’ that has been overlooked as a solution. 

At a few points during the paper, an approach is emphasized: conditionally generating the u(t+1) from an interpolant with initial condition u(t), but I see that this has been explored in a few papers including the work by Chen et al (https://arxiv.org/abs/2403.13724) that the authors cite. I would also understand more if there was a framing of this work was that the observations in Chen et al (and related papers) are shown in this paper to extend to new types of data.

From the reviewing perspective, I would really benefit from one of the following:

- clarification on the nature of reframing or contributions that I may have missed
- an attempt to narrate the paper more closely with what it accomplishes, which is a super solid benchmarking of generative modeling methods on hard PDE tasks. But if so, please relate this to either works that have also done this (for example, https://arxiv.org/abs/2410.16415).
- or perhaps a clarification that the work being done is showing that existing methods extend to new instances of data.

If the authors feel that these two options are a mischaracterization, or inappropriately miss another aspects of the work, I’m open to hearing about it.

Regardless, I also proceed with a few more specific technical comments in "Questions".

In summary, I feel rather positively about the work done, but I feel that it is not what the paper describes as the work being done. I am open to discussion about it with the authors and other reviewers.

### Questions
**Question #1**

I have doubts that the SDE you present in (4):

$$
dX = b dt + \frac{\dot \gamma(t)}{\sqrt{t}}dW_t
$$

has the same time-varying marginals as the stochastic interpolant and the ODE. 

I'll proceed by direct comparison to that in (Albergo et al) and in (Chen et al). Let us assume the setup with $\alpha, \beta, \gamma$ from (https://arxiv.org/abs/2303.08797). Let $b()$ denote the ODE velocity/drift that you (implicitly) define through the loss function in (3) as 

$$
b(t, x) = E[\dot \alpha_t x_0 + \dot \beta_t x_1 + \dot \gamma_t z | x_t = x]
$$

Then it is true that the ODE you present, $dX = b dt$, starting at a sample of $x_0 \sim \rho_0$, passes through the same time marginals $\rho_t$ that the interpolant does.  However, according to (Albergo et al), the SDE with solutions that pass through these same marginals is, for some $\delta(t) > 0$, 

$$
dX = f(t, x) dt + \sqrt{2 \delta(t)} dW_t, \quad f(t, x) = b(t, x) + \delta(t) s(t, x)
$$ 

Here $s(t, x) = \nabla \log \rho_t(x)$. The score can be shown to be $s(t,x)=-\frac{1}{\gamma(t)}E[z | x_t]$ so altogether:

$$
f(t, x) = E[\dot \alpha_t x_0 + \dot \beta_t x_1 + (\dot \gamma_t - \frac{\delta(t)}{\gamma(t)} )z | x_t = x]
$$

It doesn't immediately seem obvious to me that this generic $f()$ recipe equals the $b()$ mentioned in your (4) for the SDE.

Next, in (Chen et al 2024), which you mention just after your equation (4), they do make some special choices that yield an SDE that appears different to this recipe, but, can be shown to be an instance of that in (Albergo et al). Let us next review this choice., assuming their SDE has been proven to pass through the right marginals. (Chen et al) describe the interpolant directly via the Brownian motion rather than the gaussian $z$. They define it as 

$$
x_t = \alpha(t) x_0 + \beta(t) x_1 + \sigma(t) W_t
$$

where $W_t$ is the time $t$ solution of a brownian motion/Wiener process which has the same distribution as $\sqrt{t}z$ for standard gaussian $z$. Therefore this interpolant has the same distribution as 
$$
x_t = \alpha(t) x_0 + \beta(t) x_1 + \sigma(t) \sqrt{t} z
$$
for gaussian $z$, so we can see that it corresponds to the choice of your $\gamma(t) = \sigma(t) \sqrt{t}$ for some chosen $\sigma(t)$. We also note that this same $\sigma(t)$, which appears in the interpolant, also appears in (Chen et al)'s generative SDE, directly as the diffusion coefficient.

Next, In your section 3.2 (which is somewhat after your equation 4 where the SDE is given a very particular form), you set $\alpha(t)=1-t, \beta(t)=t$, and $\gamma(t)=(1-t)\sqrt{t}$. By your choice of $\gamma$ and by the previous reasoning step that $\gamma$ relates to $\sigma$ via $\gamma(t) = \sigma(t) \sqrt{t}$, we can conclude that the $\sigma(t)$ that is implicitly in play in your work is $\sigma(t)=1-t$, like in (Chen et al). However, (Chen et al) show that the appropriate SDE has diffusion coefficient exactly equal to $\sigma(t)$ i.e., $dX = b dt + \sigma(t) dW_t$. But the diffusion coefficient you choose in (4) is $\frac{\dot \gamma(t)}{\sqrt{t}}$. So we must next compare these choices. Taking $\gamma = (1-t)\sqrt{t}$ (as you choose in section 3) and computing your $\frac{\dot \gamma(t)}{\sqrt{t}}$ seems to give a diffusion coeffient of $\frac{1-3t}{2t}$, while (Chen et al) just had the diffusion coefficient $\sigma(t) = 1-t$.

Therefore, with the background of Albergo et al (https://arxiv.org/abs/2303.08797) and Chen et al (https://arxiv.org/abs/2403.13724), I don’t quite have the tools to see why the SDE in your (4) has solutions following the marginals of the interpolant and the ODE (thereby yielding the integration of this SDE as a method that samples from the generative model being trained)

Regardless of whether my reasoning about the mismatch holds up (sorry if any gaps, please let me know), could you please provide some sort of explanation (preferably proof) of how the SDE samples the desired distribution? If the specific form only holds under choices of $\alpha,\beta,\gamma$ (and maybe an implicit $\delta(t)$) used later on in the methods/experiments, could you please make sure to state these choices right next to (4) and clarify whether it only holds under those choses?

**Question #2 (relatively minor stuff)**

Some more confusion on ODE and SDE for stochastic interpolant: 

- table 11 (appendix) reportedly describes the sampling procedures for each model, and only shows the Euler discretization of the ODE generative model defined by the interpolant. However, in the main text, the ODE (sampled with Euler) and SDE (sampled with Euler-Maruyama) are shown in the results tables. Could you please update table 11 to include the stochastic interpolant as well, or explain why it’s not listed. 

- for the ODE: if the previous time step u(t) of the PDE solution does not determine the next step u(t+1) (either due to stochastic forcing or unobserved variables), then there is a distribution of u(t+1) given u(t). This is seemingly your motivation for using generative models. However, the ODE does need stochasticity in the usual random gaussian initial condition in order to model a whole distribution. How can the ODE starting at u(t) model a distribution?


**small comment** 

Since velocity-trained stochastic interpolant with no gamma function and gaussian base distribution is equal to flow matching, especially when both are sampled with the ODE, could you help the reader by the reminding them that the difference is the $\gamma$, the gaussian base, and the choice of sampling algorithm, and especially provide clarity when flow matching and the interpolant Euler ODE show up in the same result tables?

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
The paper proposes a framework for the physics prediction. The problem is defined as learning p(x_t|x_t+1) in a physical system, where x represents the state. The whole process is vae+diffusion/flow models. The key innovation is the diffusion process. Instead of using guassian noise, the model choose x_t as the start.

### Strengths
I think the design is reasonable and promising. The choice of vae, DiT, flow/diffusion, works well on the physical prediction.

### Weaknesses
I am confused about ODE setting. Assuming for a single x_t, there are two possible x_{t+1}^a and x_{t+1}^b, where does the uncertainty come from? With the same start, the model can only learn the average ( x_{t+1}^a  +  x_{t+1}^b), and the model becomes more likely a purely prediction model with rmse loss: learn the average of the output. During the sampling, the start is fixed, combined with an ODE process, this is impossible to generate different results. Though the result may be good, it is possible the whole process is not a probabilistic model.

### Questions
See weakness. I think this question is important to this submission. And I think the biggest challenge for the single start with ODE process is how to get different samples.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper benchmarks generative models for deterministic and probabilistic forecasting of physical systems (fluids and weather datasets), contrasting methods that transform from Gaussian noise with those that transform from current states to future states. The authors focus on stochastic interpolants for the latter case and diffusion models/flow matching for the former. They argue that the proximity of successive physical distributions makes stochastic interpolants better suited for forecasting problems, achieving superior accuracy across various metrics.

### Strengths
The paper investigates the choice of source distributions (Gaussian noise versus current states) in generative modeling for physical systems, a natural question with practical relevance across many problems. The authors conducted comprehensive benchmark experiments using various evaluation metrics and made useful observations.

### Weaknesses
1. The set-up of stochastic interpolants needs more clarification. 
First, on page 2, equation 4, the ODE and SDE formulations appear to be inconsistent. The SDE formulation looks incorrect: the drift b already generates the exact marginal distribution at time t=1, so directly adding a stochastic term seems not right. A drift modification should be made to compensate for this additional noise.

In Section 4, it is unclear from the results how SI-E and SI-EM, which correspond to the ODE and SDE formulations respectively, are implemented. The above concern regarding the SDE formulation raises questions about the reliability of these results.

2. For the stochastic interpolant methods, it appears the authors did not feed the current state vector as an additional input to the drift vector b (as seen in Figure 1). 

If so, the framework will only guarantee generation of the marginal distribution of u(t+1), rather than the conditional distribution u(t+1)|u(t). When the time lag is small, there is strong correlation between successive states, so the generative model in this marginal setting may still capture some conditional dependence / coupling, particularly for deterministic dynamical systems, so the authors could obtain reasonable accuracy. But in general, this approach appears less mathematically sound than the conditional formulation and requires further clarification.

### Questions
1. The diffusion models and flow matching methods use Gaussian source distributions. To generate the conditional distribution u(t+1)|u(t), the authors need to feed the current state u(t) as an additional input to the drift vector. This appears to be what the authors have done based on Figure 1 (Page 2), but it contradicts what is presented in Table 11 (Page 23), where the setup generates the marginal distribution rather than the conditional distribution. The authors provide some explanation on Page 4 under "architecture," specifically noting that "the current state of the system is concatenated to the noisy estimate of the future state as conditioning." I think making this distinction clearer throughout the paper would be helpful.

2. The authors have considered metrics for both deterministic accuracy (i.e., using these models to predict deterministic outputs, such as pointwise values or spectral values—specifically for the Kolmogorov flow and Rayleigh-Benard convection examples in this paper) and probabilistic accuracy (for the climate forecasting example). The latter is natural since these generative models produce inherently probabilistic outputs. For the former, could the authors clarify how the stochasticity in the generative model's output is handled to obtain deterministic predictions? Is it simply averaged to get a point estimate? This is not clear in the current presentation.

3. For the climate emulation example, on page 7 the authors state: "Therefore, while 10-day forecasting errors converge and are reported after 30 epochs, models are fine-tuned for an additional 20 epochs for bias evaluations." Could the authors elaborate on how the fine-tuning and bias reduction are performed? Given the long rollout period required for climatology, short-term prediction methods may reasonably suffer from instability or inaccuracy. Specialized techniques are often needed to address this, and further clarification from the authors would be helpful.

### Soundness
3

### Presentation
2

### Contribution
3

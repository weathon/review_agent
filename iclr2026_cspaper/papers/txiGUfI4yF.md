# Latent Stochastic Interpolants

Saurabh Singh 
∗
Poetiq AI saurabh@poetiq.ai Dmitry Lagun Google DeepMind dlagun@google.com

## Abstract

Stochastic Interpolants (SI) is a powerful framework for generative modeling, capable of flexibly transforming between two probability distributions. However, its use in jointly optimized latent variable models remains unexplored as it requires direct access to the samples from the two distributions. This work presents Latent Stochastic Interpolants (LSI) enabling joint learning in a latent space with endto-end optimized encoder, decoder and latent SI models. We achieve this by developing a principled Evidence Lower Bound (ELBO) objective derived directly in continuous time. The joint optimization allows LSI to learn effective latent representations along with a generative process that transforms an arbitrary prior distribution into the encoder-defined aggregated posterior. LSI sidesteps the simple priors of the normal diffusion models and mitigates the computational demands of applying SI directly in high-dimensional observation spaces, while preserving the generative flexibility of the SI framework. We demonstrate the efficacy of LSI through comprehensive experiments on the standard large scale ImageNet generation benchmark.

## 1 Introduction

Diffusion models have achieved remarkable success in modeling complex, high-dimensional data distributions across various domains. These models learn to transform a simple "prior" distribution p0, such as a standard Gaussian, into a complex data distribution p1. While early formulations were constrained to use specific prior distributions that are Lévy Stable, recent advancements, particularly Stochastic Interpolants (SI) (Albergo et al., 2023) offer a powerful, unifying framework capable of bridging arbitrary probability distributions. However, SI assumes that both the prior p0 and the target p1 distributions are fixed and the samples from both are directly *observed*. This requirement limits their use in jointly learned latent variable models where the generative model is learned, along with an encoder and a decoder, in a latent unobserved space. Further, the latent space, often lower dimensional, evolves as the encoder and decoder are jointly optimized. Lack of support for joint optimization implies that arbitrary fixed latent representations may not be optimally aligned with the generative process resulting in inefficiencies. To address this, we present Latent Stochastic Interpolants (LSI), a novel framework for end-to-end learning of a generative model in an *unobserved* latent space. Our key innovation lies in deriving a principled, flexible and scalable training objective as an Evidence Lower Bound (ELBO) directly in continuous time. This objective, like SI, provides data log-likelihood control, while enabling scalable end-to-end training of the three components: an encoder mapping high-dimensional observations to a latent space, a decoder reconstructing observations from latent representations, and a latent SI model operating entirely within the learned latent space. Our approach allows transforming arbitrary prior distributions into the encoder-defined aggregated posterior, simultaneously aligning data representations with a high-fidelity generative process using that representation. LSI's single ELBO objective provides a unified, scalable framework that avoids the need for simple priors of the normal diffusion models, mitigates the computational demands of applying SI directly in high-dimensional observation spaces and offers an alternative to ad-hoc multi-stage training. Our formulation admits simulation-free training analogous to observation-space diffusion and SI models, while preserving the flexibility of SI framework. We empirically validate LSI's strengths through
∗Work done while at DeepMind.

1 comprehensive experiments on the challenging ImageNet generation benchmark, demonstrating competitive generative performance and highlighting its advantages in efficiency. Our key contributions are: 1) **Latent stochastic interpolants (LSI):** a novel and flexible framework for scalable training of a latent variable generative model with continuous time dynamic latent variables, where the encoder, decoder and latent generative model are jointly trained, 2) **Unifying** perspective: a novel perspective on integrating flexible continuous-time formulation of SI within latent variable models, leveraging insights from continuous time stochastic processes, 3) **Principled** ELBO objective: a new ELBO as a principled training objective that retains strengths of SI - simple simulation free training and flexible prior choice - while enabling the benefits of joint training in a latent space.

## 2 Background

Notation. We use small letters *x, y, t* etc. to represent scalar and vector variables, *f, g* etc. to represent functions, Greek letters *β, θ* etc. to represent (hyper-)parameters. Lower case letters x are used to represent both the random variable and a particular value x ∼ p(x). Dependence on an argument t is indicated as a subscript ut or argument u(t) interchangeably.

Our work builds upon two key results briefly reviewed below. The first result (Li et al., 2020; Theodorou, 2015) states an Evidence Lower Bound (ELBO) for models using continuous time dynamic latent variables. We state a more general form than the original to aid the discussion of the prior distributions. The second result is a well known method for constructing a stochastic mapping between two distributions. We exploit it to construct a variational approximation in the latent space.

## 2.1 Variational Lower Bound Using Dynamic Latent Variables

Consider two SDEs, starting with the same starting point z˜0 = z0 at t = 0, sharing the same dispersion coefficient σ(zt, t) but potentially different initial distributions - z0 ∼ p0(z0) for the model, with path measure Pθ, and z0 ∼ q0(z0) for the variational posterior, with path measure Q:

$$\begin{array}{l l}{{d\bar{z}_{t}=h_{\theta}(\bar{z}_{t},t)d t+\sigma(\bar{z}_{t},t)d\bar{w}_{t},}}&{{\quad\quad\quad(\mathrm{model,~path~measure~}\mathbb{P}_{\theta})}}\\ {{d z_{t}=h_{\phi}(z_{t},t)d t+\sigma(z_{t},t)d w_{t},}}&{{\quad\quad\quad(\mathrm{variational~posterior,~path~measure~}\mathbb{Q})}}\end{array}$$

Where w˜t and wt are Wiener processes under corresponding path measures. The first equation can be viewed as the latent dynamics under the model hθ we are interested in learning and the second as the latent dynamics under some variational approximation to the posterior that can be used to produce samples zt. Further, let xti be observations at time tithat are assumed to only depend on the corresponding unobserved latent state zti, then the ELBO can be written as

$$\ln p_{\theta}(x_{t_{1}},\ldots,x_{t_{n}})\geq\mathbb{E}_{\mathbb{Q}}\left[\sum_{i=1}^{n}\ln p_{\theta}(x_{t_{i}}|z_{t_{i}})-\ln\frac{q_{0}(z_{0})}{p_{0}(z_{0})}-\frac{1}{2}\int_{0}^{T}\|u(z_{t},t)\|^{2}\,dt\right]$$ $$=\mathbb{E}_{\mathbb{Q}}\left[\sum_{i=1}^{n}\ln p_{\theta}(x_{t_{i}}|z_{t_{i}})\right]-\text{KL}(\mathbb{Q}\|\mathbb{P}_{\theta})$$  with $\mathbb{E}_{\mathbb{Q}}$.  
$$\begin{array}{l}{(1)}\\ {(2)}\end{array}$$
(3)  $\frac{1}{2}$  (4)  ... 
2dt#(3)

Where u satisfies

$$\sigma(z,t)u(z,t)=h_{\phi}(z,t)-h_{\theta}(z,t)$$
σ(*z, t*)u(z, t) = hϕ(z, t) − hθ(*z, t*) (5)
We provide the proof of the above general form in Section A. Similar to the ELBO for the VAEs (Kingma et al., 2013), the first term in eq. (4) explains observations given the latent path and the second term penalizes the mismatch between the variational and model path distributions. In the following, we focus on the case of q0 = p0 and draw attention to the general case when needed.

## 2.2 Diffusion Bridge

Given two arbitrary points z0 and z1, a diffusion bridge between the two is a random process constrained to start and end at the two given end points. A diffusion bridge can be used to specify the stochastic dynamics of a particle that starts at z0 at t = 0 and is constrained to land at z1 at t = 1.

$$({\boldsymbol{5}})$$

Consider a stochastic process starting at z0 with the dynamics specified by eq. (2). Using Doob's h-transform, the SDE for the end point conditioned diffusion bridge, constrained to end at z1 at time t = 1 can be written as dzt = [hϕ(zt, t) + σ(zt, t)σ(zt, t)
T ∇zt ln p(z1|zt)]dt + σ(zt, t)dwt (6)
where p(z1|zt) is the conditional density for z1 under the original dynamics in eq. (2) and depends on hϕ. Note that a Brownian bridge is a special case of a Diffusion bridge where the dynamics are specified by the standard Brownian motion. Diffusion bridges can be used to construct a stochastic mapping between two distributions by considering the end points z0 ∼ p0(z0) and z1 ∼ p1(z1) to be sampled from the two distributions of interest.

## 3 Latent Stochastic Interpolants

Stochastic Interpolants (SI) and their limitation: SI (Albergo et al., 2023) is a powerful framework for generative modeling, capable of learning a model that can flexibly transform between two probability distributions. Let x1 ∼ p(x1) be an observation from the data distribution p(x1) that we want to sample from. In SI framework, another distribution p0(x0) is chosen as a prior with samples x0 ∼ p0(x0). Typically, p0 is easy to sample from, e.g. a Gaussian distribution. A stochastic interpolant xt is then constructed with the requirement that the marginal distribution pt(xt) of xt equals p0 at t = 0 and p1 at t = 1. For example, the interpolant xt = (1−t)x0+tx1+pt(1 − t)ϵ, ϵ ∼ N(0, I)
satisfies this requirement. The velocity field and the score function for the generative model are then estimated as solutions to particular least squares problems. The learned velocity field and the score function can then be used to transform a sample from p0 to produce a sample from p1. SI requires that the samples x0 and x1 are observed, though x1 could be an output of a *fixed* model, hence still observed. We use the term observation space SI to emphasize this. However, we are interested in jointly learning a generative model in a latent space to leverage efficiency of low dimensional representations while also aligning the latents with the generative process. Therefore, we want to jointly optimize an encoder pθ(z1|x1) that represents high dimensional observations in the latent space and a decoder pθ(x1|z1) that maps a given latent representation to the observation space, along with the generative model in latent space. To use SI, we need to interpolate between a fixed prior p0(z0) in the latent space and the true marginal posterior p1(z1) ≡Rp(z1|x1)dx1. However, we only have access to the posterior model pθ(z1|x1) that is optimized concurrently and is an approximation to the true intractable posterior. Consequently, we can not directly construct an interpolant in the latent space that satisfies the requirements of SI. In the following, we address this issue by deriving Latent Stochastic Interpolants (LSI), though from an entirely different perspective than is considered by SI. Generative model with dynamic latent variables: Since we want to jointly learn the generative model in a latent space, we propose a latent variable model where the unobserved latent variables are assumed to evolve in continuous time according to the dynamics specified by an SDE of the form in eq. (1). Let pθ(x1|z1) be a parameterized stochastic decoder and hθ parameterized drift for eq. (1). Then, the generation process using our model is as following - first a sample z0 ∼ p0(z0)
is produced from a prior p0(z0), then z0 evolves according to the dynamics specified by eq. (1) using hθ from t = 0 to t = 1 to yield a z1, and finally an observation space sample is produced using the decoder pθ(x1|z1). In theory, we can now utilize the ELBO presented in section 2.1 to train this model. Note that, although the ELBO in eq. (3) supports arbitrary number of observations xtiat arbitrary times ti, in this paper we focus on a single observation x1 at t = 1. The ELBO in eq. (3) needs a variational approximation to the posterior pθ(zt|x1) which can be used to sample zt. This approximation is constructed as another dynamical model specified by the SDE in eq. (2). Unfortunately, for a general variational approximation specified by an arbitrary hϕ, simulating eq. (2)
would lead to significant computational burden for large problems during each training iteration and open the door to additional issues resulting from approximations needed for simulation of the SDE.

Instead, we explicitly construct the drift hϕ in eq. (2) such that zt can be sampled directly without simulation for any time t. Our scheme provides a scalable alternative that allows simulation free efficient training, as is common in the observation space diffusion models. Variational posterior with simulation free samples: Next we construct a variational posterior approximation, that enables easy sampling of zt without requiring the simulation of the SDE in eq. (2). Let z1 ∼ pθ(z1|x1) be a stochastic encoding of the observation x1 providing direct access to z1 at t = 1. Next, using the Diffusion Bridge specified by eq. (6) we construct a stochastic mapping between the prior p0(z0) and the aggregated approximate posterior Rpθ(z1|x1)dx1 at t = 1. The diffusion bridge, coupled with the encoder pθ(z1|x1) yields our approximate posterior pθ(zt|x1). However, p(z1|zt) is unknown in general. If we additionally assume that hϕ(zt, t) ≡ htzt and σ(zt, t) ≡ σt, then the original SDE in eq. (2) becomes linear with additive noise

$$d z_{t}=h_{t}z_{t}d t+\sigma_{t}d w_{t}$$
$$(7)$$
$$({\mathfrak{g}})$$
dzt = htztdt + σtdwt (7)
It is well known that for linear SDEs of the above form, the transition density p(zt|zs)*, t > s* is gaussian N(zt; astzs, bstI) (see section G) for some functions ast, bst that depend on ht, σt. Consequently, we can compute ∇ztln p(z1|zt) for a given zt as

$$\nabla_{z_{t}}\ln p(z_{1}|z_{t})=\frac{a_{t1}(z_{1}-a_{t1}z_{t})}{b_{t1}}\tag{8}$$

$$(10)$$

The transformed SDE in terms of the simplified drift and dispersion coefficients can be expressed as

$$d z_{t}=[h_{t}z_{t}+\sigma_{t}^{2}\nabla z_{t}\ln p(z_{1}|z_{t})]d t+\sigma_{t}d w_{t}$$
ln p(z1|zt)]dt + σtdwt (9)
Further, if we condition on the starting point z0, then the conditional density p(zt|z1, z0) can be expressed as following using the Bayes rule

$$p(z_{t}|z_{1},z_{0})={\frac{p(z_{1}|z_{t},z_{0})p(z_{t}|z_{0})}{p(z_{1}|z_{0})}}={\frac{p(z_{1}|z_{t})p(z_{t}|z_{0})}{p(z_{1}|z_{0})}}$$

where p(z1|zt, z0) = p(z1|zt) because of the Markov independence assumption inherent in eq. (2).

Note that all the factors on the right are gaussian. It can be shown that the conditional density p(zt|z1, z0) is also gaussian if the transition densities are gaussian and takes the following form

$$p(z_{t}|z_{1},z_{0})=\left({\frac{1}{2\pi}}{\frac{b_{01}}{b_{0t}b_{t1}}}\right)^{\frac{d}{2}}\exp\left(-{\frac{1}{2}}{\frac{b_{01}}{b_{0t}b_{t1}}}\left\|z_{t}-{\frac{b_{0t}a_{t1}z_{1}+b_{t1}a_{0t}z_{0}}{b_{01}}}\right\|^{2}\right)$$
$$(11)$$

Where a(·), b(·) are constant or time dependent scalars and d is the dimensionality of zt. Their specific forms depends on the choice of ht, σt. Refer to section G for details. zt can now be directly sampled without simulating the SDE, given a sample z0 and the encoded observation z1. Note that the assumptions made for eq. (7), while restrictive, do not limit the empirical performance. Latent stochastic interpolants: We can now define latent stochastic interpolants using reparameterization trick in conjuction with eq. (11) to parameterize zt as

$$(12)$$
$$z_{t}=\eta_{t}\epsilon+\kappa_{t}z_{1}+\nu_{t}z_{0},\quad\epsilon\sim N(0,I)$$
zt = ηtϵ + κtz1 + νtz0, ϵ ∼ N(0, I) (12)
For some functions ηt, κt, νt that depend on a(·), b(·). Note that η0 = η1 = 0, κ0 = ν1 = 0, κ1 =
ν0 = 1 since zt is sampled from a diffusion bridge with the two end points fixed at z0, z1. Equation (12) specifies a general stochastic interpolant, akin to the proposal in (Albergo et al., 2023), but now in the latent space. If we choose the encoder and decoder to be identity functions, then above can be viewed as an alternative way to construct stochastic interpolants in the observation space. Instead of choosing ht, σt first, we can instead choose κt, νt and infer the corresponding ht, σt. For example, choosing κt = *t, ν*t = 1 − t leads to σt = σ, a constant, and we arrive at the following

$$z_{t}=\sigma\sqrt{t(1-t)}\epsilon+tz_{1}+(1-t)z_{0},\quad\epsilon\sim N(0,I)\tag{13}$$

See section J for a detailed derivation. We use the above form for all the experiments in the paper. Further, if p0(z0) is chosen to be a standard gaussian then the interpolant simplifies to zt = tz1 +p(1 − t)(σ 2t + 1 − t)z0 (section M). With the above interpolants, we can now define the ELBO and optimize it efficiently with simulation free samples zt. We also derive the expressions for variance preserving choices of κt =
√*t, η*2 t + ν 2 t = 1 − t in section K, however we do not explore this interpolant empirically.

Constructing training objective using ELBO (eq. (3)): We first define u(zt, t) using eq. (9) as

$$u(z_{t},t)=\sigma_{t}^{-1}[h_{t}z_{t}+\sigma_{t}^{2}\nabla_{z_{t}}\ln p(z_{1}|z_{t})-h_{\theta}(z_{t},t)]\tag{14}$$  For the general latent stochastic interpolant $z_{t}=\eta_{t}\epsilon+\kappa_{t}z_{1}+\nu_{t}z_{0}$ (eq. (12)), we show that $u(z_{t},t)$
takes the following form
takes the following form  $$u(z_{t},t)=\sigma_{t}^{-1}\left[\left(\frac{d\eta_{t}}{dt}-\frac{\sigma_{t}^{2}}{2\eta_{t}}\right)\epsilon+\frac{d\kappa_{t}}{dt}z_{1}+\frac{d\nu_{t}}{dt}z_{0}-h_{\theta}(z_{t},t)\right]\tag{15}$$  See section H for the proof. This $u(z_{t},t)$ can be substituted into the ELBO in eq. (3) to construct a 
training objective. For example, with the choices κt = *t, ν*t = 1 − t, we get
$$(15)$$
$$u(z_{t},t)=\sigma^{-1}\left[-\sigma{\sqrt{\frac{t}{1-t}}}\epsilon+z_{1}-z_{0}-h_{\theta}(z_{t},t)\right]$$
$$(16)$$

See section J for details. We write a generalized loss based on the ELBO as

$$\mathbb{E}_{p(t)p(x_{1},x_{0})p_{\theta}(z_{1}|x_{1})p(z_{1}|z_{1},z_{0})}\left[-\ln p_{\theta}(x_{1}|z_{1})+\frac{\beta_{t}}{2}\left\|\sigma\sqrt{\frac{t}{1-t}}\epsilon+z_{1}-z_{0}-h_{\theta}(z_{t},t)\right\|^{2}\right]\right]$$
$$(17)$$

Where βt (discussed further in section 4) is a relative weighting term, similar in spirit to β-VAE
(Higgins et al., 2017; Alemi et al., 2018), allowing empirical re-balancing for metrics of interest, e.g. FID. Above loss is reminiscent of the SI training objective, but with an additional reconstruction term and the interpolants zt arising from the variational posterior. We use this training objective for all the experiments, and optimize it using stochastic gradient descent to jointly train all three components –
encoder pθ(z1|x1), decoder pθ(x1|z1) and latent SI model hθ(zt, t). Note that we choose pθ(x1|z1) to be a conditional gaussian in all experiments, resulting in a simple L2 decoder loss.

Observation-space stochastic interpolants: To elucidate the connection with observation-space SI (Albergo et al., 2023) we derive the corresponding training objective in our framework, yielding:

$$\mathbb{E}_{p(t)p(x_{1},x_{0})p(x_{1}|x_{1},x_{0})}\left[\frac{\partial_{t}}{2}\left\|\sigma\sqrt{\frac{t}{1-t}}\epsilon+x_{1}-x_{0}-h_{\theta}(x_{t},t)\right\|^{2}\right]\tag{18}$$

where βt has the same interpretation as in eq. (17), with βt = σ
−2corresponding to exact ELBO. See Section B for detailed proof. Comparing with the LSI loss (eq. (17)), the observation-space ELBO is precisely the LSI objective with the reconstruction term − ln pθ(x1|z1) removed and z replaced by x.

LSI recovers observation-space stochastic interpolants when the encoder and decoder are identity functions. All parameterizations (Section 4) and sampling procedures (Section 5) apply directly with z replaced by x. Lastly, the likelihood control property of the above objective is trivially established
- the objective corresponds to KL(Q∥Pθ) for βt = σ
−2and KL(p1∥pθ) ≤ KL(Q∥Pθ) (eq. (41)),
where p1 is the true data distribution and pθ is the data likelihood under the model.

Learnable priors: When the prior p0 is parameterized (e.g., pθ(z0) = N (µθ, Σθ)), the default construction above uses the same learnable prior for both processes (q0 = pθ), so KL(q0∥p0) =
0 and the ELBO retains the same form. The prior parameters are still learned: they affect the distribution of z0 in the path integral EQ[R∥u∥
2 dt], and gradients flow through z0 ∼ pθ(z0) via the reparameterization trick. Alternatively, if the variational process uses a fixed reference q0 ̸= pθ, the KL(q0∥pθ) term appears as an additional regularizer penalizing deviation from the reference. Same carries over to the observation-space stochastic interpolants as well.

## 4 Parameterization

Directly using the loss in eq. (17) leads to high variance in gradients and unreliable training due to the 
√1 − t in the denominator of the second term. Consequently, we consider several alternative
parameterizations for the second term, including denoising and noise prediction (see section C for details). Among the alternatives considered, we found the following parameterization, referred to as InterpFlow, to reliably lead to better results and we use it in all our experiments.
$$\frac{\beta_{t}}{2}\left\|-\sigma\sqrt{t}\epsilon+\sqrt{1-t}(z_{1}-z_{0})+\sqrt{t}z_{t}-\hat{h}_{\theta}(z_{t},t)\right\|^{2}$$
2(19)
$$(19)$$
Where hˆθ(zt, t) ≡
√tzt +
√1 − thθ(zt, t) and βt ≡ β/(1 − t) is a time t dependent weighting term, with β a constant. Instead of explicitly using the weights βt, due to 1 − t in the denominator, we consider a change of variable for t with the parametric family t(s) = 1 − (1 − s)
c with s ∼ U[0, 1]
uniformly sampled. It can be shown that p(t) ∝ (1−t)
1 c −1, therefore the change of variable provides the reweighting and we simply set βt = β, a constant. Empirically, we found that a value of c = 1
(i.e. a uniform schedule) works the best for all parameterizations during training and sampling, except for NoisePred and Denoising, which preferred c ≈ 2 during sampling. c < 1 led to degradation in FID. Figure 4 in appendix visualizes t(s) for various values of c. While the ELBO suggests using β = 1/σ2, we compute the two terms in eq. (17) as averages and experiment with different weightings. When used with optimizers like Adam or AdamW, β can be interpreted as the relative weighting of the gradients from the two terms for the encoder pθ(z1|x1). A lower value of β leads the encoder to focus purely on the reconstruction and is akin to using a pre-trained encoder-decoder pair as β → 0. A higher value of β forces the encoder to adapt its representation for the second term as well. We empirically study the effect of β in the experiments.

## 5 Sampling

For the InterpFlow parameterization, the learned drift hˆθ(zt, t) is related to the original drift hθ(zt, t) as hθ(zt, t) = (hˆ(zt, t) −
√tzt)/
√1 − t (see section F.2). We can sample from the model by discretizing the SDE in eq. (1), where σt = σ for the choices of κt = *t, ν*t = 1 − t. However, to derive a flexible family of samplers where we can independently tune the dispersion σ without retraining, we exploit Corollary 1 from Singh & Fischer (2024) to introduce a family of SDEs with the same marginal distributions as that for eq. (1)

$$d z_{t}=\left[h_{\theta}(z_{t},t)-{\frac{(1-\gamma_{t}^{2})\sigma^{2}}{2}}\nabla_{z_{t}}\ln p_{t}(z_{t})\right]d t+\gamma_{t}\sigma d w_{t}$$

Where γt ≥ 0 can be chosen to control the amount of stochasticity introduced into sampling. For example, setting γt = 0 yields the probability flow ODE for deterministic sampling. In general, to use eq. (20) for γt ̸= 1, the score function ∇ztln pt(zt) is needed as well. For the interpolant zt = σpt(1 − t)ϵ + tz1 + (1 − t)z0, the score can be estimated using

$$(20)$$
$$\nabla_{z_{t}}\ln p_{t}(z_{t})=-\frac{\mathbb{E}[\epsilon|z_{t}]}{\sigma\sqrt{t(1-t)}}$$

$$\nabla_{x}\ln p_{t}(z_{t})=-z_{t}+t h_{\theta}(z_{t},t)$$
$$(22)$$
$$(21)$$

See section E for the proof. However, for Gaussian z0, score can be computed from the drift hθ(zt, t)
(Singh & Fischer, 2024) as following (see section D for details)
∇x ln pt(zt) = −zt + thθ(zt, t) (22)
Section F provides detailed derivation of samplers for various parameterizations. For classifier free guided sampling (Ho & Salimans, 2022; Xie et al., 2024; Dao et al., 2023; Zheng et al., 2023; Singh
& Fischer, 2024), we define the guided drift as a linear combination of the conditional drift hθ(zt*, t, c*)
and the unconditional drift hθ(zt*, t, c* = ∅) as

$$h^{\mathrm{cfg}}(z_{t},t,c)\equiv(1+\lambda)h_{\theta}(z_{t},t,c)-\lambda h_{\theta}(z_{t},t,c=\varnothing)$$

where λ is the relative weight of the guidance, c is the conditioning information and c = ∅ denotes no conditioning. Note that λ = −1 corresponds to unconditional sampling, λ = 0 corresponds to conditional sampling and λ > 0 further biases towards the modes of the conditional distribution.

## 6 Experiments

We evaluate LSI on the standard ImageNet (2012) dataset (Deng et al., 2009; Russakovsky et al., 2015). We train models at various image resolutions and compare their sample quality using the Frechet Inception Distance (FID) metric (Heusel et al., 2017) for class conditional samples. All models were trained for 1000 epochs, except for the comparison in table 1 which reports FID at 2000 epochs. All results use deterministic sampler, using γt = 0, unless otherwise specified. A key implementation detail to note is that the encoder uses normalization and tanh to bound the scale of the latents. See sections O and P for additional details.

$$(23)$$

Table 1: **LSI enables joint learning for SI and cheaper sampling:** The latent space models achieve FID similar to observation space models of comparable size. However, the latent space model L has fewer parameters (reported in millions (M)) and FLOPs (reported in Giga (G)), as part of the parameters live in the encoder E and the decoder D. During sampling, encoder is not used, decoder is used only once, while the latent model L is run repeatedly, once for each sampling step. Therefore, FLOP savings from a computationally cheaper latent model accumulate with sampling steps.

| FID @ 2K epochs   | # Params (M)   | Flops (G)   |                |         |                |         |
|-------------------|----------------|-------------|----------------|---------|----------------|---------|
| Resolution        | Latent         | Observ.     | Latent (E/D/L) | Observ. | Latent (E/D/L) | Observ. |
| 64 × 64           | 2.62           | 2.57        | 392 (5/5/382)  | 398     | 15/15/161      | 201     |
| 128 × 128         | 3.12           | 3.46        | 392 (5/5/382)  | 400     | 59/59/327      | 466     |
| 256 × 256         | 3.91           | 3.87        | 393 (5/5/383)  | 405     | 240/240/450    | 1288    |

![6_image_0.png](6_image_0.png)

LSI enables joint learning for SI : While SI doesn't allow latent variables, LSI enables joint learning of Encoder (E), Decoder (D), and Latent SI models (L). In table 1 we compare FID across various resolutions for LSI models against SI models trained directly in observation (pixel) space. LSI models achieve FIDs similar to the observation space models indicating on par performance in terms of the final FID. Models for both were chosen with similar architecture and number of parameters and trained for 2000 epochs. Reference comparison with other methods is provided in section R. LSI enables computationally cheaper sampling: In table 1 we also report the parameter counts (in millions) as well as FLOPs (in Giga) for the observation space SI model as well as E, D and L models for the LSI. For the latent L model, FLOPs are reported for a single forward pass. First note that the parameters in LSI are partitioned across the encoder E, the decoder D and the latent L models. At sampling time, encoder is not used, decoder is used only once, while the latent model is run multiple times, once for each step of sampling. Therefore, while the overall FLOP count for LSI and Observation space SI models is similar for a single forward pass, sampling with multiple steps becomes significantly cheaper. For example, sampling with 100 steps leads to 73.6% reduction in FLOPs for sampling 128 × 128 images and 48.6% for 256 × 256 images. Joint learning is beneficial: In fig. 1(left panel) we plot the FID as the weighting term β is varied (eq. (19)). A higher β forces the encoder to adapt the latents more for the second term of the loss. We observe that FID improves as β increases, going from 4.53 (for β → 0) to 3.75 (≈ 17% improvement) for β = 0.0001, indicating that this adaptation is beneficial for the overall performance. Eventually, FID worsens as β is increased further. We also plot the reconstruction PSNR for each of these models Table 2: **Joint training helps mitigate capacity shift:** We evaluate the effect of moving first k and last k convolutional blocks from the latent model L to encoder and decoder respectively, for 128×128 resolution models. This results in the overall parameter count staying roughly the same, but the number of FLOPs required for sampling changing significantly. We observe that the model trained with β > 0 perform better and maintains FID well, in comparison to the independently trained model
(β → 0), even when capacity is shifted away from the latent model L, resulting in 8.5% reduction in FLOPs for sampling from k = 0 to k = 6.

| k   | FID (β > 0)   | FID (β → 0)   | #Params. (E/D/L)   | FLOPs (E/D/L)   |
|-----|---------------|---------------|--------------------|-----------------|
| 0   | 3.76          | 4.31          | 392 (5/5/382)      | 59/59/327       |
| 3   | 3.91          | 4.55          | 389 (9/8/372)      | 68/66/313       |
| 6   | 3.96          | 4.87          | 387 (13/12/362)    | 75/73/299       |
| 9   | 4.61          | 4.98          | 383 (16/16/351)    | 82/80/284       |

in orange and observe that increasing β essentially trades-off reconstruction quality with generative performance. For too large a β, poor reconstruction quality leads to worsening FID. The dashed line indicates the performance when the encoder-decoder are trained independently of the latent model, limit of β → 0. We implement it as a stop gradient operation in implementation, where the gradients from the second term of the loss are not backpropagated into z1. To further assess the benefits of joint training, in table 2 we compare the FIDs between jointly trained model (β > 0) and independently trained model (β → 0) as parameters are shifted from the latent model L to the encoder E and decoder D models, by moving first k and last k convolutional blocks from the latent model to the encoder and the decoder respectively. While this keeps the total parameter count roughly the same, the number of FLOPs required for sampling changes significantly. The jointly trained model performs better and maintains FID well even when capacity shifts away from the latent model, resulting in 8.5% reduction in FLOPs required for sampling from k = 0 to k = 6. Encoder noise scale affects performance: The stochasticity of the encoder pθ(z1|x) has a significant impact on the performance. We parameterize the encoder as a conditional Gaussian N(z1; µθ(x), Σθ(x)) where Σ(x) is assumed to be diagonal. We experimented with a purely deterministic encoder (Σθ(x) = 0), learned Σθ(x) and constant noise Σθ(x) = cI. In fig. 1(right panel) we plot FID as the encoder output stochasticity c is varied. Dashed line indicates performance with learned Σθ(x). A deterministic encoder (c = 0) performs poorly. FID improves as the noise scale c is increased, until eventually it degrades again. While learned Σθ(x) (dashed line) performs well, fixed c models achieved higher FID.

InterpFlow **parameterization performs better than alternatives:** In table 3 we compare different parameterizations discussed in section 4 and section C. The InterpFlow parameterization consistently led to better FID. Both OrigFlow and NoisePred parameterizations exhibited higher variance gradients and noisy optimization. While Denoising parameterization resulted in less noisy training, InterpFlow parameterization led to fastest improvement in FID.

LSI supports diverse p0: In table 4 we report FID achieved by LSI using different prior p0(z0) distributions. While Gaussian p0 performs the best, other choices for p0 yield competitive results indicating that LSI retains one of the key strengths of SI - support for diverse p0 distributions. See section N for additional details. To allow flexible sampling using eq. (20), we modified latent SI
model to output extra output channels and augmented the loss with another term to estimate E[ϵ|zt].

Equation (21) was used to compute the score and sample with the deterministic sampler using γt = 0.

LSI supports flexible sampling: In fig. 2 and fig. 3 we qualitatively demonstrate flexible sampling with LSI model for popular use cases. Figure 2 demonstrates compatibility of classifier free guidance (CFG) with LSI, using eq. (22). Increasing guidance weight λ results in more typical samples. First z0 is sampled from p0(z0), Gaussian in this example, following which eq. (20) is simulated forward in time, using class conditional drift with different guidance weights λ. In fig. 3 a given 'Original' image (shown leftmost) is first encoded to yield it's representation z1, which is then inverted by simulating probability flow ODE (setting γt = 0 in eq. (20)) backward in time from t = 1 to t = 0, yielding z0 (similar to DDIM inversion (Song et al., 2020a)). Using this z0 as starting point, eq. (20)

| Parameterization   | FID @1K epochs   |
|--------------------|------------------|
| OrigFlow           | 4.56             |
| NoisePred          | 4.73             |
| Denoising          | 4.28             |
| InterpFlow         | 3.76             |

Table 3: **Effect of parameterization:** We compare various parameterization schemes at 128 × 128 resolution. InterpFlow parameterization performs better against the alternatives.

Table 4: **LSI supports diverse** p0: LSI retains one of the key strengths of SI - support for arbitrary p0 distribution. Different p0 achieve competetive FID for 128 × 128 resolution model.

![8_image_0.png](8_image_0.png)

| p0               | FID @1K epochs   |
|------------------|------------------|
| Uniform          | 4.81             |
| Laplacian        | 4.45             |
| Gaussian         | 3.76             |
| Gaussian Mixture | 4.26             |

is simulated forward is time using γt ≡ γ(1 − t) for different values of γ. We show three samples for each value of γ and observe increasing diversity with increasing γ. See section Q for additional details and results.

## 7 Related Work

Latent Stochastic Interpolants (LSI) draw from insights in diffusion models, latent variable models, and continuous-time generative processes. We discuss key works from these areas in the following. Diffusion Models: Diffusion models, originating from foundational work on score matching (Vincent, 2011; Song & Ermon, 2019) and early variational formulation (Sohl-Dickstein et al., 2015), gained prominence with Denoising Diffusion Probabilistic Models (DDPMs) (Ho et al., 2020). Subsequent improvements focused on architectural choices and learned variances (Nichol & Dhariwal, 2021), faster sampling via Denoising Diffusion Implicit Models (DDIMs) (Song et al., 2020a), progressive distillation (Salimans & Ho, 2022), and powerful conditional generation through techniques like classifier-free guidance (Ho & Salimans, 2022). Further exploration of the design space (Karras et al., 2022; 2024) has lead to highly performant models. More recently, diffusion inspired consistency models (Song et al., 2023) have emerged, offering efficient generation. LSI complements these with a flexible method for jointly learning in a latent space using richer prior distributions. Latent Variable Models and Expressive Priors: Variational Autoencoders (VAEs) (Kingma et al.,
2013; Rezende et al., 2014) learn a compressed representation z of data x, but are limited by the expressiveness of the prior p(z) (NVAE (Vahdat & Kautz, 2020), LSGM (Vahdat et al., 2021)), as they typically use simple priors (e.g., isotropic Gaussian). LSI addresses this by jointly learning a flexible generative process in the latent space, enabling powerful transformations of the simple prior.

Early work (Sohl-Dickstein et al., 2015) derived ELBO for discrete time diffusion models, while Variational Diffusion Models (VDM) (Kingma et al., 2021) interpret diffusion models as a specific type of VAE with Gaussian noising process. In contrast, while LSI also optimizes an ELBO, it allows for a broader choice of the prior p(z0) and the transforms mapping the prior to the learned aggregated posterior. Our work is similar in spirit to models like NVAE, which employed deep hierarchical latent representations, and LSGM, which proposed training score-based models in the latent space of a VAE, but offers a flexible framework similar to SI allowing a rich family of priors and latent space

![9_image_0.png](9_image_0.png)

dynamics. Note that LDM (Rombach et al., 2022) train a diffusion generative model in the latent space of a *fixed* encoder-decoder pair - making their latents actually *observed* from the point of view of generative modeling. Continuous-Time Generative Processes: While diffusion models have been formulated and studied using continuous time dynamics (Song et al., 2020b;a; Kingma et al., 2021; Vahdat et al., 2021), their relation to Continuous Normalizing Flows (CNFs) (Chen et al., 2018; Grathwohl et al., 2019) offers another perspective on continuous-time transformations. Early training challenges with the CNFs have been addressed by newer methods like Flow Matching (FM) (Lipman et al., 2022; Xu et al., 2022), Conditional Flow Matching (CFM) (Neklyudov et al., 2023; Tong et al., 2023), and Rectified Flow (Liu et al., 2022). These approaches propose simulation-free training by regressing vector fields of fixed conditional probability paths. However, likelihood control is typically not possible (Albergo et al., 2023), consequently extension to jointly learning in latent space is ill-specified. In contrast, LSI optimizes an ELBO, offering likelihood control along with joint learning in a latent space. Stochastic Interpolants (SI) (Albergo et al., 2023) provides a unifying perspective on generative modeling, capable of bridging any two probability distributions via a continuous-time stochastic process, encompassing aspects of both flow-based and diffusion-based methods. While SI formulates learning the velocity field and score function directly in the observation space using pre-specified stochastic interpolants, LSI arrives at a similar objective in the latent space, as part of the ELBO, from the specific choices of the approximate variational posterior. LSI reduces to SI when encoder and decoder are chosen to be Identity functions. SI is related to the Optimal Transport and the Schrödinger Bridge problem (SBP) which have been explored as a basis for generative modeling (De Bortoli et al., 2021; Wang et al., 2021; Shi et al., 2023). While LSI learns a transport, its primary objective is data log-likelihood maximization via the ELBO, rather than solving a specific OT or SBP.

## 8 Conclusion

In this paper, we introduced Latent Stochastic Interpolants (LSI), generalizing Stochastic Interpolants to enable joint end-to-end training of an encoder, a decoder, and a generative model operating entirely within the learned latent space. LSI overcomes the limitation of simple priors of the normal diffusion models and mitigates the computational demands of applying SI directly in high-dimensional observation spaces, while preserving the generative flexibility of the SI framework. LSI leverage SDE-based Evidence Lower Bound to offer a principled approach for optimizing the entire model. We validate the proposed approach with comprehensive experimental studies on standard ImageNet benchmark. Our method offers scalability along with a unifying perspective on continuous-time generative models with dynamic latent variables. However, to achieve scalable training, our approach makes simplifying assumptions for the variational posterior approximation. While restrictive, and common with other methods, these assumptions do not seem to limit the empirical performance.

## Acknowledgments

We would like to thank Kevin J. Shih and Ian Fischer for proofreading early drafts of this manuscript and providing valuable feedback.

## Reproducibility Statement

We have included detailed proofs of all the key theoretical results in the appendix. Sections 6 and O provide key training and evaluation setup details. Section P provides the necessary architecture details to reproduce the models used in the experiments. Section Q provides additional sampling setup details.

## References

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. *arXiv preprint arXiv:2303.08797*, 2023.

Alexander Alemi, Ben Poole, Ian Fischer, Joshua Dillon, Rif A Saurous, and Kevin Murphy. Fixing a broken elbo. In *International conference on machine learning*, pp. 159–168. PMLR, 2018.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper_files/paper/ 2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.

Quan Dao, Hao Phung, Binh Nguyen, and Anh Tran. Flow matching in latent space. arXiv preprint arXiv:2307.08698, 2023.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. Advances in Neural Information Processing Systems, 34:17695–17709, 2021.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE conference on computer vision and pattern recognition*, pp. 248–255. Ieee, 2009.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

Will Grathwohl, Ricky T. Q. Chen, Jesse Bettencourt, and David Duvenaud. Scalable reversible generative models with free-form continuous dynamics. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=rJxgknCcK7.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. *Advances in neural* information processing systems, 30, 2017.

Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In *International conference on learning representations*, 2017.

Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*,
2022.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In *International Conference on Machine Learning*, pp. 13213–13232. PMLR, 2023.

Emiel Hoogeboom, Thomas Mensink, Jonathan Heek, Kay Lamerigts, Ruiqi Gao, and Tim Salimans.

Simpler diffusion (sid2): 1.5 fid on imagenet512 with pixel-space diffusion. arXiv preprint arXiv:2410.19324, 2024.

Allan Jabri, David Fleet, and Ting Chen. Scalable adaptive computation for iterative generation.

arXiv preprint arXiv:2212.11972, 2022.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusionbased generative models. *Advances in Neural Information Processing Systems*, 35:26565–26577, 2022.

Tero Karras, Miika Aittala, Jaakko Lehtinen, Janne Hellsten, Timo Aila, and Samuli Laine. Analyzing and improving the training dynamics of diffusion models. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pp. 24174–24184, 2024.

Patrick Kidger, James Foster, Xuechen Chen Li, and Terry Lyons. Efficient and accurate gradients for neural sdes. *Advances in Neural Information Processing Systems*, 34:18747–18761, 2021.

Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Pagoda: Progressive growing of a one-step generator from a low-resolution diffusion teacher. *Advances in Neural Information Processing Systems*, 37:
19167–19208, 2024.

Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. *Advances in Neural Information Processing Systems*, 36:65484–65516, 2023.

Diederik Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. *Advances* in neural information processing systems, 34:21696–21707, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. *arXiv preprint* arXiv:1412.6980, 2014.

Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. Xuechen Li, Ting-Kam Leonard Wong, Ricky TQ Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In International Conference on Artificial Intelligence and Statistics, pp. 3870–3882. PMLR, 2020.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic dynamics from samples. In *International conference on machine learning*, pp. 25858– 25889. PMLR, 2023.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In *International conference on machine learning*, pp. 8162–8171. PMLR, 2021.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In *International conference on machine learning*, pp. 1278–1286. PMLR, 2014.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10684–10695, June 2022.

Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision (IJCV)*, 115
(3):211–252, 2015. doi: 10.1007/s11263-015-0816-y.

Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models. arXiv preprint arXiv:2202.00512, 2022.

Simo Särkkä and Arno Solin. *Applied stochastic differential equations*, volume 10. Cambridge University Press, 2019.
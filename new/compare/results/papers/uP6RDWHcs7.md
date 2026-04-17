Anonymous authors Paper under double-blind review

## Abstract

Current density modeling approaches suffer from at least one of the following shortcomings: expensive training, slow inference, approximate likelihood, mode collapse or architectural constraints like bijective mappings. We propose a simple yet powerful framework that overcomes these limitations altogether. We define our model qθ(x) through a parametric distribution q(x|w) with latent parameters w. Instead of directly optimizing the latent variables w, our idea is to marginalize them out by sampling them from a learnable distribution qθ(w), hence the name Marginal Flow. In order to evaluate the learned density qθ(x) or to sample from it, we only need to draw samples from qθ(w), which makes both operations efficient.

The proposed model allows for exact density evaluation and is orders of magnitude faster than competing models both at training and inference. Furthermore, Marginal Flow is a flexible framework: it does not impose any restrictions on the neural network architecture, it enables learning distributions on lower-dimensional manifolds (either known or to be learned), it can be trained efficiently with any objective (e.g. forward and reverse KL divergence), and it easily handles multimodal targets. We evaluate Marginal Flow extensively on various tasks including synthetic datasets, simulation-based inference, distributions on positive definite matrices and manifold learning in latent spaces of images.

## 1 Introduction

000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Density estimation models are ubiquitous in machine learning and have been used for a wide range of purposes. Their overarching characteristic is to provide an approximation to some probability distribution. The most popular use case is probabilistic modeling of data with the goal of generating new instances. The underlying assumption is that there exists an unknown generative process that generated the data in the first place. Successful applications include generation of images, e.g. Rombach et al. (2022), text-to-audio, e.g. Liu et al. (2023), and text-to-video, e.g. Singer et al. (2023). Other popular applications of deep generative models include protein structure prediction, e.g. Abramson et al. (2024), and drug discovery, e.g. Zeng et al. (2022). Rather than focusing on generating new samples, another interesting use case of density estimation models lies in modeling and reasoning about the probability distribution itself, which has relevant applications in the sciences. Common settings include computation of high-dimensional integrals and intractable likelihoods or posteriors. This is maybe best exemplified by Bayesian inference, e.g. Rezende & Mohamed (2015). Applications include cosmology, e.g. Alsing et al. (2018), neurosciences, e.g. Goncalves et al. (2020), simulation-based inference, e.g. Cranmer et al. (2020), and many more. Learning probability distributions on manifolds is also a challenging problem that can be addressed with density estimation models, e.g. Gemici et al. (2016); Chen & Lipman (2024). The two fundamental operations that characterize a density estimation model are sampling from the learned distribution and evaluating its probability density. Most models show a trade-off in efficiency between the two operations, which have their own specific challenges. On the one hand, evaluating the probability density often requires restricting the learned transformations to bijections that are carefully designed to avoid computing expensive Jacobian determinants, as in the case of Normalizing Flows (NF) (Kobyzev et al., 2020). Alternatively, the true density can be bounded like in VAEs (Kingma & Welling, 2014; Rezende et al., 2014) and afterwards estimated (Burda et al., 2015), which is still very expensive. Therefore, most generative models rely on surrogate objectives

# Marginal Flow: A Flexible And Efficient Framework For Density Estimation

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Table 1: Comparison of Marginal Flow with other deep generative models: GANs, VAEs, Energy- Based models (EB), Flow Matching (FM), Normalizing Flow (NF), and Free-form Flows (FFF). The Table is inspired by Bond-Taylor et al. (2021).

| Feature                           | GANs   | VAEs   | EB   | FM   | NF   | FFF   | Ours   |
|-----------------------------------|--------|--------|------|------|------|-------|--------|
| Efficient exact likelihood        | ✗      | ✗      | ✗    | ✗    | ✓    | ✗     | ✓      |
| Efficient (single-step) sampling  | ✓      | ✓      | ✗    | ✗    | ✓    | ✓     | ✓      |
| Efficient training                | ✗      | ✓      | (✓)  | (✓)  | ✗    | (✓)   | ✓      |
| Free-form Jacobian                | ✓      | ✗      | ✓    | ✓    | ✗    | ✓     | ✓      |
| Lower dim. base distr. (manifold) | ✓      | ✓      | ✗    | ✗    | ✗    | ✓     | ✓      |

that do not require the evaluation of the probability densities, while still allowing for high-fidelity sample generation. This is the case for Energy-Based (EB) models (Swersky et al., 2011), Diffusion models (Sohl-Dickstein et al., 2015) and Flow Matching (FM) (Lipman et al., 2023). On the other hand, sampling often requires multi-step processes that transform samples from a simple distribution into samples from the learned distribution, e.g. Flow Matching and Diffusion models. The tradeoff between efficient log-likelihood evaluation and efficient sampling is clear in NF, which can be efficient only at either sampling or evaluating the density. Which of the two operations is more efficient also determines which objective function can be used for training. In many applications it is beneficial to learn a density on a lower-dimensional space. For instance, real data is often assumed to live on a lower-dimensional manifold (Fefferman et al., 2016). Most models, like Diffusion, FM and NF, cannot account for a change in the dimensionality while others like GANs (Goodfellow et al., 2014) or Free-form Flows (Draxler et al., 2024) can, but suffer from other disadvantages like approximate likelihood and unstable training. Contribution. We propose a novel density estimation framework that alleviates altogether the common shortcomings of current approaches. We define our model through a parametric distribution q(x|w) with latent parameters w. Instead of directly optimizing the latent variables w, we marginalize them out by sampling w from a learnable distribution qθ(w). As we do not need to evaluate qθ(w) at any point, but only to sample from it, we are free to generate samples in a very flexible and efficient way. To generate w, we feed-forward samples from a base distribution of choice through an unconstrained learnable neural network. Overall, the proposed approach allows for efficient exact density evaluation and efficient sampling. Furthermore, it does not pose any restrictions (e.g. bijectivity) on the neural network and allows for learning a lower-dimensional manifold alongside the density. In Table 1, we provide a high-level comparison between popular density estimation models and Marginal Flow. Overall, our contributions can be summarized as follows:
- We introduce a novel density estimation framework called Marginal Flow. - We demonstrate the flexibility of the framework: it allows for learning lower-dimensional manifolds, it can easily handle multi-modal distributions, and can be tailored to the data with the choice of the parametric distribution q(x|w).

- We show empirically that Marginal Flow is orders of magnitude faster than competing models both at training and inference.

- Lastly, we showcase Marginal Flow on extensive experiments with synthetic data (trained via log-likelihood and reverse KL divergence), simulation-based inference, distributions over positive-definite matrices, and finally on MNIST digits and the JAFFE faces dataset.

## 2 Marginal Flow 2.1 Model Definition

Marginalization Let q(x|w) with x ∈ R
d be a family of distributions parametrized by w ∈ R
p and assume that, for given w, it is easy to evaluate the density of q(x|w) to sample from it. We can compute q(x) by marginalizing out w over some q(w):

$$q(\mathbf{x})=\int q(\mathbf{x}|\mathbf{w})q(\mathbf{w})d\mathbf{w}=\mathbb{E}_{\mathbf{w}\sim q(\mathbf{w})}\left[q(\mathbf{x}|\mathbf{w})\right]\ .$$
Zq(x|w)q(w)dw = Ew∼q(w)[q(x|w)] . (1)
2 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 In our model, we let q(x|w) be a distribution of choice parametrized by w and we let q(w) be freely learnable: q(w) → qθ(w). The resulting marginal q(x) is universal for many families of distributions q(x|w), e.g. if q(x|w) is a kernel (Micchelli et al., 2006). We will often assume q(x|w) = N (x|µ = w, Σ = diag(σ1*, . . . , σ*d)), for which p = d, and learnable variances (alongside θ). However, we show that other choices of q(x|w) can be beneficial, depending on the setting. Definition. Motivated by the marginalization in Eq. 1, we define our model as follows:

$$q_{\theta}(\mathbf{x}):={\frac{1}{N_{c}}}\sum_{i=1}^{N_{c}}q(\mathbf{x}|\mathbf{w}_{\theta,i})\qquad{\mathrm{where}}\qquad\mathbf{w}_{\theta,i}\sim q_{\theta}(\mathbf{w})~.$$
$$(2)$$
q(x|wθ,i) where wθ,i ∼ qθ(w) . (2)
The density qθ(x) can be exactly evaluated and efficiently sampled from. Nc is the number of parameters drawn from qθ(w) and is not required to be fixed. In fact, the parameters wθ,i are not fixed themselves but rather *resampled* from qθ(w) at each iteration, which effectively renders the marginalization in Eq. 1. As we will argue in the next paragraph, there is a crucial difference with respect to directly optimizing a finite set of mixtures {wi}
Nc i=1. Another important aspect is that we do not need to evaluate qθ(w) but only to sample from it. Therefore, we can construct samples in a very flexible way and in a single step: we first sample from a distribution of choice pbase(z) with z ∈ R
m and then transform them via a learnable mapping to the space of latent parameters w ∈ R
p.

Relevantly, to do so we can use an unconstrained learnable function fθ : z ∈ R
m 7→ w ∈ R
p:
wθ,i := fθ(zi) with zi ∼ pbase(z) . (3)
The resulting samples wθ,i := fθ(zi) will be samples from some (learnable) distribution qθ(w). The neural network fθ(z) is thus the trainable part of the model. In our experiments, a small MLP with 3-5 layers and 256 neurons was enough. Unlike most density estimation models, Marginal Flow is efficient both at sampling and at evaluating the probability density, as we will see in Section 2.2. Furthermore, in contrast to competing models, we can learn a density with support on a lowerdimensional manifold by simply choosing a base distribution with support in R
m with *m < d*.

Motivation for marginalization. In order to understand the importance of the marginalization aspect, consider the case where we have a finite number of wi and, instead of integrating them out, we optimize them. Without marginalization, the model reduces to a simple mixture model optimized over a fixed set of mixture components {wi}
Nc i=1, e.g. a Gaussian Mixture Model (GMM)
if q(x|w) = N (x|µ = w, Σ = σ1). In this case, learning a target distribution amounts to placing the Nc Gaussians in an optimal way. The expressiveness and scalability of the model are then fundamentally limited by the number of mixtures Nc. Instead of optimizing over fixed {wi}
Nc i=1, our approach relies on the marginalization of w, sampled from qθ(w). We optimize the parameters θ of the neural network fθ(z), and we resample w ∼ qθ(w) at each iteration. The resampling induces an approximation to the marginal distribution in Eq. 1, rather than just a finite mixture. As illustrated in Figure 1, even with the same nominal number of mixtures (e.g. 10), only the marginalized model is able to learn a smooth density. As such, the modeling capacity is not directly linked to Nc anymore. The marginalization prevents the collapse to a GMM and spreads qθ(w) to cover the entire target.

![2_image_0.png](2_image_0.png) 

## 2.2 Efficient Evaluation And Sampling

Sampling the parameters wi Figure 2 (**left**). In order to evaluate the modeled density qθ(x) or to sample from it, we first need to sample wi, which parametrize q(x|wi). This is done ef162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

![3_image_0.png](3_image_0.png)

ficiently by feed-forwarding samples {zi}
Nc i=1 from a base distribution of choice: wi = fθ(zi)
with zi ∼ pbase(z). With the sampled {wi}
Nc i=1, our model in Eq. 2 resembles a mixture model with Nc components. Note, however, that the {wi}
Nc i=1 are not fixed but sampled again for each evaluation or sampling of qθ(x). The neural network fθ is unconstrained. **Evaluation: Figure 2** (*center*). In order to evaluate the density qθ(x) at a given point x, we use the definition in Eq. 2. Given the sampled parameters {wi}
Nc i=1, we only need to evaluate each q(x|wi) on x, which is chosen to have a simple closed-form density function. Note that, in contrast to other density estimation models, the evaluation of the density does not require inverting fθ(zi), computing detJfθ or solving an ODE. Sampling from qθ(x): Figure 2 (**right**). Sampling as in Eq. 2 is also efficient, just like sampling from a mixture model. Given the sampled parameters {wi}
Nc i=1, we first need to sample a component wj and then sample from the associated distribution q(x|wj ), with j ∈ {1*, . . . , N*c}. To draw N samples, we sample N indices with replacement from {1*, . . . , N*c}.

Empirical runtime. We now empirically measure runtime for sampling and evaluating the exact density and compare against competing models. Note that only Marginal Flow and Normalizing Flow (NF) provide exact density by construction. As shown in Figure 3, Marginal flow is orders of magnitude faster than competing methods in terms of both sampling and density evaluation, where FM is Flow Matching and FFF is Free-form Flows. Sampling is as efficient as in FFF, since both only require drawing from a base distribution and passing the samples through a neural network. For further details, see the Appendix in Section A.3.1.

![3_image_1.png](3_image_1.png)

## 2.3 Flexibility Of Marginal Flow

Lower-dimensional latent distribution. Most density estimation models, like Flow Matching and Normalizing Flows, learn mappings that preserve the dimensionality and cannot learn densities on lower-dimensional manifolds. Some work tries to overcome this issue either by resorting on approximations (Brehmer & Cranmer, 2020) or by restricting the transformations (Khorashadizadeh et al., 2023; Negri et al., 2025). In contrast, with our model in Eq. 2, we have the freedom of choosing the dimensionality of the base distribution, i.e. pbase(z) with support in R
m with *m < d*. Also in this case we can evaluate qθ(x) exactly and learn the manifold alongside the density. In Figure 4 we showcase Marginal Flow and competing models on a density defined on a (unknown) 1D manifold.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Conditional distribution. As wo do not have any requirements on the neural network fθ(z),
Marginal Flow can be readily extended to model conditional distributions. The conditioning variables could be appended to the input fθ(z) → fθ(z; c) or one could use a hypernetwork that takes c as input and returns the neural network parameters fθ(z) → fθ(c)(z). Furthermore, the base distribution can also be conditioned on c: pbase(z) → pbase(z; c).

Multi-modal targets. Marginal Flow can naturally account for multi-modal targets thanks to the unconstrained neural network fθ(z). Most generative models, like Normalizing Flows and Flow Matching, learn (directly or indirectly) a bijection between a base distribution and the target distribution. However, bijections struggle to learn new modalities and have limited expressiveness (Liao & He, 2021). Even with a multi-model base distribution, bijections will still struggle to match the modalities in the target with those of the base distribution. Furthermore, many density estimation models suffer from mode collapse during training (He et al., 2019; Kossale et al., 2022). In Figure 5 we showcase how easily Marginal Flow can learn multi-modal targets compared to other models.

Ground truth Training samples **Marginal Flow** Flow Matching Normalizing Flow Free-form Flow

![4_image_1.png](4_image_1.png)

Training objectives. Density estimation models are usually trained through an objective that requires sampling, evaluating the (exact) density or both. However, current approaches are efficient only at either one or the other. For instance, models trained on data via forward KL divergence (i.e. log-likelihood) require efficient density evaluation while models trained on unnormalized targets via reverse KL divergence require efficient sampling. However, one could wish to use both objectives to combine information from observations and unnormalized targets or to mitigate the mean-seeking (mode-seeking) behavior of the forward (reverse) KL divergence. Since Marginal Flow is efficient both at sampling and evaluation, it can be trained efficiently with most objectives; see Appendix A.2. Extension to other mixtures. The proposed model in Eq. 2 leaves complete freedom in the choice of q(x|w), as long as it can be parametrized by some w. In most experiments we employ a Gaussian with learnable variances, i.e. q(x|w) = N (x|µ = w, Σ = diag(σ1*, . . . , σ*d)). However, other choices are possible depending on the application. For instance, when modeling distributions on the probabilistic simplex, we can use the Dirichlet distribution. We can model distributions on symmet-

![4_image_0.png](4_image_0.png)

270 271 272 273

## 274

275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 ric positive-definite matrices by choosing q(x|w) to be a Wishart, which we showcase in Section 4.3. Relevantly, the choice of q(x|w) does not affect the structure of the proposed framework.

## 3 Related Work

One of the earliest attempts to use deep learning for generative modeling are Energy-based (EB) models (Swersky et al., 2011). Instead of modeling a normalized density, EB models learn the negative log-probability. Despite their flexibility, computing the exact density and sampling from the model is generally expensive (Song & Kingma, 2021). Closely related are diffusion models (Sohl- Dickstein et al., 2015), which learn how to reverse a fixed noising process by estimating at each step the gradient of the log-density. Diffusion models can produce high-quality samples (Rombach et al., 2022; Liu et al., 2023), but still require multi-step sampling and do not provide the exact density. Another approach is to model the observed density with unobserved latent variables. VAEs (Kingma & Welling, 2014; Rezende et al., 2014) encode data into a latent space and are trained via a lower bound on the log-likelihood. In contrast to EB models, VAEs can be sampled in a single step. However, VAEs have limited expressiveness and suffer from posterior collapse (He et al., 2019). Another latent variable model - GANs (Goodfellow et al., 2014) - consists of a generator that creates samples from a latent distribution and a discriminator trained to distinguish generated samples from real ones. GANs can generate high-fidelity images (Karras et al., 2019) but are unstable and suffer from mode collapse (Kossale et al., 2022). Neither GANs nor VAEs provide the exact likelihood. Normalizing Flows (NFs) (Papamakarios et al., 2021) provide a principled way to compute the exact density. NFs transform a base distribution through bijections and account for the probability change via the Jacobian determinant, which is expensive to compute. Thanks to their exact density, NF have been applied for posterior approximations (Rezende & Mohamed, 2015). Additional limitations of NFs arise from the limited expressivity of bijective layers (Liao & He, 2021). Efficiency could be obtained using approximate bijections and by approximating the Jacobian determinant (Draxler et al., 2024), which however precludes sound statistical understanding and evaluation of the exact log-likelihood. Lipman et al. (2023) proposed to learn instead a velocity field that transforms the base distribution into the target. While this approach scales to high-dimensions, it cannot handle lower-dimensional base distributions and still requires expensive ODE solvers to compute the exact density. For a comprehensive review on generative models we refer to Bond-Taylor et al. (2021).

## 4 Experiments

First, we show on synthetic data that Marginal Flow can learn complex distributions both via loglikelihood and reverse KL divergence training. We also show that it converges more quickly than competing models. Second, we showcase how Marginal Flow can learn complex conditional distributions and achieve state-of-the-art results for simulation-based inference. Third, we show that Marginal Flow can be easily adapted to learn distributions on positive-definite matrices by simply changing the parametric form of q(x|w). Lastly, we showcase applications in computer vision as well: we learn densities on lower-dimensional manifolds on MNIST and on the JAFFE face dataset.

## 4.1 Synthetic Datasets

Log-likelihood training. As illustrative examples, we picked 4 common synthetic datasets (Two moons, Pinwheel, *Swiss Roll* and *Checkerboard*) and 1 additional multi-modal distribution (Mixture of Gaussians). We train Marginal Flow by maximizing the log-likelihood, which is reported explicitly in the Appendix 6. In Figure 6 we showcase that Marginal Flow can perfectly learn all densities without needing any fine-tuning. Next, we study the ability of Marginal Flow to learn densities when a limited number of observations is available. In particular, we compare against Flow Matching, Normalizing Flow and Free-form Flows with an increasing number of training points
{100, 200, 500, 1000}. For a fair comparison we used a comparable amount of parameters in each model. In the Appendix in Figure 13, we show the learned densities, which are particularly accurate for Marginal Flow, already in few-sample regimes. In Figure 7 we showcase the test log-likelihood during training for all models and datasets when train on 1000 points. Marginal Flow converges orders of magnitude quicker than competing models.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_1.png](6_image_1.png)

Reverse KL divergence training We additionally show that Marginal Flow can be trained in the reverse KL direction as well, namely without observations and only guided by the (unnormalized) density of the target distribution. This type of training requires an efficient computation of the exact log-likelihood, which is possible only for Normalizing Flow. Some attempts to make Flow Matching work in this direction have been made but remain limited (Tong et al., 2024). We tried with a score-matching objective but it led to unstable training. We trained Marginal Flow and Normalizing Flow with a reverse KL objective and compared the learned densities in terms of test KL. Marginal Flow achieved superior or comparable performance with Normalizing Flow, see Figure 8 (*left*), and showed better density reconstruction quality, see Figure 8 (*right*). Note that we do not use the Checkerboard dataset because its density is constant and has gradients equal to zero everywhere.

![6_image_2.png](6_image_2.png)

## 4.2 Simulation-Based Inference

As argued in Section 2.3, with the proposed framework we can easily learn conditional distributions as well. We showcase Marginal Flow on complex conditional distributions by training it on the Simulation-Based Inference (SBI) benchmark (Lueckmann et al., 2021). SBI data consists of tuples
{xi, θi}i, where θi are parameters sampled from a prior p(θ) and xi are samples from a simulator p(x|θi) parameterized by θi. Given tuples of observations {xi, θi}i, the goal is to learn the posterior p(θ|xj ) of a new xj . Evaluation is performed in terms of Classifier 2-Sample Tests (C2ST) on a held-out test set. Due to space constraints we report results in the Appendix in Figure 14. Marginal Flow achieves state-of-the-art results and proves to be particularly effective in low data regimes. Figure 6: Marginal Flow trained via log-likelihood on 2D synthetic datasets. We show 10'000 samples from the true distribution and from Marginal Flow.

![6_image_0.png](6_image_0.png)

## 4.3 Wishart Mixture Distribution

One interesting aspect of Marginal Flow is that the parametric family q(x|w) in Eq. 2 can be adjusted depending on the application and on the noise assumption. Consider the case of learning a Wishart mixture distributions (Haff et al., 2011; Cappozzo & Casa, 2025): observations consist of sample covariances, which lie on the cone of positive-definite (p.d.) matrices. One design choice would be to use a Gaussian assumption in q(x|w) and then transform the samples into positive definite matrices through bijective layers as in Negri et al. (2023). Alternatively, one could directly choose q(x|w) to be Wishart distributions. We showcase this second option, and, in particular, we parametrize the scale matrices of Wishart via wi, in addition to a parametrized global ν. We consider a target distribution t(x) where the generating parameters live on a 1D manifold:
t(x) = W(x; ν, Σ(λ)) s.t. Σ(λ) *∈ M ∀*λ ∈ [0, 1] . (4)
We showcase training using both the reverse and forward KL divergence (log-likelihood). Our goal is to approximate t(x) while reconstructing the manifold M. We showcase two settings. (i) A lowdimensional setting with 10 × 10 matrices using the reverse KL and we compare to Normalizing Flows (NFs) parameterizing the Cholesky factor. (ii) A high-dimensional setting with 100 × 100 matrices using the forward KL, which was computationally prohibitive for NFs. In Figure 9 we show test KL divergence in the low-dim setting and plot the manifold reconstruction using a PCA
projection to 2D. Marginal Flow perfectly recovers the manifold in both training directions and approximates t(x) better than NFs. For more details on the target manifold M see Appendix A.4.2.

![7_image_0.png](7_image_0.png) 

## 4.4 Manifolds In Image Latent-Spaces

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Most modern image generative models rely on non-trivial latent spaces, e.g. Rombach et al. (2022), which can still be relatively high-dimensional and show non-Euclidean behavior (Shao et al., 2018). It would then be relevant to traverse such latent spaces on a lower-dimensional manifold. Marginal Flow is well-suited for this task since it allows for learning a lower-dimensional manifold alongside the density. We showcase this on MNIST digits (LeCun et al., 1998) and the JAFFE face dataset (Lyons et al., 1998). The JAFFE dataset contains 214 face images of ten Japanese women mimicking certain emotions. Each image is associated with a score quantifying the emotions, e.g. "happiness" or "surprise". Note that learning a manifold with such little data is very challenging. In both settings, we first train a VAE without conditional information to encode images into a latent space (20- and 10-dimensional, respectively). Then, we train a single Marginal Flow in the latent space to learn a low-dimensional manifold conditioned on the digit label (or emotion score). The exact loss function is reported in the Appendix in Eq. 8. In particular, we use a 1-dim uniform base distribution pbase = U([−1, 1]). We learn conditional manifolds via the network fθ(z; c), with z ∈ [−1, 1] and c the class label (or scores). In Figure 10, we explore the 1-dim manifold conditioned on each label of **MNIST**. Results show similarities across digits in the learned manifold: some sections look approximately bold, *bold italic* and normal font, with smooth transitions in between them. For **JAFFE**, the manifold smoothly interpolates the different faces (horizontally) at fixed emotion levels, as shown in Figure 11. We observe disentanglement of faces and emotions, as faces tend to align within columns. Some inconsistencies are probably the result of the extremely low-data regime. For further visualizations, see the Appendix, Figure 15 and 16.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| Bold   | Bold italic   | Normal   |
|--------|---------------|----------|

Figure 10: Each row shows the 1-dim manifold conditioned on the label learned by Marginal Flow on MNIST (in a 20-dim VAE latent space). We observe disentanglement of digits and writing style.

![8_image_0.png](8_image_0.png)

## 5 Conclusions

In this work we introduced a flexible and efficient density estimation framework called Marginal Flow. We showed empirically that Marginal Flow is orders of magnitude faster than competing methods in terms of runtime, both at sampling and exact density evaluation. Unlike most density estimation models, Marginal Flow provides exact density evaluation by construction. Marginal Flow is also a very flexible framework: it allows for learning lower-dimensional manifolds, it can easily handle multi-modal distributions, and it can be easily tailored to the data with the choice of the parametrized distribution q(x|w). Experimentally, we showcase Marginal Flow on several datasets and various tasks. First, we showed that Marginal Flow can perfectly reconstruct synthetic datasets both when trained via log-likelihood and via reverse KL divergence. Additionally, Marginal Flow converges orders of magnitude faster than competing models. Then, we showed that it can achieve state-of-the-art results on the Simulation-based Inference benchmark. We also showed that we can easily adapt Marginal Flow to learn distributions on positive definite matrices by choosing the Wishart distribution as the parametrized family q(x|w). Lastly, we applied Marginal Flow to learn a (conditional) manifold alongside the density for MNIST digits and the JAFFE face dataset.

## Reproducibility

We made an effort to make every aspect of the model and of the experiments reproducible. In particular, as part of the submission we provide code with a PyTorch implementation of the model and code for reproducing figures and experiments. Furthermore, in Appendix A.1 we discuss implementation details of Marginal Flow concerning sampling, log density evaluation and neural network architecture. Finally, in Appendix A.3 we provide detailed description of the experiments conducted including data pre-processing for real-world experiments.

## References

Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. *Nature*, 630(8016):493–500, 2024.

Justin Alsing, Benjamin Wandelt, and Stephen Feeney. Massive optimal data compression and density estimation for scalable, likelihood-free inference in cosmology. Monthly Notices of the Royal Astronomical Society, 477(3):2874–2885, 03 2018. ISSN 0035-8711.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Jens Behrmann, Will Grathwohl, Ricky T. Q. Chen, David Duvenaud, and Joern-Henrik Jacobsen.

Invertible residual networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 573–582. PMLR, 09–15 Jun 2019.

Sam Bond-Taylor, Adam Leach, Yang Long, and Chris Willcocks. Deep generative modelling: A
comparative review of vaes, gans, normalizing flows, energy-based and autoregressive models.

IEEE Transactions on Pattern Analysis and Machine Intelligence, PP, 09 2021.

Johann Brehmer and Kyle Cranmer. Flows for simultaneous manifold learning and density estimation. *Advances in neural information processing systems*, 33:442–453, 2020.

Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519, 2015.

Andrea Cappozzo and Alessandro Casa. Model-based clustering for covariance matrices via penalized wishart mixture models. *Computational Statistics & Data Analysis*, pp. 108232, 2025.

Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In The Twelfth International Conference on Learning Representations, 2024.

Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference.

Proceedings of the National Academy of Sciences, 117(48):30055–30062, 2020. doi: 10.1073/
pnas.1912789117.

Felix Draxler, Peter Sorrenson, Lea Zimmermann, Armand Rousselot, and Ullrich Kothe. Free- ¨
form flows: Make any architecture a normalizing flow. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, pp. 2197–2205. PMLR, 02–04 May 2024.

Charles Fefferman, Sanjoy Mitter, and Hariharan Narayanan. Testing the manifold hypothesis.

Journal of the American Mathematical Society, 29(4):983–1049, 2016.

Mevlana C. Gemici, Danilo Rezende, and Shakir Mohamed. Normalizing flows on riemannian manifolds, 2016.

Pedro J Goncalves, Jan-Matthis Lueckmann, Michael Deistler, Marcel Nonnenmacher, Kaan Ocal, Giacomo Bassetto, Chaitanya Chintaluri, William F Podlaski, Sara A Haddad, Tim P Vogels, David S Greenberg, and Jakob H Macke. Training deep neural density estimators to identify mechanistic models of neural dynamics. *eLife*, pp. e56261, sep 2020. ISSN 2050-084X.

Ian J Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in neural information processing systems, 27, 2014.

Leonard R Haff, Peter T Kim, J-Y Koo, and D St P Richards. Minimax estimation for mixtures of wishart distributions. *The Annals of Statistics*, 2011.

Junxian He, Daniel Spokoyny, Graham Neubig, and Taylor Berg-Kirkpatrick. Lagging inference networks and posterior collapse in variational autoencoders. In ICLR, 2019.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

L Jeff Hong and Sandeep Juneja. Estimating the mean of a non-linear function of conditional expectation. In *Proceedings of the 2009 Winter Simulation Conference (WSC)*, pp. 1223–1236. IEEE, 2009.

Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In *International conference on machine learning*, pp. 448–456. pmlr, 2015.

Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4401–4410, 2019.

AmirEhsan Khorashadizadeh, Konik Kothari, Leonardo Salsi, Ali Aghababaei Harandi, Maarten de Hoop, and Ivan Dokmanic. Conditional injective flows for bayesian imaging. ´ IEEE Transactions on Computational Imaging, 9:224–237, 2023.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. 2014.

S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi. Optimization by simulated annealing. *Science*, 220
(4598):671–680, 1983. doi: 10.1126/science.220.4598.671.

Ivan Kobyzev, Simon JD Prince, and Marcus A Brubaker. Normalizing flows: An introduction and review of current methods. *IEEE transactions on pattern analysis and machine intelligence*, 43 (11):3964–3979, 2020.

Youssef Kossale, Mohammed Airaj, and Aziz Darouichi. Mode collapse in generative adversarial networks: An overview. In *ICOA*, pp. 1–6, 2022. doi: 10.1109/ICOA55659.2022.9934291.

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to ´
document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Huadong Liao and Jiawei He. Jacobian determinant of normalizing flows, 2021. Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023.

Haohe Liu, Zehua Chen, Yi Yuan, Xinhao Mei, Xubo Liu, Danilo Mandic, Wenwu Wang, and Mark D Plumbley. AudioLDM: Text-to-audio generation with latent diffusion models. In Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, pp. 21450–21474. PMLR, 23–29 Jul 2023.

Jan-Matthis Lueckmann, Jan Boelts, David Greenberg, Pedro Goncalves, and Jakob Macke. Benchmarking simulation-based inference. In *Proceedings of The 24th International Conference on* Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, pp. 343–351. PMLR, 13–15 Apr 2021.

M. Lyons, S. Akamatsu, M. Kamachi, and J. Gyoba. Coding facial expressions with gabor wavelets.

In Proceedings Third IEEE International Conference on Automatic Face and Gesture Recognition, pp. 200–205, 1998.

Charles A Micchelli, Yuesheng Xu, and Haizhang Zhang. Universal kernels. Journal of Machine Learning Research, 7(12), 2006.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Marcello Massimo Negri, Fabricio Arend Torres, and Volker Roth. Conditional matrix flows for gaussian graphical models. *Advances in Neural Information Processing Systems*, 36:25095– 25111, 2023.

Marcello Massimo Negri, Jonathan Aellen, and Volker Roth. Injective flows for star-like manifolds.

In *The Thirteenth International Conference on Learning Representations*, 2025.

George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Ed- ¨ ward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Bowen Tang, Yunjing Li, Michael Fang, Jing Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In *Advances in Neural Information Processing Systems 32*, pp. 8024–8035. Curran Associates, Inc., 2019.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*,
12:2825–2830, 2011.

Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pp. 1530–1538. PMLR, 2015.

Danilo Jimenez Rezende, Shakir Mohamed, and Daan Wierstra. Stochastic backpropagation and approximate inference in deep generative models. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of *Proceedings of Machine Learning Research*, pp. 1278–1286. PMLR, 2014.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Hang Shao, Abhishek Kumar, and P Thomas Fletcher. The riemannian geometry of deep generative models. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 315–323, 2018.

Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, Devi Parikh, Sonal Gupta, and Yaniv Taigman. Make-a-video: Text-to-video generation without text-video data. In The Eleventh International Conference on Learning Representations, 2023.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pp. 2256–2265. pmlr, 2015.

Yang Song and Diederik P. Kingma. How to train your energy-based models, 2021. Kevin Swersky, Marc'Aurelio Ranzato, David Buchman, Nando D Freitas, and Benjamin M Marlin. On autoencoders and score matching for energy based models. In *Proceedings of the 28th* international conference on machine learning (ICML-11), pp. 1201–1208, 2011.

Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. *Advances in neural information processing* systems, 33:7537–7547, 2020.

Alexander Tong, Nikolay Malkin, Kilian Fatras, Lazar Atanackovic, Yanlei Zhang, Guillaume Huguet, Guy Wolf, and Yoshua Bengio. Simulation-free schrodinger bridges via score and flow ¨ matching, 2024.

Xiangxiang Zeng, Fei Wang, Yuan Luo, Seung-gu Kang, Jian Tang, Felice C Lightstone, Evandro F
Fang, Wendy Cornell, Ruth Nussinov, and Feixiong Cheng. Deep generative molecular design reshapes drug discovery. *Cell Reports Medicine*, 3(12), 2022.
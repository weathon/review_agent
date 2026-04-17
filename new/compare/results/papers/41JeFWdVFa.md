000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Ldp: A Lightweight Denoising Plugin Enhanc- Ing Generalization In Single-Image Super- Resolution

Anonymous authors Paper under double-blind review

## Abstract

Current single-image super-resolution (SISR) models struggle to generalize to real-world degradations. To address this challenge, we propose LDP, an innovative lightweight denoising autoencoder (DAE) plug-in. It improves the generalization ability of SR models via low-resolution (LR) images prediction-based cyclic regularization. LDP models the SISR degradation process within the DAE framework. It leverages a property of diffusion models, where after noise is added, high-resolution (HR) images and LR features become aligned, so that denoising noisy HR features is equivalent to denoising noisy LR features. During the corruption process, noise is added independently to each HR patch. During the denoising process, a convolutional denoiser uses learned filters to approximate blur kernels. In addition, LR degradation is used to distinguish different LR from the same HR. LDP can be applied to SR models in two modes: as a training loss to improve reconstruction quality, or as an inference post-processing step to correct artifacts. Extensive experiments demonstrate that LDP substantially improves the generalization of existing SR models to unseen degradations.

![0_image_0.png](0_image_0.png) 

Figure 1: Our LDP is a lightweight denoising autoencoder-based plug-in that can be seamlessly integrated into arbitrary SR models, operating as a training-time loss or an inference-time module.

## 1 Introduction

Single Image Super-Resolution (SISR) aims to reconstruct high-resolution (HR) images from their low-resolution (LR) counterparts. SISR is widely applied in various fields, such as medical imaging Li et al. (2024a) and remote sensing Dong et al. (2024). Deep learning has advanced SISR
architectures from Convolutional Neural Network (CNN) Dong et al. (2014) to Transformer Liang et al. (2021); Chen et al. (2023b) and State-Space Model Guo et al. (2024; 2025), achieving higher reconstruction accuracy. Meanwhile, generative methods, including Generative Adversarial Network (GAN) Chen et al. (2022) and Diffusion Model Wang et al. (2024); Yue et al. (2025); Zhang et al. (2025), have been explored to improve perceptual quality.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Despite advances in SR architectures, existing models struggle to generalize to unseen degradations. Recent approaches leverage data augmentation and self-supervised learning techniques to tackle this challenge. Data augmentation approaches typically fall into two categories: generating synthetic distortions Zhang et al. (2021a); Wang et al. (2021), or employing generative models Li et al. (2022); Chen et al. (2025) to synthesize paired data from unpaired LR and HR images. However, these methods may harm performance Zhang et al. (2023) or are limited to in-distribution datasets. Selfsupervised approaches rely on either image-specific training Shocher et al. (2018); Ulyanov et al. (2018) or test-time adaptation Hussein et al. (2020); Zhou et al. (2023); Chen et al. (2024), utilizing internal image statistics and priors. However, they suffer from high computational cost or the need for model-specific adaptation. Addressing unseen degradations efficiently remains a key challenge. To address these limitations, we propose LDP, a lightweight denoising autoencoder (DAE) plug-in. It improves the generalization ability of SR models via LR prediction-based cyclic regularization. LDP models the SISR degradation process within the DAE framework. It leverages a property of diffusion models, where after noise is added, high-resolution (HR) images and LR features become aligned Wang et al. (2023b), making denoising noisy HR features equivalent to denoising noisy LR features. LDP takes high-resolution images (ground-truth HR or SR outputs) as input for degradation modeling, with LR high-frequency components as a condition to distinguish different LR images from the same HR. During the corruption process, LDP introduces patch-dependent Gaussian noise.

This enables the model to learn fine-grained degradation in local patches, rather than assuming the same degradation for the whole image. During the denoising process, a lightweight convolutional denoiser learns the blur kernels associated with the degradation model. Built on these designs, LDP accurately generates corresponding LR image and generalizes well to unseen degradations. LDP applies to SR models in two modes: as a training-time loss function to improve reconstruction quality, or as an inference-time post-processing step that corrects artifacts independently of training. Extensive experiments verify that LDP significantly improves the generalization ability of existing SR models on unknown complex degradations. Overall, our contributions are three-fold: - We propose LDP, an innovative lightweight denoising autoencoder plug-in for single-image superresolution that enhances the generalization of existing SR models.

- LDP is a conditional degradation model that generates LR images from HR inputs by explicitly conditioning on LR high-frequency components. LDP operates in two modes: as a degradationaware training-time loss function, or as an inference-time correction module (e.g., Posterior Sampling for diffusion models).

- LDP enhances reconstruction quality during training as a loss function and mitigates artifacts at inference independently of training. Both modes improve SR model generalization to unknown complex degradations.

## 2 Related Work 2.1 Improving Generalization In Sr

The limited generalization ability of SR models to unseen degradations remains a major challenge for real-world applications. Existing SR methods address this issue using two main approaches: data augmentation and self-supervised learning. Data augmentation methods seek to bridge the training–inference gap by creating synthetic data with degradations that approximate real-world scenarios. One line of works explicitly model degradations using predefined operations. BSR- GAN Zhang et al. (2021a) generates complex degradations by sequentially combining downsampling, blur, noise, and compression in random order, producing varied LR images for training. RealESRGAN Wang et al. (2021) introduces higher-order degradations to reflect real-world degradation chains. While BSRGAN and RealESRGAN enable non-blind SR models to handle blind scenarios through multi-degradation training, such strategies may compromise performance on indistribution benchmarks Zhang et al. (2023). Alternatively, implicit modeling methods leverage generative models to synthesize paired data from real LR and unpaired HR images. GAN Yuan et al. (2018); Li et al. (2022); Yin et al. (2023) or diffusion-based Chen et al. (2025) methods learn degradation priors to create realistic training pairs. However, their generalization remains limited to in-distribution data. Self-supervised learning enables SISR training using only LR images without paired HR supervision. ZSSR Shocher et al. (2018) and DIP Ulyanov et al. (2018) exploit internal patterns or implicit priors without external data. CorrectFilter Hussein et al. (2020); Zhou et al. (2023) aligns inputs with the training distribution of pre-trained models. Lway Chen et al. (2024) uses a degradation model to synthesize LR images from SR outputs for test-time fine-tuning. Although effective, these methods are computationally expensive or require model-specific adaptation.

## 2.2 Constraining The Sr Solution Space Via Degradation Modeling

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Degradation modeling, applied jointly with the SR model, introduces structural constraints that ensure reconstructed LR outputs align with the LR input, effectively narrowing the solution space to favor LR-consistent reconstructions. DRN Guo et al. (2020) adds a degradation branch that projects SR outputs back to the LR domain, enforcing reconstruction consistency and improving stability.

DualSR Emad et al. (2021) introduces a dual-path framework where a GAN-based downsampler and an upsampler are jointly trained with cycle consistency to model and reverse image-specific degradations. SCL-SASR Chen et al. (2023a) adopts a similar bidirectional design under MAP estimation, coupling SR and degradation networks to adapt to test-time degradations. Lway Chen et al. (2024) introduces test-time adaptation with pre-trained degradation models to fine-tune SR models, increasing generalization to unseen degradations. Despite their benefits, these methods face several limitations: DRN handles only bicubic downsampling; DualSR and SCL-SASR require image-specific optimization or joint training; and Lway introduces significant computational overhead due to its large model size. In contrast, our method supports a wide range of degradations through an explicitly modeled degradation process within a lightweight denoising autoencoder framework. Our degradation modeling framework is adaptable to various training settings, from large-scale supervised learning to image-specific fine-tuning, and can also be applied directly at test time. The framework is lightweight and does not incur significant computational cost. Degradation modeling is also applied during inference in diffusion-based image restoration to enforce LR consistency. ILVR Choi et al. (2021) guides the sampling process of DDPM Ho et al. (2020) using a reference image to maintain low-frequency consistency across the denoising steps. DR2 Wang et al. (2023b) shows that injecting additional Gaussian noise makes LR and HR distributions less distinguishable, allowing noise-corrupted LR images to be treated as noise-corrupted HR images during sampling. MCG Chung et al. (2022) ensures samples stay close to the data manifold by projecting the gradient of the measurement function onto its tangent space. DPS Chung et al. (2023) further leverages the degradation process to connect the LR observation to the predicted clean image at each step. In our method, LDP degrades each predicted clean image during diffusion inference, treating it as SR to produce a predicted LR image. We then enforce LR cyclic consistency by applying the tailored loss L
FT
sym (Eq. 16), which penalizes the discrepancy between the predicted LR and the ground-truth LR. This degradation-aware constraint enhances fidelity by suppressing artifacts and promoting structural consistency in the SR results.

## 3.1 Motivation 3 Proposed Method

Section 3.1 outlines the motivation behind LDP. Section 3.2 introduces the overall framework of LDP. Section 3.3 then details its training and inference modes, describing LDP's own training, its application in fine-tuning SR models, and its role as a post-processing step for diffusion models. To improve the generalization of existing SR models on unknown complex degradations, we adopt a degradation modeling approach applied jointly with the SR model. This introduces structural constraints that ensure the reconstructed LR outputs are aligned with the LR input, effectively narrowing the solution space to favor LR-consistent reconstructions. Our LDP integrates degradation modeling Yue et al. (2022) into the denoising autoencoder, reinterpreting denoising as a controllable degradation applied to HR images. In the classical degradation formulation, this can be expressed as:
y = ((x + n) ⊗ k) ↓s, (1)
where x ∈ R
H×W×3is the HR image, y ∈ R
H
s × W
s ×3is the LR image, n is the noise, k is the blur kernel, and s is the downsampling scale. We further leverage a property of diffusion models,

$$y=((x+n)\otimes k)\downarrow_{s},$$

162

![3_image_0.png](3_image_0.png) 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 whereby after noise is added, HR features and LR features become aligned Wang et al. (2023b), making denoising noisy HR features equivalent to denoising noisy LR features. This allows us to perform degradation modeling on HR images using a denoising autoencoder. However, there remains a challenge: since the SR task is inherently ill-posed, a condition is required to differentiate between different LR images generated from the same HR image under varying degradations. This condition must satisfy three criteria: (1) it cannot be the LR image itself, otherwise the network might take shortcuts and fail to learn meaningful degradations; (2) it must be discriminative for different LR images corresponding to the same HR image; and (3) it should be simple and easy to obtain. We define this condition as LRhf , obtained by subtracting the s
′-fold downsampled-thenupsampled LR image from the original LR image. In summary, we use a denoising autoencoder to perform degradation modeling on the input HR image, with the condition LRhf controlling the type of degradation in the output. During application, this approach constrains the super-resolution (SR) model to produce outputs whose LR reconstructions (via our LDP) are consistent with the original LR input, thus enforcing LR cyclic consistency and effectively guiding the SR model.

## 3.2 Framework

Figure 2 (a) illustrates the framework of our proposed LDP, which consists of four main modules: the Degradation Prediction Module (DPM), Noise Addition Module (NAM), Denoiser Module and Downsample Module. Designed as a denoising autoencoder, LDP functions as a conditional degradation model that generates LR images from HR inputs by conditioning on LR high-frequency components. To facilitate both implementation and interpretability, we adopt the noise corruption process from diffusion models Ho et al. (2020). The overall process of LDP is formulated as:

$$\begin{array}{l}{{x_{t}=N A M(x,t),}}\\ {{y^{\prime}=D(D e n o i s e r(x_{t}|D P M(y_{h f}),t)),}}\end{array}$$

Where y
′is the predicted LR images, and yhf is the LR high-frequency component. t is a patchdependent timestep, xt is the noised HR features, NAM(·) is the Noise Addition Module, DPM(·)
is the Degradation Prediction Module and D(·) is the Downsample Module.

$$(2)^{\frac{1}{2}}$$
$\eqref{eq:walpha}$. 
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 where ↓s
′ and ↑s
′ denote the downsampling and upsampling operations with scale factor s
′, respectively. To extract degradation information, we use prompts to encode degradation-specific details Potlapalli et al. (2023). First, a weight map w is derived from yhf , and then resized to match the spatial dimensions of x (i.e., H × W). This resized weight map is multiplied element-wise with the Degradation Prompt PD. It forms a degradation map C
′ ∈ R
H×W×C and serves as the condition for the denoiser. The process can be formulated as:

$$\begin{array}{l}{{w=(\mathrm{RB}_{4}\circ\mathrm{RB}_{3}\circ\downarrow_{2}\circ\mathrm{RB}_{2}\circ\mathrm{RB}_{1})\circ C o n v(y_{h f}),}}\\ {{C^{\prime}=P_{D}\otimes\mathrm{Resize}(w,H,W),}}\end{array}$$

where RB(·) denotes a residual block consisting of two 3 × 3 convolutional layers with a SiLU activation in between, *Conv*(·) represents a convolutional layer, ◦ denotes function composition applied sequentially from right to left, and ⊗ denotes element-wise multiplication. The downsampling operator ↓2 further reduces spatial resolution and disrupts local structures. The degradation prompt PD ∈ R
Np×C is jointly learned to encode degradation-specific information.

Noise Addition and Denoiser Module. Our framework integrates degradation modeling Yue et al. (2022) into the denoising autoencoder, reinterpreting denoising as a controllable degradation applied to HR images. During the corruption process, we perturb HR images using a patch-wise noise schedule. Specifically, following the diffusion noise schedule, each patch xi ∈ R
P ×P ×C is assigned a random timestep ti, and its noisy version is obtained as:

$$x_{i}^{(t_{i})}=\sqrt{\hat{\alpha}_{t_{i}}}\,x_{i}+\sqrt{1-\hat{\alpha}_{t_{i}}}\,\epsilon_{i},\quad\epsilon_{i}\sim\mathcal{N}(0,\mathbf{I}),$$
$$\left(7\right)$$

where αˆtidenotes the cumulative product of noise scheduling coefficients at time ti and ϵiis standard Gaussian noise. This patch-wise formulation enables each image region to undergo a different level of degradation, allowing the model to better capture spatially varying corruption. The final noisy image is denoted as xt. During the denoising process, a lightweight CNN acting as the denoiser module estimates the blur kernel and extracts intermediate feature F conditioned on the degradation map C
′. The feature F are then downsampled to produce the predicted LR image. Specifically, the denoiser module comprises L Condition Residual Blocks (CRBs) that leverage Adaptive Layer Normalization (AdaLN) Perez et al. (2018); Li et al. (2024b) for conditional modulation. For each P × P patch, the assigned timestep tiis embedded and combined with C
′to produce a patch-specific condition z. This condition is passed through a SiLU activation and a linear layer to generate modulation parameters α, β, and γ corresponding to scaling, bias, and gating. In the residual path, features are first normalized via LayerNorm and modulated by α and β, then processed by a residual block, gated with γ, and finally added back to the input. The CRB can be formulated as:

$$t_{emb}=TEmb(t),$$  $\alpha,\beta,\gamma=Linear(SiLU(C^{\prime}+t_{emb})),$  $x^{\prime}_{t}=\alpha\otimes(LN(F_{i-1}))+\beta,$  $F_{i}=\gamma\otimes RB(x^{\prime}_{t})+F_{i-1},$
$$({\mathfrak{s}})$$
$\eqref{eq:walpha}$. 
$$(10)^{\frac{1}{2}}$$
$\eqref{eq:walpha}$
′ + temb)), (9)
t) + Fi−1, (11)
where *T Emb*(·) is the timestep embedder, Fi−1 is the output of the previous CRB, and the initial feature is set as F0 = xt. The RB(·) in the final CRB is simplified to a single convolutional layer. Downsample Module. The module adjusts the feature map to match the spatial resolution of the
original LR image. Features F are first downsampled by a factor of s, then processed by a residual block and a convolutional layer:
$$y^{\prime}=C o m v(\mathrm{RB}(F\downarrow_{s})).$$
′ = *Conv*(RB(F ↓s)). (12)
Here, RB and the final convolutional layer are used to enhance feature representation and maintain smooth transitions between downsampled regions. Degradation Prediction Module. Figure 2 (b) shows the DPM diagram. Its input is the highfrequency component of the LR image, computed by subtracting the s'-fold downsampled-thenupsampled LR image from the original LR image, which can be formulated as:

$$y_{h f}=y-y\downarrow_{s^{\prime}}\uparrow_{s^{\prime}},$$
$$(4)$$

$$(5)$$
(6) $\frac{1}{2}$
yhf = y − y ↓s′ ↑s′ , (4)
$$(12)^{\frac{1}{2}}$$

## 3.3 Training And Inference Modes Of Ldp

4.1 IMPLEMENTATION DETAILS

## 4 Experiment

L
T
sym = λ1L1(M ⊗ y
′, M ⊗ y) + λ2L*LP IP S*(M ⊗ y
′, M ⊗ y), (13)
where λ1 and λ2 are the corresponding loss weights. Fine-Tuning SR Models with LDP. In fine-tuning, the original loss of pretrained SR models is augmented with a frequency loss Xie et al. (2023) that supervises the amplitude and phase components of SR and HR images in the frequency domain:

$$\begin{array}{c}{{{\mathcal L}_{f r e}=\frac{1}{H W}\sum_{u=0}^{H-1}\sum_{v=0}^{W-1}D({\mathcal F}(x^{\prime})(u,v),{\mathcal F}(x)(u,v)),}}\\ {{{\mathrm{}}}}\\ {{D({\mathcal F}(x^{\prime}),{\mathcal F}(x))=\left(\left({\mathcal R}\left({\mathcal F}(x^{\prime})\right)-{\mathcal R}\left({\mathcal F}(x)\right)\right)^{2}+\left({\mathcal I}\left({\mathcal F}(x^{\prime})\right)-{\mathcal I}\left({\mathcal F}(x)\right)\right)^{2}\right)^{\gamma/2},}}\end{array}$$
$$L P I P S(M\otimes y^{\prime},M\otimes y),$$
$$(13)^{\frac{1}{2}}$$
$$(14)$$
$$(15)$$
$$(17)^{\frac{1}{2}}$$

where x and x
′are the HR image and SR result, F(x) denotes the 2D Fourier transform of x, and R(·) and I(·) denote its real and imaginary parts. γ controls the sharpness of the frequency distance and is set to 1 by default. (*u, v*) indexes the frequency domain. In addition, LDP enforces cycle consistency by reconstructing the LR image from the SR output and minimizing a symmetric loss:
L
F T
sym = λ1L1(M′⊗y
′, M′⊗y)+λ2LLP IP S(M′⊗y
′, M′⊗y)+λ3L*f re*(M′⊗y
′, M′⊗y), (16)
where M′ = τ · M, τ scales the high-frequency weight map M by a scalar τ .

Diffusion Posterior Sampling with LDP. Our LDP can also be applied during inference in diffusion models via Diffusion Posterior Sampling (DPS) Chung et al. (2023), which uses the gradient of a data fidelity term to guide sampling and better align the results with the LR input:

$$\nabla_{\mathbf{x}_{t}}\log p_{t}(x_{t}|y)\simeq\mathbf{s}_{\theta^{*}}(x_{t},t)-\rho\nabla_{x_{t}}{\mathcal{L}}_{s y m}^{F T}(L D P({\hat{x_{0}}},y_{h f}),y),$$
$$\mathbf{\hat{r}}(x_{0},y_{h f}),y),$$
sym(LDP( ˆx0, yhf ), y), (17)
Training LDP. We train LDP on LSDIR Li et al. (2023) dataset using BSRGAN Zhang et al. (2021a) to synthesize diverse degradation datasets. For a scale factor of s = 4, the key hyperparameters are s
′ = 2, L = 3, P = 16, Np = 32, λ1 = λ2 = 1, and C = 64, resulting in 642k parameters. We use the Adam Kingma & Ba (2015) optimizer with β1 = 0.9 and β2 = 0.99, with a fixed learning rate of 0.001. The batch size is 12, with 256 × 256 HR patches. The timesteps ti are sampled from [500, 1000] to align the noisy HR and LR features. We adopt the diffusion batch multiplier Li et al. (2024b) with a value of 4 to perform multiple noise realizations per HR image. Training is conducted on a single NVIDIA RTX A6000 for 60K iterations, taking approximately 16 hours.

Fine-Tuning SR Models. We fine-tune existing SR models on the DF2K dataset (DIV2K Agustsson
& Timofte (2017) and Flickr2K Lim et al. (2017)) using BSRGAN degradation patterns, with our LDP employed as an auxiliary loss. Details are provided in the Appendix D. Testing. For synthetic testing, we generate five distinct datasets from the DIV2K validation set using bsrgan plus (BSRGAN Zhang et al. (2021a) and Real-ESRGAN Wang et al. (2021)), corresponding to the following degradation types: (1) downsampling, (2) noise, (3) blur, (4) JPEG compression, and (5) hybrid degradations following bsrgan plus defaults. For real-world testing, Training LDP. Following Lway Chen et al. (2024), LDP is trained by supervising only the highfrequency components of the predicted LR images. We apply the Discrete Wavelet Transform
(DWT) to decompose the predicted LR image y
′into four subbands (LL, LH, HL, HH). The highfrequency subbands (LH, HL, HH) are then summed and normalized to form a weight map M,
which is subsequently used to compute both the L1 loss and the LPIPS loss Zhang et al. (2018):
where sθ
∗ (xt, t) denotes the score function (the noise predictor in DDPM Ho et al. (2020)), and LDP(·) represents our LDP degradation model. xˆ0 denotes the predicted clean image at each time step, and we treat it as the SR output. In latent diffusion models, xˆ0 is first decoded into the pixel space before computing the gradient.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

| Methods Metrics                                                                                                                                                                                                                                                                                                                  | Down   | Noise   | Blur   | JPEG Hybrid   |       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|---------|--------|---------------|-------|
| PSNR↑                                                                                                                                                                                                                                                                                                                            | 32.05  | 27.25   | 26.38  | 29.65         | 27.03 |
| SSIM↑ 0.9539 0.7812 0.8273 0.9270 0.8098 LPIPS↓ 0.0794 0.2474 0.3207 0.0826 0.3360 PSNR↑ 19.58 18.77 19.36 18.57 19.36 SSIM↑ 0.4814 0.4712 0.4911 0.4612 0.4883 LPIPS↓ 0.1408 0.1399 0.1844 0.1492 0.2130 PSNR↑ 29.15 26.71 28.41 28.01 27.94 SSIM↑ 0.9283 0.8978 0.9159 0.9243 0.9173 LPIPS↓ 0.0985 0.1248 0.1417 0.0877 0.1025 |        |         |        |               |       |

![6_image_0.png](6_image_0.png)

| Methods Metrics   | Down   | Noise   | Blur   | JPEG Hybrid   |
|-------------------|--------|---------|--------|---------------|
| DRN DualSR LDP    |        |         |        |               |

we evaluate on RealSR Cai et al. (2019), RealSRSet Zhang et al. (2021b), and DPED Ignatov et al. (2017) datasets. We evaluate using PSNR, SSIM Wang et al. (2004), and LPIPS Zhang et al. (2018) as reference metrics, and NIQE Mittal et al. (2012), MANIQA Yang et al. (2022), CLIPIQA Wang et al. (2023a), MUSIQ Ke et al. (2021), and QAlign Wu et al. (2024) as non-reference metrics. For diffusion models, synthetic datasets are center-cropped to 512 × 512, and real-world datasets follow the StableSR Wang et al. (2024).

## 4.2 Effectiveness Of Ldp In Lr Prediction

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 To thoroughly evaluate the effectiveness of the proposed LDP, we conduct extensive experiments under five degradation scenarios and compare it with two existing degradation models, DRN Guo et al. (2020) and DualSR Emad et al. (2021). In this experiment, we first generate SR images using SwinIR Liang et al. (2021), and then apply the degradation models provided by LDP, DRN, and DualSR to obtain predicted LR images from the SR outputs. These predictions are compared with the LR inputs to the SR model, and the results are reported in Table 1. In addition, Table 2 reports the similarity between the LR images produced by each degradation model and the downsampled SR images. A higher similarity indicates that the degradation model collapses into trivial downsampling rather than applying the specific degradations implied by the input LR. As shown in the tables, LDP performs consistently well across all degradation types. Importantly, the similarity between the LDP-generated LR and the downsampled SR is significantly lower than that between the LDP-generated LR and the input LR, demonstrating that LDP does not degenerate into simple downsampling. In contrast, DRN behaves almost identically to bicubic downsampling: because its inputs include only HR (SR results) images without any conditional signals, it fails to map an SR
image to the multiple possible LR variants implied by different degradations. DualSR also struggles to properly handle diverse degradation types, particularly under complex mixed settings. As illustrated in **Fig.** 3, LDP effectively degrades high-frequency structures, further validating its ability to generate perceptually realistic LR images even under challenging degradations. In contrast, DRN and DualSR largely produce LR outputs that resemble simple downsampled versions of the SR images, indicating that they fail to apply the intended degradations.

Datasets Scale Metrics FeMaSR +LDP StableSR +LDP SwinIR +LDP MambaIR +LDP

Down

×4 PSNR↑ 24.22 **25.06** (+0.84) 20.35 **21.73** (+1.38) 25.44 **25.86** (+0.42) 26.58 **26.63** (+0.05) ×4 SSIM↑ 0.6793 **0.7105** (+0.0312) 0.4998 **0.5642** (+0.0644) 0.7210 **0.7242** (+0.0032) 0.7393 **0.7403** (+0.0010) ×4 LPIPS↓ 0.2637 **0.2490** (-0.0147) 0.3746 **0.2870** (-0.0876) 0.2579 **0.2538** (-0.0041) 0.2054 **0.2005** (-0.0049)

Noise

×4 PSNR↑ 22.82 **23.84** (+1.02) 19.95 **21.48** (+1.53) 24.34 **25.04** (+0.70) 26.11 **26.34** (+0.23) ×4 SSIM↑ 0.6519 **0.6957** (+0.0438) 0.4569 **0.5599** (+0.1030) 0.7130 **0.7198** (+0.0068) 0.7382 **0.7411** (+0.0029) ×4 LPIPS↓ 0.2788 **0.2624** (-0.0164) 0.4279 **0.3040** (-0.1239) 0.2676 **0.2659** (-0.0017) 0.2279 **0.2219** (-0.0060)

Blur

×4 PSNR↑ 24.12 **24.42** (+0.30) 19.98 **21.50** (+1.52) 24.03 **24.67** (+0.64) 24.99 **25.33** (+0.34) ×4 SSIM↑ 0.6639 **0.6787** (+0.0148) 0.4373 **0.5437** (+0.1064) 0.6764 **0.6833** (+0.0069) 0.6892 **0.6942** (+0.0050) ×4 LPIPS↓ **0.3168** 0.3199 (+0.0031) 0.5112 **0.4763** (-0.0349) 0.3197 **0.3168** (-0.0029) 0.2768 **0.2751** (-0.0017)

JPEG×4 PSNR↑ 22.92 **23.87** (+0.95) 20.17 **21.91** (+1.74) 24.55 **25.27** (+0.72) 26.36 **26.59** (+0.23)

×4 SSIM↑ 0.6696 **0.7068** (+0.0372) 0.5141 **0.5943** (+0.0802) 0.7301 **0.7372** (+0.0071) 0.7497 **0.7538** (+0.0041) ×4 LPIPS↓ 0.2633 **0.2508** (-0.0125) 0.3682 **0.2767** (-0.0915) 0.2535 **0.2506** (-0.0029) 0.2113 **0.2063** (-0.0050)

Hybrid

×4 PSNR↑ 23.40 **23.72** (+0.32) 19.27 **21.43** (+2.16) 23.52 **24.35** (+0.83) 24.35 **24.71** (+0.36) ×4 SSIM↑ 0.6211 **0.6392** (+0.0181) 0.3656 **0.5197** (+0.1541) 0.6458 **0.6492** (+0.0034) 0.6587 **0.6636** (+0.0049) ×4 LPIPS↓ **0.3453** 0.3516 (+0.0063) 0.5727 **0.4461** (-0.1266) 0.3634 **0.3571** (-0.0063) 0.3244 **0.3210** (-0.0034)

Figure 4: Qualitative results on synthetic datasets with ×4 scale factor. (**Zoom in for details**)

![7_image_0.png](7_image_0.png)

4.3 IMPROVING EXISTING SR MODELS VIA FINE-TUNING WITH LDP
378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Table 3: Performance improvements of blind SR models across diverse architectures using our proposed LDP on synthetic multi-degradation benchmarks. We generate synthetic benchmarks from the DIV2K validation set using five types of degradation: (1) Downsampling (Down), (2) Noise, (3) Blur, (4) JPEG, and (5) Hybrid degradations following bsrgan plus defaults.

We evaluate LDP on Blind SR models, including the GAN-based FeMaSR Chen et al. (2022),
Diffusion-based StableSR Wang et al. (2024), Transformer-based SwinIR Liang et al. (2021), and Mamba-based MambaIR Guo et al. (2024). In these experiments, LDP is applied only during the fine-tuning stage and is not used at inference. Improving SR Models on Synthetic Benchmarks. Quantitative and qualitative results are presented in Tab. 3 and Fig. 4 (Fig. 7 in **Appendix**). As listed in Tab. 3, incorporating LDP consistently improves all baseline models across all degradation types. Among them, MambaIR+LDP achieves the best overall performance. SwinIR and StableSR also benefit significantly from LDP. StableSR, in particular, shows substantial relative gains under challenging conditions such as Blur and Hybrid. These results highlight LDP's effectiveness in narrowing the solution space via cycle consistency, enabling stronger generalization to unknown degradations. Although FeMaSR+LDP outperforms the original model in most metrics, its LPIPS values in Blur and Hybrid remain higher. As shown in Fig. 4, LDP effectively reduces GAN artifacts and corrects texture distortions, significantly improving perceptual quality. The low LPIPS scores of the original FeMaSR are likely due to severe GAN artifacts misinterpreted as texture. Improving SR Models on Real-World Benchmarks. Quantitative and qualitative results are presented in Tab. 4 and Fig. 5 (Fig. 8 in **Appendix**). Table 4 shows that incorporating LDP consistently improves the performance of existing blind SR models across almost all datasets and metrics, demonstrating its enhanced generalization to unseen degradations. For FeMaSR, LDP suppresses GAN-induced artifacts, producing more stable, natural outputs. This can lower no-reference metrics, e.g., the CLIPIQA score drops on RealSR, as such metrics may favor visually striking but structurally inaccurate results. As shown in Fig. 5, the visual results explain the numerical improvements, with LDP mitigating ringing and GAN-induced artifacts, thereby enhancing visual fidelity and contributing to the better no-reference metrics scores.

## 4.4 Ldp For Posterior Sampling Of Pretrained Diffusion Models

We evaluated how LDP enhances pre-trained diffusion models through posterior sampling, including LDM Rombach et al. (2022), StableSR Wang et al. (2024), ResShift Yue et al. (2025), and UPSR Zhang et al. (2025). Quantitative and qualitative results are presented in Tab. 5 and 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Figure 5: Qualitative results on real-world benchmarks with ×4 scale factor. (**Zoom in for details**)
Fig. 6 (Fig. 9 in **Appendix**). As listed in Tab. 5, after applying LDP, the baselines show improvements across nearly all metrics on most datasets. For instance, StableSR demonstrates notable gains in MANIQA, CLIPIQA, and MUSIQ scores after applying LDP, while ResShift and UPSR also achieve higher metric values in most cases. For StableSR, we applied the noise-subtraction technique (Appendix E), which accounts for the differences from Tab. 4. As showed in Fig. 6, our LDP effectively reduces texture artifacts while preserving structural consistency.

## 5 Ablation Study

In ablation study, we examine the loss components, patch size, frequency band selection, scale factor for high-frequency acquisition, performance of LDP under severe degradations, and computational burden of LDP. Further details are provided in Appendix F. Ablation of Loss Terms in the Fine-Tuning Stage. Table 6 presents the impact of different loss components in L
F T
sym (Equ. 16) and L*f re* (Equ. 14) during fine-tuning of pretrained SwinIR models, evaluated on the synthetic Hybrid dataset. In all experiments, we set τ = 100 and the weight of each loss term is set to 1. All variants using any combination of the proposed losses outperform the baseline. Incorporating both symmetric and frequency losses (LDPV5–LDPV7) consistently improves perceptual quality (lower LPIPS) and reconstruction accuracy (higher PSNR and SSIM), with LDPV7 achieving the best overall performance, highlighting the complementary nature of these loss components. The LDP parameters can be universally configured as τ = 100 and λ1 = λ2 = λ3 = 1 for any super-resolution model, leading to improved generalization performance.

Ablation of the weight of tau. Table 7 presents the impact of different weight of tau when finetuning SwinIR. All values of tau outperform the baseline, with tau = 100 achieving the best overall performance.

## 6 Limitations And Conclusion

We propose LDP, a lightweight denoising autoencoder plug-in. By integrating HR images and the high-frequency component of LR, the model achieves realistic degradation modeling while maintaining efficiency. Experiments show LDP significantly improves the generalization of existing SR models on unseen degradations after fine-tuning, and enables test-time artifact correction. However, LDP has two main limitations: (1) in posterior sampling, it lacks generative ability and only per-

Datasets Scale Metrics FeMaSR +LDP StableSR +LDP SwinIR +LDP MambaIR +LDP

RealSR

×4 NIQE↓ **4.708** 5.533 (+0.825) 7.446 **6.331** (-1.115) **4.773** 4.838 (+0.065) **5.330** 5.350 (+0.020) ×4 MANIQA↑ 0.3430 **0.3654** (+0.0224) 0.3303 **0.3548** (+0.0245) 0.3510 **0.3742** (+0.0232) 0.2882 **0.3374** (+0.0492) ×4 CLIPIQA↑ **0.5645** 0.4482 (-0.1163) 0.4886 **0.5213** (+0.0327) 0.4739 **0.5478** (+0.0739) 0.3989 **0.4642** (+0.0653) ×4 MUSIQ↑ 58.94 **60.70** (+1.76) 52.99 **59.26** (+6.27) 59.67 **61.91** (+2.24) 51.87 **57.85** (+5.98) ×4 QAlign↑ 3.695 **3.860** (+0.165) 2.347 **2.646** (+0.299) 3.820 **3.877** (+0.057) 3.631 **3.766** (+0.135)

DPED

×4 NIQE↓ **5.045** 5.704 (+0.659) 7.616 **7.228** (-0.388) 4.982 **4.821** (-0.161) 5.983 **5.430** (-0.553) ×4 MANIQA↑ **0.3102** 0.2719 (-0.0383) **0.3056** 0.2970 (-0.0086) 0.2637 **0.2832** (+0.0195) 0.2334 **0.2767** (+0.0433) ×4 CLIPIQA↑ **0.5570** 0.3610 (-0.1960) **0.3968** 0.3843 (-0.0125) 0.3402 **0.4538** (+0.1136) 0.3083 **0.3850** (+0.0767) ×4 MUSIQ↑ **49.14** 44.07 (-5.07) 42.97 **45.08** (+2.11) 42.10 **45.91** (+3.81) 35.25 **44.64** (+9.39) ×4 QAlign↑ **3.429** 3.262 (-0.167) 2.033 **2.311** (+0.278) 2.988 **3.090** (+0.102) 3.192 **3.248** (+0.056)

RealSRSet

![8_image_0.png](8_image_0.png)

×4 NIQE↓ **5.236** 5.952 (+0.716) 6.090 **5.586** (-0.504) **5.424** 5.441 (+0.017) **5.726** 5.893 (+0.167) ×4 MANIQA↑ **0.4006** 0.4002 (-0.0004) 0.3904 **0.4012** (+0.0108) 0.3740 **0.3938** (+0.0198) 0.2978 **0.3555** (+0.0577) ×4 CLIPIQA↑ **0.6874** 0.5683 (-0.1191) 0.6057 **0.6214** (+0.0157) 0.5843 **0.6376** (+0.0533) 0.4793 **0.5428** (+0.0635) ×4 MUSIQ↑ **64.65** 64.07 (-0.58) 60.15 **62.84** (+2.69) 63.60 **65.33** (+1.73) 55.96 **61.28** (+5.32) ×4 QAlign↑ 3.776 **3.870** (+0.094) 2.916 **3.247** (+0.331) 2.749 **3.322** (+0.573) 3.434 **3.632** (+0.198)

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

Datasets Scale Metrics LDM +LDP StableSR +LDP ResShift +LDP UPSR +LDP

RealSR

×4 NIQE↓ **6.651** 6.830 (+0.179) 5.948 **5.636** (-0.312) **8.021** 8.027 (+0.006) 4.854 **4.834** (-0.020) ×4 MANIQA↑ **0.2904** 0.2810 (-0.0094) 0.3552 **0.3644** (+0.0092) **0.3487** 0.3486 (-0.0001) 0.3901 **0.3908** (+0.0007) ×4 CLIPIQA↑ **0.4564** 0.4319 (-0.0245) 0.4840 **0.5031** (+0.0191) 0.5353 **0.5354** (+0.0001) 0.5278 **0.5361** (+0.0083)

×4 MUSIQ↑ **52.09** 50.37 (-1.72) 55.11 **56.56** (+1.45) 56.85 56.85 **64.82** 64.70 (-0.12) ×4 QAlign↑ **2.685** 2.610 (-0.075) 2.607 **2.716** (+0.109) 3.036 3.036 3.218 **3.231** (+0.013)

DPED

×4 NIQE↓ **8.724** 8.770 (+0.046) 6.456 **6.267** (-0.189) 9.429 **9.415** (-0.014) **6.266** 6.281 (+0.015) ×4 MANIQA↑ 0.2381 **0.2418** (+0.0037) 0.3255 **0.3341** (+0.0086) **0.3107** 0.3104 (-0.0003) 0.3151 **0.3163** (+0.0012) ×4 CLIPIQA↑ **0.3718** 0.3681 (-0.0037) 0.4041 **0.4053** (+0.0012) 0.4875 **0.4879** (+0.0004) **0.4094** 0.4026 (-0.0068) ×4 MUSIQ↑ **32.92** 32.55 (-0.37) 45.55 **49.25** (+3.70) **44.63** 44.59 (-0.04) 46.47 **46.52** (+0.05) ×4 QAlign↑ 1.901 **1.917** (+0.016) 2.302 **2.343** (+0.041) 2.422 **2.423** (+0.001) **2.271** 2.257 (-0.014)

RealSRSet

![9_image_0.png](9_image_0.png)

×4 NIQE↓ 6.349 **6.258** (-0.091) 4.898 **4.687** (-0.211) **6.979** 7.011 (+0.032) **4.864** 4.878 (+0.014)

×4 MANIQA↑ 0.3407 **0.3470** (+0.0063) 0.4411 **0.4573** (+0.0162) 0.4004 0.4004 0.4647 **0.4720** (+0.0073)

×4 CLIPIQA↑ **0.5439** 0.5311 (-0.0128) 0.6384 **0.6584** (+0.0200) 0.6656 **0.6658** (+0.0002) 0.6709 **0.6753** (+0.0044) ×4 MUSIQ↑ 58.54 **59.52** (+0.98) 62.73 **62.96** (+0.23) 66.05 **66.06** (+0.01) 69.68 **69.74** (+0.06) ×4 QAlign↑ 3.046 **3.089** (+0.043) **3.193** 3.192 (-0.001) **3.561** 3.560 (-0.001) **3.705** 3.656 (-0.049)

Figure 6: Qualitative results of LDP enhances diffusion models through posterior sampling at ×4 scale SR. (**Zoom in for details**) Table 6: Ablation study of the loss terms used in the fine-tuning stage of pretrained SwinIR models.

Methods L

Sym

1 L

Sym

LP IP S L

Sym

f re L

SR

f re PSNR↑ SSIM↑ LPIPS↓

baseline *× × × ×* 23.52 0.6458 0.3634

LDPV1 *× × ×* ✓ 23.99 0.6481 0.3591

LDPV2 ✓ ✓ × × 24.08 0.6406 0.3585

LDPV3 × × ✓ × 24.01 0.6404 0.3582

LDPV4 ✓ ✓ ✓ × 24.13 0.6406 0.3609

LDPV5 ✓ ✓ × ✓ 24.33 0.6499 0.3578

LDPV6 × × ✓ ✓ 24.28 **0.6500** 0.3580

LDPV7 ✓ ✓ ✓ ✓ **24.35** 0.6492 **0.3571**

Table 7: Ablation study of the τ weight.

forms texture rectification; (2) It does not support unpaired degradation modeling, as the generated LR image inevitably retains information from the input LR high-frequency components.

## References

Eirikur Agustsson and Radu Timofte. NTIRE 2017 challenge on single image super-resolution: Dataset and study. In *IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pp. 1122–1131, 2017.

Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Cold diffusion: Inverting arbitrary image transforms without noise. In Advances in Neural Information Processing Systems, 2023.

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, and Lei Zhang. Toward real-world single image superresolution: A new benchmark and a new model. In *IEEE International Conference on Computer Vision*, pp. 3086–3095, 2019.

Chaofeng Chen, Xinyu Shi, Yipeng Qin, Xiaoming Li, Xiaoguang Han, Tao Yang, and Shihui Guo. Realworld blind super-resolution via feature matching with implicit high-resolution priors. In ACM International Conference on Multimedia, pp. 1329–1338, 2022.

Table 5: Improving Diffusion models via posterior sampling with LDP on real-world benchmarks.

10

| tau   | PSNR↑   | SSIM↑   | LPIPS↓   |
|-------|---------|---------|----------|
| -     | 23.52   | 0.6458  | 0.3634   |
| 0.1   | 24.15   | 0.6547  | 0.3601   |
| 1     | 24.27   | 0.6547  | 0.3595   |
| 10    | 24.30   | 0.6500  | 0.3596   |
| 100   | 24.35   | 0.6492  | 0.3571   |

Haoyu Chen, Wenbo Li, Jinjin Gu, Jingjing Ren, Haoze Sun, Xueyi Zou, Zhensong Zhang, Youliang Yan, and Lei Zhu. Low-res leads the way: Improving generalization for super-resolution by self-supervised learning. In *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 25857–25867, 2024.

Honggang Chen, Xiaohai He, Hong Yang, Yuanyuan Wu, Linbo Qing, and Ray E. Sheriff. Self-supervised cycle-consistent learning for scale-arbitrary real-world single image super-resolution. Expert Systems with Applications, 212:118657, 2023a.

Xiangyu Chen, Xintao Wang, Jiantao Zhou, Yu Qiao, and Chao Dong. Activating more pixels in image superresolution transformer. In *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 22367–22377, 2023b.

Yuying Chen, Mingde Yao, Wenbo Li, Renjing Pei, Jinjing Zhao, and Wenqi Ren. Unsupervised diffusionbased degradation modeling for real-world super-resolution. In *Proceedings of the AAAI Conference on* Artificial Intelligence, pp. 2348–2356, 2025.

Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. ILVR: conditioning method for denoising diffusion probabilistic models. In *IEEE International Conference on Computer Vision*, pp. 14347–14356, 2021.

Hyungjin Chung, Byeongsu Sim, Dohoon Ryu, and Jong Chul Ye. Improving diffusion models for inverse problems using manifold constraints. In *Advances in Neural Information Processing Systems*, 2022.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Hyungjin Chung, Jeongsol Kim, Michael Thompson McCann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In *International Conference on Learning* Representations, 2023.

Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. Learning a deep convolutional network for image super-resolution. In *European conference on computer vision*, pp. 184–199, 2014.

Runmin Dong, Shuai Yuan, Bin Luo, Mengxuan Chen, Jinxiao Zhang, Lixian Zhang, Weijia Li, Juepeng Zheng, and Haohuan Fu. Building bridges across spatial and temporal resolutions: Reference-based superresolution via change priors and conditional diffusion model. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 27674–27684, 2024.

Mohammad Emad, Maurice Peemen, and Henk Corporaal. DualSR: Zero-shot dual learning for real-world super-resolution. In *IEEE Winter Conference on Applications of Computer Vision*, pp. 1629–1638, 2021.

Hang Guo, Jinmin Li, Tao Dai, Zhihao Ouyang, Xudong Ren, and Shu-Tao Xia. MambaIR: A simple baseline for image restoration with state-space model. In *European conference on computer vision*, pp. 222–241, 2024.

Hang Guo, Yong Guo, Yaohua Zha, Yulun Zhang, Wenbo Li, Tao Dai, Shu-Tao Xia, and Yawei Li. MambaIRv2: Attentive state space restoration. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2025.

Yong Guo, Jian Chen, Jingdong Wang, Qi Chen, Jiezhang Cao, Zeshuai Deng, Yanwu Xu, and Mingkui Tan.

Closed-loop matters: Dual regression networks for single image super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 5406–5415, 2020.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In *Advances in Neural* Information Processing Systems, 2020.

Shady Abu Hussein, Tom Tirer, and Raja Giryes. Correction filter for single image super-resolution: Robustifying off-the-shelf deep super-resolvers. In *IEEE Conference on Computer Vision and Pattern Recognition*,
pp. 1425–1434, 2020.

Andrey Ignatov, Nikolay Kobyshev, Radu Timofte, Kenneth Vanhoey, and Luc Van Gool. DSLR-Quality photos on mobile devices with deep convolutional networks. In *IEEE International Conference on Computer Vision*, pp. 3297–3305, 2017.

Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. MUSIQ: multi-scale image quality transformer. In *IEEE/CVF International Conference on Computer Vision*, pp. 5128–5137, 2021.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Represent, 2015.

Guangyuan Li, Chen Rao, Juncheng Mo, Zhanjie Zhang, Wei Xing, and Lei Zhao. Rethinking diffusion model for multi-contrast MRI super-resolution. In *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 11365–11374, 2024a.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. In *Advances in Neural Information Processing Systems*, 2024b.

Xiaoming Li, Chaofeng Chen, Xianhui Lin, Wangmeng Zuo, and Lei Zhang. From face to natural image:
Learning real degradation for blind image super-resolution. In *European conference on computer vision*, pp. 376–392, 2022.

Yawei Li, Kai Zhang, Jingyun Liang, Jiezhang Cao, Ce Liu, Rui Gong, Yulun Zhang, Hao Tang, Yun Liu, Denis Demandolx, Rakesh Ranjan, Radu Timofte, and Luc Van Gool. LSDIR: A large scale dataset for image restoration. In *IEEE Conference on Computer Vision and Pattern Recognition Workshops*, pp. 1775– 1787, 2023.

Jingyun Liang, Jiezhang Cao, Guolei Sun, Kai Zhang, Luc Van Gool, and Radu Timofte. SwinIR: Image restoration using swin transformer. In *IEEE International Conference on Computer Vision*, pp. 1833–1844, 2021.

Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee. Enhanced deep residual networks for single image super-resolution. In IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 136–144, 2017.

Anish Mittal, Rajiv Soundararajan, and Alan C Bovik. Making a "completely blind" image quality analyzer.

IEEE Signal Processing Letters, 20(3):209–212, 2012.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron C. Courville. Film: Visual reasoning with a general conditioning layer. In *Proceedings of the AAAI Conference on Artificial Intelligence*, pp.

3942–3951, 2018.

Vaishnav Potlapalli, Syed Waqas Zamir, Salman H. Khan, and Fahad Shahbaz Khan. Promptir: Prompting for all-in-one image restoration. In *Advances in Neural Information Processing Systems*, 2023.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution im- ¨
age synthesis with latent diffusion models. In *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 10674–10685, 2022.

Assaf Shocher, Nadav Cohen, and Michal Irani. "Zero-Shot" super-resolution using deep internal learning. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 3118–3126, 2018.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In *International* Conference on Learning Representations, 2021.

Dmitry Ulyanov, Andrea Vedaldi, and Victor S. Lempitsky. Deep image prior. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 9446–9454, 2018.

Jianyi Wang, Kelvin C. K. Chan, and Chen Change Loy. Exploring CLIP for assessing the look and feel of images. In *AAAI Conference on Artificial Intelligence*, pp. 2555–2563, 2023a.

Jianyi Wang, Zongsheng Yue, Shangchen Zhou, Kelvin C. K. Chan, and Chen Change Loy. Exploiting diffusion prior for real-world image super-resolution. *International Journal of Computer vision*, 2024.

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind superresolution with pure synthetic data. In *International Conference on Computer Vision Workshops*, pp. 1905– 1914, 2021.

Zhixin Wang, Ziying Zhang, Xiaoyun Zhang, Huangjie Zheng, Mingyuan Zhou, Ya Zhang, and Yanfeng Wang.

DR2: diffusion-based robust degradation remover for blind face restoration. In IEEE Conference on Computer Vision and Pattern Recognition, pp. 1704–1713, 2023b.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4):600–612, 2004.

Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtai Zhai, and Weisi Lin. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. In *International Conference on Machine Learning*, 2024.

Jiahao Xie, Wei Li, Xiaohang Zhan, Ziwei Liu, Yew-Soon Ong, and Chen Change Loy. Masked frequency modeling for self-supervised visual pre-training. In *International Conference on Learning Representations*, 2023.
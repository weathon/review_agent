# Spherical Watermark: Encryption-Free, Loss- Less Watermarking For Diffusion Models

Xiaoxiao Hu1, Jiaqi Jin1, Sheng Li1, Wanli Peng2, Xinpeng Zhang1**, Zhenxing Qian**1,∗
1Fudan University, 2China Agricultural University
{xxhu23,jqjin24}@m.fudan.edu.cn {lisheng,zhangxinpeng,zxqian}@fudan.edu.cn wlpeng@cau.edu.cn

## Abstract

Diffusion models have revolutionized image synthesis but raise concerns around content provenance and authenticity. Digital watermarking offers a means of tracing generated media, yet traditional schemes often introduce distributional shifts and degrade visual quality. Recent lossless methods embed watermark bits directly into the latent Gaussian prior without modifying model weights, but still require per-image key storage or heavy cryptographic overhead. In this paper, we introduce Spherical Watermark, an encryption-free and lossless watermarking framework that integrates seamlessly with diffusion architectures. First, our binary embedding module mixes repeated watermark bits with random padding to form a highentropy code. Second, the spherical mapping module projects this code onto the unit sphere, applies an orthogonal rotation, and scales by a chi-square-distributed radius to recover exact multivariate Gaussian noise. We theoretically prove that the watermarked noise distribution preserves the target prior up to third-order moments, and empirically demonstrate that it is statistically indistinguishable from a standard multivariate normal distribution. Adopting Stable Diffusion, extensive experiments confirm that Spherical Watermark consistently preserves high visual fidelity while simultaneously improving traceability, computational efficiency, and robustness under attacks, thereby outperforming both lossy and lossless approaches.

## 1 Introduction

Diffusion models have demonstrated transformative potential in creative applications (Rombach et al.,
2022; Sahoo et al., 2024), but also raise concerns over authenticity and ownership (Craver et al., 1997; Grinbaum & Adomaitis, 2022). Malicious users can exploit them to fabricate images and spread disinformation, eroding public trust and creating legal and ethical challenges. As governments and platforms face mounting pressure to address harmful content (Biden, 2023; Wiggers, 2023), reliable provenance mechanisms are urgently needed to trace and identify malicious actors. Image watermarking offers a promising direction by embedding imperceptible identifiers into images. However, traditional schemes alter the data distribution and degrade visual fidelity, whether operating in the spatial (Li et al., 2009; Bender et al., 1995) or frequency (Al-Haj, 2007; Navas et al., 2008) domain. Additionally, some approaches inject watermarks by training or fine-tuning generative models. For example, Fernandez et al. (Fernandez et al., 2023) fine-tune the Stable Diffusion (Rombach et al., 2022) decoder to bake in a hidden mark. To avoid costly retraining and improve flexibility, Wen et al. (Wen et al., 2023) embed ring-patterns in the frequency domains of the latent space. Although robust to lossy transmission, these methods introduce perceptual artifacts and reduced fidelity. Recently, the concepts of lossless or undetectable watermarking have been proposed. These methods seek to establish an invertible mapping from watermark bits to standard Gaussian noise, embedding watermarks without any modifications to the pretrained generative model. For example, Yang et al. (Yang et al., 2024) introduce Gaussian Shading which uses repeated watermarks and stream cipher for sampling but demands a unique key and nonce per image, *incurring substantial storage*
∗Corresponding author.

1 and management overhead. Gunn et al. (Gunn et al., 2025) later replace the stream cipher with fixed-key pseudorandom error-correcting codes (Christ & Gunn, 2024). Nonetheless, the heavyweight cryptographic constructs also introduce drawbacks: they incur nontrivial computational and decoding latency, demand careful parameter tuning to balance code rate and error-correction capability, and fail under strong attacks that exceed the code's designed distortion bounds. In this paper, we propose Spherical Watermark, a simple yet effective lossless scheme that is encryption-free and robust against common attacks. Our method integrates seamlessly with pretrained diffusion models via three modules: binary embedding, spherical mapping, and diffusion integration. The binary embedding module mixes watermark bits with random paddings to produce a 3-wise independent bitstream. The spherical mapping module then projects this bitstream onto the unit sphere, applies an orthogonal rotation, and scales it by a chi-square-distributed radius. We theoretically analyze each intermediate distribution and prove that the final noise is statistically indistinguishable from standard Gaussian noise. In addition, our encryption-free design eliminates the need for perimage key storage. The diffusion integration module then feeds the watermarked noise into Stable Diffusion (Rombach et al., 2022) to produce watermarked images. Experiments show that our scheme preserves fidelity and surpasses lossy methods. Compared to lossless approaches (Gunn et al., 2025), our method offers stronger traceability, reduced complexity, and enhanced reliability.

In summary, our key contributions are three-folded: 1) We propose a novel lossless watermarking framework, which seamlessly integrates with diffusion-based architectures. Our method guarantees robust watermark extraction while preserving the original generation fidelity. 2) We introduce a simple yet effective mapping strategy that transforms binary watermarks into Gaussian noise inputs. We provide both theoretical analysis and empirical evidence that the watermarked noise distribution is statistically indistinguishable from a standard multivariate normal distribution. 3) Compared to existing lossless watermarking schemes, our encryption-free approach omits key storage overhead, enabling an excellent trade-off between undetectability and watermark robustness.

## 2 Related Works

Digital image watermarking has been extensively studied to safeguard intellectual property. Traditional watermarking methods can be applied directly to diffusion outputs, whether operating in the spatial domain (Li et al., 2009; Bender et al., 1995), the frequency domain (Navas et al., 2008; Liu et al., 2017; Kashyap & Sinha, 2012), or via neural-network embedding (Zhang et al., 2019; Zhu et al., 2018; Tancik et al., 2020). In addition, several works embed watermarks by fine-tuning diffusion models (Fernandez et al., 2023; Xiong et al., 2023; Kim et al., 2024; Wang et al., 2025). For example, SleeperMark (Wang et al., 2025) introduces a trigger mechanism to decouple watermark information from semantic content, keeping the watermark extractable after model fine-tuning. More recently, latent-based watermarking has gained attention. Wen et al. propose the Tree-Ring (Wen et al., 2023) watermarking scheme, which embeds ring-shaped patterns into frequency domains of the latent space to enable detection. Subsequent works such as RingID (Ci et al., 2024), SEAL (Arabi et al., 2025b), and WIND (Arabi et al., 2025a) design alternative patterns. Beyond pattern-based designs, Wei et al. (Wei et al., 2024) provide a unified analytical framework for diffusion watermarking and instantiate several distribution-preserving schemes, including truncated Gaussian sampling and Gaussian ring watermarking. However, these methods are limited to merely verifying the presence of watermark, not supporting large-scale provenance. To overcome this limitation, Yang et al. (Yang et al., 2024) introduce Gaussian Shading, a provably lossless watermarking method that employs repetition codes and a stream cipher to sample from the standard Gaussian distribution. However, the reliance on a distinct cipher key and nonce for each generated image imposes a huge key-management overhead that is impractical in the real world. Gunn et al. (Gunn et al., 2025) advocate replacing the stream cipher with the pseudorandom error-correcting codes (PRC) (Christ & Gunn, 2024), which allow the generation of distinct pseudorandom sequences from a fixed secret key. PRC's extensive cryptographic operations also introduce several challenges. Encoding and belief-propagation decoding (Pearl, 2014) incur substantial computation and latency. Finding a trade-off between code rate and error-correction strength requires careful tuning. Moreover, under aggressive post-processing or shifts in the data distribution, the scheme can hit an irreducible error floor and fail to recover the watermark. In this paper, we introduce Spherical Watermark, a framework that eliminates per-image key management, ensures lossless watermark embedding, and demonstrates superior robustness with high computational efficiency.

![2_image_0.png](2_image_0.png)

## 3 Method

As illustrated in Figure 1(a), our method constructs a tracing mechanism from the model developer's perspective. In the offline build phase, the model developer generates a fixed "Signature", a set of invertible transforms that encode distinct binary watermarks into the diffusion model's Gaussian noise input. During the online runtime phase, API-driven image request automatically applies the same signature to embed a user-related watermark into the latent code before it is passed through the diffusion model, ensuring that synthesized images carry traceable provenance. Finally, the developer inverts generated images to extract watermarks for reliable provenance tracking.

## 3.1 Problem Formulation

The secret watermark m encodes API metadata (e.g., user ID, timestamp). Let G : R
lx → I denote a fixed, pretrained diffusion generator that maps standard Gaussian noise z to a generated image O. Since diffusion models admit an approximate inverse mapping, we use G
−1to recover the latent representation from a generated image. Assume the watermark length is lm. Our goal is to design two efficient procedures in the latent space:

$${\mathrm{Embed}}:\mathbf{m}\in\{0,1\}^{l_{m}}\ \to\ \mathbf{z}_{w}\in\mathbb{R}^{l_{x}},$$
$$\text{Extract}:\hat{\mathbf{z}}_{w}\in\mathbb{R}^{l_{x}}\ \rightarrow\ \hat{\mathbf{m}}\in\{0,1\}^{l_{m}}.\tag{1}$$

Specifically, Embed takes m to produce the watermarked latent zw = Embed(m), and Extract predicts mˆ from the inverted latent zˆw = G
−1(Ow), where lx denotes the latent dimensionality of zw

and Ow is the generated image with tracable watermark. Let Pr-·denotes probability, and negl(ρ)
is a function that vanishes faster than any inverse polynomial in the security parameter ρ. We require:
Undetectability (Losslessness). For any probabilistic polynomial-time adversary A,
Pr[A(zw) = 1] − Pr[A(z) = 1] ≤ negl(ρ). (2)
In other words, watermarked noise zw is computationally indistinguishable from standard Gaussian noise z. Thus, for any polynomial-time adversary A′, the generated images remain indistinguishable:
$$\left|\mathrm{Pr}\big[A^{\prime}({\mathcal{G}}(\mathbf{z}_{w}))=1\big]~-~\mathrm{Pr}\big[A^{\prime}({\mathcal{G}}(\mathbf{z}))=1\big]\right|~\leq~\mathrm{negl}(\rho).$$
′(G(z)) = 1 ≤ negl(ρ). (3)
Traceability (Exact Extraction). There exists an Extract such that, given watermarked image Ow,
$$\left|\mathrm{Pr}[A(\mathbf{z}_{w})=1]\ -\ \mathrm{Pr}[A(\mathbf{z})=1]\right|\ \leq\ \mathrm{negl}(\rho).$$
$$\mathrm{Pr}\big[\mathrm{Extract}({\mathcal{G}}^{-1}(\mathbf{O}_{w}))=\mathbf{m}\big]\ \geq\ 1-\mathrm{negl}(\rho).$$
−1(Ow)) = m≥ 1 − negl(ρ). (4)
$\text{ability}(\text{Exact E})$  . 
That is, the recovered watermark matches the original except with only negligible error in ρ. For watermarking generated samples, losslessness is the central design principle. It preserves visual fidelity and underpins robustness in adversarial settings. We formally justify this in Appendix E and provide empirical evidence in Section 4.2, showing that lossy watermarking can be easily broken by adversarial attacks, whereas lossless watermarking remains unaffected.

$$(2)^{\frac{1}{2}}$$

$$({\mathfrak{I}})$$

$$\mathbf{U}_{\mathrm{{uc}}},$$
$$(4)$$

## 3.2 Methodological Design

Watermark Preprocessing. We represent watermark m as independent Bernoulli( 1 2
) bits. To enhance randomness and error correction, we repeat m across N blocks and append a padding vector r ∈ {0, 1}
lr, drawn i.i.d. from a Bernoulli( 1 2
) distribution on each invocation. The resulting vector

$$\mathbf{x}=\begin{bmatrix}\mathbf{m}&\mathbf{m}&\cdots&\mathbf{m}&\mathbf{r}\end{bmatrix}^{\top}\in\{0,1\}^{l_{x}},l_{x}=N\times l_{m}+l_{r},$$
lx, lx = N × lm + lr, (5)
serves as the sole input to the subsequent transforms.

Build Phase. In the build phase, the model developer constructs the *Signature* K =T, C	. To reduce the correlation introduced by repeating m, we inject randomness from the padding vector r. Accordingly, the embedding matrix T ∈ {0, 1}
lx×lx is designed to mix watermark bits with random paddings while remaining invertible. The rotation matrix C, also invertible, then maps the binary sequence into Gaussian-like noise. K is kept fixed and secret during runtime to prevent unauthorized removal. The embedding matrix T is constructed from the identity matrices IlNm and Ilr of sizes lNm and lr, together with a sparse binary matrix R ∈ {0, 1}
lNm×lr generated by Algorithm 1:

$$(S)$$

$$\mathbf{T}=\begin{bmatrix}\mathbf{I}_{l_{N m}}&\mathbf{R}\\ \mathbf{0}&\mathbf{I}_{l_{r}}\end{bmatrix},l_{N m}=N\times l_{m}.$$
$$(6)^{\frac{1}{2}}$$

, lNm = N × lm. (6)
The core design lies in R, which injects randomness from the padding vector into the watermark bits. Two parameters govern this construction. The row sparsity s specifies how many random paddings each watermark bit is mixed with: a larger s improves indistinguishability at the cost of amplified error propagation (see Section 4.3). In addition, redundancy is introduced through N repetitions, which enable majority vote decoding. Algorithm 1 ensures that the N copies of each bit are mixed with disjoint subsets of paddings, guaranteeing the independence property proved in Theorem 3.1. Algorithm 1 Construction of Binary Matrix R
Require: Positive integers N, lm, lr, s such that lr ≥ N × s Ensure: Binary matrix R ∈ {0, 1}
lNm×lr, Indices Set P
1: **Initialize** R ← 0 N×lm×lr 2: P ← ∅
3: for j = 1 to lm do 4: π ← RandomPermutation([1, 2*, . . . , l*r])
5: TMP ← π[1 : N × s] 6: for i = 1 to N do 7: G ← TMP[(i − 1) × s + 1 : i × s] 8: R[*i, j, G*] ← 1 9: P ← P ∪ {(*i, j, G*)}
10: **end for** 11: **end for**
12: **Return** Reshape(R,(lNm, lr)), P
By design, T is bijective over the binary field F2 and its inverse T−1follows that T−1 = T. And the rotation matrix C ∈ R
lC ×lC is orthogonal, so its inverse satisfies C−1 = CT. We obtain C by drawing a matrix lC × lC with i.i.d. N (0, 1) and then applying a QR decomposition, retaining the orthogonal factor. C maps the binary sequence into a continuous noise compatible with the latent input of diffusion models. For notational convenience, we set lC = lx in the following descriptions1.

Runtime Phase. Latent-based diffusion models adopt the encoder and decoder of VAE (Kingma & Welling, 2014) to construct bidirectional mappings between the latent and pixel space.

EVAE : I → R
lx, DVAE : R
lx → I, (7)
denote the pretrained VAE encoder and decoder, respectively. Let zT be standard Gaussian noise in latent space, and let z0 = EVAE(O) denote the clean latent encoding of an image O. To transform zT into z0, the diffusion model iteratively perform denoising steps over T discrete timesteps:
1In practice, lC is chosen as a factor of lx (e.g. lC = ⌊
√lx⌋) to balance rotational expressiveness with computational and storage efficiency.

$${\mathsf{T}}_{\mathrm{t}}$$

zT → zT −1 *→ · · · →* z0. At each diffusion timestep, the marginal distribution of zt is governed
by the probability-flow ordinary differential equation (ODE) (Song et al., 2021b):
$${\frac{d\mathbf{z}_{t}}{d t}}=f_{t}(\mathbf{z}_{t})\ -\ {\frac{1}{2}}\,g_{t}^{2}\,\nabla_{\mathbf{z}_{t}}\log p_{t}(\mathbf{z}_{t}),$$
t ∇ztlog pt(zt), (8)
where ft and gt are drift and diffusion coefficients determined by the pre-defined noising schedule.

The score function ∇ztlog pt(zt) is approximated by a neural network sθ(zt, t). We now describe how watermark embedding and extraction are seamlessly integrated into the Stable Diffusion pipeline. Our approach decomposes into three reversible modules: Binary Embedding Module B, Spherical Mapping Module S, and Diffusion Integration Module G. As illustrated in Figure 1(b), for watermarked image generation, we first construct the preprocessed input x by repeating m and appending random padding r. Then binary embedding module B performs the matrix multiplication

$$\mathbf{z}^{(1)}\;=\;\mathbf{T}\,\mathbf{x}$$  in $\mathbb{F}_{2}$. Next, spherical mapping module $\mathcal{S}$ converts $\mathbf{z}^{(1)}\in\{0,1\}^{l_{x}}$ into Gaussian noise by  $$\mathbf{v}=2\,\mathbf{z}^{(1)}-1,\mathbf{z}^{(2)}=\frac{\mathbf{v}}{\|\mathbf{v}\|_{2}},\mathbf{z}^{(3)}=\mathbf{C}\,\mathbf{z}^{(2)},$$  $$\text{draw}r\text{such that}r^{2}\sim\chi^{2}(l_{x}),\mathbf{z}_{w}=r\,\mathbf{z}^{(3)}.$$  Here, $\|\cdot\|_{2}$ denotes the Euclidean norm, and $\chi^{2}(l_{x})$ is the chi-square distribution with $l_{x}$ de 
$$\mathbf{z}^{(1)}\ =\ \mathbf{T}\,\mathbf{x}$$
$$({\boldsymbol{\delta}})$$
$$(9)$$
$$(10)$$

of freedom. The diffusion integration module G then generates the watermarked image. We set the initial noise zT = zw, and by solving Eq. 8 from t = T to t = 0, recover the clean latent z0 from zT ,
z0 = ODESolvezT ; sθ, cond*, T,* 0. (11)
Here, cond denotes sampling conditions (e.g. text prompts), and ODESolve may be instantiated with different solvers such as DDIM (Song et al., 2021a), DPM-Solver (Lu et al., 2022; 2025),
or other ODE integrators. z0 is then passed through DVAE to produce the watermarked image Ow = DVAEz0.

For watermark extraction, the developer applies the inverse modules in the order G
−1, S
−1, B
−1 on a suspect image Oˆ w. Specifically, the developer uses EVAE to estimate the latent zˆ0 = EVAE(Oˆ w),
and then solves Eq. 8 from t = 0 to t = T to obtain an estimate of the initial noise:
zˆT = ODESolvezˆ0; sθ, ∅, 0, T. (12)
Here, ∅ denotes the empty condition (no text prompt). Finally, the developer inverts zˆT as,

$$(12)$$
$${\hat{\mathbf{z}}}_{T}=\mathrm{ODESolve}\big({\hat{\mathbf{z}}}_{0};\,s_{\theta},\,\varnothing,\,0,\,T\big).$$

$$\Gamma\ \mathrm{as}$$
$$(13)^{\frac{1}{2}}$$

zˆ
(2) = C−1zˆT , zˆ
(1) = roundzˆ
(2)+1 2, xˆ = T
−1zˆ
(1), (13)
where round(·) refers to the rounding operation. The first lNm entries of xˆ correspond to N repeated copies of the watermark message. We therefore apply a majority-vote rule across each group of N bits to obtain the final decoded watermark mˆ . To avoid ties, N is chosen to be odd. Our embedding and extraction pipeline guarantees high-precision watermark retrieval for reliable provenance tracking.

## 3.3 Theoretical Analysis

In this section, we provide theoretical guarantees that, after the successive mappings x → z
(1) →
z
(2) → z
(3) → zw, the final latent code zw is distributed as N (0, Ilx) in R
lx . The detailed proofs of all lemmas and theorems stated in this section are provided in the Appendix C. First, we analyze the distribution of z
(1) in Theorem 3.1. By introducing r and carefully designing T,
we ensure that the resulting high-entropy code z
(1) exhibits strong independence properties.

Theorem 3.1. If m and r *consist of independent* Bernoulli( 12
) *bits, then for* z
(1) *in Eq. 9, we have* z
(1)
i ∼ Bernoulli12 for every i ∈ {1, . . . , lx}*, and* z
(1) *is both 2-wise and 3-wise independent.*
Building on the properties established in Theorem 3.1, we show that z
(2) satisfies the conditions of a spherical 3–design. A spherical t-design (Bannai, 1979; Bajnok, 1992) is a finite set of points on the unit sphere that, *up to degree* t, exactly matches the averages of all real polynomials with those of the continuous uniform distribution. Consequently, it can be regarded as an *approximate* uniform distribution on the unit sphere. The rigorous mathematical definition of a spherical t–design is as, Definition 3.1 (Spherical t-Design). A finite set of points X = {x1, . . . , xN } ⊂ S
n−1 on the unit sphere in R
n *is called a* spherical t-design if, for every real polynomial f *of total degree at most* t,

$${\frac{1}{N}}\sum_{x\in X}f(x)\;=\;{\frac{1}{|S^{n-1}|}}\int_{S^{n-1}}f(x)\,d\sigma(x),$$

where dσ *is the uniform surface measure on* S
n−1*, and* |S
n−1| denotes the total surface area of the unit (n − 1)*-sphere.* Equivalently, X is a t-design if and only if it *matches all moments* of the uniform distribution on the sphere up to degree t. Consequently, a spherical t-design is indistinguishable from the uniform distribution on S
n−1 by any statistic of degree ≤ t, and thus may be viewed as an *approximation* to the uniform spherical distribution. We derive that the set of z
(2) is a spherical 3-design in Theorem 3.2.

Theorem 3.2. z
(2) *satisfies that each* z
(2)
i*takes values* ± √
1 lx with Pr[zi = + √
1 lx
] = Pr[zi =
− √
1 lx
] = 12
, i ∈ (1, · · · , lx); z
(2) *is 2-wise and 3-wise independent. Then the finite set of* z
(2) on the unit sphere S
lx−1*is a spherical 3–design.*
Finally, the following two lemmas analyze the distributions of z
(3) and zw. In Lemma 3.3 we show that the orthogonally rotated vector z
(3) remains uniformly distributed on S
lx−1. In Lemma 3.4 we prove that scaling by r ∼ χ(lx) yields zw = r z
(3) ≈ N (0, Ilx) in R
lx . The detailed proofs are given in the Appendix C, and our experiments confirm that the empirical distribution of zw is statistically indistinguishable from standard Gaussian distribution in Section 4.2. Lemma 3.3. Let z
(2) ∈ S
lx−1 be a spherical 3*–design. If we apply a fixed orthogonal rotation* z
(3) = C z(2)*, then* z
(3) is also a spherical 3*–design. For each coordinate* z
(3)
i*, one has* E[zi] = 0 and E[z 2 i
] = 1/lx, and as lx → ∞*, the marginal law of* z
(3)
i*converges to* N (0, 1/lx).

Lemma 3.4. Let n ∼ N (0, In) *be a standard multivariate normal vector in* R
n. Then n admits a polar decomposition of the form n = r · u, where r 2 ∼ χ 2(n), and u *is uniformly distributed on the unit sphere* S
n−1. Furthermore, r and u are statistically independent. Conversely, if r 2 ∼ χ 2(n), u *is uniformly distributed on* S
n−1*, and* r ⊥ u, then the product n = r · u follows a standard multivariate normal distribution, i.e., n ∼ N (0, In).

## 4 Experiment

4.1 EXPERIMENTAL SETTINGS
Implementation Details. We adopt Stable Diffusion (SD) v1.52and v2.13as backbone generative models. Generated images are 512 × 512 color images with latent size 4 × 64 × 64. During the diffusion process, we use a 50-step DPM-Solver++ (Lu et al., 2025) for image generation with a guidance scale of 7.5 and a 50-step DDIM inversion (Song et al., 2021a) with a guidance scale of 1.0. To simulate real-world scenarios, DDIM inversion uses empty prompts. Default settings are N = 31, lm = 512, lr = 512, and s = 1, giving lNm = 15872 and lx = 16384, which matches the diffusion latent dimensionality. All experiments are conducted in PyTorch on four NVIDIA RTX 4090 GPUs.

Watermark baselines. We consider the following baselines: traditional watermarking methods include DwtDct (Al-Haj, 2007), DwtDctSvd (Navas et al., 2008), and RivaGAN (Zhang et al., 2019),
all configured to embed 32-bit watermarks. Latent-based baselines include Tree-Ring (Wen et al.,
2023), Gaussian Shading (Yang et al., 2024), and PRC Watermark (Gunn et al., 2025). All schemes are evaluated with 512-bit watermarks, except Tree-Ring, which supports detection only. For latentbased methods, we generate five fixed keys (or signatures) and report the mean and standard deviation of each metric over five independent runs. Unless noted otherwise, baselines use their default settings. Note that with fixed keys, Gaussian Shading no longer achieves true losslessness. Datasets & Evaluation metrics. For text prompts, we use two datasets, termed COCO and SDP. Each comprises 1000 text prompts randomly sampled from the MS-COCO val2017 set (Lin et al., 2014)
2https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 3https://huggingface.co/stabilityai/stable-diffusion-2-1-base Table 1: FID value for different watermarking methods. Lower FID indicates higher image quality.

MeanStd represents the mean value with 1-sigma error bar.

| Method           | COCO          | SDP           |               |               |
|------------------|---------------|---------------|---------------|---------------|
| SD v1.5          | SD v2.1       | SD v1.5       | SD v2.1       |               |
| Original         | 48.12561.3744 | 46.81461.0617 | 49.70410.5425 | 46.40600.5231 |
| DwtDct           | 48.29751.3918 | 46.97711.0702 | 49.98530.5385 | 46.73040.5163 |
| DwtDctSvd        | 48.71791.4075 | 47.40491.0121 | 51.01600.6162 | 47.50440.6439 |
| RivaGan          | 48.79561.3952 | 47.61241.1012 | 51.27730.6320 | 47.82980.6748 |
| Tree-Ring        | 49.33181.5108 | 47.87211.1320 | 50.64911.0197 | 47.39170.7127 |
| Gaussian Shading | 50.69681.3200 | 49.43791.0326 | 51.52210.8773 | 48.25390.4859 |
| PRC Watermark    | 48.13481.3074 | 46.75441.0748 | 49.52500.7651 | 46.41570.3445 |
| Ours             | 48.12241.5489 | 46.81321.0962 | 49.38940.7475 | 46.43110.3695 |

![6_image_0.png](6_image_0.png) 
Figure 2: Classification performance over training epochs for distinguishing watermarked from unwatermarked samples. Left Two: Training loss and test accuracy at latent-level. Right Two: Training loss and test accuracy at image-level on SDP dataset with SD v2.1.

![6_image_1.png](6_image_1.png)

and the Stable Diffusion Prompt dataset4, respectively. To evaluate the performance of our method, we focus on two core criteria: undetectability and tracing accuracy. For undetectability, we first assess any degradation introduced by watermark embedding. To detect subtle distributional shifts, we employ the Fréchet Inception Distance (FID) (Heusel et al., 2017) measured against the unwatermarked output distribution. We also train binary classifiers on both image-level pixels and latent-space inputs to distinguish watermarked from non-watermarked samples, reporting classification accuracy to reveal detectable artifacts introduced by the watermark embedding. Next, we evaluate the reliability of watermark extraction for 100 distinct users under common storage and transmission degradations, including post-processing attacks and adversarial attacks from WEvade (Jiang et al., 2023). Extraction performance is quantified by bit-level accuracy (ACC) and the true positive rate at 1% false positive rate (TPR@1%FPR). For simplicity, we abbreviate TPR@1%FPR as TPR in the sequel. We report mean and standard deviation over five runs for all metrics. Additional experimental results are provided in Appendix F, including further undetectability experiments and ablation studies.

## 4.2 Performance Analysis

Undetectability. To assess undetectability, we train classifiers to capture distributional shifts. First, we train a two-layer MLP (Rumelhart et al., 1986) for latent-level classification. According to Figure 2, both Tree-Ring and Gaussian Shading (with fixed keys) are easily detected with accuracies of 100% and 97%, while PRC Watermark and our method remain indistinguishable. Second, we 4https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts

| Method           | Metrics     |            |             |             |            |           |
|------------------|-------------|------------|-------------|-------------|------------|-----------|
| ACC (Clean)      | ACC (Post.) | ACC (Adv.) | TPR (Clean) | TPR (Post.) | TPR (Adv.) |           |
| DwtDct           | 90.141.15   | 64.751.08  | 49.280.00   | 92.803.14   | 52.233.41  | 16.150.02 |
| DwtDctSvd        | 100.000.00  | 93.210.17  | 48.950.01   | 100.000.00  | 91.940.68  | 17.050.02 |
| RivaGan          | 99.680.10   | 96.780.22  | 52.310.01   | 100.000.00  | 99.130.22  | 26.750.02 |
| Tree-Ring        | -           | -          | -           | 100.000.00  | 98.850.31  | 6.710.02  |
| Gaussian Shading | 100.000.00  | 98.430.04  | 88.060.11   | 100.000.00  | 99.970.04  | 99.230.00 |
| PRC Watermark    | 100.000.00  | 93.520.20  | 97.690.07   | 100.000.00  | 87.030.39  | 95.380.00 |
| Ours             | 99.990.01   | 95.020.08  | 98.120.04   | 100.000.00  | 97.500.18  | 99.830.00 |

sample one prompt and generate ten watermarked images per user across 100 distinct users for image-level evaluation, with qualitative examples shown in Figure 3. In Figure 2, we also train a ResNet-18 classifier (He et al., 2016) for image-level classification. Tree-Ring and Gaussian Shading are detectable, while PRC Watermark and ours show near-chance detection (50%). Table 1 shows that only PRC Watermark and our method match the original in FID, whereas other methods incur distribution shifts. These results support our theoretical analysis in Section 3.3 by showing that watermarked samples are statistically indistinguishable from unwatermarked ones.

![7_image_0.png](7_image_0.png) 
Computational Efficiency. To demonstrate the advantages of our encryption-free design, we evaluate the embedding and extraction times of latent-based watermarking schemes, with each result averaged over 100 trials. In this comparison, we focus exclusively on the transformation between the watermark and its latent noise representation, excluding any diffusion sampling or inversion procedures. As illustrated in Figure 4, we employ a logarithmic scale on the yaxis for visualization. The extraction time of the PRC Watermark is much higher than that of ours, roughly four orders of magnitude slower on extraction. This difference reflects the computational burden introduced by belief-propagation decoding in the PRC scheme. In contrast, our approach eliminates the need for complex key design, thereby enhancing execution speed, improving computational efficiency. Tracing Accuracy. In Table 2, we evaluate tracing accuracy under varied conditions. "Clean" refers to PNG storage, "Post-Processing" reports common post-processing distortions, and "Adversarial" refers to attacks from (Jiang et al., 2023) (See Appendix F.4 for details). Compared to lossy schemes, our method achieves comparable accuracy above 95% in both Clean and Post-Processing settings. We introduce a tunable parameter s, which entails a slight robustness trade-off relative to Gaussian Shading. Under Adversarial conditions, however, the accuracy of lossy schemes degrades sharply, as their embeddings enable effective classifiers to be trained for watermark detection, which can then be adversarially attacked. In contrast, lossless schemes demonstrate clear superiority: our method improves accuracy by more than 10%, consistent with the theoretical analysis in Appendix E.

Comparison with PRC Watermark. In Table 2 and Figure 5, we compare our method with PRC Watermark under varied distortions. Our method consistently achieves higher TPR and ACC, with a larger margin at stronger distortions. In addition, Figure 6(a) examines the effect of watermark capacity lm on tracing accuracy under JPEG–70 compression. As lm increases, PRC Watermark's decoding performance deteriorates rapidly and fails entirely beyond lm = 2000. In contrast, Spherical Watermark sustains high detection rates across the full capacity range. Furthermore, the computational efficiency comparisons show that our embedding and extraction incur significantly lower overhead than PRC Watermark, with extraction being about four orders of magnitude faster. These results confirm the superior robustness of our method.

![8_image_0.png](8_image_0.png) 

![8_image_1.png](8_image_1.png)

(b) Ablation on Modules. 

(c) Ablation on Modules.

(a) Ablation on lm.

## 4.3 Ablation Experiments

Ablation on Modules. In our ablation study, we isolate the effects of each module. In one variant, we omit the spherical mapping S and substitute the Gaussian Shading transform; in another, we skip the binary embedding B and apply only the spherical mapping to x. We then evaluate both latent-level undetectability and tracing accuracy. In Figure 6(b), omitting the binary embedding makes the latent noise trivially distinguishable. Figure 6(c) shows that robustness under brightness adjustment drops dramatically without spherical mapping. These results confirm that binary embedding enforces independence, while spherical mapping is essential for restoring robustness. A rigorous analysis of why our orthogonal rotation design achieves optimal robustness is provided in Appendix D. Ablation on Parameters. We further investigate the sensitivity of our method to hyperparameters: the watermark length lm, the padding length lr, the row sparsity parameter s, and the repetition count N. In Figure 6(d), we vary these parameters and train a latent-level classifier to evaluate their effect. The results show that classification accuracy remains near 50%, indicating that parameter changes do not impair undetectability. As s increases, each watermark bit depends on more paddings, making errors more likely to propagate and amplify. Similarly, reducing N decreases redundancy for majority-vote correction. Thus, both larger s and smaller N reduce accuracy by design, a trend also confirmed by the experimental results in Table 3. In addition, Figure 6(a) shows that *Spherical Watermark* maintains high detection rates across all watermark capacities under JPEG–70 compression.

Ablation on Diffusion Sampling Settings. We conduct ablation studies on the COCO dataset using the SD v2.1 model to assess the sensitivity of our method to diffusion sampling configurations. Table 4 compares watermark extraction accuracy under various attacks across three ODE solvers: DDIM (Song et al., 2021a), PNDM (Liu et al., 2022), and DPM-Solver++ (Lu et al., 2025). Settings of each attack type are provided in Appendix F.5. We then investigate the role of generation and

| Case   | sparsity parameter s   | repetition count N   |           |           |           |           |           |            |
|--------|------------------------|----------------------|-----------|-----------|-----------|-----------|-----------|------------|
| 1      | 2                      | 3                    | 4         | 1         | 11        | 21        | 31        |            |
| 1      | 100.000.00             | 99.840.12            | 99.080.25 | 97.580.34 | 99.400.14 | 99.960.05 | 99.940.05 | 100.000.00 |
| 2      | 99.940.05              | 99.460.26            | 96.640.42 | 92.380.74 | 98.080.32 | 99.860.14 | 99.920.12 | 99.940.05  |
| 3      | 99.720.12              | 98.000.21            | 93.120.53 | 83.680.71 | 95.100.45 | 99.500.24 | 99.680.19 | 99.720.12  |

Solver Post-processing Perturbations

PNG Brightness Gaussian Blur Median Filter JPEG Resize

DDIM 99.980.01 96.060.23 99.430.02 99.200.03 98.390.16 99.850.01 PNDM 99.980.01 96.170.23 99.400.02 99.150.03 98.410.15 99.840.01 DPM-Solver++ 99.980.01 96.020.26 99.440.01 99.210.03 98.400.15 99.850.01

inversion timesteps under PNG storage, as summarized in Table 5. Results show that neither the choice of ODE solver nor the variation in timestep schedules introduces meaningful degradation. The minor numerical discrepancies caused by switching solvers or adjusting timesteps are effectively absorbed by the inherent redundancy of our spherical mapping, which provides robustness against moderate inversion inaccuracies. Further quantitative analysis is provided in Appendix F.5.

## 5 Discussion And Limitations

Our Gaussian-noise guarantee depends on spherical 3-design definition. While watermarked and random noise are empirically indistinguishable, higher-order moments may deviate from the true prior. Extremely strong inversion-breaking attacks (e.g., perturbations targeting the VAE encoder or ODE solver) can still compromise recovery. We provide an extended analysis of our method against re-generation and editing attacks in Appendix F.2, showing that the proposed approach retains robustness in these scenarios. Nevertheless, our primary focus is on tracing the origin of maliciously generated content. Since editing and forgery may involve different adversarial, such cases are outside our scope. Overall, our method can generalize to any generative model with a Gaussian prior and invertible mappings, with detailed analysis in Appendix F.1.

## 6 Conclusion

In this paper, we introduce Spherical Watermark, a novel watermarking framework for image generation that requires no modifications to the diffusion model. Our key innovation is the binary embedding and spherical mapping module that converts binary watermark bits into Gaussian noise input. Watermarked latent inputs are provably and empirically indistinguishable from a standard Gaussian prior. Additionally, we eliminate per-image key management while delivering superior robustness under realistic distortions. Extensive experiments demonstrate that our method outperforms existing schemes in terms of undetectability, traceability, and computational efficiency.

| 10   | 20        | 30        | 40        | 50        |           |
|------|-----------|-----------|-----------|-----------|-----------|
| 10   | 99.850.88 | 99.920.71 | 99.950.57 | 99.950.58 | 99.950.60 |
| 20   | 99.960.52 | 99.970.48 | 99.980.35 | 99.980.36 | 99.980.38 |
| 30   | 99.970.28 | 99.990.18 | 99.990.16 | 99.990.12 | 99.990.12 |
| 40   | 99.970.56 | 99.980.52 | 99.980.48 | 99.980.47 | 99.980.48 |
| 50   | 99.970.36 | 99.980.36 | 99.980.29 | 99.990.29 | 99.990.26 |

## Acknowledgments

We sincerely thank the anonymous reviewers and area chairs for their constructive comments and suggestions. This work was supported by the National Natural Science Foundation of China (Grant 62572125) and the Natural Science Foundation of Shanghai (Grant 25ZR1401019).

## Ethics Statement

We have carefully reviewed and adhered to the ICLR Code of Ethics throughout the development of this work. We have ensured that our methodologies and findings are transparent, reproducible, and free from discrimination, bias, or unfairness concerns. Any potential ethical concerns have been carefully considered, and we encourage responsible use of our contributions in future research.

## Reproducibility Statement

We are committed to ensuring the reproducibility of our results. All theoretical claims are supported with complete proofs provided in Appendix C, Appendix D, and Appendix E. For empirical studies, we specify datasets, model architectures, hyperparameters, and implementation details in Section 4, with additional information in Appendix F. The source code is included in the supplementary material, with a README file that clearly documents the execution steps. All experiments are repeated five times, and we report the mean and standard deviation to mitigate randomness and measurement error.

## References

Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4432–4441, 2019.

Ali Al-Haj. Combined dwt-dct digital image watermarking. *Journal of computer science*, 3(9):
740–746, 2007.

Kasra Arabi, Benjamin Feuer, R. Teal Witter, Chinmay Hegde, and Niv Cohen. Hidden in the noise: Two-stage robust watermarking for images. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025a. URL https://openreview.net/forum?id=ll2nz6qwRG.

Kasra Arabi, R Teal Witter, Chinmay Hegde, and Niv Cohen. Seal: Semantic aware image watermarking. *arXiv preprint arXiv:2503.12172*, 2025b.

Bela Bajnok. Construction of spherical t-designs. *Geometriae Dedicata*, 43:167–179, 1992. Eiichi Bannai. On tight spherical designs. *Journal of Combinatorial Theory, Series A*, 26(1):38–47, 1979.

Eiichi Bannai and Etsuko Bannai. A survey on spherical designs and algebraic combinatorics on spheres. *European Journal of Combinatorics*, 30(6):1392–1425, 2009.

Walter R Bender, Daniel Gruhl, and Norishige Morimoto. Techniques for data hiding. In Storage and Retrieval for Image and Video Databases III, volume 2420, pp. 164–173. SPIE, 1995.

Joseph R Biden. Executive order on the safe, secure, and trustworthy development and use of artificial intelligence. 2023.

BlackForestLabs. Flux.1-dev. https://blackforestlabs.ai/, 2024.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In *Proceedings of the IEEE/CVF conference on computer vision and pattern* recognition, pp. 18392–18402, 2023.

Tu Bui, Shruti Agarwal, and John Collomosse. Trustmark: Robust watermarking and watermark removal for arbitrary resolution images. In *Proceedings of the IEEE/CVF International Conference* on Computer Vision, pp. 18629–18639, 2025.

Louis HY Chen and Qi-Man Shao. Normal approximation under local dependence. 2004.

Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. In Annual International Cryptology Conference, pp. 325–347. Springer, 2024.

Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. Ringid: Rethinking tree-ring watermarking for enhanced multi-key identification. In *European Conference on Computer Vision*, pp. 338–354. Springer, 2024.

Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Scott A Craver, Nasir D Memon, Boon-Lock Yeo, and Minerva M Yeung. Can invisible watermarks resolve rightful ownerships? In *Storage and Retrieval for Image and Video Databases V*, volume 3022, pp. 310–321. SPIE, 1997.

Ph Delsarte, JM Goethals, and JJ Seidel. Spherical codes and designs. *Geometriae Dedicata*, 6(3):
363–388, 1977.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA, pp. 248–255. IEEE
Computer Society, 2009. doi: 10.1109/CVPR.2009.5206848. URL https://doi.org/10. 1109/CVPR.2009.5206848.

Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In Marc'Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), *Advances in Neural Information Processing Systems 34: Annual Conference on* Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pp. 8780–8794, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.

Mucong Ding, Tahseen Rabbani, Bang An, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, et al. Waves: Benchmarking the robustness of image watermarks. In ICLR 2024 Workshop on Reliable and Responsible Foundation Models, 2024.

Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. *arXiv preprint arXiv:1410.8516*, 2014.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale.

In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id= YicbFdNTTy.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable signature: Rooting watermarks in latent diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22466–22477, 2023.

Alexei Grinbaum and Laurynas Adomaitis. The ethical need for watermarks in machine-generated language. *AI and Ethics*, 2022.
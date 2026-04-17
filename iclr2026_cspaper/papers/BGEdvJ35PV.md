000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Diffusion models have achieved state-of-the-art performance across diverse domains, yet their application to molecular generation remains challenging. Unlike many data types where values can tolerate slight variations, such as pixel intensities in images, molecules are governed by strict geometric and chemical constraints: minor variations in the atomic coordinates of even a single atom can lead to totally invalid or unstable molecules. These constraints give rise to highly concentrated data distributions, forming sharp probability peaks. Moreover, these peaks are *densely packed* in configuration space: changing one atom's type, along with small but precise adjustments to its position and that of its neighbors, can result in a distinct molecule, whereas images generally require much larger perturbations to change semantic meaning. This dense-concentrated structure makes diffusion modeling fragile: because valid regions are narrow and tightly clustered, even small deviations at intermediate timesteps can easily cross validity boundaries. Once entering the invalid regions, the generative process provides unreliable guidance, causing errors that accumulate over timesteps and drift generative trajectories off-distribution, ultimately leading to irreparable structural violations. To address this challenge, we formalize the notion of dense-concentrated structure in molecular distributions and analyze how discrepancies at intermediate steps propagate under reverse inference. Building on this insight, we propose **DIST**, a plug-in corrective method that DIffuses and STeers the intermediate distribution, thereby realigning inference trajectories toward a valid molecular distribution. Our method is model-agnostic and can be integrated into a wide range of existing diffusion models, achieving significant improvements in performance while reducing the computational cost to nearly half the standard number of timesteps.

## 1 Introduction

Generative models are probabilistic frameworks that aim to approximate an underlying data distribution and generate new samples from the learned distribution. By providing a principled approach to learning and sampling from complex, high-dimensional distributions, generative modeling has emerged as a promising paradigm with broad implications for design automation, simulation, and scientific discovery. Recently, diffusion models (DMs) (Ho et al., 2020; Song et al., 2021b) have become a prominent generative paradigm due to their outstanding performance in natural image synthesis and beyond (Song et al., 2020; Rombach et al., 2021; Watson et al., 2023). A DM consists of a forward process and a reverse process. In the forward process, data samples are gradually corrupted by a Markovian noise injection until they become indistinguishable from pure Gaussian noise. The reverse process is parameterized by a neural network, which is trained to approximate the time-reversed dynamics by iteratively denoising the corrupted states. At inference time, the model generates new samples by simulating this learned reverse trajectory, reconstructing structured data from pure noise. Recent work has extended DMs to 3D molecular generation (Hoogeboom et al.,
2022; Xu et al., 2023). However, molecular data presents unique challenges that make direct application of diffusion models less effective. Specifically, 3D molecules are represented by continuous 3D atomic coordinates together with discrete features such as atom types. Unlike images, where pixel intensities are only loosely correlated and can tolerate a wide range of variations, molecules are governed by strict geometric and chemical constraints, such that even small perturbations to atomic coordinates or atom types can result Anonymous authors Paper under double-blind review

## Abstract

# Diffuse And Steer: Corrective Sampling For Stable 3D Molecular Generation

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 in completely invalid or unstable structures (Choi et al., 2025). These constraints result in highly concentrated data distributions with narrow probability peaks, where each peak represents a valid and stable molecular configuration. Even slight displacements can shift the molecular configuration off-peaks into regions of negligible probability, corresponding to invalid or unstable states (Reymond et al., 2012; Martin & Cao, 2015; Bohde et al., 2025). Moreover, these peaks are densely packed but clearly separated: changes in one atom's type, along with small (densely packed) but precise adjustments (well separated) to its position and that of its neighbors, can result in a distinct molecule. **Overall, the molecular distribution exhibits an evident dense and concentrated** structure, where each probability peak corresponds to a chemically valid molecule, and the regions between the peaks are of near-zero density. We provide an illustrative analogy to compare the distribution and diffusion process of images with those of molecules in Fig. 1, to highlight the consequences of such a dense and concentrated structure to the diffusion process. Notably, such denseness breaks the clear supervision signal required for denoising, introduces learning difficulties, and leads to errors that accumulate over time; and because of the concentration of the molecular distribution, such errors cannot be tolerated, ultimately resulting in invalid and unstable generations. Under the same forward noising process, the peaks of molecular distributions quickly merge creating overlap regions where samples become indistinguishable. In contrast, image distributions exhibit broader peaks that overlap smoothly and only at later stages. However, for the reverse process of molecular diffusion, a critical problem arises: **overlap regions create intersections or crossings of** generative trajectories which make the score field inherently ambiguous, where multiple plausible directions coexist, but the model can only represent a single *averaged* vector. As a result, the learned score is systematically inaccurate in these regions (Liu et al., 2022; Lee et al., 2023; Ni et al., 2025). Because the peaks are thin, discretization error (Zhang et al., 2023), model limitations, and imperfect score estimation in overlap regions can push the reverse updates too far, placing samples into low-density regions (see Fig. 1). The resulting discrepancy between the true data distribution and the model distribution, caused by artificial inflation of probability mass in invalid regions, then accumulates and propagates (Li & van der Schaar, 2023), ultimately leading to irreversible structural failures. We further analyze this phenomenon in Sec. 3.1. To address this challenge, we focus on the unique nature of molecular data distributions. Since chemically valid molecules occupy only the densely packed distribution peaks, which are confined to narrow and well-separated regions of the representation space, we describe this property as *dense-concentrated structure (DC-structure)*, formally introduced in Definition 3.1 in Sec . 3.1. This definition provides a quantitative handle on the geometry of molecular distributions and lays the theoretical foundation for our analysis. Building on this, we show in Sec. 3.2 how such analysis motivates a corrective method, **DIST**, which DIffuses the intermediate distribution and STeers trajectories back toward valid high-density regions. DIST improves the stability and overall performance of molecular generation, while also providing efficiency gains as an additional benefit. In this work, our main contributions are:
- **Observation.** We are the first to highlight that molecular data distributions are highly concentrated and *dense* that makes diffusion-based generative processes fragile.

- **Theory.** We formalize the notion of DC-structure in molecular distributions and analyze its implications for the intermediate distributions during the diffusion process and error propagation in reverse inference.

- **Method.** Building on this analysis, we design a plug-in corrective module, **DIST**, that can be seamlessly integrated into diverse diffusion-based molecular generation methods.

- **Performance.** Extensive experiments on multiple benchmarks and backbones demonstrate that DIST not only improves stability and overall performance, but also reduces computational cost to nearly half the standard number of timesteps.

## 2 Preliminaries 2.1 Diffusion Models

Diffusion models (DMs) (Ho et al., 2020; Song et al., 2021b) are latent-variable generative models that learn to transform Gaussian noise into data samples through a forward–reverse Markov chain.

Let x ∼ p(x) denote a clean data sample, and let zt denote its progressively noised version at timestep t ∈ {0*, . . . , T*}. Here T is the total number of timesteps, βt ∈ (0, 1) is a variance-schedule 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

![2_image_0.png](2_image_0.png)

parameter, and we use the shorthand z1:T = (z1*, . . . ,* zT ). The forward process gradually corrupts data by adding Gaussian noise:

$$p(\mathbf{z}_{1:T}\mid\mathbf{x})=\prod_{t=1}^{T}p(\mathbf{z}_{t}\mid\mathbf{z}_{t-1}),\qquad p(\mathbf{z}_{t}\mid\mathbf{z}_{t-1})=\mathcal{N}\Big{(}\sqrt{1-\beta_{t}}\;\mathbf{z}_{t-1},\;\beta_{t}I\Big{)}\,.$$  By composition, the marginal conditional distribution admits a closed form:
. (1)
Here αs = 1 − βs controls the noising pace (Ho et al., 2020; Nichol & Dhariwal, 2021b). The unconditional marginal at step t is then

$$p(\mathbf{z}_{t})=\int p(\mathbf{x})\mathcal{N}\big(\mathbf{z}_{t}\,|\,\sqrt{\bar{\alpha}_{t}}\,\mathbf{x},\,(1-\bar{\alpha}_{t})I\big)\ d\mathbf{x},$$

which interpolates between the data distribution p(x) and the Gaussian prior p(zT ) ≈ N (0, I) (Albergo et al., 2023). The reverse process reconstructs data from noise, factorized as qθ(z0:T ) =
q(zT )QT
t=1 qθ(zt−1 | zt), with transitions qθ(zt−1 | zt) = Nµθ(zt, t), ρ2
tI, where µθ is predicted by a neural network and ρt is typically fixed. DMs are trained with the noise-prediction
objective (Song et al., 2021b):
$${\mathcal{L}}_{\mathrm{DM}}=\mathbb{E}_{\mathbf{x},\mathbf{e},t}\big[\|\mathbf{\varepsilon}-\mathbf{\varepsilon}_{\theta}(\mathbf{z}_{t},t)\|^{2}\big],\qquad\mathbf{z}_{t}={\sqrt{\bar{\alpha}_{t}}}\,\mathbf{x}+{\sqrt{1-\bar{\alpha}_{t}}}\,\mathbf{\varepsilon},\quad\mathbf{\varepsilon}\sim{\mathcal{N}}(0,I).$$
√1 − α¯t ε, ε ∼ N (0, I). (4)
The network εθ can be interpreted as learning the score field ∇zt log p(zt) (Song et al., 2021a;b).

New samples are generated by starting from pure Gaussian noise zT ∼ N (0, I) and iteratively applying the reverse update:

$$\mathbf{z}_{t-1}={\frac{1}{\sqrt{1-\beta_{t}}}}\Big(\mathbf{z}_{t}-{\frac{\beta_{t}}{\sqrt{1-\alpha_{t}}}}\,\mathbf{\varepsilon}_{\theta}(\mathbf{z}_{t},t)\Big)+\rho_{t}\mathbf{\varepsilon},\quad\mathbf{\varepsilon}\sim\mathcal{N}(0,I).$$

A 3D molecule with N atoms contains both continuous atomic coordinates and discrete atomic features (Hong et al., 2024). The atomic coordinates are represented as x = (x1*, . . . ,* xN ) ∈

$$p(\mathbf{z}_{t}\mid\mathbf{x})={\mathcal{N}}{\big(}{\sqrt{\alpha_{t}}}\,\mathbf{x},\,(1-{\bar{\alpha}}_{t})I{\big)}\,,\quad{\bar{\alpha}}_{t}=\prod_{s=1}^{t}\alpha_{s}=\prod_{s=1}^{t}(1-\beta_{s}).$$
$$(1)$$
$$(2)$$
$$(3)$$
$$(4)$$

## 2.2 Dms For Molecular Generation

$$(5)$$

3 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 R

N×3, where each xi denotes the coordinates of an atom in R
3. The atomic features, such as charges and atom types, are represented as h = (h1*, . . . ,* hN ) ∈ R
N×d. While the atomic features are scalar quantities invariant to translations and rotations (SE(3)-transformations), the coordinates transform equivariantly under these transformations (Thomas et al., 2018; Hoogeboom et al., 2022; Dumitrescu et al., 2024). However, arbitrary SE(3)-transformations of the coordinates can cause issues for standard denoising networks, since a rotated or translated molecule may be perceived as an entirely different sample. To overcome such issues, existing works often design SE(3)-equivariant frameworks to ensure symmetry-awareness. Specifically, translations can be handled by subtracting the centroid of atomic coordinates x to remove translational degrees of freedom (Garcia Satorras et al., 2021; Xu et al., 2022). However, rotations are much complicated and often handled by using carefully designed equivariant neural networks (Hoogeboom et al., 2022; Xu et al., 2023) or by canonicalization (Ding & Hofmann, 2025; Kaba et al., 2023; Rempe et al., 2020). In addition, the hybrid discrete–continuous nature of molecular data (Dunn & Koes, 2024) introduces unique challenges for generative modeling. Several recent works attempt to address the challenge by learning smoother latent representations (Xu et al., 2023; Ding & Hofmann, 2025; Chen et al., 2025; Luo et al., 2025). These approaches typically employ a VAE-based (Kingma & Welling, 2013) encoder-decoder framework, carrying out the diffusion process in a latent space rather than directly on molecular coordinates and features. While this alleviates some modeling challenges, latent-space methods introduce new sources of approximation error, and discrepancies remain between generated molecules and chemically valid structures. Importantly, the error introduced by the learned score model (see equation 4) is ubiquitous and largely independent of architectural choices (Song et al., 2023; 2024; Joshi et al., 2025); we observe such failures across GNN- and Transformer-based models, as well as in both equivariant and non-equivariant molecular generation methods. Moreover, the discrepancy between the true data marginal distribution and the model distribution grows as errors accumulate across timesteps. This observation indicates that **performance cannot be guaranteed solely by architectural** choices intended to simplify score-matching (Song et al., 2021b). Instead, it highlights the necessity of correcting inference trajectories at intermediate timesteps in order to reduce distributional discrepancies and thereby improve the stability and validity of generated molecules. Moreover, a detailed discussion on the comparison of our work with corrective method is provided in Appendix B.

## 3 Method

In this section, we delve into three key questions: (1) How can the unique structure of molecular distributions, constrained by chemical rules, be formally characterized? (2) What issues arise due to this structure for 3D molecular diffusion models? (3) Can these issues be mitigated through correction? We answer the first two questions by formally investigating the DC-structure of molecular distributions in Sec. 3.1. Building on this insight, we propose DIST together with its theoretical analysis in Sec. 3.2, which addresses the last question.

## 3.1 Dense-Concentrated Structure Issue

As illustrated in Fig. 1, molecular data distribution over the representation space exhibits an evident DC-structure, where each peak corresponds to a chemically valid molecule, and regions between the peaks are of near-zero density. This contrasts with images, where the pixel values can tolerate a wide range of variations, resulting in wider peaks and smoother transitions. **To rigorously capture** this phenomenon and further analyze its implications, we next formalize the DC-structure in probabilistic terms. Consistent with Sec. 1 and prior work, we denote the true and model marginals by p(zt) and qθ(zt), respectively. Unless otherwise stated, all analysis in this work is carried out under the molecular setting rather than the universal diffusion machinery. For notational simplicity, we write the true marginal as pt and also omit the learnable parameter θ and write the model marginal as qt.

Definition 3.1 (Dense-concentrated Structure). There exist K0 centers {mk}*, a scale* σ∗ > 0, a separation ∆ > 0, and weights {wk} *such that, for the operative noise level* t,

$$p_{t}\;\simeq\;\sum_{k=1}^{K_{0}}w_{k}\,{\cal N}(m_{k},\Sigma_{k,t}),\qquad\Sigma_{k,t}\preceq\sigma_{*}^{2}I,\qquad\|m_{k}-m_{\ell}\|\geq\Delta\;\;(k\neq\ell),$$

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 and for each k there exists some ℓ ̸= k with ∥mk − mℓ∥ ≤ O(∆), and

$$p_{t}\left(\bigcup_{k=1}^{K_{0}}B(m_{k},c\sigma_{*})\right)\ \geq\ 1-\delta_{t}$$

for some c > 0 and small δt ∈ [0, 1)*, where* B(m, r) = {x ∈ R
d: ∥x − m∥ ≤ r} denotes the Euclidean ball of radius r *centered at* m.

Under this definition, pt is a mixture of narrow peaks {B(mk*, cσ*∗)}∀k separated by low-density gaps. For molecular data, the parameter σ∗ is small, reflecting that each valid configuration is concentrated in a narrow neighborhood of configuration space. At moderate timesteps t, forward noising smooths these peaks (see equation 2 and equation 3) and creates overlap regions between them, thus a sample zt may lie in the overlap, close to the midpoint between two peaks (see Fig. 1). In this circumstance, the score field points outward, pushing zt toward the nearest peak with magnitude
∥∇ log p(zt)∥ ∼ ∆
σ2∗
(Song et al., 2021b), and the reverse update step based on equation 5 is

$\|\mathbb{Z}_{t-1}-\mathbb{Z}_{t}\|_{\text{det}}\approx\mathbb{B}_{t}\cdot\frac{\mathbb{A}}{\sigma_{\frac{1}{2}}^{2}}$.  
$\left(\widehat{\mathbb{M}}\right)$
$\left(\sqrt{\lambda}\right)$. 
. (6)
Because σ∗ is small for molecules under Definition 3.1, this step can easily overshoot the distribution radius cσ∗ and land in a low-density area:
$\beta_{t}\frac{\Lambda}{\sigma_{*}^{2}}>\alpha_{*}\quad\Longrightarrow\quad2_{t-1}\notin\bigcup B(m_{k},\alpha_{*})$.  
k
B(mk*, cσ*∗). (7)
The derivation and toy examples are provided in Appendix C. In other words, when zt originates from an overlap region created by forward noising, the reverse step is prone to push it across a thin peak and into a low-density region. Subsequent denoising cannot recover from this drift. For images, by contrast, peaks are broad (σ∗ is large) and can overlap smoothly, so the condition in equation 7 is rarely triggered. Consequently, the overshoot mechanism in equation 7, which arises directly from the concentration property in Definition 3.1, explains the fragility of reverse inference. The score field ∇ log pt indeed points *toward* high-density peaks; however, because molecular peaks are narrow, the reverse update can step *past* the peak and cross the high-density into the opposite regions. Once outside the distribution, subsequent updates are driven by the model score ∇ log qt **in a low-density region**
where estimation and discretization errors are large (Zhang et al., 2023; Li & van der Schaar, 2023), leading to oscillation or further drift rather than reliable re-entry into the correct peak. Moreover, Cao et al. (2023) also analyzed this re-entry problem and demonstrated the benefits of stochastic samplers, which further underscores the importance of trajectory correction in SDE simulation. This phenomenon is more obvious in molecular generation due to the DC-structure, and we provide a detailed comparison and explanation specific to molecules in Appendix D.

Table 1: Effect of starting timestep t on sample quality. t = 0 uses clean data; t = 1000 starts from pure Gaussian noise (standard diffusion). Intermediate t

forms zt ∼ p(zt | x), and then we run t reverse

steps for generated results. The experiment setting follows EDM on QM9. Higher numbers are better. Please refer to Sec. 4.1 for further details.

t Atom Sta (%) Mol sta (%) Valid (%) 0 99.0 95.2 97.7

100 99.0 92.7 96.4 300 98.9 89.1 95.5 500 98.7 86.2 94.3

1000 98.7 82.0 91.9

As discussed in Sec. 3.1 above, the unique characteristics of the molecular data distribution lead to severe inference and learning difficulties, such that the learned denoiser can be very inaccurate. In practice, discrepancies between the true marginal pt and the model qt accumulate across timesteps, and low-density region excursions become effectively unrecoverable.

As shown in Table 1, inference quality degrades monotonically with t increasing, reflecting the growing deviation between pt and qt. This motivates the need for a corrective mechanism at intermediate timesteps to prevent off-distribution drift. An overview of our proposed method, DIffuse and STeer (**DIST**), is illustrated in Fig. 2, and we formalize how DIST selectively realigns qt with pt in Sec. 3.2.

## 3.2 Diffuse And Steer

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

![5_image_0.png](5_image_0.png)

Figure 2: Illustration of **DIST**. In standard reverse inference, trajectories diffuse backward from Gaussian noise qT toward the data distribution p0, but the model distribution qt may drift away from the true distribution pt due to the DC-structure of molecular data (see Sec. 3.1). At an intermediate timestep t, **DIST** steers qt toward pt via a correction module that evaluates discrepancies and discards invalid samples. The resulting corrected distribution q ct better approximates pt, realigning trajectories and improving both stability and validity in the final generation.

As a result, the intermediate model distribution qt often deviates from the true marginal distribution pt. Moreover, during training, the diffusion model is trained on the true marginal distribution pt from the dataset. In other words, the reverse process is implicitly learned under the assumption that the intermediate states follow the true marginals. Intuitively, when qt drifts away from pt, this will create a mismatch between the final distribution obtained by applying the reverse process to qt and that obtained from pt. Mathematically, we can show that this is true in Corollary 3.1 below.

Corollary 3.1 (TV–contraction Step). Let Kt→0 be the ideal reverse Markov kernel, which can be intuitively understood as the perfect diffusion model with the true score functions; in other words, when the ideal reverse Markov kernel is applied to the true marginal distribution, we obtain the true data distribution p0 = Kt→0pt. Then, for any probability measure qt, there exists a TV–contraction coefficient κ ∈ [0, 1] *such that*

$$\left\|q_{0}-p_{0}\right\|_{\mathrm{TV}}=\left\|K_{t\to0}q_{t}-K_{t\to0}p_{t}\right\|_{\mathrm{TV}}\leq\kappa\left\|q_{t}-p_{t}\right\|_{\mathrm{TV}},$$

where if qt is the intermediate model distribution, q0 can be understood as the final model distribution obtained by applying the perfect diffusion model on qt.

The proof and explanation are deferred to the Appendix E.1. **Specifically, Corollary 3.1 reveals** that if the intermediate model distribution qt **is closer to the true marginal distribution** pt, the final model distribution q0 is closer to the true data distribution p0 **that we aim to obtain.** Therefore, to achieve high-quality generation despite the difficulties posed by the molecular data distribution, **our goal is to obtain an improved intermediate distribution** q c t**that remains closer**
to the true marginal pt **rather than blindly using the model distribution.** To achieve this goal, we propose **DIST** (DIffuse and STeer), a corrective sampling approach for 3D molecular diffusion.

Specifically, we perform the reverse process normally as in the standard diffusion pipelines; however, we incorporate an additional correction step to steer the intermediate distribution qt toward a
"corrected" version q ctcloser to the true marginal pt. An overview of DIST is provided in Fig. 2.

We now present the details of DIST concretely. Building on Definition 3.1, which states that the distribution pt concentrates around a finite number of peaks separated by low-density regions, we next introduce a finer partition of the support into small neighborhoods. Specifically, we divide the space into radius-r batches {Bj}
J
j=1, which can be regarded as local regions within or around the peaks, each carrying probability mass πj := pt(Bj ), πˆj := qt(Bj ),
together with the conditional distributions pt|j and qt|jrestricted to each batch Bj .

Each batch j is further associated with a model-side pilot score sj ∈ R (e.g., round-trip residual, self-consistency, ensemble variance, or chemistry-based penalty), which reflects whether the region 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

$$\alpha(\tau):=\sum_{j\in J^{\ast}(\tau)}\pi_{j},\qquad\beta(\tau):=\sum_{j\in J^{\ast}(\tau)}\hat{\pi}_{j}.$$
$i\leq\tau$ }. 
Here, α(τ ) represents the *true coverage*, i.e., the portion of the ground-truth distribution preserved by the selection, while β(τ ) denotes the *model coverage*, i.e., the portion of the model distribution retained. Smaller thresholds τ restrict the selection to batches that are more likely to correspond to valid regions, reducing coverage; larger thresholds broaden the selection and capture more mass, but at the cost of admitting regions inconsistent with the true distribution. The selected model distribution at threshold τ is then given by

$$q_{t}^{\mathbb{C}}(\tau):=\sum_{j\in J^{*}(\tau)}\tilde{\pi}_{j}\,q_{t|j},\qquad\tilde{\pi}_{j}=\frac{\tilde{\pi}_{j}}{\sum_{k\in J^{*}(\tau)}\tilde{\pi}_{k}}.\tag{9}$$

Intuitively, qt consists of both samples consistent with pt, lying within valid regions, and samples that fall outside. The corrected distribution q c tacts as a filtered version of qt, removing invalid batches in order to improve approximation of the true distribution. The following proposition establishes a quantitative error bound that illustrates the effectiveness of DIST.

Proposition 3.1 (Selective Reverse Error Bound). Under the DC-structure in Definition 3.1 and the batch construction described above, for any threshold τ *the deviation between the* selectively corrected reverse distribution Kt→0q c t(τ ) *and the true distribution* p = Kt→0pt admits an upper bound of the form

$$\left\|K_{t\to0}q_{t}^{c}(\tau)-p\right\|_{\mathrm{TV}}\;\leq\;f\big(\alpha(\tau),\beta(\tau),(\pi_{j},\hat{\pi}_{j})_{j\in J^{*}(\tau)},\;\sup_{j\in J^{*}(\tau)}\mathrm{TV}(q_{t|j},p_{t|j})\big),$$

where f(·) is an explicit function of the true coverage α(τ ), the model coverage β(τ ), the selected batch weights, and the conditional discrepancies. The exact form of f(·) is provided in Appendix E.2.

The proof and explanation are provided in Appendix E.2. This error bound provides a theoretical guarantee for DIST; that is, **selective correction ensures that** q c t**is steered toward convergence**
with the true distribution p at intermediate timestep t**, stabilizing the sampling trajectory.**
Corrective Sampling We now describe how the corrected distribution q c tis achieved in the reverse inference procedure (see Fig. 2). At a given intermediate timestep t, DIST constructs a candidate pool by reverse-simulating a small set of samples from Gaussian noise at T. Each candidate is duplicated and perturbed with a sufficiently small amount of noise to form batches {Bj}
J
j=1, which collectively follow the model distribution qt and remain within the prescribed radius-r constraint (see Definition 3.1). To evaluate whether these batches {Bj}
J
j=1 are consistent with the true distribution pt, DIST runs a full reverse inference on a pilot subset {Bsub j| Bsub j ∈ Bj}
J
j=1 drawn from each batch. This pilot inference provides an empirical assessment of how well the current model trajectory aligns with pt, and serves as a diagnostic of potential drift away from the true distribution. Based on the pilot outcomes sj ∈ R, DIST applies a filter π˜j to each batch using a universal threshold τ , obtaining a corrected distribution q c t
(τ ) (see equation 9) that better approximates pt. In effect, q c tconcentrates the reverse trajectories around valid molecular peaks. Beyond improved approximation quality, DIST also provides an efficiency advantage by reducing unnecessary inference on invalid regions, as demonstrated in Sec. 4.3. 4.1 SETUPS Datasets Following prior work (Hoogeboom et al., 2022; Xu et al., 2023; Song et al., 2024), we evaluate DIST on two widely used datasets in molecular generation: QM9 (Ramakrishnan et al., 2014) and GEOM-Drugs (Axelrod & Gomez-Bombarelli, 2022). QM9 contains 130K small ´ We then measure how much probability mass remains after this selection by defining is consistent with the true marginal distribution or potentially invalid. Given a threshold τ , we select batches whose scores fall below τ :
J
⋆(τ ) := { j : sj ≤ τ }.

## 4 Experiments

| bold. Global best results are underlined.   | QM9          | GEOM-Drugs   |           |                  |              |           |
|---------------------------------------------|--------------|--------------|-----------|------------------|--------------|-----------|
| # Metrics                                   | Atom Sta (%) | Mol Sta (%)  | Valid (%) | Valid×Unique (%) | Atom Sta (%) | Valid (%) |
| Data                                        | 99.0         | 95.2         | 97.7      | 97.7             | 86.5         | 99.9      |
| ENF                                         | 85.0         | 4.9          | 40.2      | 39.4             | -            | -         |
| G-SchNet                                    | 95.7         | 68.1         | 85.5      | 80.3             | -            | -         |
| EDM                                         | 98.7         | 82.0         | 91.9      | 90.7             | 81.3         | 92.6      |
| EDM+DIST                                    | 99.2±0.0     | 89.9±0.3     | 96.9±0.2  | 94.1±0.3         | 82.2         | 96.0      |
| GeoLDM                                      | 98.9         | 89.4         | 93.8      | 92.7             | 84.4         | 99.3      |
| GeoLDM+DIST                                 | 99.4±0.0     | 93.4±0.3     | 96.3±0.2  | 93.1±0.2         | 85.4         | 99.7      |
| RADM                                        | 98.5         | 87.3         | 94.1      | 91.7             | 85.0         | 99.3      |
| RADM+DIST                                   | 99.1±0.0     | 91.4±0.3     | 96.2±0.1  | 92.3±0.4         | 86.0         | 99.8      |

molecules, restricted to at most 9 heavy atoms (29 atoms including hydrogen atoms). We follow the standard partition from Hoogeboom et al. (2022), with 100K molecules for training, 18K for validation, and 13K for testing. GEOM-Drugs is substantially larger, comprising 420K molecules with an average of 44.4 atoms and up to 181 atoms. Following Hoogeboom et al. (2022), we retain the 30 lowest-energy conformations for each molecule. Metrics Consistent with prior work, we evaluate generated molecules using the following metrics: atom stability, molecule stability, validity, and validity×uniqueness (Simonovsky & Komodakis, 2018; Garcia Satorras et al., 2021). *Atom Stability*: the percentage of atoms whose number of bonds matches their valence (e.g., H:1, C:4, O:2). *Molecule Stability*: the percentage of molecules in which all atoms are stable. *Validity*: the percentage of molecules satisfying valence rules for all atoms. Uniqueness: the percentage of molecules that are distinct from one another. Note for GEOM-Drugs, following prior work, we omit the stability and uniqueness metrics, since they are consistently close 0% and 100%, respectively, for all evaluated methods including the baseline methods.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Baselines We employ several representative state-of-the-art diffusion models for 3D molecular generation, including EDM (Hoogeboom et al., 2022), GeoLDM (Xu et al., 2023), and RADMDiT-B (Ding & Hofmann, 2025), as backbone models for our proposed DIST and compare with the original without DIST. These backbone diffusion models cover a range of model types, including GNN-based or Transformer-based, equivariant and non-equivariant, and those operating in regular space and latent space. In addition, we include comparisons with well-known non-diffusionbased models, such as ENF (Garcia Satorras et al., 2021) and G-SchNet (Gebauer et al., 2019). The results of backbone models and baseline methods are directly obtained from their original work. Implementation Details To demonstrate the plug-in capability of our DIST and ensure fair comparison, for all backbone models, we strictly use the officially released model weights without altering any hyperparameters or settings for noise schedule, encoder-decoder configurations, and dataset partition. For detailed settings of DIST, please refer to Appendix F.

## 4.2 Main Results And Analysis

To evaluate the performance of each model on QM9 and GEOM-Drug, following prior work, we generate 10,000 3D molecules using each model. The main results are summarized in Table 2. For QM9 dataset, we report averages over three runs together with standard deviations. Across both datasets and all metrics, every backbone model combined with DIST consistently outperforms its original counterpart. The improvements are significant and universal: **all bold numbers in Table 2** indicate that DIST significantly improves the quality of generated molecules, with particularly large margins observed on the most critical stability metrics. In addition, methods based on our DIST set the new state-of-the-art for molecular generation on both QM9 and GEOM-Drug datasets. Notably, the margins of improvement observed before and after applying our method highlight the generality of DC-structure issue. Across GNN-based equivariant EDM (Hoogeboom et al., 2022), GeoLDM (Xu et al., 2023) and Transformer-based non-equivariant RADM (Ding & Hofmann, 2025), where GeoLDM and RADM perform in latent space, the issue remains consistently 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 evident. This observation cautions against relying solely on architectural choices. Our experimental results confirm that, as a plug-in component, DIST effectively steers inference trajectories and thus mitigates distributional discrepancies in the sampling process, providing a valuable complement to architectural innovations to improve 3D molecular generation quality.

## 4.3 Efficiency Analysis

Since the batches {Bj}
J
j=1 are created by duplication and perturbation, DIST requires only T −t |B| expected timesteps per inference from T (1000 is adopted in backbone models) to t, where |B| is the batch size. For example, setting t = 300 with |B| = 100, each accepted batch after threshold filtering requires only 307 (
1000−300 100 + 300) steps instead of the 1000 steps as used in standard counterparts. A detailed comparison of efficiency is provided in Table 3, which shows DIST can substantially reduce the overall timestep by nearly **half** compared to baselines, while significantly improving the generation quality as shown in Table 2. We also provide a detailed quantification of the expected computational cost of our DIST in Appendix G.1.

## 4.4 Ablation Study

The number of pilot samples drawn from each batch plays a critical role. A larger set of pilot samples provides a more accurate representation of the model distribution qt, and leads to a better corrected distribution q c t by DIST. However, increasing the number of pilot samples also leads to higher computational costs. In practice, we may choose a pilot set size that is sufficiently representative while remaining computationally affordable. We conduct an ablation study to compare the final sample quality and computational costs under different numbers of pilot samples, with results reported in Table 4. As expected, increasing the number of pilot samples improves the quality of generated molecules monotonically. At the same time, computational costs (measured by the number of time steps) also increase monotonically. Nevertheless, even under a relatively small budget (30, 50, 100), DIST still demonstrates superior performance, significantly improving the original EDM in both sample quality and computational efficiency. Moreover, we also constructed the ablation study on hyperparameters, including batch score threshold, intermediate timestep, and perturbation intensity, as shown in Appendix H.

Table 3: Average number of timesteps required for a full inference procedure. The values are computed from the total timestep consumption needed to generate 10,000 molecules, corresponding to the experiments in Table 2. All baseline methods use the standard 1000-step schedule, whereas DIST significantly reduces the computational cost.

Methods QM9 GEOM-Drugs EDM+DIST 556.1 503.3 GeoLDM+DIST 416.9 636.7 RADM+DIST 413.7 438.8 Baselines 1000 1000

| Size   | Atom Sta (%)   | Mol Sta (%)   | Valid (%)   | Valid×Unique (%)   | Timesteps   |
|--------|----------------|---------------|-------------|--------------------|-------------|
| 30     | 99.2           | 89.5          | 96.7        | 94.3               | 428.3       |
| 50     | 99.2           | 89.9          | 96.9        | 94.1               | 556.1       |
| 100    | 99.3           | 90.5          | 97.3        | 94.9               | 644.7       |

## 5 Conclusion And Future Work

In this work, we investigated the unique challenge of applying diffusion models to molecular generation. Molecular data are confined to concentrated regions of the representation space, with chemically valid structures corresponding to densely packed sharp peaks separated by regions of nearzero density. This DC-structure makes diffusion modeling fragile, since small errors at intermediate timesteps are amplified, causing generative trajectories to drift off-distribution and accumulate irreparable structural violations. To address this issue, we proposed DIST, which is a selective correction method that filters and rescales intermediate distributions, steering the inference trajectories toward valid molecular peaks. DIST is model-agnostic and can be integrated into a wide range of 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 diffusion-based molecular generators. We also provided both theoretical analysis and experimental results to demonstrate that our method consistently improves the performance across multiple architectures for molecular generation, while nearly halving the inference cost. Looking forward, our work opens several promising directions. First, as a general and principled framework, DIST can be extended to other data domains with a similar distribution structure. An intriguing question is whether the DIST framework can be adapted to protein generation, although this constitutes a fundamentally different and substantially more complex task. Second, adaptive selection or other strategies for filtering may further improve correction efficiency. Finally, while our study focuses on diffusion models, the DC-structure issue is not exclusive to them. Exploring analogous corrective strategies in alternative generative paradigms, such as normalizing flows (Rezende & Mohamed, 2015), autoregressive models (Li et al., 2024), or energy-based frameworks (Du & Mordatch, 2019), may broaden the impact of our approach and provide a unifying principle for modeling highly constrained distributions.

## 540

541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 All theoretical results are stated with explicit assumptions, and complete proofs are provided in Appendix E.1 and Appendix E.2. The datasets used in our experiments (QM9 and GEOM-Drugs) are publicly available, and we describe all preprocessing steps in Sec. 4.1. After acceptance, we will publicly release the code and provide detailed guidance to facilitate reproduction of all results.

## References

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. *arXiv preprint arXiv:2303.08797*, 2023.

Simon Axelrod and Rafael Gomez-Bombarelli. Geom, energy-annotated molecular conformations ´
for property prediction and molecular generation. *Scientific Data*, 9(1):185, 2022. doi: 10.1038/
s41597-022-01288-4. URL https://doi.org/10.1038/s41597-022-01288-4.

Montgomery Bohde, Mrunali Manjrekar, Runzhong Wang, Shuiwang Ji, and Connor W Coley. Diffms: Diffusion generation of molecules conditioned on mass spectra. arXiv preprint arXiv:2502.09571, 2025.

Yu Cao, Jingrun Chen, Yixin Luo, and Xiang Zhou. Exploring the optimal choice for generative processes in diffusion models: Ordinary vs stochastic differential equations. Advances in Neural Information Processing Systems, 36:33420–33468, 2023.

Zitao Chen, Yinjun Jia, Zitong Tian, Wei-Ying Ma, and Yanyan Lan. Manipulating 3d molecules in a fixed-dimensional se (3)-equivariant latent space. *arXiv preprint arXiv:2506.00771*, 2025.

Seungyeon Choi, Hwanhee Kim, Chihyun Park, Dahyeon Lee, Seungyong Lee, Yoonju Kim, Hyoungjoon Park, Sein Kwon, Youngwan Jo, and Sanghyun Park. Controllable 3d molecular generation for structure-based drug design through bayesian flow networks and gradient integration. arXiv preprint arXiv:2508.21468, 2025.

Yuhui Ding and Thomas Hofmann. Scalable non-equivariant 3d molecule generation via rotational alignment. *arXiv preprint arXiv:2506.10186*, 2025.

Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. *Advances* in neural information processing systems, 32, 2019.

Alexandru Dumitrescu, Dani Korpela, Markus Heinonen, Yogesh Verma, Valerii Iakovlev, Vikas Garg, and Harri Lahdesm ¨ aki. E (3)-equivariant models cannot learn chirality: Field-based molec- ¨ ular generation. *arXiv preprint arXiv:2402.15864*, 2024.

Ian Dunn and David Ryan Koes. Mixed continuous and categorical flow matching for 3d de novo molecule generation. *ArXiv*, pp. arXiv–2404, 2024.

Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. *arXiv preprint arXiv:2410.12557*, 2024.

Victor Garcia Satorras, Emiel Hoogeboom, Fabian Fuchs, Ingmar Posner, and Max Welling. E (n)
equivariant normalizing flows. *Advances in Neural Information Processing Systems*, 34:4181– 4192, 2021.

This work adheres to general ethical principles of scientific research. Our goal is to contribute to society and scientific progress by improving generative modeling for molecular data. We have carefully considered possible harms: our method is purely methodological and does not involve sensitive personal data, human subjects, or confidential information. All experiments rely on publicly available molecular datasets, and no privacy concerns arise. We believe our work will benefit the community as a complementary tool for advancing generative modeling, without introducing foreseeable risks of discrimination or misuse beyond the general risks associated with generative models.

## Reproducibility Statement Ethics Statement

Niklas Gebauer, Michael Gastegger, and Kristof Schutt. Symmetry-adapted generation of 3d point ¨
sets for the targeted discovery of molecules. *Advances in neural information processing systems*, 32, 2019.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. arXiv preprint arxiv:2006.11239, 2020.

Haokai Hong, Wanyu Lin, and Kay Chen Tan. Accelerating 3d molecule generation via jointly geometric optimal transport. *arXiv preprint arXiv:2405.15252*, 2024.

Emiel Hoogeboom, Vıctor Garcia Satorras, Clement Vignac, and Max Welling. Equivariant diffu- ´
sion for molecule generation in 3d. In *International conference on machine learning*, pp. 8867–
8887. PMLR, 2022.

Chaitanya K Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, and Zachary W Ulissi. All-atom diffusion transformers: Unified generative modelling of molecules and materials. *arXiv preprint arXiv:2503.03965*, 2025.

Sekou-Oumar Kaba, Arnab Kumar Mondal, Yan Zhang, Yoshua Bengio, and Siamak Ravanbakhsh. ´
Equivariance with learned canonicalization functions. In International Conference on Machine Learning, pp. 15546–15566. PMLR, 2023.

Joowon Kim, Ziseok Lee, Donghyeon Cho, Sanghyun Jo, Yeonsung Jung, Kyungsu Kim, and Eunho Yang. Early timestep zero-shot candidate selection for instruction-guided image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 18844–18854, 2025.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. *arXiv preprint* arXiv:1312.6114, 2013.

Sangyun Lee, Beomsu Kim, and Jong Chul Ye. Minimizing trajectory curvature of ode-based generative models. In *International Conference on Machine Learning*, pp. 18957–18973. PMLR, 2023.

Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and Marie-Francine Moens. Alleviating exposure bias in diffusion models through sampling with shifted time steps. arXiv preprint arXiv:2305.15583, 2023.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. *Advances in Neural Information Processing Systems*, 37: 56424–56445, 2024.

Yangming Li and Mihaela van der Schaar. On error propagation of diffusion models. arXiv preprint arXiv:2308.05021, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. *arXiv preprint arXiv:2209.03003*, 2022.

Eric Martin and Eddie Cao. Euclidean chemical spaces from molecular fingerprints: Hamming distance and hempel's ravens. *Journal of computer-aided molecular design*, 29(5):387–395, 2015.

Yuyan Ni, Shikun Feng, Haohan Chi, Bowen Zheng, Huan-ang Gao, Wei-Ying Ma, Zhi-Ming Ma, and Yanyan Lan. Straight-line diffusion model for efficient 3d molecular generation. arXiv preprint arXiv:2503.02918, 2025.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Yanchen Luo, Zhiyuan Liu, Yi Zhao, Sihang Li, Hengxing Cai, Kenji Kawaguchi, Tat-Seng Chua, Yang Zhang, and Xiang Wang. Towards unified and lossless latent space for 3d molecular latent diffusion modeling. *arXiv preprint arXiv:2503.15567*, 2025.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In *International conference on machine learning*, pp. 8162–8171. PMLR, 2021a.

Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models.

In *International conference on machine learning*, pp. 8162–8171. PMLR, 2021b.
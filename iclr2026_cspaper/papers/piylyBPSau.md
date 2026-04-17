000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

## 1 Introduction

Few-shot novel view synthesis (NVS) aims to synthesize images of the target scene from unseen viewpoints given a set of sparse images from limited known viewpoints. This task demonstrates significant practical value in highquality rendering upon data sparsity (Zhu et al., 2024). Existing methods focus on adapting general NVS models, *e.g.*, Neural Radiance Fields (NeRF) (Mildenhall et al., 2020) and 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023), for few-shot NVS via prior knowledge (Chen et al., 2021; Niemeyer et al., 2022; Kulhanek ´ et al., 2022; Yu et al., 2021a; Wang et al., 2023; Yang et al., 2023; Li et al., 2024a; Zhu et al., 2024; Paliwal et al., 2024; Zhang et al., 2024).

In particular, those high-fidelity and efficient few-shot NVS methods based on 3DGS are generally characterized by a two-phase pipeline: (1) 3D Gaussian initialization based on fused stereo points generated from training views (Zhu et al., 2024) or image pixels in training views using corresponding depth maps (Paliwal et al., 2024), and (2) 3D Gaussian optimization based on enhanced priors from training views Conventional few-shot novel view synthesis (NVS) methods using 3D Gaussian Splatting (3DGS) have demonstrated significance, but remain constrained by their overdependence on the limited information from training views. Their unsatisfactory scene completion capability would underrepresent those scene regions either unobserved in training views or with local details and thus cause floating artifacts against high fidelity. To address these challenges, we propose GenCoGS, a unified 3DGS-based few-shot NVS method focusing on initializing and optimizing 3DGS representation using generative completion-based strategies to enhance scene completion. Specifically, our generative point cloud completion-based strategy produces and filters complementary points toward a complete point cloud with refined structural and appearance information for Gaussian initialization; The generative pseudo view completion-based strategy leverages an image-to-video diffusion model to synthesize complete pseudo views, which benefits Gaussian optimization especially within unobserved scene regions and mitigates hallucination for less appearance distortion. Integrating both strategies enables accurate and coherent scene completion for high-fidelity few-shot NVS. Extensive experiments on three benchmark datasets demonstrate the superiority of our GenCoGS for fewshot NVS evaluated in common metrics compared to baseline methods. Compared to those 3DGS-based few-shot NVS methods, our GenCoGS achieves improvements of up to 2.40 dB, 0.08 and 0.125 in PSNR, SSIM and LPIPS.

Figure 1: Limited scene completion capability

![0_image_0.png](0_image_0.png) of existing 3DGS-based few-shot methods, represented by (a) insufficient local details due to the incomplete initialization of Gaussians; and (b) floating artifacts in unobserved regions due to the optimization guided by pseudo views.

Anonymous authors Paper under double-blind review

## Abstract

1

# Gencogs: Generative Completion-Based 3D Gaussian Splatting For High-Fidelity Few- Shot Novel View Synthesis

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 with additional supervision from sampled pseudo views (Zhu et al., 2024; Zhang et al., 2024). Despite the significant performances achieved, these methods are fundamentally confined by the nature of solely leveraging observed information, causing considerably less competent results within certain scene regions. As shown in Figure 1 (a), the initial Gaussians unsatisfactorily represent the scene's structure and appearance in those regions unobserved in training views or with local details; Meanwhile, pseudo views sampled from training views contribute primarily to the observed regions during Gaussian optimization, but lead to floating artifacts within the unobserved regions, as illustrated in Figure 1 (b). These challenges suggest that these methods lack the human *imagination* for imagery generation as scene completion (Pearson, 2019). It inspires us to explore if few-shot NVS, which is less-constrained and under-determined, can be transformed into a sufficiently constrained and observed task by exploiting the mechanism of human imagination. Considering the notable completion capabilities of recently boosted generative models (Song et al., 2020; Yu et al., 2024a; Wu et al., 2024), we propose a novel unified few-shot NVS method, Generative Completion-based 3DGS (**GenCoGS**), to address the aforementioned challenges. This unified method is characterized by two *generative completion-based strategies* on initializing and optimizing scene representation for 3DGS. The former strategy generates a complementary point set and filters this point set to complete the initial point cloud obtained by the SfM (Zhu et al., 2024) regarding structural and appearance details for 3D Gaussian initialization. The latter strategy for 3D
Gaussian optimization adopts a perturbed camera trajectory to sample pseudo camera poses probably covering unobserved regions, and an image-to-video (I2V) diffusion model (Yu et al., 2024a) for conditional completion of pseudo views; Meanwhile, a generative consistency loss is designed to provide additional supervision. Both strategies jointly enhance the 3DGS' capability of scene completion while mitigating appearance distortion and floating artifacts caused by the hallucination of generative models (Aithal et al., 2024). Extensive experiments on LLFF (Mildenhall et al., 2019), DTU (Jensen et al., 2014) and Shiny (Wizadwongsa et al., 2021) benchmark datasets, demonstrate that GenCoGS can achieve the state-of-the-art performance under representative few-shot settings with 3, 6 and 9 input training views. The contributions of this paper can be summarized as follows: - Inspired by the mechanism of human imagination, we propose a unified few-shot NVS method based on generative completion with focus on initializing and optimizing scene representation.

- To the best of our knowledge, we design, for the first time, a generative point cloud completionbased Gaussian initialization strategy leveraging complementary point generation and filtering; and a generative pseudo view completion-based Gaussian optimization strategy exploiting imageto-video diffusion models against hallucination.

- Based on the scene completion capability, the proposed method can outperform representative few-shot NVS solutions across three benchmark datasets.

## 2 Related Works

Few-shot Novel View Synthesis Few-shot NVS aims to reconstruct accurate and visually compelling 3D scenes from sparse training views, yet suffers from geometric–radiance ambiguity due to insufficient observations. NeRF-based methods mitigate overfitting through strategies such as geometric and color regularization (Niemeyer et al., 2022), depth supervision (Deng et al., 2022), depth distillation (Wang et al., 2023), and generalizable priors via pretrained models (Yu et al., 2021a; Chen et al., 2021; Li et al., 2024b). Despite these advances, implicit MLP-based representations remain computationally demanding and challenging to combine with explicit 3D scene models. Explicit 3DGS-based methods offer advantages in rendering efficiency and quality and have introduced dedicated regularizations to handle sparse inputs. Notably, FSGS (Zhu et al., 2024) and DNGaussian (Li et al., 2024a) use sparse depth supervision to align Gaussians with geometric priors, while CoherentGS (Paliwal et al., 2024) ensures spatial coherence through optical flow constraints. Nevertheless, these methods are constrained to the observed regions in training views and struggle to model unobserved structure. Unlike prior-based methods, our GenCoGS performs generative completion over unobserved regions by employing strategies on Gaussian initialization and optimization, which jointly enable high-fidelity few-shot NVS with structurally sound and realistic results. Diffusion Priors for Novel View Synthesis Recent advances in diffusion models have suggested their utility as informative priors for text-driven 3D generation. DreamFusion (Poole et al., 2022)
108

![2_image_0.png](2_image_0.png) 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 adopts score distillation sampling to leverage pre-trained 2D diffusion models for 3D object synthesis from text prompts, influencing subsequent studies (Tang et al., 2023a; Yi et al., 2024). To improve 3D consistency, Zero-1-to-3 (Liu et al., 2023) and MVDream (Shi et al., 2023) incorporate 3D-aware learning into diffusion models, though they depend on large-scale training data and computation-expensive pipelines. Alternative methods, such as HiFi-123 (Yu et al., 2024b) and Make-It-3D (Tang et al., 2023b), employ a single image with diffusion-based priors for 3D reconstruction but require per-scene optimization that limits scalability. The successes of these methods in 3D generation or reconstruction, however, have exhibited limitations in high-fidelity few-shot NVS. Meanwhile, ReconFusion (Wu et al., 2024) and IPSM (Wang et al., 2024) demonstrate that diffusionguided NeRF and 3DGS can accomplish high-quality few-shot NVS using 2D diffusion-based priors. To ensure multi-view consistency, image-to-video diffusion models have been adapted with camera-controlled generation techniques (Blattmann et al., 2023; Chen et al., 2024; Melas-Kyriazi et al., 2024). ViewCrafter (Yu et al., 2024a), CAT3D (Gao et al., 2024) and ReconX (Liu et al., 2025) have further extended this approach to the few-shot setting by integrating image-to-video diffusion models with iterative point cloud refinement. However, these attempts tend to hallucinate within the target scene's unobserved regions, causing structural and appearance inconsistencies and thus constraining their effectiveness in high-fidelity few-shot NVS. Furthermore, they neglect the importance of the initialization of scene representation for 3DGS.

## 3 Methods 3.1 Generative Point Cloud Completion-Based Gaussian Initialization

The sparse point cloud used to initialize 3D Gaussians in FSGS (Zhu et al., 2024) from SfM (Schonberger & Frahm, 2016), provides the initial information on the scene's structure and appearance. In particular, the initial Gaussians' means follow the corresponding points' spatial positions. Since sparse views may cause the corresponding point cloud to become considerably less informative, i.e., *incomplete* regarding the scene's structural representation in under-observed regions.

A straightforward solution is to generate points for completion, which often results in a dilemma: generative models fill structural hollows, but also introduce significant outliers due to unconstrained hallucination. As shown in Figure 3 (b), the Gaussians initialized using such points cause structural distortion in those regions with details and degrade the few-shot NVS performance. Hence, as shown in Figure 2, our unified Generative point cloud Completion-based Gaussian Initialization (GCGI) strategy produces refined complementary points to enhance the representation of initial point cloud. Specifically, GCGI comprises two sequential modules on complementary point generation and filtering with the *generate-and-filter* paradigm.

## 3.1.1 Complementary Point Generation

Inspired by previous studies (Yu et al., 2021b), we design an end-to-end complementary point generation (CPG) module to produce a complementary set of points for point cloud completion.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Given the point cloud P0 = {p1, p2*, . . . , p*n} that has been used to initialize a provisional set of 3D Gaussians Θ0 = {θ1, θ2*, . . . , θ*n}, the CPG module starts by using the furthest point sampling (FPS) algorithm (Eldar et al., 1997) to downsample P0 for a set of point proxies C0 = {c1, c2*, . . . , c*n},
and adopts a light-weight backbone (i.e., DGCNN (Wang et al., 2019)) F to extract a representation for each point proxy cithat represents the corresponding local structural details, as follows:
fi = F(ci) + P E(ci), (1)
where P E(ci) denotes the position embedding of proxy ci.

To exploit the structural representations and the long-range dependencies among different local parts of the point cloud, the CPG module leverages the Transformer model (Vaswani et al., 2017) that comprises an encoder ME and a decoder MD for the set-to-set generation. In particular, the k-NN algorithm (Kramer, 2013) is employed in each Transformer block to capture the structural relationships among point proxies for the enhancement of the local geometric information, *i.e.*, each query representation is enhanced by processing it and its k nearest representations altogether using a linear layer followed by the max pooling operation. The encoder ME outputs a set of high-level representations F
′from F = {f1, f2*, . . . , f*n}, as follows:
F
′ = ME(F). (2)
Following the idea of dynamic query mechanism (Dai et al., 2021), the decoder MD takes as input both F
′and dynamic queries Q = {q1, q2*, . . . , q*m}, and generates a new set of point proxies C1 =
{c
′1
, c′2
, . . . , c′m}, as follows:C1 = MD(F
′, Q). (3)
Afterward, the CPG employs a point auto encoder-decoder H, *i.e.*, FoldingNet (Yang et al., 2018),
to output a set of complementary points P1 = {P
′1, P′2*, . . . , P*′m} with structural details, as follows, P
′
i = H(c
′
i), (4)
where P
′
idenotes the neighboring points centered at c
′
i.

## 3.1.2 Complementary Point Filtering

As shown in Figure 3 (b), the combined point cloud Pc = P0SP1 with the naive combination of sparse point cloud P0 and the complementary points P1 output by CPG module contains significant outliers. To address this points generative *hallucination*, we devise an additional complementary point filtering (CPF) module to prune outliers in P1 while maintaining the scene's structural details. Previous studies have demonstrated that structures like anchor grids or octrees can contribute to enhancing the local structural details for 3DGS (Lu et al., 2024; Ren et al., 2025). Since few-shot NVS is an ill-posed problem, introducing additional structural information that needs to be optimized would cause training to crash. Therefore, we design a filtering mask in the CPF module to detect outliers for pruning based on K-Dimensional Tree (kd-Tree) (Zhou et al., 2008), an optimize-free space-partitioning data structure. In the absence of ground truth structural information, the incomplete point cloud P0 initially obtained through the SfM is used as a high-confidence reference, for which the CPF module constructs a kd-tree T = {t1, t2*, . . . , t*d} that comprises d parts using the nearest-neighbor search algorithm. For each complementary point p
′i ∈ P1, the CPF module samples k = 3 nearest points
{pi,1, pi,2, . . . , pi,k} ∈ P0, as reference anchors, in its corresponding part ti of T , as follows, pi,k = k-minp∈(P0∩ti)∥p
′i − p∥, p′i ∈ ti. (5)
The reference anchors are adopted to calculate a distance-based outlier indicator yi for p
′ias follows, Figure 3: Comparison of initial point cloud P0,

![3_image_0.png](3_image_0.png) combined point cloud Pc, and final complete point cloud Pf .

$$y_{i}={\frac{1}{k}}\sum_{i=1}^{k}\|p_{i}^{\prime}-p_{i,k}\|.$$
i − pi,k∥. (6)
4 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 3.2.2 GENERATIVE CONSISTENCY LOSS Afterward, the CPF module conducts a binary classification on each complementary point p
′i ∈ P1, i.e., p
′ias an outlier if yi exceeds a predefined threshold δ1 = 1.0 and the mean distance of P0.

Accordingly, the filtering mask is obtained as follows,

$$M={\bf1}(y\leq\delta_{1}\cdot\mu({\bf P}_{0}))\quad\mu({\bf P}_{0})=\frac{1}{n(n-1)}\sum_{i=1}^{n-1}\sum_{j\neq i}^{n}\|p_{i}-p_{j}\|.$$
$$\Theta=\Theta_{0}\cup\Theta_{1},$$
$$\left(7\right)$$

$\left(8\right)$. 
The module then leverages this mask to filter those points distant to high-confidence reference P0 from P1. And, the complete point cloud Pf , which possesses enhanced structural information
barely affected by outliers.
$$\mathbf{P}_{1}^{\prime}=\mathbf{P}_{1}\odot M,\quad\mathbf{P}_{f}=\mathbf{P}_{0}\cup\mathbf{P}_{1}^{\prime}.$$

## 1. (8)
Given The Complete Point Cloud Pf , A Set Of 3D Gaussians Θ For Optimization Can Be Initialized As
Follows,
Θ = Θ0 ∪ Θ1, (9)
Where Θ1 Represents The Complementary 3D Gaussians Initialized Using P′1. Specifically, The Position Of Each Gaussian Θ
′I ∈ Θ1 Follows A Point P
′I ∈ P′1, Whereas The Remaining Attributes Of Θ
′I
Are
Cloned From Those Of Gaussian In Θ0 Corresponding To Nearest Point Pj ∈ P0 Of P
′Iaccording To T . 3.2 Generative Pseudo View Completion-Based Gaussian Optimization

To exploit the sparse training views while preventing overfitting, existing methods (Zhu et al., 2024; Zhang et al., 2024) have attempted to employ pseudo views generated from interpolated camera poses as additional guidance for training. Since such pseudo views are essentially based on the observed regions of the scene, this strategy often still causes *hollows* or incomplete structural details in the reconstruction of those regions unobserved by the input training views after Gaussian optimization. As a countermeasure, our GenCoGS adopts a Generative point cloud Completion-based Gaussian Optimization (GCGO) strategy based on an I2V diffusion model (Yu et al., 2024a) in Figure 2, which is capable of maintaining spatial-temporal consistency, for structurally-aware pseudo view completion against hollows. Specifically, the input training views are processed by the image encoder of a pre-trained languageimage model, *e.g.*, CLIP (Radford et al., 2021), to obtain high-level representations Fc that hold
multi-view consistency information. These representations are then integrated with each initial
pseudo view Ip to provide the conditional information that guides the diffusion model to reach
the corresponding complete pseudo view ˆIp via a multi-step denoising process, as follows:
$$z_{t-1}=p_{\theta}(z_{t},\mathbb{E}[z_{0}\mid z_{t},F_{c},I_{p}]),\quad\hat{I}_{p}=\mathcal{G}(z_{T}),$$

## Where Pθ Denotes The Denoising Process, Zt Denotes The High-Level Representations From Vae Of
Ldms (Metzer Et Al., 2023) At Denoising Step T, G Refers To The Image Generator And T Denotes The
Final Step. Please Refer To **Preliminary In Appendix** For Detials. 3.2.1 Perturbed Camera Trajectory

To explore those unobserved regions of the scene by the input training views, we introduce a perturbed camera trajectory that benefits pseudo camera pose sampling. Specifically, uniform poses are first sampled in a circular camera trajectory generated from the camera poses of training views (Ovren & Forss ´ en, 2018) as candidate pseudo camera pose positions. Afterward, each ´
pseudo pose ciis defined by a position ti and a quaternion on the rotation qi averaged from two training cameras. In particular, our strategy applies periodic perturbations alongside the x-axis and y-axis of the camera coordinate system using the sin function to it, which may cover horizontally and vertically distributed unobserved regions, as follows:

$$\mathbf{c}_{i}=[t_{i}+A s i n(2\pi f\cdot t_{i})\begin{bmatrix}1\\ 1\\ 0\end{bmatrix},q_{i}],$$

where A represents the x-axis and y-axis perturbation amplitudes, and f denotes the wave frequency. We set f = 1.0, and A = 2.0 as the trade-off between exploiting unobserved regions and avoiding generative model hallucination.

$$(11)$$

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

$$\mathbf{j}\,\mathrm{AUSSIAN}$$

## 3.2.3 Gaussian Optimization

The proposed method adopts a two-phase optimization for 3D Gaussians. At the first phase, *i.e.*,
during the first m iterations, Gaussians are optimized solely using an image reconstruction loss Limg between the synthesized views and training views; At the second phase, *i.e.*, during the following iterations, pseudo camera poses are sampled based on our camera trajectory perturbation strategy to generate the corresponding pseudo views, which contribute to the optimization of 3D Gaussians. Overall, the training loss is formulated as follows, Similar to the generative points, the generative hallucination in the complete pseudo views ˆIp could result in multi-view inconsistency and appearance distortion in the rendered details, as shown in Figure 4. To attenuate this impact, we design a generative consistency loss composed of two key terms on constraining those regions' representations with appearance distortion and improving the scene completion capability while maintaining the multi-view consistency.

Specifically, the first loss term is based on a pixellevel confidence mask Mr, which firstly evaluates the appearance gap ∆C between the color C of Ip and ˆIp via the L2-norm, formulated for a pixel (u, v) as follows,

$$\Delta_{C}(u,v)=\|C_{I_{p}}(u,v)-C_{\hat{I}_{p}}(u,v)\|$$
$\psi$). 
(u, v)∥ (12)
Subsequently, we generated an adaptive threshold T(*u, v*) to robustly identify significant distortion. Specifically, a Gaussian blur kernel is adopted to generate the local mean µ∆(*u, v*) and standard deviation σ∆(*u, v*) as the local statistics of the gap ∆C , and T(*u, v*) are derived as follows, T(u, v) = µ∆(*u, v*) + δ2 · σ∆(*u, v*), (13)
where δ2 = 20 denotes as a variance coefficient. Finally, the binary confidence mask Mr is obtained by applying the adaptive threshold to the difference map:

$M_{r}(u,v)=\begin{cases}1&\text{if}\Delta_{C}(u,v)>T(u,v),\\ 0&\text{otherwise}.\end{cases}$
To further improve the coherence and smoothness of Mr for training stability, a sequence of expansion K1, erosion K2, and connected components filtering operations is performed as follows:
Ri, (15)

$M_{r}^{\prime}=(M_{r}\oplus\mathcal{K}_{1})\ominus\mathcal{K}_{2},\quad\hat{M}_{r}=\bigcup_{\begin{subarray}{c}R_{i}\in\mathcal{R}\mid\text{Area}(R_{i})\geq\delta_{3}\\ \end{subarray}}R_{i},$
where R denotes the set of connected components in M′r, and δ3 = 8 refers to a threshold.

Afterward, the first loss term is formulated to constrain the appearance of those regions identified by Mˆr and suppress the hallucination using the L1 loss as follows,

$${\mathcal{L}}_{r e g}(I_{p},{\hat{I}}_{p})=\|I_{p}-{\hat{I}}_{p}\|_{1}\odot{\hat{M}}_{r},$$

The second loss term provides a feature-level constraint between Ip and ˆIp based on a VGG network Simonyan & Zisserman (2015), to benefit structural completion and keep multi-view consistency, as follows:
$$\mathcal{L}_{str}(I_{p},\hat{I}_{p})=\mathcal{L}_{L\,P\,IPS}(I_{p},\hat{I}_{p}).\tag{1}$$  Hence, generative consistency loss is formulated with the weight coefficient $\alpha=10.0$ as follows,  $$\mathcal{L}_{GC}=\mathcal{L}_{img}+\alpha(\mathcal{L}_{reg}+\mathcal{L}_{str}),\tag{1}$$
where, Limg represents reconstruction loss Kerbl et al. (2023) between Ip and ˆIp using λ = 0.2,
ˆIp). (19)
$${\mathcal{L}}_{i m g}(I_{p},{\hat{I}}_{p})={\mathcal{L}}_{1}(I_{p},{\hat{I}}_{p})+\lambda{\mathcal{L}}_{D S S I M}(I_{p},{\hat{I}}_{p}).$$
$$(12)$$
$$(14)$$
$$(15)$$

![5_image_0.png](5_image_0.png)

where k represents the iteration index and β denotes a weight coefficient. We set β = 0.1 in practice.

$$(16)$$
$$(17)$$ vs. 
$$(18)$$
$$(19)$$
al. (2023) between $I$. 
$${\mathcal{L}}=\left\{\begin{array}{c l}{{{\mathcal{L}}_{i m g},}}&{{\mathrm{if}\;k<m,}}\\ {{{\mathcal{L}}_{i m g}+\beta{\mathcal{L}}_{G C},}}&{{\mathrm{otherwise},}}\end{array}\right.$$
$$(20)$$

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

| second-best , and third-best scores are highlighted. Method PSNR↑ SSIM↑                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | LPIPS↓   | AVGE↓   |    |    |    |    |    |    |    |    |    |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------|----|----|----|----|----|----|----|----|----|
| 3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | 6        | 9       | 3  | 6  | 9  | 3  | 6  | 9  | 3  | 6  | 9  |
| SparseNeRF (Wang et al., 2023) 19.86 23.26 24.27 0.714 0.741 0.781 0.243 0.235 0.228 0.127 0.117 0.113 ReconFusion (Wu et al., 2024) 21.34 24.25 25.21 0.724 0.815 0.848 0.203 0.152 0.134 0.110 0.090 0.081 MuRF (Xu et al., 2024) 21.26 23.54 24.66 0.722 0.796 0.836 0.245 0.199 0.164 0.118 0.103 0.094 FrugalNeRF (Lin et al., 2025) 19.87 - - 0.610 - - 0.300 - - 0.125 - - CAT3D (Gao et al., 2024) 21.58 24.71 25.63 0.731 0.833 0.860 0.181 0.121 0.107 0.097 0.067 0.059 3DGS (Kerbl et al., 2023) 15.52 19.45 21.13 0.405 0.627 0.715 0.408 0.268 0.214 0.209 0.154 0.137 FSGS (Zhu et al., 2024) 20.31 24.20 25.32 0.652 0.811 0.856 0.288 0.173 0.136 0.136 0.095 0.082 DNGaussian (Li et al., 2024a) 19.12 22.18 23.17 0.591 0.755 0.788 0.294 0.198 0.180 0.132 0.110 0.105 BinoGS (Han et al., 2024) 21.44 24.87 26.17 0.751 0.845 0.877 0.168 0.106 0.090 0.101 0.061 0.051 IPSM (Wang et al., 2024) 20.44 23.91 25.13 0.702 0.818 0.855 0.207 0.135 0.111 0.109 0.080 0.071 ReconX (Liu et al., 2025) 21.05 - - 0.768 - - 0.178 - - 0.111 - - GenCoGS (Ours) 22.13 25.61 26.64 0.762 0.857 0.880 0.164 0.108 0.090 0.084 0.051 0.044 |          |         |    |    |    |    |    |    |    |    |    |

Table 2: Comparison of GenCoGS and other methods regarding performance on the

DTU (Jensen et al., 2014) under 3-view setting.

Method PSNR↑ SSIM↑ LPIPS↓ **AVGE**↓ SparseNeRF (Wang et al., 2023) 19.47 0.829 0.183 0.120 ReconFusion (Wu et al., 2024) 20.74 0.875 0.124 0.109 MuRF (Xu et al., 2024) 21.31 0.885 0.127 0.103 CAT3D (Gao et al., 2024) 22.02 0.844 0.121 0.099 FSGS (Zhu et al., 2024) 17.34 0.818 0.169 0.123 DNGaussian (Li et al., 2024a) 18.91 0.790 0.176 0.124 BinoGS (Han et al., 2024) 20.71 0.862 0.111 0.096 IPSM (Wang et al., 2024) 19.99 0.856 0.121 0.077 ReconX (Liu et al., 2025) 19.78 0.476 0.378 0.142 GenCoGS (Ours) 23.11 0.910 0.082 0.049

![6_image_0.png](6_image_0.png)

## 4 Experiments

Following previous methods (Zhu et al., 2024; Paliwal et al., 2024), we conducted experiments on three benchmark datasets: LLFF (Mildenhall et al., 2019), DTU (Jensen et al., 2014), and Shiny (Wizadwongsa et al., 2021) with 3, 6, and 9 training views as few-shot settings. We implemented GenCoGS using the PyTorch framework, with the initial point cloud computed from SfM in FSGS (Zhu et al., 2024). During optimization, we densify the Gaussians every 100 iterations and start densification after 1000 iterations. The total optimization steps are set to 5000, and we set the GCGO after m = 4, 000 iterations. For hyper-parameters, we set k = 3 and δ1 = 1.0 in *GCGI*, we set the wave frequency f = 1.0, perturbation amplitude A = 2.0, δ2 = 20, and δ3 = 8 in *GCGO*,
and the loss weight coefficients are set as α = 10.0, and β = 0.1 for Gaussian optimization. All results are obtained using a NVIDIA A6000 GPU. Furthermore, please refer to the **Appendix for** details on Datasets and Evaluation Metrics.

## 4.1 Quantitative Comparison

As shown in Table 1, 2 and 3, our GenCoGS consistently outperformed other representative fewshot NVS methods, nearly in all metrics. On the LLFF dataset, GenCoGS achieved improvements of 0.55 dB / 0.74 dB / 0.47 dB in PSNR, 0.011 / 0.012 / 0.003 in SSIM, and 0.013 / 0.029 / 0.027 in AVGE under 3-view / 6-view / 9-view settings, respectively, compared to the methods with second-best performances. On the DTU dataset, the improvements by GenCoGS under 3-view setting were 2.40 dB in PSNR, 0.025 in SSIM, 0.029 in LPIPS, and 0.045 in AVGE compared to the second-best 3DGS-based method. Please refer to **Appendix** for detailed results on the DTU dataset. Notably, the substantial boosts over other diffusion-based methods (Wang et al., 2024; Wu et al.,
Table 3: Comparison of GenCoGS and other methods regarding performance on the Shiny (Jensen et al., 2014) under 3-view setting.

Method PSNR↑ SSIM↑ LPIPS↓ AVGE ↓
RegNeRF 18.10 0.574 0.378 0.136 FreeNeRF 18.65 0.586 0.360 0.127 SparseNeRF 18.81 0.591 0.354 0.124 3D-GS 17.83 0.547 0.385 0.142 FSGS 19.63 0.612 0.327 0.111 GenCoGS (Ours) 21.10 0.692 0.202 0.099 378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Figure 7: Visualization of example images in the ablation studies on LLFF under 3-view setting.

2024; Gao et al., 2024) achieved by GenCoGS stem from the hallucination attenuation capability of our strategies. On the more challenging Shiny dataset, GenCoGS also outperformed existing methods, achieving improvements of 1.47 dB in PSNR, 0.080 in SSIM, 0.125 in LPIPS, and 0.012 in AVGE under the 3-view setting, which further validates the superiority of GenCoGS in high-fidelity few-shot NVS.

## 4.2 Qualitative Comparison

We visualized example views synthesized by GenCoGS, alongside DNGaussian (Li et al., 2024a), BinoGS (Han et al., 2024) and the diffusion-based ViewCrafter (Yu et al., 2024a), on both DTU and LLFF datasets under 3-views setting, as shown in Figure 5 and 6. DNGaussian and BinoGS attempted to exploit priors on the structure and appearance of the input training views, but resulted in considerable ambiguity, *e.g.*, first and second rows in Figure 6, due to the lack of scene completion capability. Furthermore, the results of ViewCrafter (Yu et al., 2024a) suggest that its generative completion pipeline toward synthesized views suffers from significant generative model hallucination and unsatisfactory scene reconstruction capability via synthesized view completion, as shown in Figure 6 (e) highlighted regions. Integrating both generative completion-based strategies, our GenCoGS provided a high-quality scene using the complete initial Gaussians followed by the optimization additionally guided by pseudo views less influenced by generative model hallucination. In particular, our GCGO strategy effectively filled the hollows within the synthesized views, *e.g.*, the highlighted regions in Figure 6 second and third rows. These examples further demonstrate the improvements of GenCoGS across different benchmark datasets, demonstrating its consistency and effectiveness in delivering highfidelity few-shot NVS results.

## 4.3 Ablation Studies

To investigate the contributions of individual strategies, we conducted ablation studies on the LLFF dataset under the 3-view setting. The results in Table 4 indicate that each strategy positively impacts few-shot NVS performance, with the combination of both achieving the best performance.

Impact of the GCGI Strategy Compared to the baseline, adopting the GCGI strategy reached the improvements of 0.66 dB, 0.024, 0.016 and 0.009 in PSNR, SSIM, LPIPS and AVGE, respectively. These results suggest that the complete initial Gaussians with the complete point cloud from the Figure 6: Visualization of example synthesized views by GenCoGS, DNGaussian (Li et al., 2024a),

![7_image_1.png](7_image_1.png) BinoGS (Han et al., 2024) and ViewCrafter (Yu et al., 2024a) on LLFF under the 3-view setting.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| PSNR SSIM LPIPS AVGE                         |                         |
|----------------------------------------------|-------------------------|
| Baseline                                     | 20.79 0.733 0.184 0.096 |
| + GCGI                                       | 21.45 0.757 0.168 0.087 |
| + GCGO                                       | 21.65 0.752 0.184 0.088 |
| + GCGI + GCGO (Ours) 22.13 0.762 0.164 0.084 |                         |

![8_image_0.png](8_image_0.png) diffusion model using different A.
Table 4: Ablation of our GCGI and GCGO strategies on LLFF under 3-view.

Table 5: Ablation of pseudo camera sampling and LGC in GCGO on LLFF under 3-view.

| Sampling            | LGC PSNR SSIM LPIPS   |                   |
|---------------------|-----------------------|-------------------|
| Random              | !                     | 21.83 0.755 0.188 |
| Camera Trajectory % | 21.59 0.749 0.181     |                   |
| Camera Trajectory ! | 22.13 0.762 0.164     |                   |

| Sampling   | w/ CPG   | w/ CPF   | PSNR   | SSIM   | LPIPS   |
|------------|----------|----------|--------|--------|---------|
| Full       | 21.65    | 0.752    | 0.184  |        |         |
| Full       | !        | 22.04    | 0.760  | 0.178  |         |
| Full       | !        | !        | 22.13  | 0.762  | 0.164   |
| 1/4        | 21.24    | 0.730    | 0.199  |        |         |
| 1/4        | !        | 21.61    | 0.733  | 0.195  |         |
| 1/4        | !        | !        | 21.78  | 0.741  | 0.191   |

GCGI strategy is capable of avoiding floating artifacts in those scene regions with details, as also illustrated in Figure 7. As shown in Figure 3, our CPG and CPF modules work jointly to refine the sparse initial point cloud into a more complete one while effectively removing outliers to avoid hallucination. As shown in Table 6, both modules consistently contributed to improvements even when the quality of the initial point cloud P0 was degraded by randomly sampling only a quarter of the points. This demonstrates the strong generalization capability and robustness of our GCGI strategy. Impact of the GCGO Strategy Compared to the baseline, leveraging the GCGO strategy achieved the improvements of 0.86 dB, 0.019, and 0.008 in PSNR, SSIM, and AVGE, respectively.

As illustrated in Figure 7, Gaussians optimized using the GCGO strategy mitigated hollows and floating artifacts, which benefited the synthesis of high-fidelity views.

In particular, as shown in Table 5, the pseudo camera poses sampled from a perturbed camera trajectory facilitated better scene completion in unobserved regions compared to randomly sampled poses. It is noteworthy that our LGC further improved performance by focusing on reducing generative model hallucination. Furthermore, we identified a critical see-saw effect between generative model hallucination and unobserved region exploration based on the perturbed camera trajectory. As shown in Figure 8, the I2V model generated significant hallucination when trying to cover more unobserved regions, leading to low-fidelity outcomes. Hence, we set A = 2.0 as a balanced trade-off in our experiments. Furthermore, please kindly refer to **Appendix** for additional experiments results.

## 5 Conclusions

In this paper, we addressed a critical limitation of existing 3DGS-based few-shot NVS methods, i.e., unsatisfactory scene completion capability caused by the overdependence on the observed regions of sparse training views. Our unified method, GenCoGS, enhances scene completion by incorporating two generative completion-based strategies focusing on Gaussian initialization and optimization. For Gaussian initialization, GenCoGS generates and filters complementary points to establish a complete point cloud with refined structural and appearance information; For Gaussian optimization, GenCoGS leverages an image-to-video (I2V) diffusion model to generate complete pseudo views, providing effective guidance over unobserved scene regions while attenuating generative model hallucination. By enabling accurate and coherent scene completion, GenCoGS outperformed representative 3DGS-based few-shot NVS methods and achieved significant improvements, demonstrating the superiority of GenCoGS.

## Reproducibility Statement

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 To ensure the reproducibility of our work, we provided comprehensive details on our methodology and experiments. The motivation and architectural design of our proposed strategies are elaborated in Section 3. A complete description of the experiments implementation, including all hyperparameter configurations, was provided in Section 4. To justify our hyperparameter choices, we also present extensive ablation studies. The source code will be made publicly available under an open-source license upon the acceptance of this paper. We also performed the video qualitative visualizations in Supplementary Materials, please kindly refer to them for comparison with other methods.

## References

Sumukh K Aithal, Pratyush Maini, Zachary C. Lipton, and J. Zico Kolter. Understanding hallucinations in diffusion models through mode interpolation, 2024. URL https://arxiv.org/ abs/2406.09358.

A. Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, and Dominik Lorenz. Stable video diffusion: Scaling latent video diffusion models to large datasets. *ArXiv*, abs/2311.15127, 2023. URL https://api.semanticscholar.org/ CorpusID:265312551.

Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su.

Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 14124–14133, 2021.

Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3d: Video diffusion models are effective 3d generators. *arXiv preprint arXiv:2403.06738*, 2024.

Xiyang Dai, Yinpeng Chen, Jianwei Yang, Pengchuan Zhang, Lu Yuan, and Lei Zhang. Dynamic detr: End-to-end object detection with dynamic attention. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 2988–2997, 2021.

Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12882–12891, 2022.

Y. Eldar, M. Lindenbaum, M. Porat, and Y.Y. Zeevi. The farthest point strategy for progressive image sampling. *IEEE Transactions on Image Processing*, 6(9):1305–1315, September 1997. ISSN
1941-0042. doi: 10.1109/83.623193. URL http://dx.doi.org/10.1109/83.623193.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam ¨
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https://arxiv.org/abs/ 2403.03206.

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo Martin-Brualla, Pratul Srinivasan, Jonathan T. Barron, and Ben Poole. Cat3d: Create anything in 3d with multi-view diffusion models, 2024. URL https://arxiv.org/abs/2405.10314.

Liang Han, Junsheng Zhou, Yu-Shen Liu, and Zhizhong Han. Binocular-guided 3d gaussian splatting with view consistency for sparse view synthesis. *Advances in Neural Information Processing* Systems, 37:68595–68621, 2024.

Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 406–413, 2014.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- ¨
ting for real-time radiance field rendering. *ACM Transactions on Graphics*, 42(4), July 2023. URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.

Jari Korhonen and Junyong You. Peak signal-to-noise ratio revisited: Is simple beautiful?

In *2012 Fourth International Workshop on Quality of Multimedia Experience*. IEEE, July 2012. doi: 10.1109/qomex.2012.6263880. URL http://dx.doi.org/10.1109/QoMEX. 2012.6263880.

Oliver Kramer. *K-Nearest Neighbors*, pp. 13–23. Springer Berlin Heidelberg, 2013. ISBN
9783642386527. doi: 10.1007/978-3-642-38652-7 2. URL http://dx.doi.org/10. 1007/978-3-642-38652-7_2.

Jona´s Kulh ˇ anek, Erik Derner, Torsten Sattler, and Robert Babu ´ ska. Viewformer: Nerf-free neural ˇ
rendering from few images using transformers. In *European Conference on Computer Vision*, pp. 198–216. Springer, 2022.

Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20775–20785, 2024a.

Jinke Li, Xiao He, Chonghua Zhou, Xiaoqiang Cheng, Yang Wen, and Dan Zhang. Viewformer: Exploring spatiotemporal modeling for multi-view 3d occupancy perception via view-guided transformers. In *European Conference on Computer Vision*, pp. 90–106. Springer, 2024b.

Chin-Yang Lin, Chung-Ho Wu, Chang-Han Yeh, Shih-Han Yen, Cheng Sun, and Yu-Lun Liu. Frugalnerf: Fast convergence for extreme few-shot novel view synthesis without learned priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11227–11238, June 2025.

Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model, 2025. URL https://arxiv.org/abs/2408.16767.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick.

Zero-1-to-3: Zero-shot one image to 3d object. In *Proceedings of the IEEE/CVF international* conference on computer vision, pp. 9298–9309, 2023.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20654–20664, 2024.

Luke Melas-Kyriazi, Iro Laina, Christian Rupprecht, Natalia Neverova, Andrea Vedaldi, Oran Gafni, and Filippos Kokkinos. Im-3d: Iterative multiview diffusion and reconstruction for high-quality 3d generation. *arXiv preprint arXiv:2402.08682*, 2024.

Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, and Daniel Cohen-Or. Latent-nerf for shape-guided generation of 3d shapes and textures. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12663–12673, 2023.

Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. *ACM Transactions on Graphics (ToG)*, 38(4):1–14, 2019.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp. 5480–5490, 2022.

Hannes Ovren and Per-Erik Forss ´ en. Trajectory representation and landmark projection for ´
continuous-time structure from motion, 2018. URL https://arxiv.org/abs/1805. 02543.

Avinash Paliwal, Wei Ye, Jinhui Xiong, Dmytro Kotovenko, Rakesh Ranjan, Vikas Chandra, and Nima Khademi Kalantari. Coherentgs: Sparse novel view synthesis with coherent 3d gaussians. In *European Conference on Computer Vision*, pp. 19–37. Springer, 2024.

Joel Pearson. The human imagination: the cognitive neuroscience of visual mental imagery. Nature Reviews Neuroscience, 20(10):624–634, August 2019. ISSN 1471-0048. doi: 10.1038/ s41583-019-0202-9. URL http://dx.doi.org/10.1038/s41583-019-0202-9.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. *arXiv preprint arXiv:2209.14988*, 2022.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pp. 8748–8763. PmLR, 2021.

Kerui Ren, Lihan Jiang, Tao Lu, Mulin Yu, Linning Xu, Zhangkai Ni, and Bo Dai. Octree-gs:
Towards consistent real-time rendering with lod-structured 3d gaussians. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-motion revisited. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. *arXiv preprint arXiv:2308.16512*, 2023.

K Simonyan and A Zisserman. Very deep convolutional networks for large-scale image recognition.

pp. 1–14. Computational and Biological Learning Society, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv* preprint arXiv:2010.02502, 2020.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Jiaxiang Tang, Jiawei Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Dreamgaussian: Generative gaussian splatting for efficient 3d content creation. *arXiv preprint arXiv:2309.16653*, 2023a.

Junshu Tang, Tengfei Wang, Bo Zhang, Ting Zhang, Ran Yi, Lizhuang Ma, and Dong Chen. Makeit-3d: High-fidelity 3d creation from a single image with diffusion prior. In *Proceedings of the* IEEE/CVF international conference on computer vision, pp. 22819–22829, 2023b.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS'17, pp. 6000–6010, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9065–9076, 2023.

Qisen Wang, Yifan Zhao, Jiawei Ma, and Jia Li. How to use diffusion priors under sparse views?

In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL https://openreview.net/forum?id=i6BBclCymR.

Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E Sarma, Michael M Bronstein, and Justin M Solomon.

Dynamic graph cnn for learning on point clouds. *ACM Transactions on Graphics (tog)*, 38(5): 1–12, 2019.

Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. *IEEE Transactions on Image Processing*, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861.

Suttisak Wizadwongsa, Pakkapon Phongthawee, Jiraphon Yenphraphai, and Supasorn Suwajanakorn. Nex: Real-time view synthesis with neural basis expansion. In *Proceedings of the* IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8534–8543, 2021.
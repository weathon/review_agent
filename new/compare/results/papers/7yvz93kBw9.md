# D 2Gs: Depth-And-Density Guided Gaussian Splatting For Stable And Accurate Sparse- View Reconstruction

Meixi Song1,2∗ Xin Lin3∗ Dizhe Zhang2 † ‡ Haodong Li3 Xiangtai Li4 Bo Du5 Lu Qi2,5‡
1 Tsinghua University 2Insta360 Research 3 University of California, San Diego 4 Nanyang Technological University 5 Wuhan University

![0_image_0.png](0_image_0.png)

Sparse views (3 views) - Ours Gaussian Primitives Rendered **Image**
Figure 1: Comparison of Gaussian primitives and rendered images between dense views (55 views) and sparse views (3 views) settings. Overfitting occurs in the near field (green box), while underfitting appears in the far field (red box). The number of Gaussian primitives in the corresponding field is shown below the images.

## Abstract

Recent advances in 3D Gaussian Splatting (3DGS) enable real-time, high-fidelity novel view synthesis (NVS) with explicit 3D representations. However, performance degradation and instability remain significant under sparse-view conditions. In this work, we identify two key failure modes under sparse-view conditions: overfitting in regions with excessive Gaussian density near the camera, and underfitting in distant areas with insufficient Gaussian coverage. To address these challenges, we propose a unified framework D2GS, comprising two key components: a Depth-and-Density Guided Dropout strategy that suppresses overfitting by adaptively masking redundant Gaussians based on density and depth, and a Distance-Aware Fidelity Enhancement module that improves reconstruction quality in under-fitted far-field areas through targeted supervision. Moreover, we introduce a new evaluation metric to quantify the stability of learned Gaussian distributions, providing insights into the robustness of the sparse-view 3DGS.

Extensive experiments on multiple datasets demonstrate that our method significantly improves both visual quality and robustness under sparse view conditions. The project page can be found at: https://insta360-research-team.github.io/DDGS- website/.

## 1 Introduction

Recently, novel view synthesis (NVS) (Kerbl et al., 2023; Lin et al., 2025; Yu et al., 2024; Ye et al., 2024; Lee et al., 2024; Niedermayr et al., 2024; Zhang et al., 2024b; Zhou et al., 2024) and its applications have witnessed significant progress due to advances in 3D Gaussian splatting (3DGS), which provides a favorable trade-off between reconstruction quality and computational efficiency. While previous methods perform well under densely-sampled multi-view settings, acquiring such data in real-world scenarios is often impractical. This limitation has led to growing interest in the sparse-view reconstruction task (Wang et al., 2023a; Yang et al., 2023; Truong et al., 2023; Zhang et al., 2024a; Bao et al., 2025), where only a few input views are available, posing additional challenges for accurate and consistent novel view synthesis. To address this challenge, existing works (Park et al., 2025) suggest that 3DGS models trained on sparse views tend to overfit a limited set of Gaussian primitives. Therefore, they typically adopt a dropout strategy during training, uniformly dropping Gaussian primitives to reduce overreconstruction. However, we observe that uniform dropout can inadvertently hurt both well-fitted and under-fitted regions, thereby degrading reconstruction quality in critical areas, as shown in the bottom-right of Figure 1. Moreover, the visualizations of Gaussian distributions in dense- and sparse-view settings (55 and 3 input images) reveal two key issues: over-reconstruction in texturerich and near-camera regions, leading to dense, aliased Gaussians and rendering artifacts; underreconstruction in distant areas, where sparse Gaussians fail to capture structural details, resulting in blurry reconstructions. Based on these observations, we shift our focus to enhancing and evaluating the robustness of 3DGS models under sparse-view settings through both methodological design and evaluation metric. The proposed D2GS method aims to dynamically adjust the degree of reconstruction based on depth and density information, while the evaluation metric is used to assess the robustness of 3DGS models in a consistent training setting. Specifically, the D2GS mainly consists of two key components: a Depthand-Density guided Dropout (DD-Drop) mechanism and Distance-Aware Fidelity Enhancement (DAFE), to improve the stability and spatial completeness of scene reconstruction under sparseview settings. DD-Drop assigns each Gaussian a dropout score based on local density and camera distance, indicating regions prone to overfitting. High-scoring Gaussians would be dropped with a higher probability to suppress aliasing and improve rendering fidelity. In addition, DAFE avoids underfitting by boosting supervision in distant regions using depth priors. To further assess the robustness of 3DGS models under sparse-view constraints, we propose a novel evaluation metric, Inter-Model Robustness (IMR), which measures the stability of the learned 3D Gaussian distributions. Specifically, IMR quantifies the consistency of independently trained models by comparing their output Gaussian distributions under identical input settings, reflecting robustness to initialization and training noise. This distribution-based metric complements traditional image-space metrics such as PSNR and SSIM, providing a more direct evaluation of 3D representation quality. We comprehensively evaluate the proposed D2GS framework on the LLFF and Mip-
NeRF360 datasets. Extensive ablation studies further validate the effectiveness of each proposed module. In summary, the main contributions can be summarized as follows:
- We systematically analyze the failure modes of 3DGS in sparse-view settings, revealing consistent patterns of overfitting and underfitting in Gaussian primitives.

- Based on these insights, we propose a unified D2GS framework that incorporates two complementary modules: a Depth-and-Density Guided Dropout mechanism to suppress overfitting in redundant and dense regions, and a Distance-Aware Fidelity Enhancement module to enhance reconstruction fidelity in underfitting areas.

- To better evaluate the quality of 3D Gaussian representations, we introduce a Gaussiandistribution-based metric to assess robustness and fidelity beyond conventional 2D evaluations.

Extensive experiments demonstrate that D2GS achieves state-of-the-art novel view synthesis while yielding more robust 3D reconstructions.

## 2 Related Work

Novel View Synthesis. Novel View Synthesis (NVS) aims to generate unseen views of a scene from given images. Neural Radiance Fields (NeRF) (Mildenhall et al., 2021) reconstruct scenes as

![2_image_0.png](2_image_0.png)

F
a eld G
T
G
T
Distance-Aware Fidelity *Enhancement* 1 +  + 
F a eld R G
B
R
e n d e R G B
Render
Figure 2: The overall framework of D2GS consists of a Depth-and-Density Guided Dropout (DD-
Drop) module and a Distance-Aware Fidelity Enhancement (DAFE) module. The DD-Drop module adaptively removes Gaussian primitives based on depth and density indication through a dual localglobal mechanism. The DAFE module enhances supervision for far-field regions using distanceaware masks. implicit volumetric radiance fields, with many works improving rendering quality (Barron et al., 2021; 2022; Verbin et al., 2022; Chen et al., 2022; Barron et al., 2023) and efficiency (Garbin et al., 2021; Yu et al., 2021; Fridovich-Keil et al., 2022; Muller et al. ¨ , 2022; Sun et al., 2022; Li et al., 2023; Wang et al., 2023b; Hu et al., 2023). Despite impressive visual fidelity, NeRF-based methods suffer from high computational costs and long training times. To address these limitations, 3D Gaussian Splatting (3DGS) represents scenes with Gaussian primitives and renders via differentiable splatting, achieving real-time synthesis. Building on this, recent methods further enhance reconstruction in diverse 3D vision tasks (Yu et al., 2024; Ye et al., 2024; Lee et al., 2024; Kheradmand et al., 2024; Shi et al., 2025; Niedermayr et al., 2024; Zhang et al., 2024b; Yue et al., 2025). Novel View Synthesis with Sparse Views. NeRF- and 3DGS-based methods have achieved remarkable performance with dense views, but collecting many images is often impractical in realworld scenarios, resulting in significant performance degradation for conventional approaches. To mitigate this, previous NeRF variants introduce architectural enhancements such as semantic consistency (Jain et al., 2021; Qi et al., 2022; 2023), depth supervision (Deng et al., 2022; Niemeyer et al., 2022; Yang et al., 2025; Roessle et al., 2022; Qi et al., 2024; Wang et al., 2023a), frequency regularization (Yang et al., 2023), and cross-view consistency (Truong et al., 2023; Qi et al., 2021). With more efficient 3DGS frameworks, recent methods improve scene understanding via pseudoview generation (Zhang et al., 2024a), address sparse initialization with additional priors (Bao et al.,
2025), and mitigate overfitting to training views (Park et al., 2025; Chen et al., 2025). Recently, some feed-forward methods further advance sparse-view NVS: PixelSplat (Charatan et al., 2024) predicts 3D Gaussian parameters directly from images, MVSplat (Chen et al., 2024) incorporates multi-view stereo cues to improve depth reliability under sparse inputs, and HiSplat (Tang et al., 2024) adopts a hierarchical Gaussian representation to enhance geometric detail and view consistency.

## 3 Proposed Method

Figure 2 presents the overall pipeline of the proposed D2GS, which takes sparse-view images as input and generates initial point clouds and camera poses through Structure-from-Motion (SfM). During training, two key modules are introduced: Depth-and-Density Guided Dropout, which regularizes near-field Gaussians via depth- and density-aware dropout; and Distance-Aware Fidelity Enhancement, which strengthens far-field supervision using depth-derived masks predicted by a monocular depth estimator. In the following subsections, we detail the motivation, design, and function of each component, and introduce a dedicated robustness metric for 3DGS under sparse supervision.

## 3.1 Motivation

Our motivation arises from a comprehensive analysis of key factors affecting the performance and stability of sparse-view 3D Gaussian Splatting (3DGS). Figure 1 compares the trained Gaussian primitives under dense- and sparse-view settings. It reveals a significant spatial imbalance: Gaussians are over- and under-fitted in near-and far-field regions. Specifically, in near-field regions, models trained with only three views (e.g., DropGaussian) produce a much higher density of Gaussians than the dense-view model. In the green box, previous methods generated 11,450 Gaussian primitives, far exceeding the 6,112 Gaussian primitives of the dense view, indicating clear local overfitting. After rendering, we observe that local overreconstruction in the near field can introduce artifacts that propagate globally, which significantly degrade the rendered image quality. In contrast, far-field regions suffer from underfitting due to limited visibility in training data and frequent occlusion by densely populated near-field Gaussians. In the red box, previous methods generated 3,082 Gaussian primitives, which is noticeably fewer than the 5,224 Gaussian primitives of the dense view, preventing the optimizer from effectively supervising these regions. Therefore, the model is unable to capture accurate geometry and texture in distant areas, leading to blurred or discontinuous structures in the rendered outputs.

## 3.2 Depth-And-Density Guided Dropout

As observed, near-field regions with high Gaussian density are more susceptible to overfitting. To alleviate this, we propose a spatially adaptive dropout strategy guided by both depth and density. Furthermore, to tackle the problem from both continuous and discrete perspectives, we incorporate two complementary penalty mechanisms operating from local and global viewpoints. We first introduce the local dropout mechanism, which evaluates the spatial variation of each Gaussian primitive i = 1, 2*, . . . , N* based on its depth di (Euclidean distance to the camera) and local density ρi (estimated via k-nearest neighbors). Both di and ρi are processed with min–max normalization to obtain the depth score ˜di and density score ρ˜i, respectively. The dropout score Siis then computed as a weighted combination of the two:
Si = ωdepth ˜di + ω*density* ρ˜i, (1)
where ω*depth* and ω*density* are weighting coefficients that satisfy ωdepth + ωdensity = 1. This continuous scoring function captures fine-grained local spatial variation, but local information alone is insufficient to characterize overfitting patterns across the entire scene. The global mechanism is motivated by a depth-induced imbalance: regions at different depth ranges receive markedly different visibility, leading to significantly different overfitting behaviors at a global level. To model this pattern, we further divide the point cloud into three depth-based layers: near, middle, and far. The division is determined by the first and second tertiles of the depth distribution, denoted as thresholds Dnear and Dmiddle. Here, our method aims to introduce depth prior information without strongly relying on such partitioning. Each layer is assigned a different attenuation factor, where λmiddle and λfar satisfy 0 < λfar < λmiddle < 1, and the near layer uses no attenuation. This combination of locally continuous and globally discrete mechanisms facilitates fine-grained local tuning while preserving global structural coherence, ultimately leading to efficient control over the overall spatial distribution. This combined design controls the probability of per-Gaussian dropout in a soft and progressive manner, and the corresponding formulation is given by:

$$P_{i}=1$$
$$\left\{\begin{array}{l l}{{S_{i},}}&{{d_{i}\leq D_{n e a r},}}\\ {{\lambda_{m i d d l e}\,S_{i},}}&{{D_{n e a r}<d_{i}\leq D_{m i d d l e},}}\\ {{\lambda_{f a r}\,S_{i},}}&{{d_{i}>D_{m i d d l e},}}\end{array}\right.$$
(2)
where Piindicates dropout rate of i th Gaussian primitive. Based on experimental experience, we set λfar = 0.3 and λmiddle = 0.7 in practice.

As the training progresses, the number of Gaussian primitives increases through continuous optimization and refinement. To maintain effective regularization, we gradually increase the dropout ratio over training iterations using a time-dependent global rate r(t), which progressively increases the fraction of Gaussians discarded in later training stages:

$$r(t)=r_{\mathrm{min}}+\left(r_{\mathrm{max}}-r_{\mathrm{min}}\right){\frac{\operatorname*{min}(t,T)}{T}},$$

$$({\mathfrak{I}})$$

T, (3)
where t denotes the current training step, rmax and rmin are the minimum and maximum dropout rates, and T is the total number of training steps. 3.3 DISTANCE-AWARE FIDELITY ENHANCEMENT To address underfitting in distant regions with missing details, we introduce a Distance-Aware Fidelity Enhancement (DAFE) module that reinforces dedicated supervision in these areas. Specifically, we first employ a monocular depth estimation model to generate depth maps for each input image. These maps are then processed using a depth-thresholding strategy to construct a binary mask that separates the image into near and far regions. The binary distant-region mask Mdis ∈ {0, 1}
H×W is constructed as follows:

$$M_{\mathrm{dis}}(x,y)={\begin{cases}1,&{\mathrm{if~}}D(x,y)>\tau\,D_{m a x},\\ 0,&{\mathrm{otherwise}},\end{cases}}$$

where D(*x, y*) is the estimated depth value at pixel (x, y), Dmax is the maximum depth value, and τ is the predefined depth threshold.

We then leverage the distant-region mask Mdis(*x, y*) to modulate the training objective, with the aim of amplifying the supervision signal in under-fitted far-field regions. Specifically, the mask is applied to both the ground-truth image and the rendered output to isolate distant content. A dedicated distance-enhanced loss is computed by measuring their difference in these masked regions:

$$(4)$$
$${\cal L}_{\mathrm{DAFE}}=\frac{1}{\sum M_{\mathrm{dis}}}\sum_{x,y}M_{\mathrm{dis}}(x,y)\cdot\left\|\hat{I}(x,y)-I(x,y)\right\|_{1},$$
$$(5)$$

where ˆI and I denote the rendered and ground-truth images respectively. By incorporating LDAFE,
the model is guided to allocate greater attention to distant regions during training, which in turn encourages the generation of a denser set of Gaussian primitives in these areas. The improved coverage of Gaussians facilitates more accurate reconstruction of fine-grained details, thereby enhancing the visual quality of novel views in far-field regions. Following 3D Gaussian splatting, the color reconstruction loss consists of an L1 loss and a D-SSIM loss. Accordingly, the overall training objective is formulated as:
Ltotal = L1(ˆI, I) + λSSIM LD-SSIM(ˆI, I) + λDAFELDAFE(ˆ*I, I*), (6)
where λSSIM and λDAFE are weighting coefficients that balance the contributions of the D-SSIM and the DAFE loss. 3.4 INTER-MODEL ROBUSTNESS ASSESSMENT

As shown in Figure 3 (left), repeated training using the same algorithm and configuration can produce results with considerable variance, leading to large discrepancies in rendering quality. This highlights the importance of quantifying the divergence among independently trained models under identical settings to assess model robustness. To this end, we propose Inter-Model Robustness (IMR), a novel metric specifically designed for 3DGS, grounded in the theory of 2-Wasserstein Distance (Vaserstein, 1969) and Optimal Transport (OT) (Kantorovich, 1960) over Gaussian point clouds, as illustrated in Figure 3 (right).
Let G1, G2*, . . . , G*n denote n independently trained 3DGS models, where each model Gi consists of Ki Gaussian primitives:
$$G_{i}=\{(m_{i,j},s_{i,j},q_{i,j},\alpha_{i,j},f_{i,j})\}_{j=1}^{K_{i}},$$
j=1, (7)
$$(7)$$

![5_image_0.png](5_image_0.png)

Opacity-weighted
=
 
,
 +  
,

Soft-Matching *(OT)*
2, 2
(1, 2)
, =
 , 

$$(9)$$

Gaussian Mixture Distribution  = 
 , (,
, ,)
where mi,j ∈ R3is the center, si,j ∈ R3is the scaling factor, qi,j ∈ R4is the rotation, αi,j ∈ R is the opacity for rendering ,and fi,j ∈ RL is an L-dimensional color feature. Each Gaussian influences a 3D point x in 3D space following the 3D Gaussian distribution:

$$G_{i,j}(x)=\frac{1}{(2\pi)^{\frac{3}{2}}|\Sigma_{i,j}|^{\frac{1}{2}}}\exp(-\frac{1}{2}(x-m_{i,j})^{T}\Sigma_{i,j}^{-1}(x-m_{i,j})),\tag{8}$$  hence matrix $\Sigma_{i,j}$ is computed from the scale $s_{i,j}$ and rotation $q_{i,j}$.  
where the covariance matrix Σi,j is computed from the scale si,j and rotation qi,j .

To enable robustness analysis, each model is abstracted as a Gaussian mixture distribution:

$$G_{i}=\sum_{j=1}^{K_{i}}w_{i,j}\cdot N(m_{i,j},\Sigma_{i,j}),\quad w_{i,j}={\frac{\alpha_{i,j}}{\sum_{k=1}^{K_{i}}\alpha_{i,k}}}.$$
. (9)
Here, opacity αi,j serves as a proxy for the importance of each Gaussian in the final rendering, enabling a principled weighting of geometric features during comparison. For two Gaussian point clouds, it is difficult to directly pair tens of thousands of Gaussian primitives one by one. Therefore, to quantify the difference between two such Gaussian mixtures, we employ the 2-Wasserstein distance and OT theory to establish a soft matching. For two Gaussian distributions µ1 = N(m1, Σ1) and µ2 = N(m2, Σ2), the Wasserstein distance admits a closed-form via the Bures metric (Bures, 1969; Dowson & Landau, 1982):

$$W_{2}^{2}(\mu_{1},\mu_{2})=\|m_{1}-m_{2}\|^{2}+\mathrm{tr}(\Sigma_{1}+\Sigma_{2}-2(\Sigma_{2}^{\frac{1}{2}}\Sigma_{1}\Sigma_{2}^{\frac{1}{2}})^{\frac{1}{2}}).$$

This expression captures both the positional distance and the shape difference between two ellipsoidal Gaussians. To avoid expensive matrix square roots and improve numerical stability, we approximate the Bures shape term via a first-order Taylor expansion, resulting in following expression:

$$\tilde{W}_{2}^{2}(\mu_{1},\mu_{2})=\|m_{1}-m_{2}\|^{2}+\frac{1}{4}\,\operatorname{tr}\bigl{(}(\Sigma_{1}-\Sigma_{2})\Sigma_{2}^{-1}(\Sigma_{1}-\Sigma_{2})\bigr{)}.$$  **the-dimensional derivation is presented in the Appendix A.** Let $C_{1}$ and $C_{2}$ denote 
The detailed mathematical derivation is presented in the Appendix A. Let G1 and G2 denote two 3DGS models. The corresponding mixture Wasserstein distance is then formulated as an OT problem over the Gaussian components (Rubner et al., 2000):

$$\mathrm{MW}_{2}^{2}(G_{1},G_{2})=\min_{\gamma\geq0}\sum_{i=1}^{K_{1}}\sum_{j=1}^{K_{2}}\gamma_{ij}\tilde{W}_{2}^{2}(\mu_{1,i},\mu_{2,j}),\quad\text{s.t.}\sum_{j}\gamma_{ij}=w_{1,i},\quad\sum_{i}\gamma_{ij}=w_{2,j}.\tag{12}$$  This formulation performs soft structure-aware alignment established by the optimal transport plan 
$$(10)$$

γ ∈ RK1×K2, eliminating the need for explicit correspondence. To compute the distance at scale, we introduce entropic regularization and solve the relaxed problem using the Sinkhorn algorithm (Sinkhorn & Knopp, 1967; Cuturi, 2013):

$$\mathrm{MW}_{2,\varepsilon}^{2}(G_{1},G_{2})=\operatorname*{min}_{\gamma}\sum_{i,j}\gamma_{i j}C_{i j}+\varepsilon\sum_{i,j}\gamma_{i j}\log\gamma_{i j},$$
γij log γij , (13)
$$(11)$$

$$(13)$$

3DGS CoR-GS DropGaussian DDGS GT

![6_image_0.png](6_image_0.png) 

MethodsLLFF (3-view 1/8 Resolution) LLFF (3-view 1/4 Resolution)

PSNR(↑) SSIM(↑) LPIPS (↓) AVGE(↓) PSNR(↑) SSIM(↑) LPIPS (↓) AVGE(↓)

NeRF-bas

edMip-NeRF (Barron et al., 2021) 16.11 0.401 0.460 0.206 15.22 0.351 0.540 0.236

DietNeRF (Jain et al., 2021) 14.94 0.370 0.496 0.233 13.86 0.305 0.578 0.271 RegNeRF (Niemeyer et al., 2022) 19.08 0.587 0.336 0.139 18.66 0.535 0.411 0.156 FreeNeRF (Yang et al., 2023) 19.63 0.612 0.308 0.128 19.13 0.562 0.384 0.146 SparseNeRF (Wang et al., 2023a) 19.86 0.624 0.328 0.128 19.07 0.564 0.392 0.147

3D

G

S-

ba

sed

3DGS (Kerbl et al., 2023) 19.22 0.649 0.229 0.118 16.94 0.488 0.402 0.180 DNGaussian (Li et al., 2024) 19.12 0.591 0.294 0.132 18.47 0.578 0.330 0.145 FSGS (Zhu et al., 2024) 20.43 0.682 0.248 0.108 19.71 0.642 0.283 0.122 CoR-GS (Zhang et al., 2024a) 20.45 0.712 0.196 0.092 19.96 0.696 0.250 0.119 LoopSparseGS (Bao et al., 2025) 20.85 0.717 0.205 0.096 20.19 0.680 0.274 0.114 DropGaussian (Park et al., 2025) 20.76 0.713 0.200 0.097 20.01 0.690 0.258 0.113

D

2GS (Ours) 21.35 0.746 0.179 0.087 20.56 0.695 0.254 0.107

where Cij = W˜ 2 2
(N(m1,i, Σ1,i), N(m2,j , Σ2,j )) is the cost matrix, and ε > 0 is the regularization strength. With the introduction of entropic regularization to the original discrete optimal transport objective, the mixture Wasserstein distance between 3DGS models admits a unique and well-defined optimal solution (Delon & Desolneux, 2020).

Direct computation of transport between tens of thousands of Gaussians is computationally infeasible. To further improve tractability, we adopt a depth-stratified importance sampling strategy to select approximately 10,000 Gaussians primitives. Given that far-field Gaussians are more prone to noise and instability due to underfitting, they are oversampled accordingly.

Let Sij = MW22(Gi, Gj ) denote the pairwise distances between N independently trained models.

To specifically penalize model pairs with large divergence, we use a weighted formulation that amplifies the impact of inconsistent models. Finally, we define the Inter-model Robustness (IMR) metric as:

$$\mathrm{IMR}=l n(\frac{\sum_{1\leq i<j\leq N}S_{i j}^{2}}{\sum_{1\leq i<j\leq N}S_{i j}})$$
) (14)

| Methods      | PSNR(↑)   | SSIM(↑)   | LPIPS(↓)   | AVGE(↓)   |
|--------------|-----------|-----------|------------|-----------|
| 3DGS         | 18.52     | 0.523     | 0.415      | 0.159     |
| FSGS         | 18.80     | 0.531     | 0.418      | 0.156     |
| CoR-GS       | 19.52     | 0.558     | 0.418      | 0.146     |
| DropGaussian | 19.74     | 0.577     | 0.364      | 0.136     |
| D 2GS (Ours) | 20.09     | 0.587     | 0.356      | 0.130     |

Table 2: Performance comparisons of sparse-view synthesis on MipNeRF360 dataset. The best, second-best, and third-best entries are marked in red, orange, and yellow, respectively.

## 4 Experiments

We conduct experiments on LLFF (Mildenhall et al., 2019) and Mip-NeRF360 (Barron et al., 2022), following the same data splits and downsampling as prior work. Our implementation is built on DropGaussian, with 10k training iterations per dataset. The evaluation process uses PSNR, SSIM,
LPIPS, and AVGE (the geometric mean of MSE = 10−
PSNR
10 ,
√1 − SSIM, LPIPS), along with our proposed IMR for robustness. All experiments run on a single H20 GPU. More Implementation Details are presented in the Appendix B.

Quantitative evaluation. We compare D2GS with some NeRF-based methods (Mip-NeRF, Diet-
NeRF, RegNeRF, FreeNeRF, SparseNeRF) and 3DGS-based methods (3DGS, DNGaussian, FSGS, CoR-GS, LoopSparseGS, DropGaussian) on LLFF and MipNeRF360. As shown in Tables 1 and 2, D
2GS consistently achieves the best results. On LLFF (1/8 res.), D2GS surpasses FSGS, CoR-GS,
and LoopSparseGS by 0.92/0.9/0.5 dB PSNR with notable SSIM/LPIPS/AVGE gains, and outperforms DropGaussian by 0.59/0.55 dB at 1/8 and 1/4 res. On MipNeRF360, it also improves over CoR-GS and DropGaussian by 0.57 dB and 0.35 dB PSNR, respectively, confirming its superior reconstruction quality. These gains largely stem from the proposed DD-Drop and DAFE modules, which jointly suppress overfitting in near-field regions while enhancing distant details. More results are presented in the Appendix E.

| Methods       | IMR(↓)        |       |
|---------------|---------------|-------|
| LLFF (3-view) | LLFF (6-view) |       |
| 3DGS          | 3.162         | 3.234 |
| CoR-GS        | 3.136         | 3.270 |
| DropGaussian  | 3.205         | 3.143 |
| D 2GS (Ours)  | 3.039         | 3.109 |

Table 3: IMR comparison on LLFF Dataset with 3view and 6-view Settings. All results are tested on ten independent training models.

To assess the robustness of the trained 3D Gaussian primitives, we report the metric IMR, measuring the dispersion across independently trained models. The number of Gaussian primitives in the scenes of LLFF ranges from 20k to 310k. Table 3 shows that our method achieves the lowest IMR in both sparse settings: 3.039 (3-view) and 3.109 (6-view), respectively. This indicates more stable and consistent Gaussian reconstructions across runs. Qualitative evaluation. Figure 4 shows qualitative results on LLFF, comparing 3DGS, CoR-
GS, DropGaussian, D2GS, and GT. As highlighted by the red boxes, D2GS yields sharper details and fewer artifacts, preserving more high-frequency structures than DropGaussian with a random dropout strategy. This visual comparison highlights the superiority of D2GS in reconstructing finegrained geometry under sparse views. These improvements come mainly from the targeted suppression of redundant Gaussians by DD-Drop module and the enhancement of distant structures. Ablation study on the proposed components. We conduct ablation experiments to validate the effectiveness of each proposed modules on LLFF, as summarized in Table 4. Starting from the baseline without any proposed component, we progressively add the density score, depth score, and depth-based layering for DD-Drop, each of which steadily improves PSNR, SSIM, LPIPS, and IMR. Finally, incorporating the DAFE module further enhances reconstruction quality, leading to the best overall performance. These results confirm that all components contribute complementary benefits, with the full model achieving the highest visual fidelity.

Ablation study on DD-Drop. The upper left part of Table 5 shows different weights ω*depth* and ω*density* to balance the influence of normalized depth and density scores in the dropout process. The best performance is achieved when ω*depth* = 0.5 and ω*density* = 0.5, suggesting that both depth and density contribute positively, and overly increasing the weight of either factor results in

| Density Score   | Depth Score   | Depth-based Layering   | DAFE   | PSNR(↑)   | SSIM(↑)   | LPIPS(↓)   | IMR(↓)   |
|-----------------|---------------|------------------------|--------|-----------|-----------|------------|----------|
| 19.22           | 0.649         | 0.229                  | 3.162  |           |           |            |          |
| ✓               | ✓             | 21.02                  | 0.732  | 0.191     | 3.119     |            |          |
| ✓               | ✓             | 20.92                  | 0.728  | 0.200     | 3.155     |            |          |
| ✓               | ✓             | 21.10                  | 0.735  | 0.187     | 3.111     |            |          |
| ✓               | ✓             | ✓                      | 21.17  | 0.740     | 0.181     | 3.088      |          |
| ✓               | ✓             | ✓                      | ✓      | 21.35     | 0.746     | 0.179      | 3.039    |

Table 4: Ablation Study on proposed components. The ✓indicates adding the module.

rmin rmax PSNR(↑) SSIM(↑) LPIPS(↓) τ (%) PSNR(↑) SSIM(↑) LPIPS(↓) 0.05 0.3 21.16 0.740 0.181 5 21.25 0.744 0.180

0.1 0.3 21.11 0.740 0.181 10 21.26 0.743 0.180

0.05 0.5 21.06 0.738 0.187 15 21.20 0.741 0.181

ωdepth ω*density* PSNR(↑) SSIM(↑) LPIPS(↓) λDAFE PSNR(↑) SSIM(↑) LPIPS(↓)

0.2 0.8 21.07 0.737 0.183 0.5 21.27 0.743 0.180 0.5 0.5 21.16 0.740 0.181 1.0 21.30 0.744 0.179 0.8 0.2 21.04 0.734 0.190 1.5 21.25 0.743 0.182

Table 5: Ablation study on different parameters in our model. In DD-Drop module, rmin and rmax denote the minimum and maximum dropout rates, while ω*depth* and ω*density* are used in computing the dropout score. In DAFE module, τ denotes the depth threshold controlling the proportion of far regions retained, and λDAFE denotes the weight of the DAFE loss.

a performance drop. The lower left part of Table 5 presents results under different combinations of minimum and maximum dropout thresholds rmin and rmax, which control the dynamic range of the time-dependent dropout rate r(t). We observe that setting rmin = 0.05 and rmax = 0.3 achieves the best performance, indicating that maintaining a mild dropout rate in the early training stages helps to preserve the geometry of the essential scene, while gradually increasing the dropout rate to a moderate level encourages effective regularization during the later stages. Ablation study on DAFE. We further conduct ablations on the components of the proposed DAFE loss. As shown in the upper right part of Table 5, we compare different values for the depth-based masking ratio, where selecting the top 5% of the farthest depth values yields the best performance, indicating that enforcing depth fidelity in distant regions is particularly beneficial under sparse-view settings. In the lower right part of Table 5 investigates the impact of the weighting hyperparameter in the DAFE loss. A moderate value (e.g., 1.0) provides the best trade-off across all metrics. Table 6 compares different depth estimation models on DAFE supervision. DepthAnything V2 is used by default. While different depth estimator has an impact on the performance, our method demonstrates consistent improvements across all models, indicating that DAFE is compatible with a variety of depth priors and can effectively enhance rendering quality under sparse-view settings.

Methods PSNR(↑) SSIM(↑) LPIPS(↓) MiDaS 21.21 0.740 0.182 DPT 21.27 0.743 0.181 DepthAnything V2 **21.35 0.746 0.179**

Table 6: Ablation Study on different monocular depth estimators: MiDas (Ranftl et al., 2022) with VIT-small backbone, DPT (Ranftl et al., 2021) with VIT-Hybrid backbone, and DepthAnything V2 (Yang et al., 2024).

## 5 Conclusion

In this work, we present a novel D2GS for enhancing sparse-view 3D reconstruction. We introduce a depth-and-density guided dropout that selectively removes over-fitted Gaussians in texture-dense, near-camera regions. To complement this, the proposed Distance-Aware Fidelity Enhancement loss leverages depth priors to reinforce geometric consistency—particularly in distant regions prone to underfitting. Beyond accuracy, we also assess robustness with an inter-model robustness metric, showing more stable Gaussian distributions across runs. Extensive experiments on standard benchmarks confirm consistent gains in both quantitative metrics and visual fidelity over strong baselines. Acknowledgements. This work has been supported by the New Cornerstone Science Foundation through the XPLORER PRIZE.

## References

Zhenyu Bao, Guibiao Liao, Kaichen Zhou, Kanglin Liu, Qing Li, and Guoping Qiu. Loopsparsegs:
Loop based sparse-view friendly gaussian splatting. *IEEE TIP*, 2025. 2, 3, 7 Jonathan T Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and Pratul P Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields.

In ICCV, 2021. 3, 7 Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In *CVPR*, 2022. 3, 8, 15, 17 Jonathan T Barron, Ben Mildenhall, Dor Verbin, Pratul P Srinivasan, and Peter Hedman. Zip-nerf:
Anti-aliased grid-based neural radiance fields. In *ICCV*, 2023. 3 Donald Bures. An extension of kakutani's theorem on infinite product measures to the tensor product of semifinite w*-algebras. *Transactions of the American Mathematical Society*, 1969. 6 David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, pp. 19457– 19467, 2024. 3 Kangjie Chen, Yingji Zhong, Zhihao Li, Jiaqi Lin, Youyu Chen, Minghan Qin, and Haoqian Wang.

Quantifying and alleviating co-adaptation in sparse-view 3d gaussian splatting. *arXiv*, 2025. 3 Tianlong Chen, Peihao Wang, Zhiwen Fan, and Zhangyang Wang. Aug-nerf: Training stronger neural radiance fields with triple-level physically-grounded augmentations. In *CVPR*, 2022. 3 Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-
Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In *ECCV*, pp. 370–386. Springer, 2024. 3 Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS*, 2013. 6 Julie Delon and Agnes Desolneux. A wasserstein-type distance in the space of gaussian mixture models. *SIAM Journal on Imaging Sciences*, 2020. 7 Kangle Deng, Andrew Liu, Jun-Yan Zhu, and Deva Ramanan. Depth-supervised nerf: Fewer views and faster training for free. In *CVPR*, 2022. 3 D. C. Dowson and B. V. Landau. The frechet distance between multivariate normal distributions. ´
Journal of Multivariate Analysis, 1982. 6 Sara Fridovich-Keil, Alex Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In *CVPR*, 2022. 3 Stephan J Garbin, Marek Kowalski, Matthew Johnson, Jamie Shotton, and Julien Valentin. Fastnerf:
High-fidelity neural rendering at 200fps. In *ICCV*, 2021. 3 Wenbo Hu, Yuling Wang, Lin Ma, Bangbang Yang, Lin Gao, Xiao Liu, and Yuewen Ma. Tri-miprf:
Tri-mip representation for efficient anti-aliasing neural radiance fields. In *ICCV*, 2023. 3 Ajay Jain, Matthew Tancik, and Pieter Abbeel. Putting nerf on a diet: Semantically consistent few-shot view synthesis. In *ICCV*, 2021. 3, 7 Rasmus Jensen, Anders Dahl, George Vogiatzis, Engin Tola, and Henrik Aanæs. Large scale multiview stereopsis evaluation. In *CVPR*, pp. 406–413, 2014. 15, 16 Leonid V Kantorovich. Mathematical methods of organizing and planning production. Management science, 1960. 5 Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- ¨
ting for real-time radiance field rendering. TOG, 2023. 2, 7, 14 Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, and Kwang Moo Yi. 3d gaussian splatting as markov chain monte carlo. *NeurIPS*, 2024. 3 Joo Chan Lee, Daniel Rho, Xiangyu Sun, Jong Hwan Ko, and Eunbyung Park. Compact 3d gaussian representation for radiance field. In *CVPR*, 2024. 2, 3 Jiahe Li, Jiawei Zhang, Xiao Bai, Jin Zheng, Xin Ning, Jun Zhou, and Lin Gu. Dngaussian: Optimizing sparse-view 3d gaussian radiance fields with global-local depth normalization. In CVPR, 2024. 7 Sicheng Li, Hao Li, Yue Wang, Yiyi Liao, and Lu Yu. Steernerf: Accelerating nerf rendering via smooth viewpoint trajectory. In CVPR, 2023. 3 Xin Lin, Shi Luo, Xiaojun Shan, Xiaoyu Zhou, Chao Ren, Lu Qi, Ming-Hsuan Yang, and Nuno Vasconcelos. Hqgs: High-quality novel view synthesis with gaussian splatting in degraded scenes. In *ICLR*, 2025. 2 Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ToG, 2019. 7, 8, 14, 15, 16 Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 2 Thomas Muller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics prim- ¨
itives with a multiresolution hash encoding. TOG, 2022. 3 Simon Niedermayr, Josef Stumpfegger, and Rudiger Westermann. Compressed 3d gaussian splatting ¨
for accelerated novel view synthesis. In *CVPR*, 2024. 2, 3 Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs.

In *CVPR*, 2022. 3, 7 Hyunwoo Park, Gun Ryu, and Wonjun Kim. Dropgaussian: Structural regularization for sparse-view gaussian splatting. In *CVPR*, 2025. 2, 3, 7, 14 Lu Qi, Jason Kuen, Jiuxiang Gu, Zhe Lin, Yi Wang, Yukang Chen, Yanwei Li, and Jiaya Jia. Multiscale aligned distillation for low-resolution detection. In *CVPR*, 2021. 3 Lu Qi, Jason Kuen, Yi Wang, Jiuxiang Gu, Hengshuang Zhao, Philip Torr, Zhe Lin, and Jiaya Jia.

Open world entity segmentation. In *TPAMI*, 2022. 3 Lu Qi, Jason Kuen, Weidong Guo, Tiancheng Shen, Jiuxiang Gu, Jiaya Jia, Zhe Lin, and Ming-
Hsuan Yang. High-quality entity segmentation. In *ICCV*, 2023. 3 Lu Qi, Lehan Yang, Weidong Guo, Yu Xu, Bo Du, Varun Jampani, and Ming-Hsuan Yang. Unigs:
Unified representation for image generation and segmentation. In *Proceedings of the IEEE/CVF* Conference on Computer Vision and Pattern Recognition, 2024. 3 Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. ´
ICCV, 2021. 9 Rene Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust ´
monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. *IEEE TPAMI*, 2022. 9 Barbara Roessle, Jonathan T Barron, Ben Mildenhall, Pratul P Srinivasan, and Matthias Nießner.

Dense depth priors for neural radiance fields from sparse input views. In *CVPR*, 2022. 3 Yossi Rubner, Carlo Tomasi, and Leonidas J Guibas. The earth mover's distance as a metric for image retrieval. *IJCV*, 2000. 6 Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, and Xiangtai Li. Dreamrelation: Bridging customization and relation generation. In *CVPR*, 2025. 3 Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. *Pacific Journal of Mathematics*, 1967. 6 Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In CVPR, 2022. 3 Shengji Tang, Weicai Ye, Peng Ye, Weihao Lin, Yang Zhou, Tao Chen, and Wanli Ouyang. Hisplat:
Hierarchical 3d gaussian splatting for generalizable sparse-view reconstruction. *arXiv*, 2024. 3 Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. Sparf: Neural radiance fields from sparse and noisy poses. In CVPR, 2023. 2, 3 Leonid Nisonovich Vaserstein. Markov processes over denumerable products of spaces, describing large systems of automata. *Problemy Peredachi Informatsii*, 1969. 5 Dor Verbin, Peter Hedman, Ben Mildenhall, Todd Zickler, Jonathan T Barron, and Pratul P Srinivasan. Ref-nerf: Structured view-dependent appearance for neural radiance fields. In *CVPR*, 2022. 3 Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. Sparsenerf: Distilling depth ranking for few-shot novel view synthesis. In *ICCV*, 2023a. 2, 3, 7 Peng Wang, Yuan Liu, Zhaoxi Chen, Lingjie Liu, Ziwei Liu, Taku Komura, Christian Theobalt, and Wenping Wang. F2-nerf: Fast neural radiance field training with free camera trajectories. In CVPR, 2023b. 3 Jiawei Yang, Marco Pavone, and Yue Wang. Freenerf: Improving few-shot neural rendering with free frequency regularization. In *CVPR*, 2023. 2, 3, 7 Lehan Yang, Lu Qi, Xiangtai Li, Sheng Li, Varun Jampani, and Ming-Hsuan Yang. Unified dense prediction of video diffusion. In *CVPR*, 2025. 3 Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In *arXiv*, 2024. 9 Zongxin Ye, Wenyu Li, Sidun Liu, Peng Qiao, and Yong Dou. Absgs: Recovering fine details in 3d gaussian splatting. In *ACM MM*, 2024. 2, 3 Alex Yu, Ruilong Li, Matthew Tancik, Hao Li, Ren Ng, and Angjoo Kanazawa. Plenoctrees for real-time rendering of neural radiance fields. In *ICCV*, 2021. 3 Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Aliasfree 3d gaussian splatting. In *CVPR*, 2024. 2, 3 Jingtong Yue, Zhiwei Lin, Xin Lin, Xiaoyu Zhou, Xiangtai Li, Lu Qi, Yongtao Wang, and Ming-
Hsuan Yang. Roburcdet: Enhancing robustness of radar-camera fusion in bird's eye view for 3d object detection. In *ICLR*, 2025. 3 Jiawei Zhang, Jiahe Li, Xiaohan Yu, Lei Huang, Lin Gu, Jin Zheng, and Xiao Bai. Cor-gs: sparseview 3d gaussian splatting via co-regularization. In ECCV, 2024a. 2, 3, 7 Zheng Zhang, Wenbo Hu, Yixing Lao, Tong He, and Hengshuang Zhao. Pixel-gs: Density control with pixel-aware gradient for 3d gaussian splatting. In *ECCV*, 2024b. 2, 3 Junwei Zhou, Xueting Li, Lu Qi, and Ming-Hsuan Yang. Layout-your-3d: Controllable and precise 3d generation with 2d blueprint. In *ICLR*, 2024. 2 Zehao Zhu, Zhiwen Fan, Yifan Jiang, and Zhangyang Wang. Fsgs: Real-time few-shot view synthesis using gaussian splatting. In *ECCV*, 2024. 7, 14
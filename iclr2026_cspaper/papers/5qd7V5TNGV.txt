000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Cp4D: Compositional Physics-Aware 4D Scene Generation

Anonymous authors Paper under double-blind review

## Abstract

4D generation (i.e., dynamic 3D generation) has recently emerged as a rapidly growing research frontier due to its powerful spatiotemporal modeling capabilities. However, despite notable advances, existing approaches typically fail to capture the underlying physical principles, producing results that are both physically inconsistent and visually implausible. To overcome this limitation, we present CP4D, a novel paradigm for photorealistic 4D scene synthesis with faithful adherence to complex physical dynamics. Drawing inspiration from the compositional nature of real-world scenes, where immutable static backgrounds coexist with dynamic, physically plausible foregrounds, CP4D reformulates 4D generation as the integration of a static 3D environment with physically grounded dynamic objects. On this basis, our framework follows a three-stage pipeline: 1) Firstly, we leverage pre-trained expert models to generate high-fidelity 3D representations of the environment and foreground objects respectively. 2) Subsequently, to produce physically plausible trajectories and realistic interactions for these objects, we propose a hybrid motion synthesis strategy that integrates priors from physical simulators with the common sense embedded in video diffusion models. 3) Finally, we develop an automated composition mechanism that seamlessly fuses the static environment and dynamic objects into coherent, physically consistent 4D scenes. Extensive experiments demonstrate that CP4D can generate explorable and interactive 4D scenes with high visual fidelity, strong physical plausibility, and finegrained controllability, significantly outperforming existing methods. The anonymous project page: https://anonymous.4open.science/w/CP4D/.

## 1 Introduction

Empowered by recent progress in generative models (Ho et al., 2020; Song et al., 2020) and largescale data available, 4D generation (*i.e.*, dynamic 3D generation) (Ren et al., 2023; Xie et al., 2024b; YU et al., 2025; Ma et al., 2025) has emerged as a prominent research focus. Through joint modeling of spatial structure and temporal dynamics, 4D generation enables the synthesis of realistic and coherent 4D scenes, holding great promise for a wide range of applications such as AR/VR (Li et al., 2024a), robotics (Liu et al., 2025a), and world models (Chen et al., 2025b). Existing approaches for 4D generation can be broadly divided into two categories. The first class of methods exploits priors distilled from pre-trained video or 3D generative models (Bahmani et al., 2024b;a; Jiang et al., 2023; Zeng et al., 2024), employing them as auxiliary supervisory signals to constrain the generation process and improve fidelity. In contrast, the second class follows a datadriven paradigm (Xie et al., 2024b; Ren et al., 2024; Liang et al., 2024; Bai et al., 2025), where crossview videos are directly synthesized as intermediate proxies and subsequently transformed into full 4D content through classical reconstruction pipelines. While producing seemingly plausible results, these approaches typically lack an explicit characterization of the underlying physical principles. As a consequence, the generated content often suffers from physical inconsistencies and visual artifacts, leading to scenes that deviate from realistic dynamics. To mitigate this issue, inspired by the compositional nature of real-world scenes (Xu et al., 2024; Zhu et al., 2024), where static backgrounds co-exist with physically plausible dynamic foregrounds, we reformulate 4D scene generation as the integration of a static 3D environment with physically 1

![1_image_0.png](1_image_0.png)

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 grounded dynamic 3D objects. Building upon this formulation, there arise three key technical challenges: 1) How to construct plausible 3D representations of background environments and foreground objects that conform to user-specified instructions? 2) How to model the motion dynamics of foreground objects that encompass physically plausible trajectories and realistic interactions? 3) How to seamlessly compose the generated dynamic foregrounds with the static background into a consistent 4D scene? To tackle these challenges, in this paper we introduce CP4D, a novel paradigm for photorealistic 4D scene generation with faithful adherence to complex physical dynamics. Specifically, as shown in Fig. 1, CP4D follows a three-stage pipeline: 1) Firstly, given a textual prompt, we first synthesize a background image using a text-to-image generative model, after which an image editing model, conditioned on this background, is employed to generate foregrounds that are visually compatible with it. Both the background and the foregrounds are then reconstructed into their respective 3D representations using pre-trained expert models. In contrast to the naive baseline that independently applies text-to-3D models to each component, our approach enforces stylistic coherence across background and foreground, thereby mitigating implausible artifacts such as realistic environments juxtaposed with cartoon-like objects. 2) Secondly, to endow foreground objects with physically plausible trajectories and realistic interactions, we introduce a hybrid motion synthesis strategy. In particular, we first leverage physical simulators to produce coarse object trajectories that comply with fundamental physical laws. These initial dynamics are subsequently refined using the commonsense knowledge embedded in video generative models, thereby enhancing inter-object interactions and yielding motion that is both more realistic and visually convincing. 3) Thirdly, to seamlessly fuse the dynamic foregrounds with the static background into a unified 4D scene, we develop an automated composition mechanism. By leveraging monocular depth estimation and a depth-aware heuristic rule, this mechanism first estimates the relative spatial attributes of foreground objects (e.g., positions and scales) within the background, which are subsequently calibrated through optimization to ensure coherent integration and visually compelling compositions. Notably, owing to its compositional design, CP4D not only enables the synthesis of 4D scenes that faithfully comply with physical laws, but also provides strong interactive controllability. In particular, users are afforded the flexibility to edit different scene elements, such as foreground objects, background environments, and motion trajectories, thus facilitating diverse 4D generation. In summary, our key contributions can be concluded as follows:

## 2 Related Works 2.1 4D Generation

- We present CP4D, a novel compositional framework designed to generate photorealistic 4D scenes with accurate adherence to complex physical dynamics.

- We propose a hybrid motion synthesis strategy that integrates physical priors from differentiable simulators with commonsense knowledge from video generative models, yielding physically plausible trajectories and realistic interactions.

- We develop an automated composition mechanism that harmoniously fuses dynamic foregrounds with the static background, producing a coherent and visually compelling 4D scene.

- Extensive experiments demonstrate that CP4D is capable of synthesizing explorable and interactive 4D scenes characterized by high visual fidelity, robust physical realism, and precise controllability, consistently outperforming prior methods.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Generating 4D assets from textual prompts has drawn growing attention owing to its wide-ranging applications in AR, VR (Wang et al., 2025), and spatial intelligence. Early approaches (Jiang et al., 2023; Zhu et al., 2025; Li et al., 2024b; Zeng et al., 2024; Gao et al., 2024) towards this goal predominantly relied on distilling knowledge from pre-trained generative models to guide the generation process. For instance, DreamGaussian4D (Ren et al., 2023) pioneered the use of SDS (Poole et al., 2022) in the 4D generation domain, demonstrating the capability to produce realistic 4D objects conditioned on text prompts. Consistent4D (Jiang et al., 2023) realized video-to-4D generation by integrating SDS with dynamic NeRF (Park et al., 2021), and further employed a video enhancer to improve the quality of the synthesized 4D assets. Recently, the availability of large-scale datasets (Deitke et al., 2023; Nan et al., 2024) has enabled methods that directly train feed-forward video diffusion models to synthesize multi-view videos (Xie et al., 2024b; Ren et al., 2024; YU et al., 2025; Bai et al., 2025; He et al., 2024; Namekata et al., 2024), which are subsequently reconstructed into 4D scenes using standard reconstruction techniques (Wu et al., 2024). However, despite their ability to produce seemingly plausible results, these approaches generally overlook the explicit characterization of underlying physical dynamics. Consequently, the generated content often exhibits physically inconsistent behaviors and visual artifacts. In contrast, we present CP4D, a physics-aware framework for text-driven 4D scene generation, delivering photorealistic quality, reliable physical consistency, and precise generation control.

## 2.2 Physics-Based Simulation

Given an initial 3D representation (*e.g.*, 3D gaussian splatting (Kerbl et al., 2023)), recent works (Xie et al., 2024a) have explored the use of physical solvers, such as the Material Point Method (MPM) (Hu et al., 2018; Jiang et al., 2017), to update the state of Gaussian primitives under external forces at different timestamps. To automate the specification of material parameters, multimodal large language models (MLLMs) have been employed to infer properties such as density, Young's modulus, and Poisson's ratio (Zhao et al., 2024; Mao et al., 2025). Complementary to this, other approaches (Huang et al., 2025; Liu et al., 2024a; Lin et al., 2025; Liu et al., 2025b) exploit implicit physical regularities in video diffusion models by incorporating Score Distillation Sampling (SDS) (Poole et al., 2022) to refine these preliminary estimates. While the above methods assume access to well-defined 3D representations, more recent works (Lin et al., 2024a;b; Chen et al., 2025a; Tan et al., 2024; Liu et al., 2024b) aim to generate physics-driven videos directly from a single image. These methods first generate a full 3D representation using image-to-3D models
(either mesh-based (Chen et al., 2025a) or Gaussian-based (Lin et al., 2024a;b; Tan et al., 2024)) before applying physical simulations as described above. However, existing solutions remain limited: they typically handle only elastic or rigid bodies, lack support for realistic multi-material Gao et al. (2025) and multi-object interactions, and often employ either 2D backgrounds or 3D environments with fixed viewpoints, restricting the ability to render consistent novel views.

## 3 Preliminaries: Score Distillation Sampling 4 Methodology

Score Distillation Sampling (SDS) (Poole et al., 2022) is a widely used technique for optimizing a differentiable generator g(θ) under the guidance of a pre-trained diffusion model. Its core idea is to exploit the score function of the diffusion model to supply gradient that steer the generator's outputs towards alignment with a target text prompt, without the need for explicit likelihood computation.

Formally, let ϵϕ(·, T, ζ) denote the denoiser of a pre-trained text-conditioned diffusion model parameterized by ϕ, with timestep ζ and text prompt T, the SDS gradient is given by:

$$\nabla_{\theta}{\mathcal{L}}_{\mathrm{SDS}}=\mathbb{E}_{\epsilon,\zeta}\left[\omega(\zeta)\left(\epsilon_{\phi}(g(\theta),\mathbf{T},\zeta)-\epsilon\right){\frac{\partial g(\theta)}{\partial\theta}}\right]$$
$$(\mathbf{l})$$
$\uparrow$ . 
, (1)
where g(θ) denotes the generator's output (e.g., a rendered video), ϵ is Gaussian noise sampled at timestep ζ, ω(ζ) is a weighting function, and θ are the learnable parameters of the generator. Overview. Given a textual prompt T, our objective is to synthesize a 4D scene that faithfully adheres to complex physical dynamics while supporting flexible viewpoint changes. To this end, we adopt a compositional formulation grounded in the nature of real-world scenes (*i.e.*, static backgrounds coexisting with physically governed, dynamic foregrounds), and cast 4D generation as the integration of a static 3D environment with physically grounded dynamic objects. To achieve this goal, we introduce a three-stage pipeline. To begin with, we leverage pre-trained expert models to construct plausible 3D representations for both the background environment and the foreground objects (Sec. 4.1). Subsequently, we propose a hybrid motion synthesis strategy utilizing physical simulators and video generative models to produce foreground motions with physical consistency and realistic interactions (Sec. 4.2). Finally, we develop an automated composition mechanism that seamlessly integrates the generated background and foreground into a coherent 4D scene (Sec. 4.3).

## 4.1 S**Tage** I: Background–Foreground 3D Representation Synthesis

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 To achieve text-guided compositional physics-aware 4D scene generation (CP4D), constructing plausible 3D representations of both the background environment and the foreground objects constitutes an essential prerequisite, providing the foundation for subsequent motion modeling and scene composition. To this end, we first invoke a large language model (e.g., GPT-4o (Achiam et al.,
2023)) to decompose the input textual prompt T into two sub-prompts (*i.e.*, T = {Tb, Tf }), each describing the background and foreground to be generated. Subsequently, to obtain the corresponding 3D representations of the background and foreground, one intuitive approach is to independently apply pretrained text-to-3D generative models. However, such a straightforward strategy typically yields implausible outcomes, *e.g.*, generating a realistic background paired with cartoon-like foregrounds, which in turn undermines the coherence and overall quality of the synthesized 4D scene. To overcome this limitation, we adopt a simple yet effective strategy for 3D representation synthesis.

Specifically, we first synthesize a background image Ib from the input prompt Tb using a text-toimage generative model Ft2i. Next, conditioned on Ib and Tf , we employ an image editing model F*edit* to generate a composite image Ib,f that simultaneously contains both the background and foreground in a visually coherent manner. We then apply an image segmentation model Fseg to Ib,f to isolate the foreground region Mf (Mf = 1 corresponds to foreground pixels and Mf = 0 to background pixels), yielding the foreground image If . Finally, with the harmonized background image Ib and foreground image If , we leverage pretrained image-to-3D generative models F
b3d and F
f 3dto construct their respective 3D representations. The overall pipeline can be formally expressed as follows:

$$(2)^{\frac{1}{2}}$$
$$\mathbf{G}_{b}=\mathbf{F}_{3d}^{b}(\mathbf{I}_{b}),\ \mathbf{G}_{f}=\mathbf{F}_{3d}^{f}(\mathbf{I}_{f}),$$  $\mathbf{I}_{b}=\mathbf{F}_{t2i}(\mathbf{T}_{b}),\mathbf{I}_{b,f}=\mathbf{F}_{e d i t}(\mathbf{I}_{b},\mathbf{T}_{f}),\ \{\mathbf{I}_{f},\mathbf{M}_{f}\}=\mathbf{F}_{s e g}(\mathbf{I}_{b,f}),$
where Gb and Gf denote the 3D representations of the background and foreground, instantiated using *3D gaussian splatting*. For clarity, we use Gf as a unified notation to represent the foreground representation, which may correspond to either a single object or multiple different objects.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

## 4.2 Stage Ii: Physically Grounded Motion Simulation

Given the generated Gf , the second stage aims to endow the foreground objects with motions that are both physically consistent and visually realistic. To this end, we adopt a hybrid motion synthesis framework: physical simulators are first employed to generate trajectories constrained by fundamental physical laws, which are subsequently refined using the commonsense priors embedded in video generative models. This design ensures that the resulting motions remain faithful to physics while exhibiting naturalistic interactions.

Physical simulator-based motion synthesis. To simulate the dynamics of Gf conditioned on the textual description Tf , we begin by leveraging vision-language models (VLMs) to infer essential physical attributes of the objects, including material properties (*e.g.*, Young's modulus E, Poisson's ratio µ, and density ρ) and external forces Q. These inferred parameters provide the initialization required for physically grounded motion simulation. More details are provided in Appendix B.

We then employ heterogeneous physical solvers Φ to simulate object dynamics. Specifically, elastic or flexible objects are handled using an MPM solver Φmpm, rigid objects are modeled with a dedicated rigid-body solver Φ*rigid*, while fluid objects are simulated with a Position-Base-Dynamic (PBD) solver Φ*f luid* (More details are provided in Appendix C). Initialized with the estimated material parameters Θ = {*ρ, E, µ*} and external forces Q, the solvers evolve the foreground into deformed 3D representations Gtf over time t, which can be expressed as:

$$\mathbf{G}_{f}^{t}=\Phi(\mathbf{G}_{f},\mathbf{Q},\Theta,t).$$
$$({\mathfrak{I}})$$
Gtf = Φ(Gf , Q, Θ, t). (3)
Video generative model-based refinement. Although Eq. 3 produces motions that are broadly consistent with physical principles, two critical limitations persist. 1) As VLMs are not explicitly trained on physics-oriented datasets, the inferred material parameters, while generally reasonable, often lack the numerical accuracy required to reflect precise physical behavior. 2) As shown in Fig. 2, physics solvers generally rely on grid-based approximations of Gf to model interactions such as collisions. However, the limited fidelity of these approximations often fails to capture the intricate geometry of the underlying 3D structures, leading to perceptually implausible outcomes, e.g., collisions may be registered between objects despite no apparent contact in the rendered scene. To mitigate these issues, we resort to commonsense knowledge embedded in video diffusion models. Specifically, to solve the first problem, we employ the SDS loss to optimize the estimated physical parameters Θ, which is denoted as follows:

$$\nabla_{\Theta}{\mathcal{L}}_{\mathrm{SDS}}=\mathbb{E}_{\epsilon,\zeta}\left[\omega(\zeta)(\hat{\epsilon}_{\psi}(V;\mathbf{T}_{f};\zeta)-\epsilon){\frac{\partial V}{\partial\Theta}}\right],$$
, (4)
where V denotes the rendered video based on Gtf, ϵˆψ represents the predicted noise using pre-trained video diffusion model ψ, ω(ζ) is a weighting function over the diffusion timestep ζ. To alleviate the second issue, namely the inaccuracies introduced by coarse grid-based approximations during inter-object interactions, we similarly employ SDS-based optimization. Specifically, assuming Gf comprises K individual objects, {Gfi }
K
i=1, we assign to each object a learnable global displacement variable ∆Γi, which adjust their relative positions. These displacement variables are

$$(4)$$

270

![5_image_0.png](5_image_0.png) 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

optimized via SDS supervision to ensure that the rendered video adheres to the textual prompt Tf
while exhibiting interaction patterns aligned with human perceptual priors, which is formulated as follows:
follows.  $$\nabla_{\Delta\Gamma}\,{\cal L}_{\rm SBS}=\mathbb{E}_{\epsilon,\zeta}\left[\omega(\zeta)\big{(}\hat{\epsilon}_{\psi}(V_{\Delta\Gamma};{\bf T}_{f},\zeta)-\epsilon\big{)}\frac{\partial V_{\Delta\Gamma}}{\partial\Delta\Gamma}\right],$$  where $V_{\Delta\Gamma}$ denotes the rendered video after applying displacements $\Delta\Gamma$.  
, (5)
$$({\boldsymbol{S}})$$

## 4.3 Stage **Iii:** Automated 4D Scene Composition

$$\mathbf{G}_{f}^{*}=S\times\mathbf{G}_{f}+P,$$

After obtaining physically grounded motions of the foreground object(s) Gf , our next goal is to fuse them with the background Gb into a coherent 4D scene. To this end, we introduce an automated scene composition mechanism that estimates the relative spatial attributes of Gf (*e.g.*, its position and scale) with respect to Gb using monocular depth cues and heuristic priors, and further refines them through optimization to ensure both geometric consistency and visual plausibility. A detailed illustration is provided below. Relative spatial attributes initialization. Since Gb and Gf are generated independently by different pre-trained expert models, their 3D representations lie in distinct coordinate spaces, making direct integration infeasible. Therefore, to reasonably place Gf into Gb with correct size and location, we propose to transform Gf into an aligned representation G∗fusing the following equation:
G∗f = S × Gf + P, (6)
where S ∈ R
+ denotes the relative scale and P = (P
x, Py, Pz) ∈ R
3the relative translation. For clarity, we simplify Gf as Gf = (Gx f, G
y f
, Gzf) ∈ R
U×3, considering only the transformation of its U spatial coordinates.

Subsequently, to estimate the translation parameter P (*i.e.*, the spatial location of Gf within Gb), we employ a monocular depth estimator F*depth* on the composite image Ib,f (as defined in Eq. 2) to recover a dense depth map of the scene. Guided by the foreground mask Mf , depth values associated with the target region are isolated, from which the centroid depth of the foreground object is derived. This depth estimate is further back-projected into 3D space, providing an initialization of the foreground position P in the coordinate frame of Gb, which can be formulated as follows:
(P
x, Py, Pz) = Φ(Db,f [(Mf = 1)cen]), Db,f = Fdepth(Ib,f ), (7)
where Db,f denotes the depth map estimated from the composite image Ib,f , (Mf = 1)cen indicates the centroid pixel of the segmented foreground region, Φ(·) represents the back-projection function that maps a 2D pixel into 3D space based on its depth value. Notably, since we unify the world coordinate system of the background with the camera coordinate system, the z-coordinate of P (*i.e.*, P
z) is directly equal to the corresponding depth value Db,f [(Mf = 1)cen].

For scale estimation, *i.e.*, determining the size of Gf within Gb, we employ a depth-aware heuristic. The key insight is that, under the reference viewpoint corresponding to Ib,f , the foreground object 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 should be entirely visible within the image plane. This implies that, in 3D space, Gf **must be**
fully contained within the camera frustum of the reference view. Given the estimated depth P
z, as shown in Fig. 3, the scale S is constrained such that all points of Gf fall within the valid frustum slice at depth P
z, *i.e.*, their x- and y-coordinates remain bounded by the image-plane limits defined at that depth. Accordingly, we initialize S as the maximum feasible scale that satisfies these geometric bounds, which is formulated as follows:

$$S=\frac{\operatorname*{min}(\operatorname*{min}(P^{x}-B_{\operatorname*{min}}^{x},B_{\operatorname*{max}}^{x}-P^{x})\,,\,\operatorname*{min}(P^{y}-B_{\operatorname*{min}}^{y},B_{\operatorname*{max}}^{y}-P^{y}))}{\operatorname*{max}\Bigl(({\bf G}_{f}^{x})_{\operatorname*{max}}-({\bf G}_{f}^{x})_{\operatorname*{min}},\,({\bf G}_{f}^{y})_{\operatorname*{max}}-({\bf G}_{f}^{y})_{\operatorname*{min}}\Bigr)\,/2},$$
$$(9)$$
(8) $\frac{1}{2}$
where Bxmin, Bxmax, Bymin, Bymax denote the horizontal and vertical boundaries of the camera frustum
at depth P
z, (Gx
f
)max,(Gxf
)min,(G
y f
)max,(G
y f
)min represent the maximum and minimum x- and
y-coordinates of the original foreground representation Gf , respectively. Optimization-based refinement. After obtaining the initial estimates of P and S, we further
refine them to improve perceptual fidelity. The objective is to ensure that the rendered reference
view of the composed scene closely aligns with the composite image Ib,f . Accordingly, we optimize
P and S by minimizing the discrepancy between the rendered image ˆIb,f (*P, S*) and Ib,f , formulated
as:
$$(P,S)=\arg\operatorname*{min}_{P,\,S}\;\left\|\hat{\mathbf{I}}_{b,f}(P,S)-\mathbf{I}_{b,f}\right\|_{2}^{2}.$$
. (9)
Notably, our experiments reveal that simultaneously optimizing S and P introduces substantial ambiguity, often leading to suboptimal local minima. To address this, we employ a sequential refinement strategy: first optimizing the scale S, followed by refining the translation P. This progressive scheme significantly reduces uncertainty and consistently yields more robust and reliable composition results.

## 5 Experiments 5.1 Experimental Setups

Implementation details. We curate a dataset of 17 examples for evaluation, where each instance consists of a foreground prompt Tf and a background prompt Tb. Qwen-Image (Wu et al., 2025) is employed to generate the background image Ib from Tb, and Qwen-Image-Edit is further applied to synthesize the composite image Ib,f . The foreground mask Mf is extracted from Ib,f using SAM (Kirillov et al., 2023), and its depth map is estimated with Depth Anything (Yang et al.,
2024). Foreground 3D representations Gf are reconstructed with Trellis (Xiang et al., 2025), and the background 3D representation Gb is produced using Viewcrafter (Yu et al., 2024).

Baselines. We compare CP4D against three categories of baselines: physics-driven simulation methods, conditional video generation models, and text-to-4D approaches. For physics-driven methods, we include PhysGen (Liu et al., 2024b), PhysGen3D (Chen et al., 2025a), and Omni- PhysGS (Lin et al., 2025). For conditional video generation, we evaluate open-source models such as CogVideoX (Yang et al., 2025) and Wan (Wan et al., 2025), as well as proprietary systems including Sora (OpenAI, 2024) and Runway (Runway, 2024). Finally, DreamGaussian4D (Ren et al., 2023) is selected as a representative text-to-4D baseline. Metrics. To assess the quality of generated videos, we adopt VBench (Huang et al., 2024) for evaluating motion smoothness, subject consistency, and image quality. In addition, WorldScore (Duan et al., 2025) is employed to measure photo consistency, 3D consistency, and motion smoothness. To further assess prompt adherence, following PhysGen3D (Chen et al., 2025a), we leverage GPT-4o to score generated videos across three dimensions: physical realism, photorealism, and semantic alignment with the input prompt. Please refer to more details in Appendix A. 5.2 COMPARISONS WITH STATE-OF-THE-ART METHODS As illustrated in Fig. 4, we present two challenging cases for qualitative comparison. In the deformable object motion scenario (*i.e.*, the left side of Fig. 4), Sora (OpenAI, 2024) demonstrates limited capability in accurately identifying the target object and modeling its physical dynamics, and further synthesizes spurious motion patterns involving entities absent from the input image. PhysGen3D (Chen et al., 2025a) reconstructs 3D meshes with low geometric fidelity and spatial arrangements that violate physical plausibility, substantially degrading visual realism. Wan (Wan et al., 2025) exhibits pronounced temporal instability due to color flickering, and fails to respond to 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

| Model                                                                   | VBench   | WorldScore   |       |       |       |       |
|-------------------------------------------------------------------------|----------|--------------|-------|-------|-------|-------|
| Motion↑ Consistency↑ Imaging↑ Photo Consist↑ 3D Consist↑ Motion Smooth↑ |          |              |       |       |       |       |
| Runway Runway (2024)                                                    | 0.995    | 0.936        | 0.644 | 62.66 | 86.34 | 68.43 |
| Sora (OpenAI, 2024)                                                     | 0.993    | 0.904        | 0.592 | 52.95 | 64.26 | 33.44 |
| CogVideoX-I2V-5B (Yang et al., 2025) 0.993                              | 0.932    | 0.603        | 70.06 | 81.90 | 73.66 |       |
| Wan2.2-TI2V-5B (Wan et al., 2025)                                       | 0.991    | 0.934        | 0.599 | 72.66 | 77.50 | 47.04 |
| PhysGen (Liu et al., 2024b)                                             | 0.996    | 0.966        | 0.621 | 88.34 | 90.04 | 81.67 |
| PhysGen3D (Chen et al., 2025a)                                          | 0.997    | 0.963        | 0.599 | 93.07 | 92.99 | 90.95 |
| OmniPhysGS (Lin et al., 2025)                                           | 0.995    | 0.960        | 0.356 | 22.54 | 48.80 | 92.88 |
| DreamGaussian4D (Ren et al., 2023)                                      | 0.969    | 0.846        | 0.477 | 14.59 | 40.29 | 34.73 |
| Ours                                                                    | 0.998    | 0.972        | 0.641 | 97.42 | 95.55 | 93.52 |

the motion prompt, resulting in static garments throughout the sequence. In contrast, our method produces coherent, artifact-free motion grounded in the input image, with significantly improved physical fidelity and temporal consistency. In the rigid-body collision scenario (*i.e.*, the right side of Fig. 4), PhysGen3D is restricted to elastic material simulation, causing the bottle to collapse unrealistically upon impact. Sora and Wan (Wan et al., 2025) further undermine plausibility by replacing the bottle with a different object post-collision, thereby breaking object identity and disrupting motion continuity. Compared to these methods, our approach preserves object identity throughout the interaction and yields physically consistent collision outcomes. Kindly refer to more results in Appendix E and F. Quantitatively, as shown in Tab. 1, our method achieves superior motion coherence and temporal smoothness, consistently outperforming both video generative models and physics-driven methods across key dynamic metrics. Moreover, the generated videos exhibit high static visual quality, rivaling or even surpassing the strong closed-source baselines, particularly in terms of 3D consistency.

| Model                              | Physical realism↑   | Photorealism↑   | Semantic alignment↑   |
|------------------------------------|---------------------|-----------------|-----------------------|
| Sora (OpenAI, 2024)                | 0.547               | 0.729           | 0.665                 |
| Runway Runway (2024)               | 0.670               | 0.753           | 0.732                 |
| Wan2.2-TI2V-5B (Wan et al., 2025)  | 0.576               | 0.626           | 0.635                 |
| PhysGen (Liu et al., 2024b)        | 0.524               | 0.615           | 0.588                 |
| PhysGen3D (Chen et al., 2025a)     | 0.624               | 0.624           | 0.626                 |
| OmniPhysGS (Lin et al., 2025)      | 0.347               | 0.265           | 0.170                 |
| DreamGaussian4D (Ren et al., 2023) | 0.229               | 0.112           | 0.176                 |
| Ours                               | 0.694               | 0.759           | 0.747                 |

Table 2: **GPT-4o Evaluation Results**. Our proposed method can achieve the best results.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Regarding physical plausibility, as demonstrated in Tab. 2, our method surpasses all competing approaches on the physics realism metric, while simultaneously maintaining strong alignment with the input text, thereby ensuring high semantic consistency.

## 5.3 Ablation Study

As illustrated in Sec. 4.2, to address inaccuracies in VLM-estimated physical parameters and the limited precision of physics simulators, we employ SDS to separately optimize the material parameters predicted by VLMs and the relative positions of foreground objects. To verify the necessity of these designs, we provide ablation studies here. As shown in Fig. 5, omitting material optimization causes the VLM-predicted density and Young's modulus to yield overly compliant simulations, leading to unstable or non-physical object motion. Without relative position optimization, the simulation of multi-object interactions produces spurious collisions in the absence of true spatial overlap. When both optimization modules are applied, our method yields more stable dynamics and visually plausible object interactions. More ablation studies are provided in the Appendix D.

## 5.4 Applications On Controllable Editing

The compositional design of our method endows it with the inherent ability to edit individual concepts, *e.g.*, varying background environments and foreground objects with distinct motions. As shown in Fig. 6, we can seamlessly replace them in a zero-shot manner while preserving scene coherence, physical plausibility, and temporal consistency, thereby enabling flexible and diverse 4D content generation.

## 6 Conclusion

In this work, we have presented CP4D, a novel framework for photorealistic 4D scene generation with faithful modeling of complex physical dynamics. Drawing inspiration from the compositional nature of real-world scenes, CP4D follows a three-stage pipeline: 1) constructing 3D representations of background environments and foreground objects from textual prompts using pre-trained expert models; 2) producing physically grounded trajectories and realistic interactions through a hybrid motion synthesis strategy; and (3) seamlessly integrating static environments with dynamic objects via an automated composition mechanism. Extensive experiments have demonstrated that our proposed method consistently outperforms state-of-the-art baselines.

## References

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Sherwin Bahmani, Xian Liu, Wang Yifan, Ivan Skorokhodov, Victor Rong, Ziwei Liu, Xihui Liu, Jeong Joon Park, Sergey Tulyakov, Gordon Wetzstein, et al. Tc4d: Trajectory-conditioned textto-4d generation. In *European Conference on Computer Vision*, pp. 53–72. Springer, 2024a.

Sherwin Bahmani, Ivan Skorokhodov, Victor Rong, Gordon Wetzstein, Leonidas Guibas, Peter Wonka, Sergey Tulyakov, Jeong Joon Park, Andrea Tagliasacchi, and David B Lindell. 4d-fy: Text-to-4d generation using hybrid score distillation sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7996–8006, 2024b.

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. Recammaster: Camera-controlled generative rendering from a single video. *arXiv preprint arXiv:2503.11647*, 2025.

Hritik Bansal, Zongyu Lin, Tianyi Xie, Zeshun Zong, Michal Yarom, Yonatan Bitton, Chenfanfu Jiang, Yizhou Sun, Kai-Wei Chang, and Aditya Grover. Videophy: Evaluating physical commonsense for video generation. *arXiv preprint arXiv:2406.03520*, 2024.

Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, and Shenlong Wang. Physgen3d: Crafting a miniature interactive world from a single image. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6178–6189, 2025a.

Junyi Chen, Haoyi Zhu, Xianglong He, Yifan Wang, Jianjun Zhou, Wenzheng Chang, Yang Zhou, Zizun Li, Zhoujie Fu, Jiangmiao Pang, et al. Deepverse: 4d autoregressive video generation as a world model. *arXiv preprint arXiv:2506.01103*, 2025b.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13142–13153, 2023.

Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, and Jiajun Wu. Worldscore: A unified evaluation benchmark for world generation. *arXiv preprint arXiv:2504.00983*, 2025.

Quankai Gao, Qiangeng Xu, Zhe Cao, Ben Mildenhall, Wenchao Ma, Le Chen, Danhang Tang, and Ulrich Neumann. Gaussianflow: Splatting gaussian dynamics for 4d content creation. arXiv preprint arXiv:2403.12365, 2024.

Yue Gao, Hong-Xing Yu, Bo Zhu, and Jiajun Wu. Fluidnexus: 3d fluid reconstruction and prediction from a single video. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 26091–26101, 2025.

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101, 2024.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Yuanming Hu, Yu Fang, Ziheng Ge, Ziyin Qu, Yixin Zhu, Andre Pradhana, and Chenfanfu Jiang. A
moving least squares material point method with displacement discontinuity and two-way rigid body coupling. *ACM Transactions on Graphics (TOG)*, 37(4):1–14, 2018.

Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson WH
Lau. Dreamphysics: Learning physics-based 3d dynamics with video diffusion priors. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3733–3741, 2025.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and* Pattern Recognition, pp. 21807–21818, 2024.

Chenfanfu Jiang, Theodore Gast, and Joseph Teran. Anisotropic elastoplasticity for cloth, knit and hair frictional contact. *ACM Transactions on Graphics (TOG)*, pp. 1–14, 2017.

Yanqin Jiang, Li Zhang, Jin Gao, Weimin Hu, and Yao Yao. Consistent4d: Consistent 360 {\deg}
dynamic object generation from monocular video. *arXiv preprint arXiv:2311.02848*, 2023.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, and George Drettakis. 3d gaussian splat- ¨
ting for real-time radiance field rendering. *ACM Trans. Graph.*, pp. 139–1, 2023.

Diederik P Kingma. Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*,
2014.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.

Renjie Li, Panwang Pan, Bangbang Yang, Dejia Xu, Shijie Zhou, Xuanyang Zhang, Zeming Li, Achuta Kadambi, Zhangyang Wang, Zhengzhong Tu, et al. 4k4dgen: Panoramic 4d generation at 4k resolution. *arXiv preprint arXiv:2406.13527*, 2024a.

Zhiqi Li, Yiming Chen, and Peidong Liu. Dreammesh4d: Video-to-4d generation with sparsecontrolled gaussian-mesh hybrid representation. Advances in Neural Information Processing Systems, 37:21377–21400, 2024b.

Hanwen Liang, Yuyang Yin, Dejia Xu, Hanxue Liang, Zhangyang Wang, Konstantinos N Plataniotis, Yao Zhao, and Yunchao Wei. Diffusion4d: Fast spatial-temporal consistent 4d generation via video diffusion models. *arXiv preprint arXiv:2405.16645*, 2024.

Jiajing Lin, Zhenzhong Wang, Yongjie Hou, Yuzhou Tang, and Min Jiang. Phy124: Fast physicsdriven 4d content generation from a single image. *arXiv preprint arXiv:2409.07179*, 2024a.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jiajing Lin, Zhenzhong Wang, Dejun Xu, Shu Jiang, YunPeng Gong, and Min Jiang. Phys4dgen:
Physics-compliant 4d generation with multi-material composition perception. *arXiv preprint* arXiv:2411.16800, 2024b.

Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong Mu. Omniphysgs: 3d constitutive gaussians for general physics-based dynamics generation. *arXiv preprint arXiv:2501.18982*, 2025.

Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan.

Physics3d: Learning physical properties of 3d gaussians via video diffusion. *arXiv preprint* arXiv:2406.04338, 2024a.

Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In *European Conference on Computer Vision*, pp. 360–378.

Springer, 2024b.

Zeyi Liu, Shuang Li, Eric Cousineau, Siyuan Feng, Benjamin Burchfiel, and Shuran Song.

Geometry-aware 4d video generation for robot manipulation. *arXiv preprint arXiv:2507.01099*, 2025a.

Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, and Di Zhang. Unleashing the potential of multi-modal foundation models and video diffusion for 4d dynamic physical scene simulation. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 11016–11025, 2025b.

Yue Ma, Kunyu Feng, Xinhua Zhang, Hongyu Liu, David Junhao Zhang, Jinbo Xing, Yinhan Zhang, Ayden Yang, Zeyu Wang, and Qifeng Chen. Follow-your-creation: Empowering 4d creation through video inpainting. *arXiv preprint arXiv:2506.04590*, 2025.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Haotian Mao, Zhuoxiong Xu, Siyue Wei, Yule Quan, Nianchen Deng, and Xubo Yang. Live-gs:
Llm powers interactive vr by enhancing gaussian splatting. In 2025 IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW), pp. 1234–1235. IEEE, 2025.

Koichi Namekata, Sherwin Bahmani, Ziyi Wu, Yash Kant, Igor Gilitschenski, and David B Lindell. Sg-i2v: Self-guided trajectory control in image-to-video generation. arXiv preprint arXiv:2411.04989, 2024.

Kepan Nan, Rui Xie, Penghao Zhou, Tiehan Fan, Zhenheng Yang, Zhijie Chen, Xiang Li, Jian Yang, and Ying Tai. Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371, 2024.

OpenAI. Sora, 2024. URL https://openai.com/sora. Accessed: 2025-09-15. Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M
Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5865–5874, 2021.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. *arXiv preprint arXiv:2209.14988*, 2022.

Runway. Runway, 2024. URL https://runwayml.com. Accessed: 2025-09-15.

JC Simo and TJR Hughes. *Computational Inelasticity*, volume 7. Springer Science & Business Media, 2006.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. *arXiv* preprint arXiv:2010.02502, 2020.

Alexey Stomakhin, Craig Schroeder, Lawrence Chai, Joseph Teran, and Andrew Selle. A material point method for snow simulation. *ACM Transactions on Graphics (TOG)*, 32(4):1–10, 2013.

Xiyang Tan, Ying Jiang, Xuan Li, Zeshun Zong, Tianyi Xie, Yin Yang, and Chenfanfu Jiang. Physmotion: Physics-grounded dynamics from a single image. *arXiv preprint arXiv:2411.17189*, 2024.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. *arXiv preprint arXiv:2503.20314*, 2025.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In *Proceedings* of the IEEE/CVF conference on computer vision and pattern recognition, pp. 20310–20320, 2024.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In *Proceedings of the Computer Vision and Pattern Recognition Conference*, pp. 21469–21480, 2025.

Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. Advances in Neural Information Processing Systems, 37:56828–56858, 2024.

Jiawei Ren, Liang Pan, Jiaxiang Tang, Chi Zhang, Ang Cao, Gang Zeng, and Ziwei Liu. Dreamgaussian4d: Generative 4d gaussian splatting. *arXiv preprint arXiv:2312.17142*, 2023.

Cong Wang, Xianda Guo, Wenbo Xu, Wei Tian, Ruiqi Song, Chenming Zhang, Lingxi Li, and Long Chen. Drivesplat: Decoupled driving scene reconstruction with geometry-enhanced partitioned neural gaussians. *arXiv preprint arXiv:2508.15376*, 2025.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. *arXiv preprint arXiv:2508.02324*,
2025.
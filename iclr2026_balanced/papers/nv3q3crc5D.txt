# FLASH-MONO: FEED-FORWARD ACCELERATED GAUSSIAN SPLATTING MONOCULAR SLAM

**Zicheng Zhang** [1] **, Ke Wu** [1] **, Xiangting Meng** [2] **, Keyu Liu** [3] **, Jieru Zhao** [3] _[∗]_ **, Wenchao Ding** [1] _[∗]_

1Fudan University 2ShanghaiTech University 3Shanghai Jiao Tong University
[https://victkk.github.io/flash-mono](https://victkk.github.io/flash-mono)


Figure 1: **Our Results for Reconstruction and Rendering & Tracking & Speed Metrics.**
Our method reconstructs high-quality Gaussian maps in complex scenes with multiple rooms and
varying lighting conditions. The right-side radar chart shows our rendering quality (PSNR, SSIM,
LPIPS) and trajectory tracking accuracy (ATE), with reciprocals of LPIPS, ATE, and Depth L1
plotted for clarity. Our method outperforms others in both rendering quality and trajectory accuracy,
offering a **10x** speedup over contemporary monocular GS-SLAM methods.


ABSTRACT


Monocular Gaussian Splatting SLAM suffers from critical limitations in time efficiency, geometric accuracy, and multi-view consistency. These issues stem from
the time-consuming _Train-from-Scratch_ optimization and the lack of inter-frame
scale consistency from single-frame geometry priors. We contend that a feedforward paradigm, leveraging multi-frame context to predict Gaussian attributes
directly, is crucial for addressing these challenges. We present Flash-Mono, a
system composed of three core modules: a feed-forward prediction frontend, a
2D Gaussian Splatting mapping backend, and an efficient hidden-state-based loop
closure module. We trained a recurrent feed-forward frontend model that progressively aggregates multi-frame visual features into a hidden state via cross attention
and jointly predicts camera poses and per-pixel Gaussian properties. By directly
predicting Gaussian attributes, our method bypasses the burdensome per-frame
optimization required in optimization-based GS-SLAM, achieving a **10x** speedup
while ensuring high-quality rendering. The power of our recurrent architecture
extends beyond efficient prediction. The hidden states act as compact submap
descriptors, facilitating efficient loop closure and global Sim(3) optimization to
mitigate the long-standing challenge of drift. For enhanced geometric fidelity, we
replace conventional 3D Gaussian ellipsoids with 2D Gaussian surfels. Extensive


_∗_ Corresponding author.


1


experiments demonstrate that Flash-Mono achieves state-of-the-art performance
in both tracking and mapping quality, highlighting its potential for embodied perception and real-time reconstruction applications.


1 INTRODUCTION


Recent advancements in real-time 3D scene reconstruction using a single RGB camera have attracted
considerable attention. Its ability to provide dense and information-rich maps is crucial for applications ranging from robotic navigation and spatial intelligence. While traditional representations
like point clouds, voxels, and surfels have been widely used, 3D Gaussian Splatting (3DGS) (Kerbl
et al., 2023) has recently emerged as a highly promising approach for 3D reconstruction, owing
to its capabilities in differentiable rendering, self-supervised training from RGB images, and highfidelity novel view synthesis. Consequently, integrating 3DGS into a real-time monocular SLAM
framework presents a significant opportunity for advancing embodied perception.


An early attempt at monocular GS-SLAM (Matsuki et al., 2024) initializes Gaussians randomly and
relies on hundreds of optimization iterations per frame to maintain a consistent map. Subsequent
methods (Zheng et al., 2025; Wu et al., 2025a) employ depth or optical flow prediction networks
to provide geometric priors, which are used to initialize Gaussian geometric attributes. However,
their performance remains limited to around 1 FPS, insufficient for real-time SLAM, as they do not
abandon the _Train-from-Scratch_ paradigm (Gaussian appearance attributes are randomly initialized
and trained). Moreover, these approaches also suffer from severe multi-view inconsistencies, as
monocular depth predictions are inherently scale-inconsistent. On a different front, feed-forward
methods such as VGGT (Wang et al., 2025a) have demonstrated excellent multi-frame consistency
by applying cross-attention over image batches. While the feed-forward approach supplies a consistent geometric prior, its offline requirement of processing all frames at once makes it fundamentally
incompatible with the streaming input and low-latency pose estimation required by SLAM.


Based on this analysis, we identify three critical challenges that impede the development of a truly
real-time and globally consistent monocular GS-SLAM system. First, the prevalent _Train-from-_
_Scratch_ paradigm of Gaussian Splatting requires dozens to hundreds of iterations of optimization
per keyframe, fundamentally preventing real-time performance. Second, incremental feed-forward
reconstruction methods are susceptible to cumulative pose and scale drift, as past predictions cannot
be refined by future observations, leading to poor multi-frame geometric consistency. Third, vanilla
3DGS representations often suffer from poor geometry quality.


To overcome these challenges, we propose **Flash-Mono**, a monocular GS-SLAM system designed
to deliver exceptional speed performance and high-quality mapping. At its core is a recurrent feedforward reconstruction model that incrementally predicts camera poses together with a dense, pixelaligned Gaussian representation for each incoming frame. This design directly addresses the efficiency bottleneck of optimization-based GS-SLAM: instead of training Gaussians from scratch
at every keyframe, we predict high-quality Gaussians and only apply lightweight backend refinement. To combat the drift that is common in incremental feed-forward reconstruction, we leverage
the model’s hidden state as a compact submap descriptor: when revisiting a location, a single conditional forward pass produces an accurate Sim(3) loop constraint, which we integrate into pose graph
optimization for global correction. Finally, to improve geometric fidelity, we adopt 2D Gaussian
surfels as our map primitive, providing a stronger surface prior than vanilla 3DGS. With these components, Flash-Mono supports streaming inputs while achieving real-time performance and globally
consistent reconstructions. In summary, our main contributions are:


    - We propose a real-time (10 FPS+) monocular GS-SLAM framework that leverages a recurrent feed-forward model to predict poses and Gaussians directly. Compared to all previous
methods that require training Gaussians entirely from scratch, our framework achieves remarkable speed improvements while still ensuring high-quality results.

    - We design a novel and efficient loop closure method based on the hidden state of the feedforward model, and through Sim(3) graph optimization, we mitigate accumulated errors
while preserving the global consistency of the reconstructed map.

    - We conduct extensive experiments on large-scale and challenging datasets, evaluating rendering, geometry, tracking, and efficiency metrics. Our work achieves state-of-the-art re

2


sults in both tracking and rendering quality, while significantly surpassing previous methods in processing speed.


2 RELATED WORKS


**SLAM** **with** **3D** **Foundation** **Model.** Feed-forward architectures have recently emerged as a
powerful alternative to classical Structure-from-Motion (SfM) pipelines, which rely on iterative
feature matching and bundle adjustment (Schönberger & Frahm, 2016). Early works such as
DUSt3R (Wang et al., 2024) and its extension MASt3R (Murai et al., 2025) pioneered this paradigm
by directly predicting point maps from image pairs within a single forward pass. To overcome
the limitation of pairwise inputs, Fast3R (Yang et al., 2025) introduced a transformer-based design
capable of processing multiple images in parallel, thereby accelerating large-scale 3D reconstruction. CUT3R (Wang et al., 2025b) further advanced this direction by adopting a recurrent framework that accommodates a variable number of images and supports diverse input modalities, enabling online processing of video streams. Extending this line of work, VGGT (Wang et al., 2025a)
demonstrated the potential of large-scale, multi-task learning for feed-forward reconstruction, while
FLARE (Zhang et al., 2025) and Splatt3r (Smart et al., 2024) extended the idea to renderable Gaussian Splatting representations directly from unposed images.


Nevertheless, directly applying feed-forward methods to SLAM remains highly challenging due to
the need for accurate pose consistency, temporal stability, and long-horizon robustness. For example, although MASt3R-SLAM (Murai et al., 2025) partially mitigates some of these issues with
improved correspondence strategies, its design is not tailored for persistent SLAM. Later, VGGTSLAM (Maggio et al., 2025) builds on the strong backbone of VGGT (Wang et al., 2025a), feeding
submaps into it and optimizing poses on the SL(4) manifold to achieve more accurate tracking.


**Monocular** **GS-SLAM.** 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) has recently gained
attention in monocular SLAM research due to its differentiable nature and real-time rendering efficiency. MonoGS (Matsuki et al., 2024) and PhotoSLAM (Huang et al.) are early monocular
GS-SLAM methods that initialize Gaussian ellipsoids through feature points or random sampling
and incorporate ORB-SLAM3 (Mur-Artal et al., 2015) for pose estimation, enabling applications
in small indoor environments. SEGS-SLAM (Tianci Wen, 2025) further enhances structural consistency by modeling appearance variations. DroidSplat (Homeyer et al., 2025) leverages dense
optical flow and depth priors to achieve robust tracking and reconstruction. However, these monocular systems suffer from scalability issues and often generate floating artifacts in dynamic or largescale scenes. Building upon these limitations, approaches like WildGS-SLAM (Zheng et al., 2025),
DepthGS (Zhao et al., 2025), and Dy3DGS-SLAM (Li et al., 2025) introduced geometry prior and
pixel-level uncertainty estimation to enhance robustness in real-world dynamic scenes. Furthermore,
S3PO-GS (Cheng et al., 2025) addresses the challenges of scale drift and the lack of geometric priors commonly encountered in outdoor scenarios by introducing a scale self-consistent pointmap.
However, existing GS-based SLAM methods are generally limited to around 1 FPS, which is clearly
insufficient to meet the inherent real-time requirements of SLAM. The main reason lies in the fact
that these methods train the Gaussians from scratch for each keyframe, typically requiring tens to
hundreds of iterations. Since a single iteration takes approximately 20 ms, the total training time per
keyframe is roughly one second, inevitably resulting in slow overall performance.


3 PRELIMINARIES: 2D GAUSSIAN FOR GEOMETRIC ACCURACY


The original 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) often produces noisy geometry
with “floater” artifacts, as its volumetric primitives lack explicit surface constraints. To address
this, 2D Gaussian Splatting (2DGS) was introduced in (Huang et al., 2024), representing scenes
as a collection of 2D planar Gaussian surfels. Their work demonstrated that this representation
provides stronger geometric priors, yielding significantly improved surface accuracy and multi-view
consistency over 3DGS.


We adopt 2DGS as scene representation, where each surfel is defined by its position ( _**µ**_ ), color ( _**c**_ ),
opacity ( _σ_ ), rotation ( _**r**_ ), and 2D scale ( _**s**_ ). The final pixel color ( _I_ [ˆ] ), depth ( _D_ [ˆ] ), and accumulation


3


Figure 2: **Pipeline.** For each new frame, our recurrent model jointly infers the camera pose and perpixel 2DGS attributes conditioned on a hidden state. The hidden state is updated simultaneously. To
avoid catastrophic forgetting, the stream is partitioned into submaps. The hidden state is reinitialized
for each submap. Past hidden states are cached in the Bag of Hidden States. Upon loop detection,
i.e., revisiting a location, we perform a single forward pass on the loop frame conditioned on the past
hidden state to relocalize the current frame in the past submap. A following pose graph optimization
is then performed to correct the full trajectory. In the backend, per-frame 2DGS attributes prediction
is voxelized, merged, and refined to build a global 2DGS map.


( _A_ [ˆ] ) are rendered via volumetric alpha blending:


            -             _wi_ ( _p_ ) = _σi ·_ exp _−_ [1] _i_ [(] _[p][ −]_ _**[µ]**_ _[i]_ [)]

2 [(] _[p][ −]_ _**[µ]**_ _[i]_ [)] _[T]_ [ Σ] _[−]_ [1]


_i−_ 1 (1)
�(1 _−_ _wj_ )


_j_ =1


( _I,_ [ˆ] _D,_ [ˆ] _A_ [ˆ] ) =


_N_
�( _**c**_ _i,_ _**z**_ _i,_ 1) _wi_


_i_ =1


Here, _p_ denotes a pixel coordinate, Σ _i_ _∈_ R [2] _[×]_ [2] is the screen-space covariance induced by the surfel’s
rotation _**r**_ _i_ and 2D scale _**s**_ _i_, and _**z**_ _i_ is the surfel depth along the camera optical axis used for depth
accumulation. Compared to 3D Gaussian ellipsoids, the planar 2DGS representation provides a
stronger surface prior that suppresses floaters and improves geometric fidelity, which is particularly
beneficial for SLAM where small geometric inconsistencies can quickly accumulate into drift. In
the remainder of this paper, we use 2DGS as our map primitive: our recurrent feed-forward frontend
predicts per-pixel surfel attributes in the current camera frame, and our backend incrementally fuses
and refines these predictions into a global, renderable map that can be efficiently updated after pose
graph optimization.


4 OUR APPROACH


In this section, we introduce our approach in the following order. We first describe our recurrent
feed-forward frontend, which constitutes the core of our system by incrementally estimating camera
poses and per-frame 2DGS attributes (§4.1). We then present our loop closure mechanism, which
leverages the model’s hidden state to enable global drift correction via Sim(3) optimization (§4.2).
Finally, we detail the backend mapping method that incrementally fuses the frontend’s raw predictions into a globally consistent 2DGS map (§4.3).


4.1 RECURRENT FEED-FORWARD FRONTEND MODEL


The input of our system is a monocular RGB stream _{It}_ . For each incoming frame _It_ _∈_ R _[H][×][W][ ×]_ [3]
at timestep _t_, our feed-forward model, denoted by _f_, takes the current frame and a hidden state _Mt−_ 1


4


as input. The function of model _f_ is to jointly predict three outputs: (a) the camera pose _T_ [ˆ] _t_ _∈_ SE(3),
representing the transformation from the current camera frame to the coordinate system of the initial
frame ( _t_ = 1); (b) a dense, pixel-aligned 2DGS map _G_ [ˆ] _t_ = _{Gn}_ _[H]_ _n_ =1 _[×][W]_ [, where the attributes of each]
Gaussian surfel are defined in the local coordinate system of the current camera; and (c) an updated
hidden state _Mt_, which carries aggregated information forward to the next timestep (the initial state
_M_ 0 is initialized to zero). Formally, the per-frame prediction process is expressed as:

_T_ ˆ _t,_ _G_ ˆ _t, Mt_ = _f_ ( _It, Mt−_ 1) (2)


**Model** **Architecture.** Inspired by Wang et al. (2025b) and Wu et al. (2025b), we design a stateful transformer architecture to incrementally reconstruct the scene. Each incoming image is first
converted into a set of visual tokens _Ft_ _∈_ R _[K][×][C]_ by a ViT encoder. The model then employs two
interconnected decoders that facilitate bidirectional information exchange between visual tokens _Ft_
and the persistent hidden state _Mt−_ 1 via cross-attention. A learnable pose token _zt_, concatenated
with _Ft_, is processed by the decoders to aggregate geometric cues for pose estimation. This fusion
can be expressed as:


_Ft_ = Encoder( _It_ ) (3)

_Ft_ _[′][, z]_ _t_ _[′][, M][t]_ [= Decoders((] _[F][t][, z][t]_ [)] _[, M][t][−]_ [1][)] (4)

Finally, two DPT heads (Ranftl et al., 2021) decode the image tokens _Ft_ _[′]_ [to predict 2DGS attributes:]
the means and confidences _{_ _**µ**_ ˆ _t,_ _C_ [ˆ] _t}_, and other parameters _{_ _**σ**_ ˆ _t,_ ˆ _**r**_ _t,_ ˆ _**s**_ _t,_ ˆ _**c**_ _t}_ . Concurrently, an MLP
head extracts the absolute camera pose _T_ [ˆ] _t_ from the output pose token _zt_ _[′]_ [.]

_**µ**_ ˆ _t,_ _C_ [ˆ] _t_ = Headmeans( _Ft_ _[′]_ [)] (5)

_**σ**_ ˆ _t,_ ˆ _**r**_ _t,_ ˆ _**s**_ _t,_ ˆ _**c**_ _t_ = Headgs( _Ft_ _[′]_ [)] (6)
_T_ ˆ _t_ = Headpose( _zt_ _[′]_ [)] (7)


**Training** **Objective.** Our model is trained on large-scale datasets with ground-truth RGB, depth,
and camera pose data. The training objective consists of three loss components, summed over a
sequence of length _L_ . The predicted pose _T_ [ˆ] _t_ is parameterized as a quaternion _q_ ˆ _t_ and a translation
vector _τ_ ˆ _t_ . The total loss is a weighted sum of the pose loss, geometric loss, and rendering loss:


_L_ total = _λ_ pose _L_ pose + _λ_ geo _L_ geo + _L_ render (8)


_L_ pose =


_L_ geo =


_L_ render =


_L_

- ( _∥q_ ˆ _t −_ _qt∥_ 2 + _∥τ_ ˆ _t −_ _τt∥_ 2) (9)


_t_ =1


_L_


_t_ =1


_L_


_t_ =1


- _λmse∥It −_ _I_ [ˆ] _t∥_ [2] 2 [+] _[ λ][lpips][L][lpips]_ [(] _[I][t][,]_ [ ˆ] _[I][t]_ [) +] _[ λ][depth][∥][D][t]_ _[−]_ _[D]_ [ˆ] _[t][∥]_ [2] 2 (11)


_H×W_


_n_ =1


- _c_ ˆ _t,n · ∥µ_ ˆ _t,n −_ _µt,n∥_ 2 _−_ _α_ log(ˆ _ct,n_ )� (10)


where _I_ [ˆ] _t_ and _D_ [ˆ] _t_ are the RGB and depth rendered from the predicted 2DGS map _G_ [ˆ] _t_ through standard
rasterization as described in §3. Here, ( _qt, τt_ ), _It_, and _Dt_ denote the ground-truth camera pose, RGB
image, and depth map, respectively; _µ_ is obtained by unprojecting the ground-truth depth map using
the camera intrinsics. Our model is trained on datasets including DL3DV and ScanNet++, which
cover both indoor and outdoor scenes. Please refer to Appendix D for the detailed training setup,
and Appendix C.2 for model acceleration.


**Incremental** **Tracking** **with** **Submaps.** While our model can theoretically process an arbitrarily
long sequence, we observe in practice that cumulative drift increases with sequence length, _L_, a result of catastrophic forgetting in recurrent models. To ensure robust tracking, we partition the input
stream into shorter subsequences (submaps). For each submap, the hidden state is re-initialized;
consequently, all predicted poses _{T_ [ˆ] _t}_ are expressed in the coordinate frame of that submap’s first
frame. A one-frame overlap between consecutive submaps allows us to compute the relative transformation between them, enabling us to chain the local pose estimates into a continuous trajectory.
This overlap also provides an explicit inter-submap alignment constraint, which is later incorporated
into the pose graph for global optimization.


5


4.2 LOOP CLOSURE VIA HIDDEN STATE


Monocular SLAM systems inevitably suffer from accumulated pose and scale drift. To address
this, we introduce a novel mechanism to compute a geometric constraint between the current frame
and a past frame with a single forward pass when the camera revisits a previously mapped area (a
loop closure (Tsintotas et al., 2022)). This enables pose graph optimization to produce a globally
consistent trajectory, mitigating the long-standing problem of drift.


**Bag of Hidden States as Long-Term Memory.** As our recurrent model processes frames sequentially, the hidden state, _M_, incrementally aggregates local, multi-frame geometric and visual information. The hidden state becomes a rich, contextual summary of the local scene that the system has
just observed. We leverage this by caching the final hidden state _Ma_ for each submap _Ca_ in a bag
of hidden states. When the system later revisits an area, it can retrieve a historical hidden state from
the bag of hidden states to reload past geometric and visual context.


**Loop Frame Feed-Forward for Relocalization.** The process is triggered when a loop candidate is
detected between the current keyframe _Ij_ of submap _Cb_ and a historical keyframe _Ii_ of submap _Ca_
using an appearance-based method (Izquierdo & Civera, 2024). We retrieve the cached hidden state
_Ma_ associated with the historical submap _Ca_ containing keyframe _Ii_ . Intuitively, conditioning on
_Ma_ encourages the model to interpret _Ij_ in the coordinate system of the past submap _Ca_, yielding a
direct cross-submap constraint. A single forward pass _f_ ( _Ij, Ma_ ) on the current frame _Ij_ conditioned
on this past context yields two key outputs: (1) the relocalized pose **T** _[a]_ _j_ _[∈]_ [SE(3)][,] [and] [(2)] [the]
corresponding point cloud interpretation _Pj_ _[a]_ [=] _[ {]_ _**[µ]**_ _k_ _[a][}]_ [. The relative pose is then computed as] **[ T]** _[j][→][i]_ [=]
( **T** _[a]_ _j_ [)] _[−]_ [1] **[T]** _i_ _[a]_ [.] [To resolve scale ambiguity, we compare this historical interpretation] _[ P]_ _j_ _[a]_ [against] _[ P]_ _j_ _[b]_ [=]
_{_ _**µ**_ _[b]_ _k_ _[}]_ [, which is the point cloud generated from the standard, incremental tracking of frame] _[ I][j]_ [(i.e.,]
using the current hidden state). Crucially, since both point clouds are in the camera coordinate frame
and originate from the same image _Ij_, they differ only by a scale factor. This allows us to robustly
solve for the relative scale _s_ _[∗]_ via least-squares:


_b_ 2
�� _**µ**_ _k_ _[−]_ _[s][ ·]_ _**[ µ]**_ _k_ _[a]_ �� _._ (12)


_s_ _[∗]_ = argmin
_s_


_k_


Finally, the estimated scale and relative pose are combined into the complete Sim(3) loop closure
constraint:


**H** _j→i_ = - _s∗R_ **0** _[T]_ _j→i_ _tj_ 1 _→i_


(13)


where _Rj→i_ and _tj→i_ are the rotation and translation components of **T** _j→i_ .


**Pose** **Graph** **Optimization.** The computed Sim(3) constraint enables global optimization of the
entire trajectory via a pose graph. In this graph, nodes represent the keyframe poses **T** _[W]_ _k_ _[∈]_ [Sim(3)][,]
and edges represent three types of geometric constraints: _**Sequential**_ _**Constraint**_, a factor linking
consecutive frames within a submap, computed as the relative transformation ( **T** _[−]_ _k_ [1] **[T]** _[k]_ [+1][)][;] _**[Inter-]**_
_**Submap Constraint**_, an alignment factor connecting adjacent submaps, which is derived by estimating the relative scale between the two point cloud predictions of their shared frame; and our novel
_**Loop Closure**_ factors that connect distant, revisited parts of the trajectory. The globally optimal set
of poses _T_ _[W][ ∗]_ is found by minimizing a non-linear least-squares cost function over all constraints:


_T_ _[W][ ∗]_ = arg min
_T_ _[W]_


( _i,j_ ) _∈E_


��log - **H** _[−]_ _j→_ [1] _i_ _[·]_ [ ((] **[T]** _i_ _[W]_ [)] _[−]_ [1] _[ ·]_ **[ T]** _j_ _[W]_ [)] ���2Ω (14)


where the residual error is computed in the Lie algebra sim(3) using the logarithmic map log( _·_ ).
This formulation finds the trajectory that best satisfies all geometric constraints simultaneously. We
solve this efficiently using GTSAM (Dellaert & Contributors, 2022), and the resulting corrected
poses are passed to the backend to update the 2DGS map, as detailed in §4.3.


4.3 2DGS MAP OPTIMIZATION


The backend runs in a separate thread and incrementally builds and optimizes a globally consistent
2DGS map. For each new keyframe, it takes as input the RGB image _Ik_, the globally optimized camera pose _Tk_ _∈_ Sim(3), and the per-pixel 2DGS map _G_ [ˆ] _k_ of _Ik_ predicted by the frontend. The backend


6


pipeline then consists of four key stages: (1) pre-processing the dense predictions via adaptive voxelization; (2) merging the 2DGS map of new frame into the global map; (3) applying a lightweight
local refinement; and (4) executing global map corrections after a successful loop closure.


**Adaptive** **Voxelization.** We empirically found that the per-pixel 2DGS predicted by the frontend
is sometimes overly dense. To reduce memory consumption, we first process each incoming 2DGS
map _G_ [ˆ] _k_ with an adaptive voxelization filter prior to merging. The map is partitioned into blocks of
2 _×_ 2 2DGS primitives. Primitives within each block are consolidated into a single merged primitive
by averaging their attributes:


_**θ**_ merged = [1]

_N_


_N_

- _**θ**_ _n,_ for _**θ**_ _∈{_ _**µ**_ _,_ _**σ**_ _,_ _**c**_ _,_ _**s**_ _}._ (15)


_n_ =1


   - _N_
_n_ =1 [align][(] _**[r]**_ _[n][,]_ _**[ r]**_ [1][)]
_**r**_ merged = (16)

_||_ [�] _n_ _[N]_ =1 [align][(] _**[r]**_ _[n][,]_ _**[ r]**_ [1][)] _[||]_


where align( _·_ ) is the standard process to ensure consistent quaternion averaging. To preserve geometric details, blocks with a depth variation exceeding a threshold _τd_ are excluded from this process.


**Map Fusion.** With each new frame, we first maintain the existing global map by pruning erroneous
Gaussians. This is done by rendering the map from the current camera pose _Tk_ using the formula
from §3, and removing any primitive that contributes to pixels with high RGB or depth reconstruction error. Subsequently, the incoming voxelized 2DGS primitives are fused into the global map.
First, they are transformed from camera to world coordinates:


_µ_ world = _skRkµ_ cam + _tk,_ _r_ world = _qk · r_ cam (17)


where ( _sk, Rk, tk_ ) are components of _Tk_ ; _qk_ is the quaternion form of _Rk_ . To avoid redundant densification, we only add these new primitives in regions that are not yet well-reconstructed, identified
by rendering an accumulation map from the global map and checking against a threshold _τaccum_ .


**Lightweight** **Map** **Refinement.** A key advantage of our _Predict-and-Refine_ paradigm is the dramatically reduced optimization workload for the backend. The high-quality per-frame predictions
from our frontend serve as a strong geometric and appearance prior. Consequently, after the fusion
step, we only need to refine the local map region associated with the latest _K_ keyframes for only 20
iterations. This stands in stark contrast to existing 3DGS-SLAM methods that require hundreds to
thousands of optimization iterations per frame, especially during initialization.


**Loop** **Correction** **of** **Gaussian** **Map.** As described in §4.2, upon receiving the set of globally
optimized poses _T_ _[W][ ∗]_ after a loop closure, the backend initiates map correction to ensure the 2DGS
map aligns with the corrected trajectory. A naive approach would be to re-run rendering-based
optimization using the corrected poses, but we found this to be prohibitively slow. Therefore, we
adopt a more efficient rigid transformation strategy. We rigidly bind the 2DGS primitives to their
originating keyframe. When a keyframe’s pose is updated from _T_ old to _T_ new, we compute the delta
transformation ∆ _T_ = _T_ new _· T_ old _[−]_ [1] [and apply it to all associated primitives.] [This process efficiently]
warps the map to align with the corrected trajectory without costly re-rendering.


5 EXPERIMENTS


5.1 EXPERIMENTAL SETUP


We evaluate our system on three challenging real-world datasets: ScanNet (Dai et al., 2017a),
BundleFusion (Dai et al., 2017b), and KITTI (Geiger et al., 2012). ScanNet and BundleFusion
consist of large-scale indoor scenes with motion blur and diverse lighting conditions. We treat
ScanNet as in-domain evaluation and BundleFusion as out-of-domain evaluation. KITTI features
large-scale outdoor driving scenarios with high scale variance and dynamic objects. We evaluate
tracking accuracy using Absolute Trajectory Error (ATE RMSE) and rendering quality via PSNR,
SSIM, and LPIPS. Since monocular SLAM has inherent scale ambiguity, we compute ATE after
Sim(3) alignment to ground truth. For ScanNet and BundleFusion, we further evaluate geometric
quality with scale-aligned Depth _L_ 1 error. All experiments are conducted on a single RTX 4090
GPU paired with an Intel Xeon 6133 CPU (2.50GHz).


7


Figure 3: **Qualitative Rendering Results.**


**Baselines.** We compare Flash-Mono with three state-of-the-art monocular GS-SLAM systems on
both mapping and tracking quality: MonoGS (Matsuki et al., 2024), DepthGS (Zhao et al., 2025),
and S3POGS (Cheng et al., 2025). We also compare against leading monocular SLAM systems
renowned for pose accuracy, although they do not produce dense renderings, including ORBSLAM3 (Campos et al., 2021), DROID-SLAM (Teed & Deng, 2021), and MASt3R-SLAM (Murai
et al., 2025). On KITTI, we primarily compare against S3POGS, as we encountered frequent failures
while evaluating other indoor-focused GS-SLAM baselines due to the large-scale and high dynamic
nature of KITTI.


5.2 TRACKING PERFORMANCE


As shown in Table 1, Flash-Mono significantly outperformed all traditional and GS-SLAM baseline methods. On most scenes, we also surpassed MASt3R-SLAM, a recent feed-forward SLAM
system. This validates the effectiveness of multi-frame context and our novel hidden-state-based
relocalization mechanism.


Table 1: ATE RMSE (cm) on **ScanNetV1** and **BundleFusion** datasets. Lower is better. We mark
the **first** and second best results.


**ScanNetV1** **BundleFusion**
**ATE [cm]** _↓_

**0054** **0059** **0106** **0169** **0233** **0465** **apt0** **apt2** **copyroom** **office0** **office2**


ORB-SLAM3 243.26 90.67 178.13 60.15 25.01 181.86 87.37 265.64 27.60 116.33 49.33
DROID-SLAM 161.22 69.92 89.11 28.26 74.01 117.27 89.38 148.04 19.71 31.41 73.91
MonoGS 70.19 97.24 150.89 191.98 62.45 113.19 122.59 142.54 53.41 62.67 127.02
DepthGS 192.18 93.69 140.19 205.92 81.90 121.01 67.52 119.74 14.59 40.42 16.05
S3PO-GS 69.36 16.52 26.15 87.04 27.09 96.35 92.49 97.90 21.88 64.22 69.88
MASt3R-SLAM 13.25 10.89 15.83 15.24 **10.99** 15.74 **9.65** 13.66 9.28 9.97 9.92
Ours **11.69** **8.89** **10.83** **10.16** 12.13 **13.00** 11.44 **12.36** **7.34** **8.74** **9.34**


5.3 MAPPING PERFORMANCE


Table 2 presents the rendering quality results. Although we perform only 20 optimization iterations
per keyframe (a **10x** reduction compared to the 250 iterations used by MonoGS (Matsuki et al., 2024)
and S3PO-GS (Cheng et al., 2025)), our method achieves superior or competitive rendering quality.


8


Table 2: Mapping quality on **ScanNetV1** and **BundleFusion** . Higher is better for SSIM/PSNR,
lower is better for LPIPS. We mark the **first** and second best results.


|ScanNetV1 BundleFusion<br>Method Metric<br>0054 0059 0106 0169 0233 0465 FPS ↑ apt0 apt2 copyroom offci e0 office2 FPS ↑|Col2|Col3|
|---|---|---|
|**MonoGS**<br>SSIM_ ↑_<br>LPIPS_ ↓_<br>PSNR_ ↑_|**0.80**<br>**0.74**<br>0.72<br>0.77<br>0_._68<br>0_._59<br>0.69<br>0.61<br>0_._60<br>0.54<br>0_._66<br>0.67<br>0.74<br>19_._24<br>16_._54<br>16_._09<br>**18.86**<br>17_._65<br>14.52|0.70<br>0_._39<br>0.70<br>0_._52<br>0.60<br>1.00<br>0_._67<br>0_._82<br>0_._63<br>0_._78<br>0_._71<br>13_._68<br>11_._50<br>14_._37<br>13_._38<br>13_._96|
|**DepthGS**<br>SSIM_ ↑_<br>LPIPS_ ↓_<br>PSNR_ ↑_|0_._31<br>0_._32<br>0_._34<br>0_._42<br>0_._36<br>0_._26<br>1.57<br>0_._79<br>0_._78<br>0_._78<br>0_._73<br>0_._84<br>0_._81<br>12_._29<br>12_._42<br>11_._76<br>13_._64<br>13_._17<br>11_._11|0_._38<br>0_._41<br>0_._58<br>0_._56<br>0_._58<br>1.28<br>0_._67<br>0.69<br>0.51<br>0.62<br>0.63<br>13_._65<br>14_._85<br>17_._00<br>15.96<br>16_._51|
|**S3PO-GS**<br>SSIM_ ↑_<br>LPIPS_ ↓_<br>PSNR_ ↑_|**0.80**<br>0.71<br>**0.75**<br>**0.78**<br>**0.73**<br>0.61<br>0.71<br>0_._62<br>0.58<br>0.54<br>0.55<br>0_._69<br>0_._75<br>20.79<br>17.19<br>17.60<br>18.52<br>18.37<br>14_._14|**0.74**<br>**0.64**<br>0_._47<br>0.63<br>**0.64**<br>0.94<br>0.57<br>0.71<br>0_._78<br>0_._71<br>0_._64<br>18.98<br>15.72<br>18.56<br>15_._23<br>16.59|
|**Ours**<br>SSIM_ ↑_<br>LPIPS_ ↓_<br>PSNR_ ↑_|0.79<br>0_._66<br>**0.72**<br>0_._73<br>0.69<br>**0.66**<br>**12.71**<br>**0.39**<br>**0.41**<br>**0.43**<br>**0.39**<br>**0.44**<br>**0.45**<br>**21.73**<br>**17.83**<br>**17.75**<br>18.52<br>**21.60**<br>**19.51**|0_._66<br>0.60<br>0.72<br>**0.69**<br>**0.64**<br>**11.99**<br>**0.49**<br>**0.54**<br>**0.45**<br>**0.50**<br>**0.51**<br>**19.03**<br>**16.48**<br>**19.50**<br>**17.10**<br>**17.63**|


Figure 4: **Qualitative Analysis on Rendered Depth.**


This highlights the effectiveness of our _Predict-and-Refine_ paradigm: high-quality Gaussians predicted by our foundation model reduce the need for costly backend optimization. The scale-aligned
Depth _L_ 1 error is evaluated in Table 5. We achieve a lower Depth L1 error, suggesting a more
accurate underlying 3D scene reconstruction. Qualitative rendered RGB and depth are presented in
Figure 3 and Figure 4.


5.4 OUTDOOR EVALUATION ON KITTI


We further evaluate Flash-Mono on the KITTI benchmark to assess generalization to large-scale
outdoor environments. Since MonoGS and DepthGS are designed primarily for indoor scenes, they
often fail under the large scale variance and dynamics in KITTI; therefore, we mainly compare with
S3PO-GS (Cheng et al., 2025), which is designed for outdoor scenarios. Table 3 reports tracking
accuracy and Table 4 reports rendering quality.


Table 3: ATE RMSE (m) on **KITTI Odometry** . Lower is better.


**ATE RMSE [m]** _↓_ **00** **05** **06** **07** **08** **28**


**Ours** **12.85** **16.58** **9.93** **12.08** **45.25** **16.75**
**S3PO-GS** 32.49 34.76 16.43 fail 64.74 23.64


9


Table 4: Rendering quality on **KITTI Odometry** . Higher is better for PSNR/SSIM, lower is better
for LPIPS.


**Method** **Metric** **00** **05** **06** **07** **08** **28**


PSNR _↑_ 16.65 15.64 13.55 fail **17.25** 15.30
**S3PO-GS** SSIM _↑_ 0.5409 0.5320 0.4726 fail 0.5912 0.5053
LPIPS _↓_ 0.6254 0.6352 0.7241 fail **0.4626** 0.6131


PSNR _↑_ **17.41** **17.01** **15.13** **17.89** 16.12 **17.47**
**Ours** SSIM _↑_ **0.6584** **0.6278** **0.5922** **0.6036** **0.6221** **0.5633**
LPIPS _↓_ **0.5358** **0.4871** **0.5333** **0.4854** 0.4710 **0.4581**


Table 5: Mean Depth L1 Error
(m) on **ScanNet** and **BundleFu-**
**sion** . We mark the **best** results.


**L1(m)** _↓_ **Scan.** **Bundle.**


Figure 5: **Ablation studies.** (a) Refine Iterations vs. PSNR.
(b) Submap Length vs. ATE RMSE. (c) Loop Closure Settings. (d) PSNR vs. Model Size.


5.5 ABLATION


MonoGS 1.19 1.20
DepthGS 0.49 0.23
S3PO-GS 0.52 0.85
Ours **0.34** **0.21**


We conducted ablation studies to analyze the impact of key system components. The results are
shown in Figure 5. First, we evaluated the effect of backend refinement iterations on rendering
quality (PSNR). Without refinement (0 iterations), the direct output from our feed-forward model
achieves a PSNR of 20.14. Applying 10 refinement iterations increases the PSNR to 22.41, indicating that the model provides a strong initial prediction that can be efficiently improved by a few
optimization steps. Second, we examined the influence of submap clip length on tracking accuracy (ATE RMSE). The lowest error of 0.106 was observed with a clip length of 8 frames. Shorter
lengths resulted in higher error, suggesting insufficient temporal context, while lengths greater than
16 frames also increased the error, which points to the accumulation of intra-submap drift caused
by the forgetting characteristic of RNN models. This supports the strategy of partitioning the input
stream. Third, we compared our hidden state-based loop closure against a traditional PnP+RANSAC
baseline and a configuration with no loop closure. Our original system beats the other two settings on
tracking performance by a large margin, suggesting our approach generates more accurate _Sim_ (3)
constraints. Finally, our adaptive voxelization module reduced the total number of Gaussian primitives by over 58% (from 1.35M to 0.56M), which corresponded to a minor PSNR decrease from
19.70 to 19.44. This demonstrates the module’s role in creating a more compact map representation
at a small cost to rendering fidelity.


6 CONCLUSION


We presented Flash-Mono, a real-time monocular Gaussian Splatting SLAM system that fundamentally shifts from the time-consuming _Train-from-Scratch_ paradigm to an efficient _Predict-and-Refine_
approach. As our experiments show, Flash-Mono achieves state-of-the-art rendering quality with a
**10x** reduction in computation time. Furthermore, we introduced a novel loop closure mechanism
that enables robust Sim(3) optimization to correct scale and pose drift inherent in monocular systems, leading to superior tracking accuracy on complex indoor scenes.


10


ACKNOWLEDGMENTS


This work was supported in part by the National Natural Science Foundation of China (NSFC) under
Grant 62403142, and in part by the Science and Technology Commission of Shanghai Municipality
under Grant 24511103100.


REFERENCES


Carlos Campos, Richard Elvira, Juan J. Gomez, José M. M. Montiel, and Juan D. Tardós. ORBSLAM3: An accurate open-source library for visual, visual-inertial and multi-map SLAM. _IEEE_
_Transactions on Robotics_, 37(6):1874–1890, 2021.


Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction
as test-time training. _arXiv preprint arXiv:2509.26645_, 2025.


Chong Cheng, Sicheng Yu, Zijian Wang, Yifan Zhou, and Hao Wang. Outdoor monocular slam with
global scale-consistent 3d gaussian pointmaps. _arXiv preprint arXiv:2507.03737_, 2025.


Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias
Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In _Proceedings of the_
_IEEE conference on computer vision and pattern recognition_, pp. 5828–5839, 2017a.


Angela Dai, Matthias Nießner, Michael Zollhöfer, Shahram Izadi, and Christian Theobalt. Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration.
_ACM Transactions on Graphics (ToG)_, 36(4):1, 2017b.


Frank Dellaert and GTSAM Contributors. borglab/gtsam, May 2022. [URL https://github.](https://github.com/borglab/gtsam))
[com/borglab/gtsam).](https://github.com/borglab/gtsam))


Dapeng Feng, Zhiqiang Chen, Yizhen Yin, Shipeng Zhong, Yuhua Qi, and Hongbo Chen. Cartgs:
Computational alignment for real-time gaussian splatting slam, 2024. [URL https://arxiv.](https://arxiv.org/abs/2410.00486)
[org/abs/2410.00486.](https://arxiv.org/abs/2410.00486)


Bin Fu, Jialin Li, Bin Zhang, Ruiping Wang, and Xilin Chen. Gs-lts: 3d gaussian splatting-based
adaptive modeling for long-term service robots. _arXiv preprint arXiv:2503.17733_, 2025.


Andreas Geiger, Philip Lenz, and Raquel Urtasun. Are we ready for autonomous driving? the kitti
vision benchmark suite. In _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_ _(CVPR)_,
2012.


Christian Homeyer, Leon Begiristain, and Christoph Schnörr. Droid-splat combining end-to-end
slam with 3d gaussian splatting. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _International_ _Conference_ _on_
_Computer Vision (ICCV) Workshops_, pp. 2767–2777, October 2025.


Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting
for geometrically accurate radiance fields. In _ACM_ _SIGGRAPH_ _2024_ _conference_ _papers_, pp.
1–11, 2024.


Huajian Huang, Longwei Li, Hui Cheng, and Sai-Kit Yeung. Photo-slam: Real-time simultaneous
localization and photorealistic mapping for monocular, stereo, and rgb-d cameras-supplementary
material.


Sergio Izquierdo and Javier Civera. Optimal transport aggregation for visual place recognition. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_,
2024.


Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM Trans. Graph._, 42(4):139–1, 2023.


Mingrui Li, Yiming Zhou, Hongxing Zhou, Xinggang Hu, Florian Roemer, Hongyu Wang, and
Ahmad Osman. Dy3dgs-slam: Monocular 3d gaussian splatting slam for dynamic environments.
_arXiv preprint arXiv:2506.05965_, 2025.


11


Dominic Maggio, Hyungtae Lim, and Luca Carlone. Vggt-slam: Dense rgb slam optimized on the
sl (4) manifold. _arXiv preprint arXiv:2505.12549_, 2025.


Hidenobu Matsuki, Riku Murai, Paul HJ Kelly, and Andrew J Davison. Gaussian splatting slam.
In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_, pp.
18039–18048, 2024.


Raul Mur-Artal, Jose Maria Martinez Montiel, and Juan D Tardos. Orb-slam: A versatile and
accurate monocular slam system. _IEEE transactions on robotics_, 31(5):1147–1163, 2015.


Riku Murai, Eric Dexheimer, and Andrew J Davison. Mast3r-slam: Real-time dense slam with 3d
reconstruction priors. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_ _Confer-_
_ence_, pp. 16695–16705, 2025.


Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, and
Luc Van Gool. Unidepthv2: Universal monocular metric depth estimation made simpler. _arXiv_
_preprint arXiv:2502.20110_, 2025.


René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In
_Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, pp. 12179–
12188, October 2021.


Johannes Lutz Schönberger and Jan-Michael Frahm. Structure-from-motion revisited. In _Confer-_
_ence on Computer Vision and Pattern Recognition (CVPR)_, 2016.


Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot
gaussian splatting from uncalibrated image pairs. _arXiv preprint arXiv:2408.13912_, 2024.


Zachary Teed and Jia Deng. Droid-slam: Deep visual slam for monocular, stereo, and rgb-d cameras.
_Advances in neural information processing systems_, 34:16558–16569, 2021.


Yongchun Fang Tianci Wen, Zhiang Liu. Segs-slam: Structure-enhanced 3d gaussian splatting slam
with appearance embedding. In _Proceedings of the IEEE/CVF Conference on Computer Vision_,
2025.


Konstantinos A Tsintotas, Loukas Bampis, and Antonios Gasteratos. The revisiting problem in
simultaneous localization and mapping: A survey on visual loop closure detection. _IEEE Trans-_
_actions on Intelligent Transportation Systems_, 23(11):19929–19953, 2022.


Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David
Novotny. Vggt: Visual geometry grounded transformer. In _Proceedings of the Computer Vision_
_and Pattern Recognition Conference_, pp. 5294–5306, 2025a.


Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_
_Pattern Recognition Conference_, pp. 10510–10522, 2025b.


Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_, pp. 20697–20709, 2024.


Ke Wu, Zicheng Zhang, Muer Tie, Ziqing Ai, Zhongxue Gan, and Wenchao Ding. Vingsmono: Visual-inertial gaussian splatting monocular slam in large scenes. _arXiv_ _preprint_
_arXiv:2501.08286_, 2025a.


Yuqi Wu, Wenzhao Zheng, Jie Zhou, and Jiwen Lu. Point3r: Streaming 3d reconstruction with
explicit spatial pointer memory, 2025b. [URL https://arxiv.org/abs/2507.02863.](https://arxiv.org/abs/2507.02863)


Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai,
Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one
forward pass. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_ _Conference_, pp.
21924–21935, 2025.


12


Shangzhan Zhang, Jianyuan Wang, Yinghao Xu, Nan Xue, Christian Rupprecht, Xiaowei Zhou,
Yujun Shen, and Gordon Wetzstein. Flare: Feed-forward geometry, appearance and camera estimation from uncalibrated sparse views. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition Conference_, pp. 21936–21947, 2025.


Linqing Zhao, Xiuwei Xu, Yirui Wang, Hao Wang, Wenzhao Zheng, Yansong Tang, Haibin Yan,
and Jiwen Lu. Pseudo depth meets gaussian: A feed-forward rgb slam baseline. _arXiv preprint_
_arXiv:2508.04597_, 2025.


Min Zhao, Xin Guo, Le Song, Baoxing Qin, Xuesong Shi, Gim Hee Lee, and Guanghui Sun. A general framework for lifelong localization and mapping in changing environment. In _2021 IEEE/RSJ_
_International Conference on Intelligent Robots and Systems (IROS)_, pp. 3305–3312. IEEE, 2021.


Jianhao Zheng, Zihan Zhu, Valentin Bieri, Marc Pollefeys, Songyou Peng, and Iro Armeni. Wildgsslam: Monocular gaussian splatting slam in dynamic environments. In _Proceedings of the Com-_
_puter Vision and Pattern Recognition Conference_, pp. 11461–11471, 2025.


13


A LLM USAGE STATEMENT


In the preparation of this manuscript, we utilized Large Language Models (LLMs) as assistive tools,
in accordance with the ICLR policy. The specific roles of these models are detailed below.


We employed **GPT-5** primarily for **language polishing and refinement** . After drafting the paper,
we used the model to improve grammar, clarity, and overall readability. The core ideas, experimental
setup, results, and conclusions were conceived and articulated entirely by the human authors. We
reviewed and edited all model-generated suggestions to ensure the final text accurately reflects our
original research and contributions.


We also used **Gemini 2.5 Pro** with its deep research capabilities to assist in the **literature review**
**process** . This tool helped in identifying relevant prior work, summarizing existing literature, and
locating publicly available publications. All cited works were subsequently read, analyzed, and
contextualized by the authors to build the foundation for our research.


The authors take full responsibility for all content presented in this paper, including its scientific validity, accuracy, and originality. LLMs were used strictly as productivity tools and are not considered
authors of this work.


B MORE EXPERIMENTAL SETUP AND RESULTS


B.1 BASELINE


For MASt3R-SLAM (Murai et al., 2025), we adopted the official experimental configuration with
_ωk_ = 0 _._ 333 _, ωl_ = 0 _._ 1 _, ωr_ = 0 _._ 005, and a maximum of 10 matching iterations. For MonoGS (Matsuki et al., 2024), we followed their TUM settings, as TUM shares the closest characteristics with
the BundleFusion (Dai et al., 2017b) and ScanNetV1 (Dai et al., 2017a) datasets. During evaluation, MonoGS encountered out-of-memory (OOM) failures on three sequences: ScanNet 0054 and
0465, and the apt0 sequence in BundleFusion. For these cases, the reported metrics are computed
only on the subsequences successfully reconstructed before the crash. This truncation may lead to
an optimistic bias, as drift accumulation over the full sequence would likely degrade reconstruction
and rendering quality further. For S3PO-GS (Cheng et al., 2025), we used their official base configuration while loading the ground-truth intrinsics for the test datasets. For the ScanNetV1 sequence
0465, accumulated pose drift caused _PnP_ failure, and results are reported only on the valid subsequence. For DepthGS (Zhao et al., 2025), we followed the official repository guidelines, generating
monocular depth maps for each sequence using the _UniDepthV2-large_ checkpoint and benchmarking under the provided experimental settings. Importantly, the reported FPS includes the runtime
required for UniDepthV2 (Piccinelli et al., 2025), ensuring a fair comparison across methods.


B.2 COMPARISON DETAILS ON DEPTH RENDERING QUALITY


As shown in Table 6, we record the depth rendering results in detail. To avoid bias caused by large
errors in failure scenarios, we report in the main text the mean values excluding the maximum and
minimum.


Table 6: Depth L1 Error (m) on **ScanNetV1** and **BundleFusion** datasets. Lower is better. We mark
the best results.


**ScanNetV1** **BundleFusion**
**Depth L1 [m]** _↓_

**0054** **0059** **0106** **0169** **0233** **0465** **apt0** **apt2** **copyroom** **office0** **office2**


**MonoGS** 1 _._ 06 1 _._ 27 1 _._ 41 1 _._ 56 0 _._ 89 0 _._ 82 0 _._ 96 1 _._ 28 1 _._ 18 1 _._ 15 1 _._ 26
**DepthGS** 0 _._ 45 0 _._ 66 0 _._ 52 0 _._ 48 0 _._ 41 0 _._ 47 0 _._ 37 **0.29** 0 _._ 13 0 _._ 18 0 _._ 21
**S3PO-GS** 0 _._ 58 0 _._ 35 0 _._ 55 0 _._ 66 0 _._ 28 0 _._ 89 0 _._ 72 0 _._ 99 0 _._ 41 1 _._ 01 0 _._ 85
**Ours** **0.16** **0.23** **0.51** **0.17** **0.35** **0.44** **0.33** 0 _._ 35 **0.11** **0.11** **0.18**


14


B.3 MORE QUALITATIVE RESULTS


Figure 6 provides a qualitative comparison of camera trajectories from different methods. We plot
the estimated trajectory (colored line) against the ground truth (dashed gray line), projected onto the
XY plane. The color of the path indicates the magnitude of the error, following a gradient from blue
(low error) to red (high error).


Figure 7 provides a qualitative comparison of the reconstructed map on ScanNet scene 0054. Scene
0054 is a multi-room apartment with varying lighting conditions. All baselines failed to reconstruct
the scene.


Figure 6: **Qualitative Analysis on Estimated Trajectory**


15


Figure 7: **Qualitative** **Analysis** **on** **reconstructed** **ScanNet** **scene** **0054.** All baselines failed to
reconstruct the scene.


C MODEL SIZE AND ACCELERATION


C.1 MODEL SIZE


To address concerns regarding the feasibility of deployment on resource-constrained devices (e.g.,
edge devices or laptops), we provide a detailed breakdown of our model size and a performance
analysis on lower-end hardware.


**Model** **Size** **and** **Memory** **Usage.** The detailed parameter breakdown of our architecture is presented in Table 7. The complete model consists of 795.7 million parameters. In terms of memory
consumption, the system requires approximately **3GB of VRAM** to run during inference.


Table 7: Detailed breakdown of Flash-Mono model parameters.


**Component** **Total Parameters**


Encoder 303.1 M
Decoder 380.8 M
Heads & Tokens 111.8 M


**Total** **795.7 M**


C.2 MODEL ACCELERATION


The feed-forward frontend is the primary computational bottleneck in the Flash-Mono system. To
enhance its practicality for SLAM applications on more accessible, resource-constrained hardware,
we tested the effectiveness of several acceleration methods.


First, we converted the attention module parameters from float32 to float16 precision. This strategy
compresses the model size and accelerates inference without degrading downstream task accuracy.
Second, we addressed an inefficiency in the single-image inference pipeline. With a batch size of 1,


16


frequent CPU-side operator launches create a bottleneck that underutilizes the GPU. By employing
CUDA Graphs, we merged multiple operator calls into a single, efficient launch. See Figure 8.


To validate these improvements under a resource-constrained SLAM setting, we benchmarked the
system on a laptop version NVIDIA RTX 4060 GPU (8GB). These optimizations reduced frontend
inference latency from 283 ms to just 85 ms, a **3.33** _×_ **speedup** . Notably, the inference time on the
laptop RTX 4060 after acceleration (83 ms) is comparable to the inference time on the high-end
RTX 4090 (24GB) used in our main experimental setup (62 ms). In addition, as our model is based
on the transformer architecture, further optimizations, such as quantization and efficient attention
mechanisms, remain promising directions for future inference acceleration.


Figure 8: CUDA Graph optimization


D TRAINING SETUP


D.1 DATASETS


We train our model on a combination of indoor and outdoor datasets, including ScanNet++, DL3DV,
and Replica. For each training sequence, we utilize the provided RGB video stream, ground truth
camera poses, and depth maps. The ground truth point cloud( _µt_ ) required for supervising the geometry loss is generated by unprojecting the ground truth depth map _Dt_ using the corresponding
camera intrinsics _K_ .


D.2 EXTRA RENDERING LOSS


While the pose and geometry loss terms adhere to the standard formulations outlined in the main
paper, the rendering loss incorporates a more sophisticated strategy. Our empirical investigation
revealed that a naive rendering loss, computed solely on the merged Gaussian point cloud from an
entire sequence, encourages the model to excessively shrink the scale of individual Gaussians to
prevent inter-frame conflicts. In our incremental SLAM setting, this will lead to a bad rendering
result on the first few frames of each submap. Thus, our rendering loss consists of two parts with
the same weights. The first is a per-frame rendering loss, where the predicted 2DGS map for each
frame, _Gt_, is rendered independently and compared against its corresponding ground truth image
and depth. For the second rendering loss, the 2DGS predictions from the entire input sequence,
_{Gt}_ _[N]_ _t_ =1 [, are merged into a global 2DGS map and then rendered and supervised against the ground]
truth RGB and depth.


17


D.3 TRAINING CURRICULUM


We designed a three-stage training curriculum. We first warm up the GS head for 5,000 steps.
In this initial phase, we initialize our model parameters from CUT3R (Wang et al., 2025b) and
freeze all network parameters except for the final 2DGS attribute prediction head and employ a
relatively high learning rate of 2 _×_ 10 _[−]_ [4] on short input sequences of 1-4 frames. The freezing
prevents the gradients from excessive rendering loss from backpropagating into the well-trained
model backbone. Following this, we unfreeze the parameters of the decoder, the pose head, and the
point-mean prediction head, while significantly reducing the learning rate to 1 _×_ 10 _[−]_ [5] and fixing
the sequence length at 4 frames. This intermediate stage is intended to allow the model to adapt
to the specific data distribution of the task and mitigate the risk of gradient explosion common in
recurrent architectures. Finally, we adapt the model to a longer sequence. The learning rate is further
decreased to 5 _×_ 10 _[−]_ [6], and the maximum sequence length is extended to 32 frames.


E DETAILED RUNTIME BREAKDOWN


To substantiate the claimed efficiency and clarify the FPS calculation protocol mentioned in the
main paper, we provide a comprehensive runtime breakdown of Flash-Mono.


**FPS Calculation Protocol.** The reported FPS is an end-to-end metric, calculated as Total Runtime [Total Frames] [. This]

metric explicitly accounts for **all** system components, including frontend inference, backend map
refinement, loop closure detection, pose graph optimization (PGO), and other system overheads.


**Runtime Analysis.** Flash-Mono operates using two parallel threads: a **Frontend** thread responsible
for tracking and loop closure, and a **Backend** thread handling mapping and refinement. The detailed
time consumption for each module is presented in Table 8.


As shown in the table, the Backend thread (77.5 ms per frame) is slightly slower than the Frontend thread (65 ms per frame), making it the bottleneck of the system. It is important to note that
computationally intensive loop closure operations (such as Loop Frame Feedforward and PGO) are
sparse events. Consequently, their amortized cost is minimal, allowing the system to maintain high
real-time performance.


Table 8: Runtime breakdown of Flash-Mono. The system runs in parallel threads, with the Backend
being the primary bottleneck. Loop closure operations are sparse events.


**Thread** **Module** **Time (ms)** **Note**


Feedforward Inference 62 Per frame
Loop Closure Detection 3 Per frame


**Frontend**


**Backend**


**Total (Per Frame)** **65**


Loop Frame Feedforward 62 Per loop closure
PGO (Sim3 Optimization) 32 Per loop closure


Merge & Voxelization 0.5 Per frame
Refine 77 Per frame


**Total (Per Frame)** **77.5**


GS Correction 2 Per loop closure


F ANALYSIS OF GAUSSIAN MAP COMPACTNESS


To evaluate the spatial efficiency of our map representation, we conducted a quantitative analysis of
the total number of Gaussian primitives required to represent a complete scene. This metric provides
insight into the trade-off between reconstruction quality and memory usage.


Table 9 presents a comparison of the total Gaussian count against several state-of-the-art dense
SLAM and Gaussian Splatting methods on three sequences from the TUM RGB-D dataset. As


18


illustrated in Table 9, Flash-Mono maintains a moderate level of map compactness. The baseline
statistics are sourced from CaRtGS (Feng et al., 2024).


Table 9: Quantitative comparison of the total Gaussian count on the TUM dataset. Our method
maintains a balance between map density and compactness.


**Method** **fr1/desk** **fr2/xyz** **fr3/office**


MonoGS 26.64k 43.59k 35.24k
Photo-SLAM 40.00k 0.10m 81.16k
SplaTAM 0.96m 6.36m 0.79m
Gaussian-SLAM 0.76m 0.69m 1.47m
GS-ICP-SLAM 0.53m 1.91m 2.09m


**Ours** **0.63m** **0.98m** **0.61m**


G ANALYSIS OF HIDDEN STATE FOR LIFE-LONG MAPPING


While the hidden state mechanism has been demonstrated to be highly effective for loop closure
relocalization and in-session long-term consistency (as shown in our ablation study in the main
paper), we believe this mechanism can be naturally extended to address the challenges of **life-long**
**mapping** —the ability to construct and maintain an up-to-date map as the environment changes over
time (e.g., furniture rearrangement, lighting variations, seasonal changes).


**Core Challenges.** Life-long mapping presents two fundamental challenges that must be addressed
to maintain a consistent and accurate representation of a dynamic environment:


_Challenge 1:_ _Relocalizing Against an Outdated Map._ The system must be capable of relocalizing a
new observation of the changed environment against an old, potentially outdated map. Our hidden
state mechanism can address this challenge using the same feed-forward relocalization approach
described in Section 4.3 of the main paper.


To demonstrate this capability, we conducted a case study on a scene that underwent significant environmental changes. We first input 8 frames captured at night (with curtains closed and a seat back
in place) to generate a hidden state _M_ night representing the historical environment. Subsequently, we
fed the model with a new observation of the same scene captured during daytime, where the curtains
were open and a person was sitting in the chair. As illustrated in Figure 9, our feed-forward model
successfully relocalized the new frame against the outdated hidden state and predicted geometrically
consistent results, despite the substantial appearance and content changes.


This result suggests that our current architecture already possesses inherent robustness to environmental variations. We anticipate that training specifically on datasets featuring temporal changes
(e.g., time-of-day variations, seasonal shifts) would further enhance this capability.


_Challenge_ _2:_ _Updating_ _the_ _Map_ _Representation._ Beyond relocalization, the system must update
its map representation with new observations to remain synchronized with the current environment
state. Our system maintains three core representations: the Gaussian Map, the Pose Graph, and
the **Hidden** **State** (which serves as a compact submap descriptor). While strategies for updating
Gaussian primitives (Fu et al., 2025) and pose graphs (Zhao et al., 2021) to changing environment
are well-established in the SLAM literature, updating the hidden state descriptor in a life-long setting
presents unique challenges.


A naive approach of continuously aggregating new observations into a fixed-capacity hidden state
vector inevitably leads to saturation and catastrophic forgetting of historical information. To address
this challenge for life-long mapping, we propose two potential strategies:


**1.** **Discrete** **State** **Replacement.** A straightforward yet effective strategy is to detect significant
environmental changes during relocalization (e.g., via monitoring the photometric residual or geometric consistency). When substantial changes are identified, rather than attempting to update the
obsolete hidden state _M_ old, we generate a _fresh_ hidden state _M_ new from the current observations.
This new state can either replace the outdated state in the Bag of Hidden States or be appended as


19


an alternative descriptor for the same physical location, effectively maintaining multiple hypotheses
(e.g., "daytime" vs. "nighttime" appearance).


**2.** **Model Adaptation** Instead of discrete replacement, we can implement a continuous update strategy inspired by TTT3R (Chen et al., 2025). This approach reframes the hidden state update as an
online learning problem, treating the state as "fast weights" optimized via gradient descent during
inference. Crucially, TTT3R introduces a confidence-guided update rule, where the learning rate
is dynamically derived from the alignment between the current memory state and the incoming observation. This acts as a self-supervised gating mechanism: it allows the hidden state to selectively
integrate persistent environmental changes (where alignment confidence is high) while suppressing
transient noise or inconsistent updates (where confidence is low). By adopting this formulation,
our system can evolve to capture gradual domain shifts—such as seasonal changes or furniture rearrangement—while mitigating the catastrophic forgetting typically associated with recurrent updates,
ensuring the map remains both up-to-date and geometrically consistent over the long term.


**Future** **Directions.** While our current system demonstrates promising initial capabilities for handling environmental changes, a comprehensive life-long mapping system would require dedicated
training on temporally-varying datasets and careful engineering of the state update mechanisms. We
believe this represents an exciting direction for future work, building upon the flexible hidden state
architecture introduced in this paper.


Figure 9: **Case Study:** **Robust Relocalization Under Environmental Changes.** The model generates a hidden state from 8 context views captured at night (curtains closed, empty chair). When
presented with a new observation from the same location but under drastically different conditions
(daytime, curtains open, person sitting), the feed-forward model successfully relocalizes and reconstructs accurate geometry. This demonstrates the hidden state’s potential for life-long mapping
scenarios where environments undergo temporal changes.


20
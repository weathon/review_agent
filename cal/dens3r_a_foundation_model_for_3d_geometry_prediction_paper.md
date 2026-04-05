# DENS3R: A FOUNDATION MODEL FOR 3D GEOMETRY PREDICTION


**Xianze Fang** [1] _[,][∗]_ **Jingnan Gao** [2] _[,][∗]_ **Zhe Wang** [1] **Zhuo Chen** [2] **Xingyu Ren** [2] **Jiangjing Lyu** [1] _[,][†]_

**Qiaomu Ren** [1] **Zhonglei Yang** [1] **Xiaokang Yang** [2] **Yichao Yan** [2] _[,][‡]_ **Chengfei Lv** [1]

1Alibaba Group. 2Shanghai Jiao Tong University.


**[https://g-1nonly.github.io/Dens3R/](https://g-1nonly.github.io/Dens3R/)**


Figure 1: Dens3R is a feed-forward visual foundation model that takes unposed images as input and
outputs high-quality 3D pointmap with unified geometric dense prediction. Dens3R also accepts
generalized inputs, supporting both multi-view and multi-resolution inputs. As a versatile backbone,
Dens3R achieves robust dense prediction under several scenarios and can be easily extended to
downstream applications.


ABSTRACT


Recent advances in dense 3D reconstruction have led to significant progress, yet
achieving accurate unified geometric prediction remains a major challenge. Most
existing methods are limited to predicting a single geometry quantity from input
images. However, geometric quantities such as depth, surface normals, and point
maps are inherently correlated, and estimating them in isolation often fails to ensure consistency, thereby limiting both accuracy and practical applicability. This
motivates us to explore a unified framework that explicitly models the structural
coupling among different geometric properties to enable joint regression. In this
paper, we present Dens3R, a 3D foundation model designed for joint geometric
dense prediction and adaptable to a wide range of downstream tasks. Dens3R
adopts a two-stage training framework to progressively build a pointmap representation that is both generalizable and intrinsically invariant. Specifically, we
design a lightweight shared encoder-decoder backbone and introduce positioninterpolated rotary positional encoding to maintain expressive power while enhancing robustness to high-resolution inputs. By integrating image-pair matching
features with intrinsic invariance modeling, Dens3R accurately regresses multiple geometric quantities such as surface normals and depth, achieving consistent
geometry perception from single-view to multi-view inputs. Additionally, we
propose a post-processing pipeline that supports geometrically consistent multiview inference. Extensive experiments demonstrate the superior performance of
Dens3R across various tasks and highlight its potential for broader applications.


_∗_ : Equal Contribution. _†_ : Project Leader. _‡_ : Corresponding Author.


1


1 INTRODUCTION


Recovering 3D geometric structures from static images is a long-standing and fundamental problem in computer vision. Classical approaches, such as Structure-from-Motion (SfM) and MultiView Stereo (MVS), demonstrate strong performance in controlled settings and have been widely
adopted in a broad range of 3D reconstruction applications. However, in unconstrained scenarios—where camera intrinsics, extrinsics, or viewpoint information are unavailable—achieving accurate and dense geometric prediction remains highly challenging. These conditions demand more
generalizable and robust solutions capable of handling diverse and unstructured visual inputs.


Existing methods for dense geometric prediction primarily fall into two categories. The first category
mostly adopts generative models, utilizing strong image priors from pre-trained diffusion models or
large-scale training datasets for dense prediction. For example, GenPercept Xu et al. (2025) is used
for depth prediction, and StableNormal Ye et al. (2024) for normal estimation. This raises a key
issue: while image generation tasks typically benefit from their inherent ambiguity and multi-modal
output characteristics, geometric prediction is fundamentally different. Geometric prediction is essentially a deterministic task that needs to closely reflect the structural information of the underlying
scene. Moreover, the pixel continuity and spatial smoothness required by geometric representations
are difficult to naturally obtain through standard diffusion sampling mechanisms without structural
constraints. Therefore, the direct application of diffusion models in geometric regression tasks faces
significant challenges, especially in such tasks where a strict one-to-one correspondence between
input and output needs to be maintained. Based on this, we adopt a regression-oriented framework
to construct geometric mapping models in a more efficient and interpretable way. Furthermore, the
aforementioned methods mainly handle only one geometric quantity prediction and cannot generalize to output multiple geometric quantities in a single forward pass. The second category includes
DUSt3R Wang et al. (2024) and its follow-up works Leroy et al. (2024); Wang et al. (2025b;a); LAN
et al. (2026). These methods use regression models that can regress 3D point map representations
with geometric properties, applied to dense prediction, including image pair matching and depth
estimation. However, these methods typically focus on a single prediction task, and other geometric
quantities suffer severe performance degradation due to representation influences.


This raises a natural question: can we build a unified model that simultaneously regresses multiple
geometric quantities with high quality? We observe that existing methods like DUSt3R, when handling dense geometric regression tasks, overlook a crucial geometric information—surface normals.
Traditionally, normals have been used to add high-frequency details to rough geometric structures
to enhance rendering quality. However, our research finds that introducing normal information during geometric prediction can significantly improve the accuracy of point maps, resulting in more
detailed and structurally consistent 3D representations. This is mainly because: 1) From the perspective of normal prediction, the inherent image pair matching capability in dense vision backbone
networks helps alleviate monocular ambiguity and improve the stability and accuracy of normal
prediction; 2) From the feature modeling perspective, normals possess good intrinsic invariance,
which simplifies the mapping learning process and aids in model convergence and generalization.
This modeling approach enables the model to simultaneously predict multiple geometric quantities
(such as depth, normals, and point maps) from a single view, effectively reducing dependence on
multi-view supervision and simplifying the training process. However, training such a multi-task,
multi-output 3D foundation model still faces significant challenges. Geometric quantities are tightly
coupled, and how to coordinate these relationships to achieve optimal overall performance requires
carefully designed training strategies and architectural support.


In this paper, we present **Dens3R**, a foundation model for high-quality geometric prediction. To
this end, we design a two-stage training framework that gradually builds a versatile pointmap representation, which generalizes well to various downstream tasks. Specifically, we first construct a
dense vision backbone network with multi-task prediction capabilities. This network adopts a shared
encoder-decoder architecture, which significantly reduces model parameters while maintaining expressive power. To accommodate high-resolution inputs, we introduce position-interpolated rotary
positional encoding, which effectively mitigates prediction degradation caused by increased input
resolution. For the training strategy, we propose a novel two-staged approach. In the first stage, the
model leverages image pair matching features to learn scale-invariant point maps, capturing consistent spatial geometric structures across viewpoints. Subsequently, to fully exploit the one-to-one
mapping property in normal estimation, we extend the pointmap representation into an intrinsically


2


invariant form. This allows the model to independently attend to each viewpoint, thereby improving
the accuracy of normal prediction. The learned geometric structures also assist in estimating other
geometric quantities, such as depth, thereby simplifying their training processes. Finally, we design
a simple and efficient post-processing pipeline that supports multi-view inputs during inference,
which enhances the geometric consistency of the model in real-world applications. In summary, we
make the following contributions:


    - We introduce **Dens3R**, a dense 3D visual foundation model that demonstrates **high-quality**
**performance** in various 3D tasks including pointmap reconstruction, depth estimation,
normal prediction and image matching under several benchmark evaluations.


    - We design a novel training strategy with the **intrinsic-invariant** **pointmap** and shared
Encoder-Decoder visual backbone to incorporate surface normals in unconstrained imagebased dense 3D reconstruction, simplifying the training complexity of other 3D quantities
and achieving better results without requiring excessive computation resources.


    - We employ a **position-interpolated** **rotary** **positional** **encoding** to preserve prediction
accuracy at higher resolutions and support multi-resolution inputs.


    - Extensive experiments on various benchmarks showcase our high-quality predictions of 3D
geometric quantities, which further enable a wide range of applications.


2 RELATED WORKS


2.1 MONOCULAR DEPTH AND NORMAL PREDICTION


Monocular depth prediction has been extensively investigated and demonstrates strong capability in
providing geometric priors for a multitude of downstream tasks like image understanding and 3D
reconstruction. The earliest pioneering researchers Bhat et al. (2021; 2023); Eigen et al. (2014);
Yin et al. (2023); Hu et al. (2024); Piccinelli et al. (2024) addressed this issue by estimating depth
with a metric scale. These methods usually rely heavily on data from specific sensors, which restricts the applicability and deteriorates the performance when confronted with complex scenes.
Subsequently, deep learning approaches involve predicting relative depth either through direct regression Chen et al. (2016; 2020); Godard et al. (2019); Li & Snavely (2018a); Ranftl et al. (2022);
Yang et al. (2024a;b) or via generative modeling based on diffusion priors Fu et al. (2024); Gui et al.
(2024); Ke et al. (2024); Wan et al. (2023). While monocular depth estimation has made significant strides, accurate 3D shape reconstruction from depth maps remains fundamentally dependent
on precise camera intrinsic parameters. Meanwhile, normal maps serve as a supervision for neural
scene representation, bridging 2D and 3D worlds. The accurate estimation of the normal map can
open up broader applications like material decomposition and relighting. On one hand, regressionbased methods Eftekhar et al. (2021); Bansal et al. (2016); Wang et al. (2015); Ranftl et al. (2021)
utilize large-scale training datasets for robust estimation. DSINE Bae & Davison (2024) proposes to
leverage the per-pixel ray direction and try to model the inductive biases for surface normal estimation correctly. On the other hand, diffusion-based methods Long et al. (2024); Fu et al. (2024); Ye
et al. (2024) adapt the pretrained diffusion model as a geometric cues predictor. Geowizard includes
a geometry switcher to disentangle mixed-sourced data into distinct sub-distributions for normal
prediction. StableNormal repurposes the diffusion model for deterministic estimation tasks and can
estimate sharp normals steadily. Nevertheless, these normal estimation methods often suffer from
monocular ambiguity, leading to inaccurate and inconsistent results for complex scenes. In contrast,
our method allows the communication between 3D geometric representation and normal prediction
without known camera poses. This not only resolves the ambiguity but also achieves accurate 3D
reconstruction with accurate normals.


2.2 IMAGE PAIR MATCHING IN 3D


Dense matching Edstedt et al. (2023; 2024); Efe et al. (2021); Melekhov et al. (2019); Truong et al.
(2020; 2021; 2023); Zhu & Liu (2023); Sarlin et al. (2020); Sun et al. (2021) has been proved to
be effective in many scenarios and results in top performance in many benchmarks. However, these
approaches cast matching as a 2D problem, which restricts the application for visual localization.
Thus anchoring image correspondence in 3D space is essential when these 2D-based methods fall


3


Figure 2: Overview of Dens3R. We propose Dens3R, a dense visual transformer backbone featuring
a shared encoder-decoder architecture and multiple task-specific heads for geometric prediction. To
train this foundation model, we adopt a two-stage strategy. In Stage 1, we learn a scale-invariant
pointmap by enforcing cross-view mapping consistency across multiple viewpoints. In Stage 2,
we incorporate surface normals and leverage one-to-one correspondence constraints to transform
the representation into an intrinsic-invariant pointmap. Built upon this unified backbone, additional
geometric prediction heads and downstream task branches can be seamlessly integrated to support a
wide range of applications.


short. Early methods Bhalgat et al. (2023); He et al. (2020); Wang et al. (2020a); Yao et al. (2019);
Yifan et al. (2022); Zhou et al. (2021); Toft et al. (2020) leverage epipolar constraints in order to
improve accuracy or robustness. Recently, researchers Zhang et al. (2024); Wang et al. (2023a)
leverage diffusion models for pose estimation and demonstrate promising results by incorporating
3D geometric constraints into estimation formulation. MASt3R Leroy et al. (2024) retrieves correspondences via 3D reconstruction from uncalibrated images by explicitly training local features
for pairwise matching. However, MASt3R only grounds image-pair matching and overlooks other
geometric predictions like depth and normal, while Dens3R achieves unified geometric predictions
and better matching.


2.3 DENSE UNCONSTRAINED GEOMETRIC REPRESENTATIONS


Neural scene reconstructions Mildenhall et al. (2020); Wang et al. (2021a); Kerbl et al. (2023); Barron et al. (2021); Martin-Brualla et al. (2021); Barron et al. (2023); Yariv et al. (2021); Lu et al.
(2024); Yu et al. (2024); Wang et al. (2023b) usually require the camera intrinsic parameters and
poses for optimization. The reconstruction quality of these methods is highly dependent on the accuracy of the camera intrinsics and poses. Later methods Smart et al. (2024); Ye et al. (2025); Hong
et al. (2024) propose to optimize the scene without known camera poses, but these methods usually
take longer time and sacrifice reconstruction quality. To bypass estimation of camera parameters and
poses, DUSt3R Wang et al. (2024) proposes to directly map two input images in a single forward
pass, leading to a more straightforward geometry representation. Subsequently, Spann3R Wang &
Agapito (2025) and Fast3R Yang et al. (2025) augment DUSt3R to process an ordered set of images.
MoGe Wang et al. (2025b) further proposes affine-invariant pointmaps for monocular geometry estimation. VGGT Wang et al. (2025a) utilizes 3D pointmaps and multiple prediction heads to predict
geometric quantities from multi-view images input. However, the aforementioned methods overlook
the normal attribute and fall short in prediction for complex scenarios. In contrast, our model takes
advantage of pointmap representation and employs several prediction heads including the normal
head to achieve unified geometric predictions.


3 METHOD


This work aims to utilize a single model to predict various geometric data from unconstrained images, including 3D pointmaps, depth maps, normal maps, and image-pair matching. To this end, we
built a backbone network based on dense visual transformers and designed input configurations that
adapt to multi-resolution and multi-view requirements (Sec. 3.1). Since achieving accurate results
through direct training with a single model is challenging, we adopted a two-stage training approach.
In the first stage, we train the backbone and heads to obtain scale-invariant pointmaps. In the second


4


stage, we fine-tune the backbone on this foundation to obtain intrinsic-invariant pointmaps (Sec. 3.2).
Finally, we further fine-tune the prediction heads for each downstream task to adapt to different application scenarios. Meanwhile, extending the model inputs to multi-view images in the inference
stage significantly improves the overall inference quality. (Sec. 3.3).


3.1 MODEL FORMULATION


**Shared** **Backbone.** Motivated by recent advances in 3D vision Wang et al. (2024); Leroy et al.
(2024); Wang et al. (2025a); Jin et al. (2025), we aim to build a foundation model capable of predicting diverse geometric quantities across different scenes and tasks. To this end, we adopt a dense
visual transformer as the backbone, learning from rich 3D annotated data. Given an image pair of
image sequence ( _Ii_ ) [2] _i_ =1 _[∈R]_ [3] _[×][H][×][W]_ [, Dens3R’s dense visual transformer is a function] _[ f]_ [that maps]
the input to a corresponding set of 3D quantities per frame:
( _Ci, Pi, Di, Ni, Mi_ ) [2] _i_ =1 [=] _[ f]_ [((] _[I][i]_ [)] _i_ [2] =1 [)] _[,]_ (1)
where _Ci_ _∈R_ [9] is the camera parameters including both intrinsics and extrinsics, _Di_ _∈R_ _[H][×][W]_ is
the depth map, _Ni_ _∈R_ [3] _[×][H][×][W]_ is the normal map, and _Mi_ _∈R_ _[C][×][H][×][W]_ is the image-pair-matching
with _C_ -dimensional features.


The overall architecture is illustrated in the upper part of Fig. 2. Similar to prior DUSt3R-based
approaches Wang et al. (2024); Leroy et al. (2024); Wang et al. (2025a;b), we first employ a sharedweight encoder to process input image sequences and extract image features _Feai_, which are then
fed into the decoder. Unlike previous works, our approach introduces a novel weight-sharing mechanism within the decoders, allowing the backbone to better capture spatial relationships across viewpoints and to model the holistic 3D scene structure. Given the need to predict a wider range of geometric outputs, this design also significantly reduces memory and computational overhead, keeping
the training and inference efficient. Moreover, the shared-weight strategy facilitates high-resolution
input processing while effectively preventing memory overflow.


**Multi-resolution Input.** Existing methods represented by DUSt3R perform excellently at fixed resolutions (such as 512), but their prediction accuracy significantly decreases when processing higherresolution inputs. The main challenge for this issue lies in the rotary positional encoding (RoPE)
used in their ViT structure, which becomes unstable when inferring images beyond the training resolution range. Inspired by context window extension techniques in LLMs Chen et al. (2023), we
incorporate the position-interpolated RoPE into the ViT as a simple yet effective improvement. We
adapt the idea from context window to image resolution in the image domain, addressing the instability at higher resolutions. Considering the smooth properties of trigonometric functions in RoPE,
interpolation is more stable than direct extrapolation when handling high resolutions. Specifically,
let the original RoPE be _R_, the input sequence length be _L_, and for any RoPE embedding vector _x_,
we obtain a new encoding representation _R_ _[′]_ through interpolation. That is:

_R_ _[′]_ ( _x, m_ ) = _R_ ( _x,_ _[mL]_ _L_ _[′]_ [)] _[,]_ (2)

where _m_ is the position index and _L_ _[′]_ is the longer sequence. This position-interpolation encoding strategy significantly enhances the model’s robustness under high-resolution inputs, effectively
avoiding the performance degradation caused by RoPE extrapolation.


3.2 FOUNDATION MODEL TRAINING


The main challenge in training 3D geometric foundation models lies in the coupling among multiple prediction outputs, where mutual interference often leads to performance degradation. Existing
methods typically focus on only one or two geometric tasks, resulting in poor generalization to others. To this end, we propose to build upon a unified geometric representation since all geometric representations are inherently interconvertible. We adopt a two-stage training paradigm, progressively
learns a strong geometric prior, which can be efficiently transferred to a variety of 3D geometry
prediction tasks via lightweight fine-tuning.


**Scale-Invariant Pointmap Training.** In the first stage, we train the ViT backbone, pointmap head,
and matching head to obtain a scale-invariant pointmap _Pi_ . Following MASt3R’s Leroy et al.
(2024), we adopted (1) local 3D regression loss _L_ pts ~~l~~ oc, (2) Global 3D Regression Loss _L_ pts ~~g~~ lb,
(3) Pointmap Normal Loss _L_ pts n, (4) Pixel Matching Loss _L_ match. The details are as follows:


5


(1) Local 3D Regression Loss _L_ pts ~~l~~ oc . For a predicted camera, we use the local 3D regression loss
to quantify the pointmap in its own coordinate frame. We apply a mask derived from the groundtruth data to the pointmap and only evaluate the valid points when calculating the loss. We also
employ a normalization factor to handle the scale ambiguity between ground-truth and the predicted
pointmaps. We set the factor _zv_ as the average distance of all valid points in _vth_ camera coordinate
frame to the origin:

_zv_ = ��� _P_ 1 _masked,v_ ��� + ��� _P_ 2 _masked,v_ ��� _, v_ _∈{_ 1 _,_ 2 _},_

(3)
_z_ ¯ _v_ = ��� _P_ ¯ 1 _masked,v_ ��� + ��� _P_ ¯ 2 _masked,v_ ��� _, v_ _∈{_ 1 _,_ 2 _},_


where _z_ ¯ _v_ is the corresponding factor of the ground-truth. Then the local 3D regression loss can be
formulated as:


1
_L_ pts ~~l~~ oc = _Pmasked_ _[v,v]_ _[−]_ [1]
���� _zv_ _z_ ¯


_z_ ¯ _v_ _P_ ¯ _masked_ _[v,v]_


_, v_ _∈{_ 1 _,_ 2 _},_ (4)
����


where _P_ _[n,m]_ denotes the pointmap from camera _n_ expressed in the coordinate frame of camera _m_ .


(2) Global 3D Regression Loss _L_ pts glb . The global 3D regression loss is applied to quantify the
pointmap expressed in another camera’s coordinate frame. This loss function simultaneously optimizes for two objectives. It not only constrains the network to fit the pointmap shape of the image,
but also aligns the pointmap to another paired image. The global regression loss is formulated as:


1
_L_ pts glb = _Pmasked_ _[v,t]_ _[−]_ [1] _P_ ¯ _masked_ _[v,t]_
���� _zt_ _z_ ¯ _t_


_, v, t ∈{_ 1 _,_ 2 _}, v_ = _t,_ (5)
����


where _zt_ and _z_ ¯ _t_ is the normalization factor of the pointmap and the ground-truth.


(3) Pointmap Normal Loss _L_ pts n. To train an intrinsic-invariant pointmap from the scale-invariant
pointmap, we use a pointmap normal loss to encourage the pointmap learn smooth surface and sharp
edge, making the pointmap perceives the normal information and the intrinsic-invariant property.
Suppose _N_ _[v,v]_ is the ground-truth view-space normal expressed in its own camera coordinate frame
and _N_ _[v,t]_ is the ground-truth normal expressed in another camera coordinate frame, the pointmap
normal loss is the absolute error loss between the transformed normal and the ground-truth normal:


_L_ pts ~~n~~ = _L_ 1( _N_ _[v,v]_ _,_ _N_ [ˆ] _[v,v]_ ) + _L_ 1( _N_ _[v,t]_ _,_ _N_ [ˆ] _[v,t]_ ) _, v, t ∈{_ 1 _,_ 2 _}, v_ = _t,_ (6)


where the _N_ [ˆ] _[v,v]_ is the normal transformed from the local pointmap and _N_ [ˆ] _[v,t]_ is the normal transformed from the global pointmap.


(4) Pixel Matching Loss _L_ match. We utilize the pixel matching loss proposed in MASt3R Leroy et al.
(2024) to learn accurate image-matching. This loss is based on the infoNCE Oord et al. (2018) loss
and ensures that each pixel’s descriptor in the first image match at most one pixel’s descriptor in
another image. Suppose _M_ [ˆ] = ( _i, j_ ) is the set of ground-truth correspondences where the _ith_ pixel
in the first image matches the _jth_ pixel in another, the loss can then be formulated as:


 - log _sτ_ ( _i, j_ )

  
_k∈P_ [1] _[ s][τ]_ [(]

( _i,j_ ) _∈M_ [ˆ]


_L_ match = _−_ 


_sτ_ ( _i, j_ ) _sτ_ ( _i, j_ )

    _k∈P_ [1] _[ s][τ]_ [(] _[k, j]_ [)] [+ log] _k∈P_ [2] _[ s][τ]_ [(]


_k∈P_ [2] _[ s][τ]_ [(] _[i, k]_ [)] _[,]_


(7)


_sτ_ ( _i, j_ ) = exp                 - _−τDi_ [1] _[⊤]_ _Dj_ [2]                 - _,_


where _τ_ is a hyper-parameter, and _Di_ and _Dj_ are the corresponding descriptors in each image.


With the above losses, we summarize the training objective as:


_Lstage_ 1 = _L_ pts ~~l~~ oc + _η_ 1 _L_ pts ~~g~~ lb + _η_ 2 _L_ pts ~~n~~ + _η_ 3 _L_ match _,_ (8)


where the loss weights _η_ 1, _η_ 2, and _η_ 3 are set as 1 _._ 0, 0 _._ 1 and 0 _._ 075, respectively. After training,
we obtained a scale-invariant pointmap capable of capturing rich spatial information. However, as
shown in Fig. 3, the accuracy of normals obtained directly from the pointmap at this stage is still not
ideal.


**Intrinsic-Invariant** **Pointmap** **Training.** Although the point-based representation learned in the
first stage achieves good performance, it remains limited in its ability to generalize to other


6


Figure 3: Normal comparison. We demonstrate that the normal derived directly from the scaleinvariant pointmap and MoGe both are not accurate enough.

tasks—particularly surface normal estimation. Existing methods often struggle with monocular
ambiguity in normal prediction, leading to inaccurate and inconsistent results.


To this end, we expand the pointmap representation in the second stage, proposing an **intrinsic-**
**invariant** **pointmap** . This representation is inspired by the affine-invariant formulation of
MoGe Wang et al. (2025b), which disentangles shift factors from pointmaps. For a given depth map,
multiple valid solutions can exist due to shift/scale ambiguities in the 3D coordinates. In contrast,
surface normals provide an intrinsic, locally deterministic geometric property: given an underlying
surface, there is an exactly one corresponding normal map, as also discussed in works such as StableNormal Ye et al. (2024) and DSINE Bae & Davison (2024). We use this property to improve
geometric consistency by anchoring the pointmap to a more deterministic geometric interpretation,
which also improve the stability normal estimation effectively.


Specifically, we introduce high-quality normal supervision based on the first stage’s point map, and
jointly fine-tune the encoder-decoder module, point map prediction head, and newly added normal
prediction head to achieve end-to-end optimization. In terms of supervision mechanism, we adjusted
the initial ”one-to-many” mapping (one image corresponding to multiple view supervisions) to a
”one-to-one” mapping, enabling the model to independently optimize normal prediction under a
single viewpoint. This strategy not only significantly reduces the ambiguity brought by multi-view
supervision but also simplifies the training process and improves training efficiency and stability.
In addition, it enables the model to independently optimize geometric prediction under a single
viewpoint and to leverage additional high-quality monocular data during training.


We observe that the commonly-used confidence loss in previous works Wang et al. (2024); Leroy
et al. (2024); Wang et al. (2025a) tends to cause models to **ignore** **complex** **scenarios** such as reflective surfaces and low textured areas. However, naively removing the loss without additional constraints leads to degraded performance, since previous models rely heavily on confidence weighting
for point-view regression. In contrast, by utilizing the deterministic nature of normals, we obviate
the need to rely on additional views, which further enables stable and accurate prediction.


For detailed implementation, we explicitly connect normal to the pointmap representation, that is

_Pi_ _[n]_ [=] _[ P][i]_ _[⊕]_ _[n,]_ (9)

where _⊕_ represents feature concatenation operation. The normal prediction head is connected after
the initial point map training is completed, allowing the model to consistently output coherent normal mappings from the same input image, thereby internalizing this intrinsic invariance in the point
map and maintaining geometric consistency across different views.


In the second stage, we add a normal loss _L_ n for finetuning.


(5) Predicted Normal Loss _L_ n. Apart from the intrinsic-invariant pointmap, we also design a normal
head to predict the view-space normal of each frame in input image pairs. We also use the _L_ 1 loss
to supervise the normal prediction:

_L_ n = _L_ 1( _N_ _[v,v]_ _,_ _N_ [¯] _[v,v]_ ) _, v_ _∈{_ 1 _,_ 2 _},_ (10)

where _N_ is the ground-truth normal and _N_ [¯] is the direct prediction of the normal prediction head.
The complete training objective for training stage 2 is as follows:


_Lstage_ 2 = _L_ pts loc + _λ_ 1 _L_ pts ~~g~~ lb + _λ_ 2 _L_ pts ~~n~~ + _λ_ 3 _L_ n _,_ (11)


where the loss weights _λ_ 1, _λ_ 2, and _λ_ 3 are set as 1 _._ 0, 0 _._ 1 and 1 _._ 0, respectively.


To further improve the performance of Dens3R on high-resolution inputs, we introduce a coarse-tofine training strategy. Specifically, we first fine-tune the model on 512 resolution images to establish


7


Figure 4: Qualitative comparison of normal prediction. Dens3R generates more accurate and detailed normal maps than previous methods for both object-centric and unbounded scenes,. Our
method is capable of predicting accurate normals for reflective surfaces and in backgrounds.


a stable geometric prior, and then fine-tune it on 1024 resolution images to further improve the prediction accuracy. In addition, combining high-resolution data also significantly improves the fidelity
of point-based representations, ultimately enhancing the overall quality of dense 3D prediction.


3.3 MODEL INFERENCE


**Heads** **Training.** After training, we fine-tune it for downstream tasks by optimizing task-specific
prediction heads on top of the frozen backbone network. Training only these DPT heads enables
extension to various tasks such as depth estimation, normal estimation, matching estimation, and
even segmentation and object detection. It is noteworthy that the depth head is instantiated in Stage
1, similar to the depth branch in MASt3R Leroy et al. (2024), and is trained jointly within our multitask objective. At the model architectural level, however, Dens3R differs from DUSt3R Wang et al.
(2024) and MASt3R Leroy et al. (2024) by using a shared decoder rather than separate decoders for a
main and a reference view. **This design removes the need to explicitly define main and reference**
**views** **and** **alleviates** **the** **reliance** **on** **selecting** **a** **fixed** **reference** **view.** It also improves training
efficiency, since predictions are obtained from a single forward pass instead of two passes with
view swapping as in previous 3R-based methods. Building on this, after introducing the one-to-one
mapping in Stage 2, depth prediction can be optimized at the single-view level. Similarly, we can
fine-tune all the DPT heads separately with additional monocular datasets in this final heads-training
stage.


**Multi-view Inputs.** To enable Dens3R to efficiently process multi-view inputs during inference, we
design a simple yet effective post-processing step. This step ensures both computational efficiency
in multi-view data processing and the consistency and accuracy of results. Specifically, based on
Dens3R’s high-precision image pair matching predictions, we establish geometric mappings between different viewpoints by constructing and optimizing a dense correspondence network across
views. This approach effectively guides the model to understand geometric consistency between
multiple viewpoints and accurately captures spatial relationships between views. Additionally, it
significantly improves the performance and stability of multi-view processing. In practice, we first
compute matches in a one-versus-all strategy using our model, and then triangulate these matches to
obtain multi-view point clouds, following the MASt3R pipeline Leroy et al. (2024). We can also utilize the MASt3R-SfM for surface reconstruction. This pipeline inherits MASt3R’s ability to handle
large-scale scenes with hundreds of images.


8


|Method|NYUv2<br>Mean ↓ Med ↓ δ11.25◦↑|ScanNet<br>Mean ↓ Med ↓ δ11.25◦↑|IBims-1<br>Mean ↓ Med ↓ δ11.25◦↑|Sintel<br>Mean ↓ Med ↓ δ11.25◦↑|DIODE-outdoor<br>Mean ↓ Med ↓ δ11.25◦↑|
|---|---|---|---|---|---|
|DSINE<br>Lotus*<br>GeoWizard<br>StableNormal<br>Ours|18.6<br>9.9<br>56.1<br>17.5<br>8.6<br>58.7<br>20.4<br>11.9<br>47.0<br>19.7<br>10.5<br>53.0<br>16.1<br>7.4<br>62.5|18.6<br>9.9<br>56.1<br>18.1<br>8.8<br>58.2<br>21.4<br>13.9<br>37.1<br>18.1<br>10.1<br>56.0<br>16.9<br>7.1<br>64.0|18.8<br>8.3<br>64.1<br>19.2<br>5.6<br>66.2<br>19.7<br>9.7<br>58.4<br>17.2<br>8.1<br>66.7<br>16.0<br>4.3<br>72.2|34.9<br>28.1<br>21.5<br>35.7<br>28.0<br>20.5<br>41.6<br>34.3<br>11.8<br>35.0<br>27.0<br>19.5<br>30.7<br>21.4<br>28.9|22.0<br>14.5<br>39.6<br>24.7<br>15.9<br>32.9<br>27.0<br>19.8<br>24.0<br>26.9<br>16.1<br>36.1<br>20.8<br>12.8<br>43.0|


Table 1: Quantitative comparison of normal prediction. We report the mean and median angular
errors with each cell colored to indicate the best and the second . Dens3R achieves accurate
normal prediction for both indoor and outdoor scenes. *We utilize Lotus-G for a fair comparison.

|Method|Mean<br>AUC@5◦↑|Real AUC@5◦↑<br>GL3 BLE ETI ETO KIT WEA SEA NIG|Simulate AUC@5◦↑<br>MUL SCE ICL GTA|
|---|---|---|---|
|SIFT<br>SuperGlue<br>LoFTR<br>DKM<br>ROMA<br>MASt3R<br>Ours|31.8<br>34.3<br>39.1<br>51.2<br>53.2<br>59.9<br>64.5|43.5<br>33.6<br>49.9<br>48.7<br>35.2<br>21.4<br>44.1<br>14.7<br>43.2<br>34.2<br>58.7<br>61.0<br>29.0<br>28.3<br>48.4<br>18.8<br>50.6<br>43.9<br>62.6<br>61.6<br>35.9<br>26.8<br>47.5<br>17.6<br>63.3<br>53.0<br>73.9<br>76.7<br>43.4<br>34.6<br>52.5<br>24.5<br>61.8<br>53.8<br>76.7<br>82.7<br>43.2<br>36.7<br>53.2<br>26.6<br>57.8<br>52.3<br>66.2<br>78.1<br>46.2<br>52.8<br>70.5<br>43.7<br>61.3<br>59.2<br>74.7<br>81.1<br>55.6<br>57.4<br>71.7<br>50.4|33.4<br>7.6<br>14.8<br>43.9<br>34.8<br>2.8<br>15.4<br>36.5<br>41.4<br>10.2<br>25.6<br>45.0<br>56.6<br>32.2<br>42.5<br>61.6<br>60.7<br>33.8<br>45.4<br>64.3<br>70.1<br>53.9<br>60.1<br>67.7<br>71.3<br>53.7<br>66.3<br>71.7|


Table 2: Benchmark on image matching on ZEB dataset. We report the AUC values with each cell
colored to indicate the best and the second .


4 EXPERIMENTS


4.1 NORMAL AND MATCHING PREDICTION


We evaluate our Dens3R on several surface normal prediction datasets that include both indoor and
outdoor scenes. We compare our method with regression-based methods such as DSINE Bae &
Davison (2024) and diffusion-based methods like StableNormal Ye et al. (2024), GeoWizard Fu
et al. (2024) and Lotus He et al. (2025). Quantitative results are shown in Tab. 1, where Dens3R
outperforms other methods across multiple benchmarks. Qualitative comparisons are provided in
Fig. 4, also demonstrating that Dens3R generates more accurate and detailed normal maps. On the
DIODE dataset, our method produces more accurate normals for reflective surfaces ( _e.g._, car window) and finer details in backgrounds and tree structures. On in-the-wild scenes, Dens3R handles
both object-centric and unbounded scenarios, producing more stable and intricate surface normals.
Our method effectively reduces the ambiguity from monocular estimation, enabling more accurate
and detailed predictions across various settings.


For the image-matching task, we evaluate our method on the ZEB benchmark as shown in Tab. 2.
We compare our method with previous dense image-matching methods and MASt3R Leroy et al.
(2024). It can be seen that our method yields higher accuracy and surpasses previous methods across
nearly all datasets, demonstrating our superior performance across various evaluation protocols.


4.2 POINTMAP AND DEPTH PREDICTION


For monocular depth prediction and pointmap prediction, we evaluate our model on several datasets
containing both indoor and outdoor scenes. We compare our method with MoGe Wang et al.
(2025b), VGGT Wang et al. (2025a), MASt3R Leroy et al. (2024) and DUSt3R Wang et al. (2024).
The qualitative comparison is shown in Fig. 5. Our method achieves high-quality pointmap prediction and depth estimation with the intrinsic-invariant pointmap and the novel training strategy. As
for pointmap prediction, MoGe and VGGT often fail to recover depth for reflective surfaces and
tend to produce flattened pointmaps in background regions. In contrast, our method accomplishes
to predict accurate depth with high-quality pointmaps. Moreover, Dens3R yields more stable and
high-quality predictions than MASt3R. Our method also generates more accurate depth maps than
DUSt3R, which can be reflected from the depth predictions for the Chandeliers.


5 CONCLUSION


We propose Dens3R, a 3D foundation model for dense geometric prediction that jointly regresses
multiple geometric quantities, including depth, surface normals, and pointmaps, from unconstrained


9


Figure 5: Qualitative comparison of depth maps and pointmaps. We compare our method with
previous DUSt3R-based methods and demonstrate high-quality depth prediction results. Dens3R
also reconstructs more stable and accurate pointmap than previous methods.


image inputs. Unlike previous approaches that estimate geometry in isolation, Dens3R explicitly
models the structural coupling among these properties to ensure consistency and improves overall
accuracy. We utilize a two-stage training framework with coarse-to-fine strategy and build an accurate intrinsic-invariant pointmap representation. In addition, we design a lightweight encoderdecoder architecture and position-interpolated rotary positional encoding to enable scalable and
high-fidelity inference for high-resolution inputs. Moreover, Dens3R incorporates a geometrically
consistent post-processing pipeline for multi-view inputs. Extensive experiments demonstrate our
superior performance across various 3D prediction benchmarks and highlight the potential as a versatile backbone for broader downstream applications.


10


REFERENCES


Eduardo Arnold, Jamie Wynn, Sara Vicente, Guillermo Garcia-Hernando, Aron [´] Monszpart, Victor Adrian Prisacariu, Daniyar Turmukhambetov, and Eric Brachmann. Map-free visual relocalization: Metric pose relative to a single image. In _ECCV_, volume 13661, pp. 690–708, 2022.


Gwangbin Bae and Andrew J. Davison. Rethinking inductive biases for surface normal estimation.
In _CVPR_, pp. 9535–9545, 2024.


Aayush Bansal, Bryan C. Russell, and Abhinav Gupta. Marr revisited: 2d-3d alignment via surface
normal prediction. In _CVPR_, pp. 5965–5974, 2016.


Jonathan T. Barron, Ben Mildenhall, Matthew Tancik, Peter Hedman, Ricardo Martin-Brualla, and
Pratul P. Srinivasan. Mip-nerf: A multiscale representation for anti-aliasing neural radiance fields.
In _ICCV_, pp. 5835–5844, 2021.


Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Zip-nerf:
Anti-aliased grid-based neural radiance fields. In _ICCV_, pp. 19697–19705, 2023.


Gilad Baruch, Zhuoyuan Chen, Afshin Dehghan, Tal Dimry, Yuri Feigin, Peter Fu, Thomas Gebauer,
Brandon Joffe, Daniel Kurz, Arik Schwartz, and Elad Shulman. ARKitscenes  - a diverse realworld dataset for 3d indoor scene understanding using mobile RGB-d data. In _NeurIPS Datasets_
_and Benchmarks_, 2021.


Yash Bhalgat, Jo˜ao F. Henriques, and Andrew Zisserman. A light touch approach to teaching transformers multi-view geometry. In _CVPR_, pp. 4958–4969, 2023.


Shariq Farooq Bhat, Ibraheem Alhashim, and Peter Wonka. Adabins: Depth estimation using adaptive bins. In _CVPR_, pp. 4009–4018, 2021.


Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias M¨uller. Zoedepth: Zeroshot transfer by combining relative and metric depth. _arxiv preprint arXiv:2302.12288_, 2023.


Ming-Fang Chang, John W Lambert, Patsorn Sangkloy, Jagjeet Singh, Slawomir Bak, Andrew Hartnett, De Wang, Peter Carr, Simon Lucey, Deva Ramanan, and James Hays. Argoverse: 3d tracking
and forecasting with rich maps. In _CVPR_, pp. 8748–8757, 2019.


Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window
of large language models via positional interpolation. _arxiv preprint arXiv:2306.15595_, 2023.


Weifeng Chen, Zhao Fu, Dawei Yang, and Jia Deng. Single-image depth perception in the wild. In
_NeurIPS_, pp. 730–738, 2016.


Weifeng Chen, Shengyi Qian, David Fan, Noriyuki Kojima, Max Hamilton, and Jia Deng. OASIS:
A large-scale dataset for single image 3d in the wild. In _CVPR_, pp. 676–685, 2020.


Jaehoon Cho, Dongbo Min, Youngjung Kim, and Kwanghoon Sohn. A large rgb-d dataset for semisupervised monocular depth estimation. _arXiv preprint arXiv:1904.10230_, 2019.


Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu,
Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin,
and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. In
_CVPR_, pp. 21094–21104, 2022.


Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati,
Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha
Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi.
Objaverse-xl: A universe of 10m+ 3d objects. In _NeurIPS_, 2023.


Johan Edstedt, Ioannis Athanasiadis, M˚arten Wadenb¨ack, and Michael Felsberg. DKM: dense kernelized feature matching for geometry estimation. In _CVPR_, pp. 17765–17775, 2023.


Johan Edstedt, Qiyu Sun, Georg B¨okman, M˚arten Wadenb¨ack, and Michael Felsberg. Roma: Robust
dense feature matching. In _CVPR_, pp. 19790–19800, 2024.


11


Ufuk Efe, Kutalmis Gokalp Ince, and A. Aydin Alatan. DFM: A performance baseline for deep
feature matching. In _CVPRW_, pp. 4284–4293, 2021.


Ainaz Eftekhar, Alexander Sax, Jitendra Malik, and Amir Zamir. Omnidata: A scalable pipeline for
making multi-task mid-level vision datasets from 3d scans. In _ICCV_, pp. 10766–10776, 2021.


David Eigen, Christian Puhrsch, and Rob Fergus. Depth map prediction from a single image using
a multi-scale deep network. In _NeurIPS_, pp. 2366–2374, 2014.


Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and
Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a
single image. In _ECCV_, volume 15080, pp. 241–258, 2024.


Adrien Gaidon, Qiao Wang, Yohann Cabon, and Eleonora Vig. Virtual worlds as proxy for multiobject tracking analysis. In _CVPR_, pp. 4340–4349, 2016.


Cl´ement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel J. Brostow. Digging into selfsupervised monocular depth estimation. In _ICCV_, pp. 3827–3837, 2019.


Ming Gui, Johannes S. Fischer, Ulrich Prestel, Pingchuan Ma, Dmytro Kotovenko, Olga
Grebenkova, Stefan Andreas Baumann, Vincent Tao Hu, and Bj¨orn Ommer. Depthfm: Fast
monocular depth estimation with flow matching. _arxiv preprint arXiv:2403.13788_, 2024.


Jose L. G´omez, Manuel Silva, Antonio Seoane, Agn´es Borr`as, Mario Noriega, German Ros, Jose A.
Iglesias-Guitian, and Antonio M. L´opez. All for one, and one for all: Urbansyn dataset, the third
musketeer of synthetic driving scenes. _Neurocomputing_, 637:130038, 2025.


Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Liu, Bingbing
Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality
dense prediction. In _ICLR_, 2025.


Yihui He, Rui Yan, Katerina Fragkiadaki, and Shoou-I Yu. Epipolar transformers. In _CVPR_, pp.
7776–7785, 2020.


Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jiaolong Yang, Seungryong Kim, and Chong Luo.
Unifying correspondence, pose and nerf for pose-free novel view synthesis from stereo pairs. In
_CVPR_, 2024.


Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu,
Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation
model for zero-shot metric depth and surface normal estimation. _IEEE Trans. Pattern Anal. Mach._
_Intell._, 46(12):10579–10596, 2024.


Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely,
and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. In _ICLR_,
2025.


Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad
Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In
_CVPR_, pp. 9492–9502, 2024.


Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM Trans. Graph._, 42(4):139:1–139:14, 2023.


Yushi LAN, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Bo Dai,
Shuai Yang, Chen Change Loy, and Xingang Pan. STream3r: Scalable sequential 3d reconstruction with causal transformer. In _ICLR_, 2026.


Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r.
In _ECCV_, volume 15130, pp. 71–91, 2024.


12


Chengshu Li, Ruohan Zhang, Josiah Wong, Cem Gokmen, Sanjana Srivastava, Roberto Mart´ınMart´ın, Chen Wang, Gabrael Levine, Michael Lingelbach, Jiankai Sun, Mona Anvari, Minjune
Hwang, Manasi Sharma, Arman Aydin, Dhruva Bansal, Samuel Hunter, Kyu-Young Kim, Alan
Lou, Caleb R. Matthews, Ivan Villa-Renteria, Jerry Huayang Tang, Claire Tang, Fei Xia, Silvio
Savarese, Hyowon Gweon, C. Karen Liu, Jiajun Wu, and Li Fei-Fei. Behavior-1k: A benchmark
for embodied ai with 1, 000 everyday activities and realistic simulation. In _CORL_, volume 205,
pp. 80–93, 2022.


Yixuan Li, Lihan Jiang, Linning Xu, Yuanbo Xiangli, Zhenzhi Wang, Dahua Lin, and Bo Dai.
Matrixcity: A large-scale city dataset for city-scale neural rendering and beyond. In _ICCV_, pp.
3205–3215, 2023.


Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet
photos. In _CVPR_, pp. 2041–2050, 2018a.


Zhengqi Li and Noah Snavely. Megadepth: Learning single-view depth prediction from internet
photos. In _CVPR_, pp. 2041–2050, 2018b.


Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo,
Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d
vision. In _CVPR_, pp. 22160–22169, 2024.


Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma,
Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3d: Single
image to 3d using cross-domain diffusion. In _CVPR_, pp. 9970–9980, 2024.


Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs:
Structured 3d gaussians for view-adaptive rendering. _CVPR_, 2024.


Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo
collections. In _CVPR_, pp. 7210–7219, 2021.


Lukas Mehl, Jenny Schmalfuss, Azin Jahedi, Yaroslava Nalivayko, and Andr´es Bruhn. Spring: A
high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In
_CVPR_, pp. 4981–4991, 2023.


Iaroslav Melekhov, Aleksei Tiulpin, Torsten Sattler, Marc Pollefeys, Esa Rahtu, and Juho Kannala.
Dgc-net: Dense geometric correspondence network. In _WACV_, pp. 1034–1042, 2019.


Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In _ECCV_, 2020.


Simon Niklaus, Long Mai, Jimei Yang, and Feng Liu. 3d ken burns effect from a single image. _ACM_
_Trans. Graph._, 38(6):184:1–184:15, 2019.


Aaronvanden Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding. _arXiv preprint arXiv:1807.03748_, 2018.


Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Seg`u, Siyuan Li, Luc Van Gool, and
Fisher Yu. Unidepth: Universal monocular metric depth estimation. In _CVPR_, pp. 10106–10116,
2024.


Alexander Raistrick, Lingjie Mei, Karhan Kayan, David Yan, Yiming Zuo, Beining Han, Hongyu
Wen, Meenal Parakh, Stamatis Alexandropoulos, Lahav Lipson, Zeyu Ma, and Jia Deng. Infinigen
indoors: Photorealistic indoor scenes using procedural generation. In _CVPR_, pp. 21783–21794,
2024.


Ren´e Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction.
In _ICCV_, pp. 12159–12168, 2021.


Ren´e Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, and Vladlen Koltun. Towards robust
monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. _IEEE_ _Trans._
_Pattern Anal. Mach. Intell._, 44(3):1623–1637, 2022.


13


Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and
David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In _ICCV_, pp. 10881–10891, 2021.


Stephan R Richter, Vibhav Vineet, Stefan Roth, and Vladlen Koltun. Playing for data: Ground truth
from computer games. In _ECCV_, volume 9906, pp. 102–118, 2016.


Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan
Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for
holistic indoor scene understanding. In _ICCV_, pp. 10892–10902, 2021.


German Ros, Laura Sellart, Joanna Materzynska, David Vazquez, and Antonio M. Lopez. The
synthia dataset: A large collection of synthetic images for semantic segmentation of urban scenes.
In _CVPR_, pp. 3234–3243, 2016.


Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Superglue:
Learning feature matching with graph neural networks. In _CVPR_, pp. 4937–4946, 2020.


Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain,
Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, Devi Parikh, and Dhruv Batra. Habitat: A
platform for embodied ai research. In _ICCV_, pp. 9338–9346, 2019.


Philipp Schr¨oppel, Jan Bechtold, Artemij Amiranashvili, and Thomas Brox. A benchmark and a
baseline for robust multi-view depth estimation. In _3DV_, pp. 637–645, 2022.


Brandon Smart, Chuanxia Zheng, Iro Laina, and Victor Adrian Prisacariu. Splatt3r: Zero-shot
gaussian splatting from uncalibrated image pairs. _arxiv preprint arXiv:2408.13912_, 2024.


Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, and Xiaowei Zhou. Loftr: Detector-free local
feature matching with transformers. In _CVPR_, pp. 8922–8931, 2021.


Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui,
James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan
Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi,
Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for
autonomous driving: Waymo open dataset. In _CVPR_, pp. 2443–2451, 2020.


Carl Toft, Daniyar Turmukhambetov, Torsten Sattler, Fredrik Kahl, and Gabriel J. Brostow. Singleimage depth prediction makes feature matching easier. In _ECCV_, volume 12361, pp. 473–492,
2020.


Fabio Tosi, Yiyi Liao, Carolin Schmitt, and Andreas Geiger. Smd-nets: Stereo mixture density
networks. In _CVPR_, pp. 8942–8952, 2021.


Prune Truong, Martin Danelljan, and Radu Timofte. Glu-net: Global-local universal network for
dense flow and correspondences. In _CVPR_, pp. 6257–6267, 2020.


Prune Truong, Martin Danelljan, Luc Van Gool, and Radu Timofte. Learning accurate dense correspondences and when to trust them. In _CVPR_, pp. 5714–5724, 2021.


Prune Truong, Martin Danelljan, Radu Timofte, and Luc Van Gool. Pdc-net+: Enhanced probabilistic dense correspondence network. _IEEE Trans. Pattern Anal. Mach. Intell._, 45(8):10247–10266,
2023.


Qiang Wan, Zilong Huang, Bingyi Kang, Jiashi Feng, and Li Zhang. Harnessing diffusion models
for visual perception with meta prompts. _arxiv preprint arXiv:2312.14733_, 2023.


Hengyi Wang and Lourdes Agapito. 3d reconstruction with spatial memory. In _3DV_, 2025.


Jianyuan Wang, Christian Rupprecht, and David Novotn´y. Posediffusion: Solving pose estimation
via diffusion-aided bundle adjustment. In _ICCV_, pp. 9739–9749, 2023a.


Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David
Novotny. Vggt: Visual geometry grounded transformer. In _CVPR_, 2025a.


14


Kaixuan Wang and Shaojie Shen. Flow-motion and depth network for monocular stereo and beyond.
_IEEE Robotics and Automation Letters_, 5(2):3307–3314, 2020.


Peng Wang, Lingjie Liu, Yuan Liu, Christian Theobalt, Taku Komura, and Wenping Wang. Neus:
Learning neural implicit surfaces by volume rendering for multi-view reconstruction. In _NeurIPS_,
pp. 27171–27183, 2021a.


Qiang Wang, Shizhen Zheng, Qingsong Yan, Fei Deng, Kaiyong Zhao, and Xiaowen Chu. Irs:
A large naturalistic indoor robotics stereo dataset to train deep models for disparity and surface
normal estimation. In _ICME_, pp. 1–6, 2021b.


Qianqian Wang, Xiaowei Zhou, Bharath Hariharan, and Noah Snavely. Learning feature descriptors
using camera pose supervision. In _ECCV_, volume 12346, pp. 757–774, 2020a.


Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang.
Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal
training supervision. In _CVPR_, 2025b.


Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and J´erˆome Revaud. Dust3r: Geometric 3d vision made easy. In _CVPR_, pp. 20697–20709, 2024.


Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu,
Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. pp.
4909–4916, 2020b.


Xiaolong Wang, David F. Fouhey, and Abhinav Gupta. Designing deep networks for surface normal
estimation. In _CVPR_, pp. 539–547, 2015.


Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt, and Lingjie Liu.
Neus2: Fast learning of neural implicit surfaces for multi-view reconstruction. In _ICCV_, 2023b.


Yuang Wang, Xingyi He, Sida Peng, Haotong Lin, Hujun Bao, and Xiaowei Zhou. Autorecon:
Automated 3d object discovery and reconstruction. In _CVPR_, 2023c.


Hongchi Xia, Yang Fu, Sifei Liu, and Xiaolong Wang. RGBD objects in the wild: Scaling real-world
3d object learning from RGB-D videos. In _CVPR_, pp. 22378–22389, 2024.


Guangkai Xu, Yongtao Ge, Mingyu Liu, Chengxiang Fan, Kangyang Xie, Zhiyue Zhao, Hao Chen,
and Chunhua Shen. What matters when repurposing diffusion models for general dense perception tasks? In _ICLR_, 2025.


Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai,
Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one
forward pass. In _CVPR_, 2025.


Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth
anything: Unleashing the power of large-scale unlabeled data. In _CVPR_, pp. 10371–10381, 2024a.


Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang
Zhao. Depth anything v2. In _NeurIPS_, 2024b.


Yao Yao, Zixin Luo, Shiwei Li, Jingyang Zhang, Yufan Ren, Lei Zhou, Tian Fang, and Long Quan.
Blendedmvs: A large-scale dataset for generalized multi-view stereo networks. In _CVPR_, pp.
1787–1796, 2020.


Yuan Yao, Yasamin Jafarian, and Hyun Soo Park. MONET: multiview semi-supervised keypoint
detection via epipolar divergence. In _ICCV_, pp. 753–762, 2019.


Lior Yariv, Jiatao Gu, Yoni Kasten, and Yaron Lipman. Volume rendering of neural implicit surfaces.
In _NeurIPS_, pp. 4805–4815, 2021.


Botao Ye, Sifei Liu, Haofei Xu, Li Xueting, Marc Pollefeys, Ming-Hsuan Yang, and Peng Songyou.
No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. In
_ICLR_, 2025.


15


Chongjie Ye, Lingteng Qiu, Xiaodong Gu, Qi Zuo, Yushuang Wu, Zilong Dong, Liefeng Bo, Yuliang
Xiu, and Xiaoguang Han. Stablenormal: Reducing diffusion variance for stable and sharp normal.
_ACM Trans. Graph._, 43(6):250:1–250:18, 2024.


Chandan Yeshwanth, Yueh-Cheng Liu, Matthias Nießner, and Angela Dai. Scannet++: A highfidelity dataset of 3d indoor scenes. In _ICCV_, pp. 12–22, 2023.


Wang Yifan, Carl Doersch, Relja Arandjelovic, Jo˜ao Carreira, and Andrew Zisserman. Input-level
inductive biases for 3d reconstruction. In _CVPR_, pp. 6166–6176, 2022.


Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, and Chunhua
Shen. Metric3d: Towards zero-shot metric 3d prediction from A single image. In _ICCV_, pp.
9009–9019, 2023.


Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Aliasfree 3d gaussian splatting. _CVPR_, 2024.


Amir R Zamir, Alexander Sax,, William B Shen, Leonidas Guibas, Jitendra Malik, and Silvio
Savarese. Taskonomy: Disentangling task transfer learning. In _CVPR_, pp. 3712–3722, 2018.


Jason Y. Zhang, Amy Lin, Moneish Kumar, Tzu-Hsuan Yang, Deva Ramanan, and Shubham Tulsiani. Cameras as rays: Pose estimation via ray diffusion. In _ICLR_, 2024.


Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3d: A large
photo-realistic dataset for structured 3d modeling. In _ECCV_, volume 12354, pp. 519–535, 2020.


Qunjie Zhou, Torsten Sattler, and Laura Leal-Taix´e. Patch2pix: Epipolar-guided pixel-level correspondences. In _CVPR_, pp. 4669–4678, 2021.


Shengjie Zhu and Xiaoming Liu. Pmatch: Paired masked image modeling for dense geometric
matching. In _CVPR_, pp. 21909–21918, 2023.


16


A APPENDIX


In addition to the results presented in the main paper and appendix, we also provide further experiments and visualizations in the _**Supplementary**_ _**Materials**_, including unified geometric prediction
from monocular inputs, image-matching visualizations, and multi-view reconstruction results.


A.1 ABLATION STUDY


We present high-quality geometric predictions for high-resolution inputs and various scenarios in
Fig. 6 and Fig. 7. We then conducted comprehensive ablation studies for our key components: the
position-interpolated rotary positional encoding, the intrinsic-invariant training and the coarse-tofine training strategy.

|NYUv2<br>Mean ↓ δ11.25◦ ↑|ScanNet<br>Mean ↓ δ11.25◦ ↑|IBims<br>Mean ↓ δ11.25◦ ↑|Sintel<br>Mean ↓ δ11.25◦ ↑|
|---|---|---|---|
|17.8<br>50.6<br>17.6<br>50.5<br>**16.1**<br>**62.5**|18.6<br>49.4<br>17.8<br>58.8<br>**16.9**<br>**64.0**|20.2<br>56.8<br>18.6<br>63.9<br>**16.0**<br>**72.2**|35.9<br>18.9<br>35.8<br>22.3<br>**30.7**<br>**28.9**|


Table 3: Normal quantitative metrics for ablation. We demonstrate that both the intrinsic-invariant
training and coarse-to-fine strategy contributes to accurate normal predictions.


Figure 6: High-quality geometric predictions for high-resolution (2K) inputs. Please zoom in to
better observe the fine-grained details.


**Position-Interpolated** **Rotary** **Positional** **Encoding.** Dens3R can support multi-resolution image
inputs. With the position-interpolated rotary positional encoding and the coarse-to-fine training
strategy, our method can prevent performance degradation when handling high-resolution inputs.
As shown in Fig. 8a, we can generate accurate and well-structured pointmaps with the positioninterpolated RoPE, preventing the model from producing overlapping or inconsistent pointmaps at
higher resolutions.


**Intrinsic-Invariant** **Training.** Our approach first learns a scale-invariant pointmap, which is then
transformed into an intrinsic-invariant pointmap via subsequent intrinsic-invariant training. We find
that jointly training the pointmap and normal at the initial scale-invariant stage leads to instability
and poor convergence. This is because pointmaps and normal maps lie in different data domains,
and coupling their supervision potentially increases training complexity. While GeoWizard Fu et al.


17


Figure 7: High-quality unified geometric predictions for various scenarios. We demonstrate accurate
normal and depth predictions with high-quality 3D pointmaps for challenging object-centric, indoor
and outdoor scenes.


Setting Compute Cost Memory Cost Network Params


w/o Shared 1.362 TFlops 4.6 GB 737.591 M
w/ Shared 1.362 TFlops **4.1 GB** **624.152 M**


Table 4: Ablation on shared encoder-decoder structure. We conduct experiments for both of the
model on image pairs with 512 resolution. With the shared encoder-decoder structure, our model
yields lower memory cost and less network parameters.


(2024) addresses this domain gap with a task switcher, we adopt a two-stage training scheme to learn
an intrinsic-invariant pointmap, ensuring stable learning. As shown in Tab. 3, Tab. 6 and in Fig. 8b,
the performance of the model will degrade without the intrinsic-invariant training.


We also go beyond purely image-domain monocular normal estimation, where methods like StableNormal Ye et al. (2024) operate on a single view and therefore still suffer from monocular ambiguity. We concatenate pointmap and normal features so that the normal head can exploit the
multi-view geometric information encoded in the Stage 1 pointmap, which helps resolve ambiguity
that cannot be solved from a single image alone. At the same time, the normal predictions provide
additional geometric information from the normal domain that refine the pointmap representation
as shown in Fig. 11. We therefore view the pointmap–normal interaction as a bidirectional mechanism: the multi-view pointmap supplies information that helps the normal head resolve monocular
geometric ambiguities, while the normals, in turn, regularize and refine the 3D geometric representation.


**Coarse-to-Fine** **Training.** Our model is trained on diverse training dataset of varying quality. To
better utilize the full training set, we implement a coarse-to-fine training strategy that gradually
increases resolution and data fidelity. In the coarse stage, we set the max resolution of the training
images as 512 and enable all the training data. In the fine stage, we increase the image resolution to
1024 pixels and restrict training to the high-resolution data only. As demonstrated in Tab. 3 and in
Fig. 8b, this strategy improves prediction accuracy, particularly for high-resolution outputs.


**Shared Encoder-Decoder Backbone Ablation** Dens3R employs a dense visual transformer backbone designed to capture spatial relationships across viewpoints and capture the global 3D geometric
information of scenes. Different from previous methods, both the encoder and decoder components
in our architecture share weights. The comparison of the network parameters and the memory cost is
shown in Tab. 4. Since our model deals with more 3D quantities than previous methods, the framework initially requires a higher memory cost. Employing the shared encoder-decoder structure also
resolves this issue, reducing the memory cost and network parameters without losing the prediction
quality.


18


(a) High-resolution inference comparison. Our
method supports high-resolution input and generates
accurate and well-structured pointmaps.


(c) Segmentation results. Our model can be easily
extended to segmentation tasks by training a new prediction head with the backbone frozen.


(b) Normal comparison for ablation. The intrinsicinvariant training enables accurate normal prediction
and the coarse-to-fine training enhances details.


(d) Normal supervision results. We demonstrate the
effectiveness of using our normal as the supervision
of surface reconstruction.


Figure 8: Ablation and downstream applications.


A.2 DOWNSTREAM APPLICATIONS


**Segmentation Head Training.** Dens3R serves as a visual foundation model that can be finetuned for
several downstream tasks. We demonstrate this by training a new prediction head for segmentation
task while keeping our backbone frozen. As shown in Fig. 8c, the segmentation head can generate
accurate results, with much more effortless training than a large segmentation model.


**Surface** **Reconstruction.** Dens3R can improve surface reconstruction quality by its sharp and
accurate normals. We demonstrate this by utilizing our predicted normals as the supervision for
NeuS Wang et al. (2021a) training. The results are showcased in Fig. 8d. It can be seen that the final
reconstruction results are improved due to the strong normal prior provided by our Dens3R.


Dens3R can also facilitate surface reconstruction by providing accurate 3D priors, including point
maps, depth, and normals. We implement an end-to-end automated surface reconstruction pipeline
following AutoRecon Wang et al. (2023c). We showcase the high-quality reconstruction results in
Fig. 9.


A.3 IMPLEMENTATION DETAILS


**Datasets.** To train the visual foundation model, we collect and reorganize a large-scale training
dataset containing various data types. The dataset includes indoor scenes, outdoor scenes, and
object-level data. It is noteworthy that the quality of training data has a substantial impact on model
performance. We then make the most of high-quality synthetic data in the training process for more
accurate and robust predictions. We divide all the data into three types based on their quality. Data
of type A is collected from synthetic rendering process with the highest quality. Data of type B also
originates from synthetic rendering, but they possess certain quality issues like insufficient resolution or absence of background or imprecise original 3D models, _etc_ . Data of type C is obtained from
the real world using cameras and depth sensors. We also carefully allocate the proportions of each


19


Figure 9: High-quality automated surface reconstruction results. We implement an end-to-end automated surface reconstruction pipeline using Dens3R and showcase the results.

|Dataset|Type|Applied Losses<br>L, L L L L<br>pts loc pts glb pts n match n|Image<br>Pairs|Ratio|
|---|---|---|---|---|
|Hypersim Roberts et al. (2021)<br>UnrealStereo4K Tosi et al. (2021)<br>MatrixCity Li et al. (2023)<br>Infnigen Raistrick et al. (2024)<br>Behavior Li et al. (2022)<br>Structure3D Zheng et al. (2020)<br>GTASFM Wang & Shen (2020)<br>GTAV Richter et al. (2016)<br>VirtualKitti Gaidon et al. (2016)<br>IRS Wang et al. (2021b)<br>UrbanSyn G´omez et al. (2025)<br>Spring Mehl et al. (2023)|A<br>A<br>A<br>A<br>A<br>A<br>A<br>A<br>A<br>A<br>A<br>A|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|1.8M<br>0.9M<br>0.7M<br>2.8M<br>6.8M<br>0.2M<br>0.2M<br>0.6M<br>4.0M<br>74K<br>7.0K<br>10K|6.77%<br>6.77%<br>6.77%<br>6.77%<br>6.77%<br>4.06%<br>13.53%<br>13.53%<br>13.53%<br>0.41%<br>0.41%<br>0.41%|
|ScanNet++ Yeshwanth et al. (2023)<br>ABO Collins et al. (2022)<br>GObjaverseXL Deitke et al. (2023)<br>StaticThings3D Schr¨oppel et al. (2022)<br>BlendedMVS Yao et al. (2020)<br>Habitat Savva et al. (2019)<br>Taskonomy Zamir et al. (2018)<br>ARKitScenes Baruch et al. (2021)<br>Tartanair Wang et al. (2020b)<br>Synthia Ros et al. (2016)<br>KenBurns Niklaus et al. (2019)|B<br>B<br>B<br>B<br>B<br>B<br>B<br>B<br>B<br>B<br>B|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|3.5M<br>2.0M<br>6.8M<br>0.3M<br>1.1M<br>1.3M<br>1.8M<br>2.2M<br>4.5M<br>2.6M<br>0.3M|1.35%<br>1.35%<br>1.35%<br>1.35%<br>1.35%<br>0.68%<br>0.68%<br>0.68%<br>0.68%<br>0.68%<br>0.68%|
|MegaDepth Li & Snavely (2018b)<br>Waymo Sun et al. (2020)<br>Co3dv2 Reizenstein et al. (2021)<br>WildRGBD Xia et al. (2024)<br>NianticMapFree Arnold et al. (2022)<br>DL3DV Ling et al. (2024)<br>DIMLIndoor Cho et al. (2019)<br>ArgoverseStereo Chang et al. (2019)|C<br>C<br>C<br>C<br>C<br>C<br>C<br>C|✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|1.8M<br>1.1M<br>1.2M<br>1.1M<br>3.7M<br>1.2M<br>0.9M<br>4.0K|1.35%<br>1.35%<br>1.35%<br>1.35%<br>1.35%<br>1.35%<br>0.68%<br>0.68%|


Table 5: Training dataset information. We reorganize a large-scale training dataset and divide the
data into three types based on their quality. We also showcase the training objectives we apply
during training, the number of image pairs and the corresponding dataset ratio.


dataset to attain the optimal model training performance. We summarize and present this dataset
information in Tab. 5.


**Training Details.** During our coarse-to-fine training, we first utilize all the images with 512 resolution and train our model for about two weeks in the coarse-stage training. Then we only utilize
the images from type A dataset and a minor portion of type B dataset and set the image resolution
to 1024 for the fine-stage training. We utilize 32 Nvidia H20 GPUs for both the coarse and fine
stage training. As for model inference, our model only requires a single Nvidia RTX3090 GPU for
1024-resolution image inputs.


20


Figure 10: Normal prediction comparison of different training stages.


A.4 NORMAL AND DEPTH COMPARISON


Dens3R predicts robust and accurate normal and depth for various scenarios. As shown in Fig. 10,
we demonstrate that the intrinsic-invariant training assists the pointmap to capture the geometric
information from normal. Then the normal prediction head further predicts sharper edges and more
accurate results.


We provide the normal prediction comparison of the Kitti dataset in Fig. 13. It can be seen that
our method generates the most accurate and sharp normals. We also provide more comparison of
normal map prediction in Fig 15 using in-the-wild images and in Fig. 16 using DL3DV dataset. It
can be seen that our method predicts sharper and more accurate normal across various scenarios. We
also compare our method with the normal map derived from the pointmap of DUSt3R Wang et al.
(2024), MASt3R Leroy et al. (2024) and the predicted depth map of MoGe Wang et al. (2025b),
the results are shown in Fig. 14. It can be seen that Dens3R can handle normal predictions for
reflective surfaces and accomplishes to generate richer details. We also provide the full quantitative
comparison in Tab. 6 which are partly shown in Tab. 1 in the main paper correspondingly.


We provide the quantitative depth comparison in Tab. 7. It can be seen that our method achieves
accurate results in depth estimation. We also provide additional qualitative depth prediction comparison in Fig. 17, it can be seen that our method generates the most accurate depth maps even
for reflective surfaces. Since VGGT Wang et al. (2025a) also predicts multiple quantities including
depth and matching, we further compare our predicted depth map with VGGT. We demonstrate more
accurate depth predictions of NYUv2 dataset in Fig. 18. We also showcase the accurate prediction
of both indoor scenes of NYUv2 dataset and outdoor scenes of Kitti dataset. It can be seen in Fig. 19
and Fig. 20 that our model also achieves accurate human depth estimation that can be further utilized
for detection and autonomous driving.


A.5 CAMERA POSE ESTIMATION COMPARISON


Dens3R can also perform accurate camera pose estimation through a single feed-forward pass.
We conduct extended experiments to demonstrate its accuracy. We utilize the map-free benchmark Arnold et al. (2022) following the MASt3R protocol Leroy et al. (2024), which is a challenging dataset aiming at localizing the camera in metric space given a single reference image without
any map. We present the camera pose estimation (Map-free relocalization) comparison in Tab. 8.
It can be seen that Dens3R outperforms previous methods in nearly all the metrics, demonstrating
highly accurate camera pose estimation results.


A.6 IMAGE MATCHING COMPARISON


For image-matching, apart from the ZEB dataset, we also provide the quantitative comparison of the
Scannet-1500 dataset in Tab. 9 and the MegaDepth-1500 dataset in Tab. 10. The comparisons on the
ScanNet-1500 and the MegaDepth-1500 benchmarks further demonstrate our superior performance
over pervious DUSt3R-based method MASt3R Leroy et al. (2024) and VGGT Wang et al. (2025a).


A.7 HIGH-RESOLUTION INFERENCE COMPARISON


We showcase more comparison of high-resolution inputs with DUSt3R Wang et al. (2024) and
VGGT Wang et al. (2025a) in Fig. 21. It can be seen that our method can handle higher-resolution


21


Method Mean _↓_ Med _↓_ _δ_ 11 _._ 25 _◦_ _↑_ _δ_ 22 _._ 5 _◦_ _↑_ _δ_ 30 _◦_ _↑_

NYUv2 (indoor)

ScanNet (indoor)

IBims-1 (indoor)

Sintel (outdoor)

DIODE-outdoor (outdoor)


Table 6: Full quantitative comparison of normal prediction. We report the mean and median angular
errors with each cell colored to indicate the best and the second .

|Method|NYUv2<br>REL↓ RMSE↓ δ1 ↑ δ2 ↑ δ3 ↑|DIODE-indoor<br>REL↓ RMSE↓ δ1 ↑ δ2 ↑ δ3 ↑|DIODE-outdoor<br>REL↓ RMSE↓ δ1 ↑ δ2 ↑ δ3 ↑|
|---|---|---|---|
|GenPercept<br>Lotus*<br>DepthAnythingV2<br>DUSt3R<br>VGGT<br>MoGe<br>Ours|0.052<br>0.214<br>96.7<br>99.3<br>99.8<br>0.053<br>0.262<br>96.5<br>99.1<br>99.7<br>0.049<br>0.204<br>97.3<br>99.3<br>99.8<br>0.046<br>0.197<br>97.1<br>99.3<br>99.8<br>0.038<br>0.194<br>98.0<br>99.4<br>99.8<br>0.035<br>0.167<br>97.9<br>99.4<br>99.9<br>0.042<br>0.189<br>97.5<br>99.3<br>99.8|0.107<br>0.924<br>89.1<br>96.0<br>98.1<br>0.111<br>1.123<br>88.7<br>96.0<br>98.4<br>0.091<br>0.878<br>92.5<br>97.3<br>98.6<br>0.083<br>0.375<br>92.0<br>97.7<br>99.0<br>0.064<br>0.404<br>93.1<br>98.0<br>99.2<br>0.080<br>0.879<br>92.6<br>97.3<br>98.7<br>0.072<br>0.372<br>93.7<br>97.5<br>98.8|0.727<br>5.571<br>67.3<br>84.2<br>90.6<br>0.488<br>9.960<br>47.1<br>63.3<br>71.8<br>0.705<br>5.525<br>67.8<br>83.4<br>89.7<br>0.451<br>5.217<br>67.7<br>84.3<br>90.7<br>0.400<br>4.861<br>70.6<br>84.9<br>90.6<br>0.578<br>5.177<br>72.8<br>86.7<br>91.9<br>0.387<br>4.740<br>72.2<br>87.0<br>92.3|


Table 7: Quantitative comparison on monocular depth prediction. We report the relative point error
(REL), root mean square error (RMSE) and the percentage of inliers _δ_ 1 _, δ_ 2 _, δ_ 3 with each cell colored
to indicate the best and the second . *We utilize Lotu-G disparity model for comparison.


22


Method Reproj. Error _↓_ Precision _↑_ AUC _↑_ Median Error (m) _↓_ Median Error (°) _↓_ Pose Precision _↑_ Pose AUC _↑_


DUSt3R 125.8 px 45.2% 0.704 1.10 m 9.4° 17.0% 0.344


Table 8: Camera pose estimation results of the Map-free dataset. We report the metrics with each
cell colored to indicate the best and the second .


Method AUC@5 _[◦]_ _↑_ AUC@10 _[◦]_ _↑_ AUC@20 _[◦]_ _↑_


Table 9: Two-view matching comparison on ScanNet-1500 Dataset. We report the AUC values with
each cell colored to indicate the best and the second . Our method achieves state-of-the-art for
two-view matching, surpassing all the previous methods.


Method AUC@5 _[◦]_ _↑_ AUC@10 _[◦]_ _↑_ AUC@20 _[◦]_ _↑_


Table 10: Two-view matching comparison on MegaDepth-1500 Dataset. We report the AUC values
with each cell colored to indicate the best and the second . Our method also achieves state-of-theart for the two-view matching using the MegaDepth-1500 Dataset.


23


Figure 11: Pointmap comparison of Stage 1 and Stage 2. We demonstrate that the normal predictions
provide additional geometric information that refine the pointmap representation.


Figure 12: Limitations. Despite that our method outperforms previous methods in geometric predictions, the prediction quality for thin structures still require further improvement.


inputs without causing degenerated predictions like previous methods with our proposed positioninterpolated rotary positional encoding. We also provide additional qualitative comparisons in
Fig. 22 to demonstrate the effect of position-interpolated RoPE. It can be observed that adding
this design on top of DUSt3R already improves the inference quality for higher-resolution inputs.
We also empirically validate that when the network has not been exposed to high-resolution inputs
during training, using position-interpolated RoPE alone is not sufficient because it has no guidance
on the “correct” way to extrapolate. In our full model, we combine it with the intrinsic-invariant
pointmap and a coarse-to-fine training scheme, and we obtain more accurate reconstructions at high
resolution. We believe these results show that our use of position-interpolated RoPE is not a trivial
plug-in, but a natural component of a dedicated training and representation framework for highresolution inference.


A.8 LIMITATION


Although Dens3R outperforms previous methods in geometric predictions, predicting accurate results for inputs with thin structures remains a significant challenge. Restricted by the network’s
limited capacity and the presence of noisy training data, our method may predict inaccurate results
for these inputs. As shown in Fig. 12, the prediction quality for thin structures still require further
improvement.


24


Figure 13: Normal comparison of Kitti dataset. We present more normal comparison of outdoor
scenes, our method produces more accurate and sharper normals than previous methods.


Figure 14: Normal comparison with DUSt3R, MASt3R and MoGe. We provide more normal comparison with the normal maps derived from DUSt3R, MASt3R and MoGe. Dens3R yields sharper
and more accurate predictions.


25


Figure 15: More qualitative comparison of normal map. We provide more normal comparison of
both object-centric and human scenes. Dens3R is able to produce more accurate and sharper results


26


Figure 16: More qualitative comparison of normal map. We provide more normal comparison of
both indoor and outdoor scenes. Dens3R is able to produce sharper and more accurate results and
surpasses previous methods.


Figure 17: Additional depth comparison. We provide more depth comparison with previous methods
and our method can predict more accurate and detailed results.


27


Figure 18: Additional depth comparison with VGGT. We compare our depth prediction results with
VGGT and Dens3R demonstrates more robust and accurate predictions.


Figure 19: Additional depth comparison of indoor scenes with VGGT. Dens3R demonstrates more
accurate results for human depth estimation.


28


Figure 20: Additional depth comparison of outdoor scenes with VGGT. We compare our depth
prediction results of autonomous driving dataset. Our methods achieves much more accurate predictions.


29


Figure 21: Additional high-resolution inference comparison. We provide more high-resolution inference results to demonstrate the effectiveness of the proposed position-interpolated rotary positional
encoding. We present the pointmap of the main frame and our method accomplishes to prevent the
degeneration problem that occured in previous methods.


30


Figure 22: Additional high-resolution inference comparison. We empirically validate that using
position-interpolated RoPE alone is not sufficient for high-resolution inference.


31
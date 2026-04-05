# VOMP: PREDICTING VOLUMETRIC MECHANICAL PROPERTY FIELDS


**Rishit Dagli** [1] _[,]_ [2] **Donglai Xiang** [1] **Vismay Modi** [1] **Charles Loop** [1] **Clement Fuji Tsang** [1]

**Anka He Chen** [1] **Anita Hu** [1] **Gavriel State** [1] **David I.W. Levin** [1] _[,]_ [2] **Maria Shugrina** [1]

1NVIDIA 2University of Toronto

[https://research.nvidia.com/labs/sil/projects/vomp](https://research.nvidia.com/labs/sil/projects/vomp)


**Volumetric**
**Property**
**Fields**


metal


leaves


VoMP

**8s**


bedframe


Young’s ModulusPoisson’s Ratio Density


Poisson’s Ratio Density


Figure 1: **VoMP** predicts physically accurate volumetric mechanical property fields across 3D representations
in just a few seconds (top), enabling their use in realistic deformable simulations (bottom).


ABSTRACT


Physical simulation relies on spatially-varying mechanical properties, often laboriously hand-crafted. VoMP is a feed-forward method trained to predict Young’s
modulus ( _E_ ), Poisson’s ratio ( _ν_ ), and density ( _ρ_ ) throughout _the_ _volume_ of 3D
objects, in any representation that can be rendered and voxelized. VoMP aggregates per-voxel multi-view features and passes them to our trained Geometry
Transformer to predict per-voxel material latent codes. These latents reside on a
space of physically plausible materials, which we learn from a real-world dataset,
guaranteeing the validity of decoded per-voxel materials. To obtain object-level
training data, we propose an annotation pipeline combining knowledge from segmented 3D datasets, material databases, and a vision-language model, along with
a new benchmark. Experiments show that VoMP estimates accurate volumetric
properties, far outperforming prior art in accuracy and speed.


1 INTRODUCTION


Accurate physics simulation is a critical part of modern design and engineering, for example, in
workflows like creating Digital Twins (virtual replicas of real systems) (Grieves & Vickers, 2017),
Real-2-Sim (generating digital simulation from the real world) (NVIDIA, 2019), and Sim-2-Real
(transferring policies trained in simulation to real-world deployment) (Rudin et al., 2021). However,
setting up reliable simulations remains labor-intensive, partially due to the necessity to provide accurate mechanical properties _throughout_ _the_ _volume_ of every object, namely the spatially-varying
Young’s Modulus ( _E_ ), Poisson’s ratio ( _ν_ ), and density ( _ρ_ ). Common 3D capture methods (Kerbl
et al., 2023) and 3D repositories (Deitke et al., 2023) rarely contain such annotations, forcing artists
and engineers to guess or copy-paste coarse material presets in a subjective, time-consuming process. We focus on automatic prediction of these parameters, addressing important limitations of
prior art.


We propose VoMP, _the_ _first_ _feed-forward_ _model_ _trained_ _to_ _estimate_ _simulation-ready_ _mechanical_
_property_ _fields_ ( _E, ν, ρ_ ) _within_ _the_ _volume_ _of_ _3D_ _objects_ _across_ _representations_ . Rather than specializing on inputs like Gaussian Splats (Shuai et al., 2025; Xie et al., 2024), our method works


1


for any geometry that can be voxelized and rendered from turnaround views, including meshes,
Gaussian Splats, NeRFs and SDFs (Fig. 1). Unlike virtually all prior works, VoMP is fully feedforward, requiring no per-object optimization of feature fields (Zhai et al., 2024; Shuai et al., 2025)
or run-time aggregation of Vision-Language Model (VLM) (Lin et al., 2025a) or Video Model (Lin
et al., 2025b) supervision. Uniquely among others, VoMP outputs true mechanical properties (a.k.a.
material parameters), like those measured in the real world. Many existing pipelines target fast,
approximate simulators, resulting in simulator-specific parameters (Zhang et al., 2025; Huang et al.,
2024b) that may not transfer reliably across frameworks (Fig. 2), whereas our result is directly
compatible with any accurate simulator. Finally, unlike prior art, our method is designed to assign
materials throughout the object volume, which is critical for simulation fidelity.


To enable learning physically valid mechanical properties, we first train a latent space on a database
of real-world values ( _E, ν, ρ_ ) using a variational auto-encoder MatVAE (§3). To predict mechanical
property fields for 3D objects, our method first voxelizes the input geometry and aggregates multiview image features across the voxels (§4.1). This process accepts many representations [1] and is fast,
unlike optimization used in concurrent work (Le et al., 2025). We pass the voxel features through
the Geometry Transformer (§4.2), trained to output per-voxel material latents. The MatVAE latent
space decouples learning material assignments for objects from learning what materials are valid,
ensuring that the final volumetric properties ( _E, ν, ρ_ ) decoded by MatVAE are physically valid, even
in the case of interpolation. To create material property fields for training, we propose a pipeline
(§5) combining the knowledge from part-segmented 3D assets, material databases, visual textures,
and a VLM. Our experiments (§6) show that VoMP estimates simulation-ready spatially-varying
mechanical properties across a range of object classes and representations, resulting in realistic
elastodynamic simulations. We evaluate our method on an existing mass prediction benchmark and
contribute a new material estimation benchmark (§6.3), consistently outperforming prior art (Shuai
et al., 2025; Lin et al., 2025a; Zhai et al., 2024). In summary, our contributions are:


- The first (to our knowledge) method to estimate object mechanical material property fields that _(1)_
is a trained feed-forward model with minimal preprocessing, _(2)_ generalizes across 3D representations, _(3)_ predicts physically valid properties that can be used with an accurate simulator, and
_(4)_ predicts mechanical properties _within the volume_ of objects (§4).


- The first (to our knowledge) mechanical properties latent space (§3).


- An automatic data annotation pipeline and a new benchmark for volumetric physics materials (§5).


- Thorough evaluation through high-fidelity simulations and quantitative metrics on existing and
new benchmarks, significantly outperforming the prior art (§6).


2 RELATED WORK


2.1 BACKGROUND


All algorithms for continuum-based simulation of solids and
liquids require material models as input. The material, or constitutive, model is the function that determines the force response of a class of materials (e.g., rubbers, snow, water) to
internal strains and strain rates. To produce the correct constitutive behavior for a given material, the model requires an accurate set of corresponding material parameters for every point

XPBD MPM FEM

in the simulated volume. For locally isotropic material models,

Figure 2: **Simulator** **differences**

Young’s modulus ( _E_, in the 1D linear regime, the proportion
when dropping a solid sphere with

ality constant between stress and strain), Poisson’s ratio ( _ν_,

( _E, ν, ρ_ ) = (10 [4] _Pa,_ 0 _._ 3 _,_ 10 [3] kg/m [3] )

the negative ratio of transverse to axial strain under uniaxial

with XPBD (Macklin et al., 2016) and

loading) and density ( _ρ_, unit mass per volume) are ubiquitous. MPM (Sulsky et al., 1994) vs. more acGiven an accurate and valid triplet ( _E_, _ν_, _ρ_ ) along with a rea- curate FEM.
sonable material model, a consistent numerical simulation can
produce accurate predictions of an object’s behavior under load. Measured, real-world parameters


1We describe available methods for meshes, SDFs, and NeRFs, and present a method for Splats in §6.1.


2


are portable to any consistent simulation algorithm (we use high-resolution Finite Element Methods). Further, they are portable across any material model that relies on density, Young’s modulus
and Poisson’s ratio, or derived quantities, such as shear or bulk modulus (e.g., Neo-Hookean, St.
Venant–Kirchhoff, As-Rigid-As-Possible, Co-Rotated Elastic, Mooney–Rivlin, and Ogden models).
On the other hand, many physics simulation algorithms are not implemented or applied in a consistent fashion, favoring speed over accuracy (Macklin et al., 2016; Sulsky et al., 1994). In these cases,
material parameters must be modified to avoid inaccurate behavior (Fig. 2).


2.2 INFERRING MECHANICAL PROPERTIES OF STATIC OBJECTS


Our goal is to predict volumetric mechanical properties given only shape and appearance, a challenging inverse problem, which research suggests humans learn good intuition about (Adelson, 2001;
Fleming, 2014; Fleming et al., 2013; Sharan et al., 2009). However, progress in learning-based approaches has been hampered by limited data. Existing datasets are small (Gao et al., 2022; Downs
et al., 2022; Chen et al., 2025c), contain noisy labels (Lin et al., 2018), use simulator-specific parameters (Mishra, 2024; Xie et al., 2025; Belikov et al., 2015), provide only coarse annotations (Ahmed
et al., 2025; Slim et al., 2023; Li et al., 2022) or are biased towards rigid or man-made objects (Cao
et al., 2025). Worse, data collection is difficult, relying on rigorous physical experiments (ASTM
Committee D20, 2022; ASTM Committee E28, 2024; Pai, 2000), and even then lacking spatial
material fields (Loveday et al., 2004) due to digitization and annotation challenges.


As a result, works that infer physical properties from appearance often leverage knowledge from
large pre-trained models. NeRF2Physics (Zhai et al., 2024) and PUGS (Shuai et al., 2025) optimize language-embedded feature fields for a NeRF (Mildenhall et al., 2020) or 3D Gaussians (Kerbl
et al., 2023), respectively, to predict coarse stiffness categories and density, but require per-object
optimization and are limited in their ability to predict values inside objects due to the lack of meaningful features inside NeRFs or splats. Many approaches distill signals from a Video Generation
Model and optimize physics parameters by backpropagating through fast, approximate physics simulators, resulting in a slow optimization process, yielding materials deviating from real-world values
and overfit to a specific simulation setup (Zhang et al., 2025; Huang et al., 2024b; Liu et al., 2025;
Cleac’h et al., 2023; Liu et al., 2024a; Lin et al., 2025b) (§2.1). Many methods are also tailored to a
specific 3D representation or real-time simulation implementations, such as Splats (Xie et al., 2024)
or explicit Material Point Methods (Sulsky et al., 1994; Le et al., 2025), or work with coarse material
categories (Fischer et al., 2024; Hsu et al., 2024; Lin et al., 2025a; Xia et al., 2025) that must be
manually mapped to simulation parameters. Instead, we aim to augment objects across 3D representations with fine-grained spatially-varying mechanical properties that are physically accurate and
compatible across accurate simulators. Like our method, many techniques leverage vision-language
(VLM) models. PhysGen (Liu et al., 2024b) and PhysGen3D (Chen et al., 2025a) use a VLM to infer
mass, elasticity, and friction for segmented parts of a single image. Phys4DGen (Lin et al., 2025a)
uses a VLM to annotate parts of a 3D model with coarse material labels, which are then mapped
to physical parameters, a baseline used in our evaluation. Most works above rely on aggregation of
large model outputs for every input shape, which can be brittle and time-consuming at run-time, and
can only leverage external segmentation. Instead, our method uses a VLM paired with other data
sources to annotate a _training dataset_ for a feed-forward model leveraging 3D data to annotate and
learn internal material composition.


Like our method, SOPHY (Cao & Kalogerakis, 2025), PhysX-3D (Cao et al., 2025), PhysSplat
(Zhao et al., 2024a;b) (a.k.a. SimAnything) and the concurrent Pixie (Le et al., 2025) leverage pretrained models and 3D data to annotate a _training_ dataset with physical materials. PhysSplat trains
a network to predict spatially-varying simulator-specific material offset weights for MPM by using
outputs from video distillation (Liu et al., 2024a), not focusing on material accuracy. SOPHY and
PhysX-3D are 3D generative models, designed to generate new shapes augmented with physical
attributes, and cannot augment existing assets, which is our goal. Still, we detail similar aspects of
these works. Like these works, our method uses a VLM to annotate 3D objects with Young’s Modulus, Poisson’s ratio, and density, but we do not rely on the human-in-the-loop and instead leverage
multiple data sources, not just VLM knowledge, to ensure more accurate physical properties. As a
baseline, SOPHY does implement a material decoder, but it has not been made available, and only
considers object surface, while we aim to estimate volumetric properties. Like our method, PhysX3D adopts the structural latent space of TRELLIS, but trains a joint generative model over these and
learned shape-aware physical properties latents in order to generate physics-augmented shapes from


3


DINOv2


volumetric voxelization (§4.1). A trained GeometryTransformer (§4.2) predicts per-voxel material latents,
decoded by MatVAE (§3) into mechanical properties ( _E_, _ν_, _ρ_ ).

scratch. In contrast, we treat material prediction as deterministic inference for simplicity, and further
adjust the TRELLIS pipeline to facilitate accurate material prediction inside the object. Pixie (Le
et al., 2025), a concurrent work and the only other feed-forward approach, is trained on semanticallysegmented objects and uses points from filtering NeRF densities. Thus, Pixie is trained on segments
biased toward the surface, as we show in Fig. 15, while we demonstrate being able to estimate
volumetric properties with internal structures. Furthermore, unlike Pixie, we specifically focus on
estimating physically plausible material properties, such as those measured in the real world.


3 MECHANICAL PROPERTIES LATENT SPACE


To learn a latent space of valid Young’s modulus, Poisson’s ratio, and density triplets ( _E_, _ν_, _ρ_ )
(§2.1), we propose MatVAE, a variational autoencoder (VAE) trained on a dataset of real-world
values _{mi_ := ( _Ei,_ _νi,_ _ρi_ ) _}_ (§5.1). The model’s objective is to map these triplets _m_ into a 2dimensional latent space, _z_ _∈_ R [2], from which they can be accurately reconstructed. While this
offers only minor compression (R [3] _→_ R [2] ), this latent 2D space of material properties is now easy
to visualize, sample, and interpolate within, and results in consistent distances between material
triplets with disparate units (Fig. 7,§6.4). MatVAE acts like a continuous tokenizer that allows us to
always ensure VoMP output properties that fall inside the range of some materials.


We build on VAE (Kingma & Welling, 2022), with the reconstruction component of the loss defined
as mean-squared error between the input ( _Ei, νi, ρi_ ) and reconstructed material values ( _E_ [ˆ] _i,_ ˆ _νi,_ ˆ _ρi_ ):


_L_ Recon = [1]

_N_


_N_
���(( _Ei, νi, ρi_ ) _N_ )T _−_ (( ˆ _Ei,_ ˆ _νi,_ ˆ _ρi_ ) _N_ )T��22 _[,]_ (1)

_i_ =1


where T denotes transpose and _N_ per-property normalization, where _E_ and _ρ_ are first logtransformed (log10( _E_ ), log10( _ρ_ )), then normalized to [0 _,_ 1], while _ν_ is directly normalized to [0 _,_ 1].
We find other normalization schemes without log-transform or standard _z_ -score normalization induce a heavy-tailed feature distribution, which is poorly conditioned for learning (§C).


We make several modifications over standard VAE. _First_, to capture a complex posterior beyond
a simple Gaussian, the encoder’s output is transformed by a (radial) Normalizing Flow (Rezende
& Mohamed, 2015), giving us a more flexible variational distribution _qϕ_ ( _z|m_ ) since we observe
heavy-tailed distribution for Young’s Modulus and Density while Poisson’s Ratio concentrates
near the boundaries after normalization. _Second_, we decompose the KL-divergence term of the
ELBO following (Chen et al., 2018). This allows us to directly penalize the total correlation
TC( _z_ ) = KL(¯ _qϕ_ ( _z_ ) _||_ [�] _j_ _[q]_ [¯] _[ϕ]_ [(] _[z][j]_ [))] [where] _[q]_ [¯] _[ϕ]_ [(] _[z]_ [)] [is] [the] [aggregated] [posterior,] _[z][j]_ [is] [the] _[j]_ _[∈{]_ [1] _[,]_ [ 2] _[}]_ [-th]
coordinate of the latent vector _z_ . Penalizing TC allowed us to reduce the high dependence between latent coordinates which caused MatVAE to encode density in both dimensions. _Third,_ we
observe imbalanced reconstruction, _i.e._ the latent space collapses to one property, giving us low
reconstruction errors for one property and high reconstruction error for others (§C). Thus, to ensure
the 2 latent dimensions are actively utilized, we introduce a capacity constraint ( _δ_ _× z_ dim) based


4


on (Higgins et al., 2017), resulting in the following final objective:


- �� Dimension-wise KL


_,_ (2)


_L_ MatVAE = _L_ Recon +
���     MSE


Latent Space Regularization

- �� _γ ·_ MI( _z_ ) + _β ·_ TC( _z_ )

 - ��  -  - ��  Mutual Information Total Correlation


+ _α ·_


_d_


max� _δ,_ KL( _qϕ_ ( _zj_ ) _∥_ _p_ ( _zj_ ))�

_j_ =1 - �� 


where we set _γ, β, α_ = (1 _._ 0 _,_ 2 _._ 0 _,_ 1 _._ 0), with a free nats constraint _δ_ = 0 _._ 1. See §F.1 for more details.


4 PREDICTING MECHANICAL PROPERTY FIELDS


To predict volumetric mechanical properties across 3D representation, VoMP first aggregates volumetric features for the input geometry (§4.1), which are then processed by a trained feed-forward
transformer model (§4.2) that learns in the latent space of MatVAE (§3). See §2.2.


4.1 AGGREGATING FEATURES


Our method accepts any 3D representation that can be voxelized and rendered from multiple views.
Following recent works (Wang et al., 2023; Dutt et al., 2024; Xiang et al., 2025), we compute rich
DINOv2 (Oquab et al., 2024) image features across 3D views and lift them to 3D by projecting
each voxel center into every view using the camera parameters to retrieve the corresponding image
features. The retreived image features are then averaged to obtain a feature for every voxel. A critical
difference with these prior works is that we also voxelize and process the interior of the objects and
not just their surface, which allows us to learn and predict material properties _inside_ the objects (See
§6.1 for voxelization schemes and see §F.3 for details on voxelization for training). Let’s denote all
active voxel center positions in a 3D grid of size _N_ [3] as _{_ **p** _i}_ _[L]_ _i_ =1 [where] _[L]_ [denotes] [the] [number] [of]
voxels, **p** _i_ _∈_ R [3] denotes the voxel center, and Π _j_ : R [3] _→_ [ _−_ 1 _,_ 1] [2] the camera projection for view
_j_ _∈_ _J_ where _J_ is the set of rendered views. Let the DINOv2 patch-token map be _Tj_ _∈_ R [1024] _[×][n][×][n]_

which is bilinearly sampled to get a feature map _Fj_ : [ _−_ 1 _,_ 1] [2] _→_ R [1024] . Then for each voxel
_i ∈{_ 1 _,_ 2 _, . . ., L}_, we obtain a feature **f** _i_ :
**f** _i_ = Average( _Ci_ =              - _Fj_ �Π _j_ ( **p** _i_ )��� _j_ _∈_ _J_ �) _∈_ R [1024] (3)
This propagates multi-view information to the voxels in the interior of the object, encoding useful
information that our model learns to process to predict internal material composition.


4.2 GEOMETRY TRANSFORMER


The main component of VoMP is a Transformer **F** that maps voxelized image features to our trained
material latent representation. The backbone of our model follows TRELLIS (Xiang et al., 2025)
encoder/decoder, and the backbone layers of our model are initialized with TRELLIS weights. The
encoder processes a variable-length set of active voxels, represented by their positions and features
**X** = _{_ ( **p** _i,_ **f** _i_ ) _}_ _[L]_ _i_ =1 [.] [To make this data suitable for a Transformer, we first serialize the voxel features]
into a sequence and then inject spatial awareness by adding sinusoidal positional encodings derived
from each voxel’s 3D coordinates. Similar to TRELLIS and state-of-the-art 3D Transformers, we
adopt a 3D shifted window attention mechanism (Liu et al., 2021; Yang et al., 2025). Contrary to
TRELLIS (Xiang et al., 2025), to handle assets of various sizes, we define a maximum sequence
length of _LN_ . For assets with fewer voxels _L_ _≤_ _LN_, we use the complete set. However, for larger
assets where _L_ _>_ _LN_, we use a stochastic sampling strategy, selecting a random subset of _LN_
voxels at the start of each training epoch. This dynamic resampling ensures the model is exposed to
different parts of the asset over epochs and have a larger number of "effective" max voxels.


For each training asset, we first define _S_ as the set of voxel indices to be processed in the current
iteration. The corresponding sequence of image features **X** _S_ obtained from voxel features (§4.1),
is passed to **F** . The resulting latent representation is then fed into the frozen decoder of pre-trained
MatVAE to predict material properties. The MatVAE is run _L_ times _i.e._ once per voxel, which gives
us material triplets ( _E, ν, ρ_ ) _for each voxel_ . We train this transformer with the mean squared error
between the predicted materials and the ground truth materials, averaged over all voxels in the set
_S_,


_L_ **F** = _|S|_ [1]


- _∥µθ_ ( **F** ( **X** _S_ ) _i_ ) _−_ (( _Ei, νi, ρi_ ) _[N]_ ) [T] _∥_ [2] 2 _[,]_ (4)

_i∈S_


5


where _µθ_ ( _·_ ) denotes the output of the frozen MatVAE decoder, (( _Ei, νi, ρi_ ) _[N]_ ) [T] is the ground truth
material vector for voxel _i_, and **F** ( **X** _S_ ) _i_ is the latent representation for voxel _i_ .


To transfer voxel materials back to the original representation ( _i.e._ splat means, tets for FEM simulation, quadrature points for simulation, etc.), we use nearest neighbour interpolation as outlined
in §G.1. The per-voxel latents are passed into the decoder model of MatVAE (§3), which yields
per-voxel material triplets, as shown in §2.2.


5 TRAINING DATA GENERATION


5.1 MATERIAL TRIPLETS DATASET (MTD)


To train MatVAE (§3), we collect Material Triplet Dataset(MTD), containing 100,562 triplets
( _E, ν, ρ_ ) for real-world materials. We first collect a dataset of measured material properties from
multiple online databases (MatWeb, LLC, 2025; Wikipedia contributors, 2024a;b;c; The Engineering Toolbox, 2024; Department of Engineering, University of Cambridge, 2011), containing values
obtained experimentally, typically with valid _ranges_ for all three properties for all materials. We
sample numeric triplets from each material, with the number of samples proportional to the range
size. Finally, we filter out duplicates resulting from overlapping ranges for some materials.


5.2 GEOMETRY WITH VOLUMETRIC MATERIALS (GVM) DATASET


age a pre-trained VLM, but overcome its limita- PR: 0.33 - 0.35RHO: 7700 -7900 kg/m^3

We collect high-quality 3D meshes from (NVIDIA

Figure 4: **Training Data** annotation leverages ac
Corporation, 2025a;c; NVIDIA Developer, 2025;

curate 3D data labels together with a VLM.

NVIDIA Corporation, 2025d), containing 1624 partsegmentated 3D models, with a total of 8089 parts, and treat each part as having isotropic material.
Each part contains an English material name and its own realistic PBR texture, which can be used as
additional cues to the VLM. For each part in each object, we pass the following information to the
VLM: rendering of the full object, detail rendering of the part’s visual material mapped onto a sphere
(showing visual aspects that tend to correlate with material composition), the material names, and
the ranges of three closest real-world materials in the MTD (§5.1) based on the material names (See
Fig.4, detailed prompt in Fig. 23). The vision-language model then outputs material triplets for each
part, and we map to all volumetric voxels within it, resulting in a total of 37M voxels annotated with
( _E_, _ν_, _ρ_ ). By guiding VLM with real-world material values and extra clues, we avoid inaccuracies
and implausible material values. See additional details in §6.1, §E.


YM: 110 - 120 GPa
PR: 0.33 - 0.35
RHO: 7700 -7900 kg/m^3


Figure 4: **Training Data** annotation leverages accurate 3D data labels together with a VLM.


6 EXPERIMENTS AND RESULTS


We evaluate VoMP end-to-end, showing diverse realistic simulations in §6.2. Quantitative results
are presented in §6.3, with MatVAE evaluated separately in §6.4. See video and §A for many
additional results, §B for extra comparisons with concurrent work and §C for ablations.


6.1 IMPLEMENTATION DETAILS


**Voxelization:** For voxelizing 3D Gaussian splats (Kerbl et al., 2023), we present _a_ _new voxelizer_,
that works in three phases: (1) 3D Gaussians are voxelized over a 3D grid as solid ellipsoids defined
by the 99th percentile iso-surface, (2) this set of voxels is rendered from several dozen viewpoints
sampled over a sphere to form depth maps, (3) these depthmaps are used to carve away empty space
around the exterior of the object, but leaving unseen _interior_ voxels to form a solid approximation
of the object. We then sample this solid at jittered sample points on a regular grid. We employ


6


tion, 2025b) and Blender (Blender Online Community, 2021), and for DINOv2 we use an optimized
implementation (NVIDIA, 2025). During training and testing, we set the maximum number of nonempty voxels per object _LN_ = 32768 (sampled stochastically, §4.2), and sparse data structures for
efficiency. See §F for more details. All experiments were performed on a machine with four 80GB
A100 GPUs, where training took about 12 hours for MatVAE and 5 days for the Transformer.


**Simulation:** We used FEM simulator for meshes and sparse Simplicits (Modi et al., 2024;
Fuji Tsang et al.) for our large-scale simulations combining splats and meshes. Details in §G.


6.2 END-TO-END QUALITATIVE EVALUATION


We qualitatively evaluate VoMP by using it to annotate volumetric mechanical fields for several
meshes and 3D Gaussian Splats, and running physics simulation with these exact spatially varying ( _E_, _ν_, _ρ_ ) values, resulting in realistic simulations without any hand-tweaks (Fig. 5, Fig. 8, :
0:36). We also show that our approach can work across more representations, including meshes, 3D
Gaussian Splats, SDF, and NeRFs ( Fig. 8a, with additional results in §A.2.


6.3 QUANTITATIVE EVALUATION


**Datasets** **and** **Metrics:** The 10% hold-out test set of GVM (§5) consists of 166 high-quality 3D
objects with per-voxel mechanical properties for a total of 4.9 million point annotations, significantly
larger than previous works, e.g. 31 points across 11 objects (Zhai et al., 2024). We contribute this as
_a new benchmark_ and use it for evaluation against baselines. We measure standard metrics, Average
Log Displacement Error (ALDE), Average Displacement Error (ADE), Average Log Relative Error
(ALRE), and Average Relative Error (ARE) for each mechanical property, further detailed in §D.1.
We provide additional intuition for interpreting these errors through targeted simulations in §D.4.


Table 1: **Wall-clock** comparisons and breakdown. **Baselines:** We compare against prior art

NeRF2Physics (Zhai et al., 2024) and PUGS (Shuai
et al., 2025), where we look up material proper
NeRF2Physics 1454.55 ( _±_ 1118) ties at the voxel locations (with proper scaling)
PUGS 1058.33 ( _±_ 6.94)

using their optimized representations. Note that

Pixie 201.63 ( _±_ 27.74)
Phys4DGen _[∗]_ 51.65 ( _±_ 4.07) these techniques do not output Poisson’s ratio.
Ours **3.59** ( _±_ 1.36) Phys4DGen (Lin et al., 2025a) is an important

baseline, aggregating VLM prediction directly, but

Rendering 2.11 ( _±_ 0.0540)
Voxelization 0.03 ( _±_ 0.0016) does not provide code. We used our best effort to
DINO-v2 Computation 0.86 ( _±_ 0.0020) replicate their method and used prompts provided
DINO-v2 Reconstruction 0.58 ( _±_ 0.0053) by the authors, designating this implementation
Geometry Transformer 0.0082 ( _±_ 0.0063) Phys4DGen _[⋆]_ . More baseline details in §F.5. We
MatVAE 0.00032 ( _±_ 0.00026) also include early comparisons against concurrent

(and as yet unpublished) Pixie (Le et al., 2025), with additional explorations in §B.


Table 1: **Wall-clock** comparisons and breakdown.


NeRF2Physics 1454.55 ( _±_ 1118)
PUGS 1058.33 ( _±_ 6.94)
Pixie 201.63 ( _±_ 27.74)
Phys4DGen _[∗]_ 51.65 ( _±_ 4.07)
Ours **3.59** ( _±_ 1.36)


Rendering 2.11 ( _±_ 0.0540)
Voxelization 0.03 ( _±_ 0.0016)
DINO-v2 Computation 0.86 ( _±_ 0.0020)
DINO-v2 Reconstruction 0.58 ( _±_ 0.0053)
Geometry Transformer 0.0082 ( _±_ 0.0063)
MatVAE 0.00032 ( _±_ 0.00026)


**Estimating Mechanical Properties:** Quantitative evaluation of material estimates ( _E_, _ν_, _ρ_ ) of our
method against prior art on our new detailed benchmark shows a _dramatic quality boost across all_
_properties and metrics_ (Fig. 6b). According to our explorations (§D.4), ALRE under 0 _._ 05 for _E_ and
ARE under 0 _._ 15 for other properties result in similar simulations, suggesting that our materials will


7


Table 2: **Mechanical Property Estimates** of our method on the _publicly released dataset_ are very close to the
full dataset. Per-voxel error rate is first computed per object, then averaged across all objects in the test set
to avoid weighing some objects more. Global voxel-level normalization yields similar results, see Supplement
Tb. 3.


NeRF2Physics 2.8000 ( _±_ 1.05) 0.1346 ( _±_ 0.05) - - 1432.0343 ( _±_ 964.88) 1.0365 ( _±_ 0.63)
PUGS 3.3942 ( _±_ 1.72) 0.1688 ( _±_ 0.10) - - 3568.2150 ( _±_ 2839.13) 3.2429 ( _±_ 3.56)
Phys4DGen _[⋆]_ 4.8967 ( _±_ 3.17) 0.2227 ( _±_ 0.14) 0.0407 ( _±_ 0.04) 0.1467 ( _±_ 0.18) 1865.5673 ( _±_ 2176.90) 1.4394 ( _±_ 2.35)


Ours **0.3794** **(** _±_ **0.29)** **0.0409** **(** _±_ **0.04)** **0.0241** **(** _±_ **0.01)** **0.0818** **(** _±_ **0.03)** **142.7017** **(** _±_ **166.92)** **0.0921** **(** _±_ **0.07)**


lead to more faithful simulations than competitors when using an accurate simulator. Qualitatively
(Fig. 6a), we observe that this performance difference may be due to baselines occasionally mislabeling segments (e.g. by Phys4DGen), due to noisy estimates (e.g. NeRF2Physics and PUGS), and
less accurate values in the objects’ interior due to the baselines’ design.


We are unable to make the vegetation subset of our dataset publicly available. Thus, we compute
the mechanical property estimations on the public version of the dataset in Tb. 2 and 3. We find that
our results averaged over the public dataset are highly similar to the full dataset.


**Run-Time:** To show approximate speed difference, we report average material estimate speeds
across 100 runs on objects with an average of 53.9K Gaussians for our method and the baselines in
Tb. 1. To ensure fair compute between CPU and GPU heavy methods, we ran this experiment on a
machine with only one A100 GPU and 64 CPUs. While we do not provide timing breakdown of the
other methods, this result suggests a speed up of 5-100x achieved by our method, which is not surprising given that it is the only feed-forward model among previous work. Concurrent Pixie, which
is also feed-forward, involves a heavier pre-processing step, including per-object optimization, affecting its end-to-end time. In the timing breakdown of our method, rendering and pre-processing
take the most time, and could be further optimized.


**Mass Estimation:** Following NeRF2Physics (Zhai et al., 2024) and PUGS (Shuai et al., 2025), we
also evaluate our dataset on the ABO-500 (Collins et al., 2022) object mass estimation benchmark,
following the evaluation protocol of PUGS. We run our model to estimate density _ρ_ for upto 32678
voxels per object, then average these values and multiply by the known object volume to obtain
mass. While this is only an imperfect proxy for measuring the accuracy of volumetric density _ρ_, it is
a benchmark used by prior works, and we include it for completeness. We achieve better or on-par
performance across most metrics (Fig. 6c), with qualitative results in §A.3.


**Validity:** To gauge how well different methods are at predicting realistic materials, such as those
measured in the real world, which is our goal, we leverage our MTD dataset of real materials. First,
we run all methods on GVM test set objects, and for each test voxel compute relative errors to
the nearest possible material range from MTD (error is 0 for estimates within an existing material
range). These errors are averaged across all the voxels and reported in Fig. 6d. We observe that our
method, on average, outputs much more realistic materials, as it was explicitly designed to do so.


6.4 RECONSTRUCTING AND GENERATING MATERIALS WITH MATVAE


Given no prior works exploring a latent space of material triplets ( _E_, _ν_, _ρ_ ), we evaluate MatVAE
on the MTD test set (§6.1), achieving low reconstruction errors in Fig. 7a (See Appendix D.1,
Appendix D.4) for metrics). Further, in Fig. 7 we show the desirable properties of this learned latent
space. In (a), samples throughout the 2D latent space map to real-world material ranges in MTD. In
(b), we show that ( _E_, _ν_, _ρ_ )values of real materials encoded to the latent space vary smoothly. Further,
the latent space ensures valid interpolation points between materials (c), facilitating valid assignment
from predicted voxel materials back to the original geometry. We include detailed ablations of
MatVAE design (§C), and additional latent space explorations (§A.4).


7 DISCUSSION


We introduce a representation-agnostic method that maps any 3D asset (mesh, SDF, Gaussian splat,
or voxel grid) to a volumetric field of physically valid mechanical properties ( _E, ν, ρ_ ). We show


8


Ours **0.3793** **(** _±_ **0.29)** **0.0409** **(** _±_ **0.04)** **0.0241** **(** _±_ **0.01)** **0.0818** **(** _±_ **0.03)** **142.6949** **(** _±_ **166.90)** **0.0921** **(** _±_ **0.07)**


(b) **Mechanical Property Estimates** of our method significantly outperform the baselines on all metrics. Pervoxel error rate is first computed per object, then averaged across all objects in the test set to avoid weighing
some objects more. Global voxel-level normalization yields similar results, see Supplement Tb. 4.


NeRF2Physics 0.736 12.725 1.040 0.564
PUGS 0.661 9.461 **0.767** **0.576**
Phys4DGen _[⋆]_ 0.664 9.961 0.825 0.566


Ours **0.631** **8.433** 0.887 **0.576**


(c) **Mass Estimate:** We show the errors for estimating mass of objects on the ABO-500 (Collins et al.,
2022) dataset, the only existing benchmark, approximating the accuracy of our _ρ_ estimates.


NeRF2Physics 1.62 **(** _±_ **4.96)**  - 19.75 **(** _±_ **46.60)**
PUGS 1.87 **(** _±_ **4.50)** - 13.24 **(** _±_ **12.63)**
Phys4DGen _[⋆]_ 1.77 **(** _±_ **8.53)** 0.85 **(** _±_ **3.01)** 39.49 **(** _±_ **35.47)**
Pixie 11.90 **(** _±_ **17.41)** 3.46 **(** _±_ **4.42)** 46.58 **(** _±_ **36.35)**


Ours **0.29** **(** _±_ **1.23)** **0.00** **(** _±_ **0.00)** **11.75** **(** _±_ **4.02)**


(d) **Material** **Validity:** We report mean values and
relative errors (in %) with the closest physically measured material range in MTD (§5.1).


Figure 6: **Quantitative** **Results** **and** **Comparisons:** We compare our method against prior art
NeRF2Physics (Zhai et al., 2024), PUGS (Shuai et al., 2025) and Phys4DGen (Lin et al., 2025a), and include
limited early results comparing with concurrent method Pixie (Le et al., 2025).

that our method significantly outperforms prior art in accuracy and speed, lowering the barrier for
integrating accurate physics into digital workflows across 3D representations, with potential impact
across digital twins, robotics, and beyond.


While we show important advances over existing works, our method is not without limitations,
which we hope will open exciting avenues of future research. Due to fixed-grid voxelization, our
output resolution is limited, causing oversmoothing in highly heterogeneous regions, and may result
in approximation errors when transferring results to more detailed input geometry. During annotation, we assume part-level materials are isotropic, which is not a true assumption for some common
materials like wood. Further, future work could extend our method to predict additional properties
like yield strength, shear modulus and thermal expansion, or to adapt true material properties output
by our method to simulator-specific scales required for faster algorithms or implementations. We
hope to support future directions in this area by releasing our material estimation benchmark, and
trained models.


9


log( _E_ ) ( _↓_ ) _ν_ ( _↓_ ) _ρ_ ( _↓_ ) log( _E/ρ_ ) ( _↓_ ) log( _G_ ) ( _↓_ ) log( _K_ ) ( _↓_ ) L.S. ( _↓_ ) E.A. ( _↓_ ) Bray–Curtis ( _↓_ )


0.0034 0.0426 0.0330 0.0054 0.0036 0.0036 0.0131 0.4439 0.0411


(a) **MatVAE shows excellent reconstruction errors** on the MTD test set across all metrics.


6


4


2


0


2


4


6


(d) **Interpolating in latent space** results in valid intermediate
materials, unlike naive ( _E_, _ν_, _ρ_ ) interpolation.


(b) **Decoding latent samples** leads to plausible
( _E_, _ν_, _ρ_ ) values within real-world materials.


Figure 7: **Material Latent Space** learned by MatVAE (§3) ensures faithful (a), valid (b), smoothly varying (c),
and interpolatable (d) materials. "Invalid" values (c) fall outside all material ranges in MTD (§5.1).

ACKNOWLEDGMENTS


We thank Gilles Daviet for help in setting up some of the simulations. We thank Jean-Francois
Lafleche for help with rendering. We thank Beau Perschall for help in using the datasets.


10


REFERENCES


Edward H. Adelson. On seeing stuff: the perception of materials by humans and machines. In
Bernice E. Rogowitz and Thrasyvoulos N. Pappas (eds.), _Human Vision and Electronic Imaging_
_VI_, volume 4299, pp. 1  - 12. International Society for Optics and Photonics, SPIE, 2001. doi:
10.1117/12.429489. [URL https://doi.org/10.1117/12.429489.](https://doi.org/10.1117/12.429489)


Mahmoud Ahmed, Xiang Li, Arpit Prajapati, and Mohamed Elhoseiny. 3dcompat200: Languagegrounded compositional understanding of parts and materials of 3d shapes. _arXiv_ _preprint_
_arXiv:2501.06785_, 2025.


Michael F Ashby and David Cebon. Materials selection in mechanical design. _Le_ _Journal_ _de_
_Physique IV_, 3(C7):C7–1, 1993.


ASTM Committee D20. Standard test method for tensile properties of plastics. Astm standard d638,
ASTM International, West Conshohocken, PA, 2022. Approved June 30, 2022.


ASTM Committee E28. Standard test methods for tension testing of metallic materials. Astm
standard e8/e8m, ASTM International, West Conshohocken, PA, 2024. ANSI approved.


ASTM International. Standard Test Method for Rubber Property—Durometer Hardness.
ASTM Standard D2240-15, 2015. URL [https://doi.org/10.1520/D2240-15.](https://doi.org/10.1520/D2240-15)
doi:10.1520/D2240-15.


Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang
Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. [URL https://arxiv.org/abs/2308.12966.](https://arxiv.org/abs/2308.12966)


Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan,
Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng,
Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025.
[URL https://arxiv.org/abs/2502.13923.](https://arxiv.org/abs/2502.13923)


VV Belikov, NP Vabishchevich, PN Vabishchevich, UV Katishkov, and NA Mosunova. Material
property database. _Mathematical Models and Computer Simulations_, 7:95–102, 2015.


[Jan Bender and contributors. Positionbaseddynamics: Physically-based simulation library. https:](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics)
[//github.com/InteractiveComputerGraphics/PositionBasedDynamics,](https://github.com/InteractiveComputerGraphics/PositionBasedDynamics)
2015. Commit retrieved 3 Aug 2025.


Kiran S. Bhat, Steven M. Seitz, Jovan Popovi´c, and Pradeep K. Khosla. Computing the physical
parameters of rigid-body motion from video. In Anders Heyden, Gunnar Sparr, Mads Nielsen,
and Peter Johansen (eds.), _Computer_ _Vision_ _—_ _ECCV_ _2002_, pp. 551–565, Berlin, Heidelberg,
2002. Springer Berlin Heidelberg. ISBN 978-3-540-47969-7.


Blender Online Community. _Blender - a 3D modelling and rendering package_ . Blender Foundation,
Blender Institute, Amsterdam, 2021. [URL http://www.blender.org.](http://www.blender.org)


J. Roger Bray and J. T. Curtis. An ordination of the upland forest communities of southern wisconsin. _Ecological_ _Monographs_, 27(4):325–349, 1957. doi: https://doi.org/10.2307/
1942268. URL [https://esajournals.onlinelibrary.wiley.com/doi/abs/](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1942268)
[10.2307/1942268.](https://esajournals.onlinelibrary.wiley.com/doi/abs/10.2307/1942268)


Marcus A. Brubaker, Leonid Sigal, and David J. Fleet. Estimating contact dynamics. In _2009 IEEE_
_12th_ _International_ _Conference_ _on_ _Computer_ _Vision_, pp. 2389–2396, 2009. doi: 10.1109/ICCV.
2009.5459407.


Arthur Brussee. Brush: 3d reconstruction for all. [https://github.com/ArthurBrussee/](https://github.com/ArthurBrussee/brush)
[brush, 2025.](https://github.com/ArthurBrussee/brush) GitHub repository.


Junyi Cao and Evangelos Kalogerakis. Sophy: Learning to generate simulation-ready objects with
physical materials, 2025. [URL https://arxiv.org/abs/2504.12684.](https://arxiv.org/abs/2504.12684)


11


Ziang Cao, Zhaoxi Chen, Liang Pan, and Ziwei Liu. Physx-3d: Physical-grounded 3d asset generation. _arXiv preprint arXiv:2507.12465_, 2025.


Boyuan Chen, Hanxiao Jiang, Shaowei Liu, Saurabh Gupta, Yunzhu Li, Hao Zhao, and Shenlong
Wang. Physgen3d: Crafting a miniature interactive world from a single image. In _Proceedings of_
_the Computer Vision and Pattern Recognition Conference (CVPR)_, pp. 6178–6189, June 2025a.


Chuhao Chen, Zhiyang Dou, Chen Wang, Yiming Huang, Anjun Chen, Qiao Feng, Jiatao Gu, and
Lingjie Liu. Vid2sim: Generalizable, video-based reconstruction of appearance, geometry and
physics for mesh-free simulation. _IEEE Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, 2025b.


Ricky T. Q. Chen, Xuechen Li, Roger B Grosse, and David K Duvenaud. Isolating sources of disentanglement in variational autoencoders. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 31. Curran Associates, Inc.,
2018. URL [https://proceedings.neurips.cc/paper_files/paper/2018/](https://proceedings.neurips.cc/paper_files/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf)
[file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2018/file/1ee3dfcd8a0645a25a35977997223d22-Paper.pdf)


Yunuo Chen, Tianyi Xie, Zeshun Zong, Xuan Li, Feng Gao, Yin Yang, Ying Nian Wu, and Chenfanfu Jiang. Atlas3d: Physically constrained self-supporting text-to-3d for simulation and fabrication, 2024. [URL https://arxiv.org/abs/2405.18515.](https://arxiv.org/abs/2405.18515)


Yuzhen Chen, Hojun Son, and Arpan Kusari. Matpredict: a dataset and benchmark for learning material properties of diverse indoor objects, 2025c. [URL https://arxiv.org/abs/2505.](https://arxiv.org/abs/2505.13201)
[13201.](https://arxiv.org/abs/2505.13201)


An-Chieh Cheng, Hongxu Yin, Yang Fu, Qiushan Guo, Ruihan Yang, Jan Kautz, Xiaolong Wang,
and Sifei Liu. Spatialrgpt: Grounded spatial reasoning in vision-language models. In _NeurIPS_,
2024.


Simon Le Cleac’h, Hong-Xing Yu, Michelle Guo, Taylor A. Howell, Ruohan Gao, Jiajun Wu,
Zachary Manchester, and Mac Schwager. Differentiable physics simulation of dynamicsaugmented neural objects, 2023. [URL https://arxiv.org/abs/2210.09420.](https://arxiv.org/abs/2210.09420)


Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu,
Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin,
and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. In
_Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_,
pp. 21126–21136, June 2022.


Abe Davis, Katherine L Bouman, Justin G Chen, Michael Rubinstein, Fredo Durand, and William T
Freeman. Visual vibrometry: Estimating material properties from small motion in video. In
_Proceedings of the ieee conference on computer vision and pattern recognition_, pp. 5335–5343,
2015.


Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig
Schmidt, Kiana Ehsanit, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A Universe of Annotated 3D Objects . In _2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition_
_(CVPR)_, pp. 13142–13153, Los Alamitos, CA, USA, Jun 2023. IEEE Computer Society. doi:
10.1109/CVPR52729.2023.01263. URL [https://doi.ieeecomputersociety.org/](https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.01263)
[10.1109/CVPR52729.2023.01263.](https://doi.ieeecomputersociety.org/10.1109/CVPR52729.2023.01263)


Department of Engineering, University of Cambridge. _Materials_ _Data_ _Book_ . University of
Cambridge, 2011. URL [https://teaching.eng.cam.ac.uk/sites/teaching.](https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf)
[eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%](https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf)
[20(2011)%20version%20for%20Moodle.pdf.](https://teaching.eng.cam.ac.uk/sites/teaching.eng.cam.ac.uk/files/Documents/Databooks/MATERIALS%20DATABOOK%20(2011)%20version%20for%20Moodle.pdf) Data extracted from the Cambridge
Engineering Selector (CES EduPack), courtesy of Granta Design Ltd. For educational use.


Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann,
Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of
3d scanned household items, 2022. [URL https://arxiv.org/abs/2204.11918.](https://arxiv.org/abs/2204.11918)


12


Niladri Shekhar Dutt, Sanjeev Muralikrishnan, and Niloy J. Mitra. Diffusion 3d features (diff3f):
Decorating untextured shapes with distilled semantic features. In _Proceedings of the IEEE/CVF_
_Conference on Computer Vision and Pattern Recognition (CVPR)_, pp. 4494–4504, June 2024.


Yutao Feng, Yintong Shang, Xuan Li, Tianjia Shao, Chenfanfu Jiang, and Yin Yang. Pie-nerf:
Physics-based interactive elastodynamics with nerf, 2024. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2311.13099)
[2311.13099.](https://arxiv.org/abs/2311.13099)


Michael Fischer, Iliyan Georgiev, Thibault Groueix, Vladimir G Kim, Tobias Ritschel, and
Valentin Deschaintre. Sama: Material-aware 3d selection and segmentation. _arXiv_ _preprint_
_arXiv:2411.19322_, 2024.


Roland W. Fleming. Visual perception of materials and their properties. _Vision_ _Research_, 94:62–
75, 2014. ISSN 0042-6989. doi: https://doi.org/10.1016/j.visres.2013.11.004. URL [https:](https://www.sciencedirect.com/science/article/pii/S0042698913002782)
[//www.sciencedirect.com/science/article/pii/S0042698913002782.](https://www.sciencedirect.com/science/article/pii/S0042698913002782)


Roland W. Fleming, Christiane Wiebel, and Karl Gegenfurtner. Perceptual qualities and material
classes. _Journal_ _of_ _Vision_, 13(8):9–9, 07 2013. ISSN 1534-7362. doi: 10.1167/13.8.9. URL
[https://doi.org/10.1167/13.8.9.](https://doi.org/10.1167/13.8.9)


frankaemika. franka_description: Official models of franka robotics gmbh robots. [https://](https://github.com/frankaemika/franka_description)
[github.com/frankaemika/franka_description, 2025. GitHub repository, accessed](https://github.com/frankaemika/franka_description)
June 2025.


Clement Fuji Tsang, Maria Shugrina, Jean Francois Lafleche, Or Perel, Charles Loop, Towaki
Takikawa, Vismay Modi, Alexander Zook, Jiehan Wang, Wenzheng Chen, Tianchang Shen, Jun
Gao, Krishna Murthy Jatavallabhula, Edward Smith, Artem Rozantsev, Sanja Fidler, Gavriel
State, Jason Gorski, Tommy Xiang, Jianing Li, Michael Li, and Rev Lebaredian. Kaolin: A
pytorch library for accelerating 3d deep learning research. URL [https://github.com/](https://github.com/NVIDIAGameWorks/kaolin)
[NVIDIAGameWorks/kaolin.](https://github.com/NVIDIAGameWorks/kaolin)


Ruohan Gao, Zilin Si, Yen-Yu Chang, Samuel Clarke, Jeannette Bohg, Li Fei-Fei, Wenzhen Yuan,
and Jiajun Wu. Objectfolder 2.0: A multisensory object dataset for sim2real transfer, 2022. URL
[https://arxiv.org/abs/2204.02389.](https://arxiv.org/abs/2204.02389)


Michael Grieves and John Vickers. _Digital_ _Twin:_ _Mitigating_ _Unpredictable,_ _Undesirable_ _Emer-_
_gent Behavior in Complex Systems_, pp. 85–113. Springer International Publishing, Cham, 2017.
ISBN 978-3-319-38756-7. doi: 10.1007/978-3-319-38756-7_4. URL [https://doi.org/](https://doi.org/10.1007/978-3-319-38756-7_4)
[10.1007/978-3-319-38756-7_4.](https://doi.org/10.1007/978-3-319-38756-7_4)


Minghao Guo, Bohan Wang, Pingchuan Ma, Tianyuan Zhang, Crystal Elaine Owens, Chuang
Gan, Joshua B. Tenenbaum, Kaiming He, and Wojciech Matusik. Physically compatible 3d object modeling from a single image. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), _Advances_ _in_ _Neural_ _In-_
_formation_ _Processing_ _Systems_, volume 37, pp. 119260–119282. Curran Associates, Inc.,
2024. URL [https://proceedings.neurips.cc/paper_files/paper/2024/](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7af02c8a8e26608199c087f50a21d37-Paper-Conference.pdf)
[file/d7af02c8a8e26608199c087f50a21d37-Paper-Conference.pdf.](https://proceedings.neurips.cc/paper_files/paper/2024/file/d7af02c8a8e26608199c087f50a21d37-Paper-Conference.pdf)


Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick,
Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a
constrained variational framework. In _International conference on learning representations_, 2017.


Hao-Yu Hsu, Zhi-Hao Lin, Albert Zhai, Hongchi Xia, and Shenlong Wang. Autovfx: Physically
realistic video editing from natural language instructions, 2024. [URL https://arxiv.org/](https://arxiv.org/abs/2411.02394)
[abs/2411.02394.](https://arxiv.org/abs/2411.02394)


Yuanming Hu and contributors. taichi_mpm: High-performance mls-mpm solver. [https://](https://github.com/yuanming-hu/taichi_mpm)
[github.com/yuanming-hu/taichi_mpm, 2018.](https://github.com/yuanming-hu/taichi_mpm) Commit retrieved 3 Aug 2025.


Kemeng Huang, Floyd M. Chitalu, Huancheng Lin, and Taku Komura. Gipc: Fast and stable gaussnewton optimization of ipc barrier energy. _ACM Trans. Graph._, 43(2), mar 2024a. ISSN 07300301. doi: 10.1145/3643028.


13


Kemeng Huang, Xinyu Lu, Huancheng Lin, Taku Komura, and Minchen Li. Stiffgipc: Advancing
gpu ipc for stiff affine-deformable simulation. _ACM_ _Trans._ _Graph._, 44(3), May 2025. ISSN
0730-0301. doi: 10.1145/3735126.


Tianyu Huang, Haoze Zhang, Yihan Zeng, Zhilu Zhang, Hui Li, Wangmeng Zuo, and Rynson W. H.
Lau. Dreamphysics: Learning physics-based 3d dynamics with video diffusion priors, 2024b.
[URL https://arxiv.org/abs/2406.01476.](https://arxiv.org/abs/2406.01476)


Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. _ACM_ _Transactions_ _on_ _Graphics_, 42(4), July 2023.
[URL https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)


Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2022. URL [https:](https://arxiv.org/abs/1312.6114)
[//arxiv.org/abs/1312.6114.](https://arxiv.org/abs/1312.6114)


Jochen Lang, Dinesh K Pai, and Hans-Peter Seidel. Scanning large-scale articulated deformations.
In _Graphics Interface_, pp. 265–272, 2003.


Long Le, Ryan Lucas, Chen Wang, Chuhao Chen, Dinesh Jayaraman, Eric Eaton, and Lingjie Liu.
Pixie: Fast and generalizable supervised learning of 3d physics from pixels, 2025. [URL https:](https://arxiv.org/abs/2508.17437)
[//arxiv.org/abs/2508.17437.](https://arxiv.org/abs/2508.17437)


Jinxi Li, Ziyang Song, Siyuan Zhou, and Bo Yang. Freegave: 3d physics learning from dynamic
videos by gaussian velocity. _CVPR_, 2025.


Minchen Li, Zachary Ferguson, Teseo Schneider, Timothy Langlois, Denis Zorin, Daniele Panozzo,
Chenfanfu Jiang, and Danny M. Kaufman. Incremental potential contact: intersectionand inversion-free, large-deformation dynamics. _ACM_ _Trans._ _Graph._, 39(4), August 2020a.
ISSN 0730-0301. doi: 10.1145/3386569.3392425. URL [https://doi.org/10.1145/](https://doi.org/10.1145/3386569.3392425)
[3386569.3392425.](https://doi.org/10.1145/3386569.3392425)


Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu
Jiang, and Chuang Gan. Pac-nerf: Physics augmented continuum neural radiance fields for
geometry-agnostic system identification, 2023. URL [https://arxiv.org/abs/2303.](https://arxiv.org/abs/2303.05512)
[05512.](https://arxiv.org/abs/2303.05512)


Yuchen Li, Ujjwal Upadhyay, Habib Slim, Ahmed Abdelreheem, Arpit Prajapati, Suhail Pothigara,
Peter Wonka, and Mohamed Elhoseiny. 3d compat: Composition of materials on parts of 3d
things. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Maria Farinella, and Tal
Hassner (eds.), _Computer_ _Vision_ _–_ _ECCV_ _2022_, pp. 110–127, Cham, 2022. Springer Nature
Switzerland. ISBN 978-3-031-20074-8.


Yunzhu Li, Toru Lin, Kexin Yi, Daniel Bear, Daniel Yamins, Jiajun Wu, Joshua Tenenbaum, and
Antonio Torralba. Visual grounding of learned physical models. In Hal Daumé III and Aarti
Singh (eds.), _Proceedings_ _of_ _the_ _37th_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume
119 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 5927–5936. PMLR, 13–18 Jul 2020b.
[URL https://proceedings.mlr.press/v119/li20j.html.](https://proceedings.mlr.press/v119/li20j.html)


Hubert Lin, Melinos Averkiou, Evangelos Kalogerakis, Balazs Kovacs, Siddhant Ranade, Vladimir
Kim, Siddhartha Chaudhuri, and Kavita Bala. Learning material-aware local descriptors for 3d
shapes. In _2018 International Conference on 3D Vision (3DV)_, pp. 150–159. IEEE, 2018.


Ji Lin, Hongxu Yin, Wei Ping, Yao Lu, Pavlo Molchanov, Andrew Tao, Huizi Mao, Jan Kautz,
Mohammad Shoeybi, and Song Han. Vila: On pre-training for visual language models, 2024a.
[URL https://arxiv.org/abs/2312.07533.](https://arxiv.org/abs/2312.07533)


Jiajing Lin, Zhenzhong Wang, Yongjie Hou, Yuzhou Tang, and Min Jiang. Phy124: Fast physicsdriven 4d content generation from a single image, 2024b. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2409.07179)
[2409.07179.](https://arxiv.org/abs/2409.07179)


Jiajing Lin, Zhenzhong Wang, Dejun Xu, Shu Jiang, YunPeng Gong, and Min Jiang. Phys4dgen:
Physics-compliant 4d generation with multi-material composition perception, 2025a. URL
[https://arxiv.org/abs/2411.16800.](https://arxiv.org/abs/2411.16800)


14


Yuchen Lin, Chenguo Lin, Jianjin Xu, and Yadong MU. OmniphysGS: 3d constitutive gaussians for general physics-based dynamics generation. In _The_ _Thirteenth_ _International_ _Confer-_
_ence on Learning Representations_, 2025b. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=9HZtP6I5lv)
[9HZtP6I5lv.](https://openreview.net/forum?id=9HZtP6I5lv)


Fangfu Liu, Hanyang Wang, Shunyu Yao, Shengjun Zhang, Jie Zhou, and Yueqi Duan.
Physics3d: Learning physical properties of 3d gaussians via video diffusion. _arXiv_ _preprint_
_arXiv:2406.04338_, 2024a.


Shaowei Liu, Zhongzheng Ren, Saurabh Gupta, and Shenlong Wang. Physgen: Rigid-body physicsgrounded image-to-video generation. In _European Conference on Computer Vision_, pp. 360–378.
Springer, 2024b.


Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo.
Swin transformer: Hierarchical vision transformer using shifted windows. In _Proceedings of the_
_IEEE/CVF international conference on computer vision_, pp. 10012–10022, 2021.


Zhuoman Liu, Weicai Ye, Yan Luximon, Pengfei Wan, and Di Zhang. Unleashing the potential of
multi-modal foundation models and video diffusion for 4d dynamic physical scene simulation.
In _Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)_, pp. 11016–
11025, June 2025.


John E Lloyd and Dinesh K Pai. Robotic mapping of friction and roughness for reality-based modeling. In _Proceedings_ _2001_ _ICRA._ _IEEE_ _International_ _Conference_ _on_ _Robotics_ _and_ _Automation_
_(Cat. No. 01CH37164)_, volume 2, pp. 1884–1890. IEEE, 2001.


Malcolm S Loveday, Tom Gray, and Johannes Aegerter. Tensile testing of metallic materials: A
review. _Final report of the TENSTAND project of work package_, 1, 2004.


Miles Macklin. Warp: A high-performance python framework for gpu simulation and graphics.
[https://github.com/nvidia/warp,](https://github.com/nvidia/warp) March 2022. NVIDIA GPU Technology Conference (GTC).


Miles Macklin, Matthias Müller, and Nuttapong Chentanez. Xpbd: position-based simulation
of compliant constrained dynamics. In _Proceedings_ _of_ _the_ _9th_ _International_ _Conference_ _on_
_Motion_ _in_ _Games_, MIG ’16, pp. 49–54, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450345927. doi: 10.1145/2994258.2994272. URL [https:](https://doi.org/10.1145/2994258.2994272)
[//doi.org/10.1145/2994258.2994272.](https://doi.org/10.1145/2994258.2994272)


[MatWeb, LLC. Matweb: Online materials information resource. https://www.matweb.com/,](https://www.matweb.com/)
2025. Accessed: 2025-06-25.


Mariem Mezghanni, Théo Bodrito, Malika Boulkenafed, and Maks Ovsjanikov. Physical simulation
layer for accurate 3d modeling. In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition (CVPR)_, pp. 13514–13523, June 2022.


Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In _ECCV_, 2020.


Akshansh Mishra. Latticeml: A data-driven application for predicting the effective young modulus
of high temperature graph based architected materials, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2404.09470)
[abs/2404.09470.](https://arxiv.org/abs/2404.09470)


Vismay Modi, Nicholas Sharp, Or Perel, Shinjiro Sueda, and David I. W. Levin. Simplicits: Meshfree, geometry-agnostic elastic simulation. _ACM_ _Trans._ _Graph._, 43(4), July 2024. ISSN 07300301. doi: 10.1145/3658184. [URL https://doi.org/10.1145/3658184.](https://doi.org/10.1145/3658184)


Roozbeh Mottaghi, Hessam Bagherinezhad, Mohammad Rastegari, and Ali Farhadi. Newtonian
scene understanding: Unfolding the dynamics of objects in static images. In _Proceedings of the_
_IEEE Conference on Computer Vision and Pattern Recognition (CVPR)_, June 2016.


Junfeng Ni, Yixin Chen, Bohan Jing, Nan Jiang, Bin Wang, Bo Dai, Puhao Li, Yixin Zhu, SongChun Zhu, and Siyuan Huang. Phyrecon: Physically plausible neural scene reconstruction, 2024.
[URL https://arxiv.org/abs/2404.16666.](https://arxiv.org/abs/2404.16666)


15


NVIDIA. Nvidia unveils omniverse - open, interactive 3d design collaboration platform for multi-tool workflows, 2019. URL [https://blogs.nvidia.com/blog/](https://blogs.nvidia.com/blog/omniverse-collaboration-platform/)
[omniverse-collaboration-platform/.](https://blogs.nvidia.com/blog/omniverse-collaboration-platform/) NVIDIA Blog.


NVIDIA. Nv-dinov2. NVIDIA AI Foundation Models (Model card), 2025.


NVIDIA Corporation. Commercial assets pack. [https://docs.omniverse.nvidia.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[com/usd/latest/usd_content_samples/downloadable_packs.html,](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) 2025a.
URL [https://docs.omniverse.nvidia.com/usd/latest/usd_content_](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[samples/downloadable_packs.html.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) Accessed: 2025-06-13.


NVIDIA Corporation. Nvidia omniverse replicator. Developer Documentation, 2025b.
URL [https://docs.omniverse.nvidia.com/extensions/latest/ext_](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)
[replicator.html.](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)


NVIDIA Corporation. Residential assets pack. [https://docs.omniverse.nvidia.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[com/usd/latest/usd_content_samples/downloadable_packs.html,](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) 2025c.
URL [https://docs.omniverse.nvidia.com/usd/latest/usd_content_](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[samples/downloadable_packs.html.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) Accessed: 2025-06-13.


NVIDIA Corporation. Vegetation assets pack. [https://docs.omniverse.nvidia.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[com/usd/latest/usd_content_samples/downloadable_packs.html,](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) 2025d.
URL [https://docs.omniverse.nvidia.com/usd/latest/usd_content_](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html)
[samples/downloadable_packs.html.](https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html) Accessed: 2025-06-13.


NVIDIA Developer. Simready assets. [https://developer.nvidia.com/omniverse/](https://developer.nvidia.com/omniverse/simready-assets)
[simready-assets,](https://developer.nvidia.com/omniverse/simready-assets) 2025. URL [https://developer.nvidia.com/omniverse/](https://developer.nvidia.com/omniverse/simready-assets)
[simready-assets.](https://developer.nvidia.com/omniverse/simready-assets) Accessed: 2025-06-13.


[OpenAI and Josh Achiam et al. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/](https://arxiv.org/abs/2303.08774)
[2303.08774.](https://arxiv.org/abs/2303.08774)


Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov,
Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael
Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision,
2024. [URL https://arxiv.org/abs/2304.07193.](https://arxiv.org/abs/2304.07193)


Dinesh K Pai. Robotics in reality-based modeling. In _Robotics Research:_ _the Ninth International_
_Symposium_, pp. 353–358. Springer, 2000.


Dinesh K. Pai, Kees van den Doel, Doug L. James, Jochen Lang, John E. Lloyd, Joshua L. Richmond, and Som H. Yau. Scanning physical interaction behavior of 3d objects. In _Proceed-_
_ings_ _of_ _the_ _28th_ _Annual_ _Conference_ _on_ _Computer_ _Graphics_ _and_ _Interactive_ _Techniques_, SIGGRAPH ’01, pp. 87–96, New York, NY, USA, 2001. Association for Computing Machinery.
ISBN 158113374X. doi: 10.1145/383259.383268. URL [https://doi.org/10.1145/](https://doi.org/10.1145/383259.383268)
[383259.383268.](https://doi.org/10.1145/383259.383268)


Dinesh K Pai, Jochen Lang, John Lloyd, and Robert J Woodham. Acme, a telerobotic active measurement facility. In _Experimental Robotics VI_, pp. 391–400. Springer, 2008.


Lerrel Pinto, Dhiraj Gandhi, Yuanfeng Han, Yong-Lae Park, and Abhinav Gupta. The curious robot:
Learning visual representations via physical interactions, 2016. [URL https://arxiv.org/](https://arxiv.org/abs/1604.01360)
[abs/1604.01360.](https://arxiv.org/abs/1604.01360)


PlayCanvas contributors. SuperSplat: 3d gaussian splat editor. [https://github.com/](https://github.com/playcanvas/supersplat)
[playcanvas/supersplat, 2025.](https://github.com/playcanvas/supersplat) GitHub repository.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision, 2021. URL
[https://arxiv.org/abs/2103.00020.](https://arxiv.org/abs/2103.00020)


16


Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In Francis
Bach and David Blei (eds.), _Proceedings of the 32nd International Conference on Machine Learn-_
_ing_, volume 37 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 1530–1538, Lille, France,
07–09 Jul 2015. PMLR. [URL https://proceedings.mlr.press/v37/rezende15.](https://proceedings.mlr.press/v37/rezende15.html)
[html.](https://proceedings.mlr.press/v37/rezende15.html)


Nikita Rudin, David Hoeller, Philipp Reist, and Marco Hutter. Learning to walk in minutes using
massively parallel deep reinforcement learning. In _5th_ _Annual_ _Conference_ _on_ _Robot_ _Learning_,
2021. [URL https://openreview.net/forum?id=wK2fDDJ5VcF.](https://openreview.net/forum?id=wK2fDDJ5VcF)


Katalin Schäffer, Yasemin Ozkan-Aydin, and Margaret M. Coad. Soft wrist exosuit actuated by
fabric pneumatic artificial muscles. _IEEE_ _Transactions_ _on_ _Medical_ _Robotics_ _and_ _Bionics_, 6(2):
718–732, May 2024. ISSN 2576-3202. doi: 10.1109/tmrb.2024.3385795. [URL http://dx.](http://dx.doi.org/10.1109/TMRB.2024.3385795)
[doi.org/10.1109/TMRB.2024.3385795.](http://dx.doi.org/10.1109/TMRB.2024.3385795)


Lavanya Sharan, Ruth Rosenholtz, and Edward Adelson. Material perception: What can you see in
a brief glance? _Journal of Vision_, 9(8):784–784, 2009.


Nicholas Sharp et al. Polyscope, 2019. www.polyscope.run.


Haochen Shi, Huazhe Xu, Samuel Clarke, Yunzhu Li, and Jiajun Wu. Robocook: Long-horizon
elasto-plastic object manipulation with diverse tools. _arXiv preprint arXiv:2306.14447_, 2023.


Yinghao Shuai, Ran Yu, Yuantao Chen, Zijian Jiang, Xiaowei Song, Nan Wang, Jv Zheng, Jianzhu
Ma, Meng Yang, Zhicheng Wang, Wenbo Ding, and Hao Zhao. Pugs: Zero-shot physical understanding with gaussian splatting, 2025. [URL https://arxiv.org/abs/2502.12231.](https://arxiv.org/abs/2502.12231)


Habib Slim, Xiang Li, Yuchen Li, Mahmoud Ahmed, Mohamed Ayman, Ujjwal Upadhyay, Ahmed
Abdelreheem, Arpit Prajapati, Suhail Pothigara, Peter Wonka, et al. 3dcompat++: An improved
large-scale 3d vision dataset for compositional recognition. _arXiv_ _preprint_ _arXiv:2310.18511_,
2023.


Trevor Standley, Ozan Sener, Dawn Chen, and Silvio Savarese. image2mass: Estimating the
mass of an object from its image. In Sergey Levine, Vincent Vanhoucke, and Ken Goldberg
(eds.), _Proceedings_ _of_ _the_ _1st_ _Annual_ _Conference_ _on_ _Robot_ _Learning_, volume 78 of _Proceed-_
_ings_ _of_ _Machine_ _Learning_ _Research_, pp. 324–333. PMLR, 13–15 Nov 2017. URL [https:](https://proceedings.mlr.press/v78/standley17a.html)
[//proceedings.mlr.press/v78/standley17a.html.](https://proceedings.mlr.press/v78/standley17a.html)


D. Sulsky, Z. Chen, and H.L. Schreyer. A particle method for history-dependent materials.
_Computer_ _Methods_ _in_ _Applied_ _Mechanics_ _and_ _Engineering_, 118(1):179–196, 1994. ISSN
0045-7825. doi: https://doi.org/10.1016/0045-7825(94)90112-0. URL [https://www.](https://www.sciencedirect.com/science/article/pii/0045782594901120)
[sciencedirect.com/science/article/pii/0045782594901120.](https://www.sciencedirect.com/science/article/pii/0045782594901120)


Matthew Tancik, Ethan Weber, Evonne Ng, Ruilong Li, Brent Yi, Justin Kerr, Terrance Wang,
Alexander Kristoffersen, Jake Austin, Kamyar Salahi, Abhik Ahuja, David McAllister, and
Angjoo Kanazawa. Nerfstudio: A modular framework for neural radiance field development.
In _ACM SIGGRAPH 2023 Conference Proceedings_, SIGGRAPH ’23, 2023.


The Engineering Toolbox. Engineering materials - properties. [https://www.](https://www.engineeringtoolbox.com/engineering-materials-properties-d_1225.html)
[engineeringtoolbox.com/engineering-materials-properties-d_1225.](https://www.engineeringtoolbox.com/engineering-materials-properties-d_1225.html)
[html, 2024.](https://www.engineeringtoolbox.com/engineering-materials-properties-d_1225.html) Accessed: 2025-06-25.


Yuang Wang, Xingyi He, Sida Peng, Haotong Lin, Hujun Bao, and Xiaowei Zhou. Autorecon:
Automated 3d object discovery and reconstruction. In _CVPR_, 2023.


Wikipedia contributors. Density. [https://en.wikipedia.org/wiki/Density,](https://en.wikipedia.org/wiki/Density) 2024a.
Accessed: 2025-06-25.


Wikipedia contributors. Poisson’s ratio. [https://en.wikipedia.org/wiki/Poisson%](https://en.wikipedia.org/wiki/Poisson%27s_ratio)
[27s_ratio, 2024b.](https://en.wikipedia.org/wiki/Poisson%27s_ratio) Accessed: 2025-06-25.


Wikipedia contributors. Young’s modulus. [https://en.wikipedia.org/wiki/Young%](https://en.wikipedia.org/wiki/Young%27s_modulus)
[27s_modulus, 2024c.](https://en.wikipedia.org/wiki/Young%27s_modulus) Accessed: 2025-06-25.


17


Jiajun Wu, Ilker Yildirim, Joseph J Lim, Bill Freeman, and Josh Tenenbaum. Galileo:
Perceiving physical object properties by integrating a physics engine with deep learning. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (eds.), _Ad-_
_vances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 28. Curran Associates, Inc.,
2015. URL [https://proceedings.neurips.cc/paper_files/paper/2015/](https://proceedings.neurips.cc/paper_files/paper/2015/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)
[file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2015/file/d09bf41544a3365a46c9077ebb5e35c3-Paper.pdf)


Jiajun Wu, Joseph J Lim, Hongyi Zhang, Joshua B Tenenbaum, and William T Freeman. Physics
101: Learning physical object properties from unlabeled videos. In _BMVC_, volume 2, pp. 7,
2016.


Jiajun Wu, Erika Lu, Pushmeet Kohli, Bill Freeman, and Josh Tenenbaum. Learning to see physics
via visual de-animation. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), _Advances in Neural Information Processing Systems_, volume 30.
Curran Associates, Inc., 2017. URL [https://proceedings.neurips.cc/paper_](https://proceedings.neurips.cc/paper_files/paper/2017/file/4c56ff4ce4aaf9573aa5dff913df997a-Paper.pdf)
[files/paper/2017/file/4c56ff4ce4aaf9573aa5dff913df997a-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2017/file/4c56ff4ce4aaf9573aa5dff913df997a-Paper.pdf)


Hongchi Xia, Zhi-Hao Lin, Wei-Chiu Ma, and Shenlong Wang. Video2game: Real-time interactive realistic and browser-compatible environment from a single video. In _Proceedings_ _of_ _the_
_IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_ _(CVPR)_, pp. 4578–4588,
June 2024.


Hongchi Xia, Entong Su, Marius Memmel, Arhan Jain, Raymond Yu, Numfor Mbiziwo-Tiapo, Ali
Farhadi, Abhishek Gupta, Shenlong Wang, and Wei-Chiu Ma. Drawer: Digital reconstruction
and articulation with environment realism. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition Conference (CVPR)_, pp. 21771–21782, June 2025.


Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin
Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation, 2025.
[URL https://arxiv.org/abs/2412.01506.](https://arxiv.org/abs/2412.01506)


Han Xie, Ru Jia, Yonglin Xia, Lei Li, Yue Hu, Jiaxuan Xu, Yufei Sheng, Yuanyuan Wang, and
Hua Bao. An ab initio dataset of size-dependent effective thermal conductivity for advanced
technology transistors. _arXiv preprint arXiv:2501.15736_, 2025.


Tianyi Xie, Zeshun Zong, Yuxing Qiu, Xuan Li, Yutao Feng, Yin Yang, and Chenfanfu Jiang.
Physgaussian: Physics-integrated 3d gaussians for generative dynamics. In _Proceedings_ _of_ _the_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 4389–4398, 2024.


Zhenjia Xu, Jiajun Wu, Andy Zeng, Joshua B. Tenenbaum, and Shuran Song. Densephysnet:
Learning dense physical object representations via multi-step dynamic interactions, 2019. URL
[https://arxiv.org/abs/1906.03853.](https://arxiv.org/abs/1906.03853)


Haotian Xue, Antonio Torralba, Joshua B. Tenenbaum, Daniel LK Yamins, Yunzhu Li, and HsiaoYu Tung. 3d-intphys: Towards more generalized 3d-grounded visual intuitive physics under challenging scenes, 2023. [URL https://arxiv.org/abs/2304.11470.](https://arxiv.org/abs/2304.11470)


Yandan Yang, Baoxiong Jia, Peiyuan Zhi, and Siyuan Huang. Physcene: Physically interactable
3d scene synthesis for embodied ai. In _Proceedings of the IEEE/CVF Conference on Computer_
_Vision and Pattern Recognition (CVPR)_, pp. 16262–16272, June 2024.


Yu-Qi Yang, Yu-Xiao Guo, Jian-Yu Xiong, Yang Liu, Hao Pan, Peng-Shuai Wang, Xin Tong, and
Baining Guo. Swin3d: A pretrained transformer backbone for 3d indoor scene understanding.
_Computational Visual Media_, 11(1):83–101, 2025.


Shaoxiong Yao and Kris Hauser. Estimating tactile models of heterogeneous deformable objects
in real time. In _2023_ _IEEE_ _International_ _Conference_ _on_ _Robotics_ _and_ _Automation_ _(ICRA)_, pp.
12583–12589, 2023. doi: 10.1109/ICRA48891.2023.10160731.


Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari,
Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library
for gaussian splatting, 2024. [URL https://arxiv.org/abs/2409.06765.](https://arxiv.org/abs/2409.06765)


18


Ilker Yildirim, Jiajun Wu, Yilun Du, and Joshua B. Tenenbaum. Interpreting dynamic scenes by a
physics engine and bottom-up visual cues. _arXiv preprint_, 1605., 2016. arXiv:1605.02470.


Samson Yu, Kelvin Lin, Anxing Xiao, Jiafei Duan, and Harold Soh. Octopi: Object property reasoning with large tactile-language models, 2024. URL [https://arxiv.org/abs/2405.](https://arxiv.org/abs/2405.02794)
[02794.](https://arxiv.org/abs/2405.02794)


Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Pan Lu, Zhi Huang, Carlos Guestrin,
and James Zou. Optimizing generative ai by backpropagating language model feedback. _Nature_,
639:609–616, 2025.


Albert J. Zhai, Yuan Shen, Emily Y. Chen, Gloria X. Wang, Xinlei Wang, Sheng Wang, Kaiyu
Guan, and Shenlong Wang. Physical property understanding from language-embedded feature
fields, 2024. [URL https://arxiv.org/abs/2404.04242.](https://arxiv.org/abs/2404.04242)


Kaifeng Zhang, Baoyu Li, Kris Hauser, and Yunzhu Li. Adaptigraph: Material-adaptive graph-based
neural dynamics for robotic manipulation. _arXiv preprint arXiv:2407.07889_, 2024.


Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely,
Jiajun Wu, and William T. Freeman. Physdreamer: Physics-based interaction with 3d objects via
video generation. In Aleš Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler,
and Gül Varol (eds.), _Computer Vision – ECCV 2024_, pp. 388–406, Cham, 2025. Springer Nature
Switzerland. ISBN 978-3-031-72627-9.


Haoyu Zhao, Hao Wang, Xingyue Zhao, Hao Fei, Hongqiu Wang, Chengjiang Long, and Hua Zou.
Efficient physics simulation for 3d scenes via mllm-guided gaussian splatting. _arXiv_ _preprint_
_arXiv:2411.12789_, 2024a.


Haoyu Zhao, Hao Wang, Xingyue Zhao, Hongqiu Wang, Zhiyu Wu, Chengjiang Long, and Hua Zou.
Automated 3d physical simulation of open-world scene with gaussian splatting. _arXiv_ _e-prints_,
pp. arXiv–2411, 2024b.


19


# **Supplementary Material for VoMP: Predicting** **Volumetric Mechanical Property Fields**


SUPPLEMENTARY CONTENTS


**A** **Additional Results** **22**


A.1 End-to-end Examples with Simulation . . . . . . . . . . . . . . . . . . . . . . . . 22


A.2 More Mechanical Property Prediction Results . . . . . . . . . . . . . . . . . . . . 22


A.3 Mass Estimation Example Results . . . . . . . . . . . . . . . . . . . . . . . . . . 22


A.4 Additional MatVAE Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27


**B** **Comparison with Concurrent work** **29**


**C** **Ablations** **30**


**D** **Metrics** **31**


D.1 Metrics for Mass and Field Estimation . . . . . . . . . . . . . . . . . . . . . . . . 31


D.2 Metrics to Measure Differences in Mechanical Properties . . . . . . . . . . . . . . 32


D.3 Metrics to Measure Differences in Distributions . . . . . . . . . . . . . . . . . . . 32


D.4 Interpreting Errors for Material Property Estimation . . . . . . . . . . . . . . . . . 33


**E** **Dataset Details** **34**


E.1 Annotation with Vision-Language Model . . . . . . . . . . . . . . . . . . . . . . 35


E.2 Dataset Statistics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36


**F** **Additional Implementation Details** **38**


F.1 Design of MatVAE . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 38


F.2 Network Design . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41


F.3 Training . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41


F.4 Simulation and Rendering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44


F.5 Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 44


**G** **Additional Details on the Simulations** **44**


G.1 Interpolation Scheme . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45


G.2 Preparing Scenes and Assigning Materials for the FEM Solver . . . . . . . . . . . 45


G.3 Details of the FEM Solver . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 46


G.4 Preparing Scenes and Assigning Materials for the Simplicits Solver . . . . . . . . 47


G.5 Deforming Splats and Rendering Deformed Splats . . . . . . . . . . . . . . . . . . 48


G.6 Details of the Simplicits Solver . . . . . . . . . . . . . . . . . . . . . . . . . . . . 48


G.7 Preparing Scenes and Assigning Materials for the XPBD Solver . . . . . . . . . . 49


20


G.8 Preparing Scenes and Assigning Materials for the MPM Solver . . . . . . . . . . . 49


**H** **Other Related Works** **50**


21


Table 3: **Voxel Mechanical Property Estimation.** Errors for predicting mechanical properties on the publicly
released dataset, which does not include the vegetation subset, are very close to the full dataset. These metrics
are computed by averaging across all voxels across all 3D objects in the public test set.


NeRF2Physics 2.8000 ( _±_ 1.05) 0.1346 ( _±_ 0.05) - - 1432.0343 ( _±_ 964.88) 1.0365 ( _±_ 0.63)
PUGS 3.3942 ( _±_ 1.72) 0.1688 ( _±_ 0.10) - - 3568.2150 ( _±_ 2839.13) 3.2429 ( _±_ 3.56)
Phys4DGen _[⋆]_ 4.8967 ( _±_ 3.17) 0.2227 ( _±_ 0.14) 0.0407 ( _±_ 0.04) 0.1467 ( _±_ 0.18) 1865.5673 ( _±_ 2176.90) 1.4394 ( _±_ 2.35)


Ours **0.3766** **(** _±_ **0.39)** **0.0421** **(** _±_ **0.05)** **0.0250** **(** _±_ **0.01)** **0.0837** **(** _±_ **0.03)** **113.4683** **(** _±_ **302.19)** **0.0909** **(** _±_ **0.14)**

A ADDITIONAL RESULTS


A.1 END-TO-END EXAMPLES WITH SIMULATION


We demonstrate the process that enables creating simulation-ready, realistic assets from Gaussian
Splats and meshes in Fig. 8, demonstrating convincing simulations without any fine-tuning. For
example, in Fig. 8c, we first capture a video of an object, and then train a 3D Gaussian Splatting
model. Then, we pass it to VoMP which estimates the mechanical properties in a couple of seconds.
We then use these properties in a simulator to produce a realistic (see : 0:36), greatly reducing the
barrier toward constructing realistic interactive digital worlds directly from our physical reality. We
demonstrate a Gaussian Splat scene with multiple simulated objects each of which has properties
estimated with VoMP, we then place a robot in the scene interacting with the splats in Fig. 14.
Through our experiments on Gaussian Splats we find the Gaussian Splat voxelization scheme (§6.1)
we introduce empirically qualitatively accurately voxelizes complex, noisy real-world splat objects.


A.2 MORE MECHANICAL PROPERTY PREDICTION RESULTS


We show qualitative results for inferring mechanical property fields in Fig. 9 and 10. We notice
from Fig. 9 row 1, column 2 that our model can pick up small details like the stem of the orange
at the top of the object, which is given a different Young’s modulus, though it only spans a few
voxels (see : 1:38). We notice from Fig. 9 row 2, column 2 that our model finds that the space
inside the pot should be made up of properties that fall in the range of dirt, even though the inside
of the pot was not observed through external renders (see : 1:50). We notice from Fig. 9 row 3,
column 2 that our model can tolerate some noise in assets such as the Gaussian splat of a bowl with
fruits segmented from a larger Gaussian splat (see : 2:01). We notice from Fig. 10 row 1, column
1 that our model can accurately predict thin segments and thin boundaries, like for the seat of the
chair (see : 2:10). We notice from Fig. 9 row 3, column 2 that our models can handle complex
assets and complex materials like trees and accurately handle fine details, such as understanding
where all the leaves lie and giving them different material properties than wood (see : 2:16). We
notice from Fig. 9 row 4, column 1 that our models can handle complex volumetric materials, such
as annotating the properties of wood under the flowers (see : 2:28). We notice from Fig. 9 row
4, column 2 that our models can handle thin materials and identify the vein of the leaves (see :
2:35). For completeness, we also include our metrics normalized by total test set voxels in Tb. 4 and
observe the same performance boost compared to prior art as with the per-object normalization as
presented in the main paper Fig. 6b.


We present dataset voxel-averaged results on the publicly-released dataset in Tb. 3. The publiclyreleased dataset does not include the vegetation subset.


A.3 MASS ESTIMATION EXAMPLE RESULTS


We show qualitative results from our model on mass estimation on the ABO-500 (Collins et al.,
2022) dataset in Fig. 11.


22


VoMP


VoMP


VoMP


VoMP


3D Gaussian Splat


Fields


Poisson’s Ratio


(e) **Simulating** **Gaussian** **Splats** **at** **scale:** an elastodynamic simulation of a Gaussian Splat bulldozer going
through a forest of 100 Gaussian splat ficuses in the presence of wind; all materials predicted by VoMP( :
3:00).

Figure 8: **End To End Results:** We test VoMP material estimates on a variety of input representations (a), and
show realistic simulations without any hand-tuning for meshes and splats across diverse scenarios. No handtuning of our predicted material parameters was performed, showing that VoMP directly predicts simulationready parameters. See our video for the simulation of these examples.


23


Object Young’s
Modulus
( _E_, Pa)


Object Young’s
Modulus
( _E_, Pa)


Poisson’s Density
Ratio ( _ν_ ) ( _ρ,_ _m_ _[kg]_ [3] [)]


Poisson’s Density
Ratio ( _ν_ ) ( _ρ,_ _m_ _[kg]_ [3] [)]


Figure 9: **Inferred** **Mechanical** **Property** **Fields.** We show additional mechnical property fields and slice
planes through mechanical property fields estimated by VoMP ( : 1:40).


Table 4: **Voxel Mechanical Property Estimation.** Errors for predicting mechanical properties from 3D objects
averaged across all voxels in the test set.


NeRF2Physics (Zhai et al., 2024) 2.5719 ( _±_ 1.15) 0.4122 ( _±_ 0.08) - - 1354.9458 ( _±_ 1315.71) 1.1496 ( _±_ 0.67)
PUGS (Shuai et al., 2025) 3.8619 ( _±_ 2.01) 0.4512 ( _±_ 0.11) - - 3641.0715 ( _±_ 3320.78) 4.0413 ( _±_ 4.16)
Phys4DGen _[⋆]_ (Lin et al., 2025a) 5.2977 ( _±_ 3.36) 0.4825 ( _±_ 0.14) 0.0394 ( _±_ 0.05) 0.1425 ( _±_ 0.21) 1285.9489 ( _±_ 1981.11) 1.0445 ( _±_ 2.53)


Ours **0.3765** **(** _±_ **0.39)** **0.0421** **(** _±_ **0.05)** **0.0250** **(** _±_ **0.01)** **0.0837** **(** _±_ **0.03)** **113.3807** **(** _±_ **301.90)** **0.0908** **(** _±_ **0.14)**


24


Object Young’s
Modulus
( _E_, Pa)


Object Young’s
Modulus
( _E_, Pa)


Poisson’s Density
Ratio ( _ν_ ) ( _ρ,_ _m_ _[kg]_ [3] [)]


Poisson’s Density
Ratio ( _ν_ ) ( _ρ,_ _m_ _[kg]_ [3] [)]


Figure 10: **Inferred** **Mechanical** **Property** **Fields.** We show additional mechnical property fields and slice
planes through mechanical property fields estimated by VoMP ( : 2:10).


25


Predicted: 1.58 kg Predicted: 0.01 kg Predicted: 0.02 kg Predicted: 0.71 kg
Ground Truth: 1.58 kg Ground Truth: 0.01 kg Ground Truth: 0.01 kg Ground Truth: 0.70 kg


Predicted: 9.09 kg Predicted: 0.37 kg Predicted: 7.52 kg Predicted: 0.14 kg
Ground Truth: 9.07 kg Ground Truth: 0.34 kg Ground Truth: 7.54 kg Ground Truth: 0.10 kg


Predicted: 0.38 kg Predicted: 0.25 kg Predicted: 0.05 kg Predicted: 0.19 kg
Ground Truth: 0.32 kg Ground Truth: 0.19 kg Ground Truth: 0.11 kg Ground Truth: 0.12 kg


Predicted: 7.68 kg Predicted: 4.56 kg Predicted: 1.23 kg Predicted: 0.80 kg
Ground Truth: 7.60 kg Ground Truth: 4.65 kg Ground Truth: 1.12 kg Ground Truth: 0.91 kg


Predicted: 4.47 kg Predicted: 0.42 kg Predicted: 2.40 kg Predicted: 2.85 kg
Ground Truth: 4.36 kg Ground Truth: 0.29 kg Ground Truth: 2.27 kg Ground Truth: 2.72 kg

Figure 11: **Mass Estimation.** We show qualitative results of estimating mass from the ABO-500 (Collins et al.,
2022) dataset.
26


Table 5: **Distribution learned by MatVAE** compared to the distribution of MTD test set.


0.0405 0.0798 0.1379 0.0317 0.0437 0.0342 0.0132 0.0172 0.0260


A.4 ADDITIONAL MATVAE RESULTS


A.4.1 DISTRIBUTION LEARNED BY MATVAE


We measure standard metrics used for measuring the difference in the distribution learned by MatVAE and the distribution of the test set in Tb. 5, observing small errors which suggest that MatVAE
learned a good approximation of the true material distribution.


A.4.2 MOVING ACROSS MATVAE LATENT SPACE


We show an example of moving across the latent space in Fig. 13. For each setting, we take a point in
our latent space and move across both of its dimensions to obtain multiple smoothly varying material
properties. We apply these properties to a bunny and simulate dropping it to the ground with a FEM
simulator and these various material properties. The color plots show the average displacements of
the mesh from its rest state across simulation steps (e.g. Fig. 13b), demonstrating that even the actual
physical behavior correlates with the dimensions of the latent space. Thus, we find that MatVAE
learns a rich, meaningful latent space with smooth interpolation and ensures generating physically
valid material triplets.


A.4.3 INTERPOLATION WITH MATVAE


We show additional examples of interpolating in the MatVAE latent space (§6.4) in Fig. 12.


Figure 12: **Interpolating in MatVAE latent space:** an additional example of interpolation, complementary to
Fig. 7d.


27


Setting Position Material Young’s Modulus (Pa) Poisson’s Ratio Density (kg/m [3] )


Interpolated True Range Interpolated True Range Interpolated True Range


Fig. 13b Top-left ( _↖_ ) Aerographite 4 _._ 4 _×_ 10 [5] 1 _._ 0 _×_ 10 [5]   - 1 _._ 0 _×_ 10 [6] 0.241 0.2–0.3 0.2 0.2–0.2
Top-right ( _↗_ ) Polyurethane Foam 4 _._ 8 _×_ 10 [6] 1 _._ 0 _×_ 10 [5]        - 5 _._ 0 _×_ 10 [6] 0.304 0.30–0.30 298.2.0 50–300
Bottom-left ( _↙_ ) Rubber (soft) 3 _._ 1 _×_ 10 [6] 3 _._ 0 _×_ 10 [6]        - 5 _._ 0 _×_ 10 [6] 0.488 0.48–0.50 952.0 950–950
Bottom-right ( _↘_ ) Styrofoam 1 _._ 6 _×_ 10 [6] 1 _._ 0 _×_ 10 [6]        - 3 _._ 0 _×_ 10 [6] 0.322 0.3–0.35 22.6 15–35


Fig. 13c Top-left ( _↖_ ) Aerogel 4 _._ 4 _×_ 10 [6] 1 _._ 0 _×_ 10 [6]   - 1 _._ 0 _×_ 10 [7] 0.257 0.2–0.3 1.0 1.0–1.0
Top-right ( _↗_ ) Neoprene 1 _._ 0 _×_ 10 [7] 1 _._ 0 _×_ 10 [6]        - 1 _._ 0 _×_ 10 [7] 0.494 0.45–0.5 1232.0 1230–1250
Bottom-left ( _↙_ ) EPDM Rubber 6 _._ 6 _×_ 10 [6] 5 _._ 0 _×_ 10 [6]        - 1 _._ 0 _×_ 10 [7] 0.488 0.49–0.49 1100.9 1100–1100
Bottom-right ( _↘_ ) Flexible PVC (Plasticized) 4 _._ 8 _×_ 10 [7] 2 _._ 0 _×_ 10 [7]        - 1 _._ 0 _×_ 10 [8] 0.450 0.45–0.45 1209.5 1200–1400


Fig. 13d Top-left ( _↖_ ) Polystyrene Foam (EPS) 2 _._ 6 _×_ 10 [6] 1 _._ 0 _×_ 10 [6]   - 5 _._ 0 _×_ 10 [6] 0.104 0.10–0.10 59.1 30–100
Top-right ( _↗_ ) Chloroprene Rubber (Neoprene) 5 _._ 0 _×_ 10 [6] 5 _._ 0 _×_ 10 [6]        - 5 _._ 0 _×_ 10 [6] 0.490 0.49–0.49 1200.8 1200–1200
Bottom-left ( _↙_ ) Polystyrene (Foam) 5 _._ 8 _×_ 10 [6] 2 _._ 5 _×_ 10 [6]        - 7 _._ 0 _×_ 10 [6] 0.371 0.34–0.4 34.8 15–35
Bottom-right ( _↘_ ) Polybutylene (PB) 2 _._ 5 _×_ 10 [8] 2 _._ 5 _×_ 10 [8]        - 3 _._ 0 _×_ 10 [8] 0.400 0.4–0.42 932.0 930–950

(a) Corner Materials for our experiments on moving across the latent space (Fig. 13b to 13d).


(b) Setting 1. (c) Setting 2. (d) Setting 3.


Figure 13: **Moving Across MatVAE latent space.** We sample 3 different valid mechanical property triplets ( _E_,
_ν_, _ρ_ ) (Setting 1,2,3), corresponding to the middle square in the three color diagrams. We encode each of these
triplets with MatVAE, and then traverse the 2D latent space to build a 5 _×_ 5 grid of latents around the starting
value, which are each decoded to actual mechanical properties. To visualize if latent space dimensions correlate
with actual simulation performance, we apply each meachnical property triplet to a dropping bunny simulation
and measure its mean displacement from rest, which is color coded in the graphs below. Each diagram (b, c, d)
thus corresponds to 25 simulation runs with different parameters. We observe a clear correlation between latent
dimensions and simulation behavior. ( : 4:40)


(b) Setting 1.


(c) Setting 2.


time


Young’s Modulus Density Poisson’s Ratio


Voxelized Splats

Figure 14: **Simulating** **a** **Large** **Gaussian** **Splat** **Scene.** We demonstrate an elastodynamic simulation of a
large Gaussian Splat scene with multiple objects segmented out and being assigned properties with VoMP.


28


Figure 15: **Data Annotation.** We compare data annotation process of VoMP and Pixie.


Table 6: **Comparison of Mapped Material Properties** between Pixie’s in-context physics examples (Le et al.,
2025) and the datasets of known material properties.


Item Mapped Materials Pixie _ρ_ Dataset _ρ_ Pixie _E_ Dataset _E_ Pixie _ν_ Dataset _ν_


Clay Brick [1900, 1900] [2.000e+09, 6.000e+09] [0.20, 0.20]
tree/pot Porcelain (Ceramic) [400, 400] [2400, 2400] [2.000e+08, 2.000e+08] [7.000e+10, 7.000e+10] [0.40, 0.40] [0.20, 0.20]
Glass Ceramic [2400, 2600] [9.000e+10, 1.100e+11] [0.24, 0.25]


Wood [700, 700] [8.000e+09, 1.100e+10] [0.30, 0.50]
tree/trunk Oak (White) [400, 400] [770, 800] [2.000e+06, 2.000e+06] [1.200e+10, 1.500e+10] [0.40, 0.40] [0.30, 0.40]
Maple Wood (Sugar) [630, 690] [1.000e+10, 1.300e+10] [0.30, 0.40]


tree/leaves  - [200, 200]  - [2.000e+04, 2.000e+04]  - [0.40, 0.40]  

Glass (Soda-Lime) [2500, 2500] [7.200e+10, 7.400e+10] [0.23, 0.23]
flowers/vase [500, 500] [1.000e+06, 1.000e+06] [0.30, 0.30]
Glass (Borosilicate) [2300, 2300] [6.200e+10, 8.100e+10] [0.20, 0.20]


flowers/flowers - [100, 100] - [1.000e+04, 1.000e+04] - [0.40, 0.40] 

shrub/stems Wood [300, 300] [700, 700] [1.000e+05, 1.000e+05] [8.000e+09, 1.100e+10] [0.35, 0.35] [0.30, 0.50]


shrub/twigs Wood [250, 250] [700, 700] [6.000e+04, 6.000e+04] [8.000e+09, 1.100e+10] [0.38, 0.38] [0.30, 0.50]


shrub/foliage - [150, 150] - [2.000e+04, 2.000e+04] - [0.40, 0.40] 

Rubber (soft) [950, 950] [3.000e+06, 5.000e+06] [0.48, 0.50]
grass/blades EPDM Rubber [80, 80] [1100, 1100] [1.000e+04, 1.000e+04] [1.000e+07, 1.000e+07] [0.45, 0.45] [0.49, 0.49]
Neoprene [1230, 1250] [1.000e+06, 1.000e+07] [0.45, 0.50]


soil (if visible) Sandy Loam [1200, 1200] [1600, 1800] [5.000e+05, 5.000e+05] [1.000e+08, 5.000e+08] [0.30, 0.30] [0.31, 0.31]


Rubber (soft)


[950, 950]


[3.000e+06, 5.000e+06]


rubber_ducks_and_toys/toy


Rubber (soft) [950, 950] [3.000e+06, 5.000e+06] [0.48, 0.50]

EPDM Rubber [1100, 1100] [1.000e+07, 1.000e+07] [0.49, 0.49]

[80, 150] [3.000e+04, 5.000e+04] [0.40, 0.45]

Neoprene [1230, 1250] [1.000e+06, 1.000e+07] [0.45, 0.50]
Flexible PVC (Plasticized) [1200, 1400] [2.000e+07, 1.000e+08] [0.45, 0.45]


[80, 150]


[3.000e+04, 5.000e+04]


[0.40, 0.45]


Rubber (soft) [950, 950] [3.000e+06, 5.000e+06] [0.48, 0.50]
sport_balls/ball EPDM Rubber [80, 150] [1100, 1100] [3.000e+04, 5.000e+04] [1.000e+07, 1.000e+07] [0.40, 0.45] [0.49, 0.49]
Neoprene [1230, 1250] [1.000e+06, 1.000e+07] [0.45, 0.50]


Aluminium [2700, 2700] [7.000e+10, 7.000e+10] [0.35, 0.35]
soda_cans/can Aluminum 2024-T3 [2600, 2800] [2780, 2780] [5.000e+10, 8.000e+10] [7.240e+10, 7.240e+10] [0.25, 0.35] [0.33, 0.33]
Aluminum 7075-T6 [2810, 2810] [7.100e+10, 7.100e+10] [0.33, 0.33]


Steel [7700, 7700] [2.000e+11, 2.000e+11] [0.31, 0.31]
metal_crates/crate Stainless Steel 17-7PH [2500, 2900] [7800, 7800] [8.000e+07, 1.200e+08] [2.040e+11, 2.040e+11] [0.25, 0.35] [0.30, 0.30]
Stainless Steel 440A [7800, 7800] [2.000e+11, 2.000e+11] [0.30, 0.30]


sand/sand Sandy Loam [1800, 2200] [1600, 1800] [4.000e+07, 6.000e+07] [1.000e+08, 5.000e+08] [0.25, 0.35] [0.31, 0.31]


jello_block/jello  - [40, 60]  - [8.000e+02, 1.200e+03]  - [0.25, 0.35]  

snow_and_mud/snow_and_mud Sandy Loam [2000, 3000] [1600, 1800] [8.000e+04, 1.200e+05] [1.000e+08, 5.000e+08] [0.15, 0.25] [0.31, 0.31]

B COMPARISON WITH CONCURRENT WORK


Although Pixie (Le et al., 2025) is concurrent with us, we still compare aspects of our approach with
Pixie. We discuss differences with Pixie in §2.2.


**Data Annotation Process.** We compare our data annotation process with the annotation process
of Pixie (Le et al., 2025) in Fig. 15. Our method performs annotation from meshes while Pixie (Le
et al., 2025) gets points from training a NeRF, which often produces noisy points, and the segmentation is performed based on CLIP features, which often produces noisy segmentation for difficult
objects.


**Validity** **of** **Materials** **for** **Data** **Annotation.** Although we do not have access to the Pixie (Le
et al., 2025) dataset, Pixie uses in-context physics examples, which include material names and
ranges of mechanical property triplets in the annotation process. We analyze these in-context physics
examples and compare them with real material ranges from MTD (§5.1) in Tb. 6. We find that some
of these in-context properties might create pleasing simulations with a particular simulator but can
fall outside the range of real materials.


29


Modulus


VoMP


Pixie


0.37


1.4e3


Ratio


0.36


1e3


C ABLATIONS


We provide an in-depth analysis motivating our MatVAE and Geometry Transformer training
scheme by ablating each component in Tb. 7 and 8. Our ablations require changing the hyperparameters for fair comparisons; thus, for each ablation, we tune our hyperparameters within an
identical compute budget.


**MatVAE vs Vanilla VAE.** Technically, MatVAE (§3) is built on top of the vanilla VAE (Kingma
& Welling, 2022) and we can use a vanilla VAE (Kingma & Welling, 2022) in its place. In Tb. 7,
we show material property reconstruction and distributional metrics between a vanilla VAE and
our MatVAE. MatVAE outperforms the vanilla VAE in almost all metrics. We find that Vanilla
VAE collapses to the Young’s Modulus property, giving us a low reconstruction error for Young’s
Modulus but significantly higher errors for other properties.


**Image** **Features.** For aggregating image features (§4.1), we experiment with using DINOv2 (Oquab et al., 2024), CLIP (Radford et al., 2021), and RGB colors by average pooling in
the voxel. The results are shown in Tb. 8. Our model had many layers in the Geometry Transformer
(§4.2 and appendix F.2) initialized from a generation model (Xiang et al., 2025). This set of ablations was trained starting from random weights, due to the absence of generation weights for these
settings. We find that using DINOv2 (Oquab et al., 2024) and CLIP (Radford et al., 2021) without
initializing the weights from TRELLIS (Xiang et al., 2025) performs slightly worse, whereas simply
using RGB colors performs significantly worse.


**MatVAE.** While MatVAE acts as a continuous tokenizer, it is possible to have the Geometry Transformer directly predict a R [3] vector _i.e._ directly predict the material triplets. We find this produces
significantly worse results for Young’s Modulus estimation and Poisson’s Ratio estimation.


**Normalization Scheme.** We experiment with different normalization schemes (§3) like _Z_ -score,
and not using log-space transform for either Young’s Modulus or Density. All of these normalization
schemes lead to a significant degradation in performance. Most notably, removing the logarithmic
scaling for Young’s Modulus (w/o log( _E_ )) or using a simple _Z_ -score severely harms prediction
accuracy.


**Loss.** Our Geometry Transformer (§4.2) is trained with _ℓ_ 2 loss for reconstruction (Equation (4)).
We test the effect of replacing this with an _ℓ_ 1 loss. This change results in a substantial drop in
performance across all metrics, with errors increasing by a factor of 2-3x for most properties. This
indicates that the squared error penalty of the _ℓ_ 2 loss is more effective for this material property
regression task.


30


D METRICS


We present an explanation of the metrics we use, and experiments on interpreting these metrics.


D.1 METRICS FOR MASS AND FIELD ESTIMATION


To evaluate the accuracy of predicted scalar quantities such as object mass, as well as continuous
scalar fields like density or stiffness, we use several commonly adopted metrics. Let _y_ denote a
ground-truth scalar value or voxel-wise field (e.g., density), and _y_ ˆ its predicted counterpart.


**Absolute** **Difference** **Error** **(ADE).** The average absolute error between predicted and groundtruth values:


ADE = [1]

_N_


_N_

- _|yi −_ _y_ ˆ _i|._ (5)


_i_ =1


This metric is scale-sensitive and reports the error in physical units (e.g., kg _/_ m [3] for density, kg for
mass).


**Absolute Log Difference Error (ALDE).** The average absolute error in logarithmic space:


ALDE = [1]

_N_


_N_

- _|_ log _yi −_ log ˆ _yi|._ (6)


_i_ =1


This metric captures multiplicative error and is particularly useful for quantities that vary over several orders of magnitude.


**Average** **Relative** **Error** **(ARE).** The mean relative deviation between predictions and ground
truth:


_._ (7)
����


ARE = [1]

_N_


_N_


_i_ =1


_yi −_ _y_ ˆ _i_
���� _yi_


This dimensionless metric penalizes over- and under-estimates proportionally, making it appropriate
for comparing across varying scales.


**Minimum Ratio Error (MnRE).** A symmetric and bounded measure of relative accuracy:


MnRE = [1]

_N_


_N_


- min - _yi_ _,_ _[y]_ [ˆ] _[i]_

_y_ ˆ _i_ _yi_
_i_ =1


_yi_


_._ (8)


This metric ranges from 0 to 1 and is maximized when predictions are perfectly accurate. As suggested in prior work (Standley et al., 2017), MnRE avoids bias toward systematic over- or underestimation and reduces sensitivity to outliers, making it particularly effective for evaluating physical
quantity predictions across heterogeneous samples.
Table 7: **MatVAE Ablation.** We ablate MatVAE by comparing it against a vanilla VAE and present reconstruction and distirbutional metrics.


Model Young’s Modulus ( _E_ ) Poisson’s Ratio ( _ν_ ) Density ( _ρ_ )


W1 ( _↓_ ) W2 ( _↓_ ) _D_ KL ( _↓_ ) W1 ( _↓_ ) W2 ( _↓_ ) _D_ KL ( _↓_ ) W1 ( _↓_ ) W2 ( _↓_ ) _D_ KL ( _↓_ )


Vanilla VAE (Kingma & Welling, 2022) 0.0653 0.0868 **0.0547** 0.0849 0.1057 0.0689 0.0547 0.0744 **0.0175**
MatVAE **0.0405** **0.0798** 0.1379 **0.0317** **0.0437** **0.0342** **0.0132** **0.0172** 0.0260
**(-0.025)** **(-0.007)** **(+0.083)** **(-0.053)** **(-0.062)** **(-0.035)** **(-0.042)** **(-0.057)** **(+0.009)**
w/o NF 0.0339 0.0447 0.0441 0.0417 0.0504 0.0848 0.0599 0.0819 0.0529
w/o TC penalty 0.0633 0.0855 0.0500 0.0844 0.1052 0.0672 0.0525 0.0715 0.0109
w/o free nats 0.0749 0.1168 0.1311 0.2014 0.2064 0.6376 0.0421 0.0507 0.0223


log( _E_ ) ( _↓_ ) _ν_ ( _↓_ ) _ρ_ ( _↓_ ) log( _E/ρ_ ) ( _↓_ ) log( _G_ ) ( _↓_ ) log( _K_ ) ( _↓_ ) L.S. ( _↓_ ) E.A. ( _↓_ ) Bray–Curtis ( _↓_ )


Vanilla VAE (Kingma & Welling, 2022) 0.0512 15366.8750 0.8306 0.0542 0.0544 0.0447 0.2384 2.1893 0.4690
MatVAE **0.0034** **0.0426** **0.0330** **0.0054** **0.0036** **0.0036** **0.0131** **0.4439** **0.0411**
**(-0.048)** **(-15366.8)** **(-0.798)** **(-0.049)** **(-0.051)** **(-0.041)** **(-0.225)** **(-1.745)** **(-0.428)**
w/o NF 0.0020 nan 0.0160 0.0033 0.0021 0.0021 0.0086 0.1729 0.0234
w/o TC penalty 0.0499 15567.0986 0.8298 0.0537 0.0530 0.0437 0.2382 2.1514 0.4562
w/o free nats 0.0036 16332.7829 0.0276 0.0053 0.0037 0.0041 0.0116 0.2966 0.0436


31


Table 8: **Ablations.** We ablate VoMP with choice of image features, using MatVAE, normalization scheme,
and choice of a loss function. We report the voxel-level mechanical property difference errors.


w/ DINOv2 (Oquab et al., 2024) **0.2888** **(** _±_ **0.41)** 0.0536 **(** _±_ **0.06)** 0.0259 **(** _±_ **0.02)** 0.0803 **(** _±_ **0.08)** 373.5183 **(** _±_ **675.90)** 0.3126 **(** _±_ **0.79)**
w/ CLIP (Radford et al., 2021) 0.2695 **(** _±_ **0.42)** 0.0508 **(** _±_ **0.06)** 0.0250 **(** _±_ **0.02)** 0.0771 **(** _±_ **0.07)** 383.5844 **(** _±_ **766.41)** 0.3110 **(** _±_ **0.85)**
w/ RGB colors 1.2176 **(** _±_ **0.88)** 0.6593 **(** _±_ **0.49)** 0.1379 **(** _±_ **0.06)** 1.1642 **(** _±_ **0.78)** 3678.4451 **(** _±_ **8421.17)** 4.7430 **(** _±_ **2.85)**


MatVAE (§3)
w/o MatVAE 1.1284 **(** _±_ **0.52)** 0.1289 **(** _±_ **0.08)** 0.0480 **(** _±_ **0.02)** 0.1638 **(** _±_ **0.08)** 917.5879 **(** _±_ **428.50)** 0.8637 **(** _±_ **0.61)**


Normalization Scheme (§3)
w/ _Z_ -score 0.8838 **(** _±_ **0.61)** 0.0996 **(** _±_ **0.08)** 0.0814 **(** _±_ **0.04)** 0.2938 **(** _±_ **0.20)** 5269.2900 **(** _±_ **946.03)** 6.4656 **(** _±_ **4.00)**
w/o log( _ρ_ ) 0.6654 **(** _±_ **0.54)** 0.0748 **(** _±_ **0.07)** 0.0542 **(** _±_ **0.03)** 0.1806 **(** _±_ **0.11)** 549.9512 **(** _±_ **513.57)** 0.5976 **(** _±_ **0.39)**
w/o log( _E_ ) 0.9033 **(** _±_ **0.60)** 0.1024 **(** _±_ **0.09)** 0.1182 **(** _±_ **0.05)** 0.4189 **(** _±_ **0.25)** 4051.9121 **(** _±_ **838.98)** 5.0428 **(** _±_ **3.22)**


Loss
w/ _ℓ_ 1 0.8947 **(** _±_ **0.62)** 0.1038 **(** _±_ **0.09)** 0.0468 **(** _±_ **0.04)** 0.1666 **(** _±_ **0.16)** 568.7543 **(** _±_ **734.10)** 0.6337 **(** _±_ **0.88)**


Ours 0.3765 **(** _±_ **0.39)** **0.0421** **(** _±_ **0.05)** **0.0250** **(** _±_ **0.01)** **0.0837** **(** _±_ **0.03)** **113.3807** **(** _±_ **301.90)** **0.0908** **(** _±_ **0.14)**


D.2 METRICS TO MEASURE DIFFERENCES IN MECHANICAL PROPERTIES


We use multiple commonly used metrics for measuring differences between mechanical properties.


**Relative** **Error** **in** log( _E_ ) **.** Relative error between predicted and true values of the logarithm of
Young’s modulus _E_ reported in units of Pa. This captures relative error in material stiffness across
several orders of magnitude.


**Relative Error in** _ν_ **.** Relative Error in linear space for Poisson’s ratio _ν_, a dimensionless measure
of lateral contraction under uniaxial loading.


**Relative** **Error** **in** _ρ_ **.** Relative Error between predicted and true values of material density _ρ_, reported in units of kg _/_ m [3] .


**Relative** **Error** **in** log( _E/ρ_ ) **.** Relative Error in the logarithm of specific modulus, where _E_ is
Young’s modulus and _ρ_ is density. Reflects relative deviation in stiffness-to-weight efficiency.


**Relative Error in** log( _G_ ) **.** Relative Error in the logarithm of shear modulus _G_ = 2(1+ _E_ _ν_ ) [,] [repre-]
senting resistance to shear deformation.


**Relative Error in** log( _K_ ) **.** Relative Error in the logarithm of bulk modulus _K_ = 3(1 _−E_ 2 _ν_ ) [, charac-]
terizing resistance to uniform volumetric compression.


**Lightweight** **Stiffness** **Ashby** **Index** **(** _P_ = _E_ [1] _[/]_ [2] _/ρ_ **).** The Relative Error in log( _P_ ), where _P_ =
_E_ [1] _[/]_ [2] _/ρ_, reflecting relative error in predicting material efficiency for maximizing stiffness per unit
weight (Ashby & Cebon, 1993).


**Energy** **Absorption** **Ashby** **Index** **(** _P_ = _E_ [1] _[/]_ [3] _/ρ_ **).** The Relative Error in log( _P_ ), where _P_ =
_E_ [1] _[/]_ [3] _/ρ_, quantifying relative deviation in predicted energy absorption efficiency (Ashby & Cebon,
1993).


**Bray–Curtis** **dissimilarity.** Bray–Curtis dissimilarity (Bray & Curtis, 1957) between predicted
and ground-truth property vectors **x** and **y** :

          _i_ _[|][x][i][ −]_ _[y][i][|]_
BC( **x** _,_ **y** ) = (9)

          _i_ [(] _[x][i]_ [ +] _[ y][i]_ [)] _[.]_


A normalized, dimensionless measure in [0 _,_ 1] capturing overall distributional divergence across
multiple material properties.


D.3 METRICS TO MEASURE DIFFERENCES IN DISTRIBUTIONS


We use multiple commonly-used metrics for measuring differences between distributions.


32


time

Figure 17: **Simulations** **to** **Interpret** **Errors.** We demonstrate simulations performed to show the relation
between relative error and simulation error.

**Wasserstein–1 Distance (** _W_ 1 **).** For probability measures _µ, ν_ on a space _X_,

                  


_W_ 1( _µ, ν_ ) = inf
_γ∈_ Π( _µ,ν_ )


_∥x −_ _y∥_ _dγ_ ( _x, y_ ) _,_ (10)
_X×X_


where Π( _µ, ν_ ) is the set of all couplings of _µ_ and _ν_ . _W_ 1 equals the minimum average “work” to
move mass from _µ_ to _ν_ .


**Wasserstein–2 Distance (** _W_ 2 **).** For probability measures _µ, ν_ on a space _X_,


     _W_ 2( _µ, ν_ ) = inf
_γ∈_ Π( _µ,ν_ )


- �1 _/_ 2

_∥x −_ _y∥_ [2] _dγ_ ( _x, y_ ) _,_ (11)
_X×X_


where Π( _µ, ν_ ) is the set of all couplings of _µ_ and _ν_ . The root-mean-square transport cost between
_µ_ and _ν_ .


**Kullback–Leibler Divergence (** _D_ KL **).** For densities _p, q_ on _X_,


     _D_ KL( _p∥q_ ) =


(12)
_q_ ( _x_ ) _[dx.]_


_p_ ( _x_ ) log _[p]_ [(] _[x]_ [)]
_X_ _q_ ( _x_ )


_D_ KL measures the expected extra log-likelihood of data drawn from _p_ when it is coded using _q_
instead of _p_ .


D.4 INTERPRETING ERRORS FOR MATERIAL PROPERTY ESTIMATION


We experimentally demonstrate an interpretation of how relative changes in material properties affect simulations of the finite element method (FEM) solver. We do so by simulating the deformation
of unit cubes under many different material properties and scenarios with an FEM solver.


For each baseline triplet ( _E_ 0 _, ν_ 0 _, ρ_ 0) representing Young’s modulus, Poisson’s ratio, and density,
we introduce variations following the scaling laws: density variations follow linear scaling _ρ_ new =
_ρ_ 0(1 + ∆), Poisson’s ratio variations use the same linear relationship _ν_ new = _ν_ 0(1 + ∆), while
Young’s modulus variations use exponential scaling _E_ new = _E_ 0 _e_ [∆] to accommodate the wide range
of stiffness values. We then apply every such unique material triplet to a unit cube and perform a
simulation under some external forces.


During each of these simulations, we measure the final volume and potential energy of the cube after
the Newton iterations have converged.


**Measuring Volume.** For a body undergoing deformation, the deformation gradient **F** = _∇_ **u** + I
maps material points from the reference to the current configuration, where **u** represents the displacement field and I is the 3 _×_ 3 identity tensor. The local volume change is quantified by the
Jacobian _J_ = det( **F** ), which represents the ratio of deformed to reference volume at each material
point. The total deformed volume is: _V_ def = �Ω0 _[J dV]_ [, where][ Ω][0] [denotes the reference configura-]

tion. The relative volume change, defined as ∆ _V/V_ = ( _V_ def _−_ _V_ 0) _/V_ 0, provides a dimensionless
measure of volumetric deformation.


**Measuring Potential Energy.** We compute the total potential energy by combining elastic strain
energy and kinetic-potential contributions. We use corotated linear elasticity, where we calculate the
deformation gradient **F** = _∇_ **u** + I and symmetric strain tensor **S** = [1] [(] **[F]** [ +] **[ F]** _[T]_ [ )] _[ −]_ [I][ to obtain the]


deformation gradient **F** = _∇_ **u** + I and symmetric strain tensor **S** = 2 [1] [(] **[F]** [ +] **[ F]** _[T]_ [ )] _[ −]_ [I][ to obtain the]

energy density _W_ = _µ_ tr( **S** [2] )+ _[λ]_ [(][tr][(] **[S]** [))][2][, with Lamé parameters] _[ µ]_ [ and] _[ λ]_ [ derived from the Young’s]


energy density _W_ = _µ_ tr( **S** [2] )+ _[λ]_ 2 [(][tr][(] **[S]** [))][2][, with Lamé parameters] _[ µ]_ [ and] _[ λ]_ [ derived from the Young’s]

modulus and Poisson’s ratio. We use three distinct contributions in the kinetic-potential term: an


33


Young’s Modulus ( _E_ ) Poisson’s Ratio ( _ν_ ) Density ( _ρ_ )


PE
PE


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V
V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


3


2


1


0


1


2


1.5


1.0


0.5


0.0


0.5


1.0


2.5

2.0

1.5

1.0

0.5

0.0

0.5


0.03


0.02


0.01


0.00


0.01


0.02


0.03


0.4


0.2


0.0


0.2


0.4


V
V


0.2 0.1 0.0 0.1 0.2 0.3

E
E


0.2 0.1 0.0 0.1 0.2 0.3

E
E


Figure 18: **Gripping Force by Robots.** We demonstrate the relation between relative errors in materials and
relative change in P.E. (top) and volume (bottom). We then show the confidence bounds in light shaded regions.
inertial component _E_ inertia = - _ρ_ [2] _[|]_ **[u]** _[n]_ [+1] _[ −]_ **[u]** _[n][|]_ [2] _[ dV]_ [that captures displacement changes between]


inertial component _E_ inertia = �Ω 2∆ _ρt_ [2] _[|]_ **[u]** _[n]_ [+1] _[ −]_ **[u]** _[n][|]_ [2] _[ dV]_ [that captures displacement changes between]

iterations in our quasi-static solver, a gravitational potential _E_ gravity = _−_ - _[ρ]_ **[ u]** _[ ·]_ **[ g]** _[ dV]_ [accounting]


iterations in our quasi-static solver, a gravitational potential _E_ gravity = _−_ �Ω _[ρ]_ **[ u]** _[ ·]_ **[ g]** _[ dV]_ [accounting]

for body forces, and an external work term _E_ ext = _−_ - **[u]** _[ ·]_ **[ f]** [ext] _[ dV]_ [representing the applied loads.]


for body forces, and an external work term _E_ ext = _−_ �Ω **[u]** _[ ·]_ **[ f]** [ext] _[ dV]_ [representing the applied loads.]

We thus compute the total potential energy as _E_ total = - _[W]_ _[dV]_ [+] _[E]_ [inertia][ +] _[E]_ [gravity][ +] _[E]_ [ext][, evaluated]


We thus compute the total potential energy as _E_ total = Ω _[W]_ _[dV]_ [+] _[E]_ [inertia][ +] _[E]_ [gravity][ +] _[E]_ [ext][, evaluated]

at the converged displacement field.


We perform the simulations in the following scenarios,


**Gripping force by robots.** We simulate a 140 N compressive force, which is common in robotic
gripping applications, for example, the Franka Emika (frankaemika, 2025) “Hand” end effector
applies a maximum of 70 N per finger with a maximum clamping force of 140 N. We demonstrate
the results from 486 simulations in this setting, all of which were run to convergence in Fig. 18.


**Impact** **Force** **on** **Dropping** **Objects.** We simulate a 120 N impact force that simulates package
drop scenarios, calculated from the impact dynamics of a 1 kg package dropped from 0.6 m height
with a 5 cm deformation distance. We demonstrate the results from 486 simulations in this setting,
all of which were run to convergence in Fig. 19.


**Tensile** **Testing** **Machines.** We simulate a 330 N force corresponding to standard tensile testing
conditions employed in bench-top universal testing machines (ASTM Committee D20, 2022; ASTM
Committee E28, 2024). We demonstrate the results from 486 simulations in this setting, all of which
were run to convergence in Fig. 20.


**Tension.** We simulate a 200 N force, which represents typical pretension in tendon-driven robotic
systems, where continuum arms and wearable assistive devices maintain structural stiffness through
cable tensions (Schäffer et al., 2024). We demonstrate the results from 486 simulations in this setting
all of which were run to convergence in Fig. 21.


E DATASET DETAILS


We present additional details about our datasets for training MatVAE (§3) and Geometry Transformer (§4.2).


34


Young’s Modulus ( _E_ ) Poisson’s Ratio ( _ν_ ) Density ( _ρ_ )


PE
PE


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


2


1


0


1


2


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.04

0.03

0.02

0.01

0.00

0.01

0.02

0.03

0.04


1.0


0.5


0.0


0.5


1.0


1.5


1.0


0.5


0.0


0.5


1.0


6


4


2


0


2


4


V
V


0.2 0.1 0.0 0.1 0.2 0.3

E
E


0.2 0.1 0.0 0.1 0.2 0.3

E
E


Figure 19: **Impact Force on Dropping Objects.** We demonstrate the relation between relative errors in materials and relative change in P.E. (top) and volume (bottom). We show the confidence bounds in light shaded
regions.


Young’s Modulus ( _E_ ) Poisson’s Ratio ( _ν_ ) Density ( _ρ_ )


0.2 0.1 0.0 0.1 0.2 0.3

E
E


0.2 0.1 0.0 0.1 0.2 0.3

E
E


PE
PE


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


PE
PE


V
V


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.0125

0.0100

0.0075

0.0050

0.0025

0.0000

0.0025

0.0050

0.0075


10.0

7.5

5.0

2.5

0.0

2.5

5.0

7.5


0.3


0.2


0.1


0.0


0.1


0.2


0.3


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.3 0.2 0.1 0.0 0.1 0.2 0.3


Figure 20: **Tensile** **Testing** **Machine.** We demonstrate the relation between relative errors in materials and
relative change in P.E. (top) and volume (bottom). We show the confidence bounds in light shaded regions.

E.1 ANNOTATION WITH VISION-LANGUAGE MODEL


To create our training dataset (§5) for the Geometry Transformer, we use a Vision-Language Model
(VLM) coupled with multiple other data sources like 3D assets, component-wise part segmentations, material databases (§5.1), visual textures, and material names to annotate our dataset. We
run the VLM on every segment of every object individually. We experiment with Qwen2.5-VL 7B,
Qwen2.5-VL 32B, Qwen2.5-VL 72B (Bai et al., 2023; 2025), VL-Rethinker (Wang et al., 2023),
SpatialRGPT (Cheng et al., 2024), and Cosmos Nemotron (Lin et al., 2024a). We experimentally
choose Qwen2.5-VL 72B for the data annotation. We show the system prompts and the user prompts
that we use in Fig. 22 to 25. We find the best performing system prompts with TextGrad (Yuksek

35


Young’s Modulus ( _E_ ) Poisson’s Ratio ( _ν_ ) Density ( _ρ_ )


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.2 0.1 0.0 0.1 0.2 0.3

E
E


0.2 0.1 0.0 0.1 0.2 0.3

E
E


PE
PE


0.3 0.2 0.1 0.0 0.1 0.2 0.3


V


0.3 0.2 0.1 0.0 0.1 0.2 0.3


0.020

0.015

0.010

0.005

0.000

0.005

0.010

0.015


0.015

0.010

0.005

0.000

0.005

0.010

0.015

0.020


PE
PE


V
V


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


0.4

0.3

0.2

0.1

0.0

0.1

0.2

0.3


PE
PE


V
V


0.6


0.4


0.2


0.0


0.2


0.4


Figure 21: **Tension.** We demonstrate the relation between relative errors in materials and relative change in
P.E. (top) and volume (bottom). We show the confidence bounds in light shaded regions.


Table 9: **VLM Annotation Errors.** Errors for the VLM annotation for mechanical property annotation.


log( _E_ ) ( _↓_ ) _ν_ ( _↓_ ) _ρ_ ( _↓_ ) log( _E/ρ_ ) ( _↓_ ) log( _G_ ) ( _↓_ ) log( _K_ ) ( _↓_ ) L.S. ( _↓_ ) E.A. ( _↓_ ) Bray–Curtis ( _↓_ )


0.0295 0.0426 0.1348 0.1961 0.0303 0.0330 0.2022 0.2162 0.2342


gonul et al., 2025). We show a response from the model for one of the segments from an object in
our dataset in Fig. 26 and 27.


We construct a tiny dataset consisting of complex objects that are manually annotated and compare
these properties with the annotations from the VLM (Qwen2.5-VL 72B ) in Tb. 9. We observe
that the VLM, given significant additional information as we provide, performs close to human
annotator performance. We report the commonly used metrics we list in Appendix D.2 for measuring
differences in mechanical properties.


E.2 DATASET STATISTICS


Our dataset comprises a diverse collection of 1,692 objects, sourced from four datasets: simready (NVIDIA Developer, 2025), residential (NVIDIA Corporation, 2025c), vegetation (NVIDIA
Corporation, 2025d), and commercial (NVIDIA Corporation, 2025a). As shown in Tb. 10, the majority of objects belong to the simready and residential categories, with vegetation and commercial
objects providing additional variety. Each object is decomposed into multiple segments, with a total
of 8,128 segments across the dataset. Most parts are labeled with English material names, and for
few parts that do not have these material names, we infer these from the PBR texture names that
were applied to these parts. To characterize the physical realism and diversity of the dataset, we
analyze the distribution of key material properties for all segments in Tb. 11. The wide range of
material properties highlights the heterogeneity of the dataset, which is essential for robust learning
and evaluation of material-aware models. We summarize the most frequent material categories (e.g.,
metal, plastic, wood, cardboard) and object classes (e.g., residential, shelf, container), along with
their respective counts and proportions in Tb. 12.


We report the statistics of our Material Triplet dataset in Tb. 13. To train MatVAE, we use the
"Filtered Dataset".


36


Figure 22: **System Prompt.** The System Prompt we use for every segment of every object.


Probability


0.12


0.08


0.04


6 7 8 9 10 11
Young's Modulus (log Pa)


Figure 28: **Young’s** **Modulus** **Pa**
**(** _E_ **).** Histogram of Young’s Modulus in our Geometry with Volumetric Materials Dataset (§5).


Probability


0.10


0.08


0.06


0.04


0.02

|Col1|Col2|
|---|---|
|||
|||
|||
|||
|||


Poisson's Ratio


Figure 29: **Poisson’s** **Ratio** **(** _ν_ **).**
Histogram of Poisson’s Ratio in our
Geometry with Volumetric Materials Dataset (§5).


37


Probability

0.16


0.12


0.08


0.04


2.5 3.0 3.5 4.0
Density (log kg/m³)

Figure 30: **Density** _mkg_ [3] **[.]** [Histogram]
of Density in our Geometry with
Volumetric Materials Dataset (§5).


Table 10: **Dataset Statistics.** Number of objects, total segments, total points, average segments per object (std.
dev.), and average points per object (std. dev.) for each dataset.


Dataset Total Objects Segments (%) Voxels (%) Avg. Segments/Object Avg. Voxels/Object


commercial 82 650 (8.0) 1,812,064 (4.9) 7.93 **(** _±_ **7.19)** 22,098 **(** _±_ **22,774)**
residential 449 4225 (52.2) 9,109,380 (24.4) 9.41 **(** _±_ **21.82)** 20,288 **(** _±_ **21,714)**
simready 1029 2544 (31.5) 24,148,660 (64.7) 2.47 **(** _±_ **1.33)** 23,468 **(** _±_ **25,032)**
vegetation 104 670 (8.3) 2,267,848 (6.1) 6.44 **(** _±_ **4.53)** 21,806 **(** _±_ **19,428)**


train 1333 6477 (80.1) 28,709,190 (76.9) 4.86 **(** _±_ **12.69)** 21,537 **(** _±_ **23,431)**
validation 165 552 (6.8) 3,719,996 (10.0) 3.35 **(** _±_ **3.19)** 22,545 **(** _±_ **23,095)**
test 166 1060 (13.1) 4,908,766 (13.1) 6.39 **(** _±_ **11.33)** 29,571 **(** _±_ **25,987)**


**Total** 1664 8089 (100.0) 37,337,952 (100.0) 4.86 **(** _±_ **11.97)** 22,439 **(** _±_ **23,786)**


Table 11: **Material property statistics for all segments in the dataset.** We report the minimum, maximum,
mean, median, standard deviation, and outlier count (% of values outside _±_ 3 _σ_ ) for Young’s modulus, Poisson’s
ratio, and Density.


Property Min Max Mean Median Std Dev Outliers (%)


Density (kg _/_ m [3] ) 5 _._ 0 _×_ 10 [1] 1 _._ 93 _×_ 10 [4] 2 _._ 28 _×_ 10 [3] 1 _._ 20 _×_ 10 [3] 2 _._ 44 _×_ 10 [3] 25 (0.3)
Young’s Modulus (Pa) 1 _._ 0 _×_ 10 [5] 2 _._ 8 _×_ 10 [11] 4 _._ 19 _×_ 10 [10] 1 _._ 0 _×_ 10 [10] 6 _._ 53 _×_ 10 [10] 165 (2.0)
Poisson’s Ratio 1 _._ 6 _×_ 10 _[−]_ [1] 4 _._ 9 _×_ 10 _[−]_ [1] 3 _._ 36 _×_ 10 _[−]_ [1] 3 _._ 5 _×_ 10 _[−]_ [1] 4 _._ 36 _×_ 10 _[−]_ [2] 88 (1.1)

F ADDITIONAL IMPLEMENTATION DETAILS


We present additional implementation details.


F.1 DESIGN OF MATVAE


We explain the motivation behind the design of MatVAE.


**Normalizing** **Flow.** Material triplets remain statistically non-Gaussian even after normalization (heavy-tailed/multi-modal log10 _E,_ log10 _ρ_ ; boundary-concentrated _ν_ _∈_ [0 _,_ 0 _._ 5)), so a
diagonal-Gaussian _qϕ_ ( _z_ _|_ _m_ ) tends to mode-average and miscalibrate tails. We therefore parameterize the posterior with a bijective normalizing flow (Rezende & Mohamed, 2015) _fψ_ where _ψ_
is the parameter of the flow network f, and Ψ is the space of all parameters _ψ_ . This _fψ_ applied
to a Gaussian base _q_ 0: sample _u_ _∼_ _q_ 0( _u_ _|_ _m_ ) = _N_ ( _u_ ; _µϕ_ ( _m_ ) _,_ diag _σϕ_ [2][(] _[m]_ [))][,] [set] _[z]_ [=] _[f][ψ]_ [(] _[u]_ [)][,] [and]
compute the density by change of variables
log _qϕ_ ( _z_ _| m_ ) = log _q_ 0� _fψ_ _[−]_ [1][(] _[z]_ [)] _[ |][ m]_          - + log��det _Jf −_ 1 [(] _[z]_ [)] ��


- �� base density


+ log��det _Jf −ψ_ 1 [(] _[z]_ [)] ��

 - ��  log-Jacobian


(13)
��� _u_ = _fψ_ _[−]_ [1][(] _[z]_ [)] _[.]_


= log _q_ 0( _u | m_ )

 - ��  base density


_−_ log��det _Jfψ_ ( _u_ )��

 - ��  log-Jacobian


with a standard normal prior _p_ ( _z_ ) = _N_ (0 _, I_ ) and decoder likelihood _pθ_ ( _m | z_ ).
Table 12: **Most** **frequent** **high-level** **material** **categories** **and** **object** **classes.** We report the top high-level
material categories (aggregated and deduplicated) and the most common object classes in the dataset, with
their respective counts and percentages.


Metal 1434 (17.7) residential 477 (28.2)
Plastic 553 (6.8) shelf 250 (14.8)
Wood 707 (8.7) container 193 (11.4)
Cardboard 315 (3.9) cardboard box 106 (6.3)
Leather 140 (1.7) vegetation 105 (6.2)
Chrome 171 (2.1) crate 101 (6.0)
Glass 85 (1.0) commercial 82 (4.8)
Fabric 60 (0.7) pallet 67 (4.0)
Rubber 55 (0.7) barricade tape 59 (3.5)
Stone 40 (0.5) inclined plane 22 (1.3)


38


Table 13: **Dataset Details.** We present statistics of our dataset for training MatVAE (§3).


Material Ranges 249
Extracted Materials 105456
Filtered Materials 101517


This keeps the ELBO form unchanged while strictly enlarging the variational family (the identity
map recovers the Gaussian case), allowing _qϕ_ to match the true posterior and avoid mode-averaging
on ( _E, ν, ρ_ ). We instantiate _fψ_ as a single radial flow,


_fψ_ ( _u_ ) = _u_ + _β h_ ( _u_ )

���      radial scale


displacement

- �� ( _u −_ _z_ 0) _,_


(14)


1
_h_ ( _u_ ) = _,_

                   - _α_ + _∥u −_ _z_ 0 _∥_ 2�


whose log-determinant has the closed form (substitute _h_ _[′]_ = _−h_ [2] in the closed form from (Rezende
& Mohamed, 2015)),
log det _Jfψ_ ( _u_ ) = ( _D −_ 1) log�1 + _βh_ ( _u_ )�


                   - ��                   angular


+ log�1 + _βh_ ( _u_ ) _−_ _β h_ ( _u_ ) [2] _r_ ( _u_ )� _,_


(15)


                    - ��                    radial

_r_ ( _u_ ) = _∥u −_ _z_ 0 _∥_ 2 _,_


where _D_ is the dimensionality of the latent space and _z_ 0 is a _trainable D_ -dimensional vector (one
per flow layer) representing the centre of the deformation.


Radial flows are invertible iff _α_ _>_ 0 and _β_ _>_ _−α_, we satisfy these by _α_ = softplus(˜ _α_ ) _,_ _β_ =
_−α_ + softplus( _β_ [˜] ) (Rezende & Mohamed, 2015) with unconstrained trainable parameters _α,_ ˜ _β_ [˜] _∈_ R.
We show our implementation in Algorithm 1.


**Penalizing TC.** We observed high dependence between latent coordinates in the aggregated posterior _q_ ¯ _ϕ_ ( _z_ ) (both dimensions tended to encode _ρ_ ). Thus, we decompose the KL-divergence term
of the ELBO following (Chen et al., 2018). For MatVAE, this allows us to directly penalize the
total correlation TC( _z_ ) = KL(¯ _qϕ_ ( _z_ ) _||_ [�] _j_ _[q]_ [¯] _[ϕ]_ [(] _[z][j]_ [))][ where] _[q]_ [¯] _[ϕ]_ [(] _[z]_ [)][ is the aggregated posterior,] _[ z][j]_ [is the]
_j_ _∈{_ 1 _,_ 2 _}_ -th coordinate of the latent vector _z_ . This allowed us to reduce the high dependence
between latent coordinates, causing both dimensions to encode density. During training, we follow (Chen et al., 2018) and approximate the aggregated posterior _q_ ¯ _ϕ_ ( _z_ ) = E _m∼pdata_ [ _qϕ_ ( _z_ _|_ _m_ )]
using samples from a mini-batch where _pdata_ is the empirical data distribution.


**Preventing Posterior Collapse.** For a fixed _m_, define per-dimension marginals _qϕ_ ( _zj_ _| m_ ) and
KL _j_ ( _z_ _| m_ ) = KL ( _qϕ_ ( _zj_ _| m_ ) _∥_ _p_ ( _zj_ )) _._ (16)


Then, for each _m_,
KL ( _qϕ_ ( _z_ _| m_ ) _∥_ _p_ ( _z_ )) =           - KL _j_ ( _m_ )


_j_


+ TC ( _qϕ_ ( _z_ _| m_ ))

   - ��    


(17)
_._


KL( _qϕ_ ( _z|m_ ) _∥_ [�]


_j_ _[q][ϕ]_ [(] _[z][j]_ _[|][m]_ [)][)]


So the total KL contains a per-dimension rate term plus a per-sample total correlation. In the Gaussian (no-flow) case, KL _j_ ( _z_ _|_ _m_ ) = 21 - _µj_ ( _m_ ) [2] + _σj_ ( _m_ ) [2] _−_ log _σj_ ( _m_ ) [2] _−_ 1�, whose gradients
drive _µj_ _→_ 0, _σj_ _→_ 1 under posterior collapse. We therefore impose a capacity constraint (“free
nats”) on the dim-wise term to prevent collapse:
_d_

     


max� _δ,_ E _p_ dataKL _j_ ( _z_ )�

_j_ =1 - �� 


_,_ (18)


- �� free-nats


39


**Algorithm 1** MatVAE posterior update with a radial normalizing fow.
**Require:**
Batch _x ∈_ R _[B][×]_ [3]
Encoder outputs _µ_ ( _x_ ) _∈_ R _[B][×][D]_
Encoder outputs log _σ_ [2] ( _x_ ) _∈_ R _[B][×][D]_
Flow param _z_ 0 _∈_ R [1] _[×][D]_
Flow param log _α ∈_ R
Flow param _β_ raw _∈_ R
Prior _p_ ( _z_ ) = _N_ (0 _, I_ )
**Ensure:**
Flowed latent _z_ _∈_ R _[B][×][D]_
Post-flow log-density log _q_ ( _z_ _| x_ ) _∈_ R _[B]_
KL term denoted as KL
Reconstruction loss _L_ recon


1: _▷_ Encode and reparameterize
2: _µ ←_ Encoder _µ_ ( _x_ )
3: log _σ_ [2] _←_ Encoderlog _σ_ 2( _x_ )
4: _ε ∼N_ (0 _, I_ )
5: _z_ base _←_ _µ_ + exp( [1] [log] _[ σ]_ [2][)] _[ ⊙]_


5: _z_ base _←_ _µ_ + exp( [1] 2 [log] _[ σ]_ [2][)] _[ ⊙]_ _[ε]_

6:
7: _▷_ Base posterior log-density
8: log _q_ 0 _←_ [�] _[D]_ _d_ =1 [log] _[ N]_ - _z_ base _,d_ ; _µd,_ exp(log _σd_ [2][)] 


9:
10: _▷_ Radial flow parameters ( _α >_ 0, _β_ _> −α_ )
11: _α ←_ softplus(log _α_ ) + _εα_
12: _β_ _←−α_ + softplus( _β_ raw)
13:
14: _▷_ Radial flow transform
15: diff _←_ _z_ base _−_ _z_ 0
16: _r_ _←∥_ diff _∥_ 2 + _εr_
17: _h ←_ ( _α_ +1 _r_ )
18: _z_ _←_ _z_ base + _β × h ×_ diff
19:
20: _▷_ Log-determinant of Jacobian
21: _bh ←_ _β × h_
22: _bh_ stab _←_ clamp( _bh,_ _−c,_ _c_ )
23: term1 _←_ ( _D −_ 1) _·_ log(1 + _bh_ stab)
24: term2 _←_ log(1 + _bh_ stab _−_ _β h_ [2] _r_ )
25: ∆log _|J| ←_ term1 + term2
26:
27: _▷_ Change of variables and KL pieces
28: log _q_ _←_ log _q_ 0 _−_ ∆log _|J|_
29: log _pz_ _←_ [�] _[D]_ _d_ =1 [log] _[ N]_ [(] _[z][d]_ [; 0] _[,]_ [ 1)]
30:
31: _▷_ Losses (non-relevant details shown as _. . ._ )
32: ( _E,_ [ˆ] ˆ _ν,_ ˆ _ρ, . . ._ ) _←_ Decoder( _z_ )
33: _L_ recon _←_ MSE/NLL in transformed space ( _. . ._ )
34: KL _←_ log _q −_ log _pz_
35: _L ←L_ recon + _. . ._
36: **return** ( _z,_ log _q,_ KL _,_ _L_ recon)


which enforces a minimum information budget _δ_ = 0 _._ 1 per coordinate (zero subgradient below _δ_ ).
This allows us to fix the empirically observed imbalance where one latent carried most information
and the other collapsed. An aggregated alternative consistent with the KL decomposition is max� _ϕ_ _·_
_d,_ [�] _j_ [KL(] _[q][ϕ]_ [(] _[z][j]_ [)] _[ ∥]_ _[p]_ [(] _[z][j]_ [))] �.


40


Table 14: **Training Hyperparameters.** We show the hyperparameters for the MatVAE and Geometry Transformer.


MatVAE


Training Precision FP-32
Hidden Width 256
Network Depth 3 ( _×_ 2)
Latent Dimensions 2
Dropout Rate 0.05
Epochs 850
Batch Size 256
Optimizer AdamW
Learning Rate 10 _[−]_ [4]

Weight Decay 10 _[−]_ [4]
LR Scheduler Cosine Annealing
Final Learning Rate 10 _[−]_ [5]
Gradient Clipping 5.0

_α_ = 1 _._ 0 (KL)
_β_ -TC Loss Weights _β_ = 2 _._ 0 (TC)
_γ_ = 1 _._ 0 (MI)
Free Nats 0.1
KL Annealing Epochs 200
Data Normalization Log Min-Max


F.2 NETWORK DESIGN


We now present our network architecture.


Geometry Transformer


Training Precision FP-16
Voxel Grid Resolution 64³
Input Channels 1024
Model Channels 768
Latent Channels 2
Transformer Blocks 12
Attention Heads 12
MLP Ratio 4
Attention Mode Swin
Window Size 8
Max Training Steps 200,000
Batch Size per GPU 4
Total Batch Size 16
Optimizer AdamW
Learning Rate 10 _[−]_ [4]

Weight Decay 5 _×_ 10 _[−]_ [2]
Gradient Clipping 1.0
Loss Function _ℓ_ 2
EMA Rate 0.9999


**MatVAE.** The _encoder_ architecture begins by projecting the 3-dimensional material triplet
through a linear transformation into a 256-dimensional hidden space, followed by SiLU activation.
The resulting representation then passes through three "ResidualBlocks", each using a bottleneck
design that compresses the 256-dimensional vector to 128 dimensions via LayerNorm and SiLU
activation, applies another linear transformation, and restores the original dimensionality through a
second LayerNorm-SiLU sequence. Each "ResidualBlock" maintains a skip connection that adds
the input directly to the final output. The encoder finally has separate linear heads that project
the processed representation into the latent space parameters: one head predicts the posterior mean
_µϕ_ ( _m_ ) and another predicts the log-variance log _σϕ_ [2][(] _[m]_ [)][ for the 2-dimensional latent code] _[ z]_ [.]


The _decoder_ mirrors this architecture in reverse, beginning with a linear projection from the 2dimensional latent space back to the 256-dimensional hidden representation, followed by SiLU activation. The latent encoding then goes through three "ResidualBlocks" with an identical bottleneck
structure and skip connections as the encoder. Finally, three separate linear heads decode the processed representation into the reconstructed material properties: Young’s modulus, Poisson’s ratio,
and density, each predicted as scalar values in the normalized space.


**Geometry** **Transformer.** Our model is based on TRELLIS (Xiang et al., 2025). We use a
transformer-based architecture specifically designed for processing sparse voxel representations
with associated material properties. The model operates on a 64 [3] resolution voxel grid, accepting 1024-dimensional DINOv2 visual features as input and compressing them to a compact 2dimensional latent representation through a 12-layer transformer backbone. Each transformer block
utilizes 12 attention heads with a 4:1 MLP expansion ratio, using Swin attention with 8 _×_ 8 local windows. During training, the Geometry Transformer operates in conjunction with a frozen MatVAE
that decodes the latent into material properties.


F.3 TRAINING


We present our voxelization scheme for training on meshes in Algorithms 2 and 3. We present the
hyperparameters used for training MatVAE and Geometry Transformer in Tb. 14.


41


**Algorithm 2** Segment-aware volumetric voxelization for meshes.
**Require:**
Full-mesh vertices _V_ all _∈_ R _[N]_ _[×]_ [3], faces _F_ all
Segments _S_ = _{_ ( _Vi_ _∈_ R _[N][i][×]_ [3] _,_ _Fi,_ sid _i_ ) _}_ _[M]_ _i_ =1
Grid resolution _r_ _∈_ N (voxel pitch _h_ = 1 _/r_ )
Per-segment cap _K_ seg _∈_ N
Global cap _K_ all _∈_ N
**Ensure:**
Combined voxel centers _C_ all _∈_ R _[L][×]_ [3] within [ _−_ 0 _._ 5 _,_ 0 _._ 5] [3]
Segment identifiers sidall _∈{_ str _}_ _[L]_

Discretized centers _C_ [ˆ] all _∈_ R _[L][×]_ [3] on an _r_ [3] grid


1: _▷_ Global normalization from the full mesh
2: _v_ min _←_ min( _V_ all)
3: _v_ max _←_ max( _V_ all)
4: _c ←_ ( _v_ min + _v_ max) _/_ 2
5: _s ←_ max( _v_ max _−_ _v_ min)
6: _ε ←_ 10 _[−]_ [6]

7:
8: _C_ acc _←_ [ ]
9: sidacc _←_ [ ]
10: **for** _i_ = 1 to _M_ **do**
11: _▷_ Normalize segment to [ _−_ 0 _._ 5 _,_ 0 _._ 5] [3] and ensure triangles
12: _Vi_ _[′]_ _[←]_ [clip((] _[V][i][ −]_ _[c]_ [)] _[/s,]_ _[−]_ [0] _[.]_ [5 +] _[ ε,]_ [0] _[.]_ [5] _[ −]_ _[ε]_ [)]
13: _Fi_ _[′]_ _[←]_ [triangulate(] _[F][i]_ [)]
14:
15: _▷_ Voxelize segment and solid-fill (Algorithm 3)
16: ( _Ci,_ _Yi_ ) _←_ VOXELIZESOLID( _Vi_ _[′][,]_ _[F]_ _i_ _[ ′][,]_ _[r]_ [)]
17: **if** _K_ seg is given and _|Ci| > K_ seg **then**
18: _I_ _←_ choice( _|Ci|,_ _K_ seg _,_ without replacement)
19: _Ci_ _←_ _Ci_ [ _I_ ]

20: **if** _|Ci|_ = 0 **then**
21: **continue**
22: _C_ acc _._ append( _Ci_ )
23: sidacc _._ append([sid _i_ ] _[|][C][i][|]_ )


24:
25: **if** _|C_ acc _|_ = 0 **then**
26: **return** (∅ _,_ ∅ _,_ ∅)

27: _C_ all _←_ concat( _C_ acc)
28: sidall _←_ concat(sidacc)
29:
30: _▷_ Optional global subsampling
31: **if** _K_ all is given and _|C_ all _| > K_ all **then**
32: _I_ _←_ choice( _|C_ all _|,_ _K_ all _,_ without replacement)
33: _C_ all _←_ _C_ all[ _I_ ]
34: sidall _←_ sidall[ _I_ ]


35:
36: _▷_ Discretize to an _r_ [3] grid aligned with [ _−_ 0 _._ 5 _,_ 0 _._ 5] [3]

37: _J_ _←_ clip( _⌊_ ( _C_ all + 0 _._ 5) _· r⌋,_ 0 _,_ _r −_ 1)
38: _C_ [ˆ] all _←_ _J/r −_ 0 _._ 5
39: **return** ( _C_ all _,_ sidall _,_ _C_ [ˆ] all)


**Voxelization For Training.** Our training dataset contains Universal Scene Description (USD) files
with multi-segment meshes. Each mesh is normalized to the range [ _−_ 0 _._ 5 _,_ 0 _._ 5] using a global bounding box computed across all segments to preserve relative spatial relationships. We use volumetric
voxelization using a regular 3D grid with a resolution of 641 [,] [where] [each] [voxel] [center] [is] [tested] [for]
interior containment within the mesh volume through point-in-polyhedron testing, followed by vol

42


**Algorithm 3** Voxelization and food fll primitives for meshes.


1: **procedure** VOXELIZESOLID( _V,_ _F,_ _r_ )
2: _▷_ Grid setup over a padded mesh AABB
3: _h ←_ 1 _/r_
4: _a_ min _←_ min( _V_ ); _a_ max _←_ max( _V_ )
5: _b_ min _←_ _a_ min _−_ _h_
6: _b_ max _←_ _a_ max + _h_
7: _nx_ _←⌈_ ( _b_ max _,x −_ _b_ min _,x_ ) _/h⌉_
8: _ny_ _←⌈_ ( _b_ max _,y −_ _b_ min _,y_ ) _/h⌉_
9: _nz_ _←⌈_ ( _b_ max _,z_ _−_ _b_ min _,z_ ) _/h⌉_
10: _S_ [ _nx, ny, nz_ ] _←_ false
11: _X_ [ _nx, ny, nz_ ] _←_ false
12:
13: _▷_ Triangle rasterization: mark surface cells
14: **for** each triangle _t_ = ( _v_ 0 _, v_ 1 _, v_ 2) _∈_ _F_ **do**
15: Compute triangle AABB [ _t_ min _, t_ max] in world coordinates
16: Convert to grid index ranges ( _i_ min : _i_ max _,_ _j_ min : _j_ max _,_ _k_ min : _k_ max)
17: **for** _i_ = _i_ min to _i_ max **do**
18: **for** _j_ = _j_ min to _j_ max **do**
19: **for** _k_ = _k_ min to _k_ max **do**
20: Cell box _B_ = [ _b_ min + ( _i, j, k_ ) _h,_ _b_ min + ( _i_ + 1 _, j_ + 1 _, k_ + 1) _h_ ]
21: **if** TRIANGLEBOXINTERSECT( _t,_ _B_ ) **then**
22: _S_ [ _i, j, k_ ] _←_ true


23:
24: _▷_ Exterior marking by flood fill on non-surface cells
25: Initialize queue _Q_ with boundary indices ( _i, j, k_ ) where _S_ [ _i, j, k_ ] = false
26: **while** _Q_ not empty **do**
27: _u ←_ _Q._ pop()
28: **if** _X_ [ _u_ ] **then**
29: **continue**
30: _X_ [ _u_ ] _←_ true
31: **for** each 6-neighbor _v_ of _u_ within bounds **do**
32: **if** _S_ [ _v_ ] = false and _X_ [ _v_ ] = false **then**
33: _Q._ push( _v_ )


34:
35: _▷_ Solid fill (interior) and center extraction
36: _Y_ _←¬X_ _∧¬S_
37: _C_ _←_ [ ]
38: **for** all indices ( _i, j, k_ ) where _Y_ [ _i, j, k_ ] = true **do**
39: _c ←_ _b_ min + ( _i_ + 0 _._ 5 _,_ _j_ + 0 _._ 5 _,_ _k_ + 0 _._ 5) _· h_
40: _C._ append( _c_ )


41: **return** ( _C,_ _Y_ )


umetric filling to generate solid voxel representations rather than surface-only discretizations. All
the voxels inside a given segment receive the material properties of the segment they lie in.


**Rendering** **for** **Training.** For multi-view image rendering of meshes, we use a path-tracing renderer to produce photorealistic renderings of 3D objects. Camera viewpoints are sampled using a
quasi-random Hammersley sequence distributed uniformly across a sphere. For training and testing,
we render 150 views, though our method can work by rendering as many views as needed, with cameras positioned at a fixed radius of 2 units from the object center and configured with a 40-degree
field of view. Images are rendered at 512 _×_ 512 pixel resolution. The rendering pipeline outputs both
the RGB images and the corresponding camera extrinsics and intrinsics. For rendering splats, we
simply replace the renderer with the 3D Gaussian Splat renderer (Kerbl et al., 2023) in our workflow.
For rendering SDFs, we render meshes using many points collected from the SDF. For rendering
NeRFs (Mildenhall et al., 2020), we simply replace the renderer with nerfstudio (Tancik et al.,
2023) in our workflow.


43


**Feature Aggregation.** For visual feature extraction, we employ DINOv2-ViT-L/14 (Oquab et al.,
2024) with registers. We use a patch size of 14 _×_ 14 pixels and process input images resized to
518 _×_ 518 pixels, resulting in a 37 _×_ 37 patch. We use the nv-dinov2 [2] (NVIDIA, 2025) implementation.


F.4 SIMULATION AND RENDERING


For our mesh simulations (Fig. 5), we simulate with the finite-element method (FEM) using the
libuipc (Huang et al., 2025; 2024a) implementation, and we render the simulations in a pathtracing-based renderer. While comparing with other simulators (Fig. 2) we use MPM (Sulsky
et al., 1994) using taichi-mpm (Hu & contributors, 2018), XPBD (Macklin et al., 2016) using PositionBasedDynamics (Bender & contributors, 2015), and FEM using Warp (Macklin,
2022).


For our large-scale splat simulations or splat + mesh simulations, we use Simplicits (Modi et al.,
2024) using the sparse simplicits implementation using Kaolin (Modi et al., 2024; Fuji Tsang et al.).
For rendering our large-scale splat simulations or splat + mesh simulations, we use Polyscope (Sharp
et al., 2019) and composite splat renders from gsplat (Ye et al., 2024). For these simulations,
we apply material property tolerances to reduce numerical noise: voxels with Young’s modulus
differing by less than 10 [1] Pa, Poisson’s ratio by less than 10 _[−]_ [3], or density by less than 10 [1] kg/m³
are assigned identical values for the respective property. We present additional details for deforming
and rendering deformed Gaussian Splats in Appendix G.5.


F.5 BASELINES


**Converting Hardness to Young’s Modulus.** NeRF2Physics (Zhai et al., 2024) does not estimate
a numerical value of Young’s Modulus, but instead predicts Shore A-Shore D hardness. Thus, to
compare our method with NeRF2Physics (Zhai et al., 2024) we convert these Shore hardness values
to average Young’s Modulus values.


**Shore A.** For Shore A hardness, we follow (ASTM International, 2015) and use:
_E_ MPa = _e_ [(] _[S][A][×]_ [0] _[.]_ [0235)] _[−]_ [0] _[.]_ [6403] (19)
where _SA_ is the Shore A hardness value and _E_ MPa is Young’s modulus in megapascals.


**Shore D.** For Shore D hardness, we follow (ASTM International, 2015) and use:
_E_ MPa = _e_ [((] _[S][D]_ [+50)] _[×]_ [0] _[.]_ [0235)] _[−]_ [0] _[.]_ [6403] (20)
where _SD_ is the Shore D hardness value and _E_ MPa is Young’s modulus in megapascals.


**Point** **or** **Voxel** **Sampling.** The baselines NeRF2Physics (Zhai et al., 2024) and PUGS (Shuai
et al., 2025) in their methods sample points from the NeRF or Gaussian splat, respectively, and
predict mechanical properties at those points. To ensure fair comparisons in Tb. 4 and Fig. 6b, we
explicitly make these methods work on the same set of points in the object on which our method is
evaluated.


**Implementation** **details** **of** **Baselines.** The baseline NeRF2Physics (Zhai et al., 2024) uses
gpt-3.5-turbo for certain parts of their pipeline. We replace gpt-3.5-turbo in their
pipeline with a better performing model, GPT-4o (OpenAI & et al., 2024). The baseline
Phys4DGen (Liu et al., 2024b) does not have code available. Thus, we faithfully reproduce the parts,
"Material Grouping and Internal Discovery" and "MLLMs-Guided Material Identification". We reproduce these parts of their pipeline using GPT-4o (OpenAI & et al., 2024) for the MLLMs-Guided
Material Identification. Furthermore, we obtained the prompts from the authors of Phys4DGen (Liu
et al., 2024b) and use the same prompts.


G ADDITIONAL DETAILS ON THE SIMULATIONS


We experiment with Simplicits (Modi et al., 2024), a reduced-order simulator (Fig. 1, 5, 8c and 8e)
and an accurate finite-element method (FEM) simulator (Fig. 1, 5 and 8b) with our material prop

[2https://build.nvidia.com/nvidia/nv-dinov2](https://build.nvidia.com/nvidia/nv-dinov2)


44


Table 15: Hyperparameters for FEM simulation.


Hyperparameter Value Hyperparameter Value


Time Integrator Backward Euler Linear Solver pre-conditioned CG
Nonlinear Solver Newton’s w/ line search Linear tolerance 10 _[−]_ [3]
Newton max iters. 1024 Line search
Velocity tol. 0.05 _ms_ _[−]_ [1] max iters 8
CCD tol. 1.0 Collision
Transform rate tol. 0.1/s Friction 0.5
_dt_ 0.02 Contact Resistance 1.0
Gravity [0 _._ 0 _, −_ 9 _._ 8 _,_ 0 _._ 0] _d_ ˆ 0.01


erties. We also use a FEM simulator for our experiments on interpreting errors in properties (Appendix D.4). We use a material point method (MPM) (Sulsky et al., 1994), and an Extended Position
Based Dynamics (XPBD) (Macklin et al., 2016) simulator for our experiments to compare between
simulators (Fig. 2). We share details on these simulations. We also share details on the interpolation
we use across all our simulations. We share the hyperparameters used for all the FEM simulations
in Tb. 15.


G.1 INTERPOLATION SCHEME


Our simulations receive a material field sampled on a voxel grid predicted by VoMP, i.e., values
_m_ ( **X** _i_ ) given at lattice points _{_ **X** _i}_ _⊂_ Ω. When the simulator needs material values at arbitrary
query locations **X** (e.g., element centroids or vertices), we evaluate a nearest-neighbour interpolation
of the voxel field:
_i_ _[∗]_ ( **X** ) = arg min
_i_ _[∥]_ **[X]** _[ −]_ **[X]** _[i][∥]_ [2] _[,]_

(21)
_m_ _[∗]_ ( **X** ) = _m_        - **X** _i∗_ ( **X** )� _._

We intentionally avoid higher-order interpolation of material fields since real objects are piecewiseconstant across label regions, and convex blending across parts of the objects invents intermediate
materials. These intermediate materials might not be physically present or admissible, while our outputs fall into a valid material due to the MatVAE (§3). Nearest-neighbour preserves sharp interfaces
and is usually robust for arbitrary query locations.


G.2 PREPARING SCENES AND ASSIGNING MATERIALS FOR THE FEM SOLVER


Mechanical properties are set either uniformly (like in Appendix D.4) or heterogeneously from a
voxel field. For uniform assignment, given _E_ and _ν_ we compute Lamé parameters

_E ν_ _E_
_λ_ = _µ_ = (22)
(1 + _ν_ )(1 _−_ 2 _ν_ ) _[,]_ 2(1 + _ν_ ) _[,]_

which are used elementwise together with a constant mass density _ρ_ .


For heterogeneous assignment, a voxel lattice provides _E_ ( **X** ), _ν_ ( **X** ), and _ρ_ ( **X** ) at voxel centers.
After applying the same rigid/scale transform as the mesh, each tetrahedron takes _λ, µ_ from the
nearest voxel to its centroid, and each vertex takes _ρ_ from the nearest voxel to its position. This
produces per-tetrahedron _λ, µ_ and per-vertex _ρ_ fields that are directly used in the elastic strain energy
density per unit reference volume ( _W_ ), first variation of the incremental potential ( _R_ ), and NewtonJacobian ( _K_ ).


During simulation, a visual mesh is embedded into the physics mesh by assigning each visual vertex
**x** _v_ to a containing (or nearest) tetrahedron with vertices _{_ **X** _a}_ [4] _a_ =1 [and barycentric weights] _[ {][w][a][}]_ _a_ [4] =1
satisfying [�] _a_ _[w][a]_ [=] [1][ and][ �] _a_ _[w][a]_ **[ X]** _[a]_ [=] **[x]** _[v]_ [; its deformed position is then the barycentric interpo-]
lation of current nodal positions _{_ **x** _a}_ [4] _a_ =1 [:]


**x** _v_ [def] =


4

- _wa_ **x** _a ._ (23)


_a_ =1


The state update in our simulation experiments is computed time-step by time-step, and we also
deform and move the visual mesh according to the physics mesh at each time step.


45


G.3 DETAILS OF THE FEM SOLVER


For FEM simulations, we use a simulator based on the libuipc (Huang et al., 2025; 2024a)
implementation and the Warp (warp.fem) (Macklin, 2022) implementation. We first explain the
details for our simulations in §D.4.


We consider a deformable continuum body with reference configuration Ω _⊂_ R [3] and boundary
_∂_ Ω= Γ _D_ _∪_ Γ _N_, where Γ _D_ denotes boundary points with Dirichlet boundary conditions, and Γ _N_
denotes boundary points with Neumann boundary conditions. The unknown to solve for is the
displacement field **u** : Ω _→_ R [3] . Time is discretized into frames with a fixed step ∆ _t_ . At each
frame we compute an increment ∆ **u** that advances the configuration **u** _←_ **u** + ∆ **u** while enforcing
Dirichlet constraints on Γ _D_ . The deformation map is _φ_ ( **X** ) = **X** + **u** ( **X** ), with deformation gradient
**F** ( **u** ) = **I** + _∇_ **u**, Jacobian _J_ = det **F**, and isochoric invariant _Ic_ = tr( **F** _[⊤]_ **F** ). For corotational
modeling, we use the stretch tensor **S** from the polar/SVD decomposition: if **F** = **U** diag( _**σ**_ ) **V** _[⊤]_
then **S** = **V** diag( _**σ**_ ) **V** _[⊤]_ . Given Young’s modulus _E_ and Poisson ratio _ν_, the Lamé parameters are
_λ_ = _Eν/_ ((1 + _ν_ )(1 _−_ 2 _ν_ )) and _µ_ = _E/_ (2(1 + _ν_ )). Here _∇_ denotes the gradient with respect to
reference coordinates, **A** : **B** = tr( **A** _[⊤]_ **B** ) is the Frobenius inner product, and _∥·∥_ is the Euclidean
norm.


The elastic response we use is the corotational Hookean model. Define the small strain _**ε**_ = **S** _−_ **I** .
The strain energy density and Kirchhoff stress are
_W_ CR( **S** ) = _µ_ _**ε**_ : _**ε**_ + _[λ]_ 2 [tr(] _**[ε]**_ [)][2] _,_
shear (deviatoric)���                                      -                                      - ��                                      


_[λ]_ 2 [tr(] _**[ε]**_ [)][2] _,_

- �� volumetric


_,_


(24)
_,_


_**τ**_ ( **S** ) = 2 _µ_ _**ε**_ + _λ_ tr( _**ε**_ ) **I**

����     - ��     shear volumetric


with a consistent linearization obtained via the variation of **S** with respect to **F** and projected to
maintain symmetry and positive semidefiniteness.


Each frame solves an incremental variational problem. Given the previous increment ∆ **u** _[n][−]_ [1], we
seek ∆ **u** _[n]_ that approximately minimizes the incremental potential


Ω _ρ_ - 12 _∥_ ∆ **u** _−_ ∆∆ _t_ **u** [2] _[n][−]_ [1] _∥_ [2]


�Ω _ρ_ - 12 _∥_ ∆ **u** _−_ ∆∆ _t_ **u** [2] _[n][−]_ [1] _∥_ [2] - d _V_

- �� inertial regularization


    Π(∆ **u** ) =


∆ _t_ [2]


 +


  - _−_ _ρ_ **g** _·_ ∆ **u** _−_ **f** ext _·_ ∆ **u**  - d _V_

Ω

- �� body and external work


(25)


Ω


 +


+ _W_ CR� **S** ( **u** _[n][−]_ [1] + ∆ **u** )� d _V_

Ω

 - ��  elastic energy

+ Πint(∆ **u** ) _,_

    - ��     interior/boundary regularization


_,_


where _ρ_ is the mass density, **g** is the gravitational acceleration vector, **f** ext denotes prescribed volumetric loads, and Πint denotes any interior/boundary regularization term (e.g., a jump penalty in
discontinuous settings). The admissible test function **v** is any sufficiently smooth virtual displacement that vanishes on Γ _D_ . The first variation _δ_ Π(∆ **u** ; **v** ) = 0 for all such **v** yields the residual


46


functional


_ρ_ [∆] **[u]** _[ −]_ [∆] **[u]** _[ n][−]_ [1]
Ω ∆ _t_ [2]


     _R_ (∆ **u** )[ **v** ] =


_ρ_ _·_ **v** d _V_
Ω ∆ _t_ [2]

- �� inertia


 +


  - _−_ _ρ_ **g** _·_ **v** _−_ **f** ext _·_ **v**  - d _V_

Ω

- �� body/external


(26)


Ω


 
_−_


_−_ _**τ**_ - **S** ( **u** _[n][−]_ [1] + ∆ **u** )� : _∇_ **v** d _V_

Ω

 - ��  elastic (internal) virtual work

+ _R_ int(∆ **u** )[ **v** ] _,_

 - ��  regularization


_,_


which is set to zero for all **v** . Newton’s method is applied to _R_ (∆ **u** ) = 0. At iterate ∆ **u** [(] _[k]_ [)] we
assemble the consistent tangent operator _K_ = D _R_ �∆ **u** [(] _[k]_ [)][�] (the Gàteaux derivative of _R_ ) and solve
the linear system
_K δ_ **u** = _−R_ �∆ **u** [(] _[k]_ [)][�] _,_

(27)
∆ **u** [(] _[k]_ [+1)] = ∆ **u** [(] _[k]_ [)] + _α δ_ **u**


where _α ∈_ (0 _,_ 1] is chosen by a backtracking Armijo rule to guarantee sufficient decrease of Π. The
operator _K_ contains an inertial mass-like term �Ω _[ρ]_ [ ∆] _[t][−]_ [2] _[ δ]_ **[u]** _[ ·]_ **[ v]** [ d] _[V]_ [,] [the] [consistent] [elastic] [tangent]

from the linearization of _**τ**_ ( **S** ( _·_ )), and any interior/boundary penalty contributions. This procedure
is repeated until the update norm or residual falls below a prescribed tolerance.


For all our other simulations ( _i.e._ except the simulations in §D.4) we use a closely related variant
whose differences are in the constitutive law, material assignment, mesh preparation/interpolation,
and contact handling. First, the stored energy and stress are taken to be compressible Neo-Hookean
with volumetric regularization. Writing **C** = **F** _[⊤]_ **F** and **B** = **F F** _[⊤]_, the energy and Kirchhoff stress
are
_W_ NH( **F** ) = _µ_ 2 �tr **C** _−_ 3 _−_ 2 ln _J_      - + _[λ]_ 2 [(ln] _[ J]_ [)][2] _[,]_


_[λ]_

2 [(ln] _[ J]_ [)][2] _[,]_


(28)
_**τ**_ NH( **F** ) = _µ_ ( **B** _−_ **I** ) + _λ_ ln _J_ **I** _._


This change only affects the elastic terms in Π, _R_, and _K_ ; the kinematics and inertial terms remain
the same. For the simulation experiments, we also use IPC (Li et al., 2020a) for collision handling.


G.4 PREPARING SCENES AND ASSIGNING MATERIALS FOR THE SIMPLICITS SOLVER


Each object is specified by a set of quadrature points _Q_ = _{_ **X** _q}_ that sample its volume (used for
elasticity and inertia), a set of collision particles _C_ = _{_ **X** _c}_ for contact, and a set of visual vertices
for rendering. We position objects with a rigid transform (origin and rotation) and an object scale;
these transforms are applied consistently when evaluating kinematics, gravity, and material fields.


We embed all objects into a regular grid domain and attach to this grid a low-dimensional Simplicits
subspace. The displacement basis is the product of a trilinear grid shape and a per-object handle
shape, with multiple duplicated handles per grid vertex. At each quadrature point, we evaluate and
cache per-node subspace weights and their spatial gradients. These weights modulate the duplicated
handle functions during assembly, instantiating the Simplicits subspace on the grid.


Material parameters are assigned per quadrature point from a voxel lattice providing _E_ ( **X** ), _ν_ ( **X** ),
and _ρ_ ( **X** ). After applying the same rigid/scale transform as the object, each quadrature location **X** _q_
takes its material from the nearest voxel to **X** _q_ . We compute Lamé parameters per point as

_E_ ( **X** _q_ ) _ν_ ( **X** _q_ )
_λq_ =
(1 + _ν_ ( **X** _q_ ))(1 _−_ 2 _ν_ ( **X** _q_ )) _[,]_

(29)
_E_ ( **X** _q_ )
_µq_ =
2(1 + _ν_ ( **X** _q_ )) _[,]_

and use _ρq_ = _ρ_ ( **X** _q_ ) in the inertial terms.


47


The quadrature weights are set uniformly as

_v_
_wq_ = _|Q|_ _[,]_ (30)

where _v_ is the object volume estimate and _|Q|_ is the number of quadrature points. This makes elastic
and inertial energies invariant to the sampling density.


Collision particles _C_ are used for detecting and resolving contact against other particles and registered kinematic triangle meshes (containers and obstacles). We use an IPC-style barrier with
Coulomb friction, and scale the contact stiffness by object volume and the number of collision
particles to obtain comparable penalties across scenes.


G.5 DEFORMING SPLATS AND RENDERING DEFORMED SPLATS


We render each object as a set of anisotropic Gaussian splats. At rest, a splat is parameterized by its
mean _**µ**_ 0 _∈_ R [3], a unit quaternion **q** 0 (with rotation **R** 0 _∈_ SO(3)), and axis scales **s** 0 _∈_ R [3] _>_ 0 [.] [We]
define the rest-frame shape operator
**L** = **R** 0 diag( **s** 0) _._ (31)
����
rest anisotropy


During simulation, the displacement field yields a world-space deformation gradient **F** at
����
local deformation

the splat center and a world-space position _**µ**_ (obtained by evaluating the embedded deformation at
the visual vertex). We map the rest anisotropy through the local deformation to obtain the worldspace covariance of the splat as


Σ = ( **F L** )
world covariance���� deformed axes� ���


deformed axes _[⊤]_

��� ( **F L** ) _[⊤]_ + _ε_ **I** _._ (32)
����
SPD padding


Here _ε >_ 0 is a small scalar that guarantees positive-definiteness under extreme compression.


For rasterization we pass _**µ**_ and Σ to the Gaussian renderer. The renderer (gsplat (Ye et al., 2024))
expects a symmetric 6-vector parameterization; we therefore pack the lower-triangular entries as

**c** =            - Σ11 _,_ Σ12 _,_ Σ13 _,_ Σ22 _,_ Σ23 _,_ Σ33            - _⊤_ _._ (33)
����
packed covariance

Color appearance (spherical-harmonic coefficients) and opacity are carried from the rest representation; only the mean _**µ**_ and covariance Σ change over time. We also support a scalar scale multiplier
applied to **s** 0 for interactive control in qualitative visualizations.


Given the view (extrinsic) matrix **V** and vertical field-of-view, we synthesize camera intrinsics


**K** =


- _fx_ 0 _cx_ 0 _fy_ _cy_
0 0 1


_,_


_fx_ = _W_


2 _[x]_ ) _[,]_ _[f][y]_ [=] 2 tan( _H_


_[y]_ ) _,_

2


(34)


2 tan( [fov] _[x]_


2 tan( [fov] _[y]_


_cx_ = _W_


2 _[,]_ _[c][y]_ [=] _H_ 2


2 _[,]_


with image size ( _W, H_ ). We then render the set of splats _{_ _**µ**_ _,_ **c** _}_ under ( **V** _,_ **K** ) to produce RGB
(and depth) frames. At every frame, we interpolate _**µ**_ and **F** at the visual vertices, form Σ =
**F L L** _[⊤]_ **F** _[⊤]_ + _ε_ **I**, pack it as **c**, and feed the Gaussian rasterizer together with the stored colors and
opacities.


G.6 DETAILS OF THE SIMPLICITS SOLVER


We use the Simplicits (Modi et al., 2024) simulator based on the implementation in
Kaolin (Fuji Tsang et al.). Simplicits solves for a displacement field represented in a lowdimensional subspace attached to a regular grid. This subspace is a product basis between trilinear grid polynomials and duplicated per-vertex “handle” functions whose influence is modulated
by per-point weights and weight gradients evaluated at quadrature points. We assemble inertia and
compressible Neo-Hookean elasticity on this subspace, using the per-quadrature Lamé parameters


48


( _λq, µq_ ) and measures _wq_ . Each frame performs Newton steps on the incremental potential with
backtracking line search. Linear systems are solved by preconditioned conjugate gradients.


_In_ _the_ _Kaolin_ _implementation_, splat–splat contact uses particle pairs: for a pair ( _a, b_ ) with current
positions **x** _a,_ **x** _b_ and contact radius _r_, we set **n** _c_ = ( **x** _a −_ **x** _b_ ) _/∥_ **x** _a −_ **x** _b∥_, _rc_ = _r_, and use the relative
offset **o** _c_ (∆ **u** ) = ∆ **u** _a −_ ∆ **u** _b_, so that _dc_ = **n** _c·_ ( **x** _a −_ **x** _b_ ) and **v** _t,c_ measures tangential slip between
the two.


_However, in our implementation_, splat–mesh contact is handled differently than splat-splat contact.
For splat-mesh contact, instead of using the collision points, we use triangle meshes as kinematic
colliders: for a particle at **x** and its closest point **p** on a nearby triangle (with interpolated mesh
normal/velocity **n** _c,_ **v** _m_ ), we take _rc_ = 2 _r_ and

**o** _c_ (∆ **u** ) = ∆ **u** +   - _∥_ **x** _−_ **p** _∥−_ **n** _c_ _·_ ∆ **u**   - **n** _c −_ **v** _m,_ _dc_ = _∥_ **x** _−_ **p** _∥−_ **n** _c_ _·_ **v** _m ._ (35)
Only the simulated-object DOFs enter **o** _c_ ; mesh or splat motion appears through **v** _m_ .


These terms are used in the Newton system as
**K** _←_ **K** + _α_ **H** _[⊤]_ **C H** _,_ **r** _←_ **r** _−_ _α_ **H** _[⊤]_ **g** _,_ (36)
contact stiffness� ��                              - contact force� ���

where **H** is the Jacobian of contact offsets, and **g** _,_ **C** are the per-contact gradient/Hessian with respect to **o** _c_ . For splat–mesh contacts only the simulated-object block of **H** is present; for splat–splat
contacts the two object blocks appear with opposite signs.


G.7 PREPARING SCENES AND ASSIGNING MATERIALS FOR THE XPBD SOLVER


We also use an Extended Position-Based Dynamics (XPBD) solver (Macklin et al., 2016) based on
PositionBasedDynamics (Bender & contributors, 2015).


**Particles and initialization.** Objects are represented by particles positioned at rest locations _{_ **X** _i}_ .
For each particle we initialize position **x** _i ←_ **X** _i_ and a previous position equal to **X** _i_ (Verlet), set mass
_mi_ and inverse mass _wi_ = 1 _/mi_ (pinned points use _wi_ = 0). A soft sphere uses one center particle
and a set of surface particles sampled on a UV sphere; the center forms simple tetrahedra with
nearby surface points for volume preservation.


**Material parameters and compliance.** To target an elastic behavior with Young’s modulus _E_ and
Poisson ratio _ν_ (bulk modulus _K_ = _E/_ (3(1 _−_ 2 _ν_ ))), we choose distance- and volume-constraint
compliances _α_ dist _, α_ vol inversely proportional to _E_ and _K_ . XPBD uses the scaled compliance
_α_ ∆ = _α/_ ∆ _t_ [2] so that smaller _α_ yields stiffer response.


**Prediction, projections, and updates.** Each frame predicts positions with Verlet integration, then
iteratively projects distance, volume, and collision constraints. For a pairwise distance constraint,
the XPBD update displaces endpoints along the edge with a factor

_α_ ∆ **x** _i −_ **x** _j_
_γ_ = _,_ ∆ **x** _i_ _∝−γ Cij_ ∆ **x** _j_ = _−_ _[w][i]_ ∆ **x** _i,_ (37)
_α_ ∆ + _wi_ + _wj_ _∥_ **x** _i −_ **x** _j∥_ _[,]_ _wj_

with the analogous formulation for volume constraints using their gradients. After projections, we
update positions and apply ground-plane contact with Coulomb friction.


**Visualization.** We track the evolving surface by updating a triangular mesh whose vertices coincide with the surface particles.


G.8 PREPARING SCENES AND ASSIGNING MATERIALS FOR THE MPM SOLVER


We use a material point method (MPM) simulator (Sulsky et al., 1994) based on taichi-mpm (Hu
& contributors, 2018). Scenes are specified by a uniform Cartesian grid, a set of particles sampling
each object’s volume, and per-object mechanical properties.


**Domain, grid, and timestep.** We embed all objects in a unit cube domain discretized by an _n_ grid _×_
_n_ grid _× n_ grid grid (cell size ∆ _x_ = 1 _/n_ grid). We use a fixed time step ∆ _t_ small enough for stability
(e.g., ∆ _t_ = 5 _×_ 10 _[−]_ [5] in our experiments).


49


**Particles and initialization.** Each object is sampled with material points (particles) positioned at
rest locations _{_ **X** _p}_ . For each particle we initialize position **x** _p_ _←_ **X** _p_, velocity **v** _p_ _←_ **0**, affine
velocity field **C** _p ←_ **0**, and deformation gradient **F** _p ←_ **I** . Mass is set as _mp_ = _ρ Vp_ with density
_ρ_ and particle volume _Vp_ consistent with the grid resolution. In the simple drop test, we sample a
sphere at a height and let gravity act.


**Material parameters and constitutive law.** We assign per-object Young’s modulus _E_ and Poisson ratio _ν_, compute Lamé parameters _µ_ = _E/_ (2(1 + _ν_ )) and _λ_ = _Eν/_ ((1 + _ν_ )(1 _−_ 2 _ν_ )), and
use a fast corotated (FCR) elastic model. With polar/SVD decomposition **F** = **U** diag( _**σ**_ ) **V** _[⊤]_ and
rotation **R** = **UV** _[⊤]_, the Kirchhoff stress is
_**τ**_ FCR( **F** ) = 2 _µ_ ( **F** _−_ **R** ) **F** _[⊤]_ + _λ J_ ( _J_ _−_ 1) **I** _,_ _J_ = det **F** _._ (38)


**Transfers** **and** **updates.** Each step performs Particle-to-Grid (P2G) transfers using quadratic Bspline weights: we scatter particle mass, momentum, and internal forces _−Vp_ _**τ**_ _∇w_ to grid nodes.
On the grid, we (i) convert momentum to velocity, (ii) add gravity, and (iii) enforce box boundary
conditions by clamping outward normal velocities to zero. We then perform Grid-to-Particle (G2P)
to interpolate grid velocities back to particles, update the affine field, and integrate
**x** _p_ _←_ **x** _p_ + ∆ _t_ **v** _p,_ **F** _p_ _←_ ( **I** + ∆ _t ∇_ **v** ) **F** _p._ (39)


**Visualization.** We embed a coarse surface at rest, align it to the particle center of mass, and update
its vertices via interpolation of nearby particle displacements.


H OTHER RELATED WORKS


For completeness, we include other tangentially related works here. A different setting from ours
is inferring physical properties given additional observations, such as video (Davis et al., 2015;
Mottaghi et al., 2016; Bhat et al., 2002; Chen et al., 2025b; Liu et al., 2024a; Xue et al., 2023; Li
et al., 2025; Brubaker et al., 2009; Yildirim et al., 2016; Li et al., 2023; 2020b; Wu et al., 2016; 2015;
2017; Xia et al., 2024; Xu et al., 2019; Feng et al., 2024; Lin et al., 2024b) or physical manipulation
of real objects (Yu et al., 2024; Pai et al., 2001; Lang et al., 2003; Lloyd & Pai, 2001; Pai et al.,
2008; Pai, 2000; Yao & Hauser, 2023; Pinto et al., 2016). Other related works focus on generating
new physically plausible shapes, e.g. stable under gravity or other interactions, but cannot augment
existing 3D assets with mechanical properties, which is our goal (Lin et al., 2025b; Guo et al., 2024;
Chen et al., 2024; Ni et al., 2024; Yang et al., 2024; Mezghanni et al., 2022; Chen et al., 2025a; Cao
et al., 2025; Cao & Kalogerakis, 2025). Other methods predict displacements (Zhang et al., 2024;
Shi et al., 2023), bypassing mechanical properties, or focus on other aspects such as articulation (Xia
et al., 2025).


50


Figure 23: **User Prompt I.** The User Prompt we use for every segment of every object.


51


Figure 24: **User Prompt II.** The User Prompt we use for every segment of every object.


52


Figure 25: **User Prompt III.** The User Prompt we use for every segment of every object.


53


Figure 26: **Example Response I.** We demonstrate an example response for a segment from one of the objects
from our dataset. The given object has one part.


54


Figure 27: **Example Response II.** We demonstrate an example response for a segment from one of the objects
from our dataset. The given object has two parts, and we show the response for the "rubber cap" part.


55
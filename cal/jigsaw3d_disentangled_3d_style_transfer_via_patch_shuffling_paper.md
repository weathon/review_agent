# JIGSAW3D: DISENTANGLED 3D STYLE TRANSFER
#### VIA PATCH SHUFFLING AND MASKING


**Anonymous authors**
Paper under double-blind review


Figure 1: We propose the JIGSAW3D, a versatile 3D stylization framework that transfers stylistic
statistics from 2D images to 3D meshes. Our method achieves high stylistic consistency across
multiple, diverse objects in a scene ( **top** ). Furthermore, it demonstrates high versatility with various
art styles and supports partial reference stylization for fine-grained user control ( **bottom** ).


ABSTRACT


Controllable 3D style transfer seeks to restyle a 3D asset so that its textures match
a reference image while preserving the integrity and multi-view consistency. The
pravelent methods either rely on direct reference style token injection or scoredistillation from 2D diffusion models, which incurs heavy per-scene optimization and often entangles style with semantic content. We introduce Jigsaw3D, a
multi-view diffusion based pipeline that decouples style from content and enables
fast, view-consistent stylization. Our key idea is to leverage the jigsaw operation—spatial shuffling and random masking of reference patches—to suppress object semantics and isolate stylistic statistics (color palettes, strokes, textures). We
integrate these style cues into a multi-view diffusion model via reference-to-view
cross-attention, producing view-consistent stylized renderings conditioned on the
input mesh. The renders are then style-baked onto the surface to yield seamless textures. Across standard 3D stylization benchmarks, Jigsaw3D achieves
high style fidelity and multi-view consistency with substantially lower latency,
and generalizes to masked partial reference stylization, multi-object scene styling,
and tileable texture generation.


1 INTRODUCTION


The field of 3D object generation has advanced rapidly, driven by progress in 3D generative modeling (Zhang et al., 2024; Xiang et al., 2025; Li et al., 2025; Wu et al., 2024; Zhang et al., 2023a),
neural representations (Mildenhall et al., 2021; Park et al., 2019; Mescheder et al., 2019), and the
availability of large-scale 3D datasets (Chang et al., 2015; Deitke et al., 2023b). Within this landscape, 3D stylization—transferring the artistic style of a 2D reference image to a 3D asset while
preserving object identity and multi-view consistency—has emerged as a practical requirement for


1


virtual reality, game development, and animated content creation. Progress, however, is hampered
by the absence of large-scale 3D style corpora containing paired texture and style supervision, which
makes end-to-end supervised training impractical.


In response, recent approaches typically follow one of two directions. Training-free methods inject reference style cues into frozen attention layers to achieve multi-view style fusion; for example, 3D-style-LRM (Oztas et al., 2025) combines CLIP-derived features within attention modules, and Style3D (Song et al., 2024) modifies self-attention keys/values to propagate style across
views. Score-distillation strategies instead leverage diffusion-based style objectives to fine-tune neural rendering pipelines (e.g., StyleTex (Xie et al., 2024)). While effective in limited settings, these
paradigms often (i) struggle to disentangle style from semantic content—leading to texture leakage
and degraded geometry/appearance fidelity—or (ii) require computationally expensive, per-asset
optimization, limiting scalability.


To address the lack of explicit “style–texture” image pairs for training 3D stylization models, we
revisit what constitutes a style reference. Conventional practice treats a “reference style image” as a
natural image that entangles global semantics (object layout, parts, viewpoint) with style attributes
(color palette, stroke-like texture, frequency statistics), which in turn requires substantial variation in
both content and style for effective learning. We instead posit that an effective style reference should
convey style independently of semantics, and that local image patches are sufficient carriers of such
statistics (Wang et al., 2023). Building on this insight, we introduce a jigsaw transform that randomly
shuffles and sparsely masks non-overlapping patches, destroying global structure while preserving
local style cues. This enables us to synthesize style–texture supervision from textured 3D assets.
Concretely, given a textured asset (e.g., from Objaverse (Deitke et al., 2023b)), we render multiview images, apply the jigsaw transform to one rendered view to obtain a semantics-agnostic style
reference, and use the remaining views as supervision targets. This procedure yields large quantities of pseudo-paired data without requiring curated style–texture pairs. We then train a multi-view
stylized image generator conditioned on the style reference and geometry. The style image is encoded by a pretrained text-to-image (T2I) diffusion network; intermediate activations are extracted
as disentangled style conditions. These conditions guide a multi-view diffusion model built on a
U-Net backbone that integrates three complementary attention mechanisms: (a) self-attention for
intra-view coherence, (b) multi-view attention to enforce cross-view consistency, and (c) reference
attention that injects the style conditions to perform dynamic, style-adaptive feature recombination.
In addition, geometric signals (normal and position maps) are processed by a conditional encoder
and injected into the U-Net’s spatial features to respect object geometry during generation. By construction, the jigsawed reference suppresses content leakage while retaining style statistics, enabling
explicit content–style disentanglement and scalable training without per-asset optimization or manually paired 3D style datasets. Finally, we style-bake the multi-view outputs into a textured asset:
reproject stylized views to the UV atlas with visibility/z-tests and fuse per-texel observations via
seam-aware, confidence-weighted blending to obtain albedo. We then bake tangent-space normals
and complete missing texels with UV-space inpainting, yielding a complete, cross-view-consistent
texture.


Experiments demonstrate that our method attains state-of-the-art results on multiple 3D stylization
benchmarks and generalizes to partial stylization, multi-object scene styling, and tileable texture
generation, without per-asset optimization.


The main contributions of this work are summarized as follows:


    - **Jigsaw-based** **style** **reference** **construction.** We introduce a semantics-destroying jigsaw transform—spatial patch shuffling with random masking—that disentangles style from
content and synthesizes style–texture pseudo-pairs from textured 3D assets for supervised
stylization training.

    - **Reference-attention for stylization.** We design a trainable reference-attention module that
injects disentangled style conditions to enable dynamic, style-aware feature recombination.


2 RELATED WORK


**3D Texture Generation.** 3D texture generation aims to create visually consistent and semantically
meaningful texture maps for 3D objects. A central challenge in this task lies in achieving multi

2


view geometric consistency and maintaining texture coherence across different viewpoints. Some
approaches relied on optimization-based methods (Poole et al., 2022; Lin et al., 2023) that leverage pre-trained 2D diffusion models (Rombach et al., 2022) through score distillation sampling,
suffering high computational costs. More related to our work are methods focusing on novel view
generation with T2I models. These approaches build upon text-to-image diffusion models (Rombach et al., 2022), which provide a strong prior of 2D image appearance, and extend them to generate
geometrically consistent multi-view images. Zero-1-to-3 (Liu et al., 2023a) serves as a foundational
model that predicts novel views from a single image using viewpoint-conditioned diffusion. MVDream (Shi et al., 2023) extends this further by injecting camera parameters into the self-attention
mechanism, enabling explicit 3D awareness and cross-view consistency. SyncDreamer (Liu et al.,
2023b) introduces synchronized multi-view generation through feature-level fusion across views,
while MV-Adapter (Huang et al., 2024) employs lightweight adapter modules to efficiently fine-tune
pre-trained T2I models for multi-view synthesis. These methods commonly integrate camera embeddings or geometric constraints to maintain multi-view consistency while leveraging large-scale
3D data (Deitke et al., 2023b;a) for training.


**Image-Guided** **Stylization.** Image-guided stylization aims to transfer the style from a reference
image to a target while preserving its semantic structure. A central challenge lies in effectively
representing and transferring style features. Early approaches typically relied on statistical summarization of deep features, such as Gram matrices (Gatys et al., 2016) or channel-wise mean and
variance alignment (Huang & Belongie, 2017; Lu et al., 2019). With recent advancements, research
in this area has evolved along two main directions: 2D and 3D style transfer approaches.


In the domain of 2D style transfer, the rise of diffusion models has spurred the development of
various fine-tuning strategies. These include full model fine-tuning (Zhang et al., 2023b; 2022),
lightweight adapter-based approaches (Wang et al., 2023; Mou et al., 2024; Ye et al., 2023) that
insert trainable modules into pre-trained networks, and low-rank adaptation (LoRA) (Hu et al., 2022;
Frenkel et al., 2024) that captures style characteristics via weight updates. More recently, attentionbased methods have attracted increasing interest. Among these, StyleAligned (Hertz et al., 2024)
ensures consistent style across generated images by sharing self-attention and aligning the query
and key features of target images with a reference via AdaIN (Huang & Belongie, 2017). Visual
Style Prompting (Jeong et al., 2024) enables training-free style transfer by replacing the key and
value features in the target’s self-attention. Building on these attention designs, StyleAdapter (Wang
et al., 2023) reduces semantic interference by removing feature class tokens and shuffling positional
embeddings. Although the shuffling strategy in (Wang et al., 2023; Gu et al., 2018) indicates that
style information can be preserved within feature patches, it has not been applied to 3D stylization
tasks.


Research progress in 3D-aware stylization remains considerably limited compared to 2D stylization.
Previous NeRF-based approaches (Fujiwara et al., 2024) typically depend on multi-view images and
require per-asset optimization. StyleTex (Xie et al., 2024) decomposes style diffusion loss via orthogonal projection in a semantic-aware feature space, yet its test-time rendering optimization incurs
significant computational overhead. Other training-free methods inject style features through attention fusion. Style3D (Song et al., 2024) directly transfers self-attention features from a 2D reference
image to multi-view generation. 3D-style-LRM (Oztas et al., 2025) integrates style information
through linear combinations of CLIP-based reference features.


In contrast to existing methods, we introduce a jigsaw-based disentanglement strategy to create styletexture pairs, enabling the training of dynamic style-aware feature recombination. To the best of our
knowledge, our work presents the first approach to incorporate image-jigsaw for 3D stylization.


3 METHODS


Our method first constructs style-texture pairs from existing 3D assets to serve as training data for
the framework (Sec. 3.1). Subsequently, it operates to generate multi-view stylized images and
bakes these views into a stylized 3D object (Sec. 3.2, see Figure 2).


3


|Encoder<br>Style U-net<br>Normal & Position<br>Self Attention Attention<br>Block<br>Multi-view Attention C<br>Res Cross<br>R Ae tf te er ne tn ioc ne Q K V Projection|Col2|Col3|
|---|---|---|
|**Encoder**<br>Normal & Position<br>**C**<br>**Self Attention**<br>**Cross Attention**<br>**Multi-view Attention**<br>**Reference**<br>**Attention**<br>**Q**<br>**K V**<br>**Res Block**<br>**Style U-net**<br>**Projection**|||
|**Encoder**<br>Normal & Position<br>**C**<br>**Self Attention**<br>**Cross Attention**<br>**Multi-view Attention**<br>**Reference**<br>**Attention**<br>**Q**<br>**K V**<br>**Res Block**<br>**Style U-net**<br>**Projection**|||


|Col1|Classifier Score<br>Gram Matrix|
|---|---|
|||
|||
|||
|||
|||
|||


Figure 3: Analysis of style-content disentanglement through patch shuffling and masking. We
apply different degrees of shuffling and a fixed mask ratio. **Left:** Quantitative evaluation of content
and style attributes under increasing shuffle intensity. As _N_ (number of divisions per image side)
increases, the CNN-based classification score (blue line) of shuffled images decreases sharply. At
_N_ = 8, semantic content is almost entirely lost. Meanwhile, the Gram matrix similarity (Gatys
et al., 2016) (denoted as green dashed line) calculated between shuffled images and source images
increases gradually for _N_ _≤_ 8, indicating well-preserved style fidelity. The setting _N_ = 8 strikes
a good balance between semantic suppression and style preservation. **Right:** Visual examples of
shuffled images using different values of _N_ and a fixed mask ratio.


4


Figure 2: **Our Method Pipeline.** The whole framework contains multi-view stylized image generation and 3D style baking. **Multi-View Style Generation:** position and normal maps from the mesh
_M_ are encoded and injected into a style U-Net via feature modulation, while the reference image _I_
is processed by a jigsaw operation involving image patch shuffling and random masking to extract
style features. These style features are sent to a pre-trained reference U-Net to extract intermediate
features that serve as keys and values in a reference attention module. Our style U-Net uses reference attention for aligning with the reference style and multi-view attention to ensure cross-view
consistency. **3D Style Baking:** The generated multi-view images are projected onto the mesh’s UV
space, yielding a seamless UV map ready for final rendering.


3.1 STYLE-TEXTURE PAIRS CREATION


Unlike heavy score distillation-based approaches (Fujiwara et al., 2024; Song et al., 2024), our
method adopts a data-driven manner by constructing a style-3D dataset, enabling the model to acquire stylization transfer capabilities through supervised training. However, current large-scale 3D
datasets such as Objaverse (Deitke et al., 2023b) typically exhibit complex representations where
semantic content and style attributes are intricately entangled within texture maps, making style extraction particularly challenging. A critical initial step involves developing an effective approach to
disentangle style information from texture maps.


1.0


0.8


0.6


0.4


0.2


0.0


1 2 4 8 16 32
Number of Divisions per Side (N)


N=1 (Source) N=2 N=4 N=8 N=16 **…**


35


30


25


20


15


10


5


0


**Disentanglement** **Motivation.** Generally disentanglement requires finding a common representation to express variations among different statistical dimensions. It has been observed in 2D style
transfer that image patches can serve as effective carriers of style information (Wang et al., 2023).
Based on this insight, we further posit that deliberately shuffling and masking image patches can
disrupt object structures and suppress global semantics. At the same time, such patch-level shuffling
preserves first- and second-order style statistics ( _e.g._ mean and variance) (Huang & Belongie, 2017).
This behavior is further quantitatively demonstrated in Figure 3, where beyond a certain shuffling intensity, semantic content is largely eliminated while style fidelity remains well-preserved. Motivated
by these observations, we introduce a **jigsaw** **operation** to perform style-content disentanglement
and construct our style-texture pairs.


**Jigsaw** **Operation.** As shown in Figure 2, for a given reference image _I_ _∈_ R _[C][×][H][×][W]_, we first
partition it into a grid of non-overlapping patches _Pi,j_, where each patch has size _S_ _× S_ . These
patches are then shuffled using a permutation function _σ_, which randomly reassembles them to
disrupt structural semantics:


      _I_ shuffled = _Pσ_ ( _i,j_ ) _,_ where each _Pi,j_ _∈_ R _[C][×][S][×][S]_ (1)


_i,j_


This shuffling operation suppresses semantic information while preserving style attributes. We further apply stochastic masking with a mask ratio _p_ to control the proportion of patches that are
masked:


where _Mi,j_ is a binary mask with elements drawn from Bernoulli(1 _−_ _p_ ), and _µ_ is the masking background value. Similar to He et al. (2022), we encourage the remaining visible regions to reconstruct
the styles of the masked patches.


During training, we use the jigsaw operation to process the current 3D object from the Objaverse
to create style-texture pairs. For each 3D object, we render _K_ orthogonal views as texture targets,
along with several additional random views as reference images. Each reference image is processed
through the jigsaw operation to obtain _I_ jigsaw as model input, while the original texture targets serve
as ground truth supervision. During inference, the reference image is the provided by the user.
We apply shuffling operation to reference image to produce the input for stylization. Our model is
trained only on the pseudo-paired dataset and remains frozen during inference.


3.2 MULTI-VIEW STYLE GENERATION


After processing the reference image with the jigsaw operation, we create a large style-texture pair
dataset and train a multi-view style generation model using this dataset. As illustrated in Figure 2,
the multi-view generation aims to produce multi-view consistent images, combining the geometric
structure of _M_ and the stylistic attributes of _I_ .


**Geometric information injection** . To ensure the generated multiview images retain structural information with _M_, we leverage geometric cues from _M_ and inject them into the denoising U-Net.
Specifically, we first render both position and normal maps from the _M_ from _K_ predefined camera
viewpoints, following the setup of Li et al. (2023) and Bensadoun et al. (2024). These maps are
concatenated along the channel dimension to form the geometry condition **G** _∈_ R _[B][×]_ [2] _[K][×][H][×][W]_,
where the 2 _K_ channels comprise _K_ position maps and _K_ normal maps. Then the condition **G** is
processed by a trainable condition encoder _G_ based on T2I-Adapter (Mou et al., 2024), which consists of a series of convolutional and downsampling layers. The resulting multi-scale features are
injected directly into the corresponding scales of the Style U-Net denoiser through additive feature
modulation, providing persistent geometric guidance throughout the denoising process.


**Style** **information** **injection** . To transfer style from the reference image _I_, we first apply the jigsaw operation to suppress semantic information and disentangle style features, resulting in _I_ jigsaw.
This processed _I_ jigsaw is then encoded through a pre-trained VAE encoder _E_ to obtain latent features, which are fed into a pre-trained diffusion U-Net at timestep _t_ = 0. We extract intermediate
hidden state features _f_ ref from the self-attention layers of this U-Net, which subsequently guide the
stylization process in the style U-Net. Our diffusion model employs a style U-Net architecture that


5


_I_ jigsaw = Mask ( _I_ shuffled) = 

_i,j_


- _Mi,j_ _⊙_ _Pσ_ ( _i,j_ ) + (1 _−_ _Mi,j_ ) _⊙_ _µ_ - (2)


incorporates a multi-branch attention block (Huang et al., 2024) after each residual layer. These
blocks consist of three parallel attention mechanisms: self-attention captures contextual relationships within each view; multi-view attention enforces consistency across different viewpoints using
row-wise self-attention (Li et al., 2024); and reference attention aligns the generation with the reference style by attending to _f_ ref, as formalized below.


**Reference** **Attention.** We employ cross-attention for style transfer. In this module, the original
input feature map _f_ in serves as the query, while _f_ ref serves as both the key and value. The reference
attention operation is defined as:


The softmax output represents relevance scores between the input features _f_ in and style features
_f_ ref, enabling dynamic style-aware recombination. After computing the three attention outputs,
the results are summed with the original input feature _f_ in. The combined representation is further
refined through a text-conditioned cross-attention layer. During training, ground-truth text captions
with random dropout are used to improve robustness; during inference, generic prompts such as
“high quality” are employed to maintain generalization and output quality.


3.2.1 3D STYLE BAKING


3D Style Baking projects the pre-generated multi-view stylized images onto the UV texture space
to produce fully textured 3D objects. This baking process consists of three main steps: **Visibility-**
**aware** **reprojection** establishes accurate pixel-to-UV correspondences while filtering occluded or
invalid regions using camera and depth information; **3D inpainting** fills missing or invisible regions
by computing a weighted average of the nearest neighboring pixels on the object surface. **Seamless**
**composition** performs 2D inpainting in UV space to eliminate seam artifacts and ensure texture
continuity.


4 EXPERIMENTS


**Implementation Details.** Our approach is built upon Stable Diffusion XL (Rombach et al., 2022).
During training, we render each object from the Objaverse (Deitke et al., 2023b) to generate 6
orthogonal views as ground-truth and 4 random views as reference images. All images are scaled to
a resolution of 512 _×_ 512. In the jigsaw operation, the reference image is split into patches of size
64 _×_ 64 during training and 128 _×_ 128 during inference. A mask ratio between 0 and 0.25 achieves a
balance between prediction capability and geometric consistency. For model configuration, we apply
a combined conditioning dropout strategy with a probability of 0.1, which independently drops the
text condition, the image condition, or both simultaneously. The model is optimized using AdamW
with a learning rate of 5 _×_ 10 _[−]_ [5] for 10 epochs. We employ a DDPM sampler with 50 denoising steps
during inference, with classifier-free guidance scale set to 3.0. Additionally, we adjust the log-SNR
offset by log( _n_ ) where _n_ = 6 is the number of views.


**Evaluation** **Dataset.** For 3D objects, we select 20 objects from Objaverse (Deitke et al., 2023b)
covering diverse categories, including both flat-surfaced and geometrically sharp shapes. Importantly, all selected meshes are distinct from those used during training to ensure a fair evaluation.
For reference images, we first select style images from WikiArt (WikiArt, 2014), and additionally
collect supplementary images manually from public sources. The **WikiArt** **dataset** includes 30
style images, with 5 examples each from 6 artistic genres: cityscape, figurative, flower painting,
landscape, marina, and still-life. Furthermore, **our collected dataset** contains 40 extra images from
the internet and existing publications to cover a broader spectrum of styles, such as Chinese ink
painting, bronze/gold effects, Van Gogh-style art and cartoon illustrations. All images used comply
with the Creative Commons Attribution 4.0 International (CC BY 4.0) license.


**Evaluation Metrics.** We employ several metrics to quantitatively evaluate performance between 6
orthogonal rendered views and the reference image. **Gram Matrix Similarity** and **AdaIN Distance**
serve as style-fidelity measurements. Specifically, we extract style features from a pre-trained VGG19 network. The Gram matrix similarity is calculated using the Frobenius norm between the style
correlation matrices, and the AdaIN distance is computed as the sum of the _L_ 2 norms between the


6


RefAttention( _f_ in _, f_ ref) = softmax - _f_ ~~_√_~~ in _f_ ref _T_
_dk_


_f_ ref (3)


Mesh Reference Mv-adapter Styletex 3D-style-LRM Ours


Figure 4: **Qualitative comparison between 3D stylization methods on our collected dataset and**
**WikiArt** . The left side of the dashed line displays the input object mesh and reference image. On
the right, four groups of comparative results are shown, and each group has two selected viewpoints.


feature means and standard deviations. **CLIP** **Score** is used to measure style-content disentanglement. A lower CLIP score indicates weaker semantic correlation and more effective disentanglement
of style information from the reference. The final metrics are reported as averages across multi-view
computations.


**Baseline** **Methods.** We compare our approach against several state-of-the-art methods in 3D texture/style generation. The selected baselines include two feed-forward generation methods (Oztas
et al., 2025; Huang et al., 2024) as well as one SDS-based optimization method (Xie et al., 2024).
For a fair comparison, all methods use the same reference image and object mesh. Among baselines,
**3D-style-LRM** (Oztas et al., 2025) generates initial multi-view images using InstantMesh (Xu et al.,
2024) and fuses style information by blending attention outputs from both the original multi-view
images and the style reference within the cross-attention module. We provide the required source
images that maintain strict alignment with the input mesh. **StyleTex** (Xie et al., 2024) disentangles
style from reference images through orthogonal projection in the CLIP embedding space and guides
texture generation iteratively using a diffusion-based style loss. **MV-Adapter** (Huang et al., 2024)
employs a text-to-image diffusion model enhanced with adapter-based feature injection to produce
multi-view consistent images with unified texture and style.


4.1 QUALITATIVE AND QUANTITATIVE COMPARISONS


**Qualitative Comparison.** Figure 4 presents a qualitative comparison of different methods for transferring reference styles onto geometric meshes. The results show that 3D-Style-LRM exhibits significant limitations in preserving the geometric fidelity of the original mesh, resulting in inconsistent
surfaces. All baseline methods suffer from style infidelity compared to the reference image, such
as color shifts in the clothing (row 3) or loss of texture patterns in the sofa (row 4). Additionally,
MV-Adapter incorrectly transfers the entire texture map layout onto the target object. In contrast,
our method demonstrates superior visual quality, with color distribution and texture details consistent with the reference. Furthermore, our disentanglement and recombination strategy effectively


7


**Collected Data** **WikiArt**
**Method** **Cost Time**
**Gram** _↓_ **AdaIN** _↓_ **CLIP** _↓_ **Gram** _↓_ **AdaIN** _↓_ **CLIP** _↓_


StyleTex (TOG 2025) 5.35 124.54 **0.205** 6.54 149.27 **0.208** 15min
MV-Adapter (ICCV 2025) 4.85 114.04 0.214 4.91 122.19 0.213 _∼_ 40s
3D-style-LRM (SIGGRAPH 2025) 5.78 136.22 0.215 5.49 139.86 0.210 _∼_ 35s


**ours** **4.81** **113.38** 0.213 **4.82** **120.54** 0.210 _∼_ 40s


Table 1: **Quantitative comparison between 3D stylization methods on our collected dataset and**


Figure 5: **Ablation study on the Jigsaw module.** The left side shows the input object mesh and
reference style image. The right side presents groups of stylization results under different Jigsaw
settings: **(a) w/o Train & Infer Jigsaw** : training and reference process without jigsaw operation; **(b)**
**w/o** **Infer** **Jigsaw** : only inference process without jigsaw operation; **(c)** **w/ Train** **& Infer** **Jigsaw**
**(Ours)** : our approach applies the jigsaw operation in both training and inference phases.


applies style attributes to semantically appropriate structures. As shown in row 2, our approach
applies a floral pattern to the roof of a building while maintaining pure-colored walls, enabling the
assignment of distinct styles to different structural components.


**Quantitative Comparison.** Table 1 presents a quantitative comparison of 3D stylization methods.
As demonstrated, our method achieves significantly superior performance on style-related metrics
including Gram matrix similarity and AdaIN, indicating exceptional style consistency with the reference and multi-view coherence. Furthermore, our approach attains competitive CLIP scores, second
only to StyleTex, which effectively demonstrates successful semantic disentanglement. The slightly
lower CLIP performance compared to StyleTex can be attributed to the fact that StyleTex utilizes
style text descriptions as additional input conditions, providing ground-truth information. Besides,
compared to SDS-based methods like StyleTex, our approach is more efficient in terms of computational time.


4.2 ABLATION STUDY


**Ablation Study on Jigsaw Module.** Figure 5 presents an ablation study evaluating the effectiveness
of the jigsaw module under different configurations. Figure (a) shows that without the jigsaw operation, the first row lacks the flower style, while the second row exhibits semantic entanglement of the
elephant. When applying the jigsaw module during training as shown in Figure (b), the flower style
is attributed to the house roof and the elephant semantics are compressed. Furthermore, applying
jigsaw during both training and inference, as shown in Figure (c), preserves better style fidelity to
the reference images, as demonstrated by the color distribution of the elephant and the detailed floral
patterns on the house.


We further provide additional ablation studies, including those on the **patch size** _S_ and **mask ratio**
_p_ in the jigsaw operation, in the Section A.5.


8


Figure 6: The first figure demonstrates strong geometric awareness, as the generated **sketch lines**
**align precisely with the 3D feature lines of the objects** . Across all four figures, local regions also
exhibit reasonable style attribution, as the house roof and wall maintain color consistency with the
reference images.


Figure 7: **Limitations of our method.** The dashed red boxes indicate failure cases in style transfer,
such as text or symbolic patterns that do not match the reference image.


4.3 MORE APPLICATIONS AND LIMITATIONS


**Multiple Objects Scene Stylization** . We collected several scene-related 3D objects and applied our
method to stylize them. As shown in Figure 6, our approach successfully achieves consistent style
coherence both within individual objects and across different objects in the scene.


We further demonstrate other applications, including **tileable texture generation** and **partial styl-**
**ization**, in the appendix Section A.6.


**Limitations.** During the style transfer process, our method struggles to accurately preserve finegrained patterns such as text or symbols, as shown in Figure 7. This limitation is attributable to
Stable Diffusion backbone (SDXL) we used, which lacks the capability to reliably generate or reconstruct precise textual and symbolic structures.


5 CONCLUSION


In this paper, we propose a new framework to transfer the style of a reference image to a 3D object
while preserving its geometric structure. We propose a novel framework centered around the jigsaw operation to create a style-3D pair dataset. We apply a multiview generation pipeline to utilize
the extracted style features from the jigsawed images. Geometric cues such as normal and position
maps are further incorporated to enhance structural alignment. Extensive experiments demonstrate
that our method achieves state-of-the-art performance across several 3D stylization benchmarks
and shows strong generalization to downstream tasks such as partial object stylization, multi-object
scene styling, and tileable texture generation.


9


**Ethical** **Statement.** Our work presents a method for 3D style transfer using multi-view diffusion
models and a jigsaw-based disentanglement mechanism. We acknowledge that our approach could
potentially be misused to create misleading or harmful content, such as generating stylized 3D objects that infringe upon intellectual property, appropriate cultural or artistic styles without permission, or propagate visual misinformation. We strongly emphasize that the use of this technology
should adhere to the relevant legal frameworks and community standards. All 3D assets and reference images should be properly licensed or used in accordance with fair use principles.


**Reproducibility** **Statement.** To facilitate the reproducibility of our work, we provide code in the
supplementary materials. A detailed README document is included, which covers the environment
configuration and instructions for loading pre-trained weights.


10


REFERENCES


Raphael Bensadoun, Yanir Kleiman, Idan Azuri, Omri Harosh, Andrea Vedaldi, Natalia Neverova,
and Oran Gafni. Meta 3d texturegen: Fast and consistent texture generation for 3d objects. _arXiv_
_preprint arXiv:2407.02430_, 2024.


Angel X Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li,
Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, et al. Shapenet: An information-rich 3d
model repository. _arXiv preprint arXiv:1512.03012_, 2015.


Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan
Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of
10m+ 3d objects. _Advances in Neural Information Processing Systems_, 36:35799–35813, 2023a.


Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig
Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _conference_ _on_ _computer_ _vision_ _and_ _pattern_
_recognition_, pp. 13142–13153, 2023b.


Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation
using b-lora. In _European Conference on Computer Vision_, pp. 181–198. Springer, 2024.


Haruo Fujiwara, Yusuke Mukuta, and Tatsuya Harada. Style-nerf2nerf: 3d style transfer from stylealigned multi-view images. In _SIGGRAPH Asia 2024 Conference Papers_, pp. 1–10, 2024.


Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional
neural networks. In _Proceedings of the IEEE conference on computer vision and pattern recog-_
_nition_, pp. 2414–2423, 2016.


Shuyang Gu, Congliang Chen, Jing Liao, and Lu Yuan. Arbitrary style transfer with deep feature
reshuffle. In _Proceedings of the IEEE conference on computer vision and pattern recognition_, pp.
8222–8231, 2018.


Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pp. 16000–16009, 2022.


Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation
via shared attention. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern_
_Recognition_, pp. 4775–4785, 2024.


Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _ICLR_, 1(2):3, 2022.


Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In _Proceedings of the IEEE international conference on computer vision_, pp. 1501–1510,
2017.


Zehuan Huang, Yuan-Chen Guo, Haoran Wang, Ran Yi, Lizhuang Ma, Yan-Pei Cao, and
Lu Sheng. Mv-adapter: Multi-view consistent image generation made easy. _arXiv_ _preprint_
_arXiv:2412.03632_, 2024.


Jaeseok Jeong, Junho Kim, Yunjey Choi, Gayoung Lee, and Youngjung Uh. Visual style prompting
with swapping self-attention. _arXiv preprint arXiv:2402.12974_, 2024.


Peng Li, Yuan Liu, Xiaoxiao Long, Feihu Zhang, Cheng Lin, Mengfei Li, Xingqun Qi, Shanghang
Zhang, Wei Xue, Wenhan Luo, et al. Era3d: High-resolution multiview diffusion using efficient
row-wise attention. _Advances in Neural Information Processing Systems_, 37:55975–56000, 2024.


Weiyu Li, Rui Chen, Xuelin Chen, and Ping Tan. Sweetdreamer: Aligning geometric priors in 2d
diffusion for consistent text-to-3d. _arXiv preprint arXiv:2310.02596_, 2023.


Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. Sparc3d: Sparse representation
and construction for high-resolution 3d shapes modeling. _arXiv preprint arXiv:2505.14521_, 2025.


11


Chia-Hao Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis,
Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_,
pp. 300–309, 2023.


Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Carl Vondrick, and Ali Fathi. Zero-1to-3: Zero-shot one image to 3d object. In _Proceedings of the IEEE/CVF International Conference_
_on Computer Vision_, pp. 22026–22037, 2023a.


Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang.
Syncdreamer: Generating multiview-consistent images from a single-view image. _arXiv preprint_
_arXiv:2309.03453_, 2023b.


Ming Lu, Hao Zhao, Anbang Yao, Yurong Chen, Feng Xu, and Li Zhang. A closed-form solution to
universal style transfer. In _Proceedings of the IEEE/CVF international conference on computer_
_vision_, pp. 5952–5961, 2019.


Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In _Proceedings of the IEEE/CVF_
_conference on computer vision and pattern recognition_, pp. 4460–4470, 2019.


Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and
Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. _Communications_
_of the ACM_, 65(1):99–106, 2021.


Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan.
T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion
models. In _Proceedings of the AAAI conference on artificial intelligence_, volume 38, pp. 4296–
4304, 2024.


Ipek Oztas, Duygu Ceylan, and Aysegul Dundar. 3d stylization via large reconstruction model.
In _Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques_
_Conference Conference Papers_, pp. 1–11, 2025.


Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove.
Deepsdf: Learning continuous signed distance functions for shape representation. In _Proceedings_
_of the IEEE/CVF conference on computer vision and pattern recognition_, pp. 165–174, 2019.


Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d
diffusion. _arXiv preprint arXiv:2209.14988_, 2022.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF Con-_
_ference on Computer Vision and Pattern Recognition_, pp. 10684–10695, 2022.


Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view
diffusion for 3d generation. _arXiv preprint arXiv:2308.16512_, 2023.


Bingjie Song, Xin Huang, Ruting Xie, Xue Wang, and Qing Wang. Style3d: Attention-guided
multi-view style transfer for 3d object generation. _arXiv preprint arXiv:2412.03571_, 2024.


Zhouxia Wang, Xintao Wang, Liangbin Xie, Zhongang Qi, Ying Shan, Wenping Wang, and Ping
Luo. Styleadapter: A unified stylized image generation model. _arXiv preprint arXiv:2309.01770_,
2023.


WikiArt. The online visual art encyclopedia. [https://www.wikiart.org/, 2014.](https://www.wikiart.org/)


Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao
Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. _Advances in_
_Neural Information Processing Systems_, 37:121859–121881, 2024.


Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen,
Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation.
In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_ _Conference_, pp. 21469–21480,
2025.


12


Zhiyu Xie, Yuqing Zhang, Xiangjun Tang, Yiqian Wu, Dehan Chen, Gongsheng Li, and Xiaogang
Jin. Styletex: Style image-guided texture generation for 3d models. _ACM Transactions on Graph-_
_ics (TOG)_, 43(6):1–14, 2024.


Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh:
Efficient 3d mesh generation from a single image with sparse-view large reconstruction models.
_arXiv preprint arXiv:2404.07191_, 2024.


Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt
adapter for text-to-image diffusion models. _arXiv preprint arXiv:2308.06721_, 2023.


Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape
representation for neural fields and generative diffusion models. _ACM Transactions On Graphics_
_(TOG)_, 42(4):1–16, 2023a.


Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan
Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d
assets. _ACM Transactions on Graphics (TOG)_, 43(4):1–20, 2024.


Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image
diffusion models. In _Proceedings of the IEEE/CVF international conference on computer vision_,
pp. 3836–3847, 2023b.


Yuxin Zhang, Fan Tang, Weiming Dong, Haibin Huang, Chongyang Ma, Tong-Yee Lee, and Changsheng Xu. Domain enhanced arbitrary image style transfer via contrastive learning. In _ACM_
_SIGGRAPH 2022 conference proceedings_, pp. 1–8, 2022.


Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng,
Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for
high resolution textured 3d assets generation. _arXiv preprint arXiv:2501.12202_, 2025.


13


A APPENDIX


A.1 USE OF LARGE LANGUAGE MODELS (LLMS)


We declare that Large Language Models (LLMs) were used solely as an auxiliary tool in the writing
process of this paper, specifically for tasks such as checking and correcting grammatical errors and
ensuring consistency in formatting and terminology. We emphasize that the core ideas, theoretical
derivations, experimental design, and result analysis were entirely conceived and conducted by the
authors.


A.2 PROOF OF STYLE STATISTICS PRESERVATION UNDER SHUFFLE OPERATION


Let _I_ _∈_ R [3] _[×][H][×][W]_ denote an input image with 3 channels, height _H_, and width _W_ . We partition _I_
into _N_ _× N_ non-overlapping patches _{Pi,j}_, each of size _[H]_ _N_ _[×]_ _[W]_ _N_ [.] [Let] _[ I]_ [shuffled] [be the image after]

applying a permutation _σ_ to these patches:


Hence,
( _σc_ [2][)] _[′]_ [=] _[ σ]_ _c_ [2]


According to the above, shuffling patches changes the spatial arrangement of pixels but does not
alter their first-order (mean) or second-order (variance) statistics.


A.3 STYLE CONSISTENCY METRICS.


To quantitatively evaluate style consistency between generated multi-view images and the reference
style image, we employ two widely adopted perceptual style evaluation metrics: **Gram** **Matrix**
**Similarity** and **AdaIN Distance** . These are considered perceptual metrics because they operate on
deep feature representations rather than pixel-level values, thereby aligning with human perception


14


_I_ shuffled = - _Pσ_ ( _i,j_ )


_i,j_


**Mean Preservation.** The mean of channel _c_ of the original image is:


_W_

- _I_ ( _c, i, j_ )


_j_ =1


1
_µc_ =
_HW_


_H_


_i_ =1


After shuffling, only the spatial positions of pixel values are permuted. Therefore, the sum over all
positions remains identical:


_W_

- _I_ ( _c, i, j_ ) =


_j_ =1


_W_

- _I_ shuffled( _c, i, j_ ) = 1

_HW_
_j_ =1


_H_


_i_ =1


_W_

- _I_ shuffled( _c, i, j_ )


_j_ =1


Thus,


_H_


_i_ =1


_W_

- _I_ ( _c, i, j_ ) = _µc_


_j_ =1


_W_


1
_µ_ _[′]_ _c_ [=]
_HW_


_H_


_i_ =1


_H_


_i_ =1


_W_


**Variance Preservation.** The variance of channel _c_ is defined as:


_W_

- ( _I_ ( _c, i, j_ ) _−_ _µc_ ) [2]


_j_ =1


1
_σc_ [2] [=]
_HW_


_H_


_i_ =1


Since both the pixel values _I_ ( _c, i, j_ ) and the mean _µc_ remain unchanged under shuffling, each
squared term ( _I_ ( _c, i, j_ ) _−_ _µc_ ) [2] is preserved. Therefore, the sum of squared deviations remains
the same:
_H_ _W_ _H_ _W_

   -    - [2]    -    - _[′]_ [2]


_W_


_W_


_H_


_i_ =1


_i_ =1


- ( _I_ ( _c, i, j_ ) _−_ _µc_ ) [2] =


_j_ =1


- ( _I_ shuffled( _c, i, j_ ) _−_ _µ_ _[′]_ _c_ [)][2]


_j_ =1


of artistic style that relies on texture patterns and statistical characteristics. Features are extracted
from five key ReLU layers (1 ~~1~~, 2 ~~1~~, 3 ~~1~~, 4 ~~1~~, 5 ~~1~~ ) of a pre-trained VGG-19 network.


**Gram** **Matrix** **Similarity** captures the correlations between feature channels. For a feature map
**F** _∈_ R _[C][×][H][×][W]_, the Gram matrix **G** _∈_ R _[C][×][C]_ is computed as:


1
**G** =
_C · H_ _· W_ **[FF]** _[⊤]_

The style similarity between the reference image and a generated view is measured using the Frobenius norm:
_L_ Gram = _∥_ **G** ref _−_ **G** gen _∥F_


**AdaIN Distance** measures the discrepancy in first-order (mean) and second-order (standard deviation) statistics of deep features:


_Lµ_ = _∥µ_ ( **F** ref) _−_ _µ_ ( **F** gen) _∥_ 2 _,_ _Lσ_ = _∥σ_ ( **F** ref) _−_ _σ_ ( **F** gen) _∥_ 2
_L_ AdaIN = _Lµ_ + _Lσ_


Both metrics are computed across five VGG-19 layers and averaged over all views. Lower values indicate better style consistency. Together, these perceptual metrics provide a comprehensive
assessment of style transfer quality at multiple feature levels.


A.4 MORE RESULTS


**Further** **Qualitative** **Results** **of** **Our** **Methods.** We present additional multi-view rendering results of our method in Figure 8 and Figure 9. Figure 8 illustrates the outcomes of stylizing a
house object using a collection of reference images with distinct artistic styles. Figure 9 extends
this by applying different stylistic references to a diverse set of objects. Both sets of results highlight our method’s ability to maintain strong stylistic consistency across multiple viewpoints. Besides, Figure 10 presents our method’s generalization capability by applying celebrated artistic styles
to amusement park scene objects. For artistic stylization, we select representative works including
Monet’s Water Lilies, Piet Mondrian’s compositions, Edvard Munch’s The Scream, Van Gogh’s The
Starry Night, and Ukiyo-e prints. The results show stylistic consistency and overall harmony.


**Further Qualitative Comparisons with SOTA.** Figure 11 and Figure 12 present qualitative comparisons with other methods, including Hunyuan-V2 (Zhao et al., 2025) as an additional baseline.


A.5 MORE ABLATION STUDY


**Ablation on Mask Ratio** _p_ **in equation 2.** In Figure 13, we conduct an ablation study to analyze
the impact of the mask ratio. The mask ratio is designed to encourage the model to learn diverse and
enhanced feature representations from masked reference images. The results show employing an
appropriate mask ratio can enhance the diversity of the learned feature expressions, while setting it
too high (e.g., 0.75) proves detrimental. An excessive mask ratio leads to lost geometric information
with generated multi-views, which in turn causes an uneven appearance with artifacts during the
baking process.


**Ablation on Training and Inference Patch Size** _S_ **in equation 1.** The image patch serves as the
primary carrier of style information, making its size a critical hyperparameter. Figure 14 showcases
the qualitative rendering results of our model under different patch size configurations. A training
patch size that is too small may not sufficiently contain the style information, whereas a size that is
too large can cause the model to focus excessively on content details. We find that for a majority of
images, a training patch size of 64 _×_ 64 coupled with an inference patch size of 128 _×_ 128 produces
the stable results. This is likely due to the nature of the data domain, where this configuration strikes
an optimal balance.


A.6 MORE APPLICATION RESULTS


**Partial** **Object** **Stylization.** To evaluate our method’s ability to understand and transfer style attributes from limited visual cues, we conduct experiments using only cropped regions of the reference image. As shown in Figure 15, our approach can successfully infer a globally consistent


15


Views


Reference

Image


Figure 8: **More 3D Stylization Results of our method.** Multi-view rendering results by applying
diverse reference style images to a “house” object.


style from a partial reference and apply it coherently to the full 3D object. This demonstrates the
robustness of our model in handling partial style references while maintaining semantic and stylistic
coherence.


**Tileable Texture Generation.** We further demonstrate that our method is not capable of transferring
styles but also can create consistent texture tiles from an example image. As shown in Figure 16, in
comparison with the direct texture mapping in Blender with noticeable irregularities and artifacts,
especially around UV seams, our jigsaw-based strategy disrupts spatial biases while maintaining
strong consistency and continuity across the entire 3D surface.


A.7 VISUALIZATION ON OUR SELF-COLLECTED REFERENCE IMAGES


The visualization of our self-collected reference images is presented in Fig. 17, which includes
images from the internet and existing publications to cover a broader spectrum of styles, such as
Chinese ink painting, bronze and gold effects, Van Gogh-style art, and cartoon illustrations.


16


Figure 9: **More 3D Stylization Results of our method.** Multi-view rendering results by applying
various reference styles to a wide range of object meshes from the Objaverse dataset.


A.8 VISUALIZATION OF DISENTANGLEMENT RESULTS


**Disentanglement results between Jigsaw3D and MV-Adapter** are shown in Fig. 18, on flat mesh
surfaces, MV-Adapter frequently overlays both the semantic content and style of the reference image
onto the target geometry. In contrast, our approach effectively disentangles and transfers only the
stylistic attributes, eliminating interference from irrelevant semantic elements in the reference.


A.9 FAILURE CASES IN UV BAKING AND IMPROVEMENT STRATEGY


**Failure** **Cases** **in** **UV** **Baking** **and** **Improvement** **Strategy.** Fig. 19 shows the limitations of UV
baking in handling occluded regions and our proposed solution. During multi-view generation,
we generate 2 random viewpoints to ensure comprehensive texture coverage in the UV inpainting


17


Reference Object
Render Results


Image


Object


Mesh


Figure 10: **Additional** **qualitative** **results** **of** **our** **method** **applied** **to** **park** **scene** **objects** **and**
**artistic styles.**


process, which effectively resolves missing or incomplete texture regions. This enhancement consistently produces more coherent and complete surface textures.


A.10 COMPARISON WITH 3DGS-BASED STYLIZATION METHODS


The comparison with 3DGS-based stylization methods is shown in Fig. 20. For our method, we
first generate a scene-level mesh using Hunyuan3D (Zhao et al., 2025) with three different views
as input. The mesh and reference image are then processed by our Jigsaw3D framework to render the output 3D asset as stylized scene images. For 3DGS stylization, we provide 120 multiview images to StyleGaussian for Gaussian splatting reconstruction and stylization. Comparative
results demonstrate that StyleGaussian fails to preserve reference texture patterns and exhibits texture incompleteness, whereas our approach achieves superior scene-style consistency and maintains
coherent stylization across objects.


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


Hunyuan-v2 3D-LRM Styletex Ours


Figure 11: **Qualitative comparison of multi-view stylization results.** Results below the dashed
line show outputs from different methods across multiple viewpoints.


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


Figure 12: **Qualitative comparison of multi-view stylization results.** Results below the dashed
line show outputs from different methods across multiple viewpoints.


20


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


Multi-view generation results (Stage 1)
Reference Image


Reference Image


Figure 13: **Ablation study on the mask ratio** _p_ **.** This figure illustrates the impact of different mask
ratio settings on our two-stage process. The panels on the right side of the dashed line show the
multi-view generation results and the final rendered results. Each row corresponds _p_ set to a specific
ratio: 0.0, 0.25, 0.5, and 0.75, from top to bottom.


21


Render results (stage 2)


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


training and inference. The entire image size is fixed at 512 _×_ 512, and the image patch size is set to
_S × S_ . From left to right, the columns show rendering results using training patch sizes of 32 _×_ 32,
64 _×_ 64, and 128 _×_ 128, respectively. From top to bottom, the rows correspond to inference patch
sizes of 32 _×_ 32, 64 _×_ 64, 128 _×_ 128.


Figure 15: **Partial** **stylization** **results.** For reference image, we preserve the partial region (as
presented as red box) and mask other regions for partial inference.


22


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


Figure 16: **Comparison with conventional renderers on tileable texture generation.** Our method
effectively eliminates seam artifacts compared to Blender, which often produces inconsistent texture
directionality across UV boundaries.


Figure 17: **Visualization on our self-collected reference images.**


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


Figure 18: **Disentanglement results between Jigsaw3D and MV-Adapter.** MV-Adapter tends to
transfer the entire reference appearance directly to the target mesh, whereas our method successfully
isolates and extracts only the style from the reference image.


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Figure 19: **Failure Cases in UV Baking and Improvement Strategy.** By generating complementary random views, our method provides more complete texture coverage and effectively addresses
missing or obscured areas via robust texture completion.


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


##### Multi-view Input Images Reference Image StyleGaussian Ours


Figure 20: **Comparative results between our method and 3DGS-based stylization approaches.**


26
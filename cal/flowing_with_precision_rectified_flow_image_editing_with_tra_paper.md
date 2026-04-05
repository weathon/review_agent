# FLOWING WITH PRECISION: RECTIFIED FLOW IMAGE EDITING WITH TRAJECTORY AND FREQUENCY GUID
## ANCE


**Anonymous authors**
Paper under double-blind review


cat dog wearing earrings Remove paraglider Cartoon Style


Figure 1: Our method is designed to modify high level attributes to match the prompt, all while
maintaining the core structure and background of the source image.


ABSTRACT


Rectified flow text-to-image models have shown remarkable progress. However,
editing complex scenes containing multiple objects remains challenging due to
semantic entanglement and structural inconsistency. To address this, we propose
a dual-domain framework that jointly refines temporal editing trajectories and
adapts frequency domain. First, we design a **Starting Point Optimization (SPO)**
**strategy**, which intelligently determines the optimal editing starting point based
on the structural complexity of different images. Second, we introduce a **Tra-**
**jectory Optimization (TO) strategy** . In the time domain, it performs semanticaware vector orthogonalization to suppress source bias while preserving target
semantics. In the frequency domain, it adaptively re-weights high and low frequency residuals according to stage-specific spectral characteristics. Furthermore,
we leverage the frequency-aware capabilities of MM-DiT to dynamically inject
structural priors from the source image at different denoising steps. Our method
allows users to add, replace, or modify multiple objects, making it highly efficient
for editing complex scenes. Experiments show that our method significantly outperforms existing methods for image editing and achieving higher user preference
in human evaluations.


1


bird crochet bird cat owl opened eyes cat closed eyes dog cat dog

rabbit squirrel

branch gold branch bean bag chair pumpkin wooden floor green grass


cat
rabbit


cat
bean bag chair


owl
pumpkin


opened eyes cat closed eyes dog


wooden floor green grass


black
stones


colorfor

stones


shack castle
pure blue Milky Way sky


pickup rusty pickup
green trees yellow trees


tree sunflower
full moon crescent moon


1 INTRODUCTION


The goal of image editing is to align a region of interest with text prompt while preserving the nonedited areas. Recently, Rectified Flow (RF) models (Labs, 2024) have demonstrated superior performance over diffusion models (DMs) (Rombach et al., 2022; Tang et al., 2023; Wang et al., 2024;
Song et al., 2022) in both image quality and text alignment, by leveraging flow matching (Lipman
et al., 2023; Liu et al., 2022) and a multi-modal diffusion transformer (MM-DiT) (Esser et al., 2024;
Peebles and Xie, 2023; Huang et al., 2024). However, their effectiveness is limited when it comes to
fine-grained editing of specific, detailed regions within images containing multiple objects or complex scenes. In practice terms, we aim to design a powerful and effective framework that excels at
image editing across a diverse range of image types.


The challenge in image editing is to precisely edit multi-object at specific locations within an image.
Existing text-guided image editing methods that leverage flow models (Wang et al., 2025; Deng
et al., 2024; Avrahami et al., 2025) operate by inverting the source image into a latent space and
then performing conditional denoising. However, such methods often lead to significant deviations
between the edited and original images (Xie et al., 2025; Kulikov et al., 2025). While some approaches utilize attention modification to enhance control (Xu et al., 2024; Tewel et al., 2024; Lv
et al., 2025), they often face challenges. While attention-based methods are good at preserving the
original image structure, this strong control can also restrain the editing strength. When multiple
objects are present, these approaches often suffer from appearance leakage (Zhang et al., 2025; Sun
et al., 2025). Current multi-object editing methods (Zhu et al., 2025a; Yang et al., 2024; Huang
et al., 2025) rely on masks, such methods often struggles to precisely bind specific attributes to their
geometrically defined regions, which can lead to inaccurate edits or content leakage.


To overcome the above challenges, we revisit the editing process. Unlike DM, RF allows the latent
noise to be inferred at each time step through linear interpolation. FlowEdit builds an ODE between the source and target images without inversion. Inspired by this, we propose an inversion-free
method that modifies specific regions directly on the source image at each step. We clearly divide the
editing process in the noise space into three distinct stages: the **Chaos Phase**, the **Layout Phase**,
and the **Refinement Phase** . Our key finding is that beginning the edit too early can compromise the
source image’s structural integrity, whereas starting too late may result in an ineffective edit. This
is further complicated by the fact that the structural complexity of different images, leading to a
non-uniform end point for the Chaos phase. To solve this issue, we introduce **Starting Point Opti-**
**mization (SPO)**, a method that adaptively determines the optimal editing start point by calculating
the low frequency Mean Squared Error (MSE) between the source and target images.


Although the SPO strategy enhance structural fidelity, it still suffers from obvious visual artifacts and
insufficient editing strength. To this end, we propose a trajectory optimization strategy. This strategy
decomposes the editing direction into two orthogonal components, cross-cue and cross-track, in the
time domain, and eliminates its projection in the cross-cue direction by orthogonalizing the crosstrack term. In the frequency domain, we employ dynamic frequency weighting to adaptively adjust
the editing strength based on the frequency characteristics of each denoising stage. Furthermore,
we leverage the frequency-aware properties of MM-DiT. Instead of applying the same strategy to
all attention layers (Feng et al., 2025), we select and utilize specific attention layers at different
diffusion stages to inject corresponding source image features.


In summary, our main contributions are as follows:


(a) We propose a novel **SPO** strategy that adaptively selects the optimal editing onset for different images based on their structural complexity.


(b) We introduce a **Trajectory Optimization** method. This method performs optimization in
both time and frequency domains. Additionally, it selectively injects structural features
from the source image into appropriate attention layers of the MM-DiT, based on the current denoising step.


(c) We evaluate our method across various editing tasks,as shown in Figure 1. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art baselines
in single-object editing tasks and achieves competitive results in multi-object editing scenarios.


2


attention injection,scale takes different values at different phases.


2 RELATED WORK


2.1 INVERSION FOR IMAGE EDITING


Image editing methods fall into two main categories: inversion-based and inversion-free. Inversion
methods obtain the initial noise by iteratively adding predicted noise (Brack et al., 2024; Deutch
et al., 2024). Null-text Inversion (Mokady et al., 2022) achieves a more precise inverse recovery by
training null-text embeddings. Negative Prompt Inversion (Miyake et al., 2024) replaces the negative prompt with the source prompt. RF models still reverse the ODE to gradually add noise. RFInversion (Rout et al., 2024) constructs a controlled ODE through source image and noise interpolation. RF-solver (Wang et al., 2025) and FireFlow (Deng et al., 2024) introduce second-order Taylor
expansions to reduce reconstruction errors. Early inversion-free methods such as SDEdit (Meng
et al., 2022) strike a balance between realism and fidelity by adding moderate noise, Delta Denoising Score (Hertz et al., 2023) refines text-guided edits by subtracting noise gradients using a
reference-guided image-text pair. Infedit (Xu et al., 2023) uses a special variance schedule such that
the denoising step takes the same form as multi-step consistency sampling. Recently, FlowEdit (Kulikov et al., 2025) and FlowAlign (Kim et al., 2025) circumvents inversion by constructing a direct
flow drive between the source and target images.


2.2 MULTI-OBJECT IMAGE EDITING


Balancing structural preservation and semantic alignment in multi-object editing remains a challenging task. Early methods leveraged U-Net based DMs (Jiang et al., 2025; Simsar et al., 2024)
focusing on attention modification. P2P (Wang et al., 2022) manipulated cross-attention maps for
feature-prompt alignment, PnP (Tumanyan et al., 2022b) injected aligned internal controls, and
MastCtrl (Cao et al., 2023) achieved edits while preserving overall texture and consistency. Recent methods (Sanjyal, 2025) introduce refined control strategies. OIR (Yang et al., 2024) separates
editing pairs with masks and distinct inversion steps, LoMOE (Chakrabarty et al., 2024) restricts
edits to specified mask regions through a multi-diffusion process, and Paralleledits (Huang et al.,
2025) employs parallel branches for a editing strategy. While RF models primarily (Rombach et al.,
2022; Labs, 2024) utilize the MM-DiT architecture, this domain remains underexplored. Existing
methods (Deng et al., 2024; Zhu et al., 2025b; Avrahami et al., 2025) adapt attention injection to
preserve fidelity, while others (Xu et al., 2025) use Adaln to manipulate features for control. The
MM-DiT architecture is an area still requiring deeper investigation.


3


**MM-DiT**


**MM-DiT**


�푠���


�����


Figure 2: **Pipeline** **overview** **of** **our** **method** . (a) **Trajectory** **Optimization** . _λ_ type( _ti_ ) = 1 +


 _α_ 1 _−_


( _kx,ky_ ) _[|][U][t]_ _i_ [(] _[k][x][,k][y]_ [)] _[|]_


( _kx,ky_ ) _∈R_ type _[|][U][t]_ _i_ [(] _[k][x][,k][y]_ [)] _[|]_


_,_ where type _∈{_ low _,_ high _}_ . (b) **Token** **mapping** based cross

Figure 3: **Analyse on intermediate results** . In the early stages of the denoising process, predictions
are mainly guided by textual cues, and _Vtar_ [src] [and] _[ V]_ _src_ [src] [have high similarity.] [Later, image comes into]
play, and _Vtar_ [src] [and] _[ V]_ _src_ [src] [have high similarity.]


3 METHOD
Our method is able to achieve semantic alignment while effectively maintaining the consistency of
image structure. Our method mainly consists of three parts: (i) based on the analysis of frequency
changes during denoising, We propose a novel **Start** **Point** **Optimization** **(SPO)** strategy; (ii) introduce a **Frequency aware trajectory optimization** method; (iii) introduces our feature injection
process along with attention adaptation methods to enhance editability. An overview of our method
is presented in Fig. 2.


3.1 PRELIMINARY


3.1.1 RECTIFIED FLOW BASED MODELS.


Rectified flow models learn the probability paths between two distributions. Specifically, they _lin-_
_early interpolate_ between two observed distributions **x** 0 _∼_ _p_ 0 and **x** 1 _∼_ _p_ 1, and model such probability transport paths using the following ordinary differential equation (ODE):


**x** _t_ = _t_ **x** 1 + (1 _−_ _t_ ) **x** 0 _,_ _t ∈_ [0 _,_ 1] _,_ (1)

_d_ **x** _t_ = _vθ_ ( **x** _t, t_ ) _dt,_ **x** 0 _∼_ _p_ 0 _,_ _t ∈_ [0 _,_ 1] _,_ (2)
where _vθ_ ( **x** _t, t_ ) denotes a time-aware velocity field governing the transport dynamics. The training
objective is to directly regress the velocity field using the least-squares loss:


                    _L_ = E _t∼U_ [0 _,_ 1] _,_ **x** 1 _∼p_ 1 _∥_ ( **x** 1 _−_ **x** 0) _−_ _vθ_ ( **x** _t, t_ ) _∥_ [2][�] _._ (3)


3.1.2 INVERSION-FREE TEXT-BASED EDITING.
In text-based image editing using flow models, we aim to translate a source image _X_ [src] to a target
image _X_ [tar] based on the text description of each image or the editing instruction. In particular, such
translation can be represented through a linear conditional flow between two image distributions:

_Xt_ [edit] = _Xt_ [tar] _[−]_ _[X]_ _t_ [src] + _X_ 0 [src] _[.]_ (4)

Consequently, we can simulate the ODE for image editing by

_dXt_ [edit] = _V_ edit( _t_ ) = _v_ ( _Xt_ [tar] _[, t, c]_ [tar][)] _[ −]_ _[v]_ [(] _[X]_ _t_ [src] _[, t, c]_ [src][)] _[.]_ (5)
_dt_


4


3.2 OBSERVATIONS ON PHASED EDITING IN NOISE SPACE


Previous work (Yu et al., 2023; Bao et al.,
2025) have shown that diffusion models naturally emphasize different frequency components at different sampling timesteps. Inspired
by this, we analyze intermediate results during the editing process at different denoising
timesteps, we categorize the image editing process in the noise space into three distinct stages:
**Chaotic** **Phase**, **Layout** **Phase**, and **Refine-**
**ment Phase** .


To address this, we orthogonalize the **cross-**
**prompt** term. By retaining only the component that is independent of the **cross-trajectory**
term, we avoid redundant superposition and
achieve a more precise and controllable edit.
Specifically, we define the cross-prompt vector as ∆ _ts_ and the cross-trajectory vector as
∆ _ss_ . The orthogonalized cross-trajectory vector ∆ _[orth]_ _ts_ is computed as:


_′_
∆ _ts_ [= ∆] _[ts]_ _[−]_ [∆] _[ts][ ·]_ [ ∆] _[ss]_ [∆] _[ss]_ (7)

_∥_ ∆ _ss∥_ [2]


where ∆ _ts_ and ∆ _ss_ represent the cross-prompt
and cross-trajectory vectors, respectively.


_′_
_Vedit_ [(] _[t]_ [) = ∆] _[ss]_ [+] _[ ω][ ·]_ [ ∆] _[orth]_ _ts_ (8)


**Chaotic** **Phase:** We introduce intermediate
variables _V_ ( _Xt_ [tar] _[, t, c]_ [src][)][.] [As] [shown] [in] [Fig.] [3,] Figure 4: CosSim1 is the cosine similarity of
during the early stage of the denoising process, _Vtar_ [src] [)] [and] _[V]_ _tar_ [tar] [,] [CosSim2] [is] [the] [cosine] [similarity]
the predictions are primarily guided by the text of _Vtar_ [src] [and] _[V]_ _src_ [src] [)][.] _[T]_ [0] [is] [the] [point] [where] [the] [low]
prompt, containing very little structural and se- frequency and overall values of CosSim2 are first
mantic information about the source image. By equal.
analyzing the cosine similarity between the predictions and the target at different frequency components as shown in Fig. 4, we identify a key point _T_ 0. From this point, the image information begins
to guide the predictions, enabling the model to effectively reconstruct the source image’s structure.
Consequently, we propose a novel **SPO** strategy and define this transition point as the optimal starting timestep, which adaptively determines the starting point based on an image’s characteristics.


**Layout** **Phase** **and** **Refinement** **Phase:** As illustrated in Fig. 3, after skipping the chaotic phase,
_V_ src is able to reconstruct the source image layout with high fidelity. The editing result then mainly
depends on the accuracy of the denoising direction _V_ edit. We found that there is an intersection _Tturn_
between CosSim1 and CosSim2. Before this, it is mainly responsible for low frequency layout, and
after this, it is responsible for high frequency refinement. We use this observation to delineate the
**Layout Phase** and **Refinement Phase** .


3.3 TRAJECTORY OPTIMIZATION


3.3.1 SEMANTIC AWARE VECTOR DECOUPLING


After determining the editing start point, we use Eq. (6) to build a direct ODE process between the
source and target distributions. However, without explicit latent inversion, the generated _V_ _[edit]_ ( _t_ )
vector remains heavily constrained by the source structure, which limits the overall editing strength.


_V_ _[edit]_ ( _t_ ) = _v_ ( _Xt_ _[tar]_ _, t, ctar_ ) _−_ _v_ ( _Xt_ _[tar]_ _, t, csrc_ )

     - ��      cross-prompt


+ ( _v_ ( _Xt_ _[tar]_ _, t, csrc_ ) _−_ _v_ ( _Xt_ _[src]_ _, t, csrc_ ))

 - ��  cross-trajectory


(6)


Source Ours Flowedit Turboedit Orthogonalized


fruit—>pizza


bird —> chicken  tree—>bamboo


Figure 5: Qualitative comparison. Orthogonal
calculation helps maintain background structure,
and frequency domain control helps increase editing power.


5


3.3.2 FREQUENCY AWARE TRAJECTORY OPTIMIZATION


In multi-object editing scenarios, we found that simply amplifying the **cross-prompt** term does not
effectively improve editing quality and can instead introduce unnecessary distortion, as shown in
Fig. 5. By performing a frequency domain analysis of _Vedit_, we observed that the distribution of its
high and low frequency components changes dynamically with the timestep. Therefore, we propose
a **frequency-adaptive scaling strategy** to more precisely control the editing process.


Specifically, we first transform the editing vector into the frequency domain and apply adaptive
scaling to its low- and high-frequency components. This process is defined as:


_U_ ∆ _ti_ = [ _λlow_ ( _ti_ ) _· Mlow_ + _λhigh_ ( _ti_ ) _· Mhigh_ ] _⊙F_ (∆ _ti_ ) (9)

where _F_ denotes the Fourier transform, and _Mlow_ and _Mhigh_ are binary masks isolating the low
and high frequency components, respectively. _λlow_ ( _ti_ ) and _λhigh_ ( _ti_ ) are adaptive frequency scaling
coefficients. The transformed vector is then converted back to the time domain:
_V_ ˆ∆ _ti_ = _F_ _[−]_ [1] ( _U_ ∆ _ti_ ) (10)

The frequency scaling coefficient _λtype_ ( _ti_ ) is computed based on the relative energy concentration
within the residual spectrum _U_ ∆ _ti_ . This mechanism, controlled by parameter _α_, compensates for
missing frequency information and balances the contribution of low- and high-frequency components during image editing. The coefficient is defined as:


Inspired by (Hertz et al., 2022), we propose an

with subsequent text alignment.

attention reconstruction method to correct these
errors. The MMDiT joint attention module can be decoupled into four core components: I2I-SA,
I2T-CA, T2I-CA, and T2T-SA. To evaluate their roles in image editing, we separately inject each
component into the target branch. Our experiments showed that while both I2I-SA and I2T-CA can
effectively preserve source image structure, I2I-SA interferes with subsequent text alignment. We
therefore selected I2T-CA as the final choice.


For the _j_ -th token in the target prompt, if a corresponding source token exists with an index of _ϕ_ ( _j_ ),
we reuse the cross-attention value from the source as _Bϕ_ ( _j_ ); otherwise, we amplify the original
attention _Aj_ by a dynamic scaling factor _β_ . Here, _Aj_ denotes the source’s cross-attention (CA)
value computed for the _j_ -th token in the target prompt, and _Bϕ_ ( _j_ ) is the corresponding attention
value from the source prompt at index _ϕ_ ( _j_ ). The final attention fusion is formulated as:

_A_ _[′]_ _j_ [=] _[ ω][j]_ _[·][ B]_ _ϕ_ ( _j_ ) [+ (1] _[ −]_ _[ω][j]_ [)] _[ ·][ β][ ·][ A][j][,]_ (12)


where _ωj_ = 1 if a mapping exists, and _ωj_ = 0 otherwise. The mapping function _ϕ_ ( _j_ ) assigns the
index of the source prompt token corresponding to the _j_ -th target token.


The amplification factor _β_ is dynamically adjusted based on the denoising timestep _t_ :

_β_ = �1 _,_ if _t < T_ turn _,_ (13)
_β_ 0 _,_ if _t ≥_ _T_ turn _._


This dynamic adjustment mechanism ensures that the source’s CA prevents editing leakage or artifacts during the layout phase and the target’s CA increases the editing intensity during the refinement
phase. Furthermore, we found that each layer block of MMDiT also has a related frequency pattern. Please see the appendix for details. For both the layout and refinement phases, we select the
corresponding attention layers to inject.


6


( _kx,ky_ ) _∈Rtype_ _[|][U][t]_ _i_ [(] _[k][x][, k][y]_ [)] _[|]_


( _kx,ky_ ) _∈Rall_ _[|][U][t]_ _i_ [(] _[k][x][, k][y]_ [)] _[|]_


_λtype_ ( _ti_ ) = 1 + _α_


1 _−_


(11)


where _type ∈{low, high}_, _Rlow_ and _Rhigh_ denote the low and high frequency regions.


3.4 ATTENTION REMAPPING.

Source I2I SA I2T CA T2I CA T2T SA w/o Injection


We construct the latent variable _Zt_ by linearly
interpolating between a clean image and Gaussian noise. However, as shown in the figure,
a conflict between the source image’s structure
and the target prompt can lead to reconstruction
errors such as **editing leakage** .


Figure 6: Edited results generated by injecting
features. I2I-SA and I2T-CA can effectively preserve source image structure, I2I-SA interferes
with subsequent text alignment.


a bowl of fry rice (dumpling) on the left and knife (chopsticks) on the right


a round painting of a forest with ( deer ), flowers, trees, ( rocks ), and ( a cat )


Figure 7: **Qualitative** **Comparison** . Unlike existing methods, our method allows users to add,
replace, or modify multiple objects, making it highly efficient for editing complex scenes.


4 EXPERIMENT


4.1 IMPLEMENTATION DETAILS


4.1.1 DATASETS.


For single-object editing, we evaluate our method and baseline methods on 9 tasks from PIEBench (Ju et al., 2023). For multi-object edit, we evaluate our method on three multi-object datasets:
PIE-Bench++ (Huang et al., 2025), an augmented version of PIE-Bench designed for mixed edits
involving 2-3 object categories; and the OIR (Yang et al., 2024) include mixed edits across two
different task types. All three datasets provide paired sources and target prompts.


4.1.2 METRICS.


We evaluated our method from two perspectives: (a) source preservation and (b) text alignment. For
source preservation, we measured Structure Distance Tumanyan et al. (2022a), PSNR Huynh-Thu
and Ghanbari (2008), LPIPS Zhang et al. (2018), MSE, and SSIM. Note that the numbers of these


7


metrics reported in this paper are scaled. For text alignment, we measure the CLIP similarity Radford et al. (2021) between the whole image and the target prompt (Whole) and


4.1.3 BASELINES AND IMPLEMENTATION DETAILS


We mainly compare our method with previous state-of-the-art training-free image editing methods.
For RF-based models, we evaluate RF-Inversion (Rout et al., 2024), RF-solver (Wang et al., 2025),
FireFlow (Deng et al., 2024), Stable Flow (Avrahami et al., 2025), and FlowEdit (Kulikov et al.,
2025). For DMs, we evaluated DDIM+P2P (Hertz et al., 2022), Direct Inversion+PnP (Tumanyan
et al., 2022b), Infedit (Xu et al., 2023), MasaCtrl (Cao et al., 2023). In addition, we also include
multi-object editing method OIR (Yang et al., 2024). We follow their official implementations for
evaluation. We implement our method based on FLUX (Labs, 2024). Throughout the comparative
evaluations, our hyperparameters remain fixed: _β_ = 4 for cross attention injection. Specifically, during the Layout Phase, cross attention from the source image is injected into layers 5-20 of the target
branch. During the Refinement Phase, the injection occurs in layers 20-45. Further implementation
details are provided in Appendix.


4.2 EDITING RESULTS


4.2.1 QUALITATIVE EVALUATION.


Qualitative results are shown in Fig. 7. From our experiments, we observe the following: Firstly,
methods such as StableFlow, Fireflow, and RF-inversion often exhibit omitted edits when dealing
with complex scenes involving multiple materials, colors, or object modifications, leading to noticeable text-image misalignment. Secondly, for multi-object editing approaches like OIR, their
performance is suboptimal in non-rigid editing tasks, such as object addition or removal. This is
primarily because these methods heavily rely on precise masks for editing, which makes it challenging to generate natural and contextually consistent results when an object needs to be completely
removed or created from scratch. Thirdly, FlowEdit demonstrates better editing performance in certain scenarios; however, it overlooks the variations in inversion steps across different images, which
can lead to suboptimal editing outcomes.


**Structure** **Background Preservation** **CLIP Similarity**


**Method** **Distance** _**×**_ **10** **[3]** _**↓**_ **PSNR** _**↑**_ **LPIPS** _**×**_ **10** **[2]** _**↓**_ **MSE** _**×**_ **10** **[3]** _**↓**_ **SSIM** _**×**_ **10** **[2]** _**↑**_ **Whole** _**↑**_ **Edited** _**↑**_


DDIM+P2P 69.43 17.87 20.88 21.99 71.14 25.01 22.44
DI+PnP 24.29 22.46 10.61 8.045 79.68 25.41 **22.62**
InfEdit 13.78 **28.51** 4.758 3.209 85.66 25.03 22.22
MasaCtrl 28.07 22.18 10.15 8.677 80.26 24.96 21.40
RF-Inversion 32.62 22.03 15.96 9.601 73.26 24.89 21.89
RF-Solver 24.17 26.12 11.88 4.064 86.50 25.19 22.07
StableFlow 14.41 25.98 7.246 4.471 92.08 24.20 20.86
FireFlow 22.42 25.91 11.45 4.396 86.56 25.41 22.08
FlowEdit 21.07 23.59 8.889 6.631 88.89 24.90 21.66


Ours **8.754** 28.50 **4.143** **2.111** **94.69** **25.44** 22.00


Table 1: Comparison with different baselines for single-object edits in PIE benchmark. The best
score is highlighted in **bold**, and the second-best score is underlined.


4.2.2 QUANTITATIVE EVALUATION.


For single-object edits, the quantitative results are summarized in Table 1. Our method performs
well on most metrics, with the exception of the edited CLIP score and PSNR. Although DI+PnP
obtains a higher CLIP score, its ability to preserve the background and structure of the source image
is inferior to that of our method. For multi-object edits, the quantitative results are summarized in
Table 2. Our method achieves good performance on most metrics. Specifically, while InfEdit shows
better PSNR, its CLIP similarity is significantly lower than other methods. This suggests that its
strong background preservation capability hinders its editing ability. FlowEdit achieves a high CLIP
score but shows weak background and structure preservation. In summary, our method performs
well in both background preservation and editing, enabling precise editing without compromising
structural consistency or editability. Quantitative results for OIR are shown in the Appendix.


8


**Structure** **Background Preservation** **CLIP Similarity**


**Method** **Distance** _**×**_ **10** **[3]** _**↓**_ **PSNR** _**↑**_ **LPIPS** _**×**_ **10** **[2]** _**↓**_ **MSE** _**×**_ **10** **[3]** _**↓**_ **SSIM** _**×**_ **10** **[2]** _**↑**_ **Whole** _**↑**_ **Edited** _**↑**_


DDIM+P2P 43.57 18.48 18.83 19.01 73.55 20.72 19.58
DI+PnP 27.07 22.73 10.32 7.597 80.73 20.79 19.07
InfEdit 22.60 24.61 10.40 16.05 78.85 24.69 22.63
MasaCtrl 29.76 22.50 10.22 8.758 81.59 24.15 22.14
RF-Inversion 42.29 21.41 18.19 11.24 73.70 25.06 23.29
RF-Solver 23.83 26.65 11.523 3.778 87.26 24.46 22.80
StableFlow 16.43 26.38 6.582 4.038 91.76 22.79 21.48
FireFlow 20.49 27.06 10.50 3.854 88.21 24.52 22.80
FlowEdit 22.52 24.00 8.638 6.206 89.78 25.11 23.43
OIR 24.66 27.79 **5.715** 2.460 86.97 23.56 20.95


Ours **12.82** **27.84** 6.523 **2.430** **91.83** **25.65** **23.55**


Table 2: Comparison with different baselines for multi-object edits in PIEBench++. The best score
is highlighted in **bold**, and the second-best score is underlined.


4.3 ABLATION STUDY


We conducted an ablation study on three core
technical components of our method: starting Source Ours w/o SPO w/o Injection w/o TO
point optimization, feature injection, and difference modulation. Table 3 presents quantitative results. Without starting point optimization, a significant portion of the source struc- mockup of a wooden frame with a white ~~(red)~~ rose on a white marble background
ture is lost in the edited images. This may stem
from extracting features from the latent space
that contain inaccurate information about the
source image. Meanwhile, without feature in- an origami rabbit wearing glasses lay on a newspaper ~~(book)~~
jection, the edits fail to fully align with the target prompt, leading to a lower text-alignment
score. Furthermore, without trajectory optimization(TO), edits maintain high image sim- a little girl wearing ( hat ~~) sunglasses~~ and a gray (dress ~~)shirt~~ leaning against a wall
ilarity but yield lower CLIP scores, potentially

Figure 8: Ablation examples for assessing the im
due to insufficient editing strength. These find
pact of each technique in our method.

ings collectively underscore the importance of
both source structure preservation and semantic alignment in image editing, objectives that our
method effectively achieves. Fig. 8 provides a qualitative comparison. As long as the correct starting
point is determined, the original image structure can be maintained. The other two strategies will
improve the text alignment.


**Structure** **Background Preservation** **CLIP Similarity**


**Method** **Distance** _**×**_ **10** **[3]** _**↓**_ **PSNR** _**↑**_ **LPIPS** _**×**_ **10** **[3]** _**↓**_ **MSE** _**×**_ **10** **[3]** _**↓**_ **SSIM** _**×**_ **10** **[2]** _**↑**_ **Whole** _**↑**_ **Edited** _**↑**_


w/o SPO 17.26 25.82 8.072 4.057 90.10 23.78 22.28
w/o Injection 15.00 26.14 7.749 3.374 90.11 24.54 22.70
w/o Modulation 13.45 27.72 6.256 2.650 91.06 24.24 22.65


Ours **12.82** **27.84** **5.523** **2.430** **91.83** **25.65** **23.55**


Table 3: **Ablations Study** . Three strategies com plement each other and result in improved metrics.


5 CONCLUSION


In conclusion, we analyze the intermediate features at each timestep of inversion-free methods,
and propose a simple yet effective starting point optimization strategy. In addition, we introduce
Trajectory Optimization method to address the editing omission and detail loss problems in multiobject and complex scene editing. We hope that our analysis of start point and frequency will serve
as a building block for future advancements in image editing.


9


Source Ours w/o SPO w/o Injection w/o TO


mockup of a wooden frame with a white ~~(red)~~ rose on a white marble background


an origami rabbit wearing glasses lay on a newspaper ~~(book)~~


a little girl wearing ( hat ~~) sunglasses~~ and a gray (dress ~~)shirt~~ leaning against a wall


Figure 8: Ablation examples for assessing the impact of each technique in our method.


REFERENCES


Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel
Cohen-Or. Stable flow: Vital layers for training-free image editing, 2025.


Yuxiang Bao, Huijie Liu, Xun Gao, Huan Fu, and Guoliang Kang. Freeinv: Free lunch for improving ddim
inversion, 2025.


Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting,
and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models, 2024.


Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuningfree mutual self-attention control for consistent image synthesis and editing, 2023.


Goirik Chakrabarty, Aditya Chandrasekar, Ramya Hebbalaguppe, and Prathosh AP. Lomoe: Localized multiobject editing via multi-diffusion, 2024.


Yingying Deng, Xiangyu He, Changwang Mei, Peisong Wang, and Fan Tang. Fireflow: Fast inversion of
rectified flow for image semantic editing, 2024.


Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or. Turboedit: Text-based image
editing using few-step diffusion models, 2024.


Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex
Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image
synthesis, 2024. [URL https://arxiv.org/abs/2403.03206.](https://arxiv.org/abs/2403.03206)


Haoran Feng, Zehuan Huang, Lin Li, Hairong Lv, and Lu Sheng. Personalize anything for free with diffusion
transformer, 2025.


Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt
image editing with cross attention control, 2022.


Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score, 2023.


Mingzhen Huang, Jialing Cai, Shan Jia, Vishnu Suresh Lokhande, and Siwei Lyu. Paralleledits: Efficient
multi-object image editing, 2025.


Zemin Huang, Zhengyang Geng, Weijian Luo, and Guo jun Qi. Flow generator matching, 2024.


Q. Huynh-Thu and M. Ghanbari. Scope of validity of psnr in image/video quality assessment. _Electronics_
_Letters_, page 800, Jan 2008. doi: 10.1049/el:20080522. URL [http://dx.doi.org/10.1049/el:](http://dx.doi.org/10.1049/el:20080522)
[20080522.](http://dx.doi.org/10.1049/el:20080522)


Rui Jiang, Xinghe Fu, Guangcong Zheng, Teng Li, Taiping Yao, and Xi Li. Energy-guided optimization for
personalized image editing with pretrained text-to-image diffusion models, 2025.


Xuan Ju, Ailing Zeng, Yuxuan Bian, Shaoteng Liu, and Qiang Xu. Direct inversion: Boosting diffusion-based
editing with 3 lines of code, 2023.


Jeongsol Kim, Yeobin Hong, Jonghyun Park, and Jong Chul Ye. Flowalign: Trajectory-regularized, inversionfree flow-based image editing, 2025.


Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free
text-based editing using pre-trained flow models, 2025.


Black Forest Labs. Flux. [https://github.com/black-forest-labs/flux, 2024.](https://github.com/black-forest-labs/flux)


Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for
generative modeling, 2023.


Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data
with rectified flow, 2022.


Zhengyao Lv, Tianlin Pan, Chenyang Si, Zhaoxi Chen, Wangmeng Zuo, Ziwei Liu, and Kwan-Yee K. Wong.
Rethinking cross-modal interaction in multimodal diffusion transformers, 2025.


Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit:
Guided image synthesis and editing with stochastic differential equations, 2022.


10


Daiki Miyake, Akihiro Iohara, Yu Saito, and Toshiyuki Tanaka. Negative-prompt inversion: Fast image inversion for editing with text-guided diffusion models, 2024.


Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing
real images using guided diffusion models, 2022.


William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable
visual models from natural language supervision, 2021.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution
image synthesis with latent diffusion models, 2022.


Litu Rout, Yujia Chen, Nataniel Ruiz, Constantine Caramanis, Sanjay Shakkottai, and Wen-Sheng Chu. Semantic image inversion and editing using rectified stochastic differential equations, 2024.


Ankit Sanjyal. Local prompt adaptation for style-consistent multi-object generation in diffusion models, 2025.


Enis Simsar, Alessio Tonioni, Yongqin Xian, Thomas Hofmann, and Federico Tombari. Lime: Localized image
editing via attention regularization in diffusion models, 2024.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022.


Jiayang Sun, Hongbo Wang, Jie Cao, Huaibo Huang, and Ran He. Marmot: Multi-agent reasoning for multiobject self-correcting in improving image-text alignment, 2025.


Siao Tang, Xin Wang, Hong Chen, Chaoyu Guan, Zewen Wu, Yansong Tang, and Wenwu Zhu. Post-training
quantization with progressive calibration and activation relaxing for text-to-image diffusion models. Nov
2023.


Yoad Tewel, Rinon Gal, Dvir Samuel, Yuval Atzmon, Lior Wolf, and Gal Chechik. Add-it: Training-free object
insertion in images with pretrained diffusion models, 2024.


Narek Tumanyan, Omer Bar-Tal, Shai Bagon, and Tali Dekel. Splicing vit features for semantic appearance
transfer, 2022a.


Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven
image-to-image translation, 2022b.


Changyuan Wang, Ziwei Wang, Xiuwei Xu, Yansong Tang, Jie Zhou, and Jiwen Lu. Towards accurate posttraining quantization for diffusion models. Apr 2024.


Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying
Shan. Taming rectified flow for inversion and editing, 2025.


Ziyi Wang, Xumin Yu, Yongming Rao, Jie Zhou, and Jiwen Lu. P2p: Tuning pre-trained image models for
point cloud analysis with point-to-pixel prompting, 2022.


Chenxi Xie, Minghan Li, Shuai Li, Yuhui Wu, Qiaosi Yi, and Lei Zhang. Dnaedit: Direct noise alignment for
text-guided rectified flow editing, 2025.


Pengcheng Xu, Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Jiangning Zhang, Chengjie Wang,
Yunsheng Wu, Charles Ling, and Boyu Wang. Unveil inversion and invariance in flow transformer for
versatile image editing, 2025.


Sihan Xu, Yidong Huang, Jiayi Pan, Ziqiao Ma, and Joyce Chai. Inversion-free image editing with natural
language, 2023.


Yu Xu, Fan Tang, Juan Cao, Yuxin Zhang, Xiaoyu Kong, Jintao Li, Oliver Deussen, and Tong-Yee Lee. Headrouter: A training-free image editing framework for mm-dits by adaptively routing attention heads, 2024.


Zhen Yang, Ganggui Ding, Wen Wang, Hao Chen, Bohan Zhuang, and Chunhua Shen. Object-aware inversion
and reassembly for image editing, 2024.


Jiwen Yu, Yinhuai Wang, Chen Zhao, Bernard Ghanem, and Jian Zhang. Freedom: Training-free energy-guided
conditional diffusion model, 2023.


11


Liheng Zhang, Lexi Pang, Hang Ye, Xiaoxuan Ma, and Yizhou Wang. Richcontrol: Structure- and appearancerich training-free spatial control for text-to-image generation, 2025. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2507.02792)
[2507.02792.](https://arxiv.org/abs/2507.02792)


Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness
of deep features as a perceptual metric. In _2018_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition_, Jun 2018. doi: 10.1109/cvpr.2018.00068. [URL http://dx.doi.org/10.1109/cvpr.](http://dx.doi.org/10.1109/cvpr.2018.00068)
[2018.00068.](http://dx.doi.org/10.1109/cvpr.2018.00068)


Hongyang Zhu, Haipeng Liu, Bo Fu, and Yang Wang. Mde-edit: Masked dual-editing for multi-object image
editing via diffusion models, 2025a.


Tianrui Zhu, Shiyi Zhang, Jiawei Shao, and Yansong Tang. Kv-edit: Training-free image editing for precise
background preservation, 2025b.


12


A APPENDIX


THE USE OF LLM


We acknowledge the use of a large language model (ChatGPT, GPT-5 by OpenAI) to assist in improving the
clarity and readability of the manuscript. The model was used only for language polishing (e.g., grammar
checking, wording refinement) and not for generating novel scientific content, experimental design, data analysis, or results. All technical ideas, methods, and contributions are solely the work of the authors.


A. DETAILED ANALYSIS


A.1 START POINT


As shown in Fig. 9, we present three examples of images with different levels of complexity. For a 50-step
process, their starting point is approximately the 10th timestep.


_a blue and yellow (_ _**crochet)**_ _parrot is sitting on a (_ _**gold**_ _) branch_


As shown in Fig. 10, The MM-DiT architecture exhibits consistent frequency-domain characteristics across
different timesteps. The model, which consists of 57 blocks, processes information in a hierarchical manner.
As the layer number increases, the attention outputs progressively incorporate higher frequency components.
Specifically, the early layers establish the low frequency structure of the image, such as object placement
and motion, while the later blocks focus on high frequency details. Consequently, to inject low frequency
structural information, we should concentrate on the early layers. Conversely, for injecting high frequency
details, attention should be directed to the later layers.


13


_Source_


A.2 ATTENTION LAYERS


Figure 9: **Analysis of start point**


### prompt：A dog wearing a chef’s hat in a kitchen. Visualization word: dog


Figure 10: **Visual analysis of the layers of MM-DiT at each timesteps** .


B. IMPLEMENTATION DETAILS


B.1 INJECTION LAYERS


The MM-DiT architecture exhibits similar frequency-domain properties. To leverage this, we conducted ablation studies to determine the optimal attention injection strategy for both the **layout phase** and the **refinement**
**phase** .


14


As shown in Fig. 10, we observed that the first five layers contain relatively limited information. For the layout
phase, we set the injection starting point at layer 5. As can be seen from Fig. 11. Injecting source image
attention from layers 5 to 20 effectively removes editing artifacts. It is noteworthy that a narrower injection
range, such as layers 5-10 or 5-15, tends to generate unrelated objects. Conversely, extending the range to layer
25 excessively preserves source details, which constrains editing flexibility. Furthermore, if the injection starts
later at layer 10, editing artifacts reappear, proving that layers 5-10 are indispensable.


For the refinement phase as shown in Fig. 12, we found that injecting attention from the 20th block effectively enhances editing strength. This strategy aims to prevent editing omissions because the first 20 blocks
are primarily responsible for the image’s macro layout. Experiments indicate that layers 20-45 are the optimal
injection range. Using layers 20-35 still leads to editing leakage, while extending the range to 20-55 significantly limits the editing effect. Additionally, we found that injecting only into layers 5-20 does not improve
performance, and the results from layers 5-45 are similar to those from layers 20-45.


Figure 11: **Analysis of injection layers in layout phase**


B.2 BASELINE IMPLEMENTATION


In this section, we describe the implementation details of the baselines we used.
For **RF-Inversion**, we follow their Github official implementation, the stopping timestep is set to 7/28, and the
strength is set to 0.9.
For **RF-Solver-Edit**, we follow their Github official implementation, guidance is set to 2 and inject is set to 5.
For **Stable** **Flow**, we follow their Github official implementation, the vital layers are the same as the official
implementation.
For **FireFlow**, we follow their Github official implementation, the number of steps is set to 8, and the inject is
set to 1.
For **FlowEdit**, we follow their Github official implementation, and use their default setting for FLUX.
For **OIR**, we follow their Github official implementation, reinversion step and reassembly step are both set to
10.
For **InfEdit**, we follow their Github official implementation.
For **DDIM+P2P**, **DI+PnP** and **MasaCtrl**, we follow the implementations from **Direct Inversion** codebase


15


_a (_ _**dog**_ _)_ ~~_woman_~~ _jumps over (_ _**a zombie**_ _)_ ~~_rocks_~~ _at sunset_


Figure 12: **Analysis of injection layers in refinement phase**


C. ADDITIONAL QUALITATIVE EVALUATION


We performed additional qualitative comparisons against baselines. Extended comparison results on singleobject and multi-object are presented in Fig. 13 and Fig. 14, with more examples in Fig. 15 and Fig. 16.


D. ADDITIONAL QUANTITATIVE COMPARISON


We present the comprehensive results of our quantitative evaluation on the OIR-Bench and LoMOE-Bench
dataset in Tab 5 and Tab **??** . To assess how well the original image’s structure is preserved, we used several
metrics: Structure Distance, PSNR, LPIPS, MSE, and SSIM. To measure how closely the edited image aligns
with the text prompt, we computed CLIP text similarity. This was done in two ways: one for the entire image
and another specifically for the editing mask region, which we refer to as ”Whole Image Clip Similarity” and
”Edit Region Clip Similarity.”


D.1 OIR-BENCH


Our method achieves better perfomance across most metrics. Concretely, the results show that LoMOE has
a stronger ability to preserve source content, but this also limits its editing flexibility. As shown in the CLIP
scores, LoMOE performs relatively worse in terms of edited similarity. In summary, our method performs
well in both background preservation and editing, enabling accurate edits without compromising structural
consistency or editability.


D.2 DETAILS ON USER STUDY


We conducted two user studies, comparing our method against eight multi-object editing techniques, and eight
single-object editing methods. For the studies, we selected 8 images each from single-object and multi-object
editing tasks within our dataset. These images were generated from the same source image and target prompt
but using different methods. More than 20 participants are asked to select the image that best conformed to the
target prompt while effectively preserving the source image structure. As show in Table 4. We carried out a user
study to compare image editing methods, involving 28 anonymous Prolific users and 14 questions. Participants
were shown a source image and an editing instruction. Their task was to select one of eight edited images based


16


on two criteria: edit accuracy and backgroud preservation . This process, exemplified in Fig. 17, allows us to
analyze user preferences and inform the development of more effective and precise editing methods.


**Comparison on single-object edits**


Method DI+PnP MasaCtrl Infedit RF-Inversion RF-Edit FlowEdit Stableflow **Ours**
User Preference 4.3% 3.4% 8.4% 6.7% 7.3% 9.1% 10.3% **50.5%**


**Comparison on multi-object edits**


Method DI+PnP Inf-Edit RF-Inversion RF-Edit FlowEdit Stableflow OIR **Ours**
User Preference 5.6% 3.2% 6.8% 7.2% 9.8% 7.2% 5.4% **54.8%**


Table 4: User study results comparing our method with nine methods in single-object edits and
multi-object edits.


E.ADDITIONAL EXAMPLES OF ABLATION STUDIES ON EACH TECHNIQUE


E.1 EFFECT OF VARYING _β_ IN CROSS-ATTENTION INJECTION


The effect of the hyperparameter _β_ is presented in Fig. 19. With _β_ =1, the model fails to fully align the edited
image with the text prompt. Increasing _β_ significantly improves this alignment, as indicated by better comparison scores. However, an excessively large _β_ can’t increase the editing effect Therefore, selecting an appropriate
_β_ is crucial. Based on this observation, we set _β_ =4 for all subsequent experiments.


E.2 EFFECT OF VARYING START POINT


The effect of the hyperparameter start point is presented in Fig. 18. Starting editing too early will destroy the
original image structure, and starting editing too late will not align with the text prompt.


F.FUTURE WORK


We believe that image processing and video processing are inseparable from the frequency domain. Later we
will continue to explore the role of the frequency domain in the image and video fields.


**Structure** **Background Preservation** **CLIP Similarity**


**Method** **Distance** _**×**_ **10** **[3]** _**↓**_ **PSNR** _**↑**_ **LPIPS** _**×**_ **10** **[2]** _**↓**_ **MSE** _**×**_ **10** **[3]** _**↓**_ **SSIM** _**×**_ **10** **[2]** _**↑**_ **Whole** _**↑**_ **Edited** _**↑**_


DI+PnP 33.74 22.60 10.88 6.796 83.30 28.18 25.56
MasaCtrl 26.62 22.76 10.67 7.261 81.44 21.81 19.79
RF-Inversion 44.89 21.01 16.56 9.95 79.81 27.72 25.32
RF-Solver 27.02 25.17 10.39 4.25 87.45 27.23 25.14
StableFlow **16.03** 24.46 7.28 5.23 91.61 22.51 20.29
FireFlow 26.84 24.44 11.28 4.65 86.10 26.95 24.26
FlowEdit 31.59 22.81 9.256 6.92 88.43 27.07 24.70
OIR 21.56 28.65 4.608 2.342 88.11 28.74 **26.51**


Ours 18.47 **30.53** **3.295** 2.910 **92.27** **28.88** 26.09


Table 5: Comparison with different baselines for multi-object edits in OIR Bench. The best score is
in **bold**, and the second-best score is underlined.


17


_white_ ~~_tiger (_~~ _cat_ _)_ _on brown ground_


_a_ _silver_ _cat_ _sculpture_ _sitting next to a mirror_


_a dog wearing space suit (_ _with flowers in mouth_ _)_


_a cartoon woman in a red cloak walking through the woods_ ~~_with bat_~~


_a woman in a hat and dress_ ~~_walking_~~ _(_ _running_ _) down a path at sunset_


_(_ _a pixel art of_ _) a woman in a blue dress leaning against a wall_


_a_ ~~_white (_~~ _yellow_ _)_ _kitten sitting on a leopard print blanket_


Figure 13: Additional evaluation comparisons on single-object editing.


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


_a_ ~~_girl (_~~ _boy_ _) and_ ~~_her dog (_~~ _his monkey_ _)in a field_


_a_ ~~_cat_~~ _(_ _dog_ _) and_ ~~_fish_~~ _(_ _frog_ _) in a fish bowl_


_a_ ~~_monkey_~~ _(_ _man_ _) wearing colorful_ ~~_goggles_~~ _(_ _sunglasses_ _)_


_a cup of coffee (_ _with spoon_ _) and a notebook on a checkered tablecloth (_ _with a pen_ _)_


_a young boy_ ~~_standing_~~ _(_ _similing_ _) in the dirt with a jacket on (_ _looking up at the sky_ _)_


_a_ ~~_man (_~~ _robot_ _) wearing a_ ~~_shirt_~~ _(_ _sweater_ _)_


_a photo of_ ~~_couples_~~ _(_ _robots_ _) wearing_ ~~_shirts_~~ _(_ _dress_ _) dancing together_


_a cartoon painting of a cute_ ~~_owl_~~ _(_ _cat_ _) with a_ ~~_heart (_~~ _circle_ _) on its body_


Figure 14: Additional evaluation comparisons on multi-object editing.


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


Figure 15: **Diverse** **edited** **results** **of** **our** **method** **on** **single-object** **editing** . Our method allows
users to add, replace, change object, change color and change material.


20


Source Image Edited Image


_white_ _brown_


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


Source Image Edited Image Source Image Edited Image Source Image Edited Image


Figure 16: **Diverse** **edited** **results** **of** **our** **method** **on** **multi-object** **editing** . Our method allows
users to add, replace, change object, change color and change material.


21


_bird_ _crochet bird_ _cat_ _owl_ _cat_ _dog_
_branch_ _gold branch_ _bean bag chair_ _pumpkin_ _rabbit_ _squirrel_


_cat_
_rabbit_


_cat_
_bean bag chair_


_owl_
_pumpkin_


_pickup_ _rusty pickup_
_green trees_ _yellow trees_


_tree_ _sunflower_
_full moon_ _crescent moon_


_black stones_ _colorfor stones_


_bulb_
_book_


_pink_
_cotton balls_


_rabit_
_daisies_


_four gray_
_cushions_


_shack_ _castle_
_pure blue_ _Milky Way sky_


_bird_

_tree_


_chicken_
_bamboo_


_roses_
_gift box_


_clolorful_
_cotton balls_


_fox_
_roses_


_pumpkins_ _carved_
_pumpkins_


_two blue,two_
_white cushions_


_cat_
_wooden floor_


_dog_
_green grass_


_cat_ _dog_


**1134**

**1135**

**1136**

**1137**


**1138**

**1139**

**1140**


**1179**

**1180**

**1181**

**1182**

**1183**


**1184**

**1185**

**1186**

**1187**


Figure 17: Example screenshot from the user study, displaying images generated using different
methods, where participants selected the one that best represents the intended edit.


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


_a (_ _**dog**_ _)_ ~~_cat_~~ _sitting on a (_ _**pumpkin**_ _)_ ~~_bean bag chair_~~


a ( _**rusty**_ ) pickup surrounded by ( _**yellow)**_ ~~green~~ trees and houses


Figure 18: The effect of start point


_a (_ _**Lego**_ _)_ ~~_colorful v_~~ _an parked on a snowy street, next to (_ _**an ancient roman architecture**_ _)_ ~~_a building_~~


Figure 19: The effect of _β_


23


Source
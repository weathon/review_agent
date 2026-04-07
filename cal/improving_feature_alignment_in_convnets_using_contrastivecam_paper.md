## IMPROVING FEATURE ALIGNMENT IN CONVNETS US### ING CONTRASTIVECAMS AND CORE-FOCUSED CROSS## ENTROPY

**Anonymous authors**
Paper under double-blind review


ABSTRACT


Despite the ubiquity of modern deep learning, accurate explanations of network predictions remain largely elusive. HiResCAM is a popular interpretability technique
used to visualize attention maps (i.e., regions-of-interest) over input images. In this
paper, we theoretically show a limitation of HiResCAM: the HiResCAMs for a
given input are not uniquely determined, allowing an arbitrary spurious shift by a
common matrix _M_ while corresponding to the same prediction. We further propose
_ContrastiveCAMs_, which are invariant to the spurious shift _M_ hence improving
robustness of explanations, while additionally providing granular class-versus-class
explanations. With the additional granular explanations, experiments reveal that
networks often focus on regions unrelated to the class label. To address this issue,
we leverage the knowledge of core image regions and propose _Core-Focused Cross-_
_Entropy_, an extension of cross entropy, which encourages attention on core regions
while suppressing unrelated regions, improving feature alignment. Experiments on
Hard-ImageNet and Oxford-IIIT Pets show that ContrastiveCAM provides more
faithful attention maps and our method effectively improves feature alignment by
primarily extracting predictive performance from core image regions.


1 INTRODUCTION


The vast applications of convolutional neural networks in safety-critical domains such as medical
imaging (Kc et al., 2021; Rajpurkar et al., 2017), forensic investigation (Murthy and Siddesh,
2023) and self-driving (Kim and Canny, 2017) make accurate (a.k.a faithful) interpretations of their
predictions paramount (Haufe et al., 2024). Approaches to explain predictions include featureattribution based interpretability techniques (Zhou et al., 2016; Selvaraju et al., 2017; Draelos and
Carin, 2020), input-based interpretability with saliency maps (Simonyan et al., 2013; Smilkov et al.,
2017), and more recently, mechanistic interpretability for image circuit discovery (Olah et al., 2020).


In addition to faithful interpretability, ensuring that only target-relevant (a.k.a. core) regions influence
model predictions is a critical determination to make. A model-agnostic approach for evaluating the
impact of core regions involves input ablation experiments as introduced in recent work on Core
Risk Minimization (Singla et al., 2022; Moayeri et al., 2022). Images are modified to systematically
corrupt core regions, following which the change in performance is reported. Singla and Feizi
(2022) demonstrate that both convolutional and transformer-based architectures are vulnerable to
learning non-core regions of the input, caused by features like co-occurring backgrounds. These
encourage learning ‘tricks’ - shortcuts to learning that improve in-distribution accuracy while
inhibiting generalization over core features (Geirhos et al., 2020). A concrete example of shortcut
learning is illustrated within the introduction of Invariant Risk Minimization (Arjovsky et al., 2020).


In this work, we develop and leverage faithful interpretability to encourage feature alignment in
convolutional models. We theoretically observe that HiResCAMs (Draelos and Carin, 2020) may
not explain true factors that contribute towards predictions as a consequence of _softmax_ activation.
Specifically, we prove that HiResCAMs are not uniquely determined and admit arbitrary, spurious
shifts by a common matrix _M_ while corresponding to the same prediction (Theorem 3.2). This
spurious shift from _M_ can, in principle, completely corrupt HiResCAM explanations. To remove
this redundancy, we propose _ContrastiveCAMs_ (Definitions 3.3, 3.4), resulting in attention maps that


1


are invariant to the aforementioned spurious shift while additionally providing granular class-versusclass explanations. Using class-versus-class comparisons, we experimentally reveal circumstances
wherein different comparisons leverage different regions to base their predictions. Further, these
differing regions do not always correspond to core regions of the input image, i.e., there are spurious
contributions. We demonstrate that cross entropy loss encourages leveraging these unrelated regions,
especially in settings where the target represents a small portion of the image (Section 4.1). Finally,
we propose a modification to cross-entropy, termed _Core-Focused Cross-Entropy_ (Definition 4.5),
which: a) suppresses user-specified non-core regions despite the presence of spurious factors, and b)
generates contrast within user-specified target regions to solve for the underlying classification task.
This improves feature alignment by encouraging the model to learn target-relevant features only.


We demonstrate the effectiveness of our proposed method by reporting experimental results in
multiclass, multiple-class, and binary classification settings. We supplement this evidence by showing
that core-focused models may be trained competitively even with coarse or auto-generated masks,
and that they outperform backbones trained using cross-entropy in downstream segmentation tasks.


1.1 RELATED WORK


**Feature Attribution in Convolutional Networks.** A prominent family of interpretability techniques stems from the seminal CAMs (short for Class Activation Mappings) (Zhou et al., 2016)
literature. CAMs help identify regions-of-interest in the form of attention maps. It’s success led to
the introduction of a vast set of derivative works, that extend CAMs in various ways (Selvaraju et al.,
2017; Chattopadhay et al., 2018; Wang et al., 2020; Draelos and Carin, 2020).


**Representation Learning.** Arjovsky et al. (2020) introduces the notion of predictors that learn feature representations that are invariant to spurious factors. Bau et al. (2017) quantifies the interpretability of learned representations in convolutional models by evaluating hidden units within convolutional
layers on segmentation tasks. Recently, Zou et al. (2023) motivates neuroscience-inspired top-down
approached for inducing interpretability. It encourages the analysis of representations (representation
reading) and it’s subsequent modification (representation control).


**Feature Alignment.** Spurious factors in images encourage extracting predictions from unrelated
regions, termed _shortcuts_, and are discussed extensively by Geirhos et al. (2020). Feature alignment
seeks to ensure predictions are made using relevant features only, and is deeply connected with
robustness in neural networks (Wang, 2023). Preventing shortcut learning is thus a crucial goal
of feature alignment. Approaches to alignment include region masking (Kc et al., 2021), tiered
training (Aniraj et al., 2023), and regularization via saliency maps (Ismail et al., 2021), each having an
empirical focus. For a thorough exposition to recent advancements and challenges in interpretabilityguided feature alignment, we direct the reader to Weber et al. (2023) and Gao et al. (2024).


2 PRELIMINARIES


**Notation.** We denote vectors using bold lowercase letters (e.g., **v** ), matrices using uppercase letters
(e.g., _M_ ), and tensors using bold uppercase letters (e.g., **T** ), with partial indexing implying selection
of the subtensor across the remaining subsequent dimensions (e.g., **T** _i_ _∈_ R _[b][×][c]_ for **T** _∈_ R _[a][×][b][×][c]_ ). We
use the operator _⊙_ to represent elementwise multiplication, and define [ _C_ ] := _{_ 1 _,_ 2 _, . . ., C}_ .


**Setup.** In this paper, we consider image classification tasks. The dataset _D_ = _{_ ( **X** [(] _[i]_ [)] _,_ **y** [(] _[i]_ [)] ) _}_ _[n]_ _i_ =1
contains image-label pairs where images are represented using rank-3 tensors **X** consisting of two
spatial dimensions and one channel dimension, and labels are one-hot vectors **y** _∈_ R _[C]_, where _C_
denotes the total number of classes in the dataset. A neural network _f_ is trained to learn the relation
between the images **X** and labels **y** . The output of _f_ contains _C_ _logits_ : _fc_, _c ∈_ [ _C_ ]. Let _σ_ ( _·_ ) be the
softmax function, _f_ [˜] ( **X** ) = _σ_ ( _f_ ( **X** )) is then interpreted as the class-specific _probability predictions_ .
The standard training procedure is to optimize a cross-entropy loss function so that _f_ [˜] ( **X** ) matches
the label **y** as closely as possible for each training image-label pair.


In prominent approaches such as VGG (Simonyan and Zisserman, 2015), ResNet (He et al., 2016)
& ViT (Dosovitskiy et al., 2020), the neural network _f_ mainly consists of two consecutive parts, a


2


Each logit _fc_ is the summation of the HiResCAM over its spatial dimensions, up to a scalar **b** _c_ .


3 CONTRASTIVE CLASS ACTIVATION MAPS


In this section, we first discuss the theoretical limitations of HiResCAM in explaining model predictions, and then introduce a surrogate method, ContrastiveCAM, which offers more faithful and
class-specific explanations.


**HiResCAMs Admit Spurious Shifts.** A key observation is that HiResCAMs are only related to
_logits f_, not _probability predictions_ _f_ [˜] = _σ_ ( _f_ ) belonging to each class, see Eq. (3). The drawback is
that, for the same probability prediction _f_ [˜], there are infinitely many possible logit outputs _f_, hence
infinitely many HiResCAMs, each of which explain the same prediction differently. This drawback
arises intrinsically from the nature of the _softmax_ function.

**Proposition 3.1** (Contrastiveness of _softmax_ ) **.** _The softmax function is invariant to a universal shift_
_of all its input components:_


_σ_ ( **x** ) = _σ_ ( **x** + _a_ **1** _C_ ) _∀_ **x** _∈_ R _[C]_ _,_ _a ∈_ R (4)


3


backbone module _g_ followed by a classifier _h_ : _f_ = _h ◦_ _g_ . In this paper, we focus on convolutional
neural networks, i.e., the backbone _g_ is convolutional. We denote the output of the backbone _g_ as
**A** _∈_ R _[d]_ [0] _[×][d]_ [1] _[×][d]_ [2], termed as feature embedding (a.k.a. feature maps) of the image, where _d_ 0 is the
number of features (a.k.a. channels) and _d_ 1 and _d_ 2 are the spatial dimensions of the final convolutional
layer. The feature embedding **A** is then reduced to a vector **z**, either by flattening **z** = _vec_ ( **A** ), or by
Global Average Pooling (GAP). **z** is then processed by the classifier _h_, which outputs the logits _f_,
that are passed through _softmax_ to obtain the class prediction vector, denoted _f_ [˜] .


The recent trend is that the classifier _h_ becomes as simple as a single layer, such as in ConvNext (Liu
et al., 2022), ViT (Dosovitskiy et al., 2020), EfficientNet (Tan and Le, 2019), ResNet (He et al., 2016)
& DenseNet (Iandola et al., 2014):
_h_ ( **z** ) = _W_ **z** + **b** _,_ (1)

This simplification of _h_ is largely due to the fact that the backbone _g_, which encapsulates the bulk of
the model’s predictive power, extracts high quality and comprehensive features **A**, based on which a
single layer is enough to obtain accurate final predictions. In this paper, we assume that the classifier
is of the form in Eq. (1).


**HiResCAMs.** HiResCAMs (short for High-Resolution Class Activation Maps), introduced in
(Draelos and Carin, 2020), is a method designed to provide interpretable explanations of convolutional
neural networks. It renders the contribution of each spatial location in an image to the final logit
output _fc_, thereby revealing which regions are most critical to the models prediction. Specifically,
given a feature embedding **A** of an image **X** and a class index _c ∈_ [ _C_ ], the HiResCAM is defined as:


**CAM** [HiRes] _c_ =


_d_ 0


( _∇_ **A** _j_ _fc_ ) _⊙_ **A** _j,_ **CAM** [HiRes] _c_ _∈_ R _[d]_ [1] _[×][d]_ [2] (2)

_j_ =1


**CAM** [HiRes] _c_ shares spatial dimensions with the backbone output **A** . Each element within **CAM** [HiRes] _c_
represents a contribution to the logit output _fc_ from a corresponding patch within the original image.
A higher absolute value implies a greater contribution.


HiResCAMs have been widely used for incorporating explainability in a variety of tasks, such as
CT scan abnormality classification (Draelos and Carin, 2022), malware visualization (Brosolo et al.,
2025), coffee leaf rust classification (Chavarro et al., 2024), counterfeit banknote detection (Pachón
et al., 2023) & flow estimation (Chen and Wu, 2025).


Particularly, for single-layer classifiers _h_, Draelos and Carin (2020) show that the expression of
HiResCAMs, Eq. (2), can be simplified and has the following close connection with output logits _fc_ :


_fc_ ( **X** ) =


_d_ 1 _,d_ 2

 - **CAM** [HiRes] _c,i,j_ [(] **[X]** [) +] **[ b]** _[c][,]_ _c ∈_ [ _C_ ] _._ (3)

_i_ =1 _,j_ =1


|0|0 0|0|0|
|---|---|---|---|
|0|1|1|0|
|0|1|1|0|
|0|0|0|0|


|0|0 0|0|0|
|---|---|---|---|
|**0**|**0**|**0**|**0**|
|**0**|**0**|**0**|**0**|
|**0**|**0**|**0**|0|


|-05|07 0|-01|-04|
|---|---|---|---|
|**0**<br>-2|**0**<br>10|**0**<br>3|**0**<br>-2|
|**0**<br>4|**0**<br>3|**0**<br>-4|**0**<br>4|
|**0**|0<br>-8|0<br>1|6|


|-05|07 0|-01|-04|
|---|---|---|---|
|**0**<br>-2|**0**<br>11|**0**<br>4|**0**<br>-2|
|**0**<br>4|**0**<br>4|**0**<br>-3|**0**<br>4|
|**0**|0<br>-8|0<br>1|6|


|-05|07 0|-01|-04|
|---|---|---|---|
|**0**<br>-2|**0**<br>10|**0**<br>3|**0**<br>-2|
|**0**<br>4|**0**<br>3|**0**<br>-4|**0**<br>4|
|**0**|0<br>-8|0<br>1|6|


|Col1|Col2|Col3|
|---|---|---|
|0<br>0<br>0<br>1<br>0<br>0<br>1<br>0<br>**0**<br>**0**<br>**0**<br>**0**<br>0<br>**0**<br>**0**<br>**0**<br>,<br>,<br>0<br>**0**<br>**0**<br>**0**<br>0<br>0<br>**0**<br>**0**<br>**0**<br>6<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>4<br>4<br>-3<br>1<br>-5<br>-2<br>4<br>-8<br>4<br>11<br>7<br>-1<br>-4<br>-2<br>4<br>0<br>16<br>12<br>0<br>**0**<br>**0**<br>**0**<br>0<br>0<br>**0**<br>**0**<br>**0**<br>6<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>**0**<br>3<br>4<br>-4<br>1<br>-5<br>-2<br>4<br>-8<br>3<br>10<br>7<br>-1<br>-4<br>-2|4|0|


**CAM** [Recon] _ct_ thus removes redundancy _R_ = _−_ [1] _/C_ _·_ [�] _c_ _[C]_ =1 **[CAM]** _c_ [HiRes] . We report the ratio of
redundancy to the original explanation as _γ_ = _[∥][R][∥][F]_ _/∥_ **CAM** [HiRes] _ct_ _∥F_ for various datasets in Table 1.


Crucially, ContrastiveCAMs are invariant to spurious contributions as exposed by Theorem 3.2.


4


_Proof._ All proofs are deferred to Appendix A.


This invariance to _a ∈_ R is amplified to a matrix _M_ _∈_ R _[d]_ [1] _[×][d]_ [2] when assessing HiResCAMs.

**Theorem** **3.2.** _HiResCAM_ _explanations_ **CAM** [HiRes] _∈_ R _[C][×][d]_ [1] _[×][d]_ [2] _corresponding_ _to_ _probability_
_predictions_ _f_ [˜] ( **X** ) _∈_ R _[C]_ _are_ _not_ _uniquely_ _determined,_ _admitting_ _a_ _universal_ _shift_ _of_ _class-level_
_explanations_ **CAM** [HiRes] _c_ _by an arbitrary matrix M_ _∈_ R _[d]_ [1] _[×][d]_ [2] _∀c ∈_ [ _C_ ] _._




_f_ ˜( **X** ) = _σ_ 


 - **CAM** [HiRes] : _,i,j_ + **b**


_i_ =1 _,j_ =1


_d_ 1 _,d_ 2





 = _σ_




~~H~~ iRes
**CAM** : _,i,j_ + **b**  _∀M_ _∈_ R _[d]_ [1] _[×][d]_ [2] (5)


_i_ =1 _,j_ =1








_d_ 1 _,d_ 2


HiRes
_Where_ **CAM** _is defined as:_


~~H~~ iRes
**CAM** _c_ := **CAM** [HiRes] _c_ + _M_ _∀c ∈_ [ _C_ ] (6)


Thus explanations from HiResCAMs are accurate only upto a summand _M_ which is unknown. These
explanations may be misleading, and _fail to guarantee a faithful interpretation_ of the model prediction.
An example of such a misinterpretation is illustrated in Figure 1.


.98 .02


~~H~~ iRes
Figure 1: Shifting **CAM** [HiRes] by arbitrary matrix _M_ results in a change to explanations **CAM**

which subsequently changes the corresponding logit vector. However, the model’s final prediction
probabilities are identical and remain unchanged.


To remove this redundancy, we define a contrastive representation of HiResCAMs, which recovers
faithful attention maps at the class probability level.


**Definition 3.3** (ContrastiveCAMs) **.** Given a set of classes [ _C_ ] with _ct_ being the index of the target
class for a given image, ContrastiveCAM is defined as follows:


      -      **CAM** [Cntrst] _ct_ := **CAM** [Cntrst] ( _ct,c_ _[′]_ ) [:] _[ c][′]_ _[∈]_ [[] _[C]_ []] _[ \][ c]_ _,_ **CAM** [Cntrst] ( _ct,c_ _[′]_ ) [:=] **[ CAM]** _c_ [HiRes] _t_ _−_ **CAM** [HiRes] _c_ _[′]_ (7)


Further, we also reconstruct single-class interpretations of ContrastiveCAMs:


**Definition 3.4** (Class-Reconstructed ContrastiveCAMs) **.** Given a set of classes [ _C_ ] with _ct_ being the
index of the target class for a given image, reconstructed ContrastiveCAMs are defined as follows:


**CAM** [Recon] _ct_ := _C_ [1]


_C_


- **CAM** [Cntrst] ( _ct,c_ ) [=] **[ CAM]** _c_ [HiRes] _t_ _−_ _C_ [1]

_c_ =1


_C_


_C_

- **CAM** [HiRes] _c_ (8)

_c_ =1


~~H~~ iRes
**Theorem** **3.5** (ContrastiveCAMs are _M_ -invariant) **.** _Let_ **CAM** [HiRes] _and_ **CAM** _be_ _two_
_HiResCAMs corresponding to probability predictions_ _f_ [˜] ( **X** ) _∈_ R _[C]_ _such that:_


HiRes
**CAM** _c_ = **CAM** [HiRes] _c_ + _M_ _∀c ∈_ [ _C_ ] (9)

_Then, for every M_ _∈_ R _[d]_ [1] _[×][d]_ [2] _, it holds that:_


~~C~~ ntrst ~~R~~ econ
**CAM** [Cntrst] = **CAM** and **CAM** [Recon] = **CAM** (10)


**Class-versus-Class Explanations.** While explanations from the CAM-family only involve visualizing _fct_, softmax activation uses every logit in computing class probabilities. Making inferences based
on individual logits may thus misinterpret the internal model state, as the training objective induced
by cross-entropy loss over softmax activation is to maximize the **difference between class logits**, see
Eq. (44). We demonstrate the value of additional granularity provided by pairwise explanations by
reporting observations on a three-class subset of Hard-ImageNet in Figure 2.


Figure 2: We plot ContrastiveCAM and HiResCAM explanations on a ResNet-18 model trained to
classify: (‘dog sled’, ‘volleyball’, ‘baseball player’), ordered by label index. ContrastiveCAMs reveal
circumstances wherein: a) regions that contribute towards prediction are hidden by HiResCAMs, and
b) differing parts of the image contribute towards various class-versus-class predictions.


From Figure 2, we also observe that the model often leverages irrelevant regions (e.g., environmental
cues), to inform predictions. Following Moayeri et al. (2022), we refer to these regions as _non-core_
_regions_ . In principle, _core regions_ are those that causally influence the prediction (modification of
this region could mean the ground truth itself may change), while _non-core regions_ represent spurious
correlations – modifications to these regions do not change the ground truth labels.


Table 1: ContrastiveCAM explanations bifurcated by core-region maps across various datasets. The
average contributions of core / non-core regions and ratio of redundancy removed is reported below.


**Dataset** Core ( _↑_ ) Non-Core ( _↓_ ) Core _/_ Total ( _↑_ ) Redundancy ( _γ_ ) Accuracy (%)


Hard-ImageNet 14.817 **42.138** .2601 .201 95.73
Oxford-IIIT Pets **3.925** 2.150 .6461 .367 99.34
PASCAL VOC 1.581 **1.719** .4791 _−_ [1] 87.32


This undesired influence is consistently observed, as evidenced by high overall non-core contribution
in Table 1 above. Despite strong accuracy, large contributions arise from non-core regions.


4 LEARNING WITH CONTRASTIVECAMS


The dependency on non-core regions observed above is evidence of misalignment, which inhibits
generalization. In this section, we prove a desirable theoretical property of ContrastiveCAMs and
leverage it to incorporate interpretability within model optimization, mitigating this weakness.


5


Where _ℓ_ ( _f_ ( **X** ) _,_ **y** ) = 1 (arg max( _f_ [˜] ( **X** )) _̸_ = arg max( **y** )) is 0/1 loss for the multiclass setting.


6


Specifically, we prove that any _input-dependent_ change to probability predictions _f_ [˜] (e.g., caused by
updating model weights) is precisely reflected by a proportionate change to **CAM** [Cntrst] _ct_ .

**Proposition 4.1** (Correctness of ContrastiveCAMs) **.** _Softmax-activated class probabilities_ _f_ [˜] _can be_
_expressed as a direct function of ContrastiveCAMs and the bias vector._


_f_ ˜ _ct_ ( **X** ) =


- _C_

 - exp - **b** _c −_ **b** _ct_ _−_ - **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


_∀ct_ _∈_ [ _C_ ] (11)


_Where_ **CAM** [Cntrst] ( _ct,ct_ ) [=] **[ 0]** _[d]_ 1 _[×][d]_ 2 _[.]_


By zero-ing the final bias vector (i.e., **b** := **0** _C_ for _h only_ ), we can precisely disassociate the role of
specific regions in computing cross-entropy. We leverage this property to study feature misalignment,
and later in our proposed modification of cross-entropy to penalize the use of non-core regions.


4.1 CROSS-ENTROPY CAN MOTIVATE FEATURE MISALIGNMENT


To encode core-region information, for each sample from our dataset of size _N_, we extend dataset _D_
by specifying a binary mask _H_, which indicates whether or not downsampled regions from the input
image may be used to determine the prediction.


�1 region contains target
_D_ := _{_ ( **X** [(] _[i]_ [)] _,_ ( _H_ [(] _[i]_ [)] _,_ **y** [(] _[i]_ [)] )) _}_ _[N]_ _i_ =1 where _Hjk_ := 0 region doesn’t contain target _[∀][j, k]_ _[∈]_ [[] _[d]_ [1][]] _[,]_ [ [] _[d]_ [2][]]


We can restate cross-entropy as a function of ContrastiveCAMs and core-region information in _D_ .

**Proposition** **4.2.** _Given_ _bias-free_ _classifier_ _h,_ _we_ _can_ _precisely_ _associate_ _the_ _impact_ _of_ _specific_
_regions, encoded by binary mask H, to the computation of cross-entropy loss._


_L_ CE( _f_ ( **X** ) _,_ **y** _, H_ ) = log


- _C_

 - exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[−]_ �(1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ ) - [�]

_c_ =1


(12)

**Remark 4.3.** _Equivalently, we disassociate the logit and use the standard cross-entropy formulation:_





 _−_



_d_ 1 _,d_ 2

 - _H_ _⊙_ **CAM** [Cntrst] ( _ct,_ :) _,i,j_

_i_ =1 _,j_ =1 - �� core









+ (1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,_ :) _,i,j_  _,_ **y**

 - ��  non _−_ core





_L_ CE( _f_ ( **X** ) _,_ **y** _, H_ ) = _L_ CE





 _σ_



(13)


We observe from Proposition 4.2 that cross-entropy loss does not inherently favor using the core or
non-core regions for classification. Provided the prediction is accurate with high confidence, error
remains low. This presents a theoretical basis for feature misalignment in convolutional networks.


**Scale-Sensitivity** **of** **Convolutional** **Approaches.** In training classification models, an implicit
assumption is that the strongest indicator of the class label is the target itself (i.e., the core regions).
From Table 1, we observe through the significant influence of non-core regions that this assumption
does not universally hold. In cases where the target is far from the camera, as commonly observed
in Hard-ImageNet, the emphasis is placed on **learning the best non-core surrogate to the actual**
**target**, rather than obtaining an accurate feature representation using just the fewer relevant regions.


Learning a non-core surrogate does reduce cross-entropy loss, but at the cost of misrepresenting the
underlying classification target, thus inducing feature misalignment. The model should, through the
course of training, distinguish and ignore non-core regions in determining the final prediction.


This leads us to propose an alignment-motivated constraint to empirical risk minimization.

**Definition 4.4** (Core-Constrained Risk Minimization) **.**


             -              _R_ CCRM( _f_ ) := E( **X** _,_ ( _H,_ **y** )) _∼D_ _ℓ_ ( _f_ [˜] ( **X** ) _,_ **y** ) s.t.


_C_


_c_ =1


���(1 _−_ _H_ ) _⊙_ **CAM** Cntrst( _ct,c_ ) ��� = 0 (14)


The divergence term motivates similarity in the _shape_ of ContrastiveCAMs to _H_ . The normalizing
behavior of softmax, analogous to its effect on the logits, means that absolute scale is invariant; that
information comes exclusively from _L_ CFCE.


Supplemental formulations and adaptations of core-focused optimization are deferred to Appendix B.


5 EXPERIMENTS


For our experiments, we evaluate the performance of ResNet-50 with a set of interpretabilitymotivated modifications. These are detailed in Appendix C. For consistency, we include baselines
with (denoted by ‘w/ Arch’) and without these modifications. We initialize each training run on
ImageNet pre-trained weights, and report fine-tuning performance.


**Datasets.** We present training results for Oxford IIIT-Pets (Parkhi et al., 2012), Hard-ImageNet
(Moayeri et al., 2022), and the Semantic Boundaries Dataset (Hariharan et al., 2011). These datasets
span image classification tasks with binary, multiclass & multilabel targets. In addition to reporting
raw prediction performance, we also report intersection-over-union (IoU) scores, indicating the
overlap between ground-truth core regions and those used by the models for classification.


5.1 HARD-IMAGENET


Hard-ImageNet (Moayeri et al., 2022) is a subset of ImageNet (Deng et al., 2009) that only contains
classes that have been observed to use spurious features to inform predictions (Singla and Feizi, 2022).


7


4.2 CORE-FOCUSED CROSS-ENTROPY


We have shown that cross-entropy motivates generating predictions using either core or non-core
features. To correct this, we propose Core-Focused Cross-Entropy, which penalizes the contribution
from non-core regions to the final classification.


**Definition** **4.5** (Core-Focused Cross-Entropy) **.** We integrate masked region suppression to the
definition of cross-entropy using the following formulation:


        - _C_

     -     -     -     -     - [�]
_L_ CFCE( _f_ ( **X** ) _,_ **y** _, H_ ) := log exp _−_ _H ⊙_ **CAM** [Cntrst] ( _ct,c_ ) [+] (1 _−_ _H_ ) _⊙|_ **CAM** [Cntrst] ( _ct,c_ ) _[|]_

_c_ =1

(15)


We can show that the above loss function is consistent with our constrained optimization objective.


**Theorem** **4.6** (Consistency of Core-Focused Cross-Entropy) **.** _A_ _sequence_ _of_ _predictors_ _fn_ _that_
_converges to the optimal R_ CFCE _-risk also converges to the Bayes-optimal R_ CCRM _-risk._ _Equivalently,_
_in the realizable setting, L_ CFCE _is classification-calibrated._

_R_ CFCE( _fn_ ) _→R_ _[∗]_ CFCE [=] _[⇒R]_ [CCRM][(] _[f][n]_ [)] _[ →R]_ CCRM _[∗]_ (16)


_Where R_ CFCE( _f_ ) _is defined as:_


_R_ CFCE( _f_ ) := E( **X** _,_ ( _H,_ **y** )) _∼D_ [ _L_ CFCE( _f_ ( **X** ) _,_ **y** _, H_ )] (17)


**Divergence Regularization.** Using ContrastiveCAMs, we observe a tendency for cross-entropy
to only generate contrast in regions where feature differences are prominent within the training set.
Successful test predictions rely on the prominence of the same set of differing features even if there
exist subtleties in the training set that can be used to offer more nuanced classifications. We thus
propose regularization by minimizing divergence between target mask _H_ and **CAM** [Cntrst] _c_ . This
encourages contrast for every region in which the target is present, even when the difference is subtle.


**Definition** **4.7** (Regularized Core-Focused Cross-Entropy) **.** We regularize _L_ CFCE to encourage
contrast over the entire target region using KL Divergence:


_L_ RCFCE( _f_ ( **X** ) _,_ **y** _, H_ ) := _L_ CFCE + _λ_ 1
_C −_ 1


 - _DKL_ - _σ_ ( _λ_ 2 _H_ ) _|| σ_ - _λ_ 3 **CAM** [Cntrst] ( _ct,c_ ) �� (18)

_c∈_ [ _C_ ] _\ct_


The core regions from these classes typically constitute a minority of the overall image (13 _._ 96% on
average), lending further evidence to the scale-sensitivity of convolutional models (Section 4.1).


To evaluate the performance of models using core regions only, Moayeri et al. (2022) introduces
an evaluation suite that reports a) accuracy when core regions are removed from the image using
segmentation masking, bounding-box masking and tiling over the foreground; b) _relative foreground_
_sensitivity_ (RFS) which evaluates performance degradation under corruption of the foreground; and c)
saliency alignment measured by intersection over union of core masks to regions used for prediction.


Table 2: Hard-ImageNet benchmarks on finetuned ResNet-50 models trained using varying approaches. Models trained using our proposed core-focused loss functions show significant improvement across all evaluations, at the cost of some un-ablated performance.

|Accuracy under Core-Region Ablation (%)<br>Method<br>None (↑) Gray Mask (↓) Gray BBOX (↓) Tile (↓)|GradCAM Contrastive-<br>RFS (↑) IoU (↑) CAM IoU (↑)|
|---|---|
|Cross-Entropy<br>94.25<br>75.94<br>69.39<br>67.38|-0.18<br>18.44<br>_−_|
|CORM (Singla et al., 2022)<br>92.91<br>76.20<br>69.12<br>68.32<br>DFR (Kirichenko et al., 2022)<br>**94.39**<br>73.53<br>67.51<br>66.71<br>CORM + DFR<br>91.31<br>72.59<br>63.64<br>63.90|-0.08<br>20.43<br>_−_<br>-0.27<br>18.39<br>_−_<br>-0.23<br>20.35<br>_−_|
|CE w/ Arch<br>93.69_±_0.77<br>76.53_±_2.15<br>72.49_±_2.19<br>71.02_±_2.4<br>CFCE (Ours)<br>90.53_±_0.69<br>**41.78**_±_**1.49**<br>**31.66**_±_**1.26**<br>**34.31**_±_**1.04**<br>CFCE + KL (Ours)<br>90.35_±_1.59<br>45.49_±_5.15<br>37.07_±_4.57<br>39.47_±_4.12|-0.23_±_0.05<br>16.25_±_14.07<br>30.27_±_3.99<br>.224_±_0.10<br>18.88_±_1.13<br>89.22_±_0.31<br>**.236**_±_**0.10**<br>**51.52**_±_**1.07**<br>**93.39**_±_**0.11**|


IoU for this benchmark was computed using GradCAMs (Selvaraju et al., 2017) only for consistency
with baselines, as GradCAMs have been shown to present unfaithful explanations (Draelos and Carin,
2020). We thus include additional evaluations using ContrastiveCAMs for core-focused models. We
also qualitatively evaluate improvements using core-focused approaches in Figure 3 below.


Figure 3: Models trained using CFCE exhibit suppressed contributions from non-core regions.


5.2 OXFORD IIIT-PETS


The Oxford IIIT-Pets dataset contains images of 37 breeds of cats and dogs, paired with segmentation
trimaps that denote the foreground and background regions within the image. In the binary setting,
the objective is to classify cats and dogs; individual breed labels are merged. This creates a class
imbalance (4978 dogs to 2371 cats), however no training modifications are made to account for this.
There is virtually no class imbalance in the multiclass setting.


**Applicability of Approximate Masks.** Core-region masks _H_ have a smaller resolution compared
to input **X** as a consequence of the convolutional backbone _g_ . Thus, in the absence of ground-truth
core-region masks, approximate pixel-level masks or weaker supervision such as bounding boxes can
be used to effectively suppress contributions from non-core regions. We demonstrate this empirically
through competitive alignment achieved both with auto-generated masks obtained using Segment
Anything (Kirillov et al., 2023) (SAM), and with weaker supervision via bounding boxes (BBOX).


8


|Core Binary<br>Method Region Accuracy (%) IoU (%)<br>Masks Train Valid Train Valid|Multiclass<br>Accuracy (%) IoU (%)<br>Train Valid Train Valid|
|---|---|
|Cross-Entropy<br>_−_<br>99.82_±_0.26<br>99.40_±_0.07<br>78.37_±_1.12<br>78.37_±_1.14<br>CE w/ Arch<br>_−_<br>99.99_±_0.02<br>99.4_±_0.22<br>38.58_±_16.95<br>39.07_±_16.98|99.92_±_0.21<br>**94.41**_±_**1.07**<br>80.04_±_0.66<br>80.16_±_0.48<br>100_±_0<br>95.3_±_0.3<br>59.86_±_17.09<br>60.6_±_17.2|
|CFCE<br>GT<br>99.88_±_0.10<br>99.32_±_0.25<br>83.22_±_1.13<br>82.92_±_1.18<br>CFCE + KL<br>GT<br>99.71_±_0.27<br>99.32_±_0.15<br>**94.93**_±_**0.88**<br>**92.72**_±_**0.73**|99.96_±_0.03<br>92.96_±_0.15<br>87.93_±_0.24<br>88.16_±_0.33<br>99.74_±_0.13<br>90.08_±_1.47<br>**96.22**_±_**3.58**<br>**93.12**_±_**2.22**|
|CFCE<br>SAM<br>99.92_±_0.06<br>99.37_±_0.15<br>83.96_±_2.1<br>83.95_±_2.33<br>CFCE + KL<br>SAM<br>99.88_±_0.07<br>99.19_±_0.24<br>83.46_±_1.73<br>83.54_±_1.96|99.6_±_0.19<br>93.26_±_0.67<br>84.79_±_1.26<br>85.26_±_1.22<br>99.6_±_0.2<br>93.7_±_0.28<br>84.67_±_1.16<br>85.16_±_1.2|
|CFCE<br>BBOX<br>_±_**0.01**<br>**99.42**_±_**0.22**<br>79.09_±_2.26<br>79.13_±_2.28|**99.98**_±_**0**<br>93.83_±_0.33<br>84.26_±_1.86<br>84.61_±_1.91|


|Col1|Col2|Fine-Tune with CE-Trained Init<br>Fine-Tune with CFCE+KL-Trained Init<br>E2E with CE-Trained Init<br>E2E with CFCE+KL-Trained Init|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||Fine-Tune with CE-Trained Init<br>Fine-Tune with CFCE+KL-Trained Init<br>E2E with CE-Trained Init<br>E2E with CFCE+KL-Trained Init|||||


6 DISCUSSION


In this work, we establish a connection between interpretability and feature alignment. We demonstrate the impact of utilizing _post-hoc_ (i.e., post-training) explainability methods, primarily used as
sanity checks, as a guiding factor during training to improve feature alignment with encouraging
effect. Core-Focused Cross Entropy is a direct result of the desirable theoretical properties of ContrastiveCAMs, establishing the value of correctness guarantees in interpretability. Reductive metrics
inevitably present a partial view of factors that influence model prediction, and comprehensively
ensuring that deep neural networks faithfully learn to solve the intended, underlying objective remains a significant challenge for the research community. We hope that our work motivates further
exploration towards connections between interpretability and alignment of deep neural networks.


9


Notably, KL regularization must not be applied when bounding boxes are used in place of masks, as
fitting to the shape of the box mischaracterizes the target. Also note that ground-truth (GT) masks are
used for validation in every setting to ensure a fair comparison.


5.3 SEMANTIC BOUNDARIES DATASET (PASCAL VOC)


The Semantic Boundaries Dataset introduces segmentation annotations to the entire Pascal VOC 2011
Dataset (Everingham et al., 2011). We use this dataset to demonstrate performance improvements for
both classification and downstream detection settings.


**Classification.** PASCAL VOC encodes a 20-class _multilabel_ classification task; thus input image
may contain multiple positive classifications. We report a pareto improvement with increased Average
Precision (AP) and Intersection-over-Union (IoU) scores when using core-focused loss formulations.


**AP (%)** **IoU (%)**
**Method**
Train Valid Train Valid


Cross-Entropy **99.75** _±_ **0.30** 87.32 _±_ 2.58 46.08 _±_ 16.54 44.50 _±_ 16.57
CE w/ Arch 99.57 _±_ 0.74 **88.85** _±_ **0.79** 40.69 _±_ 16.37 38.55 _±_ 16.43
CFBCE 98.38 _±_ 2.49 88.39 _±_ 1.23 85.00 _±_ 1.32 82.07 _±_ 0.91
CFBCE + KL 97.92 _±_ 1.00 87.19 _±_ 0.46 **89.53** _±_ **1.89** **85.39** _±_ **0.60**


**Segmentation.** We also report improvements in IoU performance of core-focused backbones on
downstream segmentation, both when fine-tuned (i.e., with a frozen backbone) and trained end-to-end.


80


60


40


20


0


REFERENCES


Ananthu Aniraj, Cassio F Dantas, Dino Ienco, and Diego Marcos. Masking strategies for background
bias removal in computer vision models. In _Proceedings of the IEEE/CVF International Conference_
_on Computer Vision_, pages 4397–4405, 2023.


Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization,
2020. [URL https://arxiv.org/abs/1907.02893.](https://arxiv.org/abs/1907.02893)


David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection:
Quantifying interpretability of deep visual representations. In _Proceedings of the IEEE conference_
_on computer vision and pattern recognition_, pages 6541–6549, 2017.


Matteo Brosolo, P Vinod, and Mauro Conti. Through the static: Demystifying malware visualization
via explainability. _Journal of Information Security and Applications_, 91:104063, 2025.


Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader, and Vineeth N Balasubramanian. Gradcam++: Generalized gradient-based visual explanations for deep convolutional networks. In _2018_
_IEEE Winter Conference on Applications of Computer Vision (WACV)_ . IEEE, March 2018. doi:
10.1109/wacv.2018.00097. [URL http://dx.doi.org/10.1109/WACV.2018.00097.](http://dx.doi.org/10.1109/WACV.2018.00097)


Adrian Chavarro, Diego Renza, and Ernesto Moya-Albor. Convnext as a basis for interpretability in
coffee leaf rust classification. _Mathematics (2227-7390)_, 12(17), 2024.


Yu-Hsi Chen and Chin-Tien Wu. Reynoldsflow: Exquisite flow estimation via reynolds transport
theorem, 2025. [URL https://arxiv.org/abs/2503.04500.](https://arxiv.org/abs/2503.04500)


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pages 248–255. Ieee, 2009.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale. _arXiv_ _preprint_
_arXiv:2010.11929_, 2020.


Rachel Lea Draelos and Lawrence Carin. Use hirescam instead of grad-cam for faithful explanations
of convolutional neural networks. _arXiv preprint arXiv:2011.08891_, 2020.


Rachel Lea Draelos and Lawrence Carin. Explainable multiple abnormality classification of chest ct
volumes. _Artificial Intelligence in Medicine_, 132:102372, 2022.


M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, and A. Zisserman. The PASCAL Visual Object Classes Challenge 2011 (VOC2011) Results. http://www.pascal-network.org/challenges/VOC/voc2011/workshop/index.html, 2011.


Yuyang Gao, Siyi Gu, Junji Jiang, Sungsoo Ray Hong, Dazhou Yu, and Liang Zhao. Going beyond
xai: A systematic survey for explanation-guided learning. _ACM Computing Surveys_, 56(7):1–39,
2024.


Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias
Bethge, and Felix A Wichmann. Shortcut learning in deep neural networks. _Nature_ _Machine_
_Intelligence_, 2(11):665–673, 2020.


Bharath Hariharan, Pablo Arbeláez, Lubomir Bourdev, Subhransu Maji, and Jitendra Malik. Semantic
contours from inverse detectors. In _2011_ _international_ _conference_ _on_ _computer_ _vision_, pages
991–998. IEEE, 2011.


Stefan Haufe, Rick Wilming, Benedict Clark, Rustam Zhumagambetov, Danny Panknin, and Ahcene
Boubekki. Position: Xai needs formal notions of explanation correctness. In _Interpretable AI: Past,_
_Present and Future_, 2024.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,
pages 770–778, 2016.


10


Forrest Iandola, Matt Moskewicz, Sergey Karayev, Ross Girshick, Trevor Darrell, and Kurt Keutzer.
Densenet: Implementing efficient convnet descriptor pyramids. _arXiv preprint arXiv:1404.1869_,
2014.


Aya Abdelsalam Ismail, Hector Corrada Bravo, and Soheil Feizi. Improving deep learning interpretability by saliency guided training. _Advances in Neural Information Processing Systems_, 34:
26726–26739, 2021.


Kamal Kc, Zhendong Yin, Dasen Li, and Zhilu Wu. Impacts of background removal on convolutional
neural networks for plant disease classification in-situ. _Agriculture_, 11(9):827, 2021.


Jinkyu Kim and John Canny. Interpretable learning for self-driving cars by visualizing causal attention.
In _Proceedings of the IEEE international conference on computer vision_, pages 2942–2950, 2017.


Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014.


Polina Kirichenko, Pavel Izmailov, and Andrew Gordon Wilson. Last layer re-training is sufficient
for robustness to spurious correlations. _arXiv preprint arXiv:2204.02937_, 2022.


Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick.
Segment anything, 2023. [URL https://arxiv.org/abs/2304.02643.](https://arxiv.org/abs/2304.02643)


Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
A convnet for the 2020s. In _Proceedings of the IEEE/CVF conference on computer vision and_
_pattern recognition_, pages 11976–11986, 2022.


Ilya Loshchilov and Frank Hutter. Sgdr: Stochastic gradient descent with warm restarts. _arXiv_
_preprint arXiv:1608.03983_, 2016.


Mazda Moayeri, Sahil Singla, and Soheil Feizi. Hard imagenet: Segmentations for objects with
strong spurious cues, June 2022.


Jamuna S Murthy and GM Siddesh. Ai based criminal detection and recognition system for public
safety and security using novel criminalnet-228. In _International_ _Conference_ _on_ _Frontiers_ _in_
_Computing and Systems_, pages 3–20. Springer, 2023.


Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter.
Zoom in: An introduction to circuits. _Distill_, 5(3):e00024–001, 2020.


César G Pachón, Dora M Ballesteros, and Diego Renza. An efficient deep learning model using
network pruning for fake banknote recognition. _Expert Systems with Applications_, 233:120961,
2023.


Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In _2012_
_IEEE conference on computer vision and pattern recognition_, pages 3498–3505. IEEE, 2012.


Pranav Rajpurkar, Jeremy Irvin, Kaylie Zhu, Brandon Yang, Hershel Mehta, Tony Duan, Daisy Ding,
Aarti Bagul, Curtis Langlotz, Katie Shpanskaya, et al. Chexnet: Radiologist-level pneumonia
detection on chest x-rays with deep learning. _arXiv preprint arXiv:1711.05225_, 2017.


Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh,
and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings of the IEEE international conference on computer vision_, pages 618–626,
2017.


Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image
recognition, 2015. [URL https://arxiv.org/abs/1409.1556.](https://arxiv.org/abs/1409.1556)


Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks:
Visualising image classification models and saliency maps. _arXiv preprint arXiv:1312.6034_, 2013.


11


HiRes
_Where_ **CAM** _is defined as:_


~~H~~ iRes
**CAM** _c_ := **CAM** [HiRes] _c_ + _M_ _∀c ∈_ [ _C_ ] (21)


12


Sahil Singla and Soheil Feizi. Salient imagenet: How to discover spurious features in deep learning?
In _International Conference on Learning Representations_, 2022. [URL https://openreview.](https://openreview.net/forum?id=XVPqLyNxSyh)
[net/forum?id=XVPqLyNxSyh.](https://openreview.net/forum?id=XVPqLyNxSyh)


Sahil Singla, Mazda Moayeri, and Soheil Feizi. Core risk minimization using salient imagenet. _arXiv_
_preprint arXiv:2203.15566_, 2022.


Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. Smoothgrad:
removing noise by adding noise. _arXiv preprint arXiv:1706.03825_, 2017.


Mingxing Tan and Quoc Le. Efficientnet: Rethinking model scaling for convolutional neural networks.
In _International conference on machine learning_, pages 6105–6114. PMLR, 2019.


Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, and
Xia Hu. Score-cam: Score-weighted visual explanations for convolutional neural networks. In
_Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops_,
pages 24–25, 2020.


Zifan Wang. _On_ _the_ _Feature_ _Alignment_ _of_ _Deep_ _Vision_ _Models_ _Explainability_ _and_ _Robustness_
_Connected at Hip_ . PhD thesis, Carnegie Mellon University, 2023.


Leander Weber, Sebastian Lapuschkin, Alexander Binder, and Wojciech Samek. Beyond explaining:
Opportunities and challenges of xai-based model improvement. _Information Fusion_, 92:154–176,
2023.


Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo.
Cutmix: Regularization strategy to train strong classifiers with localizable features. In _Proceedings_
_of the IEEE/CVF international conference on computer vision_, pages 6023–6032, 2019.


Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Learning deep
features for discriminative localization. In _Proceedings of the IEEE conference on computer vision_
_and pattern recognition_, pages 2921–2929, 2016.


Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan,
Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A
top-down approach to ai transparency. _arXiv preprint arXiv:2310.01405_, 2023.


A MATHEMATICAL DERIVATIONS


**Proposition 3.1.** _The softmax function is invariant to a universal shift of all its input components:_


_σ_ ( **x** ) = _σ_ ( **x** + _a_ **1** _C_ ) _∀_ **x** _∈_ R _[C]_ _,_ _a ∈_ R


_Proof._


1 1
_σ_ ( **x** + _a_ **1** _C_ ) = - _C_ [(] _[e]_ **[x]** [1][+] _[a][, e]_ **[x]** [2][+] _[a][,][ · · ·]_ _[, e]_ **[x]** _[C]_ [+] _[a]_ [) =] - _C_ [(] _[e]_ **[x]** [1] _[, e]_ **[x]** [2] _[,][ · · ·]_ _[, e]_ **[x]** _[C]_ [) =] _[ σ]_ [(] **[x]** [)] _[.]_
_c_ =1 _[e]_ **[x]** _[c]_ [+] _[a]_ _c_ =1 _[e]_ **[x]** _[c]_

(19)


**Theorem** **3.2.** _HiResCAM_ _explanations_ **CAM** [HiRes] _∈_ R _[C][×][d]_ [1] _[×][d]_ [2] _corresponding_ _to_ _probability_
_predictions_ _f_ [˜] ( **X** ) _∈_ R _[C]_ _are_ _not_ _uniquely_ _determined,_ _admitting_ _a_ _universal_ _shift_ _of_ _class-level_
_explanations_ **CAM** [HiRes] _c_ _by an arbitrary matrix M_ _∈_ R _[d]_ [1] _[×][d]_ [2] _∀c ∈_ [ _C_ ] _._





_d_ 1 _,d_ 2


_d_ 1 _,d_ 2

 

_i_ =1 _,j_ =1





 = _σ_




~~H~~ iRes
**CAM** : _,i,j_ + **b**  _∀M_ _∈_ R _[d]_ [1] _[×][d]_ [2] (20)


_f_ ˜( **X** ) = _σ_





 - **CAM** [HiRes] : _,i,j_ + **b**


_i_ =1 _,j_ =1








Proving the desired statement.


~~H~~ iRes
**Theorem 3.5.** _Let_ **CAM** [HiRes] _and_ **CAM** _be two HiResCAMs corresponding to probability_
_predictions_ _f_ [˜] ( **X** ) _∈_ R _[C]_ _such that:_


HiRes                 -                 **CAM** = **CAM** [HiRes] _c_ + _M_ : _c ∈_ [ _C_ ] (28)


_Then, for every M_ _∈_ R _[d]_ [1] _[×][d]_ [2] _, it holds that:_


~~C~~ ntrst ~~R~~ econ
**CAM** [Cntrst] _ct_ = **CAM** _ct_ and **CAM** [Recon] = **CAM** (29)


_Proof._ For some _ct_ _∈_ [ _C_ ], we have:


~~C~~ ntrst        - Cntrst        **CAM** _ct_ = **CAM** ( _ct,c_ ) [:] _[ c][ ∈]_ [[] _[C]_ []] _[ \][ c][t]_ (30)


         - HiRes ~~H~~ iRes          = **CAM** _ct_ _−_ **CAM** _c_ : _c ∈_ [ _C_ ] _\ ct_ (31)


HiRes
By definition of **CAM**, we have:


         -          = **CAM** [HiRes] _ct_ + _M_ _−_ **CAM** [HiRes] _c_ _−_ _M_ : _c ∈_ [ _C_ ] _\ ct_ (32)


         -          = **CAM** [HiRes] _ct_ _−_ **CAM** [HiRes] _c_ : _c ∈_ [ _C_ ] _\ ct_ (33)


         -          = **CAM** [Cntrst] ( _ct,c_ ) [:] _[ c][ ∈]_ [[] _[C]_ []] _[ \][ c][t]_ = **CAM** [Cntrst] _ct_ (34)


13


_Proof._ First, we define the set of all valid shifts _M_ :


Let _M_ := _{_ **M** _∈_ R _[C][×][d]_ [1] _[×][d]_ [2] : **M** _i_ = **M** _j_ _∀i, j_ _∈_ [ _C_ ] _}_ (22)


The matrix **M** _i_ _∈_ R _[d]_ [1] _[×][d]_ [2] can be arbitrary, provided it is constant _∀i_ _∈_ [ _C_ ]. Thus _|M|_ = _∞_ . We
will show that all HiResCAM explanations that differ by **M** _∈M_ form an equivalence class under
the _softmax_ operation. Consider the following set:


[ **CAM** ~~H~~ iRes] = _{_ **CAM** HiRes + **M** : **M** _∈M}_ (23)

We then show that any **CAM** ~~H~~ iRes with a corresponding shift **M** _′_ is a valid explanation (i.e., preserves
the final prediction). With logits _f_ deconstructed into HiResCAMs following Eq. (3), we have:










HiRes
**CAM** : _,i,j_ + **b**  = _σ_


_d_ 1 _,d_ 2




- ~~H~~ iRes **CAM** : _,i,j_ + **M** _[′]_ : _,i,j_ + **b**  (24)


_σ_








_d_ 1 _,d_ 2


_i_ =1 _,j_ =1


_i_ =1 _,j_ =1


Let _a_ = [�] _i_ _[d]_ =1 [1] _[,d]_ _,j_ [2] =1 **[M]** _c,i,j_ _[′]_ [for some] _[ c][ ∈]_ [[] _[C]_ []][.] [By property of] **[ M]** [:]











 - **CAM** [HiRes] : _,i,j_ + **b**


_i_ =1 _,j_ =1


_d_ 1 _,d_ 2


_i_ =1 _,j_ =1




- **CAM** [HiRes] : _,i,j_ + _a_ **1** _C_ + **b**  (25)





Applying Proposition 3.1, we have:


Thus, we have:


= _σ_


= _σ_





_d_ 1 _,d_ 2


 = _f_ ˜( **X** ) (26)





_d_ 1 _,d_ 2




~~H~~ iRes
**CAM** : _,i,j_ + **b**  _∀M_ _∈_ R _[d]_ [1] _[×][d]_ [2] (27)





 = _σ_


_f_ ˜( **X** ) = _σ_





 - **CAM** [HiRes] : _,i,j_ + **b**


_i_ =1 _,j_ =1


 _d_ 1 _,d_ 2

 


_i_ =1 _,j_ =1


(47)
We can thus compute class probabilities as a direct function of ContrastiveCAMs and the bias
vector.


14


~~C~~ ntrst
∴ **CAM** [Cntrst] _ct_ = **CAM** _ct_ (35)

This proves the first statement. Now, we can tend to the **CAM** [Recon] case:


~~R~~ econ HiRes
**CAM** _ct_ = **CAM** _ct_ _−_ _C_ [1]


_C_


_c_ =1


HiRes
**CAM** _c_ (36)


HiRes
By definition of **CAM**, we have:


= **CAM** [HiRes] _ct_ + _M_ _−_ _C_ [1]


_C_


_c_ =1


- **CAM** [HiRes] _c_ + _M_ (37)


= **CAM** [HiRes] _ct_ + _M_ _−_ _[C][ ·][ M]_


_C_


_[ M]_

_−_ [1]
_C_ _C_


_C_

- **CAM** [HiRes] _c_ (38)

_c_ =1


= **CAM** [HiRes] _ct_ _−_ _C_ [1]


_C_

- **CAM** [HiRes] _c_ = **CAM** [Recon] _ct_ (39)

_c_ =1


~~R~~ econ
∴ **CAM** [Recon] = **CAM** (40)
Proving the desired statements.


**Proposition 4.1.** _Softmax-activated class probabilities_ _f_ [˜] _can be expressed as a direct function of_
_ContrastiveCAMs and the bias vector._


_f_ ˜ _ct_ ( **X** ) =


- _C_

 - exp - **b** _c −_ **b** _ct_ _−_ - **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


_∀ct_ _∈_ [ _C_ ] (41)


_Where_ **CAM** [Cntrst] ( _ct,ct_ ) [=] **[ 0]** _[d]_ 1 _[×][d]_ 2 _[.]_


_Proof._ Individual class probabilities for logit vector _f_ are defined as:

_e_ _[f][ct]_
_f_ ˜ _ct_ = _σct_ ( _f_ ) =                - (42)

_i_ _[e][f][i]_

For some _ct_ _∈_ [ _C_ ].


We define our logit vector in terms of the elementwise difference to a target class _c_ :


**d** := _f_ _−_ _fct_ = _⇒_ _f_ = _fct_ + **d** (43)


Based on this definition, class probabilities can equivalently be computed as:


_e_ _[f][ct]_
_f_ ˜ _ct_ = 


_[f][ct]_ _e_ _[f][ct]_

[=]     _i_ _[e][f][i]_ _i_ _[e][f][ct]_


_e_ _[f][ct]_ _e_ _[f][ct]_

_i_ _[e][f][ct]_ [+] **[d]** _[i]_ [=] _e_ _[f][ct]_ [�]


_e_ _[f][ct]_ [�]


1

[=]     _i_ _[e]_ **[d]** _[i]_ _i_


(44)
_i_ _[e]_ **[d]** _[i]_


This re-contextualizes softmax as a direct function of the differences of class logits. We can further
deconstruct the difference by logit values:


_d_ 1 _,d_ 2

 - **CAM** [HiRes] _ct,i,j_ _[−]_ **[b]** _[c]_ _t_ (45)

_i_ =1 _,j_ =1


**d** _c_ = _fc −_ _fct_ =


_d_ 1 _,d_ 2

 - **CAM** [HiRes] _c,i,j_ + **b** _c −_


_i_ =1 _,j_ =1


Applying Definition 3.3, we have:

**d** _c_ = **b** _c −_ **b** _ct_ _−_           - **CAM** [Cntrst] ( _ct,c_ ) (46)


Substituting **d** _i_ from Eq. (46) into Eq. (44), we have:


1
_f_ ˜ _ct_ ( **X** ) = - _Ci_ =1 [exp] - **b** _c −_ **b** _ct_ _−_ [�] **CAM** [Cntrst] ( _ct,c_ ) - =


- _C_

 - exp - **b** _c −_ **b** _ct_ _−_ - **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


**Proposition 4.2.** _Given bias-free classifier h, we can precisely associate the impact of specific regions,_
_encoded by binary mask H, to the computation of cross-entropy loss._


_L_ CE( _f_ ( **X** ) _,_ **y** _, H_ ) = log


- _C_

 - exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[−]_ �(1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ ) - [�]

_c_ =1

(48)


_Proof._ Setting **b** = 0 to the result from Proposition 4.1, we have:


_f_ ˜ _ct_ ( **X** ) =


- _C_

 - exp - _−_ - **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


(49)


For target class _ct_ _∈_ [ _C_ ]. Let _H_ and (1 _−_ _H_ ) define core and non-core masks respectively; these are
disjoint. We can use this to further disassociate ContrastiveCAMs:


_f_ ˜ _ct_ =


- _C_

 - exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[−]_ �(1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


(50)


For one-hot encoded target vector **y** and target class index _ct_, cross-entropy loss is defined as:


_LCE_ ( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_


_C_

- **y** _c_ log _f_ [˜] _c_ = _−_ log _f_ [˜] _ct_ (51)


_c_ =1


To which we can substitute softmax using Eq. (50):


_LCE_ ( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ log


- _C_

 - exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[−]_ �(1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ ) - [�] _[−]_ [1]

_c_ =1


- _C_

 - exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[−]_ �(1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ ) - [�]

_c_ =1


- _C_

 


= log


(52)


As core and non-core masks are disjoint, Eq. (52) enables us to identify the logit contributions from
the core and non-core regions respectively.


**Theorem** **4.6.** _A_ _sequence_ _of_ _predictors_ _fn_ _⊂F_ _that_ _converges_ _to_ _the_ _optimal_ _R_ CFCE _-risk_ _also_
_converges_ _to_ _the_ _Bayes-optimal_ _R_ CCRM _-risk._ _Equivalently,_ _in_ _the_ _realizable_ _setting,_ _L_ CFCE _is_
_classification-calibrated._


_R_ CFCE( _fn_ ) _→R_ _[∗]_ CFCE [=] _[⇒R]_ [CCRM][(] _[f][n]_ [)] _[ →R]_ CCRM _[∗]_ (53)


_Where R_ CFCE( _f_ ) _is:_


_R_ CFCE( _f_ ) := E( **X** _,_ ( _H,_ **y** )) _∼D_ [ _L_ CFCE( _f_ ( **X** ) _,_ **y** _, H_ )] (54)


_Proof._ We start by restating Definition (4.5):


        - _C_
_L_ CFCE( _f_ ( **X** ) _,_ **y** _, H_ )) = log 


- exp - _−_ - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ )

_c_ =1


=1 (55)

 - - [�]
+ (1 _−_ _H_ ) _⊙|_ **CAM** [Cntrst] ( _ct,c_ ) _[|]_


�� Cntrst  exp (1 _−_ _H_ ) _⊙|_ **CAM** ( _ct,c_ ) _[|]_

�� Cntrst�
exp _H_ _⊙_ **CAM** ( _ct,c_ )


15





 (56)


= log


 _C_

 

_c_ =1


-      - [�]
+ (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] ( **X** ) _i_


16


We can observe that _R_ CFCE( _f_ ) takes the following form:





_R_ CFCE( _f_ ) = E( **X** _,_ ( _H,_ **y** ) _∼D_ log







 _C_
�


 _c_ =1



�� Cntrst  exp (1 _−_ _H_ ) _⊙|_ **CAM** ( _ct,c_ ) _[|]_

�� Cntrst�
exp _H_ _⊙_ **CAM** ( _ct,c_ )


- �� _sc_







 (57)





_R_ _[∗]_ CFCE [= inf] _[f][ R]_ [CFCE][(] _[f]_ [)][ is predicated on each summand] _[ s][c]_ _[→]_ [0][.] [We have that:]


�� Cntrst  - 
exp ��(1 _−_ _H_ ) _⊙|_ **CAM** Cntrst( _ct,c_ �) _[|]_  _≥_

exp _H_ _⊙_ **CAM** ( _ct,c_ )


_C_


_c_ =1


  - �� Cntrst ��
inf _f_ exp (1 _−_ _H_ ) _⊙|_ **CAM** ( _ct,c_ ) _[|]_

   - �� Cntrst��
sup _f_ exp _H_ _⊙_ **CAM** ( _ct,c_ )


inf
_f_


 _C_

 

_c_ =1


(58)
Given sufficiently expressive _F_ by assumption of realizability of _R_ _[∗]_ CCRM [, as] _[ n][ →∞][,]_ _[f][n]_ [converges]
uniformly towards the equality case thus admitting the following dual objective for each _sc_ :


              - �� Cntrst ��
_R_ CFCE( _fn_ ) _→R_ _[∗]_ CFCE _[⇐⇒]_ inf _f_ exp� ��(1 _−_ _H_ ) _⊙|_ **CAM** Cntrst( _ct,c_ ��) _[|]_ _∀c ∈_ [ _C_ ] (59)

sup _f_ exp _H_ _⊙_ **CAM** ( _ct,c_ )


With the absolute _| · |_ operator over numerator’s exponent and the realizability assumption, we have:


inf
_f_


- �� ��
exp (1 _−_ _H_ ) _⊙|_ **CAM** [Cntrst] ( _ct,c_ ) _[|]_ = 1 _⇐⇒∥_ (1 _−_ _H_ ) _⊙_ **CAM** [Cntrst] ( _ct,c_ _[′]_ ) _[∥]_ [= 0] (60)


This satisfies the constraint from Definition 4.4 and further implies (by absolute homogeneity of the
norm) that each non-core region has no contribution to the final classification.


Next, we can tend to the denominator.


Let _f_ _[∗]_ = arg sup
_f_


By convexity of exp, we have that:


�� _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) (61)


exp �� _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _,f_ _[∗]_ [�] _≥_ sup
_f_


- �� ��
exp _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) (62)


The realization of _f_ _[∗]_ satisfies the following condition:

    - _H_ _⊙_ **CAM** [Cntrst] ( _ct,c_ ) _[>]_ [ 0] _∀c ∈_ [ _C_ ] (63)


Which is sufficient to show the largest logit is that of the target class _ct_ . Thus arg max( _f_ ( **X** )) =
arg max( **y** ) _∀_ ( **X** _,_ ( _H,_ **y** )) _∼D_ = _⇒_ E( **X** _,_ ( _H,_ **y** )) _∼D_ [ _ℓ_ ( _f_ ( **X** _,_ **y** ))] = 0 which gives us:


_R_ CFCE( _fn_ ) _→R_ _[∗]_ CFCE [=] _[⇒R]_ [CCRM][(] _[f][n]_ [)] _[ →R]_ CCRM _[∗]_ (64)


Proving the consistency of _L_ CFCE as a surrogate minimizer to _R_ CCRM.


**Proposition B.1.** _We can integrate background suppression to the definition of binary cross-entropy_
_using the following formulation:_





 [�]


[�] _Hi ⊙_ **CAM** [HiRes] _i,j,k_ _−_ 

_j,k_ _j,k_








 _φ_


_L_ CFBCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ [1]

_C_


_C_


_i_ =1


**y** _i_ log


(1 _−_ _Hi_ ) _⊙|_ **CAM** [HiRes] _i,j,k_ _[|]_
_j,k_











(65)


-      - [�]
+ (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] ( **X** ) _i_


17


_Proof._ We will prove for the multilabel setting, which is a generalization of binary cross-entropy. For
binary vector **y** (i.e., **y** _i_ _∈{_ 0 _,_ 1 _} ∀i_ ), class-specific core masks _Hi_, and sigmoid _φ_ activated logits _f_,
denoted _f_ [˜], binary cross-entropy is defined as:


_L_ BCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ [1]

_C_


_C_


_i_ =1


- - ��
**y** _i_ log _f_ [˜] _i_ + (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] _i_ (66)


= _−_ [1]

_C_


_C_

- [ **y** _i_ log _φ_ ( _fi_ ) + (1 _−_ **y** _i_ ) log (1 _−_ _φ_ ( _fi_ ))] (67)


_i_ =1


Setting **b** = 0, we can substitute Eq. (3) within the first term:




 [�] **CAM** [HiRes] _i,j,k_


_j,k_





 

      -        + (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] _i_ 


 (68)





 **y** _i_ log _φ_


= _−_ [1]

_C_


_C_


_i_ =1


Similar to Proposition 4.5, we can break down each HiResCAM to core and spurious components.
For non-target indices, we seek to reducing logit values across the entire input image. Therefore, we
do not disassociate logit values for the second term.





 [�]


[�] _Hi ⊙_ **CAM** [HiRes] _i,j,k_ + 

_j,k_ _j,k_


_L_ BCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ [1]

_C_


_C_


_i_ =1


**y** _i_ log _φ_


(1 _−_ _Hi_ ) _⊙_ **CAM** [HiRes] _i,j,k_

_j,k_








                 -                 - [�]
+ (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] _i_


(69)


The current formulation motivates activating either the core or non-core for positive classification, and
motivates de-activating every pixel of the non-positive class. We penalize activation on the non-core
regions for the positive class only:


[�] _Hi ⊙_ **CAM** [HiRes] _i,j,k_ _−_ 

_j,k_ _j,k_








 _φ_





_L_ CFBCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ [1]

_C_


_C_ 
- **y** _i_ log


_i_ =1


 [�]


(1 _−_ _Hi_ ) _⊙|_ **CAM** [HiRes] _i,j,k_ _[|]_
_j,k_











                  -                  - [�]
+ (1 _−_ **y** _i_ ) log 1 _−_ _f_ [˜] ( **X** ) _i_


This gives us the core-focused binary cross-entropy formulation.


B CORE-FOCUSED CROSS-ENTROPIC ADAPTATIONS


B.1 CORE-FOCUSED BINARY CROSS-ENTROPY


(70)


For sigmoid-activated binary / multilabel classification tasks, we leverage similar principles to define
core-focused binary cross-entropy. Since we do not have the contrastive process in softmax-activation,
this definitions relies only on HiResCAMs. We represent sigmoid activation using _φ_ and admit _C_
target-region masks, denoted _Hi_ for each class _i ∈_ [ _C_ ]. In addition, instead of one-hot encoding, we
now have binary vector **y** (i.e., **y** _i_ _∈{_ 0 _,_ 1 _} ∀i_ ).
**Proposition B.1** (Core-Focused Binary Cross-Entropy) **.** _We can integrate background suppression to_
_the definition of binary cross-entropy using the following formulation:_


[�] _Hi ⊙_ **CAM** [HiRes] _i,j,k_ _−_ 

_j,k_ _j,k_








 _φ_





_L_ CFBCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _−_ [1]

_C_


_C_ 
- **y** _i_ log


_i_ =1


 [�]


�(1 _−_ _Hi_ ) _⊙_ ��� **CAM** HiRes _i,j,k_ ���

_j,k_








(71)


Where the third newly introduced term within the exponent expresses differential contrast.


C TRAINING DETAILS


**Hyperparameters.** To _mitigate reward-hacking_ our proposed approach, we selected a consistent
set of hyperparameters that generally performs well and use it across all our experiments. We train
each model using the Adam optimizer (Kingma, 2014) for 150 epochs with a learning rate of 5 _·_ 10 _[−]_ [4],
using a linear warmup of 5 epochs followed by Cosine Annealing (Loshchilov and Hutter, 2016)
for the remaining 145 epochs. We use a weight decay of 10 _[−]_ [4], a batch size of 768. For divergence
regularized approaches, we used _**λ**_ = _{_ 50 _,_ 10 [3] _,_ 10 _}_ .


**Reproducibility.** The source code, datasets, experiments, evals, and model weights are published
under a permissive license and can be found at _[redacted for double blind peer-review]._


C.1 ARCHITECTURE MODIFICATIONS


The architecture used for training was ResNet-50 (He et al., 2016), initialized with ImageNet. We
introduce the following three key modifications:


18


**Divergence Regularization** Similar to Definition 4.5, we define a divergence term for the target
class to motivate activation of the entire core region within the training objective.


**Definition B.2** (Regularized Core-Focused Binary Cross-Entropy) **.**


_L_ RCFBCE( _f_ ( **X** ) _,_ **y** _, H_ ) = _L_ CFBCE + _λ_ 1
_∥_ **y** _∥_ 1


_C_

- **y** _iD_ KL - _σ_ ( _λ_ 2 _Hi_ ) _|| σ_ - _λ_ 3 **CAM** [HiRes] _i_ �� (72)


_i_ =1


B.2 CUTMIX WITH CORE-FOCUSED CROSS-ENTROPY


CutMix (Yun et al., 2019) is a batch-wise augmentation technique that encourages better regularization
by a) “cutting” out a randomized rectangle (randomized portion remaining consistent across the
batch) of a given image and b) “mixing” the cut-out with it’s neighbor. The corresponding labels are
mixed by a randomly sampled parameter _λ_ .


**Definition** **B.3** (CutMix with Core-Focused Cross-Entropy) **.** Let segmentation mask _H_ take the
following form:

         - _−_ 1 pixel does not contain any class
_H_ :=
_c_ pixel contains class _c_


Also, let 1 _a_ be the indicator function applied elementwise for some _a ∈_ R.


Then, Core-Focused Cross Entropy (4.5) with CutMix is formulated as follows:


��
_L_ CM_CFBCE( _f_ ( **X** ) _, H,_ **y** ) = log


  
  exp _−_ 1 _c_ ( _H_ ) _⊙_ **CAM** [Cntrst] ( _c,i_ )

_i_


 - + _|_ 1 _−_ 1( _H_ ) _⊙_ **CAM** [Cntrst] ( _c,i_ ) _[|]_ [ +] 1 _i_ ( _H_ ) _⊙_ **CAM** [Cntrst] ( _c,i_ )


(73)
��


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


**Removed** **final** **downsampling.** For images of size (224 _,_ 224), the final downsampling layer
converts the latent feature embeddings from _d_ 1 = _d_ 2 = 14 to 7. This prohibitively reduces the size of
the activation map, and making it hard to capture relevant features. We replace the stride of the final
downsampling convolution to (1 _,_ 1), matching that of the definition used through the rest of ResNet.


**Removed final bias.** The bias vector **b** within _h_ is not involved in the computation of the class
activation map. However, it does affect predictions in a way that is not explained by ContrastiveCAMs.
To maintain faithfulness of the explanations, we omit the bias from the final model architecture.


**Removed final BatchNormalization & ReLU.** Since the HiResCAM construction establishes
convolution followed immediately by GAP, the standard architecture which uses BatchNormalization
& ReLU layers after each convolution, does not directly explain the class score. We therefore
neutralize those functions for the final convolutional block. This recovers the faithfulness guarantee.


Note that the above changes correspond only to the final convolutional block of the backbone _g_ and
the bias of the linear classifier _h_ ; the rest of the architecture remains consistent.


19
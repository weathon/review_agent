# PARAMETER RELEASE AND KNOWLEDGE REUSE FOR CLASS-INCREMENTAL SEMANTIC SEGMENTATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Class-incremental semantic segmentation aims to progressively learn new classes
while preserving previously acquired knowledge. This task becomes particularly
challenging when prior training samples are unavailable due to data privacy or
storage restrictions, resulting in catastrophic forgetting. To address this issue,
knowledge distillation is widely adopted as a constraint by maximizing the similarity of representations between the current model (learning new classes) and
the previous model (retaining old ones). However, knowledge distillation inherently preserves the old-knowledge distribution with minimal modification. This
constraint limits the parameters available for learning new classes when substantial
information from old classes is retained. Furthermore, the acquired old knowledge
is often ignored to facilitate the learning of new knowledge, resulting in a waste
of previously learned procedures. The above two problems result in the risk of
class confusion and deviating from the performance of joint learning. Based on
such analysis, we propose **D** istribution-based **K** nowledge **D** istillation (DKD) via a
minimization–maximization distribution strategy. On the one hand, to alleviate the
parameter competition between old and new knowledge, we minimize the distribution of old knowledge after releasing low-sensitivity parameters to old classes. On
the other hand, to effectively utilize the valuable knowledge previously acquired,
we maximize shared-knowledge distribution between the old and new knowledge
after approximating the new knowledge distribution via Laplacian-based projection
estimation. The proposed method achieves an excellent balance between stability
and plasticity in nine diverse settings on Pascal VOC and ADE20K. Notably, its
average performance approaches that of joint learning (upper bound) while effectively reducing class confusion. The source code is provided in the supplementary
material and will be made publicly available upon acceptance.


1 INTRODUCTION


Recently, supervised semantic segmentation Xie et al. (2021); Jain et al. (2023); Liao & Kong
(2025) has made significant progress, which typically requires closed-set datasets where all classes
are obtained at once for manual annotation. However, in real-world applications, new classes
continuously emerge and models trained on old data struggle to adapt to new class data. A naive
approach is to retrain the model on a combined dataset of old and new classes, known as _joint training_,
but this is time-consuming and the old data is often partially inaccessible due to privacy or storage
restrictions Yang et al. (2023); Zhao et al. (2023); Li & Hoiem (2017); Baek et al. (2022). Therefore,
class-incremental semantic segmentation (CISS) Cha et al. (2021b); Maracani et al. (2021); Zhang
et al. (2022b) emerged with the goal of continuously learning new classes while mitigating forgetting
old ones, even when old classes are unavailable. This task is essential in real-world scenarios such as
autonomous driving, medical image analysis, and environmental monitoring.


In recent years, CISS methods have focused on catastrophic forgetting Cha et al. (2021b); Douillard
et al. (2021b); Yuan & Zhao (2024) and background shift Cermelli et al. (2020); Park et al. (2024);
Qiu et al. (2023), enabling models to retain previously learned knowledge while adapting to new data.
To alleviate the aforementioned issues, some methods Yoon et al. (2017); Qin et al. (2021) attempt to
dynamically expand the modules, but this strategy introduces an additional inference burden due to
the extra parameters. Hence, most CISS methods Cha et al. (2021b); Douillard et al. (2021b); Baek
et al. (2022); Shang et al. (2023); Wang et al. (2024) rely on static architectures, meaning the capacity


1


Aeroplane Person Sheep Sofa Cat TVmonitor


Figure 1: Problems and Analysis. KD, a widely adopted strategy, inherently preserves the old
knowledge distribution with minimal change. However, as the number of new classes grows, this
strategy aggravates parameter competition and causes the capacity for new-class distributions to
become crowded. In addition, acquired old knowledge remains unused for step-wise guidance,
neglecting its valuable role in supporting the learning of new knowledge.


of the overall parameter-fitting knowledge space is fixed. Pseudo-label based cross-entropy (CE) Cha
et al. (2021b); Cermelli et al. (2020) and knowledge distillation (KD) Zhao et al. (2023); Wang
et al. (2024); Yang et al. (2022); Shang et al. (2023); Baek et al. (2022); Cong et al. (2023) are
typical strategies in CISS. Mainstream KD approaches used in static architectures can be broadly
categorized into two types. The first performs multi-level feature-based knowledge distillation (MKD)
by enforcing similarity between the frozen old model and the new model Zhao et al. (2023); Wang
et al. (2024). The second, to alleviate guidance from potentially incorrect pixel-level information in
the frozen old model, employs confidence map-based knowledge distillation (CKD), which retains
high-confidence pixel information for feature distillation Yang et al. (2022); Shang et al. (2023).
These mainstream KD approaches under a static parameter-fitting space hide a parameter competition:
excessive parameter fitting for old knowledge hinders the learning of new classes, while overfitting to
new knowledge leads to forgetting of old classes. When using KD to construct the similarity of the old
representations during different steps, the fitting capacity for new knowledge becomes increasingly
crowded as more classes are learned. This leads to a crowded distribution of new knowledge, as
illustrated in the light-blue region of Fig. 1. Moreover, this constraint causes the valuable knowledge
previously acquired to remain unused when learning new classes, as shown in the light-yellow region
of Fig. 1. Consequently, the model struggles to balance stability and plasticity and becomes prone to
class confusion, as illustrated in the right half of Fig. 1. This raises a critical question: **Can we release**
**parameters and simultaneously reuse the acquired knowledge when learning new knowledge**
**without introducing an additional inference burden?**


To address the above issue, we propose a **D** istribution-based **K** nowledge **D** istillation (DKD) via the
minimization–maximization strategy. DKD releases previously occupied low-sensitivity parameters
to alleviate competition between old and new knowledge. Additionally, Laplacian-based projection
estimation produces two attention maps that identify reusable knowledge to guide the learning
of new knowledge: (1) A position map that represents the spatial regions in which all categories
coexist harmoniously, and (2) a confidence map that indicates the necessity of spatial reuse of old
category information within the distribution of the new model. Subsequently, we maximize the
shared knowledge distribution between old and new knowledge. As shown in Fig. 2(a-c), unlike
prior KD methods, DKD emphasizes parameter release and knowledge reuse for the learning of
new knowledge to mitigate parameter competition and reduce wasted knowledge. Experiments
demonstrate that dynamic parameter release facilitates greater parameter differences between the
incremental and initial steps (Fig. 2(d)). Furthermore, knowledge reuse in DKD not only mitigates
catastrophic forgetting (as seen in the slower performance drop across incremental steps in the green
curve of Fig. 2(e)), but also significantly enhances new class learning, achieving higher pixel-level
classification accuracy with increasing incremental steps (Fig. 2(f)). Experiments on Pascal VOC


2


... Step


...


Frozen Adjustable


Confidence map


**~** Constraint


Figure 2: Comparison with existing KD methods. Unlike previous approaches that freeze the
old model, our method dynamically adjusts knowledge distribution through parameter release and
knowledge reuse (as shown in a–c). This enables dynamic parameter adjustment (as shown in d),
mitigates catastrophic forgetting (as shown in e), and enhances new class learning (as shown in f).


2012 and ADE20K datasets demonstrate the effectiveness of our approach. Our main contributions
are summarized as follows:


    - We show that mainstream KD for CISS overlooks _parameter_ _competition_ and _underuti-_
_lization of acquired knowledge_, motivating our Distribution-based Knowledge Distillation
(DKD) with a **minimization–maximization** distribution strategy.

    - To mitigate parameter competition, we **minimize** the distribution of old knowledge after
releasing low-sensitivity occupied parameters; to reduce the underutilization of acquired
knowledge, we reuse the acquired old knowledge to guide new class learning and **maximize**
the shared knowledge distribution. These two components constitute DKD.

    - Theoretical analysis and extensive experiments validate our approach, delivering state-ofthe-art average performance and results close to the joint-training (upper bound).


2 RELATED WORK


In Class-Incremental Semantic Segmentation (CISS), the model is required to continuously learn
new classes for semantic segmentation while preserving the knowledge of previously learned classes,
addressing the challenges of catastrophic forgetting and background shift. Current research on CISS
can be broadly divided into three main types of methods: 1) Replay-based approaches Cha et al.
(2021b); Maracani et al. (2021); Rebuffi et al. (2017); Chen et al. (2025); Yu et al. (2024); Oh et al.
(2022), which mitigate forgetting by storing or generating past data for rehearsal. Recall Maracani
et al. (2021) leverages generative adversarial networks and web-crawled data to synthesize replayable
samples of previously learned classes for continual learning. To avoid the burden of storing old
data, TiKP Yu et al. (2024) employs text-to-image generation to replay samples of previously
learned classes. 2) Dynamic architecture-based approaches Yoon et al. (2017); Qin et al. (2021);
Aljundi et al. (2017); Yan et al. (2021), which expand the network structure to accommodate new
classes. BNS Qin et al. (2021) dynamically constructs task-specific networks to mitigate catastrophic
forgetting while facilitating knowledge transfer across tasks. EG Aljundi et al. (2017) progressively
introduces task-specific experts as the number of tasks increases, enabling dynamic expert selection
during inference. 3) Regularization-based approaches preserve previously learned knowledge by
constraining representations. To address background shift, MIB Cermelli et al. (2020) remodels
the background in the ground truth based on the output of the previous steps. MicroSeg Zhang
et al. (2022b) introduces mask proposals to refine pseudo-labels. Building on the effectiveness
of knowledge distillation to alleviate catastrophic forgetting Yang et al. (2019); Heo et al. (2019);


3


Figure 3: Overview of the proposed Distribution-based Knowledge Distillation (DKD) via a minimization–maximization distribution strategy. Our strategy introduces distributional constraints between
old and new knowledge—minimizing the old knowledge distribution via _LMin_, while maximizing
the shared knowledge through _LMax_ after estimating the new knowledge distribution with _LEsti_ .
This approach achieves parameter release (via _LMin_ ) and knowledge reuse (via _LEsti_ and _LMax_ ).


Wang & Yoon (2021), IL Michieli & Zanuttigh (2019) leverages the outputs of the old model,
typically with frozen parameters, to supervise the trainable new model by maximizing the similarity
between intermediate representations and logits via distillation losses, a process known as multilevel feature map-based knowledge distillation (MKD). Methods such as Comformer Cermelli et al.
(2023), PLOP Douillard et al. (2021b), and CLIS Zheng et al. (2021) adopt a confidence map-based
knowledge distillation (CKD), where high-confidence regions from the old model are selected and
retained at the pixel level to guide the distillation process. However, our method differs in that our
goal is to mitigate parameter competition between old and new knowledge while promoting the reuse
of acquired knowledge, thereby effectively balancing stability and plasticity in CISS.


3 METHOD


3.1 PROBLEM DEFINITION


In CISS, the learning process involves a sequence of incremental steps, denoted as t=1,2,3,..., T.
Following prior works Cha et al. (2021b); Douillard et al. (2021b); Cermelli et al. (2020; 2023), we
adopt the overlapped setting, where the ground truth only includes labels for the currently learned
class _Ct_, while all other classes, both previously learned and those to be presented in future steps, are
treated as background _C_ 0. The CISS model _ft,θ_ in the t-th incremental learning step is parameterized
by _θ_, comprising a feature extractor and a classifier. Given an input image _x_, the predicted result is
obtained as _yt_, where the predicted category may belong to the old classes _C_ 1: _t−_ 1, the new classes
_Ct_, or the background class _C_ 0.


3.2 DISTRIBUTION-BASED KNOWLEDGE DISTILLATION


**(a)** **Parameter-driven** **minimization** **of** **old** **knowledge** **distribution.** Rather than freezing the
parameters _θt−_ 1 of the old model, we dynamically update their utilization at each step to enable
adaptive knowledge retention and effective parameter reuse. To achieve this, as shown in Fig. 3(a),
we first retrieve the weight matrix _Wl_ for each layer _l_ in the model. Then we compute the layer-wise
Euclidean norm (L2) of the weight matrix to assess its importance. For layer _l_, the weight matrix for a
convolutional layer is _Wl_ [Conv] _∈_ R _[C]_ [out] _[×][C]_ [in] _[×][H][k][×][W][k]_, and for a fully connected layer, _Wl_ [FC] _∈_ R _[C]_ [out] _[×][C]_ [in] .
Here, _C_ out and _C_ in denote the output and input channels (e.g., 3 for RGB), and _Hk_ and _Wk_ represent


4


the height and width of the kernel. The calculation for each layer is as follows:


**(c) Entropy-induced optimization of overlap between new and old knowledge distribution.** To
maximize the shared knowledge between the old and new distributions, we propose a loss constraint
_L_ Max based on information theory Ash (2012). The objective is to encourage the model to exhibit


5


_o_ =1


_C_ out

 


_Wk_


_Hk_


_C_ in


_i_ =1


_C_ in


_h_ =1


(1)


_∥Wl∥_ 2 =






 - - - - ( _Wl,o,i,h,w_ ) [2] _,_ if _Wl_ [Conv] is a convolutional layer _._

_o_ =1 _i_ =1 _h_ =1 _w_ =1

_C_ out _C_ in

 - - [2] [FC]





_o_ =1


- ( _Wl,o,i_ ) [2] _,_ if _Wl_ [FC] is a fully connected layer _._

_i_ =1


After computing the Euclidean norm, we apply a pruning threshold _τ_ = 0 _._ 1 (with experimental analysis in Appendix C.5) to retain significant weights and prune those below the threshold. Specifically,
if the norm of a filter or output unit is less than _τ_, it is pruned, and its weights and bias are set to zero.
We define a binary pruning mask _Pl_ for each layer, where _Pl,i_ = 1 means the _i_ -th unit in the _l_ -th
layer is kept, and _Pl,i_ = 0 means it is pruned. The selection decision is expressed as _Pl,i_ = 1 when
_∥Wl,i∥_ 2 _≥_ _τ_, and _Pl,i_ = 0 otherwise. Based on the resulting binary mask _Pl_, the weights _Wl_ and
biases _Bl_ of the old model are updated according to _W_ [ˆ] _l_ = _Wl ⊙_ _Pl_ and _B_ [ˆ] _l_ = _Bl ⊙_ _Pl_, respectively.
By adjusting the utilization of the old parameters in this way, the model maintains essential knowledge
while freeing up capacity for new classes. For each pixel location ( _h, w_ ) associated with previously
learned classes c _∈_ 1 _, ..., Ct−_ 1, the label _yc_ _[∗]_ [(] _[h, w]_ [)][ is updated to alleviate background shift:]
_yc_ _[∗]_ [(] _[h, w]_ [) =] [ 1] [[] _[y][t]_ [(] _[h, w]_ [) =] _[ c]_ []ˆ] _[y]_ _t_ _[c]_ _−_ 1 [(] _[h, w]_ [)] _[,]_ (2)
where _yt_ ( _h, w_ ) is the prediction of step _t_ and _y_ ˆ _t−_ 1 represents the prediction after parameter release.
1 denotes an indicator function that outputs 1 if the condition inside the parentheses is satisfied, and
0 otherwise. To dynamically minimize the distribution of old classes in _yt_ ( _h, w_ ) during the current
learning step _t_ after parameter release, an optimization loss is introduced:


_W_

- _yc_ _[∗]_ [(] _[h, w]_ [)] _[y][t]_ [(] _[h, w]_ [)] _[.]_ (3)

_w_ =1


1
_LMin_ = _−_
_H_ _× W_


_H_


_h_ =1


Through this computation, the output distribution of the old classes in the current-step model is
encouraged to closely match the output of the old model after parameter release. In this process, the
output of the parameter-released old model serves as the bound for this optimization, enabling the
current step model to dynamically minimize the usage of old-class parameters via the loss constraint.
Further theoretical analysis is provided in Appendix A.1.


**(b) Laplacian-based projection estimation for constructing the new knowledge distribution.** To
effectively leverage previously acquired knowledge when forming the new knowledge distribution,
we calculate a position map and a confidence map to identify where old knowledge is reusable
and how strongly it applies, and we constrain their utilization via the loss _LEsti_ . We compute the
second-order gradient to identify low-curvature regions and store them in the position map _Pt_ ( _h, w_ ),
where the representations of old and new knowledge exhibit minimal difference and can coexist.


    - _∂_ 2
_Pt_ ( _h, w_ ) =


_∂w_ [2]


_∂_ 2 _[∂]_ [2]

_∂h_ [2] [+] _∂w_


��
_yc_ _[∗]_ [(] _[h, w]_ [)] �� _ft_ ( _h, w_ ) _−_ _ft−_ 1( _h, w_ )��2


(4)


Additionally, the confidence map _Ct_ ( _h, w_ ) is generated to quantify the projection strength of old
knowledge along the direction of the new knowledge distribution. Given the current model feature
_ft_ ( _h, w_ ) and the label _yc_ _[∗]_ [(] _[h, w]_ [)][ of Eq. 2, the confidence map is computed as:]


_c_ [(] _[h, w]_ [)] _[, f][t]_ [(] _[h, w]_ [)] _[ >]_
_Ct_ ( _h, w_ ) = _[< y][∗]_


_._ (5)
_||ft_ ( _h, w_ ) _||_ 2


Here, _<_ _·, ·_ _>_ denotes the dot product. A higher value of _Ct_ ( _h, w_ ) indicates a stronger projection
between _yc_ _[∗]_ [(] _[h, w]_ [)] [and] _[f][t]_ [(] _[h, w]_ [)][,] [suggesting] [a] [higher] [likelihood] [of] [reusing] [previous] [knowledge]
in that region. Finally, to enforce the alignment between the feature representation _ft_ and the
inferred coexisting region, we define the loss _Llap_ with the guidance of _P_ ( _h, w_ ). Besides, we
leverage the confidence map to guide the _Lpro_ loss to constrain the model to preserve old knowledge
representations in high-confidence regions. As shown in Fig. 3(b), the overall loss of Laplacian-based
projection estimation for the construction of a new knowledge distribution is defined as:


_Lpro_

- �� 
- _H_ _W_ 


_W_


+ _γ_


_H_


_h_ =1


_LEsti_ =


1
1 _−_
_H_ _× W_


_Llap_

- �� _∥ft −Pt∥_ 2 _,_ (6)


_Ct_ ( _h, w_ ) _yc_ _[∗]_ [(] _[h, w]_ [)]
_||ft_ ( _h, w_ ) _||_ 2


_w_ =1


where _γ_ = 1 by default, with experiments in Section 4.3 and theoretical analysis in Appendix A.2.


low conditional entropy and high marginal entropy in its predictions. Given a batch of predicted
probabilities _y_ ( _[b]_ _t,c_ ) _[∈]_ [R][ for category] _[ c][ ∈{]_ [1] _[, . . ., C][t][}]_ [ and sample] _[ b][ ∈{]_ [1] _[, . . ., B][}]_ [, we define marginal]
probability _y_ ( _t,c_ ) as the average prediction across the entire batch for class _c_ :


where _|Ct−_ 1 _|_ is the number of old classes for normalization. This loss encourages more general
knowledge between the old and new class distributions by maximizing shared knowledge representation while maintaining discriminative diversity across classes. Related theoretical analysis is provided
in Appendix A.3.


The overall optimization objective of DKD is formulated as follows:
_L_ total = _L_ Min + _L_ Esti + _L_ Max + _L_ CE _,_ (11)
where _L_ CE denotes the cross-entropy loss Zhang & Sabuncu (2018).


4 EXPERIMENTS


4.1 EXPERIMENT SETUPS


Following previous works Cermelli et al. (2020); Yang et al. (2023); Zhang et al. (2023); Park et al.
(2024), we evaluate our method on Pascal VOC 2012 Everingham et al. (2010) and ADE20K Zhou
et al. (2017). We use ViT-B/16 Dosovitskiy et al. (2021) pretrained on ImageNet Deng et al. (2009)
as the backbone and a decoder with two transformer blocks at 512×512 resolution. Optimization is
performed using SGD Ketkar (2017) with momentum 0.9 and weight decay 1 _×_ 10 _[−]_ [5] for 64 epochs
per step. The learning rate starts at 1 _×_ 10 _[−]_ [3] and is reduced to 1 _×_ 10 _[−]_ [4] (Pascal VOC) or 5 _×_ 10 _[−]_ [4]
(ADE20K) in incremental steps. Experiments are run on 6 RTX 3090 GPUs with an Intel Xeon Gold
6226R CPU using PyTorch. Performance is evaluated using mean Intersection over Union (mIoU).


4.2 COMPARISON WITH THE STATE-OF-THE-ARTS
**Quantitative** **analysis** **on** **Pascal** **VOC** **2012** **and** **ADE20K.** In addition to the widely explored
incremental settings of 15-1, 19-1, and 15-5 in previous works, we further evaluate the effectiveness
of our method in more challenging scenarios, specifically 10-1 and 2-2. These two settings involve
a larger number of incremental steps, making them more representative of practical incremental
learning scenarios. As shown in Tab. 1, our method achieves notable improvements over previous
approaches. For example, under the 2-2 setting, our method achieves a 3 _._ 9% improvement on the
old classes and a 1 _._ 8% gain on the new classes, leading to an overall improvement of 4 _._ 7% in the
combined class performance. When averaged across all five incremental settings, our method achieves
a gain of 1 _._ 7%, which demonstrates its effectiveness and generalizability in various CISS settings. We
further evaluate the effectiveness of our method on the ADE20K dataset under 4 different incremental
settings, as shown in Tab. 2. The results demonstrate robustness in handling more complex and
large-scale CISS tasks. The average performance closely matches that of joint training, which is
considered the upper bound under this architecture in CISS.


**Qualitative analysis of forgetting resistance on old classes and plasticity on new classes.** As
shown in Fig. 4(a), prior methods tend to misclassify parts of the class “horse” as “sheep”, indicating
partial forgetting of old knowledge. Besides, previous methods show unsatisfactory sensitivity to new


6


_y_ ( _t,c_ ) = _B_ [1]


_B_

- _y_ ( _[b]_ _t,c_ ) _[.]_ (7)

_b_ =1


The marginal entropy is then calculated as:


_H_ ( _Yt_ ) = _−_


_Ct_

- _y_ ( _t,c_ ) log( _y_ ( _t,c_ )) _,_ (8)

_c_ =1


where high marginal entropy _H_ ( _Yt_ ) reflects a well-balanced category distribution in the new feature
space, helps to avoid overfitting to specific old or new categories and promotes diversity. The
conditional entropy, measuring the uncertainty of new knowledge given the old, is defined as:


_Ct_


_H_ ( _Yt_ _| Yt−_ 1) = _−_ [1]

_B_


_B_


_b_ =1


- _y_ ( _[b]_ _t,c_ ) [log(] _[y]_ ( _[b]_ _t,c_ ) [)] _[,]_ (9)

_c_ =1


where low conditional entropy _H_ ( _Yt_ _| Yt−_ 1) indicates that the new knowledge distribution establishes
effective dependencies on the old knowledge, facilitating knowledge retention and constructive
transfer of prior information to form class-specific knowledge of step _t_ . Finally, as shown in Fig. 3(c),
we define the _L_ Max loss to maximize the overlap between the old and new distributions as:


_[|][ Y][t][−]_ [1][)]
_L_ Max = _−_ _[H]_ [(] _[Y][t]_ [)] _[ −]_ _[H]_ [(] _[Y][t]_


_,_ (10)
log _|Ct−_ 1 _|_


Table 1: Comparative experiments on VOC dataset Everingham et al. (2010). The optimal and
suboptimal performance are respectively represented in **red** and **blue** bold. The _†_ symbol indicates
results reproduced following the publicly released code. Across five incremental settings, our method
achieves the highest average mIoU, demonstrating robust learning ability.


|Method Backbone|10-1 (11 steps)<br>Old New All|2-2 (10 steps)<br>Old New All|15-1 (6 steps)<br>Old New All|19-1 (2 steps)<br>Old New All|15-5 (2 steps)<br>Old New All|Average|
|---|---|---|---|---|---|---|
|Joint<br>ViT<br>MIB Cermelli et al. (2020)<br>Resnet101<br>SDR Michieli & Zanuttigh (2021)<br>Resnet101<br>PLOP Douillard et al. (2021a)<br>Resnet101<br>SSUL Cha et al. (2021a)<br>Resnet101<br>MicroSeg Zhang et al. (2022b)<br>Resnet101<br>REMINDER Phan et al. (2022)<br>Resnet101<br>RCIL Zhang et al. (2022a)<br>Resnet101<br>EWF Xiao et al. (2023)<br>Resnet101<br>LGKD Yang et al. (2023)<br>Resnet101<br>IDEC Zhao et al. (2023)<br>ResNet101<br>GSC Cong et al. (2023)<br>ResNet101<br>CoMasTRe Gong et al. (2024)<br>ResNet101<br>Adapter Zhu et al. (2025)<br>Resnet101<br>MIB Cermelli et al. (2020)<br>ViT<br>SSUL† Cha et al. (2021a)<br>ViT<br>MicroSeg† Zhang et al. (2022b)<br>ViT<br>CoinSeg Zhang et al. (2023)<br>ViT<br>MBS† Park et al. (2024)<br>ViT<br>Nest Xie et al. (2024)<br>ViT<br>Adapter-T Zhu et al. (2025)<br>ViT<br>Ours<br>ViT<br><br><br>|85.0<br>84.7<br>84.9<br>12.3<br>13.1<br>12.7<br>32.1<br>17.0<br>24.9<br>44.0<br>15.5<br>30.4<br>74.0<br>53.2<br>64.1<br>77.2<br>57.2<br>67.7<br>-<br>-<br>-<br>55.4<br>15.1<br>36.2<br>71.5<br>30.3<br>51.9<br>-<br>-<br>-<br>70.7<br>46.3<br>59.1<br>50.6<br>17.3<br>34.7<br>-<br>-<br>-<br>74.9<br>54.3<br>65.1<br>-<br>-<br>-<br>74.3<br>51.0<br>63.2<br>73.5<br>53.0<br>63.7<br>**80.1**<br>60.0<br>70.5<br>80.0<br>**72.9**<br>**76.6**<br>65.2<br>35.8<br>51.2<br>-<br>-<br>-<br>**81.7**<br>**72.8**<br>**77.5**<br>|77.3<br>85.5<br>84.3<br>41.1<br>23.4<br>25.9<br>13.0<br>5.1<br>6.2<br>24.1<br>11.9<br>13.6<br>-<br>-<br>-<br>60.0<br>50.9<br>52.2<br>-<br>-<br>-<br>28.3<br>19.0<br>20.3<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>62.8<br>57.9<br>58.6<br>-<br>-<br>-<br>60.3<br>40.6<br>43.4<br>64.8<br>43.4<br>46.5<br>**70.1**<br>63.3<br>64.3<br>67.5<br>**73.4**<br>**70.3**<br>-<br>-<br>-<br>-<br>-<br>-<br>**74.0**<br>**75.2**<br>**75.0**<br><br>|83.9<br>79.1<br>82.8<br>34.2<br>13.5<br>29.3<br>44.7<br>21.8<br>39.2<br>65.1<br>21.1<br>54.6<br>78.4<br>49.0<br>71.4<br>81.3<br>52.5<br>74.4<br>68.3<br>27.7<br>58.6<br>70.6<br>23.7<br>59.4<br>77.7<br>32.7<br>67.0<br>70.6<br>30.9<br>61.1<br>77.0<br>36.5<br>67.4<br>72.1<br>24.4<br>60.7<br>69.8<br>43.6<br>63.5<br>79.9<br>51.9<br>73.2<br>72.6<br>23.5<br>60.9<br>78.1<br>33.4<br>67.5<br>80.5<br>40.8<br>71.0<br>82.7<br>52.5<br>75.5<br>81.9<br>**65.6**<br>**78.0**<br>77.0<br>53.3<br>71.4<br>**83.3**<br>60.1<br>77.8<br>**83.4**<br>**66.1**<br>**79.3**<br><br>|84.4<br>79.6<br>84.2<br>71.4<br>23.6<br>69.1<br>69.1<br>32.6<br>67.4<br>75.4<br>37.4<br>73.6<br>77.8<br>49.8<br>76.5<br>79.3<br>62.9<br>78.5<br>76.5<br>32.3<br>74.4<br>68.5<br>12.1<br>65.8<br>77.9<br>6.7<br>74.5<br>77.3<br>42.9<br>75.7<br>-<br>-<br>-<br>76.9<br>42.7<br>75.3<br>75.1<br>69.5<br>74.9<br>-<br>-<br>-<br>80.4<br>47.8<br>78.8<br>80.8<br>31.5<br>78.5<br>79.0<br>25.3<br>76.4<br>81.5<br>44.8<br>79.8<br>**83.0**<br>**72.6**<br>**82.5**<br>79.7<br>60.0<br>78.8<br>-<br>-<br>-<br>**82.8**<br>**74.1**<br>**82.4**<br>|85.5<br>80.3<br>84.3<br>76.4<br>50.0<br>70.1<br>75.4<br>52.6<br>70.0<br>75.7<br>51.7<br>70.0<br>78.4<br>55.8<br>73.0<br>82.0<br>59.2<br>76.6<br>76.1<br>50.7<br>70.1<br>78.8<br>52.0<br>72.4<br>-<br>-<br>-<br>79.5<br>54.8<br>73.6<br>78.0<br>51.8<br>71.8<br>78.3<br>54.2<br>72.6<br>79.7<br>51.9<br>73.1<br>-<br>-<br>-<br>78.5<br>63.2<br>74.9<br>79.7<br>55.3<br>73.9<br>81.9<br>54.0<br>75.3<br>82.1<br>63.2<br>77.6<br>**83.9**<br>**72.6**<br>**81.2**<br>81.2<br>67.4<br>77.9<br>-<br>-<br>-<br>**84.8**<br>**76.4**<br>**82.8**<br><br>|84.1<br>41.4<br>41.5<br>48.4<br>-<br>69.9<br>67.7<br>50.8<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>65.3<br>66.6<br>73.5<br>**77.7**<br>-<br>-<br>**79.4**<br>|
|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|Image<br>MIB<br>LGKD<br>Coinseg<br>MBS<br>Ours<br>GT<br>**Old classes from the base stage**<br>**New classes from the incremental stage**<br>**MBS**<br>**Ours**<br>**Coinseg**<br>**LGKD**<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>0.0<br>0.2<br>0.4<br>0.6<br>0.8<br>1.0<br>1.0<br>1.0<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tvmonitor<br>(a)<br>(b)<br>Figure 4: (a) Visual comparison under the 15-1 setting. (b) T-SNE visualizations under the 15-1<br>setting. Our method achieves more accurate pixel-level segmentation of old classes with strong<br>resistance to forgetting, while also reducing misclassifcation of new-class pixels. The t-SNE results<br>further demonstrate that our method achieves more compact intra-class distributions and more<br>|


effectively mitigating class confusion.


classes, misclassifying “sheep” as “horse” or “cow”—classes seen during the base step. Our method
better preserves old knowledge and adapts more effectively to new classes.


**Qualitative analysis of class-specific knowledge distribution and confusion.** As shown in Fig. 4(b),
we visualize the distributions of class-specific feature using t-SNE Van der Maaten & Hinton (2008)
for our method and recent approaches under the 15-1 incremental setting. The plots reveal both
intra-class compactness and inter-class separability. Compared to prior methods, our approach yields
tighter clustering of features within the same class, reflected by denser colored point groups, and more
distinct separation between different classes. This improved feature distribution reduces inter-class
entanglement and effectively mitigates class confusion. For example, the “person” class (in pink)
forms a noticeably more compact and isolated cluster, indicating enhanced class discrimination in
incremental learning scenarios.


**Training time and convergence.** While our approach achieves consistent effectiveness across various
incremental settings without increasing inference time, it introduces a slight training overhead for
single-class learning: about 7 seconds per epoch longer than CKD on a single GPU (MKD: 51s,
CKD: 53s, DKD: 60s). To evaluate convergence, we analyze the losses of MKD, CKD, and DKD in
step 1 of the 19-1 incremental setting (Fig. 5). All methods show a rapid loss decline in the early
epochs, but MKD and CKD exhibit higher initial loss and greater fluctuations, indicating less stable


7


Table 2: Comparative experiments on ADE20K Zhou et al. (2017). Our method is capable of
effectively learning new knowledge and resisting catastrophic forgetting without accessing old-class
data for rehearsal. Notably, the average performance of our method across the four incremental
settings is very close to that of joint training, which is commonly regarded as the upper bound of


|performance in CISS.|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Method**<br>**Backbone**|**100-5 (11 steps)**<br>Old<br>New<br>All|**100-10 (6 steps)**<br>Old<br>New<br>All|**50-50 (3 steps)**<br>Old<br>New<br>All|**100-50 (2 steps)**<br>Old<br>New<br>All|**Average**|
|Joint<br>ViT<br>SDR Michieli & Zanuttigh (2021)<br>ResNet101<br>PLOP Douillard et al. (2021a)<br>ResNet101<br>RCIL Zhang et al. (2022a)<br>ResNet101<br>SSUL Cha et al. (2021a)<br>ResNet101<br>REMINDER Phan et al. (2022)<br>ResNet101<br>Microseg Zhang et al. (2022b)<br>ResNet101<br>EWF Xiao et al. (2023)<br>ResNet101<br>IDEC Zhao et al. (2023)<br>ResNet101<br>GSC Cong et al. (2023)<br>ResNet101<br>LAG Yuan et al. (2024)<br>ResNet101<br>CoMasTRe Gong et al. (2024)<br>ResNet101<br>Adapter Zhu et al. (2025)<br>ResNet101<br>MIB† Cermelli et al. (2020)<br>ViT<br>SSUL† Cha et al. (2021a)<br>ViT<br>Microseg† Zhang et al. (2022b)<br>ViT<br>Coinseg Zhang et al. (2023)<br>ViT<br>CoMFormer Cermelli et al. (2023)<br>ViT<br>INC Shang et al. (2023)<br>ViT<br>MBS† Park et al. (2024)<br>ViT<br>Ours<br>ViT|49.5<br>38.0<br>45.7<br>36.7<br>5.7<br>26.4<br>39.1<br>7.8<br>28.7<br>38.5<br>11.5<br>29.6<br>39.9<br>17.4<br>32.4<br>36.1<br>16.4<br>29.6<br>40.4<br>20.5<br>33.8<br>41.4<br>13.4<br>32.1<br>39.2<br>14.6<br>31.0<br>-<br>-<br>-<br>40.0<br>17.2<br>32.5<br>40.8<br>15.8<br>32.5<br>42.6<br>18.0<br>34.5<br>40.2<br>26.6<br>35.7<br>41.3<br>16.0<br>32.9<br>41.2<br>21.0<br>34.5<br>43.1<br>24.1<br>36.8<br>39.5<br>13.6<br>30.9<br>**46.9**<br>**31.3**<br>**41.7**<br>45.7<br>22.7<br>38.1<br>**47.2**<br>**30.0**<br>**41.5**|49.7<br>38.4<br>46.0<br>28.9<br>11.7<br>23.2<br>40.5<br>13.6<br>31.6<br>39.3<br>17.7<br>32.1<br>40.2<br>18.8<br>33.1<br>39.0<br>21.3<br>33.1<br>41.5<br>21.6<br>34.9<br>41.5<br>16.3<br>33.2<br>42.3<br>17.6<br>34.1<br>40.8<br>17.6<br>33.1<br>41.0<br>18.7<br>33.6<br>42.3<br>18.4<br>34.4<br>42.9<br>19.9<br>35.3<br>43.0<br>30.8<br>39.0<br>40.7<br>19.0<br>33.5<br>41.0<br>22.6<br>34.9<br>42.1<br>24.5<br>36.3<br>40.6<br>15.6<br>32.3<br>**48.5**<br>34.6<br>**43.9**<br>48.1<br>**35.2**<br>43.8<br>**48.7**<br>**37.3**<br>**44.9**|55.0<br>41.1<br>59.7<br>42.9<br>25.4<br>39.9<br>48.8<br>21.0<br>37.5<br>48.3<br>24.6<br>40.9<br>48.4<br>20.2<br>36.5<br>47.1<br>20.4<br>36.3<br>48.6<br>24.8<br>41.2<br>-<br>-<br>-<br>47.4<br>26.0<br>42.0<br>46.2<br>26.2<br>41.8<br>47.7<br>26.1<br>42.0<br>-<br>-<br>-<br>49.3<br>27.3<br>44.0<br>52.2<br>35.6<br>53.2<br>49.5<br>21.3<br>38.0<br>49.8<br>23.9<br>40.7<br>49.0<br>28.9<br>45.4<br>-<br>-<br>-<br>**56.2**<br>37.8<br>56.8<br>55.6<br>**39.8**<br>**58.6**<br>**56.6**<br>**40.5**<br>**59.6**|48.9<br>38.2<br>45.4<br>37.5<br>25.5<br>33.5<br>41.9<br>14.9<br>33.0<br>42.3<br>18.8<br>34.5<br>41.3<br>18.0<br>33.6<br>41.6<br>19.2<br>34.2<br>40.2<br>18.8<br>33.1<br>41.2<br>21.3<br>34.6<br>42.0<br>18.2<br>34.1<br>42.4<br>19.2<br>34.7<br>41.6<br>19.7<br>34.3<br>45.7<br>26.0<br>39.2<br>43.1<br>23.6<br>36.6<br>46.4<br>35.0<br>42.6<br>41.9<br>20.1<br>34.7<br>41.1<br>24.1<br>35.5<br>41.6<br>26.7<br>36.7<br>44.7<br>26.2<br>38.6<br>**49.4**<br>35.6<br>44.8<br>49.4<br>**37.6**<br>**45.5**<br>**49.3**<br>**39.9**<br>**46.2**|49.2<br>30.8<br>32.7<br>34.3<br>33.9<br>33.3<br>35.8<br>-<br>35.3<br>-<br>35.6<br>-<br>37.6<br>42.6<br>34.8<br>36.4<br>38.8<br>-<br>**46.8**<br>46.5<br>**48.1**|


|Grp.|LMin LEsti LMax|10-1<br>Old New All|
|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|✗<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|63.5 21.3 43.4<br>69.9 46.7 58.9<br>72.7 45.2 59.6<br>68.7 35.7 53.0<br>82.1 70.8 76.7<br>82.1 71.2 76.9<br>71.2 49.6 60.9<br>81.7 72.8 77.5|


**Impact of DKD.** To evaluate the effectiveness of the proposed DKD in reducing class confusion,
we compute the class similarity matrix for the 10 incremental classes (from class 11 to 20) in the


8


Figure 5: Epoch-wise(left) and iteration-wise loss curves (right). DKD stabilizes quickly and
maintains a smoother trajectory, indicating better training stability and reliable convergence.


optimization. In contrast, DKD consistently achieves the lowest average loss and stabilizes quickly,
demonstrating superior training stability and robustness in class-incremental semantic segmentation.


4.3 ABLATION STUDIES
**Impact of Adjusting the Distribution of Old Knowledge.** To evaluate the effectiveness of adaptive
adjustment to the old model, we conduct an ablation study under the 10-1 setting. As shown
in Fig. 6(a), the narrower orange shaded area compared to the blue one demonstrates that _L_ Min
effectively balances the performance between old and new classes. In addition, the performance of
the new classes improves significantly after incorporating _L_ Min, as indicated by _**×**_ . This suggests
that dynamic adjustment of the old model’s parameters facilitates the learning of new knowledge.


**Effectiveness of components in DKD.** To assess the contributions Table 3: Ablation study on the

by _L_ Min allows better adaptation to new classes. To ensure the best
overall performance across old and new classes, Grp. 8 is adopted as the configuration in this paper.


Table 3: Ablation study on the


Table 4: Ablation study on the hyperparameter _γ_ . In settings with fewer incremental classes, _γ_
indicates hyperparameter insensitivity; conversely, _γ_ = 0 _._ 4 performs best.


|Col1|10-1 (11 steps)<br>γ= 0 γ= 0.2 γ= 0.4 γ= 0.5 γ= 0.6 γ= 0.8 γ= 1.0|19-1 (2 steps)<br>γ= 0 γ= 0.2 γ= 0.4 γ= 0.5 γ= 0.6 γ= 0.8 γ= 1.0|
|---|---|---|
|Old<br>New<br>All|80.4<br>76.3<br>81.7<br>**82.6**<br>80.0<br>80.5<br>80.2<br>69.0<br>66.6<br>**72.8**<br>70.8<br>69.9<br>70.0<br>70.3<br>75.0<br>71.7<br>**77.5**<br>77.0<br>75.2<br>75.5<br>75.5|**82.8**<br>**82.8**<br>82.7<br>**82.8**<br>**82.8**<br>82.7<br>**82.8**<br>74.2<br>**74.3**<br>74.0<br>**74.3**<br>74.0<br>74.0<br>74.1<br>**82.4**<br>82.4<br>82.3<br>**82.4**<br>**82.4**<br>82.3<br>**82.4**|


In this paper, we propose **D** istribution-based **K** nowledge **D** istillation (DKD), a minimization–
maximization strategy designed to address parameter competition and knowledge reuse. DKD
first releases low-sensitivity parameters of the old model and applies _L_ Min to minimize the oldknowledge distribution, thereby alleviating parameter competition under a static architecture. To
better estimate the distribution of new knowledge and promote the reuse of acquired knowledge,
we introduce _L_ Esti guided by Laplacian-based projection estimation. To further mitigate the underutilization of previously acquired knowledge, we then employ _L_ Max to maximize the shared
knowledge distribution through an entropy-induced optimization. We establish the rationality of
the method via theoretical analysis and demonstrate its effectiveness through extensive experiments.
This minimization–maximization strategy reduces class confusion and achieves **near-upper-bound**
average performance on ADE20K and Pascal VOC 2012, without incurring additional inference cost.


9


11

12

13

14

15

16

17

18

19


Mean performance of **baseline** **baseline with** ����
across old and new classes

(a)


11


12

13

14

15

16

17

18

19

20


20
(b) (c)


Figure 6: (a) compares the baseline with and without _LMin_, showing that it enables adaptive
adjustment of old parameters and knowledge distribution, facilitating new class learning while
mitigating forgetting. (b) and (c) show that DKD lowers similarity between incrementally learned
classes, reducing class confusion.


10-1 setting. This is shown in Fig. 6(b) (without DKD) and Fig. 6(c) (with DKD). In the figures, red
indicates higher similarity between classes, while lighter red and blue represent lower similarity. As
shown in Fig. 6(b), without DKD, the similarity between the classes learned during the incremental
steps (classes 11–20) is relatively high, resulting in an overall similarity matrix with a reddish hue.
However, after applying DKD, as shown in Fig. 6(c), the similarity between the learned classes
significantly decreases. This suggests that the proposed DKD method effectively reduces class
confusion, which in turn enhances pixel-level CISS performance.


**Ablation of hyperparameters.** Based on the experimental results from 14 sets of hyperparameters,
as shown in Tab. 4, it is observed that for tasks with fewer incremental classes, such as 19-1, _γ_ has a
minimal impact on the performance of both old and new classes, as well as their average performance.
Thus, _γ_ in Eq. 6 is set to the default value of 1. As the number of incremental classes increases,
new knowledge tends to be underfit without strengthened knowledge reuse; setting _γ_ too large or
too small disrupts the performance balance between old and new knowledge. Thus, for tasks with
more incremental classes, such as the 10-1, both the individual performance of old and new classes,
as well as their average performance, reach optimal results when _γ_ is around 0.4. Accordingly, the
default value of _γ_ is used for the 15-5, 15-1, and 19-1 tasks on Pascal VOC. For settings involving
more incremental steps (10-1 and 2-2 on Pascal VOC) or a larger number of classes (100-50, 100-10,
50-50, and 100-5 on ADE20K), _γ_ is set to 0.4.


**Error** **Analysis.** As shown in Tab. 5, we evaluate experimental
error by repeating the 10-1 configuration three times. The overall
standard deviation for combined old and new class performance is
approximately 0.1, confirming the stability and robustness of our
metrics. Additional analysis is provided in Appendix C.7.


5 CONCLUSION


Table 5: Repeated experiments
for error analysis.

1 2 3 Avg. Std.
Old 81.8 82.0 81.7 81.83 0.15
New 72.6 72.4 72.8 72.60 0.20
All 77.4 77.4 77.5 77.43 **0.06**


ETHICS STATEMENT


We adhere to the ICLR Code of Ethics in data use, experimentation, and manuscript preparation. The
primary ethical considerations and compliance measures in this work are as follows.


**Human** **subjects** **and** **identifiable** **information.** This research does not involve human-subject
experiments and does not collect or process any personally identifiable information (PII) or sensitive
data. All experiments are conducted on publicly available datasets.


**Data and licensing.** We use the publicly available PASCAL VOC 2012 and ADE20K datasets under
their original licenses and terms of use. Beyond standard preprocessing (e.g., label usage), we do not
modify the data, synthesize new personal content, or inject information that could reveal identities.


**Potential harms.** Our continual-learning method can improve the stability and plasticity of semantic
segmentation in long-term incremental scenarios, and could, in principle, be adapted to sensitive
applications (e.g., surveillance). To reduce misuse risk: (1) we do not provide any data, models,
or scripts tailored for face recognition, tracking, or other privacy-intrusive tasks; (2) we encourage
practitioners to conduct application-level risk assessments (privacy, compliance, and security reviews)
prior to deployment, and to use our method only in lawful, legitimate, and ethical contexts.


**Fairness,** **bias,** **and** **interpretability.** Our experiments follow standard CISS architectures and
protocols from prior works, changing only the incremental-learning strategy. We provide theoretical
analysis in the appendix to improve interpretability.


**Privacy and security.** We do not introduce raw images or metadata that could identify individuals.
Training and evaluation do not involve inversion, re-identification, model stealing, or other high-risk
procedures. Released code will not include scripts or interfaces for downloading personal data.


**Legal** **and** **regulatory** **compliance.** The research complies with dataset licenses and applicable
copyright/usage terms. For any follow-up or regional extensions, users are responsible for ensuring
compliance with local laws and ethical review requirements.


**Conflicts** **of** **interest** **and** **funding** **disclosure.** We have no undisclosed commercial conflicts of
interest.


REPRODUCIBILITY STATEMENT


We are committed to ensuring the reproducibility of our work. In the following, we summarize the
measures we have taken:


**Code Availability.** To facilitate the review process, the related code is included in the supplementary
material. The released package includes model definitions, training scripts, and evaluation tools.
Detailed instructions are also provided, covering the environment setup (Python version, PyTorch
dependencies, and GPU drivers), execution commands, and hyperparameter settings, ensuring that
the experiments can be easily replicated.


**Details of Experimental Setups.** The descriptions of the dataset, implementation details, metrics,
baselines, and implementation configuration are provided in Appendix B. In particular, we specify
the dataset splits, preprocessing steps, evaluation protocols, and the baseline implementations to
guarantee fair comparisons. All hyperparameters used in training and testing are documented in both
the main paper and the appendix.


**Computational Resources.** All experiments are conducted on six NVIDIA GeForce RTX 3090 GPUs
and an Intel(R) Xeon(R) Gold 6226R CPU, with a batch size of 16, using PyTorch for implementation,
as mentioned in Section 4.1 and Appendix B. We also record average training times and convergence
to provide a clear view of the computational requirements in Section 4.2.


**Statistical Significance of Results.** To verify the robustness of our conclusions, we conduct multiple
repeated experiments on complex tasks with a larger number of incremental steps. We observe that
the average performance deviation across all classes remains around 0.1, as detailed in Section 4.3
and Appendix C.7. In addition, we provide variance and confidence interval analyses for the main
results, which further demonstrate the stability and reliability of our method.


We believe these efforts ensure that our results can be reliably reproduced and extended by the
research community.


10


REFERENCES


Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars. Expert gate: Lifelong learning with a
network of experts. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 3366–3375, 2017.


Robert B Ash. Information theory. _Courier Corporation_, 2012.


Donghyeon Baek, Youngmin Oh, Sanghoon Lee, Junghyup Lee, and Bumsub Ham. Decomposed
knowledge distillation for class-incremental semantic segmentation. _Neural Information Processing_
_Systems (NIPS)_, 35:10380–10392, 2022.


Fabio Cermelli, Massimiliano Mancini, Samuel Rota Bulo, Elisa Ricci, and Barbara Caputo. Modeling
the background for incremental learning in semantic segmentation. In _Computer Vision and Pattern_
_Recognition (CVPR)_, pp. 9233–9242, 2020.


Fabio Cermelli, Matthieu Cord, and Arthur Douillard. Comformer: Continual learning in semantic
and panoptic segmentation. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 3010–3020,
2023.


Sungmin Cha, Beomyoung Kim, Young Joon Yoo, and Taesup Moon. Ssul: Semantic segmentation
with unknown label for exemplar-based class-incremental learning. _Neural Information Processing_
_Systems (NIPS)_, 34:10919–10930, 2021a.


Sungmin Cha, YoungJoon Yoo, Taesup Moon, et al. Ssul: Semantic segmentation with unknown label
for exemplar-based class-incremental learning. _Neural Information Processing Systems (NIPS)_,
34:10919–10930, 2021b.


Jinpeng Chen, Runmin Cong, Yuxuan Luo, Horace Ho Shing Ip, and Sam Kwong. Replay without saving: Prototype derivation and distribution rebalance for class-incremental semantic segmentation.
_IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)_, 2025.


Wei Cong, Yang Cong, Jiahua Dong, Gan Sun, and Henghui Ding. Gradient-semantic compensation
for incremental semantic segmentation. _IEEE Transactions on Multimedia (TMM)_, 26:5561–5574,
2023.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _Computer vision and pattern recognition (CVPR)_, pp. 248–255.
Ieee, 2009.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image
is worth 16x16 words: Transformers for image recognition at scale. _International Conference on_
_Learning Representations (ICLR)_, 2021.


Arthur Douillard, Yifu Chen, Arnaud Dapogny, and Matthieu Cord. Plop: Learning without forgetting
for continual semantic segmentation. In _Computer Vision and Pattern Recognition (CVPR)_, pp.
4040–4050, 2021a.


Arthur Douillard, Yifu Chen, Arnaud Dapogny, and Matthieu Cord. Plop: Learning without forgetting
for continual semantic segmentation. In _Computer Vision and Pattern Recognition (CVPR)_, pp.
4040–4050, 2021b.


Mark Everingham, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The
pascal visual object classes (voc) challenge. _International Journal of Computer Vision (IJCV)_, 88:
303–338, 2010.


Yizheng Gong, Siyue Yu, Xiaoyang Wang, and Jimin Xiao. Continual segmentation with disentangled
objectness learning and class recognition. In _Computer Vision and Pattern Recognition (CVPR)_,
pp. 3848–3857, 2024.


Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, and Jin Young Choi. A comprehensive overhaul of feature distillation. In _International Conference on Computer Vision (ICCV)_,
pp. 1921–1930, 2019.


11


Jitesh Jain, Anukriti Singh, Nikita Orlov, Zilong Huang, Jiachen Li, Steven Walton, and Humphrey
Shi. Semask: Semantically masked transformers for semantic segmentation. In _International_
_Conference on Computer Vision (ICCV)_, pp. 752–761, 2023.


Nikhil Ketkar. Stochastic gradient descent. In _Deep learning with Python:_ _A hands-on introduction_,
pp. 113–132. Springer, 2017.


Zhizhong Li and Derek Hoiem. Learning without forgetting. _IEEE Transactions on Pattern Analysis_
_and Machine Intelligence (TPAMI)_, 40(12):2935–2947, 2017.


Chenzhong Liao and Lingguo Kong. A comprehensive survey of semantic segmentation based
on computer vision. In _Fourth International Conference on Computer Vision, Application, and_
_Algorithm (CVAA 2024)_, volume 13486, pp. 235–250. SPIE, 2025.


Andrea Maracani, Umberto Michieli, Marco Toldo, and Pietro Zanuttigh. Recall: Replay-based continual learning in semantic segmentation. In _International conference on computer vision (ICCV)_,
pp. 7026–7035, 2021.


Umberto Michieli and Pietro Zanuttigh. Incremental learning techniques for semantic segmentation.
In _International Conference on Computer Vision Workshops (ICCVW)_, 2019.


Umberto Michieli and Pietro Zanuttigh. Continual semantic segmentation via repulsion-attraction
of sparse and disentangled latent representations. In _Computer_ _Vision_ _and_ _Pattern_ _Recogni-_
_tion (CVPR)_, pp. 1114–1124, 2021.


Youngmin Oh, Donghyeon Baek, and Bumsub Ham. Alife: Adaptive logit regularizer and feature
replay for incremental semantic segmentation. _Neural Information Processing Systems (NIPS)_, 35:
14516–14528, 2022.


Gilhan Park, WonJun Moon, SuBeen Lee, Tae-Young Kim, and Jae-Pil Heo. Mitigating background shift in class-incremental semantic segmentation. In _European Conference on Computer_
_Vision (ECCV)_, pp. 71–88. Springer, 2024.


Minh-Hieu Phan, The-Anh Ta, Son Lam Phung, Long Tran-Thanh, and Abdesselam Bouzerdoum.
Class similarity weighted knowledge distillation for continual semantic segmentation. In _Computer_
_Vision and Pattern Recognition (CVPR)_, pp. 16866–16875, 2022.


Qi Qin, Wenpeng Hu, Han Peng, Dongyan Zhao, and Bing Liu. Bns: Building network structures
dynamically for continual learning. _Neural Information Processing Systems (NIPS)_, 34:20608–
20620, 2021.


Yiqiao Qiu, Yixing Shen, Zhuohao Sun, Yanchong Zheng, Xiaobin Chang, Weishi Zheng, and
Ruixuan Wang. Sats: Self-attention transfer for continual semantic segmentation. _Pattern Recogni-_
_tion (PR)_, 138:109383, 2023.


Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, and Christoph H Lampert. icarl: Incremental classifier and representation learning. In _Computer Vision and Pattern Recognition (CVPR)_,
pp. 2001–2010, 2017.


Chao Shang, Hongliang Li, Fanman Meng, Qingbo Wu, Heqian Qiu, and Lanxiao Wang. Incrementer:
Transformer for class-incremental semantic segmentation with knowledge distillation focusing on
old class. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 7214–7224, 2023.


Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. _Journal of machine_
_learning research (JMLR)_, 9(11), 2008.


Lin Wang and Kuk-Jin Yoon. Knowledge distillation and student-teacher learning for visual intelligence: A review and new outlooks. _IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_
_Intelligence (TPAMI)_, 44(6):3048–3068, 2021.


Qilong Wang, Yiwen Wu, Liu Yang, Wangmeng Zuo, and Qinghua Hu. Layer-specific knowledge
distillation for class incremental semantic segmentation. _IEEE Transactions on Image Process-_
_ing (TIP)_, 2024.


12


Jianqiang Xiao, Chang-Bin Zhang, Jiekang Feng, Xialei Liu, Joost van de Weijer, and Mingg-Ming
Cheng. Endpoints weight fusion for class incremental semantic segmentation. In _Computer Vision_
_and Pattern Recognition (CVPR)_, pp. 7204–7213, 2023.


Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. Segformer:
Simple and efficient design for semantic segmentation with transformers. _Neural Information_
_Processing Systems (NIPS)_, 34:12077–12090, 2021.


Zhengyuan Xie, Haiquan Lu, Jia-wen Xiao, Enguang Wang, Le Zhang, and Xialei Liu. Early
preparation pays off: New classifier pre-tuning for class incremental semantic segmentation. In
_European Conference on Computer Vision (ECCV)_, pp. 183–201. Springer, 2024.


Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynamically expandable representation for class
incremental learning. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 3014–3023, 2021.


Chenglin Yang, Lingxi Xie, Chi Su, and Alan L Yuille. Snapshot distillation: Teacher-student
optimization in one generation. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 2859–
2868, 2019.


Guanglei Yang, Enrico Fini, Dan Xu, Paolo Rota, Mingli Ding, Moin Nabi, Xavier Alameda-Pineda,
and Elisa Ricci. Uncertainty-aware contrastive distillation for incremental semantic segmentation.
_IEEE_ _Transactions_ _on_ _Pattern_ _Analysis_ _and_ _Machine_ _Intelligence_ _(TPAMI)_, 45(2):2567–2581,
2022.


Ze Yang, Ruibo Li, Evan Ling, Chi Zhang, Yiming Wang, Dezhao Huang, Keng Teck Ma, Minhoe
Hur, and Guosheng Lin. Label-guided knowledge distillation for continual semantic segmentation
on 2d images and 3d point clouds. In _International Conference on Computer Vision (ICCV)_, pp.
18601–18612, 2023.


Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang. Lifelong learning with dynamically
expandable networks. _arXiv preprint arXiv:1708.01547_, 2017.


Zhidong Yu, Wei Yang, Xike Xie, and Zhenbo Shi. Tikp: Text-to-image knowledge preservation for
continual semantic segmentation. In _Association for the Advancement of Artificial Intelligence_
_Conference (AAAI)_, volume 38, pp. 16596–16604, 2024.


Bo Yuan and Danpei Zhao. A survey on continual semantic segmentation: Theory, challenge, method
and application. _IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)_, 2024.


Bo Yuan, Danpei Zhao, and Zhenwei Shi. Learning at a glance: Towards interpretable data-limited
continual semantic segmentation via semantic-invariance modelling. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence (TPAMI)_, 46(12):7909–7923, 2024.


Chang-Bin Zhang, Jianqiang Xiao, Xialei Liu, Ying-Cong Chen, and Ming-Ming Cheng. Representation compensation networks for continual semantic segmentation. In _Computer Vision and Pattern_
_Recognition (CVPR)_, pp. 7053–7064, 2022a.


Zekang Zhang, Guangyu Gao, Zhiyuan Fang, Jianbo Jiao, and Yunchao Wei. Mining unseen classes
via regional objectness: A simple baseline for incremental segmentation. _Neural_ _Information_
_Processing Systems (NIPS)_, 35:24340–24353, 2022b.


Zekang Zhang, Guangyu Gao, Jianbo Jiao, Chi Harold Liu, and Yunchao Wei. Coinseg: Contrast
inter- and intra- class representations for incremental segmentation. In _International Conference_
_on Computer Vision (ICCV)_, pp. 843–853, 2023.


Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks
with noisy labels. _Neural Information Processing Systems (NIPS)_, 31, 2018.


Danpei Zhao, Bo Yuan, and Zhenwei Shi. Inherit with distillation and evolve with contrast: Exploring
class incremental semantic segmentation without exemplar memory. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence (TPAMI)_, 45(10):11932–11947, 2023.


13


Hanbin Zhao, Fengyu Yang, Xinghe Fu, and Xi Li. Rbc: Rectifying the biased context in continual
semantic segmentation. In _European Conference on Computer Vision (ECCV)_, pp. 55–72. Springer,
2022.


Ervine Zheng, Qi Yu, Rui Li, Pengcheng Shi, and Anne Haake. A continual learning framework for
uncertainty-aware interactive image segmentation. In _Association for the Advancement of Artificial_
_Intelligence Conference (AAAI)_, volume 35, pp. 6030–6038, 2021.


Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene
parsing through ade20k dataset. In _Computer Vision and Pattern Recognition (CVPR)_, pp. 633–641,
2017.


Guilin Zhu, Dongyue Wu, Changxin Gao, Runmin Wang, Weidong Yang, and Nong Sang. Adaptive
prototype replay for class incremental semantic segmentation. In _AAAI Conference on Artificial_
_Intelligence (AAAI)_, volume 39, pp. 10932–10940, 2025.


14


APPENDIX


A THEORETICAL ANALYSIS


A.1 THEORETICAL ANALYSIS ABOUT _LMin_


where the first term encourages the current model to increase the probability _yk_ _[t]_ [for this old class,]
making it closer to the old model’s output _yk_ _[∗]_ [,] [and] [the] [second] [term] [balances] [contributions] [from]
other old classes to maintain probability normalization. For old classes _k_ _∈_ _C_ _[t][−]_ [1], the gradient
direction encourages the current model’s output at the retained parameter locations to not fall below
the reference provided by the pruned old model, thereby achieving “soft target alignment”.


**(ii) When** _k_ _∈/_ _C_ _[t][−]_ [1] **(non-old classes):**
_∂ℓ_ = _yk_ _[t]_                - _yc_ _[∗][y]_ _c_ _[t][,]_ (18)
_∂zk_ _[t]_ _c∈C_ _[t][−]_ [1]

this positive gradient indirectly shrinks the knowledge distribution for non-old classes, as the optimizer will adjust the logits during gradient descent to reduce this term, thereby alleviating crowded
parameter-fitting knowledge space for old classes.


A.2 THEORETICAL ANALYSIS ABOUT _LEsti_


At pixel ( _h, w_ ), let _f_ := _ft_ ( _h, w_ ) _∈_ R _[D]_ be the current feature and _p_ := _Pt_ ( _h, w_ ) _∈_ R _[D]_ the “lowcurvature coexistence point” inferred from the second-order information of _LCKD_ . Let _yc_ _[∗]_ [(] _[h, w]_ [)] _[≥]_ [0]
be the old-knowledge weight at this pixel. Define the pixel-wise distillation term _ℓCKD_ ( _f_ ) :=
_yc_ _[∗]_ [(] _[h, w]_ [)] _[ ∥][f]_ _[−]_ _[f][t][−]_ [1][(] _[h, w]_ [)] _[∥]_ [2][,] [the] [image-level] [distillation] _[L][CKD]_ [:=] _HW_ 1 - _h,w_ _[ℓ][CKD]_ - _ft_ ( _h, w_ )�,
and the Laplacian consistency _Llap_ := _HW_ 1 - _h,w_ _[∥][f][t]_ [(] _[h, w]_ [)] _[ −P][t]_ [(] _[h, w]_ [)] _[∥]_ [2][.] [Here] _[ ∥· ∥]_ [2][ denotes the]
Euclidean norm. For any pixel,

             -             _ℓCKD_ ( _f_ ) _−_ _ℓCKD_ ( _p_ ) = _yc_ _[∗]_ [(] _[h, w]_ [)] _∥f_ _−_ _ft−_ 1 _∥_ 2 _−∥p −_ _ft−_ 1 _∥_ 2 _≤_ _yc_ _[∗]_ [(] _[h, w]_ [)] _[ ∥][f]_ _[−]_ _[p][∥]_ [2] _[,]_ (19)
where the inequality follows from the _reverse triangle inequality_ _∥u∥_ 2 _−∥v∥_ 2 _≤∥u −_ _v∥_ 2 with
_u_ = _f_ _−_ _ft−_ 1 and _v_ = _p −_ _ft−_ 1. Averaging over all pixels and letting _y_ max := max _h,w yc_ _[∗]_ [(] _[h, w]_ [)]


15


In this paper, to address the crowded distribution of old and new knowledge caused by parameter
competition, we propose a strategy to minimize the discrepancy between the output of the parameterreleased old model and the newly trained model over the old knowledge distribution in Section 3.2
(a). Here, the release is performed once per step for the old model, rather than at every epoch, and
therefore does not introduce additional training overhead. Given the current training step _t_, the
softmax output probability _yc_ _[t]_ [(] _[h, w]_ [)][ represents the predicted probability of class] _[ c]_ [ at position][ (] _[h, w]_ [)][:]


exp( _zc_ _[t]_ [(] _[h, w]_ [))]
_yc_ _[t]_ [(] _[h, w]_ [) =] - _[t]_


(12)
_j_ [exp(] _[z]_ _j_ _[t]_ [(] _[h, w]_ [))] _[,]_


the standard derivative of the softmax with respect to the unnormalized prediction score (logit)
_zk_ _[t]_ [(] _[h, w]_ [)][ for class] _[ k]_ [ at position][ (] _[h, w]_ [)][ is]
_∂yc_ _[t]_ = _yc_ _[t]_ [(] _[δ][ck]_ _[−]_ _[y]_ _k_ _[t]_ [)] _[,]_ (13)
_∂zk_ _[t]_


where _δck_ is the Kronecker delta:


�1 _,_ _c_ = _k,_
_δck_ = (14)
0 _,_ _c ̸_ = _k._


Thus, the loss gradient at the pixel ( _h, w_ ) of _LMin_ can be written as:
_∂ℓ_ ( _h, w_ ) _∂y_ _[t]_


_c∈_ - _C_ _[t][−]_ [1] _yc_ _[∗]_ _∂z∂yk_ _[t]_ _c_ _[t]_ (15)

- _yc_ _[∗]_ _[y]_ _c_ _[t]_ [(] _[δ][ck]_ _[−]_ _[y]_ _k_ _[t]_ [)] _[,]_ (16)

_c∈C_ _[t][−]_ [1]


= _−_   _∂zk_ _[t]_ _[t]_


= _−_ 


**(i) When** _k_ _∈_ _C_ _[t][−]_ [1] **(old classes):**
_∂ℓ_ = _−yk_ _[∗][y]_ _k_ _[t]_ [(1] _[ −]_ _[y]_ _k_ _[t]_ [) +] _[ y]_ _k_ _[t]_            _∂zk_ _[t]_ _[t][−]_ [1]


 - _yc_ _[∗][y]_ _c_ _[t][,]_ (17)

_c∈C_ _[t][−]_ [1] _,c_ = _k_


yields


_H_ ( _Yt_ ) = _−_


1
_LCKD_ ( _ft_ ) _−LCKD_ ( _Pt_ ) _≤_
_HW_


- _yc_ _[∗]_ [(] _[h, w]_ [)] _[ ∥][f][t]_ _[−P][t][∥]_ [2] _[≤]_ _[y]_ [max] _[L][lap][.]_ (20)


_h,w_


Decreasing _Llap_ guarantees at least a _linear_ reduction of the distillation gap, with proportionality
governed by _yc_ _[∗]_ [—this is the direct, mathematical form of “reusing old knowledge.”]


Fix a pixel ( _h, w_ ) and write
_f_ _≡_ _ft_ ( _h, w_ ) _∈_ R _[D]_ _,_ (21)


_y_ _≡_ _yc_ _[∗]_ [(] _[h, w]_ [)] _[ ∈]_ [R] _[D][,]_ (22)


_f_
_f_ ˆ := _._ (23)
_∥f_ _∥_ 2


_Ct_ ( _h, w_ ) preserves the degree of reusable old knowledge


_[f]_ _[⟩]_
_Ct_ ( _h, w_ ) = _ϕ_ ( _f_ ) := _[⟨][y,]_


_,_ (24)
_∥f_ _∥_ 2


and denote a scalar weight by _w_ ˆ. The pixel-wise part of _L_ pro is then
_J_ ( _f_ ) = _−_ _w_ ˆ _Ct_ ( _h, w_ ) = _−_ _w ϕ_ ˆ ( _f_ ) _,_ (25)
which matches the global definition;


1
_L_ pro = 1 _−_
_HW_


- _w_ ˆ( _h, w_ ) _Ct_ ( _h, w_ ) _,_ (26)


_h,w_


_c_ [(] _[h, w]_ [)] _[,]_ _[f][t]_ [(] _[h, w]_ [)] _[⟩]_
_Ct_ ( _h, w_ ) = _[⟨][y][∗]_ _∥ft_ ( _h, w_ ) _∥_ 2 _._ (27)


**(i) Gradient of** _Ct_ ( _h, w_ ) **.** Write _ϕ_ ( _f_ ) = _N/D_ with _N_ := _y_ _[⊤]_ _f_ and _D_ := _∥f_ _∥_ 2 = ( _f_ _[⊤]_ _f_ ) [1] _[/]_ [2] . Then
_∇f_ _N_ = _y_ and _∇f_ _D_ = _f/∥f_ _∥_ 2 = _f_ [ˆ] . By the quotient rule,


_[−]_ _[N][ ∇][f]_ _[D]_ _[f]_ [ˆ]

= _[∥][f]_ _[∥]_ [2] _[ y][ −]_ [(] _[y][⊤][f]_ [)]
_D_ [2] _∥f_ _∥_ [2]


_[−]_ _[N][ ∇][f]_ _[D]_
_∇f_ _ϕ_ ( _f_ ) = _[D][ ∇][f]_ _[N]_


_∥f_ _∥_ [2] 2


1
=
_∥f_ _∥_ 2


- - 1
_y −_ ( _f_ [ˆ] _[⊤]_ _y_ ) _f_ [ˆ] =
_∥f_ _∥_ 2


 - _I_ _−_ _f_ [ˆ] _f_ [ˆ] _[⊤]_ [�] _y._ (28)


**(ii) Descent direction for the pixel-wise objective.** Since _J_ ( _f_ ) = _−w ϕ_ ˆ ( _f_ ) with _w_ ˆ treated as a
local constant,


_w_ ˆ
_∇f_ _J_ ( _f_ ) = _−_ _w_ ˆ _∇f_ _ϕ_ ( _f_ ) = _−_ _∥f_ _∥_ 2


- _I_ _−_ _f_ [ˆ] _f_ [ˆ] _[⊤]_ [�] _y._ (29)


A gradient descent step _f_ [+] = _f_ _−_ _η ∇f_ _J_ ( _f_ ) ( _η_ _>_ 0) yields the increment


_w_
∆ _f_ := _f_ [+] _−_ _f_ = _η_
_∥f_ _∥_ 2


- _I_ _−_ _f_ [ˆ] _f_ [ˆ] _[⊤]_ [�] _y_ = _⇒_ ∆ _f_ _∝_ - _I_ _−_ _f_ [ˆ] _f_ [ˆ] _[⊤]_ [�] _y_ _._ (30)


This indicates that our update direction is the steepest ascent direction of the projection-based
alignment, thereby aligning the current representation—without increasing its norm—under the
guidance of the knowledge. The larger _w_ ˆ( _h, w_ ) emphasizes high-confidence regions where old
knowledge can be reused, guiding the update equation 30 to align the features with old knowledge.
In contrast, low-confidence regions are scarcely affected, preserving capacity for new knowledge.


A.3 THEORETICAL ANALYSIS ABOUT _LMax_


Suppose _y_ ( _[b]_ _t,c_ ) _[∈]_ [[0] _[,]_ [ 1]][ denotes the predicted probability (softmax output) of sample] _[ b][ ∈{]_ [1] _[, . . ., B][}]_


at step _t_ for class _c ∈{_ 1 _, . . ., Ct}_, with [�] _c_ _[C]_ =1 _[t]_ _[y]_ ( _[b]_ _t,c_ ) [= 1][.] [Define the batch marginal (per class)]


_y_ ( _t,c_ ) = _B_ [1]


_B_


- _y_ ( _[b]_ _t,c_ ) _[,]_ (31)

_b_ =1


the marginal entropy


_Ct_


_c_ =1


_y_ ( _t,c_ ) log _y_ ( _t,c_ ) _,_ (32)


16


and the conditional entropy proxy


_y_ ( _[b,]_ _t,c_ [new] ) _≈_ _y_ ( _[b]_ _t,c_ ) [+] _[ η]_ _B_ log _|_ 1 _Ct−_ 1 _|_ [log] _yy_ (( _[b]_ _t,ct,c_ )) _._ (38)


If _y_ ( _[b]_ _t,c_ ) _[>]_ _[y]_ [(] _[t,c]_ [)][, then][ log] _[y]_ _y_ _[>]_ [0][ so] _[ ∂][L]_ [Max] _[/∂y]_ _[<]_ [0][; a gradient descent step] _[ y]_ _[←]_ _[y][ −]_ _[η ∂][L]_ [Max] _[/∂y]_

( _η_ _>_ 0) _increases y_, making the per-sample distribution _sharper_ (lower _H_ ( _Yt_ _| Yt−_ 1)). If _y_ ( _[b]_ _t,c_ ) _[<]_
_y_ ( _t,c_ ), the sign reverses and _y_ decreases; aggregated over the batch, this pushes _y_ toward _balance_
(higher _H_ ( _Yt_ )), mitigating collapse. Since _L_ Max = _−I_ ( _Yt_ ; _Yt−_ 1) _/_ log _|Ct−_ 1 _|_, minimizing _L_ Max is
equivalent to maximizing _I_ ( _Yt_ ; _Yt−_ 1): it enforces _low conditional entropy_ (sample-wise certainty
given old knowledge) and _high_ _marginal_ _entropy_ (batch-wise class balance). The former injects
old discriminative knowledge into current predictions (retention/transfer), while the latter preserves
capacity and diversity for new classes. Together, the update sharpens per-sample distributions
and balances class usage across the batch—maximizing shared information between old and new
distributions and thereby enabling “old-to-new” knowledge reuse.


B DETAILS OF EXPERIMENTAL SETUPS


**Dataset.** Following previous works Cermelli et al. (2020); Yang et al. (2023); Zhang et al. (2023);
Park et al. (2024), our method is evaluated on the Pascal VOC 2012 Everingham et al. (2010) and
ADE20K Zhou et al. (2017) datasets. The Pascal VOC dataset Everingham et al. (2010), a widely
used benchmark for incremental segmentation, comprises 10,582 training images and 1,449 validation
images across 20 categories. Additionally, the more challenging ADE20K dataset Zhou et al. (2017)
includes 150 categories, with 20,210 training images and 2,000 validation images.


**Implementation details.** Following established practices Park et al. (2024); Zhang et al. (2023),
we use ViT pretrained on ImageNet-1K as the backbone network. The decoder is composed of two
transformer blocks, processing image patches at a fixed resolution of 512×512. Model optimization
is conducted using stochastic gradient descent (SGD) with a momentum of 0.9 and a weight decay
of 1 _×_ 10 _[−]_ [5] . Each training step in both the base and incremental steps is conducted for 64 epochs.
The learning rate starts at 1 _×_ 10 _[−]_ [3] in the base step ( _t_ = 1) and is adaptively adjusted in subsequent
incremental steps ( _t ≥_ 1). For Pascal VOC, the learning rate is reduced to 1 _×_ 10 _[−]_ [4] in the incremental
steps, while for ADE20K, it is set to 5 _×_ 10 _[−]_ [4] . We set _τ_ = 0 _._ 1 for all experiments, and _γ_ is initialized
to 1. For tasks with more incremental classes, including the 10-1 and 2-2 configuration of VOC
and all incremental configurations of ADE20K, _γ_ is set to 0.4. All experiments are conducted on 6
Nvidia GeForce RTX 3090 GPUs and an Intel(R) Xeon(R) Gold 6226R CPU, with PyTorch used for
implementation.


17


_Ct_

- _y_ ( _[b]_ _t,c_ ) [log] _[ y]_ ( _[b]_ _t,c_ ) _[.]_ (33)

_c_ =1


_H_ ( _Yt_ _| Yt−_ 1) = _−_ [1]

_B_


_B_


_b_ =1


The objective is


_[|][ Y][t][−]_ [1][)]
_L_ Max = _−_ _[H]_ [(] _[Y][t]_ [)] _[ −]_ _[H]_ [(] _[Y][t]_


(34)
log _|Ct−_ 1 _|_ _[.]_


_[ −]_ _[H]_ [(] _[Y][t]_ _[|][ Y][t][−]_ [1][)]

= _−_ _[I]_ [(] _[Y][t]_ [;] _[ Y][t][−]_ [1][)]
log _|Ct−_ 1 _|_ log _|Ct−_ 1 _|_


**(i) Gradient of the marginal entropy (chain rule).**
_∂H_ ( _Yt_ )          
[1]


_∂H_ ( _Yt_ )

= [1]
_∂y_ ( _[b]_ _t,c_ ) _B_


_B_


- 
_−_ 1 _−_ log _y_ ( _t,c_ ) _._ (35)


**(ii) Gradient of the conditional entropy (per-sample derivative).**
_∂H_ ( _Yt_ _| Yt−_ 1)         
[1] _[b]_


( _Yt_ _| Yt−_ 1)

= _−_ [1]
_∂y_ ( _[b]_ _t,c_ ) _B_


_B_


- 1 + log _y_ ( _[b]_ _t,c_ ) _._ (36)


**(iii) Gradient of** _L_ **Max** **(difference and normalization).** Combining equation 35–equation 36,


_∂L_ Max 1 _y_ ( _[b]_ _t,c_ )
= _−_
_∂y_ ( _[b]_ _t,c_ ) _B_ log _|Ct−_ 1 _|_ [log] _y_ ( _t,c_ )


(37)


**Metrics.** In this paper, the performance of incremental semantic segmentation is evaluated using the
commonly adopted mean Intersection over Union (mIoU) metric, consistent with previous methods.
In the supplemental material, we additionally report Accuracy (ACC) to compare performance
differences among subcategories. In our terminology, “Old” refers to the mIoU of old classes from
the base step, “New” denotes the mIoU of new classes introduced in incremental steps, and “All”
represents the mIoU across all classes, including background.


**Baselines.** In CISS, most current methods Cermelli et al. (2020); Yang et al. (2023); Zhang et al.
(2023); Park et al. (2024) focus on optimizing incremental learning strategies rather than the architectural design. The primary architectural difference among existing methods lies in the choice
of backbone—typically either ResNet101 or ViT. In this paper, we compare our method with both
ResNet101-based and ViT-based approaches on the VOC and ADE20K datasets. To demonstrate
that our method is not limited to ViT-based architectures, we replace the backbone in CoinSeg
with ResNet101 and apply our proposed DKD strategy for performance validation. We also report
re-implementation results of existing methods with ViT as the backbone, such as MBS _†_ and SSUL _†_ .
Additionally, we provide the performance of our architecture under joint training across multiple
incremental configurations, which is currently considered the upper bound of incremental learning
for models in this field Yang et al. (2023); Baek et al. (2022).


**Incremental** **configurations.** In addition to the widely explored incremental configurations on
the VOC dataset—15-1, 15-5, and 19-1. This paper also conducts experiments on configurations
with more incremental steps, such as 10-1 and 2-2. Furthermore, we evaluate our method on
four configurations on ADE20K: 100-50, 50-50, 100-10, and 100-5. These diverse experimental
setups allow for a comprehensive assessment of the robustness of our approach under various
incremental learning settings. For example, the 19-1 setting represents a typical setting where most
foreground classes (19 in total, excluding background) are learned in the base step, with only one
new class introduced in the incremental step. In the 2-2 configuration, the model is initially trained
on 2 foreground classes, with 2 additional classes added at each of the 9 subsequent incremental
steps—culminating in 18 new classes. This setup reflects a representative case of a substantial number
of classes to be acquired incrementally.


C ADDITIONAL EXPERIMENTS


C.1 PERFORMANCE COMPARISON IN THE DISJOINT SETTING


As shown in Tab. 6, in addition to the commonly explored overlap setting, we also conduct a
quantitative analysis on the VOC dataset using an additional disjoint setting. In the overlap setting,
the background includes previously learned classes, the classes to be learned at the current step, and
the classes to be learned in future steps. In the disjoint setting, the background excludes classes
required for future steps. Taking into account both old and new classes, our method achieves the best
results in all three incremental configurations under the disjoint setting, with overall improvements of
0 _._ 2%, 0 _._ 1%, and 0 _._ 9%, respectively. In cases with fewer incremental steps and fewer new classes
to learn, our method demonstrates better learning of new classes. Specifically, in the 19-1 task, our
method improves the MIoU metric by 5 _._ 2%. In the 15-1 task, with more incremental steps, our
method achieves a 4% improvement on new classes. This demonstrates that our method is not only
effective for the overlap setting mentioned in Tab. 1 in the main paper, but also excels in CISS under
the disjoint setting, showing both resistance to forgetting old classes and adaptability to new ones.


C.2 PERFORMANCE COMPARISON ON INDIVIDUAL CLASSES


To analyze the impact of our method on the 20 foreground classes in VOC under the widely explored
overlap setting, we compute the MIoU values of the 20 subcategories for both our method and the
MBS method across three different incremental configurations. Considering the average performance
across the 20 foreground classes as shown in Tab. 7, our method achieves the best results in multiple
incremental configurations, with improvements of 1 _._ 6%, 1 _._ 3%, and 2 _._ 5%, respectively. In the 2-2 task
with more incremental steps, our method demonstrates significant resistance to forgetting, particularly
for the “Aeroplane” and “Bottle” classes, with improvements of 17 _._ 8% and 29 _._ 0%, respectively.
Additionally, in the 15-5 incremental configuration, our method improves the performance of new
classes—“Potted plant”, “Sheep”, “Sofa”, “Train”, and “TV monitor”—by 3 _._ 2%, 2 _._ 1%, 5 _._ 6%, 4 _._ 0%,


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


Table 6: Performance comparison of the disjoint setting on the VOC dataset. Unlike the overlapping
setting, which has been widely explored in Tab. 1 of the main paper, the disjoint setting only includes
the classes from the current step _Ct_ and the previous old classes _C_ 1: _t−_ 1, without including any future
classes. The optimal and suboptimal performances are respectively represented in **red** and **blue**
bold. The _†_ symbol indicates results reproduced following the publicly released code. Our method
demonstrates notable performance gains across most incremental configurations, including a 5 _._ 2%
improvement in new classes for 19-1 and a 4% boost in new classes for 15-1.

|Method Publication Backbone|19-1 (2 steps)<br>Old New All|15-5 (2 steps)<br>Old New All|15-1 (6 steps)<br>Old New All|
|---|---|---|---|
|Joint<br>-<br>ViT<br>MiB Cermelli et al. (2020)<br>CVPR 2020<br>ResNet101<br>MiB† Cermelli et al. (2020)<br>CVPR 2020<br>ViT<br>SDR Michieli & Zanuttigh (2021)<br>CVPR 2021<br>ResNet101<br>PLOP Douillard et al. (2021a)<br>CVPR 2021<br>ResNet101<br>RBC Zhao et al. (2022)<br>ECCV 2022<br>ResNet101<br>RBC† Zhao et al. (2022)<br>ECCV 2022<br>ViT<br>MBS† Park et al. (2024)<br>ECCV 2024<br>ViT<br>Ours<br>-<br>ViT|84.5<br>80.6<br>84.3<br>69.6<br>25.6<br>67.5<br>**80.6**<br>45.2<br>78.9<br>69.9<br>37.3<br>68.3<br>75.4<br>38.9<br>73.7<br>76.4<br>45.8<br>74.9<br>80.9<br>42.1<br>79.1<br>**84.4**<br>**70.8**<br>**83.8**<br>**84.4**<br>**76.0**<br>**84.0**|85.3<br>80.7<br>84.2<br>71.8<br>43.3<br>65.0<br>75.0<br>59.9<br>71.4<br>73.5<br>47.3<br>67.3<br>71.0<br>42.8<br>64.3<br>75.1<br>49.7<br>69.1<br>77.7<br>59.1<br>73.3<br>**82.7**<br>**68.6**<br>**79.3**<br>**82.8**<br>**68.5**<br>**79.4**|85.3<br>80.4<br>84.1<br>46.2<br>12.9<br>38.3<br>66.7<br>26.3<br>57.1<br>59.2<br>12.9<br>48.2<br>57.9<br>13.7<br>47.4<br>61.7<br>19.5<br>51.7<br>69.0<br>28.4<br>59.3<br>**81.0**<br>**62.0**<br>**76.5**<br>**80.9**<br>**66.0**<br>**77.4**|


and 4 _._ 3%, respectively. Based on the above analysis, our method is shown to be effective not only
for configurations with fewer incremental steps (e.g., 15-5) but also for tasks with more incremental
steps (e.g., 2-2).


Table 7: Performance comparison of the overlapping setting between our method and the recent
MBS method on 20 individual classes in VOC (excluding the background class). _↑_ and _↓_ represent
the magnitude of improvement and decline in MIoU, respectively, when compared to MBS Park
et al. (2024). Considering the 20 foreground classes in VOC, our method achieves the best multiclass average performance in all 3 configurations, with improvements of 1 _._ 6%, 1 _._ 3%, and 2 _._ 5%,
respectively.


**MBS** Park et al. (2024) **Ours** **Comparison**
**Class Name**
15-5 15-1 2-2 15-5 15-1 2-2 15-5 15-1 2-2


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


C.3 ABLATION STUDY OF THE BACKBONE


To further assess the generalizability and compatibility of our proposed DKD strategy beyond ViTbased architectures, we also evaluate its effectiveness when integrated into a ResNet101-based CISS
framework. In particular, we adopt CoinSeg, built upon ResNet101 as backbone, and incorporate our
DKD module. We then conduct experiments under the 19-1 task setting to measure its performance.
As shown in Tab. 8, we report the detailed performance of the original CoinSeg (with ResNet101
backbone) and CoinSeg enhanced with DKD across all 20 foreground classes and the background.
For the new class “Tv monitor”, the original CoinSeg achieves a MIoU of 43 _._ 5%. After integrating
DKD, the performance improves significantly to 48 _._ 3%, resulting in a 4 _._ 8% gain. Averaged over all
20 foreground classes and background, the ResNet101-based architecture obtains a 0 _._ 3% increase
in MIoU and a 0 _._ 5% improvement in overall accuracy (ACC) with DKD. These results clearly
demonstrate that our DKD strategy is not limited to ViT-based models, but also exhibits compatibility
and orthogonality with ResNet101-based architectures.


Table 8: Performance Comparison on CoinSeg (ResNet101) and CoinSeg + DKD (ResNet101).
When applied to architectures with ResNet101 as the backbone, our method yields substantial gains
in the segmentation performance of newly introduced categories. Specifically, under the 19-1 setting,
the mIoU for the new class (TV monitor) improves by 4 _._ 8%, highlighting the adaptability and

|effectiveness of|our approach beyond ViT-based models.|Col3|
|---|---|---|
|**Class Name**|**Coinseg (ResNet101)**<br>MIoU<br>Acc|**Coinseg + DKD (ResNet101)**<br>MIoU<br>Acc|
|Background<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tv monitor<br>Average|92.0<br>95.6<br>90.2<br>95.8<br>38.3<br>89.2<br>91.2<br>95.9<br>68.6<br>88.5<br>76.3<br>92.8<br>92.3<br>95.6<br>86.2<br>92.8<br>95.1<br>98.6<br>31.5<br>40.8<br>84.1<br>91.2<br>55.1<br>62.4<br>88.5<br>95.0<br>79.9<br>88.8<br>87.5<br>95.5<br>88.0<br>91.1<br>53.4<br>68.9<br>76.1<br>89.9<br>45.8<br>53.8<br>88.4<br>92.7<br>43.5<br>85.8<br>73.9<br>85.7|92.2<br>95.7<br>89.1<br>96.1<br>37.1<br>90.5<br>91.5<br>96.3<br>68.0<br>90.2<br>76.3<br>93.2<br>93.2<br>96.3<br>85.0<br>93.1<br>93.5<br>98.0<br>33.2<br>43.1<br>83.6<br>90.7<br>56.2<br>65.4<br>87.6<br>94.1<br>80.1<br>90.4<br>87.7<br>95.9<br>87.6<br>91.5<br>55.3<br>72.7<br>76.4<br>89.6<br>48.0<br>57.0<br>88.9<br>93.3<br>48.3<br>77.0<br>74.2<br>86.2|


C.4 MORE EXPERIMENTS ABOUT THE PROPOSED DKD.


To evaluate the effectiveness of our method within other CISS methods, we integrate the proposed
DKD strategy into the original CoinSeg, employing a ViT backbone under the 19-1 incremental
setting. We then compare the performance with the original CoinSeg method. As shown in Tab. 9,
by augmenting CoinSeg with our DKD while retaining its original learning strategy, the MIoU for
the new class “Tv monitor” is significantly improved by 20 _._ 2%. Moreover, considering the average
performance over all 20 foreground classes and the background class, our method brings an overall
mIoU improvement of 1 _._ 3% and an accuracy (ACC) gain of 0 _._ 6%. These results demonstrate that our
DKD strategy not only enhances the model’s ability to acquire new class knowledge but also helps
preserve knowledge of previously learned classes by mitigating catastrophic forgetting. Besides, we
also conducted related experiments on incremental remote sensing data. On the iSAID dataset, joint


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


training achieved upper bounds of 45.2 and 52.3 for old and new classes, respectively. Our method
achieved 44.6 and 49.3, showing that it can perform close to joint training, which is considered the
upper bound for incremental tasks.


Table 9: Performance Comparison on CoinSeg (ViT) and CoinSeg + DKD (ViT). The integration of
the proposed DKD loss into the existing CoinSeg framework highlights its flexibility as a plug-andplay module. Notably, DKD boosts the mIoU of the novel class “TV monitor” by 20 _._ 2%.

|Class Name|Coinseg (ViT)<br>MIoU Acc|Coinseg + DKD (ViT)<br>MIoU Acc|
|---|---|---|
|Background<br>Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tv monitor<br>Average|93.0<br>96.3<br>88.1<br>92.1<br>42.9<br>86.6<br>95.1<br>97.1<br>74.9<br>88.0<br>85.3<br>94.6<br>95.0<br>97.3<br>90.5<br>92.4<br>96.2<br>98.3<br>48.1<br>59.5<br>93.8<br>97.7<br>58.4<br>60.1<br>92.9<br>97.7<br>91.7<br>94.6<br>92.4<br>95.9<br>89.7<br>92.6<br>68.6<br>85.0<br>93.2<br>95.3<br>59.5<br>67.9<br>90.8<br>92.2<br>38.6<br>86.1<br>79.9<br>88.9|94.0<br>97.2<br>90.5<br>95.0<br>42.7<br>89.7<br>95.1<br>97.4<br>75.6<br>89.1<br>85.1<br>94.8<br>95.6<br>97.5<br>90.3<br>92.4<br>96.2<br>98.3<br>49.0<br>62.3<br>93.7<br>98.0<br>59.1<br>61.0<br>92.6<br>97.8<br>92.2<br>95.2<br>92.5<br>96.1<br>89.8<br>93.0<br>68.6<br>85.8<br>92.5<br>95.6<br>60.1<br>68.6<br>90.9<br>92.4<br>58.8<br>82.8<br>81.2<br>89.5|


C.5 ABLATION STUDY OF THRESHOLD _τ_


Through the ablation study, we validate the necessity of releasing old parameters based on the
value of _τ_ . Specifically, by comparing Groups (Grps.) 1 and 2 in Tab. 3 of the main paper, we
observe that releasing old knowledge parameters with the assistance of _τ_ and minimizing the old
distribution can enhance the plasticity of new knowledge. To assess the reasonableness and impact
of selecting the threshold _τ_ during the release of old model parameters, we conduct experiments
with five different values of _τ_ on the 10-1 and 19-1 incremental configurations. As shown in Tab.
10, we evaluate the performance of different _τ_ values on old classes (including the background
class), new classes emerging in the incremental steps, and overall performance across all classes
(including the background class). In the 10-1 task, when _τ_ increases from 0.05 to 0.1, we observe a
0.3% performance improvement on new classes, while in the 19-1 task, the improvement reaches
1.6%. Thus, increasing _τ_ can enhance the ability to learn new classes. By observing the performance
changes of _τ_ values between 0.15 and 0.4, we find that excessively large values of _τ_ lead to the
forgetting of old classes. Based on the experiments in Tab. 10, we conclude that if the priority is
to preserve the performance on old classes, _τ_ = 0 _._ 05 is a good choice. However, if the focus is on
learning new knowledge, _τ_ = 0 _._ 1 is a better option. For all experiments in this paper, we use _τ_ = 0 _._ 1,
which is more favorable for learning new classes in the incremental steps.


To further validate that pruning with a predefined threshold does not lead to significant forgetting
of previously learned classes, we conduct experiments under the 19-1 setting. Specifically, we
first load the model at step 0 of the 19-1 task and evaluate its performance without pruning (W/O
pruning), measuring the mIoU and ACC on the 19 classes already learned. Next, we apply pruning
and re-evaluate the same model (W pruning) on the identical set of 19 classes. As shown in Tab.
11, the results indicate that this pruning strategy induces only negligible performance degradation.


21


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


Table 10: Ablation study on the threshold _τ_ . Considering both tasks with more incremental steps (e.g.,
10-1) and tasks with fewer incremental steps (e.g., 19-1), we evaluate the resistance to forgetting for
old classes, the ability to learn new classes, and the average performance across all classes (including
the background class). Based on the performance on new classes, we select _τ_ =0.1 as the value for all

|experiments.|Col2|
|---|---|
|**10-1**<br>_τ_ = 0.05<br>_τ_ = 0.1<br>_τ_ = 0.15<br>_τ_ = 0.2<br>_τ_ = 0.3<br>_τ_ =0.4|**19-1**<br>_τ_ = 0.05<br>_τ_ = 0.1<br>_τ_ = 0.15<br>_τ_ = 0.2<br>_τ_=0.3<br>_τ_=0.4|
|Old<br>83.1<br>81.7<br>81.8<br>81.6<br>81.1<br>76.9<br>New<br>72.5<br>72.8<br>72.9<br>72.1<br>64.5<br>49.4<br>All<br>78.1<br>77.5<br>77.6<br>77.1<br>73.2<br>63.8|83.1<br>82.8<br>82.6<br>82.4<br>82.4<br>1.1<br>72.5<br>74.1<br>72.8<br>72.3<br>71.9<br>3.1<br>82.6<br>82.4<br>82.1<br>81.9<br>81.9<br>1.2|


Table 11: Impact of parameter pruning on the performance of learned classes. The results show that
pruning leads to only a minor performance drop on the learned classes, indicating that it does not
cause signifcant forgetting.

|Class Name|W pruning<br>MIoU Acc|W/O pruning<br>MIoU Acc|
|---|---|---|
|Aeroplane<br>Bicycle<br>Bird<br>Boat<br>Bottle<br>Bus<br>Car<br>Cat<br>Chair<br>Cow<br>Dining table<br>Dog<br>Horse<br>Motorbike<br>Person<br>Potted plant<br>Sheep<br>Sofa<br>Train<br>Tv monitor<br>Average|93.2<br>97.5<br>43.9<br>89.6<br>93.1<br>96.2<br>78.5<br>91.0<br>86.4<br>95.4<br>92.8<br>95.5<br>88.8<br>95.7<br>94.4<br>97.1<br>51.7<br>67.4<br>95.0<br>97.2<br>62.6<br>65.1<br>93.2<br>97.1<br>92.7<br>95.2<br>85.9<br>91.5<br>88.5<br>93.6<br>71.7<br>85.8<br>91.4<br>93.9<br>57.2<br>63.2<br>90.8<br>98.3<br>Unseen classes<br>Unseen classes<br>82.3<br>90.2|94.3<br>97.6<br>45.2<br>90.5<br>94.8<br>97.6<br>78.2<br>91.8<br>87.7<br>96.2<br>92.7<br>95.5<br>89.8<br>96.3<br>95.4<br>97.4<br>55.7<br>74.6<br>95.3<br>97.7<br>63.9<br>66.5<br>94.3<br>97.9<br>92.6<br>94.9<br>86.7<br>91.7<br>89.3<br>94.4<br>72.5<br>96.4<br>94.4<br>96.6<br>57.6<br>63.2<br>90.6<br>98.3<br>Unseen classes<br>Unseen classes<br>83.2<br>91.1|


Furthermore, ablation results confirm that parameter release, under the guidance of _L_ Min, effectively
promotes the learning of new classes in subsequent steps (refer to Tab.12 and Sec. C.6 for more
details). These findings demonstrate the effectiveness of our parameter release strategy.


C.6 FURTHER EVALUATION OF DKD COMPONENT EFFECTIVENESS.


To evaluate the individual contributions of the 3 loss components ( _L_ Min, _L_ Esti, and _L_ Max) in DKD,
we conduct an ablation study focusing on their respective roles. As shown in Tab. 12, experiments
are conducted under 10-1, 15-1, and 19-1 settings to evaluate the effectiveness of the component
in different complex incremental scenarios. Comparing Grp. 1 with Grps. 2–4 show that each loss
function individually helps slightly mitigate the performance imbalance between old and new classes.
After adding the _L_ Min loss, the new classes in the 10-1 setting, which involve more incremental steps,
show a 25.4% improvement over the baseline Grp. 1. In settings with fewer incremental steps, such
as 15-1 and 19-1, the performance on new classes improves by 4.1% and 3.3%, respectively. These
results demonstrate that the adaptive adjustment and optimization of the old model’s parameters
during the distillation process significantly enhance the plasticity of new knowledge in continuous
learning. Comparisons between Grp. 5 and Grp. 8, as well as Grp. 6 and Grp. 8, show that
combining _L_ Min with either _L_ Esti or _L_ Max significantly improves the balance between old and new


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


Table 12: Ablation study of the components of the proposed DKD across multiple incremental
configurations. Through various incremental learning setups, it is observed that each loss function
helps promote a better balance between the performance on old and new classes.

|Grp.|L L L<br>Min Esti Max|10-1 (11 Steps)<br>Old New All|15-1 (6 Steps)<br>Old New All|19-1 (2 Steps)<br>Old New All|Average<br>Old New All|
|---|---|---|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8|✗<br>✗<br>✗<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓|63.5<br>21.3<br>43.4<br>69.9<br>46.7<br>58.9<br>72.7<br>45.2<br>59.6<br>68.7<br>35.7<br>53.0<br>82.1<br>70.8<br>76.7<br>82.1<br>71.2<br>76.9<br>71.2<br>49.6<br>60.9<br>81.7<br>72.8<br>77.5|74.9<br>38.3<br>66.2<br>81.8<br>42.4<br>72.4<br>81.0<br>36.6<br>70.4<br>82.2<br>50.1<br>74.6<br>82.2<br>68.0<br>78.8<br>83.3<br>67.0<br>79.4<br>81.2<br>39.9<br>71.4<br>82.6<br>70.0<br>79.6|83.3<br>66.7<br>82.5<br>82.5<br>70.0<br>81.9<br>82.9<br>69.7<br>82.3<br>82.9<br>71.0<br>82.3<br>82.6<br>69.8<br>82.0<br>82.6<br>72.3<br>82.1<br>82.9<br>70.2<br>82.3<br>82.8<br>74.1<br>82.4|73.9<br>42.1<br>64.0<br>78.1<br>53.0<br>71.1<br>78.9<br>50.5<br>70.8<br>77.9<br>52.3<br>70.0<br>82.3<br>69.5<br>79.2<br>82.7<br>70.2<br>79.5<br>78.4<br>53.2<br>71.5<br>82.4<br>72.3<br>79.8|


class performance, enhancing resistance to catastrophic forgetting and knowledge plasticity. The Grp.
7 vs. Grp. 8 comparison indicates that refining the old knowledge by _L_ Min allows better adaptation
to new classes. Ablation experiments conducted under 3 different incremental settings effectively
validate the role of the 3 distinct losses in class-incremental semantic segmentation. This further
demonstrates that the combination of these three losses enhances the model’s ability to learn new
classes while resisting catastrophic forgetting of old classes across various incremental settings.


C.7 FURTHER EXPERIMENTS OF ERROR ANALYSIS


Table 13: Error analysis. The overall performance error of the mean is approximately 0.1

|Col1|10-1 (11 steps)<br>1 2 3 Average Std|2-2 (10 steps)<br>1 2 3 Average Std|15-1(6 steps)<br>1 2 3 Average Std|19-1 (2 steps)<br>1 2 3 Average Std|15-5 (2 steps)<br>1 2 3 Average Std|
|---|---|---|---|---|---|
|Old<br>New<br>All|81.8<br>82.0<br>81.7<br>81.83<br>0.15<br>72.6<br>72.4<br>72.8<br>72.60<br>0.20<br>77.4<br>77.4<br>77.5<br>77.43<br>0.06|74.0<br>74.1<br>74.0<br>74.03<br>0.06<br>75.2<br>74.7<br>75.0<br>74.97<br>0.25<br>75.0<br>74.6<br>74.9<br>74.83<br>0.21|83.4<br>83.2<br>83.3<br>83.30<br>0.10<br>66.1<br>66.4<br>66.0<br>66.17<br>0.21<br>79.3<br>79.2<br>79.2<br>79.23<br>0.06|82.8<br>82.8<br>82.6<br>82.73<br>0.12<br>74.0<br>74.1<br>74.0<br>74.03<br>0.06<br>82.4<br>82.4<br>82.2<br>82.33<br>0.12|84.8<br>84.8<br>84.7<br>84.77<br>0.06<br>76.4<br>76.2<br>76.2<br>76.27<br>0.12<br>82.8<br>82.8<br>82.7<br>82.77<br>0.06|


As shown in Tab. 13, we evaluate the experimental variability by conducting repeated experiments
under the 10-1 and 19-1 settings. Among them, 10-1 represents a complex incremental setting with
numerous incremental steps, while 19-1 involves a greater number of classes learned during the base
step. We select these two representative and challenging settings for error analysis. Each setting is
run three times, recording the mIoU for old classes (Old), newly introduced classes (New), and all
classes excluding the background (All). We report the mean and standard deviation (Std) across three
repeated experiments. The results show that the overall variability for the combined performance of
both old and new classes is consistently close to 0.1, further validating the stability and reliability of
the performance under varying incremental learning conditions.


D MORE QUALITATIVE ANALYSIS


To further validate the effectiveness of our proposed method in improving pixel-level semantic classification accuracy, we present more qualitative comparisons with recent state-of-the-art approaches
in Figs. 7-10 under 15-1 setting. As illustrated in the second row of Fig. 7, existing methods tend
to confuse the background “cloth” with the “sofa” class after learning sofa-related knowledge in
the incremental step. In the third row, “windows” are mistakenly segmented as “TV monitor” or
“sofa”, indicating that these methods struggle to distinguish fine-grained class segmentation when
integrating new knowledge. Our method maintains the stability of previously learned classes while
effectively acquiring new concepts, demonstrating improved plasticity in adapting to new classes. Fig.
8 provides additional results, where the first row shows a case of “motorcycle” being largely forgotten
after five incremental steps. Some methods fails to retain the object’s contour and misclassifies
it as other irrelevant categories. A similar confusion is more evident in Fig. 9 in the background
of the first row. Our approach successfully preserves fine-grained class distinctions, significantly
reducing pixel-level misclassification. In the final row of Fig. 10, existing methods confuse the “chair”
occupied by a baby with a “sofa”. Our method achieves much more precise segmentation, with
only a small portion misclassified. These qualitative results clearly demonstrate that our approach
substantially enhances pixel-wise classification accuracy in class-incremental semantic segmentation,
while effectively mitigating class confusion during continual learning.


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


Image MIB LGKD Coinseg MBS Ours GT


Figure 7: Qualitative analysis of results under 15-1 setting. Our method provides more accurate
pixel-level segmentation for old classes with strong resistance to forgetting, while also reducing
misclassification of new class pixels, demonstrating superior plasticity in learning new classes.


E BROADER IMPACTS


In semantic segmentation, a common approach to handling newly emerging classes is to train the
new and old class data together. However, due to storage limitations and data privacy concerns, old
data is often inaccessible, rendering this approach impractical. Additionally, fine-tuning strategies are
prone to catastrophic forgetting. In this paper, we propose a distribution-based incremental semantic
segmentation learning strategy that mitigates forgetting without requiring access to old class data,


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


Image MIB LGKD Coinseg MBS Ours GT


Figure 8: Qualitative analysis of results under 15-1 setting. Our method provides more accurate
pixel-level segmentation for old classes with strong resistance to forgetting, while also reducing
misclassification of new class pixels, demonstrating superior plasticity in learning new classes.


while continuously learning new class knowledge. This method has promising applications in fields
such as autonomous driving, medical image segmentation, and environmental monitoring.


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


Image MIB LGKD Coinseg MBS Ours GT


Figure 9: Qualitative analysis of results under 15-1 setting. Our method provides more accurate
pixel-level segmentation for old classes with strong resistance to forgetting, while also reducing
misclassification of new class pixels, demonstrating superior plasticity in learning new classes.


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


Image MIB LGKD Coinseg MBS Ours GT


Figure 10: Qualitative analysis of results under 15-1 setting. Our method provides more accurate
pixel-level segmentation for old classes with strong resistance to forgetting, while also reducing
misclassification of new class pixels, demonstrating superior plasticity in learning new classes.


27
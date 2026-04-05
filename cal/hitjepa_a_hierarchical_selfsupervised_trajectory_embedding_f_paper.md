# HIT-JEPA: A HIERARCHICAL SELF-SUPERVISED TRA## JECTORY EMBEDDING FRAMEWORK FOR SIMILARITY
# COMPUTATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The representation of urban trajectory data plays a critical role in effectively analyzing spatial movement patterns. Despite considerable progress, the challenge of
designing trajectory representations that can capture diverse and complementary
information remains an open research problem. Existing methods struggle in incorporating trajectory fine-grained details and high-level summary in a single model,
limiting their ability to attend to both long-term dependencies while preserving
local nuances. To address this, we propose HiT-JEPA ( **H** ierarchical **I** nteractions
of **T** rajectory Semantics via a **J** oint **E** mbedding **P** redictive **A** rchitecture), a unified framework for learning multi-scale urban trajectory representations across
semantic abstraction levels. HiT-JEPA adopts a three-layer hierarchy that progressively captures point-level fine-grained details, intermediate patterns, and high-level
trajectory abstractions, enabling the model to integrate both local dynamics and
global semantics in one coherent structure. Extensive experiments on multiple
real-world datasets for trajectory similarity computation show that HiT-JEPA’s
hierarchical design yields richer, multi-scale representations. Code is available at:
[https://anonymous.4open.science/r/HiT-JEPA.](https://anonymous.4open.science/r/HiT-JEPA)


1 INTRODUCTION


With the widespread use of location-aware devices, trajectory data is now produced at an unprecedented rate Zhu et al. (2024); Qian et al. (2024). Effectively representing trajectory data powers
critical applications ranging from urban computing applications, such as travel time estimation Chen
et al. (2022b; 2021); Lin et al. (2023), trajectory clustering Fang et al. (2021); Yao et al. (2024); Bai
et al. (2020), and traffic analysis Yu et al. (2017). Trajectories exhibit multi-scale attributes, ranging
from short-term local transitions (e.g., turns and stops) to long-term strategic pathways or routines,
whereas capturing both the fine-grained point-level details of individual trajectories and higher-level
semantic patterns of mobility behavior within a unified framework is challenging. This necessitates a
representation learning model that accommodates this complexity.


Early trajectory analysis methods (heuristic methods) Alt & Godau (1995); Chen & Ng (2004); Chen
et al. (2005); Yi et al. (1998) relied on handcrafted similarity measures and point-matching heuristics.
Recently, deep-learning-based approaches have been applied to learn low-dimensional trajectory
embeddings, alleviating the need for manual feature engineering Yang et al. (2024); Yao et al. (2019);
Yang et al. (2021). Self-supervised learning frameworks Li et al. (2018); Cao et al. (2021), especially
contrastive learning (as shown in Fig. 1, left), further advanced trajectory representation learning by
leveraging large unlabeled datasets Chang et al. (2023); Liu et al. (2022); Li et al. (2024a). However,
these deep learning models usually generate a single scale embedding of an entire trajectory and cannot integrate different semantic levels, i.e., they often neglect fine-grained point-level information in
favor of broader trajectory-level features. On the other hand, most representation frameworks Chang
et al. (2023); Li et al. (2018) are restricted to a single form of trajectory data encoding and lack a mechanism to incorporate global context or higher-level information. Recent work Li et al. (2024b) (as
shown in Fig. 1, middle) explores alternative self-supervised paradigms that capture higher-level semantic information without manual augmentation. Nevertheless, a flexible and semantically aware representation architecture that unifies multiple levels of trajectory information remains an open question.


1


Figure 1: Structural comparisons among Contrastive Learning, JEPA, and Hierarchical JEPA.

Sequence models Vaswani et al. (2017); Hochreiter & Schmidhuber (1997), such as recurrent neural
networks (RNNs) and Transformers, are a natural choice for trajectory representation due to their
ability to process temporally ordered data. However, they exhibit inherent limitations when representing hierarchical semantics of trajectory data. Specifically, these models often operate at a single
temporal granularity: they either overemphasize point-level nuances, making them susceptible to
noise, or focus too heavily on coarse trajectory-level summaries and thus oversimplify critical details.
This single-scale bias in sequential models prevents them from integrating complementary information across abstraction levels and inhibits explicit semantic interactions between local (point-level),
intermediate (segment-level), and global (trajectory-level) representations, making it challenging for
sequence models to capture long-term dependencies while maintaining the detailed local nuances.
Besides, different from uniformly sampled time series data without spatial topology, trajectories are
more capricious due to their irregular, geometry-aware, and network-constrained characteristics.


A new framework is thus required to facilitate the model’s understanding of various levels of trajectory
representation information, to allow predictions to be grounded on more extensive, multi-dimensional
knowledge. In this paper, we propose HiT-JEPA (as shown in Fig. 1, right), a hierarchical framework
for urban trajectory representation learning, which is designed to address the gaps mentioned above
by integrating trajectory semantics across three levels of granularity. Its three-layer architecture
that explicitly captures (1) point-level details, modeling fine-grained spatial-temporal features of
consecutive points; (2) intermediate-level patterns, learning representations of local displacement
patterns that reflect mesoscopic movement structures; and (3) high-level abstractions, distilling
the overall semantic context as summarized moving behaviors of an entire trajectory. The model
unifies multiple information scales within a single representation framework through this hierarchy.
Moreover, HiT-JEPA enables interactions between adjacent levels to enrich and align the learned
trajectory embeddings across scales. By leveraging a joint embedding predictive architecture, the
framework learns to predict and align latent representations between these semantic levels, facilitating
semantic integration in a self-supervised manner. For clarity, we summarize our contributions as
follows:


    - We propose HiT-JEPA, a novel hierarchical trajectory representation learning architecture
that encapsulates movement information across different semantic levels inside a cohesive
framework. HiT-JEPA is the first architecture to explicitly unify both fine-grained and
abstract trajectory patterns within a single model.


    - HiT-JEPA introduces a joint embedding predictive architecture that unifies the entire trajectory across multiple levels of abstraction, resulting in a flexible representation that can
seamlessly incorporate local trajectory nuances and global semantic context. By striking
a balance between coarse-to-fine trajectory representations by our proposed hierarchical
interaction module, we address the limitations of single-scale or single-view models.


    - We conduct extensive experiments on real-world urban trajectory datasets spanning diverse
cities and movement patterns, demonstrating that HiT-JEPA’s semantically enriched, hierarchical embeddings exhibit comparative trajectory similarity search and remarkably superior
zero-shot performance across heterogeneous urban and maritime datasets.


2


Input trajectory
Input trajectory


2 RELATED WORK


**Urban Trajectory Representation Learning on Similarity Computation.** Self-supervised learning
methods for trajectory similarity computation are proposed to cope with robust and generalizable
trajectory representation learning on large, unlabeled datasets. t2vec Li et al. (2018) divides spatial
regions into rectangular grids and applies Skip-gram Mikolov et al. (2013) models to convert grid cells
into word tokens, then leverages an encoder-decoder framework to learn trajectory representations.
TrajCL Chang et al. (2023) applies contrastive learning on multiple augmentation schemes with a dualfeature attention module to learn both structural and spatial information in trajectories. CLEAR Li
et al. (2024a) proposes a ranked multi-positive contrastive learning method by ordering the similarities
of positive trajectories to the anchor trajectories. Recently, T-JEPA Li et al. (2024b) employs a Joint
Embedding Predictive Architecture that shifts learning from trajectory data into representation
space, establishing a novel self-supervised paradigm for trajectory representation learning. It is also
worth noting that robust trajectory representations are often the prerequisite for effective trajectory
clustering Yao et al. (2017); Wang et al. (2022); Fang et al. (2021), which focuses on uncovering
latent behavioral patterns by grouping trajectories with high semantic affinity. However, none of
the above methods manages to explicitly capture hierarchical trajectory information. We propose
HiT-JEPA to support coarse-to-fine, multi-scale trajectory abstraction extraction in a hierarchical
JEPA structure.


**Hierarchical** **Self-supervised** **Learning** **(HSSL).** Self-supervised learning methods have significantly advanced the capability to extract knowledge from massive amounts of unlabeled data.
Recent approaches emphasize multi-scale feature extraction to achieve a more comprehensive
understanding of complex data samples (e.g., lengthy texts or high-resolution images with
intricate details). In Computer Vision (CV), Chen _et_ _al._ Chen et al. (2022a) stack three Vision
Transformers Dosovitskiy et al. (2020) variants (varying patch size configurations) to learn cell, patch,
and region representations of gigapixel whole-slide images in computational pathology. Kong _et_
_al._ Kong et al. (2023) design a hierarchical latent variable model incorporating Masked Autoencoders
(MAE) He et al. (2022) to encode and reconstruct multi-level image semantics. Xiao _et al._ Xiao
et al. (2022) split the hierarchical structure by video semantic levels and employ different learning
objectives to capture distinct semantic granularities. In Natural Language Processing (NLP), Zhang _et_
_al._ Zhang et al. (2019) develop HIBERT, leveraging BERT Devlin et al. (2019) to learn sentence-level
and document-level text representations for document summarization. Li _et_ _al._ Li et al. (2022)
introduce HiCLRE, a hierarchical contrastive learning framework for distantly supervised relation
extraction, utilizing Multi-Granularity Recontextualization for cross-level representation interactions
to effectively reduce the influence of noisy data. In contrast to these methods, which partition inputs
into discrete fragments and directly propagate representations across levels, HiT-JEPA encodes the
entire trajectory at multiple abstraction levels by coupling adjacent-level attention weights from a
hierarchical JEPA to learn multi-scale urban trajectory representations.


3 METHODOLOGY


Compared to previous methods that only model trajectories at point-level, our primary goal in
designing HiT-JEPA is to bridge the gap between simultaneous modeling of local trajectory details
and global movement patterns by embedding explicit, cross-level trajectory abstractions into a JEPA
framework. To that end, as Fig. 2 illustrates, given a trajectory _T_, we apply three consecutive
convolutional layers followed by max pooling operations to produce point-level representation _T_ [(1)],
intermediate-level semantics _T_ [(2)] and high-level summary _T_ [(3)], where higher layer representations
consist of coarser but semantically richer trajectory patterns. Trajectory abstraction at layer _l_ is
learned by the corresponding JEPA layer JEPA [(] _[l]_ [)] to capture multi-scale sequential dependencies.


**Spatial** **region** **representation.** Considering the continuous and high-precision nature of GPS
coordinates, we partition the continuous spatial regions into fixed cells. But different from previous
approaches Chang et al. (2023); Li et al. (2024b;a) that use grid cells, we employ Uber H3 [1] to map
GPS points into hexagonal grids to select the grid cell resolutions adaptively according to the study
area size. Each hexagonal cell shares six equidistant neighbors, with all neighboring centers located
at the same distance from the cell’s center. Therefore, we structurally represent the spatial regions by


1https://h3geo.org/


3


Original Trajectory 𝑇


Figure 2: HiT-JEPA builds a three-level JEPA hierarchy to extract multi-scale trajectory semantics: (1) Level 1 encodes fine-grained, local point-level features; (2) Level 2 abstracts mesoscopic
segment-level patterns; (3) Level 3 captures coarse, global route structures. Trajectory information is
propagated from top to bottom, consecutive levels via attention weights.


a graph _G_ = ( _V, E_ ) where each node _vi_ _∈_ _V_ is a hexagon cell connecting to its neighboring cells
_vj_ _∈_ _V_ by an undirected edge _eij_ _∈_ _E_ . We pretrain the spatial node embeddings _H_ of graph _G_ using
node2vec Grover & Leskovec (2016), which produces an embedding set:

_H_ =         - _hi_ _∈_ R _[d]_ : _vi_ _∈_ _V_         - _,_ (1)


where each _hi_ encodes the relative position of node _vi_ . For a GPS location _P_ = ( _lon, lat_ ), we first
assign it to its grid cell index via:

_δ_ : R [2] _→{_ 1 _, . . ., |V |},_ (2)


and then look up its embedding _hδ_ ( _p_ ) _∈H_ .


**Hierarchical trajectory abstractions.** After obtaining the location embeddings, we construct trajectory representations at multiple semantic levels, which are termed hierarchical trajectory abstractions.
Given a trajectory _T_ with length _n_, we obtain its location embeddings and denote the input trajectory
as _T_ = ( _hδ_ ( _t_ 1) _, hδ_ ( _t_ 2) _, . . ., hδ_ ( _tn_ )) _∈_ (R _[d]_ ) _[n]_ . Then, we create its multi-level abstractions _T_ [(1)], _T_ [(2)],
_T_ [(3)] by a set of convolutions with kernel size of 3 and stride of 2, and max pooling layers:

_T_ [(1)] = LayerNorm(MaxPool1D(Conv1D( _T_ ))) _∈_ (R _[d]_ ) _[n]_ [1] _,_ _n_ 1 = _n,_ (3)


_T_ [(2)] = LayerNorm(MaxPool1D(Conv1D( _T_ [(1)] ))) _∈_ (R _[d]_ ) _[n]_ [2] _,_ _n_ 2 =    - _n_ 21    - _,_ (4)


_T_ [(3)] = LayerNorm(MaxPool1D(Conv1D( _T_ [(2)] ))) _∈_ (R _[d]_ ) _[n]_ [3] _,_ _n_ 3 =    - _n_ 22    - _._ (5)


where _T_ [(1)] in layer 1 preserves the channel dimension _d_ and sequence length _n_ 1 = _n_, _T_ [(2)] in layer
2 keeps the channel dimension and halves the sequence length to _n_ 2 = _n/_ 2, and _T_ [(3)] in layer 3 also
keeps the channel dimension and halves the sequence length to _n_ 3 = _n/_ 4. Higher-layer trajectory
abstractions contain aggregated, high-level trajectory semantic behaviors, while lower layers preserve
fine-grained, local dynamic details.


**Target encoder branch.** For the target encoder branch, at each level _l ∈{_ 1 _,_ 2 _,_ 3 _}_ the target trajectory
representation is extracted by:
_S_ [(] _[l]_ [)] = _Eθ_ [(] ¯ _[l]_ [)][(] _[T]_ [ (] _[l]_ [)][)] (6)

where _Eθ_ [(] _[l]_ [)] is the target encoder at layer _l_ . Similar to previous JEPA methods LeCun (2022); Assran
et al. (2023); Li et al. (2024b); Bardes et al. (2023), we randomly sample _M_ times from target


4


representation to create the targets, where _S_ [(] _[l]_ [)] ( _i_ ) = _{Sj_ [(] _[l]_ [)] _[}][j][∈M]_ _i_ [.] [Therefore,] _[S]_ [(] _[l]_ [)][(] _[i]_ [)] [is] [the] _[i]_ [-th]
sampled target and _Mi_ is the _i_ -th sampling mask starting from a random position. To ensure the
diversity of learning targets, we follow T-JEPA Li et al. (2024b) and introduce a set of masking
ratios _r_ = _{r_ 1 _, r_ 2 _, r_ 3 _, r_ 4 _, r_ 5 _}_ where each ratio value specifies the fraction of the representation to
mask. At each sampling step, we uniformly draw one ratio from r. We also introduce a probability
_p_ : with probability _p_, we apply successive masking, and with probability 1 _−_ _p_, we scatter the
masks randomly. Successive masking encourages the encoder to learn both local and long-range
dependencies.


**Context encoder branch.** For the context encoder branch, we initially sample a trajectory context
_C_ [(] _[l]_ [)] from _T_ [(] _[l]_ [)] at level _l_ by a mask _CT_ at with sampling ratio _pγ_ . Next, to prevent any information
leakage, we remove from _C_ [(] _[l]_ [)] all positions that overlap with the targets _S_ [(] _[l]_ [)] to obtain the context
input _T_ _[′]_ [(] _[l]_ [)] . The context trajectory representation _S_ _[′]_ [(] _[l]_ [)] at level _l_ is extracted by:

_S_ _[′]_ [(] _[l]_ [)] = _Eθ_ [(] _[l]_ [)][(] _[T][ ′]_ [(] _[l]_ [)][)] (7)


where _Eθ_ [(] _[l]_ [)] is the context encoder at level _l_ . During inference, we use _S_ _[′]_ [(1)] from _Eθ_ [(1)][,] [enriched]
by the full hierarchy of multi-scale abstractions, as the final output of trajectory representations for
similarity comparison or downstream fine-tuning.


**Predictions.** Once we have both context representations _S_ _[′]_ [(] _[l]_ [)] and targets _S_ [(] _[l]_ [)] at level _l_, we apply
JEPA predictor _Dϕ_ [(] _[l]_ [)] [on] _[ S][′]_ [(] _[l]_ [)] [to approximate] _[ S]_ [(] _[l]_ [)] [with the help of the mask tokens] _[ z]_ [(] _[l]_ [)][:]

_S_         - _[′]_ [(] _[l]_ [)] ( _i_ ) = _Dϕ_ [(] _[l]_ [)][(CONCAT(] _[S][′]_ [(] _[l]_ [)] _[,]_ [ PE(] _[i]_ [)] _[ ⊕]_ [(] _[z]_ [(] _[l]_ [)][)))] (8)


where CONCATE( _·_ ) denotes concatenation and PE( _i_ ) refers to the positional embedding after
applying the target sampling mask _Mi_ . _⊕_ is element-wise addition between these masked positional
embeddings and the mask tokens. Then, we concatenate the mask tokens with positional information
with the context representations to guide the predictor in approximating the missing components in
the targets at the representation space.


**Hierarchical** **interactions.** By applying JEPA independently at each level, we learn trajectory
representations at multiple scales of abstractions. However, the encoders at each level remain siloed
and retain only their scale-specific information without leveraging insights from other layers. To
enable hierarchical and multi-scale feature extraction, we propagate high-level information down to
the next lower abstraction layer.


We adopt Transformer encoders Vaswani et al. (2017) for both context and target encoders as their
self-attention module is proven highly effective in sequential modeling. Therefore, for both branches,
we inject attention weights to the next lower level as a “top-down spotlight” where the high-level
encoder casts its attention maps to the lower layer, lighting up where the lower-level encoder should
attend. For clarity, we illustrate the process using the target encoder branch as an example. At level
_l_, given the query and key matrices _Q_ [(] _[l]_ [)] and _K_ [(] _[l]_ [)] of an input trajectory abstraction _T_ [(] _[l]_ [)], we first
retrieve the attention coefficient by:


( _l−_ 1)

            - _n_
_A_ [(] _[l]_ [)] _∈_ [0 _,_ 1] _[n]_ [(] _[l]_ [)] _[×][n]_ [(] _[l]_ [)] _,_ _n_ [(] _[l]_ [)] =

2


5


(12)


_dk_ = _[d]_ _H_ [(] _[l]_ [)] _[, Q]_ _i_ [(] _[l]_ [)] = _Q_ [(] _[l]_ [)] _Wi_ _[Q,]_ [(] _[l]_ [)] _,_ _Ki_ [(] _[l]_ [)] = _K_ [(] _[l]_ [)] _Wi_ _[K,]_ [(] _[l]_ [)] _,_ _A_ [(] _i_ _[l]_ [)] = softmax� _Q_ [(] _i_ _[l]_ ~~_√_~~ [)] _Kdki_ [(] _[l]_ [)] _[ ⊤]_


_,_ _i_ = 1 _, . . ., H_ (9)


where _H_ is the number of attention heads, _Wi_ _[Q,]_ [(] _[l]_ [)] and _Wi_ _[K,]_ [(] _[l]_ [)] are head- _i_ projections, _d_ [(] _[l]_ [)] is the
channel dimension, and _A_ [(] _i_ _[l]_ [)] is the attention coefficient of the head- _i_ . The multi-head attention
coefficient _A_ [(] _[l]_ [)] are concatenated and projected by:

_A_ [(] _[l]_ [)] = Concat� _A_ [(] 1 _[l]_ [)] _[, . . ., A]_ _H_ [(] _[l]_ [)]         - _W_ _[O,]_ [(] _[l]_ [)] (10)

where _W_ _[O,]_ [(] _[l]_ [)] is the multi-head projection. To construct the output representation _S_ [(] _[l]_ [)] at level _l_, we
simply apply the value matrix _V_ [(] _[l]_ [)] by:


_S_ [(] _[l]_ [)] = _A_ [(] _[l]_ [)] _V_ [(] _[l]_ [)] (11)


Since the dimension of _A_ [(] _[l]_ [)] is:


where _n_ [(] _[l]_ [)] is the length of trajectory abstractions at level _l_, which is half of _n_ [(] _[l][−]_ [1)] at level _l −_ 1 due
to Eq. 4 and Eq. 5. We need to upsample the attention coefficients:

_A_      - [(] _[l]_ [)] = Interpolatebilinear� _A_ [(] _[l]_ [)][�] _∈_ [0 _,_ 1] _[n]_ [(] _[l][−]_ [1)] _[×][n]_ [(] _[l][−]_ [1)] (13)


where we adopt bilinear interpolation to upsample the attention weights at level _l_ . To propagate the
upsampled _A_ [(] _[l]_ [)] to the next lower level, We refer to Chang et al. (2023) to calculate a weighted sum

[�]
between _A_ [(] _[l]_ [)] and lower level attention coefficient _A_ [(] _[l][−]_ [1)] . Therefore, we obtain the updated attention

[�]
coefficient _A_ [(] _[l][−]_ [1)] at level _l −_ 1 by:


_A_ [(] _[l][−]_ [1)] = ( _σA_ [(] _[l][−]_ [1)] + (1 _−_ _σ_ ) _A_ [(] _[l]_ [)] ) (14)

[�]

where _σ_ is a learnable scale factor weighting the importance of _A_ [(] _[l]_ [)] . Attention coefficient _A_ _[′]_ [(] _[l]_ [)] from
the context encoders follows an identical procedure. This way, the coarse, global insights guide the
fine-grained feature extraction in the next layer to focus on the most semantically important trajectory
segments. This alignment sharpens local feature extraction so it stays consistent with the overall
context.

**Loss function.** After obtaining the predicted representation _S_ _[′]_ [(] _[l]_ [)] ( _i_ ) and the _i_ -th target representation

[�]
_S_ [(] _[l]_ [)] ( _i_ ) at level _l_, we apply SmoothL1 to calculate the loss _L_ [(] _[l]_ [)] between them:


           - ��           
_L_ [(] VICReg _[l]_ [)]


where we sum over the channel and sequence length dimension _d_ [(] _[l]_ [)] and _N_ [(] _[l]_ [)], and average over the
batch and number of target masks dimension _B_ and _M_ to obtain JEPA loss _L_ [(] JEPA _[l]_ [)] [.] [We also add]
VICReg Bardes et al. (2021) to prevent representation collapse, yielding more discriminative representations. We obtain the regularization term _L_ [(] VICReg _[l]_ [)] [by summing up the variance loss][ VarLoss(] _[·]_ [)]

and covariance loss CovLoss( _·_ ) of both expanded context representation _z_ ctx [(] _[l]_ [)] [=] [MLP(] _[S][′]_ [(] _[l]_ [)][)][ and]
expanded target representation _z_ tar [(] _[l]_ [)] [= MLP(] _[S]_ [(] _[l]_ [)][)][ via a single-layer MLP. Afterwards,] _[ L]_ VICReg [(] _[l]_ [)] [is]
added to the loss _L_ [(] _[l]_ [)] at level _l_ .


For level _l ∈{_ 1 _,_ 2 _,_ 3 _}_, we calculate a weighted sum to obtain the final loss _L_ :


_L_ = _λ ∗L_ [(1)] + _µ ∗L_ [(2)] + _ν ∗L_ [(3)] (16)


where _λ_, _µ_ and _ν_ are the scale factors for loss at each level.


4 EXPERIMENTS
We conduct experiments on three real-world urban GPS trajectory datasets: Porto [2], T-Drive Yuan et al.
(2011; 2010) and GeoLife Zheng et al. (2008; 2009; 2010), two FourSquare datasets: FourSquareTKY and FourSquare-NYC Yang et al. (2014), and one vessel trajectory dataset: Vessel Tracking
Data Australia, which we call “AIS(AU)” [3] . The dataset details can be found in Appendices A.1.
We compare HiT-JEPA with the three most recent self-supervised methods on trajectory similarity
computation: TrajCL Chang et al. (2023), CLEAR Li et al. (2024a) and T-JEPA Li et al. (2024b).
The details of these methods are listed in Appendices A.2


4.1 QUANTITATIVE EVALUATION


In this section, we evaluate HiT-JEPA and compare it to baselines in three experiments: most similar
trajectory search, robustness of learn representations, and generalization with downstream fine-tuning.
We combine the first two experiments as “Self-similarity”.


2https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data
3https://www.operations.amsa.gov.au/spatial/DataServices/DigitalData


6


_d_ [(] _[l]_ [)]

- SmoothL1�� _S′_ ( _l_ )( _i_ ) _b,n,k,_ _S_ ( _l_ )( _i_ ) _b,n,k_ 

_k_ =1


_B_


_b_ =1


_N_ [(] _[l]_ [)]


_n_ =1


1
_L_ [(] _[l]_ [)] =
_M B_


_M_


_i_ =1


- �� 
_L_ [(] JEPA _[l]_ [)]


+ VarLoss� _z_ tar [(] _[l]_ [)] - + VarLoss� _z_ ctx [(] _[l]_ [)] - + CovLoss� _z_ tar [(] _[l]_ [)] - + CovLoss� _z_ ctx [(] _[l]_ [)] 


(15)


_._


4.1.1 SELF-SIMILARITY


Following similar experimental settings of previous work Chang et al. (2023); Li et al. (2024b), we
construct a Query trajectory set _Q_ and a database trajectory _D_ for the testing set given a trajectory.
_Q_ has 1,000 trajectories for Porto, T-Drive, and GeoLife, 600 for TKY, 140 for NYC, and 1400 for
AIS(AU). And _D_ has 100,000 trajectories for Porto, 10,000 for T-Drive and Geolife, 3000 for TKY,
700 for NYC, and 7000 for AIS(AU). Detailed experimental settings can be found in Appendices A.4.


Table 1: Mean-rank comparison of methods across meta ratios _R_ 1 _∼R_ 5. For each meta ratio, we
report the mean ranks under varying DB size _|D|_, downsampling rate _ρs_, and distortion rate _ρd_ . **Bold**
value are the lowest mean ranks and underlined values are the second lowest.


|Dataset Method R1 R2 R3 R4 R5<br>|D| ρs ρd |D| ρs ρd |D| ρs ρd |D| ρs ρd |D| ρs ρd|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|Porto<br>TrajCL<br>**1.004**<br>**1.047**<br>**1.017**<br>**1.007**<br>**1.170**<br>**1.029**<br>**1.008**<br>**1.905**<br>**1.036**<br>**1.011**<br>**6.529**<br>**1.060**<br>**1.014**<br>68.557<br>**1.022**<br>TrjSR<br>3.240<br>12.553<br>12.509<br>5.321<br>16.945<br>15.401<br>7.073<br>37.150<br>15.901<br>8.740<br>65.413<br>28.914<br>10.192<br>149.950<br>32.730<br>CLEAR<br>3.235<br>7.796<br>4.250<br>4.012<br>13.323<br>4.442<br>4.088<br>22.814<br>4.284<br>4.137<br>44.865<br>4.438<br>4.204<br>123.921<br>4.399<br>T-JEPA<br>1.029<br>1.455<br>1.097<br>1.048<br>2.304<br>1.084<br>1.053<br>4.413<br>1.115<br>1.061<br>9.599<br>1.110<br>1.074<br>**23.900**<br>1.123<br>HiT-JEPA<br>1.026<br>1.369<br>1.074<br>1.043<br>2.624<br>1.077<br>1.048<br>5.541<br>1.085<br>1.058<br>13.773<br>1.093<br>1.065<br>28.806<br>1.119|
|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|T-Drive<br>TrajCL<br>1.111<br>1.203<br>1.267<br>1.128<br>1.348<br>3.320<br>1.146<br>1.668<br>1.355<br>1.177<br>1.936<br>1.513<br>1.201<br>3.356<br>1.179<br>TrjSR<br>110.726<br>674.16<br>581.776<br>223.841<br>795.331<br>572.944<br>356.941<br>870.73<br>566.816<br>475.872<br>960.404<br>545.278<br>592.146<br>1033.404<br>566.696<br>CLEAR<br>1.047<br>1.305<br>1.111<br>1.062<br>1.484<br>1.110<br>1.077<br>1.964<br>1.171<br>1.088<br>3.497<br>1.152<br>1.104<br>3.902<br>1.172<br>T-JEPA<br>**1.032**<br>1.088<br>1.054<br>**1.034**<br>1.225<br>1.061<br>**1.036**<br>1.617<br>1.069<br>1.045<br>3.226<br>1.067<br>1.049<br>4.115<br>1.078<br>HiT-JEPA<br>1.040<br>**1.057**<br>**1.035**<br>1.040<br>**1.085**<br>**1.029**<br>1.040<br>**1.172**<br>**1.039**<br>**1.040**<br>**1.389**<br>**1.033**<br>**1.040**<br>**2.222**<br>**1.034**|
|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|GeoLife<br>TrajCL<br>1.130<br>1.440<br>7.973<br>1.168<br>1.435<br>19.266<br>1.195<br>1.720<br>12.397<br>1.234<br>1.616<br>10.560<br>1.256<br>2.675<br>11.035<br>TrjSR<br>6.765<br>8.332<br>7.747<br>7.393<br>8.594<br>7.942<br>7.661<br>8.688<br>7.648<br>7.767<br>8.566<br>8.534<br>8.350<br>8.770<br>9.460<br>CLEAR<br>1.110<br>1.196<br>1.212<br>1.124<br>1.318<br>1.211<br>1.144<br>1.818<br>1.189<br>1.145<br>2.237<br>1.239<br>1.155<br>3.712<br>1.333<br>T-JEPA<br>**1.019**<br>**1.052**<br>**1.047**<br>1.034<br>**1.030**<br>**1.093**<br>1.036<br>**1.103**<br>**1.101**<br>1.040<br>**1.150**<br>**1.154**<br>1.047<br>**1.218**<br>**1.197**<br>HiT-JEPA<br>1.033<br>1.061<br>1.170<br>**1.033**<br>1.111<br>1.370<br>**1.033**<br>1.247<br>1.357<br>**1.033**<br>1.377<br>1.509<br>**1.033**<br>1.573<br>1.511|
|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|TKY<br>(zero-shot)<br>TrajCL<br>17.590<br>66.963<br>75.397<br>32.377<br>67.835<br>79.228<br>46.958<br>116.677<br>59.222<br>62.145<br>170.460<br>69.642<br>78.722<br>211.487<br>65.258<br>TrjSR<br>8.673<br>31.770<br>27.505<br>17.120<br>37.070<br>30.758<br>22.310<br>48.985<br>30.923<br>26.820<br>64.380<br>33.113<br>29.318<br>84.302<br>34.043<br>CLEAR<br>119.561<br>591.345<br>583.863<br>242.493<br>626.075<br>591.460<br>349.132<br>646.160<br>587.138<br>456.525<br>662.553<br>588.212<br>577.238<br>709.903<br>591.107<br>T-JEPA<br>1.948<br>3.060<br>3.245<br>2.272<br>4.227<br>3.165<br>2.617<br>7.975<br>3.313<br>2.913<br>18.173<br>3.202<br>3.275<br>19.135<br>3.127<br>HiT-JEPA<br>**1.508**<br>**2.490**<br>**2.060**<br>**1.707**<br>**2.962**<br>**2.002**<br>**1.835**<br>**4.985**<br>**2.067**<br>**1.930**<br>**10.268**<br>**2.045**<br>**2.057**<br>**14.755**<br>**1.988**|
|NYC<br>(zero-shot)|TrajCL<br>TrjSR<br>CLEAR<br>T-JEPA<br>HiT-JEPA|4.336<br>16.886<br>15.093<br>3.929<br>5.457<br>6.307<br>19.693<br>68.843<br>68.057<br>1.450<br>1.950<br>1.714<br>**1.343**<br>**1.743**<br>**1.493**|6.457<br>18.857<br>16.971<br>4.793<br>5.171<br>7.950<br>32.171<br>74.964<br>68.321<br>1.514<br>3.050<br>1.736<br>**1.364**<br>**2.143**<br>**1.500**|9.129<br>22.007<br>16.443<br>5.457<br>8.350<br>6.679<br>43.214<br>75.121<br>69.221<br>1.571<br>2.400<br>1.679<br>**1.414**<br>**1.636**<br>**1.500**|12.350<br>37.579<br>11.236<br>5.821<br>12.757<br>7.443<br>55.507<br>79.514<br>70.507<br>1.636<br>2.457<br>1.771<br>**1.457**<br>**2.407**<br>**1.550**|15.071<br>36.650<br>6.543<br>6.007<br>14.329<br>7.907<br>67.207<br>84.421<br>65.914<br>1.714<br>5.850<br>1.807<br>**1.500**<br>**3.343**<br>**1.471**|
|AIS(AU)<br>(zero-shot)|TrajCL<br>TrjSR<br>CLEAR<br>T-JEPA<br>HiT-JEPA|9.057<br>37.721<br>37.866<br>692.000<br>3658.400<br>3649.450<br>38.042<br>188.171<br>184.600<br>2.156<br>5.661<br>4.753<br>**1.483**<br>**4.119**<br>**2.759**|18.771<br>9.878<br>37.879<br>1390.364<br>3661.421<br>3649.407<br>73.164<br>187.914<br>184.579<br>3.176<br>6.849<br>4.753<br>**1.954**<br>**6.357**<br>**2.759**|26.538<br>41.068<br>37.862<br>2136.271<br>3675.043<br>3649.150<br>112.371<br>192.571<br>184.600<br>3.889<br>**9.486**<br>4.755<br>**2.311**<br>10.233<br>**2.758**|33.004<br>45.352<br>37.911<br>2942.586<br>3714.564<br>3649.086<br>150.050<br>191.629<br>184.871<br>4.364<br>**13.055**<br>4.758<br>**2.579**<br>15.180<br>**2.757**|37.866<br>48.651<br>38.399<br>2892.264<br>3700.221<br>3649.371<br>184.600<br>198.843<br>184.593<br>4.754<br>**16.986**<br>4.749<br>**2.758**<br>20.267<br>**2.755**|


Table 1 shows the mean ranks of all methods. HiT-JEPA achieves the overall lowest mean ranks
across five of the six datasets. For urban GPS datasets, Porto, T-Drive, and GeoLife, we have the
lowest ranks in the T-Drive dataset. For example, the mean ranks of DB size _|D|_ across 20% _∼_ 100%
and distortion rates _ρd_ across 0.1 _∼_ 0.5 remains very steady (1.040 _∼_ 1.041 and 1.031 _∼_ 1.038). This
dataset has taxi trajectories with much longer irregular sampling intervals (3.1 minutes on average).
By leveraging a hierarchical structure to capture the global and high-level trajectory abstractions,
HiT-JEPA learns features that remain invariant against noise and sparse sampling, resulting in more
robust and accurate representations against low and irregularly sampled trajectories with limited
training samples. We achieve comparative mean ranks (only 2.8% higher) with T-JEPA on GeoLife,
and overall, the second best on Porto. This is because Porto trajectories inhabit an especially dense
spatial region, so TrajCL can exploit auxiliary cues such as movement speed and orientations to tease
apart nearly identical paths. However, relying on these features undermines the generalization ability
in lower-quality trajectories (e.g., in T-Drive) and knowledge transfer into other cities.


Next, we evaluate zero-shot performance on TKY, NYC, and AIS(AU). HiT-JEPA consistently
achieves the lowest mean ranks across all database sizes, downsampling, and distortion rates. Both
TKY and NYC consist of highly sparse and coarse check-in sequences, lacking trajectory waypoints,
which challenge the summarization ability of the models. Benefiting from the hierarchical structure,
HiT-JEPA first summarizes the mobility patterns at a coarse level, then refines the check-in details
at finer levels. Crucially, the summarization knowledge is transferred from dense urban trajectories in
Porto, demonstrating that HiT-JEPA learns more generalizable representations than TrajCL in Porto
with more essential spatiotemporal information captured in trajectories. Even on AIS(AU) with trajectories across the ocean-wide scales, HiT-JEPA maintains overall the lowest mean ranks, demonstrating
its ability to handle multiple forms of trajectories that spread over various regional scales. We find that
even though CLEAR outperforms TrajCL on T-Drive and GeoLife, it exhibits weak generalization
in zero-shot experiments on TKY, NYC, and AIS(AU). TrjSR showed the weakest overall perfor

7


Table 2: Comparisons with fine-tuning 2-layer MLP decoder. **Bold** value are the lowest mean ranks
and underlined values are the second lowest.


|Dataset|Method|EDR|LCSS|Hausdorff|Fre´chet|Average|
|---|---|---|---|---|---|---|
|Dataset|Method|HR@5_↑_HR@20_↑_R5@20_↑_|HR@5_↑_HR@20_↑_R5@20_↑_|HR@5_↑_HR@20_↑_R5@20_↑_|HR@5_↑_HR@20_↑_R5@20_↑_|HR@5_↑_HR@20_↑_R5@20_↑_|
|Porto|TrajCL<br>TrjSR<br>CLEAR<br>T-JEPA<br>HiT-JEPA|0.137<br>0.179<br>0.301<br>0.085<br>0.083<br>0.157<br>0.078<br>0.075<br>0.142<br>0.154<br>0.194<br>0.336<br>**0.163**<br>**0.197**<br>**0.337**|0.329<br>0.508<br>0.663<br>0.162<br>0.197<br>0.292<br>0.164<br>0.198<br>0.293<br>0.365<br>0.551<br>0.713<br>**0.369**<br>**0.558**<br>**0.720**|0.456<br>0.574<br>0.803<br>0.166<br>0.192<br>0.304<br>0.152<br>0.131<br>0.232<br>**0.525**<br>**0.633**<br>**0.869**<br>0.466<br>0.599<br>0.835|0.412<br>0.526<br>0.734<br>0.157<br>0.173<br>0.288<br>0.192<br>0.165<br>0.316<br>0.433<br>0.565<br>0.771<br>**0.450**<br>**0.587**<br>**0.810**|0.468<br>0.188<br>0.178<br>**0.509**<br>0.508|
|T-Drive|TrajCL<br>TrjSR<br>CLEAR<br>T-JEPA<br>HiT-JEPA|0.094<br>0.131<br>0.191<br>0.076<br>0.068<br>0.114<br>0.093<br>0.084<br>0.143<br>0.094<br>0.147<br>0.215<br>**0.112**<br>**0.170**<br>**0.260**|0.159<br>0.289<br>0.366<br>0.076<br>0.080<br>0.118<br>0.126<br>0.166<br>0.216<br>0.205<br>0.366<br>0.469<br>**0.221**<br>**0.384**<br>**0.493**|0.173<br>0.256<br>0.356<br>0.095<br>0.090<br>0.143<br>0.142<br>0.158<br>0.243<br>0.158<br>0.229<br>0.329<br>**0.222**<br>**0.316**<br>**0.456**|0.138<br>0.187<br>0.274<br>0.098<br>0.094<br>0.145<br>0.135<br>0.170<br>0.283<br>0.125<br>0.159<br>0.249<br>**0.158**<br>**0.219**<br>**0.325**|0.218<br>0.100<br>0.163<br>0.229<br>**0.278**|
|GeoLife|TrajCL<br>TrjSR<br>CLEAR<br>T-JEPA<br>HiT-JEPA|0.193<br>0.363<br>0.512<br>0.138<br>0.246<br>0.443<br>0.175<br>0.164<br>0.311<br>**0.195**<br>0.383<br>0.527<br>0.183<br>**0.414**<br>**0.564**|0.232<br>0.484<br>0.584<br>0.229<br>0.330<br>0.479<br>0.224<br>0.224<br>0.342<br>0.242<br>0.515<br>0.586<br>**0.250**<br>**0.525**<br>**0.609**|0.479<br>0.536<br>0.745<br>0.492<br>0.439<br>0.692<br>0.347<br>0.308<br>0.499<br>0.606<br>0.656<br>0.857<br>**0.643**<br>**0.700**<br>**0.885**|0.398<br>0.463<br>0.708<br>0.383<br>0.362<br>0.614<br>0.397<br>0.273<br>0.539<br>**0.488**<br>0.406<br>0.731<br>0.467<br>**0.555**<br>**0.842**|0.475<br>0.404<br>0.320<br>0.516<br>**0.553**|


mance across all datasets. This is because image-based representations have difficulty distinguishing
fine-grained trajectory differences, a challenge exacerbated by lower data quality (e.g., T-Drive).


4.1.2 DOWNSTREAM FINE-TUNING


To evaluate the generalization ability of HiT-JPEA, we conduct downstream fine-tuning on its learned
representations. Specifically, we retrieve and freeze the encoder of HiT-JEPA and other baselines,
concatenated with a 2-layer MLP decoder, then train the decoder to approximate the computed
trajectory similarities by heuristic approaches. This setting is first proposed by TrajCL Chang et al.
(2023), then followed by T-JEPA Li et al. (2024b), to quantitatively assess whether the learned
representations can generalize to approach the computational processes underlying each heuristic
measure. In real applications, fine-tuned models can act as efficient, “fast” approximations of
traditional heuristic measures, alleviating their quadratic time-complexity bottleneck. We report hit
ratios HR@5 and HR@20 to evaluate the correct matches between top-5 predictions and each of the
top-5 and top-20 ground truths. We also report the recall R5@20 to evaluate the correct matches of
top-5 ground truths from predicted top-20 predictions. We approximate all model representations to
4 heuristic measures: EDR, LCSS, Hausdorff and Discret Frechet.´ We do not include TrjSR here as
its results are proven to be less competitive in Chang et al. (2023).
From Table 2, we can observe that HiT-JEPA achieves the highest overall performance. In the column
“Average”, we calculate the average of all reported results for each model on each dataset. HiT-JEPA
outperforms T-JEPA on T-Drive and GeoLife for 12.6% and 6.4%, with only 3.7% lower on Porto.
For results on T-Drive, HiT-JEPA consistently outperforms the T-JEPA across all measures, especially
in Hausdorff and Discret Frechet´ measures, where we achieve relative average improvements of
14.7% and 19.9%, respectively. For GeoLife, even though we have some cases that achieve slightly
lower results than T-JEPA in EDR and Hausdorff, we are overall 6.1% and 1.8% higher on average in
these two measures. For Porto, although our results are 3.7% lower than T-JEPA on average across
all measures, we have successfully made minor improvements in LCSS measure. Visualizations of
predictions can be found in Fig. 12 and Fig. 13 in Appendices A.8.


4.2 VISUALIZATIONS OF HIT-JEPA EMBEDDINGS.


HiT-JEPA encodes and predicts trajectory information only in the representation space, making
it more difficult than generative models such as MAE He et al. (2022) to evaluate the learned
representation quality at the data level. To assess and gauge the validity of the representations of
HiT-JEPA, we project the encoded _S_ _[′]_ [(1)] from _Eθ_ [(1)] (on full trajectories) and predicted _S_ [�] _[′]_ [(1)] from
_Dϕ_ [(1)] (on masked trajectories) back onto the hexagonal grid at their GPS coordinates for visual
comparisons.


First, we freeze the context encoders and predictors across all levels in a pre-trained HiT-JEPA. Then
we encode and predict the masked trajectory representations to simulate the training process, and
encode the full trajectory representations to simulate the inference process. Next, we concatenate and
tune a 2-layer MLP for each of the representations to decode to the hexagonal grid cell embeddings


8


to which they belong. We denote the decoded predicted masked trajectory representations as _S_ 1 and
the decoded encoded full trajectory representations as _S_ 2. Finally, for each trajectory position, we
search for the _k_ most similar embeddings in the spatial region embedding set _H_ and retrieve their
hexagonal cell IDs. We choose _k_ = 3 in our visualizations.


(a) Predicted masked points (b) Encoded full trajectory


Figure 3: Visualizations of decoded learned trajectory representations by HiT-JEPA on hexagonal
cells: (a) blue points are sampled trajectory points, gray points are masked trajectory points labeled
with ”target”, and orange hexagons are projected predictions. (b) blue points are full trajectory points,
green hexagons are projected encoded representations.


Origin attention


Stay point/pattern
change attention


Destination intent
attention


Figure 4: A case study of hierarchical semantic information captured by HiT-JEPA. **(Top-Left)** The
raw attention map visualizes the absolute attention weights, showing the overall intensity distribution.
**(Top-Mid)** The deviation heatmap by displaying areas of active focus (red) versus suppression (blue)
relative to the mean attention value. **(Top-Right)** The statistical profiles quantify the peak attention
intensity at each time step. **Bottom** The corresponding physical trajectory with index labeled every
10 steps, where colored boxes spatially ground the salient attention regions identified in the top row.


Fig. 3a shows the comparisons between decoded cells (orange hexagons) and masked points (gray
points) labeled as “targets”. The decoded locations lie in close proximity to their corresponding
masked targets, confirming that the model effectively learns accurate representations for masked
points during training. Fig. 3b overlays the decoded cells green hexagons) on each blue trajectory
point, demonstrating that the model can encode each point with even greater accuracy with access to
the full trajectory during inference.


4.3 INTERPRETATION OF HIERARCHICAL ATTENTION WEIGHTS.


9


Figure 5: Averaged attention weight visualizations at each JEPA layer. Left to right: _A_ [(3)] to _A_ [(1)] .


From Fig. 4, by corroborating the raw attention map, deviation heatmap, and statistical profiles, we
identify three distinct semantic phases localized within the bottom-left, top-right, and middle regions
around the 20 [th] trajectory point. These phases correspond to the peak intensity of attention allocated
to specific trajectory segments: the origin anchoring (red boxes), the local attention peak triggered
by a pattern change (orange boxes), and the destination intent (green boxes). This spatial-semantic
alignment confirms that HiT-JEPA successfully learns the critical semantic waypoints from raw GPS
tracks and verifies the interpretability of its learned representations. By comparing the raw attention
weights across 3 JEPA layers in Fig. 5, it is obvious to discern a coarse-to-fine attention evolution,
where the _A_ [3] layer highly summarizes the trajectory origin-destination patterns and is fused into lower
layers with more smoothed local details. This validates that HiT-JEPA learns consistent trajectory
semantics through the hierarchical interactions while preserving distinct layer-specific granularity.
We further visualize the attention map for each head in Appendices A.6.


4.4 ABLATION STUDY


can learn individually, the hierarchical interactions
bind different levels into a cohesive multi-scale structure.


5 CONCLUSION
In summary, HiT-JEPA introduces a unified three-layer hierarchy that captures point-level finegrained details, intermediate trajectory patterns, and high-level trajectory semantics within a single
self-supervised framework. By leveraging a Hierarchical JEPA, it enables a more powerful trajectory
feature extraction in the representation space and produces cohesive multi-granular embeddings.
Extensive evaluations on diverse urban and maritime trajectory datasets show that HiT-JEPA
outperforms single-scale self-supervised methods in trajectory similarity computation, especially
zero-shot generalization and downstream fine-tuning. These results validate its effectiveness and
robustness for real-world, large-scale trajectory modeling.


10


We study the effect of removing the key designs in
HiT-JEPA. We compare HiT-JEPA with 4 variants: 1) **HiT** ~~**e**~~ **mb** which replaces the hierarchical interaction method from attention upsampling to directly concatenate the upsampled encoder embeddings between _S_ _[′]_ [(] _[l]_ [)] and _S_ _[′]_ [(] _[l][−]_ [1)] . 2)
**HiT** ~~**s**~~ **ingle** ~~**l**~~ **ayer** where we only level _l_ = 1 to
train and predict. 3) **HiT** ~~**n**~~ **o** ~~**a**~~ **ttn** with no hierarchical interactions between each pair of successive
layers. 4) **HiT** **rect** with spatial location tokenization method changed to rectangular grid cells. We
train these variants and conduct self-similarity experiments on Porto.


Table 3 shows the comparisons between HiT-JEPA
and its variants. The performance drops without
any key designs, especially for HiT ~~e~~ mb, as directly concatenating the embedding from the previous layers causes representation collapse. Results
from the other two variants demonstrate that in our
model design, even though each layer of JEPA _[l]_


Table 3: Ablation Study of HiT-JEPA on Porto


**Varying DB Size** _|D|_


**Model** **20%** **40%** **60%** **80%** **100%**


HiT ~~e~~ mb 106.568 209.746 297.919 394.111 497.064
HiT ~~s~~ ingle ~~l~~ ayer 1.031 1.061 1.066 1.077 1.091
HiT ~~n~~ - ~~a~~ ttn 1.026 1.049 1.054 1.062 1.069
HiT ~~r~~ ect 1.032 1.062 1.069 1.080 1.093
HiT-JEPA **1.026** **1.043** **1.048** **1.058** **1.065**


**Downsampling Rate** _ρs_


**Model** **0.1** **0.2** **0.3** **0.4** **0.5**


HiT ~~e~~ mb 569.322 706.831 1004.246 2047.699 2171.331
HiT ~~s~~ ingle ~~l~~ ayer 1.378 2.659 5.626 14.123 **26.875**
HiT ~~n~~ - ~~a~~ ttn 1.405 2.867 5.761 17.143 27.324
HiT ~~r~~ ect 1.508 3.054 7.735 18.912 36.768
HiT-JEPA **1.369** **2.624** **5.541** **13.773** 28.806


**Distortion Rate** _ρd_


**Model** **0.1** **0.2** **0.3** **0.4** **0.5**


HiT ~~e~~ mb 502.259 503.876 506.333 507.738 507.082
HiT ~~s~~ ingle ~~l~~ ayer 1.088 1.099 1.120 1.100 1.137
HiT ~~n~~ - ~~a~~ ttn 1.079 1.095 1.105 1.093 1.120
HiT ~~r~~ ect 1.095 1.111 1.123 1.122 1.124
HiT-JEPA **1.074** **1.077** **1.085** **1.093** **1.119**


6 ETHICS STATEMENT


We claim that we adhere to the ICLR Code of Ethics. All the datasets used in the manuscript are
publicly available with no user information revealed. HiT-JEPA encodes the trajectory location
information in hexagonal cell tokens, where exact GPS traces are blurred. And such tokens are the
only input to our model, thereby preventing any leakage of precise location data. The code for all
baselines is publicly available and used under their respective licenses.


7 REPRODUCIBILITY STATEMENT


We provide an anonymous GitHub link [https://anonymous.4open.science/r/](https://anonymous.4open.science/r/HiT-JEPA)
[HiT-JEPA](https://anonymous.4open.science/r/HiT-JEPA) to prove that our work is reproducible. This repository contains the code for the
HiT-JEPA method implementation in Section 3 and any data processing and evaluation files in
Section 4. The details, such as dataset statistics A.1, implementation configurations A.3, and further
experiment details A.4, can also be found in the repository.


REFERENCES


Helmut Alt and Michael Godau. Computing the frechet distance between two polygonal curves.´
_International Journal of Computational Geometry & Applications_, 5(01n02):75–91, 1995.


Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat,
Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding
predictive architecture. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 15619–15629, 2023.


Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent
network for traffic forecasting. _Advances in neural information processing systems_, 33:17804–
17815, 2020.


Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization
for self-supervised learning. _arXiv preprint arXiv:2105.04906_, 2021.


Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mido
Assran, and Nicolas Ballas. V-jepa: Latent video prediction for visual representation learning.
2023.


Hanlin Cao, Haina Tang, Yulei Wu, Fei Wang, and Yongjun Xu. On accurate computation of trajectory
similarity via single image super-resolution. In _2021 International Joint Conference on Neural_
_Networks (IJCNN)_, pp. 1–9. IEEE, 2021.


Yanchuan Chang, Jianzhong Qi, Yuxuan Liang, and Egemen Tanin. Contrastive trajectory similarity learning with dual-feature attention. In _2023 IEEE 39th International conference on data_
_engineering (ICDE)_, pp. 2933–2945. IEEE, 2023.


Lei Chen and Raymond Ng. On the marriage of lp-norms and edit distance. In _Proceedings of the_
_Thirtieth international conference on Very large data bases-Volume 30_, pp. 792–803, 2004.


Lei Chen, M Tamer Ozsu, and Vincent Oria. [¨] Robust and fast similarity search for moving object
trajectories. In _Proceedings of the 2005 ACM SIGMOD international conference on Management_
_of data_, pp. 491–502, 2005.


Richard J Chen, Chengkuan Chen, Yicong Li, Tiffany Y Chen, Andrew D Trister, Rahul G Krishnan,
and Faisal Mahmood. Scaling vision transformers to gigapixel images via hierarchical selfsupervised learning. In _Proceedings of the IEEE/CVF conference on computer vision and pattern_
_recognition_, pp. 16144–16155, 2022a.


Yile Chen, Xiucheng Li, Gao Cong, Zhifeng Bao, Cheng Long, Yiding Liu, Arun Kumar Chandran,
and Richard Ellison. Robust road network representation learning: When traffic patterns meet
traveling semantics. In _Proceedings of the 30th ACM International Conference on Information &_
_Knowledge Management_, pp. 211–220, 2021.


11


Zebin Chen, Xiaolin Xiao, Yue-Jiao Gong, Jun Fang, Nan Ma, Hua Chai, and Zhiguang Cao.
Interpreting trajectories from multiple views: A hierarchical self-attention network for estimating
the time of arrival. In _Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery_
_and Data Mining_, pp. 2771–2779, 2022b.


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. In _Proceedings of the 2019 conference of_
_the North American chapter of the association for computational linguistics:_ _human language_
_technologies, volume 1 (long and short papers)_, pp. 4171–4186, 2019.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale. _arXiv_ _preprint_
_arXiv:2010.11929_, 2020.


Ziquan Fang, Yuntao Du, Lu Chen, Yujia Hu, Yunjun Gao, and Gang Chen. E 2 dtc: An end to end
deep trajectory clustering framework via self-training. In _2021 IEEE 37th International Conference_
_on Data Engineering (ICDE)_, pp. 696–707. IEEE, 2021.


Aditya Grover and Jure Leskovec. node2vec: Scalable feature learning for networks. In _Proceedings_
_of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining_, pp.
855–864, 2016.


Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar,´ and Ross Girshick. Masked
autoencoders are scalable vision learners. In _Proceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pp. 16000–16009, 2022.


Sepp Hochreiter and Jurgen¨ Schmidhuber. Long short-term memory. _Neural_ _computation_, 9(8):
1735–1780, 1997.


Lingjing Kong, Martin Q Ma, Guangyi Chen, Eric P Xing, Yuejie Chi, Louis-Philippe Morency,
and Kun Zhang. Understanding masked autoencoders via hierarchical latent variable models.
In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp.
7918–7928, 2023.


Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. _Open_
_Review_, 62(1):1–62, 2022.


Dongyang Li, Taolin Zhang, Nan Hu, Chengyu Wang, and Xiaofeng He. Hiclre: A hierarchical contrastive learning framework for distantly supervised relation extraction. _arXiv_ _preprint_
_arXiv:2202.13352_, 2022.


Jialiang Li, Tiantian Liu, and Hua Lu. Clear: Ranked multi-positive contrastive representation
learning for robust trajectory similarity computation. In _2024 25th IEEE International Conference_
_on Mobile Data Management (MDM)_, pp. 21–30. IEEE, 2024a.


Lihuan Li, Hao Xue, Yang Song, and Flora Salim. T-jepa: A joint-embedding predictive architecture
for trajectory similarity computation. In _Proceedings of the 32nd ACM International Conference_
_on Advances in Geographic Information Systems_, pp. 569–572, 2024b.


Xiucheng Li, Kaiqi Zhao, Gao Cong, Christian S Jensen, and Wei Wei. Deep representation
learning for trajectory similarity computation. In _2018 IEEE 34th international conference on data_
_engineering (ICDE)_, pp. 617–628. IEEE, 2018.


Yan Lin, Huaiyu Wan, Shengnan Guo, Jilin Hu, Christian S Jensen, and Youfang Lin. Pre-training
general trajectory embeddings with maximum multi-view entropy coding. _IEEE Transactions on_
_Knowledge and Data Engineering_, 36(12):9037–9050, 2023.


Xiang Liu, Xiaoying Tan, Yuchun Guo, Yishuai Chen, and Zhe Zhang. Cstrm: Contrastive selfsupervised trajectory representation model for trajectory similarity computation. _Computer Com-_
_munications_, 185:159–167, 2022.


Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. _arXiv preprint arXiv:1301.3781_, 2013.


12


Tangwen Qian, Junhe Li, Yile Chen, Gao Cong, Tao Sun, Fei Wang, and Yongjun Xu. Contextenhanced multi-view trajectory representation learning: Bridging the gap through self-supervised
models. _arXiv preprint arXiv:2410.13196_, 2024.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing_
_systems_, 30, 2017.


Chao Wang, Fangzheng Lyu, Sensen Wu, Yuanyuan Wang, Liuchang Xu, Feng Zhang, Shaowen
Wang, Yongheng Wang, and Zhenhong Du. A deep trajectory clustering method based on sequenceto-sequence autoencoder model. _Transactions in GIS_, 26(4):1801–1820, 2022.


Fanyi Xiao, Kaustav Kundu, Joseph Tighe, and Davide Modolo. Hierarchical self-supervised
representation learning for movie understanding. In _Proceedings of the IEEE/CVF conference on_
_computer vision and pattern recognition_, pp. 9727–9736, 2022.


Chuang Yang, Renhe Jiang, Xiaohang Xu, Chuan Xiao, and Kaoru Sezaki. Simformer: Single-layer
vanilla transformer can learn free-space trajectory similarity. _arXiv preprint arXiv:2410.14629_,
2024.


Dingqi Yang, Daqing Zhang, Vincent W Zheng, and Zhiyong Yu. Modeling user activity preference
by leveraging user spatial temporal characteristics in lbsns. _IEEE Transactions on Systems, Man,_
_and Cybernetics:_ _Systems_, 45(1):129–142, 2014.


Peilun Yang, Hanchen Wang, Ying Zhang, Lu Qin, Wenjie Zhang, and Xuemin Lin. T3s: Effective
representation learning for trajectory similarity computation. In _2021 IEEE 37th international_
_conference on data engineering (ICDE)_, pp. 2183–2188. IEEE, 2021.


Di Yao, Chao Zhang, Zhihua Zhu, Jianhui Huang, and Jingping Bi. Trajectory clustering via deep
representation learning. In _2017 international joint conference on neural networks (IJCNN)_, pp.
3880–3887. IEEE, 2017.


Di Yao, Gao Cong, Chao Zhang, and Jingping Bi. Computing trajectory similarity in linear time: A
generic seed-guided neural metric learning approach. In _2019 IEEE 35th international conference_
_on data engineering (ICDE)_, pp. 1358–1369. IEEE, 2019.


Di Yao, Jin Wang, Wenjie Chen, Fangda Guo, Peng Han, and Jingping Bi. Deep dirichlet process mixture model for non-parametric trajectory clustering. In _2024 IEEE 40th International Conference_
_on Data Engineering (ICDE)_, pp. 4449–4462. IEEE, 2024.


Byoung-Kee Yi, Hosagrahar V Jagadish, and Christos Faloutsos. Efficient retrieval of similar time
sequences under time warping. In _Proceedings 14th International Conference on Data Engineering_,
pp. 201–208. IEEE, 1998.


Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep
learning framework for traffic forecasting. _arXiv preprint arXiv:1709.04875_, 2017.


Jing Yuan, Yu Zheng, Chengyang Zhang, Wenlei Xie, Xing Xie, Guangzhong Sun, and Yan Huang.
T-drive: driving directions based on taxi trajectories. In _Proceedings of the 18th SIGSPATIAL_
_International conference on advances in geographic information systems_, pp. 99–108, 2010.


Jing Yuan, Yu Zheng, Xing Xie, and Guangzhong Sun. Driving with knowledge from the physical
world. In _Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery_
_and data mining_, pp. 316–324, 2011.


Xingxing Zhang, Furu Wei, and Ming Zhou. Hibert: Document level pre-training of hierarchical
bidirectional transformers for document summarization. _arXiv preprint arXiv:1905.06566_, 2019.


Yu Zheng, Quannan Li, Yukun Chen, Xing Xie, and Wei-Ying Ma. Understanding mobility based
on gps data. In _Proceedings of the 10th international conference on Ubiquitous computing_, pp.
312–321, 2008.


13


Yu Zheng, Lizhu Zhang, Xing Xie, and Wei-Ying Ma. Mining interesting locations and travel
sequences from gps trajectories. In _Proceedings of the 18th international conference on World_
_wide web_, pp. 791–800, 2009.


Yu Zheng, Xing Xie, Wei-Ying Ma, et al. Geolife: A collaborative social networking service among
user, location and trajectory. _IEEE Data Eng. Bull._, 33(2):32–39, 2010.


Yuanshao Zhu, James Jianqiao Yu, Xiangyu Zhao, Xuetao Wei, and Yuxuan Liang. Unitraj: Learning
a universal trajectory foundation model from billion-scale worldwide traces. _CoRR_, 2024.


14


A APPENDIX


A.1 DATASETS


Here we list the details of the datasets:


    - **Porto** includes 1.7 million trajectories from 442 taxis in Porto, Portugal. The dataset was
collected from July 2013 to June 2014.

    - **T-Drive** contains trajectories of 10,357 taxis in Beijing, China from Feb. 2 to Feb. 8, 2008.
The average sampling interval is 3.1 minutes.

    - **GeoLife** contains trajectories of 182 users in Beijing, China from April 2007 to August
2012. There are 17,6212 trajectories in total with most of them sampled in 1–5 seconds.

    - **Foursquare-TKY** is collected for 11 months from April 2012 to February 2013 in Tokyo,
Japan, with 573,703 check-ins in total.

    - **Foursquare-NYC** is collected for 11 months from April 2012 to February 2013 in New
York City, USA, with 227,428 check-ins in total.

    - **AIS(AU)** comprises vessel traffic records collected by the Craft Tracking System (CTS) of
Australia. In this paper, we use vessel trajectories in February 2025.


Table 4: Statistics of Datasets after preprocessing.


|Data type|Dataset|#points #trajectories|
|---|---|---|
|Urban trajectories|Porto<br>T-Drive<br>GeoLife|65,913,828<br>1,372,725<br>5,579,067<br>101,842<br>8,987,488<br>50,693|
|Check-in sequences|TKY<br>NYC|106,480<br>3,048<br>28,858<br>734|
|Vessel trajectories|AIS(AU)|485,424<br>7,095|


We first keep trajectories in urban areas with the number of points ranging from 20 to 200, where
the statistics of the datasets after preprocessing are shown in Table 4. We use 200,000 trajectories
for Porto, 70,000 for T-Drive, and 35000 for GeoLife as training sets. Each dataset has 10% of
data used for validation. As there are many fewer trajectories in TKY, NYC, and AIS(AU), we use
all trajectories in these datasets for testing. For the testing set, we select 100,000 trajectories for
Porto, 10,000 for T-Drive and GeoLife, 3000 for TKY, 700 for NYC, and 7000 for AIS(AU). For
the downstream fine-tuning task, we select 10,000 trajectories for Porto and T-Drive, and 5000 for
GeoLife, where the selected trajectories are split by 7:1:2 for training, validation, and testing. We
train Hit-JEPA and all baselines from scratch for Porto, T-Drive, and GeoLife datasets. Then, we
load the pre-trained weights from Porto and conduct zero-shot self-similarity experiments on each of
the TKY, NYC, and AIS(AU) to evaluate the generalization ability of all models.


A.2 BASELINES


We compare HiT-JEPA with four most recent self-supervised free space trajectory similarity
computation methods: TrajCL Chang et al. (2023), TrjSR Cao et al. (2021), CLEAR Li et al. (2024a),
and T-JEPA Li et al. (2024b). TrajCL is a contrastive learning method that adopts a dual-feature
attention module to capture the trajectory details, which has achieved impactful performance on
trajectory similarity computation in multiple datasets and experimental settings. TrjSR is a generative
model that converts trajectories into gray-scale images. This method reconstructs the high-resolution
trajectory image from the low-resolution image by leveraging single-image super-resolution to
learn better spatial trajectory representations. CLEAR improves the contrastive learning process
by ranking the positive trajectory samples based on their similarities to anchor samples, capturing
detailed differences from similar trajectories. T-JEPA is the most recent method utilizing Joint
Embedding Predictive Architecture to encode and predict trajectory information in the representation
space, which effectively captures necessary trajectory information. We run these two models from
their open-source code repositories with default parameters.


15


A.3 IMPLEMENTATION DETAILS


We use Adam Optimizer for training and optimizing the model parameters across all levels, except
for the target encoders. The target encoder at each level _l_ updates its parameters via the exponential
moving average of the parameters of the context encoder at the same level. The maximum number
of training epochs is 20, and the learning rate is 0.0001, decaying by half every 5 epochs. The
embedding dimension _d_ is 256, and the batch size is 64. We apply 1-layer Transformer Encoders for
both context and target encoders at each level, with the number of attention heads set to 8 and hidden
layer dimension to 1024. We use a 1-layer Transformer Decoder as the predictor at each level _l_ with
the number of attention heads set to 8. We use learnable positional encoding for all the encoders and
decoders. We set the resampling masking ratio to be selected from _r_ = _{_ 10% _,_ 15% _,_ 20% _,_ 25% _,_ 30% _}_
and the number of sampled targets _M_ to 4 for each trajectory at each model level _l_ . The successive
sampling probability _p_ is set to 50%, and the initial context sampling ratio _pγ_ is set to range from
85% to 100%. The scale factors for the final loss are _λ_ = 0 _._ 05, _µ_ = 0 _._ 15, and _ν_ = 0 _._ 8. We use a
hexagonal cell resolution of 11 for Porto, resolution 10 for T-Drive, GeoLife, TKY, and NYC, and
resolution 4 for AIS(AU). All experiments are conducted on servers with Nvidia A5000 GPUs, 24GB
of memory, and 250GB of RAM.


A.4 EXPERIMENTAL SETTINGS


A.4.1 SELF-SIMILARITY


For each query trajectory _q_ _∈_ _Q_, we create two sub-trajectories _qa_ = _{p_ 1 _, p_ 3 _, p_ 5 _, . . .}_ containing
the odd-indexed points and _qb_ = _{p_ 2 _, p_ 4 _, p_ 6 _, . . .}_ even-indexed points of _q_ . We separate them by
putting _qa_ into the query set _Q_ and putting _qb_ into the database _D_, with the rest of the trajectories
in _D_ randomly filled from the testing set. Each _qa_ and _qb_ pair exhibits similar overall patterns in
terms of shape, length, and sampling rate. We apply HiT-JEPA context-encoders to both query and
database trajectories, compute pairwise similarities, and sort the results in descending order. Next,
we report the mean rank of each _qb_ when retrieved by its corresponding query _qa_ : ideally, the true
match appears at rank one. We choose _{_ 20% _,_ 40% _._ 60% _,_ 80% _,_ 100% _}_ of the total database size _|D|_
for evaluation. To further evaluate the robustness of learned trajectory representations, we also apply
down-sampling and distortion on _Q_ and _D_ . Specifically, we randomly mask points (with start and
end points kept) with down-sampling probability _ρs_ and shift the point coordinates with distortion
probability _ρd_ . Both _ρs_ and _ρd_ represent the number of points to be down-sampled or distorted,
ranging from _{_ 0 _._ 1 _,_ 0 _._ 2 _,_ 0 _._ 3 _,_ 0 _._ 4 _,_ 0 _._ 5 _}_ .


For the convenience of comparing results under these settings together, we denote meta ratio
_Ri_ = _{|D|i, ρsi, ρdi}_ and compare the **mean rank** of all models at each _Ri_ on each dataset, smaller
values are better.


A.5 HYPERPARAMETER ANALYSIS


We analyze the impact of two sets of hyperparameters with the implementation and experimental
settings in the Appendices section A.3 and A.4.


**Number** **of** **attention** **layers** **at** **each** **abstraction** **level.** We change the number of Transformer
encoder layers for each level to 2 and 3, then compare them with the default setting (1 layer) for
self-similarity search with varying _|D|_, _ρs_ and _ρd_ on Porto. From Fig. 6, we can find that with only 1
attention layer, we can achieve the lowest mean ranks for all settings. This is due to higher chances
of overfitting with more attention layers.


(a) DB size (20% _∼_ 100%) (b) Downsampling rate (0.1 _∼_ 0.5) (c) Distortion rate(0.1 _∼_ 0.5)


Figure 6: Effect of different numbers of attention layers at each abstraction level.

16


(a) DB size (20% _∼_ 100%) (b) Downsampling rate (0.1 _∼_ 0.5) (c) Distortion rate(0.1 _∼_ 0.5)


Figure 7: Effect of different batch sizes.
**Weighting coefficient for the multi-level loss** _L_ **.** The values of the loss weighting coefficients _λ_, _µ_,
and _ν_ are carefully tuned. In Table 5, 6, and 7, we compare our selected coefficients with other 3 sets
of parameters in a wide range on the Porto dataset. From the tables, we can see that HiT-JEPA is
robust against various loss combinations. Even though the loss coefficients with _λ_, _µ_, and _ν_ equal to
0.33, 0.33, and 0.33 perform better on the downsampling experiment, our selected combination with
_λ_ = 0 _._ 05, _µ_ = 0 _._ 15, and _ν_ = 0 _._ 8 still learns overall the most accurate, stable, and consistent results
across all experimental settings.


_λ_ _µ_ _ν_ 20% 40% 60% 80% 100%


0.1 0.2 0.7 1.026 1.050 1.056 1.067 1.079
0.33 0.33 0.33 1.036 1.072 1.080 1.102 1.120

0.6 0.3 0.1 1.035 1.063 1.066 1.079 1.099
**0.05** **0.15** **0.8** **1.026** **1.043** **1.048** **1.058** **1.065**


Table 5: Loss weighting coefficients for varying DB sizes _|D|_ .


_λ_ _µ_ _ν_ 0.1 0.2 0.3 0.4 0.5


0.1 0.2 0.7 **1.334** 2.844 5.868 13.864 25.009
0.33 0.33 0.33 1.393 2.664 **4.616** **11.210** **20.730**

0.6 0.3 0.1 1.449 2.763 5.629 14.104 23.985
**0.05** **0.15** **0.8** 1.369 **2.624** 5.541 13.773 28.806


Table 6: Loss weighting coefficients for varying downsampling rates _ρs_ .


A.6 ATTENTION HEADS VISUALIZATIONS


The attention weights in Fig. 8 demonstrate functional specialization among attention heads. For
example, Head 3 focuses on local kinematics, while Heads 2, 4, 6, and 8 act as global anchors that
attend to long-term trajectory semantics. This diversity ensures a comprehensive representation that
integrates fine-grained motion dynamics with high-level trip intent.


A.7 TRAINING EFFICIENCY


We compare HiT-JEPA with baselines in terms of training time per iteration in Table 8. While TrajCL
and CLEAR achieve lower training time due to their lightweight structures, HiT-JEPA remains highly
competitive at rank 3 among 5 methods. Moreover, by incorporating convolution-based trajectory
semantics aggregation and learning on multi-level trajectory abstractions with 1-layer Transformer
backbones, HiT-JEPA remains efficient while achieving generalizable and robust performance.


A.8 VISUALIZATIONS


We visualize two sets of comparisons of 5-NN queries after fine-tuning by Hausdorff measure
in Fig. 12b and Fig. 13b, where each row shows the rank 1 to 5 matched trajectories from left to
right, given red query trajectories. The rightmost figures are the indices of the query and matched


17


_λ_ _µ_ _ν_ 0.1 0.2 0.3 0.4 0.5


0.1 0.2 0.7 1.081 1.109 1.095 1.115 1.135
0.33 0.33 0.33 1.130 1.138 1.133 1.152 1.196

0.6 0.3 0.1 1.092 1.107 1.133 1.117 1.191
**0.05** **0.15** **0.8** **1.074** **1.077** **1.085** **1.093** **1.119**


Table 7: Loss weighting coefficients for varying distortion rates _ρd_ .


Table 8: Comparison of training efficiency (seconds per iteration).


Method Time (s)


TrajCL 0.196
TrjSR 0.476
CLEAR 0.292
T-JEPA 1.022
HiT-JEPA 0.341


trajectories. We can find that the improvements of HiT-JEPA can find more similar trajectories on
ranks 4 and 5, resulting in a higher average HR@5 than T-JEPA.


Figure 8: Visualization of each of the 8 attention heads at the JEPA layer _A_ [(1)] .


A.9 REPRESENTATION VISUALIZATION VIA CLUSTERING


We cluster and visualize the embedding of 3000 random trajectories in Porto in Fig. 9. We use a
K-Means Clusterer with a number of cluster centers _K_ = 6 acquired from the Elbow Method. We
can find that, although the boundaries between clusters are soft without a specific self-clustering
design in recent clustering methods Yao et al. (2017); Fang et al. (2021), distinct semantic groups are
clearly visually discernible. This demonstrates the strong potential of HiT-JEPA, which is trained on
regression loss, to be fine-tuned to generalize to multiple trajectory tasks.


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


Figure 9: t-SNE Visualization of Trajectories.


A.10 LIMITATIONS AND FUTURE WORK


By upsampling and fusing attention weights across adjacent layers, HiT-JEPA demonstrates one form
of hierarchical interaction common to Transformer-based JEPA models. Therefore, one extension
could be developing a unified hierarchical interaction framework for all kinds of learning architectures
(e.g., CNNs, Mambas, LSTMs, etc.). This will enable each architecture to plug in its customized
hierarchy module while preserving a consistent multi-level learning paradigm.


Figure 10: Visualization of predicted masked trajectories.


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


Figure 11: Visualization of encoded full trajectories.


(a) T-JEPA Visualizations


(b) HiT-JEPA Visualizations


Figure 12: Comparisons of 5-NN search between T-JEPA and HiT-JEPA on Porto after being finetuned by Hausdorff measure.


(a) T-JEPA Visualizations


(b) HiT-JEPA Visualizations


Figure 13: Comparisons of 5-NN search between T-JEPA and HiT-JEPA on GeoLife after being
fine-tuned by Hausdorff measure.


20
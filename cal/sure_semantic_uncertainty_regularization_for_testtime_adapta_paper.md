# SURE: SEMANTIC UNCERTAINTY REGULARIZATION FOR TEST-TIME ADAPTATION IN VISION-LANGUAGE MODELS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Test-time adaptation (TTA) aims to improve model robustness under distribution
shift by exploiting unlabeled test data. Existing methods often rely on pseudolabels, which are noisy and treated independently, ignoring both their temporal
reliability and the semantic structure of the label space. We introduce SURE
(Semantic Uncertainty REgularization), a framework that regularizes predictions
through a dynamically evolving prototype-reliability graph (PRG). PRG captures
semantic affinity across classes and the stability of confidence over time, enabling
the selective propagation of reliable predictions while suppressing errors. This
structure-driven regularization enforces semantic consistency and prevents error
amplification. Across diverse domain-shift benchmarks, SURE consistently outperforms prior methods, offering a principled and generalizable approach to reliable TTA.


1 INTRODUCTION


Vision-language models (VLMs) such as CLIP (Radford et al., 2021) and ALIGN (Jia et al., 2021)
achieve strong zero-shot transfer by aligning images and text in a shared embedding space. Yet,
their performance degrades under distribution shifts— _e.g._, changes in appearance, context, or class
priors—resulting in uncertainty and semantic misalignment. Compared to unimodal vision models,
this degradation is particularly severe for VLMs, since distribution shifts may not only distort visual
features but also disrupt their alignment with textual prototypes. This motivates test-time adaptation
(TTA) for robust deployment.


Recent work has explored TTA for VLMs without labeled supervision or source data (Shu et al.,
2022; Zanella & Ben Ayed, 2024; Ma et al., 2024; Yoon et al., 2024). Entropy-based methods (Shu
et al., 2022; Sui et al., 2024; Zhang et al., 2024a) adapt by minimizing prediction entropy over augmented views, while prototype-based methods (Zanella & Ben Ayed, 2024; Zhou et al., 2025) refine
class prototypes guided by text embeddings. Despite different formulations, both families rely heavily on model predictions, making them vulnerable to noisy pseudo-labels under distribution shift. In
practice, these approaches suffer from distinct failure modes: entropy minimization can render the
model overconfident in incorrect predictions, whereas early prototype updates may propagate noise
and destabilize adaptation.


Confidence thresholding used in Shu et al. (2022) partly alleviates noise but discards informative
low-confidence samples. We argue that adaptation should instead exploit the evolving reliability
of class-level predictions, reflecting the model’s accumulated certainty about each category. Rather
than treating adaptation as isolated prediction correction, we advocate a structured paradigm where
adaptation emerges from the interaction between evolving predictions and semantic priors. To this
end, we formulate adaptation as a structured process that propagates reliable information while
suppressing semantic noise, laying the foundation for our proposed method.


As shown in Fig. 1, we move beyond instance-level heuristics and propose a structured formulation
of test-time adaptation. The key idea is to regularize predictions through an evolving PrototypeReliability Graph (PRG), where edges encode (i) semantic affinity derived from textual prototypes
and (ii) class-wise reliability measured by the temporal stability of pseudo-label confidences. PRG
acts as a semantic regularizer: it propagates support from confident and semantically related classes


1


**Prototype Update**


|Cat|Dog|Fox|Tiger|
|---|---|---|---|
|0.35|0.32|0.21|0.12|


|Cat|Dog|Fox|Tiger|
|---|---|---|---|
|0.52|0.21|0.15|0.08|


Figure 1: Conceptual diagram of SURE. At test time, raw predictions are refined through a
prototype-reliability graph, where edges encode both semantic similarity and class-wise reliability.
As the input stream arrives, the graph is dynamically updated via prototype and reliability updates,
aligning predictions with evolving semantic priors. This structured propagation emphasizes support
from semantically related and statistically stable neighbors while suppressing noisy signals, thereby
enforcing semantic consistency and mitigating error amplification under distribution shift.


to uncertain ones, while suppressing unreliable associations. Concretely, logits are updated via
graph-based propagation, aligning outputs with structure-induced consistency constraints. We instantiate this principle as SURE (Semantic Uncertainty REgularization), a closed-loop framework
in which predictions, prototypes, and graph structure co-evolve, thus stabilizing adaptation and mitigating semantic drift under distribution shift.


    - We propose SURE, a test-time adaptation framework that regularizes prediction dynamics
via a Prototype-Reliability Graph, jointly modeling semantic affinity and class-wise reliability.


    - We design a closed-loop adaptation mechanism where predictions and graph structure coevolve, realized through prototype refinement, reliability update, and graph-based logit
propagation.


    - We demonstrate the effectiveness and generality of SURE across four domain-shift benchmarks and two VLM backbones, achieving consistent robustness gains and state-of-the-art
performance.


2 RELATED WORK


**Adapting vision-language models.** Vision-language models such as CLIP (Radford et al., 2021),
ALIGN (Jia et al., 2021), and BLIP (Li et al., 2022) achieve strong zero-shot transfer through
large-scale contrastive pretraining. To further improve downstream performance, parameter-efficient
adaptation methods have been proposed. Prompt-learning approaches such as CoOp (Zhou et al.,
2022b) and CoCoOp (Zhou et al., 2022a) optimize soft prompts, while adapter-based methods (Zhang et al., 2022b; 2023) insert lightweight modules into frozen backbones. Structure-aware
adapters like GraphAdapter (Li & Jiang, 2025) further introduce graph neural networks to model
class correlations within the adapter framework, enhancing the semantic consistency of learned features. Other variants such as LoRA (Hu et al., 2022) and fine-tuning extensions have also been
explored. While effective, these approaches typically require labeled supervision and offline training, making them unsuitable for deployment-time adaptation under distribution shift.


**General-purpose test-time adaptation.** Standard TTA aims to adapt a pre-trained model to unlabeled test streams via self-supervised objectives. Pioneering works like TENT (Wang et al., 2020)
update batch normalization parameters by minimizing prediction entropy. Schneider et al. (Schneider et al., 2020b) demonstrated that simply replacing training-time batch statistics with test-time
statistics significantly improves robustness against corruptions. T3A (Iwasawa & Matsuo, 2021) adjusts the classifier prototype using pseudo-labeled samples, while SAR (Niu et al., 2022) introduces
sharpness-aware minimization to select reliable samples. Other approaches focus on distribution
alignment (Liang et al., 2020), conjugate pseudo-labels (Goyal et al., 2022), or robustifying normalization statistics (Zhang et al., 2022a; Schneider et al., 2020a). However, these methods generally
operate on uni-modal (vision-only) architectures. Directly applying them to VLMs is often suboptimal because they fail to exploit the rich semantic prior encapsulated in the pre-trained text encoder.


2


Input Image


**Semantic Uncertainty Regularization**


**Test-time adaptation of VLMs.** TTA offers a label-free paradigm for adapting VLMs at inference
time using only unlabeled inputs. Existing methods can be broadly grouped into: (i) _entropy-based_
approaches (Shu et al., 2022; Sui et al., 2024; Imam et al., 2025; Yoon et al., 2024; Sheng et al., 2025)
that minimize prediction entropy to calibrate features; (ii) _prototype-based_ approaches (Zanella &
Ben Ayed, 2024; Zhang et al., 2024a;b; Zhou et al., 2025) that build or refine prototypes guided
by text embeddings; (iii) _ensemble and augmentation_ methods such as ZERO (Farina et al., 2024);
and (iv) _optimization-based_ _extensions_ including reward-driven (Zhao et al., 2023), MAP estimation (Fuchs et al., 2025), and retrieval augmentation (Lee et al., 2025). Most of these methods treat
classes independently and rely on per-instance confidence, overlooking inter-class structure. While
we also target VLM adaptation, we diverge from these works by explicitly modeling the topological
relationships between classes to rectify unreliable predictions.


**Graph-based** **reasoning** **and** **uncertainty** **modeling.** Graph-based reasoning captures structured
dependencies across entities for tasks like scene understanding and relational prediction (Kipf, 2016;
Veliˇckovi´c et al., 2017). In the context of adaptation, PROGRAM (Sun et al., 2024) recently proposed a prototype graph model to propagate pseudo-labels between prototypes and test samples,
utilizing message passing to improve pseudo-label quality. While sharing the graph-based spirit,
our approach differs from PROGRAM in two key aspects: (1) **Reliability-driven topology** : Instead
of a static or purely feature-distance-based graph, we dynamically modulate edges using class-wise
uncertainty statistics (reliability), ensuring that noise does not propagate through the graph. (2)
**VLM-specific** **design** : PROGRAM is a general TTA method designed for uni-modal classifiers,
whereas our framework is tailored for VLMs, leveraging the frozen text encoder to initialize semantically meaningful graphs. More recently, uncertainty has been integrated into graph learning to
yield calibrated inference (Ni et al., 2025; Huang et al., 2023; Han et al., 2025). Unlike these efforts,
our approach dynamically constructs a class-level graph during test-time adaptation, coupling semantic similarity with reliability estimation to regularize predictions in a lightweight and principled
manner.


3 PRELIMINARIES


**CLIP.** CLIP (Radford et al., 2021) is a vision-language model pretrained on large-scale image-text
pairs via a contrastive objective that aligns visual and textual embeddings. It consists of an image
encoder _EI_ ( _·_ ) and a text encoder _ET_ ( _·_ ). Given an input image **x**, the image feature is **f** = _EI_ ( **x** ).
Textual prompts such as “a photo of a _{_ class name _}_ ” are encoded into class prototypes **T** = _{_ **t** _i}_ _[C]_ _i_ =1 [,]
where **t** _i_ = _ET_ (“a photo of a _{_ class name _}_ ”) and _C_ is the number of classes. Classification is then
performed via cosine similarity followed by a softmax:


exp(cos( **f** _,_ **t** _i_ ) _/τ_ )
_p_ ( _yi|_ **x** ) =             - _C_ _,_ (1)
_j_ =1 [exp(cos(] **[f]** _[,]_ **[ t]** _[j]_ [)] _[/τ]_ [)]


where cos( **f** _,_ **t** _i_ ) = _∥_ **ff** _∥∥_ _[⊤]_ **tt** _ii∥_ [denotes cosine similarity, and] _[ τ]_ _[>]_ [ 0][ is a temperature.]


**Likelihood Adaptation.** A prominent line of TTA methods adapts CLIP by updating class prototypes according to the test distribution, thereby refining the likelihood _p_ ( _yi|_ **x** ). Zhou et al. (2025)
identify such likelihood-level adaptation as a key mechanism. Specifically, when the predicted class
_yi_ has confidence _p_ ( _yi|_ **x** ) above a threshold _θ_, its prototype **t** _i_ is updated using a normalized running
average with the image feature **f** . A counter _Ni_ tracks the number of updates per class:


where Norm( _·_ ) denotes L2 normalization to keep prototypes on the unit sphere. This continual
refinement gradually aligns prototypes with the evolving test distribution, improving zero-shot robustness. However, it also inherits a critical limitation: early updates based on noisy pseudo-labels
may accumulate errors, motivating the need for more structured reliability modeling.


3


    - _Ni ·_ **t** _i_ + **f**
**t** _i_ _←_ Norm
_Ni_ + 1


_,_ _Ni_ _←_ _Ni_ + 1 _,_ (2)


where _σ_ max is a fixed hyperparameter constant (set to 0.5 in our experiments) to ensure stable normalization, independent of batch statistics. In implementation, _Rj_ is clipped to [0 _,_ 1] for numerical


4


Figure 2: Overview of the proposed test-time adaptation framework SURE. The process comprises
three stages: (1) Embedding: Given a test image **x**, we use a frozen CLIP model to extract its visual
embedding and compute initial predictions via cosine similarity to class-specific textual prototypes.
(2) Confidence-aware logit regularization: A cached PRG is retrieved, constructed from a semantic
similarity matrix **S** (computed from textual embeddings) and a joint reliability matrix **R** joint (derived
from historical pseudo-label confidence statistics). The PRG is then used to regularize logits and
output updated predictions. (3) Graph evolution: Based on updated predictions, we update class
prototypes, reliability estimates, and the PRG structure itself via temporal smoothing. This forms a
closed-loop semantic regularization process under distribution shift. .


4 METHOD


4.1 PROTOTYPE-RELIABILITY GRAPH


We propose the Prototype-Reliability Graph (PRG), a dynamically constructed structure that captures robust inter-class relations under distribution shift. Instead of assuming static or uniform similarity, PRG incorporates both semantic affinity and uncertainty-aware reliability to model a classlevel graph that reflects the evolving confidence landscape of the target domain.

At each test-time step _ℓ_, PRG is instantiated based on the current class prototypes _{_ **t** _i}_ _[C]_ _i_ =1 [and]
pseudo-label confidence statistics. Formally, it can be viewed as a semantic graph whose adjacency
structure is modulated by information-theoretic Shannon (1948); Huang et al. (2023) estimates of
prediction uncertainty, thereby regularizing connections in favor of reliable, low-entropy class associations. Crucially, unlike standard self-training methods that trust instantaneous high-confidence
predictions—which are prone to calibration errors under shift—SURE leverages the temporal statistics of predictions. By computing the variance over a sliding window, we filter out classes that
oscillate in confidence, ensuring that only statistically stable predictions guide the adaptation.


**Prototype similarity matrix S.** We compute a semantic similarity matrix **S** _∈_ R _[C][×][C]_ that encodes
pairwise affinities between class prototypes:


**S** _jk_ = cos( **t** _j,_ **t** _k_ ) _,_ (3)


where cos( _·, ·_ ) denotes cosine similarity. In practice, we set **S** _jj_ = 0 to avoid trivial self-loops. This
matrix captures semantic proximity in the embedding space, but remains sensitive to noise when
prototypes are distorted by incorrect pseudo-labels under distribution shift.


**Class-wise** **reliability** **scores** _Rj_ **.** We modulate the similarity graph using a class-wise reliability
score _Rj_, which quantifies the prediction stability associated with class _yj_ . Specifically, we maintain
two descriptive statistics in a cache: the mean pseudo-label confidence _µj_ _∈_ [0 _,_ 1] and its standard
deviation _σj_, which jointly characterize the historical uncertainty of predictions assigned to class
_yj_ . These statistics are updated online as new pseudo-labels arrive. At the beginning of test-time
adaptation, we initialize _µj_ = 1 _._ 0 and _σj_ = 0 _._ 0 for all classes to reflect maximal confidence and no
variance. The reliability score _Rj_ is then computed as:


   - _σj_
_Rj_ = _µj_ _·_ 1 _−_
_σ_ max


_,_ (4)


stability. This formulation reflects an information-theoretic intuition: reliable classes exhibit lowentropy confidence distributions (high _µj_, low _σj_ ), while noisy or ambiguous classes are associated
with high entropy. Although _Rj_ is not a direct entropy measure, it serves as a practical proxy for
inverse uncertainty and thus quantifies each class’s susceptibility to semantic drift.


We next integrate **S** and _{Rj}_ into a unified adjacency structure that governs graph-based propagation. Specifically, we construct **R** joint and then **W** as follows.


**Joint** **reliability** **matrix** **Rjoint.** To suppress unreliable semantic interactions, we compute a joint
reliability score between class pairs:


**R** joint _,jk_ = _Rj_ _· Rk._ (5)


This multiplicative form ensures that the trustworthiness of an edge depends on both endpoints: if
either class is unstable, the resulting connection is weakened. In this way, **R** joint acts as an edge-wise
gating mechanism that penalizes unreliable prototypes and prevents fragile classes from dominating
the graph. This design provides the basis for integrating reliability into the subsequent semantic
similarity structure.


**Reliability-weighted** **matrix** **W.** We then construct a reliability-aware similarity matrix by combining semantic affinity with joint reliability:


**W** = **S** _⊙_ **R** joint _,_ **W** _jj_ = 0 _,_ (6)


where _⊙_ denotes element-wise multiplication and self-connections are set to zero. This formulation
downweights unreliable or ambiguous prototype associations, ensuring that unstable classes cannot
reinforce themselves. Intuitively, even if a corrupted prototype **t** _j_ appears close to **t** _k_ due to semantic
drift, a low reliability score _Rj_ suppresses the edge **W** _jk_ . The resulting **W** defines a reliabilityweighted semantic graph, where edges reflect both semantic proximity and prediction stability.


**Construction** **of** **sparse** **adjacency** **matrix** **A.** To obtain a sparse and tractable PRG, we identify
the top- _k_ neighbors _Tj_ _[k]_ [for] [each] [class] _[y][j]_ [based] [on] [the] [reliability-weighted] [similarities] [in] **[W]** [.] [We]
retain only these top- _k_ connections and normalize them to define the directed adjacency matrix at
test step _ℓ_ :


yielding **A** [(] _[ℓ]_ [)] _∈_ R _[C][×][C]_ as a sparse, directed, and weighted graph. This structure fuses semantic
similarity with class-wise reliability and forms the structural backbone for subsequent propagation
and prediction refinement.


**Graph Formalization.** With the structural backbone **A** [(] _[ℓ]_ [)] established, we formally define our PRG
framework as a dynamic system _G_ [(] _[ℓ]_ [)] = ( _V, E_ [(] _[ℓ]_ [)] _, M, U_ ):


    - **Nodes** _V_ **:** The set of class prototypes _{_ **t** 1 _, . . .,_ **t** _C}_ . Each node carries a state vector **h** _i_ =

[ **t** _i, µi, σi_ ] comprising the semantic prototype and its reliability statistics (Eq. 4).


    - **Edges** _E_ [(] _[ℓ]_ [)] **:** The directed connections defined by the sparse adjacency **A** [(] _[ℓ]_ [)] . An edge
( _j_ _→_ _i_ ) represents a reliability-weighted semantic dependency derived from Eq. 6.


    - **Messages** _M_ **(Inference):** The belief propagation mechanism described in Sec. 4.2, where
node _j_ transmits prediction evidence to node _i_ weighted by edge strength.


    - **Update Rules** _U_ **(Evolution):** The temporal dynamics described in Sec. 4.3, which update
the node states ( **h** _i_ ) and consequently the edge topology ( _E_ [(] _[ℓ]_ [+1)] ) for the next step.


This formalization highlights that our method is not merely a static graph regularization, but a
closed-loop dynamic system.


4.2 CONFIDENCE-AWARE LOGIT REGULARIZATION


Building on the reliability-aware graph construction in Sec. 4.1, this module leverages class-wise
confidence _Rj_ to inject reliability into the adjacency structure. In this sense, the logit regularization


5


**W** _jk_


**A** [(] _jk_ _[ℓ]_ [)] [=]






_yk′ ∈Tj_ _[k]_ (7)

0 _,_ otherwise _,_


**W** _jk′_ _[,]_ _yk_ _∈Tj_ _[k][,]_


_yk′ ∈Tj_ _[k]_





is confidence-aware: predictions are not only smoothed by semantic neighbors but also weighted by
their reliability.


To stabilize the discrete top- _k_ edge selection, we apply a sliding window average over the recent _L_
adjacency matrices. At test-time step _ℓ_, the smoothed adjacency is


1
_µi_ = _|Qi|_


- _c_ [(] _i_ _[n]_ [)] _−_ _µi_ �2 _._ (13)


**A** ¯ [(] _[ℓ]_ [)] = [1]

_L_


**A** ¯ [(] _[ℓ]_ [)] = [1]


_L−_ 1

- **A** [(] _[ℓ][−][i]_ [)] _,_ (8)


_i_ =0


where **A** [(] _[ℓ][−][i]_ [)] is the adjacency at step _ℓ_ _−_ _i_ . The buffer is updated online by enqueuing the latest matrix and discarding the oldest once its size exceeds _L_ . This temporal smoothing avoids introducing
an additional coefficient—only the window size _L_ is required—and makes the graph less sensitive
to spurious pseudo-labels or abrupt prototype shifts.

Given a test input **x**, we refine its raw prediction scores _{p_ ( _yi|_ **x** ) _}_ _[C]_ _i_ =1 [by propagating them over the]
smoothed graph, we aggregate incoming messages to _yi_ as:


_p_ graph( _yi|_ **x** ) =


_yj_ _∈Ti_ _[k]_ **[A]** [¯] _j,i_ [(] _[ℓ]_ [)] _[p]_ [(] _[y][j][|]_ **[x]** [)]

 - [¯] [(] _[ℓ]_ [)]


_i_ _,_ (9)

_yj_ _∈Ti_ _[k]_ **[A]** [¯] _j,i_ [(] _[ℓ]_ [)]


where **A** [¯] [(] _j,i_ _[ℓ]_ [)] [denotes] [the] [normalized] [weight] [of] [the] [edge] [from] [class] _[y][j]_ [to] _[y][i]_ [(row-normalized] [on] _[j]_ [).]
Since the sliding-window averaging can slightly break exact row-normalization, the denominator in
_p_ graph re-normalizes the incoming weights. We then combine local and graph-based scores as


_p_ ( _yi|_ **x** ) + _p_ graph( _yi|_ **x** )
_p_ ˆ( _yi|_ **x** ) =


_._ (10)

_C_

- ( _p_ ( _ym|_ **x** ) + _p_ graph( _ym|_ **x** ))

_m_ =1


_C_


This process can be interpreted as one-step belief propagation in a class-level Markov random field,
where local classifier outputs serve as node evidence and **A** [¯] [(] _[ℓ]_ [)] defines edge potentials. By coupling each class prediction with semantically and statistically reliable neighbors, the model reduces
variance, suppresses noise, and mitigates semantic drift. The final pseudo-label is assigned as

_y_ _[∗]_ ( **x** ) = arg max _p_ ˆ( _yj|_ **x** ) _._ (11)
_j_


4.3 GRAPH EVOLUTION


At each test-time step, SURE refines its predictions using the current PRG, while in turn updating
the graph based on reliable pseudo-labels. This feedback loop enables mutual calibration: the graph
constrains prediction dynamics via semantic uncertainty regularization, and reliable predictions reinforce graph consistency by refining class prototypes and local connectivity. These components
form a dynamic regularization system that anchors adaptation to semantically trustworthy regions
of the target domain.


**Prototype** **and** **reliability** **update.** Given a high-confidence pseudo-label _y_ _[∗]_ ( **x** ) ( _i.e._, prediction
confidence _>_ _θ_ ), we update the corresponding class prototype with a normalized moving average
following Eq. 2. The update is tracked by a class-specific counter _Ni_ [proto] :


**t** _i_ _←_ Norm� _NNi_ proto _i_ [proto] _·_ **t** + 1 _i_ + **f**


_,_ _Ni_ [proto] _←_ _Ni_ [proto] + 1 _._ (12)


To assess the reliability of class _i_, we maintain a fixed-size sliding window buffer _Qi_ =
_{c_ [(1)] _i_ _[, . . ., c]_ _i_ [(] _[L]_ [)] _}_, which stores the most recent confidence scores _c_ ( **x** ) from test inputs assigned to
class _i_ . Only confidently assigned samples (confidence _≥_ _θ_ ) contribute to the reliability buffer, using
the same confidence as in _y_ _[∗]_ ( **x** ). When a new score arrives, the oldest entry is removed if the buffer
exceeds size _L_ . The class-wise reliability statistics are then computed as:


_|Qi|_


_n_ =1


~~�~~


- [1]
_|Qi|_


_|Qi|_

- _c_ [(] _i_ _[n]_ [)] _,_ _σi_ =

_n_ =1


6


**Algorithm 1:** Overview of the SURE algorithm.

**Input:** Pretrained CLIP, test stream _{_ **x** _}_ _[T]_ _t_ =1 [, threshold] _[ θ]_ [, neighbor size] _[ k]_ [, cache size] _[ L]_
**Output:** Adapted prediction _y_ ˆ( **x** ) for each input

**1** Initialize reliability stats: _µi_ _←_ 1 _._ 0, _σi_ _←_ 0 _._ 0, _Qi_ _←∅_ for all _i_ ;

**2** **for** _each test sample_ **x do**

**3** Compute CLIP scores _p_ ( _yi|_ **x** );


**13** **if** _c_ ( **x** ) _> θ_ **then**

**14** Update prototype **t** _y∗_ via Eq. 12;

**15** _Ny_ [proto] _[∗]_ _←_ _Ny_ [proto] _[∗]_ + 1;

**16** Append _c_ ( **x** ) to _Qy∗_ and keep last _L_ entries;

**17** Update _µy∗_ _, σy∗_ via Eq. 13;


Table 1: Results of natural distribution shifts for SURE and recent baselines using ResNet-50 and
CLIP-ViT-B/16. We report the top-1 accuracy (%) for each dataset, along with the average accuracy
for the five datasets and average OOD accuracy for ImageNet-A, -V2, -R, -Sketch. The best average
and OOD average results are highlighted in **bold** .


**Method** **ImageNet** **ImageNet-A** **ImageNet-V2** **ImageNet-R** **ImageNet-Sketch** **Average** **OOD Average**


CLIP-RN50 (Radford et al., 2021) 59.81 23.24 52.91 60.72 35.48 46.43 43.09


TPT (Shu et al., 2022) 60.74 26.67 54.70 59.11 35.09 47.26 43.89
TDA (Karmanov et al., 2024) 61.35 30.29 55.54 62.58 38.12 49.58 46.63
DPE (Zhang et al., 2024a) 63.41 30.15 56.72 63.72 40.03 50.81 47.66
BCA (Zhou et al., 2025) 61.81 30.35 56.58 62.89 38.08 49.94 46.98
R-TPT (Sheng et al., 2025) 60.9 28.4 54.9 57.6 34.0 47.1 43.73


**SURE (Ours)** 64.08 29.57 57.75 63.38 40.83 **51.12** **47.88**


CLIP-ViT-B (Radford et al., 2021) 68.34 49.89 61.88 77.65 48.24 61.20 59.40


TPT (Shu et al., 2022) 68.98 54.77 63.45 77.06 47.94 62.44 60.81
MTA (Zanella & Ben Ayed, 2024) 70.08 58.06 64.24 78.33 49.61 64.06 62.56
TDA (Karmanov et al., 2024) 69.51 60.11 64.67 80.24 50.54 65.01 63.89
DPE (Zhang et al., 2024a) 71.91 59.63 65.44 80.40 52.26 65.93 64.43
ZERO (Farina et al., 2024) 71.17 62.75 65.23 80.75 50.59 66.10 64.83
TTL (Imam et al., 2025) 70.23 60.51 64.55 77.54 48.61 64.29 62.80
BCA (Zhou et al., 2025) 70.22 61.14 64.90 80.72 50.87 65.37 64.16


**SURE (Ours)** 71.20 61.45 65.67 79.96 52.88 **66.23** **64.99**


This buffered estimation provides a temporally smoothed and statistically stable summary of pseudolabel confidence for class _i_, avoiding fragile per-step updates. The resulting ( _µi, σi_ ) are then used
in Eq. 4 to update the reliability score _Ri_, thereby enabling robust graph evolution over time.


**Inference protocol.** Our overall process is shown in Algorithm 1. Final predictions are made using
the adjusted posterior _p_ ˆ( _yi|_ **x** ), which integrates both the model’s raw prediction and the structural
consensus induced by the graph.


5 EXPERIMENT


5.1 EXPERIMENTAL SETUP


**Datasets.** We evaluate generalization under two complementary settings. For _natural_ _distribu-_
_tion_ _shifts_, we use ImageNet (Deng et al., 2009) and its OOD variants—ImageNet-V2 (Recht


7


**4** Compute **S** _ij_ = cos( **t** _i,_ **t** _j_ );


_σj_
**5** Compute _Ri_ = _µi ·_ (1 _−_ _σ_ max [)][;]


**6** Form **W** _i,j_ = **S** _i,j_ _· Ri · Rj_ ;


**7** Sparsify to top- _k_ neighbors _⇒_ **A** [(] _[ℓ]_ [)] ;


**8** Maintain buffer _{_ **A** [(] _[ℓ][−][i]_ [)] _}_ _[L]_ _i_ =0 _[−]_ [1][;]


**9** Update: **A** [¯] [(] _[ℓ]_ [)] _←_ [1]


_L_ [1] - _Li_ =0 _−_ 1 **[A]** [(] _[ℓ][−][i]_ [)][;]


**10** Perform graph-based smoothing _⇒_ _p_ ˆ( _yi|_ **x** ) (Eq. 10);


**11** Assign pseudo-label _y_ _[∗]_ _←_ arg max _i_ _p_ ˆ( _yi|_ **x** );


**12** Compute confidence _c_ ( **x** ) _←_ _p_ ˆ( _y_ _[∗]_ _|_ **x** );


Table 2: Results of cross-dataset generalization for SURE and recent baselines using ResNet-50 and
CLIP-ViT-B/16. We report the top-1 accuracy (%) for each dataset, as well as the average accuracy
across the ten datasets. The best average results are highlighted in **bold** .


**Method** **SUN397** **Aircraft** **EuroSAT** **Cars** **Food101** **Pets** **Flower** **Caltech** **DTD** **UCF101** **Average**


CLIP-RN50 (Radford et al., 2021) 60.85 16.11 25.79 55.89 74.82 82.97 62.77 87.26 40.37 59.48 56.63


TPT (Shu et al., 2022) 61.46 17.58 28.33 58.46 74.88 84.49 62.69 87.02 40.84 60.82 57.66
TDA (Karmanov et al., 2024) 62.53 17.61 42.11 57.78 77.75 86.18 68.74 89.70 43.74 64.18 61.03
DPE (Zhang et al., 2024a) 64.23 19.80 41.67 59.26 77.83 85.97 67.60 90.83 50.18 61.98 61.93
BCA (Zhou et al., 2025) 63.38 19.89 42.12 58.13 77.19 85.58 66.30 89.70 48.58 63.51 61.44


**SURE (Ours)** 64.18 20.45 41.41 62.07 79.78 85.35 68.56 89.98 50.90 65.24 **62.79**


CLIP-ViT-B (Radford et al., 2021) 65.63 23.22 50.42 66.11 82.86 86.92 66.99 93.55 45.04 65.16 64.59


TPT (Shu et al., 2022) 65.50 24.78 42.44 66.87 84.67 87.79 68.98 94.16 47.75 68.04 65.10
MTA (Zanella & Ben Ayed, 2024) 66.67 25.20 45.36 68.47 85.00 88.24 68.06 94.21 45.90 68.69 65.58
TDA (Karmanov et al., 2024) 67.62 23.91 58.00 67.28 86.14 88.63 71.42 94.24 47.40 70.66 67.53
DPE (Zhang et al., 2024a) 70.07 28.95 55.79 67.31 86.17 91.14 75.07 94.81 54.20 70.44 69.40
Zero (Farina et al., 2024) 66.90 24.42 43.77 68.48 84.58 87.20 66.82 94.14 45.86 68.57 65.07
ZERO (Farina et al., 2024) 67.63 25.21 42.17 68.97 86.77 87.83 67.17 94.41 45.86 69.18 65.52
BCA (Zhou et al., 2025) 68.41 28.59 56.63 66.86 85.97 90.43 73.12 94.69 53.49 67.59 68.59
TTL (Imam et al., 2025) 66.32 23.82 42.02 67.96 85.05 88.72 70.48 93.63 46.69 69.20 65.39


**SURE (Ours)** 70.82 28.92 53.60 69.31 87.47 89.81 77.75 94.89 55.26 72.56 **70.04**


Table 3: Comparison of our proposed SURE with baselines in
terms of test time (s) and mean accuracy (%) on natural distribution shifts (ImageNet and its variants). Test time represents
the average inference time per sample, measured on an NVIDIA
RTX A6000 GPU.


**Method** Test Time (s) Accuracy (%) ∆ Gain (%)


CLIP-ViT-B (Radford et al., 2021) 0.004 61.20 +0.00
TPT (Shu et al., 2022) 0.706 62.44 +1.24
MTA (Zanella & Ben Ayed, 2024) 0.060 63.16 +1.96
DPE (Zhang et al., 2024a) 0.189 65.93 +4.73
BCA (Zhou et al., 2025) 0.023 65.37 +4.17
ZERO (Farina et al., 2024) 0.082 66.10 +4.90


**SURE** 0.067 66.23 +7.12


et al., 2019), ImageNet-A (Hendrycks et al., 2021b), ImageNet-R (Hendrycks et al., 2021a), and
ImageNet-Sketch (Wang et al., 2019). For _cross-dataset generalization_, we assess transfer to diverse
domains, including objects (Caltech101 (Fei-Fei et al., 2004)), fine-grained categories (OxfordPets (Parkhi et al., 2012), StanfordCars (Krause et al., 2013), Flowers102 (Nilsback & Zisserman, 2008), Food101 (Bossard et al., 2014), FGVC-Aircraft (Du et al., 2020)), and scenes or textures (SUN397 (Xiao et al., 2010), EuroSAT (Helber et al., 2019), DTD (Cimpoi et al., 2014),
UCF101 (Soomro et al., 2012)).


**Baselines.** We compare with representative TTA methods, including gradient-based approaches
(TPT (Shu et al., 2022), TPS (Sui et al., 2024), DPE (Zhang et al., 2024a)) and gradient-free alternatives (BCA (Zhou et al., 2025), MTA (Zanella & Ben Ayed, 2024)). Among them, DPE, TDA, and
BCA leverage historical test streams, while TPT and TPS perform online adaptation based solely on
the current input.


**Implementation details.** We use ResNet-50 (RN50) (He et al., 2016) and ViT-B/16 (ViT-B) (Dosovitskiy et al., 2020) as CLIP image encoders (Radford et al., 2021), initialized with pretrained
weights. All models are implemented using the official CLIP codebase. Following (Zhou et al.,
2025), each class prototype is initialized with _Ni_ [proto] = 30000 confident samples and updated via
normalized moving averages. For reliability estimation, we maintain a per-class sliding window of
size _L_ = 5. The graph neighbor size is set as _k_ = 3 _·_ log( _C_ ), and the confidence threshold is _θ_ = 0 _._ 3.
Prompts are derived by majority vote over 80 handcrafted templates (Radford et al., 2021).


5.2 COMPARISONS WITH STATE-OF-THE-ART


**Results on natural distribution shifts.** As shown in Tab. 1, **SURE** consistently outperforms prior
methods on both RN50 and ViT-B backbones. The gains are most pronounced under RN50 (+4.79%
over CLIP), where limited capacity amplifies semantic drift. Notably, SURE significantly improves


8


Table 4: Ablation study of key SURE components under natural distribution shifts on the ViT-B/16
backbone. Accuracy is reported for each dataset (%) along with average and OOD-only average
across ImageNet-A/V2/R/Sketch. Each method is shown in two rows: the first for absolute accuracy,
the second for the incremental gain compared to the preceding variant (or the baseline).


**Method** **ImageNet** **ImageNet-A** **ImageNet-V2** **ImageNet-R** **ImageNet-Sketch** **Average** **OOD Average**


CLIP-ViT-B/16 68.34 49.89 61.88 77.65 48.24 61.20 59.40


69.31 57.92 63.40 78.48 50.13 63.85 62.48
ProtoOnly
0.97 _↑_ 8.03 _↑_ 1.52 _↑_ 0.83 _↑_ 1.89 _↑_ 2.65 _↑_ 3.08 _↑_

69.63 57.68 63.80 78.63 50.70 64.09 62.70
+Graph w/o Rel
0.32 _↑_ -0.24 _↓_ 0.40 _↑_ 0.15 _↑_ 0.57 _↑_ 0.24 _↑_ 0.22 _↑_

70.07 60.49 64.36 78.17 52.73 65.16 63.94
+Graph + Rel
0.44 _↑_ 2.81 _↑_ 0.56 _↑_ -0.46 _↓_ 2.03 _↑_ 1.07 _↑_ 1.24 _↑_

71.20 61.45 65.67 79.96 52.88 66.23 64.99
+LogitProp (Full)
1.13 _↑_ 0.96 _↑_ 1.31 _↑_ 1.79 _↑_ 0.15 _↑_ 1.07 _↑_ 1.05 _↑_


Figure 3: We show average accuracy under natural distribution shifts and cross-dataset generalization while varying confidence threshold _θ_, neighbor size _k_, and buffer length _L_ .


robustness on ImageNet-Sketch, a dataset characterized by abstract visual cues, reflecting its ability
to stabilize adaptation when prototypes face severe drift. While the numerical margin over DPE
on ImageNet-R is narrower—likely because low-level style cues are less influenced by semantic
drift—SURE demonstrates superior stability across seeds. Under ViT-B, SURE reaches 66.23% average accuracy, exceeding strong baselines like ZERO and DPE, validating that structured semantic
regularization provides complementary benefits beyond scaling backbone capacity.


**Results** **on** **cross-dataset** **generalization.** Tab. 2 shows that SURE achieves the highest average
accuracy across ten diverse datasets (+5.49% over CLIP-ViT-B). Gains are especially notable in
structure-sensitive domains (e.g., Aircraft, DTD) and fine-grained tasks (e.g., Flowers), indicating
that reliability-aware propagation effectively suppresses misleading interactions in high-variance
domains. In contrast, improvements on visually consistent datasets like Pets and Cars are relatively
modest. This is likely because prototypes in these domains are already compact, leaving less room
for graph-based refinement. Overall, SURE primarily benefits scenarios with ambiguous or highly
variable semantics, where our structure-aware refinement offers the most value.


**Efficiency** **analysis.** Tab. 3 confirms that SURE offers a strong balance of performance and efficiency (66.23% accuracy at 0.067s/sample). It runs over 10 _×_ faster than TPT and outperforms
ZERO (+0.13%) while reducing latency by _∼_ 18%, avoiding the computational overhead of processing multiple augmented views. This efficiency stems from our lightweight design: graph updates scale linearly with class count _C_ and are amortized through a sliding buffer. Compared to
BCA, SURE improves accuracy by +0.86% with negligible latency increase, confirming it is both
deployment-friendly and robust in streaming adaptation.


5.3 ABLATION STUDY


We analyze key components of SURE using ViT-B/16 under natural distribution shifts.


**Effectiveness** **of** **model** **components.** As shown in Tab. 4, the minimal baseline (ProtoOnly)
improves over raw CLIP by enabling continual adaptation. While adding graph structure (+Graph
w/o Rel) incorporates semantic context, it can be risky if unreliable edges dominate; indeed,
graph smoothing alone may even hurt performance in noisy domains. The reliability-aware modula

9


(a) Initial Semantic Graph (T = 0)

Ambiguous Connections


Figure 4: **Evolution** **of** **the** **Adjacency** **Matrix.** (Left) Initially, CLIP exhibits dense, ambiguous
connections (e.g., ‘Television’ strongly connects to ‘Tabby’). (Right) SURE suppresses unreliable
outliers like ‘Television’ (weight drops to 0.13) while preserving valid semantic clusters.


tion (+Graph + Rel) solves this, boosting OOD accuracy by +1.24% via down-weighting noisy
classes. Finally, enabling logit-level propagation (+LogitProp) yields the best result (+1.05%).
The trajectory confirms that SURE’s advantage comes from component synergy: the reliability
mechanism corrects graph imbalances, and logit propagation consolidates information from statistically reliable neighbors to reduce prediction variance.


**Hyperparameter analysis.** Fig. 3 analyzes the confidence threshold _θ_, neighbor size _k_, and window
size _L_ . Results show _θ_ =0 _._ 4 optimally balances noise suppression and information retention: lower
thresholds admit unreliable classes, while higher ones discard too much signal. The optimal _k_ varies
(4 for natural shifts, 3 for cross-dataset), reflecting semantic density: broader domains benefit from
denser graphs, while fine-grained ones risk semantic leakage with large _k_ . Performance stabilizes
for _L>_ 3, indicating that a short buffer provides reliable estimation without excessive smoothing.
These smooth variations confirm SURE’s robustness to hyperparameter tuning.


5.4 VISUALIZATION OF GRAPH EVOLUTION


To validate our Prototype-Reliability Graph (PRG), we visualize adjacency matrix evolution on a
“micro-universe” of 5 classes (three ‘Cat’ species, plus ‘Television’ and ‘Carton’) under simulated
distribution shift.


**Results.** As shown in Fig. 4, the initial graph ( _T_ = 0) exposes CLIP’s ambiguity, where the irrelevant ‘Television’ holds a dangerously strong connection to ‘Tabby’ ( **0** _._ **75** ), which could easily
lead to error propagation during prototype updates. After adaptation ( _T_ = _End_ ), SURE’s reliability
gating identifies ‘Television’ as unstable due to high variance. Consequently, its influence is drastically attenuated, with its diagonal weight dropping to 0.13 and spurious connections suppressed.
Crucially, the visualization reveals a clear hierarchy: the easy control class ‘Carton’ remains highly
trusted ( **0** _._ **81** ), the hard fine-grained class ‘Tiger Cat’ is preserved but modulated, while the pure
noise ‘Television’ is effectively silenced. This demonstrates that SURE learns a soft, statistically
grounded topology rather than a simple hard threshold.


6 CONCLUSION


We presented SURE, a test-time adaptation framework that leverages a dynamic prototype-reliability
graph to couple semantic similarity with class-wise reliability. By propagating structure-aware signals through this graph, SURE mitigates semantic drift and stabilizes predictions under distribution
shifts. Extensive experiments confirm consistent gains across natural and cross-dataset benchmarks,
highlighting both the effectiveness and efficiency of our closed-loop design. We believe SURE
provides a principled foundation for semantic uncertainty modeling in TTA and can inspire future
extensions toward graph-based reasoning in vision-language adaptation.


10


(b) Evolved Reliability Graph (T = End)

Targeted Suppression


1.0


0.8


0.6


0.4


0.2


0.0


REFERENCES


Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In _Computer vision–ECCV 2014:_ _13th European conference, zurich,_
_Switzerland, September 6-12, 2014, proceedings, part VI 13_, pp. 446–461. Springer, 2014.


Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In _Proceedings_ _of_ _the_ _IEEE_ _conference_ _on_ _computer_ _vision_ _and_
_pattern recognition_, pp. 3606–3613, 2014.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pp. 248–255. Ieee, 2009.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An
image is worth 16x16 words: Transformers for image recognition at scale. _arXiv_ _preprint_
_arXiv:2010.11929_, 2020.


Ruoyi Du, Dongliang Chang, Ayan Kumar Bhunia, Jiyang Xie, Zhanyu Ma, Yi-Zhe Song, and
Jun Guo. Fine-grained visual classification via progressive multi-granularity training of jigsaw
patches. In _European Conference on Computer Vision_, pp. 153–168. Springer, 2020.


Matteo Farina, Gianni Franchi, Giovanni Iacca, Massimiliano Mancini, and Elisa Ricci. Frustratingly easy test-time adaptation of vision-language models. _Advances in Neural Information Pro-_
_cessing Systems_, 37:129062–129093, 2024.


Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training
examples: An incremental bayesian approach tested on 101 object categories. In _2004 conference_
_on computer vision and pattern recognition workshop_, pp. 178–178. IEEE, 2004.


Cl´ement Fuchs, Maxime Zanella, and Christophe De Vleeschouwer. Online gaussian test-time adaptation of vision-language models. In _Proceedings of the Computer Vision and Pattern Recognition_
_Conference_, pp. 128–137, 2025.


Sachin Goyal, Mingjie Sun, Aditi Raghunathan, and J Zico Kolter. Test time adaptation via conjugate pseudo-labels. _Advances in Neural Information Processing Systems_, 35:6204–6218, 2022.


Shen Han, Zhiyao Zhou, Jiawei Chen, Zhezheng Hao, Sheng Zhou, Gang Wang, Yan Feng, Chun
Chen, and Can Wang. Uncertainty-aware graph structure learning. In _Proceedings of the ACM on_
_Web Conference 2025_, pp. 4863–4874, 2025.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings_ _of_ _the_ _IEEE_ _conference_ _on_ _computer_ _vision_ _and_ _pattern_ _recognition_, pp.
770–778, 2016.


Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset
and deep learning benchmark for land use and land cover classification. _IEEE Journal of Selected_
_Topics in Applied Earth Observations and Remote Sensing_, 12(7):2217–2226, 2019.


Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul
Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In _Proceedings of the IEEE/CVF international_
_conference on computer vision_, pp. 8340–8349, 2021a.


Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial
examples. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recogni-_
_tion_, pp. 15262–15271, 2021b.


Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. _ICLR_, 1(2):3, 2022.


Kexin Huang, Ying Jin, Emmanuel Candes, and Jure Leskovec. Uncertainty quantification over
graph with conformalized graph neural networks. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 36:26699–26721, 2023.


11


Raza Imam, Hanan Gani, Muhammad Huzaifa, and Karthik Nandakumar. Test-time low rank adaptation via confidence maximization for zero-shot generalization of vision-language models. In
_2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_, pp. 5449–5459.
IEEE, 2025.


Yusuke Iwasawa and Yutaka Matsuo. Test-time classifier adjustment module for model-agnostic domain generalization. _Advances in Neural Information Processing Systems_, 34:2427–2440, 2021.


Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan
Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning
with noisy text supervision. In _International_ _conference_ _on_ _machine_ _learning_, pp. 4904–4916.
PMLR, 2021.


Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing. Efficient
test-time adaptation of vision-language models. In _Proceedings of the IEEE/CVF Conference on_
_Computer Vision and Pattern Recognition_, pp. 14162–14171, 2024.


TN Kipf. Semi-supervised classification with graph convolutional networks. _arXiv_ _preprint_
_arXiv:1609.02907_, 2016.


Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained
categorization. In _Proceedings_ _of_ _the_ _IEEE_ _international_ _conference_ _on_ _computer_ _vision_ _work-_
_shops_, pp. 554–561, 2013.


Youngjun Lee, Doyoung Kim, Junhyeok Kang, Jihwan Bang, Hwanjun Song, and Jae-Gil Lee.
Ra-tta: Retrieval-augmented test-time adaptation for vision-language models. In _The Thirteenth_
_International Conference on Learning Representations_, 2025.


Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In _International conference on_
_machine learning_, pp. 12888–12900. PMLR, 2022.


Ruizhou Li and Haiyun Jiang. Graph-to-vision: Multi-graph understanding and reasoning using
vision-language models. _arXiv preprint arXiv:2503.21435_, 2025.


Jian Liang, Dapeng Hu, and Jiashi Feng. Do we really need to access the source data? source
hypothesis transfer for unsupervised domain adaptation. In _International conference on machine_
_learning_, pp. 6028–6039. PMLR, 2020.


Xiaosong Ma, Jie Zhang, Song Guo, and Wenchao Xu. Swapprompt: Test-time prompt adaptation
for vision-language models. _Advances in Neural Information Processing Systems_, 36, 2024.


Balamurali Murugesan, Julio Silva-Rodr´ıguez, Ismail Ben Ayed, and Jose Dolz. Robust calibration
of large vision-language adapters. In _European_ _Conference_ _on_ _Computer_ _Vision_, pp. 147–165.
Springer, 2024.


Mahdi Pakdaman Naeini, Gregory Cooper, and Milos Hauskrecht. Obtaining well calibrated probabilities using bayesian binning. In _Proceedings of the AAAI conference on artificial intelligence_,
volume 29, 2015.


Bo Ni, Yu Wang, Lu Cheng, Erik Blasch, and Tyler Derr. Towards trustworthy knowledge graph
reasoning: An uncertainty aware perspective. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 39, pp. 12417–12425, 2025.


Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number
of classes. In _2008 Sixth Indian conference on computer vision, graphics & image processing_, pp.
722–729. IEEE, 2008.


Shuaicheng Niu, Jiaxiang Wu, Yifan Zhang, Yaofo Chen, Shijian Zheng, Peilin Zhao, and Mingkui
Tan. Efficient test-time model adaptation without forgetting. In _International_ _conference_ _on_
_machine learning_, pp. 16888–16905. PMLR, 2022.


Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In _2012_
_IEEE conference on computer vision and pattern recognition_, pp. 3498–3505. IEEE, 2012.


12


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In _International conference on machine learning_, pp.
8748–8763. PMLR, 2021.


Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers
generalize to imagenet? In _International conference on machine learning_, pp. 5389–5400. PMLR,
2019.


Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias
Bethge. Improving robustness against common corruptions by covariate shift adaptation. _Ad-_
_vances in neural information processing systems_, 33:11539–11551, 2020a.


Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias
Bethge. Improving robustness against common corruptions by covariate shift adaptation. _Ad-_
_vances in neural information processing systems_, 33:11539–11551, 2020b.


Claude E Shannon. A mathematical theory of communication. _The Bell system technical journal_,
27(3):379–423, 1948.


Lijun Sheng, Jian Liang, Zilei Wang, and Ran He. R-tpt: Improving adversarial robustness of visionlanguage models through test-time prompt tuning. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_
_Pattern Recognition Conference_, pp. 29958–29967, 2025.


Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and
Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models.
_Advances in Neural Information Processing Systems_, 35:14274–14289, 2022.


Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. A dataset of 101 human action classes
from videos in the wild. _Center for Research in Computer Vision_, 2(11):1–7, 2012.


Elaine Sui, Xiaohan Wang, and Serena Yeung-Levy. Just shift it: Test-time prototype shifting for
zero-shot generalization with vision-language models. _arXiv preprint arXiv:2403.12952_, 2024.


Haopeng Sun, Lumin Xu, Sheng Jin, Ping Luo, Chen Qian, and Wentao Liu. Program: Prototype
graph model based pseudo-label learning for test-time adaptation. In _The_ _twelfth_ _international_
_conference on learning representations_, 2024.


Petar Veliˇckovi´c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua
Bengio. Graph attention networks. _arXiv preprint arXiv:1710.10903_, 2017.


Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully
test-time adaptation by entropy minimization. _arXiv preprint arXiv:2006.10726_, 2020.


Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. _Advances in Neural Information Processing Systems_,
32, 2019.


Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database:
Large-scale scene recognition from abbey to zoo. In _2010 IEEE computer society conference on_
_computer vision and pattern recognition_, pp. 3485–3492. IEEE, 2010.


Hee Suk Yoon, Eunseop Yoon, Joshua Tian Jin Tee, Mark Hasegawa-Johnson, Yingzhen Li, and
Chang D Yoo. C-tpt: Calibrated test-time prompt tuning for vision-language models via text
feature dispersion. _arXiv preprint arXiv:2403.14119_, 2024.


Maxime Zanella and Ismail Ben Ayed. On the test-time zero-shot generalization of vision-language
models: Do we really need prompt learning? In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_
_Computer Vision and Pattern Recognition_, pp. 23783–23793, 2024.


Ce Zhang, Simon Stepputtis, Katia Sycara, and Yaqi Xie. Dual prototype evolving for test-time
generalization of vision-language models. _arXiv preprint arXiv:2410.12790_, 2024a.


13


Jingyi Zhang, Jiaxing Huang, Xiaoqin Zhang, Ling Shao, and Shijian Lu. Historical test-time
prompt tuning for vision foundation models. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Sys-_
_tems_, 37:12872–12896, 2024b.


Marvin Zhang, Sergey Levine, and Chelsea Finn. Memo: Test time robustness via adaptation and
augmentation. _Advances in neural information processing systems_, 35:38629–38642, 2022a.


Renrui Zhang, Wei Zhang, Rongyao Fang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free adaption of clip for few-shot classification. In _European_
_conference on computer vision_, pp. 493–510. Springer, 2022b.


Renrui Zhang, Xiangfei Hu, Bohao Li, Siyuan Huang, Hanqiu Deng, Yu Qiao, Peng Gao, and
Hongsheng Li. Prompt, generate, then cache: Cascade of foundation models makes strong fewshot learners. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition_, pp. 15211–15222, 2023.


Shuai Zhao, Xiaohan Wang, Linchao Zhu, and Yi Yang. Test-time adaptation with clip reward for
zero-shot generalization in vision-language models. _arXiv preprint arXiv:2305.18010_, 2023.


Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for
vision-language models. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _conference_ _on_ _computer_ _vision_ _and_
_pattern recognition_, pp. 16816–16825, 2022a.


Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. _International Journal of Computer Vision_, 130(9):2337–2348, 2022b.


Lihua Zhou, Mao Ye, Shuaifeng Li, Nianxin Li, Xiatian Zhu, Lei Deng, Hongbin Liu, and Zhen
Lei. Bayesian test-time adaptation for vision-language models. In _Proceedings of the Computer_
_Vision and Pattern Recognition Conference_, pp. 29999–30009, 2025.


14


A APPENDIX


This appendix provides additional details and analyses to complement the main paper. Specifically,
we include:


    - A detailed overview of the datasets used, including their label granularity and domain characteristics;


    - The complete set of handcrafted textual prompts used for zero-shot and ensemble-based
classification;


    - An evaluation of prediction consistency across random seeds, highlighting the stability of
SURE;


    - A systematic hyperparameter study covering graph structure, confidence thresholds, and
update window size;


    - A comparative analysis of different prompt initialization strategies and their impact on
adaptation;


    - Calibration performance analysis using Expected Calibration Error (ECE) across all benchmark settings;


    - Additional remarks summarizing the robustness and generality of the SURE framework
across evaluation scenarios.


These supplementary results further support the effectiveness and robustness of the proposed SURE
framework under various evaluation perspectives.


A.1 DATASET OVERVIEW


To assess both robustness under natural distribution shifts and generalization to novel domains, we
adopt a total of 15 standard benchmarks, consistent with prior TTA literature such as TPT (Shu et al.,
2022). Tab. 5 outlines each dataset’s label granularity and test size.


ImageNet Deng et al. (2009) serves as the canonical large-scale object classification dataset, with
1,000 diverse classes. ImageNet-V2 Recht et al. (2019) introduces subtle distribution shifts via resampling. ImageNet-A Hendrycks et al. (2021b) contains adversarially filtered, high-difficulty examples. ImageNet-R Hendrycks et al. (2021a) includes artistic renditions and stylized depictions of
objects, while ImageNet-Sketch Wang et al. (2019) features hand-drawn black-and-white sketches.


Beyond the ImageNet family, Caltech101 Fei-Fei et al. (2004), StanfordCars Krause et al. (2013),
Flowers102 Nilsback & Zisserman (2008), Food101 Bossard et al. (2014), and OxfordPets Parkhi
et al. (2012) focus on fine-grained visual categorization within constrained domains. FGVCAircraft Du et al. (2020) emphasizes viewpoint and structural variation in aircraft types, while
DTD Cimpoi et al. (2014) consists of texture-centric categories. SUN397 Xiao et al. (2010) presents
a challenging scene recognition task with high intra-class variability. EuroSAT Helber et al. (2019)
introduces modality shift through multispectral satellite imagery, and UCF101 Soomro et al. (2012)
samples human action frames from video data, adding temporal and motion-related visual variability.


Together, these datasets offer a comprehensive evaluation protocol covering object-, texture-, scene-,
and action-level recognition across diverse real-world domains.


A.2 PROMPT TEMPLATE SPECIFICATION


Following the common practice in vision-language pretraining Radford et al. (2021), we adopt a
fixed set of handcrafted textual prompts to encode class semantics. Specifically, our method’s results
are obtained under the 80-way prompt ensemble, where each prompt serves as a natural language
template for class name insertion. The full list of these handcrafted templates, originally proposed
in Radford et al. (2021), is reproduced in Tab. 6.


15


Table 5: Summary of datasets used in our evaluation,
categorized by the nature of distribution shift. The first group
reflects natural variants of ImageNet, while the second spans
diverse domains to assess generalization.


**Type** **Dataset** **#Classes** **#Test Samples**


Table 6: Complete list of 80 handcrafted prompt templates used for generating text embeddings,
following the standard CLIP ensemble Radford et al. (2021). The placeholder _{}_ is substituted by
class names during inference.


a bad photo of a _{}_ a photo of many _{}_ a sculpture of a _{}_ a photo of the hard to see _{}_
a low resolution photo of the _{}_ a rendering of a _{}_ graffiti of a _{}_ a bad photo of the _{}_
a cropped photo of the _{}_ a tattoo of a _{}_ the embroidered _{}_ a photo of a hard to see _{}_
a bright photo of a _{}_ a photo of a clean _{}_ a photo of a dirty _{}_ a dark photo of the _{}_
a drawing of a _{}_ a photo of my _{}_ the plastic _{}_ a photo of the cool _{}_
a close-up photo of a _{}_ a black and white photo of the _{}_ a painting of the _{}_ a painting of a _{}_
a pixelated photo of the _{}_ a sculpture of the _{}_ a bright photo of the _{}_ a cropped photo of a _{}_
a plastic _{}_ a photo of the dirty _{}_ a jpeg corrupted photo of a _{}_ a blurry photo of the _{}_
a photo of the _{}_ a good photo of the _{}_ a rendering of the _{}_ a _{}_ in a video game
a photo of one _{}_ a doodle of a _{}_ a close-up photo of the _{}_ a photo of a _{}_
the origami _{}_ the _{}_ in a video game a sketch of a _{}_ a doodle of the _{}_
a origami _{}_ a low resolution photo of a _{}_ the toy _{}_ a rendition of the _{}_
a photo of the clean _{}_ a photo of a large _{}_ a rendition of a _{}_ a photo of a nice _{}_
a photo of a weird _{}_ a blurry photo of a _{}_ a cartoon _{}_ art of a _{}_
a sketch of the _{}_ a embroidered _{}_ a pixelated photo of a _{}_ itap of the _{}_
a jpeg corrupted photo of the _{}_ a good photo of a _{}_ a plushie _{}_ a photo of the nice _{}_
a photo of the small _{}_ a photo of the weird _{}_ the cartoon _{}_ art of the _{}_
a drawing of the _{}_ a photo of the large _{}_ a black and white photo of a _{}_ the plushie _{}_
a dark photo of a _{}_ itap of a _{}_ graffiti of the _{}_ a toy _{}_
itap of my _{}_ a photo of a cool _{}_ a photo of a small _{}_ a tattoo of the _{}_


A.3 STABILITY ACROSS RANDOM SEEDS


To assess the robustness of our method under varying initialization and test-time orderings, we report
the standard deviation of performance across three independent runs in Tab. 7 and Tab. 8. SURE
exhibits consistently low variance across both natural distribution shifts and cross-dataset generalization benchmarks. On the ImageNet variants, deviations remain below 0 _._ 3%, with particularly
stable performance on ImageNet-Sketch and ImageNet-V2. Similarly, on the cross-domain benchmarks, the standard deviations for SURE-ViT-B do not exceed 0 _._ 3%, with most domains exhibiting
fluctuations within 0 _._ 2%. These results confirm that SURE not only achieves strong accuracy but
also maintains stable behavior across random seeds and domain shifts.


A.4 HYPERPARAMETER TUNING


We perform a systematic grid search over the key hyperparameters in our framework. For the graph
neighbor size _k_, we evaluate values from log _C_ to 10 log _C_ in integer multiples. The confidence
threshold _θ_ is searched within the range [0 _._ 1 _,_ 0 _._ 9] with a step size of 0 _._ 1, while the temporal smoothing window _L_ is varied from 1 to 10 in steps of 1. All hyperparameters are selected based on
performance on the ImageNet validation set and then fixed for evaluation on both the natural distribution shift benchmarks and the cross-dataset generalization tasks. Following Zhou et al. (2025),
the initialization count _Ni_ [proto] for each class prototype is set to a constant prior to adaptation.


16


Natural Shift


Cross-Domain


ImageNet (Deng et al., 2009) 1000 50,000
ImageNet-V2 (Recht et al., 2019) 1000 10,000
ImageNet-A (Hendrycks et al., 2021b) 200 7,500
ImageNet-R (Hendrycks et al., 2021a) 200 30,000
ImageNet-Sketch (Wang et al., 2019) 1000 50,889


Caltech101 (Fei-Fei et al., 2004) 100 2,465
OxfordPets (Parkhi et al., 2012) 37 3,669
StanfordCars (Krause et al., 2013) 196 8,041
Flowers102 (Nilsback & Zisserman, 2008) 102 2,463
Food101 (Bossard et al., 2014) 101 30,300
FGVCAircraft (Du et al., 2020) 100 3,333
SUN397 (Xiao et al., 2010) 397 19,850
DTD (Cimpoi et al., 2014) 47 1,692
EuroSAT (Helber et al., 2019) 10 8,100
UCF101 (Soomro et al., 2012) 101 3,783


Table 7: Performance of SURE on natural distribution shifts with standard deviation across three
runs. Both ResNet-50 and ViT-B backbones are evaluated. SURE consistently improves accuracy
over CLIP while maintaining low variance, indicating stable adaptation.


**Method** **ImageNet** **ImageNet-A** **ImageNet-V2** **ImageNet-R** **ImageNet-Sketch** **Average** **OOD Average**


CLIP-RN50 (Radford et al., 2021) 59.81 23.24 52.91 60.72 35.48 46.43 43.09


64.08 29.57 57.75 63.38 40.83 **51.12** **47.88**
**SURE (Ours)**
_±_ .16 _±_ .18 _±_ .17 _±_ .28 _±_ .16 _±_ .12 _±_ .14


CLIP-ViT-B (Radford et al., 2021) 68.34 49.89 61.88 77.65 48.24 61.20 59.40


71.20 61.45 65.67 79.96 52.88 **66.23** **64.99**
**SURE (Ours)**
_±_ .14 _±_ .23 _±_ .21 _±_ .13 _±_ .16 _±_ .11 _±_ .16


Table 8: Cross-dataset generalization results with standard deviation across three random seeds.
SURE achieves state-of-the-art accuracy on all datasets while exhibiting small performance variation, demonstrating robustness across diverse domains and categories.


**Method** **SUN397** **Aircraft** **EuroSAT** **Cars** **Food101** **Pets** **Flower** **Caltech** **DTD** **UCF101** **Average**


CLIP-RN50 (Radford et al., 2021) 60.85 16.11 25.79 55.89 74.82 82.97 62.77 87.26 40.37 59.48 56.63


64.18 20.45 41.41 62.07 79.78 85.35 68.56 89.98 50.90 65.24 **62.79**
**SURE (Ours)**
_±_ .04 _±_ .15 _±_ .22 _±_ .21 _±_ .09 _±_ .15 _±_ .15 _±_ .16 _±_ .21 _±_ .14 _±_ .11


CLIP-ViT-B (Radford et al., 2021) 65.63 23.22 50.42 66.11 82.86 86.92 66.99 93.55 45.04 65.16 64.59


70.84 28.92 53.60 69.31 87.47 89.81 77.75 94.89 55.26 72.56 **70.04**
**SURE (Ours)**
_±_ .08 _±_ .17 _±_ .24 _±_ .27 _±_ .09 _±_ .14 _±_ .25 _±_ .24 _±_ .18 _±_ .17 _±_ .14


A.5 PROMPT VARIATION ANALYSIS


To examine how different textual initializations influence test-time adaptation, we report the performance of our method under several representative prompt configurations. While our main results are
based on 80-way prompt ensemble (+Emsemble), we additionally evaluate two alternatives: a standard handcrafted prompt (+Basic) and CoOp-learned prompt embeddings (+CoOp). This analysis
provides insights into the sensitivity of adaptation performance with respect to prompt design and
semantic expressiveness.


    - +Basic: A single handcrafted prompt “a photo of a [class]”, which is commonly used in CLIP-based zero-shot classification.


    - +CoOp: A learned prompt embedding introduced in Zhou et al. (2022b), trained with 16shot supervision on ImageNet and composed of four class-specific tokens. We directly
use the pretrained prompt embeddings without additional tuning, following Zanella &
Ben Ayed (2024).


As shown in Tab. 9, prompt initialization has a non-trivial impact on adaptation performance. Compared to the basic prompt, both +CoOp and +Ensemble consistently improve accuracy across all
ImageNet variants. Notably, +CoOp achieves the highest accuracy on most individual datasets and
leads in both overall and OOD averages, reflecting the advantage of learned prompt embeddings.
Meanwhile, +Ensemble also performs strongly, benefiting from linguistic diversity and offering
competitive robustness under domain shift. These results highlight that SURE is not limited to a
specific prompt format and can flexibly adapt to various textual initializations. The ability to incorporate both handcrafted and learned prompts further demonstrates the generality and robustness of
our framework in real-world settings.


A.6 CALIBRATION ANALYSIS


In addition to accuracy, we assess model confidence calibration using the Expected Calibration
Error (ECE) Naeini et al. (2015), a standard metric that quantifies the alignment between predicted
probabilities and actual correctness. Formally, ECE is computed as:


ECE =


_|Bm|_ _|_ acc( _Bm_ ) _−_ conf( _Bm_ ) _|,_ (14)

_N_


17


_M_


_m_ =1


Table 9: Performance of SURE under different prompt initialization strategies on ImageNet and its
OOD variants using the CLIP-ViT-B/16 backbone. Basic uses a single handcrafted prompt (“a
photo of a [class]”), CoOp adopts learned prompts from Zhou et al. (2022b), and Ensemble averages 80 handcrafted prompts. +Ensemble yields the best average OOD accuracy, while +CoOp
achieves the highest overall performance.


**Method** **ImageNet** **ImageNet-A** **ImageNet-V2** **ImageNet-R** **ImageNet-Sketch** **Average** **OOD Average**


CLIP-ViT-B (Radford et al., 2021) 68.34 49.89 61.88 77.65 48.24 61.20 59.40


SURE (Basic) 69.68 60.12 63.98 80.34 51.56 65.13 64.00
SURE (CoOp) 74.82 62.38 67.04 81.35 53.79 67.88 66.14
SURE (Emsemble) 71.20 61.45 65.67 79.96 52.88 66.23 64.99


Table 10: Comparison of classification accuracy ( _↑_ ) and Expected Calibration Error (ECE, _↓_ ) on nature-shift distributions
(ImageNet and its 4 variants) and cross-dataset generalization
benchmarks. While CLIP shows strong calibration without adaptation, it lags in accuracy. Calibration-aware baselines such as
SaLS Murugesan et al. (2024) and C-TPT Yoon et al. (2024)
improve ECE moderately but sacrifice generalization. SURE
achieves the best accuracy overall while maintaining competitive
calibration.


ImageNet OOD Cross-dataset
**Method**
Acc. _↑_ ECE _↓_ Acc. _↑_ ECE _↓_


CLIP-ViT-B(Radford et al., 2021) 57.54 6.29 65.12 3.75
TPT (Shu et al., 2022) 60.93 10.8 60.08 9.90
TPT (Shu et al., 2022) + C-TPT (Yoon et al., 2024) 60.70 8.32 65.47 4.64
TPT (Shu et al., 2022) + SaLS (Murugesan et al., 2024) 60.89 10.86 64.59 9.88
ProtoOnly (Tab. 4) 63.85 11.23 67.83 9.86
**SURE (Ours)** 66.23 7.48 70.08 6.54


where _M_ denotes the number of confidence bins, _Bm_ is the _m_ -th bin containing _|Bm|_ samples, _N_
is the total number of samples, acc( _Bm_ ) is the empirical accuracy within the bin, and conf( _Bm_ ) is
the mean predicted confidence. Lower ECE values indicate better-calibrated predictions.


**Calibration-performance** **trade-off.** Tab. 10 compares SURE with several test-time adaptation
methods in terms of both classification accuracy and model calibration, measured by ECE. The zeroshot CLIP baseline exhibits strong calibration due to its lack of adaptation dynamics, but its accuracy is notably limited, especially under domain shift. TPT Shu et al. (2022) improves accuracy via
test-time prompt tuning, but significantly increases ECE. Notably, the ProtoOnly baseline achieves
competitive accuracy gains but suffers from the highest calibration error (11.23 on ImageNet OOD),
indicating that naive prototype updates without constraints tend to generate overconfident predictions and exacerbate semantic drift.


To mitigate miscalibration, calibration-aware extensions such as SaLS Murugesan et al. (2024) and
C-TPT Yoon et al. (2024) introduce entropy-based or post-hoc smoothing mechanisms. While they
reduce ECE to some extent, their performance on cross-dataset benchmarks remains inferior. In
contrast, SURE achieves a superior balance: it consistently outperforms all baselines in accuracy
across both OOD and cross-dataset settings, while keeping ECE relatively low (7.48 and 6.54, respectively). Crucially, SURE significantly reduces ECE compared to ProtoOnly (7.48 vs. 11.23),
demonstrating that the proposed reliability-aware graph acts as a safety net, effectively suppressing
noise to enhance predictive performance while preserving trustworthy confidence estimation.


A.7 ADDITIONAL REMARKS


The extended results presented in this appendix offer further evidence for the effectiveness and
robustness of the proposed SURE framework. Across all 15 datasets, SURE consistently improves
accuracy under both natural distribution shifts and cross-domain generalization scenarios, while
maintaining low variance and competitive calibration.


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


The stability analysis shows that our method performs reliably under different random seeds and testtime permutations, which is crucial for real-world deployment. The hyperparameter search confirms
that SURE is not overly sensitive to tuning, especially when initialized using prior knowledge from
zero-shot models.


Our prompt variation experiments indicate that SURE is not limited to a specific prompt initialization. While learned prompts such as CoOp can lead to strong performance, SURE also demonstrates
robust adaptation when combined with handcrafted templates, particularly under distribution shifts.
This flexibility highlights SURE’s ability to accommodate diverse prompt configurations, benefiting
from either semantic richness or data-driven optimization.


Moreover, the calibration results suggest that SURE mitigates the overconfidence typically caused
by adaptation, preserving the trustworthiness of predictions. Taken together, these findings highlight
the practicality of SURE as a simple, effective, and robust solution for adapting vision-language
models without access to labeled target data.


We hope this supplementary analysis provides readers with deeper insights into the design choices
and empirical behavior of our method.


19
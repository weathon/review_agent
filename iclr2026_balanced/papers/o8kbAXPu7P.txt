# INDUCTIVE REASONING FOR TEMPORAL KNOWLEDGE GRAPHS WITH EMERGING ENTITIES


**Ze Zhao** [1] **, Yuhui He** [1] **, Lyuwen Wu** [1] **, Gu Tang** [1] **, Bin Lu** [1] **, Xiaoying Gan** [1] _[∗]_ **, Luoyi Fu** [1] **,**
**Xinbing Wang** [1] **, Chenghu Zhou** [2]

1Shanghai Jiao Tong University, 2IGSNRR, Chinese Academy of Sciences
_{_ zhaoze,hyhuiiiii,wlw2016,gutang,robinlu1209,ganxiaoying,
yiluofu,xwang8 _}_ @sjtu.edu.cn, zhouch@lreis.ac.cn


ABSTRACT


Reasoning on Temporal Knowledge Graphs (TKGs) is essential for predicting future events and time-aware facts. While existing methods are effective at capturing
relational dynamics, their performance is limited by a closed-world assumption,
which fails to account for emerging entities not present in the training. Notably,
these entities continuously join the network without historical interactions. Empirical study reveals that emerging entities are widespread in TKGs, comprising
roughly 25% of all entities. The absence of historical interactions of these entities
leads to significant performance degradation in reasoning tasks. Whereas, we observe that entities with semantic similarities often exhibit comparable interaction
histories, suggesting the presence of transferable temporal patterns. Inspired by
this insight, we propose TRANSFIR ( **Transf** erable **I** nductive **R** easoning), a novel
framework that leverages historical interaction sequences from semantically similar known entities to support inductive reasoning. Specifically, we propose a
codebook-based classifier that categorizes emerging entities into latent semantic
clusters, allowing them to adopt reasoning patterns from similar entities. Experimental results demonstrate that TRANSFIR outperforms all baselines in reasoning
on emerging entities, achieving an average improvement of 28.6% in Mean Reciprocal Rank (MRR) across multiple datasets. The implementations are available
[at https://github.com/zhaodazhuang2333/TransFIR.](https://github.com/zhaodazhuang2333/TransFIR)


1 INTRODUCTION


Reasoning on **Temporal Knowledge Graphs(TKGs)** facilitates the prediction of future events and
time-aware facts, significantly enhancing the utility and applicability of temporal knowledge graphs.
By explicitly modeling relation dynamics as the graph evolves, TKG reasoning captures temporal
dependencies and interaction patterns, thereby supporting event forecasting and time-aware inference (Liang et al., 2024; Zhang et al., 2025a). These capabilities form the foundation for applications
such as temporal question answering, clinical risk analysis, and recommendation systems (Xue et al.,
2024; Postiglione et al., 2024; Hu et al., 2024).


However, existing reasoning methods focus on modeling relation dynamics while neglecting the
emergence of new entities. In real-world graphs, both entities and relations evolve continuously.
**Emerging entities** often join the network without _historical interactions_ . This phenomenon is observed in various contexts, from social platforms adding new users (Wang et al., 2024a) to molecular networks coming new compounds (Hadipour et al., 2025). Although current methods effectively
capture relation dynamics and achieve strong forecast performance (Li et al., 2021; Xu et al., 2023b),
they typically assume a _closed_ entity set. Due to the absence of historical interactions, these models
lack adequate supervision and representation for emerging entities, which significantly limits their
reasoning capability. For instance, as shown in Fig. 1, when Barack Obama first assumes office,
predicting his first state visit becomes challenging due to the absence of historical interactions.


To clarify the challenges and opportunities for reasoning on emerging entities, we conduct an empirical study from three progressively deeper perspectives (see Sec. 3). From the **Data** perspective, we


_∗_ Correspondence to: Xiaoying Gan (ganxiaoying@sjtu.edu.cn).


1


observe that emerging entities are widespread in TKGs, with nearly 25% of entities not appearing in
the training set. Meanwhile, existing models show a significant performance drop on related events.
From the **Representation** perspective, we attribute this degradation to **representation** **collapse**,
caused by the lack of supervision signal from historical interactions . Finally, from the **Feasibility**
perspective, we explore invariant patterns to transfer to emerging entities and find that entities of
semantically similar type often exhibit comparable interaction histories.


As a inspiring method, inductive learning provides a promising approach for reasoning on new entities in knowledge graphs. Unlike transductive methods, which rely on entity-specific embeddings,
inductive approaches learn transferable patterns from subgraphs (Chen et al., 2022). For instance,
InGram (Lee et al., 2023b) constructs relation-affinity graphs to capture neighbor interactions, while
ULTRA (Galkin et al., 2024) generalizes to unseen entities through relative interaction representation. However, these methods are primarily designed for static KGs, where new entities already
have known interactions. In contrast, emerging entities in TKGs often arrive without any interactions. This lack of supervision signals can lead to representation collapse, raising a significant
challenge: **how can we prevent representation collapse in the absence of historical interactions.**


To address this challenge, we propose TRANSFIR
( **Transf** erable **I** nductive **R** easoning), an inductive reasoning framework designed to handle emerging entities
in TKGs. Inspired by our empirical observation that semantically similar entities exhibit transferable patterns,
we propose Interaction Chain to model such structures.
TRANSFIR extracts these patterns from Interaction Chains
and employs a codebook-based classifier to map entities
into latent semantic clusters, thereby transferring patterns
to emerging entities. Specifically, TRANSFIR follows a
Classification–Representation–Generalization pipeline: (i)
Classification maps entities to latent semantic clusters via
an interaction-aware codebook; (ii) Representation encodes
entity’s Interaction Chain to capture reasoning patterns; (iii)
Generalization propagates learned patterns within each cluster,
enabling emerging entities to obtain informative embeddings.
Together, these steps help prevent representation collapse and
improve forecasting performance for emerging entities.


In summary, the main contributions of our work are as follows:


known interaction
predict interaction


Figure 1: Illustration of Transductive vs. Inductive Reasoning on
Emerging Entities.


 - **Novel** **Framework.** We propose TRANSFIR, an inductive framework designed to transfer reasoning patterns from semantically similar entities to enable reasoning on emerging ones.


 - **Codebook-based Classifier for Transfer.** We propose an interaction-aware VQ codebook that
maps entities into latent semantic clusters. This facilitates reasoning pattern transfer while preventing representation collapse.


 - **Problem & Evidence.** We formally define the task of inductive reasoning on emerging entities
without historical interactions. Additionally, an empirical study demonstrates that such entities
are widespread in TKGs and existing methods suffer significant performance degradation.


 - **State-of-the-art Results.** TRANSFIR outperforms strong baselines across multiple benchmarks,
with an average improvement of 28 _._ 6% in MRR on four datasets.


2 PRELIMINARIES AND PROBLEM FORMALIZATION


**Reasoning on Temporal Knowledge Graphs.** A temporal knowledge graph (TKG) is structured
as a sequence of timestamped snapshots _G_ = _{Gt}t∈T_, where each snapshot _Gt_ = ( _E_ 1: _t, R, Ft_ ).
Here _E_ 1: _t_ is the set of entities observed up to time _t_, _R_ is the relation set, and _Ft_ _⊆E_ 1: _t ×R×E_ 1: _t ×_
_{t}_ represents the set of timestamped facts.


Given a query ( _es, r,_ ? _, tq_ ) with _tq_ in the future, temporal KG reasoning aims to predict the missing
entity based on the historical context _Htq_ = [�] _i<tq_ _[F][i]_ [.] [It follows a standard chronological time split]

to prevent leakage of future information. However, a non-trivial fraction of entities remain unseen
during training, posing significant challenges for generalizing to emerging entities.


2


Reasoning Patterns


transfer to


President
Tag:Country A


**Emerging**


Government
Tag:Country C


President
Tag:Country C


**Entity**


Figure 2: **(a) Entity emergence over time.** Across four TKGs, new entities continuously emerge;
about _≈_ 25% of entities are unseen during training. **(b) Performance comparison.** Under _Vanilla_
vs. _Emerging_ settings, strong baselines consistently drop on emerging entity triples. **(c) Represen-**
**tation** **collapse.** On ICEWS14, t-SNE of LogCL shows representation collapsing after training,
while known entities drift to a separate manifold. **(d) Transferable patterns.** Semantically similar
entities share relation-conditioned patterns, enabling transfer to emerging entities.


**Inductive** **Reasoning** **for** **Emerging** **Entities** **in** **TKGs.** We formalize inductive reasoning for
TKGs: reasoning for emerging entities that enter the graph without historical interactions. Define
the first-appearance time of entity _e_ be
_te_ ( _e_ ) = min _{ t ∈T_ _| e_ participates in some ( _es, r, eo, t_ ) _∈Ft }._
At timestamp _t_, _e_ is considered as **emerging** **entity** if _e_ _∈E_ 1: _t_ _\ E_ 1: _t−_ 1. The goal is to answer
temporal queries involving such entities at the moment they emerge— i.e., queries of the form
( _e, r,_ ? _, tq_ ) or (? _, r, e, tq_ ), where _tq_ = _te_ ( _e_ ), and no historical interactions are available. This setting
reflects real-world scenarios—such as the introduction of new users, proteins, or organizations—and
highlights the difficulty of reasoning in the absence of historical interaction.


3 EMPIRICAL INVESTIGATION


In this section, we empirically investigate the proposed inductive reasoning task for emerging entities from three complementary angles. From the **Data** perspective, we address the prevalence and
impact of emerging entities on forecasting by answering **Q1** : _How frequently do emerging entities_
_appear_ _in_ _TKGs,_ _and_ _how_ _do_ _they_ _influence_ _forecasting_ _performance?_ From the **Representation**
perspective, we explore the underlying causes of performance degradation by addressing **Q2** : _What_
_factors contribute to failures on emerging entities?_ From the **Feasibility** perspective, we investigate
potential alternatives to overcome the limitations of sparse interactions by answering **Q3** : _Are there_
_transferable_ _temporal_ _patterns_ _that_ _support_ _reasoning_ _without_ _historical_ _interactions?_ Details of
datasets, models, and metrics are provided in Sec. 5.


To address **Q1**, We quantify entity emergence and its impact on performance across multiple TKG
datasets. Specifically, we track both the number and the proportion of emerging entities over time.
To assess impact on forecasting, we evaluate representative baselines in two settings: (i) overall test
triples (Vanilla) and (ii) triples involving at least one emerging entity(Emerging).


**Observation** **1.** From Fig. 2(a), we find that new entities continuously emerge over time. Nearly
**25%** of entities appear only in inference set, having no historical interactions available for training,
indicating that entity emergence is widespread in TKGs. From Fig. 2(b), all models exhibit significant performance degradation on emerging triples compared with vanilla triples, underscoring the
challenge of generalizing to entities without historical interactions.


To address **Q2**, we evaluate the representation quality for all entities using t-SNE (known vs. emerging). We visualize both the initial embeddings (all entity embeddings are randomly initialized) and
the learned entity embeddings after training in baseline model, LogCL (Chen et al., 2024). Additionally, inspired by Zbontar et al. (2021), we propose a rotation-invariant mertic, **Collapse Ratio**, to
quantify the degree of collapse. Collapse ratio measures the geometric spread (log-det covariance)
of emerging-entity embeddings relative to a reference set; lower values indicate stronger collapse.
See Appendix C.2 for the full definition and details.


**Observation 2.** From Fig. 2(c), we find that after training, emerging entities deviate sharply from
known entities in the embedding space. Quantitatively, their Collapse Ratio drops from 1.0201 to
0.0055 after training, evidencing severe **representation collapse** .


3


Figure 3: The overall architecture of proposed TRANSFIR.


To address **Q3**, we investigate whether models can perform reasoning that is independent of entity
embedding. Inspired by prior work on inductive and path-based reasoning, we analyze the feasibility
to transfer interaction patterns across entities, and identify concrete instances of this phenomenon.


**Observation** **3.** We observe that certain reasoning patterns can be transferred across entities of
similar semantic types. For example, as shown in Fig. 2(d), visit–negotiation sequences patterns
can be reused when a new president takes office in another country with no interaction history. This
suggests that invariant event-sequence patterns can be captured and extended to semantically similar
entities, thereby supporting inductive inference for emerging entities.


4 METHODOLOGY


In this section, we present TRANSFIR, an inductive framework designed for _emerging_ _entities_
without interaction history. As shown in Fig. 3, TRANSFIR employs a three-stage _Classifica-_
_tion–Representation–Generalization_ pipeline that to transform raw interactions into transferable representations:


(1) **Codebook Mapping (Classification)** : Assign emerging and known entities to latent types (semantic clusters) via a vector-quantized (VQ) codebook, providing history-free categorical priors.


(2) **Interaction Chain Encoding (Representation)** : Construct and encode _Interaction Chains_ (ICs)
around query entities to capture transferable interaction sequences.


(3) **Temporal Pattern Transfer (Generalization)** : Propagate learned temporal patterns within each
cluster, enabling emerging entities to acquire informative, time-aware embeddings.


4.1 CODEBOOK MAPPING (CLASSIFICATION)


Entities of similar semantic types often share comparable interaction history (e.g., states show recurring diplomatic rhythms, while individuals follow distinct patterns). Inspired by this, categorizing
entities into latent semantic clusters offers a promising way to import type-level priors for emerging
entities. However, two straightforward strategies prove inadequate: (i) updating entity embeddings
directly risks collapse for emerging entities lack of supervision; (ii) based on frozen embeddings
fails to adapt to dynamic interactions within temporal knowledge graphs.


To address these challenges, we propose a learnable vector quantization (VQ) codebook: entity embeddings are fixed for stability, and cluster prototypes are trained to become interaction-aware. This
results in an adaptive latent semantic clustering mechanism to facilitate effective transfer learning.


4


**Vector-quantized** **clustering.** For each entity _e_ _∈E_, we first obtain a static textual embedding
**h** _e_ _∈_ R _[d]_ from its title using a pretrained BERT encoder. These embeddings remain fixed during
training, allowing emerging entities to be encoded even without any interaction history.


We maintain a learnable codebook _C_ = _{_ **c** 1 _, . . .,_ **c** _K}_, where each codeword **c** _k_ _∈_ R _[d]_ denotes a
latent cluster. Entity entity embedding is quantized by mapping it to the nearest codeword:


_π_ ( _e_ ) = arg min 2 _[,]_ (1)
_k_ _[∥]_ **[h]** _[e][ −]_ **[c]** _[k][∥]_ [2]


where _π_ ( _e_ ) is the cluster index of entity _e_ . This process groups both observed and emerging entities
into consistent categories, forming an adaptive semantic cluster structure for downstream reasoning.


**Codebook** **optimization.** To ensure meaningful latent semantic clusters, we adopt two complementary objectives. The _codebook loss_ updates prototypes toward their assigned embeddings:


_L_ cb = _∥_ sg[ **h** _e_ ] _−_ **c** _π_ ( _e_ ) _∥_ [2] 2 _[,]_ (2)


where sg[ _·_ ] denotes the stop-gradient operator. The _commitment loss_ encourages embeddings to stay
close to their prototypes:
_L_ commit = _∥_ **h** _e −_ sg[ **c** _π_ ( _e_ )] _∥_ [2] 2 _[.]_ (3)
The overall objective is
_L_ codebook = _αL_ cb + _βL_ commit _,_ (4)
with _α, β_ _>_ 0 as weighting coefficients. This optimization refines the codewords into semantically
coherent clusters and stabilizes the assignment of entities to prototypes. Unlike static clustering
methods, our approach jointly learns the prototypes with the task objective, making it suitable for
fixed entity embeddings while enabling adaptive category representations for effective classification.


4.2 INTERACTION CHAIN ENCODING (REPRESENTATION)


To capture transferable interaction sequence patterns for emerging entities, we introduce an **Interac-**
**tion Chain (IC)** around each query entity. Unlike unordered temporal neighborhoods, ICs preserve
the sequential structure of interactions, thereby reflecting entity-invariant temporal dynamics—such
as periodic behaviors or events that follow a specific order.


**Definition.** Given a temporal query _q_ = ( _eq, rq,_ ? _, tq_ ) and a window size _T_, the IC of _eq_ collects
its past interactions in chronological order:


_Cq_ = �( _eq, r, o, ti_ ) or ( _s, r, eq, ti_ ) �� _tq −_ _T_ _≤_ _ti_ _< tq_         - _,_ (5)


which captures the behavioral trajectory of _eq_ piror to time _tq_ . This chain-based structure is motivated by **Observation 3.**, which suggests that such sequential patterns are largely independent of
specific entities and are more effectively modeled through chains than through unordered neighborhoods (see Appendix D.1 for further details).


**Construction.** At query time _tq_, for each query _q_ = ( _eq, rq,_ ? _, tq_ ) we collect past interactions of
_eq_ within window _T_ to form _Cq_ . Let _ri_ be the relation of the _i_ -th interaction in _Cq_, with trainable
relation embeddings _hrq_ _, hri_ _∈_ R _[d]_ . We keep the _k_ interactions whose relations are most similar to
the query relation:
_Cq_ [(] _[k]_ [)] = TopK _i_ �sim� _hrq_ _, hri_            - _,_ _Cq_            - _,_ (6)


where sim is cosine similarity. Selected interactions are then kept in chronological order; doing this
for all queries at _t_ yields ICs _{Cq_ [(] _[k]_ [)] _}_ as the temporal context for the snapshot.


**Encoding.** Each interaction ( _si, ri, oi, ti_ ) _∈_ _Cq_ [(] _[k]_ [)] is first mapped by component-specific transforms _ϕ∗_ ( _·_ ) and fused by _f_ ( _·_ ):


_xi_ = _f_ ( _ϕe_ ( _hsi_ ) _,_ _ϕr_ ( _hri_ ) _,_ _ϕe_ ( _hoi_ ) _,_ _ϕτ_ ( _h_ ∆ _ti_ )) _,_ (7)

where _hsi, hoi_ _∈_ R _[d]_ are _frozen_ entity embeddings (from a pretrained encoder; see Sec. 4.1), _hri_ _∈_
R _[d]_ is a _trainable_ relation embedding, and _h_ ∆ _ti_ encodes the relative time gap ∆ _ti_ = _tq −_ _ti_ .


5


The sequence _{xi}_ _[n]_ _i_ =1 [is then contextualized by a Transformer encoder to yield] _[ {][h][i][}]_ _i_ _[n]_ =1 [.] [We apply]
relation-guided attention, modulated by the query-relation embedding _hrq_ :


_αi_ = exp� _w_ _[⊤]_ tanh( _Whhi_ + _Wqhrq_ )�


   - _q_   - _,_ **h** [IC] _eq_ [=]
_j_ [exp] _w_ _[⊤]_ tanh( _Whhj_ + _Wqhrq_ )


_n_

- _αi hi,_ (8)


_i_ =1


which produces the query-specific chain representation **h** [IC] _eq_ [, emphasizing interactions most relevant]
to _rq_ while down-weighting irrelevant context.


4.3 CHAIN PATTERN TRANSFER (GENERALIZATION)


Although IC encodings capture query-specific dynamics, entities with limited interactions remain
static, limiting generalization to emerging cases. To address this, we propose **Chain Pattern Trans-**
**fer**, a mechanism that propagates interaction patterns across semantic clusters. This approach enables even newly emerging entities to acquire time-aware representations.


**Cluster** **pooling.** At each timestamp _t_, we aggregate IC embeddings _{_ **h** [IC] _e_ _[}]_ [based] [on] [codebook]
assignments. Let _π_ ( _e_ ) be the cluster index of entity _e_ . The dynamic prototype of cluster _k_ is


1
**c** [dyn] _k_ = _|Qk|_


- **h** [IC] _e_ _[,]_ _Qk_ = _{e ∈E_ _| π_ ( _e_ ) = _k},_ (9)

_e∈Qk_


which summarizes the shared temporal evolution within semantic cluster _k_ .


**Pattern Transfer.** Each entity _e_ combines its static embedding **h** _e_ with the cluster-level prototype:

_ze_ = [ **h** _e ∥_ **c** [dyn] _π_ ( _e_ ) []] _[,]_ (10)

where _∥_ denotes concatenation. A parametric mapping Ψ( _·_ ) generates the transfer vector:

_ωe_ = Ψ( _ze_ ) _,_ **h** ˜ _e_ = **h** _e_ + _ωe ·_ **c** [dyn] _π_ ( _e_ ) _[.]_ (11)


Through the Pattern Transfer module, we transfer Interaction Chain’s information from semantically
similar known entities to emerging ones, resulting in informative entity representations.


**Ranking and optimization.** Given a query ( _eq, rq,_ ? _, tq_ ), candidate entities _eo_ are scored as


                 -                 _ϕ_ ( _eq, rq, eo, t_ ) = _σ_ _f_ ( **h** [˜] _eq_ _,_ **h** _rq_ _,_ **h** [˜] _eo_ ) _,_ (12)


where _f_ ( _·_ ) is implemented with ConvTransE (Shang et al., 2019), a strong score function that is
widely adopted for the recent TKG reasoning methods. The training objective is cross-entropy loss
over all candidate entities:


_L_ lp = _−_


_T_

- 

_t_ =1 ( _eq,rq,eo,tq_ ) _∈Ft_


- _yt_ _[e]_ _q_ [log] _[ ϕ]_ [(] _[e][q][, r][q][, e, t][q]_ [)] _[,]_ (13)

_e∈E_


where _yt_ _[e]_ _q_ [is the one-hot indicator for the correct entity.] [The overall objective is]

_L_ = _L_ lp + _λL_ codebook _._ (14)


In our work, both link prediction loss and codebook loss are trained simultaneously. A complete
algorithmic workflow, detailed pseudo code, and complexity analysis are provided in Appendix D.2.


5 EXPERIMENTS


We evaluate the effectiveness of TRANSFIR through extensive experiments and analyses, guided by
the following research questions:


 - **RQ1:** How does TRANSFIR compare with SOTA baselines in emerging entity reasoning?

 - **RQ2:** What insights can be obtained from the learning behavior of TRANSFIR?

 - **RQ3:** How does each component of TRANSFIR contribute to its overall effectiveness?

 - **RQ4:** How well does TRANSFIR generalize to new inductive scenarios?


6


Table 1: Performance comparison of inductive reasoning on emerging entities on four benchmarks.
In each column, best results are highlighted in **bold** and second-best are underlined. For generative
model GenTKG, MRR is unavailable due to it’s reliance on multiple generations for each query.


ICEWS14 ICEWS18 ICEWS05-15 GDELT
Method
MRR Hits@3 Hits@10 MRR Hits@3 Hits@10 MRR Hits@3 Hits@10 MRR Hits@3 Hits@10


CyGNet(2021) 0.0111 0.0098 0.0202 0.0031 0.0020 0.0047 0.0031 0.0020 0.0048 0.0067 0.0031 0.0147


REGCN(2021) 0.1175 0.1263 0.2232 0.0947 0.1004 0.1797 0.0887 0.0961 0.1803 0.0222 0.0209 0.0393


HiSMatch(2022) 0.0284 0.0285 0.0418 0.0055 0.0058 0.0076 0.0242 0.0238 0.0443 0.0159 0.0141 0.0270


MGESL(2024) 0.0309 0.0361 0.0603 0.0747 0.0809 0.1031 0.1069 0.1166 0.1563 0.0516 0.0471 0.0815


LogCL(2024) 0.1354 0.1770 0.2273 0.0903 0.1064 0.1548 0.1917 0.2452 0.2855 0.0473 0.0479 0.0973


HisRes(2025) 0.1169 0.1107 0.2132 0.0445 0.0434 0.0735 0.1325 0.1332 0.1407 0.0416 0.0737 0.0932


MLEMKD(2025) 0.0685 0.0728 0.1303 0.0402 0.0382 0.0831 0.0833 0.0848 0.1717 0.0229 0.0215 0.0436


TLogic(2022) 0.0122 0.0107 0.0257 0.0141 0.0131 0.0262 0.0121 0.0108 0.0285 0.0733 0.0749 0.1131


TILP(2024) 0.0397 0.0471 0.1114 0.0498 0.0669 0.1659 0.0358 0.0374 0.1030 0.0053 0.0025 0.0084


ECEformer(2024) 0.0461 0.0496 0.0915 0.0323 0.0680 0.0454 0.0587 0.0642 0.1141 0.0426 0.0410 0.0872


GenTKG(2024)   - 0.0983 0.1919   - 0.0540 0.1512   - 0.1105 0.1873   - 0.0734 0.1013


CompGCN(2020) 0.0682 0.0906 0.1213 0.0638 0.0745 0.1049 0.1885 0.2103 0.2479 0.0472 0.0791 0.0934


ICL(2023) 0.0252 0.0261 0.0388 0.0639 0.0727 0.0938 0.0254 0.0302 0.0373 0.0277 0.0326 0.0362


PPT(2023) 0.0093 0.1062 0.1716 0.0368 0.0386 0.0650 0.0015 0.0005 0.0022 0.0406 0.0425 0.0764


MorsE(2022) 0.0136 0.0074 0.0185 0.0078 0.0075 0.0126 0.0381 0.0167 0.0439 0.0039 0.0040 0.0152


InGram(2023) 0.0563 0.0596 0.1138 0.0254 0.0265 0.0518 0.0771 0.0793 0.1454 0.0471 0.0430 0.0847


TRANSFIR **0.1687** **0.1935** **0.3246** **0.1177** **0.1344** **0.2324** **0.2204** **0.2617** **0.3827** **0.1103** **0.1129** **0.2278**


Improvements **24.6%** **9.3%** **42.8%** **24.3%** **26.3%** **29.3%** **15.0%** **6.7%** **34.0%** **50.5%** **42.7%** **101.4%**


5.1 SETUP


**Datasets.** Our experiments are conducted on four widely used benchmark datasets for TKG reasoning: ICEWS14, ICEWS18, ICEWS05-15, and GDELT. Unlike the conventional 8:1:1 split, we
adopt a 5:2:3 chronological split. This approach helps reveal more emerging entities and better evaluates inductive reasoning performance. A detailed description of the datasets and their statistics can
be found in Appendix E.1.


**Baselines.** To demonstrate the effectiveness of TRANSFIR, we compare it with thirteen strong baselines across three complementary categories: (1) _Graph-based methods_ ; (2) _Path-based methods_ ; (3)
_Inductive methods_ . The details of the description and implementation of all methods are provided in
the Appendix E.2.


**Evaluation Metrics.** We report results using Mean Reciprocal Rank (MRR) and Hits@k (k=3,10),
the standard metrics for link prediction. We pay particular attention to triples involving _emerging_
_entities_, which directly reflects the ability to generalize beyond entities observed during training.


5.2 PERFORMANCE COMPARISON ( _RQ1_ )


The overall performance of TRANSFIR and all baseline methods on the four benchmark datasets
is summarized in Table 1. The best scores are highlighted in **bold**, and the second-best scores are
underlined. From the experimental results, we draw the following observations:


Firstly, TRANSFIR achieves the highest average performance across all four benchmarks, consistently ranking first in both MRR and Hits@k on every dataset. Its consistent superiority over graphbased, path-based, and static inductive baselines confirms the effectiveness of TRANSFIR for inductive reasoning on emerging entities (e.g., average MRR gain of **28.6%** over the strongest baseline).


Secondly, on the ICEWS series, TRANSFIR demonstrates notable gains. On ICEWS14 it surpasses
the best baseline by **24.6%** in MRR. Notably, the advantage persists on ICEWS05-15 (longer temporal horizon) and ICEWS18 (larger, denser graph), with improvements of **15.0%** and **24.3%**, respectively. Such robustness across varying time spans and graph dynamics indicates that the proposed
Classification-Representation-Generalization pipeline enables reliable inductive generalization.


7


Thirdly, on GDELT, a large and rapidly evolving dataset, TRANSFIR still outperforms all baselines,
with an MRR gain of 50.5%. We attribute this to the latent semantic cluster that supplies strong
categorical priors for emerging entities and propagates cluster-level dynamics.


5.3 REPRESENTATION AND LEARNING ANALYSIS (RQ2)


We analyze what TRANSFIR learns during training and how it addresses the challenges of emerging
entities, as summarized in Fig. 4.


**A. Country**


(a) Representation Collapse of baseline and our model (b) Case Analysis for Cluster and Emerging Entities


Figure 4: (a) t-SNE visualization showing the improved separation of clusters in TRANSFIR compared to LogCL, with Collapse Ratio improvement from 0.0055 to 0.8677. (b) Case analysis of three
representative clusters and how TRANSFIR transfers reasoning patterns to emerging entities.


**(a)** **Representation** **Quality** **and** **collapse.** Compared to LogCL, TRANSFIR produces wellseparated clusters in embedding space, rather than a single dense cloud. The Collapse Ratio improves markedly from 0.0055 to 0.8677, indicating that embeddings of emerging entities remain
well distributed and informative. These results suggest that the VQ codebook, combined with pattern generalization, jointly prevent representation collapse and yield informative embeddings.


**(b)** **Cluster** **Structure** **and** **Emerging** **Entities.** A closer look at three latent semantic clusters
identified by the codebook reveals semantically coherent, type-like groupings: **A.** **Country** (e.g.,
_France_, _Canada_, _United_ _States_ ); **B.** **Civic** **&** **Parties** (e.g., _Prime_ _Minister_ _(Romania)_, _Officials_
_(Mexico)_, _Presidential Candidate (Mexico)_ ); **C. Citizen** (e.g., _Citizen (Nigeria)_, _Student (Thailand)_,
_Children_ _(Canada)_ ). Emerging entities (red) are consistently categorizes to appropriate clusters
alongside known ones (blue), providing type-level priors even without historical interactions.


**(c) Case study.** Consider the query: ”Where did the presidential candidate in Mexico make a statement at _tq_ ?” This query is structured as( _Presidential Candidate (Mexico)_, MAKE STATEMENT, ?, _tq_ ).
TRANSFIR retrieves transferable pattern chains from the **B. Civic & Parties** cluster, including: (i)
Cross patterns, such as ( _Prime Minister (Romania_ _−−−−−−−−−→_ MAKE STATEMENT _Gov (Romania)_ ), and (ii)Within-country
patterns ( _Officials (Mexico)_ _−−−−−−−−−→_ MAKE STATEMENT _Gov (Mexico)_ ). By leveraging such transferable supervision
signals, TRANSFIR successfully predicts _Gov_ _(Mexico)_ . It illustrates how TRANSFIR utilizes
cluster-level priors to extract transferable patterns, enabling reasoning on emerging entities.


5.4 ABLATION STUDY (RQ3)


We conduct ablation experiments to evaluate the contribution of each module in TRANSFIR. The
following variants are considered:


 - **-IC** : removing the Interaction Chain construction and using only entity embeddings.

 - **-Codebook** : removing the codebook mapping and using static clustering features only.

 - **-Pattern Transfer** : removing the pattern transfer mechanism and using static representations.

 - **-Textual encoding** : removing frozen textual embeddings and using random initialization.

The results are summarized in Fig 5. Across all four benchmarks, removing any individual module
leads to a decline in performance. While the relative impact of each ablation varies across datasets,


8


Figure 5: Ablation study results on four benchmarks, showing the performance impact of removing
different components. Reported results are for Hits@10, with additional metrics in Appendix F.2.


two consistent patterns emerge (i) removing the codebook-driven mapping or the pattern-transfer
typically results in the most performance drops, highlighting the need to both mapping entities and
propagate pattern signals; (ii) removing IC construction or textual encoding also degrades performance. These findings demonstrate the complementary functions of the modules.


Besides, we observe that in GDELT, removing textual encoding can sometimes lead to better performance. We believe this is due to the quality of the input text in GDELT, where entity titles often include abbreviations and symbolic elements (e.g., ”EGYPT (EGY@ OPP REF LEG SPY...)”, which
makes it challenging for the textual encoding module to extract clear semantic information. Incorporating external knowledge sources to enrich entity descriptions may further enhance TRANSFIR
’s performance under such conditions.


5.5 EXTENDED EXPERIMENTS (RQ4)


To evaluate the generalization capability and robustness of TRANSFIR in inductive scenarios, we

ciency in GPU memory and computational time.


sults and experimental details are available in Appendix F.3.


Figure 6: Experiment results on
ICEWS14 under the _Unknown_
and _Emerging_ settings.


**Robustness** **under** **Different** **Temporal** **Splits.** We construct
four chronological data splits by varying the test horizon to 10%, 30%, 50%, and 70% of the full
timeline. For each ratio, we we re-partition the dataset chronologically into training, validation, and
test sets. This setup reduces the observed historical context and increases the proportion of emerging entities over time. We compare TRANSFIR against strong baselines strong baselines—LogCL,
REGCN, and MLEMKD. Across all splits, TRANSFIR achieves the best MRR/Hits@10 and exhibits the smallest degradation as emergence increases, demonstrating robustness to reduced historical coverage. Detailed partitioning protocols and full results are provided in Appendix F.4.


**Hyperparameter** **Sensitivity.** We analyze the sensitivity of TRANSFIR to several key hyperparameters: codebook size _K_, Interaction Chain length _k_, hidden dimension _d_, and number of layers


9


GPU Memory Usage


Training Time Analysis on ICEWS14


REGCN
MGESL

|Col1|LogCL|Col3|
|---|---|---|
||~~HisRes~~<br>TransF|IR|
||||


0 1000 2000 3000 4000 5000 6000 7000
Training Time (seconds)


12500


10000


7500


5000


2500


0


|Col1|641|2.2<br>558|4.0|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|342|8.5|||357|1.9<br>330|2.3|
||||||||


REGCN MGESL LogCL HisRes MLEMKD TransFIR
Model


~~11654.5~~


0.15


0.10


0.05


0.00


Figure 7: GPU memory usage and empirical running time on ICEWS14. TRANSFIR achieves
significantly lower peak GPU memory usage while maintaining competitive training speed.


_L_ . TRANSFIR demonstrates consistent performance across a wide range of parameters. Our experiments show that datasets with greater diversity (e.g., ICEWS18) require a larger number of
codebooks. For chain length, most datasets achieve best performance at _k_ = 30, longer chains
potentially introducing noise. Comprehensive experiment results can be found in Appendix F.5.


**Different** **Textual** **Encoder.** We investigate the impact of different pretrained language models(PLM)
on TRANSFIR, as its latent semantic clustering relies on textual representations. Specifically, we evaluate four widely-used PLMs: BERT (Devlin et al.,
2019), RoBERTa (Liu et al., 2019), T5 (Raffel et al.,
2020), and Qwen3-Embedding (Zhang et al., 2025b).
As shown in Table 2, TRANSFIR consistently outperforms the strongest baseline across all pretrained language models, demonstrating both its robustness and
adaptability.


Table 2: Different Textual Encoder experiment result on TRANSFIR.


**PLM** ICEWS14 ICEWS18 GDELT


Baseline 0.2273 0.1797 0.1131


T5 0.3057 0.2061 0.2082


RoBERTa 0.2934 0.1939 0.2289


Qwen3 0.2567 0.2009 0.2030


BERT 0.3246 0.2324 0.2278


**Model Efficiency in GPU Memory and Training Time.** We evaluate the efficiency of TRANSFIR in terms of the runtime and GPU memory usage, comparing it against several strong baselines,
including HisRes and LogCL, on the ICEWS14 dataset. As shown in Fig. 7, TRANSFIR achieves
significantly lower peak GPU memory usage while maintaining competitive training speed, demonstrating strong efficiency and scalability. This suggests that TRANSFIR is not only effective in
performance but also efficient in resource utilization, making it scalable for large-scale datasets and
suitable for long-term applications in temporal knowledge graphs (TKGs). This efficiency is crucial
for real-world deployment, where both computational resources and time are limited.


6 CONCLUSION


In this work, we introduce TRANSFIR, a novel inductive reasoning framework designed to handle
emerging entities in temporal knowledge graphs. By leveraging transferable reasoning patterns and
utilizing an interaction-aware codebook, TRANSFIR effectively bridges the gap for emerging entities in the absence of historical interactions. Experimental results demonstrate that TRANSFIR
outperforms strong baselines across multiple benchmarks, with a significant improvement in MRR
on four datasets, showcasing its ability to perform effective inductive reasoning on emerging entities.


In the future, we plan to improve TRANSFIR by enhancing entity textual embeddings through external knowledge and LLMs to handle noisy or sparse entity descriptions. We also aims to extend
TRANSFIR to handle emerging relations and explore its application in more open-world scenarios.
Furthermore, we will investigate methods to model long-term evolution of entity semantics, enabling
TRANSFIR to adapt to changing knowledge over time.


ACKNOWLEDGMENTS


The authors of this paper were supported by National Natural Science Foundation of China (No.
92579211, 62272301, T2421002, 62525209, 62432002), Postdoctoral Fellowship Program of CPSF
under Grant Number No. GZB20250806, the AI for Science Seed Program of Shanghai Jiao
Tong University (project number 2025AI4S-QY01) and Natural Science Foundation of Shanghai
No.21TQ1400214.


10


7 ETHICS STATEMENT


This work complies with the ICLR Code of Ethics. Our research involves no human subjects,
sensitive personal data, or experiments that could cause harm to individuals, communities, or the
environment. All datasets utilized are publicly available and commonly used within the research
community. The methods introduced in this work are intended for general machine learning research
purposes and present no foreseeable risks of misuse or harmful applications. To the best of our
knowledge, this study raises no conflicts of interest or ethical concerns.


8 REPRODUCIBILITY STATEMENT


We have taken concrete steps to ensure the reproducibility of our work. The full implementation
details of our models, training setup, and baselines are provided in Appendix E. To further enhance reproducibility, we make our code publicly available at the following anonymous repository:
[https://github.com/zhaodazhuang2333/TransFIR. These resources should enable independent verifi-](https://github.com/zhaodazhuang2333/TransFIR)
cation of our results.


REFERENCES


Theodore Wilbur Anderson, Theodore Wilbur Anderson, Theodore Wilbur Anderson,
Theodore Wilbur Anderson, and Etats-Unis Math´ematicien. _An introduction to multivariate sta-_
_tistical analysis_, volume 2. Wiley New York, 1958.


Elizabeth Boschee, Jennifer Lautenschlager, Sean O’Brien, Steve Shellman, James Starz, and
[Michael Ward. ICEWS Coded Event Data, 2015. URL https://doi.org/10.7910/DVN/](https://doi.org/10.7910/DVN/28075)
[28075.](https://doi.org/10.7910/DVN/28075)


Borui Cai, Yong Xiang, Longxiang Gao, He Zhang, Yunfeng Li, and Jianxin Li. Temporal knowledge graph completion: a survey. In _Proceedings of the Thirty-Second International Joint Con-_
_ference on Artificial Intelligence_, pp. 6545–6553, 2023.


Jiajun Chen, Huarui He, Feng Wu, and Jie Wang. Topology-aware correlations between relations
for inductive link prediction in knowledge graphs. In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_
_artificial intelligence_, volume 35, pp. 6271–6278, 2021.


Mingyang Chen, Wen Zhang, Yushan Zhu, Hongting Zhou, Zonggang Yuan, Changliang Xu, and
Huajun Chen. Meta-knowledge transfer for inductive knowledge graph embedding. In _Proceed-_
_ings of the 45th international ACM SIGIR conference on research and development in information_
_retrieval_, pp. 927–937, 2022.


Wei Chen, Huaiyu Wan, Yuting Wu, Shuyuan Zhao, Jiayaqi Cheng, Yuxin Li, and Youfang Lin.
Local-global history-aware contrastive learning for temporal knowledge graph reasoning. In _2024_
_IEEE 40th International Conference on Data Engineering (ICDE)_, pp. 733–746. IEEE, 2024.


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding. In _Proceedings of the 2019 conference of_
_the_ _North_ _American_ _chapter_ _of_ _the_ _association_ _for_ _computational_ _linguistics:_ _human_ _language_
_technologies, volume 1 (long and short papers)_, pp. 4171–4186, 2019.


Zifeng Ding, Heling Cai, Jingpei Wu, Yunpu Ma, Ruotong Liao, Bo Xiong, and Volker Tresp. zrllm: Zero-shot relational learning on temporal knowledge graphs with large language models. In
_Proceedings_ _of_ _the_ _2024_ _conference_ _of_ _the_ _North_ _American_ _chapter_ _of_ _the_ _association_ _for_ _com-_
_putational linguistics:_ _Human language technologies (Volume 1:_ _Long papers)_, pp. 1877–1895,
2024.


Zhiyu Fang, Shuai-Long Lei, Xiaobin Zhu, Chun Yang, Shi-Xue Zhang, Xu-Cheng Yin, and Jingyan
Qin. Transformer-based reasoning for learning evolutionary chain of events on temporal knowledge graph. In _Proceedings_ _of_ _the_ _47th_ _international_ _ACM_ _SIGIR_ _conference_ _on_ _research_ _and_
_development in information retrieval_, pp. 70–79, 2024.


11


Mikhail Galkin, Xinyu Yuan, Hesham Mostafa, Jian Tang, and Zhaocheng Zhu. Towards foundation
models for knowledge graph reasoning. In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_
_Representations_, 2024.


Alberto Garcia-Duran, Sebastijan Dumanˇci´c, and Mathias Niepert. Learning sequence encoders
for temporal knowledge graph completion. In _Proceedings of the 2018 Conference on Empirical_
_Methods in Natural Language Processing_, pp. 4816–4821, 2018.


Hamid Hadipour, Yan Yi Li, Yan Sun, Chutong Deng, Leann Lac, Rebecca Davis, Silvia T Cardona,
and Pingzhao Hu. Graphban: An inductive graph-based approach for enhanced prediction of
compound-protein interactions. _Nature Communications_, 16(1):2541, 2025.


Hengchang Hu, Wei Guo, Xu Liu, Yong Liu, Ruiming Tang, Rui Zhang, and Min-Yen Kan. User
behavior enriched temporal knowledge graphs for sequential recommendation. In _Proceedings of_
_the 17th ACM International Conference on Web Search and Data Mining_, pp. 266–275, 2024.


Woojeong Jin, Meng Qu, Xisen Jin, and Xiang Ren. Recurrent event network: Autoregressive
structure inference over temporal knowledge graphs. _arXiv preprint arXiv:1904.05530_, 2019.


Li Jing, Pascal Vincent, Yann LeCun, and Yuandong Tian. Understanding dimensional collapse in
contrastive self-supervised learning. In _10th International Conference on Learning Representa-_
_tions, ICLR 2022_, 2022.


Timoth´ee Lacroix, Guillaume Obozinski, and Nicolas Usunier. Tensor decompositions for temporal
knowledge base completion. In _International Conference on Learning Representations_, 2020.


Dong-Ho Lee, Kian Ahrabian, Woojeong Jin, Fred Morstatter, and Jay Pujara. Temporal knowledge
graph forecasting without knowledge using in-context learning. _arXiv preprint arXiv:2305.10613_,
2023a.


Jaejun Lee, Chanyoung Chung, and Joyce Jiyoung Whang. Ingram: Inductive knowledge graph
embedding via relation graphs. In _International_ _conference_ _on_ _machine_ _learning_, pp. 18796–
18809. PMLR, 2023b.


Kalev Leetaru and Philip A Schrodt. Gdelt. In _ISA annual convention_, volume 2, pp. 1–49. Citeseer,
2013.


Zixuan Li, Xiaolong Jin, Wei Li, Saiping Guan, Jiafeng Guo, Huawei Shen, Yuanzhuo Wang, and
Xueqi Cheng. Temporal knowledge graph reasoning based on evolutional representation learning.
In _Proceedings of the 44th international ACM SIGIR conference on research and development in_
_information retrieval_, pp. 408–417, 2021.


Zixuan Li, Zhongni Hou, Saiping Guan, Xiaolong Jin, Weihua Peng, Long Bai, Yajuan Lyu, Wei Li,
Jiafeng Guo, and Xueqi Cheng. Hismatch: Historical structure matching based temporal knowledge graph reasoning. In _Findings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_ _EMNLP_
_2022_, pp. 7328–7338, 2022.


Ke Liang, Lingyuan Meng, Meng Liu, Yue Liu, Wenxuan Tu, Siwei Wang, Sihang Zhou, Xinwang
Liu, Fuchun Sun, and Kunlun He. A survey of knowledge graph reasoning on graph types: Static,
dynamic, and multi-modal. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, 46
(12):9456–9478, 2024.


Ruotong Liao, Xu Jia, Yangzhe Li, Yunpu Ma, and Volker Tresp. Gentkg: Generative forecasting
on temporal knowledge graph with large language models. In _Findings_ _of_ _the_ _Association_ _for_
_Computational Linguistics:_ _NAACL 2024_, pp. 4303–4317, 2024.


Shuwen Liu, Bernardo Grau, Ian Horrocks, and Egor Kostylev. Indigo: Gnn-based inductive knowledge graph completion using pair-wise encoding. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 34:2034–2045, 2021.


Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining
approach. _arXiv preprint arXiv:1907.11692_, 2019.


12


Yushan Liu, Yunpu Ma, Marcel Hildebrandt, Mitchell Joblin, and Volker Tresp. Tlogic: Temporal
logical rules for explainable link forecasting on temporal knowledge graphs. In _Proceedings_ _of_
_the AAAI conference on artificial intelligence_, volume 36, pp. 4120–4127, 2022.


Xin Mei, Libin Yang, Xiaoyan Cai, and Zuowei Jiang. An adaptive logical rule embedding model
for inductive reasoning over temporal knowledge graphs. In _Proceedings of the 2022 Conference_
_on Empirical Methods in Natural Language Processing_, pp. 7304–7316, Abu Dhabi, United Arab
Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.
emnlp-main.493.


Shi Mingcong, Chunjiang Zhu, Detian Zhang, Shiting Wen, and Li Qing. Multi-granularity history
and entity similarity learning for temporal knowledge graph reasoning. In _Proceedings_ _of_ _the_
_2024 Conference on Empirical Methods in Natural Language Processing_, pp. 5232–5243, 2024.


Jiaxin Pan, Mojtaba Nayyeri, Osama Mohammed, Daniel Hernandez, Rongchuan Zhang, Cheng
Cheng, and Steffen Staab. Towards foundation model on temporal knowledge graph reasoning.
_arXiv preprint arXiv:2506.06367_, 2025.


Marco Postiglione, Daniel Bean, Zeljko Kraljevic, Richard JB Dobson, and Vincenzo Moscato.
Predicting future disorders via temporal knowledge graphs and medical ontologies. _IEEE Journal_
_of Biomedical and Health Informatics_, 28(7):4238–4248, 2024.


Ye Qian, Xiaoyan Wang, Fuhui Sun, and Li Pan. Compressing transfer: Mutual learning-empowered
knowledge distillation for temporal knowledge graph reasoning. _IEEE_ _Transactions_ _on_ _Neural_
_Networks and Learning Systems_, 2025.


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer. _Journal of machine learning research_, 21(140):1–67, 2020.


Chao Shang, Yun Tang, Jing Huang, Jinbo Bi, Xiaodong He, and Bowen Zhou. End-to-end structureaware convolutional networks for knowledge base completion. In _Proceedings of the AAAI con-_
_ference on artificial intelligence_, volume 33, pp. 3060–3067, 2019.


Komal Teru, Etienne Denis, and Will Hamilton. Inductive relation prediction by subgraph reasoning.
In _International conference on machine learning_, pp. 9448–9457. PMLR, 2020.


Christos Thrampoulidis, Ganesh Ramachandra Kini, Vala Vakilian, and Tina Behnia. Imbalance
trouble: Revisiting neural-collapse geometry. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Sys-_
_tems_, 35:27225–27238, 2022.


Rakshit Trivedi, Hanjun Dai, Yichen Wang, and Le Song. Know-evolve: Deep temporal reasoning
for dynamic knowledge graphs. In _international conference on machine learning_, pp. 3462–3471.
PMLR, 2017.


Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha P Talukdar. Composition-based multirelational graph convolutional networks. In _ICLR_, 2020.


Jianling Wang, Haokai Lu, James Caverlee, Ed H Chi, and Minmin Chen. Large language models
as data augmenters for cold-start item recommendation. In _Companion Proceedings of the ACM_
_Web Conference 2024_, pp. 726–729, 2024a.


Jiapu Wang, Sun Kai, Linhao Luo, Wei Wei, Yongli Hu, Alan Wee-Chung Liew, Shirui Pan, and
Baocai Yin. Large language models-guided dynamic adaptation for temporal knowledge graph
reasoning. _Advances in Neural Information Processing Systems_, 37:8384–8410, 2024b.


Siheng Xiong, Yuan Yang, Faramarz Fekri, and James Clayton Kerce. Tilp: Differentiable learning
of temporal logical rules on knowledge graphs. _arXiv preprint arXiv:2402.12309_, 2024a.


Siheng Xiong, Yuan Yang, Ali Payani, James C Kerce, and Faramarz Fekri. Teilp: Time prediction
over knowledge graphs via logical reasoning. In _Proceedings of the AAAI conference on artificial_
_intelligence_, volume 38, pp. 16112–16119, 2024b.


13


Wenjie Xu, Ben Liu, Miao Peng, Xu Jia, and Min Peng. Pre-trained language model with prompts
for temporal knowledge graph completion. _arXiv preprint arXiv:2305.07912_, 2023a.


Yi Xu, Junjie Ou, Hui Xu, and Luoyi Fu. Temporal knowledge graph reasoning with historical contrastive learning. In Brian Williams, Yiling Chen, and Jennifer Neville (eds.), _Thirty-_
_Seventh_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence,_ _AAAI_ _2023,_ _Thirty-Fifth_ _Conference_ _on_ _In-_
_novative_ _Applications_ _of_ _Artificial_ _Intelligence,_ _IAAI_ _2023,_ _Thirteenth_ _Symposium_ _on_ _Educa-_
_tional_ _Advances_ _in_ _Artificial_ _Intelligence,_ _EAAI_ _2023,_ _Washington,_ _DC,_ _USA,_ _February_ _7-14,_
_2023_, pp. 4765–4773. AAAI Press, 2023b. doi: 10.1609/AAAI.V37I4.25601. URL [https:](https://doi.org/10.1609/aaai.v37i4.25601)
[//doi.org/10.1609/aaai.v37i4.25601.](https://doi.org/10.1609/aaai.v37i4.25601)


Chao Xue, Di Liang, Pengfei Wang, and Jing Zhang. Question calibration and multi-hop modeling
for temporal question answering. In _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelli-_
_gence_, volume 38, pp. 19332–19340, 2024.


Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and St´ephane Deny. Barlow twins: Self-supervised
learning via redundancy reduction. In _International conference on machine learning_, pp. 12310–
12320. PMLR, 2021.


Jinchuan Zhang, Ming Sun, Chong Mu, Jinhao Zhang, Quanjiang Guo, and Ling Tian. Historically
relevant event structuring for temporal knowledge graph reasoning. In _2025 IEEE 41st Interna-_
_tional Conference on Data Engineering (ICDE)_, pp. 3179–3192. IEEE, 2025a.


Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 embedding: Advancing text embedding and reranking through foundation models. _arXiv preprint arXiv:2506.05176_,
2025b.


Cunchao Zhu, Muhao Chen, Changjun Fan, Guangquan Cheng, and Yan Zhang. Learning from
history: Modeling temporal knowledge graphs with sequential copy-generation networks. In
_Proceedings of the AAAI conference on artificial intelligence_, volume 35, pp. 4732–4740, 2021.


14


A LLM USAGE DISCLOSURE


In this work, we used GPT-4o to assist with grammar checking and polishing. All LLM-generated
content was thoroughly reviewed and validated by the authors to ensure the accuracy of the presented information. Additionally, the items representing ”President” and ”Government” in Fig. 1
and Fig. 2 are generated by GPT for illustrative purposes. The use of LLM aligns with ICLR’s
ethical guidelines, and all contributions from the LLM have been transparently acknowledged and
reviewed to avoid any false or misleading statements.


B RELATED WORK


**Reasoning on Temporal Knowledge Graphs.** Reasoning on temporal knowledge graphs (TKGs)
aims to infer missing or future facts by modeling temporal evolution. Prior work falls into interpolation (filling unobserved facts within the time window) and extrapolation (forecasting beyond
training) Cai et al. (2023). Representative interpolation approaches extend static models with temporal mechanisms: TTransE introduces temporal constraints across adjacent facts Garcia-Duran
et al. (2018); TNTComplEx employs a fourth-order tensor for time-aware entity/relation embeddings Lacroix et al. (2020); TEILP parameterizes temporal logical rules with neural modules Xiong
et al. (2024b).


In contrast, the extrapolation setting focuses on predicting future events using only historical interactions, without access to future information during training. Extrapolative methods typically
aggregate historical interactions and capture cross-time dependencies: CyGNet uses time-aware
copy mechanisms for recurrence Zhu et al. (2021); CENET applies contrastive learning to disentangle historical vs. non-historical influences Xu et al. (2023b); LogCL blends local and global
temporal context Chen et al. (2024). More recent approaches leverage Transformer architectures
and large language models (LLMs): GenTKG combines retrieval-augmented generation with instruction tuning Liao et al. (2024); LLM-DA dynamically updates temporal rules for domain adaptation Wang et al. (2024b); and ECEformer encodes chronological event chains using a Transformer
structure Fang et al. (2024).


Despite these advances, a key challenge remains unaddressed: handling emergent entities that appear during graph evolution. Current methods operate under a closed-world assumption and typically initialize the embeddings of all new entities randomly. As these emerging entities lack any
historical interactions during the training phase, the absence of sufficient supervision often results
in representation collapse.


**Knowledge** **Graph** **Inductive** **Learning.** Inductive learning on static KGs aims to generalize to
unseen entities/relations, or even entirely new graphs without retraining under a fixed vocabulary.
Classical approaches such as GraIL Teru et al. (2020) and TACT Chen et al. (2021) reason from
local subgraph structure and relational patterns, reducing reliance on pre-learned entity embeddings.
Recent work strengthens inductive reach from complementary angles: INDIGO enables fully inductive link prediction directly from GNN outputs Liu et al. (2021); MorsE employs meta-learning to
transfer knowledge for initializing unseen-entity embeddings Chen et al. (2022); InGram integrates
relation-aware attention to better handle novel relations Lee et al. (2023b); and ULTRA learns conditional relational representations for zero-shot generalization across different graphs Galkin et al.
(2024).


However, these methods are designed for static KGs, where new entities typically possess at least
some known relations. Few works focus on inductive reasoning in TKGs. ALRE-IR (Mei et al.,
2022) combines embedding-based and logical rule-based methods to capture deep causal logic,
demonstrating strong zero-shot reasoning capabilities. zrLLM (Ding et al., 2024) leverages large
language models to generate relation representations from text descriptions, enabling reasoning for
unseen relations. POSTRA (Pan et al., 2025) enables cross-dataset knowledge transfer through sinusoidal positional encoding.


Despite these advances, they overlook the fact that emerging entities in temporal knowledge graphs
often arrive without any historical interactions, a common scenario in real-world applications. The
absence of relational context makes it particularly challenging to derive meaningful representations
for such entities.


15


C EMPIRICAL STUDIES


C.1 VISUALIZATION DETAILS FOR **Q2**


For the visualization study, we adopt **LogCL** as the base model. We record entity embeddings at two
stages: (i) _init_, right after model initialization, and (ii) _trained_, after convergence on the training set.
Entities are categorized as _known_ if they are present in the training data, and as _emerging_ otherwise.


All embeddings are first reduced to 50 dimensions via PCA and then projected into a 2D space using
_t_ -SNE (perplexity=30, 2000 iterations). The main text presents visualization results on ICEWS14,
while additional plots for other datasets are provided in Appendix F.2 for comparison with TRANSFIR. Known and emerging entities are distinguished by color to facilitate comparative analysis.


C.2 REPRESENTATION COLLAPSE AND COLLAPSE RATIO


**Representation** **Collapse.** In representation learning, _collapse_ refers to a degradation in the expressiveness of the embedding space, where multiple input instances are mapped to (approximately)
identical points or confined to a low-rank subspace. This phenomenon typically manifests as vanishing variance along principal directions, rank deficiency, or excessively homogeneous node representations in graph models Thrampoulidis et al. (2022); Jing et al. (2022). Common causes include
inadequate supervision, degenerate learning objectives, or limited contextual information. As a result, collapsed representations exhibit poor separability and diminished generalization performance.


**Collapse Ratio.** In TKGs, emerging entities arrive with no historical interactions, so their learning signal is under-constrained and easily pulled toward generic priors. To quantify this, let
_X_ = _{zi}_ _[n]_ _i_ =1 _[⊂]_ [R] _[d]_ [be] [a] [centered] [set of embeddings with covariance estimator][ Σ] _[X]_ [.] [We measure]
dispersion via the _generalized variance_ (the geometric mean of principal-axis standard deviations)


GS( _X_ ) =             - det Σ _X_             - 2 [1] _d,_


which decreases whenever variance collapses along any eigen-direction and is rotation-invariant Anderson et al. (1958); Zbontar et al. (2021). For numerical stability when _n_ _<_ _d_ or directions are
nearly collinear, we compute log det(Σ _X_ ) from the (nonnegative) eigenvalues of Σ _X_ . Given an
_emerging_ set _X_ emerg and a _reference_ set _X_ ref (e.g., the set of known entities), we define


CR = [GS(] _[X]_ [emerg][)]

GS( _X_ ref ) _[.]_


Values _<_ 1 indicate collapse (e.g., CR = 0 _._ 2 means the average per-axis scale is 5 _×_ smaller). Because GS summarizes the available variance across all informative directions, lower Collapse Ratio
corresponds to reduced separability and weaker discriminative capacity of emerging-entity representations. We report Collapse Ratio alongside t-SNE visuals as quantitative evidence of representation
collapse.


D ADDITIONAL DETAILS OF METHODOLOGY


D.1 CHAIN STRUCTURE MOTIVATION


In this section, we provide additional motivation for modeling historical interactions as **Interaction**
**Chains (ICs)** .


**Sequential nature of temporal reasoning.** Reasoning over temporal knowledge graphs often involves sequential dependencies akin to multi-step inference paths. For example, consider an entity representing a person with interactions such as _“visited Country A at t_ 1 _”_, followed by _“visited_
_Country B at t_ 2 _”_ . At a later time _tq_, predicting that this person may _“visit Country C”_ often depends
on the sequential chain of prior visits, rather than an unordered set of neighbors. Such sequential
dynamics are difficult to capture when historical interactions are aggregated as a bag-of-events.


16


**Entity-invariant temporal patterns.** Many chains reflect patterns that are largely _entity-invariant_
(e.g., _successive state visits_ ). Organizing history into chains exposes such transferable regularities,
enabling generalization to _emerging entities_ with no prior representations; in contrast, updating static
embeddings tends to overfit well-observed entities and fails to extrapolate.


**Benefit** **of** **chain** **formulation.** By preserving the temporal order of events, the chain formulation naturally captures the progression of interaction dynamics, making it particularly suitable for
inductive temporal reasoning. Our proposed Interaction Chain (IC) design offers a principled approach to extracting reusable temporal patterns directly from raw interaction logs, thereby forming
the foundation of our framework.


D.2 ALGORITHM FLOW AND PSEUDOCODE


Here we include a detailed pseudocode of our framework TRANSFIR, covering the Classification–Representation–Generalization pipeline.


**Initialization** . Entity textual embeddings _{_ **h** _e}e∈E_ are obtained with a pretrained BERT encoder
and kept _frozen_ . Learnable parameters include relation embeddings _{_ **h** _r}r∈R_, the IC encoder Θenc
(component-wise MLPs, Transformer, query-aware attention), the VQ codebook _C_ = _{_ **c** _k}_ _[K]_ _k_ =1 [, the]
drift MLP Ψ, and the scoring module _f_ ( _·_ ) (ConvTransE). Training and inference proceed strictly in
chronological order.


**Training-time** **Flow** **(per** **timestamp)** At each timestamp _t_ with query set _Qt_ = _{_ ( _es, r,_ ? _, t_ ) _}_,
TRANSFIR executes:


(i) **Classification** - quantize frozen **h** _e_ to the nearest codeword in _C_ ; get VQ losses _L_ codebook;

(ii) **Representation** - build and encode an IC for each query _q_, yielding **h** [IC] _eq_ [;]

(iii) **Generalization** - form cluster-level dynamic prototypes _{_ **c** [dyn] _k_ _[}]_ [ by pooling] _[ {]_ **[h]** _e_ [IC] _[}]_ [ per cluster]
of the _query entity_ ; propagate _temporal transfer_ to non-query entities via **h** [˜] _e_ = **h** _e_ +Ψ([ **h** _e∥_ **c** [dyn] _π_ ( _e_ ) [])] _[·]_

**c** [dyn] _π_ ( _e_ ) [.]


(iv) **Ranking & Loss** - score candidates with ConvTransE and optimize _L_ = _L_ lp + _λL_ codebook. For
implementation details and the step-by-step routine, please refer to Alg. 1.


D.3 COMPLEXITY ANALYSIS


We analyze the time and space complexity of TRANSFIR per timestamp _t_ . Let _nt_ = _|Qt|_ be the
number of queries at _t_, _k_ the Interaction Chain length (Top- _k_ ), _d_ the hidden size, _L_ the number
of Transformer layers, _K_ the codebook size, _m_ the hidden width of the drift MLP, and _E_ = _|E|_,
_R_ = _|R|_ .


**Codebook (classification).** Vector-quantized assignment has worst-case time _O_ ( _EKd_ ) per update
(nearest-prototype search) and space _O_ ( _Kd_ ) for the codebook. Because entity text embeddings are
frozen, assignments can be cached and updated lazily; thus the amortized assignment cost is small
relative to encoding.


**IC** **construction(representation).** IC construction keeps a bounded chain of length _k_ for each
query, yielding time _O_ ( _ntkd_ ) for token projections. The Transformer encoder dominates with


_O_           - _nt L_ ( _k_ [2] _d_ + _kd_ [2] )�


(attention and feed-forward), and memory _O_ ( _ntkd_ ) for activations.


**Pattern** **transfer** **(generalization).** Forming cluster prototypes requires _O_ ( _ntd_ + _Kd_ ). Broadcasting drift via the MLP costs _O_ ( _Emd_ ) with space _O_ ( _Ed_ ) for (temporary) updated embeddings.
In practice we apply drift only to non-query entities at _t_ .


17


**Algorithm 1** TRANSFIR Training (per epoch, chronological)
**Require:** Train timestamps _{_ 1 _, . . ., t_ train _}_ ; frozen entity embeddings _{_ **h** _e}e∈E_ ; learnable _{_ **h** _r}r∈R_,
IC encoder Θenc (MLPs+Transformer+attn), VQ codebook _C_ = _{_ **c** _k}_ _[K]_ _k_ =1 [,] [transfer] [MLP] [Ψ][,]
scorer _f_ (ConvTransE); window _T_, Top- _k_ .
1: **for** epoch = 1 _,_ 2 _, . . ._ **do**
2: **for** timestamp _t_ = 1 to _t_ train **do**
3: _Qt_ _←{_ ( _eq, rq,_ ? _, t_ ) _}_ _▷_ All queries at time _t_


4: **(1) Codebook Mapping (Classification)**
5: **for** each _e ∈E_ **do**
6: _π_ ( _e_ ) _←_ arg min _k ∥_ **h** _e −_ **c** _k∥_ [2] 2 _▷_ VQ assignment
7: **end for**

2 2

8: _L_ cb _←_ [�] _e_ ��sg[ **h** _e_ ] _−_ **c** _π_ ( _e_ )�� [;] _L_ commit _←_ [�] _e_ �� **h** _e −_ sg[ **c** _π_ ( _e_ )]��


2
_e_ ��sg[ **h** _e_ ] _−_ **c** _π_ ( _e_ )��2 [;] _L_ commit _←_ [�]


2 2

8: _L_ cb _←_ [�] _e_ ��sg[ **h** _e_ ] _−_ **c** _π_ ( _e_ )��2 [;] _L_ commit _←_ [�] _e_ �� **h** _e −_ sg[ **c** _π_ ( _e_ )]��2

9: _L_ codebook _←_ _α L_ cb + _β L_ commit


10: **(2) IC Encoding (Representation)**
11: **for** each _q_ = ( _eq, rq,_ ? _, t_ ) _∈Qt_ **do**
12: _Cq ←{_ ( _si, ri, oi, ti_ ) _| t −_ _T_ _≤_ _ti_ _< t,_ _eq ∈{si, oi}}_

13: _Cq_ [(] _[k]_ [)] _←_ TopK _i_ �sim( **h** _rq_ _,_ **h** _ri_ ) _,_ _Cq_ - _▷_ cosine sim

14: Encode _Cq_ [(] _[k]_ [)] with Θenc; relation-guided attn _⇒_ **h** [IC] _eq_
15: **end for**


16: **(3) Temporal Pattern Transfer (Generalization)**


1
17: Group _{_ **h** [IC] _eq_ _[}]_ [ by] _[ π]_ [(] _[e][q]_ [)][; for] _[ k]_ [=1] _[. . .K]_ [:] **[c]** [dyn] _k_ _←_
_|Qk|_


 - **h** [IC] _eq_

_eq_ : _π_ ( _eq_ )= _k_


18: _St_ _←{eq_ _|_ ( _eq, rq,_ ? _, t_ ) _∈Qt}_ _▷_ Query entities at time _t_
19: **for** each _e ∈E_ **do**
20: _ze_ _←_ [ **h** _e ∥_ **c** [dyn] _π_ ( _e_ ) []][;] _ωe_ _←_ Ψ( _ze_ )

21: **h** ˆ _e_ _←_ **h** _e_ + _ωe ·_ **c** [dyn] _π_ ( _e_ )
22: **end for**


23: **Ranking & Loss**
24: _L_ lp _←_ 0
25: **for** each _q_ = ( _es, rq,_ ? _, t_ ) _∈Qt_ **do**
26: Score all (or sampled) _eo_ : _ϕ_ ( _es, rq, eo, t_ ) = _σ_ - _f_ ( **h** [ˆ] _es_ _,_ **h** _rq_ _,_ **h** [ˆ] _eo_ )�

27: _L_ lp += _−_ log softmax _eo_ - _ϕ_ ( _es, rq, eo, t_ )�

28: **end for**
29: **Update** by backprop on _L_ = _L_ lp + _λ L_ codebook; update _{_ **h** _r},_ Θenc _,_ Ψ _, f, C_
30: **end for**
31: **end for**


**Overall.** Ignoring the shared scoring cost, the dominant _model-specific_ complexity of TRANSFIR
per timestamp is

_O_       - _nt L_ ( _k_ [2] _d_ + _kd_ [2] )� + _O_ ( _EKd_ ) + _O_ ( _Emd_ )


Since _k_, _L_, _K_, and _m_ are small constants (e.g., _k_ _≤_ 32), TRANSFIR scales _linearly_ with the number
of queries and entities, and its controllable chain length avoids dependence on the full neighborhood
size.


E ADDITIONAL EXPERIMENTAL SETTINGS


E.1 DETAILED DATASET INFORMATION


Table E.1 presents comprehensive statistics for all datasets, encompassing entity counts, relation
counts, fact counts, and the proportion of emerging entities in validation and test splits. We uti

18


lize four temporal event datasets spanning crisis early - warning contexts and diverse global event
landscapes to evaluate the model’s multi - dimensional performance.


Table 3: Statistics of all datasets, including ICEWS14, ICEWS18, ICEWS05-15 and GDELT.


**Dataset** **Entities** **Relation** **Time Snapshots** **Total Triples** **Emerging Entities**


ICEWS14 7128 230 365 90730 1301


ICEWS18 23033 256 304 468558 3434


ICEWS05-15 10488 251 4017 461329 1954


GDELT 7691 240 2976 2278405 1020


**•** **ICEWS14(Trivedi** **et** **al.** **(2017)):** A subset of the Integrated Crisis Early Warning System
(ICEWS) dataset for 2014, focusing on short-term conflict events within a single year. After preprocessing (e.g., entity standardization, confidence filtering), it contains 8 high-frequency event types
(e.g., protests, attacks). It is used to evaluate the model’s performance in local temporal window
event prediction.


**• ICEWS18 (Boschee et al. (2015)):** The 2018 ICEWS dataset, maintaining the core focus on crisis
events but introducing emerging subtypes (e.g., ”economic sanctions”) to reflect modern conflict
dynamics. It tests the model’s cross-year stability and adaptability to emerging event types.


**• ICEWS05-15(Jin et al. (2019)):** A long-term crisis dataset covering 2005–2015, including historical events such as financial crises and regional conflicts. Characterized by sparse daily events and
a large time span, it serves as the primary training set to validate the model’s long-term temporal
dependency modeling and generalization under low-resource scenarios.


**• GDELT(Leetaru & Schrodt (2013)):** The Global Database of Events, Language, and Tone, covering political, economic, and cultural events beyond crises. It complements ICEWS by including
non-conflict scenarios, enabling validation of the model’s cross-domain generalization and utilization of multi-dimensional information.


E.2 BASELINES (OVERVIEW AND IMPLEMENTATION)


**Families.** **Graph-based** (temporal GNN/embedding; mostly transductive), **Path-based** (querycentered relational paths or reasoning rules), and **Static** **inductive** (inductive graph learning but
without temporal encoder).


**Implementation.** We follow chronological splits (5:2:3) consistent with the main paper. For
**Graph-based** and **Path-based** methods, we keep the original settings and only adjust the temporal
split and test set to fit the emerging-entity evaluation. For rule-mining approaches (e.g., _TILP_ ) with
high search complexity, we reduce the maximum rule length from 5 to **3 (ICEWS)** and **2 (GDELT)**
to control computation while preserving the core mechanism. For **Static inductive** methods, which
assume a static graph, we merge a small window of timestamps (e.g., **7** ) into a subgraph to run, and
we inject relative time into features to enable comparison under the same prediction protocol.


**Baseline briefs.**


**CyGNet** [GRAPH] Zhu et al. (2021). Sequential copy-generation with a time-aware dual-mode inference to predict recurrent and de-novo events.


**REGCN** [GRAPH] Li et al. (2021). Recurrent GCN that learns evolving entity/relation states by
capturing temporal–structural patterns and injecting static constraints.


**HiSMatch** [GRAPH] Li et al. (2022). Historical structure matching with entity/relation/time semantics and sequential cues; background knowledge improves matching.


**MGESL** [GRAPH] Mingcong et al. (2024). TKG reasoning model combines multi-granularity history and entity similarity via hypergraph convolution, includes candidate-known setting.


19


**LogCL** [GRAPH] Chen et al. (2024). Local–global contrastive learning with entity-aware attention
to mine query-relevant histories and suppress noise.

**HisRes** [GRAPH] Zhang et al. (2025a). Historically relevant event structuring with multi-granularity
evolution and global relevance encoders, fused adaptively.

**MLEMKD** [GRAPH] Qian et al. (2025). Mutual-learning KD for temporal KGs using soft-label
filtering and adaptive distillation to curb anomaly diffusion with minimal drop.

**TLogic** [PATH] Liu et al. (2022). Time-constrained random-walk rule mining that yields timeconsistent explanations and competitive forecasting.

**TLIP** [PATH] Xiong et al. (2024a). Differentiable temporal rule learner extracting interpretable
patterns via constrained walks and temporal features.

**ECEformer** [PATH] Fang et al. (2024). Transformer over Evolutionary Chains of Events with intraquadruple representation and inter-quadruple context mixing.

**GenTKG** [PATH] Liao et al. (2024). Retrieval-augmented generation: temporal rule retrieval +
few-shot instruction tuning for LLM-based forecasting.

**CompGCN** [INDUCTIVE] Vashishth et al. (2020). Multi-relational GCN with relation composition
operators, unifying KG embedding tricks beyond plain graph conv.

**ICL** [INDUCTIVE]Lee et al. (2023a) TKG forecasting via in-context learning with LLMs requires no
fine-tuning or prior semantic knowledge and performs competitively on benchmarks.

**PPT** [INDUCTIVE]Xu et al. (2023a) TKG completion uses pre-trained LMs and time prompts, via
masked token prediction, with competitive benchmark results.

**MorsE** [INDUCTIVE] Chen et al. (2022). Meta-knowledge transfer that learns entity-agnostic structural priors for unseen entities via relation-aware initialization.

**InGram** [INDUCTIVE] Lee et al. (2023b). Inductive KG embedding using relation-affinity graphs
and attention-based aggregation to form embeddings for unseen nodes/relations.


E.3 EVALUATION


**Metrics** We use two standard metrics: Mean Reciprocal Rank (MRR) and Hits@K. MRR is defined as:


MRR = [1]

_N_


_N_


_i_ =1


1
_,_
_ri_


where _ri_ is the rank of the correct answer for the _i_ -th query. Hits@K measures the proportion of
queries for which the correct answer is ranked in the top _K_ .


**Experimental Setup** We evaluate all models on emerging entity-related quadruples using MRR,
Hits@3, and Hits@10. For inverse relation triples ( _eo, r_ _[−]_ [1] _, es, tq_ ), we also perform tests, and report
the average of both directions. During testing, we follow the same filtering strategy as LogCL Chen
et al. (2024), excluding quadruples involving the same query entity and relation at the same timestamp to avoid redundant results.


All experiments of TRANSFIR are conducted with three random seeds, and the reported results
are the averages across these runs. Detailed results are presented in Table 1. Note that GenTKG
generates 10 samples to compute Hits, so MRR values are not available for this method.


F EXTENDED EXPERIMENTAL RESULTS


F.1 REPRESENTATION AND LEARNING ANALYSIS (RQ2)


**Representation** **quality** **and** **collapse.** We further evaluate TRANSFIR’s ability to represent
emerging entities through t-SNE visualizations across multiple datasets. As illustrated in Fig. 8,
TRANSFIR consistently yields well-separated clusters in the embedding space, in contrast to
LogCL, which only distinguishes between emerging and known entities, resulting in a distribution shift between their embeddings. In comparison, our approach clearly groups emerging entities
into distinct latent semantic clusters. The Collapse Ratio is significantly improved across all four


20


datasets, underscoring the effectiveness of our VQ codebook and pattern transfer mechanism in
preventing representation collapse. This enhancement enables the model to produce informative
embeddings that support inductive reasoning for emerging entities.


Figure 8: t-SNE visualizations comparing LogCL and TRANSFIR on multiple datasets. The top
row shows LogCL embeddings, with a clear representation collapse for emerging entities (red).
The bottom row shows TRANSFIR, where emerging entities are well-separated into latent semantic
clusters, significantly improving the Collapse Ratio across all datasets.


**Failure** **case** **analysis.** To provide a deeper understanding of the model’s limitations,
we provide a failure case due to insufficient semantic information in dataset ICEWS14:
( _(Court Judge (Nigeria)_, INVESTIGATE, _(Bala Ngilari)_, _t_ 184 ). In this case, the emerging entity _Bala Ngi-_
_lari_ lacks sufficient semantic information in the textual input. Since TRANSFIR relies on semanticbased clustering to align emerging entities with known entities, the absence of meaningful textual
features prevents the model from assigning _Bala Ngilari_ to the correct latent semantic cluster. Consequently, the model fails to infer that _Court Judge (Nigeria)_ will investigate _Bala Ngilari_ at _t_ 184.


F.2 ABLATION: ADDITIONAL METRICS (RQ3)


Beyond Hits@10 reported in the main paper, we further evaluate the ablations on _MRR_ and _Hits@3_ .
As shown in Fig. 9, the qualitative conclusions remain unchanged across four benchmarks: (i) removing the **codebook mapping** yields the largest drop, confirming the importance of aligning entities into latent semantic clusters for reliable transfer; (ii) both **IC construction** and **pattern transfer**
contribute consistently; (iii) discarding **textual encoding** degrades performance, since text provides
a stable prior for emerging entities. Results are averaged over three random seeds; error bars denote
standard deviation.


F.3 GENERALIZATION TO THE UNKNOWN SETTING(RQ4)


**Definition.** We keep the temporal KG notation _G_ = _{Gt}t∈T_ with _Gt_ = ( _E_ 1: _t, R, Ft_ ). Let the
timeline be split into disjoint windows _T_ tr _, T_ val _, T_ te. For any window _W_ _⊂T_, define the entity set
_EW_ = _{ e | ∃_ ( _es, r, eo, t_ ) _∈Ft,_ _t_ _∈_ _W,_ _e_ _∈{es, eo} }_ . The **Unknown** entity set is
_E_ unk = _ET_ te _\_              - _ET_ tr _∪ET_ val� _._
During testing, we evaluate queries of the form ( _es, r,_ ? _, tq_ ) or (? _, r, eo, tq_ ), where _tq_ _∈T_ te and
the queried entity _e_ _∈E_ unk. Unlike the _Emerging_ setting (Sec. 2) which enforces _tq_ = _te_ ( _e_ ) (zero
history), the Unknown setting allows the model to _observe_ _local_ _pre-query_ _history_ _within_ _the_ _test_
_window_, defined as
_Ht_ [te] _q_ [=]              - _Fi,_

_i∈T_ te _, i<tq_

while future facts ( _≥_ _tq_ ) remain hidden.


21


Figure 9: Ablation results on four benchmarks under **MRR** (top row) and **Hits@3** (bottom row).
The ranking of variants mirrors the main-paper Hits@10.


Figure 10: Results on ICEWS18, ICEWS05-15, and GDELT under the _Unknown_ (blue; observable
_G<t_ ) vs. _Emerging_ (green; zero history) settings. All methods improve with pre-query history, and
TRANSFIR remains best on both MRR and Hits@10 across datasets.


**Relation** **to** **the** _**Emerging**_ **setting** We distinguish between two test settings. Let _T_ te denote the
test window, and let _Ht_ [te] _q_ [=][ �] _i<tq,,i∈T_ te _[F][i]_ [ represent the test-time history available prior to time] _[ t][q]_ [.]


_Emerging_ . In this setting, queries are restricted to the first appearance of an entity. For a target entity
_e_, the query time is set to _tq_ = _te_ ( _e_ ) (its emergence time). Consequently, _Ht_ [te] _q_ [contains] [no] [prior]
interactions involving _e_ (strict zero-history condition).


_Unknown_ . Here, entities are also unseen during training and validation. However, queries can occur
at any time _tq_ _>_ _te_ ( _e_ ) within the test window _T_ te. Therefore, _Ht_ [te] _q_ [may] [include] [earlier] [test-time]
interactions of _e_, providing a short local history. In practice, since an unseen entity can appear
multiple times during testing, we evaluate its predictions specifically at non-first occurrences. This
allows us to isolate the benefit of having limited test-time context.


**Experiment Results.** As shown in Fig. 10, across all datasets, every method achieves higher MRR
and Hits@10 scores in the _Unknown_ setting than in the _Emerging_ setting, confirming that even brief
interaction histories ( _G<t_ ) are beneficial. TRANSFIR consistently outperforms all baselines on every dataset and metric, maintaining a clear advantage even when test-time history is provided. This


22


0.165


0.160


0.155


0.150


ICEWS14


ICEWS14


ICEWS14


ICEWS14


0.165


0.160


0.155


0.150


|Col1|Col2|Col3|
|---|---|---|
|||MRR|


|Col1|Col2|Col3|
|---|---|---|
|||MRR|


|Col1|Col2|Col3|
|---|---|---|
|||MRR|


|Col1|Col2|Col3|
|---|---|---|
|||MRR|


GDELT
0.110


0.105


0.100


0.095


0.090


0.165


0.160


0.155


0.150


GDELT
0.110


0.105


0.100


0.095


0.090


0.165


0.160


0.155


0.150


0.33


0.32


0.31


0.30


10 25 50 100
_K_

ICEWS18
0.120


0.115


0.110


0.105


7 15 30 50
_k_

ICEWS18
0.120


0.115


0.110


0.105


256 512 768 1024
_d_

ICEWS18
0.120


ICEWS05-15


0.39


0.38


0.37


256 512 768 1024
_d_


2 3 4 6
_L_

ICEWS18
0.120


0.115


0.110


0.105


0.100


0.22


0.21


0.20


0.33


0.32


0.31


0.30


0.23


0.22


0.21


0.20


0.39


0.38


0.37


0.23


0.22


0.21


0.20


0.115


0.110


0.105


0.100


0.22


0.21


0.20


HITS@10


0.23


0.22


0.21


0.20


0.100


0.22


0.21


0.20


HITS@10


10 25 50 100
_K_

ICEWS05-15


10 25 50 100
_K_


0.100


0.22


0.21


0.20


HITS@10


7 15 30 50
_k_

ICEWS05-15


7 15 30 50
_k_


HITS@10


2 3 4 6
_L_

ICEWS05-15


2 3 4 6
_L_


256 512 768 1024
_d_


0.110


0.105


0.100


0.095


0.090


GDELT
0.110


0.105


0.100


0.095


0.090


0.33


0.32


0.31


0.30


0.23


0.22


0.21


0.20


0.39


0.38


0.37


0.23


0.22


0.21


0.20


0.23


0.22


0.21


0.20


0.33


0.32


0.31


0.30


0.23


0.22


0.21


0.20


0.39


0.38


0.37


0.23


0.22


0.21


0.20


|Col1|Col2|Col3|
|---|---|---|
|||MRR<br>HITS@10|


|Col1|Col2|Col3|
|---|---|---|
|||MRR<br>HITS@10|


|Col1|Col2|Col3|
|---|---|---|
|||MRR<br>HITS@10|


|Col1|Col2|Col3|
|---|---|---|
|||MRR<br>HITS@10|


10 25 50 100
_K_


7 15 30 50
_k_


256 512 768 1024
_d_


2 3 4 6
_L_


Figure 12: Hyperparameter study on four benchmarks, exploring effects of codebook size _K_, Interaction Chain length _k_, hidden dimension _d_, and the number of layers _l_ . Brown and green represent
MRR and HITS@10, respectively.


suggests that TRANSFIR effectively leverages both local historical patterns and type-level regularities, while baseline methods rely primarily on entity-specific history and still fall short. Overall,
these results demonstrate the robust inductive generalization capability of TRANSFIR: its performance gains do not hinge on the zero-history setup, and its superiority persists as more historical
context becomes available.


F.4 DETAILED RESULTS FOR DIFFERENT TEMPORAL SPLITS(RQ4)


To test generalization under _varying_ _emergence_, we build four
chronological splits with test horizons of _{_ 10%, 30%, 50%,
70% _}_, corresponding to train:val:test timeline ratios [8 : 1 : 1],

[5 : 2 : 3], [3 : 2 : 5], [2 : 1 : 7]. For each split, we re-partition the
data strictly in time (validation is re-cut per split), which shortens training history and increases the share of first-appearance
entities. We evaluate on **ICEWS14** and **ICEWS05-15**, reporting MRR and Hits@10, and compare TRANSFIR against strong
baselines ( **LogCL**, **REGCN**, **MLEMKD** ); all models are retrained for each split with the main hyperparameters.


|Col1|Col2|Hits@10|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||


0.34


0.22


0.10


TransFIR(Ours) LogCL MLEMKD REGCN


0.39


0.26


0.13


0.00

0.46


different time split


**Results** **and** **discussion.** As shown in Fig. F.4, performance
drops for all methods as the test horizon expands and emer
Figure 11: Experiment results on

gence increases. TRANSFIR consistently attains the best scores

ICEWS14 and ICEWS05-15 un
across splits and exhibits the _smallest_ degradation, indicating ro
der different time splits.

bustness when historical coverage is reduced. A mild uptick
for some baselines at the 70% horizon likely stems from undertraining with a shorter history, which narrows the gap between known and emerging entities and partially curbs collapse. Overall, these trends support that TRANSFIR ’s Classification–Representation–Generalization pipeline remains effective across diverse temporal partitions.


23


F.5 HYPERPARAMETER SENSITIVITY(RQ4)


We investigate the hyperparameter sensitivity of TRANSFIR, focusing on codebook size _K_, Interaction Chain length _k_, hidden dimension _d_, and the number of layers _L_ in the IC Encoder.


First, we examine the impact of codebook size _K_, testing values _{_ 10, 25, 50, 100 _}_ . As shown in
Figure 12, performance improves with increasing _K_, with _K_ = 50 yielding the best results across
most datasets.


Next, we analyze the effect of Interaction Chain length _k_ by testing values _{_ 10, 15, 30, 50 _}_ . While
the best length varies across datasets, the performance remains stable across different lengths for all
datasets, with no significant drop in performance.


Additionally, we assess the hidden dimension _d_ and the number of layers _L_ . Performance is stable
across a range of hidden dimensions, with _d_ = 768 providing optimal results in most cases. Similarly, two to three layers in the Chain Encoder provide the best performance, with no significant
improvement from adding more layers.


24
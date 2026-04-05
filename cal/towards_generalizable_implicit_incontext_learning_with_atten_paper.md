# TOWARDS GENERALIZABLE IMPLICIT IN-CONTEXT LEARNING WITH ATTENTION ROUTING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Implicit in-context learning (ICL) has newly emerged as a promising paradigm
that simulates ICL behaviors in the representation space of Large Language Models (LLMs), aiming to attain few-shot performance at zero-shot cost. However, existing approaches largely rely on injecting shift vectors into residual flows, which
are typically constructed from labeled demonstrations or task-specific alignment.
Such designs fall short of utilizing the structural mechanisms underlying ICL
and suffer from limited generalizability. To address this, we propose **In-Context**
**Routing (ICR)**, a novel implicit ICL method that _internalizes_ generalizable ICL
patterns at the attention logits level. It extracts reusable structural directions that
emerge during ICL and employs a learnable input-conditioned router to modulate
attention logits accordingly, enabling a train-once-and-reuse framework. We evaluate ICR on 12 real-world datasets spanning diverse domains and multiple LLMs.
The results show that ICR consistently outperforms prior implicit ICL methods
that require task-specific retrieval or training, while demonstrating robust generalization to out-of-domain tasks where existing methods struggle. These findings
position ICR to push the boundary of ICL’s practical value.


1 INTRODUCTION


Large Language Models (LLMs) have been widely adopted for text understanding and generation
tasks. As applications broaden, the ability to adapt these models efficiently at inference time has
become increasingly important (Brown et al., 2020; Wang et al., 2020b). In-context learning (ICL)
is a central mechanism for this adaptation (Dong et al., 2022; Min et al., 2021): by conditioning on
a few labeled examples inserted before the query, known as in-context demonstrations (ICDs), the
model can perform new tasks without any parameter updates (Wies et al., 2023; Pan, 2023).


Despite its broad adoption, ICL faces two practical limitations: (i) inserting ICDs into the prompt
inflates sequence length and inference cost compared to zero-shot use (Peng et al., 2024; Li et al.,
2025a), and (ii) performance is brittle, varying with small changes in ICD order or format (Wu
et al., 2022; Guo et al., 2024). To address these issues, recent work has explored **implicit** **ICL**,
which converts ICDs into dense vectors that steer intermediate residual flows to approximate the
effect of explicit prompting (Hendel et al., 2023; Todd et al., 2023; Liu et al., 2023; Li et al., 2024).


While vector-based implicit ICL offers a new way to simulate ICL behaviors in LLMs, it struggles to
generalize across real-world tasks. First, using fixed-size vectors as carriers is inherently restrictive.
They can only encode a limited amount of prompt information. Attempts to add new knowledge
or transfer it to other models require constructing new vectors. Moreover, this approach lacks a
theoretical foundation that is both model-agnostic and input-agnostic. Second, they push LLMs to
mimic ICL rather than internalize it, since by the time vectors are applied, the backbone has already
settled into a distribution shaped by its own attention dynamics. As a result, they perform well
mainly on tasks where explicit ICL already succeeds, but fail to generalize to more challenging
cases, such as tasks lacking manually labeled ICDs. To this end, we ask:


_“Can we design an implicit ICL method that enables models to truly_ _**internalize**_ _ICL, thus allowing_
_seamless generalization across diverse ICL scenarios?”_


To examine if there exists a generalizable cross-task ICL pattern, we take explicit multi-task ICL as
an empirical probe, which incorporates ICDs from diverse, potentially out-of-domain (OOD) tasks
to support those lacking their own labeled examples. This setting provides a unique lens in that it


1


CSQA+PIQA → CREAK


|Col1|Col2|Col3|Col4|41.|Col6|Col7|44.<br>40|Col9|Col10|20|
|---|---|---|---|---|---|---|---|---|---|---|
||32.|32.|32.|00|00||||||
|25.|80|80|||||||||
||||||||||||
||||||||||||
||||||||||||


|51.|80 52.|Col3|Col4|10 51.|Col6|Col7|40 50.|Col9|Col10|80|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
||||||||||||
||||||||||||


|22.|24.<br>40|Col3|Col4|40|Col6|Col7|24.|Col9|Col10|80|
|---|---|---|---|---|---|---|---|---|---|---|
||||||19.|19.|80|80|||
||||||19.||||||
||||||||||||
||||||||||||
||||||||||||


Figure 1: Multi-task ICL on OOD targets. Multi-task few-shot prompting sometimes surpasses both
zero-shot and the best single-source few-shot (SST-5, CSQA), but may also degrade performance at
times (CREAK). ∆ denotes the difference from the best single-source few-shot prompting.


can sometimes outperform zero-shot prompting and few-shot baselines from single source tasks, but
can also yield worse results (Fig 1). This indicates that ICDs from different tasks may embed a latent
cross-task pattern beneficial for ICL, yet explicit prompting introduces noise that may obscure it.


Motivated by this, we move deeper than additive residual vectors to investigate the attention space
to identify and leverage the cross-domain ICL pattern. We formally analyze how such patterns can
be decomposed and embedded directly into attention logits during zero-shot inference, a strategy
which we term _attention routing_ . Building on this, we propose **In-Context Routing (ICR)**, which
extracts the cross-task ICL pattern and employs a router to synthesize it as a low-rank weighted
composition, guiding attention computation in a task-adaptive manner.


Empirically, ICR consistently outperforms vector-based implicit ICL baselines across five in-domain
and seven out-of-domain (OOD) datasets. It exhibits strong OOD generalization without performance degradation, whereas existing baselines often suffer deficits on certain OOD tasks. ICR also
retains key advantages of implicit ICL, including fewer cached parameters and faster inference than
few-shot prompting. To the best of our knowledge, ICR is the first implicit ICL method that can be
directly adopted for zero-shot inference in diverse new tasks without retrieval or retraining.


Our contributions are three-fold. 1) Recognizing the challenges of post-hoc steering, we propose a
new paradigm, _attention routing_ . It leverages generalizable ICL patterns that emerge in the attention
space across tasks to steer attention logits. 2) Building on this paradigm, we propose **In-Context**
**Routing (ICR)** . Without modifying LLM parameters, ICR introduces a small number of learnable
parameters and an end-to-end training strategy that adaptively adjusts routing based on the input
query. 3) Extensive experiments validate the effectiveness of ICR, and comprehensive analyses
demonstrate that it internalizes ICL patterns while achieving strong adaptivity and generalization.


2 ATTENTION ROUTING


This section introduces attention routing, a paradigm that leverages general ICL patterns to intrinsically steer model behavior in zero-shot settings. We begin in Sec. 2.1 by revisiting existing implicit
ICL paradigms and their challenges. Sec. 2.2 then presents the formation of attention routing, and
Sec. 2.3 analyzes why the general ICL pattern underlying it can be extracted from LLM attention.


2.1 PRELIMINARIES AND CHALLENGES OF EXISTING WORK

An ICL prompt input **p** to the LLM is typically constructed from several labeled examples serving
as in-context demonstrations (ICDs) and a query sample. We denote it as **p** = [ **D** _, xq_ ], where
**D** = _{_ ( _xi, yi_ ) _}_ _[n]_ _i_ =1 [represents the set of] _[ n]_ [ ICDs and] _[ x][q]_ [is the query sample.] [The model is expected]
to infer the input-label mappings illustrated by the ICDs and then predict the label associated with
the query sample. Extensive studies have shown that the multi-head attention (MHA) module in
transformer-based models plays a central role in learning from **D** (Olsson et al., 2022; Chen et al.,
2024), which performs a soft query-conditioned retrieval over the ICDs to acquire key knowledge.


**Vector-based implicit ICL** replaces explicit token-level ICDs with dense vectors injected into the
model’s internal layers. They find that ICDs can be viewed as additive modifications to the MHA
outputs in the zero-shot setting and steer the model using vectors that represent ICL (Peng et al.,
2024). A typical approach is to add the activation differences induced by ICDs as shift vectors to the
zero-shot hidden states. Formally, given an LLM with hidden dimension _d_ and an input sequence of
_T_ tokens, the MHA output **h** **[˜]** _[l]_ _t_ [of token] _[ t]_ [ at layer] _[ l]_ [ is given by:]

**h** _[l]_ = Concat _h_ �softmax( **A** _[l,h]_ ) _V_ _[l,h]_ [�] = Concat _h_ �softmax( _[Q][l,h]_ ~~_√_~~ _[K]_ _[l,h][⊤]_ + **M** ) _V_ _[l,h]_ [�] _,_ (1)

_dk_


2


18


12


6


0
Zero-shot PIQA TREC PIQA+TREC


PIQA+TREC → CSQA


40


30


20


10


SST-2+MR → SST-5


0
Zero-shot SST-2 MR SST-2+MR


45


30


15


0
Zero-shot CSQA PIQA CSQA+PIQA


**˜h** _[l]_ _t_ [=] **[ h]** _t_ _[l]_ [+] _[ β][l][ ·]_ **[ V]** shift _[l]_ _[,]_ (2)
where **h** _[l]_ _∈_ R _[T][ ×][d]_ denotes the zero-shot MHA output at layer _l_ and _Q_ _[l,h]_ _, K_ _[l,h]_ _, V_ _[l,h]_ _∈_ R _[T][ ×][d][k]_ are
head projections of the final output from layer _l −_ 1. _dk_ is the dimensionality of each head and **M** is
a causal mask. **A** _[l,h]_ _∈_ R _[T][ ×][T]_ is the matrix of attention logits at layer _l_ and head _h_ . **V** shift _[l]_ _[∈]_ [R] _[d]_ [is a]
shift vector. It is typically derived from explicit ICL, for example, by averaging the hidden states of
_n_ ICDs’ last tokens. The scalar coefficient _β_ _[l]_ _∈_ R controls the magnitude of this shift.


**Challenges.** The steering approach in Eq. 2, while effective for task-specific adaptation, is **inher-**
**ently limited in generalizability** . It operates in a post-hoc manner where a shift vector is directly
injected into the residual stream. Such additive interventions cannot structurally control how information flows, and thus often remain tied to task-specific representations. In contrast, more generalizable ICL patterns are expected to lie in how queries are routed through alternative attention paths.
This motivates our hypothesis that modulating the matching geometry in the attention space, rather
than perturbing outputs post hoc, better reflects the mechanism of ICL, where query tokens attend to
the most relevant directions (Olsson et al., 2022; Cho et al., 2025). We therefore argue that attention
logits provide a principled basis for extracting task-agnostic and transferable ICL patterns. Since
it intrinsically directs model attention to desired routes, we refer to steering attention logits during
zero-shot inference as _attention routing_ .


2.2 HOW ATTENTION ROUTING WORKS


As shown in Eq. 1, attention logits are

**(a) Vector-based** **(b) Explicit ICL** **(c) Attention Routing**

governed by query-key interactions, mak- **implicit ICL**

ence of ICDs across diverse tasks. These

space capturing generalizable ICL dynam
Figure 2: Illustration of attention routing compared

ics. To recover this subspace, we first per
with vector-based implicit ICL, with head-level details

form explicit ICL across multiple domains

omitted for clarity.

to obtain high-dimensional mixed-domain
attention representations. Specifically, we iteratively input ICL prompts into the LLM, each prompt
containing ICDs and a query sample from the same domain. We then collect the last-token _Q_ and _K_
projections across domains and stack them to form two **ICL bases** . Principal Component Analysis
(PCA) is applied separately to each base, yielding two sets of layer-wise **Principal ICL Directions**
**(PIDs)**, denoted for each layer _l_ as _Uq_ _[l][, U]_ _k_ _[ l]_ _[∈]_ [R] _[d][×][r]_ [, where] _[ r]_ [ is the rank of the PID subspace.]


We define a _routing_ _vector_ _α_ _[l]_ _∈_ R _[r]_ that assigns weights to the PIDs at layer _l_ . _α_ _[l]_ controls
the strength with which each PID modulates the attention. During zero-shot inference, the layerlevel query and key projections are formed by concatenating the per-head projections _Q_ _[l,h]_ **zs** _[, K]_ **zs** _[l,h]_ _[∈]_
R _[T][ ×][d][k]_, yielding _Q_ _[l]_ **zs** _[, K]_ **zs** _[l]_ _[∈]_ [R] _[T][ ×][d]_ [.] [The] [routing] [vector] [specifies] [a] [low-rank] [modulation] [of] [the]
attention logits and thereby biases the attention dynamics toward the extracted PIDs:

∆ **A** _[l]_ =       - _Q_ _[l]_ **zs** _[U]_ _q_ _[ l]_       - diag( _α_ _[l]_ )       - _K_ **zs** _[l]_ _[U]_ _k_ _[ l]_       - _⊤_ _∈_ _RT ×T ._ (3)

The layer-level bias ∆ **A** _[l]_ is shared across all _H_ heads in layer _l_, so that each head’s routed logits
become **A** **[˜]** _[l,h]_ = **A** _[l,h]_ + ∆ **A** _[l]_ . Figure 2 shows the key difference between attention routing and
vector-based implicit ICL. We further provide a kernel-based perspective in Appendix A.1.


2.3 WHY PIDS CAPTURE GENERAL ICL PATTERN
We now explain why the low-dimensional subspaces defined by the PID sets _{Uq_ _[l][}][L]_ _l_ =1 [and] _[ {][U]_ _k_ _[ l]_ _[}][L]_ _l_ =1 [,]
derived from multi-domain ICL, capture a general attention pattern to enable ICL. As described in
Sec. 2.2, at each layer we derive two ICL bases, _Q_ and _K_, by stacking projections across multiple
domains. Considering the rows of _Q_ from a particular domain `d`, we can model its covariance under
the Spiked Covariance Model (Johnstone, 2001) (see Appendix A.2) as a mixed spiked form:

Σ [(] _Q_ `[d]` [)] = _Sq_ Λ _qSq_ _[⊤]_ + _Bq,_ `d` Γ _q,_ `d` _Bq,_ _[⊤]_ `d` [+] _[σ]_ [2] _[I,]_ (4)


3


**(a) Vector-based**


**(b) Explicit ICL** **(c) Attention Routing**


**implicit ICL**


Figure 2: Illustration of attention routing compared
with vector-based implicit ICL, with head-level details
omitted for clarity.


where _Sq_ _∈_ R _[d][×][r]_ captures a low-dimensional subspace of attention structures shared across domains, while _Bq,_ `d` encodes domain-specific variations with energy Γ _q,_ `d` . _σ_ [2] _I_ represents isotropic
noise. An analogous decomposition holds for _K_ . Let _{D_ 1 _, . . ., D_ `D` _}_ denote all `D` domains involved
in the ICDs. We define the pooled covariance of _Q_ as:


The same expansion holds for Σ [�] _K_ . The first term corresponds to the ICL structure shared across
domains, while the last term aggregates domain-specific variations. If the domain-specific subspace
set _{Bq,_ `d` _}_ are sufficiently diverse and lack consistent alignment, their aggregate contribution averages out toward isotropy. In this case, they primarily increase background variance rather than
forming dominant eigen-directions. In contrast, the shared component _Sq_ Λ _qSq_ _[⊤]_ [accumulates] [con-]
sistently across all domains. In this way, PIDs obtained by PCA on multi-domain ICL bases recover
a domain-stable ICL pattern. Appendix A.3 provides perturbation analysis supporting this claim,
and Appendix A.4 further examines the validity of the extracted pattern in OOD settings.


3 METHOD


Building on the foundation of attention routing, we propose a new implicit ICL method, termed
**In-Context** **Routing** **(ICR)** . ICR leverages attention routing to dynamically integrate extracted
Principal ICL Directions (PIDs) into the attention space, thereby enhancing zero-shot inference of
LLMs. We instantiate ICR in three stages: (i) PIDs extraction across multiple domains, (ii) a queryconditioned router that determines low-rank routing vectors and head gates, and (iii) multi-objective
training that combines supervision with stable and sparse routing. The pipeline of ICR is illustrated
in Figure 3 and presented in pseudocode in Appendix C.


3.1 PRINCIPAL ICL DIRECTIONS EXTRACTION

To implement ICR, we first extract the ICL bases from the model’s ICL across multiple domains,
along with the PIDs contained within them. For `D` domains, we construct a set of ICL prompts for
each domain `d`, denoted as _P_ `d` . Let _N_ = [�] `d` `[D]` =1 _[|P]_ `[d]` _[|]_ [ be the total number of constructed ICL prompts]
across all domains. These prompts are fed into the LLM domain by domain. During inference of
the _i_ -th prompt from domain `d`, we extract the query and key projections of its _last token_ in the layer
_l_ and the head _h_, denoted _q_ `d` _[l,h]_ _,i_ _[, k]_ `d` _[ l,h]_ _,i_ _[∈]_ [R][1] _[×][d][k]_ [.] [We] [then] [concatenate] [them] [across] [heads] [to] [obtain]
layer-level vectors _q_ `d` _[l]_ _,i_ [=] [Concat] _[h][ q][ l,h]_ `d` _,i_ _[∈]_ [R][1] _[×][d]_ [and] _[ k]_ `d` _[ l]_ _,i_ [=] [Concat] _[h][ k]_ `d` _[ l,h]_ _,i_ _[∈]_ [R][1] _[×][d]_ [.] [Finally,] [these]
vectors are stacked across prompts and domains to yield the ICL bases across `D` domains.

_Q_   - _[l]_ = stack `[D]` `d` =1 [stack] _i_ _[|P]_ =1 `[d]` _[|]_ _[q][ l]_ `d` _,i_ _[∈]_ [R] _[N]_ _[×][d][,]_ _K_   - _[l]_ = stack `[D]` `d` =1 [stack] _i_ _[|P]_ =1 `[d]` _[|]_ _[k]_ `d` _[ l]_ _,i_ _[∈]_ [R] _[N]_ _[×][d][.]_ (7)


From _Q_ _[l]_ _,_ _K_ _[l]_ constructed above, we then obtain the top- _r_ principal directions by PCA to form the

[�] [�]
PIDs _Uq_ _[l][, U]_ _k_ _[ l]_ _[∈]_ [R] _[d][×][r]_ [.] [These] [PIDs] [serve] [as] [reusable] [routing] [directions] [for] [downstream] [control] [of]
attention logits during both training and inference.


3.2 QUERY-CONDITIONED ROUTER

After obtaining the PIDs, our goal is to construct the attention routing form introduced in Sec. 2.2.
To apply these cross-domain ICL patterns during inference on various input queries, we employ a
learnable router to optimize the routing process. Given a query sample _x_, it is fed into the LLM and
a frozen text encoder, which produces a representation E( _x_ ). E( _x_ ) is then passed to a two-branch
router consisting of two two-layer MLPs, _gθα_ and _gθγ_ . The two branches generate a routing matrix
_α_ ( _x_ ) _∈_ R _[L][×][r]_ and a gating matrix _γ_ ( _x_ ) _∈_ R _[L][×][H]_ in parallel, computed as


_α_ ( _x_ ) = tanh� _gθα_ (E( _x_ ))� _∈_ R _[L][×][r]_ _,_ (8)

_γ_ ( _x_ ) = _σ_             - _gθγ_ (E( _x_ ))� _∈_ R _[L][×][H]_ _,_ (9)


4


- _QiQ_ _[⊤]_ _i_ _[,]_ _N_ =

_i∈D_ `d`


1
Σ� _Q_ = _N_


```
 D
```


`d` =1


```
 D
```

- _|D_ `d` _|._ (5)


`d` =1


We compute the expectation of Σ [�] _Q_ and expand it under the mixed spiked form defined in Eq. 4 as:


E[Σ [�] _Q_ ] = _Sq_ Λ _qSq_ _[⊤]_ [+] _[ σ]_ [2] _[I]_ [+] [1]

_N_


```
 D
```

- _|D_ `d` _| Bq,_ `d` Γ _q,_ `d` _Bq,_ _[⊤]_ `d` _[.]_ (6)

`d` =1


The final training objective is a weighted combination of the above three terms:
_L_ = _L_ CE + _λ_ conf _L_ conf + _λ_ spar _L_ spar + _λ_ gate _L_ gate _,_ (14)
where _λ_ conf, _λ_ spar, and _λ_ gate are hyperparameters that weight each corresponding loss term.


5


(a) Principal ICL Direction Extraction


�-th layer


ICL prompts


(b) ICR Training

Zero-shot

prompts


Figure 3: Pipeline of In-Context Routing (ICR). (a) We perform ICL across multiple domains to
extract PIDs, which can be stored and reused. (b) We train the router with zero-shot inputs while
keeping the LLM frozen, and it generates query-conditioned matrices to control the routing.


where _σ_ ( _·_ ) denotes the sigmoid function. _α_ _[l]_ ( _x_ ) _∈_ R [1] _[×][r]_ denotes the _r_ -dimensional routing vector
at layer _l_, and _γ_ _[l,h]_ ( _x_ ) _∈_ R [1] _[×]_ [1] provides head-specific gates at layer _l_ and head _h_ . Together, _α_ ( _x_ )
adaptively amplifies or attenuates the extracted PIDs according to query semantics, and _γ_ ( _x_ ) regulates the contributions of individual heads. They jointly produce a low-rank bias that leverages the
PIDs in a query-conditioned manner to modulate the zero-shot attention logits for input _x_ :
**˜A** _[l,h]_ ( _x_ ) = **A** _[l,h]_ ( _x_ ) + _γ_ _[l,h]_ ( _x_ )         - _Q_ _[l]_ **zs** _[U]_ _q_ _[ l]_         - diag� _α_ _[l]_ ( _x_ )�� _K_ **zs** _[l]_ _[U]_ _k_ _[ l]_         - _⊤,_ (10)

Again, _Q_ _[l]_ **zs** _[, K]_ **zs** _[l]_ _[∈]_ [R] _[T][ ×][d]_ [are] [the] [concatenation] [of] [head-level] [projections] _[Q][l,h]_ **zs** _[, K]_ **zs** _[l,h]_ _∈_ R _[T][ ×][d][k]_ .
**˜A** _[l,h]_ ( _x_ ) is then applied to the subsequent attention computation and final answer generation.


3.3 TRAINING OBJECTIVE

During ICR training, only the router parameters ( _θα, θγ_ ) are updated. The training set is constructed
by sampling and mixing subsets from each domain _D_ `d` _∈D_ . We then construct mini-batches of
size _B_, each denoted as _{_ ( _xi, yi_ ) _, D_ `d` _}_ _[B]_ _i_ =1 [, where][ (] _[x][i][, y][i]_ [)][ is an input–label pair and] _[ D]_ `[d]` [indicates its]
domain. Within each mini-batch, we obtain (i) the zero-shot output _p_ [zs] _i_ _[∈]_ [R] _[|V|]_ [and] [(ii)] [the] [output]
under ICR _p_ [ICR] _i_ _∈_ R _[|V|]_ of the generated answer, where _V_ is the model’s vocabulary.


**(1)** **Supervised** **cross-entropy.** To provide solid semantic supervision for training ICR, we first
adopt the standard cross-entropy loss. For each input and its ground-truth label ( _xi, yi_ ), the loss is:


_L_ CE = _−_ [1]

_B_


_L_ CE = _−_ [1]


_B_

- log _P_ [ICR][�] _yi|xi_ - _._ (11)


_i_ =1


**(2)** **Confidence** **alignment.** We encourage routed predictions to be at least as confident as zeroshot ones via an entropy drop objective. This prevents the router from taking a shortcut of producing
over-uncertain predictions and ensures routed inference does not reduce confidence:


_qv_ log _qv._ (12)
_v∈V_


_L_ conf = [1]

_B_


_B_


- - - - - - [�] 
ReLU _H_ softmax( _p_ [ICR] _i_ ) _−_ _H_ softmax( _p_ [zs] _i_ [)] _,_ _H_ ( _q_ ) = _−_
_i_ =1 _v∈V_


**(3) Sparse routing.** We regularize the per-layer routing vectors _α_ _[l]_ ( _x_ ) _∈_ R _[r]_ and gates _γ_ _[l]_ ( _x_ ) _∈_ R _[H]_
to encourage sparsity in the modulation that ICR introduces to MHA. Because later layers are closer
to the final prediction and should depend on fewer but more decisive routing directions, we scale the
sparsity penalty with a layer-dependent weight _w_ _[l]_ that increases linearly with depth:


- _[∥][α][l]_ [(] _[x]_ [)] _[∥]_ [1]

_w_ _[l]_

_r_

_l_ =1


_r_


      1

_,_ _L_ gate = E _x_

_L_


_L_


_l_ =1


_∥γ_ _[l]_ ( _x_ ) _∥_ 1

_H_


_L_ spar = E _x_


1

_L_


_L_


_._ (13)


3.4 INFERENCE

During inference, ICR is implemented by adding low-rank biases to the attention logits of the corresponding heads, as defined in Eq. 10, while keeping the backbone parameters frozen. When given
a zero-shot prompt, ICR adaptively forms **A** **[˜]** _[l,h]_ ( _x_ ), which the model then uses for subsequent prefilling and complete decoding. The entire procedure operates purely on the query representations
and does not access any label space or task-specific supervision at test time. In this way, ICR implicitly equips zero-shot inference with the effect of ICL by fundamentally routing attention dynamics
along shared structural directions via query-conditioned composition, regardless of whether the input belongs to a domain seen during training.


4 EXPERIMENTS


4.1 SETUPS

This section introduces the models employed and the settings for cross-domain collections, training,
and evaluation of ICR. Further details are provided in Appendix D.


**Models** ICR is evaluated on three open-source LLMs: Llama2-7B (Touvron et al., 2023),
Qwen2.5-7B (Yang et al., 2025), and Llama3.1-8B (Grattafiori et al., 2024). All ablation and analysis studies are conducted on Llama2-7B as an example.


**Cross-domain** **collections** We consider five datasets with distinct task types: AGNews (Zhang
et al., 2015), SST-2 (Socher et al., 2013), TREC (Li & Roth, 2002), CSQA (Talmor et al., 2019),
and PIQA (Bisk et al., 2020), and treat each dataset as a separate domain. For each dataset, we
construct ICL prompts by first sampling a query and a balanced set of ICDs, both from the training
split, where 5 ICDs are drawn from each class of the same dataset. We construct 10k prompts for
AGNews and 5k prompts for each of the remaining datasets. After feeding each prompt into the
LLM, we extract the layer-wise _Q_ and _K_ representations of the last token. They are aggregated
across all prompts to obtain per-layer ICL bases as in Eq. 7, enabling PIDs extraction via PCA.
More details about ICL prompts construction during collection are presented in Appendix D.


**Training** We train the router on a set of 25k queries, obtained by randomly sampling 5k queries
from the training split of each of the five datasets and shuffling them together. Each query is first
encoded by a frozen MiniLM encoder (Wang et al., 2020a), and its pooled representation is fed into
the router. The ICR is applied only to the **last** one-third of the LLM layers. We set _λ_ conf = 0 _._ 01,
_λ_ spar = 10 _[−]_ [3], and _λ_ gate = 0 _._ 02 during training.


**Evaluation** We evaluate on 500 randomly sampled test instances (or the full set if smaller) using
dataset-specific prompts and a batch size of 4. Each experiment is run with three seeds, and we
report the average results. We treat the five datasets used for training as **in-domain (ID)** and select
seven additional datasets for out-of-domain (OOD) evaluation. Based on their task similarity to the
training datasets, we further categorize them into **near OOD** and **far OOD** . The near OOD datasets
include SST-5 (Socher et al., 2013), MR (Pang & Lee, 2005), and MRPC (Dolan & Brockett, 2005),
while the far OOD datasets include CB (De Marneffe et al., 2019), COPA (Roemmele et al., 2011),
CREAK (Onoe et al., 2021), and AI2SciE (Clark et al., 2018). In addition to zero-shot and few-shot
prompting, we choose three vector-based methods with calibration or training as baselines: I2CL
(Li et al., 2024), LIVE (Peng et al., 2024), and M [2] IV (Li et al., 2025a). We further compare the
in-domain performance of ICR with five training-free methods: TV (Hendel et al., 2023), FV (Todd
et al., 2023), ICV (Liu et al., 2023), ELICIT (Wang et al., 2024a), and IV (Liu & Deng, 2025).


4.2 MAIN RESULTS

As shown in Table 1, ICR closely matches and can even surpass few-shot prompting on ID tasks.
It consistently outperforms all implicit ICL baselines. Notably, these methods often require additional task-specific retrieval or training, whereas ICR operates in a train-once-and-reuse manner,
further highlighting its practical value. On OOD tasks, multi-task few-shot prompting is unstable,
performing well on some tasks but collapsing on others, which corroborates the limitations observed in Figure 1. By design, vector-based implicit ICL inherits the drawbacks of explicit ICL,
leading to higher failure rates. In contrast, ICR improves over the best implicit baseline by +3.0%
on Llama2-7B and +6.5% on Qwen2.5-7B, and even surpasses few-shot prompting by +2.7% on
Qwen2.5-7B. These results establish ICR as a generalizable paradigm for implicit ICL. We also
compare ICR with vector-based ICL variants that inject dataset-specific vectors into hidden states


6


Table 1: Baseline comparison across benchmarks. [*] For ID datasets, few-shot uses 5-shot balanced
sampling per class. For OOD datasets, we adopt multi-task few-shot prompting where each ID
dataset provides 3-shot ICDs. The _Collapse_ column reports the number of cases where a method
underperforms the zero-shot baseline. Results on Llama3.1-8B are shown in Appendix E.1.


**In-Domain (ID)** **Near OOD** **Far OOD** **Overall**
**Method**

AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE **Average** **Collapse**


**Llama2-7B**


Zero-shot 67.0 78.6 56.6 22.4 52.2 25.8 72.2 44.4 37.5 63.0 51.8 34.8 50.5  Few-shot [*] 81.0 95.2 84.6 58.0 59.8 37.4 98.6 68.2 41.1 82.0 50.8 45.4 66.8 1


I2CL 85.5 86.0 78.6 23.8 55.6 27.6 71.6 42.4 38.2 63.6 52.6 35.0 55.0 2
LIVE 86.0 86.2 81.0 24.2 56.4 32.8 73.8 47.6 40.8 64.8 51.0 34.6 56.6 2
M [2] IV 86.4 **86.4** 81.5 **24.8** 56.8 30.8 74.0 46.0 42.6 64.8 54.0 35.2 56.9 **0**
**ICR** **86.6** **86.4** **83.8** **24.8** **57.0** **38.6** **79.8** **53.4** **46.4** **68.0** **56.4** **37.2** **59.9** **0**


**Qwen2.5-7B**


Zero-shot 66.8 54.0 65.8 80.4 76.2 31.4 64.4 72.4 83.9 92.0 77.8 90.4 71.3  Few-shot [*] 80.2 95.6 67.6 82.2 86.0 37.2 70.2 76.2 83.9 95.0 59.7 95.8 77.5 1


I2CL 77.0 86.4 68.6 81.6 81.2 34.6 69.0 70.8 80.6 92.6 74.8 91.8 75.6 3
LIVE 79.0 87.8 70.4 81.6 82.0 30.8 68.6 69.4 81.0 93.2 72.8 91.8 75.7 4
M [2] IV 79.6 89.0 **70.8** 81.8 82.5 31.6 71.2 71.0 76.0 93.5 74.6 92.4 76.2 3
**ICR** **80.4** **91.0** 70.6 **82.0** **82.6** **41.4** **89.4** **73.2** **84.6** **95.0** **79.2** **93.2** **80.2** **0**


Table 2: Baseline comparison on in-domain benchmarks.


**Llama2-7B** **Qwen2.5-7B**
**Method**

AG SST-2 TREC CSQA PIQA **Overall** AG SST-2 TREC CSQA PIQA **Overall**


TV 82.8 83.4 73.4 22.6 53.0 63.0 70.4 78.2 64.6 80.6 74.6 73.7
FV 83.6 82.8 72.8 22.4 52.5 62.8 68.4 76.8 66.2 78.8 80.0 74.0
ICV 83.6 84.2 74.2 23.0 52.8 63.5 74.6 83.0 67.2 81.3 77.2 76.7
ELICIT 84.0 84.4 75.8 22.4 53.9 64.1 70.4 78.5 65.0 79.2 76.4 74.3
IV 83.8 85.6 73.8 23.2 54.6 64.2 73.8 78.4 66.0 81.2 77.8 75.4
**ICR** **86.6** **86.4** **82.2** **24.8** **57.0** **67.4** **80.4** **91.0** **70.6** **82.0** **82.6** **81.2**


(Table 2). These ad-hoc methods lack transferability and are evaluated only on five ID datasets. ICR
consistently outperforms them by a clear margin, indicating that attention routing captures deeper
and more general ICL patterns. Appendix E.2 further compares ICR with few-shot LoRA (Hu et al.,
2021), a PEFT-based finetuning method. Appendix F provides an efficiency analysis of ICR.


4.3 ABLATION STUDY


In this section, we provide ablations on the extraction of PIDs and the key components of ICR.
Further analyses on the strategy for sampling ICDs in constructing the ICL bases and on the effect
of routing layer positions are presented in Appendix G.2 and Appendix G.3.


**Key Components** Table 4 presents ablations of the key components of ICR, including the auxiliary loss terms in Eq. 14 and the query-conditioned modulation of _α_ and _γ_ . Dropping _L_ spar or _L_ gate
has little impact on ID and near-OOD tasks but leads to clear degradation on far-OOD datasets, consistent with their role in constraining over-intervention and enhancing transferability. Removing the
confidence-alignment loss _L_ conf produces less systematic changes, suggesting that its primary effect
is stabilizing routing by suppressing entropy inflation rather than directly improving ICL accuracy.
For _α_ and _γ_, we preserve their magnitude but redistribute it uniformly across PID directions or


7


**PIDs** **Extraction** To understand the role of

Table 3: Ablation on PIDs Extraction. “R.O.”

PIDs extraction, we conduct two ablations (Ta
denotes the replacement of PIDs with a random

ble 3). First, we vary the PCA rank _r_ _∈_ 4 _,_ 8 _,_ 12.

orthogonal basis. Scores are averaged within

Compared to _r_ = 8, reducing _r_ to 4 improves

ID, near-OOD, and far-OOD groups.

in-domain and near-OOD results but sharply reduces far-OOD accuracy, as the stronger bottle- **Setting** **ID** **Near OOD** **Far OOD**
neck regularizes domain signals but suppresses r=4 (PCA) **67.8** **57.5** 45.6
the diversity needed for transfer. Increasing _r_ to r=8 (PCA) 67.7 57.3 **52.0**
12 consistently hurts, likely due to the enlarged r=12 (PCA) 53.2 54.4 43.4

r=8 (R. O.) 63.9 48.1 46.7

subspace introducing degrees of freedom that remain under-trained. Second, we replace PCA with a random orthogonal basis ( _r_ = 8). While ID
performance remains close to PCA, both near- and far-OOD collapse. This shows that low-rank
routing alone is insufficient: OOD robustness crucially depends on aligning with meaningful ICL
directions extracted by PCA. A more detailed study on PIDs extraction is provided in Appendix G.1.


Table 3: Ablation on PIDs Extraction. “R.O.”
denotes the replacement of PIDs with a random
orthogonal basis. Scores are averaged within
ID, near-OOD, and far-OOD groups.


**Setting** **ID** **Near OOD** **Far OOD**


r=4 (PCA) **67.8** **57.5** 45.6
r=8 (PCA) 67.7 57.3 **52.0**
r=12 (PCA) 53.2 54.4 43.4
r=8 (R. O.) 63.9 48.1 46.7


Table 4: Ablation of key components in ICR.


**In-Domain (ID)** **Near OOD** **Far OOD**
**Ablation**

AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE


FULL **86.6** 86.4 83.8 24.8 **57.0** **38.6** 79.8 53.4 **46.4** **68.0** **56.4** **37.2**


w/o _L_ conf 84.4 **88.8** **84.6** 23.8 54.2 38.0 **84.0** 53.8 33.9 66.0 54.8 31.4
w/o _L_ gate 86.0 88.4 80.6 **27.4** 56.6 37.6 83.4 44.8 17.9 61.0 **56.4** 33.0
w/o _L_ spar 84.6 87.6 80.2 26.6 54.4 38.2 82.6 38.0 **46.4** 66.0 52.4 35.2


w/o _α_ ( _x_ ) 68.2 80.4 47.6 21.4 52.0 30.2 72.0 49.2 39.3 57.0 52.4 33.2
w/o _γ_ ( _x_ ) 64.8 82.2 49.2 21.0 54.8 29.6 73.0 **57.4** 39.3 56.0 52.8 33.0


heads. Both ablations cause consistent drops, showing that query-conditioned allocation is crucial:
uniform _α_ or _γ_ erases direction- and head-specific selectivity that underpins effective routing.


5 ANALYSES


5.1 ICR EXHIBITS INTERPRETABLE EFFECTS.


Though ICR modulates zero-shot inference in the attention space, its effects are interpretable.
Probing next-token distributions across ID and OOD datasets reveals systematic vocabulary-level
shifts that remain stable across datasets. Specifically, ICR consistently upweights tokens linked to
reasoning-oriented structures such as _’capture’_, _’connections’_, and _’signs’_, rather than task-specific
label words. Full method details and the top-50 ranked token list are provided in Appendix H.


5.2 ALIGNED AND DIVERSE DOMAIN DISTRIBUTIONS MATTER.


We study the impact of domain distribution in PIDs extraction and router training by varying the
extraction and training data. Table 5 compares three configurations: (i) MATCHED-3: both extraction and training on _{_ AGNews, SST-2, TREC _}_ ; (ii) MISMATCHED: extraction on _{_ AGNews,
SST-2, TREC _}_ with _{_ CSQA, PIQA _}_ additionally included during training; (iii) MATCHED-5: extraction and training on all five datasets. Two key findings emerge. (1) Enlarging the training pool
without aligning the extraction (MISMATCHED) degrades performance in most cases, as the router
receives conflicting supervision signals that distort the extracted ICL patterns. (2) Jointly expanding both extraction and training (MATCHED-5) yields clear gains on OOD tasks, suggesting that
the extracted ICL pattern becomes more generalizable (providing empirical support for our claim
in Sec. 2.3). It also improves performance on ID tasks that appear unrelated to the added datasets
(e.g., AGNews, TREC). This indicates that heterogeneous domains provide complementary ICL
cues, enabling cross-task transfer and mutual reinforcement across domains.


5.3 ICR HIERARCHICALLY INTERNALIZES ICL DYNAMICS OF LLMS


In this section, we present a hierarchical importance analysis, spanning layers, heads, and PIDs,
which progresses from coarse to fine granularity. This reveals how ICR adaptively composes ICL
patterns across tasks and internalizes them at multiple levels of abstraction.


**Layer** We quantify per-layer contribution by combining two router signals: the mean head-level
gate strength and the averaged weights in the routing vectors ( _α_ ) across _r_ directions. For each input,
both streams are min–max normalized across layers, multiplied to form a layer-importance profile,
and renormalized to sum to one. We then report dataset-level means restricted to the intervened
layers. Figure 4 **Left** plots results for six representative datasets spanning ID, near-OOD, and farOOD groups. The curves show that a few hub layers (notably 23 and 26) consistently dominate,
suggesting that ICR identifies shared structural anchors for routing. Moreover, semantically related
datasets (e.g., SST-5/MR, CB/MRPC) exhibit nearly parallel profiles, indicating that ICR adaptively reweights layers in a task-aware yet structurally consistent manner. A more detailed analysis
with figures covering all 12 datasets is provided in Appendix I.


**Head** For each dataset, we record the gate values of all heads across layers for every zero-shot
input and average them to obtain per-head importance scores. The head with the highest average
value in each layer is selected as the Top-1 head, producing a routing sequence per dataset. We
analyze six representative datasets from in-domain, near OOD, and far OOD groups, and visualize
their routing sequences with a radar plot (Figure 4 **Middle** ). The consensus hubs, marked with green
stars, reveal that certain heads dominate the ICR process (e.g., head 26 in layer 22, head 21 in layer
23). In contrast, some layers exhibit task-specific divergence, where different tasks rely on different


8


Table 5: PIDs extraction and training with different domain combinations.


**Method** AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE


MATCHED-3 86.4 **87.6** 79.6 21.4 51.0 35.6 **80.2** 60.4 37.5 57.0 52.8 34.2
MISMATCHED 65.0 82.8 63.6 23.4 54.6 29.8 76.4 **64.0** 32.1 65.0 53.6 30.8
MATCHED-5 **86.6** 86.4 **83.8** **24.8** **57.0** **38.6** 79.8 53.4 **46.4** **68.0** **56.4** **37.2**


21


Figure 4: **Left** : Layer-importance visualization under ICR. **Middle** : Visualization of top-1 head in
each layer, with rings for heads, spokes for layers (starting at layer 21), and green stars marking
consensus heads (numbers denote head indices). **Right** : Correlation of per-dataset PID importance.


heads (e.g., at layer 28 the six tasks split across three heads, indicating three routing modes). These
results show that ICR identifies shared hub heads while flexibly adapting routing in non-hub layers.


**PIDs** We estimate per-dataset PID importance by combining the absolute weights in _α_ with the
average head-gate strength in each layer, and then averaging these weighted values across layers.
For each dataset, this yields a vector whose entries correspond to the importance of individual PIDs.
Pairwise Spearman correlations of these vectors are calculated and clustered (Figure 4 **Right** ). The
results show that ICR flexibly combines and routes along different ICL directions: for example, MR
aligns more with SST-2/TREC, while AI2SCIE and COPA correlate more with CSQA/PIQA,
reflecting a greater dependence on reasoning-oriented patterns than sentiment- or classificationoriented patterns. This differentiated behavior confirms that our attention routing-based design can
dynamically select and exploit relevant ICL directions, enabling adaptation across diverse OOD scenarios. These results demonstrate the deep alignment between ICR and the attention mechanisms,
which can benefit continually evolving transformer-based models.


6 RELATED WORK


**Implicit In-context Learning.** To better understand and exploit ICL, prior work has emphasized
the role of MHA. Building on these insights, researchers have proposed implicit ICL, which converts
ICDs into vectors injected into LLM activations, typically within MHA (Merullo et al., 2023). Task
Vectors (Hendel et al., 2023) are extracted from specific layers, while Function Vectors (Todd et al.,
2023) come from attention heads critical to ICL; both are applied during zero-shot inference to
provide task-relevant knowledge. Liu et al. (2023) modeled ICDs as shifts on MHA outputs and
introduced the in-context vector, while Peng et al. (2024); Jiang et al. (2025); Li et al. (2025a)
developed training strategies to enhance vector expressiveness. Although these methods alleviate
the latency and instability of token-level ICDs (Chen et al., 2022; Xiang et al., 2024), their limited
theoretical grounding in attention restricts generalization. Our approach, ICR, addresses this gap
and opens a new direction for implicit ICL. Additional related works are introduced in Appendix J.


7 CONCLUSION


We introduce In-Context Routing (ICR), a query-conditioned framework that extracts and exploits
generalizable ICL patterns within the MHA module of LLMs. Extensive experiments demonstrate
that ICR delivers robust performance across diverse ID and OOD tasks. Moreover, it requires only
a single round of training and transfers to new tasks without additional retrieval or retraining. By
operationalizing the mechanism of ICL within the implicit ICL paradigm, ICR improves both effec

9


0.40


0.35


0.30


0.25


0.20


0.15


0.10


0.05


0.00


ai2scie

piqa

csqa

copa

agnews


1.00


0.95


0.90


0.85


0.80


0.75


0.70


0.65


sst2

mr

cb

trec

mrpc

sst5

creak


29


24


21 22 23 24 25 26 27 28 29 30 31
Layer (global index)


27 26


agnews (ID)
csqa (ID)


mr (OOD)
cb (OOD)


creak (OOD)
ai2scie (OOD)


agnews
csqa


mr
cb


creak
ai2scie


tiveness and efficiency and further extends the benefits of ICL to tasks without labeled examples.
ICR provides valuable insights for reshaping zero-shot inference in the next generation of LLMs.


REPRODUCIBILITY STATEMENT


The LLMs adopted in this study are presented in Sec.4.1. The training procedures with full hyperparameter settings are reported in Appendix D.2, and details of the datasets used in this study are
provided in Appendix D.3.1. Due to our institution’s privacy policy and the requirements of double
blind review, we will release all code used for data reprocessing and for conducting experiments
upon the publication of the paper. The code will be distributed under a license that permits free use
for research purposes.


REFERENCES


Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about
physical commonsense in natural language. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, 2020.


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _Advances in neural information processing systems_, 33:1877–1901, 2020.


Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head
softmax attention for in-context learning: Emergence, convergence, and optimality. _arXiv preprint_
_arXiv:2402.19442_, 2024.


Yanda Chen, Chen Zhao, Zhou Yu, Kathleen McKeown, and He He. On the relation between
sensitivity and accuracy in in-context learning. _arXiv preprint arXiv:2209.07661_, 2022.


Hakaze Cho, Mariko Kato, Yoshihiro Sakai, and Naoya Inoue. Revisiting in-context learning inference circuit in large language models, 2025. URL [https://arxiv.org/abs/2410.](https://arxiv.org/abs/2410.04468)
[04468.](https://arxiv.org/abs/2410.04468)


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
_arXiv preprint arXiv:1803.05457_, 2018.


Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, and Furu Wei. Why can gpt
learn in-context? language models implicitly perform gradient descent as meta-optimizers. _arXiv_
_preprint arXiv:2212.10559_, 2022.


Chandler Davis and W. M. Kahan. The rotation of eigenvectors by a perturbation. iii. _SIAM Journal_
_on Numerical Analysis_, 7:1–46, 1970. doi: 10.1137/0707001. [URL https://doi.org/10.](https://doi.org/10.1137/0707001)
[1137/0707001.](https://doi.org/10.1137/0707001)


Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In _proceedings_ _of_ _Sinn_ _und_ _Bedeutung_,
volume 23, pp. 107–124, 2019.


Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In
_Third international workshop on paraphrasing (IWP2005)_, 2005.


Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu,
Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. _arXiv preprint arXiv:2301.00234_,
2022.


Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for
transformer circuits. _Transformer Circuits Thread_, 1(1):12, 2021.


Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn
in-context? a case study of simple function classes. _Advances in neural information processing_
_systems_, 35:30583–30598, 2022.


10


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, and et al. Akhil Mathur. The llama 3 herd of models, 2024. URL
[https://arxiv.org/abs/2407.21783.](https://arxiv.org/abs/2407.21783)


Qi Guo, Leiyu Wang, Yidong Wang, Wei Ye, and Shikun Zhang. What makes a good order of
examples in in-context learning. In _Findings_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics:_
_ACL 2024_, pp. 14892–14904, 2024.


Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. _arXiv_
_preprint arXiv:2310.15916_, 2023.


Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models.
_https://arxiv.org/abs/2106.09685_, 2021.


Yuchu Jiang, Jiale Fu, Chenduo Hao, Xinting Hu, Yingzhe Peng, Xin Geng, and Xu Yang. Mimic
in-context learning for multimodal tasks. In _Proceedings_ _of_ _the_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition Conference_, pp. 29825–29835, 2025.


Iain M Johnstone. On the distribution of the largest eigenvalue in principal components analysis.
_Annals of Statistics_, 29(2):295–327, 2001.


Ivan Lee, Nan Jiang, and Taylor Berg-Kirkpatrick. Is attention required for icl? exploring
the relationship between model architecture and in-context learning ability. _arXiv_ _preprint_
_arXiv:2310.08049_, 2023.


Xin Li and Dan Roth. Learning question classifiers. In _Proceedings of COLING_, 2002.


Yanshu Li, Yi Cao, Hongyang He, Qisen Cheng, Xiang Fu, Xi Xiao, Tianyang Wang, and Ruixiang
Tang. M [2] iv: Towards efficient and fine-grained multimodal in-context learning via representation
engineering. In _Second Conference on Language Modeling_, 2025a.


Yanshu Li, JianJiang Yang, Ziteng Yang, Bozheng Li, Hongyang He, Zhengtao Yao, Ligong Han,
Yingjie Victor Chen, Songlin Fei, Dongfang Liu, et al. Cama: Enhancing multimodal in-context
learning with context-aware modulated attention. _arXiv preprint arXiv:2505.17097_, 2025b.


Yanshu Li, Tian Yun, Jianjiang Yang, Pinyuan Feng, Jinfa Huang, and Ruixiang Tang. Taco: Enhancing multimodal in-context learning via task mapping-guided sequence configuration. _arXiv_
_preprint arXiv:2505.17098_, 2025c.


Zhuowei Li, Zihao Xu, Ligong Han, Yunhe Gao, Song Wen, Di Liu, Hao Wang, and Dimitris N
Metaxas. Implicit in-context learning. _arXiv preprint arXiv:2405.14660_, 2024.


Jiachang Liu, Dinghan Shen, Yizhe Zhang, Bill Dolan, Lawrence Carin, and Weizhu Chen. What
makes good in-context examples for gpt-3?, 2021. [URL https://arxiv.org/abs/2101.](https://arxiv.org/abs/2101.06804)
[06804.](https://arxiv.org/abs/2101.06804)


Sheng Liu, Haotian Ye, Lei Xing, and James Zou. In-context vectors: Making in context learning
more effective and controllable through latent space steering. _arXiv preprint arXiv:2311.06668_,
2023.


Yiting Liu and Zhi-Hong Deng. Iterative vectors: In-context gradient steering without backpropagation. In _Forty-Second_ _International_ _Conference_ _on_ _Machine_ _Learning_, 2025. URL
[https://openreview.net/forum?id=1v3XEcRMyP.](https://openreview.net/forum?id=1v3XEcRMyP)


Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Language models implement simple word2vecstyle vector arithmetic. _arXiv preprint arXiv:2305.16130_, 2023.


Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn
in context. _arXiv preprint arXiv:2110.15943_, 2021.


Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan,
Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction
heads. _arXiv preprint arXiv:2209.11895_, 2022.


11


Yasumasa Onoe, Michael JQ Zhang, Eunsol Choi, and Greg Durrett. Creak: A dataset for commonsense reasoning over entity knowledge. _arXiv preprint arXiv:2109.01653_, 2021.


Jane Pan. What in-context learning “learns” in-context: Disentangling task recognition and task
learning. Master’s thesis, Princeton University, 2023.


Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization
with respect to rating scales. In Kevin Knight, Hwee Tou Ng, and Kemal Oflazer (eds.), _Pro-_
_ceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05)_,
pp. 115–124, Ann Arbor, Michigan, June 2005. Association for Computational Linguistics. doi:
10.3115/1219840.1219855. [URL https://aclanthology.org/P05-1015/.](https://aclanthology.org/P05-1015/)


Yingzhe Peng, Xinting Hu, Jiawei Peng, Xin Geng, Xu Yang, et al. Live: Learnable in-context
vector for visual question answering. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 37:
9773–9800, 2024.


Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. Choice of plausible alternatives:
An evaluation of commonsense causal reasoning. In _AAAI spring symposium:_ _logical formaliza-_
_tions of commonsense reasoning_, pp. 90–95, 2011.


Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng,
and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment
treebank. In _Proceedings of EMNLP_, 2013.


Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question
answering challenge targeting commonsense knowledge. In _Proceedings of NAACL-HLT_, 2019.


Eric Todd, Millicent L Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau.
Function vectors in large language models. _arXiv preprint arXiv:2310.15213_, 2023.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher,
Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy
Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn,
Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel
Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee,
Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra,
Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi,
Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh
Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen
Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic,
Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models,
2023. [URL https://arxiv.org/abs/2307.09288.](https://arxiv.org/abs/2307.09288)


Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient
descent. In _International Conference on Machine Learning_, pp. 35151–35174. PMLR, 2023.


Futing Wang, Jianhao Yan, Yue Zhang, and Tao Lin. Elicit: Llm augmentation via external incontext capability. _arXiv preprint arXiv:2410.09343_, 2024a.


Qixun Wang, Yifei Wang, Yisen Wang, and Xianghua Ying. Can in-context learning really generalize to out-of-distribution tasks? _arXiv preprint arXiv:2410.09695_, 2024b.


Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. _Advances in neu-_
_ral information processing systems_, 33:5776–5788, 2020a.


Yaqing Wang, Quanming Yao, James T Kwok, and Lionel M Ni. Generalizing from a few examples:
A survey on few-shot learning. _ACM computing surveys (csur)_, 53(3):1–34, 2020b.


Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language
models. _arXiv preprint arXiv:2206.07682_, 2022.


12


Noam Wies, Yoav Levine, and Amnon Shashua. The learnability of in-context learning. _Advances_
_in Neural Information Processing Systems_, 36:36637–36651, 2023.


Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, and Lingpeng Kong. Self-adaptive in-context learning:
An information compression perspective for in-context example selection and ordering. _arXiv_
_preprint arXiv:2212.10375_, 2022.


Yanzheng Xiang, Hanqi Yan, Lin Gui, and Yulan He. Addressing order sensitivity of in-context
demonstration examples in causal language models. _arXiv preprint arXiv:2402.15637_, 2024.


Sang Michael Xie, Aditi Raghunathan, Percy Liang, and Tengyu Ma. An explanation of in-context
learning as implicit bayesian inference. _arXiv preprint arXiv:2111.02080_, 2021.


Steve Yadlowsky, Lyric Doshi, and Nilesh Tripuraneni. Pretraining data mixtures enable narrow
model selection capabilities in transformer models. _arXiv preprint arXiv:2311.00871_, 2023.


An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li,
Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang,
Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang,
Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL
[https://arxiv.org/abs/2412.15115.](https://arxiv.org/abs/2412.15115)


Kayo Yin and Jacob Steinhardt. Which attention heads matter for in-context learning? _arXiv_
_preprint arXiv:2502.14010_, 2025.


Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In _Advances in Neural Information Processing Systems (NeurIPS)_, 2015.


13


A SUPPLEMENTARY THEORETICAL ANALYSIS


A.1 KERNEL VIEW OF ATTENTION ROUTING
Self-attention can be viewed as a kernel machine, where the dot-product _q_ _[⊤]_ _k_ defines a _linear kernel_
_K_ 0( _q, k_ ) = _q_ _[⊤]_ _k_ . From this perspective, attention routing does not merely add a bias to the logits,
but reparameterizes the kernel itself. Formally, let _Q_ _[l]_ **zs** _[, K]_ **zs** _[l]_ _[∈]_ [R] _[T][ ×][d]_ [be the layer-level projections]
during zero-shot inference. We define a reparameterized kernel function


_Kα_ _[l]_ [(] _[q, k]_ [) =] _[q][⊤][M][ l]_ [(] _[α][l]_ [)] _[ k,]_ (15)


where the reparameterization matrix is


_M_ _[l]_ ( _α_ _[l]_ ) = _Id_ + _Uq_ _[l]_ [diag(] _[α][l]_ [)] _[ U]_ _k_ _[ l][⊤][.]_ (16)


Here _Uq_ _[l][, U]_ _k_ _[ l]_ _[∈]_ [R] _[d][×][r]_ [are the PID bases and] _[ α][l]_ _[∈]_ [R] _[r]_ [is the routing vector.] [The resulting correction]
is
∆ _A_ _[l]_ = _Q_ _[l]_ **zs** _[M][ l]_ [(] _[α][l]_ [)] _[K]_ **zs** _[l]_ _[−]_ _[Q]_ **zs** _[l]_ _[K]_ **zs** _[l]_ _[,]_

which is then broadcast to heads to produce


_A_ ˜ _[l,h]_ = _A_ _[l,h]_ + ∆ _A_ _[l]_ _._


This kernel view shows that attention routing replaces the fixed linear kernel with a reparameterized
kernel whose deviation from _K_ 0 is low-rank, since rank( _M_ _[l]_ ( _α_ _[l]_ ) _−_ _I_ ) _≤_ _r_ . The modification is
structural, as it is confined to PID directions.


A.2 SPIKED COVARIANCE MODEL

The _spiked_ _covariance_ _model_ (Johnstone, 2001) is a widely studied framework in random matrix
theory and high-dimensional statistics. It assumes that the population covariance matrix Σ _∈_ R _[d][×][d]_
can be decomposed into an isotropic noise component plus a small number of low-rank “spikes”:


where _σ_ [2] _Id_ represents homogeneous noise, _ui_ _∈_ R _[d]_ are orthonormal eigen-directions corresponding to signal components, and _θi_ are the spike strengths (eigenvalues above the noise level). In
this setting, most eigenvalues of Σ concentrate around _σ_ [2], while a few leading eigenvalues (the
spikes) separate from the bulk, capturing the essential low-dimensional structure of the data. This
model provides the foundation for our mixed spiked formulation, where we separate shared lowdimensional attention structures from domain-specific variations to analyze in-context reasoning
signals across datasets.


A.3 FORMAL ANALYSIS OF POOLED PCA

We provide a high-level analysis supporting the claim in Sec. 2.3 that pooled PCA over multiple
domains can better recover the general ICL pattern. Our argument is based on the classical Davis–
Kahan sin Θ theorem (Davis & Kahan, 1970), which bounds the deviation between the estimated
and true subspaces under perturbations. Let _U_ [�] _q_ be the top- _r_ eigenspace of the pooled covariance
Σ� _Q_, and let _Sq_ denote the ground-truth shared subspace. Then

sin Θ�span( _U_ [�] _q_ ) _,_ span( _Sq_ )� ≲ _O_ ˜( _N_ _[−]_ [1] _[/]_ [2] ) + _ρD_ _,_ (18)

gap _Q_


where gap _Q_ is the eigengap separating the shared spikes from the bulk spectrum. Here, sin Θ( _U, V_ )
denotes the operator norm of the sine of the canonical angles between subspaces _U_ and _V_ . The
numerator of the bound contains two sources of error: the _O_ [˜] ( _N_ _[−]_ [1] _[/]_ [2] ) term from finite-sample noise
and the residual _ρD_ from domain-specific variations. Both decrease with larger _N_ and _D_ : increasing
_N_ reduces sampling fluctuations, while increasing _D_ averages out heterogeneous domain-specific
directions.


At the same time, the denominator gap _Q_ becomes larger as _N_ and _D_ grow. With more samples, the
leading eigenvalues of the shared component are estimated more accurately, and with more domains,


14


Σ =


_r_

- _θiuiu_ _[⊤]_ _i_ [+] _[ σ]_ [2] _[I][d][,]_ (17)

_i_ =1


domain-specific contributions cancel out, making the shared spikes stand out more prominently from
the bulk.


Together, these effects tighten the Davis–Kahan bound: the numerator shrinks while the denominator
enlarges, so the subspace distance sin Θ( _U_ [�] _q, Sq_ ) decreases. Consequently, pooled PCA on multidomain ICL bases becomes increasingly reliable for recovering the shared subspace _Sq_ .


A.4 PERTURBATION ANALYSIS OF OOD STABILITY


We continue our analysis by showing that the shared ICL subspace recovered by pooled PCA is not
only stable under test-time distribution shifts but also becomes more accurate for out-of-distribution
(OOD) generalization as the number of domains increases. Specifically, we model OOD shifts in
the query/key statistics as additive perturbations to the pooled covariances:


Σ� _[′]_ _Q_ [=] [Σ][�] _[Q]_ [+ ∆] _[Q][,]_ Σ� _[′]_ _K_ [=] [Σ][�] _[K]_ [+ ∆] _[K][,]_


where _∥_ ∆ _Q∥_ op _, ∥_ ∆ _K∥_ op _≤_ _ϵ_ capture bounded changes in second-order statistics.


Let _Uq_ be the top- _r_ eigenspace of Σ [�] _Q_, and _U_ [�] _q_ be the corresponding eigenspace of the perturbed
matrix Σ [�] _[′]_ _Q_ [.] [The Davis–Kahan][ sin Θ][ theorem (Davis & Kahan, 1970) gives the bound:]

sin Θ�span( _U_ [�] _q_ ) _,_ span( _Uq_ )� _≤_ _[∥]_ [∆] _[Q][∥]_ [op] (19)

gap _Q_


Thus, the subspace stability depends on the relative size of the perturbation versus the eigengap. An
identical argument applies to _Uk_ .


Importantly, pooling across multiple domains helps enlarge gap _Q_ by amplifying the shared signal
while averaging out domain-specific variations (see Sec.2.3). This increases the separation between
the top- _r_ eigenvalues and the noise floor, which tightens the Davis–Kahan bound and ensures that the
perturbed subspace _U_ [�] _q_ remains closer to the in-domain subspace _Uq_ under test-time shifts. Together,
these explain why increasing the number of training domains leads to more reliable OOD routing in
practice.


B CHALLENGES OF VECTOR-BASED IMPLICIT ICL


Although vector-based methods can reproduce certain _input-output_ _statistics_ of ICDs and enable
efficient ICL without token-level ICDs, they suffer from two fundamental challenges.


1. **Weak** **theoretical** **grounding** **limits** **scalability.** Vector-based methods convert certain explicit
ICDs into free-form residual biases of a specific model **without** structural connections to the
query/key space, which makes them relatively black-box and detached from the theoretical framework of MHA. Thus, these methods witness large performance fluctuations when transferred across
architectures. Moreover, incorporating new knowledge into these vectors or resizing them to fit
novel models requires curated training, and the results of such training can also be unstable.


2. **Post-hoc residual steering limits generalization.** Vector-based implicit ICL intervenes only after
attention aggregation, injecting additive shifts into the MHA output. Such post-hoc adjustments
lack structural control: the resulting representations are often entangled with task-specific content,
limiting their ability to transfer beyond the training task. Since the underlying attention logits **A** _[l,h]_,
which more fundamentally encode ICL patterns, remain unaffected, the model tends to mimic ICL
by fitting specific feature patterns rather than developing the attention dynamics needed to exploit
context. This design inherits the potential attention deficits in explicit ICL (Lee et al., 2023), while
also lacking the adaptability necessary for multi-task or OOD scenarios.


C ICR PSEUDOCODE


ICR consists of two key phases: PIDs extraction and router training. Algorithm 1 presents the
pseudocode for multi-task query/key representation collection and the subsequent PIDs extraction,
while Algorithm 2 illustrates the core mechanism and training procedure of ICR.


15


**Algorithm 1:** Collecting PIDs _Uq, Uk_ across multiple domains

**Input:** Model _M_, datasets _{D_ 1 _, . . ., DN_ _}_ with _Mn_ prompts each, target layers _L_, PCA rank _r_
**Output:** _Uq_ _[l][, U]_ _k_ _[ l]_ [for each] _[ l][ ∈]_ _[L]_

**1** **foreach** _l ∈_ _L_ **do**

**2** _Q_ pool[ _l_ ] _←∅_, _K_ pool[ _l_ ] _←∅_


**6** Run _M_ ( _p_ ) with Q/K hooks;

**7** **foreach** _l ∈_ _L_ **do**

**8** _Q_ _[l]_ last _[←]_ [Concat] _h_ _[H]_ =1 _[Q][l,h]_ [[] _[t]_ [last][]][;]

**9** _K_ last _[l]_ _[←]_ [Concat] _h_ _[H]_ =1 _[K][l,h]_ [[] _[t]_ [last][]][;]

**10** Append _Q_ _[l]_ last [to] _[ Q]_ [pool][[] _[l]_ []][;]

**11** Append _K_ last _[l]_ [to] _[ K]_ [pool][[] _[l]_ []][;]


**12** **foreach** _l ∈_ _L_ **do**

**13** _Q ←_ Concat( _Q_ pool[ _l_ ]);

**14** _K_ _←_ Concat( _K_ pool[ _l_ ]);

**15** _Uq_ _[l]_ _[←]_ [Top-] _[r]_ [ PCA][(] _[Q]_ [)][;]

**16** _Uk_ _[l]_ _[←]_ [Top-] _[r]_ [ PCA][(] _[K]_ [)][;]

**17** Save _Uq_ _[l][, U]_ _k_ _[ l]_ [;]


D EXPERIMENTAL SETUP


D.1 COLLECTION DETAILS


To construct the ICL bases, we collect 10k examples from AGNEWS and 5k examples each from
SST-2, TREC, CSQA, and PIQA. This allocation is motivated by the complementary characteristics of these datasets: AGNEWS focuses on topic-level categorization that captures broad semantic
content, SST-2 and TREC emphasize sentence-level classification with a sharper focus on specific linguistic distinctions, while CSQA and PIQA represent QA-style tasks that require more
reasoning-oriented processing. Overall, this balanced collection is designed to provide approximately uniform coverage of semantic, classification, and reasoning patterns.


D.2 TRAINING DETAILS

Optimization uses AdamW (lr 1 _×_ 10 _[−]_ [4], batch size 4) for 2 epochs with gradient clipping (1 _._ 0)
and PIDs rank _r_ =8. The training objective combines cross-entropy with a confidence-improvement
term ( _λ_ conf=0 _._ 01), an _ℓ_ 1 sparsity penalty on routing vectors ( _λ_ spar=10 _[−]_ [3] ), and a gate sparsity term
( _λ_ gate=0 _._ 02). To stabilize training, we employ two simple schedules: (i) a late-layer weighting
scheme that increases sparsity strength toward the late layers (up to 3 _._ 0) ( _w_ _[l]_ in Eq.13), and (ii) a
cosine annealing of the routing scale _α_ across epochs (from 1 _._ 0 to 0 _._ 8). Inputs to both the encoder
and the LLM are truncated to 512 tokens. All runs use a single V100 GPU under deterministic
settings (seed 42; TF32 and non-deterministic SDPA disabled).


D.3 EVALUATION DETAILS


Predictions follow a unified next-token scoring protocol: each answer option is mapped to the variant
that tokenizes into a single token, and the prediction is taken as the arg max over the logits at the
next position restricted to these candidate ids. When ICR is enabled, the router is conditioned on a
mean-pooled MiniLM sentence embedding, while the backbone remains frozen.


D.3.1 DATASETS

**In-Domain** We treat the five datasets used for cross-domain collection and router training as indomain: AGNews, SST-2, TREC, CSQA, and PIQA. **AGNews** provides large-scale topic classification over news articles spanning four domains. **SST-2** evaluates binary sentiment classification on
movie reviews, emphasizing subtle polarity cues. **TREC** focuses on open-domain question classification into several semantic types. **CSQA** targets commonsense reasoning through multiple-choice


16


**3** **foreach** _dataset Dn_ **do**


**4** **for** _i ←_ 1 **to** _Mn_ **do**


**5** _p ←_ GenerateFewShotPrompt( _Dn_ );


**Algorithm 2:** In-Context Routing (ICR) training

**Input:** Frozen backbone _M_ with _L_ layers and _H_ heads per layer;
Frozen encoder _E_ ; Router MLP with parameters _θ_ = ( _θα, θγ_ );
Subspaces _{Uq_ _[l][, U]_ _k_ _[ l]_ _[}][L]_ _l_ =1 [;]
Late-layer set _L_ late = _{_ [2] 3 _[L]_ [+ 1] _[, . . ., L][ }]_ [;]

Datasets _{D_ 1 _, . . ., DN_ _}_, where each sample is ( _x, y, d_ ) with input _x_, label _y_, dataset index _d_ ;
Optimizer Opt( _θ_ )
**Output:** Trained router parameters _θ_

**1** **while** _not converged_ **do**

// sample a minibatch from the union of datasets

**2** _B_ _←{_ ( _xi, yi, di_ ) _}_ _[B]_ _i_ =1 [;]

**3** **for** _i_ = 1 **to** _B_ **do**

**4** _zi_ _←_ _E_ ( _xi_ ) ; // pooled representation of the query

**5** ( _αi, γi_ ) _←_ RouterMLP _θ_ ( _zi_ ) ; // _αi_ [ _l,_ :] _∈_ R _[r]_, _γi_ [ _l, h_ ] _∈_ R

**6** **for** _l_ = 1 **to** _L_ **do**

**7** **if** _l_ _∈L/_ late **then continue** ;

**8** _Q ←_ _Wq_ _[l][h][l]_ [(] _[x][i]_ [)] _[,]_ _K_ _←_ _Wk_ _[l]_ _[h][l]_ [(] _[x][i]_ [)] _[,]_ _V_ _←_ _Wv_ _[l][h][l]_ [(] _[x][i]_ [)][;]

**9** _Zq_ _←_ _Q Uq_ _[l]_ _[∈]_ [R] _[T][ ×][r][,]_ _Zk_ _←_ _K Uk_ _[l]_ _[∈]_ [R] _[T][ ×][r]_ [;]

**10** _B_ shared _←_ einsum( _Zq,_ _αi_ [ _l,_ :] _,_ _Zk_ ) _∈_ R _[T][ ×][T]_ ;

**11** **for** _h_ = 1 **to** _H_ **do**


**15** _O_ [(] _[h]_ [)] _←_ _A_ [(] _[h]_ [)] _V_ [(] _[h]_ [)] ;

// replace layer- _l_ attention output

**16** _O_ ˜ _[l]_ _←_ Concat _[H]_ _h_ =1 _[O]_ [(] _[h]_ [)] _[ W][ O]_


// obtain task logits at the last token

**17** _ℓi_ _←_ _M_ ( _xi_ )��last-token

// training loss

**18** _L_ task _←{ℓ_ [ICR] _i_ _, yi}_ ;

**19** _L_ conf _←{ℓ_ [ICR] _i_ _, ℓ_ [zs] _i_ _[}]_ [ ;]

**20** _Lα_ -spar _←{α}_ ;

**21** _Lγ_ -spar _←{γ}_ ;

**22** _L ←_ _L_ task + _λ_ conf _L_ conf + _λαLα_ -spar + _λγLγ_ -spar // router update

**23** Opt _._ zero ~~g~~ rad(); _∇θL_ ; Opt _._ step()


questions grounded in everyday knowledge. **PIQA** assesses physical knowledge by requiring plausibility judgments over everyday actions.


**Out-of-Domain** For out-of-domain (OOD) evaluation, we consider seven representative datasets
that are disjoint from the collection and training sources. We divide them into _near_ _OOD_ and _far_
_OOD_ groups depending on their proximity to the training tasks in terms of domain, label space, and
input format.


Firstly we articulate a clear framework for how we distinguish near-OOD and far-OOD. In the
context of ICL generalization, we consider three key axes: (i) the compatibility of the label ontology
with the ID family, (ii) the similarity of the required operation or reasoning structure, and (iii) the
degree of semantic and domain shift. Under this framework, near-OOD tasks are those that deviate
from ID along at most one of these axes while preserving the overall ICL structure. SST-5, MR, and
MRPC fall into this category. SST-5 extends SST-2 from a binary polarity to a five-point sentiment
scale, so the label ontology changes in granularity but remains sentiment-based, while the operation
type and domain are essentially identical. MR keeps the same binary sentiment ontology as SST2 and the same sentence-level classification operation, but shifts the underlying review corpus, so
the main change is semantic and domain shift. MRPC, although cast as sentence-pair paraphrase,


17


**12** _B_ head [(] _[h]_ [)] _[←]_ _[γ][i]_ [[] _[l, h]_ []] _[ ·][ B]_ [shared][ ;]


**13** _S_ [(] _[h]_ [)] _←_ _[Q]_ [(] _[h]_ ~~_√_~~ [)] _[K]_ [(] _[h]_ [)] _[⊤]_


~~_√_~~ _[K]_ _dk_ + _B_ head [(] _[h]_ [)] [;]


               **14** _A_ [(] _[h]_ [)] _←_ softmax _S_ [(] _[h]_ [)][�] ;


Table 6: Datasets, task types, and prompt templates used in ICR.


**Dataset** **Task Type** **Template**


AGNews Topic classification News: _{_ text _}_ ; Type: [World, Sports, Business, Technology]


SST-2 Sentiment (binary) Review: _{_ text _}_ ; Sentiment: [negative, positive]


TREC Question type classification Question: _{_ text _}_ ; Answer Type: [Abbreviation, Entity,
Description, Person, Location, Number]


CSQA Commonsense MCQ (5-class) Question: _{_ question _}_ ; A. _{_ optA _}_ ; B. _{_ optB _}_ ; C. _{_ optC _}_ ;
D. _{_ optD _}_ ; E. _{_ optE _}_ ; Answer (A/B/C/D/E); Options:

[A, B, C, D, E]


PIQA Physical commonsense (2-choice) Goal: _{_ goal _}_ ; A. _{_ optA _}_ ; B. _{_ optB _}_ ; Answer (A/B); Options: [A, B]


SST-5 Sentiment (5-class) Sentence: _{_ text _}_ ; Sentiment: [terrible, negative, neutral,
positive, great]


MR Movie Review (binary) Review: _{_ text _}_ ; Sentiment: [negative, positive]


MRPC Paraphrase _{_ pair _}_ ; A. Paraphrase; B. Not paraphrase; Answer (A/B);
Options: [A, B]


CB NLI (3-class) _{_ pair _}_ ; A. Entailment; B. Contradiction; C. Neutral; Answer (A/B/C); Options: [A, B, C]


CREAK Claim verification Claim: _{_ claim _}_ ; Label: yes / no; Options: [yes, no]


COPA Causal reasoning (2-choice) _{_ context _}_ ; A. _{_ optA _}_ ; B. _{_ optB _}_ ; Answer (A/B); Options: [A, B]


AI2SciE Science MCQ (K-choice) Question: _{_ question _}_ ; A. _{_ optA _}_ ; B. _{_ optB _}_ ; C. _{_ optC _}_ ;
D. _{_ optD _}_ ; E. _{_ optE _}_ ; F. _{_ optF _}_ ; G. _{_ optG _}_ ; H. _{_ optH _}_ ;
Answer (A/B/C/...); Options: [A, B, C, D, E, F, G, H]


still uses a binary decision geometry that is compatible with SST-2 and the A/B decision format of
PIQA, and the reasoning required is largely surface-level alignment such as lexical and syntactic
rewrites. In this sense it introduces a mild change in operation type but remains close to ID along
label ontology and general language style. By contrast, far-OOD tasks shift along multiple axes at
once. CB, COPA, CREAK, and AI2SciE all introduce new label inventories and reasoning structures
that are not present in any ID dataset, together with nontrivial semantic shifts. CB is a three-way NLI
task with labels entailment, contradiction, and neutral, which do not align with any ID label space,
and it requires directional inference from premise to hypothesis. COPA formulates explicit causal
reasoning over alternatives, which differs from the recognition-style decisions in ID and entails
a different type of relational reasoning. CREAK focuses on claim verification, relying on world
knowledge and on reasoning about when seemingly plausible statements fail in specific cases, rather
than on shallow sentence-level judgments. AI2SciE requires scientific explanatory reasoning over
domain-specific content that is absent from the ID datasets. Together, these shifts in label ontology,
operation type, and semantics alter the effective ICL geometry in more than one dimension.


_Near_ _OOD._ **SST-5** evaluates fine-grained sentiment prediction beyond the binary labels seen in
training, requiring models to calibrate over a five-class space. **MR** further tests domain transfer
by shifting sentiment analysis to the movie-review domain. Finally, **MRPC** evaluates robustness
under input format shift, where the model must generalize from single-sentence classification to
sentence-pair paraphrase detection. These tasks remain relatively close to the training distribution
(sentiment or classification-style tasks) but introduce moderate shifts in label granularity, domain,
or input structure.


_Far_ _OOD._ In contrast, **CommitmentBank** **(CB)** stresses generalization under shifts in semantic
judgment criteria, where decisions hinge on subtle pragmatic or syntactic cues absent from typical
training tasks. **COPA** introduces a pairwise choice format grounded in causal reasoning. **CREAK**
evaluates plausibility judgments in commonsense relational contexts. Finally, **AI2SciE** requires
elementary science question answering, representing a shift toward multi-hop reasoning. These


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


datasets constitute far OOD scenarios, as they deviate more substantially from the training distribution in both task format and reasoning requirements.


Taken together, the near and far OOD sets cover complementary axes of generalization, ranging
from finer-grained variants of familiar tasks to entirely novel reasoning paradigms, thus providing a
comprehensive testbed for out-of-domain robustness. On these datasets we report comparisons only
with zero-shot and few-shot prompting, since current vector- or retrieval-based methods require
labeled in-domain ICDs and are not directly applicable.


**Templates** The datasets used for extraction, training, and evaluation are listed in Table 6, along
with their task types and templates. For in-domain datasets, the templates serve a dual role: they
are applied when constructing ICL prompts prior to collecting query/key representations for PCAbased PIDs extraction, and again during evaluation. For out-of-domain datasets, the templates are
employed only for evaluation.


D.3.2 PRELIMINARY EXPERIMENT SETUP

For the preliminary cross-task ICL experiments in Section 1 (Figure 1), the inference-time model is
Llama-2-7B, and all prompts follow the template shown in Table 6. Each experiment uses a total
of 16 in-context demonstrations. In the single-source setting, we sample 16 demonstrations without
replacement from the training split of a single source task. In the cross-task setting, we sample 8
demonstrations from each of two source tasks, concatenate them, and uniformly shuffle their order
before inserting them into the prompt. In both settings, demonstrations from any dataset are selected
using label-balanced sampling.


For evaluation, each target task is assessed on a subset of 500 test instances, and we report accuracy
averaged over 5 independent seeds. Decoding is performed using greedy search.


D.3.3 BASELINES

For in-domain evaluation, we compare our method against several representative vector-based implicit ICL baselines, including Task Vector (TV), Function Vector (FV), In-Context Vector (ICV),
ELICIT, Iterative Vectors (IV), Implicit ICL (I2CL), Learnable In-context VEctor (LIVE), and
M2IV, in addition to standard zero-shot and few-shot prompting. For out-of-domain evaluation,
we select three methods that involve calibration or training with data: I2CL, LIVE, and M [2] IV, as
other training-free methods **cannot** be applied to OOD tasks. For methods requiring training, we
follow the original setups and conduct a hyperparameter search to achieve the best performance.
The details of the baselines are as follows:


    - Task Vector (TV): TV frames ICL as compressing the demonstrations into a single task
vector that encodes the task rule. This vector is then patched into the transformer’s intermediate layers during the query’s forward pass, steering the model’s prediction without
direct access to the demonstrations.


    - Function Vector (FV): FVs identify a small set of causal attention heads that transport
a compact vector representation of the demonstrated task during ICL. By extracting this
function vector and inserting it into the hidden states of new contexts, the model can execute
the task in zero-shot or natural text settings. The approach shows that LLMs internally
encode portable and composable task representations.


    - In-Context Vector (ICV): ICVs recast ICL by extracting a single vector from the latent
states of demonstration examples, which summarizes the task. At inference, this vector is
added to the hidden states of all layers during the query’s forward pass. This approach improves controllability, reduces context length, and supports vector arithmetic for combining
tasks.


    - ELICIT: ELICIT introduces a modular framework that builds a capability library of task
vectors extracted from in-context learning prompts. At inference, a retrieval module dynamically selects and injects relevant task vectors into the model’s hidden states, enabling
it to reuse learned capabilities without extra tokens or fine-tuning.


    - Iterative Vectors (IV): IVs enhance ICL by extracting activation-based meta-gradients,
the differences between activations with and without demonstrations, and refining them


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


Table 7: Baseline comparison across benchmarks. [*] For ID datasets, few-shot uses 5-shot balanced
sampling per class. For OOD datasets, we adopt multi-task few-shot prompting where each ID
dataset provides 3-shot ICDs. Under **Overall**, _Average_ is the mean accuracy across all datasets, and
_Collapse_ counts datasets where a method underperforms the zero-shot baseline.


**In-Domain (ID)** **Near OOD** **Far OOD** **Overall**
**Method**

AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE **Average** **Collapse**


**Llama3.1-8B**


Zero-shot 70.0 87.8 49.0 65.0 62.6 27.6 82.2 68.8 41.1 65.0 53.6 78.4 62.6  Few-shot [*] 88.2 91.4 57.4 72.8 70.4 42.2 91.8 72.4 51.4 63.0 50.8 89.6 70.1 2


I2CL 79.8 86.4 63.8 66.2 62.0 30.8 82.0 64.8 40.6 61.2 46.8 61.4 62.2 8
LIVE 82.6 87.8 66.0 66.8 61.4 32.4 78.6 69.0 41.8 58.8 51.0 65.2 63.5 5
M [2] IV 83.4 88.2 64.8 **67.2** 64.8 35.0 81.8 67.8 42.6 60.8 49.8 67.6 64.5 5
**ICR** **85.2** **88.6** **76.8** 66.6 **66.4** **36.6** **83.6** **69.4** **42.9** **67.0** **54.6** **82.6** **68.4** **0**


Table 8: Comparison of ICR and LoRA. **Param.** denotes the number of trainable parameters relative
to ICR (with ICR set as _×_ 1 _._ 0).


**In-Domain (ID)** **Near OOD** **Far OOD** **Overall**
**Method**

AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE **Average** **Param.**


**Qwen2.5-7B**


LoRA **83.6** **93.2** **71.6** **84.0** **84.2** 40.8 88.5 73.2 83.0 92.6 74.6 91.5 80.1 _×_ 2 _._ 1
**ICR** 80.4 91.0 70.6 82.0 82.6 **41.4** **89.4** **73.2** **84.6** **95.0** **79.2** **93.2** **80.2** _×_ 1 _._ 0


**Llama-3.1-8B**


LoRA **86.8** **90.4** **77.2** **67.2** 65.8 **37.4** 83.0 69.0 40.0 65.4 52.6 79.8 67.9 _×_ 2 _._ 8
**ICR** 85.2 88.6 76.8 66.6 **66.4** 36.6 **83.6** **69.4** **42.9** **67.0** **54.6** **82.6** **68.4** _×_ 1 _._ 0


through an iterative process. These vectors are then injected back into the model’s activations during inference, effectively simulating gradient updates without backpropagation.


    - Implicit ICL (I2CL): I2CL extracts vectors from each ICD and aggregates them into a
unified context vector. During inference, it injects a linear combination of this context
vector and the query activations into each layer’s residual streams to simulate the effect of
ICL. Additionally, I2CL employs a noisy self-calibration step to optimize the layer-wise
fusion coefficients.


    - Learnable In-context VEctor (LIVE): LIVE distills task information from ICDs into a set
of learnable vectors. During training, it aligns the model’s outputs using ICDs with those
using LIVE, and at inference, the learned vectors are added to each layer’s MHA outputs
to simulate the effect of ICDs.


    - M [2] IV: M²IV assigns learnable vectors and weight factors to both the MHA and MLP
branches at each layer of an LVLM. During training, it uses a self-distillation framework
with mimicry, synergistic, and supervised losses to align with Vanilla ICL outputs. At inference, the trained vectors are injected into residual streams to emulate n-shot ICL without
explicit ICDs.


E ADDITIONAL RESULTS


E.1 RESULTS ON LLAMA3.1-8B

Table 7 presents additional results comparing ICR with zero-shot, few-shot, and baseline methods
on Llama3.1-8B. Overall, ICR approaches and sometimes surpasses few-shot performance, while
consistently outperforming other task-specific implicit ICL baselines in both accuracy and stability.
Notably, ICR shows no collapses below zero-shot performance on any OOD task, outperforming
multi-task few-shot prompting and all other baselines. This reinforces our conclusions in Sec. 4.2.


E.2 COMPARISON WITH LORA

We further compare ICR with LoRA in Table 8. The LoRA module is applied to the token classification head of the last layer with rank 32. For training, we use the same number of few-shot
examples as those contained in an ICL prompt during the construction of ICL bases, drawn from
five in-domain datasets. Although LoRA requires 2–3 _×_ more trainable parameters than ICR, it
achieves slightly weaker overall performance. Moreover, ICR exhibits clear advantages in OOD settings, which shows its better generalizability and efficiency compared to the PEFT-based methods
in few-shot scenarios.


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


Table 9: Baseline comparison across benchmarks on Qwen3-32B and Llama3.1-70B. [*] For ID
datasets, few-shot uses 5-shot balanced sampling per class. For OOD datasets, we adopt multitask few-shot prompting where each ID dataset provides 3-shot ICDs.


**In-Domain (ID)** **Near OOD** **Far OOD** **Overall**
**Method**

AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE **Average**


**Qwen3-32B**


Zero-shot 69.8 83.0 51.2 76.0 78.0 46.0 95.2 75.0 91.1 96.0 51.8 85.8 74.9
Few-shot [*] 84.4 89.8 77.8 86.8 89.0 43.8 99.4 76.4 91.1 98.0 51.4 80.3 80.7


FV 74.6 82.6 58.2 75.2 63.4 35.6 93.4 74.8 85.2 93.8 47.0 79.4 71.9
I2CL 78.4 85.6 64.7 74.6 74.2 38.6 94.8 76.0 89.6 94.2 52.0 84.6 75.6
**ICR** **81.6** **86.4** **77.2** **82.0** **82.8** **50.4** **97.2** **79.8** **94.6** **96.0** **53.6** **88.6** **80.9**


**Llama3.1-70B**


Zero-shot 48.8 93.2 62.6 80.2 70.4 44.0 82.0 71.4 91.0 96.0 92.6 68.6 77.6
Few-shot [*] 70.8 91.0 68.0 83.2 88.6 46.4 85.2 78.6 92.9 97.0 88.0 96.2 82.2


FV 52.4 86.4 58.4 75.6 71.2 42.6 78.8 68.4 85.6 86.4 85.8 80.8 72.7
I2CL 62.6 88.8 63.0 73.8 75.4 46.8 77.6 70.0 90.2 89.0 88.0 84.4 75.8
**ICR** **66.4** **93.8** **66.0** **82.4** **83.2** **48.4** **86.8** **80.2** **93.4** **92.0** **93.2** **92.0** **81.5**


E.3 RESULTS ON MODELS WITH LARGER SCALE


To test the robustness of ICR on larger-scale models, we additionally report experiments on increased model sizes in Table 9. Across the models with increasing scale, ICR consistently outperforms the two residual-injection baselines. It approaches few-shot performance in in-domain
settings and typically surpasses cross-task few-shot prompting in OOD settings, with only a few
exceptions where the large-scale model is already very strong and results become slightly unstable. These trends further validate that ICR achieves stable transfer by avoiding reliance on noisy
cross-task ICDs, and demonstrate its effectiveness across models of different scales.


F EFFICIENCY ANALYSIS


To assess the efficiency of In-Context Routing (ICR), we benchmark it against baselines along two
dimensions. Following Li et al. (2024), we report cached parameter size in Table 11. For ICR,
the cached parameter is 2 _rdL_, as both _Uq_ and _Uk_ of the shape _d × r_ must be stored in each layer.
Although this appears larger than some baselines, _r_ is typically a small constant (e.g., 4–16), so
the asymptotic complexity remains _O_ ( _dL_ ), on par with methods such as I2CL or LIVE. Moreover,
since _r_ _≪_ _M_ in few-shot settings, ICR still provides a far lighter memory footprint compared to
explicit ICL.


We also report the average per-sample inference time over five in-domain datasets in Figure 5.
The results show that ICR consistently requires less inference time than the 5-shot setting. More
importantly, as the input length increases, the inference time of few-shot grows much faster than
that of ICR. This demonstrates that ICR preserves the efficiency of implicit ICL, with the advantage
becoming especially pronounced for longer contexts.


For a better understanding of offline computational cost of ICR, we provide an explicit comparison
with I2CL, M2IV, and LIVE in Table 10. The cost is measured in NVIDIA V-100 GPU hours for
representation collection and training/calibration. Because ICR extracts PIDs and trains the router
using five in-domain datasets, and the baselines are not inherently designed for OOD scenarios,
we report their offline cost on ID tasks only. Concretely, ICR performs a single round of representation collection and training shared across all in-domain datasets, while the baselines are run
and trained/calibrated separately for each task, as they are originally designed, and their offline cost
is obtained by averaging over the five ID tasks. From the results, our ICR, as a train-once-andreuse method, has a comparable GPU hour cost to the per-task averages reported for the baselines
(except I2CL, which only performs calibration rather than training).. This implies that once two
or more tasks are evaluated, the amortized cost of ICR becomes lower, making our method more
time-efficient in practice.


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


Table 10: GPU hours comparison across methods.


**Method** **I2CL** **LIVE** **M2IV** **ICR**


GPU Hours 1.2h 6.8h 7.0h 9.7h


Table 11: **Cached** **parameter** **size** of different methods. _M_ = #demonstration tokens, _d_ = hidden
dimension, _L_ = #layers, _r_ = PID subspace rank ( _r_ _≪_ _M_ ).


Method Zero-shot Few-shot TV FV ICV I2CL LIVE M [2] IV ICR


Cached Param. 0 2 _MdL_ _d_ _d_ _dL_ 2 _dL_ _dL_ 2 _dL_ 2 _rdL_


G ADDITIONAL ABLATION STUDY


G.1 PIDS EXTRACTION

In Sec. 4.3 we reported the impact of varying the PCA rank and replacing PCA with a random basis.
Here we provide additional details and observations.


For the random orthogonal subspace ( _r_ = 8), we generate a _d × r_ Gaussian matrix per layer and
apply QR decomposition to obtain an orthogonal basis. This ensures the comparison isolates the
role of PCA-extracted directions from generic low-rank projections.


While Sec.4.3 reports the performance trade-offs, we note that the degradation at _r_ = 12 is not only
consistent across settings but also more unstable across runs, suggesting that the enlarged subspace
introduces degrees of freedom that remain under-trained with fixed data and epochs. This further
supports the interpretation that OOD robustness benefits from a carefully constrained subspace.


Although in-domain accuracy is relatively preserved under the random basis (indicating the model
can adapt with enough supervision), both near- and far-OOD performance collapse. This highlights
that OOD generalization is not a byproduct of low-rank routing alone: it specifically requires alignment with meaningful directions identified by PCA. Without such alignment, routing vectors fail to
capture exemplar-derived cues, and the model effectively loses its cross-task transfer ability.


G.2 ICD SAMPLING

We vary strategies for constructing ICL prompts in PIDs extraction. Specifically, BALANCE/ _k_ denotes sampling _k_ ICDs per class in a balanced manner, while SIMILARITY selects ICDs based on
BERT embedding similarity to the query (Liu et al., 2021), with the total number of ICDs matched
to that of BALANCE/5. Table 12 shows that although SIMILARITY performs comparably in-domain,
it substantially degrades near- and far-OOD accuracy, indicating overfitting to query-local patterns
rather than capturing cross-domain invariances. This result highlights that exemplar diversity, rather
than local similarity, is most critical for robust PIDs extraction. Within the balanced scheme, _k_ = 5
achieves the best trade-off: fewer exemplars ( _k_ = 3) reduce coverage, while more ( _k_ = 7) add
redundancy without benefit.


G.3 ROUTING LAYERS

We investigate the effect of applying ICR at different depths within the model by evenly dividing it
into early, middle, and late segments. Table 13 shows that intervening at the late layers yields the
best overall performance. This outcome reflects a fundamental difference between ICR and prior
vector-based methods like I2CL. Vector-based approaches add interventions on the residual stream
whose effects tend to accumulate linearly, making adjustments from early or middle layers relatively
stable. In contrast, ICR directly modulates Q/K alignment via gated subspace coefficients. The resulting changes to attention distributions are nonlinear and softmax-amplified, which may propagate
through subsequent layers. When such routing is altered too early, small misalignments can cascade
and erode the low-level syntactic structure, causing all settings that involve early-layer intervention (including Early and All) to collapse. Focusing the intervention on late layers instead acts as a
high-level readout reweighting, preserving early representations while concentrating adaptation near
semantic integration and decision formation.


G.4 INFORMATION USAGE IN PID EXTRACTION

In PIDs extraction, we collect Q/K representation from the last token of ICL prompts. To explore
alternative ways of extracting Q/K representations Specifically, we test (i) mean pooling over the last


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


4 tokens, (ii) mean pooling over the last 8 tokens, and (iii) an attention-rollout–based pooling that
aggregates token-level Q/K using attention-flow weights computed across all layers. Conceptually,
the rollout variant constructs a cumulative attention map by multiplying layer-wise attention matrices
and uses the resulting contribution scores to weight each token’s Q/K before pooling. Experimental
results for the above variants are reported in Table 14.


Across all benchmarks including ID, near-OOD, and far-OOD, none of the alternative pooling
strategies outperform the last-token extraction. Performance degrades as the pooling window expands, and the attention-rollout variant yields the weakest results. This pattern suggests that incorporating a broader set of tokens introduces noise from heterogeneous token roles, diluting the
ICL-related signal that PIDs aim to isolate. A clear trend emerges that the more tokens included in
the pooling region, the more the essential alignment signal is blurred. The effectiveness of the lasttoken extraction is actually consistent with the functional role of this position in ICL. The final token
before answering is where the model synthesizes the full prefix (query and demonstrations) into a
single attention computation immediately before prediction. This makes it a coherent integration
point where the demonstration-induced structure is concentrated. Moreover, it is precisely the position at which ICR injects its attention-logits bias during inference. Extracting PIDs from the same
locus where the intervention is later applied provides a natural alignment between the raw attention
geometry and the added low-rank bias. These results indicate that the last-token Q/K captures the
most stable and transferable ICL-related structure, while broader pooling mixes in context that is not
directly relevant for the ICL computation. From an interpretability perspective, the fact that PIDs
extracted at the same position where we intervene work best shows that ICR is indeed leveraging
the attention structure that few-shot ICL forms at this locus.


G.5 TEXT ENCODER


To assess whether the choice of frozen text encoder affects routing quality and cross-domain generalization, we conduct an ablation in which we replace all-MiniLM-L6-v2 with a stronger
encoder, all-mpnet-base-v2. The latter has more layers and a higher embedding dimension
(768 vs. 384), providing richer semantic representations at the cost of slower encoding.


23


250


200


150


100


50


0
AGNews SST-2 TREC CSQA PIQA


Figure 5: Comparison of average per-sample inference time across five datasets for 5-shot, zeroshot, and ICR methods.


Table 12: Ablation on ICD sampling in ICL
bases construction. Scores are averaged over
ID, near-OOD, and far-OOD groups.


**Method** **ID** **Near OOD** **Far OOD**


SIMILARITY 67.3 55.0 49.4
BALANCE/3 62.1 52.4 48.8
BALANCE/5 **67.7** **57.3** **52.0**
BALANCE/7 66.8 56.9 50.2


Table 13: Ablation on routing layers. Scores
are averaged over ID, near-OOD, and far-OOD
groups.


**Layers** **ID** **Near OOD** **Far OOD**


Early 40.6 47.7 41.0
Middle 60.3 56.3 37.3
Late **67.7** **57.3** **52.0**
All 48.6 41.4 40.2


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


Table 14: Effect of different Q/K pooling strategies for PID extraction on ICR performance.


**In-Domain (ID)** **Near OOD** **Far OOD**
**Method** **Average**
AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE


Default (last token) 86.6 86.4 83.8 24.8 57.0 38.6 79.8 53.4 46.4 68.0 56.4 37.2 59.9
Last 4 tokens 84.6 83.2 83.8 25.6 55.0 35.6 73.8 46.6 39.3 61.0 53.4 31.2 56.1
Last 8 tokens 67.8 77.2 71.4 23.8 54.0 27.6 70.4 45.6 28.6 62.0 55.8 31.6 51.3
Attention rollout 67.2 78.8 67.0 22.2 52.6 26.8 72.8 44.4 32.2 62.0 52.4 32.8 50.9


Table 15: Effect of frozen text encoder choice on ICR performance.


**In-Domain (ID)** **Near OOD** **Far OOD**
**Encoder** **Average**
AG SST-2 TREC CSQA PIQA SST-5 MR MRPC CB COPA CREAK AI2SciE


miniLM 86.6 86.4 83.8 24.8 57.0 38.6 79.8 53.4 46.4 68.0 56.4 37.2 59.9
mpnet 86.8 87.0 88.2 25.0 58.6 37.0 86.4 56.6 48.2 64.0 57.0 38.0 61.1


∆ +0.2 +0.6 +4.4 +0.2 +1.6 -1.6 +6.6 +3.2 +1.8 -4.0 +0.6 +0.8 +1.2


The results, summarized in Table 15, show that all-mpnet-base-v2 yields slightly better performance in both ID and OOD settings, while the overall trend and relative performance of ICR
remain consistent. This indicates that (i) the router is able to effectively exploit the semantic features provided by the frozen encoder, and (ii) ICR’s generalization behavior is robust to the encoder
choice rather than being tied to a particular model. As expected, larger encoders offer marginally
better semantic retrieval at the cost of slower usage, so the choice involves a tradeoff between speed
and capacity.


H ”ICLNESS” TOKENS


For each dataset _d_ (including all ID, near-OOD, and far-OOD tasks), we run the model in both
zero-shot and ICR-augmented settings, compute the next-token log-probabilities, and obtain


∆log _p_ [(] _[d]_ [)] = log _p_ ICR _−_ log _p_ zs _._


Averaging over all examples in _d_ yields a token-level bias vector _b_ [(] _[d]_ [)] _∈_ R _[|V|]_, where each coordinate
indicates the systematic up- or down-weighting of a token by ICR on that dataset. We then aggregate
across datasets with the following statistics for each vocabulary token _v_ :


 - mean _v_ : mean ∆log _p_ across datasets


 - std _v_ : standard deviation across datasets


 - pos ~~r~~ ate _v_ : fraction of datasets with ∆log _p >_ 0


 - borda _v_ : Borda rank fusion across datasets


 - stability _v_ = mean _v/_ (std _v_ + _ϵ_ )


The final score is defined as


score _v_ = stability _v ·_ pos ~~r~~ ate _v ·_ log(1 + borda _v_ ) _,_


which rewards tokens that are (i) strongly upweighted on average, (ii) consistently positive across
datasets, and (iii) highly ranked across tasks. The top-50 tokens are listed in Table 16, with tokens
strongly related to in-context reasoning or structural semantics (”ICLness tokens”) highlighted in
red.


One might argue that because we explicitly require consistency across datasets, the resulting tokens
are trivially “cross-dataset”. However, cross-dataset consistency alone does not guarantee interpretability: many tokens that satisfy this criterion are function words (e.g., the, and) or generic
terms (e.g., _people_, _year_ ) that carry little connection to in-context reasoning. The notable observation is that the tokens emerging at the very top of the ranking are not such trivial items, but words
with structural and explanatory semantics (e.g., _illustrated_, _constitution_, _protected_ ). This indicates
that ICR does not merely enforce consistency on generic vocabulary, but systematically biases the
model toward dimensions plausibly linked to reasoning and explanation, aligning with our hypothesis about generalizable “ICLness.”


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


Table 16: Top-50 dataset-invariant “ICLness” tokens. A higher score indicates a more stable and
consistent positive bias across ID, near-OOD, and far-OOD datasets.


Rank Token Score Mean ∆log _p_ Std Pos. Rate Borda Norm


1 dep +28.79 +0.73 0.02 1.00 0.825
2 court +22.31 +0.75 0.02 1.00 0.828
3 _forme_ (French form) +21.92 +0.74 0.02 1.00 0.823
4 _illustrated_ +19.80 +0.21 0.00 1.00 0.538
5 _constitution_ +18.92 +0.48 0.01 1.00 0.704
6 _protected_ +18.35 +0.75 0.02 1.00 0.829
7 network +17.01 +0.76 0.03 1.00 0.836
8 thoughts +13.51 +0.47 0.02 1.00 0.695
9 colonial +13.49 +0.71 0.03 1.00 0.815
10 drie +13.41 +0.72 0.03 1.00 0.816
11 acres +12.50 +0.50 0.02 1.00 0.711
12 fro +12.22 +1.11 0.06 1.00 0.934
13 _protection_ +12.14 +0.83 0.04 1.00 0.861
14 reve +11.79 +0.68 0.03 1.00 0.797
15 leur +11.14 +0.70 0.04 1.00 0.809
16 _trouv_ (French find) +10.72 +0.77 0.04 1.00 0.839
17 _clause_ +10.09 +0.56 0.03 1.00 0.744
18 pipe +10.07 +1.12 0.07 1.00 0.923
19 _column_ +10.04 +0.52 0.03 1.00 0.723
20 Tot +9.21 +0.33 0.01 1.00 0.618
21 catt +9.17 +1.01 0.07 1.00 0.914
22 networks +9.16 +0.69 0.04 1.00 0.805
23 cyl +9.12 +1.28 0.09 1.00 0.958
24 duch +8.69 +0.87 0.06 1.00 0.868
25 bro +8.67 +0.32 0.02 1.00 0.609
26 _enumerate_ +8.54 +0.45 0.03 1.00 0.686
27 surv +8.34 +0.74 0.05 1.00 0.824
28 burst +8.27 +0.65 0.05 1.00 0.788
29 _connections_ +8.08 +0.85 0.07 1.00 0.868
30 _presente_ (French present) +8.08 +0.59 0.04 1.00 0.760
31 colors +7.99 +0.63 0.05 1.00 0.776
32 _signs_ +7.78 +0.41 0.03 1.00 0.662
33 _filter_ +7.55 +1.07 0.09 1.00 0.916
34 indust +7.37 +0.26 0.02 1.00 0.571
35 _returns_ +7.24 +0.88 0.08 1.00 0.879
36 _filters_ +7.23 +1.19 0.11 1.00 0.943
37 alles +7.22 +0.88 0.08 1.00 0.880
38 _zusammen_ (German jointly) +7.11 +0.74 0.06 1.00 0.820
39 neces +7.08 +0.94 0.08 1.00 0.886
40 tandis +7.07 +0.85 0.08 1.00 0.867
41 _separately_ +6.94 +1.14 0.11 1.00 0.946
42 bird +6.69 +0.42 0.03 1.00 0.670
43 blieb +6.57 +0.52 0.04 1.00 0.722
44 _comprend_ (French comprehend) +6.53 +0.93 0.09 1.00 0.888
45 _contrib_ +6.45 +0.60 0.05 1.00 0.765
46 _capture_ +6.41 +0.57 0.05 1.00 0.745
47 strict +6.40 +0.73 0.07 1.00 0.813
48 happy +6.28 +0.45 0.04 1.00 0.681
49 lange +6.21 +0.55 0.05 1.00 0.744
50 condem +6.18 +0.64 0.06 1.00 0.789


I LAYER IMPORTANCE


Figures 6a and 6b report the normalized layer-importance profiles across all in-domain (ID) and
out-of-domain (OOD) datasets, respectively. Each curve corresponds to one dataset, and the _x_ -axis
denotes the global transformer layer index. By comparing the two figures, several observations can
be made. First, both ID and OOD datasets consistently highlight a few dominant “hub” layers (e.g.,
around layers 23 and 26), indicating that ICR relies on these shared layers as primary routing points.
Notably, such hub layers are concentrated in the earlier–middle part of the intervened layers, while
later layers no longer exhibit clear global hubs, suggesting that they play a more task-specific role.


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


(a) (b)

Figure 6: Layer importance profiles. Curves show per-layer importance, computed from head gates
and routing coefficients.


Second, certain OOD datasets exhibit importance profiles that closely resemble those of particular
ID datasets, suggesting that ICR is able to adjust its routing behavior in a task-aware manner rather
than collapsing to a uniform pattern. Third, the importance peaks in OOD settings are sharper,
implying that under distribution shift the model leans more heavily on these hub layers as stable
anchors to preserve generalization.


J ADDITIONAL RELATED WORK


**Mechanisms** **of** **In-context** **Learning.** To better exploit ICL, considerable efforts have been devoted to understanding the mechanisms of ICL (Li et al., 2025c;b). ICL was initially regarded as an
ability that emerges as LLMs scale up in parameters and training data (Wei et al., 2022). Subsequent
work has sought to provide theoretical interpretations through two main perspectives. Garg et al.
(2022) modeled ICL as a form of gradient descent. Based on this, Von Oswald et al. (2023); Dai
et al. (2022) explained ICL via meta-optimization. Alternatively, Xie et al. (2021) framed ICL as
implicit Bayesian inference, suggesting that LLMs infer a shared latent concept across ICDs. Beyond modeling of model behavior, the connection between MHA and ICL has also been extensively
studied. Induction heads, which are attention heads that learn repeated patterns in the prompt and are
considered key contributors to ICL, were identified by Elhage et al. (2021) and empirically analyzed
by Olsson et al. (2022). Todd et al. (2023) further employed causal mediation analysis to identify
the heads that contribute most to ICL, denoted as FV heads. Yin & Steinhardt (2025) provided a
systematic synthesis of these findings. In contrast to these works, we develop a deeper theoretical
framework for ICL through attention routing, which can be effectively applied to enhance ICL performance. Whether ICL can truly generalize to OOD tasks is another central question. Yadlowsky
et al. (2023) find that ICL struggles to generalize to function classes unseen during training, such
as convex combinations or extreme variants of the pretraining functions. Wang et al. (2024b) further argue that ICL fails to generalize to new task instances even within a seen distribution, instead
exposing its limitation in handling unseen input–label distributions.


K MECHANISM DISCUSSION


One might argue that certain attention behaviors in transformers, such as induction heads, implement
global attention from demonstrations to the query, and that a method like ICR, which operates under
zero-shot inputs without explicit demonstrations, should be unable to reconstruct such patterns. Yet
empirically, ICR can match or even surpass vanilla few-shot ICL in several settings, which calls
for a more refined view of what demonstrations contribute. Our position is that, in the zero-shot
regime, the benefits commonly attributed to demonstrations in vanilla ICL can be reinterpreted as
local, intra-query attention routing. ICR explicitly operates in this regime that it does not reconstruct demo-to-query links, instead, it modulates attention logits within the query so that the model
allocates attention along task-useful paths. Concretely, the low-rank update ∆ _A_ encodes crosstask, reusable priors over intra-query routing, learned from pooled Q/K statistics across tasks, such
as: **(i)** **role-typing** (e.g., anchors such as question stems, label markers, options, premises vs. hypotheses); **(ii) long-range links between these roles** (e.g., question _↔_ option, premise _↔_ hypothesis,
number _↔_ unit); and **(iii) competition/sparsity priors** that sharpen relevant links and suppress distractors. Applying ∆ _A_ rotates the query–key geometry to reinstate these priors on a new query,


26


0.40


0.35


0.30


0.25


0.20


0.15


0.10


0.05


0.00


0.35


0.30


0.25


0.20


0.15


0.10


0.05


0.00


21 22 23 24 25 26 27 28 29 30 31
Layer (global index)


21 22 23 24 25 26 27 28 29 30 31
Layer (global index)


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


yielding attention maps that functionally resemble those induced by good demonstrations, without
requiring any demo content. This explains why zero-shot ICR can match or exceed vanilla few-shot
when demonstrations are noisy or misaligned. Importantly, ∆ _A_ transfers a routing prior rather than
learning new content at inference time. When a task truly relies on demo-specific content (beyond
routing), explicit few-shot prompting can be stronger, since such content cannot be reinstated by
intra-query attention routing alone. This explains why, on in-domain benchmarks, ICR may underperform vanilla few-shot ICL: some queries benefit directly from the information contained in the
demonstrations. By contrast, in OOD settings the main benefit of few-shot prompting often lies in
inducing a robust intra-query routing pattern, while its demo content can be misaligned or even misleading. By extracting and reusing this pattern, ICR attains few-shot–like gains without exposure to
OOD demo-content mismatch, yielding more comparable and stable performance under distribution
shift. This may provide a useful new perspective for future work on understanding the mechanisms
underlying in-context learning.


THE USE OF LARGE LANGUAGE MODELS (LLMS)


In preparing this submission, we used large language models (LLMs) solely for language refinement. Specifically, LLMs were employed to polish the writing style and improve readability, such
as rephrasing sentences and adjusting grammar.


27
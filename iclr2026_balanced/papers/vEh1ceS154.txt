# PARTITION GENERATIVE MODELING: MASKED MODELING WITHOUT MASKS


**Justin Deschenaux** [1] _[∗]_ **Lan Tran** [1] **Caglar Gulcehre** [1]


1EPFL, Lausanne, Switzerland


ABSTRACT


Masked generative models (MGMs) can generate tokens in parallel and in any
order, unlike autoregressive models (ARMs), which decode one token at a time,
left-to-right. However, MGMs process the full-length sequence at every sampling
step, including [MASK] tokens that carry no information. In contrast, ARMs process only the previously generated tokens. We introduce “Partition Generative
Models” (PGMs), which replace masking with partitioning. Tokens are split into
two groups that cannot attend to each other, and the model learns to predict each
group conditioned on the other, eliminating [MASK] tokens entirely. Because the
groups do not interact, PGMs can process only the clean tokens during sampling,
like ARMs, while retaining parallel, any-order generation, like MGMs. On OpenWebText, PGMs achieve 5–5 _._ 5 _×_ higher throughput than MDLM while producing
samples with lower Generative Perplexity. On ImageNet, PGMs reach comparable
FID to MaskGIT with a 7 _._ 5 _×_ throughput improvement. With twice as many steps,
the FID improves to 4.56 while remaining 3 _._ 9 _×_ faster than MGMs. Finally, PGMs
remain compatible with existing MGM samplers and distillation methods.


6


5


4


3


2


1


|GPT-2 Small (KV ca<br>MDLM (dim=768)|che)|Col3|Col4|
|---|---|---|---|
|~~PGM (6/6, dim=10~~<br>PGM (8/8, dim=76|~~ 24)~~<br> 8)|~~ 24)~~<br> 8)||
|||||
|||||
|||||
|||||


10 [3] 10 [4]

Throughput (tok/sec)


8


6


4


2


0


180


160


140


120


100


0


|5.54 5.55<br>5.35|7.93<br>7.37|
|---|---|
|4.56<br>~~4.76~~||
||~~4.12~~<br>|
||3.80|
|||
||1.05|
|Halton (FID<br>)<br>Throughput|Halton (FID<br>)<br>Throughput|


Figure 1: **(Left)** : On ImageNet, using the Halton sampler, PGM (ours), reaches similar FID as
MaskGIT with a 7 _._ 5 _×_ speedup. By sampling with twice as many steps, PGM reaches an FID of
4 _._ 56 while remaining 3 _._ 9 _×_ faster. **(Right)** : On OpenWebText, PGM achieves a better Generative
Perplexity with a 5 _._ 3 _×_ higher sampling throughput compared to MDLM (an MGM for text), at a
context length of 1024.


1 INTRODUCTION


Masked generative models (MGMs) offer two key advantages over autoregressive models (ARMs):
they can generate tokens in parallel and in any order, rather than one-by-one, left-to-right. These
properties have led to strong results across images (Chang et al., 2022), video (Yu et al., 2023;
Villegas et al., 2022), audio (Comunità et al., 2024), and language (Austin et al., 2023; Lou et al.,
2024; Sahoo et al., 2024; Shi et al., 2025; Campbell et al., 2024; Gat et al., 2024). However, MGMs


_∗_ Correspondence to justin.deschenaux@epfl.ch.


1


are slow at inference. Indeed, at every sampling step, they process the full-length sequence, including

[MASK] tokens that carry no information, whereas ARMs process only the previously generated
tokens. This limits the practicality of MGMs in large-scale and real-time settings, and is a crucial
disadvantage for test-time compute scaling (Snell et al., 2024; Wu et al., 2024) compared to ARMs.


Addressing the inference inefficiency of MGMs is not trivial because training and sampling must be
consistent. MGMs are trained with bidirectional architectures over the full sequence, so every hidden
representation depends on all _L_ positions, including masked ones. Furthermore, naively decoding
tokens block-by-block means feeding the model shorter sequences at inference, which differs from
training and leads to poor sample quality (Deschenaux & Gulcehre, 2024).


Prior work addresses the slow inference of MGMs from different angles. Decoding more tokens per
step increases throughput, but degrades sample quality. Distillation (Deschenaux & Gulcehre, 2025;
Zhu et al., 2025; Sahoo et al., 2025a) reduces the number of sampling steps, but each step remains
equally expensive, and distillation can affect the sample diversity (Gandikota & Bau, 2025). Block
Diffusion (Arriola et al., 2025) enables partial KV-caching by generating tokens block-by-block, but
sacrifices the any-order generation capability. None of these approaches make _individual sampling_
_steps cheaper_ while preserving the full flexibility of MGMs.


We introduce _Partition Generative Models_ (PGMs), which replace masking with partitioning. Tokens
are split into two disjoint groups, and a group-wise attention mechanism ensures that no information
flows between them. The model learns to predict each group conditioned on the other, eliminating

[MASK] tokens entirely. Because the two groups do not interact, PGMs process only the clean tokens
during sampling, just like ARMs, while retaining the ability to generate tokens in parallel and in
any order, like MGMs. We propose the _Partition Transformer_, a dedicated architecture that prevents
information flow between groups.


**Contributions** (1) We introduce PGMs and the Partition Transformer, a new architecture that
enables MGM-style parallel, any-order generation without [MASK] tokens. PGMs are compatible
with existing MGM samplers (Besnier et al., 2025) and distillation methods (Deschenaux & Gulcehre,
2025), making them a drop-in replacement. (2) On OpenWebText (Gokaslan & Cohen, 2019), PGMs
generate samples with lower Generative Perplexity than MDLM (Sahoo et al., 2024) and reach
similar downstream task performance, before and after distillation, while **achieving 5–5** _._ **5** _×_ **higher**
**throughput** . On ImageNet, PGMs reach comparable FID to MaskGIT (Chang et al., 2022) with a
**7** _._ **5** _×_ **throughput improvement** . (3) We show that PGM are trained with denser supervision than
MGM. Since each group predicts the other, a single sequence yields two complementary training
signals. This reduces gradient variance and yields a **1.95 reduction in validation perplexity** on
LM1B (Chelba et al., 2014) compared to MDLM with the same number of layers.


2 BACKGROUND


2.1 SEQUENCE MODELING


We consider the task of generating sequences **x** = ( _x_ 1 _, . . ., xL_ ) of length _L_ over a vocabulary
_V_ = _{_ 0 _, . . ., N_ _−_ 1 _}_ . The training dataset _D_ contains finitely many sequences drawn from an
unknown data distribution _p_ data over _V_ _[L]_ . **Autoregressive models** (ARMs) factorize the distribution
as _pθ_ ( **x** ) = [�] _i_ _[L]_ =1 _[p][θ]_ [(] **[x]** _[i]_ _[|]_ **[ x]** _[<i]_ [)][, where] **[ x]** _[<i]_ [denotes the prefix before position] _[ i]_ [.] [Tokens are sampled]
sequentially, and because each conditional only depends on the prefix **x** _<i_, ARMs process only the
previously generated tokens instead of the whole sequence.


2.2 MASKED GENERATIVE MODELS


MGMs augment the vocabulary with a special [MASK] token, absent from the training data. Given
**x** _∈D_, let **z** _t_ denote a corrupted sequence where the token **z** _[ℓ]_ _t_ [at position] _[ ℓ]_ [is [][MASK][] with probability]
_pt_, or the clean value **x** _[ℓ]_ otherwise. The masking probability _pt_ is an increasing function of _t ∈_ [0 _,_ 1]
with _p_ 0 = 0 and _p_ 1 = 1. MGMs train a denoiser **x** _θ_ : _V_ _[L]_ _→_ R _[L][×][N]_ whose sampling distribution
is modeled as factorized marginals: [�] _ℓ_ : **z** _[ℓ]_ _t_ [=[][MASK][]] _[p]_ _θ_ _[ℓ]_ [(] _[.][ |]_ **[ z]** _[t]_ [)][.] [Because MGMs sample independently]

from each _p_ _[ℓ]_ _θ_ [(] _[.][ |]_ **[ z]** _[t]_ [)][, they cannot model arbitrary joint distributions, unlike ARMs.] [However, they]


2


|PGM Decoder|Col2|Col3|
|---|---|---|
||||
||||


|GroupSwap|Col2|
|---|---|
|||
|PGM Encoder|PGM Encoder|
|**No masked tokens**<br>**to process**⚡ <br>**Group 2**<br>**A**<br>**D**<br>**F**|**No masked tokens**<br>**to process**⚡ <br>**Group 2**<br>**A**<br>**D**<br>**F**|


Figure 2: **Masked Generative Modeling (MGM) vs.** **Partition Generative Modeling (PGM). Left:**
MGMs learn at masked positions only, and must process [MASK] tokens at inference. **Right:** PGMs
partition tokens in two groups, learn from _all_ positions and process clean tokens only during sampling.
PGMs achieve >5.3x higher throughput on OpenWebText (context length 1024, 128 sampling steps).


can decode tokens in parallel. The training objective for MGMs is:


_L_ MGM := E **x** _∼D,t∼U_ [0 _,_ 1] [ _w_ ( _t_ )CE( **x** _θ_ ( **z** _t_ ; _t_ ) _,_ **x** )] _,_ (1)


where _w_ : [0 _,_ 1] _→_ R _≥_ 0 is a weighting function and CE(ˆ **x** _,_ **x** ) denotes the cross-entropy loss over
masked positions. To generate samples, MGMs start from a fully masked sequence and iteratively
unmask subsets of positions over multiple evaluations of **x** _θ_, selecting positions at random or based
on confidence scores. We now describe two instantiations used in this work.


**MDLM** Masked Diffusion Language Models (MDLM; Sahoo et al. (2024); Ou et al. (2025); Shi
et al. (2025)) are MGMs for language modeling. Analogously to continuous diffusion (Sohl-Dickstein
et al., 2015; Song & Ermon, 2020; Ho et al., 2020; Kingma et al., 2023), MDLM defines a forward
process that corrupts clean data and a generative process that recovers samples from noise. The
forward process is:
_qt_ ( _.|_ **x** ) := Cat( _._ ; _αt_ **x** + (1 _−_ _αt_ ) _**π**_ ) _,_ (2)
where **x** is the one-hot representation, _**π**_ = **m** is the one-hot encoding of [MASK], and _αt_ is a strictly
decreasing noise schedule with _α_ 0 = 1 _, α_ 1 = 0. (2) is applied independently at every position. The
posterior distribution is:


_ps|t_ ( _.|_ **z** _t,_ **x** ) =


- Cat( _._ ; **z** _t_ ) _,_ **z** _t_ = **m** _,_


Cat - _._ ; [(1] _[−][α][s]_ [)] **[m]** [+(] _[α][s][−][α][t]_ [)] **[x]**


(1 **[m]** _−_ [+(] _α_ _[α]_ _t_ ) _[s][−][α][t]_ [)] **[x]** - _,_ **z** _t_ = **m** _._ (3)


To generate samples, we fix a decreasing sequence of times 1 = _τT_ _> · · · > τ_ 0 = 0, set **z** _τT_ to the
all [MASK] tokens sequence, and iteratively sample from


**z** _τi−_ 1 _∼_ _pθ,τi−_ 1 _|τi_ ( _. |_ **z** _τi_ ) = _pτi−_ 1 _|τi_ ( _.|_ **z** _t,_ **x** _θ_ ( **z** _τi, τi_ )) _._ (4)

MDLM optimizes a variational bound on log-likelihood that reduces to (1) with _w_ ( _t_ ) = 1 _−αα_ _[′]_ _t_ _t_ [.] [Only]
masked positions contribute to the loss.


**MaskGIT** MaskGIT (Chang et al., 2022) is an MGM that operates in the latent space of a pretrained VQGAN (Esser et al., 2021) tokenizer. MaskGIT proposes tokens **x** ˜ _[ℓ]_ at masked position and
uses the predicted likelihood of the sampled token **x** ˜ _[ℓ]_ as a confidence score: _c_ _[ℓ]_ = **x** _[ℓ]_ _θ_ [(] **[z]** _[t]_ [;] _[ t]_ [)] **[x]** [˜] _[ℓ]_ [.] [A]
predefined schedule determines the number of positions to unmask, and the most confident positions
are kept. Tokens generated in earlier steps are kept unchanged. This differs from MDLM, which
denoises at random positions. Besnier et al. (2025) observed that confidence-based sampling tends
to decode spatially clustered tokens, since the denoiser is most confident near previously generated
positions. Because MGMs sample independently from a product of marginals [�] _ℓ∈S_ _[p][θ]_ [(] **[x]** _[ℓ]_ _[|]_ **[z]** _[τ]_ [)]

rather than the joint token distribution, decoding nearby tokens increases the risk of generating


3


inconsistent samples. By sampling according to a low-discrepancy sequence (Halton, 1964), Besnier
et al. (2025) enforce a more uniform coverage of the space, which improves the FID and IS compared
to confidence sampling.


**Classifier-Free Guidance** Let _pθ_ ( **x** _| c_ ) denote a class-conditional distribution learned by an MGM,
where _c ∈{_ 0 _, . . ., C −_ 1 _}_ is a class label and _pθ_ ( **x** _|_ ø) denotes the class-unconditional distribution.
Let _ω_ _≥_ 0 control the guidance strength. Classifier-Free Guidance (CFG; Ho & Salimans (2022);
Chang et al. (2022)) steers generation toward class _c_ by replacing log _pθ_ ( **x** _| c_ ) during sampling with


log ˜ _pθ_ ( **x** _| c_ ) = (1 + _ω_ ) log _pθ_ ( **x** _| c_ ) _−_ _ω_ log _pθ_ ( **x** _|_ ø) _,_ (5)


**Self-Distillation Through Time** Self-Distillation Through Time (SDTT; Deschenaux & Gulcehre
(2025)) accelerates sampling from MGMs by distilling a teacher trained for denoising with many
steps into a few-steps student. Let _p_ [(] _θ_ _[m]_ [)] denote the distribution of samples generated with _m_ steps
using a denoiser **x** _θ_, and let _p_ [(] _ν_ _[k]_ [)] denote the distribution when using _k_ _<_ _m_ steps with a student
denoiser **x** _ν_ . SDTT trains **x** _ν_ with the following objective:

min E **z** 0 _∼D,_ **z** _t∼qt_ ( **z** _t|_ **z** 0)        - _δ_ ( **x** _ν_ ( **z** _t, t_ ) _||_ **x** ˜ [teacher] _θ_ ( **z** _t, t,_ _[m]_ _/k_ ))� _,_ (6)
_ν_

where _δ_ is a divergence measure (e.g., KLD) and **x** ˜ [teacher] _θ_ ( **z** _t, t,_ _[m]_ _/k_ ) are the distillation targets. These
targets are constructed using _[m]_ _/k_ sampling steps with the teacher, starting from **z** _t_ and collecting the
predicted log-probabilities for each token at the step where a token was denoised. After training,
one step of the student should match _[m]_ _/k_ teacher steps. SDTT can be applied iteratively by reusing
the student as teacher in each round (Salimans & Ho, 2022), progressively halving the number of
required steps. Empirically, distilling 2 steps per round is most effective.


3 PARTITION GENERATIVE MODELING


At each sampling step, MGMs process the entire sequence, including many [MASK] tokens that
will not be decoded yet. In contrast, ARMs process clean tokens only, but generate one token at a
time. _Partition Generative Models_ (PGMs) combine the strengths of both approaches, by generating
multiple tokens in parallel, like MGMs, while processing only the clean tokens, like ARMs. _PGMs_
_are a direct extension of the MGM paradigm_ . As a result, sampling algorithms, guidance mechanisms,
and distillation methods developed for MGMs apply directly to PGMs. Only the neural network
architecture must be adapted (Sec. 4).


3.1 TRAINING


**From Masking to Partitioning** Instead of replacing tokens with [MASK], PGMs partition the
sequence into two complementary groups. Given **x** _∈D_ and _t ∼U_ [0 _,_ 1], each token is assigned to
group 1 with probability _pt_ = 1 _−_ _αt_, and to group 0 otherwise. Let **g** _∈{_ 0 _,_ 1 _}_ _[L]_ denote the group
membership vector. We propose a Transformer variant in Sec. 4 that ensures that information cannot
flow between groups. Predictions at positions in group 0 depend only on tokens in group 1, and
vice-versa (Figure 2). This is consistent with MGMs, where masked tokens are predicted from clean
ones, except that PGMs learn from both groups.


**Connection to the MDLM Variational Bound** In MDLM, the forward process (2) masks each
position independently, so at time _t_ an expected fraction _αt_ of tokens remain clean. PGMs assign an
expected fraction _αt_ of tokens to group 0, which plays the same role as the clean tokens in MDLM.
By treating group 0 as clean and group 1 as masked, the MDLM loss weight _w_ ( _t_ ) = 1 _−αα_ _[′]_ _t_ _t_ [is applied]
to tokens in group 1. By symmetry, tokens in group 0 are weighted by _w_ (1 _−_ _t_ ). Therefore, in a
single forward pass, PGMs evaluate the MDLM training objective at two complementary masking
rates. Hence, the training objective is


_L_ PGM := E **x** _∼D,t∼U_ [0 _,_ 1]       - _w_ [PGM] ( **g** _, t_ )CE( **x** _θ_ ( **x** ; **g** ; _t_ ) _,_ **x** )� _,_ (7)


where

_w_ [PGM] ( **g** _, t_ ) _i_ =         - _w_ ( _t_ ) if **g** _i_ = 0 (8)
_w_ (1 _−_ _t_ ) if **g** _i_ = 1 _._


4


Figure 3: **Partition Transformer.** RoPE is applied before every attention layer (omitted for clarity).
**(C)** Encoder: group-wise self-attention (no cross-group flow). **(B)** GroupSwap: cross-attention that
routes each position’s representation to the opposite group. **(A)** Decoder: group-wise cross-attention
to the encoder output (no self-attention).


**Variance Reduction** Unlike MGMs, which compute the loss over masked positions only (Figure 2),
PGMs compute the loss at every position, yielding two gradient contributions per training sample. By
training on two complementary copies, PGMs reduce the variance. Empirically, training diffusion
models with lower variance improves the validation likelihood (Kingma et al., 2023; Sahoo et al.,
2024). We study the variance reduction in Sec. 5.3.


3.2 SAMPLING


During inference, PGMs process clean tokens only, like ARMs, yet decode tokens in parallel at
arbitrary positions, like MGMs (Figure 2). Let _Cτ_ _⊆{_ 1 _, . . ., L}_ denote the clean token indices at
step _τ_ _∈{_ 1 _, . . ., T_ _}_, with _nτ_ = _|Cτ_ _|_ and _mτ_ = _L −_ _nτ_ . At each step, we select _kτ_ masked positions,
sample from _pθ_ ( _·_ _|_ **x** _Cτ_ ), and add the decoded tokens to _Cτ_ +1. For text, we find that using a fixed
schedule with _kτ_ = _k_ tokens per step (Algo. 2) improves sample quality and throughput compared
to the MDLM posterior that decodes each position with probability _[α]_ 1 _[s]_ _−_ _[−]_ _α_ _[α]_ _t_ _[t]_ [and requires padding for]

batched generation (Algo. 3; Suppl. E.2). For images, we experiment with both the confidence and
Halton samplers (Suppl. B, Besnier et al. (2025)). The Halton sampler performs better empirically,
so we report confidence-based sampler results in Suppl. D.3.


4 THE PARTITION TRANSFORMER


PGMs require a careful architectural design. In particular, since our goal is to process a single group
only during inference, tokens across groups should not attend to each other. As shown in Figure 2
and Figure 3, we build the _Partition Transformer_ such that the predictions for tokens in group 0 are
based on tokens in group 1 only. The Partition Transformer implements a mechanism to _swap_ the
physical location of information across groups. During training, this allows using the input sequence
**x** as target. During sampling, it moves information about the input tokens from the clean positions to
the positions to predict. Our architecture consists of an encoder, a _GroupSwap_ layer, and a decoder,
which we describe below.


**Encoder** The encoder is made of partition-wise self-attention blocks, which are similar to standard
bidirectional transformer blocks except that tokens in separate groups do not attend to each other.


5


Table 1: Validation perplexity, sampling latency, and throughput (TP) on LM1B and OpenWebText.
_PGM k / m_ uses _k_ encoder and _m_ decoder layers. The best PGM per dataset is highlighted. Latency
and TP are measured at batch size 32. _[†]_ Trained with a 2 _×_ larger batch size (Sec. 5.3). See Table 5
for architecture ablations.


**Model** **#Params** **Val.** **PPL** _↓_ **Latency (sec)** _↓_ **TP (tok/sec)** _↑_


_LM1B (ctx len._ _128)_
MDLM 170M 27.67 3.78 1’081.57
MDLM _[†]_ (Compl. masking) 170M **25.72** 3.78 1’081.57

PGM 6 / 6 171M 26.80 **2.12** **1’930.93**


_OpenWebText (ctx len._ _1024)_
MDLM 170M 23.07 31.41 1’043.22
MDLM _[†]_ (Compl. masking) 170M 22.98 31.41 1’043.22
PGM 8 / 8 203M 22.61 **5.86** **5’585.57**

PGM 6 / 6 (dim. 1024) 268M **21.43** 5.93 5’518.09


**Decoder** The decoder uses cross-attention layers, whose keys and values are computed based on the
output of the encoder. In contrast, the queries are computed using either the output of the GroupSwap
layer (for the first block of the decoder) or the output of the previous decoder block (see Sec. 4.1).
Importantly, _there is no self-attention layer in the decoder_, which allows efficient generation, as we
can compute predictions solely at the positions that we will decode.


4.1 THE GROUPSWAP LAYER


In the encoder, information remains localized. If a token belongs to group 0, its hidden representation
only depends on tokens in group 0. For prediction, however, we require the opposite: representations
at positions in group 0 must depend exclusively on group 1, and vice versa. To enforce this, we
introduce the _GroupSwap_ layer (Figure 3B), which exchanges information between groups. The
GroupSwap layer is implemented using cross-attention, and to prevent information leakage, the
queries used in cross-attention cannot depend on tokens in the other group. We describe two ways of
initializing queries.


**Data-Independent** **Queries** Let **u** _∈_ R _[H]_ be a learnable vector. To initialize the queries, we
replicate **u** across the sequence length, add fixed positional encodings, and apply layer normalization
followed by a linear projection. The query matrix _V_ _∈_ R _[L][×][H]_ (where _Vi_ ; _·_ is the _i_ -th row) satisfies


_Vi_ ; _·_ = _W_ �LN                - _u_ + pos _i_ ; _·_                - + _b_                - _,_ (9)


where _W_ _∈_ R _[H][×][H]_, _b ∈_ R _[H]_ are learnable parameters and LN denotes layer normalization (Ba et al.,
2016). We use sinusoidal positional encoding (Vaswani et al., 2023):


          - cos           - _i_           - if _j_ _<_ _[H]_ _/_ 2
pos _i,j_ = sin             - 10000 [2] _i_ _[j/H]_             - otherwise (10)
10000 [2] _[j/H][−]_ [1]


**Data-Dependent Queries** Let _X_ _∈_ R _[L][×][H]_ be the encoder output. We first perform a group-wise
aggregation over the sequence length (e.g., logsumexp or mean) to obtain vectors _Y_ 0 _, Y_ 1 _∈_ R _[H]_,
the aggregate representations of groups 0 and 1. The queries _V_ _[′]_ are then

_Vi_ _[′]_ ; _·_ [=] _[ V][i]_ [;] _[·]_ [+]                - _Y_ 1 _,_ if _gi_ = 0 (11)
_Y_ 0 otherwise _._


5 EXPERIMENTS


We compare PGM with MDLM (Sahoo et al., 2024) on standard language modeling datasets, training
on LM1B (Chelba et al., 2014) and OpenWebText (OWT; Gokaslan & Cohen (2019)) in Sec. 5.1.


6


We evaluate them using the validation perplexity and downstream task accuracy before and after
distillation with SDTT (Deschenaux & Gulcehre, 2025). We compare PGM with MaskGIT (Chang
et al., 2022) on VQGAN-quantized (Esser et al., 2021) ImageNet256 (Deng et al., 2009) (Sec. 5.2).
As described in Sec. 3, by predicting each group from the other, PGMs implement a mechanism
akin to training on two complementary masked sequences per batch, while also introducing a new
architecture (Sec. 4). The effect of complementary masking is studied in isolation in Sec. 5.3. Our
experiments show that, for both language and image modeling, and after distillation, PGMs are
competitive with MDLM and MaskGIT, while **providing a 5–5** _._ **5** _×_ **throughput improvement for**
**text and a 7** _._ **5** _×_ **improvement for images** . Find more experimental details in Suppl. C.


5.1 LANGUAGE MODELING


**Experimental** **settings** We closely follow the settings of Sahoo et al. (2024). MDLM uses a
modified Diffusion Transformer (Peebles & Xie, 2023; Lou et al., 2024) with RoPE (Su et al., 2023),
with 12 layers and an embedding dimension of 768, without time conditioning. We train with a global
batch size of 512 for 1M steps, dropout of 0.1, and the Adam optimizer with learning rate 3 _×_ 10 _[−]_ [4]
and no weight decay. We maintain an Exponential Moving Average (EMA) of the weights with
decay 0.9999. For PGM, we use the Partition Transformer architecture (Sec. 4) with 12 or 16 layers,
embedding dimensions of 768 or 1024, and varying numbers of encoder and decoder layers. On
LM1B, all models use a context length of 128, with shorter documents padded and tokenized using
the bert-base-uncased (Devlin et al., 2019) tokenizer. On OWT, we use a context length of
1024 with sentence packing (Raffel et al., 2023) with the GPT-2 tokenizer and insert an [EOS] token
between documents. Since the dataset lacks an official validation split, the last 100k documents are
reserved for validation. To evaluate the sample quality, we use the Generative Perplexity (Gen. PPL),
computed using GPT-2 Large (Radford et al., 2019), following Sahoo et al. (2024). We cast the logits
in float64 prior to sampling, following Zheng et al. (2025).


**Likelihood Evaluation** After 1M steps, PGMs with as many layers as MDLM achieve a **validation**
**perplexity of 1.95 lower than MDLM on LM1B** (Table 1). Table 5 (left) shows that balanced models
with equal numbers of encoder and decoder layers outperform imbalanced variants. Interestingly,
data-independent queries perform comparably to data-dependent queries, so we use the simpler, dataindependent version in all subsequent experiments. On OpenWebText, PGMs with the same number
of layers and embedding dimension as MDLM slightly underperform (Table 5, right). Increasing the
number of encoder and decoder layers by two, or increasing the embedding dimension to 1024, allows
PGMs to surpass MDLM in validation perplexity, while **achieving at least 5** _×_ **higher sampling**
**throughput** . This improved efficiency makes PGMs particularly attractive for scaling test-time
computation (Madaan et al., 2023; Yao et al., 2023; Snell et al., 2024; Wu et al., 2024; Chen et al.,
2024; Brown et al., 2024; Goyal et al., 2024).


**Downstream Evaluation** Following Deschenaux & Gulcehre (2024); Nie et al. (2025), we evaluate
MDLM and PGMs trained on OpenWebText using the lm-eval-harness suite (Gao et al.,
2024). As shown in Table 2, PGMs slightly outperform MDLM on six out of eight tasks, although
the overall accuracy across models is similar. This suggests that **PGM achieves faster inference**
**without sacrificing downstream performance** . Since lm-eval-harness is originally designed
for ARMs, we must adapt it for MGMs. Fortunately, both MDLM and PGM can compute a variational
bound on the likelihood, which is used in place of the true likelihood to select the most probable
answer in multiple-choice tasks. Additional details and tasks are provided in Suppl. D.5.


**Distillation of PGMs** After likelihood training, PGMs achieve 5 _−_ 5 _._ 5 _×_ higher throughput than
MDLM. To further accelerate sampling, we apply Self-Distillation Through Time (SDTT; Deschenaux
& Gulcehre (2025)). To remain as faithful as possible to the implementation of Deschenaux & Gulcehre (2025), we apply the distillation loss to a single group while treating the other as [MASK] tokens.
This shows that PGMs are compatible with distillation methods designed for MGMs. We leave the
development of new distillation strategies for PGMs to future work. Hence, the setup naturally favors
MDLM. Figure 4 (right) and Table 6 compare the Gen. PPL, unigram entropy, and sampling speed
of PGM and MDLM. After five rounds of distillation, and with standard ancestral sampling, PGMs
achieve higher Generative Perplexity and entropy than MDLM. With nucleus sampling ( _p_ = 0 _._ 9)
(Holtzman et al., 2020), PGMs produce samples with comparable perplexity and entropy. Due to


7


the overhead of nucleus sampling, the speed advantage of PGMs decreases from at least 5 _×_ to
approximately 4 _._ 6 _×_ faster than MDLM for the same number of steps (Fig. 4). Generative perplexity
alone does not fully capture language model performance, hence we also evaluate distilled models on
downstream tasks. As shown in Table 2, distillation slightly shifts accuracy across tasks, but overall
performance remains similar. **PGMs still achieve slightly higher accuracy than MDLM on most**
**tasks after distillation** .


5.2 IMAGE MODELING


**Experimental** **Settings** We train MaskGIT
(Chang et al., 2022) and PGM on ImageNet256.
Images are cropped to a centered square along
the longer side and then rescaled to 256 _×_ 256.
We use the MaskGIT implementation of Besnier
et al. (2025), including their pre-trained VQGAN tokenizer. We train for 500k steps with
a batch size of 256 using AdamW (weight decay 0.03, learning rate 1e-4, cosine schedule
with 2500 warmup steps). We use a dropout of
0.1 in the Transformer. All models are classconditional, with a class-label dropout of 0.1 to
enable classifier-free guidance (CFG) at sampling time. As Besnier et al. (2025), we train
with one register (Darcet et al., 2024) for the
MaskGIT baseline, and two (one per group) for
PGM, so that we can use one register during
sampling. We sample with the confidence and
Halton samplers.


90


60


50


40


35


|P<br>P|MDLM<br>GM<br>GM+nucleus|(p=0.9)<br>(<br>.5)|5.5)|(5|.5)|
|---|---|---|---|---|---|
|P<br>P|(5.5)<br>(5|(5.5)<br>(5|(5.5)<br>(5|(5.5)<br>(5|(5.5)<br>(5|
|(5.4)|~~(5.4)~~<br>(5.4)<br>|||~~(5.5)~~||
|(5.4)<br>(5.4)<br>(5.4)<br>|(5.4)<br><br>(5.<br>|(5.<br>4)|4)|||
|(5.3)<br>(5.4)|(5.4)<br>(5.4)|||||


0 5000 10000 15000 20000
Throughput (tokens/sec.)


Figure 4: After distillation, PGM (6 / 6, dim. 1024)
with nucleus sampling remains significantly faster
than MDLM, at matching entropy and Gen. PPL.


**Results** In Figure 1 (left), we compare the Fréchet Inception Distance (FID; Heusel et al. (2018))
of samples from MaskGIT (Chang et al., 2022) and PGM, using the Halton sampler and classifierfree guidance with the guidance weight _w_ _∈{_ 0 _,_ 1 _, . . .,_ 6 _}_ that yields the lowest FID. PGM 12/12
achieves a 7 _._ 5 _×_ higher throughput with only a slight FID degradation (5.54 vs. 5.35). **Increasing**
**the sampling steps to 64 further improves the FID to 4.56, while remaining 3** _._ **9** _×_ **faster than**
**MaskGIT** . See Suppl. D.3 for full results across guidance strengths.


5.3 ISOLATING THE EFFECT OF COMPLEMENTARY MASKING


**Experimental Setup** To disentangle the contributions of PGM, we isolate the effect of complementary masking (Sec. 3) by training a standard bidirectional Transformer with double the batch size.
Each input sequence is turned into two complementary masked copies: if the token at position _ℓ_ is
masked in one copy, it remains unmasked in the other. This setup provides an upper bound on the
potential gains, as it directly measures the benefit of complementary masks during training.


**Results** Table 1 shows that complementary masking improves the validation perplexity on LM1B
and OWT, though with smaller gains on OWT. On both datasets, a gap remains between PGM and
MDLM with complementary masking. This suggests that the current neural network architecture can
be improved further. Because of the smaller improvement on OWT, we must increase the parameter
count to surpass MDLM. Nonetheless, recall that despite having more parameters, PGMs remain at
least 5 _×_ faster than MDLM during sampling. In Suppl. D.1, we present preliminary experiments
exploring why complementary masking improves performance on LM1B but not on OpenWebText.


6 RELATED WORK


**Discrete** **Diffusion** Although autoregressive models currently dominate text generation, recent
advances in discrete diffusion (Austin et al., 2023; Lou et al., 2024; Shi et al., 2025; Sahoo et al.,
2024; von Rütte et al., 2025; Schiff et al., 2025; Haxholli et al., 2025; Sahoo et al., 2025a) and
discrete flow matching (Campbell et al., 2024; Gat et al., 2024) have demonstrated that MGMs can


8


Table 2: **Accuracy on downstream tasks** (Gao et al., 2024). HS: HellaSwag, OQA: OpenBook
QA. Arc: Arc-easy. We select the tasks following Nie et al. (2025). We see that distillation slightly
changes the downstream tasks performance, but that PGMs continue to outperform MDLM on most
tasks. The best performance is **bolded**, while the second best is underlined.


**LAMBADA** **Arc** **BoolQ** **HS** **OQA** **PIQA** **RACE** **SIQA**


_Before Distillation_
MDLM 38.52 37.88 49.42 31.36 **28.60** 58.27 **28.04** 38.84
PGM 8 / 8 **46.98** **40.40** **53.49** 33.20 26.60 58.92 26.89 39.97
PGM 6 / 6 (1024) 41.39 39.98 49.82 **34.27** 25.40 **59.19** 27.37 **40.28**


_After Distillation (SDTT)_
MDLM 41.34 33.80 48.59 30.75 **28.80** 57.73 27.94 38.79
PGM 8 / 8 **47.22** **37.42** **51.50** 31.62 25.80 59.03 **30.62** **39.61**
PGM 6 / 6 (1024) 44.48 36.70 49.36 **32.55** 25.00 **59.85** 27.37 39.25


approach AR models in generation quality. We propose a simple framework that allows sampling
without processing any [MASK] tokens, but remains compatible with methods developed for MGMs
(such as distillation and alternative samplers).


**Variable Length Masked Diffusion** Block Diffusion (BD; Arriola et al. (2025)) enables partial
KV-caching (Pope et al., 2022) by generating tokens block-by-block using discrete diffusion. BD
improves throughput but sacrifices the any-order generation capabilities of MGMs. We do not
experiment with integrating causal attention to enable KV-caching. However, Ma et al. (2025); Wu
et al. (2025) show that KV caching can be integrated post-hoc into MGMs despite being trained
without causal attention. FlexMDM (Kim et al., 2025) and Edit Flows (Havasi et al., 2025) enable
variable-length generation via insertion, deletion, and replacement. While promising, these depart
from the simplicity of MGM and PGM. Finally, Eso-LMs (Sahoo et al., 2025b) train with a hybrid
AR-MGM objective. Sahoo et al. (2025b) first sample a draft in MGM mode, then fill in the remaining
tokens autoregressively. During training, Eso-LMs must choose the fraction of examples to process in
AR versus MGM mode, which adds a hyperparameter to tune. Eso-LMs use [MASK] tokens during
training in MGM mode, whereas PGMs do not because of the Partition Transformer.


**Non-Autoregressive Language Models** Any-order and any-subset autoregressive models (Yang
et al., 2020; Pannatier et al., 2024; Shih et al., 2022; Guo & Ermon, 2025) factorize the sequence
distribution autoregressively over permutations of tokens. Hence, these models use causal attention
and generate tokens one by one. In contrast, MGMs use bidirectional attention and generate multiple
tokens in parallel, which is the setting PGM builds on.


7 CONCLUSION


We introduce Partition Generative Modeling (PGM), a novel approach to masked generative modeling
that eliminates [MASK] tokens entirely. PGM achieves significant improvements in inference speed
on both text and images, with minimal effect on quality. The significant improvements suggest
that PGM might be suited for domains that benefit from test-time scaling, such as coding and
reasoning. We show that PGMs can be distilled for further acceleration. Future work should explore
optimizations to the PGM architecture, investigate distillation techniques specifically designed for
PGMs, and extend the approach to multimodal settings. In summary, PGM offers an alternative
to masked generative models, with particular advantages for applications where inference speed is
critical.


9


8 ACKNOWLEDGEMENTS


This work has received funding from the Swiss State Secretariat for Education, Research and
Innovation (SERI). We are grateful to Razvan Pascanu, Sungjin Ahn, Jaesik Yoon, Mingyu Jo,
Subham Sahoo and Zhihan Yang for insightful discussions and suggestions. We acknowledge the
SCITAS team at EPFL for providing access to their cluster, and the Swiss National Supercomputing
Centre for the Alps platform. We are grateful to Karin Gétaz for her administrative assistance.


REFERENCES


Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025. [URL https://arxiv.org/abs/2503.09573.](https://arxiv.org/abs/2503.09573)


Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured
denoising diffusion models in discrete state-spaces, 2023. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2107.03006)
[2107.03006.](https://arxiv.org/abs/2107.03006)


Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer normalization, 2016. URL

[https://arxiv.org/abs/1607.06450.](https://arxiv.org/abs/1607.06450)


Victor Besnier, Mickael Chen, David Hurych, Eduardo Valle, and Matthieu Cord. Halton scheduler
for masked generative image transformer, 2025. URL [https://arxiv.org/abs/2503.](https://arxiv.org/abs/2503.17076)
[17076.](https://arxiv.org/abs/2503.17076)


Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and
Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling,
2024. [URL https://arxiv.org/abs/2407.21787.](https://arxiv.org/abs/2407.21787)


Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative
flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design,
2024. [URL https://arxiv.org/abs/2402.04997.](https://arxiv.org/abs/2402.04997)


Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked generative
image transformer, 2022. [URL https://arxiv.org/abs/2202.04200.](https://arxiv.org/abs/2202.04200)


Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony
Robinson. One billion word benchmark for measuring progress in statistical language modeling,
2014. [URL https://arxiv.org/abs/1312.3005.](https://arxiv.org/abs/1312.3005)


Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Ion Stoica, Matei Zaharia, and James
Zou. Are more llm calls all you need? towards scaling laws of compound inference systems, 2024.
[URL https://arxiv.org/abs/2403.02419.](https://arxiv.org/abs/2403.02419)


Marco Comunità, Zhi Zhong, Akira Takahashi, Shiqi Yang, Mengjie Zhao, Koichi Saito, Yukara
Ikemiya, Takashi Shibuya, Shusuke Takahashi, and Yuki Mitsufuji. Specmaskgit: Masked generative modeling of audio spectrograms for efficient audio synthesis and beyond, 2024. URL
[https://arxiv.org/abs/2406.17672.](https://arxiv.org/abs/2406.17672)


Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need
registers, 2024. [URL https://arxiv.org/abs/2309.16588.](https://arxiv.org/abs/2309.16588)


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE conference on computer vision and pattern recognition_,
pp. 248–255. Ieee, 2009.


Justin Deschenaux and Caglar Gulcehre. Promises, outlooks and challenges of diffusion language
modeling, 2024. [URL https://arxiv.org/abs/2406.11473.](https://arxiv.org/abs/2406.11473)


Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast llms via self-distillation
through time, 2025. [URL https://arxiv.org/abs/2410.21035.](https://arxiv.org/abs/2410.21035)


10


Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep
bidirectional transformers for language understanding, 2019. URL [https://arxiv.org/](https://arxiv.org/abs/1810.04805)
[abs/1810.04805.](https://arxiv.org/abs/1810.04805)


Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H.
Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi
Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data, 2022. URL
[https://arxiv.org/abs/2211.15089.](https://arxiv.org/abs/2211.15089)


Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image
synthesis, 2021. [URL https://arxiv.org/abs/2012.09841.](https://arxiv.org/abs/2012.09841)


Rohit Gandikota and David Bau. Distilling diversity and control in diffusion models, 2025. URL

[https://arxiv.org/abs/2503.10637.](https://arxiv.org/abs/2503.10637)


Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster,
Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff,
Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika,
Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation
harness, 07 2024. [URL https://zenodo.org/records/12608602.](https://zenodo.org/records/12608602)


Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky T. Q. Chen, Gabriel Synnaeve, Yossi Adi,
and Yaron Lipman. Discrete flow matching, 2024. [URL https://arxiv.org/abs/2407.](https://arxiv.org/abs/2407.15595)
[15595.](https://arxiv.org/abs/2407.15595)


Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. [http://Skylion007.github.io/](http://Skylion007.github.io/OpenWebTextCorpus)
[OpenWebTextCorpus, 2019.](http://Skylion007.github.io/OpenWebTextCorpus)


Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh
Nagarajan. Think before you speak: Training language models with pause tokens, 2024. URL
[https://arxiv.org/abs/2310.02226.](https://arxiv.org/abs/2310.02226)


Gabe Guo and Stefano Ermon. Reviving any-subset autoregressive models with principled parallel
sampling and speculative decoding, 2025. [URL https://arxiv.org/abs/2504.20456.](https://arxiv.org/abs/2504.20456)


John H Halton. Algorithm 247: Radical-inverse quasi-random point sequence. _ACM_, 7(12):701–702,
1964.


Marton Havasi, Brian Karrer, Itai Gat, and Ricky T. Q. Chen. Edit flows: Flow matching with edit
operations, 2025. [URL https://arxiv.org/abs/2506.09018.](https://arxiv.org/abs/2506.09018)


Etrit Haxholli, Yeti Z. Gurbuz, O˘gul Can, and Eli Waxman. Efficient perplexity bound and ratio matching in discrete diffusion language models. In _The Thirteenth International Conference on Learning_
_Representations_, 2025. [URL https://openreview.net/forum?id=Mri9WIfxSm.](https://openreview.net/forum?id=Mri9WIfxSm)


Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash equilibrium, 2018. URL
[https://arxiv.org/abs/1706.08500.](https://arxiv.org/abs/1706.08500)


Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. [URL https://arxiv.](https://arxiv.org/abs/2207.12598)
[org/abs/2207.12598.](https://arxiv.org/abs/2207.12598)


Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. URL

[https://arxiv.org/abs/2006.11239.](https://arxiv.org/abs/2006.11239)


Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text
degeneration, 2020. [URL https://arxiv.org/abs/1904.09751.](https://arxiv.org/abs/1904.09751)


Jaeyeon Kim, Lee Cheuk-Kit, Carles Domingo-Enrich, Yilun Du, Sham Kakade, Timothy Ngotiaoco,
Sitan Chen, and Michael Albergo. Any-order flexible length masked diffusion, 2025. URL
[https://arxiv.org/abs/2509.01025.](https://arxiv.org/abs/2509.01025)


Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL

[https://arxiv.org/abs/1412.6980.](https://arxiv.org/abs/1412.6980)


11


Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models, 2023.
[URL https://arxiv.org/abs/2107.00630.](https://arxiv.org/abs/2107.00630)


Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios
of the data distribution, 2024. [URL https://arxiv.org/abs/2310.16834.](https://arxiv.org/abs/2310.16834)


Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion
language models, 2025. [URL https://arxiv.org/abs/2505.15781.](https://arxiv.org/abs/2505.15781)


Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder,
Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative
refinement with self-feedback, 2023. [URL https://arxiv.org/abs/2303.17651.](https://arxiv.org/abs/2303.17651)


Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan
Li. Scaling up masked diffusion models on text, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2410.18514)
[2410.18514.](https://arxiv.org/abs/2410.18514)


Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li.
Your absorbing discrete diffusion secretly models the conditional distributions of clean data, 2025.
[URL https://arxiv.org/abs/2406.03736.](https://arxiv.org/abs/2406.03736)


Arnaud Pannatier, Evann Courdier, and François Fleuret. Sigma-gpts: A new approach to autoregressive models, 2024. [URL https://arxiv.org/abs/2404.09562.](https://arxiv.org/abs/2404.09562)


William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. [URL https:](https://arxiv.org/abs/2212.09748)
[//arxiv.org/abs/2212.09748.](https://arxiv.org/abs/2212.09748)


Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm
Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling
transformer inference, 2022. [URL https://arxiv.org/abs/2211.05102.](https://arxiv.org/abs/2211.05102)


Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language
models are unsupervised multitask learners. _OpenAI blog_, 1(8):9, 2019.


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer, 2023. [URL https://arxiv.org/abs/1910.10683.](https://arxiv.org/abs/1910.10683)


Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T
Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language
models, 2024. [URL https://arxiv.org/abs/2406.07524.](https://arxiv.org/abs/2406.07524)


Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and
Volodymyr Kuleshov. The diffusion duality, 2025a. [URL https://arxiv.org/abs/2506.](https://arxiv.org/abs/2506.10892)
[10892.](https://arxiv.org/abs/2506.10892)


Subham Sekhar Sahoo, Zhihan Yang, Yash Akhauri, Johnna Liu, Deepansha Singh, Zhoujun Cheng,
Zhengzhong Liu, Eric Xing, John Thickstun, and Arash Vahdat. Esoteric language models, 2025b.
[URL https://arxiv.org/abs/2506.01928.](https://arxiv.org/abs/2506.01928)


Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models, 2022.
[URL https://arxiv.org/abs/2202.00512.](https://arxiv.org/abs/2202.00512)


Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans, 2016. [URL https://arxiv.org/abs/1606.03498.](https://arxiv.org/abs/1606.03498)


Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre,
Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple
guidance mechanisms for discrete diffusion models, 2025. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2412.10193)
[2412.10193.](https://arxiv.org/abs/2412.10193)


Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and
generalized masked diffusion for discrete data, 2025. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2406.04329)
[2406.04329.](https://arxiv.org/abs/2406.04329)


12


Andy Shih, Dorsa Sadigh, and Stefano Ermon. Training and inference on any-order autoregressive
models the right way, 2022. [URL https://arxiv.org/abs/2205.13554.](https://arxiv.org/abs/2205.13554)


Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally
can be more effective than scaling model parameters, 2024. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2408.03314)
[2408.03314.](https://arxiv.org/abs/2408.03314)


Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised
learning using nonequilibrium thermodynamics. In Francis Bach and David Blei (eds.), _Proceedings_
_of the 32nd International Conference on Machine Learning_, volume 37 of _Proceedings of Machine_
_Learning_ _Research_, pp. 2256–2265, Lille, France, 07–09 Jul 2015. PMLR. URL [https://](https://proceedings.mlr.press/v37/sohl-dickstein15.html)
[proceedings.mlr.press/v37/sohl-dickstein15.html.](https://proceedings.mlr.press/v37/sohl-dickstein15.html)


Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution,
2020. [URL https://arxiv.org/abs/1907.05600.](https://arxiv.org/abs/1907.05600)


Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced
transformer with rotary position embedding, 2023. [URL https://arxiv.org/abs/2104.](https://arxiv.org/abs/2104.09864)
[09864.](https://arxiv.org/abs/2104.09864)


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
Kaiser, and Illia Polosukhin. Attention is all you need, 2023. [URL https://arxiv.org/](https://arxiv.org/abs/1706.03762)
[abs/1706.03762.](https://arxiv.org/abs/1706.03762)


Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang,
Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable
length video generation from open domain textual description, 2022. [URL https://arxiv.](https://arxiv.org/abs/2210.02399)
[org/abs/2210.02399.](https://arxiv.org/abs/2210.02399)


Dimitri von Rütte, Janis Fluri, Yuhui Ding, Antonio Orvieto, Bernhard Schölkopf, and Thomas
Hofmann. Generalized interpolating discrete diffusion, 2025. [URL https://arxiv.org/](https://arxiv.org/abs/2503.04482)
[abs/2503.04482.](https://arxiv.org/abs/2503.04482)


Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song
Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache
and parallel decoding, 2025. [URL https://arxiv.org/abs/2505.22618.](https://arxiv.org/abs/2505.22618)


Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. An empirical analysis
of compute-optimal inference for problem-solving with language models, 2024. [URL https:](https://arxiv.org/abs/2408.00724)
[//arxiv.org/abs/2408.00724.](https://arxiv.org/abs/2408.00724)


Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V.
Le. Xlnet: Generalized autoregressive pretraining for language understanding, 2020. URL
[https://arxiv.org/abs/1906.08237.](https://arxiv.org/abs/1906.08237)


Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models, 2023.
[URL https://arxiv.org/abs/2305.10601.](https://arxiv.org/abs/2305.10601)


Lijun Yu, Yong Cheng, Kihyuk Sohn, José Lezama, Han Zhang, Huiwen Chang, Alexander G.
Hauptmann, Ming-Hsuan Yang, Yuan Hao, Irfan Essa, and Lu Jiang. Magvit: Masked generative
video transformer, 2023. [URL https://arxiv.org/abs/2212.05199.](https://arxiv.org/abs/2212.05199)


Kaiwen Zheng, Yongxin Chen, Hanzi Mao, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Masked
diffusion models are secretly time-agnostic masked models and exploit inaccurate categorical
sampling, 2025. [URL https://arxiv.org/abs/2409.02908.](https://arxiv.org/abs/2409.02908)


Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, and Vicky Kalogeiton. Di[ `M` ]o: Distilling masked
diffusion models into one-step generator, 2025. URL [https://arxiv.org/abs/2503.](https://arxiv.org/abs/2503.15457)
[15457.](https://arxiv.org/abs/2503.15457)


13


**Algorithm 1** Building the Halton Unmasking Schedule


1: **Input:** Grid size _H_
2: **Output:** Ordered list of _L_ = _H_ [2] grid positions
3: schedule _←_ []
4: seen _←∅_
5: _i ←_ 1
6: **while** _|_ schedule _| < L_ **do**
7: cell _←_ ( _⌊_ Φ2( _i_ ) _· H⌋,_ _⌊_ Φ3( _i_ ) _· H⌋_ )
8: **if** cell _∈/_ seen **then**
9: seen _←_ seen _∪_ {cell}
10: schedule.append(cell)
11: **end if**
12: _i ←_ _i_ + 1
13: **end while**
14: **return** schedule


A LIMITATIONS


To match the validation perplexity of the MDLM baseline at a context length of 1024, our models
require a slight increase in parameters. We attribute this to the GroupSwap layer, and future work
will explore more efficient mechanisms for information exchange between groups in PGMs. While
PGMs offer faster inference, their training is slightly more computationally expensive (Appendix E),
as we use torch’s default attention implementation (“sdpa”) for simplicity. By reordering tokens
according to their group assignment, the self-attention matrices becomes block-diagonal. Future
work will explore efficient kernel implementations that exploit this block-diagonal sparsity. Partition
Generative Modeling is a general framework, and its application to multimodal settings remains an
open direction for future research.


B SAMPLING WITH HALTON SEQUENCES (MASKGIT)


Let _S_ = _{ℓ_ _|_ _zτ_ _[ℓ]_ [=] [[][MASK][]] _[}]_ [denote] [the] [set] [of] [masked] [positions] [at] [step] _[τ]_ [.] [MGMs] [samples]
independently at all the masked positions from a product of marginal predictions [�] _ℓ∈S_ _[p][θ]_ [(] **[x]** _[ℓ]_ _[|]_ **[ z]** _[τ]_ [)][,]

not from the joint _p_ ( **x** _|_ **z** _τ_ ). The KL divergence between the joint and the product of marginals is the
mutual information (MI) (Besnier et al., 2025):


= MI( _{x_ _[ℓ]_ _}ℓ∈S_ _|_ **z** _τ_ ) _._ (12)


DKL


_p_ ( _{x_ _[ℓ]_ _}ℓ∈S_ _|_ **z** _τ_ ) ���


- _pθ_ ( _x_ _[ℓ]_ _|_ **z** _τ_ )


_ℓ∈S_


Empirically, the denoiser is most confident at positions close to previously generated tokens. Therefore, the decoded tokens tend to cluster together. This leads to a larger MI, with a higher risk of
generating inconsistent tokens. Besnier et al. (2025) propose to replace the confidence-based ordering
with a _Halton sequence_ (Halton, 1964), a low-discrepancy sequence whose consecutive points are far
apart in space. Formally, let _aj_ ( _i_ ) denote the _j_ -th digit of the representation of _i ∈_ N+ in base _b_, so
that _i_ = [�] _j_ _[a][j]_ [(] _[i]_ [)] _[ b][j]_ [.] [Let the] _[ radical-inverse function]_ [ Φ] _[b]_ [denote the inverse of] _[ i]_ [ in base] _[ b]_ [:]


Φ _b_ ( _i_ ) =


_m_

- _aj_ ( _i_ ) _b_ _[−]_ [(] _[j]_ [+1)] _∈_ [0 _,_ 1) _._ (13)


_j_ =0


Recall that MaskGIT operates over a square grid of VQGAN tokens (Esser et al., 2021), of size
_H × H_ ( _L_ = _H_ [2] ). To build the unmasking order (Algo. 1), Besnier et al. (2025) iterate over positive
integers _i_ and compute the cell coordinates ( _⌊_ Φ2( _i_ ) _· H⌋,_ _⌊_ Φ3( _i_ ) _· H⌋_ ). If the cell was already visited
for some _i_ _[′]_ _<_ _i_, it is skipped. This continues until all _L_ cells are visited. The Halton scheduler
consistently improves the FID and, unlike confidence-based sampling, continues to improve with
more inference steps (Besnier et al., 2025). In our image experiments, we use both confidence and
Halton schedulers.


14


**(a)** **Halton** **Sampling**


**(b)** **Random** **Sampling**


**(c)** **Confidence-based** **Sampling**


_t_ = 1 **SAMPLING** **TIME** _t_ = 0


Figure 5: Comparison of unmasking schedules on a 2D grid. **(a)** Halton sampling covers space
uniformly. **(b)** Random sampling can leave large areas empty. **(c)** With confidence-based sampling,
new tokens are decoded close to existing ones, leading to high mutual information and a risk of
inconsistent generation.


C EXPERIMENTAL DETAILS


We trained all models from scratch. Our baselines achieve similar performance as reported by Sahoo
et al. (2024). On LM1B, we obtain a validation perplexity of 27.67 after 1M steps (compared to
MDLM’s reported 27.04), while on OWT, we reach 23.07 (versus MDLM’s 23.21).


Minor differences can be expected since estimating the perplexity of diffusion language models
involves a Monte-Carlo approximation of the NELBO (1) with finitely many samples. Although we
used libraries (e.g., PyTorch) with the same version as MDLM, differences in compute environments
and underlying software stacks may also contribute to these variations. Since the performance gap is
small, we are confident that we used the code of MDLM correctly.


C.1 LM1B


For the LM1B dataset, we employed the bert-base-uncased tokenizer with a context length of
128 tokens, padding shorter sequences. Our architecture consisted of a Diffusion Transformer (DiT)
with 12 transformer blocks, 12 attention heads, a hidden dimension of 768, and a dropout rate of 0.1.
We optimized the model using Adam (Kingma & Ba, 2017) (learning rate 3e-4, betas of (0.9, 0.999),
epsilon 1e-8) without weight decay. [We based our implementation on the official MDLM codebase.](https://github.com/kuleshov-group/mdlm)
We trained with a global batch size of 512 across 8 GPUs (2 nodes with 4 GPUs), gradient clipping at
1.0, and a constant learning rate with 2,500 steps of linear warmup. We trained for 1 million steps
with an EMA rate of 0.9999. Besides the neural network hyperparameters, the other parameters were
unchanged when training the PGM.


C.2 OWT


For the OpenWebText (OWT) dataset, we used the GPT-2 tokenizer with a context length of 1024
tokens. Our architecture consisted of a Diffusion Transformer (DiT) with 12 transformer blocks,


15


Table 3: Latency and throughput for a single forward+backward pass of the MDLMs and PGMs,
computed on a single A100-SXM4-80GB GPU. On LM1B, PGM introduces a negligible overhead
over MDLM. On OWT, our PGM with 6 encoder and decoder layers and an embedding dimension of
1024 achieves around 75% of the training throughput of MDLM. Recall that at inference, the same
PGM is around 5 _×_ faster than MDLM.


**Forward Pass** **Forward + Backward**
**Model**

**Latency (ms)** **Seq/sec** **Latency (ms)** **Seq/Sec**


_LM1B (context length 128, batch size 64, trained on 8 GPUs)_
MDLM 0 _._ 03 _±_ 0 _._ 00 1 _[′]_ 978 _._ 87 _±_ 44 _._ 21 0 _._ 08 _±_ 0 _._ 00 714 _._ 80 _±_ 15 _._ 47
PGM 6 / 6 0 _._ 03 _±_ 0 _._ 00 1 _[′]_ 966 _._ 60 _±_ 102 _._ 14 0 _._ 08 _±_ 0 _._ 00 794 _._ 42 _±_ 18 _._ 81


_OpenWebText (context length 1024, batch size 32, trained on 16 GPUs)_
MDLM 0 _._ 13 _±_ 0 _._ 00 233 _._ 28 _±_ 2 _._ 58 0 _._ 39 _±_ 0 _._ 00 80 _._ 86 _±_ 0 _._ 15
PGM 8 / 8 0 _._ 17 _±_ 0 _._ 00 188 _._ 07 _±_ 0 _._ 75 0 _._ 47 _±_ 0 _._ 00 68 _._ 04 _±_ 0 _._ 08
PGM 6 / 6 (dim. 1024) 0 _._ 18 _±_ 0 _._ 00 176 _._ 47 _±_ 0 _._ 65 0 _._ 50 _±_ 0 _._ 00 62 _._ 85 _±_ 0 _._ 19


12 attention heads, a hidden dimension of 768, and a dropout rate of 0.1. We optimized the model
using Adam (Kingma & Ba, 2017) with a learning rate of 3e-4, betas of (0.9, 0.999), and epsilon
of 1e-8, without weight decay. We trained with a global batch size of 512 across 16 GPUs (4 nodes
with 4 GPUs). We applied gradient clipping at 1.0 and used a constant learning rate schedule with
2,500 steps of linear warmup. The model was trained for 1 million steps with an EMA rate of 0.9999.
Besides the neural network hyperparameters, the other parameters were unchanged when training the
PGM.


C.3 IMAGENET


For the ImageNet experiments, we used a pre-trained VQGAN tokenizer (Esser et al., 2021; Besnier
et al., 2025), following the setup of Besnier et al. (2025). The images are tokenized into sequences
of 1024 tokens. This allowed for a direct comparison between PGM and MaskGIT, both trained
in the codebase of Besnier et al. (2025) and the FID is evaluated using the Halton sampler and the
confidence sampler. We compute the FID between 50k generated images and the validation set,
following Besnier et al. (2025)


All models use 24 transformer blocks. For PGM, we add a GroupSwap layer to enable information
exchange between partition groups. We use the same hyperparameters as HaltonMaskGIT for all
models, except we reduce the training duration to 500k steps (from 2M) due to computational
constraints. All models are trained to be class-conditional, which enables the use of classifier-free
guidance to significantly improve performance.


C.4 IMPACT OF NUMERICAL PRECISION ON SAMPLING


Zheng et al. (2025) identified that Masked Diffusion Models often achieve lower Generative Perplexity
results because of underflow in the logits when sampling using low precision. The resulting decrease
in token diversity can make evaluations based solely on Generative Perplexity misleading. Hence, we
always cast the logits to FP64 before sampling.


C.5 SAMPLE-BASED EVALUATION


**Generative Perplexity** We use the Generative Perplexity to evaluate the quality of samples, following prior work (Lou et al., 2024; Sahoo et al., 2024; Deschenaux & Gulcehre, 2025). The Generative
Perplexity measures how well a reference model (in our case, GPT-2 Large) can predict the next token
in generated sequences. Specifically, we generate 1 _[′]_ 024 samples from each model being evaluated.
For each generated sample, we compute the Generative Perplexity using GPT-2 Large as follows:


_,_ (14)


_L_


Perplexity = exp


_−_ [1]


_L_


log _p_ GPT-2 Large( _xi|x<i_ )

_i_ =1


16


PGM (6/6, dim=1024)


0.0 0.2 0.4 0.6 0.8 1.0
Training step 1e6


10

9

8

7

6

5

4

3


MDLM


0.0 0.2 0.4 0.6 0.8 1.0
Training step 1e6


10

9

8

7

6

5

4

3


MDLM (Complementary masking)


0.0 0.2 0.4 0.6 0.8 1.0
Training step 1e6


10

9

8

7

6

5

4

3


Figure 6: Training loss of MDLM, MDLM with Complementary Masking (Section 5.3) and PGM.
Complementary masking seems to introduce spikes in the loss, even though it did not cause the
models to diverge.


where _L_ is the length of the sequence, _xi_ is the _i_ -th token, and _p_ GPT-2 Large( _xi|x<i_ ) is the probability
assigned by GPT-2 Large to token _xi_ given the preceding tokens _x<i_ .


**Unigram** **Entropy** Unfortunately, a low Generative Perplexity can be achieved by generating
repetitive text. To catch such cases, we compute the average unigram entropy of the generated
samples:


_v∈V_


_,_ (15)
_L_


Unigram Entropy = _−_ [1]

_N_


_N_


_i_ =1


_c_ ( _v,_ **x** [(] _[i]_ [)] )


**x** [(] _[i]_ [)] )

log _[c]_ [(] _[v,]_ **[ x]** [(] _[i]_ [)][)]
_L_ _L_


where _V_ is the vocabulary, _v_ is a token of the vocabulary, and _c_ ( _v,_ **x** ) is the empirical appearance
count of the token _v_ in the sequence **x** . Low unigram entropy helps us to catch degenerate generation,
as shown by Dieleman et al. (2022).


**Fréchet Inception Distance and Inception Score** On image generation tasks, we evaluate the
quality of samples using the Fréchet Inception Distance (FID) (Heusel et al., 2018) and Inception
Score (IS) (Salimans et al., 2016). Both metrics are computed using 50 _[′]_ 000 images, following the
standard practice.


D ADDITIONAL RESULTS


D.1 IMPACT OF CONTEXT LENGTH ON THE EFFECTIVENESS OF COMPLEMENTARY MASKING


There are three key differences between our experiments on LM1B and OWT. First, we used different
tokenizers: bert-base-uncased for LM1B and GPT2’s tokenizer for OWT, following the setup
of MDLM (Sahoo et al., 2024). Second, the context lengths differ significantly: 128 tokens for LM1B
versus 1024 for OWT. Third, we train on different datasets that might have different characteristics.


We observed that complementary masking helps when training on OWT using a shorter context
length of 128 tokens with the GPT-2 tokenizer. Indeed, after 200k training step, the MDLM with
complementary masking achieved a validation PPL of 37 _._ 92, outperforming the standard MDLM,
which reached 39 _._ 90. This suggests that PGMs may not need extra parameters when the sequence
length is short. Exploring the use of PGMs in domains where the sequence length is short, such as
modeling chemical sequences, is a promising direction for future work.


D.2 MDLM+SDTT VS PGM+SDTT


The precision of logits during sampling can have a significant effect on sample quality, as noted in
Appendix C.4. Hence, we cast all logits to FP64 prior to sampling, unlike the original MDLM and
SDTT implementations.


17


Using higher precision also affects distillation, which compresses two sampling steps into one. As
shown in Table 7, models distilled with float32 achieve lower Generative Perplexity than those trained
with mixed precision (bfloat16). We therefore report float32 results in the main body.


D.3 ADDITIONAL RESULTS ON IMAGENET


Table 8 and Table 9 show the FID, IS, latency, and throughput for the Confidence and Halton
samplers. Overall, the Halton sampler works best for both MaskGIT and PGM. With 32 steps and the
confidence-based sampler, PGM 12/12 gets a better FID than MaskGIT and is 3 _._ 58 _×_ faster. With
32 steps and the Halton sampler, PGM has a slightly higher FID than MaskGIT (5.54 vs 5.35), but
is 7 _._ 5 _×_ faster. If we increase the number of sampling steps to 64, PGM achieves an FID of 4.56,
which is better than MaskGIT, and is 3 _._ 92 _×_ faster. Generally, the 12/12 variant outperforms the
14/10 variant, which suggests that balanced number of layers in the encoder and decoder is beneficial,
just as for language modeling (Table 5).


D.4 TRAINING STABILITY


Complementary masking introduces occasional spikes in the training loss in both MDLMs and
PGMs, as shown in Figure 6. This phenomenon should be kept in mind when scaling PGMs to larger
sizes. Despite these spikes, all runs converged on the first attempt. We observed different precision
requirements between models. For loss computations, MDLMs performed best with BF16 precision,
while PGMs achieved better results with FP32 precision. Both models use mixed precision within
the neural network; the precision difference only affects computations performed outside the model,
such as the loss calculation.


D.5 ADDITIONAL DOWNSTREAM TASKS


Table 4 reports additional downstream results as in Deschenaux & Gulcehre (2025), where PGM
outperforms MDLM on all but one benchmark, with only a small gap on the latter. We evaluate models
with the lm-eval-harness library (Gao et al., 2024), originally designed for autoregressive LMs
and adapted here for MDLM. For multiple-choice tasks, lm-eval-harness computes the loglikelihood of each candidate answer **y** _i_ given a prefix **x**, i.e., _p_ ( **y** _i|_ **x** ), and selects the answer with the
highest score.


While lm-eval-harness uses the log-likelihood of the continuation, the NELBO objective (1)
bounds the log-likelihood of the _complete_ sequence ( **x** _,_ **y** _i_ ). However, we only need to know which
continuation achieves the highest log-likelihood, not to compute the exact log-likelihood. Using
Bayes’ theorem, we note that
log _p_ ( **y** _i|_ **x** ) = log _p_ ( **x** _,_ **y** _i_ ) _−_ log _p_ ( **x** ) _∝_ log _p_ ( **x** _,_ **y** _i_ ) _,_ (16)
since log _p_ ( **x** ) is constant with respect to **y** _i_ . Therefore, we can simply evaluate the variational bound
on log _p_ ( **x** _,_ **y** _i_ ) to select the most likely continuation _yi_ .


D.6 PERFORMANCE ON LONGER CONTEXT LENGTH


Due to the high computational cost, we were unable to train models with context lengths greater
than 1024. Nevertheless, we report the latency and throughput of both MDLM and PGM at a context
length of 4096. As shown in Table 10, PGM remains substantially faster than MDLM in this setting.


E COMPUTATIONAL COSTS


This section presents the computational costs associated with the models reported in this paper. We
exclude costs associated with exploratory experiments that yielded inferior results and were not
included in this manuscript.


E.1 TRAINING COSTS


Training PGMs is currently slower than training MGMs since we use torch.sdpa with dense
tensor masks. Future work should explore efficient kernels to address this limitation. We measure the


18


Model LAMBADA ARC-e ARC-c HSwag MathQA PIQA WinoG


MDLM 38.52 34.26 **24.66** 31.54 20.70 57.89 51.93
PGM 8 / 8 **46.98** 37.37 24.06 33.10 21.24 59.09 51.30
PGM 6 / 6 (1024) 41.39 **38.80** 22.95 **33.92** **21.71** **61.43** **54.30**


Table 4: Accuracy on downstream tasks. We evaluate MDLM and PGM on LAMBADA, ARC
Easy and Challenge, HellaSwag, MathQA, PIQA, and WinoGrande. Both models show comparable
performance across tasks. PGM outperforms MDLM on all but one benchmark, where the difference
between MDLM and PGM 8 / 8 is small.


latency and throughput using a single NVIDIA A100-SXM4-80GB GPU, with results reported in
Table 3. We compute the mean and standard deviation over 100 batches after 2 warmup batches.


The total training duration approximately equals the per-step latency multiplied by the number of
steps. Experiments with complementary masking required twice the computational resources due
to larger batch sizes and gradient accumulation. Training times for 1M steps varied by dataset:
approximately 22 hours for LM1B, 4.5 days for OWT, and 3.8 days for ImageNet.


Despite the current training overhead, we are confident that future work can improve the training
efficiency of PGMs, thanks to their block-diagonal attention patterns, once the tokens are grouped
together along the sequence-length axis.


E.2 INFERENCE COSTS


We evaluate the inference efficiency of PGMs compared to MDLMs and GPT-2 with KV caching. As
shown in Figure 1, PGMs achieve around 5 _−_ 5 _._ 5 _×_ improvements in throughput over MDLM while
reaching superior Generative Perplexity. For inference measurements, we use a single NVIDIA A100SXM4-80GB GPU. The efficiency gain stems from the ability of PGMs to process only unmasked
tokens during inference, as illustrated in Figure 2. Table 6 compares MDLM and PGMs on the
Generative Perplexity, unigram entropy, latency, and throughput. We compute the mean and standard
deviation of the latency and throughput over 20 batches after two warmup batches.


E.3 LICENSING


Our code and model artifacts will be released under the MIT license. The OWT dataset (Gokaslan
& Cohen, 2019) is available under the Apache License 2.0. We were unable to identify a specific
license for the LM1B dataset (Chelba et al., 2014). The images in ImageNet remain the property of
their respective copyright holders.


**Algorithm 2** Simplifed Sampling for PGMs


1: **Input:** Batch size BS, number of steps K, model length L, special BOS index
2: **Output:** Generated samples x
3: x _←_ empty_tensor(BS, 1) _▷_ _Initialize_
4: x[:, 0] _←_ BOS _▷_ _Set BOS as first token_
5: k _←_ L/K _▷_ _Number of tokens to denoise at each step_
6: decoded_positions _←_ zeros(BS, 1) _▷_ _Keep track of already-decoded and positions to decode_
7: positions_to_decode _←_ 1+ rand_row_perm(BS, L-1) _▷_ _Each rows is a permutation of {_ 1 _, ..., L}_
8: **for** _ in range(K) **do**
9: pos_to_decode _←_ positions_to_decode[:, :k] _▷_ _Random positions to be predicted_
10: new_values _←_ pgm_predict(x, decoded_positions, pos_to_decode)
11: _x ←_ concat([x, new_values], dim=1) _▷_ _Add new values to the sequence length dimension_
12: decoded_positions _←_ concat([decoded_positions, pos_to_decode], dim=1)
13: positions_to_decode _←_ positions_to_decode[:, k:] _▷_ _Remove the k decoded positions_
14: **end for**
15: out _←_ reorder(x, decoded_positions) _▷_ _Sort based on positions_
16: **return** out


19


**Algorithm 3** MDLM-equivalent sampling for PGMs.


1: **Input:** Batch size BS, number of steps K, model length L, special BOS index
2: **Output:** Generated samples x
3: x _←_ empty_tensor(BS, 1) _▷_ _Initialize_
4: x[:, 0] _←_ BOS _▷_ _Set BOS as first token_
5: k _←_ L/K _▷_ _Number of tokens to denoise at each step_
6: clean_positions _←_ zeros(BS, 1) _▷_ _Keep track of clean and noisy positions_
7: concrete_lengths _←_ ones(BS, 1) _▷_ _Keep track of the actual length of each sequence (some are_
_padded)._
8: noisy_positions _←_ 1+ rand_row_perm(BS, L-1)
9: **for** _ in range(K) **do**
10: n_denoise_per_seq, noisy_pos_input _←_ **sample_noisy** (noisy_positions, k) _▷_ _Algorithm_
_Algo. 4_
11: new_values _←_ pgm_predict(x, clean_positions, noisy_pos_input)
12: x, clean_positions, noisy_positions, concrete_lengths _←_ **extract_predictions** (
13: x, _▷_ _Algorithm Algo. 5_
14: clean_positions,
15: noisy_positions,
16: noisy_pos_input,
17: concrete_lengths,
18: n_denoise_per_seq,
19: new_values)
20: **end for**
21: out _←_ reorder(x, clean_positions) _▷_ _Sort based on clean_positions_
22: **return** out


**Algorithm 4** Sample the number of tokens to denoise from a binomial distribution and pad the input.


1: **Input:** Noisy positions tensor, probability of denoising prob_denoise, model length L, concrete
lengths tensor
2: **Output:** Noisy positions to denoise
3: n_denoise_per_seq _←_ binomial(BS, L, prob_denoise) _▷_ _Sample from binomial distribution_
4: n_denoise_per_seq _←_ min(n_denoise_per_seq, L - concrete_lengths) _▷_ _Don’t denoise more_
_than available_
5: denoise_seq_len _←_ max(n_denoise_per_seq, 0) _▷_ _Maximum number of tokens to denoise_
6: **if** denoise_seq_len = 0 **then**
7: **return** empty_tensor() _▷_ _Nothing to denoise_
8: **end if**
9: noisy_pos_input _←_ noisy_positions[:, :denoise_seq_len] _▷_ _Some predictions won’t be used_
10: **return** n_denoise_per_seq, noisy_pos_input


20


**Algorithm 5** Extract the correct number of predictions per sequence


1: **Input:** x, concrete_lengths, n_denoise_per_seq, denoised_token_values, clean_positions,
noisy_positions, noisy_pos_input
2: **Output:** Updated x, clean_positions, noisy_positions, concrete_lengths
3: new_concrete_lengths _←_ concrete_lengths + n_denoise_per_seq _▷_ _Update sequence lengths_
4: n_tok_to_add _←_ max(new_concrete_lengths) - shape(x, 1) _▷_ _Calculate padding needed_
5: **if** n_tok_to_add > 0 **then**
6: pad _←_ zeros(BS, n_tok_to_add) _▷_ _Create padding tensor_
7: x _←_ concat(x, pad, dim=1) _▷_ _Pad the sequences_
8: clean_positions _←_ concat(clean_positions, pad, dim=1) _▷_ _Pad the positions_
9: **end if**
10: **for** i in range(BS) **do**
11: **if** n_denoise_per_seq[i] = 0 **then**
12: continue _▷_ Skip if no tokens to denoise
13: **end if**
14: x[i, concrete_lengths[i]:new_concrete_lengths[i]] _←_
15: denoised_token_values[i, :n_denoise_per_seq[i]]
16: clean_positions[i, concrete_lengths[i]:new_concrete_lengths[i]] _←_
17: noisy_pos_input[i, :n_denoise_per_seq[i]]
18: noisy_positions[i, :shape(noisy_positions, 1) - n_denoise_per_seq[i]] _←_
19: noisy_positions[i, n_denoise_per_seq[i]:]
20: **end for**
21: **return** x, clean_positions, noisy_positions, new_concrete_lengths


**Model (LM1B)** **Val.** **PPL** _↓_


_200k steps_
MDLM 34.29
MDLM (Compl. masking) **30.87**
PGM 8 / 4 32.83
PGM 10 / 2 33.55
PGM 4 / 8 32.84

PGM 6 / 6 (lsm) 32.70
PGM 6 / 6 (mean) 33.89


_1M steps_
MDLM 27.67
MDLM (Compl. masking) **25.72**


**Model (OWT)** **Val.** **PPL** _↓_


_200k steps_
MDLM 25.35
MDLM (Compl. masking) 25.32
PGM 6 / 6 26.96
PGM 8 / 8 25.10
PGM 10 / 6 25.19

PGM 6 / 6 (dim. 1024) **23.75**


_1M steps_
MDLM 23.07
MDLM (Compl. masking) 22.98
PGM 8 / 8 22.61


Table 5: Perplexity evaluations. Validation perplexity of the Masked Diffusion Language Model
(MDLM) and PGMs (ours) on LM1B and OpenWebText (OWT). The row _MDLM (Compl._ _masking)_
denotes an MDLM trained with the complementary masking strategy discussed in Section 5.3. The
row _PGM k / m_ denotes a PGM with _k_ encoder and _m_ decoder layers, and we highlighted the best
PGM results in gray. _lsm_ and _mean_ denote the _logsumexp_ and _mean_ queries initializations (Section 4).
**Takeaway:** using the same number of layers in the encoder and decoder, and data-independent
queries performed best. On LM1B, our PGM reaches 1.95 lower perplexity than MDLM after 1M
steps. On OWT, we grow the embedding dimension or the number of layers to outperform MDLM
on OWT.


21


Table 6: Sample quality and efficiency on OpenWebText with different numbers of sampling steps.
We generate sequences of 1024 tokens with a batch size of 32 to measure the latency and throughput.
PGM 6 / 6 with a hidden dimension of 1024 and uniform sampling achieves at least a 5 _×_ latency and
throughput improvement over MDLM, with better Generative Perplexity and matching entropy.


**Model** **Gen.** **PPL** _↓_ **Entropy** _↑_ **Latency** _↓_ **Throughput** _↑_
**(ms)** **(tok/s)**


_MDLM_
32 steps 192 _._ 31 5 _._ 73 8 _._ 037 _±_ 0 _._ 01 4 _[′]_ 077 _._ 08 _±_ 3 _._ 06
64 steps 142 _._ 58 5 _._ 69 15 _._ 82 _±_ 0 _._ 01 2 _[′]_ 070 _._ 67 _±_ 0 _._ 69
128 steps 122 _._ 89 5 _._ 67 31 _._ 41 _±_ 0 _._ 01 1 _[′]_ 043 _._ 22 _±_ 0 _._ 16
256 steps 113 _._ 96 5 _._ 66 62 _._ 54 _±_ 0 _._ 01 523 _._ 90 _±_ 0 _._ 06
512 steps 109 _._ 05 5 _._ 64 124 _._ 94 _±_ 0 _._ 16 262 _._ 26 _±_ 0 _._ 33
1024 steps 106 _._ 75 5 _._ 64 249 _._ 31 _±_ 0 _._ 11 131 _._ 42 _±_ 0 _._ 05


_PGM 8 / 8 (uniform sampling)_
32 steps 189 _._ 02 5 _._ 73 1 _._ 55 _±_ 0 _._ 01 21 _[′]_ 120 _._ 99 _±_ 83 _._ 59
64 steps 143 _._ 79 5 _._ 69 3 _._ 00 _±_ 0 _._ 01 10 _[′]_ 914 _._ 91 _±_ 41 _._ 69
128 steps 122 _._ 21 5 _._ 66 5 _._ 86 _±_ 0 _._ 02 5 _[′]_ 585 _._ 57 _±_ 24 _._ 49
256 steps 112 _._ 48 5 _._ 65 11 _._ 64 _±_ 0 _._ 03 2 _[′]_ 814 _._ 99 _±_ 9 _._ 33
512 steps 108 _._ 76 5 _._ 64 22 _._ 98 _±_ 0 _._ 02 1 _[′]_ 425 _._ 89 _±_ 1 _._ 61
1024 steps 107 _._ 03 5 _._ 63 45 _._ 84 _±_ 0 _._ 03 714 _._ 71 _±_ 0 _._ 50


_PGM 8 / 8 (non uniform sampling)_
32 steps 194 _._ 09 5 _._ 73 2 _._ 07 _±_ 0 _._ 02 15 _[′]_ 764 _._ 09 _±_ 192 _._ 12
64 steps 143 _._ 60 5 _._ 69 3 _._ 90 _±_ 0 _._ 07 8 _[′]_ 405 _._ 14 _±_ 158 _._ 01
128 steps 124 _._ 38 5 _._ 67 7 _._ 41 _±_ 0 _._ 08 4 _[′]_ 419 _._ 77 _±_ 53 _._ 27
256 steps 116 _._ 85 5 _._ 66 14 _._ 73 _±_ 0 _._ 19 2 _[′]_ 223 _._ 63 _±_ 28 _._ 47
512 steps 111 _._ 11 5 _._ 64 28 _._ 15 _±_ 0 _._ 32 1 _[′]_ 163 _._ 79 _±_ 13 _._ 25
1024 steps 108 _._ 24 5 _._ 63 54 _._ 62 _±_ 0 _._ 66 599 _._ 97 _±_ 7 _._ 27


_PGM 6 / 6 (dim._ _1024, uniform sampling)_
32 steps 185 _._ 16 5 _._ 73 1 _._ 59 _±_ 0 _._ 01 20 _[′]_ 569 _._ 99 _±_ 95 _._ 63
64 steps 138 _._ 87 5 _._ 70 3 _._ 03 _±_ 0 _._ 01 10 _[′]_ 805 _._ 31 _±_ 14 _._ 11
128 steps 116 _._ 95 5 _._ 67 5 _._ 93 _±_ 0 _._ 01 5 _[′]_ 518 _._ 09 _±_ 13 _._ 46
256 steps 108 _._ 51 5 _._ 65 11 _._ 77 _±_ 0 _._ 01 2 _[′]_ 782 _._ 78 _±_ 3 _._ 46
512 steps 101 _._ 94 5 _._ 63 23 _._ 25 _±_ 0 _._ 01 1 _[′]_ 408 _._ 88 _±_ 1 _._ 05
1024 steps 99 _._ 64 5 _._ 62 46 _._ 31 _±_ 0 _._ 02 707 _._ 52 _±_ 0 _._ 34


_PGM 6 / 6 (dim._ _1024, non-uniform sampling)_
32 steps 191 _._ 30 5 _._ 74 2 _._ 12 _±_ 0 _._ 07 15 _[′]_ 415 _._ 56 _±_ 467 _._ 20
64 steps 138 _._ 67 5 _._ 69 3 _._ 940 _±_ 0 _._ 06 8 _[′]_ 318 _._ 72 _±_ 135 _._ 47
128 steps 118 _._ 17 5 _._ 67 7 _._ 60 _±_ 0 _._ 09 4 _[′]_ 311 _._ 80 _±_ 54 _._ 92
256 steps 108 _._ 93 5 _._ 65 14 _._ 84 _±_ 0 _._ 20 2 _[′]_ 207 _._ 71 _±_ 29 _._ 71
512 steps 105 _._ 41 5 _._ 64 28 _._ 56 _±_ 0 _._ 33 1 _[′]_ 147 _._ 17 _±_ 13 _._ 47
1024 steps 102 _._ 93 5 _._ 62 55 _._ 50 _±_ 0 _._ 36 590 _._ 37 _±_ 3.85


22


Table 7: Generative perplexity of MDLM and PGM after distillation with varying precision.


**Model** **Gen.** **PPL** _↓_ **Entropy** _↑_ **Latency** _↓_ **Throughput** _↑_
**(ms)** **(tok/s)**


_MDLM + SDTT (loss in BF16)_
32 steps 66 _._ 26 5 _._ 49 8 _._ 037 _±_ 0 _._ 01 4 _[′]_ 077 _._ 08 _±_ 3 _._ 06
64 steps 53 _._ 98 5 _._ 46 15 _._ 82 _±_ 0 _._ 01 2 _[′]_ 070 _._ 67 _±_ 0 _._ 69
128 steps 48 _._ 02 5 _._ 44 31 _._ 41 _±_ 0 _._ 01 1 _[′]_ 043 _._ 22 _±_ 0 _._ 16
256 steps 45 _._ 86 5 _._ 42 62 _._ 54 _±_ 0 _._ 01 523 _._ 90 _±_ 0 _._ 06
512 steps 44 _._ 21 5 _._ 40 124 _._ 94 _±_ 0 _._ 16 262 _._ 26 _±_ 0 _._ 33
1024 steps 43 _._ 19 5 _._ 38 249 _._ 31 _±_ 0 _._ 11 131 _._ 42 _±_ 0 _._ 05


_MDLM + SDTT (loss in FP32)_
32 steps 61 _._ 65 5 _._ 46 8 _._ 037 _±_ 0 _._ 01 4 _[′]_ 077 _._ 08 _±_ 3 _._ 06
64 steps 50 _._ 65 5 _._ 43 15 _._ 82 _±_ 0 _._ 01 2 _[′]_ 070 _._ 67 _±_ 0 _._ 69
128 steps 45 _._ 06 5 _._ 40 31 _._ 41 _±_ 0 _._ 01 1 _[′]_ 043 _._ 22 _±_ 0 _._ 16
256 steps 41 _._ 70 5 _._ 37 62 _._ 54 _±_ 0 _._ 01 523 _._ 90 _±_ 0 _._ 06
512 steps 40 _._ 63 5 _._ 36 124 _._ 94 _±_ 0 _._ 16 262 _._ 26 _±_ 0 _._ 33
1024 steps 39 _._ 50 5 _._ 32 249 _._ 31 _±_ 0 _._ 11 131 _._ 42 _±_ 0 _._ 05


_PGM 6 / 6 (dim._ _1024) + SDTT (loss in BF16)_
32 steps 91 _._ 61 5 _._ 56 1 _._ 59 _±_ 0 _._ 01 20 _[′]_ 569 _._ 99 _±_ 95 _._ 63
64 steps 72 _._ 73 5 _._ 52 3 _._ 03 _±_ 0 _._ 01 10 _[′]_ 805 _._ 31 _±_ 14 _._ 11
128 steps 63 _._ 83 5 _._ 49 5 _._ 93 _±_ 0 _._ 01 5 _[′]_ 518 _._ 09 _±_ 13 _._ 46
256 steps 58 _._ 74 5 _._ 47 11 _._ 77 _±_ 0 _._ 01 2 _[′]_ 782 _._ 78 _±_ 3 _._ 46
512 steps 58 _._ 77 5 _._ 47 23 _._ 25 _±_ 0 _._ 01 1 _[′]_ 408 _._ 88 _±_ 1 _._ 05
1024 steps 56 _._ 47 5 _._ 46 46 _._ 31 _±_ 0 _._ 02 707 _._ 52 _±_ 0 _._ 34


_PGM 6 / 6 (dim._ _1024) nucleus (p=0.9)+ SDTT (loss in BF16)_
32 steps 68 _._ 33 5 _._ 50 1 _._ 74 _±_ 0 _._ 01 18 _[′]_ 866 _._ 12 _±_ 18 _._ 35
64 steps 53 _._ 88 5 _._ 45 3 _._ 18 _±_ 0 _._ 01 10 _[′]_ 307 _._ 16 _±_ 6 _._ 58
128 steps 46 _._ 99 5 _._ 42 6 _._ 10 _±_ 0 _._ 01 5 _[′]_ 375 _._ 20 _±_ 2 _._ 40
256 steps 43 _._ 22 5 _._ 40 11 _._ 95 _±_ 0 _._ 01 2 _[′]_ 742 _._ 74 _±_ 1 _._ 32
512 steps 42 _._ 79 5 _._ 39 23 _._ 63 _±_ 0 _._ 01 1 _[′]_ 386 _._ 79 _±_ 0 _._ 69
1024 steps 40 _._ 99 5 _._ 38 46 _._ 83 _±_ 0 _._ 02 699 _._ 80 _±_ 0 _._ 24


_PGM 6 / 6 (dim._ _1024) + SDTT (loss in FP32)_
32 steps 84 _._ 97 5 _._ 52 1 _._ 74 _±_ 0 _._ 01 20 _[′]_ 569 _._ 99 _±_ 95 _._ 63
64 steps 67 _._ 60 5 _._ 49 3 _._ 18 _±_ 0 _._ 01 10 _[′]_ 805 _._ 31 _±_ 14 _._ 11
128 steps 60 _._ 06 5 _._ 47 6 _._ 10 _±_ 0 _._ 01 5 _[′]_ 518 _._ 09 _±_ 13 _._ 46
256 steps 55 _._ 97 5 _._ 45 11 _._ 95 _±_ 0 _._ 01 2 _[′]_ 782 _._ 78 _±_ 3 _._ 46
512 steps 54 _._ 13 5 _._ 44 23 _._ 13 _±_ 0 _._ 01 1 _[′]_ 408 _._ 88 _±_ 1 _._ 05
1024 steps 52 _._ 77 5 _._ 44 46 _._ 83 _±_ 0 _._ 02 707 _._ 52 _±_ 0 _._ 34


_PGM 6 / 6 (dim._ _1024) nucleus (p=0.9)+ SDTT (loss in FP32)_
32 steps 63 _._ 46 5 _._ 45 1 _._ 59 _±_ 0 _._ 01 18 _[′]_ 866 _._ 12 _±_ 18 _._ 35
64 steps 49 _._ 94 5 _._ 41 3 _._ 03 _±_ 0 _._ 01 10 _[′]_ 307 _._ 16 _±_ 6 _._ 58
128 steps 43 _._ 84 5 _._ 39 5 _._ 93 _±_ 0 _._ 01 5 _[′]_ 375 _._ 20 _±_ 2 _._ 40
256 steps 40 _._ 76 5 _._ 36 11 _._ 77 _±_ 0 _._ 01 2 _[′]_ 742 _._ 74 _±_ 1 _._ 32
512 steps 39 _._ 46 5 _._ 36 23 _._ 25 _±_ 0 _._ 01 1 _[′]_ 386 _._ 79 _±_ 0 _._ 69
1024 steps 38 _._ 81 5 _._ 35 46 _._ 31 _±_ 0 _._ 02 699 _._ 80 _±_ 0 _._ 24


_PGM 8 / 8 + SDTT (loss in BF16)_
32 steps 102 _._ 64 5 _._ 54 1 _._ 55 _±_ 0 _._ 01 21 _[′]_ 120 _._ 99 _±_ 83 _._ 59
64 steps 82 _._ 93 5 _._ 50 3 _._ 00 _±_ 0 _._ 01 10 _[′]_ 914 _._ 91 _±_ 41 _._ 69
128 steps 73 _._ 19 5 _._ 48 5 _._ 86 _±_ 0 _._ 02 5 _[′]_ 585 _._ 57 _±_ 24 _._ 49
256 steps 70 _._ 30 5 _._ 47 11 _._ 64 _±_ 0 _._ 03 2 _[′]_ 814 _._ 99 _±_ 9 _._ 33
512 steps 68 _._ 07 5 _._ 46 22 _._ 98 _±_ 0 _._ 02 1 _[′]_ 425 _._ 89 _±_ 1 _._ 61
1024 steps 65 _._ 87 5 _._ 44 45 _._ 84 _±_ 0 _._ 03 714 _._ 71 _±_ 0 _._ 50


_PGM 8 / 8 + SDTT (loss in FP32)_
32 steps 87 _._ 64 5 _._ 51 1 _._ 55 _±_ 0 _._ 01 21 _[′]_ 120 _._ 99 _±_ 83 _._ 59
64 steps 70 _._ 47 5 _._ 48 3 _._ 00 _±_ 0 _._ 01 10 _[′]_ 914 _._ 91 _±_ 41 _._ 69
128 steps 62 _._ 66 5 _._ 46 5 _._ 86 _±_ 0 _._ 02 5 _[′]_ 585 _._ 57 _±_ 24 _._ 49
256 steps 59 _._ 38 5 _._ 45 11 _._ 64 _±_ 0 _._ 03 2 _[′]_ 814 _._ 99 _±_ 9 _._ 33
512 steps 57 _._ 57 5 _._ 44 22 _._ 98 _±_ 0 _._ 02 1 _[′]_ 425 _._ 89 _±_ 1 _._ 61
1024 steps 56 _._ 12 5 _._ 44 45 _._ 84 _±_ 0 _._ 03 714 _._ 71 _±_ 0 _._ 50


23


Table 8: Sample quality and efficiency on ImageNet for different numbers of sampling steps using
the _Confidence-based_ sampler. We generate images in batches of 32 to measure throughput, and use
a batch size of 1 to measure latency. Throughput is lower with CFG because each step requires two
forward passes (conditional and unconditional). The throughput and latency are averaged over 10
batches.


**Model** **FID** _↓_ **IS** _↑_ **Latency** _↓_ **Throughput** _↑_
**(ms)** **(img/s)**


_MaskGIT (32 steps; 458M)_
w = 0 14.30 82.41 0.70 2.05
w = 1 7.80 151.62 1.21 1.05
w = 2 **6.78** 208.92 1.21 1.05
w = 3 7.37 255.69 1.21 1.05
w = 4 7.46 289.93 1.21 1.05
w = 5 9.61 250.86 1.21 1.05
w = 6 21.68 149.61 1.21 1.05


_MaskGIT (64 steps; 458M)_
w = 0 15.62 79.23 1.39 1.03
w = 1 9.40 140.48 2.41 0.52
w = 2 8.06 195.35 2.41 0.52
w = 3 8.19 239.89 2.41 0.52
w = 4 **7.61** 267.26 2.41 0.52
w = 5 11.44 202.92 2.41 0.52
w = 6 26.41 113.63 2.41 0.52


_PGM 12 / 12 (32 steps; 464M)_
w = 0 18.77 67.22 1.04 6.86
w = 1 8.96 135.03 1.08 3.76
w = 2 **6.67** 201.66 1.08 3.76
w = 3 7.09 255.43 1.08 3.76
w = 4 8.30 290.18 1.08 3.76
w = 5 9.59 307.52 1.08 3.76
w = 6 10.84 313.27 1.08 3.76


_PGM 12 / 12 (64 steps; 464M)_
w = 0 19.45 64.44 2.04 3.52
w = 1 10.08 124.90 2.04 1.90
w = 2 **7.35** 188.77 2.04 1.90
w = 3 7.39 238.31 2.04 1.90
w = 4 8.13 276.55 2.04 1.90
w = 5 9.18 297.44 2.04 1.90
w = 6 10.38 302.29 2.04 1.90


_PGM 14 / 10 (32 steps; 464M)_
w = 0 21.97 60.07 1.04 7.09
w = 1 11.24 121.39 1.04 3.90
w = 2 8.05 183.20 1.04 3.90
w = 3 **7.76** 232.62 1.04 3.90
w = 4 8.47 263.73 1.04 3.90
w = 5 9.39 288.60 1.04 3.90
w = 6 10.40 291.46 1.04 3.90


_PGM 14 / 10 (64 steps; 464M)_
w = 0 22.74 56.77 2.03 3.63
w = 1 12.32 112.98 2.04 1.97
w = 2 8.80 171.11 2.04 1.97
w = 3 **8.11** 219.33 2.04 1.97
w = 4 8.44 253.10 2.04 1.97
w = 5 9.12 270.16 2.04 1.97
w = 6 9.90 279.00 2.04 1.97


24


Table 9: Sample quality and efficiency on ImageNet for different numbers of sampling steps using
the _Halton_ sampler. We generate images in batches of 32 to measure throughput, and use a batch
size of 1 to measure latency. Throughput is lower with CFG because each step requires two forward
passes (conditional and unconditional). The throughput and latency are averaged over 10 batches.


**Model** **FID** _↓_ **IS** _↑_ **Latency** _↓_ **Throughput** _↑_
**(ms)** **(img/s)**


_MaskGIT (32 steps; 458M)_
w = 0 25.72 57.70 0.70 2.02
w = 1 **5.35** 267.49 1.21 1.05
w = 2 12.82 365.36 1.21 1.05
w = 3 17.24 **408.30** 1.21 1.05
w = 4 15.65 365.97 1.21 1.05
w = 5 25.33 182.14 1.21 1.05
w = 6 48.97 74.74 1.21 1.05


_MaskGIT (64 steps; 458M)_
w = 0 18.61 69.68 1.39 1.03
w = 1 **6.76** 283.96 2.41 0.52
w = 2 14.97 372.74 2.41 0.52
w = 3 18.30 **410.60** 2.41 0.52
w = 4 16.18 312.75 2.41 0.52
w = 5 32.69 126.69 2.41 0.52
w = 6 60.12 51.40 2.41 0.52


_PGM 12 / 12 (32 steps; 464M)_
w = 0 22.59 66.32 1.03 13.44
w = 1 10.21 134.22 1.03 7.93
w = 2 6.04 203.28 1.03 7.93
w = 3 **5.54** 263.53 1.03 7.93
w = 4 6.38 311.53 1.03 7.93
w = 5 7.58 345.18 1.03 7.93
w = 6 8.83 **372.07** 1.03 7.93


_PGM 12 / 12 (64 steps; 464M)_
w = 0 16.19 79.26 1.96 7.17
w = 1 6.67 151.44 2.01 4.12
w = 2 **4.56** 218.98 2.01 4.12
w = 3 4.98 276.16 2.01 4.12
w = 4 6.47 322.53 2.01 4.12
w = 5 7.95 352.56 2.01 4.12
w = 6 9.47 **379.39** 2.01 4.12


_PGM 14 / 10 (32 steps; 464M)_
w = 0 25.42 62.11 1.01 12.70
w = 1 11.57 128.22 1.00 7.37
w = 2 6.60 196.56 1.00 7.37
w = 3 **5.55** 253.76 1.00 7.37
w = 4 6.00 301.66 1.00 7.37
w = 5 7.04 334.29 1.00 7.37
w = 6 8.16 **365.56** 1.00 7.37


_PGM 14 / 10 (64 steps; 464M)_
w = 0 18.03 75.40 1.93 6.69
w = 1 7.77 144.26 1.99 3.80
w = 2 **4.76** 212.39 1.99 3.80
w = 3 4.85 268.55 1.99 3.80
w = 4 5.88 309.88 1.99 3.80
w = 5 7.23 342.40 1.99 3.80
w = 6 8.63 **366.57** 1.99 3.80


25


Table 10: Throughput (TP) of MDLM and PGM with a context length of 4096, for varying number
of inference steps. PGM is significantly faster than MDLM.


**Model** **TP (4096)** **TP (1024)** **TP (256)** **TP (64)**

MDLM 30 _._ 45 _±_ 0 _._ 06 121 _._ 25 _±_ 0 _._ 02 483 _._ 53 _±_ 0 _._ 25 1 _[′]_ 912 _._ 16 _±_ 1 _._ 44
PGM 8/8 128 _._ 99 _±_ 0 _._ 23 697 _._ 36 _±_ 32 _._ 83 **2’216.91** _±_ 3 _._ 06 **8’203.82** _±_ 6 _._ 60
PGM 6/6 (dim=1024) **129.01** _±_ 0 _._ 67 **706.65** _±_ 36 _._ 23 2 _[′]_ 146 _._ 60 _±_ 15 _._ 12 8 _[′]_ 175 _._ 69 _±_ 7 _._ 85


26
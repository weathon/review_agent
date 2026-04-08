# AREUREDI: ANNEALED RECTIFIED UPDATES FOR REFINING DISCRETE FLOWS WITH MULTI-OBJECTIVE GUIDANCE


**Anonymous authors**
Paper under double-blind review


ABSTRACT

Designing sequences that satisfy multiple, often conflicting, objectives is a central challenge in therapeutic and biomolecular engineering. Existing generative
frameworks largely operate in continuous spaces with single-objective guidance,
while discrete approaches lack guarantees for multi-objective Pareto optimality. We
introduce **AReUReDi** ( **A** nnealed **Re** ctified **U** pdates for **Re** fining **Di** screte Flows),
a discrete optimization algorithm with theoretical guarantees of convergence to the
Pareto front. Building on Rectified Discrete Flows (ReDi), AReUReDi combines
Tchebycheff scalarization, locally balanced proposals, and annealed MetropolisHastings updates to bias sampling toward Pareto-optimal states while preserving distributional invariance. Applied to peptide and SMILES sequence design,
AReUReDi simultaneously optimizes up to five therapeutic properties (including
affinity, solubility, hemolysis, half-life, and non-fouling) and outperforms both
evolutionary and diffusion-based baselines. These results establish AReUReDi as
a powerful, sequence-based framework for multi-property biomolecule generation.


1 INTRODUCTION


The design of biological sequences must account for multiple, often conflicting, objectives (Naseri
& Koffas, 2020). Therapeutic molecules, for example, must combine high binding affinity with
low immunogenicity and favorable pharmacokinetics (Tominaga et al., 2024); CRISPR guide RNAs
require both high on-target activity and minimal off-target effects (Mohr et al., 2016; Schmidt
et al., 2025); and synthetic promoters must deliver strong expression while remaining tissue-specific
(Artemyev et al., 2024). These examples illustrate that biomolecular engineering is inherently a
multi-objective optimization problem.


Yet, most computational frameworks continue to optimize single objectives in isolation (Zhou et al.,
2019; Nehdi et al., 2020; Nisonoff et al., 2025). While such approaches can reduce toxicity (Kreiser
et al., 2020; Sharma et al., 2022) or improve thermostability (Komp et al., 2025), they often create
adverse trade-offs: high-affinity peptides may be insoluble or hemolytic, and stabilized proteins may
lose specificity (Bigi et al., 2023; Rinauro et al., 2024). Black-box multi-objective optimization
(MOO) methods such as evolutionary search and Bayesian optimization have long been applied to
molecular design (Zitzler & Thiele, 1998; Deb, 2011; Ueno et al., 2016; Frisby & Langmead, 2021),
but these approaches scale poorly in high-dimensional sequence spaces.


To overcome this, recent generative approaches have incorporated controllable multi-objective
sampling (Li et al., 2018; Sousa et al., 2021; Yao et al., 2024). For instance, ParetoFlow (Yuan
et al., 2024) leverages continuous-space flow matching to generate Pareto-optimal samples. However,
extending such guarantees to biological sequences is challenging, since discrete data typically require
embedding into continuous manifolds, which distorts token-level structure and complicates propertybased guidance (Beliakov & Lim, 2007; Michael et al., 2024).


A more direct path lies in discrete flow models (Campbell et al., 2024; Gat et al., 2024; Dunn &
Koes, 2024). These models define probability paths over categorical state spaces, either through
simplex-based interpolations (Stark et al., 2024; Davis et al., 2024; Tang et al., 2025a) or jump-process
flows that learn token-level transition rates (Campbell et al., 2024; Gat et al., 2024). Recent advances
have shown their promise for controllable single-objective generation (Nisonoff et al., 2025; Tang
et al., 2025a), but no framework yet achieves Pareto guidance across multiple objectives.


1


Here, the notion of rectification provides a crucial building block. In the continuous setting, _Rectified_
_Flows_ (Liu et al., 2023) learn to straighten ODE paths between distributions, thereby reducing convex
transport costs and enabling efficient few-step or even one-step sampling. Recently, **ReDi** ( _Rectified_
_Discrete Flows_ ) (Yoo et al., 2025) extended this principle to discrete domains. By iteratively refining
the coupling between source and target distributions, ReDi provably reduces factorization error
(quantified as conditional total correlation) while maintaining distributional fidelity. This makes
ReDi highly effective for efficient discrete sequence generation. However, ReDi does not address
the multi-objective setting, as it lacks a mechanism to steer sampling toward the _Pareto front_, where
improvements in one objective cannot be made without degrading another. This is a critical limitation
for biomolecular design, where trade-offs define practical success.


To address this, we introduce **AReUReDi** ( **A** nnealed **Re** ctified **U** pdates for **Re** fining **Di** screte Flows),
a new framework that extends rectified discrete flows with multi-objective guidance. AReUReDi
integrates three innovations: (i) _annealed Tchebycheff scalarization_, which gradually sharpens the
focus on balanced solutions across objectives (Lin et al., 2024a); (ii) _locally balanced proposals_, which
combine the generative prior of ReDi with multi-objective guidance while ensuring reversibility;
and (iii) _Metropolis-Hastings updates_, which preserve exact distributional invariance and guarantee
convergence to Pareto-optimal states. Together, these mechanisms refine rectified discrete flows into
a principled Pareto sampler.


Our key contributions are:


1. We propose AReUReDi, the first multi-objective extension of rectified discrete flows, integrating
annealed scalarization, locally balanced proposals, and MCMC updates.


2. We provide theoretical guarantees that AReUReDi preserves distributional invariance and
converges to the Pareto front with full coverage.


3. We demonstrate that AReUReDi can optimize up to five competing biological properties
simultaneously, including affinity, solubility, hemolysis, half-life, and non-fouling.


4. We benchmark AReUReDi against classical MOO algorithms and state-of-the-art discrete
diffusion approaches, showing superior trade-off navigation and biologically plausible sequence
designs.


2 PRELIMINARIES


2.1 DISCRETE FLOW MATCHING


Let _S_ = _V_ _[L]_ denote the discrete state space, where _V_ is a vocabulary of size _K_ and each _x_ =
( _x_ 1 _, . . ., xL_ ) _∈S_ is a sequence of tokens. A _discrete flow matching (DFM)_ model (Campbell et al.,
2024; Gat et al., 2024; Dunn & Koes, 2024) defines a probability path _{pt}t∈_ [0 _,_ 1] interpolating
between a simple source distribution _p_ 0 and a target distribution _p_ 1 by means of a coupling _π_ ( _x_ 0 _, x_ 1)
and conditional bridge distributions _pt_ ( _xt_ _| x_ 0 _, x_ 1). The model is trained to approximate conditional
transitions _ps|t_ ( _xs_ _| xt_ ) for 0 _≤_ _t < s ≤_ 1.


Since the joint distribution over _L_ coordinates is intractable, DFMs employ a factorization


This quantity captures the inter-dimensional dependencies neglected under factorization, and grows
with larger step sizes (Stark et al., 2024; Davis et al., 2024; Tang et al., 2025a). As a result, DFMs are
accurate in the many-step regime but degrade under few-step or one-step generation.


2


_ps|t_ ( _xs_ _| xt_ ) _≈_


_L_

- _ps|t_ - _x_ _[i]_ _s_ _[|][ x][t]_ - _,_

_i_ =1


which introduces a discrepancy measured by the conditional total correlation


- _ps|t_ ( _x_ _[i]_ _s_ _[|][ x][t]_ [)]

_i_ =1


_._


�����


_L_


TC _s|t_ = KL


_ps|t_ ( _xs_ _| xt_ )


2.2 RECTIFIED DISCRETE FLOW


To mitigate factorization error, **Rectified** **Discrete** **Flow** **(ReDi)** (Yoo et al., 2025) introduces an
iterative rectification of the coupling _π_ . Starting from an initial coupling _π_ [(0)] ( _x_ 0 _, x_ 1), a DFM is
trained under _π_ [(] _[k]_ [)] to produce new source–target pairs, defining an empirical joint distribution _π_ ˆ [(] _[k]_ [)] .
The coupling is then updated via

_[|][ x]_ [0][)]
_π_ [(] _[k]_ [+1)] ( _x_ 0 _, x_ 1) _∝_ _π_ [(] _[k]_ [)] ( _x_ 0 _, x_ 1) _[p][θ]_ [(] _[k]_ [)][(] _[x]_ [1] _,_

_pθ_ ( _k_ )( _x_ 1)

where _pθ_ ( _k_ )( _x_ 1 _| x_ 0) is the conditional distribution learned at iteration _k_ . This yields a sequence of
couplings _{π_ [(] _[k]_ [)] _}k≥_ 0 with provably decreasing conditional TC,

TC _s|t_ ( _π_ [(] _[k]_ [+1)] ) _≤_ TC _s|t_ ( _π_ [(] _[k]_ [)] ) _._
By progressively reducing factorization error, ReDi produces a well-calibrated base distribution _p_ 1
with low inter-dimensional correlation. This base distribution provides reliable marginal transition
probabilities _p_ _[i]_ _t_ [(] _[·]_ _[|]_ _[x][t]_ [)] [for] [each] [coordinate] _[i]_ [at] [time] _[t]_ [,] [which] [serve] [as] [the] [generative] [prior] [in] [the]
AReUReDi framework. Rectification follows the same principle as _Rectified Flow_ in continuous
domains (Liu et al., 2023), where iterative refinement straightens ODE paths and decreases transport
costs.


3 AREUREDI: ANNEALED RECTIFIED UPDATES FOR REFINING DISCRETE
FLOWS


With an efficient discrete flow-based generation framework in hand, we develop AReUReDi that
extends ReDi (Yoo et al., 2025) to the multi-objective optimization setting, where the goal is
to generate discrete samples that approximate the Pareto front of multiple competing objectives.
Starting from a pre-trained ReDi model, AReUReDi incorporates annealed guidance, locally balanced
proposals, and Metropolis-Hastings updates to progressively bias the sampling process toward Paretooptimal states while preserving the probabilistic guarantees of the underlying flow (Algorithm 1).


3.1 PROBLEM SETUP


Let the discrete search space be _S_ = _V_ _[L]_, where _V_ is a finite vocabulary of size _K_ and each state
_x_ = ( _x_ 1 _, . . ., xL_ ) _∈S_ is a sequence of tokens. We assume access to a pre-trained ReDi model that
provides marginal transition probabilities _p_ _[i]_ _t_ [(] _[· |][ x][t]_ [)][ for each position] _[ i]_ [ and time] _[ t]_ [.] [In addition, we are]
given _N_ pre-trained scalar objective functions _sn_ : _S_ _→_ R, where _n_ = 1 _, . . ., N_, and ˜ _sn_ ( _x_ ) are their
normalized counterparts with outputs mapped to [0 _,_ 1] to support balanced updates for each objective.
The sampling task is to construct a Markov chain whose stationary distribution concentrates on states
that approximate the Pareto front of the normalized objectives ˜ _s_ 1 _, . . .,_ ˜ _sN_ .


3.2 ANNEALED MULTI-OBJECTIVE GUIDANCE


To direct sampling toward the Pareto front, AReUReDi introduces a scalarized reward
_Sω_ ( _x_ ) = min
1 _≤n≤N_ _[ω][n]_ [ ˜] _[s][n]_ [(] _[x]_ [)] _[,]_

where the weight vector _ω_ = [ _ω_ 1 _, . . ., ωN_ ] lies in the probability simplex ∆ _[N]_ _[−]_ [1] and balances the
different objectives. This Tchebycheff scalarization promotes solutions that are simultaneously strong
across all objectives rather than excelling in only a subset (Miettinen, 1999). The scalarized reward is
converted into a guidance weight
_Wηt,ω_ ( _x_ ) = exp               - _ηtSω_ ( _x_ )� _,_
where the parameter _ηt_ _>_ 0 controls the strength of the guidance at each iteration _t_ . AReUReDi
incorporates an annealing schedule for _ηt_ :

_ηt_ = _η_ min +                - _η_ max _−_ _η_ min� _t_
_T_ _−_ 1 _[,]_

so that the chain begins with a small value of _ηt_ to encourage wide exploration of the state space and
gradually increases _ηt_ to focus sampling on high-quality Pareto candidates. This annealing strategy
mirrors simulated annealing but operates directly on the scalarized objectives within the discrete flow
framework.


3


3.3 LOCALLY BALANCED PROPOSALS


Given the current state _xt_, AReUReDi updates one coordinate _i_ _∈{_ 1 _, . . ., L}_ at a time using a
locally balanced proposal that blends the generative prior of ReDi with the multi-objective guidance.
First, a candidate set of replacement tokens is drawn from the ReDi marginal _p_ _[i]_ _t_ [(] _[· |][ x][t]_ [)][, optionally]
pruned using top-p to retain only the most promising alternatives for computational efficiency. For
each candidate token _y_, the algorithm computes the ratio

_ri_ ( _y_ ; _xt_ ) = _[W][η][t][,ω]_                    - _x_ [(] _t_ _[i][←][y]_ [)]                    - _,_

_Wηt,ω_ ( _xt_ )


which measures the change in scalarized reward if _x_ _[i]_ _t_ [were] [replaced] [by] _[y]_ [.] [The] [ratio] _[r][i]_ [(] _[y]_ [;] _[ x][t]_ [)] [is]
then transformed by a balancing function _g_ : R+ _→_ R+ that satisfies the symmetry condition
_g_ ( _u_ ) = _u g_ (1 _/u_ ). Typical choices include Barker’s function _g_ ( _u_ ) = 1+ _uu_ [and] [the] [square-root]
function _g_ ( _u_ ) = _[√]_ _u_ . This symmetry ensures that the resulting Markov chain admits the desired
stationary distribution. Using the balanced function, the unnormalized proposal for a candidate token
_y_ takes the form
_q_ ˜ _i_ ( _y_ _| xt_ ) = _p_ _[i]_ _t_ [(] _[y]_ _[|][ x][t]_ [)] _[ g]_               - _ri_ ( _y_ ; _xt_ )� _,_

which is then normalized over the candidate set to yield the final proposal distribution _qi_ ( _y_ _|_ _xt_ ).
This construction allows the proposal to favor states with higher scalarized reward while remaining
reversible with respect to the target distribution.


3.4 METROPOLIS-HASTINGS UPDATE


A candidate token _y_ _[⋆]_ is drawn from the final proposal distribution _qi_ ( _· | xt_ ) and forms the proposed
state _x_ prop = _x_ [(] _t_ _[i][←][y][⋆]_ [)] . The proposal is accepted with the standard Metropolis-Hastings probability
(Hastings, 1970)


where we define _πηt,ω_ ( _x_ ) _∝_ _p_ 1( _x_ ) _Wηt,ω_ ( _x_ ) = _p_ 1( _x_ ) exp� _ηtSω_ ( _x_ )�. With Barker’s balancing
function, the acceptance probability simplifies to one, ensuring automatic acceptance of proposals
and faster mixing. Other choices, such as the square-root function, trade higher acceptance rates for
more conservative moves.


The annealed, locally balanced updates are repeated for _T_ iterations and end with the final sample
_x_ 1 whose objective scores are jointly optimized. Building on the ReDi model’s well-calibrated base
distribution with low inter-dimensional correlation, AReUReDi safely biases this base toward Paretooptimal regions while preserving full coverage of the state space, thereby guaranteeing convergence
to Pareto-optimal solutions with complete coverage of the Pareto front.


4 EXPERIMENTS


To the best of our knowledge, no public datasets exist for benchmarking multi-objective optimization
algorithms on biological sequences. We therefore developed two benchmarks to evaluate AReUReDi,
focusing on the generation of wild-type peptide sequences and chemically-modified peptide SMILES.
These tasks are supported by two core components: the generative models described in Appendix B
and the objective-scoring models validated in Appendix E. Leveraging these models, we demonstrate
AReUReDi’s efficacy on a wide range of tasks and examples.


Although AReUReDi provides theoretical guarantees of Pareto optimality and full coverage, in
practice, these guarantees hold only in the limit of an infinitely long Markov chain. Reaching
the Pareto front with high probability can therefore require a vast number of sampling steps. To
improve sampling efficiency in all reported experiments, we introduce a monotonicity constraint that
accepts only token updates that increase the weighted sum of the current objective scores. Empirical
results prove the accelerated convergence toward high-quality Pareto solutions without altering the
underlying optimization objectives (Table 6). Therefore, this monotonicity constraint was involved in
all the following experiments.


4


      _t_ _[|][ x]_ [prop][)]
_αi_ ( _xt, x_ prop) = min 1 _,_ _[π][η][t][,ω]_ [(] _[x]_ [prop][)] _[ q][i]_ [(] _[x][i]_
_πηt,ω_ ( _xt_ ) _qi_ ( _y_ _[⋆]_ _| xt_ )


_,_


Figure 1: **(A), (B)** Complex structures of PDB 1B8Q with an AReUReDi-designed binder and its pre-existing
binder. **(C),** **(D)** Complex structures of OX1R and EWS::FLI1 with an AReUReDi-designed binder. Five
property scores are shown for each binder, along with the ipTM score from AlphaFold3 and docking score from
AutoDock VINA. Interacting residues on the target are visualized. **(E)** Plots showing the mean scores for each
property across the number of iterations during AReUReDi’s design of binders of length 12-aa for EWS::FLI1.
**(F)** A density plot illustrating the distribution of predicted property scores for AReUReDi-designed EWS::FLI1
binders of length 12-aa, compared to the peptides generated unconditionally by PepReDi [3] .


4.1 AREUREDI EFFECTIVELY BALANCES EACH OBJECTIVE TRADE-OFF


With pre-trained PepReDi in hand, we first focus on validating AReUReDi’s capability of balancing
multiple conflicting objectives. We performed two sets of experiments for wild-type peptide binder
generation with three property guidance, and in ablation experiment settings, we removed one or more
objectives. In the binder design task for target 7LUL (hemolysis, solubility, affinity guidance; Table 7),
omitting any single guidance causes a collapse in that property, while the remaining guided metrics
may modestly improve. Likewise, in the binder design task for target CLK1 (affinity, non-fouling,
half-life guidance; Table 8), disabling non-fouling guidance allows half-life to exceed 96 hours but
drives non-fouling near zero, and disabling half-life guidance preserves non-fouling yet reduces
half-life below 2 hours. In contrast, enabling all guidance signals produces the most balanced profiles
across all objectives. These results confirm that AReUReDi precisely targets chosen objectives while
preserving the flexibility to navigate conflicting objectives and push samples toward the Pareto front.


4.2 AREUREDI GENERATES WILD-TYPE PEPTIDE BINDERS UNDER FIVE PROPERTY GUIDANCE


We next benchmark AReUReDi on a wild-type peptide binder generation task guided by five different
properties that are critical for therapeutic discovery: hemolysis, non-fouling, solubility, half-life,


5


Table 1: AReUReDi generates wild-type peptide binders for 8 diverse protein targets, optimizing five therapeutic
properties: hemolysis, non-fouling, solubility, half-life (in hours), and binding affinity. Each value represents the
average of 100 AReUReDi-designed binders.


**Name** **Binder Length** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


AMHR2 8 0.9156 0.8613 0.8564 45.73 7.0608

AMHR2 12 0.9384 0.8872 0.8810 52.52 7.2284

AMHR2 16 0.9420 0.8914 0.8755 63.34 7.2533

EWS::FLI1 8 0.9186 0.8630 0.8619 44.77 5.8424

EWS::FLI1 12 0.9345 0.8819 0.8796 59.11 6.2007

EWS::FLI1 16 0.9416 0.8875 0.8807 64.32 6.4195

MYC 8 0.9180 0.8627 0.8627 44.13 6.4082

OX1R 10 0.9302 0.8687 0.8563 50.14 7.1882

DUSP12 9 0.9240 0.8669 0.8633 48.14 6.1276

1B8Q 8 0.9214 0.8680 0.8654 42.63 5.7130

5AZ8 11 0.9293 0.8732 0.8605 58.33 6.2792

7JVS 11 0.9313 0.8840 0.8743 56.49 6.8449


and binding affinity. To evaluate AReUReDi in a controlled setting, we designed 100 peptide
binders per target for 8 diverse proteins, structured targets with known binders (3IDJ, 5AZ8, 7JVS),
structured targets without known binders (AMHR2, OX1R, DUSP12), and intrinsically disordered
targets (EWS::FLI1, MYC) (Table 1). Across all targets and across multiple binder lengths, the
generated peptides achieve superior hemolysis rates (0.91-0.94), high non-fouling ( _>_ 0.86) and
solubility ( _>_ 0.85), extended half-life (42-64 h), and strong affinity scores (5.7-7.3), demonstrating
both balanced optimization and robustness to sequence length.


For the target proteins with pre-existing binders, we compared the property values between their
known binders with AReUReDi-designed ones (Figure 1A,B, A1). The designed binders significantly
outperform the pre-existing binders across all properties without compromising the binding potential,
which is further confirmed by the ipTM scores computed by AlphaFold3 (Abramson et al., 2024)
and docking scores calculated by AutoDock VINA (Trott & Olson, 2010). Although the AReUReDidesigned binders bind to similar target positions as the pre-existing ones, they differ significantly in
sequence and structure, demonstrating AReUReDi’s capacity to explore the vast sequence space for
optimal designs. For target proteins without known binders, complex structures were visualized using
one of the AReUReDi-designed binders (Figure A2). The corresponding property scores, as well as
ipTM and docking scores, are also displayed. Some of the designed binders showed longer half-life,
while others excelled in non-fouling and solubility, underscoring the comprehensive exploration of
the sequence space by AReUReDi.


To evaluate our guided generation strategy, we tracked the mean and standard deviation of five
property scores across 100 generated binders (length 12) targeting EWS::FLI1 at each iteration
(Figure 1E). All five properties steadily improved, with average scores for solubility and non-fouling
properties increasing markedly from 0.4 to 0.9. [˜] The large standard deviation observed in the final
half-life and binding affinity values reflects this property’s high sensitivity to guidance, as AReUReDi
balances the trade-offs between multiple conflicting objectives. We further visualized AReUReDi’s
impact by comparing the property distribution of the 100 guided peptides to that of 100 peptides
unconditionally sampled from PepReDi [3] . The results show that AReUReDi effectively shifted the
distribution towards peptides with higher binding affinity. Collectively, these findings demonstrate
AReUReDi’s capability to steer generation toward simultaneous multi-property optimization.


We benchmarked AReUReDi against four established multi-objective optimization (MOO) baselines
(NSGA-III (Deb & Jain, 2013), SMS-EMOA (Beume et al., 2007), SPEA2 (Zitzler et al., 2001),
and MOPSO (Coello & Lechuga, 2002)) on two protein targets: 1B8Q, a small protein with known
peptide binders (Zhang et al., 1999), and PPP5, a larger protein without characterized binders
(Yang et al., 2004) (Table 2). Each method generated 100 candidate binders optimized for five
properties: hemolysis, non-fouling, solubility, half-life, and binding affinity. While AReUReDi
required longer runtimes than evolutionary baselines, it consistently produced the best trade-offs. For
both targets, it designed targets with top hemolysis scores, increased non-fouling and solubility by


6


Table 2: AReUReDi outperforms traditional multi-objective optimization algorithms in designing wild-type
peptide binders guided by five objectives. Each value represents the average of 100 designed binders. The table
also records the average runtime for each algorithm to design a single binder. The best result for each metric is
highlighted in bold.


**Target** **Method** **Time (s)** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


30-50%, maintained competitive binding affinity, and even extended the half-life by a factor of 3-13
relative to the next-best method. These results underscore AReUReDi’s effectiveness in navigating
high-dimensional property landscapes to yield peptide binders with balanced, optimized profiles.


We also compared against PepTune (Tang et al., 2025b), a recent masked discrete diffusion model
for peptide design that couples generation with Monte Carlo Tree Search for MOO. PepTune’s
backbone was adapted to the existing DPLM model (Wang et al., 2024) for wild-type peptide
sequence generation. Despite longer runtimes, AReUReDi substantially outperformed PepTune
across all objectives, yielding nearly threefold improvements in non-fouling and solubility and a
22-fold increase in half-life. Together, these comparisons demonstrate that AReUReDi surpasses not
only traditional MOO algorithms but also the current state-of-the-art diffusion-based approach for
multi-objective-guided wild-type peptide binder design.


Since AReUReDi requires more computation than PepTune to design the same number of binders, we
compare both methods under a matched wall-clock budget (Table 11). Specifically, the time PepTune
needs to generate 100 binders approximately matches the time AReUReDi needs to generate four 8mer binders for 1B8Q and three 16-mer binders for PPP5. For both tasks, the top-2 AReUReDi binders
achieve substantially higher non-fouling, solubility, and half-life, while maintaining comparable
hemolysis and affinity. This comparison shows that AReUReDi produces better multi-objective
trade-offs, even when PepTune is allowed a much larger candidate pool under the same time budget.


4.3 AREUREDI GENERATES THERAPEUTIC PEPTIDE SMILES UNDER FOUR PROPERTY
GUIDANCE


To demonstrate the broad applicability of AReUReDi for multi-objective guided generation of
biological sequences, we employed the rectified SMILESReDi model to design chemically-modified
peptide binder SMILES sequences for five diverse therapeutic targets. These included the metabolic
hormone receptor Glucagon-like peptide-1 receptor (GLP1), the iron transport protein Transferrin
receptor (TfR), the Neural Cell Adhesion Molecule 1 (NCAM1), the neurotransmitter transporter
GLAST, and the developmental Anti-Müllerian Hormone Receptor Type 2 (AMHR2). For each
target, sequence generation was jointly conditioned on a predicted binding-affinity score to the target
protein, as long as hemolysis, solubility, and non-fouling, to ensure both potency and desirable
physicochemical profiles. Although PepTune is also able to perform multi-property guided design
of peptide-binder SMILES sequences, it does not report average property scores for its generated
binders, making a direct quantitative comparison with AReUReDi infeasible (Tang et al., 2025b).


We selected and visualized representative binders with the highest predicted binding affinities for
each target (Figure 2A, A3A,C, A4A,C). All selected binders achieved high scores across hemolysis,


7


1B8Q


PPP5


MOPSO 8.54 0.8934 0.4763 0.4684 4.45 6.0594

NSGA-III 33.13 0.9138 0.5715 0.5825 7.32 7.2178

SMS-EMOA 8.21 0.8804 0.3450 0.3511 3.02 5.955

SPEA2 17.48 0.9181 0.4973 0.5057 4.13 **7.3240**

PepTune + DPLM **2.46** 0.8547 0.3085 0.3213 1.17 5.2398

**AReUReDi** 55 **0.9214** **0.8680** **0.8654** **22.93** 5.7130


MOPSO 11.34 0.9117 0.4711 0.4255 1.77 6.6958

NSGA-III 37.30 **0.9521** 0.7138 0.7066 2.90 7.3789

SMS-EMOA 8.43 0.8758 0.4269 0.4334 1.03 6.2854

SPEA2 19.02 0.9445 0.6221 0.6098 2.61 **7.6253**

PepTune + DPLM **4.80** 0.8816 0.2752 0.2636 1.27 5.8454

**AReUReDi** 195 0.9412 **0.896** **0.8832** **38.28** 6.7186


Figure 2: **(A)** Example 2D SMILES structure of AReUReDi-designed peptide binders with four property scores.
**(B)** Plots showing the mean scores for each property across the number of iterations during AReUReDi’s design
of binders of length 200 for NCAM1.


solubility, non-fouling, and binding affinity. During generation, we recorded the mean and standard
deviation of all four property scores over 100 binders at each iteration to assess the effectiveness
of the multi-objective guidance (Figure 2B, A3B,D, A4B,D). Across all targets, binding affinity
scores and non-fouling scores showed steady upward trends throughout the generation process,
while hemolysis and solubility scores fluctuated, indicating AReUReDi’s effort to balance the four
conflicting objectives. Moreover, AReUReDi produces valid sequences with substantially higher
diversity and lower SNN than PepTune, indicating both superior novelty and structural variability
(Table 5). These findings highlight the versatility and reliability of AReUReDi for the _de novo_ design
of chemically modified peptide binders across a wide range of therapeutic targets.


4.4 ABLATION STUDIES FOR RECTIFICATION AND ANNEALED GUIDANCE STRENGTH


To determine if rectification offers an advantage over standard discrete flow matching, we compared
the performance of AReUReDi using three generative models: the base PepReDi model (no rectification), PepReDi (three rounds of rectification), and PepDFM, a standard discrete flow model that
follows Gat et al. (2024) and was trained on the same data (Appendix C.3). Under the three settings,
wild-type binders were designed for two distinct protein targets: 5AZ8 and AMHR2 (Table 9). For
the AMHR2 target, the rectified model achieved the highest scores across all five properties, with
its predicted half-life surpassing the next-best method by nearly 13 hours. For the 5AZ8 target, the
rectified model yielded a significantly higher half-life while maintaining comparable performance
on other metrics. These results indicate that by lowering conditional TC and improving the quality
of the probability path, rectification enables AReUReDi to achieve stronger Pareto trade-offs on the
more demanding objectives.


We further demonstrated the advantage of using an annealed guidance strength (Table 10). AReUReDi
was applied to design wild-type peptide binders for two distinct proteins: a structured protein with
known binders (PDB 1DDV) and an intrinsically disordered protein without known binders (P53).
Across both targets, any fixed guidance strength, whether set to _η_ min, _η_ max, or their midpoint, failed to
match the performance achieved with an annealed schedule. For 1DDV, annealing produced binders
with markedly higher half-life and the best solubility, while maintaining hemolysis, non-fouling,
and affinity scores that meet or exceed those of all fixed- _η_ settings. A similar trend holds for P53,
where the annealing schedule consistently delivers the strongest results across all objectives. These
findings confirm that gradually increasing the guidance strength enables AReUReDi to attain more
favorable Pareto trade-offs, enhancing challenging properties such as half-life without sacrificing
other therapeutic metrics.


8


5 RELATED WORKS


**Online** **Multi-Objective** **Optimization.** Recent work in multi-objective guided generation has
focused on online or sequential decision-making, where solutions are refined with new data (Gruver
et al., 2023; Jain et al., 2023; Stanton et al., 2022; Ahmadianshalchi et al., 2024). A common
approach is Bayesian optimization (BO), which builds a surrogate model and proposes evaluations
via acquisition functions (Yu et al., 2020; Shahriari et al., 2015). Multi-objective BO often uses
advanced criteria such as EHVI (Emmerich & Klinkenberg, 2008), information gain (Belakaria et al.,
2021), or scalarization (Knowles, 2006; Zhang & Li, 2007; Paria et al., 2020). While AReUReDi also
employs Tchebycheff scalarization, it operates in an offline setting, where each sequence requires
costly evaluation. This contrasts with the sequential, feedback-driven nature of online methods,
making direct comparison inappropriate.


**Tchebycheff Scalarization.** Tchebycheff scalarization can identify any Pareto-optimal point and
is widely used in multi-objective optimization (Miettinen, 1999). Recent variants include smooth
scalarization for gradient-based algorithms (Lin et al., 2024b) and OMD-TCH for online learning (Liu
et al., 2024). AReUReDi is, to our knowledge, the first to apply Tchebycheff scalarization for offline
generative design of discrete therapeutic sequences. Future work may extend to many-objective
problems or alternative utility functions (Lin et al., 2024a; Tu et al., 2023).


**Diffusion and Flow Matching.** Generative approaches such as ParetoFlow and PGD-MOO adapt
flow matching or diffusion models for multi-objective optimization (Yuan et al., 2024; Annadani et al.,
2025). These operate in continuous or latent spaces, whereas AReUReDi is designed for discrete
token spaces inherent to biological sequences. This domain mismatch precludes direct benchmarking.


**Biomolecule** **Generation.** Offline multi-objective frameworks such as EGD and MUDM have
optimized molecules with multiple properties (Sun et al., 2025; Han et al., 2023), but these emphasize
3D structural representations. By contrast, AReUReDi is sequence-only, operating directly over
amino acids or SMILES, which makes structural methods unsuitable as direct comparators.


6 DISCUSSION


In this work, we have presented **AReUReDi**, a multi-objective optimization framework that extends
rectified discrete flows to generate biomolecular sequences satisfying multiple, often conflicting,
properties. By integrating annealed Tchebycheff scalarization, locally balanced proposals, and
Metropolis-Hastings updates, AReUReDi provides theoretical guarantees of convergence to the Pareto
front while maintaining full coverage of the solution space. Built on high-quality base generators
such as PepReDi and SMILESReDi, the method demonstrates broad applicability across amino acid
sequences and chemically modified peptide SMILES. Superior _in silico_ results establish AReUReDi
as a general, theoretically-grounded tool for multi-property-guided biomolecular sequence design.


While AReUReDi excels in domains like wild-type and chemically-modified peptide designs, future
work will extend to other biological modalities, including DNA, RNA, antibodies, and combinatorial
genotype libraries, where multi-objective trade-offs are central. From a theoretical perspective,
improving AReUReDi’s efficiency while maintaining the Pareto convergence guarantees and incorporating uncertainty-aware or feedback-driven guidance remain key directions to explore. Ultimately,
AReUReDi provides a foundation for designing the next generation of therapeutic molecules that are
not only potent but also explicitly optimized for the diverse properties required for clinical success.


REPRODUCIBILITY STATEMENT


We ensure reproducibility through detailed theoretical, algorithmic, and experimental descriptions
of AReUReDi. The complete procedure is formally described in the main text with proofs of
convergence guarantees, including the rectified discrete flow foundation, annealed Tchebycheff
scalarization, locally balanced proposals, and Metropolis-Hastings updates. Architectures, training
details, and datasets for all base generators (PepReDi, SMILESReDi, and PepDFM) are reported
with quantitative metrics in the Results and Appendix. Hyperparameter settings, annealing schedules,
and sensitivity analyses are provided to facilitate replication, and ablation studies are included to
assess the impact of key design choices. Benchmark comparisons against classical multi-objective


9


optimization baselines and diffusion-based methods are tabulated for reference. All datasets used in
this work (PepNN, BioLip2, PPIRef, peptide property datasets, and peptide SMILES collections) are
publicly available. We will release code, pretrained checkpoints, and sampling scripts for AReUReDi
to enable full reproducibility.


ETHICS STATEMENT


This work develops a general generative modeling framework for multi-objective sequence design,
with demonstrations on peptide and peptide-SMILES generation. All datasets are publicly available
and non-sensitive, consisting of peptide property measurements, protein-peptide interaction sets,
and peptide SMILES representations. No human subjects, patient data, or animal experiments were
involved. Potential risks include the misuse of generative models for harmful molecule design or
the uncontrolled release of potent sequences. To mitigate these risks, we will release code and
pretrained models strictly under a research-only license and provide documentation that emphasizes
safe and responsible use. The anticipated societal benefits, such as improving therapeutic peptide
design, enhancing drug safety profiles, and enabling efficient exploration of biological sequence
space, substantially outweigh these potential risks. We encourage future users of AReUReDi to adopt
similar safeguards when applying the method to other molecular domains.


10


REFERENCES


Osama Abdin, Satra Nim, Han Wen, and Philip M Kim. Pepnn: a deep attention model for the
identification of peptide binding sites. _Communications biology_, 5(1):503, 2022.


Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf
Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure
prediction of biomolecular interactions with alphafold 3. _Nature_, pp. 1–3, 2024.


Alaleh Ahmadianshalchi, Syrine Belakaria, and Janardhan Rao Doppa. Pareto front-diverse batch
multi-objective bayesian optimization. In _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_
_Intelligence_, volume 38, pp. 10784–10794, 2024.


Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A nextgeneration hyperparameter optimization framework. In _International Conference on Knowledge_
_Discovery and Data Mining_, pp. 2623–2631, 2019.


Yashas Annadani, Syrine Belakaria, Stefano Ermon, Stefan Bauer, and Barbara E Engelhardt. Preference-guided diffusion for multi-objective offline optimization. _arXiv_ _preprint_
_arXiv:2503.17299_, 2025.


Valentin Artemyev, Anna Gubaeva, Anastasiia Iu Paremskaia, Amina A Dzhioeva, Andrei Deviatkin,
Sofya G Feoktistova, Olga Mityaeva, and Pavel Yu Volchkov. Synthetic promoters in gene therapy:
Design approaches, features and applications. _Cells_, 13(23):1963, 2024.


Syrine Belakaria, Aryan Deshwal, and Janardhan Rao Doppa. Output space entropy search framework
for multi-objective bayesian optimization. _Journal of artificial intelligence research_, 72:667–715,
2021.


Gleb Beliakov and Kieran F Lim. Challenges of continuous global optimization in molecular structure
prediction. _European journal of operational research_, 181(3):1198–1213, 2007.


Nicola Beume, Boris Naujoks, and Michael Emmerich. Sms-emoa: Multiobjective selection based
on dominated hypervolume. _European journal of operational research_, 181(3):1653–1669, 2007.


Alessandra Bigi, Eva Lombardo, Roberta Cascella, and Cristina Cecchi. The toxicity of protein
aggregates: new insights into the mechanisms, 2023.


Jenny Bostrom, Chingwei V Lee, Lauric Haber, and Germaine Fuh. Improving antibody binding
affinity and specificity for therapeutic development. In _Therapeutic_ _Antibodies:_ _Methods_ _and_
_Protocols_, pp. 353–376. Springer, 2008.


Anton Bushuiev, Roman Bushuiev, Petr Kouba, Anatolii Filkin, Marketa Gabrielova, Michal Gabriel,
Jiri Sedlar, Tomas Pluskal, Jiri Damborsky, Stanislav Mazurenko, et al. Learning to design
protein-protein interactions with enhanced generalization. _arXiv preprint arXiv:2310.18515_, 2023.


Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative
flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In
_Forty-first International Conference on Machine Learning_, 2024. [URL https://openreview.](https://openreview.net/forum?id=kQwSbv0BR4)
[net/forum?id=kQwSbv0BR4.](https://openreview.net/forum?id=kQwSbv0BR4)


Shengfu Chen, Zhiqiang Cao, and Shaoyi Jiang. Ultra-low fouling peptide surfaces derived from
natural amino acids. _Biomaterials_, 30(29):5892–5896, 2009.


CA Coello Coello and Maximino Salazar Lechuga. Mopso: A proposal for multiple objective particle
swarm optimization. In _Proceedings of the 2002 Congress on Evolutionary Computation. CEC’02_
_(Cat. No. 02TH8600)_, volume 2, pp. 1051–1056. IEEE, 2002.


Oscar Davis, Samuel Kessler, Mircea Petrache, Ismail Ceylan, Michael Bronstein, and Joey Bose.
Fisher flow matching for generative modeling over discrete data. _Advances in Neural Information_
_Processing Systems_, 37:139054–139084, 2024.


11


Kalyanmoy Deb. Multi-objective optimisation using evolutionary algorithms: an introduction.
In _Multi-objective_ _evolutionary_ _optimisation_ _for_ _product_ _design_ _and_ _manufacturing_, pp. 3–34.
Springer, 2011.


Kalyanmoy Deb and Himanshu Jain. An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, part i: solving problems with box
constraints. _IEEE transactions on evolutionary computation_, 18(4):577–601, 2013.


Ian Dunn and David Ryan Koes. Exploring discrete flow matching for 3d de novo molecule generation.
_ArXiv_, pp. arXiv–2411, 2024.


Vera D’Aloisio, Paolo Dognini, Gillian A Hutcheon, and Christopher R Coxon. Peptherdia: database
and structural composition analysis of approved peptide therapeutics and diagnostics. _Drug_
_Discovery Today_, 26(6):1409–1419, 2021.


Michael Emmerich and Jan-willem Klinkenberg. The computation of the expected improvement in
dominated hypervolume of pareto front approximations. _Rapport technique, Leiden University_, 34:
7–3, 2008.


Keld Fosgerau and Torsten Hoffmann. Peptide therapeutics: current status and future directions.
_Drug discovery today_, 20(1):122–128, 2015.


Trevor S Frisby and Christopher James Langmead. Bayesian optimization with evolutionary and
structure-based regularization for directed protein evolution. _Algorithms for Molecular Biology_, 16
(1):13, 2021.


Itai Gat, Tal Remez, Neta Shaul, Felix Kreuk, Ricky TQ Chen, Gabriel Synnaeve, Yossi Adi, and
Yaron Lipman. Discrete flow matching. _Advances in Neural Information Processing Systems_, 37:
133345–133385, 2024.


Nate Gruver, Samuel Stanton, Nathan Frey, Tim GJ Rudner, Isidro Hotzel, Julien Lafrance-Vanasse,
Arvind Rajpal, Kyunghyun Cho, and Andrew G Wilson. Protein design with guided discrete
diffusion. _Advances in neural information processing systems_, 36:12489–12517, 2023.


Chakradhar Guntuboina, Adrita Das, Parisa Mollaei, Seongwon Kim, and Amir Barati Farimani.
Peptidebert: A language model based on transformers for peptide property prediction. _The Journal_
_of Physical Chemistry Letters_, 14(46):10427–10434, 2023.


Xu Han, Caihua Shan, Yifei Shen, Can Xu, Han Yang, Xiang Li, and Dongsheng Li. Trainingfree multi-objective diffusion model for 3d molecule generation. In _The Twelfth International_
_Conference on Learning Representations_, 2023.


W Keith Hastings. Monte carlo sampling methods using markov chains and their applications. 1970.


Moksh Jain, Sharath Chandra Raparthy, Alex Hernández-Garcıa, Jarrid Rector-Brooks, Yoshua
Bengio, Santiago Miret, and Emmanuel Bengio. Multi-objective gflownets. In _International_
_conference on machine learning_, pp. 14631–14653. PMLR, 2023.


Shipra Jain, Srijanee Gupta, Sumeet Patiyal, and Gajendra PS Raghava. Thpdb2: compilation of fda
approved therapeutic peptides and proteins. _Drug Discovery Today_, pp. 104047, 2024.


Joshua Knowles. Parego: A hybrid algorithm with on-line landscape approximation for expensive
multiobjective optimization problems. _IEEE transactions on evolutionary computation_, 10(1):
50–66, 2006.


Evan Komp, Christian Phillips, Lauren M Lee, Shayna M Fallin, Humood N Alanzi, Marlo Zorman,
Michelle E McCully, and David AC Beck. Neural network conditioned to produce thermophilic
protein sequences can increase thermal stability. _Scientific Reports_, 15(1):14124, 2025.


Ryan P Kreiser, Aidan K Wright, Natalie R Block, Jared E Hollows, Lam T Nguyen, Kathleen
LeForte, Benedetta Mannini, Michele Vendruscolo, and Ryan Limbocker. Therapeutic strategies
to reduce the toxicity of misfolded protein oligomers. _International journal of molecular sciences_,
21(22):8651, 2020.


12


Yibo Li, Liangren Zhang, and Zhenming Liu. Multi-objective de novo drug design with conditional
graph generative model. _Journal of cheminformatics_, 10:1–24, 2018.


Xi Lin, Yilu Liu, Xiaoyuan Zhang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Few for many:
Tchebycheff set scalarization for many-objective optimization. _arXiv preprint arXiv:2405.19650_,
2024a.


Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang. Smooth
tchebycheff scalarization for multi-objective optimization. _arXiv_ _preprint_ _arXiv:2402.19078_,
2024b.


Zeming Lin, Halil Akin, Roshan Rao, Brian Hie, Zhongkai Zhu, Wenting Lu, Nikita Smetanin,
Robert Verkuil, Ori Kabeli, Yaniv Shmueli, et al. Evolutionary-scale prediction of atomic-level
protein structure with a language model. _Science_, 379(6637):1123–1130, 2023.


Meitong Liu, Xiaoyuan Zhang, Chulin Xie, Kate Donahue, and Han Zhao. Online mirror descent for
tchebycheff scalarization in multi-objective optimization. _arXiv preprint arXiv:2410.21764_, 2024.


Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer
data with rectified flow. In _The Eleventh International Conference on Learning Representations_,
2023. [URL https://openreview.net/forum?id=XVjTT1nw5z.](https://openreview.net/forum?id=XVjTT1nw5z)


Deepika Mathur, Satya Prakash, Priya Anand, Harpreet Kaur, Piyush Agrawal, Ayesha Mehta, Rajesh
Kumar, Sandeep Singh, and Gajendra PS Raghava. Peplife: a repository of the half-life of peptides.
_Scientific reports_, 6(1):36617, 2016.


Richard Michael, Simon Bartels, Miguel González-Duque, Yevgen Zainchkovskyy, Jes Frellsen,
Søren Hauberg, and Wouter Boomsma. A continuous relaxation for discrete bayesian optimization.
_arXiv preprint arXiv:2404.17452_, 2024.


K. Miettinen. _Nonlinear multiobjective optimization_ . Kluwer, Boston, USA, 1999.


Stephanie E Mohr, Yanhui Hu, Benjamin Ewen-Campen, Benjamin E Housden, Raghuvir Viswanatha,
and Norbert Perrimon. Crispr guide rna design for research applications. _The FEBS journal_, 283
(17):3232–3238, 2016.


Gita Naseri and Mattheos AG Koffas. Application of combinatorial optimization strategies in
synthetic biology. _Nature communications_, 11(1):2446, 2020.


Atef Nehdi, Nosaibah Samman, Vanessa Aguilar-Sánchez, Azer Farah, Emre Yurdusev, Mohamed
Boudjelal, and Jonathan Perreault. Novel strategies to optimize the amplification of single-stranded
dna. _Frontiers in Bioengineering and Biotechnology_, 8:401, 2020.


Hunter Nisonoff, Junhao Xiong, Stephan Allenspach, and Jennifer Listgarten. Unlocking guidance for
discrete state-space diffusion and flow models. _Proceedings of the 13th International Conference_
_on Learning Representations (ICLR)_, 2025.


Biswajit Paria, Kirthevasan Kandasamy, and Barnabás Póczos. A flexible framework for multiobjective bayesian optimization using random scalarizations. In _Uncertainty in Artificial Intelli-_
_gence_, pp. 766–776. PMLR, 2020.


Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier
Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, Jake Vanderplas,
Alexandre Passos, David Cournapeau, Matthieu Brucher, Matthieu Perrot, and Édouard Duchesnay.
Scikit-learn: Machine learning in python. _J. Mach. Learn. Res._, 12(null):2825–2830, November
2011. ISSN 1532-4435.


William Peebles and Saining Xie. Scalable diffusion models with transformers. _arXiv_ _preprint_
_arXiv:2212.09748_, 2022.


M Pirtskhalava, B Vishnepolsky, and M Grigolava. Transmembrane and antimicrobial peptides.
hydrophobicity, amphiphilicity and propensity to aggregation. _arXiv preprint arXiv:1307.6160_,
2013.


13


Dillon J Rinauro, Fabrizio Chiti, Michele Vendruscolo, and Ryan Limbocker. Misfolded protein
oligomers: Mechanisms of formation, cytotoxic effects, and pharmacological approaches against
protein misfolding diseases. _Molecular Neurodegeneration_, 19(1):20, 2024.


Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
image segmentation. In _Medical image computing and computer-assisted intervention–MICCAI_
_2015:_ _18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III_
_18_, pp. 234–241. Springer, 2015.


Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T
Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language
models. _Advances in Neural Information Processing Systems_, 2024.


Henri Schmidt, Minsi Zhang, Dimitar Chakarov, Vineet Bansal, Haralambos Mourelatos, Francisco J
Sánchez-Rivera, Scott W Lowe, Andrea Ventura, Christina S Leslie, and Yuri Pritykin. Genomewide crispr guide rna design and specificity analysis with guidescan2. _Genome biology_, 26(1):
1–25, 2025.


Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P Adams, and Nando De Freitas. Taking the
human out of the loop: A review of bayesian optimization. _Proceedings_ _of_ _the_ _IEEE_, 104(1):
148–175, 2015.


Neelam Sharma, Leimarembi Devi Naorem, Shipra Jain, and Gajendra PS Raghava. Toxinpred2: an
improved method for predicting toxicity of proteins. _Briefings in bioinformatics_, 23(5):bbac174,
2022.


Tiago Sousa, João Correia, Vitor Pereira, and Miguel Rocha. Combining multi-objective evolutionary
algorithms with deep generative models towards focused molecular design. In _Applications of_
_Evolutionary Computation:_ _24th International Conference, EvoApplications 2021, Held as Part of_
_EvoStar 2021, Virtual Event, April 7–9, 2021, Proceedings 24_, pp. 81–96. Springer, 2021.


Samuel Stanton, Wesley Maddox, Nate Gruver, Phillip Maffettone, Emily Delaney, Peyton Greenside,
and Andrew Gordon Wilson. Accelerating bayesian optimization for biological sequence design
with denoising autoencoders. In _International conference on machine learning_, pp. 20459–20478.
PMLR, 2022.


Hannes Stark, Bowen Jing, Chenyu Wang, Gabriele Corso, Bonnie Berger, Regina Barzilay, and
Tommi Jaakkola. Dirichlet flow matching with applications to dna sequence design. _Proceedings_
_of the 41st International Conference on Machine Learning (ICML)_, 2024.


Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced
transformer with rotary position embedding. _Neurocomputing_, 568:127063, 2024.


Ruiqing Sun, Dawei Feng, Sen Yang, Yijie Wang, and Huaimin Wang. Evolutionary trainingfree guidance in diffusion model for 3d multi-objective molecular generation. _arXiv_ _preprint_
_arXiv:2505.11037_, 2025.


R Swanson. Long live peptides—evolution of peptide half-life extension technologies and emerging
hybrid approaches. _Drug Discovery World_, 15:57–61, 2014.


Sophia Tang, Yinuo Zhang, and Pranam Chatterjee. Gumbel-softmax flow matching with straightthrough guidance for controllable biological sequence generation. _arXiv preprint arXiv:2503.17361_,
2025a.


Sophia Tang, Yinuo Zhang, and Pranam Chatterjee. Peptune: De novo generation of therapeutic
peptides with multi-objective-guided discrete diffusion. _Proceedings_ _of_ _the_ _41st_ _International_
_Conference on Machine Learning (ICML)_, 2025b.


Masahiro Tominaga, Yoko Shima, Kenta Nozaki, Yoichiro Ito, Masataka Someda, Yuji Shoya,
Noritaka Hashii, Chihiro Obata, Miho Matsumoto-Kitano, Kohei Suematsu, et al. Designing strong
inducible synthetic promoters in yeasts. _Nature Communications_, 15(1):10653, 2024.


14


Oleg Trott and Arthur J Olson. Autodock vina: improving the speed and accuracy of docking with
a new scoring function, efficient optimization, and multithreading. _Journal_ _of_ _computational_
_chemistry_, 31(2):455–461, 2010.


Kotaro Tsuboyama, Justas Dauparas, Jonathan Chen, Elodie Laine, Yasser Mohseni Behbahani,
Jonathan J Weinstein, Niall M Mangan, Sergey Ovchinnikov, and Gabriel J Rocklin. Mega-scale
experimental analysis of protein folding stability in biology and design. _Nature_, 620(7973):
434–444, 2023.


Ben Tu, Nikolas Kantas, Robert M Lee, and Behrang Shafei. Multi-objective optimisation via the r2
utilities. _arXiv preprint arXiv:2305.11774_, 2023.


Tsuyoshi Ueno, Trevor David Rhone, Zhufeng Hou, Teruyasu Mizoguchi, and Koji Tsuda. Combo:
An efficient bayesian optimization library for materials science. _Materials discovery_, 4:18–21,
2016.


Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion
language models are versatile protein learners. _arXiv preprint arXiv:2402.18567_, 2024.


Jing Yang, S Mark Roe, Matthew J Cliff, Mark A Williams, John E Ladbury, Patricia T W Cohen, and
David Barford. Molecular basis for tpr domain-mediated regulation of protein phosphatase 5. _The_
_EMBO Journal_, 24(1):1–10, December 2004. ISSN 1460-2075. doi: 10.1038/sj.emboj.7600496.
[URL http://dx.doi.org/10.1038/sj.emboj.7600496.](http://dx.doi.org/10.1038/sj.emboj.7600496)


Yinghua Yao, Yuangang Pan, Jing Li, Ivor Tsang, and Xin Yao. Proud: Pareto-guided diffusion
model for multi-objective generation. _Machine Learning_, 113(9):6511–6538, 2024.


Jaehoon Yoo, Wonjung Kim, and Seunghoon Hong. Redi: Rectified discrete flow. _arXiv preprint_
_arXiv:2507.15897_, 2025.


Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn.
Gradient surgery for multi-task learning. _Advances in neural information processing systems_, 33:
5824–5836, 2020.


Ye Yuan, Can Chen, Christopher Pal, and Xue Liu. Paretoflow: Guided flows in multi-objective
optimization. _arXiv preprint arXiv:2412.03718_, 2024.


Chengxin Zhang, Xi Zhang, Peter L Freddolino, and Yang Zhang. Biolip2: an updated structure
database for biologically relevant ligand–protein interactions. _Nucleic Acids Research_, 52(D1):
D404–D412, 2024.


Mingjie Zhang, Hidehito Tochio, Qiang Zhang, Pravat Mandal, and Ming Li. Solution structure of
the extended neuronal nitric oxide synthase pdz domain complexed with an associated peptide.
_Nature Structural Biology_, 6(5):417–421, May 1999. ISSN 1072-8368. doi: 10.1038/8216. URL
[http://dx.doi.org/10.1038/8216.](http://dx.doi.org/10.1038/8216)


Qingfu Zhang and Hui Li. Moea/d: A multiobjective evolutionary algorithm based on decomposition.
_IEEE Transactions on evolutionary computation_, 11(6):712–731, 2007.


Ruochi Zhang, Haoran Wu, Yuting Xiu, Kewei Li, Ningning Chen, Yu Wang, Yan Wang, Xin
Gao, and Fengfeng Zhou. Pepland: a large-scale pre-trained peptide representation model for
a comprehensive landscape of both canonical and non-canonical amino acids. _arXiv_ _preprint_
_arXiv:2311.04419_, 2023.


Ruimin Zhou, Zhaoyan Jiang, Chen Yang, Jianwei Yu, Jirui Feng, Muhammad Abdullah Adil, Dan
Deng, Wenjun Zou, Jianqi Zhang, Kun Lu, et al. All-small-molecule organic solar cells with over
14% efficiency by optimizing hierarchical morphologies. _Nature communications_, 10(1):5393,
2019.


Eckart Zitzler and Lothar Thiele. Multiobjective optimization using evolutionary algorithms—a
comparative case study. In _International conference on parallel problem solving from nature_, pp.
292–301. Springer, 1998.


Eckart Zitzler, Marco Laumanns, and Lothar Thiele. Spea2: Improving the strength pareto evolutionary algorithm. _TIK report_, 103, 2001.


15


A THEORETICAL GUARANTEES


In this section, we establish that AReUReDi converges to Pareto-optimal solutions while preserving
coverage of the entire Pareto front. We assume throughout that the state space _S_ is finite, all objective
functions _sn_ are bounded, and their normalized versions ˜ _sn_ map to [0 _,_ 1].


A.1 PRELIMINARY DEFINITIONS


**Definition (Pareto Optimality).** A state _x_ _[∗]_ _∈S_ is _Pareto optimal_ if there exists no _y_ _∈S_ such that
_s_ ˜ _n_ ( _y_ ) _≥_ _s_ ˜ _n_ ( _x_ _[∗]_ ) for all _n ∈{_ 1 _, . . ., N_ _}_ with strict inequality for at least one _n_ .


**Definition (Pareto Front).** The Pareto front is _P_ = _{x ∈S_ : _x_ is Pareto optimal _}_ .


**Definition (Interior Weight Vector).** A weight vector _ω_ _∈_ ∆ _[N]_ _[−]_ [1] is _interior_ if _ωn_ _>_ 0 for all _n_ .


A.2 MAIN THEORETICAL RESULTS


**Theorem (Invariance).** The Markov kernel defined by the Locally Balanced Proposal (LBP) and
Metropolis–Hastings update leaves the distribution


_πη,ω_ ( _x_ ) _∝_ _p_ 1( _x_ ) exp� _ηSω_ ( _x_ )�


invariant for every guidance strength _η_ _>_ 0 and weight vector _ω_ _∈_ ∆ _[N]_ _[−]_ [1] .


_Proof._ We prove this in two steps: first showing that single-coordinate updates preserve detailed
balance, then that random-scan mixtures preserve invariance.

**Step 1:** **Single-coordinate detailed balance.** Let _x_ and _x_ _[′]_ differ only at coordinate _i_, where _x_ _[′]_ _i_ [=] _[ y]_
for some token _y_ . The proposal probability is


_p_ _[i]_ _t_ [(] _[y]_ _[|][ x][t]_ [)] _[g]_ [(] _[r][i]_ [(] _[y]_ [;] _[ x][t]_ [))]
_qi_ ( _y_ _| x_ ) =               - _z∈_ candidates _[p]_ _t_ _[i]_ [(] _[z]_ _[|][ x][t]_ [)] _[g]_ [(] _[r][i]_ [(] _[z]_ [;] _[ x][t]_ [))] _[,]_


_t_ )
where _ri_ ( _y_ ; _xt_ ) = _[W][ηt,ω]_ _Wηt,ω_ [(] _[x]_ [(] ( _[i]_ _x_ _[←]_ _t_ ) _[y]_ [)] and _g_ satisfies _g_ ( _u_ ) = _u · g_ (1 _/u_ ).


The acceptance probability is


By the symmetry property of _g_ and the construction of the proposal, we have


_qi_ ( _y_ _| x_ )

_[W][η,ω]_ [(] _[x][′]_ [)]
_qi_ ( _xi_ _| x_ _[′]_ ) [=] _Wη,ω_ ( _x_ ) _[.]_


Since _πη,ω_ ( _x_ ) = _Z_ _[−]_ [1] _p_ 1( _x_ ) _Wη,ω_ ( _x_ ), it follows that

_πη,ω_ ( _x_ _[′]_ ) _qi_ ( _xi_ _| x_ _[′]_ )

= 1 _._
_πη,ω_ ( _x_ ) _qi_ ( _y_ _| x_ )


Therefore, _αi_ ( _x, x_ _[′]_ ) = 1 and detailed balance is satisfied.

**Step 2:** **Random-scan mixture.** The overall kernel is _K_ ( _x, x_ _[′]_ ) = _L_ [1] - _Li_ =1 _[K][i]_ [(] _[x, x][′]_ [)][, where] _[ K][i]_ [ is]

the kernel for updating coordinate _i_ . Since each _Ki_ satisfies detailed balance with respect to _πη,ω_,
their convex combination also satisfies detailed balance and hence preserves invariance.


**Theorem** **(Convergence** **to** **Pareto** **Front).** Fix any _ω_ _∈_ int ∆ _[N]_ _[−]_ [1] with strictly positive entries
and let _Sω_ ( _x_ ) = min _n ωns_ ˜ _n_ ( _x_ ). If _η_ _→∞_, samples drawn from _πη,ω_ ( _x_ ) _∝_ _p_ 1( _x_ ) exp( _ηSω_ ( _x_ ))
concentrate on the set
_Fω_ = arg max _Sω_ ( _x_ ) _,_
_x_

and every element of _Fω_ is Pareto optimal.


16


     - _[|][ x][′]_ [)]
_αi_ ( _x, x_ _[′]_ ) = min 1 _,_ _[π][η,ω]_ [(] _[x][′]_ [)] _[q][i]_ [(] _[x][i]_

_πη,ω_ ( _x_ ) _qi_ ( _y_ _| x_ )


_._


_Proof._ **Step** **1:** **Maximizers** **of** _Sω_ **are** **Pareto** **optimal.** Suppose _x_ _[∗]_ _∈Fω_ but _x_ _[∗]_ is not Pareto
optimal. Then there exists _y_ _∈S_ with

_s_ ˜ _n_ ( _y_ ) _≥_ _s_ ˜ _n_ ( _x_ _[∗]_ ) _∀n,_ and _s_ ˜ _m_ ( _y_ ) _>_ _s_ ˜ _m_ ( _x_ _[∗]_ ) for some _m._


Since _ωn_ _>_ 0 for all _n_, multiplying preserves inequalities. If _m_ is the bottleneck coordinate of
_x_ _[∗]_, then _Sω_ ( _y_ ) _>_ _Sω_ ( _x_ _[∗]_ ), contradiction. Otherwise, equality requires special weight alignments
(measure zero). Thus maximizers are Pareto optimal almost surely.

**Step 2:** **Concentration as** _η_ _→∞_ **.** Let _Sω_ _[∗]_ [= max] _[x]_ _[S][ω]_ [(] _[x]_ [)][ and][ ∆] _[ω]_ [=] _[ S]_ _ω_ _[∗]_ _[−]_ [max] _x/∈Fω_ _[S][ω]_ [(] _[x]_ [)] _[ >]_ [ 0][.]
Then for _x_ _∈F/_ _ω_,

_πη,ω_ ( _x_ ) _≤_ _e_ _[−][η]_ [∆] _[ω]_ _·_               - _p_ 1( _x_ )

_z∈Fω_ _[p]_ [1][(] _[z]_ [)] _[.]_

Summing gives _πη,ω_ ( _S \ Fω_ ) _→_ 0 as _η_ _→∞_ . Hence the mass concentrates on _Fω_ .


**Theorem** **(Pareto** **Point** **Representability).** For every Pareto-optimal state _x_ _[†]_ _∈P_ there exists
_ω_ _∈_ ∆ _[N]_ _[−]_ [1] such that _x_ _[†]_ _∈_ arg max _x Sω_ ( _x_ ). Moreover, if ˜ _sn_ ( _x_ _[†]_ ) _>_ 0 for all _n_, then _x_ _[†]_ can be made
the unique maximizer.


_Proof._ If ˜ _sn_ ( _x_ _[†]_ ) _>_ 0, define

1 _/s_ ˜ _n_ ( _x_ _[†]_ )
_ωn_ =              - _N_ _._
_k_ =1 [1] _[/][s]_ [˜] _[k]_ [(] _[x][†]_ [)]


**Theorem** **(Coverage** **Guarantee).** Let _µ_ be any probability distribution with full support on
int ∆ _[N]_ _[−]_ [1] . If _ω_ _∼_ _µ_ and _η_ _→∞_, then the induced sampler visits every Pareto-optimal state
with positive probability.


_Proof._ By representability, each Pareto point _x_ _[†]_ maximizes _Sω_ for some interior _ω_ _[†]_ . By continuity,
there exists a neighborhood _Ux†_ where _x_ _[†]_ remains optimal. Since _µ_ ( _Ux†_ ) _>_ 0, randomizing _ω_
ensures _x_ _[†]_ is visited with positive probability in the high- _η_ limit.


**Remark.** The guarantees hold for any finite _S_ and bounded objectives. In practice, convergence
depends on the chain mixing rate, the annealing schedule for _η_, and the choice of balancing function
_g_ .


B PEPREDI AND SMILESREDI GENERATE DIVERSE AND BIOLOGICALLY
PLAUSIBLE SEQUENCES


To enable the efficient generation of peptide binders, we developed an unconditional peptide generator,
**PepReDi**, based on the ReDi framework. The model backbone of PepReDi is a Diffusion Transformer
(DiT) architecture (Peebles & Xie, 2022). We trained PepDFM on a custom dataset comprising
approximately 15,000 peptides from the PepNN and BioLip2 datasets, as well as sequences from the
PPIRef dataset, with lengths ranging from 6 to 49 amino acids (Abdin et al., 2022; Zhang et al., 2024;
Bushuiev et al., 2023). Using this trained model, we generated new data couplings containing 10,000
sequences for each peptide length and used them to fine-tune PepReDi in an iterative rectification
procedure. This rectification was performed three times and yielded substantial improvements in
training loss, validation negative log-likelihood (NLL), perplexity (PPL), and conditional TC (Table
4). Notably, the conditional TC rises after the first rectification, likely due to the distributional shift
from the large, model-generated coupling, whose absolute TC can be higher even though ReDi
guarantees a monotonic decrease within each coupling. The low validation NLL and PPL metrics
showcase PepReDi’s reliability to generate biologically plausible wild-type peptide sequences.


SMILESReDi adopts the same backbone structure as PepReDi, enhanced with Rotary Positional
Embeddings (RoPE), which effectively captures the relative inter-token interactions in peptide
SMILES (Su et al., 2024). SMILESReDi also incorporates a time-dependent noising schedule to


17


Then _Sω_ ( _x_ _[†]_ ) = - 1


1 [and] [for] [any] _[y]_ [=] _[x][†]_ [,] [some] _[m]_ [satisfies] _[s]_ [˜] _[m]_ [(] _[y]_ [)] _[<]_ _[s]_ [˜] _[m]_ [(] _[x][†]_ [)][,] [implying]

_k_ [1] _[/][s]_ [˜] _[k]_ [(] _[x][†]_ [)] [,]


_Sω_ ( _y_ ) _< Sω_ ( _x_ _[†]_ ). If some ˜ _sn_ ( _x_ _[†]_ ) = 0, perturb objectives by _ε >_ 0 and take the limit.


improve its capability to generate valid peptide SMILES sequences (C.2). We applied the same
training data as PepMDLM, a state-of-the-art diffusion model that generates valid peptide SMILES
sequences (Tang et al., 2025b). After only two training epochs, SMILESReDi converged to a
validation NLL of 0.722 and achieved a sampling validity of 76.3% using just 16 generation steps. One
hundred SMILES sequences were then generated by the trained SMILESReDi for each length from 4
to 1035, forming a large and diverse new data coupling. Following a single round of rectification,
the validation NLL further decreased to 0.608, and the sampling validity rose dramatically to 98.6%
with 16 steps and 100% with 32 steps 5. While its similarity-to-nearest-neighbor (SNN) score and
diversity are comparable to those of PepMDLM (details on metrics are provided in Appendix C.2),
SMILESReDi substantially outperforms PepMDLM in validity, highlighting its superior capability of
generating diverse chemically-modified peptide SMILES sequences.


C BASE MODEL DETAILS


C.1 PEPREDI


**Model Architecture.** The backbone of PepReDi is built on a Diffusion Transformer (DiT) framework
implemented within a Masked Diffusion Language Model (MDLM) paradigm (Peebles & Xie, 2022;
Sahoo et al., 2024). Input amino acid sequences are transformed to discrete tokens using the ESM-2650M tokenizer (Lin et al., 2023). Tokenized amino acid sequences and time-steps are converted
to continuous embedding vectors using two separate layers, which are then fused and processed
by stacked DiT transformer blocks equipped with multi-head self-attention to capture long-range
dependencies in the amino-acid sequence. Residual connections and layer normalization stabilize the
training dynamics, and a final projection layer outputs token logits for each position.


**Dataset Curation.** The dataset for PepReDi training was curated from the PepNN, BioLip2, and
PPIRef dataset (Abdin et al., 2022; Zhang et al., 2024; Bushuiev et al., 2023). All peptides from
PepNN and BioLip2 were included, along with sequences from PPIRef ranging from 6 to 49 amino
acids in length. The dataset was divided into training, validation, and test sets at an 80/10/10 ratio.


**Training Strategy.** Training was conducted on a single node equipped with one NVIDIA GPU and
128 GB of GPU memory using the SLURM workload manager. The model was trained for 100
epochs using the Adam optimizer and a learning rate of 1e-4 with weight decay of 1e-5. A learning
rate scheduler with 10 warm-up epochs and cosine decay was used, with initial and minimum learning
rates both 1e-5. The network architecture included a model dimension of 512, 6 transformer layers,
and 8 attention heads, with a vocabulary size of 24 and a maximum sequence length of 100 tokens.
Conditional total correlation estimation was performed using 20 batches and 50 samples per batch to
monitor rectification quality during training. The model checkpoint with the lowest total correlation
was saved. For training rectified models, the same hyperparameter setting was applied, except for the
loaded pre-trained model checkpoint and the weight decay being increased to 2e-5.


**Dynamic** **Batching.** To enhance computational efficiency and manage variable-length token sequences, we implemented dynamic batching. Drawing inspiration from ESM-2’s approach (Lin et al.,
2023), input peptide sequences were sorted by length to optimize GPU memory utilization, with a
maximum token size of 100 per GPU.


**Rectification.** The trained model applied 16 sampling steps to generate 10k sequences for each
peptide length, ranging from 6 to 49, with a temperature hyperparameter set to 1. After generation,
dynamic batching was used to optimize GPU memory utilization for future rectified training.


C.2 SMILESREDI


**Model Architecture.** SMILESReDi follows the ReDi paradigm and uses a Diffusion Transformer
(DiT) backbone embedded in a Masked Diffusion Language Model (MDLM) design to generate
molecular SMILES sequences (Peebles & Xie, 2022; Sahoo et al., 2024). Input SMILES sequences
are transformed to discrete tokens using the PeptideCLM -23M tokenizer. Tokenized amino acid
sequences and time-steps are converted to continuous embedding vectors using two separate layers.
Both embeddings are then fused and processed by stacked DiT transformer blocks that incorporate
Rotary Positional Embeddings (RoPE) and multi-head attention modules to capture long-range


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


structural dependencies while preserving positional information (Su et al., 2024). A final layer
normalization and linear projection outputs token logits for each position.


**Time-dependent bond-aware noising schedule.** Peptide SMILES share a conserved backbone of
alternating carbonyl and amide groups connected by chemically constrained peptide bonds, while
their side chains remain highly diverse. Standard discrete flow matching can corrupt these critical
bond tokens too early, hindering the flow from recovering the backbone along the probability path.
Inspired by previous work in bond-dependent masking, we devised a time-dependent bond-aware
noising schedule that preserves backbone tokens longer than side-chain tokens, allowing the model
to reconstruct the invariant scaffold before generating variable side chains. Specifically, for each
position _j_ with a bond indicator _bj_ _∈{_ 0 _,_ 1 _}_, the time- _t_ marginal of the probability path is


           -            -            -            _pt_ ( _x_ [(] _t_ _[j]_ [)] _| x_ [(] 0 _[j]_ [)] _[, x]_ 1 [(] _[j]_ [)][) =] _bjt_ _[γ]_ + (1 _−_ _bj_ ) _t_ _δx_ (1 _j_ ) + 1 _−_ _bjt_ _[γ]_ _−_ (1 _−_ _bj_ ) _t_ _δx_ (0 _j_ ) _[,]_ _t ∈_ [0 _,_ 1] _,_ _γ_ _>_ 1 _,_


so each token is equal to _x_ [(] 1 _[j]_ [)] with the indicated mixture coefficient and to _x_ [(] 0 _[j]_ [)] otherwise, ensuring
that backbone tokens ( _bj_ = 1) transition more slowly than non-bond tokens along the DFM probability
path.


**Training Strategy.** The training is conducted on a 4*A6000 NVIDIA RTX 6000 Ada GPU system
with 48 GB of VRAM for 5 epochs. The model checkpoint with the lowest evaluation loss was saved.
The Adam optimizer was employed with a learning rate of 1e-4. A learning rate scheduler with 10%
total training steps and cosine decay was used, with initial and minimum learning rates both 1e-5.
The network architecture included a model dimension of 768, 8 transformer layers, and 8 attention
heads. Gradient clip value was set to 1.0 and _γ_ to 2.0 in the time-dependent bond-aware noising
schedule. For training rectified models, the same hyperparameter setting was applied, except for the
loaded pre-trained model checkpoint and the total training epochs set to 10.


**Rectification.** The trained model applied 100 sampling steps to generate 100 sequences for each
peptide length, ranging from 4 to 1035, with a temperature hyperparameter set to 1. After generation,
dynamic batching was used to optimize GPU memory utilization for future rectified training.


**Evaluation Metrics.**


    - **Validity** is defined as the fraction of peptide SMILES that pass the SMILES2PEPTIDE filter
(Tang et al., 2025b), indicating that it translates to a synthesizable peptide.


    - **Uniqueness** is defined as the fraction of mutually distinct peptide SMILES.


    - **Diversity** is defined as one minus the average Tanimoto similarity between the Morgan
fingerprints of every pair of generated sequences, which measures the similarity in structure
across generated peptides.


C.3 PEPDFM


**Model Architecture.** The base model is a time-dependent architecture based on U-Net (Ronneberger
et al., 2015). It uses two separate embedding layers for sequence and time, followed by five
convolutional blocks with varying dilation rates to capture temporal dependencies, while incorporating
time-conditioning through dense layers. The final output layer generates logits for each token. We
used a polynomial convex schedule with a polynomial exponent of 2.0 for the mixture discrete
probability path in the discrete flow matching.


19


Diversity = 1 _−_ - _N_ generated1 - 2 _i,j_


**f** ( **x** _i_ ) _·_ **f** ( **x** _j_ )
_|_ **f** ( **x** _i_ ) _|_ + _|_ **f** ( **x** _j_ ) _| −_ **f** ( **x** _i_ ) _·_ **f** ( **x** _j_ )


where **f** ( **x** _i_ ) and **f** ( **x** _j_ ) are the 2048-dimensional Morgan fingerprint with radius 3 for a pair
of generated sequences **x** _i_ and **x** _j_ .


- **Similarity to Nearest Neighbor (SNN)** is defined as the maximum Tanimoto similarity
between a generated sequence **x** _i_ with a sequence in the dataset **x** ˜ _j_ .


SNN = max
_j∈|D|_


- **f** ( **x** _i_ ) _·_ **f** (˜ **x** _j_ )
_|_ **f** ( **x** _i_ ) _|_ + _|_ **f** (˜ **x** _j_ ) _| −_ **f** ( **x** _i_ ) _·_ **f** (˜ **x** _j_ )


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


**Dataset Curation.** The dataset for PepDFM training was curated from the PepNN, BioLip2, and
PPIRef dataset (Abdin et al., 2022; Zhang et al., 2024; Bushuiev et al., 2023). All peptides from
PepNN and BioLip2 were included, along with sequences from PPIRef ranging from 6 to 49 amino
acids in length. The dataset was divided into training, validation, and test sets at an 80/10/10 ratio.


**Training Strategy.** The training is conducted on a 2xH100 NVIDIA NVL GPU system with 94 GB
of VRAM for 200 epochs with batch size 512. The model checkpoint with the lowest evaluation loss
was saved. The Adam optimizer was employed with a learning rate 1e-4. A learning rate scheduler
with 20 warm-up epochs and cosine decay was used, with initial and minimum learning rates both
1e-5. The embedding dimension and hidden dimension were set to be 512 and 256 respectively for
the base model.


**Performance.** PepDFM achieved a validation loss of 3.1051. Its low generalized KL loss during
evaluation demonstrates PepDFM’s strong capability to generate sequences with high biological
plausibility (Gat et al., 2024).


D OBJECTIVE DESCRIPTION


In this work, five key property objectives are considered in the peptide binder tasks: hemolysis,
non-fouling, solubility, half-life, and binding affinity. Each of these properties plays a crucial role
in optimizing the therapeutic potential of peptides. Hemolysis refers to the peptide’s ability to
minimize red blood cell lysis, ensuring safe systemic circulation (Pirtskhalava et al., 2013). Nonfouling properties describe the peptide’s resistance to unwanted interactions with biomolecules, thus
enhancing its stability and bioavailability in vivo (Chen et al., 2009). Solubility is critical for ensuring
adequate peptide dissolution in biological fluids, directly influencing its absorption and therapeutic
efficacy (Fosgerau & Hoffmann, 2015). Half-life indicates the duration for which the peptide remains
active in circulation, which is vital for reducing dosing frequency (Swanson, 2014). Finally, binding
affinity measures the strength of the peptide’s interaction with its target, directly correlating to its
biological activity and potency in therapeutic applications (Bostrom et al., 2008).


E SCORE MODEL DETAILS


We applied the score models from Tang et al. (2025b) to guide the generation of chemically-modified
peptide binders. We now introduce the score model developed for the wild-type peptide binder
generation task. We collected hemolysis (9,316), non-fouling (17,185), solubility (18,453), and
binding affinity (1,781) data for classifier training from the PepLand and PeptideBERT datasets
(Zhang et al., 2023; Guntuboina et al., 2023). All sequences taken are wild-type L-amino acids and
are tokenized and represented by the ESM-2 protein language model (Lin et al., 2023).


E.1 BOOSTED TREES FOR CLASSIFICATION


For hemolysis, non-fouling, and solubility classification, we trained XGBoost boosted tree models
for logistic regression. We split the data into 0.8/0.2 train/validation using stratified splits from
scikit-learn (Pedregosa et al., 2011) and generated mean-pooled ESM-2-650M (Lin et al., 2023)
embeddings as input features to the model. We ran 50 trials of OPTUNA (Akiba et al., 2019) search to
determine the optimal XGBoost hyperparameters (Table 3), tracking the best binary classification F1
scores. The best models for each property reached F1 scores of 0.58, 0.71, and 0.68 on the validation
sets respectively.


E.2 BINDING AFFINITY SCORE MODEL


We developed an unpooled reciprocal attention transformer model to predict protein-peptide binding
affinity, leveraging latent representations from the ESM-2 650M protein language model (Lin
et al., 2023). Instead of relying on pooled representations, the model retains unpooled token-level
embeddings from ESM-2, which are passed through convolutional layers followed by cross-attention
layers. The binding affinity data were split into a 0.8/0.2 ratio, maintaining similar affinity score
distributions across splits. We used OPTUNA (Akiba et al., 2019) for hyperparameter optimization,
tracing validation correlation scores. The final model was trained for 50 epochs with a learning rate


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


Table 3: XGBoost Hyperparameters for Classification


**Hyperparameter** **Value/Range**


Objective binary:logistic
Lambda [1e _−_ 8 _,_ 10 _._ 0]
Alpha [1e _−_ 8 _,_ 10 _._ 0]

Colsample by Tree [0 _._ 1 _,_ 1 _._ 0]

Subsample [0 _._ 1 _,_ 1 _._ 0]

Learning Rate [0 _._ 01 _,_ 0 _._ 3]

Max Depth [2 _,_ 30]

Min Child Weight [1 _,_ 20]

Tree Method hist


of 3.84e-5, a dropout rate of 0.15, 3 initial CNN kernel layers (dimension 384), 4 cross-attention
layers (dimension 2048), and a shared prediction head (dimension 1024) in the end. The classifier
reached 0.64 Spearman’s correlation score on validation data.


E.3 HALF-LIFE SCORE MODEL


**Dataset Curation.** The half-life dataset is curated from three publicly available datasets: PEPLife,
PepTherDia, and THPdb2 (Mathur et al., 2016; D’Aloisio et al., 2021; Jain et al., 2024). Data related
to human subjects were selected, and entries with missing half-life values were excluded. After
removing duplicates, the final dataset consists of 105 entries.


**Pre-training on stability data.** Given the small size of the half-life dataset, which is insufficient for
training a model to capture the underlying data distribution, we first pre-trained a score model on
a larger stability dataset to predict peptide stability (Tsuboyama et al., 2023). The model consists
of three linear layers with ReLU activation functions, and a dropout rate of 0.3 was applied. The
model was trained on a 2xH100 NVIDIA NVL GPU system with 94 GB of VRAM for 50 epochs.
The Adam optimizer was employed with a learning rate of 1e-2. A learning rate scheduler with 5
warm-up epochs and cosine decay was used, with initial and minimum learning rates both 1e-3. After
training, the model achieved a validation Spearman’s correlation of 0.7915 and an _R_ [2] value of 0.6864,
demonstrating the reliability of the stability score model.


**Fine-tuning on half-life data.** The pre-trained stability score model was subsequently fine-tuned on
the half-life dataset. Since half-life values span a wide range, the model was adapted to predict the
base-10 logarithm of the half-life (h) values to stabilize the learning process. After fine-tuning, the
model achieved a validation Spearman’s correlation of 0.8581 and an _R_ [2] value of 0.5977.


F SAMPLING DETAILS


**Score Model Settings.** We cap the predicted log-scale half-life at 2 (i.e., 100 h) to prevent it from
dominating the optimization and ensure balanced trade-offs across all properties. For the remaining
objectives, hemolysis, non-fouling, solubility, and binding affinity, we directly employ their model
outputs during sampling.


**Wild-Type** **Peptide** **Binder** **Generation** **Task** **Settings.** The total sampling steps are set to 20
multiplied by the binder length. All possible candidate token transitions are evaluated during each
sampling step. We applied the same weight for each objective in all wild-type peptide binder
generation tasks.


**Chemically-Modified Peptide Binder Generation Task Settings.** The total sampling steps are
set to 128. With a vocabulary size of 586, evaluating all the possible candidate tokens is too
computationally intensive. We therefore only evaluated the top 200 candidate tokens during each
sampling step. We applied weight 0.7 for binding affinity, and 0.1 for hemolysis, non-fouling, and


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


Table 4: Training and validation performance of PepReDi over successive rectification rounds. Each row reports
the training loss, validation negative log-likelihood (NLL), validation perplexity (PPL), and conditional total
correlation (TC). PepReDi without superscript denotes the base model, while PepReDi [1], PepReDi [3], PepReDi [3]

indicate the first, second, and third rounds of rectification, respectively.


**Train Loss** **Val NLL** **Val PPL** **Conditional TC**


PepReDi 1.6567 1.6458 5.19 10.6027


PepReDi [1] 1.6170 1.6101 5.00 12.6250
PepReDi [2] 1.5347 1.5238 4.59 11.7279
PepReDi [3] **1.3538** **1.3548** **3.88** **11.2339**


solubility, respectively. Instead of random initialization, the initial sequences _x_ 0 are sampled from
the pre-trained SMILESReDi [1] with 16 generation steps. During generation, AReUReDi rejects any
transitions that will make the SMILES sequence an invalid peptide.


G ABLATION STUDIES


**Computational Cost.** We performed an ablation to study how AReUReDi’s performance scales
with the number of generation steps (Table 12). For wild-type binder design (MYC, 12-mers) and
chemically-modified binder SMILES design (NCAM1, length 200), we generated 100 binders using
64, 128, and 256 sampling steps. In both tasks, all optimized properties consistently improve as the
number of steps increases, while runtime grows approximately linearly with the step budget. However,
for NCAM1 the marginal gains in property scores from 128 to 256 steps are small compared to the
more than twofold increase in runtime. Based on this quality–compute trade-off, we use 128–256
steps for wild-type binder design tasks and 128 steps for chemically-modified binder design tasks in
the main experiments.


**Weight Vectors.** To directly assess how AReUReDi explores the Pareto front, we ran experiments
on two three-objective tasks with varied Tchebycheff weights: wild-type peptide binder design for
CLK1 (Table 13) and chemically-modified peptide binder design for GFAP (Table Table 14). In
both cases, a balanced weight vector produces balanced improvements across all objectives, while
emphasizing a single objective systematically shifts the generated sequences toward that objective,
with corresponding trade-offs in the others. These results indicate that changing _ω_ indeed steers
AReUReDi to different regions of the Pareto front rather than merely re-sampling the same trade-off
point.


**ReDi Priors.** To directly assess the role of ReDi’s prior, we ran an ablation where we replaced the
ReDi prior _p_ 1( _x_ ) with a completely uninformed prior and kept the rest of AReUReDi unchanged,
across both wildtype binder (PPP5, 1B8Q) and chemically-modified binder (TfR, GLP1) design tasks.
In all cases, using the learned prior _p_ 1 yields consistently better multi-objective performance. This
indicates that the discrete flow prior is not a redundant factor in the reward-tilted distribution, but a
crucial reference that anchors sampling in realistic, high-quality regions of sequence space, whereas
removing it degrades the quality of the discovered trade-offs.


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


Table 5: Evaluation metrics for the generative quality of peptide SMILES sequences of max token length set to
200. SMILESReDi without superscription denotes the base model, while SMILESReDi [1] refers to the model
that has undergone one round of rectification.


**Model** **Validity (↑)** **Uniqueness (↑)** **Diversity (↑)** **SNN (↓)**


Data 1.000 1.000 0.885 1.000

PepMDLM 0.450 1.000 0.705 0.513

**SMILESReDi** **0.763** 1.000 0.719 0.593
**SMILESReDi** [1] **0.986** 1.000 0.665 0.579


PepTune 1.000 1.000 0.677 0.486

**AReUReDi** 1.000 1.000 **0.789** **0.392**


Table 6: **Adding a sampling constraint greatly improves AReUReDi’s performance.** Wild-type binders for
two protein targets (PDB 8CN1 and 4EBP2) were generated with or without a sampling constraint using the
same number of generation steps. The table reports the average score for each objective, calculated from 100
generated binders per setting. The best score for each objective is highlighted in bold.


**Target** **Method** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


w/o constraints 0.8650 0.4782 0.4627 2.54 5.2412
8CN1
w/ constraints **0.9213** **0.8676** **0.8697** **44.70** **5.5143**


w/o constraints 0.8879 0.4288 0.4257 1.8781 5.7132
4EBP2
w/ constraints **0.9356** **0.8767** **0.8692** **53.95** **6.4571**


Table 7: Ablation results for wild-type peptide binder design targeting PDB 7LUL with different guidance
settings. For each setting, 100 binders of length 7 were designed.


**Guidance Settings**
**Hemolysis** **Solubility** **Affinity**
**Hemolysis** **Solubility** **Affinity**


✓ ✓ ✓ 0.9389 0.9398 6.2559


_×_ ✓ ✓ 0.8964 0.9465 6.3272


✓ _×_ ✓ 0.9502 0.4013 6.9798


✓ ✓ _×_ 0.9535 0.9642 5.2611


_×_ _×_ ✓ 0.8812 0.2877 7.5057


_×_ ✓ _×_ 0.9036 0.9725 5.2449


✓ _×_ _×_ 0.9802 0.6135 5.0985


_×_ _×_ _×_ 0.8431 0.5810 4.8919


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


Table 8: Ablation results for wild-type peptide binder design targeting PDB CLK1 with different guidance
settings. For each setting, 100 binders of length 12 were designed.


**Guidance Settings**
**Non-Fouling** **Half-Life (h)** **Affinity**
**Non-Fouling** **Half-Life (h)** **Affinity**


✓ ✓ ✓ 0.8285 74.04 6.8099


_×_ ✓ ✓ 0.2902 96.59 7.3906


✓ _×_ ✓ 0.9365 1.33 7.2029


✓ ✓ _×_ 0.9479 75.68 6.3437


_×_ _×_ ✓ 0.9625 1.23 6.2319


_×_ ✓ _×_ 0.3540 100.00 6.4116


✓ _×_ _×_ 0.2531 2.96 8.6580


_×_ _×_ _×_ 0.4988 1.82 5.4739


Table 9: **Rectification of the base generation model improves AReUReDi’s performance.** Wild-type binders
for two protein targets (PDB 5AZ8 and AMHR2) were generated using AReUReDi with three different base
models: PepDFM, PepReDi (without rectification), and PepReDi [3] (with three rounds of rectification). The table
reports the average score for each objective, calculated from 100 generated binders per setting. The best score
for each objective is highlighted in bold.


**Target** **Base Model** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


PepDFM 0.9296 **0.8867** **0.8743** 37.30 6.2291

5AZ8 PepReDi **0.9326** 0.8759 0.8572 50.16 **6.4391**
PepReDi [3] 0.9293 0.8732 0.8605 **58.33** 6.2792


PepDFM 0.9412 0.8774 0.8612 47.84 7.2373

AMHR2 PepReDi 0.9127 0.8602 0.8460 50.92 7.0101
PepReDi [3] **0.9420** **0.8914** **0.8755** **63.34** **7.2533**


Table 10: **Annealed guidance strength improves AReUReDi’s performance.** Wild-type binders for two protein
targets (PDB 1DDV and P53) were generated under four guidance schedules: (1) fixed at the minimum strength
_ηmin_ = 1 _._ 0, (2) fixed at the maximum strength _ηmax_ = 20 _._ 0, (3) fixed at the midpoint [1] 2 [(] _[η][min]_ [+] _[η][max]_ [) = 10] _[.]_ [5][,]

and (4) an annealed schedule where _ηt_ increases from _ηmin_ to _ηmax_ over optimization steps. The table reports
the average score for each objective, calculated from 100 generated binders per setting. The best score for each
objective is highlighted in bold.


**Target** **Method** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


1DDV


P53


_η_ = _ηmin_ 0.9130 0.8575 0.8429 38.70 5.3554
_η_ = _ηmax_ 0.9156 0.8512 0.8479 40.27 5.4359
_η_ = [1] 2 [(] _[η][min]_ [ +] _[ η][max]_ [)] 0.9108 0.8641 0.8544 40.43 5.5396

_ηt_ = _η_ min + - _η_ max _−_ _η_ min� _T −t_ 1 0.9128 0.8545 **0.8565** **44.73** 5.4482


_η_ = _ηmin_ 0.9335 0.8800 0.8706 49.97 6.2538
_η_ = _ηmax_ 0.9293 0.8693 0.8657 61.76 6.3043
_η_ = [1] 2 [(] _[η][min]_ [ +] _[ η][max]_ [)] 0.9294 0.8713 0.8653 59.43 6.3060

_ηt_ = _η_ min + - _η_ max _−_ _η_ min� _T −t_ 1 **0.9353** **0.8818** **0.8785** **62.83** **6.3508**


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


Table 11: **Best-of-** _N_ **comparison between PepTune+DPLM and AReUReDi under matched wall-clock time.**
For each target, PepTune+DPLM is allowed to generate 100 binders while AReUReDi generates only 4 (PDB
1B8Q) or 3 (PPP5). Top-2 sequences from each method were reported. The table reports the average score for
each objective.


**Target** **Method** **Rank** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


Top 1 0.9323 0.4379 0.3624 9.82 7.0534
PepTune + DPLM


Table 12: **Increasing** **generation** **steps** **improves** **AReUReDi’s** **performance.** AReUReDi designed 100
generated binders for MYC (12-mer wild-type peptides) and NCAM1 (chemically-modified peptides of length
200) using different numbers of generation steps. The table reports the average score for each objective. Half-life
is not optimized for NCAM1 and is indicated by “*”.


**Target** **# Steps** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity** **Time**


64 0.9279 0.8571 0.8519 5.49 6.5167 67
MYC 128 0.9301 0.8721 0.8627 16.54 6.5811 131

256 0.9357 0.8820 0.8740 34.83 6.5293 265


64 0.8801 0.2468 0.7954       - 5.3936 112
NCAM1 128 0.8840 0.2657 0.8109 - 5.4377 198

256 0.8900 0.3015 0.8202       - 5.5929 423


Table 13: Ablation results for wild-type peptide binder design targeting CLK1 with different weight vector
settings. For each setting, 100 binders of length 12 were designed. The table reports the average score for each
objective.


**Weight Vectors**
**Non-Fouling** **Half-Life (h)** **Affinity**
**Non-Fouling** **Half-Life** **Affinity**


0.3 0.3 0.3 0.8285 74.04 6.8099

0.8 0.1 0.1 0.9367 6.94 6.5231

0.1 0.8 0.1 0.5642 85.47 6.3649

0.1 0.1 0.8 0.6698 48.94 7.4922


25


1B8Q


PPP5


Top 2 0.8718 0.2573 0.2391 38.67 6.5605


Top 1 0.8651 0.8638 0.8892 100.00 5.6008
AReUReDi

Top 2 0.9354 0.8567 0.9331 49.25 6.5605


Top 1 0.7984 0.3338 0.2342 80.27 7.6117
PepTune + DPLM

Top 2 0.7901 0.0966 0.1328 100.00 6.7571


Top 1 0.9407 0.9378 0.9131 100.00 6.8193
AReUReDi

Top 2 0.9606 0.8750 0.8399 90.16 6.8969


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


Table 14: Ablation results for chemically-modified peptide binder design targeting GFAP with different weight
vector settings. For each setting, 100 binders of length 200 were designed. The table reports the average score
for each objective.


**Weight Vectors**
**Non-Fouling** **Solubility** **Affinity**
**Non-Fouling** **Solubility** **Affinity**


0.3 0.3 0.3 0.2754 0.8169 5.3011

0.8 0.1 0.1 0.3322 0.7528 5.3487

0.1 0.8 0.1 0.2273 0.8327 5.3378

0.1 0.1 0.8 0.2498 0.7910 5.8827


Table 15: **PepReDi** **provides** **prior** **knowledge** **that** **helps** **AReUReDi** **to** **generate** **samples** **with** **better**
**multi-objective trade-offs.** 100 wild-type binders were designed for PDB 1B8Q (8-mer) and PPP5 (16-mer),
respectively. The table reports the average score for each objective. The best score for each objective is
highlighted in bold.


**Target** **Prior** **Hemolysis** **Non-Fouling** **Solubility** **Half-Life (h)** **Affinity**


Uniform Prior 0.9009 0.8191 0.8049 14.20 5.8432
1B8Q

PepReDi Prior **0.9214** **0.8680** **0.8654** **22.93** **5.7130**


Uniform Prior 0.9265 0.8263 0.7993 17.52 6.7122
PPP5

PepReDi Prior **0.9412** **0.896** **0.8832** **38.28** **6.7186**


Table 16: **SMILESReDi provides prior knowledge that helps AReUReDi to generate samples with better**
**multi-objective trade-offs.** For each setting, 100 chemically-modified binders of length 200 were designed.
The table reports the average score for each objective. The best score for each objective is highlighted in bold.


**Target** **Prior** **Hemolysis** **Non-Fouling** **Solubility** **Affinity**


Uniform Prior 0.8652 0.2381 **0.7777** 5.5535
TfR

SMILESReDi Prior **0.8665** **0.3234** 0.7408 **6.1271**


Uniform Prior 8.3414 0.2123 **0.7777** 7.5731
GLP1

SMILESReDi Prior **0.8743** **0.3438** 0.7661 **8.3414**


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


Figure A1: **Complex structures of target proteins with pre-existing binders.** **(A)-(B)** 5AZ8 **(C)-(D)** 7JVS.
Each panel shows the complex structure of the target with either an AReUReDi-designed binder or its preexisting binder. For each binder, five property scores are provided, as well as the ipTM score from AlphaFold3
and the docking score from AutoDock VINA. Interacting residues on the target are visualized.


27


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


Figure A2: **Complex structures of target proteins without pre-existing binders.** **(A)-(C)** AMHR2, **(D)-(E)**
EWS::FLI1, **(F)** MYC, **(G)** DUSP12. Each panel shows the complex structure of the target with an AReUReDidesigned binder. For each binder, five property scores are provided, as well as the ipTM score from AlphaFold3
and the docking score from AutoDock VINA. Interacting residues on the target are visualized.


28


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


Figure A3: **(A), (C)** Example 2D SMILES structure of AReUReDi-designed peptide binders with four property
scores for GLP1 and GLAST, respectively. **(B), (D)** Plots showing the mean scores for each property across the
number of iterations during AReUReDi’s design of binders of length 200 for GLP1 and GLAST, respectively.


29


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


Figure A4: **(A), (C)** Example 2D SMILES structure of AReUReDi-designed peptide binders with four property
scores for TfR and AMHR2, respectively. **(B), (D)** Plots showing the mean scores for each property across the
number of iterations during AReUReDi’s design of binders of length 200 for TfR and AMHR2, respectively.


30


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


**Algorithm 1** AReUReDi: Annealed Rectifed Updates for Refning Discrete Flows


1: **Input:** Pre-trained ReDi model _p_ _[i]_ _t_ [(] _[·|][x][t]_ [)][, objective functions][ ˜] _[s]_ [1] _[, . . .,]_ [ ˜] _[s][N]_ [, weight vector] _[ ω]_ _[∈]_
∆ _[N]_ _[−]_ [1], annealing parameters _ηmin, ηmax_ .
2: **Output:** Sequence _xT_ with multi-objective optimized properties.
3: **Initialize:**
4: Sample an initial sequence _x_ 0 uniformly from the discrete state space _S_
5: Sample or specify a weight vector _ω_ _∈_ ∆ _[N]_ _[−]_ [1]

6: **for** _t_ = 0 to 1 with step size _h_ = _T_ [1] **[do]**

7: **Step 1:** **Annealing and Coordinate Selection**
8: Update guidance strength: _ηt_ _←_ _ηmin_ + ( _ηmax −_ _ηmin_ ) _T −t_ 1
9: Select a position _i_ in the sequence to update: _i ∼_ Uniform( _{_ 1 _, . . ., L}_ )
10: **Step 2:** **Proposal Generation via Local Balancing**
11: Let _Ci_ be the set of candidate tokens from _p_ _[i]_ _t_ [(] _[·|][x][t]_ [)][.]
12: For each candidate token _y_ _∈_ _Ci_ :
13: Compute scalarized reward ratio _ri_ ( _y_ ; _xt_ ):

_ri_ ( _y_ ; _xt_ ) _←_ [exp(] _[η][t]_ [ min] _[n][ ω][n][s]_ [˜] _[n]_ [(] _[x]_ [(] _[i][←][y]_ [)][))]

exp( _ηt_ min _n ωns_ ˜ _n_ ( _x_ ))
14: Compute unnormalized proposal distribution _q_ ˜ _i_ ( _y|xt_ ) using a balancing function _g_ ( _·_ ):


_q_ ˜ _i_ ( _y|xt_ ) _←_ _p_ _[i]_ _t_ [(] _[y][|][x][t]_ [)] _[g]_ [(] _[r][i]_ [(] _[y]_ [;] _[ x][t]_ [))]
15: Normalize to get the final proposal distribution _qi_ ( _y|xt_ ).
16: **Step 3:** **Metropolis-Hastings Acceptance**
17: Sample a candidate token _y_ _[∗]_ _∼_ _qi_ ( _·|xt_ ).
18: Form the proposed state _xprop_ _←_ _x_ [(] _[i][←][y][∗]_ [)] .
19: Compute acceptance probability _αi_ ( _x, xprop_ ):


20: With probability _αi_ ( _x, xprop_ ), accept the proposal: _x ←_ _xprop_ .
21: Update time: _t →_ _t_ + _h_
22: **end for**
23: **Return:** Final sequence _x_ 1.


31


      _αi_ ( _x, xprop_ ) _←_ min 1 _,_ _[π][η][t][,ω]_ [(] _[x][prop]_ [)] _[q][i]_ [(] _[x][i][|][x][prop]_ [)]

_πηt,ω_ ( _x_ ) _qi_ ( _y_ _[∗]_ _|x_ )


_πηt,ω_ ( _z_ ) _∝_ _p_ 1( _z_ ) exp( _ηt_ min _n_ _[ω][n][s]_ [˜] _[n]_ [(] _[z]_ [))]


**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


H USE OF LARGE LANGUAGE MODELS (LLMS)


We acknowledge the use of large language models (LLMs) to assist in polishing and editing parts of
this manuscript. LLMs were used to refine phrasing, improve clarity, and ensure consistency of style
across sections. All technical content, experiments, analyses, and conclusions were developed by the
authors, with LLM support limited to language refinement and editorial improvements.


32
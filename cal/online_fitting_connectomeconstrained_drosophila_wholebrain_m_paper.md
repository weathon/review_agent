# ONLINE FITTING CONNECTOME-CONSTRAINED DROSOPHILA WHOLE-BRAIN MODEL REPRODUCES CRITICAL RESTING-STATE DYNAMICS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The rapid growth of large-scale synaptic connectome maps and neural activity
datasets has created an urgent need for connectome-constrained whole-brain models that can fit and interpret experimentally recorded neural data. A promising approach to bridge this gap is to train biologically inspired models using backpropagation through time (BPTT), which enables data-driven optimization of unknown
model parameters. However, BPTT is inherently an offline method, with memory
requirements that grow linearly with simulation time, making it impractical for
training large-scale whole-brain networks over biologically relevant timescales.
To address this challenge, we introduce an online learning framework for fitting
whole-brain models using online gradient-based optimization. By updating parameters in a strictly forward-time manner, our method reduces memory consumption to a single time step, scaling only with the number of parameters rather than
the entire temporal sequence. Using this framework, we construct a _Drosophila_
whole-brain network comprising over 130,000 neurons and millions of synapses,
where the network topology is fixed from the FlyWire connectome, and unknown
parameters such as synaptic weights and cellular time constants are optimized to
match _in vivo_ resting-state neural activity. Our results show that this approach enables the training of large-scale _Drosophila_ models over experimental timescales
on a single GPU, a feat that is computationally prohibitive with BPTT. Remarkably, the optimization not only captures target dynamics but also spontaneously
produces synaptic weight distributions that closely match empirical connectome
statistics and drives the network toward a hallmark feature of resting state – critical dynamics. Together, this work establishes an online, scalable, and data-driven
framework for integrating anatomical and functional datasets, paving the way toward mechanistic whole-brain models at unprecedented scales.


1 INTRODUCTION


The quest to understand how the brain’s structural connectivity gives rise to its complex dynamics
represents one of the fundamental challenges in neuroscience. Recent technological advances have
produced unprecedented datasets documenting both the anatomical wiring and functional activity of
neural systems (Amunts et al., 2013; Cook et al., 2019; Dorkenwald et al., 2024; Oh et al., 2014;
Ding et al., 2025). The complete synaptic-resolution connectome of _Drosophila_, comprising over
130,000 neurons and millions of synapses, now provides a comprehensive structural blueprint of
an entire fly brain (Dorkenwald et al., 2024). In parallel, large-scale calcium imaging and electrophysiological recordings have captured the rich spatiotemporal dynamics of neural activity across
whole-brain populations (Mann et al., 2017; Turner et al., 2021). However, a critical gap remains:
we lack computational frameworks capable of integrating these anatomical and functional datasets
into mechanistic models that can explain how connectivity patterns shape neural dynamics.


Current approaches to modeling neural activity fall into two distinct paradigms, each with fundamental limitations. Task-optimized artificial neural networks (ANNs) have revolutionized our ability
to predict neural responses across diverse brain systems. Deep convolutional networks trained for
object recognition in macaque inferotemporal cortex, with improved task performance correlating
with higher neural predictivity (Yamins et al., 2014; Nayebi et al., 2018). Despite their impressive


1


predictive accuracy, these models suffer from critical shortcomings in biological interpretability.
Without incorporating known anatomical constraints or biological mechanisms, they typically operate as black-box function approximators that can predict what neurons will do but not why specific
representations emerge or how they relate to known circuit mechanisms.


At the opposite extreme, detailed biophysical models prioritize biological realism by incorporating precise anatomical connectivity and physiological properties. Recent efforts have successfully
integrated multi-modal experimental data to construct models with tens of thousands of neurons,
reproducing visual responses in the mouse cortex (Billeh et al., 2020; Chen et al., 2022) and motion detection in the _Drosophila_ visual system (Lappalainen et al., 2024). These models respect
the constraints imposed by synaptic-resolution connectomes and train unknown parameters such as
synaptic weights and physiological parameters using the backpropagation through time (BPTT) algorithm. However, this detailed modeling paradigm faces insurmountable computational barriers
when scaling to whole-brain systems. BPTT requires storing complete activation histories throughout training, with memory consumption that scales linearly with both network size and simulation
duration. Training a 50,000-neuron mouse visual cortex model required 160 GPUs, and even then
only for relatively short behavioral tasks (Chen et al., 2022). For whole-brain networks comprising
hundreds of thousands of neurons operating over biologically relevant timescales, BPTT becomes
computationally prohibitive. This fundamental limitation has created an impasse: researchers must
choose between biologically grounded models that cannot scale (Zhu et al., 2025) and scalable models that lack biological grounding (Pathak et al., 2022).


In this work, we introduce an online learning framework to resolve this computational bottleneck.
Rather than accumulating gradients across entire trajectories, our method updates parameters in a
strictly forward-time manner, reducing memory requirements from scaling with simulation length
to depending only on the number of model parameters. This approach makes it feasible, for the first
time, to optimize large-scale brain networks over experimental timescales on standard computational
resources. We demonstrate this framework by constructing a whole-brain _Drosophila_ model where
the complete FlyWire connectome provides the fixed anatomical scaffold (Dorkenwald et al., 2024),
while unknown parameters, including synaptic weights and cellular time constants, are optimized to
match experimentally recorded neural dynamics (Mann et al., 2017; Turner et al., 2021).


Our results reveal that this data-driven optimization process not only successfully reproduces target neural activity patterns but also yields emergent properties that were not explicitly enforced.
The optimized synaptic weight distributions closely match those observed in the empirical connectome, suggesting that functional constraints shape synaptic strengths in predictable ways. Moreover,
the trained network spontaneously develops complex dynamical features characteristic of biological
neural systems. These findings demonstrate that online gradient-based optimization can serve as a
powerful bridge between structure and function, enabling whole-brain models that are simultaneously constrained by anatomy, fitted to experimental data, and capable of revealing principles of
neural organization.


2 RELATED WORK


**Neural** **activity** **fitting.** Modeling neural activity through network fitting has become an important tool in neuroscience. One widely used approach represents a brain region as a probabilistic
recurrent spiking network, with parameters optimized to maximize the likelihood of observed spike
trains (Gerwinn et al., 2010; Gerhard et al., 2013). Although effective for capturing certain statistical features of neural responses, these models often fail to reproduce realistic large-scale population
dynamics and circuit-level activity patterns (Bellec et al., 2021). Since 2014, deep learning has
catalyzed a complementary line of work that uses task-optimized ANNs as brain models. Notably,
Yamins, DiCarlo, and colleagues showed that deep convolutional networks trained for object recognition yield intermediate representations that closely track macaque IT responses (Yamins et al.,
2014; Nayebi et al., 2018), with better task performance accompanying higher neural predictivity
(Kubilius et al., 2019; Schrimpf et al., 2020). This task-driven paradigm has broadened beyond the
visual system: sequence-to-sequence RNNs have been used to model sentence-level responses in
language areas (Hosseini et al., 2024); speech recognition networks help predict auditory-cortex encoding (Ahmed et al., 2025); and deep reinforcement-learning agents sometimes develop grid-like
codes reminiscent of hippocampal representations (Banino et al., 2018). However, ANN models face


2


challenges in biological interpretability. They often act as black boxes, accurately predicting neural
responses without explaining why certain representations emerge or how they relate to real circuits.
Lacking anatomical, physiological, and biophysical constraints, their correspondence to biological
neurons is ambiguous, and apparent similarities to neural data may reflect task-driven coincidences
rather than true mechanisms.


**Connectome-constrained brain modeling.** The field of connectomics has progressively mapped
neural wiring across species, from the complete nervous system of _C._ _elegans_ (Cook et al., 2019)
to the synaptic-resolution connectome of _Drosophila_ (Dorkenwald et al., 2024) and emerging maps
of mouse (Oh et al., 2014; Ding et al., 2025) and human brains (Amunts et al., 2013). These resources have opened the door to connectome-constrained modeling, where the anatomical scaffold
of the connectome is used to build computational models that link structure to function. At the
macroscopic scale, whole-brain models using neural mass models prioritize scale over detail by abstracting each brain region as a neural mass (Pathak et al., 2022). These models use anatomical
connectivity from diffusion MRI as structural scaffolding, with nodes representing mean-field activity of brain areas that evolve according to differential equations under coupling influences from other
regions (Griffiths et al., 2021). While computationally tractable at whole-brain scales, this approach
suffers from severely limited predictive resolution. More critically, these models typically rely on
only a few global coupling parameters, fundamentally constraining their capacity for individualized
prediction and task generalization. At the cellular scale, neuron-resolved models pursue the opposite
strategy, prioritizing biological detail at the expense of scale. Recent advances have demonstrated
remarkable success in this direction: Billeh et al. (2020) systematically integrated multimodal data
to create biologically realistic models of mouse primary visual cortex with over 50,000 neurons,
which Chen et al. (2022) successfully trained on visual tasks using backpropagation across 160
GPUs. Similarly, Lappalainen et al. (2024) trained a _Drosophila_ visual system model constrained
by synapse-level connectome data, reproducing motion detection and experimental neural responses
without requiring precise physiological parameters for individual neurons. Zhu et al. (2025) further
demonstrated how optimized barrel cortex models can replicate network dynamics during whisker
processing. However, this detailed modeling paradigm faces critical limitations that prevent its
extension to whole-brain scales. Scaling BPTT for whole-brain-scale modeling is exceedingly difficult due to its massive computational resource demands(Lillicrap et al., 2020). More practically,
the memory and computational costs grow prohibitively with simulation length, rendering activity
fitting over biological time scales infeasible for large-scale networks.


**Critical** **state** **of** **resting-state** **neural** **activity.** Extensive experimental studies across multiple
species consistently demonstrate that resting-state neural activity in the brain exhibits hallmark characteristics of critical systems, such as neuronal avalanches with power-law distributions (Beggs &
Plenz, 2003; Ponce-Alvarez et al., 2018; Fusc`a et al., 2023; Fontenele et al., 2019). These observations raise two fundamental questions: Why does the resting brain tend to operate near criticality?
And how does the neural system spontaneously maintain such delicate dynamical balance? A prevailing view is that criticality is the outcome of functional optimization (O’Byrne & Jerbi, 2022).
Operating at the critical point provides the unique computational advantages: it enables the brain
to elegantly navigate between disordered random states and overly synchronized states, establishing
an optimal balance between stability and flexibility while simultaneously maximizing information
transmission capacity and processing efficiency (Shew & Plenz, 2013; Tkaˇcik et al., 2015). To test
this hypothesis, researchers have developed diverse computational models to reproduce critical-state
features of neural activity. At the network level, studies using excitatory-inhibitory neural networks
with random graphs have shown that when network parameters (such as connection strength and
excitatory-inhibitory balance) are precisely tuned near a phase transition point, the system can faithfully reproduce experimentally observed power-law distributed neuronal avalanches and long-range
correlated dynamics (Shew et al., 2009). At the statistical level, Ising models abstract the brain as
a spin network with phase transition behavior. Near the critical temperature, these models exhibit
multi-scale fluctuation patterns and complex correlation structures that closely resemble those of the
biological brain (Nicoletti et al., 2020; Cabral-Carvalho et al., 2025). Despite these advances, existing models are overly simplified, focusing on statistical patterns like power-law distributions while
neglecting the prediction of complete neural signal dynamics. They also rely on idealized network
topologies rather than real brain anatomical connectivity and require manual parameter tuning to
maintain criticality. These shortcomings highlight the need for biologically realistic models that can
self-organize to sustain criticality.


3


Figure 1: **Online fitting workflow of** _**Drosophila**_ **whole-brain neural activity.** (A) Online fitting
paradigm, which employs online gradient optimization to minimize the discrepancy between network activity and experimental data. (B) Example traces of neuropil calcium signals ∆ _F/F_ and corresponding firing rate conversions. (C) Single-neuron dynamics with free parameters (highlighted in
red) optimized during training. (D) Neuropil firing-rate readout constrained by the synapse-resolved
connectome. (E) Online optimization diagram during whole-brain neural activity fitting.


3 ONLINE FITTING OF _Drosophila_ WHOLE-BRAIN NEURAL ACTIVITY


3.1 ONLINE FITTING FRAMEWORK


We develop an online fitting framework that bridges anatomical structure and functional dynamics
by integrating connectome constraints with scalable gradient-based algorithms to model whole-brain
_Drosophila_ neural activity (Fig. 1A). The workflow begins with large-scale calcium imaging data
acquired from _Drosophila_ brains during spontaneous resting states (Mann et al., 2017; Turner et al.,
2021), which provide ground-truth spatiotemporal patterns of neural activity that our biological
model must reproduce. To build the model, we use the FlyWire synaptic-resolution connectome
(Dorkenwald et al., 2024) as the fixed anatomical scaffold and describe the dynamics of each neuron
with firing-rate threshold-linear models (Miller & Fumarola, 2012). While the connectome specifies
the network topology, several biophysical parameters remain unknown, including synaptic weights,
neuronal time constants, and background input strengths. These parameters are treated as free variables to be optimized against experimental activity data. Model optimization is performed with an
online gradient optimization algorithm implemented in the BrainScale platform (Wang et al., 2024).
Unlike BPTT, our online approach scales in memory only with the number of model parameters,
making it computationally feasible to fit a whole-brain, connectome-constrained model directly to
experimental recordings over biologically relevant timescales.


3.2 WHOLE-BRAIN CALCIUM IMAGING NEURAL ACTIVITY


Whole-brain _in vivo_ calcium imaging datasets of resting-state activity in _Drosophila_ have been collected across 18 brains (Mann et al., 2017; Turner et al., 2021). Recordings were acquired at a
sampling rate of 1.2 Hz and subsequently registered to a standardized template brain atlas comprising 73 neuropils (Jenett et al., 2012). The resulting data are represented as ∆ _F/F_ fluorescence
signals for each neuropil, providing population-level activity traces across the entire brain (Fig. 1B).


4


model parameters


model


**C** **D** **E**


states


Single neuron dynamics


neuron _a_


d𝑟&


[&] (-%% + 𝐼&


𝜏&


d𝑡 [= −𝑟][&][ + 𝑓(𝐼][&]


'%()


...


Recurrent synaptic input


𝐼&(-%%,! = 0

1∈𝒩!


sgn&1 |𝑤&1| 𝑟1!


!


Background input


'%(,! = ReLU( 0


%'+,-./0,!"#* W*,&


𝐼&


FR*


'%()


FR%'+,-./0,!* =


8! > n67%8,*


FR [!"#] FR [!] FR [!$%"#]


𝑟5 [!] - n67%5,*


67%5,* + 𝑟8!


n67%5,* + n67%8,*


n5,*


67%


67%


To enable direct comparison with model outputs, which are expressed in firing rates (section 3.3),
the calcium fluorescence signals were transformed into estimated firing-rate dynamics. This conversion was performed using a sparse deconvolution method (Appendix A), which infers underlying neuronal activity from the slower calcium signal dynamics. The resulting dataset thus provides neuropil-level firing-rate activity patterns that serve as training targets for our whole-brain
connectome-constrained model.


3.3 SINGLE-NEURON DYNAMICS


The activity of each neuron _i_ is modeled by a first-order rate equation (Fig. 1C):


d _ri_
_τi_ d _t_ = _−ri_ + _f_ ( _Ii_ [conn] + _Ii_ [enc] ) _,_ (1)

where _ri_ denotes the firing rate of neuron _i_, and _τi_ is its membrane time constant. The term _Ii_ [conn]
represents the recurrent synaptic input arising from the connectome-defined network, while _Ii_ [enc]
denotes external driving input, such as background or stimulus-related signals. The nonlinear activation function _f_ ( _·_ ) determines how inputs are transformed into firing-rate responses. To ensure
firing rates remain non-negative while avoiding artificial saturation effects, we employ the rectified
linear function: _f_ ( _x_ ) = max(0 _, x_ ). This choice provides a simple yet effective approximation
of neuronal response functions, consistent with threshold-linear models widely used in theoretical
neuroscience (Miller & Fumarola, 2012).


To simulate the model, we use the exponential Euler approximation with time step ∆ _t_ :

_ri_ ( _t_ ) = _αiri_ ( _t −_ 1) + (1 _−_ _αi_ ) _f_ ( _Ii_ [conn] ( _t_ ) + _Ii_ [enc] ( _t_ )) _,_ (2)

where the decay factor is given by _αi_ = exp( _−_ ∆ _t/τi_ ) and is learned during optimization.


3.4 CONNECTOME-BASED RECURRENT CONNECTION


Neurons are interconnected according to the synaptic-resolution _Drosophila_ connectome provided
by the FlyWire project (Dorkenwald et al., 2024; Schlegel et al., 2024). We used the version 783
release, which contains 138,639 neurons and 15,091,982 synaptic connections. In this work, we
incorporated two key features from the connectome: (i) the binary connectivity structure, defined as
the set of presynaptic partners _Ni_ for each neuron _i_, and (ii) the synaptic polarity of each connection,
sgn _ij_ . Polarity was assigned based on neurotransmitter identity: acetylcholine and dopamine were
considered excitatory (+1), while GABA, glutamate, octopamine, and serotonin were considered
inhibitory ( _−_ 1), consistent with their dominant physiological effects in the fly brain. Therefore, the
recurrent synaptic input to neuron _i_ at time _t_ is thus expressed as (Fig. 1C):

_Ii_ [conn] ( _t_ ) =                   - sgn _ij |wij| rj_ ( _t_ ) _,_

_j∈Ni_

where _rj_ ( _t_ ) is the firing rate of presynaptic neuron _j_, sgn _ij_ _∈{_ +1 _, −_ 1 _}_ denotes synaptic polarity,
and _|wij|_ represents the magnitude of the effective synaptic weight. Synaptic weights are initialized
randomly (Appendix B) and subsequently optimized during model fitting.


3.5 RESTING-STATE BACKGROUND INPUT


While our model explicitly captures direct synaptic connections through the connectome, numerous
sources of input remain unaccounted for during resting states. To capture these unknown input
influences without explicitly modeling each mechanism, we introduce a data-driven background
input term that allows the network to generate realistic spontaneous activity patterns. Specifically,
we model the background input _Ii_ [enc] ( _t_ ) to each neuron as a learned function of the population-level
activity state (Fig. 1C):


where FR [neuropil] _k_ ( _t −_ 1) denotes the firing rate of neuropil _k_ at time _t −_ 1, and _Wk,i_ [enc] [represents]
the effective coupling strength from neuropil _k_ to neuron _i_ . The ReLU nonlinearity ensures nonnegative input values, consistent with excitatory drive. This formulation reflects the hypothesis that


5


_Ii_ [enc] ( _t_ ) = ReLU


��

FR [neuropil] _k_ ( _t −_ 1) _Wk,i_ [enc]
_k_


_,_ (3)


each neuron’s background input is shaped by large-scale population activity at the neuropil level,
thereby coupling single-neuron dynamics to global brain-wide activity patterns.


3.6 NEUROPIL FIRING RATE READOUT


Our model operates at single-cell resolution with over 130,000 individual neurons, while the calcium
imaging data are spatially averaged within anatomically defined neuropil regions. To enable quantitative comparison between model predictions and experimental observations, we need to transform
single-neuron firing rates into neuropil-level population activity. We leverage the anatomical organization of the _Drosophila_ brain, where neurons extend processes across multiple neuropils, forming
region-specific synaptic territories. Specifically, we compute the population firing rate for each neuropil by aggregating contributions from all neurons that form synapses within that region, weighted
by their relative synaptic density (Fig. 1D). Formally, the firing rate FR [neuropil] _k_ ( _t_ ) of neuropil _k_ at
time _t_ is computed as:


where _Mk_ denotes the set of all neurons with presynaptic terminals in neuropil _k_, and _n_ [syn] _j,k_ [quan-]
tifies the number of synaptic connections that neuron _j_ forms within neuropil _k_ . This synapseweighted averaging reflects the principle that neurons with more extensive arborizations in a given
neuropil contribute more strongly to the calcium signal measured from that region.


3.7 NEURAL ACTIVITY FITTING WITH ONLINE GRADIENT OPTIMIZATION


We train the _Drosophila_ whole-brain model using the D-RTRL online algorithm implemented in
BrainScale (Wang et al., 2024). Unlike BPTT, D-RTRL computes gradients in a strictly forwardtime manner by propagating eligibility traces that accumulate local parameter sensitivities at each
time step. This makes the algorithm both scalable and memory-efficient.


For recurrent weights _wij_, the gradient is given by:


_∇wij_ _L ≈_        - _∂L_ ( _t_ ) _/∂ri_ ( _t_ ) _· ϵwij_ ( _t_ ) _,_ (5)


_t_

_ϵwij_ ( _t_ ) = _αi · ϵwij_ ( _t −_ 1) + (1 _−_ _αi_ ) _· f_ _[′]_ ( _xi_ ( _t_ )) _·_ sgn _ij_ _· rj_ ( _t_ ) _._ (6)


For encoding weights _Wk,i_ [enc][, the gradient is computed by:]


_∇Wk,i_ [enc] _[L ≈]_    - _∂L_ ( _t_ ) _/∂ri_ ( _t_ ) _· ϵWk,i_ [enc] [(] _[t]_ [)] _[,]_ (7)

_t_

_ϵWk,i_ [enc] [(] _[t]_ [) =] _[ α][i][ ·][ ϵ][W]_ [ enc] _k,i_ [(] _[t][ −]_ [1) + (1] _[ −]_ _[α][i]_ [)] _[ ·][ f][ ′]_ [(] _[x][i]_ [(] _[t]_ [))] _[ ·][ g][′]_ [(] _[y][i]_ [(] _[t]_ [))] _[ ·]_ [ FR] _k_ [neuropil] ( _t −_ 1) _._ (8)


For decay factors _αi_, which control temporal integration, the gradient is given by:


_∇αiL ≈_          - _∂L_ ( _t_ ) _/∂ri_ ( _t_ ) _· ϵαi_ ( _t_ ) _,_ (9)


_t_

_ϵαi_ ( _t_ ) = _αi · ϵαi_ ( _t −_ 1) + _ri_ ( _t −_ 1) _−_ _f_ ( _xi_ ( _t_ )) _._ (10)


Here, _xi_ ( _t_ ) = _Ii_ [conn] ( _t_ ) + _Ii_ [enc] ( _t_ ) is the total input to neuron _i_, _yi_ ( _t_ ) = [�] _k_ [FR] _k_ [neuropil] ( _t −_ 1) _· Wk,i_ [enc]

is the pre-ReLU encoding input, _f_ _[′]_ ( _·_ ) is the derivative of the activation function _f_, _g_ _[′]_ ( _y_ ) = **1** _y>_ 0
is the derivative of the ReLU function, _ϵθ_ ( _t_ ) represents the eligibility trace for parameter _θ_, and the
loss _L_ is defined as the mean squared error between recorded neural activity and simulated activity.


During training, the network states and all eligibility traces evolve in real time, updated locally at
each step without storing the full temporal history (Fig. 1E and Fig. S8). Once the loss is computed,
parameter gradients are obtained directly from the current eligibility traces.


6


FR [neuropil] _k_ ( _t_ ) =


_j∈Mk_ _[r][j]_ [(] _[t]_ [)] _[ ·][ n]_ _j,k_ [syn]

_,_ (4)

 
_j∈Mk_ _[n]_ _j,k_ [syn]


4 ONLINE FITTED WHOLE-BRAIN MODELS RECOVER RESTING-STATE

CRITICALITY


4.1 CONNECTOME-CONSTRAINED ONLINE OPTIMIZATION ENABLES WHOLE-BRAIN FITTING


Figure 2: **Scalability and training performance of the online fitting workflow.** (A) GPU memory consumption of our online fitting algorithm across different data lengths and batch sizes. (B)
Training loss comparison between the online fitting algorithm and BPTT when optimizing low-rank
recurrent weights. (C) Training loss comparison between connectome-constrained readout and linear readout across training epochs.


We first assessed the scalability of our _Drosophila_ online fitting framework. We found that even
with a small training batch, the memory consumption of BPTT exceeded the capacity of a 32 GB
GPU, and training failed regardless of the dataset length. In contrast, our online learning approach
successfully trained the model while exhibiting favorable scaling properties. Specifically, memory usage grew linearly with the training batch size but remained independent of the dataset length
(Fig. 2A). This property highlights a key advantage of our framework: by decoupling memory demand from sequence length, it enables fitting of large-scale whole-brain networks over biologically
realistic timescales.


We next evaluated the training performance of our online fitting approach in comparison with BPTT.
To ensure a fair comparison, we replaced the connectome-derived recurrent weight matrix with a
synthetic low-rank factorization (Appendix C). Under this setting, BPTT was able to successfully
train on a single GPU. We trained models for 200 epochs across datasets of varying lengths and compared their final training losses. The results show that our online learning approach achieves training
performance comparable to BPTT (Fig. 2B), while retaining its memory-efficiency advantages (SI).


Having established the computational efficiency of our online fitting framework, we next examined the role of biological constraints in shaping model performance. Specifically, we compared
our connectome-based readout mechanism (section 3.6) with an unconstrained linear readout (Appendix D). We found that the connectome-constrained readout consistently converges to a lower
training loss, whereas the purely linear readout fails to do so (Fig. 2C). This result highlights the
critical role of anatomical priors, which provide an inductive bias that enables the model to more
accurately capture the brain’s functional dynamics.


4.2 TRAINED MODEL REPRODUCES NEURAL DYNAMICS AND FUNCTIONAL CONNECTIVITY


We trained the connectome-constrained model to reproduce neural activity over the first 500 time
steps (“Train” phase in Fig. 3A). To assess its capacity for generalization beyond the training window, we simulated the model over longer timescales and compared its predicted activity with unseen
experimental data. Remarkably, the trained model was able to generate spontaneous oscillatory dynamics that closely matched those observed during training (“Test” phase in Fig. 3A). The oscillation
patterns were preserved across neuropil populations, and the model reproduced characteristic fluctuations in activity, including rising and falling phases evident in the experimental recordings (see
markers **a** and **b** in Fig. 3A). These results demonstrate that the model captures intrinsic resting-state
dynamics rather than overfitting to the training segment.


Beyond reproducing individual neuropil activity, we examined whether the model also preserved
coordinated interactions among neuropils across the whole brain. To this end, we evaluated functional connectivity (FC) patterns during both training and testing phases. On the training data, the


7


Figure 3: **Neural activity and functional connectivity in trained** _**Drosophila**_ **whole-brain mod-**
**els.** (A) Simulated neural activity generated in the whole-brain model trained on the first 500 time
steps and tested on the subsequent 500. (B) Comparison between ground-truth and model-predicted
functional activity across 73 neuropils.


model achieved near-perfect reconstruction of FC, with a correlation of 0.998 relative to the groundtruth experimental connectivity (Fig. 3B). Crucially, during testing on unseen data, the predicted
FC remained strongly aligned with empirical measurements (correlation = 0.556), outperforming
the direct similarity between the experimental test and training datasets themselves (correlation =
0.474). Furthermore, the model’s own predicted activity exhibited robust temporal consistency, as
reflected by a correlation of 0.750 between training and test segments of the generated sequences.


4.3 TRAINED SYNAPTIC WEIGHTS ALIGN WITH CONNECTOME STATISTICS


Figure 4: **Recurrent** **Weight** **distribution** **of** **trained** **whole-brain** **models.** (A) Distribution of
recurrent weights in models before and after training. (B) Distribution of recurrent weights in the
FlyWire _Drosophila_ connectome. (C) Q-Q plot comparisons of weight distributions between untrained and trained models relative to the connectome.


Next, we analyzed the distribution of recurrent synaptic weights in the trained whole-brain
_Drosophila_ models. Before training, weights were narrowly distributed within a limited range
(Fig. 4A). After training, the distribution broadened and developed a pronounced heavy tail, with a
small subset of connections becoming substantially stronger (Fig. 4A). This organization, with many
weak connections and few strong ones, mirrors the empirical pattern observed in the FlyWire connectome reconstruction (Dorkenwald et al., 2024), where synaptic connection counts span several
orders of magnitude and a minority of connections dominate in strength (Fig. 4B).


To quantitatively assess similarity, we rescaled both pre- and post-training model weights to the same
range as the experimental connectome and performed quantile–quantile (Q–Q) analysis. The posttraining weights exhibited a markedly higher correlation with the connectome distribution compared


8


**A** **Train** **Test** **B**


**B**


**Train** **Test** **B**


R=0.474 R=0.750


to the initial weights (Fig. 4C). In the Q–Q plot, points for the trained model clustered closely along
the _y_ = _x_ reference line, indicating that both the variability and quantile structure of the learned
distribution strongly matched the experimental data. This alignment was further corroborated by
higher correlations in the binned count distributions of weight values (Fig. S9).


4.4 NETWORK DYNAMICS EXHIBIT CRITICALITY IN THE RESTING STATE


**A** **B** **C**


Figure 5: **Analysis** **of** **critical** **dynamics** **in** **experimental** **data** **and** **model** **predictions.** (A)
Power-law distribution of avalanche durations in experimentally recorded neural activity, with fitted
exponent _α_ = 1.78 ( _R_ [2] = 0.916). (B) Power-law distribution of avalanche durations in modelpredicted activity from 138,639 neurons, with fitted exponent _α_ = 1.90 ( _R_ [2] = 0.912). (C) Evolution
of the magnitudes of the top 1000 eigenvalues as the training decreases.


A hallmark of resting-state neural activity is criticality, characterized by neuronal avalanches whose
duration distribution follows a power law, _P_ ( _D_ ) _∝_ _D_ _[−][α]_ . Across species, critical states are associated with an exponent _α_ close to 2, as reported in zebrafish (Ponce-Alvarez et al., 2018), rat cortex
(Beggs & Plenz, 2003), awake macaque (Gireesh & Plenz, 2008), and humans (Shriki et al., 2013).
Using the experimental calcium imaging data (Mann et al., 2017; Turner et al., 2021), we confirmed
that the resting-state _Drosophila_ brain also operates near criticality, exhibiting a power-law distribution of avalanche durations with a fitted exponent of _α_ = 1 _._ 78 ( _R_ [2] = 0 _._ 916; Fig. 5A).


Recording avalanche dynamics at single-neuron resolution across the entire brain is technically challenging in experiments. Our model provides a way to overcome this limitation. We analyzed the
firing rates of 138,639 neurons simulated over 1,400 time steps from a well-trained model (final loss
_≈_ 0 _._ 03). Importantly, the last 900 steps, unseen during training, were used for analysis. The model
robustly reproduced avalanche criticality, with a fitted exponent of _α_ = 1 _._ 90 ( _R_ [2] = 0 _._ 912; Fig. 5B).


To further probe the mechanistic basis of this phenomenon, we examined the spectral radius of
the recurrent connectivity matrix during training. As optimization progressed and the training loss
decreased, the spectral radius increased and asymptotically approached 1 (Fig. 5C). In dynamical
systems theory, a spectral radius near unity signifies operation at the edge of stability and near
criticality. These results demonstrated that online optimization not only improved data fitting but
also spontaneously drove the network from a stable initial regime toward critical dynamics, enabling
the model to replicate a fundamental feature of resting-state brain activity.


5 CONCLUSION


In conclusion, we presented a connectome-constrained, online learning framework that makes
whole-brain data fitting computationally tractable at cellular resolution. By updating parameters
strictly in forward time, our method reduces memory from scaling with sequence length to scaling only with the number of parameters, enabling optimization of a _Drosophila_ whole-brain network with over 130 _,_ 000 neurons and millions of synapses on a single GPU. Trained on resting-state
calcium recordings, the model reproduces neuropil-level dynamics and functional connectivity on
held-out data. Moreover, optimization yielded emergent structure–function alignment: Post-training
synaptic weights developed heavy-tailed statistics that more closely match the connectome measured
spine counts, and the network’s dynamics self-organized toward a critical regime. These results
indicate that fitting to whole-brain activity, under anatomical constraints, can recover organizing
principles of neural computation, and link circuit topology, parameter distributions, and population
dynamics within a single mechanistic model.


9


REFERENCES


Bilal Ahmed, Joshua D. Downer, Brian J Malone, and Joseph G Makin. Deep neural networks
explain spiking activity in auditory cortex. _PLOS Computational Biology_, 21(8):e1013334, 2025.


Katrin Amunts, Claude Lepage, Louis Borgeat, Hartmut Mohlberg, Timo Dickscheid, Marc-Etienne [´]
Rousseau, Sebastian Bludau, Pierre-Louis Bazin, Lindsay B Lewis, Ana-Maria Oros-Peusquens,
et al. Bigbrain: an ultrahigh-resolution 3d human brain model. _science_, 340(6139):1472–1475,
2013.


Andrea Banino, Caswell Barry, Benigno Uria, Charles Blundell, Timothy Lillicrap, Piotr Mirowski,
Alexander Pritzel, Martin J Chadwick, Thomas Degris, Joseph Modayil, et al. Vector-based
navigation using grid-like representations in artificial agents. _Nature_, 557(7705):429–433, 2018.


John M Beggs and Dietmar Plenz. Neuronal avalanches in neocortical circuits. _The_ _Journal_ _of_
_Neuroscience_, 23(35):11167–11177, 2003.


Guillaume Bellec, Shuqi Wang, Alireza Modirshanechi, Johanni Brea, and Wulfram Gerstner. Fitting summary statistics of neural data with a differentiable spiking network simulator. _Advances_
_in Neural Information Processing Systems_, 34:18552–18563, 2021.


Yazan N Billeh, Binghuang Cai, Sergey L Gratiy, Kael Dai, Ramakrishnan Iyer, Nathan W
Gouwens, Reza Abbasi-Asl, Xiaoxuan Jia, Joshua H Siegle, Shawn R Olsen, et al. Systematic integration of structural and functional data into multi-scale models of mouse primary visual
cortex. _Neuron_, 106(3):388–403, 2020.


Rodrigo M Cabral-Carvalho, Walter HL Pinaya, and Jo˜ao R Sato. A graph neural network approach
to investigate brain critical states over neurodevelopment. _Network Neuroscience_, pp. 1–16, 2025.


Guozhang Chen, Franz Scherr, and Wolfgang Maass. A data-based large-scale model for primary
visual cortex enables brain-like robust and versatile visual processing. _Science advances_, 8(44):
eabq7592, 2022.


Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of
gated recurrent neural networks on sequence modeling. _arXiv preprint arXiv:1412.3555_, 2014.


Steven J Cook, Travis A Jarrell, Christopher A Brittin, Yi Wang, Adam E Bloniarz, Maksim A
Yakovlev, Ken CQ Nguyen, Leo T-H Tang, Emily A Bayer, Janet S Duerr, et al. Whole-animal
connectomes of both caenorhabditis elegans sexes. _Nature_, 571(7763):63–71, 2019.


Zhuokun Ding, Paul G Fahey, Stelios Papadopoulos, Eric Y Wang, Brendan Celii, Christos Papadopoulos, Andersen Chang, Alexander B Kunin, Dat Tran, Jiakun Fu, et al. Functional connectomics reveals general wiring rule in mouse visual cortex. _Nature_, 640(8058):459–469, 2025.


Sven Dorkenwald, Arie Matsliah, Amy R Sterling, Philipp Schlegel, Szi-Chieh Yu, Claire E McKellar, Albert Lin, Marta Costa, Katharina Eichler, Yijie Yin, et al. Neuronal wiring diagram of an
adult brain. _Nature_, 634(8032):124–138, 2024.


Antonio J Fontenele, Nivaldo AP De Vasconcelos, Tha´ıs Feliciano, Leandro AA Aguiar, Carina
Soares-Cunha, B´arbara Coimbra, Leonardo Dalla Porta, Sidarta Ribeiro, Ana Jo˜ao Rodrigues,
Nuno Sousa, et al. Criticality between cortical states. _Physical_ _review_ _letters_, 122(20):208101,
2019.


Marco Fusc`a, Felix Siebenh¨uhner, Sheng H Wang, Vladislav Myrov, Gabriele Arnulfo, Lino Nobili,
J Matias Palva, and Satu Palva. Brain criticality predicts individual levels of inter-areal synchronization in human electrophysiological data. _Nature Communications_, 14(1):4736, 2023.


Felipe Gerhard, Tilman Kispersky, Gabrielle J Gutierrez, Eve Marder, Mark Kramer, and Uri Eden.
Successful reconstruction of a physiological circuit with known connectivity from spiking activity
alone. _PLoS computational biology_, 9(7):e1003138, 2013.


Sebastian Gerwinn, Jakob H Macke, and Matthias Bethge. Bayesian inference for generalized linear
models for spiking neurons. _Frontiers in computational neuroscience_, 4:1299, 2010.


10


Elakkat D Gireesh and Dietmar Plenz. Neuronal avalanches organize as nested theta-and
beta/gamma-oscillations during development of cortical layer 2/3. _Proceedings_ _of_ _the_ _National_
_Academy of Sciences_, 105(21):7576–7581, 2008.


John D Griffiths, Sorenza P Bastiaens, and Neda Kaboodvand. Whole-brain modelling: past,
present, and future. In _Computational_ _modelling_ _of_ _the_ _brain:_ _Modelling_ _approaches_ _to_ _cells,_
_circuits and networks_, pp. 313–355. Springer, 2021.


Eghbal A Hosseini, Martin Schrimpf, Yian Zhang, Samuel Bowman, Noga Zaslavsky, and Evelina
Fedorenko. Artificial neural network language models predict human brain responses to language
even after a developmentally realistic amount of training. _Neurobiology of Language_, 5(1):43–63,
2024.


Arnim Jenett, Gerald M Rubin, Teri-TB Ngo, David Shepherd, Christine Murphy, Heather Dionne,
Barret D Pfeiffer, Amanda Cavallaro, Donald Hall, Jennifer Jeter, et al. A gal4-driver line resource
for drosophila neurobiology. _Cell reports_, 2(4):991–1001, 2012.


Jonas Kubilius, Martin Schrimpf, Kohitij Kar, Rishi Rajalingham, Ha Hong, Najib Majaj, Elias Issa,
Pouya Bashivan, Jonathan Prescott-Roy, Kailyn Schmidt, et al. Brain-like object recognition with
high-performing shallow recurrent anns. _Advances in neural information processing systems_, 32,
2019.


Janne K Lappalainen, Fabian D Tschopp, Sridhama Prakhya, Mason McGill, Aljoscha Nern,
Kazunori Shinomiya, Shin-ya Takemura, Eyal Gruntman, Jakob H Macke, and Srinivas C Turaga.
Connectome-constrained networks predict neural activity across the fly visual system. _Nature_, 634
(8036):1132–1140, 2024.


Timothy P Lillicrap, Adam Santoro, Luke Marris, Colin J Akerman, and Geoffrey Hinton. Backpropagation and the brain. _Nature Reviews Neuroscience_, 21(6):335–346, 2020.


Kevin Mann, Courtney L. Gallen, and Thomas R. Clandinin. Whole-brain calcium imaging reveals
an intrinsic functional network in drosophila. _Current_ _biology:_ _CB_, pp. S0960982217308138,
2017.


Kenneth D Miller and Francesco Fumarola. Mathematical equivalence of two common forms of
firing rate models of neural networks. _Neural computation_, 24(1):25–31, 2012.


Aran Nayebi, Daniel Bear, Jonas Kubilius, Kohitij Kar, Surya Ganguli, David Sussillo, James J
DiCarlo, and Daniel L Yamins. Task-driven convolutional recurrent models of the visual system.
_Advances in neural information processing systems_, 31, 2018.


Giorgio Nicoletti, Samir Suweis, and Amos Maritan. Scaling and criticality in a phenomenological
renormalization group. _Physical Review Research_, 2(2):023144, 2020.


Seung Wook Oh, Julie A. Harris, Lydia Ng, Brent Winslow, Nicholas Cain, Stefan Mihalas, Quanxin
Wang, Chris Lau, Leonard Kuan, and Alex M. Henry. A mesoscale connectome of the mouse
brain. _Nature_, 508(7495):207–14, 2014.


Jordan O’Byrne and Karim Jerbi. How critical is brain criticality? _Trends in Neurosciences_, 45(11):
820–837, 2022.


Anagh Pathak, Dipanjan Roy, and Arpan Banerjee. Whole-brain network models: from physics to
bedside. _Frontiers in Computational Neuroscience_, 16:866517, 2022.


Adri´an Ponce-Alvarez, Adrien Jouary, Martin Privat, Gustavo Deco, and Germ´an Sumbre. Wholebrain neuronal activity displays crackling noise dynamics. _Neuron_, 100(6):1446–1459, 2018.


Philipp Schlegel, Yijie Yin, Alexander S Bates, Sven Dorkenwald, Katharina Eichler, Paul Brooks,
Daniel S Han, Marina Gkantia, Marcia Dos Santos, Eva J Munnelly, et al. Whole-brain annotation
and multi-connectome cell typing of drosophila. _Nature_, 634(8032):139–152, 2024.


Martin Schrimpf, Jonas Kubilius, Michael J Lee, N Apurva Ratan Murty, Robert Ajemian, and
James J DiCarlo. Integrative benchmarking to advance neurally mechanistic models of human
intelligence. _Neuron_, 108(3):413–423, 2020.


11


Woodrow L Shew and Dietmar Plenz. The functional benefits of criticality in the cortex. _The_
_neuroscientist_, 19(1):88–100, 2013.


Woodrow L Shew, Hongdian Yang, Thomas Petermann, Rajarshi Roy, and Dietmar Plenz. Neuronal avalanches imply maximum dynamic range in cortical networks at criticality. _Journal_ _of_
_neuroscience_, 29(49):15595–15600, 2009.


Philip K Shiu, Gabriella R Sterne, Nico Spiller, Romain Franconville, Andrea Sandoval, Joie Zhou,
Neha Simha, Chan Hyuk Kang, Seongbong Yu, Jinseop S Kim, et al. A drosophila computational
brain model reveals sensorimotor processing. _Nature_, 634(8032):210–219, 2024.


Oren Shriki, Jeff Alstott, Frederick Carver, Tom Holroyd, Richard NA Henson, Marie L Smith,
Richard Coppola, Edward Bullmore, and Dietmar Plenz. Neuronal avalanches in the resting meg
of the human brain. _Journal of Neuroscience_, 33(16):7079–7090, 2013.


Gaˇsper Tkaˇcik, Thierry Mora, Olivier Marre, Dario Amodei, Stephanie E Palmer, Michael J Berry,
and William Bialek. Thermodynamics and signatures of criticality in a network of neurons. _Pro-_
_ceedings of the National Academy of Sciences_, 112(37):11508–11513, 2015.


Maxwell H Turner, Kevin Mann, and Thomas R Clandinin. The connectome predicts resting-state
functional connectivity across the drosophila brain. _Current biology_, 31(11):2386–2394, 2021.


Chaoming Wang, Xingsi Dong, Zilong Ji, Jiedong Jiang, Xiao Liu, and Si Wu. Brainscale, enabling
scalable online learning in spiking neural networks. _bioRxiv_, pp. 2024–09, 2024.


Daniel LK Yamins, Ha Hong, Charles F Cadieu, Ethan A Solomon, Darren Seibert, and James J
DiCarlo. Performance-optimized hierarchical models predict neural responses in higher visual
cortex. _Proceedings of the national academy of sciences_, 111(23):8619–8624, 2014.


Tianfang Zhu, Dongli Hu, Jiandong Zhou, Kai Du, and Anan LI. Biologically constrained barrel cortex model integrates whisker inputs and replicates key brain network dynamics. In
_The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2025. URL [https:](https://openreview.net/forum?id=UvfI4grcM7)
[//openreview.net/forum?id=UvfI4grcM7.](https://openreview.net/forum?id=UvfI4grcM7)


12


A CONVERTING CALCIUM IMAGING DATA INTO FIRING RATE


The observed calcium signal _c_ ( _t_ ) is modeled as the convolution of an unobserved spike train _s_ ( _t_ )
(where _s_ ( _t_ ) _∈_ 0 _,_ 1) with an exponential calcium response kernel _k_ ( _t_ ) = _A · exp_ ( _−t/τ_ ) for _t_ _≥_ 0,
plus additive noise _ε_ ( _t_ ):

_c_ ( _t_ ) =                - _s_ ( _τ_ ) _· k_ ( _t −_ _τ_ ) + _ε_ ( _t_ ) _._ (S1)


_τ_


To recover the firing rate, we first solved for the most likely spike sequence _s_ ˆ( _t_ ) given the calcium
trace _c_ ( _t_ ) using a sparse deconvolution algorithm with an _l_ 1-norm constraint to enforce sparsity.
This estimated spike train was then convolved with a smoothing window function _w_ (∆ _t_ ) (e.g., a
Gaussian or rectangular window) to produce a continuous estimate of the instantaneous firing rate
f(t) in Hz:


                  _f_ ( _t_ ) = (ˆ _s · w_ )( _t_ ) = _s_ ˆ( _τ_ ) _· w_ ( _t −_ _τ_ ) _dτ._ (S2)


This method provides a higher temporal resolution estimate of neural activity by effectively inverting
the slow calcium dynamics, yielding a signal more suitable for driving rate-based network models.


B RECURRENT WEIGHT INITIALIZATION


The initial recurrent synaptic weights (section 3.4) were drawn from a truncated normal distribution,
_w_ _∼_ _ψ_ ( _µ_ = 0 _, σ_ = _σ_ 0 _, a_ = _−_ 2 _, b_ = 2) _,_ (S3)
where the truncation interval [ _−_ 2 _,_ 2] ensures numerical stability by preventing extreme outliers. The
standard deviation _σ_ ~~0~~ was set according to a variance-scaling principle widely used in deep learning
to stabilize activity at initialization. Specifically, it was computed as


where ˜ _σN_ [[] _[−]_ [2] _[,]_ [2]] = 0 _._ 87962566103423978 is the standard-deviation correction factor for the truncated
normal distribution on the interval [ _−_ 2 _,_ 2], and _n_ eff denotes the effective number of synaptic inputs per neuron. The latter was estimated as the total number of non-zero entries in the structural
adjacency matrix divided by two, reflecting the sparsity imposed by the connectome.


Importantly, only the non-zero synaptic weights specified by the connectome were initialized in
this way and treated as trainable parameters, ensuring that optimization remained constrained by
anatomical structure.


C LOW-RANK WEIGHT APPROXIMATION FOR RECURRENT CONNECTIVITY


We observed that training synaptic weights under connectome-constrained connectivity (Section 3.4)
creates a severe memory bottleneck for BPTT (Section 4.1). This limitation arises because gradient
computation requires storing a large number of intermediate states across both time and the densely
connected recurrent graph, causing memory usage to grow prohibitively with sequence length. To
address this challenge, we investigated a memory-efficient alternative based on a low-rank approximation of the recurrent connectivity matrix.


Specifically, the recurrent synaptic input is expressed as
**I** [conn] ( _t_ ) = **Wr** ( _t −_ 1) = **LRr** ( _t −_ 1) _,_ (S5)

where **W** _∈_ R _[N]_ _[×][N]_ denotes the full recurrent weight matrix, **L** _∈_ R _[N]_ _[×][k]_, **R** _∈_ R _[k][×][N]_ are low-rank
factors, and _N_ is the neuron number. During training, we set _k_ = 10, which drastically reduces the
number of parameters and the memory footprint while still enabling rich recurrent dynamics.


With this low-rank factorization, BPTT was able to successfully train the model on neural activity
fitting tasks for short to moderate sequence lengths (Fig. S1B). Nevertheless, even with this reduction, scalability remained limited: at batch size 32, training failed with an out-of-memory error once
the dataset length exceeded approximately 400 time steps.


13


_σ_ 0 =


�2 _/n_ eff
_,_ (S4)
_σ_ ˜ _N_ [[] _[−]_ [2] _[,]_ [2]]


D NEUROPIL FIRING RATE READOUT WITH A LINEAR TRANSFORMATION


In addition to the connectome-constrained readout mechanism (section 3.6), we also considered a
more flexible approach in which neuropil activity is obtained through a learned linear transformation
of neuron-level firing rates. Specifically, the firing rate of neuropil _k_ at time _t_, denoted FR [neuropil] _k_ ( _t_ ),
is computed as


where _Wk,j_ [out] [is a learnable weight mapping the firing rate of neuron] _[ j]_ [onto neuropil] _[ k]_ [,] [and] _[ b][k]_ [is a]
bias parameter. The ReLU activation enforces non-negativity of the predicted firing rate, consistent
with experimental observations that calcium-imaging-derived neuropil signals are bounded below
by zero.


This linear readout framework serves as an anatomically unconstrained baseline, in contrast to the
connectome-based aggregation rule. The comparison between the two readout strategies highlights
the role of anatomical priors in improving both accuracy and interpretability of whole-brain activity
fitting.


E SUPERIORITY OF ONLINE LEARNING FOR TRAINING BASELINE MODEL


To establish a performance benchmark, a Gated Recurrent Unit (GRU) network (Chung et al., 2014)
with 256 hidden units was implemented. The training of this baseline model revealed two critical
advantages of the D-RTRL online learning method (Wang et al., 2024) over BPTT: convergence
stability and memory efficiency.


Figure S1: **Comparison of BPTT and online learning in convergence and memory efficiency.**
(A) Training loss of the GRU baseline model. When trained with BPTT (blue), the model failed
to converge to a low-error solution, exhibiting unstable dynamics. In contrast, D-RTRL (orange)
enabled stable convergence to a significantly lower final loss, demonstrating its efficacy for this
task. (B) Memory usage of BPTT versus online learning. To evaluate GPU memory consumption,
we compared both methods using low-rank connections (as BPTT is infeasible with the full connectome). BPTT memory usage grew rapidly with sequence length, exceeding typical GPU limits
(dashed line) at a length of 400. The online learning method maintained substantially lower and
sequence-length-independent memory consumption.


**Convergence** **stability:** Initial attempts to train the GRU using BPTT were unsuccessful, as the
model failed to converge to a low-error solution (Fig. S1A). The BPTT loss curve exhibited high
variance, reflecting the optimization challenges for this long-horizon task. Conversely, D-RTRL
training resulted in stable convergence to a significantly lower final loss (Fig. S1A), enabling a fair
architectural comparison.


**Memory efficiency:** Beyond convergence, BPTT is limited by its substantial memory footprint. As
shown in Fig. S1B, BPTT memory consumption scales rapidly with sequence length, exceeding
GPU capacity for long sequences (with low-rank connections). This constraint would be prohibitive


14





 _,_ (S6)


FR [neuropil] _k_ ( _t_ ) = ReLU




 [�] _Wk,j_ [out] _[, r][j]_ [(] _[t]_ [) +] _[ b][k]_

_j_


at the full connectome scale. In contrast, online learning maintains low, largely sequence-lengthindependent memory usage, making it practical for long neural recordings.


These results demonstrate that online learning (D-RTRL) offers both convergence stability and memory efficiency advantages over BPTT, justifying its use for training and comparison in this study.


F BASELINE MODEL COMPARISONS


To evaluate the efficacy of different model architectures in capturing neural dynamics, we compared
our connectome-based model with a standard GRU network (256 hidden units). Both models were
trained using the D-RTRL online learning algorithm and evaluated under the same protocol: for each
test trial, the models were first warmed up for 250 time steps by feeding the ground-truth activity
at each step, followed by 750 steps of autoregressive prediction where each step’s input was the
model’s prediction from the previous step.


Figure S2: **Evaluation of connectome-based and artificial neural network models on neuropil**
**activity.** (A) Connectome-based network model. The model successfully maintains diverse and
realistic neural activity patterns throughout both the warmup and autonomous prediction phases,
demonstrating robust capture of the underlying dynamics. (B) GRU network with 256 hidden units.
The model’s generalization is poor, particularly on unseen test data. it collapses into simplistic limit
cycles during autonomous prediction on unseen test data, failing to sustain realistic dynamics. The
vertical dashed lines demarcate the evaluation stages: the red line separates the 250-step warmup
phase (left) from the 750-step autonomous prediction phase (right); the green line separates the first
500 steps (familiar training data) from the subsequent 500 steps (unseen test data).


**Qualitative analysis:** The GRU model exhibited poor generalization on unseen test data, frequently
collapsing into simplistic limit cycles (Fig. S2B). In contrast, our model sustained rich and diverse
activity patterns that more accurately mimicked the experimental data (Fig. S2A).


**Quantitative analysis:** The functional connectivity (FC) matrix derived from the GRU’s predictions
showed a low correlation with the empirical FC, with a Pearson correlation coefficient of _r_ = 0 _._ 211
(Fig. S3). This performance was substantially lower than the correlation achieved by our model
( _r_ = 0 _._ 556). Furthermore, avalanche analysis revealed that the GRU’s output failed to exhibit the


15


power-law scaling characteristic of neural criticality (Fig. S7A), indicating its inability to capture
this fundamental dynamical property.


Figure S3: **Comparison** **between** **ground-truth** **and** **GRU-predicted** **neuropil** **functional** **con-**
**nectivity.** The FC matrix derived from the GRU model’s predictions shows low correlation with
the empirical FC on test data( _r_ = 0 _._ 211), indicating poor accuracy in replicating the network-level
correlations observed in the neuropil data.


G PARAMETER STABILITY AND REPRODUCIBILITY ANALYSIS


To assess the robustness and reproducibility of the learned model—key concerns for large-scale,
high-capacity networks—we analyzed the stability of the final trained connectivity weights across
multiple independent runs with different initializations.


**Systematic** **Shift** **Induced** **by** **Learning:** Comparing the initial uniform distribution (Fig. S4A,
blue) with the trained distribution (orange) reveals a consistent, systematic shift. This indicates that
the model purposefully adjusts parameters to capture underlying data structure rather than randomly
memorizing training data.


**Cross-Initialization** **Consistency:** Models initialized from different prior distributions—a truncated normal distribution and a uniform distribution—converged to final weight profiles that were
highly consistent across neuropil (Fig. S4B). The two profiles are statistically indistinguishable,
indicating that the learned solution is not an artifact of a specific initialization scheme.


**Post-training consistency of neuropil weights:** Independent training runs starting from the same
initial uniform distribution yielded virtually identical final weight vectors (Fig. S4C). This demonstrates the high determinism and numerical stability of the training process itself.


16


Figure S4: **Stability** **and** **reproducibility** **of** **learned** **connectivity** **weights** **under** **different** **ini-**
**tializations.** (A) Systematic shift induced by learning. Neuropil weights initialized from a uniform
distribution (blue) are compared with the trained weights (orange). The consistent shift demonstrates that learning systematically adjusts parameters to capture data structure rather than random
memorization. (B) Cross-initialization consistency. Final weights for each neuropil are shown for
models initialized with a truncated normal distribution (blue) versus a uniform distribution (orange).
Despite different starting points, the final weight profiles across brain regions are highly similar. (C)
Post-training consistency of neuropil weights. Final weights from two independent training runs,
both initialized from the same uniform distribution, are plotted. The near-perfect overlap demonstrates that the training process is highly deterministic and reproducible. In both panels, the x-axis
represents different neuropils, and the y-axis represents the final learned neuropils weight values.


17


These results collectively demonstrate that our model robustly converges to a unique and stable
parametric solution. This convergence is independent of the initial conditions, which effectively
mitigates concerns of overfitting to idiosyncrasies of the training set and underscores the reliability
of the parameters and the insights derived from them.


H DISTRIBUTION OF TIME CONSTANTS


The intrinsic time constant of individual neurons is a fundamental parameter governing their temporal dynamics. To understand how learning shapes the network’s timescales, we analyzed the timeconstant distribution across all single neurons in the model at three stages: initial (pre-training),
trained (post-training), and the per-neuron change between them.


Figure S5: **Evolution of intrinsic time constants during learning.** (A) Initial distribution of time
constants before training. The time constants are initialized within a biologically plausible range.
(B) Distribution of time constants after model training. The learned time constants remain within
a biologically plausible range, indicating that the model maintains realistic dynamical timescales.
(C) Distribution of per-neuron changes in time constants (post-training minus pre-training). Most
neurons exhibit relatively small changes, while a minority show substantial increases or decreases,
indicating targeted adaptation of temporal processing in specific subsets of units.


Both the initial and learned time constant distributions (Fig. S5A, B) span ranges consistent with
biological observations, confirming that the model’s dynamics operate within physiologically relevant timescales. The distribution of per-unit changes (Fig. S5C) reveals that the majority of neurons
underwent only modest adjustments, with a small subset exhibiting more pronounced shifts toward
either longer or shorter time constants. This pattern suggests that learning selectively modulates
the temporal dynamics of specific neuronal subpopulations, rather than uniformly shifting all units,
potentially enabling the model to capture multi-timescale dynamics essential for generating realistic
neural activity patterns.


I POST-TRAINING WEIGHTS EXHIBIT HEAVY-TAILED CHARACTERISTICS


The statistical distribution of synaptic weights is a fundamental characteristic of biological neural
networks. We analyzed whether our model, after training, recovers a key statistical feature observed
in the Drosophila connectome: a heavy-tailed distribution of connection strengths.


The empirical Drosophila connectome exhibits a heavy-tailed, approximately scale-free distribution
of synaptic weights, as evidenced by the power-law fit to the tail of the absolute weight distribution
(Fig. S6A). Remarkably, our model’s weight distribution evolves toward a similar structure through
learning. Starting from its initial state, the distribution of absolute model weights gradually develops
a pronounced heavy tail as training loss decreases (Fig. S6B). This convergence suggests that the
learning process not only optimizes for task performance but also implicitly drives the network’s
connectivity toward a statistical organization that mirrors a fundamental feature of biological brain
networks.


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


Figure S6: **Heavy-tailed** **weight** **distributions** **emerge** **during** **training** (A) Weight distribution
of the Drosophila connectome. (Top) Histogram of connection synapse count. (Bottom) Probability density of the absolute connection weights. The tail region is well-fit by a power-law function
(dashed line) indicating a heavy-tailed scale-free structure. (B) Evolution of the model’s weight distribution during training. (Top) Histograms show the progression of the model weight distribution
from initialization through different stages of training (with decreasing loss). (Bottom) The corresponding probability densities of the absolute weights. As training progresses and loss decreases,
the distribution of absolute weights progressively develops a heavier tail and becomes better approximated by a power-law fit, converging toward a statistical structure resembling the biological
connectome.


J AVALANCHE ANALYSIS OF NEURAL DYNAMICS


J.1 SIGNAL PROCESSING AND AVALANCHE DETECTION


To analyze the critical dynamics of neural activity, time series were binarized and neuronal
avalanches were detected using established methods.


**Signal** **binarization:** Activity traces _xi_ ( _t_ ) for each unit (neuron or neuropil) were converted to
binary events _bi_ ( _t_ ) by applying a unit-specific threshold:


_bi_ ( _t_ ) = �1 _,_ if _xi_ ( _t_ ) _>_ 3 _σi,_
0 _,_ otherwise _,_


where the baseline noise level _σi_ was robustly estimated from the median absolute deviation (MAD)
of the signal.


**Avalanche detection:** Avalanches were defined as contiguous periods of global activity, where at
least one unit was active ( _A_ ( _t_ ) = [�] _i_ _[b][i]_ [(] _[t]_ [)] [=] [1][).] [For each avalanche, its duration] _[ D]_ [ (in time steps)]

was recorded.


**Power-law fitting:** To assess scale-free criticality, we tested whether the avalanche duration distribution _P_ ( _D_ ) followed a power law, _P_ ( _D_ ) _∝_ _D_ _[−][α]_ . The scaling exponent _α_ was estimated via
linear regression in log-log space.


J.2 AVALANCHE DURATION DISTRIBUTIONS REVEAL DIVERGENT DYNAMICS


The analysis of avalanche durations highlights a fundamental difference in the dynamical regimes
captured by the GRU baseline and our model.


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


Figure S7: **Avalanche** **duration** **distributions** **of** **the** **GRU** **baseline** **and** **our** **model** (A) GRU
model. The distribution of avalanche durations does not follow a power law, indicating a lack of
critical, scale-free dynamics. (B) Our model (early training). The distribution is well-fit by a power
law with an exponent _α ≈_ 3 _._ 86 (red line). This high exponent may suggest a supercritical dynamical
regime.


As shown in Fig. S7A, the avalanche durations generated by the GRU baseline do not adhere to a
power-law distribution, failing to capture the scale-free signature of neural criticality. In contrast, our
model, even in early training stages, produces avalanches whose duration distribution is consistent
with a power law (Fig. S7B). The fitted exponent of _α ≈_ 3 _._ 86 is notably higher than typical critical
exponents, potentially indicating that the network operates in a supercritical regime characterized
by amplified and prolonged activity cascades. This divergence underscores our model’s enhanced
capability to generate biologically plausible, collective neural dynamics compared to the standard
RNN baseline.


K APPLYING SUGAR SENSORY STIMULATION TO THE MODEL


Our model architecture and training framework are fully compatible with fitting stimulus-driven
or task-based neural activity. The key distinction from modeling resting-state activity lies in the
introduction of additional parameters to represent external stimuli, while the core training paradigm
remains unchanged.


Furthermore, a model trained on resting-state data can be directly used to probe its dynamical response to external inputs. To illustrate this, we conducted a simulation in which a simulated sugar
stimulus was applied to the left gustatory receptor neurons within the model, analogous to experimental paradigms used to study taste processing (Shiu et al., 2024). The consequent activity changes
in downstream mouthpart motor neurons (MN9) were then monitored and analyzed.


The model successfully captured a basic sensorimotor transformation: activation of the appetitive
taste pathway led to excitation of the motor neurons. A key observation was a pronounced lateral
asymmetry in the motor response. Following unilateral stimulation on the left side, the contralateral
(right) MN9 neuron exhibited a markedly stronger increase in firing rate compared to the ipsilateral
(left) MN9 neuron, as detailed in Table S1.


Table S1: Motor neuron response to simulated sugar stimulation

cell type side firing rate (no stimulus) firing rate (with stimulus) increment magnification


MN9 Right 0.10054 5.35326 5.25272 53.24
MN9 Left 0.09425 2.23371 2.13946 23.70


The **increment** is defined as the difference between the average firing rate during the stimulus period
and the average firing rate during the baseline (no stimulus) period. The **magnification** is the ratio
of the average firing rate with stimulus to the average firing rate without stimulus. This demonstration confirms that the resting-state-trained model retains a functionally structured input-output map,


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


capable of generating biologically plausible, stimulus-specific motor dynamics under connectome
constraints.


L ADDITIONAL SUPPLEMENTARY FIGURES


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


/01234 41567897:;


Figure S8: **Two-stage training pipeline.** The model was trained using an online learning framework consisting of a warm-up phase and a prediction phase. This framework updates synaptic
weights with a combination of instantaneous loss and eligibility traces, thereby avoiding the high
memory cost of backpropagation through time. In the warm-up phase, the model received the actual
neuropil firing rate from the previous time step to predict the current rate. The resulting prediction error was used to compute gradients, which were accumulated over multiple steps before each
weight update. In the prediction phase, the model generated autonomous activity: initialized from
experimental data, it then used its own predicted output from the previous step as input. At each
time step, the prediction was compared against the target sequence, training the model to sustain
its dynamics independently. The online framework enabled efficient and scalable weight updates
throughout this recursive process.


22


target FR [!$#]


target FR [!"#]


target FR [!]


target FR [!$%]


Update


target FR [!$&]


Update


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


Figure S9: **Comparison** **of** **bin-count** **correlations** **within** **connectivity** **groups.** (A) Correlation between the trained model’s recurrent weight distribution and the empirical connectome weight
distribution. The scatter plot shows the correspondence of frequency counts across identical valuerange bins, with both axes representing log-transformed frequencies (log(bin count + 1)). The
red dashed line indicates the linear regression fit, yielding a Pearson correlation of _R_ = 0 _._ 904,
demonstrating a strong alignment between the trained and biological distributions. (B) Correlation
between the untrained weight distribution and the connectome distribution. The red dashed regression fit shows a Pearson correlation of _R_ = 0 _._ 479, indicating only modest similarity prior to training
and highlighting the emergence of biologically consistent weight patterns through optimization.


23
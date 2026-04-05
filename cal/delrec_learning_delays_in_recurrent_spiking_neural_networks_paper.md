# DELREC: LEARNING DELAYS IN RECURRENT SPIKING
### NEURAL NETWORKS.


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Spiking neural networks (SNNs) are a bio-inspired alternative to conventional
real-valued deep learning models, with the potential for substantially higher energy efficiency. Interest in SNNs has recently exploded due to a major breakthrough: surrogate gradient learning (SGL), which allows training SNNs with
backpropagation, strongly outperforming other approaches. In SNNs, each
synapse is characterized not only by a weight but also by a transmission delay. While theoretical works have long suggested that trainable delays significantly enhance expressivity, practical methods for learning them have only recently emerged. Here, we introduce “DelRec”, the first SGL-based method to
train axonal or synaptic delays in recurrent spiking layers. DelRec leverages
a differentiable interpolation technique to handle non-integer delays with welldefined gradients at training time. We show that trainable recurrent delays outperform feedforward ones, leading to new state-of-the-art (SOTA) on two challenging temporal datasets (Spiking Speech Command, an audio dataset, and Permuted Sequential MNIST, a vision one), and match the SOTA on the now saturated Spiking Heidelberg Digit dataset using only vanilla Leaky-Integrate-andFire neurons with stateless (instantaneous) synapses. Our results demonstrate
that recurrent delays are critical for temporal processing in SNNs and can be
effectively optimized with DelRec, paving the way for efficient deployment on
neuromorphic hardware with programmable delays. Our code is available at
[https://anonymous.4open.science/r/Recdel-4175.](https://anonymous.4open.science/r/Recdel-4175)


1 INTRODUCTION


Inspired by the architecture and dynamics of biological neural networks, Recurrent Spiking Neural Networks (RSNNs) provide a compelling and energy-efficient framework for processing timevarying data. Their recurrent structure enables them to maintain an internal state, integrating information over extended periods, an essential capability for tasks involving temporal dependencies,
such as speech recognition and time-series prediction (Bellec et al., 2018). However, despite their
promise, RSNNs remain underutilized in machine learning due to significant training challenges,
particularly the pervasive issues of vanishing and exploding gradients.


Recent advancements have sought to address these limitations by enhancing the models of spiking
neurons (Yin et al., 2021; Bittar & Garner, 2022; Baronig et al., 2025). Innovations such as adaptive
leaky integrate-and-fire (AdLIF) models and other sophisticated neuron dynamics have led to notable improvements, achieving state-of-the-art performance on widely used spiking neural network
benchmarks. Yet, these neuron-centric approaches put their emphasis on modulating current inputs,
rather than reactivating past signals, which limits their ability to model extended temporal dynamics.


An alternative approach involves incorporating transmission delays in synaptic connections, a feature observed in biological systems. In the brain, these delays are modulated by myelin, an insulating
sheet around axons that accelerates conduction speeds. Evidence further indicates that myelin levels
are plastic (Monje, 2018). Delays enhance network expressivity: neurons detect coincident spike
time latencies, which, in the presence of heterogeneous delays, correspond to arbitrary spike onset
latency sequences. While previous studies have successfully demonstrated the benefits of using delays in feedforward connections (Shrestha & Orchard, 2018; Sun et al., 2022; 2023; Timcheck et al.,
2023; Hammouamri et al., 2024; Deckers et al., 2024; G¨oltz et al., 2025; Ghosh et al., 2025), their


1


potential in recurrent connections remains largely unexplored. Recurrent delays could offer even
greater advantages, facilitating self-sustained activity (see Fig. 1A), modeling long-term dependencies, and supporting complex patterns like oscillations and polychronization. Theoretical work by
Izhikevich demonstrated that recurrent delays transform a neuron’s differential equation, expanding the range of possible solutions and enabling richer dynamics. Additionally, recurrent delays
may mitigate gradient challenges by implementing temporal skip connections, improving gradient
propagation during training (see Fig 1B).


|Vanilla RSNN B|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|
|**Wff**|**Wff**|**Wff**|**Wff**||
|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|
|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|


|S S S S<br>Wrec Wrec Wrec<br>H V H V H V H V|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**Wff**|**Wff**|**Wff**|**Wff**||
|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|**H**<br>**V**<br>**S**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**<br>**H**<br>**V**<br>**S**<br>**Wrec**|
|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|**X**<br>**X**<br>**X**<br>**X**|


Figure 1: **A** : The optimization of a single delay in a recurrent connection can transform two recurrently connected neurons in a pattern generator. Two different behaviors of two neurons with the
same inputs. Each neuron is recurrently connected to itself and to the other neuron, with a weight
equal to 1. The recurrent connections each have a delay, indicated by the circled number on the
connection. The neurons spike if they receive inputs strictly superior to 1 spike. At time _t −_ 1, the
neurons do not receive any input. The blue neuron receives an input spike at times _t_ and _t_ + 3,
while the pink neuron only receives two spikes at time _t_ + 1. _Top_ : The inputs trigger the firing of
one spike per neuron, working as a coincidence detector for spikes reaching the two neurons in a
short time interval. _Bottom_ : When the delay of the pink neuron’s recurrent connection (blue arrow)
is increased from 1 to 3 time steps, the same input triggers a regular and sustained firing pattern.
**B** : Delays in recurrent connections reduce the risks of exploding or vanishing gradients by bridging
distant time steps. Computational graphs of a vanilla RSNN with a intrinsic delay of 1 time step
in all recurrent connections ( _Top_ ), and of a RSNN with different and longer delays in the recurrent
connections ( _Bottom_ ).


To date, however, only a handful of studies have explored the potential of delays in recurrent connections, and even fewer have focused on learning optimal delay configurations. Recent approaches
have introduced algorithms to optimize these temporal parameters, demonstrating promising improvements in temporal tasks. For instance, Xu et al. achieved state-of-the-art results by learning
a single recurrent delay parameter per layer using backpropagation. Their approach selects delays
from a fixed set via a softmax function with a decreasing temperature, showcasing the potential of
more flexible, parameterized methods. To the best of our knowledge, only M´esz´aros et al. (2025)
have proposed an algorithm specifically designed to learn optimal delays in recurrent connections.
Their method, tailored for the EventProp algorithm (Wunderlich & Pehle, 2021), leverages exact
gradient computation. However, it inherits common limitations of EventProp, including scalability


2


**Coincidence detection**


**t**


**t+1**


**t+2**


**t+3**


**Pattern generation**


**Delayed RSNN**


**t**


**t+1**


**t+2**


**t+3**


challenges and suboptimal performance on real-world temporal benchmarks. Currently, all state-ofthe-art spiking approaches on these benchmarks rely on surrogate gradient learning (SGL).


In this paper, we introduce ”DelRec”, the first method to train axonal or synaptic delays in recurrent
connections using surrogate gradient learning (SGL) and backpropagation. Our method operates
in discrete time and eliminates the need to predefine a maximum delay range. During training,
we relax the constraint of integer delays by employing a differentiable interpolation process, then
round delays to the nearest integer for inference. Our approach is implemented using the Pytorchbased Spikingjelly library (Fang et al., 2023), and is compatible with any spiking neuron model. We
achieved new state-of-the-art results on two challenging temporal datasets (Spiking Speech Command, an audio dataset, and Permuted Sequential MNIST, a vision one), even when using simple
Leaky-Integrate-and-Fire neurons. Additionally, our study is the first to combine the optimization
of feedforward delays using DCLS (Hammouamri et al., 2024) and delays in recurrent connections.
We believe this work establishes a foundation for leveraging delays in recurrent networks and provides an accessible tool to explore the potential improvements offered by such methods. Our results
highlight the essential role of recurrent delays in temporal processing for spiking neural networks
(SNNs). Using DelRec, these delays can be optimized effectively, opening new opportunities for
deployment on neuromorphic hardware equipped with programmable delays.


2 METHODS


2.1 NEURON MODEL


Most spiking neuron models can be described by three discrete-time equations (Fang et al., 2021):


_H_ [ _t_ ] = _f_ ( _V_ [ _t −_ 1] _, I_ [ _t_ ]) _,_ (1)
_S_ [ _t_ ] = Θ( _H_ [ _t_ ] _−_ _V_ th) _,_ (2)

        - _H_ [ _t_ ] _·_ (1 _−_ _S_ [ _t_ ]) + _V_ reset _· S_ [ _t_ ] _,_ if hard reset
_V_ [ _t_ ] = _,_ (3)
_H_ [ _t_ ] _−_ _V_ th _· S_ [ _t_ ] _,_ if soft reset


where _f_ is the neuronal charge function (which depends on the neuron model), _I_ [ _t_ ] is the input current, _H_ [ _t_ ] is the membrane potential after charging but before firing, _V_ [ _t_ ] is the membrane potential
after firing, and _S_ [ _t_ ] is the output spike. _V_ th in Eq. 2 is the threshold, and _V_ reset in Eq. 3 is the reset
potential. Θ( _x_ ) is the Heaviside step function, i.e. Θ( _x_ ) = 1 for all _x_ _≥_ 0, otherwise Θ( _x_ ) = 0.
We use the surrogate gradient method () defining Θ _[′]_ ( _x_ ) = _σ_ _[′]_ ( _x_ ) during the backward step, where _σ_
is the surrogate function.


An example of a spiking neuron model, which is popular due to its simplicity, is the leaky integrateand-fire neuron (LIF). The neuronal charge equation for the LIF neuron is


where _τ_ is the membrane time constant. We used the LIF neuron in all our experiments. Yet our
method is compatible with any spiking neuron model that fits in the Eq. 1- 3 formalism.


Our method is essentially a way to compute _I_ in the presence of delayed recurrent connections. The
total input current is the sum of the feedforward input _X_ and the recurrent one _X_ [rec] :


_I_ [ _t_ ] = _X_ [ _t_ ] + _X_ [rec] [ _t_ ] (5)


In a vanilla Recurrent Spiking Neural Network (RSNN), the outputs of each neuron of a layer _L_ at
time _t_ are connected to the neurons of the same layer _L_ at time _t_ + 1. If we denote _{Sj_ ( _t_ ) _, j_ _∈_ ( _L_ ) _}_
the spikes emitted by all neurons of layer _L_ at time _t_, then we have for a neuron _i_ of the same layer:


_Xi_ _[rec]_ [ _t_ ] =             - _wij_ [rec] _[S][j]_ [[] _[t][ −]_ [1]] (6)

_j∈_ ( _L_ )


where _{w_ [rec] _}_ are the local recurrent weights of layer _L_ . We extend this regular definition of a RSNN
by allowing neurons to project their outputs to neurons of the same layer, but with a longer delay.
To this end, we set parameters _{dj_ _∈_ N _, j_ _∈_ _L}_ to model the time delay in a recurrent connection.


3


_f_ ( _V_ [ _t −_ 1] _, I_ [ _t_ ]) = (1 _−_ [1]


(4)
_τ_ _[·][ I]_ [[] _[t]_ []] _[,]_


[1] [1]

_τ_ [)] _[ ·][ V]_ [ [] _[t][ −]_ [1] +] _τ_


More specifically, if neuron _j_ in layer _L_ emits a spike _Sj_ ( _t_ ) at time _t_, then neuron _i_ in layer _L_ will
receive a recurrent input _wij_ [rec] _[S][j]_ [(] _[t]_ [)][ at time] _[ t]_ [ + 1 +] _[ d][j]_ [.] [As a result, we modify Eq. 6:]

_Xi_ [rec][[] _[t]_ [] =]            - _wij_ [rec] _[S][j]_ [[] _[t][ −]_ [(1 +] _[ d][j]_ [)]] (7)

_j∈_ ( _L_ )


Here, for simplicity, we assume an identical delay for all outgoing connections of a given neuron.
This is referred to as “axonal delay” in the literature, and we will use this setting in all our experiments. Yet our method/code is also compatible with synaptic delays (i.e., one different delay for
each synapse). In the last equation, we used the arbitrary convention that a recurrent connection has
a minimum delay of 1 time-step. Therefore, in our method, a delay parameter _d_ = 0 leads to an
effective delay of 1 in the recurrent connection.


2.2 LEARNING DELAYS IN RECURRENT CONNECTIONS


Let’s consider one layer of _N_ different neurons, with input sequences of temporal dimension _T_ . In
order to learn the delay parameters _{dj_ _∈_ N _, j_ = 1 _, ..., N_ _}_, we take a ”future-oriented” perspective:
when a neuron _j_ fires a spike at date _t_, we will schedule an input _wij_ [rec] [at] [date] _[t]_ [ + 1 +] _[ d][j]_ [for] [all]
neurons _i_ of the same layer. To this end, we consider in practice _X_ _[rec]_ _∈_ R _[N]_ _[×][T]_ as a scheduling
matrix storing weighted spikes for future time steps. More specifically:

_Xi_ [rec][[] _[t]_ [ +] _[ τ]_ [] =]             - _wij_ [rec] _[S][j]_ [[] _[t]_ []] (8)

_j|_ 1+ _dj_ = _τ_


In this framework, the parameters we learn in our method are the set of delays _d_ and the weights _w_ .


However, for the purpose of optimization, we consider real-valued delays _{dj_ _∈_ R _, j_ = 1 _, ..., N_ _}_,
which leads to a modification of Eq. 8. If a spike is scheduled at date _t_ + 1 + _d ∈_ R, we temporally
spread the prediction over time steps around _t_ + 1 + _d_, with a triangle function _hσ,d_ (KhalfaouiHassani et al., 2023) with a width parameter _σ_ . More specifically, for all current time steps _t_ we
consider the spread prediction at the target date _t_ + _τ_ :


_[−]_ [(1 +] _[ d]_ [)] _[|]_
_hσ,d_ ( _τ_ ) = max(0 _,_ [1 +] _[ σ][ −|][τ]_ ) (9)

(1 + _σ_ ) [2]


and we decrease the parameter _σ_ throughout training down to 0, as depicted in Figure 2C, so that
by the end of training _h_ 0+ _,d_ ( _τ_ ) leads to a linear interpolation between the two closest integer delay
positions.


Using this method, at each time step _t_, we update future time steps of the scheduling matrix. So the
prediction at a future time _t_ + _τ_ is updated (in the case of axonal delays) with:


where _dj_ is the axonal delay in the recurrent connection between neurons _j_ and _i_ .


One can notice in Eq.15 that the function _hσ,d_ ( _τ_ ) has a finite support supp( _hσ,d_ ( _τ_ )), depending only
of _σ_ and _d_ . Indeed:

_∀τ,_ _hσ,d_ ( _τ_ ) = 0 _⇔_ _τ_ _∈_ supp( _hσ,d_ ) = �(1 + _d_ ) _−_ (1 + _σ_ ) ; (1 + _d_ ) + (1 + _σ_ )�

=             - _d −_ _σ_ ; 2 + _d_ + _σ_             - (12)


So, it is in fact sufficient to schedule recurrent inputs only in supp( _hσ,d_ ), and as _σ_ decreases during
training, the range of time steps when we can schedule recurrent inputs becomes narrower. At the
scale of a layer, we need to schedule inputs for multiple neurons at the same time, which means it
suffices to compute and schedule inputs for a limited range of time steps **E**, such that:

_τ_ _∈_ **E** ( _σ, D_ ) =                - supp( _hσ,d_ )


_d_ _∈_ _D_


4


_Xi_ [rec][[] _[t]_ [ +] _[ τ]_ []] _[ ←]_ _[X]_ _i_ [rec][[] _[t]_ [ +] _[ τ]_ [] +]


_←_ _Xi_ [rec][[] _[t]_ [ +] _[ τ]_ [] +]


_N_

- _wij_ [rec] _[·][ h][σ]_ epoch _[,d]_ _j_ [(] _[τ]_ [)] _[ ·][ S][j]_ [[] _[t]_ []] (10)

_j_ =1


- _wij_ [rec] _[·]_ [ max(0] _[,]_ [1 +] _[ σ][ −|][τ]_ _[−]_ [(1 +] _[ d][j]_ [)] _[|]_

(1 + _σ_ ) [2]

_j_ =1


_N_


) _· Sj_ [ _t_ ] (11)
(1 + _σ_ ) [2]


### C


Figure 2: **A** : At time step _t_, the neurons in the studied layer receive weighted spikes from the
previous layer, which are then summed with the inputs scheduled for time step _t_ in the scheduling
matrix.. The subsequent evolution of the internal state of the neuron (membrane potential) may
produce output spikes. **B** : Each neuron of the layer receives a weighted sum of the output spikes,
and schedules it on a spread of future dates determined by its delay parameter _d_ and the spread of the
epoch _σ_ . Spread values are represented by the purple gradients. **C** : We modify the spread function
at each epoch by reducing its _σ_ . At the beginning of training, the scheduled values are widely spread
around the true delay _d_ . When the training ends, _σ_ is close to 0 and the spread function only performs
linear interpolation between the closest integers from the floating delay. Then we manually round
the delay to the closest integer in evaluation mode.


with _N_ the number of neurons in the layer, and _D_ = _{dj_ ; _j_ = 1 _,_ _..._ _, N_ _}_ . In practice, as _σ_ is
decreasing to 0 in Eq. 12, we ignore the lower bound of **E**, and we approximate this set with:


**E** ˜ ( _σ, D_ ) =      - 0 ;      - 1 + max [+ (1 +] _[ σ]_ [)] �� _⊃_ **E** ( _σ, D_ ) _._ (13)
1 _≤j≤N_ _[d][j]_


In other words, we only need to compute and store _hσ,d_ ( _τ_ ) for _τ_ in **E** [˜] ( _σ, D_ ), so the scheduling
matrix _X_ _[rec]_ has in fact a dimension of _N ×_ dim( **E** [˜] ( _σ, D_ )), then at time _t_, the future recurrent input
at time _t_ + _τ_ is updated with Eq. 11. We use _X_ [rec] as a buffer with a pointer mechanism in order to
efficiently schedule the future recurrent inputs (see Algorithm 1).


The initial value of _σ_ allows the recurrent connections to capture broad temporal dependencies at
the beginning of training and leads to a loose optimization of delays on long time scales, while the
subsequent refining of _σ_ during training pushes the delay parameters towards more precise locations.
A similar strategy was used in (Hammouamri et al., 2024). The described method is illustrated in
Fig. 2 and leads to Algorithm 1, which describes the update of the internal parameters of our neuron.


5


3 RESULTS


3.1 STATE-OF-THE-ART PERFORMANCE ON AUDIO AND VISION TASKS


We evaluated our method on two state-of-the-art datasets: the SSC (Spiking Speech Commands) and
the PS-MNIST (Permuted Sequential MNIST). The SSC dataset is a spiking audio dataset which
demands leveraging temporal patterns in spike trains to reach good classification accuracies. While
it is one of the most widely used datasets in the SNN community for benchmarking models’ temporal
processing capabilities, it also stands out as one of the largest, featuring over 100k samples across 35
classes of spoken commands. It is worth noting that this dataset has dedicated training, validation
and test sets, and is far from saturated (with best accuracies around 80%). The PS-MNIST is a
vision dataset which is obtained by flattening all images of the MNIST (28 _×_ 28) into one sequence
(1 _×_ 784), and permuting the pixel positions. This transformation requires to integrate long range
dependencies, making the PS-MNIST a reference benchmark to evaluate recurrent SNNs.


For both datasets, our models do not include any normalization layers, and training is performed
without data augmentation. Our implementation builds upon the code of Xu et al.. For the SSC
dataset, we used 3 fully connected hidden layers, with 256 neurons per layer, and we train our model
on 3 different seeds. For the PS-MNIST, we use one layer of 64 neurons, then 2 layers of 212 neurons, and we only test one seed as all the previous state-of-the-art models on the dataset. Complete
implementation details and hyperparameters are provided in A.2.5. Table 1 summarizes the accuracies of most competitive spiking, LIF-derived models on both datasets. We deliberately leave out of
this table models that rely on substantially more complex neuron models, such as multi-compartment
neurons (Zheng et al., 2024; Chen et al., 2024), attention or GRU based neurons (Dampfhoffer et al.,
2022; Wang et al., 2024), whose additional mechanisms make direct comparison less meaningful [1] .


Table 1: **Classification accuracy on SSC and PS-MNIST datasets**, ranked by accuracy.


Model Rec. Rec. Delays Ff. Delays LIF Param Test Acc. [%]


**SSC**
Adaptive RSNN (Yin et al., 2021) ✓ 0.78M 74.20%
EventProp (M´esz´aros et al., 2025) ✓ ✓ _∼_ 5M [a] 76.1 _±_ 1.0%
RadLIF (Bittar & Garner, 2022) ✓ 3.9M 77.40%
cAdLIF (Deckers et al., 2024) ✓ 0.35M 77.60%
SE-adLIF (Baronig et al., 2025) ✓ 1.6M 80.44 _±_ 0.26%
DCLS (Hammouamri et al., 2024) ✓ ✓ 2.5M 80.69 _±_ 0.21%
ASRC-SNN (Xu et al.) ✓ ✓ ✓ 0.37M 81.54%*
SiLIF (Fabre et al., 2025) ✓ 0.35M 82.03 _±_ 0.25%
**DelRec (Rec.** **and Ff.** **delays)** _**Ours**_ ✓ ✓ ✓ ✓ **0.55M** **82.19** _±_ **0.16%**
**DelRec (only Rec.** **delays)** _**Ours**_ ✓ ✓ ✓ **0.37M** **82.58** _±_ **0.08%**


**PS-MNIST**
GLIF (Yao et al., 2022) ✓ 0.15M 90.47%
Adaptive RSNN (Yin et al., 2021) ✓ 0.15M 94.30%
BRF (Higuchi et al., 2024) ✓ 69k 95.20%
ASRC-SNN (Xu et al.) ✓ ✓ ✓ 0.15M 95.77%*
**DelRec (only Rec.** **delays)** _**Ours**_ ✓ ✓ ✓ **0.16M** **96.21%**


a The parameter count is not clearly specified in the associated publication. Estimated from Figure 6.

 - Results reproduced with publicly available code, using dedicated validation and test sets.


Overall, DelRec models set new state-of-the-art accuracy scores on both SSC and PS-MNIST
datasets and using competitive numbers of parameters. Remarkably, this performance is achieved
using simple LIF neurons, in contrast to many competing approaches that rely on more complex


1Dampfhoffer et al. (2022) report 77 _±_ 0.4% on SSC using a spiking GRU, Zheng et al. (2024) report 82.46%
on SSC (on only one seed) using multi-compartment neurons, Wang et al. (2024) report 83.69% on SSC (only
one seed) using attention and distillation, Chen et al. (2024) report 97.78% on the PS-MNIST (one seed) with
a multi-compartment neuron and using the test set as the validation set.


6


neuron models incorporating adaptive mechanisms, resonant dynamics, or structured state-space
formulations. (Baronig et al., 2025; Fabre et al., 2025; Higuchi et al., 2024). This underscores
the importance of incorporating synaptic delays in RSNNs (see also Hammouamri et al. (2024);
M´esz´aros et al. (2025); Xu et al. for similar approaches) and suggests that even higher performance
could be achieved by combining delays with more sophisticated neuron models. Our findings also
demonstrate that optimizing synaptic delays can significantly enhance the performance of spiking
models, even in recurrent networks. Furthermore, they indicate that optimizing delays in recurrent
connections may yield greater benefits than optimizing feedforward delays, particularly for tasks
involving long-range temporal dependencies (see Section 3.2).


3.2 FUNCTIONAL STUDY OF DELAYS


To determine whether the high accuracies in the previous section stemmed from the use of learnable recurrent delays, we compared our method with the state-of-the-art feedforward delay learning
approach Hammouamri et al. (2024), and vanilla SNN architectures. It is worth noting that we are
comparing synaptic feedforward delays (one delay per synapse), with axonal recurrent delays (one
delay per neuron). For this study, we used much smaller models, and a much smaller dataset: SHD
(Spiking Heidelberg Digits), a widely used spiking dataset (Cramer et al., 2022) of 10k recordings
of spoken digits ranging from zero to nine, in English and German. This dataset demands leveraging
temporal patterns in spike trains to reach good classification accuracies.


We adopted the following procedure. _Validation phase_ : we first verified that our method performed
competitively with feedforward delays on the SHD dataset. _Simplification phase_ : we then reduced
the size of the layers, and simplified our networks to obtain smaller versions of state-of-the-art
models, with equivalent numbers of parameters. _Comparative_ _phase_ : we set the layer sizes such
that each model contains around 10k parameters, then gradually reduced the number of neurons
in layers, training a separate network for each model at each step. We compared the evolution
of accuracies on the SHD as a function of the parameter count. We also enforced sparsity in the
networks with spikes penalization, and study models’ performance as a function of mean firing rate.
Details of the architecture, parameters and training methodology can be found in A.2.3. In the next
sections, we detail the results obtained during each if the phases described above.


_**Validation phase**_ : To compete with state-of-the-art models on the SHD dataset, we designed a network inspired by Hammouamri et al. (2024), incorporating delays in recurrent connections. The
architecture also included regularizing modules as described in Table 3. We augmented the training
data following M´esz´aros et al. (2025) and Nowotny et al. (2022) (see A.2.3) to reduce overfitting.
In all experiments on this dataset, we used networks with 2 hidden layers of 256 neurons, but recurrent connections (and recurrent delays) only in the second layer to avoid overfitting (Fig. 3A). As
with the SSC and PS-MNIST datasets, we used only simple LIF neurons. However, we compared
our architectural innovation to the best-performing models, which often rely on more sophisticated
intrinsic neuronal dynamics.


SHD lacks a dedicated validation set, and historical evaluations of SNN performance on this dataset
have relied solely on the test set. This approach is methodologically flawed and leads to an overfitting of the test set. More recent works have set more rigorous standards by using a fraction of the
training set as a validation set, before reporting the best model’s accuracy on the test set (Baronig
et al., 2025; M´esz´aros et al., 2025). In line with this effort, we use 20% of the non-augmented
training set as a validation set, and we report the results we obtained on the test set in Table 2 (our
models are trained on 10 different seeds). Yet, while the best models report around 93% of accuracy
on the test set (using a clean split), M´esz´aros et al. (2025) explain that further improvements in
performance are likely not statistically significant given the small size of the test set (2264) : with
naive assumptions on error rates, the Bayesian confidence intervals of accuracies over 93% overlap.
For all these reasons, we decided not to include SHD in Table 1.


Whether using both feedforward and recurrent delays or only recurrent delays, our models achieve
state-of-the-art performance on SHD. Notably, the combination of recurrent and feedforward delays
yields the highest mean accuracy among our tested configurations, demonstrating the effectiveness
of integrating both delay types when overfitting is controlled (here with data augmentations). Given
that our method improves the state-of-the-art on larger and more challenging datasets, it suggests
that the SHD dataset has become overly saturated for benchmarking the processing capabilities of


7


Table 2: **Classification accuracies on SHD using a clean split**, ranked by accuracy.


Model Rec. Rec. Delays Ff. Delays LIF Param Test Acc. [%]


BRF (Higuchi et al., 2024) ✓ 0.1M 92.70 _±_ 0.70%
SE-adLIF (1L) (Baronig et al., 2025) ✓ 37.5k 93.18 _±_ 0.74%
EventProp [b] (M´esz´aros et al., 2025) ✓ ✓ _∼_ 1M [a] 93.24 _±_ 1.00%
**DelRec (Only Rec.** **delays)** _**Ours**_ **[b]** ✓ ✓ ✓ **0.17M** **93.39** _±_ **0.45%**
**DelRec (Rec.** **and Ff.** **delays)** _**Ours**_ **[b]** ✓ ✓ ✓ ✓ **0.24M** **93.73** _±_ **0.69%**
DCLS [b] (Hammouamri et al., 2024) ✓ ✓ 0.22M 93.77 _±_ 0.68%
SE-adLIF (2L) (Baronig et al., 2025) ✓ 0.45M 93.79 _±_ 0.76%


a The parameter count is not clearly specified in the associated publication. Estimated from Figure 5.
b Models trained using augmentations.


new spiking models. We therefore recommend its use only as an initial validation step for proof-ofconcept studies.


_**Simplification phase**_ : In Table 2, our configuration was exactly the same as the one in DCLS-Delays,
and described in A.2.3. At this stage, we focused on simplifying both models to compare them at
a lower yet equivalent number of parameters. The changes between the state-of-the-art models and
the small models are detailed in Table 3.


Table 3: Network parameters for models used on the SHD dataset, ranked by accuracy.


Model #Layers Hidden _τ_ (ms) BN Bias Epochs Augm. #Params


Large 2 256 10.05 Yes Yes 150 Yes 0.2M
Small 2 _≤_ 52 20 No No 30 No _≤_ 10k


Additionally, we increased the learning rate to 0 _._ 1 and we applied a one cycle scheduler to the delay
parameters. Feedforward delays were enabled only between the first and second hidden layers,
while recurrent delays were restricted to the second hidden layer, ensuring that both types of delays
operated on weighted spikes (see Fig. 3A).


_**Comparative phase**_ : Recent works suggested that delays in SNNs improved robustness under low
number of parameters and sparsity constraints (Hammouamri et al., 2024; M´esz´aros et al., 2025). To
explore this further, we performed an ablation study on the SHD dataset, evaluating model performance under varying delay settings and constraints. In total, we compared 6 different models, with
the architecture presented in Figure. 3A: a vanilla SNN, a vanilla RSNN with a uniform delay of 1
time-step in recurrent connections, a model with learned feedforward delays using DCLS-delays, a
RSNN with fixed random delays in recurrent connections, a model with learned delays in recurrent
connections and a model with learned feedforward and recurrent delays.


These studies highlight several observations regarding the role of delays. First, as theory suggests,
models with either type of delays strongly outperform the equivalent architecture without delays
(Fig. 3B), proving that delays, including those in recurrent connections, offer an invaluable tool for
temporal structure extraction. Moreover, the comparison between a vanilla RSNN and the same
network with random fixed recurrent delays illustrates how the simple introduction of delays in
recurrent connections mitigates the training difficulties of RSNNs due to gradient issues.


Second, our results in Fig. 3C indicate that under low parameters constraints, a model with recurrent
delays consistently outperforms all other models, with accuracy degrading less steeply as network
size decreases. It suggests that recurrent delays allow for more efficient use and reuse of temporal
information when representational capacity is limited. In contrast, we found no advantage in using
both types of delays in these small configurations, despite this combination achieving our highest
score on the SHD with larger models(Table. 2). However, Fig. 3 reveals a tradeoff between accuracy and energy consumption : while recurrent delays achieve better performance than feedforward
delays for an equivalent number of parameters and without firing rate constraints, we found that
feedforward delays reached their best accuracies with a lower mean firing rate than recurrent delays


8


in the second layer can have recurrent connections, possibly with delays. **B** : Histogram of model
accuracy on the SHD. The values on top of the bars are the number of parameters in the models.
**C** : Model accuracy on the SHD as a function of the number of parameters in the network ( _top_ ), and
as a function of the mean number of spikes per neuron per time-step ( _bottom_ ). The shaded areas
represent the standard error mean (sem). For more details see A.3.


required for the same performance. Though the model with recurrent delays achieves its best accuracy under an already low energy cost (0 _._ 08 spikes per neuron per time-step), our study suggest that
feedforward delays can provide a more energy-efficient alternative when computational efficiency
is preferred over performance.


Finally, in line with the findings of Hammouamri et al. (2024) with DCLS-Delays, we observe in
Fig. 3B that the benefit of learning delays in recurrent connections is relatively small, yet consistent
and significant, underlining the utility of optimizing recurrent dynamics for temporal processing.


4 CONCLUSION


This work introduces a new method (DelRec) to optimize delays in the recurrent connections of
spiking neural networks with surrogate gradient learning and backpropagation, leveraging differentiable interpolation and a progressive spike scheduling process. Using the simplest spiking neuron
model, i.e., a LIF with instantaneous synapses, DelRec outperforms the previous state-of-the-art
accuracy on both the PS-MNIST vision dataset and the SSC audio dataset, two widely recognized
benchmarks for evaluating temporal processing capabilities. Moreover, we present a study suggesting that recurrent delays can achieve better performance than feedforward delays. We believe that
further improvements could be obtained by using more complex neurons with DelRec, and by better
combining DelRec with feedforward delays. Finally, our method also offers new tools for modeling neural populations dynamics in the brain and could offer insights on how delays shape sensory
processing.


9


Vanilla SNN Feedforward delays
Vanilla RSNN Recurrent delays


_2k_ _4k_ _6k_ _8k_ _10k_

Number of parameters


80


80


70


60


50


40


30


20


80


75


70


65


60


70


60


50


40


55


Mean spiking rate


5 REPRODUCIBILITY STATEMENT


All the results presented in this work can be reproduced using the anonymous repository: [https:](https://anonymous.4open.science/r/Recdel-4175)
[//anonymous.4open.science/r/Recdel-4175.](https://anonymous.4open.science/r/Recdel-4175) We used publicly available datasets,
[downloadable at the following address : https://zenkelab.org/datasets/. We also used](https://zenkelab.org/datasets/)
the PS-MNIST dataset, which directly derived from Pytorch’s MNIST. Our implementation builds
upon the Spiking Jelly framework (Fang et al., 2023), an open-source library providing optimized
tools for developing spiking neural networks. The hyperparameters we used are provided in the
Appendix (see A.2.5), and can also be found in the configuration files of our repository. Finally,
our results were produced using NVIDIA A100 GPUs for the SSC and PS-MNIST datasets, and
NVIDIA A40 GPUs for the SHD dataset.


REFERENCES


Markus Baronig, Romain Ferrand, Stefan Sabathiel, et al. Advancing spatio-temporal processing through adaptation in spiking neural networks. _Nature_ _Communications_, 16:
5776, 2025. doi: 10.1038/s41467-025-60878-z. URL [https://doi.org/10.1038/](https://doi.org/10.1038/s41467-025-60878-z)
[s41467-025-60878-z.](https://doi.org/10.1038/s41467-025-60878-z)


Guillaume Bellec, Darjan Salaj, Anand Subramoney, Robert Legenstein, and Wolfgang Maass. Long
short-term memory and learning-to-learn in networks of spiking neurons. _CoRR_, abs/1803.09574,
2018. [URL http://arxiv.org/abs/1803.09574.](http://arxiv.org/abs/1803.09574)


Alexandre Bittar and Philip N. Garner. A surrogate gradient spiking baseline for speech command recognition. _Frontiers_ _in_ _Neuroscience_, 16, 2022. ISSN 1662-453X. doi: 10.3389/fnins.
2022.865897. [URL https://www.frontiersin.org/articles/10.3389/fnins.](https://www.frontiersin.org/articles/10.3389/fnins.2022.865897/full)
[2022.865897/full.](https://www.frontiersin.org/articles/10.3389/fnins.2022.865897/full)


Xinyi Chen, Jibin Wu, Chenxiang Ma, Yinsong Yan, Yujie Wu, and Kay Chen Tan. Pmsn: A
parallel multi-compartment spiking neuron for multi-scale temporal processing. _arXiv_ _preprint_
_arXiv:2408.14917_, 2024.


Benjamin Cramer, Yannik Stradmann, Johannes Schemmel, and Friedemann Zenke. The heidelberg
spiking data sets for the systematic evaluation of spiking neural networks. _IEEE_ _Transactions_
_on Neural Networks and Learning Systems_, 33(7):2744–2757, 2022. doi: 10.1109/TNNLS.2020.
3044364.


Manon Dampfhoffer, Thomas Mesquida, Alexandre Valentian, and Lorena Anghel. Investigating
current-based and gating approaches for accurate and energy-efficient spiking recurrent neural networks. In Elias Pimenidis, Plamen Angelov, Chrisina Jayne, Antonios Papaleonidas,
and Mehmet Aydin (eds.), _Artificial_ _Neural_ _Networks_ _and_ _Machine_ _Learning_ _–_ _ICANN_ _2022_,
pp. 359–370, Cham, 2022. Springer Nature Switzerland. ISBN 978-3-031-15934-3. doi:
10.1007/978-3-031-15934-3 ~~3~~ 0.


Lucas Deckers et al. Co-learning synaptic delays, weights and adaptation in spiking neural networks.
_Frontiers in Neuroscience_, 18:1360300, 2024. doi: 10.3389/fnins.2024.1360300.


Maxime Fabre, Lyubov Dudchenko, and Emre Neftci. Structured State Space Model Dynamics and
Parametrization for Spiking Neural Networks, June 2025.


Wei Fang, Zhaofei Yu, Yanqi Chen, Timoth´ee Masquelier, Tiejun Huang, and Yonghong Tian. Incorporating learnable membrane time constant to enhance learning of spiking neural networks. In
_Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)_, pp. 2661–
2671, 2021.


Wei Fang, Yanqi Chen, Jianhao Ding, Zhaofei Yu, Timoth´ee Masquelier, Ding Chen, Liwei Huang,
Huihui Zhou, Guoqi Li, and Yonghong Tian. Spikingjelly: An open-source machine learning
infrastructure platform for spike-based intelligence. _Science_ _Advances_, 9(40):eadi1480, 2023.
doi: 10.1126/sciadv.adi1480. URL [https://www.science.org/doi/abs/10.1126/](https://www.science.org/doi/abs/10.1126/sciadv.adi1480)
[sciadv.adi1480.](https://www.science.org/doi/abs/10.1126/sciadv.adi1480)


10


Marcus Ghosh, Karim G. Habashy, Francesco De Santis, Tomas Fiers, Dilay Fidan Erc¸elik, Bal´azs
M´esz´aros, Zachary Friedenberger, Gabriel B´ena, Mingxuan Hong, Umar Abubacar, Rory T.
Byrne, Juan Luis Riquelme, Yuhan Helena Liu, Ido Aizenbud, Brendan A. Bicknell, Volker
Bormuth, Alberto Antonietti, and Dan F. M. Goodman. Spiking Neural Network Models of
Interaural Time Difference Extraction via a Massively Collaborative Process. _eneuro_, 12(7):
ENEURO.0383–24.2025, jul 2025. ISSN 2373-2822. doi: 10.1523/ENEURO.0383-24.2025.
[URL https://www.eneuro.org/content/12/7/ENEURO.0383-24.2025.](https://www.eneuro.org/content/12/7/ENEURO.0383-24.2025)


J. G¨oltz, J. Weber, L. Kriener, et al. Delgrad: exact event-based gradients for training delays and
weights on spiking neuromorphic hardware. _Nature_ _Communications_, 16(1):8245, 2025. doi:
10.1038/s41467-025-63120-y.


Ilyass Hammouamri, Ismail Khalfaoui-Hassani, and Timoth´ee Masquelier. Learning Delays in Spiking Neural Networks using Dilated Convolutions with Learnable Spacings. In _ICLR_, pp. 1–12,
2024. [URL http://arxiv.org/abs/2306.17670.](http://arxiv.org/abs/2306.17670)


Saya Higuchi, Sebastian Kairat, Sander M. Bohte, and Sebastian Otte. Balanced resonate-and-fire
neurons, 2024. [URL https://arxiv.org/abs/2402.14603.](https://arxiv.org/abs/2402.14603)


Eugene M. Izhikevich. Polychronization: Computation with Spikes. 18(2):245–282. ISSN 08997667, 1530-888X. doi: 10.1162/089976606775093882. [URL https://direct.mit.edu/](https://direct.mit.edu/neco/article/18/2/245-282/7033)
[neco/article/18/2/245-282/7033.](https://direct.mit.edu/neco/article/18/2/245-282/7033)


Ismail Khalfaoui-Hassani, Thomas Pellegrini, and Timoth´ee Masquelier. Dilated Convolution with
Learnable Spacings: beyond bilinear interpolation. In _ICML_ _Workshop:_ _Differentiable_ _Al-_
_most_ _Everything_, pp. 1–7, jun 2023. URL [http://arxiv.org/abs/2306.00817id=](http://arxiv.org/abs/2306.00817 id=j8FPBCltB9)
[j8FPBCltB9.](http://arxiv.org/abs/2306.00817 id=j8FPBCltB9)


Michelle Monje. Myelin Plasticity and Nervous System Function. _Annual_ _Re-_
_view_ _of_ _Neuroscience_, 41(1):61–76, jul 2018. ISSN 0147-006X. doi: 10.1146/
annurev-neuro-080317-061853. URL [https://www.annualreviews.org/doi/10.](https://www.annualreviews.org/doi/10.1146/annurev-neuro-080317-061853)
[1146/annurev-neuro-080317-061853.](https://www.annualreviews.org/doi/10.1146/annurev-neuro-080317-061853)


Bal´azs M´esz´aros, James C. Knight, and Thomas Nowotny. Efficient event-based delay learning in
spiking neural networks, 2025. [URL https://arxiv.org/abs/2501.07331.](https://arxiv.org/abs/2501.07331)


Thomas Nowotny, James P. Turner, and James C. Knight. Loss shaping enhances exact gradient
learning with Eventprop in spiking neural networks. 2022. doi: 10.48550/ARXIV.2212.01232.


Thomas Pellegrini, Romain Zimmer, and Timothee Masquelier. Low-Activity Supervised Convolutional Spiking Neural Networks Applied to Speech Commands Recognition. In _2021_ _IEEE_
_Spoken Language Technology Workshop (SLT)_, pp. 97–103. IEEE, jan 2021. ISBN 978-1-72817066-4. doi: 10.1109/SLT48900.2021.9383587. [URL https://ieeexplore.ieee.org/](https://ieeexplore.ieee.org/document/9383587/)
[document/9383587/.](https://ieeexplore.ieee.org/document/9383587/)


Sumit Bam Shrestha and Garrick Orchard. Slayer: Spike layer error reassignment in time, 2018.
[URL https://arxiv.org/abs/1810.08646.](https://arxiv.org/abs/1810.08646)


Pengfei Sun, Zhu Longwei, and Dick Botteldooren. Axonal Delay as a Short-term Memory for Feed
Forward Deep Spiking Neural Networks. _Proc. of ICASSP 2022_, 2022.


Pengfei Sun, Yansong Chua, Paul Devos, and Dick Botteldooren. Learnable axonal delay in spiking
neural networks improves spoken word recognition. _Frontiers_ _in_ _Neuroscience_, 2023. doi: 10.
3389/fnins.2023.1275944.


Jonathan Timcheck, Sumit Bam Shrestha, Daniel Ben Dayan Rubin, Adam Kupryjanow, Garrick Orchard, Lukasz Pindor, Timothy Shea, and Mike Davies. The Intel neuromorphic DNS
challenge. _Neuromorphic_ _Computing_ _and_ _Engineering_, 3(3), 2023. ISSN 26344386. doi:
10.1088/2634-4386/ace737.


Jiaqi Wang, Liutao Yu, Liwei Huang, Chenlin Zhou, Han Zhang, Zhenxi Song, Min Zhang, Zhengyu
Ma, and Zhiguo Zhang. Efficient speech command recognition leveraging spiking neural network
and curriculum learning-based knowledge distillation, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2412.12858)
[abs/2412.12858.](https://arxiv.org/abs/2412.12858)


11


Timo C. Wunderlich and Christian Pehle. Event-based backpropagation can compute exact gradients
for spiking neural networks. _Sci._ _Rep._, 11(12829):1–17, June 2021. ISSN 2045-2322. doi:
10.1038/s41598-021-91786-z.


Shang Xu, Jiayu Zhang, Ziming Wang, Runhao Jiang, Rui Yan, and Huajin Tang. ASRC-SNN:
Adaptive Skip Recurrent Connection Spiking Neural Network. URL [http://arxiv.org/](http://arxiv.org/abs/2505.11455)
[abs/2505.11455.](http://arxiv.org/abs/2505.11455)


Xingting Yao, Fanrong Li, Zitao Mo, and Jian Cheng. Glif: A unified gated leaky integrate-andfire neuron for spiking neural networks. In _Advances in Neural Information Processing Systems_,
volume 35, pp. 32160–32171, 2022.


Bojian Yin, Federico Corradi, and Sander M. Bohte. Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. _Nature_ _Machine_ _Intelligence_, 3(10):
905–913, oct 2021. doi: 10.1038/s42256-021-00383-x.


H. Zheng, Z. Zheng, R. Hu, et al. Temporal dendritic heterogeneity incorporated with spiking neural
networks for learning multi-timescale dynamics. _Nature Communications_, 15(1):277, 2024. doi:
10.1038/s41467-023-44614-z.


A APPENDIX


A.1 LEARNING ALGORITHM


We detail here the algorithm we use to learn delays in recurrent connections. Algorithm 1 describes a forward pass for _N_ neurons. Our spike scheduling employs a circular buffer of size
_N_ _×_ dim( **E** [˜] ( _σ, D_ )) with **E** [˜] ( _σ, D_ ) defined in Eq. 13.


A.2 LEARNING STRATEGY, ARCHITECTURES AND HYPERPARAMETERS


A.2.1 _hσ,d_


Our algorithm uses a scheduling of spikes for future time steps, which is, for one recurrent connection, spread around the delay value _d_ . This spread has a triangular shape determined by _hσ,d_, with
the sigma decreasing from its initial value to zero throughout the training. We use an exponential
decay value _decay_ = 0 _._ 95, and we update _σepoch_ at each epoch such that :


_σepoch_ _←_ _σinit × decay_ 100 _×_ _N_ _[epoch]_ epochs (14)


with _N_ epochs the total number of epochs. We always take _σinit_ = 10.


However, we use an additional strategy on the SSC dataset in order to allow for a quicker decay of
_σepoch_ for specific neurons. Specifically, we introduce a parameter _p_ per neuron, and consider for a
neuron _i_ the modified spread function:


_[−]_ [(1 +] _[ d][i]_ [)] _[|]_
_hσ,di,pi_ ( _τ_ ) = max(0 _,_ [1 + 2] _[ ·]_ [ sig] (1 + 2 [(] _[p][i]_ [)] _·_ _[ ·]_ sig _[ σ][ −|]_ ( _pi_ _[τ]_ ) _· σ_ ) [2] ) (15)


with sig( _·_ ) the sigmoid function, and parameters _{pi}_ all initialized at 0, and added to the learnable
parameters. To remain neuromorphic hardware friendly and have a single integer delay in each
recurrent connection at the end of training, the _σ_ parameter is still updated with Eq. 14.


A.2.2 SSC AND PS-MNIST DATASETS


On the SSC dataset, we reduced the input size by binning every 5 neurons of the original 700,
resulting in a spatial input dimension of 140. We also temporally binned the inputs using a discrete
time-step of ∆= 5 _._ 6ms.


12


**Algorithm 1:** Learning Delays in Recurrent Connections


**Inputs:** Sequence time point at date _t_ : _X_ [ _t_ ], for _N_ neurons.
**Parameters:** Number of neurons _N_, recurrent weights _W_ [rec] _∈_ R _[N]_ _[×][N]_, recurrent delays _d_ _∈_ R _[N]_,
scheduled _σepoch_, firing threshold _V_ th, neuron reset type, reset potential _Vreset_, neuronal
charge function _f_ .
**Outputs:** Spikes at time _t_ : _S_ [ _t_ ].
Compute **E** [˜] ( _σepoch, D_ )
_L ←_ dim� **E** ˜ ( _σepoch, D_ )�

`spread` _←_ - _hσepoch,d_ ( _x_ ) for _x ∈_ **E** [˜] ( _σepoch, D_ )�

_B_ _←_ **0** _[N]_ _[×][L]_ // zero-out buffer matrix
`pointer` _←_ 0
**for** _t ←_ 0 **to** _T_ **do**

**for** _i ←_ 1 **to** _N_ **do**

_Xi_ [rec] [ _t_ ] _←_ _Bi_ [ `pointer` ] // take current time step in the buffer
_Ii_ [ _t_ ] _←_ _Xi_ [ _t_ ] + _Xi_ [rec] [ _t_ ] // get current spikes with neuron equation
_Hi_ [ _t_ ] _←_ _f_ ( _Vi_ [ _t −_ 1] _, Ii_ [ _t_ ])
_S_ [ _t_ ] _←_ Θ� _Hi_ [ _t_ ] _−_ _V_ th�

**if** _Hard reset_ **then**

_Vi_ [ _t_ ] _←_ _Hi_ [ _t_ ] _·_ (1 _−_ _Si_ [ _t_ ]) + _V_ reset _· Si_ [ _t_ ]
**else**

_Vi_ [ _t_ ] _←_ _Hi_ [ _t_ ] _−_ _V_ th _· Si_ [ _t_ ]
**end**
_Bi_ [ `pointer` ] _←_ 0 // update the buffer
`pointer` _←_ ( `pointer` + 1) mod _L_
**for** _j_ _←_ 1 **to** _N_ **do**

**for** _τ_ _∈_ **E** [˜] ( _σepoch, D_ ) **do**

**if** `pointer` + _τ_ _≤_ _L −_ 1 **then**

_Bj_ [ `pointer` + _τ_ ] _←_ _Bj_ [ `pointer` + _τ_ ] + _wji_ _[rec]_ _·_ `spread` [ _τ_ ] _· Si_ [ _t_ ]
**else**

`p` _[′]_ _←_ ( `pointer` + _τ_ ) mod _L_
_Bj_ [ `p` _[′]_ ] _←_ _Bj_ [ `p` _[′]_ ] + _wji_ _[rec]_ _·_ `spread` [ _τ_ ] _· Si_ [ _t_ ]
**end**
**end**
**end**
**end**
**end**


For both the SSC and PS-MNIST, the readout is a simple linear layer mapping the last hidden layer
to the _n_ classes = 35 output neurons. Then, denoting _N_ the batch size, _y_ ˆ _i_ the output of neuron _i_ and _y_
the ground truth, we compute the cross-entropy loss for one batch as:


A.2.3 SHD DATASET


For this dataset, we also reduced the input size by binning every 5 neurons of the original 700,
resulting in a spatial input dimension of 140. As well, we binned the inputs temporally, using a
discrete time-step of ∆= 10ms. The architecture used is composed of two fully connected hidden
layers (see Fig. 3.A), and each of these layers can be followed by batch normalization, a neuron
module and dropout. The readout consists of _n_ classes = 20 LIF neurons with an infinite threshold.
Following Bittar & Garner (2022), we consider _Vi_ _[r]_ [the] [membrane] [potential] [of] [the] [readout] [neuron]
_i_, then we compute the softmax over output neurons at each time-step before summing across the
temporal dimension. Thus, the model output for this neuron:


13


_L_ = [1]

_N_


_N_

- _−_ log(ˆ _yn,_ _yn_ [ _n_ ]) _._ (16)


_n_ =1


where fan ~~i~~ n denotes the number of input units to the layer. The recurrent delays were initialized
with a random draw from a uniform distribution:


_d_ [rec] _i_ _∼U_ (10 _,_ 30) _._


For all tasks, in the models with feedforward delays, we initialized them uniformly with :


_d_ [ff] _i_ _[∼U]_ [(0] _[,]_ [50)] _[.]_


A.2.5 HYPERPARAMETERS


We detail here all the hyperparameters that we used to perform the experiments presented in this
work.


In our models, the feedforward delays were optimized with the DCLS method. However, while
Hammouamri et al. (2024) use gaussians centered on the delays positions in their temporal kernels
and reduce the variance during training, we only spread the weights using a simple linear interpolation between the two closest integer positions to the floating delay parameter.


On the SSC dataset, we changed the feedforward dropout to 0.2 when feedforward delays were used,
in order to avoid overfitting.


14


- softmax( _Vi_ _[r]_ [[] _[t]_ []) =] 
_t∈T_ _t∈T_


_y_ ˆ _i_ = 


_t∈T_


_e_ _[V]_ _i_ _[r]_ [[] _[t]_ []]

- _n_ classes _j_ _[r]_ [[] _[t]_ []] (17)
_j_ =1 _[e][V]_


with _T_ the temporal dimension. Then, denoting _N_ the batch size and _y_ the ground truth, we compute
the cross-entropy loss for one batch as:


_L_ = [1]

_N_


_N_

- _−_ log(ˆ _yn,_ _yn_ [ _n_ ]) _._ (18)


_n_ =1


We used augmentations on the training set of the SHD in order to limit overfitting. We use this
method only on our large models when trying to reach high accuracies, not in our ablation study.
We followed the approach of Nowotny et al. (2022) and M´esz´aros et al. (2025):


    - We applied a random temporal shifting to inputs, with a shift uniformely drawn in the
interval      - _−_ 100 time steps _,_ 100 time steps�.


    - Each input was blended with an input from the same class by aligning their center of mass
and, for each time step, picking a spike from either sample with a probability of 0 _._ 5.


A.2.4 WEIGHTS AND DELAYS INITIALIZATION


**SSC** **and** **PS-MNIST** : We initialized the feedforward weights with the default linear weights and
biases method in Pytorch:


   -    -    _wij, bi_ _∼U_ _−_ in features1 _[,]_ [+] in ~~f~~ eatures1


_._


The recurrent delays were initialized with a random draw from a half gaussian centered on 0:


_d_ [rec] _i_ _∼|N_ (0 _, σ_ [2] ) _|,_ with _σ_ = 12 _._

_√_
**SHD** : We initialized the feedforward weights using Kaiming uniform initialization, with _a_ =


5:


   -   - ~~�~~
_wij_ _∼U_ _−_ in ~~f~~ eatures6 _[,]_ [+] in ~~f~~ eatures6


_,_


_bi_ _∼U_ - _−_ ~~_√_~~ 1


in ~~f~~ eatures


1 [+] ~~_√_~~ 1

in ~~f~~ eatures _[,]_ in ~~f~~


_,_


Table 4: General hyperparameters used for the SSC, PS-MNIST and SHD datasets.


**Hyperparameter** **SSC** **PS-MNIST** **SHD**


`epochs` 100 200 150
`batch size` 128 256 256
`learning rate max weights` 1e-3 1e-3 5e-3
`learning rate max positions` 5e-2 5e-2 5e-2
`optimizer` Adam AdamW AdamW
`weight decay` 1e-5 1e-2 1e-5

```
batch norm False False True

bias False False True
```

`feedforward dropout` 0.1 0.1 0.4
`recurrent dropout` 0.3 0.2 0.2
`scheduler weights` One cycle One cycle One cycle
`scheduler positions` One cycle One cycle Cosine annealing


Table 5: Neuron-related hyperparameters used for the SSC, PS-MNIST and SHD datasets.


**Hyperparameter** **SSC** **PS-MNIST** **SHD**


_τ_ (time-steps) 2 2 1.005
_Vthreshold_ 1. 1. 1.
`reset type` Soft Soft Hard
_Vreset_ 0

```
detach reset False False True
```

surrogate function Triangle Triangle Arctan


A.3 ABLATION STUDY METHODOLOGY


In this section, we aim to explain in more details our methodology to produce Fig. 3.


Fig 3B : The goal was to obtain acceptable accuracies with a minimal number of parameters. We
found that around 10k parameters was an overall effective choice. To this end, we picked the number
of neurons reported in Table 6 for our different models, and we trained them on 3 seeds. The standard
error mean (sem) is also reported on the figure.


Table 6: Neurons per layer in the models of the ablation study.


**Model** Hidden layers sizes Approx. number of parameters


Vanilla RSNN [42, 42] _∼_ 10k
Vanilla SNN [52, 52] _∼_ 11k
Learned Ff. and Rec. delays [38, 38] _∼_ 10k
Learned Ff. delays [42, 42] _∼_ 10k
Fixed Rec. delays [42, 42] _∼_ 10k
Learned Rec. delays [42, 42] _∼_ 10k


Fig 3C, _Top_ : For each model, we started with the same layer size reported in Table. 6, and we
reduced gradually the layer sizes with steps of 6 neurons, training on 3 seeds each time, to produce
the points in the figure.


Fig 3C, _Bottom_ : We started with the same models as in Table. 6, but we gradually penalized the
mean firing rate of the model in order to introduce more sparsity. To do so, we added the following
term to the loss, as in Pellegrini et al. (2021) :


1
_L_ spike =
2 _TBN_


_T_


_t_ =1


_N_

- _Sb,n_ [ _t_ ] [2] _,_


_n_ =1


_B_


_b_ =1


15


where _T_ is the length of the temporal dimension, _N_ is the number of neurons in the model and _B_ is
the batch size. Thus, we considered a total loss:


_L_ = _L_ cross entropy + _λL_ spike _,_


and we trained our models (3 seeds again) for _λ_ values ranging from 1e-4 to 10. We recorded the
number of spikes per neuron, and we plotted the model accuracy as a function of the number of
spike per neuron and per time-step on the test set.


A.4 USE OF LARGE LANGUAGE MODELS


We declare the following use of LLMs in the preparation of this work:


    - For writing: OpenAI’s ChatGPT was used in the writing of this article to improve the
phrasing and the clarity of some sentences.


    - For coding: OpenAI’s ChatGPT and GitHub copilot were used to enhance coding efficiency, more particularly for the factorization of existing scripts and the generation of plots.


16
# On Biologically Plausible Learning in Continuous Time

- Decision: Reject
- Scores: 2, 2, 6, 2

## Abstract
Biological learning unfolds continuously in time, yet most algorithmic models rely on discrete updates and separate inference and learning phases. We study a continuous-time neural model that unifies several biologically plausible learning algorithms and removes the need for phase separation. Rules including stochastic gradient descent (SGD), feedback alignment (FA), direct feedback alignment (DFA), and Kolen–Pollack (KP) emerge naturally as limiting cases of the dynamics. Simulations show that these continuous-time networks stably learn at biological timescales, even under temporal mismatches and integration noise. Our results reveal that, in the absence of longer-range memory mechanisms, learning is constrained by the temporal overlap of inputs and errors. Robust learning requires potentiation timescales that outlast the stimulus window by at least an order of magnitude, placing the effective eligibility regime in the few-second range. More broadly, this identifies a unifying principle: learning succeeds when input and error are temporally correlated at each synapse, a rule that yields testable predictions for neuroscience and practical design guidance for analog hardware.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes coupled differential equations to study the activity and plasticity of neural networks. At the digital implementation level, this removes the separation between activity generation and weight update phases as all variables will evolve jointly according to the differential equations.

Using feedforward neural networks and rate-based neurons, the paper studies the impact of different schemes of delivering task error feedback to individual units, both theoretically for each synapse and numerically at the level of task performance. By analyzing the results of the simulations, the paper makes multiple observations on the neurobiology of synaptic learning, such as the delay between neuronal input and error feedback signals.

### Strengths
- An ODE based formulation of neuronal activity and plasticity dynamics

- The paper consider three relevant timescales: (i) synaptic transmission, (ii) plasticity via intra-neuronal signaling cascades, (iii) synaptic weakening via protein turnover, etc

### Weaknesses
- For a manuscript on bio-plausible synaptic learning, the experimental setup is not appropriate:
    - Only feedforward networks are studied and the input, output, and the task does not involve dynamics.
    - Rate-based neuronal units are important tools in computational neuroscience. However, their use here is not justified because millisecond-scale timing considerations are studied.

- If timescales are as given in Table 1, then separate activity and update phases appear justified. I believe essentially the same insights would be gleaned if separate phases were used.

- All the findings of this manuscript on feedback delays are already known and formally studied as eligibility traces in the literature (e.g., Gerstner).

- (line 146) A key aspect of backpropagation is the back propogation of the error signal. Without showing the dynamics of the error signal, it would be wrong to say that the backpropagation rule is recovered.

- In Eq. 2 and below, the role of the post-synaptic neuron is ignored. The concept of the eligibility trace formalizes this, which can be considered to either change the integration limits or shift the kernel function. In a 'continuous time' learning framework, what the integration limits correspond to in Eq. 2 is not clear. This shows up in a few places in the manuscript. For instance, the asymmetry considered around line 267 emerges from the "validation" of the pre-synaptic activity by the post-synaptic activity.

- For rate-based neurons and in a continuous time learning framework, it is not clear what it means for the neuron to be active in $[0,T]$ with $T \ll \tau_\text{pot}$.

- The paper studies two datasets: downsampled MNIST digits and a 'synthetic two-dimensional circles dataset'. The second dataset is not clear to me. More importantly, the tasks are not defined for either of the two datasets!

### Questions
- What are the tasks studied in this paper?

- Would the results hold for a recurrent network model?

- Would the results hold for a spiking neuron model?

- Would the results change significantly if forward-activity and update phases were separated, as in common machine learning?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Summary:

This paper presents a time-continuous, phase-less and biologically-plausible approximation of error backpropagation. It analyzes the impact of time lag in the forward- and error propagation and compares the timescales beneficial to model function to biological timescales.
While the topic is interesting and worth studying, this paper lacks novelty and does not acknowledge a whole body of work that tackles the same problems (see below). On top of that it lacks clarity in methods and result presentation.

### Strengths
Strengths: 

The topic is interesting and I appreciate the goal of modeling a time-continuous system without the approximating step of discretizations. Additionally I like the finding that the time scales that prove favorable for model functionality seem to correspond to biological time scales.

### Weaknesses
Lack of novelty and lacking acknowledgement of prior works:
- l. 42: "By contrast, existing algorithmic models are cast in discrete steps that alternate between inference and learning phases, effectively assuming that forward and backward signals are globally and instantaneously synchronized [1]". This is a strong misrepresentation of the content of the cited paper. This review paper compares 4 different biologically plausible algorithms for error backpropagation. Of which at least 2 (predictive coding as well as the dendritic microcircuit [2, 3]) do not rely on a split between forward propagation and error phase during learning. Also, while the review paper formulates these methods in time-stepped variant for ease of simulation, the original publication of [3] and some of the many publications on predictive coding models use time-continuous formulations of the models.
- Contribution 1: this is not novel as illustrated by the previous point. Other examples of pre-existing continuous time models are [4, 5, 6, 7, 8]
- Contribution 2: the fact that learning in these types of models, which include either transmission delays or slowly evolving dynamics, is only possible if there is a synchronization and co-temporal arrival of error and forward signal, is already known. In fact, there exist already solutions to mitigate this limitation [4, 9].
- l. 135 "Unlike existing algorithms that explicitly separate inference and learning phases, these networks evolve in continuous time: outputs and weights change together in response to input and error signals". This is incorrect. There exist multiple prior works that emphasize the removal of a split between inference and learning phases (see above). 
- The related works section misses a whole body of work  [1-9]. These works attack similar problems as this paper and need to be discussed.

Lack of clarity:
- Eq 1: in the text the error is defined as "e" but there is no "e" in the equation. Presumably the notation there has switched to the "epsilon". This is unnecessarily confusing.
- Eq1: In the surrounding text the weights are parametrized as bold(v) and bold(w) (not capitalized) while in Eq 1: Capital (and not bold) W and V are used. This is unnecessarily confusing.
- l. 186: More details on the type of ODE solvers and simulation techniques used are required (e.g. in the appendix). E.g. pseudo-code would be good.
- The results are presented as color maps of accuracies, which is hard to read because of the big range of accuracy on the color-bar. Tables of top-accuracies would have been very helpful.
- Additionally, the color map plots are unable to convey a notion of variance between multiple runs, therefore neglecting the stability of performance over different initialization which is also an important information about a model.

Claims not backed by enough evidence:
- l. 137 - 161: The derivations/proofs of these statements should be detailed in the appendix. Without that these are completely unproven statement. Also, the paper assumes "Given the error eL as the gradient of the loss function with respect to the output layer". A huge part of the literature on biologically plausible error backpropagation is to investigate how this error signal is transported and shaped in a bio-plausible manner. Just assuming that the error automatically gets transported in the correct fashion leaves out one of the difficult open questions.
- The fact that the best achieved accuracies of the networks with multiple hidden layers in Fig 5 and 7 are inferior to the ones achieved by the single hidden layer versions, shows that this models lacks the capabilities to properly transport errors into the lower layers and make use of neurons in those layers. As a model/approximation of backpropagation this is however its main job.
- Comparisons to baseline models (i.e. MLPs trained with backprop) are hidden in the appendix even though, in my opinion this comparison is crucial to evaluate the quality of the model. Additionally, the results presented in these comparisons are very surprising. The approximation of error backpropagation, which has to deal with time-lag between forward and error signals and other non-idealities seems to consistently outperform the vanilla MLPs that are trained with full error backpropagation (even Adam instead of SGD). I find this extremely surprising and have the suspicion that a good hyperparameter optimization would change those results (in particular, just because Adam is used, that does not mean that the learning rate does not need to be adjusted).

References:

[1] Whittington, James CR, and Rafal Bogacz. "Theories of error back-propagation in the brain." _Trends in cognitive sciences_ 23.3 (2019): 235-250.

[2] Whittington, James CR, and Rafal Bogacz. "An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity." _Neural computation_ 29.5 (2017): 1229-1262.

[3] Sacramento, João, et al. "Dendritic cortical microcircuits approximate the backpropagation algorithm." _Advances in neural information processing systems_ 31 (2018).

[4] Haider, Paul, et al. "Latent equilibrium: A unified learning theory for arbitrarily fast computation with arbitrarily slow neurons." _Advances in neural information processing systems_ 34 (2021): 17839-17851.

[5] Aceituno, Pau Vilimelis, et al. "Target learning rather than backpropagation explains learning in the mammalian neocortex." _bioRxiv_ (2024).

[6] Meulemans, Alexander, et al. "Credit assignment in neural networks through deep feedback control." _Advances in Neural Information Processing Systems_ 34 (2021): 4674-4687.

[7] Payeur, Alexandre, et al. "Burst-dependent synaptic plasticity can coordinate learning in hierarchical circuits." _Nature neuroscience_ 24.7 (2021): 1010-1019.

[8] Max, Kevin, et al. "Learning efficient backprojections across cortical hierarchies in real time." _Nature Machine Intelligence_ 6.6 (2024): 619-630.

[9] Ellenberger, Benjamin, et al. "Backpropagation through space, time, and the brain." _arXiv preprint arXiv:2403.16933_ (2024).

### Questions
- Can you provide more details on the simulation techniques that were applied? The text seems to imply that the step to a time-stepped system is never made and everything is simulated fully in continuous time. How is this achieved?
- Can you comment on the variability between runs with different seeds and except for the weight initialization identical parameters? How stable is the system under hyperparameter changes?
- Can you explain why the approximation of the backpropagation algorithm in an unideal system (temporal mismatch between forward signals and errors) seems to outperform the vanilla error backpropagation algorithm in an ideal (instantaneous) MLP?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a continuous-time dynamical systems formulation of learning in feedforward rate-based neural networks in which inference and plasticity co-evolve without explicit phase separation. Neuronal activities $z_l(t)$ and synaptic parameters $(W_l,V_l)$ obey coupled first-order ODEs with three timescales: fast propagation $\tau_{\mathrm{prop}}$, intermediate potentiation $\tau_{\mathrm{pot}}$, and slow decay $\tau_{\mathrm{dec}}$. Under timescale separation $(\tau_{\mathrm{prop}}\ll \tau_{\mathrm{pot}}\ll \tau_{\mathrm{dec}})$, several learning rules are recovered as limiting cases by suitable choices or dynamics of $V_l$ and local error definitions. The authors analyze how temporal mismatch between input and error modulates learning and derive an “overlap law,” showing that the essential feature for learning is that "input and error must be temporally correlated at each synapse". Simulations on small synthetic dataset explore delay tolerance, depth effects (propagation lags), and biological plausibility via chosen $\tau$-ranges.

### Strengths
This work introduces a clean theoretical framework for continuous-time learning in biological rate-based network which recovers well-known learning strategies (e.g. Feedback Alignment) in appropriate limiting cases of the time constants and shapes of the error signals. On top of the welcomed unification of several previous learning proposals, the new framework offers a new intuitive insight for learning efficiency in terms of input/error time overlap. This represents a valuable/testable contribution.

The author also ground their modelling attempts on known biological constraints and - in light of their gained insight - articulate a new hypothesis for the role skip connections in deeper biological architectures: they are needed to preserve the overlap.

The work is in general well written with clean mathematical expositions and figure designs, with an easy-to-follow and compelling narrative.

### Weaknesses
While we believe this paper represents a good contribution, we want to also highlight the following shortcomings.

1. The proposed model suffers from high computational complexity, which inevitably limits the scope of the experiments. While this fact is already acknowledged by the authors in their Limitation section, the empirical results can only offer partial support to the core theoretical claims. In particular one is left to wonder whether such technique could handle more challenging problems such as CIFAR-10/100 (or even TinyImageNet) where depth is necessary for accurate performance, which however in turn worsen then overlap condition, making the end result of the depth scaling uncertain.
2. The "no phase separation" claim seems somewhat at odds with the fact that evaluation requires freezing $W,V$ and clamping $e=0$, which indeed operationally sounds like a separate phase. This fact becomes more relevant if one wants to employ such mechanism for a continuously learning agent with undefined training period.
3. No system noise stability is performed (e.g. small additive Gaussian noise to neural output). Given the complexity of coupled ODEs, resulting system robustness is hard to assess without proper experiments.

### Questions
1. In general, how would an "agent" address the problem of knowing when to switch on the plasticity? It seems that if one doesn't "manually turn off" the error signal in the model (i.e. effectively inserting a "no-train-phase"), then the synapses would drift based on whatever (random) error signals happens to impinge on the neuron, causing disruption to the network.
2. How would the overlap rule change if one starts to introduce "elegibility traces" that pick after the error onset or some other memory-like mechanism? Could such a principle be used to mitigate the problem with depth scaling?
3. How sensitive is the continuous time setup to system noise? Biological neurons are stochastic, so it is natural to wonder whether how resilient is this model to random (e.g. Gaussian) input or neural output perturbations.
4. The author propose that skip-connection might help overcome the overlap problem for deeper networks. Can you show a proof-of-principle scenario (by adding skip connections to depth-3 MLP layers) showing this effect?
5. Why does the continuous time model outperform the discrete analogs in Table 2?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies learning in continuous time through a coupled system of ordinary differential equations for neural activations, forward weights, and feedback weights. Under specific timescale separations, the authors show that common learning rules such as SGD, feedback alignment (FA), direct feedback alignment (DFA), and Kolen–Pollack (KP) emerge as limiting cases. They analyze how temporal mismatches between input and error signals affect learning, proposing that successful adaptation requires temporal overlap between input and error at each synapse. Simulations on small datasets illustrate the influence of propagation delay, potentiation timescales, and network depth. The authors claim that this continuous-time formulation removes the need for separate inference and learning phases, yielding biologically and physically plausible conditions for learning.

### Strengths
* The modeling framework is coherent and mathematically consistent. The coupled ODE formulation produces recognizable update rules, and the derivations are cleanly presented.
* The exposition connects well-known biologically inspired algorithms within a single dynamical formalism. This can serve as a pedagogical bridge between biologically motivated learning rules and continuous-time dynamical systems used in neural ODEs.
* The analytical treatment of delay and temporal overlap is technically correct and illustrated with simple examples. The results align with simulations and provide an intuitive picture of how propagation delays degrade performance.
* The empirical evaluation, though small in scope, is well executed and matches theoretical expectations. The relationship between potentiation timescale and sample duration is shown convincingly.

### Weaknesses
- The conceptual novelty is limited. Continuous-time learning with eligibility traces and temporally overlapping pre- and postsynaptic activity has been extensively studied in computational neuroscience (e.g., Frémaux & Gerstner, 2016; Urbanczik & Senn, 2014; Izhikevich, 2007). The central claim that learning succeeds when input and error overlap in time is a restatement of long-established three-factor plasticity theory. The paper does not acknowledge this continuity or differentiate its contribution from that body of work.
- The purported unification of SGD, FA, DFA, and KP through continuous-time dynamics is primarily formal. All stepwise update rules admit continuous-time limits under small step sizes, so expressing them as ODEs does not constitute a substantive theoretical synthesis. The true differences among these algorithms lie in feedback structure and alignment, not in discretization.
- Missing engagement with the spiking backprop literature. There is a substantial body of ML and comp-neuro work showing gradient-based learning in spiking networks via surrogate gradients or analytically derived rules (e.g., Lee et al., 2016; Shrestha & Orchard, 2018; Zenke & Ganguli, 2018; Neftci et al., 2019; Bellec et al., 2020). These already provide continuous-time, biologically motivated training signals and explicit analyses of temporal credit assignment in spiking dynamics. The paper neither compares to nor positions itself relative to these approaches.
- The biological grounding is superficial. While time constants are drawn from plausible ranges, the model remains rate-based and omits critical biological features such as spiking and neuromodulation. The work does not generate new physiological predictions beyond known timescale requirements for eligibility traces.

### Questions
- In what sense their formulation advances beyond established continuous-time or eligibility-trace models in computational neuroscience? 
- How does this framework relate to spiking backprop/surrogate-gradient methods (Lee et al., 2016; Shrestha & Orchard, 2018; Zenke & Ganguli, 2018; Bellec et al., 2020, Neftci et al., 2019)? Are there concrete advantages (theoretical or empirical) over these continuous-time, gradient-propagating spiking formulations?
- Since FA and DFA differ from SGD mainly in feedback structure, what is unified by casting them in continuous time?

### Soundness
2

### Presentation
2

### Contribution
1

# Homeostatic Adaptation of Optimal Population Codes under Metabolic Stress

- Avg Score: 5.33
- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Information processing in neural populations is inherently constrained by metabolic resource limits and noise properties, with dynamics that are not accurately described by existing mathematical models. Recent data, for example, shows that neurons in mouse visual cortex go into a "low power mode" in which they maintain firing rate homeostasis while expending less energy. This adaptation leads to increased neuronal noise and tuning curve flattening in response to metabolic stress. We have developed a theoretical population coding framework that captures this behavior using two novel, surprisingly simple constraints: an approximation of firing rate homeostasis and an energy limit tied to noise levels via biophysical simulation. A key feature of our contribution is an energy budget model directly connecting adenosine triphosphate (ATP) use in cells to a fully explainable mathematical framework that generalizes existing optimal population codes. Specifically, our simulation provides an energy-dependent dispersed Poisson noise model, based on the assumption that the cell will follow an optimal decay path to produce the least-noisy spike rate that is possible at a given cellular energy budget. Each state along this optimal path is associated with properties (resting potential and leak conductance) which can be measured in electrophysiology experiments and have been shown to change under prolonged caloric deprivation. We analytically derive the optimal coding strategy for neurons under varying energy budgets and coding goals, and show how our method uniquely captures how populations of tuning curves adapt while maintaining homeostasis, as has been observed empirically.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce a new method for modelling the interaction of metabolic stress and tuning curves. They show that different from previous models, their model is capable of reproducing the experimental findings reported in (Padamsey et al., 2022) - mainly that under metabolic stress, tuning curves widen and reduce in height. To derive model parameters, they hypothesise that cells adjust neuron parameters (rest potential, leak conductance) to minimise noise levels under varying energy levels.

### Strengths
1. The presented model is a simple extension of previous ones, which can be recovered by setting the function eta and parameter alpha accordingly.
2. Analytical solutions are provided for their model for different optimisation objectives.
3. They use simulations inspired by the experimental protocol used in (Padamsey et al., 2022) to fit alpha and eta in a biologically grounded way. This is the main difference between previous models and the proposed one, leading to alpha = 1 (which has also been used in Ganguli & Simoncelli, 2010), and a functional relationship for eta.
4. Their model faithfully reproduces experimental findings. They compare this with two alternative models, which fail to fully capture experimental observations. In particular, only their model shows firing rate homeostasis.

### Weaknesses
1. Since the introduced model is calibrated using simulations, it is not clear how well it will generalise to other experimental setups.
2. In general, the part about extracting eta_k and alpha from simulations is quite dense and hard to follow, which could be improved.
3. Some of the used variables have to be explained better, as it is not evident what they are or how they are determined (see questions).

### Questions
1. As far as I can see, the main novelties are: 1) The activity and energy-dependent dispersion, 2) The generalised energy constraint, although in the derived model, alpha = 1 is used which aligns with previous models. 3) The fit to simulated results (with energy estimates tying neural activity to ATP needs) to  obtain alpha and eta. Is that correct?
2. Eq. (1) is for Poisson neurons? A citation would be helpful.
3. What exactly is kappa? How is it related to R(s), R_n, R, etc.? Also, how is R(s) determined?
4. Eq. (7): E is stated to be energy. However, since this expression is valid for any alpha >= 1, this is weird unit-wise (and in fact, for alpha = 1, the authors identify E as the mean rate R). 
5. The final model is more complex than the two comparison models (Ganguli & Simoncelli; Wang et al.), and parameters are actually fit - not on experimental data, but using simulations. So it seems like increased expressivity is paid for by an increase in complexity. Could you comment on how well you expect this to generalise to other experimental setups? In particular, does your model predict any other effects?
6. If I understand correctly, eta_k is obtained for different combinations of v_rest and g_leak. For activity kappa and a given energy level, you then determine the values of (v_rest, g_leak) that minimise eta_k (which is later used to fit a function for eta_k)? This part was quite dense to read, so a few more explanations might be helpful here.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper tackles the adaptation of tuning curves in optimal networks in situations of metabolic stress. Authors model the observation that under metabolic stress, tuning curves of neurons in the mouse visual cortex show flattening while the average firing rates are maintained. A theoretical framework describing tuning curves is developed that captures such flattening of tuning curves in situation of energy limit that affects noise levels in single neurons. Simulation of biophysical neurons are carried out to show that the mechanism suggested by the theory can be also seen in the biophysical model of single neurons.

### Strengths
The theoretical framework as well as numerical simulations seem technically sound and carefully conducted (even though I had not checked all the details). The question attacked by the paper is  seldom explored and is of biological relevance. Some of the relevant weaknesses are appropriately discussed.

### Weaknesses
1) Main weakness is the assumption of conditionally independent firing rates. Cortex, including the visual cortex of the mouse, consists of highly recurrently connected networks of Excitatory and inhibitory neurons that have strong influence on each-other's activity (see for example Chettih and Harvey, Nature 2019). I am not convinced that the assumption of independent neurons can bring crucial insights about the neural code in the cortex, including the study of tuning curves. Can authors comment why do they think that a model with independent neurons is relevant?

2) There is a lack of citations on several occasions. 
Examples:
[line 50 ] Authors mention "Simple models of energy constrains" but not cite any so it remains unclear which models they are referring to. 
[line 54] ..."previous population coding models capture either the shortening or the widening of energy-limited tuning curves, but not both". Which previous models show this and why are they not cited at this point? 

3) Authors situate the current work among the existing literature in a way that seems rather problematic. In line [122] authors comment on the importance of the activity of the sodium/potassium pump for the metabolic expenditure of neurons. The activity of the sodium/potassium pump seems to be a direct consequences of spiking activity, and the maintenance of the reversal potential also depends on neuron's firing rate. It can be argued that models that formulate the metabolic cost that is dependent on the firing rate or the number of fired spikes nevertheless capture, even though indirectly, sources of metabolic expenditure that are biologically highly relevant. Such efficient models simply do not model the situation of important metabolic stress, but instead use the metabolic cost as a means to regulate the activity levels in the network and constrain the solution appropriately (in a general situation without energy deprivation). While a state of energy deprivation seems an important constraint from the  evolutionary perspective, I am not convinced that neural coding would only adapt to this particular constraint.

Recent research (Gutierrez and Deneve, eLife 2019, Koren et al., eLife 2025) has discussed the role of metabolic efficiency in optimal spiking networks. Both studies use a metabolic constraint that is formulated as a cost on the number of spikes fired. The first study (Gutierrez et al.) found a formulation of efficiency that gives rise to a transient adaptation in single neurons on a time scale of seconds. The second study (Koren et al.) found that a number of biophysical parameters in primary cortex can be captured when the metabolic efficiency is taken into account, and that metabolic efficiency was shown to have an important effect on optimal coding solutions. Can authors comment on their results in light of these two studies and also touch on the difference in the use of the metabolic efficiency in their model compared to the one in papers mentioned above? 

4) The paper is difficult to read and could use a rewrite with focus on clarity. I believe bringing in some more simplicity when possible would make the paper much more appealing. In particular, the first paragraph of discussion gives a good intuitive motivation for the model and I suggest to move this part earlier in the paper, possibly to the introduction.

### Questions
1) What kind of neuron model is used in simulations? I believe it is a Hodgkin-Huxley model, but this is not specified.
2) In line [300], authors mention mean spike count of 0.8 and 0.2. It is unclear to me what are the units used here?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The idea that brain representations are fundamentally shaped not only by input statistics but also on metabolic constraints has proven very important for our understanding of sensory coding. Nonetheless traditional efficient coding models don;t capture the exact nature of the constraints which leads to incorrect predictions in the low energy regime compared to recent experimental measurements (Padamsey et al., 2022). The paper introduces a new way of expressing these energetic constraints that is mathematically tractable and biophysically well justified, part of a general framework that has past model as special cases, and accounts for the experimental observations that demonstrate increased noisiness as the price of more stringent energy constraints.

### Strengths
Clean mathematical formalism for coding efficiency with new constraints.

Biophysical simulations that link mechanistic considerations with coding level abstraction. 

Explains for the first time recent experimental observations on the effect of limited energy availability on neural coding.

### Weaknesses
While i do enjoy the mathematically clean formulation, the change in the constraint is in and of itself an incremental contribution at the technical level. 

The link to data and discussion sections are very brief and need expansion.

Numerical results are very minimal.

### Questions
Neural results and discussion: i would have liked to see some concrete predictions that the model makes and perhaps some comments of how the energy-dependent encoding process affects processing downstream

some additional validation of results in simulations, e.g. documenting how other coding considerations affect the degree of flattening would make the coding numerics more substantial.

### Soundness
3

### Presentation
3

### Contribution
3

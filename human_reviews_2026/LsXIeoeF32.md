# Ubiquity of Hebbian Dynamics in Complex Learning Rules

- Decision: Reject
- Scores: 8, 4, 2, 4, 4

## Abstract
Hebbian and anti-Hebbian plasticity are widely observed in the biological brain, yet their theoretical understanding remains limited. In this work, we find that when a learning method is regularized with L2 weight decay, its learning signal will gradually align with the direction of the Hebbian learning signal as it approaches stationarity. This Hebbian-like behavior is not unique to SGD: almost any learning rule, including random ones, can exhibit the same signature long before learning has ceased. We also provide a theoretical explanation for anti-Hebbian plasticity in regression tasks, demonstrating how it can arise naturally from gradient or input noise, and offering a potential reason for the observed anti-Hebbian effects in the brain. Certainly, our proposed mechanisms do not rule out any conventionally established forms of Hebbian plasticity and could coexist with them extensively in the brain. A key insight for neurophysiology is the need to develop ways to experimentally distinguish these two types of Hebbian observations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
There is wide empirical evidence for Hebbian learning in the brain. This paper challenges this by showing that strong correlation with Hebbian learning can be achieved by many learning rules that have, in principle, nothing to do with Hebbian learning, as long as they are coupled with weight decay. Additionally, it shows noise on the parameters makes the learning signal more anti-Hebbian. The claims are supported by heuristic theoretical derivations at stationarity as well as experiments on relatively small neural networks.

### Strengths
The arguments presented in the paper are simple yet powerful, and raise significant questions regarding our understanding of synaptic plasticity in the brain. The theoretical arguments provide solid intuition that is shown to hold in practice through careful experimentation. The discussion related to neuroscience is well done, and I had the pleasant surprise of finding all the questions the paper raised during my first read carefully addressed.

### Weaknesses
What follows should be considered as suggestions rather than weaknesses:

- **Figure 4 could be moved to the appendix.** The results it shows are very intuitive (as mentioned in the main text), and the setup differs from the rest of the paper, requiring considerable time to understand what is happening. As a result, reading this figure alone may bring more confusion than clarity.
- **The analysis in Section 4 could be made more precise**. The setup considered in this section is simple enough that closed-form solutions of the learning dynamics can be derived (see e.g., Saxe et al. 2013), allowing the evolution of alignment with the Hebbian update to be precisely characterized. This could help better understand the alignment dynamics.
- **The connection to empirical evidence could be strengthened.** The paper shows that traces of Hebbian learning do not necessarily imply that the underlying learning rule is Hebbian, without directly commenting on whether current empirical evidence suffers from this problem. The following experiment would help: reproduce the methodology of empirical neuroscience papers establishing the existence of Hebbian learning in the brain that the authors cite, and show that the same results can be achieved by a version of SGD. I suspect this would be possible given the current results of the paper, and having this experiment would help make the argument that current evidence is insufficient to conclude that brains learn with Hebbian learning.

### Questions
-

### Soundness
4

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
The authors show that in the presence of weight decay, the "learning signal" caused by SGD will tend to align with Hebbian updates (input time output), whereas in the presence of high weight noise it will instead tend to align with anti-Hebbian updates.

Several experiments confirm the intuitive results.

### Strengths
The general approach of predicting which conditions will align learning with Hebbian or anti-Hebbian updates seems novel, to my knowledge.  The experiments, if correct, seem to confirm the proposals.

Some of the results are genuinely unexpected, e.g. the neat quadratic boundary shown in Figure 5.

### Weaknesses
The main problem is that the paper consistently presents itself as an explanation of *why* Hebbian/anti-Hebbian learning occurs, while being based entirely on homeostatic mechanisms and completely ignoring functional outcomes of Hebbian learning.

For example, Hebbian learning is a powerful pattern learner, and anti-Hebbian learning enforces decorrelation (a famous example of both together is Foldiak 1991 https://pubmed.ncbi.nlm.nih.gov/2291903/ ).

Since (as the authors acknowledge) Hebbian updates will tend to be aligned with the weight vectors (at least after a certain amount of learning) and thus increase W magnitude, then keeping weights stationary in the face of weight decay basically requires some kind of alignment with Hebbian updates. 

To make this obvious homeostatic necessity a general principle, you first need to assume that something like uniform weight decay occurs in the brain, which is not at all obvious (e.g. I understand daily turnover of synapses tends to erase weak synapses but preserve strong ones). But even if it did, it would not follow that this should explain all, or even most, of Hebbian/anti-Hebbian learning in the brain, which the paper basically hints at repeatedly.

Similarly, observing anti-Hebbian updates in the presence of noise would not exactly confirm the theory, since this is precisely the expected outcome of several well-known Hebbian learning rules, including BCM and most STDP rules (which involve larer negative than positive windows). The reason for this is precisely to degrade synapses between neurons that fire uncorrelatedly.

Thus, while some intriguing results are reported, the paper needs to be significantly "toned down" and clarify the reach of its proposals.

### Questions
- Eq.4: please add some more parentheses, or spaces - it's not immediately clear what is included in the gradient sign (I think it is just the gradient of l, but please clarify)

- Please expand the maths  between eq 4 and eq 5 (maybe in an appendix). How do you get from 4 to 5 ? Why did the h_a become a second h_b? Why are there traces involved?

- In equation 9, what exactly is theta, and how does it differ from W? Why "cosine similarity between the learning rule and the Hebbian rule" is somehow cos Theta? And, again, where do the traces come from?

- Please explain Eq 17 a bit more. What are c0 and c1, and are they really "constants"?

- Where's the data to support the next-to-last paragraph in p. 8 (line 420-424)?

- In Figure 7, what does "Init: 0.5x", etc. mean?

- I note that the Hebbian update is simply the gradient of y^2, so the initial bump of Hebbian-ness in Figure 7 may be caused by the need to increase response outputs, regardless of overall W magnitude. Perhaps plotting y (output) magnitude would be useful.

- Minor: fix the parentheses in line 131.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper claims that Hebbian and anti-Hebbian plasticity emerge universally from gradient-based optimization methods when combined with L2 weight decay or noise. The authors derive analytic results showing that learning rules with L2 regularization align with Hebbian updates near stationarity, while stochastic noise induces anti-Hebbian alignment. They support their theory with experiments on small MLPs and Transformers, suggesting that observed Hebbian/anti-Hebbian plasticity in the brain might be a signature of general optimization processes.

### Strengths
- The paper is clearly written and structured.
- The attempt to bridge biological learning and gradient-based optimization is an important unsolved question.
- The experiments provide systematic and broad qualitative confirmation of the claimed effects.

### Weaknesses
The following weaknesses highlight fundamental conceptual and theoretical issues that undermine the relevance of the paper’s main claims, distinguishing between superficial equilibrium effects and genuine learning dynamics.
- The paper equates the norm-stabilizing effect of L2 regularization with Hebbian dynamics. L2 weight decay enforces contraction of weight norms, while Hebbian learning refers to directional correlation between pre- and post-synaptic activity (e.g., feature extraction). The alignment the authors observe is a trivial consequence of regularization equilibrium (($\nabla_\theta \ell + \gamma W = 0$)), not evidence of Hebbian computation.
- Classical linear Hebbian rules, when combined with weight normalization or decay, are mathematically equivalent to stochastic gradient descent on the PCA objective (Oja, 1982). This connection is well established and forms the foundation of Hebbian learning theory. The authors fail to acknowledge or engage with this known property and instead conflate PCA-type Hebbian learning with the L2 norm stabilization. 
- Hebbian learning, in particular nonlinear forms, implements unsupervised feature learning objectives such as PCA, ICA, or sparse coding (e.g. Oja 1982; Oja 1991; Clopath et al. 2010; Zylberberg et al. 2011). These rules learn statistical structure beyond norm stability. The paper’s framing ignores this and instead treats any gradient-norm correlation as Hebbian, which overlooks decades of theoretical and experimental work.
- The claim that anti-Hebbian alignment results from noise or that Hebbian signatures imply hidden global optimization lacks quantitative or mechanistic justification. Real synaptic plasticity involves nonlinear, spike- or voltage-dependent mechanisms (e.g., BCM, triplet STDP, Clopath rules) absent from this discussion.
- The paper ignores key models showing anti-Hebbian plasticity as a structured, biologically grounded mechanism for decorrelation and stability, not a noise artifact (Vogels et al. 2011; King et al. 2013).
- The manuscript overlooks nonlinear Hebbian learning frameworks that unify Oja’s, BCM, and triplet-STDP rules, as well as recent work linking these to ICA and sparse coding. These studies demonstrate that Hebbian-like rules do much more than stabilize norms, by extracting higher-order structure from data. The omission leads to incorrect generalization about Hebbian dynamics.

### Questions
1) Can the authors clarify whether their “Hebbian alignment” has any relationship to the PCA or ICA objectives known to be optimized by Hebbian-like rules? If not, how is it meaningful beyond norm equilibrium?
2) How does the proposed theory differentiate between trivial weight-norm alignment and genuine correlational structure learning? Would linear networks on whitened data still exhibit the claimed effects?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work explores under what conditions Hebbian and anti-Hebbian learning signal emerges from gradient-based optimization rules. They thereotically and experimentally find that using L2 weight decay, a broad class of learning rules exhibit Hebbian-like alignment of the learning signal near stationarity. With sufficiently strong noise, the alignment flips to anti-Hebbian. However, experiments on more broad class of neural network architecture are missing.

### Strengths
- A mathematical framework  shows why any learning rule with weight decay should exhibit Hebbian-like alignment at stationarity
- In linear settings (and empirically in nonlinear ones) the work shows that sufficiently large parameter/gradient noise reverses alignment.

### Weaknesses
* The paper claims that Hebbian/anti-Hebbian signals may be auxiliary emergent phenomena of complex learning rules. However, this claim is too strong. The presented theory and experiments more naturally support a stronger statement: Hebbian learning signals can arise from gradient-based rules under L2 weight decay regularization. Or, Hebbian-like updates can be implemented by multiple learning paradigms.

* The current validation is limited to small networks. The paper should include larger-scale models (e.g., ResNet, compact LLMs) and analyze how depth, initialization, activation functions, or others shape Hebbian versus anti-Hebbian orientation. Demonstrating the effect across a broader model family would substantially strengthen the conclusions.

* The paper should investigate the trade-off between the L2 regularization coefficient and task performance, and quantify how “more Hebbian-like” alignment relates to accuracy/generalization.

### Questions
* The theoretical analysis focuses on L2 weight decay. Although L1 and Dropout are mentioned, the manuscript should make explicit what learning directions these regularizers induce and how they compare to L2 in driving Hebbian or anti-Hebbian tendencies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes the idea that 
 phenomenologically Hebbian and anti-Hebbian plasticity can emerge as byproduct of much more general learning rules (gradient based) that include weight decay (a contractive force) and/or noise (an expansive force). 


The authors postulate that gradient descent training with weight decay (especially with high weight decay values) results into weight updates that are similar to those happening under purely Hebbian learning.
They offer a clear argument that since weight decay is contractive, the learned gradient part of the update must be expansive on average, and an expansive, rank-1-looking update will look Hebbian. That is an interesting explanation for why Hebbian-looking updates can show up during learning with weight decay even when the underlying algorithm is not Hebbian. They find that  stronger weight decay, larger learning rate, and larger batch size
lead to better alignment between gradient-based and Hebbian weight updates.
Moreover they find that strong noise in learning results in a learning signal that is anti-Hebbian. They mention that when noise co-exists with weight decay there is a competition between the two forces that contribute Hebbian and anti-Hebbian aligned updates, and they identify a “phase transition” when the interplay between these two forces changes polarity. 

In general the authors try to push the argument that Hebbian and anti-Hebbian plasticity might be a byproduct or components of a more general gradient-like weight optimisation. While I find it an interesting argument, I find their evidence and argumentation not strong enough to support this argument, while their theory mostly holds for stationary states. This argumentation requires stronger biological anchoring and a clearer mapping from the employed weight decay to biological decay processes.

### Strengths
- The authors have performed an extensive number of numerical experiments where they explore the alignment of the weight updated with the Hebbian weight updates for different parameter values of weight decay, noise amplitude, network sparsity, network size, learning rate, batch size, and for different learning rules (namely Adam, stochastic gradient descent, direct feedback alignment, and randomNN) and regularisers.
- The authors explicitly demonstrate that is the gradient part of the weight update that aligns with the Hebbian updates and not the full weight update, and thus there is no issue with weights growing infinitely large (since there is also a weight decay)
- The authors propose an interesting theory: gradients under realistic constraints project onto Hebbian-looking directions, and thus observing Hebbian plasticity in experiments does not rule out gradient-like learning.

### Weaknesses
- I feel the assumption of stationarity  does the heavy lifting in the arguments of the paper, and there is no concrete proof or evidence whether this Hebbian alignment arguments hold also in the out-of equilibrium state.
- the title of the paper overstates the finding, since in essence this holds only under the assumption of stationarity and weak coupling.
- No statistical reporting in many of the figures, the results seem like single runs (or the authors omit to mention over how many realisations they average).

### Questions
# Questions


- In figure 15 you are trying to show that the alignment persists all over training, however isn’t the Hebbian alignment of the learning signal a bit too weak throughout this experiment? 
- What do the blue and red lines in Figure 3 left indicate? Is it Layer 1 and Layer 2 weight update alignment with Hebbian? Please put a legend key.
- What experiments do you think one could perform to validate your proposal?  if Hebbian signatures can be a by-product of many learning rules, which empirical measurement/experiment would falsify/validate the argument?
- You show that any update rule with decay will, on average, look Hebbian. Does that mean that much of the experimental evidence for Hebbian STDP could be reinterpreted as "we only ever observed the projected, stationary part of a richer gradient estimator"?
- You identified a Hebbian - anti-Hebbian transition controlled by (noise, weight decay). Can this be re-read as having Hebbian-like updates when the gradient estimation is clean, and anti-Hebbian when the estimation is noisy, i.e. interpret the boundary as a quality-of-gradient axis?
- In [1]  the authors mention heterosynaptic pathways as ways to direct gradient information. In the paper you show that heterosynaptic  rules also become Hebbian under decay. Does that mean these anatomical pathways could be there mainly to improve the pre-projection gradient, with Hebbianity just the surface readout?
- In Figure 16 caption the authors mention that batch normalisation seems to have anti-Hebbian effect, however as I understand the plot the effect is very small (alignment value <-0.1). Can you comment on how you support this statement?



### Typos and other comments:

- Line 86: Eq. equation
- In eq. 4 what is $\ell$?
- Line 174: “with” is missing

### Papers to be cited

[1] Richards, Blake Aaron, and Konrad Paul Kording. "The study of plasticity has always been about gradients." The Journal of Physiology 601.15 (2023): 3141-3149.

### Soundness
3

### Presentation
3

### Contribution
2

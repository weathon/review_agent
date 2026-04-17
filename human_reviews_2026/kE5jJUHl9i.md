# Stable and Scalable Deep Predictive Coding Networks with Meta-Prediction Errors

- Decision: Accept (Poster)
- Scores: 8, 6, 2, 6

## Abstract
Predictive Coding Networks (PCNs) offer a biologically inspired alternative to conventional deep neural networks.
However, their scalability is hindered by severe training instabilities that intensify with network depth. 
Through dynamical mean-field analyses, we identify two fundamental pathologies that impede deep PCN training: 
(1) prediction error (PE) imbalance that leads to uneven learning across layers, characterized by error concentration at network boundaries; and
(2) exploding and vanishing prediction errors (EVPE) sensitive to weight variance.
To address these challenges, we propose Meta-PCN, a unified framework that incorporates two synergistic components: 
(1) a loss based on meta-prediction error, which minimizes PEs of PEs to linearize the nonlinear inference dynamics; and 
(2) weight regularization that employs normalization to regulate weight variance and mitigate EVPE.
Extensive experimental validation on CIFAR-10/100 and TinyImageNet demonstrates that Meta-PCN \rev{achieves} statistically significant improvements over conventional PCNs, outperforming backpropagation in most tested configurations, while preserving the local learning rules of PCNs.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper empirically illustrates 2 problems, imbalance and unstable PEs, of PCN with high numbers of layers. Based on that analysis, the paper propose a framework to mitigate the problems based on Meta-PC objective and weight regularization. The paper then demonstrates the effectiveness of the framework on VGGs and Resnet-18 in CIFAR10/100 and TinyImagenet.

### Strengths
I find the reasoning flow of the paper clear and strong. I am also impressed in how the proposed technicalities can improve PCN significantly, matching traditional back-propagation approach.

### Weaknesses
It will be an excellence paper if the authors can demonstrate the method in larger dataset like Imagenet or demonstrate the scalability in text data on more recent language models. Nevertheless, I think this is a strong, solid paper already.

### Questions
1) Resnet-18 seems to have lower gap between PCN and Meta-PCN. Why is that?
2) How do you implement PCN and Meta-PCN into VGGs and Resnet-18? Are there any modifications in the architectures or the only changes are in inference and learning procedure?
3) Ablation results on Meta-PCN objective and Weight regularization, i.e., how each modifications improve PCN?
4) Are there any hyper/tuning-parameters related to Meta-PCN? It seems this method is parameter-free to me, but I would like to have a clarification on this.
5) Are there any future research directions can be beneficial from this framework based on PCN?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
### Summary

The paper introduces Meta-PCN, a framework designed to stabilize deep Predictive Coding Networks, which traditionally face severe scalability issues. The authors identify two causes for this instability (1) *prediction error (PE) imbalance* and (2) *exploding/vanishing prediction errors (EVPE)*.

The authors employ dynamical mean-field theory to characterize these instabilities and propose two core innovations:

1. **Meta-prediction error loss**, which minimizes prediction errors of prediction errors, effectively linearizing inference dynamics.
2. **Variance-based weight regularization**, which controls weight variance to prevent EVPE and ensure balanced signal propagation.

Empirical results across CIFAR-10/100 and TinyImageNet show substantial stability gains and accuracy improvements of Meta-PCN over PCNs while maintaining biological plausibility.

### Strengths
### Strengths

- **Convincing Theoretical results:** The paper provides a rigorous dynamical mean-field analysis of PCN inference dynamics, clearly identifying root causes of instability (Sections 3.2–3.3).
- **Experiments supporting theoretical claims:** Results show experimentally the root causes of instability. Figures 2–3 nicely illustrate the characteristic U-shaped PE imbalance and exponential EVPE dynamics; Figures 4–5 convincingly show that Meta-PCN mitigates these.
- **Novel conceptual contribution:** The idea of *meta prediction errors* offers an alternative to avoid gradient starvation.
- **Strong connection to neuroscience:** The work maintains biological plausibility, an property of high relevance for both ML and computational neuroscience.

### Weaknesses
- **Implementation comparison gaps:** The predictive coding baseline underperforms relative to prior works (e.g., Pinchetti 2025), raising questions about whether the reference implementation is optimal.
- **Connection to backpropagation not fully unpacked:** The equilibrium of the new loss (Section 4.1) appears mathematically equivalent to an iterative local implementation of backpropagation (δₗ = g(δₗ₊₁, hₗ₊₁⁰)), but the paper doesn’t explicitly acknowledge this equivalence or discuss implications for biological plausibility.
- **Missing mention of the assumptions of the length analysis**: eg. linear activation function and all layers share the same size. These assumptions should be clearly stated in the main text (section 3).
- **Limited experimental details**: The main text should contain necessary information to understand how the experiment were run broadly. Some additional details should be added, for instance the activity initialisation used for PCNs in figure 2.

### Questions
I really enjoyed reading your manuscript, especially section 3. However, there remain a few open questions that would need to be addressed before I could confidently raise my score to an 8. Could you please clarify the following:

**Major**

- **Ablation study:** Based on Figure 2, EVPE appears to be the dominant source of instability in PCNs. However, the ablation results presented by the authors do not include a direct comparison of a standard PCN augmented only with the proposed weight regularization. The experiments instead compare Meta-PCN variants with and without this regularization. Demonstrating that weight regularization alone is insufficient for achieving scalable learning in PCNs would be important to substantiate the necessity of the full Meta-PCN framework.
- **Precision learning:** It appears that optimizing the precision (1/variance) of PCNs layers would eliminate the PE imbalance by rescaling each error based on its variance (Bogacz 2017). Do you expect this to be the case? While the precisions are often assumed to be identity in PCNs for computational simplicity, they remain an essential part of PCNs that should be considered in your analysis.
- **backpropagation equivalence:** At equilibrium, the meta-PE formulation appears exactly equivalent to backpropagation ($\delta_l = g(\delta_{l+1}, h_{l+1}^0)$) but obtained using a local objective, could you clarify this relationship formally?
- **Implementation discrepancy:** Why does your predictive coding baseline perform so much worse than previous works (e.g., Pinchetti 2025)?

**Minor**

- **Energy propagation:** Why does error concentration persist at the output boundary (Fig. 4a)? Does this suggest incomplete energy flow across the network?
- **Baselines:** Did you apply *weight normalization* to the BP baseline for a fair comparison?
- **Biological interpretation:** Can Meta-PCN be implemented in a local neural circuit model consistent with cortical predictive coding mechanisms?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes a theoretical analysis of predictive coding models, as well as a solution to the problem of training very deep networks. This solution is based on two novel proposals: the use of meta-prediction errors, and a new kind of normalization of the weight matrices. A large empirical evaluation shows the effectiveness of the proposed method on a large number of architectures, where the comparison is made agains variations of PC, and backprop.

### Strengths
The paper is well motivated, tackling a timely problem in the field of predictive coding. The empirical evaluation is well done, as it tests on a large number of models and datasets, showing clear improvements in performance. Also, the use of dynamical mean field theory to study  the instability/vanishing error problems of PCNs is novel and interesting.

### Weaknesses
There are a couple of problems in the solutions proposed by the authors, that affect the novelty and bio-plausibility of the meta prediction errors, that is, the fact that they rely on the value of the neurons during the forward pass (so, at t=0). The problems are the following:

1) This is not novel, but it has already been done in three other works, that you also cite: Whittington&Bogacz,2017, B.Millidge 2020 (Predictive coding approximates backprop along arbitrary computation graphs), and C.Qi 2025 (Towards the Training of Deeper Predictive Coding Neural Networks). How is the energy function you describe different from theirs? 

2) This is also biologically implausible, as already stated in the three works above, as it breaks the locality in time that define what biologically plausible algorithms are  (operations must rely on information that is locally available at that time step). 

A second problem in terms of novelty, is the claim that you have identified the problems of vanishing PE, and PE imbalance across the network. While I agree that the theoretical analysis you propose is novel, I believe you should state that both problems have been already identified in previous work: [1] shows this problem in a small neural network, while [2,3,4] propose their theoretical analysis. So i would rephrase your list of contributions by acknowledging existing work first, the current state of the art, and from there list your contributions. On a side note, I'm not a big fan of works that place the related works in the supplementary material, as it leads exactly to this kind of confusion. Maybe you could move a summary that discusses only these points in the main body?

Minor: 

you propose two methods to improve the results, but you only test them simultaneously. I would add an ablation study so that the reader can understand which method has the largest impact in the final test accuracy.

In figure 2, what is the exact setup? Is it a randomly initialized model? Is it already trained? On which data?

At some point you cite:

*Beren Millidge, Alexander Tschantz, and Christopher L Buckley. Predictive coding networks for
temporal prediction. Neural Networks*

This is a wrong citation, maybe you meant this one, with the same title?
https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183



[1] L .Pinchetti et al. Benchmarking predictive coding networks -- Made Simple (2024)

[2] C.Qi et al. Towards the Training of Deeper Predictive Coding Neural Networks (2025)

[3] Cédric Goemaere et al. Overcoming Exponential Signal Decay in Deep Predictive Coding Networks (2025)

[4] Francesco Innocenti et al. uPC: Scaling Predictive Coding to 100+ Layer Networks (2025)

### Questions
See above. My main concern at the moment is the novelty of the work, so I'd be happy to reconsider my score if this concern is properly addressed. (Also, the score I gave is a little too harsh in my opinion, as the work does have value. However, ICLR this year does not allow for intermediate scores, and 4 is already considered borderline...)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper uses a dynamical mean-field theory (DMFT) analysis to diagnose two failure modes in deep predictive coding networks (PCNs): (1) an imbalance of the local energy functions (prediction errors) across depth, (2) exploding/vanishing prediction errors (EVPE). To address these, it introduces Meta-PCN, which (1) uses a meta prediction error objective that linearises the PC equilibrium map around the feed-forward state and trains errors to satisfy the PC delta relation, and (2) applies variance-based weight normalisation for efficient spectral control. In empirical results, Meta-PCN outperforms conventional PC and is competitive with backprop on CIFAR-10/100 and TinyImageNet; ablations indicate the meta-objective is critical. 
This paper makes a significant contribution to the predictive coding literature, both in its analysis of PCNs in terms of DMFT and the proposed Meta-PCN. I therefore recommend that this paper be accepted, under the constraint that the weaknesses highlighted below are sufficiently addressed.

### Strengths
Clear diagnosis of PCN failure modes via DMFT. Identifying depth-wise energy imbalance and EVPE gives concrete targets for stabilisation.

Simple, principled meta-objective. Linearising around the feed-forward state and training errors to satisfy the PC delta relation is elegant and leads to stable inference without heavy tuning. 

Compelling empirical evidence. Consistent gains over conventional PC and competitive performance with backprop across CIFAR-10/100 and TinyImageNet, plus ablations that isolate key components.

### Weaknesses
Unclear optimisation of parameters. The paper introduces a new error-based objective to stabilise inference, but does not establish how parameter updates relate to this objective. In effect, inference is performed under one criterion while learning appears to optimise another. Please (i) state explicitly what loss the weights optimise, and (ii) provide diagnostics showing the stated training objective consistently decreases over epochs (a descent argument or convergence guarantee would be even better).

Clarify ablation definitions: One of the most convincing results is in Appendix G, where you compare with and without the Meta-PCN objective. However, it is unclear if this includes the fixed prediction (i.e., you fix the initial forward pass and then perform standard free-energy minimisation). If this is the case, please clarify; if not (you do not fix the forward pass), this should be included as an additional ablation to tease apart the Meta-PCN objective from the fixed predictions.

Definition of the stabilised error: The objective states that the stabilised error is a fixed top-down target, but the paper does not give an explicit construction, presumably from the last iterate, but this is not clear.

Acknowledge trade-offs in freezing feed-forward prediction: In the PC literature, iterative inference over hidden variables is a key benefit. Freezing the feed-forward prediction and optimising errors removes that benefit; this is a reasonable engineering choice, but it should be clearly flagged as a departure from traditional PC.

Weight “variance normalisation” claims overreach given non-iid, structured weights: The normalisation argument assumes random, dense matrices. Convolutional weights are structured and evolve during training. Without measuring actual spectral behaviour before/after normalisation, claims of “robust spectral control” are speculative. This should be clarified in the main text.

Linear DMFT assumptions don’t extend to the nonlinear networks used in experiments: Theory is developed in a linearised regime with random weights; experiments use deep nonlinear models with structure. The paper leans on the theory to justify design choices, but doesn’t bridge the gap. This should be highlighted in the main text rather than dealt with in the appendices.

Backprop baselines aren’t tuned to standard practice for the architectures/datasets you use: Given the modest margins, any claims of “outcompeting backprop” should be softened until stronger baselines are included.
Training loop pseudocode: Provide a short, concrete algorithm box covering initialisation, T inference steps, parameter update, and when normalisation is applied.

Computational overhead: The Meta-PCN framework adds multiple components (meta-objective computation, weight normalisation, blocked sweeps) that likely increase computational cost, but this isn't discussed or quantified. 
Minor points Abstract: “demonstrates that Meta-PCN achieves statistically significant improvements …”.
§3: “depend critically on the variance of the weights”.

EVG/VG phrasing: “are immediately reflected in the magnitude …” (or “are immediately reflected”).

Meta-PCN reintroduces temporal non-locality and activation caching similar to backprop.
The method freezes feed-forward activations and runs a distinct inference phase. This requires storing c and h^{0}, analogous to backprop activation caching, and introduces explicit phase separation (forward → inference), which traditional predictive coding avoids. Thus, the biological advantage over backprop is reduced and should be acknowledged in the main text. Although high level justifications are given in the appendix, the authors do not discuss whether the same arguments could be applied to backprop.

### Questions
1) What loss do the weights actually optimise?
You introduce a new error-based meta-objective J to stabilise inference, but learning appears to proceed as if optimising a different loss. Could you explicitly state the loss function used for weight updates, and, if it is distinct from J, explain why optimising different objectives makes sense conceptually?
2) How is the “stabilised error” constructed in practice?
The text treats the stabilised error as a fixed top-down target, but its computation is not specified.  Is it the last error iterate, an EMA, or something else?
3) In the Appendix G ablation (“without meta-PC objective”), are feed-forward predictions fixed or iterated?
This ablation is central to understanding what Meta-PCN contributes. Does the ablation also freeze the feed-forward predictions? If not, could you include that variant to disentangle the contributions of fixed predictions vs. the meta-objective?

### Soundness
3

### Presentation
2

### Contribution
3

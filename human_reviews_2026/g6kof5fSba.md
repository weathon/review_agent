# Barriers for Learning in an Evolving World:  Mathematical Understanding of Loss of Plasticity

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
Deep learning models excel in stationary settings but suffer from loss of plasticity (LoP) in non-stationary environments. While prior literature characterizes LoP through symptoms like rank collapse of representations, it often lacks a mechanistic explanation for why gradient descent fails to recover from these states. This work presents a first-principles investigation grounded in dynamical systems theory, formally defining LoP not merely as a statistical degradation, but as an entrapment of gradient dynamics within invariant sub-manifolds of the parameter space. We identify two primary mechanisms that create these traps: frozen units from activation saturation and cloned-unit manifolds from representational redundancy. Crucially, our framework uncovers a fundamental tension: the very mechanisms that promote generalization in static settings, such as low-rank compression, actively steer the network into these LoP manifolds. We validate our theoretical analysis with numerical simulations and demonstrate how architectural interventions can destabilize these manifolds to restore plasticity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper provides a theoretical foundation for understanding and subsequently tackling the LoP phenomenon encountered in learning in dynamic settings. The paper pinpoints properties of solutions that inhibit the ability of the model to adapt and effectively learn in the future.

### Strengths
The paper is well written, has a clear structure and is easy to follow.

The paper identifies a geometrical property of the loss of plasticity manifolds that prevents gradient based methods from escaping them. Namely that the gradients are tangent to the manifold. Then they proceed to identify to categories of such manifolds that might occur while training, as the networks effective dimension decreases.

The discussion about escaping them through noise, dropout or some form of randomness is an interesting idea and apparently according to the experiments an effective one as well.

The paper also provides an explanation for the emergence of these manifolds through training, which is interesting.

### Weaknesses
The paper has no significant weaknesses in my opinion. As in the sense that it provides an explanation for the LoP in practical settings, from a theoretical standpoint. The explanation might be somewhat incomplete. Mostly I have some concerns which I put in the questions, as I would prefer to discuss them.

For the writing. In **line 109** there is a double the and in the appendix there is a double network in **line 646** and it should be in my opinion a represents in **line 647**.

### Questions
I have the following questions regarding your work:

1. Do you expect other types of manifolds to exist that cause the loss of plasticity? Apart from the frozen unit and the cloning manifold, especially given that the addition of the dropout layer is not effective consistently in non artificial experiments.
2. My second question revolves around neighboring points to the manifolds you describe. While it is true that if an algorithm reaches an exact cloning manifold then it will be impossible to escape, what happens when we are in the points around it?
3. Furthermore as a continuation to my previous question. What happens when you use SGD in these scenarios of inexact solutions. It seems hard for me to believe that if you reduce the batch size to 1 (even though this goes against general practice) there is not enough noise from the individual and new samples to help the model escape the neighborhood of the manifold. I am curious about this question since SGD with Gaussian noise manages to escape these manifolds.
4. Is the fact that we arrive at a neighborhood of the manifold and not the exact manifold a potential reason for the differentiation between the behavior of dropout layers in the artificial experiment and the non-artificial one?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies loss of plasticity (LoP), the gradual inability to learn as training progresses in continual learning settings. The authors define LoP in terms of manifolds and show theoretically that GD/SGD updates that occur in frozen-unit manifolds (from saturated units) and cloning manifolds (from duplicate/redundant units) will remain in their respective manifolds, preventing subsequent learning. They also show that if a network is composed of individual “modules” (layers or blocks) that satisfy some properties, then the entire network resides on a cloning manifold. Experimentally, they compare a base network and the same network with cloned units and show that they exhibit the same behavior as training progresses, indicating an inability to escape the cloning manifold. However, perturbations like dropout and noisy SGD can help to escape the manifold because they act as symmetry-breaking. Studying CL tasks, they show that as training progresses, the fraction of dead units and fraction of duplicated units tends to increase and effective rank and performance decrease. They also show how batch norm and layer norm can help to prevent the saturation of units (and subsequent LoP), preserving effective rank, and that training with a noisy learning rule like continual backpropagation (CBP) can help networks recover from LoP.

### Strengths
The paper formalizes and studies an interesting explanation for LoP. Indeed, low-rank connectivity is generally associated with better representational quality in static settings, but the paper shows how this breaks down in continual learning settings. It further highlights two ways this behavior can happen through frozen, saturated units and through cloned units. The paper is well-contextualized in the literature. The paper complements theory with several experiments in different architectures, showing the application of their results in explaining behavior in standard models and providing methods to prevent or escape LoP based on their analysis.

### Weaknesses
The specific contributions of the paper could be outlined more, such that specific results are highlighted. There are many simulations but it’s somewhat difficult to remember and follow what is or isn’t shown in them, especially as many figures are in the appendix and experiments are interspersed with theory. I found some parts (like Theorem 3.1) difficult to follow. Minor point, but there are many in-text typos.

In the paper, it’s mentioned that “initial growth is followed by a compression phase” and that this is “consistent with the information bottleneck principle.” However, the existence of a compression phase in neural network training is disputed and Saxe et al., 2018 showed that the central claims of Schwart-Ziv and Tishby, 2017 did not hold true in the general case. I’m thus skeptical of the claim of a compression phase in training.

Related, although the authors briefly mention the NTK and rank, the paper could link its results and study to feature learning more explicitly as it is very related to their analysis. For example, low-rank connectivity is typically related to the so-called rich learning regime, while higher-rank connectivity is related to lazy learning. Hence, the presence of frozen units or cloned manifolds may be a symptom of rich learning, and might be prevented by a network in the lazy learning regime. This could further highlight whether and how learning regime plays a role in LoP.

I acknowledge that the authors do show experiments of cases where there are duplicate or dead neurons and correlate this to worsened learning. However, I’m somewhat skeptical that GD/SGD always pushes the network to these manifolds, and whether they might be other underlying factors related to LoP? For example, in Figure B.5 and B.6, the behavior across different architectures and layers varies dramatically and the fraction of duplicate units is quite small. Nevertheless, I think the contribution of the paper still stands on its own but there is more to be learned in future work.

### Questions
1. In Figure 2.1, why does adding dropout worsen performance if it manages to reduce cloning and increase effective rank?

2. I understand that dropout provides symmetry breaking, but I’m wondering if you can discuss why this increases rank and decreases cloning? Dropout increases redundancy and is used as a form of regularization, so it’s surprising to me that it would have this effect. Is it because dropout is applied after cloning? Would LoP be worsened if it were applied at the onset of training?

3. In Figure B.3, batch and layer norm only appear effective in removing saturated units for the MLP and duplicated for the MLP and VIT. Can you discuss why this behavior is not consistent across all architectures?

Suggestion: you could perhaps use weight decay to prevent saturation of units/formation of dead units?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors begin their work by asking what structural properties of gradient flow lead to loss of plasticity, and in response, what sorts of algorithms or architectures can be designed to mitigate this. The authors proceed by defining a notion of a loss of plasticity manifold, which is a manifold in weight space where for each point in the manifold the gradient of the loss evaluated at that point is tangent to that manifold. Or in other words, a loss of plasticity manifold is a subset of possible network from which gradient flow cannot escape. The authors prove that for manifolds characterized by dead (saturated) neurons or those characterized by duplicated features, gradient flow cannot escape, resulting in the persistence of neuron death or redundant features under gradient descent or SGD. Empirically, the authors show that noisey GD/SGD and dropout can escape these loss of plasticity manifolds and that these manifolds can arise during continual learning.

### Strengths
They give a mathematical formalism of loss of plasticity due to neuron death or redundant units via the lens of stable manifolds. Specifically, the authors prove that gradient flow cannot escape a loss of plasticity manifold, and this modeling of plasticity is one way of unifying two well known-correlates: neuron death and rank collapse. This also introduces a new framework for modeling plasticity loss which future researchers may build upon.

### Weaknesses
While the introduction of stable manifolds is novel to this field, the results are not necessarily as novel. For instance, the implications of Theorem 2.1 are often stated in most papers in this space, notably those focused on neuron resets or spectral regularization. Moreover, the implications of this theorem are easily observed via a simple analysis of the gradient of the loss when neuron death or duplicate features are present, which many existing works explicitly state. Similarly, the empirical results appear to reproduce many of the phenomena already observed. At some level, the paper reads like it is restating existing results and observations under a more abstract formalism of stable manifolds. 
The paper does not analyze theoretically the stability of empirically observed LoP manifolds.
While the paper is generally well written, it could make use of better sign-posting or a clear summary of contributions in the introduction. The motivating question is quite vague and only half way through the main body does the reader begin to understand the scope of contributions.

### Questions
In the bit-flipping benchmark, you switch from SGD to CBP at half-point and see recovery. Can you show sensitivity to the switch point?

### Soundness
2

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
2

### Summary
This paper introduces a dynamical-systems framework for loss of plasticity (LoP) in gradient-based learning. The authors define LoP manifolds, subsets of parameter space invariant under gradient flow, corresponding to frozen or cloned units. The key result (Theorem 2.1) shows that certain linear constraints (row/column-sum equalities) define manifolds on which gradients remain tangent, preventing escape from such manifolds. Empirical experiments use cloned networks to demonstrate these manifolds and show that noise injection or dropout can help escape them. Further experiments and analyses connect LoP to representational compression and duplicated units, and test normalisation and perturbation strategies to prevent LoP.

### Strengths
The work addresses an important unsolved topic and is of high relevance to the ICLR community. The paper is well written and presented, with extensive experiments and analysis. 

While the general principle (group symmetries resulting in an invariant subspaces for GD) is known. The specific algebraic characterisation via row/column-sum constraints across block pairs, within a unified LoP-manifold framework, appears novel and is a clear, well-stated contribution.

### Weaknesses
The connection between theoretical results in section 2 and mitigation strategy experiments in section 4 is weak. The perturbations are an obvious way to recover from loss of plasticity, and the normalisation strategies seem largely heuristic. They do not directly test the assumptions of theorem 2.1 (tangency of the gradient to the manifold).

### Questions
- Could the authors clarify how the mitigation experiments in Section 4 directly relate to the LoP manifold framework of Theorem 2.1? Specifically, which theoretical assumptions or predictions of the theorem are being tested or illustrated by the normalization and noise-perturbation experiments?
- Line 164: Typo. “weigh(t) decay.” 
- Line 186: Why is Adam said to violate the symmetry conditions in Theorem 2.1 if Remark 2.3 claims the opposite?
- In Fig. 2.1, did you vary dropout rates or noise scales to examine stability/escape times from LoP manifolds?
- Line 271: The statement “training that enhances decorrelation also creates units that are nearly always inactive or saturated” should be empirically supported or cited.

### Soundness
3

### Presentation
3

### Contribution
3

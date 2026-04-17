# Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 4

## Abstract
When a deep ReLU network is initialized with small weights, gradient descent (GD) is at first dominated by the saddle at the origin in parameter space. We study the so-called escape directions along which GD leaves the origin, which play a similar role as the eigenvectors of the Hessian for strict saddles. We show that the optimal escape direction features a \textit{low-rank bias} in its deeper layers: the first singular value of the $\ell$-th layer weight matrix is at least $\ell^{\frac{1}{4}}$ larger than any other singular value. We also prove a number of related results about these escape directions. We suggest that deep ReLU networks exhibit saddle-to-saddle dynamics, with GD visiting a sequence of saddles with increasing bottleneck rank.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper provides an analytical study of the low-rank bias observed in the first escape direction in multi-layer ReLU networks. In fact, when the network is initialized with small weights, it needs to escape from the saddle point present at the origin, and the direction of this escape was previously shown to be biased towards low ranks. The paper formally proves that the escape direction is biased to have approximately rank 1 in deeper layers. A simple neural network trained on MNIST validates the theory and an analytical counterexample is provided to confirm that the rank of the optimal escape direction can be higher than one and thus an approximate notion of rank one is necessary.

### Strengths
- The background literature in Section 1 is very well presented. The reader gets a clear picture of this line of research, how previous works contributed to it, and how this paper fits into the narrative.
- The main results (Theorem 3.1. and Proposition 3.2.) are novel and provide an interesting theoretical advancement in the context of understanding saddle-to-saddle dynamics.

### Weaknesses
Although both theoretical and experimental examples were given to corroborate the claim, the latter are very limited in scope, with only one experiment performed on one simple architecture and dataset. I recognize that this is mainly a theoretical work, but a more robust numerical evaluation would have been welcome to validate the theory.

The results mainly concern the specific case of standard ReLU MLPs with no regularization (otherwise the loss is not homogeneous) and trained with gradient flow. These are standard assumptions in the literature but their strict nature limits the scope of the results.

I believe that the authors could improve the readability of the paper for a reader that is not an expert in this line of research:
- The approximation around the origin of Eq. (1) should be derived in the text, in the appendix, or a reference where such derivation is made should be referenced.
- The same holds for the homogeneity of the loss at L173. A good reference plus an intuition on where it comes from would be enough I think.
- The derivation of the equation at L180-181 (which should be numbered) in the appendix should be referenced in the main text. This should hold for all other derivations in the appendix.

### Minor weaknesses
- Please provide full forms of all abbreviations before using them, e.g. DNN (L025), NTK (L029), CNN (L076)

### Questions
- At L171 it is claimed that "since the ReLU is not differentiable, neither is the loss at the origin, so that we cannot use the traditional strategy of approximating L0 by a polynomial.". How is the approximation of Eq. (1) justified in light of this fact? 
- How much do the authors think the result extend (if they do) to the case of regularized losses or other activation functions?

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
3

### Summary
This paper gives a description of the saddle at the origin in deep ReLU networks, and the possible escape directions that GD could take as it escapes this first saddle. It were shown that the optimal escape directions feature a low-rank bias that gets stronger in deeper layers, and the optimal escape speed is non-decreasing in depth. A simple dataset is presented whose optimal escape direction is rank two for the first weight matrix. Finally, a few conjectures/hypotheses describing the complete Saddle-to-Saddle dynamics are discussed.

### Strengths
1. The theoretical analysis seems sound, although I did not check the details of proof.
2. Both optimal escape direction and speed are analysed. 
3. Figure 1 intuitively illustrates the saddle escape behavior and the low-rank bias feature on the MNIST dataset.

### Weaknesses
1. In experiments, the low-rank bias feature is only demonstrated on the MNIST dataset. Results on more datasets are preferred.
2. The motivation is not clear enough. Why should we study the saddle-to-saddle regime for very small initializations? In practice, the weights are initialized with, e.g. Kaiming initialization, which may not fall into the very small initialization regime.
3. What is the practical implications of low-rank bias feature for network training? Does it mean pathological loss landscape? How to use it to escape the saddle more effectively and efficiently? 
4. The complete Saddle-to-Saddle dynamics are only given as hypotheses without further proofs or empirical evidences. The main technical contribution is focused on the first saddle.

### Questions
see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Training of neural networks can happen in two main regimes: lazy and rich, the latter further split into an active and a mean-field regime.
This paper studies the early training dynamics of deep ReLU networks initialized with small weights, focusing on the active training regime, specifically the saddle-to-saddle phase where gradient flow (GF) escapes from the origin saddle in parameter space.
After introducing a *localized loss* approximating the true loss at small scale, norm and spherical dynamics are separated with the latter evolving faster and identifying escape directions, which are minima of the loss constrained to a sphere and therefore points where the norm dynamics can take the lead.
The theoretical results of the paper revolve around conditions imposed on the parameter at escape directions, in particular a low rank bias (seen through singular values of weights and activations matrices) and the collapse of inner activation layer to identity layer, with both these facts amplified with depth of the layer.

A validation numerical experiments is provided and a counter-example is given to disprove a conjecture that the escape direction is always rank one and, finally, the authors propose conjectures about saddle-to-saddle dynamics.

### Strengths
The submission appears technically sound, with a mathematically precise yet elegant formalism. The main claims are well supported by both theoretical proofs in the appendix and numerical validations (though not extensive).
The paper is clearly written and well structured, conveying a complete and coherent message.
While the authors do not provide code, the experimental setup is described in sufficient detaisl in Appendix B to allow reproduction, except for the unspecified method used to compute the top 10 singular values, which I think should be precised.
Overall, I have a positive impression of this work and consider it to be of high quality.

The analysis of where the rank-one layer are located, along with the proposal of a half-bottleneck structure, is particularly insightful as it helps reveal (during training) where the network’s interesting computations occur and therefore help with interpretability.

With the insights provided by the paper, there are clear ways forward, specifically, extending the analysis beyond the first saddle escape and exploring how learning might understood iteratively across subsequent saddles.
These potential research paths are further reinforced by the conjectures presented in the discussion section.

### Weaknesses
The paper presents a descriptive rather than prescriptive analysis and combined with the theoretical treatment, it limits the immediate practical consequences but the approach is appropriate given the paper’s objectives.
Also, the experimentation are limited in scope and the setup on MNIST seems a bit specific (see the related question below).
I think the paper could benefit from a stronger connection to practical implications, even long term, to help engage a casual reader: although the introduction situates the work well within related literature, the paper overall offers few explicit takeaways or long-term motivations for practitioners. (see last question)

Limitations are:
- The setup does not include biases
- The main result applies for optimal escape directions, however it is not guaranteed that the found escape direction will be optimal under gradient flow (though with the nuance that another result, proposition 3.4 which applies more broadly)
- The formalism is continuous while in practice discrete gradient descent is used
- Emphasis on deep networks:
    - the condition $l>c^2$ is proven to hold on all but finitely many initial layers; which is not a clear practical information
    - the bound is in $l^{\frac{1}{4}}$ which is not so strong (ratio $\approx 3$ for depth $l=100$), which is nuanced by experiments where the effect is clearly visible

## Minor and additional feedback
- line 453: "is greedily searched by first searching among BN-rank one functions" consider rephrasing to avoid repetition e.g. "is greedily search, first among BN-rank one functions"
- line 96: the word "step" is a bit ambiguous, something like "Dynamics with multiple saddle-to-saddle sequences" could be clearer.
- line 109: consider changing "activation" to "preactivation"
- notation overlap in sections 2.1 and 2.2: $s$ denotes both the reparamtrized time and the escape speed
- line 203 & 239: in my understanding this dynamics holds because $\|\theta(t)\|$ evolves more slowly than $\overline\theta(t)$ as it can be seen in equations lines 190. If this is correct, I would recommend precising it explicitly somewhere.
- theorem 3.1 is named A.8 is appendix
- line 323: appendix 3 but appendices are labelled with letters
- line 427: probably you meant "escape speed" instead of "escape direction"

### Questions
I have a few questions:
1. **On the parameter norm $r$ of line 228**: can $r$ be estimated to know the scale at which the approximation of GF by the localized loss ceases to hold ?
1. **On the MNIST experiment**: $1000$ hidden neurons are used in each of the 6 layers. In my experience,this level of expressivity exceeds what is needed to fit MNIST with an MLP.
Is there a particular reason for choosing this large width?
More generally, the training setup seems carefully tuned: many training steps ($1000$ epochs of relatively small batches), learning rate scheduling, input standardization.
How necessary are these design choices ?
Could you comment on the sensitivity of the results to the training details ? (Except the initialization scale since it is the focus of the paper)
1. **On figure 3**: what do you observe if you plot $\frac{s_i}{s_1}$ i.e. the ratio of each singular values by the highest one ?
This ratio plot might make it easier to visualize the dominance of the top singular value discussed in the paper.
1. **On conjecture 4**: the paper seems to be a step toward a full characterization of incremental learning in ReLU networks beyond the first saddle.
In your view, what are the most promising long-term practices this line of work could lead to, e.g. improved interpretability via hierarchical learning, overfitting diagnostics, redundancy-based architecture search, or new training strategies that separately address the spherical and explosion phases (as in Kunin et al., 2025)?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper analyzes how gradient flow (GF) escapes the saddle point at the origin when training deep ReLU networks initialized with small weights. The authors characterize the optimal escape direction of GF and show that it exhibits a low-rank structure in the deeper layers, where the leading singular value of layer $ \\ell $ is at least $ \\ell^{1/4} $ larger than the other singular values combined. They further suggest that deep ReLU networks follow saddle-to-saddle dynamics, where the weights transition between saddles of progressively higher rank during training.

### Strengths
The paper studies an interesting topic of escaping saddle points. The problem setting and assumptions are clearly presented, and the authors support their analysis with illustrative examples and well-designed figures. Overall, the paper is clearly written and effectively communicates its main ideas.

### Weaknesses
Although the authors depicted a detailed picture of the dynamics of GD in the vicinity of the saddle point at the origin, I am still not convinced that this is an actual phenomenon for the following reasons.

**Initialization magnitude:**
The current initialization schemes avoid initialization at the plateau of origin. In other words, using the standard initializations, we would not encounter such a setting.

**GF vs. GD:**
The analysis presented in the paper applies to GF. However, these results cannot be translated directly to GD. Specifically, GD with non-vanishing step size exhibits richer dynamics with dynamical stability constraints along its trajectory (e.g., Edge of Stability). Moreover, the optimal escape speed should be adjusted according to the stability constraints.

**Optimial vs. typical escape direction:**
The presented results on low rank are about the optimal escape direction. However, there is no proof that GD (or GF) will follow this escape trajectory. It might be the case that the typical escape routes of GD are non-optimal. In this regard, I don't understand the claim of GD bias towards low rank solutions, since there is no guarantee that GD will follow this optimal direction.

**Effect on the fully trained model:**
The main result deals with the state of the model in the first stage of training. I am not sure what effect a low-rank model early in the training has on the fully trained model. I am not convinced by the claim of saddle-to-saddle dynamics without any proof.

### Questions
1. Please explain the relation between the optimal escape direction from the saddle at the origin and the actual dynamics of GD. 
2. Can you give a way to convert the speed defined by GF to GD?
3. In line 189, it says that you consider the case of adaptive learning rate, where $ \\eta = \\| \\theta \\|^{2-L}  $. Wouldn't it be extremely large near the origin for deep networks? 
4. In the equation of the escape time (line 234), the dependency of $  t(r\_1) -t\_0$ on $ r $ for deep networks, given by $ r\^{2-L} $, is strange. It predicts that crossing smaller thresholds $ r $ takes a longer time (since $ L > 2 $), which does not make sense. Is there a problem with this result?

### Soundness
2

### Presentation
3

### Contribution
2

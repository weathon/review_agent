# Variational Semantic Decomposition of Compositional Representations

- Decision: Reject
- Scores: 4, 4, 6, 2, 2

## Abstract
Humans exploit the world’s compositional structure by combining familiar concepts to form new ones. A key goal in artificial intelligence is to model this capability by learning compositional representations that reflect the structure of the world, and its inverse operation, semantic decomposition, which involves recovering meaningful constituents from composite representations to enable generalization. The binding operation provides a way to compose vector representations, but existing decomposition methods, such as resonator networks, are limited by non-differentiable dynamics, have no guarantees for convergence, and assume quasi-orthogonality of constituent factors. We introduce the Variational Resonator Network (VRN), which reframes decomposition as Bayesian inference. The VRN is a fully differentiable model derived from a variational free energy objective, which allows it to be integrated into neural architectures. It generalizes decomposition beyond representations generated from quasi-orthogonal bipolar vectors and is guaranteed to converge. Experiments show that VRN is comparable to the resonator network on decomposition accuracy when factors are quasi-orthogonal and outperforms it when factors are correlated or real, while also integrating naturally into probabilistic models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work proposes extending Resonator Networks with a variational training objective. Said networks were proposed as a way to solve the problem of semantic decomposition: given a vector obtained by binding elements from different code books (values) into different roles (or variables), how can we recover the original unbound values? The original approach is hard to train given that it is not differentiable and also requires codebooks (sets of possible values) to be (quasi) orthogonal, which may not be realistic in practice. By casting the problem as a Bayesian Inference one, the authors can use a variational approximation to train the model effectively.

### Strengths
The authors explain their approach very well, and the paper is very easy to understand. They provide a useful extension to a previous method which makes it easier to use in practical settings. They provide simple experiments which make it easy to understand the benefits of their approach.

### Weaknesses
The main weakness is that the evaluation is fairly limited and some of the results cast some doubt on how scalable the approach is. 

* For example, one of the main potential benefits of the model is its ability to perform semantic decomposition over several factors, yet according to their own results the model experiences a sharp drop in performance once the number of possible codebooks increases beyond 2, which seems like the bare minimum. Now, they still show strong performance when the number of combinations is in the thousands, which may be more than enough for most practical considerations.
* A second issue is the limited evaluation beyond the toy examples. These are useful, in my opinion, to help the readers understand the benefits of the approach, but should be complemented with other datasets.
* There is no discussion on some relevant literature, which was not discussed in the original resonator network either. Specifically, Adaptive Resonance Theory by Grossberg.

### Questions
1. Shouldn't the correlated examples be harder than the non-correlated ones? The results show the opposite which is quite puzzling to me, but maybe I am not understanding something.
2. How does the model compare other approaches such as VQ-VAEs or the Neural Systematic Binder? These baseline comparisons are needed to strengthen the paper.
3. What is the relation between this approach and ART (mentioned above).
4. Can the authors provide results for more datasets? I would at least expect some of the classical ones used in disentangled representation learning: (colored) dSprites, Shapes3d, MPI3D etc. If the authors can provide examples for other modalities (e.g. language) that would also be useful.

I do really like the approach and would like to recommend acceptance, but right now the evaluation is weak and for better or worse AI is a very empirical field.

References:
1. Neural Systematic Binder: https://arxiv.org/abs/2211.01177

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a variational-based decomposition method VRN to recover meaningful constituent factors from a composite vector representation. VRN is a differentiable model derived from a variational free energy objective, which performs probabilistic inference over latent factors that compose a bound vector representation, and can generalize to correlated and real-valued cases. In the experiments, VRN achieves comparable accuracy to standard resonator networks in the quasi-orthogonal regime and outperforms resonator networks when factors are correlated or non-binary.

### Strengths
The paper reframes the resonator network’s iterative dynamics as variational inference and provide a probabilistic interpretation. The proposed formulation allows VRN to embed naturally within generative or inference-based frameworks, which can be a tool for compositional representation learning in probabilistic modeling.

VRN generalizes decomposition from quasi-orthogonal bipolar codebooks to correlated or real-valued vector spaces, broadening its applicability to realistic neural embeddings like object-centric representations.

### Weaknesses
The concern lies in the interpretability of the decomposed representations. The constructed codebook in the paper is quasi-orthogonal, but it remains unclear whether these representations actually correspond to distinct semantic factors. While the experiments on MNIST demonstrate the potential of extending VRN to probabilistic graphical models, the paper lacks qualitative analyses of the codebook. For example, do different codebooks correspond to distinct semantic factors such as background color or digit type?

The idea of factor decomposition via variational inference has been explored in earlier models, such as disentanglement-based VAE variants (e.g., beta-VAE [1]). The authors are encouraged to include a discussion clarifying how VRN differs from or improves upon these approaches.

Constructing representations based on quasi-orthogonal codes is not novel; for instance, NVSA [2] has already shown that it is possible to construct a series of orthogonal bases when the dimensionality D is large.

[1] beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework

[2] A Neuro-vector-symbolic Architecture for Solving Raven’s Progressive Matrices.

### Questions
In Figure 7, the reconstructed results are blurry. Does this indicate that the proposed method may impair the model’s image reconstruction quality?

Figure 1 presents examples from the CLEVR-style dataset, which is an good visual benchmark. Could VRN be effectively applied to such compositional visual reasoning datasets?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Variational Resonator Network (VRN), which formulates the vector decomposition problem as Bayesian inference. The resulting framework is fully differentiable and can be integrated into deep learning systems. Experimental results show that VRN matches the accuracy of the original resonator network under quasi-orthogonal settings, and outperforms it when vectors are correlated or real-valued.

### Strengths
- The paper provides a clear probabilistic reformulation of resonator dynamics, resulting in a fully differentiable inference framework with an explicit optimization objective.
- The method extends beyond quasi-orthogonal binary vectors and exhibits greater robustness under correlated and real-valued codebooks.
- The approach is compatible with modern machine learning pipelines, enabling VSA-style semantic decomposition to be integrated into gradient-based models.
- The formulation naturally fits within a probabilistic modeling framework and can be embedded into generative architectures.

### Weaknesses
- **Dependence on high-dimensional quasi-orthogonality for the derivation.** The key simplification (dropping the normalization / partition term) relies on "quasi-orthogonality under binding", which is an asymptotic argument; finite-dimensional error is not quantified.

- **Mean-field assumption ignores cross-factor dependencies.** The variational posterior is fully factorized across slots, potentially underfitting when factors are correlated; the paper does not analyze failure modes of this assumption.

- **Evaluation is largely on synthetic codebooks.** Main results use randomly sampled bipolar/real vectors rather than pretrained or task-induced embeddings; only a small VRAE demo is shown on colorized MNIST.

- **Sample size for reported metrics appears small.** Accuracy is averaged over 100 composite samples per setting; given that evaluations are synthetic (and search spaces can be large), 100 may be insufficient for stable estimates.

- **Scalability characterization is limited.** While accuracy is plotted versus search-space size n^K and K, the computational cost, convergence speed, and initialization sensitivity as n/K grow are not theoretically or systematically profiled.

- **VRAE integration is preliminary.** The generative-model integration is shown qualitatively with limited quantitative metrics or ablations, leaving practical benefits under-explored.

### Questions
Please refer to the *Weaknesses* section for my main questions regarding theoretical assumptions and evaluation settings.

### Soundness
3

### Presentation
2

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
This paper studies the “semantic decomposition” problem: given a single bound vector created by elementwise-multiplying several latent factor vectors (e.g., color, shape, position), recover which factor vectors were used. A resonator network solves this for high-dimensional quasi-orthogonal bipolar codes, but it’s non-differentiable, can fail to converge, and assumes near-orthogonality. 

The authors propose the Variational Resonator Network (VRN), which reframes decomposition as approximate Bayesian inference. VRN defines a variational free energy objective, infers a posterior over factor assignments, and optimizes it with gradient descent. This gives (i) a differentiable energy, (ii) an argument that the updates should converge because they perform loss minimization, and (iii) applicability beyond quasi-orthogonal binary codes, including correlated and real-valued codebooks. 

Empirically, the paper shows:
- VRN matches resonator networks on recovery accuracy in the standard bipolar / quasi-orthogonal regime.
- VRN outperforms resonator networks when codebooks are correlated or real-valued, where the resonator network is not originally designed to operate. 
- VRN can be embedded into a generative model (“Variational Resonator Autoencoder”, VRAE) to demonstrate integration in a neural pipeline. 

The paper positions VRN as a general, convergent, differentiable alternative to resonator networks, suitable for compositional inference inside deep networks.

### Strengths
### 1. Variational formulation with an explicit energy
VRN gives an explicit variational free energy objective whose gradients define the update dynamics. This answers two long-standing criticisms of resonator networks: (i) they are not directly differentiable and therefore awkward to slot into larger neural systems, and (ii) their dynamics can enter limit cycles and are not guaranteed to converge. The authors claim VRN resolves both by turning the problem into standard gradient-based inference. 

This “we turned a heuristic iterative retrieval rule into principled variational inference” is, in my view, the conceptual core contribution.

### 2. Beyond ~orthogonal codes
Classical resonator networks assume that each codebook is made of nearly orthogonal ±1 vectors. VRN is shown to work with (a) correlated codebooks and (b) real-valued codebooks, and it outperforms a tuned resonator baseline in those cases. This directly broadens applicability to realistic learned embeddings, where factors are correlated rather than perfectly orthogonal. 

### 3. Ablations
The Appendix looks at optimizer choice (Adam vs SGD), entropy regularization strength, and activation choices for resonator-style baselines on real codes. This shows some care in exploring design choices and stability.

### Weaknesses
### 1. Could be challenged as “variationalizing an existing iterative method,” not a fundamentally new capability.
Although the framing as Bayesian inference is elegant, one coould say: VRN is basically “take resonator-style iterative factor recovery, write down an ELBO-like objective for the posterior over factor assignments, relax to continuous logits, and optimize with gradients.” The paper needs to work harder to argue why this is conceptually more than “make resonator networks differentiable,” and to formalize the convergence advantage in a theorem rather than an assertion. 

### 2. Convergence is promised but not nailed down.
The authors repeatedly assert VRN is “guaranteed to converge,” in contrast to resonator networks which “are not guaranteed to converge” and “can enter limit cycles.” 
However, the paper (as provided) does not present:
- a formal statement like “gradient descent on L(ϕ) monotonically decreases L and therefore converges to a stationary point under step size α < …,” nor
- plots showing monotone decrease of that loss vs iteration alongside examples of resonator cycling.

Right now it’s closer to an informal claim than an established property.

### 3. Reliance on asymptotic quasi-orthogonality with limited validation.
A key simplification in the derivation is that in high dimensions, codewords within a codebook are quasi-orthogonal, so certain log-normalizer terms in the variational objective are approximately constant and can be dropped. The paper then still claims robustness in correlated regimes — where quasi-orthogonality explicitly fails. 
There is no quantitative study of:
- how large D must be for that approximation to hold,
- how badly it breaks when codebooks are intentionally correlated, or
- how much that approximation error affects final accuracy.

### 4. Limited experimental breadth / realism.
Most quantitative experiments involve synthetic setups that bind K factors by elementwise multiplication. The authors try to recover these factors, measuring recovery accuracy vs combinatorial search space size n^K. 
The only downstream/vision-ish demonstration, the Variational Resonator Autoencoder, is shown qualitatively (e.g. colorized MNIST-like reconstructions).

### 5. Baselines are narrow.
The authors mainly compare to resonator networks, occasionally with tweaks (tanh, normalization) in regimes the original resonator wasn’t intended for. How about the following as well:
- modern Hopfield-style associative memories,
- amortized neural decoders that directly predict factors from the bound vector,
- object-centric or slot-attention models for multi-factor decomposition.
The existing baseline is perhaps too easy, especially in correlated regimes where resonator networks are known to struggle.

### 6. No runtime / scaling story.
The paper stresses that naive decomposition is exponential in K (n^K), and positions VRN as an efficient inference routine. 
But there is no wall-clock or iteration-scaling analysis: How fast is VRN vs resonator vs brute force, as K and n grow? How does it scale to K=8 or K=10 factors? Without this, it’s hard to judge practicality.

### 7. The correlation narrative is suggested but under-supported.
The paper claims that using correlated codebooks actually helps VRN (e.g. smoother landscape), while resonator networks degrade. But this remains mostly anecdotal; there’s no explicit diagnostic (loss landscape visualization, gradient norms, iteration traces) backing that intuition.

### Questions
1. Convergence guarantee.
When you say VRN is “guaranteed to converge,” do you mean:
    - provably, via a Lyapunov / energy argument (“our variational free energy is strictly decreased by each update under mild conditions”),
or
    - empirically, “we never saw cycles in practice”?
Please clarify and, if it’s the former, include the actual convergence statement and its assumptions.

2. Constant-term approximation.
The derivation drops a log-normalizer term using a quasi-orthogonality / high-dimensionality argument. How large is that term (mean ± std) for realistic D and especially for correlated codebooks, where vectors are deliberately not quasi-orthogonal? Can you empirically bound the error introduced by discarding it, and show that VRN’s gradients are not dominated by that omission?

3. Practical downstream task.
Can you provide a quantitative benchmark (not just qualitative images) demonstrating that VRN-powered decomposition improves controllable generation, disentanglement, or systematic generalization in an actual model (e.g. CLEVR-style scenes, or compositional attribute editing)? Right now the downstream story (VRAE) is promising but anecdotal. 

4. Baselines beyond resonator networks.
Why is the resonator network essentially the only baseline? Could a strong amortized MLP decoder trained to map x → factor indices perform similarly or better with less per-instance optimization? How do modern Hopfield-style associative memories compare on your tasks?

5. Robustness and scaling.
How robust is VRN to noise or corruption in the bound vector x?
And how does accuracy / runtime scale as K increases beyond those in the main experiments? Even if accuracy falls, showing VRN degrades more gracefully than resonators would strengthen the story that it’s the more scalable inference mechanism.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a variational extension of resonator networks. The authors re-expressed the decomposition problem presented in resonator networks in a probabilistic way, deriving the ELBO for this specific setting. After going through the derivation, the authors proposed to optimise the obtained learning objective using a VAE and showcase its capacities through several synthetic settings.

### Strengths
- The paper is well-written and generally easy to follow.
- The idea is interesting, and while I am not an expert in resonator networks, I believe this paper would be of interest to this community
- As far as I know (but once again I am not an expert in resonator networks) the proposed extension is novel

### Weaknesses
I have several concerns regarding the soundness of this paper:
- The mathematical notation is inconsistent in several places, especially regarding the definition of $\mathbf{x}$ (see Q1), which is defined as three different things in the paper. This makes proof-checking very challenging, and I was unable to properly assess the mathematical correctness of the paper due to this. I am happy to give it another go if the authors provide a consistent notation during rebuttal. Some derivations would also benefit from being detailed in the appendix. (see Q1-2)
- The experimental section is limited to synthetic settings. I know this is a theoretical paper, and I don't expect a very extensive experiment, but it would be great to see at least one practical application, if only to illustrate potential applications of the proposed method to the reader. For now, it is quite hard to have an idea of the type of tasks on which VRN would work well, especially given the number of strong assumptions that were made in sections 2 and 3. (see Q3)

### Questions
Q1:
$\\mathbf{x}$ is defined as a composite vector l. 106, then, line 117, we have $\\mathbf{x} \\in \\mathbf{X}_j$, should it be $\\mathbf{x}_j$ as in l. 105? Furthermore, l.105 $\\mathbf{x}_j$ is a choice of vector from the $j^{th}$ codebook, but l. 115-120, $\\mathbf{x}_t$ seems to have another meaning, $t$ is never defined, but I guess this is probably the update step? If I followed correctly, the codebook index is now denoted by an upperscript. However, in Eq. 1, is $\\mathbf{x}$ a composite or a choice of vector from a given codebook? 

Same question for Eq. 3. l. 150, $\\mathbf{x}$ is now defined as a composite vector obtained from a binding. Same l. 243 and 263. However, where we previously had $\\mathbf{x} = \\mathbf{x}_1 \\odot \\cdots \\odot  \\mathbf{x}_K$.  we now have $\\mathbf{x} = \\mathbf{x}_1^{(i_1)} \\odot \\cdots \\odot  \\mathbf{x}_K^{(i_K)}$ where $\\mathbf{x}^{(i_j)}_j$ is the $i^{th}_j$ column of $\\mathbf{X}_j$. 
Did the authors mean $i^{th}$? 

l. 377 $\\mathbf{x}$ is now defined as $f_\\psi^{enc}(\\mathbf{y})$ which confused me as we prevously had l. 373 $\\delta_{Dirac}(\\mathbf{x} - f_\\psi^{enc}(\\mathbf{y}))$ and I assume we don't want to do $\\delta_{Dirac}(\\mathbf{x} - \\mathbf{x})$.

Given these inconsistencies, I strongly encourage the authors to update their notation so that each variable has a unique, consistent definition throughout the paper.

Q2: It would be great to have the detailed derivations for Eqs. 7, 17 and 18 in appendix to ease the reading.

Q3: Could the authors add a small real-life example application in the experimental section or at least in the appendix if space is an issue?

Q4: What are the training and inference times of the proposed model? Is it faster than resonator networks? Slower?

### Soundness
2

### Presentation
3

### Contribution
3

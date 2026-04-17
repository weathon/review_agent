# Convex Efficient Coding

- Decision: Accept (Poster)
- Scores: 8, 2, 6, 8

## Abstract
Why do neurons encode information the way they do? Normative answers to this question model neural activity as the solution to an optimisation problem; for example, the celebrated efficient coding hypothesis frames neural activity as the optimal encoding of information under efficiency constraints. Successful normative theories have varied dramatically in complexity, from simple linear models (Atick & Redlich, 1990), to complex deep neural networks (Lindsay, 2021). What complex models gain in flexibility, they lose in tractability and often understandability. Here, we split the difference by constructing a set of tractable but flexible normative representational theories. Instead of optimising the neural activities directly, following (Sengupta et al. 2018), we instead optimise the representational similarity, a matrix formed from the dot products of each pair of neural responses. Using this, we show that a large family of interesting optimisation problems are convex. This includes problems corresponding to linear and some non-linear neural networks, and problems from the literature not previously recognised as convex such as modified versions of semi-nonnegative matrix factorisation or nonnegative sparse coding. We put these findings to work in two ways. First, we extend previous results on modularity and mixed selectivity in neural activity; in so doing we provide the first necessary and sufficient identifiability result for a form of semi-nonnegative matrix factorisations. Second, we seek to understand the meaningfulness of single neural tuning curves as compared to neural representations. In particular we derive an identifiability result stating that, for an optimal representational similarity matrix, if neural tunings are `different enough' then they are uniquely linked to the optimal representational similarity, partially justifying the use of single neuron tuning analysis in neuroscience. In sum, we identify an interesting space of convex problems, and use that to derive neural coding results.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes an intriguing and original analysis of normative approaches to neural coding by casting optimization problems as problems of representational similarity. Using the representational similarity, which describes the population structure, authors show that some known optimization problems are convex, even when they appear not to be convex because of the constraint of non-negative firing rates or because of regularisation. They derive analytical conditions that a network has to obey to be convex. These conditions are crucially related to the scatter (spread) of neural activity vectors. An application of the theory is found in the implication that each neuron in an optimal network has a unique tuning curve, i.e., a tuning curve that does exceedingly overlap with the tuning curve of any other neuron.

### Strengths
The analysis seems technically correct and carefully conducted. The paper builds on previous work, proposing a refreshing and rather general approach to characterising efficient coding models. There is enough of high-level description and intuition to follow the ideas and the main results. I find that the paper makes useful and insightful contribution on normative approaches to neural coding.

### Weaknesses
1) It remains unclear why would be the solution where each neuron only encodes a single source optimal? This claim also seems rather unrealistic from the point of view of biological neural networks.

2) On some occasions, the definitions are not given much introduction and intuitive explanation. For example, the Definition 3.1 (Tight scattering) comes out of the blue and is not introduced with much justification, let alone intuition and is thus not well prepared. Definitions of the matrix D and F in Definition 3.1 are complex but not much effort is done to explain why they have to be as they are. 

3) On several occasions (line [228], for example) authors make a specific claim and promise to "review"or "explain" later. I find this is not particularly helpful for the reader. If a full explanation feels too lengthy, at least a short and intuitive explanation would fit better than saying it will be explained later. If it is about a commentary that will come later (as is the case of [228]), it would be better to cite the previous work that is of relevance.

4) It remains unclear why do networks have to recover the representation exactly and why they cannot be approximators of some exact solution. 

Minor:
line [152] there is a sum over t and then a normalization with 1/T - not clear.
line [77-79] "identifiability of related matrix factorization problems" reads unclear. Related to what?

### Questions
1) Typically, optimization problems are defined through minimization of a representational error (see for example Olshausen & Field, 1996), while authors here define the optimization problem in Eq. 2 only through the neural activity and decoding weights, requiring that the representation is linearly decodable from the neural activity. How does this ensure that the reconstruction is accurate and thus efficient?

2) Authors first introduce a general optimization problem in Eq. 2, and then a more specific problem (Nonnegative Affine Autoencoding) in Eq. 5. Why is introducing the nonnegative affine autoencoding necessary? Does Tight scattering in definition 3 not solve the problem in Eq.2?

3) In the simplified case where A is an identity matrix (no mixing), does the Tight Scattering condition hold? If so, does it resolve to an intuitive solution?

4) Why does the number of neurons has to be large than the number of datapoints?

5) Why is Frobenius norm used in the loss in Eq. 2? This could be justified when it is introduced, together with the intuition why it is used.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies a set of constrained optimization problems that linearly encodes the sources subject to resource constraints. Under certain conditions, the optimization problems were found to be convex. Some analytical results on the identifiability of nonnegative affine auto-encoders were presented.  Preliminary implications of the theoretical results to the neuroscience problems were mentioned.

### Strengths
****The paper carefully studies a set of constrained optimization problems that are akin to semi-negative matrix factorization. Some analytical results are derived and presented.

****Some discussions on the connections to neuroscience were presented.

### Weaknesses
****The work is quite incremental. The results, in particular the applications of the modeling framework, is rather preliminary. 


****Despite of the claim that the model framework analyzed is widely used in neuroscience, it is actually quite limited.  Maybe the framework and the results are indeed potentially useful to understanding some phenomena in neuroscience. If that is the case, it would be helpful to show these applications directly in the paper. Section 4 mentioned some potential applications but these are very pereininary. For example, it is mentioned that under certain conditions, one might be able to uniquely constrain the single neuron responses. But it is unclear whether theses conditions would be met in neural systems and whether the theoretical results are indeed capable of revealing new insights into biological systems. 


****The writing of the paper need substantial improvement. The paper has a lot of jargons, and   difficult to follow at the moment. The results need to be more carefully interpreted, and the implications and limitations need to be more explicitly described.  Overstates need to be toned down. 

****It was claimed in the paper that the paper serves to bridge the gap between deep networks and simple linear sparse coding models. But the framework is too simplistic when comparing to deep networks. It is in fact quite similar to the sparse coding models and matrix factorization.  The examples used in the papers are only toy examples and it remain unclear whether the results would generalize to more complex settings.

### Questions
Would it be possible to show some real applications of the framework to the biological data?

### Soundness
2

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
This paper reframes efficient-coding objectives as convex programs over representational similarity matrices. Optimizing over similarity structure, rather than activities, allows the authors to identify broad convex families of efficient-coding problems, including linear, affine, and some nonlinear networks.

Two key theoretical results follow:

1. Necessary and sufficient identifiability for semi-nonnegative factorization implemented by a nonnegative affine autoencoder with arbitrary mixing matrix A. This yields the first tight "scattering condition" guaranteeing recovery of sources up to scaling and permutation.
2. Uniqueness of single-neuron tuning: sufficient conditions under which nonnegativity plus convex optimality break rotational symmetry, making neuron tuning curves unique up to permutation.

The paper also provides a catalog of convex constraints (firing cost, readout norm, nonnegativity as complete positivity) and acknowledges limits such as $d_z > N$ and the need for perfect fits.

### Strengths
* Originality: Unifies a large class of efficient-coding models under a convex RS-matrix formulation. The "tight-scattering" condition provides an elegant, necessary-and-sufficient identifiability test not seen in prior work.
* Quality: Derivations are correct and geometrically motivated. Proofs connect convex geometry to neural interpretability.
* Clarity: Writing is clean, with an explicit "menu" of convex constraints and objectives.
* Significance: Offers theoretical justification for single-neuron tuning in convexly optimal codes, which is relevant for both computational neuroscience and interpretable ML

### Weaknesses
* The proposed semi-nonnegative factorization closely mirrors nonnegative sparse coding and semi-NMF [1]. In both cases, only the code (not the dictionary) is nonnegative, which already produces parts-based modularity. The paper should explicitly benchmark against these baselines under matched sparsity or energy constraints, clarifying whether convex RS-matrix optimization yields new behavior or simply a reformulation.
* The paper equates "representational similarity" with geometry preservation, which resembles RSA. However, high RSA correlation does not ensure good single-neuron fits. The authors should complement RSA with CKA [2] or other subspace-alignment measures and report per-unit, sparsity, or selectivity to substantiate claims about unique tuning.
* It remains unclear whether the key advance is the convex reformulation itself, the new identifiability theorem, or an explanatory account of neural modularity. Competing frameworks (NSC, sparse coding, convex ReLU networks) can explain similar results [3, 4].
* Demonstrations are illustrative but not quantitative; no finite-sample stress tests or noisy-data robustness.
* Completely positive programming is computationally intractable in general; implementable relaxations or surrogate constraints are not provided.

Refs:

[1] Hoyer PO (2002). Non-negative sparse coding. In: Proceedings of the 2002 12th IEEE Workshop on Neural Networks for Signal Processing; 2002; Martigny, Switzerland. Piscataway, NJ: IEEE; 2002. p. 557–565.

[2] Kornblith S, Norouzi M, Lee H, Hinton G (2019). Similarity of Neural Network Representations Revisited. ICML 2019

[3] Hoyer PO (2003). Modeling receptive fields with non-negative sparse coding. Neurocomputing 52–54:547–552.

[4] Beyeler M, Rounds EL, Carlson KD, Dutt N, Krichmar JL (2019). Neural correlates of sparse coding and dimensionality reduction. PLoS Comput Biol 15(6): e1006908

### Questions
* Under identical datasets and regularization, how does your convex RS-matrix method compare to nonnegative sparse coding and semi-NMF in reconstruction error, sparsity, and tuning modularity? Are there cases where NSC fails but your method succeeds?
* Does your notion of representational similarity reduce to RSA? If so, how do your claims about single-unit uniqueness hold under more robust geometry measures such as linear CKA?
* Can you provide an algorithmic test or relaxation for the tight-scattering and dual-cone conditions, and quantify how finite-sample noise affects identifiability?
* Could classic sparse-coding or convex ReLU formulations reproduce your observed modularity transitions? If not, what falsifiable predictions distinguish your theory?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This submission develops multiple theoretical contributions in the domain of representation learning, which are made possible by the observation that a class of optimization problems, which may not be convex in the representation itself, are in fact convex in the representation similarity matrix (RSM). The contributions are as follows:
- Demonstrating that non-negative similarity matching, in tandem with standard regularizers and a linear readout constraint, is a convex optimization problem in the RSM. In a similar spirit the paper discusses how this observation can be extended to describe a larger family of optimization problems.
- Leveraging this convexity result to derive necessary and sufficient conditions under which non-negative affine autoencoding admits an identifiable (up to permutation) solution. In particular this result holds for more general conditions than previous similar examples in the literature. 
- Leveraging the convexity result to derive sufficient conditions under which single-unit's tuning properties are necessarily unique (i.e. only one set of tuning curves can produce an optimal RSM).

### Strengths
- The heart of the paper, that optimizations problems that may not be convex in a representation $\bf{Z}$ can be convex in the RSM $\bf{Q} = \bf{Z}^T \bf{Z}$, is both quite interesting, well placed in the literature of neural circuits (via the anchor of similarity matching objectives), and to my knowledge novel.
- This key observation is used to substantial effect: it allows for the generalization of previous results in matrix factorization, and can be used to derive interpretable insights into neural tuning properties (at least in a theoretical setting). 
- The presentation of the work is for the most part very clear. The presentation of technical results is coherent, there is some effort to build intuition around these results (particularly in the example about when single-neuron tuning properties are identifiable), and there is a thoughtful effort to couch the contributions of this work in the broader literature (including acknowledging key limitations of the theory as it stands).

### Weaknesses
- Theorem 1 generalizes a previous results on problem 1 to the case of arbitrary (linear) source mixing (from orthogonal linear mixing). It is not obvious to me how significant this generalization is in practice. I.e. the value of this contribution could be made more clear if the author's outlined some real world cases where this contribution would be necessary to deliver theoretical predictions. 
- Regarding the limitations of $N_{neurons} > N_{samples}$ and the assumption of perfect fitting: it was not obvious to me whether the results for factorization with orthogonal source mixing relied on a similar set of assumptions, or if these were relatively stronger or weaker than the dependencies of previous theories?
- As noted in the limitations section of the discussion, though these problems can be demonstrated to be convex in $\bf{Q}$ the payoff of this observation in terms of computational approaches may be limited by the fact that the convex set of possible similarity matrices is not well suited for use in current numerical convex optimization solvers.
- I don't have much of an intuition for the types of circumstances in which the tight scattering condition would hold. Clearly one requirement is that the level set of the relevant quadratic form is elliptical, but it would be nice to develop a clearer way of seeing when this condition holds in terms of the structure of A and S (perhaps even with some practical examples).
- Nit: Typo in problem statement 1 (~ line 237) $\bf{b}_{\text{in}} \in \mathbb{R}^{d_x}$ --> $\bf{b}_{\text{in}} \in \mathbb{R}^{d_z}$

### Questions
- Can the author's expand on their thought in the discussion about the limitation of perfectly fitting the training data? What are the utilities and drawbacks of constraining your view of a system to the set of points that are perfectly fit?
- Can the author's answer the question from the weaknesses section about the severity of the assumptions used to define problem 1 relative to prior work on this problem? 
- The author's content their "results sound an optimistic note for classic neuroscience" but I am not sure this is necessarily the case. For example if the "true" generative model of signals encoded by neural systems do not obey the tight scattering condition (or similarly if the conditions of theorem 2 are not met) wouldn't the interpretation be reversed? It is in general not obvious to me which scenario we are in! Either way it is good to know the lay of the land from a theoretical perspective.

### Soundness
4

### Presentation
4

### Contribution
3

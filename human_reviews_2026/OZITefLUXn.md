# Representation Gap: Explaining the Unreasonable Effectiveness of Neural Networks from a Geometric Perspective

- Decision: Reject
- Scores: 2, 8, 6, 2, 2

## Abstract
Understanding generalization is a central issue in machine learning. Recent work has identified two key mechanisms to explain it: the strong memorization capabilities of neural networks, and the task-aligned invariants imposed by their architecture as well as training procedure. Remarkably, it is possible to characterize the neural network behavior for some classes of invariants widely used in practice. Leveraging this characterization, we introduce the representation gap, a metric that generalizes empirical risk and enables asymptotic analysis across three common settings: (i) unconditional generative modeling, where we obtain a precise asymptotic equivalent; (ii) supervised prediction; and (iii) ambiguous prediction tasks. A central outcome is that generalization is governed by a single parameter -- the intrinsic dimension of the task -- which captures task difficulty. As a corollary, we prove that popular strategies such as equivariant architectures improve performance by explicitly reducing this intrinsic dimension.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper propose to present an "explanation for the generalization properties" of neural networks, with theoretical contributions and an experimental validation.

They propose a metric called the "representation gap" which is the discrepency between the training data (e.g. manifold of (x,y) points) and the model predictions (e.g. (x, f(x)) predictions).

### Strengths
I'm failing to see the significance or novelty of this work, as described below.

### Weaknesses
Introduction:
--

The starting point of the discussion on generalization in NNs are (1) works on the implicit biases of gradient descent, and (2) "structural constraints [in diffusion models] to explain their impressive creativity". 

However, (1) was shown to have very little influence on the generalization properties of NNs [1,2]

And (2) diffusion models are an odd choice to talk about structural constraints, as this point was made just as clearly much earlier, e.g. with CNNs.

[1] Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent, Chiang et al. ICLR 2023

[2] Neural Redshift, Teney et al. CVPR 2024


The authors say:

> *the set of points (x, f(x)) that are reachable by a model f.*

This sounds a bit off: this is just the predictions of the model over the input space`?

--------------
Technical section
--

There are mentions of a relation with earlier work on manifold learning ("related work" section) and the manifold hypothesis (e.g. L770). I think there's a critical distinction between this work and the traditional manifold hypothesis that isn't stated. If I understand correctly, the latter is about the distribution of the data in the ambient input space $\mathcal{X}$, whereas this paper is focusing on the manifold defined by the data with its labels/predictions viewed as a manifold in $\mathcal{X} \times \mathcal{Y}$.

--------------

A central result is about equivariant architectures reducing the intrinsic dimensionality of the task. This seens trivial or perhaps tautological. The key assumption is that the enforced equivariance matches the symmetries present in the data. Therefore many possible points are equivalent under those symmetries, hence the intrinsic dimensionality of the data manifold is smaller, and the needs for training data (derived in the paper) are obviously improved.

--------------

The paper sets off trying to "explain generalization". The main results seem about the effect of enforcing hard invariances and equivariances in models. It's not clear how this helps with explaining generalization in a general sense, because most models don't have well-defined invariances, and the large body of work on enforcing such hard constraints in neural architectures has mostly been a failure (see e.g. Section 2 in [3]).

[3] Deep Learning is Not So Mysterious or Different, Wilson 2025

--------------

Name "representation gap": seems ill-chosen/misleading since this work is only considering the inputs and predictions of models, not its actual (internal/intermediate) representations.

--------------

Earlier work on dataset intrinsic dimensionality: not cited nor discussed. Here are some paper titles in order of relevance. The first in the list seems most important (e.g. related to the experiments with MNIST). The others may not be relevant enough to need to be cited, but I can't tell if this is why they're not cited, or if it's because the authors didn't know about this whole line of work.
- Intrinsic Dimensionality of Images and Its Impact on Learning
- Intrinsic Dimensionality of Image Representations
- Intrinsic dimension of data representations in deep neural networks
- Geometry of hidden representations of large transformer models
- Bridging Information-Theoretic and Geometric Compression in Language Models

--------------

L832: assumption of neural networks trained to interpolation with TV regularizer: does this match neural networks used in practice?

--------------
Empirical section
--

Empirical result in 5.1: "This is a remarkable result, since the models are trained with the generic L2 loss and have no knowledge of the structure of the data manifold."
On the contrary, the hard-coded equivariance is giving models full knowledge of this structure! This result is therefore fully expected. This experiments seems more like a "sanity check" that a demonstration of a "remarkable result".

--------------

L396 "Translation or rotation equivariance is added on top of the corresponding architecture*": how is this implemented?

--------------

Experiments "on real-world data" only use MNIST. Seems like a big claim for such a small experiment. It's fine not to have large-scale experiments and to focus on theoretical contributions, but the abstract/intro should not highlight "experiments on real-world data" as a contribution then.

--

There is a lot of math in the appendix that I didn't check given the much higher-level issues about the central claims of the paper mentioned above.

------------
Presentation
--

- The quality of the writing and presentation is subpar and makes the reading of the paper difficult.
- The bibliography is a complete mess. Most entries have no publication year, most have no venue, etc. Also please use correctly \citep and \citet.
- Missing sublot Fig.1(c) (mentioned in the caption)
- Figures 2 and 4 are identical???
- I don't understand the meaning of the first 4 words of the paper: "Implicit specification through data" (or what I'm guessing they mean makes no sense: this could apply to any learned model, it's not specific to NNs).
- L49 "*[quote fit without fear]*"??
- L79 ""**the** *standard technique to improve model performance*": what is **the** standard technique`?
- Appendix D "PREREQUISITE" -> "Preliminaries"
- L832 "minimal-norm hypothesis 6" -> "assumption 6"
- L478 "*could be leverage*" -> "*leveraged*"
- L485 "*distribution shift at test time, novelty introduction*": not sure what either of these mean; distribution shifts already refer typically to a shift between training and test time.
- L383 "*rotation-invariances*" -> no dash and "*rotational invariances*"

### Questions
I put a lot of questions above, but it is very unlikely that I will raise my score given its the many deficiencies at the time of its submission.

Overall this paper feels closer to a draft than a polished submission. I feel it borderline irrespectful to the community to submit half-finished papers for review.

In terms of contents, the main issues are that the main claims seem trivial (e.g. invariances) and/or well known (e.g. intrinsic dimensionality), and the empirical validation is straightforward yet claimed to show a "remarkable result", while being limited to simplistic toy data/MNIST. None of these issues would be a cause for rejection if these points were additions to another strong contribution. But here these points are all what the paper is about (unless I completely missed something major). For example, I don't have any problem with a weak experimental section if there was a strong conceptual contribution. Lastly, there is a disconnect between the technical contents of the paper and the stated goal at the beginning of the paper of "explaining generalization" (the paper basically shows that hard invariances/equivariances allow learning with less data, but this doesn't exist/usually doesn't work well in real architectures).

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper studies the geometry of representations in deep neural networks and how that geometry connects to their generalization. The authors propose an eleganbt metric, the representation gap, and show how it is affected only by the intrinsic dimension of the task.

### Strengths
- The paper, despite its very technical nature, is generally very well written. The theoretical statements generally have intuitive/high-level explanations or informal versions
- The paper's subject is very relevant
- The experiments clearly back up the claims, and are generally illustrative
- I find the phrasing of the implication of Thm. 1 amazing and very insightful ("architecture = virtual augmentation")

### Weaknesses
I mostly have minor concerns.

- It is unclear to me whether (and how) this paper's contribution is related to the linear representation line of works by Victor Veitch et al. As far as I understand, they also take a geometric perspective (by defining a causal inner product). To better ascertain the novelty of the contribution, it'd be very useful to compare against to those works.
- *Abstract:* the two factors of generalization (and the related works) omit the large works on identifiable representation learning (particularly, nonlinear ICA), some of which results even formulate theoretical results for OOD (compositional) generalization.
- Eq (1): please explain in words what it means
- Thm. 3.: I think it'd be beneficial for the reader to better define what an "ambiguous task" means
- 5.1.: could you please add a bit more details how you could draw the two conclusions?
- 5.2.: please also elaborate the two observations as well
- Fig. 5: the label size is too small + please define what a quantizer means

### Minor points
- The citations do not show the year
- L049: a comment was presumable not removed after the citation
- L269: maybe it would be less confusing to denote constants $c$ and not $J$
- App. C.: since you list your assumptions, I'd rename this to "Assumptions"

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper delves into the crucial problem of generalization in neural networks. The attempt to understand how design choices, like model architecture or training procedures, affect model behavior beyond the training data is genuinely intriguing. The authors propose a novel concept called the 'representation gap' ($\mathcal{R}_n$), moving away from traditional statistical learning theory frameworks like VC-dimension or Rademacher complexity. Instead, they offer a geometric perspective, measuring the discrepancy between the data manifold ($\Omega$) and the representation learned by the model ($\Omega_f$).

The core argument is that this representation gap asymptotically converges as $n^{-2/d}$, where $n$ is the dataset size and $d$ is the 'intrinsic dimension' of the task. This dimension $d$ is supposedly determined by the geometry of the data manifold and the symmetries (like equivariance) inherent in the model. I find the claim that techniques like equivariance provably improve generalization by reducing this intrinsic dimension particularly compelling. The analysis spans three scenarios: unconditional generative modeling (especially diffusion models), supervised prediction (assuming minimal-norm interpolation), and ambiguous prediction tasks.

Frankly speaking, the geometric approach itself is a refreshing and potentially insightful direction. Trying to explain the asymptotic nature of generalization performance with a single parameter—the intrinsic dimension—is quite ambitious. Providing a theoretical rationale for why equivariance works is also commendable.

However, I have some reservations. The theoretical development seems quite heavily reliant on specific assumptions about the models (mainly certain diffusion models or minimal-norm interpolators) and the data manifold. This makes me wonder about the broader applicability of these results to the messy reality of deep learning models and data. The concept of 'intrinsic dimension' $d$ is appealing, but the paper seems a bit lacking in discussing how one might estimate or know this value for complex, real-world data. While the experiments support the theory, they are mostly confined to low-dimensional synthetic data or MNIST, which might not be enough to demonstrate validity for more complex, high-dimensional modern problems.

### Strengths
1.  **Novel Perspective:** Offers a fresh geometric viewpoint on generalization, potentially moving beyond the limitations of classical statistical learning theory. The representation gap and intrinsic dimension concepts are thought-provoking.
2.  **Attempt at Unified Explanation:** The effort to connect implicit bias, structural invariants (like equivariance), and generalization through the concept of 'intrinsic dimension' is valuable.
3.  **Derivation of Asymptotic Equivalence:** Deriving the precise asymptotic convergence form ($n^{-2/d}$) for the representation gap, rather than just upper bounds, is a theoretically strong result.
4.  **Explanation of Equivariance:** Provides a clear and compelling theoretical explanation for the effectiveness of equivariant architectures – they reduce the task's intrinsic dimension $d$.

### Weaknesses
1.  **Strong Assumptions:** The theory seems to lean heavily on specific model classes (e.g., diffusion models satisfying certain conditions, minimal-norm interpolators) and geometric assumptions about the data lying on a Riemannian manifold. It's unclear how well these assumptions hold in typical deep learning scenarios. The claim that the learned representation $\Omega_f$ is either the dataset $\mathbb{D}$ itself ($\Omega_f = \mathbb{D}$) or its group-expanded version ($\Omega_f = G(\mathbb{D})$) might only apply under ideal convergence of diffusion models or strong invariance conditions.
2.  **Practicality of 'Intrinsic Dimension' $d$:** While theoretically interesting, the paper doesn't provide much guidance on how to practically estimate or determine the intrinsic dimension $d$ (either $d_\Omega$ or $d_{\Omega/G}$) for real high-dimensional data manifolds or quotient spaces. The post-hoc estimation of $d \approx 14$ for MNIST is shown, but its general applicability is questionable. This raises doubts about the practical utility of the theory's predictive power.
3.  **Limited Scope of Analyzed Models:** The focus is primarily on diffusion models and minimal-norm interpolators. How might this analysis extend to other important architectures, like standard CNNs or general Transformers?
4.  **Sensitivity of 'Representation Gap' $\mathcal{R}_n$ Definition:** The representation gap is defined using an integral of squared $\ell_2$ distances. Does the $n^{-2/d}$ scaling law hold if other loss functions or distance metrics are used? Could the result be overly specific to the $\ell_2^2$ metric?
5.  **Analysis of Ambiguous Prediction Tasks Seems Weaker:** The analysis for ambiguous prediction tasks (Section 4.2, Theorem 3) only provides an upper bound $O(n^{-2/d_\Omega})$ and relies on a somewhat vague "smooth covering model" assumption, making this part feel less solid compared to other sections.
6.  **Limited Experimental Validation:** The synthetic data experiments, while illustrative, are quite simple and low-dimensional. The MNIST experiment is interesting ($d \approx 14$), but more convincing real-world experiments on complex, modern deep learning problems are needed to truly validate this geometric perspective.

### Questions
1.  How valid do you believe the core assumptions of the theory (e.g., $\Omega_f = G(\mathbb{D})$ for diffusion models, minimal-norm interpolation, Riemannian manifold structure of data) are for complex, practical deep learning models (like ViTs, LLMs) and real-world data distributions? What are the limitations on the theory's scope if these assumptions don't hold perfectly?
2.  Are there practical methods to estimate or approximate the 'intrinsic dimension' $d$ ($d_\Omega$ or $d_{\Omega/G}$) for real-world tasks? If this dimension is unknown, how can the predictive power of the theory be utilized?
3.  Do you expect a similar $n^{-2/d}$ scaling law to hold if the representation gap were defined using a different distance metric (e.g., Wasserstein distance, KL divergence) instead of the squared $\ell_2$ distance? Or is this result specific to the $\ell_2^2$ metric?
4.  Could you elaborate on the "smooth covering model" assumption (Assumption 7) used in the analysis of ambiguous prediction tasks (Theorem 3)? Specifically, how do diffusion models satisfy this, and how might the results change if this assumption is violated?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates the approximation error for equivariant vs non equivariant neural networks, in the setting of diffusion models, showing that due to equivariant networks being constant on group action orbits, this lowers the intrinsic dimension, helping for better sampling bounds.

### Strengths
The main positive side is that the underlying principle shown is clearly presented and correct as far as I followed.

### Weaknesses
1) While the paper is aimed at equivariant neural networks, this is not stated explicitly enough, making it necessary to read it at least twice in order to follow the message.

2) I see a lot of the main principles of the paper are not new, and the paper just sets known results in the setting of diffusion models (which was not the focus of previous work). This may be due to authors being unaware of some important literature that contained very similar results (I say so because this literature is not cited). I will give only the main examples that come to mind.

a) For the role of "virtual augmentation" (theorem 1 of the paper under review), this principle was already shown in this work with similar ideas, with main difference being that the setup is not with diffusion models: 
Elesedy. Provably strict generalisation benefit for invariance in kernel methods. Neurips 2021.
See also
Elesedy, Zaidi. Provably strict generalisation benefit for equivariant models, ICML 2021

b) For the dimension dependence (theorem 2), similar principles appear in 
Chen, Dobriban, Lee. A group-theoretic framework for data augmentation. The Journal of Machine Learning Research, 21(1):9885–9955, 2020
See also
Tahmasebi, Jegelka. The exact sample complexity gain from invariances for kernel regression. Neurips 2023

c) For theorem 3, again results like this were present in the works in (b) and see also
Petrache, M., & Trivedi, S. (2023). Approximation-generalization trade-offs under (approximate) group equivariance. Neurips 2023

These are just a few works that contain similar results, and none of which are cited in this work, so I invite the authors to do a thorough literature search starting from these results, and to include a comparison with them.

### Questions
My main question is if the authors could present a detailed comparison to the results described in weakness 2, points a-b-c, highlighting if there are any new principles present, or if the main novelty compared to these is the application to diffusion models.

### Soundness
2

### Presentation
2

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
This paper presents a novel geometric framework for analyzing generalization in neural networks, centered around a new metric termed the "representation gap." This metric quantifies the discrepancy (as measured by an integral of squared distances) between the true data manifold, $\Omega$, and the manifold effectively learned by the model, $\Omega_f$. The central theoretical contribution is an asymptotic scaling law for this gap: $\mathcal R_n \sim n^{-2/d}$, where $n$ is the dataset size and $d$ is the "intrinsic dimension of the task." 

The authors provide a  partial theoretical explanation for the effectiveness of equivariant architectures, proving that they reduce this intrinsic dimension to that of the quotient space ($d = d_{\Omega/G}$), thereby accelerating the convergence of the representation gap in the asymptotic limit when $n \rightarrow \infty$.

The claims are supported by empirical validation on synthetic datasets and MNIST.

### Strengths
- The paper is overall well-written.
- The "representation gap" is an intuitive metric that generalizes empirical risk to the setting of manifold learning.
- The core idea of formalizing the sample efficiency gains of equivariance through a data-scaling law with a changing exponent ($d_{\Omega}$ vs. $d_{\Omega/G}$) is intriguing.

### Weaknesses
- In my view, the authors are overstating their results implications for generalization. For example, see l. 78: “We show that generalization is governed by the intrinsic dimension of the task, a single parameter which captures the difficulty level of the task, and may be directly linked to the data manifold geometry and the model invariants.”. Intuitively, the generalization gap might indeed be related to generalization, but this work provides no generalization bounds that rigorously connects those two. Obviously, if the representation gap goes to zero, the generalization error must also go to zero. It is however left completely unanswered what the decay rate of the generalization error is as a function of the rate of the representation error $\mathcal R_n$.

- The results w.r.t. representation gaps are fundamentally asymptotic, and it is not clear how close these asymptotic rates are to the ones in (realistic) finite sample regimes. In deep learning, it is well-known that typical models can memorize large training sets, and given enough samples, eventually perfectly learn any sufficiently regular distribution (due to uniform-convergence-type results). The actually important question is why these complex models manage to _efficiently learn_ natural distributions— a question which fundamentally requires finite-sample type results.

- The strength of distributional assumptions: see l. 61: “our analysis do not rely on any assumption about the data distribution $\rho$, 
but only on the geometry of the manifold $\Omega$ on which this distribution is supported.” l. 92: “we do not make any assumption about the data distribution $\rho$ and adopt a geometric perspective instead.” In my view, knowing the support of a distribution is in fact a very strong assumption. I suspect that the authors intend to say that their results are governed by a single, informative quantity (the intrinsic dimension) that is very informative about the ground-truth distribution. However, saying that no distributional assumption is made seems a bit misleading.

### Questions
- See weaknesses.

- Do you think your results could be extended to the more realistic scenario of approximate symmetries in the data, and/or settings where the model's invariances do not exactly match the symmetries in the data?

Typos:
l. 49: "todo" marker

### Soundness
2

### Presentation
2

### Contribution
2

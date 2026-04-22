# FINDER: Feature Inference on Noisy Datasets using Eigenspace Residuals

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 2

## Abstract
“Noisy'' datasets (regimes with low signal to noise ratios, small sample sizes, faulty data collection, etc) remain a key research frontier for classification methods with both theoretical and practical implications. We introduce FINDER, a rigorous framework for analyzing generic classification problems, with tailored algorithms for noisy datasets. FINDER incorporates fundamental stochastic analysis ideas into the feature learning and inference stages to optimally account for the randomness inherent to all empirical datasets. We construct “stochastic features'' by first viewing empirical datasets as realizations from an underlying random field (without assumptions on its exact distribution) and then mapping them to appropriate Hilbert spaces. The Karhunen-Loève (KL) transform breaks these stochastic features into computable irreducible components, which allow classification over noisy datasets via an eigen-decomposition: data from different classes resides in distinct regions, identified by analyzing the spectrum of the associated operators. We validate FINDER on several challenging, data-deficient scientific domains, producing state of the art breakthroughs in: (i) Alzheimer's Disease stage classification, (ii) Remote sensing detection of deforestation. We end with a discussion on when FINDER is expected to outperform existing methods, its failure modes, and other limitations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes FINDER (Feature Inference on Noisy Datasets using Eigenspace Residuals), a framework for binary classification in settings with high noise or limited data. The approach models datasets as random elements in a Hilbert space and applies a generalized Karhunen-Loève expansion to represent stochastic features in terms of orthogonal deterministic and random components. Classification is then performed in residual eigenspaces derived from class-specific covariance operators. In practice, the method estimates class-specific covariance matrices, computes eigen-decompositions, and projects data onto various residual subspaces before feeding into standard classifiers (e.g., SVMs, HMMs). 

Three variants for constructing residual subspaces are described: (i) direct residual subspaces, (ii) a multilevel subspace (MLS) method adapting the algorithm of Tausch and White (2003) to construct nested orthogonal bases, and (iii) two anomalous-class-adapted methods (ACA-S and ACA-L) that minimize or maximize the projected variance of one class in the residual space. The framework is validated on noisy biomedical and remote-sensing datasets, including Alzheimer’s disease proteomics data and deforestation detection from optical and radar imagery, showing improved performance and runtime compared to standard classifiers. The paper also discusses theoretical bounds based on the covariance operators, and computational complexity (though not in the main text).

### Strengths
The paper presents a mathematical framework that is elegant and general. It allows for a unified treatment of residual subspace methods that applies to both finite- and infinite-dimensional settings, and allows for quite general distribution-agnostic bounds. 

The algorithms presented in the paper appear simple to implement and achieve high performance on real datasets (though the lack of references made it difficult to verify state of the art performance).

### Weaknesses
**Related work**. The paper includes some background information in the introduction that is relevant to the main exposition, but has almost no discussion of recent related work. The experimental results are difficult to evaluate given the lack of references to related work and/or public leaderboards. 

**Clarity**. Writing is confusing at times. Basic ideas are often obscured by abstract formalism and idiosyncratic terminology. As far as I understand, the paper essentially constructs a general mathematical formalism for discussing simple covariance-based preprocessing methods. Clarity would be significantly improved if this were made clear from the very beginning. Numerous typos further hinder understanding (see detailed comments below). 

**Novelty**. The paper seems to deal exclusively with published algorithms or minor variants thereof based on per-class covariance estimation. The abstract mathematical formalism, while elegant, obscures this fact. 

**Applicability of formalism**. The formalism seamlessly handles infinite dimensional settings - which is one of its main strengths - but this doesn’t seem to be necessary for any of the considered applications. The Hilbert-space generalization is not shown to yield any new capabilities, numerical advantages, or insights beyond standard matrix algebra. Likewise, proofs of measurability and operator properties are irrelevant to the experiments.

### Questions
**Major comments/questions**

1. The paper would benefit from an expanded discussion of related work. Some specific questions:
    - Are there other published theoretical frameworks for approaching noisy/incomplete datasets of the type that you consider? 
    - Are there standard datasets of this type? What is the current state of the art performance on these? 
2. Why did you consider these particular datasets? In the introduction it is claimed that FINDER is particularly effective in “data-deficient or otherwise noisy settings”. Later, when discussing the two datasets, little is said to connect these particular problems to claimed theoretical strengths of FINDER - though section 3.2 does mention, “...clouds can obstruct or completely block the ground, significantly reducing the amount of usable data.”
3. A significant amount of space is devoted to development of abstract aspects of the mathematical formalism that don’t seem to be relevant to applications, eg. measureability, integrability, etc. Consider moving some of the more abstract material into appendices, and using the freed space to explain the mechanistic core of the algorithms using intuition from finite-dimensional linear algebra/geometry. 
4. Relatedly, I don’t feel that I’ve understood the utility of the “stochastic feature” concept. As far as I understand, the algorithms simply involve estimating (top) feature covariance eigenspaces, putting the user right back in the familiar setting of “datasets as samples from a distribution”. 
5. One of the main algorithms proposed by the paper (MLS), is not explained in the main text, and even the description in the supplement is quite high-level. 
6. Given that all applications occur in finite-dimensional feature spaces - and therefore do not invoke the generality of the mathematical formalism - it’s my understanding that the “adaptations” of Direct Residual Subspaces (DRS) and Multi-level Subspaces (MLS) would actually be identical to the previously published versions. If that is the case, it should be made clear. 


**Minor**

1. Line 031-032, should be $F\gg N$ or “F is much greater than the sample size $N$. 
2. I don’t understand the phrase “in-built separability" used at several points in the text (eg. line 062). 
3. Line 131-132, an Hilbert-Schmidt -> a Hilbert-Schmidt
4. Line 142, subspaces -> subspace
5. Line 188, “This is the one of the…”
6. Line 215, “For Balanced…” confused me at first. Consider changing to “For the ‘Balanced’ condition…”
7. Line 355-356, “robust with respect to the choice to pre-balance or not pre-balanced the data…”
8. Line 399, data missing comma between day and year. 
9. Line 429, “For 71 days of optical training day”
10. Line 461-462, “examlpe”

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors introduce a rigorous theoretical framework for binary classification, building upon standard functional data analysis techniques. The method maps data into a suitable Hilbert space and defines a residual eigenspace that captures the directions where the two classes most differ. The approach is tailored to improve robustness in noisy datasets and is validated on two scientifically relevant tasks, showing improved performance over previous methods.

### Strengths
The paper addresses the relevant problem of classification under noise, combining theoretical rigor with applications to two scientifically meaningful case studies. The framework is mathematically sound and versatile, grounded in functional data analysis. The empirical results are convincing and strengthen the authors’ claims. The authors also acknowledge several current limitations of their method and provide a transparent discussion of them.

### Weaknesses
1. The presentation is mathematically heavy and not sufficiently intuitive, which reduces accessibility for a broader ML audience that could benefit from this approach. In particular, the functional analysis formalism could be complemented by a schematic pipeline in a simpler setting (e.g. $\mathcal{H} = \mathbb{R}^D$), clarifying how the theoretical constructs translate to the experiments.

2. The novelty of the proposed framework should be more explicitly highlighted. For instance, the connection between the proposed residual eigenspace approach and existing methods such as FDA techniques remains unclear. A concise explanation in the main text of how Theorem 2.1 generalizes the KLE and a discussion of what fundamentally differs in practice from classical PCA-based methods, would improve clarity. 

3. Some of the discussed limitations, such as the heuristic choice of truncation parameters, limit the practical applicability and impact of the method.

### Questions
See 2. in Weaknesses. Apart from this, I have no additional major questions.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces FINDER, a classification framework for small-sample, noisy, high-dimensional scientific data. It models observations as random fields in a Hilbert space, applies a (generalized) Karhunen–Loève expansion to extract class-relevant eigenspaces, and then builds residual eigenspaces that are informative for the other class. Standard classifiers (SVM, HMM) trained on these projected features outperform common baselines on Alzheimer’s proteomics and remote-sensing deforestation tasks, especially under data imbalance. The main claim is that this stochastic-feature combined with the residual-eigenspace pipeline is distribution-agnostic and computationally lighter than more elaborate ML methods.

### Strengths
* The paper is well-organized and clearly written. The paper has clean definitions of the embedding and sufficient experimental data.
* The experiments were carried out on real datasets.

### Weaknesses
Major comments:
* It is unclear to me the significance of the contribution by defining such a pipeline. It seems usual PCA but generalized to Hilbert space (which is also not novel in kernel clustering). To make this compelling, the authors need to articulate what FINDER achieves beyond (a) standard FPCA/FDA pipelines that learn class-specific subspaces and (b) PCA/kPCA followed by a linear classifier — e.g., is the ACA residual step provably better for imbalanced or low-SNR settings, or does it offer a computational advantage that existing RKHS methods do not? As written, the mathematical setup looks heavier than the actual algorithmic novelty.
* It is also unclear to me how the underlying feature vectors (i.e. the implicit kernel/geometry in the Hilbert space) are to be chosen. In the current presentation, this choice is taken as given, but in practice it is at least as hard as the classification/clustering problem itself: the performance of FINDER is entirely determined by how well this geometry aligns with class structure. Moreover, because the method is advertised as “distribution-agnostic,” one would expect the feature-space selection to be unsupervised or weakly supervised (e.g. data-driven kernel selection, multiple-kernel aggregation, or intrinsic manifold estimation), but no such mechanism is described. As a result, the strongest assumption of the paper — “we already have the right Hilbert embedding” — is exactly the part that needs justification.
* Computational lightness of ACA may be overstated. The paper claims that ACA is computationally light, but the core step — minimizing the residual while enforcing a rank (i.e. low-dimensional) constraint — is, in general, a combinatorial problem once you move beyond the trivial “take the top/bottom eigenvectors” case. No convex relaxation or spectral surrogate is described. In the experiments the residual dimension is fixed to 5, which makes the search cheap, but this is effectively hard-coding a small model. It is not clear what happens when the informative residual subspace is larger (say 15–30 dimensions, as is common in HDLSS FDA): does ACA remain “light,” or does it become an intractable subset-selection over eigen-directions? As it stands, the claim of computational efficiency seems to rely on a very favorable experimental setting rather than on an algorithm whose complexity is controlled in general.
* Except for my concerns about novelty, there are additional limitations including restriction to binary case: the use of loose Markov bounds---a Hanson-Wright type tail bounds provide sharper characterization immediately, etc.

Minor comments:
* More references and discussions to standard PCA-based classification in functional data are required. I think at least a dedicated paragraph should exist.
* In Theorem 2.1. "there exists $R \in N \cup \aleph_0$ ". It should be " $\mathbb{N} \cup\left\\{\aleph_0\right\\}$ " or " $R \in \mathbb{N} \cup\\{\infty\\}$ ". It is also unnecessary to use $\aleph_0$ over $\infty$ here as the summation $\sum_{r=1}^R$ encodes this information.
* In Line 461, "examlpe" should be "example".

### Questions
* My main questions are in the weaknesses section.
* Why controlling residuals using Markov instead of sub-Gaussian tail bounds (e.g. Gaussian convex inequality or Hanson-Wright)?
* How did the authors choose the hyperparameters? How were the underlying kernels (eigenfunctions) chosen?

### Soundness
2

### Presentation
2

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
This paper introduces FINDER, a theoretical and algorithmic framework for classification under high-noise, low-sample-size conditions. FINDER models datasets as realizations from underlying random fields and constructs stochastic features mapped into Hilbert spaces via generalized Karhunen–Loève expansions (KLE). Theoretical results (e.g., Lemma 2.1–2.3) provide stochastic interpretations of feature separability, leading to several algorithmic variants (MLS, ACA-S/L). The proposed framework is tested on two domains: (1) Alzheimer’s disease diagnosis from blood plasma proteomics, and (2) remote sensing for deforestation detection, reporting improvements over standard ML baselines and domain-specific benchmarks.

While the authors combine functional data analysis concepts with machine-learning motivation, the resulting work falls short of both theoretical originality and empirical rigor. The mathematical material is largely recycled from standard stochastic analysis, the empirical evaluation is insufficiently convincing, and the exposition is difficult to follow. The paper would benefit from a major refocus—either toward a concrete algorithmic contribution with clear empirical value, or toward a purely theoretical paper with genuinely new results in stochastic feature representation.

### Strengths
1. The focus on learning in noisy and data-deficient regimes is important and relevant.
2. The paper is written with care regarding functional analytic details, correctly invoking concepts like Bochner integrals, covariance operators, and Hilbert-Schmidt isomorphisms.
3. The attempt to demonstrate FINDER on both biomedical and remote-sensing tasks shows some versatility.

### Weaknesses
1. The current presentation lack true methodological novelty, most of the theoretical developments are restatements of known results:
- The “generalized” KLE theorem (Thm. 2.1) is essentially the standard Hilbert-space KLE without the separability assumption, a minor technical relaxation known in the stochastic analysis literature (e.g., Schwab & Todor 2006).
- The link to classification through eigen-decomposition of the covariance operator is well-known (PCA, kernel PCA, functional data analysis). FINDER’s interpretation as “stochastic features” adds no genuinely new insight.
- Lemma 2.3 and the subsequent “bounds” are mathematically weak (simple Markov inequalities) and do not yield actionable or tight probabilistic guarantees.
2. The connection between the theoretical and practical setting is missing. The main theoretical results operate in infinite-dimensional Hilbert spaces with no clear justification for their finite-sample or computational counterparts. Claims such as “properties pass to truncations” are asserted but not proved or empirically verified. The presented algorithms (MLS, ACA-S/L) are described heuristically, with parameters (M_A, M_{\text{res}}) chosen ad hoc.
3. Empirical evaluation is weak and not reproducible.
- The experiments lack ablation and baseline parity. No comparison to simple but strong baselines (e.g., PCA + SVM, denoising autoencoders) is shown, so the reported improvements (often ≤ 2–3% AUC) are hard to interpret.
- The Alzheimer’s and remote-sensing results rely on heavy preprocessing and bespoke pipelines. It is unclear whether the same data splits, normalization, and computational resources are used for the benchmarks.
- Reported computational speedups are not standardized (e.g., wall-clock time on what hardware?). Some results appear cherry-picked without confidence intervals or statistical tests.
4. Some writing and presentation related issues
- The paper is overly dense, with long derivations that obscure the core ideas. Key algorithms are buried in appendices; the reader struggles to see what is actually new.
- The tone is over-claiming, with phrases like “state-of-the-art breakthroughs” not substantiated by evidence.
- Figures and tables are cluttered, with inconsistent notation and missing units.
5. Limited scope and scalability
- Current work is restricted to binary classification; the multi-class setting is not addressed beyond trivial pairwise decomposition.
- The approach depends on computing and decomposing large covariance operators, which is unlikely to scale beyond small datasets.
- There is no discussion of how this work would integrate with modern large-scale ML pipelines or stochastic training methods.

### Questions
1. It would be necessary to sharpen the algorithmic contribution, show a concrete, implementable advantage of stochastic features over standard PCA/kernel methods.
2. Can you provide theoretical novelty beyond textbook KLE—e.g., new concentration bounds or generalization guarantees.
3. Can you add rigorous experimental analysis, including ablations, fairness checks, and statistical significance.
4. Substantially simplify and reorganize the exposition for readability.
5. Temper the claims and clearly delimit the scope of the method.

### Soundness
2

### Presentation
2

### Contribution
1

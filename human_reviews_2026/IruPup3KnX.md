# ConfHit: Conformal Generative Design with Oracle-Free Guarantees

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
The success of deep generative models in scientific discovery requires not only the ability to generate novel candidates but also reliable guarantees that these candidates indeed satisfy desired properties.  Recent conformal-prediction methods offer a path to such guarantees, but its application to generative modeling in drug discovery is limited by budget constraints, lack of oracle access, and distribution shift. To this end, we introduce ConfHit, a distribution-free framework that provides validity guarantees under these conditions. ConfHit formalizes two central questions: (i) Certification: whether a generated batch can be guaranteed to contain at least one hit with a user-specified confidence level, and (ii) Design: whether the generation can be refined to a compact set without weakening this guarantee. ConfHit leverages weighted exchangeability between historical and generated samples to eliminate the need for an experimental oracle, constructs multiple-sample density-ratio weighted conformal p-value to quantify statistical confidence in hits, and proposes a nested testing procedure to certify and refine candidate sets of multiple generated samples while maintaining statistical guarantees. Across representative generative molecule design tasks and a broad range of methods, ConfHit consistently delivers valid coverage guarantees at multiple confidence levels while maintaining compact certified sets, establishing a principled and reliable framework for generative modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors present ConfHit, a conformal prediction technique that can be used for two applications:

(i) Certification: Given a set of examples, assert whether at least one example fullfills some property (called *hit*), with the guarantee that incorrectly detecting a hit comes at probability $\leq \alpha$.

(ii) Design: Generate a minimal set of examples using a generative model, such that there exists at least one hit, with probability $\geq 1- \alpha$ (built on top of the method for (i)).

Importantly, and in contrast to prior work, ConfHit does not assume access to an oracle for performing calibration, but only assumes access to a set of examples without a hit.

The method is evaluated on constrained molecule optimisation and structure-based drug discovery and performs favorably in comparison to naive baselines.

### Strengths
Overall, I believe that this paper is strong and should be accepted.

1. The writing is phenomenal. Everything is clear, the notation is wisely chosen. Also, Figure 1 is excellent.

2. The motivation is well-justified. The authors tackle a relevant limitation of existing approaches.

3. Rigorous theoretical analysis is present.

4. The method is relatively simple.

5. Strong experimental evaluation.

### Weaknesses
A great concern I have is the fact that ConfHit crucially hinges on a perfect estimate of the density ratio $w$. While Theorem 3.4 nicely quantifies the effect of estimation error of $w$ on the guarantee, the corrected bound is obviously not computable in practice. The authors motivate their technique via the need for guarantees for erroneous generative models. However, the technique itself relies on a erroneous density estimate, for which the guarantee does no longer hold. Thus, it seems that ConfHit shifts the problem from one ML model to another ML model, but it does not actually solve the problem. The authors write that they use kernel density estimation on top of a neural network representation to estimate $w$. Is there anything that suggests that trusting this density ratio estimate is more reasonable than simply trusting the outputs of the generative model? Density estimation is a harder problem than generative modeling (especially in high dimensions), so I have doubts.

The authors briefly discuss this point in their limitations paragraph of the conclusion, but I think it would be important to stress this crucial issue more (ideally, even in the manuscript's abstract) and go into why estimating $w$ should be more reliable than estimating the generative model. I am willing to increase my score if the authors convincingly address this point.

### Questions
* l.10: *"Recent advances extend CP to provide guarantees in language generation (Angelopoulos et al., 2021; Quach et al., 2023)"* I believe that [1, 2] should also be cited here.

* l.169: That means we discard all data that fulfills the property of interest? I wonder whether the statistical efficiency of the method could be improved by incorporating these examples somehow.

* Algorithm 1: Maybe I missed it, but how large should we choose the "original" set $\lbrace X_{n+j} \rbrace_{j=1}^N$? Obviously, if $N$ is too small, the set $\lbrace k \in [N]: p_k \leq \alpha \rbrace$ will be empty. What do we do in this case? Reject returning a set? I believe that this corner case is missing in the algorithm.

[1] Kladny, Klaus-Rudolf, Bernhard Schölkopf, and Michael Muehlebach. "Conformal Generative Modeling with Improved Sample Efficiency through Sequential Greedy Filtering." International Conference on Learning Representations (2025).

[2] Shahrokhi, Hooman, et al. "Conformal prediction sets for deep generative models via reduction to conformal regression." Uncertainty in Artificial Intelligence (2025).

### Soundness
2

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
CONFHIT is a wrapper you put around any generator so that a small shortlist of molecules comes with a statistical promise: with confidence , at least one on this list is a hit. It does this without calling an oracle and while correcting for distribution shift between old data  new generations made from the generator.

### Strengths
Main strength is that this paper tries to tackle one of the critical problems in drug discovery. The fact that it is model-agnostic and no oracle is required is another strength. This makes the method actually usable in resource-constrained settings. THe Nested Testing Framework seems useful because it doesn't just certify N samples but returns the smallest certified subset

### Weaknesses
1. Coverage hinges on density-ratio correction (and the covariate-shift assumption). 

2. May return an empty shortlist under tight α or small N. The paper notes that empty sets are inevitable without strong assumptions when the generation budget is limited, and empirically Bonferroni is almost always empty while CONFHIT still has ~16% empty sets in SBDD—improving on baselines but still a practical failure mode.

3. Needs an independently trained scoring model; power hinges on that predictor. The method seems to be only valid if the conformity/score function V is trained independently of the calibration/test data

### Questions
1. How robust is coverage when the density-ratio w(x) is misspecified—can you bound degradation or provide diagnostics a practitioner can run before trusting certification?

2. Can you offer principled guidance for choosing N and \alpha (or adaptive rules) that minimize empty-set rates while preserving guarantees?

3. How sensitive is CONFHIT to mild leakage or correlated training (e.g., feature reuse) between mu hat and calibration/test data, and can cross-fitting or data-splitting strategies be recommended to preserve coverage without sacrificing too much power?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The article describes variant of conformal prediction for guaranteeing that a batch of samples generated from a generative model satisfies a predefined property. The use case in mind is molecular generation, where such a predefined property could be a valid base-scaffold or a certain amount of improvement over an existing seed molecule.

### Strengths
The article is overall well-written and the concepts are presented in a clear and coherent manner. Numerical examples indicate that the method can indeed be applied to molecular generation problems.

### Weaknesses
- The core contribution is rather incremental from a statistical/mathematical perspective. The ConfHit algorithm (Alg. 1) is a variant of a permutation test that has been known in the literature for a long time. The idea of incorporating nested testing is a relatively straightforward addition (and I would be surprised if similar techniques have not been used before, maybe in a slightly different context).

- The significance of the numerical results are difficult to interpret (I am also not an expert in molecular generation). However, baseline comparisions against earlier methods are omitted with the argument that a completely new set-up is studied, where there is a distribution shift between the calibration data and the distribution of the generative model. Nonetheless, in the absence of such distribution shift a comparison to the relevant baselines Quach et al., 2023, and Kladny et al., 2024 seems doable and provides an indication of the size and the quality of the generated set.

- The method requires the estimation of a covariate shift w between the distribution used for calibration and the distribution of the generated samples. It might be difficult to reliably estimate w in practice.

- The latex-style file has been modified to ommit the label Figure 5, possibly to deal with page restrictions. I recommend the authors not to do this in future submissions, as it is an unfair practice and seems to be against the submission rules.

- The guarantees hold marginal and not hold conditional on the calibration data. This could be emphasized and explained more as it might otherwise mislead and misguide the interpretation of the statistical guarantee. Moreover, for getting a guarantee conditional on the calibration data, one would potentially need a large calibration dataset, exacerbating the computational challenge (permutation testing) of the proposed approach.

### Questions
- ConfHit is based on a permutation test, where permutations are subsampled to tame the computational effort. However, the subsampling introduces variance that only decreases with 1/B. Hence estimation of p-values might be difficult, in particular if these are small and might require large B and significant computational effort. Have the authors experienced issues along these lines? In order words, would one be able to obtain similar results if computational time is reduced from 30min (App. C3) to 5min.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces CONFHIT, a framework that brings formal reliability guarantees to conditional generative design under tight experimental budgets. Instead of assuming an oracle that can instantly validate new samples, the method leverages historical labeled data and corrects distribution shift between past and newly generated candidates through density ratio weighting. It builds joint conformal p values over batches to certify that a generated set contains at least one viable hit and extends this to a nested testing strategy that prunes a large batch to a compact shortlist while retaining finite sample error control. The authors ground the approach with theory and evaluate it on constrained molecule optimization and structure based drug discovery, showing tight error control across target levels and models, practical reductions in certified set size, and sensible behavior in ablations that remove density correction.

### Strengths
The problem is well framed around the realities of discovery work where assay budgets are limited and oracle access is infeasible, and the split between certification and design clarifies when guarantees are possible and how to act when they are. The use of weighted exchangeability to address covariate shift is principled and connects cleanly to modern conformal theory, and the nested testing idea is simple to implement yet surprisingly powerful, needing only monotone individually valid p values to avoid multiplicity penalties. The empirical study spans multiple model classes and tasks and reports both coverage and power with clear plots, while the ablation on removing density correction provides convincing evidence that the weighting is not cosmetic. The paper is mostly model agnostic in how it defines conformity scores and offers several reasonable choices, which improves the portability of the framework. The budget allocation vignette is a useful bridge from statistics to program level decision making.

### Weaknesses
The guarantees depend critically on accurate density ratio estimates built on learned features, and the robustness analysis, while helpful, still leaves open how to diagnose and mitigate misspecification in practice when calibration data are scarce or shift is large. The approach relies on a property predictor to define the conformity score, so end to end performance will be sensitive to that model and feature extractor, yet guidance on model selection, calibration, and failure modes is limited. Comparisons to prior conformal generative methods are discussed as not directly applicable because of oracle assumptions, but the experimental section would be stronger with alternative practical baselines beyond Bonferroni, for example heuristic pruning driven by predictor uncertainty, or variants that reuse unlabeled generated data for semi supervised calibration. Computational cost may be nontrivial due to permutation sampling and kernel density estimation, and the paper does not provide a thorough accounting of wall clock cost versus gains in certified set size. The empirical scope focuses on small molecule scenarios with in silico oracles rather than wet lab measurements, so external validity to other domains such as proteins or materials remains uncertain. Finally, some design choices are only briefly justified, such as training the density model on a subset and the specific feature layers used, and readers may benefit from clearer guidance on hyperparameters like the number of permutations, batch budgets, and how to pick among score statistics without post hoc tuning. 


---


Itemized weaknesses for rebuttal and discussion:

1. Heavy reliance on accurate density ratio estimates with limited guidance for diagnosing misspecification
2. Sensitivity to the property predictor and feature extractor with sparse advice on model selection and calibration
3. Limited baselines beyond Bonferroni; lacks comparisons to stronger practical heuristics or semi supervised variants
4. Computational overhead from permutation testing and density modeling without clear accounting of wall clock cost
5. Evaluation focused on in silico small molecule settings, leaving external validity to other domains uncertain
6. Several design and hyperparameter choices are under explained, reducing reproducibility and practitioner guidance

### Questions
-

### Soundness
3

### Presentation
3

### Contribution
3

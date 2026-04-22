# Causal Partial Identification with Data Augmentation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
We present a novel framework for using knowledge of data symmetries to sharpen bounds in causal *partial identification (PI)*. The causal effect of the treatment $X$ on outcome $Y$ is generally not identifiable from observational data alone if their common causes, also known as confounders, are unobserved. PI entails estimating bounds on such treatment effects by solving a constrained optimization problem that encodes different assumptions imposed on data generation. PI has use in many application domains where such bounds are sufficient to inform policy decisions, even if the treatment effect itself is not identifiable. We show that knowledge of symmetries in data generation—formalized as invariance under transformation groups—provides additional constraints that tighten these bounds. We operationalize this insight through two approaches: (1) adding explicit invariance error constraints to existing PI methods, and (2) applying symmetry-preserving *data augmentation (DA)* as a pre-processing step. Under a linear Gaussian model, we show that the later yields bounds that provably valid (containing the true causal effect), sharper (smaller identified sets), and more robust (lower worst-case error). The key mechanism being that randomized symmetry transformations introduce exogenous variation in $X$ that cannot be attributed to confounding, thereby reducing ambiguity in the identified set. Experiments on synthetic and real data validate our approach. More broadly, our findings establish known data symmetries—ubiquitously employed in DA for variance reduction—can be repurposed as a principled tool for causal inference when point-identification is impossible.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper seeks to provide theoretical bounds for using data augmentation when coupled with partial identifications. The bounds proposed give both best- and worst-case scenarios for partially identifying the causal averages they employ, with respect to predefined causal risks.

### Strengths
Although I am no expert in neither domain augmentation nor partial identification, it seems to me that combining these two ideas is quite novel. The quality of writing is not incredibly good (see Weaknesses). 

The theorems appear to be sound and (without taking a close look at the proofs) the conclusions do seem natural enough. 

The concepts presented are clearly explained and it is clear that the theorems are algebraic derivations based on these definitions.

### Weaknesses
Adding an image to illustrate an observation of the dataset described in Section 6.2 would make the explanation more intelligible.

The abuse of notation in Section 3 (Line 202) was quite confusing at first. It seems to me that the notation employed could be formally introduced, to prevent confusions. 

The first equation hints to a very general functional model, that is later replaced by a linear model. Possibly because the results of Vankadara et al., (2022) are only valid for linear models. Is this the only bottleneck that prevents extending the current framework to more general models?

### Typos and errors: 
Several citations in the first pages (and possibly other parts of the text) should be citep, instead of citet. This makes reading of these pages quite painful and annoying.

- Line 20: ubiquitous.
- Line 273: extra 'e' in perturbs.
- Line 281: I suspect there are some parentheses or mathematical notation missing for $f^T x \in H_{da+pi}(x), H_{pi}(x)$.
- Line 461: out -> our.
- Line 464: identification.
- Line 466: Rosenbaum is inconsistently-cited (no year).
- Equation (5), the norms used are not clearly defined. The (very subtle) difference between $X$ and $\mathbf{x}$ inside the do operator, makes these hard to distinguish on a first read.

### Questions
The authors claim these bounds are valid for the infinite-data setting and naturally are only able to assess in a finite-data setting. However, it would be good to numerically assess how the derived bounds behave for different data-size regimes.

Outside of the optical device data used in Section 6, it is hard for me to envision the type of causal questions where the proposed tools would be useful. Could the authors point out other settings where this could be useful? Note that I am not requesting more experiments with this question, but they would be welcome.

In Theorems 1 and 2, I assume that the '_equality iff_' would also correspond to having slack equal to zero. Is this intuition correct? If so, how does the slack increase for different degrees of independence? Since the models are linear (and possibly Gaussian), measuring this dependence could be easily done by correlations/covariances.

On that note, the Gaussian assumption (over $G, U, N_X$, and $N_Y$) in Example 1 is quite strong (and it later affects all the theoretical developments), in what key ways does your contribution depend on this?

### Soundness
3

### Presentation
1

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
The authors investigated whether outcome‑invariant data augmentation (DA) can sharpen partial identification (PI) bounds for causal effects under the presence of unmeasured confounders. The key idea lies in constructing an outcome-invariant data augmentation transformations G as a transformation intervention do(x=Gx). A very simple linear Gaussian case study is provided. By experiments, the authors showed that DA could help reduce the length for the pointwise interval width. For the theoretical analysis, the authors showed that DA lowers worst‑case causal excess risk over the identified set. This seems to be a flexible approach as the DA approach is model‑agnostic and compatible with many PI frameworks.

### Strengths
1.	The conceptual idea is very clear and straightforward. Constructing a outcome-invariant DA as transformation intervation and then use it as a plug-in module in any PI pipeline.
2.	The theoretical analysis is solid, especially for the and lower worst case excess risk (Theorem 2)
3.	The proposal is a pre‑processing step that can be composed with existing PI methods without changing their solvers or constraint sets.

### Weaknesses
1.	The linear–Gaussian SEM seems still restrictive with additive noise and a particular sensitivity model. Under other common cases, such as non-Gaussian, non-linear outcome model, many guarantees may not work.
2.	The outcome-invariant assumption is hard to be testable, there is no diagnostic or stress test for misspecified augmentations.
3.	The numerical results are not comprehensive enough. Only one real dataset (Optical Device) is used with fixed sample size (n=1000).

### Questions
1.	on the Optical Device data, why should flips/rotations/Gaussian noise be outcome‑invariant for f Can you provide predictive‑invariance checks?
2.	Can any of the theoretical results be extended to non-Gaussian case? Or with some additional mild assumption?
3.	How to choose G seems very important and tricky. A practical guideline for choosing G is very necessary.
4.	A comprehensive sensitivity analysis to mis‑specified DA is very important for understanding the role of DA.
5.	There are many augmentation parameters, such as noise scale, angle, etc. How these parameters affect the PI and the prediction performance?
6.	The authors should test DA in multiple PI framework to demonstrate compatibility and gains beyond the partial R^2 model.
7.	Since authors argue compatibility with IVs, a toy example where an IV is present, showing that DA preserves IV validity and may further tightens bounds, should be very convincing.

### Soundness
2

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
3

### Summary
This paper proposes a method based on data augmentation to improve partial identification bounds in causal inference when unobserved confounding prevents point identification. The key idea is to use data augmentation, specifically, an outcome-invariant augmentation, as an auxiliary source of information that can act as a ``soft intervention.''
If we can transform data in ways that leave the outcome function unchanged (say, rotations of a picture in a classification task), these transformations can help tighten the partial id bounds.
The method requires background knowledge about the regression function, for instance the null space of the coefficients matrix in a linear regression model, which can be quite limiting in practice.

### Strengths
1. Even though the analysis is restricted to linear-Gaussian systems, I believe most of the results have straightforward generalizations to more complex models.

---
2. Synthetic and semi-synthetic experiments consistently show that DA+PI yields tighter bounds than baseline PI.

### Weaknesses
1. Significant overlap with Akbar et al [1]:
Taking a look at the first reference of the paper, one can notice immediately that this paper has a significant overlap with, and is essentially not adding much to Akbar et al. Surprisingly, a great deal of content is copied without even rewording. Ideas are copied from that paper: DA as soft intervention, Figure 2 of the paper with its caption, the running example of the paper, and most significantly, the theoretical results of this paper such as proposition 2, Lemma 3 and Lemma 4, all appear in Akbar et. al., and the rest of the results provided here are quite trivial or straightforward at best to prove given those.
Compared to Akbar et al., I don't see much of a novel idea, novel proof, novel result, or novel presentation, begging the question what is the merit of this paper given that? Only bringing up the observation that DA can be used for partial identification too? Then I do not believe this paper is contributing enough to the literature.

---
2. The paper presents its main claims (“valid bounds”, “sharpened partial identification”, etc) early as if fairly general (Sections 1–3). But when you dig into Section 4, you find that the proofs are only in the linear Gaussian “Example 1” setting with heavy structure/assumptions. This can be quite misleading. The authors should either restrict their claims explicitly in the first few sections, or lift their proofs to more general settings (if they can). 

---
3. The manuscript repeatedly introduces notation and small definitions close together; readers must hunt back and re-read definitions frequently. I believe this paper can be presented in a much better (readable, at the least) way if a bit of time is spent on it.

---
4. I am not convinced by the idea of using the background knowledge (say symmetries of f) in the way that this paper suggests. Read my question below too.

### Questions
If you already have symmetry knowledge about f, why do DA instead of directly imposing constraints?
If the researcher truly knows those symmetries, you can (in many cases) impose them directly in the inference/optimization (e.g., enforce invariance in hypothesis class H, add equality constraints, or augment the objective with invariance penalties) rather than applying DA as a pre-processing step. The manuscript should either 1) clearly justify why DA is preferable to directly injecting invariance into the estimator (computational simplicity? easier to combine with off-the-shelf solvers? better finite-sample behavior?), and provide a brief theoretical or experimental comparison; or 2) explicitly treat DA as one practical way to operationalize symmetry knowledge and discuss tradeoffs (what you lose/gain compared to e.g. constraining H).

### Soundness
4

### Presentation
1

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
This paper studies partial identification bounds under known symmetries of the causal effect, i.e., under invariances of $f$ when $Y = f(X) + \xi$ when $\xi$ may be correlated with $X$, and hence acts as an unobserved confounder. Primarily focusing on the population setting, they define a hypothesis class $H_{pi}$ of possible causal effects functions (each $h \in H_{pi}$ is a function from $x$ to $y$), which are consistent with the observational distribution $P_{X,Y}$ and satisfy the invariance constraints. For each $x$, this hypothesis class induces a set of possible causal effects $H_{pi}(x)$, which in turn can be used to define a worst-case excess risk. Assuming a multivariate Gaussian distribution over $(X, Y, \xi)$, they show that the excess risk strictly decreases under these constraints, as long as the symmetries are not almost surely orthogonal to the expectation of $X$ given $\xi$.

In practice, rather than strictly enforcing the invariances, they are captured using data augmentation. This approach is corroborated through a simulation experiment on a linear model, giving sharper identification bounds, and in real-world experiments on the Optical Device dataset.

### Strengths
**Originality and significance:** The use of known symmetries to sharpen partial identification bounds is an interesting and (to the best of my knowledge) novel direction. As the authors discuss, such symmetries are common in many applications, especially in scientific machine learning, where causal inference is quite important, so the combination of the two is a very good match.

**Quality and clarity:** The work is well-executed, the motivation is clear, and the mathematical details are well-written.

### Weaknesses
## Major weaknesses

1. **Limitation to multivariate Gaussian setting:** To the best of my understanding, the results are limited to multivariate Gaussian distributions on $(X, Y, \xi)$, and this limitation is not made as transparent as it should be. Based on the text after Assumption 1, I understand that the partial R-squared sensitivity model is not a necessary restriction, but I'm less certain about the Gaussianity assumption (or at least, a linearity assumption). For example, Proposition 1 invokes a Lebesgue measure over $H_{pi}$, which is initially defined as a function space, but in the proofs, $h$ is taken to be a vector (i.e., the coefficients of a linear function). Overall, this lack of transparency gives the feeling that the results are being oversold.
2. **Not enough focus on the quantitative form of the results:** In connection to Weakness 1, I would be much more interested to see Lemma 5 in the main paper and a more quantitative discussion of *how much* the invariances sharpen the partial identification bounds. Theorem 1, Proposition 2, and Theorem 2 don't provide any intuition about how much the invariance sharpens the bounds, which I think would be the most interesting part.

## Minor weakness
3. **Overly focused on data augmentation:** I think a more logical way to present the results would be to focus on how known symmetries/invariances improve the partial identification bound, and afterwards connect these results to data augmentation. The results are really about the restriction of the hypothesis space, and would hold even when strictly enforcing the invariance - the approach of using data augmentation is more of a practical implementation detail.
-

### Questions
Please address the Major Weaknesses, especially (1).

### Soundness
3

### Presentation
3

### Contribution
2

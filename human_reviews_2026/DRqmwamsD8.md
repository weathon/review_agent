# WHY TEACHER–STUDENT SELF-SUPERVISED LEARNING WORKS: A MUTUAL INFORMATION PERSPECTIVE

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
We study teacher-student (TS) self-supervised learning methods equipped with a prediction head (e.g., BYOL, SimSiam), which learn meaningful representations without relying on negative samples. Building on the InfoMax perspective that unifies many multi-view Self-Supervised Learning (SSL) families, we show that TS-SSL implicitly maximizes a lower bound on the mutual information $I(Z_\theta; X)$ between the inputs $X$ and the teacher representations $Z_\theta$. Concretely, we prove that, assuming an optimal predictor, BYOL and SimSiam's loss is an approximation $H(Z_\theta \mid Z_\phi, X)$. Building on this results, we prove that, under a mild assumption, verified empirically on six different datasets, the alternating optimization—student prediction (with stop-gradient) followed by teacher updates—implicitly optimizes $\theta$ so that it maximizes $I(Z_\theta; (X, Z_\phi))$ a lower bound on $I(Z_\theta; X)$. Then, we derive increment convergence dynamics of the teacher representation’s entropy and alignment during training. Eventually, motivated by these theoretical insights, we introduce a simple mutual-information–based regularizer on the student latent space that enforces monotonic growth of $I(Z_\theta; X)$ and yields consistent downstream improvements on both natural-image and medical-imaging benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a new multi-view self-supervised learning (SSL) loss that is motivated by the InfoMax principle, aiming to maximise a lower bound on $I(X;Z)$.

The derived loss is empirically evaluated against self-implemented baselines (Barlow Twins, VicReg, SimCLR, Dino, Barlow Twins, Mocov2, SimSiam, BYOL) on natural image datasets (Cifar10, Cifar100, STL10, Imagenet100) as well as medical image datasets (BloodMNIST, PathMNIST, DermaMNIST, Camelyon16, BRACS) and compares mostly favourably to those.

### Strengths
- The overview of related literature in introduction and background and related work is extensive 

- It is great to see SSL methods tried out on medical imaging benchmarks

### Weaknesses
- The core equations of the method (Eqns 18 and 19) seem to have mistakes in them: in Eq 18 the entropy term $H(Z_{\phi})$ should be subtracted not added I believe, in Eq 19 the last term should be subtracted not added, and also the $\beta$ from Equation 18 is nowhere to be found in Eq 19. Lastly, why would the term that I think is meant to estimate $H(Z_{\phi})$ in Eq 19 (i.e. this last term) involve $f_{\theta}$?

- The natural image experiments are not conducted on full ImageNet, which has been the standard in the literature (BYOL, SimCLR, …). This raises doubts about the scalability of the proposed approach and also raises questions about fair comparison, since no independent source can be compared against for accuracy (e.g. baselines could be suboptimally trained by authors - not even on purpose. As someones who has trained these beasts before I know how hyperparameter sensitive they can be)

- Without the last term in Eq 18, the suggested modification to BYOL would look almost identical to what is proposed and investigated with an entropy regulariser in Rodriguez-Galvez et al. (2023). This warrants an ablation of its effect to better understand novelty and significance of the contribution, but it cannot be found in the paper.

### Questions
- Contribution 1. - what is ‘thus bringing to an estimator of […].’ supposed to mean? Bringing what?
- Eq 2, 3 and 4: how is H(p || q) to be understood? I don’t know the notation in this case, I only know || for divergence notations but not for entropy. 
- Eq 2: how is $q$ defined here?
- Eq 9: how is H(A|B;C) to be understood? I would understand H(A|B) as the conditional entropy of A given B, but with a third variable separated by a semicolon I am not familiar with the notation. 
- Assumption 1: how can this assumption be justified or otherwise verified in practice?
- Figure 2: how is the MI measured here?
- Where did $\beta$ from Eq 18 go in Eq 19?
- Line 478: where is this premature saturation shown in the paper?
- Line 480: typo? Should it not be $I(Z_{\phi};X)$ instead according to Eq 18 and 19?

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates teacher-student (TS) style self-supervised learning and aims to provide an information-theoretic explanation for its effectiveness. Initially, the authors analyze BYOL from an information-theoretic perspective and demonstrate that the student update implicitly maximizes a lower bound on the mutual information between the representation $Z_\theta$ and the input X. Subsequently, under the assumption of a Gaussian isotropic latent space, they derive the incremental dynamics of the teacher's entropy and find that increasing the variance ratio can promote better alignment between the teacher and student. Based on this insight, the paper introduces an additional regularization term into BYOL's optimization objective. Experimental results show that this regularization improves performance.

### Strengths
The paper presents interesting research with adequate theoretical analysis.

### Weaknesses
1. Insufficient discussion of related work: The paper claims several key contributions, including theoretical analyses of the predictor, stop-gradient mechanism, and EMA, explanations for the non-collapsing behavior of TS-SSL, and the introduction of mutual information-based constraints. However, similar arguments have been made in existing literature [1, 2, 3], and the paper fails to explicitly discuss distinctions from these prior works.


2. Overstated title: TS-SSL encompasses diverse methods, such as the classic DINO. Yet, this paper only analyzes BYOL and SimSiam, which are insufficient to represent the broader TS-SSL framework, making the title overly broad.


3. Limited experimental validity: Experiments are conducted on small-scale datasets, which lack representativeness in the current era. This raises concerns about the method’s effectiveness in real-world application scenarios.

### Questions
In Line 344, the assumption $Z_{\theta^{t+1}} = \tau Z_{\theta^t} + (1 - \tau) Z_{\phi^t}$ is introduced. Does the EMA update in the parameter space directly translate to the same update rule in the representation space?

### Soundness
2

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
The paper looks to understand Teacher-Student SSL methods from an information theoretic perspective to justify their performance, analogously to the mutual information maximisation perspectives of other SSL methods, such as infoNCE.

### Strengths
The aim of understanding these SSL methods from an information theoretic perspective, interpreting the implicit distributional assumptions they make and making principled improvements is a sound approach.

The material improvement in results on a number of benchmark datasets suggest the proposed method is useful.

### Weaknesses
The main weaknesses of the paper are readability and the mathematical arguments don't seem correct or well-presented and I believe need to be materially re-worked.

* readability - the notation is hard to follow (SSL typically considers 2 related samples x, x' and their representation z, z', which is easier to follow than tracking subscripts).
   - The paper overloads symbols and mixes random variables with their realizations. For instance, $Z_\theta=f_\theta(t(X))$ and $Z_\phi=f_\phi(t(X))$ have the same augmentation $t$, then later switch to multi‑view $v_i^{(1)}$, $v_i^{(2)}$ (Eq. (4)). The dependence structure (one view vs two independent views) must be fixed up front, e.g. with $t,t'\stackrel{iid}{\sim}\mathcal{T}$. As written, it implies $Z_\theta, Z_\phi$ are deterministic functions of the same $t(X)$, which is not the BYOL/SimSiam setup and creates confusion for all MI statements. 
* The mathematical arguments are difficult to follow, e.g.
   - it would help if the loss function being explained were stated upfront (is this RHS of Eq 5?).
   - definition of conditional entropy (Eq 1) is already an expectation over X (e.g. see https://en.wikipedia.org/wiki/Conditional_entropy)
   - "alignment" is typically considered between representations of related samples, Eq 1 doesn't consider that so hard to see how this relates, e.g. to Wang & Isola.
   - Eqs 2/3 seems to be the Barber & Agakov bound, eg see "On Variational Bounds of Mutual Information" by Poole et 
al (as cited!). This is simply an instance of cross entropy lower bounding entropy (Eq 2), which is a fundamental of machine learning and far from a "novelty".
   - the last term in Eq 2 should be in expectation (over X).
   - rather than referring to "kernel density estimation" (vague/general), it should be stated in explicit terms if the first term of Eq 3 is equivalent to Eq 4 under specific Gaussian assumptions for p and q (assuming that is the case)? It is well known that the cross entropy of two Gaussians has the general form in Eq 4 (under specific assumptions that are not made clear here).
   - Eq 4 mixes exact expectations and MC estimates (from samples). 
   - Eq 4 is a function of z under two different distributions (where z is a representation of a particular view of x), it is unclear how that then becomes a function of different z's/views.
      - From the outset, it would be clearer to refer to x and x' as different views (or similar), as is common to avoid confusion.
   - it is unclear what happens to the entropy term in Eq 3.
   - in the context of the number of operations in a neural network, it is invalid to suggest that a multiplicative factor is set to 1 for "computational efficiency" (rather this is part of the p/q assumptions above)
* the assumptions (including Eq 11) seem strong, unintuitive or not very well justified, particularly Assumption 2 and Eq 11.

Details
* 268 - t already used to define augmentation
* 269 - Z's are defined as deterministic functions of the same t(X) and therefore of each other? It is unclear why this relationship would be time invariant given that other functions are not.
* 321 - this seems highly unintuitive for a relationship presumed to hold throughout training for a finite model.

Minor
* 177 - include the domain of v = t(x) (presumably $\mathcal{X}$)
* 192 - an unusual way of writing conditional cross entropy (double lines usually reserved for divergence)

### Questions
See weaknesses

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study teacher-student self-supervised learning (TS-SSL) methods and focus on addressing the issue that these methods lack a clear information-theoretic explanation. They show that TS-SSL implicitly maximizes a lower bound on the mutual information between inputs and the teacher representations. They also give convergence results characterizing the evolution of the teacher representation’s entropy and alignment during training. By introducing a mutual-information–based regularizer on the student latent space, the authors give empirical results that show improvements on natural-image and medical-imaging benchmarks.

### Strengths
The authors show theoretically that TS-SSL maximizes a lower bound on the mutual information between inputs and the teacher representations, which helps to design a mutual-information–based regularizer that leads to empirical improvements on real datasets. The results are interesting.

### Weaknesses
This manuscript contains some technical weaknesses, such as referring to missing Sections, lacking of justification for important assumptions and inconsistent derivations, which I list in detail in the QUESTIONS part.

### Questions
1. In Section 4.1 the authors refer to Sections A.5 and A.6 in the appendix. However, I cannot find Sections A.5 and A.6 in the manuscript.

2. In Assumption 2, the function f is assumed to exhibit first-order variation (i.e., the derivative remains constant), which is claimed to hold for both the teacher and the student terms. This assumption is somewhat too strong. Please justify. Furthermore, the parameter variation is intertwined with the variation of the mutual information term (Eq. 12) in Lemma 1, which serves as a core of the analyses and impacts the validity of the conclusion.

3. From Lemma 1 and Eq. 9 in Section 3.1, the authors analyze the variation of the teacher term given the student term and input indicating that this conditional entropy can be used to assess the lower bound of mutual information. However, the conditional entropy in Eq. 16 does not incorporate the student term as a condition. Instead it directly asserts that the entropy conditioned solely on the input should be reduced. This is somewhat inconsistent with the previous derivations.

4. Regarding Eq. 17, the authors’ explanation of this equation can be insufficient. They do not clarify the rationale for introducing the entropy constraint in Eq. 17. In the preceding content, the authors illustrate that the improvement direction involves reducing the conditional entropy and increasing the marginal entropy, but they fail to explain why the conditional entropy constraint is specifically introduced here.

5. Regarding Eq. 18, the authors previously mentioned the need to reduce one variance ratio and increase another. However, in Eq. 18, both regularization terms are assigned coefficients greater than zero, which is strange. 

6. Typos/small mistakes:

In Eq. (18), is the first Lagrangian multiplier term \lambda H(Z_{\phi})?

In Table 1, ROBYOL is actually lower than BYOL on IN100 (R50).

In Abstract, SSL is used without definition.

### Soundness
2

### Presentation
3

### Contribution
3

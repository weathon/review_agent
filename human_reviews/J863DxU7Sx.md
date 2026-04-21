# Exploiting Hidden Symmetry to Improve Objective Perturbation for DP Linear Learners with a Nonsmooth L1-Norm

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Objective Perturbation (OP) is a classic approach to differentially private (DP) convex optimization with smooth loss functions but is less understood for nonsmooth cases. In this work, we study how to apply OP to DP linear learners under loss functions with an implicit $\ell_1$-norm structure, such as $\max(0,x)$ as a motivating example. We propose to first smooth out the implicit $\ell_1$-norm by convolution, and then invoke standard OP. Convolution has many advantages that distinguish itself from Moreau Envelope, such as approximating from above and a higher degree of hyperparameters. These advantages, in conjunction with the symmetry of $\ell_1$-norm, result in tighter pointwise approximation, which further facilitates tighter analysis of generalization risks by using pointwise bounds. Under mild assumptions on groundtruth distributions, the proposed OP-based algorithm is found to be rate-optimal, and can achieve the excess generalization risk $\mathcal{O}(\frac{1}{\sqrt{n}}+\frac{\sqrt{d\ln(1/\delta)}}{n\varepsilon})$. Experiments demonstrate the competitive performance of the proposed method to Noisy-SGD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper studies differentially-private optimization of nonsmooth convex functions. A subclass of such functions with "hidden $\ell_1$ structure" is defined, for which a smoothing+objective perturbation is shown to achieve the optimal rate (which is not the case in general).
Some experiments show the algorithm preforms well compared to baselines.

### Strengths
The paper studies an important problem, and does a good job motivating it with an excellent introduction, posing it compared to prior work.

There are several technical contributions throughout the paper, in order to apply the OP approach and prove its utility. These are explained fairly well in my opinion, and are generally of general interest.

The experiments make a convincing point that the approach considered in this work might have practical importance.

### Weaknesses
The assumptions made throughout the paper, the formulation of which are arguably the main contribution, seem pretty limiting.

Two things particularly seem limiting:

1. Although the word "hidden" structure appears in the title and in the name of the main assumption, it actually **needs to be known explicitly** by the algorithm, if I am not mistaken. The algorithm seems to need to know the decomposition into the "$\ell_1$-component" and the rest in the first place, which only holds for very structured problems such as GLMs or computing quantiles (which is indeed the one considered in the experiment).

2. The results neglect the dependence on the matrix dimension $m$, which is considered a small constant. This is what really allows the smoothing approach to work here, which would otherwise have a strong dimension dependence (preventing the optimal rate, as far as I can tell). While this is true for some structured problems considered in the paper, this seems pretty specific.

### Questions
I would appreciate the author's comments on the two issues I raised in the previous "weakness" box.

Some additional minor comments:

- Line 076: It is unclear at this point of the paper what $\theta$ vs. $z$ are.
- Line 082: "and etc." should be simply "etc.".
- Several places, including Lines 092 and 107, use \cite (or \citet) where \citep is more appropriate.
- Line 130: $B(R)$ is the ball around the origin, I suppose? This is unclear.
- If possible, rather slightly move Algorithm 1 so that it appears at the beginning or end of the paragraph it currently cuts-off in the middle.
- Assumption 4.4 and related discussion: the notation $\|\cdot\|_{-\infty}$ is used, perhaps should be $\infty$ instead of $-\infty$? Otherwise I don't understand this notation.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work proposes C-OP, an algorithm for differential privacy. C-OP is designed for a specific class of problems, namely one with a smooth component and a non-smooth component containing the $\ell_1$ norm. The key design is the random perturbation within the $\ell_1$ norm. The algorithm achieves $\epsilon-\delta$ privacy with optimal excess risk.

### Strengths
1. The theoretical analysis is technically impactful. Lemma 4.3 as a technical lemma is very useful.
2. The presentation follows good logic.
3. The proposed algorithm is effective in experiments.

### Weaknesses
1. It would be better if the authors could discuss the explicit result of the convolution. The reviewer did not check it, but they believe that the $\mathbb{E}_k[A(z)\theta+\mu k]$ can be explicitly written in all choices of $k$ of interest. If not, the authors are expected to discuss how the convolution is implemented.
2. It would be interesting to see the intuition of applying convolution to the part of $\ell_1$ loss only instead of the entire objective function. See Question 1.
3. In several subfigures of Figure 5, the plot goes out of the boundary.
4. The discussion about the choice of $\mu$ and $\sigma$ is not detailed enough. It would be better if the authors could discuss whether $\mu$ and $\sigma$ should increase or decrease with the change of other hyperparameters.
5. According to my understanding, the algorithm cannot be applied to the smooth setting. See also Question 2.

### Questions
1. As the reviewer is not quite familiar with the literature, it would be interesting to see why previous methods that try to use convolutional smoothing fail to achieve the optimal risk but the algorithm in this paper can. Is it due to the new analysis techniques in this paper? If not, what consequence does the convolution on the smooth component bring about?
2. As the scale of $A(z)$ decays, the problem becomes close to a smooth problem, and it seems that more noise needs to be injected into the convolution to preserve privacy. How can the authors explain this using Theorem 3?

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
3

### Summary
The authors extend the Objective Perturbation (OP) framework for differentially private (DP) convex optimization, traditionally applied to smooth loss functions, to accommodate a specific class of nonsmooth convex losses. Specifically, they address nonsmooth functions characterized by an implicit $\ell_1$-norm structure. Their method involves smoothing only the nonsmooth component of the objective function via convolution, followed by the application of an OP-based algorithm to uphold DP guarantees.

### Strengths
The problem is well-motivated, and the core ideas of the paper are clearly articulated. Although the use of convolution to obtain a smooth approximation of a nonsmooth function is not novel in itself, it enables the derivation of interesting and novel theoretical results. The primary innovation lies in leveraging an approximation function that provides an upper bound on the original nonsmooth objective. This approach results in a new decomposition of the excess generalization risk, which is subsequently used to derive optimal generalization rates.

### Weaknesses
In my opinion, the main limitation of the paper lies in the applicability of Assumption 4.4. For example: the $\ell_1$-norm structured nonsmooth functions which is the focus of this paper is often utilized in sparse linear regression where $\theta^*$ is sparse. The  Assumption 4.4 does not seem to hold for such cases. Besides, without the knowledge of $\theta^*$, one cannot verify this assumption easily.

### Questions
Can the authors comment on the applicability and verifiability of Assumption 4.4? It would strengthen the paper to provide some concrete examples related to real-world applications where the proposed assumption holds.

### Soundness
4

### Presentation
4

### Contribution
3

# High-dimensional limit theorems for SGD: Momentum and Adaptive Step-sizes

- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
We develop a high-dimensional scaling limit for Stochastic Gradient Descent with Polyak Momentum (SGD-M) and adaptive step-sizes. This provides a framework to rigourously compare online SGD with some of its popular variants. We show that the scaling limits of SGD-M coincide with those of online SGD after an appropriate time rescaling and a specific choice of step-size. However, if the step-size is kept the same between the two algorithms, SGD-M will amplify high-dimensional effects, potentially degrading performance relative to online SGD. We demonstrate our framework on two popular learning problems: Spiked Tensor PCA and Single Index Models. In both cases, we also examine online SGD with an adaptive step-size based on normalized gradients. In the high-dimensional regime, this algorithm yields multiple benefits: its dynamics admit fixed points closer to the population minimum and widens the range of admissible step-sizes for which the iterates converge to such solutions. These examples provide a rigorous account, aligning with empirical motivation, of how early preconditioners can stabilize and improve dynamics in settings where online SGD fails.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
This paper extends the high-dimensional limit theory of Ben Arous et al. (2022, 2024) to analyze Stochastic Gradient Descent (SGD) with Polyak momentum (SGD-M) and adaptive step-size variants (including a normalized-gradient variant, denoted SGD-U). The authors derive the asymptotic stochastic differential equations governing the evolution of summary statistics of SGD-M and adaptive SGD, under suitable regularity assumptions (delta-localizability and delta-closability). The theory establishes an equivalence between SGD-M and online SGD up to a time rescaling, and provides rigorous justification for the stabilizing role of gradient normalization in high-dimensional regimes. These results are illustrated on two canonical problems: spiked tensor PCA and single-index models, supported by both analytical fixed-point analyses and simulations.

### Strengths
I would like to note that I am not a specialist in stochastic differential equations or high-dimensional diffusion limits. While I have done my best to assess the work carefully, my understanding of some of the deeper technical aspects is limited.

1. The paper presents a non-trivial and technically rigorous generalization of existing high-dimensional diffusion limit results to momentum and adaptive-step-size algorithms. Extending the practical dynamics framework of Ben Arous et al. to handle additional algorithmic components is a clear and meaningful step forward. The identification of the equivalence between SGD-M and SGD via time rescaling (Theorem 2.3 and Remark 2.4) is elegant and provides theoretical insight on known empirical observations.

2. The analysis of normalized-gradient SGD (SGD-U) offers a rigorous interpretation of empirical stabilization mechanisms (e.g., gradient clipping, normalization), bridging stochastic process theory and optimization practice.

### Weaknesses
1. The paper is extremely technical and assumes strong familiarity with high-dimensional probability, weak convergence, and diffusion processes. While this is common in this line of work, some readers may find it challenging to connect the formal definitions. For instance, the concepts of delta-localizability and delta-closability are poorly described

2. I am not entirely sure about this claim, but it seems he localizability and closability conditions are mathematically strong (requiring smoothness up to third derivatives and bounded high-order moments). It seems to me that this condition is basically assuming the algorithm’s stochastic trajectories converge to an SDE, which compresses both randomness and nonlinearity into a self-contained stochastic differential equation. These may hold for the idealized models considered but might limit its applicability to more realistic scenarios (where similar behavior of SGDM and SGD have been observed, like in deep learning).

3. I might have missed this part, but it seems the authors does not provide examples of problem that satisfies delta-localizability and delta-closability. Moreover, it seems these are never explicitly verified for either of the two worked examples (spiked tensor PCA or single-index models).

### Questions
1. Is it possible to verify, even heuristically, that the paper's assumption are satisfied in one of the examples, or provide sufficient conditions under which they hold?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper extends the high-dimensional effective dynamics of summary statistics framework of online SGD in [1] to online SGD with Polyak momentum (SGD-M) and online SGD with adaptive step-sizes. Based on these results, the authors looked at two applications: the spiked tensor PCA and single index models. In these applications, the authors 

(1) consider a specific online SGD with adaptive step-sizes defined by normalizing gradient, referred as SGD-U;

(2) derive the ballistic limits of the summary statistics of SGD-M and SGD-U and study the fixed points in different regimes;

(3) derive the diffusive limits of the summary statistics of SGD-U at the equator in spiked tensor PCA.

[1] Gerard Ben Arous, Reza Gheissari, and Aukosh Jagannath. *High-dimensional limit theorems for
SGD: Effective dynamics and critical scaling*. Communications on Pure and Applied Mathemat-
ics, 77(3):2030–2080, 2024.

### Strengths
1. The paper provides a **valuable generalization** of high-dimensional diffusion limits to include both momentum and adaptive step-size variants of SGD. This extension broadens the understanding of optimization dynamics beyond standard online SGD, offering a unified framework for several widely used algorithms.

2. The main theorems and assumptions are rigorously stated and internally consistent.

### Weaknesses
1. The manuscript does not clearly highlight its key novelty—the extension from online SGD to momentum and adaptive-step methods. The relation to existing online SGD results is underdeveloped theoretically, which makes the contribution appear incremental despite being substantial.

2. The effective limits of SGD-U is difficult to follow. Neither a general limiting theorem or a comparison to SGD/SGD-M is provided.

3. The paper’s structure could better emphasize the main theoretical message. Sections on motivation, scaling intuition, and comparison to prior results are brief, while technical proofs dominate early sections, making it hard to identify the central takeaway.

### Questions
1. How does $\beta$ affect the $\delta_n$-localizable condition defined in Definition 2.1?

2. in line 322-323, the coefficient of the diffusion term is incorrect;

3. Why do you scaling by $\sqrt{n}$ in SGD-U? As explained in the paper, if $\lVert \nabla L_n \rVert=O(\sqrt{n})$, wouldn't the adaptive step-size of constant order, which contradicts to that we want the step-size tend to zero?  

4. When is there a diffusive limit in the single index model example?

5. Please double check for typos. Some that I noticed are listed below:

    (1) $\mu$ in line095 should be $P_n$;
   
    (2) $\nabla L$ in equation (1) should be $\nabla L_n$;

    (3) in line 230: $R^2$ should be $m^2+r_\perp^2$;

    (4) $r$ and $r_\perp$ are the same. Please be consistent in notations.

### Soundness
2

### Presentation
1

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
The paper extends the main ideas of high-dimensional scaling limit for Stochastic Gradient Descent presented in Ben Arous et al. (2024) to two important variants: SGD with momentum and SGD with adaptive step-sizes. The paper provides a solid theory, and extensive experiments verify the results and justify the need for the exploration of these ideas.

### Strengths
The paper is well written and well-organized with clear contributions. I enjoy reading the motivation of the main ideas, and in particular, i find the writing of section 2 on the main result of the paper very informative. Theorem 3.2 is the main theoretical contribution of the work, with the examples in Section 3 to justify the need for the theoretical result.

The proofs for the major steps I checked seem correct, and the theoretical statements are reasonable.

### Weaknesses
I do not have to point out a specific weakness. I like this work and I believe it is worth being accepted to ICLR.

Some questions:
Typically, the HIGH-DIMENSIONAL LIMIT THEOREMS of an algorithm are purely theoretical results. In this work, the authors show that SGD-M will amplify high-dimensional effects, potentially degrading performance relative to online SGD. Can they comment on what else can be an interesting follow-up to these ideas? What other algorithms might have a similar outcome?

I believe having such remarks and paragraphs at the end of the paper would be particularly valuable, in a section, let's say, named Conclusion, “Remarks and Future impact”.

### Questions
See the Weaknesses part.

### Soundness
3

### Presentation
3

### Contribution
3

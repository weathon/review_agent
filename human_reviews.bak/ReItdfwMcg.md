# Optimality and Adaptivity of Deep Neural Features for Instrumental Variable Regression

- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
We provide a convergence analysis of \emph{deep feature instrumental variable} (DFIV) regression (Xu et al., 2021), a nonparametric approach to IV regression using data-adaptive features learned by deep neural networks in two stages. We prove that the DFIV algorithm achieves the minimax optimal learning rate when the target structural function lies in a Besov space. This is shown under standard nonparametric IV assumptions, and an additional smoothness assumption on the regularity of the conditional distribution of the covariate given the instrument, which controls the difficulty of Stage 1. We further demonstrate that DFIV, as a data-adaptive algorithm, is superior to fixed-feature (kernel or sieve) IV methods in two ways. First, when the target function possesses low spatial homogeneity (i.e., it has both smooth and spiky/discontinuous regions), DFIV still achieves the optimal rate, while fixed-feature methods are shown to be strictly suboptimal. Second, comparing with kernel-based two-stage regression estimators, DFIV is provably more data efficient in the Stage 1 samples.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides theoretical guarantees for the DFIV method, which uses adpative deep features to solve the non-parametric IV regression problem. The authors prove an upper bound for the risk of the DFIV estimator when the true structual function lies in a Besov space. Secondly, the paper presents an information-theoretic lower bounds for NPIV in Besov spaces and shows that DFIV can achieve the minimax optimal rate with a balanced number of Stage 1 and 2 samples. Lastly, the paper demonstrates that DFIV has superior empirical performances when compared with linear fixed-feature methods.

### Strengths
- Showed that DFIV can achieve the minimax optimal rates, which is positive result. 
- Included an analysis on the sample sizes required for Stage 1 and Stage 2, which can provide more insights into the algorithm.

### Weaknesses
- This paper would benefit from including a brief discussion on the technical novelty in the analysis/proofs, which is important since this paper mainly presents theoretical results. 
- Section 4 (separation with fixed-feature IV) would also be better supported with some experimental results showcasing the improvement of DFIV over linear estimators. 
- DFIV is an iterative method, but the paper did not include any results/rates of convergence, which can be an important practical issue.

### Questions
I am curious to see a discussion on the smoothness assumptions that is used in this work and the $\beta$-source condition that is often assumed for inverse problems (and also used to study NPIV).

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
4

### Summary
The article provides a convergence analysis of the DFIV algorithm under certain assumptions and proves its minimax optimality. It compares DFIV with fixed feature methods, demonstrating its superiority in low spatial homogeneity and data efficiency.

### Strengths
1.	The convergence and optimality analysis in this paper is necessary to understand the DFIV algorithm. 

2.	The motivation is well-presented.

3.	The theoretical analysis is solid and It provides a rigorous foundation for DFIV.

### Weaknesses
1.	This article is symbol-dense and hard to follow. It lacks some illustrative diagrams or symbol tables.

2.	The rationality of some assumptions (e.g., Equation (2)), the reasons for transforming problems (e.g., from Equation (3) to (4), and the role of mapping \(T\) are not thoroughly explained in its introduction part. In addition, where and why is Definition 2.2 used… The text lacks explanations for these points, and the necessity of many techniques used is not clearly articulated.

3.	See Questions.

### Questions
1.	The article mentions many assumptions; so what does the term "Additional assumption" in the abstract specifically refer to? Is it Equation (2) or others? If this assumption is crucial for the derivation, the abstract should succinctly highlight its necessity rather than merely emphasizing its existence.

2.	Regarding Equation (2), a graphical illustration could be added to provide a more intuitive understanding of this part.

3.	The title mentions "adaptivity." Is there a distinction between "data adaptivity" and "data efficiency"(in abstract)?

4.	Line 41: The functional equation is not direct, and a reference to Equation (4) in Section 2 should be added here.

5.	Comparing existing statistical analyses in studying ordinary regression and here two-stage setting, what’s the difference between these two types of theoretical tools? Under the current setup, what technical challenges are brought?

6.	Line 128-143: Why is Problem/Equation (3) transformed into Problem/Equation (4)? Given that the resulting problem (4) is generally ill-posed and the solution may be ill-behaved or not unique, why is it necessary to introduce the operator \(T\)? Can't the original problem be directly solved? The role of \(T\) needs to be explained in detail here.

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
2

### Summary
**Summary:**
Nonparametric two-stage approaches to instrumental variable regression can be classed as either using fixed features  features (e.g. RKHS, spline, dictionary) or data-adaptive features (e.g. deep neural network). Of the latter approaches. deep feature instrumental variable regression has been shown to empirically perform very well, but lacks theoretical analysis. The authors prove that DFIV achieves minimax optimal rates when the target structural equation belongs to a certain class (Besov space). In order to do so, the authors use standard nonparametric IV assumptions plus one more assuption on the conditioanl distribution of the covariate given the instrument. DFIV is also superior to fixed-feature methods for two other reasons: when the target function has low spatial homogeneity (it can be smooth or bumpy in places), DFIV still attains the optimal rate but fixed feature methods fail. Secondly, the two-stage regression models are more data efficient in the first stage than kernel methods.

### Strengths
**Strengths:**
- It is mentioned on line 450 that DFIV is shown to require strictly less data in stage 1 than in stage 2, whereas for kernel IV, an assymetric amount of data is needed in stage 1 and stage 2. 
- I think this paper provides some significant results: namely, it provides a rigorous analysis of DFIV and shows advantages in terms of adaptivity, sample efficiency and balacning in stage 1 and 2, and justifies the use of neural features.

### Weaknesses
**Questions/Weaknesses:**
- "The above loss generally cannot be directly optimized for θx as ˆθz cannot be backpropagated through". Is this for practical or conceptual reasons? One can imagine either running some iterations of an iterative solver (e.g. Newton's method, SGD, ...) and then backpropagating through the iterates of the solver (often called "unrolling"). Alternatively, one can compute the derivatives using the implicit function theorem, by computing the solution and then plugging directly into a known form. But my guess is that either of these approaches fails to yield a useful gradient in practice?
- Assumption 5 - $\mathcal{P}_{\mathcal{X}}$ has Lebesgue density bounded below. Does this exclude e.g. Gaussian densities? Is this a restrictive assumption? One way I can see this assumption satisfied is if the support is compact and the density is bounded from below (otherwise, without a compact support, I think the density may not even be integrable?)
- As far as I understand, neural network models are used (belonging to class on line 245) as the class in stage 1 and stage 2 problems (7) and (8). It seems to be assumed that a minimum exists (and perhaps is unique?), and even more strongly, that this minimum can be obtained by the algorithm. The derived minimax rates assume that the actual minimum is found, which is not realistic. Or maybe with sufficient overparameterisation, it is realistic?
- The smoothness of stage 2 needs to be controlled, and the authors propose two ways to do this. One is to restrict the function class, which in practice requires somehow discarding ill-behaved bumpy solutions after random intialisation. However, practically instantiating this procedure is not really discussed. Another option is to penalise the Besov norm, and I am not sure how hard it is to actually compute a Besov norm for a given neural network. Are these smoothness penalties/constraints actually practical? 
- This paper is very theoretical, containing no experiments. This is a little bit (but probably not unheard of) for ICLR. A venue like COLT might attract readers more strongly aligned with its contents. Alternatively, experiments could be done to actually demonstrate that the assumptions made are realistic. For example, it could be investigated how hard it is in practice for the smoothness of the stage 2 by discarding ill-behaved solutions, or whether a zero stage one (and/or small stage 2 loss) can be obtained with a sufficiently rich neural network. None of these experiments are hinted at.

**Minor:**
- Grammar. "...while the rate in Stage 1 samples m is depends on the smoothness of both..."
- The sentence on line 368 is a bit confusing. Are you saying $\log(\cdot)$ is replaced by $(\log(\cdot))^3$?
- Typo on line 446.

### Questions
Please address the weakness/questions above. Referring to the details above:
- Why can't stage 2 derivatives go through to stage 1?
- Is Assumption 5 restrictive (in particular, does it include Gaussian?)
- Is it assumed that stage 1 and stage 2 is solved over this neural network class? How realistic is this assumption?
- Are the smoothness penalties/constraints actually practical, or do they only allow the theory to work? If they are not practical, then perhaps it is overselling to claim that the method of Xu et al., 2021 really achieves the minimax rate, because it is after all a practical algorithm.
- Is it possible to show how mild these assumptions or regularisers are by doing a toy experiment?

### Soundness
3

### Presentation
3

### Contribution
3

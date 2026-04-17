# CLASP: An online learning algorithm for Convex Losses And Squared Penalties

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
We study Constrained Online Convex Optimization (COCO), where a learner chooses actions iteratively, observes both unanticipated convex loss and convex constraint, and accumulates loss while incurring penalties for constraint violations. We introduce CLASP (Convex Losses And Squared Penalties), an algorithm that minimizes cumulative loss together with squared constraint violations. Our analysis departs from prior work by fully leveraging the firm non-expansiveness of convex projectors, a proof strategy not previously applied in this setting. For convex losses, CLASP achieves regret $O\left(T^{\max\\{\beta,1-\beta\\}}\right)$ and cumulative squared penalty $O\left(T^{1-\beta}\right)$ for any $\beta \in (0,1)$. Most importantly, for strongly convex losses, CLASP provides the first logarithmic guarantees on both regret and cumulative squared penalty. In the strong convex case, the regret is upper bounded by $O( \log T )$ and the cumulative squared penalty is also upper bounded by $O( \log T )$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the problem of Constrained Online Convex Optimization (COCO), and introduces a nove algorithm named Convex Losses And Squared Penalties (CLASP). Theoretically, this paper establishes an $O(T^\max{\beta, 1-\beta})$ regret bound and an $O(T^{1-\beta})$ CCV bound, for general convex functions, and an $O(\log T)$ regret bound and an $O(\log T)$ CCV bound, for strongly convex functions. Moreover, some empirical studies are conducted to support the theoretical findings.

### Strengths
- The paper is well-organized, with clear problem introduction, and methodology explanation.
- The authors present concise theorem proofs to enhance readability.

### Weaknesses
- It is unclear which term, $CCV\_{T,1}$ or $CCV_{T,2}$, is more general. The explanation in Lines 68–74 is somewhat ambiguous and should be clarified.
- The statements of **Lemmas 1–3** and **Theorems 1–2** are not sufficiently formal. All necessary assumptions and variable definitions must be explicitly provided.
- In the experimental section, it is recommended to include comparative results with [1], since [1] also investigates COCO problems with strongly convex loss functions.
- It appears that the open-source code in the url is missing.
- Overall, the paper presents a simple and clear algorithm along with sound theoretical analysis. However, the theoretical contributions are limited compared with existing works [2, 3]. The authors should highlight the technical challenges addressed in this study to better justify its contribution.

[1] Revisiting projection-free online learning with time-varying constraints. 2025.

[2] Optimal algorithms for online convex optimization with adversarial constraints. 2024.

[3] Online convex optimization for cumulative constraints. 2018.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies Constrained Online Convex Optimization (COCO), where a learner chooses actions at each round while incurring penalties for constraint violations. Most of existing work focus on cumulative constraint violation (CCV) bound, while this work investigates squared constraint violations. Previous work on $\text{CCV}_{T,2}$ (Yuan & Lamperski, 2018) studies static constraints, and this work extend their result to the setting of dynamic constraints. Furthermore, they also provide logarithmic guarantees on both regret and squared CCV.

### Strengths
This paper is clearly written, with a thorough review and discussion of related work. After reading the entire paper, I have gained a comprehensive understanding of COCO.

### Weaknesses
The main weakness of this paper lies in its limited technical novelty. Extending the previous result on squared constraint violations from static to dynamic constraints does not appear to involve substantial technical challenges. In my view, as long as the constraints satisfy Assumption 3, Lemma 3 always holds, making the derivation of the $\text{CCV}_{T,2}$ bound rather straightforward. Therefore, this work seems to be a combination of existing techniques without introducing new technical tools.

Several sections could be written more clearly. Specifically, Sections 3 and 4 present the proposed algorithm and the main theoretical guarantees, yet the authors dedicate only a very short space to describing them. Moreover, it would be more appropriate to merge Sections 3 and 4 into a single section. I strongly encourage that the authors elaborate on Section 3 and 4 in more detail, as the current presentation is too brief.

Minor suggestion: theorems and lemmas should be presented in a more formal manner (e.g., Theorems 1 and 2, Lemmas 1, 2 and 3), clearly specifying which assumptions are used.

### Questions
* What are the technical challenges of this paper? As mentioned in the Weaknesses, I believe that the technical contribution of this work is rather limited.

* Have similar guarantees to Lemmas 1–3 been used in previous studies (Yuan & Lamperski, 2018)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies constrained online convex optimization (COCO) with adversary constraints. Performance is measured by static regret and the cumulative squared constraint violation. The authors propose CLASP, which at each round takes a gradient step on the current loss and then projects onto the current feasible set. For general convex losses with step sizes, CLASP attains regret $O\big(T^{\max\{\beta,1-\beta\}}\big)$ and squared violation $O\big(T^{1-\beta}\big)$. Under strong convexity, CLASP yields logarithmic bounds on both regret and violation. The main technique is to use non-expansiveness (FNE) of projections.

### Strengths
+ This paper studies constrained online convex optimization (COCO) with adversarial varying constraints, evaluated by static regret and the cumulative squared violation. It provides theoretical guarantees in the both convex and strongly convex regime.

### Weaknesses
- The paper does not justify why the squared violation metric is necessary or reasonable in the dynamic setting ($CCV_{T,1}$ seems more reasonable), nor does it provide a clear comparison against the hard violation $CCV_{T,1}$ results. By Cauchy-Schwarz, $CCV_{T,1} \leq \sqrt{T \cdot CCV_{T,2}}$. Hence in the convex setting, the theoretical guarantees on $CCV_{T,2}$ translate to $CCV_{T,1}$ that are strictly weaker than recent bounds (e.g., Sinha and Vaze, 2024). Even under strong convexity, the paper still implies only $O\big(\sqrt{T\log T}\big)$ for $CCV_{T,1}$. To make the contribution convincing, the paper should present lower bounds specific to squared violation and a principled discussion of when squared violation is the right target. 

- The method requires maintaining a per-round safe set and computing a projection onto $K\cap C_t$, which can be substantially more expensive than the previous projection-free methods (e.g., RECOO and AdaGrad) when $C_t$ is a complicated set. 

- Theorem 2 requires knowing the strong-convexity parameter $m$ of all future loss functions. This could be infeasible or non-casual in a practical setting. 

- The experiments show AdaGrad performing poorly despite stronger theoretical guarantees. It would be better to provide a detailed discussion.

### Questions
Please see weakness above.

### Soundness
2

### Presentation
3

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
This paper studies COCO problems with time-varying convex losses and constraints. The authors propose an algorithm called CLASP that performs a gradient step followed by a projection onto the current feasible set. The main claimed contribution is achieving logarithmic regret and squared constraint violation under strong convexity, and sublinear trade-offs in the general convex case. The analysis relies on the firm non-expansiveness (FNE) property of projection operators, which the authors argue provides a cleaner proof and modular structure compared to prior work.

### Strengths
The paper is clearly written and easy to follow (e.g., all assumptions and lemmas are clearly stated). The structure is standard and logical. The logarithmic bound for the strongly convex case is a nice result.  The experiments use relevant baselines on synthetic and real-world tasks and the results show that CLASP performs competitively.

### Weaknesses
The proposed algorithm is essentially a standard projected gradient method for COCO extended with time-varying constraints: at each round, it performs one gradient step on the loss followed by projection onto the time-varying constraint set. The only new element is the squared penalty measure $CCV_{T,2}$, but it was already analyzed in prior work [Yuan & Lamperski, 2018] for static constraints. The claimed novelty that leveraging FNE in the analysis is mainly a technical proof refinement rather than a new algorithmic idea. So the theoretical analysis, while clean, does not seem to introduce a fundamentally new technique (e.g., new regret decomposition inequality in [Sinha & Vaze, 2024]) that would warrant publication at a top-tier conference like ICLR.

### Questions
Regarding the strongly convex result: how practically significant is the difference between bounding the squared violation CCV_{T,2} at $O(\log T)$ versus the linear violation CCV_{T,1} at $O(\log T)$ or $O(\sqrt{T\log T})$ as in prior work? Can you provide an intuition for when the squared penalty guarantee is substantially more useful?

### Soundness
3

### Presentation
3

### Contribution
3

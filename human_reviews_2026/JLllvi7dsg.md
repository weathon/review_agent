# A Faster Parameter-Free Regret Matching Algorithm

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Regret Matching (RM) and its variants are widely employed to learn a Nash equilibrium (NE) in large-scale games. However, most existing research only establishes a theoretical convergence rate of $O(1/\sqrt{T})$ for these algorithms in learning an NE. Recent studies have shown that smooth RM$^+$ variants, the advanced variants of RM, can achieve an improved convergence rate of $O(1/T)$. Despite this improvement, smooth RM$^+$ variants lose the parameter-free property, i.e., no parameters that need to be tuned, a highly desirable feature in practical applications. In this paper, we propose a novel smooth RM$^+$ variant called Monotone Increasing Smooth Predictive Regret Matching$^+$ (MI-SPRM$^+$), which retains the parameter-free property while still achieving a theoretical convergence rate of $O(1/T)$. To achieve these properties, MI-SPRM$^+$ employs a technology called Adaptive Regret Domain (ARD), which ensures that the lower bound for the 1-norm of accumulated regrets increases monotonically by adjusting the decision space at each iteration. This design is motivated by the observation that the range of step-sizes supporting the $O(1/T)$ convergence rate in existing smooth RM$^+$ variants is contingent on the lower bound for the 1-norm of accumulated regrets. Experimental results confirm that MI-SPRM$^+$ empirically attains an $O(1/T)$ convergence rate.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a smooth regret matching+ variant called Monotone Increasing Smooth Predictive Regret Matching+ (MI-SPRM+), which retains the O(1/T) convergence guarantee of current SOTA methods, while also being parameter-free. To achieve this, the authors introduce the Adaptive Regret Domain (ARD) method, which relies on the insight that the range of permissible stepsizes depends on the lower bound of the 1-norm of accumulated regrets. MI-SPRM+ dynamically adjusts the decision space after each iteration, ensuring that the lower bound monotonically increases and recovering the O(1/T) convergence property. Experiments show that MI-SPRM+ exhibits the theoretical convergence rate in standard benchmark games, and seems to outperform other existing methods in terms of duality gap.

### Strengths
- The paper is fairly well written, and the motivation of reducing the parameters required for tuning (especially for solving large games) is sound.
- The technical contribution and insight behind the ARD method is strong, resulting in a main theoretical result that successfully obtains strong convergence rates while being parameter free.
- The experimental results are surprisingly strong, particularly in EFGs, showing faster convergence than all prior approaches. Exploring this behavior is a crucial step for future work.

### Weaknesses
- The algorithm description for MI-SPRM+ is very densely written, and some more intuition would greatly improve the readability of this work.
- The paper presents a single convergence guarantee for NFGs, but it would be improved if a similar result can be shown for EFGs, even in the non-CFR variant (i.e. MI-SPRM+ directly applied to the sequence form representation of the EFG). If anything, the fact that MI-SPCFR+ seems to outperform SOTA methods makes it all the more compelling to see if there is some theoretical reasoning for this. 
- While the paper shows a comparison between the compute times of the main algorithms implemented, I believe more discussion or experiments could help clarify the relationship between 1) convergence rate in terms of duality gap and 2) compute time required. In particular, it seems that MI-SPRM+ achieves parameter freeness by incurring additional adaptive steps in the process. It is however not made clear to the reader what the computational cost of parameter tuning actually is (for instance in the case of SPRM+). It seems that the parameter tuning for SPRM+ amounted to stepsizes in $\{0.01, 0.1, 1\}$ but it is not clear to me if a grid search was performed, and if that would impact the compute times accordingly.
- The sentence in Line  307: "Unfortunately, to the best of my knowledge..." is oddly phrased and could be restructured.
- "do" in Line 48 should be "are".

### Questions
- The authors use the phrasing "even faster than O(1/T) convergence" several times in the paper, but this is a strong statement to make, given the lack of theoretical justification. Could the faster convergence be attributed to constant factors? I would suggest changing the wording to avoid confusing readers who might misconstrue the complexity of MI-SPRM+.
- In Fig 7, we see that for some games, the value of $R^t$ seems to plateau then subsequently grow as the number of iterations increases. Is there some intuition for what this means? Would this imply that the algorithm shifts to a different NE, since the decision space changes after some period of stability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes MI-SPRM$^+$, a smooth RM$^+$ variant that introduces an adaptive regret-domain floor so that the 1-norm of cumulative regrets grows to a level where the standard stability inequality holds. Once the floor is high enough, the scheme behaves like SPRM$^+$ without requiring tuned step sizes. The main theoretical claim is an $O(1/T)$ rate for a weighted average strategy with weights proportional to the domain floor sequence. Experiments include NFGs and EFGs in both two-player and multiplayer settings, and demonstrates favorable performance of the proposed method, when using uniform averaging for the algorithms compared to.

### Strengths
1. The paper studies an important problem of trying to design faster parameter-free regret-matching algorithms for equilibrium computation in games and recover parameter-free behavior while keeping the O(1/T) rate associated with smooth RM+ variants. A simple modification to SPRM+ is provided that achieves O(1/T) convergence for weighted average iterates.  

2. The empirical section is broad. The method is tested in both NFGs and EFGs and compares against common RM-style and OCO baselines.

### Weaknesses
1. Notation and presentation need work; there is a lot of inline math in the preliminaries section which makes it difficult to read, and the notation feel excessively verbose.
2. The experiments only use uniform averaging, which is not particularly interesting from a practical perspective. It is well-established that linear or quadratic averaging should be used, or even the last iterate in some cases.
3. The experiments do not use alternation. While it is good to test on the simultaneous case to verify the theory, numerical take-aways should most likely be concluded for the alternating variant.
4. The proposed change does retain parameter-freeness, but it does not retain stepsize-invariance, another property cojectured to be important for RM-based algorithms.

### Questions
1. The theoretical result is based on a weighted average of the iterates, which is not standard in decentralized learning settings. While this doesn’t take away from the result, it would be good to emphasize this in the abstract and intro.

2. Why not include alternation and quadratic averaging for the CFR+/predictive CFR+ baselines, as is standard in the literature? These choices often improve empirical performance substantially on the EFGs you report.

3. In Table 1, are the reported times for a fixed number of iterations (1e5?) or for reaching a fixed duality gap?

4. Given that you run experiments for computing NE in multiplayer games (and use duality gap as the potential function), it might sense to not specify the notation/preliminaries section to use 2 players (even though the theoretical work is focused on the 2p0s setting) and let it be more general (some of the notation is already more general).

5. Why is a proof for $O(1/T)$ of SPRM$^+$ given? Isn't that just restating results from Farina et al?

### Soundness
3

### Presentation
2

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
This paper proposes a new variant of regret matching, called Monotone Increasing Smooth Predictive Regret Matching+ (MI-SPRM+), that enjoys a parameter-free setting and achieves an O(1/T) convergence rate. It differs from SPRM+ by (1) eliminating the need of using \eta, the step size of accumlating regret and (2) introduce an adaptaive scheduling of R^t, a lower bound of the regret values. The authrors also present relevant experimental results.

### Strengths
The technical contribution is sound.

### Weaknesses
The experiments lack important baselines.

### Questions
I think MI-SPRM+ is neat and achieving a provable O(1/T) convergence rate is great. My biggest concern is about the experiments on EFG: It would be great if baselines like CFR+ and DCFR is included.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Monotone Increasing Smooth Predictive Regret Matching+ (MI-SPRM+), a novel variant of regret matching (RM) that simultaneously achieves two highly desirable but previously incompatible properties in the RM family:

Parameter-free: The algorithm requires no hyperparameter tuning (e.g., step size), making it robust and practical for real-world deployment.
O(1/T) theoretical convergence rate to Nash equilibrium in two-player zero-sum normal-form games (NFGs).
To our knowledge, this is the first RM-based algorithm that attains both properties. Prior smooth RM+ methods (e.g., SPRM+, Farina et al., 2023) achieve O(1/T) convergence but lose parameter-freeness due to their dependence on a carefully tuned step size η. In contrast, classical RM+ and PRM+ are parameter-free but only guarantee O(1/√T) rates.

The key technical innovation is the Adaptive Regret Domain (ARD) mechanism, which dynamically expands the lower bound of the decision space (i.e., the 1-norm of accumulated regrets) to ensure the conditions for O(1/T) convergence are eventually satisfied—without any user-specified parameters. This approach is conceptually distinct from existing parameter-free methods like DS-OptMD, which adaptively shrink the step size and suffer from slow empirical convergence.

### Strengths
A rigorous theoretical analysis proving the O(1/T) convergence of the weighted average strategy under MI-SPRM+.
Comprehensive experiments on NFGs and extensive-form games (EFGs), showing that MI-SPRM+ consistently outperforms state-of-the-art baselines—including SPRM+ with optimally tuned η—in both convergence speed and final solution quality.
Empirical validation that the algorithm achieves O(1/T) (or faster) rates in practice across diverse game settings.
Overall, the work makes a clear and meaningful contribution to the regret minimization literature by resolving a concrete limitation in existing RM algorithms, with both theoretical depth and practical relevance.

### Weaknesses
1) Larger-scale game benchmarks: The experiments are conducted on standard small to medium-sized games (e.g., Kuhn/ Leduc Poker, random NFGs). To better demonstrate the algorithm’s scalability and practical relevance, it would be valuable to include results on larger, more challenging domains—such as a heads-up no-limit Texas Hold’em (HUNL) subgame, which is commonly used in recent game-solving literature.
2) Comparison with state-of-the-art CFR variants: The paper compares MI-SPRM+ against PRM+ and SPRM+, but does not include recently proposed advanced CFR algorithms like DDCFR[1] or PDCFR+[2]. Given that these methods also aim to accelerate convergence in EFGs, a direct comparison would help contextualize the empirical advantage of MI-SPRM+ more convincingly.

[1] Xu Hang,et al. Dynamic discounted counterfactual regret minimization.  ICLR 2024.
[2] Xu Hang,et al. Minimizing weighted counterfactual regret with optimistic online mirror descent. IJCAI 2024.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

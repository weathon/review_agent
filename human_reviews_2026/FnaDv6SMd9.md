# Non-Convex Federated Optimization under Cost-Aware Client Selection

- Decision: Accept (Oral)
- Scores: 6, 6, 4, 6

## Abstract
Different federated optimization algorithms typically employ distinct client-selection strategies: some methods communicate only with a randomly sampled subset of clients at each round, while others need to periodically communicate with all clients or use a hybrid scheme that combines both strategies. However, existing metrics for comparing optimization methods typically do not distinguish between these strategies, which often incur different communication costs in practice. To address this disparity, we introduce a simple and natural model of federated optimization that quantifies communication and local computation complexities. This new model allows for several commonly used client-selection strategies and explicitly associates each with a distinct cost. Within this setting, we propose a new algorithm that achieves the best-known communication and local complexities among existing federated optimization methods for non-convex optimization. This algorithm is based on the inexact composite gradient method with a carefully constructed gradient estimator and a special procedure for solving the auxiliary subproblem at each iteration. The gradient estimator is based on SAGA, a popular variance-reduced gradient estimator. We first derive a new variance bound for it, showing that SAGA can exploit functional similarity. We then introduce the Recursive-Gradient technique as a general way to potentially improve the error bound of a given conditionally unbiased gradient estimator, including both SAGA and SVRG. By applying this technique to SAGA, we obtain a new estimator, RG-SAGA, which has an improved error bound compared to the original one.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes CGM-RG (Composite Gradient Method with Recursive Gradient Estimators), a framework for communication-efficient federated optimization that exploits functional similarity among client objectives without requiring frequent full-client participation. The key contribution is showing that SAGA-type variance-reduced estimators can exploit second-order dissimilarity, traditionally thought to require SARAH-type methods with full synchronization. By combining SAGA with recursive gradient updates and a composite gradient method, the authors establish state-of-the-art communication complexity bounds that match or exceed SARAH-based methods while supporting partial client participation.

### Strengths
- The paper makes a genuine algorithmic and theoretical advance. The central insight, that SAGA estimators can be analyzed to depend on $\delta$ rather than individual smoothness $L_{\max}$, is novel and non-obvious. 
- The introduction of three communication oracles (ACO, RCO, DCO) with associated costs (C_A, C_R) is a methodologically sound contribution.
- The paper is technically rigorous. The proofs appear sound (though I have not verified every detail). The generality of the framework, incorporating both SAGA and SVRG as instances of the same recursive gradient estimator approach, demonstrates good mathematical design. 
- Table 1 shows CGM-RG-SAGA achieves state-of-the-art communication complexity while avoiding periodic full synchronization.

### Weaknesses
- The experimental section is underdeveloped for a paper making strong theoretical claims: The quadratic with log-sum penalty (Section 4, Figure 1) is a well-controlled synthetic problem where theory-practice alignment is most likely.  Experiments use $n \in (10, 26)$ clients. Real federated learning involves thousands to millions of clients. It's unclear whether the proposed method scales and whether the theoretical constants are practical.
- EMNIST and CIFAR-10 experiments (Appendix I.3-I.4) are crucial for demonstrating practical relevance but are deferred and treated briefly.
- Missing practical insights: No wall-clock time comparisons, no investigation of parameter sensitivity (how to choose $\beta, \lambda, p$ in practice?).
- The damping factor $\alpha$ is added "to enhance empirical performance" (I.3), suggesting the base algorithm needed modification. This raises questions about whether the theory-driven design actually works well in practice or requires ad-hoc tuning. Why does the algorithm require adding a damping factor $\alpha$ in Section I.3? This suggests the base theory-driven design needs tuning. How sensitive are results to this choice?
- The paper assumes $\delta$ is small enough to provide benefits. However, no guidance on when $\delta ≪ L_{\max}$ in practice. No empirical method for estimating $\delta$ a priori.
- The CGM subproblem solver (Section 3.4) is referenced as CGM_const and CGM_rand, but the actual algorithms are not presented in the main paper.
- The discussion of SAG vs. SAGA could be clearer. The paper defers to Appendix H, but the main text should give readers intuition for why SAG fails.

### Questions
Refer to the weaknesses above.

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
The paper studies non-convex federated optimization under partial participation and proposes CGM-RG-SAGA/CGM-RG-SVRG—Composite Gradient Method (CGM) variants that plug a “recursive gradient” (RG) variance-reduced estimator into SAGA/SVRG to exploit a second-order similarity condition ($\delta$-SOD) alongside first-order dissimilarity ($\Delta_1$-ED). The core technical idea is a new multi-round variance bound for SAGA tightened by a factor of $ns$ via a tunable memory parameter $\beta=\Theta(1/ns)$, which then yields improved iteration and communication complexities for CGM with partial participation size $s$ (Theorem 3.8, Cor. 3.9). Under the proposed oracle model with costs $C_A$ (arbitrary/active) and $C_R$ (random), the authors claim communication $\tilde O(C_A nm + C_R(\Delta_1+\sqrt{ns}\,\delta_s)F_0/\varepsilon^2)$ for RG-SAGA, minimized at $s^\star=m$ and argued to be best among existing methods in the studied regime. :

### Strengths
- **Originality.** Introduces an RG-SAGA/SVRG estimator composition inside CGM that explicitly leverages $\delta$-SOD; the variance recursion is sharpened by a factor $ns$ relative to classic SAGA analyses (Cor. 3.7), which is technically neat and nontrivial. 
- **Quality.** Theory is stated with clear assumptions ($\delta$-SOD, $\Delta_1$-ED) and tracks the impact of estimator error and subproblem accuracy on iteration complexity (Theorem 3.8). The communication model (ACO/RCO/DCO) is formalized and tied to partial-participation costs, enabling proper comparisons. 
- **Clarity.** The CGM subproblem and its use of $\Delta_1$ are clearly exposed; the proof sketch for controlling $\|\nabla f(x_{r+1})\|^2$ via function decrease plus estimator error is transparent (Eq. (2) and subsequent inequalities). 
- **Significance.** If the stated bounds hold, the work sharpens the communication-vs-computation frontier for similarity-aware FL under partial participation, highlighting when $s<m$ suffices to reach the “full-gradient” rate (discussion after Cor. 3.9).

### Weaknesses
- **Novelty gaps vs. prior variance-reduced FL.** The “recursive” flavor has strong parallels to PAGE/SARAH-type recursions used in SABER and related CGM variants; several ingredients (e.g., plugging VR gradient trackers into CGM, periodic/full-grad syncs) exist in recent work. The paper’s comparison table is helpful, but the “best known” claim would benefit from a tighter, side-by-side theorem-level comparison against SABER’s second-order-aware CGM under matching assumptions and oracles
- **Assumption realism/tightness.** Core results require both $\delta$-SOD and $\Delta_1$-ED, and solving each CGM subproblem so that $\sum_{r}\|\nabla F_r(x_{r+1})\|^2=O(\varepsilon^2 R+\lambda F_0+\sum_r \Sigma_r^2)$ (Theorem 3.8), which is strong and hides nontrivial inner accuracy. The paper argues CGM can ensure this, but the constants are heavy (e.g., $\lambda\ge 5\Delta_1$ plus terms in $\sqrt{ns}\delta_s$), and the practicality of meeting the aggregate subproblem condition under stochastic local solvers is not empirically validated. 
- **Initialization costs and partial-participation realism.** RG-SAGA needs two full synchronizations (one to initialize the table, one composite step) incurring $\Theta(ns)$ ACO calls; while the text notes a heuristic “no-sync” variant often works, guarantees rely on the expensive path. In truly cross-device settings where ACO is much costlier than RCO, this may erase the asymptotic advantage for moderate accuracy. 
- **Proof-level concerns.** (i) The claimed $ns$ improvement relies on choosing $\beta=\Theta(1/ns)$; however, sensitivity of constants to $\beta$ and to client sampling variance is only partially exposed (Cor. 3.7 cites (E.8) but hides constants). (ii) The jump from Lemma-level recursion to Theorem 3.8 aggregates several bounds (58, 24, etc.); while Appendix F spells them out, the dependence on $L$ vs. $\Delta_1$ and $\delta_s$ could be more explicitly tracked in-text to make the gain scenarios unmistakable. 
- **Experimental scope.** Main-text experiments are small-scale: one synthetic quadratic with nonconvex penalty and two LIBSVM datasets with $n=10$, $s=m=1$; deep-learning tasks are deferred to the appendix. There are no error bars/seeds, limited heterogeneity controls, and no ablations on $s$ or the two required synchronizations; baselines like CE-LSGD/FedRed or stabilized proximal-point variants are not included empirically. 
- **Comparisons.** The paper presents a textual complexity accounting for SABER (App. D) but does not provide a single theorem/table that places the proposed and competing bounds under the same $\{\Delta_1,\delta_s\}$ regime and the same cost model ($C_A,C_R$), making it hard to verify the “best among all existing” statement unambiguously. 
- **Clarity/notation.** The roles of $\Delta_{\max}$ vs. $\Delta_1$ (and when each baseline is analyzed under which) are easy to lose; some constants and definitions (e.g., Assumption 3.5, (E.8) quantities $\sigma_r,\Sigma_r,\chi_r$) are only in the appendix while being central to Cor. 3.7/Theorem 3.8. A concise in-main-text summary table of symbols would help. 
- **Claim on Partial Participation:** The central motivation is to avoid the full participation requirement of SARAH. However, the theoretical analysis for CGM-RG-SAGA (Theorem 3.8, Corollary F.5) requires a heavy initialization phase: $g^0=\nabla f(x^0)$, $g^1=\nabla f(x^1)$, and $b_{i,0}=\nabla f_i(x^1)$. This appears to require *at least two* (and possibly three) full-gradient-equivalent synchronization rounds involving all $n$ clients. This is a significantly stronger requirement than standard SAGA (which needs one) and substantially weakens the practical claim of being a true "partial participation" method that avoids full synchronization.
- **Disconnect Between Theory and Experiments:** The main theoretical communication benefit (Corollary 3.9) hinges on the assumption that the random oracle is much cheaper than the arbitrary one ($C_R \ll C_A$). However, all experiments explicitly state (Sec 4, line 424) that they set $C_A = C_R = 1$. This experimental setup completely neutralizes the key theoretical advantage of the method. If $C_A = C_R$, the SAGA variant's communication complexity is no better than the SVRG variant (Corollary F.7), which the authors claim is "worse". This is a major gap in the paper's validation.
- **Weak Empirical Validation (Deep Learning):** In the deep learning experiments (EMNIST, CIFAR-10), the performance of the proposed CGM-RG-SAGA is only marginally better than baselines. For example, in Table I.2, CGM-RG-SAGA achieves 86.2% accuracy, while SCAFFOLD achieves 85.9% and CGM-RG-SVRG achieves 86.0%. These minor gains do not provide strong empirical evidence for the superiority of the method, especially given its increased complexity and number of hyperparameters ($\lambda, p, \beta, q$ in Table I.1).
- **Experimental Comparisons:** In the non-convex quadratic experiment (Figure 1), several key baselines, particularly SABER-partial, are plotted for a very small number of communication steps before terminating. This makes a direct comparison difficult and may present an overly favorable view of the proposed methods, which are run for a much larger budget.
- **Method Complexity:** The final method is a complex composition of three distinct algorithms. This introduces numerous hyperparameters (e.g., $\lambda$ from CGM, $\beta$ from RG, $p$ and $\eta$ from the local solver) that require careful tuning, as evidenced by Tables I.1 and I.3.

### Questions
1. **Necessity of two full synchronizations.** Can you prove Theorem 3.8 (or a slightly weaker version) without the initial $\Theta(ns)$ ACO calls—e.g., with a statistically warm-started memory (random mini-batch initialization) and one composite step only? If not, what failure mode appears? 
2. **Tightness of the $ns$ improvement.** Is the $ns$ factor in Cor. 3.7 minimax-optimal under $\delta$-SOD, or could a SARAH/PAGE-style recursion match it without SAGA’s memory (thus avoiding ACO entirely)? A direct theorem-level comparison against SABER under identical $(\Delta_1,\delta_s)$ would clarify. 
3. **Regime demarcation.** The text highlights an “interesting regime” $m\ge \sqrt n$, $\delta\lesssim \Delta_1$ where oracle complexity matches full-gradient CGM (while sampling only $s=\sqrt n$). Can you characterize the largest regime (in $m,\Delta_1,\delta$) where your method strictly dominates SABER/SCAFFOLD in both $C_A$ and $C_R$ usage? A plotted phase diagram would help.
4. **Local solver accuracy.** What concrete inner solver and stopping rule (measurable on devices) guarantee the subproblem accuracy term in Theorem 3.8, and how many local steps (or gradient calls) does this imply as a function of $\lambda,\Delta_1,\delta_s$? Any mismatch with the empirical inner tolerances?   
5. **Empirics under stronger heterogeneity.** Can you include experiments with larger $n$, varying $s$ (e.g., $s\in\{\sqrt n,\,m\}$), and tasks where $\delta\approx L$ to test the boundary where similarity gives little advantage? Also add multiple seeds and error bars.
6.  The primary theoretical communication gain depends on $C_R \ll C_A$, but the experiments (line 424) use $C_R = C_A = 1$. This setting does not validate the theoretical claim. Can you provide experiments in a simulated environment where $C_A$ is set to be significantly larger than $C_R$ (e.g., $C_A = C_R \cdot n/m$) to demonstrate the practical benefit of your $C_R$-dependent bound?
7.  Could you clarify the initialization requirements in Theorem 3.8? It appears to require computing $g^0=\nabla f(x^0)$, $g^1=\nabla f(x^1)$, and $b_{i,0}=\nabla f_i(x^1)$. How many full-client synchronization rounds does this entail? How does this heavy initialization align with the core motivation of avoiding the full-participation burden of SARAH?
8.  The paper argues that SAG (used by SCAFFOLD) cannot exploit functional similarity $\delta$ (Appendix H). However, in the deep learning experiments (Tables I.2, I.4), SCAFFOLD performs almost identically to your $\delta$-aware method. How do you reconcile this theoretical limitation of SAG with its strong empirical performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel framework for non-convex federated optimization called Similarity-Aware Non-Convex Federated Optimization. The primary goal is to reduce communication costs, a major bottleneck in FL, especially in settings with partial and intermittent client participation. The authors focus on bridging the gap between SARAH-type and SAGA-type methods. The paper demonstrates for the first time that SAGA-type estimators can also leverage functional similarity to achieve state-of-the-art communication complexity, rivaling SARAH-type methods, without the need for frequent full client synchronization. The proposed framework CGM-RG unifies these ideas and provides a practical algorithm that outperforms existing methods in both communication and local computation complexity under certain conditions.

### Strengths
The work demonstrates that SAGA-type methods can exploit second-order similarity, which is a novel theoretical contribution. The work designs a method that does not require frequent full synchronization, relevant to practical applications. The paper provides a thorough theoretical analysis, including a new framework for comparing algorithm complexities. The theoretical claims are well-supported by numerical experiments on both synthetic and real-world datasets (LIBSVM, EMNIST, CIFAR-10). The proposed CGM-RG-SAGA method outperforms a range of strong baselines, including FedAvg, SCAFFOLD, and SABER. The metric is communication complexity.

### Weaknesses
The authors mention scenarios where the second-order dissimilarity constant $\delta$ is smaller than $L_{\max}.$ However, in many practical FL settings with highly distinct data distributions client data can be dissimilar.

The analysis does not extend to the stochastic setting, which limits its direct applicability to problems involving online learning at the client level.

The proposed method CGM-RG-SAGA is a combination of well known techniques such as Composite Gradient Method, SAGA variance reduction estimator, recursive gradient updates related to SARAH. The contribution is primarily in the novel analysis of a clever combination of existing tools, rather than in the invention of a new algorithmic principle.

The experiments are conducted on datasets with a relatively small number of clients. This scale is not representative of large-scale cross-device FL. The experiments use standard, relatively small models.

The paper lacks an ablation study to isolate the impact of each component of CGM-RG framework.

The paper is technically dense and can be complicated for a broader audience.

### Questions
I have several questions and suggestions.

How the method performance degrades as $\delta$ increases? Is it possible to characterize it theoretically and empirically?

How much of the performance gain comes from the recursive gradient component versus the base SAGA estimator with the new analysis?

How sensitive is the algorithm to its hyperparameters $\lambda, \beta$ and local step parameters? What is the intuition behind setting these parameters?

The SAGA estimator requires an initial full gradient computation to initialize the gradient table querying all clients. You briefly mention an alternative in Remark E.12 initializing only on a sampled subset. What is the theoretical and practical impact of this inexact initialization?

You state that the memory cost of SAGA is moderate. Can you provide a more concrete quantification of the memory overhead? How does it compare to other stateful methods like SCAFFOLD?

Table 1 is extremely informative but also very dense and contains a lot of notation.

While the components are described, a single, clear algorithm box for the final proposed method would be very helpful.

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
4

### Summary
Second-order similarity is an assumption under which we can reduce the amount of communication needed in distributed optimization problems by leveraging more local work, typically done through (approximate) proximal point operations on clients and using some sort of variance reduction. Prior work mostly relied on SVRG-style variance reduction, as it lends itself well to leveraging second-order similarity, see e.g. [1]. This paper introduces a way to use SAGA-style estimators for variance-reduction, which allows us to do away with having to compute full gradients  after the first iteration (which we need to do for SVRG-style variance reduction) at the cost of additional memory. In fact, the paper introduces a general framework that subsumes both SAGA-style, SVRG-style, and even SARAH-style (recursive)  estimators; The latter kind is the best for nonconvex optimization. Combining this framework with the SARAH estimators gives the best-known convergence rates under second-order similarity compared to prior work.

### Strengths
- The proofs incorporating the SAGA estimator are quite nice, the notion of variance the authors define in the line right under (E.9) is different from the notion of variance used in the SVRG estimator (which is also used by prior work). This is quite novel and I believe to be a very useful insight.
- The paper explicitly includes details on how to solve the local problems. Even though the stopping criterion requires knowing many problem parameters, this is still appreciated as much prior work ignores this aspect or assumes perfectly solved inner problems.

Overall, I believe the paper makes a strong theoretical contribution but the experimental evaluation is limited (which is fine, it is not the main point) and the presentation could use significantly better work. I lean towards acceptance and can improve my score if my concerns (below) are addressed.

### Weaknesses
- It is quite surprising that the algorithm just chooses the first function at each timestep to calculate the prox with respect to it. This seems suboptimal (e.g. what if that first function is just extra dissimilar? The avg similarity can be low and this one function could just be an outlier).
- The accuracy of CIFAR10 in the experiments section is far too low. 77%? A three minute run with SGD can reach 96% (see https://github.com/KellerJordan/cifar10-airbench). I suggest the baselines should be better tuned with grid searches and using the latest techniques. This is not expensive.
- The main advantage of this work is obtained under the regime where the cost constants \( C_A, C_R \) differ significantly. The paper makes the claim that "In large-scale federated systems, ACO is more expensive than RCO" but this is not really substantiated by any numbers or analysis. Furthermore, the difference between the ACO and the RCO reminds me of the work (Arjevani et al., On the Complexity of Minimizing Convex Finite Sums Without Using the Indices of the Individual Functions, 2020). I think using their formulation can automatically lead to an algorithm where only the RCO is needed. I believe this could actually perform better than the rate you give depending on the relation of \( C_{R} \) to \( C_A \)? (In fact, their oracle is even weaker than the RCO, since it does not even give them the indices of the functions). They do not use second-order similarity but it seems possible to modify their algorithm to use that, as it's just SVRG based?
- The paper is difficult to read, in part because many important details on the communication model are relegated to the appendix. It would be better to defer more of the technical discussions (and even Table 1) to the appendix and instead focus on introducing this background material.

### Questions
- Please address my concerns in the weaknesses section.
- The use of Big-O notation in lines 235-237 is very handwavy. You actually need it to be bounded by a constant smaller than 1. If the variance were bounded by, say, 100 times the left hand side, the guarantee becomes vacuous. Can you fix this?
- Why do you call Definition 2.2 "Second-order Dissimilarity" when prior work calls it second-order similarity? We call \( L \) the smoothness constant even though the higher L is, the less smooth the function is. We should keep the same convention for \( \delta \) (since it is related to \( L \)).
- Is there any novelty in Section E.2.3 compared to prior work? It seems SVRG-style estimators have already been studied well.

### Soundness
3

### Presentation
1

### Contribution
3

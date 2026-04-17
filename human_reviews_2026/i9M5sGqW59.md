# Instance-Optimal Best-Arm Identification in Non-Stationary Linear Bandits

- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
We investigate the fixed-budget best-arm identification (BAI) problem in non-stationary linear bandits.  Concretely, given a fixed budget $T\in \mathbb{N}$, finite arm set $\mathcal{X} \in \mathbb{R}^d$, and a potentially adversarial sequence of unknown parameters $\lbrace \theta_t\rbrace_{t=1}^{T}$ (hence non-stationary), a learner aims to identify the arm with the largest average reward $x_* = \arg\max_{x \in \mathcal{X}} x^\top\sum_{t=1}^T \theta_t$ with high probability. It is well-known that uniformly sampling arms from the G-optimal design yields a minimax-optimal error probability of order $\exp\left(-T\Delta_{(1)}^2 / d \right)$, where $\Delta_{(1)}$ denotes the average reward gap between the first and second best arms. However, this can be suboptimal in certain arm sets as it only aims to minimize the worst-case variance of each arm's estimated reward. To emphasize this, we establish an arm-set-dependent lower bound and show that uniformly sampling from the G-optimal design fails to achieve it. Motivated by this gap, we propose the *Adjacent-optimal design*, a specialization of the $\mathcal{XY}$-optimal design tailored to the non-stationary setting, and develop the **Adjacent-BAI** algorithm. We prove that the error probability of **Adjacent-BAI** matches our lower bound, establishing the optimality of **Adjacent-BAI**, and highlighting the gap between the G-optimal and Adjacent-optimal designs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an "instance-dependent" lower bound for fixed-budget BAI under non-stationary linear bandits, introduces an "adjacent-optimal" design and an adjacent-BAI algorithm (based on $\mathcal{X}\mathcal{Y}$-optimal design), and claims instance-optimality by matching the lower bound (up to constants). The core technical ingredients are: (i) a two-phase hard-instance construction for the lower bound; (ii) a relaxation to adjacent extreme-point pairs to obtain a closed-form; and (iii) a static allocation rounded from the Adjacent-optimal design, executed in random order, with a martingale analysis to control the least-squares estimator.

### Strengths
The paper’s main contribution is the observation that restricting the complexity measure to adjacent extreme-point pairs yields a closed-form objective that depends only on pairwise differences along those edges. This edge-restricted formulation (to my knowledge) is novel and it directly drives both the algorithm— by allocating on the 1-skeleton contrasted to the canonical $\mathcal{X}\mathcal{Y}$-allocation — and the lower-bound analysis, which replaces the set of pair-wise extreme points with an adjacent-edge-based surrogate.

### Weaknesses
1. “First instance-dependent lower bound” claim is overstated.
Theorem 1, as written, is a Le Cam/Bayes (mixture) lower bound, not an instance-dependent (per-instance) bound. For any prior $\rho$ on bandits $\nu$, 

$\inf_{\mathfrak{A}} \sup_{\nu}\mathbb{P}_{\nu}^{\mathfrak{A}}(J\neq 1)=\inf\_{\mathfrak{A}}\sup\_{\rho}\mathbb{E}\_{\nu\sim\rho}[\mathbb{P}\_{\nu}^{\mathfrak{A}}(J\neq 1)]\geq\inf\_{\mathfrak{A}}\mathbb{E}\_{\nu\sim\rho}[\mathbb{P}\_{\nu}^{\mathfrak{A}}(J\neq 1)]$.

If $\rho$ is supported on {$\nu,\nu^\prime$}, we have:

$\inf_{\mathfrak{A}} \mathbb{E}\_{\nu\sim\mu}[\mathbb{P}\_{\nu}^{\mathfrak{A}}(J\neq 1)]\leq \inf\_{\mathfrak{A}} \max  \{  \mathbb{P}\_{\nu}^{\mathfrak{A}}(J\neq 1), \mathbb{P}\_{\nu^\prime}^{\mathfrak{A}}(J\neq 1)  \}$. i.e., the Bayes (two-point) risk is not larger than the two-point max risk. By contrast, an instance-dependent bound fixes an instance $\nu$ and lower bounds
$\inf\_{\mathfrak{A}}\mathbb{P}\_{\nu}^{\mathfrak{A}}(J\neq 1)$ 
in terms of a complexity $H(\nu)$, with no outer $\sup_\nu$ or $\mathbb{E}_{\nu\sim\rho}$. As it stands, Theorem 1 lower-bounds a Bayes/minimax quantity; it does not provide a per-instance lower bound. If the authors intend an instance-dependent statement, please either (a) restate Theorem 1 explicitly as a per-instance result (fixing $\nu$ and quantifying over its alternatives) or (b) revise the claim to “Bayes/minimax lower bound.”

2. Step 2 (enumerating adjacent pairs) lacks a complexity/procedure. The algorithm requires all adjacent pairs of $\mathcal{X}$, i.e., edges of conv$(\mathcal{X})$. For simple structured sets (simplex, hypercube) this has closed-form, but for general $\mathcal{X}$, this entails:

- computing the extreme points $\mathcal{V}_{\mathcal{X}}$ (vertex enumeration), and,
- extracting the edge graph.

Please (i) give a concrete routine (e.g., convex-hull + edge graph extraction), and (ii) state its computational complexity. In general, I think, Step 2 is not computationally trivial and should be acknowledged as such. If I am missing something, I would be happy to stand corrected.

### Questions
1. Should it be $\in$ instead of $=$ in line 123?
2. The authors state: "after computing the Adjacent-optimal design $\lambda_*$, we do not sample directly from it, as this leads to heavy-tailed behavior". Why is that the case? With bounded features and sub-Gaussian noise, the empirical design under i.i.d. sampling from $\lambda_*$ should concentrate via matrix Bernstein—am I missing something?
3. How large is $r(\epsilon)$? 
4. In line 194, it should be $\frac{4}{3}d$.
5. Please fix the grammar in the first sentence of Lemma 2.
6. The flow of writing could be improved. Many Lemmas referred in the main paper are not explained, and hence, requires switching between the main paper and appendix for the statements. Please note that there is more space to accommodate these in the main paper — the current write-up is shy of 9 pages by at least half a page.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper investigates the problem of best arm identification in linear bandits with a fixed budget and non-stationary parameters. The set of arms is finite, and the parameter defining the linear reward may evolve arbitrarily over time. The objective is to find the arm that performs best for the parameter averaged over the $T$ rounds. The authors derive an instance-specific lower bound on the error probability. They also propose a new algorithm, referred to as Adjacent-BAI, whose error probability resembles that of the lower bound. Numerically, the algorithm appears to perform better than an algorithm based on the G-optimal design.

### Strengths
•	The analysis is sound and well conducted;
•	The lower bound and the resulting “adjacent” design are new;
•	Compared to G-BAI, the proposed design performs better both analytically (see Proposition 1) and numerically.

### Weaknesses
- The authors do not justify scenarios where learning the best average arm would make sense. What is the intended application scenario? Other non-stationary bandit formulations typically consider cases where the evolution is controlled (i.e., not many changes over the time horizon) and where the decision maker should adapt the arm choice over time—such settings seem more realistic.

- Instance-optimality. The proposed algorithm is not instance-optimal, mainly because the constant in the exponential differs, leading to error probabilities that can diverge substantially. The constant $c$ in Theorem 2 is 1/322, which yields an error probability that is quite far from the lower bound.

- The rounding procedure and the function r(ϵ)are not described in the paper. What does this function look like?

- Numerical experiments are on toy examples only.

### Questions
•	What are the practical applications of your problem? Can you identify real-world data where you could illustrate the proposed approach?

### Soundness
3

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
This paper studies the best-arm identification (BAI) problem in a non-stationary linear bandit setting. It introduces a new instance-dependent lower bound that depends on the gaps between the best arms, and proposes the Adjacent-BAI algorithm, which is claimed to achieve this bound.

The paper should be rejected due to (1) unclear positioning with respect to related work, and (2) weak empirical validation.

**Main Argument**
The paper proposes an algorithm for non-stationary BAI but fails to convincingly demonstrate its advantage over existing methods, both theoretically and empirically. The relationship to prior works such as P1-RAGE (Xiong et al., 2024) is not clearly discussed. The experimental evaluation is limited to seemingly benign environments and does not substantiate the claimed optimality.

### Strengths
- The theoretical setup is well-motivated, and the paper identifies an important gap in extending instance-optimal BAI to non-stationary environments.

### Weaknesses
- The paper inconsistently refers to the “pure non-stationary” and “stationary” settings. While it claims to focus on the pure non-stationary case (line 164), it later asserts that Adjacent-BAI is optimal in the stationary setting as well (line 357). The relationship between these claims is unclear, and the contribution is not well positioned relative to P1-RAGE, which explicitly handles both stationary and non-stationary environments.
- The experimental section lacks motivation for the chosen environment designs. The tested environments appear benign, as the parameter vector $\theta$ oscillates around the mean $\theta$ rather than being adversarial, chosen. Moreover, no comparison is provided against standard stationary algorithms or against P1-RAGE, the most relevant benchmark given its ability to address both stationary and non-stationary cases. Without such comparisons, the empirical claims of optimality remain unconvincing.

- Several implementation details are insufficiently explained, including the “efficient rounding” step (line 287) and the introduction of “necessary randomness” (line 294). The rationale and theoretical implications of these components are not clear.

### Questions
- How does the proposed algorithm perform empirically against stationary baselines and against P1-RAGE?
- How does the proposed method compare theoretically to P1-RAGE? In particular, does the derived lower bound or optimality claim extend to the mixed (stationary/non-stationary) setting considered by P1-RAGE, or are the guarantees specific to the pure non-stationary regime?
- How does defining the gap in terms of the neighborhood (rather than the global best arm) influence both theoretical guarantees and empirical performance?

### Soundness
1

### Presentation
3

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
This paper focused on the best arm identification problem in nonstationary/adversarial bandit setting. It proposed the adjacent-BAI algorithm and derived an upper bound on its failure probability. It also proposed a lower bound to show the superiority of the adjacent-BAI algorithm. Some numerical experiments are presented.

### Strengths
1. The author(s) discussed several related works.
3. The paper is overall with a good structure.

### Weaknesses
My key concern lies on the the soundness of Theorem 2:
1. Lemma 11: This is an important lemma to show Theorem 2 which assumes that $\sum_{t=1}^T \theta_t=0$. I wonder why this is a reasonable assumption. 
    1. The assumption should not be hidden in the appendix. 
    2. And if this assumption is troublesome, the superiority of the proposed algorithm is not convincing.

Other conerns/suggestions are also listed as below:
1. Abstract: the author(s) first discussed the non-stationary bandit setting and then claimed that G-optimal is known to be optimal. To my understanding, G-optimal algorithm is known to be optimal in a stationary bandit setting.
1. LINE 31: 'The linear bandit problem is a classical framework for addressing the exploration-exploitation dilemma in settings where the actions are embedded in a finite-dimensional space. ' To my understanding, the tradeoff between exploration and exploitation is important in bandits but definitely not the target.
1. Algorithm 1 and Theorem 2: the impact of $\epsilon$ is clear in neither Algorithm 1 nor Theorem 2. More clarifications are required. Besides, the value of $c$ should be clearly shown in the statement of Algorithm 1. 
1. Numerical simulations: I failed to find the value of $\epsilon$ and more numerical experiments on different instances and related algoirthms are appreciated.
1. How does the proposed algorithm perform comparing to all existing algorithms numerically and analytically? Comparison on analytical bounds in a table is appreciated.

### Questions
See the **Weaknesses** section.

### Soundness
1

### Presentation
2

### Contribution
1

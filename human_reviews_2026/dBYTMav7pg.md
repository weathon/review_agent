# VATT-EG: Vanishingly-Anchored Two-Timescale Extragradient

- Decision: Reject
- Scores: 2, 4, 2, 4

## Abstract
We resolve the open problem of whether a first-order algorithm can converge only to local
minimax equilibria in smooth nonconvex-nonconcave zero-sum games. Building on the two-
timescale extragradient method, we develop VATT-EG (Vanishingly-Anchored Two-Timescale
Extragradient), which introduces vanishing Tikhonov-style anchors to eliminate neutral modes
without biasing the stationary set. We show that under the standard calmness (bounded-
ratio) assumption: (i) a point is asymptotically stable for VATT-EG if and only if it is a calm
local minimax equilibrium; (ii) every non-minimax stationary point is a hyperbolic repeller
with a quantitative spectral margin; and lastly (iii) from almost every initialization, iterates
converge with probability one to a calm local minimax point. The proposed analysis depends
on a discrete-time extragradient expansion with constants; a Lyapunov metric built from the
restricted Schur complement, and a Robbins-Siegmund argument combined with a measure-zero
stable-manifold theorem. Experiments on canonical two-dimensional games confirm the neutral-
mode damping of vanishing anchors, while a delayed-corrector stress test shows that VATT-EG
stabilizes regimes where GDA and standard extragradient diverge or spiral out. Finally, in
a practical adversarial debiasing example (namely Colored-MNIST), VATT-EG achieves digit
accuracy comparable to baselines and consistently reducing adversary accuracy, yielding more
invariant features. Together, these results close the first-order selection gap and establish VATT-
EG as both a theoretically exact and practically robust algorithm for nonconvex-nonconcave
minimax optimization.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work focused on algorithms for finding local minimax solutions for smooth zero-sum two-player games. Its main result is that for a vanishingly regularized version of the two-timescale extragradient algorithm, its asymptotically stable solutions consist of precisely calm local minimax solutions. Experiments on toy examples and an adversarial debiasing problem on MNIST were conducted.

### Strengths
- natural modification of the extragradient algorithm to eliminate certain stationary points

- almost sure convergence analysis that combine a number of proof techniques

### Weaknesses
- the problem is not sufficiently motivated and the paper is poorly written (full of technicalities whose significance are hard to appreciate). It is not clear why the derived theoretical results are significant and how they will affect practice. Why are calm local minimax solutions relevant? What advantage do they have? On what applications do they really make a difference? The writing also fails to give an informed and ideally insightful discussion of the relevant literature and prior works. 

- poorly designed experiments leaving a significant gap between theory and experiments: the choices of the various parameters in the experiments do not satisfy Assumption 4.2. What is the point of running experiments that do not adhere to the assumptions? (Or how relevant is the theory if they do not reflect the practice?)

- unsubstantiated claims: Figure 1 and Figure 3 hardly showed any difference among the different algorithms and yet the authors did not shy away from claiming (slight) advantage of VATT-EG. It is not clear how Figure 5 supports the theory of this work (accuracy was never part of the goal and verification of convergence was lacking). 

- missing obvious baselines: GDA could easily adopt (vanishing) regularization or even two-timescale. GDA with ergodic averaging is also widely known to stabilize training (even for convex-concave games). The last experiment on color debiasing should at least compare to these obvious variants. 

- results are only asymptotic and the compactness in Assumption 4.1 may be too strong. If the trajectory is assumed to be contained in a compact set, then it already cannot diverge (although could cycle) and repulsion is trivial.

### Questions
My overall impression is that this work (as is currently presented) is too narrowly focused, with unclear significance and relevance. 

Below are some other comments that hopefully may be useful.

Adding quadratic regularization to smooth zero-sum two-player games is a well-known idea. In fact, it is well-known that under strong convexity-concavity, the iterates of GDA converge properly. By appropriately decreasing regularization over iteration, it is possible for GDA to converge too. There is no discussion of these points and the current writing gives the misleading impression that vanishing regularization is somehow "novel." Same goes for two-timescale. 

Line 186: shouldn't T(z+e) = T(z) + DT(z) e + .. instead of z + DT(z)e? how does this affect the proof? 

Lemma 5.4: from equation (6) we know the spectrum is strictly larger than 1 but the gap diminishes as k tends to infinity. Is it correct to conclude that the trajectory will have to diverge? It is not possible to discern as the authors never defined what they mean by a hyperbolic repeller in Theorem 5.5.

Line 239: why vanishing anchors do not alter the stationary set? I think you need the regularization to vanish sufficiently fast, but how fast?

Line 260: the constant choice of tau clearly violates Assumption 4.2 and invalidates the point of the experiments. 

Figure 2: what is the scale of the y-axis on the middle figure? what does 1e-5+1.0008 mean?

Line 317-318: how do the experiments support the theoretical claims?

Line 361: "We use Adam, cosine decay, and global gradient clipping?" Do you run VATT-EG or Adam? Is VATT-EG also delayed? This experiment section is so roughly written that it is hard to understand the details of the experiment.

### Soundness
1

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
2

### Summary
This paper introduces VATT-EG, a first-order optimization algorithm for smooth nonconvex–nonconcave zero-sum games.
The key contribution is that VATT-EG is the first purely first-order method whose attractors coincide exactly with the set of calm local minimax points, addressing a long-standing open problem in game-theoretic optimization.

The main idea is to augment standard extragradient dynamics with vanishing Tikhonov-style anchors, which damp neutral (non-contracting) modes without altering the stationary point set. The authors provide detailed theoretical analysis—based on discrete-time Jacobian expansions, Lyapunov constructions, and calmness assumptions—showing that all non-minimax stationary points become strict repellers.
Empirical results on toy games and an adversarial debiasing task (Colored-MNIST) illustrate the claimed stabilization effect and practical viability.

### Strengths
**1. Strong and rigorous theoretical contribution**:
The paper presents a technically sound and self-contained theoretical analysis, supported by explicit derivations. The work convincingly addresses a nontrivial open problem in first-order game dynamics.

**2. Algorithmic succinct**:
The optimization algorithm is succinct yet powerful, introduces vanishing anchors, offering an interpretable way to handle neutral directions in game dynamics without resorting to second-order methods, making it relevant to scalable deep learning.

### Weaknesses
**1. Limited Experimental Breadth and Depth**: The empirical results, while carefully constructed, are restricted to synthetic low-dimensional games and a single moderate-scale practical task (Colored-MNIST). No experiments address scalability on standard large-scale ML benchmarks, deep adversarial training (e.g., GANs), or tasks with higher-dimensional or less controlled geometries. This restricts the claims of “practical robustness”—there is little evidence that VATT-EG would consistently outperform GDA/EG in genuinely large-scale modern settings.

**2. Incomplete mention of Related Work:** In the related work part, the author only mentions 4 related papers, which cannot help readers to get familiar with this field.

### Questions
**Q1:** Though I like the clear and succinct algorithm, which is easy to follow. I find the paper meets some layout problems (i.e., too less references and the main text only contains 7 pages).  As I am not deeply experienced in this research area, I will finalize my rating after reviewing other reviewers’ comments and the authors’ responses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work proposes a new first-order method for minimax optimization, called *VATT-EG (vanishingly-anchored two-timescale extragradient)*, which improves upon existing algorithms that fail to converge to certain local minimax equilibria. 
By introducing the trick of "vanishing anchors", VATT-EG overcomes the instability that arises when the restricted/generalized Schur complement has zero eigenvalues, thereby ensuring convergence only to calm local minimax points.

### Strengths
Minimax optimization remains a challenging problem, and thus, a method offering improved convergence guarantees represents a meaningful contribution to the community, given its wide range of applications.
The overall structure follows the standard research-paper format, and the claimed goal of improving convergence guarantees for first-order minimax methods is, in principle, a worthwhile direction of study.

### Weaknesses
Despite the relevance of the general topic, the paper exhibits significant issues. The theoretical claims appear to contain flaws, and the presented experiments do not meaningfully support or validate the proposed method. In addition, several references are cited incorrectly, including mismatched author names, which raises serious concerns about the paper’s authenticity and scientific validity. Overall, the submission seems to fail to meet the standards of scholarly work expected at ICLR.

### Questions
1. There is something in equation (13) I cannot verify. From Assumption 4.2 asserting $\gamma_k, \beta_k \geq c_2 \eta_k$, it seems like $c_2 P$ in equation (13) should actually be $c_2 \eta_k P$. Consequently, the bound on $\mathrm{Sym}(\Delta_k)$ along the $y_0$ subspace would be $O(\eta_k^2)$, not $O(\eta_k)$ as claimed. This would further invalidate the energy decrement inequality (line 676), as the $O(\eta_k)$ term in the right hand side cannot have full $\|\|e\|\|_P$, but a truncated one (not having the part corresponding to $y_0$), potentially even affecting the stated convergence guarantees in Theorem 5.5. Can you check this?
2. Can the authors provide more details on the Lowener inequality regarding $\mathrm{Sym}(P \Lambda_{\tau_k} J)$ around line 662? It is currently stated without proof, but I don't think it is trivial enough, and I can't see why it should hold. 
3. The experiments appear to be implemented or interpreted incorrectly. For instance, the extragradient (EG) method, regardless of whether it is used in a two-timescale or single-timescale form, should converge on simple bilinear games. However, Figure 1 shows that the final gradient norm remains large, suggesting that even basic convergence behavior is not achieved. This discrepancy raises doubts about the correctness of the experimental setup or the evaluation procedure. Could the authors recheck their implementation and confirm whether the results in Figure 1 are consistent with known behavior of EG on bilinear problems?
4. It should be straightforward to construct synthetic quadratic minimax games that possess local minimax equilibria where VATT-EG is theoretically guaranteed to converge, while standard methods such as no-anchor TT-EG or GDA lack such guarantees. Experiments on these controlled examples would more directly demonstrate the claimed advantages of VATT-EG, and would provide a clearer empirical validation than the current toy tasks. Could the authors explain why such experiments were not included, and whether they plan to add them in a future revision?

### Soundness
2

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
The goal of this paper is to resolve the open problem posed by Chae et al. (2023): finding a gradient-based method for minimax optimization whose only attractors are (calm) local minimax (LM) points. In particular, since certain (calm) local minimax points remain neutrally stable for two-timescale extragradient (TT-EG), this paper proposed a vanishing anchor approach that modifies TT-EG to make such equilibria asymptotically stable.

### Strengths
Minimax optimization is notoriously challenging, so a clear theoretical framework that improves its solvability, as this paper provides, is both important and interesting. Furthermore, demonstrating that this can be achieved through a weight-decay-like mechanism such as vanishing anchoring adds practical value and insight.

### Weaknesses
- Incorrect motivation?: In line 54, the authors claim that the remaining challenge is that certain calm local minimax points are neutrally stable under TT-EG, which their method resolves. However, I was not able to find this statement in Chae et al. (2023). After reading the main related papers [1,2] by the same authors, I found that the primary issue is not neutral stability but rather some calm local minimax points act as hyperbolic repeller. 

[1] Chae, Kim, Kim, Two-timescale extragradient for finding local minimax points, ICLR, 2024.

[2] Kim, Kim, Double-step alternating extragradient with increasing timescale separation for finding local minimax points: provable improvements, ICML, 2024

- Second-order necessary condition?: In Chae et al. (2023), I found that the condition in line 142 is not exactly the necessary condition for a calm local minimax point. It becomes necessary only after an additioinal modification in the $y$ neighborhood (see Chae et al. (2023)). Moreover, the paper seems to use this condition interchangeably with the definition of a clam local minimax point, which is inaccurate. I recommend clarifying this distinction carefully.

For these two reasons, I suggest that the authors clarify their contributions and ensure consistency with prior definitions and motivations.

### Questions
* Lines 74-88: This paragraph is quite difficult to follow. I suggest that the authors revise it to make it more accessible to readers.
* Line 120: Is it correct that (Wang et al. 2019) has the only-to-LM selection property? My understanding is that their result assumes the invertibility of $B$.
* Line 143: The matrix $U$ is never used.
* Line 165: Although one can infer this from the parameter conditions, I recommend explicitly stating that $\eta_k\to0$.
* Line 261: What does $\tau$ denote here?
* Figure 1: Why do all methods exhibit large gradient norms at the final iteration? Shouldn't your algorithm at least converge to a stationary point? Overall, I found the experimental section somewhat confusing. It would be more informative to show that your proposed VATT-EG is attracted to a calm local minimax point (and eventually converges to it), whereas TT-EG remains neutrally stable. At present, the experiments emphasize relative performance rather than clearly demonstrating the claimed stability improvement. 
* Line 306: Could you explain why VATT-EG appears even more stable than TT-EG in the one-step delay case? Are you using a diminishing step size for TT-EG, as you do for VATT-EG? Some explanation of why VATT-EG remains robust under this more realistic setting would be helpful.

### Soundness
2

### Presentation
2

### Contribution
2

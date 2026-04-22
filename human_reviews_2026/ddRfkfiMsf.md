# Improved Last-iterate Convergence Properties for the FLBR Dynamics

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
The recent years have seen a surge of interest in algorithms with last-iterate convergence for 2-player games, motivated in part by applications in machine learning. 
Driven by this, we revisit a variant of Multiplicative Weights Update (MWU), defined recently by Fasoulakis et al. (2022), and denoted as Forward Looking Best Response MWU (FLBR-MWU). These dynamics are based on the approach of extra gradient methods, with the tweak of using a different learning rate in the intermediate step. So far, it has been proved that this algorithm attains asymptotic convergence but no explicit rate has been known. We answer the open question from Fasoulakis et al. by establishing a geometric convergence rate for the duality gap. In particular, we first show such a rate, of the form $O(c^t)$, till we reach an approximate Nash equilibrium, where $c<1$ is independent of the game parameters. We then prove that from that point onwards, the duality gap keeps getting decreased with a geometric rate, albeit with a dependence on the maximum eigenvalue of the Jacobian matrix. Finally, we complement our theoretical analysis with an experimental comparison to OGDA, which ranks among the best last-iterate methods for solving 0-sum games. Although in practice it does not generally outperform OGDA, it is often comparable, with a similar average performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper studies the FLBR-MWU dynamics, which is an extragradient version of OMWU, but where the intermediate stepsize is large. The authors show new convergence results for these dynamics, giving a concrete rate for the last-iterate performance.

### Strengths
- The OMWU dynamics are an important learning algorithm, and the extragradient variant is of interest as well
- Having a concrete rate on the last iterate is of interest

### Weaknesses
1. A missing interpretation in the paper is the following: if we are choosing "large $\xi$" then we are turning the algorithm into a well-known algorithm. In particular, for very large $\xi$, we get that each player is simply best responding to the other player. At that point, you have decoupled the dynamics, and we are simply looking at the intermediate step as a subgradient calculation. At that point, you can instead describe FLBR as follows: we are running subgradient descent, but instead of using projected gradient descent, we use mirror descent with an entropy regularizer. This has implications e.g. for the forgetfulness question. More generally, this interpretation suggests that for large $\xi$, this is not really a "dynamics," it is simply running entropy-based subgradient descent on each player's problem $\min_x f(x), \max_y g(y)$, where $f,g$ encode the fact that the other player best responds. Since the paper repeatedly appeals to "sufficiently large $\xi$" with no upper bound on how large $\xi$ needs to be, I think it is fair to say that the main interpretation of this algorithm should be as subgradient descent.
2. A second issue with the "large enough $\xi$" occurs in the proofs. In the proof of Theorem 1, you rely on choosing  "large enough $\xi$" such that Corollary 1 holds. However, Corollary 1 is a limit statement where you take a limit on $\xi$  after committing to $x^{t-1},y^{t-1}$. On the other hand, Theorem 1 is a statement about choosing  and then generating the sequence of iterates. Thus, I do not think that Corollary 1 can be invoked the way you are doing, since Corollary 1 is only proved for a fixed pair of iterates, whereas you are now having iterates that are determined by your choice of $\xi. More generally, the proof of Theorem 1 is not sufficiently rigorous or detailed. For one, I think you should provide a complete proof rather than say "following the proof of Fasoulakis et al." If you want to invoke Fasoulaskis et al., you should invoke a precise result proven in their paper, and then build from there, not say "the reader can look at their paper and retrace our left-out steps."
3. On forgetfulness experiments: I am not sure these experiments are all that convincing regarding whether FLBR is forgetful or not. First, I couldn't find a specification of what $\delta$ is in these plots. Was it sufficiently small? In either case, ideally you'd try several smaller and smaller $\delta$ choices, to get a sense of whether you chose a small enough $\delta$. Intuitively, it appears plausible that FLBR is forgetful for large $\xi$, due to the "subgradient descent" interpretation that I described above, but these experiments are far too underspecified for me to feel sure either way.

### Questions
1. Can the magnitude of $\xi$ be quantified? "Large enough" is too vague in my opinion.
2. Is there any sense in which this is not essentially a result for "subgradient MWU" applied to each player's nonsmooth optimization problem?
3. What were the parameters for the forgetfulness experiments?

### Soundness
1

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
3

### Summary
This paper studies the Forward Looking Best Response Dynamics (FLBR) and investigates several properties of the dynamics in two player zero sum games. In particular, the paper explores the last iterate convergence rate (to both approximate and exact NE), showing that a geometric rate dependent on game dimension can be obtained for approximate NE, while the convergence rate to exact (unique) NE depends on the Jacobian of the game. The paper also shows that FLBR is not no-regret, and is forgetful in the sense of Cai et al 2024. Finally, some empirical evidence is shown that FLBR performs comparatively or even better than OGDA, though this seems to be a game-class dependent behavior.

### Strengths
- The paper has a clear structure and the research problems investigated are reasonable. This results in a stronger understanding of the FLBR dynamics in comparison to other 'standard' accelerated methods.
- The use of ideas from information theory in the proof of Theorem 2 is interesting and could be a useful tool moving forward.

### Weaknesses
- The main selling point of FLBR compared to OGDA/OMWU is the geometric rate of convergence to approximate NE, as opposed to a $O(1/\sqrt{t})$ for OGDA for example. However, this is very much dependent on the approximation required of the NE, as once the game dependent geometric rate kicks in, the separation seems to disappear and OGDA seems to practically still perform better (except in 'structured' games). I think this reduces the significance of the results somewhat, as there is no clear reason to implement FLBR unless you know the game is 'structured' in some way. One way to address this would be to theoretically justify some classes of games (e.g. symmetric) for which the game dependent constant is small, thereby explaining the speedup of FLBR compared to OGDA.
- The discussion on forgetfulness is interesting but not convincing to me. The example used in Cai et al 2024 was a degenerate example for the case of OFTRL, but while the paper states that the larger intermediate update avoids forgetfulness, this statement is not substantiated. The Cai paper shows negative (non-forgetful) results, but to prove that FLBR is forgetful would require more work. For example consider the setting of Braverman et al (2018) and Kumar et al (2024), both cited in Cai's paper, who studied a similar property to forgetfulness. Kumar et al proved that OGDA is not exploitable by an adversary, and is thus in some sense 'forgetful'. However this is in the context of no-regret algorithms, which FLBR is not. I believe there is more depth and subtlety here to be explored, and would be very curious to know if the authors have studied this in more detail as it would improve the paper.
- While the paper is quite clear in it's structure, I found the writing to overall be very stylized and occasionally imprecise, which reduces the clarity of exposition. For instance, in the last paragraph of the introduction, the authors describe 'positive' and 'negative' results without clarifying what they are, ending with the vague statement that 'the landscape is overall less clear'. In the experimental section, FLBR is described to perform 'close enough' to OGDA in random gaussian games, and 'FLBR comes on top/OMWU is far away' in RPS games. The imprecise language makes it unclear what the threshold for 'close enough' should be. 

In light of the above, I find the paper is marginally under the threshold of acceptance to ICLR, as the writing needs more polishing and more theoretical justification is needed for FLBR compared to existing methods.

### Questions
- Purely out of curiosity, how does FLBR perform in other game classes which are non-zero-sum? For example, I would expect FLBR to converge quickly in potential games, but does it also outperform OGDA/OMWU in this class of games?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studied the dynamical behaviors of Forward Looking Best Response MWU (FLBR-MWU), which was first proposed by Fasoulakis et al. (2022), in two-player zero-sum games. The main results include:  
* An analysis of the convergence rate of FLBR-MWU. Specifically, the behavior of the algorithm can be divided into two phases: First, it converges to an approximate equilibrium at an inverse exponential rate. Second, by analyzing the Jacobian of the dynamics at equilibrium, the authors show that the dynamics also exhibit an inverse exponential rate when the strategy is sufficiently close to the equilibrium.  
* The authors provide examples to show that FLBR-MWU is indeed not a no-regret algorithm.  
* Experiments on the forgetful and convergence rate behaviors are provided for FLBR-MWU, which indicate that FLBR-MWU is comparable with other popular algorithms such as Optimistic Gradient Descent-Ascent.

### Strengths
* The theoretical results enchance our understanding of the convergence behaviors of FLBR-MWU algorithms.
* Numerical experiments provide intuitions on how the performance of FLBR-MWU compared with several other popular algorithms like Optimistic Gradient Descent-Ascent and Optimistic MWU.

### Weaknesses
* The theoretical results, which state that FLBR-MWU first converges to an approximate equilibrium at an inverse exponential rate depending only on the game dimension (i.e., Theorem 1), represent a slightly modified version of the theorem in (Fasoulakis et al., 2022) in terms of the duality gap. Moreover, the Jacobian-type analysis of dynamics near an equilibrium in Theorem 2 is also well-known in the literature; for example, see Theorem 4 in (Fasoulakis et al., 2022) and Section 3 in (Daskalakis & Panageas, 2018). These types of Jacobian analyses usually provide an exponential convergence rate related to the eigenvalues of the Jacobian at equilibrium. Comparing the results of Theorem 2 with these related works, it is unclear what new information is provided. This makes the theoretical contribution incremental.

* Given that Extra Gradient methods are not no-regret algorithms, and the FLBR-MWU algorithm is very similar to the extragradient method, with the only difference being the use of a different step size in the intermediate step (Mertikopoulos et al., 2018), it is not surprising that FLBR-MWU is also not a no-regret algorithm.

* The discussion of the forgetfulness property of FLBR-MWU (Section 4.2) remains at a rough level, with only one toy $2 \times 2$ example provided to explain the phenomenon. The discussion is very high-level and makes it difficult for readers unfamiliar with the work of (Cai et al., 2024) to grasp the key points. For example, it is stated that *"if a method is not forgetful, the produced strategies can get stuck at almost the same profile over many iterations, which slows down convergence."* However, in Figure 1, it is not explained how the algorithms get stuck at nearly the same profile. I suggest that the authors further clarify this phenomenon by providing a comparison with algorithm trajectories that lack the forgetfulness property.

Reference:

Mertikopolous et al., Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile

Daskalakis & Panageas, The Limit Points of (Optimistic) Gradient Descent in Min-Max Optimization

### Questions
1. Compared with the Jacobian-type analysis appearing in related works such as (Daskalakis & Panageas, 2018; Fasoulakis et al. 2022)), which provides an exponential convergence rate in terms of the eigenvalues of the Jacobian, what new information does Theorem 2 provide? 

The authors claimed 

> *"in order to obtain a rate of convergence, we give a more refined analysis, based on a technique utilized in Nakagawa et al. (2021)"* 
(lines 298–299). 

Could you provide more explanation on how the technique in Nakagawa et al. (2021) can provide new insights into the local convergence behavior of the algorithms that the standard Jacobian-type analysis in (Daskalakis & Panageas, 2018; Fasoulakis et al., 2022) cannot provide?

---

2. Compare with Optimistic MWU and Optimistic GDA (Wei et al., 2021), what is the benifits of FLBR-MWU in terms of convergence rate or other aspects?

In the paper, the authors claimed that 

> *"We view as advantages of our analysis that it yields a simpler and more intuitive proof compared to Wei et al. (2021), and it also establishes fast (non-game-dependent) convergence to an approximate equilibrium before approaching the exact solution"* (lines 85–88). 

Can the authors provide more explanation on this point? Specifically, what are the game-dependent constants that appear in the results of Wei et al. (2021)?

Moreover, in Section 4.2 of the current work, the authors state that Optimistic GDA has a better convergence rate than OMWU, and from the experiments it can be observed that Optimistic GDA performs even better than FLBR-MWU. Thus, I wonder whether the existing convergence rate bounds of Optimistic GDA also suffer from the issue of game-dependent constants? 

---

3. As the authors stated in Section 4.2, Optimistic-GDA has better performance than Optimistic-MWU. I'd like to know whether it is possible that FLBR-GDA could also perform better than the FLBR-MWU algorithm proposed in the current work.

### Soundness
2

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
3

### Summary
his paper investigates the last-iterate convergence properties of an algorithm called Forward Looking Best-Response Multiplicative Weights Update (FLBR-MWU) in two-player bilinear zero-sum games. FLBR-MWU is a variant of the Multiplicative Weights Update (MWU) method inspired by the extragradient idea, characterized by using a larger learning rate ξ in its intermediate prediction step and a smaller learning rate η in the final update step. The authors establish, for the first time, explicit convergence rates for this algorithm:

- Prior to reaching a approximate Nash equilibrium, the duality gap converges at a geometric rate independent of the game matrix.
- If the game admits a unique Nash equilibrium, once sufficiently close to equilibrium, the duality gap continues to converge to zero at a geometric rate dependent on the spectral radius of the Jacobian matrix.

Additionally, the authors prove that FLBR is not a no-regret algorithm and demonstrate experimentally that it performs comparably to or better than Optimistic Gradient Descent Ascent (OGDA) in normal-form games (NFGs). The paper also notes that FLBR exhibits a "forgetfulness" property, which may contribute to accelerated convergence.

### Strengths
1. **Clear theoretical contribution**: The paper provides the first explicit last-iterate geometric convergence rates for FLBR-MWU, resolving an open question posed by Fasoulakis et al. (2022).
2. **Well-structured two-phase analysis**: The convergence process is cleanly divided into two phases—(i) rapid approximation toward an approximate equilibrium with a game-matrix-independent rate, and (ii) refined convergence to the exact equilibrium with a rate dependent on the Jacobian’s spectral radius but offering greater precision.
3. **Elegant and intuitive proof**: Compared to the KL-divergence-based analysis of OMWU in Wei et al. (2021), the current paper’s analysis of the duality gap is more direct and cleverly adapts convergence techniques from the Arimoto–Blahut algorithm in information theory.

### Weaknesses
1. **Incomplete coverage of related work**: While discussing last-iterate convergence methods, the paper focuses exclusively on optimistic/extragradient-type approaches (e.g., OGDA, EG) and overlooks other relevant directions, such as:
   - Regularization-based methods [1, 2];
   - Negative momentum-based methods [3].
     These approaches have demonstrated effectiveness not only in bilinear games but also in more general settings like extensive-form games (EFGs). Including them in the Related Work section would better contextualize the paper’s contribution.
2. **Strong theoretical assumptions**: The geometric convergence in the second phase relies on the assumption of a unique Nash equilibrium. Although the authors note that this condition holds "almost always" in a measure-theoretic sense, it may limit the practical applicability of the result. Nevertheless, I do not believe this significantly diminishes the paper’s overall contribution.
3. **Typo**: For instance, an extraneous comma appears in line 148 (“w.r.t.,”).

[1] Sokota, S., et al.; A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games. 

[2] Liu, M., et al.; The Power of Regularization in Solving Extensive-Form Games.

[3] Fang, Z., et al.; Rapid Learning in Constrained Minimax Games with Negative Momentum.

### Questions
**Generalization capability**: In Appendix D.3, the paper attempts to apply FLBR to non-bilinear convex-concave problems but observes non-convergence. Could the algorithm be adapted—e.g., by incorporating regularization or negative momentum—to achieve convergence in broader classes of min-max problems?

### Soundness
3

### Presentation
3

### Contribution
2

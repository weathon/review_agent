# Monotone Near-Zero-Sum Games: A Generalization of Convex-Concave Minimax

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Zero-sum and non-zero-sum (aka general-sum) games are relevant in a wide range of applications. 
While general non-zero-sum games are computationally hard, researchers focus
on the special class of monotone games for gradient-based algorithms.
However, there is a substantial gap between the gradient complexity of monotone zero-sum and monotone general-sum games. 
Moreover, in many practical scenarios of games the zero-sum assumption needs to be relaxed.
To address these issues, we define a new intermediate class of monotone near-zero-sum games that contains monotone zero-sum games as a special case. Then, we present a novel algorithm that transforms the near-zero-sum games into a sequence of zero-sum subproblems, improving the gradient-based complexity for the class. Finally, we demonstrate the applicability of this new
class to model practical scenarios of games motivated from the literature.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the problem of computing NE in monotone games beyond the zero-sum regime. In particular, the authors define the class of monotone $\delta$-near-zero-sum games, where the $\delta$ parameter denotes that the coupled component of the game $g(\cdot, \cdot)$ is $\delta$-smooth. This class interpolates between zero-sum ($\delta=0$) and general-sum ($\delta=L$) monotone games. While the zero-sum case has been well studied and admits minimax optimal algorithms, the general-case complexity from Tseng (1995) remains less well-studied. The paper describes a new algorithm called Iterative Coupling Linearization (ICL), which is shown to converge faster than existing methods when $\delta$ is small enough, in both strongly and non-strongly monotone games. The algorithm relies on a linearization of the coupling part $g$, leading to a sequence of zero-sum subproblems. Finally, the paper exhibits several applications including regularized matrix games and competitive games with small cooperation incentives. The experiments are run on randomized examples of the former class, showing corroboration for the theoretical result.

### Strengths
- The paper is very well written and clear. The research question studied is of clear interest to the community, and is well-motivated in the introduction.
- The use of the potential function and linearization in the design of ICL is a nice idea, and leads to improved/optimal convergence rates in certain game classes.
- The applications given are nicely motivated and meaningfully extend the class of games for which we can guarantee fast convergence to Nash equilibria. The subsequent study of other similarly computationally 'easy' games seems to be a fruitful direction.
- The technical components of the paper are clean and easy to digest.

### Weaknesses
- The presence of the quadratic log factor naturally brings up the question of whether this is an artifact of the algorithm proposed or a fundamental separation between zero-sum and non-zero-sum games. The paper contains only one 'main' theoretical result, which is itself not a weakness, but I would have liked to see more discussion or even some preliminary results towards establishing lower bounds. This is briefly mentioned in the concluding remarks, but I believe this should be given more focus by the authors. Can the authors comment on this?
- The experiments given only pertain to Example 1. The ICL algorithm would be more compelling if some results can be shown in other examples or game classes.

### Questions
- Is it correct to say that recognizing that a game is $\delta$-near-zero-sum is not a trivial task, since Assumption 3 is an existence statement? Are there other effective notions of closeness that are easier to verify?
- No-regret algorithms have seen some success in converging to NE in multiplayer zero-sum monotone games, can a similar result potentially be established for ICL? Specifically, ICL's design relies heavily on the minimax (i.e. two player) formulation -- is there an extension that can work for games with more players?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a new class of two player games, termed “Monotone Near-Zero-Sum Games”, in between the classes of zero sum and general sum games. The class consists of games where averaging the utilities of the two players results in a smooth function, with the smoothness parameter $\delta$ indicating the interpolation between the classes. One of the motivations of the paper for defining the class lies in the gap of gradient complexity, which is the theoretical focus of the paper, between zero and general sum games. The authors propose a novel algorithm whose complexity depends on $\delta$ and matches, up to logarithmic terms, the state of the art results for both classes. Finally, they present some experimentation comparing their algorithm with previous methods in some families of games.

### Strengths
On a conceptual level, the paper does take a step in filling the gap between zero sum and general sum games, introducing an interesting class. Moreover, from the technical side, the proofs require technical work. On the application side, the transaction fee games are interesting as a class so I appreciate a method that solves them efficiently.

### Weaknesses
I think that the weaker aspect of the paper lies in its applications. Aside from the example mentioned above, I am not sure about the practical relevance of the classes that can be captured within the presented framework.

### Questions
Could you provide some additional examples of games that can be solved with your approach? Or some more illustrations, especially for the competitive games with small additional incentives? Also, could you include some references about the transaction fee games, even outside the literature of gradient complexity, as well as comparisons with different approaches to solve them? 

I will also add two minor comments:
Equation (13) is incorrect. It should be 2 instead of $1/2$. (The substitution later on is correct)
Figure 1 is somewhat confusing. Based on the text, I think that the best out of the 3 methods are depicted but that it is not clear from the figure itself. The legend is also confusing, with 21 entries but much fewer lines depicted. Also, it is not easy to tell some colors apart.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors targets on the gap between gradient complexity of monotone general game ($\mathcal{O}(\frac{L}{\min\{\mu, \nu\}}\cdot \log(D^2/\varepsilon)$) and monotone zero-sum game  ($\mathcal{O}(\frac{L}{\sqrt{\mu \nu}}\cdot \log(D^2/\varepsilon)$). To bridge this gap, the authors propose monotone near-zero-sum games , characterized by a smoothness paraemeter $\delta$ describing the game's proximity to a zero-sum game. Then, the author present Iterative Coupling Linearization (ICL) algorithm that provide black-box reduction from monotone near-zero sum game to  zero-sum games and derive gradient complexity in-between the known ones of zero-sum and general-sum games.

### Strengths
1. The problem is theoretically well-motivated. The authors clearly identify a gap between the gradient complexities of monotone zero-sum and monotone general-sum games, and take a principled step toward bridging this gap through the introduction of a new intermediate class of monotone near-zero-sum games.
2. The presentation is clear and well-structured. The decomposition of the potential function into a jointly convex coupling part and a convex–concave zero-sum part provides strong intuition that connects the problem formulation, algorithm design, and convergence analysis in a coherent way.
3. The proposed Iterative Coupling Linearization (ICL) algorithm is conceptually novel and algorithmically elegant, and the the convergence analysis is technically rigorous.

### Weaknesses
My main concern is generalizability of the method presented in the paper and the overall contribution of the paper. 
1. The main theoretical results hinge on the near-zero-sum smoothness parameter $\delta$ that measures how strongly coupled the game is. However, it remains unclear in which practical settings the $\delta$-smoothness assumption naturally holds or how to estimate it empirically. Although it is formally bounded by the overall smoothness $L$, it is uncertain how often $\delta$ would be sufficiently small in realistic problems to yield the claimed acceleration in convergence.
2. In addition, the presented results are developed specifically for two-player games, whereas the classical gradient complexity results for monotone general-sum games (e.g., Tseng, 1995; Nemirovski, 2004) extend to arbitrary numbers of players. It would strengthen the contribution to discuss whether the proposed framework and analysis can be generalized to multi-player settings, or to explain the technical barriers that currently prevent such an extension.

### Questions
Would you expect a similar analysis and complexity improvement to extend to the multi-player setting? If not, could you elaborate on the main technical barriers that prevent such an extension? In particular, are there specific steps in the convergence proof or properties of the two-player potential function that fail to generalize when the number of players exceeds two?

Moreover, could you provide additional examples or intuitions for practical game settings where the coupling part $g$ is expected to have a small smoothness parameter $\delta$? In other words, in what types of real-world or commonly studied games would the near-zero-sum assumption $(\delta<<L)$ naturally hold?

### Soundness
4

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
The paper examines a new class of games that interpolates between zero-sum monotone games and general-sum monotone games. It is motivated by a gap between solving monotone zero-sum games and monotone general-sum games. The main contribution is an algorithm that improves the iteration complexity for that class. It finally provides some motivating examples together with some numerical results.

### Strengths
The paper is well written and examines a concrete problem. To my knowledge, the gap in the number of gradients needed between monotone zero-sum and monotone general-sum has been unexplored. This gap can be substantial, which motivates the main contribution of the paper. The paper also does a good job at explaining the technical steps and highlighting the technical contribution compared to existing techniques. The proposed algorithm is quite natural and clean; I didn't find any notable issues in the technical steps.

### Weaknesses
Overall, I believe the paper is lacking in motivating the significance of the new class of problems and providing good applications of it. Some of the examples discussed in Section 4 look pretty artificial. For example, this setting of matrix games with transaction fees hasn't been studied before, to my knowledge, and its significance is unclear. Concerning Example 2, combing competition with cooperation is, of course, very interesting. But the precise way it is formulated in that example appears to be again artificial. The paper would benefit a lot from having a good application that has been studied in the past and the community would appreciate. Otherwise, it is hard to see the significance of the paper.

The experimental section also doesn't go far enough. The only experiment is run on this setting of matrix games with transaction fees, and the improvement compared to OGDA only manifests itself for a narrow range of the parameter. The paper would benefit from expanding the set of experiments to include other domains where the method has an advantage.

### Questions
- In the experiments, can the authors also report the gap of the average iterate of OGDA instead of its last iterate? In theory that should converge faster. It would also be interesting to see whether alternating instead of simultaneous updates make a difference in that experiment. 
- Can the authors provide more experiments using the class of competitive games with small cooperation incentives? Is there some specific setting in economics that naturally satisfies this assumption?

### Soundness
3

### Presentation
3

### Contribution
2

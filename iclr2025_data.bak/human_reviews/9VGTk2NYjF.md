## Human Reviewer 1

### Summary
In this work, the authors investigate the computational complexity of computing a Nash equilibrium in two-team zero-sum polymatrix games where one team consists of independent players (i.e., players who do not interact with one another). Specifically, they prove that this problem is complete for the complexity class CLS. To demonstrate hardness, they first reduce from MinQuadraticKKT—the problem of computing a KKT point of a quadratic function with box constraints—to MinmaxIndKKT, a min-max problem with an independence property they define. In a second step, they reduce this problem to a two-team zero-sum polymatrix game. Membership in CLS follows fairly straightforwardly from the recent result that QuadraticKKT is complete for the class CLS, as shown in [1], and using LP duality for transforming a min-max problem into a minimization.

### Strengths
The paper is generally well-written, though there are areas where phrasing could be improved. The problem under consideration is quite interesting and represents a step forward in establishing complexity results for two-team (zero-sum) games, i.e., min-max optimization problems beyond the case of having coupled constraints as in [2]. Of course, the more general case of having dependent adversaries (or that of a single adversary) remains open, as the authors highlight in Section 5.

### Weaknesses
I cannot identify any obvious weaknesses. Although the techniques and ideas are not particularly complex—as is often the case in results of this kind—this should not in itself be considered a weakness. However, the simplicity of the proof and the lack of novel ideas makes me more skeptical about my final score.

### Questions
- **Line 386**: In the reduction from MinmaxIndKKT, the authors define the candidate KKT point $(x_i, y_i)$ for the case where neither $x_i$ nor $y_i$ is in ${0, 1\}$ as $x_i = a_i$ and $y_i = d_i$. I assume that $a_i$ is simply a typo, as $a_i$ is already used to denote player $i$ on the first team. I think the authors likely intended to use $p_i$ and $q_i$ for $x_i$ and $y_i$, which would also align with the statement in line 415 indicating that these variables are close to their respective counterparts, $p_i$ and $q_i$.

References  
[1] The complexity of computing KKT solutions of quadratic programs.

[2] Constantinos Daskalakis, Stratis Skoulakis, and Manolis Zampetakis. The complexity of constrained min-max optimization.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper studies the problem of finding Nash Equilibria in two-team polymatrix games. Polymatrix games are a special class of n-player games with a succinct representation of the payoff functions. Each player's payoff is a sum of payoffs resulting from two-player games played with all the other players. This problem is known to be tractable when all interactions are zero-sum, and to be PPAD-hard in general. A special subclass of these games are team games, where each pair of interactions are either zero-sum (different teams) or coordination games (same team). Three team games are known to be PPAD complete. The main result of the paper is in showing that two team games are CLS hard (CLS is a structured subclass of PPAD).  This result holds even when one of the team consists of independent adversaries (their games consist of the zero matrix for all the payoffs). They also show that computing the minimax/ KKT point of a bilinear polynomial is also CLS hard.

### Strengths
The paper solves a well formulated problem about the complexity of finding Nash equilibria. This problem is a natural continuation of prior results about team polymatrix games. The technical proofs and reductions are interesting and well written.

### Weaknesses
The main weakness is in the lack of appeal to a broader ICLR audience. The paper has solid results in complexity theory and game theory but requires some connection to the machine learning audience. That such a connection exists is not in itself in question, there are a plethora of papers about learning equilibria in team games, but the paper offers no discussion about the broader significance of studying team games. The open problems section also mentions gradient based methods that converge to equilibria in time poly(1/epsilon), but there is not further discussion.

### Questions
Could you add some discussion about the broader landscape of team games, why we might care about them (if not necessarily the two-player polymatrix team games), and about the best-known algorithmic results in this space, particularly in the context of learning dynamics?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 3

### Summary
This paper studies the complexity of finding a Nash equilibrium in two-team polymatrix zero-sum games. They show that this problem is CLS-hard, and is in CLS if the adversaries are independent (thus establishing CLS-completeness in the latter case).

### Strengths
I think this is a good paper and vote to accept. The paper is clearly written and presents an interesting result. The hardness result about minimax KKT points is also rather clean, and may be of independent interest as a CLS-complete problem that may be relatively easy to make reductions from. The concerns below are very minor.

### Weaknesses
The section about ex-ante coordination contains some strange choices of phrasing. For example, all of the papers in that paragraph study extensive-form games (not just the last one), and the paper that shows "efficient algorithms exist under some assumptions about the players’ information set" is Zhang and Sandholm (2022), not Zhang et al. (2021).

To get parenthetical citations like (Lastname et al. 2023) instead of Lastname et al. (2023), use \citep.

### Questions
Perhaps the most obvious gap in this paper is the CLS-membership without the independent adversaries assumption. Do you think there is any hope to extend your techniques to that case?

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3
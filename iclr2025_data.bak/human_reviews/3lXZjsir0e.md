## Human Reviewer 1

### Summary
This paper presents a robust model-based algorithm for offline two-player zero-sum Markov games (RTZMGs), effectively addressing the challenges of learning under partial coverage and environmental uncertainty. The key contributions of the paper are as follows:

- The authors introduce the robust tabular zero-sum Markov game framework by extending the standard tabular zero-sum Markov game to a robust setting. Under this framework, they propose a new algorithm, RTZ-VI-LCB, which integrates robust value iteration with a data-informed penalty term to estimate robust Nash equilibria.
- The authors provide a finite-sample complexity analysis for RTZ-VI-LCB, demonstrating its optimal dependency on the number of actions. This represents the first set of optimal sample complexity bounds for RTZMGs.
- The authors establish a lower bound on the sample complexity for learning RTZMGs, confirming the tightness of their upper bound and demonstrating the near-optimality of RTZ-VI-LCB across varying levels of uncertainty.

### Strengths
The primary strengths of this paper can be summarized in the following two aspects:

1. This paper introduces the first algorithm that achieves optimal sample complexity with respect to the dependence on action spaces.  
2. The paper offers a comprehensive analysis of robust tabular zero-sum Markov games, presenting both upper and lower bounds on the sample complexity.

### Weaknesses
There are two major weaknesses from my perspective:

1. The authors do not discuss whether a Nash equilibrium exists under their definition of the robust zero-sum Markov game. It is well known that in robust Markov games, the existence of a Nash equilibrium can be affected by the choice of uncertainty sets and specific problem settings. Therefore, I believe it is essential to provide a discussion on the existence of Nash equilibrium within their framework.
2. Another weakness is the limited technical novelty of the work. The presentation in Sections 3.1 and 3.2 closely resembles that of [1], and the overall methodology appears to be a direct combination of [1] and [2]. The primary contribution seems to be the incorporation of the two-fold subsampling trick from [3] to sharpen the sample complexity bounds.

### Questions
Based on the discussion of the paper's strengths and weaknesses, I have the following questions for the authors:

1. The authors focus on the finite-horizon setting. Can the methodology presented in the paper be extended to analyze the infinite-horizon setting, as in [1]? Additionally, why did the authors choose to focus on the finite-horizon case rather than the infinite-horizon scenario?

2. The algorithmic framework follows from [1]. What are the specific technical challenges in extending the techniques of [1] from standard zero-sum Markov games to robust zero-sum Markov games?


 [1] Yan Y, Li G, Chen Y, et al. Model-based reinforcement learning is minimax-optimal for offline zero-sum markov games[J]. arXiv preprint arXiv:2206.04044, 2022.

   [2]Shi, Laixi, and Yuejie Chi. "Distributionally robust model-based offline reinforcement learning with near-optimal sample complexity." Journal of Machine Learning Research 25.200 (2024): 1-91.

   [3]Li G, Shi L, Chen Y, et al. Settling the sample complexity of model-based offline reinforcement learning[J]. The Annals of Statistics, 2024, 52(1): 233-260.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper studies robust two-player zero-sum Markov games.  Recent papers have provided near optimal sample complexity bounds in this setting under partial and limited coverage of historical data individually, but cannot handle both settings simultaneously.  This work provides an algorithm that can achieve near-optimal sample complexity under partial and limited coverage simultaneously while also providing information theoretic lower bounds.

### Strengths
The question of robust learning in strategic multi-agent settings has received considerable attention in recent years. This paper builds on recent work, providing a clear contribution by combining technical and algorithmic ideas from recent work to provide tighter and more general results than the state-of-the-art.  

Additionally, the work provides interesting lower bounds, whose proofs provide important new insight for the area.

### Weaknesses
The algorithmic novelty in the work is not clear.  A core component of the algorithm is a natural extension of work by Li et al the setting in this paper.  From my read, the key algorithmic novelty is primarily in the penalty term. 

The technical novelty of the paper is not clearly presented.  It seems that key components of the proof follow recent work (e.g. much of the work in step 1 of the proof follows closely the approach of Shi et al.).  That said, there are certainly new ideas in the proof, it is just that the paper does not do a good job of highlighting the new techniques in the analysis.  

Numerical comparisons to the work of Blanchet et al and Shi et al are not included.  Such comparisons would increase the potential impact of the paper and highlight the extent to which the improvement in theoretical bounds represent empirical improvements.

### Questions
Can you please clarify the new technical ideas in the proof as compared to the work of Blanchet et al and She et al?

Can you please clarify the relationship of the algorithmic ideas to prior work, highlighting which components are natural extensions and which represent new algorithmic ideas?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper proposed an algorithm RTZ-VI-LCB, designed to efficiently find the robust Nash Equilibrium(NE) in Robust Two-player Zero-sum Markov Games (RTZMGs). The authors employ confidence bounds innovatively in the algorithm, enabling it to achieve a sample complexity close to the lower bound except for the order of the horizon.

### Strengths
The paper is well-written and easy to follow. The problem of efficiently finding the robust NE in RTZMGs is significant for the field. The theoretical results are strong, as the sample complexity of the RTZ-VI-LCB algorithm nearly matches the lower bound for this problem class. Additionally, the lower bound analysis indicates that RTZMGs are not notably easier than traditional TZMGs.

### Weaknesses
There are still some technical problems to justify, which will be discussed in the Questions section.

### Questions
1. In remark 1, the paper mentions that the coefficient $C_r^\star$ could be $\frac{AB}{A + B}$. Given that the sample complexity result is $\tilde{O}\left(\frac{C_r^*(A + B)}{\epsilon^2}\right)$, does this imply that the complexity is reduced to $\tilde{O}\left(\frac{A B}{\epsilon^2}\right)$ in terms of $A$ and $B$, which is the same as the result in DR-NVI? 
2. How should we compare the term $\min\left(f(\sigma^+,\sigma^-),H\right)$ in the upper bound and the term $\min\left(1/\min(\sigma^+,\sigma^-),H\right)$ in the lower bound? Additional discussion on this comparison would clarify the practical implications of the upper bound's tightness relative to the lower bound.
3. From the similarity between the lower bounds of RTZMGs and RZMGs (Shi et al., 2024b), I assume that RTZMGs are not significantly easier than RZMGs. Given this, is it feasible to extend the RTZ-VI-LCB algorithm to more than two players?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
2

---

## Human Reviewer 4

### Summary
This work focuses on developing provable algorithm for distributionally robust multi-agent reinforcement learning in the face of environmental shift, in offline setting using only a history dataset. Considering two-player zero-sum games, it proposes RTZ-VI-LCB with an upper bound and a lower bound for this problem.

### Strengths
1. This is the first work that targets offline settings for robust MARL problems, which is an interesting topic.
2. It provides both upper and lower bounds for understanding this problem.

### Weaknesses
1. The writing and presentation need to be revised a lot. A lot of parts of the paper are similar to prior art. For instance, the two-fold sampling method in Algorithm 1 is almost the same as Algorithm 3 in [1]. Although cited the prior works, the algorithm needs to be rewritten entirely.
2. The contributions are a little bit overclaimed from the reviewer's viewpoint. In line 104, this work claims that "To the best of our knowledge, this is the first time optimal dependency on actions {A, B} has been achieved". While the concentrability coefficient also involves potential terms of A and B. So it is better to also say this is only for offline settings.
3. Some writing issues such as in line 107. The "transition kernel" does not need to be solved, it seems to need to be revised to "RTZMG". In line 113, "across a range of uncertainty levels", it seems there is something missing in this half sentence.
4. In the discussion part after showing the theorems, the reviewer highly suggests that the author check the claims again. For instance, in line 511-512, it seems the upper bound and lower bound do not match in $H$ even if $\min\\{\sigma^+, \sigma^- \\} \geq \frac{1}{H}$. The upper bound has $O(H^5)$, while the lower bound has $O(H^4)$? So it is not optimal yet, which is also claimed in the second paragraph of the discussion.

[1] Li, Gen, et al. "Settling the sample complexity of model-based offline reinforcement learning." The Annals of Statistics 52.1 (2024): 233-260.

### Questions
1. Why target two-player zero-sum games? Is there any special structure that helps the results, which hinders the authors from considering more general general-sum multi-agent games?

Other minors:
1) For presentation, as actually the max-player and min-player enjoys very similar formulation, algorithm update rules, and others, the presentation is a little bit redundant. It will be better to only write one time of them, such as equations 8(a), 8(b) can be represented as one if we let the min-player's everything be its negative version. The same for equation 9, 10, 18, the two terms in 22, and etc.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 5

### Summary
This paper addresses robust multi-agent reinforcement learning (MARL) in two-player zero-sum Markov games (TZMGs) by introducing the RTZ-VI-LCB algorithm, a sample-efficient approach to handle offline settings with environmental uncertainties. The algorithm improves robustness by applying value iteration with data-driven penalties and establishing sample complexity bounds without requiring full state-action space coverage.

### Strengths
1. The paper provides a rigorous theoretical framework, including upper and lower sample complexity bounds, which supports the robustness and efficiency claims of the RTZ-VI-LCB algorithm.

2. The design of RTZ-VI-LCB is explained in a step-by-step manner, making it easy to follow the rationale behind each component, such as the use of lower confidence bounds and two-player-wise rectangularity.

3. The paper adapts the robust Bellman equations specifically for two-player games, enhancing the clarity and relevance of the methodology in the context of MARL.

### Weaknesses
1. The paper assumes that both players in the two-player zero-sum game have identical uncertainty sets (same divergence function for both players). This simplifies the model but may limit its applicability to real-world scenarios where players could have different levels of uncertainty.

2. The penalty term introduced in the RTZ-VI-LCB algorithm is crucial for the robust value estimation, but the paper does not clearly explain how the penalty is calibrated or how different choices of penalty function influence the algorithm’s performance.

3. The paper assumes that historical datasets can be treated as independent samples after applying subsampling techniques, but it does not fully address the potential temporal dependencies within offline data.

### Questions
1. How does the algorithm's performance vary with different types of divergence functions beyond total variation, such as Kullback-Leibler divergence?

2. Would the RTZ-VI-LCB framework be adaptable to handle more complex multi-agent settings with more than two players?

3. How sensitive is the model’s performance to variations in the clipping parameter $C_r^*$, and what guidelines can be provided for choosing this parameter effectively?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
3
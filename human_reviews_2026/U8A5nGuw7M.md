# AMPED: Adaptive Multi-objective Projection for balancing Exploration and skill Diversification

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4, 4

## Abstract
Skill-based reinforcement learning (SBRL) enables rapid adaptation in environments with sparse rewards by pretraining a skill-conditioned policy. Effective skill learning requires jointly maximizing both exploration and skill diversity. However, existing methods often face challenges in simultaneously optimizing for these two conflicting objectives. In this work, we propose a new method, Adaptive Multi-objective Projection for balancing Exploration and skill Diversification (AMPED), which explicitly addresses both: during pre-training, a gradient-surgery projection balances the exploration and diversity gradients, and during fine-tuning, a skill selector exploits the learned diversity by choosing skills suited to downstream tasks. Our approach achieves performance that surpasses SBRL baselines across various benchmarks. Through an extensive ablation study, we identify the role of each component and demonstrate that each element in AMPED is contributing to performance. We further provide theoretical evidence that, with a greedy skill selector, greater skill diversity reduces fine-tuning sample complexity. These results highlight the importance of explicitly harmonizing exploration and diversity and demonstrate the effectiveness of AMPED in enabling robust and generalizable skill learning. https://geonwoo.me/amped/

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces "AMPED", a method for unsupervised skill learning and efficient adaptation to downstream tasks.

**Main contribution:** During pretraining, the method explicitly handles two potentially conflicting objectives:  
1. Achieving wide exploration of the state space  
2. Achieving a diverse distribution of skills  

The pretraining method relies on gradient surgery to ensure that conflicts between the gradients for the two objectives are resolved, thereby avoiding that the competing objectives negatively affect the pretraining process.

After pretraining, the downstream task can efficiently be solved by exploiting the skills discovered during pretraining.  
For this, AMPED learns a selector over the skills to maximize the downstream reward, which allows for more efficient adaptation compared to randomly sampling or fine-tuning skills.

The method is evaluated against a large number of unsupervised skill discovery baselines, on both didactic maze environments and more challenging continuous control tasks, all showing strong performance gains of AMPED compared to prior works.
An ablation over AMPED's components helps understanding the importance of each.

### Strengths
- The motivation is clear: conflicting objectives can deteriorate performance during pretraining.  
- The paper is well written and easy to follow. Key mathematical objects and issues are presented and explained clearly.  
- The paper formally connects the diversity of the pretrained skills with the number of samples needed to discover the optimal skill during the downstream adaptation step.  
- The experiments include a wide range of representative baselines, and the evaluation environments feature both a didactic 2D maze example and challenging URLB environments.  
- The results are strong, convincing, and consistent with the central claims of the paper.

### Weaknesses
- The key insights are already well understood, and the novelty is somewhat limited. The paper’s key insight, line 053, that conflicts can arise between gradients for different objectives, is well understood in the multi-objective RL and optimization literature, and thus not a strong novel finding.  

- To the best of my knowledge, the paper does not introduce fundamentally new approaches to solving the challenges in multi-objective optimization, but rather combines existing techniques and applies them to unsupervised skill discovery and downstream adaptation. The contribution is still valuable, but the paper is relatively applied in nature, IMO.  
- Without digging into the related works, it is unclear whether estimating the mutual information with AInfoNCE is a contribution of the authors. If so, I acknowledge it as a strength, slightly weakening my limited-novelty critique.  
- The learned hierarchical selector during fine-tuning is not conceptually different from prior works such as DIAYN, which also features a hierarchical selector over discovered skills.  

**Clarity:** Lines 300–304 do not sufficiently explain how the skill selector is learned exactly, and what skill fine-tuning entails. Please elaborate and explain this more clearly.  
  - Is “fine-tuning” used interchangeably with learning the skill selector?  
  - Are the skills themselves being fine-tuned to maximize the downstream reward?  
  This needs to be made more explicit. Consider describing clearly which parameters of your architecture are updated during the adaptive skill selection step, and how.  

**Nitpicks:**
- Lines 183–184: $z$ is defined as a skill, but $z^\ast$ is defined as a best policy, which is confusing. Does $z^\ast$ always induce the optimal state visitation distribution? What if that \$z^\ast$ is given to a suboptimal policy that cannot execute the optimal skill well?  
- Lines 251–252: I assume (1) and (2) are indices of different skills? The wording can be understood like they belong to the same skill.  
- Line 86 mentions “statistically significant results,” but no formal hypothesis test ($p < 0.05$) is performed. Consider changing the wording to “consistent improvements” or similar.

### Questions
As part of the ablation study, lines 425–431 discuss using different projection ratios. Can you elaborate a bit more on what is actually meant by *AMPED (ours)* here? Is it the procedure outlined in lines 287–295?  
  If yes, I think it would improve clarity to make that connection more explicit in the main text.

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
3

### Summary
This paper studies unsupervised skill learning and proposed AMPED, a novel method that aims to learn skills that are both diverse and explore the state space well. To do this, AMPED relies on an intrinsic reward based on an entropy bonus combined with RND for exploration, and an intrinsic reward based on contrastive learning for incentivizing the skills to be diverse. Finally, AMPED uses a gradient-surgery projection procedure to balance the exploration and diversity gradients and avoid gradient conflict. On the URLB benchmark, they find their method to outperform all baselines in terms of mean, IQM, and optimality gap. The authors also provide ablation studies to test the importance of the entropy bonuses, the RND bonus, the Anisotropic InfoNCE reward, the gradient surgery procedure, and the skill selector.

### Strengths
1. Evaluation was done over 10 random seeds and reports results according to recommendations from the Rliable framework.
2. Strong results: outperforms all baselines in terms of mean, IQM, and optimality gap.
3. The motivation for having explicit exploration and diversity rewards is clear.
4. Figure 4 provides a very helpful visualization of the gradient surgery procedure.
5. Figure 5 provides a good qualitative visualization of the exploration performance and the distinguishabiltiy of the learned skills for all considered methods in a toy maze.

### Weaknesses
1. Part 2 of Figure 2 is nothing new: this seems to be a standard HRL setup for skill discovery methods? In general, the paper emphasizes the benefits of the skill selector, while to me this just seems like HRL over the learned skills which has been done in prior work already (see METRA or CSF). Could the authors comment on this? Is there any differences or contributions that I’m missing?
2. Page 4: “is the discounted total state distribution” → do the authors mean the discounted state occupancy measure here? If so, I’d recommend replacing it with that since it’s the canonical terminology for this.
3. Section 4.4: “… this indicates faster convergence of AMPED” → based on Table 3, I disagree with the authors on this claim. It appears to me all the numbers reported fall within each other’s error bars, questioning the statistical significance. Could the authors comment on this?

### Questions
1. Page 7: “In order to ensure a fair comparison, we fine-tuned all methods under identical conditions without using the skill selector.” → how does that work? Every skill gets fine-tuned towards the same task reward? Does this preserve the skill behavior? Do the skills change a lot during fine-tuning?
2. Section 3.1: "and ρ⋆ ∈∆(S^H) be a corresponding state distribution." -> If this is a state distribution, shouldn't it just be ∆(S) instead of ∆(S^H)? The latter seems like a distribution over trajectories. This also confuses me later where it says "Draw n i.i.d. trajectories from optimal policy S(1),...,S(n) ∼ρ⋆". This seems like drawing n states instead of n trajectories?
3. Could the authors expand a bit on exactly why it's necessary to have both RND and entropy exploration? There's some discussion about it on page 5 but it wasn't super clear to me.
4. Page 5: "However, if the supports of different skill coverages become disjoint, the diversity loss Ldiversity no longer enforces inter-skill distributional separation, which can lead to skill clustering rather than promoting broad coverage of the state space." -> Could the authors provide some clarification on this statement? I couldn't really follow what exactly is meant here.

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
4

### Summary
This work proposes an adaptive training method to balance exploration and diversity in the application of unsupervised reinforcement learning (URL) for skill discovery. The authors argue that optimizing exploration and diversity in skill discovery tasks can be contradictory. Therefore, they design the algorithm from a multi-objective optimization perspective, defining "enhancing skill exploration" and "improving skill diversity" as two distinct objectives for the URL algorithm. They then propose using a gradient clipping approach to balance these two optimization goals. The proposed method demonstrates superior performance compared to existing baselines across multiple benchmarks. Furthermore, the paper includes comprehensive ablation studies that experimentally validate the necessity and correctness of the various components within the proposed algorithm.

### Strengths
- This paper is easy to follow.
- The results of the ablation study are comprehensive.

### Weaknesses
- Some content is unclearly explained, such as the specific form of $\rho$ and the meaning of the several metrics in Figure 6.
- The proof of Theorem 1 is currently difficult to ascertain as valid (see Questions for details).
- The experimental settings are too simple, primarily consisting of basic maze environments.
- The presentation of the conflicts during the skill learning process is not intuitive.

### Questions
- In the proof of Theorem 1, the authors state that a larger $\delta$ increases $\Delta$ and thus reduces the required sample size. However, Inequality 3 only provides a lower bound for $n$ and does not explicitly demonstrate that increased diversity reduces the necessary sample size. Could the authors clarify this point?
- What are the exact calculation of median, IQM (is it Interquartile Mean?), and mean scores and optimality gap in Figure6?
- Table 1 shows the percentage of conflicts. What does the corresponding learning curve look like for these conflicts?
- How does the dimension of $z$ affect the final policy performance?
- The presentation of the conflicts during the skill learning process is not intuitive. Could the authors make it clearer, perhaps by reporting the curve of gradient similarity changes?
- Is the gradient similarity related to the type of task? For instance, in some scenarios, the conflict between exploration and diversity might be small, while in others it could be large.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a new framework for skill-based reinforcement learning called AMPED. During the pre-training phase, the method uses a "gradient-surgery projection" to mitigate conflicting gradient components. During the fine-tuning phase, it employs an "adaptive skill selector" to leverage the learned skills for downstream tasks. Experiments demonstrate that this method surpasses baselines on the URLB and Maze benchmarks, and proves that greater diversity.

### Strengths
1. The experimental analysis is thorough. In particular, extensive ablation studies are provided to demonstrate the effectiveness of each component in different scenarios.

2. The paper is exceptionally clear and logically fluent. The progression from the problem definition in the Introduction, to the Methods section, and then to the Experimental Analysis is easy to follow. Furthermore, the Appendix provides extremely detailed further analysis, enhancing comprehensibility.

3. The paper models this problem as a "gradient conflict" in multi-objective optimization and introduces "gradient-surgery projection" to resolve it. This is a very insightful and novel approach.

### Weaknesses
1. I find that the method presented in the paper largely consists of combining existing techniques, and it lacks significant methodological innovation.

2. Directly using SAC to learn a policy over the skill repertoire (i.e., changing the prior z) seems feasible. However, the paper does not couple the two RL training processes well, which may lead to unstable convergence and, as noted in Table 6, instances of ineffectiveness.

3. The experimental environments are relatively simple, with experiments primarily compared on the Maze and 12 URLB downstream tasks. I believe the contribution to practical applications (e.g., pixel-based environments) is minimal.

4. The algorithmic complexity is high (integrating CIC, RND, SAC, etc.), making the training process relatively heavy. Moreover, the efficiency analysis provided is not intuitive enough.

### Questions
1. The authors state that METRA was omitted because it does not exhibit performance improvements on the URLB relative to CeSD. However, I noted in the paper[1] that METRA outperforms CIC on some tasks. Given that CIC is a relatively strong performer in the authors' own experiments, could the authors please add supplementary experiments comparing against METRA?

2. Could you provide more comparisons and analysis of the algorithm's efficiency? For instance, by plotting learning curves that show the evaluation metrics as a function of training steps?

[1] METRA: SCALABLE UNSUPERVISED RL WITH METRIC-AWARE ABSTRACTION

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose AMPED, a skill-based reinforcement learning framework that balances exploration and skill diversity objectives during pretraining. The method treats these as competing objectives in a multi-objective RL setting and applies gradient surgery (PCGrad) to resolve conflicts between diversity gradients (from AnInfoNCE) and explroation gradients (from particle-based entropy and RND). Those skills are exploited in the following fine-tuning phase, in which a SAC-based skill selector adaptively chooses skills for downstream tasks.

### Strengths
- The paper is very clear and easy to follow, and the proposed method is well-motivated.
- Strong results on several tasks from the URLB benchmark with comparisons against 7 baselines, with consistent improvements.
- Table 2 effectively demonstrates that each component (RND, AnInfoNCE, gradient surgery, skill selector) contributes meaningfully to performance.
- Figure 5 (Tree Maze) provides intuitive evidence that AMPED achieves both skill separation and state coverage where baselines fail at one or both.

### Weaknesses
Section 4.4 claims are problematic in multiple ways:
- Line 200 claims "empirical validation appears in Section 4.4," but Theorem 1 assumes sampling trajectories from the **optimal** policy (requiring access to it), while Section 4.4 measures convergence speed to the optimal policy during fine-tuning. Theorem 1 establishes that diversity reduces sample complexity for skill identification given optimal trajectories—not that diversity accelerates convergence during learning.
- The statement "Combined with higher final returns, this indicates faster convergence of AMPED" is inaccurate, as other variants finish converging before (e.g. BeCL actually converges 90% faster on Flip, and many other comparisons are not statistically significant). The paper should also report the **rate of improvement** (reward gained per step) rather than steps-to-threshold, especially when final returns differ across methods.
- The logic "AMPED has higher diversity than CIC/BeCL" + "AMPED improves faster" $\Rightarrow$ "diversity causes faster improvement" seems flawed. Other algorithmic differences (objective functions, gradient surgery, skill selector) could explain the improvement rate. To isolate diversity's effect, ablations should vary diversity within AMPED itself (e.g., different numbers of skills or projection ratios $p$), not compare across different methods.
- Section 4.4 only evaluates Walker tasks, making it difficult to establish a general relationship between diversity and rate of improvement.

### Questions
Table 10 shows skill dimension is fixed at 16 for all URLB experiments. How do diversity and performance vary with different numbers of skills (e.g., 4, 8, 32)? This would help validate whether increased diversity consistently improves results.

### Soundness
4

### Presentation
4

### Contribution
3

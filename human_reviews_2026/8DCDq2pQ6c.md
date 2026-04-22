# Directional-based Wasserstein Distance for Efficient Multi-Agent Diversity

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 6

## Abstract
In the domain of cooperative Multi-Agent Reinforcement Learning (MARL), agents typically share the same policy network to accelerate training. However, the use of shared policy network parameters among agents often leads to similar behaviors, restricting effective exploration and resulting in suboptimal cooperative policies. To promote diversity among agents, recent works have focused on differentiating trajectories of different agents given agent identities by maximizing the mutual information objective. However, these methods do not necessarily enhance exploration. To promote efficient multi-agent diversity and more robust exploration in multi-agent systems, we introduce a novel exploration method called Directional Metric-based Diversity (DMD). This method aims to maximize an inner-product-based Wasserstein distance between the trajectory distributions of different agents in a latent trajectory representation space, providing a more efficient and structured Wasserstein distance metric. Since directly calculating the Wasserstein distance is intractable, we introduce a kernel method to compute it with low computational cost. Empirical evaluations across a variety of complex multi-agent scenarios demonstrate the superior performance and enhanced exploration of our method, outperforming current state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes **Directional Metric-based Diversity (DMD)**, a new intrinsic-reward mechanism for cooperative multi-agent RL that encourages **policy diversity** by maximizing an **inner-product–weighted Wasserstein distance** between trajectory distributions of different agents.  
Key technical ingredients are:

- A **contrastive representation space** for trajectories obtained by supervising a CPC-style encoder with **learnable “identity” embeddings**; the authors argue this prevents collapse when all agents start from the **shared-parameter** policy.  
- A **kernel-based efficient estimator** of the 2-Wasserstein distance between trajectory distributions.  
- An **intrinsic reward** that aligns the Wasserstein vector between agent *i* and every other agent *j* with a **randomly sampled direction** *zj*; maximizing this reward is claimed to yield **structured, task-relevant diversity**.  
- Integration into both value-based (QMIX) and policy-based (MAPPO) algorithms with an **auxiliary utility network** that learns from the intrinsic reward.

Experiments are conducted on **Pac-Men**, **SMAC**, and **SMACv2**. The authors report **higher win-rates / returns** than QMIX, MAPPO, and several prior diversity-promoting baselines (MAVEN, EOI, SCDS, MAPD, DiCo, etc.).

### Strengths
- **Originality**: Using the **Wasserstein distance as a directional diversity signal** is new in the MARL literature; prior work either maximizes mutual information or the scalar Wasserstein norm. The **inner-product formulation with random direction vectors** is a creative twist.  
- **Quality**: The **kernel SGD estimator** for Wasserstein distance is implemented carefully; the **contrastive loss with learnable identity prototypes** is technically sound and helps alleviate the “all-identical” initial policy issue.  
- **Clarity**: The paper is **well structured** (background → method → ablations → large-scale experiments). Pseudocode and open-source code are provided.  
- **Significance**: If the claims hold, the work offers a **principled way to maintain diversity under parameter sharing**, which is the **de-facto training paradigm** in large-scale MARL systems.

### Weaknesses
### W1. **Conceptual gap: “Wasserstein ↑ ⇒ better performance” is never justified**
- The manuscript **assumes** that **larger Wasserstein distance between trajectory distributions** automatically translates into **better exploration and higher team reward**.  
- **No theorem, example, or ablation** shows that **performance correlates with the magnitude** of the proposed diversity measure; Figure 4 only shows that **removing the term hurts**, but does **not validate the directional part**.  
- **Counter-evidence**: in homogeneous tasks (8m, 5m\_vs\_6m) the **optimal policy is almost identical** for all agents; forcing diversity could **harm performance**, yet the paper still reports gains without explaining why.

### W2. **Problem motivation is tied to parameter sharing, but the link is weak**
- The paper claims parameter sharing makes **“policy similarity severe”**, but **no ablation removes sharing** to quantify how much diversity is lost.  
- **Missing baseline**: compare **separate policy networks** (no sharing) vs **shared + DMD**; if separate networks already give enough diversity, DMD may be **redundant**.  
- **Parameter sharing is never revisited** in SMACv2 or GRF experiments—readers cannot tell whether the **gains come from diversity or from better exploration** that any count-based bonus could deliver.

### W3. **Algorithmic description and reproducibility gaps**
- **Figure 1** is **ambiguous**: arrows do **not show gradients**, the **intrinsic utility network** is introduced **only in words**, and the **dimensionality** of *zj*, *da*, *ct* is **missing**.  
- **Appendix G pseudocode** omits **learning rates, batch sizes, number of SGD steps** for the dual variables, and **how many Monte-Carlo samples** are used for the Wasserstein estimate—**reproducing the estimator is non-trivial**.  
- **Hyper-parameter sensitivity (Table 7)** is **too narrow** (only α); **no sensitivity** to the **kernel bandwidth**, **dimension of random features**, or **number of directions** *zj* is reported.

### W4. **Empirical evaluation overstates generality**
- **All environments are cooperative, fully-observable to the critic, and discrete-action**; the **benefit in partial-observation, competitive or continuous-control domains** is **untouched**.  
- **Baselines are not fairly tuned**: QMIX and MAPPO use **default hyper-parameters** while DMD uses **extra networks and larger buffer**—**compute budget asymmetry** is **not controlled**.  
- **Statistical significance**: error bars in Fig. 2–3 are **standard deviation over 5 seeds**, **not standard error or confidence intervals**; with **only 32 test episodes** the **variance is high** and **improvements may not survive stricter testing**.

The paper introduces an **interesting directional extension** of Wasserstein-based diversity, but the **central premise**—that **maximizing this particular distance yields better cooperative policies**—is **not theoretically or empirically substantiated**. The **evaluation protocol** does **not disentangle** the **effects of parameter sharing**, **exploration**, and **increased network capacity**, and the **writing omit key implementational details** that are **essential for reproducibility**. I encourage the authors to **tighten the motivation**, **provide ablations that isolate the role of parameter sharing**, and **supply rigorous statistical tests** before resubmission.

### Questions
Q1. **Directional diversity vs. performance**:  
Can you provide **scatter plots** of **episode return vs. empirical Wasserstein distance** across training? If the correlation is weak, how do you defend the **causal chain** “larger *W* ⇒ better exploration ⇒ higher return”?

Q2. **Parameter-sharing ablation**:  
Is it possible to run **“separate networks (no sharing)”** and **“shared + DMD”** on 3s5z and Pac-Men. How much of the gain survives when agents are already diverse by construction?

Q3. **Homogeneous-task consistency**:  
In **focus-fire** scenarios the **optimal joint policy is nearly identical**. Why does **forcing directional diversity not degrade performance**? Does the **intrinsic reward vanish** when agents naturally converge to the same actions?

Q4. **Estimator variance**:  
What is the **variance** of the **kernel-based Wasserstein gradient** when the **number of random features** *m* = 64? Did you try any ablations or **adaptive bandwidth**? A **learning curve with different m** would clarify robustness.

Q5. **Clarity of architecture**:  
Could you release a **single-file diagram** that shows **shapes of all tensors**, **gradient flows**, and **where each loss term** is computed? The current figure mixes **data flow** and **conceptual blocks**, which makes implementation error-prone.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Directional Metric-based Diversity (DMD), an exploration method for cooperative MARL. This paper focus on heterogeneous MARL without the paremeter sharing setting. The proposed DMD first learns distinguishable trajectory representations for agents via a contrastive loss. Then, within this contrastive representation space, DMD computes the Wasserstein distance between agents as an intrinsic to encourage exploration. Experimental results indicate the efficiency of DMD on PAC-MEN, GRF, SMAC and SMACv2 environments.

### Strengths
1. The experimental results are extensive, and competitive compared to SOTA MARL methods. 
2. The use of Wasserstein distance for measuring agents' differences is novel.
3. The writting is good and easy-to-understand, with clear notations.

### Weaknesses
1. The novelty of the use of Contrastive Predictive Coding is fair. What is the difference between the contrastive learn modules of DMD and [1] ?
2. Fig. 2 and fig. 3 are distorted by incorrect scaling. 
3. The motivation of the use of the Wasserstein distance is not clear. The paper only empirically compare the proposed inner-product based Wasserstein distance with other versions of Wasserstein distance, and the comparison with other distance measurements, both methodologically and empirically, is missing. Why choose the Wasserstein distance rather than other distribution difference measurements?
4. The implementation of the directionality of the proposed inner-product based Wasserstein distance is confusing. The paper uses a latent variable to provide directional guidance. However, in Line 248, the authors mention that this latent variable ***"is sampled from a fixed uniform distribution"***. So why this variable is informative is unclear. 
5. It seems the performance improvement comes from the contrastive learning module, which is identical in [1] and [2], since the learning curves, visualization heatmaps and the t-nse trajectory representations in this paper looks similar with [1] and [2]. 



[1] Toward Efficient Multi-Agent Exploration With Trajectory Entropy Maximization, ICLR 2025

[2] Learning distinguishable trajectory representation with contrastive loss, NeurIPS 2024

### Questions
1. The authors claim in Line 50 that, parameter sharing can result  in poor multi-agent diversity and exploration. And in Line 63, the authors claim that parameter sharing can adversely affect the performance of traditional Wasserstein distance based method, which is the main motivation of the proposed Wasserstein distance. So if I understand correctly, the proposed Wasserstein distance based method runs under parameter sharing. Then one question arises, **parameter sharing learns identical policies for all agents, thus how can the proposed method encourage diversity?**
2. Following Q1, if the proposed DMD fits for non-parameter sharing, as [1] does, then where does the following motivation in Line 63 come from? ***"...these methods fail to account for the similarity in agents’ initial policies due to shared policy network parameters."*** Moreover, why is the similarity in agents’ **initial policies** important?
3. In experiements, is parameter sharing or not? It is reported that in SMAC, sharing parameter can lead to better performance. So it is better to compare DMD with baselines under both parameter sharing and non-parameter sharing settings. 

[1] Measuring policy distance for multi-agent reinforcement learning, arXiv.

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
5

### Summary
This paper introduces a multi-agent exploration method, Directional Metric-based Diversity (DMD) to address the issue of homogeneous agent behaviors resulting from parameter sharing. DMD leverages contrastive learning to encode trajectories and identities into distinguishable embeddings, utilizes an inner-product based Wasserstein distance in a latent contrastive trajectory representation space to calculate intrinsic rewards, and integrates seamlessly with existing MARL algorithms such as QMIX and MAPPO. Experiments on the Pac-Men, SMAC, and SMACv2 benchmarks demonstrate its superior effectiveness.

### Strengths
The paper is well-written and clearly presented.
The method effectively combines Wasserstein distance with contrastive learning to address the diversity challenge in MARL, showing strong empirical performance.
The experiments in the Pac-Men section are very intuitive—particularly the heatmap visualizations, which clearly demonstrate the method’s effectiveness.
The t-SNE visualization (Figure 5) provides a clear insight into the learned representations and diversity of trajectories.
The paper includes comprehensive ablation studies and comparisons with multiple baseline methods.
In SMAC, DMD shows significant improvements over baselines, especially in more challenging scenarios.

### Weaknesses
1. Introducing contrastive learning into MARL to learn identity-aware representations is not novel. Prior works such as [1], [2], and [3] have already employed contrastive learning to enhance identity-aware representation, improve credit assignment, and facilitate efficient exploration—albeit with different information forms. These works are not adequately discussed in this paper.

2. The core idea of this paper is quite similar to [2] and [3]. The main concept—using contrastive learning to obtain distinguishable trajectory representations and then measuring the distance between them to encourage diversity—has been explored previously. Consequently, the only new component appears to be the inner-product based Wasserstein distance (Section 3.2). However, it remains unclear what advantage this metric offers over, for example, the trajectory entropy measure used in [2].

3. Weak theoretical motivation for the inner-product based Wasserstein distance:
   Equation (6) defines the intrinsic reward $r^{a}_{w}$, where $z_j$ is randomly sampled from a fixed uniform distribution $p(z)$. Since $z_j$ is purely random, it is unclear why this design would “result in the visitation of diverse trajectories with significant variations.” Although the ablation studies empirically demonstrate improvements, a more theoretically grounded explanation and insight are needed—especially because this is the paper’s main claimed contribution.

4. The content and experimental figures closely resemble those in [2] and [3]. Some visualizations (e.g., Figure 8) appear very similar to those in Figure 8 of [3] (TEE). The authors are encouraged to include new visualization results that better highlight the unique insights and contributions of this work.

5. The comparison experiments omit recent exploration methods, especially the precursor works [1] and [2]. This omission weakens the overall contribution. Notably, in Table 1, DMD reports average returns of 0.93, 0.89, and 0.86 on the three SMACv2 maps. In contrast, TEE [3] reports 0.96, 0.95, and 0.87 on the same maps. Similar trends appear in SMAC results, suggesting that the Wasserstein-based metric may perform no better, and sometimes worse than trajectory entropy. Thus, DMD may effectively be an ablation or variant of TEE rather than a distinct new method.

6. Comparing the main experimental results with [2] and [3], it seems that even the base algorithm (CTR [2]) already achieves strong performance. This raises doubts about whether the trajectory diversity rewards introduced in DMD (and TEE) genuinely contribute significant additional value. The authors should clearly articulate the core advantages of the Wasserstein-based distance and provide direct empirical comparisons with other metrics to justify its use and significance.

  

  [1] Liu S, Zhou Y, Song J, et al. Contrastive identity-aware learning for multi-agent value decomposition[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023, 37(10): 11595-11603.

  [2] Li T, Zhu K, Li J, et al. Learning distinguishable trajectory representation with contrastive loss[J]. Advances in Neural Information Processing Systems, 2024, 37: 64454-64478.

  [3] Li T, Zhu K. Toward Efficient Multi-Agent Exploration With Trajectory Entropy Maximization[C]//The Thirteenth International Conference on Learning Representations.

### Questions
See other sections.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an exploration method DMD based Wasserstein distance between different agents’ trajectory distributions in a latent trajectory representation space. Theoretical analysis and numerical examples are given to demonstrate the advantages of DMD.

### Strengths
The paper introduces an exploration method DMD based Wasserstein distance between different agents’ trajectory distributions in a latent trajectory representation space. Theoretical analysis and numerical examples are given to demonstrate the advantages of DMD over other alternatives in MARL.

### Weaknesses
The examples used are still small size. It is not clear how the proposed DMD method scale as the number of agents grow to larger size.

### Questions
What is the computational cost by DMD and whether it fits large scale muti-agent systems?

### Soundness
3

### Presentation
3

### Contribution
3

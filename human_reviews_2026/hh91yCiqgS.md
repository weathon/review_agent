# General Exploratory Bonus for Optimistic Exploration in RLHF

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 6

## Abstract
Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods often fail to realize true optimism. We provide a theoretical analysis showing that current formulations, under KL or $\alpha$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $\alpha$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper addresses the issue of exploration in LLM finetuning with RLHF. The authors show that a previously proposed novelty bonus does not induce more exploration and extend this insight from the proposed KL-divergence to the class of $\alpha$-divergences. Finally the authors define a condition they claim to be optimistic and develop a general exploratory bonus (GEB) based on it. Evaluation against other exploratory bonuses appears promising.

### Strengths
The addressed problem of insufficient exploration during LLM fine tuning is relevant and the analysis of the existing exploratory bonus is insightful.

### Weaknesses
I need to clarify that I am not an LLM expert, and am not familiar with any of the cited literature. As such I had huge problems following the paper's story and math. While I would reject this paper due to my lack of comprehension, I am willing to follow the recommendation of reviewers more experienced in this field. I admit that some of my problems might be trivial for experts in the field, but the fact that an outsider cannot follow some of the notations and definitions should not be ignored either. 

In more detail:
- While the intuition in Figure 1 is very clear, I was wondering whether optimistic exploration (as used in RL literature [1]) is even desirable in LLM fine tuning. Most of the possible trajectory space is not sampled by the reference model and exploring those parts seems counter-productive: most of those trajectories will not follow proper grammar and are out-of-distribution for the learned RLHF reward function (which might therefore prefer them although they are garbage). Both are reasons to stay away from trajectories that have very small probabilities under the reference policy. So which trajectories do the authors *actually* propose to explore?	
- Even after repeated attempts, I was unable to make sense of Definition 3.1. In RL literature there is a clear definition of optimism for exploration [1], which requires to estimate an epistemic upper bound on the true return/reward of an action. How is Def.3.1 related to that? What is the intuition behind an "ideal sampling distribution" $\pi_s$? Later the authors mention they "adopt the commonly use $\pi_{ref}$ as $\pi_s$", but I fail to see what the 2nd order derivative of the exploration bonus w.r.t. the current policy $\pi$ (that's what it is, right?) and the reference policy $\pi_{ref}$ has to with optimism. 
- The introduced formalism has many undefined symbols or at least does not explain some default symbols. For example, I do not understand what the "policy-reparameterized reward model" in line 167 is. The left-hand side is a function of a policy, but the right hand side contains queries $x$ and completions $y$. Where do these come from? In line 163 it is defined as $r(x,y)=r(\pi)$, so are $x$ and $y$ just omitted in the function signature? What is the idea behind this reward and what does it have to do with Def.3.1? Another example is the appearance of $f'$ and $f''$ in the proof of Lemma 3.2: is this a partial derivative? If so which one? Why did the authors not use the clearer $\frac{\partial f}{\partial x}$ formalism they relied on earlier? 
- Theorem 4.2 appears grammatically malformed. The first sentence contains an if, but no then part. Later there is a $\forall (x,y)$ quantifier, but no $x$ or $y$ appear in the theorem. The meaning of these things might be obvious to someone familiar with the literature, but a mathematical statement must be self-contained.
- The experiments do not contain standard deviation and I could not find out whether they have been repeated at all. This might be common in LLM literature, where repeating experiments can be prohibitively expensive, but I was unable to gauge whether improving from 79.71 to 81.00 is a worthwhile improvement or just stochastic noise.

 
[1] Lattimore T, Szepesvari C. Bandit Algorithms. Cambridge University Press; 2020.

### Questions
See above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper aims to address limitations in existing exploratory bonus formulations for reinforcement learning from human feedback (RLHF). The authors claim that prior approaches under KL or α-divergence regularization fail to achieve true “optimism in the face of uncertainty.” They propose the General Exploratory Bonus (GEB) framework, which theoretically corrects these failures and empirically improves performance on several RLHF benchmarks.

While the paper tackles an important question, it suffers from serious clarity, organization, and presentation issues. The writing quality is poor, making the technical content extremely difficult to follow. Furthermore, the claimed theoretical and empirical contributions are not clearly or convincingly demonstrated.

### Strengths
The topic (exploration in RLHF) is timely and relevant. The attempt to provide a unified theoretical treatment across divergence families could be valuable if clearly presented.

### Weaknesses
The paper’s language is riddled with grammatical errors and awkward phrasing. Many sentences are either ungrammatical or semantically unclear, which severely hinders readability. Several paragraphs are almost unreadable due to convoluted sentence structures and a lack of coherent logical flow.
While Figure 1 conveys an important idea, its design is confusing: it is unclear why a line connects the four sections instead of simply displaying their values. This creates the impression of a continuous transition between them, which undermines the intended clarity.

### Questions
Your optimism condition (Definition 3.1 and Theorem 4.2) is central to the paper, but its mathematical and conceptual justification remains unclear. Could you clarify why it is an appropriate formalization of the “optimism in the face of uncertainty” principle, and how it connects to standard exploration metrics (e.g., uncertainty, entropy, or epistemic variance)?

Most reported improvements (e.g., in Tables 3 and 4) appear to be within 1–2 % over baselines. Could you provide more evidence that these differences are statistically significant and not due to noise or hyperparameter sensitivity?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper identifies a fundamental theoretical failure in existing exploratory bonus methods for reinforcement learning with human feedback (RLHF) and proposes General Exploratory Bonus (GEB) as a principled solution. The authors prove that current formulations under KL and α-divergence regularization paradoxically bias exploration toward high-probability regions of the reference model rather than uncertain regions, contradicting the "optimism in the face of uncertainty" principle. To address this, they introduce GEB with a reference-dependent reward regulation that provably satisfies optimism, unifies prior heuristic bonuses as special cases, and extends naturally across the α-divergence family. Empirical validation on LLM alignment tasks demonstrates consistent improvements over baselines.

### Strengths
1. The failure modes of existing bonuses are rigorously proven, not merely observed. Lemmas 3.1-3.2 and Theorem 3.3 are solid contributions that clarify fundamental issues.

2. The paper provides a general framework (GEB) that unifies prior work and extends beyond KL divergence.

3. Figure 1 provides intuitive understanding, while mathematical sections deliver rigor. Both audiences (practitioners and theorists) can engage.

4. Testing across two LLM backbones, three divergence settings, and both in-domain and out-of-domain tasks demonstrates breadth.

### Weaknesses
1. The practical implementation restricts bonus computation to rejected responses for stability, but this isn't theoretically justified. Does this modification preserve the optimism guarantee? If not, the theory's practical applicability is weakened.

2. Win-rate gains are ~1-2% on UltraFeedback and marginal on out-of-domain tasks (Table 4). While consistent, these are not compelling. Concern: Is the gap between theoretical promise and empirical gains expected? Could better tuning of κ yield larger improvements?

3. Missing Ablations: What is the effect of restricting the bonus to rejected responses? Ablating this would isolate theoretical vs. practical design choices.

4. The regret bound in Theorem C.1 has the same dependence on T as prior work. Does GEB offer better constants? This isn't discussed, weakening the theoretical justification for improved sample efficiency.

5. The paper doesn't characterize problem instances where GEB would yield larger improvements. When is exploration in low-pi_{ref}
πref​ regions most critical?

### Questions
1. The practical restriction of bonus computation to rejected responses (mentioned in Section 4) lacks theoretical justification. Does this modification affect the optimism guarantee? If so, shouldn't this be analyzed formally?

2. Does GEB improve the constants in the regret bound (Theorem C.1) compared to Cen et al. (2025)? The T-dependence appears identical—is there an improvement in problem-dependent terms?

3. How do the three u designs in Table 2 compare empirically? This would clarify the practical importance of theoretical flexibility.

4. What happens if you remove the restriction to rejected responses and compute the bonus on all responses? This would isolate the theoretical contribution from practical engineering.

5. Are there problem instances or regimes where GEB provides substantially larger improvements than standard methods? Characterizing this would strengthen motivation.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper addresses optimistic exploration in reinforcement learning from human feedback (RLHF),  arguing that some existing exploratory bonus methods under KL or $\alpha$-divergence  regularization end up sampling actions with high probability under the reference policy regions  instead of sampling low probability actions. The authors prove theoretically that current  formulations fail to satisfy the optimism principle and introduce the General Exploratory Bonus  (GEB), a framework that introduces reference-dependent reward regulation to counteract  divergence-induced bias. The paper argues that GEB achieves optimism and extends naturally  across the $\alpha$-divergence family. Experiments on large language model alignment tasks  across different divergence settings (KL, Hellinger, forward KL) and model backbones (Llama-3-8B SFT, Mistral-Instruct-v0.3) demonstrate that GEB outperforms baselines.

### Strengths
Originality 
- The theoretical analysis is insightful, formally proving through Lemmas 3.1-3.2 and Theorem 3.3  that some existing exploratory bonus formulations fail to achieve optimism under KL, $\alpha$- divergence, and general $f$-divergence regularization, with the bonus encouraging sampling from  high probability regions under the reference policy.  
- The framework unifies prior heuristic methods (SELM, XPO, VPO) as special cases while extending  beyond the KL-divergence to the $\alpha$-divergence family. 

Quality 
- The experimental section covers multiple divergence classes, two LLM backbones, both in-domain  and out-of-domain evaluations (AlpacaEval2, MATH-500), with improvements demonstrated  through win-rates, average rewards, and diversity metrics (dist-n scores). 

Clarity 
- The paper is well-written with clear motivation and effective visualizations (Figure 1).

Significance
- Figure 2 provides evidence that GEB encourages sampling in low-probability reference regions,  validating the theoretical claims.  
- GEB is shown to be integrable into iterative online RLHF without additional sampling cost.

### Weaknesses
1. Theorem 4.2's optimism condition requires Equation 12 to hold, and $u > \alpha$, but the paper  provides no empirical verification that these conditions hold during training for the proposed $u$ designs.  
2. All experiments use only three online iterations following prior work, which may not reveal long term dynamics or potential failure modes.  
3. The paper does not report computational overhead relative to baselines - how much wall-clock  time does the additional bonus computation and gradient calculations require? 
4. The distinction between the three GEB variants ($u = 1 + \alpha - \pi, 1/\pi, arctanh(1 - \pi) +  \alpha)$ is insufficiently explored - why do they perform differently, and how should practitioners  choose among them? 
5. The analysis of $\mathcal{k}$ selection (Figure 3) reveals performance is sensitive to this hyperparameter, with suitable ranges varying across tasks and divergences. Yet, no principled  automatic selection method is provided beyond empirical search. 
6. The evaluation focuses exclusively on language model alignment; applicability to other RLHF  domains (robotic control, game playing) remains unexplored.
7. Some of the gains reported in Tables 3 and 4 are marginal. This is possibly a task/training issue. Choosing a domain where the reference model/policy converges on a local suboptimum and GEB performing exploration to converge on the optimal solution might be a better demonstration of GEB empirical success.

### Questions
1. How should practitioners choose among the three GEB variants ($u = 1 + \alpha - \pi, 1/\pi, arctanh(1 - \pi) + \alpha$)? Can the authors provide guidance based on task characteristics or on the choice of divergence metric?

2. Have the authors evaluated the performance of GEB beyond language model alignment, such as robotic control or game playing, especially in sparse-reward tasks where optimism-based exploration is generally helpful?

### Soundness
3

### Presentation
3

### Contribution
2

# A Non-Linear Ranking Surrogate based Stochastic Bandits for top-m arm Selection

- Decision: Reject
- Scores: 2, 2, 6

## Abstract
The top-m arm selection problem has multiple applications, particularly in example selection for enhancing in-context learning in Large Language Models (LLMs). Existing approaches assume a linear relationship between features and rewards, which limits their ability to capture the complex reward landscapes induced by LLMs. 
Moreover, they typically perform static task-level selection, choosing subsets once offline, which can fail to generalize to unseen queries. This motivates the need for learning a surrogate that can be employed, for instance-level ranking of exemplar subsets. To address these challenges, we formulate the top-m arm selection as a learning to rank problem and  propose GRASS (Gap-indexed bandits with {RA}nking-based non-linear {Surrogate} for {S}election). It is  a novel gap-index bandit framework with non-linear differential sorting  based surrogate to model the scores of the example subsets (arms) for the top-m arm (example subset) selection problem. The non-linear surrogate is learned offline using gap-index framework with challenger arm sampling to clearly distinguish borderline arms in a fixed-confidence setting and also provides top-m examples. Hence, it can be used in a task-level or instance-level setting. GRASS is as sample-efficient as the linear bandit variants, while providing performance gains of 9.4-15.2\% in smaller open-source LLMs while converging faster (2.35 x) than existing state-of-the-art approaches.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper considers the problem of exemplar selection for in-context learning (ICL) for large language models (LLMs). Authors formulate it as a top-m arm identification problem in a multi-armed bandit framework. Importantly, an arm in their setting is something complex: it is a set of exemplars for the prompt of LLM. The authors found that existing approaches assume linear reward relationships, which are insufficient for capturing complex, non-linear dependencies of the answer of a neural network (LLM) on the characteristics of the exemplars. To overcome this, they develop GRASS, a novel gap-index bandit method with a non-linear ranking surrogate trained offline.

### Strengths
TBD: strong point will be discussed when the conceptual problem on the formulation is resolved.

### Weaknesses
I see a deep conceptual drawback in the problem formulation of this paper. In a nutshell, the formulation follows a previous work, “Sample Efficient Demonstration Selection for In-Context Learning” by Purohit et al. Unfortunately, I need to continue with a closer examination of that previous paper, since there are some tricky points I cannot grasp unless the authors broaden my understanding. In the Problem Setup section (3.1), the initial problem is correctly formulated as  $\arg \max_U$ … (Equation 2), where $U$ is a set of $m$ arms (sets), which (in my mind) should COVER the diversity of the validation set of examples. However, immediately in the next subsection, this problem somehow transforms to a top-m arm selection, where the problem is already $\arg \max_a$, with $a$ being a single arm, which should therefore cover the whole validation set  ON AVERAGE.

Could you elaborate on the connection between the initial (correct) problem in Section 3.1 and Section 3.2 of the paper by Purohit et al.?

To be honest, there are some inaccurate statements in the paper by Purohit et al. which I cannot agree, so I cannot draw conclusions based on their results. For example, they argue in section 3.2 that “the number of arms is exponential in both k and n.” It is surely not true. The number of arms is  $n!/(k! (n-k)!)$, what is exponential in k, and polynomial in n.

### Questions
See Weaknesses above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper introduces GRASS, a novel framework that addresses the top-m arm selection problem, e.g., selecting optimal demonstration subsets for LLM in-context learning. The primary contribution is the integration of a non-linear ranking surrogate based on differentiable sorting within the existing gap-index bandit framework, which explicitly models the complex, non-linear reward, a limitation of prior linear-based approaches. GRASS provides an efficient solution for both static (task-level) and dynamic (instance-level) subset selection, yielding significant performance gains and faster convergence compared to existing state-of-the-art methods.

### Strengths
- The paper introduces a novel non-linear approach for exemplar selection in in-context learning, supported by theoretical analysis on the gap index.
- GRASS significantly improves the performance of in-context learning while yielding comparable efficiency.

### Weaknesses
- The writing could be improved; it's very difficult for readers who are not familiar with multi-arm bandit for LLM in-context learning to understand the approach. I'd suggest putting more details in the problem definition, including step-by-step examples of sample selection with multi-arm bandit, notations, and important definitions such as demonstrations, empirical mean estimator, gap index, rewards, etc.
- If I understand correctly, the main contribution and also the difference between GRASS and CASE is the non-linearity, which is the differentiable sorting model with MLP in Eqn. 2. The novelty seems limited, and the motivation for non-linearity is unclear to me.
- Section 3.3 should have a Remark paragraph explaining the theoretical results and their implication. 
- The paper only conducts experiments on two models, llama 3b and gpt 4o. Evaluating more models will strengthen the claim and effectiveness of GRASS.
- Minor: missing argument in the definition of gap-index (end of Page 5) and format problem at the end of Page 6.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses **top-$m$ subset selection for in-context learning (ICL)** with LLMs, formulated as a fixed-confidence multi-armed bandit problem where each $k$-sized demonstration subset is an arm. The challenge is identifying the best $m$ subsets from an exponentially large space while minimizing expensive LLM queries.

The authors propose **GRASS (Gap-indexed bandits with RAnking-based non-linear Surrogate for Selection)**, which extends gap-index bandits to use a **differentiable sorting neural network** instead of linear surrogates. Prior work (CASE) assumes linear relationships between features and rewards, failing to capture complex interactions between examples in a subset. GRASS trains the neural surrogate online using ranking loss ($-$NDCG), updating it incrementally (one epoch per round) as the bandit algorithm tests arms.

**Key technical contribution**: Adapting gap-index confidence bounds for non-linear surrogates, accounting for MC-dropout variance, SGD stability, and model bias (Theorem 1), with sample complexity guarantees (Theorem 2). The algorithm maintains a top-$m$ set and challenger list, iteratively testing the most ambiguous pairs and retraining the surrogate.

### Strengths
**Originality:** The paper makes a genuine technical contribution by extending gap-index bandits to non-linear surrogates with valid theoretical guarantees. Prior work (CASE) is limited to linear models, which is genuinely restrictive for modeling LLM rewards. The key insight—using differentiable sorting networks within a bandit framework and deriving modified confidence bounds that account for neural network uncertainty (MC-dropout variance, SGD stability, model bias)—is non-trivial. 

**Quality:** The experimental design is solid with evaluation across three diverse datasets (math reasoning, translation) and multiple model sizes. The comparison to relevant baselines is comprehensive, and the inclusion of both static and dynamic settings demonstrates practical versatility.

**Clarity:** The paper is well-structured with clear motivation and intuitive explanations. Algorithm 1 is easy to follow, and the learning-to-rank framing is well-articulated. The problem formulation (Section 3.1) clearly distinguishes static vs. dynamic selection and positions the work relative to prior bandit formulations. 

**Significance:** The problem is practically relevant—ICL example selection is important for LLM deployment, and non-linear modeling is clearly more appropriate than linear assumptions for LLM rewards. The 9-15% improvements on smaller models and 2× speedup over CASE demonstrate real value. The framework's generality (applicable beyond ICL to any subset selection problem) broadens impact. Providing both offline top-$m$ identification and a trained ranking model is useful for practitioners.

### Weaknesses
**Limited Experimental Scope and Missing Ablations:** The experiments are restricted to one primary open-source LLM (Llama3.2-3B) with GPT-4o-mini results relegated to the appendix. This is insufficient to validate the claim that non-linear surrogates are universally better—perhaps the gains are model-specific. It would be interesting to see results on reasoning models.

**Scalability Concerns Not Addressed:** The paper claims dynamic selection has "negligible latency overhead" (Figure 1d, ~few milliseconds), but this analysis is incomplete. Figure 1d only shows average inference time, not how it scales with problem size. For dynamic selection, GRASS must score **all** $\binom{|Q|}{k}$ possible subsets per test query. With $|Q|=100$ and $k=5$, that's ~75 million subsets. Even if the neural network is fast (say, 1ms per forward pass with batching), this becomes prohibitive. The paper never discusses: (1) How many subsets were actually ranked in experiments? (2) What happens when $|Q|$ grows to 1000 or 10,000? (3) Can approximate nearest-neighbor search or pruning strategies reduce this cost? The claim of "efficient instance-level selection" is overstated without addressing exponential growth. 

**Insufficient Comparison to State-of-the-Art ICL Methods:** The baselines are primarily other bandit methods, but the paper ignores recent specialized ICL example selection methods. For instance, retrieval-augmented approaches using dense embeddings  are only briefly mentioned. The comparison to KNN and MMR is superficial—these are implemented as simple baselines without tuning. More sophisticated methods like learned retrievers are missing entirely. The claim that GRASS achieves "state-of-the-art" is therefore not substantiated.

### Questions
see weakness above

### Soundness
3

### Presentation
3

### Contribution
3

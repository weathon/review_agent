# Similarity-Constrained Reweighting for Complex Query Answering on Knowledge Graphs

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Machine learning models for answering complex queries on knowledge graphs estimate the likelihood of answers that are not reachable via direct traversal. Prior work in this area has focused on structured queries whose constraints are expressed in first-order logic. Recent work has proposed to extend such logical constraints with *soft entity constraints*, which require the answer to a query (also known as the *target variable*) to be similar or dissimilar to specified sets of entities.

A natural but unexplored generalization of this extension is to allow specifying similarity constraints not only on the answer to a query, but also on the values assigned to *intermediate variables*, which frequently occur in complex queries. In this work, we study this more general formulation and introduce SCORE: a computationally efficient and interpretable method for incorporating similarity constraints at arbitrary positions of a query. Unlike approaches that rely on deep neural networks, SCORE is based on a lightweight and interpretable score adjustment function that only requires tuning two parameters on a validation set.

Our experiments on a challenging benchmark over three different knowledge graphs demonstrate that on the special case of constraints on the target variable, SCORE is able to incorporate preferences without degrading overall query answering performance, with significantly increased ranking performance over a learned neural baseline. Moreover, SCORE maintains its performance in the more general setting with constraints on intermediate variables.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the new CQA tasks considering the soft constraints over any variables, which extends the constraints for the free variable in previous works. To address this new setting, this paper proposes a new re-weighting method to interfere with the symbolic search based on the new constraints, where this new method is light and has linear complexity involving two parameters. Then the experiments showed that this new method has advantages over the previous method considering the free variable and the naive baseline.

### Strengths
1. Propose a new setting for CQA: consider the additional soft constraints over intermediate variables. 
2. Propose a light and effective method to rerank the answer sets based on the new soft constraints and the results are good.

### Weaknesses
1. In my opinion, for the paper proposing new tasks, it’s key to introduce the motivation and application of this new setting. After reading this manuscript, I know the soft constraints over intermediate variables are new compared with existing methods.  However, I don’t know why to extend this new setting. In real situation, it's hard to prepare such preference set for re-weighting.
2. The presentation of method is not clear. Though this manuscript provide the details of the re-weighting operations, it's lacking for how this operation integrated with the  symbolic search process.  Is the re-weighting  operation applied over each updating for fuzzy vectors?  
3. The compared baselines are limited and all symbolic search methods. It’s important to include more baselines for new task. Query embeddings methods are mainstreams lines for complex query answering, thus I suggest authors can include some classic methods in query embedding methods.

### Questions
Typos:
I found some potential typos in this manuscript and listed them in the following:
1.Equation 1 in Line 147, this formula lacks the existential qualifier \exists based on the semantics of the natural language question. 
2.Equation 3 in Line 161: the symbolic \exist v_{\lneg i} \in \mathcal{E} only is right for two variables. Please consider the general formula for any variables.
3.Line 69: e as a symptom,” a user

### Soundness
3

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
This paper proposes a new method for complex query answering. Unlike traditional settings, it focuses on scenarios where similarity constraints are applied to either intermediate nodes or answer nodes within complex queries. The key idea is to represent the potential answers of each variable as fuzzy sets, and then perform Similarity-Constrained Reweighting to adjust these fuzzy vectors accordingly. Overall, the method is conceptually simple yet addresses an important and underexplored problem.

### Strengths
The paper is clearly written and easy to follow. The writing quality is excellent, and the authors address a novel and meaningful problem—introducing similarity constraints on both intermediate variable nodes and answer nodes in complex query answering.

The inclusion of theoretical analysis enhances the soundness and credibility of the proposed algorithm, while the experimental results convincingly demonstrate its effectiveness across different datasets and settings.

### Weaknesses
The core idea of the paper is simple and intuitive, with the primary contribution being the introduction of the Similarity-Constrained Reweighting mechanism. However, this contribution appears somewhat limited in scope.

The experimental section requires significant improvement for better clarity and completeness. First, although the paper introduces a new problem setting, it does not provide sufficient details on how existing methods (such as SCORE, NQR, and other baselines) were adapted to this new domain. This information is essential and should be described explicitly.

Second, the presentation of experiments is not very clear. Both the experimental setup and the performance analysis are difficult to follow. The section would benefit from a thorough revision to improve organization, explanation, and readability.

### Questions
no

### Soundness
2

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
4

### Summary
This paper proposes a new extension to complex query answering (CQA) over knowledge graphs (KGs), the paper names it similarity-based constraints CQA. The similarity constraint can be applied to any variable within this query. To address this new question, the paper proposes SCORE (Similarity-COnstrained REweighting). It achieves this via a logit-space reweighting mechanism that only contains two new hyperparameters.

### Strengths
1.	It is praiseworthy to introduce the novel generalization of similarity constraints. 
Notably, the extension of SimCQA allows for similarity constraints of intermediate variables. Overall, the paper addresses a realistic but previously unstudied case in CQA.
2.	SCORE has good interpretability. Unlike black-box neural methods, SCORE’s update mechanism is transparent and traceable to individual preference contributions.
3.	The methodology of SCORE is generally easy to comprehend and follows, like its logit-space reweighting mechanism, the paper also proposes some theoretical results to show its soundness.
4.	The code, datasets are provided.

### Weaknesses
1.	The setting of similarity function is far too simple. The paper just uses binary classification of True/False to determine whether one entity is similar to another one in the problem setting. This is quite simple but another problem is bigger: the paper does not explain very clearly how it decides the ground truth of similarity. To my understanding, the paper uses other answers from same query as ground truth which can be very problematic.

2.	The performance is highly dependent on the backbone model as it only introduces two new hyperparameters. I also found that the experimental performance falls behind NQR in pairwise accuracy and also only outperforms very simple baselines like MeanCosine slightly. Therefore, I suspect the effectiveness of SCORE.

### Questions
Perhaps using numerical attribute (which is provided in NELL and other dataset) instead of Boolean similar/not similar is a better alternative. Have you considered that?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a method for complex query answering (CQA) on knowledge graphs, where entity similarity constraints are exploited to guide the solver towards solutions that are consistent with such constraints. In particular, the method extends a previous one in order to exploit an entity similarity constraint defined on arbitrary variables, rather than on only the answer entity (also called target). While the method is rather simple, it turns out that it can achieve a better CQA accuracy with little computational overhead.

### Strengths
I have found the writing and the plots to be extremely clear. The writing slowly introduces concepts when they are needed with many examples and a sufficiently detailed notation. Overall, I have also appreciated the simplicity of the proposed method, as well as the execution of the experiments. From a quick look, it looks like the code can help reproducing all the presented results.

### Weaknesses
Although the CQA benchmarks used in the paper are particularly established in the community, some queries can suffer from link leakage from the training set. For instance, most of 2p complex queries can actually be decomposed into the simplest task--link prediction (1p queries)--if one considers also the training triples at test time. This means that ranking metrics for certain query types are actually inflated. Recently, [A] showed this problem and proposed a new set of much more challenging CQA benchmarks, where all triples in a query are missing in the observed knowledge graph and therefore must be predicted. Evaluating the baselines and the proposed method on these other recent benchmarks could strengthen the value of the obtained empirical results. Although I do not expect a huge difference w.r.t. the presented results, I suggest the authors to evaluate their method on these other datasets as well.

[A] C. Gregucci, B. Xiong, D. Hernández, L. Loconte, P. Minervini, S. Staab, A. Vergari. Is Complex Query Answering Really Complex? ICML 2025.

### Questions
- The definition of similarity-constrained complex queries assumes that there exists a single similarity constraint for one of free variables (e.g., see Eq. 6). Do you think this framework can be further extended to a setting where several of the free variables take part in a similarity constraint?

- What is the "unconstrained" baseline reported in Figure 2? Is it evaluated on the same dataset for SimCQA? I do not understand why the unconstrained baseline performs better than the baseline NQR, which instead takes into account similarity constraints. Could you please clarify this aspect?

### Soundness
4

### Presentation
4

### Contribution
4

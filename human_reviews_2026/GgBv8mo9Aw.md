# When Students Surpass Teachers: Hypergraph-Aware Knowledge Distillation with Spectral Guarantees

- Avg Score: 3.20
- Decision: Reject
- Scores: 6, 0, 4, 2, 4

## Abstract
Many real-world systems involve complex many-to-many relationships naturally represented as hypergraphs, from social networks to molecular interactions. While hypergraph neural networks (HGNNs) have shown promise, existing attention mechanisms fail to handle hypergraph-specific asymmetries between node-to-node, node-to-hyperedge, and hyperedge-to-node interactions, leading to suboptimal structural encoding. We introduce \textbf{CuCoDistill}, a novel framework that challenges fundamental assumptions in knowledge distillation by demonstrating that student models can systematically outperform their teachers through hypergraph-aware adaptive attention with provable spectral guarantees. Our approach features: (1) set-aware attention fusion that handles variable-sized hyperedge sets with approximation error bounds of $\epsilon\sqrt{|\mathcal{V}|}\max_i|\mathcal{E}_i|$; (2) co-evolutionary unified architecture where teacher and student jointly discover structural patterns in a single forward pass; and (3) theoretically-grounded curriculum distillation based on hypergraph spectral properties. We prove that when student's constrained attention aligns with the hypergraph's intrinsic spectral dimension, superior generalization emerges through beneficial regularization. Extensive experiments across nine benchmarks show our students achieve up to 1.8\% higher accuracy than teachers while delivering 6.25× inference speedup and 10× memory reduction, consistently outperforming state-of-the-art methods and establishing new efficiency-performance frontiers for hypergraph learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a knowledge distillation (KD) method for learning on hypergraphs where a compact 'student' and a high-capacity 'teacher' hypergraph neural network are trained together so that they learn from each other at the same time. Through a step-by-step process guided by the hypergraph's spectral properties, the student's constrained attention mechanism acts as a beneficial regulariser, filtering noise and helping both models focus on essential structural patterns. This approach enables the student model to not only be significantly more efficient but to also provably outperform its teacher on large-scale and noisy datasets, challenging traditional assumptions in knowledge distillation.

### Strengths
1.  This work reframes KD as a powerful regularization mechanism, proving both theoretically (Theorem 2) and empirically that a constrained student can systematically generalize better than its unconstrained teacher under specific, predictable conditions (e.g., noisy, feature-redundant data).
2. There are formal guarantees for (i) spectral preservation of the attention mechanism, (ii) convergence under co-evolution + curriculum, and (iii) generalization benefits from curriculum with reduced hypothesis complexity; plus a complexity corollary that explains the measured efficiency. These align tightly with the architectural choices.
3. The paper evaluates on nine diverse hypergraph datasets and provides robustness (feature/structural/label noise), sensitivity analyses (e.g., K-factor, temperature), and scaling studies (time/memory vs. N), which substantiate both accuracy and efficiency claims.

### Weaknesses
1. All main results and ablations target node classification. That limits external validity for hypergraph tasks where higher-order relations matter most (e.g., hyperedge/link prediction, group recommendation, set expansion). Actionable: add at least one hyperedge prediction benchmark (e.g., on DBLP/IMDB subsets) and one inductive split to test transfer. Even a single well-designed hyperedge task would strengthen the “hypergraph-aware” claim.
2. The ablation study in Table 3 is performed on three datasets where the student model is either superior or nearly on par with the teacher. To provide a more complete picture, it would be highly insightful to include an ablation study on a clean, well-structured dataset where the teacher clearly dominates (e.g., CC-Cora or DBLP-Conf). This would help answer a key question: Do components like co-evolutionary training still offer significant benefits (e.g., faster convergence) even when the final student accuracy doesn't surpass the teacher? This would strengthen the case for the framework's general utility beyond the specific 'student-superiority' scenario.

### Questions
1. Under what conditions is the Frobenius-norm approximation bound in Theorem 1expected to be tight in practice?
2. The curriculum combines time-varying quantiles and loss-weight schedules. Which individual component contributes most to stability and performance?
3. In the set-level attention, which elements are learned versus fixed normalization?
4. For dense regimes (e.g., IMDB), what preprocessing affects hyperedge size/degree distributions, and how might this interact with K-sparsification in evaluation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper claims to improving the performance of Hyper-Graph Neural Network by designing the attention mechanism for hyper-graph asymmetries, introducing constrained attention, and creating a co-evolve training mechanism.

### Strengths
None.

### Weaknesses
1. The organization is poor, which is confusing and hard to follow.

2. The writing is poor and some of the key explanations are missed, for example, what is 'structural inductive bias'?

3. The definition of the hyper-graph is confusing. For the hyper-edges, is it denotes the edges between nodes? Or the edges between hyper-nodes?

4. Figure.1 is confusing and hard to understand. There are lots of meaningless texts with emphasis, such as the Unified Backbone. Moreover, the presentation of the data flow is a disaster, which is hard to understand. 

5. The annotations is confusing, e_i and e_j are used as the feature of nodes, however, the e is said to belongs to the edge set in the very initial definition.

6. The proof of Theorem 1 is meaningless. There is no explanation where A_ours comes from and how the bound is computed.

7. There are lots of unexplained variables, such as the w_i in Eq.(10).

8. Lacks of discussion with related work, such as 'Distilling Knowledge from Graph Convolutional Networks. CVPR 2020', which is highly relative with this paper.

### Questions
Please refer to the Weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an improved knowledge distillation framework for better hypergraph learning. The authors first point out some limitations or challenges of existing techniques, including the shortcomings of prior hypergraph attentions and the gap between distillation and hypergraph learning tasks. The proposed distillation framework contains three different parts. Part 1 focuses on improving the hypergraph attention via an adaptive multi-scale fusion (combining node-node and node-hyperedge attentions) to support a more comprehensive knowledge extraction (both global and local interactions). It also solves the variable-size challenge of hyperedges. The authors give a theorem to guarantee that this part can encode intrinsic knowledge of the vanilla hypergraph by bounding the gap between the proposed attention matrix and an ideal one. Part 2 proposes a co-trained teacher-student distillation framework, where the teacher is an attention-based GNN and the student is its dynamic top-k sparse variant. This part incorporates both attention alignment and embedding alignment for better performance. In this part, the authors also provide a theorem, showing that when K is greater than the effective spectral dimension of the vanilla hypergraph, the student can approach the teacher in a large probability. Part 3 incorporates contrastive learning and curriculum learning to further improve the above framework, where the authors use contrastive and distillation gaps to design a “difficulty” score for their curriculum, supporting easy-to-hard learning. The experiments are generally comprehensive, including performance comparisons among nine benchmarks from different domains, ablation studies, teacher-student comparisons, running time and memory comparisons. These experimental results show that the proposed framework can achieve better performance with reduced time and memory costs.

### Strengths
1)	Based on some experimental results, the proposed framework effectively improves the performance of hypergraph learning tasks, with reduced time and memory costs, among some benchmarks from different domains.

2)	The authors provide theorems to show that their framework can learn intrinsic knowledge from the input hypergraph and the student can approach the teacher when K is large, showing that the framework has some theoretical merits.

3)	The presentation of their specific methodology designs is clear (with clear mathematical formulas), which makes their method understandable.

4)	The ablation studies are detailed and comprehensive.

### Weaknesses
1)	The analysis of the ablation studies is missing. Please check Line 265. The submitted manuscript seems incomplete.

2)	The title is misleading. After carefully reading the main text, from my point of view, the central aim of this paper is to propose a framework to improve the hypergraph learning performance, rather than study whether, when, and why the student can surpass the teacher (a critical question in the knowledge distillation domain). Thus, the title of this paper is very misleading, giving a sense that the authors propose a hypergraph-based solution to solve the above-mentioned general question in the knowledge distillation domain. While the authors attempt to discuss the student-teacher relationship regarding hypergraph tasks in this paper, it is not enough to highlight that contribution in the title.

3)	The main motivation is unclear. The proposed framework contains three parts. Part 1 focuses on improving hypergraph attention. Part 2 focuses on a co-trained distillation framework. Part 3 focuses on a contrastive curriculum. From my point of view, Part 2 is directly related to the main aim of this work, since the authors attempt to use distillation to improve hypergraph learning. However, the motivation of incorporating Part 1 and Part 3 into the comprehensive framework is unclear. While they have merits and benefits in performance gain, whether the distillation must rely on them remains confusing. Thus, incorporating them significantly harms the generality and effectiveness of Part 2 and makes the holistic framework heavy. In summary, the three parts focus on different challenges, and do not align with the same main motivation. Besides, regarding distillation itself, why distillation matters in hypergraph learning still requires further elaboration.

4)	The novelty seems limited. First, are the authors the first to introduce distillation (the main idea) to hypergraph learning? Second, the holistic distillation framework has three parts. According to the main text, I see limited novelty in each of them. For example, Part 1 combines local and global knowledge, which seems common in graph transformers. The embedding distillation and attention distillation in Part 2 can also be found in graph transformers or GNNs. Part 3 is interesting in defining a “difficulty” score via gaps for a curriculum. Yet, based on the structure of the paper presentation, it is not a main contribution of this paper. And I think the novelty still needs to be highlighted by contrasting it with some prior existing curriculum designs related to distillation or contrasting learning. In summary, the authors should clearly state which component is novel and highlight it with enough support.

5)	Both co-trained distillation and sequential distillation have their own merits. The authors should clearly point out that the former one requires more memory.

### Questions
See Weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors investigate a knowledge distillation framework for hypergraph neural networks (HNNs).

To this end, the authors introduce CuCoDistill, an attention-based distillation framework for HNNs.

CuCoDistill combines contrastive learning and an attention mechanism to distill the teacher's knowledge to the student model in an effective manner.

Through experiments, the authors demonstrate the effectiveness of the proposed method.

### Strengths
- S1. The authors conduct an in-depth analysis of the hypergraph density, which provides important insight into the use case.

- S2. Embedding the similarity of the teacher model and student model is interesting.

### Weaknesses
- **W1 [Theory]** While the authors present several theoretical results, I think the statements are not formal enough. For instance, what does it mean by structural encoding? Moreover, to my understanding, the attention matrix is a learnable component that is derived from the model output. Then, how can this be used for theoretical analysis, given that the learning process of the attention matrix depends on the model hyperparameters and training configurations?

- **W2 [Research goal]** The authors criticize the limitations regarding the current usage of contrastive learning and attention mechanisms within the HNN domain. However, I cannot understand why the teacher-student-based distillation framework overcomes this limitation. What is the key research question of this work? Is it proposing a new HNN design or proposing a new distillation method? The key research question and its presentation should be further improved.

- **W3 [Baselines]** The method only includes outdated HNNs as baselines, which were published in 2019. The authors need to compare the proposed method with more recent HNNs, such as [1, 2, 3].

- **W4 [Incomplete manuscript]** The writing of Section 3.1 is incomplete

- **[References]**
  - [1] Chien et al., You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks, ICLR 2022
  - [2] Wang et al., Equivariant Hypergraph Diffusion Neural Operators, ICLR 2023
  - [3] Wang et al., From hypergraph energy functions to hypergraph neural networks, ICML 2023

### Questions
See Weakness.

### Soundness
2

### Presentation
2

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
This paper introduces CuCoDistill, a highly novel and complex framework for knowledge distillation (KD) in Hypergraph Neural Networks (HGNNs). The authors address the failure of existing HGNN attention mechanisms to handle hypergraph asymmetries and the limitations of standard KD in preserving higher-order structures . The framework's core innovations include: (1) a hypergraph-aware adaptive attention mechanism with provable spectral guarantees; (2) a unified co-evolutionary architecture where teacher and student models train simultaneously rather than sequentially ; and (3) a spectral curriculum scheduler that dynamically adjusts learning difficulty based on hypergraph properties. The paper theoretically and empirically demonstrates the counter-intuitive finding that, under certain conditions (e.g., noisy datasets), the compressed student model can systematically outperform the larger teacher model .

### Strengths
1. The framework is innovative, particularly its "co-evolutionary" architecture and the theoretical demonstration that a student model can surpass its teacher. 



2. The work is theoretically deep, providing provable guarantees for its attention mechanism (Theorem 1) and formalizing the conditions for student superiority (Theorem 2) , lending rigor to its claims.





3. The empirical results are good, showing state-of-the-art performance, efficiency gains (6.25x speedup, 10x memory reduction), and, crucially, validating the "student surpasses teacher" phenomenon on several large-scale, noisy datasets.

### Weaknesses
1. The framework's complexity is extremely high, potentially hindering reproducibility and adoption. It integrates multiple complex components (multi-scale attention, co-evolution, spectral curriculum, multi-level KD losses ), creating a system that is very difficult to implement and tune.

2. The claimed "student superiority" is highly conditional and not a general outcome. The results clearly show this phenomenon occurs only on large, noisy, or feature-redundant datasets (e.g., DBLP, IMDB, Yelp). On clean, well-structured datasets (e.g., CC-Cora), the teacher model remains superior, a critical nuance that limits the generality of the titular claim.

3. The method introduces a very large number of new hyperparameters. The spectral curriculum (adaptive thresholds, loss weights $\lambda(t)$) , attention mechanism (Top-K $\alpha$) , and various loss component weights create a complex tuning space, even with the sensitivity analysis provided in the appendix.

### Questions
1. The ablation study shows the "Spectral Curriculum" has the smallest individual impact (0.9-1.1%). Given its complexity (calculating dual difficulties, quantile thresholds), is this component truly necessary, or could a simpler regularization suffice?

2. In the t-SNE analysis (Figure 4, ), the student embedding space for DBLP shows a worse silhouette score (0.327) than the teacher (0.614), yet the student model outperforms the teacher on the DBLP task (Table 1). This is counter-intuitive. Could the authors explain why degraded cluster quality in the embedding space leads to better classification accuracy in this case?


3. There is a citation error in the baseline description (Section D.2.2). The text cites (Zhang et al., 2019b) for Hyper-SAGNN but then describes HyGCL-AdT (Qian et al., 2024) . This should be corrected.

### Soundness
3

### Presentation
3

### Contribution
2

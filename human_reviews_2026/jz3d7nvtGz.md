# Any-Subgroup Equivariant Networks via Symmetry Breaking

- Decision: Accept (Poster)
- Scores: 8, 8, 2, 8

## Abstract
The inclusion of symmetries as an inductive bias, known as *equivariance*, often improves generalization on geometric data (e.g. grids, sets, and graphs). However, equivariant architectures are usually highly constrained, designed for symmetries chosen *a priori*, and not applicable to datasets with other symmetries. This precludes the development of flexible, multi-modal foundation models capable of processing diverse data equivariantly. In this work, we build a single model --- the Any-Subgroup Equivariant Network (ASEN) --- that can be simultaneously equivariant to several groups, simply by modulating a certain auxiliary input feature. In particular, we start with a fully permutation-equivariant base model, and then obtain subgroup equivariance by using a symmetry-breaking input whose automorphism group is that subgroup. However, finding an input with the desired automorphism group is computationally hard. We overcome this by relaxing from exact to approximate symmetry breaking, leveraging the notion of 2-closure to derive fast algorithms. Theoretically, we show that our subgroup-equivariant networks can simulate equivariant MLPs, and their universality can be guaranteed if the base model is universal.  Empirically, we validate our method on symmetry selection for graph and image tasks, as well as multitask and transfer learning for sequence tasks, showing that a single network equivariant to multiple permutation subgroups outperforms both separate equivariant models and a single non-equivariant model.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes using networks equivariant under a group H, together with a symmetry breaking input to obtain equivariance under a subgroup G of H. Specifically, H is taken to be the symmetric group on n elements, where n is the dimensionality of the input. Symmetry breaking is obtained by designing edge features that define a graph that has automorphism group $G^{(2)}$, the 2-closure of G. $G^{(2)}$-equivariance is obtained by using a graph network on the constructed colored graph. When G is 2-closed, this yields equivariance precisely under G.

### Strengths
I found the paper well written and engaging. The proposed method is well thought out and deserves to be discussed at the conference in my opinion. Theorem 1 was especially interesting to me, showing that equivariant MLPs can be simulated by the proposed graph networks.

### Weaknesses
The experimental evaluation seems not so convincing to me: 

1. There are no comparisons to equivariant MLPs.
2. In Figure 6 it looks like just training the yellow models longer will allow them to catch up. Is the effective training time here the same for the blue and yellow models or do the blue models train for longer due to training on multiple tasks? 
3. There is an example in the appendix where $G$ is not 2-closed, and then the method does not work.


**Typos:**

I believe that the symmetry group of the intersection operation in table 3 should not be a direct product since permuting the two sets does not commute with permuting internally in one set.

### Questions
1. I am not very familiar with the prior work “Approximately Equivariant Graph Networks” by Huang et al. Is it correct to state that they propose using ordinary MLPs equivariant under the automorphism group of a fixed graph to obtain more expressive networks for that specific graph, whereas the submitted paper does the opposite of proposing to design a graph with a specific automorphism group G to obtain G-equivariant networks through a graph network?
2. In table 1, the lowest error is consistently obtained by the non-equivariant version, which, if I understand correctly, corresponds to a unique identifier for each edge. Is this a standard approach in graph networks on fixed graphs?

As I found the paper interesting I have some potentially more philosophical questions.

3. As far as I understand, the suggested approach requires knowing the orbits of the node pairs under G, but from this information we can also construct a basis for the equivariant linear maps (Ravanbakhsh et al, 2017). So is the following statement in the introduction fair? “(I) equivariant architectures typically require deriving and implementing group-specific equivariant layers, so substantial research and engineering must be done for architectural design whenever a new type of symmetry arises,”
4. A linear $G$ equivariant layer from $\mathbb{R}^n$ to $\mathbb{R}^n$ under the permutation action $\rho(g)$ can be defined from a general linear map $W\in \mathbb{R}^{n\times n}$ through group averaging as $W_\rho = \sum_{g\in G}\rho(g) W \rho(g)^{-1}$. Thus the same $W$ can define equivariant linear layers for different groups $G$ acting on $\mathbb{R}^n$. Is this somehow analogous to how different symmetry breaking objects lead to equivariance under different $G$ for the same message passing layer in the paper?
5. In some sense, the paper suggests an approach that enables the use of graph networks instead of equivariant MLPs. Aren't equivariant MLPs easier to work with, and more aligned with current hardware, due to the focus on linear layers?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a simple framework to build neural networks that are equivariant to any subgroup of a given symmetry group. By conditioning the network on a symmetry-breaking input $v$ whose automorphism $Aut(v)$ equals the desired subgroup, the model naturally inherits the correct equivariance. The approach is general, easy to apply, and shown to work well on several examples involving rotation and reflection subgroups.

### Strengths
- Presents an interesting and elegant idea that is both simple and flexible for constructing subgroup-equivariant networks.

- Provides a solid mathematical foundation, with clear and rigorous proofs of the main results.

- Includes several illustrative examples that effectively clarify the concept and demonstrate its broad applicability.

- Offers a unifying framework that can reproduce or extend many existing equivariant architectures with minimal effort.

### Weaknesses
Overall, I find the paper is convincing and I am not aware of any similar work, therefore the weaknesses below should be regarded as minor:

- Computational efficiency and scalability are not analyzed, leaving open how well the approach performs for larger models or groups with respect to alternative approaches.

- The experiments demonstrate flexibility, but not whether this is the preferred method compared to specialized architectures; it may mainly serve as a prototyping tool.

- The experimental validation is limited and does not convincingly show advantages on more challenging or large-scale tasks.

### Questions
The framework offers flexibility to easily test different equivariances. Do the authors view it mainly as a prototyping system, or do they expect it to be competitive with specialized architectures in practical applications?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents a method to obtain foundational models equivariant to any chosen subgroup of a larger symmetry group while reusing a single backbone. It introduces a fixed, symmetry-breaking hypergraph whose automorphisms match the target subgroup, so the backbone effectively enforces just the chosen symmetries. The authors provide proofs of this property and show that their models inherit universality from a universal base model. They then present experiments evaluating the framework on human pose estimation, traffic flow prediction, and image classification benchmarks, along with synthetic tests that explore multitask and transfer learning.

### Strengths
- The authors tackle a relevant problem: building foundation models equivariant with respect to adaptive symmetries. 
- The proposed solution, using symmetry breaking to realize adaptable group equivariance, is a brilliant idea.

### Weaknesses
1. Although the proposed solution is interesting, the analysis is broad but shallow. The paper aims to span both theory and practice, yet it defers the hardest, and most relevant, questions to future work, even though they are essential to validate the framework as a practical path to adaptive foundation models. In particular:
	- The limited analysis of scalability and computational cost is a critical omission, and should be of paramount importance in the design of foundation models.
	- A thorough analysis of how the $k$-closure approximation affects expressivity, equivariance, and performance is fundamental to assessing the framework’s practical relevance and implementability.
2. Readability is limited: the introduction is technical from the introduction, the exposition is overly compact, and there are too few worked examples to illustrate the core mechanisms.

### Questions
1. How can the presented baselines be scaled to test the scalability potential of the proposed model?
2. In this direction, would it be possible to report scaling laws for this models: how do preprocessing and training costs grow with input size, the number of hypergraph automorphisms, and with $k$?
3. At scale, when does this approach help with respect to a non-equivariant baseline?
4. Which are the groups for which do you envision a particular difference in employing its $2$-closure?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new method for constructing equivariant neural networks by first having a base network equivariant to a larger base group and then fixing a symmetry-breaking input feature which is precisely invariant with respect to the target (sub)group. The authors focus on the specific case of base group being Sn, for which the authors draw upon a classical result that one can always construct a hypergraph of which automorphism group is equal to the target group and propose an approximate scheme that restricts the search space to graphs. The authors prove theoretical sanity-checks on equivariance and expressive power. Through experiments, the authors show that the proposed method can use different symmetries under one architecture and provide a diagnostic insight into the choice of the target group structure, and also show evidence that the proposed method exhibits knowledge sharing and transfer across tasks with different symmetries.

### Strengths
S1. While there has been prior work on architecture agnostic invariance/equivariance and knowledge sharing/transfer across symmetries [1-3], they consider taking an unconstrained base model and adding symmetry constraints through (randomized) symmetry breaking, while this work considers an opposite and original direction of starting at an overly constrained base model and reducing symmetry constraints using a fixed symmetry-breaking input. This is an interesting direction and could facilitate future work.

S2. The paper is very well written and easy to follow.

S3. The proposed method is simple and technically sound as far as I can confirm.

S4. The experimental results support the claims made in the main text.

S5. The shown connections to hypergraphs and higher-order GNNs are interesting and might lead to renewed interests of related literature.

[1] Kim et al., Learning Probabilistic Symmetrization for Architecture Agnostic Equivariance, NeurIPS 2023.

[2] Mondal et al., Equivariant Adaptation of Large Pretrained Models, NeurIPS 2023.

[3] Kim et al., Revisiting Random Walks for Learning on Graphs, ICLR 2025.

### Weaknesses
W1. There are several potential weaknesses stemming from the fact that the work takes an over-constrained base model and relaxes its constraints in downstream tasks. One weaknesses is related to practical implication of Theorem 2. For large groups G such as Sn as considered in this work, constructing universal and equivariant networks is in general quite hard and requires combinatorially large feature spaces [4] or some kind of randomized symmetry breaking themselves [5], making the setup of Theorem 2 unlikely in practice. Another potential weaknesses is that, with the base model overly constrained, the scope of knowledge it can accumulate from pretraining without knowing what downstream tasks are can be substantially limited. As far as I understand, pretraining in Section 5.2.2 is aware of downstream tasks and uses task-specific embedding modules, so it bypasses this problem; but in general, one may not know the details of downstream tasks during pretraining.

[4] Maron et al., On the Universality of Invariant Networks, ICML 2019.

[5] Abboud et al., The Surprising Power of Graph Neural Networks with Random Node Initialization, IJCAI 2021.

### Questions
I don't have particular questions but would like to hear the authors' opinions on W1.

### Soundness
3

### Presentation
4

### Contribution
3

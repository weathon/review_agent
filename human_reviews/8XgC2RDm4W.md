# Graphon Neural Differential Equations and Transferabilty of Graph Neural Differential Equations

- Avg Score: 3.67
- Decision: Reject
- Scores: 5, 3, 3

## Abstract
Graph Neural Differential Equations (GNDEs) extend Graph Neural Networks (GNNs) to a continuous-depth framework, providing a robust tool for modeling complex network dynamics. In this paper, we investigate the potential of GNDEs for transferring knowledge across different graphs with shared convolutional structures. To bridge the gap between discrete and continuous graph representations, we introduce Graphon Neural Differential Equations (Graphon-NDEs) as the continuous limit of GNDEs. Using tools from nonlinear evolution equations and graph limit theory, we rigorously establish this continuum limit and develop a mathematical framework to quantify the approximation error between a GNDE and its corresponding Graphon-NDE, which decreases as the number of nodes increases, ensuring reliable transferability. We further derive specific rates for various graph families, providing practical insights into the performance of GNDEs. These findings extend recent results on GNNs to the continuous-depth setting and reveal a fundamental trade-off between discriminability and transferability in GNDEs.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper considers the transferability of Graph Neural ODEs (G-NODEs) in the sense of how much difference two solutions of G-NODEs derived from different graphs have. First, the Graphon-ODE, a time-continuous version of the Graphon GNN, is formulated, and its well-posedness is shown.
Next, two variants of Graphon-ODE (Models I and II) are considered for a graph sequence made by spatially discretizing a graphon $W$. The approximation error between each solution of the Graphon-ODEs and the solution of Graphon-ODE derived from the original graphon is evaluated. As a corollary, the difference between the solutions of graphon-ODEs derived from two sequence graphs converging to the same graphon was evaluated.
The types of numerical experiments verify the validity of the theoretical analyses:
1. The approximation ability of models learned on small graphs by simulating the heat equation on large graphs.
2. The dependence of the relative error on the box dimension on the CheckerBoard Graphon.
3. Performance of node prediction tasks of models trained with subgraphs on whole graphs using the Cora citation network.

### Strengths
- As far as I know, this is the first study to extend Graph Neural ODEs to graphons.
- The background knowledge on graphons is adequately described, making the paper accessible for readers unfamiliar with the graph limit theory.

### Weaknesses
- I have questions about whether the setup and evaluation methods of the numerical experiments are appropriate to claim that support the correctness of the theoretical analyses.
- I have questions about the interpretation of the experimental results (Table 1).
- The clarity of the paper may be improved. Several similar concepts appear, making it difficult to understand the relationship between them. Also, the description of experimental settings has room for improvement.

See Questions for details.

### Questions
- Theorem 3.2 shows that the solution of a Graphon ODE induced from $\mathcal{G}\_{n}$ that converges to graphon $W$ approximates the Graphon ODE induced from $W$. We cannot take arbitrary sequence $\mathcal{G}\_{n}$ that converges to $W$, but specifically constructs a specific convergence sequence $\mathcal{G}\_{n}$ from graphon $W$ (Models I, II). Also, we need to know $W$ to construct $\mathcal{G}\_{n}$. These assumptions limit the applicability of the theory.
- This paper compares the solution $X_n(t)$ of the GNDE (12) with the solution $X(\cdot, t)$ of the Graphon-CNDE (6) by transforming (12) into the Graphon-CNDE (15), solve (15), and compare its solution $X_n(\cdot, t)$ with that of (6). While this is one way to compare the solutions, since both are Graphon-NODE solutions, I think it is not appropriate to say that it quantifies the approximation error between GNDE and Graphon-ODE, as described in the abstract. Wouldn't it be more natural to regard the solution $X_n(T)$ of (12) as a graphon by interpreting it as a piecewise-constant function?
- What is the definition of the true dynamics $Y_N(1)$ in the numerical experiments? Is it the solution of the heat equation in (18) made by a numerical simulation method? In that case, I have a question about whether it is appropriate to call it *true dynamics* since the simulation is an approximate numerical solution. Also, where does the dependence on $N$ come from?
- In Figure 3, the authors claim that *GNDEs can learn complex physical dynamics on smaller graphs [...] and effectively transfer this knowledge to larger systems [...].* from the fact the relative error is around $10^{-2}$. I have a question about whether this interpretation is appropriate. First, the theoretical bound $O(1/500)$ is an abuse of the O-nation. Second, it needs to be clarified why we can claim the relative error $10^{-2}$ is sufficiently small without comparison with other approximated solutions.
- The meaning of *transferability* seems to differ between theoretical analyses and numerical experiments. If I do not miss any information, the theoretical analyses do not explicitly define transferability. As far as we can infer from Theorem 3.3, it refers to the difference between the outputs of Graphon-NODEs on small and large graphs is small. On the other hand, in the experiments in Section *Transfer Learning of Nonlinear Heat Equations on Complete Weighted Graphs*, transferability means that true dynamics and the model outputs are close. Therefore, whether the numerical experiment results validate the theoretical analyses is questionable.
- How are *the GNDE outputs on the original and subgraphs* calculated in numerical experiments on $\{0, 1\}$-valued Checkerboad graphon? In particular, what do original and subgraph mean, respectively?
- Which test data are used for *Subgraph Accuracy* and *Full Graph Accuracy* in Table 1 respectively? If I do not miss any information, *Subgraph Accuracy* is not mentioned in the text. For *Full Graph Accuracy*, does it correspond to the *test accuracy for the full dataset* in the text?
- The paper draws the following conclusion from Table 1: *As we train on a larger proportion of nodes, we gain accuracy on full graph prediction*. However, I do not think it is appropriate for the following reasons. First, since the standard deviation of the results in Table 1 is large, it is difficult to say that there is a significant difference between Subgraph Accuracy and Full Graph Accuracy. Second, even if we ignore the standard deviation, the Subgraph accuracy value exceeds the full graph accuracy at 20%. As the percentage increases over 20%, the difference between Subgraph accuracy and Full graph accuracy decreases.
- Can we interpret the subsampling method in the Cora experiments as Model I or II in the theoretical analysis? If so, an explanation should be provided for the justification.


**Minor Comments**

- Citations are not appropriate; citep and citet should be used appropriately. For example, *this framework generalizes Neural ODEs Chen et al. (2018)* in Section 1 should be *this framework generalizes Neural ODEs (Chen et al., 2018)*.
- The paper writes that *One of the most prevalent GNN architectures is the GCNs, introduced by Bruna et al. (2013) [...]*. However, the reference Bruna et al. (2013) does not use the term *graph convolution* nor GCN. Also, GCN is considered a proper term for the architecture introduced by Kipf and Welling (2016). Therefore, I think it is more appropriate to write like *GCNs, whose origin can be traced back to Bruna et al. (2013) and popularized by Kipf & Welling (2016).*
- The homomorphism count $\mathrm{hom}(\mathcal{F}, \mathcal{G})$ is undefined.
- The homomorphism density $t(\mathcal{F}, \mathbf{W})$ is undefined for the pair of a graph $\mathcal{F}$ and a graphon $\mathbf{W}$.
- Many similar concepts related to neural ODEs appear, such as GNDE (1), GCNDE (4), Graphon-CNDE (6), and Graphon-NDE induced by GNDE (15). Since it is difficult to understand their relationships at first reading, I would suggest clarifying them by, e.g., making diagrams to show their relationships.
- $G(u)$ is in $L^{\infty}(I; \mathbb{R}^{1\times F})$ in Eq. (6). However, the graphon convolution assumes $L^2$-integrability in the previous section. It should be commented that $G(u)$ is $L^2$ (we can verify it because $L^\infty \subset L^2$ using the fact that $I$ is compact.) Also, rigorously speaking about well-definedness, it should be shown that $X(\cdot, t)$ is in $L^{\infty}$ for any $t$.
- In Theorem 3.1, it should be made clear what IVP stands for (Initial Value Problem?)
- It is not easy to read when formulas are referenced without prefixes (e.g., IVP 6, solution of 15). It is preferable to use a form such as Eq. (6) to clarify that it references a formula.
- P8: *We mention that [...], so in general, AS1 is not satifsfied for Model II.*: AS1 -> AS3?
- It is better to add a reference to Adam.
- *The relative error [...] is shown in Figure 3 (a).*: Figure 3 does not have sub-items such as (a).
- Dormand-Price -> Dormand-Prince

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a study on the transferability of neuro graph ODEs through the perspective of graphons. The key idea is to present conditions for when can transferability happen, based on the concept of graph limits.
The authors define graphon neuro ODEs, and then present several experiments to verify their method.

### Strengths
1.  **Relevance:** The question of the paper is interesting. It is important to know and quantify the transferability of GNNs.

2. **Correness:** The method itself seems to be correct.

### Weaknesses
1. **Related works**: A significant lack of discussion of relevant works is missing in this paper. From neuro ODEs [1-2], to graph neuro ODEs [3-8]. Also, in terms of continuous ODE based GNNs, the authors ignore works like [9-11]. This raises doubts about how thoroughly the study proposed in this paper was done.

2. **Claims with no proofs/references:** In the introduction section (section 1), the authors make major claims such as "Recent advances have introduced Graphon Neural Networks (Graphon-NNs) as limit objects of GNNs, establishing theoretical bounds on the approximation error between GNNs and their corresponding Graphon-NNs. These results reveal a fundamental trade-off between discriminability and transferability." However the authors do not provide references/proofs of these claims. 

3. **Limited use of GNNs**: The authors focus on the GCN architecture, which is limiting and does not show whether the proposed method can work with other graph neural architectures.

4. **Definition of Graphon**: Since this paper revolves around graphons, I find that the late formal introduction of graphons in the paper makes it hard to follow.  

5. **Low quality of presentation:** The paper suffers from an overall lack of quality in terms of its presentation. For example, references are broken in the sense that they do not tell to which element references are (e.g., Equation, Section, etc.). Also, the paper is hard to follow and should be significantly edited to be pleasant to read and easy to follow.

6. **Missing comparison with Graph Transformer:** The authors consider the case of a fully connected graph with edge weights. This is almost identical to Graph Transformers, see [12-15] for examples. I would expect the authors to discuss and compare these methods.

7. **Missing discussion on computational cost, and potentitally high complexity:** The complexity of the method is not discussed in the paper. Morever, to the best of my understanding the method is also built on the use of a fully-connected graph, which makes it very expensive. Can the authors please elaborate?

8.  **Limited and unconvincing experiments:** The experiments provided in this paper are simple and not convincing. That is due to several reasons:
A. The experiment of heat diffusion is rather simplistic, and does not really show that any transferable knowledge was studied. Employing the diffusion equation on different graph sizes will yield the same process, so the result shown here is not surprising.
B. The results on Cora look very low compared to standard results on this dataset (which are usually around 80% accuracy). I understand that the authors use a subgraph of the Cora network to train the GNN, but it is well-known that even simple diffusion can yield strong results on this dataset. Therefore I am not convinced that the results are valid, and code is not provided, so it is hard to understand how the results were obtained.
C. While the experiments in A and B are simple, they are welcome given that they work. However, to show the transferability of learned models, more experiments are required. For example, recent papers on GNNs show multiple benchmarks on graph transferability in [16-18].


**References:**

[1] Stable Architectures for Deep Neural Networks

[2] A Proposal on Machine Learning via Dynamical Systems

[3] GRAND: Graph Neural Diffusion

[4] PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations

[5] GRAND++: Graph Neural Diffusion with A Source Term

[6] GREAD: Graph Neural Reaction-Diffusion Networks

[7] Graph-Coupled Oscillator Networks

[8] Unleashing the Potential of Fractional Calculus in Graph Neural Networks with FROND

[9] Monotone Operator Theory-Inspired Message Passing for Learning Long-Range Interaction on Graphs

[10] Implicit graph neural networks: a monotone operator viewpoint

[11] Long Range Propagation on Continuous-Time Dynamic Graphs

[12] Graph Transformer Networks

[13] Attending to Graph Transformers

[14] Recipe for a General, Powerful, Scalable Graph Transformer

[15] A Generalization of Transformer Networks to Graphs

[16] AnyGraph: Graph Foundation Model in the Wild

[17] GraphAny: A Foundation Model for Node Classification on Any Graph

[18] Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting

### Questions
Please see in the enlisted weaknesses.

### Soundness
3

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces Graphon Neural Differential Equations (Graphon-NDEs) as a continuous-depth extension of Graph Neural Differential Equations (GNDEs) to enhance transferability across graphs with shared convolutional structures. GNDEs generalize Graph Neural Networks (GNNs) to continuous-depth frameworks but face challenges when transferring learned knowledge to larger, structurally similar graphs. The authors propose Graphon-NDEs as continuum limits of GNDEs, enabling a smoother transition between discrete and continuous graph representations. Using tools from dynamical systems theory and graph limit theory, they develop a mathematical framework to quantify the approximation error between GNDEs and their Graphon-NDE counterparts, showing that this error decreases as graph size grows and providing explicit convergence rates for different graph families. Empirical validation on various graph structures, such as complete weighted graphs and checkerboard graphons, supports their theoretical results, demonstrating how structural complexity impacts transferability. This work establishes a foundational approach for scaling GNDEs to larger graphs, advancing the theoretical basis for transferability and generalization in continuous-depth graph models.

### Strengths
- The paper presents a well-structured and theoretically sound framework. Using rigorous mathematical tools, including dynamical systems and graph limit theories, the authors derive error bounds for GNDE and Graphon-NDE approximation, with explicit convergence rates for different graph families

- The paper is generally clear and logically structured, with a coherent flow from the introduction of GNDE limitations to the development of Graphon-NDEs and their potential to mitigate these issues.

### Weaknesses
- The citation format is incorrect; please ensure proper use of \citep for consistency.

- The third and fourth paragraph of the Introduction lacks relevant references to support its arguments. Adding citations here would strengthen the background context.

- Writing needs refinement for clarity. For instance, the contributions are not clearly articulated. It would be helpful to list the contributions as bullet points in the Introduction, aligning them with the theoretical sections.

- Details in notation and references to equations require more precision. For example, IVP 6 in Theorem 3.1 is unclear and could benefit from a clearer explanation or reference.

- I have concerns regarding the results in Theorem 3.3. Specifically, it appears that $b + \epsilon$ could exceed 2, and there is no explicit dependence on $\epsilon$ in the result, which needs clarification.

- The paper raises important theoretical insights but does not sufficiently address their practical relevance. For instance, while Theorem 3.3 introduces box-counting dimension to capture boundary complexity, its real-world implications are less clear. It would be beneficial if the authors included a discussion on how these theoretical bounds might affect practical performance in real applications, such as how varying graph complexity might influence training times, prediction accuracy, or the stability of the Graphon-NDEs in dynamic environments.

### Questions
- Could you expand the empirical validation to include more complex and widely used graph structures, such as scale-free and small-world networks?

- How does the performance of Graphon-NDEs compare with standard GNNs or GNDEs without the Graphon-NDE extension?

-  How does the structural complexity of the graph (e.g., measured by the box-counting dimension) quantitatively affect the transferability and convergence rates in practical scenarios?


- Why does Theorem 3.2 achieve a better bound with fewer assumptions compared to previous works? Could you elaborate on the specific differences in assumptions and techniques that allow for this improvement?

### Soundness
3

### Presentation
2

### Contribution
2

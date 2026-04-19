# Relation-Aware Diffusion for Heterogeneous Graphs with Partially Observed Features

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
Diffusion-based imputation methods, which impute missing features through the iterative propagation of observed features, have shown impressive performance in homogeneous graphs. However, these methods are not directly applicable to heterogeneous graphs, which have multiple types of nodes and edges, due to two key issues: (1) the presence of nodes with undefined features hinders diffusion-based imputation; (2) treating various edge types equally during diffusion does not fully utilize information contained in heterogeneous graphs. To address these challenges, this paper presents a novel imputation scheme that enables diffusion-based imputation in heterogeneous graphs. Our key idea involves (1) assigning a {\it virtual feature} to an undefined node feature and (2) determining the importance of each edge type during diffusion according to a new criterion. Through experiments, we demonstrate that our virtual feature scheme effectively serves as a bridge between existing diffusion-based methods and heterogeneous graphs, maintaining the advantages of these methods. Furthermore, we confirm that adjusting the importance of each edge type leads to significant performance gains on heterogeneous graphs. Extensive experimental results demonstrate the superiority of our scheme in both semi-supervised node classification and link prediction tasks on heterogeneous graphs with missing rates ranging from low to exceedingly high. The source code is available at https://github.com/daehoum1/hetgfd.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new diffusion-based imputation method for heterogeneous graphs to improve classification. The method is applied to each dimension. Given a dimension d, it starts by creating a transition Matrix \tilde{M}^(d), where the rows corresponding to the known attributes are set to 1 in the main diagonal (no transition) and the other rows correspond to the probability of going from one node to another. This probability is proportional to the number of edges existing per type of edge. Once the method converges it generates the pre-imputed matrix which is used to compute the Edge-type-wise homophily ranking.

The Edge-type-wise homophily ranking corresponds to ranking from 1 to r where a value of 1 corresponds to the edge type with the highest homophily. The highest homophily corresponds to the average cosine similarity among nodes of each type of edge standardized by the expected similarity ("calculated from randomly sampled node pairs"). This ranking is used to generate a weight adjacency matrix given by equation 6 (high homophily implies lower weights), which is later used to estimate the pseudo-confidence of a node. 

The pseudo-confidence of a node is given by $\alpha \times$ the shortest path of the node with respect to a node with a known attribute. So, a high pseudo-confidence implies that the node is closer and its value can be trusted. The pseudo-confidence is again used to estimate a new transition matrix to generate the final values in a new iterative process. 

In summary, the first transition matrix determines "the importance of nodes based on their similarity/homophily" and the second transition matrix generates the final attributes based on the imputed importance of the nodes. Then, the imputed features are used for prediction.

### Strengths
The abstract and the introduction define the problem correctly. Even though it seems that the problem will be feature imputation, the problem is clearly defined as the generation of imputed features to improve classification rather than generating unobserved features. 

The method converges thanks to the absorbing state defined by the known nodes. The convergence is also demonstrated in the appendix, but it should be easier to connect it to Markov random models. 

The main contribution of the paper seems to be the inclusion of heterogeneous matrices, which are considered through the summation of the different dimensions.

### Weaknesses
The clarity of the paper is very low. The main idea of the paper can be explained with simple words and the process simplified. For some reason, the authors prefer to use a mathematical explanation instead of a simple algorithm or explanation. Most equations are left for the readers to understand, instead of explaining their meaning and their use in the next steps of the paper. These reduce the readability of the paper. Moreover, the idea of the transition matrix is quite similar to Um et al 2023. So, the paper could be rewritten based on the Um idea, and explaining that the main contribution is the "summation" of the matrices, leaving some technical details to the appendix or the Um paper.

The quality of the paper can also be improved. Besides the complex readability, the authors should try to explain why this method actually works. The paper should state the main difference in comparison to the PCFI method (UM paper). Moreover, the main paper lacks several components, including time complexity, experiment details, and hyperparameter analysis. Even though these parts are included in the Appendix, we are not required to read it for the evaluation of the paper. 

Regarding the time complexity, the method is impossible to apply to a large network. As expected, the time complexity is proportional to |V|^2. 

Based on the final results, it seems that the differences are not statistically significant. This reduces the significance of this method in comparison to PCFI. Even though the hyperparameters analysis is given in the appendix, the main paper does not discuss their values or their influence on the model. Moreover, no training process is given for these values (\alpha and \beta). 

In summary, the paper is complex to read, the most important discussion and experiment details are left in the appendix, and it has a large time complexity.

### Questions
Could you give more details about the time complexity analysis?
Is the size of the network from Table 4 the main connected component?
What is the behavior when you include \alpha and \beta equal to 1?

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
This paper proposes a method for feature imputation in heterogeneous graphs. It is assumed that only some types of nodes may have a certain feature and for some of them the feature value is not given and has to be imputed. The idea is the following. First, virtual features are assigned to nodes for which a particular feature is undefined. Then, preliminary propagation is used to get initial assignments for all the nodes. Using these assignments, homophily of different edge types is defined. Then, these homophily values are used to re-weight the edges for the final propagation of features. In the experiments, it is shown that the proposed approach outperforms the baselines for different missing feature rates.

### Strengths
- The paper is in general well-written and easy to follow;
- The proposed approach, while being relatively simple, shows good results in the experiments;
- The authors conduct extensive ablation studies to demonstrate that different components of their solution are important.

### Weaknesses
- It seems that there are two assumptions needed for the approach to work: 1) the graph should be homophilic (nodes with small graph distances between them have similar features) so that feature propagation works, and 2) features are of the same nature/scale so that equation (5) is meaningful. These limitations should be discussed.
- Some of the components of the proposed solution are designed based on some intuition, but particular formulas are not well motivated (e.g., by models or theoretical analysis). Examples include a particular weighting of the edge types in eq. (2), ranking edge types according to homophily and eq. (6), and equation (8). I acknowledge the ablation studies in Appendix, but still many potential solutions could be used and it would be insightful to understand why a particular form is chosen.
- One of the baselines (PCFI+VF) shows comparable performance in most cases, I think it would be useful to discuss this method and compare its advantages and disadvantages to the proposed solution in more detail.
- While the theoretical result in Appendix A seems relatively straightforward and following from existing results (see the question below), its importance is repeatedly mentioned many times in the text (e.g., lines 78, 125, 206, 247).
- Code for reproducing the results is not provided (if I am not mistaken).

Minor: the first parts of the paper assume that only nodes of a certain type have features. The fact that it is possible to apply the method in a general setup is discussed in L360-362. I advise moving this explanation to earlier in the text.

Typos:
- L143: \citet should be \cite
- L220: one of V_k^d should be V_u^d
- L376: "we apply missing to a feature matrix"
- L1102: "highest.Since"

### Questions
Q1. I could not find in the text the details on how the models are evaluated for link prediction (dataset splits, negative sampling, details for metric computation, etc.) Could you please provide the details or point to a particular part of the text?

Q2. Regarding the theoretical analysis in Appendix A, it is written in line 725 "Here, we prove the case when non-attributed nodes participate in feature diffusion with virtual features." Do virtual features bring additional challenges to the proof? It seems that you treat nodes with missing features and with no features exactly the same (in line 735), so it reduces to the standard scenario.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel imputation method named HetGFD. It is designed for heterogeneous graphs with partially observed features. By utilizing a virtual feature mechanism and introducing edge-type-wise homophily, HetGFD enables feature diffusion across different node types and edge connections. The method shows improvements in semi-supervised node classification and link prediction tasks, even under high missing feature rates.

### Strengths
S1. HetGFD introduces virtual features for nodes with missing attributes. This approach effectively bridges gaps in the feature set, enhancing the model's robustness.

S2. The integration of edge-type-wise homophily allows for a flexible adjustment of edge weights during feature propagation. It improves the quality of the imputed features.

S3. The paper demonstrates the effectiveness of HetGFD through experiments on multiple benchmark datasets. The results show that HetGFD outperforms traditional imputation methods, highlighting its practical applicability in real-world scenarios.

### Weaknesses
W1. The novelty of this paper is limited. The main innovation lies in assigning different propagation weights based on node similarity, which is similar to the k-NN imputation method.

W2. Calculating edge-type weights requires complete pairwise feature distance calculations among all attributed nodes, leading to high computational costs that could limit scalability.

W3. Using incomplete features for similarity calculations makes the results less reliable.

### Questions
Q1. Can the authors provide additional experimental results to show that the proposed method outperforms k-NN? The ablation study suggests similar performance, which raises concerns about its novelty.

Q2. The fine-grained similarity calculation method raises questions. Additional discussions and experimental validations regarding the algorithm's efficiency and scalability would be beneficial. Could the authors also demonstrate the effectiveness of a coarser propagation weight calculation method? For instance, using the difference between the feature means of the specific set of attribute nodes and the overall feature means of all attribute nodes could provide a simpler and more efficient approach. Additionally, Section D3 does not fully outline the model's complexity, particularly regarding similarity and shortest path computations.

Q3. Could the authors provide further motivation or theoretical guarantees, such as lower-bound proofs, to validate the similarity calculation strategy for incomplete features? Additionally, using attention mechanisms or learnable weight matrices for dynamic adjustment of propagation weights for the specific downstream task may be a more effective approach. The authors can further discuss the motivation behind their strategy.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Authors provide the framework that can diffuse missing feature values in the heterogeneous graphs. Authors divide nodes into 3 groups for each feature channel -- nodes with known features, nodes with unknown features, and non-attributed nodes (virtual features) -- and then define the diffusion across those groups along with graph links. In this diffusion process, authors introduce 3 kinds of weights: (1) normalization by the frequency of links, (2) edge-type aware weights based on edge-type homophily, and (3) pseudo-confidence aware weights. After multiple steps of diffusion, the diffused features for attributable nodes are obtained by removing the virtual features of the non-attributed nodes.

Authors conduct experiments for semi-supervised node classification as well as link prediction tasks by applying the diffused features into a common GNN algorithm. From the empirical experimental results, authors provide the superior performance from the proposed method as opposed to the baselines.

### Strengths
- The idea of virtual features to allow the diffusion over heterogeneous graphs is novel and worth studying.
- Authors show the superior performance of the proposed algorithm for structural as well as random missing features. And its gain increases as the missing rate increases.
- The proposed algorithm is straightforward.

### Weaknesses
- There is almost no learning process and the proposed method is closer to the well-designed preprocessing process. Some model learning from data would fit better to the venue.
- There are crucial hyperparameters, but setting them up is decoupled from the task learning. Sensitivity analysis and the proposal for tuning them would be great to have.
- Homophily is assumed for the proposed diffusion process, while the underlying GNN could capture more general interactions. It would be great to have experiments on the non-homophily dataset.

### Questions
- In Eq (5), how can the homophily be defined when the edge type r connects two different types of nodes? For instance, if r connects actor and movie type, then actor node is the non-attributable node for the move type feature. How can we define the similarity here?

- For the link prediction task, applying the diffusion process should be careful since using the links in the test split for the diffusion path is information leakage. Despite this demand for carefulness, it is not addressed in the paper. If authors have not done this way, experiments should be performed in this way. If authors already performed this way, the procedure should be clearly addressed.

- It would be great to see the ablation study such as the impact of convergence (i.e. k steps in diffusion), the importance of edge type ranking , and so on.

### Soundness
3

### Presentation
3

### Contribution
2

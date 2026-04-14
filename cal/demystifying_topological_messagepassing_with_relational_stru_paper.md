

{0}------------------------------------------------

# DEMYSTIFYING TOPOLOGICAL MESSAGE-PASSING WITH RELATIONAL STRUCTURES: A CASE STUDY ON OVERSQUASHING IN SIMPLICIAL MESSAGE-PASSING

\*Diaeldin Taha<sup>1</sup>, \*James Chapman<sup>2</sup>, †Marzieh Eidi<sup>1,3</sup>, †Karel Devriendt<sup>4</sup>, Guido Montúfar<sup>1,2</sup>

<sup>1</sup>Max Planck Institute for Mathematics in the Sciences, Leipzig, Germany

<sup>2</sup>UCLA, CA, USA

<sup>3</sup>Center for Scalable Data Analytics and Artificial Intelligence, Leipzig, Germany

<sup>4</sup>University of Oxford, Oxford, UK

taha@mis.mpg.de, chapman20j@math.ucla.edu, meidi@mis.mpg.de, karel.devriendt@maths.ox.ac.uk, montufar@math.ucla.edu

## ABSTRACT

Topological deep learning (TDL) has emerged as a powerful tool for modeling higher-order interactions in relational data. However, phenomena such as oversquashing in topological message-passing remain understudied and lack theoretical analysis. We propose a unifying axiomatic framework that bridges graph and topological message-passing by viewing simplicial and cellular complexes and their message-passing schemes through the lens of relational structures. This approach extends graph-theoretic results and algorithms to higher-order structures, facilitating the analysis and mitigation of oversquashing in topological message-passing networks. Through theoretical analysis and empirical studies on simplicial networks, we demonstrate the potential of this framework to advance TDL.

## 1 INTRODUCTION

Recent years have witnessed a growing recognition that traditional machine learning, rooted in Euclidean spaces, often fails to capture the complex structure and relationships present in real-world data. This shortcoming has driven the development of geometric deep learning (GDL) (Bronstein et al., 2021) and, more recently, topological deep learning (TDL) (Hajij et al., 2023), for handling non-Euclidean and relational data. TDL, in particular, has emerged as a promising frontier for relational learning, that extends beyond graph neural networks (GNNs). TDL offers tools to capture and analyze higher-order interactions and topological features in complex data and higher-order structures, such as simplicial complexes, cell complexes, and sheaves (Hajij et al., 2023). However, the TDL field is young, and the TDL community has yet many open theoretical and practical questions relating to, e.g., oversquashing and rewiring (research directions 2 and 9 of Papamakarios et al., 2024).

Oversquashing is a challenging failure mode in GNNs, where information struggles to propagate across long paths due to the compression of an exponentially growing number of messages into fixed-size vectors (Alon & Yahav, 2021). This phenomenon has been examined through various perspectives, including curvature (Topping et al., 2022), graph expansion (Banerjee et al., 2022), effective resistance (Black et al., 2023), and spectral properties (Karhadkar et al., 2023). Despite the potential of higher-order message passing architectures—such as simplicial neural networks (Ebli et al., 2020), message passing simplicial networks (Bodnar et al., 2021b), and CW networks (Bodnar et al., 2021a)—there remains a lack of unified frameworks for analyzing and mitigating oversquashing in these settings.

In this paper, we take a first step toward studying oversquashing in TDL by showing that simplicial complexes and their message passing schemes can be interpreted as relational structures, making it

\*Equal contribution.

†Equal contribution.

{1}------------------------------------------------

possible to extend key GNN insights and tools to higher-order message passing architectures. The conceptual framework and theoretical results developed in this paper address pressing questions in the TDL community (e.g., research directions 2 and 9 of Papamakarios et al., 2024).

**Contributions.** Our contributions are threefold:

- **Axiomatic:** We provide a unifying view of simplicial complexes and their message passing schemes through the lens of relational structures.
- **Theoretical:** We introduce *influence graphs* which enable novel extensions of prior graph analyses to higher-order structures, where existing methods for analysis do not apply. We extend graph-theoretic concepts and results on oversquashing to relational structures, analyzing network sensitivity (Lemma 3.2), local geometry (Proposition 3.4), and the impact of network depth (Theorem 3.5) and hidden dimensions (Section 3.4).
- **Practical:** We propose a heuristic to extend oversquashing-mitigation techniques from graph-based models to relational structures.

**Related Work.** Our work sits at the intersection of graph neural networks, topological deep learning, relational learning, and the study of oversquashing and graph rewiring in graph neural networks. We review related work in Appendix A.

The rest of this paper is organized as follows: Section 2 provides the axiomatic groundwork for relating simplicial and relational message passing. Section 3 presents our theoretical analysis of oversquashing in this context. Section 4 introduces a heuristic rewiring strategy to mitigate oversquashing in relational message passing. Section 5 presents our experimental results, followed by a discussion and conclusions in Section 6.

## 2 SIMPLICIAL COMPLEXES ARE RELATIONAL STRUCTURES

In this section, we reinterpret simplicial complexes and message passing through the lens of *relational structures*. We begin by recalling simplicial complexes and a representative simplicial message passing scheme, then reframe these notions within a relational framework. We illustrate the definitions in this section with a small worked example in Appendix H. We note that the connection we establish here extends to other higher-order structures, such as cellular complexes (Hansen & Ghrist, 2019; Bodnar et al., 2021a; Giusti et al., 2024).

### 2.1 SIMPLICIAL COMPLEXES AND MESSAGE PASSING

Simplicial complexes are mathematical structures that generalize graphs to higher dimensions, capturing relationships among vertices, edges, triangles, and higher-dimensional objects.

**Definition 2.1** (Simplicial Complex, Nanda, 2021). *Let  $V$  be a non-empty set. A simplicial complex  $\mathcal{K}$  is a collection of non-empty subsets of  $V$  that contains all the singleton subsets of  $V$  and is closed under the operation of taking non-empty subsets.*

A member  $\sigma = \{v_0, v_1, \dots, v_d\} \in \mathcal{K}$  with cardinality  $|\sigma| = d + 1$  is called a *d-simplex*. Geometrically, 0-simplices are vertices, 1-simplices are edges, 2-simplices are triangles, and so on.

**Definition 2.2** (Boundary Incidence Relation). *We say that  $\tau$  covers  $\sigma$ , written  $\sigma \prec \tau$ , iff  $\sigma \subsetneq \tau$  and there is no  $\delta \in \mathcal{K}$  such that  $\sigma \subsetneq \delta \subsetneq \tau$ .*

The incidence relations from Definition 2.2 can be used to construct four types of (local) adjacencies.

**Definition 2.3.** *Consider a simplex  $\sigma \in \mathcal{K}$ . Four types of adjacent simplices can be defined:*

1. Boundary adjacency:  $\mathcal{B}(\sigma) = \{\tau: \tau \prec \sigma\}$ ;
2. Co-boundary adjacency:  $\mathcal{C}(\sigma) = \{\tau: \sigma \prec \tau\}$ ;
3. Lower adjacency:  $\mathcal{N}_1(\sigma) = \{\tau: \exists \delta \text{ such that } \delta \prec \tau \text{ and } \delta \prec \sigma\}$ ;
4. Upper adjacency:  $\mathcal{N}_\uparrow(\sigma) = \{\tau: \exists \delta \text{ such that } \tau \prec \delta \text{ and } \sigma \prec \delta\}$ .

In Figure 1, we illustrate an example of a simplicial complex and its adjacency relations.

We now, following Bodnar et al. (2021b, Section 4), review a general scheme for message passing on simplicial complexes. In Appendix A, we provide references for topological message passing architectures that fit this scheme. We refer readers to Appendix F.5 for specific instantiations of this scheme in our graph and topological message passing models.

{2}------------------------------------------------

![Figure 1: (a) Simplicial Complex K: A graph with 5 vertices (0, 1, 2, 3, 4). Vertex 0 is connected to 1, 1 to 2, and 2 is part of a triangle with 3 and 4. (b) Adjacency Relations on K: A Hasse diagram showing relations between simplices. Vertices are {0}, {1}, {2}, {3}, {4}. Edges are {0,1}, {1,2}, {2,3}, {2,4}, {3,4}. A triangle is {2,3,4}. Arrows represent boundary, co-boundary, lower, and upper relations.](1b7d539e02a202c2cf2d97698b911447_img.jpg)

(a) Simplicial Complex  $\mathcal{K}$ 
(b) Adjacency Relations on  $\mathcal{K}$

→ Boundary  
 → Co-boundary  
 → Lower  
 → Upper

Figure 1: (a) Simplicial Complex K: A graph with 5 vertices (0, 1, 2, 3, 4). Vertex 0 is connected to 1, 1 to 2, and 2 is part of a triangle with 3 and 4. (b) Adjacency Relations on K: A Hasse diagram showing relations between simplices. Vertices are {0}, {1}, {2}, {3}, {4}. Edges are {0,1}, {1,2}, {2,3}, {2,4}, {3,4}. A triangle is {2,3,4}. Arrows represent boundary, co-boundary, lower, and upper relations.

Figure 1: The left panel shows a simplicial complex  $\mathcal{K}$  consisting of five vertices, five edges, and one triangle. The right panel shows the corresponding adjacency relations depicted as arrows to each simplex  $\sigma \in \mathcal{K}$  emanating from each of its adjacent simplices in  $\mathcal{B}(\sigma)$ ,  $\mathcal{C}(\sigma)$ ,  $\mathcal{N}_\downarrow(\sigma)$ ,  $\mathcal{N}_\uparrow(\sigma)$ .

Simplicial message passing extends graph message passing from pairwise node connections to higher-dimensional adjacency connections between simplices. Message passing schemes on simplicial complexes iteratively update feature vectors assigned to simplices by exchanging messages between adjacent simplices. For a simplicial complex  $\mathcal{K}$ , we denote the feature vector of a simplex  $\sigma \in \mathcal{K}$  as  $\mathbf{h}_\sigma \in \mathbb{R}^p$ . At each iteration (layer)  $t$ , the feature vectors  $\mathbf{h}_\sigma^{(t)}$  of simplices  $\sigma \in \mathcal{K}$  are updated by aggregating messages from adjacent simplices. For a simplex  $\sigma \in \mathcal{K}$ , the messages passed from adjacent simplices are defined as follows:

$$\begin{aligned}
 \mathbf{m}_B^{(t+1)}(\sigma) &= \text{AGG}_{\tau \in \mathcal{B}(\sigma)} (M_B(\mathbf{h}_\sigma^{(t)}, \mathbf{h}_\tau^{(t)})), \\
 \mathbf{m}_C^{(t+1)}(\sigma) &= \text{AGG}_{\tau \in \mathcal{C}(\sigma)} (M_C(\mathbf{h}_\sigma^{(t)}, \mathbf{h}_\tau^{(t)})), \\
 \mathbf{m}_\downarrow^{(t+1)}(\sigma) &= \text{AGG}_{\tau \in \mathcal{N}_\downarrow(\sigma)} (M_\downarrow(\mathbf{h}_\sigma^{(t)}, \mathbf{h}_\tau^{(t)})), \\
 \mathbf{m}_\uparrow^{(t+1)}(\sigma) &= \text{AGG}_{\tau \in \mathcal{N}_\uparrow(\sigma)} (M_\uparrow(\mathbf{h}_\sigma^{(t)}, \mathbf{h}_\tau^{(t)})).
 \end{aligned} \tag{1}$$

Here AGG is an aggregation function (e.g., sum or mean), and  $M_B$ ,  $M_C$ ,  $M_\downarrow$ ,  $M_\uparrow$  are message functions (e.g., linear or MLP). Then, an update operation UPDATE (e.g., MLP) incorporates these four different types of incoming messages:

$$\mathbf{h}_\sigma^{(t+1)} = \text{UPDATE} \left( \mathbf{h}_\sigma^{(t)}, \mathbf{m}_B^{(t+1)}(\sigma), \mathbf{m}_C^{(t+1)}(\sigma), \mathbf{m}_\downarrow^{(t+1)}(\sigma), \mathbf{m}_\uparrow^{(t+1)}(\sigma) \right). \tag{2}$$

Finally, after the last iteration, a read-out function is applied to process the features to perform a desired task, such as classification or regression.

### 2.2 RELATIONAL STRUCTURES AND MESSAGE PASSING

We model simplicial complexes and the above message passing scheme using *relational structures*.

**Definition 2.4** (Relational Structure, Hodges, 1993). A relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$  consists of a finite set  $\mathcal{S}$  of entities, and relations  $R_i \subseteq \mathcal{S}^{n_i}$ , where  $n_i$  is the arity of  $R_i$ .

We note that modeling simplicial complexes as relational structures generalizes a powerful perspective in which simplicial complexes and similar constructs are treated as augmented Hasse diagrams as demonstrated, for example, by Hajj et al. (2023), Eitan et al. (2024), and Papillon et al. (2024).

We now introduce a general scheme for message passing on relational structures which encompasses the simplicial message passing scheme from Section 2.1. This scheme is an extension of the relational graph convolution model from Schlichtkrull et al. (2018) which allows for relations of different arities, not just binary relations.

**Definition 2.5** (Relational Message Passing Model). A relational message passing model on a relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$  consists of:

- Feature vectors:  $\mathbf{h}_\sigma^{(t)} \in \mathbb{R}^{p_t}$  for each  $\sigma \in \mathcal{S}$  at layer  $t \geq 0$ , initialized as  $\mathbf{h}_\sigma^{(0)} = \mathbf{x}_\sigma$  (input features). Here,  $p_t$  denotes the dimensionality of the feature vectors at layer  $t$ .

{3}------------------------------------------------

- Message functions  $\psi_i^{(t)}: \mathbb{R}^{p_i} \times \dots \times \mathbb{R}^{p_i} \rightarrow \mathbb{R}^{p_{i,t}}$  (with  $n_i$  arguments) for each relation  $R_i$ , where  $i = 1, \dots, k$ . Each message function takes  $n_i$  input feature vectors (corresponding to the target simplex and its  $n_i - 1$  related simplices) and outputs a message vector of dimension  $p_{i,t}$ . The parameter  $n_i$  represents the arity of the relation  $R_i$ .
- Update function  $\phi^{(t)}: \mathbb{R}^{p_{1,t}} \times \dots \times \mathbb{R}^{p_{k,t}} \rightarrow \mathbb{R}^{p_{t+1}}$ . The output dimension  $p_{t+1}$  specifies the dimensionality of the feature vectors at layer  $t + 1$ .
- Shift operators  $\mathbf{A}^{R_i} \in \mathbb{R}_{\geq 0}^{|\mathcal{S}|^{n_i}}$  associated with each relation  $R_i$ , for  $i = 1, \dots, k$ . For each  $\sigma \in \mathcal{S}$  and  $\xi = (\xi_1, \dots, \xi_{n_i-1}) \in \mathcal{S}^{n_i-1}$  with  $(\sigma, \xi) \in R_i$ , the element  $\mathbf{A}_{\sigma, \xi}^{R_i}$  represents the strength of the signal passed from  $\xi$  to  $\sigma$ . More specifically, for any combination of entities  $(\zeta_1, \zeta_2, \dots, \zeta_{n_i}) \in \mathcal{S}^{n_i}$  where the relation  $R_i$  does not hold among the entities  $\zeta_1, \zeta_2, \dots, \zeta_{n_i}$  (i.e.,  $(\zeta_1, \zeta_2, \dots, \zeta_{n_i}) \notin R_i$ ), the tensor  $\mathbf{A}^{R_i}$  satisfies  $\mathbf{A}_{\zeta_1, \zeta_2, \dots, \zeta_{n_i}}^{R_i} = 0$ .

The update rule is given by:

$$\mathbf{h}_\sigma^{(t+1)} = \phi^{(t)} \left( \mathbf{m}_{\sigma, 1}^{(t)}, \dots, \mathbf{m}_{\sigma, k}^{(t)} \right), \quad (3)$$

where for each  $i = 1, \dots, k$ , the message  $\mathbf{m}_{\sigma, i}^{(t)}$  received by  $\sigma$  over  $R_i$  is computed as:

$$\mathbf{m}_{\sigma, i}^{(t)} = \sum_{\xi \in \mathcal{S}^{n_i-1}} \mathbf{A}_{\sigma, \xi}^{R_i} \psi_i^{(t)} \left( \mathbf{h}_\sigma^{(t)}, \mathbf{h}_{\xi_1}^{(t)}, \dots, \mathbf{h}_{\xi_{n_i-1}}^{(t)} \right). \quad (4)$$

**Remark 2.6.** The shift operators in Definition 2.5 extend the definition of graph shift operators (Mateos et al., 2019; Gama et al., 2020; Dasoulas et al., 2021) from graphs to relational structures. Whereas relations indicate whether entities are connected, shift operators numerically encode these connections with weights.

In the context of message passing, a simplicial complex  $\mathcal{K}$  can be viewed as a relational structure  $\mathcal{R}(\mathcal{K}) = (\mathcal{S}, R_1, \dots, R_5)$ , where  $\mathcal{S} = \mathcal{K}$  are the entities, and  $R_i$  are relations defined as follows:  $R_1 = \{(\sigma) : \sigma \in \mathcal{K}\}$  (identity),  $R_2 = \{(\sigma, \tau) : \sigma \in \mathcal{K}, \tau \in \mathcal{B}(\sigma)\}$  (boundary),  $R_3 = \{(\sigma, \tau) : \sigma \in \mathcal{K}, \tau \in \mathcal{C}(\sigma)\}$  (co-boundary),  $R_4 = \{(\sigma, \tau, \delta) : \sigma \in \mathcal{K}, \tau \in \mathcal{N}_+(\sigma), \delta = \sigma \cap \tau\}$  (lower adjacency),  $R_5 = \{(\sigma, \tau, \delta) : \sigma \in \mathcal{K}, \tau \in \mathcal{N}_+(\sigma), \delta = \sigma \cup \tau\}$  (upper adjacency). The message functions  $\psi_i^{(t)}$  correspond to  $M_B, M_C, M_L, M_U$ , the update function  $\phi^{(t)}$  to UPDATE, and aggregation uses shift operators  $\mathbf{A}^{R_i}$ . This establishes an equivalence between message passing on the simplicial complex  $\mathcal{K}$  and the relational structure  $\mathcal{R}(\mathcal{K})$ .

**Remark 2.7.** The relational message passing scheme in Definition 2.5 encompasses relational graph neural networks (Schlichtkrull et al., 2018), simplicial neural networks (Bodnar et al., 2021b), higher-order graph neural networks (Morris et al., 2019), and cellular complex neural networks (Bodnar et al., 2021a). We demonstrate how higher-order graphs fit the relational framework in Appendix G.

#### Takeaway Message 1 (Axiomatic)

Simplicial complexes can be represented as *relational structures*, where the entities are simplices, and the relations capture the adjacency among simplices of different dimensions. Simplicial message passing is an instance of relational message passing on these structures.

## 3 OVERSQUASHING IN RELATIONAL MESSAGE-PASSING

The existing literature on oversquashing in GNNs does not directly address relational message passing. In this section, we address that gap by deriving new extensions of key results on oversquashing in GNNs to relational message passing. We illustrate the definitions in this section with a small worked example in Appendix H.

In our analysis of relational structures and message passing schemes, we naturally encounter matrices and graphs that capture the aggregated influence of the underlying shift operators. For convenience, we introduce notation for these matrices and graphs. For each relation  $R_i$  of arity  $n_i$  with

{4}------------------------------------------------

shift operator  $\mathbf{A}^{R_i}$ , we define the matrix  $\tilde{\mathbf{A}}^{R_i} \in \mathbb{R}_{\geq 0}^{|\mathcal{S}| \times |\mathcal{S}|}$  as:

$$\tilde{\mathbf{A}}_{\sigma, \tau}^{R_i} = \sum_{j=1}^{n_i-1} \sum_{\xi \in \mathcal{S}^{n_i-2}} \mathbf{A}_{\sigma, \xi_1, \dots, \xi_{j-1}, \tau, \xi_j, \dots, \xi_{n_i-2}}^{R_i}, \quad \sigma, \tau \in \mathcal{S}. \quad (5)$$

This matrix captures all possible ways an entity  $\tau$  can influence entity  $\sigma$  via the relation  $R_i$ . Specifically, it sums over all positions  $j$  where  $\tau$  can appear among the arguments of the shift operator  $\mathbf{A}^{R_i}$ , and over all possible combinations of the other entities  $\xi$ .

We aggregate these matrices over all relations to form the *aggregated influence matrix*  $\tilde{\mathbf{A}} \in \mathbb{R}_{\geq 0}^{|\mathcal{S}| \times |\mathcal{S}|}$ :

$$\tilde{\mathbf{A}} = \sum_{i=1}^k \tilde{\mathbf{A}}^{R_i}. \quad (6)$$

Next, we define the *augmented influence matrix*  $\mathbf{B}$ , which plays the role of an augmented adjacency matrix in our analysis:

$$\mathbf{B} = \gamma \mathbf{I} + \tilde{\mathbf{A}}, \quad (7)$$

where  $\gamma = \max_{\sigma} \sum_{\xi \in \mathcal{S}^{n_i-1}} \tilde{\mathbf{A}}_{\sigma, \xi}$  is the maximum row sum of  $\tilde{\mathbf{A}}$ .

Lastly, we introduce graphs that capture the aggregated message passing dynamical structure implied by the relational structure and the message passing scheme.

**Definition 3.1 (Influence Graph).** *Given a relational structure  $\mathcal{R} = (S, R_1, \dots, R_k)$  and a relational message passing scheme with update rule given by Equation 3, and given  $\mathbf{Q} \in \{\tilde{\mathbf{A}}, \mathbf{B}\}$ , where  $\tilde{\mathbf{A}}$  and  $\mathbf{B}$  are defined by Equations 6 and 7 respectively, we define the influence graph  $\mathcal{G}(\mathcal{S}, \mathbf{Q}) = (\mathcal{S}, \mathcal{E}, w)$  as follows: The set of entities (i.e., nodes) is  $\mathcal{S}$ . The set of edges  $\mathcal{E}$  consists of directed edges from entity  $\tau$  to entity  $\sigma$  for each pair  $(\sigma, \tau) \in \mathcal{S} \times \mathcal{S}$  with  $\mathbf{Q}_{\sigma, \tau} > 0$ . Each edge from  $\tau$  to  $\sigma$  is assigned a weight  $w_{\tau \rightarrow \sigma} = \mathbf{Q}_{\sigma, \tau}$ .*

As we will see next, these graphs make it possible to leverage and extend graph-theoretic concepts, results, and intuition to understand and analyze the behavior of relational message passing schemes.

### 3.1 SENSITIVITY ANALYSIS

We now analyze the sensitivity of relational message passing to changes in the input features. This analysis is crucial for understanding how information propagates through the network and for identifying potential bottlenecks or oversquashing effects. We begin with a standard assumption about the boundedness of the Jacobians of the message and update functions.

**Assumption 1 (Bounded Jacobians).** *All message functions  $\psi_i^{(\ell)}$  and update functions  $\phi^{(\ell)}$  are differentiable with bounded Jacobians: There exist constants  $\beta_i^{(\ell)}$  and  $\alpha^{(\ell)}$  such that  $\|\partial \psi_i^{(\ell)} / \partial \mathbf{h}_{\sigma}\|_1 \leq \beta_i^{(\ell)}$  for any input feature vector  $\mathbf{h}_{\sigma}$ , and  $\|\partial \phi^{(\ell)} / \partial \mathbf{m}_j\|_1 \leq \alpha^{(\ell)}$  for any message input  $\mathbf{m}_j$ . We write  $\beta^{(\ell)} = \max_i \beta_i^{(\ell)}$ .*

Our main result on sensitivity is the following, which is a novel extension of GNN sensitivity analysis results (e.g., Topping et al., 2022, Lemma 1 and Di Giovanni et al., 2023, Theorem 3.2) to relational (and topological) message passing. We provide the proof in Appendix C.1.

**Lemma 3.2 (Sensitivity Bound for Relational Message Passing).** *Consider a relational structure  $\mathcal{R} = (S, R_1, \dots, R_k)$  with update rule given by Equation 3 and satisfying Assumption 1. Then, for any  $\sigma, \tau \in \mathcal{S}$  and  $t > 0$ , the Jacobian at layer  $t$  with respect to the input features ( $t = 0$ ) satisfies*

$$\left\| \frac{\partial \mathbf{h}_{\sigma}^{(t)}}{\partial \mathbf{h}_{\tau}^{(0)}} \right\|_1 \leq \left( \prod_{\ell=0}^{t-1} \alpha^{(\ell)} \beta^{(\ell)} \right) (\mathbf{B}^t)_{\sigma, \tau}. \quad (8)$$

Thus, the bound on the Jacobian of the  $\sigma$ -feature with respect to the input  $\tau$ -feature depends on the  $(\sigma, \tau)$ -entry of the  $t$ -th matrix power  $\mathbf{B}^t$ , which reflects the number and strength of  $t$ -length paths from  $\tau$  to  $\sigma$  in the graph  $\mathcal{G}(\mathcal{S}, \mathbf{B})$ . Structural properties of  $\mathcal{G}(\mathcal{S}, \mathbf{B})$  that lead to small values of  $(\mathbf{B}^t)_{\sigma, \tau}$ , such as bottlenecks or long distances between nodes, therefore contribute to the phenomenon of oversquashing, where the influence of distant entities is diminished.

{5}------------------------------------------------

As demonstrated throughout this work, our result offers a systematic framework for extending other theoretical findings on oversquashing in graphs, which do not directly apply to simplicial complexes and similar relational structures. This includes the influential works by Topping et al. (2022), Di Giovanni et al. (2023), and Fesser & Weber (2023). By leveraging our axiomatic framework, we derive principled extensions on the impact of local geometry (Section 3.2), depth (Section 3.3), and hidden dimensions (Section 3.4) in higher-order message passing, addressing settings where prior results are not applicable. Additionally, this result offers a clear approach for deriving analogs of key quantities such as curvature (Definition 3.3), and can serve as a guide for future work.

### 3.2 THE IMPACT OF LOCAL GEOMETRY

Lemma 3.2 shows that the entries of the matrix  $\mathbf{B}^t$ , which encode the number and strength of connections in a relational message passing scheme, control feature sensitivity. Prior works relate similar bounds to notions of discrete curvature for unweighted undirected graphs, such as balanced Forman curvature (Topping et al., 2022), Ollivier-Ricci curvature (Nguyen et al., 2023), and augmented Forman curvature (Fesser & Weber, 2023), via counting local motifs, such as triangles and squares, in the underlying graphs. Following this approach, we derive a result analogous to Fesser & Weber (2023, Proposition 3.4), introducing a motif-counting quantity inspired by the augmented Forman curvature, but adapted for the particular weighted directed graphs arising in our setting.

**Definition 3.3.** Let  $\mathcal{G} = (\mathcal{S}, \mathcal{E}, w)$  be a weighted directed graph with entities (nodes)  $\mathcal{S}$ , edges  $\mathcal{E}$ , and edge weights  $w: \mathcal{E} \rightarrow \mathbb{R}_{\geq 0}$ . For each entity  $\tau \in \mathcal{S}$ , define the weighted out-degree  $w_\tau^{\text{out}} = \sum_{(\tau \rightarrow \sigma) \in \mathcal{E}} w_{\tau \rightarrow \sigma}$  and the weighted in-degree  $w_\tau^{\text{in}} = \sum_{(\sigma \rightarrow \tau) \in \mathcal{E}} w_{\sigma \rightarrow \tau}$ . For an edge  $(\tau \rightarrow \sigma) \in \mathcal{E}$ , define the weighted triangle count  $w_T = \sum_{\xi \in \mathcal{S}} w_{\tau \rightarrow \xi} \cdot w_{\xi \rightarrow \sigma}$  and the weighted quadrangle count  $w_F = \sum_{\xi_1, \xi_2 \in \mathcal{S}} w_{\tau \rightarrow \xi_1} \cdot w_{\xi_1 \rightarrow \xi_2} \cdot w_{\xi_2 \rightarrow \sigma}$ . Then, the extended Forman curvature of the edge  $(\tau \rightarrow \sigma)$  is defined as:

$$\text{EFC}_{\mathcal{G}}(\tau, \sigma) = 4 - w_\tau^{\text{out}} - w_\sigma^{\text{in}} + 3w_T + 2w_F. \quad (9)$$

We immediately get the following result, inspired by Nguyen et al. (2023, Proposition 4.4) and Fesser & Weber (2023, Proposition 3.4), and which is exemplary of results connecting sensitivity analysis to notions of discrete curvature. We provide the proof in Appendix C.2.

**Proposition 3.4.** Consider a relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$  with update rule given by Equation 3 and satisfying Assumption 1. Denote  $\mathcal{G} = \mathcal{G}(\mathcal{S}, \mathbf{B})$ . Then, for any  $\sigma, \tau \in \mathcal{S}$  with an edge  $(\tau \rightarrow \sigma) \in \mathcal{G}$ , the following holds:

$$\left\| \frac{\partial \mathbf{h}_\tau^{(2)}}{\partial \mathbf{h}_\tau^{(0)}} \right\|_1 \leq \frac{1}{3} \left( \prod_{\ell=0}^1 \alpha^{(\ell)} \beta^{(\ell)} \right) [\text{EFC}_{\mathcal{G}}(\tau, \sigma) + w_\tau^{\text{out}} + w_\sigma^{\text{in}} - 4]. \quad (10)$$

In principle, a similar result using balanced Forman curvature (as in Topping et al., 2022, Theorem 4) is possible using our framework, and we leave that extension for future work. Connections to Ollivier-Ricci curvature are discussed in Appendix B.1.

We present experimental analyses related to curvature on relational structures in Section D.3 (edge curvature distribution) and Appendices D.1 (edge curvature visualization) and D.2 (weighted curvature). We further propose a relational extension of curvature-based rewiring techniques in Section 4 and empirically analyze the impact of relational rewiring using real-world and synthetic benchmarks in Sections 5.1 and 5.2, respectively.

### 3.3 THE IMPACT OF DEPTH

To facilitate our analysis of depth, we make the following non-restrictive assumptions:

**Assumption 2** (Row-Normalized Shift Operators). Each shift operator  $\mathbf{A}^{R_i}$  associated with relation  $R_i$  is row-normalized, such that for all  $\sigma \in \mathcal{S}$ ,

$$\sum_{\xi \in \mathcal{S}^{n_{R_i}-1}} A_{\sigma, \xi}^{R_i} = \begin{cases} 1, & \text{if } \sum_{\xi \in \mathcal{S}} A_{\sigma, \xi}^{R_i} \neq 0, \\ 0, & \text{if } \sum_{\xi \in \mathcal{S}} A_{\sigma, \xi}^{R_i} = 0. \end{cases} \quad (11)$$

**Assumption 3** (Bounded  $\alpha^{(\ell)}$  and  $\beta^{(\ell)}$ ). There exist constants  $\alpha_{\max} > 0$  and  $\beta_{\max} > 0$  such that for all layers  $\ell$ ,  $\alpha^{(\ell)} \leq \alpha_{\max}$  and  $\beta^{(\ell)} \leq \beta_{\max}$ .

{6}------------------------------------------------

We now present our main result on the impact of depth in relational message passing, extending a previous result by Di Giovanni et al. (2023, Theorem 4.1) to our setting. We provide the proof in Appendix C.3. By the *combinatorial distance* from  $\tau$  to  $\sigma$  in the graph  $\mathcal{G}(\mathcal{S}, \tilde{\mathbf{A}})$ , we mean the smallest number of edges in a directed path from  $\tau$  to  $\sigma$  in the graph. Similarly, by *combinatorial length* of a directed path, we mean the number of edges in the path.

**Theorem 3.5** (Impact of Depth on Relational Message Passing). *Consider a relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$  with update rule given by Equation 3 and satisfying Assumptions 1, 2, and 3. Let  $\sigma, \tau \in \mathcal{S}$  be entities such that the combinatorial distance from  $\tau$  to  $\sigma$  in the graph  $\mathcal{G}(\mathcal{S}, \tilde{\mathbf{A}})$  is  $r$ . Denote by  $\omega_\ell(\sigma, \tau)$  the number of directed paths from  $\tau$  to  $\sigma$  of combinatorial length at most  $\ell$  in  $\mathcal{G}(\mathcal{S}, \tilde{\mathbf{A}})$ . Then, for any  $0 \leq m < r$ , there exists a constant  $C > 0$ , depending only on  $\alpha_{\max}$ ,  $\beta_{\max}$ ,  $k$ , and  $m$ , but not on  $r$  nor the specific relations in  $\mathcal{R}$ , such that*

$$\left\| \frac{\partial \mathbf{h}_\sigma^{(r+m)}}{\partial \mathbf{h}_\tau^{(0)}} \right\|_1 \leq C \omega_{r+m}(\sigma, \tau) (2\alpha_{\max} \beta_{\max} M)^r, \quad (12)$$

where  $M = \max_{\sigma, \tau} \tilde{\mathbf{A}}_{\sigma, \tau}$ .

This result indicates that the sensitivity can decay exponentially with depth when  $M < 1/(2\alpha_{\max} \beta_{\max})$ , particularly when the number of walks  $\omega_\ell(\sigma, \tau)$  is limited by the structure of  $\mathcal{G}(\mathcal{S}, \tilde{\mathbf{A}})$ . Such exponential decay is a characteristic of the oversquashing phenomenon, where information from distant nodes becomes increasingly compressed, reducing its influence on the output.

We present experimental validation of this result in Section 5.2.

### 3.4 THE IMPACT OF HIDDEN DIMENSIONS

In situations where the Lipschitz constants of the message and update functions from Assumption 1 are affected by hyperparameters, such as the widths of neural networks implementing said functions, one can have  $\beta_i^{(\ell)} = O(p_{i,\ell})$  and  $\alpha^{(\ell)} = O(p_{\ell+1})$ . This is the case, for instance, when the message and update functions are shallow neural networks (see Appendix B.2 and Appendix C.4).

Writing  $p'_\ell = \max_i p_{i,\ell}$  and substituting  $\beta^{(\ell)} = O(p'_\ell)$  and  $\alpha^{(\ell)} = O(p_{\ell+1})$  into the bound from Lemma 3.2, one gets:

$$\left\| \frac{\partial \mathbf{h}_\sigma^{(t)}}{\partial \mathbf{h}_\tau^{(0)}} \right\|_1 \leq C \cdot \left( \prod_{\ell=0}^{t-1} p_{\ell+1} \cdot p'_\ell \right) (\mathbf{B}^t)_{\sigma, \tau}, \quad (13)$$

where  $C$  is a constant independent of the layer widths,  $p'_\ell$  is the maximum dimension of the message vectors at layer  $\ell$ , and  $p_{\ell+1}$  is the output dimension of the update function at layer  $\ell$ .

This implies that low hidden dimensions in the message and update functions contribute to a low sensitivity upper bound, which can exacerbate the oversquashing problem. Increasing the hidden dimensions will raise the upper bound, which can help improve the model's ability to propagate information effectively and enhance performance on tasks. However, increasing the hidden dimensions risks overfitting due to the increased model complexity (Bartlett et al., 2017).

We present experimental validation of this result in Section 5.2.

#### Takeaway Message 2 (Theoretical)

By reformulating higher order structures as relational structures, key results on oversquashing in graph neural networks extend to relational message passing schemes through the *aggregated influence matrix* and the *influence graph*. This conceptual framework enables analysis of the impact of local geometry, depth, and hidden dimensions in relational message passing schemes, just as in graph neural networks.

## 4 REWIRING HEURISTICS FOR RELATIONAL STRUCTURES

Inspired by First-Order Spectral Rewiring (FoSR) (Karthikar et al., 2023), we propose a rewiring heuristic that integrates additional connections into a relational structure without altering its original

{7}------------------------------------------------

connections. To capture the overall connectivity of a relational structure, we define the collapsed adjacency matrix, which counts the number of direct connections between entities.

**Definition 4.1** (Collapsed Adjacency Matrix). *Given a relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$ , the collapsed adjacency matrix  $\mathbf{A}^{\text{col}}$  for the structure  $\mathcal{R}$  is defined by:*

$$\mathbf{A}_{\sigma, \tau}^{\text{col}} = \sum_{i=1}^k \sum_{\xi \in \mathcal{S}^{n_i-1}} \mathbf{1}_{\{(\sigma, \xi) \in R_i, \tau = \xi_j \text{ for some } j \in \{1, \dots, n_i\}\}}, \quad \sigma, \tau \in \mathcal{S}. \quad (14)$$

This matrix captures direct connections between entities through any relation, effectively collapsing the relational structure into a graph. Our proposed relational rewiring algorithm is as follows.

### --- Algorithm 1 Relational Rewiring Algorithm ---

**Require:** Relational structure  $\mathcal{R} = (\mathcal{S}, R_1, \dots, R_k)$ ; graph rewiring algorithm REWIREALGO

- 1: Construct the collapsed adjacency matrix  $\mathbf{A}^{\text{col}}$  (Definition 4.1)
 - 2: Build the graph  $\mathcal{G}^{\text{col}} = (\mathcal{S}, \mathbf{A}^{\text{col}})$
 - 3: Apply REWIREALGO to  $\mathcal{G}^{\text{col}}$  to obtain additional edges  $E_{\text{new}}$
 - 4: Define a new relation  $R_{k+1} = E_{\text{new}}$
 - 5: Update the relational structure:  $\mathcal{R}' = (\mathcal{S}, R_1, \dots, R_k, R_{k+1})$
- 

Adding new connections ( $E_{\text{new}}$ ) without removing existing ones improves the model capacity to capture long-range dependencies while preserving the original relational structure. We experimentally analyze the impact of relational rewiring using real-world and synthetic benchmarks in Sections 5.1 and 5.2, respectively. Future work could explore rewiring algorithms that remove or reclassify edges, such as spectral pruning (Jamadandi et al., 2024), either by re-labeling the edges as new relations or by deletion. We report preliminary empirical results on pruning in Appendix D.

#### Takeaway Message 3 (Practical)

Graph rewiring techniques for improving information flow and mitigating oversquashing can be adapted to relational structures. This approach improves long-range connectivity and enhances message propagation while maintaining the integrity of the original connections.

## 5 EXPERIMENTS AND RESULTS

### 5.1 REAL-WORLD BENCHMARK: GRAPH CLASSIFICATION

We run an empirical analysis with real-world datasets to compare the performance of different graph, relational graph, and simplicial message passing models, and the impact of relational rewiring on said models. We provide more details in Appendix E.

**Task and Datasets.** We use graph classification tasks ENZYMES, IMDB-B, MUTAG, NCII, and PROTEINS from the TUDataset (Morris et al., 2020) for evaluation.

**Graph Lifting.** For the topological message passing models, graphs are separately treated as complexes with only graph nodes (0-dimensional simplices) and upper adjacencies, and also lifted into clique and ring complexes (see appendix).

**Models.** We evaluate three types of models: a) *Graph message passing models*: SGC (Wu et al., 2019), GCN (Kipf & Welling, 2017), and GIN (Xu et al., 2019); b) *Relational graph message passing models*: RGCN (Schlichtkrull et al., 2018) and RGIN; c) *Topological message passing models*: SIN (Bodnar et al., 2021b), CIN (Bodnar et al., 2021a), and CIN++ (Giusti et al., 2024).

**Relational Rewiring.** We apply relational rewiring for 40 iterations (Section 4) using three choices for REWIREALGO: SDRF (Topping et al., 2022), FoSR (Karthadkar et al., 2023), and AFRC (Fesser & Weber, 2023). Due to computational constraints, we run the three choices on all datasets except IMDB with Clique graph lifting, where we only run FoSR. We use fixed, dataset- and model-agnostic hyperparameters, diverging from prior work where hyperparameter sweeps are carried out. It is

{8}------------------------------------------------

important to note that hyperparameter tuning can significantly impact performance on downstream tasks, as highlighted by, e.g., Tori et al. (2024).

**Training.** Models are trained for up to 500 epochs, with early stopping and learning rate decay based on a validation set. Additional details can be found in Appendix E. Results are reported as *mean  $\pm$  standard error* over 10 trials.

**Results.** Table 1 shows test accuracies for the TUDataset experiments. Rewiring generally boosts performance for base graphs across models and datasets, and the impact of rewiring with our dataset- and model-agnostic choice of hyperparameters varies across datasets, with relational and topological models performance responding to rewiring similarly to graph models.

| Lift | Model | ENZYMES |  | IMDB-B |  | MUTAG |  | NCII |  | PROTEINS |  |
|-|-|-|-|-|-|-|-|-|-|-|-|
|  |  | No Rew. | Best Rew. | No Rew. | FeSR | No Rew. | Best Rew. | No Rew. | Best Rew. | No Rew. | Best Rew. |
| None | SGC | 18.3 $\pm$ 1.2 | 21.5 $\pm$ 1.6 | 40.5 $\pm$ 1.5 | 50.0 $\pm$ 1.8 | 64.5 $\pm$ 5.8 | 70.0 $\pm$ 2.6 | 55.2 $\pm$ 1.0 | 54.4 $\pm$ 0.7 | 62.2 $\pm$ 1.4 | 65.0 $\pm$ 1.5 |
|  | GCN | 32.2 $\pm$ 2.0 | 30.7 $\pm$ 1.5 | 49.1 $\pm$ 1.4 | 47.9 $\pm$ 1.0 | 71.0 $\pm$ 3.8 | 83.0 $\pm$ 1.5 | 48.3 $\pm$ 0.6 | 49.1 $\pm$ 0.8 | 73.1 $\pm$ 1.2 | 77.8 $\pm$ 1.7 |
|  | GIN | 47.2 $\pm$ 1.9 | 50.0 $\pm$ 2.7 | 74.7 $\pm$ 1.5 | 67.1 $\pm$ 1.5 | 83.0 $\pm$ 3.1 | 88.0 $\pm$ 2.4 | 77.2 $\pm$ 0.5 | 77.9 $\pm$ 0.5 | 70.6 $\pm$ 1.6 | 72.2 $\pm$ 0.8 |
|  | RGCN | 33.8 $\pm$ 1.6 | 42.2 $\pm$ 1.3 | 47.6 $\pm$ 1.4 | 68.0 $\pm$ 1.3 | 72.5 $\pm$ 2.5 | 83.5 $\pm$ 1.8 | 63.2 $\pm$ 0.7 | 63.4 $\pm$ 1.1 | 71.9 $\pm$ 1.6 | 76.6 $\pm$ 1.9 |
|  | RGIN | 46.8 $\pm$ 1.8 | 49.8 $\pm$ 2.0 | 69.6 $\pm$ 1.6 | 48.9 $\pm$ 2.9 | 81.5 $\pm$ 1.7 | 85.5 $\pm$ 2.0 | 76.8 $\pm$ 1.1 | 77.0 $\pm$ 1.1 | 70.8 $\pm$ 1.2 | 72.4 $\pm$ 1.4 |
|  | SIN | 47.5 $\pm$ 2.3 | 46.8 $\pm$ 2.1 | 70.0 $\pm$ 1.4 | 63.0 $\pm$ 2.7 | 88.5 $\pm$ 3.0 | 85.5 $\pm$ 1.7 | 77.0 $\pm$ 0.6 | 76.4 $\pm$ 0.4 | 70.2 $\pm$ 1.3 | 73.2 $\pm$ 1.5 |
| Clique | CIN | 50.0 $\pm$ 1.9 | 49.9 $\pm$ 2.0 | 58.1 $\pm$ 4.0 | 58.4 $\pm$ 2.8 | 86.5 $\pm$ 2.8 | 87.0 $\pm$ 2.4 | 51.4 $\pm$ 2.5 | 66.2 $\pm$ 2.0 | 70.7 $\pm$ 1.0 | 71.0 $\pm$ 1.4 |
|  | CIN++ | 48.5 $\pm$ 1.9 | 51.0 $\pm$ 1.5 | 66.6 $\pm$ 3.7 | 56.0 $\pm$ 3.9 | 85.0 $\pm$ 3.4 | 91.0 $\pm$ 2.3 | 60.8 $\pm$ 3.8 | 64.8 $\pm$ 3.1 | 67.9 $\pm$ 1.9 | 71.4 $\pm$ 1.4 |
|  | SGC | 14.5 $\pm$ 1.4 | 16.8 $\pm$ 0.9 | 48.7 $\pm$ 2.2 | 47.8 $\pm$ 1.6 | 70.0 $\pm$ 3.3 | 60.5 $\pm$ 2.6 | 50.0 $\pm$ 1.3 | 56.8 $\pm$ 0.8 | 59.9 $\pm$ 0.8 | 59.1 $\pm$ 1.4 |
|  | GCN | 30.7 $\pm$ 1.2 | 30.2 $\pm$ 2.4 | 64.0 $\pm$ 3.1 | 65.5 $\pm$ 3.1 | 67.0 $\pm$ 3.5 | 81.5 $\pm$ 2.9 | 48.4 $\pm$ 0.4 | 49.6 $\pm$ 0.6 | 69.9 $\pm$ 0.6 | 75.0 $\pm$ 1.4 |
|  | GIN | 44.0 $\pm$ 1.7 | 48.5 $\pm$ 2.2 | 69.1 $\pm$ 1.2 | 70.8 $\pm$ 1.1 | 83.0 $\pm$ 2.8 | 82.5 $\pm$ 2.6 | 78.8 $\pm$ 0.7 | 78.2 $\pm$ 0.6 | 68.7 $\pm$ 1.6 | 72.8 $\pm$ 1.2 |
|  | RGCN | 48.8 $\pm$ 1.2 | 45.2 $\pm$ 1.5 | 71.0 $\pm$ 1.0 | 69.7 $\pm$ 1.3 | 79.5 $\pm$ 1.7 | 81.5 $\pm$ 3.8 | 72.9 $\pm$ 0.8 | 75.0 $\pm$ 0.9 | 72.4 $\pm$ 1.6 | 74.2 $\pm$ 1.2 |
|  | RGIN | 50.8 $\pm$ 1.5 | 55.5 $\pm$ 2.5 | 71.6 $\pm$ 0.9 | 69.0 $\pm$ 1.4 | 86.0 $\pm$ 2.3 | 85.0 $\pm$ 2.4 | 79.2 $\pm$ 0.6 | 79.0 $\pm$ 0.4 | 71.5 $\pm$ 1.5 | 71.8 $\pm$ 1.7 |
|  | SIN | 51.0 $\pm$ 2.4 | 46.5 $\pm$ 1.2 | 53.0 $\pm$ 1.9 | 64.0 $\pm$ 2.3 | 87.0 $\pm$ 3.2 | 83.5 $\pm$ 1.4 | 76.0 $\pm$ 1.3 | 76.4 $\pm$ 0.7 | 66.9 $\pm$ 1.3 | 70.4 $\pm$ 1.2 |
|  | CIN | 49.8 $\pm$ 1.9 | 46.7 $\pm$ 1.3 | 52.6 $\pm$ 2.4 | 68.1 $\pm$ 1.6 | 85.5 $\pm$ 2.8 | 86.5 $\pm$ 2.6 | 51.8 $\pm$ 2.3 | 72.5 $\pm$ 0.8 | 70.7 $\pm$ 1.2 | 70.3 $\pm$ 0.8 |
|  | CIN++ | 50.5 $\pm$ 2.1 | 52.7 $\pm$ 1.6 | 62.8 $\pm$ 3.8 | 64.7 $\pm$ 1.5 | 90.5 $\pm$ 2.2 | 84.5 $\pm$ 3.3 | 61.5 $\pm$ 4.6 | 76.8 $\pm$ 0.4 | 68.3 $\pm$ 1.3 | 71.9 $\pm$ 1.0 |
| Ring | SGC | 16.5 $\pm$ 1.6 | 19.3 $\pm$ 1.2 | 50.1 $\pm$ 1.9 | 40.9 $\pm$ 1.9 | 65.5 $\pm$ 3.6 | 75.0 $\pm$ 5.7 | 51.5 $\pm$ 1.2 | 51.4 $\pm$ 0.4 | 44.8 $\pm$ 2.3 | 49.3 $\pm$ 3.6 |
|  | GCN | 34.8 $\pm$ 1.3 | 32.0 $\pm$ 1.4 | 46.9 $\pm$ 1.4 | 48.0 $\pm$ 1.2 | 72.0 $\pm$ 2.7 | 77.5 $\pm$ 2.4 | 49.3 $\pm$ 0.9 | 49.4 $\pm$ 0.6 | 72.2 $\pm$ 1.3 | 72.7 $\pm$ 1.2 |
|  | GIN | 46.7 $\pm$ 2.4 | 47.0 $\pm$ 1.6 | 70.1 $\pm$ 1.7 | 70.7 $\pm$ 1.7 | 88.0 $\pm$ 2.1 | 89.0 $\pm$ 1.9 | 78.8 $\pm$ 0.6 | 77.5 $\pm$ 0.9 | 69.8 $\pm$ 1.4 | 72.1 $\pm$ 0.9 |
|  | RGCN | 35.2 $\pm$ 1.7 | 45.7 $\pm$ 1.5 | 71.1 $\pm$ 1.4 | 74.0 $\pm$ 1.6 | 83.5 $\pm$ 2.7 | 84.0 $\pm$ 2.1 | 73.4 $\pm$ 0.5 | 73.3 $\pm$ 0.5 | 70.7 $\pm$ 1.6 | 71.3 $\pm$ 1.2 |
|  | RGIN | 45.3 $\pm$ 1.3 | 49.2 $\pm$ 1.5 | 68.6 $\pm$ 1.2 | 67.2 $\pm$ 1.8 | 87.0 $\pm$ 2.9 | 87.5 $\pm$ 2.4 | 78.4 $\pm$ 0.7 | 79.8 $\pm$ 0.7 | 68.8 $\pm$ 1.5 | 71.3 $\pm$ 1.5 |
|  | SIN | 40.3 $\pm$ 2.2 | 48.0 $\pm$ 2.0 | 50.6 $\pm$ 1.9 | 60.9 $\pm$ 2.1 | 85.0 $\pm$ 2.1 | 88.5 $\pm$ 2.5 | 80.0 $\pm$ 0.8 | 79.1 $\pm$ 0.9 | 70.6 $\pm$ 1.1 | 72.1 $\pm$ 0.7 |
|  | CIN | 47.5 $\pm$ 2.0 | 49.5 $\pm$ 2.0 | 48.6 $\pm$ 1.6 | 66.1 $\pm$ 2.0 | 83.5 $\pm$ 2.1 | 90.0 $\pm$ 1.3 | 51.6 $\pm$ 3.2 | 76.5 $\pm$ 0.5 | 68.7 $\pm$ 1.4 | 68.5 $\pm$ 1.6 |
|  | CIN++ | 47.5 $\pm$ 1.7 | 46.3 $\pm$ 1.8 | 66.0 $\pm$ 1.4 | 67.8 $\pm$ 1.3 | 85.5 $\pm$ 2.0 | 90.0 $\pm$ 2.7 | 56.8 $\pm$ 4.5 | 76.0 $\pm$ 0.6 | 68.1 $\pm$ 1.2 | 70.1 $\pm$ 1.2 |

Table 1: Test accuracy for TUDataset experiments. Each value is presented as the mean  $\pm$  standard error across ten trials. The best-performing result for each dataset is highlighted in gold, while the second-best is in silver. The results after rewiring are shown with green text if the mean increased and red text if the mean decreased.

### 5.2 SYNTHETIC BENCHMARK: RINGTRANSFER

We confirm the theoretical results from Section 3 using the RINGTRANSFER benchmark, a graph feature transfer task designed to tease out the effect of long-range dependencies in message-passing models using rings of growing size. We follow the experimental setup of Karhadkar et al. (2023) and Di Giovanni et al. (2023), and provide more details in Appendix E.2. We test the impact of neural network hidden dimensions (Section 3.4), relational structure depth (Section 3.3), and relational structure local geometry (Sections 3.4 and 4) on task performance by varying the hidden dimensions, ring sizes, and rewiring iterations. The results, consistent with the theory, demonstrate that increasing network hidden dimensions improves performance up to a point, after which it declines, potentially due to overfitting. Larger ring sizes lead to performance deterioration, as the effects of long-range dependencies and bottlenecks start to take over. At the same time, rewiring improves performance by facilitating communication between distant nodes and mitigating oversquashing. As illustrated in Figure 2, message passing on graphs and simplicial complexes demonstrate similar trends, consistent with our theoretical predictions.

### 5.3 ADDITIONAL EXPERIMENTS AND ANALYSES

We report additional analyses in Appendix D.1 and Appendix D.2. There, we visualize the curvature of relational structures for dumbbell graphs and their corresponding clique complexes. We also reports a statistically significant linear relationship between the weighted curvature of graphs and their lifted clique complexes. These interesting patterns merit further investigation. We also present the following additional experiments: (1) neighbors match for path of cliques and tree datasets in

{9}------------------------------------------------

![Figure 2: Performance on RINGTRANSFER. The figure contains three line plots. Plot (a) shows Accuracy (%) vs Hidden Dimension (log scale from 10^0 to 10^2). Plot (b) shows Accuracy (%) vs Nodes (6 to 14). Plot (c) shows Accuracy (%) vs Rewiring Iterations (0 to 10). All plots compare GIN/None (green), RGCN/None (blue), RGCN/Clique (orange), and RGCN/Ring (purple).](4e0ade2f41b66d5602160da5cc978274_img.jpg)

Figure 2 consists of three line plots showing performance on the RINGTRANSFER dataset. The legend for all plots is: GIN/None (green), RGCN/None (blue), RGCN/Clique (orange), and RGCN/Ring (purple). Plot (a) shows Accuracy (%) on the y-axis (0 to 100) against Hidden Dimension on a log scale (10<sup>0</sup> to 10<sup>2</sup>). Plot (b) shows Accuracy (%) on the y-axis (0 to 100) against Nodes (6 to 14). Plot (c) shows Accuracy (%) on the y-axis (0 to 100) against Rewiring Iterations (0 to 10). In all plots, GIN/None generally achieves the highest accuracy, while RGCN/Ring generally achieves the lowest.

Figure 2: Performance on RINGTRANSFER. The figure contains three line plots. Plot (a) shows Accuracy (%) vs Hidden Dimension (log scale from 10^0 to 10^2). Plot (b) shows Accuracy (%) vs Nodes (6 to 14). Plot (c) shows Accuracy (%) vs Rewiring Iterations (0 to 10). All plots compare GIN/None (green), RGCN/None (blue), RGCN/Clique (orange), and RGCN/Ring (purple).

Figure 2: Performance on RINGTRANSFER obtained by varying model hidden dimensions (left), ring size (middle), and number of rewiring iterations (right).

Appendix D.4, (2) graph regression for ZINC in Appendix D.5, (3) node classification for CORNELL, WISCONSIN, TEXAS, CORA, and CITeseer in Appendix D.6, (4) simplex pruning on the MUTAG dataset in Appendix D.8, and (5) full TUDataset results in Appendix E.1.

## 6 DISCUSSION AND CONCLUSIONS

This work addresses pressing questions about oversquashing in topological networks and higher-order generalizations of rewiring algorithms raised by the TDL community (Questions 2 and 9 of Papamakou et al., 2024). We introduce a theoretical framework for unifying graph and topological message passing via relational structures, extending key graph-theoretic results on oversquashing and rewiring strategies to higher-order networks such as simplicial complexes via *influence graphs* that capture the aggregated message passing dynamical structure on relational structures. Our approach applies broadly to other message-passing schemes, including relational GNNs, high-order GNNs, and CW networks, providing a foundation for future theoretical and empirical research. Empirical results on real-world datasets show that simplicial networks respond to rewiring similarly to graph networks, and synthetic benchmarks further confirm our theoretical findings.

Certain aspects are worthy of further investigation. In particular, we compare message passing on graphs and their clique complexes through proxies (e.g., performance on tasks), as the significant differences in size and structure make direct empirical comparisons, e.g., of curvature, less theoretically rigorous. While we observe statistically significant patterns when comparing weighted curvatures, further theoretical and empirical investigation is needed. Furthermore, the rewiring algorithms we applied our relational rewiring heuristic to were not originally designed with weighted directed influence graphs in mind. Potentially, further improvements could be obtained by implementing algorithms specifically tailored for rewiring weighted directed graphs.

For future work, exploring global geometric properties of relational structures, studying oversmoothing, and empirically analyzing more relational message-passing schemes are promising directions. Developing theoretical tools and tailored rewiring heuristics for weighted directed graphs will be crucial, as will be tools for direct comparisons of message-passing across different relational structures. Furthermore, systematically and empirically assessing our framework’s higher-order extensions of more state-of-the-art (SoTA) graph rewiring solutions is essential. By unifying topological message passing into message passing on relational structures and generalizing graph-based analysis to this setting, we hope that the present work can aid in both rigorous analysis and direct comparison between different higher-order message passing schemes.

Lastly, for practitioners, we recommend topological message passing as yet another relational learning tool with relational rewiring as a preprocessing step.

**Reproducibility Statement** The code for replicating our experiments is available at <https://github.com/chapman20j/Simplicial-Oversquashing>. Experimental settings and implementation details are described in Section 5, and Appendices E and F.

 Rest of paper (reference and Appendix) is removed.
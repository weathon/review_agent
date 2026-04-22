# LEAP: Local ECT-Based Learnable Positional Encodings for Graphs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Graph neural networks (GNNs) largely rely on the message-passing paradigm, where nodes iteratively aggregate information from their neighbors. Yet, standard message passing neural networks (MPNNs) face well-documented theoretical and practical limitations. Graph positional encoding (PE) has emerged as a promising direction to address these limitations. The Euler Characteristic Transform (ECT) is an efficiently computable geometric–topological invariant that characterizes shapes and graphs. In this work, we combine the differentiable approximation of the ECT (DECT) and its local variant ($\ell$-ECT) to propose LEAP, a new end-to-end trainable local structural PE for graphs. We evaluate our approach on multiple real-world datasets as well as on a synthetic task designed to test its ability to extract topological features. Our results underline the potential of LEAP-based encodings as a powerful component for graph representation learning pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Building upon the local Euler Characteristic Transform ($\ell$-ECT), this paper introduces a new class of learnable *local strucutral encodings* (LSE) that can be integrated in any GNN architecture. To be precise, the main novelty is that such structural encodings are made learnable, since they were already adopted as static descriptors in another work, although very recent. Another novelty lies into a new "projection" layer in the proposed neural architecture, allowing one to reduce a matrix containing the differentiable approximations of $\ell$-ECT, one per vertex, into a final vector: the positional encoding for that vertex. The approach is extremely versatile and the experimental section clearly shows that i) the proposed LSE allows standard GNNs to improve their performance in a variety of tasks and ii) learning the LSEs further increase the improvement and iii)  the choice of the hyper-parameters (e.g. number of directions, time grids) in not constraining, meaning that different choices still lead to good results, except for the hop's ray, in this case the smaller the better.

### Strengths
This paper is really well written, clear and concise.  The experimental setup is solid and the results are impressive (more comments later), however I would say the contribution is just "fair" since local and global ECT descriptors already existed. However, I think that this paper has the potential to be accepted at ICLR, so I am **temporarily** ranking it with a 6, but I am ready to raise my note to 8 after the discussion, provided some key points are discussed/addressed.

### Weaknesses
A few points deserves to be discussed and/or presented differently. One major drawback, discussed below, and this is the only reason why I do not immediately rate this paper with a 8.

### Questions
Major remarks: 

i) It is unfortunate that the word "projection" is used in two different contexts: former concerning the definition of ECT, Eq. (4), latter concerning the $\phi$ map at l.198. Moreover, this word plays a role in your acronym and it is quite clear that you refer to the second projection ($\phi$) there but a few words about it would be appreciated. 

ii) At line 225 you introduce the notion of *valid graph embedding*. It would be wise to recall the reader what it means.

iii) The experiments of the synthetic dataset are the only ones not being very convincing. I mean: what happens when you provide GCN/GAT with LaPE or RWPE? Isn't the prediction score equal to 1? Even if it were the case it would not be an issue, in my view, since in the rest of the section, by the way, you beat the state-of-the-art.

iv) Here's the weakest point of the paper in my view. There is a number of message passing architectures known to be more expressive w.r.t. GCNs. The first and maybe most important I can think of is graphs isomorphism networks (GINs, [1]). Including them in your Table 4 would turn your paper into a master  piece.  
 
v) At l. 355 I read: "An interesting observation from Table 4 ... ". I think you were too modest in your commentary! What I see in Table 4 is that on 5 columns over 7 the best accuracy score is obtained by a neural architecture **not** being a GNN, provided that it is equipped with LEAP-L. Did about Alchemy and HIV datasets? Did you test NoMP on them?

Minor remarks: 

i) At line 239, you need a space between *to* and $|\\Theta |$  
ii) Line 248: *rojection* -> *projection*

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes LEAP, a learnable graph positional encoding method based on a local Euler Characteristic Transform (ℓ-ECT). The key idea is to compute differentiable, direction-dependent topology-aware signatures localized around each node and incorporate them as positional embeddings in standard GNN architectures. By leveraging both geometric and topological structure, LEAP aims to address limitations of message passing GNNs that struggle to capture higher-order structural patterns. The authors evaluate the method on synthetic and real-world graph classification datasets, comparing to standard positional encodings such as RWPE and Laplacian-based LaPE. Results indicate that LEAP improves performance in many settings, particularly when used with architectures that are otherwise limited in structural awareness.

### Strengths
1. The use of local, differentiable Euler characteristic transforms as graph positional encoding is an interesting new direction that meaningfully blends geometric and topological features.
2. The paper clearly explains the limitations of pure message passing networks and why positional encoding is useful.
3. The learnable direction variants of LEAP show thoughtful design toward practical applicability.

### Weaknesses
1. While LEAP is motivated as addressing expressivity limits of MPNNs, the paper does not provide formal analysis or guarantees regarding what structural distinctions LEAP can or cannot capture. The argument remains primarily empirical.
2. While LEAP performs well in several TU datasets, its benefits diminish or disappear on datasets where global structure is more important (e.g., HIV), where LaPE performs best. This suggests that LEAP may be inherently limited in capturing global structural contexts, which should be more explicitly acknowledged or explored.

### Questions
1. Since LEAP is intrinsically local, have you considered augmenting it with a complementary global encoding? Could LEAP be combined with LaPE without redundancy?
2. What is the computational overhead per graph and per layer when incorporating LEAP? How does it scale with graph size and number of directions?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes LEAP, a learnable positional encoding for graphs based on the local Euler Characteristic Transform ($l$-ECT), which integrates both geometric and topological information and supports end-to-end training. The authors validate LEAP on synthetic tasks and multiple real-world graph datasets, demonstrating its ability to capture structural information even when node features are non-informative.

### Strengths
(1) LEAP is the first work to integrate the local ECT into GNNs in an end-to-end trainable manner, combining geometric and topological insights.

(2) The authors evaluate their method on synthetic data, small molecules, image graphs, and large-scale quantum chemistry datasets, showing consistent improvements. The synthetic task clearly demonstrates LEAP’s ability to classify graphs based solely on structural information, highlighting limitations of standard MPNNs.

(3) The paper includes thorough ablation studies on various aspects (Projection strategies, Locality, PE dimension, and DECT hyperparameters) of LEAP, strengthening its conclusions.

### Weaknesses
(1) As the authors acknowledge in the limitations, LEAP is not a purely structural PE. It requires node features $x(v)$ to compute the ECT. While the synthetic experiment shows LEAP works even with uninformative features, it is unclear how, or if, LEAP would be applied to graphs without any node features (e.g., graphs represented only by an adjacency matrix).

(2) LEAP introduces more hyperparameters (e.g., number of directions, smoothing, discretization steps) compared to LaPE or RWPE.

(3) Although not analyzed in depth, computing ECT for m-hop subgraphs for every node could be expensive for large graphs.

### Questions
The ablation study (Table 2 ) shows that 1-hop neighborhoods perform better than 2-hop or a 1,2-hop combination. This seems counter-intuitive to the MPNN notion that larger receptive fields are better. Why is 1-hop sufficient?

How does LEAP perform on very large graphs?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LEAP, an end-to-end trainable local structural positional encoding that integrates geometric and topological information via the local Euler Characteristic Transform (ℓ-ECT). LEAP computes node-level embeddings by extracting m-hop neighborhoods, normalizing local features, applying a differentiable ℓ-ECT, and projecting the resulting matrix into a compact representation through learnable modules (e.g., linear layers, 1D convolutions, or DeepSets). Two variants are proposed: LEAP-F, which uses fixed ECT directions, and LEAP-L, which learns them during training. Extensive experiments on both synthetic and real-world datasets, using three GNN architectures, confirm that LEAP substantially enhances graph representation learning.

### Strengths
S1: The paper proposes an end-to-end trainable framework for encoding positional information, eliminating the need for static or precomputed preprocessing.

S2: The experimental setup is well-structured, evaluating three mainstream GNN architectures across multiple datasets, including both synthetic and real-world benchmarks. 

S3: Ablation studies demonstrate LEAP’s robustness to key hyperparameters and locality configurations. Moreover, the five projection strategies showcase LEAP’s adaptability to diverse model architectures and task requirements.

### Weaknesses
W1: Although the paper discusses LEAP’s robustness to hyperparameters (e.g., number of directions, thresholds), it omits computational efficiency analysis.

W2: Since no single projection strategy yields the best results across all tasks, do the authors have heuristic or data-driven guidelines for choosing among them

W3: Theoretical analysis and justification are insufficient, which is critical for this method-oriented paper.

### Questions
Q1: On the HIV dataset, the locally encoded LEAP underperforms the globally informed LaPE. Could the authors elaborate on why global structural information is crucial for this task?

Q2: The injectivity of the Euler Characteristic Transform underpins its discriminative power, yet LEAP employs a differentiable local approximation. Do the authors have theoretical or empirical evidence that this approximation preserves key topological information compared with the exact, global ECT?

Q3: Is LEAP applicable to geometric graphs? If so, how does it leverage spatial coordinates or geometric relationships within the ℓ-ECT framework?

### Soundness
2

### Presentation
3

### Contribution
2

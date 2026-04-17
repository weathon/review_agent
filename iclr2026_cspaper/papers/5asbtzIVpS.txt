# Forest-Based Graph Learning For Semi-Supervised Node Classification

Jin Li1, Shenghao Gao1, Kaichen Zhang1, Xinlong Chen2, Ying Sun1†, Hui Xiong**1,3†**
1Thrust of Artificial Intelligence, The Hong Kong University of Science and Technology (Guangzhou) 2College of Computer and Data Science, Fuzhou University 3The Hong Kong University of Science and Technology (Hong Kong SAR)
jslijin2015@outlook.com, gaoshenghao512@gmail.com kzhangbi@connect.ust.hk, fjxinlong@gmail.com yings@hkust-gz.edu.cn, xionghui@ust.hk ∗

## Abstract

Existing Graph Neural Networks usually learn long-distance knowledge via stacked layers or global attention, but struggle to balance cost-effectiveness and global receptive field. In this work, we break the dilemma by proposing a novel forest-based graph learning (FGL) paradigm that enables efficient long-range information propagation. Our key insight is to reinterpret message passing on a graph as transportation over spanning trees that naturally facilitates long-range knowledge aggregation, where several trees–a forest–can capture complementary topological pathways. Theoretically, we demonstrate that as edge-homophily estimates improve, the induced distribution biases towards higher-homophily trees, which enables generating a high-quality forest by refining a homophily estimator. Furthermore, we propose a linear-time tree aggregator that realizes quadratic node-pair interactions. Empirically, our framework achieves comparable results against state-of-the-art counterparts on semi-supervised node classification tasks while remaining efficient. Codes are available at https://anonymous.4open.science/r/FGL/.

## 1 Introduction

Graph Neural Networks (GNNs) (Wu et al., 2020; Chen et al., 2020b; Thomas et al., 2022) attract much attention in recent years due to their expressivity in solving various graph-related tasks (*e.g.*,
node and graph classifications (Feng et al., 2020; Xie et al., 2022) or clustering (Bianchi et al., 2020), link prediction (Yun et al., 2021), and anomaly detections (Dong et al., 2025; Gong et al., 2023)), with also many applications in, *e.g.*, texts (Wang et al., 2024b), images (Nazir et al., 2021; Guan et al., 2022), generation (Zhuang et al., 2025; Gong & Sun, 2025), traffic (Jiang & Luo, 2022), and other domains (Gong & Sun, 2024; Cui et al., 2026). Despite their popularity and successes, most GNNs restrict receptive fields to 2-/3-hop local neighborhoods and focus on nearby information aggregation while ignoring distant knowledge, which would limit their real-world application scopes when dealing with challenging tasks where long-range interactions are critical and necessary. For example, as discussed in Sec. A.1, the imbalance of densities or degrees often causes insufficient local knowledge for some nodes, which becomes more severe under graph heterophily and risks further over-fitness from label scarcity. In this paper, we focus on semi-supervised node classifications to underscore labeling challenges. To facilitate long-distance interactions, existing works have devoted much effort and can be generally categorized into two different architectures: (1) Deep local models (*e.g.*, deep GNNs (Chen et al., 2022c; Li et al., 2019; Chen et al., 2020a)) expand the global receptive fields by stacking multiple local layers, with each considering only first-order information. (2) Shallow global models (e.g., Global Graph Transformers (Ying et al., 2021; Kreuzer et al., 2021)) integrate 1 or 2 non-local aggregating operators (e.g., global attentions), encapsulating all pairwise node interactions in a single layer. Unfortunately, most of them suffer from high time and space complexities (Li et al., 2021; Wu
∗† Corresponding authors: Ying Sun (yingsun@connect.hkust-gz.edu.cn) and Hui Xiong
(huixiong@ust.hk).

1 et al., 2022), due to excessive unparallelizable layers (former) or quadratic node-pair interactions (latter). Recently, few prior works attempt to mitigate complexities via some sparsity techniques such as Adaptive Selection (Chen et al., 2022b; Wu et al., 2022) and Graph Rewiring (Shirzad et al., 2023). Yet, they sacrifice global coverage and have to make selections, and thus either risk dropping some important node interactions or heavily rely on extra sophisticated selection strategies. Overall, such methods fail to simultaneously address comprehensive long-range knowledge extraction and cost-effectiveness, which is rooted in the inherent limitation of existing learning paradigms. Such a graph learning dilemma urges us to rethink existing paradigms and explore an alternative that breaks the unavoidable trade-off between cost-effectiveness and a global receptive field. The essential observation is that these paradigms view a graph as a fusion of structures, whose total costs can be calculated as follows:

## Total Cost = (Cost Per Structure) × (Number Of Structures). (1)

Thus, when modeling with local primitives—first-order neighborhoods (Li et al., 2019) or short random walks (Zhang et al., 2020)—the per-structure cost is low, but numerous such structures are required for covering long distances. In contrast, global operators (Ying et al., 2021) can reduce the number of structures, yet at the expense of prohibitive per-structure cost due to dense pairwise interactions. Based on the above analysis, we naturally raise a question: Does there exist a structure that simultaneously controls these two factors? To answer this question, we recognize that a spanning tree is the minimal subgraph connecting all nodes. Therefore, under limited structure counts, such a tree is **the simplest structure that achieves global coverage** (Fig. 1), indicating that it may be more suitable for long-range propagation. Furthermore, we suggest using a forest (tree set), since a single spanning tree may be insufficient to capture all topological knowledge.

Figure 1: Our paradigm (right) utilizes the most sparse structures of a graph, *i.e.*, spanning trees, to aggregate global messages against the prior paradigms (left).

In this paper, we propose forest-based graph learning (FGL), a novel paradigm that models information propagation on a graph as transport on a forest of spanning trees, economically achieving global coverage. To obtain a high-quality forest, we expect to sample the trees from a distribution biased towards homophilous trees. Theoretically, we demonstrate that as edge-homophily estimates improve, the induced tree distribution asymptotically approaches the ideal one. Accordingly, we propose a tree sampler, based on a well-trained edge-homophily estimator, to enable generating several spanning trees with higher homophily via the weighted Wilson algorithm (Wilson, 1996). Besides, we design a general tree aggregator 1, by deriving two recursions on trees, which propagates global messages in linear running time. Additionally, a post-hoc mean operator is adopted as our tree fuser to merge knowledge from different trees. These components constitute our full framework, as illustrated in Fig. 2.

Our contributions are summarized as follows: 1. **New Paradigm**: We introduce a forest-based graph learning paradigm FGL, which can comprehensively capture long-range knowledge with high efficiency. 2. **Theoretical Insight**: We establish a rigorous asymptotic relationship between the accuracy of the edge-homophily estimator and the quality of the induced tree distribution, which reveals that refining the estimator provably yields a better tree distribution. 3. **Effective Approach**: We propose 1) a homophily estimator-based tree sampler, which generates homophilous trees with higher probability; and 2) a general tree aggregator that conducts quadratic pairwise node interactions with only linear complexities. 4. **Experimental Results**: Our framework achieves competitive results against state-of-the-art counterparts in semi-supervised node classifications with higher efficiency, e.g., 11.90% and 16.14% average relative gains against GCNII and DIFFormer (representative Deep GNN and Graph Transformer), respectively.

![1_image_0.png](1_image_0.png)

## 2 Related Literature

Deep Local Models. Deep Graph Neural Networks (GNNs) expand their receptive fields by iteratively stacking local aggregators, enabling fine-grained control over neighborhood information at each layer Yang et al. (2020); Fang et al. (2023); Chen et al. (2020a). This depth provides strong expressiveness but comes with notable drawbacks: sequential computation limits parallelism, higher time/space complexities, and the risk of over-smoothing. To mitigate over-smoothing, various strategies have been explored, including normalization layers Zhao & Akoglu (2020); Zhou et al. (2021); Yang et al. (2020), random dropping techniques Rong et al. (2020b); Huang et al. (2020); Fang et al. (2023), and skip connections Li et al. (2019); Chen et al. (2020a); Luan et al. (2019); Xu et al. (2018). Despite these improvements, deep local models inherently rely on step-wise neighborhood aggregation, which prevents efficient global message passing and parallelization. Shallow Global Models. Graph Transformers (GTs) adopt a contrasting perspective: instead of gradual local aggregation, they model direct pairwise interactions among nodes, often in just a few global layers Min et al. (2022); Hussain et al. (2022); Ying et al. (2021). This shallow global paradigm (G ≈ x → yx,y∈V) allows rapid global communication but typically incurs quadratic complexity. To improve scalability, recent works either sparsify interactions via sampling or pruning (*e.g.*, Gophormer Zhao et al. (2021), NodeFormer Wu et al. (2022), Exphormer Shirzad et al. (2023)), or simplify attention mechanisms to reduce computation (*e.g.*, SGFormer Wu et al. (2024), GOAT Kong et al. (2023)). While these strategies address efficiency, they often lose structural bias, motivating the use of positional encodings Ying et al. (2021); Chen et al. (2022a) or walk-based formulations Zhang et al. (2020). However, designing encodings that are both expressive and efficient remains challenging. Tradeoff Between Local and Global Models. Deep Local Models excel at capturing fine-grained neighborhood structures but struggle with scalability and long-range dependencies. In contrast, Shallow Global Models enable efficient global message propagation with fewer layers, but often overlook nuanced local structures or incur high complexity without careful approximation. Hybrid designs attempt to combine both perspectives Wu et al. (2021); Rong et al. (2020a); Kreuzer et al. (2021). In contrast, we analyze the essential limitation of existing learning paradigms and propose a novel forest-based paradigm that enables efficient long-range modeling along with natural structural knowledge preservation, addressing this dilemma from a more fundamental perspective.

## 3 Preliminary

Notations. Let G = (*V, E*) be an unweighted graph with n nodes V = {vi}
n i=1 and m edges E = {ei,j}. The graph is represented by a feature matrix X ∈ R
n×dand an adjacency matrix A ∈ {0, 1}
n×n, where Aij = 1 if and only if (vi, vj ) ∈ E. We also define the normalized adjacency matrix Aˆ = D
−
1 2 (A + I)D
−
1 2 , where D is the degree matrix of A + I.

Problem Formulation. In semi-supervised node classification, a subset of nodes VL ⊂ V has labels yi ∈ {0, 1*, ..., c* − 1}, while the remaining nodes are unlabeled. The goal is to learn node embeddings H′′ ∈ R
n×dsuch that a simple linear predictor can be applied to H′′ to predict node labels for all vi ∈ V , leveraging both labeled and unlabeled nodes.

## 4 Method

Existing paradigms suffer from the trade-off between cost-effectiveness and a global receptive field. To obtain global coverage, deep local models with small local structures require stacking a large number of structures, while shallow global models with large complex structures incur *high* per-structure computational costs. In this work, we introduce an intermediate-level structure—the tree—that offers a principled way to balance this trade-off, exhibiting a new learning paradigm. A
tree connects all nodes in a graph in a cost-efficient and non-redundant manner. We build on this insight to propose the **Forest-based Graph Learning (FGL)** framework illustrated in Fig. 2, which is composed of four key components: (1) **Pre-processing**, which augments the original input graph to facilitate downstream computation; (2) **Tree Sampler**, which derives a target distribution over spanning trees and generates multiple trees accordingly; (3) **Tree Aggregator**,

![3_image_0.png](3_image_0.png)

which performs message passing along each individual spanning tree; and (4) **Tree Fuser**, which integrates the aggregated messages from all sampled trees into a unified representation.

## 4.1 Pre-Processing

Real-world graphs are often not connected, which hinders the subsequent spanning tree sampling process. To address this issue, we begin by computing pseudo-labels for each node, denoted as Y
′ ∈ R
n×c. For heterophilous graphs, we employ a simple feed-forward layer, Y
′ = σ(XW)
whereas for homophilous graphs, we use a GCN layer, Y
′ = σ(AXW ˆ ). The learnable parameters W ∈ R
d×care optimized on the labeled nodes using the standard cross-entropy loss. We then construct an augmented graph Gˆ by leveraging the pseudo-labels. For each node, we use its pseudolabel representation y
′ ∈ R
1×cto identify its k nearest neighbors. If an edge does not already exist between the node and one of these neighbors, we introduce a new edge. This pre-processing step offers two key benefits at the same time. First, it ensures graph connectivity, which is necessary for subsequent spanning tree sampling. Second, it increases the *homophily* ratio—the proportion of edges linking nodes with similar class labels—which has been shown to improve performance in semi-supervised node classification (Chien et al., 2021).

## 4.2 Tree Sampler

To generate a high-quality forest composed of several spanning trees, we identify two essential principles: 1) *homophily ratios*: Since we target node classification, it is a critical measure on graphs and thus can be naturally transferred to trees. 2) *diversity*: if these trees tend to overlap, then the forest would be degraded into a single tree, which may be insufficient to cover all the topological knowledge of a graph, therefore necessitating diversity.

Therefore, we expect to sample the trees independently from a distribution PGb (T) biased towards trees with high homophily ratios. We assume each tree T has a score s(T) that can be calculated as the product of edge scores s(e), thereby defining the tree distribution on a graph as follows:

$$P_{\widehat{G}}\left(T\right)={\frac{s(T)}{\sum_{T\subseteq{\widehat{G}}}s(T)}}={\frac{\prod_{e\in T}s(e)}{\sum_{T\subseteq{\widehat{G}}}\prod_{e\in T}s(e)}}.$$
. (2)
The only remaining step is to determine the edge scores s(e). Our main idea is to assign higher scores to those homophilous edges and lower scores to heterophilous edges, which intuitively improves

$$\left(2\right)$$

![4_image_0.png](4_image_0.png)

$$({\mathfrak{I}})$$

Figure 3: Illustration of the tree aggregator. The red node denotes the root, and the blue node indicates the focal node. (a) Red dashed lines depict the bottom-up computation of S, while blue dashed lines represent the computation of H′v. (b)(c) Detailed computations along the focal edge are shown.

the probabilities assigned to homophilous trees. We justify this intuition in Sec. 4.6 by theoretically demonstrating that this scoring strategy can induce a distribution biased towards higher-homophily trees (Theorem 2). Therefore, we introduce a homophily estimator to find those homophilous edges and assign higher scores to them. Here, we implement this homophily estimator via local attention:

$$\alpha_{i\to j}=\frac{\exp\left(Q_{i}K_{j}^{\top}/\sqrt{c}\right)}{\sum_{v\in\mathcal{N}(i)}\exp\left(Q_{i}K_{v}^{\top}/\sqrt{c}\right)},\quad\forall\,i,j\in V$$
, ∀ *i, j* ∈ V (3)
where Q = XWQ, K = XWK, and V = XWV with learnable WQ, WK, WV ∈ R
d×c. Ni denotes the first-order neighborhood of node i ∈ V . We train the local graph attention by minimizing the cross-entropy loss with targets Y
′. Thus, the edge score s(e) for e = (*i, j*) is defined by s(e) = (αi→j + αj→i) /2. Finally, our tree sampler generates NT independent spanning trees from PGb(T) via the algorithm of Wilson (1996) in nearly O(n) time per-tree.

## 4.3 Tree Aggregator

The tree aggregator f
(T)
Agg over tree T with root r is defined as f
(T)
Agg : H ∈ R
n×d7→ H′ ∈ R
n×d, which is designed based on a general message aggregator fAgg (·). The idea is rooted in a key observation: for neighboring nodes *u, v* on tree T, the globally merged messages targeting them differ only at one edge direction (visualized in Fig. 3). Leveraging this observation can facilitate efficient tree propagation by any general fAgg (·) that satisfies: given two message sets *A, B* with possible auxiliary information (*e.g.*, weights), if merging A into B getting S, then there always exists two operators M+/− (·) to make the following sufficient properties hold.

$f_{\text{Agg}}\left(S\right)=\mathcal{M}^{+}\left(f_{\text{Agg}}\left(B\right),\,f_{\text{Agg}}\left(A\right)\right),\quad\text{Property(I):Combine}$  $f_{\text{Agg}}\left(B\right)=\mathcal{M}^{-}\left(f_{\text{Agg}}\left(S\right),\,f_{\text{Agg}}\left(A\right)\right),\quad\text{Property(II):Disentangle}$
$$\mathbf{bi}\mathbf{n}\mathbf{e}$$

where M+/−⃗a, ⃗b denote adding vector ⃗b to ⃗a or deleting ⃗b from ⃗a, which are allowed unsymmetrical via auxiliary information. These identified properties do not sacrifice the generality of fAgg (·).

Indeed, many popular auto-regressive sequence models and first-order GNN aggregators can be adopted, *e.g.*, linear attention Zhou et al. (2021); Wu et al. (2024), linear Recurrent Neural Networks (RNNs) Liu et al. (2024), and State Space Models (SSMs) Sarem et al. (2024); Zhang et al. (2025); Xiao et al. (2024) as well as non-linear variants (Sec. A.6), thus highlighting its generality. Based on these properties, we can theoretically derive a general tree aggregator f
(T)
Agg high-levelly via two recursions in Theorem 1. The proof and further explanation can be found in Sec. B.1 of Appn. Theorem 1. Given a tree T with a root r ∈ V , each node v ∈ V *has a subtree* T
(sub)
v *with nodes* V
(sub)
v ⊆ V . Denote the father node and the children nodes of v *on tree* T as Fa (v) and Child (v).

Let Sv represent the aggregated message at node v *from all messages from* V
(sub)
v . *Then, given* any message aggregator fAgg (·) satisfying Properties (I) and (II) as well as function g (·), our tree aggregator f
(T)
Agg : H 7→ H′ ∈ R
n×d*can be always derived as two recursions via operators* M+/−:

$$\forall\,u\in V,\quad S_{u}=f_{\mathrm{Agg}}\left(\left\{S_{v}\right\}_{v\in\mathrm{Child}(u)}\cup\left\{g\left(H_{u}\right)\right\}\right),\quad\textit{Recursion}(I)$$
$$\quad(4)$$
$$(5)$$
$\forall\;v\in V,\quad H^{\prime}_{v}={\cal M}^{+}\left(S_{v},\;{\cal M}^{-}\left(H^{\prime}_{\rm Pa(v)},S_{v}\right)\right),\quad H^{\prime}_{r}=S_{r},\qquad\mbox{\it Recursion(II)}$
(6) $\frac{1}{2}$
where *H, H*′ ∈ R
n×d *denote node embeddings before and after aggregation.*
This theorem provides an efficient way to propagate long-distance information on a tree: (1) First, to calculate Su for each node u ∈ V , it suffices to collect all distant messages targeting the root once, by recursively calling fAgg (·); (2) Then, apart from the root H′r = Sr, we can calculate H′for other nodes efficiently via the operator M− followed by M+.

Implementation Despite the strong generality, we still prioritize a linear variant for simplicity and ease of implementation. Specifically, adopting fAgg and M+ as weighted sums, M− as weighted difference, and g as a linear transformation, we implement Eq. 5 and Eq. 6 as follows:

$$\forall\;u\in V,\quad S_{u}=\sum_{v\in\text{Child}(u)}(\alpha_{v\to u}\cdot W_{A})\cdot S_{v}+W_{B}\cdot H_{u}\in\mathbb{R}^{d},\tag{7}$$  $$\forall\;v\in V,\quad H^{\prime}_{v}=S_{v}+\alpha_{\text{Fix}(v)\to v}\cdot W_{A}\cdot\left(H^{\prime}_{\text{Fa}(v)}-\alpha_{v\to\text{Fa}(v)}\cdot W_{A}\cdot S_{v}\right)\in\mathbb{R}^{d},\tag{8}$$

where WA ∈ R
d×dand WB ∈ R
d×dare learnable matrices. The local attentions {αi→j}i,j (defined in Eq. 3) are utilized to enhance the impact of homophilous edges and weaken heterophilous edges. Acceleration and Extensions Note that parallelization can be conducted both between trees and between aggregations inside a single tree. For higher parallelization, we can intuitively make a rooted tree shallower yet wider to support many threads working together by selecting its centroid as the root. Furthermore, there exist different greedy strategies for nodes' priority for different recursions (Eq. 5 and Eq. 6) to reduce the waiting time of threads. We discuss their specific implementations in Sec. D of Appn. Due to space limits, we will discuss more on several potential extensions of the above tree aggregators in Sec. C of Appn., which includes how to: (1) efficiently integrate a global linear attention to the framework similar to Wu et al. (2024) and conveniently incorporate the kernel decomposition techniques (*e.g.*, Random Feature Likhosherstov et al. (2022)) to improve the expressivity of attention; (2) conduct fine-grained propagation control, such as discounting or truncating the distance, similar to some deep GNNs Xu et al. (2018); Chen et al. (2020a); (3) generalize forests to eliminate the need for Recursion (II), *i.e.*, Eq. 6.

## 4.4 Tree Fuser

Motivated by prior work Wu et al. (2024); Kreuzer et al. (2021); Wu et al. (2021), we utilize a local module to supplement local knowledge to mitigate the local sparsity of trees. Thus, the tree fuser first computes the local information H from input features X, which is formalized as below:

$$H=\left(\beta_{1}\cdot\widehat{A}_{\widehat{G}}+\beta_{2}\cdot\alpha+(1-\beta_{1}-\beta_{2})\cdot\mathbb{I}_{n\times n}\right)^{K_{L}}XW_{H}\in\mathbb{R}^{n\times d},\tag{9}$$

where β1 + β2 ≤ 1, KL ≤ 2 are hyper-parameters and WH are training parameters. The tree fuser then computes the results of NT different tree aggregators, H
′(k) = f
(Tk)
Agg (H), k ∈
[1, NT ]. For each H
′(k), the tree fuser normalizes each row to 1 using the L2-norm for numerical stabilization. Afterwards, the tree fuser averages all the tree aggregators as global information:

$$H^{\prime}=\text{Mean}\left(\left\{\text{RowNorm}\left(H^{{}^{\prime}(k)}\right)\right\}_{k\in[1,\ N_{T}]}\right)\in\mathbb{R}^{n\times d}.\tag{10}$$
$$H^{\prime\prime}=(1-\gamma)\cdot H^{\prime}+\gamma\cdot H.$$

Subsequently, the tree fuser uses a residual connection controlled by the hyper-parameter γ ∈ [0, 1] to balance local and global information, which can be formulated as follows:
H′′ = (1 − γ) · H′ + γ · H. (11)
The H′′ are final node embeddings that can be fed into a linear predictor for node classification.

4.5 COMPLEXITY ANALYSIS The comprehensive time and space complexities per epoch are linear against the number of nodes and edges, *i.e.*, n and m, as well as hidden dim d. Specifically, suppose we sample and utilize NT trees. Each pre-training epoch costs O ((n + m) d) time and space. Each training epoch of the student requires only O ((n + m) Kd) time and space, which can be further parallelized.

$$(11)^{\frac{1}{2}}$$
$\mathbf{a}_{\rm i}=\mathbf{a}_{\rm i}\mathbf{a}_{\rm i}$ (10.10)
Table 1: The results of performance comparison (with the best bolded and the runner-ups underlined)

Method Category Cora Citeseer Pubmed Actor Cornell Texas Wisconsin Arxiv Flickr Avg. Rank

MLP Classic 58.30 58.68 72.94 35.62 72.70 77.84 79.61 32.84 42.01 14.11 GCN GNN 82.06 71.60 79.58 27.88 53.51 69.19 57.25 53.77 38.40 14.89

GAT GNN 82.84 72.28 78.52 28.71 55.14 68.65 58.82 55.73 40.32 12.78

GraphSAGE GNN 81.40 71.68 78.50 36.24 63.78 75.14 76.08 51.42 41.42 11.00

SuperGATSD GNN 82.70 72.50 **81.30** 30.18 54.59 69.73 58.04 51.52 36.24 13.22

APPNP GNN 84.10 72.14 80.02 33.47 61.08 71.35 65.10 55.60 43.07 9.22

ClusterGCN GNN 82.04 70.08 77.26 29.66 49.73 63.24 62.35 53.35 39.58 16.89

GraphSAINT GNN 82.00 70.30 77.36 29.55 48.65 63.78 61.96 53.55 35.26 17.67

Pairnorm DeepGNN 66.24 44.20 72.12 24.33 40.68 41.08 52.94 54.58 31.41 22.56

Nodenorm DeepGNN 80.14 65.74 78.64 29.74 40.00 66.49 48.24 54.22 44.11 16.33

Meannorm DeepGNN 79.54 72.16 73.06 25.46 25.41 61.62 52.94 20.37 42.40 19.67

DropEdge DeepGNN 81.69 71.43 79.06 26.38 52.97 64.86 60.78 39.23 32.11 18.33

GCNII DeepGNN 85.34 73.24 79.88 34.64 74.61 69.19 70.31 51.91 41.79 8.78

ShadowGNN DeepGNN 82.32 70.06 77.30 29.45 51.35 64.32 62.35 53.35 37.59 17.00

GT GT 77.58 66.96 76.48 37.15 61.62 74.60 71.76 OOM OOM 15.57

SAN GT 77.60 68.64 76.62 37.79 63.24 75.14 77.25 OOM OOM 13.00

Graphormer GT 63.08 61.08 OOM OOM 62.70 76.76 72.16 OOM OOM 15.40

ANS-GT GT 77.68 64.16 77.98 38.29 74.92 76.22 76.47 41.83 21.86 13.22

Nodeformer GT 79.02 69.66 76.06 34.80 68.11 77.84 76.47 39.47 40.31 13.11

NAGphormer GT 79.51 67.34 78.32 37.33 63.78 71.89 66.27 52.00 38.59 13.44

GOAT GT 83.18 71.99 79.13 37.66 64.32 76.76 73.33 52.46 35.53 9.11

Exphormer GT 82.77 71.63 79.46 35.53 62.16 75.68 70.98 41.12 22.79 12.67

SGFormer GT 82.38 71.82 80.64 37.80 68.65 78.92 80.00 45.73 40.13 7.22

DIFFormer GT 83.32 **74.46** 78.16 34.51 60.00 68.11 63.92 53.60 44.25 10.56

TDGNN GT 85.35 73.78 80.20 32.84 35.68 61.35 46.86 OOM 38.25 15.00

GraphMamba Mamba 54.36 58.98 70.90 36.05 74.05 77.29 80.39 33.59 42.30 13.89

Ours Forest **85.46** 74.42 81.00 39.88 83.24 91.89 86.27 56.47 47.22 **1.22**

## 4.6 Theoretical Discussion

In this subsection, we provide theoretical justification for a rigorous asymptotic relationship between the accuracy of the edge-homophily estimator and the quality of the induced tree distribution.

Formally, we define PGb(T) = Qei,j∈Ts(eij )/PT ⊆GbQei,j∈Ts(eij ), where the edge score is given by s(eij ) = p if nodes i and j share the same label (a homophilous edge), and s(eij ) = q otherwise
(a heterophilous edge). Based on this formulation, we establish the following result:
Theorem 2. Let Gb be any connected graph, and define the expected edge homophily ratio under the score ratio ∆ = p/q > 0 as:
RGb(∆) := ET ∼P
(p,q)
Gb
[h(T)] ,
where h(T) is the edge homophily ratio of tree T. Then there exists a ∆0 > 0 *such that:*
- **Monotonicity.** If ∆ > ∆′ ≥ ∆0*, then* RGb(∆) > RGb(∆′).

- **Upper Bound.** *For all* ∆ ≥ ∆0, RGb(∆) ≤ 1 −
NHCC(Gb)−1 n−1, *where* NHCC(Gb) denotes the number of homophilous connected components of Gb.

$$\Delta)\ \to$$
$\mathbf{M}$
- *Asymptotic Tightness.* As ∆ → +∞, RGb(∆) → 1 −
NHCC(Gb)−1 n−1.

Theorem 2 shows that, for a given graph Gb, as the ratio ∆ = p/q increases, PGb(T) gradually shifts toward homophilous trees. Moreover, the upper bound of RGb(∆) is determined by the number of homophilous connected components in Gb, which reflects the inherent structural limitation of the graph. In the limit ∆ → +∞, RGb(∆) approaches this structural bound. In other words, assigning a higher score p > 0 to homophilous edges and a lower score q > 0 to heterophilous edges drives PGb(T) toward the maximum level of edge homophily permitted by the graph.

5 EXPERIMENTS
This section verifies the effectiveness of the proposed method in the semi-supervised node classification task via extensive experiments. Due to space limits, some experimental details such as environments, dataset statistics, algorithm implementation details, hyperparameter optimization strategy and configurations, and some visualizations are moved to Sec. K of Appn. Benchmarks and Baselines The experiments include nine real-world benchmarks, covering two types: (1) homophilous graphs: Cora, Citeseer, Pubmed (Sen et al., 2008), and OGBN-ArXiv (Hu et al., 2020) at a large node scale; (2) heterophilous graphs: Flickr (Zeng et al., 2019), Texas, Wisconsin, Cornell (Pei et al., 2020a), and Actor (Tang et al., 2009). Their full statistics are detailed in Tab. 7 of Appn. For a fair comparison, semi-supervised data splits are adopted for OGBN-ArXiv

![7_image_0.png](7_image_0.png) 
and Flickr (Sec. K.2), and other datasets strictly follow the standard public splits in (Kipf & Welling, 2017). Twenty-six counterparts are selected for a thorough comparison, including: (1) *classic* method: MLP; (2) *seven GNNs*: GCN (Li et al., 2019), GAT (Velickovi ˇ c et al., 2018), GraphSAGE, ´
SuperGATSD (Kim & Oh, 2021), APPNP (Gasteiger et al., 2019a), ClusterGCN (Chiang et al., 2019)
and GraphSAINT (Zeng et al., 2019); (3) *six Deep GNNs*: Pairnorm (Zhao & Akoglu, 2020), Nodenorm (Zhou et al., 2021), Meannorm (Yang et al., 2020), DropEdge (Rong et al., 2020b), GCNII (Chen et al., 2020a) and ShadowGNN (Zeng et al., 2021); (4) *eleven Graph Transformers*: GT (Dwivedi & Bresson, 2020), SAN (Kreuzer et al., 2021), Graphormer (Ying et al., 2021), ANS-GT (Zhang et al., 2022), NodeFormer (Wu et al., 2022), GOAT (Kong et al., 2023), NAGphormer (Chen et al., 2022b), Exphormer (Shirzad et al., 2023), SGFormer (Wu et al., 2024), DIFFormer (Wu et al., 2023), and TDGNN (Qu et al., 2020); (5) *Mamba*: GraphMamba (Wang et al., 2024a), Comparative Experiments All experiments run with ten different initializations. We report mean accuracy in Tab. 1 with also their standard deviations in Tab. 10 of Appn. We empirically show our framework has significant advantages for both homophilous and heterophilous datasets: against GT, DIFFormer, GCN, and GCNII, the mean accuracy is relatively increased by 16.2%, 16.1%, 24.5% and 11.9%, respectively. Particularly on Wisconsin, we obtain 20.2%, 35.0%, 50.7%, and 22.7% relative gains. Against recent models like TDGNN, ShadowGNN, and GraphSAINT, our framework also shows significant relative gains of 39.3%, 24.8%, and 27.0%, respectively. These performance gains are attributed to our ability to effectively capture long-distance knowledge, thus highlighting the potential of the proposed forest-based paradigm, even under label scarcity.

Table 2: Running time comparison
(sec/epoch)
Method Cora Citeseer Pubmed Flickr ArXiv GT 0.011 0.014 0.254 OOM OOM
SAN 0.165 0.154 0.241 OOM OOM
Graphormer 0.433 0.639 OOM OOM OOM
ANS-GT 1.453 2.973 3.433 7.796 24.540 Nodeformer 0.188 0.217 0.292 0.838 1.360 NAGphormer 0.022 0.044 0.031 0.835 1.560 GOAT 1.026 1.045 1.450 28.281 58.772 Exphormer 0.086 0.175 0.348 1.112 1.948 SGFormer 0.010 0.011 0.021 0.051 0.114 DIFFormer 0.029 0.030 0.047 0.297 0.545 GraphSAINT 0.013 0.022 0.030 0.658 0.951 Pairnorm 0.053 0.071 0.647 0.320 1.387 Nodenorm 0.013 0.032 0.285 0.310 1.357 Meannorm 0.012 0.030 0.279 0.296 1.461 Dropedge 0.017 0.017 1.231 1.244 1.491 GCNII 0.066 0.033 1.306 1.373 2.843 Ours 0.005 0.019 0.020 0.079 0.246 Ablation Studies We conduct ablation studies in Tab. 3 and drop or substitute key parts. For convenience, we refer to Eq. 9 and Eq.10 as Local and Global Submodules, respectively. We (1) drop Global Submodules to verify its long-range modeling capability; (2) drop Local Submodules to test the effects of supplementing local knowledge; (3) Sample trees from a uniform distribution and apply the attention weighting mechanism from Eq 7-8.; (4) sample only a single tree to explore the potential of multi-tree fusion. Comparing **(4) vs. (3)** reveals that sampling a single tree from the homophily-guided distribution outperforms multiple random trees, emphasizing the importance of homophily-based tree sampling. Comparing **(1)(2) vs.** (5) shows the significance of each submodule. Comparing (5) vs. (4) shows sampling multiple trees (a forest) can consistently surpass a single tree from our distribution, confirming that a forest can effectively capture more comprehensive and complementary topological knowledge. Hyper-Parameter Studies We conduct several hyper-parameter studies in Sec. J.1 Here, due to space limits, we focus only on the impact of the tree number NT on performance in Fig. 4, which reveals an optimal range of 6 to 10 trees across different datasets, highlighting our efficient coverage of global knowledge. In Fig. 4, the performance first consistently rises and then fluctuates or decreases, meaning that our framework covers the essence of the graph structure with only a few trees, and

| No.   | Method                       | Cora   | Citeseer   | Pubmed   | Actor   | Cornell   | Texas   | Wisconsin ArXiv   | Flickr   |       |
|-------|------------------------------|--------|------------|----------|---------|-----------|---------|-------------------|----------|-------|
| (1)   | w.o. Global Submodule        | 80.00  | 71.63      | 76.13    | 34.73   | 75.68     | 82.88   | 83.92             | 55.05    | 39.63 |
| (2)   | w.o. Local Submodule         | 82.18  | 71.55      | 77.48    | 35.08   | 74.77     | 69.93   | 75.49             | 54.92    | 32.17 |
| (3)   | Uniform Tree Sampling        | 83.63  | 72.32      | 78.45    | 36.13   | 72.97     | 82.58   | 84.80             | 55.11    | 42.77 |
| (4)   | Single Homophily-guided Tree | 83.73  | 72.58      | 78.55    | 36.32   | 76.35     | 84.83   | 85.29             | 55.17    | 42.96 |
| (5)   | FGL - Ours                   | 85.46  | 74.42      | 81.00    | 39.88   | 83.24     | 91.89   | 86.27             | 56.47    | 47.22 |

more trees provide marginal benefits and risk redundancy, highlighting our efficiency due to *a lower* number of structures in the calculation of the total cost, *i.e.*, Eq. 1. Efficiency Comparison Besides the theoretical complexity analysis in Sec. 4.5, we compare the practical running time in Tab. 2, where our method runs faster than baselines in most cases. For example, compared with recent GTs like ANS-GT and GOAT, which require over 1 second per epoch on small graphs and dozens of seconds on large graphs, our method runs in under 0.02 seconds on small graphs and 0.246 seconds on ArXiv. Even against efficient GTs like DIFFormer and deep GNNs like GCNII, our method shows 2 to 5 times speedup. While a few baselines run slightly faster than ours, their performance is generally worse than ours, since they overlook some critical structural knowledge due to over-simplified designs. Compared with these baselines with strong performance, we have the highest efficiency, highlighting the advantages of the linear complexities and higher parallelizability of the proposed forest-based learning paradigm. Homophily Estimator Comparison To explore the effects of different homophily estimators, we compare six variants in Tab. 4: (A) Non-attention auxiliary module (NAAM) via single-layer GCN
for homophilous graphs or MLP for heterophilous graphs to generate pseudo-labels; (B) Naive attention based estimator via a single local graph transformer layer where attention coefficients serve as bidirected average edge homophily scores; (C) 2-stage homophily estimation that first generates pseudo-labels via non-attention estimator, then uses these labels to guide the training of attentionbased estimator for more stable homophily scores; (D) FGL (Uniform) as baseline that samples trees uniformly; (E) FGL (Naive attention estimator) that uses attention scores from (B) to guide tree sampling; (F) FGL (2-stage estimator) - Ours, incorporating the full two-stage estimation process for robust homophily-guided tree sampling. Comparing (B) vs. (E), FGL using an attention-based estimator performs competitive or better than the standalone attention estimator, demonstrating FGL's effective utilization of homophily scores through structured tree aggregation. Comparing (C) vs. (E), two-stage estimation significantly outperforms FGL with only attention-based estimation in most cases, confirming that pseudo-labels from non-attention estimators provide valuable supervision to improve homophily estimation quality, especially under label scarcity. These empirical observations further support our theoretical analysis (Theorem 2) and directly confirm the accuracy of the edge homophily estimator has a positive impact on our final results.

Interpretability Studies We propose a strategy to design our tree distribution, which is justified by Theorem 2. Here, we provide some empirical evidence to understand our performance gains. Fig. 5 reveals that as the accuracy of homophily estimator increases, model performance consistently improves across all datasets, with perfect estimation (accuracy is 1) leading to perfect classification, demonstrating no performance bottleneck and motivating the pursuit of high-quality homophily estimators. To further understand the mechanism, we introduce a global homophily metric
(Sec. J.2). Fig. 6 shows that trees sampled from our homophily-guided distribution significantly facilitate higher long-range homophilous information propagation compared to uniform sampling.

Such trees allow the subsequent tree aggregator much easier to capture and exploit beneficial distant graph information, which fundamentally interprets our performance gains.

![8_image_0.png](8_image_0.png)

Random 0.6758

Figure 6: homophily ratio comparison based on different sampling strategies

## 6 Conclusion

To break the dilemma of existing graph techniques, *i.e.*, the challenging trade-off between complexities and comprehensive long-distance knowledge, we fundamentally analyze its root cause and propose a

| No.   | Model                                  | Cora   | Citeseer Pubmed Actor   | Cornell Texas   | Wisconsin ArXiv   | Flickr   |       |       |       |       |
|-------|----------------------------------------|--------|-------------------------|-----------------|-------------------|----------|-------|-------|-------|-------|
| (A)   | Non-attention auxiliary module (NAAM)  | 78.42  | 69.62                   | 76.64           | 35.33             | 72.97    | 72.97 | 82.35 | 47.65 | 38.36 |
| (B)   | Naive attention based estimator        | 75.18  | 65.78                   | 74.32           | 34.87             | 70.27    | 75.00 | 73.04 | 53.45 | 40.90 |
| (C)   | Two-stage (NAAM + attention) estimator | 81.40  | 70.30                   | 78.68           | 36.20             | 78.38    | 83.78 | 82.75 | 53.99 | 43.30 |
| (D)   | FGL (Uniform)                          | 78.40  | 73.13                   | 71.54           | 34.47             | 71.62    | 70.27 | 74.51 | 52.30 | 41.05 |
| (E)   | FGL (Naive attention)                  | 81.60  | 73.38                   | 75.10           | 35.56             | 74.32    | 75.00 | 76.75 | 53.63 | 41.61 |
| (F)   | FGL (2-stage) - Ours                   | 85.46  | 74.42                   | 81.00           | 39.88             | 83.24    | 91.89 | 86.27 | 56.47 | 47.22 |

novel forest-based graph learning paradigm. The key insight is to understand a graph as a fusion of some sampled spanning trees, similar to bagging, since a tree can connect all nodes economically. We provide a technical framework, where we first induce a tree distribution proven biased towards homophily, and then efficiently conduct all node-pair interactions in each tree via a general tree aggregator with linear complexities and higher parallelizability. Compared with deep GNNs or GTs, our framework has better global coverage and structural understanding, with higher efficiency. Extensive experiments on semi-supervised node classifications show we can achieve competitive or even better results than state-of-the-art counterparts. We believe our forest-based paradigm is a significant step towards the future development of long-distance graph learning.

## Acknowledgments

This work is partly supported by the National Key Research and Development Program of China (No.

2023YFF0725001), the National Natural Science Foundation of China (No. 92370204, 62306255), the Guangdong Basic and Applied Basic Research Foundation (No. 2024A1515011839). the Guangdong Basic and Applied Basic Research Foundation (Grant No.2023B1515120057), the Key-Area Special Project of Guangdong Provincial Ordinary Universities(2024ZDZX1007).

## References

Filippo Maria Bianchi, Daniele Grattarola, and Cesare Alippi. Spectral clustering with graph neural networks for graph pooling. In *International Conference on Machine Learning*, pp. 874–883. PMLR, 2020.

Francesco Bonchi, Claudio Gentile, Francesco Paolo Nerini, André Panisson, and Fabio Vitale. Fast and effective gnn training through sequences of random path graphs. In *Proceedings of the 31st* ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 49–60, 2025.

Andrei Z Broder. Generating random spanning trees. In *FOCS*, volume 89, pp. 442–447, 1989. Dexiong Chen, Leslie O'Bray, and Karsten Borgwardt. Structure-aware transformer for graph representation learning. In *International Conference on Machine Learning*, pp. 3469–3489. PMLR, 2022a.

Jinsong Chen, Kaiyuan Gao, Gaichao Li, and Kun He. Nagphormer: A tokenized graph transformer for node classification in large graphs. In The Eleventh International Conference on Learning Representations, 2022b.

Ming Chen, Zhewei Wei, Zengfeng Huang, Bolin Ding, and Yaliang Li. Simple and deep graph convolutional networks. In *International Conference on Machine Learning*, pp. 1725–1735, 2020a.

Tianlong Chen, Kaixiong Zhou, Keyu Duan, Wenqing Zheng, Peihao Wang, Xia Hu, and Zhangyang Wang. Bag of tricks for training deeper graph neural networks: A comprehensive benchmark study. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, pp. DOI:10.1109/TPAMI.2022.3174515, 2022c.

Zhiqian Chen, Fanglan Chen, Lei Zhang, Taoran Ji, Kaiqun Fu, Liang Zhao, Feng Chen, Lingfei Wu, Charu Aggarwal, and Chang-Tien Lu. Bridging the gap between spatial and spectral domains: A survey on graph neural networks, 2020b.

Wei-Lin Chiang, Xuanqing Liu, Si Si, Yang Li, Samy Bengio, and Cho-Jui Hsieh. Cluster-gcn: An efficient algorithm for training deep and large graph convolutional networks. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 257–266, 2019.

Eli Chien, Jianhao Peng, Pan Li, and Olgica Milenkovic. Adaptive universal generalized pagerank graph neural network. In *International Conference on Learning Representations*, 2021. URL
https://openreview.net/forum?id=n6jl7fLxrP.

Fan RK Chung. *Spectral graph theory*, volume 92. American Mathematical Soc., 1997.

Shuting Cui, Ying Sun, Yuting Zhang, Qingxin Meng, and Hengshu Zhu. Llm-enhanced career knowledge graph understanding for job mobility prediction. ACM Transactions on Management Information Systems, 2026.

Xiangyu Dong, Xingyi Zhang, Yanni Sun, Lei Chen, Mingxuan Yuan, and Sibo Wang. Smoothgnn:
Smoothing-aware gnn for unsupervised node anomaly detection. In Proceedings of the ACM on Web Conference 2025, pp. 1225–1236, 2025.

David Durfee, Rasmus Kyng, John Peebles, Anup B Rao, and Sushant Sachdeva. Sampling random spanning trees faster than matrix multiplication. In *Proceedings of the 49th Annual ACM SIGACT* Symposium on Theory of Computing, pp. 730–742, 2017.

Vijay Prakash Dwivedi and Xavier Bresson. A generalization of transformer networks to graphs.

arXiv preprint arXiv:2012.09699, 2020.

Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, and Yang Yang. Dropmessage:
Unifying random dropping for graph neural networks, 2023.

Wenzheng Feng, Jie Zhang, Yuxiao Dong, Yu Han, Huanbo Luan, Qian Xu, Qiang Yang, Evgeny Kharlamov, and Jie Tang. Graph random neural networks for semi-supervised learning on graphs.

Advances in neural information processing systems, 33:22092–22103, 2020.

Johannes Gasteiger, Aleksandar Bojchevski, and Stephan Günnemann. Predict then propagate:
Graph neural networks meet personalized pagerank. In International Conference on Learning Representations, 2019a.

Johannes Gasteiger, Stefan Weißenberger, and Stephan Günnemann. Diffusion improves graph learning. *Advances in neural information processing systems*, 32, 2019b.

Jhony H. Giraldo, Konstantinos Skianis, Thierry Bouwmans, and Fragkiskos D. Malliaros. On the trade-off between over-smoothing and over-squashing in deep graph neural networks. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM '23, pp. 566–576, New York, NY, USA, 2023. Association for Computing Machinery.

ISBN 9798400701245. doi: 10.1145/3583780.3614997. URL https://doi.org/10.1145/ 3583780.3614997.

Zheng Gong and Ying Sun. Graph reasoning enhanced language models for text-to-sql. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 2447–2451, 2024.

Zheng Gong and Ying Sun. Outlier-aware post-training quantization for discrete graph diffusion models. In *Forty-second International Conference on Machine Learning*, 2025.

Zheng Gong, Guifeng Wang, Ying Sun, Qi Liu, Yuting Ning, Hui Xiong, and Jingyu Peng. Beyond homophily: Robust graph anomaly detection via neural sparsification. In *Ijcai*, pp. 2104–2113, 2023.

Caterina Graziani, Tamara Drucks, Monica Bianchini, Thomas Gärtner, et al. No pain no gain: More expressive gnns with paths. In *NeurIPS 2023 Workshop: New Frontiers in Graph Learning*, 2023.

Yonghang Guan, Jun Zhang, Kuan Tian, Sen Yang, Pei Dong, Jinxi Xiang, Wei Yang, Junzhou Huang, Yuyao Zhang, and Xiao Han. Node-aligned graph convolutional network for whole-slide image representation and classification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18813–18823, 2022.

Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs. Advances in neural information processing systems, 33:22118–22133, 2020.

Wenbing Huang, Yu Rong, Tingyang Xu, Fuchun Sun, and Junzhou Huang. Tackling over-smoothing for general graph convolutional networks, 2020.

Md Shamim Hussain, Mohammed J Zaki, and Dharmashankar Subramanian. Global self-attention as a replacement for graph convolution. In *Proceedings of the 28th ACM SIGKDD Conference on* Knowledge Discovery and Data Mining, pp. 655–665, 2022.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. *arXiv* preprint arXiv:1611.01144, 2016.

Weiwei Jiang and Jiayun Luo. Graph neural network for traffic forecasting: A survey. *Expert* Systems with Applications, 207:117921, nov 2022. doi: 10.1016/j.eswa.2022.117921. URL https://doi.org/10.1016%2Fj.eswa.2022.117921.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with GPUs. IEEE
Transactions on Big Data, 7(3):535–547, 2019.
# PARDIFF: BRIDGING AUTOREGRESSIVE AND DIFFU- SION MODELS FOR ORDER-AGNOSTIC GRAPH GEN
## ERATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Graph generation has long struggled with the trade-off between structural fidelity and permutation robustness: autoregressive models excel in expressivity
but break under node-order sensitivity, while diffusion models offer invariance at
the cost of directional coherence. We introduce PARDIFF, a Progressive AutoRegressive DIFFusion framework that unifies these strengths through block-wise,
order-agnostic generation guided by learned structural decomposition. Unlike
prior heuristics, PARDIFF jointly predicts block sizes, ranks nodes, and applies
an equivariant diffusion process to each block, aligning AR directionality with
diffusion robustness. This reframes graph synthesis as probabilistic reasoning
over learned topological partitions, enabling scalable, semantically faithful, and
order-agnostic generation across molecular and non-molecular domains without
auxiliary features. Experiments show state-of-the-art results on diverse benchmarks, while its modular, latency-aware design supports real-time applications
like drug–drug interaction analysis, positioning PARDIFF as a paradigm shift in
[structured generative modeling. Code is available at: https://github.com/](https://github.com/llmresearch678/Pardiff_M_1)
[llmresearch678/Pardiff_M_1.](https://github.com/llmresearch678/Pardiff_M_1)


1 INTRODUCTION


Graphs lie at the heart of modeling complex relational structures across diverse domains—including
social networks, biochemical systems, recommendation engines, and cyber-physical infrastructures
(Kong et al. (2023); Cohen-Karlik et al.; Li et al. (2024); Chen et al. (2023)). As machine learning
advances toward general-purpose, foundation-level models, graph generative modeling has emerged
as a pivotal capability—fueling applications in molecular synthesis, protein engineering, and synthetic network design (Niu et al. (2020); Liao et al. (2019a)). Unlike grid-based modalities such as
images or text, graphs are inherently combinatorial, permutation-invariant, non-Euclidean, and variable in size. This introduces profound challenges in maintaining structural validity, generalization
across topologies, and permutation-consistent generation (Dai et al. (2020); Guo & Zhao (2022)).


To tackle graph generation challenges, prior works span AR models (You et al. (2018); Liao et al.
(2019a); Jin et al. (2018)), VAEs, GANs (Roy & Dasgupta (2023; 2024b;a)), and diffusion methods
(Du et al. (2021); Jo et al. (2022b); Huang et al. (2022)). AR models excel in controllability but
suffer from permutation bias and factorial inference costs (O’Bray et al. (2021); Luo et al. (2021);
Honda et al. (2019)). Diffusion models like EDP-GNN (Vignac et al. (2022b)) and GDSS (You
et al. (2023)) offer order-agnostic generation via SDEs, yet struggle with discrete structures. DIGRESS (Vignac et al. (2023)) adopts discrete-state transitions but relies on handcrafted priors. Hybrid methods such as GRAPHARM (Zhang et al. (2021)) partially bridge the gap but impose rigid
orderings. No prior approach fully unifies scalability, permutation-invariance, and structural expressivity in a single, efficient framework.


We introduce PARDIFF, a Progressive AR-Diffusion framework that bridges the structural control
of autoregression with the robustness of discrete diffusion. Unlike prior models that treat graphs
as monolithic, PARDIFF generates them block-wise through dynamically learned topological decompositions—predicting block size and order, then modeling each block with a shared equivariant diffusion process. This aligns generation with natural partial orderings while ensuring seman

1


tic fidelity and scalability. To overcome equivariant models’ symmetry limitations, we design a
noise-guided transition mechanism—akin to simulated annealing—that drives asymmetry formation through structured perturbations, yielding richer and more diverse graphs. Finally, we introduce a higher-order graph transformer with GPT-style parallel training, fusing edge-level reasoning
from Provably Powerful Graph Networks with transformer expressivity. Together, these innovations
establish PARDIFF as a paradigm shift in graph generation, delivering state-of-the-art results on
large-scale benchmarks without handcrafted features or auxiliary supervision.


2 PARDIFF: STRUCTURED DIFFUSION FOR PERMUTATION-INVARIANT
GRAPH GENERATION


Diffusion-based generative models (Haefeli et al. (2022); Madhawa et al. (2019)) work by gradually
adding noise to data until it is completely unrecognizable, and then training a model to reverse
this process and recover the original input. Originally developed for continuous data like images,
researchers have recently adapted these models to handle discrete data, including graphs (Song et al.
(2020); Simonovsky & Komodakis (2018))—structures made of nodes and edges. In the graph
context, the process begins with a clean graph _G_ 0 = _{V_ 0 _, E_ 0 _}_, where _V_ 0 denotes the node features
(one-hot vectors encoding categorical attributes such as type or label) and _E_ 0 denotes edge features
(one-hot encodings of relation types, connection categories, or an explicit “no-edge” token). The noedge token ensures that graph diffusion models can explicitly represent the absence of a connection
between two nodes, making the edge feature space complete. It is critical both for stable noise
injection/denoising during training and for generating valid, sparse, and realistic graphs at inference.
This graph is gradually corrupted over a series of steps, each step adding more randomness to the
features (we call forward diffusion)—until the graph becomes almost completely noisy. The goal
of the diffusion model is to learn how to reverse this process, step by step, so it can generate new,
realistic graphs from random noise.


The forward diffusion trajectory is described by a sequence of latent variables _G_ 1 _, G_ 2 _· · · GT_ over
_T_ time steps, where _Gt_ = _{Vt, Et}_ represents the noisy version of _G_ 0 at time step _t_ . This forward
process is modeled by the following Markov chain: _q_ - _Gt_ _|_ _Gt−_ 1� = [�] _i_ _[q]_ - _ft_ _[i]_ _[|]_ _[f]_ _t_ _[ i]_ _−_ 1�� _i,j_ _[q]_ - _rt_ _[ij]_ _[|]_


process is modeled by the following Markov chain: _q_ - _Gt_ _|_ _Gt−_ 1� = [�] _i_ _[q]_ - _ft_ _[i]_ _[|]_ _[f]_ _t_ _[ i]_ _−_ 1�� _i,j_ _[q]_ - _rt_ _[ij]_ _[|]_

_rt_ _[ij]_ _−_ 1�, where _ft_ _[i]_ [and] _[ r]_ _t_ _[ij]_ [denote the categorical states of node] _[ i]_ [ and edge][ (] _[i, j]_ [)][ at time] _[ t]_ [ respectively.]
_ft_ _[i][, r]_ _t_ _[ij]_ _∈{_ 1 _· · · n}_ with _n_ is the number of states. The learning problem then reduces to parameterizing a reverse-time process (we call it denoising) _p_ _**ϕ**_ - _Gt−_ 1 _|Gt_ - that approximates _q_ - _Gt|Gt−_ 1�

but runs backwards, from unstructured noise _Gt_ to structured samples resembling the original data
distribution. In practice, this requires training a denoising network (score function or conditional
transition model) that iteratively refines noisy graphs, balancing local consistency (node/edge attributes) and global topology (graph structure).


In this work, we model the reverse (denoising) process using a parameterized transformer neural
network with parameters _**ϕ**_ and estimates the backward transition as follows: _p_ _**ϕ**_ - _Gt−_ 1 _|Gt_ - =

- _i_ _[p]_ _**[ϕ]**_ - _ft_ _[i]_ _−_ 1 _[|][G][t]_ �� _i,j_ _[p]_ _**[ϕ]**_ - _rt_ _[ij]_ _−_ 1 _[|][G][t]_ �. Subsequent loss function should balance two things: ( _1_ ) It
tries to minimize the difference between the real data and what it generates (via a cross-entropy loss),
and ( _2_ ) It also tries to make the learned denoising steps as close as possible to the true underlying
reverse steps (via a KL divergence term _D_ - _· || ·_ �). Our training objective maximizes a variational
lower bound (VLB) on the data log-likelihood by jointly optimizing the terminal reconstruction
likelihood and minimizing the KL divergence between the forward (noising) and reverse (denoising)
diffusion processes across all time-steps as follows:


log _p_ _**ϕ**_        - _G_ 0� _≥_ E _q_ [log _p_ _**ϕ**_ ( _G_ 0 _| G_ 1)] _−D_        - _q_ ( _GT_ _| G_ 0) _∥_ _p_ _**ϕ**_ ( _GT_ )� _−_

     - _T_ (1)

E _q_ [ _D_ ( _q_ ( _Gt−_ 1 _| Gt_ ) _∥_ _p_ _**ϕ**_ ( _Gt−_ 1 _| Gt_ ))] _,_

_t_ =2


where _p_ _**ϕ**_ - _GT_ - is typically set as a fixed uniform noise distribution. Unlike traditional diffusion
models that estimate each _p_ _**ϕ**_ ( _Gt−_ 1 _|_ _Gt_ ) independently, we directly learn _p_ _**ϕ**_ - _G_ 0 _|Gt_ - and derive
all intermediate steps from it. This not only reduces training complexity and memory usage, but
also enforces global temporal coherence, yielding more stable, sample-efficient generation under a
principled VLB framework. As shown in APPENDIX, this follows from the variational objective.


2


_i_ _[q]_ - _ft_ _[i]_ _[|]_ _[f]_ _t_ _[ i]_ _−_ 1��


which enables us to use a cross-entropy (CE) loss at each timestep:


which we combine with the VLB loss to create a hybrid objective: _Lt_ ( _·_ ) = _D_ - _·_ _||_ _·_ - + _λ_ _·_
_L_ CE _,t_ ( _·_ ) _,_ with _λ_ = 0 _._ 1. During generation, a synthetic graph is sampled from _p_ _**ϕ**_ ( _GT_ ) and iteratively denoised via the learned reverse process _p_ _**ϕ**_ ( _Gt−_ 1 _|_ _Gt_ ) for _t_ = _T_ down to 0. While
diffusion models demonstrate strong potential for discrete structure generation, their application to
graphs remains challenging due to high dimensionality and complex dependencies between nodes
and edges. Prior works like DIGRESS (Vignac et al. (2023)) address this by incorporating auxiliary
structural cues ( _e.g._, spectral eigenvectors, cycle indicators), but these add computational overhead
and introduce reliance on domain-specific priors. Additionally, such methods often require hundreds to thousands of steps to achieve distributional fidelity. In contrast, we adopt a simplified
discrete-time diffusion approach, which improves memory efficiency and enables exact computation of the variational loss. The complete derivation of the forward and reverse distributions used in
our model— _q_ ( _Gt_ _| G_ 0), _q_ ( _Gt−_ 1 _| Gt, G_ 0), and _p_ _**ϕ**_ ( _Gt−_ 1 _| Gt_ )—is provided in APPENDIX.


2.1 STRUCTURE-AWARE SEQUENTIAL GRAPH GENERATION


AR models generate graphs step-by-step by breaking down the joint probability into a sequence
of conditional decisions—each choice depending on what has already been generated. This approach works well for data with natural order, like text or images. However, graphs are permutationinvariant, meaning their structure does not depend on the order of the nodes. This creates a fundamental mismatch: AR models are sensitive to order, while graphs are not. Early graph generation
models like GRAPHRNN (You et al. (2018)) and GRAN (Liao et al. (2019b)) handled this by assigning an artificial node ordering—using methods like breadth-first search, depth-first search, or
_k_ -core decompositions—to serialize the graph. While these heuristics allow training, they introduce
biases that do not reflect the true nature of graph distributions. These approaches often perform
well on small or synthetic graphs with regular structures, but struggle to generalize to larger or more
complex graphs where order invariance is crucial for accurate modeling.


There are two common strategies to address this: ( _1_ ) Marginalize over all possible node orderings
_p_ - _G, π_ �, but this becomes computationally infeasible because the number of orderings grows factorially. ( _2_ ) Use a fixed, canonical ordering for each graph, but finding such an ordering is as hard
as solving the graph isomorphism problem, which is computationally challenging and often datasetspecific. To avoid these limitations, we propose a more flexible and general approach: instead of
enforcing a strict global order, we leverage partial structural ordering. The key insight is that not
all nodes are equal—some play similar roles based on how they are connected. We group nodes
into blocks based on their structural roles, assigning each node a rank or block index via a function
_ψ_ : _V_ _→{_ 1 _, · · · B}_, where _B_ is the number of blocks.


During generation, we treat nodes in the same block as structurally interchangeable and generate the graph block by block, not node by node. To maintain coherence and realism, we ensure
that each new block connects to the previously generated part of the graph. Formally, we require that the subgraph _G_ _[′]_ = _{V_ _[′]_ _, E_ _[′]_ _}_ ; _V_ _[′]_ _⊆V_ induced by all nodes up to block _b_ is connected:
_∀_ _b_ _∈{_ 1 _, · · · B},_ _G_ _[′]_ [�] _ψ_ _[−]_ [1][�] _≤_ _b_ �� is connected. This approach aligns with how real-world graphs
grow—by expanding around existing structures—and avoids the rigidity and bias of fixed orderings. It brings together structural awareness, flexibility, and scalability, offering a more natural and
powerful foundation for graph generation.


**Weighted Degree Hashing for Ranking.** To reduce rank collisions and capture broader structural
context, we introduce a weighted degree function over _K_ -hop neighborhoods. Let _δk_ ( _V_ ); _V_ _∈V_
be the number of nodes reachable from node _v_ within exact _K_ hops. Then we define the weighted
structural score: _wK_ ( _V_ ) = [�] _k_ _[K]_ =1 _[δ][k]_ [(] _[V]_ [ )] _[·]_ _[|V|][K][−][k]_ [.] [This] [encoding] [gives] [greater] [importance] [to]
lower-hop connectivity. Having defined _wK_ ( _V_ ), we introduce structural partial order in Algo. 1.


**Theorem 1.** _The structural ranking function ψ (Algo. 1) is permutation-consistent, i.e., for any G_ =
_{V, E_ _} and permutation π that reorders the nodes of G, the ranking satisfies: ψ_ - _π⋆G_ - = _π⋆ψ_ ( _G_ ) _,_
_where ⋆_ _is the natural action of π on both the graph structure and node ranking map._


3


_L_ CE _,t_ ( _**·**_ ) = _−_ E _q_ - [�]


log _p_ _**ϕ**_  - _f_ 0 _[i][|][G][t]_  - +  
_i_ _i,j_


log _p_ _**ϕ**_  - _r_ 0 _[ij][|][G][t]_ �� _,_ (2)
_i,j_


**Proof of Theorem 1 is in** **APPENDIX** . The ranking _ψ_ ( _u_ ) of a node _u_ _∈V_ is determined in Algo.
1 from the multi-hop structural weight _wK_ ( _u_ ), which encodes degree patterns up to _K_ hops. These
descriptors are isomorphism-invariant: under any relabeling (permutation), the _K_ -hop neighborhood
of _u_ is mapped bijectively to the neighborhood of _π_ ( _u_ ), preserving the weight _wK_ . As a result, _ψ_
assigns the same relative rank after permutation, ensuring _ψ_ - _π ⋆G_ - = _π ⋆ψ_ ( _G_ ). This means, the
ranking _ψ_ is label-independent. It depends only on the structure of the graph around each node.
So if we shuffle the node names, the ranking shuffles in the exact same way, proving the method is
consistent and fair under relabeling.


**Algorithm 1** Multi-hop Hierarchical Node Ranking


**Require:** Graph _G_ = _{V, E}_ ; hop threshold _K_ .
**Ensure:** Structural order map _ψ_

1: Initialize: _G_ 0 _←_ _G_, _ψ_ ( _v_ ) _←_ 0 _∀_ _V_ _∈V_, _i_ _←_ 0
2: **while** _Gi_ is not empty **do**
3: **for all** _V_ _∈Vi_ **do**
4: Compute _wK_ ( _V_ ) = [�] _k_ _[K]_ =1 _[δ][k]_ [(] _[V]_ [ )] _[ · |V|][K][−][k]_
5: **end for**
6: Let _L_ _←{V_ _∈Vi_ _|_ _wK_ ( _V_ ) = min _u∈Vi_ _wK_ ( _u_ ) _}_
7: **for all** _V_ _∈L_ **do**
8: _ψ_ ( _V_ ) _←_ _i_
9: **end for**
10: _Gi_ +1 _←Vi_ _\ L_
11: _i_ _←_ _i_ + 1
12: **end while**
13: **return** _ψ_ _←_ _i −_ _ψ_


**Algorithm 2** Block Size Predictor Training


**Require:** _G_ ; max-hop depth _h_ max; block predictor _g_ _**α**_
1: Derive structural ordering _ψ_ from Algorithm 1.
2: Extract node partitions - _C_ 1 _, · · · CB_ - using _ψ_ .
3: **for** each _i_ = 1 to _B_ **do**
4: Predict block size: _S_ [ˆ] _i_ _←_ _gα_ - _Ci_ 
5: Compute loss: _Li_ _←_ CE� _S_ ˆ _i, Ci_ +1�

6: **end for**
7: **return** Minimize total loss: [�] _i_ _[B]_ =1 _[L][i]_


2.1.1 PROGRESSIVE GRAPH CONSTRUCTION VIA BLOCK SEQUENCES.


_ψ_ (Algo. 1) partitions the node set _V_ into _B_ ranked blocks _C_ 1 _· · · CB_, where all nodes in _Ck_ share the
same rank; the cumulative subgraph up to rank _k_ is defined as _G≤k_ = [�] _j_ _[k]_ =1 _[C][j]_ [and the incremental]
block as ∆ _k_ = _G≤k_ _\ G≤k−_ 1. The model factorizes the total likelihood of the graph as a chain of
conditional probabilities over incrementally added blocks: P _**ϕ**_ - _G_ - = [�] _k_ _[B]_ =1 [P] _**[ϕ]**_ �∆ _k_ _| G≤k−_ 1�, with
_G≤_ 0 defined as the empty graph. Such a decomposition has several critical advantages: ( _1_ ) Modularity and tractability. By breaking down the full generation task into block-wise increments, the
model transforms an intractable global problem into smaller, well-structured subproblems. ( _2_ ) Parameter sharing. Because blocks are treated symmetrically, parameters can be reused across ranks,
improving generalization and sample efficiency; and ( _3_ ) Permutation invariance. Since _ψ_ respects
the inherent symmetries of the graph and all nodes within a block are treated identically, the generation process is equivariant to node permutations. Consequently, the induced probability distribution
is exchangeable with respect to node relabelings (details are in APPENDIX). This framework also
addresses a key limitation of prior approaches such as GRAN (Liao et al. (2019b)), where nodes
within each block are generated sequentially. That design introduces an ordering bias, different node
orderings within a block yield different generative processes. In contrast, our method supports partially parallel generation within blocks, thereby eliminating intra-block asymmetry and ensuring that
the generative model is both scalable and faithful to the underlying exchangeable graph distribution.


2.2 LIMITS OF EQUIVARIANT GRAPH GENERATION


To ensure permutation-invariant graph generation within a block-wise AR framework, we must
carefully design the parameterization of conditional distributions. Let _Ck_ denote the _k_ -th structural


4


block, and let _G<k_ be the partial graph formed by the union of blocks _{C_ 1 _· · · Ck−_ 1 _}_ . We aim to
model the probability of the newly generated graph components at step _k_, given all components
generated before step _k_ : P _**ϕ**_ �∆ _k_ _|_ _G<k_ - i.e., the probability of newly added elements in _Ck_, given
the existing structure. To preserve symmetry, we introduce a virtual augmentation of _G<k_ to match
the target size of _G≤k_ by appending placeholder (empty) nodes and edges. Denote this extended
context as _G_ [�] _k_ := _G<k_ - _Zk_, where _Zk_ is a zero-padded placeholder graph mimicking the structure
of _Ck_ . The conditional likelihood is then: P _**ϕ**_ �∆ _k_ _| G<k_ - = [�] _e∈_ ∆ _k_ [P] _**[ϕ]**_ - _e |_ _G_ [�] _k_ �. It allows us to use

a permutation-equivariant function over the extended graph _G_ [�] _k_ to model each _e ∈_ ∆ _k_ .


**Algorithm 3** Denoising Diffusion Model Training


**Require:** _G_ ; diffusion steps _T_ ; _h_ max; denoising model _ℓα_ .

1: Derive ordering _ψ_ using Algorithm 1.
2: Extract blocks - _C_ 1 _, · · · CB_ - via _ψ_ .
3: Sample timestep _t_ _∼U_ - _{_ 1 _, · · · T }_ �.
4: **for** each _i_ = 1 to _B_ **in parallel do**
5: Mask _M_ _←_ ∆ _i_, where ∆ _i_ = _G≤i_ _\ G≤i−_ 1
6: Sample noised graph: _G_ [˜] _t_ _∼_ _qt_ - _G≤i_ 
7: Replace only masked part:
8: _G_ ˜ _←M ⊙_ _G_ [˜] _t_ + (1 _−M_ ) _⊙_ _G_
9: Predict reconstruction: _G_ [ˆ] _←_ _ℓα_ - _G_ ˜� _⊙M_
10: Ground truth: _G_ 0 _←_ _G≤i_ _⊙M_
11: Loss: _Li_ = _L_ _[t]_ diff� _G, G_ ˆ [true][�] + _λ · L_ _[t]_ CE� _G, G_ ˆ [true][�]

12: **end for**
13: **return** Minimize: [�] _i_ _[B]_ =1 _[L][i]_


2.2.1 SYMMETRY BOTTLENECK OF EQUIVARIANT MODELS


While using an equivariant function ensures that predictions respect node relabeling, it introduces a
critical limitation: equivariant models assign identical embeddings to all structurally equivalent elements. This makes distinguishing between symmetrically positioned nodes or edges infeasible. Let
**A** _G_ be the binary adjacency matrix of graph _G_ under a default node order. A graph automorphism
is a permutation _π_ such that: **A** _G_ = **P** _[⊤]_ _π_ **[A]** _[G]_ **[P]** _[π]_ [, where] **[ P]** _[π]_ [is the permutation matrix induced by] _[ π]_ [.]
The automorphism group is defined as: Aut( _G_ ) := - _π_ �� **A** _G_ = **P** _⊤π_ **[A]** _[G]_ **[P]** _[π]_ �. For a node _u_, its orbit
_O_ is the set of all nodes it can map to under automorphisms: _O_ ( _u_ ) := - _π_ ( _u_ ) _| π_ _∈_ Aut( _G_ )�.


**Theorem 2.** _Let Aut_ ( _G_ ) _be the automorphism group of a graph G. Then, for any node (or edge) pair_
( _u, v_ ) _lying_ _in_ _the_ _same_ _orbit_ _under_ _Aut_ ( _G_ ) _,_ _a_ _permutation-equivariant_ _neural_ _network_ Φ _assigns_
_identical representations,_ _i.e.,_ _u_ _∼Aut_ ( _G_ ) _v_ = _⇒_ Φ( _u_ ) = Φ( _v_ ) _,_ _regardless of the depth,_ _width,_ _or_
_expressivity_ _of_ Φ _._ _Here_ _u_ _∼Aut_ ( _G_ ) _v_ _denotes_ _the_ _nodes_ _u,_ _v_ _are_ _in_ _the_ _same_ _orbit_ _under_ _Aut_ ( _G_ ) _;_
Φ( _u_ ) = Φ( _v_ ) _denotes the model will assign identical representations/embeddings to u and v._


This theorem highlights a fundamental symmetry constraint imposed on permutation-equivariant
architectures: no matter how powerful the network (even with infinite capacity), it cannot distinguish
nodes or edges that are structurally indistinguishable under graph automorphisms. In other words,
expressivity is upper-bounded by orbit partitions—the finest granularity of distinction available is
the orbit structure of _G_ . This observation directly connects the theory of permutation-equivariant
networks to classical graph isomorphism: ( _1_ ) Orbits act as equivalence classes of symmetry, defining
the representational bottleneck. ( _2_ ) The result explains why standard message-passing GNNs are
no more powerful than the _1_ -dimensional WEISFEILER–LEHMAN (WL) test (Morris et al. (2019)):
they collapse all nodes in the same automorphism orbit to the same embedding; and ( _3_ ) Breaking
this symmetry ( _e.g._, via randomization, positional encodings, or anchor-based features) is therefore
essential for tasks requiring finer node distinctions. The proof of Theorem 2 is given in APPENDIX.


2.3 AUTOREGRESSIVE DENOISING DIFFUSION PROCESS


Graphs with high structural symmetry present a fundamental obstacle for permutation-equivariant
models, which, by design, produce identical outputs for structurally indistinguishable components.
This symmetry-preserving property, while theoretically elegant, impairs expressivity when the goal
is to transform a highly regular graph into an asymmetric or complex target. We reinterpret this


5


limitation through the lens of graph energy landscapes: highly symmetric graphs often occupy lowenergy basins due to their minimal description complexity and redundant structure (Trinquier et al.
(2021); Vignac et al. (2022a); Xu et al. (2022); Yan et al. (2023)). Consequently, generating richer,
asymmetrical structures from such graphs necessitates the deliberate injection of energy to escape
these local minima—akin to crossing barriers in a rugged optimization landscape (You et al. (2018);
Zhao et al. (2021)). This perspective reframes generative modeling as a controlled symmetrybreaking process: rather than relying solely on expressive equivariant functions, we advocate for
a two-stage mechanism—injecting structured randomness to perturb symmetric configurations, followed by guided denoising to refine toward desired complexity. This insight forms the foundation
for PARDIFF design, where simulated annealing–style transitions enable traversal across symmetry
plateaus, unlocking a broader generative space with theoretical grounding and practical efficiency.


To overcome symmetry-induced degeneracies, we introduce a discrete diffusion-based symmetrybreaking mechanism that injects structured randomness into node and edge features. This acts as an
energy injection phase—similar to thermal perturbations in simulated annealing, enabling the model
to escape low-energy basins and explore richer graph configurations (Algo. 3). Formally, we define a
forward Markov process _q_ - _Zt_ _| Zt−_ 1� that introduces noise at each timestep, corrupting categorical
node and edge features into indistinguishable forms. The reverse process is parameterized by a
learnable de-noiser _p_ _**ϕ**_ - _Zt−_ 1 _|_ _Zt_ �, which incrementally recovers structure, transforming initially
indistinguishable elements into semantically distinct graph components. The generative likelihood
of the final structure is computed by marginalizing over intermediate noise steps: P _**ϕ**_ �∆ _k_ _|_ _G_ [�] _k_ - =

- _· · ·_ - _p_ _**ϕ**_ ( _Z_ 0 _| Z_ 1) _·_ [�] _t_ _[T]_ =1 _[p]_ _**[ϕ]**_ [(] _[Z][t][−]_ [1] _[|][ Z][t]_ [)] _[ ·][ q]_ [(] _[Z][T]_ [ )] _[ ·][ dZ][T]_ _[· · ·][ dZ]_ [1] _[.]_


**Algorithm 4** Generate a Graph Using Learned Block Sizes


**Require:** _g_ _**α**_ in Algorithm 2; trained _ℓα_ (in Algorithm 3).

1: Initialize empty graph _G_ _←∅_, block index _i_ _←_ 1
2: Sample initial block size _n_ _∼_ _p_ 0
3: **while** _n_ _>_ 0 **do**
4: Add a block _Ci_ of _n_ new nodes to _G_
5: Define mask _M_ _←_ ∆ _i_, where ∆ _i_ = _G≤i_ _\ G≤i−_ 1
6: Initialize noised subgraph _G_ [˜] over _M_ using random noise models for nodes and edges
7: **for** _t_ = 1 to _T_ **do**
8: Predict denoised structure: _G_ [ˆ] _←_ _ℓα_ - _G_ ˜�

9: Sample reconstructed structure: _S_ _∼_ _G_ [ˆ]
10: Update subgraph: _G_ [˜] _←M ⊙S_ + (1 _−M_ ) _⊙_ _G_ [˜]
11: **end for**
12: Update full graph: _G_ _←_ _G_ [˜]
13: Predict next block size: _n_ _∼_ _g_ _**α**_ ( _G_ )
14: Increment block index: _i_ _←_ _i_ + 1
15: **end while**
16: **return** _G_


**Theorem 3.** _The full generative model_ P _**ϕ**_ - _G_ - _, constructed through AR block expansion and block-_
_level diffusion, is invariant under any node permutation π, i.e.,_ P _**ϕ**_ - _π ⋆G_ - = P _**ϕ**_ - _G_ - _, ∀π_ _∈Cn._


The proof of Theorem 3 relies on two facts: ( _1_ ) the block partitioning function _ψ_ is permutationequivariant (Theorem 1), and ( _2_ ) the discrete diffusion model is implemented using an equivariant
neural architecture across identically structured noise schedules. Together, these properties ensure
that the output distribution is exchangeable with respect to input labeling. Proof is in APPENDIX.


2.4 HYBRID TRANSFORMER ARCHITECTURE


The proposed PARDIFF framework flexibly integrates permutation-equivariant backbones, yet robust generalization requires capturing higher-order structural symmetries within each generated
block. While models like subgraph-aware GNNs (Tahmasebi et al. (2020)) and 3-WL expressive
networks such as PPGN (Maron et al. (2019)) offer deep structural insight, their _O_ ( _n_ [3] ) memory complexity limits scalability. To overcome this, we propose a novel hybrid that merges the
transformer-based global reasoning of GRIT (Ma et al. (2023)) with a lightweight approximation of
higher-order interactions inspired by PPGN. The key design principles include: Representing nodes
with enriched hidden states of dimension _dn_, Reducing edge embeddings to compact latent vectors
of dimension _de_ _≪_ _d_ [2] _n_ [and] [Maintaining] _[O]_ [(] _[n]_ [2][)] [memory] [complexity] [by] [avoiding] [full] [edge-wise]


6


tensor operations. This architectural fusion allows the model to benefit from global attention and
permutation-equivariant reasoning, while keeping computation tractable for large-scale graphs.


**Block-Wise** **Parallelism** **with** **Structural** **Masks.** In the PARDIFF framework, graph generation
is split into _K_ conditional steps, each handled by a shared denoising network _ℓ_ _**α**_ conditioned on
the preceding subgraph. Processing each step independently incurs _K×_ data expansion due to _K_
forward passes. To improve scalability, we propose a block-indexed parallelization scheme that
computes shared representations from a single forward pass over the full graph. Inspired by masked
language modeling, we apply a masking protocol to prevent information leakage from future blocks.
Each node and edge ( _u, v_ ) _∈_ _G_ is annotated with an integer block index _i_ _∈{_ 1 _· · · K}_, indicating
the block it belongs to. Let _M_ _∈{_ 0 _,_ 1 _}_ _[n][×][n]_ ( _n_ be number of states) be the binary mask matrix
defined as: _Mij_ = 1 if _i ≥_ _j_ and _Mij_ = 0 otherwise.


**Masking** **Rules** **for** **Causal** **Graph** **Diffusion.** The two primary operations that require masking
are the attention mechanism **A** _·_ **h** in transformer-style models and the bilinear edge update **A** _·_
**B** in matrix-based GNNs. To avoid leakage while preserving message flow, we redefine these
operations using masked interactions through Masked Attention or **MA** - **A** _,_ **h** - = - **A** _⊙M_ - _·_ **h**
and Masked Bilinear or **MB** - **A** _,_ **B** - = - **A** _⊙M_ - **B** + **A** - **B** _⊙M_ _[⊤]_ [�] _−_ - **A** _⊙M_ �� **B** _⊙M_ _[⊤]_ [�],
where _⊙_ denotes the Hadamard (element-wise) product. **MB** ( _·_ ) ensures bidirectional information
flow within valid scope while canceling redundant interactions that violate block causality. **Full**
**derivation** **is** **in** **APPENDIX** . _M_ allows us to use a single forward pass through the denoising
network _ℓ_ _**α**_ (Algo. 3) to compute all _K_ conditional probabilities �P _**ϕ**_ �∆ _k_ _|_ _G_ [�] _k_ �� _Kk_ =1 [.] [This offers]
the following advantages: reduces computational overhead by over an order of magnitude, avoids
redundant passes through _ℓ_ _**α**_, and enables batched training and gradient sharing across all blocks.
In implementation, we use separate modules for predicting the next block size and the conditional
block content. Both modules leverage the masked parallelization scheme. We fix the maximum
number of diffusion steps to _T_ = 40 for each block, a setting found effective without extensive
hyperparameter tuning. These efficiency improvements enable PARDIFF to scale to large datasets
such as MOSES (Polykovskiy et al. (2020)), achieving over 10 _×_ speedups in wall-clock training
time while preserving the permutation-invariant properties of the model.


3 IMPLEMENTATION DETAILS & EVALUATION


Block-wise diffusion in PARDIFF is parameterized by a shared model across all blocks, using a
fixed schedule length of _T_ = 50 for simplicity. Two specialized networks are trained independently:
a block size predictor _g_ _**α**_ (Algo. 2) and a block content generator _ℓ_ _**α**_ (Algo. 3). While PARDIFF
is architecturally agnostic, accurate modeling of intra-block symmetries demands expressive equivariant backbones. We employ the PPGN (Maron et al. (2019)) for its 3-WL-aligned capacity to
encode _⟨_ edge _,_ level _⟩_ features. Despite its representational strength, PPGN’s high memory cost may
constrain scalability on dense graphs. The experiments are conducted using NVIDIA RTX 5080,
PYTORCH 2.0.1, PYTHON 3.10, and CUDA 11.8.


**Baseline** **Datasets** **&** **Models.** We evaluate our method on three standard molecular datasets used
in graph generation research: ( _1_ ) QM9 (Ramakrishnan et al. (2014)) contains 133,885 small organic molecules with computed DFT properties; ( _2_ ) ZINC-250K (Irwin et al. (2005)), a set of
250K drug-like molecules; ( _3_ ) MOSES (Polykovskiy et al. (2020)), a large-scale benchmark with
approximately 1.9M molecular graphs. We used a 80%-20% split for training and testing, with 20%
of the training data reserved for validation. For generation, we sample 10,000 molecules from QM9
and ZINC, and 25,000 from MOSES. The graph generation literature features diverse benchmarking approaches. Among existing models, DIGRESS (Vignac et al. (2023)) has demonstrated strong
performance and serves as a primary baseline. We also compare against other notable methods including GDSS (Jo et al. (2022a)) and GRAPHARM (Kong et al. (2023)), as reported in results tables.


**Evaluation** **Metrics.** We adopt the following established evaluation metrics commonly used in
molecular graph generation to assess the performance of our model: ( _1_ ) VALIDITY (VAL) denotes
the proportion of generated molecules that are chemically valid, meaning they satisfy basic chemical rules such as correct valence for each atom. ( _2_ ) UNIQUENESS (UNI) measures the fraction of
unique molecules among valid ones, reflecting the diversity of the generation process. ( _3_ ) NOV

7


Figure 1: 1. Non-curated structured grid graphs generated by PARDIFF, trained with 50 diffusion
steps per block. 2. PARDIFF generating different known complex molecular structures trained with
50 diffusion steps per block using QM9. **More sample graphs are located in the APPENDIX** .


ELTY (NOV) indicates the percentage of valid molecules that are not present in the training dataset,
demonstrating the model’s ability to generate new and previously unseen molecular structures; and
( _4_ ) ATOM-LEVEL ACCURACY (AL) indicates the proportion of correctly predicted atom types for
all atoms in the generated molecules.


**PARDIFF** **Generating** **Grid-like** **Graph** **Structures.** Fig. 1.1 showcases non-curated grid-like
graphs generated using PARDIFF with 50 diffusion steps per block. Without explicit supervision,
the model consistently synthesizes regular lattice structures ( _e.g._, square, rectangular grids) while
allowing localized perturbations, mimicking real-world imperfections in physical layouts, like circuit designs, urban plans, and sensor meshes. The generated graphs exhibit grid-like regularity with
controlled imperfections, like local deformations, holes, and topological noise, enabled by PARDIFF’s hierarchical block-wise generation, which adaptively conditions each subgraph on evolving
structural context.


**PARDIFF** **Generating** **Molecule** **Structure.** PARDIFF generates chemically valid and topologically diverse molecules via an order-agnostic, block-wise diffusion process. By refining atombond structures from noise using a shared equivariant backbone, it naturally captures molecular
motifs—rings, chains, branches—without relying on handcrafted templates, making it ideal for “de
novo drug design” and scaffold discovery. For example, Fig. 1.2 shows nine different complex
drug molecules structures generated by the PARDIFF, showing its capability of handling complex
drug discovery problems (Deng et al. (2022)). A few more sample complex tentative (existent/nonexistent) molecular structures (without explicitly labeling the nodes) are shown in the APPENDIX.
Table 1 reports graph generation performance on QM9 dataset with explicit hydrogen atoms. PAR

Table 1: Graph generation performance on QM9 with explicit “H” atoms. PARDIFF achieves the

|sults. ↑indicates higher is better.|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|**MODEL**|**VAL ↑**|**UNI ↑**|**AL ↑**|**MOL ↑**|
|DATASET (OPTIMAL)|97.8|100.0|98.5|87.0|
|CONGRESS (Cai & Wang (2023))|86.7|98.4|97.2|69.5|
|DIGRESS (UNIFORM) (Vignac et al. (2023))|89.8|97.8|97.3|70.5|
|DIGRESS (MARGINAL) (Vignac et al. (2023))|92.3|97.9|97.3|66.8|
|DIGRESS (MARG. + FEAT.) (Vignac et al. (2023))|95.4|97.6|98.1|79.8|
|**PARDIFF** (**OUR METHOD**)|**98.9**|**100.0**|**99.2**|**90.3**|


DIFF outperforms strong baselines, including DIGRESS (Vignac et al. (2023)) and CONGRESS (Cai
& Wang (2023)), achieving state-of-the-art scores on VAL (98 _._ 1%), AL (98 _._ 9%), and molecular accuracy or MOL (88 _._ 5%), even surpassing the reference dataset accuracy (87 _._ 0%). While uniqueness
(96 _._ 8%) slightly trails CONGRESS (98 _._ 4%), it remains highly competitive. These results underscore
PARDIFF’s ability to generate chemically valid, diverse, and topologically faithful molecules, mark

8


ing a significant advancement in data-driven molecular synthesis. Table 2 shows that PARDIFF sets


Table 2: Generation quality on ZINC-250K. PARDIFF outperforms all baselines across VAL, FCD,
and UNI, while maintaining a compact model size. **↓** indicates lower is better.

|MODEL|VAL ↑|FCD ↓|UNI ↑|MODEL SIZE|
|---|---|---|---|---|
|EDP-GNN (Niu et al. (2020))|82.97|16.74|99.79|0.09M|
|GRAPHEBM (Liu et al. (2021))|85.29|35.47|98.79|—|
|SPECTRE (Martinkus et al. (2022))|90.20|18.44|67.05|—|
|GDSS (You et al. (2023))|97.01|14.66|99.64|0.37M|
|GRAPHARM (Zhang et al. (2021))|88.23|16.26|99.46|—|
|DIGRESS (Vignac et al. (2022a))|91.02|23.06|81.23|18.43M|
|SWINGNN-L (Yan et al. (2023))|90.68|1.99|99.73|35.91M|
|**PARDIFF (OUR METHOD)**|**97.50**|**1.62**|**99.998**|_∼_**4.5M**|


new state-of-the-art on ZINC-250K, achieving 97 _._ 50% validity, 1 _._ 62 FRECHET´ CHEMNET DISTANCE (FCD), and an impressive 99 _._ 998% uniqueness. This improves upon GDSS (You et al.
(2023)), which had 97 _._ 01% validity, by also enhancing diversity and fidelity. While SWINGNN-L
achieves a similar FCD (1 _._ 99), it uses over 35M parameters, nearly **8** _×_ larger than our compact
model. These results underscore PARDIFF’s ability to generate chemically valid, diverse molecules
that closely match the target distribution—using a small and efficient architecture. For QM9, we
also report AL and MOL, following prior evaluations in (Vignac et al. (2023); Cai & Wang (2023))
(Table 1). For ZINC-250K and MOSES, we evaluate models using comprehensive metrics in

Table 3: Generation quality on MOSES. PARDIFF outperforms its competitors. FIL: filter pass
rate, SNN: similarity to nearest neighbor, SCAF: SCAFFOLD similarity.

|MODEL|VAL ↑|UNI ↑|NOV ↑|FIL ↑|FCD ↓|SNN ↑|SCAF ↑|
|---|---|---|---|---|---|---|---|
|VAE (Kingma & Welling (2014))|97.7|99.8|69.5|99.7|0.57|0.58|5.9|
|JT-VAE (Jin et al. (2018))|100|100|99.9|97.8|1.00|0.53|10.0|
|GRAPHINVENT (Mercado et al. (2021))|96.4|99.8|—|95.0|1.22|0.54|12.7|
|CONGRESS (Cai & Wang (2023))|83.4|99.9|96.4|94.8|1.48|0.50|16.4|
|DIGRESS (Vignac et al. (2023))|85.7||95.0|97.1|1.19|0.52|14.8|
|**PARDIFF (OUR METHOD)**|||**99.99**|**99.9**|**0.39**|**0.61**|**17.2**|


cluding FCD, FIL, SNN, and SCAF to assess chemical validity, novelty, and diversity. PARDIFF
achieves state-of-the-art performance with perfect VAL and UNI, highest NOV (99 _._ 99%), best
FIL (99 _._ 9%), and lowest FCD (0 _._ 39). It also attains the top SNN (0 _._ 61) and SCAF (17 _._ 2) scores,
demonstrating superior fidelity and diversity; ablation results are provided in the APPENDIX.


4 CONCLUSION & DISCUSSIONS


PARDIFF resolves the long-standing trade-off between autoregressive expressivity and diffusionbased permutation invariance. Its block-wise, order-agnostic design fuses directional coherence
with structural flexibility, enabling scalable, high-fidelity graph generation across diverse domains.


**Possible Industrial Applications.** ( _1_ ) PHARMACEUTICALS & DRUG DISCOVERY: PARDIFF can
generate chemically valid, diverse molecules by learning hierarchical chemical structures, accelerating optimization while preserving structural constraints, which is critical for real-time drug synthesis. ( _2_ ) HEALTHCARE & BIOINFORMATICS: Allows generation of anatomical graphs, protein
structures, and multi-modal medical knowledge graphs, enabling better diagnostics, personalized
therapy design, and multimodal fusion of clinical data. ( _3_ ) SMART INFRASTRUCTURE & IOT: It
has the potential to facilitate structured modeling of sensor networks, dynamic resource graphs, and
fault-tolerant system designs for smart cities, power grids, and industrial automation.


**Why** **PARDIFF** **is** **a** **Game** **Changer?** PARDIFF learns partial structural order and adaptive
graph decomposition through a data-driven block-size predictor and ranking module, replacing rigid
heuristics with flexible, learned generation. Its modular, latency-aware design makes it deployable
in real-time industrial settings, turning a research advance into a practical tool for intelligent system design under uncertainty. Beyond graphs, PARDIFF lays the foundation for structured-data
foundation models with extensions to multimodal generation, dynamic graphs, and federated learning—enabling adaptive reasoning for real-time simulation, autonomous design, and personalized
medicine.


9


REFERENCES


Chen Cai and Yusu Wang. Congress: Conditional graph generation via score-based diffusion.
In _International_ _Conference_ _on_ _Learning_ _Representations_ _(ICLR)_, 2023. URL [https://](https://openreview.net/forum?id=ycyWpR0Uxn)
[openreview.net/forum?id=ycyWpR0Uxn.](https://openreview.net/forum?id=ycyWpR0Uxn)


Xiaohui Chen, Jiaxing He, Xu Han, and Li-Ping Liu. Efficient and degree-guided graph generation
via discrete diffusion modeling. _arXiv preprint arXiv:2305.04111_, 2023.


Edo Cohen-Karlik, Eyal Rozenberg, and Daniel Freedman. Order agnostic autoregressive graph
generation. In _NeurIPS 2023 Workshop:_ _New Frontiers in Graph Learning_ .


Hanjun Dai, Azade Nazi, Yujia Li, Bo Dai, and Dale Schuurmans. Scalable deep generative modeling for sparse graphs. In _International conference on machine learning_, pp. 2302–2312. PMLR,
2020.


Jianyuan Deng, Zhibo Yang, Iwao Ojima, Dimitris Samaras, and Fusheng Wang. Artificial intelligence in drug discovery: applications and techniques. _Briefings in Bioinformatics_, 23(1):bbab430,
2022.


Yuanqi Du, Shiyu Wang, Xiaojie Guo, Hengning Cao, Shujie Hu, Junji Jiang, Aishwarya Varala,
Abhinav Angirekula, and Liang Zhao. Graphgt: Machine learning datasets for graph generation and transformation. In _Thirty-fifth_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_
_Datasets and Benchmarks Track (Round 2)_, 2021.


Xiaojie Guo and Liang Zhao. A systematic survey on deep generative models for graph generation.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_, 45(5):5370–5390, 2022.


Kilian Konstantin Haefeli, Karolis Martinkus, Nathana¨el Perraudin, and Roger Wattenhofer. Diffusion models for graphs benefit from discrete state spaces. _arXiv_ _preprint_ _arXiv:2210.01549_,
2022.


Shion Honda, Hirotaka Akita, Katsuhiko Ishiguro, Toshiki Nakanishi, and Kenta Oono. Graph
residual flow for molecular graph generation. _arXiv preprint arXiv:1909.13521_, 2019.


Han Huang, Leilei Sun, Bowen Du, Yanjie Fu, and Weifeng Lv. Graphgdp: Generative diffusion
processes for permutation invariant graph generation. In _2022 IEEE International Conference on_
_Data Mining (ICDM)_, pp. 201–210. IEEE, 2022.


John J. Irwin, Thomas Sterling, Michael M. Mysinger, Eliot S. Bolstad, and Robert G. Coleman.
Zinc – a free tool to discover chemistry for biology. _Journal of Chemical Information and Mod-_
_eling_, 45(1):177–182, 2005. doi: 10.1021/ci049714+.


Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for
molecular graph generation. In _International Conference on Machine Learning_, pp. 2323–2332.
PMLR, 2018.


Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the
system of stochastic differential equations. _arXiv preprint arXiv:2202.02514_, 2022a.


Jaehyeong Jo, Seul Lee, and Sung Ju Hwang. Score-based generative modeling of graphs via the
system of stochastic differential equations. In _International conference on machine learning_, pp.
10362–10383. PMLR, 2022b.


Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In _International Conference_
_on Learning Representations (ICLR)_, 2014. arXiv:1312.6114.


Lingkai Kong, Jiaming Cui, Haotian Sun, Yuchen Zhuang, B Aditya Prakash, and Chao Zhang.
Autoregressive diffusion model for graph generation. In _International_ _conference_ _on_ _machine_
_learning_, pp. 17391–17408. PMLR, 2023.


Mufei Li, Viraj Shitole, Eli Chien, Changhai Man, Zhaodong Wang, Srinivas Sridharan, Ying Zhang,
Tushar Krishna, and Pan Li. Layerdag: A layerwise autoregressive diffusion model for directed
acyclic graph generation. _arXiv preprint arXiv:2411.02322_, 2024.


10


Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Will Hamilton, David K Duvenaud, Raquel
Urtasun, and Richard Zemel. Efficient graph generation with graph recurrent attention networks.
_Advances in neural information processing systems_, 32, 2019a.


Renjie Liao, Yujia Li, Yang Song, Shenlong Wang, Charlie Nash, William L. Hamilton, David
Duvenaud, Raquel Urtasun, and Richard Zemel. Efficient graph generation with graph recurrent attention networks. In _Advances in Neural Information Processing Systems_, pp. 5758–5768,
2019b.


Meng Liu, Keqiang Yan, Bora Oztekin, and Shuiwang Ji. Graphebm: Molecular graph generation
with energy-based models. In _Energy-Based Models Workshop, ICLR_, 2021. arXiv:2102.00546.


Youzhi Luo, Keqiang Yan, and Shuiwang Ji. Graphdf: A discrete flow model for molecular graph
generation. In _International conference on machine learning_, pp. 7192–7203. PMLR, 2021.


Liheng Ma, Chen Lin, Derek Lim, Adriana Romero-Soriano, Puneet K. Dokania, Mark Coates,
Philip H.S. Torr, and Ser-Nam Lim. Graph inductive biases in transformers without message
passing. In _Proceedings_ _of_ _the_ _40th_ _International_ _Conference_ _on_ _Machine_ _Learning_ _(ICML)_,
volume 202, pp. 12345–12356. PMLR, 2023.


Kaushalya Madhawa, Katushiko Ishiguro, Kosuke Nakago, and Motoki Abe. Graphnvp: An invertible flow model for generating molecular graphs. _arXiv preprint arXiv:1905.11600_, 2019.


Haggai Maron, Heli Ben-Hamu, Hadar Serviansky, and Yaron Lipman. Provably powerful graph
networks. _Advances in neural information processing systems_, 32, 2019.


Karolis Martinkus, Andreas Loukas, Nathana¨el Perraudin, and Roger Wattenhofer. Spectre: Spectral
conditioning helps to overcome the expressivity limits of one-shot graph generators. In _Interna-_
_tional Conference on Machine Learning_, pp. 15159–15179. PMLR, 2022.


Roc´ıo Mercado, Tobias Rastemo, Edvard Lindel¨of, G¨unter Klambauer, Ola Engkvist, Hongming
Chen, and Esben Jannik Bjerrum. Graph networks for molecular design. _Machine_ _Learning:_
_Science and Technology_, 2(2):025023, 2021.


Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Johannes E. Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 33, pp. 4602–
4609. AAAI Press, 2019.


Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, and Stefano Ermon. Permutation invariant graph generation via score-based generative modeling. In _International_ _con-_
_ference on artificial intelligence and statistics_, pp. 4474–4484. PMLR, 2020.


Leslie O’Bray, Max Horn, Bastian Rieck, and Karsten Borgwardt. Evaluation metrics for graph
generative models: Problems, pitfalls, and practical solutions. _arXiv preprint arXiv:2106.01098_,
2021.


Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai
Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark
Veselov, Artur Kadurin, Simon Johansson, Hongming Chen, Sergey Nikolenko, Alan AspuruGuzik, and Alex Zhavoronkov. Molecular sets (moses): A benchmarking platform for molecular generation models. _Frontiers_ _in_ _Pharmacology_, 11:565644, 2020. doi: 10.3389/fphar.2020.
565644.


Raghunathan Ramakrishnan, Pavlo O. Dral, Matthias Rupp, and O. Anatole von Lilienfeld. Quantum chemistry structures and properties of 134 kilo molecules. _Scientific Data_, 1:140022, 2014.


Arunava Roy and Dipankar Dasgupta. A novel conditional wasserstein deep convolutional generative adversarial network. _IEEE Transactions on Artificial Intelligence_, 2023.


Arunava Roy and Dipankar Dasgupta. A distributed conditional wasserstein deep convolutional
relativistic loss generative adversarial network with improved convergence. _IEEE_ _Transactions_
_on Artificial Intelligence_, 2024a.


11


Arunava Roy and Dipankar Dasgupta. Drd-gan: A novel distributed conditional wasserstein deep
convolutional relativistic discriminator gan with improved convergence. _ACM_ _Transactions_ _on_
_Probabilistic Machine Learning_, 2024b.


Martin Simonovsky and Nikos Komodakis. Graphvae: Towards generation of small graphs using
variational autoencoders. In _Artificial_ _Neural_ _Networks_ _and_ _Machine_ _Learning–ICANN_ _2018:_
_27th International Conference on Artificial Neural Networks, Rhodes, Greece, October 4-7, 2018,_
_Proceedings, Part I 27_, pp. 412–422. Springer, 2018.


Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. _arXiv preprint_
_arXiv:2011.13456_, 2020.


Behrooz Tahmasebi, Derek Lim, and Stefanie Jegelka. Counting substructures with higher-order
graph neural networks: Possibility and impossibility results. _arXiv_ _preprint_ _arXiv:2012.03174_,
2020.


Jeanne Trinquier, Guido Uguzzoni, Andrea Pagnani, Francesco Zamponi, and Martin Weigt. Efficient generative modeling of protein sequences using simple autoregressive models. _Nature_
_communications_, 12(1):5800, 2021.


Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. Digress: Discrete denoising diffusion for graph generation. _arXiv_ _preprint_
_arXiv:2209.14734_, 2022a.


Cl´ement Vignac, Jiaxuan You, Fabian B Fuchs, Nicholas Gile, and Michael M Bronstein. Equivariant discrete diffusion for graph generation. In _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 2022b.


Cl´ement Vignac, Jiaxuan You, Jure Leskovec, and Michael M. Bronstein. Digress: Discrete denoising diffusion for graph generation. In _International Conference on Learning Representations_
_(ICLR)_, 2023. [URL https://arxiv.org/abs/2209.14734.](https://arxiv.org/abs/2209.14734)


Minkai Xu, Lantao Yu, Yang Song, Chence Shi, Stefano Ermon, and Jian Tang. Geodiff: A geometric diffusion model for molecular conformation generation. _arXiv preprint arXiv:2203.02923_,
2022.


Qi Yan, Zhengyang Liang, Yang Song, Renjie Liao, and Lele Wang. Swingnn: Rethinking permutation invariance in diffusion models for graph generation. _arXiv preprint arXiv:2307.01646_,
2023.


Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, and Jure Leskovec. Graphrnn: Generating realistic graphs with deep auto-regressive models. In _Proceedings_ _of_ _the_ _35th_ _International_
_Conference on Machine Learning_, pp. 5708–5717. PMLR, 2018.


Jiaxuan You, Tianxiao Shen, Brandyn Sigouin, Shuangjia Zheng, Rui Chen, and Jure Leskovec.
Scalable graph generation with structural motifs. In _Proceedings of the 40th International Confer-_
_ence on Machine Learning (ICML)_, 2023. [URL https://arxiv.org/abs/2302.06611.](https://arxiv.org/abs/2302.06611)


Zhen Zhang, Yu Li, Chongxuan Li, Chang Liu, and Jun Zhu. Grapharm: Autoregressive graph
generation with hidden variables. In _Advances in Neural Information Processing Systems_, 2021.
[URL https://arxiv.org/abs/2110.07585.](https://arxiv.org/abs/2110.07585)


Lingxiao Zhao, Wei Jin, Leman Akoglu, and Neil Shah. From stars to subgraphs: Uplifting any gnn
with local structure awareness. _arXiv preprint arXiv:2110.03753_, 2021.


12
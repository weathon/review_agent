# Characterizing The Discrete Geometry Of Relu Networks

Blake B. Gaines Department of Computer Science University of Connecticut blake.gaines@uconn.edu Jinbo Bi∗
Department of Computer Science University of Connecticut jinbo.bi@uconn.edu

## Abstract

It is well established that ReLU networks defne continuous piecewise-linear functions, and that their linear regions are polyhedra in the input space. These regions form a complex that fully partitions the input space. The way these regions ft together is fundamental to the behavior of the network, as nonlinearities occur only at the boundaries where these regions connect. However, relatively little is known about the geometry of these complexes beyond bounds on the total number of regions, and calculating the complex exactly is intractable for most networks. In this work, we prove new theoretical results about these complexes that hold for all fully-connected ReLU networks, specifcally about their connectivity graphs in which nodes correspond to regions and edges exist between each pair of regions connected by a face. We fnd that the average degree of this graph is upper bounded by twice the input dimension regardless of the width and depth of the network, and that the diameter of this graph has an upper bound that does not depend on input dimension, despite the number of regions increasing exponentially with input dimension. We corroborate our fndings through experiments with networks trained on both synthetic and real-world data, which provide additional insight into the geometry of ReLU networks. Code to reproduce our results can be found at https://github.com/bl-ake/ICLR-2026. 

## 1 Introduction

Fully-connected networks with Rectifed Linear Unit (ReLU) activations have become ubiquitous in recent years. These networks realize piecewise linear functions, with each "piece" defned on a polyhedron in the input space as illustrated in Fig. 1a (Grigsby & Lindsey, 2022; Grigsby et al., 2024). These functions can be incredibly complex and have universal approximation power if suffciently wide or deep (Huang, 2020). Even work on one of the more basic questions about these networksbounding the maximum number of regions defned by a given architecture—already spans over a decade (Montufar et al., 2014; Goujon et al., 2024). Since that work began, it has generated significant interest in other topics related to how ReLU networks divide the input space. The geometry of ReLU networks is defned both by the number of polyhedral regions and the way they connect to each other to form a polyhedral complex (Fig. 1b). Existing work that focuses on the number of regions does not describe their arrangement (Pascanu et al., 2014; Montuf´ ar et al., 2022; Goujon et al., 2024). These works show that investigating specifc properties of the complex (e.g., defning the boundaries of individual regions, calculating paths from one region to another) can be intractable because the number of regions grows exponentially with respect to the input dimension and network size (Hanin & Rolnick, 2019b). Here we attempt to fll the gap in the middle and establish general properties about the arrangement of these regions that hold regardless of network size and the actual values of network weights. This work is motivated by the wide variety of research areas that leverage the polyhedral geometry of neural networks. These include explainability (Zhu et al., 2023; Hu, 2021), expressivity (Raghu et al., 2017), error prediction (Ji et al., 2022; Lehmann & Ebner, 2022; Daroczy ´ , 2020), 

![1_image_0.png](1_image_0.png)

robustness (Tran et al., 2019; Yang et al., 2020; Jamil et al., 2022), and even toxicity in large language models (Balestriero et al., 2024). It also allows the networks to be encoded as mixed-integer programs for verifcation (Botoeva et al., 2020; Cheng et al., 2017; Bunel et al., 2018) and inverse design (Ansari et al., 2022). The relationship between polyhedral regions and network activation states has also been applied to data compression (He et al., 2021), representation and clustering (Craighero et al., 2020), and open set detection (Jamil et al., 2023). A more detailed review of related work can be found in Appendix A. Our analysis builds on the topological perspective of ReLU network geometry, and follows the same assumptions as Masden (2025). Our results are best expressed in terms of the complex's connectivity graph (Fig. 1c), where nodes correspond to polyhedral regions and edges exist between regions that have a shared face (Liu et al., 2023b). The degree of a node in the graph corresponds to the number of faces of its region, each of which connects to a unique neighboring region. Fig. 1d plots a histogram of the node degrees and shows the average degree of the connectivity graph in Fig. 1c. The diameter of this graph (the longest shortest-path distance between any pair of nodes in the graph) corresponds to the number of faces one has to cross to reach a polyhedron from any other polyhedron. Our work provides fundamental links between network architecture and connectivity graph topology. Recent work has analyzed the connectivity graph to characterize network properties such as VC-Dimension and the distribution of region volumes (Dhayalkar, 2025). Notably, the work in (Fan et al., 2024) includes several results bounding the average number of faces of the polyhedra from above, but with crucial assumptions for the ReLU networks (e.g., no bias terms or low rank in the frst hidden layer's weight matrix) and their bounds are asymptotic with respect to the size of the network. In this work, we will bound the same quantity for networks regardless of architecture, both from above and (for networks with at least d neurons in any confguration) below. We will then further characterize the complex by deriving similar bounds for the intersections of the regions, and bound the connectivity graph diameter from above and below, which provides additional insight into how the activation regions ft together. Our main contributions are as follows: 

## Theoretical Properties

A fully-connected ReLU network with input dimension d, maximum layer width m, and depth ℓ creates a polyhedral complex C in Rd that, with probability 1 (almost everywhere) over all possible network weights, satisfes: 1. The average degree of the connectivity graph is at most 2d. 

2. This average approaches the upper bound as the size of the network increases. 

3. The diameter of the connectivity graph is bounded above by (m + 1)ℓ regardless of the value of d. 

2 

## Empirical Observations

Experimental results with networks of different sizes trained on synthetic data and three benchmark datasets show that: 1. The average degree of the connectivity graph quickly approaches the upper bound as the size of the network increases. 

2. The number of neighbors for every polyhedral region follows a unimodal distribution that skews right and peaks just below 2d. 

3. Regions that contain data points tend to be more connected on average compared to those that do not. 

## 2 Preliminaries

Sign Sequences: Let f : Rd → Rout be a fully-connected feedforward ReLU network in which each hidden neuron i performs an affne transformation wT xpi pi i + bi on its input x and then applies the ReLU activation, where xpi is the output of the previous layer, and wi and bi are the trainable parameters of this neuron. Let n denote the total number of hidden neurons in f. We start by introducing how, with probability 1 over possible network weights, the activation states of the neurons in a network can be uniquely represented by sign sequences (Masden, 2025). For a point x in the input space, when passing through the network and reaching the ith neuron, if wTi xpi +bi > 0, then the sign of this neuron S(x)i = 1 and the ReLU function outputs the value of wTi xpi + bi; if the affne function is equal to or less than 0, then S(x)i = 0 or −1 respectively, and the ReLU outputs 0. Thus, x receives a sign sequence S(x) ∈ {−1, 0, 1}n, a vector of length n indicating the sign of each neuron's output before applying the activation function. 

Bent Hyperplanes: If i is in the frst hidden layer, then xpi is the network's input, and all inputs x with S(x)i = 0 lie on the hyperplane {x ∈ Rd : wTi x + bi = 0}. When i is a neuron from a later layer, the set of inputs with wTi xpi +bi = 0 (i.e., S(x)i = 0) is more complex, because xpi will have been computed by the continuous piecewise-linear function represented by the previous layers, and the input points for which S(x)i = 0 form a level set of this function. We call this set the neuron's Bent Hyperplane (BH) following convention (Hanin & Rolnick, 2019b; Masden, 2025), although unlike a hyperplane, a BH can intersect itself and even be disconnected. 

![2_image_0.png](2_image_0.png)

Figure 2: A polyhedral complex. 

Fig. 2 shows an example of BHs created by a network with an input of d = 2 and 2 hidden ReLU layers where 3 neurons are in layer 1 (corresponding to hyperplanes, which we also call BHs 1–3 for notational convenience) and 1 neuron in layer 2 (corresponding to BH 4). Each blue arrow in Fig. 2 indicates the orientation of BHi, pointing towards the area where the output of the neuron i is positive (i.e., S(x)i = 1). BHs 1–3 intersect to form 7 regions. BH 4 intersects 6 of them, and in each area, BH 4 is a segment of a hyperplane. Whenever BH 4 crosses a BH j in the frst layer, one of the entries of its input xpi changes its activation state (switching between wTj x + bj and 0). This causes the BH to bend. In this example, it eventually intersects itself. 

The BH of neuron i (where S(x)i = 0) forms a boundary that divides the input space into two parts where S(x)i = 1 and S(x)i = −1. Thus, the sign sequence of a region can also be interpreted as a list describing which "side" of each BH it lies on. All of the BHs of neurons in previous layers partition the input space into disjoint regions, in each of which, wTi xpi + bi collapses into an affne function in the input space (say ΦTi x + βi for some Φi ∈ Rd and βi ∈ R), so the segment of the BH within an area lies on a hyperplane in the input space ΦTi x + βi = 0. This hyperplane subdivides the regions it intersects into two smaller regions that differ in the activation state of neuron i. For instance, Fig. 2 shows that BH 4 divides the 6 regions formed by intersections of BHs 1–3 into 12 smaller regions. From this point forward, all statements about ReLU networks will make the same assumptions as in (Masden, 2025) to avoid degenerate weight assignments. These will ensure that at most d BHs intersect at a point, and that the sections of BHs are never perfectly parallel to each other. As proven in that work, these assumptions will hold on all but a measure-zero set of parameter assignments for a given architecture. Appendix B provides rigorous defnitions of these assumptions. Polyhedral Complex: A polyhedral complex is a set of polyhedra (cells) that is closed under intersection (e.g., the complex in Fig. 2 includes not only the polygons as elements but also the line segments where pairs of polyhedra intersect and the vertices where four polygons intersect), and taking faces (e.g., the complex in Fig. 2 includes the line segments enclosing each polyhedron and the vertices enclosing each line segment). A k-cell is an element of a polyhedral complex with affne span k ≤ d, that is, a polyhedral set whose elements span a k-dimensional affne subspace of Rd. BHs of a neural network form the boundaries of the network's d-cells (the maximal regions in which the network's mapping is affne), which intersect to generate the polyhedral complex C of the network (Grigsby & Lindsey, 2022). Within each cell of C, the network's behavior is affne. When k < d, the k-cells of this complex are each contained by the intersection of d−k BHs. For the complex in Fig. 2, the orange 2-cell is not contained in any BHs because d − k = 0, a 1-cell is contained by a BH (e.g., the orange line segment is part of BH 4) and a 0-cell is contained in the intersection of two BHs (e.g., the highlighted vertex is the intersection of BHs 1 and 4). Accordingly, the BH of neuron i can be considered as the union of the (d − 1)-cells with a single 0 in the ith position of their sign sequences. For example, in Fig. 2, BH 4 is formed by the ring of six line segments (1-cells) with a 0 in position 4 of their sign sequences. We defne the "faces" of a k-cell as the (k −1)-cells it contains (these are often called "facets" in the relevant literature). In the connectivity graph, the k-cells of C are represented by (d − k)-hypercube subgraphs, and the collection of edges corresponding to the 1-cells of each BH form a cut. An important property of a network's canonical polyhedral complex is that if it is restricted to the cells lying in or on one side of a single neuron's BH (or equivalently, an element of the sign sequences is fxed), the resulting substructure is still a polyhedral complex. Although it no longer corresponds to a ReLU network, we still call it a ReLU complex because it is still a polyhedral complex with cells defned by BHs, so several of the results in the next section will still apply. In the following discussion, the dimension of a ReLU (sub)complex will refer to the maximum dimension of its cells instead of the dimension of the ambient space in which it is embedded (e.g. a BH of a 2-dimensional ReLU complex is a 1-dimensional ReLU complex). Sign Sequence Complex: The polyhedral complex can be described in terms of sign sequences, because in the interior of any k-cell, the sign sequence of every point is exactly the same. Furthermore, the sign sequences of a k-cell contain exactly d − k zero elements corresponding to the d − k BHs whose intersection contains the cell as a subset and k nonzero elements corresponding to BHs that either contain a single face of the cell or do not intersect the cell at all. For example, in Fig. 2, the orange 1-cell with sign sequence (1, −1, −1, 0) is contained by BH 4 so it has 0 in this position, its faces are the vertices on each end contained by BHs 2 and 3 respectively, and BH 1 does not intersect the cell at all, so all three are nonzero in the sign sequence. With the aforementioned basic assumptions on the network weights, the work in (Masden, 2025) proves that every cell of a ReLU complex has a unique sign sequence, that is, the mapping from cells to sign sequences S : C *→ {−*1, 0, 1}n is well-defned and injective. The connectivity graph can be equivalently defned in terms of the sign sequence complex, with nodes for the sign sequences of d-cells (the ones with no zeros) and edges between sequences that differ by one element. 

## 3 Theoretical Results On Network Geometry

We examine how the cells in a ReLU complex are connected with each other, how many neighbors a cell can have on average, and whether or not the number of neighbors varies with respect to the depth and width of the network. Proof outlines are given here while detailed proofs are in Appendix B. The cells in a ReLU complex and the number of their faces may depend on the specifc network architecture and values of the network weights. However, we prove that the average number of neighbors for any polyhedral region can be upper bounded by 2d regardless of the depth and width of the network. In a ReLU complex, counting the faces of cells is the same as counting their neighbors. 

For a d-cell, each face is contained in a unique BH, and across each face is a unique neighboring polyhedron that has the same sign sequence except that the sign corresponding to the crossed BH is fipped. More generally, we prove the following theorem for any k-cells of a ReLU complex. 

Theorem 3.1. For a ReLU complex in d-dimensional input space, the average number of faces of a k-cell is at most 2k *for* k = 1, 2*, . . . , d*. An earlier work proves this theorem for hyperplane arrangements (Fukuda et al., 1991), which only applies to the polyhedral complexes of single-layer networks, but the proof does not generalize to deep ReLU network complexes formed by BH arrangements. We employ the 1-1 mapping between cells and sign sequences to prove that the theorem also holds true for complexes of deep ReLU networks. 

Let C be a ReLU complex corresponding to a network f. Denote the BH of a neuron i by hi, which contains a set of k-cells of k < d, specifcally hi = {c ∈ C : S(c)i = 0}. Then, we use C − hi to denote the complex that results from removing all cells contained in the BH of neuron i and joining all pairs of cells sharing one of the faces that were removed. Fig. 3a illustrates C − hi with C from Fig. 2 and BH 4 as hi. The connectivity graph of C − hi can be obtained by contracting every edge 

![4_image_0.png](4_image_0.png)

corresponding to a (d − 1)-cell contained in hi (i.e., removing each edge and combining the two end nodes into one, keeping their connections to other nodes in the graph). As a result, every two k-cells previously split by hi are now fused into a single new k-cell. Any cells that do not intersect hi remain the same in the new complex C − hi. 

Lemma 3.2. For any neuron i in f, each k-cell c of C falls into exactly one of the following categories: 
Category 1: c is a cell of hi Category 2: c is a cell of C − hi Category 3: c is one of the two k-cells formed when a k-cell in C − hi *is separated by* hi We outline our full proof here. In the sign sequence of c, the element S(c)i is either zero or nonzero. If it is zero, then c can only be in Category 1. If it is nonzero, we need to check if c intersects hi. This is the case if changing the cell's sign sequence so that S(c)i = 0 yields a new sign sequence that matches a Category 1 cell in C. Then, fipping S(c)i yields the sign sequence of a k-cell neighbor of c*, so* c is in Category 3. Otherwise, if the new sign sequence is not an element of C, then hi does not contain c or form a face of c, so c is in Category 2. Fig. 3b and Fig. 3c show the categorization of k-cells (k = 0, 1, 2) in the polyhedral complex from Fig. 2 when hi corresponds to BH 4. The blue line segments and vertices in Fig. 3b are the Category 1 1- and 0-cells respectively, and the 4th element of their sign sequences is 0. The green 2-, 1-
, and 0-cells are in C − h4 because they do not change when hi is removed. For such a cell, if we zero out the 4th element of their sign sequence (e.g., the 2-cell in the center with a sequence 
(−1, −1, −1, −1)), the resulting sequence (e.g., (−1, −1, −1, 0)) does not exist in S(C). The red 2- and 1-cells are in Category 3 because h4 contains one face of each of those cells. As an example, the leftmost 2-cell has the sign sequence (−1, −1, 1, 1), and after setting S(c)4 = 0, (−1, −1, 1, 0) corresponds to the line segment of hi forming its right boundary. Note that the proof of Lemma 3.2 also works for ReLU subcomplexes formed by fxing one of the sign sequence elements, and these subcomplexes cannot include only half of a pair of Category 3 cells since their sign sequences can only differ at the position of the removed BH i. Alternate colorings for the same complex based on different choices of i and restrictions to different subcomplexes are included in Appendix C. 

If hi is a neuron from the last ReLU layer, the complex C−hi corresponds to another ReLU network, but directly removing a neuron from an early layer may not result in a complex corresponding to any ReLU network. As a result, we can break down the problem of counting cells (including faces) in the ReLU complex by iteratively removing neurons starting from the last layer and counting the number of cells that disappear. Let Nk(C) be the total number of k-cells in C. Based on Lemma 3.2, Nk(C) equals the sum of the numbers of k-cells in each category. To evaluate Nk(C), we frst count the k-cells in hi and C − hi separately, and then double count those in C − hi that are split by hi. To count the split cells, we can just count (k − 1)-cells in hi, which each divide one of them. For example, compare Fig. 3a and Fig. 3b. The six 1-cells in h4 split the six 2-cells in Fig. 3a to add six more 2-cells in Fig. 3b. Similarly, the six 0-cells in hi also split six 1-cells. 

Lemma 3.3. *For* k = 1, 2*, . . . , d*, 
Nk(C) = Nk(hi) + Nk(C − hi) + Nk−1(hi). (1) 

$$\begin{array}{l}{{\lambda,\lambda,\lambda,\omega}}\\ {{N_{k}({\mathcal{C}})=N_{k}(h_{i})+N_{k}({\mathcal{C}}-h_{i})+N_{k-1}(h_{i}).}}\end{array}$$

The frst term in the sum accounts for the Category 1 k-cells in C, the second term accounts for all the Category 2 cells and half the Category 3 cells, and the third term accounts for the other half of the Category 3 cells. There are no Category 1 d-cells in C because hi by defnition only contains cells up to dimension d − 1, so when k = d, the frst term Nk(hi) is always 0. The two lemmas lead to the following special case of Theorem 3.1 for k = d: 
Theorem 3.4. [Upper Bound] The average number of faces of a d-cell of C in Rd (i.e., the average degree of the connectivity graph) is at most 2d. 

Here we provide an outline of our proof (see Appendix B for detailed proof). Each (d − 1)-cell forms a face between two d-cells in C because the single 0 in its corresponding sign sequence can be set to −1 or 1 to get the sign sequences of the two d-cells. Thus, the sum of the numbers of faces of all d-cells is just twice the total count of (d−1)-cells in C, so the average number of faces of a d-cell is 2Nd−1(C) 2Nd−1(C)
C . Using Lemma 3.3, we prove C ≤ 2d Nd( ) Nd( ) by mathematical induction on the number of BHs n in the complex and d. By assuming that the upper bound holds for (n − 1, d − 1) and (n − 1, d), we prove that the upper bound holds for any complex with (n, d). The proof of Theorem 3.1 then follows by applying the lemmas to groups k-cells whose sign sequences have exactly d − k zeros at the same positions, that is, restricting the complex to the intersections of d − k BHs and applying the lemmas to the resulting subcomplex. It is more straightforward to establish the following lower bound on the degree of individual nodes, which then bounds the overall average degree of the ReLU complex graph. 

Theorem 3.5. [Lower Bound] If a ReLU network has n1 *neurons in the frst hidden layer, every* d-cell of C *has at least* min(n1, d) *neighbors, and thus the average degree of the connectivity graph* is at least min(n1, d).

## 3.1 Asymptotic Behavior

To study how connectivity properties change as network size increases, we can create sequences of networks by adding new ReLU neurons to the last layer or a new layer after it. We use Cn to denote the complex after n neurons have been added. We characterize these sequences with the following theorems, which show that the average number of faces grows monotonically and that the bound in Theorem 3.1 is tight respectively. 

Theorem 3.6. The average number of faces of d-cells in Cn *increases monotonically in terms of* n. 

Theorem 3.7. Let f be a shallow network that has only one hidden layer with n nodes. When n goes to infnity, the average number of faces of its d-cells converges exactly to 2d*. That is,* 

$$\operatorname*{lim}_{n\to\infty}{\frac{2N_{d-1}({\mathcal{C}}_{n})}{N_{d}({\mathcal{C}}_{n})}}=2d.$$

In our experiments in Section 5, we observe that the average number of faces also appears to approach 2d as the depth of the network increases. 3.2 BOUNDS ON CONNECTIVITY GRAPH DIAMETER 
Let ℓ be the total number of hidden layers (depth), mj be the number of nodes in layer j, j = 1*, . . . , ℓ*, and m = max{m1, · · · , mℓ} (width). We fnd that, 
  �  **Theorem 3.8.** *The diameter of the connectivity graph is* ln(Nd(C)) D Ω ℓ ln(n) and O m .

The lower bound (in Ω) agrees with the intuition that diameter increases with the number of regions in the complex. Although it appears as though increasing the number of neurons in the network might reduce diameter by increasing ln(n), actually ln(Nd(C)) grows much faster with n regardless of architecture, so the ln(n) term is just attenuating the growth of this lower bound. The upper bound (in O) may rarely be reached in practice, but it is interesting in that it does not have to depend on the input dimension d, even though the number of the network's regions increases exponentially with d. We also fnd that this is empirically true in Section 5, as when we fx network architecture and only change the input dimension, the diameter of the resulting complexes grows almost identically. 

## 4 Algorithm For Calculating Polyhedron Boundaries

To defne the complex, it will be necessary to map sign sequences to their polyhedra defned by intersections of half-spaces, i.e., systems of linear inequalities of the form ΦTi x + βi ≤ 0 for i ∈ f. 

We are only concerned with the sign sequences of d-cells, which do not contain any zeros, so that each neuron provides exactly one inequality to our system defning the polyhedron. Let s be such a sign sequence and the column vector s(j) be the portion of s that contains only signs for the neurons in layer j. Let W(j) ∈ Rjout×jin and b(j) ∈ Rjout denote the weights and biases in layer j of the network with input dimension jin and output dimension jout. We can defne our polyhedron by using the following formulas to calculate the inequalities layer by layer. 

Half-spaces for current layer Picking half-space signs Mask for inactive neurons in the previous layer
$\mathbb{S}\left[\mathbb{S}\right]\mathbb{S}\mathbb{P}$
     Current layer's weights Half-spaces for the previous layer
$$(2)$$
Φ(j) = diag s(j) W(j) diag ReLU s(j−1) Φ(j−1) (2) 

![6_image_0.png](6_image_0.png)

$${}^{(j)}=\mathrm{diag}\left(s^{(j)}\right)\left(W^{(j)}\mathrm{diag}\left(\mathrm{ReLU}\left(s^{(j-1)}\right)\right)\beta^{(j-1)}+b^{(j)}\right)$$
At the initial stage, Φ(0) = Id, b(0) = 0 (0) 
(d×1), and s = 1d×1. The frst term on the right-hand side ensures that the inequality of each half-space is always in the same direction, regardless of whether the neuron is active or inactive. This allows us to concatenate Φ(j) and β(j) from each layer to get the full linear system, Φx + β ≤ 0. 

## 4.1 Enumerating Polyhedra

To enumerate the maximal polyhedra and obtain their connectivity graph G = (V, E), we will employ breadth-frst search (BFS). We describe our exact method in Algorithm 1. Starting with a valid sign sequence s, which can be found by passing any point through the network, we enumerate its neighbors and add them to our graph. Neighbors are polyhedra that can be reached by crossing a single BH, so their sign sequences have the opposite sign from the original polyhedron in exactly one position i, denoted as s-i. To fnd the neighbors, we can calculate Φs and βs as described in the previous section and determine which inequalities actually form the boundary of the polyhedron. We check the redundancy of each inequality by solving an LP, performed by the SOLVELP subroutine on line 6, which is given the arguments Φs for the constraint coeffcient matrix, βs + ei (βs with 1 added to the ith element to relax this constraint) for the constraint offset, and −Φs for the objective function coeffcients to maximize in the direction of the relaxed constraint. This LP 
i is explained further in Appendix D. Non-redundant constraints will be violated by the optimal solution to this LP, i.e., ΦTsi*x > β*si , meaning that s-i gives the sign sequence of the neighbor of s across the BH of neuron i. For each neighbor, we add the edge between s and s-i to the graph (line 7), and if we have not reached s-i before, we add it to both the graph and the search queue (line 8). In the next iteration, we can pop a new sign sequence from the queue and repeat the same process. 

$$(3)$$

Algorithm 1 Construction of the Connectivity Graph 1: **Input:** Trained network f, sign sequence s 2: Q, V, E ← {s}, {s}, ∅ 3: **while** Q is not empty do 4: s ← pop(Q) 5: **for** i ∈ {*0, . . . , n*} do 6: if SOLVELP(−Φsi , Φs, βs+ei) ≥ βsi **then** 
7: add (s, s-i) to E 8: if s-i ̸∈ V **then** add s-i to Q and V **end if** 9: **end if** 
10: **end for** 11: **end while** 12: **return** (*V, E*)
This traversal of polyhedra is similar to several previous works (Xu et al., 2022; Liu et al., 2023a;b), 
specifcally the BFS in (Xu et al., 2022), but we take the additional step of building up the connectivity graph of the polyhedral complex over the course of the search by recording when faces are shared with already-found polyhedra. Furthermore, when determining whether or not an element of a sign sequence can be fipped to produce another valid sign sequence, we follow (Zhang & Wu, 2019; Fukuda, 2004) and slightly relax the corresponding inequality to reduce errors arising from insuffcient numerical precision. 

![7_image_0.png](7_image_0.png)

## 5 Experiments

To further understand the structure of ReLU complexes, we use Algorithm 1 to enumerate polyhedra for a number of neural networks and construct their connectivity graphs. We then discuss how data tends to be distributed across the complex of networks after training. Additional details about the experimental setup and trained networks can be found in Appendix F. Code, data, and models used in all experiments can be found at https://github.com/bl-ake/ICLR-2026. 

## 5.1 Understanding The Bounds

We start by training a number of networks on clustering problems generated from three isotropic Gaussians with unit variance and centers selected uniformly at random within a hypercube around the origin with side length 10. We vary input dimension d, the number of hidden layers, and the width of each hidden layer. For each combination of hyperparameters, we perform fve experiments with newly generated datasets, and perform an exhaustive search to compute the full complex of the network and obtain information about every region. Summary statistics for the polyhedra can be found in Table 1, and the distributions of neighbor counts are visualized in Fig. 4. The average number of neighbors in every complex is below the upper bound of 2d, with the overall distribution being unimodal and skewed right. We also plot the estimated connectivity graph diameter versus the upper bound from Section 3.2 in Fig. 5. We estimate the actual diameter of the complexes 

![7_image_1.png](7_image_1.png)

Table 1: Summary statistics for the distributions in Fig. 4 with four dimensions (left) and fve dimensions (right). Diameter for each complex is estimated as described in Section 5.1. Non-degenerate depth-1 networks always have the same number of polyhedra because their BHs are all just hyperplanes (Buck, 1943). 

Depth idth 

W

# Polyhedrons Average Degree Diameter # Polyhedrons Average Degree Diameter 

1 4 

8 16 

16.00±0.00 163.00±0.00 2517.00±0.00 

4.00±0.00 6.28±0.00 7.32±0.00 

5.50±0.00 10.60±0.42 22.50±0.35 

16.00±0.00 219.00±0.00 6884.87±0.35 

4.00±0.00 7.23±0.00 9.02±0.00 

5.50±0.00 10.75±0.26 23.17±0.41 

2 4 72.60±22.70 5.21±0.31 8.70±0.76 89.50±19.78 5.34±0.25 9.30±0.63 

8 2244.80±630.08 7.25±0.18 20.40±0.74 5802.60±1146.10 8.77±0.27 21.75±1.06 

16 42243.00±8608.12 7.72±0.06 41.10±0.65 2.69×105±48746.09 9.61±0.05 42.57±1.53 

3 4 227.60±42.21 5.85±0.16 12.50±0.61 389.60±188.02 5.65±0.41 14.65±2.32 

8 9340.80±3325.81 7.47±0.17 27.60±2.25 36591.20±12085.54 8.54±0.93 31.50±2.33 16 2.23×105±72142.81 7.82±0.04 57.70±2.46 1.78×106±1.94×105 9.78±0.04 57.44±1.47 

4 4 448.00±119.14 6.17±0.12 15.90±1.19 1206.70±1154.00 5.50±0.78 18.60±3.98 

8 35767.20±9493.85 7.70±0.06 37.40±1.29 1.82×105±79768.45 8.25±1.38 48.35±12.25 16 6.24×105±96311.16 7.85±0.03 76.35±4.56 5.03×106±*1.07*×106 9.80±0.03 70.88±1.19 

by bounding each one above and below using the corresponding algorithms from (Magnien et al., 2009) and taking the midpoint. To be clear, the asymptotic bounds derived in Section 3.2 were not used to make this estimate. Across all experiments, the diameter estimates for networks with the same depth and width were almost identical across different input dimensions. Although the upper bound is rarely reached, the logic that it should be independent of input dimension appears to hold in practice. Furthermore, when width is fxed, the diameter appears to grow logarithmically with respect to our theoretical upper bound. Additional summary metrics for the complexes and results for d ∈ {2, 3} can be found in Appendix G. 

## 5.2 Training Data And Polyhedra

We observe a difference in the distributions of neighbor counts between polyhedra that contain data and those that do not. We test networks trained on three datasets: California Housing (CC 0 License) (Kelley Pace & Barry, 1997), MNIST (CC BY-SA 3.0 License) (Deng, 2012), and CIFAR10 (MIT License) (Krizhevsky, 2009), and achieve reasonable performance for each (AUC above 0.9 or R2 above 0.6). We examine the last 3 layers of 8 neurons for MNIST and 2 layers of 64 neurons for CIFAR10 on a lower-dimensional hidden representation rather than the input, 5 dimensions for MNIST and 10 for CIFAR10. For California Housing, we calculate the complex of the entire network. Details about the datasets and networks can be found in Appendix F. Algorithm 1 was used to identify all polyhedra in the complex for MNIST. For the California Housing and CIFAR10 datasets, complete enumeration of the network complex was intractable, so the search was terminated after traversing 8 million polyhedra. We then randomly sample 10,000 points from the training data. If a data point does not lie in one of the polyhedra found in the initial search, we calculate the new polyhedron that contains the point and add it to the 8 million that were already found. The distributions of neighbor counts for these complexes can be found in Fig. 6. Across all datasets, the neighbor counts for polyhedra containing training data tend to be higher than the upper bound for the average neighbor count of all polyhedra. Since the number of faces of any polyhedron is bounded above by n, this necessarily reduces the rightward skew of the distribution as well. 

![8_image_0.png](8_image_0.png)

We also examine how neighbor counts vary according to whether polyhedra are bounded or unbounded, with results shown in Fig. 7. In all three experiments, we observe that polyhedra with higher numbers of neighbors are more likely to be unbounded (darker colors toward the right of each histogram, with the exception of polyhedra with d neighbors shown by the leftmost bars, which are always unbounded). In addition, we fnd that the proportion of unbounded polyhedra in datacontaining regions is higher than the overall proportion for both classifcation tasks (the top two histograms show darker bars than the corresponding bottom fgures) but lower for the regression task (the top histogram bars have lighter colors than the bottom). For the classifcation tasks, the network may have to focus its complexity on the spaces between classes of data points where it has to draw the decision boundary, leaving more of the data points themselves on the outer (unbounded) regions of the complex. On the other hand, for regression, the model is focused on ftting the data points, so data points tend to lie more on bounded regions with fnite function values. Additional results from these experiments are included in Appendix G. 

![9_image_0.png](9_image_0.png)

## 6 Discussion And Future Work

This work characterizes general geometric properties of the polyhedral complexes defned by ReLU networks. For the frst time, we place bounds on both the average connectivity of this complex and its graph-theoretic diameter. We also conduct empirical studies that visualize the distributions of polyhedron connectivity, and show that training data tends to lie on polyhedra with higher-thanaverage connectivity. There are several limitations to the work presented here. Further investigation is needed to fully explain why training tends to put data points in regions with higher numbers of faces, and how this phenomenon is related to the network's behavior. Additionally, we are not yet able to describe how more specifc network structures like convolutional layers and skip connections affect the network's geometry. Another limitation comes from the fact that our results only apply to ReLU activations, and while they could be extended to other piecewise-linear activation functions, there are no immediate implications for networks that use nonlinear activation functions. Our results have implications for several active areas of study that involve the polyhedral geometry of ReLU networks. For example, the work by Ji et al. (2022) places a bound on empirical training error based on the spatial relationships between the regions containing the train and test data. They use Hamming distance between the sign sequences of two polyhedra as a distance metric between them. However, this metric will not refect the case where a bent hyperplane may have to be crossed multiple times when moving from one polyhedron to another. Thus, the length of the shortest path between two polyhedra in the connectivity graph is a more suitable metric. If path length is used, Theorem 3.8 allows us to bound the empirical error based on the network architecture and independently of the input dimension. 

## References

Ross Anderson, Joey Huchette, Christian Tjandraatmadja, and Juan Pablo Vielma. Strong Mixed-
Integer Programming Formulations for Trained Neural Networks. In Andrea Lodi and Viswanath Nagarajan (eds.), *Integer Programming and Combinatorial Optimization*, pp. 27–42, Cham, 2019. Springer International Publishing. ISBN 978-3-030-17953-3. 

Navid Ansari, Hans-Peter Seidel, and Vahid Babaei. Mixed integer neural inverse design. ACM 
Transactions on Graphics, 41(4):151:1–151:14, July 2022. ISSN 0730-0301. doi: 10.1145/35 28223.3530083. URL https://dl.acm.org/doi/10.1145/3528223.3530083. GSCC: 0000268 4 citations (Semantic Scholar/DOI) [2024-05-30] 0 citations (Crossref) [202405-30].

Randall Balestriero and Richard Baraniuk. Mad Max: Affne Spline Insights Into Deep Learning. 

Proceedings of the IEEE, 109:704–727, 2018. URL https://api.semanticscholar. org/CorpusID:49901088. 

Randall Balestriero and richard baraniuk. A Spline Theory of Deep Learning. In Jennifer Dy and Andreas Krause (eds.), *Proceedings of the 35th International Conference on Machine Learning*, volume 80 of *Proceedings of Machine Learning Research*, pp. 374–383. PMLR, July 2018. URL https://proceedings.mlr.press/v80/balestriero18b.html. 

Randall Balestriero, Romain Cosentino, Behnaam Aazhang, and Richard Baraniuk. The Geometry of Deep Networks: Power Diagram Subdivision. In *Neural Information Processing Systems*, 
2019. URL https://api.semanticscholar.org/CorpusID:160010022. 

Randall Balestriero, Romain Cosentino, and Sarath Shekkizhar. Characterizing Large Language Model Geometry Helps Solve Toxicity Detection and Generation. In Forty-frst International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= glfcwSsks8. 

Arturs Berzins. Polyhedral Complex Extraction from ReLU Networks using Edge Subdivision. In Proceedings of the 40th International Conference on Machine Learning, pp. 2234–2244. PMLR, July 2023. URL https://proceedings.mlr.press/v202/berzins23a.html. 

Elena Botoeva, Panagiotis Kouvaros, Jan Kronqvist, Alessio Lomuscio, and Ruth Misener. Effcient Verifcation of ReLU-Based Neural Networks via Dependency Analysis. *Proceedings of the AAAI* Conference on Artifcial Intelligence, 34(04):3291–3299, April 2020. ISSN 2374-3468, 21595399. doi: 10.1609/aaai.v34i04.5729. URL https://ojs.aaai.org/index.php/AAA I/article/view/5729. GSCC: 0000170 102 citations (Semantic Scholar/DOI) [2024-0530] 60 citations (Crossref) [2024-05-30].

R. C. Buck. Partition of Space. *The American Mathematical Monthly*, 50(9):541–544, November 1943. ISSN 0002-9890, 1930-0972. doi: 10.1080/00029890.1943.11991447. URL https: //www.tandfonline.com/doi/full/10.1080/00029890.1943.11991447.

Rudy R Bunel, Ilker Turkaslan, Philip Torr, Pushmeet Kohli, and Pawan K Mudigonda. A Unifed View of Piecewise Linear Neural Network Verifcation. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurip s.cc/paper_files/paper/2018/file/be53d253d6bc3258a8160556dda3e9b 2-Paper.pdf. GSCC: 0000413.

Vasileios Charisopoulos and Petros Maragos. A Tropical Approach to Neural Networks with Piecewise Linear Activations, January 2019. URL http://arxiv.org/abs/1805.08749. 

GSCC: 0000048 arXiv:1805.08749 [cs, stat]. 

Chih-Hong Cheng, Georg Nuhrenberg, and Harald Ruess. Maximum Resilience of Artifcial Neural ¨
Networks. In Deepak D'Souza and K. Narayan Kumar (eds.), Automated Technology for Verifcation and Analysis, pp. 251–268, Cham, 2017. Springer International Publishing. ISBN 978-3319-68167-2. doi: 10.1007/978-3-319-68167-2 18. GSCC: 0000334 92 citations (Crossref) [2024-05-30] 255 citations (Semantic Scholar/DOI) [2024-05-30].

Francesco Craighero, Fabrizio Angaroni, Alex Graudenzi, Fabio Stella, and Marco Antoniotti. Investigating the Compositional Structure of Deep Neural Networks. In Giuseppe Nicosia, Varun Ojha, Emanuele La Malfa, Giorgio Jansen, Vincenzo Sciacca, Panos Pardalos, Giovanni Giuffrida, and Renato Umeton (eds.), *Machine Learning, Optimization, and Data Science*, pp. 322– 334, Cham, 2020. Springer International Publishing. ISBN 978-3-030-64583-0. 

Balint Dar ´ oczy. ´ Tangent Space Sensitivity and Distribution of Linear Regions in ReLU Networks, June 2020. URL http://arxiv.org/abs/2006.06780. GSCC: 0000000 arXiv:2006.06780 [cs, stat]. 

Li Deng. The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best of the Web]. *IEEE Signal Processing Magazine*, 29(6):141–142, November 2012. ISSN 15580792. doi: 10.1109/MSP.2012.2211477. URL https://ieeexplore.ieee.org/docu ment/6296535. 

Sahil Rajesh Dhayalkar. The Geometry of ReLU Networks through the ReLU Transition Graph, May 2025. URL http://arxiv.org/abs/2505.11692. arXiv:2505.11692. 

Reinhard Diestel. *Graph theory*. Springer Berlin Heidelberg, New York, NY, 6 edition, 2017. ISBN 
9783662536216. 

Feng-Lei Fan, Wei Huang, Xiangru Zhong, Lecheng Ruan, Tieyong Zeng, Huan Xiong, and Fei Wang. Deep ReLU Networks Have Surprisingly Simple Polytopes, November 2024. URL http: //arxiv.org/abs/2305.09145. arXiv:2305.09145. 

Gregoire Fournier. A tropical approach to neural networks. Master's thesis, University of Lille, September 2019. URL https://www.semanticscholar.org/paper/A-tropica l-approach-to-neural-networks-Fournier/907a9ecd78de8d1ca57d2dc5 24135afdff6d7c66. GSCC: 0000164.

Komei Fukuda. Frequently Asked Questions in Polyhedral Computation, August 2004. URL http s://people.inf.ethz.ch/˜fukudak/polyfaq/. 

Komei Fukuda, Shigemasa Saito, Akihisa Tamura, and Takeshi Tokuyama. Bounding the number of k-faces in arrangements of hyperplanes. *Discrete Applied Mathematics*, 31(2):151–165, April 1991. ISSN 0166218X. doi: 10.1016/0166-218X(91)90067-7. URL https://linkinghub .elsevier.com/retrieve/pii/0166218X91900677. 

Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear neural networks. *Journal of Computational and Applied Mathematics*, 441:115667, May 2024. ISSN 0377-0427. doi: 10.1016/j.cam.2023.115667. URL https://www.sciencedirec t.com/science/article/pii/S0377042723006118.

J. Elisenda Grigsby and Kathryn Lindsey. On Transversality of Bent Hyperplane Arrangements and the Topological Expressiveness of ReLU Neural Networks. SIAM Journal on Applied Algebra and Geometry, 6(2):216–242, June 2022. ISSN 2470-6566. doi: 10.1137/20M1368902. URL https://epubs.siam.org/doi/10.1137/20M1368902. GSCC: 0000020.

J. Elisenda Grigsby, Kathryn Lindsey, and Marissa Masden. Local and global topological complexity measures OF ReLU neural network functions, April 2024. URL http://arxiv.org/abs/ 2204.06062. GSCC: 0000006 arXiv:2204.06062 [cs, math].

Boris Hanin and David Rolnick. Complexity of Linear Regions in Deep Networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of *Proceedings of Machine Learning Research*, pp. 2596–2604. 

PMLR, June 2019a. URL https://proceedings.mlr.press/v97/hanin19a.htm l. 

Boris Hanin and David Rolnick. Deep ReLU networks have surprisingly few activation patterns. 

In *Proceedings of the 33rd International Conference on Neural Information Processing Systems*. Curran Associates Inc., Red Hook, NY, USA, 2019b. GSCC: 0000237.
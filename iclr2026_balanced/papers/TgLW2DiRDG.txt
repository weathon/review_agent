CHARACTERIZING THE DISCRETE GEOMETRY OF
RELU NETWORKS


**Blake B. Gaines**
Department of Computer Science
University of Connecticut
[blake.gaines@uconn.edu](mailto:blake.gaines@uconn.edu)


**Jinbo Bi** _[∗]_
Department of Computer Science
University of Connecticut
[jinbo.bi@uconn.edu](mailto:jinbo.bi@uconn.edu)


ABSTRACT


It is well established that ReLU networks defne continuous piecewise-linear functions, and that their linear regions are polyhedra in the input space. These regions
form a complex that fully partitions the input space. The way these regions ft
together is fundamental to the behavior of the network, as nonlinearities occur
only at the boundaries where these regions connect. However, relatively little is
known about the geometry of these complexes beyond bounds on the total number
of regions, and calculating the complex exactly is intractable for most networks.
In this work, we prove new theoretical results about these complexes that hold for
all fully-connected ReLU networks, specifcally about their connectivity graphs
in which nodes correspond to regions and edges exist between each pair of regions connected by a face. We fnd that the average degree of this graph is upper
bounded by twice the input dimension regardless of the width and depth of the network, and that the diameter of this graph has an upper bound that does not depend
on input dimension, despite the number of regions increasing exponentially with
input dimension. We corroborate our fndings through experiments with networks
trained on both synthetic and real-world data, which provide additional insight
into the geometry of ReLU networks. Code to reproduce our results can be found
[at https://github.com/bl-ake/ICLR-2026.](https://github.com/bl-ake/ICLR-2026)


1 INTRODUCTION


Fully-connected networks with Rectifed Linear Unit (ReLU) activations have become ubiquitous in
recent years. These networks realize piecewise linear functions, with each “piece” defned on a polyhedron in the input space as illustrated in Fig. 1a (Grigsby & Lindsey, 2022; Grigsby et al., 2024).
These functions can be incredibly complex and have universal approximation power if suffciently
wide or deep (Huang, 2020). Even work on one of the more basic questions about these networks—
bounding the maximum number of regions defned by a given architecture—already spans over a
decade (Montufar et al., 2014; Goujon et al., 2024). Since that work began, it has generated significant interest in other topics related to how ReLU networks divide the input space. The geometry
of ReLU networks is defned both by the number of polyhedral regions and the way they connect
to each other to form a polyhedral complex (Fig. 1b). Existing work that focuses on the number
of regions does not describe their arrangement (Pascanu et al., 2014; Montuf´ar et al., 2022; Goujon
et al., 2024). These works show that investigating specifc properties of the complex (e.g., defning
the boundaries of individual regions, calculating paths from one region to another) can be intractable
because the number of regions grows exponentially with respect to the input dimension and network
size (Hanin & Rolnick, 2019b). Here we attempt to fll the gap in the middle and establish general
properties about the arrangement of these regions that hold regardless of network size and the actual
values of network weights.


This work is motivated by the wide variety of research areas that leverage the polyhedral geometry of neural networks. These include explainability (Zhu et al., 2023; Hu, 2021), expressivity
(Raghu et al., 2017), error prediction (Ji et al., 2022; Lehmann & Ebner, 2022; Daroczy´, 2020),


_∗_


Corresponding Author


1


Average: 3.27


4


2


0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||

2 3


10


4


2


0
2 3 4 5


8


6


4


2


0
2 3 4 5


Figure 1: (a) An example ReLU network with a 2-dimensional input. (b) The corresponding polyhedral complex where region A has neighbors B, C, D, and E. (c) The connectivity graph where nodes
represent regions and edges link neighboring regions (so region A has degree 4). (d) A histogram of
the number of neighbors for each region, or equivalently the degrees of the connectivity graph.


robustness (Tran et al., 2019; Yang et al., 2020; Jamil et al., 2022), and even toxicity in large language models (Balestriero et al., 2024). It also allows the networks to be encoded as mixed-integer
programs for verifcation (Botoeva et al., 2020; Cheng et al., 2017; Bunel et al., 2018) and inverse design (Ansari et al., 2022). The relationship between polyhedral regions and network activation states
has also been applied to data compression (He et al., 2021), representation and clustering (Craighero
et al., 2020), and open set detection (Jamil et al., 2023). A more detailed review of related work can
be found in Appendix A.


Our analysis builds on the topological perspective of ReLU network geometry, and follows the same
assumptions as Masden (2025). Our results are best expressed in terms of the complex’s **connectiv-**
**ity graph** (Fig. 1c), where nodes correspond to polyhedral regions and edges exist between regions
that have a shared face (Liu et al., 2023b). The degree of a node in the graph corresponds to the
number of faces of its region, each of which connects to a unique neighboring region. Fig. 1d plots
a histogram of the node degrees and shows the average degree of the connectivity graph in Fig. 1c.
The diameter of this graph (the longest shortest-path distance between any pair of nodes in the
graph) corresponds to the number of faces one has to cross to reach a polyhedron from any other
polyhedron. Our work provides fundamental links between network architecture and connectivity
graph topology. Recent work has analyzed the connectivity graph to characterize network properties
such as VC-Dimension and the distribution of region volumes (Dhayalkar, 2025). Notably, the work
in (Fan et al., 2024) includes several results bounding the average number of faces of the polyhedra
from above, but with crucial assumptions for the ReLU networks (e.g., no bias terms or low rank
in the frst hidden layer’s weight matrix) and their bounds are asymptotic with respect to the size of
the network. In this work, we will bound the same quantity for networks regardless of architecture,
both from above and (for networks with at least _d_ neurons in any confguration) below. We will then
further characterize the complex by deriving similar bounds for the intersections of the regions, and
bound the connectivity graph diameter from above and below, which provides additional insight into
how the activation regions ft together. Our main contributions are as follows:


**Theoretical Properties**
A fully-connected ReLU network with input
dimension _d_, maximum layer width _m_, and
depth _ℓ_ creates a polyhedral complex _C_ in R _[d]_
that, with probability 1 (almost everywhere)
over all possible network weights, satisfes:

1. The average degree of the connectivity
graph is at most 2 _d_ .

2. This average approaches the upper bound
as the size of the network increases.


3. The diameter of the connectivity graph is
bounded above by ( _m_ + 1) _[ℓ]_ regardless of
the value of _d_ .


**Empirical Observations**
Experimental results with networks of different
sizes trained on synthetic data and three benchmark datasets show that:

1. The average degree of the connectivity
graph quickly approaches the upper bound
as the size of the network increases.


2. The number of neighbors for every polyhedral region follows a unimodal distribution
that skews right and peaks just below 2 _d_ .

3. Regions that contain data points tend to be
more connected on average compared to
those that do not.


2


2 PRELIMINARIES


**Sign Sequences:** Let _f_ : R _[d]_ _→_ R [out ] be a fully-connected feedforward ReLU network in which each
hidden neuron _i_ performs an affne transformation _wi_ _[T]_ _x_ _[p][i]_ + _bi_ on its input _xpi_ and then applies
the ReLU activation, where _x_ _[p][i]_ is the output of the previous layer, and _wi_ and _bi_ are the trainable
parameters of this neuron. Let _n_ denote the total number of hidden neurons in _f_ . We start by introducing how, with probability 1 over possible network weights, the activation states of the neurons
in a network can be uniquely represented by sign sequences (Masden, 2025). For a point _x_ in the
input space, when passing through the network and reaching the _i_ th neuron, if _wi_ _[T]_ _[x][p][i]_ [+] _[b][i]_ _[>]_ [0][, then ]
the sign of this neuron _S_ ( _x_ ) _i_ = 1 and the ReLU function outputs the value of _wi_ _[T]_ _[x][p][i]_ [+] _[b][i]_ [; if the ]
affne function is equal to or less than 0, then _S_ ( _x_ ) _i_ = 0 or _−_ 1 respectively, and the ReLU outputs
0. Thus, _x_ receives a sign sequence _S_ ( _x_ ) _∈_ _{−_ 1 _,_ 0 _,_ 1 _}_ _[n]_, a vector of length _n_ indicating the sign of
each neuron’s output before applying the activation function.


**Bent Hyperplanes:** If _i_ is in the frst hidden layer, then _x_ _[p][i]_ is the network’s input, and all inputs _x_
with _S_ ( _x_ ) _i_ = 0 lie on the hyperplane _{x_ _∈_ R _[d]_ : _wi_ _[T]_ _[x]_ [+] _[b][i]_ [=] [0] _[}]_ [. When ] _[i]_ [is a neuron from a later ]
layer, the set of inputs with _wi_ _[T]_ _[x][p][i]_ [+] _[b][i]_ [= 0] [(i.e., ] _[S]_ [(] _[x]_ [)] _[i]_ [= 0][) is more complex, because ] _[x][p][i]_ [will have ]
been computed by the continuous piecewise-linear function represented by the previous layers, and
the input points for which _S_ ( _x_ ) _i_ = 0 form a level set of this function. We call this set the neuron’s
Bent Hyperplane (BH) following convention (Hanin & Rolnick, 2019b; Masden, 2025), although
unlike a hyperplane, a BH can intersect itself and even be disconnected.


_**1**_


Fig. 2 shows an example of BHs created by a network with an
input of _d_ = 2 and 2 hidden ReLU layers where 3 neurons are in
layer 1 (corresponding to hyperplanes, which we also call BHs
1–3 for notational convenience) and 1 neuron in layer 2 (corresponding to BH 4). Each blue arrow in Fig. 2 indicates the
orientation of BH _i_, pointing towards the area where the output
of the neuron _i_ is positive (i.e., _S_ ( _x_ ) _i_ = 1). BHs 1–3 intersect
to form 7 regions. BH 4 intersects 6 of them, and in each area,
BH 4 is a segment of a hyperplane. Whenever BH 4 crosses a
BH _j_ in the frst layer, one of the entries of its input _x_ _[p][i]_ changes
its activation state (switching between _wj_ _[T]_ _[x]_ [+] _[b][j]_ [and ][0][). This ]
causes the BH to bend. In this example, it eventually intersects
itself.


Figure 2: A polyhedral complex.


The BH of neuron _i_ (where _S_ ( _x_ ) _i_ = 0) forms a boundary that
divides the input space into two parts where _S_ ( _x_ ) _i_ = 1 and _S_ ( _x_ ) _i_ = _−_ 1. Thus, the sign sequence
of a region can also be interpreted as a list describing which “side” of each BH it lies on. All of the
BHs of neurons in previous layers partition the input space into disjoint regions, in each of which,
_wi_ _[T]_ _[x][p][i]_ [+] _[b][i]_ [collapses into an affne function in the input space (say ][Φ] _[T]_ _i_ _[x]_ [+] _[β][i]_ [for some ][Φ] _[i]_ _[∈]_ [R] _[d]_
and _βi_ _∈_ R), so the segment of the BH within an area lies on a hyperplane in the input space
Φ _[T]_ _i_ _[x]_ [+] _[β][i]_ [=] [0][. This hyperplane subdivides the regions it intersects into two smaller regions that ]
differ in the activation state of neuron _i_ . For instance, Fig. 2 shows that BH 4 divides the 6 regions
formed by intersections of BHs 1–3 into 12 smaller regions.


From this point forward, all statements about ReLU networks will make the same assumptions as
in (Masden, 2025) to avoid degenerate weight assignments. These will ensure that at most _d_ BHs
intersect at a point, and that the sections of BHs are never perfectly parallel to each other. As proven
in that work, these assumptions will hold on all but a measure-zero set of parameter assignments for
a given architecture. Appendix B provides rigorous defnitions of these assumptions.


**Polyhedral Complex:** A polyhedral complex is a set of polyhedra (cells) that is closed under intersection (e.g., the complex in Fig. 2 includes not only the polygons as elements but also the line
segments where pairs of polyhedra intersect and the vertices where four polygons intersect), and
taking faces (e.g., the complex in Fig. 2 includes the line segments enclosing each polyhedron and
the vertices enclosing each line segment). A _k_ -cell is an element of a polyhedral complex with
affne span _k_ _≤_ _d_, that is, a polyhedral set whose elements span a _k_ -dimensional affne subspace of
R _[d]_ . BHs of a neural network form the boundaries of the network’s _d_ -cells (the maximal regions in
which the network’s mapping is affne), which intersect to generate the polyhedral complex _C_ of the
network (Grigsby & Lindsey, 2022). Within each cell of _C_, the network’s behavior is affne. When


3


_k_ _< d_, the _k_ -cells of this complex are each contained by the intersection of _d_ _−_ _k_ BHs. For the complex in Fig. 2, the orange 2-cell is not contained in any BHs because _d_ _−_ _k_ = 0, a 1-cell is contained
by a BH (e.g., the orange line segment is part of BH 4) and a 0-cell is contained in the intersection
of two BHs (e.g., the highlighted vertex is the intersection of BHs 1 and 4). Accordingly, the BH
of neuron _i_ can be considered as the union of the ( _d_ _−_ 1)-cells with a single 0 in the _i_ th position
of their sign sequences. For example, in Fig. 2, BH 4 is formed by the ring of six line segments
(1-cells) with a 0 in position 4 of their sign sequences. We defne the “faces” of a _k_ -cell as the
( _k_ _−_ 1)-cells it contains (these are often called “facets” in the relevant literature). In the connectivity
graph, the _k_ -cells of _C_ are represented by ( _d_ _−_ _k_ )-hypercube subgraphs, and the collection of edges
corresponding to the 1-cells of each BH form a cut.


An important property of a network’s canonical polyhedral complex is that if it is restricted to
the cells lying in or on one side of a single neuron’s BH (or equivalently, an element of the sign
sequences is fxed), the resulting substructure is still a polyhedral complex. Although it no longer
corresponds to a ReLU network, we still call it a ReLU complex because it is still a polyhedral
complex with cells defned by BHs, so several of the results in the next section will still apply. In the
following discussion, the dimension of a ReLU (sub)complex will refer to the maximum dimension
of its cells instead of the dimension of the ambient space in which it is embedded (e.g. a BH of a
2-dimensional ReLU complex is a 1-dimensional ReLU complex).


**Sign Sequence Complex:** The polyhedral complex can be described in terms of sign sequences, because in the interior of any _k_ -cell, the sign sequence of every point is exactly the same. Furthermore,
the sign sequences of a _k_ -cell contain exactly _d_ _−_ _k_ zero elements corresponding to the _d_ _−_ _k_ BHs
whose intersection contains the cell as a subset and _k_ nonzero elements corresponding to BHs that
either contain a single face of the cell or do not intersect the cell at all. For example, in Fig. 2, the
orange 1-cell with sign sequence (1 _,_ _−_ 1 _,_ _−_ 1 _,_ 0) is contained by BH 4 so it has 0 in this position, its
faces are the vertices on each end contained by BHs 2 and 3 respectively, and BH 1 does not intersect
the cell at all, so all three are nonzero in the sign sequence. With the aforementioned basic assumptions on the network weights, the work in (Masden, 2025) proves that every cell of a ReLU complex
has a unique sign sequence, that is, the mapping from cells to sign sequences _S_ : _C_ _→{−_ 1 _,_ 0 _,_ 1 _}_ _[n]_ is
well-defned and injective. The connectivity graph can be equivalently defned in terms of the sign
sequence complex, with nodes for the sign sequences of _d_ -cells (the ones with no zeros) and edges
between sequences that differ by one element.


3 THEORETICAL RESULTS ON NETWORK GEOMETRY


We examine how the cells in a ReLU complex are connected with each other, how many neighbors
a cell can have on average, and whether or not the number of neighbors varies with respect to the
depth and width of the network. Proof outlines are given here while detailed proofs are in Appendix
B. The cells in a ReLU complex and the number of their faces may depend on the specifc network
architecture and values of the network weights. However, we prove that the average number of
neighbors for any polyhedral region can be upper bounded by 2 _d_ regardless of the depth and width of
the network. In a ReLU complex, counting the faces of cells is the same as counting their neighbors.
For a _d_ -cell, each face is contained in a unique BH, and across each face is a unique neighboring
polyhedron that has the same sign sequence except that the sign corresponding to the crossed BH is
fipped. More generally, we prove the following theorem for any _k_ -cells of a ReLU complex.
**Theorem 3.1.** _For a ReLU complex in d-dimensional input space, the average number of faces of a_
_k-cell is at most_ 2 _k_ _for k_ = 1 _,_ 2 _, . . ., d._


An earlier work proves this theorem for hyperplane arrangements (Fukuda et al., 1991), which only
applies to the polyhedral complexes of single-layer networks, but the proof does not generalize to
deep ReLU network complexes formed by BH arrangements. We employ the 1-1 mapping between
cells and sign sequences to prove that the theorem also holds true for complexes of deep ReLU
networks.


Let _C_ be a ReLU complex corresponding to a network _f_ . Denote the BH of a neuron _i_ by _hi_, which
contains a set of _k_ -cells of _k_ _<_ _d_, specifcally _hi_ = _{c_ _∈C_ : _S_ ( _c_ ) _i_ = 0 _}_ . Then, we use _C_ _−_ _hi_ to
denote the complex that results from removing all cells contained in the BH of neuron _i_ and joining
all pairs of cells sharing one of the faces that were removed. Fig. 3a illustrates _C −_ _hi_ with _C_ from
Fig. 2 and BH 4 as _hi_ . The connectivity graph of _C −_ _hi_ can be obtained by contracting every edge


4


_**1**_


_**1**_


(a) _C −_ _h_ 4 (b) Cells Categorized with _i_ = 4 (c) Connectivity Graph


Figure 3: (a) The complex in Fig. 2 with BH 4 removed. (b) BH 4 is added and cells are categorized
according to Lemma 3.2 with those in Categories 1, 2, and 3 shown in blue, green, and red, respectively. (c) Connectivity graph with nodes and edges colored according to their corresponding cells.


corresponding to a ( _d_ _−_ 1)-cell contained in _hi_ (i.e., removing each edge and combining the two
end nodes into one, keeping their connections to other nodes in the graph). As a result, every two
_k_ -cells previously split by _hi_ are now fused into a single new _k_ -cell. Any cells that do not intersect
_hi_ remain the same in the new complex _C −_ _hi_ .
**Lemma 3.2.** _For any neuron i_ _in f_ _, each k-cell c_ _of C_ _falls into exactly one of the following cate-_
_gories:_

_Category 1: c_ _is a cell of hi_
_Category 2: c_ _is a cell of C −_ _hi_
_Category 3: c_ _is one of the two k-cells formed when a k-cell in C −_ _hi_ _is separated by hi_


We outline our full proof here. In the sign sequence of _c_, the element _S_ ( _c_ ) _i_ is either zero or nonzero.
If it is zero, then _c_ can only be in Category 1. If it is nonzero, we need to check if _c_ intersects _hi_ . This
is the case if changing the cell’s sign sequence so that _S_ ( _c_ ) _i_ = 0 yields a new sign sequence that
matches a Category 1 cell in _C_ . Then, fipping _S_ ( _c_ ) _i_ yields the sign sequence of a _k_ -cell neighbor
of _c_, so _c_ is in Category 3. Otherwise, if the new sign sequence is not an element of _C_, then _hi_ does
not contain _c_ or form a face of _c_, so _c_ is in Category 2.


Fig. 3b and Fig. 3c show the categorization of _k_ -cells ( _k_ = 0 _,_ 1 _,_ 2) in the polyhedral complex from
Fig. 2 when _hi_ corresponds to BH 4. The blue line segments and vertices in Fig. 3b are the Category
1 1- and 0-cells respectively, and the 4th element of their sign sequences is 0. The green 2-, 1, and 0-cells are in _C_ _−_ _h_ 4 because they do not change when _hi_ is removed. For such a cell, if
we zero out the 4th element of their sign sequence (e.g., the 2-cell in the center with a sequence
( _−_ 1 _,_ _−_ 1 _,_ _−_ 1 _,_ _−_ 1)), the resulting sequence (e.g., ( _−_ 1 _,_ _−_ 1 _,_ _−_ 1 _,_ 0)) does not exist in _S_ ( _C_ ). The red
2- and 1-cells are in Category 3 because _h_ 4 contains one face of each of those cells. As an example,
the leftmost 2-cell has the sign sequence ( _−_ 1 _,_ _−_ 1 _,_ 1 _,_ 1), and after setting _S_ ( _c_ )4 = 0, ( _−_ 1 _,_ _−_ 1 _,_ 1 _,_ 0)
corresponds to the line segment of _hi_ forming its right boundary. Note that the proof of Lemma 3.2
also works for ReLU subcomplexes formed by fxing one of the sign sequence elements, and these
subcomplexes cannot include only half of a pair of Category 3 cells since their sign sequences can
only differ at the position of the removed BH _i_ . Alternate colorings for the same complex based on
different choices of _i_ and restrictions to different subcomplexes are included in Appendix C.


If _hi_ is a neuron from the last ReLU layer, the complex _C−hi_ corresponds to another ReLU network,
but directly removing a neuron from an early layer may not result in a complex corresponding to
any ReLU network. As a result, we can break down the problem of counting cells (including faces)
in the ReLU complex by iteratively removing neurons starting from the last layer and counting the
number of cells that disappear. Let _Nk_ ( _C_ ) be the total number of _k_ -cells in _C_ . Based on Lemma 3.2,
_Nk_ ( _C_ ) equals the sum of the numbers of _k_ -cells in each category. To evaluate _Nk_ ( _C_ ), we frst count
the _k_ -cells in _hi_ and _C_ _−_ _hi_ separately, and then double count those in _C_ _−_ _hi_ that are split by _hi_ .
To count the split cells, we can just count ( _k_ _−_ 1)-cells in _hi_, which each divide one of them. For
example, compare Fig. 3a and Fig. 3b. The six 1-cells in _h_ 4 split the six 2-cells in Fig. 3a to add six
more 2-cells in Fig. 3b. Similarly, the six 0-cells in _hi_ also split six 1-cells.
**Lemma 3.3.** _For k_ = 1 _,_ 2 _, . . ., d,_
_Nk_ ( _C_ ) = _Nk_ ( _hi_ ) + _Nk_ ( _C −_ _hi_ ) + _Nk−_ 1( _hi_ ) _._ (1)


5


The frst term in the sum accounts for the Category 1 _k_ -cells in _C_, the second term accounts for all
the Category 2 cells and half the Category 3 cells, and the third term accounts for the other half of
the Category 3 cells. There are no Category 1 _d_ -cells in _C_ because _hi_ by defnition only contains
cells up to dimension _d_ _−_ 1, so when _k_ = _d_, the frst term _Nk_ ( _hi_ ) is always 0. The two lemmas lead
to the following special case of Theorem 3.1 for _k_ = _d_ :
**Theorem 3.4.** _[Upper Bound] The average number of faces of a d-cell of C_ _in_ R _[d]_ _(i.e., the average_
_degree of the connectivity graph) is at most_ 2 _d._


Here we provide an outline of our proof (see Appendix B for detailed proof). Each ( _d_ _−_ 1)-cell
forms a face between two _d_ -cells in _C_ because the single 0 in its corresponding sign sequence can be
set to _−_ 1 or 1 to get the sign sequences of the two _d_ -cells. Thus, the sum of the numbers of faces of
all _d_ -cells is just twice the total count of ( _d_ _−_ 1)-cells in _C_, so the average number of faces of a _d_ -cell
is [2] _[N]_ _N_ _[d]_ _d_ _[−]_ ( [1] _C_ [(] ) _[C]_ [)] [. Using Lemma 3.3, we prove ] 2 _NNdd−_ (1 _C_ () _C_ ) _≤_ 2 _d_ by mathematical induction on the number
of BHs _n_ in the complex and _d_ . By assuming that the upper bound holds for ( _n_ _−_ 1 _, d_ _−_ 1) and
( _n_ _−_ 1 _, d_ ), we prove that the upper bound holds for any complex with ( _n,_ _d_ ). The proof of Theorem
3.1 then follows by applying the lemmas to groups _k_ -cells whose sign sequences have exactly _d_ _−_ _k_
zeros at the same positions, that is, restricting the complex to the intersections of _d_ _−_ _k_ BHs and
applying the lemmas to the resulting subcomplex.


It is more straightforward to establish the following lower bound on the degree of individual nodes,
which then bounds the overall average degree of the ReLU complex graph.
**Theorem 3.5.** _[Lower Bound] If a ReLU network has n_ 1 _neurons in the frst hidden layer, every_
_d-cell of C_ _has at least_ min( _n_ 1 _, d_ ) _neighbors, and thus the average degree of the connectivity graph_
_is at least_ min( _n_ 1 _, d_ ) _._


3.1 ASYMPTOTIC BEHAVIOR


To study how connectivity properties change as network size increases, we can create sequences of
networks by adding new ReLU neurons to the last layer or a new layer after it. We use _Cn_ to denote
the complex after _n_ neurons have been added. We characterize these sequences with the following
theorems, which show that the average number of faces grows monotonically and that the bound in
Theorem 3.1 is tight respectively.
**Theorem 3.6.** _The average number of faces of d-cells in Cn_ _increases monotonically in terms of n._
**Theorem 3.7.** _Let f_ _be a shallow network that has only one hidden layer with n_ _nodes. When n_
_goes to infnity, the average number of faces of its d-cells converges exactly to_ 2 _d. That is,_


2 _Nd−_ 1( _Cn_ )
lim = 2 _d._
_n→∞_ _Nd_ ( _Cn_ )


In our experiments in Section 5, we observe that the average number of faces also appears to approach 2 _d_ as the depth of the network increases.


3.2 BOUNDS ON CONNECTIVITY GRAPH DIAMETER


Let _ℓ_ be the total number of hidden layers (depth), _mj_ be the number of nodes in layer _j_, _j_ =
1 _, . . ., ℓ_, and _m_ = max _{m_ 1 _,_ _· · ·_ _, mℓ}_ (width). We fnd that,
**Theorem 3.8.** _The diameter D_ _of the connectivity graph is_ Ω - ln(ln( _Ndn_ () _C_ )) - _and O_ - _mℓ_ - _._


The lower bound (in Ω) agrees with the intuition that diameter increases with the number of regions
in the complex. Although it appears as though increasing the number of neurons in the network
might reduce diameter by increasing ln( _n_ ), actually ln( _Nd_ ( _C_ )) grows much faster with _n_ regardless
of architecture, so the ln( _n_ ) term is just attenuating the growth of this lower bound. The upper bound
(in _O_ ) may rarely be reached in practice, but it is interesting in that it does not have to depend on the
input dimension _d_, even though the number of the network’s regions increases exponentially with _d_ .
We also fnd that this is empirically true in Section 5, as when we fx network architecture and only
change the input dimension, the diameter of the resulting complexes grows almost identically.


6


4 ALGORITHM FOR CALCULATING POLYHEDRON BOUNDARIES


To defne the complex, it will be necessary to map sign sequences to their polyhedra defned by
intersections of half-spaces, i.e., systems of linear inequalities of the form Φ _[T]_ _i_ _[x]_ [+] _[β][i]_ _[≤]_ [0] [for ] _[i]_ _[∈]_ _[f]_ [. ]
We are only concerned with the sign sequences of _d_ -cells, which do not contain any zeros, so that
each neuron provides exactly one inequality to our system defning the polyhedron. Let _s_ be such a
sign sequence and the column vector _s_ [(] _[j]_ [)] be the portion of _s_ that contains only signs for the neurons
in layer _j_ . Let _W_ [(] _[j]_ [)] _∈_ R _[j]_ [out] _[×][j]_ [in ] and _b_ [(] _[j]_ [)] _∈_ R _[j]_ [out] denote the weights and biases in layer _j_ of the
network with input dimension _j_ in and output dimension _j_ out. We can defne our polyhedron by using
the following formulas to calculate the inequalities layer by layer.


Half-spaces for current layer Picking half-space signs Mask for inactive neurons in the previous layer


          - ��           -           - ��           _β_ [(] _[j]_ [)] = diag _s_ [(] _[j]_ [)] _W_ [(] _[j]_ [)] diag ReLU _s_ [(] _[j][−]_ [1)] _β_ [(] _[j][−]_ [1)] + _b_ [(] _[j]_ [)] (3)


At the initial stage, Φ [(0)] = _Id_, _b_ [(0)] = 0( _d×_ 1) [, and ] _[s]_ (0) = **1** _d×_ 1. The frst term on the right-hand side
ensures that the inequality of each half-space is always in the same direction, regardless of whether
the neuron is active or inactive. This allows us to concatenate Φ [(] _[j]_ [)] and _β_ [(] _[j]_ [)] from each layer to get
the full linear system, Φ _x_ + _β_ _≤_ 0.


4.1 ENUMERATING POLYHEDRA


To enumerate the maximal polyhedra and ob
**Algorithm 1** Construction of the Connectivity Graph

tain their connectivity graph _G_ = ( _V,_ _E_ ), we
will employ breadth-frst search (BFS). We de- 1: **Input:** Trained network _f_, sign sequence _s_
scribe our exact method in Algorithm 1. Start- 2: _Q,_ _V,_ _E_ _←{s},_ _{s},_ _∅_

3: **while** _Q_ is not empty **do**

ing with a valid sign sequence _s_, which can

4: _s_ _←_ pop( _Q_ )

be found by passing any point through the net
5: **for** _i_ _∈{_ 0 _, . . ., n}_ **do**

work, we enumerate its neighbors and add them
to our graph. Neighbors are polyhedra that can 6: 7: **if** Sadd (s, OLVELP _s_ - _i_ () to _−_ Φ _Esi_ _,_ Φ _s, βs_ + _ei_ ) _≥_ _βsi_ **then**
be reached by crossing a single BH, so their 8: **if** _s_ - _i_ _̸∈_ _V_ **then** add _s_ - _i_ to _Q_ and _V_ **end if**
sign sequences have the opposite sign from the 9: **end if**
original polyhedron in exactly one position _i_, 10: **end for**
denoted as _s_ - _i_ . To fnd the neighbors, we can 11: **end while**
calculate Φ _s_ and _βs_ as described in the previous 12: **return** ( _V, E_ )
section and determine which inequalities actually form the boundary of the polyhedron. We check the redundancy of each inequality by solving
an LP, performed by the SOLVELP subroutine on line 6, which is given the arguments Φ _s_ for the
constraint coeffcient matrix, _βs_ + _ei_ ( _βs_ with 1 added to the _i_ th element to relax this constraint) for
the constraint offset, and _−_ Φ _si_ for the objective function coeffcients to maximize in the direction of
the relaxed constraint. This LP is explained further in Appendix D. Non-redundant constraints will


**Algorithm 1** Construction of the Connectivity Graph


1: **Input:** Trained network _f_, sign sequence _s_
2: _Q,_ _V,_ _E_ _←{s},_ _{s},_ _∅_
3: **while** _Q_ is not empty **do**
4: _s_ _←_ pop( _Q_ )
5: **for** _i_ _∈{_ 0 _, . . ., n}_ **do**
6: **if** SOLVELP( _−_ Φ _si_ _,_ Φ _s, βs_ + _ei_ ) _≥_ _βsi_ **then**
7: add (s, _s_  - _i_ ) to _E_
8: **if** _s_  - _i_ _̸∈_ _V_ **then** add _s_  - _i_ to _Q_ and _V_ **end if**
9: **end if**
10: **end for**
11: **end while**
12: **return** ( _V, E_ )


_i_

the relaxed constraint. This LP is explained further in Appendix D. Non-redundant constraints will

be violated by the optimal solution to this LP, i.e., Φ _[T]_ _si_ _[x]_ _[>]_ _[β][s]_ _i_ [, meaning that ] _[s]_ [-] _[i]_ [gives the sign se-]
quence of the neighbor of _s_ across the BH of neuron _i_ . For each neighbor, we add the edge between
_s_ and _s_ - _i_ to the graph (line 7), and if we have not reached _s_ - _i_ before, we add it to both the graph and
the search queue (line 8). In the next iteration, we can pop a new sign sequence from the queue and
repeat the same process.


This traversal of polyhedra is similar to several previous works (Xu et al., 2022; Liu et al., 2023a;b),
specifcally the BFS in (Xu et al., 2022), but we take the additional step of building up the connectivity graph of the polyhedral complex over the course of the search by recording when faces are
shared with already-found polyhedra. Furthermore, when determining whether or not an element
of a sign sequence can be fipped to produce another valid sign sequence, we follow (Zhang & Wu,
2019; Fukuda, 2004) and slightly relax the corresponding inequality to reduce errors arising from
insuffcient numerical precision.


7


1 2 3 4


# Hidden Layers 1 2 3 4


Width: 4 Width: 8 Width: 16


2

4

8

5


1M


10k


100


1

1M


10k


100


1


4


9


8


7


6


5


4


7


6


3

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
|<br><br>|||||||
|**1**<br>~~-~~<br>|**11**<br>~~--i-~~<br>~~. ~~|**11**<br>~~i--,~~|**1**<br>~~---~~|**1**<br>~~--~~|~~----·~~||

10 20 30 40 50 60


Number of ReLU Neurons


3


4


5


120


60


40


80


60


20


100


80


_d_
2


60


40


|•|Col2|i|l|Col5|l|l|l|llllh<br>1|Col10|Col11|1|1|Col14|Col15|1|Col17|1|Col19|1|Col21|1|Col23|1|111111|Col26|Col27|Col28|Col29|Col30|Col31|Col32|Col33|Col34|Col35|Col36|Col37|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|~~**ab**~~<br>10<br>  Left) D<br> ta. Ea<br> es of n<br>ns of ne<br>  rs. (Ri<br>  imens<br>RIME<br>  ndersta<br>  r of ne<br>   distrib<br>l setup<br>  ments<br>RSTAN<br>   trainin<br> ith un<br>  ith sid<br>  h hidd<br>  genera<br>  and o<br>   Table<br>  eighbo<br> odal an<br>  ound f|<br>5<br>   i<br>  c<br>   e<br>  t<br>   g<br>  io<br>N<br>  n<br>    u<br>   ut<br> <br>   c<br> <br>   g<br>  it<br>   e<br>   e<br>  t<br>   b<br>    1<br>  r<br>  d<br>   r|~~**1**~~<br><br>   st<br>  h<br>   t<br>  w<br>   h<br>  n<br>T<br>  d<br>    r<br>   e<br>  a<br>   a<br>D<br> <br> <br>    l<br>   n<br>  e<br>   ta<br>   ,<br>  s<br> <br>   o|<br>   r<br> <br> <br> <br>   t<br>  .<br> <br> <br>    a<br>   d<br>  n<br>   n<br>I<br>    a<br>   v<br> <br>    l<br>  d<br>   i<br>     a<br>   i<br>   s<br>|~~**1**~~<br>1<br> <br>   ib<br>   c<br>   o<br>  or<br>   )<br> <br>S<br>   th<br>    l<br> <br>  d<br>    b<br>N<br> <br>   ar<br>    en<br>    a<br>   d<br>   n<br>     n<br>   n<br>   k<br>|~~**1**~~<br>0<br>N<br>   u<br>   o<br>   rk<br>  k<br>    T<br>   D<br> <br>   e<br>     ne<br>    ac<br>   tr<br>    e<br>G<br>     nu<br>   ia<br>    g<br>    y<br>   a<br>    i<br>     d<br>    e<br>   e<br>    S|~~**1**~~<br>1<br>um<br>   ti<br>   lo<br>   s<br>   w<br>    he<br>   ot<br>    st<br>     t<br>    ro<br>   ai<br>     fo<br> T<br>     m<br>   n<br>    th<br>    er.<br>   ta<br>    nf<br>      th<br>    ve<br>   we<br>    ec|~~**1**~~<br>5<br>b<br>   on<br>   re<br>    w<br>   ei<br>     m<br>   te<br>    ru<br>     wo<br>    ss<br>   ne<br>     u<br> HE<br>     b<br>   ce<br>     1<br>     F<br>   se<br>    or<br>      e<br>    ry<br>   d<br>    ti|<br>e<br>   s<br>   d<br>    i<br> <br> <br>   d<br>    c<br>     r<br> <br> <br>     n<br> <br>     e<br> <br>     0<br>     o<br>   t<br> <br>       d<br> <br>    r<br>    o|<br>e<br>   s<br>   d<br>    i<br> <br> <br>   d<br>    c<br>     r<br> <br> <br>     n<br> <br>     e<br> <br>     0<br>     o<br>   t<br> <br>       d<br> <br>    r<br>    o|~~**1**~~<br>  h<br> <br>     o<br>      e<br>     5<br>       a<br>     p<br>      R<br>       c<br> <br>    r<br>t<br> <br>       w<br>     t<br>       a<br> <br>     e<br>     b<br>       i<br>     x<br>     e<br>|<br>  b<br>      n<br> <br>      c<br> <br>       c<br>     r<br> <br>       o<br>      p<br>    k<br><br>S<br> <br>     e<br>       r<br>       o<br>     r<br> <br>       o<br> <br> <br>|<br>  o<br>      u<br> <br>      i<br>      d<br> <br> <br>      e<br>       n<br>      l<br>    s<br>p<br> <br>       o<br>     r<br>       y<br> <br>     f<br>     o<br>       n<br>      i<br> <br>       F|<br><br>  r<br> <br>     s<br>      f<br>      i<br>       h<br>     e<br>      L<br> <br>      e<br> <br>s<br>       r<br>     s<br> <br> <br> <br>     u<br> <br>      s<br>      a<br>       i|<br>1<br>  s<br> <br> <br> <br>      f<br> <br>     s<br> <br>       s<br> <br>     c<br><br>       k<br> <br>        i<br>       b<br>     o<br>     t<br>       s<br> <br>      l<br>       g|<br>0<br> <br> <br>      t<br>      c<br>      f<br>        d<br>     e<br> <br>       t<br>      x<br> <br>:<br> <br>      s<br> <br> <br>     r<br> <br> <br>       b<br>      s<br>       .|<br><br>      b<br>      h<br> <br> <br>        i<br>     n<br> <br>       r<br> <br>     a<br><br>       s<br>      e<br>        n<br>       i<br> <br> <br>        o<br>       e<br>      o<br>|<br> <br> <br> <br>      e<br>        s<br> <br> <br>       u<br>       o<br>     n<br>/<br> <br>      l<br>        p<br>       n<br> <br>      e<br>        f<br>       l<br> <br>|<br>      e<br>      e<br> <br>      r<br>        t<br>     t<br>       c<br>       c<br>       f<br> <br><br>        o<br>      e<br> <br>       a<br> <br>      v<br> <br>       o<br> <br>        5|<br>      r<br> <br>       i<br>      e<br>        r<br>      u<br>       o<br>       t<br> <br>      b<br>/<br>        n<br>      c<br>        ut<br>       t<br>      a<br>      e<br>         n<br> <br>       pl<br>        .|~~**1**~~<br>       o<br>       n<br>       d<br>      n<br>        ib<br>      p<br>       m<br>        t<br>        n<br>      e<br>g<br> <br>      t<br> <br>       io<br>      n<br>      r<br>         e<br>       w<br>       o<br>|<br> <br>       u<br>       t<br>      t<br> <br> <br> <br>        h<br>        e<br> <br><br>         c<br>      e<br>         d<br> <br> <br>      y<br>         i<br> <br>       t<br>|<br><br>       f<br> <br>       h<br>       t<br>        u<br>      p<br>       p<br> <br>        t<br>       f<br>i<br>         l<br>      d<br>         i<br>       n<br>       e<br> <br>         g<br>        t<br> <br>|<br>2<br> <br> <br>      ,<br>       r<br>        t<br>      e<br>       l<br>        e<br> <br>       o<br>t<br>         u<br> <br> <br> <br> <br>       r<br>         h<br>        h<br>        t<br>|<br>0<br>        f<br> <br> <br>       a<br>        i<br>      r<br>       e<br>        i<br> <br> <br><br> <br> <br> <br>        o<br>       x<br>       e<br> <br>        e<br>        h<br>         e|<br><br>        a<br>       b<br>        d<br>       i<br>        o<br> <br> <br>        r<br>        o<br>       u<br>h<br>         s<br>       u<br> <br>        f<br>       h<br> <br>         b<br> <br> <br>|<br>        c<br>       e<br> <br> <br> <br>       b<br>       x<br> <br> <br>       n<br><br>         t<br>       n<br>         e<br> <br> <br>       g<br>         o<br>         u<br>        e<br>          e|<br>        e<br>       r<br>        e<br>       n<br>        n<br> <br>       e<br>         c<br>        r<br> <br>u<br>         e<br> <br>         n<br>         h<br>       a<br>       i<br> <br> <br> <br>          s|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|~~** 1111**~~<br>~~,~~<br> <br>30<br>        s of po<br>        of pol<br>        pth, and<br>       ing data<br>         versus<br>       ounds fo<br>       s, we us<br>         onnecti<br>        ks after<br>       d in App<br>b.com<br>         ring pro<br>       iformly<br>         sion_d_,<br>         yperpar<br>       ustive s<br>       on. Sum<br>         r counts<br>         pper bo<br>         estimate<br>          timate|
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|~~**ab**~~<br>10<br>  Left) D<br> ta. Ea<br> es of n<br>ns of ne<br>  rs. (Ri<br>  imens<br>RIME<br>  ndersta<br>  r of ne<br>   distrib<br>l setup<br>  ments<br>RSTAN<br>   trainin<br> ith un<br>  ith sid<br>  h hidd<br>  genera<br>  and o<br>   Table<br>  eighbo<br> odal an<br>  ound f|<br>5<br>   i<br>  c<br>   e<br>  t<br>   g<br>  io<br>N<br>  n<br>    u<br>   ut<br> <br>   c<br> <br>   g<br>  it<br>   e<br>   e<br>  t<br>   b<br>    1<br>  r<br>  d<br>   r|~~**1**~~<br><br>   st<br>  h<br>   t<br>  w<br>   h<br>  n<br>T<br>  d<br>    r<br>   e<br>  a<br>   a<br>D<br> <br> <br>    l<br>   n<br>  e<br>   ta<br>   ,<br>  s<br> <br>   o|<br>   r<br> <br> <br> <br>   t<br>  .<br> <br> <br>    a<br>   d<br>  n<br>   n<br>I<br>    a<br>   v<br> <br>    l<br>  d<br>   i<br>     a<br>   i<br>   s<br>|~~**1**~~<br>1<br> <br>   ib<br>   c<br>   o<br>  or<br>   )<br> <br>S<br>   th<br>    l<br> <br>  d<br>    b<br>N<br> <br>   ar<br>    en<br>    a<br>   d<br>   n<br>     n<br>   n<br>   k<br>|~~**1**~~<br>0<br>N<br>   u<br>   o<br>   rk<br>  k<br>    T<br>   D<br> <br>   e<br>     ne<br>    ac<br>   tr<br>    e<br>G<br>     nu<br>   ia<br>    g<br>    y<br>   a<br>    i<br>     d<br>    e<br>   e<br>    S|~~**1**~~<br>1<br>um<br>   ti<br>   lo<br>   s<br>   w<br>    he<br>   ot<br>    st<br>     t<br>    ro<br>   ai<br>     fo<br> T<br>     m<br>   n<br>    th<br>    er.<br>   ta<br>    nf<br>      th<br>    ve<br>   we<br>    ec|~~**1**~~<br>5<br>b<br>   on<br>   re<br>    w<br>   ei<br>     m<br>   te<br>    ru<br>     wo<br>    ss<br>   ne<br>     u<br> HE<br>     b<br>   ce<br>     1<br>     F<br>   se<br>    or<br>      e<br>    ry<br>   d<br>    ti|<br>e<br>   s<br>   d<br>    i<br> <br> <br>   d<br>    c<br>     r<br> <br> <br>     n<br> <br>     e<br> <br>     0<br>     o<br>   t<br> <br>       d<br> <br>    r<br>    o|<br>e<br>   s<br>   d<br>    i<br> <br> <br>   d<br>    c<br>     r<br> <br> <br>     n<br> <br>     e<br> <br>     0<br>     o<br>   t<br> <br>       d<br> <br>    r<br>    o|||||||||||||||||||||||||~~, ~~||
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25||||||||||, <br>|||||||||||||||||||||||<br>|~~,.~~<br>|~~•~~<br>|<br>|
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25||||||||||<br>~~.~~<br>|||||||||||||||||||||~~**_,_**~~|<br>|<br> <br>|<br> <br>|<br> <br>|<br> <br>|
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25||||||||||<br>~~.~~<br>|||||||||||||||||||||||||||
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|~~~ ~~|~~~ ~~|||||||||||||||||||||||||||||||||||
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|~~~ ~~|~~~ ~~|||||||||||||||||||||||||||||||||||
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|<br>|||||||||||||||||||||||||||
|~~**l**~~<br>5<br> 4: (<br>ic da<br> plex<br>zatio<br> ck ba<br> by d<br>XPE<br> her u<br>  umbe<br> o be<br>menta<br>  xperi<br> UNDE<br> rt by<br>ans w<br> gin w<br> f eac<br> ewly<br> work<br> nd in<br>r of n<br> unim<br> per b<br>5<br>10<br>15<br>20<br>25|||||||||||||||||||||||||||||||||||||


|Col1|Col2|Col3|,<br>,,<br>,,,|
|---|---|---|---|
|||<br>**_p _**|<br> <br>**_1/'_**<br>....|
||||<br>|
|||~~~;;;~~|~~.~~<br>~~-~~|
|~~..~~||||
|||||


100 1000 10k


10 100


10 100 1000


Width [Depth]


Figure 5: Connectivity graph diameter vs theoretical upper bound. At each distinct value of the
theoretical upper bound (abscissa), the actual diameter of 5 network complexes was estimated as
described in Section 5.1. Each pair of dotted lines also encloses all (non-estimated) values of the
connectivity graph diameters from every experiment. Each subfgure shows networks with fxed
width _m_ and depth 1 _≤_ _ℓ_ _≤_ 4. The widths are 4 (left), 8 (middle), and 16 (right).


8


Table 1: Summary statistics for the distributions in Fig. 4 with four dimensions (left) and fve dimensions (right). Diameter for each complex is estimated as described in Section 5.1. Non-degenerate
depth-1 networks always have the same number of polyhedra because their BHs are all just hyperplanes (Buck, 1943).


# Polyhedrons Average Degree Diameter # Polyhedrons Average Degree Diameter


1 4 16.00 _±_ 0.00 4.00 _±_ 0.00 5.50 _±_ 0.00 16.00 _±_ 0.00 4.00 _±_ 0.00 5.50 _±_ 0.00
8 163.00 _±_ 0.00 6.28 _±_ 0.00 10.60 _±_ 0.42 219.00 _±_ 0.00 7.23 _±_ 0.00 10.75 _±_ 0.26
16 2517.00 _±_ 0.00 7.32 _±_ 0.00 22.50 _±_ 0.35 6884.87 _±_ 0.35 9.02 _±_ 0.00 23.17 _±_ 0.41

2 4 72.60 _±_ 22.70 5.21 _±_ 0.31 8.70 _±_ 0.76 89.50 _±_ 19.78 5.34 _±_ 0.25 9.30 _±_ 0.63
8 2244.80 _±_ 630.08 7.25 _±_ 0.18 20.40 _±_ 0.74 5802.60 _±_ 1146.10 8.77 _±_ 0.27 21.75 _±_ 1.06
16 42243.00 _±_ 8608.12 7.72 _±_ 0.06 41.10 _±_ 0.65 2.69 _×_ 10 [5] _±_ 48746.09 9.61 _±_ 0.05 42.57 _±_ 1.53
3 4 227.60 _±_ 42.21 5.85 _±_ 0.16 12.50 _±_ 0.61 389.60 _±_ 188.02 5.65 _±_ 0.41 14.65 _±_ 2.32
8 9340.80 _±_ 3325.81 7.47 _±_ 0.17 27.60 _±_ 2.25 36591.20 _±_ 12085.54 8.54 _±_ 0.93 31.50 _±_ 2.33
16 2.23 _×_ 10 [5] _±_ 72142.81 7.82 _±_ 0.04 57.70 _±_ 2.46 1.78 _×_ 10 [6] _±_ 1.94 _×_ 10 [5] 9.78 _±_ 0.04 57.44 _±_ 1.47
4 4 448.00 _±_ 119.14 6.17 _±_ 0.12 15.90 _±_ 1.19 1206.70 _±_ 1154.00 5.50 _±_ 0.78 18.60 _±_ 3.98
8 35767.20 _±_ 9493.85 7.70 _±_ 0.06 37.40 _±_ 1.29 1.82 _×_ 10 [5] _±_ 79768.45 8.25 _±_ 1.38 48.35 _±_ 12.25
16 6.24 _×_ 10 [5] _±_ 96311.16 7.85 _±_ 0.03 76.35 _±_ 4.56 5.03 _×_ 10 [6] _±_ 1.07 _×_ 10 [6] 9.80 _±_ 0.03 70.88 _±_ 1.19


4.00 _±_ 0.00
6.28 _±_ 0.00
7.32 _±_ 0.00


5.50 _±_ 0.00
10.60 _±_ 0.42
22.50 _±_ 0.35


16.00 _±_ 0.00
219.00 _±_ 0.00
6884.87 _±_ 0.35


4.00 _±_ 0.00
7.23 _±_ 0.00
9.02 _±_ 0.00


1 4
8
16


16.00 _±_ 0.00
163.00 _±_ 0.00
2517.00 _±_ 0.00


by bounding each one above and below using the corresponding algorithms from (Magnien et al.,
2009) and taking the midpoint. To be clear, the asymptotic bounds derived in Section 3.2 were not
used to make this estimate. Across all experiments, the diameter estimates for networks with the
same depth and width were almost identical across different input dimensions. Although the upper
bound is rarely reached, the logic that it should be independent of input dimension appears to hold
in practice. Furthermore, when width is fxed, the diameter appears to grow logarithmically with
respect to our theoretical upper bound. Additional summary metrics for the complexes and results
for _d_ _∈{_ 2 _,_ 3 _}_ can be found in Appendix G.


5.2 TRAINING DATA AND POLYHEDRA


We observe a difference in the distributions of neighbor counts between polyhedra that contain data
and those that do not. We test networks trained on three datasets: California Housing (CC 0 License) (Kelley Pace & Barry, 1997), MNIST (CC BY-SA 3.0 License) (Deng, 2012), and CIFAR10
(MIT License) (Krizhevsky, 2009), and achieve reasonable performance for each (AUC above 0.9
or _R_ [2] above 0.6). We examine the last 3 layers of 8 neurons for MNIST and 2 layers of 64 neurons
for CIFAR10 on a lower-dimensional hidden representation rather than the input, 5 dimensions for
MNIST and 10 for CIFAR10. For California Housing, we calculate the complex of the entire network. Details about the datasets and networks can be found in Appendix F. Algorithm 1 was used to
identify all polyhedra in the complex for MNIST. For the California Housing and CIFAR10 datasets,
complete enumeration of the network complex was intractable, so the search was terminated after
traversing 8 million polyhedra. We then randomly sample 10,000 points from the training data. If a
data point does not lie in one of the polyhedra found in the initial search, we calculate the new polyhedron that contains the point and add it to the 8 million that were already found. The distributions
of neighbor counts for these complexes can be found in Fig. 6. Across all datasets, the neighbor
counts for polyhedra containing training data tend to be higher than the upper bound for the average
neighbor count of all polyhedra. Since the number of faces of any polyhedron is bounded above by
_n_, this necessarily reduces the rightward skew of the distribution as well.


1M


100k


10k


1000


100


10


10k


1000


100


10


1M


100k


10k


1000


100


10


1
10 15 20 25 30 35 40 45 50 55


Number
of Points


625


125


25


5


1


6 8 10 12 14 16 18 20


1
10 20 30 40 50 60 70


1


(a) MNIST ( _d_ =5, _n_ =24) (b) CIFAR10 ( _d_ =10, _n_ =128) (c) CA Housing ( _d_ =8, _n_ =128)


Figure 6: Histograms of polyhedron neighbor counts (i.e., the number of polyhedra that have a
specifc number of neighbors) for polyhedra that do not contain training data (gray) and ones that do
(colored by total number of data points contained in those polyhedra).


9


We also examine how neighbor counts vary according to whether polyhedra are bounded or unbounded, with results shown in Fig. 7. In all three experiments, we observe that polyhedra with
higher numbers of neighbors are more likely to be unbounded (darker colors toward the right of
each histogram, with the exception of polyhedra with _d_ neighbors shown by the leftmost bars, which
are always unbounded). In addition, we fnd that the proportion of unbounded polyhedra in datacontaining regions is higher than the overall proportion for both classifcation tasks (the top two
histograms show darker bars than the corresponding bottom fgures) but lower for the regression
task (the top histogram bars have lighter colors than the bottom). For the classifcation tasks, the
network may have to focus its complexity on the spaces between classes of data points where it has
to draw the decision boundary, leaving more of the data points themselves on the outer (unbounded)
regions of the complex. On the other hand, for regression, the model is focused on ftting the data
points, so data points tend to lie more on bounded regions with fnite function values. Additional
results from these experiments are included in Appendix G.


100k


1000


**10k**


**1000**


**10**


**10k**


**1000**


**10**


10 10

0%


100k


1000


Percent
Bounded


100%

10 10

80%


100%


10


10


80%


100k


1000


100k


1000


60%


40%


20%


10


10


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||
|||||||||||||||||


0%


**6** **8** **10** **12** **14** **16** **18** **20**


10 20 30 40 50 60 70


10 20 30 40 50


Number of Neighbors

(a) MNIST (b) CIFAR10 (c) CH Reg


Figure 7: Histograms of polyhedron neighbor counts, separated into those that contain data points
(top) and those that do not (bottom). The vertical axis gives the number of polyhedra and each bar is
colored by the percentage of polyhedra among those with the same neighbor count that are bounded.


6 DISCUSSION AND FUTURE WORK


This work characterizes general geometric properties of the polyhedral complexes defned by ReLU
networks. For the frst time, we place bounds on both the average connectivity of this complex and
its graph-theoretic diameter. We also conduct empirical studies that visualize the distributions of
polyhedron connectivity, and show that training data tends to lie on polyhedra with higher-thanaverage connectivity.


There are several limitations to the work presented here. Further investigation is needed to fully
explain why training tends to put data points in regions with higher numbers of faces, and how this
phenomenon is related to the network’s behavior. Additionally, we are not yet able to describe how
more specifc network structures like convolutional layers and skip connections affect the network’s
geometry. Another limitation comes from the fact that our results only apply to ReLU activations,
and while they could be extended to other piecewise-linear activation functions, there are no immediate implications for networks that use nonlinear activation functions.


Our results have implications for several active areas of study that involve the polyhedral geometry
of ReLU networks. For example, the work by Ji et al. (2022) places a bound on empirical training
error based on the spatial relationships between the regions containing the train and test data. They
use Hamming distance between the sign sequences of two polyhedra as a distance metric between
them. However, this metric will not refect the case where a bent hyperplane may have to be crossed
multiple times when moving from one polyhedron to another. Thus, the length of the shortest
path between two polyhedra in the connectivity graph is a more suitable metric. If path length is
used, Theorem 3.8 allows us to bound the empirical error based on the network architecture and
independently of the input dimension.


10


REFERENCES


Ross Anderson, Joey Huchette, Christian Tjandraatmadja, and Juan Pablo Vielma. Strong MixedInteger Programming Formulations for Trained Neural Networks. In Andrea Lodi and Viswanath
Nagarajan (eds.), _Integer Programming and Combinatorial Optimization_, pp. 27–42, Cham, 2019.
Springer International Publishing. ISBN 978-3-030-17953-3.


Navid Ansari, Hans-Peter Seidel, and Vahid Babaei. Mixed integer neural inverse design. _ACM_
_Transactions on Graphics_, 41(4):151:1–151:14, July 2022. ISSN 0730-0301. doi: 10.1145/35
[28223.3530083. URL https://dl.acm.org/doi/10.1145/3528223.3530083.](https://dl.acm.org/doi/10.1145/3528223.3530083)
GSCC: 0000268 4 citations (Semantic Scholar/DOI) [2024-05-30] 0 citations (Crossref) [202405-30].


Randall Balestriero and Richard Baraniuk. Mad Max: Affne Spline Insights Into Deep Learning.
_Proceedings of the IEEE_ [, 109:704–727, 2018. URL https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:49901088)
[org/CorpusID:49901088.](https://api.semanticscholar.org/CorpusID:49901088)


Randall Balestriero and richard baraniuk. A Spline Theory of Deep Learning. In Jennifer Dy and
Andreas Krause (eds.), _Proceedings of the 35th International Conference on Machine Learning_,
volume 80 of _Proceedings of Machine Learning Research_, pp. 374–383. PMLR, July 2018. URL
[https://proceedings.mlr.press/v80/balestriero18b.html.](https://proceedings.mlr.press/v80/balestriero18b.html)


Randall Balestriero, Romain Cosentino, Behnaam Aazhang, and Richard Baraniuk. The Geometry
of Deep Networks: Power Diagram Subdivision. In _Neural Information Processing Systems_,
[2019. URL https://api.semanticscholar.org/CorpusID:160010022.](https://api.semanticscholar.org/CorpusID:160010022)


Randall Balestriero, Romain Cosentino, and Sarath Shekkizhar. Characterizing Large Language
Model Geometry Helps Solve Toxicity Detection and Generation. In _Forty-frst International_
_Conference on Machine Learning_ [, 2024. URL https://openreview.net/forum?id=](https://openreview.net/forum?id=glfcwSsks8)
[glfcwSsks8.](https://openreview.net/forum?id=glfcwSsks8)


Arturs Berzins. Polyhedral Complex Extraction from ReLU Networks using Edge Subdivision. In
_Proceedings of the 40th International Conference on Machine Learning_, pp. 2234–2244. PMLR,
[July 2023. URL https://proceedings.mlr.press/v202/berzins23a.html.](https://proceedings.mlr.press/v202/berzins23a.html)


Elena Botoeva, Panagiotis Kouvaros, Jan Kronqvist, Alessio Lomuscio, and Ruth Misener. Effcient
Verifcation of ReLU-Based Neural Networks via Dependency Analysis. _Proceedings of the AAAI_
_Conference on Artifcial Intelligence_, 34(04):3291–3299, April 2020. ISSN 2374-3468, 2159[5399. doi: 10.1609/aaai.v34i04.5729. URL https://ojs.aaai.org/index.php/AAA](https://ojs.aaai.org/index.php/AAAI/article/view/5729)
[I/article/view/5729. GSCC: 0000170 102 citations (Semantic Scholar/DOI) [2024-05-](https://ojs.aaai.org/index.php/AAAI/article/view/5729)
30] 60 citations (Crossref) [2024-05-30].


R. C. Buck. Partition of Space. _The American Mathematical Monthly_, 50(9):541–544, November
[1943. ISSN 0002-9890, 1930-0972. doi: 10.1080/00029890.1943.11991447. URL https:](https://www.tandfonline.com/doi/full/10.1080/00029890.1943.11991447)
[//www.tandfonline.com/doi/full/10.1080/00029890.1943.11991447.](https://www.tandfonline.com/doi/full/10.1080/00029890.1943.11991447)


Rudy R Bunel, Ilker Turkaslan, Philip Torr, Pushmeet Kohli, and Pawan K Mudigonda. A Unifed
View of Piecewise Linear Neural Network Verifcation. In S. Bengio, H. Wallach, H. Larochelle,
K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), _Advances in Neural Information Processing_
_Systems_ [, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurip](https://proceedings.neurips.cc/paper_files/paper/2018/file/be53d253d6bc3258a8160556dda3e9b2-Paper.pdf)
[s.cc/paper_files/paper/2018/file/be53d253d6bc3258a8160556dda3e9b](https://proceedings.neurips.cc/paper_files/paper/2018/file/be53d253d6bc3258a8160556dda3e9b2-Paper.pdf)
[2-Paper.pdf. GSCC: 0000413.](https://proceedings.neurips.cc/paper_files/paper/2018/file/be53d253d6bc3258a8160556dda3e9b2-Paper.pdf)


Vasileios Charisopoulos and Petros Maragos. A Tropical Approach to Neural Networks with Piece
[wise Linear Activations, January 2019. URL http://arxiv.org/abs/1805.08749.](http://arxiv.org/abs/1805.08749)
GSCC: 0000048 arXiv:1805.08749 [cs, stat].


Chih-Hong Cheng, Georg N¨uhrenberg, and Harald Ruess. Maximum Resilience of Artifcial Neural
Networks. In Deepak D’Souza and K. Narayan Kumar (eds.), _Automated Technology for Verif-_
_cation and Analysis_, pp. 251–268, Cham, 2017. Springer International Publishing. ISBN 978-3319-68167-2. doi: 10.1007/978-3-319-68167-2 18. GSCC: 0000334 92 citations (Crossref)

[2024-05-30] 255 citations (Semantic Scholar/DOI) [2024-05-30].


11


Francesco Craighero, Fabrizio Angaroni, Alex Graudenzi, Fabio Stella, and Marco Antoniotti. In
vestigating the Compositional Structure of Deep Neural Networks. In Giuseppe Nicosia, Varun
Ojha, Emanuele La Malfa, Giorgio Jansen, Vincenzo Sciacca, Panos Pardalos, Giovanni Giuffrida, and Renato Umeton (eds.), _Machine Learning, Optimization, and Data Science_, pp. 322–
334, Cham, 2020. Springer International Publishing. ISBN 978-3-030-64583-0.


B´ alint Dar´ oczy. Tangent Space Sensitivity and Distribution of Linear Regions in ReLU Networks, June 2020. URL [http://arxiv.org/abs/2006.06780. GSCC: 0000000](http://arxiv.org/abs/2006.06780)
arXiv:2006.06780 [cs, stat].


Li Deng. The MNIST Database of Handwritten Digit Images for Machine Learning Research [Best
of the Web]. _IEEE Signal Processing Magazine_, 29(6):141–142, November 2012. ISSN 1558[0792. doi: 10.1109/MSP.2012.2211477. URL https://ieeexplore.ieee.org/docu](https://ieeexplore.ieee.org/document/6296535)
[ment/6296535.](https://ieeexplore.ieee.org/document/6296535)


Sahil Rajesh Dhayalkar. The Geometry of ReLU Networks through the ReLU Transition Graph,
[May 2025. URL http://arxiv.org/abs/2505.11692. arXiv:2505.11692.](http://arxiv.org/abs/2505.11692)


Reinhard Diestel. _Graph theory_ . Springer Berlin Heidelberg, New York, NY, 6 edition, 2017. ISBN
9783662536216.


Feng-Lei Fan, Wei Huang, Xiangru Zhong, Lecheng Ruan, Tieyong Zeng, Huan Xiong, and Fei
[Wang. Deep ReLU Networks Have Surprisingly Simple Polytopes, November 2024. URL http:](http://arxiv.org/abs/2305.09145)
[//arxiv.org/abs/2305.09145. arXiv:2305.09145.](http://arxiv.org/abs/2305.09145)


Gregoire Fournier. A tropical approach to neural networks. Master’s thesis, University of Lille,
[September 2019. URL https://www.semanticscholar.org/paper/A-tropica](https://www.semanticscholar.org/paper/A-tropical-approach-to-neural-networks-Fournier/907a9ecd78de8d1ca57d2dc524135afdff6d7c66)
[l-approach-to-neural-networks-Fournier/907a9ecd78de8d1ca57d2dc5](https://www.semanticscholar.org/paper/A-tropical-approach-to-neural-networks-Fournier/907a9ecd78de8d1ca57d2dc524135afdff6d7c66)
[24135afdff6d7c66. GSCC: 0000164.](https://www.semanticscholar.org/paper/A-tropical-approach-to-neural-networks-Fournier/907a9ecd78de8d1ca57d2dc524135afdff6d7c66)


[Komei Fukuda. Frequently Asked Questions in Polyhedral Computation, August 2004. URL http](https://people.inf.ethz.ch/~fukudak/polyfaq/)

[s://people.inf.ethz.ch/˜fukudak/polyfaq/.](https://people.inf.ethz.ch/~fukudak/polyfaq/)


Komei Fukuda, Shigemasa Saito, Akihisa Tamura, and Takeshi Tokuyama. Bounding the number
of k-faces in arrangements of hyperplanes. _Discrete Applied Mathematics_, 31(2):151–165, April
[1991. ISSN 0166218X. doi: 10.1016/0166-218X(91)90067-7. URL https://linkinghub](https://linkinghub.elsevier.com/retrieve/pii/0166218X91900677)
[.elsevier.com/retrieve/pii/0166218X91900677.](https://linkinghub.elsevier.com/retrieve/pii/0166218X91900677)


Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear
neural networks. _Journal of Computational and Applied Mathematics_, 441:115667, May 2024.
[ISSN 0377-0427. doi: 10.1016/j.cam.2023.115667. URL https://www.sciencedirec](https://www.sciencedirect.com/science/article/pii/S0377042723006118)
[t.com/science/article/pii/S0377042723006118.](https://www.sciencedirect.com/science/article/pii/S0377042723006118)


J. Elisenda Grigsby and Kathryn Lindsey. On Transversality of Bent Hyperplane Arrangements and
the Topological Expressiveness of ReLU Neural Networks. _SIAM Journal on Applied Algebra_
_and Geometry_, 6(2):216–242, June 2022. ISSN 2470-6566. doi: 10.1137/20M1368902. URL
[https://epubs.siam.org/doi/10.1137/20M1368902. GSCC: 0000020.](https://epubs.siam.org/doi/10.1137/20M1368902)


J. Elisenda Grigsby, Kathryn Lindsey, and Marissa Masden. Local and global topological complexity
[measures OF ReLU neural network functions, April 2024. URL http://arxiv.org/abs/](http://arxiv.org/abs/2204.06062)
[2204.06062. GSCC: 0000006 arXiv:2204.06062 [cs, math].](http://arxiv.org/abs/2204.06062)


Boris Hanin and David Rolnick. Complexity of Linear Regions in Deep Networks. In Kamalika
Chaudhuri and Ruslan Salakhutdinov (eds.), _Proceedings of the 36th International Conference on_
_Machine Learning_, volume 97 of _Proceedings of Machine Learning Research_, pp. 2596–2604.
[PMLR, June 2019a. URL https://proceedings.mlr.press/v97/hanin19a.htm](https://proceedings.mlr.press/v97/hanin19a.html)
[l.](https://proceedings.mlr.press/v97/hanin19a.html)


Boris Hanin and David Rolnick. Deep ReLU networks have surprisingly few activation patterns.
In _Proceedings of the 33rd International Conference on Neural Information Processing Systems_ .
Curran Associates Inc., Red Hook, NY, USA, 2019b. GSCC: 0000237.


12


Fengxiang He, Shiye Lei, Jianmin Ji, and Dacheng Tao. Neural networks behave as hash encoders:
[An empirical study, January 2021. URL http://arxiv.org/abs/2101.05490. GSCC:](http://arxiv.org/abs/2101.05490)
0000004 arXiv:2101.05490 [cs, stat].


Xia Hu. _Understanding deep neural networks from the perspective of piecewise linear property_ .
PhD thesis, Simon Fraser University, 2021. Publisher: Simon Fraser University.


Changcun Huang. ReLU Networks Are Universal Approximators via Piecewise Linear or Constant
Functions. _Neural Computation_, 32(11):2249–2278, November 2020. ISSN 0899-7667, 1530888X. doi: 10.1162/neco a [01316. URL https://direct.mit.edu/neco/article/](https://direct.mit.edu/neco/article/32/11/2249-2278/95633)
[32/11/2249-2278/95633.](https://direct.mit.edu/neco/article/32/11/2249-2278/95633)


Joey Huchette and Juan Pablo Vielma. Nonconvex Piecewise Linear Functions: Advanced For
mulations and Simple Modeling Tools. _Operations Research_, 71(5):1835–1856, September
2023. ISSN 0030-364X, 1526-5463. doi: 10.1287/opre.2019.1973. URL [https:](https://pubsonline.informs.org/doi/10.1287/opre.2019.1973)
[//pubsonline.informs.org/doi/10.1287/opre.2019.1973.](https://pubsonline.informs.org/doi/10.1287/opre.2019.1973)


Joey Huchette, Gonzalo Mu˜noz, Thiago Serra, and Calvin Tsay. When Deep Learning Meets Poly
[hedral Theory: A Survey, April 2023. URL https://optimization-online.org/?p](https://optimization-online.org/?p=22840)
[=22840. GSCC: 0000025.](https://optimization-online.org/?p=22840)


Huma Jamil, Yajing Liu, Christina Cole, Nathaniel Blanchard, Emily J. King, Michael Kirby, and
Christopher Peterson. Dual Graphs of Polyhedral Decompositions for the Detection of Adversarial Attacks. In _2022 IEEE International Conference on Big Data (Big Data)_, pp. 2913–
2921, December 2022. doi: 10.1109/BigData55660.2022.10020880. URL [https:](https://ieeexplore.ieee.org/document/10020880)
[//ieeexplore.ieee.org/document/10020880.](https://ieeexplore.ieee.org/document/10020880)


Huma Jamil, Yajing Liu, Turgay Caglar, Christina M. Cole, Nathaniel Blanchard, Christopher Pe
terson, and Michael Kirby. Hamming Similarity and Graph Laplacians for Class Partitioning and
[Adversarial Image Detection, May 2023. URL http://arxiv.org/abs/2305.01808.](http://arxiv.org/abs/2305.01808)
GSCC: 0000006 arXiv:2305.01808 [cs, math].


Xu Ji, Razvan Pascanu, R. Devon Hjelm, Balaji Lakshminarayanan, and Andrea Vedaldi. Test
Sample Accuracy Scales with Training Sample Density in Neural Networks. In Sarath Chandar,
Razvan Pascanu, and Doina Precup (eds.), _Proceedings of The 1st Conference on Lifelong Learn-_
_ing Agents_, volume 199 of _Proceedings of Machine Learning Research_, pp. 629–646. PMLR,
[August 2022. URL https://proceedings.mlr.press/v199/ji22a.html. GSCC:](https://proceedings.mlr.press/v199/ji22a.html)
0000081.


R. Kelley Pace and Ronald Barry. Sparse spatial autoregressions. _Statistics & Probability Letters_,
33(3):291–297, May 1997. ISSN 0167-7152. doi: 10.1016/S0167-7152(96)00140-X. URL
[https://www.sciencedirect.com/science/article/pii/S0167715296001](https://www.sciencedirect.com/science/article/pii/S016771529600140X)
[40X.](https://www.sciencedirect.com/science/article/pii/S016771529600140X)


Alex Krizhevsky. _Learning Multiple Layers of Features from Tiny Images_ . PhD thesis, University
[of Toronto, 2009. URL https://www.cs.toronto.edu/˜kriz/learning-featu](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
[res-2009-TR.pdf.](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)


Daniel Lehmann and Marc Ebner. Calculating the Credibility of Test Samples at Inference by a
Layer-wise Activation Cluster Analysis of Convolutional Neural Networks. In _Proceedings of_
_the 3rd International Conference on Deep Learning Theory and Applications -_ _DeLTA_, pp. 34–
43. SciTePress, 2022. ISBN 978-989-758-584-5. doi: 10.5220/0011274000003277. GSCC:
0000001 Backup Publisher: INSTICC ISSN: 2184-9277.


Paul Lezeau, Thomas Walker, Yueqi Cao, Shiv Bhatia, and Anthea Monod. Tropical Expressiv
[ity of Neural Networks, October 2024. URL http://arxiv.org/abs/2405.20174.](http://arxiv.org/abs/2405.20174)
arXiv:2405.20174.


Yajing Liu, Turgay Caglar, Christopher Peterson, and Michael Kirby. Integrating geometries of
ReLU feedforward neural networks. _Frontiers in Big Data_, 6:1274831, November 2023a. ISSN
[2624-909X. doi: 10.3389/fdata.2023.1274831. URL https://www.frontiersin.org/](https://www.frontiersin.org/articles/10.3389/fdata.2023.1274831/full)
[articles/10.3389/fdata.2023.1274831/full. GSCC: 0000030.](https://www.frontiersin.org/articles/10.3389/fdata.2023.1274831/full)


13


Yajing Liu, Christina M. Cole, Chris Peterson, and Michael Kirby. ReLU Neural Networks, Poly
hedral Decompositions, and Persistent Homology. In _Proceedings of 2nd Annual Workshop on_
_Topology, Algebra, and Geometry in Machine Learning (TAG-ML)_, pp. 455–468. PMLR, Septem[ber 2023b. URL https://proceedings.mlr.press/v221/liu23a.html.](https://proceedings.mlr.press/v221/liu23a.html)


Cl´ emence Magnien, Matthieu Latapy, and Michel Habib. Fast computation of empirically tight
bounds for the diameter of massive graphs. _ACM Journal of Experimental Algorithmics_, 13,
[February 2009. ISSN 1084-6654, 1084-6654. doi: 10.1145/1412228.1455266. URL https:](https://dl.acm.org/doi/10.1145/1412228.1455266)
[//dl.acm.org/doi/10.1145/1412228.1455266.](https://dl.acm.org/doi/10.1145/1412228.1455266)


Marissa Masden. Algorithmic Determination of the Combinatorial Structure of the Linear Regions
of ReLU Neural Networks. _SIAM Journal on Applied Algebra and Geometry_, 9(2):374–404, June
[2025. ISSN 2470-6566. doi: 10.1137/24M1646996. URL https://epubs.siam.org/d](https://epubs.siam.org/doi/10.1137/24M1646996)
[oi/10.1137/24M1646996.](https://epubs.siam.org/doi/10.1137/24M1646996)


Guido F Montufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the Number of Lin
ear Regions of Deep Neural Networks. In _Advances in Neural Information Processing Systems_,
[volume 27. Curran Associates, Inc., 2014. URL https://papers.nips.cc/paper/542](https://papers.nips.cc/paper/5422-on-the-number-of-linear-regions-of-deep-neural-networks)
[2-on-the-number-of-linear-regions-of-deep-neural-networks. GSCC:](https://papers.nips.cc/paper/5422-on-the-number-of-linear-regions-of-deep-neural-networks)
0002803.


Guido Mont´ufar, Yue Ren, and Leon Zhang. Sharp Bounds for the Number of Regions of Maxout
Networks and Vertices of Minkowski Sums. _SIAM Journal on Applied Algebra and Geometry_,
[6(4):618–649, December 2022. ISSN 2470-6566. doi: 10.1137/21M1413699. URL https:](https://epubs.siam.org/doi/10.1137/21M1413699)
[//epubs.siam.org/doi/10.1137/21M1413699.](https://epubs.siam.org/doi/10.1137/21M1413699)


Razvan Pascanu, Guido Montufar, and Yoshua Bengio. On the number of response regions of
deep feed forward networks with piece-wise linear activations, February 2014. URL [http:](http://arxiv.org/abs/1312.6098)
[//arxiv.org/abs/1312.6098. arXiv:1312.6098.](http://arxiv.org/abs/1312.6098)


Maithra Raghu, Ben Poole, Jon Kleinberg, Surya Ganguli, and Jascha Sohl-Dickstein. On the
Expressive Power of Deep Neural Networks. In Doina Precup and Yee Whye Teh (eds.), _Pro-_
_ceedings of the 34th International Conference on Machine Learning_, volume 70 of _Proceed-_
_ings of Machine Learning Research_, pp. 2847–2854. PMLR, August 2017. URL [https:](https://proceedings.mlr.press/v70/raghu17a.html)
[//proceedings.mlr.press/v70/raghu17a.html. GSCC: 0000949.](https://proceedings.mlr.press/v70/raghu17a.html)


David Rolnick and Konrad Kording. Reverse-engineering deep ReLU networks. In _Proceedings_
_of the 37th International Conference on Machine Learning_, pp. 8178–8187. PMLR, November
[2020. URL https://proceedings.mlr.press/v119/rolnick20a.html. GSCC:](https://proceedings.mlr.press/v119/rolnick20a.html)
0000110.


Thiago Serra, Christian Tjandraatmadja, and Srikumar Ramalingam. Bounding and Counting Lin
ear Regions of Deep Neural Networks. In _Proceedings of the 35th International Conference on_
_Machine Learning_ [, pp. 4558–4566. PMLR, July 2018. URL https://proceedings.mlr.](https://proceedings.mlr.press/v80/serra18b.html)
[press/v80/serra18b.html.](https://proceedings.mlr.press/v80/serra18b.html)


Hoang-Dung Tran, Diago Manzanas Lopez, Patrick Musau, Xiaodong Yang, Luan Viet Nguyen,
Weiming Xiang, and Taylor T. Johnson. Star-Based Reachability Analysis of Deep Neural Networks. In Maurice H. Ter Beek, Annabelle McIver, and Jos´e N. Oliveira (eds.), _Formal Meth-_
_ods – The Next 30 Years_, volume 11800, pp. 670–686. Springer International Publishing, Cham,
2019. ISBN 9783030309411 9783030309428. doi: 10.1007/978-3-030-30942-8 39. URL
[http://link.springer.com/10.1007/978-3-030-30942-8_39.](http://link.springer.com/10.1007/978-3-030-30942-8_39)


Martin Trimmel, Henning Petzka, and Cristian Sminchisescu. TropEx: An Algorithm for Extracting
Linear Terms in Deep Neural Networks. In _International Conference on Learning Representa-_
_tions_ [, 2021. URL https://openreview.net/forum?id=IqtonxWI0V3. GSCC:](https://openreview.net/forum?id=IqtonxWI0V3)
0000014.


Calvin Tsay, Jan Kronqvist, Alexander Thebelt, and Ruth Misener. Partition-Based Formulations for
Mixed-Integer Optimization of Trained ReLU Neural Networks. In A. Beygelzimer, Y. Dauphin,
P. Liang, and J. Wortman Vaughan (eds.), _Advances in Neural Information Processing Systems_,
[2021. URL https://openreview.net/forum?id=jhd62iKzRuj.](https://openreview.net/forum?id=jhd62iKzRuj)


14


Shaojie Xu, Joel Vaughan, Jie Chen, Aijun Zhang, and Agus Sudjianto. Traversing the Local Poly
topes of ReLU Neural Networks. In _The AAAI-22 Workshop on Adversarial Machine Learning_
_and Beyond_ [, 2022. URL https://openreview.net/forum?id=EQjwT2-Vaba.](https://openreview.net/forum?id=EQjwT2-Vaba)


Xiaodong Yang, Hoang-Dung Tran, Weiming Xiang, and Taylor Johnson. Reachability Analysis for
[Feed-Forward Neural Networks using Face Lattices, March 2020. URL http://arxiv.or](http://arxiv.org/abs/2003.01226)
[g/abs/2003.01226. arXiv:2003.01226.](http://arxiv.org/abs/2003.01226)


Liwen Zhang, Gregory Naitzat, and Lek-Heng Lim. Tropical Geometry of Deep Neural Networks.
In Jennifer Dy and Andreas Krause (eds.), _Proceedings of the 35th International Conference on_
_Machine Learning_, volume 80 of _Proceedings of Machine Learning Research_, pp. 5824–5832.
[PMLR, July 2018. URL https://proceedings.mlr.press/v80/zhang18i.html.](https://proceedings.mlr.press/v80/zhang18i.html)
GSCC: 0000164.


Xiao Zhang and Dongrui Wu. Empirical Studies on the Properties of Linear Regions in Deep Neural
Networks. In _International Conference on Learning Representations_, September 2019. URL
[https://openreview.net/forum?id=SkeFl1HKwr.](https://openreview.net/forum?id=SkeFl1HKwr)


Tan Zhu, Fei Dou, Xinyu Wang, Jin Lu, and Jinbo Bi. Polyhedron Attention Module: Learning
Adaptive-order Interactions. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and
S. Levine (eds.), _Advances in Neural Information Processing Systems_, volume 36, pp. 9213–
[9225. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d83ad88759cef8192451543e5d59bf6-Paper-Conference.pdf)
[_files/paper/2023/file/1d83ad88759cef8192451543e5d59bf6-Paper-C](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d83ad88759cef8192451543e5d59bf6-Paper-Conference.pdf)
[onference.pdf. GSCC: 0000000.](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d83ad88759cef8192451543e5d59bf6-Paper-Conference.pdf)


15


APPENDIX


A DETAILED RELATED WORK


Several works have established lower and upper bounds on the _maximum_ possible number of polyhedra (the regions in Fig. 1(b)) in ReLU-type networks (networks with piecewise-linear activation
functions) in terms of input dimension, depth and width of the network (Pascanu et al., 2014; Montufar et al., 2014; Montuf´ ar et al., 2022; Serra et al., 2018; Montuf´ ar et al., 2022; Goujon et al., 2024).
For example, when the hidden layer width is always larger than the input dimension, the number
of linear regions grows exponentially with respect to both the input dimension and the number of
hidden layers (Montufar et al., 2014). Another direction bounds from above the _expected_ number
of polyhedra over all networks of the same architecture (Hanin & Rolnick, 2019b). These bounds
have been compared in a recent survey (Huchette et al., 2023), which summarizes how lower bounds
are found by constructing networks with many polyhedral regions while upper bounds are found by
determining the effect of adding additional layers of various sizes to existing networks. The number
of polyhedra affects how the network transforms a trajectory in the input space after each layer of
mapping (Hanin & Rolnick, 2019a), and the length of the transformed trajectory refects the network
expressivity (Raghu et al., 2017).


Several theoretical studies characterize the polyhedral structure of ReLU-like networks. A number
of them have established links to tropical geometry (Zhang et al., 2018; Fournier, 2019; Charisopoulos & Maragos, 2019; Trimmel et al., 2021; Lezeau et al., 2024), which can be used to bound the total
number of polyhedra and provide implicit formulas for the networks’ decision boundaries. Analysis of ReLU networks as max-affne spline operators (Balestriero & Baraniuk, 2018; Balestriero &
baraniuk, 2018; Balestriero et al., 2019) has identifed a correspondence between the boundaries of
the network’s polyhedral regions as roots of a polynomial. Although this does not provide a direct
way to see how the polyhedra ft together, this theory has been successfully applied in detecting toxicity in large language models by analyzing them individually (Balestriero et al., 2024). Topology
has been essential for describing how the architecture of a network affects its geometry and measuring its complexity in terms of Betti numbers (Grigsby & Lindsey, 2022; Grigsby et al., 2024).
An activation state is a sign vector indicating whether the ReLU in the corresponding hidden neuron
has been activated for an input. A topological approach proves that there is a 1-1 correspondence
between the network’s activation states and its polyhedral regions and can calculate the polyhedron
boundaries using this description of its combinatorial structure (Masden, 2025). This paper builds on
top of this work, which constructs the robust theoretical framework we use to study ReLU geometry
(see Section 2 for more details). A more effcient method for computing the boundaries is developed
later on (Berzins, 2023), which builds on a previous method that can determine the architecture and
parameters of ReLU networks via sampling (Rolnick & Kording, 2020). These works all provide
tools for deeper analysis of network geometry, but they have not examined the connectivity of the
polyhedra, a fundamental property of the complex.


There are also empirical studies that calculate the polyhedral complex and demonstrate a number
of practical applications. Due to the intractability of enumerating all polyhedra, existing methods
iteratively solve linear programs (LPs) that locally search confned regions of the input space (Liu
et al., 2023a; Xu et al., 2022; Liu et al., 2023b) to count polyhedra or perform reachability analysis (Tran et al., 2019; Yang et al., 2020). The behavior of ReLU networks on bounded input spaces
can also be exactly described with mixed-integer LPs with decision variables representing activation states, leading to a signifcant body of work improving the formulations of the networks with
constraints (Huchette & Vielma, 2023; Tsay et al., 2021; Anderson et al., 2019) and applying them
to problems such as verifcation (Botoeva et al., 2020; Cheng et al., 2017; Bunel et al., 2018) and
inverse design (Ansari et al., 2022). Network architecture has been shown to infuence the volume
and distribution of the polyhedra (Zhang & Wu, 2019), which are then applied to quantify how the
network generalizes to new data (Ji et al., 2022; Lehmann & Ebner, 2022; Daroczy´, 2020). ReLU
networks can be modifed at the polyhedral level with the aim of improving explainability (Zhu
et al., 2023; Hu, 2021). Other works study their geometric structure through the lens of activation
states of the network on various inputs. For instance, a few methods use the Hamming distance
between activation states as a proxy for similarity between the data points in different polyhedra,
with various downstream applications ranging from data compression (He et al., 2021) to open set
detection (Jamil et al., 2023) and clustering (Craighero et al., 2020). However, empirical studies


16


have not examined the connectivity of the polyhedral complex and how it relates to the distribution
of training data.


B PROOFS OF THEORETICAL RESULTS


**Defnition B.1** (Hypercube Graph) **.** A _k_ -hypercube graph is a graph formed from the vertices and
edges of a _k_ -hypercube. It can also be constructed as a graph with nodes corresponding to every
string of _k_ bits and edges between nodes with strings that differ by one bit.


**Defnition B.2** (Polyhedron) **.** A polyhedron is a (possibly unbounded) intersection of halfspaces, or
equivalently, the set of solutions to a system of linear inequalities of the form _{x_ _∈_ R _[d]_ : Φ _x_ + _β_ _≤_ 0 _}_
for some real ( _n,_ _d_ )-matrix Φ and some _β_ _∈_ R _[n]_ .


**Defnition B.3** (Polyhedral Complex) **.** A Polyhedral Complex _C_ is a set of polyhedra satisfying the
following properties:


1. Closure under intersection, that is, _c_ 1 _, c_ 2 _∈C_ implies _c_ 1 _∩_ _c_ 2 _∈C_ . Note that this can be the
empty set.


2. Closure under taking faces. That is, if _c_ _∈C_ and _c_ _[′]_ is a face of _c_, then _c_ _[′]_ _∈C_


We restrict our discussion to ReLU networks with the following two properties:


**Genericity** (Grigsby & Lindsey (2022), Defnition 2.4): A hyperplane arrangement in R _[d]_ is called
_generic_ if all sets of _k_ hyperplanes intersect in an affne space of dimension _d_ _−_ _k_ for 1 _≤_ _k_ _≤_ _d_ .
In a ReLU network, each activation region is the intersection of half-spaces created by each neuron
in the network, the boundaries of which form a hyperplane arrangement in the layer’s input space.
A neural network is generic if the hyperplane arrangements corresponding to each of its regions are
generic.


**Supertransversality** (Masden (2025), Defnition 11): Let _f_ be a ReLU network with complex _C_
and denote by _fi_ : R _[m][i][−]_ [1] _→_ R [out] the output of the last _ℓ_ _−_ _i_ layers of _f_ with complex _Ci_ _⊂_
R _[m][i][−]_ [1] . Denote by _Hi_ the polyhedral complex in R _[m][i][−]_ [1] created by the hyperplane arrangement
_{x_ _∈_ R _[m][i][−]_ [1] : _πj_ ( _W_ _[i][−]_ [1] _x_ + _b_ ) = 0 _}_ where _πj_ is the projection onto the _j_ -th coordinate for some
1 _≤_ _j_ _≤_ _mi_ . Suppose for each layer 1 _≤_ _i_ _≤_ _ℓ_ and each 1 _≤_ _k_ _≤_ _d_ that the restriction of _fi_ to the
interior of every _k_ -cell of the complex of _Hi−_ 1 is transverse to the interior of all cells of _Ci_ . Then,
the network _f_ is called _supertransversal_ .


Defne _Nk_ ( _C_ ) to be the number of _k_ -cells in the polyhedral complex _C_ .

**Theorem 3.4.** _The average number of faces of a d-cell of C_ _in_ R _[d]_ _(i.e., the average degree of the_
_connectivity graph) is at most_ 2 _d._


_Proof._ A sign sequence that has exactly one 0 in its elements corresponds to a ( _d_ _−_ 1)-cell. Each
( _d_ _−_ 1)-cell forms a boundary of exactly two _d_ -cells in the polyhedral complex _C_, and the single
0 in its corresponding sign sequence can be set to _−_ 1 or 1 to obtain the sign sequences of the two
_d_ -cells it separates. Thus, the sum of the numbers of faces of every _d_ -cell is just twice the number
of ( _d_ _−_ 1)-cells, and the average number of faces of a _d_ -cell of _C_ is 2 _NNdd−_ (1 _C_ () _C_ ) [. We want to show that ]

2 _NNdd−_ (1 _C_ () _C_ ) _≤_ 2 _d_, or equivalently, _Nd−_ 1( _C_ ) _≤_ _dNd_ ( _C_ ). We perform mathematical induction on both
the dimension of the (sub)complex _d_ and the number of neurons in the network _n_ .


**Base Case**


If _d_ = 1, each _d_ -cell is a 1-cell, which can have either 0, 1, or 2 faces. Thus, the average number of
faces of _d_ -cells is at most 2.


If _n_ = 1 (the network only consists of a single neuron), the arrangement consists of a single hyperplane, so the average number of faces of a _d_ -cell is 1 and the lemma holds regardless of the value of
_d_ .


17


**Inductive Step**


We now prove that the statement holds for complexes of ReLU networks with _n_ neurons in _d_ dimensions if it holds for both _n_ _−_ 1 neurons in _d_ dimensions and _n_ _−_ 1 neurons in _d_ _−_ 1 dimensions.


Assume for any ReLU complex _C_ _[′]_ with _n_ _−_ 1 neurons (or _n_ _−_ 1 BHs) in _d_ dimensions, we have

_Nd−_ 1( _C_ _[′]_ ) _≤_ _dNd_ ( _C_ _[′]_ ) _._ (4)


and for any ReLU complex _C_ _[′]_ with _n_ _−_ 1 neurons in _d_ _−_ 1 dimensions, we have

_Nd−_ 2( _C_ _[′]_ ) _≤_ ( _d_ _−_ 1) _Nd−_ 1( _C_ _[′]_ ) _._ (5)


Let _C_ be a ReLU complex of a network _f_ with _n_ neurons in _d_ dimensions. Let _hi_ be the BH
corresponding to a particular neuron _i_ in the last layer of _f_ . By Lemma 3.3 where _k_ = _d_,


_Nd_ ( _C_ ) = _Nd_ ( _C −_ _hi_ ) + _Nd−_ 1( _hi_ ) _._ (6)


Because _C −_ _hi_ is a complex with _n_ _−_ 1 BHs in _d_ dimensions, Eq.(4) holds. Substituting Eq.(4) into
Eq.(6) yields

1
_Nd_ ( _C_ ) _≥_ _Nd−_ 1( _C −_ _hi_ ) + _Nd−_ 1( _hi_ ) _._ (7)
_d_
Lemma 3.3 also applies to _Nd−_ 1( _C −_ _hi_ ) where _k_ = _d_ _−_ 1. Substituting it into Eq.(7) yields


1
_Nd_ ( _C_ ) _≥_ [ _Nd−_ 1( _C_ ) _−_ _Nd−_ 1( _hi_ ) _−_ _Nd−_ 2( _hi_ )] + _Nd−_ 1( _hi_ ) _,_
_d_


which can be equivalently written as:


1 1
_Nd_ ( _C_ ) _≥_ _Nd−_ 1( _C_ ) _−_ [ _Nd−_ 1( _hi_ ) + _Nd−_ 2( _hi_ )] + _Nd−_ 1( _hi_ ) _._ (8)
_d_ _d_

To prove that _Nd−_ 1( _C_ ) _≤_ _dNd_ ( _C_ ), it is equivalent to prove that _Nd_ ( _C_ ) _≥_ _d_ 1 _Nd−_ 1( _C_ ), and so it is
suffcient to show that the remaining terms on the right-hand side of Eq.(8) are at least 0. That is,


1
_Nd−_ 1( _hi_ ) _≥_ [ _Nd−_ 1( _hi_ ) + _Nd−_ 2( _hi_ )] _._
_d_

Multiplying this inequality by _d_ on both sides yields


_dNd−_ 1( _hi_ ) _≥_ _Nd−_ 1( _hi_ ) + _Nd−_ 2( _hi_ ) _._


Then, subtracting _Nd−_ 1( _hi_ ) from both sides yields


( _d_ _−_ 1) _Nd−_ 1( _hi_ ) _≥_ _Nd−_ 2( _hi_ ) _._


Since _hi_ is a ReLU complex in _d_ _−_ 1 dimensions with _n_ _−_ 1 neurons (its sign sequence contains
only one 0), the strict form of this inequality is guaranteed by Eq.(5).


**Theorem 3.1.** _For a ReLU complex in d-dimensional input space, the average number of faces of a_
_k-cell is at most_ 2 _k_ _for k_ = 1 _,_ 2 _, . . ., d._


_Proof._ For _k_ = 1 _, . . ., d_ _−_ 1, each _k_ -cell of a polyhedral complex _C_ corresponds to a sign sequence
that has exactly _d_ _−_ _k_ zeros, so the points in the _k_ -cell are contained in the intersection of these
_d_ _−_ _k_ BHs. Each _k_ -cell belongs to a unique subcomplex of sign sequences where the elements
corresponding to these BHs (neurons) are fxed to 0. Therefore, calculating the average number of
faces of a _k_ -cell involves examining the average number of faces of the _k_ -cells contained in the
subcomplexes restricted to every combination of _d_ _−_ _k_ neurons (i.e., every intersection of _d_ _−_ _k_
BHs and its corresponding subset of sign sequences with 0 in the corresponding positions). These
intersections are _k_ -dimensional ReLU complexes themselves. Thus, Theorem 3.4 guarantees that
the average number of faces for the _k_ -cells in each intersection is at most 2 _k_ . Together, the average
number of faces of the _k_ -cells contained in all different intersections of _d_ _−_ _k_ BHs is at most 2 _k_ .


18


of how we translate the gray plane, its intersection with all the BHs will be identical, and the blue
second layer BH will never intersect the intersection of the two frst-layer hyperplanes. The proof
works by showing that as you move in a direction orthogonal to the intersection of the frst-layer
hyperplanes (shown in orange), the distance to the frst layer hyperplanes stays the same, and so the
output of the frst layer (and thus, the remaining layers) stays the same.


Previous work gives a lower bound on the connectivity of the complex as follows:


**Lemma B.1** (Grigsby et al. (2024), Version 1, Corollary 5.29) **.** _If a ReLU network has at least d_
_neurons in the frst hidden layer, then every cell of C_ _contains a vertex._


If the condition of having at least _d_ neurons in the frst layer is not met, the minimum degree can be
lower. We generalize the previous result as follows:


**Theorem 3.5.** _If a ReLU network has n_ 1 _neurons in the frst hidden layer, every d-cell of C_ _has_
_at least_ min( _n_ 1 _, d_ ) _neighbors, and thus the average degree of the connectivity graph is at least_
min( _n_ 1 _, d_ ) _._


_Proof._ If _n_ 1 _≥_ _d_, then Lemma B.1 implies every cell of _C_ contains a vertex. Equivalently, every
node in the connectivity graph is part of a _d_ -hypercube graph, and the graph has minimum degree
_d_ . On the other hand, assume that _n_ 1 _[≤]_ _d_ . By the assumption of genericity, _W_ 1 _[∈]_ R _[n]_ [1] _×_ R _[d]_ has
rank _n_ 1. Thus, the space spanned by its row vectors _R_ ( _W_ 1) (the normal vectors of the frst layer’s
hyperplanes) has dimension _n_ 1 and its null space _N_ ( _W_ 1) has dimension _d−n_ 1. Since the null space
and row space of _W_ 1 [are orthogonal complements, every point ] _[x]_ _[∈]_ [R] _d_ [can be written uniquely as ]
_x_ = _v_ + _k_ for _v_ _∈R_ ( _W_ 1) and _k_ _∈N_ ( _W_ 1). However, since _W_ 1 _k_ = 0 for all _k_ _∈N_ ( _W_ 1), we
really have that _W_ 1 _x_ + _b_ = _W_ 1( _v_ + _k_ ) + _b_ = _W_ 1 _v_ + _W_ 1 _k_ + _b_ = _W_ 1 _v_ + _b_ . This shows that for
a fxed _v_, _f_ ( _x_ ) = _f_ ( _v_ + _k_ ) remains constant for all _k_ _∈N_ ( _W_ 1). It then immediately follows that
no polyhedron boundaries are ever crossed when _k_ varies. Thus, to count the number of faces of
_d_ -cells in _C_, we can count the number of faces of the corresponding _n_ 1-cells of the complex _Cn_ 1
created by restricting the network to _R_ ( _W_ 1). Applying Lemma B.1, every _n_ 1-cell in _Cn_ 1 contains


1


1

a vertex (0-cell). Every 0-cell in _Cn_ 1 corresponds to a ( _d_ _−_ _n_ 1)-cell in _C_, which means every _d_ -cell

in _C_ contains a ( _d_ _−_ _n_ 1)-cell. A ( _d_ _−_ _n_ 1)-cell’s node in the connectivity graph belongs to a _n_ 1
hypercube subgraph, which implies that the minimum degree of the connectivity graph is at least _n_ 1
when _n_ 1 _≤_ _d_ .


a vertex (0-cell). Every 0-cell in _Cn_ 1 corresponds to a


**Theorem 3.6.** _The average number of faces of d-cells in Cn_ _increases monotonically in terms of n._


_Proof._ To prove this, it is equivalent to prove that adding a new BH _hn_ to the complex (i.e., adding
a new neuron to the network) creates more ( _d_ _−_ 1)-cells than _d_ -cells, _Nd−_ 1( _C_ ) _−_ _Nd−_ 1( _C −_ _hn_ ) _>_
_Nd_ ( _C_ ) _−_ _Nd_ ( _C_ _−_ _hn_ ). Lemma 3.3 with _k_ = _d_ _−_ 1 shows that the left-hand side _Nd−_ 1( _C_ ) _−_
_Nd−_ 1( _C_ _−_ _hn_ ) = _Nd−_ 1( _hn_ ) + _Nd−_ 2( _hn_ ). Applying the lemma with _k_ = _d_ shows that the righthand side _Nd_ ( _C_ ) _−_ _Nd_ ( _C_ _−_ _hn_ ) = _Nd_ ( _hn_ ) + _Nd−_ 1( _hn_ ), but _Nd_ ( _hn_ ) = 0, so the right-hand side
of the equation just equals _Nd−_ 1( _hn_ ). Substituting these equations into the inequality on both sides
yields _Nd−_ 1( _hn_ ) + _Nd−_ 2( _hn_ ) _>_ _Nd−_ 1( _hn_ ). Then, subtracting _Nd−_ 1( _hn_ ) shows the inequality is
equivalent to _Nd−_ 2( _hn_ ) _>_ 0. Since ( _d−_ 2)-cells are created whenever _hn_ intersects other BHs in the
complex, which must happen at least once if any _d_ -cells are created, so this inequality is true. Since


19


after the _d_ -th term the sequence is monotonically increasing and bounded below and above by _d_ and
2 _d_ respectively, the average number of faces must converge to a value between these bounds.


**Theorem 3.7.** _Let f_ _be a shallow network that has only one hidden layer with n_ _nodes. When n_
_goes to infnity, the average number of faces of its d-cells converges exactly to_ 2 _d. That is,_


2 _Nd−_ 1( _Cn_ )
lim = 2 _d._
_n→∞_ _Nd_ ( _Cn_ )


_Proof._ If the network is shallow, the BHs of the neurons form a generic hyperplane arrangement.
According to Buck (1943), the number of -cells in the generic hyperplane arrangement _C_ in _d_
dimensions is given by _Nk_ ( _C_ ) = P _id_ = _d−k_ - _[k]_ _ni_ �� _d−i_ _k_ - [. Intuitively, the frst term in the sum counts]
subsets of hyperplanes that intersect in at least _k_ dimensions, and the second term counts the number
of subsets in each of these sets intersect to form a _k_ -face. Therefore, we have that


2 _Nd−_ 1( _Cn_ ) = 2 P _di_ =1               - _ni_ �� _i_               _Nd_ [(] _[C]_ _n_ [)] P _d_ �� _n_ [1]
_i_ =0 _i_

As _n_ _→∞_, the numerator is dominated by _d_ �� _[n]_ _d_ [and the denominator is dominated by ] �� _nd_ [. Ther][e-]
fore, the expression converges to 2 _d_ .


Let _ℓ_ be the total number of hidden layers (depth), _mj_ be the number of nodes in layer _j_, _j_ =
1 _, . . ., ℓ_, and _m_ = max _{m_ 1 _,_ _· · ·_ _, mℓ}_ (width).
**Theorem 3.8.** _The diameter_ _D_ _of a ReLU complex’s connectivity graph is_ Ω - ln(ln( _Ndn_ () _C_ )) - _and_
_O_ - _m_ _[ℓ]_ - _._


_Proof._ For the lower bound, we can use the fact that the maximum degree of the connectivity graph
is the number of hidden neurons _n_, as a single BH cannot form more than one face of a polyhedron.
The Moore bound (Diestel, 2017) implies that the number of vertices in the connectivity graph is at
most 1+ _n_ + _n_ ( _n−_ 1)+ _· · ·_ + _n_ ( _n−_ 1) _[D][−]_ [1] _< n_ ( _n−_ 1) _[D]_ . We can rearrange this inequality as follows:
_Nd_ ( _C_ ) _< n_ ( _n_ _−_ 1) _[D]_ becomes ln( _Nd_ ( _C_ )) _<_ ln( _n_ ( _n −_ 1) _D_ ), and _n, D_ _≥_ 1, so _D_ _>_ ln( [ln(] _n_ _[N]_ ( _n_ _[d]_ [(] _−_ _[C]_ 1)) [))] .


To get the upper bound, consider two _d_ -cells _p_ 1 _, p_ 2 _∈C_ . If the network only consists of one layer,
we can easily fnd a path of polyhedra from _p_ 1 to _p_ 2 in this hyperplane arrangement by drawing a
straight line from a point within _p_ 1 to a point within _p_ 2 and fipping the sign of each element in
_S_ ( _p_ 1) every time we cross its BH. Each fip then yields the sign sequence of the next polyhedron
on the path, and since the BHs are just hyperplanes, the line is guaranteed to cross each of the BHs
on the path and fip the corresponding sign only once. For deeper networks, the problem is more
diffcult because BHs are not necessarily hyperplanes, and we may have to cross the same one more
than once to get from one polyhedron to another. For a neural network with 2 layers, consider the
BHs of the second layer as subdivisions of the polyhedra of the hyperplane arrangement created by
the frst layer. While walking along a path between two of these layer-1 polyhedra, we have to cross
some of these subdivisions. Now, if we want to fnd a path between two polyhedra _p_ 1 and _p_ 2 in
this network, we can still fnd one that only crosses each frst-layer BH at most once, because we
can ignore the second layer and fnd a path between the polyhedra from the frst layer’s hyperplane
arrangement that contains _p_ 1 and _p_ 2. Inside each of the _m_ 1 + 1 polyhedra created from only the
frst layer’s BHs, the segments of each BH from the second layer are contained by a hyperplane.
Recursively treating them as their own hyperplane arrangement, we know we can get from one side
of a frst-layer polyhedron to the other by crossing each second-layer hyperplane at most once. Thus,
the total number of frst and second layer BHs we must pass through to get from _p_ 1 to _p_ 2 is bounded
above by ( _m_ 1 + 1) _×_ _m_ 2. For deeper networks, we can continue this recursion for each layer. Then,
the maximum number of faces we would have to pass through to get from one to any other is upper
bounded by Q1 _≤j≤ℓ_ [(] _[m][j]_ [+] [1)] _[≤]_ [(] _[m]_ [+] [1)] _ℓ_ [. The dominant term of ] ( _m_ + 1) _ℓ_ as _m,_ _ℓ_ _→∞_ is _m_ _[ℓ]_,
which shows that the connectivity graph diameter is _O_ - _m_ _[ℓ]_ �.


20


This diameter upper bound is the tightest asymptotic bound possible without factoring in the dimension of the complex. For _x_ _∈_ R, the following equations give a non-degenerate 1-dimensional
network with width _m_ = 2, depth _ℓ_, and more than _m_ _[ℓ]_ linear regions. Denote by _fi,j_ the _j_ th neuron
of layer _i_ with 1 _≤_ _i_ _≤_ _ℓ_ and _j_ _∈{_ 1 _,_ 2 _}_ .


_f_ 1 _,_ 1( _x_ ) = max( _x_ + 1 _,_ 0)
_f_ 1 _,_ 2( _x_ ) = max(2 _x_ _−_ 1 _,_ 0)

3 _j_ _−_ 2
_fi,j_ ( _x_ ) = 1 _._ 1 _jfi−_ 1 _,_ 1( _x_ ) _−_ 1 _._ 1 _jfi−_ 1 _,_ 2( _x_ ) _−_ 4 _i−_ 1
_fℓ,_ 1( _x_ ) = _fr_ ( _fℓ,_ 1( _x_ ) _−_ _fℓ,_ 2( _x_ ))


See an interactive demonstration of this network here:


[https://www.desmos.com/calculator/rc6bkezojk.](https://www.desmos.com/calculator/rc6bkezojk)


C ADDITIONAL CATEGORIZATION EXAMPLES


_**1**_


_**1**_


_**1**_


(c) Coloring with _i_ = 3 for the
subcomplex where _S_ ( _c_ )4 = 0


(a) Coloring with _i_ = 1


(b) Coloring with _i_ = 4 for the
subcomplex where _S_ ( _c_ )3 = _−_ 1


Figure 9: Alternate categorizations for Fig. 3 with different choices of _i_ and restrictions to different
subcomplexes of _C_ . Cells in Categories 1, 2, and 3 are shown in blue, green, and red, respectively.
Cells that are not included in the complex are shown in gray. In Fig. 9c, the only _hi_ cells are the
blue vertices, and each one splits a pair of 1-cells in the subcomplex.


D ALGORITHMIC DETAILS


In this section, we describe and further explain the role of the subroutine SOLVELP() in Algorithm
1, which was executed by Gurobi [1 ] during our experiments. The SOLVELP() function takes in inputs

_−_ Φ _si_, Φ _s_, and _βs_ + _ei_, which are calculated based on _s_ according to Eq. 2 and Eq. 3 to determine
the affne functions forming the boundary of the corresponding polyhedron. To check if a linear
inequality is redundant we solve the following LP for the best _x_ _[∗]_ and we evaluate whether Φ _[T]_ _si_ _[x]_ _∗_ [+]
_βsi_ _≤_ 0.


[1https://www.gurobi.com](https://www.gurobi.com)


21


SOLVELP( _−_ Φ _si,_ Φ _s, βs_ + _ei_ ) = maximize _−_ Φ _[T]_ _i_ _[x]_

subject to Φ _[T]_ 1 _[x]_ _[≤−][β]_ [1]
Φ _[T]_ 2 _[x]_ _[≤−][β]_ [2]
...

Φ _[T]_ _i−_ 1 _[x]_ _[≤−][β][i][−]_ [1]

Relaxed Constraint _→_ Φ _[T]_ _i_ _[x]_ _[≤−][β][i]_ [+ 1]

Φ _[T]_ _i_ +1 _[x]_ _[≤−][β][i]_ [+1]
...

Φ _[T]_ _n−_ 1 _[x]_ _[≤−][β][n][−]_ [1]
Φ _[T]_ _n_ _[x]_ _[≤−][β][n]_


Let _p_ _∈C_ be a _d_ -cell with sign sequence _s_ . Our goal is to fnd the neighbors of _p_, which correspond
to _d_ -cells with sign sequences that differ from _s_ in one position. If _p_ is separated by a face from the
neighbor differing in sign sequence position _i_, this face will be contained by the BH of the neuron _i_
in the network. Therefore, to fnd the neighbors of _p_, we can equivalently determine which neurons
have BHs that contain the faces of _p_, which we can do by checking each neuron individually. As
described in Section 2, when _s_ is fxed and the network defnes an affne function, the neuron _i_
defnes a half-space (a linear inequality of the form Φ _[T]_ _si_ _[x]_ [+] _[β][s]_ _i_ _[≤]_ 0) where the sign of its affne
function is indicated by _si_ . This half-space contains _p_ by defnition, and the intersection of the halfspaces of all of the neurons in _f_ is exactly _p_ . However, not all of the half-spaces contain faces of
_p_, as some may be redundant. To test whether a specifc neuron _i_ ’s BH forms a face of _p_, we push
the boundary hyperplane of _i_ ’s half-space backwards along its normal vector and check whether
that allows us to walk further in that direction before we leave one of the half-spaces. That is,
following (Fukuda, 2004), we relax the less-than inequality corresponding to neuron _i_ by adding
a small value to the right-hand side and fnd the point _x_ in the new system of inequalities that
maximizes _−_ Φ _[T]_ _six_, where _−_ Φ _[T]_ _si_ [is the opposite direction of the normal vector of the hyperplane ]
bounding the half-space. If _x_ _[∗]_ violates the original inequality Φ _[T]_ _si_ _[x]_ [+] _[β][s]_ _i_ _[≤]_ [0][, then it is outside of ]
_p_, so that half-space is not redundant and the face of _p_ contained by its boundary is a ( _d_ _−_ 1)-cell

of _hi_, which we can cross (equivalently, fip _si_ ’s sign) to get to a new polyhedron. If _x_ _[∗]_ does not
violate the original constraint, then the half-space defned by neuron _i_ is redundant and its BH does
not contain a face of _p_, so fipping the sign of _si_ does not yield a neighbor of _s_ and may not even
correspond to any element of _C_ .


E ADDING BENT HYPERPLANES TO THE DUAL GRAPH


Here we describe how the process of adding a new neuron to the network, going from _Cn_ to _Cn_ +1
via the addition of BH _hn_ +1, can be represented in the connectivity graph. Each step is visualized
in Fig. 10. When a new neuron’s BH is added to an existing complex, it splits each of a connected
set of existing _d_ -cells in _Cn_ (the red nodes identifed in step 1) in half. We frst disconnect the nodes
representing _d_ -cells getting split from the rest of the graph, which form a vertex cut (step 2). Then,
we duplicate the induced subgraph of the vertex cut, with each copy representing the Category 3
_d_ -cells in _Cn_ +1 formed on either side of the new neuron’s BH (step 3). After that, we add a new
edge between each corresponding node in the two copies, representing the Category 1 ( _d_ _−_ 1)-cell
between each of those _d_ -cells (step 4). Together, this set of new edges represents the added neuron’s
BH. We now have to add back the edges we deleted between the vertex cut and the rest of the graph,
which represent Category 2 ( _d_ _−_ 1)-cells that separated the _d_ -cells that got split from the _d_ -cells that
did not (step 5). For each edge we removed in the second step, we add a new edge from its original
outside-the-cut node to one of the two copies of its original in-the-cut node, the copy representing the
polyhedron in _Cn_ +1 that is on the same side of the new BH as the original outside-the-cut node. This
demonstrates the monotonicity of the average number of faces as network size increases (Theorem


22


3.6), since every added node uniquely corresponds to at least one new added edge regardless of the
value of _d_ .


Figure 10: Connectivity graph modifcation when adding another fnal-layer neuron to a neural
network with _d_ = 3. Red nodes represent the _d_ -cells that are split in half by a new neuron’s BH.
Note that the complex before and after is made up of cube graphs, and the subcomplex of split 3cells is made up of square graphs (2-hypercubes).


F EXPERIMENTAL DETAILS


For all experiments, datasets were randomly split into 80% train and 20% test. Models were trained
using vanilla SGD with a fxed learning rate. Data from all real datasets was normalized, and labels
were normalized for the regression task. All networks were trained on a single GPU. Calculating
the polyhedra in their complexes was distributed across 32 processors, and took up to twelve hours
for the largest networks.


For the experiments on synthetic data, all networks were trained for 10 epochs with a batch size of
64 and a learning rate of 0.01. The datasets used in these experiments were generated using ScikitLearn [2] . For the experiments with real data, setup information and performance metrics are listed in
Table 2.

Table 2: Architecture, training hyperparameters, and performance of networks trained on real-world
data. For the regression task, we report coeffcient of determination _R_ [2] and Mean Squared Error
(MSE), while for the classifcation tasks, we report accuracy and Receiver Operating Characteristic
Area Under the Curve (AUC) on the test set.


Dataset Name CA Housing CIFAR10 MNIST


Dataset Size 20640 60000 70000
Task Regression Classifcation Classifcation
# Classes NA 10.00 10.00
Batch Size 64 4 64
Epochs 60 30 50
Learning Rate 0.00 0.01 0.10
Input Dimension 8 10 5
Hidden Layer Sizes (128) (64, 64) (8, 8, 8)
Test Accuracy NA 0.64 0.90
Test AUC NA 0.94 0.99
Test _R_ [2] 0.65 NA NA
Test MSE 0.34 NA NA


For MNIST and CIFAR10, we split the trained networks into two parts and regard the frst part as
a feature extractor. Our experiments are then performed on the second part, a classifer, which has
a smaller input dimension. The California Housing dataset only has 8 features, so we regard the
full network as the classifer. We report the input dimension and hidden layer sizes of the classifer


[2https://scikit-learn.org](https://scikit-learn.org)


23


networks in Table 2. Working with the lower input dimension allows us to calculate a larger portion
of the polyhedral complex associated with the network, which gives us a more complete picture
of its structure. For the MNIST network, the feature extractor consists of a single fully connected
layer with a ReLU activation. For the CIFAR10 network, the feature extractor consists of two
convolutional layers with 5 _×_ 5 kernels computing 6 and 16 features, both followed by a 2 _×_ 2
max-pooling layer and a ReLU layer, before a fnal fully-connected layer with a ReLU activation.


G EXTENDED EXPERIMENTAL RESULTS


We performed additional experiments and included the related results here. Fig. 11 illustrates the
same results in Fig. 5 but with the theoretically computed lower diameter bound on the x-axis instead
of the upper bound.


Fig. 12 expands our results in Fig. 4 to include results for networks with input dimension equal to
2 and 3 and display the average number of neighbors for each combination of width, depth, and
dimension averaged over the 5 trials.


Table 3 provides more detailed results for the number of polyhedra, the average degree, the average
volume of each bounded polyhedron, the percentage of bounded regions among all regions, and the
graph diameter for different networks with input dimension between 2 and 5, numbers of hidden
layers between 1 and 4, and width in _{_ 4 _,_ 8 _,_ 16 _}_ . We do observe that the number of polyhedra
exponentially increases with the number of hidden layers (depth) and with the number of neurons
in each layer (width). However, the average degree quickly approaches the upper bound for each
dimension.


Fig. 13 shows how the volume and inradius of bounded polyhedra in the MNIST complex are related
to their numbers of neighbors and whether or not they contain data. Although inradius has been
previously used to estimate the volumes of polyhedra in higher-dimensional complexes where exact
calculations are intractable, this shows an example of a case where the two values are not closely
correlated. Note that unbounded polyhedra are not included in this fgure.


Fig. 14 shows how the distribution of neighbor counts in Fig. 6a changes over the course of training.
After each epoch, we recalculate the entire complex and check the positions of the same 10,000 data
points. In this instance, we observe that as the network trains, the data is gradually surrounded by
higher numbers of polyhedra with relatively many neighbors.


120


100


80


60


40


20


_d_
2


3


4


5


_d_
2


3


4


5


0

|Col1|Col2|Col3|Col4|Col5|.<br>'•|Col7|Col8|Col9|Col10|"<br>I|Col12|Col13|Col14|Col15|Col16|Col17|Col18|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||<br>II|<br>II|<br>II|<br> ~~'·~~<br> _I,_|||||<br>||||||||
|||||||||||||||||||
|||||||||||||||||||
|||||||||||||||||||
|||||||||||||||||||
|||||||||||||||||||
|||||||||||||||||||
|||||||||||||||||||

1.4 1.6 1.8 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4

Lower Diameter Bound


Figure 11: The same plot as Fig. 5 but with the lower bound from Theorem 3.8 on the _x_ -axis and
experiments from all widths plotted together.


24


# Hidden Layers 1 2 3 4
Width: 4 Width: 8 Width: 16


3.85
3.84
~~3.76~~


3.94
3.91
~~3.87~~


1M


10k


100


1


1M


10k


100


1


1M


10k


100


1


1M


10k


100


1


|♦ =-• 1d•1 lil illll 2.91|♦ •••  1lllli li 3.39|♦ i1 3.75|Col4|
|---|---|---|---|
|<br>**1d•1**<br>~~**ililll**~~<br>~~♦ ~~<br>5.29|<br>~~1llll~~<br>~~**li**~~<br>~~♦ ~~<br>5.77|~~1~~<br>~~♦ ~~<br>5.91||
|<br>♦ <br> <br>3.73<br>~~4.70~~<br>4.99<br>|<br>♦ <br> <br>5.01<br>~~5.59~~<br>5.72<br>|<br>♦ <br> <br>5.55<br>~~5.80~~<br>5.88<br>|<br> <br>|
|~~1111~~~~**1**1 ~~<br>~~: ll~~<br>~~♦ ~~<br>6.16|<br>~~lllili11 : 1llll~~<br>~~♦ ~~<br>7.71|~~  lllill11~~<br>~~♦ ~~<br>7.85|<br>|
|<br> <br>4.00<br>~~5.38~~<br>5.89<br>|<br> <br> <br>6.28<br>~~7.29~~<br>7.52<br>|<br> <br> <br>7.32<br>~~7.73~~<br>7.83<br>|<br> <br>|
|~~111111~~<br>~~: lll~~<br>~~♦ ~~<br>♦ <br><br>~~5.84~~<br>6.53<br>6.81|~~lllllilil111~~<br>~~: lllll~~<br>~~♦ ~~<br>~~**1iiii11**~~<br>♦ <br>1 <br>**1111**<br><br>~~8.86~~<br>9.07<br>9.03|~~ lllllliillll~~<br> <br>~~♦ ~~<br>•····<br>♦ <br><br>~~9.62~~<br>9.78<br>9.81|<br>~~: ~~<br> <br>|


3.67
3.46
~~3.40~~


3.84 3.91
~~3.76~~ ~~3.87~~


5 10 15


5 10 15 20 25


Number of Neighbors


5 10 15 20 25 30


Figure 12: Distributions for the number of faces of polyhedra in the decompositions of trained ReLU
networks with varying input dimension (row), width (column), and depth (color). Each colored bar
shows the average number of polyhedra with a given number of faces created by a certain architecture across 5 runs, with error bars representing the standard deviation. Diamonds represent the
average number of faces in each network.


25


Table 3: Summary statistics for the distributions in Fig. 12. Diameter for each complex is estimated
as described in Section 5.1.


_d_ # Hidden Width


# Regions Avg # Facets Average Volume % Finite Diameter


2 1 4 11.00 _±_ 0.00 2.91 _±_ 0.00 0.63 _±_ 0.11 0.27 _±_ 0.00 4.50 _±_ 0.00

8 37.00 _±_ 0.00 3.46 _±_ 0.00 1.45 _±_ 0.99 0.57 _±_ 0.00 9.10 _±_ 0.22
16 137.00 _±_ 0.00 3.74 _±_ 0.00 3.34 _±_ 4.48 0.77 _±_ 0.00 19.20 _±_ 0.76

2 4 30.80 _±_ 4.15 3.37 _±_ 0.09 33.75 _±_ 67.35 0.57 _±_ 0.05 8.70 _±_ 1.04

8 135.20 _±_ 9.88 3.75 _±_ 0.03 13.19 _±_ 11.50 0.78 _±_ 0.03 18.50 _±_ 1.46
16 541.20 _±_ 26.81 3.87 _±_ 0.01 4.81 _±_ 3.16 0.88 _±_ 0.01 41.80 _±_ 1.82

3 4 58.20 _±_ 14.58 3.52 _±_ 0.10 116.74 _±_ 147.94 0.67 _±_ 0.07 11.70 _±_ 1.44

8 275.20 _±_ 51.57 3.82 _±_ 0.02 45.21 _±_ 33.08 0.84 _±_ 0.02 27.10 _±_ 3.45
16 1141.20 _±_ 56.99 3.91 _±_ 0.01 11.43 _±_ 3.63 0.92 _±_ 0.01 60.40 _±_ 1.08

4 4 87.80 _±_ 22.86 3.63 _±_ 0.09 153.52 _±_ 121.53 0.81 _±_ 0.03 14.40 _±_ 2.22

8 497.20 _±_ 148.28 3.85 _±_ 0.03 109.09 _±_ 28.45 0.86 _±_ 0.03 37.00 _±_ 5.72
16 2128.40 _±_ 287.64 3.94 _±_ 0.01 28.05 _±_ 28.37 0.94 _±_ 0.01 80.90 _±_ 3.38

3 1 4 15.00 _±_ 0.00 3.73 _±_ 0.00 0.49 _±_ 0.65 0.07 _±_ 0.00 5.00 _±_ 0.00

8 93.00 _±_ 0.00 4.99 _±_ 0.00 3.17 _±_ 4.20 0.38 _±_ 0.00 10.10 _±_ 0.42
16 696.80 _±_ 0.45 5.55 _±_ 0.00 3.08 _±_ 2.35 0.65 _±_ 0.00 21.80 _±_ 0.27

2 4 62.00 _±_ 9.77 4.68 _±_ 0.17 5.07 _±_ 5.18 0.45 _±_ 0.05 8.80 _±_ 0.45
8 710.00 _±_ 141.76 5.56 _±_ 0.06 20.26 _±_ 16.75 0.68 _±_ 0.03 20.20 _±_ 1.68
16 5628.40 _±_ 482.33 5.80 _±_ 0.01 13.79 _±_ 4.92 0.82 _±_ 0.01 41.80 _±_ 1.35
3 4 118.80 _±_ 49.34 4.91 _±_ 0.25 60.28 _±_ 67.19 0.60 _±_ 0.07 11.00 _±_ 1.17
8 1806.80 _±_ 572.05 5.70 _±_ 0.08 56.18 _±_ 21.47 0.76 _±_ 0.07 26.90 _±_ 2.46
16 2.00 _×_ 10 [4] _±_ 2871.17 5.88 _±_ 0.01 30.68 _±_ 12.59 0.89 _±_ 0.01 59.00 _±_ 3.30
4 4 291.40 _±_ 145.44 5.13 _±_ 0.33 314.42 _±_ 189.10 0.66 _±_ 0.18 14.90 _±_ 3.11
8 4073.40 _±_ 1620.28 5.76 _±_ 0.06 204.91 _±_ 95.20 0.82 _±_ 0.06 34.20 _±_ 5.75
16 5.49 _×_ 10 [4] _±_ 4259.65 5.91 _±_ 0.02 91.56 _±_ 49.71 0.91 _±_ 0.02 81.70 _±_ 6.45
4 1 4 16.00 _±_ 0.00 4.00 _±_ 0.00 0.00 _±_ 0.00 0.00 _±_ 0.00 5.50 _±_ 0.00
8 163.00 _±_ 0.00 6.28 _±_ 0.00 4.78 _±_ 9.59 0.21 _±_ 0.00 10.60 _±_ 0.42
16 2517.00 _±_ 0.00 7.32 _±_ 0.00 4.33 _±_ 2.61 0.54 _±_ 0.00 22.50 _±_ 0.35
2 4 72.60 _±_ 22.70 5.21 _±_ 0.31 16.85 _±_ 37.67 0.33 _±_ 0.18 8.70 _±_ 0.76
8 2244.80 _±_ 630.08 7.25 _±_ 0.18 36.18 _±_ 21.01 0.56 _±_ 0.11 20.40 _±_ 0.74
16 4.22 _×_ 10 [4] _±_ 8608.12 7.72 _±_ 0.06 13.07 _±_ 4.95 0.78 _±_ 0.05 41.10 _±_ 0.65
3 4 227.60 _±_ 42.21 5.85 _±_ 0.16 31.59 _±_ 36.89 0.63 _±_ 0.06 12.50 _±_ 0.61
8 9340.80 _±_ 3325.81 7.47 _±_ 0.17 143.79 _±_ 68.60 0.69 _±_ 0.08 27.60 _±_ 2.25
16 2.23 _×_ 10 [5] 7.21 _×_ 10 [4] 7.82 _±_ 0.04 49.91 _±_ 9.37 0.85 _±_ 0.04 57.70 _±_ 2.46
4 4 448.00 _±_ 119.14 6.17 _±_ 0.12 62.72 _±_ 41.18 0.72 _±_ 0.07 15.90 _±_ 1.19
8 3.58 _×_ 10 [4] _±_ 9493.85 7.70 _±_ 0.06 233.75 _±_ 72.31 0.85 _±_ 0.02 37.40 _±_ 1.29
16 6.24 _×_ 10 [5] 9.63 _×_ 10 [4] 7.85 _±_ 0.03 107.33 _±_ 20.93 0.86 _±_ 0.02 76.35 _±_ 4.56
5 1 4 16.00 _±_ 0.00 4.00 _±_ 0.00 0.00 _±_ 0.00 0.00 _±_ 0.00 5.50 _±_ 0.00
8 219.00 _±_ 0.00 7.23 _±_ 0.00 6.80 _±_ 19.51 0.10 _±_ 0.00 10.75 _±_ 0.26
16 6884.87 _±_ 0.35 9.02 _±_ 0.00 4.35 _±_ 2.69 0.44 _±_ 0.00 23.17 _±_ 0.41
2 4 89.50 _±_ 19.78 5.34 _±_ 0.25 0.00 _±_ 0.01 0.31 _±_ 0.11 9.30 _±_ 0.63
8 5802.60 _±_ 1146.10 8.77 _±_ 0.27 62.26 _±_ 32.43 0.45 _±_ 0.11 21.75 _±_ 1.06
16 2.69 _×_ 10 [5] 4.87 _×_ 10 [4] 9.61 _±_ 0.05 21.69 _±_ 10.54 0.71 _±_ 0.04 42.57 _±_ 1.53
3 4 389.60 _±_ 188.02 5.65 _±_ 0.41 2.11 _±_ 4.30 0.55 _±_ 0.20 14.65 _±_ 2.32
8 3.66 _×_ 10 [4] 1.21 _×_ 10 [4] 8.54 _±_ 0.93 156.80 _±_ 75.65 0.68 _±_ 0.12 31.50 _±_ 2.33
16 1.78 _×_ 10 [6] 1.94 _×_ 10 [5] 9.78 _±_ 0.04 39.91 _±_ 17.83 0.82 _±_ 0.04 57.44 _±_ 1.47
4 4 1206.70 _±_ 1154.00 5.50 _±_ 0.78 1.38 _±_ 1.78 0.77 _±_ 0.16 18.60 _±_ 3.98
8 1.82 _×_ 10 [5] 7.98 _×_ 10 [4] 8.25 _±_ 1.38 192.28 _±_ 100.15 0.84 _±_ 0.07 48.35 _±_ 12.25
16 5.03 _×_ 10 [6] 1.07 _×_ 10 [6] 9.80 _±_ 0.03 152.54 _±_ 58.24 0.84 _±_ 0.03 70.88 _±_ 1.19


0.27 _±_ 0.00
0.57 _±_ 0.00
0.77 _±_ 0.00
0.57 _±_ 0.05
0.78 _±_ 0.03
0.88 _±_ 0.01
0.67 _±_ 0.07
0.84 _±_ 0.02
0.92 _±_ 0.01
0.81 _±_ 0.03
0.86 _±_ 0.03
0.94 _±_ 0.01
0.07 _±_ 0.00
0.38 _±_ 0.00
0.65 _±_ 0.00


0.63 _±_ 0.11
1.45 _±_ 0.99
3.34 _±_ 4.48
33.75 _±_ 67.35
13.19 _±_ 11.50
4.81 _±_ 3.16
116.74 _±_ 147.94
45.21 _±_ 33.08
11.43 _±_ 3.63
153.52 _±_ 121.53
109.09 _±_ 28.45
28.05 _±_ 28.37
0.49 _±_ 0.65
3.17 _±_ 4.20
3.08 _±_ 2.35


2.91 _±_ 0.00
3.46 _±_ 0.00
3.74 _±_ 0.00
3.37 _±_ 0.09
3.75 _±_ 0.03
3.87 _±_ 0.01
3.52 _±_ 0.10
3.82 _±_ 0.02
3.91 _±_ 0.01
3.63 _±_ 0.09
3.85 _±_ 0.03
3.94 _±_ 0.01
3.73 _±_ 0.00
4.99 _±_ 0.00
5.55 _±_ 0.00


11.00 _±_ 0.00
37.00 _±_ 0.00
137.00 _±_ 0.00
30.80 _±_ 4.15
135.20 _±_ 9.88
541.20 _±_ 26.81
58.20 _±_ 14.58
275.20 _±_ 51.57
1141.20 _±_ 56.99
87.80 _±_ 22.86
497.20 _±_ 148.28
2128.40 _±_ 287.64
15.00 _±_ 0.00
93.00 _±_ 0.00
696.80 _±_ 0.45


2


1


2


3


4


4
8
16
4
8
16
4
8
16
4
8
16
4
8
16


3


1


26


(b) Inradius


10k


1000


100


(a) Volume


Average
Volume


10k


1000


Average
Inradius


Number
of Points


10k
1000
100
10
1


10k
1000
100
10
1


10k
1000
100
10
1


25


3125


625


125


25


10k
1000
100
10
1

|10<br>1<br>10k<br>1000<br>100<br>11<br>10<br>6 8<br>Histograms<br>mplex, colore<br>No Trainin|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|1111<br>3<br>2<br>2<br>1<br>1111<br>1<br>5<br>16 18 20<br>Numb<br>ed polyhedr<br>ng to volume<br>1 Epoch|Col13|Col14|Col15|Col16|Col17|Col18|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|
|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|||||||||||
|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|||||||||||
|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|||||||||||
|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|~~1~~<br>~~11~~<br><br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br> Histograms<br> mplex, colore<br>No Trainin|<br><br>6<br>8 <br> ms<br> lore<br> inin|<br><br>6<br>8 <br> ms<br> lore<br> inin|<br><br> <br>2 14 <br>_und_<br>  ordi|<br><br> <br> 16 <br>_ed_<br>  ng t<br>1 E|<br><br> <br>18 2<br>pol<br>  o v<br> poc|<br><br> <br>18 2<br>pol<br>  o v<br> poc|<br><br> <br>18 2<br>pol<br>  o v<br> poc|<br><br> <br>18 2<br>pol<br>  o v<br> poc|<br><br> <br>18 2<br>pol<br>  o v<br> poc|<br><br> <br>18 2<br>pol<br>  o v<br> poc|
|||||||||||~~Il~~|~~1~~<br>|~~1~~<br> ~~1~~|||~~**1**~~<br>|~~**1**~~<br>|~~**1**~~<br>|
|||||4||E|po|chs|chs|<br>|<br>5 E|<br> poc|hs|||||
|||||||||||~~**1**~~|~~**1**~~|||||||
|<br>|<br>|<br>|<br> <br>|<br> <br>8|<br> <br>|<br> <br> E|<br> po|<br> chs|<br> chs|<br> <br>|<br> <br>9 E|<br> <br> poc|<br> <br> hs|<br> <br>|<br>|<br>|<br>|
|||||||||||~~ 1~~|||~~1~~|||||
||||<br>1|<br>|<br>2|<br>|Ep|<br> ochs|<br> ochs|<br>1|<br>3 E|<br> po|<br> ch|<br> s||||
|||||||||||||||||||
|||||||||||||||||||

5 10 15


|100<br>1111.<br>10<br>10k<br>1000<br>100<br>11.1 1<br>10<br>6 8<br>of Neighbors<br>neighbor c<br>left) and inr<br>2 Epoc|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|0.6<br>oints<br>11.1. 11<br>0.5<br>0.4<br>0.3 No<br>Data<br>0.2<br>Points<br>1111<br>0.1<br>0<br>16 18 20<br>all d-cells in<br>ht).<br>3 Epochs|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|~~.~~<br>~~1~~|<br>~~1~~|~~1~~||~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|~~1~~|
|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc||||||||||||||
|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc||||||||||||||
|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc||||||||||||||
|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|1~~11.~~<br>~~1~~<br>11~~. 11~~<br> <br> <br> <br>6<br>8 <br>10<br>100<br>1000<br>10k<br>10<br>100<br> of Neighbors<br> neighbor c<br>      left) and inr<br>2 Epoc|<br> <br> <br>6<br>8 <br>  bors<br> or c<br>       d inr<br> poc|<br> <br> <br> 10 <br> <br>  ou<br>        adi<br> hs|<br> <br> <br> 10 <br> <br>  ou<br>        adi<br> hs|<br> <br> <br>14 <br>   for<br>         rig|<br> <br> <br>16 1<br>    all<br>         ht).<br>3|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|<br> <br>8 2<br> _d_-<br> <br> Ep|
||||<br> <br>|<br> <br>|<br> <br>|~~.~~<br>~~I1~~|<br>~~11~~|<br>~~11~~|<br>~~ I~~|<br>~~ l~~||||~~,~~|~~1~~<br>~~1~~|~~1~~<br>~~1~~|~~1~~<br>~~1~~|~~1~~<br>~~1~~|
||||<br> <br> <br>|<br> <br> <br>6|<br> <br> <br> E|<br> <br> poc|<br> <br> hs|<br> <br> hs|<br>|<br> <br>7|<br> Ep|oc|h|s|||||
||||||~~**1**~~|~~**1**~~|||~~**1**~~|~~**1**~~|~~**1**~~||||||||
|<br> <br>|<br> <br>|<br> <br>|<br> <br> <br>1|<br> <br> <br>|<br> <br> <br>0 E|<br> <br> poc|<br> <br> hs|<br> <br> hs|<br> <br>|<br> <br> <br>11|<br> <br> Ep|<br> <br> oc|<br> <br>|<br> h|<br> s||<br>|<br>|
|<br>|<br>|<br>|<br>|<br>|<br>|~~1~~|~~ 1~~|~~ 1~~|~~ 1~~<br>~~1~~|<br>~~1~~|~~1~~||||||||
|<br>|<br>|<br>|<br> <br>1|<br> <br>|<br> <br>4 E|<br> poc|<br> hs|<br> hs|<br>|<br> <br>15|<br> Ep|<br> o|<br> c|<br> h|<br> s||||
||||||||||||||||||||
||||||||||||||||||||


5 10 15 5 10 15 5 10 15


5


1


1


Number of Neighbors


Figure 14: Face count distributions for a network trained on the MNIST dataset. These are constructed in the same way as in Fig. 6, but with one for each training Epoch.


27
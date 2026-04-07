# SOLVING THE 2-NORM K-HYPERPLANE CLUSTERING
## PROBLEM VIA MULTI-NORM FORMULATIONS

**Stefano Coniglio**
Department of Economics
University of Bergamo
Bergamo, Italy
stefano.coniglio@unibg.it


ABSTRACT


We propose a method to solve _k_ -HC2—the _k_ -Hyperplane Clustering problem that
asks to find _k_ hyperplanes that minimize the sum of squared 2-norm (Euclidean)
distances between each point and its closest hyperplane—to global optimality via
spatial branch-and-bound (SBB) techniques. Our method strengthens a mixed
integer quadratically-constrained quadratic programming formulation for _k_ -HC2
with constraints that arise when formulating the problem in _p_ -norms with _p_ = 2.
In particular, we show that, for every (suitably scaled) _p_ _∈_ N _∪{∞}_, one obtains a variant of _k_ -HC2 whose optimal solutions yield lower bounds within a
multiplicative approximation factor. We focus on the case of polyhedral norms
where _p_ = 1 _, ∞_ (which are disjunctive-programming representable), and prove
that strengthening the original formulation by including, on top of its 2-norm constraints, the constraints of one of the polyhedral norms leads to an SBB method
where nonzero lower bounds are obtained in a a number of nodes that is linear in
_n_ and _k_ (rather than exponential). Experimentally, our method leads to very large
speedups, reducing median solve times by up to 41× while increasing the total
number of solved instances by up to 63%, drastically improving the problem’s
solvability to global optimality.


1 INTRODUCTION


Given _m_ points _{a_ 1 _, . . ., am}_ in R _[n]_, the _k-Hyperplane_ _Clustering_ problem, or _k_ -HC2, asks for
identifying _k_ hyperplanes which minimize the sum of the squares of the distances between each point
and the hyperplane closest to it in Euclidean (2-norm) distance. _k_ -HC2 arises when relationships of
_co-linearity_ (in R [2] ) or _co-(hyper)planarity_ (in R _[n]_ ) are sought. One of the problem’s most natural
applications is line/surface detection in digitally-sampled images and in 3d environments Amaldi &
Mattavelli (2002). More applications are found in diverse areas such medical prognosis Bradely &
Mangasarian (2000), linear facility location Megiddo & Tamir (1982), discrete-time piecewise affine
hybrid system identification Ferrari-Trecate et al. (2003), piecewise-affine model fitting Amaldi et al.
(2016), principal/sparse component analysis Washizawa & Cichocki (2006); He & Cichocki (2007);
Tsakiris & Vidal (2017), nonlinear regression He & Qin (2010), dictionary learning Zhang et al.
(2013), LiDAR data classification Kong et al. (2013), and sparse matrix representation Georgiev
et al. (2007).


_k_ -HC2 was first introduced by Bradely & Mangasarian (2000), where it is shown that, with _k_ = 1,
the problem is solved by computing an eigenvalue-eigenvector pair of a suitably defined matrix
built as a function of the data points. _k_ -HC2 is _NP_ -hard in any norm since fitting _m_ points in
R _[n]_ with _k_ hyperplanes with 0 error is _NP_ -complete even for _n_ = 2 (Megiddo & Tamir, 1982).
To tackle _k_ -HC2 when _k_ _≥_ 2 without optimality guarantees, Bradely & Mangasarian (2000) proposed an adaptation of the popular _k-means_ heuristic by MacQueen (1967). An exact Mixed Integer
Quadratically Constrained Quadratic Programming (MI-QCQP) formulation is proposed by Amaldi
& Coniglio (2013), together with a heuristic for larger-scale instances. Works addressing variants of
_k_ -HC2 asking for the smallest number of hyperplanes with a distance no larger than a given _ϵ_ _>_ 0
are found in Dhyani & Liberti (2008); Amaldi et al. (2013).


1


**Contributions.** We propose a method to solve _k_ -HC2 to global optimality via a spatial branchand-bound (SBB) technique. We strengthen a classical mixed-integer quadratically-constrained
quadratic programming (MI-QCQP) formulation for _k_ -HC2 by including constraints (and variables)
that arise when formulating the problem in another _p_ -norm ( _p_ = 2). We show that, under mild
assumptions, the inclusion of constraints stemming from a version of _k_ -HC2 formulated in one of
the two polyhedral norms (where _p_ = 1 _, ∞_ ) leads to an SBB method where a nonzero global lower
bound is obtained in a linear number of SBB nodes, as opposed to the exponential number that is
necessary when the classical formulation is used. Our experiments reveal that our method leads
to very large speedups, reducing median solve times by up to 41× while increasing the total number of solved instances by up to 63%, substantially improving the problem’s solvability to global
optimality.


2 PRELIMINARIES


Given a point _a_ _∈_ R _[n]_, its _p_ -norm with _p_ _∈_ N _∪{∞}_ is _∥a∥p_ := lim _q→p_ ( [�] _[n]_ _h_ =1 _[|][a][h][|][q]_ [)][1] _[/q]_ [.] [In]

particular, for _p_ = 1 _,_ 2, and _∞_ we have _∥a∥_ 1 = [�] _h_ _[n]_ =1 _[|][a][h][|]_ [,] _[∥][a][∥]_ [2] [:=] �� _nh_ =1 _[|][a][h][|]_ [2][�][1] _[/]_ [2][,] [and]
_∥a∥∞_ = max _h∈_ [ _n_ ] - _|ah|_ �. [1] The _p_ -norm point-to-hyperplane distance _dp_ ( _a, H_ ) between a point
_a_ _∈_ R _[n]_ and a hyperplane _H_ := _{x_ _∈_ R _[n]_ : _x_ _[⊤]_ _w_ = _γ}_ of parameters ( _w, γ_ ) _∈_ R _[n]_ [+1] is defined
as the _p_ -norm distance between _a_ and the point _y_ _∈_ _H_ that is closest to it. Namely, _dp_ ( _a, H_ ) :=
min _y∈H ∥a −_ _y∥p_ . Different arguments, including Lagrangian duality—see Mangasarian (1999),


can be used to show that _dp_ ( _a, H_ ) = _[|][w][⊤][a][−][γ][|]_


[1] [1]

_p_ [+] _p_


can be used to show that _dp_ ( _a, H_ ) = _[|][w]_ _∥w_ _[a]_ _∥_ _[−]_ _p′_ _[γ][|]_ [,] [where] _[p]_ [and] _[p][′]_ [satisfy] _p_ [1] [+] _p_ [1] _[′]_ [=] [1][.][2] [For] _[p]_ [=] [2][,]

_dp_ ( _a, H_ ) is called _Euclidean_ _point-to-hyperplane_ (or _orthogonal_ ) _distance_ . In many applications,
such a distance is preferred as it leads to solutions that are invariant to rotations of the data points.


In spite of being defined on top of a _p_ -norm, the distance function _dp_ is intrinsically nonconvex w.r.t.
_w_ regardless of the choice of _p_ (the proof is in the appendix):

**Proposition 1.** _Given a hyperplane H_ := _{x_ _∈_ R _[n]_ : _x_ _[⊤]_ _w_ = _γ} and a point a_ _∈_ R _[n]_ _, the function_
_dp_ ( _a, H_ ) = _[|][w][⊤][a][−][γ][|]_ _[, where]_ [1] [+] [1] _[′]_ [= 1] _[, is a nonconvex function of]_ [ (] _[w, γ]_ [)] _[ for every][ p][ ∈]_ [N] _[ ∪{∞}][.]_


_∥w_ _[a]_ _∥_ _[−]_ _p′_ _[γ][|]_ [,] [where] _[p]_ [and] _[p][′]_ [satisfy] _p_ [1]


_[a][−][γ][|]_ [1]

_∥w∥p′_ _[, where]_ _p_


[1] [1]

_p_ [+] _p_


_p_ _[′]_ [= 1] _[, is a nonconvex function of]_ [ (] _[w, γ]_ [)] _[ for every][ p][ ∈]_ [N] _[ ∪{∞}][.]_


This makes _k_ -HC2 substantially harder than classical machine learning problems where a norm is
minimized, and motivates the adoption of SBB techniques for solving it to global optimality.


3 APPROXIMATING _k_ -HC2 USING DIFFERENT NORMS


Given _m_ points _{a_ 1 _, . . ., am}_ in R _[n]_, the most compact nonlinear programming (NLP) formulation
for _k_ -HC2 reads: [3]


_,_


- [�]


( _k_ -HC2) min
( _w,γ_ )


- _m_

 


min
_j∈_ [ _k_ ]
_i_ =1


- ( _a⊤i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2]

_∥wj∥_ [2] 2


where ( _wj, γj_ ) _∈_ R _[n]_ [+1], _j_ _∈_ [ _k_ ], are the hyperplane parameters. ( _k_ -HC2) has a non-smooth objective function due to Proposition 1 and, since _∥wj∥_ [2] 2 [=] _[w]_ _j_ _[⊤][w][j]_ [,] [it] [features] [ratios] [of] [quadratics.]
While the inner min operator can be easily dropped by introducing binary assignment variables (see
below), this formulation is unsuitable for most nonlinear programming solvers as the denominator
vanishes when _wj_ = 0.


In the remainder of the paper, we will study _k_ -HC( _p,c_ ), a generalized version of _k_ -HC2 which
employs a _p_ norm not necessarily equal to 2 and which is parametric in a constant _c_ _≥_ 0. Its
NLP formulation, where [1] [+] [1] _[′]_ [= 1][, reads:]


[1] [1]

_p_ [+] _p_


_p_ _[′]_ [= 1][, reads:]


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1


- _m_

 


          
�( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _∥wj∥_ [2] _p_ _[′]_ _[≥]_ _[c, j]_ _[∈]_ [[] _[k]_ []]


_,_


( _k_ -HC( _p,c_ )) min
( _w,γ_ )


1Throughout the paper, we adopt the notation [ _ξ_ ] := 1 _, . . ., ξ_ for every _ξ_ _∈_ N.
2Two norms where 1 [+] [1] _[′]_ [= 1][ are called] _[ dual]_ [.]


1 [1]

_p_ [+] _p_


Two norms where _p_ [+] _p_ _[′]_ [= 1][ are called] _[ dual]_ [.] [The 2-norm is self dual and the 1 and] _[ ∞]_ [-norms are dual.]

3We report mathematical programming formulations in brackets and optimization problems without them.


2


Letting, for a problem _P_, OPT( _P_ ) be its optimal solution value, the validity of ( _k_ -HC( _p,c_ )) and the
role that _c_ plays in it are shown by the following lemma (the proof is in the appendix):

**Lemma 1.** _The solutions to_ ( _k_ -HC(2 _,_ 1)) _and_ ( _k_ -HC2) _coincide._ _Also,_ ( _k_ -HC( _p,c_ )) _is quadratically_
_homogeneous w.r.t._ _c, i.e.,_ OPT( _k_ -HC( _p,c_ )) = _c_ [2] OPT( _k_ -HC( _p,_ 1)) _._


The property shown by the lemma will be useful to guide our choice of which _p_ we should use to
introduce additional norm constraints to the formulation of _k_ -HC2 (which, we recall, is the version
of the problem that we aim to solve in this paper) in order to strengthen it.


**Rationale.** Investigating _k_ -HC( _p,c_ ) with ( _p, c_ ) _̸_ = (2 _,_ 1) is of interest for two reasons. First, as shown
in this section, doing so allows us to show that, for a suitable choice of _p_ and _c_, the optimal solutions
to _k_ -HC( _p,c_ ) are approximate solutions (to within an approximation factor) of those to _k_ -HC(2 _,_ 1).
Second, as shown in the next two sections, the study of _k_ -HC( _p,c_ ) allows us to prove that, again for a
suitable choice of _p_ and _c_, the formulations ( _k_ -HC( _p,c_ )) and ( _k_ -HC(2 _,_ 1)) can be intersected to obtain
a _strengthened formulation_ which is valid for _k_ -HC2 and which is also much easier to solve both in
theory and practice.


**Novelty.** While changes of norm are frequent in the ML literature, the dual norm in the denominator of the point-to-hyperplane distance requires, for our results, switching between primal and
dual norms and applying suitable scaling factors to the problem’s constraints in a way that, to our
knowledge, is new. The idea of _intersecting_ formulations derived for different norms, which leads to
a provably tighter approximation factor, is also, to our knowledge, uncommon in the literature. We
also manage to establish lower bounds on the number of branching operations needed to compute
a nonzero lower bound (after which pruning becomes possible), a type of result which is extremely
rare in integer programming (let alone nonlinear integer programming).


3.1 THE GENERAL CASE


We show that, whichever version of _k_ -HC( _p,c_ ) one aims to solve (be it the 2-norm one with _c_ = 1 or
another one), the optimal-solution value of _k_ -HC( _q,c′_ ) for _any_ choice of _q_ and a suitable _c_ _[′]_ is within
an approximation factor of the optimal-solution value of _k_ -HC( _p,c_ ):


**Theorem** **1.** _Let_ _p, q_ _∈_ N _∪{∞}_ _and_ _c_ _>_ 0 _._ _The_ _three_ _positive_ _scalars_ _α_ ( _p, q_ ) _, β_ ( _p, q_ ) _, δ_ ( _p, q_ )
_which,_ _for_ _all_ _x_ _∈_ R _[n]_ _,_ _satisfy_ _the_ _congruence_ _inequality_ _α_ ( _p, q_ ) _||x||p_ _≤_ _β_ ( _p, q_ ) _||x||q_ _≤_
_δ_ ( _p, q_ ) _||x||p for p, q_ _∈_ N _∪{∞} also satisfy the optimal-value inequality_ _[α]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)][2][2] [OPT(] _[k]_ [-HC][(] _[p,c]_ [)][)] _[ ≤]_


  OPT _k_ -HC _β_ ( _p,q_ )
( _q,c_ _δ_ ( _p,q_ ) [)]


_≤_ OPT( _k_ -HC( _p,c_ )) _._


Theorem 1 shows that the optimal solution value of _k_ -HC( _q,c′_ ) with _c_ _[′]_ = _c_ _[β]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)] [is a lower bound]


on the optimal solution value of _k_ -HC( _p,c_ ) to within an approximation factor of _[α]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)][2][2] [.] [This] [is]

crucial, as it shows which value to pick for _c_ _[′]_ for _any q_ -norm we may choose to obtain a relaxation
of _k_ -HC( _p,c_ ) and, in particular, one of _k_ -HC(2 _,_ 1) (which is, ultimately, the problem we aim to solve).


We remark that Theorem 1 can be extended to produce an approximation of _k_ -HC( _p,c_ ) from above
to within an approximation factor—we omit the details since, here, we solely are interested in approximations from below to build tighter relaxations suitable for an SBB method.


Theorem 1 has a nice geometrical interpretation in terms of the feasible regions of ( _k_ -HC( _p,c_ )) and
( _k_ -HC( _q,c_ _βδ_ (( _p,qp,q_ )) [)][)][.] [Indeed, with] _[ c][′]_ [=] _[c]_ _[β]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)] [, the feasible region of the] _[ q]_ [-norm constraints featured]

in _k_ -HC( _q,c′_ ) is a relaxation of the region that is feasible for the _p_ -norm constraints of _k_ -HC( _p,c_ ).
An illustration is reported in Figure 1 for _p_ = 2 _, c_ = 1 and adopting _q_ = 1 (left) and _q_ = _∞_ (right),
for which we have, respectively, _c_ _[′]_ = 1 and _c_ _[′]_ = ~~_√_~~ 1 .
_n_


3.2 THE CASE OF POLYHEDRAL NORMS WITH _q_ = 1 _, ∞_


We now focus on _polyhedral_ norms ( _q_ = 1 _, ∞_ ). These are of computational interest due to their
tractability: while the constraints _∥wj∥q_ _≥_ _c_ _[′]_, _j_ _∈_ [ _k_ ], with _q_ = 1 _, ∞_, are nonconvex, they can be
stated as disjunctions over polyhedra, thus being mixed-integer-linear-programming representable.


3


_βδ_ (( _p,qp,q_ )) [)][)][.] [Indeed, with] _[ c][′]_ [=] _[c]_ _[β]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)]


_δ_ ( _p,q_ ) [, the feasible region of the] _[ q]_ [-norm constraints featured]


_w_ 2


_w_ 2


_w_ 1


_w_ 1


Figure 1: Complements of the feasible regions of _{w_ _∈_ R [2] : _||w||_ 1 _≥_ 1 _}_ and _{w_ _∈_ R [2] : _||w||∞_ _≥_
~~_√_~~ 1

2 _[}]_ [.]


In light of this, we consider the following two relaxations of _k_ -HC(2 _,_ 1) (see again Figure 1 for an
illustration of the projection of the feasible regions of these two problems onto the _w_ space for
_k_ = 1):


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1


- _m_

 


( _k_ -HC( _∞,_ 1)) min
( _w,γ_ )


( _k_ -HC(1 _,_ ~~_√_~~ 1 _n_ ) [)] (min _w,γ_ )


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1


- _m_

 


�( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _∥wj∥_ 1 _≥_ 1 _, j_ _∈_ [ _k_ ]


�( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_ _, j_ _∈_ [ _k_ ]


_,_


_._


Notice that, due to norm duality, ( _k_ -HC( _∞,_ 1)) features 1-norm constraints while ( _k_ -HC(1 _,_ ~~_√_~~ 1 _n_ ) [) fea-]
tures _∞_ -norm ones. For these two problems, Theorem 1 leads to the following result (the proof is
in the appendix):
**Corollary 1.** _k_ -HC( _∞,_ 1) _and k_ -HC(1 _,_ ~~_√_~~ 1 _n_ ) _[satisfy:]_


1
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(] _[∞][,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)]

1
~~_√_~~ 1 (2 _,_ 1) [)] _[.]_
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(1] _[,]_ _n_ ) [)] _[ ≤]_ [OPT(] _[k]_ [-HC]


With the first chain of inequalities, the corollary shows that solving _k_ -HC( _∞,_ 1), i.e., formulating _k_ HC with the constraint _||wj||_ 1 _≥_ 1 for all _j_ _∈_ [ _k_ ], leads to a relaxation to within a _n_ [1] [approximation]

factor. With the second one, the corollary shows that solving _k_ -HC(1 _,_ ~~_√_~~ 1 _n_ ) [, i.e., solving the version]

of _k_ -HC with the constraint _||wj||∞_ _≥_ ~~_√_~~ 1 _n_ for all _j_ _∈_ [ _k_ ], leads to another relaxation also to within
the same approximation factor _n_ [1] [.]


3.3 MULTI-NORM RELAXATION


Since both _∥wj∥_ 1 _≥_ 1, _j_ _∈_ [ _k_ ], and _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], are relaxations of _∥wj∥_ 2 _≥_ 1, _j_ _∈_ [ _k_ ],
a strengthened relaxation of _k_ -HC(2 _,_ 1) can be obtained by simultaneously imposing both. Such a
_multi-norm_ relaxation, which we refer to as _k_ -HC(multi _,_ 1), reads


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1


( _k_ -HC(multi _,_ 1)) min
( _w,γ_ )


- _[≥]_ [1] _[,]_ _j_ _∈_ [ _k_ ]
( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _[∥]_ _∥_ _[w]_ _w_ _[j]_ _j_ _[∥]_ _∥_ [1] _∞_ _≥_ ~~_√_~~ 1 _n_ _,_ _j_ _∈_ [ _k_ ]


_._


Letting _||w||_ multi := min _{||w||_ 1 _,_ _[√]_ _n||w||∞}_, one can see that simultaneously imposing _∥wj∥_ 1 _≥_ 1
and _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], coincides with imposing _||wj||_ multi _≥_ 1 _, j_ _∈_ [ _k_ ]. A depiction of the
corresponding feasible region is reported in Figure 2.


So far, our analysis has hinged on the possibility of translating a _p_ _[′]_ -norm constraint into the corresponding _dp_ distance, on which we applied Theorem 1. Deriving an approximation factor for
_k_ -HC(multi _,_ 1) is not straightforward, though. This is because, while vector norms are convex and
convex functions have convex sublevel sets, the sub-level sets of the function _||w||_ multi are not convex and, thus, there is no _p_ -norm, _p ∈_ N _∪{∞}_, whose adoption directly leads to _k_ -HC(multi _,_ 1).


4


_w_ 2


_w_ 1


Figure 2: Complement of the feasible region of _{w_ _∈_ R [2] : _||w||_ multi _≥_ 1 _}_ .


In spite of this, in the following we show that we can still derive an approximation factor by constructing the norm that is implicitly minimized when min _{||w||_ 1 _,_ _[√]_ _n||w||∞} ≥_ 1 is imposed.


We start with the following lemma (the proof is in the appendix), which shows what combination of
point-to-hyperplane distances is minimized in _k_ -HC when imposing min _{||w||_ 1 _,_ _[√]_ _n||w||∞} ≥_ 1:

**Lemma 2.** _Solving k-HC subject to_ min _{||w||_ 1 _,_ _[√]_ _n||w||∞}_ _≥_ 1 _coincides with solving an uncon-_
_strained_ _version_ _of_ _k-HC_ _where_ _the_ _point-to-hyperplane distance between_ _ai_ _and Hj_ _is defined as_
max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}._


We now prove two new lemmas (the proofs are in the appendix) that show that the function
max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ is a norm and construct a congruence inequality for it:

**Lemma** **3.** _The_ _function_ max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}_ _is_ _a_ _distance_ _induced_ _by_ _the_ _norm_
max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}._

**Lemma 4.** _The norm_ max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _} satisfies the congruence inequality_


_n_ _[−]_ 4 [1] _∥x∥_ 2 _≤_ max� _∥x∥∞,_ ~~_√_~~ 1 _n_ _∥x∥_ 1         - _≤∥x∥_ 2 _._


Crucially, the following holds:


**Corollary 2.** _Combining Lemma 4 with Theorem 1, the multi-norm relaxation k_ -HC(multi _,_ 1) _satisfies_

~~_√_~~ 1 _n_ OPT� _k_ -HC(2 _,_ 1)� _≤_ OPT� _k_ -HC(multi _,_ 1)� _≤_ OPT� _k_ -HC(2 _,_ 1)� _._


4 SOLVING THE STRENGTHENED FORMULATIONS OF _k_ -HC(2 _,_ 1) VIA SBB


We now focus on solving _k_ -HC(2 _,_ 1) to global optimality via SBB. We analyze the number of SBB
nodes needed to compute a nonzero global lower bound when solving a basic formulation of the
problem, and then prove that intersecting the basic formulation for _k_ -HC(2 _,_ 1) with one of our relaxations involving polyhedral norms allows for computing a nonzero global lower bound much
earlier.


4.1 SPATIAL BRANCH-AND-BOUND


The basic idea of the spatial branch-and-bound (SBB) method is to build a dual bound by optimizing
over a convex (typically polyhedral) envelope conv( _F_ ) of the feasible region _F_ of the problem. _F_
is then split into two sub-regions _F_ 1 and _F_ 2 with tighter bounds on at least one variable. This allows
for constructing tighter convex envelopes of _F_ 1 and _F_ 2 in such a way that the optimal solution
over conv( _F_ ) is cut off because it does not belong to conv( _F_ 1) _∪_ conv( _F_ 2). _F_ 1 and _F_ 2 are then
recursively optimized in a classical _divide-et-impera_ (branch-and-bound) fashion within a binarytree search scheme.


Let us consider the case of _k_ -HC(2 _,_ 1). We assume (as done by most of the state-of-the-art solvers
such as Gurobi Gurobi Optimization, LLC (2026)), that polyhedral envelopes are employed. Under
such an assumption, when considering the nonlinear constraints _||wj||_ [2] 2 [=][ �] _h_ _[n]_ =1 _[w]_ _jh_ [2] _[≥]_ [1][,] _[ j]_ _[∈]_ [[] _[k]_ []][,]


5


a classical SBB method first introduces the auxiliary variable _zjh_ for each nonlinear term _wjh_ [2] [and]
a corresponding defining constraint _zjh_ = _wjh_ [2] [.] [It then substitutes the original nonlinear constraint]
with [�] _h_ _[n]_ =1 _[z][jh]_ _[≥]_ [1][.] [Each] [defining] [constraint] [is] [then] [relaxed] [into] [a] [polyhedral] [envelope.] [The]
point-wise minimal outer envelope of a bilinear product corresponds to the well-known McCormick
envelope McCormick (1976).


4.2 BASELINE MATHEMATICAL PROGRAMMING FORMULATION FOR _k_ -HC(2 _,_ 1)


We start by considering as baseline the following classical Mixed Integer Quadratically Constrained Quadratic Programming (MI-QCQP) formulation for _k_ -HC(2 _,_ 1) Coniglio (2011); Amaldi
& Coniglio (2013):


min
( _k_ -HC(2 _,_ 1)) ( _w,γ_ ) _,x,d_






_m_


- _d_ [2] _i_ [:]

_i_ =1


- _kj_ =1 _[x][ij]_ [= 1] _∀i ∈_ [ _m_ ]
_∥wj∥_ 2 _≥_ 1 _∀j_ _∈_ [ _k_ ]
_di_ _≥_ _wj_ _[T]_ _[a][i]_ _[−]_ _[γ][j]_ _[−]_ _[d][U]_ [(1] _[ −]_ _[x][ij]_ [)] _∀i ∈_ [ _m_ ] _, j_ _∈_ [ _k_ ]
_di_ _≥−wj_ _[T]_ _[a][i]_ [+] _[ γ][j]_ _[−]_ _[d][U]_ [(1] _[ −]_ _[x][ij]_ [)] _∀i ∈_ [ _m_ ] _, j_ _∈_ [ _k_ ]





_._






In it, _xij_ _∈{_ 0 _,_ 1 _}_ takes value 1 if and only if _ai_ is assigned to the hyperplane of index _j_ _∈_ [ _k_ ];
_di_ is the distance between _ai_ and the hyperplane of index _j_ _∈_ [ _k_ ]; _d_ _[U]_ is an upper bound on the
largest distance between any point _ai_ and hyperplane of index _j_ _∈_ [ _k_ ]. The only nonconvexity of
the formulation is due to the 2-norm constraints. W.l.o.g., we assume _ai_ _≥_ 0 for all _i ∈_ [ _m_ ] (as this
can be easily obtained in a preprocessing step by translating the dataset). The following bounds on
the variables can be included. We let _d_ _[U]_ := _∥b e∥_ 2, where _e_ is the all-one vector and _b_ is the length
of the edge of the smallest hypercube that contains _{a_ 1 _, . . ., am}_ . Since _∥wj∥_ 2 = 1 holds in any
optimal solution and max _{∥wj∥∞_ : _∥wj∥_ 2 = 1 _}_ = 1, we impose _∥wj∥∞_ _≤_ 1 via _−e_ _≤_ _wj_ _≤_ _e_,
_j_ _∈_ [ _k_ ]. These bounds imply _−nb −_ _d_ _[U]_ _≤_ _γj_ _≤_ _nb_ + _d_ _[U]_, _j_ _∈_ [ _k_ ].


Since the point-to-hyperplane distance is symmetric, given any solution to _k_ -HC(2 _,_ 1), an equivalent
one can be obtained by changing the sign of _wj_ for some _j_ _∈_ [ _k_ ]. To remove such a symmetry
(symmetries are known to be a hindrance when solving mathematical programming problems to
optimality via methods based on (spatial) branch-and-bound), we impose _wj_ to belong to an arbitrary
half-space of R _[n]_ for each _j_ _∈_ [ _k_ ] by imposing _wj_ 1 _≥_ 0 _, j_ _∈_ [ _k_ ], where _wj_ 1 is the first component
of _wj_ . In this way, any solution that is obtainable by changing the sign of a component of one of
the vectors _wj_ becomes infeasible (due to being obtained from the previous one by reflection of _wj_
over the hyperplane defining the halfspace that we selected), thus breaking the symmetry. In all our
formulations, we partially remove the symmetry on _xij_, _i_ _∈_ [ _m_ ] _, j_ _∈_ [ _k_ ], that is induced by the
assignment constraints by imposing _xij_ = 0 for all _i, j_ _∈_ [ _m_ ] _×_ [ _k_ ] with _i_ _<_ _j_ . This reduces the
number of 0-1 variables by [(] _[k][−]_ 2 [1)] _[k]_ .


4.3 SOLVING THE FORMULATION ( _k_ -HC(2 _,_ 1)) VIA SBB


Let us now analyze the behavior of a standard SBB method employed for solving the classical
formulation ( _k_ -HC(2 _,_ 1)). Since the projection onto the _w_ space of the feasible region of _k_ -HC(2 _,_ 1)
is nonconvex and its complement is symmetric about the origin, any SBB method based on convex
envelopes will necessarily convexify the infeasible region, thus making the trivial solution _wj_ =
0 _, j_ _∈_ [ _k_ ], feasible. This leads to a bound as weak as possible due to the fact that the objective
function is the sum of squares [�] _i_ _[m]_ =1 _[d]_ _i_ [2] _[≥]_ [0][ and, with][ (] _[w][j][, γ][j]_ [) = 0][,] _[ j]_ _[∈]_ [[] _[k]_ []][, we obtain][ �] _[m]_ _i_ =1 _[d]_ _i_ [2] [=]
0.


The following assumption holds in most SBB codes—see, e.g., Belotti et al. (2009):


**Assumption** **1.** _Assume_ _that,_ _when_ _spatially_ _branching_ _on_ _variables_ _with_ _a_ _symmetric_ _domain,_
_branching takes place on the midpoint of the domain._


Notice that, with the bounds we included, the domain of _wjh_, _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ], is symmetric.


Crucially, under Assumption 1 the geometry of the feasible region of _k_ -HC(2 _,_ 1) makes it so that
the number of branching operations that are needed to make the 0 solution infeasible (and, thus,
compute a nonzero global lower bound) is exponentially large (the proof is in the appendix):


6


**Proposition** **2.** _Under_ _Assumption_ _1,_ _when_ _solving_ _k_ -HC(2 _,_ 1) _a_ _nonzero_ _lower_ _bound_ _is_ _obtained_
_only after generating at least_ 2 _[k]_ [(] _[n][−]_ [1)] _branching nodes._


This is particularly bad since, until the first nonzero lower bound has been calculated, no pruning
can happen on the tree due to the fact that a lower bound of 0 trivially holds at any node since the
objective function is a sum of squares.


4.4 STRENGTHENED FORMULATIONS


We now construct valid formulations for _k_ -HC2 which are strengthened by featuring not only the
2-norm constraints but also a collection of polyhedral-norm constraints. Building on the relaxations
we constructed before, we introduce the following three strengthened formulations (in each of them,
the norm constraints are imposed for all _j_ _∈_ [ _k_ ]):


( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)) min
( _w,γ_ )


( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [)] (min _w,γ_ )


( _k_ -HC(2 _,_ 1) _,_ (multi _,_ 1)) min
( _w,γ_ )


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1





min
_j∈_ [ _k_ ]
_i_ =1


�( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _[∥]_ _∥_ _[w]_ _w_ _[j]_ _j_ _[∥]_ _∥_ [2] 1 _[≥]_ _≥_ [1] 1


- _m_

 - min

_j∈_ [ _k_ ]
_i_ =1





 _[.]_






_m_


- _[≥]_ [1]
( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _[∥]_ _∥_ _[w]_ _w_ _[j]_ _j_ _[∥]_ _∥_ [2] _∞_ _≥_ ~~_√_~~ 1 _n_


_∥wj∥_ 2 _≥_ 1
�( _a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][j]_ [)][2][�] : _∥wj∥_ 1 _≥_ 1
_∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_


Before analyzing the number of branching operations needed to achieve a nonzero lower bound
with these formulations, we report the Mixed Integer Linear Programming (MILP) formulations by
which we formulate the polyhedral-norm constraints.


**1-norm.** We formulate the constraints _∥wj∥_ 1 _≥_ 1, _j_ _∈_ [ _k_ ], via the following absolute-value reformulation:


_wjh_ [+] _[−]_ _[w]_ _jh_ _[−]_ [=] _[ w][jh]_ _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] (1a)

_wjh_ [+] _[≤]_ _[s][jh]_ _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] (1b)

_wjh_ _[−]_ _[≤]_ [(1] _[ −]_ _[s][jh]_ [)] _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] (1c)

_n_
�( _wjh_ [+] [+] _[ w]_ _jh_ _[−]_ [)] _[ ≥]_ [1] _j_ _∈_ [ _k_ ] (1d)

_h_ =1

0 _≤_ _wjh_ [+] _[, w]_ _jh_ _[−]_ _[≤]_ [1] _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] (1e)

_sjh_ _∈{_ 0 _,_ 1 _}_ _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] _._ (1f)


The binary variable _sjh_ denotes the sign of the _h_ -th component of _wj_ . Consider a component _wjh_ of
index _h_ of _wj_ . Due to Constraints (1a)–(1c), if _wjh_ _>_ 0, then _wjh_ [+] _[>]_ [ 0][ (with] _[ w]_ _jh_ [+] [=] _[ w][jh]_ [ and] _[ w]_ _jh_ _[−]_ [=]
0) and _sjh_ = 1. Otherwise, if _wjh_ _<_ 0, then _wjh_ _[−]_ _[>]_ [ 0][ (with] _[ w]_ _jh_ [+] [= 0][ and] _[ w]_ _jh_ _[−]_ [=] _[ −][w][jh]_ [) and] _[ s][jh]_ [= 0][.]
Since _wj_ [+] [and] _[ w]_ _j_ _[−]_ [are component-wise complementary thanks to Constraints (1b)–(1c), we deduce]
that _wj_ [+] [+] _[ w]_ _j_ _[−]_ [=] _[|][w][j][|]_ [ holds.] [Thus, Constraint (1d) guarantees] _[ ∥][w][j][∥]_ [1] _[≥]_ [1][.] [When these constraints]
are imposed, we break symmetry as mentioned before by imposing _wj_ 1 _≥_ 0, _j_ _∈_ [ _k_ ]. This leads to
_sj_ 1 = 1 and _wj_ _[−]_ 1 [= 0][, thanks to which Constraint (1d) becomes] _[ w][j]_ [1][ +][ �] _h_ _[n]_ =2 [(] _[w]_ _jh_ [+] [+] _[ w]_ _jh_ _[−]_ [)] _[ ≥]_ [1][.]

_∞_ **-norm.** We formulate the constraints _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], i.e., max _h∈_ [ _n_ ] _{|wjh|}_ _≥_ ~~_√_~~ 1 _n_,

_j_ _∈_ [ _k_ ], as the disjunction [�] _h_ _[n]_ =1 - _wjh_ _≤−_ ~~_√_~~ [1] _n_ _∨_ _wjh_ _≥_ ~~_√_~~ 1 _n_ - _, j_ _∈_ [ _k_ ]. Differently from the pre
vious cases, in this case we break symmetry by (w.l.o.g.) always selecting _wjh_ _≥_ ~~_√_~~ 1 _n_ from each
elementary disjunction _wjh_ _≤−_ ~~_√_~~ [1] _n_ _∨_ _wjh_ _≥_ ~~_√_~~ 1 _n_ . This translates into considering the restricted

disjunction [�] _h_ _[n]_ =1 - _wjh_ _≥_ ~~_√_~~ 1 _n_ �, _j_ _∈_ [ _k_ ]. For each _j_ _∈_ [ _k_ ], we restate the resulting disjunctive set


7


via the following MILP formulation:


         - 1
_wjh_ _≥−_ 1 + 1 + ~~_√_~~
_n_


_ujh_ _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] (2a)


_n_

   - _ujh_ = 1 _j_ _∈_ [ _k_ ] (2b)


_h_ =1

_ujh_ _∈{_ 0 _,_ 1 _}_ _j_ _∈_ [ _k_ ] _, h ∈_ [ _n_ ] _._ (2c)


Due to Constraint (2a), if _ujh_ = 1 holds for some _h_ _∈_ [ _n_ ], then _wjh_ _≥_ ~~_√_~~ 1 _n_ holds (the constraint is
inactive if _ujh_ = 0, and reads _wjh_ _≥−_ 1). Constraint (2b) imposes that exactly one component of
_uj_ = ( _uj_ 1 _, . . ., ujn_ ) is equal to 1.


When imposing multiple norm constraints at once, we only have to pay attention to the way symmetry is prevented, as the symmetry-breaking constraint _wj_ 1 _≥_ 0 we introduced for the constraints
_∥wj∥_ 2 _≥_ 1, _j_ _∈_ [ _k_ ], and _∥wj∥_ 1 _≥_ 1, _j_ _∈_ [ _k_ ], is not compatible with the one-sided disjunction we
considered for _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], and imposing both would lead to an over-restriction. Whenever the _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_ constraints are imposed, we resolve the issue by dropping the symmetrybreaking constraints _wj_ 1 _≥_ 0, _j_ _∈_ [ _k_ ].


4.5 SOLVING THE STRENGTHENED FORMULATIONS VIA SBB


We extend the analysis in Proposition 2 to the strengthened formulations with the following two
propositions (the proofs of both are contained in the appendix):


**Proposition 3.** _Assume that the constraint ∥wj∥_ 1 _≥_ 1 _, j_ _∈_ [ _k_ ] _, is imposed and that branching takes_
_place on the sjh_ _variables first._ _Then, a nonzero global lower bound is obtained after generating at_
_least_ 2 _[k]_ [(] _[n][−]_ [1)] _nodes._ _If k_ -HC( _∞,_ 1) _is being solved, no further branching on w takes place._

**Proposition 4.** _Assume that ∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_ _, j_ _∈_ [ _k_ ] _, is imposed and that branching takes place on_
_the ujh variables first._ _Then, k_ ( _n_ _−_ 1) _nodes suffice to obtain a nonzero lower bound._ _If k_ -HC(1 _,_ ~~_√_~~ 1 _n_ )
_is being solved, no further branching on w takes place._


Propositions 3 and 4 show the crucial advantages of strengthening formulation ( _k_ -HC(2 _,_ 1)) as we
proposed via the two (scaled) polyhedral-norm constraints we considered. Proposition 3 indicates
that, if the _||wj||_ 1 _≥_ 1 _, j_ _∈_ [ _k_ ], constraints are imposed and branching takes place on the 0-1
variables of such norm constraints, in a complete SBB tree of depth Θ(2 _[k]_ [(] _[n][−]_ [1)] ) the polyhedralnorm constraint is satisfied in _every_ leaf node. This is in stark contrast to the 2-norm case, where the
same number of branching operations only suffices to obtain the first nonzero global lower bound,
and the number of branchings needed to completely describe the feasible region of the problem in
the _w_ space depends on the solver’s feasibility tolerance (since, for each _j_ _∈_ [ _k_ ], the complement of
the feasible region is a sphere).

Crucially, Proposition 4 shows that, when the _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], constraints are imposed
and branching takes place on their 0-1 variables, the size of the SBB tree is extremely small—only
polynomial in _k_ and _n_ . The difference between the two results is due to the geometry of the 1- and
_∞_ -norm balls with _n >_ 2, since the former has 2 _[n]_ facets while the latter only 2 _n_ .


When included in a formulation for _k_ -HC2 on top of the constraints _||wj||_ 2 _≥_ 1 _, j_ _∈_ [ _k_ ], the
polyhedral-norm constraints substantially accelerate the computation of a nonzero global lower
bound, leading to more pruning and, overall, to a faster SBB method. This is better shown in
the next section.


5 COMPUTATIONAL RESULTS


We assess the effectiveness of our strengthened formulations with Gurobi 10’s SBB using 12 threads
on a 2.6GHz Intel Core i7-9750H equipped with 32 GB RAM, with a total time limit across the 12
cores of 168,000 seconds (46 hours).


8


We consider two testbeds: Low-dim and High-dim. Low-dim contains 43 instances with
_m_ = 10 _, . . .,_ 30, _n_ = 2 _,_ 3, and _k_ = 2 _,_ 3. These instances are a superset of the 24 instances
tackled with SBB techniques in Amaldi & Coniglio (2013). High-dim contains 43 instances with
_m_ = 10 _, . . .,_ 17, _n_ = 2 _,_ 3 _,_ 4 _,_ 5, and _k_ = 2 _,_ 3 _,_ 4 _,_ 5. Both datasets are generated by randomly choosing ( _wj, γj_ ), _j_ _∈_ [ _k_ ], with a uniform distribution in [ _−_ 1 _,_ 1] and distributing uniformly at random
the _m_ points such that each of them belongs (with 0 distance) to a hyperplane. Then, an orthogonal deviation from the corresponding hyperplane is added to each point by sampling a Gaussian
distribution with 0 mean and a variance that is selected, for each hyperplane, uniformly at random
in [0 _._ 7 _·_ 0 _._ 003 _,_ 0 _._ 003]. Details on how to access and run our code as well as on how to access the
dataset we used in the experiment are reported in the appendix.


We consider four formulations: ( _k_ -HC(2 _,_ 1)), ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [),] ( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)), and

( _k_ -HC(2 _,_ 1) _,_ (multi _,_ 1)). Tables 1 and 2 report, for each formulation, the median computing time on the
subset of instances solved by all four, the median speed-up relative to ( _k_ -HC(2 _,_ 1)), and the Holmcorrected (with a family-wise error rate _α_ = 0 _._ 05) _p_ -value of a two-sided Wilcoxon signed-rank test
against ( _k_ -HC(2 _,_ 1)).


Table 1: LowDim: comparison to ( _k_ -HC(2 _,_ 1))
on the 20 instances solved by all four formulations.


**Algorithm** **Median (s)** **Speed-up** _p_ **-value**


( _k_ -HC(2 _,_ 1)) 169.9 1 _×_  ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [)] 4.15 40.9 _×_ 1 _._ 1 _×_ 10 _[−]_ [4]

( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)) 6.10 27.9 _×_ 1 _._ 1 _×_ 10 _[−]_ [4]

( _k_ -HC(multi _,_ 1)) 5.00 34.0 _×_ 1 _._ 1 _×_ 10 _[−]_ [4]


Table 2: HighDim: comparison to ( _k_ -HC(2 _,_ 1))
on the 30 instances solved by all four formulations.


**Algorithm** **Median (s)** **Speed-up** _p_ **-value**


( _k_ -HC(2 _,_ 1)) 208.6 1 _×_  ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [)] 18.20 11.5 _×_ 5 _._ 6 _×_ 10 _[−]_ [9]

( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)) 20.65 10.1 _×_ 7 _._ 5 _×_ 10 _[−]_ [9]

( _k_ -HC(multi _,_ 1)) 37.35 5.6 _×_ 8 _._ 7 _×_ 10 _[−]_ [4]


Detailed results are reported in Tables 3 and 4. Let us focus first on the Low-dim testbed. With the
three strengthened formulations ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [),] [(] _[k]_ [-HC] (2 _,_ 1) _,_ ( _∞,_ 1) [),] [and] [(] _[k]_ [-HC] (2 _,_ 1) _,_ (multi _,_ 1) [)][,] [21]

instances that are not solved in over 46 hours with the classical formulation ( _k_ -HC(2 _,_ 1)) are solved
in under 2 hours. With the strengthened formulations, the 20 instances that are also solved with the
classical formulation are solved, respectively, 41, 28, and 34 times faster. Incidentally, our results
on the Low-dim testbed prove that all the heuristic solutions found in Amaldi & Coniglio (2013)
on the 24 instances considered in that work (those with _m_ = 10 _,_ 14 _,_ 18 _,_ 22 _,_ 26 _,_ 30) are optimal.


Let us turn now to the High-dim testbed. On it, with the best-performing strengthened formulation
we manage to solve 10 more instances than with the classical formulation. With the strengthened
formulations, the 30 instances that are also solved with the classical formulation are solved, respectively, 12, 10, and 6 times faster.


Notice that the speedup obtained with ( _k_ -HC(2 _,_ 1) _,_ (multi _,_ 1)) is smaller than those obtained with
( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)) and ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [).] [Such a behavior is well explained by the results of Propo-]
sitions 3 and 4: As _n_ and _k_ increase, the difference between the exponential lower bound on the
number of nodes required to obtain a nonzero global lower bound in the first proposition and the
polynomial one in the second one becomes larger and larger. Thus, any branching operations taking
place on the constraints _∥wj∥_ 1 _≥_ 1 have a much smaller impact on the bound than those taking
place on the _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], which explains the superior performance of ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [).]


6 CONCLUDING REMARKS


We have focused on solving the 2-norm _k_ -Hyperplane Clustering problem with spatial branch-andbound (SBB) techniques by strengthening the classical formulation with constraints that arise from
(scaled) _p_ -norm formulations of the problem, with _p_ = 2. Focusing on the 1- and _∞_ -norms, we
have theoretically shown that including the constraints stemming from the 1-norm version of the
problem (featuring scaled _∞_ -norm constraints) leads to computing nonzero lower bounds in a linear
(rather than exponential) number of SBB nodes. Our experimental results show very large speedups,
reducing median solve times by up to 41× while increasing the total number of solved instances by
up to 63%, substantially improving the problem’s solvability to global optimality.


9


Table 3: Results on the LowDim dataset


_m_ _n_ _k_ time obj time obj time obj time obj


10 2 2 0.3 0.3 0.2 0.3 0.2 0.3 0.2 0.3
10 2 3 0.7 0.5 1.0 0.5 0.8 0.5 1.0 0.5
14 2 2 1.6 8.5 0.6 8.5 0.2 8.5 0.3 8.5
14 2 3 31.9 0.8 4.4 0.8 3.4 0.8 5.4 0.8
18 2 2 13.9 3.4 0.4 3.4 0.4 3.4 0.7 3.4
18 2 3 488.9 0.7 3.9 0.7 4.4 0.7 4.6 0.7
22 2 2 179.2 9.7 1.7 9.7 1.4 9.7 0.9 9.7
22 2 3 2213.3 2.4 11.2 2.4 11.2 2.4 9.8 2.4
25 2 2 28.9 8.2 0.6 8.2 0.4 8.2 1.4 8.2
25 2 3 168000.0 2.7 936.6 2.7 96.1 2.7 221.0 2.7
26 2 2 168000.0 - 6.2 5.8 10.4 5.8 2.2 5.8
26 2 3 168000.0 - 39.2 3.4 56.6 3.4 28.3 3.4
27 2 2 168000.0 - 0.7 5.1 2.6 5.1 0.8 5.1
27 2 3 168000.0 - 1678.4 3.3 2687.7 3.3 238.6 3.3
28 2 2 168000.0 - 8.6 11.7 6.3 11.7 1.8 11.7
28 2 3 168000.0 - 293.1 3.6 471.3 3.6 153.5 3.6
29 2 2 168000.0 - 0.8 7.1 0.3 7.1 0.8 7.1
29 2 3 168000.0 - 7694.9 7.1 6029.0 7.1 1476.4 7.1
30 2 2 168000.0 - 10.4 9.1 38.5 9.1 1.6 9.1
30 2 3 168000.0 - 172.9 3.4 191.2 3.4 44.3 3.4
10 3 2 1.1 0.9 0.4 0.9 1.0 0.9 0.9 0.9
10 3 3 30.2 0.0 32.6 0.0 31.9 0.0 41.9 0.0
14 3 2 8.4 0.7 0.8 0.7 0.8 0.7 1.4 0.7
14 3 3 206.4 0.1 29.7 0.1 25.5 0.1 49.7 0.1
18 3 2 160.6 0.7 3.7 0.7 7.8 0.7 4.5 0.7
18 3 3 2234.9 0.4 93.4 0.4 91.6 0.4 157.9 0.4
22 3 2 625.0 4.3 15.6 4.3 11.3 4.3 10.8 4.3
22 3 3 135362.9 1.3 1089.5 1.3 638.2 1.3 1243.7 1.3
23 3 2 6459.4 0.9 8.1 0.9 45.5 0.9 10.1 0.9
24 3 2 18049.6 6.9 66.3 6.9 474.7 6.9 34.5 6.9
24 3 3 168000.0 1.7 2470.6 1.5 2716.7 1.5 3817.0 1.5
25 3 2 22886.9 5.7 70.7 5.7 28.1 5.7 14.2 5.7
25 3 3 168000.0 1.3 1952.3 1.3 5060.3 1.3 2885.1 1.3
26 3 2 168000.0 - 6.3 4.5 4.7 4.5 4.4 4.5
26 3 3 168000.0 - 5937.9 1.3 4345.7 1.3 2300.2 1.3
27 3 2 168000.0 - 215.1 3.4 1274.8 3.4 58.5 3.4
27 3 3 168000.0 - 52548.9 2.9 65949.3 2.9 35206.1 2.9
28 3 2 168000.0 - 31.1 3.6 1.7 3.6 10.2 3.6
28 3 3 168000.0 - 4234.9 1.4 74560.6 1.4 4180.9 1.4
29 3 2 168000.0 - 143.5 8.1 34.0 8.1 12.5 8.1
29 3 3 168000.0 - 168000.0 4.9 168000.0 4.9 168000.0 4.9
30 3 2 168000.0 - 8083.1 2.5 168000.0 2.5 3014.8 2.5
30 3 3 168000.0 - 23488.8 3.2 168000.0 3.2 6541.5 3.2


**# Sol** 20 42 40 42


Table 4: Results on the HighDim dataset


_m_ _n_ _k_ time obj time obj time obj time obj


10 2 4 8.3 0.0 2.4 0.0 1.8 0.0 6.8 0.0
10 4 2 4.9 0.0 0.8 0.0 6.1 0.0 3.9 0.0
11 2 4 21.9 0.1 9.8 0.1 5.9 0.1 17.7 0.1
11 2 5 1264.3 0.0 392.8 0.0 300.2 0.0 2689.7 0.0
11 4 2 5.4 0.0 1.6 0.0 1.6 0.0 2.1 0.0
12 2 4 79.4 0.1 17.0 0.1 8.1 0.1 30.5 0.1
12 2 5 425.6 0.0 160.4 0.0 56.8 0.0 282.8 0.0
12 4 2 17.3 0.1 1.2 0.1 7.7 0.1 10.1 0.1
12 5 2 29.3 0.0 14.4 0.0 16.4 0.0 26.1 0.0
13 2 4 238.2 0.1 19.4 0.1 14.6 0.1 38.4 0.1
13 2 5 935.1 0.0 127.1 0.0 55.8 0.0 170.7 0.0
13 3 4 4143.7 0.0 7567.6 0.0 168000.0 - 168000.0 13 4 2 13.0 0.1 6.5 0.1 2.1 0.1 9.3 0.1
13 4 3 948.7 0.0 567.1 0.0 712.6 0.0 4625.7 0.0
13 5 2 47.0 0.1 11.1 0.1 19.8 0.1 28.3 0.1
14 2 4 683.1 0.2 22.4 0.2 12.2 0.2 55.8 0.2
14 2 5 6526.6 0.0 628.6 0.0 211.9 0.0 586.0 0.0
14 3 4 168000.0 - 2757.6 0.0 2784.8 0.0 7540.2 0.0
14 4 2 58.5 0.5 2.2 0.5 7.0 0.5 9.6 0.5
14 4 3 1447.5 0.0 687.9 0.0 890.5 0.0 6906.7 0.0
14 5 2 120.1 0.1 13.8 0.1 21.5 0.1 36.3 0.1
15 2 4 1350.6 0.3 32.9 0.3 23.4 0.3 54.4 0.3
15 2 5 5854.2 0.0 320.5 0.0 92.9 0.0 445.3 0.0
15 3 4 168000.0 - 2760.8 0.0 1772.1 0.0 168000.0 15 4 2 37.5 0.6 5.8 0.6 8.4 0.6 9.2 0.6
15 4 3 3803.0 0.0 515.6 0.0 439.4 0.0 2208.8 0.0
15 5 2 98.1 0.1 13.5 0.1 40.7 0.1 35.0 0.1
16 2 4 5827.2 0.2 119.6 0.2 28.9 0.2 67.3 0.2
16 2 5 168000.0 - 582.6 0.0 346.6 0.0 781.9 0.0
16 3 4 168000.0 - 4586.5 0.0 2407.2 0.0 168000.0 16 3 5 168000.0 - 168000.0 - 168000.0 - 168000.0 16 4 2 179.0 1.1 12.9 1.1 15.0 1.1 12.1 1.1
16 4 3 5144.2 0.0 554.5 0.0 601.1 0.0 2507.3 0.0
16 5 2 444.9 0.8 28.5 0.8 43.2 0.8 60.8 0.8
17 2 4 168000.0 0.2 37.1 0.2 42.1 0.2 69.2 0.2
17 2 5 168000.0 0.1 1452.3 0.1 999.4 0.1 1517.1 0.1
17 3 4 168000.0 - 4970.5 0.0 2553.9 0.0 168000.0 17 3 5 168000.0 - 168000.0 - 168000.0 - 168000.0 17 4 2 175.7 0.5 9.8 0.5 10.6 0.5 9.8 0.5
17 4 3 168000.0 - 904.1 0.0 967.5 0.0 3679.0 0.0
17 4 4 168000.0 - 8218.2 0.0 8102.3 0.0 8104.9 0.0
17 5 2 1092.7 1.4 87.0 1.4 97.4 1.4 101.0 1.4
17 5 3 168000.0 - 8116.4 0.0 8082.4 0.0 7910.9 0.0


**# Sol** 31 41 40 37


An interesting research direction for future work is exploring the connection between _k_ -HC and subspace clustering, in particular related to the recent literature on coresets for projective clustering and
subspace approximation Rademacher et al. (2005); Sohler & Woodruff (2018); Eiben et al. (2021).
These techniques allow us to construct small, weighted subsets of data that preserve the clustering
cost within a (1 + _ε_ ) factor. Integrating such coreset constructions with our exact SBB-based solver
could yield a hybrid approach (approximate in data, but exact in optimization), combining scalability
with provable global optimality guarantees.


ACKNOWLEDGEMENT OF SUPPORT


The author’s work was partially supported by the European Union under Next Generation EU

- the Italian National Recovery and Resilience Plan (PNRR), PRIN 2022 PNRR (project code
P20227CTY3, CUP D53D23018800001), project title ”HEXAGON: Highly-specialized EXact Algorithms for Grid Operations at the National level”.


10


REPRODUCIBILITY STATEMENT


The author provided all the necessary information to facilitate the reproducibility of the results. The
code developed for this work is made available online (see Appendix) and freely distributed under
the MIT license. [4]


ETHICS STATEMENT


All datasets employed in this work are publicly available for research and contain no personally
identifiable information or harmful content. The methods introduced in this paper have a societal
impact comparable to that of any other clustering algorithm.


LLM USAGE STATEMENT


All technical content presented in this paper is entirely the work of the author, with LLMs serving
only as an editorial tool.


REFERENCES


E. Amaldi and S. Coniglio. A distance-based point-reassignment heuristic for the k-hyperplane
clustering problem. _European Journal of Operational Research_, 227(1):22–29, 2013.


E. Amaldi and M. Mattavelli. The MIN PFS problem and piecewise linear model estimation. _Discrete_
_Applied Mathematics_, 118(1-2):115–143, 2002.


E. Amaldi, K. Dhyani, and A. Ceselli. Column generation for the minimum hyperplanes clustering
problem. _INFORMS Journal on Computing_, 25(3):446–460, 2013.


E. Amaldi, S. Coniglio, and L. Taccari. Discrete optimization methods to fit piecewise affine models
to data points. _Computers & Operations Research_, 75:214–230, 2016.


P. Belotti, J. Lee, L. Liberti, F. Margot, and A. W¨achter. Branching and bound tightening techniques
for non-convex MINLP. _Optimization methods and software_, 24:597–634, 2009.


P. Bradely and O.L. Mangasarian. _k_ -plane clustering. _Journal_ _of_ _Global_ _Optimization_, 16:23–32,
2000.


S. Coniglio. The impact of the norm on the k-hyperplane clustering problem: relaxations, restrictions, approximation factors, and exact formulations. In _Proceedings of the 10th CTW on Graphs_
_and Combinatorial Optimization_, pp. 118–121, 2011.


K. Dhyani and L. Liberti. Mathematical programming formulations for the bottleneck hyperplane
clustering problem. In _Proceedings of Modelling, Computation and Optimization in Information_
_Systems and Management Sciences_, volume 14, pp. 87–96, 2008.


E. Eiben, F.V. Fomin, P.A. Golovach, W. Lochet, F. Panolan, and K. Simonov. Eptas for k-means
clustering of affine subspaces. In _Proceedings_ _of_ _the_ _2021_ _ACM-SIAM_ _Symposium_ _on_ _Discrete_
_Algorithms (SODA)_, pp. 2649–2659. SIAM, 2021.


G. Ferrari-Trecate, M. Muselli, D. Liberati, and M. Morari. A clustering technique for the identification of piecewise affine systems. _Automatica_, 39:205–217, 2003.


P. Georgiev, P. Pardalos, and F. Theis. A bilinear algorithm for sparse representations. _Computa-_
_tionals Optimization and Applications_, 38(2):249–259, 2007.


Gurobi Optimization, LLC. Gurobi Optimizer Reference Manual, 2026. URL [https://www.](https://www.gurobi.com)
[gurobi.com.](https://www.gurobi.com)


[4https://choosealicense.com/licenses/mit/](https://choosealicense.com/licenses/mit/)


11


H. He and Z. Qin. A k-hyperplane-based neural network for non-linear regression. In _9th_ _IEEE_
_International Conference on Cognitive Informatics (ICCI’10)_, pp. 783–787. IEEE, 2010.


Z. He and A. Cichocki. An efficient k-hyperplane clustering algorithm and its application to sparse
component analysis. In _International Symposium on Neural Networks_, pp. 1032–1041. Springer,
2007.


D. Kong, L. Xu, X. Li, and S. Li. K-plane-based classification of airborne lidar data for accurate
building roof measurement. _IEEE_ _Transactions_ _on_ _Instrumentation_ _and_ _Measurement_, 63(5):
1200–1214, 2013.


J. MacQueen. Some methods for classification and analysis of multivariate observations. In _Pro-_
_ceedings_ _of_ _the_ _fifth_ _Berkeley_ _symposium_ _on_ _mathematical_ _statistics_ _and_ _probability_, volume 1
(14), pp. 281–297. Oakland, CA, USA, 1967.


O.L. Mangasarian. Arbitrary-norm separating plane. _Operations Research Letters_, 24(1-2):15–23,
1999.


G. McCormick. Computability of global solutions to factorable nonconvex programs: Part i - convex
underestimating problems. _Math. Progm._, 10:146–175, 1976.


N. Megiddo and A. Tamir. On the complexity of locating linear facilities in the plane. _Operations_
_research letters_, 1(5):194–197, 1982.


L. Rademacher, S. Vempala, and G. Wang. Matrix approximation and projective clustering via
iterative sampling. 2005.


C. Sohler and D.P. Woodruff. Strong coresets for k-median and subspace approximation: Goodbye
dimension. In _2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS)_,
pp. 802–813. IEEE, 2018.


M.C. Tsakiris and R. Vidal. Hyperplane clustering via dual principal component pursuit. In _Inter-_
_national conference on machine learning_, pp. 3472–3481. PMLR, 2017.


Y. Washizawa and A. Cichocki. On-line k-plane clustering learning algorithm for sparse component
analysis. In _2006_ _IEEE_ _International_ _Conference_ _on_ _Acoustics_ _Speech_ _and_ _Signal_ _Processing_
_Proceedings_, volume 5, pp. V–V. IEEE, 2006.


y. Zhang, h. Wang, W. Wang, and S. Sanei. K-plane clustering algorithm for analysis dictionary
learning. In _2013_ _IEEE_ _International_ _Workshop_ _on_ _Machine_ _Learning_ _for_ _Signal_ _Processing_
_(MLSP)_, pp. 1–4. IEEE, 2013.


A CODE REPOSITORY AND LICENSING


The code used for the experiments is freely available under the MIT license [(https://](https://choosealicense.com/licenses/mit/)
[choosealicense.com/licenses/mit/)](https://choosealicense.com/licenses/mit/) and is available at [https://github.com/](https://github.com/stefanoconiglio/khc-multinorm)
[stefanoconiglio/khc-multinorm.](https://github.com/stefanoconiglio/khc-multinorm)


B FURTHER COMPUTATIONAL RESULTS


Table 5 reports the total node counts for the HighDim dataset. This provides a clearer picture of
relative tree sizes and convergence behavior across formulations. The results confirm the theoretical
analysis, with the ( _k_ -HC(2 _,_ 1)), ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 _n_ ) [)][,] [(] _[k]_ [-HC] (2 _,_ 1) _,_ ( _∞,_ 1) [)][,] [and][ (] _[k]_ [-HC] (multi _,_ 1) [)][ formula-]
tions generating, respectively, 7,987,723.07, 3,201,881.49, 2,741,496.67, and 4,632,264.51 nodes
on average.


12


Table 5: HighDim instances: total SBB node counts by formulation.


_m_ _n_ _k_ ( _k_ -HC(2 _,_ 1)) ( _k_ -HC(2 _,_ 1) _,_ (1 _,_ ~~_√_~~ 1 ( _k_ -HC(2 _,_ 1) _,_ ( _∞,_ 1)) ( _k_ -HC(multi _,_ 1))
_n_ ) [)]


10 2 4 38 392 16 357 10 301 23 588
10 4 2 17 033 4 739 18 112 16 937
11 2 4 78 168 29 502 17 868 41 565
11 2 5 4 654 600 1 151 890 958 992 8 332 930
11 4 2 21 404 8 701 6 989 8 584
12 2 4 287 736 45 100 22 786 75 515
12 2 5 1 228 060 440 626 169 063 687 073
12 4 2 61 743 5 820 19 455 26 611
12 5 2 85 702 42 506 36 266 60 555
13 2 4 791 227 52 720 41 656 99 091
13 2 5 2 621 440 347 674 154 070 383 310
13 3 4 11 864 400 19 678 500 23 810 300 18 600 100
13 4 2 41 063 19 382 9 097 24 744
13 4 3 2 629 260 1 284 080 1 646 530 9 724 370
13 5 2 139 309 24 034 40 346 48 397
14 2 4 2 355 780 61 667 34 681 166 633
14 2 5 19 826 000 2 014 470 715 573 1 582 970
14 3 4 21 011 800 5 928 130 6 446 500 16 330 600
14 4 2 197 555 10 637 15 814 19 679
14 4 3 3 653 010 1 509 690 1 914 490 14 396 900
14 5 2 367 215 34 060 40 631 67 349
15 2 4 4 819 300 88 293 63 762 142 499
15 2 5 15 129 700 815 961 240 698 1 057 150
15 3 4 20 797 200 6 821 170 4 221 290 17 132 800
15 4 2 123 055 14 603 21 152 19 678
15 4 3 9 399 350 1 068 560 949 518 4 432 280
15 5 2 285 279 25 590 83 182 63 958
16 2 4 20 072 700 387 933 86 122 177 715
16 2 5 18 348 500 1 839 480 977 328 1 834 550
16 3 4 18 615 700 12 285 100 5 353 010 16 241 100
16 3 5 16 743 900 16 840 700 15 925 600 14 859 600
16 4 2 622 968 29 784 34 762 25 049
16 4 3 12 307 300 1 297 660 1 169 750 4 721 850
16 5 2 1 411 490 66 069 100 491 107 740
17 2 4 23 783 300 108 023 120 569 176 621
17 2 5 18 531 600 4 274 570 3 036 550 3 629 780
17 3 4 18 137 400 10 911 800 5 561 730 16 517 600
17 3 5 16 997 500 15 721 000 16 018 000 13 877 100
17 4 2 599 093 22 341 26 660 18 985
17 4 3 18 403 800 2 103 800 2 034 520 7 360 030
17 4 4 16 580 800 14 203 900 13 111 500 12 754 700
17 5 2 3 190 660 187 282 217 843 189 688
17 5 3 16 600 600 15 857 000 12 400 800 13 129 400


C LIST OF OUR THEORETICAL RESULTS WITH THE CORRESPONDING PROOFS


**Proposition 1.** _Given a hyperplane H_ := _{x_ _∈_ R _[n]_ : _x_ _[⊤]_ _w_ = _γ} and a point a_ _∈_ R _[n]_ _, the function_
_dp_ ( _a, H_ ) = _[|][w][⊤][a][−][γ][|]_ _[, where]_ [1] [+] [1] _[′]_ [= 1] _[, is a nonconvex function of]_ [ (] _[w, γ]_ [)] _[ for every][ p][ ∈]_ [N] _[ ∪{∞}][.]_


_[a][−][γ][|]_ [1]

_∥w∥p′_ _[, where]_ _p_


[1] [1]

_p_ [+] _p_


_p_ _[′]_ [= 1] _[, is a nonconvex function of]_ [ (] _[w, γ]_ [)] _[ for every][ p][ ∈]_ [N] _[ ∪{∞}][.]_


_Proof._ By definition, _[|][w][⊤][a][−][γ][|]_


_∥w∥p′_ is a convex function of ( _w, γ_ ) if and only if the following holds for


every ( _w_ 1 _, γ_ 1) and ( _w_ 2 _, γ_ 2) _∈_ R _[n]_ [+1] and _λ ∈_ [0 _,_ 1]:


1 _[a][ −]_ _[γ]_ [1] _[|]_ 2 _[a][ −]_ _[γ]_ [2] _[|]_
_λ_ _[|][w][⊤]_ + (1 _−_ _λ_ ) _[|][w][⊤]_ _≥_
_∥w_ 1 _∥p′_ _∥w_ 2 _∥p′_

_|_ ( _λw_ 1 + (1 _−_ _λ_ ) _w_ 2) _[⊤]_ _a −_ ( _λγ_ 1 + (1 _−_ _λ_ ) _γ_ 2) _|_

_._ (3)
_∥λw_ 1 + (1 _−_ _λ_ ) _w_ 2 _∥p′_


Let _p_ _[′]_ _∈_ N. Let _a_ = (0 _,_ 0) and consider two hyperplanes of parameters _w_ 1 := (1 _, −_ [1]


Let _p_ _∈_ N. Let _a_ = (0 _,_ 0) and consider two hyperplanes of parameters _w_ 1 := (1 _, −_ 5 [)] _[, γ]_ [1] [= 1][ and]

_w_ 2 := ( _−_ [1] _[,]_ [ 1)] _[, γ]_ [2] [= 1][.] [Let] _[ γ]_ [:=] _[ γ]_ [1] [=] _[ γ]_ [2][.] [Letting] _[ λ]_ [ =] [1] [, Inequality (3) reads:]


5 [1] _[,]_ [ 1)] _[, γ]_ [2] [= 1][.] [Let] _[ γ]_ [:=] _[ γ]_ [1] [=] _[ γ]_ [2][.] [Letting] _[ λ]_ [ =] [1] 2


[.] [Let] _[ γ]_ [:=] _[ γ]_ [1] [=] _[ γ]_ [2][.] [Letting] _[ λ]_ [ =] 2 [, Inequality (3) reads:]

1 1 [1] 1 1
2 _p_ [�] _[′]_ 1 _p′_ [+] 2 _p_ [�] _[′]_ 1 _p′_ _[≥]_ _p_ ~~[�]~~ _[′]_ 2 _p′_ 2 _p′_ _[,]_


1 [1] 1

1 + - 15 - _p′_ [+] 2 _p_ [�] _[′]_ 1 +


(4)

25 - _p′_ + - 52 - _p′_ _[,]_


_p_ [�] _[′]_


_p_ [�] _[′]_


1 1

1 + - 15 - _p′_ _[≥]_ _p_ ~~[�]~~ _[′]_ [�] 25 - _p′_


_p_ ~~[�]~~ _[′]_ [�] 2


or, equivalently:


~~�~~
_p_ _[′]_ - 2

5


- _p′_ - 2
+
5


 - 1
1 +
5


- _p′_ _p_ _[′]_
_≥_


- _p′_
_._


Taking both sides to the _p_ _[′]_ -th power, we have 2 - 25 - _p_ _[′]_ _≥_ 1 + - 15 - _p_ _[′]_ . After moving 1 to the lefthand side and multiplying both sides by 5 _[p][′]_, we deduce 2 _·_ 2 _[p][′]_ _−_ 1 _≥_ 5 _[p][′]_, which implies 2 _·_ 2 _[p][′]_ _>_


13


2 _·_ 2 _[p][′]_ _−_ 1 _≥_ 5 _[p][′]_ . As - 52 - _p_ _[′]_ _>_ 2 holds for every _p_ _[′]_ _∈_ N _∪{∞}_ (as one can see by setting _p_ _[′]_ to its
smallest value, i.e., setting _p_ _[′]_ := 1), Inequality (4) is proven not to hold for any choice of _p_ _[′]_ _∈_ N
( _p ∈_ N _\ {_ 1 _} ∪{∞}_ ).


Let us consider the case _p_ _[′]_ = _∞_ now. With _w_ 1 = (1 _, −_ [1]


5 [1] [)][ and] _[ w]_ [2] [=] [(] _[−]_ [1] 5


Let us consider the case _p_ = _∞_ now. With _w_ 1 = (1 _, −_ 5 [)][ and] _[ w]_ [2] [=] [(] _[−]_ 5 _[,]_ [ 1)][, we have] _[ ∥][w]_ [1] _[∥][∞]_ [=]

_∥w_ 2 _∥∞_ = 1 and, with _λ_ = [1] 2 [, we obtain] �� 21 [(] _[w]_ [1][ +] _[ w]_ [2][)] �� _∞_ [=] _[∥]_ [1] 2 [(] [4] 5 _[,]_ [4] 5 [)] _[∥][∞]_ [=] [2] 5 [.] [Substituting these]

values directly into equation 3 leads to


[1] 2 [, we obtain] �� 21 [(] _[w]_ [1][ +] _[ w]_ [2][)] �� _∞_ [=] _[∥]_ [1] 2


[1] 2 [(] [4] 5


[4] [4]

5 _[,]_ 5


[4] [2]

5 [)] _[∥][∞]_ [=] 5


1 [1]
2 [+] 2


2 _[,]_


[1]

2 _[≥]_ [5] 2


which does not hold, showing that convexity fails to hold also for _p_ _[′]_ = _∞_ ( _p_ = 1).


**Lemma 1.** _The solutions to_ ( _k_ -HC(2 _,_ 1)) _and_ ( _k_ -HC2) _coincide._ _Also,_ ( _k_ -HC( _p,c_ )) _is quadratically_
_homogeneous w.r.t._ _c, i.e.,_ OPT( _k_ -HC( _p,c_ )) = _c_ [2] OPT( _k_ -HC( _p,_ 1)) _._


_Proof._ We start by showing that ( _k_ -HC(2 _,_ 1)) and ( _k_ -HC2) are equivalent when _c_ = 1 and _p_ = 2.
5 Indeed, as _n_ points in general position fix a hyperplane in R _n_, only _n_ of the _n_ + 1 parameters
in ( _wj, γj_ ) _∈_ R _[n]_ [+1] are independent. Thus, _||wj||_ [2] 2 [=] _[||][w][j][||]_ [2] [=] [1][ can be imposed w.l.o.g.] [for all]
_j_ _∈_ [ _k_ ]. Relaxing _||wj||_ 2 = 1 as _||wj||_ 2 _≥_ 1 is w.l.o.g. as the latter is tight in any optimal solution—
indeed, if not, a strictly better solution can be found by scaling ( _wj, γj_ ) by _||wj_ 1 _||p′_ [,] _[j]_ _[∈]_ [[] _[k]_ []][.] [Let]
_{_ ( _wj, γj_ ) _}j∈_ [ _k_ ] be an optimal solution to ( _k_ -HC( _p,c_ )). As argued, _∥wj∥p′_ = _c_ holds. Let now
( _wj_ _[′]_ _[, γ]_ _j_ _[′]_ [)] [:=] [(] _[w][j]_ _c_ _[,γ][j]_ [)], _j_ _∈_ [ _k_ ]. Such a scaled solution satisfies _∥wj_ _[′]_ _[∥][p][′]_ [=] [1][ for all] _[ j]_ _[∈]_ [[] _[k]_ []][ and, thus,]


( _wj_ _[′]_ _[, γ]_ _j_ _[′]_ [)] [:=] [(] _[w][j]_ _c_ _[,γ][j]_ [)], _j_ _∈_ [ _k_ ]. Such a scaled solution satisfies _∥wj_ _[′]_ _[∥][p][′]_ [=] [1][ for all] _[ j]_ _[∈]_ [[] _[k]_ []][ and, thus,]

is feasible for ( _k_ -HC( _p,_ 1)). Its objective function value is [1][2] [times the one of] _[ {]_ [(] _[w][j][, γ][j]_ [)] _[}][j][∈]_ [[] _[k]_ []][.] [Since]


is feasible for ( _k_ -HC( _p,_ 1)). Its objective function value is _c_ [2] [times the one of] _[ {]_ [(] _[w][j][, γ][j]_ [)] _[}][j][∈]_ [[] _[k]_ []][.] [Since]

such a multiplicative difference is a constant, the scaled solution is also optimal for ( _k_ -HC( _p,_ 1)).
Thus, we have OPT( _k_ -HC( _p,c_ )) = _c_ [2] OPT( _k_ -HC( _p,_ 1)).


**Theorem** **1.** _Let_ _p, q_ _∈_ N _∪{∞}_ _and_ _c_ _>_ 0 _._ _The_ _three_ _positive_ _scalars_ _α_ ( _p, q_ ) _, β_ ( _p, q_ ) _, δ_ ( _p, q_ )
_which,_ _for_ _all_ _x_ _∈_ R _[n]_ _,_ _satisfy_ _the_ _congruence_ _inequality_ _α_ ( _p, q_ ) _||x||p_ _≤_ _β_ ( _p, q_ ) _||x||q_ _≤_
_δ_ ( _p, q_ ) _||x||p for p, q_ _∈_ N _∪{∞} also satisfy the optimal-value inequality_ _[α]_ _δ_ ( [(] _p,q_ _[p,q]_ ) [)][2][2] [OPT(] _[k]_ [-HC][(] _[p,c]_ [)][)] _[ ≤]_


  OPT _k_ -HC _β_ ( _p,q_ )
( _q,c_ _δ_ ( _p,q_ ) [)]


_Proof._ The inequality


_≤_ OPT( _k_ -HC( _p,c_ )) _._


min (5)
_x∈X_ _[f]_ [(] _[x]_ [)] _[ ≤]_ _x_ [min] _∈X_ _[f][ ′]_ [(] _[x]_ [)] _[ ≤]_ _x_ [min] _∈X_ _[f][ ′′]_ [(] _[x]_ [)]


clearly holds for any three functions _f, f_ _[′]_ _, f_ _[′′]_ : _X_ _→_ R satisfying _f_ ( _x_ ) _≤_ _f_ _[′]_ ( _x_ ) _≤_ _f_ _[′′]_ ( _x_ ) for all
_x_ _∈_ _X_ _⊆_ R _[n]_ . Since vector norms in R _[n]_ are congruent, for every _p, q_ _∈_ N _∪{∞}_ there are three
positive scalars _α_ ( _p, q_ ) _, β_ ( _p, q_ ) _, δ_ ( _p, q_ ) which satisfy _α_ ( _p, q_ ) _||x||p_ _≤_ _β_ ( _p, q_ ) _||x||q_ _≤_ _δ_ ( _p, q_ ) _||x||p_
for _p, q_ _∈_ N _∪{∞}_ . Since, by definition, _dp_ ( _a, H_ ) = min _y∈H ||a −_ _y||p_, equation 5 leads to the
following congruence relationship for point-to-hyperplane distances that holds for every hyperplane
_H_ in R _[n]_ and point _a ∈_ R _[n]_ :


_α_ ( _p, q_ ) _dp_ ( _a, H_ ) _≤_ _β_ ( _p, q_ ) _dq_ ( _a, H_ ) _≤_ _δ_ ( _p, q_ ) _dp_ ( _a, H_ ) _._ (6)


Squaring equation 6 and letting _H_ 1 _, . . ., Hk_ be an arbitrary choice of _k_ hyperplanes, another application of equation 5 leads to


_α_ ( _p, q_ ) [2] min (7)
_j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[p][} ≤]_ _[β]_ [(] _[p, q]_ [)][2][ min] _j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[q][} ≤]_ _[δ]_ [(] _[p, q]_ [)][2][ min] _j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[p][}][.]_


5This was already observed in Amaldi & Coniglio (2013). The proof we provide here will be useful in the
following.


14


Summing equation 7 over all data points with unit multipliers, we obtain the following surrogate
inequality:


_m_
_α_ ( _p, q_ ) [2]           - min

_j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[p][} ≤]_
_i_ =1

_m_
_β_ ( _p, q_ ) [2]            - min

_j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[q][} ≤]_
_i_ =1

_m_
_δ_ ( _p, q_ ) [2]                - min

_j∈_ [ _k_ ] _[{][d]_ [2][(] _[a][i][, H][j]_ [)] _[p][}][.]_
_i_ =1


Applying again equation 5 by letting the minimization consider the choice of optimal parameters for
the hyperplanes _Hj, j_ _∈_ [ _k_ ], we deduce _α_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,_ 1)) _≤_ _β_ ( _p, q_ ) [2] OPT( _k_ -HC( _q,_ 1)) _≤_
_δ_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,_ 1)). Multiplying through by _c_ [2] and using Lemma 1, we obtain
_α_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,c_ )) _≤_ _β_ ( _p, q_ ) [2] OPT( _k_ -HC( _q,c_ )) _≤_ _δ_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,c_ )). By using the
quadratic homogeneity property of Lemma 1 one more time, we deduce _β_ ( _p, q_ ) [2] OPT( _k_ -HC( _q,c_ )) =
OPT( _k_ -HC( _q,cβ_ ( _p,q_ ))), which allows us to write:

_α_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,c_ )) _≤_ OPT( _k_ -HC( _q,cβ_ ( _p,q_ ))) _≤_ _δ_ ( _p, q_ ) [2] OPT( _k_ -HC( _p,c_ )) _._


Dividing all three terms by _δ_ ( _p, q_ ) [2] and applying Lemma 1 one last time to remove the coefficient
_δ_ ( _p,q_ 1 ) [2] [that would otherwise multiply the inner term, the claim follows.]


**Corollary 1.** _k_ -HC( _∞,_ 1) _and k_ -HC(1 _,_ ~~_√_~~ 1 _n_ ) _[satisfy:]_


1
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(] _[∞][,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)]

1
~~_√_~~ 1 (2 _,_ 1) [)] _[.]_
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(1] _[,]_ _n_ ) [)] _[ ≤]_ [OPT(] _[k]_ [-HC]


_Proof._ We rely on the following congruence relationships (see Proposition 5 for their derivation):

1 1 1
~~_√_~~ _∥x∥_ 2 _≤∥x∥∞_ _≤∥x∥_ 2 ~~_√_~~ _∥x∥_ 2 _≤_ ~~_√_~~ _∥x∥_ 1 _≤∥x∥_ 2 _._
_n_ _n_ _n_


Thanks to Theorem 1, ~~_√_~~ 1 _n_ _∥x∥_ 2 _≤∥x∥∞_ _≤∥x∥_ 2 implies


1
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(] _[∞][,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[.]_

Thanks to Theorem 1, ~~_√_~~ 1 _n_ _∥x∥_ 2 _≤_ ~~_√_~~ 1 _n_ _∥x∥_ 1 _≤∥x∥_ 2 implies


1
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ _n_ [1] [OPT(] _[k]_ [-HC][(1] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)]

which, due to Lemma 1, implies

1
~~_√_~~ 1 (2 _,_ 1) [)] _[.]_
_n_ [OPT(] _[k]_ [-HC][(2] _[,]_ [1)][)] _[ ≤]_ [OPT(] _[k]_ [-HC][(1] _[,]_ _n_ ) [)] _[ ≤]_ [OPT(] _[k]_ [-HC]


**Lemma 2.** _Solving k-HC subject to_ min _{||w||_ 1 _,_ _[√]_ _n||w||∞}_ _≥_ 1 _coincides with solving an uncon-_
_strained_ _version_ _of_ _k-HC_ _where_ _the_ _point-to-hyperplane distance between_ _ai_ _and Hj_ _is defined as_
max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}._

_Proof._ As a consequence of Lemma 1, imposing min _{||w||_ 1 _,_ _[√]_ _n||w||∞}_ _≥_ 1 in the context of _k_ HC implies imposing min _{||w||_ 1 _,_ _[√]_ _n||w||∞}_ = 1 in any optimal solution and, thus, accounting for

_|a_ _[⊤]_ _i_ _[w][j]_ _[−][γ][|]_
the distance between _ai_ and _Hj_ as _|a_ _[⊤]_ _i_ _[w][j]_ _[−]_ _[γ][|]_ [=] min _{||w||_ 1 _,_ ~~_[√]_~~ _n||w||∞}_ [.] [We can rewrite the latter as]


max _{_ _[|][a]_ _i_ _[⊤][w][j]_ _[−][γ][|]_


_||_ _[w]_ _w_ _[j]_ _||_ _[−]_ 1 _[γ][|]_ _,_ _[|]_ ~~_√_~~ _[a]_ _i_ _[⊤]_ _n|_ _[w]_ _|w_ _[j]_ _[−]_ _||∞_ _[γ][|]_ _}_ = max _{_ _[|][a]_ _i_ _[⊤]_ _||_ _[w]_ _w_ _[j]_ _||_ _[−]_ 1 _[γ][|]_


_||_ _[w]_ _w_ _[j]_ _||_ _[−]_ 1 _[γ][|]_ _,_ ~~_√_~~ 1 _n_ _|a||_ _[⊤]_ _i_ _w_ _[w]_ _||_ _[j]_ _∞_ _[−][γ][|]_ _}_ = max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}_ .


15


_y_


_x_


Figure 3: Sets of points satisfying _∥x∥_ 2 = 1 (inner circle) and max _{∥x∥∞,_ ~~_√_~~ 1 _n_ _∥x∥_ 1 _}_ = 1 (outer
octagon).


**Lemma** **3.** _The_ _function_ max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}_ _is_ _a_ _distance_ _induced_ _by_ _the_ _norm_
max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}._


_Proof._ Let us show that max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ is a norm in three steps.

_I._ _Positive_ _definiteness_ . First, it is clear that max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ _≥_ 0 and that
max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ = 0 if and only if _x_ = 0.

_II._ _Absolute_ _homogeneity_ . Second, it is also clear that _|λ|_ max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ =
max _{λ||x||∞, λ_ ~~_√_~~ [1] _n_ _||x||_ 1 _}_ for all _λ ∈_ R.


_III. Triangle inequality_ . Third, we must show that

max _{||x_ + _y||∞,_ ~~_√_~~ [1] _||x_ + _y||_ 1 _} ≤_ max _{||x||∞,_ ~~_√_~~ [1] _||x||_ 1 _}_ + max _{||y||∞,_ ~~_√_~~ [1] _||y||_ 1 _}_ (8)
_n_ _n_ _n_


holds for any _x, y_ _∈_ R _[n]_ . To see this, we first notice that


1 1 1
_||x_ + _y||∞_ _≤||x||∞_ + _||y||∞_ and ~~_√_~~ _||x_ + _y||_ 1 _≤_ ~~_√_~~ _||x||_ 1 + ~~_√_~~ _||y||_ 1
_n_ _n_ _n_


hold since these functions are norms. Taking the maximum of the left-hand and right-hand sides of
these two inequalities, thanks to the monotonicity of max we have:


1
max _{||x_ + _y||∞,_ ~~_√_~~ [1] _||x_ + _y||_ 1 _} ≤_ max _{||x||∞_ + _||y||∞,_ ~~_√_~~ [1] _||x||_ 1 + ~~_√_~~ _||y||_ 1 _}._ (9)
_n_ _n_ _n_


To show that this implies that the triangle inequality is satisfied, we show that, for any _a, b, c, d ≥_ 0,
we have
max _{a_ + _c, b_ + _d} ≤_ max _{a, b}_ + max _{c, d}._ (10)


Trivially, we have _a ≤_ max _{a, b}_, _b ≤_ max _{a, b}_, _c ≤_ max _{c, d}_, and _d ≤_ max _{c, d}_ . Adding the
inequalities in pairs, we obtain _a_ + _c ≤_ max _{a, b}_ +max _{c, d}_ and _b_ + _d ≤_ max _{a, b}_ +max _{c, d}_ .
Taking the maximum of the left- and right-hand sides and applying again the monotonicity of max,
we deduce equation 10.

Letting now _a_ := _||x||∞_, _c_ := _||y||∞_, _b_ := ~~_√_~~ 1 _n_ _||x||_ 1, and _d_ := ~~_√_~~ 1 _n_ _||y||_ 1, from equation 10 we have

max _{||x||∞_ + _||y||∞,_ ~~_√_~~ [1] _||x||_ 1 + ~~_√_~~ [1] _||y||_ 1 _} ≤_ max _{||x||∞,_ ~~_√_~~ [1] _||x||_ 1 _}_ +max _{||y||∞,_ ~~_√_~~ [1] _||y||_ 1 _}._
_n_ _n_ _n_ _n_

(11)
Combining equation 11 with equation 9, equation 8 is proven.


16


We have shown that max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ is a norm. Showing that
max _{d∞_ ( _ai, Hj_ ) _,_ ~~_√_~~ 1 _n_ _d_ 1( _ai, Hj_ ) _}_ is a distance follows straightforwardly by following the
classical construction of point-to-hyperplane distances.


An illustration of the function max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _}_ is provided in Figure 3.

**Lemma 4.** _The norm_ max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _} satisfies the congruence inequality_

_n_ _[−]_ 4 [1] _∥x∥_ 2 _≤_ max� _∥x∥∞,_ ~~_√_~~ 1 _n_ _∥x∥_ 1         - _≤∥x∥_ 2 _._


_Proof._ We prove the congruence relationship in two steps.


_I. Second part._ From the second part of each of the two congruence relationships


1 1 1
~~_√_~~ _∥x∥_ 2 _≤∥x∥∞_ _≤∥x∥_ 2 ~~_√_~~ _∥x∥_ 2 _≤_ ~~_√_~~ _∥x∥_ 1 _≤∥x∥_ 2 _,_
_n_ _n_ _n_


we directly deduce max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _} ≤||x||_ 2.


_II. First part._ To prove the first part of the congruence, we establish what the largest value of _∥x∥_ 2
is when _x_ is subject to max _{||x||∞,_ ~~_√_~~ 1 _n_ _||x||_ 1 _} ≤_ 1.

Let _S_ := _{x_ _∈_ R _[n]_ : _∥x∥∞_ _≤_ 1 _,_ ~~_√_~~ 1 _n_ _∥x∥_ 1 _≤_ 1 _}_, or, equivalently, _S_ := _{x_ _∈_ R _[n]_ : _∥x∥∞_ _≤_
1 _,_ _∥x∥_ 1 _≤_ _[√]_ _n}_ . Let _r_ be the fractional part of _[√]_ _n_, i.e., _r_ := _[√]_ _n_ _−⌊_ _[√]_ _n⌋_ _∈_ [0 _,_ 1). We’ll
prove that every maximizer of _∥x∥_ 2 over _S_ has at most one fractional component in (0 _,_ 1) and, in
particular, that _x_ _[⋆]_ = (1 _, . . .,_ 1 _,_ _r,_ 0 _, . . .,_ 0) is one such maximizer with objective function value

���        _⌊_ ~~_[√]_~~ _n⌋_ times


_,_ _r,_ 0 _, . . .,_ 0) is one such maximizer with objective function value


max _x∈S ∥x∥_ 2 = - _⌊_ ~~_[√]_~~ _n⌋_ + _r_ [2] .


Since _S_ is symmetric under sign flips and coordinate permutations, we can w.l.o.g. restrict ourselves
to vectors _x ∈_ R _[n]_ with _x_ 1 _≥_ _x_ 2 _≥· · · ≥_ _xn_ _≥_ 0 and consider the equivalent problem


_n_ 
- _xi_ _≤_ _[√]_ _n, x ∈_ [0 _,_ 1] _[n]_


_i_ =1


( _P_ ) max


- _n_

 - _x_ [2] _i_ [:]

_i_ =1


_._


_(i)_ First, we show that constraint [�] _i_ _[n]_ =1 _[x][i]_ _[≤√][n]_ [is tight in any optimal ][so][lution.] [This is because,]
if not, we could increase each _xi_ until either _xi_ = 1 or [�] _i_ _[n]_ =1 _[x][i]_ [=] _[√][n]_ [,] [thereby] [increasing] [the]
objective function [�] _i_ _[n]_ =1 _[x]_ _i_ [2][.]

_(ii)_ Second, we show that any optimal solution features at most one fractional component. Suppose
that _x_ is feasible with [�] _i_ _[n]_ =1 _[x][i]_ [=] _[√][n]_ [with] [0] _[<]_ _[x][i]_ _[<]_ [1] [and] [0] _[<]_ _[x][j]_ _[<]_ [1] [for] [some] _[i]_ [=] _[j]_ _[∈]_ [[] _[n]_ []][.]
W.l.o.g., assume _xi_ _≥_ _xj_ . Pick some _ε_ _>_ 0 with _xi_ + _ε_ _≤_ 1 and _xj_ _−_ _ε_ _≥_ 0, and define _x_ ˜ as
_x_ ˜ _i_ := _xi_ + _ε_, _x_ ˜ _j_ = _xj_ _−_ _ε_, and _x_ ˜ _k_ = _xk_ for all _k_ _∈_ [ _n_ ] with _k_ = _i, j_ . Then,


_n_

- _x_ ˜ [2] _i_ _[−]_

_i_ =1


_n_


- _x_ [2] _i_ [= (] _[x][i]_ [+] _[ ε]_ [)][2][ + (] _[x][j]_ _[−]_ _[ε]_ [)][2] _[ −]_ _[x]_ [2] _i_ _[−]_ _[x]_ _j_ [2] [= 2] _[ε]_ [(] _[x][i]_ _[−]_ _[x][j]_

_i_ =1 - �� _≥_ 0


) + 2 _ε_ [2] _>_ 0 _._


This shows that any _x_ with two fractional entries is suboptimal.


_(iii)_ Let _x_ be an optimal solution with _t_ ones, one fractional component _r_ _∈_ [0 _,_ 1) (or none if _r_ = 0),
and _n −_ _t −_ 1 zeros. Since [�] _i_ _[n]_ =1 _[x][i]_ _[≤√][n]_ [is] [ti][ght,] [we] [deduce] _[t]_ [ +] _[ r]_ [=] _[√][n]_ [,] [which] [(since] _[t]_ [is]
integer and _r_ _<_ 1) implies _t_ = _⌊_ _[√]_ _n⌋_ and _r_ = _[√]_ _n −⌊_ _[√]_ _n⌋_ . The objective value of ( _P_ ) is therefore

- _ni_ =1 _[x]_ _i_ [2] [=] _[ t][·]_ [1][2][+] _[r]_ [2][, leading to] _[ ∥][x][∥]_ [2] [=] - _⌊_ ~~_[√]_~~ _n⌋_ + ( ~~_[√]_~~ _n_ _−⌊_ ~~_[√]_~~ _n⌋_ ) [2] . Since 0 _≤_ _[√]_ _n−⌊_ _[√]_ _n⌋_ _<_ 1, we
have ( _[√]_ _n−⌊_ _[√]_ _n⌋_ ) [2] _≤_ _[√]_ _n−⌊_ _[√]_ _n⌋._ Therefore, _⌊_ _[√]_ _n⌋_ +( _[√]_ _n−⌊_ _[√]_ _n⌋_ ) [2] _≤⌊_ _[√]_ _n⌋_ + _[√]_ _n−⌊_ _[√]_ _n⌋_ = _[√]_ _n._
Taking square roots gives - _⌊_ ~~_[√]_~~ _n⌋_ + ( ~~_[√]_~~ _n_ _−⌊_ ~~_[√]_~~ _n⌋_ ) [2] _≤_ - ~~_√_~~ _n_ = _n_ 1 _/_ 4 _._


_(iv)_ With steps (i)–(iii), we have shown that, for every _x_ _∈_ R _[n]_ with max _{∥x∥∞,_ ~~_√_~~ 1 _n_ _∥x∥_ 1 _}_ _≤_ 1,

we have _∥x∥_ 2 _≤_ _n_ [1] _[/]_ [4] . For an arbitrary _x_ = 0 (assuming this is w.l.o.g. since, for _x_ = 0, the


17


congruence we are trying to prove is trivially satisfied), let _y_ := max _{||x||∞_ 1 _,_ ~~_√_~~ 1 _n ||x||_ 1 _}_ _[x]_ [.] [Clearly,]

max _{||y||∞,_ ~~_√_~~ 1 _n_ _||y||_ 1 _}_ = 1. Thus, _y_ _∈_ _S_ and, thus, _||y||_ 2 _≤_ _n_ 14 . Therefore


1
_∥x∥_ 2 = max _{||x||∞,_ ~~_√_~~ [1] _||x||_ 1 _}∥y∥_ 2 _≤_ max _{||x||∞,_ ~~_√_~~ [1] _||x||_ 1 _} n_ 4 _._
_n_ _n_


It follows that


which concludes the proof.


_n_ _[−]_ 4 [1] _∥x∥_ 2 _≤_ max _{||x||∞,_ ~~_√_~~ [1] _||x||_ 1 _},_

_n_


**Corollary 2.** _Combining Lemma 4 with Theorem 1, the multi-norm relaxation k_ -HC(multi _,_ 1) _satisfies_

~~_√_~~ 1 _n_ OPT� _k_ -HC(2 _,_ 1)� _≤_ OPT� _k_ -HC(multi _,_ 1)� _≤_ OPT� _k_ -HC(2 _,_ 1)� _._


_Proof._ A direct consequence of applying Theorem 1 to the congruence relationship derived in
Lemma 4.


**Proposition** **2.** _Under_ _Assumption_ _1,_ _when_ _solving_ _k_ -HC(2 _,_ 1) _a_ _nonzero_ _lower_ _bound_ _is_ _obtained_
_only after generating at least_ 2 _[k]_ [(] _[n][−]_ [1)] _branching nodes._


_Proof._ By assumption, each branching operation decides the sign of a component of _wj_ for some
_j_ _∈_ [ _k_ ] by splitting (with a half-space constraint) its feasible region with a hyperplane containing the
origin. As long as the cone, call it _C_, obtained by intersecting such half-spaces is not pointed, the
convex hull of its intersection with the feasible region of the problem contains the origin. Thus, the
solution with ( _wj, γj_ ) = 0 and _xij_ = 1, _i ∈_ [ _m_ ], which coincides with assigning every data point to
the degenerate hyperplane of index _j_ (thus achieving _di_ = 0, _i_ _∈_ [ _m_ ]), is optimal regardless of the
convex envelope that is employed. Only after branching has been carried out on each component of
_wj_ whose sign is not already restricted by the symmetry-breaking constraint (i.e., all coordinates except _wj_ 1) for each _j_ _∈_ [ _k_ ], the cone _C_ becomes pointed and, thus, the convex hull of its intersection
with the feasible region of the problem renders the trivial solution ( _wj, γj_ ) = 0, _j_ _∈_ [ _k_ ], infeasible,
allowing for the calculation of a nonzero lower bound. This requires generating at least 2 _[k]_ [(] _[n][−]_ [1)]
nodes by branching on each hyperplane _n −_ 1 times (rather than _n_ due to the symmetry breaking
constraint being already imposed). Notice that, in the general case, more branching operations are
needed due to the _x_ variables being binary.


**Proposition 3.** _Assume that the constraint ∥wj∥_ 1 _≥_ 1 _, j_ _∈_ [ _k_ ] _, is imposed and that branching takes_
_place on the sjh_ _variables first._ _Then, a nonzero global lower bound is obtained after generating at_
_least_ 2 _[k]_ [(] _[n][−]_ [1)] _nodes._ _If k_ -HC( _∞,_ 1) _is being solved, no further branching on w takes place._


_Proof._ Let’s consider a hyperplane of index _j_ _∈_ [ _k_ ]. Let _sjh_ = [1]


_Proof._ Let’s consider a hyperplane of index _j_ _∈_ [ _k_ ]. Let _sjh_ = 2 [for] [all] _[h]_ _[∈]_ [[] _[n]_ []][,] [which] [implies]

_wjh_ [+] _[≤]_ 2 [1] [and] _[ w]_ _jh_ _[−]_ _[≤]_ [1] 2 [.] [Letting] _[ w]_ _jh_ [+] [=] _[w]_ _jh_ _[−]_ [=] [1] 2 [, we have] _[ w]_ _jh_ [+] [+] _[ w]_ _jh_ _[−]_ [=] [1][.] [This feasible solution]


_wjh_ [+] _[≤]_ 2 [1] [and] _[ w]_ _jh_ _[−]_ _[≤]_ [1] 2 [.] [Letting] _[ w]_ _jh_ [+] [=] _[w]_ _jh_ _[−]_ [=] [1] 2 [, we have] _[ w]_ _jh_ [+] [+] _[ w]_ _jh_ _[−]_ [=] [1][.] [This feasible solution]

trivially satisfies the 1-norm constraint equation 1d with _wjh_ [+] _[−]_ _[w]_ _jh_ _[−]_ [=] _[ w][jh]_ [= 0][. Thus,][ (] _[w][j][, γ][j]_ [) = 0][,]
_j_ _∈_ [ _k_ ], is optimal. By branching on a variable _sjh_, we impose either _wjh_ _≤_ 0 (with _sjh_ = 0) or
_wjh_ _≥_ 0 (with _sjh_ = 1). In both cases, the solution where _wjh_ [+] [=] _[w]_ _jh_ _[−]_ [=] [1] 2 [and] _[ w][jh]_ [=] [0][ becomes]


2 [1] [and] _[ w]_ _jh_ _[−]_ _[≤]_ [1] 2


[1] 2 [.] [Letting] _[ w]_ _jh_ [+] [=] _[w]_ _jh_ _[−]_ [=] [1] 2


_wjh_ _≥_ 0 (with _sjh_ = 1). In both cases, the solution where _wjh_ [=] _[w]_ _jh_ [=] 2 [and] _[ w][jh]_ [=] [0][ becomes]

infeasible due to either _wjh_ [+] [or] _[ w]_ _jh_ _[−]_ [being forced to 0;] [the solution with] _[ w][jh][′]_ [=] [0][,] [though,] [for any]
other _h_ _[′]_ _∈_ [ _n_ ] _\{h}_, remains feasible as long as branching on it has not taken place. Thus, a nonzero
lower bound is obtained after 2 _[k]_ [(] _[n][−]_ [1)] branching nodes have been generated. If _k_ -HC( _∞,_ 1) is being
solved, when such an exponentially-large tree of depth _k_ ( _n −_ 1) is complete, though, _∥wj∥_ 1 _≥_ 1,
_j_ _∈_ [ _k_ ], holds in each leaf node and, thus, no further branching on _w_ is necessary.


**Proposition 4.** _Assume that ∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_ _, j_ _∈_ [ _k_ ] _, is imposed and that branching takes place on_
_the ujh variables first._ _Then, k_ ( _n_ _−_ 1) _nodes suffice to obtain a nonzero lower bound._ _If k_ -HC(1 _,_ ~~_√_~~ 1 _n_ )
_is being solved, no further branching on w takes place._


18


_Proof._ After branching on _ujh_ for any pair _j, h_, the (left, w.l.o.g.) child node with _ujh_ = 1 satisfies
_wjh_ _≥_ ~~_√_~~ 1 _n_ . This guarantees _||wj||∞_ _≥_ ~~_√_~~ 1 _n_ and, thus, no further branching is needed on _wj_ in the
descendants of the left node. Further branching operations on _wj_ are only necessary on the right
child node where _ujh_ = 0 has been imposed. By iteratively applying this reasoning _n −_ 1 times
(recall that no disjunction is imposed on _w_ 1 _j_ due to symmetry breaking) for each _j_ _∈_ [ _k_ ], we obtain
a tree with exactly two nodes per level (except for the root node) where each left node satisfies
the _||wj||∞_ _≥_ ~~_√_~~ 1 _n_ constraint for at least a _j_ _∈_ [ _k_ ]. Therefore, when the tree has depth _k_ ( _n −_ 1),
_||wj||∞_ _≥_ ~~_√_~~ 1 _n_ is satisfied for all _j_ _∈_ [ _k_ ]. When such an polynomially-sized tree of depth _k_ ( _n_ _−_ 1) is
complete, _∥wj∥∞_ _≥_ ~~_√_~~ 1 _n_, _j_ _∈_ [ _k_ ], holds in each leaf node and, thus, if _khcTwo_ 1 ~~_√_~~ [1] _n_ is being solved,
no further branching on _w_ is necessary.


D PROOF OF THE APPROXIMATION FACTORS AND OF THEIR TIGHTNESS


We will rely on the following Lemma:
**Lemma 5.** _Given two functions f, g_ : R _[n]_ _→_ R _with g surjective we have:_


- _f_ ( _x_ ) ��
: _g_ ( _x_ ) = _ν_ _._ (12)
_ν_


_f_ ( _x_ )
max
_x∈_ R _[n]_ _g_ ( _x_ ) [= max] _ν∈_ R


max
_x∈_ R _[n]_


_If, for all x ∈_ R _[n]_ _, f_ ( _x_ ) = _f_ ( _|x|_ ) _and g_ ( _x_ ) = _g_ ( _|x|_ ) _, then:_


- _f_ ( _x_ ) ��
: _g_ ( _x_ ) = _ν_ _._ (13)
_ν_


_f_ ( _x_ )
max [max]
_x∈_ R _[n]_ _g_ ( _x_ ) [=] _ν∈_ R+


max
_x∈_ R _[n]_ +


_Proof._ If _g_ is surjective, then _∪ν∈_ R _{x ∈_ R _[n]_ : _g_ ( _x_ ) = _ν}_ = R _[n]_ . We can therefore partition R _[n]_ into
infinitely many subsets of type _{x_ _∈_ R _[n]_ : _g_ ( _x_ ) = _ν}_ . An optimal solution to max _x∈_ R _n_ _[f]_ _g_ ( [(] _x_ _[x]_ ) [)] [thus]

corresponds to the best solution over all such subsets. The special case in equation 13 follows by a
similar argument.


**Proposition 5.** _The following relationships are satisfied for every x ∈_ R _[n]_ _:_


_∥x∥_ 2 _≤∥x∥_ 1 _≤_ _[√]_ _n∥x∥_ 2
1
~~_√_~~ _∥x∥_ 2 _≤∥x∥∞_ _≤∥x∥_ 2
_n_


_and the factors_ _[√]_ _n_ _and_ ~~_√_~~ 1 _are tight._
_n_


_Proof._ We are looking for four positive coefficients _α_ 1 _, β_ 1 _, α∞, β∞_ that satisfy the following relationships for all _x ∈_ R _[n]_ :


_α_ 1 _∥x∥_ 2 _≤∥x∥_ 1 _≤_ _β_ 1 _∥x∥_ 2
_α∞∥x∥_ 2 _≤∥x∥∞_ _≤_ _β∞∥x∥_ 2 _._


Assuming _x_ = 0 as, for _x_ = 0, _α∥x∥p_ _≤∥x∥q_ _≤_ _β∥x∥p_ holds for all _α, β_ and for all _p, q_ _∈_
N _∪{∞}_, the tightest values for _α_ 1 _, β_ 1 _, α∞, β∞_ must satisfy the following relationships:

_∥x∥_ 1 _∥x∥∞_
_β_ 1 = max _β∞_ = max
_x∈_ R _[n]_ _∥x∥_ 2 _x∈_ R _[n]_ _∥x∥_ 2

_∥x∥_ 1 _∥x∥∞_
_α_ 1 = min _α∞_ = min _._
_x∈_ R _[n]_ _∥x∥_ 2 _x∈_ R _[n]_ _∥x∥_ 2


As it is not hard to see, max _[∥][x][∥][p]_


_[∥][x][∥][p]_ _[∥][x][∥][q]_

_∥x∥q_ [= min] _∥x∥p_


As it is not hard to see, max _∥x∥_ _[p]_ _q_ [= min] _∥x∥p_ _[q]_ [holds for all] _[ p, q]_ _[∈]_ [N] _[ ∪{∞}]_ [.] [Thus, we need to solve]

the following four problems:


_β_ 1 = max _∥_ _[∥]_ _x_ _[x]_ _∥_ _[∥]_ 2 [1]

_α_ 1 = max _[∥][x][∥]_ [2]


_[∥][x][∥]_ [1] _β∞_ = max _[∥][x][∥][∞]_

_∥x∥_ 2 _∥x∥_ 2


_._
_∥x∥∞_


_∥x∥_ 2


_[∥][x][∥]_ [2] _α∞_ = max _[∥][x][∥]_ [2]

_∥x∥_ 1 _∥x∥∞_


19


Let us consider the case of _α_ 1 _, α∞_, for which we are solving max _∥_ _[∥]_ _x_ _[x]_ _∥_ _[∥]_ _q_ [2] [for] _[ q]_ [=] [1] _[,][ ∞]_ [.] [By virtue of]

Lemma 5, we are thus solving:


_αq_ = max
_ν∈_ R+


- 1 _{∥x∥_ 2 : _∥x∥q_ = _ν}_ _._
_ν_ _x_ [max] _∈_ R _[n]_ +


As the maximum of a convex function (such as _∥x∥_ 2) over a closed, convex set is achieved on the
border of the latter and, if we are optimizing over a polytope, over its extreme vertices, we can
w.l.o.g. relax _∥x∥q_ = _ν_ into _∥x∥q_ _≤_ _ν_ .

For _α_ 1, the extreme points of _{x_ _∈_ R _[n]_ : _∥x∥_ 1 _≤_ _ν}_ are of the form: _νeℓ_ for _√_ all _ℓ_ _∈_ [ _n_ ], with
_eℓ_ being the _ℓ_ -th canonical vector of R _[n]_ . For each of them, we have _∥νeℓ∥_ 2 = _ν_ [2] = _ν_ . Thus,


_eℓ_ being the _ℓ_ -th canonical vector of R _[n]_ . For each of them, we have _∥νeℓ∥_ 2 = _ν_ [2] = _ν_ . Thus,

_α_ 1 = max _[∥]_ _∥_ _[x]_ _x_ _[∥]_ _∥_ [2] 1 [=] _[ν]_ _ν_ [= 1][.]

For _α∞_, the extreme points of _{x_ _∈_ R _[n]_ : _∥x∥∞_ _≤_ _ν}_ are of the form: _√_ ( _±ν, . . ., ±ν_ ) for all
possible choices of _±_ . For each of them, we have _∥_ ( _±ν, . . ., ±ν_ ) _∥_ 2 = _ν_ [2] _n_ = _ν_ _[√]_ _n_ . Thus,


_[∥][x][∥]_ [2] _[ν]_

_∥x∥_ 1 [=] _ν_


_ν_ [= 1][.]


_√_
possible _α∞_ = maxchoices _∥_ _[∥]_ _x_ _[x]_ _∥_ _[∥]_ _∞_ [2] of [=] _±_ _[ν]_ . ~~_[√]_~~ _ν_ _[n]_ For= each _[√]_ _n_ . of them, we have _∥_ ( _±ν, . . ., ±ν_ ) _∥_ 2 = _ν_ [2] _n_ = _ν_ _[√]_ _n_ . Thus,

Let us now consider the case of _β_ 1 and _β∞_, for which we are solving max _[∥]_ _∥_ _[x]_ _x_ _[∥]_ _∥_ _[q]_ 2 [for] _[ q]_ [=] [1] _[,][ ∞]_ [.] [By]

virtue of Lemma 5, we are thus solving:


_[∥][x][∥]_ [2] _[ν]_ ~~_[√]_~~ _[n]_

_∥x∥∞_ [=] _ν_


~~_[√]_~~ _ν_ _[n]_ = _[√]_ _n_ .


Let us now consider the case of _β_ 1 and _β∞_, for which we are solving max _[∥]_ _∥_ _[x]_ _x_ _[∥]_ _∥_ _[q]_


- 1 _{∥x∥q_ : _∥x∥_ 2 = _ν}_ _._
_ν_ _x_ [max] _∈_ R _[n]_ +


- 1
_ν_ _x_ [max] _∈_ R _[n]_ +


For _β_ 1, the problem reads:


_βq_ = max
_ν∈_ R+


_β_ 1 = max
_ν≥_ 0


- _e_ _[T]_ _x_ : _x_ _[T]_ _x_ = _ν_ [2][��] _._ (14)


The KKT conditions for the relaxation of the inner problem of equation 14 obtained after dropping
the nonnegativity on _x_ read:


_∇x_ ( _e_ _[T]_ _x −_ _λ_ ( _x_ _[T]_ _x −_ _ν_ [2] )) = 0

_x_ _[T]_ _x_ = _ν_ [2] _,_

with _λ_ unrestricted in sign. From the first equation, ~~_√_~~ we deduce _x_ = 2 _eλ_ [.] [By] [substituting] [it] [in] [the]
second equation, we obtain 2 _e_ [2] _[T]_ _λe_ [2] [=] _[ν]_ [2][, that is,] _[ λ]_ [=] 2 _νn_ [.] [Thus, we have] _[ x]_ [=] ~~_√_~~ _en_ _ν_ . Since the latter


second equation, we obtain 2 _e_ [2] _λe_ [2] [=] _[ν]_ [2][, that is,] _[ λ]_ [=] 2 _νn_ [.] [Thus, we have] _[ x]_ [=] ~~_√_~~ _en_ _ν_ . Since the latter

is nonnegative, it is an optimal solution to both the relaxation of the inner problem of equation 14
with _x ∈_ R _[n]_ and its unrelaxed version with _x ∈_ R _[n]_ + [.] [We thus have] _[ ∥][x][∥]_ [1] [=] ~~_√_~~ _νn_ _∥e∥_ 1 = ~~_√_~~ _[νn]_ _n_ = _ν_ _[√]_ _n_ .


We conclude that _β_ 1 = _[ν]_ ~~_[√]_~~ _[n]_


_ν_ _[n]_ = _[√]_ _n_ .


For _β∞_, the problem reads:


_β∞_ = max
_ν≥_ 0


- 1
_ν_ _x_ [max] _∈_ R _[n]_ +


- ��
max _._
_ℓ∈_ [ _n_ ] _[{][x][ℓ][}]_ [ :] _[ x][T][ x]_ [ =] _[ ν]_ [2]


The optimal solutions to the inner problem are of the form _νeℓ_, where _eℓ_ is a canonical vector of
R _[n]_, for which we have _∥νeℓ∥∞_ = _ν_ . We conclude that _β∞_ = _[ν]_ _ν_ [= 1][.]


20
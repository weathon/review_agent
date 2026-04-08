# IMPROVED ℓp REGRESSION VIA ITERATIVELY REWEIGHTED LEAST SQUARES


**Adrian Vladu**
CNRS & IRIF
Universit´e Paris Cit´e
vladu@irif.fr


**Alina Ene**
Department of Computer Science
Boston University
aene@bu.edu


**Ta Duy Nguyen**
Department of Computer Science
Boston University
taduy@bu.edu


ABSTRACT


We introduce fast algorithms for solving _ℓp_ regression problems using the iteratively
reweighted least squares (IRLS) method. Our approach achieves state-of-the-art
iteration complexity, outperforming the IRLS algorithm by Adil-Peng-Sachdeva
(NeurIPS 2019) and matching the theoretical bounds established by the complex
algorithm of Adil-Kyng-Peng-Sachdeva (SODA 2019, J. ACM 2024) via a simpler
lightweight iterative scheme. This bridges the existing gap between theoretical and
practical algorithms for _ℓp_ regression. Our algorithms depart from prior approaches,
using a primal-dual framework, in which the update rule can be naturally derived
from an invariant maintained for the dual objective. Empirically, we show that
our algorithms significantly outperform both the IRLS algorithm by Adil-PengSachdeva and MATLAB/CVX implementations.


1 INTRODUCTION


In this paper, we study the _ℓp_ regression problem defined as follows. The input to the problem is a
matrix _A ∈_ R _[d][×][n]_ _,_ a vector _b ∈_ R _[d]_ that lies in the column span of _A_, and an accuracy parameter _ϵ_ .
The goal is to approximately solve the problem min _x∈_ R _[n]_ : _Ax_ = _b ∥x∥p_, i.e., find a solution _x_ _∈_ R _[n]_
such that _Ax_ = _b_ and _∥x∥p_ _≤_ (1 + _ϵ_ ) _∥x_ _[∗]_ _∥p_, where _x_ _[∗]_ is an optimal solution to the problem, and
_∥·∥p_ denotes the _ℓp_ norm. Solving _ℓp_ regression for all values of _p_ is a fundamental problem in
machine learning with numerous applications and has been studied in a long line of research beyond
the classical least squares regression with _p_ = 2. _Lp_ -norm regression problems with general _p_ arise
in several areas, including supervised learning, graph clustering, and wireless networks. Examples of
applications include _ℓp_ -norm based algorithms in semi-supervised learning (Alaoui, 2016; Liu and
Gleich, 2020), _k_ -clustering with _ℓp_ -norm (Huang and Vishnoi, 2020), robust regression and robust
clustering (Meng and Mahoney, 2013; Huang et al., 2023).


For this general class of convex optimization problems, designing provably fast iterative algorithms
to obtain high accuracy solutions with empirical efficiency is an important question. General convex
programming methods such as interior point methods are usually slow in practice. In theory, Bubeck
et al. (2018) show that algorithms based on interior point methods cannot improve beyond _O_ ( _[√]_ _n_ )
iterations [1] for any _p_ _∈{/_ 1 _,_ 2 _, ∞}_ . Breaking this barrier and finding iterative algorithms that are faster
than interior point methods both in theory and practice is the goal of this line of work.


Recent developments have led to new algorithmic approaches such as a homotopy method (Bubeck
et al., 2018), and an iterative refinement approach (Adil et al., 2019a;b; 2024) for _ℓp_ regression
with _p_ _∈{/_ 1 _, ∞}_ . We highlight the notable works by Adil et al. (2019a;b; 2024). On the one hand,
the algorithm with the best known theoretical runtime is given by Adil et al. (2019a; 2024) with


1For simplicity in the introduction, we assume that _d_ = Θ( _n_ ). In the regime when _n ≫_ _d_, the IPM iteration
~~_√_~~
complexity improves to _O_ ( _d_ ).

[�]


1


_O_ - _p_ [2] _n_ 3 _pp−−_ 22 log - _nϵ_ �� calls [2] to a linear system solver. This algorithm, however, relies on complex

subroutines and includes theoretical choices for several hyperparameters. In practice, to obtain an
efficient implementation, hyperparameters require tuning. Due to these reasons, this theoretical
algorithm by Adil et al. (2019a; 2024) does not provide a practical implementation. On the other hand,
an algorithm known as _p_ -IRLS by Adil et al. (2019b) has been shown to have significant speed up over
standard solvers such as CVX. This algorithm is implemented based on an Iteratively Reweighted
Least Squares (IRLS) method, which is a general iterative framework for solving regression problems.
The key element of an IRLS method is solving a weighted least squares regression problem in each
iteration. This is equivalent to solving a linear system of the form min _x∈_ R _n_ : _Ax_ = _b x_ _[⊤]_ _Rx_, where _R_ is
a diagonal matrix, which can be computed very efficiently in practice with the advance of numerical
solvers. IRLS algorithms are favored in practice (Burrus, 2012), but designing IRLS algorithms with
strong convergence guarantees is challenging. In particular, to obtain the efficiency, the algorithm by
Adil et al. (2019b) sacrifices the theoretical guarantee, requiring _O_ - _p_ [3] _n_ 2 _pp−−_ 22 log - _n_ �� linear system


_O_ - _p_ [2] _n_


Adil et al. (2019b) sacrifices the theoretical guarantee, requiring _O_ - _p_ [3] _n_ 2 _pp−−_ 22 log - _nϵ_ �� linear system

solves. This brings forth the question:


_Can we design an algorithm that retains the empirical efficiency of an IRLS approach while_
_achieving the state-of-the-art theoretical runtime?_


In this work, we give a positive answer to this question. We provide a new algorithmic framework for
_ℓp_ regression based on an IRLS approach for all values of _p ∈_ (1 _, ∞_ ). We propose an algorithm that
uses _O_ - _p_ [2] _n_ 3 _pp−−_ 22 log - _nϵ_ �� linear system solves, matching the state-of-the-art theoretical algorithm

by Adil et al. (2019a), and improving upon the guarantee of _O_ - _p_ [3] _n_ 2 _pp−−_ 22 log - _nϵ_ �� for the _p_ -IRLS

algorithm by Adil et al. (2019b). We experimentally compare our algorithm with the _p_ -IRLS algorithm
(Adil et al., 2019b) and CVX solvers, and we observe significant improvements in all instances.


3 _pp−−_ 22 log - _nϵ_ �� linear system solves, matching the state-of-the-art theoretical algorithm


by Adil et al. (2019a), and improving upon the guarantee of _O_ - _p_ [3] _n_


1.1 OUR CONTRIBUTIONS


For the simplicity of the exposition, we study the _ℓp_ regression problem in both low and high precision
regimes for _p ≥_ 2.
_Remark_ 1.1 _._ In Appendix B, we show a simple reduction for the more general problem
min _x_ : _Ax_ = _b ∥Nx −_ _v∥p_ to the form min _x_ : _Ax_ ˜ = [˜] _b_ _[∥][x][∥][p]_ [with the dependence of the runtime on the]
number of rows of _N_ instead of the dimension of _x_ . We also show in Appendix C a reduction for the
case 1 _< p <_ 2 to the case _p ≥_ 2.


In the low precision regime when the runtime dependence on _ϵ_ is poly - 1 _ϵ_ �, we have the following
theorem.


**Theorem** **1.1.** _For_ _any_ _p_ _≥_ 2 _,_ _there_ _is_ _an_ _iterative_ _algorithm_ _for_ _the_ _ℓp_ _regression_ _problem_
min _x∈_ R _[n]_ : _Ax_ = _b ∥x∥p_ _that_ _solves_ _O_ (log log _n_ + log (1 _/ϵ_ )) _subproblems,_ _each_ _of_ _which_ _makes_


_O_ ��( [1] _ϵ_ [)]


3 _p_ 3 [2] _p−_ [2] _−_ 8 _p_ 4+4 _p_ - log - _n_ ~~_p_~~


_n_ - [�]

~~_p_~~ _calls_ _to_ _solve_ _a_ _linear_ _system_ _of_ _the_ _form_
_ϵ_ _p−_ 2


2 _p−_ 3

_p−_ 2 + _n_


_p−_ 2
3 _p−_ 2 ( [1] _ϵ_ [)]


_ADA_ _[⊤]_ _ϕ_ = _b, where D is an arbitrary non-negative diagonal matrix._


1
_Remark_ 1.2 _._ When _p_ = _∞_, each subproblem makes _O_ - _ϵ_ 1 [2] [+] _[n]_ _ϵ_ 3 [log(] _[n]_ _ϵ_ [)] - calls to a linear system

solver.


Prior approaches for solving _ℓp_ regression problem in the low precision regime commonly use the
Taylor expansion of _∥x∥_ _[p]_ _p_ [,] [which then allows for deriving and bounding the updates.] [In contrast]
to this, our algorithm relies on a primal-dual approach using the dual formulation of the squared
objective min _x_ : _Ax_ = _b ∥x∥_ [2] _p_ [= min] _[x]_ [:] _[Ax]_ [=] _[b]_ _[∥][x]_ [2] _[∥]_ _p/_ 2 [= max] _[r]_ _∥Er_ ( _∥rq_ ) [where] _[ ℓ][q]_ [is the dual norm of] _[ ℓ][p/]_ [2]
and _E_ ( _r_ ) = min _x_ : _Ax_ = _b⟨r, x_ [2] _⟩_ . The term _E_ ( _r_ ) is often referred to as the energy. The high level idea
of our approach is as follows. Starting with an initial solution _r_ for the dual problem, we will increase
the coordinates of _r_ as much as possible so that the increase in the energy _E_ ( _r_ ) relative to the increase


_p_ _p_
2The original result is _O_ - _pn_ 3 _pp−−_ 22 log - _∥x_ [(0)] _∥pϵ_ _[−][∥][x][∗][∥]_ _p_ �� for finding � _x_ such that _∥x_ - _∥_ _[p]_ _p_ _[≤]_ [min] _[x]_ [:] _[Ax]_ [=] _[b][ ∥][x][∥]_ _p_ _[p]_ [+]


_p_ _p_
_ϵ_ . This translates to _O_ - _pn_ 3 _pp−−_ 22 log - _∥x_ [(0)] _pϵ∥∥px_ _[−][∗][∥]_ _∥_ ~~_[p]_~~ _p_ _[x][∗][∥]_ _p_ �� = _O_ - _p_ [2] _n_ 3 _pp−−_ 22 log - _nϵ_ �� for finding � _x_ such that _∥x_ - _∥p_ _≤_


(1 + _ϵ_ ) min _x_ : _Ax_ = _b ∥x∥p_ for _x_ [(0)] initialized to min _x_ : _Ax_ = _b ∥x∥_ 2.


2


of _∥r∥q_ is also sufficiently large, until we can obtain a (1 _−_ _ϵ_ ) optimal dual solution and whereby
recover an approximately optimal primal solution. This template is close to the approach for _ℓ∞_
regression by Ene and Vladu (2019). However, _ℓp_ regression does not have the readily decomposable
structure along the coordinates as _ℓ∞_ regression and novel technique is required in the design of
the algorithm. Our approach is also a reminiscence of the width-independent multiplicative weights
update method for solving mixed packing covering linear program, where in each step the algorithm
updates the coordinates the maximize the bang-for-buck ratio (Quanrud, 2020). In contrast to MWU,
we do not use a mirror map or regularize _ℓp_ norms to make them smooth as in standard approaches.
Our scheme allows our method to take much longer steps, where in each step, the coordinates of the
dual solution are allowed to change by large polynomial factors and thereby achieve faster running
time.


To obtain faster algorithms in the high accuracy regime with a logarithmic dependence on the accuracy,
we adapt the iterative refinement approach of Adil et al. (2019a) and obtain improved running times.


**Theorem** **1.2.** _For_ _any_ _p_ _≥_ 2 _,_ _there_ _is_ _an_ _iterative_ _algorithm_ _for_ _the_ _ℓp_ _regression_ _problem_
min _x∈_ R _n_ : _Ax_ = _b ∥x∥p that solves O_ - _p_ [2] log _n_ log - _nϵ_ �� _subproblems, each of which makes O_ - _n_ 3 _pp−−_ 22 [�]

_calls_ _to_ _solve_ _a_ _linear_ _system_ _of_ _the_ _form_ _ADA_ _[⊤]_ _ϕ_ = _z,_ _where_ _D_ _is_ _an_ _arbitrary_ _non-negative_

[�] [�]
_diagonal matrix,_ _A is a matrix obtained from A by appending a single row, and z is a vector obtained_

[�]
_from the all-zero vector by appending a single non-zero coordinate._


Using the iterative refinement template by (Adil et al., 2019a;b; 2024), we instead use an IRLS solver
for the residual problems with improved runtime. The residual solver solves a mixed _ℓp_ + _ℓ_ 2 problem
in the form min _x_ : _Ax_ = _b ∥x∥_ [2] _p_ [+] - _θ, x_ [2][�], only to a constant approximation. Here the challenge lies
in the fact that the _ℓ_ 2 term makes the dual problem no longer scale-free and thus our low precision
solver is not immediately usable. However, by an appropriate initialization of the dual solution and
careful adjustments to the step size, our algorithm achieves the desired _O_ - _n_ 3 _pp−−_ 22 [�] bound. Since

regularized _ℓp_ + _ℓ_ 2 regression problems arise in many applications in machine learning and beyond,
our algorithm for the mixed _ℓp_ + _ℓ_ 2 objective is of independent interest.


Finally, we experimentally evaluate our high-precision algorithm. Our algorithm significantly
outperforms the _p_ -IRLS algorithm (Adil et al., 2019a) both in the number of linear system solves as
well as the overall running time. Our algorithm is significantly faster than CVX solvers and is able to
run on large instances, which is not possible for CVX solvers within a time constraint.


1.2 RELATED WORK


_ℓp_ regression problems have received significant attention. Here we summarize the results that are
closest to our work. The surveyed algorithms are iterative algorithms where the running time of each
iteration is dominated by a single linear system solve.

Algorithms based on interior point methods use _O_ ( _[√]_ _n_ ) iterations for any _p_ _∈_ [1 _, ∞_ ] (Nesterov

[�]                                                        - _√_                                                        and Nemirovskii, 1994), which was improved to _O_ _d_ iterations for _p_ _∈{_ 1 _, ∞}_ (Lee and

[�]
Sidford, 2014). Bubeck-Cohen-Lee-Li (Bubeck et al., 2018) show that this iteration bound is
generally necessary for interior point methods and propose a homotopy-based algorithm that uses
_O_ ˜�poly� _pp−_ 21 - _· n_ _[|]_ [1] _[/]_ [2] _[−]_ [1] _[/p][|]_ [�] iterations for any _p_ _∈{/_ 1 _, ∞}_ . Adil et al. (2019a; 2024) introduced


an iterative refinement framework that uses _O_ - _p_ [2] _· n_


_p−_ 2
3 _p−_ 2 log( _[n]_


an iterative refinement framework that uses _O_ - _p_ [2] _· n_ 3 _p−_ 2 log( _[n]_ _ϵ_ [)] - iterations for any _p_ _>_ 2. Using

Lewis weight sampling, Jambulapati-Liu-Sidford (Jambulapati et al., 2022) improve the method
by Adil et al. (2019a; 2024) to _O_ - _p_ _[p]_ _· d_ 3 _pp−−_ 22 polylog( _[n]_ [)] �, for overconstrained regression problems


_p−_ 2
3 _p−_ 2 polylog( _[n]_


by Adil et al. (2019a; 2024) to _O_ - _p_ _[p]_ _· d_ 3 _p−_ 2 polylog( _[n]_ _ϵ_ [)] �, for overconstrained regression problems

min _x∈_ R _d ∥Ax −_ _b∥p_ where _A_ _∈_ R _[n][×][d]_ and _n_ is much larger than _d_ (the iteration complexity of
the prior algorithms will still depend on the larger dimension _n_ in this case). Bullins (2018) gives
a faster algorithm for minimizing structured convex quartics, which implies an algorithm for _ℓ_ 4
1
regression with _O_ [˜] ( _n_ 5 ) iterations. Building on the work of Christiano et al. (2011); Chin et al.
(2013) for maximum flows and regression, Ene and Vladu (2019) give an algorithm for _ℓ_ 1 and _ℓ∞_
regression using _O_ - _n_ 1 _/_ 3 log(1 _ϵ_ [2] _[/]_ [3] _/ϵ_ ) + [log] _ϵ_ [2] _[ n]_ - iterations. This work also uses a primal-dual framework


regression using _O_ - _n_ _ϵ_ [2] _[/]_ [3] _/ϵ_ ) + [log] _ϵ_ [2] _[ n]_ - iterations. This work also uses a primal-dual framework

but the algorithm and analysis are specific to the special structure of the _ℓ_ 1 and _ℓ∞_ norm and work
only in the low precision regime with poly( [1] [)][ convergence.]


_ϵ_ [)][ convergence.]


3


**Algorithm 1** _ℓ_ 2 _p_ -minimization( _A, b, ϵ_ )


**Input:** Matrix _A ∈_ R _[d][×][n]_, vector _b ∈_ R _[d]_, accuracy _ϵ_
**Output** : Vector _x_ such that _Ax_ = _b_ and _∥x∥_ 2 _p_ _≤_ (1 + _ϵ_ ) min _x_ : _Ax_ = _b ∥x∥_ 2 _p_
Initialize _x_ [(0)] = min _x_ : _Ax_ = _b ∥x∥_ 2


   _L_ = max _i_ : (1 + _ϵ_ ) _[i]_ _≤_ _[∥][x]_ 1 [(0)] _[∥]_ [1][2]


1
_n_ 2 _[−]_ 2 [1] _p_


2 _p_


; _U_ = min  - _i_ : (1 + _ϵ_ ) _[i]_ _≥_ �� _x_ (0)��2�


**while** _L < U_ :
_P_ = _⌊_ _[L]_ [+] 2 _[U]_ _⌋_, _M_ = (1 + _ϵ_ ) _[P]_

**if** SubSolver( _A, b, ϵ, M_ ) is infeasible **then**
_L_ = _P_ + 1
**else**
Let _x_ [(] _[t]_ [+1)] be the output of SubSolver( _A, b, ϵ, M_ )
_U_ = _P_ ; _t ←_ _t_ + 1
**end if**
**end while**
**return** _x_ [(] _[t]_ [)]


**Algorithm 2** SubSolver( _A, b, ϵ, M_ )

**Input:** Matrix _A ∈_ R _[d][×][n]_, vector _b ∈_ R _[d]_, accuracy _ϵ_, target value _M_
**Output** : Vector _x_ such that _Ax_ = _b_ and _∥x∥_ 2 _p_ _≤_ (1 + _ϵ_ ) _M_,
or approximate infeasibility certificate _r_, _∥r∥q_ = 1.
_t_ = 0, _r_ [(0)] = _n_ [1] 1 _[/q]_ [,] _[ t][′]_ [= 0][,] _[ s]_ [(] _[t][′]_ [)] [= 0]
**while** ��� _r_ [(] _[t]_ [)][���] _q_ _[≤]_ [1] _ϵ_

_x_ [(] _[t]_ [)] = arg min _x_ : _Ax_ = _b⟨r_ [(] _[t]_ [)] _, x_ [2] _⟩_


_γi_ [(] _[t]_ [)] =


- _x_ 2 _i_ _[∥][r][∥]_ _q_ _[q][−]_ [1] _x_ [2] _i_ _[∥][r][∥]_ _q_ _[q][−]_ [1]
if _≥_ (1 + _ϵ_ ) _M_ [2]
_M_ [2] _ri_ _[q][−]_ [1] _ri_ _[q][−]_ [1], for all _i_

1 otherwise


**if** _γ_ [(] _[t]_ [)] = 1 **then return** _x_ [(] _[t]_ [)] **end if** _▷_ _Case 1_

_α_ [(] _[t]_ [)] = - _γ_ [(] _[t]_ [)][�] _q_ [1] ; _r_ ( _t_ +1) = _r_ ( _t_ ) _· α_ ( _t_ )


[1] - 2 _[q]_ _q_ _[−]_ +1 [1] ( _t_ _[′]_ +1) ( _t_ _[′]_ ) ( _t_ ) _′_ _′_

_ϵ_ **then** _s_ = _s_ + _x_ ; _t_ = _t_ + 1 **end if**


2
**if** _α_ [(] _[t]_ [)] _≤_ _n_ 2 _q_ +1 [�] [1] _ϵ_


**if** _t_ _[′]_ _>_ 0 **and** _s_ ( _t′_ ) _/t′_ **[then return]** _[ s]_ [(] _[t][′]_ [)] _[/t][′]_ **[end if]** _▷_ _Case 2_
��� ���2 _p_ _[≤]_ [(1 +] _[ ϵ]_ [)] _[M]_


_t_ = _t_ + 1
**end while**
**return** _r_ [(] _[t]_ [)] _▷_ _Case 3_


                - 1                2 OUR ALGORITHM WITH poly CONVERGENCE
_ϵ_


In this section, we present our algorithm with guarantee provided in Theorem 1.1.


Before describing the algorithm, we first introduce some basic notations. For a constant _a ∈_ R, we
abuse the notation and use _a ∈_ R _[n]_ to denote the vector with all entries equal to _a_ (the dimension will
be clear from context). When it is clear from the context, we apply scalar operations to vectors with
the interpretation that they are applied coordinate-wise. For _p ≥_ 1, we let _q_ be such that _p_ [1] [+] [1] _q_ [= 1]

and _ℓq_ is the dual norm of the _ℓp_ norm.


2.1 OUR ALGORITHM


For ease of notation, it is convenient to consider the following equivalent formulation of the problem:
For _p_ _≥_ 1, we solve min _x_ : _Ax_ = _b ∥x∥_ [2] 2 _p_ [=] [min] _[x]_ [:] _[Ax]_ [=] _[b]_ �� _x_ 2�� _p_ [to] [(1 +] _[ ϵ]_ [)] [multiplicative] [error.] [We]
provide our algorithm in Algorithms 1 and 2. We give an overview of our approach and explain the
intuition in the following section.


4


2.2 OVERVIEW OF OUR APPROACH


Our algorithm is based on a primal-dual approach, starting with the following dual formulation of the
problem. Using _q_ as the dual norm of _p_ and by duality, we write


_x_ :min _Ax_ = _b_ _[∥][x][∥]_ [2] _[p]_ [=] _x_ : [min] _Ax_ = _b_


�� _x_ 2�� _p_ [=] _x_ : [min] _Ax_ = _b_ _r_ : _∥_ max _r∥q≤_ 1 _[⟨][r, x]_ [2] _[⟩]_ _r≥_ 0:max _∥r∥q≤_ 1 _x_ : [min] _Ax_ = _b_ _[⟨][r, x]_ [2] _[⟩]_ [= max] _r≥_ 0 _∥Er_ ( _∥rq_ ) _,_


where we defined _E_ ( _r_ ) := min _x_ : _Ax_ = _b⟨r, x_ [2] _⟩_ . The main part of our algorithm is the subroutine
shown in Algorithm 2, which takes as input a guess _M_ for the optimum value _∥x_ _[∗]_ _∥_ 2 _p_ . To find an
(1 + _ϵ_ ) approximation of the optimum value, the main Algorithm 1 performs a binary search as
follows. Since _x_ [(0)] is initialized to min _x_ : _Ax_ = _b ∥x∥_ 2, we can show that _∥x_ _[∗]_ _∥p_ is contained in the


range - _∥_ _[x]_ 1 [(0)] _∥_ [1] 2


2 [1] _p_ 2 _[,]_ �� _x_ (0)��2


. The algorithm performs binary search over the indices _i_ such that (1 + _ϵ_ ) _[i]_


1
_n_ 2 _[−]_ 2 [1]


is in that range. Note that the main algorithm only needs to perform at most log - log _ϵ n_ - iterations,

each of which makes one call to the subproblem solver.


We now focus on the subproblem when we are given a guess _M_ and a target precision _ϵ_ . The goal is to
find a primal solution _x_ that satisfies _∥x∥_ 2 _p_ _≤_ _M_ (1 + _ϵ_ ) or a dual solution _r_ (infeasibility certificate)


which can certify that min _x_ : _Ax_ = _b ∥x∥_ [2] 2 _p_ _[≥]_ _∥_ _[E]_ _r_ [(] _∥_ _[r]_ [)]


_∥_ _[E]_ _r_ [(] _∥_ _[r]_ _q_ [)] _[≥]_ [(] 1+ _[M]_


which can certify that min _x_ : _Ax_ = _b ∥x∥_ 2 _p_ _[≥]_ _∥r_ [(] _∥_ _[r]_ _q_ [)] _[≥]_ [(] 1+ _[M]_ _ϵ_ [)][2][.] [This lower bound on the optimal value]

of the problem tells us that we can increase the guess _M_ .


The objective function _E_ ( _r_ ) has a very useful monotonicity property: it increases when _r_ increases.
The overall strategy of our algorithm is to start with an initial dual solution _r_ [(0)] (which we initialize
uniformly to _n_ [1] 1 _[/q]_ [) and increase it while maintaining the following invariant]


_E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] ) _≥_ _M_ [2] ( _r_ ( _t_ +1) _r_ ( _t_ ) (1)
��� ��� _q_ _[−]_ ��� ��� _q_ [)] _[,]_


or equivalently,


_E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] )
_≥_ _M_ [2] _._
�� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_


The telescoping property of both sides of (1) will guarantee that, if the algorithm outputs a dual

solution _r_ with sufficiently large _∥r∥q_, this solution will satisfy _E_ ( _r_ ) _≥_ - 1+ _Mϵ_ �2 _∥r∥q_, i.e, _∥_ _[E]_ _r_ [(] _∥_ _[r]_ _q_ [)] _[≥]_

- 1+ _Mϵ_ �2. To maintain the invariant 1, we have two useful bounds for the change in the objective and
dual solution:


    
  - �2
_ri_ [(] _[t]_ [)] _x_ [(] _i_ _[t]_ [)]
_i_


_E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] ) _≥_ 


1 _−_ _ri_ [(] _[t]_ [)]
_ri_ [(] _[t]_ [+1)]


_,_ (2)


�� _r_ ( _t_ +1)�� _q_ 1 _[−]_ �� _r_ ( _t_ )�� _q_ _≥_                - _i_                - _ri_ [(] _q_ _[t]_ [+1)] �� _r_ (� _t_ ) _q_ �� _qq−−_ 1� _ri_ [(] _[t]_ [)]                - _q_ _._ (3)


Both inequalities allow us to decompose the invariant along the coordinates. That is, we can maintain
the invariant by ensuring for each coordinate _i_ that we increase that


_q_ �� _r_ ( _t_ )�� _qq−_ 1 _ri_ [(] _[t]_ [)] - _x_ [(] _i_ _[t]_ [)] �2

 - - _q_ - - _q_
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


1 _−_ _ri_ [(] _[t]_ [)]
_ri_ [(] _[t]_ [+1)]


_≥_ _M_ [2] _._


_q−_ 1
In order to do this, we update each _ri_ [(] _[t]_ [)] multiplicatively, via the term _γi_ [(] _[t]_ [)] = _[∥]_ - _[r]_ [(] _[t]_ [)] - _[∥]_ _qq−_ 1 _·_

_ri_ [(] _[t]_ [)]


- �2
_xM_ [(] _i_ _[t]_ [)][2] .


To guarantee fast convergence, we want to increase _ri_ [(] _[t]_ [)] as much as possible, by setting a target

                         - �1 _/q_
threshold on _γi_ [(] _[t]_ [)][:] [if] _[ γ]_ _i_ [(] _[t]_ [)] exceeds the threshold, we update _ri_ [(] _[t]_ [+1)] = _ri_ [(] _[t]_ [)] _γi_ [(] _[t]_ [)] ; otherwise, _ri_ [(] _[t]_ [)]


5


remains unchanged. When we can no longer increase _r_ while preserving the invariant, we can be
sure that we have found the corresponding primal solution _x_ with small norm. During the course of
the algorithm, we also keep track of iterations with small increases in _r_ and use the uniform average
over the corresponding primal solutions to obtain an approximately feasible primal solution, in case
the algorithm fails to return an infeasibility certificate quickly enough.


We note that our update approach is derived in a completely different way from standard iterative
frameworks such as multiplicatives weights updates and, generally, mirror descent. In contrast to
these standard approaches, we do not use a mirror map or regularize _ℓp_ norms to make them smooth.
Our update scheme allows our algorithm to take much longer steps, and the coordinates of the dual
solution are allowed to change by large polynomial factors in each step. This allows us to obtain a
fast convergence rate.


We outline the necessary lemmas needed to prove Theorem 1.1 before providing complete analysis
and proof in Appendix D.


**Correctness of Algorithm 2.** There are two possible outcomes of Algorithm 2. Either it returns a
primal solution (Case 1 and Case 2) or a dual certificate (Case 3). In the former two cases, Case 2
immediately gives us an approximate solution. We show in Lemma 2.2 that the returned vector in
Case 1 achieves the target approximation guarantee. In Case 3, we use the invariant shown in Lemma
2.1 to show that the returned dual solution is an infeasibility certificate.


We formalize these statements in the lemmas below.

**Lemma 2.1** (Invariant) **.** _For all t, we have that if γ_ [(] _[t]_ [)] = 1 _then_ _E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] ) _≥_ _M_ [2] _._
_∥_ _[r]_ [(] _[t]_ [+1)] _∥q_ _[−]_ _∥_ _[r]_ [(] _[t]_ [)] _∥q_

**Lemma** **2.2** (Case 1) **.** _Let_ _r_ _be_ _a_ _dual_ _solution_ _and_ _x_ = arg min _x_ �: _Ax_ �= _b⟨r,_ - _x_ [2] _⟩._ _If_
��� _∥r∥qq−_ 1 _·_ _r_ _[q]_ _x_ _[−]_ [2] [1] ��� _∞_ _[≤]_ [(1 +] _[ ϵ]_ [)] _[ M]_ [ 2] _[ then][ ∥][x][∥]_ [2] _[p]_ _[≤]_ _[M]_ [(1 +] _[ ϵ]_ [)] _[.]_

**Lemma 2.3** (Case 3) **.** _If the algorithm returns r_ [(] _[T]_ [ )] _, then_ _∥E_ _[r]_ ( [(] _r_ _[T]_ [(][ )] _[T]_ _∥_ [ )] ) _q_ _≥_ (1+ _Mϵ_ [2] ) [2] _[.]_


**Convergence of Algorithm 2.** We run the algorithm for _T_ iterations. The algorithm terminates
if at any point it finds a solution _x_ that satisfies the desired bound (otherwise it is unable to further
increase the dual solution). Otherwise, we show that it must finish very fast. Suppose we run it
for _T_ = _Thi_ + _Tlo_ iterations. Let the iterations in _Thi_ correspond to those where at least a single

2
coordinate of _r_ was scaled by _≥_ _S_ := _n_ 2 _q_ +1 [�] [1] _ϵ_ - 2 _[q]_ _q_ _[−]_ +1 [1] . Let _Tlo_ be the remaining iterations. The

following lemmas give an upperbound on _Thi_ and _Tlo_ .

**Lemma 2.4.** _We have Thi_ _≤_ _S_ _[q]_ _nϵ_ _[q]_ _[.]_


_q_ _[S]_ ln [1] _[/]_ _S_ [2] - _q_ 1


**Lemma 2.5.** _We have Tlo_ _≤_ _O_ �� 1 _ϵ_ [+] _q_ _[S]_ ln [1] _[/]_ [2]


_q_ +1
_ϵ_ 2


   - _n_    - [�]

+12 log _ϵ_ _[q]_ _._


2
Since _S_ = _n_ 2 _q_ +1 [�] [1] _ϵ_ - 2 _[q]_ _q_ _[−]_ +1 [1], we obtain the following convergence guarantee:


1                                    **Lemma 2.6.** _Algorithm 2 terminates in O_ ��� 1 _ϵ_ - _[q]_ [+3] 2 + _n_ 2 _q_ +1 [�] [1] _ϵ_ - _[q]_ 2 [2] _q_ [+2] +1 _[q]_ log - _ϵn_ _[q]_ - [�] _iterations._


Equipped with these lemmas, we give the proof for Theorem 1.1.


_Proof of Theorem 1.1._ Returning to the problem min _x∈_ R _n_ : _Ax_ = _b ∥x∥p_, we have the main algorithm


executes a binary search over the power of (1 + _ϵ_ ) in the range - _∥x_ 1(0) _∥_ [1] 2


_p_ [1] 2 _[,]_ �� _x_ (0)��2


, so the total


1
_n_ 2 _[−]_ _p_ [1]


number of calls to the subroutine solver is _O_ �log log _n_ + log [1]


  
[1] - _[q]_ 2 [2] _q_ [+2] +1 _[q]_ - _n_ - [�] _p_

_ϵ_ log _ϵ_ _[q]_ linear system solves, where _q_ = _p−_ 2 [is]


1
solver requires _O_ ��� 1 _ϵ_ - _[q]_ [+3] 2 + _n_ 2 _q_ +1 [�] [1] _ϵ_


[1] _ϵ_ �. By Lemma 2.6, the subroutine


the dual norm of _p/_ 2. Substituting the value of _q_, we obtain the conclusion.


6


**Algorithm 3** Iteratively Reweighted Least Squares


**Input:** Matrix _A ∈_ R _[d][×][n]_, vector _b ∈_ R _[d]_, _ϵ_
**Output** : Vector _x_ such that _Ax_ = _b_ that minimizes _∥x∥_ _[p]_ _p_
Initialize _x_ [(0)] = arg min _x_ : _Ax_ = _b ∥x∥_ [2] 2


_p_
_M_ [(0)] := _[∥][x]_ 16 [(0)] _p_ _[∥]_ _p_, _t ←_ 0; _κ_ =


�1 if _p ≤_ log2 log _n− n_ 1
_p−p_ 2 otherwise


**while** _M_ [(] _[t]_ [)] _≥_ 16 _p_ (1+ _ϵ_ _ϵ_ ) �� _x_ ( _t_ )�� _pp_
_g_ [(] _[t]_ [)] = �� _x_ ( _t_ )�� _p−_ 2 _x_ ( _t_ ); _R_ ( _t_ ) = 2 �� _x_ ( _t_ )�� _p−_ 2


˜∆ _←_ ResidualSolver� _p_ 2 _[,]_ - ( _g_ [(] _A_ _[t]_ [)] ) _[⊤]_


- _,_ �0 _,_ _[M]_ 2 [(] _[t]_ [)] - _,_ ( _M_ [(] _[t]_ [)] )


2 _−p_ 1

_p_ _R_ [(] _[t]_ [)] _,_ 2 _[√]_ _κ_ ( _M_ [(] _[t]_ [)] ) _p_


              **if** ∆ [˜] is an infeasibility certificate or _R_ [(] _[t]_ [)] _,_ ∆ [˜] [2][�] _≥_ 2 _M_ [(] _[t]_ [)] **then**


_M_ [(] _[t]_ [+1)] _←_ _M_ [(] _[t]_ [)] _/_ 2, _x_ [(] _[t]_ [+1)] = _x_ [(] _[t]_ [)]
**else**
_M_ [(] _[t]_ [+1)] _←_ _M_ [(] _[t]_ [)], _x_ [(] _[t]_ [+1)] = _x_ [(] _[t]_ [)] _−_ 64∆˜ _pκ_
**end if**
_t ←_ _t_ + 1
**end while**
**return** _x_ [(] _[t]_ [)]


               - 1               3 OUR ALGORITHM WITH log CONVERGENCE
_ϵ_


3.1 ALGORITHM


In this section, we present our algorithm with guarantee provided in Theorem 1.2. For the ease of the
exposition, we consider a slight variation of the problem: for _p_ _≥_ 2, we solve min _x_ : _Ax_ = _b ∥x∥_ _[p]_ _p_ [to]
(1 + _ϵ_ ) multiplicative error. We show our algorithm in Algorithms 3 and 4.


3.2 OVERVIEW OF OUR APPROACH


At the highest level, the main algorithm relies on a simple yet powerful observation by Adil et al.
(2019a), which is that the _ℓp_ minimization problem we are attempting to solve supports iterative
refinement. Adil et al. (2019a) show that having access to a weak solver which gives a constant
factor multiplicative approximation to a mixed objective of _ℓp_ and _ℓ_ 2 norms suffices to boost the
multiplicative error to 1 + _ϵ_ while only making _O_ [�] _p_ (log 1 _/ϵ_ ) calls to the solver. This reduces the
entire difficulty of the problem to implementing the weak solver.


More precisely, starting with an initial solution (set to arg min _x_ : _Ax_ = _b ∥x∥_ 2), we maintain _M_ [(] _[t]_ [)] as an
upper bound for the function value gap, ie. �� _x_ ( _t_ )�� _pp_ _[−∥][x][∗][∥]_ _p_ _[p]_ _[≤]_ [16] _[pM]_ [ (] _[t]_ [)][.] [We show this invariant in]
Lemma E.2. In each iteration, the algorithm makes a call to a solver for the residual problem which
approximates the function value progress _∥x∥_ _[p]_ _p_ _[−∥][x][ −]_ [∆] _[∥][p]_ _p_ [if we update the solution] _[ x][ ←]_ _[x][−]_ [∆][. The]
residual solution tells us either the progress is too small, in which case we can improve the upperbound
on the suboptimality gap by reducing _M_ [(] _[t]_ [)], or the progress is at least Ω - _M_ [(] _[t]_ [)][�], in which case we
can perform the update and obtain a new solution. This new solution improves the function value gap


by at least a factor 1 _−_ Ω - _p_ 1 �, and thus the algorithm requires only _O_ - _p_ log _[∥][x]_ [(0)] _ϵ∥_ _[∥]_ _xpp_ _[∗][−∥]_ _∥_ _[p]_ _p_ _[x][∗][∥]_ _p_ _[p]_

to the residual solver. We show this guarantee in Lemma E.2.


calls


We give the pseudocode for the residual solver in Algorithm 4 [3] . Prior works by Adil et al. (2019a;b;
2024) give algorithms for this solver either via a width-reduced multiplicative weights update method


3Note that while the residual solver takes as input the original matrix _A_ augmented with an extra row, the
least squares problems required by the residual solver reduce to least squares problems involving only _A_, using
the Sherman-Morrison formula. This guarantees that we only require a linear system solver for structured
matrices of the form _A_ _[⊤]_ _DA_, for non-negative diagonal _D_ .


7


**Algorithm 4** ResidualSolver( _p, A, b, θ, M_ )

**Input:** Matrix _A ∈_ R _[d][×][n]_, vector _b ∈_ R _[d]_, target value _M_, weight _θ_
**Output** : Vector _x_ such that _Ax_ = _b_, _∥x∥_ 2 _p_ _≤_ 2 _M_ and - _θ, x_ [2][�] _≤_ min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�]


or approximate infeasibility certificate _r_, _∥r∥q_ = 1.
**if** _p ≤_ loglog _n n−_ 1 **[then]**
_r_ = 11

[;] _[x]_ [ = arg min] _[x]_ [:] _[Ax]_ [=] _[b][⟨][r]_ [ +] _[ θ, x]_ [2] _[⟩]_


1

1
_n_ _q_ [;][ �] _[x]_ [ = arg min] _[x]_ [:] _[Ax]_ [=] _[b][⟨][r]_ [ +] _[ θ, x]_ [2] _[⟩]_


**if** _∥x_    - _∥_ 2 _p_ _≤_ 2 _M_ **then** return � _x_ **else return** _r_ **end if**
**else**
_t_ = 0, _r_ [(0)] = [2] _[q][−]_ 1 [1] [= 0][,] _[ s]_ [(] _[t][′]_ [)] [= 0]

[,] _[ t][′]_


[2] _[q][−]_ 1 [1] [= 0][,] _[ s]_ [(] _[t][′]_ [)] [= 0]

2 _qn_ _q_ [,] _[ t][′]_


**while** - _r_ [(] _[t]_ [)][�] _[q]_ [��]
��� �1 _[≤]_ [1]

_x_ [(] _[t]_ [)] = arg min _x_ : _Ax_ = _b⟨r_ [(] _[t]_ [)] + _θ, x_ [2] _⟩_


_γi_ [(] _[t]_ [)] =


- _x_ 2 _i_ _[∥][r][∥]_ _q_ _[q][−]_ [1] _x_ [2] _i_ _[∥][r][∥]_ _q_ _[q][−]_ [1]
if _≥_ 2 _M_ [2]
_M_ [2] _ri_ _[q][−]_ [1] _ri_ _[q][−]_ [1], for all _i_

1 otherwise


      - �1 _/q_
_αi_ [(] _[t]_ [)] = _γi_ [(] _[t]_ [)]

**if** _α_ [(] _[t]_ [)] = 1 **then return** _x_ [(] _[t]_ [)] **end if** _▷_ _Case 1_
_r_ [(] _[t]_ [+1)] = _α_ [(] _[t]_ [)] 2 _· r_ [(] _[t]_ [)]
**if** _α_ [(] _[t]_ [)] _≤_ _n_ 2 _q_ +1 **then** _s_ [(] _[t][′]_ [+1)] = _s_ [(] _[t][′]_ [)] + _x_ [(] _[t]_ [)] ; _t_ _[′]_ = _t_ _[′]_ + 1 **end if**
**if** _t_ _[′]_ _>_ 0 **and** _s_ ( _t′_ ) _/t′_ **[then return]** _[ s]_ [(] _[t][′]_ [)] _[/t][′]_ **[end if]** _▷_ _Case 2_
��� ���2 _p_ _[≤]_ [2] _[M]_

_t_ = _t_ + 1
**end while**
**end if**
**return** _r_ [(] _[t]_ [)] _▷_ _Case 3_


which achieves the state-of-the-art theoretical runtime but does not support a practical implementation
or via a practical IRLS method with suboptimal theoretical guarantee. In contrast, we build on ideas
from the low precision IRLS solver we have shown in the previous section and design a new IRLS
algorithm that attains the best of both worlds.


Our residual solver outputs an approximate solution to a constant factor to the objective of the form


min
_x_ : _Ax_ = _b_


�� _x_ 2�� _p_ [+] - _θ, x_ [2][�] (4)


for _p ≥_ 1 and a positive weight vector _θ_ _∈_ R _[n]_ . We also start with the dual formulation of the problem


_r_
+ _θ_
_∥r∥q_


(4) = min max
_x_ : _Ax_ = _b_ _r_ : _∥r∥q_ =1


- _r, x_ [2][�] + - _θ, x_ [2][�] = max - _r_ + _θ, x_ [2][�] = max
_r≥_ 0: _∥r∥q_ =1 _x_ : [min] _Ax_ = _b_ _r≥_ 0 _[E]_


_,_


where _q_ is the dual norm to _p_ and _E_ ( _r_ + _θ_ ) = min _x_ : _Ax_ = _b⟨r_ + _θ, x_ [2] _⟩_ . Given a target _M_, our goal
is to find a primal solution _x_ that satisfies _∥x∥_ 2 _p_ _≤_ 2 _M_ and - _θ, x_ [2][�] _≤_ min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�]

or a dual solution _r_ _∈_ R _[n]_ (infeasibility certificate) which can certify that min _x_ : _Ax_ = _b ∥x∥_ [2] 2 _p_ _[≥]_


_E_ - _∥rr∥q_ [+] _[ θ]_ - _≥_ _[M]_ 2 _κ_ [2] [, where] _[ κ]_ [ is a value set as shown in Algorithm 3.]

We distinguish between two regimes: when _p_ is sufficiently small, 1 _≤_ _p_ _≤_ loglog _n n−_ 1 [for which we]

will show that we can obtain a solution by _O_ (1) calls to the linear solver, and when _p >_ loglog _n n−_ 1 [, to]
which we need to pay more attention. In the latter case, similarly to Algorithm 2, we want to maintain
the invariant

_E_ ( _r_ [(] _[t]_ [+1)] + _θ_ ) _−E_ ( _r_ [(] _[t]_ [)] + _θ_ )
_≥_ _M_ [2] _._
�� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_

Notice the differences between this objective and the problem min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [which we solve in]
the previous section. The _ℓ_ 2 term - _θ, x_ [2][�] makes this objective no longer scale-free. However, this _ℓ_ 2


8


10 20 30 40 50
p


500 750 1000 1250 1500 1750 2000 2250 2500
n


75

70

65

60

55

50

45

40

35


(g) size= _n ×_ ( _n −_ 100)


75

70

65

60

55

50

45

40

35


3 4 5 6 p 7 8 9 10


(b) size=500 _×_ 400


3 4 5 6 p 7 8 9 10


(f) size=500 _×_ 400


17.5


15.0


12.5


10.0


7.5


5.0


2.5


0.0


50


45


40


35


30


25


20


0 500 1000 1500 2000 2500 3000
n


(a) size= _n×_ ( _n−_ 50) _, p_ = 8


50


45


40


35


30


25


20


25


20


15


10


5


0


(c) size= _n ×_ ( _n −_ 100)


(d) size= _n ×_ ( _n −_ 100)


20.0


17.5


15.0


12.5


10.0


7.5


5.0


2.5


0.0


0 500 1000 1500 2000 2500 3000
n


500 750 1000 1250 1500 1750 2000 2250 2500
n


10 20 30 40 50
p


17.5


15.0


12.5


10.0


7.5


5.0


2.5


0.0


(e) size= _n×_ ( _n−_ 50) _, p_ = 8


(h) size= _n ×_ ( _n −_ 100)


Figure 1: Performance on random matrices: min _∥Ax −_ _b∥_ _[p]_ _p_ [with] _[ϵ]_ [=] [10] _[−]_ [10][.] [We] [compare] [our]
algorithm with CVX using SDPT3 and SeDuMi solvers and _p_ -IRLS by Adil et al. (2019b). Figures
(a),(b),(e),(f) plot the average and standard deviation of number of iterations and time taken by the
solvers to find a solution over 10 runs. Figures (c),(d),(g),(h) measure over 5 runs.


term does not affect the lower bound [�] _i_ _[r]_ _i_ [(] _[t]_ [)] - _x_ [(] _i_ _[t]_ [)] �2 [�] 1 _−_ _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)]


in the change in the objective


_ri_ [(] _[t]_ [+1)]


    
  - �2
_i_ _[r]_ _i_ [(] _[t]_ [)] _x_ [(] _i_ _[t]_ [)]


1 _−_ _ri_ [(] _[t]_ [)]


(eq. (2)); thus it suffices to maintain


_ri_ _≥_ _M_ [2] in order to guarantee the

_∥_ _[r]_ [(] _[t]_ [+1)] _∥q_ _[−]_ _∥_ _[r]_ [(] _[t]_ [)] _∥q_


invariant _[E]_ [(] _∥_ _[r]_ [(] _[r][t]_ [(][+1)] _[t]_ [+1)][+] _∥_ _[θ]_ _q_ [)] _[−E][−]_ _∥_ [(] _[r][r]_ [(] _[t]_ [(][)] _[t]_ [)] _∥_ [+] _q_ _[θ]_ [)] _≥_ _M_ [2] . At the same time, if we maintain _∥r∥q_ _≤_ 1, we can show

that if the algorithm outputs a primal solution _x_, the _ℓ_ 2 term - _θ, x_ [2][�] _≤_ min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�] .
This requires us to initialize _r_ with sufficiently small _∥r∥q_ . Algorithm 4 then follows similarly to
Algorithm 2, with the note that it suffices to obtain only a constant approximation. We give the
correctness and convergence of Algorithm 4 in Lemma E.1 whose proof is based on the same idea as
the analysis for Algorithm 2.


The complete analysis of our algorithm is provided in Appendix E.


4 EXPERIMENTAL EVALUATION


**On** **synthetic** **data.** We follow the experimental setup in Adil et al. (2019b), and build on the
provided code [4] . We evaluate the performance of our high-precision Algorithm 3 on the problem
min _∥Ax −_ _b∥_ _[p]_ _p_ [on two types of instances:] **[ (1)]** [ Random matrices.] [The entries of] _[ A]_ [ and] _[ b]_ [ are generated]
uniformly at randomly between 0 and 1; **(2)** Random graphs. We use the procedure in Adil et al.
(2019b) to generate random graphs and the corresponding _A_ and _b_ . The generated graph is a weighted
graph, where the vertices are generated by choosing a point in [0 _,_ 1] [10] uniformly at random, each
vertex is connected to the 10 nearest neighbors. The edge weights are generated by a gaussian type
function (by Flores-Calder-Lerman). _k_ (around 10) nodes are labeled in [0 _,_ 1] and let _g_ be the label
vector. Let _B_ be the edge-vertex adjacency matrix, _W_ be the diagonal matrix with edge weights. We
generate _A_ = _W_ [1] _[/p]_ _B_, _b_ = _−B_ [: _, n_ : _n_ + _k_ ] _g_ .


We vary _p_ and the size of the matrices and graphs, while keeping the error _ϵ_ = 10 _[−]_ [10] . All implementations were done on MATLAB 2024a on a MacBook Pro M2 with 16GB RAM. We measure
the number of iterations and running time for each algorithm and report them in Figures 1-2. In the
appendix, we provide additional experimental results when 1 _< p <_ 2 and when _ϵ_ varies.


[4The code is available at https://github.com/fast-algos/pIRLS](https://github.com/fast-algos/pIRLS)


9


60

55

50

45

40

35

30

25

20


35


30


25


20


15


10


5


0


0 2000 4000 6000 8000 10000
Number of nodes in graph


(a) _p_ = 8


0 2000 4000 6000 8000 10000
Number of nodes in graph


(e) _p_ = 8


30


25


20


15


10


5


0


(f) Number of nodes=500


2000 3000 4000 5000 6000 7000 8000 9000 10000
Number of nodes in graph


(c) _n_ nodes


2000 3000 4000 5000 6000 7000 8000 9000 10000
Number of nodes in graph


(g) _n_ nodes


110


100


90


80


70


60


50


40


50


40


30


20


10


0


10 20 30 40 50
p


(d) _n_ nodes


10 20 30 40 50
p


(h) _n_ nodes


60


50


40


30


20


3 4 5 6 p 7 8 9 10


(b) Number of nodes=500


110


100


90


80


70


60


50


40


50


40


30


20


10


0


3 4 5 6 p 7 8 9 10


Figure 2: Performance on random graph instances: min _∥Ax −_ _b∥_ _[p]_ _p_ [with] _[ ϵ]_ [ = 10] _[−]_ [10][.] [We compare our]
algorithm with CVX using SDPT3 and SeDuMi solvers and _p_ -IRLS by Adil et al. (2019b). Figures
(a),(b),(e),(f) measure over 10 runs. Figures (c),(d),(g),(h) measure over 5 runs.


Table 1: Performance of our algorithm against _p_ -IRLS on six real-world datasets for _p_ = 8, _ϵ_ =
10 _[−]_ [10] .


|Col1|Col2|CT slices<br>Graf et al.<br>(2011)|KEGG<br>Metabolic<br>Naeem<br>and<br>Asghar<br>(2011)|Power<br>Consump-<br>tion<br>Hebrail<br>and<br>Berard<br>(2006)|Buzz in<br>Social<br>Media<br>Kawala<br>et al.<br>(2013)|Protein<br>Property<br>Rana<br>(2013)|Song<br>Year Pre-<br>diction<br>Bertin-<br>Mahieux<br>(2011)|
|---|---|---|---|---|---|---|---|
||Size|48150<br>_×_385|57248<br>_×_27|1844352<br>_×_11|524925<br>_×_77|41157_×_9|463811<br>_×_90|
|no.<br>iters|_p_-IRLS|48|50|45|50|44|45|
|no.<br>iters|Ours|36|42|36|42|36|36|
|time<br>(s)|_p_-IRLS|14.3|2.5|32.|28.|1.6|22.5|
|time<br>(s)|Ours|9.2|1.7|15.7|18.1|1.1|13.3|


**On real-world datasets.** We test our algorithm against _p_ -IRLS on six regression datasets from the
UCI repository. CVX has excessive runtime and hence is excluded from the comparison. Results are
provided in Table 1.
_Remark_ 4.1 _._ Regarding the correctness of the algorithm, we use the output by CVX as the baseline.
In all experiments, our algorithm has error within the _ϵ_ margin compared with the objective value of
the CVX solution (see appendix).


On smaller instances, we compare our algorithm with CVX using SDPT3 and Sedumi solvers and the
_p_ -IRLS algorithm by Adil et al. (2019b). While CVX solvers generally need fewer iterations to find a
solution, they are significantly slower on all instances than our algorithm and _p_ -IRLS. Our algorithm
also significantly outperforms _p_ -IRLS in both the number of iterations (calls to a linear system solver)
and running time. When the size of the problem and the value of _p_ increases, the gap between our
algorithm and _p_ -IRLS also increases. On average, our algorithm is 1-2 _._ 6 times faster than _p_ -IRLS.


ACKNOWLEDGEMENT


AE was supported in part by an Alfred P. Sloan Research Fellowship. AV was partially supported
by the French Agence Nationale de la Recherche (ANR) under grant ANR-21-CE48-0016 (project
COMCOPT).


10


REPRODUCIBILITY STATEMENT


For the reproducibility purpose, we submitted the source code in the supplementary material. We
included the MATLAB implementation by Adil et al. (2019b).


REFERENCES


Deeksha Adil, Rasmus Kyng, Richard Peng, and Sushant Sachdeva. Iterative refinement for _ℓp_ -norm
regression. In _Proceedings of the Thirtieth Annual ACM-SIAM Symposium on Discrete Algorithms_,
pages 1405–1424. SIAM, 2019a.


Deeksha Adil, Richard Peng, and Sushant Sachdeva. Fast, provably convergent irls algorithm for
p-norm linear regression. _Advances in Neural Information Processing Systems_, 32, 2019b.


Deeksha Adil, Rasmus Kyng, Richard Peng, and Sushant Sachdeva. Fast algorithms for _ℓp_ -regression.
_J. ACM_, 71(5):34:1–34:45, 2024. [URL https://doi.org/10.1145/3686794.](https://doi.org/10.1145/3686794)


Ahmed El Alaoui. Asymptotic behavior of _\_ ( _\_ ell ~~p~~ _\_ )-based laplacian regularization in semisupervised learning. In _COLT_, volume 49 of _JMLR_ _Workshop_ _and_ _Conference_ _Proceedings_,
pages 879–906. JMLR.org, 2016.


T. Bertin-Mahieux. Year Prediction MSD. UCI Machine Learning Repository, 2011. DOI:
https://doi.org/10.24432/C50K61.


Sebastien Bubeck, Michael B Cohen, Yin Tat Lee, and Yuanzhi Li.´ An homotopy method for _ℓp_
regression provably beyond self-concordance and in input-sparsity time. In _Proceedings of the_
_50th Annual ACM SIGACT Symposium on Theory of Computing_, pages 1130–1137. ACM, 2018.


Brian Bullins. Fast minimization of structured convex quartics. _arXiv preprint arXiv:1812.10349_,
2018.


C Sidney Burrus. Iterative reweighted least squares. _OpenStax CNX. Available online:_ _http://cnx._
_org/contents/92b90377-2b34-49e4-b26f-7fe572db78a1_, 12(2012):6, 2012.


Hui Han Chin, Aleksander Madry, Gary L. Miller, and Richard Peng. Runtime guarantees for
regression problems. In Robert D. Kleinberg, editor, _Innovations in Theoretical Computer Science,_
_ITCS ’13, Berkeley, CA, USA, January 9-12, 2013_, pages 269–282. ACM, 2013. ISBN 978-1-45031859-4. doi: 10.1145/2422436.2422469. URL [https://doi.org/10.1145/2422436.](https://doi.org/10.1145/2422436.2422469)
[2422469.](https://doi.org/10.1145/2422436.2422469)


Paul Christiano, Jonathan A. Kelner, Aleksander Madry, Daniel A. Spielman, and Shang-Hua Teng.
Electrical flows, laplacian systems, and faster approximation of maximum flow in undirected
graphs. In Lance Fortnow and Salil P. Vadhan, editors, _Proceedings of the 43rd ACM Symposium_
_on Theory of Computing, STOC 2011, San Jose, CA, USA, 6-8 June 2011_, pages 273–282. ACM,
2011. ISBN 978-1-4503-0691-1. doi: 10.1145/1993636.1993674. [URL https://doi.org/](https://doi.org/10.1145/1993636.1993674)
[10.1145/1993636.1993674.](https://doi.org/10.1145/1993636.1993674)


Alina Ene and Adrian Vladu. Improved convergence for _ℓ_ 1 and _ℓ∞_ regression via iteratively
reweighted least squares. In _International Conference on Machine Learning_, pages 1794–1801,
2019.


F. Graf, H.-P. Kriegel, M. Schubert, S. Poelsterl, and A. Cavallaro. Relative location of CT slices on
axial axis. UCI Machine Learning Repository, 2011. DOI: https://doi.org/10.24432/C5CP6G.


Georges Hebrail and Alice Berard. Individual Household Electric Power Consumption. UCI Machine
Learning Repository, 2006. DOI: https://doi.org/10.24432/C58K54.


Lingxiao Huang and Nisheeth K. Vishnoi. Coresets for clustering in euclidean spaces: importance
sampling is nearly optimal. In _STOC_, pages 1416–1429. ACM, 2020.


Lingxiao Huang, Shaofeng H.-C. Jiang, Jianing Lou, and Xuan Wu. Near-optimal coresets for robust
clustering. In _ICLR_ . OpenReview.net, 2023.


11


Arun Jambulapati, Yang P Liu, and Aaron Sidford. Improved iteration complexities for overconstrained p-norm regression. In _Proceedings_ _of_ _the_ _54th_ _Annual_ _ACM_ _SIGACT_ _Symposium_ _on_
_Theory of Computing_, pages 529–542, 2022.


Franois Kawala, Ahlame Douzal, Eric Gaussier, and Eustache Diemert. Buzz in social media . UCI
Machine Learning Repository, 2013. DOI: https://doi.org/10.24432/C56G6V.


Yin Tat Lee and Aaron Sidford. Path finding methods for linear programming: Solving linear
programs in  - (vrank) iterations and faster algorithms for maximum flow. In _2014_ _IEEE_ _55th_
_Annual Symposium on Foundations of Computer Science_, pages 424–433. IEEE, 2014.


Meng Liu and David F. Gleich. Strongly local p-norm-cut algorithms for semi-supervised learning
and local graph clustering. In _NeurIPS_, 2020.


Xiangrui Meng and Michael Mahoney. Robust regression on mapreduce. In _International Conference_
_on Machine Learning_, pages 888–896. PMLR, 2013.


Muhammad Naeem and Sohail Asghar. KEGG Metabolic Relation Network (Directed). UCI Machine
Learning Repository, 2011. DOI: https://doi.org/10.24432/C5CK52.


Yurii Nesterov and Arkadii Nemirovskii. _Interior-point polynomial algorithms in convex program-_
_ming_ . SIAM, 1994.


Kent Quanrud. Nearly linear time approximations for mixed packing and covering problems without
data structures or randomization. In _Symposium on Simplicity in Algorithms_, pages 69–80. SIAM,
2020.


Prashant Rana. Physicochemical Properties of Protein Tertiary Structure. UCI Machine Learning
Repository, 2013. DOI: https://doi.org/10.24432/C5QW3H.


A PROPERTY OF THE ENERGY FUNCTION


We recall the definition of energy function and its properties used in the algorithms.

**Definition A.1.** (Energy function). Given a vector _r_ _∈_ R _[n]_ + [, we let the electrical energy be] _[ E]_ [(] _[r]_ [) =]
min _x_ : _Ax_ = _b⟨r, x_ [2] _⟩_ .

**Lemma** **A.1.** _(Computing_ _the_ _energy_ _minimizer)_ _Given_ _b_ _∈_ R _[d]_ _and_ _r_ _∈_ R _[n]_ + _[,]_ _[the]_ _[least]_ _[squares]_
_problem_ min _x_ : _Ax_ = _b⟨r, x_ [2] _⟩_ _can be solved by evaluating x_ = D( _r_ ) _[−]_ [1] _A_ _[⊤]_ [�] _A_ D( _r_ ) _[−]_ [1] _A_ _[⊤]_ [�][+] _b, where_
D( _r_ ) _is the diagonal matrix whose entries are given by r._


The following lemma gives us a lower bound on the increase in electrical energy when we increase _r_ .

**Lemma A.2.** _Given r_ _[′]_ _≥_ _r and letting x_ = arg min _x_ : _Ax_ = _b⟨r, x_ [2] _⟩, one has that_


_E_ ( _r_ _[′]_ ) _−E_ ( _r_ ) _≥_ - _rix_ [2] _i_


_i_


1 _−_ _[r][i]_

_ri_ _[′]_


_._


_Proof._ This inequality follows from the standard lower bound for _E_ ( _r_ _[′]_ ) _−E_ ( _r_ ), which the reader can
find in Ene and Vladu (2019).


B REDUCING GENERAL REGRESSION PROBLEMS TO THE
AFFINE-CONSTRAINED VERSION


In this section we show that the affine constrained version of the problem we consider is in full
generality. Formally, we show that any _ℓp_ regression problem of the form min _Ax_ = _b ∥Nx −_ _v∥p_ can
be reduced to the form we consider.


12


**Lemma B.1.** _Let A_ _∈_ R _[s][×][n]_ _, b_ _∈_ R _[s]_ _, N_ _∈_ R _[m][×][n]_ _, v_ _∈_ R _[m]_ _and consider the optimization objec-_


            - _x_
_tive_ min _Ax_ = _b ∥Nx −_ _v∥p._ _Let_ _z_


_be a_ (1 + _ε_ ) _approximate solution to the affine-constrained_


_regression problem_

min

             - _N_ _−Im×m_ �� _x_
_A_ 0 _s×m_ _z_


- _v_
_b_


- _[∥][z][∥]_ _p_ _[.]_


=


_Then x is a_ (1 + _ε_ ) _approximate solution to the original objective._ _Furthermore, each least squares_
_subproblem can be solved using two calls to a linear system solver for N_ _[⊤]_ _RN_ _, and one call to a_
_linear system solver for A_ - _N_ _[⊤]_ _RN_ �+ _A⊤._


_Proof._ We augment the dimension of the iterate by introducing _m_ additional variables encoded in a
vector _z_ _∈_ R _[m]_ . Hence one can equivalently enforce the constraints


_Nx −_ _z_ = _v_

_Ax_ = _b_


and simply seek to minimize _∥z∥p_ instead of _∥Ax −_ _b∥p_, which is the suitable formulation required
by our solver. Note that while we do not have any weights on the _x_ iterate, the analysis goes through
normally, since in fact it tolerates solving a more general weighted _ℓp_ regression problem.


To solve the corresponding least squares problem, we need to compute


1
min
_Ax_ = _b_ 2


- _r,_ ( _Nx −_ _v_ ) [2][�] = min 1 - _N_ _[⊤]_ _Rv, x_ - + [1]
_Ax_ = _b_ 2 _[x][⊤][N][ ⊤][RNx][ −]_ 2 _[v][⊤][Rv]_


= max min 1 - _N_ _[⊤]_ _Rv, x_ - + [1]
_y_ _x_ 2 _[x][⊤][N][ ⊤][RNx][ −]_ 2 _[v][⊤][Rv]_ [ +] _[ ⟨][b][ −]_ _[Ax, y][⟩]_


= max
_y_


- 1 - - [�]
_⟨b, y⟩_ + min _N_ _[⊤]_ _Rv_ + _A_ _[⊤]_ _y, x_ _−_ [1]
_x_ 2 _[x][⊤][N][ ⊤][RNx][ −]_ 2 _[v][⊤][Rv .]_


where _R_ is the diagonal matrix whose entries are given by _r_ . The inner problem is minimized at


_x_ =            - _N_ _[⊤]_ _RN_ �+ � _N_ _[⊤]_ _Rv_ + _A_ _[⊤]_ _y_            - _,_


which simplifies the problem to


max _⟨b, y⟩−_ [1]
_y_ 2


[1] - _N_ _[⊤]_ _Rv_ + _A_ _[⊤]_ _y_ - _⊤_ - _N_ _[⊤]_ _RN_ �+ � _N_ _[⊤]_ _Rv_ + _A_ _[⊤]_ _y_ - _−_ [1]

2 2


2 _[v][⊤][Rv]_


= max
_y_


- _b −_ _A_ - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv, y_ - _−_ [1] - _N_ _[⊤]_ _RN_ �+ _A⊤y_

2 _[y][⊤][A]_


_−_ [1] - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv −_ 1

2 _[v][⊤][RN]_ 2 _[v][⊤][Rv,]_


which is maximized at


_y_ =         - _A_         - _N_ _[⊤]_ _RN_ �+ _A⊤_ [�][+] [�] _b −_ _A_         - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv_         - _,_


so


_x_ = - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv_ + - _N_ _[⊤]_ _RN_ �+ _A⊤_ [�] _A_ - _N_ _[⊤]_ _RN_ �+ _A⊤_ [�][+] [�] _b −_ _N_ - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv_ 

=  - _N_ _[⊤]_ _RN_ �+ [�] _N_ _[⊤]_ _Rv_ + _A_ _[⊤]_ [�] _A_  - _N_ _[⊤]_ _RN_ �+ _A⊤_ [�][+] [�] _b −_ _A_  - _N_ _[⊤]_ _RN_ �+ _N ⊤Rv_  - [�] _._


We observer that to execute this step we require two calls to a solver for _N_ _[⊤]_ _RN_, and one call to a
solver for _A_ - _N_ _[⊤]_ _RN_ �+ _A⊤_ .


13


C SOLVING _ℓp_ REGRESSION FOR 1 _≤_ _p <_ 2


In this section we show that while our solvers are defined for _ℓp_ regression when _p_ _≥_ 2, they also
provide solutions _ℓq_ regression for 1 _≤_ _q_ _<_ 2. This follows directly from exploiting duality. See Adil
et al. (2019a), section 7.2 for a proof detailed. Here we briefly explain why this is the case. Let _p, q_
such that [1] [+] [1] [= 1][,][ 1] _[ ≤]_ _[q]_ _[<]_ [ 2][, and consider the] _[ ℓ][q]_ [regression problem, along with its dual]


[1] [1]

_p_ [+] _q_


_q_ [= 1][,][ 1] _[ ≤]_ _[q]_ _[<]_ [ 2][, and consider the] _[ ℓ][q]_ [regression problem, along with its dual]


_x_ :min _Ax_ = _b_ _[∥][x][∥][q]_ [=] _∥A_ _[⊤]_ max _y∥p≤_ 1 _[⟨][b, y][⟩]_ _[.]_


We can use our solver to provide a high precision solution to the dual maximization problem, which
we then show can be used to read off a primal nearly optimal solution. Indeed, we can equivalently
solve
_⟨b,y_ min _⟩_ =1 �� _A⊤y_ �� _p_


to high precision _ε_ = _n_ _[O]_ 1 [(1)] [, based on which we construct the nearly-feasible primal solution]


_x_ = _⟨b, y⟩_ _·_             - _A_ _[⊤]_ _y_             - _p−_ 1 _._
_∥A_ _[⊤]_ _y∥_ _[p]_ _p_


To see why this is a good solution, let us assume that we achieve exact gradient optimality for _y_,
which means that for some scalar _λ_,


_A_           - _A_ _[⊤]_ _y_           - _p−_ 1 = _b · λ ._ (5)


First let us verify that _x_ is feasible. Using (5) we see that:


= _⟨b, y⟩_ _· A_ - _A_ _[⊤]_ _y_ - _p−_ 1 =
_∥A_ _[⊤]_ _y∥_ _[p]_ _p_


_⟨b, y⟩_

_· λ_
_∥A_ _[⊤]_ _y∥_ _[p]_ _p_


_Ax_ = _A_


- _⟨b, y⟩_ _·_   - _A_ _[⊤]_ _y_   - _p−_ 1
_∥A_ _[⊤]_ _y∥_ _[p]_ _p_


_· b ._


Additionally we can also use (5) again to obtain that


�� _A⊤y_ �� _pp_ [=]                     - _y, A_                     - _A_ _[⊤]_ _y_                     - _p−_ 1 [�] = _⟨y, b⟩· λ,_


which allows us to conclude that

_Ax_ = _b,_


so _x_ is feasible. Finally, we can measure the duality gap by calculating


_λ_


_∥x∥q_ = _λ_ [1]


_p−_ 1 [�]

 - _A_ _[⊤]_ _y_ [1]
��� �� _q_ [=] _λ_


[1] ��� _A_ _[⊤]_ _y_ �( _p−_ 1) _p−p_ 1 [�] _[p][−]_ _p_ [1]

_λ_ _[·]_


_p_
= [1]


�� _A⊤y_ �� _pp−_ 1


= _∥A⟨y, b_ _[⊤]_ _y⟩∥_ _[p]_ _p_ _·_ �� _A⊤y_ �� _pp−_ 1 = _∥A⟨y, b_ _[⊤]_ _y⟩∥p_ _,_


which certifies optimality for _b_ . While in general we do not solve the dual problem exactly, which
yields a slight violation in the demand for the primal iterate _x_, this can be fixed by adding to _x_ a
flow _x_ - = _A_ _[⊤]_ [�] _AA_ _[⊤]_ [�][+] ( _b −_ _Ax_ ) that routes the residual demand. This affects the _ℓq_ norm only
slightly since the residual demand is guaranteed to be very small due to the near-optimality of the
dual problem. Then we can proceed to bounding the duality gap by following the argument sketched
above, while also carrying the polynomially small error through the calculation. We refer the reader
to Adil et al. (2019a) for the detailed error analysis. We have the following theorem.


**Theorem C.1.** _For any_ 1 _<_ _p_ _≤_ 2 _, there is an iterative algorithm for the ℓp_ _regression problem_
min _x∈_ R _n_ : _Ax_ = _b ∥x∥p_ _that solves O_ - _q_ [2] log _n_ log - _nϵ_ �� _subproblems, each of which makes O_ - _n_ 3 _qq−−_ 22 [�]

_calls_ _to_ _solve_ _a_ _linear_ _system_ _of_ _the_ _form_ _AD_ [�] _A_ [�] _[⊤]_ _ϕ_ = _z,_ _where_ _q_ = _p−p_ 1 _[,]_ _[D]_ _[is]_ _[an]_ _[arbitrary]_ _[non-]_

_negative diagonal matrix,_ _A is a matrix obtained from A by appending a single row, and z is a vector_

[�]
_obtained from the all-zero vector by appending a single non-zero coordinate._


14


D PROOF OF THEOREM 1.1


_Proof of Lemma 2.1._ First we show (3).

�� _r_ ( _t_ +1)�� _q_ 1 _[−]_ �� _r_ ( _t_ )�� _q_ _≥_ �� _r_ ( _t_ +1) _q_ �� _r_ ��( _qqt_ ) _[−]_ �� _qq_ �� _−r_ 1( _t_ )�� _qq_ _._


This is equivalent to show

_q_
_r_ ( _t_ +1)
��� ���


_r_ ( _t_ +1)
_q_ _[≥]_ _[q]_ ��� ��� _q_


_q_ _q_

_r_ ( _t_ )
_q_ [+ (] _[q][ −]_ [1)] ��� ��� _q_


_q−_ 1
_r_ ( _t_ )
��� ��� _q_


which can easily be obtained from AM-GM inequality.


Using (3) and Lemma A.2 we have


��


_E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] ) _q_ �� _r_ ( _t_ )�� _qq−_ 1
_≥_
�� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_


�� - �2 [�] _ri_ [(] _[t]_ [)]

_i_ _[r]_ _i_ [(] _[t]_ [)] _x_ [(] _i_ _[t]_ [)] 1 _−_ _ri_ [(] _[t]_ [+1)]


 - - _q_ - - _q_
_i_ _ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


��


_q_ �� _r_ ( _t_ )�� _qq−_ 1
=


��


_i,α_ [(] _i_ _[t]_ [)] _>_ 1 _[r]_ _i_ [(] _[t]_ [)] - _x_ [(] _i_ _[t]_ [)] �2 [�] 1 _−_ _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)]


_i,α_ [(] _i_ _[t]_ [)] _>_ 1


- - _q_ - - _q_ _._
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


For _i_ such that _αi_ [(] _[t]_ [)] _>_ 1, we have _ri_ [(] _[t]_ [+1)] = _αi_ [(] _[t]_ [)] _[r]_ _i_ [(] _[t]_ [)][, thus]


_q_ �� _r_ ( _t_ )�� _qq−_ 1 _ri_ [(] _[t]_ [)] - _x_ [(] _i_ _[t]_ [)] �2 [�] 1 _−_ _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)]


- �2
_x_ [(] _i_ _[t]_ [)]


- - _q_ - - _q_ =
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


�� _r_ ( _t_ )�� _qq−_ 1


     �)�� _qq−_ 1�� _q−x_ 1 [(] _i_ _[t]_ [)] �2 _·_ _q_ - 1 _−_ - _qα_ 1 [(] _i_ _[t]_ [)]
_ri_ [(] _[t]_ [)] _αi_ [(] _[t]_ [)] _−_


- - _q_
_αi_ [(] _[t]_ [)] _−_ 1


1
_≥_ _γi_ [(] _[t]_ [)] _[M]_ [ 2] _[ ·]_              -              - _q_
_αi_ [(] _[t]_ [)]


= _M_ [2] _,_


where the first inequality is due to _αq_ (( _αα_ _[q]_ _−−_ 1)1) _[≥]_ _α_ 1 _[q]_ [,] [for] _[α]_ _[>]_ [1][.] [We] [can] [then] [obtain] [the] [desired]
conclusion from here.


_Proof of Lemma 2.2._ If
���� _∥r∥qq−_ 1 _·_ _rx_ _[q][−]_ [2][1] ���� _∞_ _≤_ (1 + _ϵ_ ) _M_ [2] _,_


for all _i_ we have


which gives


We obtain


as needed.


_[r]_ _i_ _[q][−]_ [1]
_x_ [2] _i_ _[≤]_ [(1 +] _[ ϵ]_ [)][2] _[ M]_ [ 2] _,_
_∥r∥_ _[q]_ _q_ _[−]_ [1]


_[r]_ _i_ _[q]_
_x_ [2] _i_ _[p]_ _≤_ (1 + _ϵ_ ) [2] _[p]_ _M_ [2] _[p]_ _∥r∥_ _[q]_ _q_ _,_


_∥x∥_ [2] 2 _[p]_ _p_ _[≤]_ [(1 +] _[ ϵ]_ [)][2] _[p][ M]_ [ 2] _[p][,]_


15


_Proof of Lemma 2.3._ We have that

_E_ ( _r_ [(] _[T]_ [ )] ) [�] _t_ _[T]_ =0 _[ −]_ [1]    - _E_ ( _r_ [(] _[t]_ [+1)] ) _−E_ ( _r_ [(] _[t]_ [)] )�
= _[E]_ [(] _[r]_ [(0)][) +]
�� _r_ ( _T_ )�� _q_ �� _r_ ( _T_ )�� _q_

_E_ ( _r_ [(0)] ) + [�] _t_ _[T]_ =0 _[ −]_ [1] ��� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_         - _· M_ [2]
_≥_ (due to the invariant)

�� _r_ ( _T_ )�� _q_


1
1 _−_
�� _r_ ( _T_ )�� _q_


_≥_


��� _r_ ( _T_ )�� _q_ _[−]_ [1] - _· M_ [2]
= _M_ [2] _·_
�� _r_ ( _T_ )�� _q_


_≥_ _M_ [2] _·_ (1 _−_ _ϵ_ ) (since _r_ ( _T_ )
��� ��� _q_ _[≥]_ [1] _ϵ_ [)]


_M_ [2]
_≥_ _[.]_

(1 + _ϵ_ ) [2]


_Proof of Lemma 2.4._ Suppose the contrary. Then we claim that the perturbations that scale the dual
solution by _≥_ _S_ will have increased it a lot to the point where _∥r∥_ _[q]_ _q_ _[≥]_ _ϵ_ 1 _[q]_ [.] [Indeed, since] _[ r]_ [ is initialized]
to _n_ [1] 1 _[/q]_ [, in the worst case each perturbation in] _[ T][hi]_ [touches a different coordinate] _[ i]_ [.] [Therefore this]
establishes a lower bound of _Thi_ _·_ _[S]_ _n_ _[q]_ [on] _[∥][r][∥][q]_ _q_ [.] [As] [this] [must] [be] [at] [most] _ϵ_ 1 _[q]_ [,] [since] [otherwise] [we]

obtained a good solution per Lemma 2.3, we obtain the conclusion.


Before showing the proof of Lemma 2.5, we claim that we can either look at the history produced in
_Tlo_ and obtain an approximately feasible solution, or a single coordinate of _r_ must have increased a
lot.
**Lemma D.1.** _Consider the set of iterates_ ( _r_ [(] _[t]_ [)] _, x_ [(] _[t]_ [)] ) _used for the iterates in Tlo._ _If_
1               - _x_ [(] _[t]_ [)] _> M_ (1 + _ϵ_ )
����� _Tlo_ _t∈T_ �����


- _x_ [(] _[t]_ [)] _> M_ (1 + _ϵ_ )

_t∈Tlo_ �����2 _p_


_then there exists a coordinate i for which_

      

_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


_q_ +1

- 2
_αi_ [(] _[t]_ [)] _≥_ _[T][lo][ϵ]_ 2 _._


_Proof._ Suppose that

1
����� _Tlo_


Note that by the update rule,


- _x_ [(] _[t]_ [)] _> M_ (1 + _ϵ_ )

_t∈Tlo_ �����2 _p_


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ �� [)] _[q]_ _r_ ( _t_ )�� _qq−_ 1


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


_x_ [(] _i_ _[t]_ [)]
_M_ _[≤]_ [(1 +] _[ ϵ]_ [)]


1
2


~~�~~


~~�~~


- - _q−_ 1
_ri_ [(] _[t]_ [)]

�� _r_ ( _t_ )�� _qq−_ 1 + **1** _αi>_ 1


- - _q−_ 1
_ri_ [(] _[t]_ [)]


�� _r_ ( _t_ )�� _qq−_ 1 + **1** _αi>_ 1


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


 _≤_ 1 + _[ϵ]_


- - _q−_ 1
_ri_ [(] _[t]_ [)]


2              - �� _r_ ( _t_ )�� _qq−_ 1 _i_              - �� _r_ ( _t_ )�� _qq−_ 1


Hence we can write


2







_i_


���������2 _p_


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 





�����


���������


_≤_
�����
2 _p_


2


~~�~~
��� ��� _rr_ [(] ( _[t]_ _t_ [)] ) [�] �� _[q]_ _qq_ _[−]_ _−_ [1] 1 +


16


 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


1 + _[ϵ]_


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


�� _r_ ( _t_ )�� _qq−_ 1


��


_t∈Tlo_


 



_t∈Tlo_


_x_ [(] _[t]_ [)]


_M_


~~�~~  ~~�~~    -    -    - _q−_ 1 

_≤_ �1 + 2 _[ϵ]_ �� _t∈Tlo_ ��������� ��� _rr_ [(] ( _[t]_ _t_ [)] ) [�] �� _[q]_ _qq_ _[−]_ _−_ [1] 1 ������2 _p_ +  _t∈Tlo_ - _,α_ [(] _i_ _[t]_ [)] _>_ 1 ��� _[α]_ _i_ [(] _[t]_ �� [)] _[q]_ _r_ ( _tr_ )�� _i_ [(] _[t]_ _qq_ [)] _−_ 1 

��������� _i_ ���������2 _p_


(by triangle inequality)







_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1





������2 _p_


 _≤_ 1 + _[ϵ]_


��


+


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


���������


 



2


_t∈Tlo_


������


~~�~~
�� - _r_ [(] _[t]_ [)][�] _[q][−]_ [1]

- �� _r_ ( _t_ )�� _qq−_ 1


�� _r_ ( _t_ )�� _qq−_ 1


_i_


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 







_i_


���������2 _p_





 = 1 + _[ϵ]_

2


���������


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


_Tlo_ +


 



~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


_._


�� _r_ ( _t_ )�� _qq−_ 1


We obtain
���������


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 





���������2 _p_


 



~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


_≥_ _[ϵ]_

2 _[T][lo]_







_i_


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


�� _r_ ( _t_ )�� _qq−_ 1


On the other hand, we have


~~~~ 2 _p_







2 _p_


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [+1)]


- _[α]_ [(] _[t]_ [)] �� _r_ ( _t_ )�� _qq−_ 1


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ �� [)] _[q]_ _r_ ( _t_ )�� _qq−_ 1


2 _p_





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1





_i_





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1







= 

_i_


_q_
_≤_ _r_ ( _T_ )
��� ��� _q_ [max] _i_


2 _p_


_αi_ [(] _[t]_ [)] 


 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


2 _p_






_≤_ 

_i_




- - _q_ _ri_ [(] _[T]_ [ )] 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


_αi_ [(] _[t]_ [)]


2 _p_






_≤_ [1]

_ϵ_ _[q]_ [max] _i_


_≤_ [1]





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


_αi_ [(] _[t]_ [)]


Therefore there exists _i_ such that




     


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


which gives us


      


2 _p_


~~�~~
_αi_ [(] _[t]_ [)]






 - _ϵT_ �2 _p_
_≥_ _ϵ_ _[q]_ _,_
2


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


Now we show the proof of Lemma 2.5.


_q_ +1

- 2
_αi_ [(] _[t]_ [)] _≥_ _[T][lo][ϵ]_ 2 _._


_Proof of Lemma 2.5._ From Lemma D.1 we know that there exists a coordinate _i_ for which


 

_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


_q_ +1

- 2
_αi_ [(] _[t]_ [)] _>_ _[T][lo][ϵ]_ 2 _._ (6)


                             -                             - _q_
Furthermore by definition for all iterates in _Tlo_ we have that pointwise (1 + _ϵ_ ) _≤_ _αi_ [(] _[t]_ [)] _≤_ _S_ _[q]_ .

This enables us to lower bound the final value of - _ri_ [(] _[T]_ [ )] - _q_ which is a lower bound on �� _r_ ( _T_ )�� _qq_ [.] [More]


17


precisely, we have

     - _ri_ [(] _[T]_ [ )]      - _q_ _≥_      - _ri_ [(0)]      - _q_ _·_      
_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


- - _q_
_αi_ [(] _[t]_ [)] = [1]


 _n_ _[·]_


- - _q_
_αi_ [(] _[t]_ [)] _._ (7)


_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


Now we can proceed to lower bound this coodinate i.e. we lower bound the product in (7) using the
lower bound we have in (6).


                       -                       - _q_
Intuitively, the worst case behavior i.e. slowest possible increase in _ri_ [(] _[T]_ [ )] is achieved in one of the
two extreme cases:

(i) the _αi_ [(] _[t]_ [)] are all minimized i.e. - _αi_ [(] _[t]_ [)] - _q_ = (1 + _ϵ_ ) in which case Θ - 1 _ϵ_ [log] - _ϵn_ _[q]_ �� such terms are

sufficient to make their product _≥_ _ϵ_ _[n][q]_ [, which means that we are done, since then we have] �� _r_ ( _T_ )�� _qq_ _[≥]_

- _ri_ [(] _[T]_ [ )] - _q_ _≥_ _ϵ_ 1 _[q]_ [;] [so] [setting] _[T][lo][ϵ]_ 2 _q_ +12 _≥_ Θ �(1 + _ϵ_ ) 21 _q_ [1] _ϵ_ [log] - _ϵn_ _[q]_ - [�] i.e _Tlo_ _≥_ Θ - _q_ 1+3 log - _ϵn_ _[q]_ - [�] is


sufficient to make their product _≥_ _[n]_


_[ϵ]_ 2 2 _≥_ Θ �(1 + _ϵ_ )


[1] _ϵ_ [log] - _ϵn_ _[q]_ - [�] i.e _Tlo_ _≥_ Θ - _q_ 1


1
2 _q_ [1]


_q_ +3
_ϵ_ 2


   - _n_    - [�]

+32 log _ϵ_ _[q]_ is


sufficient to make this happen;


(ii) all the entries are maximized, i.e. _αi_ [(] _[t]_ [)] = _S_ in which case we have that their product to power _q_ is


_ϵ_ _[n][q]_ [, so if we set] _S_ _[qT]_ [1] _[/][lo]_ [2]


   - _n_    - [�]

+12 log _ϵ_ _[q]_,


_q_ +1

_[qT][lo]_ _ϵ_ 2

_S_ [1] _[/]_ [2] 2


22 ln _S_ _≥_ log - _ϵn_ _[q]_ �, ie., _Tlo_ = Θ - _qS_ ln [1] _[/]_ _S_ [2] _q_ 1


at least _S_


_q_ +1
_qTlo_ _ϵ_ 2
_S_ [1] _[/]_ [2] 2


2
2 _≥_ _[n]_


we guarantee that the corresponding _ri_ increases to a value larger than _ϵ_ [1]


_q_ +1
_ϵ_ 2


we guarantee that the corresponding _ri_ increases to a value larger than _ϵ_ _[q]_ [.] [The fact that these two]

cases capture the slowest possible increase is shown in Lemma F.1.


Therefore we can set


�� 1
_Tlo_ = _O_ _[S]_ [1] _[/]_ [2]
_ϵ_ [+] _q_ ln _S_


_q_ +1
_ϵ_ 2


  - _n_

+12 log _ϵ_ _[q]_


- 1


- [�]
_._


E PROOF OF THEOREM 1.2


First, we give guarantee for the subproblem solver (Algorithm 4, proof follows subsequently) .


**Lemma E.1.** _For p ≥_ 1 _, κ_ =


�1 _if p ≤_ loglog _n n−_ 1 _, Algorithm 4 either returns x such that Ax_ = _b,_
_q_ _otherwise_


_∥x∥_ 2 _p_ _≤_ 2 _M_ _and_ - _θ, x_ [2][�] _≤_ min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�] _or_ _certifies_ _that_ min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+]

- _θ, x_ [2][�] _≥_ _[M]_ 2 _κ_ [2] _[in][ O]_ - _n_ 2 _q_ 1+1 - _calls to solve a linear system of the form ADA_ _[⊤]_ _ϕ_ = _b, where D is an_

_arbitrary non-negative diagonal matrix._


The next lemma provides guarantees on the iterate progress in the main algorithm (Algorithm 3).


**Lemma E.2.** _For p ≥_ 2 _κ_ =


�1 _p_ _if p ≤_ log2 log _n− n_ 1 _, Algorithm 3 maintains that_ �� _x_ ( _t_ )�� _pp_ _[−∥][x][∗][∥]_ _p_ _[p]_ _[≤]_
_p−_ 2 _otherwise_


16 _pM_ [(] _[t]_ [)] _and that if x_ [(] _[t]_ [+1)] = _x_ [(] _[t]_ [)] _then_

_p_
��� _x_ ( _t_ +1)��� _[−∥][x][∗][∥]_ _p_ _[p]_


_._


_p_ - 1

_p_ _[≤]_ 1 _−_
_p_ _[−∥][x][∗][∥][p]_ 2 [13] _pκ_


_p_
����� _x_ ( _t_ )��� _p_ _[−∥][x][∗][∥]_ _p_ _[p]_


Finally, we show the proof of Theorem 1.2.


_Proof._ Algorithm 3 terminates when _M_ [(] _[t]_ [)] _≤_ 16 _p_ (1+ _ϵ_ _ϵ_ ) �� _x_ ( _t_ )�� _pp_ [.] [This] [gives] �� _x_ ( _t_ )�� _pp_ _[−∥][x][∗][∥]_ _p_ _[p]_ _[≤]_

16 _p_ (1+ _ϵ_ _ϵ_ ) �� _x_ ( _t_ )�� _pp_ [, which implies] �� _x_ ( _t_ )�� _pp_ _[≤]_ [(1 +] _[ ϵ]_ [)] _[ ∥][x][∗][∥]_ _p_ _[p]_ [and thus] �� _x_ ( _t_ )�� _p_ _[≤]_ [(1 +] _[ ϵ]_ [)] _[ ∥][x][∗][∥][p]_ [.] [Hence,]

_x_ [(] _[t]_ [)] is a (1+ _ϵ_ ) approximate solution. Since 16 _p_ (1+ _ϵ_ _ϵ_ ) �� _x_ ( _t_ )�� _pp_ _[≥]_ 16 _p_ (1+ _ϵ_ _ϵ_ ) _[∥][x][∗][∥]_ _p_ _[p]_ [, the number of times]


       - _p_
_M_ [(] _[t]_ [)] can be reduced is _O_ log _[∥][x]_ [(0)] _[∗]_ _[∥][p]_ _p_


_p_
_ϵ∥x_ _[∗]_ _∥_ _[p]_ _p_


= _O_ - _p_ log _[n]_ _ϵ_ �. By Lemma E.2, the number of times the


18


- _p_

_p_ _[−∥][x][∗][∥]_ _p_ _[p]_

iterate makes progress is _O_ 2 [13] _pκ_ log _[∥][x]_ [(0)] _ϵ∥_ _[∥]_ _x_ _[∗]_ _∥_ _[p]_ _p_


= _O_ - _p_ [2] log _n_ log _[n]_ _ϵ_ - where _κ_ = _O_ (log _n_ ).


Therefore the total number of calls to the subroutine solver is _O_ - _p_ [2] log _n_ log _[n]_


          - 1          -          subroutine solver makes _O_ _n_ 2 _q_ +1 = _O_ _n_


          - 1          -          - _p−_ 2          subroutine solver makes _O_ _n_ 2 _q_ +1 = _O_ _n_ 3 _p−_ 2 calls to a linear system solver. This concludes


the proof.


_ϵ_ �. By lemma E.1, the


E.1 PROOF OF LEMMA E.1


We let _OPT_ = min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�] and _x_ _[∗]_ = arg min _x_ : _Ax_ = _b_ �� _x_ 2�� _p_ [+] - _θ, x_ [2][�] . We consider

two cases: when _p ≤_ loglog _n n−_ 1 [and when] _[ p >]_ loglog _n n−_ 1 [.] [We will prove for each case using the following]
lemmas:
**Lemma E.3.** _For_ 1 _≤_ _p_ _≤_ loglog _n n−_ 1 _[, Algorithm 4 either returns][ x][ such that][ Ax]_ [=] _[b][,][ ∥][x][∥]_ 2 _p_ _[≤]_ [2] _[M]_

_and_ - _θ, x_ [2][�] _≤OPT_ _or certifies that OPT_ _≥_ _[M]_ 2 [2] _[in][ O]_ [(1)] _[ call to solve a linear system.]_

**Lemma E.4.** _For p_ _>_ loglog _n n−_ 1 _[, Algorithm 4 either returns][ x][ such that][ Ax]_ [=] _[b][,][ ∥][x][∥]_ 2 _p_ _[≤]_ [2] _[M]_ _[and]_

- _θ, x_ [2][�] _≤OPT_ _or certifies that OPT_ _≥_ _[M]_ 2 _q_ [2] _[in][ O]_ - _n_ 2 _q_ 1+1 - _calls to solve a linear system._


To start, we have the following lemma that controls the _ℓ_ 2 term in the objective
**Lemma** **E.5.** _For_ _r_ _such_ _that_ _∥r∥q_ _≤_ 1 _,_ _suppose_ _x_ = arg min _x_ : _Ax_ = _b⟨r_ + _θ, x_ [2] _⟩._ _Then_ _we_ _have_

- _θ, x_ [2][�] _≤OPT ._


_Proof._ For _r_ with _∥r∥q_ _≤_ 1, we have

        - _θ, x_ [2][�] _≤⟨r_ + _θ, x_ [2] _⟩≤⟨r_ + _θ,_ ( _x_ _[∗]_ ) [2] _⟩_ (by definition of _x_ )

_≤_ ��( _x∗_ )2�� _p_ [+]        - _θ,_ ( _x_ _[∗]_ ) [2][�] _≤OPT ._


Now, let us turn to the first case when 1 _≤_ _p ≤_ loglog _n n−_ 1 [.] [We give the proof for Lemma E.3.]


_Proof of Lemma E.3._ When 1 _≤_ _p ≤_ loglog _n n−_ 1 [, we have] _[ q]_ [=] _p−p_ 1 _[≥]_ [log] _[ n]_ [.] [Algorithm 4 computes]

_x_ = min              - _r_ + _θ, x_ [2][�]

              - _x_ : _Ax_ = _b_

where _ri_ = _n_ _[−]_ _q_ [1] for all _i_ .

Since _∥r∥q_ = 1, if _∥x_ - _∥_ 2 _p_ _≤_ 2 _M_, by Lemma E.5, we immediately have _∥x_ - _∥_ 2 _p_ _≤_ 2 _M_ and - _θ, x_ [2][�] _≤_
_OPT_ .


Assume that _∥x_ - _∥_ 2 _p_ _>_ 2 _M_ . We have

_OPT_ = ���( _x∗_ )2��� _p_ [+]     - _θ,_ ( _x_ _[∗]_ ) [2][�] _≥_     - _r,_ ( _x_ _[∗]_ ) [2][�] +     - _θ,_ ( _x_ _[∗]_ ) [2][�]


        -         = _θ_ + _r,_ ( _x_ _[∗]_ ) [2][�] _≥_ _θ_ + _r,_ ( _x_ ) [2][�]

                   


_≥_ [1] 1

_n_ _q_


��� _x_ 2��1 _[≥]_ _n_ [1] _q_ 1


��� _x_ 2�� _p_ (since ��� _x_ 2��1 _[≥]_ ��� _x_ 2�� _p_ [)]


_≥_ [1] 2 _[∥][x]_ [�] _[∥]_ 2 [2] _p_ (since _q_ _≥_ log _n_ )

_≥_ 2 _M_ [2] _≥_ _[M]_ [ 2]

2 _[.]_


For the case when _p_ _>_ loglog _n n−_ 1 [,] [the] [proof] [for] [Lemma] [E.4] [follows] [similarly] [to] [the] [analysis] [of]
Algorithm 2. We proceed by showing the following invariant.


19


**Lemma E.6** (Invariant) **.** _For all t, we have that if γ_ [(] _[t]_ [)] = 1 _then_ _[E]_ [(] _[r]_ [(] _[t]_ [+1)][+] _[θ]_ [)] _[−E]_ [(] _[r]_ [(] _[t]_ [)][+] _[θ]_ [)] _≥_ _M_ [2] _._

_∥_ _[r]_ [(] _[t]_ [+1)] _∥q_ _[−]_ _∥_ _[r]_ [(] _[t]_ [)] _∥q_


_Proof._ Using Lemma A.2 we have


��


_E_ ( _r_ [(] _[t]_ [+1)] + _θ_ ) _−E_ ( _r_ [(] _[t]_ [)] + _θ_ ) _q ·_ �� _r_ ( _t_ )�� _qq−_ 1
_≥_
�� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_


_q ·_ �� _r_ ( _t_ )�� _qq−_ 1
=


_q ·_ �� _r_ ( _t_ )�� _qq−_ 1
_≥_


_q ·_ �� _r_ ( _t_ )�� _qq−_ 1
=


�� - �� �2 [�] _ri_ [(] _[t]_ [)] + _θi_
_i_ _ri_ [(] _[t]_ [)] + _θi_ _x_ [(] _i_ _[t]_ [)] 1 _−_ _ri_ [(] _[t]_ [+1)] + _θi_

   -   - _q_   -   - _q_

 - _i_ _ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


�� - �2 _ri_ [(] _[t]_ [)] + _θi_

_i_ _x_ [(] _i_ _[t]_ [)] _ri_ [(] _[t]_ [+1)] + _θi_


- - [�]
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


  -  - _q_  -  - _q_

- _i_ _ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


�� - �2 _ri_ [(] _[t]_ [)]

_i_ _x_ [(] _i_ _[t]_ [)] _ri_ [(] _[t]_ [+1)]


- - [�]
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


  -  - _q_  -  - _q_

- _i_ _ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


��


- - [�]
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


_i,α_ [(] _i_ _[t]_ [)] _>_ 1


- _x_ [(] _i_ _[t]_ [)] �2 _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)]


_i,α_ [(] _i_ _[t]_ [)] _>_


- - _q_ - - _q_ _,_
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


where in the second inequality we use _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)] ++ _θiθi_ _[≥]_ _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)] for _ri_ [(] _[t]_ [+1)] _≥_ _ri_ [(] _[t]_ [)][,] _[ θ]_ _[≥]_ [0][.] [For] _[ i]_ [ such that]

_αi_ [(] _[t]_ [)] _>_ 1, we have _ri_ [(] _[t]_ [+1)] = _αi_ [(] _[t]_ [)] _[r]_ _i_ [(] _[t]_ [)][, thus]


�1 - _x_ [(] _i_ _[t]_ [)] ��2 _q_ _ri_ [(] _r_ _[t]_ _i_ [(][+1)] - _[t]_ [)] - _r_ - _i_ [(] _q_ _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)] - = _γi_ [(] _[t]_ [)] _[M]_ [ 2] _[ ·]_ _q_ ��1 _−_ - _qα_ 1 [(] _i_ _[t]_ [)]
_ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)] _αi_ [(] _[t]_ [)] _−_


- - _q_
_αi_ [(] _[t]_ [)] _−_ 1


_q ·_ �� _r_ ( _t_ )�� _qq−_ 1


- _x_ [(] _i_ _[t]_ [)] �2 _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)]


- _ri_ [(] _[t]_ [+1)] _−_ _ri_ [(] _[t]_ [)]


1
_≥_ _γi_ [(] _[t]_ [)] _[M]_ [ 2] _[ ·]_                -                - _q_
_αi_ [(] _[t]_ [)]


= _M_ [2] _,_

where the first inequality is due to _αq_ (( _αα_ _[q]_ _−−_ 1)1) _[≥]_ _α_ 1 _[q]_ [,] [for] _[α]_ _[>]_ [1][.] [We] [can] [then] [obtain] [the] [desired]
conclusion from here.

**Lemma** **E.7** (Case 1) **.** _Let_ _r_ _be_ _a_ _dual_ _solution_ _and_ _x_ = arg min _x_ �: _Ax_ �= _b⟨r_ + _θ,_ - _x_ [2] _⟩._ _If_
��� _∥r∥qq−_ 1 _·_ _r_ _[q]_ _x_ _[−]_ [2] [1] ��� _∞_ _[≤]_ [2] _[M]_ _[then][ ∥][x][∥]_ [2] _[p]_ _[≤]_ [2] _[M]_ _[and]_ - _θ, x_ [2][�] _≤OPT ._


_Proof._ If
���� _∥r∥qq−_ 1 _·_ _rx_ _[q][−]_ [2][1]


for all _i_ we have


_≤_ 2 _M_ [2] _,_
���� _∞_


which gives


We obtain


_[r]_ _i_ _[q][−]_ [1]
_x_ [2] _i_ _[≤]_ [4] _[M]_ [ 2] _,_
_∥r∥_ _[q]_ _q_ _[−]_ [1]


_[r]_ _i_ _[q]_
_x_ [2] _i_ _[p]_ _≤_ 2 [2] _[p]_ _M_ [2] _[p]_ _∥r∥_ _[q]_ _q_ _,_


_∥x∥_ [2] 2 _[p]_ _p_ _[≤]_ [2][2] _[p][M]_ [ 2] _[p][,]_


as needed. The second claim comes directly from Lemma E.5.


20


-                 _r_ [(] _[T]_ [ )] [2]
**Lemma E.8** (Case 3) **.** _If the algorithm returns r_ [(] _[T]_ [ )] _, then E_ _∥_ _[r]_ [(] _[T]_ [ )] _∥q_ + _θ_ _≥_ _[M]_ 2 _q_ _[.]_


_Proof._ We have that


_E_ ( _r_ [(] _[T]_ [ )] + _θ_ ) [+] _[ θ]_ [) +] [�] _t_ _[T]_ =0 _[ −]_ [1]     - _E_ ( _r_ [(] _[t]_ [+1)] + _θ_ ) _−E_ ( _r_ [(] _[t]_ [)] + _θ_ )�
= _[E]_ [(] _[r]_ [(0)]
�� _r_ ( _T_ )�� _q_ �� _r_ ( _T_ )�� _q_


_≥_


_≥_


- _Tt_ =0 _−_ 1 ��� _r_ ( _t_ +1)�� _q_ _[−]_ �� _r_ ( _t_ )�� _q_ - _· M_ [2]

(due to the invariant)
�� _r_ ( _T_ )�� _q_

��� _r_ ( _T_ )�� _q_ _[−]_ �� _r_ (0)�� _q_ - _· M_ [2]

�� _r_ ( _T_ )�� _q_


2 _q−_ 1  
2 _q_
�� _r_ ( _T_ )�� _q_


(since _r_ (0) [2] _[q][ −]_ [1] )
��� ��� _q_ [=] 2 _q_


= _M_ [2] _·_


1 _−_


= _[M]_ [ 2] (since _r_ ( _T_ )

2 _q_ ��� ��� _q_ _[≥]_ [1][)] _[.]_


Finally since �� _r_ ( _T_ )�� _q_ _[≥]_ [1]


_≥_ _[E]_ [(] _[r]_ [(] _[T]_ [ )][ +] _[ θ]_ [)] _≥_ _[M]_ [ 2]
�� _r_ ( _T_ )�� _q_ 2 _q_ _[.]_


_E_


_r_ [(] _[T]_ [ )]
+ _θ_
�� _r_ ( _T_ )�� _q_


**Convergence Analysis** We run the algorithm for _T_ iterations. The algorithm terminates if at any
point it finds a solution _x_ that satisfies the desired bound (otherwise it is unable to further perturb the
dual solution). Otherwise, we show that it must finish very fast.


Suppose we run it for _T_ = _Thi_ + _Tlo_ iterations. Let the iterations in _Thi_ correspond to those where at
2
least a single _ri_ was scaled by _≥_ _S_ = _n_ 2 _q_ +1 . Let _Tlo_ be the remaining iterations.

**Lemma E.9.** _We have Thi_ _≤_ _S_ [2] _[n][q]_ _[.]_


_Proof._ Suppose the contrary. Then we claim that these perturbations alone will have increased _r_
a lot to the point where _∥r∥_ _[q]_ _q_ _[≥]_ [1][.] [Indeed,] [let] _[r][i]_ [be] [the] [current] [value] [of] [coordinate] _[i]_ [and] _[r]_ _i_ _[′]_ [be]


_i_
its value after being increased, and assume that _[r]_ _ri_ _[′]_ _[≥]_ _[S]_ [.] [Since] _[ r]_ [is initialized to] [2] _[q]_ 2 _[−]_ _q_ [1]


_i_ 1
its value after being increased, and assume that _[r]_ _ri_ _[≥]_ _[S]_ [.] [Since] _[ r]_ [is initialized to] [2] _[q]_ 2 _[−]_ _q_ [1] _n_ [1] _[/q]_ [, in the]

worst case each perturbation in _Thi_ touches a different _i_ . Therefore this establishes a lower bound


_[q]_  - 2 _q−_ 1

_n_ 2 _q_


of _Thi ·_ _[S][q]_


2 _−q_ 1 - _q_ _≥_ _Thi ·_ _[S]_ 2 _n_ _[q]_


of _Thi ·_ _[S]_ _n_ _[q]_ 2 _q_ 2 _−q_ 1 _≥_ _Thi ·_ _[S]_ 2 _n_ _[q]_ [on] _[ ∥][r][∥]_ _q_ _[q]_ [.] [As this must be at most][ 1][, since otherwise we obtained a]

good solution per Lemma E.8, we obtain the conclusion.


Now we claim that we can either look at the history produced in _Tlo_ and obtain an approximately
feasible solution, or a single coordinate _ri_ must have increased a lot.


**Lemma E.10.** _Consider the set of iterates_ ( _r_ [(] _[t]_ [)] _, x_ [(] _[t]_ [)] ) _used for the iterates in Tlo._ _If_

1                 - _x_ [(] _[t]_ [)] _>_ 2 _M_
����� _Tlo_ _t∈T_ �����


- _x_ [(] _[t]_ [)] _>_ 2 _M_

_t∈Tlo_ �����2 _p_


_then there exists a coordinate i for which_


      

_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


_αi_ [(] _[t]_ [)] _≥_ _[T]_ 4 _[lo]_ _[.]_


21


_Proof._ Suppose ��� _T_ 1 _lo_ - _t∈Tlo_ _[x]_ [(] _[t]_ [)][���] 2 _p_ _[>]_ [ 2] _[M]_ [.] [Note that by the update rule,]


�� _r_ ( _t_ )�� _qq−_ 1 + **1** _αi>_ 1


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


_x_ [(] _i_ _[t]_ [)] _√_
_M_ _[≤]_


~~�~~


2


- - _q−_ 1
_ri_ [(] _[t]_ [)]


_M_      - �� _r_ ( _t_ )�� _qq−_ 1 _i_      - �� _r_ ( _t_ )�� _qq−_ 1


Hence we can write







_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ ~~~~


 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


�� _r_ ( _t_ )�� _qq−_ 1


~~~~


_√_

���������


���������


~~�~~
��� ��� _rr_ [(] ( _[t]_ _t_ [)] ) [�] �� _[q]_ _qq_ _[−]_ _−_ [1] 1 +


�����


_≤_
�����
2 _p_


2 

_t∈Tlo_


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


_i_


���������2 _p_


_t∈Tlo_


_x_ [(] _[t]_ [)]


_M_


~~�~~  ~~�~~   -   -   - _q−_ 1 ~~~~

_≤_ _√_ 2 _t_ - _∈Tlo_ ��������� ��� _rr_ [(] ( _[t]_ _t_ [)] ) [�] �� _[q]_ _qq_ _[−]_ _−_ [1] 1 ������2 _p_ +  _t∈Tlo_ - _,α_ [(] _i_ _[t]_ [)] _>_ 1 ��� _[α]_ _i_ [(] _[t]_ �� [)] _[q]_ _r_ ( _tr_ )�� _i_ [(] _[t]_ _qq_ [)] _−_ 1 

��������� _i_ ���������2 _p_


(by triangle inequality)







���������


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ ~~~~


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


~~~~


������2 _p_


_√_
_≤_


2 


+


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


 



_t∈Tlo_


������


~~�~~
�� - _r_ [(] _[t]_ [)][�] _[q][−]_ [1]

- �� _r_ ( _t_ )�� _qq−_ 1


�� _r_ ( _t_ )�� _qq−_ 1


_i_


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 





���������2 _p_


_√_
= 2 _Tlo_ +


���������


 



~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_







_i_


_._


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


�� _r_ ( _t_ )�� _qq−_ 1


We obtain
���������


_−−−−−−−−−−−−−−−−−−−−−−−−−→_
 ~~�~~ 


���������2 _p_





 



~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ [)] _[q]_ _−_


 - _√_
_≥_ 2 _−_


 2 _Tlo_ _≥_ _[T][lo]_

2


 2 _Tlo_ _≥_ _[T][lo]_







_i_


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


�� _r_ ( _t_ )�� _qq−_ 1


On the other hand, we have


(since _r_ (0) 2 _q_
��� ��� _q_ [=] 2 _q −_ 1 [)]










~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [)]


- _[α]_ [(] _[t]_ �� [)] _[q]_ _r_ ( _t_ )�� _qq−_ 1





~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [+1)]


- _[α]_ [(] _[t]_ [)] �� _r_ ( _t_ )�� _qq−_ 1


~~�~~ - - - _q−_ 1

- _i_ _ri_ [(] _[t]_ [+1)]


- _[α]_ [(] _[t]_ [)] _−_


2 _p_


_i_





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


= 

_i_


 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


~~�~~
_αi_ [(] _[t]_ [)]


~~�~~
_αi_ [(] _[t]_ [)]





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


2 _p_







2 _p_






2 _p_






2 _p_






_≤_ 

_i_


- - _q_
_ri_ [(] _[T]_ [ )]

�� _r_ (0)�� _qq_


_≤_


�� _r_ ( _T_ )�� _qq_
�� _r_ (0)�� _qq_ max _i_


_αi_ [(] _[t]_ [)]


 - 2 _q_ - _q_
_≤_ max
2 _q −_ 1 _i_


2 _p_


_αi_ [(] _[t]_ [)] 


_≤_ 2 max
_i_





 


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


_,_ (since _q_ _≥_ 1)


Therefore there exists _i_ such that




     


_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


�2 _p_
_,_


~~�~~
_αi_ [(] _[t]_ [)]


2 _p_






_≥_ [1]

2


_≥_ [1]


- _Tlo_
2


22


which gives us

    

_t∈Tlo,α_ [(] _i_ _[t]_ [)] _>_ 1


_αi_ [(] _[t]_ [)] _≥_ _[T][lo]_


1


1
2 2 _p_ _[≥]_ _[T]_ 4 _[lo]_


_[lo]_ 1


1

2 2


[since] _[ p][ ≥]_ [1] _[.]_
4 _[,]_


This lemma enables us to upper bound _Tlo_ .

           - _S_ [1] _[/]_ [2]            **Lemma E.11.** _We have Tlo_ _≤_ Θ ln _S_ [ln] _[ n]_ [ + ln] _[ n]_ _._


_Proof._ From Lemma E.10 we know that there exists a coordinate _i_ for which


          
      - [(] _[t]_ [)] _[T][lo]_


_αi_ [(] _[t]_ [)] _>_ _[T][lo]_


_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


(8)
4 _[.]_


Furthermore by definition for all iterates in _Tlo_ we have that pointwise _αi_ [(] _[t]_ [)] = _ri_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)] _≤_ _S_ and

   - �1 _/q_ 1   -   - _q_
_αi_ [(] _[t]_ [)] = _γi_ [(] _[t]_ [)] _≥_ 2 _q_ . This enables us to lower bound the final value of _ri_ [(] _[T]_ [ )] which is a lower


( _T_ ) _q_ _i_
bound on �� _r_ �� _q_ [.] [More precisely, we have] _[r]_ [(] _r_ _[t]_ _i_ [(][+1)] _[t]_ [)] _≥_ _αi_ [(] _[t]_ [)] thus


- _ri_ [(] _[T]_ [ )] - _q_ _≥_ - _ri_ [(0)] - _q_ _·_ 
_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


- - _q_
_αi_ [(] _[t]_ [)] = [2] _[q][ −]_ [1]


_[ −]_ [1] _·_ [1] 
2 _q_ _n_ _[·]_


- - _q_
_αi_ [(] _[t]_ [)] _._ (9)


_t∈Tlo_ : _α_ [(] _i_ _[t]_ [)] _>_ 1


Now we can proceed to lower bound this _ri_ i.e. we lower bound the product in (9) using the lower
bound we have in (8).


                              -                              - _q_
Similarly to the previous section, the worst case behavior i.e. slowest possible increase in _ri_ [(] _[T]_ [ )] is
achieved in one of the two extreme cases:


1
(i) the _αi_ [(] _[t]_ [)] are all minimized i.e. _αi_ [(] _[t]_ [)] = 2 _q_ in which case Θ (ln _n_ ) such terms are sufficient to
make their product _≥_ 2 _n_ _≥_ 22 _qqn−_ 1 [, which means that we are done, since then we have] �� _r_ ( _T_ )�� _qq_ _[≥]_

- - _q_
_ri_ [(] _[T]_ [ )] _≥_ 1; so setting _Tlo_ _≥_ Θ (ln _n_ ) is sufficient to make this happen;


(ii) all the entries are maximized, i.e. _αi_ [(] _[t]_ [)] = _S_ in which case we have that their product to power _q_ is


at least _S_


_Tlo_ _q_


4 _TSlo_ [1] _[/]_ _q_ [2] _≥_ 2 _n_ _≥_ 22 _qqn−_ 1 [, so if we set] 4 _TSlo_ [1] _[/]_ _q_ [2] [ln] _[ S]_ _[≥]_ [ln 2] _[n]_ [, ie,] _[ T][lo]_ _[≥]_ [8] _[S]_ [1] _q_ _[/]_ ln [2][ ln(] _S_ _[n]_ [)]


at least _S_ 4 _S_ [1] _[/]_ [2] _≥_ 2 _n_ _≥_ 2 _q−_ 1 [, so if we set] 4 _Slo_ [1] _[/]_ [2] [ln] _[ S]_ _[≥]_ [ln 2] _[n]_ [, ie,] _[ T][lo]_ _[≥]_ _q_ ln _S_, we guarantee

that the corresponding _ri_ increases to a value larger than 2. The fact that these two cases capture the
slowest possible increase is shown in Lemma F.1.


Therefore we can set


   - _S_ 1 _/_ 2   _Tlo_ = _O_ _._

ln _S_ [ln] _[ n]_ [ + ln] _[ n]_


2
Finally, by the choice _S_ = _n_ 2 _q_ +1, we obtain the runtime guarantee.


               - 1                **Lemma E.12.** _Algorithm 4 terminates in O_ _n_ 2 _q_ +1 _iterations._


_Proof of Lemma E.4._ The proof of Lemma E.1 immediately follows from Lemmas E.7, E.8 and
E.12.


E.2 PROOF OF LEMMA E.2


_Proof of Lemma E.2._ We define the function res _x_ as follows

res _x_ (∆) = _⟨g,_ ∆ _⟩−_                - _R,_ ∆ [2][�] _−∥_ ∆ _∥_ _[p]_ _p_


23


where _g_ = _|x|_ _[p][−]_ [2] _x_, _R_ = 2 _|x|_ _[p][−]_ [2] . We use the following property of this function from Adil et al.
(2019a; 2024): For _λ_ = 16 _p_ and for all ∆


����

����


∆
_∥x∥_ _[p]_ _p_ _[−]_ _x −_
���� _p_


∆
_∥x∥_ _[p]_ _p_ _[−]_ _x −_ _λ_
���� _p_


We prove the claim by induction.


_p_


_≥_ res _x_ (∆) ; (10)
_p_


_p_


_≤_ _λ_ res _x_ (∆) _._ (11)
_p_


[(0)] _[∥]_ _p_ _p_

16 _p_ _p_ _≥_ _[∥][x]_ [(] _[t]_ [)] _[∥]_ 16 _p_


_p_
For _t_ = 0, we have _M_ [(0)] := _[∥][x]_ [(0)] _[∥]_ _p_


_p_ _[−∥][x][∗][∥]_ _p_ _[p]_
16 _p_ .


Now assume that we have �� _x_ ( _t_ )�� _pp_ _[−∥][x][∗][∥]_ _p_ _[p]_ _[≤]_ [16] _[pM]_ [ (] _[t]_ [)][.] [We have two cases.]


Case 1. ResidualSolver returns an infeasibility certificate or ResidualSolver returns a primal solution

    ˜∆ such that _R_ [(] _[t]_ [)] _,_ ∆ [˜] [2][�] _≥_ 2 _M_ [(] _[t]_ [)] . In both scenarios, using Lemma E.1 we have


min
_A_ ∆=0
_⟨_ _[g]_ [(] _[t]_ [)] _[,]_ [∆] _⟩_ [=] _[M]_ 2 [(] _[t]_ [)]


��∆2�� _p_ 2 _−p_ _p_ - _R_ [(] _[t]_ [)] _,_ ∆ [2][�] _≥_ 2( _M_ [(] _[t]_ [)] ) _p_ 2 _._
2 [+ (] _[M]_ [ (] _[t]_ [)][)]


Hence for all ∆ such that _A_ ∆= 0, - _g_ [(] _[t]_ [)] _,_ ∆� = _[M]_ [(] _[t]_ [)]


2 [(] _[t]_ [)] [, either] ��∆2�� _p_ 2 _[≥]_ [(] _[M]_ [ (] _[t]_ [)][)] _p_ 2 _⇔∥_ ∆ _∥_ _[p]_ _p_ _[≥]_ _[M]_ [ (] _[t]_ [)]


or ( _M_ [(] _[t]_ [)] )


2 _−p_


_p_ _p_ [�] _R_ [(] _[t]_ [)] _,_ ∆ [2][�] _≥_ ( _M_ [(] _[t]_ [)] ) _p_ 2 _⇔_ - _R_ [(] _[t]_ [)] _,_ ∆ [2][�] _≥_ _M_ [(] _[t]_ [)] . For all ∆ such that _A_ ∆= 0, we


can write - _g_ [(] _[t]_ [)] _,_ ∆� = _a_ _[M]_ 2 [(] _[t]_ [)] [,] [for] [some] [constant] _[a]_ _[∈]_ [R][.] [We] [obtain] [either] _[∥]_ [∆] _[∥][p]_ _p_ _[≥]_ _[a][p][M]_ [ (] _[t]_ [)] [or]

- _R_ [(] _[t]_ [)] _,_ ∆ [2][�] _≥_ _a_ [2] _M_ [(] _[t]_ [)], and thus for all ∆


      - 1      res _x_ ( _t_ ) (∆) _≤_ _M_ [(] _[t]_ [)] 2 _[a][ −]_ [min] _a_ [2] _, a_ _[p]_ [��] _≤_ _[M]_ 2 [ (] _[t]_ [)] = _M_ [(] _[t]_ [+1)] _._


We write ∆= _[x]_ [(] _[t]_ [)] _[−][x][∗]_


_λ/p_, for _λ_ = 16 _p_ . Using property (11) of the res _x_, we have

_p_ _p_
��� _x_ ( _t_ +1)��� _[−∥][x][∗][∥]_ _p_ _[p]_ [=] ��� _x_ ( _t_ )��� _[−∥][x][∗][∥]_ _p_ _[p]_


_p_ _p_

_p_ _[−∥][x][∗][∥]_ _p_ _[p]_ [=] ��� _x_ ( _t_ )��� _p_


_p_
_p_ _[−∥][x][∗][∥][p]_


_p_ _x_ ( _t_ ) _−_ _λ_ ∆

_p_ _[−]_ ���� _p_


= _x_ ( _t_ ) _p_ _x_ ( _t_ ) _−_ _λ_ ∆
��� ��� _p_ _[−]_ ���� _p_ ���� _p_


     - ~~�~~
_≤_ _λ_ res _x_ ( _t_ ) ∆


_p_
= _x_ ( _t_ )
��� ���


_≤_ 16 _pM_ [(] _[t]_ [+1)] _._


Case 2. We have - _R,_ ∆ [˜] [2][�] _<_ 2 _M_ [(] _[t]_ [)] and ���˜∆��� _p_ _[≤]_ [4] _[√][κ]_ [(] _[M]_ [ (] _[t]_ [)][)] _p_ 1 and - _g,_ ∆ [˜] - = _[M]_ 2 [(] _[t]_ [)]


_p_


_p_


_p_
_x_ ( _t_ )
��� ���


_p_ _p_

_x_ ( _t_ +1)
_p_ _[−]_ ��� ��� _p_


_p_ _p_

_x_ ( _t_ )
_p_ [=] ��� ��� _p_


_p_ _[−]_


_x_ ( _t_ ) _−_ ∆˜
64 _pκ_
�����


�����


- ∆˜

64 _κ_


�2�


_≥_ res _x_ ( _t_ )


- ∆˜

64 _κ_


�����


_p_


_p_


_R,_


�����


∆˜

64 _κ_


_−_


=


- ∆˜

_g,_
64 _κ_


_−_


[ (] _[t]_ [)]

2 [7] _κ_ _[−]_ 2 _[M]_ [11][ (] _κ_ _[t]_ [)]

[ (] _[t]_ [)]

2 [7] _κ_ _[−]_ _[M]_ 2 [11][ (] _[t]_ _κ_ [)]


(since _p ≥_ 2 _, κ ≥_ 1)
2 [8] _κ_ _[,]_


_≥_ _[M]_ [ (] _[t]_ [)]


_[M]_ [ (] _[t]_ [)] _p_

2 [11] _κ_ [2] _[−]_ 2 _[M]_ [4] _[p]_ [ (] _κ_ _[t]_ [)] 2


_p_
2 [4] _[p]_ _κ_ 2


_≥_ _[M]_ [ (] _[t]_ [)]


_[M]_ [ (] _[t]_ [)]

2 [11] _κ_ _[−]_ _[M]_ 2 [8][ (] _κ_ _[t]_ [)]


24


1

_≥_ _[M]_ [ (] _[t]_ [)]

2 [9] _κ_ _[≥]_ 2 [13] _pκ_


_p_
���� _x_ ( _t_ )��� _p_ _[−∥][x][∗][∥]_ _p_ _[p]_


_,_


from which we obtain

_p_
_x_ ( _t_ +1)
��� ���


_p_ 1

_p_ _[−]_
_p_ _[−∥][x][∗][∥][p]_ 2 [13] _pκ_


_p_ _p_

_p_ _[−∥][x][∗][∥]_ _p_ _[p]_ _[≤]_ ��� _x_ ( _t_ )��� _p_


_p_
���� _x_ ( _t_ )��� _p_ _[−∥][x][∗][∥]_ _p_ _[p]_


          - 1
_≤_ 1 _−_
2 [13] _pκ_


as needed.


F LOWER BOUND LEMMA


_p_
����� _x_ ( _t_ )��� _p_ _[−∥][x][∗][∥]_ _p_ _[p]_


1

**Lemma F.1.** _Let a set of nonnegative reals β_ 1 _, . . ., βk such that_ 1+ _ϵ ≤_ _βi_ _≤_ _S, and_ [�] _i_ _[k]_ =1 _[β]_ _ir_ _[≥]_ _[K][,]_

_where r_ _≥_ 2 _._ _Then for any k one has that_


_k_ 
- _βi_ _≥_ min _S_


_i_ =1


_K_ _K_ _S_ [1] _[/r]_ _,_ (1 + _ϵ_ ) (1+ _ϵ_ ) [1] _[/r]_ _._


_Proof._ Consider a fixed _k_, and let us attempt to minimize the product of _βi_ ’s subject to the constraints.

1

W.l.o.g. we have [�] _i_ _[k]_ =1 _[β]_ _ir_ = _K_ . Equivalently we want to minimize [�] _i_ _[k]_ =1 [log(] _[β][i]_ [)][,] [which] [is] [a]


1

W.l.o.g. we have [�] _i_ _[k]_ =1 _[β]_ _ir_ = _K_ . Equivalently we want to minimize [�] _i_ _[k]_ =1 [log(] _[β][i]_ [)][,] [which] [is] [a]

concave function. Therefore its minimizer is attained on the boundary of the feasible domain. This
means that for some 0 _≤_ _k_ _[′]_ _≤_ _k_ _−_ 1, there are _k_ _[′]_ elements equal to 1+ _ϵ_, _k_ _−_ 1 _−k_ _[′]_ equal to _S_, and one
which is exactly equal to the remaining budget, i.e. - _K −_ _k_ _[′]_ (1 + _ϵ_ ) [1] _[/r]_ _−_ ( _k −_ 1 _−_ _k_ _[′]_ ) _S_ [1] _[/r]_ [�], which
yields the product (1 + _ϵ_ ) _[k][′]_ _S_ _[k][−][k][′][−]_ [1] _·_ - _K −_ _k_ _[′]_ (1 + _ϵ_ ) [1] _[/r]_ _−_ ( _k −_ 1 _−_ _k_ _[′]_ ) _S_ [1] _[/r]_ [�] . This can be relaxed
by allowing _k_ and _k_ _[′]_ to be non-integral. Hence we aim to minimize the product (1 + _ϵ_ ) _[k][′]_ _S_ _[k][−][k][′][−]_ [1]


subject to _k_ _[′]_ (1 + _ϵ_ ) [1] _[/r]_ _−_ ( _k −_ 1 _−_ _k_ _[′]_ ) _S_ [1] _[/r]_ = _K_ .


Finally, we observe that we can always obtain a better solution by placing all the available mass on a


_K_
(1+ _ϵ_ ) [1] _[/r]_, whichever is lowest.


single one of the factors, i.e. we lower bound either by _S_


G ITERATIVE REFINEMENT


_K_
_S_ [1] _[/r]_ or (1 + _ϵ_ )


In this section we provide a general technique for solving optimization problems to high-precision,
by reducing then to an adaptive sequence of easier optimization problems, which only require
approximate solutions. This formalizes the minimal requirements for the iterative refinement scheme
employed in Adil et al. (2019a;b) to go through. We state the main lemma below.


**Lemma G.1.** _Let D_ _⊆_ R _[n]_ _be a convex set, and let f_ : _D_ _→_ R _be a convex function._ _Let η_ _≥_ 0 _be a_
_scalar, and suppose that for any x_ _∈D there exists a function hx_ _that approximates the Bregman_
_divergence at x in the sense that_

1

_[.]_
_η_ _[h][x]_ [ (] _[ηδ]_ [)] _[ ≤]_ _[f]_ [ (] _[x]_ [ +] _[ δ]_ [)] _[ −]_ _[f]_ [ (] _[x]_ [)] _[ −⟨∇][f]_ [ (] _[x]_ [)] _[, δ][⟩≤]_ _[h][x]_ [ (] _[δ]_ [)]


_Given access to an oracle that for any direction v can provide κ-approximate minimizers to ⟨v, δ⟩_ +
_hx_ ( _δ_ ) _in the sense that it returns δ_ _[♯]_ _such that v_ + _δ_ _[♯]_ _∈D and_


             -              
           - _v, δ_ _[♯]_ [�] + _hx_           - _δ_ _[♯]_ [�] _≤_ [1] min _,_

_[⟨][v, δ][⟩]_ [+] _[ h][x]_ [ (] _[δ]_ [)]


_κ_


- min _,_
_v_ + _δ∈D_ _[⟨][v, δ][⟩]_ [+] _[ h][x]_ [ (] _[δ]_ [)]


_along with an initial point x_ 0 _∈D,_ _in O_ - _κη_ [ln] _[f]_ [(] _[x]_ [0][)] _[−]_ _ε_ _[f]_ [(] _[x][∗]_ [)] - _calls to the oracle one can obtain a_

_point x such that f_ ( _x_ ) _≤_ _f_ ( _x_ _[∗]_ ) + _ε, where x_ _[∗]_ _∈_ arg min _x∈D f_ ( _x_ ) _._


_Proof._ Let _δ_ _[♯]_ be the a _κ_ -approximate minimizer of - _∇f_ ( _x_ ) _, δ_ _[♯]_ [�] + _hx_ - _δ_ _[♯]_ [�], which by definition
satisfies:

             -              
        - _∇f_ ( _x_ ) _, δ_ _[♯]_ [�] + _hx_        - _δ_ _[♯]_ [�] _≤_ [1] min _._ (12)

_[⟨∇][f]_ [ (] _[x]_ [)] _[, δ][⟩]_ [+] _[ h][x]_ [ (] _[δ]_ [)]


_κ_


- min _._ (12)
_v_ + _δ∈D_ _[⟨∇][f]_ [ (] _[x]_ [)] _[, δ][⟩]_ [+] _[ h][x]_ [ (] _[δ]_ [)]


25


Updating our iterate to _x_ _[′]_ = _x_ + _δ_ _[♯]_ we can bound the new function value as

_f_      - _x_ + _δ_ _[♯]_ [�]

= _f_ ( _x_ ) +    - _∇f_ ( _x_ ) _, δ_ _[♯]_ [�] + _hx_    - _δ_ _[♯]_ [�] (Bregman divergence upper bound)


_≤_ _f_ ( _x_ ) + _[η]_

_κ_


_≤_ _f_ ( _x_ ) + _[η]_

_κ_

= _f_ ( _x_ ) + _[η]_


- _⟨∇f_ ( _x_ ) _, x_ _[∗]_ _−_ _x⟩_ + [1] (using (12))

_η_ _[h][x]_ [ (] _[η]_ [ (] _[x][∗]_ _[−]_ _[x]_ [))]


_κ_ [(] _[⟨∇][f]_ [ (] _[x]_ [)] _[, x][∗]_ _[−]_ _[x][⟩]_ [+ (] _[f]_ [ (] _[x][∗]_ [)] _[ −]_ _[f]_ [ (] _[x]_ [)] _[ −⟨∇][f]_ [ (] _[x]_ [)] _[, x][ −]_ _[x][∗][⟩]_ [))]


(Bregman divergence lower bound)


= _f_ ( _x_ ) + _[η]_ _[,]_

_κ_ [(] _[f]_ [ (] _[x][∗]_ [)] _[ −]_ _[f]_ [ (] _[x]_ [))]


from where we equivalently obtain that

_f_           - _x_ + _δ_ _[♯]_ [�] _−_ _f_ ( _x_ _[∗]_ ) _≤_ �1 _−_ _[η]_

_κ_


( _f_ ( _x_ ) _−_ _f_ ( _x_ _[∗]_ )) _._


Therefore to reduce the initial error _f_ ( _x_ 0) _−_ _f_ ( _x_ _[∗]_ ) to _ε_ it suffices to iterate _O_ - _κη_ [ln] _[f]_ [(] _[x]_ [0][)] _[−]_ _ε_ _[f]_ [(] _[x][∗]_ [)] 
times.


The following lemma provides a sandwiching inequality for the Bregman divergence of _∥x∥_ _[p]_ _p_ [.]


**Lemma G.2** (Adil et al. (2019b), Lemma B.1) **.** _For any x, δ and p ≥_ 2 _, we have for r_ = _x_ _[p][−]_ [2] _and_
_g_ = _px_ _[p][−]_ [1] _,_
_p_ 8     - _r, δ_ [2][�] + 2 _[p]_ 1 [+1] _[∥][δ][∥]_ _p_ _[p]_ _[≤∥][x]_ [ +] _[ δ][∥]_ _p_ _[p]_ _[−∥][x][∥]_ _p_ _[p]_ _[−⟨][g, δ][⟩≤]_ [2] _[p]_ [2][ �] _r, δ_ [2][�] + _p_ _[p]_ _∥δ∥_ _[p]_ _p_ _[.]_


As a corollary we see that the function _hx_ ( _δ_ ) = 2 _p_ [2][ �] _x_ _[p][−]_ [2] _, δ_ [2][�] + _p_ _[p]_ _∥δ∥_ _[p]_ _p_ [satisfies the inequality]
required by Lemma G.1 for _η_ = 41 _p_ [.] [We] [can] [thus] [conclude] [that] [given] [access] [to] [an] [oracle] [that]
approximately minimizes mixed _ℓ_ 2 + _ℓp_ regression objectives, one can efficiently generate a high
precision solution.

**Corollary G.1.** _Consider the ℓp regression problem_ min _f_ : _B⊤f_ = _d ∥f_ _∥_ _[p]_ _p_ _[.]_ _[Given access to an oracle]_
_that can compute κ-approximate minimizers to the optimization problem_
_V_ _[∗]_ := min       - _pf_ _[p][−]_ [1] _,_ ∆ _f_       - + 2 _p_ [2][ �] _f_ _[p][−]_ [2] _,_ ∆ _f_ [2][�] + _p_ _[p]_ _∥_ ∆ _f_ _∥_ _[p]_ _p_
_f_ : _B_ _[⊤]_ ∆ _f_ =0


_in the sense that it returns_ ∆ _f_ _satisfying B_ _[⊤]_ ∆ _f_ = 0 _and_

         - _pf_ _[p][−]_ [1] _,_ ∆ _f_          - + 2 _p_ [2][ �] _f_ _[p][−]_ [2] _,_ ∆ _f_ [2][�] + _p_ _[p]_ _∥_ ∆ _f_ _∥_ _[p]_ _p_ _[≤]_ [1] _[∗]_ _[,]_

_κ_ _[V]_

_along with an initial point f_ 0 _, satisfying B_ _[⊤]_ _f_ = _d, in O_ - _κp_ ln _∥f_ 0 _∥_ _[p]_ _p_ _[−∥]_ _ε_ _[f][ ∗][∥][p]_ _p_ - _calls to the oracle one_

_can obtain a point f_ _such that ∥f_ _∥_ _[p]_ _p_ _[≤∥][f][ ∗][∥]_ _p_ _[p]_ [+] _[ ε][, where][ f][ ∗]_ _[∈]_ [arg min] _[B][⊤][f]_ [=] _[d]_ _[∥][f]_ _[∥][p]_ _p_ _[.]_


_Proof._ Using Lemma G.2 we verify that the function _hf_ (∆ _f_ ) = 2 _p_ [2][ �] _f_ _[p][−]_ [2] _,_ ∆ _f_ [2][�] + _p_ _[p]_ _∥_ ∆ _f_ _∥_ _[p]_ _p_
satisfies
1 [+ ∆] _[f]_ _[∥][p]_ _p_ _[−∥][f]_ _[∥]_ _p_ _[p]_ [+]        - _pf_ _[p][−]_ [1] _,_ ∆ _f_        - _≤_ _hf_ (∆ _f_ )
_η_ _[h][f]_ [ (] _[η]_ [∆] _[f]_ [)] _[ ≤∥][f]_

for _η_ = 41 _p_ [.] [Therefore by Lemma G.1 we can need] _[ O]_ - _κp_ ln _∥f_ 0 _∥_ _[p]_ _p_ _[−∥]_ _ε_ _[f][ ∗][∥][p]_ _p_ - iterations to obtain an
_ε_ -additive error to the regression problem.


H ADDITIONAL EXPERIMENTAL RESULTS


**Correctness of solution.** In Figure 3, we plot the error of the solutions outputted by our algorithm
and _p_ -IRLS against CVX in the random matrices and random graphs instances for _ϵ_ = 10 _[−]_ [10] . In all
cases, the error is below _ϵ_ .


26


12.5


13.0


13.5


14.0


14.5


15.0


200 400 600 800 1000
n


3 4 5 6 7 8 9 10
p


13.0


13.5


14.0


14.5


15.0


3 4 5 6 7 8 9 10
p


10.0


10.5


11.0


11.5


12.0


12.5


100 200 300 400 500
n


10.0


10.5


11.0


11.5


12.0


12.5


13.0


(a) matrix size= _n_ _×_ ( _n_ _−_
50) _, p_ = 8


(b) matrix size=500 _×_ 400


(c) Graph of _n_ nodes, _p_ = 8(d) Graph of _n_ = 500 nodes


(c) Graph of _n_ nodes, _p_ = 8


Figure 3: Error of the solution against CVX/SDPT3 solution in log10 scale.


50


40


30


20


10


2 4 6 8 10


0.0
2 4 6 8 10


(b) matrix size=500 _×_ 400


3.5


3.0


2.5


2.0


1.5


1.0


0.5


50


40


30


20


10


2 4 6 8 10


2 4 6 8 10


20


15


10


5


0


(a) matrix size=500 _×_ 400


(c) Graph of _n_ = 500 nodes(d) Graph of _n_ = 500 nodes


(c) Graph of _n_ = 500 nodes


2 4 6 8 10


25


20


15


10


5


60


50


40


30


20


10


2 4 6 8 10


2 4 6 8 10


50


40


30


20


10


2 4 6 8 10


12


10


8


6


4


2


(e) matrix size=2500 _×_ 2400


(f) matrix size=2500 _×_ 2400


(g) Graph of _n_ = 10000
nodes


(h) Graph of _n_ = 10000
nodes


Figure 4: Performance when varying _ϵ_ on random matrices and random graphs instances.


**When varying** _ϵ_ **.** In Figure 4, we plot iteration complexity and runtime in seconds of our algorithm,
_p_ -IRLS and CVX when varying _ϵ_ . Note that, CVX does not allow varying this parameter. In all
experiment, we fix _p_ = 8. For large instances, we only consider our solution against _p_ -IRLS.


**For** 1 _< p <_ 2 **.** In Figure 5, we plot iteration complexity and runtime in seconds of our algorithm,
_p_ -IRLS and CVX on random matrices of size _n ×_ ( _n −_ 100). We fix _ϵ_ = 10 _[−]_ [10] . We test with _p_ = 1 _._ 1
and _p_ = 1 _._ 9.


500 750 1000 1250 1500 1750 2000 2250 2500
n


(b) _p_ = 1 _._ 1


500 750 1000 1250 1500 1750 2000 2250 2500
n


(c) _p_ = 1 _._ 9


175


150


125


100


75


50


25


0


200


175


150


125


100


75


50


25


0


42

40

38

36

34

32

30

28

26


500 750 1000 1250 1500 1750 2000 2250 2500
n


(d) _p_ = 1 _._ 9


60


55


50


45


40


35


30


500 750 1000 1250 1500 1750 2000 2250 2500
n


(a) _p_ = 1 _._ 1


Figure 5: Performance when _p_ = 1 _._ 1 and _p_ = 1 _._ 9 on random matrices of size _n ×_ ( _n −_ 100).


27
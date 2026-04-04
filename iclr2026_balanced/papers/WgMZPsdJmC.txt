# AN ANALYSIS OF THE CAUCHY METHOD FOR DIFFER## ENT STEPLENGTH COEFFICIENT
# CONFERENCE SUBMISSIONS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


In this work we take the parameter r (recipprocal of optimal steplenth) as analysis
target and introduce steplength coefficient t for classical steepest descent method
for convex quadratic optimization problems, and we found the different coefficients affect the state of the entire system convergence. As the value of t varies,
the overall system, including the value of r, may converge towards a fixed value,
oscillate between two regions, or display chaotic behavior. We also conducted a
specific analysis in the two-dimensional case.


1 INTRODUCTION


In this paper, we consider the the unconstrained optimization problem with convex quadratic form


_minf(x)_ = [1] (1)

2 _[x][T][ Ax][ −]_ _[b][T][ x]_


where _x ∈_ R _[n]_, _b ∈_ R _[n]_, _A ∈_ R _[n][×][n]_ is a symmetric and positive definite matrix.


The common solution methods for solving Eq(1) are iterarive methods of the following form


_xk_ +1 = _xk −_ _αk∇f(xk)_ (2)


where _αk_ is a steplenth,gradient descent method and its variants are the most common optimization
method.for GD method,if we minimizes Eq.(3) with exact line search,then we get


_k_ _[∇][f][k]_ _k_ _[g][k]_
_αk_ _[SD]_ = _∇_ _[∇]_ _f_ _[f]_ _k_ _[T][ T][A][∇][f][k]_ = _g_ _[g]_ _k_ _[T][T][Ag][k]_ (3)


1 _k_ _[Ag][k]_
_rk_ = = _[g][T]_ (4)
2 _αk_ 2 _gk_ _[T]_ _[g][k]_

this method proposed by A.Cauchy (1847) is called steepest descent method,so _αk_ _[SD]_ is also called
Cauchy step length. the method’s convergence rate is very sensitive to ill condition number and may
be very slow,when the f(x) is quadratic _xk_ will satisfy the


The convergence rate of SD method is relatively slow with a zigzag phenomena which is proved by
Akaike (1959) and Forsythe (1968)


Yuan (2006) proposed a new stepsize formula for the SD method. the method alternates as follows, on even iterations it employs the Yuan step size, while on odd iterations it performs an exact
line search. For two-dimensional convex quadratic functions, this alternating scheme guarantees
convergence to the minimum in only three iterations. Yuan steplength is as follows:


1


_f_ ( _xk_ +1) _−_ _f_ ( _x_ _[∗]_ )


[1] _[n]_

) [2] (5)
_λ_ 1 + _λn_


( _xk_ +1) _−_ _f_ ( _x_ ) _≤_ ( _[λ]_ [1] _[ −]_ _[λ][n]_

_f_ ( _xk_ ) _−_ _f_ ( _x_ _[∗]_ ) _λ_ 1 + _λn_


_rk_ +1 = _G_ ( _rk_ ) (14)


we will study the functional relationship of _G_ and the effect of parameters _t_ on the function _G_ .


2


2
_αk_ _[Y]_ [=] ~~�~~ ���� 1 _−_ 1 �2 + 4
_αk_ _[SD]_ _−_ 1 _αk_ _[SD]_ ( _αk_


�2


_∥g_ 2 _k∥_ [2]
+ 4


_∥g_ 2 _k∥_ 1 1

( _αk_ _[SD]_ _−_ 1 _[∥][g][k][−]_ [1] _[∥]_ [)][2] [+] _αk_ _[SD]_ _−_ 1 + _αk_ _[SD]_


(6)


Raydan M (2002) proposed RSD which accelerates convergence by introducing a relaxation parameter between 0 and 2 in the standard SD method,In each iteration, the step size is randomly chosen from a fixed interval in [0 _,_ 2 _αK_ _[SD]_ []][,this randomization eliminates the oscillations inherent in the]
SD method. RSD method convergs monotonically to optimal point _x_ _[∗]_ Serafino;F.Riccio;G.Toraldo
(2013) observes that over-relaxation appears more suitable than under-relaxation of the Cauchy
step size,therefore, they introduces a modified version of RSD, called RSDA method, where
_αk_ _∈_ [0 _._ 8 _αk,_ 2 _αk_ ].


Kalousek (2015) presents a randomized steepest descent method for minimizing smooth functions.
Instead of using exact step sizes, it randomly selects step lengths from a specific probability distribution,where _αk_ _∈_ [ _λ_ [1] 1 _[,]_ _λ_ 1 _n_ []]


In this paper,we take the parameter _r_ (Eq.(5)) as analysis target and introduce a multiplicative factor
parameter _s_ to the SD method and analyze how different values of _s_ affect the method. its formula
is as follows:


_xk_ +1 = _xk −_ _sαk_ _[SD]_ _∇f(xk)_ (7)


the same conclusion can be obtained by comparing the simplest form of qudratic function and matrix
form. For the convenience of analysis and greater intuitiveness,consider a situation the objective
function is a simple n dimensions hyper-ellipsoid stimulating Eq(1)


_f(x)_ =


_n_

- _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)2] (8)


_i_ =1


       - _n_       - _n_
_i_ =1 _[a]_ [(] _[i]_ [)3] _[x]_ [(] _[i]_ [)2] _i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ [(] _[i]_ [)2]
_r_ =              - _n_ [=]              - _n_ (9)
_i_ =0 _[a]_ [(] _[i]_ [)2] _[x]_ [(] _[i]_ [)2] _i_ =1 _[g]_ [(] _[i]_ [)2]

where 0 _<_ _a_ [(] _[n]_ [)] _≤_ _a_ [(] _[n][−]_ [1)] _≤_ _......_ _≤_ _a_ [(1)], _g_ [(] _[i]_ [)] = 2 _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)], the initial point _X_ 0 =

[ _x_ [(1)] 0 _[, x]_ 0 [(2)] _[, ......x]_ 0 [(] _[n]_ [)][]][ we have]


 - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ =


2


- _n_
_i_ =1 _[g]_ _k_ [(] _[i]_ [)]


2 (10)


- _ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2( _rk −_ _a_ ( _i_ ))2

- _ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2( _rk −_ _a_ ( _i_ ))2


  - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ +1 =


2 (11)
( _rk −_ _a_ ( _i_ ))2


now from Eq(7) then we have

_xk_ +1 = _xk −_ _sαk_ _[SD]_ _∇f(xk)_ = _xk −_ _[∇]_ _tr_ _[f(][x]_ _k_ _[k][)]_ (12)


where _s >_ 0 _, s_ = [1] _t_


- _ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2( _trk −_ _a_ ( _i_ ))2

- _ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2( _trk −_ _a_ ( _i_ ))2


  - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ +1 =


2 (13)
( _trk −_ _a_ ( _i_ ))2


2 TWO DIMENSION


In two dimensions case, we can analyse explicitly the positive quadratic case.


from Eq(11)


2
( _trk −_ _a_ (2))2


_rk_ +1 = _[a]_ [(1)] _[g]_ _k_ [(1)]

_gk_ [(1)]


2( _trk −_ _a_ (1))2 + _a_ (2) _gk_ (2)


2( _trk −_ _a_ (1))2 + _gk_ (2)


2 (15)
( _trk −_ _a_ (2))2


we treat _rk_ as a continuous variable _r_,we have


_G_ ( _r_ ) = _[a]_ [(1)][(] _[r][ −]_ _[a]_ [(2)][)(] _[tr][ −]_ _[a]_ [(1)][)][2] _[ −]_ _[a]_ [(2)][(] _[r][ −]_ _[a]_ [(1)][)(] _[tr][ −]_ _[a]_ [(2)][)][2] (16)

( _tr −_ _a_ [(1)] ) [2] ( _r −_ _a_ [(2)] ) _−_ ( _tr −_ _a_ [(2)] ) [2] ( _r −_ _a_ [(1)] )


where _r_ _∈_ ( _a_ [(2)] _, a_ [(1)] ) _, G_ ( _r_ ) _∈_ ( _a_ [(2)] _, a_ [(1)] ),differentiate the function _G_ ( _r_ )


_G_ ( _r_ ) _′_ = ( _tr −_ _a_ (1))( _tr −_ _a_ (2))( _a_ (1) _−_ _a_ (2))2

( _tr −_ _a_ [(1)] )( _tr −_ _a_ [(2)] ) _−_ 2 _t_ ( _r −_ _a_ [(1)] )( _r −_ _a_ [(2)] ) (17)
_×_

[( _tr −_ _a_ [(1)] ) [2] ( _r −_ _a_ [(2)] ) _−_ ( _tr −_ _a_ [(2)] ) [2] ( _r −_ _a_ [(1)] )] [2]


_′_ _′_
if we set _G_ ( _r_ ) to zero,we can obtain four solutions of _G_ ( _r_ ) :


_r_ 1 = _[a]_ [(1)] (18)

_t_


_r_ 2 = _[a]_ [(2)] (19)

_t_


_r_ 3 = _[a]_ [(1)][ +] _[ a]_ [(2)] _−_

2(2 _−_ _t_ )


_r_ 4 = _[a]_ [(1)][ +] _[ a]_ [(2)] +

2(2 _−_ _t_ )


~~�~~ _t_ [2] ( _a_ [(1)] + _a_ [(2)] ) [2] _−_ 4 _t_ (2 _−_ _t_ )(2 _t −_ 1) _a_ [(1)] _a_ [(2)]

(20)
2 _t_ (2 _−_ _t_ )


~~�~~ _t_ [2] ( _a_ [(1)] + _a_ [(2)] ) [2] _−_ 4 _t_ (2 _−_ _t_ )(2 _t −_ 1) _a_ [(1)] _a_ [(2)]

(21)
2 _t_ (2 _−_ _t_ )


we can find fixed points _re_ ( _re_ = _G_ ( _re_ ))obviously.


_re_ = _[a]_ [(1)][ +] _[ a]_ [(2)] (22)

2 _t_


put the _re_ into Eq(17)


_′_ 2 _t_ ( _re −_ _a_ [(1)] )( _re −_ _a_ [(2)] )
_G_ ( _re_ ) = 1 + [(1)] [(2)]


( _[a]_ [(1)] _[−][a]_ [(2)]


2 _[a]_ ) [2]


_[a]_ [(1)][+] 2 _[a]_ [(2)]
= 1 _−_ [8(] _[ta]_ [(1)] _[a]_ [(2)] [+] [(] _t_


[+] 2 _[a]_ ) [2]


2 _t_ _[a]_ ) [2] _−_ [(] _[a]_ [(1)][+] 2 _[a]_ [(2)][)][2]


(23)


2 _t_ _−_ 2 )

( _a_ [(1)] _−_ _a_ [(2)] ) [2]


We will discuss three situations based on the different values of _t_ .


3


2.1 _t >_ 1


_′_
because _r_ 2 _< a_ [(2)], _r_ 4 _> a_ [(1)],so _r_ 2 and _r_ 4 are out of range. _G_ ( _re_ ) is a monotony decrease function
_′_ _′_
of t value.when _t_ approach 1, _G_ ( _re_ ) reach its maximum with _−_ 1. so _G_ ( _re_ ) _<_ _−_ 1. that means
_re_ is repulsion point. the _r_ value is a chaos motion.from Eq(18),if _r_ approach _a_ [(1)], _G_ ( _r_ ) will also
approach _a_ [(1)], so _re_ = _a_ [(1)] is also a fixed point


_′_ _ta_ [(1)] _−_ _a_ [(2)] _t_
_G_ ( _re_ ) = (24)

_ta_ [(1)] _−_ _a_ [(1)] _[≈]_ _t −_ 1 _[>]_ [ 1]


so _re_ = _a_ [(1)] is also repulsion point. From Figure(1a), it can be seen that the function graphs are
similar for different values of t. In Figure(1b), the intersection points of the three functions _G_ ( _r_ ),The
inverse function _G_ ( _r_ ) _[−]_ [1] of _G_ ( _r_ ), and _Y_ = _Y_ ( _x_ ) are the fixed points. It is evident that the gradient
at the fixed point forms an angle less than 90 degrees with Y, indicating that it is a repulsion point.


(a) _G_ ( _r_ ) function ( _t_ = 1 _._ 1 _,_ 1 _._ 5 _,_ 1 _._ 9 _,_ 2 _._ 5 _,_ 3 _._ 5)


(b) _G_ ( _r_ ) is orange line, _G_ ( _r_ ) _[−]_ [1] is green line, _Y_ ( _x_ ) = _x_
(t=1.5) is blue line


Figure 1: _G_ ( _r_ ) _function_ ( _a_ [(1)] = 50 _, a_ [(2)] = 1)


2.2 _t_ = 1


Obviously,when _t_ = 1,it is the most commonly used steepest descent method.


4


the initial point _X_ 0 = [ _x_ [(1)] 0 _[, x]_ 0 [(2)][]][,]


_′_

_−_ 1 _< G_ ( _re_ ) _<_ 0 (31)


,the point _a_ [(1)] is a strange attractor, so the r value will tend to the point of _a_ [(1)] So if _t_ is within a
certain range, the value of _r_ is proportional to _t_ . However, if _t_ exceeds this range, then _t_ approaches
the maximum value of the eigenvector _a_ [(1)] .


3 N DIMENSION


Similarly to the previous chapter, for the N-dimensional case, we also conduct an analysis based on
different values of _t_ .


3.1 _t_ = 1


when the case t value equal to 1. that means SD methods.Akaike (1959) and Forsythe (1968) have
conducted an in-depth analysis, and we have analyzed it from the perspective of _r_ . from Eqs(10)
and (11)


5


2 (2)3 (2)
+ _a_ _x_ 0

2 (2)2 (2)
+ _a_ _x_ 0


_g_ 0 [(1)]


_r_ 0 = _[a]_ [(1)3] _[x]_ 0 [(1)]

_a_ [(1)2] _x_ [(1)] 0


_r_ 0 = _[a]_ [(1)3] _[x]_ 0 [(1)]


2


0
2 [=] _[a]_ [(1)] _[g]_ [(1)]


2 (2) (2)
+ _a_ _g_ 0


2


2 (25)


2 (2)
+ _g_ 0


2
( _r_ 0 _−_ _a_ (2))2


_r_ 1 = _[a]_ [(1)] _[g]_ 0 [(1)]

_g_ 0 [(1)]


2( _r_ 0 _−_ _a_ (1))2 + _a_ (2) _g_ 0(2)


2( _r_ 0 _−_ _a_ (1))2 + _g_ 0(2)


2 (1) (2)
+ _a_ _g_ 0


2


2 (26)


2 _g_ 0(2) ( _r_ 0 _−_ _a_ (2))2 = _[a]_ [(2)] _[g]_ 0 [(1)]
( _r_ 0 _−_ _a_ (2))2 _g_ 0 [(1)]


_g_ 0 [(1)]


2 (2)
+ _g_ 0


2
( _r_ 1 _−_ _a_ (2))2


_r_ 2 = _[a]_ [(1)] _[g]_ 1 [(1)]

_g_ 1 [(1)]


2( _r_ 1 _−_ _a_ (1))2 + _a_ (2) _g_ 1(2)


2( _r_ 1 _−_ _a_ (1))2 + _g_ 1(2)


2 (1) (2)
+ _a_ _g_ 1


2


2 (27)


2 _g_ 1(2) ( _r_ 1 _−_ _a_ (2))2 = _[a]_ [(2)] _[g]_ 1 [(1)]
( _r_ 1 _−_ _a_ (2))2 _g_ 1 [(1)]


_g_ 1 [(1)]


2 (2)
+ _g_ 1


so
_r_ 0 = _r_ 2 _k, r_ 1 = _r_ 2 _k_ +1 (28)


_r_ 0 + _r_ 1 = _rk_ + _rk_ +1 = _a_ [(1)] + _a_ [(2)] (29)


in two dimensions, _r_ will immediately achieve stable state, and then alternate between two values,one large and one small.


_a_ [(1)] + _a_ [(2)] _′_
from the previous chapter,we know _re_ = 2, _G_ ( _r_ ) = _−_ 1.Therefore, _re_ is a critical

state,meaning it is neither attractive nor repulsive. As analyzed earlier, it alternates between the
two states.


2.3 _t <_ 1


_a_ [(1)] + _a_ [(2)]
It may be concluded that _t_ _>_

[(1)]


0 _._ 5 _[a]_ [(2)]

[(1)]


, If the _t_ value is limited in the interval of (0 _._ 5 +
2 _a_ [(1)]


0 _._ 5 _a_ _[a]_ [(1)][(2)] _[,]_ [ 1)][,] _[|][G]_ [(] _[r][e]_ [)] _′|_ _<_ 1,the point _re_ is a strange attractor, so the _r_ value will tend to the point

of _re_ when _r_ approach _a_ [(1)] then


_′_ _ta_ [(1)] _−_ _a_ [(2)] _t_
_G_ ( _re_ ) = (30)

_ta_ [(1)] _−_ _a_ [(1)] _[≈]_ _t −_ 1 _[<][ −]_ [1]


so _re_ = _a_ [(1)] is also repulsion point. when _t_ value has been smaller which means _[a]_ [(1)][+] _[a]_ [(2)]


that means _t <_ 0 _._ 5 + 0 _._ 5 _[a]_ [(2)]

[(1)]


2 [+] _t_ _[a]_ _>_ _a_ [(1)],


_[a]_ _a_ [(1)] [so in the field only includes one equilibrium point which] _[ r]_ _[≈]_ _[a]_ [(1)][,]


for the case where _t_ _>_ 1,according to the analysis above, the _r_ value is no longer stable and still
appear to be chaotic. However, unlike in 2 dimensions where there is only one definite single
stable orbit, in higher dimensions there are several different orbits are actually narrow bands. At the
beginning, the system may be in one state, and with increasing iterations, other orbital states will
emerges until finally it stabilizes.there is a small amount of data outside these main orbits.As shown
in Figure(3), The blue points are generated by the function _G_ ( _r_ ), the orange points are generated by
the function _G_ ( _r_ ) _[−]_ [1], and the green points are generated by the function _Y_ ( _x_ ) = _x_ .


6


_n_

- _gk_ [(] _[i]_ [)]

_j_ =1


2
_A_ ( _a_ ( _i_ ) _, a_ ( _j_ ))


(32)
2
_B_ ( _a_ ( _i_ ) _, a_ ( _j_ ))


_rk_ + _rk_ +1 =


_n_


_i_ =1

_n_


_i_ =1


_n_

- _gk_ [(] _[i]_ [)]

_j_ =1


2 ( _j_ )
_gk_


2 ( _j_ )
_gk_


where
_A_ ( _x, y_ ) = ( _x −_ _y_ ) [2] ( _x_ + _y_ ) (33)


_B_ ( _x, y_ ) = ( _x −_ _y_ ) [2] (34)


(a) _A_ ( _x, y_ ) = ( _x −_ _y_ ) [2] ( _x_ + _y_ ) (b) _B_ ( _x, y_ ) = ( _x −_ _y_ ) [2]


Figure 2: A(x,y) and B(x,y) 0 _._ 1 _≤_ _x ≤_ 20 _,_ 0 _._ 1 _≤_ _y_ _≤_ 20


then we can see _A_ ( _a_ [(] _[i]_ [)] _, a_ [(] _[j]_ [)] ) and _B_ ( _a_ [(] _[i]_ [)] _, a_ [(] _[j]_ [)] ) as the different weight of the numerator and denominator of Eq(32).


so the bigger the difference between the _a_ [(] _[i]_ [)] and _a_ [(] _[j]_ [)], the greater the weight in _a_ [(] _[i]_ [)] and _a_ [(] _[j]_ [)], from
Figure 2, the x and y more center at the top left corner area and the bottom right corner area. the
x and y in other areas lead to critical value of A and B,so only the _a_ [(] _[i]_ [)] and _a_ [(] _[j]_ [)] locate in the
maximum eigenvector direction area apporximate _a_ [(1)] and the minimum eigenvector direciton area
apporximate _a_ [(] _[n]_ [)] have the biggest weight. Based on the analysis above, Eq(32) is mainly affected
by the value at maximum eigenvalue ares and minimum eigenvalue area. after a few step, the system
will fall into a state of balance situation,


_rk_ + _rk_ +1 _≈_ _rk_ +1 + _rk_ +2 _≈_ _a_ [(1)] + _a_ [(] _[n]_ [)] (35)


3.2 _t ̸_ = 1


When _t_ not equal to 1. In a situation similar to two dimensions, the r value will converge to a single
value relatively quickly.


for the case where _t_ _<_ 1,The system quickly reaches a balanced state after a number of iterations,
and the r value will stabilize near a fixed value _re_ and slowly change.we have _re_ = _[a]_ [(1)][+] 2 _t_ _[a]_ [(] _[n]_ [)], _re_ _∈_


( _[a]_ [(1)][+] 2 _[a]_ [(] _[n]_ [)] _, a_ [(1)] )


2 _t_, _re_ _∈_


Figure 3: _G_ ( _r_ )( _a_ [(1)] = 2000 _, a_ [(10000)] = 0 _._ 01 _, t_ = 1 _._ 5)


(a) The r values in the 200 iterations (b) The distribution of r values


Figure 4: r value when _t_ = 0 _._ 9


4 EXPRIMENT


Now, considering an example as follow


, where the sequence _a_ [(] _[i]_ [)] is arithmetic progression and 0 _._ 001 _≤_ _a_ [(] _[i]_ [)] _≤_ 10000, _x_ [(] 0 _[i]_ [)] is a random
number between 0 and 10000. We take the t value of three different situations and iterate 200 times.
for t=0.9, as shown in Figure(4), the value of r stabilizes near a single value.


for t=1, as shown in Figure(5), the value of r quickly stabilizes near two values.


for t=1.1, as shown in Figure(6), the value of r no longer remains stable and may appear at any
position. Since the value of r gradually changes near the stable point, the ratio of values near the
stable point is relatively larger.


7


_f(x)_ =


10000

- _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)2] (36)


_i_ =1


(a) The r values in the 200 iterations (b) The distribution of r values


Figure 5: r value when _t_ = 1


(a) The r values in the 200 iterations (b) The distribution of r values


Figure 6: r value when _t_ = 1 _._ 1


(a) BB method
(b) SD method with _t_ = 1 _._ 5


Figure 7: _G_ ( _r_ ) function


8


we further compared the _G_ ( _r_ ) of the BB method and the SD method(t=1.5). as shown in Figure(7),It
can be observed that the _G_ ( _r_ ) of the SD method has a relatively clear trajectory, and as the number
of iterations increases, the trajectory becomes more definite. On the other hand, the BB method does
not have a trajectory and may fill up all the points in the space.


5 CONCLUSION


We analyzed the SD method by taking the reciprocal of the optimal step size _r_ and introducing a
multiplicative factor _t_ = [1] _s_ [.] [We] [found] [that] [the] [values] [of] _[r]_ [before] [and] [after] [each] [iteration] [follow]

a certain pattern, which we represented using a function _G_ ( _r_ ).Interestingly, this function actually
describes a chaotic system. We calculated the fixed points of this system and found that, depending
on the value of the multiplicative factor _t_, these fixed points correspond to different types of behavior:
one type is stable with a single fixed value, another type is in a critical state with two fixed values,
and the third type is unstable, causing r to jump along the main trajectory. Since the first two states
correspond to fixed _r_ values and descent rates, they do not offer any advantage for the components
in the direction of small eigenvalues or for overall convergence. In contrast, the unstable state allows
_r_ to take on arbitrary values. Therefore, in the future, we can explore the unstable state to potentially
accelerate convergence.


REFERENCES


A.Cauchy. M´ethode g´en´erale pour lar´esolution des syst´ems d’´equations simultan´ees. _C.R._ _Acad._
_Sci. Paris_, 25:536–538, 1847.


H. Akaike. On a successive transformation of probability distribution and its application to the
analysis of the optimum gradient method. _Ann. Inst. Stat. Math. Tokyo_, 11:1–17, 1959.


E. Forsythe, G. On the asymptotic directions of the s-dimensional optimum gradient method. _[j]._
_Numerische Mathematik_, 11:57–76, 1968.


Z Kalousek. Steepest descent method with random steplengths. _[J].Foundations of Computational_
_Mathematics_, 17(2):359–422, 2015.


Svaiter B F. Raydan M. Relaxed steepest descent and cauchy-barzilai-borwein method.

_[J]Computational Optimization and Applications_, 21:155–167, 2002.


R.De Asmundis;D.di Serafino;F.Riccio;G.Toraldo. On spectral properties of steepest descent methods. _IMA Journal of Numerical Analysis_, 4, 2013.


X. Yuan, Y. A new step size for the gradient method. _Journal of Computational Mathematics_, 24
(2):149–156, 2006.


9
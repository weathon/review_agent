# A NEW EFFICIENT METHOD FOR COMBINING GRADI## ENTS OF DIFFERENT ORDERS
# CONFERENCE SUBMISSIONS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


We present a new optimization method called GOC(Gradient Order Combination)
which a combination based on the products of Hessian matrices of different orders and the gradient. the parameter r (the recipprocal of steplenth) is taken as
analysis target, we can regard the SD method as a first-order and the CBB method
as second-order. Whave developed third-order and even higher-order, which offer
faster convergence rates.


1 INTRODUCTION


In this paper, we consider the the unconstrained optimization problem with convex quadratic form


_minf(x)_ = [1] (1)

2 _[x][T][ Ax][ −]_ _[b][T][ x]_


where _x ∈_ R _[n]_, _b ∈_ R _[n]_, _A ∈_ R _[n][×][n]_ is a symmetric and positive definite matrix.


The common solution methods for solving Eq(1) are iterarive methods of the following form


_xk_ +1 = _xk −_ _αk∇f(xk)_ (2)


_f(xk_ +1 _)_ = _f(xk −_ _αk∇f_ ( _xk_ ) _)_ (3)


where _αk_ is a steplenth,gradient descent method and its variants are the most common optimization
method.for GD method,if we minimizes Eq.(3) with exact line search,then we get


_k_ _[∇][f][k]_ _k_ _[g][k]_
_αk_ _[SD]_ = _∇_ _[∇]_ _f_ _[f]_ _k_ _[T][ T][A][∇][f][k]_ = _g_ _[g]_ _k_ _[T][T][Ag][k]_ (4)


this method proposed by A.Cauchy (1847) is called steepest descent method,so _αk_ _[SD]_ is also called
Cauchy step length. the method’s convergence rate is very sensitive to ill condition number and may
be very slow,when the f(x) is quadratic _xk_ will satisfy the


During the iteration process, the SD method exhibits a zigzag phenomena which was explained by
Akaike (1959), J.BARZILAI & J.M.BORWEIN (1988) proposed a nonmonotone steplength which
certain quisi-Newton method, it has two choice for _ak_,respectively:


_k−_ 1 _[s][k][−]_ [1]
_αk_ _[BB]_ [1] = _s_ _[s][T]_ _k_ _[T]_ _−_ 1 _[y][k][−]_ [1] (6)


_k−_ 1 _[y][k][−]_ [1]
_αk_ _[BB]_ [2] = _y_ _[s][T]_ _k_ _[T]_ _−_ 1 _[y][k][−]_ [1] (7)


1


_f_ ( _xk_ +1) _−_ _f_ ( _x_ _[∗]_ )


[1] _[n]_

) [2] (5)
_λ_ 1 + _λn_


( _xk_ +1) _−_ _f_ ( _x_ ) _≤_ ( _[λ]_ [1] _[ −]_ _[λ][n]_

_f_ ( _xk_ ) _−_ _f_ ( _x_ _[∗]_ ) _λ_ 1 + _λn_


where _sk−_ 1 = _xk −_ _xk−_ 1 and _yk−_ 1 = _gk −_ _gk−_ 1,the BB step can be seen Cauchy step with previous iteration. Barzilai and Borwein proved R-superlinear convergence rate in two dimension.Yuan
(2008) for general n dimensional convex quadratic case, the method is convergent too and has a
properties of R-linear rate of convergence. there are some optimization methods based on gradient, YH (2003) decrease the gradient norm, Yuan (2006) and YH (2005) design a alternate steps,
in two dimension case, it could convergence 3 steps. Serafino;F.Riccio;G.Toraldo (2013) propose
SDA with a fixed stepsiz in sussesive steps. and SDC (R.De Asmundisdi SerafinoD (2014)) adding
Cauchy step comparing SDA. Sun C (2020) propose new step size based on Cauchy stepsize. Z
(2015) select random stepsize at some range.Raydan M (2002) introduce RSD which accelerates
convergence by introducing a relaxation parameter between 0 and 2 in the standard Cauchy method,
they also propose CBB method which is a combination of the SD and BB method,the CBB algorithm is much more efficient than BB method.In this paper, we construct a new descent method by
combining the gradient with products of the Hessian matrix of different orders.


2 ANALYSIS OF SD AND CBB METHODS


From Eq(4), we define a parameter _rk_ as follows:


1 _k_ _[Ag][k]_
_rk_ = = _[g][T]_ (8)
2 _αk_ 2 _gk_ _[T]_ _[g][k]_

the initial point is _x_ 0 we set
_x_ _[s]_ 0 [=] _[ x]_ [0] _[−]_ _[α]_ [0] _[g]_ [0] (9)
It is evident that _x_ _[s]_ 0 [is the result obtained after applying the steepest descent method.than we search]

_−−−→_
in the _Ag_ 0 direction and find the point _x_ _[A]_ 1 [,] [the vecotrs] _[−−→]_ _x_ 0 _x_ _[s]_ 0 [and] _x_ 0 _x_ _[A]_ 1 [are perpendicular.] [than we]

_−−−→_
discover the symmetric points _x_ 1 in the direciton _x_ _[A]_ 1 _[x]_ [0][,it] [is] [obvious] _[|][x]_ [1] _[x][s]_ 0 _[|]_ [=] _[|][x][s]_ 0 _[x][A]_ 0 _[|]_ [,as] [shown]
in Fig(1). In order to make the analysis more convenient and intuitive,considering a situation the
objective function is a simple n dimensions hyper-ellipsoid stimulating Eq(1)


2 1

_i_ =1 _[a]_ [(] _[i]_ [)4] _[x]_ 0 [(] _[i]_ [)] ) 2
_l_ 0 _[A]_ [=] _[ l]_ [0] _[/cosθ]_ [0] [=] [(][�] _[n]_ (16)

_r_ 0 [2]


2


_f(x)_ =


_n_

- _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)2] (10)


_i_ =1


       - _n_       - _n_
_i_ =1 _[a]_ [(] _[i]_ [)3] _[x]_ [(] _[i]_ [)2] _i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ [(] _[i]_ [)2]
_r_ =              - _n_ [=]              - _n_ (11)
_i_ =1 _[a]_ [(] _[i]_ [)2] _[x]_ [(] _[i]_ [)2] _i_ =1 _[g]_ [(] _[i]_ [)2]

where 0 _<_ _a_ [(] _[n]_ [)] _≤_ _a_ [(] _[n][−]_ [1)] _≤_ _......_ _≤_ _a_ [(1)], _g_ [(] _[i]_ [)] = 2 _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)], the initial point _x_ 0 =

[ _x_ [(1)] 0 _[, x]_ 0 [(2)] _[, ......x]_ 0 [(] _[n]_ [)][]]


from Eqs.(9) and (10),we have

_x_ _[s]_ 0 [=] _[ x]_ [0] _[−]_ _[∇][f(][x]_ [0] _[)]_ (12)

2 _r_ 0


_x_ _[s]_ 0( _i_ ) = _x_ (0 _i_ ) [(1] _[ −]_ _[a]_ [(] _[i]_ [)] ) (13)

_r_ 0


we define _v_ 0 = _Ag_ 0, _l_ 0 = _∥x_ 0 _x_ _[s]_ 0 _[∥]_ [,] _[ l]_ 0 _[A]_ [=] _[ ∥][x]_ [0] _[x]_ 0 _[A][∥]_ [,] _[ θ]_ [0] [is the angle between] _[ g]_ [0] [and] _[ v]_ [0] [we have]


_g_ 0 _[T]_ _[v]_ [0]      - _ni_ =1 _[a]_ [(] _[i]_ [)2] _[x]_ 0 [(] _[i]_ [)]
_cosθ_ 0 =
_∥g_ 0 _∥∥v_ 0 _∥_ [=] _[ r]_ [0][[]     - _n_

_i_ =1 _[a]_ [(] _[i]_ [)4] _[x]_ 0 [(] _[i]_ [)]


2


1
2 []] 2 (14)


_∥v_ 0 _∥_ - _ni_ =1 _[a]_ [(] _[i]_ [)2] _[x]_ 0 [(] _[i]_ [)]
_∥g_ 0 _∥_ [= 2[] - _n_

_i_ =1 _[a]_ [(] _[i]_ [)4] _[x]_ 0 [(] _[i]_ [)]


2


1
2 []] 2 (15)


Figure 1: SD and CBB


we have


_x_ _[A]_ 0 ( _i_ ) = _x_ 0( _i_ )[1 _−_ ( _[a]_ _r_ [(] 0 _[i]_ [)] ) [2] ] (17)


because _x_ 1 is a symmetric points of _x_ _[A]_ 0 [about] _[ x]_ 0 _[s]_ [, so] _[ x]_ [1] [= 2] _[x][s]_ 0 _[−]_ _[x]_ 0 _[A]_


2

_x_ [(] 1 _[i]_ [)] = _x_ [(] 0 _[i]_ [)][(1] _[ −]_ _[a]_ [(] _[i]_ [)] ) [2] = _x_ [(] 0 _[i]_ [)] _[µ]_ 0 [(] _[i]_ [)] (18)

_r_ 0


It is evident that _x_ 1 is the result obtained after applying the CBB method which is equivalent to
using SD method with the same steplenth in two consecutive iterations. From the above analysis
and Figure(1), we can see that the CBB update direction is symmetric to the _Ag_ direction with the
current gradient as the axis.


3 GOC METHOD


We consider a sequence of m consecutive identical step sizes as the update point,assuming the current point is _xk_,we can obtain the values of _r_ for the three points _xk_, _x_ _[s]_ _k_ [, and] _[ x][k]_ [+1][ as follows:]


 - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ =


2


- _n_
_i_ =1 _[g]_ _k_ [(] _[i]_ [)]


2 (19)


2 ( _i_ )
_µk_


_ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2[ _rk −_ _a_ ( _i_ )]2

- _ni_ =1 _[g]_ _k_ [(] _[i]_ [)] 2[ _rk −_ _a_ ( _i_ )]2


2 (20)

[ _rk −_ _a_ ( _i_ )]2


 - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ _[s]_ [=] 2


( _i_ ) - _n_
_k_ _i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]

2 = 2


2


2 ( _i_ )
_µk_


- _n_
_i_ =1 _[g]_ _k_ [(] _[i]_ [)]


2 ( _i_ )
_µk_


_ni_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)] 2[ _rk −_ _a_ ( _i_ )]4 _m_

- _ni_ =1 _[g]_ _k_ [(] _[i]_ [)] 2[ _rk −_ _a_ ( _i_ )]4 _m_


2 (21)

[ _rk −_ _a_ ( _i_ )]4 _m_


  - _n_
_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
_rk_ +1 =


_i_ ) - _n_

_i_ =1 _[a]_ [(] _[i]_ [)] _[g]_ _k_ [(] _[i]_ [)]
4 _m_ = 2


4 _m_


2 ( _i_ )
_µk_


- _n_
_i_ =1 _[g]_ _k_ [(] _[i]_ [)]


we will analyses several situations for diffrent initial values and the effect of r.


3


1.if the current point _xk_ lies in the larger eigenvalue direction, the gradient value of the lager
eigenvector is biger also, the larger eigenvalue component account for a greater proportion,so _rk_
tend to _a_ [(1)],the value of _a_ [(] _[i]_ [)] direction near the _rk_ will fall sharply, _rk_ + _m_ and _rk_ _[s]_ [will move to] _[ a]_ [(] _[n]_ [)]

direction .in this case, _µ_ [(] _[i]_ [)4] _[m]_ _<_ _µ_ [(] _[i]_ [)2], _rk_ +1 _<_ _rk_ _[s]_ [.from] [Figure(2a),the] [bigger] [m] [value] [is] [and] [the]
relatively smaller the _µ_ value and the faster the decrease rate of different eigenvalue direction.


2.if the current point _xk_ have a huge value at the minimize eigenvector direction compared
to other eigenvector directions, _rk_ tend to _a_ [(] _[n]_ [)],the great majority of _a_ [(] _[i]_ [)] direction value are much
larger than r value. so _µ_ [(] _[i]_ [)] is far larger than 1 especially in the direction of large eigenvale. from
Figure(2b), the bigger m value is and the relatively bigger the _µ_ value, there has a sharp rise in the
bigger eigenvalue direction.


3. considering more general cases, the distribution of the current point _xk_ component are random, the _rk_ is random value between _a_ [(1)] and _a_ [(] _[n]_ [)] correspondingly.those eigenvetor direction value
agree with _rk_ will decrease more quickly. _rk_ +1 and _rk_ _[s]_ [will become larger and smaller according to]
_rk_ .if the _rk_ value is in the middle area of eigenvalues. from Figure(2c), the larger _µ_ value signify
faster decreas rate like Figure(2a).


Based on the analysis above, r value will seesaw between larger eigenvalue area and smaller eigenvalue area generally.the component of small eigenvalue determine the convergence rate and is hard
to reduce .for SD method,r will stabilize in two certain value which means to be relatively fixed
decrase rate. Comparing the SD method,the CBB method’s r value have more wider range change
, and have higher descent rate in the direction of small eigenvalue also. Randan and Svaiter have
proven that the sequence _xk_ generate by CBB method converages Q-linerarly in the norm with
convergence factor 1 _−_ _θ_ = _[λ][max]_ _λmax_ _[−][λ][min]_


(a) r=9000 (b) r=1500 (c) r=5000


Figure 2: _x ∈_ [0 _._ 1 _,_ 10000], _µ_ = 1 _−_ _[x]_ _r_


It can be seen that if we analyze from the perspective of eigenvalue, we will find that the SD method
is first-order, while the CBB method is second-order.and we can develop methods of higher order.


Assuming it is of order m, we have


_x_ 1 = _x_ 0 _−_ 3 _[g]_ [0]


_[g]_ [0] + 3 _[Ag]_ [0]

_r_ 0 _r_ 0 [2]


_x_ [(] 1 _[i]_ [)] = _x_ [(] 0 _[i]_ [)][(1] _[ −]_ _[a]_ [(] _[i]_ [)] ) _[m]_ = _x_ [(] 0 _[i]_ [)]

_r_ 0


_n_

- _Cm_ _[k]_ [(] _[−][µ]_ [(] 0 _[i]_ [)][)] _[k]_ (22)

_i_ =1


where _µ_ 0 [(] _[i]_ [)] = _[a]_ _r_ [(] _[i]_ [)]


0 _r_ 0


We konw that applying the Hessian-free method to a vector _v_ is equivalent to multiply _v_ by _A_,If the
vector is the gradient, then each application of the Hessian-free method is equivalent to multiplying

_k_

each component of the eigenvector by its corresponding eigenvalue. so _µ_ [(] 0 _[i]_ [)] is equivalent to ap
plying the Hessian-free method _k_ times.so by combining different numbers of Hessian-free method
iterations, we can achieve Eq(22). if we take m to be 3. then we have


each component of the eigenvector by its corresponding eigenvalue. so _µ_ [(] 0 _[i]_ [)]


_x_ [(] 1 _[i]_ [)] = _x_ [(] 0 _[i]_ [)][(1] _[ −]_ _[a]_ [(] _[i]_ [)] ) [3] = _x_ [(] 0 _[i]_ [)][(1] _[ −]_ [3] _[µ]_ 0 [(] _[i]_ [)] [+ 3] _[µ]_ 0 [(] _[i]_ [)]

_r_ 0


2 ( _i_ )
_−_ _µ_ 0


3
) (23)


We transform the above equation into another form as follow


[0] _−_ _[A]_ [2] _[g]_ [0]

_r_ 0 [2] _r_ 0 [3]


(24)
_r_ 0 [3]


4


(a) x takes a fixed value 10000


(b) x takes a radom value from 0 to 10000


Figure 3: BB is the blue line,CBB is the orange line, GOC is the green line.


5


From the above equation, it can be seen that if we calculate _Ag_ 0 and _A_ [2] _g_ 0, we can obtain the final
updated value. We first compute the gradient at the current point _g_ 0,and then move along the current
negative gradient with a step size of _d_ to reach the point _x_ [1] 0 [.At] _[ x]_ [1] 0 [,we compute gradient] _[ g]_ 0 [1] [and the]
value of _r_ 0 .By calculating _g_ 0 _−g_ 0 [1][,we obtain] _[ dAg]_ [0][. Next, we move along the direction of the gradient]
_g_ 0 [1] [with a step size of] _[ d]_ [ to reach the point] _[ x]_ 0 [2][,and compute the gradient] _[ g]_ 0 [2][,By calculating] _[g]_ [0] _[−]_ _[g]_ 0 [2][,we]
obtain _d_ [2] _A_ [2] _g_ 0.In this way, we obtain all the values in Eq(24), thereby obtaining the updated value
_x_ 1. From the above, it can be seen that by updating once in the negative gradient direction and once
in the positive gradient direction with a fixed step size _d_, we can calculate the final updated point.


**Algorithm 1** Gradient Order Combination Algorithm


**Require:** _f_ ( _x_ ): objective funtion; _x_ 0: initial solution; _d_ : step size; _ε_ : objective gradient norm value
**Ensure:** optimal _x_ _[∗]_

initial _x_ 0
**while** ( _|f_ ( _xk|_ ) _> ε_ ) **do**


the sequence _a_ [(] _[i]_ [)] is arithmetic progression and 0 _._ 001 _≤_ _a_ [(] _[i]_ [)] _≤_ 10000, _x_ [(] 0 _[i]_ [)] is a fixed value of
10000.the stopping parameter _ϵ_ = 10 _[−]_ [5] .we perform 5000 iterations caculation by menas of three
method(BB CBB GOC).for demonstration on the figure, the norm value is processed with logarithm
in order to limit big changing range of data. The number of times that BB satisfies the stopping
condition is 4930,the CBB method is 3194,The GOC method is 1864.as shown in Figure(3a) we set
_x_ [(] 0 _[i]_ [)] = 10000 _∗_ _rd_, _rd_ is randomly generated in (0,1),The maximum value of _x_ [(] 0 _[i]_ [)] is 9999.9531,and
the minimum value of _x_ 0 [(] _[i]_ [)][is] [0.0423.The] [number] [of] [times] [that] [CBB] [method] [satisfies] [the] [stopping]
condition is 3515,The GOC method is 2163,and the BB method could not satisfy the stop condition.as shown in Figure(3b)


5 CONCLUSION


We conducted an in-depth analysis of the SD method and the CBB method from the perspective
of optimal step size. We found that they can be regarded as methods of different orders within the
same pattern. Based on this, we designed a higher-order method, which demonstrates a faster rate
of descent.


REFERENCES


A.Cauchy. M´ethode g´en´erale pour lar´esolution des syst´ems d’´equations simultan´ees. _C.R._ _Acad._
_Sci. Paris_, 25(2):536–538, 1847.


H. Akaike. On a successive transformation of probability distribution and its application to the
analysis of the optimum gradient method. _Ann. Inst. Stat. Math. Tokyo_, 11:1–17, 1959.


6


compute _xk_ gradient _gk_ ;
compute _x_ [1] _k_ [=] _[ x][k][ −]_ _[dg][k]_ [;]
compute _x_ [1] _k_ [gradient] _[ g]_ _k_ [1] [and] _[ r][k]_ [;]


_k_
compute _Agk_ = _[g][k][−]_ _d_ _[g]_ [1] ;
compute _x_ [2] _k_ [=] _[ x]_ _k_ [1] [+] _[ dg]_ _k_ [1][;]
compute _x_ [2] _k_ [gradient] _[ g]_ _k_ [2][;]


_k_
compute _A_ [2] _gk_ = _[g][k]_ _d_ _[−]_ [2] _[g]_ [2] ;


_[g][k]_ [3] _[A]_ [2] _[g][k]_

_rk_ [+] _rk_ [2]


compute _xk_ +1 = _xk −_ [3] _[g][k]_


[2] _[g][k]_ _−_ _[A]_ [3] _[g][k]_

_rk_ [2] _rk_ [3]


_k_ +1 _k_ _rk_ _rk_ [2] _rk_ [3]

**end while**


4 NUMERICAL EXPERIMENTS


Considering an example as follow


_f(x)_ =


100000

- _a_ [(] _[i]_ [)] _x_ [(] _[i]_ [)2] (25)


_i_ =1


J.BARZILAI and J.M.BORWEIN. Two point step size gradient methods. _IMA J. Numer. Anal._, 8:
141–148, 1988.


Svaiter B F. Raydan M. Relaxed steepest descent and cauchy-barzilai-borwein method.

_[J]Computational Optimization and Applications_, 21:155–167, 2002.


et al. R.De Asmundisdi SerafinoD, Hager W W. An efficient gradient method using the yuan
steplength. _[J].ComputationalOptimizationandApplications_, 59(3):541–563, 2014.


R.De Asmundis;D.di Serafino;F.Riccio;G.Toraldo. On spectral properties of steepest descent methods. _IMA Journal of Numerical Analysis_, 4, 2013.


Liu JP. Sun C. New stepsizes for the gradientmethod. _[J].Optimization_ _Letters_, 14:1943–1955,
2020.


Dai YH. Alternate step gradient method. _Optimization_, 52(4-5):395–415, 2003.


Dai YH. Analysis of monotone gradient methods. _[J]. Journal of Industrial and ManagementOpti-_
_mization_, 2:181–192, 2005.


X. Yuan, Y. A new step size for the gradient method. _Journal of Computational Mathematics_, 24
(2):149–156, 2006.


X. Yuan, Y. step size for the gradient method. _Ams_ _Ip_ _Studies_ _in_ _Advanced_ _Mathematics_, 42(2):
785–796, 2008.


Kalousek Z. Steepest descent method with random steplengths. _[J].Foundations of Computational_
_Mathematics_, 17(2):359–422, 2015.


7
# SPECIAL UNITARY PARAMETERIZED ESTIMATORS OF ROTATION


**Akshay Chandrasekhar**


ABSTRACT


This paper revisits the topic of rotation estimation through the lens of special
unitary matrices. We begin by reformulating Wahba’s problem using _SU_ (2) to
derive multiple solutions that yield linear constraints on corresponding quaternion
parameters. We then explore applications of these constraints by formulating efficient methods for related problems. Finally, from this theoretical foundation, we
propose two novel continuous representations for learning rotations in neural networks. Extensive experiments validate the effectiveness of the proposed methods.


1 INTRODUCTION


3D rotations are fundamental objects ubiquitously encountered in domains such as physics,
aerospace, and robotics. Many representations have been developed over the years to describe them
including rotation matrices, Euler angles, and quaternions. Each method has specific strengths such
as parameter efficiency, singularity avoidance, or interpretability. While special orthogonal matrices _SO_ (3) are widely used, their complex counterparts, special unitary matrices _SU_ (2), are less
explored in areas like robotics and machine learning. This paper showcases the utility of special
unitary matrices by tackling rotation estimation from different perspectives.


1.1 WAHBA’S PROBLEM


Wahba’s problem (Wahba, 1965) is a fundamental problem in attitude estimation. The task refers
to the process of determining the orientation of a target coordinate frame relative to a reference
coordinate frame based on 3D unit vector observations. More formally, it is phrased as seeking the
optimal rotation matrix **R** minimizing the following loss:


min
**R** _∈SO_ (3)


- _wi||_ **b** _i −_ **Ra** _i||_ [2] (1)


_i_


where **a** _i_ are the reference frame observations, **b** _i_ are the corresponding target frame observations,
and _wi_ are the real positive weights for each observation pair. The problem can be solved analytically
by finding the nearest special orthogonal matrix (in a Frobenius sense) to the matrix **B** below:


**B** =           - _wi_ **b** _i_ **a** _[T]_ _i_ (2)


_i_


Today, this solution is typically computed via singular value decomposition (Markley, 1987).


Alternatively, the solution can be estimated as a unit quaternion. Davenport (1968) introduced the
first such method in 1968 by showing that the optimal quaternion **q** is the eigenvector corresponding
to the largest eigenvalue of a 4x4 symmetric gain matrix **K**, which can be constructed as:


  - _Tr_ ( **B** ) **z** _[T]_
**K** =

**z** **B** + **B** _[T]_ _−_ _Tr_ ( **B** ) **I**


1


(3)


where **I** is the identity matrix, _Tr_ ( **B** ) = [�]


_i_ **[B]** _[ii]_ [, and] **[ z]** [=] [�]


where **I** is the identity matrix, _Tr_ ( **B** ) = [�] _i_ **[B]** _[ii]_ [, and] **[ z]** [=] [�] _i_ _[w][i]_ **[a]** _[i]_ _[×]_ **[ b]** _[i]_ [.] [The solution via eigen-]

decomposition is relatively slow as it solves for all the eigenvectors of the matrix which are not
needed. Later solutions improve upon this by calculating the characteristic equation of **K** and solving for only the largest eigenvalue (Shuster and Oh, 1981; Mortari, 1997; Wu et al., 2018). For an
overview of major algorithms, see Lourakis and Terzakis (2018).


1.2 REPRESENTATIONS FOR LEARNING ROTATIONS


In recent years, there has been great interest in representing rotations within neural networks, which
often struggle with learning structured outputs. Directly predicting common parameterizations such
as quaternions or Euler angles has generally performed relatively poorly (Geist et al., 2024). In fact,
it was shown that any 3D rotation parameterization in less than five real dimensions is discontinuous, necessitating non-minimal representations for smooth learning (Zhou et al., 2019). Additionally, challenges like double cover in some representations can further hinder learning. Two leading
approaches, Levinson et al. (2020) and Peretroukhin et al. (2020), essentially interpret network outputs as **B** and **K** matrices (Eqs. (2) and (3) respectively), mapping them to rotations via solutions
to Wahba’s problem. Thus, the two tasks can be linked. For a more in depth overview of the task,
see Geist et al. (2024).


1.3 CONTRIBUTIONS


This paper establishes new theoretical results on rotation estimation by utilizing special unitary
matrices within the framework of Wahba’s problem. We explore several applications of these results,
with particular emphasis on our two novel representations for learning rotations in neural networks.


**We** **highly** **recommend** **the** **reader** **to** **first** **review** **Appendix** **A** **to** **become** **familiar** **with** **the**
**relevant mathematical background and notation used throughout the paper.**


2 SOLUTIONS TO WAHBA’S PROBLEM VIA SU(2)


Transferring Wahba’s Problem to complex projective space, we can solve for the optimal rotation as
a special unitary matrix.


2.1 STEREOGRAPHIC PLANE SOLUTION


First, we establish the proper distance metric in complex projective space corresponding to the
spherical chordal metric in Eq. (1). For points **a** _,_ **b** _∈_ _S_ [2] and their stereographic projections _ψ_ ( **a** ) =
**z** = [ _z_ 1 _, z_ 2] _[T]_ and _ψ_ ( **b** ) = **p** = [ _p_ 1 _, p_ 2] _[T]_, we can show that the metric can be expressed in the
following way (derivation in Appendix B.1.1):

_||_ **a** _−_ **b** _||_ [2] = [4] _[|][z]_ [1] _[p]_ [2] _[ −]_ _[z]_ [2] _[p]_ [1] _[|]_ [2] (4)

_||_ **z** _||_ [2] _||_ **p** _||_ [2]


We now seek to find the rotation **R** parameterized by corresponding special unitary matrix **U** in
complex projective space that minimizes the objective in Eq. (1). Applying our derived metric and
Eqs. (32) and (34), we can construct for each weighted input correspondence **z** _i_ and **p** _i_ :

_[αz]_ [¯] _[i,]_ [2][)] _[p][i,]_ [1] _[ −]_ [(] _[αz][i,]_ [1][ +] _[ βz][i,]_ [2][)] _[p][i,]_ [2] _[|]_ [2]
_wi||_ **b** _i −_ **Ra** _i||_ [2] = [4] _[w][i][|]_ [(] _[−][βz]_ [¯] _[i,]_ [1][ +]

( _|αzi,_ 1 + _βz_ 2 _,i|_ [2] + _| −_ _βz_ [¯] 1 _,_ 1 + _αz_ ¯ _i,_ 2 _|_ [2] ) _||_ **p** _i||_ [2]

_[αz]_ [¯] _[i,]_ [2][)] _[p][i,]_ [1] _[ −]_ [(] _[αz][i,]_ [1][ +] _[ βz][i,]_ [2][)] _[p][i,]_ [2] _[|]_ [2]
= [4] _[w][i][|]_ [(] _[−][βz]_ [¯] _[i,]_ [1][ +]

_||_ **Uz** _i||_ [2] _||_ **p** _i||_ [2]


where _α, β_ are the complex parameters defining **U** from Eq. (31). By definition of unitary matrices,
_||_ **Uz** _||_ [2] = _||_ **z** _||_ [2] . Thus, we can rewrite our expression as the following target constraint:


4 _w|_ ( _−βz_ [¯] _i,_ 1 + _αz_ ¯ _i,_ 2) _pi,_ 1 _−_ ( _αzi,_ 1 + _βzi,_ 2) _pi,_ 2 _|_ [2]


= 0 (5)
_||_ **z** _||_ [2] _||_ **p** _||_ [2]


_[αz]_ [¯] _[i,]_ [2][)] _[p][i,]_ [1] _[ −]_ [(] _[αz][i,]_ [1][ +] _[ βz][i,]_ [2][)] _[p][i,]_ [2][)]
= _⇒_ [2] _[√][w]_ [((] _[−][βz]_ [¯] _[i,]_ [1][ +]

~~�~~ _|zi,_ 1 _|_ [2] + _|zi,_ 2 _|_ [2] ~~[�]~~ _|pi,_ 1 _|_ [2] + _|pi,_ 2 _|_ [2]


= 0 (6)
_|pi,_ 1 _|_ [2] + _|pi,_ 2 _|_ [2]


2


The expression is now just a linear function of rotation parameters. It is a general constraint as it
handles the entire complex projective space (proof in Appendix B.1.2). However, in practice, our
inputs are more commonly given as projection coordinates on the complex plane. As such, we have:

_zi,_ 1 = _zi_ = _xi_ + _yii,_ _pi,_ 1 = _pi_ = _mi_ + _nii,_ _zi,_ 2 = _pi,_ 2 = 1
for each point correspondence ( _xi, yi, mi, ni_ _∈_ R). This simplifies the constraint to the following:
2 _[√]_ _wi_ (( _−βz_ [¯] _i_ + _α_ ¯) _pi_ _−_ _αzi_ _−_ _β_ )
= 0 (7)

       - _|zi|_ [2] + 1� _|pi|_ [2] + 1

We can rearrange the equation to the following linear form with **u** = - _α_ _β_ _α_ ¯ _β_ ¯� _T_ :


4 _wi_
_wi_ _[′]_ [=] (8)
( _|zi|_ [2] + 1)( _|pi|_ [2] + 1)

~~�~~ _wi_ _[′]_ [[] _[−][z][i]_ _−_ 1 _pi_ _−pizi_ ] **u** = ~~�~~ _wi_ _[′]_ **[A]** _[i]_ **[u]** [ = 0] (9)
Each input point pair gives us a complex constraint **A** _i_ . Stacking **A** _i_ together and multiplying the
weights through, we can write the relation succinctly as **Au** = 0 ( **A** is a complex _n_ x 4 matrix for
_n_ points). With noisy observations, the constraints do not hold exactly, so we aim to find the best
rotation that minimizes the least squares error _||_ **Au** _||_ [2] . It is nontrivial to solve for the minimizing
vector **u** while ensuring the result will form a valid special unitary matrix ( **u** 1 = **u** ¯3, **u** 2 = **u** ¯4,
**u** 1 **u** ¯1 + **u** 2 **u** ¯2 = 1). To more effectively solve this, we use Eq. (35) to transform the vector **u** to
a corresponding quaternion **q** = [ _wq_ _xq_ _yq_ _zq_ ] _[T]_ that has a simpler constraint ( **q** must be unit
norm). We carry out the complex multiplication for each **A** _i_ **u** and break the constraint into two
constraints, one for the real and imaginary parts respectively:

4 _wi_
_wi_ _[′]_ [=] (10)
(1 + _x_ [2] _i_ [+] _[ y]_ _i_ [2][)(1 +] _[ m]_ _i_ [2] [+] _[ n]_ _i_ [2][)]


**q** = - _wi_ _[′]_ **[D]** _[i]_ **[q]** [ = 0] (11)


~~�~~ _wi_ _[′]_


- _xi −_ _mi_ _−yi −_ _ni_ 1 + _mixi −_ _niyi_ _miyi_ + _nixi_
_yi −_ _ni_ _xi_ + _mi_ _miyi_ + _nixi_ 1 _−_ _mixi_ + _niyi_


Multiplying the weights through again and stacking together **D** _i_ for each correspondence into **D**
(real 2 _n_ x 4 matrix), we can arrive at the following constrained least squares objective:

_||_ **Dq** _||_ [2] = **q** _[T]_ **D** _[T]_ **Dq** = **q** _[T]_ [ ��] _wi_ _[′]_ **[D]** _i_ _[T]_ **[D]** _[i]_                - **q** = **q** _[T]_ **G** _P_ **q**

_i_

min **q** _[T]_ **G** _P_ **q** _,_ _s.t. ||_ **q** _||_ = 1 (12)
**q**


The formulated objective in Eq. (12) is equivalent to the original problem statement, and the solution
is well known as the eigenvector corresponding to the smallest eigenvalue of **G** _P_ . Using Eq. (35)
again, we can map **q** back to a special unitary matrix **U** giving a solution to the problem. Note that

_−_ **q** is also a solution since eigenvectors are only unique up to scale. However, the sign is irrelevant
as **q** and _−_ **q** map to the same rotation due to the double cover of quaternions over _SO_ (3) in Eq. (36).
For further theoretical details on this solution, see Appendix C.


2.2 APPROXIMATION VIA M ¨OBIUS TRANSFORMATIONS


We can approximate the previous solution in the complex domain by first estimating an optimal
M¨obius transformation **M** and mapping it to a special unitary matrix. Relaxing the special unitary
conditions in Eq. (9), we can treat **u** as a flattened form of **M**, leading to a modified constraint **A** _[′]_ _i_
that holds when **M** aligns a stereographic point pair:

**m** = _vec_ ( **M** ) = [ _σ_ _ξ_ _γ_ _δ_ ] _[T]_

[ _−zi_ _−_ 1 _pizi_ _pi_ ] **m** = **A** _[′]_ _i_ **[m]** [ = 0] (13)
Note that Eq. (13) does not preserve the metric in Eq. (4) between _pi_ and transformed point **ΦM** ( _zi_ ).
We can stack each **A** _[′]_ _i_ [into matrix] **[ A]** _[′]_ [(] _[n]_ [ x 4 complex matrix) and similarly estimate the best (in a]
least squares sense) M¨obius transformation aligning the points as:

**G** _M_ = **A** _[′][H]_ **A** _[′]_ =        - **A** _[′]_ _i_ _[H]_ **[A]** _i_ _[′]_

_i_


min _[s.t.]_ _[||]_ **[m]** _[||]_ [ = 1] (14)
**m** **[m]** _[H]_ **[G]** _[M]_ **[m]**


3


The constraint in Eq. (14) is necessary to prevent trivial solutions, but the choice of quadratic constraint on **m** is arbitrary. With our constraint choice, the optimal **m** is the complex eigenvector
corresponding to the smallest eigenvalue of **G** _M_ . Since **G** _M_ is positive semidefinite and Hermitian
( **G** _[H]_ _M_ [=] **[ G]** _[M]_ [) by construction, the eigenvalues are real and nonnegative, facilitating straightforward]
ordering. If _n_ _<_ 4, **m** can be obtained directly from the kernel of **A** _[′]_ . Either way, the solution is
not unique as eigenvectors and kernel vectors can be scaled arbitrarily, particulary by a phase _e_ _[iθ]_ .
However, by Eq. (42), scaled M¨obius transformations are equivalent, so our result properly defines
the transformation.

Given **m**, we can reshape it into **M** and scale **M** to **M** _[∗]_ = det( **M** ) _[−]_ 2 [1] **M** (allowed since the scale

of **M** is arbitrary) so that det( **M** _[∗]_ ) = 1. It is known that the closest unitary matrix to **M** _[∗]_ in the
Frobenius sense can be computed by **UV** _[H]_ (Keller, 1975), where **U** and **V** _[H]_ are from the singular
value decomposition **M** _[∗]_ = **UΣV** _[H]_ . Since det( **M** _[∗]_ ) = 1, the nearest unitary matrix to **M** _[∗]_ is
special unitary (proof in Appendix B.3.1) and in fact the approximate solution. Note that this matrix
is not necessarily the nearest special unitary matrix to **M** itself. By normalizing the determinant, we
prevent the rotation mapping from being affected by arbitrary phase scalings of **m** .


2.3 3D SPHERE SOLUTION


If our inputs are given as unit observations in 3D, we could project them by _ψ_ and use the earlier
solution. However, through Eqs. (37) and (38), we see that we can act directly on 3D vectors with
special unitary matrices which suggests an alternative formulation. Upon examining the structure of
the matrices that _χ_ maps to, one can show that Eq. (1) can be equivalently expressed as:


_χ_ ( **a** _i_ ) _�→_ **Z** _i,_ _χ_ ( **b** _i_ ) _�→_ **P** _i_

- [2] [1] 


_wi||_ **b** _i −_ **Ra** _i||_ [2] = [1]

2

_i_


2


_wi||_ **P** _i −_ **UZ** _i_ **U** _[H]_ _||_ [2] _F_ (15)

_i_


where _|| · ||F_ denotes the Frobenius norm and **U** is the special unitary matrix that maps to **R** . The
Frobenius norm is unitarily invariant, so we may multiply the inside expression on the right by **U** to
obtain a new target objective and corresponding constraint:


1 - _wi||_ **P** _i_ **U** _−_ **UZ** _i||_ [2] _F_ [= 0] [=] _[⇒]_ - _wi_ (16)
2 2 [(] **[P]** _[i]_ **[U]** _[ −]_ **[UZ]** _[i]_ [) = 0]

_i_


We arrive at a linear constraint again via special unitary matrices. Inspecting the matrix within the
Frobenius norm reveals that the loss contribution from the top row elements is identical to that of the
bottom row elements. Consequently, we only need to compute the loss from a single row, allowing
us to eliminate the factor of [1] 2 [from equation Eq. (16).] [With] **[ a]** _[i]_ [= (] _[x][i][, y][i][, z][i]_ [)][ and] **[ b]** _[i]_ [= (] _[m][i][, n][i][, p][i]_ [)][,]

we can write the following complex constraint:


_√_
_wi_


�( _mi −_ _xi_ ) _i_ _yi −_ _zii_ 0 _−ni −_ _pii_ _−yi −_ _zii_ ( _xi_ + _mi_ ) _i_ _ni_ + _pii_ 0 **u** = _[√]_ _wi_ **C** _i_ **u** = 0 (17)


**C** _i_ has a rank of at most 1 if **a** and **b** have the same magnitude. We reformulate the constraint, once
again breaking the complex terms of **u** into their real components. This yields the following linear
constraint in terms of quaternion parameters:





_√_
_wi_





0 _xi −_ _mi_ _yi −_ _ni_ _zi −_ _pi_
_mi −_ _xi_ 0 _−zi −_ _pi_ _yi_ + _ni_
_ni −_ _yi_ _zi_ + _pi_ 0 _−xi −_ _mi_
_pi −_ _zi_ _−yi −_ _ni_ _xi_ + _mi_ 0




 **q** = _[√]_ _wi_ **Q** _i_ **q** = 0 (18)


Note that **Q** _i_ is a 4x4 skew-symmetric matrix and has at most rank 2 if **a** and **b** have the same
magnitude. As a result, our optimization now becomes:

     - _[T]_     - [2]


_wi_ **Q** _[T]_ _i_ **[Q]** _[i]_ [=] _[ −]_  _i_ _i_


_wi_ **Q** [2] _i_ [=] **[ G]** _[S]_
_i_


min _[s.t.][ ||]_ **[q]** _[||]_ [ = 1] (19)
**q** **[q]** _[T]_ **[ G]** _[S]_ **[q]**


The solution is once again the eigenvector corresponding to the smallest eigenvalue of **G** _S_ .


4


3 OPTIMIZATION METHODS FROM LINEAR QUATERNION CONSTRAINTS


Our previous general solutions are notably distinct from other methods as they allow for the principled construction of linear constraints (Eqs. (11) and (18)) on quaternion parameters. We discuss
some applications and desirable properties of these results.


3.1 RESIDUAL BASED OPTIMIZATION


While Wahba’s problem admits a direct solution, many related rotation estimation tasks require iterative methods. These often involve repeatedly evaluating per-observation losses for a candidate
quaternion. Examples include alternative loss functions like the absolute chordal metric ( _L_ 1 distance) or robust approaches such as iteratively reweighted least squares (IRLS). In these settings, our
linear constraints serve as a drop-in, efficient method for residual computation. The stereographic
formulation in Eq. (11) is especially appealing as it is far more compact (8 elements versus 12 for
Eq. (18)) while avoiding branching in construction, especially in the general case of Appendix C.2.


3.2 CONSTRAINED OPTIMIZATION


When the constraints for an observation pair hold exactly, our formulas yield a convenient analytical
characterization of all rotations that align the pair. A practical use case for this is rotation estimation
with an axis prior (e.g. a gravity vector measurement from an IMU). Traditional methods rely on
sequential rotations or intermediate coordinate frames to simplify the problem (Magner and Zee,
2018; Chandrasekhar, 2024). In contrast, because both Eqs. (11) and (18) reduce to rank 2 in this
setting, we can linearly express two quaternion parameters in terms of the other two and solve
directly and efficiently in a reduced space, eliminating the need for intermediate frames.


3.3 TWO-POINT CASE FOR WAHBA’S PROBLEM


More generally speaking, when the constraints hold exactly for one or more observation pairs (i.e.
noiseless scenarios), we can obtain the solution from the kernel of those constraints in closed-form.
For example, with two noiseless 3D sphere observation pairs, the aligning rotation can be given by:


               - ( **a** 1 + **b** 1) _·_ ( **a** 2 _−_ **b** 2)                **q** ˜ = (20)
( **a** 1 _−_ **b** 1) _×_ ( **a** 2 _−_ **b** 2)


where ˜ **q** denotes the unnormalized form of rotation **q** . Appendix D describes our methods to robustly
and efficiently construct these rotations of exact alignment. These simple kernel formulations are
key to enabling our solutions to the case of Wahba’s problem when _n_ = 2.


**Weighted** Wahba’s problem for the two-point case is well known to have closed-form expressions (Shuster and Oh, 1981; Mortari, 1997; Markley, 2002). We propose an alternate solution
which is given by the weighted average of the two (unnormalized) rotations that each noiselessly
align the cross products of the reference and target sets, along with one of the two corresponding
observation pairs (proof in Appendix B.4.1). Using the average rotation definition from Markley
et al. (2007) (i.e. in Frobenius sense for _SO_ (3)), the solution is:


**n** 1 = **a** 1 _×_ **a** 2 _,_ **n** 2 =


_||_ **a** 1 _×_ **a** 2 _||_ [2]


_||_ **a** 1 _×_ **a** 2 _||_ [2] - ( **a** _i_ + **b** _i_ ) _·_ ( **n** 1 _−_ **n** 2)

**q** ˜ _i_ =
_||_ **b** 1 _×_ **b** 2 _||_ [2] [(] **[b]** [1] _[ ×]_ **[ b]** [2][)] _[,]_ ( **a** _i −_ **b** _i_ ) _×_ ( **n** 1 _−_ **n** 2)


_τ_ = ( _w_ 1 _−_ _w_ 2) _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2] _,_ _ω_ = 2 _w_ 1 _||_ **q** ˜2 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2)

_ν_ = 2 _w_ 2 _||_ **q** ˜1 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2) _,_ _µ_ = _τ_ + ~~�~~ _τ_ [2] + _ων_

_µ_ **q** ˜1 + _ν_ **q** ˜2
**q** = (21)

~~�~~ _||_ **q** ˜1 _||_ [2] _µ_ [2] + _||_ **q** ˜2 _||_ [2] _ν_ [2] + 2(˜ **q** 1 _·_ ˜ **q** 2) _µν_


where **q** ˜1 _·_ **q** ˜2 denotes the usual vector dot product between **q** ˜1 and **q** ˜2. See Appendix B.4.3 for
derivation and additional details.


5


(a) Gram-Schmidt (b) 2-vec (c) QCQP/SVD (d) QuadMobius


Figure 1: (a)-(b) Illustration of difference between Gram-Schmidt and 2-vec in 2D. **bx**, **by** are
predicted axes directions from the model, and **Rx**, **Ry** are the orthogonalized coordinate axes from
each mapping. Gram-Schmidt favors **bx**, aligning **Rx** with it greedily while 2-vec uses **bx** _,_ **by** in
a balanced way. (c)-(d) Conceptual illustration of QCQP, SVD, and QuadMobius maps in context
of Wahba’s problem in 3D. QCQP/SVD can be interpreted as direct projection of target points (red)
to an orthogonal frame. QuadMobius first maps those points to an intermediate representation—a
M¨obius transformation, defined by three points (blue)—before projecting to an _SU_ (2) rotation.


**Unweighted** In the case of _w_ 1 = _w_ 2, the optimal rotation simplifies to the rotation which exactly
aligns **a** 1 + **a** 2 to **b** 1 + **b** 2 and **a** 1 _−_ **a** 2 to **b** 1 _−_ **b** 2 (proof in Appendix B.4.2). This is given by:


          - 1 + **a** 1 _·_ **a** 2
**s** 1 = **a** 1 + **a** 2 _,_ **s** 2 = ( **b** 1 + **b** 2)
1 + **b** 1 _·_ **b** 2

          - 1 _−_ **a** 1 _·_ **a** 2
**d** 1 = **a** 1 _−_ **a** 2 _,_ **d** 2 = ( **b** 1 _−_ **b** 2)
1 _−_ **b** 1 _·_ **b** 2

               - ( **s** 1 + **s** 2) _·_ ( **d** 1 _−_ **d** 2)                **q** ˜ = (22)
( **s** 1 _−_ **s** 2) _×_ ( **d** 1 _−_ **d** 2)


The aligning rotation formulas are given in the form of equation Eq. (20) for simplicity, but in
practice we use the approach described in Appendix D.2 for robustness. In that case, singular cases
only arise when **a** 1 _×_ **a** 2 = 0 or **b** 1 _×_ **b** 2 = 0 where no unique solution exists, and a particular one may
be obtained via the special unitary constraints in equation Eq. (17) (see Appendix B.4.4). Notably,
the two solutions above are optimal in the sense of Wahba’s problem and simplified compared to
existing two-point methods, especially for the unweighted case (see Table 5).


An example use case of these methods is estimating the orientation of a camera given an image
of a rectangle. Under a pinhole camera model, the image of a 3D rectangle adheres to the rules of
perspective geometry. Since the rectangle’s opposite edges are parallel in 3D, their projections in the
image converge at vanishing points that represent the direction of these lines in the camera’s frame.
Because the two sets of parallel edges in the rectangle are orthogonal in 3D, the corresponding
vanishing points should also be orthogonal. However, due to measurement noise, this orthogonality
is often violated. Our two point solutions can recover the best estimate of the camera’s orientation
in these cases.


4 REPRESENTATIONS FOR LEARNING ROTATIONS


Based on previous formulations, we introduce two higher-dimensional representations for learning
rotations. See Appendix B.2 for derivation details and Appendix F for further theoretical support
and explanation of both representations.


**2-vec** The first is based on our formula for the optimal rotation from two unweighted observations
and is denoted 2-vec. Similar to the Gram-Schmidt map in Zhou et al. (2019), 2-vec interprets a 6D
output vector from a model as target 3D _x_ and _y_ axes (denoted **b** _x_, **b** _y_ ). Unlike the Gram-Schmidt
method which greedily orthogonalizes the two vectors by assuming the x-axis prediction is correct,
2-vec maps the two vectors to a rotation optimally in the sense of Wahba’s problem, balancing error
from both axis predictions (Fig. 4). Eq. (22) could be used, but since the reference points are the
_x, y_ coordinate axes, we can instead obtain a rotation matrix in a simpler fashion through the same


6


principle:


**b** _x_ + **b** _[′]_ _y_ **b** _x_ _−_ **b** _[′]_ _y_

**b** _[′]_ _y_ [=] _[||]_ **[b]** _[x][||]_ **[b]** [+] [=] **[b]** _[−]_ [=]

_||_ **b** _y||_ **[b]** _[y][,]_ _||_ **b** _x_ + **b** _[′]_ _y||_ _[,]_ _||_ **b** _x −_ **b** _[′]_ _y||_


~~_√_~~ 1
2 [(] **[b]** [+][ +] **[ b]** _[−]_ [)] _[,]_


**R** = - ~~_√_~~ 1


2 [(] **[b]** [+] _[ −]_ **[b]** _[−]_ [)] _[,]_ **[b]** _[−]_ _[×]_ **[ b]** [+][�] _∈_ _SO_ (3) (23)


This method has a similar singular region and computational complexity as that of Gram-Schmidt.


**QuadMobius** A second parameterization is based on the approximation from Section 2.2 involving M¨obius transformations. Taking inspiration from the approach in Peretroukhin et al. (2020), a
(real) 16D network output Θ = _{θi_ : _i_ = 1 _. . ._ 16 _}_ is arranged into the unique complex elements of
**G** _M_ as below:








**G** _M_ (Θ) =





_θ_ 1 _θ_ 2 + _θ_ 3 _i_ _θ_ 4 + _θ_ 5 _i_ _θ_ 6 + _θ_ 7 _i_
_θ_ 2 _−_ _θ_ 3 _i_ _θ_ 8 _θ_ 9 + _θ_ 10 _i_ _θ_ 11 + _θ_ 12 _i_
_θ_ 4 _−_ _θ_ 5 _i_ _θ_ 9 _−_ _θ_ 10 _i_ _θ_ 13 _θ_ 14 + _θ_ 15 _i_
_θ_ 6 _−_ _θ_ 7 _i_ _θ_ 11 _−_ _θ_ 12 _i_ _θ_ 14 _−_ _θ_ 15 _i_ _θ_ 16


 (24)


**G** _M_ (Θ) is Hermitian with real (and assumed distinct) eigenvalues where we can select the eigenvector **m** corresponding to its smallest eigenvalue. After reshaping **m** to a M¨obius transformation
**M**, we can map to a rotation by the approximation procedure in Section 2.2. The procedure can be
performed via singular value decomposition ( **M** = **UΣV** _[H]_ ) to obtain a special unitary matrix **Q** :


**Q** =


~~�~~
_det_ ( **UV** _[H]_ ) **UV** _[H]_ _∈_ _SU_ (2) (25)


Alternatively, we can algebraically solve for **Q** as follows:


**M** _[∗]_ =


_det_ ( **M** )
_|det_ ( **M** ) _|_ (2 _|det_ ( **M** ) _|_ + _Tr_ ( **M** _[H]_ **M** )) **[M]**

**Q** = **M** _[∗]_ + _adj_ ( **M** _[∗]_ ) _[H]_ _∈_ _SU_ (2) (26)


where _Tr_ ( _·_ ) denotes the trace and _adj_ ( _·_ ) denotes the adjugate. In both cases, **Q** is mapped to a
quaternion via Eqs. (35) and (45), and **M** is assumed to be nonsingular. We denote the SVD method
**QuadMobiusSVD** and the algebraic method **QuadMobiusAlg** . With these maps and our assumptions (observed valid in practice), we define a full mapping from Θ to **q** that has a defined numerical
derivative for backpropagation (see Appendix E for derivative formulas). We remark that this map
is motivated by ideas from Levinson et al. (2020) and Peretroukhin et al. (2020), inheriting many of
their properties (e.g. interpretation as Bingham belief (Kent, 1994), differentiability (Magnus, 1985;
Wan and Zhang, 2019)) while offering a potentially more flexible (higher-dimensional, complex)
learning representation.


5 EXPERIMENTS


5.1 WAHBA’S PROBLEM


Synthetic experiments are performed to validate the proposed methods for Wahba’s problem. For
each trial, a ground truth quaternion rotation **q** _gt_ is randomly sampled from _S_ [3], and _n_ reference
points are randomly sampled from _S_ [2] . The reference points are rotated by **q** _gt_ to obtain target observations. Gaussian noise is added to each component of each target observation, and the target
observations are subsequently re-normalized afterward. Weights are randomly sampled between 0
and 1. Accuracy is measured by the angular distance _θerr_ = _cos_ _[−]_ [1] (2( **q** _est_ _·_ **q** _gt_ ) [2] _−_ 1) in degrees between the estimated rotation **q** _est_ and **q** _gt_, where ( _·, ·_ ) denotes the usual vector dot product.
Numerical results shown in Appendix.


We first test our solutions to Wahba’s problem for both 3D and stereographic inputs (Eqs. (12)
and (19)). The input for the latter is created by projecting the 3D points by _ψ_ . We also test the
approximate solution in Section 2.2. The solutions to all three are obtained by eigendecomposition
using Jacobi’s eigenvalue algorithm. For validation, we compare against several quaternion solvers


7


introduced over the past decades. For the two-point case, we also compare against the closed-form
solutions in Markley (2002) and Shuster and Oh (1981). All solutions were reimplemented and
optimized similarly in C++17 and compiled with the flag `-O3` . We perform one million trials for
each configuration.


Table 4 confirms that our optimal solvers match the results of Davenport’s Q-method in the general
case. In contrast, our M¨obius approximation demonstrates a sensitivity to noise (potentially a benefit
in the learning context of next section). We note that this approximation could likely be improved
with a normalization step common in real homography estimation (Hartley and Zisserman, 2004).


Table 5 similarly confirms that our two-point methods achieve the same optimal results as existing
solvers. By utilizing unnormalized rotations, our weighted algorithm minimizes normalization costs,
streamlining the compute. Most notably, in the unweighted case, our tailored solution only requires
roughly a third of the multiplications of other methods, marking a significant gain in efficiency.


Chair Sofa Toilet
Mean Med. Acc5 Acc10 Mean Med. Acc5 Acc10 Mean Med. Acc5 Acc10


Euler 21.479 10.777 0.129 0.457 22.033 9.462 0.153 0.529 14.495 8.375 0.197 0.604
Quat 23.640 12.664 0.083 0.350 23.426 10.778 0.128 0.452 14.959 9.913 0.128 0.511
GS 13.606 6.320 0.350 0.738 15.015 5.469 0.441 **0.801** 6.586 3.708 0.682 0.915
QCQP 13.131 5.786 0.416 0.773 13.916 5.476 0.436 0.795 6.070 **3.452** **0.730** 0.929
SVD 13.061 5.815 0.412 0.773 14.967 5.812 0.406 0.774 6.135 3.502 0.710 **0.930**
2-vec **12.544** 6.100 0.380 0.751 15.077 6.217 0.364 0.753 6.069 3.483 0.713 0.926
QMAlg 12.604 **5.696** **0.425** **0.783** 14.336 5.657 0.419 0.793 6.079 3.590 0.714 **0.930**
QMSVD 13.157 6.211 0.366 0.748 **13.683** **5.421** **0.443** 0.799 **6.026** 3.601 0.699 0.926


Table 1: _θerr_ mean/median and accuracy (subscript indicates threshold) on 3D shape alignment for
different ModelNet10-SO3 categories (Liao et al., 2019). Bold indicates best, underline indicates
second best.


5.2 LEARNING EXPERIMENTS


We conduct several experiments to evaluate our proposed rotation representations. The primary
loss function is the squared Frobenius norm _||_ **R** _pred_ _−_ **R** _gt||_ [2] _F_ [,] [which] [we] [refer] [to] [as] **[Chordal]**
**L2**, where **R** _pred_ is the predicted rotation and **R** _gt_ is the ground truth. For quaternion outputs,
Chordal L2 is computed same as Peretroukhin et al. (2020). We compare our representations— **2-**
**vec**, QuadMobiusAlg ( **QMAlg** ), and QuadMobiusSVD ( **QMSVD** )—against several baselines: **Eu-**
**ler** angles (Tait-Bryan YXZ), **Quat** (quaternion), **GS** (Gram-Schmidt) (Zhou et al., 2019), **QCQP**
(Peretroukhin et al., 2020), and **SVD** (Levinson et al., 2020). In both QuadMobius variants, we use
the algebraic method in the forward pass to avoid SVD computation and isolate differences to the
backward pass. This section presents results on three public benchmarks. Additional synthetic experiments exploring different learning conditions are included in Appendix G.2.2, and full training
details are provided in Appendix G.1.


**ModelNet10-SO3** We first evaluate the representations on the 3D shape alignment task from Liao
et al. (2019) using the ModelNet10-SO3 dataset. This dataset comprises of images of 3D CAD
models under uniformly sampled rotations with multiple object models per category. The task is to
predict the object’s orientation directly from its image. Table 1 reports the results on three object
categories, chosen for their low rotational symmetry following the choice in Levinson et al. (2020).


**Inverse Kinematics** Next, we test the representations on an implicit learning task, applying them
to the inverse kinematics task from Zhou et al. (2019). Given 3D human pose joint locations (from
real-world motion capture data), a network predicts the joint orientations relative to a reference pose
and uses a fixed forward kinematics function to obtain predicted joint locations. The distance loss
is applied between the predicted and given joint locations. In this task, the rotations are used as
implicit representations through which the gradients must flow rather than direct prediction targets.
Fig. 2 compares the results of the different learning representations on this task.


8


Mean 25 _[th]_ 50 _[th]_ 75 _[th]_


Euler 2.653 1.447 2.062 3.146
Quat 2.945 1.529 2.356 3.543
GS 1.629 0.8767 1.256 2.015
QCQP 1.511 0.7729 1.188 1.85
SVD 1.55 0.7647 1.16 1.855
2-vec 1.574 0.809 1.174 1.854
QMAlg 1.51 0.8633 1.182 **1.757**
QMSVD **1.509** **0.7421** **1.13** 1.81


Figure 2: Results of implicit learning for Inverse Kinematics task (Zhou et al., 2019). Left: Mean
and percentile L2 distance error (cm) of predicted joint locations. Bold indicates best, underline
indicates second best. Right: Ratios of joint errors relative to QMSVD across error percentiles
(Euler/Quat omitted due to large ratios).


**Camera** **Pose** **Estimation** Finally, we replicate the experiment from Walch et al. (2017) which
utilizes an LSTM to directly regress a camera’s pose from real world images. Training requires
simultaneously optimizing over both the camera’s orientation and translation. Data comes from
the Cambridge Landmarks dataset (Kendall et al., 2015) which includes labels estimated from traditional structure from motion pipelines. The results are seen in Table 2 from training on select scenes,
following the choice of Chen et al. (2022).


**Results** Overall, the proposed representations demonstrated strong performance and versatility
across the three benchmark tasks. Despite its lower dimensionality, 2-vec proved competitive, occasionally achieving the best result. Notably, it typically outperforms Gram-Schmidt, positioning
itself as an attractive alternative. The QuadMobius approaches showed their potential by achieving
the top result in nearly all experiments over favorites like SVD and QCQP.


King’s College Shop Facade Old Hospital
Mean 25 _[th]_ 50 _[th]_ 75 _[th]_ Mean 25 _[th]_ 50 _[th]_ 75 _[th]_ Mean 25 _[th]_ 50 _[th]_ 75 _[th]_


Euler 4.192 2.403 3.684 5.509 6.826 4.129 6.050 9.305 4.748 2.204 3.247 6.162
Quat 2.759 1.367 2.251 3.499 6.604 **3.762** 5.339 8.153 4.570 2.486 3.377 5.546
GS 3.298 1.764 2.583 4.137 6.559 4.376 5.660 8.343 4.295 **1.897** 3.070 5.698
QCQP 3.204 1.540 2.537 4.129 6.802 3.901 5.797 8.539 4.454 2.156 3.304 6.267
SVD 3.292 1.589 2.624 4.110 7.117 4.157 5.647 8.370 4.574 2.420 3.485 5.961
2-vec 3.085 1.536 2.371 4.014 7.118 3.789 5.762 8.957 **4.294** 2.085 **2.950** **5.292**
QMAlg **2.631** **1.337** **2.052** **3.267** **6.317** 4.050 **5.268** **7.758** 4.426 2.035 3.238 5.640
QMSVD 2.706 1.391 2.177 3.345 6.715 4.074 5.710 8.947 4.409 2.077 3.146 5.744


Table 2: Mean and percentile _θerr_ of predicted rotations from direct pose prediction on different
scenes in Cambridge Landmarks Dataset (Kendall et al., 2015). Bold indicates best, underline indicates second best.


6 CONCLUSION


This paper demonstrated the utility of special unitary matrices for rotation estimation. Several new
formulas and algorithms were presented from this perspective for the real and complex domains,
tackling Wahba’s problem and rotation representations in neural networks. Various experiments confirmed the potential of these approaches. Future work may include further solidifying the theoretical
and empirical foundations of our rotation representations and applying special unitary matrices to
other tasks such as analytical camera pose estimation.


9


REFERENCES


Jose Agustin Barrachina, Chengfang Ren, Gilles Vieillard, Christele Morisseau, and Jean-Philippe
Ovarlez. Theory and implementation of complex-valued neural networks, 2023.


Akshay Chandrasekhar. PoseGravity: Pose estimation from points and lines with axis prior. 2024.


Jiayi Chen, Yingda Yin, Tolga Birdal, Baoquan Chen, Leonidas J Guibas, and He Wang. Projective
manifold gradient layer for deep rotation regression. In _Proceedings of the IEEE/CVF Conference_
_on Computer Vision and Pattern Recognition_, pages 6646–6655, 2022.


Daniel Choukroun. Novel results on quaternion modeling and estimation from vector observations.
In _AIAA Guidance, Navigation, and Control Conference_, 2009.


Paul B. Davenport. A vector approach to the algebra of rotations with applications. 1968.


Andreas Geist, Jonas Frey, Mikel Zhobro, Anna Levina, and Georg Martius. Learning with 3d
rotations, a hitchhiker’s guide to so(3), 2024.


R. I. Hartley and A. Zisserman. _Multiple View Geometry in Computer Vision_ . Second edition, 2004.


Joseph B. Keller. Closest unitary, orthogonal and hermitian operators to a given operator. _Mathe-_
_matics Magazine_, 48(4):192–197, 1975.


Alex Kendall, Matthew Grimes, and Roberto Cipolla. Research data supporting “posenet: A convolutional network for real-time 6-dof camera relocalization”, 2015. Dataset, King’s College,
University of Cambridge.


John T. Kent. The complex bingham distribution and shape analysis. _Journal of the Royal Statistical_
_Society. Series B (Methodological)_, 56(2):285–299, 1994.


Jake Levinson, Carlos Esteves, Kefan Chen, Noah Snavely, Angjoo Kanazawa, Afshin Rostamizadeh, and Ameesh Makadia. An analysis of svd for deep rotation estimation. 2020.


Shuai Liao, Efstratios Gavves, and Cees G. M. Snoek. Spherical regression: Learning viewpoints,
surface normals and 3d rotations on n-spheres. In _CVPR_, pages 9751–9759, 2019.


Xinyuan Liao. Complexnn: Complex neural network modules, 2023.


Manolis Lourakis and George Terzakis. Efficient absolute orientation revisited. 2018.


Ningning Ma, Xiangyu Zhang, Hai-Tao Zheng, and Jian Sun. Shufflenet v2: Practical guidelines for
efficient cnn architecture design. In _ECCV_, 2018.


Robert D. Magner and Robert E. Zee. Extending target tracking capabilities through trajectory and
momentum setpoint optimization. _32nd Annual AIAA/USU Conference on Small Satellites_, 2018


Jan R. Magnus. On differentiating eigenvalues and eigenvectors. _Econometric Theory_, 1:179 – 191,
1985.


F. Landis Markley. Fast quaternion attitude estimation from two vector measurements. _Journal of_
_Guidance, Control, and Dynamics_, 25(2):411–414, 2002.


F. Landis Markley, Yang Cheng, John L. Crassidis, and Yaakov Oshman. Averaging quaternions.
_Journal of Guidance, Control, and Dynamics_, 30(4):1193–1197, 2007.


Landis Markley. Attitude determination using vector observations and the singular value decomposition. _J. Astronaut. Sci._, 38, 1987.


Landis Markley. Attitude determination using two vector measurements. 1999.


D. Mortari. Esoq-2 single-point algorithm for fast optimal spacecraft attitude determination. 95,
1997.


D. Mortari, Landis Markley, and Puneet Singla. Optimal linear attitude estimator. _Journal of Guid-_
_ance, Control, and Dynamics - J GUID CONTROL DYNAM_, 30:1619–1627, 2007.


10


Caitong Peng and Daniel Choukroun. Singularity and error analysis of a simple quaternion estimator,
2024.


Valentin Peretroukhin, Matthew Giamou, W. Greene, David Rosen, Jonathan Kelly, and Nicholas
Roy. A smooth representation of belief over so(3) for deep rotation learning with uncertainty.
2020.


M. D. Shuster and S. D. Oh. Three-axis attitude determination from vector observations. _Journal of_
_Guidance and Control_, 4(1):70–77, 1981.


Grace Wahba. A least squares estimate of satellite attitude. _SIAM Review_, 7(3):409–409, 1965.


Florian Walch, Caner Hazirbas, Laura Leal-Taix´e, Torsten Sattler, Sebastian Hilsenbeck, and Daniel
Cremers. Image-based localization using lstms for structured feature correlation. In _ICCV_, 2017.


Zhou-Quan Wan and Shi-Xin Zhang. Automatic differentiation for complex valued svd. 2019.


Jin Wu, Zebo Zhou, Bin Gao, Rui Li, Yuhua Cheng, and Hassen Fourati. Fast linear quaternion
attitude estimator using vector observations. _IEEE Transactions on Automation Science and En-_
_gineering_, 15(1):307–319, 2018.


Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, and Hao Li. On the continuity of rotation
representations in neural networks. In _2019_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_
_Pattern Recognition (CVPR)_, pages 5738–5746, 2019.


11


# **Special Unitary Parameterized Estimators of** **Rotation**

## **Appendix**


A MATHEMATICAL BACKGROUND AND DEFINITIONS


The mathematical background for special unitary matrices and related concepts is briefly reviewed.
The formulas are all established and generally known. A complex square matrix **U** is defined as
unitary if:


**UU** _[H]_ = **U** _[H]_ **U** = **I** _,_ _|det_ ( **U** ) _|_ = 1 (27)


where _[H]_ denotes the conjugate transpose, _| · |_ denotes complex magnitude, and _det_ ( _·_ ) denotes determinant. The matrix is _special unitary_ if it has the additional restriction that _det_ ( **U** ) = 1 exactly.

Stereographic projection _ψ_ is an invertible mapping of the sphere _S_ [2] = _{_ ( _xs, ys, zs_ ) _| x_ [2] _s_ [+] _[y]_ _s_ [2][+] _[z]_ _s_ [2] [=]
1 _}_ from the point **p** _[∗]_ = (0 _,_ 0 _, −_ 1) to the complex plane and is given by:


_xs_ _ys_
_ψ_ C( **a** ) : + _i_ = _xp_ + _ypi_ = _z_ (28)
1 + _zs_ 1 + _zs_


    - 2 _xp_ 2 _yp_ _p_ _[−]_ _[y]_ _p_ [2]
_ψ_ C _[−]_ [1][(] _[z]_ [) :] _,_ _,_ [1] _[ −]_ _[x]_ [2]
1 + _x_ [2] _p_ + _yp_ [2] 1 + _x_ [2] _p_ + _yp_ [2] 1 + _x_ [2] _p_ + _yp_ [2]


(29)


where **a** _∈_ _S_ [2] and _z_ _∈_ C. This projection is visualized in Fig. 3. Note that _ψ_ C is undefined when
**a** = **p** _[∗]_ . To overcome this, the map is extended to the complex projective space CP [1] which includes
the point at infinity so we can define _ψ_ CP( **p** _[∗]_ ) = _∞_ . The projection is now redefined below with
equivalence relations:


- - _z_
_∼_ _λ_
1


_,_ **a** _̸_ = **p** _[∗]_


_ψ_ CP( **a** ) _�→_


� _z_
 1


   - _λ_

 _∞∼_ 0


- (30)
_,_ **a** = **p** _[∗]_


_λ ∈_ C _,_ _λ ̸_ = 0 _,_ _ψ_ CP _[−]_ [1][(] _[ψ]_ [CP][(] **[a]** [)) =] **[ a]**


In this paper, our use of _ψ_ generally refers to _ψ_ CP. From the above definition, _ψ_ ( **a** ) can be arbitrarily
scaled, and _ψ_ bijectively maps the entire sphere to the complex projective space. Note that this
mapping is not unique, particularly since choice of **p** _[∗]_ is arbitrary (any point on _S_ [2] is valid). We
will use the specific projection defined above for this paper as it is convenient for image processing.


A special unitary matrix **U** _∈_ _SU_ (2) can generally be written as:


  - _α_ _β_   **U** = (31)
_−β_ [¯] _α_ ¯


_αα_ ¯ + _ββ_ [¯] = 1 _,_ _α, β_ _∈_ C


where the bar denotes complex conjugation. **U** transforms a complex projective point **z** = [ _z_ 1 _, z_ 2] _[T]_
and complex plane point _z_ by:


         - _α_ _β_
**U** : **z** _�→_ **z** _[′]_ = **Uz** =
_−β_ [¯] _α_ ¯


�� _z_ 1
_z_ 2


(32)


**ΦU** : _z_ _�→_ _z_ _[′]_ = _[αz]_ [ +] _[ β]_ _[−][βz]_ [¯] [ +] _[α]_ [¯] _[ ̸]_ [= 0] (33)

_−βz_ [¯] + _α_ ¯ _[,]_


These transformations are of importance as they act analogously to rotations of the unit sphere in
R [3] . Specifically, for a 3x3 rotation matrix **R** _∈_ _SO_ (3) that rotates a unit vector **v** _∈_ _S_ [2] as **v** _[′]_ = **Rv**,
there exists some **U** such that:


**v** _[′]_ = ( _ψ_ _[−]_ [1] _◦_ **U** _◦_ _ψ_ )( **v** ) (34)


12


The exact relationship between _SU_ (2) and _SO_ (3) is made clearer by their relationships with unit
quaternions **q** _∈_ H which also act as rotations in R [3] . The isomorphism between _SU_ (2) and unit
quaternions is given as:


**q** = _wq_ + _xqi_ + _yqj_ + _zqk,_ _wq_ [2] [+] _[ x]_ _q_ [2] [+] _[ y]_ _q_ [2] [+] _[ z]_ _q_ [2] [= 1] _[,]_ _[w][q][, x][q][, y][q][, z][q]_ _[∈]_ [R]

_α_ = _wq_ + _xqi,_ _β_ = _yq_ + _zqi_ (35)


and the mapping of unit quaternions to special orthogonal matrices is given by:


**Rq** =


 1 _−_ 2 _yq_ [2] _[−]_ [2] _[z]_ _q_ [2] 2 _xqyq −_ 2 _wqzq_ 2 _xqzq_ + 2 _wqyq_

2 _xqyq_ + 2 _wqzq_ 1 _−_ 2 _x_ [2] _q_ _[−]_ [2] _[z]_ _q_ [2] 2 _yqzq −_ 2 _wqxq_
2 _xqzq −_ 2 _wqyq_ 2 _yqzq_ + 2 _wqxq_ 1 _−_ 2 _x_ [2] _q_ _[−]_ [2] _[y]_ _q_ [2]





 (36)


Eq. (36) is the well-known 2-to-1 surjective mapping between quaternions and rotation matrices. By
their isomorphism in Eq. (35), _SU_ (2) also has a similar surjective mapping with _SO_ (3), linking the
three rotation representations. Note that the mapping given by Eq. (35) is not unique. Furthermore,
special unitary matrices have the ability to act as rotations in R [3] directly by first mapping points to
2x2 complex matrices. For a point **x** = ( _x, y, z_ ) _∈_ R [3] :


                  - _xi_ _y_ + _zi_                   _χ_ : **x** _�→_ **X** = (37)
_−y_ + _zi_ _−xi_


_χ_ ( **x** 1) _�→_ **X** 1 _,_ _χ_ ( **x** 2) _�→_ **X** 2 _,_ **x** 1 _,_ **x** 2 _∈_ R [3]

**X** 2 = **UX** 1 **U** _[H]_ _,_ **U** _∈_ _SU_ (2) (38)


Note if _||_ **x** _||_ = 1, _χ_ ( **x** ) _∈_ _SU_ (2). Also note that the map _χ_ is not uniquely defined either.


Relatedly, M¨obius transformations are general 2x2 complex projective matrices, characterized similarly by:


   - _σ_ _ξ_
**M** =
_γ_ _δ_


(39)


_det_ ( **M** ) _̸_ = 0 _,_ _σ, ξ, γ, δ_ _∈_ C


         - _σ_ _ξ_
**M** : **z** _�→_ **z** _[′]_ = **Mz** =
_γ_ _δ_


�� _z_ 1
_z_ 2


(40)


**ΦM** : _z_ _�→_ _z_ _[′]_ = _[σz]_ [ +] _[ ξ]_ _[γz]_ [ +] _[ δ]_ [= 0] (41)

_γz_ + _δ_ _[,]_

**M** _∼_ _λ_ **M** _,_ _λ ∈_ C _,_ _λ ̸_ = 0 (42)


M¨obius transformations conformally map the complex projective plane onto itself. They are
uniquely determined (up to scale) by their action on three independent points, and _SU_ (2) elements
constitute a subset of them.


B PROOFS AND DERIVATIONS


B.1 PROPER METRIC IN COMPLEX PROJECTIVE SPACE


B.1.1 DERIVATION OF METRIC


Complex projective rays are equivalent if they are linearly dependent. We can test this condition
by setting up the following constraint on complex vectors **z** = [ _z_ 1 _, z_ 2] _[T]_ and **p** = [ _p_ 1 _, p_ 2] _[T]_ for
_z_ 1 _, z_ 2 _, p_ 1 _, p_ 2 _∈_ C:


_det_ - [�] _z_ 1 _p_ 1
_z_ 2 _p_ 2


��
= _z_ 1 _p_ 2 _−_ _z_ 2 _p_ 1 = 0


For vectors **a** = ( _xs, ys, zs_ ) _,_ **b** = ( _ms, ns, ps_ ) _∈_ _S_ [2] (assume **a** = **p** _[∗]_ _,_ **b** = **p** _[∗]_ ) whose projections
via _ψ_ (Eq. (30)) correspond to **z** and **p** respectively, we can show that testing the linear independence


13


Figure 3: Visualization of a stereographic projection from the sphere ( _S_ [2] ) to the complex plane. The
projection is performed by taking the line between **p** _[∗]_ and each point and intersecting that line with
the plane through the equator. The point **p** _[∗]_ itself is mathematically mapped to infinity.


of complex vectors is in fact related to the chordal distance on a sphere:


- _m_ 1 + _s_ + _p nssi_ - _,_ _λ_ 1 _, λ_ 2 _∈_ C _,_ _λ_ 1 = 0 _, λ_ 2 = 0


**z** = _λ_ 1


- _xs_ + _ysi_ 1 + _zs_ _,_ **p** = _λ_ 2


_det_ - [�] _z_ 1 _z_ 2
��� _p_ 1 _p_ 2


2
= _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] _|_ (1 + _ps_ )( _xs_ + _ysi_ ) _−_ (1 + _zs_ )( _ms_ + _nsi_ ) _|_ [2]
�����


= _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] ((1 + _ps_ ) [2] ( _x_ [2] _s_ [+] _[ y]_ _s_ [2][) + (1 +] _[ z][s]_ [)][2][(] _[m]_ _s_ [2] [+] _[ n]_ _s_ [2][)] _[ −]_ [2(1 +] _[ p][s]_ [)(1 +] _[ z][s]_ [)(] _[x][s][m][s]_ [+] _[ y][s][n][s]_ [))]

= _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] (1 + _ps_ )(1 + _zs_ )((1 + _ps_ )(1 _−_ _zs_ ) + (1 + _zs_ )(1 _−_ _ps_ ) _−_ 2( _xsms_ + _ysns_ ))

= _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] (1 + _ps_ )(1 + _zs_ )(2 _−_ 2( _xsms_ + _ysns_ + _zsps_ ))

= _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] (1 + _ps_ )(1 + _zs_ ) _||_ **a** _−_ **b** _||_ [2]


Notice that _|λ_ 1 _|_ [2] (1 + _zs_ ) = _[|][z]_ [1] _[|]_ [2][+] _[|][z]_ [2] _[|]_ [2]


[+] 2 _[|][z]_ [2] _[|]_ [2] and _|λ_ 2 _|_ [2] (1 + _ps_ ) = _[|][p]_ [1] _[|]_ [2][+] 2 _[|][p]_ [2] _[|]_ [2]


Notice that _|λ_ 1 _|_ (1 + _zs_ ) = [1] 2 [2] and _|λ_ 2 _|_ (1 + _ps_ ) = [1] 2 [2] . Substituting this into our

expression and rearranging, we arrive at the final expression for the equivalent distance metric in
complex projective space as:

4 _|z_ 1 _p_ 2 _−_ _z_ 2 _p_ 1 _|_ [2]
_||_ **a** _−_ **b** _||_ [2] =

( _|z_ 1 _|_ [2] + _|z_ 2 _|_ [2] )( _|p_ 1 _|_ [2] + _|p_ 2 _|_ [2] )
The last substitution may seem unnecessary at first; however, this form is more useful as it generalizes the metric to hold even when **a** = **p** _[∗]_ or **b** = **p** _[∗]_ (proof below). It also gives an intuitive
interpretation that the spherical chordal distance is related to a type of “cross product” magnitude
between the two projective rays’ unit directions.


4 _|z_ 1 _p_ 2 _−_ _z_ 2 _p_ 1 _|_ [2]
_||_ **a** _−_ **b** _||_ [2] =


B.1.2 PROOF OF METRIC FOR POINTS AT INFINITY


**Proposition 1** _If_ **a** = **p** _[∗]_ _or_ **b** = **p** _[∗]_ _in Eq._ (4) _, the proper metric is still valid._


_Proof_ The squared distance between unit length points **a** = ( _xs, ys, zs_ ) and **b** = **p** _[∗]_ = (0 _,_ 0 _, −_ 1) is:

_||_ **a** _−_ **b** _||_ [2] = 2 _−_ 2 **a** _[T]_ **b** = 2(1 + _zs_ )

Using vectors **z** = _ψ_ ( **a** ) = _λ_ 1[ _xs_ + _ysi,_ 1 + _zs_ ] _[T]_ _,_ **p** = _ψ_ ( **p** _[∗]_ ) = [ _λ_ 2 _,_ 0] _[T]_ with nonzero _λ_ 1 _, λ_ 2 _∈_ C
and **a** _̸_ = **p** _[∗]_, we can calculate the same quantity via the formula in Eq. (4):
4 _|z_ 1 _p_ 2 _−_ _p_ 1 _z_ 2 _|_ [2] [4] _[| −]_ _[λ]_ [1] _[λ]_ [2][(1 +] _[ z][s]_ [)] _[|]_ [2]


1 _p_ 2 _−_ _p_ 1 _z_ 2 _|_ [2]

= [4] _[| −]_ _[λ]_ [1] _[λ]_ [2][(1 +] _[ z][s]_ [)] _[|]_ [2]
_||_ **z** _||_ [2] _||_ **p** _||_ [2] 2 _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] (1 + _zs_ )


2 _|λ_ 1 _|_ [2] _|λ_ 2 _|_ [2] (1 + _zs_ ) [= 2(1 +] _[ z][s]_ [)]


thus showing that the two formulas yield the same quantity. It is easy to see that Eq. (4) is symmetric,
so the same result would hold if **a** = **p** _[∗]_ and **b** = **p** _[∗]_ . If **a** = **b** = **p** _[∗]_, we can see that _||_ **a** _−_ **b** _||_ [2] is
clearly 0. At the same time, the numerator of Eq. (4) would be 0 while the denominator is nonzero as
the projective scalars _λi_ = 0 for any valid complex projective point. Thus, both quantities are equal
in that case as well, so the formula gives the spherical chordal distance between any two points on
the sphere via their stereographic projections.


14


B.2 REPRESENTATION DERIVATIONS


B.2.1 DERIVATION OF 2-VEC


For 3D vectors **b** _x,_ **b** _y_ extracted from a model output representing predicted target _x_ and _y_ axes
respectively, we apply the method from Section 3.3 in the unweighted case to arrive at an optimal
rotation matrix (in the sense of Wahba’s problem). We assume **b** _x ×_ **b** _y_ = 0. First, **b** _x_ and **b** _y_ must

have the same norm for the method to be unweighted, so we transform **b** _y_ via **b** _[′]_ _y_ [=] - _||||_ **bb** _xy||||_ [2][2] **[b]** _[y]_ [.]
Since the reference points are constant ( **a** 1 = (1 _,_ 0 _,_ 0) _,_ **a** 2 = (0 _,_ 1 _,_ 0)), we know that their normalized sum and difference vectors are **a** [+] = ~~_√_~~ 1 ~~_√_~~ 1 [Similarly,] [we] [create]

2 [(1] _[,]_ [ 1] _[,]_ [ 0)] _[,]_ **[ a]** _[−]_ [=] 2 [(1] _[,][ −]_ [1] _[,]_ [ 0)][.]

normalized sum and difference vectors for the target points as **b** [+] = _||_ **bb** _xx_ ++ **bb** _[′]_ _y_ _[′]_ _y||_ [and] **[ b]** _[−]_ [=] _||_ **bb** _xx−−_ **bb** _[′]_ _y_ _[′]_ _y||_ [.]

The optimal rotation aligns **a** [+] to **b** [+] and **a** _[−]_ to **b** _[−]_ noiselessly. This can be achieved because all
the vectors have the same magnitude (normalizing to unit norm was found to be more stable than
matching magnitudes like **b** _[′]_ _y_ [)] [and] [because] [the] [sum] [and] [difference] [vectors] [are] [always] [orthogonal.]
Since rotation matrices naturally encode how an orthogonal coordinate frame transforms in their
columns, we can construct the aligning rotation by joining the two rotations **Ra** and **Rb** which rotate the coordinate frame to the reference sum/difference vectors and target sum/difference vectors
respectively:


~~_√_~~ 1
2 [(1] _[,]_ [ 1] _[,]_ [ 0)] _[,]_ **[ a]** _[−]_ [=]


[Similarly,] [we] [create]
2 [(1] _[,][ −]_ [1] _[,]_ [ 0)][.]


**Ra** = - **a** + _,_ **a** _[−]_ _,_ **a** [+] _×_ **a** _[−]_ [�] _,_ **Rb** = - **b** + _,_ **b** _[−]_ _,_ **b** [+] _×_ **b** _[−]_ [�]


~~_√_~~ 1
2 [(] **[b]** [+][ +] **[ b]** _[−]_ [)] _[,]_


**R** = **RbR** _[T]_ **a** [=] - ~~_√_~~ 1


**b** _[−]_ _×_ **b** [+][�]
2 [(] **[b]** [+] _[ −]_ **[b]** _[−]_ [)] _[,]_


Because the sum/difference vectors are orthogonal and have unit norm, **Ra** _,_ **Rb** _,_ **R** _∈_ _SO_ (3). Given
the natural representation of coordinate transformations in rotation matrices, using the rotation matrix formulation was more appealing for the map than the quaternion formulation in Eq. (22). It
also provided a more direct comparison with the Gram-Schmidt map. Nonetheless, the core insight was derived from the original linear constraints on quaternion parameters. The unweighted
method was chosen for its geometric and computational simplicity, but a weighted version of the
map incorporating the magnitudes of **b** _x,_ **b** _y_ can be similarly formulated from Eq. (21).


B.2.2 DERIVATION OF QUADMOBIUS FORMULAS


Following the algorithm in Section 2.2, we normalize a 2x2 complex projective matrix **M** by its
determinant and find the nearest unitary matrix, which by Appendix B.3.1 is special unitary. The
following are two different approaches to impelement this. We assume **M** has full rank.


**Linear Algebra** Instead of normalizing **M** directly, we take a more streamlined approach by utilizing the properties of polar decomposition and determinant. We express _det_ ( **M** ) in polar form
_det_ ( **M** )
as _re_ _[iθ]_ with _r_ = _|det_ ( **M** ) _|_ _∈_ R _, r_ _>_ 0 and _e_ _[iθ]_ = _|det_ ( **M** ) _|_ [lying] [on] [the] [unit] [circle.] [For] [polar]
decomposition **M** = **QP** with unitary matrix **Q** and positive definite Hermitian matrix **P**, we have
_det_ ( **M** ) = _det_ ( **Q** ) _det_ ( **P** ). Because **Q** is unitary, _|det_ ( **Q** ) _|_ = 1, and because **P** is positive definite
Hermitian, _det_ ( **P** ) is real and nonnegative. It follows then that _det_ ( **Q** ) = _e_ _[iθ]_ and _det_ ( **P** ) = _r_ .
To normalize **M**, we typically multiply it by a nonzero scalar _λ_ _∈_ C. For polar decomposition to
remain valid under this scaling, _λ_ must distribute as _λ_ **M** = - _|λλ|_ **[Q]** - ( _|λ|_ **P** ), meaning that only the
phase of _λ_ affects the unitary factor. Since the unitary factor **Q** is the nearest unitary matrix to **M**
in the Frobenius sense, the final solution is just _|λλ|_ **[Q]** [ such that] _[ det]_ [(] _|λ_ _[λ]_ _|_ **[Q]** [) = 1][ to be special unitary.]


in the Frobenius sense, the final solution is just _|λ|_ **[Q]** [ such that] _[ det]_ [(] _|λ|_ **[Q]** [) = 1][ to be special unitary.]

We can therefore reverse the order and first compute **Q** before normalizing its determinant. We find
a scalar _λ_ _[′]_ such that _det_ ( _λ_ _[′]_ **Q** ) = _λ_ _[′]_ [2] _det_ ( **Q** ) = 1 (since **Q** is 2x2) for _|λ_ _[′]_ _|_ = 1. We can easily solve
_λ_ _[′]_ = _det_ ( **Q** ) _[−]_ 2 [1] . Since **Q** = **UV** _[H]_ from SVD ( **M** = **UΣV** _[H]_ ) and _|det_ ( **Q** ) _|_ = 1, we can rewrite


_λ_ _[′]_ = _det_ ( **Q** ) _[−]_ 2 [1] . Since **Q** = **UV** _[H]_ from SVD ( **M** = **UΣV** _[H]_ ) and _|det_ ( **Q** ) _|_ = 1, we can rewrite

our expression simply as ~~�~~ _det_ ( **UV** _[H]_ ) **UV** _[H]_ . If **M** is singular, there is no unique solution as SVD


our expression simply as ~~�~~ _det_ ( **UV** _[H]_ ) **UV** _[H]_ . If **M** is singular, there is no unique solution as SVD

is no longer unique. This formula may still be used in practice with a specific SVD.


**Algebraic** First, we can normalize **M** to **M** _[′]_ = _det_ ( **M** ) _[−]_ [1] 2 **M** such that _det_ ( **M** _[′]_ ) = 1. Next,

we can utilize the isomorphism between _SU_ (2) and quaternions in Eq. (35) to algebraically solve
for the nearest special unitary matrix. It’s easy to verify that the unitary matrix **Q** that minimizes


15


the Frobenius distance to **M** _[′]_ maximizes _ℜ_ ( _Tr_ ( **M** _[′][H]_ **Q** )) where _ℜ_ ( _·_ ) denotes the real part. From
Appendix B.3.1, we know that **Q** will be special unitary. Thus, we can express the optimization
problem (using symbols from Eqs. (31) and (39)) as:


max
**Q** _∈SU_ (2) _[ℜ]_ [(] _[Tr]_ [(] **[M]** _[H]_ **[Q]** [)) =] _[ ℜ]_ [(] _[σα]_ [ +] _[ ξβ]_ [ +] _[ δα][ −]_ _[γβ]_ [)]


= max
_||_ **q** _||_ =1 [(] _[ℜ]_ [(] _[σ]_ [) +] _[ ℜ]_ [(] _[δ]_ [))] _[w][q]_ [ + (] _[ℑ]_ [(] _[σ]_ [)] _[ −ℑ]_ [(] _[δ]_ [))] _[x][q]_ [ + (] _[ℜ]_ [(] _[ξ]_ [)] _[ −ℜ]_ [(] _[γ]_ [))] _[y][q]_ [ + (] _[ℑ]_ [(] _[ξ]_ [) +] _[ ℑ]_ [(] _[γ]_ [))] _[z][q]_


for quaternion **q** = _wq_ + _xqi_ + _yqj_ + _zqk_ and _ℑ_ ( _·_ ) denoting the imaginary part. For **q** to be a
valid rotation, it must have unit norm. Thus, the optimization problem can be rephrased as finding
the unit norm vector whose dot product with the coefficients of the quaternion parameters above
is maximized. The solution is trivially obtained by the unit norm vector in the direction of those
coefficients. Using Eq. (35) again, we can express the solution as:


**q** ˜ = ( _ℜ_ ( _σ_ ) + _ℜ_ ( _δ_ )) + ( _ℑ_ ( _σ_ ) _−ℑ_ ( _δ_ )) _i_ + ( _ℜ_ ( _ξ_ ) _−ℜ_ ( _γ_ )) _j_ + ( _ℑ_ ( _ξ_ ) + _ℑ_ ( _γ_ )) _k_

_α_ ˜ = _σ_ + _δ,_ _β_ ˜ = _ξ −_ _γ_

**Q** _∼_ **M** _[′]_ + _adj_ ( **M** _[′]_ ) _[H]_


where tilde denotes unnormalized parameters and _adj_ ( _·_ ) denotes the adjugate. We can nor
           -           malize the parameters by dividing _α_ ˜ and _β_ [˜] by _|α_ ˜ _|_ [2] + _|β_ [˜] _|_ [2] = _|σ_ + _δ|_ [2] + _|ξ −_ _γ_ ~~_|_~~ [2] =

- _Tr_ ( **M** _[′][H]_ **M** _[′]_ ) + 2 _ℜ_ ( _det_ ( **M** _[′]_ )) = - _Tr_ ( **M** _[′][H]_ **M** _[′]_ ) + 2. Since that factor is real and distributes
linearly through _α_ ˜ and _β_ [˜] to the elements of **M** _[′]_, we can efficiently combine this normalization factor into the original normalization factor of _det_ ( **M** ) _[−]_ 2 [1] in the first step. The combined normalization


tor into the original normalization factor of _det_ ( **M** ) _[−]_ 2 in the first step. The combined normalization

factor can be written as:

1 1 1 1

=

    - _det_ ( **M** )    - _Tr_ ( **M** _[′][H]_ **M** _[′]_ ) + 2    - _det_ ( **M** )    - _T r_ ( **M** _[H]_ **M** )


1 1
=

- _Tr_ ( **M** _[′][H]_ **M** _[′]_ ) + 2 - _det_ ( **M** )


1

- _T r_ ( **M** _[H]_ **M** )


_|det_ ( **M** ) _|_ [+ 2]


_det_ ( **M** )
_|det_ ( **M** ) _|_ ( _Tr_ ( **M** _[H]_ **M** ) + 2 _|det_ ( **M** ) _|_ )


=


_|det_ ( **M** ) _|_
_det_ ( **M** )( _Tr_ ( **M** _[H]_ **M** ) + 2 _|det_ ( **M** ) _|_ ) [=]


Applying this normalization factor to **M** to obtain **M** _[∗]_ will ensure that **M** _[∗]_ + _adj_ ( **M** _[∗]_ ) _[H]_ _∈_ _SU_ (2).


B.3 NEAREST UNITARY MATRIX


B.3.1 PROOF OF NEAREST SPECIAL UNITARY MATRIX


**Proposition 2** _If M¨obius transformation_ **M** _has det_ ( **M** ) = 1 _, the nearest unitary matrix to_ **M** _in the_
_Frobenius sense is special unitary._


_Proof_ **M** has a singular value decomposition given as **M** = **UΣV** _[H]_ where **U** and **V** are unitary
matrices and **Σ** is a diagonal matrix with singular values. The determinant of **M** can be expressed
as:


_det_ ( **M** ) = _det_ ( **U** ) _det_ ( **Σ** ) _det_ ( **V** _[H]_ ) (43)


by product rule of determinants. Multiplying both sides by their complex conjugates, we obtain:


_|det_ ( **M** ) _|_ [2] = _|det_ ( **U** ) _|_ [2] _|det_ ( **Σ** ) _|_ [2] _|det_ ( **V** _[H]_ ) _|_ [2]


Since **U** and **V** _[H]_ are unitary matrices, the magnitude of their determinant is 1, so the expression
simplifies to:


_|det_ ( **M** ) _|_ [2] = _|det_ ( **Σ** ) _|_ [2] = _⇒|det_ ( **M** ) _|_ = _|det_ ( **Σ** ) _|_


because the determinant magnitudes are real and nonnegative. Since **Σ** is a diagonal matrix with
real, nonnegative elements, its determinant is simply the product of its diagonal entries and is in
turn real and nonnegative. If _det_ ( **M** ) = 1, then _|det_ ( **Σ** ) _|_ = _det_ ( **Σ** ) = 1. Coming back to the first
expression, we can now write:


_det_ ( **M** ) = _det_ ( **U** ) _det_ ( **V** _[H]_ ) = _det_ ( **UV** _[H]_ ) = 1


16


It is known that closest unitary matrix to **M** in the Frobenius sense is the unitary part of polar
decomposition (Keller, 1975) which can be computed by **UV** _[H]_ . From above, we can see that
_det_ ( **UV** _[H]_ ) = 1 which means that **UV** _[H]_ is special unitary by definition.


In noiseless situations, **Σ** is observed to be the identity matrix if _det_ ( **M** ) = 1. As noise is added,
the diagonal elements of **Σ** drift from 1, so **Σ** encodes a notion of how close a M¨obius transformation’s action is to a rotation or how much noise the problem contains, making it a candidate for
optimization.


B.3.2 DERIVATION OF NEAREST UNITARY MATRIX DERIVATIVE


The nearest unitary matrix in the Frobenius sense to a complex square matrix **M** is given by the
unitary factor **Q** of its polar decomposition **M** = **QP** where **P** is a positive semidefinite Hermitian
matrix (Keller, 1975). We can find the derivative of **Q** with respect to the elements of **M** by taking
the derivative of both sides of the polar decomposition:


_d_ **M** = _d_ ( **QP** )

_d_ **M** = ( _d_ **Q** ) **P** + **Q** ( _d_ **P** )

**Q** _[H]_ ( _d_ **M** ) = **Q** _[H]_ ( _d_ **Q** ) **P** + _d_ **P**


Taking the conjugate transpose of both sides and subtracting the two statements:


( _d_ **M** _[H]_ ) **Q** = **P** _[H]_ ( _d_ **Q** _[H]_ ) **Q** + _d_ **P** _[H]_

**Q** _[H]_ ( _d_ **M** ) _−_ ( _d_ **M** _[H]_ ) **Q** = **Q** _[H]_ ( _d_ **Q** ) **P** _−_ **P** _[H]_ ( _d_ **Q** _[H]_ ) **Q** + ( _d_ **P** _−_ _d_ **P** _[H]_ )


We observe that because **P** is Hermitian for all values of **M**, _d_ **P** must also be Hermitian, so the last
term cancels out. Furthermore, we can deduce the following from definition of unitary matrices:


**Q** _[H]_ **Q** = **I**

( _d_ **Q** _[H]_ ) **Q** + **Q** _[H]_ ( _d_ **Q** ) = 0

( _d_ **Q** _[H]_ ) **Q** = _−_ **Q** _[H]_ ( _d_ **Q** )


implying that ( _d_ **Q** _[H]_ ) **Q** is skew-Hermitian. Denoting **X** = **Q** _[H]_ ( _d_ **Q** ) and **C** = **Q** _[H]_ ( _d_ **M** ) _−_
( _d_ **M** _[H]_ ) **Q**, we can now write:


**C** = **XP** + **PX**


which takes the form of a Sylvester equation. Since **P** is Hermitian, it admits a diagonalization
**P** = **YΛY** _[H]_, where **Y** is unitary and **Λ** is a diagonal matrix of eigenvalues of **P** :


**C** = **XYΛY** _[H]_ + **YΛY** _[H]_ **X**

**Y** _[H]_ **CY** = ( **Y** _[H]_ **XY** ) **Λ** + **Λ** ( **Y** _[H]_ **XY** )


The right hand side has the same term **Y** _[H]_ **XY** multiplied on the left and right respectively by
diagonal matrix **Λ** . As such, we can equivalently express the result as follows in order to solve for
**X** and ultimately _d_ **Q** :


**Y** _[H]_ **CY** = ( _diag_ ( **Λ** ) _⊕_ _diag_ ( **Λ** )) _⊙_ ( **Y** _[H]_ **XY** )


**Y** _[H]_ **CY**
**Y** _[H]_ **XY** =
_diag_ ( **Λ** ) _⊕_ _diag_ ( **Λ** )


   - **Y** _[H]_ **CY**
**X** = **Y**
_diag_ ( **Λ** ) _⊕_ _diag_ ( **Λ** )


**Y** _[H]_


_H_ _H_ _H_

    - **Y** ( **Q** ( _d_ **M** ) _−_ ( _d_ **M** ) **Q** ) **Y**
_d_ **Q** = **QY**
_diag_ ( **Λ** ) _⊕_ _diag_ ( **Λ** )


**Y** _[H]_


where _⊕_ denotes an outer sum operation, _⊙_ denotes Hadamard multiplication (element-wise), the
division is Hadamard division (element-wise), and _diag_ ( _·_ ) is a vector formed from the diagonal
elements of the matrix . Note that this solution is only properly defined if **M** is nonsingular (i.e.
**Λ** has full rank). Otherwise, the polar decomposition is not unique and neither is its derivative. In
practice, we choose to replace any instances of division by 0 in the result above with multiplications
by 0 as a specific solution.


17


B.4 TWO-POINT SOLUTIONS


B.4.1 PROOF OF WEIGHTED CASE


**Proposition 3** _Let_ **a** _i and_ **b** _i represent the reference and target points respectively and_ **k** _a_ = **a** 1 _×_ **a** 2
_and_ **k** _b_ = **b** 1 _×_ **b** 2 _._ _For n_ = 2 _points,_ **k** _a_ = **0** _, and_ **k** _b_ = **0** _, the optimal rotation to Wahba’s problem_
_is given as the weighted average (in the Frobenius sense) between two rotations_ **R** 1 _and_ **R** 2 _defined_
_by_ **R** _i_ **a** _i_ = **b** _i_ _and_ **R** _i_ _||_ **kk** _aa||_ [=] _||_ **kk** _bb||_ _[.]_


_Lemma:_ _If all points lie in the plane z=0 and_ **k** _a_ = 0 _,_ **k** _b_ = 0 _, and_ **k** _a ·_ **k** _b_ _>_ 0 _, the optimal rotation_
_is a rotation around the z-axis._


Since all points lie in the plane _z_ = 0, the last column and row of **B** (Eq. (2)) are zero. As a
result, the last column and row of **BB** _[T]_ and **B** _[T]_ **B** are also zero, so they both have a kernel vector
of (0 _,_ 0 _,_ 1). For the SVD of **B** given as **UΣV** _[T]_, the optimal rotation **R** (via Markley (1987)) can
take the form:


�� _·_ _·_ 0

_·_ _·_ 0
0 0 1


**R** =


- _·_ _·_ 0

_·_ _·_ 0
0 0 1


��1 0 0
0 1 0
0 0 _det_ ( **U** ) _det_ ( **V** )


where _det_ ( **U** ) _det_ ( **V** ) is either 1 or -1 since **U** and **V** are orthogonal matrices. Thus, the last column
and row of **R** are both (0 _,_ 0 _,_ 1) or (0 _,_ 0 _, −_ 1). In order for **R** to be a valid rotation matrix, the
remaining upper 2x2 submatrix must be an orthogonal matrix which can be generated by a single
parameter _θ_ . Furthermore, the sign of the bottom right corner element of **R** must be the same as the
determinant of the upper 2x2 submatrix for _det_ ( **R** ) = 1. These conditions reduce **R** to one of the
two general forms:

       - _cos_ ( _θ_ 1) _−sin_ ( _θ_ 1) 0�        - _cos_ ( _θ_ 2) _sin_ ( _θ_ 2) 0        _sin_ ( _θ_ 1) _cos_ ( _θ_ 1) 0 _,_ _sin_ ( _θ_ 2) _−cos_ ( _θ_ 2) 0
0 0 1 0 0 _−_ 1


We denote the former as **R** _SO_ and the latter as **R** _O_ . The optimal solution to Wahba’s problem
maximizes the gain function _Tr_ ( **RB** _[T]_ ) Lourakis and Terzakis (2018). This quantity for both forms
can be expressed as below:


_Tr_ ( **R** _SO_ **B** _[T]_ ) = _λ_ 1 _,_ 1 _cos_ ( _θ_ 1) + _λ_ 1 _,_ 2 _sin_ ( _θ_ 1)

_Tr_ ( **R** _O_ **B** _[T]_ ) = _λ_ 2 _,_ 1 _cos_ ( _θ_ 2) + _λ_ 2 _,_ 2 _sin_ ( _θ_ 2)
_λ_ 1 _,_ 1 = **B** 1 _,_ 1 + **B** 2 _,_ 2 _,_ _λ_ 1 _,_ 2 = **B** 2 _,_ 1 _−_ **B** 1 _,_ 2
_λ_ 2 _,_ 1 = **B** 1 _,_ 1 _−_ **B** 2 _,_ 2 _,_ _λ_ 2 _,_ 2 = **B** 2 _,_ 1 + **B** 1 _,_ 2


The gain function in both cases is the dot product between ( _λi,_ 1 _, λi,_ 2) and ( _cos_ ( _θi_ ) _, sin_ ( _θi_ )). Its
maximum value (subject to the constraint _cos_ ( _θi_ ) [2] + _sin_ ( _θi_ ) [2] = 1) is obtained by the unit vector
aligned with ( _λi,_ 1 _, λi,_ 2), i.e.:


_λi,_ 1 _λi,_ 2
_cos_ ( _θi_ ) =           - _,_ _sin_ ( _θi_ ) =           _λ_ [2] _i,_ 1 [+] _[ λ]_ _i,_ [2] 2 _λ_ [2] _i,_ 1 [+] _[ λ]_ _i,_ [2] 2


Substituting this back into the gain function, we see that the optimal value is simply the magnitude
of ( _λi,_ 1 _, λi,_ 2):


       -        _Tr_ ( **R** _SO_ **B** _[T]_ ) = _λ_ [2] 1 _,_ 1 [+] _[ λ]_ 1 [2] _,_ 2 _[,]_ _Tr_ ( **R** _O_ **B** _[T]_ ) = _λ_ [2] 2 _,_ 1 [+] _[ λ]_ 2 [2] _,_ 2


Since the square root function is monotonically increasing, the larger of the two radicands corresponds to the larger gain value. We can compare them directly by taking their difference:


( _λ_ [2] 1 _,_ 1 [+] _[ λ]_ 1 [2] _,_ 2 [)] _[ −]_ [(] _[λ]_ [2] 2 _,_ 1 [+] _[ λ]_ 2 [2] _,_ 2 [) = 4] _[w]_ [1] _[w]_ [2][(] **[k]** _[a]_ _[·]_ **[ k]** _[b]_ [)]


where _wi_ are the weights. Since the weights are positive and the cross products are assumed nonzero,
the quantity above is positive when **k** _a_ and **k** _b_ point in the same direction and negative otherwise.
Thus, when the cross products of the reference and target sets are aligned, **R** _SO_ corresponds to the
larger gain value and is the optimal rotation. It takes the form of a rotation about the z-axis.


18


_,_


- _cos_ ( _θ_ 2) _sin_ ( _θ_ 2) 0
_sin_ ( _θ_ 2) _−cos_ ( _θ_ 2) 0
0 0 _−_ 1


_Proof_ We assume that all points lie in the plane _z_ = 0 and that the cross product of the reference
and target sets are nonzero and are aligned. This will be generalized later. We construct rotations
**R** 1 and **R** 2 to be rotations about the z-axis that align **a** 1 to **b** 1 and **a** 2 to **b** 2 respectively. Since the
input points have unit length and the vector norm is rotationally invariant, we can rewrite the loss
function as:


_w_ 1 _||_ **b** 1 _−_ **Ra** 1 _||_ [2] + _w_ 2 _||_ **b** 2 _−_ **Ra** 2 _||_ [2]

= _w_ 1 _||_ **a** 1 _−_ **R** _[T]_ 1 **[Ra]** [1] _[||]_ [2][ +] _[ w]_ [2] _[||]_ **[a]** [2] _[−]_ **[R]** _[T]_ 2 **[Ra]** [2] _[||]_ [2]

= _w_ 1 _||_ ( **I** _−_ **R** _[T]_ 1 **[R]** [)] **[a]** [1] _[||]_ [2][ +] _[ w]_ [2] _[||]_ [(] **[I]** _[ −]_ **[R]** 2 _[T]_ **[R]** [)] **[a]** [2] _[||]_ [2]

= _w_ 1 **a** _[T]_ 1 [(] **[I]** _[ −]_ **[R]** 1 _[T]_ **[R]** [)] _[T]_ [ (] **[I]** _[ −]_ **[R]** 1 _[T]_ **[R]** [)] **[a]** [1] [+] _[ w]_ [2] **[a]** _[T]_ 2 [(] **[I]** _[ −]_ **[R]** 2 _[T]_ **[R]** [)] _[T]_ [ (] **[I]** _[ −]_ **[R]** 2 _[T]_ **[R]** [)] **[a]** [2]
= 2( _w_ 1 + _w_ 2) _−_ 2 _w_ 1 **a** _[T]_ 1 **[R]** 1 _[T]_ **[Ra]** [1] _[−]_ [2] _[w]_ [2] **[a]** _[T]_ 2 **[R]** 2 _[T]_ **[Ra]** [2]

using the fact **a** _[T]_ _i_ **[R]** _i_ _[T]_ **[Ra]** _[i]_ [=] **[a]** _[T]_ _i_ **[R]** _[T]_ **[ R]** _[i]_ **[a]** _[i]_ [.] [Under] [our] [assumptions,] [the] [lemma] [establishes] [that] [the]
optimal rotation **R** is a rotation about the z-axis. Since both **R** 1 and **R** 2 are also rotations about the
z-axis, we can easily verify that the products **R** _[T]_ 1 **[R]** [ and] **[ R]** 2 _[T]_ **[R]** [ are rotations about the z-axis as well.]
Using Rodrigues’ rotation formula, we can expand the term below as follows:


**a** _[T]_ 1 **[R]** 1 _[T]_ **[Ra]** [1] [=] **[ a]** [1] _[·]_ [(] _[cos]_ [(] _[ϕ]_ [)] **[a]** [1] [+] _[ sin]_ [(] _[ϕ]_ [)] **[k]** _[ ×]_ **[ a]** [1] [+ (1] _[ −]_ _[cos]_ [(] _[ϕ]_ [))(] **[k]** _[ ·]_ **[ a]** [1][)] **[k]** [)]
= _cos_ ( _ϕ_ ) + _sin_ ( _ϕ_ )( **a** 1 _·_ ( **k** _×_ **a** 1)) = _cos_ ( _ϕ_ )


where _ϕ_ is the angle of rotation of **R** _[T]_ 1 **[R]** [ and] **[ k]** [ = [0] _[,]_ [ 0] _[,]_ [ 1]] _[T]_ [is the axis of rotation.] [The simple result]
is due to the fact that **a** 1 is orthogonal to the axis of rotation and has unit length. On the other hand,
we note that the Frobenius norm between **R** 1 and **R** computes the following:

_||_ **R** 1 _−_ **R** _||_ [2] _F_ [=] _[ Tr]_ [((] **[R]** [1] _[−]_ **[R]** [)] _[T]_ [ (] **[R]** [1] _[−]_ **[R]** [))]

= 6 _−_ 2 _Tr_ ( **R** _[T]_ 1 **[R]** [)]

= 6 _−_ 2 _Tr_ ( _cos_ ( _ϕ_ ) **I** + _sin_ ( _ϕ_ )[ **k** ] _×_ + (1 _−_ _cos_ ( _ϕ_ )) **kk** _[T]_ )
= 6 _−_ 6 _cos_ ( _ϕ_ ) _−_ 2(1 _−_ _cos_ ( _ϕ_ )) = 4 _−_ 4 _cos_ ( _ϕ_ )

_cos_ ( _ϕ_ ) = 1 _−_ [1] 4 _[||]_ **[R]** [1] _[ −]_ **[R]** _[||]_ _F_ [2]

The expansion of **R** _[T]_ 1 **[R]** [1] [above] [is] [due] [to] [the] [axis-angle] [formula] [for] [rotation] [matrices] [where] [[] **[k]** []] _[×]_
denotes the traceless skew-symmetric matrix formed from **k** representing a vector cross product.
Deriving a similar result for **a** _[T]_ 2 **[R]** 2 _[T]_ **[Ra]** [2] [and plugging both back into our reformulated loss function,]
we can rewrite it as:


[1] _F_ [)] _[ −]_ [2] _[w]_ [2][(1] _[ −]_ [1]

4 _[||]_ **[R]** [1] _[ −]_ **[R]** _[||]_ [2] 4


2( _w_ 1 + _w_ 2) _−_ 2 _w_ 1(1 _−_ [1]


_F_ [)]
4 _[||]_ **[R]** [2] _[ −]_ **[R]** _[||]_ [2]


= [1]


[1] _F_ [+] [1]

2 _[w]_ [1] _[||]_ **[R]** [1] _[ −]_ **[R]** _[||]_ [2] 2


2 _[w]_ [2] _[||]_ **[R]** [2] _[ −]_ **[R]** _[||]_ _F_ [2]


Through this expression, we can see that the rotation **R** which minimized our original loss is exactly
the rotation that represents the weighted average in the Frobenius sense between **R** 1 and **R** 2 as
specified in Markley et al. (2007). The uniform factor of [1] 2 [is irrelevant to the optimization.]


Through this expression, we can see that the rotation **R** which minimized our original loss is exactly
the rotation that represents the weighted average in the Frobenius sense between **R** 1 and **R** 2 as
specified in Markley et al. (2007). The uniform factor of [1] 2 [is irrelevant to the optimization.]


Now we generalize the result. Starting from the assumed configuration, we can extend it to general
configurations by applying arbitrary rotations **R** _a_ and **R** _b_ to the reference and target points respectively, transforming them into **a** _[′]_ _i_ [and] **[b]** _[′]_ _i_ [.] [In] [this] [new] [coordinate] [frame,] [the] [rotation] [matrix] **[R]** _[′]_ [is]
related to the original optimal matrix **R** as shown below:

    - [2]    - [2]


_wi||_ **b** _i −_ **Ra** _i||_ [2] =  
_i_ _i_


_wi||_ **R** _b_ **b** _i −_ **R** _b_ **Ra** _i||_ [2]

_i_


_wi||_ **R** _b_ **b** _i −_ **R** _b_ **R** ( **R** _[T]_ _a_ **[R]** _[a]_ [)] **[a]** _[i][||]_ [2] [=]  _i_ _i_


= 


_wi||_ **b** _[′]_ _i_ _[−]_ [(] **[R]** _[b]_ **[RR]** _a_ _[T]_ [)] **[a]** _i_ _[′]_ _[||]_ [2]
_i_


**R** _[′]_ = **R** _b_ **RR** _[T]_ _a_


Because the vector norm is invariant under rotation, the optimal loss value remains unchanged across
all coordinate frames. Since the optimal value from the original coordinate frame is preserved


19


above, **R** _[′]_ represents the optimal rotation in the new frame. Furthermore, the Frobenius norm is also
rotation-invariant, so we can apply the required rotations to estimate **R** _[′]_ as follows:

   - [2]    - _[T]_ _[T]_ [2]


_wi||_ **R** _i −_ **R** _||_ [2] _F_ [=]  _i_ _i_


_wi||_ **R** _b_ **R** _i_ **R** _[T]_ _a_ _[−]_ **[R]** _[b]_ **[RR]** _a_ _[T]_ _[||]_ _F_ [2]
_i_


= - _wi||_ **R** _b_ **R** _i_ **R** _[T]_ _a_ _[−]_ **[R]** _[′][||]_ _F_ [2]

_i_


**R** _[′]_ 1 [=] **[ R]** _[b]_ **[R]** [1] **[R]** _a_ _[T]_ _[,]_ **[R]** _[′]_ 2 [=] **[ R]** _[b]_ **[R]** [2] **[R]** _a_ _[T]_
Thus, in the general case, the optimal rotation is given by the weighted average rotation between **R** _[′]_ 1
and **R** _[′]_ 2 [.] [We can uniquely identify those rotations with at least two linearly independent points they]
transform. Starting with the reference and target sets:


**R** _i_ **a** _i_ _≡_ **b** _i_
**R** _b_ **R** _i_ ( **R** _[T]_ _a_ **[R]** _[a]_ [)] **[a]** _[i]_ [=] **[ R]** _[b]_ **[b]** _[i]_
**R** _[′]_ _i_ **[a]** _[′]_ _i_ [=] **[ b]** _i_ _[′]_
Each rotation still aligns their respective reference point to their target point. Furthermore, in our
original coordinate frame, **k** _a_ and **k** _b_ are aligned and are parallel or antiparallel to **R** _i_ ’s axis of
rotation (z-axis), so they are unchanged by **R** _i_ . As a result:

**k** _a_ **k** _b_
**R** _i_ _||_ **k** _a||_ [=] _||_ **k** _b||_

**k** _b_

**R** _b_ **R** _i_ ( **R** _[T]_ _a_ **[R]** _[a]_ [)] **[k]** _[a]_

_||_ **k** _a||_ [=] **[ R]** _[b]_ _||_ **k** _b||_

**R** _a_ ( **a** 1 _×_ **a** 2) **R** _b_ ( **b** 1 _×_ **b** 2)
**R** _[′]_ _i_ _||_ **R** _a_ ( **a** 1 _×_ **a** 2) _||_ [=] _||_ **R** _b_ ( **b** 1 _×_ **b** 2) _||_

**a** _[′]_ 1 _[×]_ **[ a]** 2 _[′]_ **b** _[′]_ 1 _[×]_ **[ b]** 2 _[′]_
**R** _[′]_ _i_ _||_ **a** _[′]_ 1 _[×]_ **[ a]** 2 _[′]_ _[||]_ [=] _||_ **b** _[′]_ 1 _[×]_ **[ b]** 2 _[′]_ _[||]_

due to rotations distributing over the cross product. Thus, we can identify **R** _[′]_ 1 [and] **[ R]** 2 _[′]_ [as the rotations]
that align their corresponding reference point to their target point along with the cross products
of the reference and target sets. As the cross products are assumed nonzero and are orthogonal
to their respective point set, the two points aligned by each rotation are always independent and
therefore uniquely define the rotations. As shown, the optimal rotation is the weighted average in
the Frobenius sense between them.


B.4.2 PROOF OF UNWEIGHTED CASE


**Proposition 4** _Let_ **a** _i,_ **b** _i, and wi_ _represent the reference points, target points, and weights respec-_
_tively._ _Given_ _n_ = 2 _points,_ _w_ 1 = _w_ 2 _,_ **a** 1 _×_ **a** 2 = **0** _,_ _and_ **b** 1 _×_ **b** 2 = **0** _,_ _the_ _optimal_ _rotation_
_to_ _Wahba’s_ _problem_ _is_ _given_ _by_ _the_ _unique_ _rotation_ **R** _defined_ _by_ **R** ( _||_ **aa** 11++ **aa** 22 _||_ [)] [=] _||_ **bb** 11++ **bb** 22 _||_ _[and]_

**R** ( _||_ **aa** 11 _−−_ **aa** 22 _||_ [) =] _||_ **bb** 11 _−−_ **bb** 22 _||_ _[.]_


_Proof_ For two 3D unit vectors **v** 1 and **v** 2, we introduce the following notation and easily verifiable
results:

**v** ˜ _[−]_ _≡_ **v** 1 _−_ **v** 2 _,_ **v** ˜ [+] _≡_ **v** 1 + **v** 2


**v** ˜ _[−]_
**v** _[−]_ =


**v** ˜ _[−]_ **v** ˜ [+]

**[v]** [+] [=]
_||_ **v** ˜ _[−]_ _||_ _[,]_ _||_ **v** ˜ [+]


**v** ˜ _[−]_ _||_ _||_ **v** ˜ [+] _||_

**v** ˜ _[−]_ _·_ ˜ **v** [+] = 0


**v** 1 _·_ ˜ **v** [+] = **v** 2 _·_ ˜ **v** [+]

**v** ˜ _[−]_ _×_ ˜ **v** [+] = 2( **v** 1 _×_ **v** 2)

**v** 1 _×_ **v** 2 = **0** = _⇒_ **v** ˜ _[−]_ = **0** _,_ **v** ˜ [+] = **0**

If **v** 1 _×_ **v** 2 = **0**, then the two vectors **v** _[−]_ and **v** [+] are well-defined and form an orthonormal basis
for the plane spanned by **v** 1 and **v** 2. Consequently, **v** _[−]_ and **v** [+] created from one pair of linearly
independent unit vectors can be perfectly aligned with those created from another pair.


20


With **a** 1 _×_ **a** 2 = **0** _,_ **b** 1 _×_ **b** 2 = **0**, we initially assume that the points are configured such that they
all lie in the plane _z_ = 0 and that **a** [+] = **b** [+] and **a** _[−]_ = **b** _[−]_ . This is generalized later. For this
configuration, we note the following:

**a** 1 _×_ **a** 2 = [1]

2 [(˜] **[a]** _[−]_ _[×]_ [ ˜] **[a]** [+][)]


= [1]


[1] [1]

2 _[||]_ **[a]** [˜] _[−][||||]_ **[a]** [˜][+] _[||]_ [(] **[a]** _[−]_ _[×]_ **[ a]** [+][) =] 2


2 _[||]_ **[a]** [˜] _[−][||||]_ **[a]** [˜][+] _[||]_ [(] **[b]** _[−]_ _[×]_ **[ b]** [+][)]


_[||]_ **[a]** [˜] _[−][||||]_ **[a]** [˜][+] _[||]_ ( **b** [˜] _[−]_ _×_ **b** [˜][+] ) = _[||]_ **[a]** [˜] _[−][||||]_ **[a]** [˜][+] _[||]_

2 _||_ **b** [˜] _[−]_ _||||_ **b** [˜][+] _||_ _||_ **b** [˜] _[−]_ _||||_ **b** [˜][+] _||_


= _[||]_ **[a]** [˜] _[−][||||]_ **[a]** [˜][+] _[||]_


( **b** 1 _×_ **b** 2)
_||_ **b** [˜] _[−]_ _||||_ **b** [˜][+] _||_


= _⇒_ ( **a** 1 _×_ **a** 2) _·_ ( **b** 1 _×_ **b** 2) _>_ 0


Thus, the cross products are aligned in this configuration, and from the lemma in the general case
proof, the optimal rotation is a rotation about the z-axis.


From the dot product equality above, we can deduce that **a** [+] is equidistant from **a** 1 _,_ **a** 2. The dot
product calculates the cosine of the angle between linearly independent unit vectors measured in the
plane spanned by the vectors ( _z_ = 0 in our case). We know from the proof in the general case that
the dot product of a unit vector in the plane _z_ = 0 with itself after a rotation about the z-axis is
the cosine of the angle of rotation. That angle is measured in the plane perpendicular to the axis of
rotation, which is also the plane _z_ = 0. Thus, constructing rotations **Ra** 1 and **Ra** 2 which rotate **a** [+]
about the z-axis to **a** 1 and **a** 2 respectively, we can write the following:

**a** 1 _·_ **a** [+] = **a** 2 _·_ **a** [+] = **a** [+] _·_ ( **Ra** 1 **a** [+] ) = **a** [+] _·_ ( **Ra** 2 **a** [+] ) = _cos_ ( _ϕ_ )

where _ϕ_ denotes the angle of rotation of **Ra** 1, making _|ϕ|_ (canonically positive) the angle between
**a** 1 and **a** [+] . In general, **Ra** 1 = **Ra** 2, otherwise **a** 1 and **a** 2 would be identical. In order for the above
to still hold, the angle of rotation of **Ra** 2 must have the same magnitude but opposite sign of _ϕ_ . A
similar statement can be made for the target points.


Let **Rb** 1 and **Rb** 2 represent rotations about the z-axis that align **b** [+] with **b** 1 and **b** 2 respectively.
Recall **a** [+] = **b** [+] . We construct the rotations **R** 1 = **Rb** 1 **R** _[T]_ **a** 1 [and] **[R]** [2] [=] **[R][b]** 2 **[R]** _[T]_ **a** 2 [which] [are] [also]
about the z-axis to align **a** 1 with **b** 1 and **a** 2 with **b** 2 respectively. If _ψ_ is the rotation angle of **Rb** 1,
then the angle of rotation for **R** 1 is _−ϕ_ + _ψ_ since **Ra** 1 and **Rb** 1 share the same axis of rotation and
transposing a rotation matrix negates the rotation angle. For **R** 2, the rotation angle is _ϕ −_ _ψ_, as **Ra** 2
rotates by _−ϕ_ and **Rb** 2 by _−ψ_ . Thus, the rotation angles of **R** 1 and **R** 2 have equal magnitudes but
opposite signs.


From the proof in the general case, the optimal rotation **R** is the weighted average in the Frobenius
sense between the rotations **R** 1 and **R** 2 recently constructed. The weighted average rotation maximizes the quantity _Tr_ ( **RB** _[′][T]_ ) where **B** _[′]_ = [�] _i_ _[w][i]_ **[R]** _[i]_ [Markley et al. (2007).] [Given the previously]
made statements and the fact that _w_ 1 = _w_ 2, we can calculate **B** _[′]_ as:


**R** 1 =


- _cos_ ( _−ϕ_ + _ψ_ ) _−sin_ ( _−ϕ_ + _ψ_ ) 0�
_sin_ ( _−ϕ_ + _ψ_ ) _cos_ ( _−ϕ_ + _ψ_ ) 0
0 0 1


_,_ **R** 2 =


- _cos_ ( _ϕ −_ _ψ_ ) _−sin_ ( _ϕ −_ _ψ_ ) 0�
_sin_ ( _ϕ −_ _ψ_ ) _cos_ ( _ϕ −_ _ψ_ ) 0
0 0 1


_,_


**B** _[′]_ = _w_ 1 **R** 1 + _w_ 2 **R** 2 = 2 _w_ 1


- _cos_ ( _−ϕ_ + _ψ_ ) 0 0�
0 _cos_ ( _−ϕ_ + _ψ_ ) 0
0 0 1


due to the fact that sine is an odd function and cosine is an even function. Since **R** is a rotation about
the z-axis, we can directly compute _Tr_ ( **RB** _[′][T]_ ) as 2 _w_ 1(2 _cos_ ( _−ϕ_ + _ψ_ ) _cos_ ( _θ_ ) + 1) where _θ_ is **R** ’s
angle of rotation. We can trivially see that _θ_ must take on a value of 0 or _π_ (mod 2 _π_ ) to be optimal,
depending on the sign of _cos_ ( _−ϕ_ + _ψ_ ) as _w_ 1 is positive. That sign can be determined considering
**a** _[−]_ and **b** _[−]_ are aligned:


**a** ˜ _[−]_ _·_ **b** [˜] _[−]_ _>_ 0

( **Ra** 1 **a** [+] _−_ **Ra** 2 **a** [+] ) _·_ ( **Rb** 1 **b** [+] _−_ **Rb** 2 **b** [+] ) _>_ 0

**a** [+] _·_ (( **Ra** 1 _−_ **Ra** 2) _[T]_ ( **Rb** 1 _−_ **Rb** 2) **a** [+] ) _>_ 0
_cos_ ( _−ϕ_ + _ψ_ ) _−_ _cos_ ( _−ϕ −_ _ψ_ ) _−_ _cos_ ( _ϕ_ + _ψ_ ) + _cos_ ( _ϕ −_ _ψ_ ) _>_ 0
2 _cos_ ( _−ϕ_ + _ψ_ ) _−_ 2 _cos_ ( _ϕ_ + _ψ_ ) _>_ 0


21


Since **a** [+] and **b** [+] are also aligned, we can similarly derive 2 _cos_ ( _−ϕ_ + _ψ_ )+2 _cos_ ( _ϕ_ + _ψ_ ) _>_ 0. Adding
both inequalities together (valid since they are positive quantities), we find that _cos_ ( _−ϕ_ + _ψ_ ) _>_ 0.
Thus, _θ_ must be 0 to maximize _Tr_ ( **RB** _[′][T]_ ), resulting in **R** being the identity matrix and indicating
that the current alignment is the optimal one.


To generalize this, we again apply arbitrary rotations **R** _a,_ **R** _b_ to the reference and target sets respectively, transforming them into **a** _[′]_ _i_ _[,]_ **[ b]** _[′]_ _i_ [.] [From the proof in the general case,] [the new optimal rotation]
**R** _[′]_ = **R** _b_ **RR** _[T]_ _a_ [=] **[ R]** _[b]_ **[R]** _a_ _[T]_ [.] [Now, we simply verify below that this rotation aligns] **[ a]** _[′]_ [+][ to] **[ b]** _[′]_ [+][ and] **[ a]** _[′−]_
to **b** _[′−]_ (combined _±_ notation for convenience):

**a** 1 _±_ **a** 2 **b** 1 _±_ **b** 2
**a** _[±]_ = **b** _[±]_ =
_||_ **a** 1 _±_ **a** 2 _||_ [=] _||_ **b** 1 _±_ **b** 2 _||_
**R** _b_ ( **a** 1 _±_ **a** 2) **b** _[′]_ 1 _[±]_ **[ b]** 2 _[′]_

=
_||_ **a** 1 _±_ **a** 2 _||_ _||_ **b** _[′]_ 1 _[±]_ **[ b]** 2 _[′]_ _[||]_

**R** _b_ **R** _[T]_ _a_ [(] **[a]** 1 _[′]_ _[±]_ **[ a]** 2 _[′]_ [)] **b** _[′]_ 1 _[±]_ **[ b]** 2 _[′]_
=
_||_ **a** _[′]_ 1 _[±]_ **[ a]** 2 _[′]_ _[||]_ _||_ **b** _[′]_ 1 _[±]_ **[ b]** 2 _[′]_ _[||]_

**R** _[′]_ **a** _[′±]_ = **b** _[′±]_


Since **a** _[′]_ [+] and **a** _[′−]_ are orthogonal, they are also linearly independent, and their transformation
uniquely defines the rotation **R** _[′]_, thereby completing the proof.


B.4.3 AVERAGE OF TWO UNNORMALIZED QUATERNIONS


In Markley et al. (2007), it was shown that the average rotation matrix in the Frobenius sense can be
calculated via the quaternion **q** which optimizes the following:


**M** =        - _wi_ **q** _i_ **q** _[T]_ _i_


_i_

max **q** _[T]_ **Mq** _s.t._ _||_ **q** _||_ = 1
**q**


Where **q** _i_ are the unit norm quaternions corresponding to the rotations being averaged (sign of **q** _i_
is irrelevant). The solution is the eigenvector corresponding to the largest eigenvalue of **M** . In the
two point approach to Wahba’s problem proposed previously, we need to construct two quaternion
rotations and average them. The formulation above assumes all quaternions have unit norm. However, it would be computationally advantageous (see Table 5) if we did not have to normalize the
constructed rotations, thereby avoiding two square root and division operations. From Markley et al.
(2007), it is known that the average rotation in the two rotation case is simply a linear combination
of the rotations being averaged. To average unnormalized quaterions **q** ˜1 and **q** ˜2, we can express **M**
and **q** as:


_||_ **q** ˜2 _||_ [2]
**M** = _w_ 1 _||_ **q** ˜1 _||_ [2] **[q]** [˜][1] **[q]** [˜] 1 _[T]_ [+] _[ w]_ [2] **[q]** [˜][2] **[q]** [˜] 2 _[T]_

**q** = _µ_ **q** ˜1 + _ν_ **q** ˜2

where _µ, ν_ are scalars. The above takes advantage of the fact that scaling **M** does not change its
eigenvectors. Thus, we reduce the problem from estimating a unit quaternion to estimating two
scalars. As a result, we can rewrite the objective as:


  - _||_ **q** ˜1 _||_ [2] **q** ˜1 _·_ ˜ **q** 2
**Γ** =
**q** ˜1 _·_ ˜ **q** 2 _||_ **q** ˜2 _||_ [2]


- - _µ_ _,_ **v** =
_ν_


**Λ** 1 _,_ 1 = _w_ 1 _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2] + _w_ 2(˜ **q** 1 _·_ ˜ **q** 2) [2]

**Λ** 1 _,_ 2 = **Λ** 2 _,_ 1 = ( _w_ 1 + _w_ 2) _||_ **q** ˜2 _||_ [2] (˜ **q** 1 _·_ ˜ **q** 2)


**Λ** 2 _,_ 2 = _||_ **q** ˜2 _||_ [2][�] _w_ 2 _||_ **q** ˜2 _||_ [2] + _[w]_ [1][(˜] **[q]** [1] _[ ·]_ [ ˜] **[q]** [2][)][2]

_||_ **q** ˜1 _||_ [2]

max **v** _[T]_ **Λv** _s.t._ **v** _[T]_ **Γv** = 1
**v**


where _·_ denotes the usual vector dot product. **Γ** is the quadratic constraint ensuring that the linear
combination of **q** ˜1 and **q** ˜2 has unit norm, and **Λ** is the new 2x2 objective to optimize over. Using


22


the method of Lagrange multipliers, we find that the solution to the above takes the form of a generalized eigenvalue problem **Λv** = _λ_ **Γv** . Note that the scaling constraint **Γ** is positive semidefinite,
generally representing the equation of an ellipse. Assuming **Γ** is invertible and well-conditioned (it
is discussed later when this is not the case), the solution is the eigenvector of **Γ** _[−]_ [1] **Λ** corresponding
to the largest eigenvalue. Through simplification and scaling, we can express the matrix similarly
as:


             - _w_ 1 _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2] _w_ 1 _||_ **q** ˜2 _||_ [2] (˜ **q** 1 _·_ ˜ **q** 2)�
**Γ** _[−]_ [1] **Λ** _∼_
_w_ 2 _||_ **q** ˜1 _||_ [2] (˜ **q** 1 _·_ ˜ **q** 2) _w_ 2 _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2]


which maintains its eigenvectors from before. Since the matrix is only 2x2, the eigenvector **v** corresponding to the largest eigenvalue can be expressed in closed form. Scaling the eigenvector by the
constraint **v** _[T]_ **Γv** = 1 and substituting it back into the original linear combination of **q** ˜1 and **q** ˜2, we
obtain the average quaternion as:


_µ_ **q** ˜1 + _ν_ **q** ˜2
**q** =

~~�~~ _||_ **q** ˜1 _||_ [2] _µ_ [2] + _||_ **q** ˜2 _||_ [2] _ν_ [2] + 2(˜ **q** 1 _·_ ˜ **q** 2) _µν_


where the values _µ_ and _ν_ can be expressed equivalently in two ways:


_τ_ [(1)] = ( _w_ 1 _−_ _w_ 2) _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2] _,_ _ω_ [(1)] = 2 _w_ 1 _||_ **q** ˜2 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2) _,_ _ν_ [(1)] = 2 _w_ 2 _||_ **q** ˜1 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2)


~~�~~
_µ_ [(1)] = _τ_ [(1)] + ( _τ_ [(1)] ) [2] + _ω_ [(1)] _ν_ [(1)]


or


_τ_ [(2)] = ( _w_ 2 _−_ _w_ 1) _||_ **q** ˜1 _||_ [2] _||_ **q** ˜2 _||_ [2] _,_ _ω_ [(2)] = 2 _w_ 2 _||_ **q** ˜1 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2) _,_ _µ_ [(2)] = 2 _w_ 1 _||_ **q** ˜2 _||_ [2] ( ˜ **q** 1 _·_ ˜ **q** 2)


          _ν_ [(2)] = _τ_ [(2)] + ( _τ_ [(2)] ) [2] + _ω_ [(2)] _µ_ [(2)]


Both yield the same result except when **q** ˜1 _·_ ˜ **q** 2 = 0 in which case the rotation corresponding to the
larger weight is chosen. If _w_ 1 = _w_ 2 in that case, then there is no unique solution and either of the
rotations can be selected. The former solution set is used when _w_ 1 _> w_ 2 and the latter is used when
_w_ 1 _≤_ _w_ 2 as to approach the correct value as **q** ˜1 _·_ ˜ **q** 2 _→_ 0.


_√_
Note that the denominator in the expression for the average quaternion is simply


Note that the denominator in the expression for the average quaternion is simply **v** _[T]_ **Γv** . Previ
ously, **Γ** was assumed non-singular and well-conditioned, but there are two cases in practice where
this fails to hold. The first is when **q** ˜1 and **q** ˜2 are linearly dependent, i.e. they represent the same
rotation. If we choose the solution constants above by the previously described strategy and examine
the expressions for _µ_ and _ν_, then it can be seen that **v** _[T]_ **Γv** is in fact strictly positive for nontrivial
solutions **v** and nonzero weights/magnitudes. Furthermore, it can also be seen that _µ_ **q** ˜1 and _ν_ **q** ˜2
share the same direction in this case and thus cannot cancel out. The second case occurs when the
magnitudes of **q** ˜1 and/or **q** ˜2 are small, causing **Γ** to be ill-conditioned. This case can be avoided by
using the strategy described in Appendix D.2 to only obtain quaternions of sufficient magnitude or
by simply scaling/normalizing the rotations when necessary.


B.4.4 DEGENERATE CASE SOLUTION


The degenerate case occurs when either of the cross products of the reference or target points vanish,
and the previous approaches for the two point case cannot be applied. This is because the solution
is no longer unique. A particular one can be efficiently found through the following approach.


We assume without loss of generality that the target points are collinear (the reference points may
or may not be) and the first target point is aligned with the x-axis (i.e. **b** 1 = (1 _,_ 0 _,_ 0)). In this case,
the last two columns of the constraint **C** _i_ (Eq. (17)) vanish. We can thus write our optimization as:


�( _m −_ _x_ ) _i_ _y −_ _zi_   -   - _α_
**C** _i_ = _−y −_ _zi_ ( _x_ + _m_ ) _i_ _,_ **u** = _β_

**Z** =    - _wi_ **C** _[H]_ _i_ **[C]** _[i]_

_i_

min _[s.t.]_ **[u]** _[H]_ **[u]** [ = 1]
**u** **[u]** _[H]_ **[Zu]**


23


This optimization is simpler than before and can now be solved directly over the special unitary
parameters. Since **Z** is Hermitian and positive semidefinite, the solution is the complex eigenvector
of **Z** corresponding to the smallest eigenvalue. For reference points **a** _i_ = ( _xi, yi, zi_ ), this can be
expressed in closed form as:


             - _w_ 1 _x_ 1 + _w_ 2 _x_ 2 + _||w_ 1 **a** 1 + _w_ 2 **a** 2 _||_              **u** ˜ =
_w_ 1 _z_ 1 + _w_ 2 _z_ 2 _−_ ( _w_ 1 _y_ 1 + _w_ 2 _y_ 2) _i_


or

             - _w_ 1 _x_ 1 _−_ _w_ 2 _x_ 2 + _||w_ 1 **a** 1 _−_ _w_ 2 **a** 2 _||_              **u** ˜ =
_w_ 1 _z_ 1 _−_ _w_ 2 _z_ 2 _−_ ( _w_ 1 _y_ 1 _−_ _w_ 2 _y_ 2) _i_


where **u** ˜ is the unnormalized eigenvector and the correct solution depends on the target points’
configuration. If the dot product of the target points is positive, then the first expression is correct.
Otherwise, the second is correct. Note that eigenvectors are only unique up to scale, so even after
normalizing the solution so that **u** _[H]_ **u** = 1, we can still apply an arbitary unitary scaling of _e_ _[θi]_ . This
corresponds to a rotation about the x-axis and parameterizes the family of optimal solutions.


For arbitrary collinear target points, we simply need to find any rotation aligning the x-axis to the
first target point **b** 1 and then compose it with **u** . If the reference points were collinear instead, we
can swap the reference and target points in the above approach and invert the rotation afterwards. In
practice, we would choose the more degenerate (i.e. larger dot product magnitude) of the two sets
to treat as collinear.


Examining the solution closer, it can be seen that **u** represents a rotation aligning a weighted combination of the reference points we refer to as the “weighted average” with the x-axis. The weighted
average takes the form of a sum ( _w_ 1 **a** 1 + _w_ 2 **a** 2) or difference ( _w_ 1 **a** 1 _−_ _w_ 2 **a** 2) depending on the
sign of the dot product between target points. This suggests that a more straightforward approach
in practice would be to simply calculate the normalized weighted average of the reference points
and align it with **b** 1 directly. This generalizes to the case when the reference points are collinear
similarly to before. If the weighted average is zero, then any rotation is optimal.


C ADDITIONAL STEREOGRAPHIC SOLUTION DETAILS


C.1 RECOVERING R


The solution **U** obtained precisely satisfies the relation in Eq. (34). However, using the maps laid
out in Eqs. (35) and (36) directly will lead to a rotation **RU** that is not necessarily equivalent to
the desired **R** in Eq. (1). This is because our choice of **p** _[∗]_ and choice of isomorphism between
quaternions and special unitary matrices can each add an implicit orthogonal transformation in their
map. Since their combined transformation **Ψ** and its inverse are applied before and after estimation
respectively, the relationship between **U** and **R** is characterized by the conjugate transformation:


**R** = **Ψ** _[T]_ **RUΨ** (44)


For our definitions, we find that **Ψ** is simply a 90 degree rotation about the y-axis. When applied
directly to the resulting **q** from the algorithm, the transformed quaternion is given as:

**q** _[∗]_ = _wq −_ _zqi_ + _yqj_ + _xqk_ (45)

which is just a permutation/negation of the elements of **q** . We can verify that mapping **q** _[∗]_ to **R** via
Eq. (36) indeed gives us the true optimal solution to the problem.


C.2 GENERAL STEREOGRAPHIC CONSTRAINT


The generalized constraint between complex rays [ _z_ 1 _, z_ 2] _[T]_ and [ _p_ 1 _, p_ 2] _[T]_ where _z_ 1 = _x_ 1 + _y_ 1 _i_,
_z_ 2 = _x_ 2 + _y_ 2 _i_, _p_ 1 = _m_ 1 + _n_ 1 _i_, and _p_ 2 = _m_ 2 + _n_ 2 _i_ is given by:


4 _wi_
_wi_ _[′]_ [=]
( _|z_ 1 _|_ [2] + _|z_ 2 _|_ [2] )( _|p_ 1 _|_ [2] + _|p_ 2 _|_ [2] )
**A** _i_ **u** = [ _−z_ 1 _p_ 2 _−z_ 2 _p_ 2 _p_ 1 _z_ 2 _−p_ 1 _z_ 1] **u** = 0


24


for complex inputs and below for real inputs:


        - _m_ 2 _x_ 1 _−_ _m_ 1 _x_ 2 + _n_ 1 _y_ 2 _−_ _n_ 2 _y_ 1 _−m_ 2 _y_ 1 _−_ _m_ 1 _y_ 2 _−_ _n_ 2 _x_ 1 _−_ _n_ 1 _x_ 2�
**D** _i,_ 0 = _m_ 2 _y_ 1 _−_ _m_ 1 _y_ 2 + _n_ 2 _x_ 1 _−_ _n_ 1 _x_ 2 _m_ 2 _x_ 1 + _m_ 1 _x_ 2 _−_ _n_ 1 _y_ 2 _−_ _n_ 2 _y_ 1


   - _m_ 1 _x_ 1 + _m_ 2 _x_ 2 _−_ _n_ 1 _y_ 1 _−_ _n_ 2 _y_ 2 _m_ 1 _y_ 1 _−_ _m_ 2 _y_ 2 + _n_ 1 _x_ 1 _−_ _n_ 2 _x_ 2
**D** _i,_ 1 = _m_ 1 _y_ 1 + _m_ 2 _y_ 2 + _n_ 1 _x_ 1 + _n_ 2 _x_ 2 _m_ 2 _x_ 2 _−_ _m_ 1 _x_ 1 + _n_ 1 _y_ 1 _−_ _n_ 2 _y_ 2


**D** _i_ **q** = [ **D** _i,_ 0 **D** _i,_ 1] **q** = 0


We can verify that with _z_ 2 = 1 and _p_ 2 = 1, we obtain the original results in Eq. (9) and Eq. (11).
Furthermore, we can use _z_ 2 = 0 and _p_ 2 = 0 to calculate results involving the projective point
at infinity. Thus, there are no singularities using the general constraint. From this, we can derive
similar formulas and algorithms for the one and two point cases as those proposed earlier.


Similarly, the following is the general constraint for estimating a M¨obius transformation from stereographic inputs:

**A** _[′]_ _i_ **[m]** [ = [] _[−][z]_ [1] _[p]_ [2] _−z_ 2 _p_ 2 _p_ 1 _z_ 1 _p_ 1 _z_ 2] **m** = 0


D ROTATIONS OF EXACT ALIGNMENT


The equations in this section are derived from the constraint in Eq. (18) for 3D points. However, we
can easily derive similar equations for stereographic points using Eq. (11).


D.1 ONE-POINT CASE


Finding a rotation that aligns two unit vectors (i.e. **b** = **Ra** ) is a special case of Wahba’s problem
where _n_ = 1. Since aligning a pair of points constrains two out of three rotational degrees of
freedom ( **D** _i_ and **Q** _i_ have rank 2), there are infinite solutions in this case. The rotation whose axis
is the cross product of the points is often chosen for geometric simplicity and can be calculated
efficiently as:


_s_ = ~~�~~ 2(1 + **a** _·_ **b** )


_[s]_ **[a]** _[ ×]_ **[ b]**

2 _[,]_ _s_


**q** = ( _[s]_


) (46)
_s_


Instead, we may choose another convention where we constrain an element of the quaternion to be
0. Since the points can be perfectly aligned, **q** _[T]_ **G** _S_ **q** = 0, so **q** _∈_ _Null_ ( **Q** _i_ ). Leveraging this fact,
we can simply take two linearly independent rows from **Q** _i_ and set them to 0 explicitly, imposing a
rank 2 constraint. Given the homogeneous nature of this system, we can disregard the weight and
determine the rotation using straightforward linear algebra techniques. Each row below is a member
of the kernel that has a quaternion element equal to 0 (note only two rows are linearly independent):

 0 _x_ + _m_ _y_ + _n_ _z_ + _p_ 
 _x_ + _m_ 0 _z −_ _p_ _n −_ _y_ 






0 _x_ + _m_ _y_ + _n_ _z_ + _p_
_x_ + _m_ 0 _z −_ _p_ _n −_ _y_
_y_ + _n_ _p −_ _z_ 0 _x −_ _m_
_z_ + _p_ _y −_ _n_ _m −_ _x_ 0





_∈_ _ker_ ( **Q** _i_ ) (47)



Normalizing any nonzero row of Eq. (47) gives an optimal rotation. Compared to Eq. (46), this
approach has several advantages. First, the rotation is simpler to construct. Second, one of its
elements is guaranteed to be 0, so composing rotations and rotating points requires fewer operations
and memory accesses. This is particularly true for the first row of Eq. (47) as it represents a 180
degree rotation whose action on a point can be more efficiently computed as a reflection about an
axis. Finally, Eq. (46) has a singularity when the cross product vanishes. Although each row of
Eq. (47) has its own singular region, it is straightforward to systematically select another row that is
well-defined in that region.


D.2 NOISELESS TWO-POINT CASE


With two independent sets of correspondences, we are able to fully constrain the rotation to a unique
one. If we assume that the two sets can be aligned perfectly, then we can recover an optimal rotation


25


from the intersection of the constraint kernels. Two independent rows of Eq. (47) can be basis
vectors for the kernel of **Q** 1. We can determine the optimal rotation by finding the member of
_ker_ ( **Q** 1) (represented as a linear combination of basis vectors) that is orthogonal to an independent
row of **Q** 2. For example, with the last two rows of Eq. (47) as a basis of **Q** 1 and the first row of **Q** 2,
we can solve for the linear combination weights _a, b_ (note scale is arbitary):

     


0
_x_ 2 _−_ _m_ 2
_y_ 2 _−_ _n_ 2
_z_ 2 _−_ _p_ 2




















_z_ 1 + _p_ 1
_y_ 1 _−_ _n_ 1
_m_ 1 _−_ _x_ 1
0





_y_ 1 + _n_ 1
_p_ 1 _−_ _z_ 1
0
_x_ 1 _−_ _m_ 1


) = 0


 + _b_





 _·_ ( _a_


_a_ = ( _x_ 2 _−_ _m_ 2)( _z_ 1 _−_ _p_ 1) + ( _z_ 2 _−_ _p_ 2)( _m_ 1 _−_ _x_ 1)
_b_ = ( _x_ 2 _−_ _m_ 2)( _y_ 1 _−_ _n_ 1) + ( _y_ 2 _−_ _n_ 2)( _m_ 1 _−_ _x_ 1)

Substituting _a_ and _b_ back into the linear combination and dividing by _m_ 1 _−_ _x_ 1 gives the result from
Eq. (20): This result is equivalent to the simple estimators found in Markley (1999); Choukroun
(2009). However, an issue with this approach is that the singular region of this estimator is not
simple, and the equation fails to produce a valid rotation under several conditions (see Peng and
Choukroun (2024)). Rather than checking each condition with a threshold or applying sequential
rotations to avoid these cases like other kernel methods, we can more systematically select the three
vectors in our computation to guarantee a valid result.


In general, we observe that for a point pair, either **a** + **b** or **a** _−_ **b** will have at least one significantly
nonzero element. We can select the two rows from Eq. (47) corresponding to a nonzero element
from these vectors for the first point pair to ensure linearly independent kernel vectors. We then
choose one of the two rows of **Q** 2 corresponding to a nonzero element of **a** + **b** or **a** _−_ **b** for the
second point pair to solve for the rotation. For instance, if _x_ 1 + _m_ 1 = 0 and _y_ 2 + _n_ 2 = 0, we can
choose the first two rows of Eq. (47) and the last row of **Q** 2 to produce another equation for the
rotation:

**k** 1 = [ _p_ 1 _−_ _z_ 1 _−y_ 1 _−_ _n_ 1 _x_ 1 + _m_ 1] _[T]_

**k** 2 = [ _z_ 1 + _p_ 1 _y_ 1 _−_ _n_ 1 _m_ 1 _−_ _x_ 1] _[T]_

**k** 3 = [ _p_ 2 _−_ _z_ 2 _−y_ 2 _−_ _n_ 2 _x_ 2 + _m_ 2] _[T]_


  - **k** 1 _×_ **k** 3
**q** ˜ =
**k** 2 _·_ **k** 3


(48)


Though the dot and cross products are in different indices from before, the formulation is equally
simple to compute. We select the nonzero elements by largest magnitude for robustness. At least
one of the two rows we select from **Q** 2 will yield a valid rotation for **a** 1 _×_ **a** 2 = 0. Otherwise, the
rotation is any kernel vector of **Q** 1. We verify row validity by checking if either coefficient _a_ or
_b_ for the relevant constraints is nonzero. Those coefficients are always reused in the final rotation
calculation (e.g. _a_ and _b_ are the second and first elements respectively in Eq. (48)). This process
therefore covers the whole domain and only requires a handful of operations and comparisons even
in the worst case.


E BACKPROPAGATION DERIVATIVES


For a simple complex square matrix **G**, the derivative of an eigenvector **v** of **G** with respect to the
elements of **G** can be computed as Magnus (1985):

_d_ **v** = ( _λ_ **I** _−_ **G** ) [+] ( **I** _−_ **[vv]** _[H]_

**v** _[H]_ **v** [)(] _[d]_ **[G]** [)] **[v]**


where _λ_ is the eigenvalue corresponding to **v**, **I** is the identity matrix, and [+] denotes the MoorePenrose pseudoinverse. Typically, **v** _[H]_ **v** = 1 by convention for most eigenvector solvers. In our
original problem (Eq. (14)), **G** _M_ is Hermitian as opposed to a general matrix, so the elements
of Θ are repeated in the matrix through conjugation. Using complex differentiation conventions
consistent with many deep learning frameworks, the loss derivative can be written as:

_dL_ �� _d_ **v**          -          - _dL_ _d_ **v** ��
= [1] _,_ _[d][L]_ +
_d_ ( **G** _M_ ) _i,j_ 2 _d_ **G** _i,j_ _d_ **v** _d_ **v** _[,]_ _d_ **G** _j,i_


��


2


�� _d_ **v**
_,_ _[d][L]_
_d_ **G** _i,j_ _d_ **v**


_d_ **v**


- - _dL_ _d_ **v**
+
_d_ **v** _[,]_ _d_ **G** _j,i_


26


(a) Gram-Schmidt (b) 2-vec


Figure 4: Density plot of loss gradient ratios for Gram-Schmidt and 2-vec. The x-axis represents
the loss _L_, and the y-axis shows the ratio of loss gradient magnitudes _∥∇_ **b** _xL∥/∥∇_ **b** _y_ _L∥_ for the
predicted rotation axes **b** _x_ and **b** _y_ . See Appendix F for details. 2-vec exhibits noticeably lower
variance, suggesting more stable gradients during learning.


where _⟨·, ·⟩_ denotes the complex inner product and _L_ is the scalar loss. _dd_ Θ _L_ [can] [be] [extracted] [from]
the upper triangular portion of _d_ **G** _dLM_ [(after reshaping to 4 x 4), multiplying by 2 for the off-diagonal]
parameters to include the lower portion contribution. This method avoids the need for the other
eigenvectors or eigenvalues of **G** _M_ that weren’t used in the forward pass.


For QuadMobiusSVD (Eq. (25)), the backpropagation must go through the SVD operation **M** =
**UΣV** _[H]_ . It is well known that the nearest unitary matrix corresponds to the unitary component **Q**
of the polar decomposition of **M** = **QP**, where **P** is a positive semidefinite and Hermitian matrix
Keller (1975). Thus, instead of backpropagating through the SVD components individually, we
can backpropagate through **Q** in a more direct manner. Appendix B.3.2 outlines the details of the
derivative of **Q** with respect to the elements of **M** . Given the well-known relationships between the
polar decomposition and SVD ( **Q** = **UV** _[H]_, **P** = **VΣV** _[H]_ ), we can reuse the SVD elements from
the forward pass to calculate the derivative more simply as:


**S** = _diag_ ( **Σ** ) _⊕_ _diag_ ( **Σ** )


_H_ _H_ _H_

   - **U** ( _d_ **M** ) **V** _−_ **V** ( _d_ **M** ) **U**
_d_ **Q** = **U**
**S**


**V** _[H]_


where _⊕_ denotes an outer sum operation, and the division is Hadamard division (element-wise).
From this equation, the numerical complex derivative can be expressed as follows (note the indices,
**F** is 2 x 2 x 2 x 2):

**F** _j,m,l,k_ = **U** _j,k_ ( **V** _[H]_ ) _l,m_


_dL_ - - **F** _Hj,m_
= **U**
_d_ **M** _j,m_ **S**


**V** _[H]_ _,_ _[d][L]_

_d_ **Q**


  - _dL_  - **F** _j,m_

**[U]**
_F_ _[−]_ _d_ **Q** _[,]_ **S**


**V** _[H]_ [�]


_F_


where _⟨·, ·⟩F_ denotes the complex Frobenius inner product.


The remaining operations in the maps are algebraically straightforward to differentiate through. We
observe that the previous formulas compute the same gradients as PyTorch’s automatic differentiation through complex functions `torch.linalg.eigh` and `torch.linalg.svd` but in a more
streamlined manner.


F THEORETICAL INVESTIGATIONS OF REPRESENTATIONS


**2-vec** The core idea behind 2-vec lies in leveraging a more optimal projection (in the sense of
Wahba’s problem) than Gram-Schmidt to improve learning performance without increasing computational cost or dimensionality. To theoretically support this, we replicate the gradient analysis


27


Figure 5: Visualization of loss ratio between Gram-Schmidt (GS) representation and 2-vec representations for all reported figures in this paper (accuracy converted to 1-Acc to maintain directionality).
Gram-Schmidt performs around 10% worse on average than 2-vec with some experiments showing
a large discrepancy. 2-vec performed better on 41/52 reported metrics.


Figure 6: Plot of mean loss (Chordal L2) against dropout rate of map representations. Θ and **M**
denote whether dropout was applied to map inputs or intermediate representation for QuadMobius.


experiment from Geist et al. (2024) which evaluates how learning signals propagate through the
representations. We first generate a thousand random 6D vectors, each with components sampled
uniformly from [-2, 2]. Each vector is split into two 3D components, **b** _x_ and **b** _y_, representing
predicted target _x, y_ coordinate axes. These are then mapped to a rotation matrix using both the
Gram-Schmidt and 2-vec methods. For each mapping, we compute the Frobenius norm loss _L_ between the resulting rotation and the identity matrix. We then calculate the gradient magnitudes of _L_
with respect to **b** _x_ and **b** _y_ and analyze their ratio. The results are plotted in Fig. 4. We can see that
the gradient ratios for 2-vec are more tightly concentrated around 1, indicating a relatively balanced
gradient flow between the two vectors. In contrast, the Gram-Schmidt method exhibits a wider
distribution with significant skew, often yielding ratios in the range of 10–100 which highlights its
disproportionate focus on **b** _x_ . These results support the hypothesis that 2-vec facilitates more stable
gradients for optimization.


**QuadMobius** In our experiments, QuadMobius has consistently shown strong performance as a
learning representation. To better understand why, we conduct two experiments to probe its behavior. We begin by generating one thousand realistic map inputs Θ for each representation using
trained models from a synthetic Wahba’s problem (trial #15 in Appendix G.2.2). All models are
fed the same noiseless inputs on which they perform equivalently for fair comparison. In the first
experiment, we test how resilient each map is to corrupted inputs by applying dropout. Fig. 6 shows


28


Figure 7: Distribution plot of loss gradient magnitudes against loss _L_ (Chordal L2). The left shows
the gradient with respect to the map inputs Θ, while the right shows the gradient with respect to the
M¨obius transformation **M** estimated from eigendecomposition in QuadMobius.


the results of applying increasing dropout probability to Θ on mean loss. For QuadMobius, we
also test applying dropout to its intermediate M¨obius transformation **M** instead (real and imaginary
parts treated independently). While we might expect the sensitivity to dropout to decrease with dimensionality, this is not necessarily the case as seen with QCQP. Notably, QuadMobius appears to
be the most resilient to dropout on Θ, but is also the most sensitive when applied to **M** . For the
second experiment, we replace 10% of the model inputs with outlier points from another rotation,
simulating out-of-domain inference. Fig. 7 plots the distribution of loss gradient magnitudes against
loss. Gradients with respect to Θ are similar across all maps, consistent with their equivalent performance on the task. In contrast, gradients with respect to **M** in QuadMobius are both significantly
larger and more tightly concentrated, following a square root trend. Together, these two experiments
suggest that QuadMobius’s eigendecomposition step enables the learning of a stable intermediate
representation that is buffered against poor inputs, while its subsequent _SU_ (2) projection ensures
predictable, high-fidelity gradient flow, leading to its strong empirical performance.


**SU** ( **2** ) A natural question is whether we can just directly predict an _SU_ (2) representation and
project it onto the manifold. This approach is simpler than QuadMobius and still provides an
overparameterized representation (8D). However, like quaternions, _SU_ (2) suffers from the issue
of double cover. Both M¨obius transformation predictions **M** and _−_ **M** map to the same 3D rotation,
introducing ambiguity in learning. Furthermore, one might hope the rows of **M** offer two different estimates of a quaternion rotation (similar to theoretical arguments of information averaging in
SVD and QCQP). However, in _SU_ (2) the rows encode the same information, so independence is
not enforced during learning. Empirically, _SU_ (2) prediction performed much worse in synthetic
experiments than QuadMobius (often close to quaternion) and was thus not included in results.


To further validate the QuadMobius approach, we conducted a toy ablation experiment in Table 3.
We took 10k random map inputs and mapped them to quaternions. We then calculate the squared
quaternion loss (accounting for sign) against a set of random ground truth quaternions and compare the loss gradient magnitudes of the inputs for the different map variants. The variants include
SVD projection only (8D -¿ _SU_ (2)), Eigendecomposition only (16D -¿ M¨obius transformation M,
taking the first row of M as a quaternion with and without normalization), and QuadMobius. The
percentiles of the gradient distributions and their subsequent percentile ranges are shown in the table
below. The QuadMobius approach yields a significantly tighter distribution and a lower amount of
large outlying values than the other isolated components, suggesting that it provides more stable
gradients for learning with both eigendecomposition and projection.


29


**Method** **10%** **25%** **50%** **75%** **90%** **25-75%** **10-90%**


Projection 2.04e-5 2.66e-5 3.23e-5 3.65e-5 4.02e-5 9.90e-6 1.98e-5
Eig. (no norm) 2.06e-5 2.87e-5 3.41e-5 3.88e-5 4.08e-5 1.00e-5 2.02e-5
Eig. (norm) 1.43e-5 2.07e-5 2.51e-5 3.01e-5 3.20e-5 9.41e-6 1.78e-5
QuadMobius 1.46e-5 1.79e-5 2.20e-5 2.49e-5 2.64e-5 **6.98e-6** **1.17e-5**


Table 3: Toy ablation experiment showing gradient magnitude distributions for isolated components
of QuadMobius algorithm. Bold indicates lowest for spread quantities.


G EXPERIMENTS


G.1 EXPERIMENT SETTINGS AND DETAILS


These are the specific experiment settings used to obtain the results in our learning experiments.


**ModelNet10-SO3** ADAM optimizer, learning rate 5e-4, NVIDIA L1 GPU, batch size 100,
Chordal L2 loss, 300/400/800 epochs respectively for chair/sofa/toilet to train for roughly equal iterations given dataset size differences. Architecture is ShuffleNetV2-1.5 backbone Ma et al. (2018)
(used for its quick training) pretrained on ImageNet weights followed by two fully connected layers featuring ReLU activation and dropout applied before the layers with probability 0.4 and 0.25
respectively. Models saved by best average rotation error.


**Inverse** **Kinematics** Original author source code and settings Zhou et al. (2019) were utilized.
Trained on NVIDIA L1 GPU for 2 million iterations. Epoch with lowest median rotation error was
chosen for results.


**Camera** **Pose** **Estimation** Training code and settings obtained from Chen et al. (2022). Model
initialized from pretrained GoogleNet weights recommended by original paper. Used NVIDIA L1
GPU and beta values 500/100/1500 for King’s College/Shop Facade/Old Hospital. Trained for 1200
epochs with batch size 75. Models saved every 5 epochs, and models from last 300 epoch were used
for testing (batch size 1 in testing). Epoch with lowest median rotation error was chosen for results.


G.2 ADDITIONAL EXPERIMENTS


G.2.1 WAHBA’S PROBLEM


_n_ = 3 _n_ = 100
Algorithm _ϵ_ = 1 _e_ _[−]_ [5] _ϵ_ = 0 _._ 1 Timings _ϵ_ = 1 _e_ _[−]_ [5] _ϵ_ = 0 _._ 1 Timings


Q-method Davenport (1968) 7.4676e-4 7.4868 3.583 1.2487e-4 1.2551 5.375
QUEST Shuster and Oh (1981) 7.4676e-4 7.4868 0.250 1.2487e-4 1.2551 1.875
ESOQ2 Mortari (1997) 7.4694e-4 7.4869 0.375 1.2487e-4 1.2551 2.000
FLAE Wu et al. (2018) 7.4676e-4 7.4868 0.333 1.2487e-4 1.2551 1.875
OLAE Mortari et al. (2007) 7.7118e-4 7.8639 0.208 1.3120e-4 1.5952 2.167
Ours ( **G** _P_, Eq. (12)) 7.4676e-4 7.4868 4.084 1.2487e-4 1.2551 9.917
Ours ( **G** _S_, Eq. (19)) 7.4676e-4 7.4868 3.625 1.2487e-4 1.2551 6.500
Ours ( **G** _M_, Eq. (14)) 1.2614e-3 12.608 0.917 3.5870e-4 3.7782 41.875


Table 4: Results of various Wahba’s Problem solvers against varying noise levels with _n_ = _{_ 3 _,_ 100 _}_ .
Accuracy values reported are median _θerr_, and timing values are median runtimes in microseconds.
Timings taken with _ϵnoise_ =0.1. See Section 5.1 for more info.


G.2.2 LEARNING WAHBA’S PROBLEM


To evaluate our rotation representations more robustly across various conditions, we replicate the
synthetic learning experiments from Peretroukhin et al. (2020); Levinson et al. (2020); Zhou et al.
(2019), using a fully-connected neural network from Peretroukhin et al. (2020) to learn the solution


30


_√_
Algorithm x _÷_ 5 _[th]_ 50 _[th]_ 95 _[th]_


QUEST (Shuster and Oh, 1981) 89 / 99 **1** / **1** **3** / **3** 3.3082 / 3.4115 9.1727 / 9.3970 27.0520 / 27.1371
Fast 2 Vec (Markley, 2002) 72 / 78 3 / 3 4 / 4 3.3082 / 3.4115 9.1727 / 9.3970 27.0520 / 27.1371
SUPER (Ours) **29** / **74** 3 / 2 **3** / **3** 3.3082 / 3.4115 9.1727 / 9.3970 27.0520 / 27.1371


Table 5: Operation counts and _θerr_ percentiles ( _ϵnoise_ = 0 _._ 1) for two-point Wahba’s problem
solvers. Values given for unweighted/weighted algorithms without edge case handling. Bold indicates best.


# _n_ LR Loss Dom Euler Quat GS QCQP SVD 2-vec QMAlg QMSVD


1 3 1e-4 L2 R 9.009/0 8.964/1 1.761/0 1.676/141 **1.641** / 1.701/1 1.658/51 1.689/110
2 3 1e-4 L2 C 119.364/0 13.632/0 5.768/0 4.237/1 4.264/1 5.781/0 3.823/109 **3.761** / 3 3 5e-4 L2 R 12.154/0 9.618/0 1.583/5 1.518/143 **1.491** / 1.560/0 1.501/217 1.527/53
4 3 5e-4 L2 C 119.403/0 12.238/0 4.016/0 3.586/2 3.735/6 3.917/0 3.447/ **3.408** /241
5 3 1e-3 L2 R 14.693/0 9.159/0 1.575/1 1.497/170 1.509/245 1.578/2 **1.486** /87 1.499/ 6 3 1e-3 L2 C 119.397/0 11.212/0 3.290/24 3.289/190 3.253/ 3.269/0 3.250/110 **3.232** /292
7 3 1e-4 L1 R 8.063/0 4.120/0 1.603/0 1.445/135 **1.421** / 1.570/2 1.469/164 1.459/77
8 3 1e-4 L1 C 119.388/0 9.812/0 4.734/0 3.259/0 3.238/1 4.663/0 2.835/492 **2.786** / 9 3 5e-4 L1 R 8.687/0 4.355/0 1.459/0 1.315/175 1.322/279 1.416/0 **1.303** / 1.306/128
10 3 5e-4 L1 C 119.334/0 7.500/0 3.290/0 2.760/3 2.857/3 3.113/0 **2.750** / 2.807/73
11 3 1e-3 L1 R 10.833/0 4.436/0 1.434/0 1.312/53 1.301/ 1.427/0 1.317/337 **1.291** /272
12 3 1e-3 L1 C 119.483/0 6.930/0 2.916/0 2.475/92 **2.447** /251 2.874/0 2.478/211 2.472/ 13 100 1e-4 L2 R 3.784/0 3.277/0 0.569/0 0.253/138 **0.243** / 0.313/0 0.255/169 0.251/304
14 100 1e-4 L2 C 48.175/0 4.988/0 1.400/0 0.638/254 0.637/136 0.850/0 **0.625** /281 0.634/ 15 100 5e-4 L2 R 5.395/0 3.712/0 0.547/0 0.249/121 0.247/175 0.303/0 0.247/ **0.242** /336
16 100 5e-4 L2 C 119.370/0 5.009/0 1.586/0 **0.831** / 0.866/223 0.940/0 0.866/66 0.848/29
17 100 1e-3 L2 R 6.608/0 3.269/0 0.537/0 **0.243** /292 0.272/112 0.297/0 0.261/ 0.253/297
18 100 1e-3 L2 C 118.381/0 5.056/0 1.480/0 0.845/121 0.836/ 0.887/0 0.859/71 **0.826** /309
19 100 1e-4 L1 R 2.249/0 1.794/0 0.356/0 0.269/293 **0.261** / 0.332/0 0.264/130 0.265/250
20 100 1e-4 L1 C 109.217/0 3.209/0 0.927/0 **0.665** /268 0.667/ 0.889/0 0.669/196 0.669/67
21 100 5e-4 L1 R 2.666/0 1.055/0 0.355/0 0.275/83 0.284/339 0.316/1 0.289/209 **0.272** / 22 100 5e-4 L1 C 119.299/0 1.954/0 0.938/0 0.883/ 0.877/101 0.956/0 **0.873** /73 0.878/46
23 100 1e-3 L1 R 3.867/0 1.384/0 0.366/0 0.280/167 0.280/316 0.331/0 **0.277** / 0.291/171
24 100 1e-3 L1 C 83.623/0 2.184/0 0.952/0 0.830/ 0.835/61 0.919/0 **0.826** /366 0.849/107


Table 6: Trial results for learning Wahba’s problem with different rotation representations. _n_ is
number of points, LR is learning rate, Loss is type of chordal loss function, Dom is the domain,
specifying whether the network is real-valued or complex-valued. Results are shown as _θerr_ /Ldr.
pairs where _θerr_ is average rotation error on validation set, and Ldr. is the number of epochs where
that representation was a leader, i.e. had the lowest _θerr_ overall as of that epoch. Bold indicates best
value, underline indicates second best.


to Wahba’s problem. Problem points and rotations are generated according to same procedure described in Section 5.1. Each epoch, we dynamically generate 25,600 training samples and validate
on a fixed set of the same size ( _ϵnoise_ = 0 _._ 01 added to all samples). The models are trained for
1000 epochs with ADAM optimizer on an NVIDIA T4 GPU. In addition to Chordal L2, we also
define the loss function Chordal L1 analogously as the sum of absolute differences between the
elements of **R** _pred_ and **R** _gt_ . Finally, given our complex representations, we also evaluate training
complex-valued networks Liao (2023); Barrachina et al. (2023) of equivalent size for the task with
stereographic complex inputs (Eq. (30)). For real-valued representations, we take the real part of the
model output in this case.


As expected, the compact representations (Euler, Quat) performed relatively poorly. Overall, the
best performers (QCQP, SVD, QuadMobiusAlg, QuadMobiusSVD) were all quite competitive with
each other, having similar results and convergence rates. However, the QuadMobius representations
together demonstrated an edge, leading most of the epochs and having the lowest error in majority
of trials. Although mathematically equivalent, the two approaches produced different results with
neither approach consistently outperforming the other. On the other hand, 2-vec outperformed the


31


other non-eigendecomposition representations (including Gram-Schmidt), beating them on most
trials, at times by a large margin. Although significant differences for the complex cases were
not observed among representations, some of the complex-valued trials featured the highest leader
counts overall by our representations (e.g. trial #2, trial #10). The leader count gives a sense of
the convergence/dominance of the learning as well how cherry-picked the results may be based on
number of training epochs. See Fig. 8 for sample training/validation curves which illustrate the
advantage of noncompact representations and the competitiveness of our approaches.


G.2.3 REPRESENTATION TIMINGS


Euler Quat GS QCQP SVD 2-vec QMAlg QMSVD


Training 0.2123 0.0691 0.4903 0.5223 0.4904 0.4447 1.2231 1.6247
Inference 0.0401 0.0056 0.1050 0.2435 0.2737 0.0803 0.4298 0.6221


Table 7: Comparison of timings of different representations run with batch size 128. Measured on
Apple M1 Silicon CPU. Values reported are median measurements of 10000 runs in milleseconds.
Training includes forward and backward passes (PyTorch train mode), and Inference includes only
forward pass (PyTorch eval mode).


Table 7 shows the compute timings of the representations. 2-vec has notably fast inference timings. QuadMobius representations are slower than others as they involve complex arithmetic and
more compute steps overall. However, training time differences were observed to be negligible between them and QCQP/SVD as bottlenecks are typically present elsewhere in the pipeline (e.g. data
loading, network compute).


32


33
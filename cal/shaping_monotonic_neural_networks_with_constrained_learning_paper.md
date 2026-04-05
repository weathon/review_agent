# SHAPING MONOTONIC NEURAL NETWORKS WITH CON## STRAINED LEARNING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The monotonicity of outputs of a neural network with respect to a subset of inputs
is a desirable property that provides an important tool to explore the interpretability,
fairness, and generalizability of the designed models, and underlies many applications in finance, physics, engineering, and many other domains. In this paper, we
propose a novel, flexible, and adaptive learning framework to induce monotonicity of neural networks with general architectures. The monotonicity serves as a
constraint during the model training, which motivates us to develop a primal-dual
learning algorithm to train the model. In particular, our framework provides an
interface to trade off between probability of monotonicity satisfaction and overall
prediction performance by introducing a chance constraint, making it more flexible
for different application scenarios. The proposed algorithm needs only small extra
computations to continuously and adaptively enforce the monotonicity until the
constraint is satisfied. Compared to the existing methods for building monotonicity,
our framework does not impose any constraints on the neural network architectures
and needs no pre-processing such as tuning of the regularization. The numerical experiments in various practical tasks show that our method can achieve competitive
performance over state-of-the-art methods.


1 INTRODUCTION


Deep neural networks have significantly promoted the development of many real-world domains
such as finance, physics, engineering (He et al., 2016; LeCun et al., 2015; Vaswani et al., 2017),
and drastically reshaped various fields like computer vision (You et al., 2017; Minaee et al., 2021),
natural language processing (Otter et al., 2020), autonomous driving (Kiran et al., 2021). Shaping
the neural network models with prior knowledge, the monotonicity dependence for example, is a
desideratum that can help improve the interpretability and generalizability of the models (Feelders,
2000; Rieger et al., 2020; Liu et al., 2020). The incorporation of monotonicity with respect to some
parts of inputs can also help the trained models produce legal, reliable, and safe results. For instance,
in the admission decisions on applicants using an machine learning model, the machine learning
model is expected to select the student with higher scores when all other factors are equal (Liu et al.,
2020). Similar monotonicity requirements are prevalent in engineering. For example, when designing
controllers for safe-critical systems, e.g., autonomous vehicles (Chen et al., 2024; Bojarski et al.,
2016) and power systems (Cui et al., 2023a;b), the stability of the physical system often relies on
monotonicity of the controllers. Enforcing monotonicity in the controller models is important to
ensure the safety and reliability of such physical systems.


Incorporating the monotonicity constraints has been widely studied in traditional machine learning
over the past decades (Archer & Wang, 1993; Potharst & Feelders, 2002; Doumpos & Zopounidis,
2009; Chen et al., 2015; Bartley et al., 2019), and recent research has extended the attention to
inducing the monotonicity constraints into complex neural network models (Lang, 2005; You et al.,
2017; Milani Fard et al., 2016; Liu et al., 2020). Existing methodologies can generally be categorized
into two groups:


_1) Monotonicity by constructed architectures._ This group enforces the monotonicity by tailoring the
model with specific architectures (Archer & Wang, 1993; Daniels & Velikova, 2010; Kim & Lee,
2024; Milani Fard et al., 2016; You et al., 2017). This approach restricts the hypothesis space of
the models, making the implementation challenging and potentially incompatible with the efficient
architectures that can yield enhanced performance like residual connections (He et al., 2016).


1


_2) Monotonicity by regularization._ Another group introduces monotonicity by modifying the training
process of general neural networks using heuristical regularization (Gupta et al., 2019a; Liu et al.,
2020; Sivaraman et al., 2020). Although this approach offers more flexibility and the monotonicity
can be guaranteed through certification (Liu et al., 2020; Sivaraman et al., 2020), it requires empirical
tuning of the regularization during training, and the certification process may involve significant
computational overhead for large neural networks.


1.1 OUR CONTRIBUTION


In this paper, we draw inspirations from conventional constrained optimization and propose a novel,
flexible, and adaptive learning framework to induce monotonicity of neural networks with general
architectures. Compared to existing methods, our framework exhibits the following three key features.


 - **High** **Flexibility:** Our framework allows to trade off between probability of monotonicity
satisfication and overall prediction performance through a chance constraint, according to the
desirability of monotonicity or users’ preference;


 - **Advanced** **Capability:** Our framework does not impose any constraints on neural network
architectures, thereby inherits many advantages of state-of-the-art deep neural networks, e.g.,
strong expressivity, easy-to-train properties;


 - **Strong Adaptability:** Our framework does not require empirical tuning of the regularization
terms prior to training. Instead, it allows adaptive learning without case-by-case pre-processing,
making it immediately ready when applying it to a new scenario.


Finally, we evaluate the proposed frameworks on several practical tasks including classification and
regression in datasets and controller design in physical systems, highlighting the above features and
its competitive performance over state-of-the-art methods in various domains.


1.2 RELATED WORK


We further review some concrete examples in each of the aforementioned categories.


**Monotonicity** **by** **constructed** **architectures:** The examples include the non-negative approach
(Archer & Wang, 1993), which constrains the weights of monotonic features as positive, MinMax Network (Sill, 1997; Daniels & Velikova, 2010), which utilizes linear embedding and maxmin-pooling layers to impose monotonicity, and Deep Lattice Network (DLN) (You et al., 2017;
Milani Fard et al., 2016), which constructs the network with an ensemble of lattice layers. The recent
works manipulate the weights of monotonic features. For example, the Lipschitz Monotonic Network
(LMN) (Nolte et al., 2022) constrains the norm of weights to attain Lipschitz continuity. In (Runje
& Shankaranarayana, 2023), the authors convert updated weights in the monotonic dense layer into
absolute values, while the Scalable Monotonic Neural Network (SMNN) (Kim & Lee, 2024) uses
the exponentiated units to guarantee monotonicity. These methods require additional operations on
the weights and compromise the flexibility, which may result in reduced performance, due to the
restricted hypothesis space.


**Monotonicity by regularization:** Various examples in this group modify the training loss by adding a
regularization term that penalizes the negative gradients or the negative weights (Sill & Abu-Mostafa,
1996; Gupta et al., 2019b). These methods work for arbitrary architectures but offer no guarantees of
monotonicity for the trained models. In (Liu et al., 2020), the authors utilize the piece-wise linear
property of ReLU neural networks to verify the monotonicity of the trained models by transforming
the verification process into a mixed integer linear programming (MILP) problem. They repetitively
increase the regularization strength on negative gradients of the points that are uniformly sampled
from the input domain until the model passes the monotonicity certification. However, the certification
can be computationally expensive for deep neural networks. Another concurrent method (Sivaraman
et al., 2020) utilizes a satisfiability modulo theories (SMT) solver (Barrett & Tinelli, 2018) to find the
counterexamples to the monotonicity definition and include the counterexamples in the training data
with adjustments to their target values to enforce monotonicity. Monotonicity can be guaranteed by
finding the upper and lower envelopes of the model. Similar to (Liu et al., 2020), SMT solver can be
computationally expensive as the size of neural networks grows. Furthermore, both (Liu et al., 2020)
and (Sivaraman et al., 2020) have been shown to only support the ReLU networks but not general
neural networks.


2


2 PRELIMINARIES AND MOTIVATIONS


2.1 MONOTONIC FUNCTIONS


Here, we provide the definition of monotonic mapping functions and its equivalent property, including
the univariate functions and multivariant functions. The equivalent property of monotonicity is
essential for developing our constrained learning algorithm.


**Monotonic Fucntions.** Let _f_ ( _x_ ) be a univariate continuously differentiable function from an input
space _X_ _∈_ R to R, where _X_ = [ _l, u_ ], with _l_ _<_ _u_ . We say that function _f_ ( _x_ ) is monotonically
non-decreasing in _X_ if _∀x, x_ _[′]_ _∈X_ _, x_ _≤_ _x_ _[′]_, the inequality _f_ ( _x_ ) _≤_ _f_ ( _x_ _[′]_ ) is satisfied. Equivalently,
we say that function _f_ ( _x_ ) is monotonically non-decreasing if _∀x ∈X_, we have _[∂f]_ _∂x_ [(] _[x]_ [)] _≥_ 0.


**Partial Monotonic Functions.** Let _f_ ( _**x**_ ) be a multivariate continuous differentiable function from an
input space _**X**_ _∈_ R _[d]_ to R, where _**X**_ := _×_ 1 _≤i≤d_ = _×_ 1 _≤i≤d_ [ _li, ui_ ], with _d_ denoting the dimensions of
the input space and _li_ _< ui_ for all 1 _≤_ _i ≤_ _d_ . Suppose the input vector _**x**_ := [ _x_ 1 _, . . ., xd_ ] _∈_ R _[d]_ can
be partitioned into [ _**x**_ _m,_ _**x**_ _¬m_ ], where _m_ is a subset of [1 _, . . ., d_ ] and _¬m_ is the associate complement.
Similarly, suppose the input space _**X**_ can be partitioned into _**X**_ _m_ and its complementary space _**X**_ _¬m_ .
We say that function _f_ ( _**x**_ ) is monotonically non-decreasing with respect to _**x**_ _m_ if

_f_ ( _**x**_ _m,_ _**x**_ _¬m_ ) _≤_ _f_ ( _**x**_ _[′]_ _m_ _[,]_ _**[ x]**_ _[¬][m]_ [)] _[,]_ _[∀]_ _**[x]**_ _[m]_ _[≤]_ _**[x]**_ _[′]_ _m_ _[,]_ _[∀]_ _**[x]**_ _[m][,]_ _**[ x]**_ _[′]_ _m_ _[∈]_ _**[X]**_ _[ m]_ [and] _**[ x]**_ _[¬][m]_ _[∈]_ _**[X]**_ _[ ¬][m][,]_ (1)

where _≤_ denotes the element-wise inequality for vectors, i.e., _**x**_ _m_ _≤_ _**x**_ _[′]_ _m_ [means] _[ x][i]_ _[≤]_ _[x][′]_ _i_ [for all] _[ i][ ∈]_ _[m]_ [.]
Equivalently, the function _f_ ( _**x**_ ) is monotonically non-decreasing with respect to _**x**_ _m_ if _[∂f]_ _∂x_ [(] _**[x]**_ _i_ [)] _≥_ 0

holds for all _i ∈_ _m_ .


2.2 MOTIVATIONS


Let _f_ _**θ**_ ( _**x**_ ) denote a neural network model with _**θ**_ being the model parameters and _**x**_ being a data
point from the training set _**D**_ . The model _f_ _**θ**_ can be a conventional neural network with commonly
used activation functions (e.g., ReLU, Leaky ReLU, Sigmoid, Tanh, etc). Denote by _J_ ( _f_ _**θ**_ ( _**x**_ )) the
typical training loss of a machine learning problem. The conventional training process of the neural
network can be formally articulated as solving the unconstrained optimization problem:


min _**θ**_ [E] _**[x]**_ _[∼]_ _**[D]**_ _[ J]_ [ (] _[f]_ _**[θ]**_ [(] _**[x]**_ [))] _[,]_ (2)


where the expectation E is with respect to the distribution of _**x**_, and the stochastic gradient descent
(SGD)-based steps are applied to iteratively approach a solution parameter _**θ**_ . When one seeks
to impose monotonically non-decreasing features on the neural network model _f_ _**θ**_, the problem
formulation becomes:


min _**θ**_ [E] _**[x]**_ _[∼]_ _**[D]**_ _[ J]_ [ (] _[f]_ _**[θ]**_ [(] _**[x]**_ [))] (3a)

s.t. _∂f_ _**θ**_ ( _**x**_ ) _≥_ 0 _,_ _∀i ∈_ _m,_ _∀_ _**x**_ _∈_ _**D**_ _._ (3b)

_∂_ _**x**_ _i_


The interpretation of the constrained problem (3) is straightforward from the optimization’s perspective. The constraint (3b) confines the feasible solution set, excluding those solutions of (2) that
may violate the partial monotonicity requirements. However, problem (3) is not well formulated
for directly applying (projected) SGD algorithms to attain a solution due to the existence of constraint (3b). Indeed, it is unclear that how (3b) reflects the constraints on _**θ**_ in general. Therefore, the
literature has mainly developed two branches to handle it: _monotonicity by constructed architectures_
and _monotonicity by regularization_ . The former designs neural networks with special architectures
so that (3b) is satisfied automatically. This could be understood as the constraints on _**θ**_ reflected
by (3b) are embedded into the neural network architecture design. The latter instead moves the
monotonicity constraint (3b) into the cost function (3a) as a regularization term, and the monotonicity
property of neural networks is then promoted through training. Nevertheless, there exist fundamental
limits for both of them. For the former, it requires one to modify conventional neural networks with
special architectures, which often results in poor expressive capability in practice compared to the
conventional neural networks. For the latter, the performance highly depends on the selection and
tuning of the regularization term, and there is no systematical method for it.


3


To overcome these bottlenecks, in the next, we borrow ideas from constrained optimization and
propose a novel, flexible, and adaptive learning framework to enforce the monotonicity of neural networks. Specifically, our framework leverages primal-dual algorithms to continuously and adaptively
enforce monotonicity for general neural networks, does not impose any constraints on the neural
network architectures, and needs no pre-processing such as tuning of the regularization terms.


3 CONSTRAINED LEARNING FOR MONOTONIC NETWORKS


Here, we present our proposed framework for training monotonic neural networks. We transform the
problem (3) into a chance constrained optimization problem, followed by a stochastic primal-dual
learning algorithm to solve it.


3.1 REFORMULATION WITH CHANCE CONSTRAINTS


To account for the statistical distribution of constraint (3b) across the entire dataset, we propose a
reformulation of problem (3) as below:


min E _**x**_ _∼_ _**D**_ _J_ ( _f_ _**θ**_ ( _**x**_ )) (4a)
_**θ**_


where _**t**_ = [ _t_ 1 _, . . ., tm_ ] _∈_ R _[|][m][|]_ . Now, the formulation (6) permits the use of stochastic primal-dual
algorithm to optimize the neural network parameter _**θ**_ thanks to the continuous differentiability of (6b).


4


s.t. Pr _**x**_ _∼_ _**D**_


- _∂f_ _**θ**_ ( _**x**_ ) _≤_ 0� _≤_ _α,_ _∀i ∈_ _m,_ (4b)
_∂_ _**x**_ _i_


where Pr _**x**_ _∼_ _**D**_ is a probability operator with respect to the distribution of data samples in the dataset,
and _α_ _∈_ [0 _,_ 1] is a small number that confines the constraint (3b) satisfaction at least 1 _−_ _α_ . We
note that when _α_ = 0, problem (4) is exactly equivalent to the original problem (3). However, when
running stochastic primal-dual algorithm to solve (4), the probability operator in (4b) prevents the
computation of gradients (i.e., the gradients of (4b) with respect to the parameter _**θ**_ ) that are required
by the existing machine learning algorithms. To resolve this issue, notice that


- 1 0 _−_ _[∂f]_ _**[θ]**_ [(] _**[x]**_ [)]

_∂_ _**x**_ _i_


��
_≤_ _α,_ (5)


Pr _**x**_ _∼_ _**D**_


- _∂f_ _**θ**_ ( _**x**_ ) _≤_ 0 _≤_ _α_ _⇐⇒_ E _**x**_ _∼_ _**D**_
_∂_ _**x**_ _i_


for all _i ∈_ _m_, where 1( _x_ ) is an indicator function that equals 1 if _x ≥_ 0 and 0 otherwise. While the
condition in the right hand side of (5) is still not continuously differentiable, we show in the next
claim a sufficient condition to ensure it yet is continuously differentiable.

**Claim** **1.** For any _**t**_ = [ _t_ 1 _, . . ., tm_ ] _[⊤]_ _∈_ R _[|][m][|]_ with _ti_ _>_ 0 for all _i_ _∈_ _m_, if it holds that


+


E _**x**_ _∼_ _**D**_


��
_**t**_ + **0** _−_ _[∂f]_ _∂_ _**[θ]**_ _**x**_ [(] _m_ _**[x]**_ [)]


_≤_ _α_ _**t**_, where [ _·_ ]+ denotes the projection onto the nonnegative orthant,


��
_≤_ _α, ∀i ∈_ _m_, holds too.


     -      then E _**x**_ _∼_ _**D**_ 1 0 _−_ _[∂f]_ _∂_ _**[θ]**_ _**x**_ [(] _i_ _**[x]**_ [)]


We reason Claim 1 here. Consider some function _g_ ( _x_ ) of random variable _x_ and the chance constraint
Pr( _g_ ( _x_ ) _≥_ 0) = E [1 ( _g_ ( _x_ ))] _≤_ _α_, we notice that 1 ( _g_ ( _x_ )) _≤_ [1 + _g_ ( _x_ ) _/t_ ]+ holds for all _g_ ( _x_ ) and
_t_ _>_ 0. Therefore, for any _t_ _>_ 0 such that E �[1 + _g_ ( _x_ ) _/t_ ]+� _≤_ _α_, the constraint E [1 ( _g_ ( _x_ ))] _≤_ _α_
holds too. Since _t >_ 0, multiplying both sides of E �[1 + _g_ ( _x_ ) _/t_ ]+� _≤_ _α_ by _t_ leads to a equivalent
condition E �[ _t_ + _g_ ( _x_ )]+� _≤_ _αt_ . This leads to Claim 1.


Leveraging Claim 1, by replacing _g_ ( _x_ ) with 0 _−_ _[∂f]_ _**[θ]**_ [(] _**[x]**_ [)]


Leveraging Claim 1, by replacing _g_ ( _x_ ) with 0 _−_ _∂_ _**[θ]**_ _**x**_ _i_ and introducing an auxiliary variable

_ti_ _>_ 0 for each _i_ _∈_ _m_, we then have a inner approximation to the chance constraint (5) as


��
E _ti_ + 0 _−_ _[∂f]_ _**[θ]**_ [(] _**[x]**_ [)]


+


_≤_ _αti,_ _∀i_ _∈_ _m_, which are continuously differentiable. Collecting these


_∂_ _**x**_ _i_


constraints over _i ∈_ _m_ in a vector form, we have:


min E _**x**_ _∼_ _**D**_ _J_ ( _f_ _**θ**_ ( _**x**_ )) (6a)
_**θ**_ _,_ _**t**_


+


s.t. E _**x**_ _∼_ _**D**_


��
_**t**_ + **0** _−_ _[∂f]_ _**[θ]**_ [(] _**[x]**_ [)]

_∂_ _**x**_ _m_


_≤_ _α_ _**t**_ _,_ (6b)


In fact, one can notice that when _α_ = 0, problem (6) exactly returns to the original problem (3).
This can be understood by checking each data point from the dataset and prohibit any data point
from violating the monotonicity requirement, leading to strict monotonicity. On the other hand,
selecting a small positive _α_ represents the case where we allow some of the data points violate the
monotonicity requirement in a low probability manner. This is particular meaningful in scenarios
where monotonicity is acting as a tendency instead of a strict requirement, as in these cases the
compromise to monotonicity of a small portion of data points might win significant improvement of
overall performance.


**Enforce** **monotonicity** **to** **the** **whole** **input** **space** _**X**_ **.** Generally speaking, the dataset _**D**_ is a
subset of the input domain _**X**_ of function _f_ _**θ**_ ( _**x**_ ). To enhance the generalizability of function
_f_ _**θ**_ ( _**x**_ ), it is essential to enforce the monotonicity requirement across the entire input domain _**X**_
(Liu et al., 2020). Let Uni( _**X**_ ) denote the uniform distribution on _**X**_ . In the rest of this paper, we compute the expectation of the chance constraint (6b) over Uni( _**X**_ ) instead of _**D**_, i.e.,
E _**z**_ _∼_ Uni( _**X**_ ) �[ _**t**_ + **0** _−_ _∂f_ _**θ**_ ( _**z**_ ) _/∂_ _**z**_ _m_ ]+� _≤_ _α_ _**t**_, to develop our constrained learning algorithm.


3.2 STOCHASTIC PRIMAL-DUAL LEARNING ALGORITHM


Having the problem formulation (6) in mind, here we develop a stochastic primal-dual learning
algorithm to approach a solution (Eisen et al., 2019). Let _**µ**_ _∈_ R _[|][m][|]_ be the dual variable of the
constraint (6b). Consider the Lagrangian function of problem (6):


We summarize the proposed stochastic primal-dual learning algorithm for training monotonic neural
networks in Algorithm 1. A key advantage of the proposed method is that we do not impose any


5


          _L_ ( _**θ**_ _,_ _**t**_ _,_ _**µ**_ ) = E _**x**_ _∼_ _**D**_ _J_ ( _f_ _**θ**_ ( _**x**_ )) + _**µ**_ _[⊤]_ E _**z**_ _∼_ Uni( _**X**_ )


��
_**t**_ + **0** _−_ _[∂f]_ _**[θ]**_ [(] _**[z]**_ [)]

_∂_ _**z**_ _m_


+


  
_−_ _α_ _**t**_


_._ (7)


The function (7) can be interpreted as a penalized version of problem (6). We penalize the violation
of constraint (6b) by a Lagrangian term associated with the dual variable _**µ**_ . Solving the problem (6)
is equivalent to finding a saddle point of the following constraint-free max-min problem:


max (8)
_**µ**_ _≥_ 0 [min] _**θ**_ _,_ _**t**_ _[L]_ [(] _**[θ]**_ _[,]_ _**[ t]**_ _[,]_ _**[ µ]**_ [)] _[.]_


The constraint-free nature of problem (8) enables the use of stochastic gradient-based algorithms in
conventional machine learning problems, e.g., SGD algorithm. The minimization over the primal
variable _**θ**_ is to optimize the performance of neural networks, while the stochastic gradient ascent over
the dual variable _**µ**_ can be understood as adaptively handling the constraint violations. Specifically,
the proposed stochastic primal-dual gradient (SPDG) algorithm reads as follows:


_**θ**_ _[k]_ [+1] = _**θ**_ _[k]_ _−_ _γ_ _**θ**_ _∇_ _**θ**_ _L_ ( _**θ**_ _[k]_ _,_ _**t**_ _[k]_ _,_ _**µ**_ _[k]_ ) _,_ (9a)

_**t**_ _[k]_ [+1] =              - _**t**_ _[k]_ _−_ _γ_ _**t**_ _∇_ _**t**_ _L_ ( _**θ**_ _[k]_ _,_ _**t**_ _[k]_ _,_ _**µ**_ _[k]_ )�+ _[,]_ (9b)

_**µ**_ _[k]_ [+1] =        - _**µ**_ _[k]_ + _γ_ _**µ**_ _∇_ _**µ**_ _L_ ( _**θ**_ _[k]_ [+1] _,_ _**t**_ _[k]_ [+1] _,_ _**µ**_ _[k]_ )�+ _[,]_ (9c)


where ( _γ_ _**θ**_ _, γ_ _**t**_ ) and _γ_ _**µ**_ are positive learning rates for the primal and dual variables, respectively.
To align with the prevalent minibatch training style, we suppose the SPDG algorithm is iteratively
performed over batched samples _{_ _**x**_ _**[s]**_ _}_ _[S]_ _s_ =1 [from] _**[ D]**_ [and] _[ {]_ _**[z][n]**_ _[}][N]_ _n_ =1 [from][ Uni(] _**[X]**_ [)][.] [Using the rule in]
stochastic gradient descent, the gradient term in the primal update (9a) can be computed as:


_−_ _α_ _**t**_

+


��
_**t**_ + **0** _−_ _[∂f]_ _**[θ]**_ [(] _**[z]**_ _[n]_ [)]

_∂_ _**z**_ _m_


_∇_ _**θ**_ _L_ = [1]

_S_


_S_


- _∇_ _**θ**_ _J_ ( _f_ _**θ**_ ( _**x**_ _[s]_ )) + [1]

_N_

_s_ =1


_N_


_N_

- _**µ**_ _[⊤]_ _∇_ _**θ**_


_n_ =1


_._ (10)


Indeed, the gradient (10) can be readily obtained through the backpropagation of the neural network
by letting _L_ as the loss function. The first term in (10) is the descent direction of the original problem
(2), while the second term rectifies the descent direction considering the constraint (6b). As for the
gradient terms in (9b) and (9c), they can be computed as:


_·_ _**µ**_ _−_ _α_ _**µ**_ _,_ and _∇_ _**µ**_ _L_ = [1]
_N_


_N_


_n_ =1


_**t**_ _−_ _[∂f]_ _**[θ]**_ [(] _**[z]**_ [)]

_∂_ _**z**_ _m_


_−_ _α_ _**t**_ _._ (11)
+


_∇_ _**t**_ _L_ = [1]

_N_


_N_


  
- 1 _**t**_ _−_ _[∂f]_ _**[θ]**_ [(] _**[z]**_ [)]

_∂_ _**z**_ _m_

_n_ =1


_∂_ _**z**_ _m_


(a) Original (b) Unconstrained (881)


(c) SMNN (901) (d) Ours (881)


|Col1|Col2|Col3|Col4|Col5|urs)|
|---|---|---|---|---|---|
|||||SMNN<br>Unconstrained<br>Constrained (|urs)|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


**end**


4 EXPERIMENTS


In this section, we demonstrate the effectiveness of our proposed method in various practical tasks.
We first conduct experiments on a 2D example from (Liu et al., 2020), and Figure 1 shows the results.
We then conduct experiments on public datasets from (Liu et al., 2020; Sivaraman et al., 2020;
Nolte et al., 2022; Runje & Shankaranarayana, 2023; Kim & Lee, 2024) and compare them with
state-of-the-art methods. The results show that our method can achieve higher accuracies while using
fewer model parameters. Finally, we conduct experiments on a real-world safe-critical frequency
control system from (Cui et al., 2023a). Our proposed method also shows improved performance.


4.1 EXPERIMENTS ON PUBLIC DATASETS


6


10 2


10 3


10 4


Number of Parameters


Figure 1: Comparison of unconstrained network, SMNN (Kim & Lee, 2024), and our method on a
2D example from (Liu et al., 2020), i.e., fitting _f_ ( _x, y_ ) = _a_ sin ( _x/_ 25 _π_ ) + _b_ ( _x −_ 0 _._ 5) [3] + _c_ exp( _y_ ) +
_y_ [2] _, a, b, c ∈{_ 0 _._ 3 _,_ 0 _._ 6 _,_ 1 _._ 0 _}, x, y_ _∈_ [0 _,_ 1]. **Left:** Contour plots when _a, b, c_ = 1 _._ 0 with the number in
the parenthesis denoting the number of parameters. Our method shows the best fitting result. **Right:**
Evaluation of three methods on all combinations of _a, b, c_ with increasing network size. The lines
denote the average MSE of 27 runs. Our method outperforms the other two methods.


constraints on the neural network architectures. This is desirable as it can take advantage of classical
neural networks, which often have stronger expressivity and are easier to train. Observing the dual
update (9c) and the gradient _∇_ _**µ**_ _L_, we can find that the dual variable _**µ**_ will automatically modulate
the penalty strength based on the degree of constraint satisfaction. This adjustment occurs seamlessly,
obviating the manual tuning of regularization like (Liu et al., 2020). In this sense, we can interpret
Algorithm 1 as an adaptive regularization approach. In practice, one may also consider to fix the
auxiliary variable _**t**_ at a small positive constant vector to further ease the training.


**Algorithm 1:** Stochastic Primal-Dual Learning Algorithm

Randomly initialize neural network _f_ _**θ**_ ( _**x**_ ), initialize _**t**_ = **0**, _**µ**_ = **0**
**Input:** Dataset _**D**_, Input domain _**X**_
**for** _epoch_ = 0 _,_ 1 _,_ 2 _, . . ._ **do**

Randomly draw samples _{_ _**x**_ _**[s]**_ _}_ _[S]_ _s_ =1 [from] _**[ D]**_ [ and] _[ {]_ _**[z]**_ _[n][}]_ _n_ _[N]_ =1 [from][ Uni(] _**[X]**_ [)]
Observe the derivatives _[∂f]_ _∂_ _**[θ]**_ _**z**_ [(] _m_ _**[z]**_ [)] _|_ _**z**_ = _**z**_ _n_ of the neural network

Use backpropagation to compute _∇_ _**θ**_ _L_ via (10), and update the neural network parameter:


_**θ**_ _[k]_ [+1] = _**θ**_ _[k]_ _−_ _γ_ _**θ**_ _∇_ _**θ**_ _L_ ( _**θ**_ _[k]_ _,_ _**t**_ _[k]_ _,_ _**µ**_ _[k]_ )


Compute the gradients _∇_ _**t**_ _L_ and _∇_ _**µ**_ _L_ by (11). Update the auxiliary and dual variables:


_**t**_ _[k]_ [+1] = - _**t**_ _[k]_ _−_ _γ_ _**t**_ _∇_ _**t**_ _L_ ( _**θ**_ _[k]_ _,_ _**t**_ _[k]_ _,_ _**µ**_ _[k]_ )�


+


+ _[,]_ [and] _**[ µ]**_ _[k]_ [+1] [=] - _**µ**_ _[k]_ + _γ_ _**µ**_ _∇_ _**µ**_ _L_ ( _**θ**_ _[k]_ [+1] _,_ _**t**_ _[k]_ [+1] _,_ _**µ**_ _[k]_ )�


Table 1: Comparison of our method with other methods presented in (Liu et al., 2020; Nolte et al.,
2022; Runje & Shankaranarayana, 2023; Kim & Lee, 2024).


COMPAS Blog Feedback Loan Defaulter
Method

Parameters Test Acc _↑_ Parameters RMSE _↓_ Parameters Test Acc _↑_


Isotonic N.A. 67 _._ 6% N.A. 0 _._ 203 N.A. 62 _._ 1%
XGBoost N.A. 68 _._ 5% _±_ 0 _._ 1% N.A. 0 _._ 176 _±_ 0 _._ 005 N.A. 63 _._ 7% _±_ 0 _._ 1%
Crystal 25840 66 _._ 3% _±_ 0 _._ 1% 15840 0 _._ 164 _±_ 0 _._ 002 16940 65 _._ 0% _±_ 0 _._ 1%
DLN 31403 67 _._ 9% _±_ 0 _._ 3% 27903 0 _._ 161 _±_ 0 _._ 001 29949 65 _._ 1% _±_ 0 _._ 2%
Min-Max Net 42000 67 _._ 8% _±_ 0 _._ 1% 27700 0 _._ 163 _±_ 0 _._ 001 29000 64 _._ 9% _±_ 0 _._ 1%
Non-Neg-DNN 23112 67 _._ 3% _±_ 0 _._ 9% 8492 0 _._ 168 _±_ 0 _._ 001 8502 65 _._ 1% _±_ 0 _._ 1%
Certified MNN 23112 68 _._ 8% _±_ 0 _._ 2% 8492 0 _._ 158 _±_ 0 _._ 001 8502 65 _._ 2% _±_ 0 _._ 1%
LMN 37 69 _._ 3% _±_ 0 _._ 1% 2225 0 _._ 160 _±_ 0 _._ 001 **65** _._ **4** % _±_ **0** _._ **0** %
Constrained MNN 2317 69 _._ 2% _±_ 0 _._ 2% 1101 0 _._ 156 _±_ 0 _._ 001 177 65 _._ 3% _±_ 0 _._ 1%
SMNN 2657 69 _._ 3% _±_ 0 _._ 9% **1421** **0** _._ _±_ **0** _._ 501 65 _._ 0% _±_ 0 _._ 1%
Ours **2069** **69** _._ **4** % _±_ **0** _._ **1** % 847 0 _._ 151 _±_ 0 _._ 001 **65** _._ **4** % _±_ **0** _._ **0** %


For the classification tasks, we use cross-entropy loss for training, and for the regression tasks, we
use mean-square-error loss. For all the tasks, we utilize the Adam optimizer to train the primal
variable (model parameter) _**θ**_ and update the dual variable using the rule (9c). We initialize the dual
variable _**µ**_ = **0**, and fix the auxiliary variable _**t**_ = 1 _×_ 10 _[−]_ [4] . The chance constraint coefficient _α_ is
set as 0 _._ 1 for all experiments. The more detailed configurations of the experiments are available in
Appendix A. The proposed method is compared with the benchmarking methods in (Liu et al., 2020;
Runje & Shankaranarayana, 2023; Kim & Lee, 2024) and other methods described in them that can
generate partial monotonic models. The methods include Isotonic Regression (Kalai & Sastry, 2009),
XGBoost (Chen et al., 2015), Crystal (Milani Fard et al., 2016), Deep Lattice Network (DLN) (You
et al., 2017), Min-Max Net (Daniels & Velikova, 2010), Non-Neg-DNN (Archer & Wang, 1993),
Certified Monotonic Neural Network (Certified MNN) (Liu et al., 2020), Counterexample-Guided
Learning (COMET) (Sivaraman et al., 2020), Lipschitz Monotonic Network (LMN) (Nolte et al.,
2022), Constrained MNN (Runje & Shankaranarayana, 2023), and Scalable Monotonic Neural
Network (SMNN) (Kim & Lee, 2024).


Tables 1 and 2 present the experimental results on the datasets above. We evaluate the models with
metrics such as test accuracy for classification and mean squared error and root mean squared error
(MSE and RMSE) for regression, and compare the model complexity (i.e., number of parameters) if
available. We run the experiments ten times per dataset after finding the optimal hyperparameters
and report the mean and standard deviation of the best five results, which is aligned with previous
studies (Runje & Shankaranarayana, 2023; Kim & Lee, 2024). Table 1 shows that our method can
attain the top positions in COMPAS and Loan Defaulter datasets. Although LMN also secures the top
position in Loan Defaulter, it needs more model parameters. As for the Blog Feedback dataset where
the RMSE of our method is slightly larger than SMNN, our method uses the smallest model sizes
while achieving comparable performance. Table 2 also demonstrates the superior performance of our
method in Auto MPG and Heart Disease datasets. We can see that our method outperforms all the


7


We conduct experiments on five publicly
available datasets that were used in previous Table 2: Comparison of our method with other methods
studies (Liu et al., 2020; Sivaraman et al., presented in (Sivaraman et al., 2020; Nolte et al., 2022;
2020; Nolte et al., 2022; Runje & Shankara- Runje & Shankaranarayana, 2023; Kim & Lee, 2024
narayana, 2023; Kim & Lee, 2024): COM- Auto MPG Heart Disease
PAS, Blog feedback, Loan defaulter, Auto Method

MSE _↓_ Test Acc _↑_

MPG, and Heart disease. COMPAS is a
classification dataset with 13 features of Min-Max Net 10 _._ 14 _±_ 1 _._ 54 0 _._ 75 _±_ 0 _._
which 4 are monotonic. Blog feedback is DLN 13 _._ 34 _±_ 2 _._ 42 0 _._ 86 _±_ 0 _._
a regression dataset with 276 features of COMET 8 _._ 81 _±_ 1 _._ 81 0 _._ 86 _±_ 0 _._
which 8 are monotonic. Loan defaulter is LMN 7 _._ 58 _±_ 1 _._ 20 0 _._ 90 _±_ 0 _._
a classification dataset with 28 features of Constrained MNN 8 _._ 37 _±_ 0 _._ 08 0 _._ 89 _±_ 0 _._

SMNN 7 _._ 44 _±_ 1 _._ 20 0 _._ 88 _±_ 0 _._

which 5 are monotonic. Auto MPG is a re
Ours **5** _._ **82** _±_ **0** _._ **26** **0** _._ **92** _±_ **0** _._

gression dataset with 7 features of which 3
are monotonic and Heart disease is a classification dataset with 13 features of which 2 are monotonic. Appendix A provides more details.


Table 2: Comparison of our method with other methods
presented in (Sivaraman et al., 2020; Nolte et al., 2022;
Runje & Shankaranarayana, 2023; Kim & Lee, 2024).


Auto MPG Heart Disease
Method


MSE _↓_ Test Acc _↑_


Min-Max Net 10 _._ 14 _±_ 1 _._ 54 0 _._ 75 _±_ 0 _._ 04
DLN 13 _._ 34 _±_ 2 _._ 42 0 _._ 86 _±_ 0 _._ 02
COMET 8 _._ 81 _±_ 1 _._ 81 0 _._ 86 _±_ 0 _._ 03
LMN 7 _._ 58 _±_ 1 _._ 20 0 _._ 90 _±_ 0 _._ 02
Constrained MNN 8 _._ 37 _±_ 0 _._ 08 0 _._ 89 _±_ 0 _._ 00
SMNN 7 _._ 44 _±_ 1 _._ 20 0 _._ 88 _±_ 0 _._ 04
Ours **5** _._ **82** _±_ **0** _._ **26** **0** _._ **92** _±_ **0** _._ **14**


|1e 2|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||Obj.|Cost|0.0|48 p|.u.|
|||||||
|||||||
|||||||


|1e 2|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||O|j. C|ost=|0.0|38 p|.u.|
||||||||
||||||||
||||||||


|1e 2|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||Ob|j. C|ost=|0.0|36|.u.|
||||||||
||||||||
||||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
||gen<br>~~gen~~|||||
||gen<br>~~gen~~|1<br>|gen 5<br>~~gen 6~~|<br>|gen 8<br>~~en 9~~|
||<br>gen<br>gen|<br> 3<br> 4|<br>gen 7|<br>|<br>gen 10|


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
|||gen 1<br>~~gen 2~~||gen 5<br>~~gen 6~~|<br>|gen 8<br>~~en 9~~|
|||<br>gen 3<br>gen 4||<br>gen 7|<br>|<br>gen 10|


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
|||gen 1<br>~~gen 2~~||gen 5<br>~~gen 6~~||gen 8<br>~~gen 9~~|
|||<br>gen 3<br>gen 4||<br>gen 7||<br>gen 10|


|1e 1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


|1e 1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||


|1e 1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||


s.t. ( _λi_ ( _k_ + 1) _, ωi_ ( _k_ + 1)) = _F_ ( _λi_ ( _k_ ) _, ωi_ ( _k_ )) + _G_ ( _uθi_ ( _ωi_ ( _k_ ))) _,_ (12b)
_uθi_ ( _·_ ) is monotonically increasing _._ (12c)


where _ωi_ and _λi_ are the frequency deviation and phase angle of bus _i_, respectively, and _uθi_ ( _·_ ) is a controller that is typically parametrized by a neural network. The bold symbols _**ωi**_ = ( _ωi_ (0) _, . . ., ωi_ ( _K_ ))


8


1


0


1


1


0


1


1


0


1


0.2


0.1


0.0


0.2


0.1


0.0


0.2


0.1


0.0


0.1


0.1


0.1


0.2
0 1 2 3 4 5 6 7


0.2
0 1 2 3 4 5 6 7


0.2
0 1 2 3 4 5 6 7


0.0


0.5


time (s)


(a)


0.0


0.5


time (s)


(b)


0.0


0.5


time (s)


(c)


Figure 2: Comparison of the control trajectories and objective costs on a real-world frequency control
system. The trajectories include frequency deviation _**w**_, control action _**uθ**_ and phase angle _**λ**_ on
selected 10 buses. The numbers in the figures denote the objective cost (12a). The monotonic
controllers are modeled by three methods: (a) the SMNN from (Kim & Lee, 2024); (b) the monotonic
SNN in the original paper (Cui et al., 2023a); (c) our constrained learning NN.


benchmarking methods with significant improvements. It is worth noting that our method, designed
with general network architectures, is easier to train than customized architecture methods such as
the recent state-of-the-art SMNN. Compared to Certified MNN, which requires manually increasing
the coefficient of regularization that may cause training failures with excessive penalties, our method
does not encounter any training failure cases.


4.2 EXPERIMENTS ON FREQUENCY CONTROL


The previous works (Liu et al., 2020; Runje & Shankaranarayana, 2023; Kim & Lee, 2024) predominantly focused on supervised learning tasks like classification and regression as discussed in
subsection 4.1, leaving uncertainties about their monotonicity frameworks in unsupervised learning
domains. In this subsection, we extend the experiments to an unsupervised learning task, namely, a
task that is trained by reinforcement learning algorithms, to further assess the performance of the
proposed method. We consider an optimal frequency control problem in a real-world power system
from (Cui et al., 2023a). The power network comprises 39 buses with 10 of them integrated with
inverter-connected resources. These resources, equipped with controllers, are deployed in the power
system to provide frequency regulation to resist power disturbances. As the power network connects
numerous end users and large amounts of vulnerable devices, the stability of the closed-loop system
built on the controllers is essential. This necessitates that the control actions exhibit monotonicity
concerning the control inputs, which is also analyzed using Lyapunov stability theory (Cui et al.,
2023a). The optimal controllers are obtained by solving the following optimization problem:


min
_θ_


_n_

- ( _∥_ _**ωi**_ _∥∞_ + _ηC_ ( _**uθi**_ )) (12a)

_i_


|Col1|Col2|SMNN|Col4|Col5|
|---|---|---|---|---|
|||~~NN~~<br>Ours|||
||||||
||||||


|Col1|Col2|SMNN|Col4|Col5|
|---|---|---|---|---|
|||~~NN~~<br>Ours|||
||||||
||||||


|Col1|Col2|SMNN|Col4|Col5|
|---|---|---|---|---|
|||~~NN~~<br>Ours|||
||||||
||||||


|Col1|Col2|SMNN|Col4|Col5|
|---|---|---|---|---|
|||~~NN~~<br>Ours|||
||||||
||||||


|Col1|Col2|SMNN|Col4|Col5|
|---|---|---|---|---|
|||NN<br>Ours|||
||||||
||||||


|Col1|Col2|SNN<br>Ours|Col4|Col5|
|---|---|---|---|---|
||||||


|Col1|Col2|SNN<br>Ours|Col4|Col5|
|---|---|---|---|---|
||||||


|Col1|Col2|SNN<br>Ours|Col4|Col5|
|---|---|---|---|---|
||||||


|Col1|Col2|SNN<br>Ours|Col4|Col5|
|---|---|---|---|---|
||||||


|Col1|Col2|SNN<br>Ours|Col4|Col5|
|---|---|---|---|---|
||||||


Figure 3: The input-output plots of the learned controllers modeled by SMNN (Kim & Lee, 2024),
monotonic SNN (Cui et al., 2023a), and our constrained learning. Each controller is confined by a
feasible region _u_ ~~_i_~~ _≤_ _uθi_ _≤_ _ui_ . All three methods can learn monotonic controllers. However, the
SMNN controllers truncate the default feasible region of generators 2 _,_ 5 _,_ 7 _,_ 8 _,_ 9 _,_ 10.


and _**uθi**_ = ( _uθi_ ( _ωi_ (0)) _, . . ., uθi_ ( _ωi_ ( _K_ ))) represent the control trajectories over _K_ steps. Equation
(12b) describes the state transition of the power system, framing (12) as a quintessential optimal
control problem. In this experiment, we utilize the reinforce-style algorithm developed in (Cui et al.,
2023a) to train the controllers but impose the monotonicity constraint (12c) into controller _uθi_ ( _·_ )
through three different methods: monotonic single-layer neural network (SNN) in the original paper
(Cui et al., 2023a), the recent state-of-the-art SMNN (Kim & Lee, 2024), and our constrained learning
NN. We set the coefficient _η_ = 4 _×_ 10 _[−]_ [4], and train the three models with Adam optimizer in identical
environments (i.e., the power system under the same set of power disturbances). As for our proposed
method, we initialize the dual variable _**µ**_ = 0 and fix the auxiliary variable _**t**_ = 1 _×_ 10 _[−]_ [4] . The chance
constraint coefficient _α_ is set as 0 _._ 1 and the learning rate _γ_ _**µ**_ for the dual variable is 10. Appendix B
provides more comprehensive descriptions of problem (12) and the corresponding configurations.


Figure 2 compares the trajectories of frequency deviation _**ω**_, control action _**uθ**_ and phase angle _**λ**_ in
the buses equipped with controllers after an identical power disturbance was injected to the power
system. We evaluate the models using the objective cost (12a). We conduct five independent runs
on each model and plot the best results in Figure 2. It shows that all three models can stabilize the
power system after a power disturbance occurs at time= 0 _._ 5s. Our method outperforms the other
two methods by achieving the lowest objective cost, exhibiting 25 _._ 0% improvement over SMNN and
5 _._ 3% improvement over monotonic SNN, respectively. This is mainly because our method allows the
use of general neural networks that are easier to train, which may help the algorithm attain better
controllers. From the upper three plots in Figure 2, we see that the frequency deviation _**ω**_ achieved
by our method is notably smaller than that of SMNN and monotonic SNN. The middle three plots
explain the reason since our method adopts the more aggressive control actions to promptly dampen
the frequency deviations. The bottom three plots further demonstrate that our method maintains a
more consistent alteration in phase angles, attributed to the smaller frequency deviations achieved.


Figure 3 compares the input-output plots of the learned controllers. It shows that all three methods
can produce monotonic controllers. However, a closer inspection reveals that the output regions of
SMNN controllers in generators 2 _,_ 5 _,_ 7 _,_ 8 _,_ 9 _,_ 10 are comparatively constrained, falling short of the
default feasible regions. Such limitations are undesirable in practical control and might consequently
result in inferior performances. Although monotonic SNN and our method yield similar plots from
generator 1 to 9, our method excels in learning a nonlinear controller in generator 10. This superiority
is also verified by the results in Figure 2, where our method achieves the lowest objective cost.


5 CONCLUSIONS AND FUTURE WORK


In this paper, we have proposed a novel, flexible, and adaptive learning framework to enforce
monotonicity of neural networks with general architectures. Our framework relies on a primal-dual
learning algorithm with only small extra computations to continuously and adaptively enforce the
monotonicity until the constraint is satisfied. It does not impose any constraints on the neural network
architectures nor needs case-by-case pre-processing such as tuning of the regularization. Experiments
on various practical tasks show that our method achieves competitive performance compared to recent
state-of-the-art methods. In the future work, we will extend the idea presented in this paper to train
neural networks with general inequality constraints, and consider its applications in other control
problems of safety-critical systems, e.g. robotic systems and sustainable energy systems.


9


0.2


0.1


0.0


0.1


0.2


0.2


0.1


0.0


0.1


0.2


Generator 1: (Hz)


0.10 0.05 0.00 0.05 0.10
Generator 6: (Hz)


0.2


0.0


0.2


0.4


0.1


0.2

Generator 2: (Hz)


0.1


0.2


Generator 3: (Hz)


0.05


0.00


0.05


0.10


Generator 4: (Hz)


0.2


Generator 5: (Hz)


0.2


0.10 0.05 0.00 0.05 0.10
Generator 10: (Hz)


0.1


0.2


0.10 0.05 0.00 0.05 0.10
Generator 9: (Hz)


0.10 0.05 0.00 0.05 0.10
Generator 7: (Hz)


0.25


0.50

0.10 0.05 0.00 0.05 0.10
Generator 8: (Hz)


ETHICS STATEMENT


In this paper, we propose a framework to train monotonic neural networks. Our method can be
utilized to address safety issues when applying machine learning in physical systems. Therefore, our
framework is a defensive method and our work does not discover any new threat. Our research also
does not include any human subjects. Accordingly, this paper does not raise ethical issues.


REPRODUCIBILITY STATEMENT


We are committed to make to all aspects of our work open-source, and provide comprehensive
instructions for guaranteeing reproducibility. All essential details necessary for reproducing our
experiments can be founded in Section 4 and Appendices A, B. The datasets, algorithms, and
pretrained models of our method are provided in supplementary materials.


REFERENCES


Julia Angwin, Jeff Larson, Surya Mattu, and Lauren Kirchner. There’s software used across the
country to predict future criminals. _And It’s biased against blacks. ProPublica_, 2016.


Norman P Archer and Shouhong Wang. Application of the back propagation neural network algorithm
with monotonicity constraints for two-group classification problems. _Decision Sciences_, 24(1):
60–75, 1993.


Clark Barrett and Cesare Tinelli. Satisfiability modulo theories. _Handbook of Model Checking_, pp.
305–343, 2018.


Christopher Bartley, Wei Liu, and Mark Reynolds. Enhanced random forest algorithms for partially
monotone ordinal classification. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
volume 33, pp. 3224–3231, 2019.


Mariusz Bojarski, Davide Del Testa, Daniel Dworakowski, Bernhard Firner, Beat Flepp, Prasoon
Goyal, Lawrence D Jackel, Mathew Monfort, Urs Muller, Jiakai Zhang, et al. End to end learning
for self-driving cars. _arXiv preprint arXiv:1604.07316_, 2016.


Krisztian Buza. Feedback prediction for blogs. In _Data Analysis, Machine Learning and Knowledge_
_Discovery_, pp. 145–152. Springer, 2013.


Li Chen, Penghao Wu, Kashyap Chitta, Bernhard Jaeger, Andreas Geiger, and Hongyang Li. End-toend autonomous driving: Challenges and frontiers. _IEEE Transactions on Pattern Analysis and_
_Machine Intelligence_, 2024.


Tianqi Chen, Tong He, Michael Benesty, Vadim Khotilovich, Yuan Tang, Hyunsu Cho, Kailong Chen,
Rory Mitchell, Ignacio Cano, Tianyi Zhou, et al. Xgboost: extreme gradient boosting. _R Package_
_Version 0.4-2_, 1(4):1–4, 2015.


Wenqi Cui, Yan Jiang, and Baosen Zhang. Reinforcement learning for optimal primary frequency
control: A lyapunov approach. _IEEE Transactions on Power Systems_, 38(2):1676–1688, 2023a.


Wenqi Cui, Yan Jiang, Baosen Zhang, and Yuanyuan Shi. Structured neural-PI control with end-toend stability and output tracking guarantees. _Advances in Neural Information Processing Systems_,
36:68434–68457, 2023b.


Hennie Daniels and Marina Velikova. Monotone and partially monotone neural networks. _IEEE_
_Transactions on Neural Networks_, 21(6):906–917, 2010.


Michael Doumpos and Constantin Zopounidis. Monotonic support vector machines for credit risk
rating. _New Mathematics and Natural Computation_, 5(03):557–570, 2009.


Mark Eisen, Clark Zhang, Luiz FO Chamon, Daniel D Lee, and Alejandro Ribeiro. Learning
optimal resource allocations in wireless systems. _IEEE Transactions on Signal Processing_, 67(10):
2775–2790, 2019.


10


Ad J Feelders. Prior knowledge in economic applications of data mining. In _European Conference_
_on Principles of Data Mining and Knowledge Discovery_, pp. 395–400. Springer, 2000.


Akhil Gupta, Naman Shukla, Lavanya Marla, and Arinbjörn Kolbeinsson. Monotonic trends in deep
neural networks. _arXiv preprint arXiv:1909.10662_, 2019a.


Akhil Gupta, Naman Shukla, Lavanya Marla, Arinbjörn Kolbeinsson, and Kartik Yellepeddi. How
to incorporate monotonicity in deep networks while preserving flexibility? _arXiv_ _preprint_
_arXiv:1909.10662_, 2019b.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_,
pp. 770–778, 2016.


Adam Tauman Kalai and Ravi Sastry. The isotron algorithm: High-dimensional isotonic regression.
In _Conference on Learning Theory_, volume 1, pp. 9, 2009.


Hyunho Kim and Jong-Seok Lee. Scalable monotonic neural networks. In _International Conference_
_on Learning Representations_, 2024.


B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A Al Sallab, Senthil Yogamani,
and Patrick Pérez. Deep reinforcement learning for autonomous driving: A survey. _IEEE_
_Transactions on Intelligent Transportation Systems_, 23(6):4909–4926, 2021.


Bernhard Lang. Monotonic multi-layer perceptron networks as universal approximators. In _Artificial_
_Neural Networks:_ _Formal Models and Their Applications–ICANN 2005_, pp. 31–37. Springer, 2005.


Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. _Nature_, 521(7553):436–444,
2015.


Xingchao Liu, Xing Han, Na Zhang, and Qiang Liu. Certified monotonic neural networks. _Advances_
_in Neural Information Processing Systems_, 33:15427–15438, 2020.


Mahdi Milani Fard, Kevin Canini, Andrew Cotter, Jan Pfeifer, and Maya Gupta. Fast and flexible
monotonic functions with ensembles of lattices. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 29, 2016.


Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, and Demetri
Terzopoulos. Image segmentation using deep learning: A survey. _IEEE Transactions on Pattern_
_Analysis and Machine Intelligence_, 44(7):3523–3542, 2021.


Niklas Nolte, Ouail Kitouni, and Mike Williams. Expressive monotonic neural networks. In
_International Conference on Learning Representations_, 2022.


Daniel W Otter, Julian R Medina, and Jugal K Kalita. A survey of the usages of deep learning for
natural language processing. _IEEE Transactions on Neural Networks and Learning Systems_, 32(2):
604–624, 2020.


Rob Potharst and Adrianus Johannes Feelders. Classification trees for problems with monotonicity
constraints. _ACM SIGKDD Explorations Newsletter_, 4(1):1–10, 2002.


Laura Rieger, Chandan Singh, William Murdoch, and Bin Yu. Interpretations are useful: Penalizing
explanations to align neural networks with prior knowledge. In _International_ _Conference_ _on_
_Machine Learning_, pp. 8116–8126. PMLR, 2020.


Davor Runje and Sharath M Shankaranarayana. Constrained monotonic neural networks. In _Interna-_
_tional Conference on Machine Learning_, pp. 29338–29353. PMLR, 2023.


Joseph Sill. Monotonic networks. _Advances in Neural Information Processing Systems_, 10, 1997.


Joseph Sill and Yaser Abu-Mostafa. Monotonicity hints. _Advances in Neural Information Processing_
_Systems_, 9, 1996.


Aishwarya Sivaraman, Golnoosh Farnadi, Todd Millstein, and Guy Van den Broeck. Counterexampleguided learning of monotonic neural networks. _Advances_ _in_ _Neural_ _Information_ _Processing_
_Systems_, 33:11936–11948, 2020.


11


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in Neural Information Processing_
_Systems_, 30, 2017.


Seungil You, David Ding, Kevin Canini, Jan Pfeifer, and Maya Gupta. Deep lattice networks and
partial monotonic functions. _Advances in Neural Information Processing Systems_, 30, 2017.


12


A DETAILS OF EXPERIMENT 4.1


A.1 DETAILS OF DATASETS


We provide a detailed description of the five real-world datasets in Table 3. The datasets are chosen
from (Liu et al., 2020) and (Sivaraman et al., 2020), which are used in previous studies (Runje &
Shankaranarayana, 2023; Nolte et al., 2022; Kim & Lee, 2024) for evaluating model performance.
Table 3 includes the tasks of the datasets, number of features, number of monotonic features, the size
of the training set and the testing set. Following (Kim & Lee, 2024), we multiply the monotonically
non-increasing features by _−_ 1 to make them non-decreasing. All the inputs are normalized into the
range [0 _,_ 1].


Table 3: Summary of the real-world datasets.

Dataset Task #Features #Monotonic features #Train #Test


COMPAS Classification 13 4 4937 1235
Blog Feedback Regression 276 8 47302 6968
Loan Defaulter Classification 28 5 418697 70212
Auto MPG Regression 7 3 313 79
Heart Disease Classification 13 2 242 61


**COMPAS** **(Angwin** **et** **al.,** **2016):** COMPAS is a classification dataset containing the criminal
records of 6127 individuals arrested in Florida. The goal of this task is to predict whether an arrested
individual will commit a crime or not within two years. We use 13 input features for prediction and
the classification is based on the predicted risk scores. In the dataset, the risk score is expected to be
monotonically non-decreasing in 4 features: _number of prior adult convictions_, _number of juvenile_
_felony_, _number of juvenile misdemeanor_, and _number of other convictions_ .


**Blog Feedback (Buza, 2013):** Blog feedback is a regression dataset comprising 54270 samples
from blog posts. The goal of this task is to predict the number of comments in the upcoming 24
hours based on 276 features. Among all the features, 8 of them are expected to be monotonically
non-decreasing with the prediction, which are (A51, A52, A53, A54, A56, A57, A58, A59).


**Loan** **Defaulter:** The Loan Defaulter [1] dataset contains complete loan data for all loans issued
between 2007 and 2015 of several banks. The dataset contains 488909 data samples and each sample
has 28 features including the current loan status, the latest payment, and other additional information.
The goal of this task is to predict the possible loan defaulters. The prediction is expected to be
monotonically non-decreasing in _number of public record bankruptcies_, _debt-to-income ratio_ and
monotonically non-increasing in _credit score_, _length of employment_, and _annual income_ .


**Auto MPG:** Auto MPG [2] is a regression dataset containing 392 data samples. The goal of the task
is to predict city-cycle fuel consumption in miles per gallon (MPG) based on 7 features. Among
them, _weights_, _displacement_, and _horse power_ are monotonically non-increasing features with the
prediction.


**Heart Disease:** Heart Disease [3] is a classification dataset containing 303 data samples. The task
is to predict the presence of heart disease for a person based on 13 features. The risk of heart
disease should be monotonically increasing with respect to _trestbp (resting blood pressure)_ and _chol_
_(cholesterol level)_ .


A.2 DETAILS OF CONFIGURATIONS


We build the neural network models using a simple MLP structure with ReLU as the activation function. The model consists of three or four layers depending on specific tasks. For the hyperparameters


[1https://www.kaggle.com/wendykan/lenging-club-loan-data](https://www.kaggle.com/wendykan/lenging-club-loan-data)
[2https://archive.ics.uci.edu/dataset/9/auto+mpg](https://archive.ics.uci.edu/dataset/9/auto+mpg)
[3https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)


13


in the proposed constrained learning algorithm, we initialize the dual variables _**µ**_ = **0** and fix the
auxiliary variable _**t**_ = 1 _×_ 10 _[−]_ [4] . We set the learning rate for the dual variable as _γ_ _**µ**_ = 10 and set
the chance constraint coefficient as _α_ = 0 _._ 1 for all experiments. The number of data points sampled
from Uni( _**X**_ ) is set as _N_ = 128. We train the model for 1000 epochs on each of the datasets. Table
4 summarizes the other hyperparameters such as numbers of parameters, network structures, batch
sizes, and learning rates for the models in the five datasets.


Table 4: Hyperparameters of the networks on the datasets. For network structure, the two numbers
in parenthesis mean that the input layer is partitioned into two parts. The first number denotes the
layer size corresponding to the monotonic features, while the second number denotes the layer size
of non-monotonic features.

Dataset #Parameters Network structure Batch size Learning rate


COMPAS 2069 (4 _,_ 32) _−_ 32 _−_ 16 _−_ 1 256 5 _×_ 10 _[−]_ [4]

Blog Feedback 847 (1 _,_ 3) _−_ 3 _−_ 1 256 5 _×_ 10 _[−]_ [4]

Loan Defaulter 673 (8 _,_ 16) _−_ 8 _−_ 4 _−_ 1 512 5 _×_ 10 _[−]_ [4]

Auto MPG 417 (16 _,_ 16) _−_ 8 _−_ 1 128 5 _×_ 10 _[−]_ [3]

Heart Disease 1353 (8 _,_ 32) _−_ 16 _−_ 16 _−_ 1 128 2 _×_ 10 _[−]_ [4]


B DETAILS OF EXPERIMENT 4.2


B.1 DESCRIPTION OF OPTIMAL FREQUENCY CONTROL TASK


In this subsection, we provide a detailed description of problem (12). The formulation of the optimal
frequency control task can be formally written as the optimization problem below (Cui et al., 2023a):


_u_ ~~_i_~~ _≤_ _uθi_ ( _ωi_ ( _k_ )) _≤_ _ui,_ (13d)
_ωi_ ( _k_ ) _uθi_ ( _ωi_ ( _k_ )) _≥_ 0 _,_ (13e)
_uθi_ ( _·_ ) is monotonically increasing _._ (13f)


where equations (13b) and (13c) are the system transition functions, which are compactly written
as ( _λi_ ( _k_ + 1) _, ωi_ ( _k_ + 1)) = _F_ ( _λi_ ( _k_ ) _, ωi_ ( _k_ )) + _G_ ( _uθi_ ( _ωi_ ( _k_ ))) in problem (12). We can interpret
equations (13b) and (13c) as a deterministic environment [4] and _uθi_ ( _·_ ) is the policy of a smart agent,
which aims to minimize the cost [�] _i_ _[n]_ [(] _[∥]_ _**[ω][i]**_ _[∥]_ _∞_ [+] _[ ηC]_ [(] _**[u][θ]**_ _**i**_ [))] [under] [the] [random] [power] [disturbance]
_pm,i_ . The infinity norm over the control trajectory _**ωi**_ = ( _ωi_ (0) _, . . ., ωi_ ( _K_ )) is defined as _∥_ _**ωi**_ _∥∞_ =
max _k_ =0 _,...,K |ωi_ ( _k_ ) _|_ . The explanation of other symbols in (13b) and (13c) can be found in their
paper (Cui et al., 2023a). Inequality (13d) confines the feasible region of controller _i_, which can be
realized by clipping the output of the controller. Each controller is single-input single-output, and the
system has 10 controllers in total. The constraints (13e) and (13f) describe the expected property of
the designed controllers to guarantee the exponential stability of the system. We use the RNN-based
reinforcement learning algorithm developed in (Cui et al., 2023a) to train the controllers based on
three different models of _uθi_ ( _·_ ). The evaluation of the model performance is simple, as the smaller
objective cost implies the better performance of the models.


4(Cui et al., 2023a): [https://github.com/Wenqi-Cui/RNN-RL-Frequency-Lyapunov](https://github.com/Wenqi-Cui/RNN-RL-Frequency-Lyapunov)


14


min
_θ_


_n_

- ( _∥_ _**ωi**_ _∥∞_ + _ηC_ ( _**uθi**_ )) (13a)

_i_


s.t. _λi_ ( _k_ ) = _λi_ ( _k −_ 1) + _ωi_ ( _k −_ 1)∆ _t,_ (13b)


_pm,i_
_Mi_


_ωi_ ( _k_ ) = _−_ [∆] _[t]_

_Mi_


_|B|_


- _Bij_ sin ( _λij_ ( _k −_ 1)) + [∆] _[t]_

_Mi_

_j_ =1


 + 1 _−_ _[D][i]_ [∆] _[t]_

_Mi_


_ωi_ ( _k −_ 1) _−_ _M_ [∆] _[t]_ _i_ _uθi_ ( _ωi_ ( _k −_ 1)) _,_ (13c)


B.2 DETAILS OF EXPERIMENTAL CONFIGURATIONS


For our proposed method, we use the ReLU neural network to build the controllers. Each controller
_uθi_ ( _·_ ) consists of three layers and the network structure is 32 _−_ 32 _−_ 1 for _i_ = 1 _, . . .,_ 10. For fair
comparisons, we use the same network structure (i.e., 32 _−_ 32 _−_ 1) for SMNN and use the default
structure for monotonic SNN in the original paper. For all three methods, we set the length of an
episode as _K_ = 200 and the batch size as 600. We use the Adam optimizer with the learning rate
5 _×_ 10 _[−]_ [3] to train the controllers for 600 episodes. In the testing stage, we evaluate the controllers
modeled by three different methods using the same power disturbance _pm,i_ for fairness.


15
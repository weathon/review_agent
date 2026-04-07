# FEEDBACK-DRIVEN RECURRENT QUANTUM NEURAL

## NETWORK UNIVERSALITY


**Lukas Gonon** _[∗]_
School of Computer Science
University of St. Gallen, Switzerland
lukas.gonon@unisg.ch


**Juan-Pablo Ortega**
School of Physical and Mathematical Sciences
Nanyang Technological University, Singapore
Juan-Pablo.Ortega@ntu.edu.sg


ABSTRACT


**Rodrigo Mart´ınez-Pe˜na**
Donostia International Physics Center
San Sebasti´an, Spain
rodrigo.martinez@dipc.org


Quantum reservoir computing uses the dynamics of quantum systems to process temporal data, making it particularly well-suited for machine learning with
noisy intermediate-scale quantum devices. Recent developments have introduced
feedback-based quantum reservoir systems, which process temporal information
with comparatively fewer components and enable real-time computation while
preserving the input history. Motivated by their promising empirical performance,
in this work, we study the approximation capabilities of feedback-based quantum
reservoir computing. More specifically, we are concerned with recurrent quantum neural networks, which are quantum analogues of classical recurrent neural
networks. Our results show that regular state-space systems can be approximated
using quantum recurrent neural networks without the curse of dimensionality and
with the number of qubits only growing logarithmically in the reciprocal of the
prescribed approximation accuracy. Notably, our analysis demonstrates that quantum recurrent neural networks are universal with linear readouts, making them
both powerful and experimentally accessible. These results pave the way for practical and theoretically grounded quantum reservoir computing with real-time processing capabilities.


1 INTRODUCTION


Recent advances in quantum computing have led to a rapid development of quantum machine learning methods. These methods aim to exploit the potential computational speed-up and reduced complexity offered by quantum computing for machine learning purposes. For learning problems with
temporal structure, quantum reservoir computing (QRC) has emerged as a promising approach for
exploiting noisy intermediate-scale quantum (NISQ) technologies. In contrast to classical machine
learning methods based on bits valued in _{_ 0 _,_ 1 _}_, quantum bits (qubits) can be in a continuum of
states. QRC aims to exploit this fundamental difference to build efficient machine learning methods
for time series prediction and learning.


In this paper, we are concerned with recurrent quantum neural networks (RQNN), a particular type
of quantum reservoir computing method. RQNNs are a quantum analogue to classical recurrent
neural networks. RQNNs are built from quantum neural networks (QNNs), with weights and biases
typically realized via quantum circuits. Thus, these networks can be evaluated directly on quantum
computers. Thereby, quantum machine learning aims to achieve a significant increase in neural
network expressivity and computational speed-up in inference and training.


_∗_ Also affiliated as Honorary Senior Lecturer with the Department of Mathematics, Imperial College, London, United Kingdom


1


Motivated by their promising empirical performance, in this work, we study the approximation
capabilities of feedback-based quantum reservoir computing methods and, specifically, RQNNs. In
particular, our work provides precise bounds on the number of qubits and the size of the underlying
quantum circuit that is required to guarantee a prescribed approximation accuracy. Our results show
that QRNNs can approximate regular state-space systems using a quantum circuit with qubit number
only growing logarithmically in the reciprocal of the prescribed approximation accuracy and with
error rates not suffering from the curse of dimensionality. Thereby, our results pave the way for
theoretically grounded quantum reservoir computing with real-time processing capabilities.


1.1 RELATED LITERATURE


Quantum reservoir computing methods have been extensively studied for a variety of time-series
prediction and learning tasks, employing different architecture types such as online protocols (Mujal et al., 2023; Franceschetto et al., 2024), mid-circuit measurements and reset operations (Hu et al.,
2024; Murauer et al., 2025), feedback protocols (Kobayashi et al., 2024), QRC with quantum memristors (Spagnolo et al., 2022) and hybrid QRC techniques (Pfeffer et al., 2022; 2023). We provide
a detailed discussion of QRC methods in Appendix A.


Despite these promising developments, key questions regarding universal approximation capabilities
and expressivity of feedback-driven QRC methods have not been addressed in the literature. For
classical neural networks, qualitative and quantitative universal approximation theorems have been
extensively studied, with seminal works including, e.g. Hornik (1991); Barron (1993); Yarotsky
(2017). Universality results for the dynamic reservoir computing setting have been obtained in
(Grigoryeva & Ortega, 2018a;b; Gonon & Ortega, 2020; 2021; Gonon et al., 2023) for echo state
networks, state-affine systems and linear systems with polynomial / neural network readouts. For
(feedforward) QNNs first qualitative results on universal approximation properties of QNNs have
been proved only very recently P´erez-Salinas et al. (2020); Schuld et al. (2021). Subsequently,
quantitative approximation error bounds for feedforward QNNs were proved in Gonon & Jacquier
(2025); Yu et al. (2024); Aftab & Yang (2024).


For RQNNs, no quantitative approximation error bounds have been previously available in the literature. Moreover, previous universality results concerning QRC models have relied on the use of
polynomial output layers (Chen & Nurdin, 2019; Chen et al., 2020; Nokkala et al., 2021; Sannia
et al., 2024b;a), which yield a polynomial algebra that can then be used with the Stone-Weierstrass
theorem to obtain universality statements. Nevertheless, most numerical and experimental implementations of reservoir computers use linear output layers due to their simplicity and fast training.


1.2 CONTRIBUTIONS


For applications of QRC methods in learning tasks with temporal dependence, a precise understanding of RQNN approximation capabilities is essential. In this paper, we derive approximation error
bounds and prove universality statements for RQNN families with a linear output layer and in the
context of the feedback protocol. Universality refers to the ability of these families to uniformly
approximate arbitrarily well a large category of dynamic processes, so-called fading memory input/output systems. Thereby, we contribute to a precise understanding of RQNN approximation
capabilities in several aspects.


    - We provide RQNN approximation error bounds for regular state-space systems. Our first
main result, Theorem 4.6, shows that RQNNs are able to approximate regular state-space
systems without the curse of dimensionality, using quantum circuits with qubit number
only growing logarithmically in the reciprocal of the prescribed approximation accuracy.


    - In our second main result, Theorem 4.8, we prove that RQNNs can uniformly approximate
the arbitrary fading memory, causal, and time-invariant filters. In particular, RQNNs have
approximation properties as competitive as those of popular reservoir computing/statespace system families like echo state networks, state-affine systems, or linear systems with
polynomial/neural network readouts.


    - To prove these results, we first derive novel qualitative and quantitative approximation error
results for using feedforward QNNs to approximate functions and their derivatives (see
Proposition 4.4 and Corollary 4.5).


2


In comparison to Gonon & Jacquier (2025), our RQNNs introduce memory through a feedback
loop. Mathematically analysing our RQNNs architecture hence requires a novel, intricate analysis
of QNN approximations of functions jointly with their derivatives. Moreover, approximation analysis in the temporal domain is inherently much more challenging due to the feedback loop. Proving
Theorems 4.6 and 4.8 thus requires new techniques specifically tailored to deal with this situation
(see Appendix C). Most previous literature on RC and QRC universality (Grigoryeva & Ortega,
2018a;b; Gonon & Ortega, 2020; 2021; Chen & Nurdin, 2019; Chen et al., 2020; Nokkala et al.,
2021; Sannia et al., 2024b;a) implicitly assumes the search for an optimal model within a class in
which all parameters are estimated. Also our results are formulated for variational quantum circuits for which all parameters are trainable. Nevertheless, the obtained results and developed proof
techniques also promise to be useful for QRC systems in which certain parameters in the recurrent
layer are randomly generated. Our RQNN architecture builds on and extends the feedforward QNN
architecture introduced in Gonon & Jacquier (2025), which also admits results for the randomized
setting. Hence, combining the techniques developed here with these randomized architectures may
provide fruitful for studying randomization in the dynamic quantum reservoir computing setting.
Moreover, the obtained approximation error bounds may serve as a crucial ingredient for bounding
the overall generalization error of QRC methods, by combining our results with suitable risk bounds
as obtained in other contexts in Gonon et al. (2020); Chmielewski et al. (2025).


1.3 OUTLINE


The paper is structured as follows. Section 2 introduces background on filters, functionals, fadingmemory and echo state properties. Section 3 describes the RQNN model, a recurrent QNN with
state feedback. Section 4.1 derives QNN approximation error bounds for functions and their first
derivatives. We then use these results (see Proposition 4.4 and Corollary 4.5) to study the properties
of the RQNN state maps for approximating more general state equations. These results are then
used in Section 4.2 to prove the universal uniform approximation properties of the filters associated
with RQNN systems. More specifically, in Theorem 4.6 we provide filter approximation bounds
that show that RQNNs can uniformly approximate the filters induced by any contracting Barrontype state-space system. Finally, Theorem 4.8 of Section 4.3 extends this universality property to
the much larger category of arbitrary fading memory, causal, and time-invariant filters. The paper
concludes with Section 5, where the main contributions and outlook of the paper are summarized.


2 BACKGROUND ON FILTERS AND FUNCTIONALS


We start by introducing the input-output maps to be learnt in the dynamic setting. In a static context,
input-output maps are given by functions of the form _f_ : R _[d]_ _→_ R _[m]_ . For learning with temporal
dependence, the relevant input-output maps are _filters_ and _functionals_ defined on sequences.


Specifically, let (R _[n]_ ) [Z] denote the set of infinite real sequences of the form _**z**_ =
( _. . .,_ _**z**_ _−_ 1 _,_ _**z**_ 0 _,_ _**z**_ 1 _, . . ._ ), _**z**_ _i_ _∈_ R _[n]_, _i ∈_ Z; (R _[n]_ ) [Z] _[−]_ is the subspace consisting of left infinite sequences:
(R _[n]_ ) [Z] _[−]_ = _{_ _**z**_ = ( _. . .,_ _**z**_ _−_ 2 _,_ _**z**_ _−_ 1 _,_ _**z**_ 0) _|_ _**z**_ _i_ _∈_ R _[n]_ _, i_ _∈_ Z _−}_ . Analogously, ( _Dn_ ) [Z] and ( _Dn_ ) [Z] _[−]_ stand
for infinite and semi-infinite sequences, with elements in the subset _Dn_ _⊂_ R _[n]_ . Let _Dn_ _⊂_ R _[n]_ and
_BN_ _⊂_ R _[N]_ . We refer to the maps of the type _U_ : ( _Dn_ ) [Z] _−→_ ( _BN_ ) [Z] as _**filters**_ and to those like
_H_ : ( _Dn_ ) [Z] _−→_ _BN_ (or _H_ : ( _Dn_ ) [Z] _[−]_ _−→_ _BN_ ) as _**functionals**_ . A filter _U_ : ( _Dn_ ) [Z] _−→_ ( _BN_ ) [Z] is
called _**causal**_ when for any two elements _**z**_ _,_ _**w**_ _∈_ ( _Dn_ ) [Z] that satisfy that _**z**_ _τ_ = _**w**_ _τ_ for any _τ_ _≤_ _t_, for
a given _t ∈_ Z, we have that _U_ ( _**z**_ ) _t_ = _U_ ( _**w**_ ) _t_ . Let _Tτ_ : ( _Dn_ ) [Z] _−→_ ( _Dn_ ) [Z], _τ_ _∈_ Z be the _**time delay**_
operator defined by _Tτ_ ( _**z**_ ) _t_ := _**z**_ _t−τ_ . The filter _U_ is called _**time-invariant**_ when it commutes with
the time delay operator, that is, _Tτ_ _◦_ _U_ = _U_ _◦_ _Tτ_, for any _τ_ _∈_ Z, with the two operators _Tτ_ defined
in the appropriate sequence spaces. Finally, there is a bijection between causal time-invariant filters
and functionals on ( _Dn_ ) [Z] _[−]_, and we can use them interchangeably (Grigoryeva & Ortega, 2018b).


A specific class of filters is given by state-space systems (such as recurrent neural networks) determined by two maps, namely the _**recurrent**_ layer or the _**state map**_ _F_ : R _[N]_ _×_ R _[n]_ _−→_ R _[N]_, _n, N_ _∈_ N,
and a _**readout**_ or _**observation**_ map _h_ : R _[N]_ _→_ R _[m]_, _m ∈_ N, given by


_**x**_ _t_ = _F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _,_
(1)
**y** _t_ = _h_ ( _**x**_ _t_ ) _,_


where _t ∈_ Z, _**z**_ _t_ denotes the input, _**x**_ _t_ _∈_ R _[N]_ is the state vector, and **y** _t_ _∈_ R _[m]_ is the output vector.


3


Consider now subsets _BN_ _⊂_ R _[N]_ and _Dn_ _⊂_ R _[n]_ and a recurrent layer defined on them, that is,
_F_ : _BN ×_ _Dn_ _−→_ _BN_ and _h_ : _BN_ _→_ R _[m]_ . Denote by _Dm_ := _h_ ( _BN_ ) _⊂_ R _[m]_ . The recurrent system
_F_ is said to have the _**echo state property**_ with respect to inputs in ( _Dn_ ) [Z] when for any _**z**_ _∈_ ( _Dn_ ) [Z]

there exists a unique element _**x**_ _∈_ ( _BN_ ) [Z] that satisfies the first equation in (1), for each _t_ _∈_ Z.
When the echo state property holds, a unique filter _U_ _[F]_ : ( _Dn_ ) [Z] _−→_ ( _BN_ ) [Z] can be associated to
the recurrent system determined by _F_, namely, _U_ _[F]_ ( _**z**_ ) _t_ := _**x**_ _t_ _∈_ _BN_, for all _t ∈_ Z. We will denote
by _Uh_ _[F]_ [: (] _[D][n]_ [)][Z] _[−→]_ [(] _[D][m]_ [)][Z] [the corresponding filter determined by the entire recurrent system, that]
is, _Uh_ _[F]_ [(] _**[z]**_ [)] _[t]_ [=] _[h]_ - _U_ _[F]_ ( _**z**_ ) _t_ - := **y** _t_ _∈_ _Dm_, for all _t_ _∈_ Z. The filters _U_ _[F]_ and _Uh_ _[F]_ [are] [causal] [and]
time-invariant by construction. The echo state property is much related with the so-called _**fading**_
_**memory**_ _**property**_ defined as the continuity of _Uh_ _[F]_ [with] [respect] [to] [weighted] [norms] [in] [its] [domain]
and codomain (Boyd & Chua, 1985) or the product topologies when _Dn_ and _Dm_ are compact
(Grigoryeva & Ortega, 2018b). It can be shown that when _Dm_ is compact, the echo state property
implies the fading memory property (Manjunath, 2020; Ortega & Rossmannek, 2025b); see Ortega
& Rossmannek (2025c) for a comprehensive account of the dynamical implications of the fading
memory property as well as Ortega & Rossmannek (2025a) for a stochastic version.


3 RECURRENT QUANTUM NEURAL NETWORK ARCHITECTURE


Before going into details about the considered RQNN architecture, let us first explain the basic
working principle of feedforward QNNs built in quantum circuits. A QNN is built by transforming
quantum bits ( _qubits_ ) in a parametric quantum circuit. Each qubit is in state _|ψ⟩_ = _α |_ 0 _⟩_ + _β |_ 1 _⟩_ for
some _α_ _∈_ C, _β_ _∈_ C with _|α|_ [2] + _|β|_ [2] = 1 and with elementary quantum bit states _|_ 0 _⟩_ and _|_ 1 _⟩_ . For
a circuit with n qubits, at any given point in the circuit, the circuit state can thus be identified with a
vector in C _[n]_ `[U]` for _n_ `U` = 2 [n] . The quantum state _|ψ⟩_ can be transformed by applying a _quantum gate_,
that is, a unitary matrix `U` _∈_ C _[n]_ `[U]` _[×][n]_ `[U]` . A QNN now applies quantum gates `U` ( _**x**_ _,_ _**θ**_ ) that depend on the
initial data and neural network parameters and transforms the circuit accordingly. The QNN output
is obtained by measuring the final quantum state after applying the circuit quantum gates.


Next, we introduce in detail the employed RQNN architecture. Our recurrent quantum circuit is
constructed based on two parametric quantum gates `U` and `V`, which we now introduce. The construction extends the feedforward QNN architecture introduced in Gonon & Jacquier (2025) to a
recurrent setting by feeding back the network’s state.


**Construction of** `U` **.** For _δ, γ_ _∈_ [0 _,_ 2 _π_ ] and _α ∈_ R, denote by `R` x( _δ_ ), `R` y( _γ_ ), and `R` z( _α_ ) the rotations
around the X-, Y-and the Z-axis, corresponding to angles _δ_, _γ_ and _α_, respectively, and obtained as
the exponentials of the Pauli matrices:


_−_ cosi sin��2 _δ_ �2 _δ_ - _−_ cosi sin��2 _δ_ �2 _δ_ ��, `R` y( _γ_ ) := �cossin �� _γ_ 2 _γ_ 2�� _−_ cossin�� _γ_ 2 _γ_ 2��


`R` x( _δ_ ) := - cos - 2 _δ_ - _δ_ _−_ i sin - _δ_ 2 _δ_ 


cossin �� _γ_ 2 _γ_ 2�� _−_ cossin�� _γ_ 2 _γ_ 2���, `R` z( _α_ ) := - _e_ _[−]_ 0 [i] _[α]_ 2


2 0
0 _e_ [i] _[α]_ 2


2


_._


For a given accuracy parameter _n_ _∈_ N, consider weights _**a**_ = ( _**a**_ [1] _, . . .,_ _**a**_ _[n]_ ) _∈_ (R _[d]_ [+] _[N]_ ) _[n]_, _**b**_ =
( _b_ [1] _, . . ., b_ _[n]_ ) _∈_ R _[n]_ and _**γ**_ = ( _γ_ [1] _, . . ., γ_ _[n]_ ) _∈_ [0 _,_ 2 _π_ ] _[n]_ . For _i_ = 1 _, . . ., n_, we define parametric gate
maps `U` [(] 1 _[i]_ [)] [:] [R] _[N]_ _[×]_ [ R] _[d]_ _[→]_ [C][2] _[×]_ [2] [that] [map] [a] [current] [system] [state] _**[x]**_ [and] [a] [current] [observation] _**[z]**_ [to] [a]
rotation gate. Gate map _i_ depends on parameters _**a**_ _[i]_ _, b_ _[i]_ and is defined by


`U` [(] 1 _[i]_ [)][(] _**[x]**_ _[,]_ _**[ z]**_ [)] := `H R` z  - _−b_ _[i]_ [�] `R` z  - _−a_ _[i]_ _N_ + _d_ _[z][d]_  - _· · ·_ `R` z  - _−a_ _[i]_ _N_ +1 _[z]_ [1]  - `R` z  - _−a_ _[i]_ _N_ _[x][N]_  - _· · ·_ `R` z  - _−a_ _[i]_ 1 _[x]_ [1]  - `H`


for any _**x**_ = ( _x_ 1 _, . . ., xN_ ) _∈_ R _[N]_ and _**z**_ = ( _z_ 1 _, . . ., zd_ ) _∈_ R _[d]_, with `H` the Hadamard gate. We may
rewrite


`U` [(] 1 _[i]_ [)][(] _**[x]**_ _[,]_ _**[ z]**_ [) =] `[ R]` [x]    - _δ_ _[i]_ [�] _,_ _δ_ _[i]_ := _−b_ _[i]_ _−_ _a_ _[i]_ _N_ + _d_ _[z][d]_ _[· · · −]_ _[a][i]_ _N_ +1 _[z]_ [1] _[−]_ _[a][i]_ _N_ _[x][N]_ _[· · · −]_ _[a][i]_ 1 _[x]_ [1] _[.]_


Moreover, we also define the gates `U` [(] 2 _[i]_ [)] := `R` y - _γ_ _[i]_ [�] and denote the circuit parameters by _**θ**_ =
( _**a**_ _[i]_ _, b_ _[i]_ _, γ_ _[i]_ ) _i_ =1 _,...,n_ _∈_ **Θ** := (R _[d]_ [+] _[N]_ _×_ R _×_ [0 _,_ 2 _π_ ]) _[n]_ .


With these notations, we are now ready to define the key element of our parametric quantum circuit,
the gate `U` := `U` _**θ**_ ( _**x**_ _,_ _**z**_ ). `U` is defined as a block matrix built from the gates `U` [¯][(] _[i]_ [)] ( _**x**_ _,_ _**z**_ ) = `U` [(] 1 _[i]_ [)][(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ ⊗]_


4


`U` [(] 2 _[i]_ [)] as follows:

 `U` ¯ [(1)] ( _**x**_ _,_ _**z**_ ) **0** 4 _×_ 4 **0** 4 _×_ 4 _· · ·_ **0** 4 _×_ 4 **0** 4 _×n_ 0







**0** 4 _×_ 4 `U` ¯ [(2)] ( _**x**_ _,_ _**z**_ ) **0** 4 _×_ 4 _· · ·_ **0** 4 _×_ 4 ...
... ... ... ...

**0** 4 _×_ 4 _· · ·_ **0** 4 _×_ 4 `U` ¯ [(] _[n][−]_ [1)] ( _**x**_ _,_ _**z**_ ) **0** 4 _×_ 4 ...
**0** 4 _×_ 4 _· · ·_ _· · ·_ **0** 4 _×_ 4 `U` ¯ [(] _[n]_ [)] ( _**x**_ _,_ _**z**_ ) **0** 4 _×n_ 0
**0** _n_ 0 _×_ 4 _· · ·_ _· · ·_ _· · ·_ **0** _n_ 0 _×_ 4 **1** _n_ 0 _×n_ 0


`U` _**θ**_ ( _**x**_ _,_ _**z**_ ) :=





_._


Here, _n_ 0 is chosen as the smallest natural number such that the matrix dimension _n_ `U` = 4 _n_ + _n_ 0 is
a power of 2, that is, _n_ `U` = 2 [n] . It can be easily shown that _n_ 0 = 4 _κ_ with _κ_ _∈_ N, since 4 _n_ + _n_ 0
and 2 _n_ + _n_ 0 _/_ 2 must be even for n _≥_ 2. Then, `U` _∈_ C _[n]_ `[U]` _[×][n]_ `[U]` is a unitary quantum gate operating on
n = log2( _n_ `U` ) = 2 + log2( _n_ + _κ_ ) = _⌈_ log2(2 _n_ ) _⌉_ qubits with a diagonal-block structure:


_n_ + _κ−_ 1

- _|i⟩⟨i| ⊗_ **1** 4 _×_ 4 _._


_i_ = _n_


`U` _**θ**_ ( _**x**_ _,_ _**z**_ ) =


_n−_ 1

- _|i⟩⟨i| ⊗_ `U` [¯][(] _[i]_ [+1)] ( _**x**_ _,_ _**z**_ ) +


_i_ =0


These unitary operators with a block structure are known as uniformly controlled quantum gates.
They are present in many quantum algorithms and are used to decompose general unitary gates and
locally prepare arbitrary quantum states (M¨ott¨onen et al., 2004; Mottonen et al., 2004; Bergholm
et al., 2005; Arrazola et al., 2022; Park et al., 2019). They are defined as multi-controlled unitaries
where each unitary block targets a set of qubits, two qubits in this case, while the other log2( _n_ + _κ_ )
qubits act as control qubits. Multi-controlled unitaries are applied depending on the state of the control qubits, which are unchanged, and only modify the target qubits. These operations generalize the
CNOT gate for two qubits, in that we can now have several control and target qubits. Notice that the
block structure of the unitary `U` _**θ**_ arises from indexing the targets as the lowest-order bits. Recently,
efficient decompositions of multi-controlled unitaries have been proposed in terms of the number
of single-qubit and two-qubit gates (Zindorf & Bose, 2024; 2025), as well as for approximations of
the multi-controlled gate (Silva et al., 2024). Code implementations of these quantum gates can be
found in the _Qclib_ library (Araujo et al., 2023). Finally, the identity blocks **1** 4 _×_ 4 do not introduce
additional gates into the quantum circuit, so the effective circuit can be reduced to the application
of the `U` [¯][(] _[i]_ [)] gates. However, the number of control qubits is fixed by log2( _n_ + _κ_ ) and we need all of
them to compute the output probabilities, as we will see below.


**Construction of** `V` **.** Next, let `V` _∈_ C _[n]_ `[U]` _[×][n]_ `[U]` be any unitary matrix mapping _|_ 0 _⟩_ _[⊗]_ [n] to the state _|ψ⟩_ =
~~_√_~~ 1 _n_ - _ni_ =0 _−_ 1 _[|]_ [4] _[i][⟩]_ [which, for] _[ n]_ _[≥]_ [2][, is also explicitly given as] _[ |][ψ][⟩]_ [=] ~~_√_~~ 1 _n_ - _ni_ =0 _−_ 1 _[|][i][⟩⊗|]_ [00] _[⟩]_ [.] [Note that]

different choices of `V` are possible and the only required property is `V` _|_ 0 _⟩_ _[⊗]_ [n] = _|ψ⟩_ . We refer to
Appendix D for an example.


**Measuring** **circuit** **outputs.** We can now measure the state of the n-qubit system after applying
the gates `V` and `U` . The possible states that we could measure are given by 0 _, . . ., n_ `U` _−_ 1 (in binary).
By running the circuit repeatedly, we can now obtain (up to well-controlled Monte Carlo error, see
Appendix E) the probabilities P _[n]_ _m_ [that the measured state is in] _[ {][m,]_ [ 4 +] _[ m, . . .,]_ [ 4(] _[n][ −]_ [1) +] _[ m][}]_ [, for]
_m ∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_, where _m_ is the binary state of the last two qubits (the target qubits).


More formally, consider the unitary gate map `C` ( _**x**_ _,_ _**z**_ ) = `C` n _,_ _**θ**_ ( _**x**_ _,_ _**z**_ ) := `U` _**θ**_ ( _**x**_ _,_ _**z**_ ) `V` acting on n =
2 + log2( _n_ + _κ_ ) qubits. This circuit acts on the initial state _|_ 0 _⟩_ _[⊗]_ [n] via the quantum gates `V` and `U` as


1
`C` n _,_ _**θ**_ ( _**x**_ _,_ _**z**_ ) _|_ 0 _⟩_ _[⊗]_ [n] = ~~_√_~~
_n_


Then, we measure


_n−_ 1

- _|i⟩⊗_ `U` [(] 1 _[i]_ [+1)] ( _**x**_ _,_ _**z**_ ) _|_ 0 _⟩⊗_ `U` [(] 2 _[i]_ [+1)] _|_ 0 _⟩_ _._

_i_ =0


           -            P _[n,]_ _m_ _**[θ]**_ = P _[n,]_ _m_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) :=][ P] ” `C` n _,_ _**θ**_ ( _**x**_ _,_ _**z**_ ) _|_ 0 _⟩_ _[⊗]_ [n] _∈{m,_ 4 + _m, . . .,_ 4( _n −_ 1) + _m}_ ” _._ (2)


This is the sum of the probabilities of being in the states _|i⟩⊗|m⟩_, where _i_ = 0 _, . . ., n −_ 1. That is,


P _[n,]_ _m_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) =] [1]

_n_


_n_

- ��� _⟨m|_ - `U` [(] 1 _[i]_ [)][(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ |]_ [0] _[⟩⊗]_ `[U]` 2 [(] _[i]_ [)] _[|]_ [0] _[⟩]_ ����2 _._

_i_ =1


5


**Parallel** **circuits.** With _n_ (or equivalently n) fixed, the quantum circuit introduced above is
uniquely defined by the choice of circuit parameters _**θ**_ _∈_ **Θ** . We will now run _N_ such circuits
in parallel, each representing a component of the state map _F_ in (1). Each circuit is described by its
parameters _**θ**_ _[j]_ _∈_ **Θ**, _j_ _∈{_ 1 _, . . ., N_ _}_ . The circuit outputs then induce maps P _m_ _[n,]_ _**[θ]**_ _[j]_ : R _[N]_ _×_ R _[d]_ _→_ [0 _,_ 1]


**Recurrent quantum neural networks (RQNN).** With these ingredients, we can now define the
RQNN that we will consider. Given the gate map `C` n _,_ _**θ**_ and _R >_ 0, we define _F_ [¯] _R_ _[n,]_ _**[θ]**_ : R _[N]_ _×_ R _[d]_ _→_ R _[N]_

by its component maps _F_ [¯] _R_ _[n,]_ _**[θ]**_ = ( _F_ [¯] _R,_ _[n,]_ 1 _**[θ]**_ _[, . . .,]_ _[F]_ [¯] _[ n,]_ _R,N_ _**[θ]**_ [)][.] [For] _[j]_ [=] [1] _[, . . ., N]_ [,] [the] _[j]_ [-th] [component] [map]
_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [:][ R] _[N]_ _[×]_ [ R] _[d]_ _[→]_ [R][ is defined by]

_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) :=] _[ R][ −]_ [2] _[R]_ [[][P] 1 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ ) + P2 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ )] _,_ ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ _,_ (3)


with _**θ**_ = ( _**θ**_ [1] _, . . .,_ _**θ**_ _[N]_ ) _∈_ **Θ** _[N]_ . Our _**recurrent quantum neural network (RQNN)**_ is then defined
by the state-space system associated to the state map _F_ [¯] _R_ _[n,]_ _**[θ]**_

_**x**_ ˆ _t_ = _F_ [¯] _R_ _[n,]_ _**[θ]**_ [(ˆ] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] _[,]_ _t ∈_ Z _−._ (4)


Figure 1 provides a schematic representation of how the RQNN acts at each time step for the _j_ th circuit: at any time _t_, the system is initialized, the gates `V` and `U` _**θ**_ _j_ (ˆ _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) are applied, and
the system is measured. This process is repeated to estimate the probabilities P1 _[n,]_ _**[θ]**_ _[j]_ (ˆ _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) and
P _[n,]_ 2 _**[θ]**_ _[j]_ (ˆ _**x**_ _t−_ 1 _,_ _**z**_ _t_ ), which are aggregated into the network output _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(ˆ] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] [according] [to] [(3).]
Once this is done for all _j_ _∈{_ 1 _, . . ., N_ _}_, the network state _**x**_ ˆ _t_ is stored to be used as feedback for
the next time step _t_ + 1.


In the next paragraphs, we aim to address the following questions:


    - Can we choose the parameters _**θ**_ in such a way that the system determined by (4) satisfies
the echo state property?

    - Can the family of systems determined by equations of the type (4) approximate general
state-space systems arbitrarily well? More specifically, given an arbitrary state-space map
_**x**_ _t_ = _F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) with _F_ : R _[N]_ _×_ R _[d]_ _→_ R _[N]_ as general as possible, can it be approximated
by equations of the type (4)?


4 RECURRENT QUANTUM NEURAL NETWORK UNIVERSALITY


This section contains approximation guarantees and universality results for the recurrent quantum
neural network (RQNN) family. To achieve this, in Section 4.1 we first prove refined approximation error bounds (that generalize those in Gonon & Jacquier (2025)) for feedforward quantum


6


neural networks (QNNs) that allow us to control the error committed when approximating a function and its derivatives simultaneously, a crucial ingredient for analysing the RQNN feedback loop.
These error bounds show how recurrent QNNs can be used to approximate state-space maps _F_
arbitrarily well as long as these are sufficiently smooth and satisfy Barron-type integrability conditions like, for example, �R _[N]_ _×_ R _[d][ ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_ _∞_, for _i_ = 1 _, . . ., N_ + _d_ and _j_ = 1 _, . . ., N_, or

_Iq_ = �R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_ _∞_, for some _q_ _≥_ 2 (see Proposition 4.2 and Corollary 4.3 below);

RQNN state maps are hence universal in that category. These bounds are devised with respect to _L_ _[∞]_
and _L_ [2] -type norms. As we shall prove, in the _L_ _[∞]_ case, the universality of the RQNN family still
holds with respect to state maps that do not necessarily satisfy the Barron condition, even though
in that framework we do not formulate approximation bounds. Finally, in the last two sections, we
exploit all these results on the approximation of state maps to obtain universality statements and
error bounds for the approximation of arbitrary causal, time-invariant, and fading memory filters
using a modified recurrent QNN. In addition to the tools developed here, our proofs of these results
rely on techniques from Gonon & Ortega (2020; 2021) and the overall strategy is reminiscent of
the so-called internal approximation approach introduced in Grigoryeva & Ortega (2018b, Theorem
3.1 (iii)) for echo state networks, which consists of obtaining approximation results for filters out of
statements of that type for the state maps that generate them.


The approximation rate in all our results is free from the curse of dimensionality: the error decays
as ~~_√_~~ 1 as we increase _n_, with this rate of decay being independent of the input dimension _d_ and the
_n_
state space dimension _N_ . Moreover, the required number of qubits n = _⌈_ log2(2 _n_ ) _⌉_ is only growing logarithmically in the accuracy parameter _n_ . Put differently, our circuit requires only _O_ ( _ε_ _[−]_ [2] )
weights and _O_ ( _⌈_ log2( _ε_ _[−]_ [1] ) _⌉_ ) qubits suffices to achieve approximation error _ε_ _>_ 0 when approximating functions with sufficiently integrable Fourier transforms.


4.1 RQNN APPROXIMATION OF STATE-SPACE MAPS AND THEIR DERIVATIVES


As a first step, we aim to establish RQNN approximation results for a function jointly with its
derivatives. Denote by _FR_ the class of integrable functions _f_ : R _[N]_ _×_ R _[d]_ _→_ R with Fourier integral
bounded above by a constant _R >_ 0, that is,

_F_ :=   - _f_ : R _[N]_ _×_ R _[d]_ _→_ R : _f_ _∈C_ �R _[N]_ _×_ R _[d]_ [�] _∩_ _L_ [1][ �] R _[N]_ _×_ R _[d]_ [�] _,_ _∥f_ [�] _∥_ 1 _< ∞_   - _,_ (5)


     -     _FR_ := _f_ _∈F,_ with _∥f_ [�] _∥_ 1 _≤_ _R_ _,_ for _R >_ 0 _._ (6)


Here, for a continuous and integrable function _f_ : R _[N]_ _×_ R _[d]_ _→_ R we denote its Fourier transform
by _f_ [�] ( _**ξ**_ 1 _,_ _**ξ**_ 2) := �R _[N]_ _×_ R _[d][ e][−]_ [2] _[π]_ [i(] _**[y]**_ [1] _[,]_ _**[y]**_ [2][)] _[·]_ [(] _**[ξ]**_ [1] _[,]_ _**[ξ]**_ [2][)] _[f]_ [(] _**[y]**_ [1] _[,]_ _**[ y]**_ [2][)d] _**[y]**_ [1][d] _**[y]**_ [2][, with][ (] _**[ξ]**_ [1] _[,]_ _**[ ξ]**_ [2][)] _[ ∈]_ [R] _[N]_ _[×]_ [ R] _[d]_ [.]


Our first result derives a representation for the QRNN output.
**Proposition** **4.1.** _For_ _any_ _n_ _∈_ N _,_ _j_ = 1 _, . . ., N_ _,_ _**θ**_ = ( _**θ**_ [1] _, . . .,_ _**θ**_ _[N]_ ) _∈_ **Θ** _[N]_ _with_ _**θ**_ _[j]_ =
( _**a**_ _[i,j]_ _, b_ _[i,j]_ _, γ_ _[i,j]_ ) _i_ =1 _,...,n_ _∈_ **Θ** _, the RQNN introduced in_ (3) _can be represented as_


_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) =] [1]

_n_


_n_

- _R_ cos - _γ_ _[i,j]_ [�] cos - _b_ _[i,j]_ + _**a**_ _[i,j]_ _·_ ( _**x**_ _,_ _**z**_ )� _,_ _for all_ ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ _._ (7)


_i_ =1


Our next result provides an approximation error bound for the QRNN state map jointly with its
derivatives. The proof is provided in Appendix B.2. Let _µ_ be an arbitrary probability measure on
(R _[N]_ _×_ R _[d]_ _, B_ (R _[N]_ _×_ R _[d]_ )). Recall the notation


�� �1 _/_ 2
_∥f_ _−_ _g∥L_ 2( _µ_ ) := _._

R _[N]_ _×_ R _[d][ |][f]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[g]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [2] _[ µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]


**Proposition** **4.2.** _Let_ _R_ _>_ 0 _and_ _suppose_ _F_ = ( _F_ 1 _, . . ., FN_ ) : R _[N]_ _×_ R _[d]_ _→_ R _[N]_ _is_ _continu-_
_ously_ _differentiable_ _and_ _satisfies_ _Fj_ _∈FR_ _and_ _∂iFj_ _∈F_ _and_ �R _[N]_ R _[d][ ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_ _∞_ _for_


_ously_ _differentiable_ _and_ _satisfies_ _Fj_ _∈FR_ _and_ _∂iFj_ _∈F_ _and_ �R _[N]_ _×_ R _[d][ ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_ _∞_ _for_

_i_ = 1 _, . . ., N_ + _d and j_ = 1 _, . . ., N_ _._ _Then, for any n ∈_ N _, there exists_ _**θ**_ _∈_ **Θ** _[N]_ _such that_


_N_ + _d_

2 2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� [2] [+]                       - ��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� [2] _[≤]_ _[C][j]_ _[,]_


2
��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ���


_n_ _[,]_


_L_ [2] ( _µ_ ) [+]


_N_ + _d_


_i_ =1


2

_L_ [2] ( _µ_ ) _[≤]_ _[C]_ _n_ _[j]_


_for any j_ _∈{_ 1 _, . . ., N_ _}, where Cj_ = _∥F_ [�] _j∥_ [2] 1 [+ 4] _[π]_ [2] _[∥]_ _F_ [�] _j∥_ 1 

7


R _[N]_ _×_ R _[d]_ - _Ni_ =1+ _d_ _[ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _._


Next, we show that it is also possible to also obtain approximation results for QNNs with bounded
network coefficients. The proof is provided in Appendix B.3.

**Corollary 4.3.** _In the setting of Proposition 4.2, assume, in addition, that_ �R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_

_∞_ _for some q_ _≥_ 2 _._ _Then, for any n ∈_ N _, there exists_ _**θ**_ _∈_ **Θ** _such that for any j_ _∈{_ 1 _, . . ., N_ _},_


2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _L_ [2] ( _µ_ ) [+]


2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ���


_N_ + _d_


_i_ =1


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ���2 _L_ [2] ( _µ_ ) _[≤]_ _C_ ¯ _nj_ _[,]_


2 _C_ ¯ _j_

_L_ [2] ( _µ_ ) _[≤]_ _n_ _[,]_


_where_ _C_ ¯ _j_ = 3 _Cj._ _Moreover,_ _we_ _can_ _choose_ _**θ**_ = ( _**θ**_ [1] _, . . .,_ _**θ**_ _[N]_ ) _∈_ **Θ** _[N]_ _with_ _**θ**_ _[j]_ =
( _**a**_ _[i,j]_ _, b_ _[i,j]_ _, γ_ _[i,j]_ ) _i_ =1 _,...,n_ _in such a way that for all i_ = 1 _, . . ., n, j_ = 1 _. . ., N_ _,_


    _∥_ _**a**_ _[i,j]_ _∥≤_ 2 _π_ 3 _n∥F_ [�] _j∥_ _[−]_ 1 [1]


- - _q_ [1]

_Fj_ ( _**ξ**_ ) _|_ d _**ξ**_ _._ (8)
R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ [�]


Next, we complement the _L_ [2] (R _[N]_ _×_ R _[d]_ _, µ_ )-error bound in Proposition 4.2 with a uniform error
bound on compact sets. For _M_ _>_ 0 and _f, g_ _∈C_ (R _[N]_ _×_ R _[d]_ ) denote


_∥f_ _−_ _g∥∞,M_ := sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ _×_ [ _−M,M_ ] _[d][ |][f]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[g]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|][ .]_


**Proposition 4.4.** _Let R, M_ _>_ 0 _and suppose F_ = ( _F_ 1 _, . . ., FN_ ) _is continuously differentiable and_
_satisfies Fj_ _∈FR and ∂iFj_ _∈F_ _and_ �R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥]_ [4] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _< ∞_ _for j_ = 1 _, . . ., N_ _._ _Then, for any_

_n ∈_ N _, there exists_ _**θ**_ _∈_ **Θ** _such that for any j_ _∈{_ 1 _, . . ., N_ _},_


_N_ + _d_

��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _∞,M_ [+] 
_i_ =1


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� _∞,M_ _[≤]_ _[C]_ ~~_√_~~ _j_ _[∞]_ _n,_ (9)


_where Cj_ _[∞]_ = 2( _π_ + 1) _∥F_ [�] _j∥_ 1 + (8 _πM_ + 4 _π_ [2] )( _N_ + _d_ ) 12 _∥F_ [�] _j∥_ 121 _[I]_ 2 [1] _,j_ _[/]_ [2] [+ 16] _[Mπ]_ [2][(] _[N]_ [+] _[ d]_ [)] _[∥]_ _F_ [�] _j∥_ [1] 1 _[/]_ [2] _I_ 4 [1] _,j_ _[/]_ [2]

_for Iq,j_ = �R _[N]_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _< ∞._


1
_where Cj_ _[∞]_ = 2( _π_ + 1) _∥F_ [�] _j∥_ 1 + (8 _πM_ + 4 _π_ [2] )( _N_ + _d_ ) 2 _∥F_ [�] _j∥_


R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _< ∞._


The proof can be found in Appendix B.4. Finally, we obtain a qualitative universal approximation
result for QNNs jointly with their derivatives. The proof can be found in Appendix B.5.

**Corollary** **4.5.** _Let_ _F_ = ( _F_ 1 _, . . ., FN_ ) _be_ _continuously_ _differentiable._ _Then_ _for_ _any_ _ε_ _>_ 0 _and_
_X_ _⊂_ R _[N]_ _×_ R _[d]_ _compact there exist n_ _∈_ N _, R_ _>_ 0 _and_ _**θ**_ _∈_ **Θ** _such that for any j_ _∈{_ 1 _, . . ., N_ _},_
_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ _[satisfies]_

sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [ +] _[ ∥∇][F][j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −∇][F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥≤]_ _[ε.]_ (10)
( _**x**_ _,_ _**z**_ ) _∈X_


4.2 RECURRENT QNN APPROXIMATION BOUNDS FOR STATE-SPACE FILTERS


The results in the previous section show that the family of RQNNs that were introduced in (3) is
capable of approximating arbitrarily well the very general class of continuously differentiable statespace maps with bounded Fourier transform, together with their derivatives. These approximations
hold with respect to both the _L_ [2] norm (Proposition 4.2 and Corollary 4.3) and the _L_ _[∞]_ norm on
compacta (Proposition 4.4 and Corollary 4.5). We will now use the uniform RQNN approximation
results for the state maps to conclude similar uniform approximation results for the corresponding
filters under additional hypotheses that guarantee that those exist.


Consider a state-space system


_**x**_ _t_ = _F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _,_ _t ∈_ Z _−,_ (11)


with state process ( _**x**_ _t_ ) _t∈_ Z _−_ valued in R _[N]_, input process ( _**z**_ _t_ ) _t∈_ Z _−_ valued in R _[d]_ and _F_ : R _[N]_ _×_ R _[d]_ _→_
R _[N]_ . We work under the assumption that _F_ is contractive and satisfies Barron-type integrability
conditions (Barron, 1992; 1993; Barron & Klusowski, 2018). Then, e.g., Proposition 1 and Remark 2
in Gonon et al. (2020) imply that, for any compact _Dd_ _⊂_ R _[d]_, the associated filter _U_ _[F]_ : ( _Dd_ ) [Z] _[−]_ _→_
( _BN_ ) [Z] _[−]_ induced by the restriction of _F_ to _BN_ _× Dd_ is well-defined and continuous.


8


Our next result shows that among the RQNNs that we discussed in Proposition 4.4 there exist systems that have the echo state property and hence have a filter associated. More importantly, those
filters can be used to uniformly approximate any of the filters corresponding to the general systems
introduced above in (11) as long as they satisfy a Barron-type integrability condition and are sufficiently contractive. The proof can be found in Appendix C.1. Here, _∥· ∥_ 2 is the spectral norm. In
particular, this result shows that the error rate is free from the curse of dimensionality: the error decays as ~~_√_~~ 1 as we increase _n_, with this rate of decay being independent of the input dimension _d_ and
_n_
the state space dimension _N_ . Thus, the RQNN requires only _O_ ( _ε_ _[−]_ [2] ) weights and _O_ ( _⌈_ log2( _ε_ _[−]_ [1] ) _⌉_ )
qubits to achieve approximation error _ε >_ 0 for the considered state-space systems.
**Theorem** **4.6.** _Suppose_ _F_ _in_ (11) _is_ _continuously_ _differentiable_ _with_ _∥∇_ _**x**_ _F_ ( _**x**_ _,_ _**z**_ ) _∥_ 2 _≤_ _λ_ _for_ _all_
_**x**_ _∈_ R _[N]_ _,_ _**z**_ _∈_ _Dd_ _for_ _some_ _λ_ _∈_ (0 _,_ 1) _and,_ _moreover,_ _F_ _satisfies_ _Fj_ _∈FR,_ _∂iFj_ _∈F_ _and_
�R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥]_ [4] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _<_ _∞_ _for_ _j_ = 1 _, . . ., N_ _._ _Denote_ _by_ _U_ _[F]_ : ( _Dd_ ) [Z] _[−]_ _→_ ( _BN_ ) [Z] _[−]_ _the_ _filter_

_associated to_ (11) _._ _Then for any n ∈_ N _with n > n_ 0 _there exists_ _**θ**_ _∈_ **Θ** _such that the system_ (4) _has_
_the echo state property and the associated filter_ _U_ [¯] : ( _Dd_ ) [Z] _[−]_ _→_ (R _[N]_ ) [Z] _[−]_ _satisfies_


_√_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ _._ (12)
_n_


sup sup
_**z**_ _∈_ ( _Dd_ ) [Z] _[−]_ _t∈_ Z _−_


1
�� _U F_ ( _**z**_ ) _t −_ _U_ ¯ ( _**z**_ ) _t_ �� _≤_
1 _−_ _λ_


_Here, n_ 0 _may be chosen as n_ 0 = _N_ [2 (max] _[j]_ [=1] (1 _−_ _[,...,N]_ _λ_ ) [2] _[C]_ _j_ _[∞]_ [)][2] _._


Notice that _N_ represents the state space dimension of the target _F_, which is matched by the QRNN
dimension to obtain the approximation error bound. Theorem 4.6 also proves an advantage of
QRNNs over classical RNNs. RNN approximation bounds for state-space systems driven by Barrontype functions were obtained in (Gonon et al., 2023, Theorem 3). While the approximation rate in
Theorem 4.6 is the same ( 2 [1] [in both cases), the Fourier integrability condition required in the quan-]

tum case is _strictly weaker_ . Specifically, the condition �R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥]_ [4] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _< ∞_ implies that the

smoothness assumption (Gonon et al., 2023, Definition 2) required for (Gonon et al., 2023, Theorem 3) is satisfied. For example, consider a Sobolev function _F_ _∈_ _H_ _[s]_ (R _[N]_ _×_ R _[d]_ ). Then, the integrability condition for the QRNN approximation result is satisfied for any _s_ _>_ _[N]_ 2 [+] _[d]_ + 4 (by (Folland,

2020, Lemma 6.5) and its proof). In contrast, the integrability condition for the RNN approximation
result in (Gonon et al., 2023, Theorem 3) would require the stronger condition _s > N_ + _d_ + 3.


4.3 UNIVERSALITY


In the previous section, we proved error bounds for the approximation using recurrent QNNs of the
filters induced by contractive state-space targets with Barron-type integrability conditions. These
bounds show, in passing, the universality of the family of RQNN filters in that category. We now
extend this universality statement (without formulating error bounds) to the much larger family of
fading memory filters by introducing a modification in the RQNN reservoir. We define _F_ [˜] _R_ _[n,]_ _**[θ]**_ :
R _[N]_ _×_ R _[d]_ _→_ R _[N]_ by its component maps _F_ [˜] _R_ _[n,]_ _**[θ]**_ = ( _F_ [˜] _R,_ _[n,]_ 1 _**[θ]**_ _[, . . .,]_ _[F]_ [˜] _[ n,]_ _R,N_ _**[θ]**_ [)][.] [For] _[j]_ [=] [1] _[, . . ., N]_ [,] [the] _[j]_ [-th]
component map _F_ [˜] _R,j_ _[n,]_ _**[θ]**_ [:][ R] _[N]_ _[×]_ [ R] _[d]_ _[→]_ [R][ is defined by]

_F_ ˜ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) :=] _[ R][ −]_ [2] _[R]_ [[][P] 1 _[n,]_ _**[θ]**_ _[j]_ ( _Pj_ _**x**_ _,_ _**z**_ ) + P2 _[n,]_ _**[θ]**_ _[j]_ ( _Pj_ _**x**_ _,_ _**z**_ )] _,_ ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ _,_ (13)

with _**θ**_ = ( _**θ**_ [1] _, . . .,_ _**θ**_ _[N]_ ) _∈_ **Θ** _[N]_ and _P_ 1 _, . . ., PN_ _∈_ R _[N]_ _[×][N]_ linear preprocessing maps. Our modified
RQNN is then defined by the state-space system associated to the state map _F_ [˜] _R_ _[n,]_ _**[θ]**_

_**x**_ ˆ _t_ = _F_ [˜] _R_ _[n,]_ _**[θ]**_ [(ˆ] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] _[,]_ _t ∈_ Z _−._ (14)


The next lemma (with proof provided in Appendix C.2) shows that adding linear preprocessing maps
to reservoir equations can lead to the echo state property without contraction assumptions.

**Lemma 4.7.** _Let_ _F_ [˜] = ( _F_ [˜] 1 _, . . .,_ _F_ [˜] _N_ ) _be a reservoir map where each component_ _F_ [˜] _j_ : R _[N]_ _×_ R _[d]_ _→_ R _,_
_for j_ = 1 _, . . ., N_ _, is defined as_
_F_ ˜ _j_ ( _**x**_ _,_ _**z**_ ) = _gj_ ( _Pj_ _**x**_ _,_ _**z**_ ) (15)

_where P_ 1 _, . . ., PN_ _∈_ R _[N]_ _[×][N]_ _are linear preprocessing maps for any maps gj_ : R _[N]_ _×_ R _[d]_ _→_ R _, j_ =
1 _, . . ., N_ _._ _Define an arbitrary partition of the state vector_ _**x**_ ˆ _t_ = [ˆ _**x**_ [(1)] _t_ _[, . . .,]_ [ ˆ] _**[x]**_ [(] _t_ _[K]_ [)] ] _∈_ R _[I]_ [1] _×· · ·×_ R _[I][K]_


9


_such_ _that_ [�] _k_ _[K]_ =1 _[I][k]_ [=] _[N]_ _[>]_ [0] _[and]_ _[I][k]_ _[≥]_ [1] _[for]_ _[all]_ _[t]_ _[∈]_ [Z] _[−][.]_ _[We]_ _[define]_ _[the]_ _[index]_ _[l][k]_ [=] [�] _s_ _[k]_ =1 _[I][s]_ _[for]_
_k_ = 1 _, . . ., K._ _For k_ = 1 _, j_ _∈{_ 1 _, . . ., l_ 1 _}, and k_ = 2 _, . . ., K −_ 1 _, j_ _∈{lk−_ 1 + 1 _, . . ., lk}, select Pj_
_as the matrix with zero entries, except for_ ( _Pj_ ) _l,l_ + _lk_ = 1 _for l_ = 1 _, . . .,_ [�] _s_ _[K]_ = _k_ +1 _[I][s]_ _[and let][ P][j]_ [=] [0]
_for j_ = _lK−_ 1 + 1 _, . . ., N_ _._ _Then, the map_ _F_ [˜] _has the echo state property for any N_ _∈_ N [+] _._


Notice that Lemma 4.7 provides the echo state property by imposing a finite memory of _K −_ 1 time
steps on the reservoir. Let _Dd_ _⊂_ R _[d]_, _Bm_ _⊂_ R _[m]_ be compact. For a readout _W_ _∈_ R _[m][×][N]_, denote
**y** _t_ = _W_ _**x**_ _t_ (16)
the output process associated to the recurrent QNNs (4) and (14). Our next result proves universality
of RQNNs. The proof is provided in Appendix C.3.
**Theorem** **4.8.** _Let_ _U_ : ( _Dd_ ) [Z] _[−]_ _→_ ( _Bm_ ) [Z] _[−]_ _be_ _a_ _causal_ _and_ _time-invariant_ _filter_ _that_ _satisfies_ _the_
_fading memory property (that is, it is continuous with respect to the product topology). Then, for any_
_ε >_ 0 _there exist n, N_ _∈_ N _, preprocessing matrices P_ 1 _, . . ., PN_ _∈_ R _[N]_ _[×][N]_ _, a readout W_ _∈_ R _[m][×][N]_ _,_
_and circuit parameters_ _**θ**_ _∈_ **Θ** _[N]_ _such that the RQNN_ (14) _has the echo state property and the filter_
_U_ ¯ _W_ : ( _Dd_ ) [Z] _−_ _→_ ( _Bm_ ) [Z] _−_ _associated to the output process_ (16) _satisfies_
sup sup �� _U_ ( _**z**_ ) _t −_ _U_ ¯ _W_ ( _**z**_ ) _t_ �� _≤_ _ε._ (17)
_**z**_ _∈_ ( _Dd_ ) [Z] _[−]_ _t∈_ Z _−_


5 CONCLUSIONS


Approximation bounds and universality properties are part of the theoretical cornerstone of machine
learning models. While some studies have addressed the question of universality for QRC models,
the combination of the two had not previously been explored in the context of recurrent QNNs. In
this paper, we derived approximation bounds and universality statements for recurrent QNNs based
on the circuit implementation presented in Gonon & Jacquier (2025), which is compatible with
hardware deployment and whose implementation with Rydberg atoms has been already discussed
in Agarwal et al. (2024). This circuit uses a uniformly controlled quantum gate to apply multicontrolled rotations to a set of control and target qubits, and it has been recently shown that it can be
efficiently implemented (Zindorf & Bose, 2024; Silva et al., 2024; Zindorf & Bose, 2025).


To prove our results, we first derived approximation bounds for the static version of the QNN and
its derivatives. These results are used in Theorem 4.6 to provide filter approximation bounds that
show that RQNNs are able to uniformly approximate the filters induced by any contracting Barrontype state-space system. Finally, Theorem 4.8 extends this universality property to the much larger
category of arbitrary fading memory, causal, and time-invariant filters. In this last result, neither
Barron-type integrability nor contractivity conditions are needed for the target filter. While our
results apply to variational systems in which all parameters are trainable, they pave the way for
results on quantum reservoir systems in which some parameters in the recurrent layer are randomly
generated and only the output layer weights are tuned. Which strategy is best in terms of speed and
accuracy will depend on the number of blocks _n_ of the circuit, the intrinsic noise of the hardware,
and the target task. Future research will focus on implementing and comparing the variational and
reservoir approaches.


This work paves the way for extending the theoretical analysis of QRC models beyond the stateaffine system (SAS) paradigm (Mart´ınez-Pe˜na & Ortega, 2023). It is important to understand in
which situations the feedback approach is preferable to other protocols. Questions such as the
exponential concentration of observables (Sannia et al., 2025; Xiong et al., 2025) and the suitability
of QRC models for learning quantum temporal tasks (Tran & Nakajima, 2021; Nokkala, 2023)
are fundamental to discerning the conditions that render QRC models more useful than classical
machine learning approaches.


While our paper obtains approximation bounds for Barron-type sate-space systems, an important
direction of future research will consist in studying approximation error rates for systems with high
degrees of roughness or non-contractive dynamics. Furthermore, our paper focuses on approximation properties of RQNNs. Gradient-based training approaches for optimizing RQNN parameters
have been proposed, e.g., in Bausch (2020); Li et al. (2023); Siemaszko et al. (2023). Quantum circuit training may face _Barren plateaus_ McClean et al. (2018); Larocca et al. (2025), flat parameter
optimization landscapes for large number of qubits. Developing efficient training algorithms and
studying these effects in detail will be a further important direction for future research.


10


ACKNOWLEDGMENTS


The authors acknowledge partial financial support from the School of Physical and Mathematical
Sciences of the Nanyang Technological University through the SPMS Collaborative Research Award
2023 entitled “Quantum Reservoir Systems for Machine Learning”. RMP acknowledges the QCDI
project funded by the Spanish Government. JPO wishes to thank the hospitality of the Donostia
International Physics Center and LG and RMP that of the Division of Mathematical Sciences of
the Nanyang Technological University, during the academic visits in which some of this work was
developed.


REFERENCES


Junaid Aftab and Haizhao Yang. Approximating korobov functions via quantum circuits. _arXiv_
_preprint arXiv:2404.14570_, 2024.


Ishita Agarwal, Taylor L Patti, Rodrigo Araiza Bravo, Susanne F Yelin, and Anima Anandkumar.
Extending quantum perceptrons: Rydberg devices, multi-class classification, and error tolerance.
_arXiv preprint arXiv:2411.09093_, 2024.


Osama Ahmed, Felix Tennie, and Luca Magri. Optimal training of finitely sampled quantum reservoir computers for forecasting of chaotic dynamics. _Quantum Machine Intelligence_, 7(1):1–16,
2025.


Israel F Araujo, Ismael C Ara´ujo, Leon D da Silva, Carsten Blank, and Adenilton J da Silva. Quantum computing library. 2023. [https://github.com/qclib/qclib.](https://github.com/qclib/qclib)


Juan Miguel Arrazola, Olivia Di Matteo, Nicol´as Quesada, Soran Jahangiri, Alain Delgado, and
Nathan Killoran. Universal quantum circuits for quantum chemistry. _Quantum_, 6:742, 2022.


Andrew R. Barron. Neural net approximation. In _Yale Workshop on Adaptive and Learning Systems_,
volume 1, pp. 69–72, 1992.


Andrew R. Barron. Universal approximation bounds for superpositions of a sigmoidal function.
_IEEE Trans. Inform. Theory_, 39(3):930–945, 1993. ISSN 0018-9448. doi: 10.1109/18.256500.
[URL https://doi.org/10.1109/18.256500.](https://doi.org/10.1109/18.256500)


Andrew R. Barron and Jason M. Klusowski. Approximation and estimation for high-dimensional
deep learning networks. _Preprint, arXiv 1809.03090_, 2018.


Johannes Bausch. Recurrent quantum neural networks. _Advances in neural information processing_
_systems_, 33:1368–1379, 2020.


Ville Bergholm, Juha J Vartiainen, Mikko M¨ott¨onen, and Martti M Salomaa. Quantum circuits
with uniformly controlled one-qubit gates. _Physical Review A—Atomic, Molecular, and Optical_
_Physics_, 71(5):052330, 2005.


S. Boyd and L. Chua. Fading memory and the problem of approximating nonlinear operators with
Volterra series. _IEEE Transactions on Circuits and Systems_, 32(11):1150–1161, 1985.


Rodrigo Araiza Bravo, Khadijeh Najafi, Xun Gao, and Susanne F Yelin. Quantum reservoir computing using arrays of rydberg atoms. _PRX Quantum_, 3(3):030325, 2022.


Jiayin Chen and Hendra I Nurdin. Learning nonlinear input–output maps with dissipative quantum
systems. _Quantum Information Processing_, 18:1–36, 2019.


Jiayin Chen, Hendra I Nurdin, and Naoki Yamamoto. Temporal information processing on noisy
quantum computers. _Physical Review Applied_, 14(2):024065, 2020.


Naomi Mona Chmielewski, Nina Amini, and Joseph Mikael. Quantum reservoir computing and risk
bounds. _arXiv preprint arXiv:2501.08640_, 2025.


Saud Cindrak, [ˇ] Brecht Donvil, Kathy L¨udge, and Lina Jaurigue. Enhancing the performance of
quantum reservoir computing and solving the time-complexity problem by artificial memory restriction. _Physical Review Research_, 6(1):013051, 2024.


11


Samudra Dasgupta, Kathleen E Hamilton, and Arnab Banerjee. Characterizing the memory capacity
of transmon qubit reservoirs. In _2022 IEEE International Conference on Quantum Computing and_
_Engineering (QCE)_, pp. 162–166. IEEE, 2022.


Gerald B. Folland. _Introduction_ _to_ _Partial_ _Differential_ _Equations:_ _Second_ _Edition_ . Princeton University Press, 2020. ISBN 9780691213033. doi: doi:10.1515/9780691213033. URL
[https://doi.org/10.1515/9780691213033.](https://doi.org/10.1515/9780691213033)


Giacomo Franceschetto, Marcin Płodzie´n, Maciej Lewenstein, Antonio Ac´ın, and Pere Mujal. Harnessing quantum back-action for time-series processing. _arXiv preprint arXiv:2411.03979_, 2024.


Jorge Garc´ıa-Beni, Gian Luca Giorgi, Miguel C Soriano, and Roberta Zambrini. Scalable photonic
platform for real-time quantum reservoir computing. _Physical_ _Review_ _Applied_, 20(1):014051,
2023.


Lukas Gonon. Random feature neural networks learn black-scholes type pdes without curse of dimensionality. _J. Mach. Learn. Res._ [, 24(189):1–51, 2023. URL http://jmlr.org/papers/](http://jmlr.org/papers/v24/21-0987.html)
[v24/21-0987.html.](http://jmlr.org/papers/v24/21-0987.html)


Lukas Gonon. Deep neural network expressivity for optimal stopping problems. _Finance_ _and_
_Stochastics_, 28:865–910, 2024.


Lukas Gonon and Antoine Jacquier. Universal approximation theorem and error bounds for quantum
neural networks and quantum reservoirs. _Preprint, arXiv 2307.12904; to appear in IEEE TNNLS_,
2025.


Lukas Gonon and Juan-Pablo Ortega. Reservoir computing universality with stochastic inputs. _IEEE_
_Trans. Neural Netw. Learn. Syst._, 31(1):100–112, 2020. ISSN 2162-237X,2162-2388.


Lukas Gonon and Juan-Pablo Ortega. Fading memory echo state networks are universal. _Neural_
_Netw._, 138:10–13, 2021.


Lukas Gonon, Lyudmila Grigoryeva, and Juan-Pablo Ortega. Risk bounds for reservoir computing.
_J. Mach. Learn. Res._, 21:Paper No. 240, 61, 2020. ISSN 1532-4435,1533-7928.


Lukas Gonon, Lyudmila Grigoryeva, and Juan-Pablo Ortega. Approximation bounds for random neural networks and reservoir systems. _Ann._ _Appl._ _Probab._, 33(1):28–69, 2023. ISSN
1050-5164,2168-8737. doi: 10.1214/22-aap1806. URL [https://doi.org/10.1214/](https://doi.org/10.1214/22-aap1806)
[22-aap1806.](https://doi.org/10.1214/22-aap1806)


Lyudmila Grigoryeva and Juan-Pablo Ortega. Universal discrete-time reservoir computers with
stochastic inputs and linear readouts using non-homogeneous state-affine systems. _Journal_ _of_
_Machine_ _Learning_ _Research_, 19(24):1–40, 2018a. URL [http://arxiv.org/abs/1712.](http://arxiv.org/abs/1712.00754)
[00754.](http://arxiv.org/abs/1712.00754)


Lyudmila Grigoryeva and Juan-Pablo Ortega. Echo state networks are universal. _Neural Networks_,
108:495–508, 2018b.


Lars H¨ormander. _The_ _analysis_ _of_ _linear_ _partial_ _differential_ _operators_ _I_ . Springer, second edition
edition, 1990.


Kurt Hornik. Approximation capabilities of muitilayer feedforward networks. _Neural Networks_, 4
(1989):251–257, 1991. doi: 10.1016/0893-6080(91)90009-T.


Fangjun Hu, Saeed A Khan, Nicholas T Bronn, Gerasimos Angelatos, Graham E Rowlands, Guilhem J Ribeill, and Hakan E T¨ureci. Overcoming the coherence time barrier in quantum machine
learning on temporal data. _Nature Communications_, 15(1):7491, 2024.


Kaito Kobayashi, Keisuke Fujii, and Naoki Yamamoto. Feedback-driven quantum reservoir computing for time-series analysis. _PRX Quantum_, 5(4):040325, 2024.


Milan Kornjaˇca, Hong-Ye Hu, Chen Zhao, Jonathan Wurtz, Phillip Weinberg, Majd Hamdan, Andrii
Zhdanov, Sergio H Cantu, Hengyun Zhou, Rodrigo Araiza Bravo, et al. Large-scale quantum
reservoir learning with an analog quantum computer. _arXiv preprint arXiv:2407.02553_, 2024.


12


Tomoyuki Kubota, Yudai Suzuki, Shumpei Kobayashi, Quoc Hoan Tran, Naoki Yamamoto, and
Kohei Nakajima. Temporal information processing induced by quantum noise. _Physical Review_
_Research_, 5(2):023057, 2023.


Martin Larocca, Supanut Thanasilp, Samson Wang, Kunal Sharma, Jacob Biamonte, Patrick J Coles,
Lukasz Cincio, Jarrod R McClean, Zo¨e Holmes, and Marco Cerezo. Barren plateaus in variational
quantum computing. _Nature Reviews Physics_, pp. 1–16, 2025.


Michel Ledoux and Michel Talagrand. _Probability in Banach Spaces_ . Springer Berlin Heidelberg,
2013.


Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng,
Guoqiang Zhong, and Yongjian Gu. Quantum recurrent neural networks for sequential learning.
_Neural Networks_, 166:148–161, 2023.


Chen-Yu Liu, En-Jui Kuo, Chu-Hsuan Abraham Lin, Jason Gemsun Young, Yeong-Jar Chang, MinHsiu Hsieh, and Hsi-Sheng Goan. Quantum-train: Rethinking hybrid quantum-classical machine
learning in the model compression perspective. _Quantum Machine Intelligence_, 7(2):80, 2025.


G Manjunath. Stability and memory-loss go hand-in-hand: three results in dynamics _\_ & computation. _Proceedings of the Royal Society London Ser. A Math. Phys. Eng. Sci._, 476(2242):1–25,
2020. doi: 10.1098/rspa.2020.0563. [URL http://arxiv.org/abs/2001.00766.](http://arxiv.org/abs/2001.00766)


Rodrigo Mart´ınez-Pe˜na and Juan-Pablo Ortega. Quantum reservoir computing in finite dimensions.
_Physical Review E_, 107(3):035306, 2023.


Jarrod R McClean, Sergio Boixo, Vadim N Smelyanskiy, Ryan Babbush, and Hartmut Neven. Barren plateaus in quantum neural network training landscapes. _Nature communications_, 9(1):4812,
2018.


Eduardo Miranda and Hari Shaji. A quantum reservoir computing approach to computer-aided
music composition. _Academia Quantum_, 2(2), 2025.


Zoubeir Mlika, Soumaya Cherkaoui, Jean Fr´ed´eric Laprade, and Simon Corbeil-Letourneau. User
trajectory prediction in mobile wireless networks using quantum reservoir computing. _IET Quan-_
_tum Communication_, 4(3):125–135, 2023.


Riccardo Molteni, Claudio Destri, and Enrico Prati. Optimization of the memory reset rate of a
quantum echo-state network for time sequential tasks. _Physics Letters A_, pp. 128713, 2023.


Tomoya Monomi, Wataru Setoyama, and Yoshihiko Hasegawa. Feedback-enhanced quantum reservoir computing with weak measurements. _arXiv preprint arXiv:2503.17939_, 2025.


Mikko M¨ott¨onen, Juha J Vartiainen, Ville Bergholm, and Martti M Salomaa. Quantum circuits for
general multiqubit gates. _Physical review letters_, 93(13):130502, 2004.


Mikko Mottonen, Juha J Vartiainen, Ville Bergholm, and Martti M Salomaa. Transformation of
quantum states using uniformly controlled rotations. _arXiv preprint quant-ph/0407010_, 2004.


Pere Mujal, Rodrigo Mart´ınez-Pe˜na, Johannes Nokkala, Jorge Garc´ıa-Beni, Gian Luca Giorgi,
Miguel C Soriano, and Roberta Zambrini. Opportunities in quantum reservoir computing and
extreme learning machines. _Advanced Quantum Technologies_, 4(8):2100027, 2021.


Pere Mujal, Rodrigo Mart´ınez-Pe˜na, Gian Luca Giorgi, Miguel C Soriano, and Roberta Zambrini.
Time-series quantum reservoir computing with weak and projective measurements. _npj Quantum_
_Information_, 9(1):16, 2023.


Jakob Murauer, Rajiv Krishnakumar, Sabine Tornow, and Michaela Geierhos. Feedback connections in quantum reservoir computing with mid-circuit measurements. _arXiv_ _preprint_
_arXiv:2503.22380_, 2025.


Johannes Nokkala. Online quantum time series processing with random oscillator networks. _Scien-_
_tific Reports_, 13(1):7694, 2023.


13


Johannes Nokkala, Rodrigo Mart´ınez-Pe˜na, Gian Luca Giorgi, Valentina Parigi, Miguel C Soriano,
and Roberta Zambrini. Gaussian states of continuous-variable quantum systems provide universal
and versatile reservoir computing. _Communications Physics_, 4(1):53, 2021.


Juan-Pablo Ortega and Florian Rossmannek. State-space systems as dynamic generative models.
_Proceedings_ _of_ _the_ _Royal_ _Society_ _A_, 481(2309):20240308, 2025a. ISSN 1471-2946. doi: 10.
1098/rspa.2024.0308. [URL https://doi.org/10.1098/rspa.2024.0308.](https://doi.org/10.1098/rspa.2024.0308)


Juan-Pablo Ortega and Florian Rossmannek. Stochastic dynamics learning with state-space systems.
_Preprint_, 2025b.


Juan-Pablo Ortega and Florian Rossmannek. Echoes of the past: a unified perspective on fading
memory and echo states. _Preprint_, 2025c.


Iris Paparelle, Johan Henaff, Jorge Garcia-Beni, Emilie Gillet, Gian Luca Giorgi, Miguel C Soriano,
Roberta Zambrini, and Valentina Parigi. Experimental memory control in continuous variable
optical quantum reservoir computing. _arXiv preprint arXiv:2506.07279_, 2025.


Daniel K Park, Francesco Petruccione, and June-Koo Kevin Rhee. Circuit-based quantum random
access memory for classical data. _Scientific reports_, 9(1):3949, 2019.


Adri´an P´erez-Salinas, Alba Cervera-Lierta, Elies Gil-Fuster, and Jos´e I. Latorre. Data re-uploading
for a universal quantum classifier. _Quantum_, 4:226, February 2020. ISSN 2521-327X. doi: 10.
22331/q-2020-02-06-226. [URL https://doi.org/10.22331/q-2020-02-06-226.](https://doi.org/10.22331/q-2020-02-06-226)


Philipp Pfeffer, Florian Heyder, and J¨org Schumacher. Hybrid quantum-classical reservoir computing of thermal convection flow. _Physical Review Research_, 4(3):033176, 2022.


Philipp Pfeffer, Florian Heyder, and J¨org Schumacher. Reduced-order modeling of two-dimensional
turbulent rayleigh-b´enard flow by hybrid quantum-classical reservoir computing. _Physical review_
_research_, 5(4):043242, 2023.


Jun Qi, Chao-Han Huck Yang, Pin-Yu Chen, and Min-Hsiu Hsieh. Theoretical error performance
analysis for variational quantum circuit based functional regression. _npj Quantum Information_, 9
(1):4, 2023.


Antonio Sannia, Gian Luca Giorgi, Stefano Longhi, and Roberta Zambrini. Skin effect in quantum
neural networks. _arXiv preprint arXiv:2406.14112_, 2024a.


Antonio Sannia, Rodrigo Mart´ınez-Pe˜na, Miguel C Soriano, Gian Luca Giorgi, and Roberta Zambrini. Dissipation as a resource for quantum reservoir computing. _Quantum_, 8:1291, 2024b.


Antonio Sannia, Gian Luca Giorgi, and Roberta Zambrini. Exponential concentration and symmetries in quantum reservoir computing. _arXiv preprint arXiv:2505.10062_, 2025.


Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. Effect of data encoding on the expressive power of variational quantum-machine-learning models. _Phys._ _Rev._ _A_, 103:032430, Mar
2021. doi: 10.1103/PhysRevA.103.032430. URL [https://link.aps.org/doi/10.](https://link.aps.org/doi/10.1103/PhysRevA.103.032430)
[1103/PhysRevA.103.032430.](https://link.aps.org/doi/10.1103/PhysRevA.103.032430)


Mirela Selimovi´c, Iris Agresti, Michał Siemaszko, Joshua Morris, Borivoje Daki´c, Riccardo Albiero,
Andrea Crespi, Francesco Ceccarelli, Roberto Osellame, Magdalena Stobi´nska, et al. Experimental neuromorphic computing based on quantum memristor. _arXiv_ _preprint_ _arXiv:2504.18694_,
2025.


Michał Siemaszko, Adam Buraczewski, Bertrand Le Saux, and Magdalena Stobi´nska. Rapid training of quantum recurrent neural networks. _Quantum Machine Intelligence_, 5(2):31, 2023.


Jefferson DS Silva, Thiago Melo D Azevedo, Israel F Araujo, and Adenilton J da Silva. Linear decomposition of approximate multi-controlled single qubit gates. _IEEE_ _Transactions_ _on_
_Computer-Aided Design of Integrated Circuits and Systems_, 2024.


Michele Spagnolo, Joshua Morris, Simone Piacentini, Michael Antesberger, Francesco Massa, Andrea Crespi, Francesco Ceccarelli, Roberto Osellame, and Philip Walther. Experimental photonic
quantum memristor. _Nature Photonics_, 16(4):318–323, 2022.


14


Yudai Suzuki, Qi Gao, Ken C Pradel, Kenji Yasuoka, and Naoki Yamamoto. Natural quantum
reservoir computing for temporal information processing. _Scientific reports_, 12(1):1–15, 2022.


Quoc Hoan Tran and Kohei Nakajima. Learning temporal quantum tomography. _Physical_ _review_
_letters_, 127(26):260401, 2021.


Hassler Whitney. Analytic extensions of differentiable functions defined in closed sets. _Transactions_
_of the American Mathematical Society_, 36(1):63–89, 1934.


Weijie Xiong, Zo¨e Holmes, Armando Angrisani, Yudai Suzuki, Thiparat Chotibut, and Supanut
Thanasilp. Role of scrambling and noise in temporal information processing with quantum systems. _arXiv preprint arXiv:2505.10080_, 2025.


Dimitry Yarotsky. Error bounds for approximations with deep ReLU networks. _Neural Networks_,
94:103–114, 2017.


Toshiki Yasuda, Yudai Suzuki, Tomoyuki Kubota, Kohei Nakajima, Qi Gao, Wenlong Zhang,
Satoshi Shimono, Hendra I Nurdin, and Naoki Yamamoto. Quantum reservoir computing with
repeated measurements on superconducting devices. _arXiv preprint arXiv:2310.06706_, 2023.


Zhan Yu, Qiuhao Chen, Yuling Jiao, Yinan Li, Xiliang Lu, Xin Wang, and Jerry Zhijian Yang.
Non-asymptotic approximation error bounds of parameterized quantum circuits. In _The_ _Thirty-_
_eighth_ _Annual_ _Conference_ _on_ _Neural_ _Information_ _Processing_ _Systems_, 2024. URL [https://](https://openreview.net/forum?id=XCkII8nCt3)
[openreview.net/forum?id=XCkII8nCt3.](https://openreview.net/forum?id=XCkII8nCt3)


Ben Zindorf and Sougato Bose. Efficient implementation of multi-controlled quantum gates. _arXiv_
_preprint arXiv:2404.02279_, 2024.


Ben Zindorf and Sougato Bose. Multi-controlled quantum gates in linear nearest neighbor. _arXiv_
_preprint arXiv:2506.00695_, 2025.


APPENDIX


A QUANTUM RESERVOIR COMPUTING PROTOCOLS


For learning problems with temporal structure, quantum reservoir computing (QRC) has emerged as
a promising approach for exploiting noisy intermediate-scale quantum (NISQ) technologies. These
include ion traps, nuclear magnetic resonance, cold atoms, photonic platforms, and superconducting
qubits (Mujal et al., 2021). When implementing QRC models experimentally, it is necessary to
consider the backaction and statistical effects introduced by quantum measurements. Backaction
refers to the modification of a quantum state after monitoring, also known as wavefunction collapse.
Due to the probabilistic nature of quantum theory, measurements must be repeated to compute the
expected values of observables, which introduces a statistical component in all these methodologies.
Most available experimental implementations rely on the quantum computer paradigm (Dasgupta
et al., 2022; Mlika et al., 2023; Suzuki et al., 2022; Yasuda et al., 2023; Chen et al., 2020; Kubota
et al., 2023; Molteni et al., 2023; Pfeffer et al., 2022; Ahmed et al., 2025; Hu et al., 2024; Miranda
& Shaji, 2025). However, there is an increasing interest in extending this technique to new settings,
such as optical pulses (Garc´ıa-Beni et al., 2023; Paparelle et al., 2025), Rydberg atoms (Bravo et al.,
2022; Kornjaˇca et al., 2024), and quantum memristors (Spagnolo et al., 2022; Selimovi´c et al., 2025).


Early QRC model implementations relied on the simplest possible approach, namely, the restarting
protocol (Dasgupta et al., 2022; Suzuki et al., 2022; Kubota et al., 2023; Chen et al., 2020; Molteni
et al., 2023). In this approach, the expected values of observables are obtained by rerunning the
algorithm from the first time step at each subsequent time step. This avoids the backaction effect
of quantum measurements. However, the complexity of this protocol scales quadratically with the
length of the input sequence, making it very time-consuming. A faster alternative is the rewinding
protocol (Mujal et al., 2021; Cindrak et al., 2024), where the fading memory of the quantum reservoir [ˇ]
is exploited to restart the algorithm with a fixed window of past time steps. This reduces the complexity of the algorithm to linear in terms of input length. Originally proposed in Chen et al. (2020),
this protocol has thus far only been considered numerically (Mujal et al., 2023; Cindrak et al., 2024). [ˇ]


15


Both the restarting and rewinding protocols use repetition of previous time steps to reproduce the
dynamics of the theoretical model and avoid the disruptive effect of projective measurements used
to extract output information. This comes at the cost of halting the quantum dynamics at each time
step and the need to buffer the input sequence. Consequently, these approaches lack one of the most
important features of traditional reservoir computing, namely, the ability to process information in
real time.


New protocols have been proposed to circumvent this problem. The online protocol (Mujal et al.,
2023; Franceschetto et al., 2024) uses weak measurements to find a balance between erasing and
extracting information. Mid-circuit measurements and reset operations (Hu et al., 2024) can split
the reservoir into two parts: memory and readout. The memory retains previous inputs, while measurements only affect the readout part. The feedback protocol (Kobayashi et al., 2024), which can be
traced back to QRC with quantum memristors (Spagnolo et al., 2022) and hybrid QRC techniques
(Pfeffer et al., 2022; 2023), reinjects the measured observables at each time step as parameters of
an input quantum channel. This ensures that no backaction effects are present and that past input
information is preserved. Note that in order to compute the observables in real time, these protocols all require several copies of the system to be run in parallel. Furthermore, these protocols can
be combined with each other. For instance, the feedback protocol has been combined with both
the online protocol (Monomi et al., 2025) and with mid-circuit measurements and reset operations
(Murauer et al., 2025).


Of all these approaches, the feedback protocol presents some particularly interesting features. First,
the feedback protocol enables us to compute the expected values of observables from a single copy
of the system by repeating one time step only. If only a few copies of the system are available, this
reduces the experimental time overhead for real-time applications compared to other approaches.
Second, in contrast to previous QRC models, where an erasure mechanism is added to provide fundamental properties such as the echo state property, simple unitary operations can provide these
properties (Kobayashi et al., 2024). Finally, the dynamical equations of quantum reservoirs under
the feedback protocol go beyond the standard state-affine system (SAS) paradigm of QRC models
(Mart´ınez-Pe˜na & Ortega, 2023). These properties make the feedback protocol a promising candidate for exploring QRC applications.


B PROOFS FOR SECTION 4.1


B.1 PROOF OF PROPOSITION 4.1


_Proof._ The proof is a modification of the argument used to obtain (Gonon & Jacquier, 2025, Proposition 1). Recall that


_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) :=] _[ R][ −]_ [2] _[R]_ [[][P] 1 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ ) + P2 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ )] _,_ ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ _._ (18)


Fix ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ and _j_ _∈{_ 1 _, . . ., N_ _}_ and write P _m_ := P _m_ _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ ) for _m_ _∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_ . To
prove the representation (7), let us first calculate P _m_ .


As a first step, write


1
`UV` _|_ 0 _⟩_ _[⊗]_ [n] = `U` _|ψ⟩_ = ~~_√_~~
_n_


3


_k_ =0


_n−_ 1

- `U` _|_ 4 _l⟩_


_l_ =0


- `U` [(] 1 _[l]_ [+1)] _⊗_ `U` [(] 2 _[l]_ [+1)]


16


1
= ~~_√_~~
_n_


_n−_ 1


_l_ =0


_k_ +1 _,_ 1 _[|]_ [4] _[l]_ [ +] _[ k][⟩]_ _[.]_


Thus, for _m ∈{_ 0 _,_ 1 _,_ 2 _,_ 3 _}_, we have


P _m_ =


_n−_ 1

- �� _⟨_ 4 _i_ + _m|_ `UV` _|_ 0 _⟩⊗_ n��2


_i_ =0


_n−_ 1


_l_ =0


1
_⟨_ 4 _i_ + _m|_ ~~_√_~~
_n_
�����


3

- - `U` [(] 1 _[l]_ [+1)] _⊗_ `U` [(] 2 _[l]_ [+1)] 

_k_ =0


�����


2


=


_n−_ 1


_i_ =0


_k_ +1 _,_ 1 _[|]_ [4] _[l]_ [ +] _[ k][⟩]_


2
_._

_m_ +1 _,_ 1����


= [1]

_n_


_n−_ 1


_i_ =0


 - `U` [(] 1 _[i]_ [+1)] _⊗_ `U` [(] 2 _[i]_ [+1)]
����


Next, we may calculate

              - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][1] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][1] _[,]_ [1][[] `[U]` [(] 2 _[i]_ [)][]][1] _[,]_ [1] [= cos]

2

              - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][2] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][1] _[,]_ [1][[] `[U]` 2 [(] _[i]_ [)][]][2] _[,]_ [1] [= sin]

2


- - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
cos
2

- - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
cos
2


_,_


_,_


            - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][3] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][2] _[,]_ [1][[] `[U]` [(] 2 _[i]_ [)][]][1] _[,]_ [1] [= i cos]

2


            - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][3] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][2] _[,]_ [1][[] `[U]` [(] 2 _[i]_ [)][]][1] _[,]_ [1] [= i cos]

2

            - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][4] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][2] _[,]_ [1][[] `[U]` [(] 2 _[i]_ [)][]][2] _[,]_ [1] [= i sin]

2


            - _γi,j_

[ `U` [(] 1 _[i]_ [)] _[⊗]_ `[U]` 2 [(] _[i]_ [)][]][4] _[,]_ [1] [= [] `[U]` [(] 1 _[i]_ [)][]][2] _[,]_ [1][[] `[U]` [(] 2 _[i]_ [)][]][2] _[,]_ [1] [= i sin]


- - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
sin
2

- - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
sin
2


_,_


_,_


and thus


P0 = [1]

_n_


_n_


- cos - _γi,j_

2

_i_ =1


2


P1 = [1]

_n_


_n_


- sin - _γi,j_

2

_i_ =1


2


�2


�2


�2


P2 = [1]

_n_


_n_


- cos - _γi,j_

2

_i_ =1


2


�2 - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
cos
2


�2 - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
cos
2


�2 - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
sin
2


�2 - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )
sin
2


�2
_._


P3 = [1]

_n_


_n_


- sin - _γi,j_

2

_i_ =1


2


Therefore, using cos( _y_ ) [2] = [cos(2] 2 _[y]_ [)+1], we obtain


2 _n_


P0 + P1 = [1]

_n_


- _n_ cos - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )

2
_i_ =1


- _n_ cos - _bi,j_ + _**a**_ _i,j_ _·_ ( _**x**_ _,_ _**z**_ )

2
_i_ =1


�2
= [1]


[1] [1]

2 [+] 2 _n_


_n_

- cos - _b_ _[i,j]_ + _**a**_ _[i,j]_ _·_ ( _**x**_ _,_ _**z**_ )� _,_


_i_ =1


2 _n_


�2
= [1]


P0 + P2 = [1]

_n_


_n_


- cos - _γi,j_

2

_i_ =1


2


[1] [1]

2 [+] 2 _n_


_n_

- cos - _γ_ _[i,j]_ [�] _._


_i_ =1


Putting it all together we obtain, for any given _R >_ 0, that

_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) =] _[ R][ −]_ [2] _[R]_ [[][P] 1 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ ) + P2 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ )]

= _R_ [1 + 4P0 _−_ 2 (P0 + P1) _−_ 2 (P0 + P2)]


= [1]

_n_


_n_

- _R_ cos - _γ_ _[i,j]_ [�] cos - _b_ _[i,j]_ + _**a**_ _[i,j]_ _·_ ( _**x**_ _,_ _**z**_ )� _._


_i_ =1


B.2 PROOF OF PROPOSITION 4.2


_Proof._ Let _j_ _∈{_ 1 _, . . ., N_ _}_ be fixed. As in the proof of Proposition 2 in Gonon & Jacquier (2025),
we may use the Fourier inversion theorem to represent


             _Fj_ ( _**x**_ _,_ _**z**_ ) = _Fj_ ( _**ξ**_ 1 _,_ _**ξ**_ 2)d _**ξ**_ 1d _**ξ**_ 2 _,_

R _[N]_ _×_ R _[d][ e]_ [2] _[π]_ [i(] _**[x]**_ _[,]_ _**[z]**_ [)] _[·]_ [(] _**[ξ]**_ [1] _[,]_ _**[ξ]**_ [2][)][�]


17


which we may rewrite as, with _**ξ**_ = ( _**ξ**_ 1 _,_ _**ξ**_ 2),


     _Fj_ ( _**x**_ _,_ _**z**_ ) =

R _[N]_ _×_ R _[d]_


- Im[ _F_ [�] _j_ ( _**ξ**_ )] d _**ξ**_


(19)


- cos (2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ )Re[ _F_ [�] _j_ ( _**ξ**_ )] + cos 2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ + _[π]_

2


- cos (2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ )Re[ _F_ [�] _j_ ( _**ξ**_ )] + cos 2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ + _[π]_


The hypothesis _∂iFj_ _∈F_ implies that �R _[N]_ _×_ R _[d][ |][ξ][i][||]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _< ∞_ . Hence, applying differentiation

under the integral sign yields


       _∂iFj_ ( _**x**_ _,_ _**z**_ ) = _−_ 2 _π_

R _[N]_ _×_ R _[d]_


- _ξi_ sin (2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ )Re[ _F_ [�] _j_ ( _**ξ**_ )] + _ξi_ sin 2 _π_ ( _**x**_ _,_ _**z**_ ) _·_ _**ξ**_ + _[π]_

2


- Im[ _F_ [�] _j_ ( _**ξ**_ )] d _**ξ**_ _._


(20)


Next, consider the random function


Φ _j_ ( _**x**_ _,_ _**z**_ ) := [1]

_n_


_n_

- _Wi_ cos( _Bi_ + **A** _i ·_ ( _**x**_ _,_ _**z**_ )) (21)


_i_ =1


for randomly selected weights _W_ 1 _, . . ., Wn_, _B_ 1 _, . . ., Bn_ and **A** 1 _, . . .,_ **A** _n_ valued in R, R, and
R _[N]_ _×_ R _[d]_, respectively (for notational simplicity we leave the dependence on _j_ implicit here). The
distributions of these random variables are chosen as follows. First, we let _Z_ 1 _, . . ., Zn_ be i.i.d.
Bernoulli random variables with


R _[N]_ _×_ R _[d]_ _[|]_ [Im[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_

_._ (22)

�R _[N]_ _×_ R _[d][ |]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_


P( _Zi_ = 1) =


R _[N]_ _×_ R _[d]_ _[|]_ [Re[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_

_,_ P( _Zi_ = 0) =

�R _[N]_ _×_ R _[d][ |]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_


R _[N]_ _×_ R _[d]_ _[|]_ [Re[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_

 


and let _ν_ Re and _ν_ Im be the probability measures on R _[N]_ _×_ R _[d]_ with densities


_|_ Re[ _F_ [�] _j_ ] _|_


_|_ Re[ _F_ [�] _j_ ] _|_ _|_ Im[ _F_ [�] _j_ ] _|_

and
R _[N]_ _×_ R _[d][ |]_ [Re[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_ �R _[N]_ _×_ R _[d][ |]_ [Im[] _F_ [�] _j_


_,_ (23)
R _[N]_ _×_ R _[d][ |]_ [Im[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_


respectively. In case �R _[N]_ _×_ R _[d][ |]_ [Re[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_ = 0, instead we choose for _ν_ Re an arbitrary probability

measure and analogously for _ν_ Im in case �R _[N]_ _×_ R _[d][ |]_ [Im[] _F_ [�] _j_ ( _**ξ**_ )] _|_ d _**ξ**_ = 0. Next, let **U** [Re] 1 _[, . . .,]_ **[ U]** _n_ [Re]

(resp. **U** [Im] 1 _[, . . .,]_ **[ U]** _n_ [Im][) be i.i.d. random variables with distribution] _[ ν]_ [Re] [(resp.] _[ ν]_ [Im][) and assume that]
**U** [Im] 1 _[. . .,]_ **[ U]** _n_ [Im][,] **[U]** [Re] 1 _[, . . .,]_ **[ U]** _n_ [Re] _[, Z]_ [1] _[, . . ., Z][n]_ [are] [independent.] [With] [these] [preparations,] [we] [are] [now]
ready to define the weights in (21):


**A** _i_ := 2 _π_ ( _Zi_ **U** [Re] _i_ + (1 _−_ _Zi_ ) **U** [Im] _i_ [)] _[,]_ _Bi_ := _[π]_

2 [(1] _[ −]_ _[Z][i]_ [)] _[,]_


(1 _−_ _Zi_ )
_|_ Im[ _F_ [�] _j_ ]( **U** [Im] _i_ [)] _[|]_


_,_


_Wi_ := _∥F_ [�] _j∥_ 1


Re[ _F_ [�] _j_ ]( **U** [Re] _i_ [)]


Re[ _F_ [�] _j_ ]( **U** [Re] _i_ [)] _Fj_ ]( **U** [Im] _i_ [)]

_Zi_ + [Im[][�]
_|_ Re[ _F_ [�] _j_ ]( **U** [Re] _i_ [)] _[|]_ _|_ Im[ _F_ [�] _j_ ]( **U** [Im] _i_ [)]


with the quotient set to zero when the denominator is null.


Our goal now is to estimate


  -   = E _∥Fj_ _−_ Φ _j∥_ [2] _L_ [2] ( _µ_ ) +


_N_ + _d_


- - 
E _∥∂iFj_ _−_ _∂i_ Φ _j∥_ [2] _L_ [2] ( _µ_ )
_i_ =1


_N_ + _d_

- _∥∂iFj_ _−_ _∂i_ Φ _j∥_ [2] _L_ [2] ( _µ_ )

_i_ =1


E


_∥Fj_ _−_ Φ _j∥_ [2] _L_ [2] ( _µ_ ) [+]


(24)


18


by estimating the summands separately. To achieve this, we first compute E[Φ _j_ ( _**x**_ _,_ _**z**_ )] and
E[ _∂i_ Φ _j_ ( _**x**_ _,_ _**z**_ )]. Indeed, inserting the definitions, using independence and representation (19) yields


E[Φ _j_ ( _**x**_ _,_ _**z**_ )] = E[ _W_ 1 cos( _B_ 1 + **A** 1 _·_ ( _**x**_ _,_ _**z**_ ))]


(1 _−_ _Z_ 1)
_|_ Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)] _[|]_


= _∥F_ [�] _j∥_ 1E


��
Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)]


Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)] _Fj_ ]( **U** [Im] 1 [)]

_Z_ 1 + [Im[][�]
_|_ Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)] _[|]_ _|_ Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)]


  - _π_ ��
cos 1 + (1 _−_ _Z_ 1) **U** [Im] _i_ [)] _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [)]
2 [(1] _[ −]_ _[Z]_ [1][) + 2] _[π]_ [(] _[Z]_ [1] **[U]** [Re]


[�] _j_ 1 cos(2 _π_ **U** [Re] 1 _·_ ( _**x**_ _,_ _**z**_ ))

_|_ Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)] _[|]_


= _∥F_ [�] _j∥_ 1


P( _Z_ 1 = 1)E


Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)]


+P( _Z_ 1 = 0)E


Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)]  - _π_  - [��]

_|_ Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)] _[|]_ cos 2 [+ 2] _[π]_ **[U]** 1 [Im] _·_ ( _**x**_ _,_ _**z**_ )


 =


               _Fj_ ]( _**ξ**_ ) cos(2 _π_ _**ξ**_ _·_ ( _**x**_ _,_ _**z**_ ))d _**ξ**_ +
R _[N]_ _×_ R _[d]_ [ Re[][�]


_Fj_ ]( _**ξ**_ ) cos( _[π]_
R _[N]_ _×_ R _[d]_ [ Im[][�] 2


2 [+ 2] _[π]_ _**[ξ]**_ _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [))d] _**[ξ]**_


= _Fj_ ( _**x**_ _,_ _**z**_ ) _._


Analogously, using the representation (20) for the partial derivative _∂iFj_ instead, we obtain


E[ _∂i_ Φ _j_ ( _**x**_ _,_ _**z**_ )] = _−_ E[ _W_ 1 _A_ 1 _,i_ sin( _B_ 1 + **A** 1 _·_ ( _**x**_ _,_ _**z**_ ))]


Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)]


[�] _j_ 1 _U_ 1 [Re] _,i_ [sin(2] _[π]_ **[U]** 1 [Re] _·_ ( _**x**_ _,_ _**z**_ ))

_|_ Re[ _F_ [�] _j_ ]( **U** [Re] 1 [)] _[|]_


= _−_ 2 _π∥F_ [�] _j∥_ 1


P( _Z_ 1 = 1)E


+P( _Z_ 1 = 0)E


Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)]  - _π_  - [��]

_U_ 1 [Im] _,i_ [sin] 1 _·_ ( _**x**_ _,_ _**z**_ )
_|_ Im[ _F_ [�] _j_ ]( **U** [Im] 1 [)] _[|]_ 2 [+ 2] _[π]_ **[U]** [Im]


��
= _−_ 2 _π_


_Fj_ ]( _**ξ**_ ) sin( _[π]_
R _[N]_ _×_ R _[d][ ξ][i]_ [Im[][�] 2


                _Fj_ ]( _**ξ**_ ) sin(2 _π_ _**ξ**_ _·_ ( _**x**_ _,_ _**z**_ ))d _**ξ**_ +
R _[N]_ _×_ R _[d][ ξ][i]_ [Re[][�]


_[π]_ 
2 [+ 2] _[π]_ _**[ξ]**_ _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [))d] _**[ξ]**_


= _∂iFj_ ( _**x**_ _,_ _**z**_ ) _._
(25)
Therefore, we may estimate the first expectation in (24) as follows:


 -  - ��
E _∥Fj_ _−_ Φ _j∥_ [2] _L_ [2] ( _µ_ ) = E


                -                =
R _[N]_ _×_ R _[d][ |][F][j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ [Φ] _[j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [2] _[µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]


R _[N]_ _×_ R _[d]_ [ V][[Φ] _[j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)]] _[µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]


_µ_ (d _**x**_ _,_ d _**z**_ )


- _n_


_Wi_ cos( _Bi_ + **A** _i ·_ ( _**x**_ _,_ _**z**_ ))

_i_ =1


= [1]

_n_ [2]


R _[N]_ _×_ R _[d]_ [ V]


= [1]

_n_


_≤_ [1]

_n_


R _[N]_ _×_ R _[d]_ [ V][ [] _[W]_ [1][ cos(] _[B]_ [1][ +] **[ A]** [1] _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [))]] _[ µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]

- 
( _W_ 1 cos( _B_ 1 + **A** 1 _·_ ( _**x**_ _,_ _**z**_ ))) [2][�] _µ_ (d _**x**_ _,_ d _**z**_ )
R _[N]_ _×_ R _[d]_ [ E]


_≤_ [1]


_n_ [1] [E] - _W_ 1 [2] - = _n_ [1]


_Fj∥_ [2] 1 _[.]_
_n_ _[∥]_ [�]
(26)


19


For the partial derivatives, we obtain analogously


      -       -       E _∥∂iFj_ _−_ _∂i_ Φ _j∥_ [2] _L_ [2] ( _µ_ ) =

R _[N]_ _×_ R _[d]_ [ V][[] _[∂][i]_ [Φ] _[j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)]] _[µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]


_µ_ (d _**x**_ _,_ d _**z**_ )


_WkAk,i_ sin( _Bk_ + **A** _k ·_ ( _**x**_ _,_ _**z**_ ))

_k_ =1


= [1]

_n_ [2]


R _[N]_ _×_ R _[d]_ [ V]


- _n_

 


= [1]

_n_


(27)


_≤_ [1]

_n_


R _[N]_ _×_ R _[d]_ [ V][ [] _[W]_ [1] _[A]_ [1] _[,i]_ [ sin(] _[B]_ [1][ +] **[ A]** [1] _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [))]] _[ µ]_ [(d] _**[x]**_ _[,]_ [ d] _**[z]**_ [)]

- 
( _W_ 1 _A_ 1 _,i_ sin( _B_ 1 + **A** 1 _·_ ( _**x**_ _,_ _**z**_ ))) [2][�] _µ_ (d _**x**_ _,_ d _**z**_ )
R _[N]_ _×_ R _[d]_ [ E]


[1] - _W_ 1 [2] _[A]_ 1 [2] _,i_ - = [1]

_n_ [E] _n_


_Fj∥_ 1
_n_ _[∥]_ [�]


_≤_ [1]


     -      _n_ [1] _[∥]_ _F_ [�] _j∥_ [2] 1 [E] _A_ [2] 1 _,i_ = [4] _n_ _[π]_ [2]


_i_ _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _,_
R _[N]_ _×_ R _[d][ ξ]_ [2]


where we used that E - _A_ [2] 1 _,i_ - = 4 _π_ [2] _∥F_ [�] _j∥_ _[−]_ 1 [1] 


R _[N]_ _×_ R _[d][ ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ .


In particular, (26) and (27) imply that there exists a scenario _ω_ _∈_ Ω such that Φ _[ω]_ _j_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] [=]
1 - _n_
_n_ _i_ =1 _[W][i]_ [(] _[ω]_ [) cos(] _[B][i]_ [(] _[ω]_ [) +] **[ A]** _[i]_ [(] _[ω]_ [)] _[ ·]_ [ (] _**[x]**_ _[,]_ _**[ z]**_ [))][ satisfies]


_∥Fj_ _−_ Φ _[ω]_ _j_ _[∥]_ _L_ [2] [2] ( _µ_ ) [+]


_N_ + _d_


_i_ =1


�� _∂iFj_ _−_ _∂i_ Φ _ωj_ ��2 _L_ [2] ( _µ_ ) _[≤]_ _[C]_ _n_ _[j]_ _[,]_ (28)


R _[N]_ _×_ R _[d]_ - _Ni_ =1+ _d_ _[ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ . Finally, _**θ**_ = ( _**θ**_ [1] _, . . .,_ _**θ**_ _[N]_ ) can then be


with _Cj_ = _∥F_ [�] _j∥_ [2] 1 [+ 4] _[π]_ [2] _[∥]_ _F_ [�] _j∥_ 1 


constructed by setting _**θ**_ _[j]_ = ( **A** _i_ ( _ω_ ) _, Bi_ ( _ω_ ) _,_ arccos( _[W][i]_ [(] _[ω]_ [)]


constructed by setting _**θ**_ _[j]_ = ( **A** _i_ ( _ω_ ) _, Bi_ ( _ω_ ) _,_ arccos( _[W][i]_ _R_ [(] _[ω]_ [)] )) _i_ =1 _,...,n_, which guarantees that Φ _[ω]_ _j_ [=]

_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [and so the proposition follows.]


B.3 PROOF OF COROLLARY 4.3


The proof of this corollary requires the following lemma, which extends Gonon (2024,
Lemma 4.10).


**Lemma** **B.1.** _Let_ _d, n, q_ _∈_ N _,_ _let_ _M_ 1 _, M_ 2 _>_ 0 _,_ _let_ _U_ _be_ _a_ _non-negative_ _random_ _variable,_ _and_ _let_
_Y_ 1 _, . . ., Yn_ _be i.i.d._ R _[d]_ _-valued random variables._ _Suppose_ E[ _U_ ] _≤_ _M_ 1 _and_ E[ _|Y_ 1 _|_ _[q]_ ] _≤_ _M_ 2 _._ _Then_


           - 1            P _U_ _≤_ 3 _M_ 1 _,_ max _q_ _>_ 0 _._
_i_ =1 _,...,n_ _[|][Y][i][| ≤]_ [(3] _[nM]_ [2][)]


_Proof._ The proof mimics that of in Gonon (2024, Lemma 4.10) by replacing the use of Markov’s
inequality for _q_ = 1 by the more general version:


1
P[ _|Y_ 1 _| >_ (3 _nM_ 2) _q_ ] _≤_ [E][[] _[|][Y]_ [1] _[|][q]_ []]


3 _n_ _[.]_


[[] _[|][Y]_ [1] _[|]_ []]

_≤_ [1]
3 _nM_ 2 3


_Proof of the corollary._ The corollary follows by replacing the argument leading to (28) in the proof
of Proposition 4.2 by Lemma B.1 and by noticing that


E [ _∥_ **A** 1 _∥_ _[q]_ ] = (2 _π_ ) _[q]_ _∥F_ [�] _j∥_ _[−]_ 1 [1]


B.4 PROOF OF PROPOSITION 4.4


_Fj_ ( _**ξ**_ ) _|_ d _**ξ**_ _._
R _[N]_ _×_ R _[d][ ∥]_ _**[ξ]**_ _[∥][q][|]_ [�]


_Proof._ It follows by combining the proof of Proposition 4.2 with the proof of Theorem 3 in Gonon
& Jacquier (2025). More specifically, the same proof can be used as for Proposition 4.2, except that


20


we need to replace the _L_ [2] ( _µ_ ) error bounds in (26) and (27) by uniform bounds. For (26), we can
follow precisely the proof of Theorem 3 in Gonon & Jacquier (2025) to obtain
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _∞,M_ _[≤]_ _C_ ~~_√_~~ _j_ _[∞]_ _n_ _[,]_ [0] (29)


R _[N]_ _×_ R _[d]_ - _Ni_ =1+ _d_ _[ξ]_ _i_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ �1 _/_ 2. Next, we


with _Cj_ _[∞][,]_ [0] = 2( _π_ + 1) _∥F_ [�] _j∥_ 1 + 8 _πM_ ( _N_ + _d_ ) 21 _∥F_ [�] _j∥_ 112 ��


1
2 ��
1


turn to the derivatives, that is, we aim to estimate ��� _∂k_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][k][F][j]_ ��� _∞,M_ [.] [Also in this case, we may]

proceed as in the proof of Theorem 3 in Gonon & Jacquier (2025) and apply the same estimates to
the random variables _Ui,_ ( _**x**_ _,_ _**z**_ ) = _WiAi,k_ sin( _Bi_ + **A** _i ·_ ( _**x**_ _,_ _**z**_ )). Let _ε_ 1 _, . . ., εn_ be i.i.d. Rademacher
random variables independent of **A** = ( **A** 1 _, . . .,_ **A** _n_ ) and **B** = ( _B_ 1 _, . . ., Bn_ ). Symmetrisation and
independence then yield


          - _n_          
��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� _∞,M_ [=][ E] sup _[N]_ [+] _[d]_ ����� _n_ 1 - - _Ui,_ ( _**x**_ _,_ _**z**_ ) _−_ E[ _Ui,_ ( _**x**_ _,_ _**z**_ )]������


_i_ =1


 - _Ui,_ ( _**x**_ _,_ _**z**_ ) _−_ E[ _Ui,_ ( _**x**_ _,_ _**z**_ )]�
�����


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_


1

_n_
�����


_n_


_n_

- _εiUi,_ ( _**x**_ _,_ _**z**_ )


_i_ =1


1

_n_
�����


�����


_≤_ 2E


= 2E


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_





 E


_n_


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_





 _._


_εiwiai,k_ sin( _bi_ + **a** _i ·_ ( _**x**_ _,_ _**z**_ ))

_i_ =1


1

_n_
�����


�����


������( **w** _,_ _**a**_ _,_ **b** )=( **W** _,_ **A** _,_ **B** )


_n_


Now fix _**a**_ = ( _**a**_ 1 _, . . .,_ _**a**_ _n_ ) _∈_ (R _[N]_ _×_ R _[d]_ ) _[n]_, **b** = ( _b_ 1 _, . . ., bn_ ) _∈_ R _[n]_, **w** = ( _w_ 1 _, . . ., wn_ ) _∈_ R _[n]_ and
denote
_T_ := _{_ ( _wiai,k_ ( _bi_ + **a** _i ·_ ( _**x**_ _,_ _**z**_ ))) _i_ =1 _,...,n_ : ( _**x**_ _,_ _**z**_ ) _∈_ [ _−M, M_ ] _[N]_ [+] _[d]_ _},_

_x_
_ϱi_ ( _x_ ) := _wiai,k_ sin( ) _,_ _x ∈_ R _,_
_wiai,k_

for _i_ = 1 _, . . ., n_ . Then, using the definitions in the first step, the comparison theorem (Ledoux
& Talagrand, 2013, Theorem 4.12) in the second step (note _ϱi_ (0) = 0 and _ϱi_ is 1-Lipschitz), and
standard Rademacher estimates (see, e.g., Gonon (2023)), we obtain


_εiwiai,k_ sin( _bi_ + **a** _i ·_ ( _**x**_ _,_ _**z**_ ))

_i_ =1


�����


1

_n_
�����


_n_


E


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_


_n_

- _εiti_


_i_ =1


_εiϱi_ ( _ti_ )

_i_ =1


�����


������


sup
**t** _∈T_


1

_n_
�����


= E


= 2E


_≤_ 2E


sup
**t** _∈T_


1

_n_
�����


_n_

- _εiwiai,kbi_


_i_ =1


_n_


1

_n_
�����


_≤_ 2E


_εi_ ( _wiai,k_ ( _bi_ + **a** _i ·_ ( _**x**_ _,_ _**z**_ ))

_i_ =1


�����


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_


_n_


1

_n_
������


  
+ 2E


sup
( _**x**_ _,_ _**z**_ ) _∈_ [ _−M,M_ ] _[N]_ [+] _[d]_


_n_


�����


�����


_n_

- _εiwiai,k_ **a** _i_


_i_ =1


_n_


+ [2] _[M]_

_n_


�1 _/_ 2


1
( _**x**_ _,_ _**z**_ ) _·_
_n_
�����


�1 _/_ 2


- _n_

 


_≤_ [2]

_n_


- _n_

 - _wi_ [2] _[a]_ _i,k_ [2] _[b]_ [2] _i_

_i_ =1


- _n_

 


_N_ + _d_


_l_ =1


- _wi_ [2] _[a]_ _i,k_ [2] _[a]_ [2] _i,l_

_i_ =1


Putting everything together, we obtain


- _n_

 - _Wi_ [2] _[A]_ _i,k_ [2] _[B]_ _i_ [2]

_i_ =1


+ [2] _[M]_

_n_


�1 _/_ 2


_._


- _n_

 - _Wi_ [2] _[A]_ _i,k_ [2] _[A]_ [2] _i,l_

_i_ =1


�1 _/_ 2 []





_N_ + _d_


_l_ =1


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� _∞,M_ _[≤]_ [2][E]




 [2]

_n_


- _n_

 





1 _/_ 2 []

+ _d_ 
- E - _Wi_ [2] _[A]_ _i,k_ [2] _[A]_ [2] _i,l_ - 

_l_ =1


4
_≤_ ~~_√_~~
_n_


          - _N_ + _d_
E - _Wi_ [2] _[A]_ _i,k_ [2] _[B]_ _i_ [2] �1 _/_ 2 + _M_ ( _N_ + _d_ )1 _/_ 2 





_≤_ _C_ ~~_√_~~ _j_ _[∞][,k]_ _,_
_n_


21


�1 _/_ 2 �1 _/_ 2 [�]
R _[N]_ _×_ R _[d][ ξ]_ _k_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ + 4 _M_ ( _N_ + _d_ ) [1] _[/]_ [2][ ��] R _[N]_ _×_ R _[d][ ξ]_ _k_ [2] _[∥]_ _**[ξ]**_ _[∥]_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ .


with _Cj_ _[∞][,k]_ = 4 _π_ [2] _∥F_ [�] _j∥_ [1] 1 _[/]_ [2]


���


Here, the last estimate follows from the inequality


E        - _Wi_ [2] _[A]_ _i,k_ [2] _[B]_ _i_ [2]        - _≤_ _π_ [4] _∥F_ [�] _j∥_ 1


and
E       - _Wi_ [2] _[A]_ _i,k_ [2] _[A]_ [2] _i,l_       - = 16 _π_ [4] _∥F_ [�] _j∥_ 1


_k_ _[ξ]_ _l_ [2] _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_ _._
R _[N]_ _×_ R _[d][ ξ]_ [2]


_k_ _[|]_ _F_ [�] _j_ ( _**ξ**_ ) _|_ d _**ξ**_
R _[N]_ _×_ R _[d][ ξ]_ [2]


Overall, we obtain (9) with _Cj_ _[∞]_ _≥_ [�] _[N]_ _k_ =0 [+] _[d]_ _[C]_ _j_ _[∞][,k]_ chosen as


1 1
_Cj_ _[∞]_ = 2( _π_ + 1) _∥F_ [�] _j∥_ 1 + (8 _πM_ + 4 _π_ [2] )( _N_ + _d_ ) 2 _∥F_  - _j∥_ 12 _[I]_ 2 [1] _,j_ _[/]_ [2] [+ 16] _[Mπ]_ [2][(] _[N]_ [+] _[ d]_ [)] _[∥]_ _F_ [�] _j∥_ [1] 1 _[/]_ [2] _I_ 4 [1] _,j_ _[/]_ [2] _[.]_


B.5 PROOF OF COROLLARY 4.5


_Proof._ First, extending the proof of Corollary 4 in Gonon & Jacquier (2025), we show that _Fj_ can
be approximated on _X_ up to error 2 _[ε]_ [in] _[ C]_ [1][-norm by a function in] _[ C]_ _c_ _[∞]_ [(][R] _[N]_ _[×]_ [ R] _[d]_ [)][.] [Indeed, first let]

_M_ _>_ 0 be such that _X_ _⊂_ [ _−M, M_ ] _[N]_ [+] _[d]_ . Then, classical approximation results (see, e.g., Whitney,
1934, Lemma 5) imply that there exists a smooth function _h_ : R _[N]_ _×_ R _[d]_ _→_ R such that

sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _h_ ( _**x**_ _,_ _**z**_ ) _|_ + _∥∇Fj_ ( _**x**_ _,_ _**z**_ ) _−∇h_ ( _**x**_ _,_ _**z**_ ) _∥≤_ _[ε]_ (30)
( _**x**_ _,_ _**z**_ ) _∈X_ 2 _[.]_


Without loss of generality we may assume that _h ∈_ _Cc_ _[∞]_ [(][R] _[N]_ _[×]_ [ R] _[d]_ [)][.] [Otherwise, we multiply] _[ h]_ [ with]
a cutoff function _ψ_ _∈_ _Cc_ _[∞]_ [(][R] _[N]_ _[×]_ [ R] _[d]_ [)][ which is equal to][ 1][ in an open set] _[ U]_ [with] _[ X]_ _[⊂]_ _[U]_ [(see, e.g.,]
H¨ormander, 1990, Theorem 1.4.1); thereby preserving (30).


In the next step, we now apply Proposition 4.4 to _h_ . Since _h_ is a Schwartz function, its Fourier
transform _h_ is also a Schwartz function and thus _h_ is integrable and

[�]

            
_[<][ ∞][.]_
R _[N]_ _×_ R _[d]_ [(1 +] _[ ∥]_ _**[ξ]**_ _[∥]_ [4][)] _[|]_ [�] _[h]_ [(] _**[ξ]**_ [)] _[|]_ [d] _**[ξ]**_


In particular, _h_ _∈FR_ for _R_ _>_ 0 large enough and, as _h_ is a Schwartz function, also _∂ih_ _∈F_ for
all _i_ . Thus, the hypotheses of Proposition 4.4 are satisfied and we obtain that there exist _n ∈_ N and
_**θ**_ _∈_ **Θ** such that

_N_ + _d_

��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[h]_ ��� _∞,M_ [+]                          - ��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][h]_ ��� _∞,M_ _[≤]_ 2 _[ε]_ _[.]_


_N_ + _d_


_i_ =1


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][h]_ ��� _∞,M_ _[≤]_ 2 _[ε]_


2 _[.]_


This estimate together with (30) then imply


sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [ +] _[ ∥∇][F][j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −∇][F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_
( _**x**_ _,_ _**z**_ ) _∈X_


_≤_ sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _h_ ( _**x**_ _,_ _**z**_ ) _|_ + _∥∇Fj_ ( _**x**_ _,_ _**z**_ ) _−∇h_ ( _**x**_ _,_ _**z**_ ) _∥_
( _**x**_ _,_ _**z**_ ) _∈X_

+ sup _|h_ ( _**x**_ _,_ _**z**_ ) _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [ +] _[ ∥∇][F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −∇][h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_
( _**x**_ _,_ _**z**_ ) _∈X_


_≤_ sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _h_ ( _**x**_ _,_ _**z**_ ) _|_ + _∥∇Fj_ ( _**x**_ _,_ _**z**_ ) _−∇h_ ( _**x**_ _,_ _**z**_ ) _∥_
( _**x**_ _,_ _**z**_ ) _∈X_


+ sup _|F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [ +]
( _**x**_ _,_ _**z**_ ) _∈X_


_N_ + _d_

- _|∂iF_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[∂][i][h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_

_i_ =1


_≤_ sup _|Fj_ ( _**x**_ _,_ _**z**_ ) _−_ _h_ ( _**x**_ _,_ _**z**_ ) _|_ + _∥∇Fj_ ( _**x**_ _,_ _**z**_ ) _−∇h_ ( _**x**_ _,_ _**z**_ ) _∥_
( _**x**_ _,_ _**z**_ ) _∈X_


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][h]_ ��� _∞,M_ _[≤]_ _[ε,]_


22


+ ��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[h]_ ��� _∞,M_ [+]


_N_ + _d_


_i_ =1


where we used that


_∥∇F_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[−∇][h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ [=]


1 _/_ 2

- _N_ + _d_ 
 - _|∂iF_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[∂][i][h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [2]

_i_ =1


_≤_


_N_ + _d_

- _|∂iF_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[−][∂][i][h]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|][,]_

_i_ =1


since _∥_ _**y**_ _∥_ 2 _≤∥_ _**y**_ _∥_ 1 for all _**y**_ _∈_ R _[N]_ [+] _[d]_ .


C PROOFS FOR SECTION 4.2


C.1 PROOF OF THEOREM 4.6


_Proof._ Choose _M_ such that _BN_ _× Dd_ _⊂_ [ _−M, M_ ] _[N]_ [+] _[d]_ and [ _−R, R_ ] _[N]_ _× Dd_ _⊂_ [ _−M, M_ ] _[N]_ [+] _[d]_ .
Firstly, our hypotheses on _F_ guarantee that _F_ satisfies the hypotheses of Proposition 4.4. Hence,
there exists _**θ**_ _∈_ **Θ** such that for any _j_ _∈{_ 1 _, . . ., N_ _}_,


_N_ + _d_

��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _∞,M_ [+]                       - ��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� _∞,M_ _[≤]_ _[C]_ ~~_√_~~ _j_ _[∞]_ _n ._ (31)


_N_ + _d_


��� _∂i_ ¯ _F n,R,j_ _**θ**_ _[−]_ _[∂][i][F][j]_ ��� _∞,M_ _[≤]_ _[C]_ ~~_√_~~ _j_ _[∞]_ _n ._ (31)


_i_ =1


Then, for all _**x**_ _∈_ [ _−M, M_ ] _[N]_ _,_ _**z**_ _∈_ _Dd_
_∥∇_ _**x**_ _F_ [¯] _R_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ [2] _[≤∥∇]_ _**[x]**_ [ ¯] _[F][ n,]_ _R_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −∇]_ _**[x]**_ _[F]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ [2][ +] _[ ∥∇]_ _**[x]**_ _[F]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ [2]





_N_

 


1 _/_ 2





_≤_


- _|∂iF_ [¯] _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[∂][i][F][j]_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[|]_ [2]

_i,j_ =1


+ _λ_


(32)


_≤_ _N_ [max] _[j]_ [=1] ~~_√_~~ _[,...,N]_ _[C]_ _j_ _[∞]_ + _λ._
_n_

Therefore, using that max _**x**_ _∈_ [ _−M,M_ ] _N_ _∥∇_ _**x**_ _F_ [¯] _R_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ [2] [is the best Lipschitz-constant for] _[F]_ [¯] _[ n,]_ _R_ _**[θ]**_ on

[ _−M, M_ ] _[N]_ for any given _**z**_ _∈_ _Dd_, we obtain for all _**x**_ _∈_ [ _−M, M_ ] _[N]_ _,_ _**z**_ _∈_ _Dd_ that

_∥F_ [¯] _R_ _[n,]_ _**[θ]**_ [(] _**[x]**_ [1] _[,]_ _**[ z]**_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R_ _**[θ]**_ [(] _**[x]**_ [2] _[,]_ _**[ z]**_ [)] _[∥]_ [2] _[≤∥]_ _**[x]**_ [1] _[ −]_ _**[x]**_ [2] _[∥]_ [2] max _R_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[∥]_ 2 [2]
_**x**_ _∈_ [ _−M,M_ ] _[N][ ∥∇]_ _**[x]**_ [ ¯] _[F][ n,]_ _**[θ]**_

_≤_            - _N_ [max] _[j]_ [=1] ~~_√_~~ _[,...,N]_ _[C]_ _j_ _[∞]_ + _λ_ �2 _∥_ _**x**_ [1] _−_ _**x**_ [2] _∥_ [2] _._
_n_


In particular, for _n_ satisfying _N_ [2] [(max] _[j]_ [=1] _[,...,N]_ [2] _[C]_ _j_ _[∞]_ [)][2]


_√_
with _BR_ = _{_ _**x**_ _∈_ R _[N]_ : _∥_ _**x**_ _∥≤_ _R_


[=1] (1 _−_ _[,...,N]_ _λ_ ) [2] _j_ _<_ _n_ we obtain that _F_ [¯] _R_ _[n,]_ _**[θ]**_ : _BR_ _× Dd_ _→_ _BR_,


with _BR_ = _{_ _**x**_ _∈_ R _[N]_ : _∥_ _**x**_ _∥≤_ _R_ _N_ _}_, is contractive in the first argument, hence the system (4) has

the echo state property by Gonon et al. (2020, Proposition 1).


By the relation between the Lipschitz-constant and the maximal norm of the Jacobian, the assumption _∥∇_ _**x**_ _F_ ( _**x**_ _,_ _**z**_ ) _∥_ 2 _≤_ _λ_ guarantees that _F_ ( _·,_ _**z**_ ) is _λ_ -contractive for any _**z**_ _∈_ _Dd_ . Hence, we may
estimate
�� _U F_ ( _**z**_ ) _t −_ _U_ ¯ ( _**z**_ ) _t_ �� = _∥_ _**x**_ _t −_ _**x**_ ˆ _t∥_ = ��� _F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _−_ _F_ ¯ _n,R_ _**θ**_ [(ˆ] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] ���

_≤∥F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _−_ _F_ (ˆ _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _∥_ + ��� _F_ (ˆ _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _−_ _F_ ¯ _n,R_ _**θ**_ [(ˆ] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] ���


1 _/_ 2





_≤_ _λ ∥_ _**x**_ _t−_ 1 _−_ _**x**_ ˆ _t−_ 1 _∥_ +


_≤_ _λ ∥_ _**x**_ _t−_ 1 _−_ _**x**_ ˆ _t−_ 1 _∥_ +


Iterating (33), we obtain


 _N_

 


_j_ =1


2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _∞,M_


_√_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ _._
_n_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_
_n_


(33)


(34)


�� _U F_ ( _**z**_ ) _t −_ _U_ ¯ ( _**z**_ ) _t_ �� _≤_ _λJ ∥_ _**x**_ _t−J_ _−_ _**x**_ ˆ _t−J_ _∥_ +


_√_

_J_

- _λ_ _[k][−]_ [1]


_k_ =1


_J_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ _._
_n_


_≤_ _λ_ _[J]_ _[√]_


_N_ ( _M_ + _R_ ) +


23


_√_

_J−_ 1

- _λ_ _[k]_


_k_ =0


_J−_ 1


Letting _J_ _→∞_, we thus arrive at the bound (12).


C.2 PROOF OF LEMMA 4.7


The proof of Lemma 4.7 is related to the approach introduced in Gonon & Ortega (2020) and subsequently used, e.g., in Gonon et al. (2023); Gonon & Ortega (2021).


_Proof._ We start by constructing a partition of _**x**_ ˆ _t_ as in the statement. If _N_ = 1, we simply
have _**x**_ ˆ _t_ = [ˆ _xt_ ] _∈_ R. Next, we define the reservoir vector _F_ [˜] _R,i_ : _j_ = ( _F_ [˜] _i, . . .,_ _F_ [˜] _j_ ). Then, for
_k_ = 1, _j_ _∈{_ 1 _, . . ., l_ 1 _}_, and _k_ = 2 _, . . ., K_ _−_ 1, _j_ _∈{lk−_ 1 + 1 _, . . ., lk}_, we have _Pj_ ˆ _**x**_ _t_ =

[ˆ _**x**_ [(] _t_ _[k]_ [+1)] _, . . .,_ ˆ _**x**_ [(] _t_ _[K]_ [)] _,_ 0 _, . . .,_ 0] and _Pj_ ˆ _**x**_ _t_ = 0 for _j_ = _lK−_ 1 + 1 _, . . ., N_ . Inserting these choices
into (15), we may rewrite the dynamics as

_**x**_ ˆ [(] _t_ _[k]_ [)] = _F_ [˜] _lk−_ 1+1: _lk_ ([ˆ _**x**_ [(] _t−_ _[k]_ [+1)] 1 _, . . .,_ ˆ _**x**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] _[,]_ _t ∈_ Z _−,_ (35)

for _k_ = 1 _, . . ., K_ _−_ 1 and _**x**_ ˆ [(] _t_ _[K]_ [)] = _F_ [˜] _lK−_ 1+1: _lK_ (0 _,_ _**z**_ _t_ ). In particular, _**x**_ ˆ [(] _t_ _[K]_ [)] = _F_ [˜] _lK−_ 1+1: _lK_ (0 _,_ _**z**_ _t_ ),
which depends only on _**z**_ _t_, is explicitly given for all _t_ _∈_ Z _−_, and for all _k_ = 1 _, . . ., K_ _−_ 1, we see
that _**x**_ ˆ [(] _t_ _[k]_ [)] only depends on _**x**_ ˆ [(] _t−_ _[k]_ [+1)] 1 _, . . .,_ ˆ _**x**_ [(] _t−_ _[K]_ 1 [)][.] [Thus, (15) admits a unique solution which can be ex-]
plicitly obtained from the recursion (35), that is, for all _t ∈_ Z _−_, we have _**x**_ ˆ [(] _t_ _[K]_ [)] = _F_ [˜] _lK−_ 1+1: _lK_ (0 _,_ _**z**_ _t_ ),
_**x**_ ˆ _t_ [(] _[K][−]_ [1)] = _F_ [˜] _lK−_ 2+1: _lK−_ 1([ˆ _**x**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)][,] _[. . .]_ [,] _**[x]**_ [ˆ][(1)] _t_ = _F_ [˜] 1: _l_ 1([ˆ _**x**_ [(2)] _t−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)][.] [This]
proves that _F_ [˜] has the echo state property.


C.3 PROOF OF THEOREM 4.8


_Proof._ Without loss of generality we may assume _ε_ _≤_ 1, because proving (17) for _ε_ _≤_ 1 also
implies that (17) holds for _ε >_ 1.


Let _HU_ : ( _Dd_ ) [Z] _[−]_ _→_ _Bm_ be the functional associated to the filter _U_ . Then, as in the proof of Gonon
& Ortega (2021, Theorem 2.1), there exists _K_ _∈_ N and a continuous function _G_ [¯] : ( _Dd_ ) _[dK]_ _→_ _Bm_
such that

_ε_

sup �� _HU_ ( _**z**_ ) _−_ _G_ ¯( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0)�� _<_ (36)
_**z**_ _∈_ ( _Dd_ ) [Z] _[−]_ 4 _[.]_


Moreover, e.g., by the argument in Gonon & Jacquier (2025, Corollary 4), there exists a function
_G ∈_ _Cc_ _[∞]_ [((][R] _[d]_ [)] _[K][, B][m]_ [)][ which satisfies]


sup
_**z**_ _∈_ (R _[d]_ ) _[K]_


_ε_
�� _G_ ( _**z**_ ) _−_ _G_ ¯( _**z**_ )�� _<_ (37)
4 _[.]_


Next, choose _N_ = ( _K −_ 1) _d_ + _m_ and consider the recurrent QNN introduced in (4). Denote

_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [) =] _[ R][ −]_ [2] _[R]_ [[][P] 1 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ ) + P2 _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ )] _,_ ( _**x**_ _,_ _**z**_ ) _∈_ R _[N]_ _×_ R _[d]_ (38)

the update maps without preprocessing matrices. For 1 _≤_ _i_ _≤_ _j_ _≤_ _N_, write _F_ [¯] _R,i_ _[n,]_ _**[θ]**_ : _j_ =
( _F_ [¯] _R,i_ _[n,]_ _**[θ]**_ _[, . . .,]_ _[F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [)][ and] _[ l][k]_ [=] _[ m]_ [ + (] _[k][ −]_ [1)] _[d]_ [ for] _[ k]_ [= 1] _[, . . ., K]_ [.] [Define the constants]


1 _/_ 2





_√_
_LG_ = max(


_d,_ sup _CG_ = 4 _LG_
_**z**_ _∈_ (R _[d]_ ) _[K][ ∥∇][G]_ [(] _**[z]**_ [)] _[∥]_ [) + 1] _[,]_


 _K_

 


_k_ =2


_K−k_ +1

- (2 _LG_ ) _[j]_


_j_ =1


_._ (39)


Then, as _G_ _∈_ _Cc_ _[∞]_ [((][R] _[d]_ [)] _[K]_ [)] [and] [the] [identity] [is] [smooth,] [Corollary] [4.5] [(applied] [componentwise)]
guarantees that there exist _nK_, _RK_ and _**θ**_ _K_ _∈_ **Θ** _[d]_ such that

_ε_
sup _∥F_ [¯] _R_ _[n]_ _K_ _[K]_ _,l_ _[,]_ _**[θ]**_ _K_ _[K]_ _−_ 1+1: _lK_ [(0] _[,]_ _**[ z]**_ [)] _[ −]_ _**[z]**_ _[∥]_ [+] [sup] _∥∇F_ [¯] _R_ _[n]_ _K_ _[K]_ _,l_ _[,]_ _**[θ]**_ _K_ _[K]_ _−_ 1+1: _lK_ [(0] _[,]_ _**[ z]**_ [)] _[ −]_ **[1]** _[d][∥]_ _[<]_ _,_ (40)
_**z**_ _∈Dd_ _**z**_ _∈Dd_ _CG_


and (recursively), for all _k_ = _K −_ 1 _, . . .,_ 2 there exist _nk_, _Rk_ and _**θ**_ _k_ _∈_ **Θ** _[d]_ such that

_ε_
sup _∥F_ [¯] _R_ _[n]_ _k_ _[k]_ _,l_ _[,]_ _**[θ]**_ _k_ _[k]_ _−_ 1+1: _lk_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _**[x]**_ [1:] _[d][∥]_ [+] _[ ∥∇][F]_ [¯] _[ n]_ _Rk_ _[k]_ _,l_ _[,]_ _**[θ]**_ _k_ _[k]_ _−_ 1+1: _lk_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ **[1]** _[d][∥]_ _[<]_ _,_
( _**x**_ _,_ _**z**_ ) _∈_ [ _−Rk_ +1 _,Rk_ +1] _[N]_ _×Dd_ _CG_

(41)


24


and there exist _n_ 1, _R_ 1 and _**θ**_ 1 _∈_ **Θ** _[d]_ such that


sup
([ _**z**_ _−K_ +1 _,...,_ _**z**_ _−_ 1] _,_ _**z**_ 0) _∈_ [ _−R_ 2 _,R_ 2] _[N]_ _×Dd_


- _∥F_ [¯] _R_ _[n]_ 1 [1] _,_ _[,]_ 1: _**[θ]**_ [1] _m_ [([] _**[z]**_ _[−][K]_ [+1] _[, . . .,]_ _**[ z]**_ _[−]_ [1] _[,]_ [ 0]] _[,]_ _**[ z]**_ [0][)] _[ −]_ _[G]_ [(] _**[z]**_ _[−][K]_ [+1] _[, . . .,]_ _**[ z]**_ [0][)] _[∥]_


+ _∥∇F_ [¯] _R_ _[n]_ 1 [1] _,_ _[,]_ 1: _**[θ]**_ [1] _m_ [([] _**[z]**_ _[−][K]_ [+1] _[, . . .,]_ _**[ z]**_ _[−]_ [1] _[,]_ [ 0]] _[,]_ _**[ z]**_ [0][)] _[ −∇][G]_ [(] _**[z]**_ _[−][K]_ [+1] _[, . . .,]_ _**[ z]**_ [0][)] _[∥]_           - _<_ 4 _[ε]_ _[.]_

(42)
Without loss of generality we may choose _R_ = _R_ 1 = _. . ._ = _RK_, since we can always replace
_Rk_ by max( _Rk, Rk_ +1) (and hence ultimately replace _R_ 1 _, . . ., RK_ by _R_ ) and absorb the change in
an adjusted choice of parameters _γ_ _[i,j]_ (see representation (7)). Moreover, by a similar reasoning
we may assume without loss of generality that _n_ = _n_ 1 = _. . ._ = _nK_ . Indeed, otherwise we may
again choose _n_ to be the maximum of _n_ 1 _, . . ., nK_, replace _n_ 1 _, . . ., nK_ by _n_ and recover the same
functions (7) by setting surplus terms _i > nk_ to 0 by appropriate choice of _γ_ _[i,j]_ . The extra factor _nnk_ [,]
in turn, can be absorbed by modifying the choice of _R_ .


Denote by _Lk_ be the best Lipschitz constant for _F_ [¯] _R,l_ _[n,]_ _**[θ]**_ _k_ _[k]_ _−_ 1+1: _lk_ [.] [Then] [(40)–(42)] [imply] [that] _[L][k]_ _[≤]_
_√_

_d_ + _ε_ _≤_ _LG_ for _k_ = _K, . . .,_ 2 and _L_ 1 _≤_ sup _**z**_ _∈_ R _d_ ) _K_ _∥∇G_ ( _**z**_ ) _∥_ + 1 _≤_ _LG_ . In particular,
_LG_ _≥_ max( _L_ 1 _, . . ., LK_ ) is a bound on the Lipschitz constant for all QNNs _F_ [¯] _R,l_ _[n,]_ _**[θ]**_ _k_ _[k]_ _−_ 1+1: _lk_ [and] _[G]_ [.]

Partition _**x**_ ˆ _t_ = [ˆ _**x**_ [(1)] _t_ _[, . . .,]_ [ ˆ] _**[x]**_ _t_ [(] _[K]_ [)] ] _∈_ R _[m]_ _×_ (R _[d]_ ) _[K][−]_ [1] . Using the triangle inequality, we then obtain


sup
_**z**_ _∈_ ( _Dd_ ) _[K]_ [+1]


��� _G_ ( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ _**x**_ ˆ(1)0 ���


= sup
_**z**_ _∈_ ( _Dd_ ) [(] _[K]_ [+1)]


��� _G_ ( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ _F_ ¯ _n,R,_ 1: _**θ**_ _m_ [([ˆ] _**[x]**_ _−_ [(2)] 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)] _[,]_ [ 0]] _[,]_ _**[ z]**_ [0][)] ���


_≤_ ��� _G_ ( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ _G_ ([ˆ _**x**_ (2) _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] ���


(2)
+ ��� _G_ ([ˆ _**x**_ _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] _[ −]_ _[F]_ [¯] _[ n,]_ _R,_ 1: _**[θ]**_ _m_ [([ˆ] _**[x]**_ _−_ [(2)] 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)] _[,]_ [ 0]] _[,]_ _**[ z]**_ [0][)] ���

_≤_ _LG_ ���( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ ([ˆ _**x**_ (2) _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] ��� + 4 _ε_ _[.]_


For the last norm, we write

2
(2)
���( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ ([ˆ _**x**_ _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] ��� =


_K−_ 2


_k_ =0


(43)


2
��� _**z**_ _−K_ + _k−_ 1 _−_ _**x**_ ˆ( _−k_ 1)��� _._


2
��� _**z**_ _−k−_ 1 _−_ _**x**_ ˆ( _−K_ 1 _−k_ )��� =


_K_


_k_ =2


We proceed by backward induction over _k_ to prove that for all _k_ = _K, . . .,_ 2 it holds


_K−k_ +1

2
( _k_ )                             - _[ε]_ [2]
��� _**z**_ _−K_ + _k_ + _t −_ _**x**_ ˆ _t_ ��� _≤_ (2 _LG_ ) _[j]_ [2] _,_


_K−k_ +1

 


_,_
_CG_ [2]


- _[ε]_ [2]

(2 _LG_ ) _[j]_

_C_

_j_ =1


for arbitrary _t ∈_ Z _−_ . Indeed, we have

2 2
��� _**z**_ _−K_ + _k_ + _t −_ _**x**_ ˆ( _tk_ )��� = ��� _**z**_ _−K_ + _k_ + _t −_ _F_ ¯ _n,R,l_ _**θ**_ _k−_ 1+1: _lk_ [([ˆ] _**[x]**_ _t_ [(] _−_ _[k]_ [+1)] 1 _, . . .,_ ˆ _**x**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] ���


and so for _k_ = _K_ it follows that

2 2
��� _**z**_ _−K_ + _k_ + _t −_ _**x**_ ˆ( _tk_ )��� = ��� _**z**_ _t −_ _F_ ¯ _n,R,l_ _**θ**_ _K−_ 1+1: _lK_ [(0] _[,]_ _**[ z]**_ _[t]_ [)] ��� _≤_ _[ε]_ [2]


_[ε]_ [2] _≤_ 2 _LG_ _ε_ [2]

_CG_ [2] _C_ [2]


_t_ _R,lK−_ 1+1: _lK_ _CG_ [2] _CG_ [2]

Assume that the bound holds for a fixed _k_ _∈{K, . . .,_ 3 _}_, then for _k_ _−_ 1 we estimate (with the
notation _fk−_ 1 = _F_ [¯] _R,l_ _[n,]_ _**[θ]**_ _k−_ 2+1: _lk−_ 1 [)]

2 2
��� _**z**_ _−K_ +( _k−_ 1)+ _t −_ _**x**_ ˆ( _tk−_ 1)��� = ��� _**z**_ _−K_ + _k−_ 2 _−_ _fk−_ 1([ˆ _**x**_ ( _t−k_ )1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] ���


2
_≤_ 2 ��� _**z**_ _−K_ + _k−_ 2 _−_ _fk−_ 1([ _**z**_ _−K_ + _k−_ 2 _,_ ˆ _**x**_ ( _t−k_ +1)1 _, . . .,_ ˆ _**x**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] ���


2
( _k_ +1)
+ 2 ��� _fk−_ 1([ _**z**_ _−K_ + _k_ + _t−_ 1 _,_ ˆ _**x**_ _t−_ 1 _, . . .,_ ˆ _**x**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[f][k][−]_ [1][([ˆ] _**[x]**_ [(] _−_ _[k]_ 2 [)] _[, . . .,]_ [ ˆ] _**[x]**_ [(] _t−_ _[K]_ 1 [)] _[,]_ [ 0] _[, . . .,]_ [ 0]] _[,]_ _**[ z]**_ _[t]_ [)] ���


_≤_
_CG_ [2]


_,_
_CG_ [2]


_≤_ 2 _[ε]_ [2]


2
_C_ _[ε]_ [2] _G_ [2] + 2 _L_ ��� _**z**_ _−K_ + _k_ + _t−_ 1 _−_ _**x**_ ˆ( _t−k_ )1��� _≤_ 2 _C_ _[ε]_ [2]


+
_CG_ [2]


_K−k_ +1


- _[ε]_ [2]

(2 _L_ ) _[j]_

_C_

_j_ =1


_K−k_


- _[ε]_ [2]

(2 _L_ ) _[j]_ [+1]

_C_

_j_ =1


25


which completes the induction. Therefore, we obtain


2
(2)
���( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ ([ˆ _**x**_ _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] ��� =


_K_


_k_ =2


2
��� _**z**_ _−K_ + _k−_ 1 _−_ _**x**_ ˆ( _−k_ 1)���


_K−k_ +1


_K_


_k_ =2


- (2 _L_ ) _[j]_ = _ε_ [2]

16 _L_

_j_ =1


16 _L_ [2] _G_


_≤_


_K_


_k_ =2


2
��� _**z**_ _−K_ + _k−_ 1 _−_ _**x**_ ˆ( _−k_ 1)��� _≤_ _C_ _[ε]_ [2] _G_ [2]


2
��� _**z**_ _−K_ + _k−_ 1 _−_ _**x**_ ˆ( _−k_ 1)��� _≤_ _[ε]_ [2][2]


From (43), we thus obtain


sup
_**z**_ _∈_ ( _Dd_ ) _[K]_ [+1]


(44)

_[ε]_

2 _[.]_


��� _G_ ( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ _**x**_ ˆ(1)0 ���


_≤_ _LG_ ���( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ ([ˆ _**x**_ (2) _−_ 1 _[, . . .,]_ [ ˆ] _**[x]**_ [(] _−_ _[K]_ 1 [)][]] _[,]_ _**[ z]**_ [0][)] ��� + 4 _ε_ _[≤]_ 2 _[ε]_


Setting _W_ to be the projection onto the first block _**x**_ ˆ [(1)] 0 [,] [(that] [is,] _[W]_ [has] [zero] [entries] [except] [for]
_Wi,i_ = 1 for _i_ = 1 _, . . ., m_ ) and putting together (36), (37) and (44) yields


�� _HU_ ( _**z**_ ) _−_ _G_ ¯( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0)��


sup sup
_**z**_ _∈_ ( _Dd_ ) [Z] _[−]_ _t∈_ Z _−_


�� _HU_ ( _**z**_ ) _−_ _H_ ¯ _UW_ ( _**z**_ )�� _≤_ sup
_**z**_ _∈_ ( _Dd_ ) [Z] _[−]_


+ sup
_**z**_ _∈_ (R _[d]_ ) _[K]_


�� _G_ ( _**z**_ ) _−_ _G_ ¯( _**z**_ )�� + sup
_**z**_ _∈_ ( _Dd_ ) _[K]_ [+1]


�� _G_ ( _**z**_ _−K_ +1 _, . . .,_ _**z**_ 0) _−_ _H_ ¯ _UW_ ( _**z**_ )��


(45)


_≤_ _[ε]_


_[ε]_ _[ε]_

4 [+] 4


_[ε]_ _[ε]_

4 [+] 2


2 [=] _[ ε.]_


It remains to be shown that (14) has the echo state property. Recall that we partition _**x**_ ˆ _t_ =

[ˆ _**x**_ [(1)] _t_ _[, . . .,]_ [ ˆ] _**[x]**_ _t_ [(] _[K]_ [)] ] _∈_ R _[m]_ _×_ (R _[d]_ ) _[K][−]_ [1] . For _k_ = 1, _j_ _∈{_ 1 _, . . ., l_ 1 _}_, and _k_ = 2 _, . . ., K_ _−_ 1,
_j_ _∈{lk−_ 1 + 1 _, . . ., lk}_, select _Pj_ as the matrix with zero entries, except for ( _Pj_ ) _l,l_ + _lk_ = 1 for
_l_ = 1 _, . . ., d_ ( _K_ _−_ _k_ ) and let _Pj_ = 0 for _j_ = _lK−_ 1 + 1 _, . . ., N_ . Then, for _k_ = 1, _j_ _∈{_ 1 _, . . ., l_ 1 _}_,
and _k_ = 2 _, . . ., K_ _−_ 1, _j_ _∈{lk−_ 1 + 1 _, . . ., lk}_, we have _Pj_ ˆ _**x**_ _t_ = [ˆ _**x**_ [(] _t_ _[k]_ [+1)] _, . . .,_ ˆ _**x**_ [(] _t_ _[K]_ [)] _,_ 0 _, . . .,_ 0] and
_Pj_ ˆ _**x**_ _t_ = 0 for _j_ = _lK−_ 1 + 1 _, . . ., N_ . Then, echo state property follows by calling Lemma 4.7.
Therefore, the approximation bound for the functional (45) immediately implies the corresponding
bound for the filter (17), which completes the proof of the theorem.


D CONSTRUCTION OF `V`


In this appendix we provide further details on the choice of `V` appearing in the quantum circuit. Our
presentation follows Gonon & Jacquier (2025).

Generally, the matrix `V` _∈_ C _[n]_ `[U]` _[×][n]_ `[U]` can be any unitary matrix mapping _|_ 0 _⟩_ _[⊗]_ [n] to the state _|ψ⟩_ =
~~_√_~~ 1 - _n−_ 1 ~~_√_~~ 1 - _n−_ 1
_n_ _i_ =0 _[|]_ [4] _[i][⟩]_ [which, for] _[ n][ ≥]_ [2][, is also explicitly given as] _[ |][ψ][⟩]_ [=] _n_ _i_ =0 _[|][i][⟩⊗|]_ [00] _[⟩]_ [.]


As `V` _|_ 0 _⟩_ _[⊗]_ [n] = _|ψ⟩_ is the only property required in the proof, many alternative choices of `V` are
possible and one may thus select the one that is most suitable from the perspective of hardware
requirements or limitations.


**Example** One explicit example for `V` is given by `V` := 2 _|φ⟩⟨φ| −_ `I`, with


_|_ 0 _⟩_ + _|ψ⟩_
_|φ⟩_ := _,_
�2 (1 + _⟨_ 0 _|ψ⟩_ )


where we write _|_ 0 _⟩_ in place of _|_ 0 _⟩_ _[⊗]_ [n] for brevity here. One easily checks that `V` _[†]_ = 2 _|φ⟩⟨φ| −_ `I` = `V`
and thus `VV` _[†]_ = `V` _[†]_ `V` = `I` . Furthermore, a straightforward computation yields that


`V` _|_ 0 _⟩_ = (2 _|φ⟩⟨φ| −_ `I` ) _|_ 0 _⟩_

= _[|]_ [0] _[⟩]_ [(1 +] _[ ⟨][ψ][|]_ [0] _[⟩]_ [) +] _[ |][ψ][⟩]_ [(1 +] _[ ⟨][ψ][|]_ [0] _[⟩]_ [)] _−|_ 0 _⟩_ = _|ψ⟩_ _._

1 + _⟨_ 0 _|ψ⟩_


26


**Construction** **of** _|ψ⟩_ In the case _n_ 0 = 0, there is an explicit construction of _ψ_ in terms of
Hadamard gates acting on the control qubits. Indeed, for n _≥_ 2, (Gonon & Jacquier, 2025,
Lemma A.2) shows that


_|ψl⟩_ n _l_ =


E MONTE CARLO ERROR


�n _l−_ 2

 


`H` _|_ 0 _⟩_

_i_ =0


_⊗|_ 00 _⟩_ _._


_√_
In practice, the empirical sampling error leads to an additional error component of order 1 _/_ _S_ for

_S_ independent shots, see, e.g., Qi et al. (2023); Liu et al. (2025). Here, we outline how this Monte
Carlo error could be taken into account in the present setting.


More specifically, our QNNs in (3) and (4) are defined using probabilities, rather than their Monte
Carlo estimates


P� _[n,]_ _m_ _**[θ]**_ := _S_ [1]


_S_


1 _{m,_ 4+ _m,...,_ 4( _n−_ 1)+ _m}_ ( _i_ [(] _[s]_ [)] ) _,_

_s_ =1


with _i_ [(] _[s]_ [)] the measured state in the _i_ -th shot. To obtain refined bounds incorporating the sampling error, one would proceed as follows. Denote by _F_ [¯] _R_ _[n,]_ _**[θ]**_ _[,S]_ the RQNN state map with output probabilities
estimated by _S_ shots, by _**x**_ ˆ _[S]_ the associated state and by _U_ [¯] _S_ the associated filter.


For the state map itself, the _L_ [2] -error can be directly controlled (as in Gonon & Jacquier (2025)) by


��

E


R _[N]_ _×_ R _[d]_


2 �1 _/_ 2
��� _F_ ¯ _n,R,j_ _**θ**_ [(] _**[x]**_ _[,]_ _**[ z]**_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ _[,S]_ ( _**x**_ _,_ _**z**_ )��� _µ_ (d _**x**_ _,_ d _**z**_ )


2

_≤_ 2 _R_ 

_i_ =1


��


_n,_ _**θ**_ _j_
R _[N]_ _×_ R _[d]_ [ E] ����P _i_


_n,_ _**θ**_ _j_

_i_ ( _**x**_ _,_ _**z**_ ) _−_ P [�] _i_ _[n,]_ _**[θ]**_ _[j]_


2 [�] �1 _/_ 2
_i_ _[n,]_ _**[θ]**_ _[j]_ ( _**x**_ _,_ _**z**_ )��� _µ_ (d _**x**_ _,_ d _**z**_ )


(46)


_≤_ ~~_√_~~ [4] _[R]_ _,_

_S_


_S_ [1] - _Ss_ =1 _[X][s][|]_ [2][] =] [Var(] _S_ _[X]_ [1][)]


using that E[ _|_ E[ _X_ 1] _−_ [1]


_S_ [1] for i.i.d. random variables _X_ 1 _, . . ., XS_ .


For the associated filter, one may proceed as follows. Firstly, (33) in the proof of Theorem 4.6 can
be adapted to
�� _U_ ¯ _S_ ( _**z**_ ) _t −_ _U_ ( _**z**_ ) _t_ �� = �� _**x**_ ˆ _St_ _[−]_ _**[x]**_ _[t]_ �� = ��� _F_ ¯ _n,R_ _**θ**_ _,S_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[F]_ [(] _**[x]**_ _[t][−]_ [1] _[,]_ _**[ z]**_ _[t]_ [)] ���

_≤_ �� _F_ ( _**x**_ _t−_ 1 _,_ _**z**_ _t_ ) _−_ _F_ (ˆ _**x**_ _St−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] �� + ��� _F_ (ˆ _**x**_ _St−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R_ _**[θ]**_ _[,S]_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] ���


1 _/_ 2


1 _/_ 2








_≤_ _λ_ �� _**x**_ _t−_ 1 _−_ _**x**_ ˆ _St−_ 1�� +


_≤_ _λ_ �� _**x**_ _t−_ 1 _−_ _**x**_ ˆ _St−_ 1�� +


 _N_

 

_j_ =1


_√_

_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ +
_n_


2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ��� _∞,M_


2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ���


2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ��� _∞,M_


+


 _N_

 

_j_ =1


2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ��� _∞,M_


2
��� _F_ ¯ _n,R,j_ _**θ**_ _[−]_ _[F][j]_ ���





(47)


 _N_

 

_j_ =1


1 _/_ 2





_._


The last error term can be bounded as





_≤_ ~~_√_~~ _[C]_

_S_





1 _/_ 2 []


 _≤_





_N_

 


2

- E ���� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ���

_j_ =1


2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ���


_∞,M_








E





_N_

 


1 _/_ 2

- []





_j_ =1


2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ _−_ _F_ [¯] _R,j_ _[n,]_ _**[θ]**_ ��� _∞,M_


for a suitable constant _C_ using techniques from statistical learning theory, provided that P [�] _[n,]_ _m_ _**[θ]**_ is
Lipschitz continuous as a function of ( _**x**_ _,_ _**z**_ ). Inserting this into (47) and proceeding precisely as in
the proof of Theorem 4.6 then yields a bound that incorporates also the sampling error.


27


Alternatively, as the Lipschitz continuity may be hard to verify, we may obtain an _L_ [2] -bound analogously to Theorem 4.6 as follows. First, using that the shots are independent across evaluations, we
may apply (46) to estimate









2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(ˆ] _**[x]**_ _t_ _[S]_ _−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] ��� 


1 _/_ 2 []








E


_N_

 


_j_ =1








1 _/_ 2


_≤_


_N_

 


_N_

2 [�]

- E ���� _F_ ¯ _n,R,j_ _**θ**_ _,S_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(ˆ] _**[x]**_ _t_ _[S]_ _−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] ��� 

_j_ =1


_√_
_≤_


_N_ ~~_√_~~ [4] _[R]_ _,_

_S_


_N_ ~~_√_~~ [4] _[R]_


where the expectations are taken with respect to sampling the probabilities to evaluate
_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ _[,S]_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)][.]


Next, by proceeding as in (47), we may estimate


_√_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_
_n_


E[�� _U_ ¯ _S_ ( _**z**_ ) _t −_ _U_ ( _**z**_ ) _t_ ��] _≤_ _λ_ �� _**x**_ _t−_ 1 _−_ _**x**_ ˆ _St−_ 1�� +









2
��� _F_ ¯ _n,R,j_ _**θ**_ _,S_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] _[ −]_ _[F]_ [¯] _[ n,]_ _R,j_ _**[θ]**_ [(ˆ] _**[x]**_ _t_ _[S]_ _−_ 1 _[,]_ _**[ z]**_ _[t]_ [)] ��� 


1 _/_ 2 []


+ E


_N_

 





(48)








_j_ =1


_≤_ _λ_ �� _**x**_ _t−_ 1 _−_ _**x**_ ˆ _St−_ 1�� +


_√_


_N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ + _√_
_n_


_N_ ~~_√_~~ [4] _[R]_

_S_


with the expectations again taken with respect to sampling the probabilities to evaluate
_F_ ¯ _R,j_ _[n,]_ _**[θ]**_ _[,S]_ (ˆ _**x**_ _[S]_ _t−_ 1 _[,]_ _**[ z]**_ _[t]_ [)][.] [In particular, taking expectations also with respect to a random process] **[ Z]** [ (taking]
values in ( _Dd_ ) [Z] _[−]_ ) and sampling at each evaluation, the estimate (48) and the same arguments as in
the proof of Theorem 4.6 yield the bound


1
sup E[�� _U F_ ( **Z** ) _t −_ _U_ ¯ _S_ ( **Z** ) _t_ ��] _≤_
_t∈_ Z _−_ 1 _−_ _λ_


_√_

- _N_ max ~~_√_~~ _j_ =1 _,...,N_ _Cj_ _[∞]_ + _√_
_n_


_N_ ~~_√_~~ [4] _[R]_

_S_


_N_ ~~_√_~~ [4] _[R]_


_._ (49)


28
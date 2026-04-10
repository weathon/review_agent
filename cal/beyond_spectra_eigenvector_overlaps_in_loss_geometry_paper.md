# BEYOND SPECTRA: EIGENVECTOR OVERLAPS IN LOSS GEOMETRY


**Gabriel C. Mel** [1] _[,]_ [2]

1 Centre de Recerca Matem`atica, Bellaterra (Barcelona), Spain
2 CRG (Barcelona Collaboratorium for Modelling and Predictive Biology),
Dr. Aiguader 88, Barcelona 08003, Spain
meldefon@gmail.com


ABSTRACT


Local loss geometry in machine learning is inherently a two-operator concept.
While a single loss is locally characterized by its Hessian spectrum, practical
learning depends on both training and test losses, whose joint geometry is determined not only by their spectra but by the alignment of their eigenspaces. We
establish general foundations for this two-loss geometry by deriving a universal local fluctuation law: the expected test-loss increment under small training perturbations is a trace combining train and test spectral data with a precise factor quantifying eigenvector overlap. We further prove a transfer law describing how overlaps
transform under noise. As a solvable model, we apply these results to ridge regression under arbitrary covariate shift, where operator-valued free probability yields
asymptotically exact overlap decompositions that identify overlaps as the natural
quantities for specifying shift, and resolve multiple descent: error peaks are governed by eigenspace misalignment rather than Hessian ill-conditioning alone. We
then validate the fluctuation law in multilayer perceptrons, develop scalable estimators for overlap functionals based on subspace iteration and kernel polynomial
methods, and apply them to a ResNet-20 trained on CIFAR-10, showing that class
imbalance reshapes train–test geometry through induced misalignment. Together,
these results establish eigenvector overlaps as the fundamental missing ingredient
in local loss geometry, providing both theoretical foundations and practical tools
for analyzing generalization in modern neural networks.


1 INTRODUCTION


Modern learning algorithms are inherently local, and sources of randomness (stochastic gradients,
finite-sample variability, and distributional drift) are often small relative to the underlying signal.
A local quadratic approximation to the loss thus provides a natural setting for analyzing learning.
When the focus is a single loss, local geometry is fully captured by the Hessian spectrum. Crucially,
however, machine learning involves (at least) _two_ losses—train and test—and so local loss geometry
involves two quadratic approximations. Their joint geometry is not captured by Hessian spectra
alone; it requires a critical additional ingredient: _eigenvector alignment_, or overlaps.


Despite the fundamental importance of eigenvector overlaps, most studies to date have centered on
Hessian eigenvalue distributions—often explicitly equating spectra with loss geometry. The literature is extensive and examines Hessians from several complementary angles, including: (i) empirical
measurement of eigenvalue distributions and their training-time evolution, with links to optimization
stability (Sagun et al., 2017; Ghorbani et al., 2019; Yao et al., 2019); (ii) random-matrix-theoretic
and mean field models (Pennington & Bahri, 2017; Pennington & Worah, 2018; 2019; Liao & Mahoney, 2021; Karakida et al., 2019); (iii) class- and layer-structured spectral phenomena, such as
identifiable outliers tied to data and architecture (Papyan, 2020; Sankar et al., 2021); and (iv) Hessian/Fisher analyses relating sharpness (as measured by eigenvalue magnitude) to stability and generalization (Keskar et al., 2017; Cohen et al., 2021; Yao et al., 2019). These studies give fundamental insight into aspects of loss curvature, but ignore directional information that becomes relevant as
soon as one compares two operators.


1


The need to go beyond spectra is well understood in random matrix theory, where eigenvector consistency and overlaps with population directions are central objects in spiked models and correlated
ensembles (Johnstone, 2001; Paul, 2007; Nadler, 2008; Benaych-Georges & Nadakuditi, 2011; Bun
et al., 2017; Landau et al., 2023). There, eigenvalues alone do not determine statistical performance;
rather, risk depends on how sample and population eigenvectors align. Related phenomena have
also been observed in machine learning, where eigenspace overlap has been used to characterize
shared Hessian structure across independently trained networks (Wu et al., 2022) and to predict
compressed embedding performance (May et al., 2019). Here, we adapt this principle to learningtheoretic questions by analyzing train-test alignment. We show that overlap measures between the
training fluctuations covariance (intimately related to the training Hessian; see Results) and the test
Hessian yield a decomposition of generalization error into spectral and alignment components.


Applying this perspective to ridge regression resolves the puzzle of anisotropy-induced multiple descent (see Chen et al. (2021); Li & Wei (2021); Mel & Ganguli (2021); Meng et al. (2023) for several
distinct forms of multiple descent). Existing high-dimensional analyses connect interpolation peaks
to eigenvalue distribution of the design matrix (Singh et al., 2022; Chen & Mei, 2022). However, in
anisotropic settings where error exhibits multiple peaks despite monotonically decreasing minimum
training eigenvalue, spectra alone do not explain generalization. Making overlaps explicit, we show
that the appearance of multiple peaks is governed by the alignment between train and test Hessian
eigenspaces. This corrects interpretations that implicitly attribute sample-wise multiple descent to
spectrum ill-conditioning (Chen & Mei, 2022; Mel & Ganguli, 2021; Mel & Pennington, 2022), and
yields a simple geometric picture that may prove useful for understanding more complex models.


A second arena where eigenvector orientation is essential is generalization under domain shift. In
covariate shift settings, high-dimensional risk formulas in random feature models (Tripuraneni et al.,
2021) can be naturally expressed in terms of train/test covariance spectra and their overlaps. More
broadly, many domain generalization methods encourage cross-domain invariance by aligning gradients, Fisher information, or Hessian statistics across domains (Rame et al., 2022; Hemati et al.,
2023; Le & Woo, 2024), or by imposing structural constraints such as elliptic regularization (Hasan
et al., 2025). These approaches seek robustness across _many_ possible unseen domains. Our perspective is complementary: rather than enforcing invariance, we derive explicit formulas for performance
on a _specific_ target domain. In particular, under covariate shift we show that, even with covariance
spectra held fixed, varying overlap structure alone can increase or decrease test risk. Overlaps therefore provide a natural quantitative measure of the shift itself, predicting when a given domain change
will help or hurt in a way that spectrum-only or domain-agnostic analyses cannot.


Loss curvature plays an important role in both classical and modern analyses of generalization.
Classical asymptotic corrections such as the Takeuchi Information Criterion (TIC) express the generalization gap in terms of the local curvature of the population loss (see, eg., Thomas et al. (2020)).
A second line of work uses curvature information at training time: sharpness-aware and curvatureregularized methods—including SAM (Foret et al., 2021) and its Fisher- and curvature-regularized
variants (Kim et al., 2022; Wu et al., 2024)—bias optimization toward flatter regions of the training
loss, motivated by the heuristic that such regions generalize better. Both perspectives are fundamentally single-loss. By contrast, our framework is explicitly two-loss: we do not assume any
relationship between train and test losses. Given a training loss and perturbation model, we characterize test performance—potentially on a distinct domain—through the joint spectra of the two
Hessians and their eigenvector overlap. The TIC emerges as a limiting single-loss case, but our
theory reveals how alignment governs generalization beyond spectrum-based criteria.


Translating our overlap theory into practice at modern scale requires algorithms that go beyond
spectral density estimation. A substantial literature has developed linear algebraic tools for implicit matrices, including polynomial/quadrature approaches and stochastic trace methods such as
Hutchinson and Lanczos-based quadrature (Golub & Meurant, 2009; Lin et al., 2016; Ubaru et al.,
2017). These and related techniques have been adapted to deep learning to estimate Hessian spectral densities and extremal eigenpairs efficiently (Adams et al., 2018; Papyan, 2019; Ghorbani et al.,
2019; Yao et al., 2019). Building on these, we develop novel estimators for overlap functionals
between pairs of Hessians (train-test, population-sample), and use them to study class imbalance
effects on train–test geometry.


2


The resulting picture is that local geometry in machine learning is fundamentally bivariate: spectra
determine the curvatures of train and test losses, while eigenvector overlaps determine how these
curvatures interact to produce test error.


2 CONTRIBUTIONS


1. **Two-loss** **theory** **of** **local** **geometry.** We introduce a novel two-loss framework for local loss geometry that incorporates both spectra and overlaps 3.1, rectifying a widespread
oversimplification that equates spectra with geometry.

2. **General** **foundations.** We derive and test a universal local fluctuation law showing how
overlaps impact generalization 3.1.1, and a general transfer law dictating how eigenvector
overlaps are transformed by noise 3.1.2.

3. **Explicit formulas for high-dimensional ridge regression.** Combining tools from random
matrix theory with our overlap transfer law, we provide closed-form expressions for the
overlap function between train and test Hessians in anisotropic ridge regression 3.2.

4. **Unified explanation of covariate shift and multiple descent.** We show that covariate shift
is naturally quantified by eigenvector overlaps 3.2.1, and that overlaps analytically resolve
the puzzle of multiple descent 3.2.2.

5. **Empirical validation in neural networks.** We confirm our theoretical predictions in multilayer perceptrons, and use overlap machinery to show that the training Hessian acts as a
filter shaping optimization 3.3.

6. **Scalable algorithms for Hessian overlaps.** We develop novel, scalable numerical methods
for estimating Hessian eigenvector overlaps in large-scale models, enabling practical use
of our theory in modern deep learning 3.4.

7. **Train-test** **misalignment** **under** **class** **imbalance.** We show that class imbalance in CIFAR induces misalignment between train and test Hessians, explaining the effects of class
imbalance in terms of train-test loss geometry. 3.4.


3 RESULTS


3.1 THEORETICAL FOUNDATIONS


Prior theoretical and empirical work frequently uses geometric descriptors of the loss landscape—
such as “sharp” versus “flat” minima or valley structures—yet the relationship between these geometric notions and generalization remains imprecise. We begin by establishing general foundations
for two-loss geometry to formalize this connection, and then derive a fluctuation law that characterizes how perturbations to the training loss propagate to changes in the test loss.


Let _w_ _∈_ R _[d]_ denote the _d_ -dimensional parameter vector of a model _fw_, and let _L_ train( _w, ϵ_ ) _, L_ test( _w_ )
be the (twice-differentiable) train and test loss functions. The train loss _L_ train is parameterized by a
small variable _ϵ_ representing a general training perturbation. We remain agnostic about the source of
the perturbation, which could be any combination of label/input noise, distributional drift, sampling
effects, etc. Throughout, we write _w_ 0 for the minimum of the unperturbed loss _L_ train( _w,_ 0).


By analogy with the one-loss case, we refer to the pair of quadratic approximations obtained by
second order expansion of _L_ train _, L_ test around a point as the _local two-loss geometry_ . Concretely,
we define the perturbation gradient, and train and test Hessians as follows:

_z_ := _d ∇wL_ train ( _w_ 0 _, ϵ_ ) _,_ _H_ train := _d ∇_ [2] _w_ _[L]_ [train][(] _[w]_ [0] _[, ϵ]_ [)] _[,]_ _H_ test := _d ∇_ [2] _w_ _[L]_ [test][(] _[w]_ [0][)] _[,]_ (1)


(note the scalings, chosen for convenience) and introduce the quadratic surrogate losses:


_L_ [quad] train [(] _[w]_ [) =] _[ L]_ [train][ (] _[w]_ [0] _[, ϵ]_ [) +] _d_ [1]


2 [1] _d_ [∆] _[w][⊤][H]_ [train][∆] _[w,]_ (2)

2 [1] _d_ [∆] _[w][⊤][H]_ [test][∆] _[w,]_ (3)


[1] [1]

_d_ _[z][ ·]_ [ ∆] _[w]_ [ +] 2


_L_ [quad] test [(] _[w]_ [) =] _[ L]_ [test] [(] _[w]_ [0][) +] _d_ [1]


[1] [1]

_d_ _[z]_ [test] _[ ·]_ [ ∆] _[w]_ [ +] 2


where _z_ test := _d ∇wL_ test( _w_ 0) is the normalized test gradient and ∆ _w_ := _w_ _−_ _w_ 0. Finally, we
define the unperturbed test loss and test loss increment as follows:

_L_ 0 := _L_ [quad] test [(] _[w]_ [0][)] _[,]_ ∆ _L_ := _L_ [quad] test [(] _[w]_ [0] [+ ∆] _[w]_ [)] _[ −L]_ [quad] test [(] _[w]_ [0][)] _[.]_ (4)


3


3.1.1 LOSS FLUCTUATIONS ARE GOVERNED BY EIGENVECTOR OVERLAPS


Generically, the effect of a perturbation is to induce a small gradient _z_ at the unperturbed minimum _w_ 0, yielding a new minimum of _L_ [quad] train [at a displacement][ ∆] _[w]_ [.] [Directly minimizing (2) gives]
the perturbation-induced displacement ∆ _w_ = _−H_ train _[−]_ [1] _[z]_ [.] [We] [sometimes] [refer] [to] [the] [perturbation]
gradient _z_ as the _injected noise_ and to ∆ _w_ as the _(inverse-Hessian) filtered noise_ . Substituting the
displacement into _L_ [quad] test [yields the following expression for the test loss increment,]


_d_ [1] _[z]_ test _[⊤]_ _[H]_ train _[−]_ [1] _[z]_ [ +] 2 [1]


∆ _L_ = _−_ [1]


2 [1] _d_ _[z][⊤][H]_ train _[−]_ [1] _[H]_ [test] _[H]_ train _[−]_ [1] _[z.]_ (5)


Equation (5) represents the simplest model capturing the interaction of nontrivial train and test geometry in the context of noisy learning. The first-order effect is structurally simple—and, in several natural cases (e.g., label noise under MSE, analyzed below), vanishes exactly in expectation.
The second order term, in contrast, involves interaction between train and test curvatures: letting
_C_ train := E[∆ _w_ ∆ _w_ _[⊤]_ ] = E[( _H_ train _[−]_ [1] _[z]_ [)(] _[H]_ train _[−]_ [1] _[z]_ [)] _[⊤]_ []][ be the displacement covariance,] [its expectation]
is 21 _d_ [tr][[] _[H]_ [test] _[C]_ [train][]][.] [This] [simple] [trace] [expression] [suggests] [the] [importance] [of] [alignment] [between]
directions of large training displacement and directions of large test Hessian eigenvalue. One of the
main theoretical contributions of this work is the following theorem making this intuition precise.

**Theorem** **1** (Overlap local fluctuation law) **.** _Let_ _µ_ test _, µ_ train _be_ _the_ _spectral_ _measures_ _of_
_H_ test _, C_ train _,_ _and_ _define_ _d_ [1] _[O]_ [(] _[λ]_ [1] _[, λ]_ [2][)] _[as]_ _[the]_ _[mean]_ _[squared]_ _[cosine]_ _[angle]_ _[between]_ _[eigenvectors]_ _[of]_

_H_ test _, C_ train _at eigenvalues λ_ 1 _, λ_ 2 _._ _Assume_ E[∆ _w_ ] = 0 _._ _Then_


E[∆ _L_ ] = 21


��
_λ_ 1 _λ_ 2 _O_ ( _λ_ 1 _, λ_ 2) _µ_ test( _dλ_ 1) _µ_ train( _dλ_ 2) _._ (6)


Equation (6) shows that neither train nor test spectra alone predict the expected generalization impact of noise: the decisive quantity is how training-induced displacement directions route into testsensitive directions via the overlap kernel _O_ ( _λ_ 1 _, λ_ 2). In particular, large test error arises when when
high-variance displacement directions (large _λ_ 2, corresponding to low-curvature train directions)
substantially overlap high-curvature test directions (large _λ_ 1).


_Proof sketch._ Letting ( _λ_ [test] _i_ _, u_ [test] _i_ ) _,_ ( _λ_ [train] _j_ _, u_ [train] _j_ ) be the eigenvalues/eigenvectors of _H_ test _, C_ train,


_d_

- _λ_ [test] _i_ _λ_ [train] _j_ [ _d_ ( _u_ [test] _i_ _· u_ [train] _j_ ) [2] ] _._ (7)


_j_ =1


1 [1] 1
2 _d_ [tr][[] _[H]_ [test] _[C]_ [train][] =] 2 _d_ [2]


_d_


_i_ =1


Writing the double sum as an integral over the spectral measures of _H_ test _, C_ train yields (6). See
Appendix B.2 for details.


While we do not treat stochastic optimization explicitly, in the same local quadratic regime, noisy
gradient descent yields a curvature-filtered steady-state covariance that, when substituted for _C_ train,
yields the same overlap fluctuation law (see Appendix B.2.2).


3.1.2 OVERLAP TRANSFER LAW


In many situations one must consider the overlaps between an operator _A_ and a noisy transformation of another operator _B_, written _B_ [ˆ] . For example, below in the context of ridge regression with
anisotropic gaussian inputs, we consider the case that _A, B_ correspond to the population test and
train covariances, while _B_ [ˆ] is the _sample_ train covariance. More generally, _A, B_ could represent the
population test and train Hessians, and _B_ [ˆ] the empirical train Hessian. In such cases, one needs a
way of combining the population overlaps _OA,B_ with the noise, specified by _OB,_ ˆ _B_ [.] [We prove the]
following appealing transfer law in Appendix B.3:

**Theorem** **2** (Free transfer law for overlap functions) **.** _Let_ _B_ [ˆ] = _F_ ( _B, X_ ) _be_ _a_ _matrix_ _rational_
_expression._ _If X_ _is free from A, B, then_


               _OA,_ ˆ _B_ [(] _[a,]_ [ˆ] _[b]_ [) =] _OA,B_ ( _a, b_ ) _OB,_ ˆ _B_ [(] _[b,]_ [ˆ] _[b]_ [)] _[ µ][B]_ [ (] _[db]_ [)] _[ .]_ (8)


4


(Freeness is a notion of independence that is suited to large random matrices and holds asymptotically for a wide range of common random matrix models; see Appendix B.3.) Theorem 2 entails
a simple overlap calculus that can be used to compute overlap functions of complex matrix models
from simpler ones. In Appendix C, we use (8) to quickly derive expressions for train-test Hessian
overlap functions in anisotropic ridge regression.


3.2 HESSIAN OVERLAPS GOVERN GENERALIZATION IN LINEAR REGRESSION


We now consider ridge regression, where the preceding theory is exact. Let training inputs _x_ _∈_ R _[d]_
have covariance Σtrain := E[ _xx_ _[⊤]_ ], and assume linear output with Gaussian label noise:

_y_ ( _x_ ) = ~~_√_~~ 1 _d_ _[w]_ _∗_ _[⊤][x]_ [+] _[ξ,]_ _ξ_ _∼N_ (0 _, σ_ [2] ) _._

We will also assume for convenience that _w∗_ _∼N_ (0 _, I_ ), so that the signal to noise ratio is
tr¯ Σtrain _/_ (¯tr Σtrain + _σ_ [2] ) (tr¯ denotes the dimension normalized trace). Given a training set consisting of _X_ _∈_ R _[m][×][d]_ (rows _x_ _[⊤]_ ) and labels _y_ _∈_ R _[m]_, ridge regression chooses _w_ _∈_ R _[d]_ to minimize

_L_ train( _w_ ) = 21 _m_ �� _y −_ ~~_√_~~ 1 _d_ _[Xw]_ ��2 + 2 _λd_ _[∥][w][∥]_ [2] _[,]_ _λ ≥_ 0 _._ (9)


We write _α_ := _m/d_ for the sampling ratio. The (excess) test loss is measured with test inputs with
(possibly different) covariance Σtest:


[1] 2 [E] _[x,ξ]_ �� ~~_√_~~ 1


2 [1] _[σ]_ [2] [=] 21 _d_ [(] _[w][ −]_ _[w][∗]_ [)] _[⊤]_ [Σ][test][ (] _[w][ −]_ _[w][∗]_ [)] _[.]_


_L_ test( _w_ ) := [1]


_d_ _[w][⊤][x][ −]_ _[y]_ [(] _[x]_ [)] �2 [�] _−_ 2 [1]


With the scalings of 3.1, one has _H_ train = Σ [ˆ] train + _λI_ and _H_ test = Σtest, where Σ [ˆ] train := _X_ _[⊤]_ _X/m_
is the training set sample covariance. Note Σ [ˆ] train _→_ Σtrain for large _α_ = _m/d_ .


We now apply the local fluctuation formula (6), which in the setting of ridge regression is exact. Our
goal is not to re-derive known high-dimensional risk formulas, but to highlight their two-operator
structure: test error decomposes into (i) train/test spectral scales and (ii) an explicit eigenspacealignment kernel. This makes covariate shift and multiple descent analyzable as overlap phenomena. Letting the label noise supply the perturbation, the injected noise _z_ := _d∇wL_ train( _w_ 0 _, ξ_ ) and
displacement covariance _C_ train := E [( _H_ train _[−]_ [1] _[z]_ [)(] _[H]_ train _[−]_ [1] _[z]_ [)] _[⊤]_ []][ are]


~~_√_~~
_z_ = _−_


_md_ _[X]_ _[⊤][ξ,]_ _C_ train = _σ_ [2] _α_ _[−]_ [1] Σ [ˆ] train(Σ [ˆ] train + _λI_ ) _[−]_ [2] _,_ (10)


where _α_ := _m/d_ is the sampling ratio. The test loss increment ∆ _L_ is obtained by substituting
into the overlap formula (6). The training-side operators Σ [ˆ] train, _H_ train = Σ [ˆ] train + _λI_ and _C_ train
commute and share eigenvectors, so for simplicity all formulas are written in terms of Σ [ˆ] train:


E [∆ _L_ ] = 2 _σα_ [2]


�� _λ_ 2
_λ_ 1 (11)
( _λ_ 2 + _λ_ ) [2] _[O]_ [Σ][test] _[,]_ [ˆΣ][train][(] _[λ]_ [1] _[, λ]_ [2][)] _[ µ]_ [Σ][test][(] _[dλ]_ [1][)] _[ µ]_ [ˆΣ][train][(] _[dλ]_ [2][)] _[,]_


where _µ_ Σtest and _µ_ ˆΣtrain are the empirical spectral measures, and _O_ Σtest _,_ ˆΣtrain( _λ_ 1 _, λ_ 2) is the
eigenvector-overlap function (see (18)). Since we will be interested primarily in the ridgeless limit
_λ →_ 0, we will loosely refer to Σ [ˆ] train as the train Hessian. See Appendix C for detailed derivations.


The fundamental conclusion from (11) that we will apply toward analyzing covariate shift and multiple descent is that error is large when training perturbations induce large variance (small training
eigenvalue _λ_ 2) in directions that align strongly (large _O_ ( _λ_ 1 _, λ_ 2)) with directions of large test loss
sensitivity (large test eigenvalue _λ_ 1).


In Appendix C, using techniques from operator-valued free probability we derive asymptotically
exact expressions for _L_ test _,_ ∆ _L_ and the overlap function _OH_ train _,H_ test in proportional asymptotics
where _m, d →∞_ with _α_ := _m/d_ fixed. The main conceptual contribution of this work is that while
the spectral densities of train/test operators set the relevant scales, it is their relative orientation—
as quantified by the overlap function—that determines how displacements translate into test loss.
We illustrate these points in two settings: first, a simple covariate shift experiment that provides
geometric intuition and positions _O_ Σtest _,_ Σtrain as the natural object quantifying shift; second, the
puzzle of multiple descent (cf. Mel & Ganguli (2021)), where the overlap function allows a full
analytical account. For clarity, in both settings we use the simplest possible model of anisotropic
data: the “two-scale” covariance with spectral measure

_µ_ Σ := _p_ 1 _δs_ 21 [+] _[ p]_ [2] _[δ][s]_ 2 [2] _[.]_ (12)


5


Figure 1: Isospectral shifts isolate overlap effects in covariate shift. (a) Two-dimensional slice of
parameter space. Points: individual parameter displacements due to label noise. Lines show train
(cyan) and test (red) Hessians with precisely controlled alignment _θ_ . (b) Eigenspaces are rotated
while Hessian spectra are fixed. Blue line shows _d, m_ _→∞_ theory. (c) For small _θ_, large displacements in learned parameters are aligned with the low-eigenvalue directions of the test Hessian
(aligned ellipses at top left, and purple overlap lines), and error is small (black line). For large _θ_, large
displacement directions are aligned with sensitive directions of the test loss and test error is large
(black line). Lines: theory; crosses: average from simulations. _d, α, λ, σ_ = 10 [2] _,_ 10 _,_ 10 _[−]_ [4] _,_ 10 _[−]_ [1] _[/]_ [2] .


3.2.1 COVARIATE SHIFT INCREASES LOSS THROUGH TRAIN-TEST MISALIGNMENT


Equation (11) expresses ∆ _L_ in terms of the overlap function _O_ Σtest _,_ ˆΣtrain. Relative to the population
overlap _O_ Σtest _,_ Σtrain, this overlap is deformed by the finite sampling ratio of the training set (cf.
transfer law of Theorem 2). In Appendix C.3 we use the transfer law to state an explicit formula for
_O_ Σtest _,_ ˆΣtrain, and then prove the following:


**Theorem 3.** _As m, d →∞_ _with α fixed, the asymptotic test loss increment satisfies_


_dλ_ [˜]

E[∆ _L_ ] _→_ _[σ]_ [2]

2 _α_ _dλ_


�� _λ_ 2
_λ_ 1 (13)
( _λ_ 2 + _λ_ [˜] ) [2] _[O]_ [Σ][test] _[,]_ [Σ][train][ (] _[λ]_ [1] _[, λ]_ [2][)] _[ µ]_ [Σ][test][ (] _[dλ]_ [1][)] _[ µ]_ [Σ][train][ (] _[dλ]_ [2][)] _[,]_


_where_ _λ_ [˜] _is the effective regularization defined by the self-consistent equation:_


_λ_    _λ_ ˜ := _r_ ( _−λ_ ) _[,]_ _r_ ( _z_ ) = 1 _−_ _α_ [1]


- _t_ - _−_ 1
_._ (14)
_z −_ _t r_ ( _z_ ) _[dµ]_ [Σ][train][ (] _[t]_ [)]


Equation (13) parallels (11) but averages out all training randomness to express ∆ _L_ purely in terms
of the population operators Σtrain _,_ Σtest. Most importantly, (13) illustrates how _O_ Σtest _,_ Σtrain—as
the only quantity that can change under isospectral transformations to Σtrain _,_ Σtest—captures bona
fide two-loss geometric effects that are invisible from either loss geometry in isolation.


To illustrate this point, we perform a simple experiment where both Σtrain _,_ Σtest have fixed twolevel spectra (12) with scales _s_ [2] 1 _[, s]_ [2] 2 [= 2][0] _[,]_ [ 2] _[−]_ [4] [and equal multiplicities.] _[λ]_ [ = 10] _[−]_ [4] [and] _[ α]_ [=] _[ m/d]_ [ =]
10 so that _H_ train _≈_ Σ [ˆ] train _≈_ Σtrain, while _H_ test = Σtest. Fig. 1(a) shows the distribution of
learned parameter displacements for different label noise realizations. As predicted, displacements
have larger variance along long directions of _C_ train _≈_ _σ_ [2] Σ _[−]_ train [1] _[/α]_ [, corresponding to low-curvature]
directions of the train Hessian. At the same time, the test loss contours are determined by the test
Hessian Σtest. We construct a controlled perturbation in which Σtest is systematically rotated with
respect to Σtrain while all spectra are kept fixed (b), isolating the effect of overlaps. Fig. 1(c)
demonstrates the consequence of varying overlap. When the long directions of Σtrain align with the
long directions of Σtest ( _θ_ = 0), displacements occur in directions where the test error is relatively
flat, yielding low excess test loss (Fig. 1(c), left column). In contrast, when the same train-long
directions align with test-short directions ( _θ_ = _π/_ 2), the same magnitude of parameter displacement
is heavily penalized, and the test loss rises sharply (Fig. 1(c), right column). This simple experiment
illustrates the central role of eigenvector overlaps in the context of covariate shift.


6


Figure 2: Multiple descent in ridge regression. (a) Loss as a function of the measurement density _α_ = _m/d_ for two-scale data. Note the peaks at critical values of _α_ = 1 _/_ 2 _,_ 1. Solid lines:
theory; crosses: simulations with _d_ = 5000. Dashed, dotted, and dash-dotted lines show theory
_L_ test _,_ ∆ _L, L_ 0 in the limit that the lower scale _s_ 2 _→_ 0, where bumps become true singularities.
(b) Theory _L_ test( _α, λ_ ). Gold and blue lines indicate slices shown in panel (a). (c) 3 and 4 scale
data which exhibit 3 and 4 peaks; legend same as (a). (d) Green histogram: empirical spectral
density of the train Hessian _X_ _[⊤]_ _X/m_ at _α_ = 0 _._ 496; solid green line: theory. Purple line: overlap
function, _O_ ( _s_ [2] 1 _[, λ]_ [2][)] _[/]_ [2][,] [giving] [overlap] [between] [a] [train] [eigenspace] [at] [eigenvalue] _[λ]_ [2] [with] [the] [entire]
large-eigenvalue test space (ie. _s_ [2] 1 [).] [Note strong overlap for high train/test eigenspaces.]


3.2.2 MULTIPLE DESCENT IS EXPLAINED BY TRAIN-TEST OVERLAPS


Double descent is a well-established phenomenon in machine learning in which test error exhibits
a non-monotonic dependence on model size. More recently, several authors have described an extension of this effect, termed multiple descent, which arises in settings where input data are highly
anisotropic and the covariance spectrum contains multiple separated scales (see introduction). Fig.
2(a,b) illustrate multiple descent for two-scale data with _s_ 1 _, s_ 2 = 1 _,_ 10 _[−]_ [2], while panel c shows how
a larger number of separated scales can create additional peaks in test error (see caption for details).


For a two-level covariance, the overlap function is determined by the solution to a cubic polynomial
that is easily solved numerically (Appendix C). Fig. 2(d) shows the spectrum of the train Hessian
(green histogram and theory line), and the overlap function (18), indicating overlap of a training
eigenspace at eigenvalue _λ_ 2 with the large-eigenvalue ( _s_ [2] 1 [) eigenspace of the test Hessian.] [Theoreti-]
cal and empirical overlaps are in excellent agreement (purple line and crosses).


The peaks of multiple descent are easily understood in terms of eigenvector overlaps. Fig. 3 reports
the error, training spectrum, and overlap map for the two-scale covariance model of Fig. 2. The testloss curve shows two singularities at critical sampling densities _α_ = _m/d_ (a). At the same densities
the training spectrum undergoes phase transitions: at _α_ = [1] _/_ 2 an initially unimodal density splits into
two bands centered near _s_ [2] 1 [and] _[ s]_ 2 [2][, and at] _[ α]_ [ = 1][ the lower] _[ s]_ [2] 2 [band develops a near-zero component]
(Fig. 3(b)). The corresponding overlap map _O_ ( _λ_ 1 _, λ_ 2) is approximately block-diagonal: modes
near _s_ [2] 1 [align] [predominantly] [with] [the] _[s]_ [2] 1 [test] [subspace,] [and] [modes] [near] _[s]_ [2] 2 [with] [the] _[s]_ [2] 2 [subspace]
(Fig. 3(c)). Thus, the first error spike occurs when near-null training directions overlap the sharp
test subspace, whereas the second arises when an even smaller training component overlaps the
flat subspace but with variance large enough to dominate its small curvature. Fig. 3(d) provides
a geometric schematic of the alignment of top and bottom eigenspaces of _H_ train _, H_ test throughout
this sequence. Until line 5, the minimum eigenvalue of _H_ train always decreases as a function of
_α_ —which, according to a spectrum-only analysis, should increase test error. Yet the error actually
_decreases_ between horizontal lines 3 and 4, precisely because the lowest train eigenspaces begin to
overlap predominately with the low test eigenspace.


Summarizing, multiple descent arises from the interplay of (i) training components developing nearzero eigenvalues as _α_ varies, and (ii) which test directions these overlap with—sharp or flat, illustrating the potentially extreme impact of (mis-)aligned train and test loss geometry.


7


Figure 3: Multiple descent is explained by train-test Hessian overlaps. (a) ∆ _L_ due to label noise
( _λ, s_ [2] 2 _[→]_ [0][ limits taken to illustrate true singularities; see Appendix C.5.1 for details).] [(b) Spectral]
density of _H_ train as a function of _α_ (each density normalized to a maximum of 1 and quantized). For
high sampling density _α_, the training density acquires two components roughly centered on the true
underlying data scales _s_ [2] 1 _[, s]_ [2] 2 [.] [(c) The overlap function of] _[ H]_ [train] _[, H]_ [test] [as a function of] _[ α]_ [.] [For small]
_α_, all nonzero _H_ train spaces overlap strongly with the large eigenspace of _H_ test. As _α_ approaches
the first critical value, a new spectral component appears, whose eigenspaces overlap almost entirely
with the small eigenspace of _H_ test. (d) Minimal model of train-test Hessian geometry. Cyan plane
and ellipses represent the top and bottom eigenspaces of _H_ train. Red ellipsoid represents level sets
of test error. Error is controlled by both train variance magnitude and overlap onto test spaces.


3.3 LOCAL THEORY PREDICTS MLP GENERALIZATION AND LEARNING DYNAMICS


To test the quantitative predictions of the quadratic two-loss theory in a controlled nonconvex setting,
we trained small, constant width multilayer perceptrons (MLPs) to reproduce the responses of an
MLP teacher network. Student networks were batch trained for a large number of iterations to ensure
near convergence to the noiseless training loss minimum. Noise was then added to the training set
and the network was trained further—beginning from the initial trained state to determine the effect
of the noise on the initial local minimum. After training, the training loss increment was computed
and compared to prediction of the local quadratic theory. Fig. 4(a,b) show the measured test loss
increment against the local quadratic prediction for several orders of magnitude of input (a) and
label (b) noise strength. All later panels refer to the noise setting corresponding to the red point in
(b).


Fig. 4(c) illustrates inverse Hessian filtering due to training dynamics. The gradient noise induced by the label noise has covariance E[ _zz_ _[⊤]_ ]. The purple scatter represents the overlap function of E[ _zz_ _[⊤]_ ] and _H_ train. Dot _x, y_ position is given by _H_ train, E[ _zz_ _[⊤]_ ] eigenvalue and size
is proportional to overlap. Note that the gradient noise and train Hessian are strongly aligned.
After training, the parameter displacement covariance predicted by quadratic approximation is
_C_ train := E [( _H_ train _[−]_ [1] _[z]_ [)(] _[H]_ train _[−]_ [1] _[z]_ [)] _[⊤]_ []][.] [The] [overlap] [function] [of] _[H]_ [train] [and] [the] [actual] [post-training]
covariance is plotted in red. Note how in accordance with quadratic predictions, variance is strongly
inflated/attenuated along low/high eigendirections of _H_ train—a phenomenon we refer to as inverse
Hessian filtering. The large displacements do not translate into large test error since the train and test
Hessians are well aligned (Fig. 6), meaning displacements occur primarily along low test Hessian
(loss-insensitive) directions.


To provide geometric intuition, loss landscape slices are shown in Fig. 4(d) for _L_ train( _w,_ 0),
_L_ train( _w, ϵ_ ), and _L_ test. A single 2D slice was chosen to contain the unperturbed minimum _w_ 0
(white crosses), perturbed minimum (white stars), and parameters predicted by the local quadratic
approximation (white “Y”s). Local geometry also predicts local gradient descent dynamics well
(Appendix E.1; Fig. 7). Together, these results validate the predictions of the two-loss local theory.


8


Figure 4: Validation of local fluctuation law in MLPs. Layer widths for both student and teacher
were (5,5,5,1); nonlinearity: tanh; teacher network has gaussian weights with scale 4 _/_ _[√]_ _din_ for
each layer; loss: MSE with _ℓ_ 2 parameter _λ_ = 1. (a,b) Predicted vs measured perturbationinduced ∆ _L/L_ 0 for increasing input (a) or label (b) noise amplitude _σ_ . (c) Eigenvector overlap
function between _H_ train and the injected noise E[ _zz_ _[⊤]_ ] (purple), and post-learning filtered noise
_C_ train (red). Note how learning inflates/dampens variance along the low/high _H_ test eigenspaces.
(d) 2-dimensional loss landscapes for 1 example simulation: noiseless _L_ train (left), perturbed _L_ train
(middle), and _L_ test (right). Cross: noiseless training minimum; star: minimum of perturbed _L_ train
(ie. the new learned minimum); tri-star: parameters predicted by quadratic theory.


3.4 CALCULATION OF OVERLAP FUNCTIONS FOR LARGE SCALE NETWORKS


Applying our theory to modern networks requires estimating the overlap function between the training and test operators. These operators have dimension equal to the number of parameters—often
millions to billions—so any approach that forms them explicitly is infeasible.


Here we give a brief overview of our approach, deferring details to Appendix F. We apply two separate algorithms, one for computing overlaps among outlier eigenspaces and another for the remaining “bulk” spaces. Outlier eigenvectors are straightforward to obtain using subspace iteration (Appendix F.2; cf. Papyan (2019)); overlaps can then be computed directly. For the bulk eigenspaces,
we generalize a well-known approach to spectral density estimation known as the kernel polynomial
method (KPM; Algorithm 1 in Appendix F.3).


Given self-adjoint matrices _A, B_ _∈_ R _[d][×][d]_ and arbitrary smoothing kernels _G_ ( _x_ ; _σ_ ) of width _σ_, the
smoothed total eigenvector overlap of _A, B_ at eigenvalues _λ_ 1 _, λ_ 2 can be written


tr¯ [ _GA,λ_ 1 _GB,λ_ 2] = _d_ [1][2]


_d_

- _G_ ( _λA,i −_ _λ_ 1; _σ_ ) _G_ ( _λB,j_ _−_ _λ_ 2; _σ_ ) - _d_ ( _vA,i · vB,j_ ) [2][�] _,_ (15)


_i,j_ =1


where _GA,λ_ 1 := _G_ ( _A −_ _λ_ 1 _I_ ; _σ_ ) and similarly for _GB,λ_ 2. To obtain the normalized overlap function
treated above, one simply divides by the ( _G_ -smoothed) spectral densities of _A, B_ at _λ_ 1 _, λ_ 2.


Computing the trace in (15) is prohibitively expensive for large _A, B_, and so we resort to Hutchinson
trace estimation, which approximates tr [ _X_ ] with the average of _v_ _[⊤]_ _Xv_ for several samples of _v_ _∼_
_N_ (0 _, I_ ). To ensure the trace is positive, instead of approximating (15), we use

tr¯ [ _GA,λ_ 1 _GB,λ_ 2] = tr¯ [ _G_ [1] _A,λ_ _[/]_ [2] 1 _[G][B,λ]_ [2] _[G]_ _A,λ_ [1] _[/]_ [2] 1 [] =][ E] _[v][ ∥][G]_ _B,λ_ [1] _[/]_ [2] 2 _[G]_ _A,λ_ [1] _[/]_ [2] 1 _[v][∥]_ [2] _[.]_ (16)


9


Figure 5: Overlap function for _H_ train _, H_ test, and class imbalanced Hessian _H_ test _[′]_ [for] [ResNet-20]
trained on CIFAR-10. Cyan/red data: spectra of _H_ train _, H_ test. Dashed lines indicate outlier eigenvalues. Purple scatters show overlap between each pair of eigenspaces/bulk spaces. Size and color
reflect overlap magnitude. (a) _H_ train _, H_ test overlaps. Note strong alignment indicated by large overlaps along the diagonal. (b) _H_ train _, H_ test _[′]_ [overlaps.] [A large fraction of each Hessian’s outlier energy]
is lost in low-outlier and bulk spaces of the other, indicating poor alignment.


The KPM proceeds by taking the smoothing kernel _G_ ( _x_ ; _σ_ ) to be gaussian of width _σ_, and then
approximates _G_ [1] _A,λ_ _[/]_ [2] 1 _[, G]_ _B,λ_ [1] _[/]_ [2] 2 [using] [truncated] [Chebyshev] [series.] [(Kernel] [width] [and] [approximation]
degree _K_ are chosen so that the truncated series sufficiently dampens the large-multiplicity near-0
eigenspaces; see Appendix F.3.) Thus (16) can be evaluated in terms of the vectors _Ti_ ( _B_ ) _Tj_ ( _A_ ) _v_,
where _Tk_ is the _k_ _[th]_ Chebyshev polynomial. These vectors in turn can be generated efficiently via
Chebyshev recurrences using only matrix-vector products (see Appendix F for detailed treatment
and application to synthetic data).


We ran a simple controlled experiment to demonstrate the scalability of our Hessian-overlap algorithms on a modern network and to illustrate how a common form of domain shift—class imbalance
in the test set—produces a clear change in two-loss geometry. We used a publicly available CIFAR10 pretrained ResNet-20 checkpoint from Chen (top-1 test accuracy: 92 _._ 6%). The train Hessian was
estimated from 5000 examples and fixed throughout the experiment. Two-loss geometry was then
compared between two scenarios: a class-balanced test Hessian estimated from 5000 randomly selected test images, and a class-imbalanced Hessian from images with class labels 0 _,_ 1 and 2. Spectra,
estimated using subspace iteration and the Lanczos algorithm, are shown in Fig. 5(a) (train in cyan;
test in red). Non-outlier eigenspaces were grouped into a single bulk space for clarity. The strong
alignment observed between the train and balanced test Hessians largely disappears when the test set
is made unbalanced (purple scatters; bulk overlaps, omitted for space, exhibit similar pattern; Fig.
10). All Hessian-vector products were computed using standard PyTorch autograd on commodity
hardware, with total runtime of a few hours. Runtimes are essentially linear in the model size and
number of examples, underscoring the scalability of our method (see complexity analysis in F.3).


4 DISCUSSION


We show how, within a two-loss geometric framework, overlaps occupy a central role linking optimization geometry, random matrix theory, and practical machine learning phenomena. We derive
novel theoretical tools for computing overlaps, illustrate through several examples how spectra set
curvatures, while eigenvector overlaps route variance into error—unifying covariate shift and multiple descent—and develop scalable estimators for overlap analysis in large models. A natural application of two-loss geometry is as a diagnostic tool for explaining why some domain shifts are more
harmful than others. Promising future directions include tracking Hessian overlaps through training time, and _alignment-aware optimization_ that attempts to improve generalization by encouraging
strong eigenvector alignment between, eg., train and validation Hessians.


10


REFERENCES


Ryan P. Adams, Jeffrey Pennington, Matthew J. Johnson, Jamie Smith, Yaniv Ovadia, Brian Patton,
and James Saunderson. Estimating the spectral density of large implicit matrices, 2018. URL
[https://arxiv.org/abs/1802.03451.](https://arxiv.org/abs/1802.03451)


Florent Benaych-Georges and Raj Rao Nadakuditi. The eigenvalues and eigenvectors of finite,
low rank perturbations of large random matrices. _Advances_ _in_ _Mathematics_, 227(1):494–521,
2011. ISSN 0001-8708. doi: https://doi.org/10.1016/j.aim.2011.02.007. [URL https://www.](https://www.sciencedirect.com/science/article/pii/S0001870811000570)
[sciencedirect.com/science/article/pii/S0001870811000570.](https://www.sciencedirect.com/science/article/pii/S0001870811000570)


John P Boyd. _Chebyshev_ _&_ _Fourier_ _Spectral_ _Methods_ . Lecture Notes in Engineering. Springer,
Berlin, Germany, September 1989.


Jo¨el Bun, Jean-Philippe Bouchaud, and Marc Potters. Cleaning large correlation matrices: Tools
from random matrix theory. _Physics Reports_, 666:1–109, January 2017. ISSN 0370-1573. doi: 10.
1016/j.physrep.2016.10.005. [URL http://dx.doi.org/10.1016/j.physrep.2016.](http://dx.doi.org/10.1016/j.physrep.2016.10.005)
[10.005.](http://dx.doi.org/10.1016/j.physrep.2016.10.005)


Lin Chen and Song Mei. Spectral multiplicity entails sample-wise multiple descent, 2022. URL
[https://openreview.net/forum?id=qaQ8kUBYhEK.](https://openreview.net/forum?id=qaQ8kUBYhEK)


Lin Chen, Yifei Min, Mikhail Belkin, and Amin Karbasi. Multiple descent: Design your own
generalization curve, 2021. [URL https://arxiv.org/abs/2008.01036.](https://arxiv.org/abs/2008.01036)


Yaofo Chen. Pytorch cifar models. [https://github.com/chenyaofo/](https://github.com/chenyaofo/pytorch-cifar-models )
[pytorch-cifar-models.](https://github.com/chenyaofo/pytorch-cifar-models ) Accessed: 2025-5-17.


Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on
neural networks typically occurs at the edge of stability. In _International Conference on Learning_
_Representations_, 2021. [URL https://openreview.net/forum?id=jh-rTtvkGeM.](https://openreview.net/forum?id=jh-rTtvkGeM)


Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization. In _International Conference on Learning Represen-_
_tations_, 2021. [URL https://openreview.net/forum?id=6Tm1mposlrM.](https://openreview.net/forum?id=6Tm1mposlrM)


Behrooz Ghorbani, Shankar Krishnan, and Ying Xiao. An investigation into neural net optimization via hessian eigenvalue density. In Kamalika Chaudhuri and Ruslan Salakhutdinov
(eds.), _Proceedings_ _of_ _the_ _36th_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume 97 of
_Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 2232–2241. PMLR, 09–15 Jun 2019. URL
[https://proceedings.mlr.press/v97/ghorbani19b.html.](https://proceedings.mlr.press/v97/ghorbani19b.html)


Gene H Golub and Gerard Meurant. _Matrices, moments and quadrature with applications_ . Princeton
Series in Applied Mathematics. Princeton University Press, Princeton, NJ, December 2009.


Ali Hasan, Haoming Yang, Yuting Ng, and Vahid Tarokh. Elliptic loss regularization. In
_The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2025. URL [https:](https://openreview.net/forum?id=YwzxpZW3p7)
[//openreview.net/forum?id=YwzxpZW3p7.](https://openreview.net/forum?id=YwzxpZW3p7)


Sobhan Hemati, Guojun Zhang, Amir Estiri, and Xi Chen. Understanding hessian alignment for
domain generalization, 2023. [URL https://arxiv.org/abs/2308.11778.](https://arxiv.org/abs/2308.11778)


Iain M. Johnstone. On the distribution of the largest eigenvalue in principal components analysis.
_The_ _Annals_ _of_ _Statistics_, 29(2), April 2001. ISSN 0090-5364. doi: 10.1214/aos/1009210544.
[URL http://dx.doi.org/10.1214/aos/1009210544.](http://dx.doi.org/10.1214/aos/1009210544)


Ryo Karakida, Shotaro Akaho, and Shun-ichi Amari. Universal statistics of fisher information in
deep neural networks: Mean field approach. In Kamalika Chaudhuri and Masashi Sugiyama
(eds.), _Proceedings of the Twenty-Second International Conference on Artificial Intelligence and_
_Statistics_, volume 89 of _Proceedings of Machine Learning Research_, pp. 1032–1041. PMLR, 16–
18 Apr 2019. [URL https://proceedings.mlr.press/v89/karakida19a.html.](https://proceedings.mlr.press/v89/karakida19a.html)


11


Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. In
_International Conference on Learning Representations_, 2017. [URL https://openreview.](https://openreview.net/forum?id=H1oyRlYgg)
[net/forum?id=H1oyRlYgg.](https://openreview.net/forum?id=H1oyRlYgg)


Minyoung Kim, Da Li, Shell X Hu, and Timothy Hospedales. Fisher SAM: Information geometry and sharpness aware minimisation. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song,
Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings_ _of_ _the_ _39th_ _International_
_Conference_ _on_ _Machine_ _Learning_, volume 162 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_,
pp. 11148–11161. PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v162/kim22f.html)
[v162/kim22f.html.](https://proceedings.mlr.press/v162/kim22f.html)


Itamar D. Landau, Gabriel C. Mel, and Surya Ganguli. Singular vectors of sums of rectangular
random matrices and optimal estimation of high-rank signals: The extensive spike model. _Phys._
_Rev._ _E_, 108:054129, Nov 2023. doi: 10.1103/PhysRevE.108.054129. URL [https://link.](https://link.aps.org/doi/10.1103/PhysRevE.108.054129)
[aps.org/doi/10.1103/PhysRevE.108.054129.](https://link.aps.org/doi/10.1103/PhysRevE.108.054129)


Binh M Le and Simon S Woo. Gradient alignment for cross-domain face anti-spoofing. In _Proceed-_
_ings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_, pp. 188–199,
2024.


Yue Li and Yuting Wei. Minimum _ℓ_ 1-norm interpolators: Precise asymptotics and multiple descent,
2021. [URL https://arxiv.org/abs/2110.09502.](https://arxiv.org/abs/2110.09502)


Zhenyu Liao and Michael W. Mahoney. Hessian eigenspectra of more realistic nonlinear models.
In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), _Advances_ _in_ _Neu-_
_ral Information Processing Systems_, 2021. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=o-RYNVOlxA8)
[o-RYNVOlxA8.](https://openreview.net/forum?id=o-RYNVOlxA8)


Lin Lin, Yousef Saad, and Chao Yang. Approximating spectral densities of large matrices. _SIAM_
_Review_, 58(1):34–65, 2016. doi: 10.1137/130934283. [URL https://doi.org/10.1137/](https://doi.org/10.1137/130934283)
[130934283.](https://doi.org/10.1137/130934283)


Avner May, Jian Zhang, Tri Dao, and Christopher R´e. On the downstream performance of compressed word embeddings. _Advances in neural information processing systems_, 32:11782–11793,
2019. [URL https://api.semanticscholar.org/CorpusID:202538924.](https://api.semanticscholar.org/CorpusID:202538924)


Gabriel Mel and Surya Ganguli. A theory of high dimensional regression with arbitrary correlations
between input features and target functions: sample complexity, multiple descent curves and
a hierarchy of phase transitions. In Marina Meila and Tong Zhang (eds.), _Proceedings_ _of_ _the_
_38th_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume 139 of _Proceedings_ _of_ _Machine_
_Learning Research_, pp. 7578–7587. PMLR, 18–24 Jul 2021. [URL https://proceedings.](https://proceedings.mlr.press/v139/mel21a.html)
[mlr.press/v139/mel21a.html.](https://proceedings.mlr.press/v139/mel21a.html)


Gabriel Mel and Jeffrey Pennington. Anisotropic random feature regression in high dimensions. In
_International Conference on Learning Representations_, 2022. [URL https://openreview.](https://openreview.net/forum?id=JfaWawZ8BmX)
[net/forum?id=JfaWawZ8BmX.](https://openreview.net/forum?id=JfaWawZ8BmX)


Xuran Meng, Jianfeng Yao, and Yuan Cao. Multiple descent in the multiple random feature model,
2023. [URL https://arxiv.org/abs/2208.09897.](https://arxiv.org/abs/2208.09897)


James A. Mingo and Roland Speicher. _Free Probability and Random Matrices_ . Springer New York,
2017. ISBN 9781493969425. doi: 10.1007/978-1-4939-6942-5. URL [http://dx.doi.](http://dx.doi.org/10.1007/978-1-4939-6942-5)
[org/10.1007/978-1-4939-6942-5.](http://dx.doi.org/10.1007/978-1-4939-6942-5)


Boaz Nadler. Finite sample approximation results for principal component analysis: A matrix perturbation approach. _The_ _Annals_ _of_ _Statistics_, 36(6), December 2008. ISSN 0090-5364. doi:
10.1214/08-aos618. [URL http://dx.doi.org/10.1214/08-AOS618.](http://dx.doi.org/10.1214/08-AOS618)


Vardan Papyan. The full spectrum of deepnet hessians at scale: Dynamics with sgd training and
sample size, 2019. [URL https://arxiv.org/abs/1811.07062.](https://arxiv.org/abs/1811.07062)


12


Vardan Papyan. Traces of class/cross-class structure pervade deep learning spectra. _Journal_ _of_
_Machine Learning Research_, 21(252):1–64, 2020. [URL http://jmlr.org/papers/v21/](http://jmlr.org/papers/v21/20-933.html)
[20-933.html.](http://jmlr.org/papers/v21/20-933.html)


Debashis Paul. Asymptotics of sample eigenstructure for a large dimensional spiked covariance
model. _Statistica_ _Sinica_, 17(4):1617–1642, 2007. ISSN 10170405, 19968507. URL [http:](http://www.jstor.org/stable/24307692)
[//www.jstor.org/stable/24307692.](http://www.jstor.org/stable/24307692)


Jeffrey Pennington and Yasaman Bahri. Geometry of neural network loss surfaces via random
matrix theory. In Doina Precup and Yee Whye Teh (eds.), _Proceedings of the 34th International_
_Conference on Machine Learning_, volume 70 of _Proceedings of Machine Learning Research_, pp.
2798–2806. PMLR, 06–11 Aug 2017. URL [https://proceedings.mlr.press/v70/](https://proceedings.mlr.press/v70/pennington17a.html)
[pennington17a.html.](https://proceedings.mlr.press/v70/pennington17a.html)


Jeffrey Pennington and Pratik Worah. The spectrum of the fisher information matrix of a singlehidden-layer neural network. In _Neural_ _Information_ _Processing_ _Systems_, 2018. URL [https:](https://api.semanticscholar.org/CorpusID:53410991)
[//api.semanticscholar.org/CorpusID:53410991.](https://api.semanticscholar.org/CorpusID:53410991)


Jeffrey Pennington and Pratik Worah. Nonlinear random matrix theory for deep learning. _Jour-_
_nal of Statistical Mechanics:_ _Theory and Experiment_, 2019(12):124005, December 2019. ISSN
1742-5468. doi: 10.1088/1742-5468/ab3bc3. URL [http://dx.doi.org/10.1088/](http://dx.doi.org/10.1088/1742-5468/ab3bc3)
[1742-5468/ab3bc3.](http://dx.doi.org/10.1088/1742-5468/ab3bc3)


Marc Potters and Jean-Philippe Bouchaud. _A_ _First_ _Course_ _in_ _Random_ _Matrix_ _Theory:_ _for_ _Physi-_
_cists,_ _Engineers_ _and_ _Data_ _Scientists_ . Cambridge University Press, November 2020. ISBN
9781108488082. doi: 10.1017/9781108768900. URL [http://dx.doi.org/10.1017/](http://dx.doi.org/10.1017/9781108768900)
[9781108768900.](http://dx.doi.org/10.1017/9781108768900)


Alexandre Rame, Corentin Dancette, and Matthieu Cord. Fishr: Invariant gradient variances for outof-distribution generalization. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Ma-_
_chine_ _Learning_, volume 162 of _Proceedings_ _of_ _Machine_ _Learning_ _Research_, pp. 18347–18377.
PMLR, 17–23 Jul 2022. URL [https://proceedings.mlr.press/v162/rame22a.](https://proceedings.mlr.press/v162/rame22a.html)
[html.](https://proceedings.mlr.press/v162/rame22a.html)


Levent Sagun, Leon Bottou, and Yann LeCun. Eigenvalues of the hessian in deep learning: Singularity and beyond, 2017. [URL https://openreview.net/forum?id=B186cP9gx.](https://openreview.net/forum?id=B186cP9gx)


Adepu Ravi Sankar, Yash Khasbage, Rahul Vigneswaran, and Vineeth N Balasubramanian. A
deeper look at the hessian eigenspectrum of deep neural networks and its applications to regularization. _Proceedings_ _of_ _the_ _AAAI_ _Conference_ _on_ _Artificial_ _Intelligence_, 35(11):9481–9488,
May 2021. doi: 10.1609/aaai.v35i11.17142. [URL https://ojs.aaai.org/index.php/](https://ojs.aaai.org/index.php/AAAI/article/view/17142)
[AAAI/article/view/17142.](https://ojs.aaai.org/index.php/AAAI/article/view/17142)


Sidak Pal Singh, Aurelien Lucchi, Thomas Hofmann, and Bernhard Sch¨olkopf. Phenomenology of
double descent in finite-width neural networks. In _International Conference on Learning Repre-_
_sentations_, 2022. [URL https://openreview.net/forum?id=lTqGXfn9Tv.](https://openreview.net/forum?id=lTqGXfn9Tv)


Valentin Thomas, Fabian Pedregosa, Bart van Merri¨enboer, Pierre-Antoine Manzagol, Yoshua Bengio, and Nicolas Le Roux. On the interplay between noise and curvature and its effect on optimization and generalization. In Silvia Chiappa and Roberto Calandra (eds.), _Proceedings_ _of_
_the_ _Twenty_ _Third_ _International_ _Conference_ _on_ _Artificial_ _Intelligence_ _and_ _Statistics_, volume 108
of _Proceedings of Machine Learning Research_, pp. 3503–3513. PMLR, 26–28 Aug 2020. URL
[https://proceedings.mlr.press/v108/thomas20a.html.](https://proceedings.mlr.press/v108/thomas20a.html)


Nilesh Tripuraneni, Ben Adlam, and Jeffrey Pennington. Covariate shift in high-dimensional random
feature regression, 2021. [URL https://arxiv.org/abs/2111.08234.](https://arxiv.org/abs/2111.08234)


Shashanka Ubaru, Yousef Saad, and Abd-Krim Seghouane. Fast estimation of approximate matrix
ranks using spectral densities. _Neural Computation_, 29(5):1317–1351, 05 2017. ISSN 0899-7667.
doi: 10.1162/NECO ~~a~~ ~~0~~ 0951. [URL https://doi.org/10.1162/NECO_a_00951.](https://doi.org/10.1162/NECO_a_00951)


13


Tao Wu, Tie Luo, and Donald C. Wunsch II. Cr-sam: Curvature regularized sharpness-aware
minimization. _Proceedings of the AAAI Conference on Artificial Intelligence_, 38(6):6144–6152,
Mar. 2024. doi: 10.1609/aaai.v38i6.28431. [URL https://ojs.aaai.org/index.php/](https://ojs.aaai.org/index.php/AAAI/article/view/28431)
[AAAI/article/view/28431.](https://ojs.aaai.org/index.php/AAAI/article/view/28431)


Yikai Wu, Xingyu Zhu, Chenwei Wu, Annie Wang, and Rong Ge. Dissecting hessian: Understand[ing common structure of hessian in neural networks, 2022. URL https://arxiv.org/abs/](https://arxiv.org/abs/2010.04261)
[2010.04261.](https://arxiv.org/abs/2010.04261)


Zhewei Yao, Amir Gholami, Kurt Keutzer, and Michael W. Mahoney. Pyhessian: Neural networks through the lens of the hessian. _2020_ _IEEE_ _International_ _Conference_ _on_ _Big_ _Data_ _(Big_
_Data)_, pp. 581–590, 2019. URL [https://api.semanticscholar.org/CorpusID:](https://api.semanticscholar.org/CorpusID:209376531)
[209376531.](https://api.semanticscholar.org/CorpusID:209376531)


A STATEMENT ON LARGE LANGUAGE MODEL USE


Large language models were used to polish writing and in conjunction with other tools to discover
relevant published work.


B THEORETICAL FOUNDATIONS


B.1 EIGENVECTOR OVERLAP FUNCTION


To connect the finite- _d_ decomposition to random-matrix and free-probability tools, we now express
eigenvector overlaps in a kernelized trace form amenable to free-probabilistic methods. Let _X, Y_ be
symmetric _d × d_ matrices with eigendecompositions


_d_

- _λ_ _[Y]_ _j_ _[v][j][v]_ _j_ _[⊤][.]_

_j_ =1


_X_ =


_d_

- _λ_ _[X]_ _i_ _[u][i][u]_ _i_ _[⊤][,]_ _Y_ =

_i_ =1


For bounded functions _f, g_,


tr¯� _f_ ( _X_ ) _g_ ( _Y_ )� = [1]

_d_ [2]


_d_


_i_ =1


_d_

- _f_ ( _λ_ _[X]_ _i_ [)] _[ g]_ [(] _[λ]_ _j_ _[Y]_ [)] - _d_ ( _u_ _[⊤]_ _i_ _[v][j]_ [)][2][�] _._ (17)

_j_ =1


If _f_ and _g_ are sharply peaked around _λ_ 1 and _λ_ 2, the sum concentrates on overlaps between eigenvectors with eigenvalues near ( _λ_ 1 _, λ_ 2).


A convenient choice is the Poisson kernel


_σ_

_K_ ( _x_ ; _µ, σ_ ) := [1]

_π_ ( _x −_ _µ_ ) [2] + _σ_ [2] _[,]_


with center _µ_ and width _σ_ _>_ 0. We define the overlap function


_O_ ( _λ_ 1 _, λ_ 2) := lim
_σ_ 1 _,σ_ 2 _→_ 0


tr¯� _K_ ( _X_ ; _λ_ 1 _, σ_ 1) _K_ ( _Y_ ; _λ_ 2 _, σ_ 2)�

_._ (18)
tr¯� _K_ ( _X_ ; _λ_ 1 _, σ_ 1)� tr¯� _K_ ( _Y_ ; _λ_ 2 _, σ_ 2)�


The denominator normalizes the total weight in the sum (17) to one, so _O_ ( _λ_ 1 _, λ_ 2) is the weighted
average of the (scaled) squared overlaps _d_ ( _u_ _[⊤]_ _i_ _[v][j]_ [)][2] [over eigenpairs near][ (] _[λ]_ [1] _[, λ]_ [2][)][.] [Now (17) can be]
rewritten

��
tr¯ [ _f_ ( _X_ ) _g_ ( _Y_ )] = _f_ ( _λ_ 1) _g_ ( _λ_ 2) _O_ ( _λ_ 1 _, λ_ 2) _dµX_ ( _λ_ 1) _dµY_ ( _λ_ 2) _._ (19)


In fact, another way to define the overlap function is to write _µX,Y_ for the measure taking _f, g_ _�→_
tr¯ [ _f_ ( _X_ ) _g_ ( _Y_ )] and then defining _O_ ( _λ_ 1 _, λ_ 2) to be the function making (19) hold, ie. _O_ = _dµdµXX,Y⊗µY_ [.]


14


B.2 OVERLAP FLUCTUATION LAW


Here we prove the fluctuation law (6). Equation (5), which uses the quadratic surrogate losses
_L_ [quad] train _[,][ L]_ test [quad] [to compute the test loss increment, reads]


_d_ [1] _[z]_ test _[⊤]_ _[H]_ train _[−]_ [1] _[z]_ [ +] 2 [1]


∆ _L_ = _−_ [1]


2 [1] _d_ _[z][⊤][H]_ train _[−]_ [1] _[H]_ [test] _[H]_ train _[−]_ [1] _[z.]_ (20)


Noting that ∆ _w_ = _−H_ train _[−]_ [1] _[z]_ [, under the assumption that][ E][[∆] _[w]_ [] = 0][, one clearly has]


2 [1] [E][ ¯][tr][[] _[H]_ [test] _[H]_ train _[−]_ [1] _[zz][⊤][H]_ train _[−]_ [1] [] =] [1] 2


E[∆ _L_ ] = [1]


[1] 2 [tr][¯][[] _[H]_ [test] _[C]_ [train][]] _[.]_ (21)


All that’s left is to show that the last trace can be expressed in the integral form (6). Letting
( _λ_ [test] _i_ _, u_ [test] _i_ ) _,_ ( _λ_ [train] _j_ _, u_ [train] _j_ ) be the eigenvalues/eigenvectors of _H_ test _, C_ train,


_d_

- _λ_ [test] _i_ _λ_ [train] _j_ [ _d_ ( _u_ [test] _i_ _· u_ [train] _j_ ) [2] ] _._ (22)


_j_ =1


1 [1] 1
2 _d_ [tr][[] _[H]_ [test] _[C]_ [train][] =] 2 _d_ [2]


_d_


_i_ =1


Defining the overlap measure


_ν_ := [1]

_d_ [2]


equation (22) can be written


_d_


_i_ =1


_d_

- - �2

[ _d_ _u_ [test] _i_ _· u_ [train] _j_ ] _δ_ ( _λ_ test _i_ _,λ_ [train] _j_ ) _[,]_ (23)
_j_ =1


1 [1]
2 _d_ [tr][[] _[H]_ [test] _[C]_ [train][] =] 2


��
_λ_ 1 _λ_ 2 _ν_ ( _dλ_ 1 _, dλ_ 2) _._ (24)


_ν_ is absolutely continuous with respect to _µ_ test _⊗_ _µ_ train, and so we may define the Radon-Nikodym
derivative _O_ ( _λ_ 1 _, λ_ 2) = _dµ_ test _dν⊗µ_ train [(] _[λ]_ [1] _[, λ]_ [2][)][ so that]


1 [1]
2 _d_ [tr][[] _[H]_ [test] _[C]_ [train][] =] 2


On any atom ( _λ_ [test] _i_ _, λ_ [train] _j_ ),


��
_λ_ 1 _λ_ 2 _O_ ( _λ_ 1 _, λ_ 2) _µ_ test( _dλ_ 1) _µ_ train( _dλ_ 2) _._ (25)


_O_ ( _λ_ [test] _i_ _, λ_ [train] _j_ ) = _µ_ test( _ν{_ - _λ{_ [test] _i_ ( _λ_ [test] _i}_ ) _µ, λ_ train [train] _j_ ( _{_ ) _λ}_ [train] _j_ - _}_ ) [=] _[ d]_ - _u_ [test] _i_ _· u_ [train] _j_ �2 _,_ (26)


as desired.


B.2.1 SURROGATE-FREE FORMULATION


For completeness, we derive the fluctuation law without the use of quadratic surrogate losses by
making a minor modification to the train Hessian. As before, let _L_ train( _w, ϵ_ ) and _L_ test( _w_ ) denote
the train and test losses, assumed twice differentiable in _w_, and let _w_ 0 be the unperturbed minimizer
of _L_ train( _w,_ 0). For small perturbation _ϵ_, write ∆ _w_ = _w_ ( _ϵ_ ) _−_ _w_ 0 for the exact displacement. By the
fundamental theorem of calculus along the line segment _w_ 0 + _t_ ∆ _w_,


�� 1                     _∇wL_ train( _w_ 0 + ∆ _w, ϵ_ ) = _∇wL_ train( _w_ 0 _, ϵ_ ) + _∇_ [2] _w_ _[L]_ [train][(] _[w]_ [0] [+] _[ t]_ [∆] _[w, ϵ]_ [)] _[ dt]_ ∆ _w._

0


Define the _effective train Hessian_


              - 1
_H_ train [eff] [:=] _∇_ [2] _w_ _[L]_ [train][(] _[w]_ [0] [+] _[ t]_ [∆] _[w, ϵ]_ [)] _[ dt.]_

0


The perturbed optimality condition _∇wL_ train( _w_ 0 + ∆ _w_ ; _ϵ_ ) = 0 therefore yields the exact displacement equation
∆ _w_ = _−_ ( _H_ train [eff] [)] _[−]_ [1] _[ z,]_

where _z_ = _∇wL_ train( _w_ 0 _, ϵ_ ). Thus, ∆ _w_ is obtained by the same inverse-Hessian filtering law as in
the quadratic case, with _H_ train replaced by _H_ train [eff] [.]


15


For the test-loss increment, apply an ordinary Taylor expansion at _w_ 0:


_L_ test( _w_ 0 + ∆ _w_ ) = _L_ test( _w_ 0) + [1]


[1]

2 _d_ [∆] _[w][⊤][H]_ [test][∆] _[w]_ [ +] _[ O]_ [(] _[∥]_ [∆] _[w][∥]_ [3][)] _[,]_


[1] [1]

_d_ _[z]_ [test] _[ ·]_ [ ∆] _[w]_ [ +] 2


where _z_ test = _d ∇wL_ test( _w_ 0) and _H_ test = _d ∇_ [2] _w_ _[L]_ [test][(] _[w]_ [0][)] [are evaluated at] [the unperturbed point]
and are independent of _ϵ_ . Substituting the displacement equation, as in the surrogate case one
obtains:


_d_ [1] _[z]_ test _[⊤]_ [(] _[H]_ train [eff] [)] _[−]_ [1] _[z]_ [ +] 2 [1]


∆ _L_ = _−_ [1]


2 [1] _d_ _[z][⊤]_ [(] _[H]_ train [eff] [)] _[−]_ [1] _[H]_ [test][(] _[H]_ train [eff] [)] _[−]_ [1] _[z]_ [ +] _[ O]_ [(] _[∥][ϵ][∥]_ [3][)] _[.]_


Taking expectations over the perturbation, the quadratic term has the same form as in Theorem 1,
21 [tr][¯][[] _[H]_ [test] _[C]_ [train][]][, where] _[ C]_ [train] [=][ E] �( _H_ train [eff] [)] _[−]_ [1] _[zz][⊤]_ [(] _[H]_ train [eff] [)] _[−]_ [1][�], and so after replacing _H_ train with
_H_ train [eff] [, one obtains a formally identical overlap decomposition as in (6) in terms of test-(effective)]
train eigenvector overlaps.


B.2.2 NOISY GRADIENT DESCENT


Consider gradient descent with small additive noise at each iteration, and take its continuum limit.
In the local quadratic regime, such noisy gradient descent can be modeled by the linear stochastic
differential equation (SDE)
_dwt_ = _−H_ train _wtdt_ + Σ [1] _[/]_ [2] _dBt,_


where Σ is the noise covariance and _Bt_ is standard Brownian motion. Its stationary covariance
_C_ train satisfies the Lyapunov equation


_H_ train _C_ train + _C_ train _H_ train = Σ _._


This covariance is curvature-filtered—variance is suppressed along high-curvature directions and
amplified along shallow ones—similarly to the perturbation-induced _C_ train used in the main text.
Substituting this covariance into the trace formula yields the same overlap fluctuation law and the
same two-loss spectral-overlap decomposition, with no modifications to the framework required.


B.3 FREE TRANSFER LAW


Here we prove the following free transfer law for overlap functions:

**Theorem** (Theorem 2, Free transfer law) **.** _Let_ _B_ [ˆ] _combine_ _B_ _with_ _a_ _source_ _of_ _noise_ _X_ _that_ _is_ _free_
_from A, B._ _Then_

               _OA,_ ˆ _B_ [(] _[a,]_ [ˆ] _[b]_ [) =] _OA,B_ ( _a, b_ ) _OB,_ ˆ _B_ [(] _[b,]_ [ˆ] _[b]_ [)] _[ µ][B]_ [ (] _[db]_ [)] _[ .]_ (27)


Note that despite its simple and appealing form, this relationship does not hold for general triples of
matrices _A, B, C_ —it suffices to check on finite dimensional matrices with simple spectra. While the
coefficients of _C_ ’s eigenvectors in the eigenbases of _A, B_ do follow a change of basis law resembling
formula (27), recall that the overlap function encodes the squared coefficients rather than the raw
coefficients themselves. This relationship holds specifically because of the freeness relationship
we’ve assumed.


To prove (27), start by noting that all of the spectral and overlap information for two matrices _A, B_
is contained in the measure on R [2] corresponding to the functional


_µA,B_ : _f, g_ _�→_ _τ_ [ _f_ ( _A_ ) _g_ ( _B_ )] _._


For example, setting _g_ _→_ 1 and _f_ ( _x_ ) = _x_ _[m]_ gives access to all moments of _A_, and therefore to
its spectrum, and similarly for _B_, while the overlap function is precisely _OA,B_ = _dµdµAA,B⊗µB_ [,] [the]
Radon-Nikodym derivative of the joint measure with respect to the marginals.


For the remainder of this section, we will work in an abstract free probability space rather than with
concrete matrices. Let ( _M, τ_ ) be a _W_ _[∗]_ -probability space (in our application this corresponds to the
space of _d × d_ matrices with _τ_ = E tr). [¯] See Mingo & Speicher (2017) for details.


Let _A, B,_ _B, X_ [ˆ] _∈M_ be random variables and consider the problem of determining the overlap
function _OA,_ ˆ _B_ [, where] _[B]_ [ˆ] [:=] _[ F]_ [(] _[B, X]_ [)][, where] _[ X]_ [ is a source of noise that is free from] _[ A, B]_ [. As noted]


16


above, all of the overlap information for the three possible pairs of variables _A, B,_ _B_ [ˆ] is contained in
the joint measures
_µA,_ ˆ _B_ _[, µ][A,B][, µ]_ _B,B_ [ˆ] _[,]_

supported on (some subset of) R [2] . We are free now to treat these measures as ordinary probability
measures of two scalar random variables. Denote by _⟨·⟩_ these scalar expectations. We now make
use of the following two identities:


_⟨f_ ( _x_ ) _g_ ( _y_ ) _⟩_ ( _x,y_ ) _∼µX,Y_ = _τ_ [ _f_ ( _X_ ) _g_ ( _Y_ )] _,_ (28)


and that for each _g_ in a suitably broad class of functions (eg. at minimum all Poisson kernels), there
is another function _Lg_ depending linearly on _g_ encoding the “expectation of _g_ - _B_ ˆ� = _g_ ( _F_ ( _B, X_ ))
over _X_, conditioned on _B_ ”, ie. such that

_τ_             - _f_ ( _A, B_ ) _g_             - _B_ ˆ�� = _τ_ [ _f_ ( _A, B_ ) _Lg_ ( _B_ )] _,_ (29)


for all bounded borel _f_ . This is Proposition 1, proved below using operator-valued free probability.


Combining these, we have - _f_ ( _a_ ) _g_ �ˆ _b_ ��


= _τ_ [ _f_ ( _A_ ) _Lg_ ( _B_ )]. Writing the right hand
( _[a,]_ [ˆ] _[b]_ ) _[∼][µ]_ _A,B_


side as a scalar expectation and then conditioning on _b_,

       - _f_ ( _a_ ) _g_ �ˆ _b_ �� = �� _f_ ( _a_


     _a∼µA|B_ = _b_ _[L][g]_ [ (] _[b]_ [)]


= �� _f_ ( _a_ )    ( _[a,]_ [ˆ] _[b]_ ) _[∼][µ]_ _A,B_


_A|B_ = _b_ _b∼µB_

= _⟨Q_ ( _b_ ) _Lg_ ( _b_ ) _⟩b∼µB,_


where _Q_ ( _b_ ) := _⟨f_ ( _a_ ) _⟩a∼µA|B_ = _b_ . Applying (28) and (29) again,

     - _f_ ( _a_ ) _g_ �ˆ _b_ �� = _τ_     - _Q_ ( _B_ ) _g_     - _B_ ˆ��

( _[a,]_ [ˆ] _[b]_ ) _[∼][µ]_ _A,B_


= - _Q_ ( _b_ ) _g_ �ˆ _b_ ��

( _[b,]_ [ˆ] _[b]_ ) _[∼][µ]_ _B,B_ ˆ


= �� _f_ ( _a_ )�


_a∼µA|B_ = _b_ - _g_ ( [ˆ] _b_ )�ˆ _b∼µB_ ˆ _|B_ = _b_


_,_
_b∼µB_


which shows that _a,_ [ˆ] _b_ are independent conditioned on _b_ :


               _µA,_ ˆ _B_ [=] _µA|B_ = _bµB_ ˆ _|B_ = _b_ _[dµ][B]_ [ (] _[b]_ [)] _[ .]_


Applying, for example, classical _ϵ_ -gaussian smoothing to the measures of _A, B,_ _B_ [ˆ], we can assume
that _µX,Y_ _≪_ _µX_ _⊗_ _µY_ for any two of the three. Thus we are free to form the Radon-Nikodym
derivative _dµdµXX,Y⊗µY_ [,] [which] [corresponds] [to] [the] [gaussian-smoothed] [overlap] [function] _[O][X,Y]_ [ ;] _[ϵ]_ [ (] _[x, y]_ [)][.]
Since _OX,Y_ ; _ϵ_ ( _x, y_ ) _µX_ ( _dx_ ) = _µX|Y_ = _y_ ( _dx_ ), we have that for any bounded measurable function

 - - - �� - - [��] - - _ϕ_ _a,_ [ˆ] _b_ _dµA,_ ˆ _B_ [=] _ϕ_ _a,_ [ˆ] _b_ _µA|B_ = _b_ ( _da_ ) _µB_ ˆ _|B_ = _b_ _d_ [ˆ] _b_ _µB_ ( _db_ )


��        -        - [�]        -        -        -        - ��
= _ϕ_ _a,_ [ˆ] _b_ ( _OA,B_ ; _ϵ_ ( _a, b_ ) _µA_ ( _da_ )) _OB,_ ˆ _B_ ; _ϵ_ _b,_ [ˆ] _b_ _µB_ ˆ _d_ [ˆ] _b_ _µB_ ( _db_ )


��        -        - [��]        -        -        -        -        = _ϕ_ _a,_ [ˆ] _b_ _OA,B_ ; _ϵ_ ( _a, b_ ) _OB,_ ˆ _B_ ; _ϵ_ _b,_ [ˆ] _b_ _µB_ ( _db_ ) _µA_ ( _da_ ) _µB_ ˆ _d_ [ˆ] _b_ _,_


so the last quantity is exactly the Radon-Nikodym density _OA,_ ˆ _B_ ; _ϵ_ [:]


          -          -          -          -          _OA,_ ˆ _B_ ; _ϵ_ _a,_ [ˆ] _b_ = _OA,B_ ; _ϵ_ ( _a, b_ ) _OB,_ ˆ _B_ ; _ϵ_ _b,_ [ˆ] _b_ _µB_ ( _db_ ) _._


Taking the smoothing to 0, one obtains the transfer law for the overlap function. Note that one may
have to interpret the _OX,Y_ as distributions (eg involving _δ_ kernels) in case of degenerate overlap
between two of the matrices (eg. _A_ = _B_ ).


17


**Proposition** **1.** _Let_ _A, B_ _be_ _free_ _from_ _X,_ _and_ _let_ _B_ [ˆ] = _F_ ( _B, X_ ) _be_ _a_ _rational_ _function_ _of_ _B, X._
_Then there is a linear operator L on functions such that for arbitrary bounded borel H,_


_τ_             - _H_ ( _A, B_ ) _g_             - _B_ ˆ�� = _τ_ [ _H_ ( _A, B_ ) _L_ [ _g_ ] ( _B_ )] _._


_Proof._ Let **B** be a _k × k_ linearization of _B_ [ˆ] such that - **B** _[−]_ [1][�] [(] _[B, X]_ [))][.] [Decompose] **[ B]** [ into]

_k,_ 1 [=] _[ g]_ [ (] _[F]_
its _X_ -dependent and _X_ -independent parts:


**B** = _X_ **B** _X_ + **B** 0 _._


Now form the linearization matrix


   - _−_ **B** 0
**L** ( _M_ ) :=
_M_ **e** _[⊤]_ _k_ _−_ 1


_,_


so that


    - _−_ **B** _[−]_ [1] 0
**L** _[−]_ [1] ( _M_ ) :=
_−M_ **e** _[⊤]_ _k_ **[B]** _[−]_ [1] _−_ 1


_._


In particular, **L** _[−]_ _k,_ [1] 1 [(] _[M]_ [) =] _[ −][g]_ [ (] _[F]_ [(] _[B, X]_ [))][ and] **[ L]** _[−]_ _k_ +1 [1] _,_ 1 [(] _[M]_ [) =] _[ −][Mg]_ [ (] _[F]_ [(] _[B, X]_ [))][.] [So now]


- _g_ **L** ( _H_ ( _A,B_ )) (0)� _k_ +1 _,_ 1 [=] _[ τ]_ ��( _−_ **L** ( _H_ ( _A, B_ ))) _[−]_ [1][�]


_k_ +1 _,_ 1


= _τ_ [ _H_ ( _A, B_ ) _g_ ( _F_ ( _B, X_ ))] _._


As with **B**, decompose **L** into _X_ -dependent and _X_ -independent parts:


**L** ( _M_ ) = _X_ **L** _X_ + **L** 0 _._


Assuming _M_ is a function of _A, B_ only, these two parts are operator-free (ie. with amalgamation
over the space of ( _k_ + 1) _×_ ( _k_ + 1) matrices with complex entries). By the additive subordination
law,
_g_ **L** ( _M_ ) (0) = _g_ **L** 0             - _−RX_ **L** _X_             - _g_ **L** ( _M_ ) (0)�� _._


From the linearization, _g_ **L** ( _M_ ) (0) is block lower triangular, and _X_ **L** _X_ only has nonzero components
in the upper left block. This implies that _RX_ **L** _X_ - _g_ **L** ( _M_ ) (0)� is also confined to the upper left block,
and that this entry is simply _R−X_ **B** _X_ - _E_ **B** _[−]_ [1][�], where _E_ := _τ_ _⊗_ id is the expectation functional of
the operator-space. These imply


_τ_ [ _H_ ( _A, B_ ) _g_ ( _F_ ( _B, X_ ))] = - _g_ **L** ( _H_ ( _A,B_ )) (0)� _k_ +1 _,_ 1

=        - _g_ **L** 0        - _−RX_ **L** _X_        - _g_ **L** ( _H_ ( _A,B_ )) (0)���


_k_ +1 _,_ 1

- _−_ 1 [�]


_k_ +1 _,_ 1


= _τ_


�� **B** 0 _−R−X_ **B** _X_ - _E_ **B** _[−]_ [1][�] 0
_−H_ ( _A, B_ ) **e** _[⊤]_ _k_ 1


= _τ_ - _H_ ( _A, B_ ) - **B** 0 _−R−X_ **B** _X_ - _E_ **B** _[−]_ [1][��] _[−]_ _k,_ [1] 1


_._


The second factor can be regarded simply as a function of _B_ ( **B** 0 is the non- _X_ part of **B** and
_E_ **B** _[−]_ [1] : _B_ _�→E_ **B** _[−]_ [1] ( _B, X_ ) _∈_ _Mk_ (C). This proves that there is some operator _L_ [ _g_ ] as in the
statement of the proposition. _L_ must obviously be linear in _g_, completing the proof.


C TWO-LOSS GEOMETRY IN ANISOTROPIC RIDGE REGRESSION


In this appendix we derive equation (11), prove Theorem 3, and obtain formulas for the loss and
overlap functions that are used in figures 1-3. We consider ridge regression on multivariate gaussian
input data with train and test covariances Σtrain _,_ Σtest and with linear ground truth, _y_ ( _x_ ) = ~~_√_~~ 1 _d_ _[w][∗][·][x]_ [.]

We will assume for simplicity that _w∗_ _∼N_ (0 _, Id_ ). A noisy training set is generated by sampling
inputs as follows. The training set consists of


_xi_ _∼N_ (0 _,_ Σtrain) _,_ _yi_ = _y_ ( _xi_ ) + _ξi,_ _ξi_ _∼N_ �0 _, σξ_ [2]          - _,_


18


for _i_ = 1 _, . . ., m_ . We define the sampling density _α_ := _m/d_ . It will occasionally be convenient to
state formulas in terms of _α_ ’s reciprocal, _q_ := _d/m_ .


Using the framing of section 3.1.1, we will regard the noise _ξ_ as perturbing a noiseless training
objective. The train and test losses are formulated as follows:


_m_


_i_ =1


    _d_ _[w][ ·][ x][i]_ �2 + 2 _[λ]_


_L_ train( _w, ξ_ ) := [1] 2


- 1

_m_


- _yi −_ ~~_√_~~ 1


_[λ]_

2 _d_ _[∥][w][∥]_ [2]


_L_ test( _w_ ) := [1] 2


- E _x_ - _y_ ( _x_ ) _−_ ~~_√_~~ 1 �2 _._

_d_ _[w][ ·][ x]_


Note 1) we keep _L_ train’s dependence on the perturbation _ξ_ explicit, and 2) the noise is not included
in the test loss (and when it is, after averaging, it changes the loss only by an additive constant).
Finally, let us write _w_ ˆ ( _ξ_ ) := argmin _w L_ train ( _w, ξ_ ) for the learned weights, _y_ ˆ ( _x_ ) := ~~_√_~~ 1 _d_ _[w]_ [ˆ] _[ ·][ x]_ [ for]

the learned model, and _H_ train := _d ∇_ [2] _L_ train and _H_ test := _d ∇_ [2] _L_ test for the train and test Hessians;
these scalings are chosen to keep spectra _O_ (1).


We begin by deriving equation (11) by applying the fluctuation law (6). To do so, we first compute
_z, H_ train _, H_ test _,_ and _C_ train. Differentiating, we find


_d_ [1] - _m_ 1 _[X]_ _[⊤][X]_ [+] _[ λI]_ - _w −_ ~~_√_~~ 1


    (30)
_d_ _[Xw][∗]_ [+] _[ ξ]_


_∇wL_ train( _w, ξ_ ) = [1]


1 ~~_√_~~ 1
_d_ _m_ _[X]_ _[⊤]_ [�]


_H_ train := _d ∇_ [2] _w_ _[L]_ [train] [=] _m_ [1] _[X]_ _[⊤][X]_ [+] _[ λI.]_ (31)


Similarly,

_L_ test( _w_ ) = 21 _d_ [(] _[w][ −]_ _[w][∗]_ [) Σ][test][ (] _[w][ −]_ _[w][∗]_ [)] _[ .]_

_H_ test := _d ∇_ [2] _w_ _[L]_ [test] [= Σ][test] _[.]_


Next, _z_ := _d ∇wL_ train( _w_ 0 _, ξ_ ) is the scaled train gradient evaluated at the unperturbed optimum _w_ 0.
Since, by definition, _∇wL_ train( _w_ 0 _,_ 0) = 0, substituting into (30) gives


~~_√_~~
_z_ = _d ∇wL_ train( _w_ 0 _, ξ_ ) = - _m_ 1 _[X]_ _[⊤][X]_ [+] _[ λI]_ - _w_ 0 _−_


    (32)
_d_ _[Xw][∗]_ [+] _[ ξ]_


_d_ ~~_√_~~ 1
_m_ _[X]_ _[⊤]_ [�]


~~_√_~~
= _−_


_md_ _[X]_ _[⊤][ξ.]_ (33)


Finally,


_C_ train = E _ξ_ [( _H_ train _[−]_ [1] _[z]_ [)(] _[H]_ train _[−]_ [1] _[z]_ [)] _[⊤]_ []]


_d_
= _m_ [2] [E] _[ξ]_ [ [] _[H]_ train _[−]_ [1] _[X]_ _[⊤][ξξ][⊤][XH]_ train _[−]_ [1] []]

= _σξ_ [2] _md_ _[H]_ train _[−]_ [1] [(] _m_ [1] _[X]_ _[⊤][X]_ [)] _[H]_ train _[−]_ [1]

= _σξ_ [2] _[α][−]_ [1] [ˆΣ][train][(ˆΣ][train] [+] _[ λI]_ [)] _[−]_ [2] _[,]_

where Σ [ˆ] train := _m_ [1] _[X]_ _[⊤][X]_ [is the (uncentered) train covariance.] [Summarizing,]


~~_√_~~
_z_ = _−_ _md_ _[X]_ _[⊤][ξ]_ (34)

_H_ train = Σ [ˆ] train + _λI_ (35)
_H_ test = Σtest (36)

_C_ train = _σξ_ [2] _[α][−]_ [1] [ˆΣ][train][(ˆΣ][train] [+] _[ λI]_ [)] _[−]_ [2] _[.]_ (37)

Since E [ _ξ_ ] = 0, we have E[∆ _w_ ] = _−_ E[ _H_ train _[−]_ [1] _[z]_ [] = 0][.] [Plugging directly into (6), we find]


E[∆ _L_ ] = 21


��
_λ_ 1 _λ_ 2 _OH_ test _,C_ train( _λ_ 1 _, λ_ 2) _µH_ test( _dλ_ 1) _µC_ train( _dλ_ 2) _._ (38)


Since Σ [ˆ] train _, C_ train commute, they share eigenvectors and we are free to replace _OH_ test _,C_ train with
_OH_ test _,_ ˆΣtrain. Replacing the integral over _C_ train’s spectrum with one over Σ [ˆ] train, we find


E[∆ _L_ ] = 2 _σαξ_ [2]


�� _λ_ 2
_λ_ 1 (39)
( _λ_ 2 + _λ_ ) [2] _[O]_ [Σ][test] _[,]_ [ˆΣ][train][(] _[λ]_ [1] _[, λ]_ [2][)] _[ µ]_ [Σ][test][(] _[dλ]_ [1][)] _[ µ]_ [ˆΣ][train][(] _[dλ]_ [2][)] _[,]_


19


which is equation (11).


High-dimensional ridge regression has been studied extensively, so rather than rederiving published
formulas, as much as possible, we restrict attention to the novel focus of this article: overlap decompositions. We treat the label noise as a fluctuation of the training objective, and derive exact
asymptotic formulas for the needed train-test spectra and overlap functions that describing the effect
of the fluctuation on test error.


All formulas are obtainable from the general trace formula stated in the following two propositions,
which we prove in Appendix D.

**Proposition 2.** _The equation_

          -          - _t_          - _−_ 1
_r_ = 1 _−_ _q_ _,_ (40)
_z −_ _tr_ _[dµ]_ [Σ][train][ (] _[t]_ [)]


_has a unique solution r_ _satisfying r_ _∈_ H _[±]_ _for z_ _∈_ H _[∓]_ _and satisfying_ 0 _<_ _r_ _<_ 1 _for z_ _<_ 0 _._ _This_
_defines_ _a_ _holomorphic_ _function_ _r_ ( _z_ ) _on_ _all_ _of_ C _\_ R _[≥]_ [0] _that_ _is_ _obtainable_ _for_ _each_ _z_ _by_ _fixed_ _point_
_iteration_ _of_ _the_ _right_ _hand_ _side_ _of_ (40) _from_ _an_ _arbitrary_ _initial_ _point_ _r_ 0 _satisfying_ _r_ 0 _∈_ H _[±]_ _for_
_z_ _∈_ H _[∓]_ _and_ 0 _< r_ 0 _<_ 1 _for z_ _<_ 0 _._

**Proposition 3.** _Let_

                 _tf_ ( _z_ ) := _tr_ [¯] _f_ (Σtest _,_ Σtrain) ( _zI_ _−_ _A_ ) _[−]_ [1][�] _,_ (41)


_for bounded function f_ _and complex scalar z_ _∈_ C _\_ R _[≥]_ [0] _._ _As m, d →∞_ _with q_ = _d/m fixed,_


               _tf_ ( _z_ ) _→_ _tr_ [¯] _f_ (Σtest _,_ Σtrain) ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�] _,_


_where r_ ( _z_ ) _is the solution of the self-consistent equation_ (40) _._


Thus to perform the calculation we simply express all quantities in terms of traces of the form _tf_ ( _z_ ),
and then apply Propositions 2,3.


C.1 TRAIN-TEST HESSIAN OVERLAP FUNCTION


Since we will be interested primarily in the ridgeless limit _λ_ _→_ 0, and since the effect of nonzero
_λ_ is simply to shift the spectrum of _H_ train := _m_ 1 _[X]_ _[⊤][X]_ [+] _[ λI]_ [, we will omit] _[ λ]_ [ in the computation of]
_OH_ test _,H_ train ( _λ_ 1 _, λ_ 2), and will write _H_ train = _A_ = _m_ 1 _[X]_ _[⊤][X]_ [from now on unless explicitly stated]
otherwise. Thus, we are interested in the overlap function of the matrices

_H_ test = Σtest _,_ _H_ train = Σ [1] train _[/]_ [2] _m_ 1 _[Z]_ _[⊤][Z]_ [Σ] train [1] _[/]_ [2] _[.]_

The computation is simplified significantly by taking advantage of asymptotic freeness of _m_ 1 _[Z]_ _[⊤][Z]_
from Σtrain _,_ Σtest. By Theorem 2, we have asymptotically


_OH_ test _,H_ train ( _λ_ te _, λ_ tr) = _O_ Σtest _,H_ train ( _λ_ te _, λ_ tr)

             = _O_ Σtest _,_ Σtrain ( _λ_ te _, λ_ ) _O_ Σtrain _,H_ train ( _λ, λ_ tr) _dµ_ Σtrain ( _λ_ ) _._ (42)


In particular this shows, somewhat intuitively, that the overlap function of the train/test Hessians
will itself depend on the overlap function of the population covariance matrices. Eq. (42) shows
the dependence is quite simple: One simply composes the overlap kernels taking an _H_ train =
Σ [1] train _[/]_ [2] _m_ 1 _[Z]_ _[⊤][Z]_ [Σ] train [1] _[/]_ [2] [eigenspace to a][ Σ][train] [one, and taking a][ Σ][train] [eigenspace to a][ Σ][test] [one.] [The]
overlap of the train and test population covariances, _O_ Σtest _,_ Σtrain, is part of the input data of the
problem and is therefore known. As for the second factor, _O_ Σtrain _,H_ train, this is simply the overlap
function of the population and sample covariance matrices for an anisotropic gaussian sample. Formulas for this quantity are known (see, eg. Potters & Bouchaud (2020)). To keep the presentation
self-contained we quickly derive an expression using operator-valued free probability.


Following Appendix B.1, the train-test Hessian overlap function can be computed via


_O_ Σtrain _,H_ train ( _λ_ 1 _, λ_ 2) := lim
_σ→_ 0


tr¯� _K_ (Σtrain; _λ_ 1 _, σ_ 1) _K_ ( _H_ train; _λ_ 2 _, σ_ 2)�

_,_
tr¯� _K_ (Σtrain; _λ_ 1 _, σ_ 1)� tr¯� _K_ ( _H_ train; _λ_ 2 _, σ_ 2)�


20


where

_σ_

_K_ ( _x_ ; _µ, σ_ ) := [1]

_π_ ( _x −_ _µ_ ) [2] + _σ_ [2] _[,]_


is the Poisson kernel with center _µ_ and width _σ_ . Moving the first denominator trace into the numerator and canceling a factor of _π_ from top and bottom, we find that computing _OH_ test _,H_ train ( _λ_ 1 _, λ_ 2)
requires the numerator and denominator traces


_σ_ 2
_h_ (Σtrain)
( _A −_ _λ_ 2 _I_ ) [2] + _σ_ 2 [2] _[I]_


tr¯


_,_ tr¯


_σ_ 2
( _A −_ _λ_ 2 _I_ ) [2] + _σ_ 2 [2] _[I]_


_,_


where _h_ (Σtrain) := _K_ (Σtrain; _λ_ 1 _, σ_ 1) - tr¯� _K_ (Σtrain; _λ_ 1 _, σ_ 1)�.


Using the definition of _tf_ ( _z_ ) (41), and the resolvent form of the Poisson kernel,

_K_ ( _x_ ; _µ, σ_ ) = _−_ _π_ [1] [Im (] _[µ]_ [ +] _[ iσ][ −]_ _[x]_ [)] _[−]_ [1] _[,]_


these become

_−_ Im _th_ ( _λ_ 2 + _iσ_ 2) _,_ _−_ Im _t_ 1 ( _λ_ 2 + _iσ_ 2) _._
Proposition 3 implies


                _th_ ( _z_ ) _→_ tr [¯] _h_ (Σtrain) ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�] (43)


                _t_ 1 ( _z_ ) _→_ tr [¯] ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�] _,_ (44)


so


_O_ Σtrain _,H_ train ( _λ_ 1 _, λ_ 2) _→_ lim
_σ_ 1 _→_ 0 _z→_ [lim] _λ_ [+] 2 _[i]_


   Im tr [¯] _h_ (Σtrain) ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�]

_,_

     Im tr [¯] ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�]


where lim _z→λ_ +2 _i_ is shorthand for lim _σ→_ 0 with _z_ = _λ_ 2 + _iσ_ . Taking _σ_ 1 _→_ 0 sends _h_ (Σtrain) to a
delta function and collapses the trace in the numerator to the _λ_ 1 eigenspace of Σtrain, so


_O_ Σtrain _,H_ train ( _λ_ 1 _, λ_ 2) _→_ lim
_z→λ_ [+] 2 _[i]_


Im _z−r_ (1 _z_ ) _λ_ 1

- Im _z−r_ 1( _z_ ) _λ_ _[dµ]_ [Σ][train] [(] _[λ]_ [)] _[.]_


Composing with _O_ Σtest _,_ Σtrain yields the overlap function _O_ Σtest _,H_ train = _OH_ test _,H_ train.


C.2 OVERLAP DECOMPOSITION OF ∆ _L_


Trace integrals are written in terms of the spectra and overlaps of the matrices involved. To explicitly
determine the spectral density of _H_ train, note that it can be written in terms of the trace in equation
(44),
_ρH_ train ( _λ_ tr) = lim           - _K_ ( _H_ train; _λ_ tr _, σ_ )� _._
_σ→_ 0 [tr][¯]

Using the same approach as above, we have the following for the _σ_ -Poisson-smoothed spectral
density of _H_ train:

_ρH_ train; _σ_ ( _λ_ tr) = _−_ _π_ [1] [Im] _[ t]_ [1][ (] _[λ]_ [tr][ +] _[ iσ]_ [)]


_→−_ [1]

_π_


- 1
Im
_z −_ _r_ ( _z_ ) _λ_ _[dµ]_ [Σ][train][ (] _[λ]_ [)] _[ .]_


Collecting the results of the previous section and the fluctuation formula (11),


∆ _L_ = _[σ]_ [2]

2 _α_


where


��
_λ_ te
����
test curvature


_λ_ tr
( _λ_ tr + _λ_ ) [2]

- �� train variance


_OH_ train _,H_ test( _λ_ te _, λ_ tr)

- �� eigenspace overlap


_µH_ test( _dλ_ te) _µH_ train( _dλ_ tr) _,_ (45)


          _OH_ test _,H_ train ( _λ_ te _, λ_ tr) = _O_ Σtest _,_ Σtrain ( _λ_ te _, λ_ ) _O_ Σtrain _,H_ train ( _λ, λ_ tr) _dµ_ Σtrain ( _λ_ ) _,_ (46)


21


with


_O_ Σtrain _,H_ train ( _λ_ 1 _, λ_ 2) _→_ lim
_z→λ_ [+] 2 _[i]_


Im _z−r_ (1 _z_ ) _λ_ 1

- Im _z−r_ 1( _z_ ) _λ_ _[dµ]_ [Σ][train] [(] _[λ]_ [)] _[.]_


This provides a complete decomposition of the test loss fluctuation in terms of spectra and overlaps
of the train and test Hessian.


C.3 PROOF OF COVARIATE SHIFT THEOREM 3


This subsection, together with the proofs of Propositions 2, 3 found in Appendix D, proves Theorem
3.


Formulas (45) and (46) show the effect of covariate shift in train/test sets decomposes naturally in
terms of the overlap function _O_ Σtest _,_ Σtrain of the two population covariances. (Note that there are
two levels of overlap decomposition: the test loss increment is decomposed in terms of the train-test
Hessian overlap function (45), which in turn is decomposed in terms of the overlaps of Σtest _,_ Σtrain.)


We can equivalently start from explicit expressions for the fluctuation. Differentiating the loss and
solving for the optimal weights directly, one has

_L_ test ( ˆ _w_ ( _ξ_ )) = _−_ [1]      - _α_ 1 _[σ]_ _ξ_ [2] [(] _[t]_ [id] [(] _[−][λ]_ [)] _[ −]_ _[λt][′]_ id [(] _[−][λ]_ [)) +] _[ λ]_ [2] _[t]_ id _[′]_ [(] _[−][λ]_ [)]      - _,_ (47)

2

(equation (51) of Appendix C.4). Since ∆ _L_ = _L_ test _−L_ 0, and _L_ 0 is obtained by simply setting the
perturbation strength _σξ_ _→_ 0, we immediately find


_σξ_ [2] _d_
∆ _L_ = _−_ (48)
2 _α_ _dλ_ _[λt]_ [id][ (] _[−][λ]_ [)] _[ .]_

Adopting the notation _λ_ [˜] := _r_ ( _−λλ_ ) [, Proposition 3 yields]

_λt_ id ( _−λ_ ) _→−λ_ [˜] tr [¯] �Σtest           - _λI_ ˜ + Σtrain� _−_ 1 [�]


_d_         -         -         - _−_ 2 [�]
ΣtestΣtrain _λI_ ˜ + Σtrain _._
_dλ_ _[λt]_ [id][ (] _[−][λ]_ [)] _[ →−][λ]_ [˜] _[′]_ [tr][¯]


Substituting into (48),

∆ _L →_ _σξ_ [2] _λ_ ˜ _[′]_ tr¯ �ΣtestΣtrain       - _λI_ ˜ + Σtrain� _−_ 2 [�] _._
2 _α_


Writing the last trace as an integral over the spectral measures of Σtest _,_ Σtrain, this becomes

∆ _L →_ 2 _σαξ_ [2] _λ_ ˜ _[′]_  - _λ_ te  - _λ_ ˜ + _λ λ_ trtr�2 _[O]_ Σtest _,_ Σtrain [(] _[λ]_ te _[, λ]_ tr [)] _[ dµ]_ Σtest [(] _[λ]_ te [)] _[ dµ]_ Σtrain [(] _[λ]_ tr [)] _[,]_ (49)


which completes the proof of Theorem 3. Equation (49) parallels (45) but averages out the random
training inputs and label noise to express ∆ _L_ purely in terms of the known objects Σtrain _,_ Σtest. This
expression shows that label noise leads to large increases in test loss when a direction of large training variance (small eigenvalue _λ_ tr of Σtrain) and a direction of large test curvature (large eigenvalue
_λ_ te of Σtest) experience significant overlap (large _O_ Σtest _,_ Σtrain ( _λ_ te _, λ_ tr)).


C.4 EXPLICIT FORMULAS FOR TEST LOSS, FLUCTUATION


Here we derive explicit expressions for the full test loss and test loss fluctuation under general
covariate shift. Since these formulas and generalizations of them are already published, this section
is mostly for internal reference—especially for calculation of theoretical loss curves in Figs. 1 and
2.

Let _X_ have rows _x_ _[⊤]_ _i_ [and] _[ ξ]_ [ have components] _[ ξ][i]_ [.] _[L]_ [train] [can be written]


_L_ train( _w, ξ_ ) := 21 _m_ _[∥]_ ~~_√_~~ [1]


~~_√_~~ 1
_d_ _[Xw][∗]_ [+] _[ ξ][ −]_


_[λ]_
_d_ _[Xw][∥]_ [2][ +] 2


_[λ]_

2 _d_ _[w][⊤][w.]_


22


Differentiating, we find


_∇L_ train( _w, ξ_ ) = [1]


1 ~~_√_~~ 1
_d_ _m_ _[X]_ _[⊤]_ [�]


[1] ~~_√_~~ 1

_d_ _[H]_ [train] _[w][ −]_

_m_ [1] _[X]_ _[⊤][X]_ [+] _[ λI.]_


    _d_ _[Xw][∗]_ [+] _[ ξ]_


_H_ train := _d ∇_ [2] _L_ train = _m_ [1] _[X]_ _[⊤][X]_ [+] _[ λI.]_


Similarly,


_L_ test( _w_ ) = 21 _d_ [(] _[w][ −]_ _[w][∗]_ [) Σ][test][ (] _[w][ −]_ _[w][∗]_ [)] _[ .]_

_H_ test := _d ∇_ [2] _L_ test = Σtest _._


Solving 0 = _∇wL_ train yields


_w_ ˆ = _H_ train _[−]_ [1] - _m_ 1 _[X]_ _[⊤][X]_ - _w∗_ + ( _H_ train) _[−]_ [1] ~~_[√]_~~ _m_


_d_
_m_ _[X]_ _[⊤][ξ.]_


Substituting into _L_ test yields


_L_ test ( ˆ _w_ ( _ξ_ )) = [1] tr¯

2


- _qσξ_ [2] _m_ 1 _[X]_ _[⊤][X]_ [+] _[ λ]_ [2] _[I]_

Σtest  - _m_ 1 _[X]_ _[⊤][X]_ [+] _[ λI]_ �2


_._


Since ( _A_ + _λI_ ) _[−]_ [2] = _−_ _dλ_ _[d]_ [(] _[A]_ [ +] _[ λI]_ [)] _[−]_ [1] _[,]_ [we can write]


_t_ id ( _−λ_ ) (50)


_L_ test ( ˆ _w_ ( _ξ_ )) = _−_ [1]

2


- _d_ _[d]_
_qσξ_ [2] _dλ_ _[λ][ −]_ _[λ]_ [2] _dλ_


- _d_ _[d]_
_qσξ_ [2] _dλ_ _[λ][ −]_ _[λ]_ [2]


= _−_ [1] - _qσξ_ [2] [(] _[t]_ [id] [(] _[−][λ]_ [)] _[ −]_ _[λt][′]_ id [(] _[−][λ]_ [)) +] _[ λ]_ [2] _[t]_ id _[′]_ [(] _[−][λ]_ [)] - (51)

2


Proposition 3 implies


             _t_ id ( _z_ ) _→_ tr [¯] Σtest ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [1][�]


              -              _t_ _[′]_ id [(] _[z]_ [)] _[ →−]_ [tr][¯] Σtest ( _zI_ _−_ _r_ ( _z_ ) Σtrain) _[−]_ [2] ( _I_ _−_ _r_ _[′]_ ( _z_ ) Σtrain) _,_


which fully specifies _L_ test ( ˆ _w_ ( _ξ_ )). The fluctuation is easily gotten by setting _σξ_ [2] _[→]_ [0][ and subtracting]
from _L_ test ( ˆ _w_ ( _ξ_ )).


**Reduction to published formulas** Letting _λ_ [˜] := _r_ ( _−λλ_ ) [and substituting into (40), we obtain]


_λ_ = _λ_ [˜] _−_ [1]

_α_


- _λt_ ˜
_dµ_ Σtrain ( _t_ ) _,_ (52)
_λ_ ˜ + _t_


which is eq. (8) of Mel & Ganguli (2021) for the “effective regularization”.


The fluctuation in (50) is

1 _d_
∆ _L_ = _−qσξ_ [2] 2 _dλ_ _[λt]_ [id][ (] _[−][λ]_ [)] _[ .]_

Since


_λ_
_λt_ id ( _−λ_ ) _→_ tr¯
_r_ ( _−λ_ )


Σtrain


- _λ_ - _−_ 1 [�]

_−_ _[−]_ [Σ][train]
_r_ ( _−λ_ ) _[I]_


= _−λ_ [˜] tr [¯] �Σtrain - _λI_ ˜ + Σtrain� _−_ 1 [�] _,_


we get


��
Σtrain
_λI_ ˜ + Σtrain


1
∆ _L_ = _qσξ_ [2] 2 _λ_ ˜ _[′]_ tr¯


�2 [�]


The authors define _ρ_ 1 _f_ [:=] _[d]_ _dλ_ _[λ]_ [˜] [, so]


��
Σtrain
_λI_ ˜ + Σtrain


23


1

∆ _L_ = [1] 2 _[qσ]_ _ξ_ [2] _ρf_ tr¯


_._


�2 [�]

_,_


which matches the fluctuation term of their formula up to constant factors differing in the loss definitions. Next, the remaining term can be written


_L_ 0 = _L_ test _−_ ∆ _L_ = [1]

2


- _−λt_ id ( _−λ_ ) + _λ_ _[d]_ _._

_dλ_ _[λt]_ [id][ (] _[−][λ]_ [)]


Using (50) again,





tr
 [¯]


- _λ_ ˜Σtrain _λI_ ˜ + Σtrain




Σ [2] train
 - _λI_ ˜ + Σtrain�2








_L_ 0 = [1]

2


_L_ 0 = [1]


_−_ _λλ_ [˜] _[′]_ tr [¯]








Comparing to (52), the first term is _α_ - _λ_ ˜ _−_ _λ_ �, and differentiating gives


_α_
_λ_ ˜ _[′]_ =

_α −_ tr �� _λ_ ˜+ΣΣtraintrain


�2 [�] _[.]_


Substituting and simplifying yields













_L_ 0 = [1]

2


 _α_
 _[α][λ]_ [˜] _[ −]_ _[αλ]_ _α −_ tr �� _λ_ ˜+ΣΣtraintrain �2 [�]


  
= [1] _λ_ ˜ _[′]_ _λ_ ˜

2


_α −_ tr


��
Σtrain
_λ_ ˜ + Σtrain


�2 [��]


  
_−_ _αλ_ _._


Once again using equation (52), _α_ - _λ_ ˜ _−_ _λ_ - can be turned back into a trace:


 
[1] _λ_ ˜ _[′]_

2


_−_ _λ_ [˜] tr





 _,_


��
Σtrain
_λ_ ˜ + Σtrain


_L_ 0 = [1]


�2 [��]


tr¯


- _λ_ ˜Σtrain _λI_ ˜ + Σtrain


= [1] _λ_ ˜ _[′]_ tr¯

2




_λ_ ˜ [2] Σtrain
 - _λI_ ˜ + Σtrain�2


which is equivalent to their second term.


C.5 _k_ -LEVEL MODEL


At several points in the main text we refer to a _k_ -level input covariance,


_µ_ Σtrain =


_k_

- _piδsi._


_i_ =1


In this case the self-consistent equation for _r_ (40) becomes


_k_

- _pi_ _si_

_z −_ _sir_
_i_ =1


_,_


_−_ 1


_r_ =


1 _−_ _q_


which can be written as _p_ ( _r, z_ ) = 0 for some polynomial in _r, z_ . Similarly, the overlap function
simplifies to a sum over the distinct eigenvalues of Σtrain:


_O_ Σtrain _,H_ train ( _λ_ 1 _, λ_ 2) _→_ lim
_z→λ_ [+] 2 _[i]_


24


Im _z−r_ (1 _z_ ) _λ_ 1 _._

- _k_ 1
_i_ =1 _[p][i]_ [Im] _z−r_ ( _z_ ) _si_


C.5.1 SEPARATED SCALES LIMIT


We now assume the scales are widely separated: _si_ +1 _≪_ _si_ . We will also work with the ridgeless
formulas corresponding to _λ →_ 0 derived in Appendix C.6. For simplicity, assume _s_ 1 = 1. We will
obtain leading order formulas for _h_ ( _α_ ) as the ratio of successive scales is taken to 0. In Appendix
C.6, _h_ is defined and found to satisfy the following self-consistent equation (equation (55)):


               - _t_
1 =
_h_ + _αt_ _[dµ]_ [Σ][train][ (] _[t]_ [)]


The right hand side is a decreasing function of _h_ and a decreasing function of _α_, we have that _h_ is
a decreasing function of _α_ . Since the integral reduces to a sum over the _k_ eigenvalues, and since all
terms with _t_ _≪_ _h_ do not contribute at leading order in _si_ +1 _/si_, we assume that _h_ _≈_ _s_ 1 = 1 and
neglect all lower terms, giving

1
1 = _p_ 1
_h_ + _α_ _[,]_


so _h_ = _p_ 1 _−_ _α_ and ∆ _L_ is


∆ _L_ = _σξ_ [2] _α_ - _sO_ ( _s,_ 1) _dµ_ Σtest ( _s_ ) _,_
2 _p_ 1 _−_ _α_


the integral in the numerator can be written

         _sO_ ( _s,_ 1) _dµ_ Σtest ( _s_ ) = [1] tr¯ [Σtest _P_ Σtrain=1] _,_

_p_ 1

where _P_ Σtrain= _a_ is the projector onto Σtrain’s _a_ -eigenspace. In other words, the integral is simply
the normalized total overlap of Σtest onto the strong training covariance space, and is equal to 1 for
Σtrain = Σtest.

Now let us assume that _h_ is near the scale _s_ [2] _i_ [.] [The self-consistent equation becomes]

1 = _pi_ _h_ + _s αs_ [2] _i_ [2] _i_ + _α_ [1] _[µ]_ [Σ][train]             - _≫_ _s_ [2] _i_             - _,_

where _µ_ Σtrain - _≫_ _s_ [2] _i_ - is the total probability mass of all scales greater than _s_ [2] _i_ [, ie][ �] _j_ _[i][−]_ =1 [1] _[p][j]_ [.] [Solving]
yields


_αs_ [2] _i_


- _pi_ = _h,_
_α −_ _µ_ Σtrain ( _≫_ _s_ [2] _i_ [)] _[−]_ [1]


which is consistent with the assumption that _h_ _∼_ _s_ [2] _i_ [.] [Since] _[ h]_ _[≥]_ [0][, we only get a valid solution for]
_α ≥_ _µ_ Σtrain - _≫_ _s_ [2] _i_ �. Substituting back into the error expression yields


_σξ_ [2]
∆ _L_ =
2


_piα_ [2][ �] ( _h_ + _stαt_ ) [2] _[O]_ [ (] _[s, t]_ [)] _[ dµ]_ [Σ][train][ (] _[t]_ [)] _[ dµ]_ [Σ][test][ (] _[s]_ [)]

- _µ_ Σtrain - _≫_ _s_ [2] _i_ +1� _−_ _α_ - ( _α −_ _µ_ Σtrain ( _≫_ _s_ [2] _i_ [))] _[.]_


Evaluating the numerator generally requires a choice of Σtest’s behavior in the limit _si_ +1 _/si_ _→_ 0,
but note that the denominator has zeros at _α_ = _µ_ Σtrain - _≫_ _s_ [2] _i_ - _, µ_ Σtrain - _≫_ _s_ [2] _i_ +1�, and so the error
will generically become infinite whenever _α_ is equal to the cumulative mass of some number of top
scales. As a simple special case, letting Σtest = Σtrain, this reduces to


_σξ_ [2]
∆ _L_ =
2


Since under this assumption,


 - _α −_ _µ_ Σtrain - _≫_ _s_ [2] _i_ ��2 + _piµ_ Σtrain - _≫_ _s_ [2] _i_ 
- _µ_ Σtrain - _≫_ _s_ [2] _i_ +1� _−_ _α_ - ( _α −_ _µ_ Σtrain ( _≫_ _s_ [2] _i_ [))] _[.]_


_L_ 0 = [1]


   
[1] Σtrain ( _hI_ + _α_ Σtrain) _[−]_ [1][�] = [1]

2 _[h]_ [ ¯][tr] 2


2 _[h]_


- _α−µ_ Σtrain _pi_ ( _≫s_ [2] _i_ [)] _[−]_ [1] _µ_ Σtrain - _≫_ _s_ [2] _i_ - _< α ≤_ _µ_ Σtrain - _≫_ _s_ [2] _i_ +1� _,_


= [1]

2


_αs_ [2] _i_


when we take _si_ +1 _/si_ _→_ 0, _L_ 0 only contributes at the highest scale, so


where _σ_ + is the ReLU function.


_L_ 0 = [1]

2 _[σ]_ [+][ (] _[p]_ [1] _[ −]_ _[α]_ [)] _[,]_


25


C.6 RIDGELESS LIMIT


Here we simplify our formula for the test error in the ridgeless limit. From (50), we have

_d_

∆ _L_ = _−_ [1] 2 _[qσ]_ _ξ_ [2] _dλ_ _[λt]_ [id][ (] _[−][λ]_ [)]

_L_ 0 := _L_ test _−_ ∆ _L_ = _−_ [1] id [(] _[−][λ]_ [)] _[ .]_

2 _[λ]_ [2] _[t][′]_

It will also be helpful to consult _t_ id and _r_ ’s explicit expressions as matrix traces (equations (41) and
(59)):


_t_ id ( _z_ ) = tr [¯]


Σtest


- - _−_ 1 [�]
_zI_ _−_ [1]

_m_ _[X]_ _[⊤][X]_


_,_ (53)


                    -                    _r_ ( _z_ ) = 1 + _q_ tr [¯] ( _z −_ _A_ ) _[−]_ [1] _A_ _._ (54)


C.6.1 OVERSAMPLED REGIME


From (53), and since for _α_ := _m/d >_ 1, the limiting spectrum of _m_ [1] _[X]_ _[⊤][X]_ [is bounded away from][ 0][,]

_t_ id ( _−λ_ ) is analytic as _λ →_ 0 [+] . Thus in the oversampled regime

∆ _L →−_ [1] _ξ_ _[t]_ [id] [(0)] _[,]_ _L_ 0 _→_ 0 _._

2 _[qσ]_ [2]

From Propositions 2 and 3,

_t_ id ( _z_ ) _→−_ _r_ (0)1 tr¯ �ΣtestΣ _[−]_ train [1]             - _,_ _r_ (0) = 1 _−_ _q,_

so that
∆ _L, L_ test _→_ 2 [1] _[σ]_ _ξ_ [2] _α −_ 1 1 tr¯ �ΣtestΣ _[−]_ train [1]        - _._


C.6.2 UNDERSAMPLED REGIME


Now assume _α <_ 1. For _λ →_ 0, _t_ id and _r_ ’s explicit expressions in (53) and (59) suggest _t_ id ( _−λ_ ) =
_O_ - _λ_ _[−]_ [1][�] and _r_ ( _−λ_ ) = _O_ ( _λ_ ). For convenience we will rewrite our formulas in terms of _h_ ( _z_ ) :=

1 _q_ - _r_ ( _−z_ _z_ ) _[−]_ _[z]_ �. Substituting into the self-consistent equation for _r_ (40) and simplifying gives

                - _t_
_h_ ( _z_ ) = ( _qh_ ( _z_ ) + _z_ )
_qh_ ( _z_ ) + _z_ + _t_ _[dµ]_ [Σ][train][ (] _[t]_ [)] _[ .]_

Now differentiating and setting _z_ _→_ 0, we find

              - _t_
1 = _q_ (55)
_qh_ + _t_ _[dµ]_ [Σ][train][ (] _[t]_ [)]


_h_ _[′]_ =


  - [�] _t_ �2
_qh_ + _t_ _dµ_ Σtrain ( _t_ )

   - [�] _t_ �2 _,_ (56)
1 _−_ _q_ _qh_ + _t_ _dµ_ Σtrain ( _t_ )


where we’ve suppressed the argument of _h, h_ _[′]_ .


We now write the error expressions in terms of these

_d_

∆ _L_ = _−_ [1] 2 _[qσ]_ _ξ_ [2] _dλ_ _[λt]_ [id][ (] _[−][λ]_ [)]

_d_            
= [1] 2 _[qσ]_ _ξ_ [2] _dλ_ [(] _[qh]_ [ (] _[λ]_ [) +] _[ λ]_ [) ¯][tr] Σtest (( _qh_ ( _λ_ ) + _λ_ ) _I_ + Σtrain) _[−]_ [1][�]


tr¯ �Σtest ( _qhI_ Σ+Σtraintrain) [2]          
[1] 2 _[qσ]_ _ξ_ [2] [�] _t_ �2


_λ→_ 0
_−−−→_ [1]


   - [�] _t_ �2
1 _−_ _q_ _qh_ + _t_ _dµ_ Σtrain ( _t_ )


_σξ_ [2]
=
2


_q_ - ( _qhst_ + _t_ ) [2] _[O]_ [ (] _[s, t]_ [)] _[ dµ]_ [Σ][train][ (] _[t]_ [)] _[ dµ]_ [Σ][test][ (] _[s]_ [)]

     - [�] _t_ �2 _,_
1 _−_ _q_ _qh_ + _t_ _dµ_ Σtrain ( _t_ )


26


while

_L_ 0 = _−_ [1] id [(] _[−][λ]_ [)]

2 _[λ]_ [2] _[t][′]_


                
= [1] Σtest ( _qhI_ + Σtrain) _[−]_ [1][�]

2 _[qh]_ [ ¯][tr]

             - _s_

= [1]

2 _[qh]_ _qh_ + _t_ _[O]_ [ (] _[s, t]_ [)] _[ dµ]_ [Σ][train][ (] _[t]_ [)] _[ dµ]_ [Σ][test][ (] _[s]_ [)] _[ .]_


Finally, the total loss is just _L_ 0 + ∆ _L_ .


D CHARACTERIZATION OF _tf_ ( _z_ )


Here we derive an asymptotically exact expression for


                 _tF_ ( _z_ ) := tr [¯] _F_ (Σtest _,_ Σtrain) ( _zI_ _−_ _A_ ) _[−]_ [1][�] _._


Let us abbreviate _F_ := _F_ (Σtrain _,_ Σtest). First,


_A_ = _m_ [1] _[X]_ _[⊤][X]_ [= Σ][1] train _[/]_ [2]


- 1 Σ [1] train _[/]_ [2] _[,]_
_m_ _[Z]_ _[⊤][Z]_


where _Z_ has standard normal entries, so that


           


- 1 - _−_ 1 [�]
_zI_ _−_ Σtrain
_m_ _[Z]_ _[⊤][Z]_


_tF_ ( _z_ ) = tr [¯]


Σ [1] train _[/]_ [2] _[F]_ [Σ] _[−]_ train [1] _[/]_ [2]


_._ (57)


Now define _B_ := _I_ 5 _−_ _E_ 5 _,_ 2 _,_ where _E_ 5 _,_ 2 is a matrix whose (5 _,_ 2) entry is 1 and has all other entries
equal to 0, and let


0 0 0 0 0
_z_ 1 [Σ] train [1] _[/]_ [2] _[F]_ [Σ] train _[−]_ [1] _[/]_ [2] 0 _z_ 1 [Σ][train] 0 0
0 0 0 0 0
0 0 0 0 0
0 0 0 0 0


0 0 0 0 0
0 0 0 0 0
0 0 0 ~~_√_~~ 1 _Z_ _[⊤]_ 0
_m_
0 0 0 0 ~~_√_~~ 1 _Z_
_m_
0 0 0 0 0











Σ =


_Q_ =

























 _._




It is straightforward to verify that ( _B −_ (Σ + _Q_ )) _[−]_ [1] has as its (5 _,_ 1) block exactly the matrix in (57),
and so _tF_ ( _z_ ) = [ _g_ Σ+ _Q_ ( _B_ )]5 _,_ 1, where _g_ Σ+ _Q_ is the operator-valued Cauchy transform of Σ + _Q_ .


By rotational invariance, Σ _, Q_ are asymptotically operator free, meaning we can apply the operatorvalued additive subordination relation (see, eg. Mingo & Speicher (2017) Chapter 10), which yields
the self-consistent equation


_g_ Σ+ _Q_ ( _B_ ) = _g_ Σ ( _B −RQ_ ( _g_ Σ+ _Q_ ( _B_ ))) _._


The blocks of _Q_ are standard normal matrices, and so its _R_ -transform is given by


_RQ_ ( _M_ ) = _E_ [ _QMQ_ ] _,_


where _E_ is the operator-valued expectation that takes normalized traces of all square blocks. Due to
the large number of zeros in _Q_, only two entries of _RQ_ ( _M_ ) are nonzero:


[ _RQ_ ( _M_ )]3 _,_ 5 = _M_ 4 _,_ 4

[ _RQ_ ( _M_ )]4 _,_ 4 = _qM_ 5 _,_ 3 _._


27


On the other hand, by definition _g_ Σ ( _M_ ) = _E_ ( _M_ _−_ Σ) _[−]_ [1][�] . Substituting back into the subordina
tion relation and writing _g_ for _g_ Σ+ _Q_ ( _B_ ), we find





 _−_ 1 []








1 0 0 0 0

_−_ [1] [Ξ] 1 _−_ [1] [Σ][train] 0 0


_z_ [1] [Ξ] 1 _−_ _z_ [1]


_z_ [Ξ] 1 _−_ _z_ [Σ][train] 0 0

0 0 1 0 _−g_ 44
0 0 0 1 _−_ _qg_ 53 0
0 _−_ 1 0 0 1


_,_



_g_ = _E_









where to simplify notation we have written Ξ := Σ [1] train _[/]_ [2] _[F]_ [Σ] _[−]_ train [1] _[/]_ [2][.] [The] [entries] [of] [the] [right] [side] [are]
straightforward to compute using elementary row operations. Performing just enough such operations to determine the (5 _,_ 1) _,_ (4 _,_ 4) _,_ and (5 _,_ 3) entries, we obtain the closed system of equations


                 _g_ 53 = tr [¯] Σtrain ( _zI_ _−_ _g_ 44Σtrain) _[−]_ [1][�]


1
_g_ 44 =
1 _−_ _qg_ 53

                 _g_ 51 = tr [¯] _F_ ( _zI_ _−_ _g_ 44Σtrain) _[−]_ [1][�] _._


We can eliminate _g_ 53 entirely, giving our trace


                    _tf_ ( _z_ ) = _g_ 51 = tr [¯] _F_ ( _zI_ _−_ _r_ Σtrain) _[−]_ [1][�] _,_


in terms of the scalar _r_ := _g_ 44 that satisfies


           -           _r_ = 1 _−_ _q_ tr [¯] Σtrain ( _zI_ _−_ _r_ Σtrain) _[−]_ [1][��] _[−]_ [1]


A few remarks are in order. First, we note that we can rewrite this trace as an integral over the
spectrum of Σtrain:

          -          - _t_          - _−_ 1
_r_ = 1 _−_ _q_ _._ (58)
_z −_ _tr_ _[dµ]_ [Σ][train][ (] _[t]_ [)]


It is helpful to compare (58) to the explicit expressions for _g_ 44 _, g_ 53 from the linearization before
applying the subordination relation, which are


                     -                     _g_ 44 = 1 + _q_ tr [¯] ( _z −_ _A_ ) _[−]_ [1] _A_ (59)


                  -                   _g_ 53 = tr [¯] ( _zI_ _−_ _A_ ) _[−]_ [1] Σtrain _._ (60)


Thus _g_ 44 ( _z_ ) is analytic in _z_ everywhere outside the spectrum of _A_, and _g_ 44 (H _[±]_ ) _⊂_ H _[∓]_ and 0 _<_
_g_ 44 �R _[<]_ [0][�] _<_ 1 (the first inequality is gotten most easily by using _g_ 53 _<_ 0 and _g_ 44 = (1 _−_ _qg_ 53) _[−]_ [1] ).
In fact, these conditions along with the self-consistent equation (58) are enough to guarantee that
the solution is unique, holomorphic, and coincides with _g_ 44 throughout all of C _\_ R _[≥]_ [0] .

**Proposition** **4.** _For_ _z_ _∈_ C _\_ R _[≥]_ [0] _,_ _there_ _is_ _a_ _unique_ _solution_ _r_ ( _z_ ) _to_ (58) _satisfying_ _the_ _conditions_
_r_ (H _[±]_ ) _⊂_ H _[∓]_ and 0 _<_ _r_ �R _[<]_ [0][�] _<_ 1 _._ _r_ ( _z_ ) _depends holomorphically on z_ _and can be obtained by_
_iteration of the right hand side of_ (58) _from an arbitrary initial point in_ C _\_ R _[≥]_ [0] _._


_Proof._ Assume _z_ _∈_ H _[−]_ . Let _f_ ( _r, z_ ) be the map defined by the right hand side of (58):


           -           - _t_           - _−_ 1
_f_ ( _r, z_ ) := 1 _−_ _q_ _._
_z −_ _tr_ _[dµ]_ [Σ][train][ (] _[t]_ [)]


It is straightforward to check that _f_ ( _·, z_ ) : H [+] _→_ H [+] . Furthermore, no point on the boundary
of H [+] is a fixed point of _f_ ( _·, z_ ), since _f_ (R _, z_ ) _⊂_ H [+], and _f_ ( _∞, z_ ) = 1. The Denjoy-Wolff
theorem then guarantees that _f_ ( _·, z_ ) has a unique fixed point in H [+] - and that this point is obtained
by iteration of _f_ ( _·, z_ ) from an arbitrary initial point in H [+] . Thus (58) together with the condition
_r_ _∈_ H [+] uniquely defines a function _r_ ( _z_ ) for all _z_ _∈_ H _[−]_ .


28


Now fix _z_ 0 _∈_ H _[−]_ . Since _f_ ( _·, z_ 0) : H [+] _→_ H [+] and _f_ ( _·, z_ 0) is not a M¨obius transformation (it only
can be if Σtrain is a scalar matrix), the Schwarz lemma implies �� _∂r∂_ _[f]_ [ (] _[r]_ [ (] _[z]_ [0][)] _[, z]_ [0][)] �� _<_ 1, which means
_∂_ _∂_
= _>_ 0 _,_
���� _∂r_ [(] _[f]_ [ (] _[r, z]_ [0][)] _[ −]_ _[r]_ [)] ���� ���� _∂r_ _[f]_ [ (] _[r, z]_ [0][)] _[ −]_ [1] ����

and so the implicit function theorem implies there is a holomorphic function solving (58) on some
neighborhood of _z_ 0 that coincides with _r_ ( _z_ 0) at _z_ 0. Since _r_ ( _z_ 0) _∈_ H [+], this function must also stay
in H [+] in some (possibly smaller) neighborhood of _z_ 0, and by uniqueness of solutions to (58), this
implies that it coincides with _r_ ( _z_ ) on this neighborhood. Thus _r_ ( _z_ ) : H _[−]_ _→_ H [+] is holomorphic at
each point of H _[−]_ . An identical argument proves the proposition for _z_ _∈_ H [+] .


Now suppose _z_ _<_ 0. Conjugating the right hand side of (58) by the map _x_ _�→_ 1 _/_ (1 _−_ _qx_ ) gives a
self-consistent equation satisfied by _g_ 53:

              - _t_
_y_ = _z −_ 1 _−_ 1 _qy_ _[tdµ]_ [Σ][train][ (] _[t]_ [)] _[ .]_ (61)


The condition 0 _<_ _r_ ( _z_ ) _<_ 1 implies _g_ 53 _<_ 0. Now letting _h_ ( _y, z_ ) be the right hand side of (61),
_h_ ( _y, z_ ) _−_ _y_ is convex in _y_ and satisfies _h_ (0 _, z_ ) _−_ 0 _<_ 0 and _h_ ( _−∞, z_ ) _−_ ( _−∞_ ) = _∞_, so there is a
unique solution to (61) with _y_ _<_ 0, and thus a unique solution to (58) with 0 _< r_ ( _z_ ) _<_ 1.


Since _∂y∂_ _[h]_ [ (] _[y, z]_ [)] _[ >]_ [ 0][, and differentiating] _[ h]_ [ at the fixed point gives]

_∂_ _[∂]_        - _t_
_∂y_ _[h]_ [ (] _[y, z]_ [) =] _∂y_ [(1] _[ −]_ _[qy]_ [)] _z_ (1 _−_ _qy_ ) _−_ _t_ _[dµ]_ [Σ][train][ (] _[t]_ [)]

1                 - (1 _−_ _qy_ ) _qt_
= 1 _−_ _[dµ]_ [Σ][train][ (] _[t]_ [)]
1 _−_ _qy_ [+] _[ z]_ ( _z_ (1 _−_ _qy_ ) _−_ _t_ ) [2]


_<_ 1 _,_
_y_ ( _z_ ) is an attracting fixed point of _h_ ( _·, z_ ). Since _h_ ( _·, z_ ) is a conjugate of _f_ ( _·, z_ ), the unique solution
of (58) satisfying 0 _<_ _r_ ( _z_ ) _<_ 1 is an attracting fixed point of _f_ ( _·, z_ ). This implies that there is
a neighborhood of _z_ that extends into the upper half plane whose iterates converge to _r_ ( _z_ ). But
since _z_ _<_ 0, _f_ (H [+] _, z_ ) _⊂_ H [+], and so the Denjoy-Wolff theorem implies that all iterates of _f_ ( _·, z_ )
initialized in H [+] converge to the same point, which therefore must be _r_ ( _z_ ).

Finally, _dyd_ [(] _[h]_ [ (] _[y]_ [)] _[ −]_ _[y]_ [) =] _[ h][′]_ [ (] _[y]_ [)] _[ −]_ [1] _[ <]_ [ 0][ implies that] _[ y]_ [ (] _[z]_ [)][ extends holomorphically to a solution of]
(61) in an entire neighborhood of _z_ . Since _h_ ( _y_ ( _w_ ) _, w_ ) _−_ _y_ ( _w_ ) = 0 for all _w_ in this neighborhood,
at the solution point,


- _t_
_t_ 2 _[dµ]_ Σtrain [(] _[t]_ [)]
( _qy−_ 1 [+] _[z]_ [)]

_∂_
_∂y_ [(] _[h]_ [ (] _[y]_ [ (] _[z]_ [)] _[, z]_ [)] _[ −]_ _[y]_ [ (] _[z]_ [))] _[<]_ [ 0] _[.]_


_y_ _[′]_ ( _z_ ) = _−_


_∂_
_∂z_ _[h]_ [ (] _[y]_ [ (] _[z]_ [)] _[, z]_ [)]

_∂_
_∂y_ [(] _[h]_ [ (] _[y]_ [ (] _[z]_ [)] _[, z]_ [)] _[ −]_ _[y]_ [ (] _[z]_ [))] [=]


A negative derivative implies that for sufficiently small neighborhood _U_ of _z_, _y_ ( _U_ _∩_ H _[−]_ ) _⊂_ H [+] .


Mapping back to _r_ ( _z_ ) := 1 _/_ (1 _−_ _qy_ ( _z_ )) yields a holomorphic function satisfying (58) in a neighborhood _U_ of _z_ _<_ 0 such that for _r_ ( _U_ _∩_ H _[−]_ ) _⊂_ H [+] . By uniqueness of solutions in the upper half
plane, _r_ must coincide with the function defined earlier on _U ∩_ H _[−]_ . Thus _r_ extends holomorphically
to the negative real axis.


The subordination relation implies _g_ 44 ( _z_ ) = _r_ ( _z_ ) in a neighborhood of _∞_, but both functions
extend holomorphically to all of C _\_ R _[≥]_ [0], implying they are equal throughout. This completes the
proof of Propositions 2 and 3.


E LOCAL GEOMETRY OF MLPS


E.1 GRADIENT DESCENT DYNAMICS


The initial gradient at _w_ 0 is _z_ := _d∇L_ train ( _w_ 0 _, ϵ_ ), and the Hessian is _H_ train := _d∇_ [2] _L_ train, so the
local approximation for the training loss is


_L_ train ( _w_ ) _≈_ [1]


[1] [1]

_d_ _[z][⊤]_ [(] _[w][ −]_ _[w]_ [0][) +] 2


2 _d_ [(] _[w][ −]_ _[w]_ [0][)] _[⊤]_ _[H]_ [train][ (] _[w][ −]_ _[w]_ [0][)] _[,]_


29


Figure 6: Eigenvector overlap function for one MLP simulation in the context of 3.3. A dot is plotted
for every pair of train and test eigenvalues, with dot size and opacity representing squared overlap of
the corresponding eigenvectors. Note the very strong train-test alignment indicated by the restriction
of almost all overlap to the diagonal.


Figure 7: Learning dynamics predicted by local two-loss geometry. Left panel: Test loss trajectories
in response to different label noise realizations (individual trajectories shown as thin green lines;
average shown as thick green line). Noise amplitude corresponds to red dot in Fig. 4(b). To reduce
clutter, only theory average is shown (black line; see (62)). Right panel: measured vs predicted
relative test loss increment at two times, corresponding to the vertical blue and gold lines in the left
panel. Points corresponding to the same trajectory are connected by gray lines. Horizontal blue and
gold lines show means.


where we’ve discarded additive constants. The gradient is then


so gradient descent does


_∇L_ train ( _w_ ) _≈_ [1]

_d_ [(] _[z][ −]_ _[H]_ [train] _[w]_ [0][ +] _[ H]_ [train] _[w]_ [)] _[,]_


_w_ _→_ _w −_ _η_ ( _z −_ _H_ train _w_ 0 + _H_ train _w_ )
= ( _I_ _−_ _ηH_ train) _w −_ _η_ ( _z −_ _H_ train _w_ 0) _._


30


Thus,


and so the test error satisfies


_[−]_ _[ηH]_ [train][)] _[t][ −]_ _[I]_
∆ _wt_ := _wt −_ _w_ 0 = [(] _[I]_ _z,_

_H_ train


_[−]_ _[ηH]_ [train][)] _[t][ −]_ _[I]_
∆ _wt_ := _wt −_ _w_ 0 = [(] _[I]_


(∆ _Ltest_ ) _t_ = [1]


_t_ _[H]_ [test][∆] _[w][t]_ (62)
2 _d_ [∆] _[w][⊤]_


[1] [1]

_d_ _[g][⊤]_ [∆] _[w][t]_ [ +] 2


[1] _[−]_ _[ηH]_ [train][)] _[t][ −]_ _[I]_

_d_ _[g][⊤]_ [(] _[I]_ _H_ train


[1] [(] _[I]_ _[−]_ _[ηH]_ [train][)] _[t][ −]_ _[I]_

2 _d_ _[z]_ _H_ train


_z,_
_H_ train


= [1]


[train][)] _[I]_

_z_ + [1]
_H_ train 2


[train][)] _[t][ −]_ _[I]_ ( _I_ _−_ _ηH_ train) _[t]_ _−_ _I_

_H_ test
_H_ train _H_ train


(63)


with _g_ := _d∇L_ test ( _w_ 0) and _H_ test := _d∇_ [2] _L_ test ( _w_ 0).


To compute the GD trajectory for a large network, we precompute


_z_ = grad ( _L_ train) ( _w_ 0)
_H_ train _w_ 0 = hvp ( _L_ train _, w_ 0) _,_


set _v_ 0 := _z −_ _H_ train _w_ 0, and simply iterate


_w_ _→_ _w −_ _η_ ( _v_ 0 + hvp ( _L_ train _, w_ )) _,_


where grad _,_ hvp compute function gradients and Hessian-vector products.


F EFFICIENT CALCULATION OF EIGENVECTOR OVERLAPS FOR LARGE

MATRICES


Here we describe the Overlap-KPM algorithm, which estimates the unnormalized or total eigenvector overlaps for two matrices. That is, for self-adjoint _A, B_ _∈_ R _[d][×][d]_, it estimates the measure


_µA,B_ := [1]

_d_ [2]


_d_


_δ_ ( _λAi_ _[,λ]_ _j_ _[B]_ [)]
_i,j_ =1


- _d_ - _vi_ _[A]_ _[·][ v]_ _j_ _[B]_ �2 [�] _,_


which accumulates all overlap of _λ_ 1 _A_ -eigenspaces with all _λ_ 2 _B_ -eigenspaces. To obtain the normalized overlap function treated in the main text (eg. equation (18)), one simply divides by the
spectral densities of _A, B_ at _λ_ 1 _, λ_ 2. The problem of estimating spectral densities for large matrices
has already received significant attention (see Papyan (2019) in machine learning context), so we
assume that one can practically transform back and forth between normalized and unnormalized
overlap functions.


F.1 RANK DEFLATION AND NORMALIZATION


As a preprocessing step, we remove the outlier eigenspaces from each matrix obtained via subspace
iteration (cf Fig. 5), and then normalize the spectra to the interval [ _−_ 1 _,_ 1].


F.2 RANK DEFLATION VIA SUBSPACE ITERATION


The overlap plots of Fig. 5 were generated via Subspace Iteration (SI). SI is a well-known method
that generalizes power iteration, so we review it only briefly here. See Papyan (2019) for an explicit
implementation in a machine learning context.


Let _M_ be a self-adjoint operator with simple spectrum _λ_ 1 _, . . ., λd_, and take _V_ _∈_ R _[d][×][k]_ with standard
normal entries and then orthonormalize the columns. SI iterates


_V_ _→_ _MV_
_V_ _→_ _V Q,_


where _Q_ is the Gram-Schmidt orthonormalizing upper triangular matrix of _V_ . Informally, each
application of _M_ amplifies each _i_ _[th]_ eigenspace coefficient of the columns of _V_ by _λi_, which generically leads to exponentially greater weight on the leading eigenspaces. The orthonormalization _Q_


31


prevents all eigenvectors from collapsing onto the same leading eigenvector. Since they are forced
to span an _k_ -dimensional space, they must converge to the top _k_ eigenvectors of _M_ . Overlaps can
then be calculated directly by computing pairwise dot products of columns of _V_ .


After convergence, outlier eigenspaces are removed from the matrices by replacing each matrix
vector product _v_ _�→_ _M_ ( _v_ ) with

_v_ _�→_ _Mdef_ ( _v_ ) = _M_ ( _v_ ) _−_ _V V_ _[⊤]_ _v._


F.2.1 SPECTRUM NORMALIZATION


After removing the outlier eigenvalues, one may obtain bounds for the remaining spectrum via
standard approaches (eg. the Lanczos algorithm; cf Papyan (2019)). Letting _λ_ min _, λ_ max denote the
minimum and maximum eigenvalue (in practice, with a small amount of padding added), we then
normalize the matrices to the interval [ _−_ 1 _,_ 1] by replacing _v_ _�→_ _Mdef_ ( _v_ ) with


2                  - _λ_ max + _λ_ min
_v_ _�→_ _Mnorm_ ( _v_ ) = _Mdef_ ( _v_ ) _−_
_λ_ max _−_ _λ_ min _λ_ max _−_ _λ_ min


F.3 OVERLAP-KPM


_v._


We now assume the previous preprocessing steps have been performed and in particular that _A, B_ ’s
spectra lie inside [ _−_ 1 _,_ 1].


First note that for kernel function _G_ one can write the kernel-smoothed overlaps exactly as a trace:


tr¯ [ _G_ ( _A −_ _λ_ 1; _σ_ ) _G_ ( _B −_ _λ_ 2; _σ_ )] = [1]

_d_ [2]


_d_

- _G_ ( _λA,i −_ _λ_ 1; _σ_ ) _G_ ( _λB,j_ _−_ _λ_ 2; _σ_ ) - _d_ ( _vA,i · vB,j_ ) [2][�] _._


_i,j_ =1

(64)


Thus the goal will be to compute such traces for each ( _λ_ 1 _, λ_ 2) for some sufficiently small fixed
kernel width _σ._ Computing such traces directly is prohibitively expensive for very large matrices,
and so a standard approach is to use Hutchinson trace estimation, ie. to average _v_ _[⊤]_ _Mv_ over several
random samples of, say, standard normal _v_, since

E _v_         - _v_ _[⊤]_ _Mv_         - = tr         - _M_ E _v_         - _vv_ _[⊤]_ [��] = tr [ _M_ ] _._


Informal experiments suggested better stability for estimation of PSD traces, so we replace the trace
on the left side of (64) with


          -           tr¯ _G_ [1] _[/]_ [2] ( _A −_ _λ_ 1; _σ_ ) _G_ ( _B −_ _λ_ 2; _σ_ ) _G_ [1] _[/]_ [2] ( _A −_ _λ_ 1; _σ_ ) _._


Now applying the Hutchinson trick, we sample probes _v_ 1 _, . . ., vP_ and approximate


tr¯ [ _G_ ( _A −_ _λ_ 1; _σ_ ) _G_ ( _B −_ _λ_ 2; _σ_ )] _≈_ [1]

_P_


_P_

- _vµ_ _[⊤][G]_ [1] _[/]_ [2][ (] _[A][ −]_ _[λ]_ [1][;] _[ σ]_ [)] _[ G]_ [ (] _[B][ −]_ _[λ]_ [2][;] _[ σ]_ [)] _[ G]_ [1] _[/]_ [2][ (] _[A][ −]_ _[λ]_ [1][;] _[ σ]_ [)] _[ v][µ]_

_µ_ =1


= [1]

_P_


_P_


_µ_ =1


2
_G_ 1 _/_ 2 ( _B −_ _λ_ 2; _σ_ ) _G_ 1 _/_ 2 ( _A −_ _λ_ 1; _σ_ ) _vµ_ _._
��� ���


To compute the summand, we generalize a standard approach known as the kernel polynomial
method. Practically speaking, this entails approximating the kernel functions _G_ [1] _[/]_ [2] ( _x −_ _λ_ ; _σ_ ) using
Chebyshev polynomials _Tj_ ( _x_ ), which can be computed efficiently using _T_ 0 ( _x_ ) = 1 _, T_ 1 ( _x_ ) = _x_,
and the recurrence
_Tj_ ( _x_ ) = 2 _xTj−_ 1 ( _x_ ) _−_ _Tj−_ 2 ( _x_ ) _,_ _j_ _≥_ 2 _._
Letting _α, β_ be the Chebyshev coefficients of the kernel functions,


_G_ [1] _[/]_ [2] ( _x −_ _λ_ 1; _σ_ ) =


_G_ [1] _[/]_ [2] ( _x −_ _λ_ 2; _σ_ ) =


32


_∞_

- _αiTi_ ( _x_ )


_i_ =0

_∞_

- _βjTj_ ( _x_ ) _,_


_j_ =0


we truncate to degree _K_ and write


2
_cµ_ := _G_ 1 _/_ 2 ( _B −_ _λ_ 2; _σ_ ) _G_ 1 _/_ 2 ( _A −_ _λ_ 1; _σ_ ) _vµ_ (65)
��� ���


������


(66)


_≈_


=


_K_


_αiβjTj_ ( _B_ ) _Ti_ ( _A_ ) _vµ_

_i,j_ =0


������2


_K_

 - _αiβjαkβℓvµ_ _[⊤][T][i]_ [(] _[A]_ [)] _[ T][j]_ [(] _[B]_ [)] _[ T][ℓ]_ [(] _[B]_ [)] _[ T][k]_ [(] _[A]_ [)] _[ v][µ]_ (67)

_i,j,k,ℓ_ =0


_K_

 - _αiβjαkβℓMi,j,k,ℓ,µ._ (68)


_i,j,k,ℓ_ =0


_K_


=:


_K_


Thus for _P_ probes and order- _K_ Chebyshev truncation, by appropriate choice of the coefficients _α, β_,
one can approximate a general function from the _P_ ( _K_ + 1) [4] dot products


_Mi,j,k,ℓ,µ_ = _vµ_ _[⊤][T][i]_ [(] _[A]_ [)] _[ T][j]_ [(] _[B]_ [)] _[ T][ℓ]_ [(] _[B]_ [)] _[ T][k]_ [(] _[A]_ [)] _[ v][µ][.]_


This can be improved somewhat using the Chebyshev product identity

_Tj_ ( _x_ ) _Tℓ_ ( _x_ ) = [1]             - _Tj_ + _ℓ_ ( _x_ ) + _T|j−ℓ|_ ( _x_ )� _,_ (69)

2


so that


_Mi,j,k,ℓ,µ_ = [1] - _vµ_ _[⊤][T][i]_ [(] _[A]_ [)] _[ T][j]_ [+] _[ℓ]_ [(] _[B]_ [)] _[ T][k]_ [(] _[A]_ [)] _[ v][µ]_ [+] _[ v]_ _µ_ _[⊤][T][i]_ [(] _[A]_ [)] _[ T]_ _|j−ℓ|_ [(] _[B]_ [)] _[ T][k]_ [(] _[A]_ [)] _[ v][µ]_ - _,_

2


and so all needed dot products can be obtained from the _P_ ( _K_ + 1) [2] (2 _K_ + 1) _∼_ 2 _PK_ [3] dot products

_M_ ˜ _i,j,k,µ_ := _vµ_ _[⊤][T][i]_ [(] _[A]_ [)] _[ T][j]_ [(] _[B]_ [)] _[ T][k]_ [(] _[A]_ [)] _[ v][µ][,]_ 0 _≤_ _i, k_ _≤_ _K,_ 0 _≤_ _j_ _≤_ 2 _K,_ 1 _≤_ _µ ≤_ _P._


Algorithm 1 efficiently generates all such probe moments with _∼_ _PK_ [2] matrix vector products.
Algorithm 1 actually stores all 2 _PK_ [2] vectors _Tj_ ( _B_ ) _Tk_ ( _A_ ) _vµ_, but in practice, our implementation
is significantly more memory efficient. We store all _Tk_ ( _A_ ) _vµ_, but as _B_ ’s are added, one only needs
to store the current and previous power of _B_ . This amounts to _∼_ _K_ vectors in memory at once.


Once the _M_ [˜] _i,j,k,µ_ are known, equation (68) is used to estimate the trace for each value of _λ_ 1 _, λ_ 2,
yielding an approximation to the unnormalized overlap function of _A, B_ .


Often in machine learning contexts, one or both of _A, B_ has spectrum that is highly peaked around
a particular value. For the trace in (64) to accurately reflect the overlaps at _λ_ 1 _, λ_ 2, the kernels—
more precisely, their finite _K_ Chebyshev series—must decay sufficiently quickly away from _λ_ 1 _, λ_ 2
to prevent the spectral spikes from overwhelming the overlap sum. Practically speaking, this can be
diagnosed by 1) forming an estimate of _A, B_ ’s spectral density, eg. using the Lanczos algorithm (see
Papyan (2019) for implementation in ML context), 2) forming truncated Chebyshev series for the
kernels, and 3) comparing kernel decay to spike height. Insufficient decay usually requires either
decreased kernel width _σ_, or increased Chebyshev degree _K_ so that polynomial approximations
accurately approximate the small tails needed to dampen the spectral spikes.


Algorithm 1 evaluates _O_ ( _PK_ [2] ) matrix vector products. When these correspond to Hessian vector
products for a model with _d_ parameters evaluated on _m_ examples, this equates to a total runtime
complexity of _O_ ( _PK_ [2] _md_ ). The number of probes _P_ and the Chebyshev degree _K_ are usually
small and can be taken to be fixed relative to _m, d_, so runtime is essentially linear in the number
of parameters and number of examples. Similarly, Algorithm 1 only requires keeping _O_ ( _K_ ) matrix
vector products in memory at once, for a memory footprint of _O_ ( _Kd_ ).


Overlap-KPM combines two standard components—Chebyshev polynomial approximation of
smooth spectral kernels and Hutchinson trace estimation—and therefore its hyperparameter behavior is straightforward. The truncation order _K_ controls only the polynomial approximation error of
the Gaussian kernel; because the kernel is analytic, this error decays exponentially fast in _K_ (Boyd,


33


1989), and in practice the estimate stabilizes rapidly once _K_ exceeds a modest threshold. The num_√_
ber of probes _P_ affects only the Monte-Carlo variance, which decreases at the usual _O_ (1 _/_ _P_ ) rate.

Empirically, we observe that the estimator is stable over wide ranges of _K, P_ (see tests on synthetic
data in F.4).


**Algorithm 1:** Overlap-KPM for Eigenvector Overlaps

**Input:** _A_ ( _v_ ), _B_ ( _v_ ) (normalized MVPs); degree _K_ ; probes _P_
**Output:** Probe moments _Mi,j,k,µ_ for 0 _≤_ _i, k_ _≤_ _K_ ; 0 _≤_ _j_ _≤_ 2 _K_ ; 1 _≤_ _µ ≤_ _P_ .
**for** _µ_ = 1 **to** _P_ **do**

sample probe _vµ_ _∼N_ (0 _, Id_ )

_v_ 0 _,_ 0 _,µ_ _←_ _vµ_ ; _v_ 0 _,_ 1 _,µ_ _←_ _A_ ( _vµ_ )
**for** _i_ = 2 **to** _K_ **do**

_v_ 0 _,i,µ_ _←_ 2 _A_ ( _v_ 0 _,i−_ 1 _,µ_ ) _−_ _v_ 0 _,i−_ 2 _,µ_


**for** _k_ = 0 **to** _K_ **do**

_v_ 1 _,k,µ_ _←_ _B_ ( _v_ 0 _,k,µ_ )
**for** _j_ = 2 **to** 2 _K_ **do**

_vj,k,µ_ _←_ 2 _B_ ( _vj−_ 1 _,k,µ_ ) _−_ _vj−_ 2 _,k,µ_


**for** _i_ = 0 **to** _K_ **do**

**for** _k_ = 0 **to** _K_ **do**

**for** _j_ = 0 **to** 2 _K_ **do**

_Mi,j,k,µ_ _←_ _v_ 0 _,i,µ · vj,k,µ_


F.4 TESTS ON SYNTHETIC DATA


Algorithm 1 with gaussian kernel is applied to synthetic data in Fig. 8. _A, B_ _∈_ R [1000] _[×]_ [1000] are
generated according to
_A_ = _W_ 1 _,_ _B_ = _W_ 2 + _A_ [2] _,_ (70)

where _W_ 1 _, W_ 2 are independent Wishart matrices with aspect ratio _α_ = 5. The left panel shows the
ground truth gaussian-smoothed overlap function of _A, B_ . Note the nontrivial alignment due to _B_ ’s
dependence on _A_ . The right panel shows the approximation generated via Overlap-KPM, showing
good qualitative match.


We performed informal experiments varying _K, P_ to test the robustness of Algorithm 1 (Figure 9).
As expected, accuracy quickly improves and eventually saturates as the Chebyshev approximation
_√_
order _K_ is increased. The variance of the estimator as a function of _P_ decays as _O_ (1 _/_ _P_ ) with a

constant of proportionality that depends on the input matrices. As figure 9 shows, even for a modest
number of probes (eg., _P_ = 4), results can be quite accurate.


F.5 HESSIAN OVERLAPS OF RESNET-20


34


Figure 8: Overlap-KPM algorithm recovers overlaps on synthetic data. _A_ = _Z_ 1 _[⊤][Z]_ [1] _[/m]_ [and] _[B]_ [=]
_Z_ 2 _[⊤][Z]_ [2] _[/m]_ [ +] _[ A]_ [2][, where] _[ Z]_ [1] _[, Z]_ [2] _[∈]_ [R] _[m,d]_ [are independent matrices with iid.] [standard normal entries.]
_d_ = 1000 _, α_ := _m/d_ = 5. Chebyshev degree: _K_ = 45; number of probes _P_ = 4. Left panel shows
actual eigenvector overlaps at eigenvalues _λA, λB_, smoothed with a gaussian kernel of width 1 _/_ 16.
Right panel shows approximation derived from the Overlap-KPM algorithm.


Figure 9: Varying _K, P_ in overlap-KPM. Gaussian kernel with width of 1 _/_ 32. Matrices _A, B_ were
generated as in 8


35


Figure 10: Overlap-KPM algorithm for Hessian overlaps of CIFAR-10-trained ResNet-20. Degree _K_ = 45 and _P_ = 4 probes with Jackson smoothing applied to gaussian kernel Chebyshev coefficients. Left column shows overlaps for _H_ train _, H_ test (ie. balanced test set), while
right column shows overlaps for _H_ train _, H_ test _[′]_ [(imbalanced] [test] [set).] [Top] [row] [shows] [normalized]
overlap functions _OH_ train _,H_ test( _λ_ train _, λ_ test) and _OH_ train _,H_ test _[′]_ [(] _[λ]_ [train] _[, λ]_ test _[′]_ [)][.] [For ease of visualiza-]
tion, bottom row shows test, train sectional densities _OH_ train _,H_ test( _λ_ train _, λ_ test) _µ_ train( _λ_ train) and
_OH_ train _,H_ test _[′]_ [(] _[λ]_ [train] _[, λ]_ test _[′]_ [)] _[µ]_ [train][(] _[λ]_ [train][)][—the average overlap of 1-D] _[ H]_ [test] _[/H]_ test _[′]_ [eigenspaces onto]
_full_ eigenspaces of _H_ train. In both rows, strong diagonal overlaps are visible in the left column that
are reduced or absent in the right column. Note also in the bottom row that the tail of the _λ_ train _≈_ 0
band extends significantly further for _H_ test _[′]_ [than for] _[ H]_ [test][, indicating significant loss of high] _[ H]_ test _[′]_
eigenspace energy into the low-eigenvalue band of _H_ train.


36
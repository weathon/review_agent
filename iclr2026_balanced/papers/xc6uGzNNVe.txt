# PRINCIPAL SPECTRAL REGULARIZATION MAKES MO## MENTUM SURPASS ADAM FOR LLM TRAINING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Adam has been the most popular optimizer for training deep neural networks for
nearly a decade. Recently, Muon, known for its momentum orthogonalization
property, has emerged as a strong alternative to training large language models
(LLMs). However, is orthogonalization over the whole learning space really necessary, especially given the high computational complexity of Newton-Schulz iteration in Muon? To the best of our knowledge, we are the first to report that
Momentum with marginal spectral regularization on very few dimensions can surprisingly surpass Adam. In this work, we mainly made three contributions. First,
from spectral visualizations of the LLM training dynamics and the optimization
of the Styblinski-Tang function, we observe that the full orthogonalization of the
matrix can be suboptimal in some cases. Second, we propose a novel principal
spectral regularization (PSR) method that selectively penalizes only the dominant
components with computational efficiency. Third, we show that the PSR approach
enables SGD with momentum to surpass Adam in pretraining LLMs.


1 INTRODUCTION


Large language models (LLMs) have emerged as a breakthrough and state-of-the-art in various natural language processing (NLP) tasks in recent years. With pretraining on enormous text corpora,
LLMs have demonstrated strong performance in handling responsibilities such as question answering (QA), code generation, and even research assistance, thereby enhancing industry production
efficiency and contributing to broader social welfare (Kumar, 2024; Gao et al., 2025). However,
pretraining LLMs from scratch remains highly resource-intensive for both academia and industry,
requiring substantial infrastructure and specialized machine learning expertise, which motivates the
search for new training techniques that prioritize efficiency.


Optimizers are at the core of training techniques as they dominate computational resources, while
Adam (AdamW) (Kingma, 2014; Loshchilov & Hutter, 2017) is widely regarded as the “king”
of optimizers (Bremen, 2020). By maintaining exponentially decaying moving averages of gradients and second-order momentum, Adam approximates per-parameter curvature with a diagonal
approximation, offering stability and fast convergence across diverse neural architectures and tasks.
Nevertheless, the memory overhead of storing and updating two moments with auxiliary tensors for
all model parameters is significant in LLM pretraining when millions or billions of parameters are
involved. Efforts to advance single-momentum optimizers to comparable performance with Adam
and its variants remain an active area of research (Chen et al., 2023; Shu, 2023; Liang et al., 2024;
Zhang et al., 2025; Glentis et al., 2025).


Recently, spectral methods for accelerating LLM pretraining have garnered increasing attention due
to Muon’s proven benefits in sample efficiency and reduced memory consumption compared to
AdamW (Jordan et al., 2024; Liu et al., 2025a). Spectral regularization has been studied from
multiple perspectives, including penalizing the spectral norm of parameters (Yoshida & Miyato,
2017; Miyato et al., 2018) or gradients (Lewandowski et al.). Another line of research is spectral
preconditioning, explored in methods including preconditioned SGD (Li, 2017), Adafactor (Shazeer
& Stern, 2018), Shampoo (Gupta et al., 2018), SOAP (Vyas et al., 2024), etc., for which matrixshaped preconditioners are maintained instead of scalar preconditioners. Building on prior attempts,
Muon employs the Newton-Schulz iteration to approximate the nearest semi-orthogonalization of
momentum, aiming at achieving nearly uniform updates at all spectral directions. Some researchers


1


even consider Muon as a strong candidate for the next standard optimizer for LLM pretraining,
according to its performance in practice (Team et al., 2025; Shah et al., 2025; Zeng et al., 2025).


In this work, by comparing Momentum, AdamW, and Muon from a spectral perspective, we notice that the spectral structure may play an essential role in LLM pretraining acceleration. We
observe a _spiked-head-heavy-tail_ structure in the original momentum covariance, while Muon has
a very flat spectrum as it orthogonalized the momentum. However, in optimizing the StyblinskiTang function, we discover that penalizing only a fraction of the dominant updates can yield better
convergence compared to SGD-M, Adam, and Muon. This motivates a deeper investigation into
how different spectral components and orthogonalizing them affect training efficiency, intending to
interpolate between the computational trade-offs of full momentum orthogonalization. Building on
our hypothesis, we propose a novel principal spectral regularization (PSR) method that selectively
penalizes dominant directions using a simplified Lanczos bidiagonalization procedure and deflation.
Our methodology is significantly more efficient than the Newton-Schulz iteration in theoretical complexity, which contributes to an in-depth understanding of optimization from a scalable perspective.


This work made three main contributions.


    - Discovering the spectral visualization results of Momentum, Adam, and Muon, and the
intuitive findings from optimizing the Styblinski–Tang function, we raise a key insight that
orthogonalization over the whole space can be questionable for LLM training, due to the
high computational complexity of Newton-Schulz iterations employed in Muon.

    - Motivated by our insight, we propose a **principal spectral regularization (PSR)** method
that selectively regularizes very few “spiked-head” components in the high-dimensional
momentum, which is more efficient than Newton-Schulz theoretically in computational
complexity and empirically in high-dimensional matrices.

    - Our extensive experiments demonstrate that the proposed PSR method can help Momentum
surpass AdamW in LLM pretraining over just a very few dimensions, which revealed the
roles of different spectral components in the momentum spectra.


2 RELATED WORKS


This section reviews previous work on spectral methods in deep learning optimization and optimizers designed with efficiency in LLM pretraining with recent benchmarking experiments.


2.1 SPECTRAL METHODS IN DEEP LEARNING OPTIMIZATION


We consider and compare two types of spectral methods in optimizing deep learning models: spectral regularization and spectral preconditioning. Previous spectral regularization methods mostly
estimate the spectral norm using the largest singular value of the weights or gradients, then add it
as a penalty term to the loss to discourage low-entropy solutions (Yoshida & Miyato, 2017; Miyato
et al., 2018; Lewandowski et al.). Spectral preconditioning optimizers typically leverage the spectral
properties of some matrices associated with the model, usually the gradient/momentum or a Hessian approximation (e.g., the Fisher information (Martens & Grosse, 2015)), and rescale parameter
updates along each spectral direction (Doikov et al., 2024). Early efforts include PSGD/Kron that
approximate the inverse Hessian with Kronecker-factored preconditioners (Li, 2017; 2022), Adafactor with a rank-1 preconditioner for memory efficiency (Shazeer & Stern, 2018), Shampoo with a
full-matrix preconditioner for each dimension of parameters (Gupta et al., 2018), and Sophia with
an online estimate of the second-order preconditioner for scalability (Liu et al., 2023).


In 2023, Jordan et al. introduced Muon, an optimizer that performs efficient momentum orthogonalization by Newton-Schulz iterations to approximate the matrix sign function. This approach
yields a semi-orthogonal momentum update matrix, effectively amplifying the ‘rare directions’, directions with small gradient components but potentially high importance for generalization (Jordan
et al., 2024). Liu et al. further extended Muon by introducing a rescaling scheme that aligns its update RMS with AdamW. Muon and its variants have since been empirically validated in large-scale
LLM pretraining, demonstrating improved sample efficiency (Liu et al., 2025a; Shah et al., 2025;
Team et al., 2025; Zeng et al., 2025). These results have renewed interest in understanding spectral
preconditioning and orthogonalization as fundamental tools for optimization in deep learning.


2


In contrast to previous spectral optimization methods, we notice a gap between regularization applied only to the top spectral direction or to the spectral norm and full-matrix preconditioning. Our
proposed method bridges this gap by first identifying the principal spectral directions and then explicitly regularizing them, offering a scalable perspective and understanding of spectral methods.


2.2 OPTIMIZERS FOR LLM PRETRAINING


AdamW is the most widely used optimizer across deep learning architectures, including both pretraining and fine-tuning of decoder-only transformers (Zhao et al., 2024b). However, it stores both
first- and second-moment estimates, effectively occupying twice the space of weights/gradients in
GPU memory. This has motivated numerous efforts to reduce the memory footprint or improve sample efficiency, particularly for large-scale LLM training. Notable approaches include Adam-mini
with block-wise learning rate schedules based on Hessian partitions (Zhang et al., 2024; Wang et al.,
2025), Lion with momentum-sign updates (Chen et al., 2023) and FOCUS that enhance Signum
with parameter attentions (Liu et al., 2025b), Cautious Adam/Lion with gradient-aligned selective
updates (Liang et al., 2024), SWAN that enhances SGD with whitening and normalization (Ma et al.,
2024), MARS with variance reduction (Yuan et al., 2024), and SOAP that apply AdamW updates to
the Shampoo eigenbasis while amortizing the cost of eigendecomposition over multiple steps.


Recent benchmarking studies show that matrix-based optimizers with spectral preconditioning (e.g.,
Kron, Muon, Soap, etc.) generally outperform scalar-based ones (e.g., AdamW, Lion, Mars, etc.),
though the optimal choice often depends on the specific scenario (Schlotthauer et al., 2025; Wen
et al., 2025; Semenov et al., 2025). We are reminded that empirical studies cannot exhaustively
cover all scenarios, even with optimal hyperparameters and experiment setup guidelines. Consequently, a promising research direction lies in characterizing the regimes where matrix-based spectral methods provide the most benefit, balancing their potential computational overhead against the
sample efficiency and performance gains.


3 INSIGHTS ON SPECTRAL REGULARIZATION


In this section, we started by reviewing the spectral distributions in LLM pretraining and investigated
the concept of principal regularization in mathematical function optimization.


(a) Spectrum of Attention Layer (b) Spectrum of MLP Layer


(c) Momentum Heatmap of Attention Layers (d) Momentum Heatmap of MLP Layers


Figure 1: (Top) Spectral distributions of the final attention layer (a) and the final mlp layer (b).
(Bottom) Heatmaps of spectral distributions across all attention (c) and mlp layers (d). Results are
from an LLaMA-350M model at 1000 training steps on the C4/en dataset. All right-hand figures
highlight the top 64 spectral values. Both gradient and momentum of all layers in the model exhibit
a _spiked-head-heavy-tail_ structure, where a few directions are dominant. Our proposed principal
spectral regularization (PSR) method selectively orthogonalizes momentum along these dominant
directions, balancing efficiency and computational cost.


3


3.1 A SPECTRAL PERSPECTIVE IN LLM PRETRAINING


We begin by analyzing spectral distributions in pretraining LLMs. As illustrated by the spectral distributions of the final attention and MLP layer in Fig. 1, gradients are dominated by a few directions,
followed by a heavy-tailed spectrum. We concluded our observation as a _spiked-head-heavy-tail_
structure, where a few dominant directions (the _spiked-head_ ) capture most of the variance, while
many others persist with smaller contributions. The momentum, as the running average of gradients, exhibits a similar structure but with fewer dominant directions, and _heavy-tail_ lifted, allowing
updates to focus more on important directions accordingly over training iterations. From the momentum spectral heatmaps of attention and mlp layers across a LLaMA-350M model (Fig. 1), a
clear distinction arises between these attention layers (q, k, v, o) and MLP layers (gate, up,
down): attention-layer momentum spectrum tend to decay more sharply, resembling a near _y_ = _x_
slope in the tail and showing greater variability across layers, whereas MLP spectrum exhibit greater
uniformity despite the same presence of dominant directions.


Adam produces updates that mirror the momentum spectrum but with attenuated spikes, using the
second moment to adapt to local curvature. By plotting Gradient, Momentum, AdamW, and Muon
sequentially, we observe that the latter methods increasingly diminish updates in the dominant directions while amplifying those in the tail. This trend continues until the update magnitudes become
nearly uniform across spectral directions, with Muon achieving this by using Newton–Schulz iterations to approximate a semi-orthogonalization of the momentum. This is thought to promote
exploration of those “rare but important directions” in the training process. Yet, such a uniform
update may also amplify noisy directions and become unnecessary in such scenarios; in particular,
the attention layers with momentum exhibited a rapidly decaying heavy-tail, as Adam predicted.


Viewing Adam’s second moment and Muon’s matrix orthogonalization as forms of momentum regularization, we hypothesize that full matrix regularization, considering the computational overhead,
may not be necessary on all occasions. This paper investigates that possibility.


3.2 PRINCIPAL REGULARIZATION IN MATHEMATICAL FUNCTION OPTIMIZATION


(a) Threshold Principal Spectral Regularization (PSR)


(b) Proportional Principal Spectral Regularization (PSR) ( _p_ = 5%)


Figure 2: (Left) Update spectrum at step 1000, (Middle) Training dynamics, and (Right) Final loss
for different numbers of regularized components of the Styblinski–Tang function ( _n_ = 1024) with
weight attribution and Gaussian noise. We present two variants of the principal spectral regularization method: (a) shrinking the _K_ dominant updates to the top- _K_ threshold magnitude, and (b)
scaling the _K_ dominant directions to a proportion of _p_ = 5%.


4


We start our investigation on a minimal benchmark of optimization, the Styblinski-Tang function
defined in _n_ -dimensional space, usually evaluated on the hypercube _xi_ _∈_ [ _−_ 5 _,_ 5]:


Its global minimum _f_ ( _x_ _[∗]_ ) = _−_ 39 _._ 16599 _n_ can be found in _x_ _[∗]_ = ( _−_ 2 _._ 903534 _, . . ., −_ 2 _._ 903534).
To better approximate realistic deep learning scenarios, we assign an additional weight to each
dimension following a power-law distribution as _wi_ = _i_ _[−][α]_ to mimic the heavy-tailed spectrum
commonly observed in neural network gradients or Hessians (Zhao et al., 2024a; Morwani et al.,
2024; Tang et al.), and inject Gaussian noise with mean=0 and std=5e-3 at each step. We set the
dimension of the function _n_ = 1024 and use the baseline optimizer for SGD with momentum _m_ =
0 _._ 9 and the learning rate _η_ = 0 _._ 01, along with Adam with ( _β_ 1 _, β_ 2) = (0 _._ 9 _,_ 0 _._ 95) for comparison.


In addition to SGD-momentum and Adam, we consider and evaluate two forms of partial spectral
regularization approaches: (a) a threshold approach, which shrinks the _K_ dominant updates to the
smallest magnitude among them, and (b) a proportional approach that scales the _K_ dominant updates
by a fixed proportion _p_, as presented in Fig. 2. The threshold regularization approach with _K_ = _n_ =
1024, which enforces uniform updates in all directions, can be viewed as a replication of Muon. All
update vectors are normalized to ensure a consistent step size across the different methods. We track
loss over iterations and report the averaged final loss over multiple runs in Tab. 1.

|Optimization Method|d|4 16 64 256 1024|
|---|---|---|
|SGD-M<br>Adam|5.7264<br>5.6698|/<br>/<br>/<br>/<br>/<br>/<br>/<br>/<br>/<br>/|
|SGD-M-PSR (Thr)<br>SGD-M-PSR (20%)<br>SGD-M-PSR (10%)<br>SGD-M-PSR (5%)<br>SGD-M-PSR (1%)|/<br>/<br>/<br>/<br>/|5.6692<br>5.6426<br>5.6406<br>5.6605<br>5.6796<br>5.6629<br>5.5679<br>5.5775<br>5.6661<br>/<br>5.6561<br>5.5721<br>5.5610<br>5.6498<br>/<br>5.6543<br>5.5664<br>**5.5561**<br>5.6461<br>/<br>5.6618<br>5.6454<br>5.6576<br>5.7096<br>/|


Table 1: Final loss on the Styblinski-Tang function ( _n_ = 1024) for Adam and SGD with momentum,
with and without principal spectral regularization across different values of _d_ .


The spectral distribution of Adam in Fig. 2 is nearly uniform, resembling the Muon spectrum in
practice, with a sharp drop to zero in the final directions. Nevertheless, while both Adam and Muon
converge to lower minima than SGD with momentum, their training trajectories are much slower
and unstable. For principal spectral regularization, the fractional approach consistently outperforms
the streamlined approach. In particular, the configuration with _p_ = 5% and _d_ = 64 achieves the
lowest final loss among all tested optimizers, outperforming standard SGD-M, Adam, and Muon.


Moreover, the loss curves across different choices of _d_ reveal that there is an optimal trade-off
between the number of directions being regularized and the strength of their penalization: too few
directions lead to under-regularization, while too many may suppress useful update components.
This finding suggests that uniform updates in parameters or spectral direction achieved by full matrix
orthogonalization, adopted in Lion and Muon, may be unnecessary or even suboptimal in some
scenarios. While these approaches explore “rare but important directions”, they also risk amplifying
noisy directions, which is especially relevant in the presence of noisy data or small-batch training.
Additional results under more robust experimental settings are reported in Tab. 7, Appendix. 3.


4 PRINCIPAL SPECTRAL REGULARIZATION FOR LLM PRETRAINING


In this section, we propose a principal spectral regularization method for momentum-based optimizers tailored for high-dimensional matrices. According to our hypothesis that regularizing only
a fraction of the dominant spectral components in the momentum can serve as an effective approximation to matrix orthogonalization, without explicitly altering the heavy tail, we approach this
problem using matrix deflation along a set of identified spectral directions. While power iteration
efficiently converges to the largest singular pair, applying it repeatedly to extract multiple adjoint singular directions through deflation becomes computationally prohibitive. To address this, we adopt


5


_f_ ( _x_ ) = [1]

2


_n_


_i_ =1


- _x_ [4] _i_ _[−]_ [16] _[x]_ _i_ [2] [+ 5] _[x][i]_ - (1)


a simplified block Lanczos bidiagonalization procedure (or Golub-Kahan bidiagonalization), which
combines power iteration with orthogonalization against both previous and current blocks. This allows us to identify a richer set of dominant directions in parallel. QR factorization, as the block
orthogonalization technique, is employed at each step to maintain orthonormality and uniqueness
of the approximate left and right singular vectors. By constructing a compact bidiagonal submatrix that captures the dominant spectral structure, we can deflate multiple principal directions in the
momentum simultaneously, reducing computational overhead.


**Algorithm 1** Principal Spectral Regularization (PSR)

**Require:** Momentum _M_ _∈_ R [(] _[m][×][n]_ [)], regularization factor _η_, Lanczos iteration _K_, rank _r_

1: ( _U, B, V_ ) _←_ BIDIAGONAL( _M, K, r_ )
2: ( _Ub, ∗, Vb_ _[⊤]_ [)] _[ ←]_ [SVD][(] _[B]_ [)]
3: _u ←_ _UUb_, _v_ _←_ _VbV_ _[⊤]_

4: _M_ _←_ _M_ _−_ _η u_ ( _u_ _[⊤]_ _Mv_ _[⊤]_ ) _v_
5: _M_ _←_ NORMALIZE( _M_ )
6: **return** _M_


1: **function** BIDIAGONAL( _M, K, r_ )
2: _B_ _←_ 0 _∈_ R _[rm][×][rm]_

3: _u_ 0 _∼N_ (0 _,_ 1) _[N]_ _[×][r]_

4: _u_ 0 _, ∗←_ QR ORTHOGONAL( _u_ )
5: **for** _k_ = 0 to _K −_ 1 **do**
6: _vk_ _←_ _M_ _[⊤]_ _uk_
7: _vk, Rα,k_ _←_ QR ORTHOGONAL( _vk, V_ ~~_b_~~ _locks_ )
8: _B_ [ _rk_ : _r_ ( _k_ + 1) _, rk_ : _r_ ( _k_ + 1)] _←_ _Rα,j_
9: **if** _k_ _≥_ _K −_ 1 **then**
10: **break**
11: **end if**
12: _uk_ +1 _←_ _Mv_
13: _uk_ +1 _, Rβ,k_ +1 _←_ QR ORTHOGONAL( _u, U_ ~~_b_~~ _locks_ )
14: _B_ [ _r_ ( _k_ + 1) : _r_ ( _k_ + 2) _, rk_ : _r_ ( _k_ + 1)] _←_ _Rβ,k_ +1
15: **end for**
16: _U_ _←_ [ _u_ 0 _, . . ., uK−_ 1] _,_ _V_ _←_ [ _v_ 0 _, . . ., vK−_ 1]
17: **return** _U, B, V_
18: **end function**


1: **function** QR ORTHOGONAL( _Q_, prev ~~b~~ locks)
2: **for** each _Q_ prev _∈_ prev ~~b~~ locks **do**
3: _Q ←_ _Q −_ _Q_ prev( _Q_ _[⊤]_ prev _[Q]_ [)]
4: **end for**
5: _Q, R ←_ QR( _Q_ )
6: **return** _Q, R_
7: **end function**


The proposed principal spectral regularization (PSR) method is presented in Alg. 1, in which the
_bidiagonal_ function can be viewed either as a block-wise adaptation of Power Iteration that produces orthonormal bases, or equivalently as an iterative randomized SVD method that incrementally
captures dominant spectral components. By constructing two semi-orthonormal bases _U_ and _V_ and
a reduced bidiagonal form _B_ of input momentum _M_ in the Krylov subspace, SVD is performed
to extract singular vectors ( _Ub, Vb_ ) on _B_ . The single vector groups are reconstructed as _u_ = _UUb_
and _v_ = _VbV_ _[⊤]_, and matrix deflation is applied to _M_ with respect to ( _u, v_ ) using a regularization
factor _η_ . This procedure attenuates the dominant directions identified to 1 _−_ _η_ of their original magnitude. The deflated matrix is then normalized and rescaled to match the update scales of Muon
and AdamW, as discussed below. By suppressing the dominant directions and normalization, the
heavy-tailed spectrum is retained and relatively lifted, effectively amplifying their contributions.


6


|Method|A|PSR(A)|Newton-Schulz(A)|QR-Decomposition(A)|
|---|---|---|---|---|
|_E_ortho(_B_)<br>_D_sub(_A, B_)<br>_D_spec(_A, B_)|8_._0_ ×_ 104<br>/<br>/|32_._0<br>3_._52_ ×_ 10_−_5<br>0_._082|9_._9<br>3_._51_ ×_ 10_−_5<br>0_._209|2_._4_ ×_ 10_−_5<br>3_._47_ ×_ 10_−_5<br>0_._367|


Table 2: Matrix orthonormality _E_ ortho, subspace distance _D_ sub and spectral fidelity _D_ spec scores
between different matrix orthogonalization methods. For spectral fidelity evaluation, the input matrices are normalized to capture the shape difference only. PSR is less accurate than Newton-Schulz
in matrix orthogonalization but achieves a comparable subspace distance and the lowest spectral
fidelity, as it preserves the tail of the spectral distribution.


To verify our methodology, we evaluated and compared (1) orthonormality _E_ ortho( _B_ ) = _∥B_ _[⊤]_ _B −_
_Ir∥F_, (2) subspace distance _D_ sub( _A, B_ ) = _∥Q_ _[⊤]_ _B_ _[Q][B]_ _[−]_ _[Q][⊤]_ _A_ _[Q][A][∥][F]_ [,] [and] [(3)] [spectral] [fidelity]
_D_ spec( _A, B_ ) = _[∥][σ]_ _∥_ _[A]_ _σ_ _[−]_ _A_ _[σ]_ _∥_ _[B]_ 2 _[∥]_ [2] for singular values _σ_ and orthonormal basis _Q_, of our proposed regu
larization to Newton-Schulz and QR decomposition in random matrix _A_ _∈_ R [(1024] _[×]_ [2048)] . Tab. 2
demonstrated that PSR provides a less accurate orthogonalization of the input matrices compared to
Newton–Schulz, yet achieves a subspace distance comparable to the other two methods. Its spectral
fidelity is the lowest among the three methods, as PSR preserves the heavy-tail structure of the original spectrum mostly unchanged. The practical regularization effect is presented in **SGD-M-PSR**
in Fig. 1, where the prominent _spiked-head_ structure nearly disappears and the tail distribution is
further elevated than with AdamW or plain momentum. In the following paragraph, we will discuss
how PSR is integrated into SGD with momentum and its theoretical computational overhead.


**SGD Momentum with PSR:** We adopt a Nesterov-style momentum in SGD according to the empirical verifications of Muon and perform PSR on the lookahead gradient. The hyperparameters of PSR
are configured according to the optimal setup in the Styblinski-tang function: optimal regularization
factor _η_ = 0 _._ 95, and Lanczos iteration _K_ = 2 _, r_ = min( _m, n_ ) _/_ 32 to regularize the top-1 _/_ 16 spectral directions in momentum with adaptivity. Empirical results demonstrated that these constants
provide an optimal balance between convergence performance and computational cost and behave
consistently over different LLM scales and architectures.


**Update RMS Rescaling:** We introduce an extra scaling factor to align the update RMS with that of
Muon and AdamW, thereby reducing the need for additional hyperparameter tuning. PSR normalizes the momentum by its _ℓ_ 2 norm at each step to an RMS of 1 _/_ _[√]_ _mn_ . According to our empirical
observations in Tab. 6 provided in Appendix. C, we multiply the momentum by 0 _._ 18 _[√]_ _mn_ to align
with update RMS with Muon and AdamW, followed by a parameter update with weight decay. The
resulting SGD with PSR momentum is summarized in Alg. 2.


**Algorithm 2** SGD-Momentum with Principal Spectral Regularization (SGD-M-PSR)
**Require:** Weight _Wt−_ 1 _∈_ R _[m][×][n]_ and Momentum _Mt−_ 1 _∈_ R _[m][×][n]_ at step _t −_ 1, _m ≤_ _n_ .


**Computational Complexity Analysis:** All operations in the PSR algorithm rely on low-rank matrix
multiplications, effectively reducing the overall complexity to below cubic order w.r.t. momentum
dimensions. The computational overhead of PSR is characterized by the following Theorem:


**Theorem** **4.1** (Upper Bound of Computation Complexity of PSR) **.** _For_ _iteration_ _number_ _K_ = 2
_and_ _implicit_ _rank_ _r_ = _m/_ 32 _,_ _the_ _extra_ _FLOPs_ _required_ _by_ _PSR_ _compared_ _to_ _SGD_ _are_ _at_ _most_
_O_ overhead(PSR) _<_ 1 _/_ 2 _m_ [2] _n when the parameter dimension satisfies_ 16 _≤_ _m ≤_ _n._


7


1: Initialize _Mt_ _←_ 0 _, t ←_ 0.
2: **for** each step **do**
3: _Gt_ _←∇Lt_ ( _Wt −_ 1) _▷_ Compute gradient
4: _Mt_ _←_ _µMt−_ 1 + _Gt_ _▷_ Accumulate momentum
5: _G_ ˆ _t_ _←_ _µMt_ + _Gt_ _▷_ Nesterov lookahead gradient
6: _O_ ˆ _t_ _←_ PSR( ˆ _Gt,_ _K_ = 2 _,_ _η_ = 0 _._ 5 _,_ _r_ = 32 _[m]_ [)] _▷_ Principal Spectral Regularization

7: _Wt_ _←_ _Wt−_ 1 _−_ _η_ (0 _._ 18 _·_ _O_ [ˆ] _t ·_ _[√]_ _mn_ + _λWt−_ 1) _▷_ Update parameter with weight decay
8: **end for**


The full proof and further discussion are provided in the Appendix. E. This overhead corresponds
only to about 2% of the extra FLOPs, the 30 _m_ [2] _n_ upper bound incurred by the 5-step Newton-Schulz
in Muon (Jordan et al., 2024) for all LLMs and most deep learning models.


**Wall-clock time Comparison:** To fully exploit CUDA acceleration, we utilize the PyTorch API for
QR factorization and SVD. As existing PyTorch linear algebra functions lack native half-precision
arithmetic support with CUDA, explicit data-type conversions are required during mixed-precision
training. Although theoretical analysis suggests that PSR introduces negligible computational overhead, in practice, it can be more time-consuming due to the sequential execution of iterative operations in small-scale LLMs, according to the runtime experiments in Fig. 3. Nevertheless, substantial
gains in speed and memory efficiency are observed for large-scale models beyond 7B parameters
and dimensions exceeding 4096, even under a naive PyTorch implementation. Future work will
focus on kernelizing the Lanczos and deflation procedures to reduce the additional cost in training.


|Method Params|Attention|MLP|
|---|---|---|
|**Method**<br>**Params**|Time<br>Memory|Time<br>Memory|
|**LLaMA-1.3B**|(2048, 2048)|(2048, 5461)|
|NewtonSchulz<br>_T_ = 5<br>PSR (K=2)<br>_r_ = _m_<br>32|2.01 (ms)<br>64.1 (MB)<br>4.85 (ms)<br>18.5 (MB)|4.68 (ms)<br>120.0 (MB)<br>4.44 (ms)<br>48.2 (MB)|
|**LLaMA-3B**|(2560, 2560)|(2560, 6848)|
|NewtonSchulz<br>_T_ = 5<br>PSR (K=2)<br>_r_ = _m_<br>32|4.09 (ms)<br>87.5 (MB)<br>5.54 (ms)<br>29.0 (MB)|8.19 (ms)<br>186.0 (MB)<br>6.09 (ms)<br>74.6 (MB)|
|**LLaMA-7B**|(4096, 4096)|(4096, 11008)|
|NewtonSchulz<br>_T_ = 5<br>PSR (K=2)<br>_r_ = _m_<br>32|14.83 (ms)<br>224.0 (MB)<br>9.63 (ms)<br>74.1 (MB)|30.22 (ms)<br>472.0 (MB)<br>11.77 (ms)<br>189.6 (MB)|
|**LLaMA-70B**|(8192, 8192)|(8192, 16384)|
|NewtonSchulz<br>_T_ = 5<br>PSR (K=2)<br>_r_ = _m_<br>32|110.79 (ms)<br>896.0 (MB)<br>29.93 (ms)<br>296.5 (MB)|184.33 (ms)<br>1536.0 (MB)<br>35.85 (ms)<br>568.5 (MB)|


Table 3: Comparison of orthogonalization methods on BFloat16 tensors. Each block presents Attention and MLP matrix shapes from a representative LLaMA model. Reported values denote the average over 1000 PyTorch runs, with peak GPU memory usage measured via TORCH.CUDA. Despite
the overhead of sequential QR and SVD steps for low-dimensional cases, their reduced complexity
yields significant runtime and memory gains on larger matrices in comparison to the Newton-Schulz
iteration in a naive PyTorch setup.


5 EXPERIMENTS


In this section, we describe our experimental setups for LLM pretraining and present the results to
analyze the impact of different spectral methods on training.


5.1 EXPERIMENT SETUP


In this paper, we follow the experimental setup described in Zhao et al. (2024a) and Raffel et al.
(2020). We focus primarily on the LLaMA architecture with four varying sizes: 350M _,_ 1B _,_ 3B _,_ 7B
parameters. The primary pre-training corpus is the C4/en dataset with a sequence length of 1024.
We consider a total batch size of 512, and evaluate models at 10000 training steps. Additionally,
we trained a LLaMA-1 _._ 3B model until 36B tokens to assess the robustness of the optimizer in extended pretraining. The final model is evaluated on nine downstream benchmarks: ARC (Easy and
Challenge), BoolQ, HellaSwag, OpenBookQA, PiQA, MMLU, WinoGrande, and SciQ. For comparison, we used standard SGD with Nesterov momentum (SGD-M) with Newton-Schulz iteration
(Muon) or principal spectral regularization (SGD-M-PSR) alongside AdamW as the baseline. The
hyperparameters are available in the Appendix. B.


8


(a) LLaMA-1.3B (b) LLaMA-7B


Figure 4: (a) The LLaMA-1.3B model trained on 36B tokens, and (b) the LLaMA-7B model trained
for 10 _,_ 000 steps on the C4/en corpus with a batch size of 1 per node. SGD-M-PSR converges the
fastest during the warm-up phase, though the speed advantage diminishes in later stages of training.


5.2 EXPERIMENT RESULTS


Fig. 3 presents the training dynamics of three LLaMA architectures with four optimizers with different spectral regularization scales: SGD-M, Adam, SGD-M-PSR, and Muon. The experiment results
indicate that while SGD with Nesterov-style momentum approaches AdamW on the LLaMA-350M
model, it is unsuitable for training larger models due to its poor scalability. However, when PSR
is applied, SGD-M consistently surpasses AdamW and even achieves a better validation perplexity
than Muon on the LLaMA-1.3B model, with lower computational FLOPs overhead and reduced
orthogonalization requirements. The results suggest that, under certain circumstances, regularizing
only the leading spectral components in the momentum can achieve performance comparable to
full-matrix orthogonalization, which updates all spectral directions with uniform magnitude.


(a) LLaMA-350M (b) LLaMA-1.3B (c) LLaMA-3B


Figure 3: LLaMA-350M, 1.3B, 3B models trained 10000 steps on the C4/en dataset with four different optimizers: SGD-M, SGD-M-PSR, AdamW, and Muon. SGD-M exhibits the most unstable
pretraining dynamics among all optimizers, whereas SGD-M-PSR matches Muon in both sample
efficiency and validation perplexity.


We further evaluate scaled training scenarios in Fig. 4: a LLaMA-1.3B model trained on 36B tokens
with a total batch size of 2048, and a LLaMA-7B model with local batch size 1 and global batch size
512. For the LLaMA-1.3B model, SGD-M-PSR exhibits the most stable and rapid convergence during warm-up, but is eventually surpassed by Muon as training progresses. Nevertheless, its sample
efficiency relative to AdamW remains consistently stronger in long-term training. The downstream
evaluation in Tab. 4 also indicated that SGD-M-PSR outperforms AdamW on the commonly used
language and knowledge benchmarks. On the other hand, for the LLaMA-7B model, SGD-M-PSR
exhibits a small gap in comparison to Muon, yet still achieves a substantial speed-up over AdamW.


9


|Optimizer|ARC-e ARC-c BoolQ HellaSwag OBQA PiQA MMLU WG SciQ|Avg.|
|---|---|---|
||LLaMA-1.3B [num fewshot = 0]|LLaMA-1.3B [num fewshot = 0]|
|AdamW<br>SGD-M-PSR<br>Muon|43.18<br>25.68<br>57.19<br>46.62<br>30.20<br>69.97<br>22.97<br>**52.64**<br>68.40<br>43.60<br>**25.85**<br>**57.95**<br>46.34<br>30.20<br>**71.16**<br>22.77<br>**52.64**<br>68.50<br>**45.08**<br>25.60<br>57.89<br>**47.52**<br>**30.80**<br>71.11<br>**22.85**<br>52.25<br>**70.30**|46.32<br>46.56<br>**47.04**|
||LLaMA-1.3B [num fewshot = 2]|LLaMA-1.3B [num fewshot = 2]|
|AdamW<br>SGD-M-PSR<br>Muon|47.26<br>25.60<br>50.31<br>46.43<br>**32.40**<br>69.70<br>**25.05**<br>52.41<br>77.80<br>48.11<br>26.54<br>53.09<br>46.48<br>30.60<br>70.46<br>24.32<br>**53.28**<br>77.80<br>**49.24**<br>**27.05**<br>**56.88**<br>**47.34**<br>31.60<br>**71.38**<br>24.39<br>51.62<br>**78.10**|47.44<br>47.86<br>**48.62**|


Table 4: A comparison of average downstream task performance in 0-shot and 2-shot settings of different optimizers on a LLaMA-1.3B model trained with 36B tokens on the C4/en corpus. WG denotes WinoGrande. SGD-M outperforms AdamW with PSR across most benchmarks while falling
short in comparison to Muon.


From the over-trained experiment on LLaMA-1.3B, we observe and report that SGD-M with PSR
is still stricly worse than running Muon, representing a poorer converging ability in the loss-steady
training stage. We hypothesize that this is due to Muon’s full matrix orthogonalization, which further
amplifies and stabilizes the low-magnitude update directions, in the heavy-tail distribution, that PSR
strengthened relatively less. By preserving updates in these weak directions, Muon appears better
equipped to navigate and settle into lower-loss regions of the landscape. Despite this, the proposed
PSR approach demonstrates the optimization trade-offs behind head-only shrinkage and complete
normalization, revealing the different roles played by the head and tail components in the momentum
spectra. We report further ablation studies to analyze the scalability of partial orthogonalization on
optimization performance in the Tab. 9, Appendix. D.


6 CONCLUSION


In this paper, we analyze the preconditioning by Momentum, Adam, and Muon through a universal
spectral regularization perspective. While prior regularization and optimization methods primarily
focus on low-rank projection-based methods to approximate the ideal preconditioner, we propose a
marginal approach that targets a small subset of spectral directions in the original parameter space.
By selectively penalizing the dominant directions in the momentum, our proposed PSR method enables SGD with momentum to surpass Adam in large-scale LLM pretraining. Although our PSR
method does not match Muon in downstream performance or scaled-up training, SGD-M-PSR consumes only 2% of the extra FLOPs required by the Newton-Schulz iteration in theoretical complexity
compared to standard SGDs, highlighting a promising direction for next-generation spectrum-aware
optimizer design. On the other hand, the geometric property of PSR also benefits future empirical
analysis in understanding the roles of different spectral components, enabling an in-depth inspection
of the critical design choices behind AdamW, Muon, and other optimizers. Future work can expand
this line of inquiry by developing more principled mathematical frameworks and more efficient partial orthogonalization strategies.


ETHICS STATEMENT


This paper aims to understand the foundation of deep learning optimization. While it may have
many potential societal consequences, we think none of them must be specifically discussed here.


REPRODUCIBILITY STATEMENT


All experiments in this paper were conducted using publicly available datasets and models. We
provide the detailed training configurations in the Section. 5.1 and Appendix. B, including model
architecture, optimization parameters, and learning rate schedules. The optimizer implementation,
along with scripts for data preprocessing, training, and evaluation, will be released upon publication.


10


REFERENCES


Bremen. Neural networks (maybe) evolved to make adam the best optimizer, 2020. URL [https://parameterfree.com/2020/12/06/](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)
[neural-network-maybe-evolved-to-make-adam-the-best-optimizer/.](https://parameterfree.com/2020/12/06/neural-network-maybe-evolved-to-make-adam-the-best-optimizer/)


Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong,
Thang Luong, Cho-Jui Hsieh, Yifeng Lu, et al. Symbolic discovery of optimization algorithms.
_Advances in neural information processing systems_, 36:49205–49233, 2023.


Nikita Doikov, Sebastian U Stich, and Martin Jaggi. Spectral preconditioning for gradient methods
on graded non-convex functions. _arXiv preprint arXiv:2402.04843_, 2024.


Mingqi Gao, Xinyu Hu, Xunjian Yin, Jie Ruan, Xiao Pu, and Xiaojun Wan. Llm-based nlg evaluation: Current status and challenges. _Computational Linguistics_, pp. 1–27, 2025.


Athanasios Glentis, Jiaxiang Li, Andi Han, and Mingyi Hong. A minimalist optimizer design for
llm pretraining. _arXiv preprint arXiv:2506.16659_, 2025.


Gene H Golub and Charles F Van Loan. _Matrix computations_ . JHU press, 2013.


Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned stochastic tensor optimization. In _International Conference on Machine Learning_, pp. 1842–1850. PMLR, 2018.


Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy
Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL [https:](https://kellerjordan.github.io/posts/muon/)
[//kellerjordan.github.io/posts/muon/.](https://kellerjordan.github.io/posts/muon/)


Diederik P Kingma. Adam: A method for stochastic optimization. _arXiv preprint arXiv:1412.6980_,
2014.


Pranjal Kumar. Large language models (llms): survey, technical frameworks, and future challenges.
_Artificial Intelligence Review_, 57(10):260, 2024.


Alex Lewandowski, Michał Bortkiewicz, Saurabh Kumar, Andr´as Gy¨orgy, Dale Schuurmans, Mateusz Ostaszewski, and Marlos C Machado. Learning continually by spectral regularization. In
_The Thirteenth International Conference on Learning Representations_ .


Xi-Lin Li. Preconditioned stochastic gradient descent. _IEEE transactions on neural networks and_
_learning systems_, 29(5):1454–1466, 2017.


Xilin Li. Black box lie group preconditioners for sgd. _arXiv preprint arXiv:2211.04422_, 2022.


Kaizhao Liang, Lizhang Chen, Bo Liu, and Qiang Liu. Cautious optimizers: Improving training
with one line of code. _arXiv preprint arXiv:2411.16085_, 2024.


Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic
second-order optimizer for language model pre-training. _arXiv preprint arXiv:2305.14342_, 2023.


Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin,
Weixin Xu, Enzhe Lu, Junjie Yan, et al. Muon is scalable for llm training. _arXiv_ _preprint_
_arXiv:2502.16982_, 2025a.


Yizhou Liu, Ziming Liu, and Jeff Gore. Focus: First order concentrated updating scheme. _arXiv_
_preprint arXiv:2501.12243_, 2025b.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. _arXiv_ _preprint_
_arXiv:1711.05101_, 2017.


Chao Ma, Wenbo Gong, Meyer Scetbon, and Edward Meeds. Swan: Sgd with normalization and
whitening enables stateless llm training. _arXiv preprint arXiv:2412.13148_, 2024.


James Martens and Roger Grosse. Optimizing neural networks with kronecker-factored approximate
curvature. In _International conference on machine learning_, pp. 2408–2417. PMLR, 2015.


11


Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization
for generative adversarial networks. _arXiv preprint arXiv:1802.05957_, 2018.


Depen Morwani, Itai Shapira, Nikhil Vyas, Eran Malach, Sham Kakade, and Lucas Janson. A new
perspective on shampoo’s preconditioner. _arXiv preprint arXiv:2406.17748_, 2024.


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text
transformer. _Journal of machine learning research_, 21(140):1–67, 2020.


Joel Schlotthauer, Christian Kroos, Chris Hinze, Viktor Hangya, Luzian Hahn, and Fabian
K¨uch. Pre-training llms on a budget: A comparison of three optimizers. _arXiv_ _preprint_
_arXiv:2507.08472_, 2025.


Andrei Semenov, Matteo Pagliardini, and Martin Jaggi. Benchmarking optimizers for large language
model pretraining. _arXiv preprint arXiv:2509.01440_, 2025.


Ishaan Shah, Anthony M Polloreno, Karl Stratos, Philip Monk, Adarsh Chaluvaraju, Andrew Hojel,
Andrew Ma, Anil Thomas, Ashish Tanwer, Darsh J Shah, et al. Practical efficiency of muon for
pretraining. _arXiv preprint arXiv:2505.02222_, 2025.


Noam Shazeer and Mitchell Stern. Adafactor: Adaptive learning rates with sublinear memory cost.
In _International Conference on Machine Learning_, pp. 4596–4604. PMLR, 2018.


Jianlin Shu. Tiger: A tight-first optimizer, Mar 2023. URL [https://spaces.ac.cn/](https://spaces.ac.cn/archives/9512)
[archives/9512.](https://spaces.ac.cn/archives/9512)


Qian-Yuan Tang, Yufei Gu, Yunfeng Cai, Mingming Sun, Ping Li, Zeke Xie, et al. Investigating the
overlooked hessian structure: From cnns to llms. In _Forty-second_ _International_ _Conference_ _on_
_Machine Learning_ .


Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen,
Yanru Chen, Yuankun Chen, Yutian Chen, et al. Kimi k2: Open agentic intelligence. _arXiv_
_preprint arXiv:2507.20534_, 2025.


Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas
Janson, and Sham Kakade. Soap: Improving and stabilizing shampoo using adam. _arXiv preprint_
_arXiv:2409.11321_, 2024.


Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Lei Wu, et al. The sharpness disparity principle in transformers for accelerating language model pre-training. _arXiv_ _preprint_
_arXiv:2502.19002_, 2025.


Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang. Fantastic pretraining optimizers and where
to find them. _arXiv preprint arXiv:2509.02046_, 2025.


Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability
of deep learning. _arXiv preprint arXiv:1705.10941_, 2017.


Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, and Quanquan Gu. Mars: Unleashing the power
of variance reduction for training large models. _arXiv preprint arXiv:2411.10438_, 2024.


Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang,
Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation
models. _arXiv preprint arXiv:2508.06471_, 2025.


Huishuai Zhang, Bohan Wang, and Luoxin Chen. Adams: Momentum itself can be a normalizer for
llm pretraining and post-training. _arXiv preprint arXiv:2505.16363_, 2025.


Yushun Zhang, Congliang Chen, Ziniu Li, Tian Ding, Chenwei Wu, Diederik P Kingma, Yinyu
Ye, Zhi-Quan Luo, and Ruoyu Sun. Adam-mini: Use fewer learning rates to gain more. _arXiv_
_preprint arXiv:2406.16793_, 2024.


12


Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong
Tian. Galore: Memory-efficient llm training by gradient low-rank projection. _arXiv_ _preprint_
_arXiv:2403.03507_, 2024a.


Rosie Zhao, Depen Morwani, David Brandfonbrener, Nikhil Vyas, and Sham Kakade. Deconstructing what makes a good optimizer for language models. _arXiv preprint arXiv:2407.07972_, 2024b.


A STATEMENT ON THE USE OF LLMS


In preparing this manuscript, LLMs (mostly GPT-4/5) are utilized for linguistic refinement, including the detection and correction of grammar errors or spelling mistakes, and sentence rephrasing to
improve clarity, coherence, and readability. LLMs were also referenced when structuring the paper contents and reviewing missing details, but were not involved in the formulation of ideas, the
execution of experiments, or the generation of experimental results in this article.


B HYPERPARAMETERS


In this paper, we follow the experimental setup described in Zhao et al. (2024a) and Raffel et al.
(2020). The model architecture and respective hyperparameters are presented in Tab. 5. Besides the
learning rate and batch size settings, we use a cosine decay learning rate scheduler with a minimum
learning rate ratio of 0 _._ 1 for all experiments. Weight decay is set to 0 _._ 1, and gradients are clipped
at 1 _._ 0. The LLaMA models are tokenized using the T5 tokenizer, and all training is performed in
_BFloat16_ mixed precision.

|Model|LLaMA-350M LLaMA-1.3B LLaMA-3B|LLaMA-7B|LLaMA-1.3B|
|---|---|---|---|
|Layer num<br>Hidden dim size<br>FFN dim size<br>Attention heads|24<br>24<br>32<br>1024<br>2048<br>2560<br>2736<br>5461<br>6848<br>16<br>32<br>32|32<br>4096<br>11008<br>32|24<br>2048<br>5461<br>32|
|Seq-len|1024|1024|2048|
|LR|3_._0_ ×_ 10_−_4<br>3_._0_ ×_ 10_−_4<br>3_._0_ ×_ 10_−_4|3_._0_ ×_ 10_−_4|3_._0_ ×_ 10_−_4|
|Batch Size<br>GradAcc|8<br>8|1<br>64|8<br>1|
|Total Batch Size|512|512|2048|
|Iterations<br>Warmup iterations|10000<br>1000|10000<br>1000|9000<br>2000|


Table 5: Training configurations for different LLaMA model scales, including architecture details,
sequence length, learning rate, batch size and gradient accumulation steps, and training schedule.


C UPDATE RESCALING


Our empirical observation in Tab.6 reported that AdamW and Muon’s update RMS falls in the range
of [0 _._ 15 _,_ 0 _._ 2]. AdamW shows lower RMS during the warm-up stage but higher at later stages, while
Muon’s update RMS is lower than 0 _._ 2 on average. According to these insights, we opt to set the
rescaling factor to 0 _._ 18 for aligning update RMS with AdamW and Muon in our experiment setups.


13


|Update Step|Update RMS|Adam|Muon|SGD-M-PSR|
|---|---|---|---|---|
|1000|Attention Avg.<br>MLP Avg.|1_._26_ ×_ 10_−_1<br>1_._58_ ×_ 10_−_1<br>|1_._53_ ×_ 10_−_1<br>1_._77_ ×_ 10_−_1<br>|1_._8_ ×_ 10_−_1|
|2000|Attention Avg.<br>MLP Avg.|1_._54_ ×_ 10_−_1<br>1_._88_ ×_ 10_−_1|1_._63_ ×_ 10_−_1<br>1_._79_ ×_ 10_−_1|1_._63_ ×_ 10_−_1<br>1_._79_ ×_ 10_−_1|


Table 6: The update RMS of the three different optimizers at 1000 and 2000 training steps of a
LLaMA-350M model on the C4/en dataset. The update RMS is averaged respectively from all
Attention layers and all MLP layers. As we observe AdamW and Muon update RMS falls below
0 _._ 2, we adopt a rescaling factor of 0 _._ 18 in SGD-M-PSR for alignment.


D ABLATION STUDIES AND ADDITIONAL RESULTS


In this section, we present further ablation studies and additional results as an extension to the main
paper, further evaluating and comparing different optimizer settings. The results for mathematical
function optimization and LLM pretraining are presented in the following subsections, respectively.


D.1 ADDITIONAL RESULTS IN MATHEMATICAL FUNCTION OPTIMIZATION


In the main paper, we only considered a single case of the Styblinski-Tang function optimization
problem. Tab. 7 presents the final loss and standard deviation results from repeated experiments
through different random initialization, under different weight distribution and noise settings. We
have reduced the function dimension and training iterations to 2000 to reduce the time for repeated experiments. We compare Adam to the proportional PSR approach, with parameters top_K_ = _n/_ 16 = 16 and proportion _p_ = 5%, which are optimal parameters in both Styblinski-Tang
function optimization and pretraining LLMs. While the connection between mathematical function
optimization and LLM pretraining is relatively vague, this toy problem served the purpose of demonstrating the motivation of PSR and the hypothesis that full orthogonalization can be sub-optimal.

|Setup|Col2|Power-law Weight w = i−α, Noise ϵ ∼N(0, σ2)<br>i|Col4|Col5|
|---|---|---|---|---|
|**Method**|**Final Loss**|_α_ = 0_._8|_α_ = 1_._5|_α_ = 2|
|Adam<br>SGD-M-PSR|_σ_ = 0_._001|6_._849_ ±_ 3_._160<br>6_._728_ ±_ 3_._056|6_._875_ ±_ 3_._174<br>6_._703_ ±_ 3_._147|6_._850_ ±_ 3_._162<br>6_._739_ ±_ 3_._069|
|Adam<br>SGD-M-PSR|_σ_ = 0_._01|6_._742_ ±_ 3_._135<br>6_._638_ ±_ 3_._101|6_._772_ ±_ 3_._109<br>6_._665_ ±_ 3_._086|6_._839_ ±_ 3_._091<br>6_._694_ ±_ 3_._114|


Table 7: Final loss and standard deviation obtained when optimizing the Styblinski-Tang function
( _n_ = 256) with Adam and SGD-M-PSR, averaged across 1000 random initializations in the range

[ _−_ 5 _,_ 5], and seed in the range [1 _,_ 1000]. The training iterations are set to 2000 and a learning rate
of 0 _._ 05. For PSR, principal spectral regularization, we choose the proportional approach with top_K_ = _n/_ 16 = 16 and _p_ = 5%, which shrinks the top-16 update directions to 5%, aligning with the
parameters used in LLM pretraining experiments.


D.2 ABLATION STUDIES IN LLM PRETRAINING


Tab. 8 presents the downstream task evaluation on the LLaMA-3B and LLaMA-7B model (reported
in Fig. 3), under a 0-shot setting. All models are trained with 2B tokens on the C4/en corpus. We
observe that in the early stage of LLM pretraining (2B tokens), larger models (7B) underperform
smaller models (3B), suggesting slower convergence. The relative ranking of optimizers remains
consistent with the results on LLaMA-1.3B (Tab. 4): Muon performs best, followed by SGD-MPSR, and then AdamW in compressive evaluation. We hope future work will extend these experiments to later training stages.


14


|Optimizer|ARC-e ARC-c BoolQ HellaSwag OBQA PiQA MMLU WG SciQ|Avg.|
|---|---|---|
||LLaMA-3B [num fewshot = 0]|LLaMA-3B [num fewshot = 0]|
|AdamW<br>SGD-M-PSR<br>Muon|17.75<br>42.30<br>57.00<br>29.76<br>**23.02**<br>**16.40**<br>64.80<br>71.70<br>50.36<br>**19.45**<br>**46.17**<br>58.26<br>**30.79**<br>22.92<br>15.00<br>64.74<br>**74.90**<br>51.07<br>19.11<br>45.29<br>**61.90**<br>30.64<br>22.98<br>**16.40**<br>**65.67**<br>73.10<br>**51.62**|41.45<br>42.59<br>**42.97**|
||LLaMA-7B [num fewshot = 0]|LLaMA-7B [num fewshot = 0]|
|AdamW<br>SGD-M-PSR<br>Muon|17.92<br>42.34<br>61.47<br>29.14<br>**23.13**<br>15.00<br>62.95<br>68.10<br>51.07<br>**18.69**<br>43.06<br>**62.26**<br>29.80<br>22.90<br>16.60<br>64.53<br>71.30<br>50.83<br>18.26<br>**43.90**<br>59.54<br>**29.95**<br>23.03<br>**17.60**<br>**64.69**<br>**74.90**<br>**51.85**|41.23<br>42.22<br>**42.64**|


Table 8: A comparison of average downstream task performance in 0-shot settings of different
optimizers on a LLaMA-3B and a LLaMA-7B model trained with 2B tokens on the C4/en corpus.
WG denotes WinoGrande. SGD-M outperforms AdamW with PSR across most benchmarks while
falling short in comparison to Muon.


Table.9 presents the ablation study of SGD-M-PSR under different hyperparameter settings alongside other optimizers. We reported that in terms of final test perplexity, an optimal regularization
factor of _η_ = 0 _._ 95 improves performance consistently across all models as the selected principal
components are punished to 5%.A larger regularization factor can introduce instability in the optimization process, as excessive head-only shrinkage may distorts the overall update directions. On
the other hand, the choice of the rank-proportion coefficient _m/r_ reveals a computational trade-off
between cost and efficiency: penalizing a larger fraction of spectral components improves sample
efficiency, but at the expense of increased computational and memory cost per step. This observation highlights a scalable perspective in spectral preconditioning and points towards the efficiency
of geometric-adaptive methodologies.

|Model|Col2|LLaMA-350M|LLaMA-1.3B|
|---|---|---|---|
|Optimizer|1_ −η_<br>_m/r_|Final Test Perplexity|Final Test Perplexity|
|SGD-M<br>Adam<br>SGD-M-PSR<br>SGD-M-PSR<br>SGD-M-PSR<br>SGD-M-PSR<br>SGD-M-PSR<br>Muon|5%<br>128<br>10%<br>32<br>5%<br>64<br>7_._5%<br>32<br>5%<br>32|24.31<br>23.85<br>23.72<br>23.49<br>23.54<br>23.36<br>22.54<br>**22.49**|22.44<br>19.69<br>18.88<br>18.52<br>18.50<br>18.42<br>**18.30**<br>18.36|


Table 9: A comparison of final test perplexity of SGD-M-PSR with different hyperparameter settings
on a LLaMA-350M and a LLaMA-1.3B model trained 2B tokens on the C4/en corpus. The regularization factor _η_ is chosen from _{_ 0 _._ 9 _,_ 0 _._ 925 _,_ 0 _._ 95 _}_ with the rank proportion coefficient _m/r_ chosen
from _{_ 32 _,_ 64 _,_ 128 _}_ . We observe that an appropriate regularization factor can slightly improve training performance, whereas increasing the proportion of principal components trades higher per-step
computational cost for improved sample efficiency.


D.3 COMPARISON WITH SOAP


Low-rank projection methods are widely used for approximating high-dimensional gradients, momentum, or second-order preconditioners in the field of optimization, including GaLore (Zhao et al.,
2024a), Shamoo (Gupta et al., 2018), SOAP (Vyas et al., 2024), etc. These methods often compute
and maintain a low-rank subspace and project future computations into that subspace, reducing the
computational cost. Therefore, the optimization dynamics often happen in an approximated geometry of the full parameter space. In comparison, PSR identifies the principal spectral components in
the original geometric parameter space, without maintaining any low-rank projectors or preconditioners across update steps/iterations.


15


(a) LLaMA-350M (b) LLaMA-1.3B


Figure 5: LLaMA-350M, 1.3B models trained 10000 steps on the C4/en dataset with five different
optimizers: SGD-M, SGD-M-PSR, AdamW, SOAP, and Muon. In our experiment setup, SOAP
with a preconditioning frequency 10 and hyperparameters _β_ 1 = 0 _._ 95 _, β_ 2 = 0 _._ 99 _, β_ shampoo = 0 _._ 99
outperforms AdamW while falling back to SGD-M-PSR and Muon.


Fig. 5 reports the optimization performance of SOAP in comparison to the optimizers discussed in
this study. We considered the best hyperparameters reported in Vyas et al. (2024) in our reproduction and observed that SOAP performs worse than SGD-M-PSR while also consuming more GPU
memory. Nevertheless, the empirical performance of the so-far mentioned optimizers requires further study in terms of performance scaling laws across different architectures trained to Chinchillaoptimal, which are beyond our computational capabilities. The purpose of our study focuses on the
new partial orthogonalization paradigm, which differs from SOAP principally, and the power of PSR
as a scalable empirical tool to inspect the roles of different spectral components and understand the
success and trade-offs behind Muon.


E COMPUTATIONAL COMPLEXITY ANALYSIS


In this section, we present our proof of Theorem. 4.1 and discuss the overhead with Muon.


**Proof of Theorem. 4.1**


_Proof._ We first analyze the computational cost of PSR by accounting for the FLOPs of each component in the PSR regularization method:


    - **QR-Orthogonal:**


**–** **Orthonormal Projection:** For the previous block input of length _k_, the iterative projection cost is _k_ (4 _mr_ [2] + _mr_ ) or _k_ (4 _nr_ [2] + _nr_ ).

**–** **QR Decomposition:** The classical (Householder) QR decomposition has complexity
in 2 _AB_ [2] _−_ 2 _B_ [3] _/_ 3 FLOPs for matrix of shape [ _A, B_ ] and _A > B_ (Golub & Van Loan
(2013), §5.2.9). Here in the QR-Orthogonal function with input _Q_ of shape [ _m, r_ ] or

[ _n, r_ ], it cost 2 _mr_ [2] _−_ 2 _r_ [3] _/_ 3 or 2 _nr_ [2] _−_ 2 _r_ [3] _/_ 3, upper bounded by 2 _mr_ [2] or 2 _nr_ [2] .


- **Principal Spectral Regularization:**


**–** **Bidiagonalization:** The complete FLOPs for _K_ steps bidiagonalization is 2 _K_ [2] ( _m_ +
_n_ ) _r_ [2] + _[K]_ [(] _[K]_ 2 _[−]_ [1)] ( _m_ + _n_ ) _r_ + (2 _K −_ 1) _mnr_


16


- **Bi-Diagonal:** The Bi-Diagonal function procedure executed two operations: the QROrthogonalization, and power iteration. The double-sided matrix-vector multiplications contribute (2 _K_ _−_ 1) _mnr_ FLOPs in _K_ iterations. The accumulative FLOPs of
QR-Orthogonalization are characterized by 2 _K_ ( _m_ + _n_ ) _r_ [2] + _[K]_ [(] _[K][−]_ [1)] (4 _mr_ [2] + _mr_ ) +


QR-Orthogonalization are characterized by 2 _K_ ( _m_ + _n_ ) _r_ [2] + _[K]_ [(] _[K]_ 2 _[−]_ [1)] (4 _mr_ [2] + _mr_ ) +

_K_ ( _K−_ 1) [2] [2] [2] _[K]_ [(] _[K][−]_ [1)]


2 _−_ 1) (4 _nr_ [2] + _nr_ ) = 2 _K_ [2] ( _m_ + _n_ ) _r_ [2] + _[K]_ [(] _[K]_ 2 _[−]_ [1)]


2 ( _m_ + _n_ ) _r_ .


**–** **SVD:** The classical LAPACK SVD has complexity in 4 _AB_ [2] + 8 _AB_ [2] + 9 _B_ [3] FLOPs
for matrix of shape [ _A, B_ ] and _A_ _>_ _B_ if both singular vectors _U_ and _V_ are required
(Golub & Van Loan (2013), §8.6.3). For our bidiagonal matrix _B_ with shape [ _Kr, Kr_ ]
as input, computing its SVD requires 21( _Kr_ ) [3] FLOPs.

**–** **Matrix Deflation:** The deflation step requires computing four matrix-vector products
and two matrix-wise scalar operations in 4 _mnr_ + 2 _mn_ FLOPs.

**–** **Normalization:** The normalization contribute 2 _mn_ FLOPs for matrix in [ _m, n_ ].


Summing the above contributions yields the total overhead _O_ overhead(PSR) = 21( _Kr_ ) [3] +2 _K_ [2] ( _m_ +
_n_ ) _r_ [2] + _[K]_ [(] _[K]_ 2 _[−]_ [1)] ( _m_ + _n_ ) _r_ + (2 _K_ + 3) _mnr_ + 4 _mn_ .


For _K_ = 2 and _r_ = _m/_ 32, we have


_O_ overhead(PSR) = 168 _r_ [3] + 8( _m_ + _n_ ) _r_ [2] + ( _m_ + _n_ ) _r_ + 7 _mnr_ + 4 _mn_


In summary, under the condition _n ≥_ _m ≥_ 16 _>_ 15 _._ 6, we have _O_ overhead(PSR) _≤_ 2 [1] _[m]_ [2] _[n]_ [.]


**Discussion:** We notice that condition 160 _≤_ _m_ _≤_ _n_ holds for all LLMs, including LLaMA-20M
( _m_ = 256) and GPT2-small ( _m_ = 768), as well as most deep learning architectures. Compared
with the additional FLOPs required by Muon, bounded by 30 _m_ [2] _n_ for 5 Newton-Schulz iterations,
PSR reduces the computational overhead to approximately 2%. Considering the baseline FLOPs
to perform a single forward-backward step of training on a linear layer is 6 _mnB_, where B is the
number of inputs, which is the batch size in tokens for LLMs; The FLOP overhead of PSR is 12 _mB_ [,]
for parameter dimension _m_ . We now calculate the overhead for two concrete training scenarios as
follows Jordan et al. (2024):


    - For GPT2-small with model dimension _m_ = 768 and the example number of tokens per
batch _B_ = 524288, the overhead of PSR is 1 _._ 2 _×_ 10 _[−]_ [4] .

    - For LLaMA-405B training with _m_ = 16384 and tokens per batch _B_ = 1 _._ 6 _×_ 10 [7], the
overhead of PSR is 8 _._ 4 _×_ 10 _[−]_ [5] .


Although the theoretical analysis suggests that PSR introduces negligible computational overhead,
in practice, it can be more time-consuming due to the sequential execution of iterative operations.
Future work will focus on kernelizing the Lanczos and deflation procedures to further reduce this
additional cost during training.


17


= [168]


[168] [1]

2 [15] _[m]_ [3][ +] 2


[1] [7]

2 [7] [(] _[m]_ [3][ +] _[ m]_ [2] _[n]_ [) +] 2 [5]


[7] [1]

2 [5] _[m]_ [2] _[n]_ [ +] 2


2 [5] [(] _[m]_ [2][ +] _[ mn]_ [) + 4] _[mn]_


53
= [29]
4096 _[m]_ [3][ +] 128


[29] [129]

128 _[m]_ [2] _[n]_ [ +] 32


[129] [1]

32 _[mn]_ [ +]


32 _[m]_ [2]


We then derive the condition for the target inequality to hold:


53 [29]
4096 _[m]_ [3][ +] 128


[29] [129]

128 _[m]_ [2] _[n]_ [ +] 32


[129] [1]

32 _[mn]_ [ +]


[1] _[≤]_ [1]

32 _[m]_ [2] 2


2 _[m]_ [2] _[n]_


53

_[≤]_ [140] _[m]_ [2] _[n]_
8 _[m]_ [3][ + 2064] _[mn]_ [ + 16] _[m]_ [2]


53


_m_ [1]

_n_ [+ 2064]


_n_ _[≤]_ [140]


53 _m_


8 _n_


[1] [1]

_m_ [+ 16] _n_


According to our matrix dimension assumption that _m ≤_ _n_, we may derive:


53


_m_ [1]

_n_ [+ 2064]


53 _m_


8 _n_


[1] [1]

_m_ [+ 16] _n_


[1]

_n_ _[≤]_ [53] 8


[53] _m_ [1]

8 _m_ [+ 2064] _m_


[1] [1]

_m_ [+ 16] _m_


_m_ _[≤]_ [140]

[1]

_m_ _[≤]_ [140]


53


53 [1]

8 [+ 2080] _m_


[1]

_m_ _[≤]_ [1067] 8


2080 [1]


_⇒_ _m ≥_ 15 _._ 6
8
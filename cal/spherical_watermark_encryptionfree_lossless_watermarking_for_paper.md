# SPHERICAL WATERMARK: ENCRYPTION-FREE, LOSS- LESS WATERMARKING FOR DIFFUSION MODELS


**Xiaoxiao Hu** [1] **, Jiaqi Jin** [1] **, Sheng Li** [1] **, Wanli Peng** [2] **, Xinpeng Zhang** [1] **, Zhenxing Qian** [1,] _[∗]_

1Fudan University, 2China Agricultural University
{xxhu23,jqjin24}@m.fudan.edu.cn
{lisheng,zhangxinpeng,zxqian}@fudan.edu.cn
wlpeng@cau.edu.cn


ABSTRACT


Diffusion models have revolutionized image synthesis but raise concerns around
content provenance and authenticity. Digital watermarking offers a means of tracing
generated media, yet traditional schemes often introduce distributional shifts and
degrade visual quality. Recent lossless methods embed watermark bits directly
into the latent Gaussian prior without modifying model weights, but still require
per-image key storage or heavy cryptographic overhead. In this paper, we introduce
**Spherical Watermark**, an encryption-free and lossless watermarking framework
that integrates seamlessly with diffusion architectures. First, our binary embedding
module mixes repeated watermark bits with random padding to form a highentropy code. Second, the spherical mapping module projects this code onto the
unit sphere, applies an orthogonal rotation, and scales by a chi-square-distributed
radius to recover exact multivariate Gaussian noise. We theoretically prove that the
watermarked noise distribution preserves the target prior up to third-order moments,
and empirically demonstrate that it is statistically indistinguishable from a standard
multivariate normal distribution. Adopting Stable Diffusion, extensive experiments
confirm that Spherical Watermark consistently preserves high visual fidelity while
simultaneously improving traceability, computational efficiency, and robustness
under attacks, thereby outperforming both lossy and lossless approaches.


1 INTRODUCTION


Diffusion models have demonstrated transformative potential in creative applications (Rombach et al.,
2022; Sahoo et al., 2024), but also raise concerns over authenticity and ownership (Craver et al., 1997;
Grinbaum & Adomaitis, 2022). Malicious users can exploit them to fabricate images and spread
disinformation, eroding public trust and creating legal and ethical challenges. As governments and
platforms face mounting pressure to address harmful content (Biden, 2023; Wiggers, 2023), reliable
provenance mechanisms are urgently needed to trace and identify malicious actors.


Image watermarking offers a promising direction by embedding imperceptible identifiers into images.
However, traditional schemes alter the data distribution and degrade visual fidelity, whether operating
in the spatial (Li et al., 2009; Bender et al., 1995) or frequency (Al-Haj, 2007; Navas et al., 2008) domain. Additionally, some approaches inject watermarks by training or fine-tuning generative models.
For example, Fernandez et al. (Fernandez et al., 2023) fine-tune the Stable Diffusion (Rombach et al.,
2022) decoder to bake in a hidden mark. To avoid costly retraining and improve flexibility, Wen et
al. (Wen et al., 2023) embed ring-patterns in the frequency domains of the latent space. Although
robust to lossy transmission, these methods introduce perceptual artifacts and reduced fidelity.


Recently, the concepts of lossless or undetectable watermarking have been proposed. These methods
seek to establish an invertible mapping from watermark bits to standard Gaussian noise, embedding
watermarks without any modifications to the pretrained generative model. For example, Yang et
al. (Yang et al., 2024) introduce Gaussian Shading which uses repeated watermarks and stream
cipher for sampling but demands a unique key and nonce per image, _incurring substantial storage_


_∗_ Corresponding author.


1


_and_ _management_ _overhead_ . Gunn et al. (Gunn et al., 2025) later replace the stream cipher with
fixed-key pseudorandom error-correcting codes (Christ & Gunn, 2024). Nonetheless, the heavyweight
cryptographic constructs also introduce drawbacks: _they incur nontrivial computational and decoding_
_latency, demand careful parameter tuning to balance code rate and error-correction capability, and_
_fail under strong attacks that exceed the code’s designed distortion bounds._


In this paper, we propose Spherical Watermark, a simple yet effective lossless scheme that is
encryption-free and robust against common attacks. Our method integrates seamlessly with pretrained
diffusion models via three modules: binary embedding, spherical mapping, and diffusion integration.
The binary embedding module mixes watermark bits with random paddings to produce a 3-wise
independent bitstream. The spherical mapping module then projects this bitstream onto the unit sphere,
applies an orthogonal rotation, and scales it by a chi-square-distributed radius. We theoretically
analyze each intermediate distribution and prove that the final noise is statistically indistinguishable
from standard Gaussian noise. In addition, our encryption-free design eliminates the need for perimage key storage. The diffusion integration module then feeds the watermarked noise into Stable
Diffusion (Rombach et al., 2022) to produce watermarked images. Experiments show that our scheme
preserves fidelity and surpasses lossy methods. Compared to lossless approaches (Gunn et al., 2025),
our method offers stronger traceability, reduced complexity, and enhanced reliability.


In summary, our key contributions are three-folded: 1) We propose a novel lossless watermarking
framework, which seamlessly integrates with diffusion-based architectures. Our method guarantees
robust watermark extraction while preserving the original generation fidelity. 2) We introduce a
simple yet effective mapping strategy that transforms binary watermarks into Gaussian noise inputs.
We provide both theoretical analysis and empirical evidence that the watermarked noise distribution
is statistically indistinguishable from a standard multivariate normal distribution. 3) Compared to
existing lossless watermarking schemes, our encryption-free approach omits key storage overhead,
enabling an excellent trade-off between undetectability and watermark robustness.


2 RELATED WORKS


Digital image watermarking has been extensively studied to safeguard intellectual property. Traditional watermarking methods can be applied directly to diffusion outputs, whether operating in the
spatial domain (Li et al., 2009; Bender et al., 1995), the frequency domain (Navas et al., 2008; Liu
et al., 2017; Kashyap & Sinha, 2012), or via neural-network embedding (Zhang et al., 2019; Zhu et al.,
2018; Tancik et al., 2020). In addition, several works embed watermarks by fine-tuning diffusion
models (Fernandez et al., 2023; Xiong et al., 2023; Kim et al., 2024; Wang et al., 2025). For example,
SleeperMark (Wang et al., 2025) introduces a trigger mechanism to decouple watermark information
from semantic content, keeping the watermark extractable after model fine-tuning. More recently,
latent-based watermarking has gained attention. Wen et al. propose the Tree-Ring (Wen et al., 2023)
watermarking scheme, which embeds ring-shaped patterns into frequency domains of the latent space
to enable detection. Subsequent works such as RingID (Ci et al., 2024), SEAL (Arabi et al., 2025b),
and WIND (Arabi et al., 2025a) design alternative patterns. Beyond pattern-based designs, Wei et
al. (Wei et al., 2024) provide a unified analytical framework for diffusion watermarking and instantiate
several distribution-preserving schemes, including truncated Gaussian sampling and Gaussian ring
watermarking. However, these methods are limited to merely verifying the presence of watermark,
not supporting large-scale provenance.


To overcome this limitation, Yang et al. (Yang et al., 2024) introduce Gaussian Shading, a provably
lossless watermarking method that employs repetition codes and a stream cipher to sample from the
standard Gaussian distribution. However, the reliance on a distinct cipher key and nonce for each
generated image imposes a huge key-management overhead that is impractical in the real world. Gunn
et al. (Gunn et al., 2025) advocate replacing the stream cipher with the pseudorandom error-correcting
codes (PRC) (Christ & Gunn, 2024), which allow the generation of distinct pseudorandom sequences
from a fixed secret key. PRC’s extensive cryptographic operations also introduce several challenges.
Encoding and belief-propagation decoding (Pearl, 2014) incur substantial computation and latency.
Finding a trade-off between code rate and error-correction strength requires careful tuning. Moreover,
under aggressive post-processing or shifts in the data distribution, the scheme can hit an irreducible
error floor and fail to recover the watermark. In this paper, we introduce Spherical Watermark, a
framework that eliminates per-image key management, ensures lossless watermark embedding, and
demonstrates superior robustness with high computational efficiency.


2


Figure 1: The overall pipeline of our framework.


3 METHOD


As illustrated in Figure 1(a), our method constructs a tracing mechanism from the model developer’s
perspective. In the offline build phase, the model developer generates a fixed “Signature”, a set of
invertible transforms that encode distinct binary watermarks into the diffusion model’s Gaussian
noise input. During the online runtime phase, API-driven image request automatically applies the
same signature to embed a user-related watermark into the latent code before it is passed through the
diffusion model, ensuring that synthesized images carry traceable provenance. Finally, the developer
inverts generated images to extract watermarks for reliable provenance tracking.


3.1 PROBLEM FORMULATION


The secret watermark **m** encodes API metadata (e.g., user ID, timestamp). Let _G_ : R _[l][x]_ _→I_ denote
a fixed, pretrained diffusion generator that maps standard Gaussian noise **z** to a generated image
**O** . Since diffusion models admit an approximate inverse mapping, we use _G_ _[−]_ [1] to recover the latent
representation from a generated image. Assume the watermark length is _lm_ . Our goal is to design
two efficient procedures in the latent space:


Embed : **m** _∈{_ 0 _,_ 1 _}_ _[l][m]_ _→_ **z** _w_ _∈_ R _[l][x]_ _,_ Extract : ˆ **z** _w_ _∈_ R _[l][x]_ _→_ **m** ˆ _∈{_ 0 _,_ 1 _}_ _[l][m]_ _._ (1)


Specifically, Embed takes **m** to produce the watermarked latent **z** _w_ = Embed( **m** ), and Extract
predicts **m** ˆ from the inverted latent ˆ **z** _w_ = _G_ _[−]_ [1] ( **O** _w_ ), where _lx_ denotes the latent dimensionality of **z** _w_
and **O** _w_ is the generated image with tracable watermark. Let Pr� _·_ - denotes probability, and negl( _ρ_ )
is a function that vanishes faster than any inverse polynomial in the security parameter _ρ_ . We require:


**Undetectability (Losslessness).** For any probabilistic polynomial-time adversary _A_,

��Pr[ _A_ ( **z** _w_ ) = 1] _−_ Pr[ _A_ ( **z** ) = 1]�� _≤_ negl( _ρ_ ) _._ (2)


In other words, watermarked noise **z** _w_ is computationally indistinguishable from standard Gaussian
noise **z** . Thus, for any polynomial-time adversary _A_ _[′]_, the generated images remain indistinguishable:

��Pr� _A_ _[′]_ ( _G_ ( **z** _w_ )) = 1� _−_ Pr� _A_ _[′]_ ( _G_ ( **z** )) = 1��� _≤_ negl( _ρ_ ) _._ (3)


**Traceability (Exact Extraction).** There exists an Extract such that, given watermarked image **O** _w_,


Pr�Extract( _G_ _[−]_ [1] ( **O** _w_ )) = **m**            - _≥_ 1 _−_ negl( _ρ_ ) _._ (4)


That is, the recovered watermark matches the original except with only negligible error in _ρ_ .


For watermarking generated samples, losslessness is the central design principle. It preserves visual
fidelity and underpins robustness in adversarial settings. We formally justify this in Appendix E and
provide empirical evidence in Section 4.2, showing that lossy watermarking can be easily broken by
adversarial attacks, whereas lossless watermarking remains unaffected.


3


3.2 METHODOLOGICAL DESIGN


**Watermark** **Preprocessing.** We represent watermark **m** as independent Bernoulli( [1]


**Watermark** **Preprocessing.** We represent watermark **m** as independent Bernoulli( 2 [)] [bits.] [To]

enhance randomness and error correction, we repeat **m** across _N_ blocks and append a padding vector
**r** _∈{_ 0 _,_ 1 _}_ _[l][r]_, drawn i.i.d. from a Bernoulli( [1] 2 [)][ distribution on each invocation.] [The resulting vector]

**x** = [ **m** **m** _· · ·_ **m** **r** ] _[⊤]_ _∈{_ 0 _,_ 1 _}_ _[l][x]_ _, lx_ = _N_ _× lm_ + _lr,_ (5)


serves as the sole input to the subsequent transforms.


**Build Phase.** In the build phase, the model developer constructs the _Signature K_ = - **T** _,_ **C** �. To
reduce the correlation introduced by repeating **m**, we inject randomness from the padding vector **r** .
Accordingly, the embedding matrix **T** _∈{_ 0 _,_ 1 _}_ _[l][x][×][l][x]_ is designed to mix watermark bits with random
paddings while remaining invertible. The rotation matrix **C**, also invertible, then maps the binary
sequence into Gaussian-like noise. _K_ is kept fixed and secret during runtime to prevent unauthorized
removal. The embedding matrix **T** is constructed from the identity matrices **I** _lNm_ and **I** _lr_ of sizes
_lNm_ and _lr_, together with a sparse binary matrix **R** _∈{_ 0 _,_ 1 _}_ _[l][Nm][×][l][r]_ generated by Algorithm 1:


2 [)][ distribution on each invocation.] [The resulting vector]


**T** = - **I** _lNm_ **R**
**0** **I** _lr_


_, lNm_ = _N_ _× lm._ (6)


The core design lies in **R**, which injects randomness from the padding vector into the watermark bits.
Two parameters govern this construction. The row sparsity _s_ specifies how many random paddings
each watermark bit is mixed with: a larger _s_ improves indistinguishability at the cost of amplified
error propagation (see Section 4.3). In addition, redundancy is introduced through _N_ repetitions,
which enable majority vote decoding. Algorithm 1 ensures that the _N_ copies of each bit are mixed
with disjoint subsets of paddings, guaranteeing the independence property proved in Theorem 3.1.


**Algorithm 1** Construction of Binary Matrix **R**


**Require:** Positive integers _N_, _lm_, _lr_, _s_ such that _lr_ _≥_ _N_ _× s_
**Ensure:** Binary matrix **R** _∈{_ 0 _,_ 1 _}_ _[l][Nm][×][l][r]_, Indices Set _P_

1: **Initialize R** _←_ **0** _[N]_ _[×][l][m][×][l][r]_

2: _P_ _←∅_
3: **for** _j_ = 1 to _lm_ **do**
4: _π_ _←_ RandomPermutation([1 _,_ 2 _, . . ., lr_ ])
5: _TMP_ _←_ _π_ [1 : _N_ _× s_ ]
6: **for** _i_ = 1 to _N_ **do**
7: _G ←_ _TMP_ [( _i −_ 1) _× s_ + 1 : _i × s_ ]
8: **R** [ _i, j, G_ ] _←_ 1
9: _P_ _←_ _P_ _∪{_ ( _i, j, G_ ) _}_
10: **end for**
11: **end for**
12: **Return** Reshape( **R** _,_ ( _lNm, lr_ )), _P_


By design, **T** is bijective over the binary field F2 and its inverse **T** _[−]_ [1] follows that **T** _[−]_ [1] = **T** . And
the rotation matrix **C** _∈_ R _[l][C]_ _[×][l][C]_ is orthogonal, so its inverse satisfies **C** _[−]_ [1] = **C** _[T]_ . We obtain **C** by
drawing a matrix _lC_ _× lC_ with i.i.d. _N_ (0 _,_ 1) and then applying a QR decomposition, retaining the
orthogonal factor. **C** maps the binary sequence into a continuous noise compatible with the latent
input of diffusion models. For notational convenience, we set _lC_ = _lx_ in the following descriptions [1] .


**Runtime Phase.** Latent-based diffusion models adopt the encoder and decoder of VAE (Kingma &
Welling, 2014) to construct bidirectional mappings between the latent and pixel space.


_E_ VAE : _I_ _→_ R _[l][x]_ _,_ _D_ VAE : R _[l][x]_ _→I,_ (7)


denote the pretrained VAE encoder and decoder, respectively. Let **z** _T_ be standard Gaussian noise
in latent space, and let **z** 0 = _E_ VAE( **O** ) denote the clean latent encoding of an image **O** . To
transform **z** _T_ into **z** 0, the diffusion model iteratively perform denoising steps over _T_ discrete timesteps:

1In practice, _lC_ is chosen as a factor of _lx_ (e.g. _lC_ = _⌊_ ~~_[√]_~~ _lx⌋_ ) to balance rotational expressiveness with
computational and storage efficiency.


4


**z** _T_ _→_ **z** _T −_ 1 _→· · ·_ _→_ **z** 0 _._ At each diffusion timestep, the marginal distribution of **z** _t_ is governed
by the probability-flow ordinary differential equation (ODE) (Song et al., 2021b):
_d_ **z** _t_ _[−]_ 12 _[g]_ _t_ [2] _[∇]_ **[z]** _t_ [log] _[ p][t]_ [(] **[z]** _[t]_ [)] _[,]_ (8)

_dt_ [=] _[ f][t]_ [(] **[z]** _[t]_ [)]

where _ft_ and _gt_ are drift and diffusion coefficients determined by the pre-defined noising schedule.
The score function _∇_ **z** _t_ log _pt_ ( **z** _t_ ) is approximated by a neural network _sθ_ ( **z** _t, t_ ). We now describe
how watermark embedding and extraction are seamlessly integrated into the Stable Diffusion pipeline.


Our approach decomposes into three reversible modules: Binary Embedding Module _B_, Spherical
Mapping Module _S_, and Diffusion Integration Module _G_ . As illustrated in Figure 1(b), for watermarked image generation, we first construct the preprocessed input **x** by repeating **m** and appending
random padding **r** . Then binary embedding module _B_ performs the matrix multiplication


**z** [(1)] = **T x** (9)

in F2. Next, spherical mapping module _S_ converts **z** [(1)] _∈{_ 0 _,_ 1 _}_ _[l][x]_ into Gaussian noise by

**v**
**v** = 2 **z** [(1)] _−_ **1** _,_ **z** [(2)] = _,_ **z** [(3)] = **C z** [(2)] _,_
_∥_ **v** _∥_ 2 (10)

draw _r_ such that _r_ [2] _∼_ _χ_ [2] ( _lx_ ) _,_ **z** _w_ = _r_ **z** [(3)] _._


Here, _∥· ∥_ 2 denotes the Euclidean norm, and _χ_ [2] ( _lx_ ) is the chi-square distribution with _lx_ degrees
of freedom. The diffusion integration module _G_ then generates the watermarked image. We set the
initial noise **z** _T_ = **z** _w_, and by solving Eq. 8 from _t_ = _T_ to _t_ = 0, recover the clean latent **z** 0 from **z** _T_,

**z** 0 = ODESolve� **z** _T_ ; _sθ,_ cond _,_ _T,_ 0� _._ (11)


Here, cond denotes sampling conditions (e.g. text prompts), and ODESolve may be instantiated
with different solvers such as DDIM (Song et al., 2021a), DPM-Solver (Lu et al., 2022; 2025),
or other ODE integrators. **z** 0 is then passed through _D_ VAE to produce the watermarked image
**O** _w_ = _D_ VAE� **z** 0� _._


For watermark extraction, the developer applies the inverse modules in the order _G_ _[−]_ [1], _S_ _[−]_ [1], _B_ _[−]_ [1] on
a suspect image **O** [ˆ] _w_ . Specifically, the developer uses _E_ VAE to estimate the latent ˆ **z** 0 = _E_ VAE( **O** [ˆ] _w_ ),
and then solves Eq. 8 from _t_ = 0 to _t_ = _T_ to obtain an estimate of the initial noise:

**z** ˆ _T_ = ODESolve� **z** ˆ0; _sθ,_ ∅ _,_ 0 _,_ _T_             - _._ (12)


Here, ∅ denotes the empty condition (no text prompt). Finally, the developer inverts ˆ **z** _T_ as,

**z** ˆ [(2)] = **C** _[−]_ [1] **z** ˆ _T,_ ˆ **z** [(1)] = round� **z** ˆ(2)2+ **1**          - _,_ ˆ **x** = **T** _[−]_ [1] **z** ˆ [(1)] _,_ (13)

where round( _·_ ) refers to the rounding operation. The first _lNm_ entries of **x** ˆ correspond to _N_ repeated
copies of the watermark message. We therefore apply a majority-vote rule across each group of _N_ bits
to obtain the final decoded watermark **m** ˆ . To avoid ties, _N_ is chosen to be odd. Our embedding and
extraction pipeline guarantees high-precision watermark retrieval for reliable provenance tracking.


3.3 THEORETICAL ANALYSIS


In this section, we provide theoretical guarantees that, after the successive mappings **x** _→_ **z** [(1)] _→_
**z** [(2)] _→_ **z** [(3)] _→_ **z** _w_, the final latent code **z** _w_ is distributed as _N_ ( **0** _,_ **I** _lx_ ) in R _[l][x]_ . The detailed proofs of
all lemmas and theorems stated in this section are provided in the Appendix C.


First, we analyze the distribution of **z** [(1)] in Theorem 3.1. By introducing **r** and carefully designing **T**,
we ensure that the resulting high-entropy code **z** [(1)] exhibits strong independence properties.

**Theorem 3.1.** _If_ **m** _and_ **r** _consist of independent_ Bernoulli( [1] 2 [)] _[ bits, then for]_ **[ z]** [(1)] _[ in Eq. 9, we have]_

_zi_ [(1)] _∼_ Bernoulli� 12 - _for every i ∈{_ 1 _, . . ., lx}, and_ **z** [(1)] _is both 2-wise and 3-wise independent._


Building on the properties established in Theorem 3.1, we show that **z** [(2)] satisfies the conditions of a
spherical 3–design. A spherical _t_ -design (Bannai, 1979; Bajnok, 1992) is a finite set of points on the
unit sphere that, _up to degree t_, exactly matches the averages of all real polynomials with those of
the continuous uniform distribution. Consequently, it can be regarded as an _approximate_ uniform
distribution on the unit sphere. The rigorous mathematical definition of a spherical _t_ –design is as,


5


**Definition 3.1** (Spherical _t_ -Design) **.** _A finite set of points X_ = _{x_ 1 _, . . ., xN_ _} ⊂_ _S_ _[n][−]_ [1] _on the unit_
_sphere in_ R _[n]_ _is called a_ spherical _t_ -design _if, for every real polynomial f_ _of total degree at most t,_
1            - _f_ ( _x_ ) = 1            _N_ _x∈X_ _|S_ _[n][−]_ [1] _|_ _S_ _[n][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[,]_

_where dσ is the uniform surface measure on S_ _[n][−]_ [1] _, and |S_ _[n][−]_ [1] _| denotes the total surface area of the_
_unit_ ( _n −_ 1) _-sphere._


Equivalently, _X_ is a _t_ -design if and only if it _matches all moments_ of the uniform distribution on
the sphere up to degree _t_ . Consequently, a spherical _t_ -design is indistinguishable from the uniform
distribution on _S_ _[n][−]_ [1] by any statistic of degree _≤_ _t_, and thus may be viewed as an _approximation_ to the
uniform spherical distribution. We derive that the set of **z** [(2)] is a spherical 3-design in Theorem 3.2.

**Theorem** **3.2.** **z** [(2)] _satisfies_ _that_ _each_ _zi_ [(2)] _takes_ _values_ _±_ ~~_√_~~ 1 _lx_ _with_ Pr[ _zi_ = + ~~_√_~~ 1 _lx_ ] = Pr[ _zi_ =

_−_ ~~_√_~~ 1 _lx_ ] = [1] 2 _[,][ i][ ∈]_ [(1] _[,][ · · ·]_ _[, l][x]_ [)] _[;]_ **[ z]** [(2)] _[is 2-wise and 3-wise independent.]_ _[Then the finite set of]_ **[ z]** [(2)] _[on the]_

_unit sphere S_ _[l][x][−]_ [1] _is a spherical 3–design._


- _f_ ( _x_ ) = 1

_|S_ _[n][−]_ [1] _|_
_x∈X_


_S_ _[n][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[,]_


_−_ ~~_√_~~ 1 _lx_ ] = [1] 2


Finally, the following two lemmas analyze the distributions of **z** [(3)] and **z** _w_ . In Lemma 3.3 we show
that the orthogonally rotated vector **z** [(3)] remains uniformly distributed on _S_ _[l][x][−]_ [1] . In Lemma 3.4 we
prove that scaling by _r_ _∼_ _χ_ ( _lx_ ) yields **z** _w_ = _r_ **z** [(3)] _≈N_ ( **0** _,_ **I** _lx_ ) in R _[l][x]_ . The detailed proofs are given
in the Appendix C, and our experiments confirm that the empirical distribution of **z** _w_ is statistically
indistinguishable from standard Gaussian distribution in Section 4.2.

**Lemma** **3.3.** _Let_ **z** [(2)] _∈_ _S_ _[l][x][−]_ [1] _be_ _a_ _spherical_ 3 _–design._ _If_ _we_ _apply_ _a_ _fixed_ _orthogonal_ _rotation_
**z** [(3)] = **C z** [(2)] _, then_ **z** [(3)] _is also a spherical_ 3 _–design._ _For each coordinate zi_ [(3)] _, one has_ E[ _zi_ ] = 0
_and_ E[ _zi_ [2][] = 1] _[/l][x][, and as][ l][x]_ _[→∞][, the marginal law of][ z]_ _i_ [(3)] _converges to N_ (0 _,_ 1 _/lx_ ) _._
**Lemma 3.4.** _Let_ **n** _∼N_ ( **0** _,_ **I** _n_ ) _be a standard multivariate normal vector in_ R _[n]_ _._ _Then_ **n** _admits a_
_polar decomposition of the form_
**n** = _r ·_ **u** _,_
_where r_ [2] _∼_ _χ_ [2] ( _n_ ) _, and_ **u** _is uniformly distributed on the unit sphere S_ _[n][−]_ [1] _._ _Furthermore, r and_ **u** _are_
_statistically independent._ _Conversely, if r_ [2] _∼_ _χ_ [2] ( _n_ ) _,_ **u** _is uniformly distributed on S_ _[n][−]_ [1] _, and r_ _⊥_ **u** _,_
_then the product_ **n** = _r ·_ **u** _follows a standard multivariate normal distribution, i.e.,_ **n** _∼N_ ( **0** _,_ **I** _n_ ) _._


4 EXPERIMENT


4.1 EXPERIMENTAL SETTINGS


**Implementation Details.** We adopt Stable Diffusion (SD) v1.5 [2] and v2.1 [3] as backbone generative
models. Generated images are 512 _×_ 512 color images with latent size 4 _×_ 64 _×_ 64. During the
diffusion process, we use a 50-step DPM-Solver++ (Lu et al., 2025) for image generation with a
guidance scale of 7.5 and a 50-step DDIM inversion (Song et al., 2021a) with a guidance scale of 1.0.
To simulate real-world scenarios, DDIM inversion uses empty prompts. Default settings are _N_ = 31,
_lm_ = 512, _lr_ = 512, and _s_ = 1, giving _lNm_ = 15872 and _lx_ = 16384, which matches the diffusion
latent dimensionality. All experiments are conducted in PyTorch on four NVIDIA RTX 4090 GPUs.


**Watermark** **baselines.** We consider the following baselines: traditional watermarking methods
include DwtDct (Al-Haj, 2007), DwtDctSvd (Navas et al., 2008), and RivaGAN (Zhang et al., 2019),
all configured to embed 32-bit watermarks. Latent-based baselines include Tree-Ring (Wen et al.,
2023), Gaussian Shading (Yang et al., 2024), and PRC Watermark (Gunn et al., 2025). All schemes
are evaluated with 512-bit watermarks, except Tree-Ring, which supports detection only. For latentbased methods, we generate five fixed keys (or signatures) and report the mean and standard deviation
of each metric over five independent runs. Unless noted otherwise, baselines use their default settings.
Note that with fixed keys, Gaussian Shading no longer achieves true losslessness.


**Datasets & Evaluation metrics.** For text prompts, we use two datasets, termed COCO and SDP. Each
comprises 1000 text prompts randomly sampled from the MS-COCO val2017 set (Lin et al., 2014)


[2https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
[3https://huggingface.co/stabilityai/stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)


6


Table 1: FID value for different watermarking methods. Lower FID indicates higher image quality.
Mean _Std_ represents the mean value with 1-sigma error bar.


COCO SDP
Method

SD v1.5 SD v2.1 SD v1.5 SD v2.1


Original 48.12561 _._ 3744 46.81461 _._ 0617 49.70410 _._ 5425 46.40600 _._ 5231
DwtDct 48.29751 _._ 3918 46.97711 _._ 0702 49.98530 _._ 5385 46.73040 _._ 5163
DwtDctSvd 48.71791 _._ 4075 47.40491 _._ 0121 51.01600 _._ 6162 47.50440 _._ 6439
RivaGan 48.79561 _._ 3952 47.61241 _._ 1012 51.27730 _._ 6320 47.82980 _._ 6748
Tree-Ring 49.33181 _._ 5108 47.87211 _._ 1320 50.64911 _._ 0197 47.39170 _._ 7127
Gaussian Shading 50.69681 _._ 3200 49.43791 _._ 0326 51.52210 _._ 8773 48.25390 _._ 4859
PRC Watermark 48.13481 _._ 3074 46.75441 _._ 0748 49.52500 _._ 7651 46.41570 _._ 3445
Ours 48.12241 _._ 5489 46.81321 _._ 0962 49.38940 _._ 7475 46.43110 _._ 3695


1.0


0.9


0.8


0.7


0.6


0.5


|Tree-Ring PRC watermark|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|Tree-Ring<br>Gaussian Shading<br>PRC watermark<br>Spherical Watermark|Tree-Ring<br>Gaussian Shading<br>PRC watermark<br>Spherical Watermark|Tree-Ring<br>Gaussian Shading<br>PRC watermark<br>Spherical Watermark|Tree-Ring<br>Gaussian Shading<br>PRC watermark<br>Spherical Watermark|Tree-Ring<br>Gaussian Shading<br>PRC watermark<br>Spherical Watermark||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


1 1001 2001 3001 4001
Epoch


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


1 1001 2001 3001 4001
Epoch


1.0


0.9


0.8


0.7


0.6


0.5


0.4


1 11 21 31 41 51 61 71 81 91
Epoch


1 11 21 31 41 51 61 71 81 91
Epoch


Figure 2: Classification performance over training epochs for distinguishing watermarked from
unwatermarked samples. Left Two: Training loss and test accuracy at latent-level. Right Two:
Training loss and test accuracy at image-level on SDP dataset with SD v2.1.


Original DwtDct DwtDctSvd RivaGan Tree-Ring Gaussian Shading PRC Watermark Ours


Figure 3: Examples of different watermarking methods. Top: COCO dataset. Bottom: SDP dataset.


and the Stable Diffusion Prompt dataset [4], respectively. To evaluate the performance of our method, we
focus on two core criteria: undetectability and tracing accuracy. For undetectability, we first assess any
degradation introduced by watermark embedding. To detect subtle distributional shifts, we employ
the Fréchet Inception Distance (FID) (Heusel et al., 2017) measured against the unwatermarked
output distribution. We also train binary classifiers on both image-level pixels and latent-space inputs
to distinguish watermarked from non-watermarked samples, reporting classification accuracy to
reveal detectable artifacts introduced by the watermark embedding. Next, we evaluate the reliability
of watermark extraction for 100 distinct users under common storage and transmission degradations,
including post-processing attacks and adversarial attacks from WEvade (Jiang et al., 2023). Extraction
performance is quantified by bit-level accuracy (ACC) and the true positive rate at 1% false positive
rate (TPR@1%FPR). For simplicity, we abbreviate TPR@1%FPR as TPR in the sequel.


We report mean and standard deviation over five runs for all metrics. Additional experimental results
are provided in Appendix F, including further undetectability experiments and ablation studies.


4.2 PERFORMANCE ANALYSIS


**Undetectability.** To assess undetectability, we train classifiers to capture distributional shifts. First,
we train a two-layer MLP (Rumelhart et al., 1986) for latent-level classification. According to
Figure 2, both Tree-Ring and Gaussian Shading (with fixed keys) are easily detected with accuracies
of 100% and 97%, while PRC Watermark and our method remain indistinguishable. Second, we


4https://huggingface.co/datasets/Gustavosta/Stable-Diffusion-Prompts


7


Table 2: Comparison results on ACC and TPR. Dataset: COCO dataset. Model: SD v2.1. “Post.”
refers to “Post-Processing” and “Adv.” refers to “Adversarial”.


Metrics
Method

ACC (Clean) ACC (Post.) ACC (Adv.) TPR (Clean) TPR (Post.) TPR (Adv.)


DwtDct 90.141 _._ 15 64.751 _._ 08 49.280 _._ 00 92.803 _._ 14 52.233 _._ 41 16.150 _._ 02
DwtDctSvd 100.000 _._ 00 93.210 _._ 17 48.950 _._ 01 100.000 _._ 00 91.940 _._ 68 17.050 _._ 02
RivaGan 99.680 _._ 10 96.780 _._ 22 52.310 _._ 01 100.000 _._ 00 99.130 _._ 22 26.750 _._ 02
Tree-Ring  -  -  - 100.000 _._ 00 98.850 _._ 31 6.710 _._ 02
Gaussian Shading 100.000 _._ 00 98.430 _._ 04 88.060 _._ 11 100.000 _._ 00 99.970 _._ 04 99.230 _._ 00

PRC Watermark 100.000 _._ 00 93.520 _._ 20 97.690 _._ 07 100.000 _._ 00 87.030 _._ 39 95.380 _._ 00
Ours 99.990 _._ 01 95.020 _._ 08 98.120 _._ 04 100.000 _._ 00 97.500 _._ 18 99.830 _._ 00


sample one prompt and generate ten watermarked images per user across 100 distinct users for
image-level evaluation, with qualitative examples shown in Figure 3. In Figure 2, we also train a
ResNet-18 classifier (He et al., 2016) for image-level classification. Tree-Ring and Gaussian Shading
are detectable, while PRC Watermark and ours show near-chance detection (50%). Table 1 shows
that only PRC Watermark and our method match the original in FID, whereas other methods incur
distribution shifts. These results support our theoretical analysis in Section 3.3 by showing that
watermarked samples are statistically indistinguishable from unwatermarked ones.


**Computational Efficiency.** To demonstrate the
advantages of our encryption-free design, we
evaluate the embedding and extraction times of
latent-based watermarking schemes, with each
result averaged over 100 trials. In this comparison, we focus exclusively on the transformation
between the watermark and its latent noise representation, excluding any diffusion sampling
or inversion procedures. As illustrated in Figure 4, we employ a logarithmic scale on the yaxis for visualization. The extraction time of the
PRC Watermark is much higher than that of ours,
roughly four orders of magnitude slower on extraction. This difference reflects the computational burden introduced by belief-propagation
decoding in the PRC scheme. In contrast, our
approach eliminates the need for complex key
design, thereby enhancing execution speed, improving computational efficiency.


10 [1]


10 [0]


10 1


10 2


10 3


10 4


Tree-Ring Gaussian Shading PRC Watermark Ours
Methods


Figure 4: The watermark embedding and extraction time for different latent-based schemes. Here,
the Y-axis is on a logarithmic scale.


**Tracing Accuracy.** In Table 2, we evaluate tracing accuracy under varied conditions. “Clean” refers
to PNG storage, “Post-Processing” reports common post-processing distortions, and “Adversarial”
refers to attacks from (Jiang et al., 2023) (See Appendix F.4 for details). Compared to lossy schemes,
our method achieves comparable accuracy above 95% in both Clean and Post-Processing settings.
We introduce a tunable parameter _s_, which entails a slight robustness trade-off relative to Gaussian
Shading. Under Adversarial conditions, however, the accuracy of lossy schemes degrades sharply, as
their embeddings enable effective classifiers to be trained for watermark detection, which can then
be adversarially attacked. In contrast, lossless schemes demonstrate clear superiority: our method
improves accuracy by more than 10%, consistent with the theoretical analysis in Appendix E.


**Comparison with PRC Watermark.** In Table 2 and Figure 5, we compare our method with PRC
Watermark under varied distortions. Our method consistently achieves higher TPR and ACC, with
a larger margin at stronger distortions. In addition, Figure 6(a) examines the effect of watermark
capacity _lm_ on tracing accuracy under JPEG–70 compression. As _lm_ increases, PRC Watermark’s
decoding performance deteriorates rapidly and fails entirely beyond _lm_ = 2000. In contrast, _Spherical_
_Watermark_ sustains high detection rates across the full capacity range. Furthermore, the computational
efficiency comparisons show that our embedding and extraction incur significantly lower overhead
than PRC Watermark, with extraction being about four orders of magnitude faster. These results
confirm the superior robustness of our method.


8


1.01


1.00


0.99


0.98


0.97


0.96


0.95


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


1.0


0.8


0.6


0.4


0.2


0.0


|Col1|PRC. ACC PRC. TPR@1%FPR Ours ACC Ours TPR@1%FPR|Col3|Col4|Col5|
|---|---|---|---|---|
||PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR||
||||||
||||||
||||||
||||||
||||||


|PRC. ACC PRC. TPR@1%FPR Ours ACC Ours TPR@1%FPR|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|
||||||||||
||||||||||
||||||||||
||||||||||


|PRC. ACC PRC. TPR@1%FPR Ours ACC Ours TPR@1%FPR|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR|PRC. TPR@1%FPR<br>Ours TPR@1%FPR||
||||||||
||||||||
||||||||
||||||||
||||||||


1 2 3 4 5 6 7 8
Brightness Factor


(c) Brightness.


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


1.0


0.8


0.6


0.4


0.2


0.0


0.1 0.2 0.3 0.4 0.5 0.6 0.7
Random Drop Ratio


(d) Drop.


COCO-SD v1.5 COCO-SD v2.1 SDP-SD v1.5 SDP-SD v2.1
Categories


(a) PNG.


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


0.4


0.3


0.05 0.10 0.15 0.20 0.25 0.30
Noise Standard Deviation


(b) Gaussian Noise.


|Col1|PRC. ACC Ours ACC|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
||||||||
||||||||
||||||||
||||||||


|Col1|PRC. ACC Ours ACC|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|PRC. ACC Ours ACC|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


5 10 15 20 25 30
Kernel Size


(e) Gaussian Blur.


90 80 70 60 50 40 30 20 10

JPEG Quality Factor


(f) JPEG.


4 6 8 10 12 14 16
Kernel Size


(g) Median Filter.


0.7 0.6 0.5 0.4 0.3 0.2 0.1

Resize Ratio


(h) Resize.


Figure 5: ACC and TPR values under Attacks, averaged over two datasets and two models.


0.508


0.506


0.504


0.502


0.500


0.498


0.496


0.494

N (1, 31) lm (256, 512) lr (256, 512) s (1, 31)


(d) Ablation on Settings.


1.0


0.9


0.8


0.7


0.6

|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
||||||||||
||||||||||
||||||||||


Brightness Factor


(c) Ablation on Modules.


|Col1|PRC. ACC Ours ACC|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||
|||||||||
|||||||||


Capacity


(a) Ablation on _lm_ .


0.9


0.8


0.7


0.6


0.5


1 501 1001 1501 2001 2501
Epoch


(b) Ablation on Modules.


Figure 6: Ablation Study. (a) ACC comparisons under different capacities, (b) and (c) Undetectability
and robustness of module ablation, (d) Undetectability analysis of varied parameter settings. Here,
(a-c) are all conducted on the COCO dataset with SD v2.1.


4.3 ABLATION EXPERIMENTS


**Ablation on Modules.** In our ablation study, we isolate the effects of each module. In one variant, we
omit the spherical mapping _S_ and substitute the Gaussian Shading transform; in another, we skip the
binary embedding _B_ and apply only the spherical mapping to **x** . We then evaluate both latent-level
undetectability and tracing accuracy. In Figure 6(b), omitting the binary embedding makes the latent
noise trivially distinguishable. Figure 6(c) shows that robustness under brightness adjustment drops
dramatically without spherical mapping. These results confirm that binary embedding enforces
independence, while spherical mapping is essential for restoring robustness. A rigorous analysis of
why our orthogonal rotation design achieves optimal robustness is provided in Appendix D.


**Ablation on Parameters.** We further investigate the sensitivity of our method to hyperparameters: the
watermark length _lm_, the padding length _lr_, the row sparsity parameter _s_, and the repetition count _N_ .
In Figure 6(d), we vary these parameters and train a latent-level classifier to evaluate their effect. The
results show that classification accuracy remains near 50%, indicating that parameter changes do not
impair undetectability. As _s_ increases, each watermark bit depends on more paddings, making errors
more likely to propagate and amplify. Similarly, reducing _N_ decreases redundancy for majority-vote
correction. Thus, both larger _s_ and smaller _N_ reduce accuracy by design, a trend also confirmed
by the experimental results in Table 3. In addition, Figure 6(a) shows that _Spherical_ _Watermark_
maintains high detection rates across all watermark capacities under JPEG–70 compression.


**Ablation on Diffusion Sampling Settings.** We conduct ablation studies on the COCO dataset using
the SD v2.1 model to assess the sensitivity of our method to diffusion sampling configurations.
Table 4 compares watermark extraction accuracy under various attacks across three ODE solvers:
DDIM (Song et al., 2021a), PNDM (Liu et al., 2022), and DPM-Solver++ (Lu et al., 2025). Settings
of each attack type are provided in Appendix F.5. We then investigate the role of generation and


9


Table 3: Ablation of parameters _s_ and _N_ on TPR under different attack. Dataset: COCO. Model: SD
v2.1. Case 1: Gaussian Blur, kernel size = 9. Case 2: JPEG-70. Case 3: Brightness, factor = 2.


sparsity parameter _s_ repetition count _N_
Case

1 2 3 4 1 11 21 31


1 100.000 _._ 00 99.840 _._ 12 99.080 _._ 25 97.580 _._ 34 99.400 _._ 14 99.960 _._ 05 99.940 _._ 05 100.000 _._ 00
2 99.940 _._ 05 99.460 _._ 26 96.640 _._ 42 92.380 _._ 74 98.080 _._ 32 99.860 _._ 14 99.920 _._ 12 99.940 _._ 05
3 99.720 _._ 12 98.000 _._ 21 93.120 _._ 53 83.680 _._ 71 95.100 _._ 45 99.500 _._ 24 99.680 _._ 19 99.720 _._ 12


Table 4: Ablation results of extraction accuracy on ODE solvers under post-processing perturbations.


Post-processing Perturbations
Solver

PNG Brightness Gaussian Blur Median Filter JPEG Resize


DDIM 99 _._ 980 _._ 01 96 _._ 060 _._ 23 99 _._ 430 _._ 02 99 _._ 200 _._ 03 98 _._ 390 _._ 16 99 _._ 850 _._ 01
PNDM 99 _._ 980 _._ 01 96 _._ 170 _._ 23 99 _._ 400 _._ 02 99 _._ 150 _._ 03 98 _._ 410 _._ 15 99 _._ 840 _._ 01
DPM-Solver++ 99 _._ 980 _._ 01 96 _._ 020 _._ 26 99 _._ 440 _._ 01 99 _._ 210 _._ 03 98 _._ 400 _._ 15 99 _._ 850 _._ 01


Table 5: Ablation results of extraction accuracy on sampling timesteps.


Inversion Timesteps
Generation Timesteps

10 20 30 40 50


10 99 _._ 850 _._ 88 99 _._ 920 _._ 71 99 _._ 950 _._ 57 99 _._ 950 _._ 58 99 _._ 950 _._ 60
20 99 _._ 960 _._ 52 99 _._ 970 _._ 48 99 _._ 980 _._ 35 99 _._ 980 _._ 36 99 _._ 980 _._ 38
30 99 _._ 970 _._ 28 99 _._ 990 _._ 18 99 _._ 990 _._ 16 99 _._ 990 _._ 12 99 _._ 990 _._ 12
40 99 _._ 970 _._ 56 99 _._ 980 _._ 52 99 _._ 980 _._ 48 99 _._ 980 _._ 47 99 _._ 980 _._ 48
50 99 _._ 970 _._ 36 99 _._ 980 _._ 36 99 _._ 980 _._ 29 99 _._ 990 _._ 29 99 _._ 990 _._ 26


inversion timesteps under PNG storage, as summarized in Table 5. Results show that neither the
choice of ODE solver nor the variation in timestep schedules introduces meaningful degradation. The
minor numerical discrepancies caused by switching solvers or adjusting timesteps are effectively
absorbed by the inherent redundancy of our spherical mapping, which provides robustness against
moderate inversion inaccuracies. Further quantitative analysis is provided in Appendix F.5.


5 DISCUSSION AND LIMITATIONS


Our Gaussian-noise guarantee depends on spherical 3-design definition. While watermarked and
random noise are empirically indistinguishable, higher-order moments may deviate from the true
prior. Extremely strong inversion-breaking attacks (e.g., perturbations targeting the VAE encoder
or ODE solver) can still compromise recovery. We provide an extended analysis of our method
against re-generation and editing attacks in Appendix F.2, showing that the proposed approach retains
robustness in these scenarios. Nevertheless, our primary focus is on tracing the origin of maliciously
generated content. Since editing and forgery may involve different adversarial, such cases are outside
our scope. Overall, our method can generalize to any generative model with a Gaussian prior and
invertible mappings, with detailed analysis in Appendix F.1.


6 CONCLUSION


In this paper, we introduce Spherical Watermark, a novel watermarking framework for image
generation that requires no modifications to the diffusion model. Our key innovation is the binary
embedding and spherical mapping module that converts binary watermark bits into Gaussian noise
input. Watermarked latent inputs are provably and empirically indistinguishable from a standard
Gaussian prior. Additionally, we eliminate per-image key management while delivering superior
robustness under realistic distortions. Extensive experiments demonstrate that our method outperforms
existing schemes in terms of undetectability, traceability, and computational efficiency.


10


ACKNOWLEDGMENTS


We sincerely thank the anonymous reviewers and area chairs for their constructive comments and
suggestions. This work was supported by the National Natural Science Foundation of China (Grant
62572125) and the Natural Science Foundation of Shanghai (Grant 25ZR1401019).


ETHICS STATEMENT


We have carefully reviewed and adhered to the ICLR Code of Ethics throughout the development
of this work. We have ensured that our methodologies and findings are transparent, reproducible,
and free from discrimination, bias, or unfairness concerns. Any potential ethical concerns have been
carefully considered, and we encourage responsible use of our contributions in future research.


REPRODUCIBILITY STATEMENT


We are committed to ensuring the reproducibility of our results. All theoretical claims are supported
with complete proofs provided in Appendix C, Appendix D, and Appendix E. For empirical studies,
we specify datasets, model architectures, hyperparameters, and implementation details in Section 4,
with additional information in Appendix F. The source code is included in the supplementary material,
with a README file that clearly documents the execution steps. All experiments are repeated five
times, and we report the mean and standard deviation to mitigate randomness and measurement error.


REFERENCES


Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the
stylegan latent space? In _Proceedings of the IEEE/CVF international conference on computer_
_vision_, pp. 4432–4441, 2019.


Ali Al-Haj. Combined dwt-dct digital image watermarking. _Journal_ _of_ _computer_ _science_, 3(9):
740–746, 2007.


Kasra Arabi, Benjamin Feuer, R. Teal Witter, Chinmay Hegde, and Niv Cohen. Hidden in the
noise: Two-stage robust watermarking for images. In _The Thirteenth International Conference_
_on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025_ . OpenReview.net, 2025a.
[URL https://openreview.net/forum?id=ll2nz6qwRG.](https://openreview.net/forum?id=ll2nz6qwRG)


Kasra Arabi, R Teal Witter, Chinmay Hegde, and Niv Cohen. Seal: Semantic aware image watermarking. _arXiv preprint arXiv:2503.12172_, 2025b.


Bela Bajnok. Construction of spherical t-designs. _Geometriae Dedicata_, 43:167–179, 1992.


Eiichi Bannai. On tight spherical designs. _Journal of Combinatorial Theory, Series A_, 26(1):38–47,
1979.


Eiichi Bannai and Etsuko Bannai. A survey on spherical designs and algebraic combinatorics on
spheres. _European Journal of Combinatorics_, 30(6):1392–1425, 2009.


Walter R Bender, Daniel Gruhl, and Norishige Morimoto. Techniques for data hiding. In _Storage_
_and Retrieval for Image and Video Databases III_, volume 2420, pp. 164–173. SPIE, 1995.


Joseph R Biden. Executive order on the safe, secure, and trustworthy development and use of artificial
intelligence. 2023.


BlackForestLabs. Flux.1-dev. [https://blackforestlabs.ai/, 2024.](https://blackforestlabs.ai/)


Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image
editing instructions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern_
_recognition_, pp. 18392–18402, 2023.


11


Tu Bui, Shruti Agarwal, and John Collomosse. Trustmark: Robust watermarking and watermark
removal for arbitrary resolution images. In _Proceedings of the IEEE/CVF International Conference_
_on Computer Vision_, pp. 18629–18639, 2025.


Louis HY Chen and Qi-Man Shao. Normal approximation under local dependence. 2004.


Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. In _Annual_ _International_
_Cryptology Conference_, pp. 325–347. Springer, 2024.


Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. Ringid: Rethinking tree-ring watermarking
for enhanced multi-key identification. In _European Conference on Computer Vision_, pp. 338–354.
Springer, 2024.


Thomas M Cover. _Elements of information theory_ . John Wiley & Sons, 1999.


Scott A Craver, Nasir D Memon, Boon-Lock Yeo, and Minerva M Yeung. Can invisible watermarks
resolve rightful ownerships? In _Storage and Retrieval for Image and Video Databases V_, volume
3022, pp. 310–321. SPIE, 1997.


Ph Delsarte, JM Goethals, and JJ Seidel. Spherical codes and designs. _Geometriae Dedicata_, 6(3):
363–388, 1977.


Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale
hierarchical image database. In _2009 IEEE Computer Society Conference on Computer Vision and_
_Pattern Recognition (CVPR 2009), 20-25 June 2009, Miami, Florida, USA_, pp. 248–255. IEEE
Computer Society, 2009. doi: 10.1109/CVPR.2009.5206848. [URL https://doi.org/10.](https://doi.org/10.1109/CVPR.2009.5206848)
[1109/CVPR.2009.5206848.](https://doi.org/10.1109/CVPR.2009.5206848)


Prafulla Dhariwal and Alexander Quinn Nichol. Diffusion models beat gans on image synthesis. In
Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman
Vaughan (eds.), _Advances in Neural Information Processing Systems 34:_ _Annual Conference on_
_Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual_, pp.
8780–8794, 2021. URL [https://proceedings.neurips.cc/paper/2021/hash/](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)
[49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html.](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)


Mucong Ding, Tahseen Rabbani, Bang An, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng,
Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, et al. Waves: Benchmarking the
robustness of image watermarks. In _ICLR 2024 Workshop on Reliable and Responsible Foundation_
_Models_, 2024.


Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components
estimation. _arXiv preprint arXiv:1410.8516_, 2014.


Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit,
and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale.
In _9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria,_
_May_ _3-7,_ _2021_ . OpenReview.net, 2021. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=YicbFdNTTy)
[YicbFdNTTy.](https://openreview.net/forum?id=YicbFdNTTy)


Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam
Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for
high-resolution image synthesis. In _Forty-first international conference on machine learning_, 2024.


Pierre Fernandez, Guillaume Couairon, Hervé Jégou, Matthijs Douze, and Teddy Furon. The stable
signature: Rooting watermarks in latent diffusion models. In _Proceedings_ _of_ _the_ _IEEE/CVF_
_International Conference on Computer Vision_, pp. 22466–22477, 2023.


Alexei Grinbaum and Laurynas Adomaitis. The ethical need for watermarks in machine-generated
language. _AI and Ethics_, 2022.


12


Sam Gunn, Xuandong Zhao, and Dawn Song. An undetectable watermark for generative image
models. In _The Thirteenth International Conference on Learning Representations, ICLR 2025,_
_Singapore,_ _April 24-28,_ _2025_ . OpenReview.net, 2025. URL [https://openreview.net/](https://openreview.net/forum?id=jlhBFm7T2J)
[forum?id=jlhBFm7T2J.](https://openreview.net/forum?id=jlhBFm7T2J)


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image
recognition. In _Proceedings of the IEEE conference on computer vision and pattern recognition_,
pp. 770–778, 2016.


Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans
trained by a two time-scale update rule converge to a local nash equilibrium. _Advances in neural_
_information processing systems_, 30, 2017.


Runyi Hu, Jie Zhang, Ting Xu, Jiwei Li, and Tianwei Zhang. Robust-wide: Robust watermarking
against instruction-driven image editing. In _European Conference on Computer Vision_, pp. 20–37.
Springer, 2024.


Zhengyuan Jiang, Jinghuai Zhang, and Neil Zhenqiang Gong. Evading watermark based detection of
ai-generated content. In Weizhi Meng, Christian Damsgaard Jensen, Cas Cremers, and Engin Kirda
(eds.), _Proceedings_ _of_ _the_ _2023_ _ACM_ _SIGSAC_ _Conference_ _on_ _Computer_ _and_ _Communications_
_Security, CCS 2023, Copenhagen, Denmark, November 26-30, 2023_, pp. 1168–1181. ACM, 2023.
doi: 10.1145/3576915.3623189. [URL https://doi.org/10.1145/3576915.3623189.](https://doi.org/10.1145/3576915.3623189)


Nikita Kashyap and GR Sinha. Image watermarking using 3-level discrete wavelet transform (dwt).
_International Journal of Modern Education and Computer Science_, 4(3):50, 2012.


Changhoon Kim, Kyle Min, Maitreya Patel, Sheng Cheng, and Yezhou Yang. Wouaf: Weight
modulation for user attribution and fingerprinting in text-to-image diffusion models. In _Proceedings_
_of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 8974–8983, 2024.


Diederik P. Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions.
In Samy Bengio, Hanna M. Wallach, Hugo Larochelle, Kristen Grauman, Nicolò Cesa-Bianchi, and
Roman Garnett (eds.), _Advances in Neural Information Processing Systems 31:_ _Annual Conference_
_on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal,_
_Canada_, pp. 10236–10245, 2018. [URL https://proceedings.neurips.cc/paper/](https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)
[2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html.](https://proceedings.neurips.cc/paper/2018/hash/d139db6a236200b21cc7f752979132d0-Abstract.html)


Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In Yoshua Bengio and
Yann LeCun (eds.), _2nd_ _International_ _Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2014,_
_Banff,_ _AB,_ _Canada,_ _April_ _14-16,_ _2014,_ _Conference_ _Track_ _Proceedings_, 2014. URL [http:](http://arxiv.org/abs/1312.6114)
[//arxiv.org/abs/1312.6114.](http://arxiv.org/abs/1312.6114)


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.


Xiaolong Li, Bin Yang, Daofang Cheng, and Tieyong Zeng. A generalization of lsb matching. _IEEE_
_signal processing letters_, 16(2):69–72, 2009.


Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In _Computer vision–_
_ECCV 2014:_ _13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings,_
_part v 13_, pp. 740–755. Springer, 2014.


Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models on
manifolds. _arXiv preprint arXiv:2202.09778_, 2022.


Shuai Liu, Zheng Pan, and Houbing Song. Digital image watermarking method based on dct and
fractal encoding. _IET image processing_, 11(10):815–821, 2017.


Giacomo Livan, Marcel Novaes, and Pierpaolo Vivo. Introduction to random matrices theory and
practice. _Monograph Award_, 63(54):914, 2018.


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast
ode solver for diffusion probabilistic model sampling in around 10 steps. _Advances in Neural_
_Information Processing Systems_, 35:5775–5787, 2022.


13


Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver++: Fast
solver for guided sampling of diffusion probabilistic models. _Machine Intelligence Research_, pp.
1–22, 2025.


Shilin Lu, Zihan Zhou, Jiayou Lu, Yuanzhi Zhu, and Adams Wai-Kin Kong. Robust watermarking
using generative priors against image editing: From benchmarking to advances. _arXiv preprint_
_arXiv:2410.18775_, 2024.


Yuqing Ma, Xianglong Liu, Shihao Bai, Lei Wang, Aishan Liu, Dacheng Tao, and Edwin R Hancock.
Regionwise generative adversarial image inpainting for large missing areas. _IEEE transactions on_
_cybernetics_, 53(8):5226–5239, 2022.


Francesco Mezzadri. How to generate random matrices from the classical compact groups. _arXiv_
_preprint math-ph/0609050_, 2006.


Robb J Muirhead. _Aspects of multivariate statistical theory_ . John Wiley & Sons, 2009.


KA Navas, Mathews Cheriyan Ajay, M Lekshmi, Tampy S Archana, and M Sasikumar. Dwt-dct-svd
based watermarking. In _2008 3rd international conference on communication systems software_
_and middleware and workshops (COMSWARE’08)_, pp. 271–274. IEEE, 2008.


Judea Pearl. _Probabilistic reasoning in intelligent systems:_ _networks of plausible inference_ . Elsevier,
2014.


John G Proakis and Masoud Salehi. _Digital communications_ . McGraw-hill, 2008.


Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF confer-_
_ence on computer vision and pattern recognition_, pp. 10684–10695, 2022.


Nathan Ross. Fundamentals of stein’s method. 2011.


David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by
back-propagating errors. _nature_, 323(6088):533–536, 1986.


Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu,
Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language
models. _Advances in Neural Information Processing Systems_, 37:130136–130184, 2024.


Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen.
Improved techniques for training gans. _Advances in neural information processing systems_, 29,
2016.


Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In _9th Interna-_
_tional Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ .
OpenReview.net, 2021a. [URL https://openreview.net/forum?id=St1giarCHLP.](https://openreview.net/forum?id=St1giarCHLP)


Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben
Poole. Score-based generative modeling through stochastic differential equations. In _9th Interna-_
_tional Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021_ .
OpenReview.net, 2021b. [URL https://openreview.net/forum?id=PxTIG12RRHS.](https://openreview.net/forum?id=PxTIG12RRHS)


Matthew Tancik, Ben Mildenhall, and Ren Ng. Stegastamp: Invisible hyperlinks in physical photographs. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_,
pp. 2117–2126, 2020.


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. _Advances in neural information processing_
_systems_, 30, 2017.


Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li,
and Ying Shan. Taming rectified flow for inversion and editing. _arXiv preprint arXiv:2411.04746_,
2024.


14


Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from
error visibility to structural similarity. _IEEE transactions on image processing_, 13(4):600–612,
2004.


Zilan Wang, Junfeng Guo, Jiacheng Zhu, Yiming Li, Heng Huang, Muhao Chen, and Zhengzhong
Tu. Sleepermark: Towards robust watermark against fine-tuning text-to-image diffusion models.
In _Proceedings of the Computer Vision and Pattern Recognition Conference_, pp. 8213–8224, 2025.


T Wei, R Qiu, Y Chen, Y Qi, J Lin, W Xu, S Nag, R Li, H Lu, Z Wang, et al. Robust watermarking for diffusion models: A unified multi-dimensional recipe, 2024b. _URL_
_https://openreview.net/forum?id=O13fIFEB81_, 2024.


Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. Tree-ring watermarks: Fingerprints for diffusion images that are invisible and robust. 2023.


Kyle Wiggers. Microsoft pledges to watermark ai-generated images and videos. _TechCrunch blog_,
2023.


Wikipedia contributors. N-sphere — Wikipedia, the free encyclopedia, 2025. [URL https://en.](https://en.wikipedia.org/wiki/N-sphere#Distribution_of_the_first_coordinate)
[wikipedia.org/wiki/N-sphere#Distribution_of_the_first_coordinate.](https://en.wikipedia.org/wiki/N-sphere#Distribution_of_the_first_coordinate)


Cheng Xiong, Chuan Qin, Guorui Feng, and Xinpeng Zhang. Flexible and secure watermarking for
latent diffusion model. In _Proceedings of the 31st ACM International Conference on Multimedia_,
pp. 1668–1676, 2023.


Zijin Yang, Kai Zeng, Kejiang Chen, Han Fang, Weiming Zhang, and Nenghai Yu. Gaussian shading:
Provable performance-lossless image watermarking for diffusion models. In _Proceedings of the_
_IEEE/CVF Conference on Computer Vision and Pattern Recognition_, pp. 12162–12171, 2024.


Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated
dataset for instruction-guided image editing. _Advances in Neural Information Processing Systems_,
36:31428–31449, 2023.


Kevin Alex Zhang, Lei Xu, Alfredo Cuesta-Infante, and Kalyan Veeramachaneni. Robust invisible
video watermarking with attention. _CoRR_, abs/1909.01285, 2019. [URL http://arxiv.org/](http://arxiv.org/abs/1909.01285)
[abs/1909.01285.](http://arxiv.org/abs/1909.01285)


Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia
Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at
scale. _Advances in Neural Information Processing Systems_, 37:3058–3093, 2024.


Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. Hidden: Hiding data with deep networks.
In _Proceedings of the European conference on computer vision (ECCV)_, pp. 657–672, 2018.


APPENDIX


In the supplement, Appendix A provides a statement on the use of large language models (LLMs)
during the preparation. Appendix B presents the technical details of baselines. Appendix C provides
the theoretical proofs for the theorem and lemmas introduced in Section 3.3 of the main paper.
Appendix D examines the rotation operation within the proposed spherical mapping module, demonstrating its contribution to robustness and showing that it constitutes a near-optimal solution under
some lossy channels. Appendix E provides our theoretical analysis of the importance of losslessness,
showing that it plays a crucial role in resisting adversarial attacks. Finally, Appendix F supplies
additional experimental details and results that further validate the effectiveness of our method.


A USE OF LARGE LANGUAGE MODELS


We used large language models (LLMs) to assist with language polishing and improving the clarity of
exposition. All technical content, including theoretical claims, proofs, algorithms, and experiments,
was developed entirely by the authors.


15


Original DwtDct DwtSvd. RivaGAN TR. GS. PRC. Ours


Figure 7: Examples of watermarked images. Generated images from top to bottom: COCO-SD v1.5,
COCO-SD v2.1, SDP-SD v1.5, SDP-SD v2.1. Here, “DwtSvd.” refers to “DwtDctSvd”, “TR.” to
“Tree-Ring”, “GS.” to “Gaussian Shading”, and “PRC.” to “PRC Watermark”.


B OVERVIEW OF BASELINE METHODS


This section systematically reviews and analyzes several previously proposed watermarking schemes
evaluated in the experiments. These methods range from traditional signal processing techniques
to state-of-the-art latent-based schemes, reflecting the evolution and diversity of watermarking
methodologies. We present several examples of watermarked images in Figure 7.


**DwtDct.** DwtDct (Al-Haj, 2007) is a hybrid watermarking technique that integrates Discrete Wavelet
Transform (DWT) and Discrete Cosine Transform (DCT). It performs multi-level wavelet decomposition on the carrier image, extracts the low-frequency sub-band to find stable regions, and then applies
block-wise DCT to embed the watermark into mid-frequency coefficients. The method balances
quality and robustness using DWT for structure retention and DCT for compression resistance.


**DwtDctSvd.** DwtDctSvd (Navas et al., 2008) is an improved watermarking scheme based on DwtDct,
enhanced by incorporating Singular Value Decomposition (SVD) to boost security and stability. The
method performs DWT on the carrier image, selects specific sub-bands for block-wise DCT, and
then performs SVD on each block to embed the watermark into the singular value matrix. It resists
geometric transformations and signal processing attacks.


**RivaGAN.** RivaGAN (Zhang et al., 2019) is a watermarking framework based on Generative Adversarial Networks (GANs), using an encoder–decoder architecture for adaptive watermark embedding
and extraction. The generator embeds the watermark covertly, while the discriminator and extraction
network are trained together to enhance attack resistance. Adversarial training enables adaptability to
diverse attacks, ensuring strong robustness.


**Tree-Ring.** The Tree-Ring (Wen et al., 2023) watermarking scheme embeds a specific ring-shaped
pattern, referred to as a key into the Fourier space of the initial noise vector in the diffusion model.
This subtle modification influences the entire image generation process during sampling, in a way
that is imperceptible to humans. By leveraging the properties of the Fourier domain, the watermark
exhibits invariance to common attacks.


**Gaussian Shading.** Gaussian Shading (Yang et al., 2024) is a provably secure watermarking scheme.
This method utilizes repetition codes and stream encryption to transform the watermark information
**m** into a binary bit sequence **s** _∈{_ 0 _,_ 1 _}_ _[n]_ that follows a uniformly random distribution, where each
bit _si_ _∼_ Bernoulli� 12 - _, i ∈_ (0 _,_ 1 _, ..., n_ ). Then, through a truncated sampling scheme, the sequence
**s** is mapped into a latent representation **x** that conforms to a standard Gaussian distribution. This


16


process can be formally expressed as:


_xi_ = (2 _si −_ 1) _· |z|,_ _z_ _∈N_ (0 _,_ 1) _._ (14)


Here, _z_ is randomly sampled in each invocation, following a standard one-dimensional Gaussian
distribution. However, it requires storing a unique key for each image to preserve the original
distribution, which introduces substantial management overhead. Moreover, the truncated sampling
scheme is suboptimal and fails to reach the theoretical upper bound of error correction, as rigorously
analyzed in Appendix D.


**PRC** **Watermark.** The PRC Watermark (Gunn et al., 2025) embeds information into the latent
representation of diffusion models by using pseudorandom error-correcting codes (PRC) (Christ
& Gunn, 2024) instead of stream cipher techniques. While their approach eliminates the need to
store a separate key for each image, it introduces heavyweight cryptographic complexity. The use of
encoding and belief-propagation decoding (Pearl, 2014) incurs substantial computational cost and
latency during watermark embedding and extraction. Moreover, their inherent uncertainty leads to
them being prone to hitting a performance ceiling, failing to recover the watermark under certain
lossy conditions.


C PROOFS OF THEOREMS AND LEMMAS IN SECTION 3.3


**Lemma 3.1.** _If_ **m** _and_ **r** _consist of independent_ Bernoulli( [1] 2 [)] _[ bits, then for]_ **[ z]** [(1)] _[ in Eq.]_ _[11, we have]_

_zi_ [(1)] _∼_ Bernoulli� 12 - _for every i ∈{_ 1 _, . . ., lx}, and_ **z** [(1)] _is both 2-wise and 3-wise independent._


_Proof._ In F2, linear independence of Boolean linear forms is equivalent to their statistical independence. We form the combined variable

**e** =           - _m_ 1 _,_ _m_ 2 _,_ _. . .,_ _mlm,_ _r_ 1 _,_ _r_ 2 _,_ _. . .,_ _rlr_           - _⊤,_ (15)


which can be viewed as the concatenation of **m** and **r**, with all entries mutually independent. We
then construct the binary matrix







**R** 









**Q** =















**I** _lm_
**I** _lm_
...
**I** _lm_

����
_N_ copies


_∈{_ 0 _,_ 1 _}_ [(] _[N]_ _[×][l][m]_ [+] _[l][r]_ [)] _[×]_ [(] _[l][m]_ [+] _[l][r]_ [)] _,_ (16)


**0** _lr×lm_ **I** _lr_


so that the encoding **z** [(1)] = **T x** of F2 in Eq. 11 of the main paper can be compactly re-written as
**z** [(1)] = **Q e** _._ Since each row of **Q** is a nonzero vector in F2 _[l][m]_ [+] _[l][r]_, we can write


_zi_ [(1)] = _⟨_ **q** _i,_ **e** _⟩_ F2 =


_lm_ + _lr_

- _qi,j ej, i ∈{_ 1 _, . . ., lx},_ (17)


_j_ =1


where **q** _i_ is the _i_ th row of **Q**, _qi,j_ represents its _j_ th entry. _⟨·⟩_ F2 denotes the binary inner product, _ej_ is
the _j_ th entry of **e**, and refers to the bitwise XOR. Because the bits _ej_ are independent and each

[�]
satisfies Pr[ _ej_ = 1] = [1] 2 [, any nontrivial XOR of them remains unbiased, hence]


Pr� _zi_ [(1)] = 1� = Pr� _zi_ [(1)] = 0� = 21 _[,]_ (18)


i.e. _zi_ [(1)] _∼_ Bernoulli( [1] 2 [)][ for all] _[ i]_ [.]


In this setting, verifying independence of **z** [(1)] reduces to a simple rank condition on **Q** . Specifically:


**z** [(1)] _is k-wise independent if and only if every k-row submatrix of_ **Q** _has full row_
_rank._


17


Equivalently, for any choice of indices _i_ 1 _, . . ., ik_,


                   -                    rankF2 **Q** _{i_ 1 _,...,ik}_ = _k,_ (19)


which ensures that the corresponding outputs _zi_ 1 _, . . ., zik_ are linearly independent, and hence statistically independent. By Algorithm 1 of the main paper, any two distinct rows of **Q** are not identical.
Consequently, every 2-row submatrix of **Q** has full row rank. It follows that the corresponding output
bits are linearly independent, and hence **z** [(1)] is 2-wise independent. Assume, for the sake of contradiction, that **z** [(1)] is not 3-wise independent. Then there exist distinct indices _i_ 1 _, i_ 2 _, i_ 3 ( _i_ 1 _< i_ 2 _< i_ 3)
and coefficients _a, b, c ∈_ F2, not all zero, such that


_a zi_ [(1)] 1 _⊕_ _b zi_ [(1)] 2 _⊕_ _c zi_ [(1)] 3 = 0 _._ (20)


However, since **z** [(1)] is already 2-wise independent, no nontrivial relation can hold among any two
bits. For example, if _c_ = 0, then we have _a zi_ [(1)] 1 _[⊕]_ _[b z]_ _i_ [(1)] 2 [= 0][ with][ (] _[a, b]_ [)] _[ ̸]_ [= (0] _[,]_ [ 0)][, which contradicts]
2-wise independence. Therefore, in the above three-term combination, _a_, _b_ and _c_ must be 1; otherwise
we would contradict 2-wise independence. Hence,


_zi_ [(1)] 1 _[⊕]_ _[z]_ _i_ [(1)] 2 _[⊕]_ _[z]_ _i_ [(1)] 3 [= 0] [=] _[⇒]_ **[q]** _[i]_ 1 _[⊕]_ **[q]** _[i]_ 2 _[⊕]_ **[q]** _[i]_ 3 [=] **[ 0]** _[.]_ (21)


The leftmost _lm_ columns of **Q** can sum to zero only in two cases:


_•_ All three indices lie in the bottom block: _i_ 1 _,_ _i_ 2 _,_ _i_ 3 _> N_ _× lm._ Then in the last _lr_ columns
their rows sum to a nonzero vector, contradicting **q** _i_ 1 _⊕_ **q** _i_ 2 _⊕_ **q** _i_ 3 = **0** .


_•_ Two indices lie in the top block but align on repeated identity rows: _i_ 1 _,_ _i_ 2 _≤_ _N_ _× lm_,
_| i_ 1 _−_ _i_ 2 _|_ mod _lm_ = 0 and _i_ 3 _> N_ _× lm_ . In the **R** block constructed by Algorithm 1 of the
main paper, the positions of the 1–entries in rows _i_ 1 and _i_ 2 are guaranteed to be disjoint, so
their sum over the last _lr_ columns is nonzero, again a contradiction.


In either case we reach a contradiction. Therefore no nontrivial three-term relation exists, and **z** [(1)] is
3-wise independent.


**Definition 3.1** (Spherical _t_ -Design) **.** _A finite set of points X_ = _{x_ 1 _, . . ., xN_ _} ⊂_ _S_ _[n][−]_ [1] _on the unit_
_sphere in_ R _[n]_ _is called a_ spherical _t_ -design _if, for every real polynomial f_ _of total degree at most t,_


(22)
_S_ _[n][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[,]_


1

_N_


- _f_ ( _x_ ) = 1

_|S_ _[n][−]_ [1] _|_
_x∈X_


_where dσ is the uniform surface measure on S_ _[n][−]_ [1] _, and |S_ _[n][−]_ [1] _| denotes the total surface area of the_
_unit_ ( _n −_ 1) _-sphere._


Equivalently, _X_ is a _t_ -design if and only if it _matches all moments_ of the uniform distribution on the
sphere up to degree _t_ . In the following theorem, we show that the finite point set sampled from **z** [(2)]
constitutes a spherical 3-design.

**Theorem** **3.2.** **z** [(2)] _satisfies_ _that_ _each_ _zi_ [(2)] _takes_ _values_ _±_ ~~_√_~~ 1 _lx_ _with_ Pr[ _zi_ = + ~~_√_~~ 1 _lx_ ] = Pr[ _zi_ =

_−_ ~~_√_~~ 1 _lx_ ] = [1] 2 _[,][ i][ ∈]_ [(1] _[,][ · · ·]_ _[, l][x]_ [)] _[;]_ **[ z]** [(2)] _[is 2-wise and 3-wise independent.]_ _[Then the finite set of]_ **[ z]** [(2)] _[on the]_

_unit sphere S_ _[l][x][−]_ [1] _is a spherical 3–design._


_Proof._ **z** [(1)] _∈{_ 0 _,_ 1 _}_ _[l][x]_ is 2-wise and 3-wise independent with Pr[ _zi_ [(1)] = 1] = Pr[ _zi_ [(1)] = 0] = 2 [1]


_Proof._ **z** _∈{_ 0 _,_ 1 _}_ _[x]_ is 2-wise and 3-wise independent with Pr[ _zi_ = 1] = Pr[ _zi_ = 0] = 2 [,]

_i, ∈{_ 1 _, . . ., lx}_ . Define


**v**
**v** = 2 **z** [(1)] _−_ **1** _,_ **z** [(2)] = _._ (23)
_∥_ **v** _∥_ 2


Since _zi_ [(1)] _∼_ Bernoulli( [1] 2


[1] 2 [)][, we have] _[ v][i]_ [= 2] _[z]_ _i_ [(1)] _−_ 1 _∈{±_ 1 _}_ with Pr[ _vi_ = 1] = Pr[ _vi_ = _−_ 1] = [1] 2


Since _zi_ _∼_ Bernoulli( 2 [)][, we have] _[ v][i]_ [= 2] _[z]_ _i_ _−_ 1 _∈{±_ 1 _}_ with Pr[ _vi_ = 1] = Pr[ _vi_ = _−_ 1] = 2 [.]

Because _∥_ **v** _∥_ 2 = _[√]_ _lx_, it follows that


_zi_ [(2)] = ~~_√_~~ _vlix_ _∈_ - _±_ ~~_√_~~ 1 _lx_


18


_,_ (24)


[1] 2 [.] [Note] [that] [the] [transformation] [from] **[z]** [(1)] [to] **[z]** [(2)]


and Pr[ _zi_ [(2)] = + ~~_√_~~ 1 _lx_ ] = Pr[ _zi_ [(2)] = _−_ ~~_√_~~ 1 _lx_ ] = [1] 2


consists solely of a affine shift followed by a normalization. Neither step mixes entries across
different coordinates: each _zi_ [(2)] depends only on the corresponding _zi_ [(1)] . Moreover, both operations
are invertible. Hence any independence property of **z** [(1)] is preserved exactly in **z** [(2)], and no new
dependencies are introduced. We can derive that **z** [(2)] is also 2-wise and 3-wise independent.


By the polynomial-averages characterization of spherical designs (Delsarte–Goethals–Seidel
1977 (Delsarte et al., 1977), Theorem 5.3; Bannai & Bannai 2009 (Bannai & Bannai, 2009), Theorem
2.2), it suffices to check that **z** [(2)] matches the uniform sphere’s moments for every monomial of total
degree _≤_ 3. Let **U** = ( _U_ 1 _, . . ., Ulx_ ) be a random vector drawn uniformly from the unit sphere _S_ _[l][x][−]_ [1] .
By symmetry,

E[ _Ui_ ] = 0 _,_ E[ _Ui_ [2][] =] [1] _,_ E[ _UiUj_ ] = 0 ( _i ̸_ = _j_ ) _,_ (25)

_lx_

and all mixed moments of total degree up to three that are odd in any coordinate equal zero. We list
all cases:


**Degree 1.**


[1] - ~~_√_~~ 1

2


 = 0 _,_ (26)
_lx_ [)]


E[ _zi_ [(2)] ] = [1] 2


~~_√_~~ 1
_lx_ [+ (] _[−]_ _l_


matching the sphere’s E[ _Ui_ ] = 0.


**Degree 2.**


E[( _zi_ [(2)] ) [2] ] = - ~~_√_~~ 1 _lx_


�2
= _l_ [1] _x_ _[,]_ E[ _zi_ [(2)] _zj_ [(2)][] =][ E][[] _[z]_ _i_ [(2)] ] E[ _zj_ [(2)][] = 0] ( _i ̸_ = _j_ ) _,_ (27)


which agrees with the uniform-sphere values E[ _Ui_ [2][] = 1] _[/l][x]_ [,][ E][[] _[U][i][U][j]_ [] = 0][.]


**Degree 3.**


E[( _zi_ [(2)] ) [3] ] = - ~~_√_~~ 1 _lx_


E[( _zi_ [(2)] ) [3] ] = - ~~_√_~~ 1


�3� 12 _[−]_ 2 [1] - = 0 _,_ (28)


E[ ( _zi_ [(2)] ) [2] _zj_ [(2)] ] = E[( _zi_ [(2)] ) [2] ] E[ _zj_ [(2)][] =] _l_ [1] _x_ _[·]_ [ 0 = 0] ( _i ̸_ = _j_ ) _,_ (29)


E[ _zi_ [(2)] _zj_ [(2)] _zk_ [(2)] ] = E[ _zi_ [(2)] ] E[ _zj_ [(2)][]][ E][[] _[z]_ _k_ [(2)][] = 0] (distinct _i, j, k_ ) _._ (30)

These all match the sphere’s vanishing of odd moments up to degree 3. Since every polynomial of
degree _≤_ 3 is a linear combination of these monomials, **z** [(2)] reproduces exactly the uniform-sphere
averages through degree 3. Hence, the finite set of **z** [(2)] is a spherical 3–design.


**Lemma** **3.3.** _Let_ **z** [(2)] _∈_ _S_ _[l][x][−]_ [1] _be_ _a_ _spherical_ 3 _–design._ _If_ _we_ _apply_ _a_ _fixed_ _orthogonal_ _rotation_
**z** [(3)] = **C z** [(2)] _, then_ **z** [(3)] _is also a spherical_ 3 _–design._ _For each coordinate zi_ [(3)] _, one has_ E[ _zi_ ] = 0
_and_ E[ _zi_ [2][] = 1] _[/l][x][, and as][ l][x]_ _[→∞][, the marginal law of][ z]_ _i_ [(3)] _converges to N_ (0 _,_ 1 _/lx_ ) _._


_Proof._ Since the point set
_X_ = _{_ **z** [(2)] _} ⊂_ _S_ _[l][x][−]_ [1] (31)


is a spherical 3–design, it integrates exactly all polynomials of total degree _≤_ 3. Hence for any
polynomial _f_ : R _[l][x]_ _→_ R with deg _f_ _≤_ 3,


1
_|X|_


- _f_ - **z** [(2)][�] = 1

_|S_ _[l][x][−]_ [1] _|_
**z** [(2)] _∈X_


(32)
_S_ _[lx][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[.]_


Define the rotated set
_X_ _′_ = _{_ **z** (3) : **z** (3) = **C z** (2) _,_ **z** (2) _∈_ _X}._ (33)


Since orthogonal rotations preserve the spherical surface (Haar) measure,

          -           


         
   - **C** _x_    - _dσ_ ( _x_ ) =
_S_ _[lx][−]_ [1] _[ f]_


(34)
_S_ _[lx][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[.]_


19


Because _f_ ( **C** _x_ ) is still a polynomial of degree _≤_ 3, the design property of _{_ **z** [(2)] _}_ gives
_|X_ 1 _′|_   - _f_   - **z** [(3)][�] = _|X_ 1 _|_   - _f_   - **C z** [(2)][�]


 - _f_ - **z** [(3)][�] = 1

_|X|_
**z** [(3)] _∈X_ _[′]_


- _f_ - **C z** [(2)][�]


**z** [(2)] _∈X_


(35)


1
=
_|S_ _[l][x][−]_ [1] _|_


   - **C** _x_    - _dσ_ ( _x_ ) = 1
_S_ _[lx][−]_ [1] _[ f]_ _|S_ _[l][x][−]_ [1] _|_


_S_ _[lx][−]_ [1] _[ f]_ [(] _[x]_ [)] _[ dσ]_ [(] _[x]_ [)] _[.]_


_′_
Therefore the rotated set _X_ also constitutes a spherical 3–design.


Recall that from the spherical 3–design property in the proof of Theorem 3.2 (or equivalently from the
degree–1 and degree–2 moment matching) we have for the original vector **z** [(2)] = ( _z_ 1 [(2)] _[, . . ., z]_ _l_ [(2)] _x_ [)] _[⊤]_ [:]

E� _zj_ [(2)]         - = 0 _,_ E� _zj_ [(2)] _zk_ [(2)]         - = _l_ [1] _x_ _δjk,_ _j, k_ = 1 _, . . ., lx._ (36)

the symbol _δjk_ denotes the _Kronecker delta_, which is defined by


�1 _,_ _j_ = _k,_
_δjk_ = (37)
0 _,_ _j_ = _k._


Let **z** [(3)] = **C z** [(2)] be the rotated vector, so that


_zi_ [(3)] =


_lx_

- _Cij zj_ [(2)] _[.]_ (38)

_j_ =1


Here, _Cij_ represents the _i_ th row, _j_ th column entry of **C** . By linearity of expectation and the above
moment identities,


E� _zi_ [(3)] - =


_lx_

- - 
_Cij_ E _zj_ [(2)] = 0 _,_ (39)
_j_ =1


and


E�( _zi_ [(3)] ) [2][�] = E�� _[l][x]_

_j_


_lx_

- _CijCik zj_ [(2)] _zk_ [(2)] 
_k_


_lx_


_,_
_lx_


(40)


_CijCik_ E[ _zj_ [(2)] _zk_ [(2)][] =] [1]

_lx_

_k_


_lx_


=


_lx_


_j_


_lx_


_lx_


_Cij_ [2] [=] [1]

_l_

_j_ =1


The classical central limit theorem for independent summands is not directly applicable here since
**z** [(] **[2]** [)] may exhibit local dependence. Instead, we construct the dependency graph according to Stein’s
method (Ross, 2011; Chen & Shao, 2004). For the rotated item _zi_ [(3)] derived by Eq. 38, we build a
dependency graph with _lx_ vertices, where vertex _j_ corresponds to _Cij zj_ [(2)][.] [Following the generation]
procedure, each entry of **z** [(2)] is given by

_[−]_ [1]
_zj_ [(2)] = [2] _[⟨]_ **[q]** _[j][,]_ ~~_√_~~ **[ e]** _[⟩]_ _lx_ [F][2] _,_ (41)

where **q** _j_ denotes the _j_ -th row of the binary matrix **Q** (see Eq. 16), and **e** is defined by Eq. 15. We
connect two distinct vertices _j_ 1 = _j_ 2 by an edge if and only if **q** _j_ 1 and **q** _j_ 2 share at least one common
index at which both entries equal 1, i.e., supp( **q** _j_ 1) _∩_ supp( **q** _j_ 2) _̸_ = ∅, where supp( **q** _j_ ) := _{k|qj,k_ =
1 _}_ denotes the support set of indices. Equivalently, an edge exists when _zj_ [(2)] 1 [and] _[ z]_ _j_ [(2)] 2 [share at least]

one underlying source bit from **m** or **r** . Moreover, E[( _zi_ [(3)] ) [2] ] = 1 _/lx_, and we can derive a standard
normal-approximation bound for locally dependent sums. By Theorem 3.6 in (Ross, 2011), if the
degree of the dependency graph is bounded by the maximum degree _D_, then for the Wasserstein
distance _dW_ of _[√]_ _lxzi_ [(3)] and _N_ (0 _,_ 1), it satisfies,


_lx_

  - 3 [�]
_dW_ _≤_ _D_ [2] E���� _lx Cijzj_ [(2)] �� +

_j_ =1


28 �� _lx_

_π_ _[D]_ [3] _[/]_ [2]


- 28


_lx_

- E��� _lx Cijzj_ [(2)] �4 [��][1] _[/]_ [2]

_j_ =1


_._ (42)


_√_ (2)
Obviously, �� _lx zj_ �� = 1, so the right-hand side reduces to terms involving _|Cij|_ . Next, we verify
that our dependency graph satisfies the conditions above.


20


- Due to the constraint imposed by Algorithm 1, the maximum degree _D_ is uniquely determined by the embedding matrix **T** and satisfies _D_ _≤_ _N_ + _lm_ _−_ 1. Since _N_ and _lm_ are
preset constants, we can treat _D_ as a constant.


    - The entries _Cij_ come from the orthogonal matrix **C**, which is sampled from the Haar-uniform
measure according to (Mezzadri, 2006). For a typical Haar-random **C** and sufficiently
large _lx_, the row-wise empirical averages concentrate around their Haar expectations in
probability over the draw of **C** . For any fixed ( _i, j_ ), the squared entry _|Cij|_ [2] follows the Beta
distribution Beta� 12 _[,]_ _[l][x]_ 2 _[−]_ [1]     - (Wikipedia contributors, 2025; Livan et al., 2018). In particular,

the third and fourth absolute moments are,


E� _|Cij|_ [3][�] = ~~_√_~~ Γ� _l_ 2 _x_ 


~~_√_~~ _π_ 2Γ� _lx_ 2+42 - = Θ� _lx_ _[−]_ [2] - _,_


~~_√_~~ _π_ ΓΓ�� _l_ 2 _xlx_ �2+3 - = Θ� _lx_ _[−]_ [3] _[/]_ [2] - _,_ E� _|Cij|_ [4][�] = ~~_√_~~ Γ� _π_ 25Γ��Γ _lx_ �2+4 _l_ 2 _x_ 


(43)
where Γ( _·_ ) denotes the Gamma function, and Θ( _·_ ) is the standard asymptotic order notation.


Therefore, ignoring constant factors, the distributional discrepancy in Eq. 42 can be simplified as,


          -           -           -           -           -           _dW_ = _O_ _D_ [2] _lx_ _[−]_ [1] _[/]_ [2] + _O_ _D_ [3] _[/]_ [2] _lx_ _[−]_ [1] _[/]_ [2] = _O_ _lx_ _[−]_ [1] _[/]_ [2] _,_ (44)


where _O_ ( _·_ ) denotes the standard Big- _O_ notation. As _lx_ _→∞_, we have _dW_ _→_ 0, which implies that
the marginal distribution of _zi_ [(3)] converges to _N_ (0 _,_ 1 _/lx_ ).


This completes the proof.


**Lemma 3.4.** _Let_ **x** _∼N_ ( **0** _,_ **I** _n_ ) _be a standard multivariate normal vector in_ R _[n]_ _._ _Then_ **x** _admits a_
_polar decomposition of the form_
**x** = _r ·_ **u** _,_ (45)

_where r_ [2] _∼_ _χ_ [2] ( _n_ ) _, and_ **u** _is uniformly distributed on the unit sphere S_ _[n][−]_ [1] _._ _Furthermore, r and_ **u** _are_
_statistically independent._ _Conversely, if r_ [2] _∼_ _χ_ [2] ( _n_ ) _,_ **u** _is uniformly distributed on S_ _[n][−]_ [1] _, and r_ _⊥_ **u** _,_
_then the product_ **x** = _r ·_ **u** _follows a standard multivariate normal distribution, i.e.,_ **x** _∼N_ ( **0** _,_ **I** _n_ ) _._


_Proof._ In the sequel, we denote the Euclidean norm _∥·∥_ 2 simply by _∥·∥_ . We begin by demonstrating
that any random vector **x** drawn from a standard multivariate normal distribution can be decomposed
into a radial component _r_ and a directional component **u** . We refer to a classical result on spherical
distributions (see Theorem 1.5.6 in (Muirhead, 2009)), which states,


**Theorem.** If **x** has an _n_ -variate spherical distribution with Pr� **x** = **0**    - = 0, and
we define the radial part as _r_ = _∥_ **x** _∥_ = ( **x** _[⊤]_ **x** ) [1] _[/]_ [2] and the normalized direction as
_T_ ( **x** ) = _∥_ **X** _∥_ _[−]_ [1] **X**, then _T_ ( **X** ) is uniformly distributed on the unit sphere _S_ _[n][−]_ [1],
and _T_ ( **X** ) and _r_ are statistically independent.


Since the multivariate normal distribution is continuous, we have Pr� **x** = **0** - = 0. Moreover, the
standard normal distribution is rotationally invariant; that is, for any orthogonal matrix **C** _∈_ R _[n][×][n]_,
_d_ _d_
we have **Cx** = **x** (The notation = denotes equality in distribution). Therefore, **x** is spherically
distributed in R _[n]_, and it satisfies all the conditions required by the theorem. By applying the above
theorem, we decompose **x** into a radial part and a direction as follows:


**x**
_,_ **u** = _T_ ( **x** ) = (46)
_∥_ **x** _∥_ _[.]_


�1 _/_ 2


_r_ = _∥_ **x** _∥_ =


- _n_

 - _x_ [2] _i_


_i_ =1


Since each _xi_ _∼N_ (0 _,_ 1) and the components are independent, it follows that _r_ [2] _∼_ _χ_ [2] ( _n_ ). The
direction **u** is uniformly distributed over the unit sphere _S_ _[n][−]_ [1], and _r_ and **u** are independent by the
theorem. This establishes the sufficiency: any standard multivariate normal vector **x** _∼N_ ( **0** _,_ **I** _n_ )
can be decomposed into an independent product of a _χn_ -distributed magnitude and a uniformly
distributed direction.


21


Conversely, suppose we independently sample **u** _∼_ Unif( _S_ _[n][−]_ [1] ) and _r_ _∼_ _χn_, and define **z** = _r ·_ **u** .
Here Unif( _S_ _[n][−]_ [1] ) represents the uniform distribution one the unit sphere _S_ _[n][−]_ [1] . Then, using the
change-of-variables formula in polar coordinates, the probability density of **z** in R _[n]_ is given by:


_· ∥_ **z** _∥_ _[−]_ [(] _[n][−]_ [1)] _,_ (47)


_f_ **z** ( **z** ) = _fr_ ( _∥_ **z** _∥_ ) _· f_ **u**


- **z**
_∥_ **z** _∥_


where _fr_ ( _·_ ) and _f_ **u** ( _·_ ) are the densities of _χn_ and the uniform sphere distribution, respectively.
Substituting in the known densities and simplifying, we recover the standard multivariate normal
density:


1      _f_ **z** ( **z** ) = _−_ _[∥]_ **[z]** _[∥]_ [2]
(2 _π_ ) _[n/]_ [2] [exp] 2


_._ (48)


Therefore, **z** _∼N_ ( **0** _,_ **I** _n_ ).


D ROLE OF ROTATION IN ENHANCING ROBUSTNESS


In the ablation experiments of the main paper, we observed that replacing the spherical mapping module with the provably secure transformation introduced in Gaussian Shading results in a degradation
of robustness. In this section, we analyze the underlying reasons for this behavior. _Note:_ _To avoid_
_notational conflicts, this section is self-contained and uses its own symbols._


First, we compare the provably secure mapping used in Gaussian Shading with the spherical mapping
in our scheme. Let the watermark-bearing bit-string be denoted by the vector **s** _∈{_ 0 _,_ 1 _}_ _[n]_ . We first
convert **s** into a _{±_ 1 _}_ _[n]_ vector via
**s** _←_ 2 **s** _−_ **1** _._ (49)
In the Gaussian Shading pipeline, one then samples a Gaussian noise vector **z** _∼N_ ( **0** _,_ **I** _n_ ) (of the
same dimension as **s** ) and forms the watermarked Gaussian noise **x** by element-wise multiplication:

**x** = �� **z** �� _⊙_ **s** _._ (50)


Here, �� _·_ �� represents the element-wise absolute value operator. By contrast, our scheme applies a fixed
orthogonal rotation:
**x** = _r_ **C s** _,_ (51)
where **C** _∈_ R _[n][×][n]_ is an orthogonal matrix satisfying **C** _[⊤]_ **C** = **I** _n_ . Here, **I** _n_ is the identity matrix of
size _n × n_ . The scalar _r_ is defined as

_√_
_q_
_r_ = ~~_√_~~ _,_ _q_ _∼_ _χ_ [2] _n_ _[,]_ (52)
_n_


i.e., _q_ follows a chi-square distribution with _n_ degrees of freedom.


**AWGN channel.** We now compare the per-bit extraction accuracy of the two mappings under an
additive white Gaussian noise (AWGN) channel. Let the received vector be

**y** = **x** + _**e**_ _,_ _**e**_ _∼N_ ( **0** _, σ_ [2] **I** _n_ ) _._ (53)


In Gaussian Shading, each coordinate of **x** takes the form


_xi_ = _|zi| si,_ _zi_ _∼N_ (0 _,_ 1) _,_ _si_ _∈{±_ 1 _}._ (54)


We extract ˆ _si_ by the sign of the noisy sample _yi_ = _xi_ + _ei_ . Hence, the per-bit extraction accuracy by
Gaussian Shading is


0 - _∞_

_ϕ_ ( _x_ ) _P_  - _x_ + _ei_ _<_ 0� d _x_ +
_−∞_ 0


        - 0
_P_ GS = _P_ - _s_ ˆ _i_ = _si_ - =


_ϕ_ ( _x_ ) _P_  - _x_ + _ei_ _>_ 0� d _x._ (55)
0


Let _ϕ_ and Φ are the standard Gaussian density and cumulative distribution function (CDF). Since


_P_       - _x_ + _ei_ _>_ 0� = Φ� _x/σ_       - _,_ _P_       - _x_ + _ei_ _<_ 0� = 1 _−_ Φ� _x/σ_       - _,_ (56)


we get


0 - _∞_

_ϕ_ ( _x_ ) �1 _−_ Φ( _x/σ_ )� d _x_ +
_−∞_ 0


2 _ϕ_ ( _x_ ) Φ( _x/σ_ ) d _x._ (57)
0


   - 0
_P_ GS =


_∞_ - _∞_

_ϕ_ ( _x_ ) Φ( _x/σ_ ) d _x_ =
0 0


22


In our spherical mapping scheme,


**x** = _r_ **C s** _,_ **C** _[⊤]_ **C** = **I** _n,_ (58)


Upon reception we compute


**y** _[′]_ = **C** _[⊤]_ **y** = **C** _[⊤]_ [�] _r_ **C s** + _**e**_ - = _r_ **s** + **C** _[⊤]_ _**e**_

                    - ��                    _∼N_ (0 _, σ_ [2] **I** _n_ )


_._ (59)


Extraction is done coordinate-wise by ˆ _si_ = sign( _yi_ _[′]_ [)][, the per-bit accuracy of our method is derived by]


_P_ Ours = _P_  - _s_ ˆ _i_ = _si_  
= _P_   - _r si_ + _ηi_ _>_ 0 _| si_ = +1� _P_ ( _si_ = +1) + _P_   - _r si_ + _ηi_ _<_ 0 _| si_ = _−_ 1� _P_ ( _si_ = _−_ 1)


= [1]


2 [1] _[P]_ - _r_ + _ηi_ _>_ 0� + 2 [1]


2 [1] _[P]_ - _−r_ + _ηi_ _<_ 0�


= Φ� _r/σ_    - _, ηi_ _∼N_ (0 _, σ_ [2] ) _._
(60)
Recall the design of _r_ in Eq. 52, so that E[ _r_ ] = 1 and Var( _r_ ) = _O_ (1 _/n_ ). In particular, for large _n_,


�E[ _χn_ ]�2
Var( _r_ ) = _[n][ −]_


_[/]_ [2]

= [1]
_n_ 2


[ _χn_ ]

_≈_ [1] _[/]_ [2]
_n_ _n_


( _n →∞_ ) _._ (61)
2 _n_ _[−→]_ [0]


Hence _r_ concentrates around 1 in probability,


E _r_ �Φ( _r/σ_ )� _−→_ Φ�1 _/σ_        - _,_ _P_ Ours _−→_ Φ�1 _/σ_        - _._ (62)


By definition,


   - _∞_
_P_ GS = 2 _ϕ_ ( _x_ ) Φ� _x/σ_ - d _x_ = E _U_ �Φ( _U/σ_ )� _,_ (63)

0


where _U_ = _|Z|_ with _Z_ _∼_ _N_ (0 _,_ 1) (so _U_ is half-normal on [0 _, ∞_ )). Since Φ is strictly concave on

[0 _, ∞_ ) and E[ _U_ ] = �2 _/π_ _<_ 1, Jensen’s inequality gives


_P_ GS = E _U_ �Φ( _U/σ_ )� _≤_ Φ�E[ _U_ ] _/σ_       - _<_ Φ�1 _/σ_       - = _P_ Ours _._ (64)


Hence _P_ GS _< P_ Ours for all _σ_ _>_ 0. In other words, under the AWGN channel, our proposed scheme
achieves a higher per-bit extraction accuracy than Gaussian Shading.


**Optimality under Equal-Energy Constraint.** In the sequel, we denote the Euclidean norm _∥· ∥_ 2
simply by _∥· ∥_ . Both Gaussian Shading and our mapping embed **s** _∈{±_ 1 _}_ _[n]_ such that the _total_
energy, measured by the squared Euclidean norm of the embedded vector, is on average _n_, yielding a
_per-bit_ energy of exactly 1. In other words, E� **xi** [2][�] = 1 _, i_ = 1 _, . . ., n._ Concretely:


**x** GS = _|_ **z** _| ⊙_ **s** _,_ **z** _∼N_ ( **0** _,_ **I** _n_ ) _,_ (65)


so _∥_ **x** GS _∥_ [2] = _∥_ **z** _∥_ [2] and E( _∥_ **x** GS _∥_ [2] ) = _n_ .


**x** Ours = _r_ **C s** _,_ **C** _[⊤]_ **C** = **I** _n,_ E[ _r_ [2] ] = 1 _,_ (66)

so _∥_ **x** Ours _∥_ [2] = _r_ [2] _∥_ **s** _∥_ [2] = _r_ [2] _n_ and E( _∥_ **x** Ours _∥_ [2] ) = _n_ . After de-rotation by **C** _[⊤]_, bit _i_ yields

_yi_ _[′]_ [=] _[ r s][i]_ [+] _[ η][i][,]_ _ηi_ _∼N_ (0 _, σ_ [2] ) _,_ (67)


so the two hypotheses correspond to symbols + _r_ and _−r_ with distance ∆Ours = 2 _r_ _→_ 2 in
probability as _n →∞_ . In Gaussian Shading, bit _i_ is observed as


_yi_ = _±|zi|_ + _ei,_ _zi, ei_ _∼N_ (0 _,_ 1) _,_ (68)


so ∆GS = 2 _|zi|_ whose expectation 2�2 _/π_ _<_ 2.


It is a classical result in digital communications (Proakis, Digital Communications (Proakis &
Salehi, 2008), Section 5.2) and information theory (Cover & Thomas, Elements of Information
Theory (Cover, 1999)) that, under equal-energy constraints in any symmetric, memoryless additivenoise channel, larger Euclidean separation between symbols strictly reduces bit-error probability.
Since our scheme attains the maximal possible distance of 2 for unit-energy symbols, it is provably
optimal among all binary mappings with the same energy constraints.


23


E WHY LOSSLESSNESS MATTERS: AN DETECTOR-DRIVEN OPTIMIZATION
ANALYSIS


Our goal is to justify, in a detector-aware threat model, that losslessness (distributional indistinguishability between watermarked and unwatermarked inputs) is more fundamental than merely increasing
robustness. If the watermarked distribution leaves any detectable shift from the clean distribution, an
adversary can exploit it to mount effective detector-aware attacks; if the two distributions coincide
(losslessness), the optimization collapses to blind perturbations with little extra damage.


Consider an input image **x**, and let _**δ**_ denote an adversarial perturbation constrained by _∥_ _**δ**_ _∥_ 2 _≤_ _ε_ .
We write the attack objective as

_∥_ _**δ**_ min _∥_ 2 _≤ε_ _[F]_ **[x]** [(] _**[δ]**_ [) =] _[ L]_ [(] _[s]_ [(] **[x]** [ +] _**[ δ]**_ [)) +] _[λ]_ 2 _[∥]_ _**[δ]**_ _[∥]_ 2 [2] _[,]_ (69)


where _s_ ( _·_ ) _∈_ R is the detector score, _L_ is a loss applied to the score, and _λ_ _>_ 0 controls the
trade-off between reducing the score and keeping the perturbation small. The loss _L_ is chosen to
encourage the detector prediction to flip, similar to the binary cross-entropy (BCE) loss used in
standard classification. For instance, if _s_ ( **x** ) is interpreted as the probability that **x** is watermarked,
an adversary may apply a BCE-type objective with the target label flipped from “watermarked” to
“clean”. In this case, the loss explicitly drives the detector to misclassify a watermarked input as
unwatermarked. The derivatives _L_ _[′]_ of such objectives are uniformly bounded, ensuring that the
gradient scaling factor they introduce remains finite.


Next, we derive the optimal perturbation _**δ**_ _[⋆]_ that minimizes _F_ **x** . A first-order expansion at _**δ**_ = **0**
gives (we use the notation _⟨·, ·⟩_ to denote the Euclidean inner product of vectors),

_F_ **x** ( _**δ**_ ) _≈_ _F_ **x** ( **0** ) + _⟨_ **ax** _,_ _**δ**_ _⟩_ + _[λ]_ 2 _[∥]_ _**[δ]**_ _[∥]_ 2 [2] _[,]_ **ax** := _L_ _[′]_ ( _s_ ( **x** )) _∇xs_ ( **x** ) _._ (70)


The optimal perturbation is given by

_**δ**_ _[⋆]_ ( **x** ) = _−_ min� _ε,_ _∥_ **a** _λ_ **x** _∥_ 2             - _∥_ **aaxx** _∥_ 2 _._ (71)

Substituting _**δ**_ _[⋆]_ into the objective yields the decrease

_F_ **x** ( _**δ**_ _[⋆]_ ) _−_ _F_ **x** ( **0** ) = _−_ [1] 2 [min]        - _∥_ **a** _λ_ **x** _∥_ [2] 2 _,_ 2 _ε∥_ **ax** _∥_ 2 _−_ _λε_ [2][�] _._ (72)


This shows that the decrease is monotone in the gradient magnitude: when _∥_ **ax** _∥_ 2 is large, the
optimizer exploits the distortion budget and achieves a significant reduction; when _∥_ **ax** _∥_ 2 _≈_ 0, the
decrease vanishes and the attack is ineffective.


Next, we will analyze how the lossless and lossy cases differ in terms of the expected gradient energy,
and prove that lossless watermarking methods yield smaller values of E _∥_ **ax** _∥_ [2] 2 [, whereas lossy methods]
result in larger values. Formally, let _P_ wm and _P_ clean denote the watermarked and clean distributions
with densities _p_ wm and _p_ clean, respectively. If they differ, the KL divergence is strictly positive. By
the Neyman–Pearson lemma, the _optimal_ detector takes the form of the log-likelihood ratio,

_s_ ( **x** ) = log _[p]_ [wm][(] **[x]** [)] (73)

_p_ clean( **x** ) _[.]_


Its gradient satisfies
_∇_ **x** _s_ ( **x** ) = _∇_ **x** log _p_ wm( **x** ) _−∇_ **x** log _p_ clean( **x** ) _,_ (74)
and therefore
E _P_ wm _∥∇_ **x** _s_ ( **x** ) _∥_ [2] 2 [=] _[ J]_ [(] _[P]_ [wm] _[∥][P]_ [clean][)] _[,]_ (75)
the Fisher divergence between the two distributions. It is known that


_J_ ( _P_ wm _∥P_ clean) ≳ _C D_ KL( _P_ wm _∥P_ clean) _,_ (76)

for some constant _C_ _>_ 0. Since **ax** only differs from _∇_ **x** _s_ ( **x** ) by the bounded factor _L_ _[′]_ ( _s_ ( **x** )), we
conclude that
E _∥_ **aX** _∥_ [2] 2 [≳] _[C D]_ [KL][(] _[P]_ [wm] _[∥][P]_ [clean][)] _[.]_ (77)


Thus, the KL divergence provides a lower bound on the expected gradient energy. In lossy cases, when
_D_ KL( _P_ wm _∥P_ clean) _>_ 0, the gradients remain informative and provide actionable descent directions,


24


Figure 8: Watermarked images generated by SD v3 and FLUX.1-DEV. Top: COCO dataset with SD
v3. Bottom: SDP dataset with FLUX.1-DEV.


Table 6: Extraction performance for different models under post-processing perturbations. The
strength of the post-processing perturbations are consistent with Table 4 in the main paper.


Post-processing Perturbations
Method Metrics

PNG Brightness Gaussian Blur Median Filter JPEG Resize


FLUX.1-DEV ACC 99 _._ 990 _._ 01 98 _._ 840 _._ 28 95 _._ 420 _._ 28 91 _._ 920 _._ 35 88 _._ 850 _._ 56 99 _._ 100 _._ 13
FLUX.1-DEV TPR 100 _._ 000 _._ 00 99 _._ 800 _._ 40 100 _._ 000 _._ 00 100 _._ 000 _._ 00 99 _._ 900 _._ 20 100 _._ 000 _._ 00
SD v3 ACC 99 _._ 990 _._ 01 98 _._ 670 _._ 11 99 _._ 830 _._ 10 99 _._ 740 _._ 11 98 _._ 970 _._ 10 99 _._ 920 _._ 06
SD v3 TPC 100 _._ 000 _._ 00 99 _._ 330 _._ 24 100 _._ 000 _._ 00 100 _._ 000 _._ 00 99 _._ 500 _._ 00 100 _._ 000 _._ 00


allowing the adversary to exploit the distortion budget and cause substantial degradation. In contrast,
in the lossless case _P_ wm = _P_ clean, we have _∇_ **x** _s_ ( **x** ) = **0**, the expected gradient energy collapses,
and the optimizer stalls at _**δ**_ _≈_ **0** . This analysis demonstrates why losslessness is fundamental: only
when the watermarked and clean distributions are indistinguishable does the adversary lose the ability
to perform adversarial attacks that evade the detection system.


F MORE DETAILS AND EXPERIMENTAL RESULTS


F.1 GENERALIZABILITY


Our method demonstrates strong generalizability and
can be readily adapted to other latent-space diffusion Table 7: Watermark tracing accuracy on
architectures, including those based on transformer Guided Diffusion and Glow.
designs (Vaswani et al., 2017; Dosovitskiy et al.,
2021). As an example, we deploy our scheme on
Stable Diffusion v3 (SD v3) (Esser et al., 2024) and Model _lm_ Accuracy (%)
FLUX.1-DEV (BlackForestLabs, 2024). For SD v3, 96 98 _._ 756 _._ 15
we use a first-order Euler ODE solver for both gener- G-Diffusion 192 98 _._ 786 _._ 00
ation and inversion, with 50 timesteps for generation 384 98 _._ 755 _._ 92
and 20 timesteps for inversion. The generated images
have a resolution of 1024 _×_ 1024 with prompts sam- 32 99 _._ 970 _._ 16
pled from the SDP dataset, and each image carries Glow 64 99 _._ 970 _._ 20
a 512-bit watermark. For FLUX.1-DEV, we adopt 128 99 _._ 730 _._ 18
RF-Solver (Wang et al., 2024) for both generation
and inversion, each using 30 timesteps. We evaluate the model on prompts from the COCO dataset.
Each generated image is sized 512 _×_ 512, with a 512-bit watermark embedded. Figure 8 presents
visual examples of watermarked images, which preserve high perceptual fidelity. To further evaluate
the adaptability of our watermarking scheme, we measure extraction performance under a variety


Table 7: Watermark tracing accuracy on
Guided Diffusion and Glow.


Model _lm_ Accuracy (%)


96 98 _._ 756 _._ 15
G-Diffusion 192 98 _._ 786 _._ 00
384 98 _._ 755 _._ 92


32 99 _._ 970 _._ 16
Glow 64 99 _._ 970 _._ 20
128 99 _._ 730 _._ 18


25


Original Regen-Diff Rinse-2xDiff Regen-VAE


Figure 9: Visualized comparison of watermarked images under re-generation.


Original MagicBrush UltraEdit InstructPix2Pix GAN-edit GAN-edit Mask


Figure 10: Visualized comparison of watermarked images under image editing.


of post-processing operations in Table 6. Across these attack settings, the extraction performance
remains high, indicating that the proposed scheme adapts well to different latent-space diffusion
architectures.


Moreover, our method is not limited to latent-space diffusion architectures. To assess its broader
applicability, we apply our scheme to two classic generative models: G-Diffusion (Dhariwal & Nichol,
2021), a pixel-space diffusion model trained on ImageNet (Deng et al., 2009), and Glow (Kingma
& Dhariwal, 2018), a flow-based model with invertible transformations trained on the CIFAR-10
dataset (Krizhevsky et al., 2009). We evaluate on 1000 images for each model and report watermark
tracing accuracy across multiple watermark lengths (Table 7). In both cases, extraction accuracy
under lossless conditions exceeds 98%, demonstrating the effectiveness of our method.


More broadly, our approach applies to any generative model that satisfies two conditions: (1) sampling
from a Gaussian prior, and (2) supporting an invertible mapping between the image and noise domains.
These conditions are met by a wide range of architectures, including diffusion models (Rombach
et al., 2022), normalizing flows (e.g., NICE (Dinh et al., 2014), Glow (Kingma & Dhariwal, 2018)),
GANs with inversion (Abdal et al., 2019).


F.2 ROBUSTNESS AGAINST RE-GENERATION AND EDITING


In Table 8, We evaluate the extraction accuracy of different watermarking methods against image
re-generation and image editing attacks. We additionally include TrustMark (Bui et al., 2025) and
Robust-Wide (Hu et al., 2024) for comparison, as both methods represent more recent advances in
robust watermarking. TrustMark provides a resolution-agnostic pixel-space watermarking framework


26


Table 8: Extraction accuracy for re-generation and editing attacks. Here, _lm_ means watermark length,
“Robust.” refers to “Robust-Wide”, and “InstructP2P.” refers to “InstructPix2Pix”.


Re-generation Editing
Method _lm_

Regen-Diff Rinse-2xDiff Regen-VAE MagicBrush UltraEdit InstructP2P. GAN-edit


DwtDct 32 49 _._ 960 _._ 73 49 _._ 910 _._ 46 50 _._ 270 _._ 72 50 _._ 251 _._ 28 49 _._ 281 _._ 96 50 _._ 561 _._ 38 49 _._ 882 _._ 21
DwtDctSvd 32 50 _._ 200 _._ 93 49 _._ 690 _._ 74 48 _._ 980 _._ 58 49 _._ 221 _._ 76 49 _._ 621 _._ 62 48 _._ 530 _._ 88 48 _._ 283 _._ 44
RivaGan 32 56 _._ 770 _._ 56 54 _._ 190 _._ 53 50 _._ 890 _._ 33 67 _._ 311 _._ 24 52 _._ 062 _._ 33 60 _._ 561 _._ 52 99 _._ 840 _._ 14
TrustMark 100 71 _._ 360 _._ 46 59 _._ 940 _._ 30 93 _._ 870 _._ 33 86 _._ 741 _._ 85 68 _._ 161 _._ 18 81 _._ 683 _._ 54 95 _._ 851 _._ 34
Robust. 64 96 _._ 900 _._ 18 93 _._ 220 _._ 23 94 _._ 150 _._ 63 94 _._ 361 _._ 54 80 _._ 090 _._ 68 96 _._ 441 _._ 10 98 _._ 330 _._ 61
PRC 100 99 _._ 260 _._ 41 94 _._ 220 _._ 88 75 _._ 911 _._ 25 94 _._ 142 _._ 33 81 _._ 533 _._ 09 83 _._ 537 _._ 82 100 _._ 000 _._ 00
Ours 100 99 _._ 630 _._ 17 97 _._ 700 _._ 37 87 _._ 480 _._ 45 93 _._ 962 _._ 94 86 _._ 111 _._ 61 92 _._ 532 _._ 38 100 _._ 000 _._ 00


Table 9: PSNR and SSIM for traditional watermarking methods. Model: SD v1.5.


COCO SDP
Method

PSNR SSIM PSNR SSIM


DwtDct 38.08990 _._ 1516 0.99980 _._ 0000 38.46490 _._ 1114 0.99940 _._ 0001
DwtDctSvd 38.07360 _._ 1596 0.99980 _._ 0000 38.92610 _._ 2489 0.99990 _._ 0000
RivaGan 40.56200 _._ 0045 0.99680 _._ 0001 40.82940 _._ 0713 0.99050 _._ 0017


with superior robustness. Robust-Wide focuses on robustness against instruction-driven image editing,
and we evaluate its 64-bit configuration following the original implementation. For a fair comparison,
we adjust the watermark length of our method to 100 bits.


For re-generation attacks, we adopt the WAVES benchmark (Ding et al., 2024), which contains three
re-generation attacks: Regen-Diff, Rinse-2xDiff, and Regen-VAE. Specific attacked examples are
shown in the Figure 9. The experimental results on the COCO dataset using SD v2.1 are reported
in Table 8. Our method achieves the highest accuracy under both Regen-Diff and Rinse-2xDiff,
demonstrating superior robustness to diffusion-based re-generation attacks. In the Regen-VAE setting,
TrustMark and Robust-Wide obtain slightly higher accuracy, while our method still outperforms the
lossless PRC Watermark across all attack scenarios.


For image editing and forgery, we employ the W-Bench benchmark (Lu et al., 2024). This evaluation
covers a range of manipulations, including MagicBrush (Zhang et al., 2023), UltraEdit (Zhao et al.,
2024), InstructPix2Pix (Brooks et al., 2023). We also introduce GAN-edit (Ma et al., 2022), a
GAN-based inpainting method. Figure 10 presents examples of the applied image editing attacks.
The experiments are conducted on the W-Bench dataset using the SD v2.1. As shown in the Table 8,
our scheme delivers high extraction accuracy across diverse editing operations, staying above 85% on
average across all attack scenarios. These results indicate that our robustness against image editing is
broadly comparable to existing state-of-the-art watermarking methods.


F.3 UNDETECTABILITY AND IMAGE QUALITY


In this section, we present additional experiments on both undetectability and image quality assessment. First, Tables 9 and 10 report the Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity
Index Measure (SSIM) (Wang et al., 2004) of traditional watermarking methods. These results
indicate that conventional approaches tend to degrade image quality, with an average PSNR of 39
dB. To further evaluate the perceptual quality of watermarked images generated by latent-based
methods, we incorporate the Inception Score (IS) (Salimans et al., 2016) as an additional metric, as
shown in Table 11. Although the IS is not sensitive to distribution shifts, we still observe that lossy
watermarking schemes tend to yield lower IS scores compared to lossless ones. This observation
aligns with our theoretical expectation that lossy watermarking inherently induces distributional
deviations.


For undetectability, we compare the classification performance of trained detectors on watermarked
versus unwatermarked images in Figure 11, and further illustrate the ability to distinguish watermarked
images generated by different keys in Figure 12. The PRC Watermark and our method exhibit strong


27


Table 10: PSNR and SSIM for traditional watermarking methods. Model: SD v2.1.


COCO SDP
Method

PSNR SSIM PSNR SSIM


DwtDct 38.50410 _._ 1484 0.99990 _._ 0000 38.06400 _._ 2543 0.99990 _._ 0000
DwtDctSvd 38.65570 _._ 1862 0.99990 _._ 0000 37.97920 _._ 2627 0.99990 _._ 0000
RivaGan 40.56280 _._ 0065 0.99680 _._ 0001 40.51110 _._ 0064 0.99680 _._ 0001


Table 11: IS value for different watermarking methods. Higher IS indicates higher image quality.


COCO SDP
Method

SD v1.5 SD v2.1 SD v1.5 SD v2.1


Original 29.35461 _._ 8784 28.87801 _._ 7803 11.59650 _._ 7757 10.29360 _._ 7868
DwtDct 29.23781 _._ 7822 28.46451 _._ 4584 10.92700 _._ 7261 10.15930 _._ 7254
DwtDctSvd 29.40931 _._ 7302 28.61211 _._ 4145 11.08620 _._ 7460 10.22260 _._ 6963
RivaGan 29.23981 _._ 7400 28.25291 _._ 3353 10.99880 _._ 7410 10.16680 _._ 7217
Tree-Ring 29.35280 _._ 3373 28.81021 _._ 3819 11.56700 _._ 2605 10.07010 _._ 7270
Gaussian Shading 29.08680 _._ 2818 28.45301 _._ 6960 11.44380 _._ 5970 10.13220 _._ 1784
PRC Watermark 29.42441 _._ 5163 28.99311 _._ 5686 11.68870 _._ 6781 10.21020 _._ 8206
Ours 29.55212 _._ 2552 28.97811 _._ 7562 11.66280 _._ 6790 10.26720 _._ 7218


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


1.0


0.9


0.8


0.7


0.6


0.5


0.4


1 1001 2001 Epoch 3001 4001


(f) Latent Level.


1 1001 2001 Epoch 3001 4001


(a) Latent Level.


1 21 41 Epoch 61 81


(b) Image Level.


1 21 41 Epoch 61 81


(c) Image Level.


1 21 41 Epoch 61 81


(d) Image Level.


1 11 21 31 41 Epoch51 61 71 81 91


(e) Image Level.


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


0.8


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1 21 41 Epoch 61 81


(g) Image Level.


1 21 41 Epoch 61 81


(h) Image Level.


1 21 41 Epoch 61 81


(i) Image Level.


1 11 21 31 41 Epoch51 61 71 81 91


(j) Image Level.


Figure 11: Training loss and test classification accuracy for watermark and unwatermarked samples
classification. (a)(f) on the latent level. (b)(g) on the COCO dataset with SD v1.5. (c)(h) on the
COCO dataset with SD v2.1. (d)(i) on the SDP dataset with SD v1.5. (e)(j) on the SDP dataset with
SD v2.1. Top: Training loss. Bottom: Test classification accuracy.


resistance to classification, maintaining strong indistinguishability in both latent and image levels. In
contrast, the Tree-Ring and Gaussian Shading methods, which rely on fixed key patterns, introduce
significant and consistent artifacts. These artifacts are easily captured by the detectors, resulting
in classification accuracies of 100% and 98%, respectively. This indicates that such methods lack
robustness against adversarial classification and are highly susceptible to reverse-engineering attacks.


F.4 TRACING ACCURACY


We evaluate tracing performance in Table 12, 13, and 14, comparing the two diffusion model variants
under clean, post-processing, and adversarial conditions using the COCO dataset and SDP dataset.
As mentioned in the main paper, “Clean” refers to PNG storage, while “Post-Processing” reports
the average performance under a range of common image perturbations. These post-processing
perturbations include Additive White Gaussian Noise with a standard deviation of 0.05, Brightness
Adjustment with a factor of 2, Random Drop with a drop ratio of 0.1, Gaussian Blur with a kernel
size of 9, JPEG compression with a quality factor of 70 (QF=70), Median Filter with a kernel size of


28


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


|Gauss|ian Shading Spherical|
|---|---|
|||
|||
|||
|||
|||
|||


1 301 601 Epoch 901 1201


(a) Latent Level.


1 21 41 Epoch 61 81


(b) Image Level.


1 21 41 Epoch 61 81


(c) Image Level.


1 21 41 Epoch 61 81


(d) Image Level.


1 21 41 Epoch 61 81


(e) Image Level.


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


1.4


1.2


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.9


0.8


0.7


0.6


0.5


|Col1|Col2|
|---|---|
|||
|||
|||
|||
|||


1 301 601 Epoch 901 1201


(f) Latent Level.


1 21 41 Epoch 61 81


(g) Image Level.


1 21 41 Epoch 61 81


(h) Image Level.


1 21 41 Epoch 61 81


(i) Image Level.


1 21 41 Epoch 61 81


(j) Image Level.


Figure 12: Training loss and test classification accuracy on watermarked images generated by different
keys. (a)(f) on the latent level. (b)(g) on the COCO dataset with SD v1.5. (c)(h) on the COCO dataset
with SD v2.1. (d)(i) on the SDP dataset with SD v1.5. (e)(j) on the SDP dataset with SD v2.1. Top:
Training loss. Bottom: Test classification accuracy.


(a) PNG. (b) Gaussian Noise. (c) Brightness. (d) Drop.


(e) Gaussian Blur. (f) JPEG. (g) Median Filter. (h) Resize.


Figure 13: Images under common attacks. (a) PNG image. (b) Additive White Gaussian Noise,
_µ_ = 0 _, σ_ = 0 _._ 05. (c) Brightness, factor=2. (d) Drop, drop ratio=0.1. (e) Gaussian Blur, kernel size=9.
(f) JPEG, QF=70. (g) Median Filter, kernel size=5. (f) 50% Resize.


5, and 50% Resize (followed by restoration to the original dimensions). Representative examples of
these perturbations are shown in Figure 13.


For the Adversarial conditions, we adopt the adversarial attack WEvade (Jiang et al., 2023). In the
White-box setting, we use WEvade-W-I with the parameters: maximum pixel perturbation _rb_ = 10,
gradient learning rate _α_ = 2, and error rate _ϵ_ = 0 _._ 01. For the Black-box setting, we employ WEvadeB-S, which relies only on binary output predictions, and additionally adopt the JPEG compression
with a quality factor of 50 (QF=50) from WEvade-B-Q, before applying the same default parameters
as in the White-box setting. A pre-trained ResNet-18 is employed as a surrogate model for generating
the adversarial perturbations. The final results are obtained by averaging over the two settings.


In Table 12, 13, and 14, our method achieves over 95% ACC and TPR across Clean and PostProcessing settings. Under Adversarial conditions, the accuracy of lossy schemes drops sharply,


29


Table 12: Comparison results on ACC and TPR. Dataset: COCO dataset. Model: SD v1.5. “Post.”
refers to “Post-Processing” and “Adv.” refers to “Adversarial”.


Metrics
Method

ACC (Clean) ACC (Post.) ACC (Adv.) TPR (Clean) TPR (Post.) TPR (Adv.)


DwtDct 88.830 _._ 74 64.340 _._ 73 49.220 _._ 00 92.701 _._ 03 51.461 _._ 72 15.800 _._ 01
DwtDctSvd 100.000 _._ 01 92.860 _._ 26 48.710 _._ 00 100.000 _._ 00 91.490 _._ 71 18.450 _._ 01
RivaGan 99.550 _._ 15 96.590 _._ 31 52.780 _._ 01 100.000 _._ 00 99.130 _._ 25 27.850 _._ 03
Tree-Ring  -  -  - 94.503 _._ 96 93.984 _._ 24 13.720 _._ 07
Gaussian Shading 100.000 _._ 00 98.550 _._ 04 89.420 _._ 11 100.000 _._ 00 99.950 _._ 03 99.300 _._ 00

PRC Watermark 100.000 _._ 00 93.750 _._ 14 98.300 _._ 07 100.000 _._ 00 87.510 _._ 28 96.610 _._ 01
Ours 99.990 _._ 00 95.270 _._ 11 98.500 _._ 03 100.000 _._ 00 97.570 _._ 18 99.950 _._ 00


Table 13: Comparison results on ACC and TPR. Dataset: SDP dataset. Model: SD v2.1. “Post.”
refers to “Post-Processing” and “Adv.” refers to “Adversarial”.


Metrics
Method

ACC (Clean) ACC (Post.) ACC (Adv.) TPR (Clean) TPR (Post.) TPR (Adv.)


DwtDct 83.070 _._ 74 63.140 _._ 70 48.830 _._ 01 87.801 _._ 25 52.772 _._ 23 15.600 _._ 03
DwtDctSvd 99.970 _._ 02 92.500 _._ 26 49.400 _._ 01 100.000 _._ 00 89.460 _._ 41 20.200 _._ 03
RivaGan 99.060 _._ 21 95.870 _._ 38 50.770 _._ 00 100.000 _._ 00 98.940 _._ 36 22.400 _._ 03
Tree-Ring  -  -  - 100.000 _._ 00 98.590 _._ 35 2.840 _._ 01
Gaussian Shading 99.990 _._ 00 98.470 _._ 04 84.500 _._ 10 100.000 _._ 00 99.960 _._ 03 99.690 _._ 00

PRC Watermark 99.970 _._ 03 93.590 _._ 18 96.190 _._ 10 99.940 _._ 05 87.190 _._ 36 92.390 _._ 00
Ours 99.890 _._ 02 94.920 _._ 09 96.940 _._ 05 100.000 _._ 00 97.690 _._ 18 99.870 _._ 00


Table 14: Comparison results on ACC and TPR. Dataset: SDP dataset. Model: SD v1.5. “Post.”
refers to “Post-Processing” and “Adv.” refers to “Adversarial”.


Metrics
Method

ACC (Clean) ACC (Post.) ACC (Adv.) TPR (Clean) TPR (Post.) TPR (Adv.)


DwtDct 84.911 _._ 55 63.230 _._ 87 49.550 _._ 01 90.503 _._ 03 51.542 _._ 54 16.200 _._ 02
DwtDctSvd 100.000 _._ 00 92.480 _._ 35 49.240 _._ 00 100.000 _._ 00 89.460 _._ 59 17.850 _._ 01
RivaGan 99.320 _._ 11 96.340 _._ 30 51.480 _._ 01 100.000 _._ 00 98.970 _._ 23 23.950 _._ 04
Tree-Ring  -  -  - 100.000 _._ 00 99.100 _._ 39 4.670 _._ 03
Gaussian Shading 99.990 _._ 01 98.490 _._ 06 85.800 _._ 10 100.000 _._ 00 99.950 _._ 02 99.810 _._ 00

PRC Watermark 99.970 _._ 02 93.580 _._ 20 97.280 _._ 09 99.940 _._ 05 87.160 _._ 39 94.580 _._ 00
Ours 99.930 _._ 01 95.010 _._ 09 97.530 _._ 05 100.000 _._ 00 97.630 _._ 19 99.810 _._ 00


Table 15: Ablation of parameters _s_ and _N_ on TPR under different attacks. Dataset: COCO. Model:
SD v1.5. Case 1: Gaussian Blur, kernel size = 9. Case 2: JPEG-70. Case 3: Brightness, factor = 2.


sparsity parameter _s_ repetition count _N_
Case

1 2 3 4 1 11 21 31


1 99.960 _._ 05 99.800 _._ 06 99.340 _._ 26 98.420 _._ 29 99.580 _._ 15 99.980 _._ 04 99.900 _._ 06 99.960 _._ 05
2 99.980 _._ 04 99.600 _._ 23 98.720 _._ 32 94.980 _._ 86 98.660 _._ 23 99.940 _._ 12 99.940 _._ 05 99.980 _._ 04
3 99.880 _._ 04 98.440 _._ 29 94.120 _._ 77 85.960 _._ 86 95.860 _._ 51 99.480 _._ 07 99.800 _._ 14 99.880 _._ 04


since lossy embeddings expose detectable patterns that enable targeted attacks. By contrast, lossless
schemes show clear superiority: our method improves accuracy by more than 10%. Compared
with PRC Watermark, our approach achieves an additional gain of nearly 10% in TPR under PostProcessing distortions, consistent with the main paper. Overall, these findings confirm that _Spherical_
_Watermark_ enables exact recovery while maintaining superior robustness over both lossy and lossless
baselines.


30


Table 16: Ablation of parameters _s_ and _N_ on TPR under different attacks. Dataset: SDP. Model: SD
v1.5. Case 1: Gaussian Blur, kernel size = 9. Case 2: JPEG-70. Case 3: Brightness, factor = 2.


sparsity parameter _s_ repetition count _N_
Case

1 2 3 4 1 11 21 31


1 99.960 _._ 05 99.600 _._ 20 98.060 _._ 41 95.800 _._ 97 98.780 _._ 26 99.920 _._ 07 99.860 _._ 05 99.960 _._ 05
2 99.860 _._ 10 98.920 _._ 27 96.260 _._ 49 92.620 _._ 86 97.360 _._ 46 99.520 _._ 23 99.940 _._ 05 99.860 _._ 10
3 99.900 _._ 06 99.040 _._ 23 95.840 _._ 62 91.740 _._ 48 97.240 _._ 28 99.600 _._ 13 99.720 _._ 15 99.900 _._ 06


Table 17: Ablation of parameters _s_ and _N_ on TPR under different attacks. Dataset: SDP. Model: SD
v2.1. Case 1: Gaussian Blur, kernel size = 9. Case 2: JPEG-70. Case 3: Brightness, factor = 2.


sparsity parameter _s_ repetition count _N_
Case

1 2 3 4 1 11 21 31


1 100.000 _._ 00 99.700 _._ 13 98.600 _._ 14 96.120 _._ 48 99.140 _._ 20 99.940 _._ 05 100.000 _._ 00 100.000 _._ 00
2 99.960 _._ 05 98.560 _._ 19 94.080 _._ 68 88.440 _._ 72 96.280 _._ 50 99.660 _._ 16 99.820 _._ 13 99.960 _._ 05
3 99.980 _._ 04 99.200 _._ 18 95.960 _._ 39 89.741 _._ 30 97.220 _._ 37 99.760 _._ 16 99.880 _._ 12 99.980 _._ 04


0.54


0.52


0.50


0.48


0.46


|Col1|Col2|Col3|Col4|Col5|l lm {{ 22 55 66,, 22 66 44|, . . ., 512}<br>, . . ., 512}|
|---|---|---|---|---|---|---|
|||||~~r~~<br>|~~r~~<br>||
||||||||
||||||||
||||||||
||||||||


300 350 Parameter400 450 500


(a) Ablation on _lm_ and _lr_ .


0.54


0.52


0.50


0.48


0.46


|Col1|Col2|Col3|Col4|Col5|Col6|N<br>s|(1, 31)<br>(1, 31)|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


0 5 10 Parameter15 20 25 30


(b) Ablation on _s_ and _N_


Figure 14: Ablation on hyperparameters undetectability.


F.5 ABLATION EXPERIMENTS


**Ablation on Modules.** We have supplemented additional module ablation experiments, isolating
each module for testing across two datasets and two models. Figure 15 details the experimental
results. Here we apply four representative types of attacks: Brightness Adjustment, Gaussian Blur,
Median Filter, and Resize. The results clearly show that the absence of the spherical mapping module
leads to a significant drop in robustness against all tested attacks. This observation is consistent with
the findings discussed in the main paper and underscores the importance of spherical mapping in
preserving watermark integrity under perturbation. Furthermore, in the Appendix D, we provide a
rigorous analysis of how orthogonal rotation enhances robustness.


**Settings for the Ablation on Parameters.** In the ablation studies, we configure parameters with
different settings. To evaluate their impact on undetectability, we set _s_ = 1 _, N_ = 31 _, lm_ = 512
and _lr_ = 512 in default. Then, we systematically vary a single parameter at a time while keeping
the others fixed to the default configuration, allowing us to isolate the impact of each parameter
on performance. Note that the size _lx_ of the latent space varies with parameter adjustments. In
Figure 6(d) of the main paper, the values of _s_ and _N_ range from 1 to 31, resulting in 31 test points.
For _lm_ and _lr_, the values range from 256 to 512 with a step size of 8, yielding 33 test points.


To evaluate the impact of the ablated parameters _s, N, lm_ on tracing accuracy, we vary one parameter
at a time while keeping the latent space size _lx_ fixed to match the input dimension of diffusion models.
Specifically, varying _s_ does not affect other parameters, whereas changing _N_ requires adjusting
_lr_ accordingly, and modifying _lm_ necessitates updates to both _N_ and _lr_, so that the constraint
_lx_ = _N_ _× lm_ + _lr_ remains satisfied. This ensures fair comparisons across different ablation settings.


31


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


4 6 8 10 12 14 16
Kernel Size


(c) Median Filter.


1 2 3 4 5 6 7 8
Brightness Factor


(a) Brightness.


5 10 15 20 25 30
Kernel Size


(b) Gaussian Blur.


0.300 0.275 0.250 0.225 0.200 0.175 0.150 0.125 0.100

Resize Ratio


(d) Resize.


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


4 6 8 10 12 14 16
Kernel Size


(g) Median Filter.


1 2 3 4 5 6 7 8
Brightness Factor


(e) Brightness.


5 10 15 20 25 30
Kernel Size


(f) Gaussian Blur.


0.300 0.275 0.250 0.225 0.200 0.175 0.150 0.125 0.100

Resize Ratio


(h) Resize.


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


4 6 8 10 12 14 16
Kernel Size


(k) Median Filter.


1 2 3 4 5 6 7 8
Brightness Factor


(i) Brightness.


1.00


0.95


0.90


0.85


0.80


0.75


0.70


1.00


0.95


0.90


0.85


0.80


0.75


0.70


0.65


1.00


0.95


0.90


0.85


0.80


0.75


0.70


1.00


0.95


0.90


0.85


0.80


0.75


0.70


0.65


5 10 15 20 25 30
Kernel Size


(j) Gaussian Blur.


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


1.0


0.9


0.8


0.7


0.6


0.300 0.275 0.250 0.225 0.200 0.175 0.150 0.125 0.100

Resize Ratio


(l) Resize.


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|w/o, w/ w/, w/|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|w/o, w/<br>w/, w/o<br>w/, w/|||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


4 6 8 10 12 14 16
Kernel Size


(o) Median Filter.


1 2 3 4 5 6 7 8
Brightness Factor


(m) Brightness.


5 10 15 20 25 30
Kernel Size


(n) Gaussian Blur.


0.300 0.275 0.250 0.225 0.200 0.175 0.150 0.125 0.100

Resize Ratio


(p) Resize.


Figure 15: The ACC result of ablation on module _B_ and _S_ . (a-d) COCO dataset with SD v1.5. (e-h)
COCO dataset with SD v2.1. (i-l) SDP dataset with SD v1.5. (m-p) SDP dataset with SD v2.1.


We do not ablate _lr_ separately, since changes in _lr_ do not affect the number of watermark–padding
mixtures (which is controlled by _s_ ), and thus have negligible influence on tracing accuracy.


**Ablation on Parameters.** We conduct ablation studies on four hyperparameters: the watermark
length _lm_, padding length _lr_, row sparsity parameter _s_, and repetition count _N_ . To evaluate the
undetectability of the four hyperparameters, we conduct experiments in the latent space, as shown in
Figure 14. It can be observed that the classification accuracy fluctuates around 50% across all settings,
indicating that variations in these parameters do not affect undetectability. In Tables 15, 16, and 17
we report the TPR under various attack scenarios for different values of _s_ and _N_ . Consistent with
the main paper, increasing _s_ and decreasing _N_ tend to degrade the model’s robustness. In Figure 16,
we evaluate the impact of watermark length _lm_, i.e., capacity, across different datasets and models.
The attacks include Brightness Adjustment with a factor of 2, Gaussian Blur with kernel size 9,
JPEG compression at quality 70, and 30% Resize. The performance of the PRC Watermark drops
sharply when the watermark length _lm_ approaches 2000, whereas the _Spherical Watermark_ maintains
relatively high ACC and TPR. This demonstrates the design flexibility of our approach, making it
well-suited for high-capacity scenarios.


**Settings for the Ablation on Diffusion Sampling Configurations.** In the ablation of diffusion
sampling settings studies, we conduct experiment on the COCO dataset using the SD v2.1 model.
Here, we evaluate the watermark extraction accuracy under various attacks across three ODE solvers.


32


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(a) Brightness.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(b) Gaussian Blur.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(c) JPEG.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(d) Resize.


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(e) Brightness.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(f) Gaussian Blur.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(g) JPEG.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(h) Resize.


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(i) Brightness.


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(j) Gaussian Blur.


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(k) JPEG.


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


1.0


0.8


0.6


0.4


0.2


0.0


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(l) Resize.


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


|Col1|Col2|PRC. ACC Ours ACC|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR|PRC. ACC<br>PRC. TPR@1%FPR<br>Ours ACC<br>Ours TPR@1%FPR||
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(m) Brightness.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(n) Gaussian Blur.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(o) JPEG.


500 1000 1500 2000 2500 3000 3500 4000
Capacity


(p) Resize.


Figure 16: The ACC and TPR of PRC Watermark and Ours under different watermark length _lm_ .
(a-d) COCO dataset with SD v1.5. (e-h) COCO dataset with SD v2.1. (i-l) SDP dataset with SD v1.5.
(m-p) SDP dataset with SD v2.1.


These attacks include Brightness Adjustment with a factor
of 2, Gaussian Blur with a kernel size of 9, JPEG compression with a quality factor of 70 (QF=70), Median Filter
with a kernel size of 5, and 50% Resize.


**Ablation on Diffusion Sampling Settings.** To quantitatively assess the impact of inversion errors on watermark
extraction, we conduct the ablation study of latent-space
perturbation on extraction accuracy. Specifically, we inject
Gaussian noise with zero mean and standard deviation _σ_
into the watermarked latent vectors, where _σ_ controls the
noise level. As shown in Figure 17, even at a noise level
1.5 _×_ larger than the latent’s original standard deviation,
the extraction success rate remains above 95%. This confirms that our method is highly tolerant to moderate latent
perturbations.


33


1.0


0.9


0.8


0.7


0.6


0.5


0.4


0.3


0 1 2 3 4 5 6
Noise Level


Figure 17: Extraction accuracy for different noise level.
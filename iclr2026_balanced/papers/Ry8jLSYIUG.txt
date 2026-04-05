|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|PSNR-only bound|
|---|---|---|---|---|---|---|---|---|
|||||<br>10 3<br>10 2<br>10 1<br>100||||PSNR-only bound<br>PSNR + Rotation 3<br>PSNR + Crop 50%<br>PSNR + linJPEG (Q<br>ChunkySeal (This W<br>VideoSeal<br>CIN<br>HiDDeN<br>MBRS<br>TrustMark<br>WAM<br>LISO-JPEG (limite|
||||||||||
||||||||||
||||||||||


Figure 1: **Existing** **image** **watermarking** **models** **have** **capacities** **well** **under** **what** **this** **paper**
**suggests to be possible.** Shown are theoretical bounds on watermarking capacity under a PSNR
constraint alone (thick line) and in combination with robustness requirements (thin lines). Recent
methods operate far below the achievable bounds, often by orders of magnitude, as seen in the
log-scale inset. Our proposed **Chunky Seal (1024 bits)** pushes capacity higher than prior work, but
is still very far from the theoretical limits, indicating a large potential for future development.


1 INTRODUCTION


Invisible image watermarking embeds an _imperceptible_ secret message of a _certain capacity_ recoverable _under a variety of perturbations_, leading to an inherent capacity-quality-robustness trade-off.
Classic methods used hand-crafted tools, such as the mid-frequencies of the discrete cosine transform
(Al-Haj, 2007; Navas et al., 2008), discrete wavelet transform (Xia et al., 1998; Barni et al., 2001) or
a combination of them (Navas et al., 2008; Feng et al., 2010; Zear et al., 2018). Deep learning led to
significant improvements in all three dimensions via attacking fixed decoders (Vukotic et al.´, 2018;
Fernandez et al., 2022), or via end-to-end training of the embedder and decoder (Mun et al., 2017;
Zhu et al., 2018; Tancik et al., 2020; Bui et al., 2023a; Xu et al., 2025; Sander et al., 2025). Yet,
despite these techniques, it seems that progress has stagnated. State-of-the-art methods successfully
embed around 100 _−_ 200 bits in a relatively imperceptible way (i.e., Peak Signal-to-Noise Ratio,
PSNR, above 40 dB) while robust to perturbations. Improvements in quality and robustness continue,
but they are only marginal, leading many to believe we are nearing the limits of what is possible.


1


# WE CAN HIDE MORE BITS:

### THE UNUSED WATERMARKING CAPACITY IN THEORY AND IN PRACTICE

**Anonymous authors**
Paper under double-blind review


ABSTRACT


Despite rapid progress in deep learning–based image watermarking, the capacity
of current robust methods remains limited to the scale of only a few hundred bits.
Such plateauing progress raises the question: _How far are we from the fundamental_
_limits of image watermarking?_ To this end, we present an analysis that establishes
upper bounds on the message-carrying capacity of images under PSNR and linear
robustness constraints. Our results indicate theoretical capacities are orders of
magnitude larger than what current models achieve. Our experiments show this
gap between theoretical and empirical performance persists, even in minimal,
easily analysable setups. This suggests a fundamental problem. As proof that
larger capacities are indeed possible, we train Chunky Seal, a scaled-up version of
Video Seal, which increases capacity 4 _×_ to 1024 bits, all while preserving image
quality and robustness. These findings demonstrate modern methods have not yet
saturated watermarking capacity, and that significant opportunities for architectural
innovation and training strategies remain.


8


6


4


2


0


10 20 30 40 50 60 70
PSNR [dB]


20 30 40


Image watermarking indeed may already be a solved problem. Unlike generative or discriminative
models that can improve as data and parameters are scaled, watermarking has an inherent performance
ceiling. Given an image resolution and a set of robustness constraints, there is a finite amount of
information that can be embedded imperceptibly. The existence of this limit and the converging
empirical performance of recent models naturally leads to a critical question: **Have** **we** **already**
**reached the theoretical ceiling of watermarking performance?**


To answer this question, we need to know what this limit actually is and to measure how close our
models are to it. We address these challenges in the current paper and offer the following findings:

i. We propose bounds on the capacity of watermarking under a PSNR constraint and robustness to
linear augmentations, indicating capacities orders of magnitude larger than seen in practice.
ii. Watermarking models are trained with constraints we cannot directly analyse so we retrain
Video Seal (Fernandez et al., 2024), a SOTA image and video watermarking model, to match our
simplest theoretical setup: watermarking a single gray image under only a PSNR constraint. Yet,
Video Seal fails to encode even 1024 bits, when we successfully encode 2048 bits with a linear
model, 32 _,_ 768 bits by tiling lower-resolution watermarks, and 456 _,_ 509 bits with a handcrafted
model. This indicates severe structural limitations.
iii. With the standard quality and robustness constraints, we train Chunky Seal, a simple scale-up of
Video Seal, which embeds 1024 bits while maintaining similar robustness and image quality. [1]


Therefore, our theory and experiments show that **it is possible to achieve much higher capacities**
**than we currently have, although that might require innovation in architectures and training.**


2 BOUNDS ON WATERMARKING CAPACITY


We first discuss previous approaches to watermarking capacity in Section 2.1. We then model images
as points on a high-dimensional grid, where capacity is determined by the number of unique points that
satisfy imperceptibility and robustness constraints. Using this model, we first establish the absolute
maximum information capacity (Section 2.2), then apply a PSNR constraint (Sections 2.3 and 2.4),
and subsequently incorporate robustness to transformations like cropping, rescaling, and rotation
(Section 2.5). We conclude by exploring the impact of data distribution on capacity (Section 2.6).


2.1 RELATED WORK ON THEORETICAL MODELS OF WATERMARKING CAPACITY

Previous work on watermarking capacity largely relied on unrealistic assumptions like Gaussian noise
(Costa, 1983; Cohen and Lapidoth, 2002; Chen and Wornell, 2002). More practical approaches were
limited to small-magnitude perturbations (Moulin and O’Sullivan, 2003; Moulin and Koetter, 2005;
Somekh-Baruch and Merhav, 2004) or specific geometric transformations (Merhav, 2005). Rather
than these information-theoretic methods, which view the problem as power-limited communication
over a super-channel with a state that is known to the encoder, our work takes a geometric approach
allowing us to study more realistic conditions. Extended related works on image watermarking
methods and theoretical approaches are discussed in App. A.


2.2 ABSOLUTE CAPACITY OF THE IMAGE SPACE

Watermarking embeds a message _m_ into an image _**x**_ . Since each message must correspond to a
distinct encoded image, the number of unique messages, that is, the watermarking _capacity_ is limited
by the number of distinct images. An _l_ -bit message requires at least 2 _[l]_ such images. We represent an
image as a vector of length _cwh_, where _c_ is the number of channels, _w_ is the width and _h_ the height,
with each element having 2 _[k]_ discrete levels when using _k_ -bit colour depth. The tuple ( _c, w, h, k_ )
defines an _image format_ . The set of all possible images in this format is _I_ = _{_ 0 _,_ 1 _, . . ., ρ}_ _[cwh]_ with
_ρ_ = 2 _[k]_ _−_ 1, which can be thought of as a finite grid [2] of points in R _[cwh]_ . This immediately gives us
a trivial upper bound on watermarking capacity: since each message must correspond to a distinct
watermarked image, it is not possible to embed more messages than there are distinct images.


**Bound 1:** **Absolute capacity of the image.** The capacity of images in the format ( _c, w, h, k_ ) is


capacity[in bits] = log2 _|I|_ = log2 �(2 _[k]_ ) _[cwh]_ [�] = _cwhk_ bits _._


1Chunky Seal code and checkpoints will be released.
2The set of valid images is a finite subset of a lattice in R _cwh_, i.e., _I_ =[0 _, ρ_ ] _cwh∩{_ [�] _cwhi_ =1 _[a][i]_ _**[e]**_ _[i]_ _[|][ a][i][∈]_ [Z] _[}]_ [ with]
_**e**_ _i_ the _i_ -th unit vector, but we use the term _grid_ for readability since no deeper lattice theory is required.


2


Interpreting PSNR as an _ℓ_ 2-ball constraint gives us an avenue for measuring the message-carrying
capacity under it by considering the amount of integer points inside both the cube and this ball.
Counting how many such points exist is not trivial, and we analyse the three possible cases (see
Figure 2): _i._ the ball is so large that it contains the entire cube (very low _τ_ ); _ii._ the ball is small
enough to lie fully inside the cube (high _τ_ ); _iii._ the ball and cube partially overlap (medium _τ_ ). We
begin by assuming the cover image _**x**_ lies at the centre of the admissible range, i.e., _**x**_ _g_ = 2 _[k][−]_ [1] **1**
as then the volume of the intersection (and thus the capacity) is maximized. In Section 2.4 we will
extend the analysis to arbitrary images.


2.3.1 CUBE IN BALL (LOW PSNR)
When _τ_ is low, _ϵ_ ( _τ_ ) is large and the ball contains the entire cube _CI_ = [0 _, ρ_ ] _[cwh]_ . The PSNR
constraint does not rule out any images, so the capacity is just the absolute maximum (Bound 1):


**Bound 2:** **Gray image, PSNR constraint (low PSNR).** The capacity of a gray image _**x**_ _g_ under a
very low minimum PSNR threshold _√ τ_ is capacity[in bits] = _cwhk._
**Bound validity:** When _ϵ_ ( _τ_ ) _≥_ _[ρ]_ _/_ 2 _cwh_, or equivalently when _τ_ _≤_ 20 log10 2 _≈_ 6 _._ 02 dB.


2.3.2 BALL IN CUBE (HIGH PSNR)

When _τ_ is high, the PSNR ball is fully inside the cube. The capacity is the number of integer points
inside the ball, a problem with no general closed form. In high dimensions _cwh_ and sufficiently large
radii _ϵ_ ( _τ_ ), this is well approximated by the ball volume Vol _Bcwh_ [ _·, ϵ_ ( _τ_ )] (see Appendix C).


**Bound 3:** **Gray image, PSNR constraint (high PSNR, volume approximation).** The capacity of a
gray image _**x**_ _g_ under a high minimum PSNR threshold _τ_ is approximately log2 Vol _Bcwh_ [ _·, ϵ√_ ( _τ_ )].
**Bound validity:** When the ball is fully inside the cube, i.e. _ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 (i.e., _τ_ _≥_ 20 log10(2 _cwh_ ))

and _ϵ_ ( _τ_ ) large enough for accurate volume approximation (see Bound 8 for small _ϵ_ ).


For small radii the volume approximation becomes inaccurate. However, then there are relatively
few integer points in the ball and we can explicitly count them, as long as the dimension _cwh_ of the
ambient space is not too high. Instead of brute-force enumeration (which scales poorly), we use a
method introduced by Mitchell (1966) leveraging symmetries for efficient counting (see Algorithm 2).


3


(a) Cube fully inside sphere
(low PSNR, Bound 2)


(b) Sphere fully inside cube
(high PSNR, Bounds 3 and 4)


(c) Non-trivial intersection
(medium PSNR, Bounds 6 and 5)


(d) Sphere at corner, i.e.,
arbitrary image (Section 2.4)


Figure 2: **The box-ball configurations of the PSNR-only constraint.** The cube _CI_ is the set of all
images and the sphere is the PSNR ball centred at the cover. Their intersection determines the set of
feasible watermarked images, with the cardinality being the watermarking capacity. **(a)**, **(b)** and **(c)**
are the cases with the cover image _**x**_ at the centre of the cube _CI_ (gray image, resulting in highest
capacity, Section 2.3). **(d)** is the case of the worst-case cover _**x**_, i.e., at the corner of _CI_ (Section 2.4).


Bound 1 simply states that the maximum number of embeddable bits is the uncompressed size of the
image in bits. We next introduce imperceptibility and robustness to measure their effect on capacity.


2.3 CAPACITY UNDER A PSNR CONSTRAINT


A standard way to quantify distortion is the _peak_ _signal-to-noise_ _ratio_ _(PSNR)_, measured in dB.
Requiring a minimum PSNR _τ_ between the cover _x_ and the watermarked image _x_ ˜ is equivalent to
bounding their _ℓ_ 2 distance (see App. B for the full derivation):


_√_
PSNR( _x,_ ˜ _x_ ) _≥_ _τ_ _⇐⇒_ _∥x −_ _x_ ˜ _∥_ 2 _≤_ _ϵ_ ( _τ_ ) _,_ with _ϵ_ ( _τ_ ) = _ρ_ _cwh_ 10 _[−][τ/]_ [20] _._ (1)


|Bou|nd 2 B<br>B|Bo ound<br>6|und 1|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
||ound<br>|5<br>|||||
|||Bo|||||
|||u|d 3||Boun|d 4|
||||||||


|1 No PSN<br>2 5 Gray<br>6 Gray im|R constraint (total capa<br>image (low / medium /<br>age (medium PSNR)|city)<br>high PSN|R)|
|---|---|---|---|
|<br>7 9  Corn<br>und 9|<br>  er image (low / medium|/ high P|SNR)|
|<br>7 9  Corn<br>und 9||||
|Bo<br>||||
|und 7||Boun|d 8|
|||||


As shown in Fig. 3 left, this simple upper Bound 6 closely tracks the exact Bound 5. Thus Bound 6
is the practical choice going forward, while Bound 5 is provided in the appendix for completeness.
Figure 3 left illustrates all the bounds from this section for a 16 _×_ 16px image. At 45 dB these
bounds give us roughly 2000 bits of capacity (more than 2 _._ 5 bpp): orders of magnitude more than the
0 _._ 001 bpp we see in practice (Figure 1).


2.4 FROM CENTRAL GRAY IMAGE TO ARBITRARY COVER IMAGES

In Section 2.3 we assumed the cover lies at the centre of the pixel range, thereby maximizing the
volume of the intersection between the PSNR ball and the cube _CI_ . Real images, however, may be
anywhere in _CI_ . Being at the corner of _CI_ minimizes overlap with the ball and thus provides a lower
bound valid for any image. When _ϵ_ is not too large, exactly [1] _/_ 2 _[cwh]_ of the PSNR ball centred at a
corner of _CI_ remains inside _CI_ . Although this may seem drastic, the penalty is in fact modest: at
most _cwh_ bits, i.e., one bit per pixel. In Appendix F we provide the formal bounds for this corner
setting. Bound 7 adapts Bound 3, the volume approximation when the ball is fully in the cube.
Bound 8 is the analogue of Bound 4, i.e., exact counting for small _ϵ_ ( _τ_ ). Bound 9 parallels Bound 5
for the case when numerical integration is needed. As shown in Figure 3, the gap from the gray-only
image bounds is at most 1 bpp, thus: **Watermarking with a PSNR constraint should allow for**
**capacity upwards of 2 bpp and does not explain the low capacities we observe in practice.**


4


8


6


4


2


0


5 10 15 20 25 30 35 40 45 50 55 60 65 70
PSNR [dB]


6000


5000


4000


3000


2000


1000


0


5 10 15 20 25 30 35 40 45 50 55 60 65 70
PSNR [dB]


Figure 3: **Watermarking capacity under PSNR constraints for a 16** _×_ **16px 3-channel cover image without**
**robustness requirements.** We show the case where the cover lies at the centre of the pixel range (left, Sections 2.2
and 2.3) and at the extreme corner where pixels are saturated (right, Section 2.4). In both cases we plot the
family of bounds we established ranging from the trivial total capacity (Bound 1) to volume approximations
(Bounds 3 and 7), exact lattice counts (Bounds 4 and 8), and numerical integration for partial overlap (Bounds 5
and 9). In the right panel, the arbitrary cover image case lies below the centred gray image case by at most
_cwh_ bits i.e., at most one bit per pixel (1 bpp). The discontinuity between Bounds 7 and 8 is due to the volume
approximation undercounting the integer points on the faces of the cube.


**Bound 4:** **Gray image, PSNR constraint (high PSNR, exact count).** For small _ϵ_ ( _τ_ ) the capacity is


capacity[in bits] = log2 PointsInHypersphereMitchell(dim = _cwh,_ radius = _ϵ_ ( _τ_ )) _._


_√_
**Bound validity:** When _ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 (i.e., _τ_ _≥_ 20 log10(2 _cwh_ )) and _ϵ_ ( _τ_ ) small enough that exact

counting is computationally feasible.


We use Bound 4 whenever we can evaluate Algorithm 2 in reasonable time, and otherwise Bound 3.
As shown in Figure 7, the transition between the two regimes is smooth.


2.3.3 NON-TRIVIAL INTERSECTION (MEDIUM PSNR)
For intermediate PSNR values _τ_, _Bcwh_ [ _**x**_ _g, ϵ_ ( _τ_ )] and _CI_ intersect non-trivially. We can approximate
this count by the volume of the intersection, using the same volume-based method as in Bound 3. One
can use exact volume computation (see Bound 5 in Appendix E), though this tends to be numerically
unstable. In practice, a simpler upper bound approximates it well:


**Bound 6:** **Gray image, PSNR constraint (medium PSNR, approximation).** The capacity of a gray
image under minimum PSNR _τ_ is upper-bounded by min[Bound 2 _,_ Bound 3].
_√_ _√_
**Bound validity:** When _[ρ]_ _/_ 2 _≤_ _ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 _cwh_, or equivalently 20 log 2 _≤_ _τ_ _≤_ 20 log (2 _cwh_ ).


_√_
_cwh_, or equivalently 20 log10 2 _≤_ _τ_ _≤_ 20 log10(2


_cwh_ ).


|Col1|No Attacks (Bounds 2-5)<br>Crop&Rescale scale=0.25|Col3|Col4|No Attacks (Bounds 2-5)<br>Rotation 15deg|Col6|Col7|Col8|No Attacks (Bounds 2-5)<br>linJPEG q=8|Col10|
|---|---|---|---|---|---|---|---|---|---|
|<br> <br>|<br>Crop&Rescale scale=0.5<br>~~Crop&Rescale scale=0.75~~|||<br>Rotation 30deg<br>~~Rotation 45deg~~|<br>Rotation 30deg<br>~~Rotation 45deg~~|||<br>linJPEG q=10<br>~~linJPEG q=15~~|<br>linJPEG q=10<br>~~linJPEG q=15~~|
|||||||||||
|||||||||||
|||||||||||
|||||||||||
|||||||||||


Figure 4: **Impact** **of** **robustness** **constraints** **on** **watermarking** **capacity.** Capacity (in bits per
pixel) under a PSNR constraint is shown for three families of transformations: _Crop&Rescale_ (left),
_Rotation_ (center), and _Linearized JPEG_ (right), using the heuristic bounds (Bounds 10 to 12). The red
lines show the PSNR-only capacity bounds without robustness constraints Bounds 2 to 4 and 5. Each
transformation reduces capacity in proportion to its severity: smaller crop scales, larger rotations, or
lower JPEG quality factors. Across all cases, robustness constraints reduce but do not eliminate the
large theoretical capacity gap with current watermarking methods.


2.5 ADDING ROBUSTNESS CONSTRAINTS


In practice, watermarking must balance imperceptibility with robustness: the message should survive
common processing, like compression, resizing, cropping, rotation, etc. In our model, we consider
linear transformations, which encompass most transformations used in practice. We also develop
LinJPEG, a linearized version of JPEG, allowing us to study the effects of compression in the same
setting (see Appendix G.4 for the construction). Take a linear transformation _M_ _∈_ R _[cwh][×][cwh]_ that
maps an image _**x**_ to a transformed _M_ _**x**_ and a quantization operation Q (element-wise rounding or
floor operation) to map the pixel values of _M_ _**x**_ to the valid images _I_ . Hence, we have the final
transformed image _**x**_ _[′]_ = Q[ _M_ _**x**_ ]. We need to find the subset of the possible watermarked images
under only the PSNR constraint that map to unique valid images after applying _M_ and Q to them.
The main complication in this setup is that Q is non-linear.


**Heuristic bounds.** A simple approach is to take a volumetric approach akin to Bounds 3, 5, 7 and 9.
We factor in how _M_ changes the volume and account for directions compressed by the transformation
which destroy capacity as different watermarked images get collapsed together. We also account for
directions fully collapsed by _M_ when it is singular. Finally, the stretched directions might result in
some watermarked images being outside _CI_ after the transformation, leading to them being clipped.
Bounds 10 to 12 use a heuristic based on the singular values of _M_ to account for the effect on
capacity. Refer to Appendix G.2 for details. In Figure 4 we plot these bounds for robustness to
rotation, cropping followed by rescaling and LinJPEG, showing that even under the most aggressive
cropping, we should expect around 0 _._ 5 bpp or almost 100 _,_ 000 bits for 256 _×_ 256px images.


**Conservative bounds.** We can show cases where these heuristic bounds under-approximate and cases
where they over-approximate the true capacity, e.g., Figures 8 and 9. Thus, the true capacity under
linear transformation could be much lower than these bounds predict. To ensure that this is not the
case, we develop an actual lower bound: Bound 13. While we reserve the details for Appendix G.3,
this bound is based on over-approximating the set of images that can be quantized by Q to the same
image after _M_ is applied to them. As a result, Bound 13 is extremely conservative and unrealistic.
We believe that despite Bounds 10 to 12 not being valid lower bounds, they are much closer to the
true capacity. Still, we report the conservative bound in Table 2: the most aggressive crop still leaves
at least 904 bits for 256 _×_ 256px images. For the other augmentations, the conservative capacity is
much higher. Therefore, **robustness to geometric transformations and compression significantly**
**reduces the capacitybut cannot fully explain the low watermarking capacity of current models.**


2.6 FROM SINGLE COVER IMAGES TO DATASETS AND DATA DISTRIBUTIONS

In a blind watermarking setup, the decoder must operate without access to the original cover image,
creating potential collisions: if multiple natural images (i.e., potential covers) are very close to each
other in pixel space, a watermarked version of one cover could be identical to a watermarked version
of another. To prevent such ambiguity, the total set of watermarked images within a given region


5


Capacity under PSNR constraint

and Crop&Rescale transforms


Capacity under PSNR constraint

and Rotation transforms


Capacity under PSNR constraint

and linJPEG transforms


8


7


6


5


4


3


2


1


0


10 20 30 40 50 60 70
PSNR [dB]


6000


5000


4000


3000


2000


1000


0


10 20 30 40 50 60 70
PSNR [dB]


10 20 30 40 50 60 70
PSNR [dB]


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|128<br>256<br>512<br>|bit models<br> bit models<br> bit models<br>||||


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
|128 bit mo<br>256 bit mo<br>512 bit mo<br>|dels<br>  dels<br>  dels<br>|||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|1<br>|024 bit<br>|models<br>|<br>|||


Figure 5: **Video Seal fails to learn how to embed** 1024 **bits into a gray image with only PSNR**
**constraint,** **whereas** **a** **linear** **embedder** **and** **extractor** **learn** **how** **to** **embed** 2048 **bits.** _(Left.)_
Video Seal trained on a single solid gray image with only the detector and MSE losses. It learns to
embed up to 512 bits but fails to embed 1024 bits. _(Centre.)_ Same setup as the left plot but trained
on a reduced 32 _×_ 32px resolution. The performance is similar; hence, the Video Seal architecture
fails to make use of the full resolution. _(Right.)_ Replacing the embedder and decoder of Video Seal
with a single linear layer each achieves 100% bit accuracy and PSNR above 40 dB for both 1024 and
2048 bits: demonstrating that Video Seal indeed has structural limitations. The results for the sweeps
over the learning rate and _λi_ are shown, with the best models highlighted.


(like the PSNR ball) must be partitioned among all the potential covers it contains. If there are
_N_ possible covers, the capacity for each is reduced by log2( _N_ ) bits. We estimate _N_ using neural
compression models like VQ-VAE (Van Den Oord et al., 2017) and VQGAN (Esser et al., 2021),
which upper-bound the number of perceptually distinct images. For instance, a 256 _×_ 256px image
can be compressed into a 32 _×_ 32 latent with a 1024-entry codebook (Muckley et al., 2023). This
representation can express at most 1024 [32] _[×]_ [32] = 2 [10240] distinct images. Conservatively assuming all
could fall in the PSNR ball of the considered image, capacity is reduced by 10 _,_ 240 bits, or about
0 _._ 05 bpp, on top of the 1 bpp loss from Section 2.4. Thus, from this perspective, **the data distribution**
**has only a negligible effect on watermarking capacity and cannot explain the low performance**
**of current models.** This aligns with prior findings for Gaussian channels that decoder knowledge of
the cover does not affect capacity (Costa, 1983; Chen and Wornell, 2002; Moulin and O’Sullivan,
2003).


3 EMPIRICAL PERFORMANCE IS MUCH LOWER THAN PREDICTED


Section 2 showed that capacities of over 2 bpp at PSNR of 40 dB without robustness constraints, and
of 0 _._ 5 bpp with robustness, are possible. Even under the very conservative Bound 13 we still would
expect capacities of at least 0 _._ 01 bpp. However, in practice, the models reported in the literature have
significantly lower capacities (less than 0 _._ 001 bpp, Figure 1). To understand the cause of this gap,
this section asks: **are existing models significantly under-performing relative to what is possible**
**in practice,or are our bounds too unrealistic?** There are five possible explanations of the large
discrepancy between the performance we see in practice (Figure 1) and the bounds in Section 2:


_**A.**_ **Real models might be near-optimal if we consider advanced robustness constraints.**
_**B.**_ **Real models might be near-optimal if we consider advanced perceptual constraints.**
_**C.**_ **Real models might be near-optimal if we consider real-world image distributions.**
_**D.**_ **Our bounds overestimate capacity and cannot be approached empirically.**

_**E.**_ **We can do much better and push the Pareto front well beyond the current state-of-the-art.**


To understand the cause of the gap between theoretical and real-world performance in image watermarking, we need to find out which of these hypotheses is the underlying cause. If it is _**A.**_, _**B.**_, _**C.**_, _**D.**_,
or a combination of them, then it is possible that, indeed, the best current models are close to what is
ultimately possible and we can expect only marginal further improvements. On the other hand, if the
cause is _**E.**_, then that means that there is plenty of space for significant improvements.


3.1 THE REAL-WORLD COMPLEXITY DOES NOT EXPLAIN THE PERFORMANCE GAP


Let’s first address cases _**A.**_, _**B.**_, _**C.**_, i.e., that our bounds cannot capture the complexity of the robustness,
quality and data constraint with which real models are trained. While we cannot bring the real-world
complexity to our analytical bounds, we can bring the models to the simplified theoretical setup.


6


Gray image + PSNR constraint (256x256px)

VideoSeal embedder and extractor


Gray image + PSNR constraint (256x256px)

Linear embedder and extractor


Gray image + PSNR constraint (32x32px)

VideoSeal embedder and extractor


25 30 35 40 45 50
PSNR [dB]


100


90


80


70


60


50


100


90


80


70


60


50


30 35 40 45 50 55
PSNR [dB]


5 10 15 20 25 30 35 40 45
PSNR [dB]


100


90


80


70


60


50


More concretely, we take the simplest of setups: a single gray image with a PSNR constraint, as
in Section 2.3. We will use Video Seal as the base for our experiments (Fernandez et al., 2024),
originally introduced as an image watermarking model with frame copying that generalizes to video.
[It was first demonstrated with a 96-bit capacity and was recently extended to a 256-bit open-source](https://github.com/facebookresearch/videoseal)
[version,](https://github.com/facebookresearch/videoseal) which we use as the strongest available baseline. To match the setup of Section 2.3, we
replace the dataset with a single solid gray image, remove all perceptual constraints but the MSE loss
and remove all augmentations. We first retrain it for _n_ bits = 128, 256, 512, and 1024 bits. We have
hereby reduced the task to simply find a way to encode _n_ bits into a single fixed image. From Figure 3
we expect capacities of around 600 _,_ 000 bits at 40 dB in this setup. Thus, the model should easily
learn these much lower _n_ bits. We train with AdamW (Loshchilov and Hutter, 2019) with batch size
256 for 600 epochs, 1000 batches per epoch, cosine learning rate schedule with a 20-epoch warm-up,
similarly to Video Seal. We sweep over the learning rate (5e-4, 5e-5, 5e-6) and _λi_, the MSE loss
weight (0.1, 0.5, 1.0), with LR=5e-5 and _λi_ = 0 _._ 5 being the values used for training Video Seal.


The results of training Video Seal on a single gray image can be seen in Figure 5 left and Table 1.
There are runs for the 128, 256 and 512 bit models that do achieve 100% bit accuracy and PSNR
values above 42dB. However, Video Seal cannot even get to 1024 bits, far from what we expect from
the bounds. This is surprising: the model cannot approach the theoretical bounds even after removing
the complexities that supposedly make watermarking difficult. This means that neither _**A.**_, _**B.**_ nor _**C.**_
can explain why we see such a gap between the theoretical and real-world performance.


3.2 OUR SIMPLEST BOUNDS ARE ACHIEVABLE, YET MODELS STRUGGLE TO GET NEAR THEM


Section 3.1 showed that Video Seal cannot match the capacity predicted by the bounds in Section 2.3
even when trained only on a single gray image and with no augmentations. Thus, the complexity of
real world watermarking cannot explain the gap between the theoretical and real-world performance.
This leaves us with two options: _**D.**_ our bounds are wrong and unachievable, or _**E.**_ our models are
under-performing. There are a couple simple experiments that can demonstrate that we can get much
closer to the bounds in Section 2.3 and hence _**D.**_ also does not explain the gap.


**Linear embedder and extractor.** We trained a simple linear embedder and extractor. The embedder
gets the 1024 bit message (shifted and scaled to _−_ 1 and +1 values) and produces a 256 _×_ 256 _×_ 3
watermark residual which gets added to the original gray image. Similarly, the decoder is a linear
layer from the flattened 256 _×_ 256 _×_ 3 image to 1024 outputs, which are thresholded to recover the
message. We train only for 50 epochs, with the same learning rate values and _λi_ _∈{_ 4 _,_ 8 _,_ 12 _,_ 20 _}_ .


7


Table 1: **Video Seal fails to learn how to embed 1024 bits into a gray image with only PSNR**
**constraint while a linear embedder and extractor learn how to embed 2048 bits.** Numerical
results for the best-performing runs from Figure 5 and their respective hyperparameters, as well as
the handcrafted embedder/decoder from Equation (2) at four representative PSNR values.


Message size if
Message size tiled to 256x256px PSNR Bit acc. _λi_ lr


VideoSeal (256x256px, 600 epochs)


VideoSeal (32x32px, 600 epochs)


128 bits 53.45 dB 100.00% 0.5 5e-4
256 bits 53.98 dB 100.00% 1.0 5e-4
512 bits 51.45 dB 100.00% 0.5 5e-4
1024 bits 40.10 dB 89.63% 1.0 5e-5


128 bits 8192 bits 51.02 dB 100.00% 1.0 5e-4
256 bits 16384 bits 48.98 dB 100.00% 1.0 5e-4
512 bits 32768 bits 41.66 dB 100.00% 1.0 5e-5
1024 bits 65536 bits 29.66 dB 84.39% 0.1 5e-5
1024 bits 65536 bits 33.20 dB 83.86% 0.5 5e-5
1024 bits 65536 bits 34.63 dB 83.78% 1.0 5e-5
1024 bits 65536 bits 50.83 dB 50.60% 0.5 5e-4


1024 bits 44.28 dB 100.00% 20.0 5e-4
Linear (256x256px, 50 epochs)
2048 bits 40.40 dB 100.00% 12.0 5e-4


Handcrafted


623232 bits 36.00 dB 100.00%
551948 bits 38.00 dB 100.00%
456509 bits 42.00 dB 100.00%
311616 bits 48.00 dB 100.00%


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||||||||||
|B|B|ounds 2-5|||||||
|H<br>Vi<br>Vi<br>|H<br>Vi<br>Vi<br>|andcrafted<br>deoSeal (25<br>deoSeal (32<br>|model<br> 6x256px)<br> x32px) Tile|d|||||
||~~Li~~|~~near~~|||||||


can fit in this ball is _d_ = _ϵ_ ( _τ_ ) _/_ _cwh_ = _ρ_ 10 _[−][τ/]_ [20] . We have that each edge of the box contains a

_cwh_ -dimensional grid of _q_ = 2 _⌊d⌋_ + 1 = 2 _⌊_ 2 _[k]_ 10 _[−][τ/]_ [20] _⌋_ + 1 points per side. Hence, that gives us


Table 2: **Conservative** **capacity**
**bounds** **under** **robustness** **con-**
**straints** **for** **PSNR** **42** **dB.** These
values are calculated via Bound 13
and are strongly conservative lower
bounds on the capacity that is
achievable while maintaining robustness to the respective transformations and PSNR under 42 dB.


10 [1]


Conservative capacity
Augmentation bpp for 16 _×_ 16px for 256 _×_ 256px


Horizontal Flip 3.064 2,352 bits 602,353 bits
Crop&Rescale 50% 0.015 11 bits 3,013 bits
Crop&Rescale 75% 0.005 3 bits 904 bits
LinJPEG q=10 0.136 104 bits 26,757 bits
LinJPEG q=15 0.137 105 bits 27,020 bits
Rotation 30deg 0.075 57 bits 14,676 bits
Rotation 45deg 0.083 64 bits 16,401 bits


10 [0]


10 [1]


10 [2]


10 [3]


10 20 30 40 50 60 70
PSNR [dB]


Figure 6: **Simple** **models** **outperform**
**Video Seal** **on** **a** **gray** **image** **with** **only** **a**
**PSNR** **constraint.** Experiments from Section 3 compare our theoretical bounds (Section 2.3) against trained models. Video Seal
falls well below the predictions, while a linear model performs slightly better and a tiled
32 _×_ 32px Video Seal is even better. Our handcrafted model nearly matches the bound.


The results in Figure 5 right and Table 1 show the linear layer learns what Video Seal could not: 100%
bit accuracy for 1024 bits with PSNR of 44 dB. We also trained a linear model for 2048 bits which
achieved 100% bit accuracy. This shows that capacities beyond 512 bits are possible in practice (at
least for a gray image and no robustness) and are learnable via gradient descent. All one needs is the
right architecture.


**Lower-resolution training and tiling.** Our experiments reveal that Video Seal does not exploit the
additional degrees of freedom available at higher image resolutions. When trained at 256 _×_ 256px, the
model achieves essentially the same capacity and PSNR as when trained at 32 _×_ 32px (see Section 3.1
and Table 1). To verify this, we train Video Seal in the setup of Section 3.1 at 32 _×_ 32px using the
same learning-rate and _λi_ sweeps for 600 epochs. As shown in Figure 5 (centre) and Table 1, the
performance at 32 _×_ 32px is nearly identical to that at 256 _×_ 256px: the 512-bit model reaches 100% bit
accuracy with 41 _._ 7 dB, despite operating on 64 _×_ fewer pixels. In other words, the effective capacity
we observe at 256 _×_ 256px is comparable to what one would expect around 20 _×_ 20px, confirming that
the architecture fails to utilize the available resolution.


Because this setup does not require robustness to geometric or valuemetric transformations and we
consider only gray images, we can use the 32 _×_ 32px model to demonstrate that higher capacities are
possible. A simple tiling strategy suffices: each tile is embedded with an independent secret using
the same model. The decoder similarly is applied per patch with the individual decoded messages
concatenated to obtain the final combined message. Using 256 _×_ 256px as the reference size, tiling
yields 64 _×_ the capacity of the base 32 _×_ 32px model. Thus, tiling the 512-bit model—which already
achieves 100% accuracy at 41 _._ 7 dB—produces a watermark with 32 _,_ 768 bits total capacity while
maintaining the same PSNR (which is resolution-independent). This effective capacity of 32 _,_ 768 bits
is already much closer to our bound of roughly 600 _,_ 000 bits, though still only about 0 _._ 167 bpp.


It is interesting that the model could not learn at the 256 _×_ 256px resolution even for 1024 bits when
it is clear that it is possible to embed 32,768 bits as seen here. More importantly, this shows that our
bounds are not that far off and capacities of at least 32,768 bits are indeed possible.


**Handcrafted embedder and extractor.** We can do even better by manually crafting an embedder
and extractor. The key observation is that mapping a hypercube to binary messages is easy. Take
_√_
the ball of radius _ϵ_ ( _τ_ ) = _ρ_ _cwh_ 10 _[−][τ/]_ [20] from Equation (1). The half-side of the largest cube that


the ball of radius _ϵ_ ( _τ_ ) = _ρ_ _cwh_ 10 _[−][τ/]_ [20] from Equation (1). The half-side of the largest cube that

_√_
can fit in this ball is _d_ = _ϵ_ ( _τ_ ) _/_ _cwh_ = _ρ_ 10 _[−][τ/]_ [20] . We have that each edge of the box contains a


8


or log2 _q_ bits per pixel. See Figure 6 for a plot of that for different PSNR values. For 42 dB, and
images of 256 _×_ 256px that gives us a capacity of 456 _,_ 509 bits (see Table 1) almost 14 _×_ what we
could embed with the 32 _×_ 32px tiling approach. Moreover, it gets us close to the theoretical bound.


Therefore, we can get much closer to the boundary, at least in the solid gray image case with PSNR
constraint and no robustness requirements. Thus, case _**D.**_, that our bounds are wrong and impossible
to achieve, is unlikely. This leaves us with one possible explanation as to why models in practice
do not exhibit performance anywhere near what our theory predicts. That would be option _**E:**_ **Our**
**models** **are** **likely** **significantly** **underperforming** **relative** **to** **what** **is** **possible** **in** **practice.** **We**
**likely can do much better and push the Pareto front well beyond the current state-of-the-art.**


4 BETTER PERFORMANCE IN PRACTICE IS POSSIBLE: CHUNKY SEAL


While it remains possible that current models approach a theoretical limit under robustness and
quality constraints, training a watermarking model with comparable quality and robustness but with
substantially higher capacity would decisively rule this out. We take Video Seal (Fernandez et al.,
2024) as the base model and train it for 1024 bits. We increased the embedding dimension to 2048,
the U-Net channel multipliers from [1 _,_ 2 _,_ 4 _,_ 8] to [4 _,_ 8 _,_ 16 _,_ 32], and enabled watermarking in all three
channels, not just the luma (Y) channel. This results in an embedder 90 _×_ larger than the original
Video Seal embedder. The ConvNeXt (Liu et al., 2022) extractor was similarly scaled: we increased
the depths for each stage from [3 _,_ 3 _,_ 9 _,_ 3] (as in ConvNeXt-tiny) to [3 _,_ 3 _,_ 27 _,_ 3] (as in ConvNeXt-base),
with their dimensions increased from [96 _,_ 192 _,_ 384 _,_ 768] to [256 _,_ 512 _,_ 1024 _,_ 2048]. The stride of the
first layer was reduced from 4 to 2. This results in an extractor that is 23 _×_ larger than the original
Video Seal extractor. Due to its significantly increased size, we name this model Chunky Seal. We
train it at the original 256 _×_ 256px resolution. We apply gradient clipping with a maximum norm of
0.01, which proved critical for stabilizing training.


As shown in Table 3, Chunky Seal shows image quality and robustness comparable to Video Seal
across a wide range of distortions, while providing a **4** _**×**_ **higher message capacity** (1024 vs. 256 bits).
Despite its much larger capacity, Chunky Seal maintains nearly identical image quality across all
metrics, and only slightly higher LPIPS. The robustness results further confirm that Chunky Seal
sustains high bit-accuracy across transformations such as rotation, resizing, cropping, brightness
and contrast changes, JPEG compression, and blurring, closely matching Video Seal. We emphasize
that these results were achieved _without hyperparameter tuning_, whereas Video Seal was extensively
optimized for quality and robustness. **Achieving** **4** _**×**_ **the** **capacity** **per** **pixel** **with** **comparable**
**robustness** **and** **quality** **through** **simple** **scaling** **strongly** **suggests** **that** **substantially** **higher**
**capacities are within reach using improved architectures and training strategies.**


9


Table 3: **Chunky Seal performance**
**on images from SA-1B (Kirillov et al.,**
**2023)** **at** **their** **original** **resolution.**
Chunky Seal has much higher capacity
(1024 bits) than Video Seal while preserving its image quality and robustness
on a wide variety of transformations.
The improvement is driven by scaling
the model size and its training. Extended results on SA-1B (Kirillov et al.,
2023) and COCO (Lin et al., 2014) as
well as qualitative results, are reported
in Appendix J.1


total capacity in bits of


**Chunky Seal (ours)** **Video Seal 256bits**


1024 bits 256 bits
Capacity
0.0052 bpp 0.0013 bpp


Embedder size 1022.7M 11.0M
Extractor size 773.7M 33.0M


PSNR _↑_ 45.32 _±_ 2.16 44.42 _±_ 2.21
SSIM _↑_ 0.995 _±_ 0.006 0.996 _±_ 0.003
MS-SSIM _↑_ 0.997 _±_ 0.002 0.997 _±_ 0.001
LPIPS _↓_ 0.0085 _±_ 0.0067 0.0019 _±_ 0.0011


Bit acc. Identity 99.74 _±_ 0.28% 99.90 _±_ 0.21%
Bit acc. Flip 99.65 _±_ 0.34% 99.89 _±_ 0.24%
Bit acc. Rotate ( _≤_ 10°) 98.27 _±_ 2.10% 98.84 _±_ 1.10%
Bit acc. Resize (71–95%) 99.74 _±_ 0.28% 99.90 _±_ 0.21%
Bit acc. Crop (77–95%) 98.25 _±_ 1.75% 98.04 _±_ 1.57%
Bit acc. Brightness (0.5–1.5×) 98.99 _±_ 1.87% 98.67 _±_ 2.67%
Bit acc. Contrast (0.5–1.5×) 99.54 _±_ 0.51% 99.56 _±_ 0.45%
Bit acc. JPEG (Q 50–80) 98.79 _±_ 0.75% 99.74 _±_ 0.47%
Bit acc. Gaussian Blur (k _≤_ 9) 99.74 _±_ 0.28% 99.90 _±_ 0.22%


Bit acc. Overall 99.15 _±_ 0.63% 99.31 _±_ 0.60%


log2


�� - - _cwh_ [�] - - 2 2 _[k]_ 10 _[−][τ/]_ [20][�] + 1 = _cwh_ log2 2 2 _[k]_ 10 _[−][τ/]_ [20][�] + 1 = _cwh_ log2 _q,_ (2)


5 DISCUSSION AND CONCLUSIONS


Higher watermarking capacities open up new avenues for content provenance. Instead of using a
watermark to retrieve a C2PA manifest from a third-party database (Collomosse and Parsons, 2024),
we could embed the entire manifest, eliminating the need for a registry. Beyond this, improvements
in capacity can be traded for greater robustness or higher image quality, depending on the application.
The fact that our theoretical capacity bounds are an order of magnitude higher than even the best
existing models also helps explain why applying and detecting multiple watermarks on a single image
is feasible, as demonstrated by (Petrov et al., 2025).


Despite achieving substantially higher capacities than prior models, Chunky Seal still remains far
from the theoretical bounds established in this work. Our controlled experiments show that this gap
cannot be attributed to factors such as data distribution, resolution, or augmentations. Instead, the
evidence consistently points to limitations in the model architecture itself. Learning an identity map
is notoriously difficult for neural networks (He et al., 2016; Hardt and Ma, 2017), a point underscored
by the fact that simple linear models outperform Video Seal in settings where the architecture should,
in principle, excel. Importantly, we do not suggest that na¨ıvely scaling Chunky Seal is a practical
path forward. The purpose of this scaling exercise was to explore feasibility, not to advocate for
large models in deployment. These results simply illustrate that current architectures fall well short
of saturating watermarking capacity, even under generous scaling. Looking ahead, we argue that
substantial progress will require new architectural designs, improved losses, and revised training
procedures that better encode the inductive biases inherent to watermarking, rather than further
scaling of existing models.


We therefore propose a set of sanity checks for the next generation of watermarking methods.
A principled approach should scale capacity linearly with image size, decrease capacity linearly
with higher PSNR, outperform simple linear or handcrafted baselines, and show predictable drops
under stronger augmentations (e.g., 4 _×_ lower capacity for a 25% crop). These are necessary for
Pareto-optimality and can steer the community toward watermarks with far higher capacity or quality.


Our analysis is not without limitations. We restricted our study to image watermarking, though the
insights likely carry over to video. Theoretical bounds are derived only for analytically tractable setups,
with some cases relying on numerical integration that becomes impractical at higher resolutions.
Our robustness bounds are heuristic rather than formal, leaving ample room for sharper theoretical
advances. Finally, while Chunky Seal delivers clear performance gains, its size and latency highlight
the need for future architectures that deliver both higher capacities _and_ efficiency.


REFERENCES


Ali Al-Haj. 2007. Combined DWT-DCT digital image watermarking. _Journal of Computer Science_,
3(9):740–746.


Matthias Althoff, Olaf Stursberg, and Martin Buss. 2010. Computing reachable sets of hybrid
systems using a combination of zonotopes and polytopes. _Nonlinear analysis:_ _Hybrid systems_,
4(2):233–249.


Yoshinori Aono and Phong Q Nguyen. 2017. Random sampling [revisited:](https://eprint.iacr.org/2017/155.pdf) Lattice enumeration
with [discrete](https://eprint.iacr.org/2017/155.pdf) pruning. In _Annual_ _International_ _Conference_ _on_ _the_ _Theory_ _and_ _Applications_ _of_
_Cryptographic Techniques_, pages 65–102.


Mauro Barni, Franco Bartolini, and Alessandro Piva. 2001. Improved wavelet-based watermarking
through pixel-wise masking. _IEEE transactions on image processing_, 10(5):783–791.


Patrick Bas, J-M Chassery, and Benoit Macq. 2002. Geometrically invariant watermarking using
feature points. _IEEE transactions on image Processing_, 11(9):1014–1028.


Adrian G Bors and Ioannis Pitas. 1996. Image watermarking using DCT domain constraints. In _ICIP_ .


Tu Bui, Shruti Agarwal, and John Collomosse. 2023a. Trustmark: [Universal](https://arxiv.org/abs/2311.18297) watermarking for
[arbitrary resolution images.](https://arxiv.org/abs/2311.18297) _arXiv preprint arXiv:2311.18297_ .


Tu Bui, Shruti Agarwal, Ning Yu, and John Collomosse. 2023b. [RoSteALS: Robust steganography](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Bui_RoSteALS_Robust_Steganography_Using_Autoencoder_Latent_Space_CVPRW_2023_paper.html)
[using autoencoder latent space.](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Bui_RoSteALS_Robust_Steganography_Using_Autoencoder_Latent_Space_CVPRW_2023_paper.html) In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition (CVPR) Workshops_ .


10


Brian Chen and Gregory W Wornell. 2002. [Quantization index modulation:](https://doi.org/10.1109/18.923725) A class of provably good
[methods for digital watermarking and information embedding.](https://doi.org/10.1109/18.923725) _IEEE Transactions on Information_
_theory_, 47(4):1423–1443.


Xiangyu Chen, Varsha Kishore, and Kilian Q Weinberger. 2023. [Learning iterative neural optimizers](https://arxiv.org/abs/2303.16206)
[for image steganography.](https://arxiv.org/abs/2303.16206) In _International Conference on Learning Representations_ .


Hai Ci, Pei Yang, Yiren Song, and Mike Zheng Shou. 2024. [RingID: Rethinking tree-ring watermark-](https://arxiv.org/abs/2404.14055)
[ing for enhanced multi-key identification.](https://arxiv.org/abs/2404.14055) _arXiv preprint arXiv:2404.14055_ .


Aaron S Cohen and Amos Lapidoth. 2002. [The Gaussian watermarking game.](https://doi.org/10.1109/TIT.2002.1003844) _IEEE Transactions on_
_Information Theory_, 48(6):1639–1667.


John Collomosse and Andy Parsons. 2024. To authenticity, and [beyond!](https://doi.org/10.1109/MCG.2024.3380168) Building safe and fair
[generative AI upon the three pillars of provenance.](https://doi.org/10.1109/MCG.2024.3380168) _IEEE Computer Graphics and Applications_ .


Denis Constales. 1997. [Solution to “The volume of the intersection of a cube and a ball in N-space”](https://epubs.siam.org/doi/10.1137/SIREAD000039000004000761000001)
[posed by Liqun Xu.](https://epubs.siam.org/doi/10.1137/SIREAD000039000004000761000001) _SIAM Review (Problems and Solutions)_, 39.4:779–786.


Max Costa. 1983. [Writing on dirty paper.](https://doi.org/10.1109/TIT.1983.1056659) _IEEE Transactions on Information Theory_, 29(3):439–441.


I.J. Cox, J. Kilian, F.T. Leighton, and T. Shamoon. 1997. [Secure spread spectrum watermarking for](https://doi.org/10.1109/83.650120)
[multimedia.](https://doi.org/10.1109/83.650120) _IEEE Transactions on Image Processing_, 6(12):1673–1687.


Patrick Esser, Robin Rombach, and Bjorn Ommer. 2021. [Taming transformers for high-resolution](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html?ref=)
[image synthesis.](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html?ref=) In _Proceedings of the IEEE/CVF conference on computer vision and pattern_
_recognition_ .


Liu Ping Feng, Liang Bin Zheng, and Peng Cao. 2010. [A DWT-DCT based blind watermarking](https://doi.org/10.1109/ICCSIT.2010.5565101)
[algorithm for copyright protection.](https://doi.org/10.1109/ICCSIT.2010.5565101) In _2010 3rd International Conference on Computer Science_
_and Information Technology_, volume 7, pages 455–458.


Pierre Fernandez, Guillaume Couairon, Herve J´ egou, Matthijs Douze, and Teddy Furon. 2023.´ [The](https://openaccess.thecvf.com/content/ICCV2023/html/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.html)
stable signature: [Rooting watermarks in latent diffusion models.](https://openaccess.thecvf.com/content/ICCV2023/html/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.html) In _International Conference on_
_Computer Vision_ .


Pierre Fernandez, Hady Elsahar, I Zeki Yalniz, and Alexandre Mourachko. 2024. [Video Seal:](https://arxiv.org/abs/2412.09492) Open
[and efficient video watermarking.](https://arxiv.org/abs/2412.09492) _arXiv preprint arXiv:2412.09492_ .


Pierre Fernandez, Alexandre Sablayrolles, Teddy Furon, Herve J´ egou, and Matthijs Douze. 2022.´

[Watermarking images in self-supervised latent spaces.](https://arxiv.org/abs/2112.09581) In _ICASSP 2022-2022 IEEE International_
_Conference on Acoustics, Speech and Signal Processing (ICASSP)_ .


Carl Friedrich Gauss. 1837. De nexu inter multitudinem classium, in quas formae binariae secundi
gradus distribuuntur, earumque determinantem. In _Werke:_ _Band 2_, pages 269–291.


S. I. Gel’fand and M. S. Pinsker. 1980. Coding for channel with random parameters. _Problems of_
_Control and Information Theory_, 9(1):19–31.


Antoine Girard. 2005. [Reachability of uncertain linear systems using zonotopes.](https://doi.org/10.1007/978-3-540-31954-2_19) In _Proceedings of_
_the 8th International Conference on Hybrid Systems:_ _Computation and Control_ .


Moritz Hardt and Tengyu Ma. 2017. [Identity matters in deep learning.](https://openreview.net/forum?id=ryxB0Rtxx) In _International Conference_
_on Learning Representations_ .


G. H. Hardy. 1915. On the expression of a number as the sum of two squares. _Quarterly Journal of_
_Mathematics_, 46:263–283.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. [Deep residual learning for image](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
[recognition.](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ .


Seongmin Hong, Kyeonghyun Lee, Suh Yoon Jeon, Hyewon Bae, and Se Young Chun. 2024. [On](https://arxiv.org/abs/2311.18387)
[exact inversion of DPM-solvers.](https://arxiv.org/abs/2311.18387) In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_ .


11


Zhaoyang Jia, Han Fang, and Weiming Zhang. 2021. [MBRS: Enhancing robustness of DNN-based](https://arxiv.org/abs/2108.08211)
[watermarking by mini-batch of real and simulated JPEG compression.](https://arxiv.org/abs/2108.08211) In _Proceedings of the 29th_
_ACM international conference on multimedia_ .


Changhoon Kim, Kyle Min, Maitreya Patel, Sheng Cheng, and Yezhou Yang. 2023. Wouaf: Weight
[modulation for user attribution and fingerprinting in text-to-image diffusion models.](https://arxiv.org/abs/2306.04744) _arXiv preprint_
_arXiv:2306.04744_ .


Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete
Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. 2023. [Segment anything.](https://arxiv.org/abs/2304.02643) In
_Proceedings of the IEEE/CVF International Conference on Computer Vision_, pages 4015–4026.


Varsha Kishore, Xiangyu Chen, Yan Wang, Boyi Li, and Kilian Q Weinberger. 2022. [Fixed neural](https://openreview.net/forum?id=hcMvApxGSzZ)
network steganography: [Train](https://openreview.net/forum?id=hcMvApxGSzZ) the images, not the network. In _International_ _Conference_ _on_
_Learning Representations_ .


Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr
Dollar,´ and C Lawrence Zitnick. 2014. Microsoft COCO: [Common](https://arxiv.org/abs/1405.0312) objects in context. In
_Computer Vision–ECCV 2014:_ _13th European Conference, Zurich, Switzerland, September 6-12,_
_2014, Proceedings, Part V 13_ .


Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie.
2022. [A Convnet for the 2020s.](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) In _Proceedings of the IEEE/CVF Conference on Computer Vision_
_and Pattern Recognition_, pages 11976–11986.


Ilya Loshchilov and Frank Hutter. 2019. [Decoupled weight decay regularization.](https://arxiv.org/abs/1711.05101) In _International_
_Conference on Learning Representations_ .


Xiyang Luo, Ruohan Zhan, Huiwen Chang, Feng Yang, and Peyman Milanfar. 2020. [Distortion](https://arxiv.org/abs/2001.04580)
[agnostic deep watermarking.](https://arxiv.org/abs/2001.04580) In _CVPR_ .


Neri Merhav. 2005. [An information-theoretic view of watermark embedding-detection and geometric](https://web.archive.org/web/20241226112735/https://www.ee.technion.ac.il/people/merhav/papers/p98.pdf)
[attacks.](https://web.archive.org/web/20241226112735/https://www.ee.technion.ac.il/people/merhav/papers/p98.pdf) _Proceedings of WaCha, 2005, First Wavila Challenge_ .


WC Mitchell. 1966. [The number of lattice points in a k-dimensional hypersphere.](https://www.ams.org/journals/mcom/1966-20-094/S0025-5718-1966-0195834-3/S0025-5718-1966-0195834-3.pdf) _Mathematics of_
_Computation_, 20(94):300–310.


Pierre Moulin and Ralf Koetter. 2005. [Data-hiding codes.](https://doi.org/10.1109/JPROC.2005.859599) _Proceedings of the IEEE_, 93(12):2083–
2126.


Pierre Moulin and Joseph A O’Sullivan. 2003. [Information-theoretic analysis of information hiding.](https://doi.org/10.1109/TIT.2002.808134)
_IEEE Transactions on information theory_, 49(3):563–593.


Matthew J. Muckley, Alaaeldin El-Nouby, Karen Ullrich, Herve Jegou, and Jakob Verbeek. 2023.

[Improving statistical fidelity for neural image compression with implicit local likelihood models.](https://proceedings.mlr.press/v202/muckley23a.html)
In _Proceedings of the 40th International Conference on Machine Learning_, pages 25426–25443.


Seung-Min Mun, Seung-Hun Nam, Han-Ul Jang, Dongkyu Kim, and Heung-Kyu Lee. 2017. [A](https://arxiv.org/abs/1704.03248)
[robust blind watermarking using convolutional neural network.](https://arxiv.org/abs/1704.03248) _arXiv preprint arXiv:1704.03248_ .


K. A. Navas, Mathews Cheriyan Ajay, M. Lekshmi, Tampy S. Archana, and M. Sasikumar. 2008.

[DWT-DCT-SVD based watermarking.](https://ieeexplore.ieee.org/abstract/document/4554423) In _2008 3rd International Conference on Communication_
_Systems Software and Middleware and Workshops (COMSWARE ’08)_, pages 271–274.


Zhicheng Ni, Yun-Qing Shi, N. Ansari, and Wei Su. 2006. [Reversible data hiding.](https://doi.org/10.1109/TCSVT.2006.869964) _IEEE Transactions_
_on Circuits and Systems for Video Technology_, 16(3):354–362.


Minzhou Pan, Yi Zeng, Xue Lin, Ning Yu, Cho-Jui Hsieh, Peter Henderson, and Ruoxi Jia. 2024.

[JIGMARK: A black-box approach for enhancing image watermarks against diffusion model edits.](https://arxiv.org/abs/2406.03720)
_arXiv preprint arXiv:2406.03720_ .


Aleksandar Petrov, Shruti Agarwal, Philip HS Torr, Adel Bibi, and John Collomosse. 2025. [On the](https://arxiv.org/abs/2501.17356)
[coexistence and ensembling of watermarks.](https://arxiv.org/abs/2501.17356) _arXiv preprint arXiv:2501.17356_ .


12


Alessandro Piva, Mauro Barni, Franco Bartolini, and Vito Cappellini. 1997. DCT-based watermark
recovering without resorting to the uncorrupted original image. In _Proceedings of international_
_conference on image processing_, volume 1, pages 520–523. IEEE.


Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-Net: [Convolutional networks for](https://arxiv.org/abs/1505.04597)
[biomedical image segmentation.](https://arxiv.org/abs/1505.04597) In _Medical image computing and computer-assisted intervention–_
_MICCAI 2015:_ _18th international conference, Munich, Germany, October 5-9, 2015, proceedings,_
_part III 18_, pages 234–241.


Tom Sander, Pierre Fernandez, Alain Oliviero Durmus, Teddy Furon, and Matthijs Douze. 2025. [Wa-](https://arxiv.org/abs/2411.07231)
termark anything models: [Localized deep-learning image watermarking for small areas, inpainting,](https://arxiv.org/abs/2411.07231)
[and splicing.](https://arxiv.org/abs/2411.07231) _International Conference on Learning Representations_ .


Rolf Schneider. 2013. _Convex_ _bodies:_ _the_ _Brunn–Minkowski_ _theory_, volume 151. Cambridge
University Press.


Anelia Somekh-Baruch and Neri Merhav. 2004. On the capacity game [of](https://doi.org/10.1109/TIT.2004.824920) public watermarking
[systems.](https://doi.org/10.1109/TIT.2004.824920) _IEEE Transactions on Information Theory_, 50(3):511–524.


Matthew Tancik, Ben Mildenhall, and Ren Ng. 2020. StegaStamp: [Invisible hyperlinks in physical](https://openaccess.thecvf.com/content_CVPR_2020/html/Tancik_StegaStamp_Invisible_Hyperlinks_in_Physical_Photographs_CVPR_2020_paper.html)
[photographs.](https://openaccess.thecvf.com/content_CVPR_2020/html/Tancik_StegaStamp_Invisible_Hyperlinks_in_Physical_Photographs_CVPR_2020_paper.html) In _Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_
_Recognition_ .


Aaron Van Den Oord, Oriol Vinyals, et al. 2017. [Neural discrete representation learning.](https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html) _Advances_
_in Neural Information Processing Systems_, 30.


Ron G Van Schyndel, Andrew Z Tirkel, and Charles F Osborne. 1994. A digital watermark. In
_Proceedings of 1st international conference on image processing_, volume 2, pages 86–90. IEEE.


Igor G Vladimirov. 2015. [Quantized linear systems on integer lattices:](https://arxiv.org/abs/1501.04237) A frequency-based approach.
_arXiv preprint arXiv:1501.04237_ .


Vedran Vukotic,´ Vivien Chappelier, and Teddy Furon. 2018. [Are deep neural networks good for](https://doi.org/10.1109/WIFS.2018.8630768)
[blind image watermarking?](https://doi.org/10.1109/WIFS.2018.8630768) In _2018 IEEE International Workshop on Information Forensics and_
_Security (WIFS)_ .


Arnold Walfisz. 1957. _Gitterpunkte in mehrdimensionalen Kuglen_, volume 33 of _Monografie Matem-_
_atyczne_ .


Gregory K. Wallace. 1991. [The JPEG still picture compression standard.](https://doi.org/10.1145/103085.103089) _Communications of the_
_ACM_, 34(4):30–44.


Guanjie Wang, Zehua Ma, Chang Liu, Xi Yang, Han Fang, Weiming Zhang, and Nenghai Yu.
2024. [MuST: Robust image watermarking for multi-source tracing.](https://doi.org/10.1609/aaai.v38i6.28344) In _Proceedings of the AAAI_
_Conference on Artificial Intelligence_ .


Yuxin Wen, John Kirchenbauer, Jonas Geiping, and Tom Goldstein. 2023. [Tree-ring watermarks:](https://arxiv.org/abs/2305.20030)
[Fingerprints for diffusion images that are invisible and robust.](https://arxiv.org/abs/2305.20030) _arXiv preprint arXiv:2305.20030_ .


Wikipedia. 2025. [JPEG — Wikipedia, the free encyclopedia.](https://web.archive.org/web/20250730062355/https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example) [Online; accessed 30-July-2025].


Xiang-Gen Xia, Charles G Boncelet, and Gonzalo R Arce. 1998. Wavelet transform based watermark
for digital images. _Optics Express_ .


Rui Xu, Mengya Hu, Deren Lei, Yaxi Li, David Lowe, Alex Gorevski, Mingyu Wang, Emily Ching,
and Alex Deng. 2025. InvisMark: Invisible and robust [watermarking](https://arxiv.org/abs/2411.07795) for ai-generated image
[provenance.](https://arxiv.org/abs/2411.07795) In _2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)_ .


Aditi Zear, Amit Kumar Singh, and Pardeep Kumar. 2018. A proposed secure multiple watermarking
technique based on DWT, DCT and SVD for application in medicine. _Multimedia_ _tools_ _and_
_applications_, 77(4):4863–4882.


Honglei Zhang, Hu Wang, Yuanzhouhan Cao, Chunhua Shen, and Yidong Li. 2020. [Robust water-](https://arxiv.org/abs/2011.10850)
[marking using inverse gradient attention.](https://arxiv.org/abs/2011.10850) _arXiv preprint arXiv:2011.10850_ .


13


Xuanyu Zhang, Runyi Li, Jiwen Yu, Youmin Xu, Weiqi Li, and Jian Zhang. 2024. [EditGuard:](https://arxiv.org/abs/2312.08883)
[Versatile image watermarking for tamper localization and copyright protection.](https://arxiv.org/abs/2312.08883) In _Proceedings of_
_the IEEE/CVF conference on computer vision and pattern recognition_ .


Jiren Zhu, Russell Kaplan, Justin Johnson, and Li Fei-Fei. 2018. [HiDDeN: Hiding data with deep](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html)
[networks.](https://openaccess.thecvf.com/content_ECCV_2018/html/Jiren_Zhu_HiDDeN_Hiding_Data_ECCV_2018_paper.html) In _Proceedings of the European Conference on Computer Vision (ECCV)_ .


14


where _S_ denotes the cover signal, _U_ an auxiliary variable chosen by the encoder, _X_ the watermarked
image, and _Y_ the attacked image. This framework underlies nearly all classical information-theoretic
analyses of watermarking.


Early works such as Costa (1983) and Cohen and Lapidoth (2002) assume Gaussian cover signals
and Gaussian, memoryless attacks. Under these assumptions, the decoder’s ignorance of the cover
does not reduce capacity: the achievable rate depends only on watermark power and noise power.
Quantization-based analyses (Chen and Wornell, 2002) operate under similarly stylized Gaussian
models.


More sophisticated approaches were introduced by Moulin and O’Sullivan (2003) and Moulin and
Koetter (2005), who cast watermarking as a strategic game between hider and attacker and defined
the notion of _hiding capacity_ . These formulations characterize the fundamental tradeoff between
embedding distortion _D_ 1 and attack distortion _D_ 2, and have inspired a long line of follow-up work


15


A EXTENDED RELATED WORK


**Classic principled methods.** Early research on image watermarking was dominated by hand-crafted
signal processing techniques, grounded in well-understood mathematical and perceptual models.
These methods could operate directly in the pixel domain (Van Schyndel et al., 1994; Bas et al.,
2002), or in transform domains, most commonly the discrete cosine transform (DCT, Bors and Pitas,
1996; Piva et al., 1997) and the discrete wavelet transform (DWT, Xia et al., 1998; Barni et al., 2001),
as well as combinations of the two (Navas et al., 2008; Feng et al., 2010; Zear et al., 2018). A
key insight was that perceptually significant frequencies tend to be preserved under transformations
(Cox et al., 1997). Other schemes, such as (Ni et al., 2006), introduced perturbations to all pixels
with specific values to enable very large payloads. Despite being principled and accompanied by
theoretical guarantees, these methods have limited robustness to perturbations and, in some cases,
cause noticeable image degradation.


**Deep learning based-watermarking.** With the development of deep learning techniques for computer vision, it was only natural to extend them to image watermarking. Vukotic et al.´ (2018) proposed
adversarially attacking a fixed image extractors, an idea later built upon by Fernandez et al. (2022)
and Kishore et al. (2022). A similar iterative approach, albeit lacking robustness, was attempted
for steganography by Chen et al. (2023). LISO operates by iteratively optimizing the embedding of
information into images, but its performance degrades sharply under even mild compression. Notably,
the only robustness-aware variant, LISO-JPEG, achieves merely _∼_ 1 bpp at 19.7 dB PSNR, which
is well below practical watermarking quality and is only robust to JPEG compression with CRF 80.
This demonstrates that LISO is non-practical for real-world watermarking applications due to its
weak robustness and low fidelity.


More popular and successful methods train purpose-built neural networks. Early convolutional
models, such as those introduced by Mun et al. (2017) and Zhu et al. (2018), established the
feasibility of CNN-based watermarking. Subsequently, architectures based on U-Net (Ronneberger
et al., 2015) gained prominence, leveraging multi-resolution representations and residual connections
to enable greater network depths. Recent work has also explored watermarking in the latent space of
diffusion models Bui et al. (2023b). Advances in perceptual loss design have proven critical: careful
loss selection and tuning improved both image quality (Bui et al., 2023a; Fernandez et al., 2024; Xu
et al., 2025) and robustness (Tancik et al., 2020; Jia et al., 2021; Pan et al., 2024). Beyond robustness,
methods for watermark localization and multi-message embedding have been proposed (Sander et al.,
2025; Wang et al., 2024; Zhang et al., 2024), while add-on techniques such as adversarial training (Luo
et al., 2020) and attention mechanisms (Zhang et al., 2020) further enhance performance. Although
combining image generation with watermarking has also been studied (Fernandez et al., 2023; Wen
et al., 2023; Kim et al., 2023; Hong et al., 2024; Ci et al., 2024), such generative approaches are
beyond the scope of this work.


**Information-theoretic** **capacity** **of** **watermarking** A natural foundation for analyzing watermarking capacity is the seminal framework of Gel’fand and Pinsker (1980), which characterizes
communication over channels with random parameters. In the watermarking context, the _cover_
_image_ acts as the channel state: it is known to the encoder (the hider) but not to the decoder. The
Gel’fand–Pinsker theorem expresses the capacity of such channels as


_C_ = max
_p_ ( _u|s_ ) _, x_ ( _u,s_ )


- _I_ ( _U_ ; _Y_ ) _−_ _I_ ( _U_ ; _S_ )� _,_


(e.g., Somekh-Baruch and Merhav, 2004 and Merhav, 2005). However, these analyses remain
grounded in Euclidean or Hamming distortion metrics and in pixelwise probabilistic models of the
cover and the attack. Even extensions that allow more complex or adversarial behavior—such as
pixel permutations—still operate within a per-pixel, additive-noise paradigm.


These classical works provide valuable theoretical insight, but they share the same foundational
assumption: watermarking is modeled as a channel coding problem over an idealized Euclidean
signal space, with attacks represented as noise in that space. As discussed next, this assumption
diverges sharply from the realities of digital images and modern watermark attacks.


**The need for a new framework for watermarking capacity** The classical Gel’fand and Pinsker
(1980) framework treats watermarking as a _per-pixel communication problem_ : attacks are modeled
as additive or memoryless noise, distortions are measured in Euclidean or Hamming metrics, and
images are assumed to be continuous signals with well-defined probability densities. The capacity
expression
_I_ ( _U_ ; _Y_ ) _−_ _I_ ( _U_ ; _S_ )

reveals the central limitation of this view: mutual information depends on the _absolute_ _proba-_
_bility_ _distribution_ of natural images. Such distributions are unknown, representation-dependent
(RGB vs. YCbCr, gamma correction, quantization, compression), and fundamentally inestimable.
Because _I_ ( _U_ ; _Y_ ) and _I_ ( _U_ ; _S_ ) require these true underlying densities—not empirical or relative
statistics—Gel’fand–Pinsker capacities cannot be instantiated for real images.


Moreover, the dominant failure modes of watermarking systems are not pixelwise perturbations
but _geometric transformations_ : cropping, resampling, rotations, perspective changes, and nonrigid
warps that _move_ pixels rather than modify their intensities. These structured distortions introduce
strong spatial dependencies that fall far outside memoryless noise models and cannot be meaningfully
captured by _ℓ_ 2 or related Euclidean metrics. Real images also inhabit discrete, quantized spaces such
as _{_ 0 _, . . .,_ 255 _}_ [3], making differential-entropy–based Gaussian capacity formulas inapplicable. As a
result, classical information-theoretic capacity predictions are tied to an idealized Euclidean signal
model that digital images simply do not satisfy.


Because real attacks are predominantly geometric and because natural-image densities are inaccessible, classical information-theoretic models cannot yield realistic or actionable capacity estimates.
They cannot answer practical questions such as:


_How many bits can be reliably embedded in a_ 128 _×_ 128 _image while remaining_
_recoverable after rotations, cropping, or other structured geometric distortions?_


Our goal is to develop a capacity theory grounded in the _geometry of images_ —one that predicts how
many bits can be reliably stored under realistic transformations such as affine and nonrigid warps,
and that provides bounds and constructions reflecting the true failure modes of modern watermarking
systems.


16


In other words, _ϵ_ ( _τ_ ) specifies the largest _ℓ_ 2 distance (equivalently, the largest total squared pixel
error) allowed by the PSNR constraint.


17


B EQUIVALENCE OF PSNR AND THE _ℓ_ 2 BALL CONSTRAINT


In the main text we stated that imposing a minimum PSNR _τ_ is equivalent to requiring the watermarked image to remain within an _ℓ_ 2 ball around the cover image. Here we give the short derivation
and clarify how the radius is defined in terms of both _ℓ_ 2 and MSE distances.


PSNR is defined from the mean squared error (MSE) between two images:


(max possible pixel value) [2]
PSNR( _**x**_ _,_ ˜ _**x**_ ) = 10 log10 MSE( _**x**_ _,_ ˜ _**x**_ ) _._


The MSE measures the _average_ squared pixel difference, while the squared _ℓ_ 2 norm measures the
_total_ squared difference across all _cwh_ pixels. The two are linked by


1
MSE( _**x**_ _,_ ˜ _**x**_ ) = 2 _[.]_
_cwh_ _[∥]_ _**[x]**_ _[ −]_ _**[x]**_ [˜] _[∥]_ [2]


Thus, a PSNR threshold on MSE can be rephrased as a maximum allowable _ℓ_ 2 distance between _**x**_
and _**x**_ ˜. The connection between the two is:


2

           - _ρ_            PSNR( _**x**_ _,_ ˜ _**x**_ ) _≥_ _τ_ _⇐⇒_ 10 log10 MSE _≥_ _τ_

_ρ_ [2]
_⇐⇒_ MSE _≤_

10 _[τ/]_ [10]


_ρ_ [2]
_⇐⇒∥_ _**x**_ _−_ _**x**_ ˜ _∥_ [2] 2 _[≤]_ _[cwh][ ·]_


_√_
_⇐⇒∥_ _**x**_ _−_ _**x**_ ˜ _∥_ 2 _≤_ _ρ_


10 _[τ/]_ [10]


_cwh_ 10 _[−][τ/]_ [20] _._


We will define


_√_
_ϵ_ ( _τ_ ) = _ρ_


_cwh_ 10 _[−][τ/]_ [20] _._


C VOLUME-BASED ESTIMATION OF THE NUMBER OF GRID POINTS IN

HYPERSPHERES


Calculating watermarking capacity under a PSNR constraint reduces to counting how many valid
images (grid points in the pixel space) lie inside an _ℓ_ 2 ball of radius _ϵ_ around the cover image. This
is equivalent to asking how many integer grid points fall inside such a ball: a problem without a
general closed-form solution. In two dimensions this becomes the well-known _Gauss circle problem_
(Gauss, 1837; Hardy, 1915). In higher dimensions, and particularly for large radii, the count can be
well-approximated by the volume of the _n_ -dimensional ball:


_π_ _[n/]_ [2]
Vol( _Bn_ [ _·, r_ ]) = Γ                - _n_ 2 [+ 1]                - _r_ _[n]_ _._ (3)


We can approximate the number of integer points inside the ball with its volume. Simply, each grid
point corresponds to a unit cube in space, so counting grid points is almost the same as measuring
volume. The only difference comes from the boundary of the ball: some cubes are cut by the surface,
so they are only partially inside. The absolute error grows as _O_ ( _r_ [(] _[n][−]_ [1)] _[/]_ [2] ) (Walfisz, 1957; Mitchell,
1966). Since the volume itself grows as _O_ ( _r_ _[n]_ ), the relative error decreases as _O_ ( _r_ _[−]_ [(] _[n]_ [+1)] _[/]_ [2] ). Thus
for large radii the volume approximation is quite accurate. For small radii though, this error is
significant, as shown in Figure 7, hence we will use exact counting (Appendix D) for these cases.


18


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


D EXACT COUNTING GRID POINTS IN HYPERSPHERES FOR SMALL RADII


In Section 2.3 we need to calculate how many integer points are in the interior of small _ℓ_ 2 balls for
which approximating the number of integer points with the volume of the ball as in Appendix C
is inaccurate. Naively, we can iterate through the points in the smallest hypercube with integer
coordinates that contains our ball and check which points would fall inside the ball. See Algorithm 1
for one implementation of this.


**Algorithm 1:** Brute Force Count of Lattice Points in a Hypersphere

**1** **Function** BruteForceCount(radius : R _≥_ 0 _,_ dim : Z [+] ) **:**

**2** rsq _←_ radius [2] ;

**3** single ~~a~~ xis ~~p~~ oints _←{−⌊_ radius _⌋, . . ., ⌊_ radius _⌋}_ ;

**4** counter _←_ 0;

**5** **foreach** _**p**_ _∈_ product(single ~~a~~ xis ~~p~~ oints _,_ repeat = dim) **do**

**6** **if** [�] _p_ _[dim]_ =1 _**[p]**_ _i_ [2] _[≤]_ _[rsq]_ **[ then]**

**7** counter _←_ counter + 1;


**8** **return** counter;


Obviously, Algorithm 1 does not scale beyond very low dimensions because it has complexity that is
exponential in the dimension. Luckily, one can leverage symmetries to reduce the number points to
be checked. We take the method by (Mitchell, 1966) with pseudo-code presented in Algorithm 2.
Note that this can also be further sped up by caching the calls to S.


**Algorithm 2:** Count Lattice Points in a Hypersphere (Mitchell’s Method)

**1** **Function** PointsInHypersphereMitchell(dim : Z [+] _,_ radius : R _≥_ 0) **:**

**2** **return** S(dim _,_ radius [2] _,_ _∞_ );


**3** **Function** S( _m_ : Z _≥_ 0 _,_ _Z_ : R _,_ _J_ : Z _∪{∞}_ ) **:**

**4** **if** _m_ = 0 **then**

**5** **if** _Z_ _≥_ 0 **then**

**6** **return** 1;


**7** **else**

**8** **return** 0;


     - ~~�~~      **9** _N_ _←_ _Z/m_ ;

**10** _r_ _←_ (2 _N_ + 1) _[m]_ ;

**11** **for** _i ←_ 1 **to** _m −_ 1 **do**

        - �� ��
**12** MIN _i_ _←_ min _Z/i,_ _J_ _−_ 1 ;

**13** **for** _Jm_ _←_ _N_ + 1 **to** MIN _i_ **do**

           - _m_           **14** _r_ _←_ _r_ + _i_ _·_ 2 _[i]_ _·_ S( _m −_ _i,_ _Z −_ _iJm_ [2] _[,]_ _[J][m]_ [)][;]


**15** **return** _r_ ;


Unfortunately, Algorithm 2 is not applicable if we want to compute the number of lattice points in the
intersection between the hypersphere and a hypercube, as often needed when computing the number of
valid images available for watermarking in the present paper. In such cases, we can use the following
simple algorithm which is faster than the naive Algorithm 1 but slower than Algorithm 2. Note that,
again, this can be significantly sped up by caching the calls to IterativeCountWithBounds.


19


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


|Col1|Col2|Col3|Col4|Volume of the ba|ll|
|---|---|---|---|---|---|
|||||||
|||||~~Count of integer~~|~~   points in the ball~~|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


PSNR [dB] 40 45 50 55 60 65

Radius 70 60 50 40 30 20 10


Figure 7: **Discrepancy between the volume of a ball and the number of integer lattice points**
**contained in it for small** _ϵ_ ( _τ_ ) **.** When the radius is small, the volume-based approximation Bound 3
underesimates the actual number of integer points. In such cases we use the exact count Bound 4
instead. Evaluated for a 16 _×_ 16 _×_ 3 = 768-dimensional ball.


**Algorithm 3:** Count Lattice Points in a Hypersphere with Bounds


**1** **Function**
PointsInHypersphereWithBounds(dim : Z [+] _,_ radius : R _≥_ 0 _,_ bounds : (R _×_ R) [dim] ) **:**

**2** int ~~b~~ ounds _←_ ( _⌈b_ 0 _⌉, ⌊b_ 1 _⌋_ ) for each ( _b_ 0 _, b_ 1) in bounds;

**3** **return** IterativeCountWithBounds(radius [2] _,_ dim _,_ int ~~b~~ ounds);


**4** **Function** IterativeCountWithBounds(rsq : R _≥_ 0 _,_ dim : Z _≥_ 0 _,_ bounds : (Z _×_ Z) _[dim]_ ) **:**

**5** **if** dim = 0 **then**

**6** **return** 1 if rsq _≥_ 0, else 0;


**7** count _←_ 0;

      - _√_      **8** _M_ _←_ rsq ;

**9** lb _←_ max( _−M,_ bounds[0][0]);

**10** ub _←_ min( _M,_ bounds[0][1]);

**11** **for** _i ←_ lb **to** ub **do**

**12** **if** _i_ [2] _≤_ rsq **then**

**13** count _←_ count+ IterativeCountWithBounds(rsq _−_ _i_ [2] _,_ dim _−_ 1 _,_ bounds[1 :]);


**14** **return** count;


20


2000


1500


1000


500


0


500


1000


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


with the volume computed by truncating the sum in Theorem 1.
**Bound validity:** If _Bcwh_ [ _**x**_ _g, ϵ_ ( _τ_ )] _̸⊂_ _CI_ and _CI_ _̸⊂_ _Bcwh_ [ _**x**_ _g, ϵ_ ( _τ_ )], which happens when _[ρ]_ _/_ 2 _≤_
_√_ _√_
_ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 _cwh_ or, equivalently from Equation (1), when 20 log10 2 _≤_ _τ_ _≤_ 20 log10(2 _cwh_ ).

Assuming _cwh_ not too large (otherwise numerical evaluation of the bound becomes intractable).


The main limitation of Bound 5 is that it becomes computationally intractable to compute it for large
_cwh_ . In our implementation, we could evaluate it up to _cwh_ of several hundred. However, as can
be seen in Figure 3 left, Bound 5 can be well-approximated by simply considering the minimum of
Bound 2 and Bound 3. Therefore, for large resolutions we will use Bound 6.


3Note that some software libraries provide alternative (normalized) definitions for the Fresnel integrals:
_S_ ˜( _x_ ) = �0 _x_ [sin(] _[πx]_ [2] _[/]_ [2)][ and] _[C]_ [˜][(] _[x]_ [) =] �0 _x_ [cos(] _[πx]_ [2] _[/]_ [2)][.] [The two are related as such:] _[S]_ [(] _[x]_ [) =] ~~�~~ _π/_ 2 _S_ [˜] ( ~~�~~ 2 _/πx_ )
and _C_ ( _x_ ) = ~~�~~ _π/_ 2 _C_ [˜] ( ~~�~~ 2 _/πx_ ).


21


E BOUNDS FOR GRAY IMAGE IN THE NON-TRIVIAL INTERSECTION CASE


We expand on the exact bound on the capacity under only PSNR constraint (no robustness) in the
non-trivial intersection case, i.e., for medium PSNR values, as discussed in Section 2.3.3. We use the
volume-based approximation approach, reducing the problem to finding the volume of an intersection
of a hypercube and a hypersphere. Unfortunately, there is no closed-form solution. However, with
some care for the numerical precision, we can compute these intersections with numerical integration.
First, observe that we can express the volume of the intersection of an arbitrary ball and hypercube as
the volume of the intersection of appropriately transformed hypercube with the unit ball:











 _n_

 

_j_ =1


- _αj_ _−_ _**x**_ _j_ _[−]_ _**[x]**_ _[j]_
_,_ _[β][j]_
_r_ _r_


_∩_ _Bn_ [ **0** _,_ 1]


 _._ (4)


Vol


_n_

 


[ _αj, βj_ ] _∩_ _Bn_ [ _**x**_ _, r_ ]

_j_ =1


 = _r_ _[n]_ Vol


The right-hand side of Equation (4) can be represented as an infinite sum, which, in practice, can be
approximated via truncation. We will use the following result originally due Constales (1997) and
generalized by Aono and Nguyen (2017, Theorem 4):

**Theorem 1** (Volume of the intersection of a cube and a ball) **.** _Let S_ ( _x_ ) = �0 _x_ [sin(] _[t]_ [2][)] _[ dt][ and][ C]_ [(] _[x]_ [) =]
�0 _x_ [cos(] _[t]_ [2][)] _[ dt][ be the Fresnel integrals.]_ [3] _[Let][ α][j]_ _[≤]_ _[β][j]_ _[for]_ [ 1] _[≤]_ _[j]_ _[≤]_ _[n][ and][ ℓ]_ [=] [�] _j_ _[n]_ =1 [max(] _[α]_ _j_ [2] _[, β]_ _j_ [2][)] _[.]_
_Then:_


Vol �� _nj_ =1 [[] _[α][j][, β][j]_ []] _[ ∩]_ _[B][n]_ [ [] **[0]** _[,]_ [ 1]] - =






_K_ _if ℓ_ _≤_ 1

- 21 _[−]_ - _nj_ =1 _[α]_ _j_ [2] 3 [+] _ℓ_ _[β]_ _j_ [2][+] _[α][j]_ _[β][j]_ + [1] _ℓ_ [+] _π_ [1] [Im][ �] _k_ _[∞]_ =1 Φ( _−_ 2 _kπk/ℓ_ ) _e_ [2] _[iπk/ℓ]_ - _K_ _if ℓ>_ 1


_π_ [1] [Im][ �] _k_ _[∞]_ =1 Φ( _−_ 2 _kπk/ℓ_ )


    _kπk/ℓ_ ) _e_ [2] _[iπk/ℓ]_ _K_ _if ℓ>_ 1





- _n_
_j_ =1 _[α]_ _j_ [2][+] _[β]_ _j_ [2][+] _[α][j]_ _[β][j]_
3 _ℓ_ + [1] _ℓ_


[1] _ℓ_ [+] _π_ [1]


_where K_ = [�] _j_ _[n]_ =1 _[|][β][j]_ _[−]_ _[α][j][|][ and]_ [ Φ] _[ is defined as]_


- _C_ ( _βj_ ~~�~~ _|ω|_ ) _−_ _C_ ( _αj_ - _|ω|_ )� + _i_ sign( _ω_ ) - _S_ ( _βj_ ~~�~~ _|ω|_ ) _−_ _S_ ( _αj_ - _|ω|_ )�

_._
( _βj_ _−_ _αj_ )� _|ω|_


Φ( _ω_ ) =


_n_


_j_ =1


A number of speed-ups and numerical performance optimization tricks can be used when computing
the terms in Theorem 1. For example, when all _αj_ = _α_ 1 and _βj_ = _β_ 1 for all 1 _≤_ _j_ _≤_ _n_ and
when _αj_ = _−βj_, which is often the case in the setups we consider. We provide such optimized
implementation in Algorithm 4.


Theorem 1 directly gives us a bound on the capacity for the non-trivial intersection case:


**Bound 5:** **Gray image, PSNR constraint (medium PSNR, numerical integration).** The capacity
of a gray image _**x**_ _g_ under a minimum PSNR constraint _τ_ in the ambient space _A_ is upper-bounded by




capacity[in bits] _≈_ log2  _ϵ_ _[cwh]_ Vol


 _cwh_

 

_j_ =1


_[ρ/]_ [2] _[ρ/]_ [2]

_ϵ_ ( _τ_ ) _[,]_ _ϵ_ ( _τ_ )


_−_ _[ρ/]_ [2]


_ϵ_ ( _τ_ )


_∩_ _Bcwh_ [ **0** _,_ 1]

















_−_ _[ρ/]_ [2]


_ϵ_ ( _τ_ )





= _cwh_ log2 _ϵ_ ( _τ_ ) + log2 Vol


 _cwh_

 


_∩_ _Bcwh_ [ **0** _,_ 1]


 _,_


_j_ =1


_[ρ/]_ [2] _[ρ/]_ [2]

_ϵ_ ( _τ_ ) _[,]_ _ϵ_ ( _τ_ )


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


**35** _S_ 4 _←_ 0;

**36** **for** _k_ _←_ 1 **to** _Nsum_ **do**

**37** _ωk_ _←−_ 2 _πk/ℓ_ ;

**38** _S_ 4 _←_ _S_ 4 + _k_ [1] _[·]_ [ Φ][(] _[ω][k][,]_ _**[ α]**_ _[′][,]_ _**[ β]**_ _[′]_ [)] _[ ·]_ [ expjpi(2] _[k/ℓ]_ [)][;]

**39** _T_ 4 _←_ Im( _S_ 4) _/π_ ;

**40** _V_ norm _←_ _T_ 1 _−_ _T_ 2 + _T_ 3 + _T_ 4;

**41** **return** log2( _V_ norm) + _V_ scale + len( _**α**_ _[′]_ ) _·_ log2( _R_ );


22


**Algorithm 4:** Computes the volume of the intersection between a hypersphere and a box


**1** **Function** ApplyWithGrouping( _f_ : function _,_ args _,_ mode : (product _,_ sum)) **:**

// Applies a function _f_ to arguments, grouping them for stability.

**2** pairs _←_ list(zip( _∗_ args));

**3** _C_ _←_ Counter(pairs) ; // Count unique argument tuples

**4** **if** mode = product **then**

**5** result _←_ 1;


**6** **else if** mode = sum **then**

**7** result _←_ 0;


**8** **foreach** _p ∈C._ keys() **do**

**9** _c ←C_ [ _p_ ];

**10** _r_ _←_ _f_ ( _p_ );

**11** **if** mode = product **then**

**12** result _←_ result _· r_ _[c]_ ;


**13** **else if** mode = sum **then**

**14** result _←_ result + _c · r_ ;


**15** **return** result;


**16** **Function** Φinner( _α_ : R _, β_ : R _, ω_ : R) **:**

~~�~~
**17** _ω_ _[′]_ _←_ _|ω|_ ;

**18** **if** _α_ = _−β_ **then**

**19** _T_ 1 _←_ 2 _·_ fresnelc( _βω_ _[′]_ );

**20** _T_ 2 _←_ _i ·_ sgn( _ω_ ) _·_ 2 _·_ fresnels( _βω_ _[′]_ );


**21** **else**

**22** _T_ 1 _←_ fresnelc( _βω_ _[′]_ ) _−_ fresnelc( _αω_ _[′]_ );

**23** _T_ 2 _←_ _i ·_ sgn( _ω_ ) _·_ (fresnels( _β_ _[√]_ _ωabs_ ) _−_ fresnels( _αω_ _[′]_ ));

**24** **return** ( _T_ 1 + _T_ 2) _/_ (( _β −_ _α_ ) _ω_ _[′]_ );

**25** **Function** Φ( _ω_ : R _,_ _**α**_ _,_ _**β**_ ) **:**

**26** _f_ inner _←_ (( _α, β_ ) _�→_ Φinner( _α, β, ω_ ) );

**27** **return** ApplyWithGrouping( _finner,_ [ _**α**_ _,_ _**β**_ ] _,_ product);

**28** **Function** BallCubeIntersection( _R_ : R _>_ 0 _,_ _**α**_ _,_ _**β**_ _,_ _Nsum_ : Z [+] ) **:**


// Trim scaled bounds to the unit ball range [ _−_ 1 _,_ 1]


**29** _**α**_ _[′]_ _←_ elementwise ~~m~~ ax( _−_ 1 _,_ _**α**_ _/R_ );


**30** _**β**_ _[′]_ _←_ elementwise ~~m~~ in(1 _,_ _**β**_ _/R_ );


**31** _ℓ_ _←_ ApplyWithGrouping(( _a, b_ ) _�→_ (max( _−a, b_ )) [2] _,_ [ _**α**_ _[′]_ _,_ _**β**_ _[′]_ ] _,_ sum);


**32** _EX_ _←_ [1] _/_ 3 ApplyWithGrouping(( _a, b_ ) _�→_ _a_ [2] + _b_ [2] + _ab,_ [ _**α**_ _[′]_ _,_ _**β**_ _[′]_ ] _,_ sum);


**33** _V_ scale _←_ ApplyWithGrouping(( _a, b_ ) _�→_ log2( _b −_ _a_ ) _,_ [ _**α**_ _[′]_ _,_ _**β**_ _[′]_ ] _,_ sum);


**34** _T_ 1 _←_ [1] _/_ 2; _T_ 2 _←_ _[E][X]_ _/ℓ_ ; _T_ 3 _←_ [1] _/ℓ_ ;


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


_√_
**Bound validity:** If _ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 or, equivalently from Equation (1), if _τ_ _≥_ 20 log10(2 _cwh_ ). Assum
ing _ϵ_ ( _τ_ ) not too small for the volume approximation to be valid (see Bound 8 for small _ϵ_ ( _τ_ )).


**Bound 8:** **Arbitrary image, PSNR constraint (high PSNR, exact count, analogous to Bound 4).**
The capacity of an image _**x**_ under a minimum PSNR constraint _τ_ in the ambient space _A_ is upperbounded by


capacity[in bits] _≈_ log2 PointsInHypersphereWithBounds( _cwh, ϵ_ ( _τ_ ) _,_ (0 _,_ 2 _[k]_ )) _,_


with PointsInHypersphereWithBounds defined in Algorithm 3. _√_
**Bound validity:** If _ϵ_ ( _τ_ ) _≤_ _[ρ]_ _/_ 2 or, equivalently from Equation (1), if _τ_ _≥_ 20 log10(2 _cwh_ ). Assuming

_ϵ_ small (otherwise the numerical evaluation becomes intractable, see Bound 7 for large _ϵ_ ).


Note that Bound 7 slightly _under-approximates_ capacity, since only half of the lattice points lying on
the faces of the hypercube are captured by the volume approximation. This explains the discontinuity
between Bound 7 and Bound 8 visible in Figure 3 right.


For the other two cases: non-trivial intersection and the cube being fully in the ball, we can directly
apply Theorem 1 with appropriate change of bounds. We simply need to adjust the bounds in Bound 5
from [ _−_ _[ρ]_ _/_ 2 _,_ _[ρ]_ _/_ 2] to [0 _, ρ_ ]:


23


F BOUNDS FOR ARBITRARY COVER IMAGES


In Section 2.4 we extended the analysis from centred covers to arbitrary images. Here we go into
detail about how the corresponding bounds were derived.


When pixels are saturated, the cover may lie on the boundary of the cube of cube _CI_ . The most
adverse case is when all pixels are saturated,i.e., the cover at a corner. This minimizes the overlap
between the PSNR ball and the grid and hence provides a lower bound on capacity for any image.


By symmetry, the ball is evenly divided among the 2 _[cwh]_ orthants of R _[cwh]_, so only a fraction [1] _/_ 2 _[cwh]_
of its volume lies within the grid. In two dimensions this corresponds to a quarter of a circle inside a
square corner, and in three dimensions to one eighth of a sphere inside a cube corner. Although this
seems drastic, the effect on capacity is limited: at most _cwh_ bits, i.e. one bit per pixel.


We now provide the detailed derivations of Bounds 7 to 9, which are the analogues of the centered
bounds from Section 2.3, adapted to the corner case.


**Bound 7:** **Arbitrary image, PSNR constraint (high PSNR, volume approximation, analogous to**
**Bound 3).** The capacity of an image _**x**_ under a minimum PSNR constraint _τ_ in the ambient space _A_
is upper-bounded by


capacity[in bits] _≈_ log2


- Vol _Bcwh_ [ _·, ϵ_ ( _τ_ )]
2 _[cwh]_


log2 _π_ + _cwh_ log2 _ϵ_ ( _τ_ ) _−_ [ln Γ]  - _cwh_ 2 + 1�
2 ln 2


= _[cwh]_


_−_ _cwh._
ln 2


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


_j_ =1


with the volume computed by truncating the sum in Theorem 1.
_√_
**Bound** **validity:** If _ϵ_ ( _τ_ ) _≥_ _[ρ]_ _/_ 2 or, equivalently from Equation (1), when _τ_ _≤_ 20 log10(2 _cwh_ ).

Assuming _cwh_ not too large (otherwise the numerical evaluation becomes intractable).


Unlike the centred case, here the symmetry condition _αj_ = _−βj_ no longer holds, so the simplifications
of Theorem 1 cannot be applied. Bound 9 is therefore numerically stable only at relatively low
resolutions. Nevertheless, the per-pixel capacity (bpp) is resolution-invariant, so we compute these
bounds at 16 _×_ 16px.


Bounds 7 to 9 extend the centred-case analysis to arbitrary cover images. Across all three regimes,
the penalty of being at the corner of the grid is at most one bit per pixel. Thus, the observed gap
between theoretical capacity and practical watermarking performance cannot be explained by image
position within the grid.


24


**Bound 9:** **Arbitrary image, PSNR constraint (medium PSNR, numerical integration, analogous**
**to Bound 5).** The capacity of an image _**x**_ under a minimum PSNR constraint _τ_ in the ambient space
_A_ is upper-bounded by






_ϵ_ _[k]_ ( _[−]_ _τ_ ) [1] - _∩_ _Bcwh_ [ **0** _,_ 1]�





capacity[in bits] _≈_ log2


 _ϵ_ ( _τ_ ) _[cwh]_ Vol� _[cwh]_ 


�0 _,_ [2] _[k][−]_ [1]


_j_ =1


= _cwh_ log2 _ϵ_ + log2 Vol� _[cwh]_ 


_ϵ_ _[k]_ ( _[−]_ _τ_ ) [1] - _∩_ _Bcwh_ [ **0** _,_ 1]� _,_


�0 _,_ [2] _[k][−]_ [1]


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


G BOUNDS FOR CAPACITY UNDER ROBUSTNESS CONSTRAINTS


In practice, watermarking requires balancing perceptual quality with robustness. That is, minor
modifications to the watermarked image should not prevent the watermark from being extractable.
Such modifications might arise in the normal processing of the image (a social media website might
compress uploaded images) or might be malicious (to strip provenance information). Typically, one
considers a set of transformations against which a watermarking method should be robust.


Robustness comes at a cost: it reduces capacity. To quantify this trade-off we study how robustness constraints reduce the number of images which we can use for watermarking, quantifying
the corresponding reduction in capacity. We will focus on robustness to linear transformations,
which, though seemingly restrictive, covers or approximates most practical transformations. Linear
transformations are also compositional, simplifying the creation of complex transformations from
basic ones. Standard augmentation can be directly represented as linear transformations. Appendix H
shows how to represent colour space changes, rotation, flipping, cropping and rescaling, as well as a
number of intermediate operators that can be used to construct the linear operators corresponding to
other transformations.


G.1 ROBUSTNESS TO LINEAR TRANSFORMATIONS

A linear transformation applied to the ball _Bcwh_ [ **0** _, ϵ_ ] of possible watermarked images can turn it into
an ellipsoid (if the transformation has more than one unique singular value) can scale it (if they are all
the same but not 1) and can also project it into a lower-dimensional space if the transformation is not
invertible. Let’s take a linear transformation _M_ _∈_ R _[cwh][×][cwh]_ that maps an image _**x**_ to a transformed
image _M_ _**x**_ . Note that we further need to apply a quantization operation to map the pixel values of
_M_ _**x**_ to the valid images _I_ . Hence, we have the final transformed image _**x**_ _[′]_ = Q[ _M_ _**x**_ ] with Q being a
quantization operator, typically an element-wise rounding or floor operation. The main complication
in this setup is that Q is non-linear. To establish the capacity under a linear transformation, we need
to find the subset of the possible watermarked images under only the PSNR constraint that map to
unique valid images after applying _M_ and Q to them.


G.2 HEURISTIC BOUNDS

A simple approach is to consider the volumetric approach for calculating capacity that we used for
Bounds 3, 5, 7 and 9. A linear operator _M_ would change the volume of _Bcwh_ [ **0** _, ϵ_ ] by a factor
of det _M_ = _λ_ [1] _M_ _[× · · · ×][ λ]_ _M_ _[cwh]_ (the product of _M_ ’s eigenvalues) if _M_ is not singular. Note that if
det _M_ _>_ 0, then we can also express it as det _M_ = _σM_ [1] _[×· · ·×][σ]_ _M_ _[cwh]_ [, the product of singular values of]
_M_ . If _M_ is singular, then _MBcwh_ [ **0** _, ϵ_ ] has 0 _cwh_ -dimensional volume as some eigenvalues (singular
values) would be 0. However, it will have a non-zero pdet _M_ = [��] _λ_ _[i]_ _M_ _[|][ λ]_ _M_ _[i]_ [= 0] _[,]_ _[i]_ [ = 1] _[, . . ., cwh]_ 
(rank _M_ )-dimensional volume, with pdet _M_ being the _pseudo-determinant_ of _M_ .


The change in volume governs the reduction in capacity. However, it is not as simple as just calculating
this volume and taking that to be the capacity because of the quantizer Q. Take for example


�2 0                   _M_ = _._ (5)
0 1 _/_ 2


The determinant is det _M_ = 1 indicating no volume change and, thus, if we ignore the quantization,
no capacity change. However, one of the dimensions is squeezed by a factor of 2, hence we should
lose about half of the capacity along this axis. The other axis, that is stretched by a factor of 2, does
not create capacity because pairs of these points would have the same preimage. Therefore, assuming
we are far from the boundaries of the cube, we should see half the original capacity, if we want to
have robustness to this augmentation. Following this observation, we provide an heuristic for the
reduction _ξ_ of capacity due to the linear operator and quantization:


_ξM_ =            - min( _σM_ _[i]_ _[,]_ [ 1)] _[,]_ (6)

_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


where Σ _M_ are the singular values of _M_ . _ξM_ captures the combined effect of all singular values of _M_ .
Each _σi_ _<_ 1 represents a compression that reduces capacity proportionally due to the quantization,
while _σi_ _>_ 1 is capped at 1 since stretching cannot create capacity. The product accounts for all
rank _M_ dimensions, so _ξM_ reflects the total fraction of capacity that remains after accounting for all
reductions.


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


**Bound** **valid** _√_ **ity:** If _ϵ_ ( _τ_ ) _≤_ _ρ/_ 2 _µ_ with _µ_ = max _{σ_ 1 _, . . ., σcwh,_ 1 _}_ or, equivalently when _τ_ _≥_
20 log10(2 _µ_ _cwh_ ). Assuming _ϵ_ ( _τ_ ) not too small (see Bound 11 for small _ϵ_ ( _τ_ )).


Similarly to Bounds 4 and 8, when the radius _ϵ_ is too small, the volume approximation is poor and
we resort to exact counting instead:


**Bound 11:** **Gray image, Linear transformation (Heuristic), PSNR constraint (high PSNR, exact**
**count).** The capacity of a gray image _**x**_ _g_ under a low minimum PSNR constraint _τ_ and a linear
transformation _M_ _∈_ R _[cwh][×][cwh]_ in the ambient space _A_ is upper-bounded as such:


capacity[in bits] _≈_ log2 _ξM_ PointsInHypersphereMitchell (rank _M, ϵ_ )

= log2 PointsInHypersphereMitchell (rank _M, ϵ_ ) +         - min(log2 _σM_ _[i]_ _[,]_ [ 0)]

_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


with PointsInHypersphereMitchell as described in Algorithm 2.
**Bound** **valid** _√_ **ity:** If _ϵ_ ( _τ_ ) _≤_ _ρ/_ 2 _µ_ with _µ_ = max _{σ_ 1 _, . . ., σcwh,_ 1 _}_ or, equivalently when _τ_ _≥_
20 log10(2 _µ_ _cwh_ ). Assuming _ϵ_ ( _τ_ ) is small (otherwise, computationally intractable, see Bound 10

for large _ϵ_ ( _τ_ )).


And finally, we have the non-trivial intersection case, where we need to account for the clipping of
the watermarked images to _CI_, the cube of all possible images:


26


If, for a moment, we ignore that we need to clip the pixel values in their valid range, we get
the following capacity under an _invertible_ linear transformation _M_ for _ϵ_ large enough so that the
volume-based approximation is applicable:


capacity under _M_ [in bits] = capacity[in bits] + log2 _ξM_
_≈_ log2 Vol _Bcwh_ [ _·, ϵ_ ( _τ_ )] + log2 _ξM_ _._


If _M_ is singular, however, then we need to compute the capacity under the lower-dimensional
projection of _Bcwh_ [ _·, ϵ_ ( _τ_ )] in order to account for the collapsed dimensions:


capacity under _M_ [in bits] _≈_ log2 Vol _B_ rank _M_ [ _·, ϵ_ ( _τ_ )] + log2 _ξM_ _._


Note that the radius _ϵ_ ( _τ_ ) is still computed in the ambient _cwh_ -dimensional space.


A further complication caused by the _σi_ _>_ 1 singular values is the possibility of the sphere going
out of the bounds of the cube after the transform. Looking again at the example in Equation (5), the
stretching along the first dimension might result in some of the watermarked images being clipped
and hence mapped to the same image after applying the transformation. We can factor this in by
adjusting the bounds corresponding to the cube boundaries in Bounds 3, 4 and 5 accordingly. This
results in the heuristic bounds in this section.


First, let’s look at the setting where the ball is fully inside the cube _before and after_ the transformation,
and its radius _ϵ_ ( _τ_ ) is large enough for us to approximate the number of images in it with its volume.
Taking into account _M_ being possibly singular, we have:


**Bound** **10:** **Gray** **image,** **Linear** **transformation** **(Heuristic),** **PSNR** **constraint** **(high** **PSNR,**
**volume approximation).** The capacity of a gray image _**x**_ _g_ under a low minimum PSNR constraint _τ_
and a linear transformation _M_ _∈_ R _[cwh][×][cwh]_ in the ambient space _A_ is upper-bounded as such:


capacity[in bits] _≈_ log2 _ξM_ Vol _B_ rank _M_ [ _·, ϵ_ ( _τ_ )]

_π_ [(rank] _[ M]_ [)] _[/]_ [2] _ϵ_ ( _τ_ ) [rank] _[ M]_
= log2 _ξM_ Γ         - rank2 _M_ + 1�


_[ M]_ log2 _π_ + (rank _M_ ) log2 _ϵ −_ [ln Γ]  - rank2 _M_ + 1�

2 ln 2


 - min(log2 _σM_ _[i]_ _[,]_ [ 0)] _[.]_

_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


= [rank] _[ M]_


2 + 
ln 2


- �2 0
_M_ = 0 1 _/_ 2


_R_ 45 _◦_


�2 0
_M_ = 0 1 _/_ 2


- �2 0
_R_ 15 _◦_ _M_ = 0 1 _/_ 2


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


The rotation does not affect the singular values hence the scaling factor _ξM_ = _ξMθ_ and Bounds 10
to 12 are unaffected. Nevertheless, the exact count of images that remain after _Mθ_ and Q (and thus
the capacity) is much higher in the rotated case. Figure 8 shows that while _ξMθ_ is 0.5 for all _θ_, the
actual capacity factor at for _θ_ = 45 _[◦]_ is 0.6, i.e., 20% higher. Thus Bounds 10 to 12 are not upper
bounds on the capacity under a linear transformation.


Unfortunately, Bounds 10 to 12 are also not lower bounds on the capacity. Take the case of
transforming with just _Rθ_ . _Rθ_ has a determinant 1 for all _θ_, hence the scaling factor _ξRθ_ is also 1
and therefore, the capacity should be unchanged. However, as can be seen in Figure 9, the empirical
_ξ_ ˆ _R_ 45 _◦_ is just 0.837 when we rotate the disk by _θ_ = 45 _[◦]_ . While this particular case has been studied
analytically by Vladimirov (2015, Theorem 19) who demonstrated that for large enough disks this
reduction is _ξ_ [ˆ] _Rθ_ = 1 _−_ (cos _θ_ + sin _θ −_ 1) [2], convenient results for general linear operators _M_ are
unlikely to be possible.


27


Original capacity (red points):
Capacity after transform (blue points):
Capacity kept after transform:
Predicted capacity kept ( ):


1789

927


51.8%
50.0%


56.4%
50.0%


60.1%
50.0%


Original capacity (red points):
Capacity after transform (blue points):
Capacity kept after transform:
Predicted capacity kept ( ):


1789
1009


Original capacity (red points):
Capacity after transform (blue points):
Capacity kept after transform:
Predicted capacity kept ( ):


~~1789~~
1 ~~0~~ 75


Figure 8: **Quantization can result in higher capacities than predicted by Equation (6).** Showing
how rotations of a matrix with singular values smaller and larger than 1 can affect the capacity of a
linear transform due to the quantization effects. Despite Equation (6) predicting factor _ξ_ = 0 _._ 5 for all
three cases, for this _M_ we observe larger factors of up to 0.60 in the case of 45 _[◦]_ rotation.


**Bound 12:** **Gray image, Linear transformation (Heuristic), PSNR constraint (medium PSNR,**
**numerical integration).** The capacity of a gray image _**x**_ _g_ under a low minimum PSNR constraint _τ_
and a linear transformation _M_ _∈_ R _[cwh][×][cwh]_ in the ambient space _A_ is upper-bounded as such:





 

_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


- _ρ_ _ρ_

_−_
2 _ϵ_ max(1 _, σM_ _[i]_ [)] _[,]_ 2 _ϵ_ max(1 _, σi_ )








_∩_ _B_ rank _M_ [ **0** _,_ 1]








capacity[in bits] _≈_ log2




 _ξM_ _ϵ_ [rank] _[ M]_ Vol


= (rank _M_ ) log2 _ϵ_ ( _τ_ )






_∩_ _B_ rank _M_ [ **0** _,_ 1]


+ log2 Vol  
_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


- _ρ_ _ρ_

_−_
2 _ϵ_ ( _τ_ ) max(1 _, σM_ _[i]_ [)] _[,]_ 2 _ϵ_ ( _τ_ ) max(1 _, σM_ _[i]_ [)]


+      - min(log2 _σM_ _[i]_ _[,]_ [ 0)] _[,]_

_σM_ _[i]_ _[∈]_ [Σ] _[M]_ [:] _[σ]_ _M_ _[i]_ _[>]_ [0]


with the volume computed by truncating the sum in Theorem 1.
**Bound** **valid** _√_ **ity:** If _ϵ_ ( _τ_ ) _>_ _ρ/_ 2 _µ_ with _µ_ = max _{σ_ 1 _, . . ., σcwh,_ 1 _}_ or, equivalently when _τ_ _≤_
20 log10(2 _µ_ _cwh_ ). Assuming rank _M_ not too large (otherwise numerical evaluation of the bound

becomes intractable).


We would like to stress that Bounds 10 to 12 are _heuristics_ and are near-exact only for axis-aligned
transformations, i.e., where _M_ has at most one non-zero value in each row or column.


Higher capacities than predicted by Bounds 10 to 12 are possible. Take _Mθ_ = _MRθ_ as in Equation (5)
but multiplied with a rotation matrix _Rθ_ :


�cos _θ_ _−_ sin _θ_
_Rθ_ = sin _θ_ cos _θ_


_._


�1 0� �1 0
_M_ = _M_ =
0 1 0 1


- �1 0�
_R_ 15 _◦_ _M_ = 0 1 _R_ 45 _◦_


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


|ingular values of the Crop&Rescale transform|Col2|Singular values of the Rotation transform|Singular values of the Linear JPEG transform|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
|Horizontal Flip<br>|Horizontal Flip<br>|Rotation 15deg<br>|LinJPEG q=15<br>|LinJPEG q=15<br>|LinJPEG q=15<br>|LinJPEG q=15<br>|LinJPEG q=15<br>|
|~~Crop&Rescale 75%~~<br>Crop&Rescale 50%<br>|~~Crop&Rescale 75%~~<br>Crop&Rescale 50%<br>|~~Rotation 30deg~~<br>Rotation 45deg|~~LinJPEG q=10~~<br>LinJPEG q=8|~~LinJPEG q=10~~<br>LinJPEG q=8|~~LinJPEG q=10~~<br>LinJPEG q=8|~~LinJPEG q=10~~<br>LinJPEG q=8|~~LinJPEG q=10~~<br>LinJPEG q=8|
|Crop&Rescale 25%|Crop&Rescale 25%|||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


Figure 11: **Singular values of the linear transformations.** Plotted are the singular values of the
linear transformations, sorted in descending order.


28


Original capacity (red points): 1789
Capacity after transform (blue points):   1789
Capacity kept after transform: 100.0%
Predicted capacity kept ( ): 100.0%


Original capacity (red points): 1789
Capacity after transform (blue points):   1701
Capacity kept after transform: 95.1%
Predicted capacity kept ( ): 100.0%


Original capacity (red points): 1789
Capacity after transform (blue points):   1497
Capacity kept after transform: 83.7%
Predicted capacity kept ( ): 100.0%


Figure 9: **Quantization can result in lower capacities than predicted by Equation (6).** Showing
how rotations of a disk can affect the capacity of a linear transform due to the quantization effects.
Despite Equation (6) predicting factor _ξ_ = 1 for all three cases, for rotations we observe larger factors
of as little as 0.837 in the case of 45 _[◦]_ rotation.


Original Horizontal Flip Crop&Rescale 75% Crop&Rescale 50% Crop&Rescale 25% Rotation 15deg Rotation 30deg Rotation 45deg LinJPEG q=15 LinJPEG q=10 LinJPEG q=8


Figure 10: **Robustness constraints considered in this paper.** The figure illustrates the linear transformations we evaluate: Horizontal Flip, Crop&Rescale, Rotation and Linearlized JPEG compression.
We show them both at full resolution and at 16 _×_ 16px at which we compute the bounds in Section 2.5.


Nevertheless, for practical purposes, we believe that Bounds 10 to 12 are mostly lower bounds. For
instance, when we have both “squish” and “stretch” axes, i.e., singular values both smaller and
larger than 1 —which is the case for the transformations we consider— then the true capacity can be
much higher than predicted due to the rounding operation filling the space better as seen in Figure 8.
Therefore, in general, we would consider this bound to be a good heuristic.


To evaluate the effect of Bounds 10 to 12 on the watermarking capacity, we considered horizontal
flipping, cropping and rescaling, rotation (around the centre of the image), as illustrated in Figure 10.
The construction of the respective matrices is described in Appendices H.3.3, H.3.5 and H.3.6.
As can be seen in Figure 11 they all, except for the horizontal flip, have singular values above
and below 1 and are rank deficient. Figure 4 has Bounds 10 to 12 for the Crop&Rescale and the
Rotation transformations and compares them with the bounds without robustness constraints from
Section 2.3. Rotations have a roughly 2 bpp decrease in capacity mostly driven by the loss of


0 100 200 300 400 500 600 700
Singular value index


0 100 200 300 400 500 600 700
Singular value index


3.5


3.0


2.5


2.0


1.5


1.0


0.5


0.0


0 100 200 300 400 500 600 700
Singular value index


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


_with_

            - 1             -             -             - _⊤_
_**β**_ = abs 2 [Σ][+] _[U][ ⊤]_ **1** _n_ + **0** _[⊤]_ rank[ _M_ ] _[, r]_ **[1]** _[⊤]_ _n−_ rank[ _M_ ] _,_ (9)


_where U_ Σ _V_ _[⊤]_ _is the SVD decomposition of M_ _and_ Σ [+] _is the pseudo-inverse of the diagonal matrix_
Σ _(i.e., the reciprocal of the non-zero elements of_ Σ _)._ _The volume of the box-ball intersection can be_
_computed with Theorem 1._


Now, Theorem 2 gives us an upper-bound for _K_, the number of images in a PSNR ball that are
“collapsed” onto the same image after a linear transformation _M_ .


29


capacity at the corners and the effects of the interpolation. As expected, aggressive crops reduce
the capacity significantly. At 40 dB, cropping to 0.25 of the image results in about 0 _._ 5 bpp, down
from more than 3 bpp without the robustness constraint. Still, 0 _._ 5 bpp implies capacity of 98 _,_ 304 bits
for a 256 _×_ 256px image, considerably larger capacity than what we observe in practice. Therefore,
robustness to augmentations does not seem to explain the much lower capacities that we see in
practice.


G.3 CONSERVATIVE BOUNDS

While we are confident in the heuristic Bounds 10 to 12, it is nevertheless possible that, in practice,
capacities suffer from a similar problem as the rotation _Rθ_ (Figure 9) and hence are lower than
predicted by the heuristic bounds. Unfortunately, due to the non-linear nature of the quantizer Q and
the curse of dimensionality, providing an actual lower bound for _ξM_ proves difficult. Still, we outline
one approach here which, while extremely conservative, is a valid lower bound.


The idea of this conservative bound is to find an upper bound for the cardinality of the preimage of any
transformed image, that is, an upper bound to how many images would be mapped to the same transformed image under _M_ and Q. Every untransformed point (image) _**x**_ in _CI_ _∩_ _Bcwh_ [ _**x**_ _g, ϵ_ ] _∩_ Z _[cwh]_

maps to a point _**x**_ _[′]_ in _CI_ _∩_ Z _[cwh]_ after being transformed by Q[ _M_ _**x**_ ]. However, multiple images can
map to the same image by the transformation, i.e., _|M_ [+] Q _[−]_ [1] [ _**x**_ _[′]_ ] _∩_ _CI_ _∩_ _Bcwh_ [ _**x**_ _g, ϵ_ ] _∩_ Z _[cwh]_ _|_
might be more than 1. Here _M_ [+] is the (Moore–Penrose) pseudo-inverse of _M_ and Q _[−]_ [1] [ _**x**_ _[′]_ ] is the
set of points that Q maps to _**x**_ _[′]_, thus _M_ [+] Q _[−]_ [1] [ _**x**_ _[′]_ ] are all points _**x**_ such that Q[ _M_ _**x**_ ] = _**x**_ _[′]_ . Define
_N_ to be the number of points available for watermarking without the robustness constraint (i.e.,
2 [capacity][[in bits]] ) and _n_ to be the number of unique images that are robust to the transformation Q _M_ :


_N_ = �� _CI_ _∩_ _Bcwh_ [ _**x**_ _g, ϵ_ ] _∩_ Z _cwh_ �� _,_

_n_ = ��� _**x**_ _[′]_ _∈_ _CI_ _∩_ Z _[cwh]_ _|_ _∃_ _**x**_ _∈_ _CI_ _∩_ _Bcwh_ [ _**x**_ _g, ϵ_ ] _∩_ Z _[cwh]_ such that _**x**_ _[′]_ = Q[ _M_ _**x**_ ]��� _._ (7)


In other words, _n_ is the capacity that we can achieve while being robust to the linear transformation
_M_ . Note that the scaling factor _ξM_ is precisely _n/N_ . Section 2.3 was concerned with finding _N_,
here we will provide an upper bound to _n_ . While obtaining _n_ directly is difficult, if we know that
every transformed point has at most _K_ preimage points, i.e.:


_|M_ [+] Q _[−]_ [1] [ _**x**_ ] _∩_ _CI_ _∩_ _Bcwh_ [ _**x**_ _g, ϵ_ ( _τ_ )] _∩_ Z _[cwh]_ _| ≤_ _K,_ _∀_ _**x**_


then we know that _n ≥⌈_ _[N]_ _/K⌉_ . In other words we have that the scaling factor must be lower-bounded
as _ξM_ _≥_ [1] _/K_ . So the problem of finding a lower bound to _ξM_ has reduced to obtaining _K_, an upper
bound to the number of preimage points that any transformed image can have.


The quantization operation Q maps a hypercube of side 1 to a given image, regardless of whether the
rounding or floor quantization is used. The maximum number of points that can fit in the preimage
under _M_ of such a unit cube is the _K_ we are looking for. The preimage of a hypercube under a linear
operation _M_, when restricted to a hypersphere of radius _ϵ_, can be over-approximated with a zonotope
(with proof in Appendix I):


**Theorem 2.** _Given a hypercube_ _C_ = _**b**_ + [0 _,_ 1] _[n]_ _⊂_ R _[n]_ _,_ _**b**_ _∈_ R _[n]_ _and a possibly singular matrix_
_M_ _∈_ R _[n][×][n]_ _giving rise to the map fM_ ( _**x**_ ) = _M_ _**x**_ _, the supremum of the volume of the preimage of C_
_under M_ _when intersected with a hypersphere of radius r is upper-bounded as:_


_r_


sup - _fM_ _[−]_ [1][(] _[C]_ [)] _[∩]_ _[B][n]_ [ [] _**[c]**_ _[, r]_ []] - _≤_ _r_ _[n]_ Vol
_**c**_ _∈_ R _[n]_ [ Vol]


- _n_


_i_ =1


_−_ _[β][i]_


_[i]_

_r_ _[, ][β]_ _r_ _[i]_


- _∩_ _Bn_ [ **0** _,_ 1]


_,_ (8)


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


where _β_ is computed as in Theorem 2, the volume of the intersection as in Theorem 1, and
capacity[in bits] is the capacity without the linear robustness constraint.
**Bound validity:** Assuming _cwh_ not too large (otherwise numerical evaluation of the bound becomes
intractable).


This bound can be numerically unstable and is tractable only for low dimensions _cwh_, high numerical
precision and large amount of sum terms kept when evaluating the intersection volume. Nevertheless,
even with this extremely conservative bound, which restricts the capacity significantly more than the
heuristic Bounds 10 to 12, we still observe 0 _._ 005 bpp for the most aggressive crop transform (see
Table 2). This might seem small but it still amounts to over 900 bits for a 256 _×_ 256px image. For
the other transformations we get capacities in excess of 3 _,_ 000 bits. Therefore, even in this strongly
conservative setup we still see capacities significantly larger than what we observe in practice.


Bound 13 relies on a severe over-approximation of a zonotope with an axis-aligned box, meaning that
it is extremely conservative. The product of singular values heuristic Bounds 10 to 12 are probably
much closer to the true capacity (and in fact, possibly also conservative, as observed in the rotated
ellipsoid case, Figure 8). Therefore in general we will use the heuristic Bounds 10 to 12.


G.4 ROBUSTNESS TO COMPRESSION


Beyond geometric transformations, watermarks should also be robust to compression. Compression
happens during normal image processing, even in the absence of attacks, and tends to strip a lot
of information from an image. Hence, it is possible that it imposes a very strong reduction on the
capacity of watermarking. The problem with compression methods, though, is that they are highly
non-linear and difficult to study analytically. Here, we analyse JPEG, arguably the most widely
used image compression method. While JPEG is non-linear, it has only one non-linear step. We
can linearize this step without deviating too much from the behaviour of classic JPEG, resulting in
LinJPEG.


The standard JPEG compression and decompression consists of the following steps (Wallace, 1991;
Wikipedia, 2025):


1. Convert the colour space from RGB to YCbCr. Linear and invertible. Appendix H.2.2.


2. Downsample the Cb and Cr channels, typically by a factor of 2 in both dimensions. Linear
but rank-deficient. Appendix H.2.5.


3. Divide each channel into 8 _×_ 8px tiles and apply steps 4–6 to each tile individually. The
Y channel would have 4 _×_ the tiles of the chroma channels because of the downsampling.
Appendix H.2.9.


4. Perform Discrete Cosine Transform (DCT) over each tile Linear and invertible. Appendix H.2.7.


5. Do element-wise multiplication with a quantization matrix (matrix values depend on the
quality setting). Linear and invertible.


6. **Round the pixel values to integers** . [non-linear, due to the rounding]


7. Perform Inverse DCT over each tile. Linear and invertible. Appendix H.2.8.


8. Upsample the Cb and Cr channels. Linear and invertible. Appendix H.2.6.


30


**Bound 13:** **Gray image, Linear transformation (Conservative), PSNR constraint.** The capacity
of a gray image _**x**_ _g_ under a linear transformation _M_ _∈_ R _[cwh][×][cwh]_ in the ambient space _A_ is
lower-bounded as

capacity under _M_ [in bits] = log2 _n_

_N_
_≥_ log2 _K_
= log2 _N_ _−_ log2 _K_





_cwh_

 


_ϵ_


_≥_ capacity[in bits] _−_ _cwh_ log2 _ϵ −_ log2 Vol


 - _[β][j]_




_∩_ _Bcwh_ [ **0** _,_ 1] _,_


_j_ =1


_[j]_

_ϵ_ _[, ][β]_ _ϵ_ _[j]_


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


Therefore, the compression mechanism of Step 6 acts by producing zeros in the lower right corner
of the DCT components _G_ . That gives us a simple way to linearize Step 6: simply drop the highest
frequencies, regardless of the value of their components. This is equivalent to a linear operator with a
diagonal matrix with 1s and 0s on the diagonal and is explicitly constructed in Appendix H.2.10. As
all the other steps are linear, we can compose them with our alternative to Step 6 to obtain LinJPEG:
a linear operator that is a very close approximation to JPEG. LinJPEG is formally constructed in
Appendix H.2.12. This is a similar strategy to JPEG-Mask proposed by (Zhu et al., 2018) as a way to


31


Quality 0 Quality 1 Quality 3 Quality 5 Quality 8 Quality 10 Quality 13 Quality 15


Figure 12: **Illustration of LinJPEG at various strengths.** We show the effect of compressing an
image with LinJPEG, our linearised variant of JPEG compression, at various quality settings. The
quality refers to the number of DCT diagonals kept for each 8 _×_ 8px tile. Quality 0 means no DCT
coefficients are kept, while quality 15 means that the image is unchanged. Quality 10 and above
produces little visual artifacts and is almost indistinguishable from the original image.


9. Convert the color space from YCbCr to RGB. Linear and invertible. Appendix H.2.3.


The compression comes from step 2, which downsamples the chroma channels, and step 6 which
efficiently encodes the frequencies which have been rounded down to 0 via entropy coding. The
lossless compression after the quantization does not change the pixel values and does not affect
capacity, hence we ignore it. Step 6 is the only non-linear step which prevents applying Bounds 10
to 13 to JPEG compression. Let’s take a closer look at a single tile (we take the example from
(Wikipedia, 2025)). When converted to DCT space ( _G_ ), the higher frequencies are represented in
the bottom and right sides of the matrix. As high-frequency components are less perceptible, the
quantization matrices _Q_ are designed to attenuate them more, resulting in more of them becoming 0
after rounding:


_−_ 415 _._ 38 _−_ 30 _._ 19 _−_ 61 _._ 20 27 _._ 24 56 _._ 12 _−_ 20 _._ 10 _−_ 2 _._ 39 0 _._ 46
4 _._ 47 _−_ 21 _._ 86 _−_ 60 _._ 76 10 _._ 25 13 _._ 15 _−_ 7 _._ 09 _−_ 8 _._ 54 4 _._ 88

_−_ 46 _._ 83 7 _._ 37 77 _._ 13 _−_ 24 _._ 56 _−_ 28 _._ 91 9 _._ 93 5 _._ 42 _−_ 5 _._ 65

_−_ 48 _._ 53 12 _._ 07 34 _._ 10 _−_ 14 _._ 76 _−_ 10 _._ 24 6 _._ 30 1 _._ 83 1 _._ 95
12 _._ 12 _−_ 6 _._ 55 _−_ 13 _._ 20 _−_ 3 _._ 95 _−_ 1 _._ 87 1 _._ 75 _−_ 2 _._ 79 3 _._ 14

_−_ 7 _._ 73 2 _._ 91 2 _._ 38 _−_ 5 _._ 94 _−_ 2 _._ 38 0 _._ 94 4 _._ 30 1 _._ 85

_−_ 1 _._ 03 0 _._ 18 0 _._ 42 _−_ 2 _._ 42 _−_ 0 _._ 88 _−_ 3 _._ 02 4 _._ 12 _−_ 0 _._ 66

_−_ 0 _._ 17 0 _._ 14 _−_ 1 _._ 07 _−_ 4 _._ 19 _−_ 1 _._ 17 _−_ 0 _._ 10 0 _._ 50 1 _._ 68


_−_ 416 _−_ 33 _−_ 60 32 48 _−_ 40 **0** **0**
**0** _−_ 24 _−_ 56 19 26 **0** **0** **0**

_−_ 42 13 80 _−_ 24 _−_ 40 **0** **0** **0**

_−_ 42 17 44 _−_ 29 **0** **0** **0** **0**
18 **0** **0** **0** **0** **0** **0** **0**
**0** **0** **0** **0** **0** **0** **0** **0**
**0** **0** **0** **0** **0** **0** **0** **0**
**0** **0** **0** **0** **0** **0** **0** **0**








_G_ =


_Q_ =


_Q_ round( _G/Q_ ) =




















16 11 10 16 24 40 51 61
12 12 14 19 26 58 60 55
14 13 16 24 40 57 69 56
14 17 22 29 51 87 80 62
18 22 37 56 68 109 103 77
24 35 55 64 81 104 113 92
49 64 78 87 103 121 120 101
72 92 95 98 112 100 103 99











**1674**

**1675**


**1676**

**1677**

**1678**

**1679**

**1680**

**1681**


**1682**

**1683**

**1684**

**1685**

**1686**

**1687**


**1688**

**1689**

**1690**

**1691**

**1692**


**1693**

**1694**

**1695**

**1696**

**1697**

**1698**


**1699**

**1700**

**1701**

**1702**

**1703**

**1704**


**1705**

**1706**

**1707**

**1708**

**1709**

**1710**


**1711**

**1712**

**1713**

**1714**

**1715**


**1716**

**1717**

**1718**

**1719**

**1720**

**1721**


**1722**

**1723**

**1724**

**1725**

**1726**

**1727**


make JPEG differentiable. However, rather than differentiating through the compression, here we are
interested in its singular values.


The way we have implemented this is to map a quality factor that is an integer from 0 to 15 inclusive
to the number of diagonals that are being zeroed out. See Figure 12 for examples. While classic
JPEG is designed to have fixed perceptual quality and variable file size, our LinJPEG instead has
variable perceptual quality and fixed file size.


As can be seen from the capacity plot Figure 4 right for LinJPEG, the higher the compression
rate, the lower the capacity, as expected. At first, it may seem strange that quality 15, where no
frequency components are dropped, still has a 50% reduction in capacity. However, that is due to
the downsampling of the chroma channels, which reduces the effective number of pixels we have
for watermarking by half. This can also be seen by half of the singular values being 0 (Figure 11
right). Interestingly, again, even with relatively high compression rates (quality 8) we still observe
capacities of more than 1 bpp at 40 dB. Finally, as can be seen in Table 2, even the conservative
bound from Appendix G.3 gives us 0 _._ 13 bpp or 25 _,_ 559 bits for a 256 _×_ 256px image. Therefore,
although (Lin)JPEG compression removes a lot of information from the image, it still leaves plenty
of watermarking capacity.


32


**1728**

**1729**


**1730**

**1731**

**1732**

**1733**

**1734**

**1735**


**1736**

**1737**

**1738**

**1739**

**1740**

**1741**


**1742**

**1743**

**1744**

**1745**

**1746**


**1747**

**1748**

**1749**

**1750**

**1751**

**1752**


**1753**

**1754**

**1755**

**1756**

**1757**

**1758**


**1759**

**1760**

**1761**

**1762**

**1763**

**1764**


**1765**

**1766**

**1767**

**1768**

**1769**


**1770**

**1771**

**1772**

**1773**

**1774**

**1775**


**1776**

**1777**

**1778**

**1779**

**1780**

**1781**


Thus, we have the bias
_**c**_ = _B_ [0 _, −_ 128 _, −_ 128] _[⊤]_ _._


To get a transformation for the whole image, we again use the pixel-wise transform we defined above:


( _A,_ _**b**_ ) = Pixel-wise Transformation[ _B,_ _**c**_ ] _._


33


H AUGMENTATIONS AS LINEAR TRANSFORMATIONS


This section describes a series of linear transformations used for the robustness bounds in Section 2.5.
Each transformation is defined by a matrix _A_ and a bias vector _**b**_, and acts on an input vector _**x**_ as
_f_ ( _**x**_ ) = _A_ _**x**_ + _**b**_ . Some transformations are compositions of others, as described below. We will use
the pipe operator _|_ for left-to-right composition, rather than the classical _◦_ operator for right-to-left
composition.


H.1 GENERIC LINEAR TRANSFORMATION


A generic linear transformation has the form _f_ ( _**x**_ ) = _A_ _**x**_ + _**b**_, where _A_ is a matrix and _**b**_ is a bias
vector. This transformation can be applied to vectors or images (flattened as vectors, with each third
corresponding to one channel).


_f_ ( _**x**_ ) = _A_ _**x**_ + _**b**_


H.2 LINJPEG AND ITS BUILDING BLOCKS


H.2.1 PIXEL-WISE TRANSFORMATION

The pixel-wise transform applies a given linear transformation _g_ ( _**p**_ ) = _B_ _**p**_ + _**c**_ independently to
each pixel across an image. If the base transformation maps _c_ in input channels to _c_ out output channels
(i.e., _B_ _∈_ R _[c]_ [out] _[×][c]_ [in] ) the overall matrix _A_ is constructed as a block matrix, where each block is an
identity matrix of size _w × h_ (the number of pixels), scaled by the corresponding entry in the base
transformation’s matrix. The bias _**b**_ is the base bias _**c**_ repeated for every pixel.








_**c**_
...
_**c**_















 _**b**_ =


_A_ =





_B_ 1 _,_ 1 _Iwh_ _. . ._ _B_ 1 _,c_ in _Iwh_
... ... ...
_Bc_ out _,_ 1 _Iwh_ _. . ._ _Bc_ out _,c_ in _Iwh_





_wh_ times



H.2.2 RGB TO YCBCR CONVERSION

Color-space transformations are pixel-wise transformations. To convert the RGB color space to the
YCbCr color space we have the base matrix:


_B_ =


- 0 _._ 299 0 _._ 587 0 _._ 114 
_−_ 0 _._ 168736 _−_ 0 _._ 331364 0 _._ 5
0 _._ 5 _−_ 0 _._ 418688 _−_ 0 _._ 081312


The bias is _**c**_ = [0 _,_ 128 _,_ 128]. We can use the pixel-wise transform we defined above to obtain the
linear transform for applying this across the image:


( _A,_ _**b**_ ) = Pixel-wise Transformation[ _B,_ _**c**_ ] _._


H.2.3 YCBCR TO RGB CONVERSION

To convert the YCbCr color space back to RGB, we need the composition of two transforms: First,
subtract 128 from Cb and Cr channels, then apply the matrix:


_B_ =


�1 0 1 _._ 402
1 _−_ 0 _._ 344136 _−_ 0 _._ 714136
1 1 _._ 772 0


**1782**

**1783**


**1784**

**1785**

**1786**

**1787**

**1788**

**1789**


**1790**

**1791**

**1792**

**1793**

**1794**

**1795**


**1796**

**1797**

**1798**

**1799**

**1800**


**1801**

**1802**

**1803**

**1804**

**1805**

**1806**


**1807**

**1808**

**1809**

**1810**

**1811**

**1812**


**1813**

**1814**

**1815**

**1816**

**1817**

**1818**


**1819**

**1820**

**1821**

**1822**

**1823**


**1824**

**1825**

**1826**

**1827**

**1828**

**1829**


**1830**

**1831**

**1832**

**1833**

**1834**

**1835**


_D_ _[′]_ = reshape(0 _._ 25 _**αα**_ _[⊤]_ _,_ (1 _,_ 64)) _⊙_ reshape( _D,_ (64 _,_ 64)) _,_

_h_ ( _**x**_ ) = _D_ _[′]_ _**x**_ _,_
_f_ ( _**x**_ ) = ( _h_ _|_ _g_ )( _**x**_ ) _._


34


H.2.4 TAKE ROWS AND COLUMNS TRANSFORMATION
This transformation selects specific rows and columns from an input image, effectively downsampling
or upsampling. The matrix _A_ is constructed such that each output pixel corresponds to a specific input
pixel, with a 1 at the appropriate position and 0 elsewhere. The bias _**b**_ is zero. Assume the indices are
provided as two lists row indices and col indices. Then we can set the non-zero elements of _A_ as:


for ir, r in enumerate(row indices):
for ic, c in enumerate(col indices):
for chan in range(channels):
A[
chan      - (len(row indices)      - len(col indices)) + ir      - len(col indices) + ic,
chan      - w      - h + r      - w + c,
] = 1


H.2.5 DOWNSAMPLING TRANSFORMATION
The downsample transformation reduces the resolution of an image by a fixed factor. The matrix _A_
selects every _k_ -th row and column. The bias _**b**_ is zero.


(A, b) = TakeRowsAndColumns�row indices = [1 _,_ 1 + _k, . . ., h −_ _k_ + 1] _,_ col ~~i~~ ndices = [1 _,_ 1 + _k, . . ., w −_ _k_ + 1]� _._


H.2.6 UPSAMPLING TRANSFORMATION
The upsampling transformation increases the resolution by repeating rows and columns _k_ times. The
matrix _A_ selects every row and column _k_ times. The bias _**b**_ is zero.


(A, b) = TakeRowsAndColumns�row ~~i~~ ndices=[1 _, . . .,_ 1 _, . . ., h . . ., h_ ] _,_ col indices=[1 _, . . .,_ 1 _, . . ., w . . ., w_ ]� _._

                       - ��                        -                        - ��                        - ���                        -                        - ��                        _×k_ _×k_ _×k_ _×k_


H.2.7 DISCRETE COSINE TRANSFORM (DCT) FOR 8X8 BLOCKS
The Dct8x8 transformation applies the DCT to a single-channel 8 _×_ 8px block. It first subtracts 128
from each pixel so that the values are centred at 0, then applies the DCT matrix. The subtraction can
be done with:
_g_ ( _**x**_ ) = _I_ 64 _**x**_ _−_ 128 _·_ **1** _._


The DCT matrix is constructed using the four-dimensional tensor _D_ _∈_ R [8] _[×]_ [8] _[×]_ [8] _[×]_ [8] :


    - (2 _x_ + 1) _uπ_
_Dx,y,u,v_ = cos
16


- - (2 _y_ + 1) _vπ_
cos
16


_D_ _[′]_ = reshape(0 _._ 25 _**αα**_ _[⊤]_ _,_ (1 _,_ 64)) _⊙_ reshape( _D,_ (64 _,_ 64)) _,_

_√_
where _**α**_ 1 = 1 _/_ 2, _**α**_ _i_ = 1 for 2 _≤_ _i ≤_ 8 and _⊙_ is an element-wise multiplication with broadcasting.

This results in
_h_ ( _**x**_ ) = _D_ _[′]_ _**x**_ _._


The overall transformation is:
_f_ ( _**x**_ ) = ( _g_ _|_ _h_ )( _**x**_ ) _._


H.2.8 INVERSE DCT FOR 8X8 BLOCKS
The iDct8x8 transformation applies the inverse DCT to a single-channel 8 _×_ 8px block. The matrix is
constructed similarly to the DCT, but with the roles of _x, y_ and _u, v_ swapped. After the transformation,
128 is added to each pixel to restore the original range.


_g_ ( _**x**_ ) = _I_ 64 _**x**_ + 128 _·_ **1** _,_


    - (2 _u_ + 1) _xπ_
_Dx,y,u,v_ = cos
16


- - (2 _v_ + 1) _uπ_
cos
16


_,_


**1836**

**1837**


**1838**

**1839**

**1840**

**1841**

**1842**

**1843**


**1844**

**1845**

**1846**

**1847**

**1848**

**1849**


**1850**

**1851**

**1852**

**1853**

**1854**


**1855**

**1856**

**1857**

**1858**

**1859**

**1860**


**1861**

**1862**

**1863**

**1864**

**1865**

**1866**


**1867**

**1868**

**1869**

**1870**

**1871**

**1872**


**1873**

**1874**

**1875**

**1876**

**1877**


**1878**

**1879**

**1880**

**1881**

**1882**

**1883**


**1884**

**1885**

**1886**

**1887**

**1888**

**1889**


H.2.12 LINJPEG TRANSFORM


The LinJPEG transformation approximates JPEG compression with quality _q_ as a sequence of linear
transforms, as explained in Appendix G.4. In a nutshell, we need to convert the colour space from
RGB to YCbCr and then downsample the Cb and Cr channels by a factor of 2. We then need to
apply the DCT, filter and iDCT operations on 8 _×_ 8px tiles of each channel. Then we need to upsample
the Cb and Cr channels to restore them to the original resolution and convert the colour space back
to RGB. We can build a single linear transform with its _A_ and _**b**_ doing all that by composing the
building blocks defined above:


35


H.2.9 TILING TRANSFORMATION


The tiling transformation divides an image into tiles and applies a linear transform to each tile. This
operates over a single channel. Define the linear transformation for a tile as _g_ ( _**t**_ ) = _B_ _**t**_ + _**c**_, with
_B_ _∈_ R [ts][2] _[×]_ [ts][2] _,_ _**t**_ _,_ _**c**_ _∈_ R [ts][2], ts being the size of the tile (typically 8 for our use-cases). We expect that
the ts divides the width _w_ and height _h_ of the image. The matrix _A_ is constructed by placing the base
transform’s matrix along the diagonal for each tile, so that each tile is transformed independently. The
bias _**b**_ is constructed by tiling the base bias for each tile. The _A_ and _**b**_ of the resulting transformation
can then be computed using this pseudo-code:


hor˙tiles = width // ts
ver˙tiles = height // ts


tiles = split(B, ts, axis=0)
tiles = [split(s, ts, axis=1) for s in tiles] # ts lists of ts tiles of ts x ts size


per˙n˙rows = block˙matrix(

[[block˙diag([tile]   - hor˙tiles) for tile in htiles] for htiles in tiles]
)


A = block˙diag([per˙n˙rows] - ver˙tiles)


bias˙tiles = split(transform.bias, tile˙side, axis=0)
bias = concatenate([tile(bias˙tile, hor˙tiles) for bias˙tile in bias˙tiles])
b = tile(bias, ver˙tiles)


H.2.10 JPEG FILTER


This filter transformation drops the lowest frequencies in an 8 _×_ 8 block according to a quality factor.
This is the linearized replacement for the quantization operation in the standard JPEG compression
algorithm, as discussed in Appendix G.4. The quality factor _q_ designates the number of diagonals
kept: from _q_ = 0 for the worst quality where all diagonals are masked off, to _q_ = 15 where all
diagonals are kept. The resulting _A ∈_ R [64] _[×]_ [64] can be constructed with the following pseudo-code:


matrix = triu(ones(8, 8), k=16 - 8 - q)[:, ::-1]
A = diag(matrix.flatten())


H.2.11 PER-CHANNEL TRANSFORMATION


The Per-Channel Transform applies a separate linear transformation to each channel of an image.
The matrix _A_ is block-diagonal, with each block being the matrix for the corresponding channel. The
bias _**b**_ is the concatenation of the biases for each channel. Given _c_ channels, each with its own linear
transformation _Ai,_ _**b**_ _i_, 1 _≤_ _i ≤_ _c_, the resulting linear transform has:


_A_ = diag( _A_ 1 _, . . ., Ac_ ) _,_








 _._


_**b**_ =





_**b**_ 1
...
_**b**_ _c_


**1890**

**1891**

**1892**

**1893**

**1894**

**1895**


**1896**

**1897**

**1898**

**1899**

**1900**

**1901**


**1902**

**1903**

**1904**

**1905**

**1906**

**1907**


**1908**

**1909**

**1910**

**1911**

**1912**


**1913**

**1914**

**1915**

**1916**

**1917**

**1918**


**1919**

**1920**

**1921**

**1922**

**1923**

**1924**


**1925**

**1926**

**1927**

**1928**

**1929**

**1930**


**1931**

**1932**

**1933**

**1934**

**1935**


**1936**

**1937**

**1938**

**1939**

**1940**

**1941**


**1942**

**1943**


( _AY,_ _**b**_ _Y_ ) = Tile [Dct8x8 _|_ JpegFilter[ _q_ ] _|_ iDct8x8 _,_ width = _w,_ height = _h_ ]
( _AC,_ _**b**_ _C_ ) = Downsample[ _k_ = 2]

_|_ Tile�Dct8x8 _|_ JpegFilter[ _q_ ] _|_ iDct8x8 _,_ width = _w/_ 2 _,_ height = _h/_ 2�


_|_ Upsample[ _k_ = 2]
( _A,_ _**b**_ ) = RGBToYCbCr _|_ PerChannelTransform[( _AY,_ _**b**_ _Y_ ) _,_ ( _AC,_ _**b**_ _C_ ) _,_ ( _AC,_ _**b**_ _C_ )] _|_ YCbCrToRGB _._


H.3 PIXEL MAPPING TRANSFORMS

A number of transforms can be defined via a map _µ_ ( _x, y_ ) _�→_ ( _u, v_ ) that says which point ( _u, v_ ) in the
original image corresponds to a pixel ( _x, y_ ) in the transformed image. Note while _x_ and _y_ are integer
coordinates, _u_ and _v_ need not be. Thus, some sort of interpolation will be needed. Here we consider
both nearest neighbour and bilinear interpolation. It is interesting that in both of these case, given a
fixed map _µ_, its application with interpolation is a linear operation. If _µ_ is itself parameterized (e.g.,
the angle of rotation), then the transformation generally is not linear in the parameter. That is why we
consider _µ_ with different parameters to be distinct transformations.


H.3.1 NEAREST NEIGHBOUR PIXEL MAPPING
The matrix _A_ is binary, with _Ai,j_ = 1 if output pixel _i_ maps to input pixel _j_ . Below we show
the construction for a pixel map mu and a single channel. For multi-channel images, the same
transformation can be applied to each channel using PerChannelTransform.


mesh˙x, mesh˙y = meshgrid(range(width), range(height))
mesh˙x = flatten(mesh˙x)
mesh˙y = flatten(mesh˙y)
mapped˙x, mapped˙y = vectorize(mu)(mesh˙x, mesh˙y)
mapped˙x = clip(round(mapped˙x), 0, width - 1)
mapped˙y = clip(round(mapped˙y), 0, height - 1)


A = zeros(width - height, width - height)
target˙indices = mesh˙x + mesh˙y - width
source˙indices = mapped˙x + mapped˙y - width
A[target˙indices, source˙indices] = 1


H.3.2 BILINEAR PIXEL MAPPING
The matrix _A_ is constructed so that each output pixel is a weighted sum of the four nearest input
pixels, with weights determined by the mapping. Below we show the construction for a pixel map
mu and a single channel. For multi-channel images, the same transformation can be applied to each
channel using PerChannelTransform.


mesh˙x, mesh˙y = meshgrid(range(width), range(height))
mesh˙x = flatten(mesh˙x)
mesh˙y = flatten(mesh˙y)
mapped˙x, mapped˙y = vectorize(mu)(mesh˙x, mesh˙y)


def get˙corners(mapped: array, range: int) -¿ tuple[array, array]:
l = floor(mapped)
u = ceil(mapped)
# we need to ensure that the lower and the upper bounds are not the same
u = where(u == l, l + 1, u)
return l, u


mapped˙x˙l, mapped˙x˙u = get˙corners(mapped˙x, width)
mapped˙y˙l, mapped˙y˙u = get˙corners(mapped˙y, height)


denom = (mapped˙x˙u - mapped˙x˙l) - (mapped˙y˙u - mapped˙y˙l)
w11 = (mapped˙x˙u - mapped˙x) - (mapped˙y˙u - mapped˙y) / denom
w12 = (mapped˙x˙u - mapped˙x) - (mapped˙y - mapped˙y˙l) / denom
w21 = (mapped˙x - mapped˙x˙l) - (mapped˙y˙u - mapped˙y) / denom
w22 = (mapped˙x - mapped˙x˙l) - (mapped˙y - mapped˙y˙l) / denom


36


**1944**

**1945**


**1946**

**1947**

**1948**

**1949**

**1950**

**1951**


**1952**

**1953**

**1954**

**1955**

**1956**

**1957**


**1958**

**1959**

**1960**

**1961**

**1962**


**1963**

**1964**

**1965**

**1966**

**1967**

**1968**


**1969**

**1970**

**1971**

**1972**

**1973**

**1974**


**1975**

**1976**

**1977**

**1978**

**1979**

**1980**


**1981**

**1982**

**1983**

**1984**

**1985**


**1986**

**1987**

**1988**

**1989**

**1990**

**1991**


**1992**

**1993**

**1994**

**1995**

**1996**

**1997**


The transformation can be applied with either the nearest neighbour or the bilinear interpolation
methods, by default we use bilinear.


37


# Create the matrix for a single channel
target˙indices = mesh˙x + mesh˙y - width


mapped˙x˙l = clip(mapped˙x˙l, 0, width - 1)
mapped˙x˙u = clip(mapped˙x˙u, 0, width - 1)
mapped˙y˙l = clip(mapped˙y˙l, 0, height - 1)
mapped˙y˙u = clip(mapped˙y˙u, 0, height - 1)


indices˙11 = mapped˙x˙l + mapped˙y˙l - width
indices˙12 = mapped˙x˙l + mapped˙y˙u - width
indices˙21 = mapped˙x˙u + mapped˙y˙l - width
indices˙22 = mapped˙x˙u + mapped˙y˙u - width


A = zeros(width - height, width - height)
A[target˙indices, indices˙11] += w11
A[target˙indices, indices˙12] += w12
A[target˙indices, indices˙21] += w21
A[target˙indices, indices˙22] += w22


H.3.3 HORIZONTAL FLIP TRANSFORMATION
The pixel mapping is _µ_ : ( _x, y_ ) _�→_ (width _−_ 1 _−_ _x, y_ ). The transformation can be applied with either
the nearest neighbour or the bilinear interpolation methods, by default we use nearest neighbour but
bilinear should give the exact same result here.


H.3.4 VERTICAL FLIP TRANSFORMATION
The pixel mapping is _µ_ : ( _x, y_ ) _�→_ ( _x,_ height _−_ 1 _−_ _y_ ). The transformation can be applied with either
the nearest neighbour or the bilinear interpolation methods, by default we use nearest neighbour but
bilinear should give the exact same result here.


H.3.5 CENTRE CROP AND RESCALE TRANSFORMATION
This transformation crops the centre of an image to a given scale and rescales it to the original size.
The pixel mapping _µ_ for a fixed scale _s_ is:


��
( _x, y_ ) _�→_ _x −_ [width]

2


- 
_· s_ + [width] _,_ _y −_ [height]
2 2


_· s_ + [height]
2


_._


The transformation can be applied with either the nearest neighbour or the bilinear interpolation
methods, by default we use bilinear.


H.3.6 ROTATION TRANSFORMATION
This rotation transformation rotates an image around its centre by a fixed angle _θ_ . The pixel mapping
_µ_ is:


- sin _θ_ + _y −_ [height]

2


sin _θ_ + [width] _,_

2


( _x, y_ ) _�→_


��
_x −_ [width]

2


 _x −_ [width]

2


- cos _θ −_ _y −_ [height]

2


cos _θ_ + [height]

2


cos _θ_ + [height]


_._


**1998**

**1999**


**2000**

**2001**

**2002**

**2003**

**2004**

**2005**


**2006**

**2007**

**2008**

**2009**

**2010**

**2011**


**2012**

**2013**

**2014**

**2015**

**2016**


**2017**

**2018**

**2019**

**2020**

**2021**

**2022**


**2023**

**2024**

**2025**

**2026**

**2027**

**2028**


**2029**

**2030**

**2031**

**2032**

**2033**

**2034**


**2035**

**2036**

**2037**

**2038**

**2039**


**2040**

**2041**

**2042**

**2043**

**2044**

**2045**


**2046**

**2047**

**2048**

**2049**

**2050**

**2051**


                   - ��                    _Z_


where we use the fact that the intersection with a ball of radius _r_ needs to be contained within the
hypercube [ _−r, r_ ] _[n]_, that _V_ [˜] has orthonormal columns and that the two sets making up the sum in _Z_
are orthogonal. [4]


Computing the volume of this intersection in the general case is computationally intractable. However,
if we over-approximate the left-hand side of the intersection ( _Z_ ) with a (rotated) box, then we can
use our previous results on the volume of box-ball intersections, in particular Theorem 1.


4Follows from the properties of the pseudo-inverse:   - _M_ [+] _M_   - _⊤_ = _M_ + _M_ and _M_ + _MM_ + = _M_ +.


38


I PROOF OF THEOREM 2


Let _**b**_ be a point in R _[n]_ and _C_ = _**b**_ + [0 _,_ 1] _[n]_ be the hypercube with a corner at _**b**_ . Given _**x**_ _∈_ _C_, the
preimage of _fM_ can be obtained using the Moore-Penrose pseudo-inverse _M_ [+] of _M_ :

_fM_ _[−]_ [1][(] _**[x]**_ [) =]         - _M_ [+] _**x**_ + [ _I_ _−_ _M_ [+] _M_ ] _**w**_ _|_ _**w**_ _∈_ R _[n]_ [�] _._


Hence:


_fM_ _[−]_ [1][(] _[C]_ [) =]     - _M_ [+] _**b**_ + _M_ [+] _**p**_ + ( _I_ _−_ _M_ [+] _M_ ) _**w**_ _|_ _**w**_ _∈_ R _[n]_ _,_ _**p**_ _∈_ [0 _,_ 1] _[n]_ [�]

=       - _M_ [+] _**b**_       - _⊕_ _M_ [+] [0 _,_ 1] _[n]_ _⊕_ ( _I_ _−_ _M_ [+] _M_ )R _[n]_ (10)


��
_⊕_ [1] _[⊕]_ [(] _[I]_ _[−]_ _[M]_ [ +] _[M]_ [)][R] _[n][,]_

2 _[M]_ [ +][[] _[−]_ [1] _[,]_ [ 1]] _[n]_


 - = _M_ [+] _**b**_ + [1]

2


 - = _M_ [+] _**b**_ + [1]


where _⊕_ is the Minkowski sum. Take _U_ Σ _V_ _[⊤]_ = _M_ to be the SVD decomposition of _M_ . _U_ and _V_
are orthonormal matrices, while Σ is a diagonal matrix with non-negative entries. We will assume
that the singular values on the diagonal are sorted in descending order, hence the first rank[ _M_ ] values
on the diagonal are non-zero and the rest are zero. The pseudo-inverse of _M_ can be conveniently
expressed as _M_ [+] = _V_ Σ [+] _U_ _[⊤]_ . We can use that the pseudo-inverse of diagonal matrices can be
constructed by taking the reciprocals of the non-zero elements on the diagonal, leaving the zeros
unchanged. Thus we have:


( _I_ _−_ _M_ [+] _M_ )R _[n]_ = ( _I_ _−_ _V_ Σ [+] _U_ _[⊤]_ _U_ Σ _V_ _[⊤]_ )R _[n]_

= ( _I_ _−_ _V_ Σ [+] Σ _V_ _[⊤]_ )R _[n]_ ( _U_ _[⊤]_ _U_ = _I_ as _U_ is orthonormal)

= ( _V V_ _[⊤]_ _−_ _V_ Σ [+] Σ _V_ _[⊤]_ )R _[n]_ ( _V V_ _[⊤]_ = _I_ as _V_ is orthonormal)

= _V_ ( _I_ _−_ Σ [+] Σ) _V_ _[⊤]_ R _[n]_

= _V_ ( _I_ _−_ diag[ **1** rank[ _M_ ] _,_ **0** _n−_ rank[ _M_ ]]) _V_ _[⊤]_ R _[n]_

= _V_ diag[ **0** rank[ _M_ ] _,_ **1** _n−_ rank[ _M_ ]] _V_ _[⊤]_ R _[n]_

= _V_ [˜] _V_ [˜] _[⊤]_ R _[n]_ ( _V_ [˜] = _V_ [: _,_ rank[ _M_ ]+1:] _∈_ R _[n][×]_ [(] _[n][−]_ [rank[] _[M]_ [])] )

= _V_ [˜] R _[n][−]_ [rank[] _[M]_ []] ( _V_ [˜] _[⊤]_ R _[n]_ = R _[n][−]_ [rank[] _[M]_ []] ) _._


Thus, combining with Equation (10), we have that the set of images that would be mapped by _M_ to
images in the cube _C_, i.e., after quantization, is the following polytope:


   -    _fM_ _[−]_ [1][(] _[C]_ [) =] _M_ [+] _**b**_ + [1]

2


��
_⊕_ [1] _[⊕]_ _[V]_ [˜][ R] _[n][−]_ [rank[] _[M]_ []] _[.]_

2 _[V]_ [ Σ][+] _[U][ ⊤]_ [[] _[−]_ [1] _[,]_ [ 1]] _[n]_


��
_⊕_ [1]


The volume of the intersection in Equation (8) is maximized when the ball centre _**c**_ coincides with
the polytope centre _M_ [+] ( _**b**_ + [1] _/_ 2). Furthermore, because the volume of the intersection is invariant
to shifts, we can simplify the left-hand side of Equation (8) to:


sup - _fM_ _[−]_ [1][(] _[C]_ [)] _[∩]_ _[B][n]_ [ [] _**[c]**_ _[, r]_ []] - = Vol �� 1 _[⊕]_ _[V]_ [˜][ R] _[n][−]_ [rank[] _[M]_ []] - _∩_ _Bn_ [ **0** _, r_ ]�
_**c**_ _∈_ R _[n]_ [ Vol] 2 _[V]_ [ Σ][+] _[U][ ⊤]_ [[] _[−]_ [1] _[,]_ [ 1]] _[n]_


�� 1   -   = Vol _[⊕]_ _[r]_ [ ˜] _[V]_ [ [] _[−]_ [1] _[,]_ [ 1]] _[n][−]_ [rank[] _[M]_ []] _∩_ _Bn_ [ **0** _, r_ ] _,_
2 _[V]_ [ Σ][+] _[U][ ⊤]_ [[] _[−]_ [1] _[,]_ [ 1]] _[n]_


**2052**

**2053**


**2054**

**2055**

**2056**

**2057**

**2058**

**2059**


**2060**

**2061**

**2062**

**2063**

**2064**

**2065**


**2066**

**2067**

**2068**

**2069**

**2070**


**2071**

**2072**

**2073**

**2074**

**2075**

**2076**


**2077**

**2078**

**2079**

**2080**

**2081**

**2082**


**2083**

**2084**

**2085**

**2086**

**2087**

**2088**


**2089**

**2090**

**2091**

**2092**

**2093**


**2094**

**2095**

**2096**

**2097**

**2098**

**2099**


**2100**

**2101**

**2102**

**2103**

**2104**

**2105**


To upper-bound _Z_ with a (rotated) box we first observe that it is the Minkowski sum of two zonotopes
and hence is a zonotope itself. A zonotope is defined as

     - _p_     


_,_


_**x**_ _∈_ R _[n]_ : _**x**_ = _**c**_ +


_p_


_ξi_ _**g**_ _i,_ _ξi_ _∈_ [ _−_ 1 _,_ 1] _∀i_ = 1 _, . . ., p_

_i_ =1


where _**c**_ _∈_ R _[n]_ is its centre and _G_ = _{_ _**g**_ _i}_ _[p]_ _i_ =1 [,] _**[ g]**_ _[i]_ _[∈]_ [R] _[n]_ [is the set of its generators.] [A zonotope is,]
equivalently the Minkowski sum of line segments. Thus, the Minkowski sum of zonotopes is also a
zonotope. Its centre is found by adding together the centres of the original zonotopes, and its set of
generators is just all the generators from both shapes combined (Schneider, 2013). Now, it is clear
that _Z_ is a zonotope as it is the Minkowski sum of two other zonotopes.


The _V_ and _V_ [˜] matrices simply rotate the resulting zonotope. As the ball _Bn_ [ **0** _, r_ ] is rotation invariant,
we can ignore this rotation. That will help us with tightening the box approximation as the _n −_
rank[ _M_ ] dimensions of the second zonotope will now be automatically axis-aligned. Thus we have
(up to a rotation):


_Z_ =    - 12 [Σ][+] _[U][ ⊤][,]_ _rIn_ [: _,_ rank _M_ +1:]� [ _−_ 1 _,_ 1] [2] _[n][−]_ [rank] _[ M]_ = _G_ [ _−_ 1 _,_ 1] [2] _[n][−]_ [rank] _[ M]_ _,_


with _G ∈_ R _[n][×]_ [(2] _[n][−]_ [rank] _[ M]_ [)] being its 2 _n −_ rank _M_ _n_ -dimensional generators.

A zonotope with generators _G_ is contained in an axis-aligned box [�] _i_ _[n]_ =1 [[] _[−][β]_ _i_ _[′][, β]_ _i_ _[′]_ []][, where] _**[ β]**_ [is the]
sum of absolute values of _G_ across the generators: _**β**_ _[′]_ = abs[ _G_ ] **1** 2 _n−_ rank[ _M_ ] = abs[ [1] 2 [Σ][+] _[U][ ⊤]_ []] **[1]** _[n]_ [ +]

[ **0** _[⊤]_ rank[ _M_ ] _[, r]_ **[1]** _[⊤]_ _n−_ rank[ _M_ ] []] _[⊤]_ [(][Girard][,][ 2005][;][ Althoff et al.][,][ 2010][).]


Note that this over-approximation can be extremely loose: this is the step that makes Bound 13 so
conservative. However, with this, we can now apply Equation (4) and Theorem 1 for the box-ball
intersection:

sup - _fM_ _[−]_ [1][(] _[C]_ [)] _[∩]_ _[B][n]_ [ [] _**[c]**_ _[, r]_ []] - = Vol �� 1 _[⊕]_ _[r]_ [ ˜] _[V]_ [ [] _[−]_ [1] _[,]_ [ 1]] _[n][−]_ [rank[] _[M]_ []] - _∩_ _Bn_ [ **0** _, r_ ]�
_**c**_ _∈_ R _[n]_ [ Vol] 2 _[V]_ [ Σ][+] _[U][ ⊤]_ [[] _[−]_ [1] _[,]_ [ 1]] _[n]_


��                = Vol _G_ [ _−_ 1 _,_ 1] [2] _[n][−]_ [rank[] _[M]_ []][�] _∩_ _Bn_ [ **0** _, r_ ]


_≤_ Vol [([ _−β_ 1 _, β_ 1] _× · · · ×_ [ _−βn, βn_ ]) _∩_ _Bn_ [ **0** _, r_ ]]


���
= _r_ _[n]_ Vol _−_ _[β]_ [1]


_r_


_r_


_[β]_ [1]

_r_ _[, ][β]_ _r_ [1]


- _× · · · ×_ _−_ _[β][n]_


_[n]_

_r_ _[, ][β]_ _r_ _[n]_


�� _∩_ _Bn_ [ **0** _,_ 1] _._


39


**2106**

**2107**


**2108**

**2109**

**2110**

**2111**

**2112**

**2113**


**2114**

**2115**

**2116**

**2117**

**2118**

**2119**


**2120**

**2121**

**2122**

**2123**

**2124**


**2125**

**2126**

**2127**

**2128**

**2129**

**2130**


**2131**

**2132**

**2133**

**2134**

**2135**

**2136**


**2137**

**2138**

**2139**

**2140**

**2141**

**2142**


**2143**

**2144**

**2145**

**2146**

**2147**


**2148**

**2149**

**2150**

**2151**

**2152**

**2153**


**2154**

**2155**

**2156**

**2157**

**2158**

**2159**


J COMPREHENSIVE RESULTS FOR CHUNKYSEAL


J.1 EXTENDED EVALUATION


Table 4: **Extended results of Chunky Seal on SA-1B (Kirillov et al., 2023).**


Chunky Seal (ours) Video Seal Video Seal
HiDDeN MBRS TrustMark WAM
1024bit, 256px 256bit, 256px 96bit, 256px


Capacity 1024 bits 256 bits 96 bits 48 bits 256 bits 100 bits 32 bits
PSNR 45.32 dB 44.42 dB 53.19 dB 30.41 dB 45.54 dB 42.29 dB 38.19 dB
SSIM 0.9945 0.9963 0.9995 0.9299 0.9962 0.9941 0.9842
MS-SSIM 0.9966 0.9972 0.9993 0.9062 0.9967 0.9944 0.9877
LPIPS 0.0085 0.0019 0.0028 0.2021 0.0044 0.0028 0.0446
Embedding Time 0.27 s 0.06 s 0.06 s 0.08 s 0.08 s 0.07 s 0.15 s
Extraction Time 0.05 s 0.01 s 0.01 s 0.01 s 0.01 s 0.01 s 0.02 s
Bit Acc. 99.74% 99.90% 97.92% 92.19% 98.74% 99.81% 100.00%
Bit Acc. (Horizontal Flip) 99.65% 99.89% 97.20% 64.06% 50.63% 99.81% 100.00%
Bit Acc. (Rotate 5°) 99.29% 99.37% 94.79% 80.99% 50.21% 56.88% 98.83%
Bit Acc. (Rotate 10°) 97.26% 98.31% 90.26% 72.22% 50.42% 48.23% 75.00%
Bit Acc. (Rotate 30°) 49.56% 51.30% 54.93% 50.09% 51.50% 49.65% 51.82%
Bit Acc. (Rotate 45°) 50.01% 50.18% 50.20% 46.88% 51.50% 50.96% 50.91%
Bit Acc. (Rotate 90°) 51.37% 81.07% 49.08% 49.57% 50.50% 49.42% 50.78%
Bit Acc. (Resize 32%) 99.75% 99.89% 97.92% 90.36% 98.24% 99.81% 100.00%
Bit Acc. (Resize 45%) 99.73% 99.90% 97.88% 91.06% 98.51% 99.81% 100.00%
Bit Acc. (Resize 55%) 99.73% 99.90% 97.88% 91.23% 98.60% 99.81% 100.00%
Bit Acc. (Resize 63%) 99.74% 99.90% 97.92% 91.67% 98.65% 99.81% 100.00%
Bit Acc. (Resize 71%) 99.73% 99.90% 97.96% 91.75% 98.69% 99.81% 100.00%
Bit Acc. (Resize 77%) 99.74% 99.89% 97.88% 91.75% 98.68% 99.81% 100.00%
Bit Acc. (Resize 84%) 99.74% 99.90% 97.92% 92.01% 98.68% 99.81% 100.00%
Bit Acc. (Resize 89%) 99.74% 99.90% 97.92% 91.84% 98.69% 99.81% 100.00%
Bit Acc. (Resize 95%) 99.74% 99.90% 97.92% 92.01% 98.71% 99.81% 100.00%
Bit Acc. (Crop 32%) 49.71% 50.24% 50.84% 48.70% 49.95% 49.65% 79.30%
Bit Acc. (Crop 45%) 65.22% 50.70% 51.52% 48.35% 50.59% 51.73% 94.14%
Bit Acc. (Crop 55%) 86.90% 52.73% 63.58% 57.03% 50.20% 51.58% 96.22%
Bit Acc. (Crop 63%) 93.42% 66.70% 79.13% 64.06% 49.77% 51.81% 97.79%
Bit Acc. (Crop 71%) 95.85% 87.34% 86.74% 69.44% 50.26% 56.42% 98.83%
Bit Acc. (Crop 77%) 97.13% 95.33% 91.83% 74.39% 50.57% 92.85% 99.35%
Bit Acc. (Crop 84%) 97.77% 98.31% 93.47% 79.86% 50.59% 99.88% 99.61%
Bit Acc. (Crop 89%) 98.81% 98.97% 94.83% 82.47% 50.38% 99.92% 99.22%
Bit Acc. (Crop 95%) 99.29% 99.56% 95.03% 82.90% 50.93% 99.92% 99.22%
Bit Acc. (Brightness 10%) 83.62% 83.17% 82.69% 51.13% 61.76% 80.04% 96.48%
Bit Acc. (Brightness 25%) 99.57% 98.93% 95.43% 56.25% 84.39% 97.19% 100.00%
Bit Acc. (Brightness 50%) 99.74% 99.76% 97.48% 75.52% 95.01% 99.54% 100.00%
Bit Acc. (Brightness 75%) 99.73% 99.84% 98.04% 86.81% 97.91% 99.62% 100.00%
Bit Acc. (Brightness 125%) 99.22% 98.91% 94.47% 94.36% 95.42% 97.00% 100.00%
Bit Acc. (Brightness 150%) 97.26% 96.18% 87.78% 93.23% 91.03% 92.42% 100.00%
Bit Acc. (Brightness 175%) 95.74% 94.48% 83.13% 93.06% 88.79% 90.35% 99.22%
Bit Acc. (Brightness 200%) 95.10% 92.85% 80.17% 92.88% 86.93% 88.31% 99.48%
Bit Acc. (Contrast 10%) 99.46% 97.88% 84.21% 51.65% 74.01% 74.96% 95.31%
Bit Acc. (Contrast 25%) 99.67% 99.76% 95.91% 56.25% 89.35% 95.19% 100.00%
Bit Acc. (Contrast 50%) 99.74% 99.82% 97.56% 75.95% 96.21% 99.19% 100.00%
Bit Acc. (Contrast 75%) 99.74% 99.85% 97.92% 86.89% 98.08% 99.54% 100.00%
Bit Acc. (Contrast 125%) 99.58% 99.59% 95.15% 94.70% 96.63% 98.65% 100.00%
Bit Acc. (Contrast 150%) 99.11% 98.96% 92.35% 96.09% 93.69% 95.23% 100.00%
Bit Acc. (Contrast 175%) 98.52% 97.49% 89.58% 96.09% 91.50% 92.15% 99.87%
Bit Acc. (Contrast 200%) 97.86% 95.88% 86.98% 96.61% 89.21% 90.08% 99.74%
Bit Acc. (Hue -0.2) 98.37% 99.40% 83.25% 60.68% 95.39% 97.31% 95.18%
Bit Acc. (Hue -0.1) 99.66% 99.56% 94.63% 73.09% 97.28% 98.92% 99.87%
Bit Acc. (Hue 0.1) 99.68% 99.06% 95.47% 80.82% 97.12% 98.54% 100.00%
Bit Acc. (Hue 0.2) 99.28% 99.04% 81.57% 59.46% 95.70% 97.50% 98.70%
Bit Acc. (JPEG 40) 97.18% 99.41% 94.91% 91.32% 98.35% 99.50% 100.00%
Bit Acc. (JPEG 50) 98.35% 99.64% 96.47% 91.58% 98.84% 99.62% 100.00%
Bit Acc. (JPEG 60) 98.62% 99.76% 96.15% 91.49% 98.54% 99.62% 100.00%
Bit Acc. (JPEG 70) 98.90% 99.74% 96.39% 91.49% 98.42% 99.62% 100.00%
Bit Acc. (JPEG 80) 99.31% 99.84% 97.40% 91.84% 98.60% 99.69% 100.00%
Bit Acc. (JPEG 90) 99.60% 99.85% 97.32% 92.10% 98.66% 99.69% 100.00%
Bit Acc. (Gaussian Blur 3) 99.74% 99.90% 97.92% 91.67% 98.66% 99.81% 100.00%
Bit Acc. (Gaussian Blur 5) 99.74% 99.90% 97.96% 90.97% 98.47% 99.81% 100.00%
Bit Acc. (Gaussian Blur 9) 99.74% 99.89% 97.88% 89.24% 98.15% 99.81% 100.00%
Bit Acc. (Gaussian Blur 13) 99.73% 99.85% 97.96% 87.15% 97.67% 99.77% 100.00%
Bit Acc. (Gaussian Blur 17) 99.70% 99.79% 98.00% 84.98% 96.83% 99.73% 100.00%


40


**2160**

**2161**


**2162**

**2163**

**2164**

**2165**

**2166**

**2167**


**2168**

**2169**

**2170**

**2171**

**2172**

**2173**


**2174**

**2175**

**2176**

**2177**

**2178**


**2179**

**2180**

**2181**

**2182**

**2183**

**2184**


**2185**

**2186**

**2187**

**2188**

**2189**

**2190**


**2191**

**2192**

**2193**

**2194**

**2195**

**2196**


**2197**

**2198**

**2199**

**2200**

**2201**


**2202**

**2203**

**2204**

**2205**

**2206**

**2207**


**2208**

**2209**

**2210**

**2211**

**2212**

**2213**


Table 5: **Extended results of Chunky Seal on COCO (Lin et al., 2014).**


Chunky Seal (ours) Video Seal Video Seal
HiDDeN MBRS TrustMark WAM
1024bit, 256px 256bit, 256px 96bit, 256px


Capacity 1024 bits 256 bits 96 bits 48 bits 256 bits 100 bits 32 bits
PSNR 44.29 dB 44.94 dB 53.33 dB 30.51 dB 45.81 dB 42.72 dB 38.73 dB
SSIM 0.9917 0.9953 0.9992 0.8469 0.9944 0.9921 0.9803
MS-SSIM 0.9968 0.9975 0.9988 0.9203 0.9976 0.9931 0.9891
LPIPS 0.0061 0.0022 0.0033 0.1850 0.0035 0.0015 0.0295
Embedding Time 0.03 s 0.01 s 0.01 s 0.01 s 0.01 s 0.01 s 0.03 s
Extraction Time 0.04 s 0.01 s 0.01 s 0.00 s 0.00 s 0.00 s 0.01 s
Bit Acc. 99.66% 99.92% 97.64% 92.40% 98.70% 99.90% 100.00%
Bit Acc. (Horizontal Flip) 99.52% 99.87% 97.16% 61.83% 49.87% 99.87% 99.97%
Bit Acc. (Rotate 5°) 97.11% 99.06% 94.69% 79.85% 50.04% 65.31% 97.84%
Bit Acc. (Rotate 10°) 94.46% 97.56% 91.53% 72.65% 49.92% 51.89% 77.16%
Bit Acc. (Rotate 30°) 50.57% 50.83% 57.08% 53.37% 49.64% 50.05% 51.00%
Bit Acc. (Rotate 45°) 49.98% 50.53% 50.55% 49.54% 50.15% 50.93% 50.75%
Bit Acc. (Rotate 90°) 56.36% 83.44% 50.58% 51.23% 49.90% 49.84% 49.28%
Bit Acc. (Resize 32%) 94.69% 97.69% 97.53% 71.19% 90.57% 99.81% 99.81%
Bit Acc. (Resize 45%) 98.60% 99.56% 97.70% 80.50% 96.16% 99.86% 100.00%
Bit Acc. (Resize 55%) 99.31% 99.79% 97.64% 84.88% 97.28% 99.87% 100.00%
Bit Acc. (Resize 63%) 99.53% 99.86% 97.66% 87.17% 97.80% 99.90% 100.00%
Bit Acc. (Resize 71%) 99.63% 99.89% 97.67% 87.92% 98.11% 99.88% 100.00%
Bit Acc. (Resize 77%) 99.66% 99.91% 97.69% 88.71% 98.21% 99.90% 100.00%
Bit Acc. (Resize 84%) 99.66% 99.91% 97.68% 89.23% 98.34% 99.90% 100.00%
Bit Acc. (Resize 89%) 99.67% 99.93% 97.67% 89.38% 98.34% 99.91% 100.00%
Bit Acc. (Resize 95%) 99.66% 99.92% 97.70% 89.75% 98.43% 99.89% 100.00%
Bit Acc. (Crop 32%) 49.84% 50.25% 50.75% 49.27% 49.54% 49.95% 73.88%
Bit Acc. (Crop 45%) 60.64% 49.73% 50.36% 50.40% 50.04% 50.79% 91.47%
Bit Acc. (Crop 55%) 82.26% 50.59% 59.70% 56.02% 49.99% 49.79% 95.72%
Bit Acc. (Crop 63%) 89.82% 58.23% 74.25% 61.19% 50.64% 51.19% 97.19%
Bit Acc. (Crop 71%) 93.68% 81.43% 85.59% 67.94% 49.80% 54.77% 97.22%
Bit Acc. (Crop 77%) 94.71% 92.29% 89.15% 73.06% 49.70% 87.79% 98.19%
Bit Acc. (Crop 84%) 96.04% 97.64% 92.94% 78.83% 49.66% 99.95% 99.12%
Bit Acc. (Crop 89%) 97.17% 98.96% 94.32% 81.27% 49.84% 99.98% 98.69%
Bit Acc. (Crop 95%) 97.88% 99.48% 95.21% 83.31% 50.82% 99.96% 99.22%
Bit Acc. (Brightness 10%) 85.06% 81.38% 81.43% 52.33% 62.51% 83.43% 96.12%
Bit Acc. (Brightness 25%) 98.76% 98.82% 94.93% 55.85% 84.75% 97.85% 99.75%
Bit Acc. (Brightness 50%) 99.53% 99.84% 97.06% 74.29% 95.43% 99.74% 100.00%
Bit Acc. (Brightness 75%) 99.65% 99.92% 97.50% 86.44% 97.98% 99.87% 100.00%
Bit Acc. (Brightness 125%) 99.16% 99.02% 96.24% 95.04% 95.24% 98.56% 100.00%
Bit Acc. (Brightness 150%) 98.47% 97.78% 94.73% 95.56% 92.20% 96.45% 100.00%
Bit Acc. (Brightness 175%) 97.34% 96.13% 92.64% 95.63% 89.16% 94.62% 100.00%
Bit Acc. (Brightness 200%) 95.83% 94.18% 89.48% 94.92% 86.91% 91.97% 99.97%
Bit Acc. (Contrast 10%) 97.77% 98.29% 83.32% 52.31% 77.41% 78.54% 93.69%
Bit Acc. (Contrast 25%) 99.43% 99.68% 95.15% 55.79% 90.76% 96.47% 99.78%
Bit Acc. (Contrast 50%) 99.62% 99.87% 97.11% 73.81% 96.71% 99.68% 100.00%
Bit Acc. (Contrast 75%) 99.66% 99.92% 97.50% 86.27% 98.19% 99.87% 100.00%
Bit Acc. (Contrast 125%) 99.23% 99.19% 95.83% 94.67% 95.67% 98.57% 100.00%
Bit Acc. (Contrast 150%) 98.55% 97.84% 93.30% 96.00% 92.83% 95.83% 99.97%
Bit Acc. (Contrast 175%) 97.75% 96.42% 91.00% 96.58% 90.38% 93.65% 99.97%
Bit Acc. (Contrast 200%) 96.89% 95.07% 89.08% 96.85% 88.58% 90.87% 99.88%
Bit Acc. (Hue -0.2) 97.37% 98.35% 82.19% 59.62% 95.41% 99.03% 96.91%
Bit Acc. (Hue -0.1) 99.41% 98.77% 94.92% 72.27% 97.34% 99.73% 100.00%
Bit Acc. (Hue 0.1) 99.37% 99.05% 95.53% 82.04% 97.63% 99.76% 99.97%
Bit Acc. (Hue 0.2) 97.70% 98.64% 78.98% 61.85% 96.64% 99.54% 98.06%
Bit Acc. (JPEG 40) 65.86% 97.79% 72.64% 87.98% 95.44% 98.34% 97.28%
Bit Acc. (JPEG 50) 72.47% 98.74% 80.22% 88.21% 96.22% 99.05% 98.31%
Bit Acc. (JPEG 60) 76.97% 99.26% 85.39% 88.44% 97.28% 99.38% 98.84%
Bit Acc. (JPEG 70) 82.93% 99.55% 89.42% 88.77% 97.50% 99.64% 99.53%
Bit Acc. (JPEG 80) 88.89% 99.79% 93.34% 89.31% 97.99% 99.73% 99.56%
Bit Acc. (JPEG 90) 93.21% 99.87% 96.06% 90.50% 98.25% 99.81% 99.84%
Bit Acc. (Gaussian Blur 3) 99.63% 99.89% 97.70% 86.23% 97.75% 99.87% 100.00%
Bit Acc. (Gaussian Blur 5) 99.29% 99.86% 97.75% 80.40% 96.09% 99.86% 100.00%
Bit Acc. (Gaussian Blur 9) 97.95% 99.74% 97.55% 71.00% 90.90% 99.86% 99.88%
Bit Acc. (Gaussian Blur 13) 93.88% 99.57% 97.12% 65.04% 84.27% 99.85% 99.62%
Bit Acc. (Gaussian Blur 17) 84.65% 99.11% 96.64% 60.94% 77.49% 99.41% 99.38%


41


**2214**

**2215**


**2216**

**2217**

**2218**

**2219**

**2220**

**2221**


**2222**

**2223**

**2224**

**2225**

**2226**

**2227**


**2228**

**2229**

**2230**

**2231**

**2232**


**2233**

**2234**

**2235**

**2236**

**2237**

**2238**


**2239**

**2240**

**2241**

**2242**

**2243**

**2244**


**2245**

**2246**

**2247**

**2248**

**2249**

**2250**


**2251**

**2252**

**2253**

**2254**

**2255**


**2256**

**2257**

**2258**

**2259**

**2260**

**2261**


**2262**

**2263**

**2264**

**2265**

**2266**

**2267**


J.2 IMAGE EXAMPLES


**TrustMark** **WAM**
Original (Bui et al., 2023a) (Sander et al., 2025) **Video Seal** **Chunky Seal (ours)**


Figure 13: Qualitative results for the different watermarking methods on images taken from the
SA-1b dataset at their original resolutions. We show the original images, the watermarked ones, and
the watermark distortions brightened for clarity.


42


**2268**

**2269**


**2270**

**2271**

**2272**

**2273**

**2274**

**2275**


**2276**

**2277**

**2278**

**2279**

**2280**

**2281**


**2282**

**2283**

**2284**

**2285**

**2286**


**2287**

**2288**

**2289**

**2290**

**2291**

**2292**


**2293**

**2294**

**2295**

**2296**

**2297**

**2298**


**2299**

**2300**

**2301**

**2302**

**2303**

**2304**


**2305**

**2306**

**2307**

**2308**

**2309**


**2310**

**2311**

**2312**

**2313**

**2314**

**2315**


**2316**

**2317**

**2318**

**2319**

**2320**

**2321**


**TrustMark** **WAM**
Original (Bui et al., 2023a) (Sander et al., 2025) **Video Seal** **Chunky Seal (ours)**


Figure 14: Qualitative results for the different watermarking methods on images taken from the
SA-1b dataset at their original resolutions. We show the original images, the watermarked ones, and
the watermark distortions brightened for clarity.


43
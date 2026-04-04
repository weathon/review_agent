# DISK: DIFFERENTIABLE SPARSE KERNEL COMPLEX FOR EFFICIENT SPATIALLY-VARIANT CONVOLUTION

**Zhizhen Wu** _[∗]_ **Zhe Cao** _[∗]_ **Yuchi Huo** _[†]_


State Key Lab of CAD&CG, Zhejiang University, China

zhizhenwu@zju.edu.cn, caozhe022@qq.com, huo.yuchi.sc@gmail.com


Figure 1: **An overview of our method.** We represent a dense filter as a _Sparse Kernel Complex_, a
sequence of sparse layers whose parameters Θ are learned via differentiable optimization. We apply
our filter _F_ Θ to an impulse _δ_ to yield a synthesized kernel _Ksyn_, and minimize a loss against the
_L_
target _Ktgt_ to learn arbitrary shapes. These optimized kernels form a basis for high-performance
spatially varying filtering, achieving quality close to ground truth with up to a 20 _×_ speedup.


ABSTRACT


Image convolution with complex kernels is common in photography, scientific
imaging, and animation, but dense convolution is too expensive for resourcelimited devices. Existing approximations, such as simulated annealing and lowrank decompositions, are either slow or struggle with non-convex kernels. We
present a differentiable kernel decomposition framework that represents a spatially variant dense kernel with a small set of sparse samples, assuming the target
dense kernel is known for both optimization and filtering. Our method provides
(i) end-to-end differentiable sparse-kernel optimization, (ii) shape-aware initialization for non-convex kernels, and (iii) kernel-space interpolation for efficient,
multi-dimensional spatially varying filtering without retraining or added runtime
cost. Across Gaussian and non-convex kernels, our method achieves higher fidelity than simulated annealing and lower cost than low-rank decomposition. It is
practical for mobile imaging and real-time rendering, and integrates cleanly into
learning pipelines.


1 INTRODUCTION


From rendering realistic depth-of-field effects (Sakurikar; Wu et al., 2022) in computational photography to modeling the intricate point spread functions (Liu et al., 2022; Shajkofci & Liebling, 2020)
of optical systems, the ability to apply large, complex convolution kernels is a fundamental building
block in modern vision and graphics computing systems. This creates a fundamental tension: while
larger, more intricate kernels enable higher-fidelity results, their quadratic computational cost renders direct implementation impractical for interactive applications on devices ranging from mobile
phones to high-end GPUs.


_∗_ Equal contribution.

_†_ Corresponding author.


1


To bridge this gap, many works have focused on approximation strategies. For specific cases like
Gaussian blur, elegant constant-time solutions (Zing, 2010; Kovesi, 2010) exploit the filter’s analytic
structure. However, such specialized methods are not applicable to the arbitrary, often non-convex
kernels needed for advanced effects. More general approaches, such as low-rank matrix decomposition (McGraw, 2015), support arbitrary kernels but typically reduce the computation to a sequence
of smaller dense convolutions, which limits sparsity and caps efficiency gains.


A more direct and efficient approach (Schuster et al., 2020) is to approximate a dense kernel with
a truly sparse one, drastically reducing the number of required computations. It relies on heuristic
search via parallel simulated annealing to identify sparse sampling patterns for arbitrary kernels.
Despite its generality, it requires many iterations and still misses high-fidelity solutions due to the
non-convex optimization landscape. This motivates a more principled and efficient way to obtain
high-quality sparse kernel representations.


In this work, we introduce a differentiable kernel decomposition framework to address this challenge. Like prior approximation methods, our formulation assumes access to the dense target kernel
throughout optimization and filtering. Unlike traditional low-rank methods, we optimize a sequence
of natively sparse kernels, yielding an efficient representation for runtime inference. We cast the
decomposition as an end-to-end optimization problem and apply gradient-based methods instead of
heuristic search such as simulated annealing, enabling more reliable convergence to high-fidelity
solutions in far fewer iterations. To improve stability for non-convex target kernels, we further propose a two-part initialization strategy: structure-aware sampling to capture fine shape details and a
deterministic radial initialization to stabilize and accelerate convergence.


Beyond single-kernel approximation, our framework supports efficient, multidimensional spatially
varying (SV) filtering. The main challenge is the cost of generating a unique kernel per pixel, which
often creates a significant performance bottleneck. We address this with filter-space interpolation:
we precompute an optimized basis of sparse filters spanning the target effect range, then synthesize
per-pixel kernels at runtime by interpolating this compact basis. This strategy reduces per-pixel
synthesis to a minimal set of multiply-add operations, decoupling kernel-generation cost from image
resolution and enabling sophisticated SV effects with negligible overhead.


Our contributions are as follows:


    - A differentiable framework for decomposing a dense, arbitrary kernel into a sequence of
optimized sparse layers, enabling efficient, high-fidelity approximation.


    - An initialization scheme, combining a general radial strategy for stable convergence with a
sparse sampling method for capturing non-convex kernels.


    - A filter-space interpolation method for high-performance, spatially-varying filtering that
decouples kernel synthesis cost from image resolution.


2 RELATED WORK


2.1 HIGH-PERFORMANCE KERNEL


Given that Gaussian blur is computationally expensive, numerous methods have been proposed to
accelerate it. Common examples include _O_ (1) approximations such as the Extended Binomial Filter (Zing, 2010) and Summed-Area Table-based methods (Kovesi, 2010). However, their reliance on
precomputation or inherently sequential steps makes them ill-suited to the massively parallel design
of modern GPUs.


A better match for real-time rendering is Kawase blur (Kawase, 2003), a multi-pass filter that uses
only four texture samples per pass. As an extension, Dual Filtering (Martin et al., 2015) introduces downsampling followed by upsampling, reducing bandwidth and computation by operating
on lower-resolution textures. However, such pipelines trade regular downsampling for extra reconstruction work (often additional convolution) and may lose fine detail, whereas we stay at a single
resolution and reduce cost via flexible sparse sampling while matching the target kernel response. A
further practical limitation is the lack of a systematic mapping from a target Gaussian strength (e.g.,
a given _σ_ ) to the corresponding Kawase or Dual Filtering parameters; our work addresses this gap.


2


2.2 SPATIALLY-VARIANT FILTERING


A substantial body of work learns spatially varying, per-pixel convolution kernels, with applications to video prediction, frame interpolation, denoising, and deblurring (Jia et al., 2016; Niklaus
et al., 2017; Mildenhall et al., 2018; Zhou et al., 2019; 2021). Unlike approaches that directly
predict a dense per-pixel kernel map, our method decouples filter generation from spatial resolution. Specifically, we learn a compact lookup table (LUT) that parameterizes a continuous family
of filters, enabling flexible and efficient spatially-variant filtering. Spatiotemporal Variance-Guided
Filtering (Schied et al., 2017) forms per-pixel filter mixtures guided by estimated spatial and temporal variance. Differently, we condition filter generation on an input per-pixel blur-intensity map.
This allows us to synthesize filters that match any target spatially varying blur, without intermediate
variance estimation or other content statistics.


2.3 KERNEL APPROXIMATION AND DECOMPOSITION


Inspired by Kawase blur (Kawase, 2003), High-Performance Image Filters (Schuster et al., 2020)
uses parallel tempering to optimize sparse sampling patterns, but its optimization stability can be
limited by sensitivity to many hyperparameters compared with our gradient-based formulation. Kernel decomposition is also used to reduce cost in other settings: 2D kernels into pairs of 1D kernels
for video interpolation (Niklaus et al., 2017); 3D spatiotemporal kernels into spatial and temporal
atoms (STDCF) (Schied et al., 2017); standard convolution into depthwise and 1 _×_ 1 pointwise operations (Howard et al., 2017; Chollet, 2017; Ramadhani et al., 2024); dynamic weights into static
bases plus residuals (Li et al., 2021; 2024); and related decompositions for graph-transformer attention (KDLGT) (Wu et al., 2023) and large depthwise kernels (LKD-Net) (Luo et al., 2023).


Despite their success, these methods are largely limited to structured factorizations or heuristic optimization. Our framework addresses the missing capability: a differentiable, general solution for
sparse approximation of arbitrary (including non-convex) kernels, together with an efficient mechanism for spatially varying kernel synthesis.


3 PRELIMINARY


3.1 KERNEL-BASED FILTERING


Kernel-based filtering is fundamental to many image processing tasks. This process takes an input
image _Iin_ and computes each pixel’s value for the output image _Iout_ as a weighted average of its
local neighbors within _Iin_ . Formally, this operation is expressed as a 2D convolution, defined as:


_k_

- _Iin_ [ _x_ + _i, y_ + _j_ ] _K_ [ _i, j_ ] _,_ (1)

_·_
_j_ = _−k_


_Iout_ [ _x, y_ ] = ( _Iin_ _K_ )[ _x, y_ ] =
_∗_


_k_


_i_ = _−k_


where the matrix _K_ is the _M_ _× M_ convolution with kernel size _M_ _∈_ R [+], whose elements _K_ [ _i, j_ ]
are weights that determine the contribution of each neighboring pixel to the final filtered value.


3.2 FILTER REPRESENTATION


The dense matrix representation for the kernel _K_ in Eq. (1) is straightforward. However, its _O_ ( _M_ [2] )
computational cost presents a significant bottleneck. This is especially true for filters with a large
spatial support, such as a Gaussian blur with a large _σ_, where the cost becomes prohibitively expensive for real-time applications that demand high frame rates.


Our key insight is to approximate this expensive operation by structuring the filter as a sequence
of lightweight convolutional layers, where the output of one layer serves as the input for the subsequent one. Each layer applies a highly efficient sparse kernel, _Ksparse_, which we define by a small
collection of _N_ samples with offset-weight pairs:

_Ksparse_ = _{_ ( **o** _i, wi_ ) _}i_ _[N]_ =1 _[,]_ (2)

where **o** _i_ R [2] is the spatial offset and _wi_ is its corresponding weight.
_∈_


3


The complete operation, consisting of _L_ such layers with kernels ( _K_ 1 _, K_ 2 _, ..., KL_ ), can be expressed
as a nested convolution:
_Iout_ = ( _..._ (( _Iin_ _K_ 1) _K_ 2) _..._ _KL_ ) _._ (3)
_∗_ _∗_ _∗_ _∗_

This multi-layer filter reduces the cost to _O_ ( [�] _[L]_ _l_ =1 _[N][l]_ [)][ per pixel.] [Since this sum is far smaller than]
the number of weights in the target dense kernel ( [�] _Nl_ _M_ [2] ), the approach offers a dramatic
_≪_
speedup.


4 METHODOLOGY


4.1 DIFFERENTIABLE MULTI-LAYER KERNEL COMPLEX


**Overview** Sparse filters offer a computationally efficient alternative to dense kernels; however,
they often fail to capture the intricate structure of large, complex filters. The core challenge lies
in determining the optimal parameters—the spatial offsets and weights—for a sequence of sparse
kernels to accurately reconstruct a target. Manually designing these parameters or using traditional,
non-differentiable methods is a formidable task.


To overcome this, our key contribution is to frame the decomposition as a differentiable optimization
problem. This enables the simultaneous end-to-end learning of all sparse kernel parameters across
all layers. We define the complete set of these learnable parameters as Θ = _{_ ( **o** _l,i, wl,i_ ) _}l_ _[L,N]_ =1 _,i_ _[l]_ =1 [,]
which includes the offsets and weights for _Nl_ samples in each of the _L_ layers.

Our goal is to find the optimal parameters Θ _[∗]_ by minimizing a loss function _L_ that measures the
discrepancy between our approximation and the target kernel:


Θ _[∗]_ = arg min
Θ (4)

_[L]_ [(] _[K][target][, F][approx]_ [(Θ))] _[,]_

_Fapprox_ (Θ) = _Ks,_ 1 _Ks,_ 2 _..._ _Ks,L,_
_∗_ _∗_ _∗_


where _Ktarget_ is the desired dense filter and _Fapprox_ (Θ) is the composite kernel formed by the
convolution of the learned sparse kernels.


**Learnable Parameter** Our optimization strategy treats the offsets and weights of each sample as
independent, learnable parameters. Specifically, for each layer _l_ and for each of the _Nl_ sampling
points within it, we simultaneously optimize both the 2D offset vector **o** _l,i_ and its corresponding
scalar weight _wl,i_ .


The complete set of learnable parameters for the entire model, denoted by Θ, is therefore the collection of all such offset-weight pairs:


Θ =


_L_

- _{_ ( **o** _l,j, wl,j_ ) _}j_ _[N]_ =1 _[l]_ _[.]_ (5)

_l_ =1


**Initialization** A well-designed parameter initialization is crucial for stable optimization convergence. Heuristic methods, such as Kawase (Kawase, 2003) and Dual Filtering (Martin et al., 2015),
have fixed schemes tailored to specific filter types; however, a general approach is required for arbitrary target kernels of different sizes.


To address this, we propose a radial initialization strategy. The core idea is to initialize the sampling
points in each layer to be uniformly distributed on the circumference of a circle, with the radius
of this circle increasing linearly with the layer index. This progressive expansion ensures that the
effective receptive field of the composite kernel grows with each subsequent layer, making the initial
configuration capable of spanning a large-area target kernel from the outset. The radius for layer _l_,
denoted _rl_, is governed by a step size ∆ _r_ derived from the target kernel’s spatial extent and the total
number of layers _L_ (see Appendix for derivation). The corresponding weights in each layer are
initialized uniformly.


This initialization is formally defined as:


4


_rl_ = _l_ ∆ _r_ for _l_ = 1 _, . . ., L,_
_·_


  -   - 2 _πi_

**o** _l,i_ = _rl_ cos
_Nl_


- - 2 _πi_
_, rl_ sin
_Nl_


��
for _i_ = 1 _, . . ., Nl,_


(6)


_wl,i_ = [1] _._

_Nl_


4.2 SPARSE SAMPLING OF ARBITRARY KERNEL


A common way to initialize filter offsets is by sampling random positions within a local neighborhood. While this approach is general, it often traps the optimization in poor local optima, especially
for kernels with complex or non-convex shapes.


Our method decomposes a dense kernel into a series of sparse ones. The first of these, _Ks,_ 1 (Eq. 4),
has the greatest influence on the final filtered output, so its initialization is critical. A simple improvement over purely random sampling is to confine samples to the minimal bounding box of the
kernel’s non-zero pixels. This ensures most samples fall near the target shape, but it is still inefficient
for non-convex kernels, whose bounding boxes can contain large empty regions.


To overcome this limitation, we propose a more sophisticated initialization strategy leveraging rejection sampling. Instead of drawing samples from the kernel’s bounding box, our method samples
directly from the support of the kernel, i.e., its non-zero locations. We first quantify the effective
sampling area, denoted by _S_, as the count of these non-zero pixels. A sampling radius _r_ is subsequently derived based on the desired number of samples, _Ns_ :


_r_ =


~~�~~ _S_
(7)
_Ns_ _π_ _[.]_
_·_


The detailed procedure is provided in the appendix. This initialization sets the first sparse kernel’s
offsets to closely match the target shape. By restricting samples to relevant regions, it avoids vanishing gradients and reduces the risk of converging to poor local optima.


4.3 SPATIALLY VARYING FILTERING


Next, we propose a decomposition method for spatially varying filtering.


Spatially varying filtering generalizes convolution by applying a unique filter at each pixel ( _x, y_ ).
The filter’s properties—such as its blur radius, orientation, or shape—are determined by a corresponding value _P_ ( _x, y_ ) from a parameter map. The core challenge lies in efficiently synthesizing
and applying these unique per-pixel kernels.


Conventional spatially varying filtering is often impractical. Generating dense kernels on the
fly (Wang et al., 2023) is too slow, while precomputing them (Kovesi, 2010) incurs prohibitive
memory costs; both are a poorly suited to modern parallel hardware. Faster alternatives (Leimk¨uhler
et al., 2018) restrict filters to simple analytic forms (e.g., Gaussians), but this sacrifices the expressiveness needed for complex, non-convex point spread functions (PSFs).


We observe that, in prior work, the cost of generating or storing spatially varying kernels grows
linearly with image resolution. To remove this bottleneck while retaining expressive, sparsely optimized kernels, we introduce _Filter-Space Interpolation_, which decouples kernel-generation complexity from image size.


Our spatially varying filtering is built on an ordered set of _M_ basis sparse filters, which discretely
sample a continuous, one-dimensional space of filters. Each basis filter, _fk_, corresponds to a
_F_
scalar parameter _pk_ (with _p_ 1 _< p_ 2 _<_ _< pM_ ) and consists of a unique set of _N_ sampling offsets
_· · ·_
and weights. This design allows our basis to represent a wide range of filter behaviors across the
parameter space, from applying arbitrary linear transformations to a kernel to simply varying the
standard deviation ( _σ_ ) of a Gaussian. We define the basis as:
= _fk_ ( _pk_ ) _k_ = 1 _, . . ., M_ _,_ where _fk_ = ( **o** _ki, wki_ ) _i_ =1 (8)
_F_ _{_ _|_ _}_ _{_ _}_ _[N]_
We divide the approach into an offline pre-computation stage and a runtime inference stage. In the
offline stage, we optimize each basis filter _fk_ individually to represent the ideal filter effect at its
parameter value _pk_ .


5


At runtime, we synthesize a unique sparse filter for each pixel ( _x, y_ ), which is guided by a perpixel parameter map, _P_ . From the parameter value at each coordinate, _P_ ( _x, y_ ), we determine a
corresponding vector of _M_ interpolation weights, _**α**_ ( _x, y_ ) = ( _α_ 1 _, . . ., αM_ ). These weights specify
how to blend a compact set of basis filters, _{fk}k_ _[M]_ =1 [, to reconstruct the final filter instance.]

The final sparse filter for a given pixel, _f_ ( _x, y_ ), is synthesized as a direct convex combination of the
basis filters:


_f_ ( _x, y_ ) =


_M_

- _αk_ ( _x, y_ ) _fk,_ (9)

_·_
_k_ =1


subject to the constraint that [�] _k_ _[M]_ =1 _[α][k]_ [(] _[x, y]_ [) = 1][ and] _[ α][k]_ [(] _[x, y]_ [)] _[ ≥]_ [0][.]

By directly interpolating basis-filter offsets and weights, we sidestep the costly on-the-fly generation of kernels from analytical functions. This reduces the computational overhead of spatially
varying kernel synthesis to a minimal set of parallelizable multiply-add operations. Furthermore,
the interpolatable nature of our basis filters makes the entire set highly compressible, allowing us
to significantly reduce the memory footprint required to achieve a wide range of expressive effects
while offering flexible control over the quality-performance trade-off.


4.4 IMPLEMENTATION DETAILS


**Training Process** To ensure our learned filter parameters are generalized and not overfit to a specific dataset, we adopt an image-agnostic optimization strategy. We leverage a core principle of
Linear Shift-Invariant (LSI) systems (Goodman, 2005): a filter is fully characterized by its impulse
response.


First, we synthesize the effective kernel of our multi-pass filter, _Fθ_, by applying it to a discrete Dirac
delta function, _δ_ . The resulting output is the synthesized impulse response, _K_ syn. The impulse _δ_ is
an image with a single non-zero pixel at its center coordinate **c** :


�1 if **n** = **c**
_K_ syn = _Fθ_ ( _δ_ ) _,_ where _δ_ [ **n** ] = (10)
0 otherwise.


Here, _θ_ represents the learnable parameters of our filter and **n** denotes the discrete pixel coordinates.


**Loss Design** Second, we define our loss function, _L_, as the Charbonnier L1 loss _C_ (Charbonnier
et al., 1994) between the synthesized kernel _K_ syn and a target kernel _K_ tgt:


= ( _K_ syn _, K_ tgt) _._ (11)
_L_ _C_

This impulse-response-based supervision allows us to ”collapse” the entire multi-layer filtering sequence into a single, equivalent kernel for direct and precise approximation of the target.


5 EXPERIMENTS


In this section, we conduct a series of experiments to evaluate our differentiable kernel decomposition framework thoroughly. We first describe the experiment details and evaluation protocol
in Section 5.1. Next, in Section 5.2, we assess our method’s ability to approximate single, complex
kernels, comparing it against state-of-the-art techniques. We extend this analysis to the more challenging task of spatially varying filtering in Section 5.3. To validate our specific design choices, we
present a series of ablation studies in Section 5.4.


5.1 SETUP


**Baselines.** We compare our method against several baselines. For both single kernel and spatially varying filtering, we include a low-rank decomposition (LowRank) (McGraw, 2015) and the
optimization-based method of Parallel Tempering (PST) (Schuster et al., 2020).


6


LPIPS


0.002


0.007


0.011


0.001


0.053


LPIPS


0.008


0.011


0.040


_σ_ =5


_σ_ =9


Ours 8 _×_ 6 PST 8 _×_ 6 Ours 12 _×_ 4 PST 12 _×_ 4 _σ_ =7 Ours 8 _×_ 6 PST 8 _×_ 6 Ours 12 _×_ 4 PST 12 _×_ 4

LPIPS 0.001 0.019 0.011 0.077 LPIPS 0.001 0.010 0.011


Ours 8 _×_ 6


Ours 8 _×_ 6


LPIPS


Ours 8 _×_ 6


0.001


0.001


PST 8 _×_ 6


0.019


Ours 12 _×_ 4


0.011


PST 12 _×_ 4


0.077


_σ_ =7


PST 8 _×_ 6


0.010


Ours 12 _×_ 4


0.011


0.055


PST 8 _×_ 6


Ours 12 _×_ 4


PST 12 _×_ 4


_σ_ =11


Ours 8 _×_ 6


PST 8 _×_ 6


Ours 12 _×_ 4


PST 12 _×_ 4


Figure 2: **Comparison** **of** **Gaussian** **kernel** **approximation** **with** **varying** _σ_ **.** We compare our
method against PST using two sparse configurations (8 layers × 6 samples and 12 layers × 4 samples). LPIPS scores appear in the top-right corner (lower is better). The top-left inset visualizes the
error map, with positive errors shown in red and negative errors in green.


**Datasets** **and** **Kernels.** To evaluate the versatility of our method, we use a diverse set of target
kernels and images. This set includes standard analytical shapes, such as Gaussian kernels (with
_σ_ values from 5 to 11). To assess performance on more complex targets, we additionally use a
suite of arbitrary kernels comprising simple geometric primitives (disks, rings), regular polygons
(4-sided and 6-sided), non-convex shapes (a heart, a four-pointed star, and an ampersand symbol),
more complex shapes (animal silhouettes), and optical PSFs (coma and spherical aberration). For the
spatially varying filtering experiments, we use five high-resolution photographs selected to represent
realistic scenarios with complex textures and both 1D and 2D spatial variations.


**Implementation** **and** **Evaluation** **Metric.** We implement our methods in PyTorch and perform
all optimization on a single GPU with 24 GB of memory, offering computational power comparable
to an NVIDIA RTX 4090. For all configurations of kernels and layers, we use the same Adam
optimizer with a learning rate linearly decayed from 1 _×_ 10 _[−]_ [3] to 1 _×_ 10 _[−]_ [4] . We use 1000 optimization
steps per kernel for our method. For comparison, we run the PST algorithm for 10,000 iterations
with 10 parallel candidates, for a total of 100,000 optimization steps. For the LowRank method, we
utilize decompositions with ranks 1,2 and 3, chosen to maintain a comparable number of samplings.


For runtime analysis, we benchmark our approach
on a representative mobile device equipped with a
Qualcomm Snapdragon 8 Gen 3 SoC, and report latency in milliseconds (ms). We evaluate both numerical fidelity and perceptual similarity using Peak
Signal-to-Noise Ratio (PSNR), Learned Perceptual
Image Patch Similarity (LPIPS) (Zhang et al., 2018),
and FLIP-LDR (Andersson et al., 2020). Higher values indicate better performance for PSNR, and lower
values are better for LPIPS and FLIP-LDR.


5.2 SINGLE KERNEL


Fig. 3 shows that our method consistently achieves a
superior balance between reconstruction quality and
inference speed compared to all other approaches.
For our method and PST, the ’S’, ’M’, and ’L’
correspond to total sample counts of 48 (12 _×_ 4),
96 (24 _×_ 4), and 128 (32 _×_ 4), respectively. The
LowRank’s ’M’ and ’L’ use 98 (49 _×_ 2) and 196
(49 _×_ 4) parameters.


9


8


7


6


5


4


3


2


6.0 6.5 7.0 7.5 8.0 8.5 9.0
Latency(ms)


Figure 3: **Speed,** **accuracy,** **and** **samples** **com-**
**parison.** The figure plots quality against latency
(lower is better for both). The size of each bubble
represents the total sample count.


Next, we present a comparison of Gaussian kernel approximation with varying standard deviations
_σ_ in Fig. 2. In a 6-layer, 8-sample (8 _×_ 6) configuration, our method achieves high-fidelity results
with low perceptual error, whereas PST exhibits visible noise and artifacts. This performance gap
widens in a sparser 12 _×_ 4 setup. As _σ_ increases, PST’s approximation degrades severely, while our
result remains visually coherent and maintains a substantially lower LPIPS error. These results show


7


0.3194


LowR. 49 _×_ 4 9.11

38.50


0.2372


Ours 24 _×_ 4 7.47

49.02


LPIPS ( _×_ 10 _[−]_ [4] )


Disk 246.83

PSNR (dB)


GT

LPIPS ( _×_ 10 _[−]_ [4] )


Star 192.48

PSNR (dB)


GT

LPIPS ( _×_ 10 _[−]_ [4] )


Coma 243.58

PSNR (dB)


GT


0.2446


Ours 32 _×_ 4 8.51

47.95


Ours 32 _×_ 4 8.36

49.09


Ours 32 _×_ 4


Ours 32 _×_ 4


0.7612


3.036


1.183


LowR. 49 _×_ 4


Ours 24 _×_ 4


Ours 32 _×_ 4 8.51

58.09


LowR. 49 _×_ 4 9.11

42.91


Ours 24 _×_ 4 7.53

56.53


Ours 32 _×_ 4


1.443


25.96


3.667


LowR. 49 _×_ 4


Ours 24 _×_ 4


LowR. 49 _×_ 6 12.91

39.61


LowR. 49 _×_ 6


Ours 24 _×_ 4 7.45

47.10


Ours 24 _×_ 4


Figure 4: **Comparison** **of** **Single** **kernel** **approximation.** Compared to baselines, SVD-based decomposition (LowR.) (McGraw, 2015) and Parallel Simulated Tempering (PST) (Schuster et al.,
2020), our approach (blue) better preserves sharp features on non-convex targets, resulting in lower
LPIPS scores (lower is better). The top-left inset visualizes the error map, with positive errors shown
in red and negative errors in green.


that our gradient-based optimization yields more accurate approximations than PST, consistently
producing stable solutions even in challenging sparse configurations.


Our method’s accuracy extends beyond Gaussian kernels to the more general case of arbitrary singlekernel filters, as shown in Fig. 4. Our method preserves structure across both simple and complex
shapes, while LowRank introduces blocky artifacts and PST yields noisy results at low sample
counts. Quantitatively, our method achieves the lowest LPIPS across all tests, often by a large
margin. It is also much more efficient, requiring only 1 _/_ 100 the iterations of PST.


5.3 SPATIALLY VARYING KERNEL


We present three spatially varying filtering examples in Fig. 5. The first is a 1D spatially varying blur
that uses a pseudo-depth map to simulate a tilt-shift camera effect. The other two are 2D anisotropic
effects: a rotational bokeh blur and a radial motion blur, both controlled by two parameters—blur
intensity and local blur angle.


Our results are visually indistinguishable from the ground truth. As highlighted in the red and
green insets, our method reproduces the complex structure of the GT kernels. In contrast, PST
introduces noise and LowRank oversmooths, and neither recovers the correct kernel shape, while
directly applying GT kernels is prohibitively slow. Quantitatively, our method achieves the highest
PSNR among all methods while maintaining real-time performance.


This performance difference stems from how well each method’s base kernels handle filter-space
interpolation. While all approaches use interpolation to generate the varying filter parameters, our
optimization-based kernels are better conditioned for this process and appear to vary more linearly.
Consequently, they interpolate smoothly to form sharp, complex patterns. PST’s kernels, however,
suffer from poor optimization quality, and interpolating between them simply produces more noise.
Similarly, interpolating the basis kernels from LowRank’s decomposition causes them to average
into indistinct blurs rather than preserving the target structure.


8


Figure 5: **Visual comparison of diverse spatially varying (SV) effects.** We evaluate three SV configurations: 1D tilt-shift blur (top), 2D rotational blur (middle), and 2D radial motion blur (bottom).
We compare our method against Parallel Simulated Tempering (PST) and Low-Rank Decomposition (LowRank).


5.4 ABLATIONS


We conduct ablation studies to validate our main design choices, focusing on both initialization
strategies and different layer configurations.


We first evaluate different initialization schemes across multiple kernels, as shown in Fig. 6. Both
our method and Parallel Simulated Tempering (PST) benefit from the proposed Sparse Sampling
(SS) initialization, which consistently outperforms the Increasing Radial (IR) initialization, while
the Random (Rand) initialization performs worst. Although SS accelerates convergence for both our
method and PST, PST still requires more than 30× the number of iterations to converge compared
with ours, and our final reconstruction quality is significantly higher.


We further study the influence of different configurations, varying the number of layers and the number of samples, as shown in Fig. 7. The convergence curves show that all configurations converge
stably, and configurations with more samples and layers tend to achieve higher quality. Compared
with PST, our method delivers more consistent behavior and better quality across all tested configurations.


For additional results, please refer to the Appendix, which includes ablations on Gaussian kernels
with fewer samples and quantitative evaluations of initialization and regularization strategies on
arbitrary kernels.


6 DISCUSSION AND CONCLUSION


We introduced a differentiable framework that recasts the challenging problem of approximating
large, complex convolution kernels as an end-to-end optimization task. Our approach supports a
wide range of kernels—from simple Gaussians to complex, non-convex forms—and converges to
high-fidelity solutions far more efficiently than prior methods. We extend this with filter-space in

9


34

32

30

28

26

24

22

20

18

16

14

12

10

8


22


20


18


16


14


12


10


8


6


LPIPS ( _×_ 10 _[−]_ [4] ) 105.7 19.4 9.371 8.577


LPIPS ( _×_ 10 _[−]_ [4] ) 35.83 11.1 9.148 7.304


LPIPS ( _×_ 10 _[−]_ [4] ) 36.7 14.98 9.477 3.441


Initialization Step 1010 (PST) Step 122 (Ours) Step 100000 (PST) Step 3000 (Ours)


LPIPS ( _×_ 10 _[−]_ [4] ) 14.81 15.96 7.156 6.669


LPIPS ( _×_ 10 _[−]_ [4] ) 14.81 12.77 4.891 5.887


LPIPS ( _×_ 10 _[−]_ [4] ) 16.63 10.72 5.719 2.054


Initialization Step 1010 (PST) Step 122 (Ours) Step 100000 (PST) Step 3000 (Ours)


34

32

30

28

26

24

22

20

18

16

14

12

10

8


22


20


18


16


14


12


10


8


6


|Col1|Our|s-Rand|
|---|---|---|
||Our|s-IR|
||Our|s-SS|
||PST|-Rand|
||PST<br>|-IR<br>|
||PST|-SS|
||||
||||
||||
|||~~14.31~~<br>~~15.35~~<br>|
|||~~13.83~~|
||||
||||
||||


4

|Ours-Rand<br>Ours-IR<br>Ours-SS<br>PST-Rand<br>PST-IR<br>PST-SS<br>15.35<br>14.31<br>13.43 13.83<br>11.81<br>9.55<br>1000 2000 3000 4000 20000 60000 100000<br>Iteration Step<br>Ours-Rand<br>Ours-IR<br>Ours-SS<br>PST-Rand<br>PST-IR<br>PST-SS<br>12.48<br>11.41<br>9.76<br>8.65<br>7.88<br>5.32|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||Ours-R<br>|and<br>|
||||||
||||~~Ours-I~~<br>Ours-S<br>|S<br>|
||||||
||||~~PST-Ra~~<br>PST-IR<br>|~~d~~|
||||||
||||PST-SS||
||||||
|11.41|||~~12~~|~~48~~|
||||9.|76|
|~~8.65~~|||7.|88|
|~~5.32~~|||||
||||||


Iteration Step


4


Figure 6: **Ablation of initialization strategies on the** _**Flower**_ **and** _**Dove**_ **kernel.** We evaluate both
our method and Parallel Simulated Annealing (PST) combined with three initialization schemes:
Random (Rand), Increasing Radial (IR), and Sparse Sampling (SS).


0.4057


0.5092


0.1971


1.28


1.042


1.192


4.634


Ours 12 _×_ 4


0.7872


_×_ 4 Ours 24 _×_ 4 Ours 32 _×_ 4 Ours 48 _×_ 4

0.8215 0.3793 0.513


14


12


10


8


6


4


2


|Col1|Col2|Col3|
|---|---|---|
|Our<br>Our<br>|s 12x4<br>s 12x6<br>|Ours 32x4<br>Ours 32x6<br>|
|<br>~~Our~~<br>Our<br>~~Our~~|<br>~~s 12x8~~<br>s 24x4<br>~~ 24x6~~|<br>~~Ours 32x8~~<br>Ours 48x4<br>~~Ours 48x6~~|
|<br>Our|<br>s 24x8|<br>Ours 48x8|
|<br>Our|||
||||
||||
||||
||||


Iteration Step


Ours 24 _×_ 4


0.3793


PST 24 _×_ 4


1.323


Ours 32 _×_ 4


0.513


0.1665


PST 12 _×_ 4


_×_ 4 PST 24 _×_ 4 PST 32 _×_ 4 PST 48 _×_ 4

0.9876 1.323 0.9697


PST 32 _×_ 4


0.9697


0.8311


Ours 12 _×_ 6


_×_ 6 Ours 24 _×_ 6 Ours 32 _×_ 6 Ours 48 _×_ 6

0.8337 0.4891 0.5075


Ours 24 _×_ 6


0.4891


PST 24 _×_ 6


1.325


Ours 32 _×_ 6


0.5075


0.1506


PST 12 _×_ 6


_×_ 6 PST 24 _×_ 6 PST 32 _×_ 6 PST 48 _×_ 6

1.182 1.325 0.9494


PST 32 _×_ 6


0.9494


1.613


Ours 24 _×_ 8


Ours 32 _×_ 8


Ours 48 _×_ 8


PST 12 _×_ 8


PST 24 _×_ 8


PST 32 _×_ 8


PST 48 _×_ 8


Ours 12 _×_ 8


Figure 7: **Ablation results for various configurations of samples and layers on Ring kernel.**


terpolation, enabling multi-dimensional spatially varying effects with minimal per-pixel overhead.
A current constraint of our formulation is that it requires access to the target dense kernel during
optimization and filtering. This work opens several promising avenues for future research, including multi-dimensional parameter maps for simultaneous control over kernel attributes and the use
of neural architecture search to discover hardware-optimized filter decompositions. Overall, our
method provides a practical, high-performance solution for advanced image filtering in real-time
applications such as computational photography, while remaining fully differentiable and thus usable as a trainable layer within modern deep learning pipelines.


ACKNOWLEDGMENTS


This work was partially supported by National Key R&D Program of China (No.
2024YFB2809104), NSFC (No. 52532013), and Key R&D Program of Zhejiang (No:
2025C01064). We also thank the anonymous reviewers for their constructive comments.


10


REFERENCES


Pontus Andersson, Jim Nilsson, Tomas Akenine-M¨oller, Magnus Oskarsson, Kalle Astr¨om, [˚] and
Mark D Fairchild. Flip: A difference evaluator for alternating images. _Proc._ _ACM_ _Comput._
_Graph. Interact. Tech._, 3(2):15–1, 2020.


Pierre Charbonnier, Laure Blanc-Feraud, Gilles Aubert, and Michel Barlaud. Two deterministic
half-quadratic regularization algorithms for computed imaging. In _Proceedings_ _of_ _1st_ _interna-_
_tional conference on image processing_, volume 2, pp. 168–172. IEEE, 1994.


Franc¸ois Chollet. Xception: Deep learning with depthwise separable convolutions. In _Proceedings_
_of the IEEE conference on computer vision and pattern recognition_, pp. 1251–1258, 2017.


Joseph W Goodman. _Introduction to Fourier optics_ . Roberts and Company publishers, 2005.


Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand,
Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for
mobile vision applications. _arXiv preprint arXiv:1704.04861_, 2017.


Xu Jia, Bert De Brabandere, Tinne Tuytelaars, and Luc V Gool. Dynamic filter networks. _Advances_
_in neural information processing systems_, 29, 2016.


Masaki Kawase. Frame buffer postprocessing effects in double-steal (wrechless). In _Game Devel-_
_opers Conference 2003, 3_, 2003.


Peter Kovesi. Fast almost-gaussian filtering. In _2010_ _International_ _conference_ _on_ _Digital_ _image_
_computing:_ _Techniques and applications_, pp. 121–125. IEEE, 2010.


Thomas Leimk¨uhler, Hans-Peter Seidel, and Tobias Ritschel. Laplacian kernel splatting for efficient depth-of-field and motion blur synthesis or reconstruction. _ACM Transactions on Graphics_
_(TOG)_, 37(4):1–11, 2018.


Yang Li, Bobo Yan, Jianxin Hou, Bingyang Bai, Xiaoyu Huang, Canfei Xu, and Limei Fang. Unet
based on dynamic convolution decomposition and triplet attention. _Scientific Reports_, 14(1):271,
2024.


Yunsheng Li, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Ye Yu, Lu Yuan,
Zicheng Liu, Mei Chen, and Nuno Vasconcelos. Revisiting dynamic convolution via matrix decomposition. _arXiv preprint arXiv:2103.08756_, 2021.


Cewen Liu, Mengyao Sun, Nanxun Dai, Wei Wu, Yanwen Wei, Mingjie Guo, and Haohuan Fu. Deep
learning-based point-spread function deconvolution for migration image deblurring. _Geophysics_,
87(4):S249–S265, 2022.


Pinjun Luo, Guoqiang Xiao, Xinbo Gao, and Song Wu. Lkd-net: Large kernel convolution network for single image dehazing. In _2023 IEEE international conference on multimedia and expo_
_(ICME)_, pp. 1601–1606. IEEE, 2023.


Sam Martin, Andrew Garrard, Andrew Gruber, Marius Bjorge, Renaldas Zioma, Simon Benge, and
Niklas Nummelin. Moving mobile graphics. In _ACM SIGGRAPH 2015 Courses_, SIGGRAPH ’15,
New York, NY, USA, 2015. Association for Computing Machinery. ISBN 9781450336345. doi:
10.1145/2776880.2787664. [URL https://doi.org/10.1145/2776880.2787664.](https://doi.org/10.1145/2776880.2787664)


Tim McGraw. Fast bokeh effects using low-rank linear filters. _The Visual Computer_, 31(5):601–611,
2015.


Ben Mildenhall, Jonathan T Barron, Jiawen Chen, Dillon Sharlet, Ren Ng, and Robert Carroll. Burst
denoising with kernel prediction networks. In _Proceedings of the IEEE conference on computer_
_vision and pattern recognition_, pp. 2502–2510, 2018.


Simon Niklaus, Long Mai, and Feng Liu. Video frame interpolation via adaptive convolution. In
_Proceedings_ _of_ _the_ _IEEE_ _conference_ _on_ _computer_ _vision_ _and_ _pattern_ _recognition_, pp. 670–679,
2017.


11


Kurniawan Nur Ramadhani, Rinaldi Munir, and Nugraha Priya Utama. Improving video vision
transformer for deepfake video detection using facial landmark, depthwise separable convolution
and self attention. _IEEE Access_, 12:8932–8939, 2024.


Parikshit Vishwas Sakurikar. Epsilon focus photography a study of focus defocus and depth of field.


Christoph Schied, Anton Kaplanyan, Chris Wyman, Anjul Patney, Chakravarty R Alla Chaitanya,
John Burgess, Shiqiu Liu, Carsten Dachsbacher, Aaron Lefohn, and Marco Salvi. Spatiotemporal variance-guided filtering: real-time reconstruction for path-traced global illumination. In
_Proceedings of High Performance Graphics_, pp. 1–12. 2017.


Kersten Schuster, Philip Trettner, and Leif Kobbelt. High-performance image filters via sparse
approximations. _Proceedings of the ACM on Computer Graphics and Interactive Techniques_, 3
(2):1–19, 2020.


Adrian Shajkofci and Michael Liebling. Spatially-variant cnn-based point spread function estimation for blind deconvolution and depth estimation in optical microscopy. _IEEE Transactions on_
_Image Processing_, 29:5848–5861, 2020.


Chao Wang, Krzysztof Wolski, Xingang Pan, Thomas Leimk¨uhler, Bin Chen, Christian Theobalt,
Karol Myszkowski, Hans-Peter Seidel, and Ana Serrano. An implicit neural representation for
the image stack: Depth, all in focus, and high dynamic range. Technical report, 2023.


Yi Wu, Yanyang Xu, Wenhao Zhu, Guojie Song, Zhouchen Lin, Liang Wang, and Shaoguo Liu.
Kdlgt: A linear graph transformer framework via kernel decomposition approach. In _IJCAI_, pp.
2370–2378, 2023.


Zijin Wu, Xingyi Li, Juewen Peng, Hao Lu, Zhiguo Cao, and Weicai Zhong. Dof-nerf: Depth-offield meets neural radiance fields. In _Proceedings of the 30th ACM International Conference on_
_Multimedia_, pp. 1718–1729, 2022.


Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable
effectiveness of deep features as a perceptual metric. In _Proceedings of the IEEE conference on_
_computer vision and pattern recognition_, pp. 586–595, 2018.


Jingkai Zhou, Varun Jampani, Zhixiong Pi, Qiong Liu, and Ming-Hsuan Yang. Decoupled dynamic
filter networks. In _Proceedings_ _of_ _the_ _IEEE/CVF_ _conference_ _on_ _computer_ _vision_ _and_ _pattern_
_recognition_, pp. 6647–6656, 2021.


Shangchen Zhou, Jiawei Zhang, Jinshan Pan, Haozhe Xie, Wangmeng Zuo, and Jimmy Ren. Spatiotemporal filter adaptive network for video deblurring. In _Proceedings of the IEEE/CVF interna-_
_tional conference on computer vision_, pp. 2482–2491, 2019.


A Zing. Extended binomial filter for fast gaussian blur. _Vienna, Austria_, 2010.


12
# DISK: Differentiable Sparse Kernel Complex for Efficient Spatially-Variant Convolution

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Image convolution with complex kernels is common in photography, scientific imaging, and animation, but dense convolution is too expensive for resource-limited devices. Existing approximations, such as simulated annealing and low-rank decompositions, are either slow or struggle with non-convex kernels.
We present a differentiable kernel decomposition framework that represents a spatially variant dense kernel with a small set of sparse samples, assuming the target dense kernel is known for both optimization and filtering. Our method provides (i) end-to-end differentiable sparse-kernel optimization, (ii) shape-aware initialization for non-convex kernels, and (iii) kernel-space interpolation for efficient, multi-dimensional spatially varying filtering without retraining or added runtime cost.
Across Gaussian and non-convex kernels, our method achieves higher fidelity than simulated annealing and lower cost than low-rank decomposition. It is practical for mobile imaging and real-time rendering, and integrates cleanly into learning pipelines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Image convolution with large and complex kernels is computationally heavy. This paper follows a line of research seeking to reduce computational costs through kernel decomposition, using an approximation that achieves both expressibility and efficiency. The paper proposes a method for initializing a differentiable set of sparse kernels and a kernel-space interpolation method for spatially varying kernels. In both single-kernel and spatially-varying-kernel applications, the proposed method shows improvements in speed, quality, and sample efficiency.

### Strengths
- Consistent and strong performance improvement over baselines across two different sets of experiments (single kernel and spatially varying kernel).
- Well-written paper, easy to follow and understandable with proper figures.

### Weaknesses
- **Role of initialization in optimization**: The relationship between initialization and optimization, which appears central to the paper, could be emphasized more clearly. Considering the components affecting both single- and spatially-varying kernel cases, the initialization of the sparse kernel set seems to be a core contribution. Other concepts, such as sparse kernel sets and Dirac delta function-based optimization, have been explored in prior work. Therefore, it is plausible that the improved results primarily stem from the proposed initialization. The paper would be strengthened by an analysis from an optimization perspective, explaining why existing methods struggle, how the proposed initialization addresses these issues, and how it leads to better performance—potentially supported by optimization curves.
- **Limited experiments**: Adding a comparison for single-kernel filtering (i.e., applying the kernels directly to images, not just approximating them) would offer a more complete evaluation of the method’s effectiveness.

### Questions
- Could the authors provide more insight from an optimization perspective on why the proposed sparse kernel set initialization outperforms existing methods? For example, how it alleviates issues in previous approaches and leads to better convergence or performance, potentially supported by optimization curves?
- For Fig. 7 in the appendix, why do the curves for 'Ours 32 x 4' and 'PST 32 x 4' fluctuate instead of showing a monotonic downward trend?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a gradient-based optimization strategy to learn a sequence of sparse convolutional kernels whose sequential application approximates a large, dense convolutional kernel using far less computation than the standard dense convolution. The main innovation in implementing this idea is an initialization strategy for the positions of the sparse (nonzero) components in each of the sequential sparse filters, which is fairly heuristic (balancing fit with the target kernel and leveraging of kernel support expansion via repeated convolution) yet empirically effective. This idea is also extended to spatially varying convolutional kernels, e.g. to model a realistic point spread function as it varies across the field of view. This is done by pre-training a set of basis filters (each of which uses sequential sparse convolution) and then combining these basis filters with spatially varying weights.

### Strengths
Figure 1a is a compelling illustration of the main idea of the method, approximating a dense (e.g. Gaussian) kernel with fairly large spatial extent as a sequence of sparse kernel convolutions whose net computational cost is far less than the cost of a standard convolution with the dense kernel.

I appreciate that the experiments seem to be fair (or more than fair) towards the baselines, in the sense that equal or greater parameters and computation time were afforded to the baselines as to the proposed method. However, I am not an expert in the kernel approximation literature so I take the appropriateness of the chosen baselines at face value.

Figure 5’s illustration of efficient spatially varying filtering is particularly compelling.

### Weaknesses
There are aspects of the presentation that could be improved:
- The abstract describes “convolution with complex kernels”--at this point in the paper, it is unclear whether “complex” means “real plus imaginary” or “complicated”. More generally, the abstract does not clearly explain the task being solved, specifically whether the dense, spatially-varying kernel to be approximated is known in advance or unknown and to be inferred from measurements (after reading the rest of the paper, it seems the former is the goal). This should be made clear early on.
- Section 2.3 summarizes a bunch of related papers, but does not explain their limitations or their relationship with the proposed method. 
- The Charbonnier L1 loss in equation 11 is not defined.
- The font sizes in figures 3 and 4 are too small. It might be preferable to choose one Gaussian scale for the main paper so that the figures could be larger and arranged with PST in one row and “ours” in the other, making the figure easier to interpret (the other Gaussian scale could be shown in the supplement if there is not enough space in the main text). Both sets of figures also include a green and red visualization in the top left of each subfigure (perhaps an error map?) that is not described. Figure 4 also has a number (perhaps a measure of computation time?) in the bottom right of each subfigure, that is not described. Figure 4 caption says “our approach (blue)” but I do not see any blue in the figure.

In terms of the method itself, the main limitation is that it requires knowledge of the target dense filter, for example by calibration. For settings where the kernel is not known or is difficult to calibrate, it would be preferable to be able to jointly solve for the (spatially varying) PSF and apply it efficiently. The primary obstacle to using the proposed method for that goal is that the initialization strategy for the sparse kernels requires knowledge of the dense kernel. I’d at least like to see this limitation mentioned/discussed, even if addressing it is beyond the scope of this paper.

### Questions
As the focus is mainly on improving the speed at which a convolutional kernel can be applied to an image, I wonder what is the computational overhead of pre-optimizing a sparse sequence of filters?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper formulates the efficient convolution problem as an optimization problem involving a few "layers" of sparse discrete convolution operators that are applied in succession. This differentiable formulation enables gradient-based optimizers to find sparse filters that approximate target convolutions by matching their impulse response to the target. The resulting method is quick to optimize, quick to run, and uses little runtime memory. The authors further propose a robust initialization scheme which they verify improves the optimization quality via ablations. Additionally, they propose an time-and memory-efficient way to allow for spatially varying convolutions -- albeit only allowing for a 1D parameter value -- by linearly interpolating between pre-computed "basis" sparse filters where the interpolation weights are determined by the per-pixel parameter value. Overall, the paper presents an effective, simple-to-implement way of performing efficient discrete image convolutions.

### Strengths
- Good-looking results! The supplementary video shows the proposed sparse convolutions producing results visually indistinguishable from the ground-truth.
- Speed and memory requirements are both greatly improved from competing efficient convolution approaches.
- Simple to implement on desktop GPUs or mobile platforms (though pseudocode or code would be appreciated)

### Weaknesses
Since (I believe) the optimization problem in eq. (11) is non-convex in the (offset, weight) parameters, partly due to the dependence of one layer's input on the previous layer's outputs, it's unclear to me how robust the optimization is with respect to vastly different number of samples, and number of layers, or how well it works with even more intricate point-spread functions. If the authors included more robustness tests and described how they tuned (if at all) the optimization hyperparameters such as the optimizer and learning rates, I would find the results much more convincing. 

(Minor weakness) Unclear how well linear interpolation will work with multi-dimensional parameter values: naively, the number of basis filters will need to grow exponentially in D, the dimensionality. On the other hand, only having one spatially-varying parameter value feels limiting to me.

### Questions
In addition to latency/FPS numbers, can you also provide peak VRAM or CPU RAM usage numbers? This seems important to me because being more memory efficient is one of the method's advantages. 

Have you considered second-order-ish optimizers? In many cases the number of parameters sounds small enough that it's probably tractable. But then the optimization is also probably not-so-convex.

Minor typos/suggestions:
I believe most of the citations in the paper should be of the form \citep instead of \citet. 
At L283, it's not clear how the actual interpolation weights are decided from P(x, y)
L341 says "single GPU with computational power equivalent to an NVIDIA RTX 4090." I'm guessing that it's just an L40S, RTX 6000 Ada, or a RTX 4090D? I think it's okay to omit the details, but it'd be helpful to include the VRAM capacity.
L347 is probably talking about a "Snapdragon" 8 Gen 3? The "Snapdragon" is missing from the text
In Fig. 4 "ring", any idea how "Ours 24×4" looks better than "Ours 36×4?" Is it just due to random initialization?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work addresses a significant limitation in practical applications for spatially varying blur effects in computation photography using  image convolution with kernels. Usually Gaussian kernel is considered a standard kernel. However, Gaussian kernels are slow on computationally low-end devices (due to their dense M^2 complexity). To encounter, this paper proposes layer-wise filtering where a kernel is represented by L layers, where each layer has a kernel K with only N samples. Each kernel K is applied to an image resulting in much smllaer number of weights compared to M^2 weights in an ordinary dense kernel of pixel footprint M^2.

### Strengths
- Propose a robust and fast approach to perform image convolutions
- Decouples filter generation from the  spatial resolution by introducing filter space interpolation
- The idea is simple and effective
- Results are very convincing.

### Weaknesses
- The paper needs to do a bit more job in explaining the relation wrt A-trous filters or max-pooling like approaches used in U-Nets for increasing the receptive field of kernels.

- Please always mention how to read different metric values (higher is better or lower is better). 

- There are no theoretical guarantees provided that explains why layered kernels would be robust in general. 

Please see my questions below that also directly highlight my concerns.

### Questions
- How do you obtain a blur intensity map that is conditioned for the filter generation process? 
- How the whole process of structuring a filter as layers different from a convolution followed by e.g, apply max pooling to an image and then applying the same filter to the image? It will automatically increases the receptive field of the kernel without any additional cost to computing kernel weights
- What is the similarity of the approach to the A-trous filters where the number of learnable parameters remain the same despite the receptive field?
- How the kernels adapt the anisotropy of the signal? What if you want nice anisotropic bokeh effect for artistic purposes?
- What are “filter objects” in L291? 
- It is not clear what objects are interpolated? Is it the interpolation between the trained parameters \alpha_k?
- I would like more explanation on the implementation details. How the optimization works? You convolve a kernel with a Dirac function, this gives you an image with a single non-zero pixel. This image is then matched to a target image with a target kernel applied to a Dirac function. Is my understanding correct?
- Also, how this idea of layered kernels different from what is used in CNNs or U-Nets?

Figure 1 should appear early in the paper (on the second page), or better as a teaser on the first page.

### Soundness
3

### Presentation
3

### Contribution
2

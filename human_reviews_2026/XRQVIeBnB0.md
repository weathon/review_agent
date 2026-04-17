# Lookup multivariate Kolmogorov-Arnold Networks

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
High-dimensional linear mappings, or linear layers, dominate both the parameter count and the computational cost of most modern deep-learning models. We introduce a general-purpose drop-in replacement, lookup multivariate Kolmogorov-Arnold Networks (lmKANs), which deliver a substantially better trade-off between capacity and inference cost. Our construction expresses a general high-dimensional mapping through trainable low-dimensional multivariate functions. These functions can carry dozens or hundreds of trainable parameters each, and yet it takes only a few multiplications to compute them because they are implemented as spline lookup tables. Empirically, lmKANs reduce inference FLOPs by up to 6.0× while matching the flexibility of MLPs in general high-dimensional function approximation. In another feedforward fully connected benchmark, on the tabular-like dataset of randomly displaced methane configurations, lmKANs enable more than 10× higher H100 throughput at equal accuracy. Within the framework of Convolutional Neural Networks, lmKAN-based CNNs cut inference FLOPs at matched accuracy by 1.6–2.1× and by 1.7× on the CIFAR-10 and ImageNet-1k datasets, respectively. Our code, including dedicated CUDA kernels, is available online at https://github.com/schwallergroup/lmkan.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes to increase KANs' inference efficiency via lookup tables. Their construction expresses a general high-dimensional mapping through trainable low-dimensional multivariate functions. Although these functions may carry a large number of trainable parameters, they sparsely activate during inference time; as a result, it can be implemented as a sparse lookup table. They show the effectiveness of the proposed method on a wide range of datasets, outperforming MLPs and FastKANs.

### Strengths
* The paper is well written and the presentation is good
* The proposed sparse lookup table method is simple yet effective
* Experiments are extensive and convincing

### Weaknesses
* The application range of the method should be discussed, i.e., when do we expect this method to be less of an advantage?
* Although the proposed method can increase inference efficiency, it does not necessarily help training when a larger than 1 batch is being used.

### Questions
* The default setup is $d=2$, how about $d=1$ or $d=3$ or larger? What's the philosophy for choosing $d=2$?
* How about training efficiency? When the batch is larger than 1, different samples can activate different basis functions.
* Section 4.3, it should be "IMKAN", not "LMKAN"

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces lookup multivariate Kolmogorov-Arnold Networks (lmKANs). The central idea is to replace the 1D univariate activation functions used in standard KANs with 2D multivariate functions. The key component of this architecture is the implementation of these 2D functions as $\mathcal{O}(1)$ spline lookup tables. The authors posit that this approach "decouples" the model's parameter capacity which could be increased by grid point from its inference cost. This, in theory, allows lmKANs to possess a significantly larger parameter count than an MLP with only a $\sim 2\times$ increase in theoretical FLOPs. The paper presents experiments on general function approximation, the methane molecule dataset, and image classification to demonstrate that lmKANs achieve a superior trade-off between accuracy and inference cost compared to MLPs.

### Strengths
- The idea of using low-dimensional multivariate functions (specifically 2D) implemented as $\mathcal{O}(1)$ spline lookup tables is novel.
- The authors have clearly put significant effort into the implementation, including custom CUDA kernels and a sophisticated multi-stage training pipeline with Hessian regularization.

### Weaknesses
- The paper's primary thesis is that lmKANs are superior because parameters can be added "for free" in terms of FLOPs by increasing the grid resolution $G$. This is the claimed advantage over MLPs, where adding parameters requires increasing width $N$ at an $\mathcal{O}(N^2)$ FLOPs cost. However, the main experimental comparisons (Figs 4, 6, 7) do not test this thesis. These experiments fix $G$ and vary $N$, just like an MLP. These results show, at best, that lmKAN's $\mathcal{O}(N^2)$ scaling factor is better than an MLP's, but they provide zero evidence for the central claim about the benefit of scaling $G$.
- The most important experiment is missing from the paper. To validate the core claim, the authors must provide an analysis where network width $N$ is fixed, while grid resolution $G$ is varied (e.g., $G \in \{4, 8, 16, 32\}$). This experiment must plot two curves on the same y-axis: (1) model performance (e.g., MSE) and (2) inference cost (FLOPs and/or wall-clock time). The paper's hypothesis predicts that performance will improve while inference cost remains flat. Without this plot, the main premise is entirely unsubstantiated.
- The only analysis of $G$ is relegated to the appendix (e.g., Fig 19, 22, 24). These plots are incomplete, showing only "Performance vs. $G$". They critically omit the corresponding "Inference Cost vs. $G$" data, which is essential to validating the trade-off. Furthermore, these plots (e.g., Fig 19) show a U-shaped curve where performance worsens after $G=12$, which contradicts the paper's narrative that multivariate functions can "accommodate" large parameters without issue. This is not adequately discussed.
- The paper motivates the move from 1D to 2D functions by claiming 1D functions with dense grids are "unstable". To support this, it compares its 2D B-spline lmKAN to FastKAN (Fig 8), which uses 1D Gaussian RBFs. This is an invalid. The degradation in FastKAN could be a property of RBFs, not a fundamental flaw of 1D spline functions. The correct baseline would be a 1D B-spline KAN using the same $\mathcal{O}(1)$ lookup technology. This baseline is missing.

### Questions
- Can the authors please provide the critical experiment described in Weakness #2? Specifically, for the "General Function Approximation" task (from Fig 4), please provide a plot with a fixed hidden dimension (e.g., $N=256$) while sweeping $G$ (e.g., $G \in \{4, 8, 16, 32\}$). This plot must show (a) Test MSE vs. $G$ and (b) Inference FLOPs vs. $G$. This is essential to validate the paper's core premise.
- Why did the authors choose to compare against the RBF-based FastKAN to justify the 1D vs. 2D claim, rather than the much more appropriate baseline of a 1D B-spline KAN using the same lookup table method? Without this direct comparison, the claim that 2D functions are inherently more stable is unsupported.
- The appendix figures (e.g., Fig 19) show that performance degrades as $G$ becomes very large. This seems to contradict the motivation that multivariate functions can "accommodate" a large number of parameters without issue. Does this imply that the "free parameter" benefit is limited in practice? How do the authors reconcile this with their primary motivation?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes lookup multivariate Kolmogorov-Arnold Networks (lmKANs), a novel architecture that replaces high-dimensional linear mappings with trainable low-dimensional multivariate functions (primarily d=2) implemented via spline lookup tables.

lmKANs split high-dimensional inputs into low-dimensional subgroups (e.g., 32D input → 16×2D subgroups), each processed by a multivariate spline function.
Spline lookup tables leverage the local support of B-splines to achieve O(1) inference per subgroup. This design reduces inference FLOPs by up to 6× for general function approximation.

Custom CUDA kernels further accelerate computation, achieving up to 88× faster inference on H100 GPUs. Across tasks, lmKANs consistently outperform MLPs and FastKANs in both accuracy and efficiency.

### Strengths
The idea is original and overall design logics is very well organized. By grouping high-dimensional inputs into low-dimensional subgroups, lmKANs avoid the exponential complexity of high-dimensional splines. The look-up table take advantage of this The Sigmoid-like σ-grid very elegant way to adapts to unbound activations into bounded region.

Experiments are solid: across four task types, the paper tests different model scales, training budgets, and data formats to show lmKANs’ robustness.
Engineering details, including CUDA implementation, are clear too.

Also, the paper does a good job of explaining a relatively complex idea in a clear and accessible way.

### Weaknesses
Limited high-dimensional multivariate validation
The paper focuses exclusively on 2D lmKANs. It would provide values to validate for other d=4/8. Even on very small experiments this would help.

CNN architecture and  implementation
The experiments on CIFAR uses “For CIFAR-10, our backbone architecture consists of five 2 × 2 convolutions, each with stride 2” and for ImageNet “For ImageNet, we downsample the images to a resolution of 81 × 81 pixels. Our backbone consists of four convolutional layers with the 3 × 3 kernel size and stride 3 and two fully connected layers” These are not typical architectures for these datasets. Does this imply lmKANs cannot integrate with standard models like ResNet or AlexNet?

Additionally, Section 4.3 mentions that lmKAN-CNNs “cast convolutions to fully connected layers via memory manipulations.” This seems like a temporary workaround rather than a native multivariate spline convolution.

### Questions
Appendix D mentions that second-order B-splines match ReLU’s smoothness. Did the authors try higher-order splines?

Also, Hessian regularization is only mentioned in the appendix. Was it applied to all experiments in the main paper? If so, how does it affect the results?

Did the author tried other options for mapping unbounded activation to bounded regions? For example, Erf-based function, such that we may assume the activation follows a normal distribution.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces lmKANs, a lookup multivariate KAN that replaces the high dimensionality of the univariate functions employed in KANs with multivariate low-dimensional functions, where the inner functions are spline lookup tables. The paper delivers custom CUDA kernels for accelerated inference on lmKANs, conferring 88x faster inference than standard KANs. The paper then compares this implementation with standard MLPs across a range of tasks and demonstrates that lmKANs outperform in terms of accuracy and performance.

### Strengths
The paper present an important contribution to KAN development, making KANs easier to train and more scalable. The method presented is novel and is a significant improvement to address hurdles when training KANs.

The paper does a good job benchmarking lmKANs against MLPs. I particularly appreciate the MLP 1/2 traces, giving a sense of the compute benefits afforded by lmKANs.

Overall, the approach is interesting and novel, and provides a way to increase the expressivity of KAN training.

### Weaknesses
In the main paper, the authors focus far more on the computational benefit afforded by the accelerations than the theoretical foundation of how this model. However, the supplementary information dives into the theory well and motivates lmKANs, all of which is missing from the main manuscript. This unfortunately detracts from the impact of the main paper. I also struggled a bit to connect the dots between the motivation of the paper and the results presented. Adding dozens or hundreds of more parameters should significantly increase expressivity, but for some plots (e.g. Figure 4) the difference in MSE is on the order of 1e-2, which is surprising given the motivation.

The paper also claims that e.g. Moradzadeh et al do not compare against MLPs, which is untrue, taking a look at the manuscript. This oversight detracts from the novelty of the proposed method, given that other similar approaches to also result in KANs that match performance or outperform MLPs.

While the idea of using a lookup table to convert internal functions to O(1) complexity is interesting, the authors don't explore the added burden of HPO to find the optimal grid size to match or exceed MLP performance.

### Questions
Do you have statistics around training time when training lmKANs to convergence versus an equivalent MLP? Figure 18 starts to get at this result with the MLP 1/2 trace, but I would love full training stats.

How sensitive/brittle is training as a function of HPO over G? Is it possible to provide some sensitivity analysis of lmKAN performance to G? I'm trying to understand how much increasing the parameter capacity of KAN's improves expressivity.

### Soundness
2

### Presentation
2

### Contribution
3

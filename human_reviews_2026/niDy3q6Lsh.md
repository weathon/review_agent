# One Scale at a Time: Scale-Autoregressive Modeling for Fluid Flow Distributions

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 4, 4, 8

## Abstract
Analyzing unsteady fluid flows often requires access to the full distribution of possible temporal states, yet conventional PDE solvers are computationally prohibitive and learned time-stepping surrogates quickly accumulate error over long rollouts. Generative models avoid compounding error by sampling states independently, but diffusion and flow-matching methods, while accurate, are limited by the cost of many evaluations over the entire mesh. We introduce scale–autoregressive modeling (SAR) for sampling flows on unstructured meshes hierarchically from coarse to fine: it first generates a low-resolution field, then refines it by progressively sampling higher resolutions conditioned on coarser predictions. This coarse-to-fine factorization improves efficiency by concentrating computation at coarser scales, where uncertainty is greatest, while requiring fewer steps at finer scales. Across unsteady-flow benchmarks of varying complexity, SAR attains substantially lower distributional error and higher per-sample accuracy than state-of-the-art diffusion models based on multi-scale GNNs, while matching or surpassing a flow-matching Transolver (a linear-time transformer) yet running $2$–$7\times$ faster than this depending on the task. Overall, SAR provides a practical tool for fast and accurate estimation of statistical flow quantities (e.g., turbulent kinetic energy and two-point correlations) in real-world settings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces Scale-Autoregressive Modeling (SAR) — a hierarchical generative model for sampling unsteady fluid flows directly on unstructured meshes, enabling both faster generation and more accurate flow statistics than prior diffusion or flow-matching approaches.

Instead of simulating PDEs forward in time, SAR models the statistically stationary distribution of flow states, sampling full fluid fields (velocity, pressure, etc.) directly from geometry and physical parameters.

### Strengths
The paper introduces a new paradigm for generative fluid modeling: instead of treating the flow field as a single object to denoise (as in diffusion or flow-matching models), it models it hierarchically across spatial scales.

This “scale-autoregressive” idea elegantly parallels the physical cascade in turbulence — from large coherent structures to small eddies — making the model physically meaningful and statistically efficient.

It provides a clear alternative to traditional diffusion schemes that require uniform sampling effort across all scales, regardless of local uncertainty.

### Weaknesses
The model uses a fixed number of scales (typically 3–4), determined by a pre-defined coarsening algorithm. It is more like a structure design.

### Questions
what kind of physics do we learn from here? Why do we need train different models for different cases, can you train them together? What is the physical meaning of “scale” in SAR? How does SAR differ from a standard multigrid solver? How does SAR handle different flow regimes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes modeling fluid distributions autoregressively from a coarse to fine scale. At each scale, an encoder passes information about the global scale and a module passes information from all prior scales to the current scale. A flow-matching denoiser is trained to denoise the current scale. This repeats until the finest scale is denoised and a prediction is output. The model improves mainly on speed and has marginal gains in accuracy.

### Strengths
- The use of modeling at each scale is interesting and seems to leverage multigrid well
- The speed benefits seem to work well
- The evaluations seem to be done well

### Weaknesses
- Progressively refining a coarse to fine grid and using a multiscale approach in PDE modeling seems to be popular, this isn’t an issue but should just be mentioned in related works:
    - https://arxiv.org/abs/2506.04528
    - https://arxiv.org/pdf/2210.02573
    - https://arxiv.org/pdf/2207.11417
    - https://www.arxiv.org/pdf/2510.16071
    - Probably more that I am missing ... 

- In particular, this work (https://arxiv.org/pdf/2505.02450) seems to use similar concepts (denoising coarse grid, then passing outputs to a finer scale, etc.), but on a regular grid rather than a mesh.

- In general, gains in accuracy seem to be marginal, however, this isn’t a huge problem since the speed is better.

- A comparison to a latent FMT would be interesting. It is known that latent diffusion can outperform pixel-space counterparts, both in image modeling and in PDEs (https://arxiv.org/abs/2507.02608). The speed gains by using latent diffusion and potential performance gains may make latent flow matching a competitive approach, also demonstrated by prior works (https://arxiv.org/abs/2503.22600). 

- More of a philosophical point, but diffusion seems to already denoise samples from a coarse to fine scale. During forward noising, Gaussian noise first corrupts high-frequency features before corrupting low-frequency ones. Many denoisers exhibit behavior where large-scale, low-frequency features are resolved first before high-frequency features are resolved later (https://arxiv.org/abs/2505.11278v1).

### Questions
These are just a few clarification questions on my part, nothing wrong with the paper:

- Is the Encoder/Autoregressive Module/Decoder jointly trained?

- Does the VAE downsample the nodes? I’m not sure what nodal compression means.

- The training seems to be on 4 denoising steps in [0,1] but the inference is evaluated with more steps/finer discretization? 

- Figure 9 may benefit from displaying a reference simulation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The work proposes a method of predicting fluid dynamics with an "autoregressive" model running on different coarse scales. In this method, fluid dynamics are considered at hierarchical mesh levels from coarse to fine. In particular, the physics are described by state variables at mesh points.  The proposed model devises a neural network to generate the entire fluid field from coarse to fine levels. The proposed method achieves performance comparable to that of previous methods, while significantly reducing computation time.

### Strengths
1. The design of generating fluid dynamics using a hierarchical model is novel. This method has been considered in image generation but not in physics simulations. 

2. The model design is reasonable. In particular, I appreciate the design of the encoder for the spatial domain and the hierarchical generation of different levels with generative modeling.

### Weaknesses
Experiment results:

1. The performance of the proposed model is on the weak side. It seems to underperform FMT-8 in Table 1. There are only four baselines in Table 1, and three of them are from a single paper. Should the proposed method be compared against more baselines? 

2. In Figure 3, it seems that more iterations of SAR do not bring much performance improvement for the last two dots (contant # vs per-scale #). Any explanation.

3. It seems that the performance in terms of W2 and R2 is not the strength of the proposed method, but the computation speed is. Therefore, the writing probably needs to better state the contribution.  

Writing issues: 

1. Many notations are used without careful definitions. The notation system is also somewhat messy, making the calculation hard to understand. Here are a few examples:

1). Node attributes: V_c, where is c from? What does it mean? Is "c" a label or a variable representing some value?
2). "SAR partitions the node set V into K disjoint subsets": it seems that the partition is not arbitrary but related to the grid hierarchy. How is the partition done? The formal paper is supposed to be self-sustained and cannot depend on the appendix. 
3). You have a definition of \Gamma, but can you also explicitly make the meaning of gamma_i explicit? Should \gamma_i be in {1, ..., K}? 

2. "The full SAR model is trained by optimizing the flow-matching objective applied to the output of the sampler network." Does the training objective include the encoder that computes Y? It says flow-matching here, but it also mentions "diffusion-based approach".  

3. A lot of key experiment details are missing from the formal text. For example, what are scales of these problems (e.g. number of mesh points)? Is computation speed an important factor? Important experiment results are put in the appendix.

### Questions
I have left my questions in the section on weaknesses. Looking forward to seeing answers.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper under review deals with fluid flow prediction using a graph neural network multiscale approach. In this, the computational mesh sis subdivided into successively refined meshes and the prediction on the sucessive, refined scale is conditioned on the information already gathered on the coarser scale. While this refinement is performed over several (mostly three) layers, the generation of the solution information is performed generatively using GNN-based conditional flow matching. To achieve awareness of relevant global information, the network geometry and boundary conditions are encoded by some neural network that passes the obtained global information on to all layers. In some cases, a dimension reduction by a rather shallow VAE is performed before the multi-scale flow matching. The authors then implement their algorithm and train it on thee recently published data sets - Ellipse, EllipseFlow and Wing - where the first two of them are 2D and the last is 3D. The training data stems from numerical simulation. The author show that their approach, when compared with statistical measures like the Wassterstein-2-distance, is comparable with the SOTA set by very recent GNN with various additional architecture elements, while mostly requiring less trainable weights. Some Ablation studies are provided,

### Strengths
- This is a very interesting and timely contribution to a rapidly developing field of learning turbulent flows with gerenative GNN-based methods. The multiscale idea is very clear and compelling. 
- The model shows very good performance over a range of data sets, be it 2D or 3D.
- The chosen architectures are original, also the combination with a VAE of modest size is an interesting experiment.
- The paper is mostly well written and easy to follow.
- The graphics and figures are clear 
- A very detailed appendix enables the complete understanding of the method

### Weaknesses
- My main criticism is the very loose usage of mathematical language. E.g. FlowMatching is called a diffusion method (i.e. a Stochastic Differential Equation), whereas it actually is a neuralODE without stochaticity trained in a special way. The underlying PDEs for the probability path are transport equations, not Focker - Planck. This should be corrected.
- Similarly, the title SAR is a misnamer. FM has nothing to do with a regression task, but it learns a (optimal) transport task operating on distributions. What the authors call "Scale auto regressive" actually is conditional FM over scales. This should be clarified.
- I find the evaluation with W_2-distance and the visual comparison of 1st and 2nd order point statistics limited. The authors should also look into spatial power spectra.
- Some formulations are misleading - e.g. there is no such thing as 'non local physics' (would contradict special relativity) - the authors presumably mean global structures in physical systems. This should be made precise.

Minor comments 
l. 98 . 'strategically' ?!
l. 106-107 Despite ... inductive bias. Unclear what that means
l. 246-247: how is  'stochastic complexity' defined and is this statement checked experimentally?
l. 255: giving the sd without reference to the scale of the input signal is meaningless
Eq. (4) what quantity is distributed according to q_{k,r}. Check all conditionings in your notation for correctness

### Questions
Can you provide a table which  element where already present in prior models and which ones are 100% original?

### Soundness
4

### Presentation
3

### Contribution
4

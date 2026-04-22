# Decoupled-Value Attention for Prior-Data Fitted Networks: GP-Inference for Physical Equations

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Prior-data fitted networks (PFNs) are a promising alternative to time-consuming Gaussian process (GP) inference for creating fast surrogates of physical systems. PFN reduces the computational burden of GP-training by replacing Bayesian inference in GP with a single forward pass of a learned prediction model. However, with standard Transformer attention, PFNs show limited effectiveness on high-dimensional regression tasks. We introduce Decoupled-Value Attention (DVA)-- motivated by the GP property that the function space is fully characterized by the kernel over inputs and the predictive mean is a weighted sum of training targets. DVA computes similarities from inputs only and propagates labels solely through values. Thus, the proposed DVA mirrors the GP update while remaining kernel-free. We demonstrate that PFNs are backbone architecture invariant and the crucial factor for scaling PFNs is the attention rule rather than the architecture itself. Specifically, our results demonstrate that (a) localized attention consistently reduces out-of-sample validation loss in PFNs across different dimensional settings, with validation loss reduced by more than 50\% in five- and ten-dimensional cases, and (b) the role of attention is more decisive than the choice of backbone architecture, showing that CNN, RNN and LSTM-based PFNs can perform at par with their Transformer-based counterparts. The proposed PFNs provide 64-dimensional power flow equation approximations with a mean absolute error of the order of $10^{-3}$, while being over $80\times$ faster than exact GP inference.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
1

### Summary
This paper proposes Decoupled Value Attention (DVA), a new architecture of Prior data Fitted networks (PFN), with the main contribution being an attention model that separates the Key-Query interaction to apply only on the features (x), to provide weights multiplied by the values y.  This decoupling makes the estimation process more intuitive as it measures the similarity between features via the Key-Query matrices to create weights multiplied by the matching labels (y), similarly to a Gaussian process. The authors claim the method they provide leads to the following main contributions: 

 
1) The authors show that the DVA reduces the difference between predicted and true posterior distribution in PFN training.  
2) The paper shows that a CNN based PFN equipped with DVA performs comparably to a Transformer based DVA-PFN  
3) The authors show that DVA enables PFNs to scale complex, high-dimensional problems. The authors demonstrate this on a 64-dimensional power flow simulation.

### Strengths
1. The premise of the paper is reasonable and seems to be well-founded. The summation of the label and feature embeddings in the original PFN appears to be hurting the results in the provided synthetic cases. 
  

2. The "Attention is More Important than Architecture" finding is a significant contribution. By showing that a CNN+DVA can match a Transformer+DVA, the authors successfully decouple the PFN concept from a strict reliance on the Transformer, opening the door to other, potentially more efficient options.

### Weaknesses
1. The key demonstration of the original PFN's power was its ability to learn a complex hierarchical model, and from my understanding this is the main reason to implement the K data set training they suggest. However, the authors only test the mechanism against a simple, fixed hyperparameter RBF kernel. Thus, significant omission to not test if DVA retains the ability to approximate these more complex, mixed priors, which was a primary advantage of the PFN framework is problematic. 
 

2. The original PFN paper validated its method on a wide range of real-world tabular data. This paper ignores those benchmarks and instead introduces a new, highly specific physics problem (power flow). Without a direct comparison on the original paper benchmarks, it is impossible to assess if DVA is an improvement or a specialized architecture that only excels on certain tasks. 
 

3. The authors claim DVA succeeds in high dimensional regimes where the VA PFN simply stops learning, they link this threshold of D = 10 in their GP tests, but when looking at the Müller et al works, they consider datasets with a large amount of features (e.g., covertype = 55), of a similar order of the 64 feature space considered in the paper “power flow” problem.  

 

4. Section 2.2 sketches the PFN training objective, but a reader unfamiliar with PFNs will still need (Muller et al) to follow training/inference. 

The paper’s central contribution is the DVA method, which adjusts the original PFN method. This idea seems sensible, as the usage of the attention Key-Query to generate weights for the label is highly intuitive and closely matches the established GP methods. Still, as the main claim of the paper is to provide a better alternative to an existing method, it was not clear enough that the advantages appear numerically and that there is a clear advantage for high-dimensional cases.

### Questions
1. A key result of the original PFN paper (Muller et al) was its ability to learn an intractable posterior from a hierarchical GP prior.  Can the authors provide results showing DVA's performance on this task? 
 

2. The original PFN paper (Muller et al) performed a high variety of tests and comparisons with a great contribution for tabular data, while this paper focuses mainly on synthetic data sets and the “power flow” problem, is there any reason to choose that specific problem? Could the method be applied to additional cases?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents decoupled-value attention for prior data fitted networks (PFNs). It computes $Q$/$K$ only from inputs and pass labels only through $V$. It ensures that attention weights depends purely on input similarity while output flows via values. The design mirrors GP conditioning and aims to reduce the bias observed in PFNs that attend over concentrated (x,y) embeddings. The method is instantiated with both Transformer and CNN based PFNs and is evaluated over synthetic GP tasks, Rosenbrock, and a AC power-flow surrogate. Reported results show lower NLL and inference speedup vs exact GPs.

### Strengths
- simple and clear method

- DVA reduces validation NLL across dimensions specifically in 5D/10D where vanilla attention saturates early.

- the method is backbone agnostic and the authors have shown similar improvements for both CNNs and Transformers.

- high dimensional experiments (64D) with inference speedups shows practical utility.

### Weaknesses
- theory is very light: the localization argument is intuitive and refers prior PFN theory but there is no formal generalization/bias-consistency result for DVA itself. The softmax non-negativity vs possibly signed GP coefficients is acknowledged but it is not analyzed beyond a short discussion.

- scalability analysis: the attention compute remains quadratic in context size. there is no complexity/memory study comparison to linear/performer style attention under PFN training.

- weak baselines: Power-flow experiments compare to exact GP and PFN+VA and are missing strong sparse GP baselines, kernel regression / kNN, RFF/linear attention variants in higher-D, and physics-informed surrogates common in this domain.

- appendix should have a consolidated hyperparameter table.

- thin ablations

### Questions
- could you add higher-D comparisons to (a) sparse/variational GP, (b) kNN / kernel regression, (c) performer/linear attention PFNs, and (d) a physics-informed baseline on power flow?

- can you report ablations on (a) capacity of $\phi_x$/$\phi_y$ and prediction head $g()$, (b) effect of bucket count on NLL/MSE.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Decoupled-Value Attention (DVA) for Prior-Data Fitted Networks (PFNs), arguing that standard PFN attention mechanisms fail to scale beyond ~10 input dimensions. DVA computes attention affinities (queries and keys) purely from inputs while propagating labels solely through values, motivated by the structure of Gaussian process inference where kernels operate on inputs and predictions are weighted sums of training outputs. The authors demonstrate that DVA reduces validation loss by >50% in 5D and 10D settings compared to vanilla attention, and show that CNN-based PFNs with DVA can match Transformer performance. They apply their method to 64-dimensional power flow equations, achieving MAE ~10^-3 while being 80× faster than exact GP inference.

### Strengths
## Strengths 

* Clear problem identification and motivation: The paper clearly identifies a real limitation of existing PFNs—their inability to scale beyond ~10 dimensions—and provides intuitive motivation for the proposed solution through the lens of GP inference. The observation that standard PFN attention couples inputs and outputs in ways that break localization is well-articulated.

* Comprehensive experimental validation: The systematic evaluation across multiple dimensions (1D, 2D, 5D, 10D, 64D), architectures (CNN, Transformer), and attention mechanisms (VA, DVA, kernel-based) is thorough. The validation loss curves in Figure 2 clearly demonstrate the benefit of DVA, especially the striking saturation of VA-based models in 10D.

* Architecture-agnostic insight: Demonstrating that CNN-based PFNs can match Transformer performance when equipped with proper attention is valuable. This challenges the implicit assumption in prior PFN work that Transformers are necessary, and suggests attention design is the critical factor.

* Practical application: The 64D power flow experiments demonstrate real-world applicability. Achieving 80× speedup over GP while maintaining acceptable accuracy (MAE ~10^-3) is practically significant for power grid applications.

* Reproducibility: The appendix provides detailed hyperparameter ranges, architectural specifications, and implementation details that aid reproducibility (relatedly, particularly appreciate the code in supplementary materials)

### Weaknesses
## Weaknesses

* Limited novelty of the core idea: Decoupling inputs and outputs in attention is not particularly novel. The paper positions DVA as specifically designed for PFNs and GP-mimicking, but the core mechanism (computing affinities from inputs only) is a straightforward design choice that has been explored in various forms in the attention literature. The contribution feels incremental rather than introducing fundamentally new concepts.

* Overstated connection to Gaussian processes: The claim that DVA "mirrors GP inference" is overstated. While there are superficial similarities, critical differences remain:

1. GP weights β(x*) can be negative and don't sum to 1, while DVA's softmax produces non-negative normalized weights
2. GPs have a principled covariance kernel with theoretical properties; DVA learns arbitrary similarity via dot products
3. The authors acknowledge this in Section 3.1 but still heavily market the "GP alignment" throughout

The connection is more of a loose analogy than a rigorous correspondence. The paper would be stronger if it positioned DVA as "inspired by" rather than "mirroring" GP inference.

* Insufficient comparison to related work: The paper lacks comparison to other localized attention mechanisms or other recent PFN improvements. For example:

1. How does DVA compare to explicit localization post-processing (Nagler 2023)?
2. What about other input-localized attention mechanisms from the broader literature?
3. The only comparison to kernel-based attention is in Figure 1— and authors test on functions mismatched to the RBF kernel rather than showing where kernel-based attention might excel

* Limited theoretical understanding: While the paper demonstrates empirically that DVA works, it provides little theoretical insight into why it works. Questions remain:

1. Under what conditions does input-only attention provably improve over joint attention?
2. Can you characterize when DVA will succeed vs. fail?
3. What is the sample complexity with DVA vs. VA?

The paper mentions Nagler (2023)'s theoretical results on bias but doesn't rigorously connect DVA to that framework to my understanding.


* Missing ablations and analysis:

1.No ablation on the value encoder φ_y—does its design matter?
2. What happens with different query/key projections W_q, W_k?
3. The paper claims DVA "remains kernel-free" as an advantage, but doesn't test whether a learnable kernel parameterization might work even better


* Evaluation metrics: For the power flow application, MSE and MAE are reported, but uncertainty quantification—one of the main advantages of GPs—is not evaluated. Can DVA-based PFNs provide calibrated uncertainty estimates? This seems like a critical missing piece given the motivation.

### Questions
## Questions for Authors

* Theoretical characterization: Can you provide any theoretical analysis of when and why DVA reduces bias compared to VA? Even in a simplified setting (e.g., linear case), formal guarantees would strengthen the contribution.

* Kernel-based attention: The experiment in Figure 1 seems designed to make kernel-based attention look bad. Can you show cases where kernel-based attention with an appropriate kernel (e.g., Matérn for less smooth functions) actually works well? What is the fundamental limitation?

* Uncertainty quantification: One major advantage of GPs is calibrated uncertainty. Do PFNs with DVA produce well-calibrated predictive distributions? Can you compare predictive variance quality to exact GP?

* Scalability limits: Where does DVA break down? The 10D experiments show clear improvements, but the 64D power flow uses only 500 training samples. Have you tested truly high-dimensional problems (e.g., 100D+) with sufficient data?

* Value encoding: You mention that "the final head g(·) and encoding outputs in the value V" help adjust DVA output toward the true GP posterior mean. Can you ablate this claim? How sensitive is performance to the value encoder design?

* Comparison to Nagler 2023: Nagler argues for post-hoc localization. Can you compare DVA to their approach? Does DVA achieve better or different localization than explicit post-processing?

* Output information: DVA completely removes output information from attention affinities. Are there cases where having some output information is beneficial? Could a weighted combination of input-only and joint attention work better?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Prior-Data Fitted Networks (PFNs) are aimed to replace costly Gaussian process (GP) inference with a single learned forward pass for fast physical surrogates, but standard PFNs struggle in high dimensions. Authors introduce Decoupled-Value Attention (DVA), which bases attention weights only on inputs and passes labels only through values, resembling GP-style weighted averaging of targets. DVA improves PFN training in 5D-10D (over 50% lower validation loss), lets CNN backbones match Transformers (implying attention design matters more than architecture), and yields a 64D power flow surrogate with ~10^-3 MAE and a reported >80x inference-time speedup over their GP baseline.

### Strengths
1) Paper includes simple, targeted architectural contribution (Decoupled-Value Attention) with clear motivation.

2) Authors demonstrate that “vanilla attention PFNs” stall in 5D-10D tasks, where validation loss flattens early and final MSE remains high. With DVA, PFNs continue improving and achieve dramatically lower validation NLL and MSE in 5D and 10D (often >50% reduction in their reported residual bias).

3) Even accounting for fairness caveats (see weaknesses), once trained, PFN+DVA can generate all bus voltages for 4,500 samples in ~0.13–0.17 seconds that is essentially instant compared to AC power flow solvers, and is about two orders of magnitude faster than their GP baseline.

### Weaknesses
1) A defining feature of GP inference is not just accurate point predictions, but calibrated posterior uncertainty. The paper does not measure calibration, posterior variance quality, credible interval coverage, or any decision-making utility based on uncertainty. All reported metrics are pointwise MSE/MAE and wallclock time. This is a mismatch between the claimes and the experimental support. 

2) Authors claim >80x faster than exact GP inference, citing ~0.13 s for PFN vs ~11 s for GP on the 64D power flow task, but the GP baseline is actually 32 separately trained single-output GPs evaluated together, while PFN is a single multi-output model that predicts all 32 voltages in one batched forward pass; PFN inference likely ran on GPU after ~14 hours of pretraining, while the GP hardware/batching setup is not described. Can you clarify exactly how GP inference time was measured (hardware, batching, parallelization) and restate the 80x claim with those details?

3) Authors claim DVA “reduces residual bias by more than 50%,” but “bias” here is defined informally as the gap between the learned posterior predictive distribution and the “true” posterior predictive distribution; in practice you just show a lower validation negative log-likelihood curve and assert that variance is negligible. There is no explicit measurement of posterior predictive bias, and no decomposition of NLL. Will you either justify formally that your NLL reduction corresponds to a 50% reduction in posterior predictive bias, or restate the claim in terms of “lower validation NLL” instead of “lower bias”?

4) There is a conclusion that attention choice (Vanilla vs DVA) matters more than the backbone (Transformer vs CNN), based on similar performance between CNN+DVA and Transformer+DVA and large gains over Vanilla Attention. But the CNN and Transformer models differ by orders of magnitude in parameter count and were tuned with different search spaces. This claim either requires more evidence or needs to be adjusted.

### Questions
Please refer to the weaknesses

### Soundness
2

### Presentation
2

### Contribution
3

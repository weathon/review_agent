# Toward Principled Flexible Scaling for Self-Gated Neural Activation

- Avg Score: 6.67
- Decision: Accept (Poster)
- Scores: 6, 6, 8

## Abstract
Neural networks necessitate nonlinearities to achieve universal approximability.
Traditional activation functions introduce nonlinearities through rigid feature rectifications.
Recent self-gated variants improve traditional methods in fitting flexibility by incorporating learnable content-aware factors and non-local dependencies, enabling dynamic adjustments to activation curves via adaptive translation and scaling.
While SOTA approaches achieve notable gains in conventional CNN layers, they struggle to enhance Transformer layers, where fine-grained context is inherently modeled, severely reducing the effectiveness of non-local dependencies leveraged in activation processes.
We refer to this critical yet unexplored challenge as the **non-local tension** of activation.
Drawing on a decision-making perspective, we systematically analyze the origins of the non-local tension problem and explore the initial solution to foster a more discriminative and generalizable neural activation methodology.
This is achieved by rethinking how non-local cues are encoded and transformed into adaptive scaling coefficients, which in turn recalibrate the contributions of features to filter updates through neural activation.
Grounded in these insights, we present **FleS**, a novel self-gated activation model for discriminative pattern recognition.
Extensive experiments on various popular benchmarks validate our interpretable methodology for improving neural activation modeling.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel self-gated activation function, FleS (Flexible Scaling for Self-gated activation), aimed at addressing the Non-Local Tension (NLT) issue in Transformer architectures. From a decision-theoretic perspective, the authors argue that conventional activation functions (e.g., GELU, SiLU) exhibit convergence limitations when handling high-response features, reducing the efficiency of non-local information utilization. FleS adaptively adjusts the activation function’s boundary and steepness via dynamic vertical and horizontal scaling factors (κ_ve, κ_ho). Experiments on ImageNet, CIFAR-100, and COCO benchmarks demonstrate significant performance improvements. The work combines solid theoretical analysis, novel design, and comprehensive experiments, providing a new interpretable perspective for activation function modeling.

### Strengths
# **Strengths**

1. **Originality and Significance:** The paper identifies and clearly defines a novel and important problem, *Non-Local Tension (NLT)*. It explains why many advanced activation functions that perform well on CNNs fail to provide similar improvements in Transformer architectures. This is not only a technical innovation but also a conceptual contribution, offering a new perspective for understanding and improving activation mechanisms in modern neural networks.

2. **Theoretical Quality:** The paper provides solid theoretical support for both the problem and the proposed solution. The authors construct a clear logical chain: from NLT to *Trivially Discernible Gating Weights (TDGW)*, and then to the root cause, *Convergence Limitation (CL)*. The analysis is rigorous, accompanied by intuitive explanations and formal theorems. FleS is tightly designed around this theory, using *effective average response* (considering only positive responses) to generate scaling factors, which is both a clever and theoretically justified heuristic.

3. **Comprehensive Experimental Validation:** The experiments are thorough and well-designed, covering multiple tasks such as image classification, object detection, and natural language processing. Various network architectures are evaluated, including Swin Transformer, PoolFormer, and ResNet. FleS consistently demonstrates significant performance gains across all settings, strongly supporting its effectiveness, generalization, and robustness.

4. **Clarity:** The paper is well-written, clearly structured, and logically coherent. Each part—from problem formulation, theoretical analysis, method design, to experimental validation—is clearly presented. Figures (e.g., Figures 1 and 2) help readers grasp core concepts and understand FleS’s operational mechanism, making complex theory and methodology accessible.

### Weaknesses
# **Weaknesses**

1. **Lack of a Unified Analysis Framework:** The paper mainly compares FleS with GELU, Meta-ACON, and other methods. Explaining the relationship between these approaches and FleS from a unified theoretical perspective would strengthen the academic rigor and interpretability of the work.

2. **Complexity and Computational Overhead:** The practical version of FleS introduces an additional MLP, increasing the parameter count (e.g., Swin-Min from 11.9M to 13.8M), approximately a 10% increase that is not entirely negligible. The paper does not report inference speed (FPS); it is recommended to quantify the trade-off between computational overhead and performance improvement.

3. **Hyperparameter Sensitivity:** FleS introduces new hyperparameters, such as the MLP channel reduction rate and the neighborhood size (9×15) for computing statistical measures. These settings are somewhat empirical, and further analysis is needed to understand the sensitivity of model performance to these parameters.

4. **Insufficient Elaboration of the Decision-Theoretic Perspective:** Although the paper claims to be inspired by decision theory, the connection between the theoretical principles and the design of FleS remains high-level. Clearly specifying which decision-theoretic principles directly guided the FleS design would strengthen the motivation and theoretical foundation.

### Questions
# **Questions**

1. **Interaction with Normalization Layers:**  
   FleS does not consider the effects of BN or LN, but these normalization layers are widely used in modern networks. Could LN, which normalizes along the channel dimension, interfere with FleS’s computation of channel-wise statistics?

2. **Scope of Non-Local Tension (NLT):**  
   FleS is also effective on CNNs such as ResNet. Does this indicate that the NLT problem exists in modern CNNs as well, or is the observed performance improvement mainly due to the general benefits of the adaptive scaling mechanism?

3. **Implementation Details:**  
   In the COCO experiments, how was the 9×15 neighborhood size chosen? Were other sizes tested? For NLP tasks, why do FleS-NLP and FleS-SeqGate use a *token-level indicator* and *depthwise separable 1D convolution*? Could the authors provide intuition or rationale for these design choices?

4. **Applicability to Large Models:**  
   Are there any numerical stability or training bottlenecks when applying FleS to large models such as ViT-L, LLaMA, or T5?

5. **Version Consistency:**  
   The paper presents multiple FleS variants (FleS-Proto, FleS, FleS-NLP, FleS-SeqGate). Is it possible to provide a unified version that supports multiple tasks without task-specific adaptations?

### Soundness
4

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
This paper introduces a new gating mechanism. It builds on the intuition that having a fixed gating function can be problematic: If all features are salient, then a classical gating function like GELU will saturate and not meaningfully discriminate between these features. To address this, the authors introduce a new gating function which works as follows:

First, for each channel they calculate the mean of the positive features. This is a measure of how likely this feature is going to saturate the activation function. Then, they feed these statistics to two small MLPs. These MLPs provide two scaling factors (of the inputs and the outputs).

The authors argue that this is particularly important for transformer networks, since the attention mechanism in transformers is likely to lead to having many salient activations within a channel. The reason the authors use an MLP rather than deriving scaling factors directly from the inputs is because it is (1) not possible to derive class-specific statistics at test time when the class is unknown, and (2) in shuffled batches the amount of information per-class can be very noisy. Hence, an MLP is a more appropriate way of estimating appropriate scaling factors given the small amount of noisy information found in a single batch.

The empirical results in the paper are very encouraging.

### Strengths
* Strong empirical results
* A very flexible method that applies to many networks/architectures
* Robust method that seems relatively insensitive to hyperparameters

### Weaknesses
* A bit heuristic (e.g. using positive-only means)
* The connection between the theory and the practical algorithm is tenuous (given the use of MLPs in the final algorithm). Although it is interesting to see how the algorithm was motivated by the authors, it does end up feeling a lot like a post-hoc justification. I would prefer it if the authors approach their work as purely experimental and use this space in the paper for more exhaustive empirical validation.
* Subpar presentation: I found the text needlessly filled with jargon, non-standard terminology and abbreviations, drawing spurious connections to other theories, too densely written, etc.
  * For example, abbreviating "activation" to "Act" and referring to pre-activations as "projected responses" really doesn't help with readability.
  * Then there is the list of newly introduced terms and accompanying abbreviations: non-local tension (NLT), convergence limitation (CL), trivially discriminative gating weights (TDGW), etc.
  * Connections to decision-making and neuronal stimulus-response mechanisms.
  * For example: Figure 1 should be a clear explanation of the problem this paper tries to tackle, in a way that readers can grok it after just reading the abstract. Instead, readers are presented with the following sentence, which is barely comprehensible (and contains several grammar mistakes): "[...] the origin of the NLT and the key insights behind FleS shows how CL triggers TDGW problem, which in turn neutralizes the influence of external non-local cues through Act. and show two qualitative insights into addressing non-local tension: vertical and horizontal dynamic scaling strategies."

### Questions
* Did you run any ablation studies over the MLP? Does a single linear layer suffice or is it essential to have non-linearities in the MLP?

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper identifies a limitation in self-gated activation, which is argued to be the reason for limited effectiveness of self-gated activation in transformers due to saturation of gating components. This introduces what the authors call convergence limitation specifically in high-importance features wrt a filter where the difference between the importance become negligible and result in the tendency to lose discriminability wrt contributions of features. In transformers, this collapse of gating discriminability causes activation to neutralize contextual cues the architecture tries to capture.

The authors proposed flexible scaling in self-gated activation using horizontal and vertical dynamic scaling. Horizontal scaling shifts or stretches the gating curve to avoid saturation and vertical scaling increases the discriminability by increasing the range of gating values. The scaling coefficients are conditioned on channel-wise statistics of positively contributing features. In practice, the proposed activation, called FleS, computes per-channel effective responses, feeds them through lightweight MLPs, and outputs the two scaling coeffs. The authors benchmarked the proposed activation across ImageNet, CIFAR-100, long-tailed recognition, COCO detection, and GLUE where FleS consistently outperformed SOTA activations. Particularly notable are results in Swin-Transformer models.

### Strengths
1. Clear identification of an important issue 

The underlying cause of non-local tension problem was clearly discussed, something that previous works have not articulated. 

2. The logical framing of the problem, its cause and the proposed approach

The paper provides an intuitive interpretation of activations as "importance modulators". Then clearly identifies the harm of saturation in self-gating activations and draws a logical connection from convergence limitation, trivially discriminative gating weights phenomenon, and non-local tension problem. 

3. Practical design of activation with strong empirical results across different models

FleS is simple, seems to be lightweight, and can easily be dropped into modern architectures. Performance improvements especially in Swin-Micro and Swin-T are substantial. The improvements in experiments in Metaformers, CNNs, detection backbones, and long-tailed classification are promising. This broad applicability suggests that the identified problem is real and not confined to a narrow architecture.

### Weaknesses
1. Some theoretical claims rely on partially informal assumptions with more room for quantification/formalization:

Although the results generally support the narrative, but some justification or quantification can show if attention-enhanced features regularly fall into the saturation regime. Also, the explanation for why positive-only feature responses should dominate importance is intuitive, but remains heuristic; the decision-theoretic interpretation could be formalized more rigorously.

2. Insufficient analysis of optimization stability and dynamics

The paper mentions initializing \gamma values but doesn't quantify sensitivity to initialization. Given that activations can strongly shape optimization trajectories, this lack of investigation is a methodological gap

3. Dependence on channel-level statistics require more investigation/illustration of failure scenarios
The batch-dependence in channel statistics brings up questions about microbatch regimes, distributed training, highly-multimodal batches. 

4. Discussion of potential failures/limitations or stress-tests:
In addition to gains in performance, it's worth discussion more about limitations and scenarios that the proposed activation may fail to be useful.

### Questions
1. Can the authors quantify how often real Transformer activations fall into the saturation regime?

2. How do $\kappa_h$ $\kappa_o$ evolve during training? Please include training curves of these scalars and variance across layers. 

3. Is FleS stable in small-batch regimes? Also, how does FleS interact with batch normalization.

4. The paper gives an intuition but more follow-ups on why to exclude negative values: What happens in architectures where negative values carry semantic meaning? Have the authors visualized the gradient contributions from negative vs positive responses?

5. Is there any other costs other than Flops to be discussed and compared? This becomes important especially when impact of MLP size is also discussed.

### Soundness
3

### Presentation
4

### Contribution
3

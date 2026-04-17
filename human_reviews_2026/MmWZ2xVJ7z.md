# Biologically Plausible Learning via Bidirectional Spike-Based Distillation

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6, 8

## Abstract
Developing biologically plausible learning algorithms that can achieve performance comparable to error backpropagation remains a longstanding challenge.
Existing approaches often compromise biological plausibility by entirely avoiding the use of spikes for error propagation or relying on both positive and negative learning signals, while the question of how spikes can represent negative values remains unresolved. 
To address these limitations, we introduce Bidirectional Spike-based Distillation (BSD), a novel learning algorithm that jointly trains a feedforward and a backward spiking network.
We formulate learning as a transformation between two spiking representations (i.e., stimulus encoding and concept encoding) so that the feedforward network implements perception and decision-making by mapping stimuli to actions, while the backward network supports memory recall by reconstructing stimuli from concept representations.
Extensive experiments on diverse benchmarks, including image recognition, image generation, and sequential regression, show that BSD achieves performance comparable to networks trained with classical error backpropagation. 
These findings represent a significant step toward biologically grounded, spike-driven learning in neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Bidirectional Spike-based Distillation (BSD), a biologically motivated training recipe for spiking neural networks. BSD jointly trains a bottom-up “perception” SNN and a top-down “recall” SNN, framing learning as the alignment of basal (forward) and apical (feedback) membrane potentials. Hidden layers are optimized with a layer-local, unsigned Relaxed Contrastive (ReCo) loss that increases same-sample similarity and suppresses positive cross-sample correlations, while the top layer uses a standard supervised objective. The method is claimed to satisfy five criteria for biological plausibility (asymmetric weights, layer-local plasticity, non-dual-phase training, spiking neurons, and unsigned signals at hidden layers). Experiments on image classification, sequence regression, and simple autoencoding show performance close to BP on modest benchmarks with MLP/CNN/RNN backbones. Conceptually, the novelty appears moderate: BSD primarily assembles known ideas from local/decoupled learning and feedback/target-prop-style alignment into an all-spiking, two-pathway realization. Moreover, the advertised “top-down memory recall” is instantiated as label-driven feedback reconstruction rather than an addressable memory or retrieval mechanism. Finally, the work lacks validation on modern architectures and large-scale datasets and does not compare against strong local-learning baselines for SNNs, so the contribution is best viewed as an effective integrated recipe rather than a fundamentally new paradigm.

### Strengths
This work presents a well-motivated learning algorithm inspired by the brain's perception-recall mechanisms. The paper clearly defines its contribution by introducing two new criteria for biological plausibility and demonstrating that the proposed BSD algorithm satisfies all five. The joint training of feedforward and backward networks via mutual distillation is an elegant approach to implement local learning. The experimental evaluation covers a diverse set of tasks and network architectures, which supports the algorithm's versatility. The results show that BSD can achieve performance competitive with backpropagation while adhering to stricter biological constraints.

### Weaknesses
The paper positions layer-local losses and decoupled feedback as distinctive, yet these ideas substantially overlap with established families (local errors/greedy layerwise, synthetic gradients, feedback alignment, target/representation propagation; in SNNs, DECOLLE and e-prop), and the batch-correlation alignment mirrors redundancy-reduction/self-correlation objectives (Barlow Twins/VICReg), so the conceptual novelty appears moderate without head-to-head baselines. 

Claim–implementation gaps further temper the contribution: the “top-down memory recall” component is effectively label-driven feedback reconstruction (no addressable memory, retrieval/gating, replay, or partial-cue/label-free recall). 

Scalability is unclear because ReCo forms a B×B cosine matrix per layer (O(B²) compute/memory) with no memory profiling or scalable approximations, leaving open questions about large-batch, streaming/micro-batch, and deeper/wider models. 

External validity is limited to small datasets and shallow backbones (MNIST/SVHN/CIFAR; 6-layer MLP/5-layer CNN), with no results on modern architectures (ResNet) or large-scale data (ImageNet), so it is difficult to judge whether the gains persist at realistic scale or against strong local-learning baselines for SNNs.

---

References 

Deep Supervised Learning Using Local Errors, https://pmc.ncbi.nlm.nih.gov/articles/PMC6127296/

Greedy Layerwise Learning Can Scale to ImageNet, https://proceedings.mlr.press/v97/belilovsky19a/belilovsky19a.pdf

Decoupled Neural Interfaces using Synthetic Gradients, https://arxiv.org/abs/1608.05343

Random synaptic feedback weights support error backpropagation for deep learning, https://www.nature.com/articles/ncomms13276

Training Neural Networks with Local Error Signals, https://www.gatsby.ucl.ac.uk/~pel/tnlectures/papers/local2019.pdf

Difference Target Propagation, https://arxiv.org/abs/1412.7525

Biologically Motivated Algorithms for Propagating Local Target Representations (LRA-E), https://ojs.aaai.org/index.php/AAAI/article/view/4389/4267

Synaptic Plasticity Dynamics for Deep Continuous Local Learning (DECOLLE), https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full

A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop), https://www.nature.com/articles/s41467-020-17236-y

Barlow Twins: Self-Supervised Learning via Redundancy Reduction, https://proceedings.mlr.press/v139/zbontar21a.html

VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning, https://arxiv.org/abs/2105.04906

### Questions
Q1. Can you report a ImageNet-1k experiment (e.g., ResNet-18/50) with accuracy and training cost versus BP and the above baselines?

Q2. How sensitive is BSD to batch size, especially small-B/micro-batch or streaming regimes where the B×B affinity becomes ill-conditioned?

Q3. How do you reconcile C5 with a signed top-layer cross-entropy; is C5 intended only for hidden layers, or can you replace CE with an unsigned alternative and report the impact?

### Soundness
2

### Presentation
3

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
The authors present a layer-wise local learning rule for two-compartment spiking neurons where each layer tries to minimize a Relaxed Contrastive (ReCo) loss between the apical and basal activation. The network is constructed with a bottom-up discriminative and top-down generative stream, where each stream has its own classification/regression task. The authors argue that in this network, just optimizing these within-layer errors inside the spiking neurons results in competitive performance on various tasks.

### Strengths
The experiments are extensive and well done. The loss as described seems to be a novel application for spiking neural networks. If correct (see below), the stability of the local learning approach is amazing and would be a big step in general towards local learning.

### Weaknesses
I have a number of smaller issues, and one main one:

while the ReCo loss has been presented before, I am not aware of its use in a setting where there is only layer-wise learning. Ie in Lin et al 2023, this loss seems to be backpropagated through a deep network. It is not obvious why this loss still works when indeed the errors are not backpropagated from the output layer. As it is now, the approach seems to mimic a layer-wise ELBO loss, with some nudging of only the final layer from errors calculated from outputs. The issue is that such layer-wise ELBO losses typically collapse and need measure to learn properly. I do not see either the mathematical argument why the proposed approach should work or the ablation study that demonstrates how this works compared to not detaching. Checking the code, I am also not convinced that necessarily the detach() operation is applied correctly.  I would very much like to see convincing arguments why the described approach works, and that it works correctly. 

Smaller issues:

The text is many places is not particularly clear. Eg. where the loss is described in 4.3 (l234), it is not immediately clear that the ReCo loss is being explained. 

Figures 1 and 3 dont print well.

The term "distillation" is prominently mentioned but it is not explained what is meant by this term.  Similarly for "embeddings" 

Layer-wise learning is not necessarily local, it should be explained in detail why the learning rule is neuron-local or instead layer-local.

"predictive coding" is a closely related approach and should be discussed. It is also shown in Table 1 but not cited, so the particular predictive coding approach is unclear. 

the approach claims biological plausibility, but the ReCo contrastive loss relies on the batch to provide negative samples, which is not a quantity locally known by the neuron. 

I do not agree with all stated biological constraints: as is well noted in the literature, symmetric forward and backward weights can arise naturally provided that the same local learning is present at forward and backward site, and weight-decay is applied (easy to see). 

There is a misunderstanding of how the brain seems to employ global error signals: first, multiple neuromodulatory signals are thought to be quite a-specific, like dopamine being broadcasted widely and non-targeted. Moreover, to the degree that dopamine signals a global reward prediction error, negative errors are believed to be encoded as a decrease from a steady baseline, thus resolving the issue with error-signs.

typo l054: "spiking trains" ??

### Questions
What is the effect of having spikes? In 5.3 it is noted that spikes degrade accuracy compared to ANNs. For the proposed architecture, it would seem the proposed approach would also work for identically structured non-spiking RNNs. Does this improve performance? 

How does the approach scale? All examples are relatively small/shallow  networks, whereas it is known that similar ELBO-like losses tend to not scale well. Can you comment Deeper networks?

How does ReCo scale with batch size?

Can you demonstrate why layer-wise ReCo works as well as it does, compared to e.g. non detached ReCo?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes the Bidirectional Spike-based Distillation (BSD) algorithm. Inspired by the brain’s bidirectional perception-recall mechanism, BSD jointly trains a feedforward spiking network (for stimulus-to-concept perception and decision-making) and a backward spiking network (for concept-to-stimulus memory reconstruction), framing learning as a transformation between stimulus encoding and concept encoding. Through experiments across multiple tasks (image classification, image generation, sequential regression) and network architectures (MLPs, CNNs, RNNs, autoencoders), BSD is verified to achieve performance comparable to classical error backpropagation while satisfying all five biological plausibility criteria.

### Strengths
1. BSD is the an algorithm to simultaneously meet five core biological plausibility criteria, resolving conflicts between existing methods and neurobiological principles in aspects such as spike signal utilization and error signal type, thus providing an effective framework for biologically grounded spike-driven learning.

2. Experiments cover multiple tasks (e.g., image classification on MNIST/CIFAR-10, image generation evaluated via FID, sequential regression including text prediction and traffic forecasting) and various network architectures, verifying BSD’s strong generalization ability and task adaptability.

3. The algorithm design is closely aligned with the brain’s bidirectional mechanism of bottom-up perception and top-down recall, utilizing the basal and apical dendrites of pyramidal neurons to process feedforward and feedback signals respectively, resulting in a theoretical foundation highly consistent with neuroscientific observations.

### Weaknesses
1. It would be beneficial to investigate the influence of different timesteps on the BSD network's performance through additional experiments.

2. Further discussions on the impact of batch normalization on the convergence and overall performance of the BSD algorithm could be included.

3. Considering the BSD algorithm's dual forward and backward branches, it presumably requires a larger amount of GPU memory, which could be a relevant consideration for practical applications.

### Questions
1. The paper mentions that the number of time steps is set to 4 for MLP, CNN, and RNN tasks (e.g., Section E.2.2, E.2.3) and 8 for autoencoder-based image generation tasks (Section E.2.4), but no analysis of the impact of different time step values on performance is provided. Do you plan to supplement experiments with varying time steps (e.g., 2, 6, 10) to explore how time step adjustments affect BSD’s performance across different tasks (image classification, image generation, sequential regression)?

2. Batch normalization is applied in BSD-CNNs (after each convolutional layer, Section E.2.2) and BSD-RNNs (for the hidden layer in Pems-bay, Section E.2.3) to stabilize spiking activity, but the paper lacks quantitative and qualitative discussions on its specific impact. Could you provide experimental results comparing the convergence speed (e.g., number of epochs to reach stable performance) and final task performance (e.g., classification accuracy, FID, MSE) of BSD with and without batch normalization?

3. The BSD algorithm adopts dual forward (Type 1 neurons) and backward (Type 2 neurons) branches with independent weight matrices (W for forward, Θ for backward, Section 4.2), which may increase GPU memory consumption. Have you quantified the GPU memory usage of BSD compared to classical error backpropagation (on ANNs/SNNs) and other biologically plausible algorithms (e.g., DLL, CCL) under the same network architecture and batch size?

### Soundness
3

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
4

### Summary
This paper introduces Bidirectional Spike-Based Distillation (BSD), a biologically plausible learning algorithm inspired by the brain's bidirectional perception-recall architecture. BSD transforms spiking representations between stimulus encoding (feedforward path for perception) and concept encoding (feedback path for recall), satisfying five biological plausibility criteria (e.g., asymmetric weights, local plasticity). Experiments on image classification, sequential regression, and generation tasks demonstrate that BSD achieves performance comparable to error backpropagation while maintaining stronger neurobiological fidelity.

### Strengths
1. BSD fulfills all five proposed biological criteria, i.e., asymmetric weights, local error signals, non-two-stage learning, spiking neuron models, and unsigned error encoding, making it more aligned with neurobiological principles than backpropagation-based methods.   

2. BSD achieves accuracy close to backpropagation on diverse benchmarks and robustly handles sequential regression and image generation, proving practicality without sacrificing biological grounding.

### Weaknesses
1. Discrete spike-based communication reduces information capacity compared to continuous activations, leading to performance degradation in SNNs across all tasks.   

2. The description of the spike generation and information encoding lacks depth. While the LIF neuron dynamics are provided, the paper does not specify the exact strategy for converting inputs (like images) and targets into the initial spike trains (s1 and $\hat{s}$) for the feedforward and backward paths. The choice of encoding scheme (e.g., rate coding, temporal coding) can significantly impact performance, and its omission is a notable gap. Furthermore, the distillation process between the two pathways is described at a high level as feature alignment using the ReCo loss, but the precise mechanisms of how the error signals from this loss guide the updates of the distinct weight matrices (W and $\theta$) need a more detailed explanation, potentially including a mathematical derivation.

3. The selection of the Relaxed Contrastive (ReCo) loss over other alternatives is justified by its unsigned nature and performance in ablations. However, a more rigorous theoretical or empirical justification for why it is particularly suited for aligning voltage signals in spiking neurons, as opposed to other contrastive losses, would strengthen the argument. Similarly, the choice of specific hyperparameters, such as the penalty weight $\lambda$ = 0.6 and the firing thresholds for different neuron types and architectures, appears to be determined empirically. The paper would benefit from discussing the sensitivity of the results to these choices or providing a principled rationale for them.

4. The paper introduces two new biological plausibility criteria (C4 and C5) and claims BSD satisfies all five. However, the argument for satisfying C2 (local synaptic plasticity) is somewhat superficial. It states that error computation is localized using the ReCo loss, but a more detailed analysis is needed to explain how the weight updates for W and $\theta$ rely only on locally available information (e.g., pre- and post-synaptic spikes and local voltages) without any implicit non-local dependencies. A detailed diagram or step-by-step explanation of the local update rule for a single synapse would make this claim more credible.

5.  The experiments demonstrate performance on standard benchmarks but do not thoroughly test the robustness of the BSD-trained models. For instance, how do these models perform on corrupted data, out-of-distribution examples, or under varying noise conditions? Including such robustness evaluations would be crucial for claiming the practical viability of the algorithm. Furthermore, the scalability of BSD is asserted but not conclusively proven. Testing on a larger-scale, more complex dataset (e.g., ImageNet) would provide much stronger evidence for its scalability beyond relatively small datasets like CIFAR-100.

6. A key claim of the paper is that BSD is more biologically plausible. However, the experiments primarily focus on task performance metrics (accuracy, FID, MSE). There is a lack of direct analysis and quantification of the purported biological properties themselves. For example, the paper could analyze the degree of weight symmetry/alignment between the forward (W) and backward ($\theta$) pathways during training. Figure 7 shows this alignment is low for most layers, which supports adherence to C1, but a more quantitative discussion of this property across different tasks and architectures would be valuable. Similarly, the energy efficiency of the spiking models, a major claimed advantage of SNNs, is not measured or compared to analog networks.

7. In the image generation task, the qualitative results (Figure 2) and FID scores are presented. However, the FID scores for BSD are generally worse than those for BP-trained models. The paper states the quality is comparable, but a more critical discussion of the performance gap and the specific challenges BSD faces in generative tasks (e.g., possibly due to the discrete nature of spikes) would be more balanced. The visual samples in Figure 2 also lack a detailed description of what specific shortcomings (e.g., blurriness, lack of detail) are observed in the BSD-generated images compared to the BP baseline.

### Questions
See Weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces Bidirectional Spike-based Distillation (BSD), a novel biologically plausible learning algorithm for neural networks. The work addresses a fundamental challenge in neural network research: developing learning methods that satisfy biological plausibility criteria while achieving competitive performance with backpropagation. BSD is inspired by the brain's bidirectional architecture integrating bottom-up sensory perception with top-down memory recall. The algorithm jointly trains feedforward and backward spiking networks, where the feedforward pathway maps stimuli to conceptual representations (analogous to perception and decision-making), while the backward pathway reconstructs stimuli from concepts (analogous to memory recall). The authors demonstrate that BSD satisfies five biological plausibility criteria: asymmetric forward and backward weights, local synaptic plasticity, single phase learning (eliminating sequential forward-backward passes), spiking neurons, and unsigned error signals. They validate the algorithm across diverse architectures and tasks.

### Strengths
**Biological motivation and plausibility:** The paper draws inspiration from the neuroscience principle of bidirectional processes that integrate bottom-up sensory perception with top-down memory recall. BSD implements this concept with high fidelity and satisfies five relevant criteria of biological plausibility, extending beyond the three criteria proposed by prior work.

**Comprehensive experimental validation:** The authors test the algorithm on multiple architectures (MLPs, CNNs, RNNs, autoencoders) and diverse datasets spanning image classification (MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100), text prediction and time-series forecasting (Harry Potter, Electricity, Metr-la, Pems-bay), and image generation tasks, demonstrating the generalizability of the approach.

**Novel algorithmic contributions:** The integration of bidirectional spiking networks trained via mutual distillation represents a unique approach to biologically plausible learning, distinguishing it from existing methods that typically compromise on some biological plausibility criteria. Moreover, BSD achieves performance comparable to backpropagation on all introduced tasks.

**Ablation study:** The paper includes an ablation experiment examining the impact of different loss functions (MSE, InfoNCE) and demonstrates that ReCo loss consistently outperforms alternatives.

### Weaknesses
**Limited discussion of practical implications and broader impact:** The authors do not adequately address the practical implications of their work. Key questions remain unanswered:
- Will BSD be adopted by the machine learning community, and if so, for what primary applications?
- Will the code be released as an integrated Python library compatible with current frameworks (PyTorch, JAX)?
- Alternatively, will BSD primarily serve the neuroscience community for computational experiments addressing neuroscience-related problems?

**Insufficient computational performance analysis:** Performance metrics including time complexity, memory consumption, and GPU usage should be clearly stated and compared to backpropagation in the main text.

**Minor:**
- In the code, clear instructions for reproducing the experimental results are not given.
- In Section 4.2, the notation is difficult to follow and contains apparent errors. Specifically, $\hat{v}'$ is never defined, while $v'$ and $\hat{v}$ are defined but never used in equations. The same symbol $v'$ is used to define Type 1 neurons in Equation 4 and Type 2 neurons in Equation 5, and vice versa for symbol $\hat{v}$. Additionally, in Section 4.3 (line 239), matrices of membrane voltages are defined but never used in subsequent equations.
- The application of FFT decomposition to input images and membrane voltages is mentioned but not adequately explained. This technique appears important for the method's performance on generation tasks, yet it receives minimal emphasis and should be expanded at least in the appendix.
- Font sizes in figures (especially Figure 1c and Figure 3) are excessively small and difficult to read while going through the main text.

### Questions
See weaknesses. Moreover, could you expand the observation that "The network attains its best performance at a firing threshold of 0.6, suggesting that thresholds that are either too low or too high impair effective learning" (line 461) with a possible explanation for this phenomenon?

### Soundness
3

### Presentation
3

### Contribution
3

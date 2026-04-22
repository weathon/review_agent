# Exact Online Learning with Gamma-memory delays for Accurate Feedforward SNNs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6, 4

## Abstract
Spiking Neural Networks (SNNs) promise energy-efficient, low-latency AI through sparse, event-driven computation. Neuromorphic hardware can realize this efficiency by exploiting high temporal resolution, as precise spike timing supports compact and sparse information processing. Yet, training SNNs under fine temporal discretization remains a major challenge. In state-of-the-art approaches, spiking neurons are modeled as self-recurrent units, embedded into recurrent networks to maintain state, and trained with BPTT or RTRL variants based on surrogate gradients. We show that these methods scale poorly with temporal resolution, while online approximations methods are inherently unstable. We solve this problem by developing recursive memory structures combined with a linear–nonlinear interpretation of spike-train generation in spiking neurons: the SpikingGamma model. We show that SpikingGamma models support direct error backpropagation without surrogate gradients, can learn fine temporal patterns with minimal spiking in an online manner, and scale feedforward SNNs to complex tasks with competitive accuracy, all while being insensitive to the temporal precision of the model. Our approach offers both an alternative to current recurrent SNNs trained with surrogate gradients, and a direct route for mapping SNNs to neuromorphic hardware.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SpikingGamma, a temporal-coding training framework for SNNs using adaptive recursive memory and sigma-delta spike generation. The method supports direct gradient backpropagation without surrogate gradients, eliminates neuron self-recurrency, enables online learning, and achieves competitive performance on DVS Gesture, SHD, and SSC datasets with significantly improved scalability under high temporal resolution. The authors also demonstrate precise delay learning and coincidence detection tasks to support biological interpretability.

### Strengths
1. Temporal coding capability. Demonstrated capability in: precise spike timing learning, delay-based feature detection, biologically plausible mechanisms similar to auditory localization.

3. Scalability to high temporal resolution. Maintains accuracy while other online training methods collapse.

4. Strong experimental results on multiple neuromorphic datasets. Particularly SHD/SSC, where online methods typically underperform.

### Weaknesses
1. Novelty positioning is insufficiently clear. The proposed model largely builds on previously known components (Gamma-model, temporal coding, fractional predictive mechanisms), but the exact conceptual innovation versus prior models remains fuzzy.

2. Limited comparison with modern direct-training SNN baselines. Experiments omit:
   * Advanced temporal-coded SNNs.
   * State-of-the-art surrogate gradient-based deep SNNs.

3. Models beyond small FC/CNN architectures remain untested. No exploration of:
   * deeper networks (10+ layers).
   * spiking Transformer/general sequence architectures.

### Questions
1. Where is the fundamental theoretical novelty compared to Fractionally Predictive SNNs (Bohte 2011) or Gamma models (De Vries 1992)?

2. How does the model behave without adaptive thresholding? This is a core contribution in paper [3], which is also about temporally coded SNNs.

3. The proposed method is only validated on event-driven benchmarks. It remains unclear whether the model can be applied to static datasets (e.g., CIFAR-10, ImageNet).

4. Missing comparisons with state-of-the-art temporal-coding SNNs[1~3]. Although the paper emphasizes precise spike-timing learning capabilities, the comparisons mainly involve online training or BPTT approximation methods, not true timing-based SNN baselines.

[1] Training spiking neural networks with event-driven backpropagation. NeurIPS 2022.

[2] Exploring loss functions for time-based training strategy in spiking neural networks. NeurIPS 2023.

[3] Temporal spike sequence learning via backpropagation for deep spiking neural networks. NeurIPS 2020.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors of this paper present a new approach/or spiking model called SpikingGamma, which they claim is able to produce better results than the standard spiking models trained with BPTT.

### Strengths
**Pros**:

- Looking at new spiking models is an interesting direction, and if successful, can yield improved results. 
- The paper is easy to follow.

### Weaknesses
**Cons:**

- In the abstract and the main body, the authors position their work as an alternative approach to BPTT with surrogate gradients. This is simply a misstatement since the proposed SpikingGamma is a modified spiking model as opposed to a new training method.

- SpikingGamma uses a ReLu activation - this is why surrogate gradients are not necessary; however, this also means the model is less biologically plausible.

- SpikingGamma integrates ideas from several earlier works: the Gamma-model (De Vries & Principe, 1992), the Temporal Kernel RNN (Sutskever & Hinton, 2010) and the Fractionally Predictive SNN (Bohte, 2011; Rombouts & Bohte, 2010). Not clear what new innovations this work presents.
 
- The proposed model is a lot more complex than the standard LIF model; this complexity needs to be justified. 

- While being motivated by the earlier Gamma-model, the authors don't motivate the adoption of sigma-delta spike coding and provide intuition about why this improves the performance at the network level. 

- The experimental validation is inadequate; only very small datasets have been used (no Imagenet level benchmarks). The demonstrated improvement is not significant, particularly given the significant amount of increase in the model complexity.

### Questions
- In (2) and elsewhere, does superscript k indicate time step?

- In (5), how is $\hat{y}^k$ initialized at the first time step (t = 0)?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper discusses the issue of approximate gradients when training SNNs using STBP, and introduces the Gamma model to circumvent approximate gradients during training. The effectiveness of this method has been verified on some small datasets and network architectures.

### Strengths
This paper exhibits robust logical reasoning, featuring a meticulously organized overview of current methods and an introduction to this paper's motivation. The schematic representations of the methods are visually appealing, and the description of the technique is exhaustive. However, given my limited expertise in this domain, I am challenged to evaluate the novelty of the method.

### Weaknesses
The method presented in this paper is comprehensive. However, I have several inquiries regarding the experimental section.

1. Dataset: The experiments documented in this study predominantly utilize the DVS dataset. It would be beneficial to ascertain whether the efficacy of SpikingGamma is sustained when applied to more common datasets, such as RGB video data.

2. Network Architecture: An observation from Table 1 indicates that the majority of the networks are characterized by a reduced structure, specifically comprising four layers or fewer. A pertinent question arises as to whether SpikingGamma can be effectively adapted for use in networks with an increased number of parameters, such as those that are deeper and wider.

### Questions
I notice that this paper mentions "to reduce the memory footprint of BPTT" in lines 93-94. Does "memory" here refer to the GPU memory required for training? I also notice that OTTT and its successor SLTT [1], as well as some network architecture design-related work, T-RevSNN [2], both can save GPU memory consumed during BPTT training. Could you discuss these methods in more detail?

[1] Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks. ICCV 2023.

[2] High-Performance Temporal Reversible Spiking Neural Networks with $O(L)$ Training Memory and $O(1)$ Inference Cost, ICML 2024.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses the significant challenge of training Spiking Neural Networks (SNNs), particularly the poor scaling of current methods when using the fine temporal discretization required for neuromorphic hardware. State-of-the-art approaches, which treat SNNs as RNNs trained with BPTT/RTRL and Surrogate Gradients (SGs), scale poorly with temporal resolution and can be unstable

### Strengths
The paper directly addresses the inability of current SNN training methods to scale to high temporal resolutions. The most significant result is in Figure 5, which shows that SpikingGamma's performance is stable as the number of timesteps increases, whereas other online methods like FPTT and ES-D-RTRL see their accuracy "deteriorate sharply" or "completely collapse". This robustness is a critical property for real-world neuromorphic applications

The model is feedforward  and does not require unrolling through time for gradient computation (like BPTT). Figure 5 (right) shows that its memory usage is "theoretically constant" with time, a stark contrast to the linear memory growth of BPTT. This makes the method genuinely suitable for online processing and resource-constrained hardware.

### Weaknesses
The experimental evaluation is confined to datasets (SHD, SSC, DVS Gesture) that are relatively small in scale. The paper does not provide evidence of how the SpikingGamma model's performance, stability, and training efficiency would scale to much larger and more complex event-based benchmarks, such as N-ImageNet or UCF-DVS.

The paper's claim of an "online" and "constant memory" learning mechanism (a key advantage over BPTT) requires clarification. The method computes a gradient at each timestep $t$ based on a loss $L(t)$ available at that same step. This is effective for the chosen benchmarks, which allow for dense or accumulated loss signals.

However, it is unclear how this framework applies to tasks with sparse, delayed rewards or a single classification label available only at the end of a long sequence (e.g., $T=3000$). These scenarios are a primary motivation for using SNNs.

 If the loss is only available at the final timestep $T$, does the model need to store the intermediate "bucket" states ($\hat{y}_i^k$) from all preceding timesteps to compute the gradients?

 If so, this would seem to negate the "constant memory" advantage, as it would reintroduce a memory cost that scales linearly with sequence length, similar to BPTT.

 Alternatively, if gradients are computed at each step, accumulated, and applied at the end, how does this fit the definition of "online" learning? This ambiguity regarding the model's update rule and memory footprint in sparse-reward settings is a significant weakness.

### Questions
see weakness below

### Soundness
3

### Presentation
2

### Contribution
3

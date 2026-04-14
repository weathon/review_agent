# SPLR: A Spiking Neural Network for Long-Range Temporal Dependency Learning

- Decision: Reject
- Scores: 5, 3, 8, 5, 5

## Abstract
Spiking Neural Networks (SNNs) offer an efficient framework for processing event-driven data due to their sparse, spike-based communication, making them ideal for real-time tasks. However, their inability to capture long-range dependencies limits their effectiveness in complex temporal modeling. To address this challenge, we present a **SPLR (SPiking Network for Learning Long-range Relations)**, a novel architecture designed to overcome these limitations. The core contribution of SPLR is the **Spike-Aware HiPPO (SA-HiPPO)** mechanism, which adapts the HiPPO framework for discrete, spike-driven inputs, enabling efficient long-range memory retention in event-driven systems. Additionally, SPLR includes a convolutional layer that integrates state-space dynamics to enhance feature extraction while preserving the efficiency of sparse, asynchronous processing. Together, these innovations enable SPLR to model both short- and long-term dependencies effectively, outperforming prior methods on various event-based datasets. Experimental results demonstrate that SPLR achieves superior performance in tasks requiring fine-grained temporal dynamics and long-range memory, establishing it as a scalable and efficient solution for real-time applications such as event-based vision and sensor fusion in neuromorphic computing.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper integrates state-space models (SSMs) and spiking neural networks (SNNs), and proposes the Spiking Network for Learning Long-Range Relations (SPLR) to enhance the ability of SNNs to capture long-range dependencies. Theoretical proofs and experimental results support the performance advantages of SPLR.

### Strengths
1. The integration of SA-HiPPO and SPLR convolution enhances the model's ability to model long-range dependencies.
2. The SPLR model designed in this paper introduces a dendrite-based pooling layer, which further improves the performance using the DH-LIF neuron model.
3. The theoretical and experimental results in this paper confirm the effectiveness of the proposed method.

### Weaknesses
1. The core innovation of this paper is the SPLR convolution layer to enhance the long-range temporal modeling capability of SNNs. However, while this improves the performance of the SNN, the integration of the SSM and the SNN requires significant computational overhead, which counteracts the power consumption advantage of the SNN. In addition, as the authors mention SPLR is difficult to implement in hardware, making the significance of this work seem small.

2. Can the authors clarify the key differences between the DH-LIF model in this paper and the one presented in [1]? How does the DH-LIF used in the Dendrite Attention Layer relate to attention? Is the author's claim of dendrite-based spatio-temporal pooling just spatial pooling after the output of DH-LIF? Where is there temporal pooling?

3. I suggest that the authors compare their method with other optimized spiking neural networks capable of long-range temporal modeling, such as [2,3,4].


[1] Zheng H, Zheng Z, Hu R, et al. Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics. Nature Communications, 2024.

[2] Wang L, Yu Z. Autaptic synaptic circuit enhances spatio-temporal predictive learning of spiking neural networks. ICML, 2024.

[3] Zhang S, Yang Q, Ma C, et al. Tc-lif: A two-compartment spiking neuron model for long-term sequential modelling. AAAI, 2024.

[4] Shen S, Zhao D, Shen G, et al. TIM: An efficient temporal interaction module for spiking transformer. IJCAI, 2024.

### Questions
Please see weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This work proposes a spike SSM method named Spiking Network for Learning Long-Range Relations (SPLR). The proposed SPLR convolutional layer leverages state-space dynamics to enhance feature extraction while retaining the efficiency of sparse, event-based
processing, and incorporates a Spike-Aware HiPPO (SA-HiPPO) matrix that allows SPLR to effectively maintain long-range memory by adapting the HiPPO framework for discrete, spike-driven inputs. The authors tested their method on several datasets, such as Celex HAR, DVS128 Gesture, Sequential CIFAR-10/100

### Strengths
1. This work explores the combination of Spiking Neural Networks (SNN) and State Space Models (SSM), which is an interesting direction. Using state-space methods to improve SNNs' ability to model long-term dependencies holds promise.
2. The experimental comparisons in this work are extensive.

### Weaknesses
This work requires comprehensive improvements, with the main weaknesses outlined as follows.

1. Writing: The writing in this work requires careful and comprehensive improvement, covering the overall organization of the paper, paragraph structure, and numerous details that need refinement. (1) The authors placed the related work section in the supplementary materials and omitted citations to many key works, which can confuse readers. For instance, the authors did not cite relevant papers when HIPPO was first mentioned (in fact, there are almost no citations in paragraphs 3 and 4 of the introduction). (2) The methodology section is not clearly explained. What type of spiking neurons does this work use? How does SPLR integrate with spiking neurons? The authors repeat content introduced in the main text within the supplementary materials. Lines 147-150 reiterate the significance of SPLR, but readers are likely more interested in the methodological details and the rationale behind the proposed approach’s significance. Unfortunately, these critical details are missing. (3) In the theoretical discussion, the authors present several theorems but do not clarify why these are necessary. While sparse properties and FLOPS reduction are mentioned, the evaluation details are not provided. 
(4) The supplementary materials contain excessive repetition and are overly lengthy, making it difficult for readers to stay engaged. (5) Overuse of Abbreviations: The excessive use of abbreviations makes the paper difficult to follow. For instance, I cannot understand why "Spiking Network for Learning Long-Range Relations" is abbreviated as "SPLR"—what does "P" represent here? Is there a difference between "Spiking Network" and "Spiking Neural Networks (SNNs)"? Additionally, what does HIPPO stand for? Shouldn’t the authors explain these abbreviations first? (6) Overstatements: The paper is filled with terms like "spike-driven," "asynchronous," and "real-time." As I understand it, "spike-driven" implies a purely additive network[2], yet the pink section in Figure 1 seems unable to achieve this. Regarding "asynchronous," the authors’ explanation in lines 92-95 is too brief, making it difficult to discern what kind of preprocessing the network applies to the data.

2. Motivation. The authors repeatedly state that the proposed SPLE can address the challenge of modeling both short- and long-term dependencies in SNNs. However, they fail to analyze why SNNs have limitations in this area and why their proposed method can solve this issue. For instance, this is mentioned in lines 58-67, 147-149, and 1846-1850 without providing the necessary analysis.

3. Innovation.The originality of this work is limited. The dendrite modeling directly uses DH-LIF, and the SSM modeling is simply an SNN combined with HIPPO. I did not observe any standout contributions in this approach.

4. Experiments. The datasets chosen by the authors, DVS128 Gesture and Sequential CIFAR-10/100, do not effectively test the model's ability to handle long-range dependencies. The authors could consider more challenging datasets, such as LRA. Additionally, the fact that no one in the SNN field has addressed the Celex HAR dataset does not imply that SNNs cannot handle it. The authors have not even provided a complete description of the size and scope of the Celex HAR dataset. If the authors aim to compare with other SNNs on challenging DVS datasets, they might try HAR-DVS[1]. Furthermore, the authors overlook comparisons with many recent SOTA methods on the Gesture dataset, where SNN performance has already surpassed 99.5%[2].

---
[1] Hardvs: Revisiting human activity recognition with dynamic vision sensors. In AAAI 2024.
[2] Spike-driven transformer. In NeurIPS 2023.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper presents a Spiking Network for Learning Long-Range Relations (SPLR). The proposed SPLR model comprises the dendrite attention layer, the Spike-Aware HiPPO (SA-HiPPO) layer, and the SPLR convolution layer. These modules enhance the long-range temporal dependency learning capability of SPLR. Experimental results demonstrate that SPLR outperforms prior methods in tasks requiring both fine-grained temporal dynamics and the retention of long-range dependencies.

### Strengths
1. This paper is well-written and technically solid. Each module in SPLR is introduced in detail and highlighted in different colors. This paper presents a detailed theoretical analysis, including the long-range dependency capability and stability of SPLR.
2. The proposed SPLR model achieves competitive accuracy with less computational overhead than other state-of-the-art models on the Celex-HAR dataset.

### Weaknesses
The proposed SPLR model incorporates several non-spike operations, including the NPLR decomposition and FFT convolution. It makes SPLR a hybrid architecture instead of a pure spiking neural network. The hybrid nature may compromise its hardware compatibility and make it difficult to deploy on neuromorphic hardware.

### Questions
1. This paper only compares FLOPs vs. accuracy between the proposed SPLR and other models. Does SPLR have an advantage over other methods in terms of inference latency?
2. The ablation studies only examine the effects of removing the dendrite attention layer and replacing SA-HiPPO with LIF. What if we replace NPLR decomposition and FFT convolution with standard convolution?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This work introduces SPLR (Spiking Network for Learning Long-Range Relations), designed to efficiently capture long-range temporal dependencies while maintaining the hallmark efficiency of spike-driven architectures. SPLR integrates a state-space convolutional layer and a Spike-Aware HiPPO (SA-HiPPO) layer, addressing the limitations of conventional SNNs in complex temporal modeling. The SPLR convolutional layer leverages state-space dynamics to enhance feature extraction, capturing spatial and temporal complexities in event-driven data while preserving the efficiency of sparse spike-driven processing. The SA-HiPPO layer adapts the HiPPO framework to spike-based formats, enabling efficient long-term memory retention. Through dendrite-based spatiotemporal pooling and FFT-based convolution techniques, SPLR demonstrates scalability when processing high-resolution event streams and outperforms traditional methods across various event-driven tasks.

### Strengths
1. The authors propose SPLR which effectively captures long-range temporal dependencies, addressing limitations in traditional SNNs and enhancing temporal modeling capabilities for complex event-driven tasks.
2. The experiments show good results. SPLR achieves both computational efficiency and scalability.

### Weaknesses
The primary weakness of this paper lies in its writing, which significantly hinders clarity and understanding. A reorganization is recommended to improve readability and logical flow.

1. Writing and Structure. For instance, Section 2 presents the SPLR components sequentially but lacks an overview that connects each part to the overall model structure, making it difficult for readers to understand how the parts interact. Section 3 is dense with theoretical content and proofs but does not clearly convey the main ideas, making it hard to follow the section’s intended focus.

2. Lack of Citations. The paper frequently omits citations in crucial areas. For example, although modifications to the HiPPO framework are proposed, no supporting references are provided. Furthermore, DH-LIF is introduced without citation, and the reference for this component is missing from the bibliography, weakening the academic rigor of the paper.

3. Confusions and Errors. There are several errors and confusions throughout the paper, such as the incorrect abbreviation of the Spiking State-Space Model as SPLR in line 855. Such errors further impact the readability and precision of the work.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The manuscript introduces SPLR, a spiking neural network model designed to capture long-range temporal relationships by integrating state-space dynamics with spiking neuron models and augmenting the HiPPO framework to handle spike-driven inputs. The proposed model reportedly achieves high performance comparable to other models on event-based datasets.

### Strengths
The manuscript presents an innovative approach by augmenting spiking dynamics with state-space model dynamics, potentially enabling spiking models to tackle more challenging tasks that require capturing long-range temporal dependencies. This direction could be of significant interest in broadening the applications of spiking models in complex temporal tasks.

### Weaknesses
While the proposed method is intriguing and addresses the relevant challenge of enabling spiking models to capture long-term dependencies, the manuscript has several critical weaknesses. 

Firstly, the presentation lacks clarity, making it difficult to fully grasp how the method works, interpret the experimental results, or potentially reproduce the findings. Essential details expected in a research paper, such as a discussion of related works, are missing (e.g., [1]). Additionally, fundamental concepts necessary to understand the work are not well-introduced; although state-space models (SSMs) have gained popularity recently, they are not widely understood in machine learning, so a brief overview would be beneficial.

The manuscript also omits essential citations, including the original HiPPO framework, which is central to this work, and does not offer a proper explanation of how it functions. The equations are unclear; while convolutions are frequently mentioned, no equations illustrate how or where convolutions are applied. Variable definitions are sometimes confusing or incomplete; for instance, on line 187, $\Delta t$ is described as the time difference between spikes $i$ and $j$, but it is unclear what $i$ and $j$ refer to in the context of the matrix $F_{ij}$, as it operates over the hidden state rather than directly on spikes.

Regarding the experiments, the manuscript lacks details about the setup, hindering the interpretability of the results. For example, it’s unclear what “Sequential CIFAR-10” entails, such as the sequence length or frame generation process. Similarly, for the DVS Gesture dataset, it's ambiguous whether the processing was done for independent events or if events were accumulated into event frames.

[1] Stan, MI., Rhodes, O. Learning long sequences in spiking neural networks. Sci Rep 14, 21957 (2024). https://doi.org/10.1038/s41598-024-71678-8

### Questions
- How does the dendritic attention layer differ from a current-based (CUBA) leaky integrate-and-fire (LIF) neuron model?
- How are spikes produced between the SPLR convolution layers?
- How are convolutions applied within the proposed model?
- In equation (1), is the variable $u(t)$ a binary vector representing input spikes?
- How does the inclusion of a decay matrix in the HiPPO framework enhance memory retention?
- Could you clarify the setup for the Sequential CIFAR-10 and CIFAR-100 tasks? How are frames sequenced? Similarly, could you elaborate on the experimental setup for the other datasets?
- For clarification, could you specify what spikes $i$ and $j$ refer to in line 187?
- Is the manuscript proposing a new type of spiking neuron, or an entire network architecture?
- Since the manuscript emphasizes improving SNNs' capacity to handle long-term dependencies, could you elaborate on why simple LIF models face challenges with this?

### Soundness
3

### Presentation
2

### Contribution
2

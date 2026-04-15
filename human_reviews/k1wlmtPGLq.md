# TAB: Temporal Accumulated Batch Normalization in Spiking Neural Networks

- Decision: Accept (poster)
- Scores: 6, 8, 8, 6

## Abstract
Spiking Neural Networks (SNNs) are attracting growing interest for their energy-efficient computing when implemented on neuromorphic hardware. However, directly training SNNs, even adopting batch normalization (BN), is highly challenging due to their non-differentiable activation function and the temporally delayed accumulation of outputs over time.
    For SNN training, this temporal accumulation gives rise to Temporal Covariate Shifts (TCS) along the temporal dimension, a phenomenon that would become increasingly pronounced with layer-wise computations across multiple layers and multiple time-steps. 
    In this paper, we introduce TAB (Temporal Accumulated Batch Normalization), a novel SNN batch normalization method that addresses the temporal covariate shift issue by aligning with neuron dynamics (specifically the accumulated membrane potential) and utilizing temporal accumulated statistics for data normalization. 
    Within its framework, TAB effectively encapsulates the historical temporal dependencies that underlie the membrane potential accumulation process, thereby establishing a natural connection between neuron dynamics and TAB batch normalization. 
    Experimental results on CIFAR-10, CIFAR-100, and DVS-CIFAR10 show that our TAB method outperforms other state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper discusses the challenges of training SNNs, such as non-differentiability of the spiking activation function and the temporal nature of data. It proposes a method called Temporal Accumulated Batch Normalization (TAB) which uses temporal accumulated statistics for data normalization to solve a well-established problem called Temporal Covariate Shift, enhancing the efficiency of SNN training. The paper presents experimental results on CIFAR-10, CIFAR-100, and DVS-CIFAR10 datasets, demonstrating the effectiveness of TAB in improving accuracy. The paper's strength lies in its novelty, detailed explanation of the method, and well-presented experimental results.

### Strengths
This paper has the following strengths:

The authors relate the proposed algorithm with the closed-form of the LIF dynamics and give principled insights into what components in SNN should be normalized.

The authors obtain better performance than previous algorithms.

The proposed method only records the past information, which can be some basic support to online training.

### Weaknesses
The paper lacks a detailed analysis of the computational complexity of TAB, which could be crucial for practical applications. Although it mentions that TAB requires more computations than other normalization techniques, it does not provide a detailed cost analysis or the impact of different hyperparameters on computational complexity, making it difficult to evaluate its practicality in real-world applications.

There seems to be a gradual improvement on the SNN batch normalization technique with the suggested method.

### Questions
What is the impact of TAB on energy efficiency, and how does it compare to other methods for training SNNs? While the authors mention that SNNs are attractive for their energy-efficient computing when implemented on neuromorphic hardware, they do not provide a detailed analysis of the impact of TAB on energy efficiency. This is an important question, as it would help to assess the potential impact of TAB on energy efficiency and identify potential areas for future research.




Can you provide a more detailed analysis of the computational complexity of TAB and how it compares to other methods?




Can you provide evidence that TAB solves TCS? I think the illustration of distribution is capable of demonstrating the solution.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In the paper, the authors introduce a Temporal Accumulated Batch Normalization to address the temporal covariate shift issue by aligning with neuron dynamics and utilizing temporal accumulated statistics for data normalization.  They did experiments on CIFAR10, CIFAR100, and DVS-CIFAR10.

### Strengths
Ideas on low-bit precision neural networks are important and should be encouraged.
The paper focuses on temporal Covariate Shifts along the temporal dimension for SNN, which is an important problem in SNN.

### Weaknesses
The authors should provide an ablation study of the TAB, so we can know if the TAB can really increase the accuracy and how much the method improves. 
Recent work all did experiments on ImageNet, while the paper ignore this.
Some references are missing.
[1] GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks.
[2] Reducing Information Loss for Spiking Neural Networks
[3] Surrogate Module Learning: Reduce the Gradient Error Accumulation in Training Spiking Neural Networks

### Questions
Please see weakness.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors point out the challenging issue currently recognized in SNN-based models, which involves non-differentiable activation functions and temporally delayed accumulation of outputs over time. In this manuscript, the authors introduce TAB (Temporal Accumulated Batch Normalization), a novel SNN batch normalization method that addresses the temporal covariate shift issue by aligning with neuron dynamics (specifically the accumulated membrane potential) and utilizing temporal accumulated statistics for data normalization.

### Strengths
1. A very detailed theoretical analysis discusses the challenging problems of directly training SNN models through BN structures.
2. The authors introduce TAB, a novel SNN batch normalization method that addresses the temporal covariate shift issue by aligning with neuron dynamics and utilizing temporal accumulated statistics for data normalization.

### Weaknesses
1. The authors mention the phenomenon of covariance concept drift encountered by SNN models, so why don't they verify it on time-domain related datasets?
2. We can see that the improvement on the CIFAR-10 and CIFAR-100 datasets is very small, even just 0.6%. Can the current approach be regarded as a theoretically feasible strategy?
3. In addition, the authors have not looked at the impact of other network skeletons, such as ResNet34.
4. Another issue reviewer is concerned about is the relationship between TAB and neuron dynamics. Does the exponential nonlinear operation of the SNN-based model on the time coefficient lead to an error in approximating a first-order linear ODE?

Feedback: The author added a large number of revisions in the limited time, I think I have no new concerns.

### Questions
Please see details of weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In Artificial Neural Networks (ANNs), Internal co-variate shift denotes alterations in the input distribution due to preceding layer updates. In Spiking Neural Networks (SNNs), the Temporal Covariate Shift (TCS) problem arises from both previous layer updates and prior time steps, extending along the temporal dimension. Within SNNs,  neurons accumulate incoming spikes in their membrane potential over time and only generate a spike when this accumulation surpasses a threshold, staying inactive otherwise within the current time-step.
Hence, the accumulation of synaptic currents in the membrane potential is affected by temporal dependencies, which could magnify the internal covariate shift as time progresses. The central concept of this study revolves around addressing the challenge of applying batch normalization during SNN training while considering the temporal data dependencies and the issue of temporal covariate shift. TAB (Temporal Accumulated Batch Normalization) is proposed which closely follows neuron dynamics, especially the accumulated membrane potential, enhancing the accuracy of batch statistics. This alignment establishes a natural link between the behavior of neurons and the application of batch normalization in Spiking Neural Networks (SNNs). TAB uses Temporal Accumulated Statistics (dynamically using a moving averaging approach) for data normalization, introducing learnable weights to differentiate the impact of each time-step on the final result. The study also provides a theoretical connection between TAB method and the neural dynamics.

### Strengths
The paper is well written and easy to read. The authors presented an extensive set of experimental results to evaluate the proposed approach. The theoretical connection between the underlying neural dynamics and the propose normalization technique is very well explained and presented.

### Weaknesses
The experimental results does not show a significant improvement over SOTA for example TEBN, I would suggest including a demsar plot to show the improvement also explain fairly why the improvement is very small. Taking into account all the added complexity regarding trainable weights for sequential neurons, etc.

### Questions
You have used trainable weights for the each point in the neurons sequence, however it is not explain well how the weights for a new data can be used. Imagine you see a shift in the data distribution are these weights still valid?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

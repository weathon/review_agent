# Online Stabilization of Spiking Neural Networks

- Avg Score: 7.00
- Decision: Accept (spotlight)
- Scores: 6, 8, 8, 6

## Abstract
Spiking neural networks (SNNs), attributed to the binary, event-driven nature of spikes, possess heightened biological plausibility and enhanced energy efficiency on neuromorphic hardware compared to analog neural networks (ANNs). Mainstream SNN training schemes apply backpropagation-through-time (BPTT) with surrogate gradients to replace the non-differentiable spike emitting process during backpropagation. While achieving competitive performance, the requirement for storing intermediate information at all time-steps incurs higher memory consumption and fails to fulfill the online property crucial to biological brains. 
Our work focuses on online training techniques, aiming for memory efficiency while preserving biological plausibility. 
The limitation of not having access to future information in early time steps in online training has constrained previous efforts to incorporate advantageous modules such as batch normalization. 
To address this problem, we propose Online Spiking Renormalization (OSR) to ensure consistent parameters between testing and training, and Online Threshold Stabilizer (OTS) to stabilize neuron firing rates across time steps. Furthermore, we design a novel online approach to compute the sample mean and variance over time for OSR. Experiments conducted on various datasets demonstrate the proposed method's superior performance among SNN online training algorithms.
Our code is available at https://github.com/zhuyaoyu/SNN-online-normalization.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
When batch normalization (BN) is implemented in spiking neural network (SNN) structures, it is a common practice to compute the statistics across all time steps for running BN. However, this commonly used BN is ill-suited for the online training of SNNs. To enable BN for online training, this paper introduces two strategies called Online Spiking Renormalization (OSR) and Online Threshold Stabilizer (OTS), which preserve the online training property and memory efficiency. Experimental results demonstrate the efficacy of the proposed methods.

### Strengths
1. The proposed tricks are both simple to implement and logically sound, contributing to their practicality and ease of use.
2. The performance is good when compared to online training methods and conventional methods.

### Weaknesses
1. If the reviewer understands the OTS method correctly, the threshold $\theta[t]$ is dynamically adjusted for each sample batch during both the training and inference phases. However, the reviewer thinks that $\theta[t]$ should be precalculated and fixed during the inference stage. Otherwise, the batch size will highly influence the performance. Much worsely, the obtained network cannot be implemented on normal neuromorphic chips because the chips do not support calculating the mean and variance. the reviewer suggests the authors to maintain a running $\theta[t]$ used for inference.


2. This paper could be regarded as an engineering work. Then the reviewer thinks that the ablation experiments are not abundant:
 (i). When OSR is incorporated individually, there appears to be a significant performance degradation. It is important to clarify whether such a phenomenon is typical across various datasets.
 (ii). A simpler approach to integrating BN into the online training regime involves calculating the statistics solely based on data from each time step. In this approach, at every time step during training, the normalized I[t] is computed based on $\mu[t]$ and $\sigma[t]$, as illustrated in eq. 8. Additionally, the running mean and variance are also updated at each time step. This vanilla BN method should be considered as the baseline. It would be valuable to assess whether this method outperforms OSR and whether the combination of the baseline with OTS outperforms the combination of OSR with OTS.
 (iii). In the backward stage, the authors utilize a "double transformation" trick to facilitate more meaningful backpropagation. What if we directly conduct bachprop based on the ``linear transformation''? Is OSR better than that?


Minor:
1. eq.2: $s^{l-1}[t]W^l$ -> $W^ls^{l-1}[t]$, eq.4: $u^{l} [t] (1-s^{l} [t] )$  -> $u^{l}[t] \odot (1-s^{l}[t])$
2. The ``Online Calculation of All-time Mean and Variance'' part appears somewhat trivial. The reviewer thinks that it might be unnecessary to include this information in the list of contributions. The audiences may expect a more significant contribution when such a detail is highlighted.
3. The statement preceding Section 5: ``our OTS mechanism helps our OTS mechanism''.

### Questions
1. The line right before Section 4.3: is there typos in the formula? Should the formula be $...(\theta[1]-\mu[1])/\sigma[1]$? Then does it mean that $\theta[1]$ is fixed all the time?
2. How to implement the proposed method on NF-Resnet-34 which is a normalizer-free architecture? The reviewer knows that the goal of this experiment is to compare the proposed method and OTTT. But still, the proposed method and normalizer-free nets are totally orthogonal.

### Soundness
2 fair

### Presentation
2 fair

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
This paper focuses on the online training of SNNs. It adds the normalization mechanism into online training, which have not yet been fully explored by previous works. It proposes two modules to improve the standard Batch Normalization, named Online Spiking Renormalization and Online Threshold Stabilizer, which ensure consistent parameters and stable neuron firing rates across time steps. The paper demonstrates the effectiveness of the proposed methods on various datasets and shows that they outperform existing state-of-the-art online training methods.

### Strengths
1. The paper presents a novel approach to training spiking neural networks that integrates essential batch normalization into the online training process by introducing online spiking renormalization and online threshold stabilizers to enhance training stability.

2. The paper is well-organized and well-written. Also, the figures and tables are well-designed and provide a clear representation of the results.

3. The proposed method outperforms existing state-of-the-art online training methods.

### Weaknesses
1. Although the online training approaches save the memory cost, the proposed method falls short of BPTT in performance.

### Questions
1. What is the difference between OSR and batch renormalization [1] ?

2. The online calculation of all-time mean and variance is interesting. However, where is it used? Is it a part of OSR?

3. The authors have made an assumption that the membrane potential follows a normal distribution in OTS. Although experiments have shown the effectiveness of OTS+OSR, I am still curious about the real distribution of the membrane potential.

[1]Sergey Ioffe. Batch renormalization: Towards reducing minibatch dependence in batch-normalized models. Advances in neural information processing systems, 30, 2017

I consider increasing my score if the authors can solve my concerns.

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
The paper considers the online setting of SNN, and points out one important mismatch of BN happens in the training and testing stages. The authors proposed one nice solution to solve the issue, and the experimental results support the benefits with the new algorithms. Importantly, in addition to the experimental verification, the authors also provide necessary theoretical analysis to the new algorithm.

### Strengths
1. The paper have a very clear motivation: mismatch of BN happens in the training and testing stages. 
2. The authors proposed one nice solution to solve the issue, and the experimental results support the benefits with the new algorithms. 
3. The propblem of this paper is unsolved in the comminity before. The solution of this paper is interesting and novel. Importantly, it signicantly improve the performance. Considering the important role of BN in ML, I think the signicance and novelty of this work is good.
4. The authors also provide the necessary theoretical analysis to the new algorithm.
5. Presentation is good. Especially, I like figure 1 to intutiively explain the proposed algorithm.

### Weaknesses
The paper is with a nice story, and I especially like the figure 1 to intutiively explain the proposed algorithm. However, I still have some questions or comments below.

1. What is the intuitive reason to have double normalization in (12)
2. I suggest to provide the experimental results to verify the Gaussion assumption.
3. I did not find the calculation form of $\hat{\mu}$ and $\\hat{\sigma}$. I suggest to provide a clear definition of them in the paper.

Minor issue:
1. I did not find  the meaning of m in (8).

### Questions
Please find the comments above

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The work focuses on online training techniques, especially addressing the limitation of not having access to future information in early time steps in online training. This paper tries to incorporate BN into online training, by two new modules, Online Spiking Renormalization (OSR) and Online Threshold Stabilizer (OTS).  
This online training setting will benefit memory consumption. However, there are several problems: 
the presentation of the new idea is not clear at all, it depends on the running mean and running variance, but how to calculate them is not mentioned clearly. we still don't know how to save memory by using the proposed modules. 
The other issue is that in the theoretical parts, authors tend to make unrealistic assumptions. How can the spike train be iid. and expect the weights to be iid. Another thing is that the conclusions of the theorems, why do we need expectation of the average of u[t], and expectation of the average of sigma[t], how can these results help with the main new modules?
For the experiments, there is only one method using the same structure as the new method, shouldn't compared with different network architectures? Shouldn't compare with different BN methods? That is absent in the main paper.

### Strengths
This paper proposes a novel approach to training spiking neural networks that is memory-efficient and biologically plausible. The proposed Online Spiking Renormalization and Online Threshold Stabilizer techniques ensure consistent parameters and stable neuron firing rates across time steps.

### Weaknesses
The main idea should be better explained to make it easier to understand. 
How to compute gradients in the backward stage if we use this forward transformation? The authors asked themselves the question, but did not give clear answers. Please make the core part of the paper clear.

### Questions
This online training setting will benefit memory consumption. However, there are several problems: 
the presentation of the new idea is not clear at all, it depends on the running mean and running variance, but how to calculate them is not mentioned clearly. we still don't know how to save memory by using the proposed modules. 
The other issue is that in the theoretical parts, authors tend to make unrealistic assumptions. How can the spike train be iid. and expect the weights to be iid. Another thing is that the conclusions of the theorems, why do we need expectation of the average of $u[t]$, and expectation of the average of $\sigma[t]$, how can these results help with the main proposed modules?
For the experiments, there is only one method using the same structure as the new method, shouldn't compared with different network architectures? Shouldn't compare with different BN methods? That is absent in the main paper. 
How to calculate $\hat{u}, \hat{\sigma}$? please explain. 
Is $\gamma$, $\beta$ learnable or fixed?
No point of Online Threshold Stabilizer (OTS) to stabilize the firing rate of each layer?
No experiment results on resnet on cifar datasets.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

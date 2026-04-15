# Sparse Spiking Neural Network: Exploiting Heterogeneity in Timescales for Pruning Recurrent SNN

- Decision: Accept (poster)
- Scores: 5, 8, 8, 6

## Abstract
Recurrent Spiking Neural Networks (RSNNs) have emerged as a computationally efficient and brain-inspired machine learning model. The design of sparse RSNNs with fewer neurons and synapses helps reduce the computational complexity of RSNNs. Traditionally, sparse SNNs are obtained by first training a dense and complex SNN for a target task and, next, eliminating neurons with low activity (activity-based pruning) while maintaining task performance. In contrast, this paper presents a task-agnostic methodology for designing sparse RSNNs by pruning an untrained (arbitrarily initialized) large model. 
We introduce a novel Lyapunov Noise Pruning (LNP) algorithm that uses graph sparsification methods and utilizes Lyapunov exponents to design a stable sparse RSNN from an untrained RSNN. We show that the LNP can leverage diversity in neuronal timescales to design a sparse Heterogeneous RSNN (HRSNN). Further, we show that the same sparse HRSNN model can be trained for different tasks, such as image classification and time-series prediction. The experimental results show that, in spite of being task-agnostic, LNP increases computational efficiency (fewer neurons and synapses) and prediction performance of RSNNs compared to traditional activity-based pruning of trained dense models.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors significantly improved the quality of the manuscript during the revision. However, the value of the pruning strategy compared to the SOTA SNN training and quantization approach is not sufficiently significant. Based on such effort and shortcomings, I would love to increase my rating to 5. 
--------------------------------------------------------
This paper proposes an innovative method called LNP, which prunes an untrained recurrent spike network via the utilization of graph theory and Lyapunov spectra. The approach proposed in this paper is better suited for discovering a remarkable sub-network in liquid-state machines or similar situations. In the SNN, energy consumption is the factor that measures the network efficiency, and not only network sparsity but also spike frequency will also affect it. Due to the variation in spike frequency before and after training, it is almost impossible to predict whether the pruned substructure remains efficient after the training process.

### Strengths
The LNP enables pruning the network prior to training, and effectively enhancing the sparsity while maintaining commendable performance.

### Weaknesses
1. The authors should elucidate the motivation behind their proposed method. What significance does their method hold for the field of spiking neural networks?
2. The network model and task employed in the experiments are too simple and do not need pruning. I am more intrigued by the adaptability of the LNP method under more intricate situations.
3. The authors provided a detailed description of the latest pruning works in SNN in the second paragraph of the Introduction. However, their results are not referenced in Table 1. Meanwhile, the compared works in Table 1 lack reference links.
4. The equation 3 appears abrupt. I don’t understand why the spike frequency of untrained SNN can be used to construct this linearized model. And how to calculate the matrix D and L?
5. Activity Pruning needs to provide citation or explanation.
6. Lorenz and Rossler are unfamiliar datasets in the field of SNN. Could you please provide the citations?

### Questions
See the weakness please.

### Soundness
3 good

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this study, the task-agnostic Lyapunov Noise pruning (LNP) on Heterogeneous RSNN (HRSNN) is first proposed, and designed in a graph sparsification manner. HRSNN has a similar hierarchical structure as the Liquid State Machine (LSM), which includes a subsequent readout layer. LNP includes four steps 1) random dropout in HRSNN synaptic weights, where dropout rate is determined by the Lyapunov exponent, 2) mask out neurons with the lowest betweenness centrality from HRSNN, 3) add edges to maximize degree heterogeneity within local neighborhoods, 4) a Bayesian optimization for fine-tuning hyperparameter set of neurons. Results show that LNP maintains performance on time-series prediction on two chaotic systems, and two real-world datasets when FLOPS/#parameters are reduced.

### Strengths
LNP consider the equivalence between dense and sparse network, rather than simply optimizing the performance, which makes it suitable for random graph model like HRSNN. The design of LNP makes it not only task-agnostic but also learning-rule-agnostic, endowing SNNs from distinct research backgrounds with abundant flexibility in choice. Stability in pruning is also an interesting topic considering that SNN is a dynamic system.

### Weaknesses
As an SNN pruning algorithm, there are still some vague parts influencing my rate of score:

1. The calculation of sparsity is unclear to me. In most previous SNN pruning studies, the parameter/neuron sparsity includes all learnable parameters and should also include readout layers here. This vagueness could lead to bias in evaluating the power of the pruning algorithm.
2. The results on CIFAR-10 display no error bar. Considering that the spectral graph pruning includes a random dropout, the accuracy variance should also be reported.
3. LNP is essentially composed of four individual and loosely related parts with different goals. None of those are indispensable for one another. The authors should exhibit an ablation study on the contribution of each component.

### Questions
Considering that non-recurrent SNNs are still popular among SNN communities, is it possible to generalize and experiment on purely feed-forward SNNs like previous works do?

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
This paper presents a novel task-agnostic method called Lyapunov Noise Pruning (LNP) for designing sparse heterogeneous Recurrent Spiking Neural Networks (HRSNNs). Unlike conventional approaches that prune dense networks after training them for a specific task, LNP starts with an untrained and arbitrarily initialized dense HRSNN model. It uses the Lyapunov spectrum and spectral graph sparsification algorithms to prune synapses and neurons while maintaining network stability. The resulting untrained sparse HRSNN can then be trained for various target tasks using supervised or unsupervised methods.

### Strengths
1. In this article, authors introduced a novel method for task-agnostic pruning, achieved by pruning a Recurrent Spiking Neural Network (RSNN) prior to the training process.

2. The study demonstrates that Lyapunov Noise Pruning (LNP) outperforms AP in terms of network efficiency enhancement.

3. This research proposes a technique that capitalizes on the heterogeneous timescales of RSNN, offering both biological plausibility and a supportive role in the pruning process.

### Weaknesses
1. The author did not show the difference between random initialization and LNP, and didnot analyze the dynamic property of the hetergeneous timescale of LNP and how it influence the training process of RSNN.

2. The author didnot carefully organize the citation and references. For example,  in Reference section, 'Yanqi Chen, Zhaofei Yu, Wei Fang, Tiejun Huang, and Yonghong Tian. Pruning of deep spiking neural networks through gradient rewiring. arXiv preprint arXiv:2105.04916, 2021a.', and 'Yanqing Chen, Zhaofei Yu, Wei Fang, Tiejun Huang, and Yonghong Tian. Pruning of deep spiking neural networks through gradient rewiring. 2021b. doi: 10.24963/ijcai.2021/236. URL https://doi.org/10.24963/ijcai.2021/236.' are same paper, but sited twice.

### Questions
1. In the study conducted by Panda et al., it was demonstrated that their approach resulted in a higher degree of sparsity and a lower impact on accuracy when compared to the LNP method. Could you please elucidate the underlying reasons for this observed outcome?

2. Could you provide a comprehensive description of the HRSNN architecture? It appears that there is a readout layer characterized by dense connections that are not subjected to pruning via the LNP technique.

3. The Lyapunov spectrum is customarily analyzed in continuous dynamical systems, such as RNNs. How did the researchers address the discontinuity of spikes in RSNNs when determining the Lyapunov spectrum?

4. Kindly elucidate the principal dynamic characteristic of LNP RSNNs prior to the completion of the final training process and provide an explanation as to why randomly sparse Initialized RSNNs lack this particular dynamical property. Could you also discuss dynamic property of LNP-RSNN, for example, using temporal correlation function and the Lyapunov spectrum in this context?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel Lyapunov noise pruning algorithm to design a sparse RSNN from an untrained RSNN. The method is task-agnostic, which can be trained later for different down-streaming tasks. The experimental results shows the effectiveness of this method.

### Strengths
- The paper is well-written, with very clear illustration on the methods and implementations. 
- This paper proposes a novel task-agnostic method for designing sparse RSNN, which is quite novel compared with conventional approach of designing task-dependent pruning way.
- Compare to conventional network pruning methods of RSNN, the proposed method shows a very good generalization ability, proving by several down-streaming tasks’ consistent performance.

### Weaknesses
- The motivation is not fully explained, see the questions below.
- Though in the experimental parts, it is showed to be advantageous compared other pruning methods, overall, it is still hard to see how useful this method can be in real cases. Namely, one might ask, compared to a SOTA method (not only RSNN) in a specific problem (say tasks listed in the experimental parts), how much gain (in terms of performance or efficiency) does this method bring?

### Questions
1.	Why focusing on recurrent SNN? I understand Lyapunov noise pruning (LNP) is originally developed for RNNs, why do you choose to consider this method in spiking cases?
2.	RSNN has two hierarchical dynamics, one is the recurrency of network, the other one is the intrinsic recurrency of spiking neurons, do you distinguish these two dynamics? Or on the other hand, whether this method can be used directly in spiking neural networks?
3.	What does the word “heterogeneous” mean here? Is it just the time constant is variable compared with conventional neuron models? (though in many literatures it is already considered)

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

# A Dendritic-Inspired Network Science Generative Model for Topological Initialization of Connectivity in Sparse Artificial Neural Networks

- Decision: Reject
- Scores: 6, 6, 0, 2

## Abstract
Artificial neural networks (ANNs) achieve remarkable performance but at the unsustainable cost of extreme parameter density. In contrast, biological networks operate with ultra-sparse, highly organized structures, where dendrites play a central role in shaping information integration. Here we introduce the Dendritic Network Model (DNM), a generative framework that bridges this gap by embedding dendritic-inspired connectivity principles into sparse artificial networks. Unlike conventional random initialization, DNM defines connectivity through parametric distributions of dendrites, receptive fields, and synapses, enabling precise control of modularity, hierarchy, and degree heterogeneity. This parametric flexibility allows DNM to generate a wide spectrum of network topologies, from clustered modular architectures to scale-free hierarchies, whose geometry can be characterized and optimized with network-science metrics. Across image classification benchmarks (MNIST, Fashion-MNIST, EMNIST, CIFAR-10), DNM consistently outperforms classical sparse initializations at extreme sparsity (99\%), in both static and dynamic sparse training regimes. Moreover, when integrated into state-of-the-art dynamic sparse training frameworks and applied to Transformer architectures for machine translation, DNM enhances accuracy while preserving efficiency. By aligning neural network initialization with dendritic design principles, DNM demonstrates that sparse bio-inspired network science modelling is a structural advantage in deep learning, offering a principled initialization frame to train scalable and energy-efficient machine intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the Dendritic Network Model (DNM), a parametric framework for generating network topologies to initialize neural architectures that can be trained in either static or dynamic sparse training setups. Drawing inspiration from neuroscience, the framework is evaluated across multiple architectures (MLPs and Transformers) through an extensive set of experiments spanning image classification and machine translation tasks. The experimental results suggest that the proposed methodology outperforms competing approaches.

### Strengths
A significant merit of this work is its exploration of an underrepresented research direction in the AI community, attempting to transfer knowledge from neuroscience to artificial intelligence. The paper addresses interesting scientific questions and supports its claims through a comprehensive experimental evaluation.

### Weaknesses
The paper overlooks a related line of research that could enhance the related works section: artificial neural networks with complex topologies [1,2,3,4]. In a sense, these studies can be considered a generalization of what is proposed here, as they are not limited to architectures constrained to bi/multipartite topologies (i.e., layered MLPs).

**Major**

Extreme Sparsity Regime: The presented results primarily focus on an extreme sparsity regime (99%). I wonder whether such a low number of connections might push the tested topologies into degenerate cnfigurations where the topological characteristics of each graph family could be at least partially compromised or diminished.

Hyperparameter Optimization and Fair Comparison: A more significant concern relates to how the results in the tables are constructed. Each result is presented as mean ± standard deviation over 3 different seeds. However, according to the appendix, if I understand correctly, the parameters selected for the DNM configurations appear to be the result of hyperparameter optimization. I am concerned that this may constitute a form of cherry-picking if the same level of hyperparameter search is not applied to the baseline initializations. 

In other words, suppose the results presented for a specific task represent the 3 best networks from a sweep of 30 configurations. Would DNM still demonstrate superiority if we also selected the 3 best random networks from a sweep of 30 configurations?

Dense Baseline Comparison: I am curious about the performance of a dense MLP with the same number of connections as the tested sparse networks but with fewer neurons across the various experiments. I believe this additional baseline could make the significance of the presented results more convincing, especially given that current ML frameworks (both software and hardware) are particularly optimized for operations on dense MLPs.

### Questions
I would greatly appreciate comments addressing the major weaknesses reported above. I believe resolving these concerns could eliminate potential misunderstandings and lead to a more objective and informed evaluation of this work.

[1] R. L. S. Monteiro et al. “A Model for Improving the Learning Curves of Artificial Neural Networks”. en. In: PLOS ONE 11.2 (2016)

[2] S. Xie et al. “Exploring Randomly Wired Neural Networks for Image Recognition”. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). (2019)

[3] J. Stier and M. Granitzer. “deepstruct – linking deep learning and graph theory”. In: Software Impacts 11 (2022)

[4] T. Boccato et al. “Beyond multilayer perceptrons: Investigating complex topologies in neural networks”. In: Neural Networks 171 (2024)

### Soundness
3

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
2

### Summary
The paper introduces the Dendritic Network Model (DNM), a biologically inspired generative framework for initializing sparse artificial neural networks. Inspired by dendritic processing in biological neurons, DNM defines sparse connectivity through parametric distributions of dendrites, receptive fields, and synapses. It enables flexible control over modularity, hierarchy, and degree heterogeneity, allowing the generation of diverse network topologies. Experiments on image classification (MNIST, Fashion-MNIST, EMNIST, CIFAR-10) and machine translation tasks (Multi30k, IWSLT14, WMT17) show that DNM consistently outperforms classical sparse initialization methods at extreme sparsity levels, improving both efficiency and accuracy.

### Strengths
1. The paper introduces a novel and biologically grounded generative framework for sparse network initialization.
2. Extensive experiments across multiple datasets and architectures convincingly validate the proposed model.
3. The analysis establishes a clear link between network topology and task complexity, offering theoretical insights into structure–function relationships.

### Weaknesses
I am not a researcher in this area, but I feel interested in this topic. I have several concerns:
1. How sensitive is DNM to hyperparameter choices, and how would one select optimal configurations for unseen tasks?
2. The model’s computational overhead and scalability on very large-scale networks are not thoroughly discussed.
3. The work focuses primarily on initialization; it lacks exploration of how DNM interacts with other training dynamics beyond sparsity.
4. Can DNM be effectively combined with pruning or regularization techniques for additional efficiency gains?
5. Does the proposed framework generalize to non-visual or non-language domains, such as reinforcement learning or graph-based tasks? Please discuss.

Hope that the authors could answer the above questions. Thanks!

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
2

### Summary
This paper, inspired by dendrites in the brain, proposes DNM, a parameterization of a broad class of sparse neural network layer topologies, using parameters sparsity, dendritic distribution, receptive field width distribution, degree distribution, synaptic distribution, and layer border wiring pattern. A given topology replaces a fully connected layer. The authors evaluate the performance of different DNM topologies when replacing hidden layers with DNM layers on various image classification tasks as well as in transformers for machine translation tasks. The authors find that DNM perform slightly better than alternative sparse methods across the board.

### Strengths
- The paper has a thorough set of experiments, comparing to four different sparse baselines, on four different images datasets, plus two machine translation tasks. They evaluate for static sparse training as well as multiple dynamic sparse training methods. They conduct a sensitivity analysis and analyze the structure of the best performing networks.
- The parameterization of sparse topologies is well motivated by dendritic networks in the brain, and is novel.

### Weaknesses
- The design of dendritic sparse topologies is not very novel — it seems to be similar to several prior parameterizations of sparsity, perhaps just with more tunable parameters (see Robinett and Kepner, 2018 as an example). 
- The performance benefit of the DNM layers is very minimal
- Some of the evaluation is suspect. For example, the reported standard error of accuracy on MNIST are in the 0.01% range, which seems unusually narrow. When I've done experiments on MNIST in the past, the variance between runs is much higher than 0.01%. 
- The experiments are lacking many other crucial details (see questions)
- The authors suggest that they choose the best DNM model for each experiment from a hyperparameter sweep. Thus, the higher performance could just be due to trying more random architectures for DNM than other architectures.
- There is no single DNM architecture that works the best. So for any new problem, a new hyperparameter sweep would be required, rather than a "one size fits all" sparse architecture that outperforms fully connected networks. As a result, I'm not sure there is much impact of having a high performing sparse architecture.
- These results seem to be of minor impact to the community. The authors do not discuss why these architectures should be impactful, or if there's any potential for them to be adopted into mainstream state of the art neural network architecture design.

### Questions
Q1. what is the architecture of the models for MNIST? how many parameters are in the models? How many parameters are in the sparse models vs the fully connected models? 

Q2. what is the evaluation strategy for the DNM models and other baselines on MNIST? how many different DNM variants are tested? How is the best model selected — on a held-out validation set? It could be useful to see a table of training, validation, and test accuracies for the different models tested. 

Q3. Why are the error levels so low for the MNIST and other experiments? Is there other work that finds similar error levels when training different initializations on MNIST?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces the Dendritic Network Model (DNM), a generative framework for initializing sparse neural networks using biologically inspired connectivity principles. The model is evaluated across image classification and translation benchmarks and compared with several static and dynamic sparse training (DST) methods. DNM is shown to consistently outperform baselines at high sparsity levels and performs competitively with data-informed methods such as SNIP and CSTI.

### Strengths
The topic is timely and relevant: understanding the link between biological structure and sparse AI architectures is of broad interest to the ICLR community.

The model is novel as a generative topological initialization approach and appears to integrate concepts from network science and computational neuroscience.

The paper includes comprehensive experiments across multiple architectures and datasets, including transformer-based translation experiments in addition to image benchmarks.

### Weaknesses
The biological connection is not clearly articulated. The work lacks a discussion of DNM’s relationship to what is known about biological dendritic connectivity patterns. There has been significant progress here as a result of connectomics datasets (e.g. the MICrONs dataset and related papers) and it is a weakness that DNMs are not contextualised better here.

The paper’s clarity is a major limitation. Key terms and references to other methods are often introduced without adequate explanation or focus. For example, CHTs, SET and RigL are not adequately introduced and motivated. Why are they good comparisons to DNMs? Section 3 mixes implementation detail and inspiration in a way that is difficult to follow. Figure 2 is not motivated and described well enough. What is the purpose of the characterisation? It is not clear what the reader should take away from this figure, other than that the DNM can generate a range of networks. 

Experimental methodology isn’t clearly communicated to the reader and is underexplained. For example, learning rate hyperparameter search doesn’t seem to have been done for MLPs, instead all models were trained with the same learning rate (Appendix B1).  This seems unlikely given the range of image datasets trained on. 

Section 5 does analyse somewhat the structure of best performing DNM models. But this analysis is descriptive and speculative, and doesn’t clearly communicate the principles the best performing DNM models are leveraging for good task performance.

Minor:
- The term “sandwich layer” (L137) is unconventional and confusing; “hidden layer” would suffice unless a specific distinction is intended.

- Figure 4 is referenced in the main text but only appears in the appendix.

- Several claims (e.g., about scale-free structure, L252) require citations or clearer definitions.

### Questions
Line 51: Could you clarify the claimed relationship between dendritic computation and convolutional models?

Line 40-43: The efficiency statements should either be framed as hypotheses or supported by explicit citations.

Line 57: How do you operationally define “dendritic topology”?

Figure 4 is an appendix figure. 

Line 211: Please describe how the coalescent embedding in hyperbolic space is computed and what it reveals about network structure.

Line 252: Claim about scale free networks should be cited.

### Soundness
2

### Presentation
1

### Contribution
2

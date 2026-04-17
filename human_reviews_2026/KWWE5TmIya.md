# Feature–Label Embedding Alignment for Backprop-Free tiny Networks

- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Forward-only training methods, such as Forward-Forward (FF) learning, provide an alternative to backpropagation by eliminating the backward pass. This reduces memory usage and makes them well-suited for forward-propagation accelerators increasingly common in modern embedded devices. However, current forward-only methods struggle to scale to deep networks and challenging visual recognition tasks, resulting in a substantial performance gap compared to backpropagation.  
We address this limitation with the $\textit{Feature-Label Embedding Alignment (FLEA)}$ block, a novel architectural component that allows FF networks to scale effectively while remaining forward-only. FLEA introduces $\textit{layer-wise discriminative learning}$, where each layer independently optimizes its parameters by aligning its feature embedding with the corresponding label embedding that maximizes class separability. This produces more discriminative representations in deeper layers and significantly narrows the performance gap with backpropagation.  
Experiments demonstrate that FLEA-equipped FF networks achieve competitive accuracy on complex visual benchmarks while retaining the memory efficiency and accelerator compatibility that make forward-only training appealing for resource-constrained systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper enhances Forward-Forward (FF) learning — a forward-only alternative to backpropagation — by introducing the Feature-Label Embedding Alignment (FLEA) block. While FF methods reduce memory usage and suit hardware accelerators, they struggle to scale to deep networks and complex visual tasks. FLEA overcomes this limitation through layer-wise discriminative learning, aligning each layer’s feature embeddings with label embeddings to maximize class separability. This design enables FF networks to learn more discriminative representations, close the performance gap with backpropagation, and maintain the efficiency and hardware compatibility ideal for resource-constrained environments.

### Strengths
The paper proposed a novel method in Forward-Forward Learning, and provided systematic analysis from the preliminary background to their framework. Also they proved that their method is more efficient than conventional ways from the theoretical level. Based on their experiments, we could truly find their advantages than previous works and the experimental results are significant from the results they presented.

### Weaknesses
1) Firstly, even though they provided a detailed description of their method, the novelty is not enough. They are based on the basical chain rule and introduce another way on processing the 'label' in the traditional FF learning. The method is not so novel even if they gave a lot of descriptions. The main motivation aligns with the traditional FF learning principles and they did not design a novel method in FF learning. 2) Based on the experimental results, they truly present the significance of their method. However, they just performed experiments on CIFAR10/100, and also they just focused on the classification tasks. The models they used are convolutional NN. From my opinion, extensive experiments on other datasets, other tasks, and other various models not just CNN should be conducted to support their method's effectiveness.  Even if they offered the results on MNIST, it's a simple dataset and easier than CIFAR datasets. 3)  I am not sure if they compared to enough baselines in the relavant field. They just used 3 baselines in the main text. Considering the second weakness I proposed, I think the current experiments are not sufficient to support their method.

### Questions
N/A.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a novel architectural component, FLEA(Feature-Label Embedding Alignment), aimed at addressing the performance limitations of current forward-only methods, which struggle to scale to deep networks and challenging visual recognition tasks. FLEA introduces layer-wise discriminative learning, where each layer independently optimizes its parameters. The experiments on CIFAR-10/100 and MNIST show that FLEA-equipped FF networks achieve competitive accuracy on complex visual benchmarks.

### Strengths
- This paper introduces the FLEA block, a new architectural component designed to overcome the limitations of existing forward-only models, which often struggle to scale effectively to deeper networks and more complex visual recognition tasks. Building upon this module, the authors further develop a complete network architecture and a tailored training strategy. Together, these innovations enable the model to achieve substantial performance improvements across multiple benchmarks, clearly surpassing prior forward-only approaches.
- Beyond empirical results, the paper offers a theoretical analysis of the recognition capability of forward-only models, shedding light on their representational properties and limitations. This analytical perspective provides valuable guidance for future research, helping to establish a more principled foundation for advancing the understanding and development of forward-only neural architectures.

### Weaknesses
- The experimental evaluation primarily focuses on CIFAR-10 and CIFAR-100, with no tests on other datasets such as ImageNet, COCO, or more complex visual recognition tasks. This narrow scope raises questions about the generalizability of the proposed approach to diverse or large-scale settings.
- The method relies on numerous hyperparameters, yet the paper provides limited ablation studies or comparative experiments to assess their impact. This lack of analysis makes it difficult to determine whether the reported high accuracy stems from the intrinsic design of the FLEA block or from careful hyperparameter tuning, weakening the persuasiveness of the experimental results.
- Previous related works typically validated their methods on widely used architectures like ResNet, whereas this paper evaluates the approach solely on a self-designed network. This raises concerns about the practical applicability of the method and whether it can be effectively transferred or adapted to a broad range of existing architectures.

### Questions
- Could the authors elaborate on the potential practical applications of the proposed approach? Specifically, it would be valuable to clarify whether this method could be extended to scenarios where backpropagation incurs high memory or computational costs, such as the training of large-scale models or LLMs. If such deep and large models could be effectively trained in a layer-by-layer fashion, this would represent a significant contribution to the machine learning community.
In its current form, the manuscript appears relatively simple. For example, a straightforward baseline could involve injecting a projection at each layer or block to directly predict the label. While the label encoder/decoder design is novel, evaluating it solely on classification tasks may not fully demonstrate the method’s potential or effectiveness across broader applications. Further discussion and experiments would help clarify the generalizability and practical impact of the approach.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Current neural networks are typically trained using backpropagation. To remove the backward pass computation, Forward-Forward training is previously studied. However, FF learning has a large accuracy drop compared to backpropagation. This paper proposes a feature-label embedding alignment (FLEA) to enhance the performance of FF training. FLEA introduces layer-wise discriminative learning, where each layer independently optimizes its parameters by aligning its feature embedding with the corresponding label embedding that maximizes class separability. Experiments demonstrate that FF with FLEA outperforms previous FF training methods.

### Strengths
1. A good background and explanation is provided for Forward-Forward learning.
2. The hard example learning is proposed.
3. The results show that the proposed FLEA outperforms previous FF method by large.

### Weaknesses
1. The explanation of Forward–Forward learning in section 3 provides a good background. However, the analysis is conventional, there is no tight connection between this explanation and the proposed method. Moreover, the analysis (specifically, the separability of equation 2) will not stand for other layers except the first layer.
2. The label encoder is basically the same with MLP classifer layer. In my understand, the proposed FLEA is a layer-wise classification learning method. From this point, the proposed method is similar with DeeperForward.
3. Some details are not clear, such as how to collect the layer-wise classification into the final classification, the gradient is only provided for features, but not for the weights to be learned. 
4. The experiments use different network architectures, making the comparison not on the same basis and making it hard to evaluate the advantages of the proposed method. From the layer-wise classification viewpoint, concatenating features of previous layers (Appendix A.2) maybe the key for the performance improvement.

### Questions
1. What's the difference of the proposed method with a mlp classification on the representation encoder? What if CE loss is used instead of the positive-negative loss of line 361?
2. What's the difference of the proposed method with DeeperForward?
3. How the proposed method performs on simple feed-forward networks?
4. The FF method of Hinton 2022 also use hard negative labels, what are the differences?

### Soundness
2

### Presentation
2

### Contribution
2

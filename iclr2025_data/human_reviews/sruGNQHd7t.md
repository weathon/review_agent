## Human Reviewer 1

### Summary
The researchers propose a privacy-preserving method for deep learning services that transforms input data into an obfuscated form before sending it to cloud models and then decodes the results locally, enabling the secure use of external AI services while protecting sensitive data.

### Strengths
The paper addresses the privacy issues in deep learning and proposes a privacy-preserving strategy that users can implement independently,
without requiring any modification to the deep learning models.

### Weaknesses
1. The introduction part mentions different approaches to privacy preservation, including differential privacy, holomorphic encryption, and multi-party computation. Then, they claim that their strategy is different and better. However, in their experiments part, they did not compare these existing approaches.
2. In the contribution part, the paper mentions three aspects. However, these claims are not clear. The first aspect is "An evaluation of the feasibility of applying domain shifting as an input obfuscation method and a demonstration of domain shifting theory on pre-trained DNN models." It seems it does not correspond to later sections.
3. The third aspect is "A comprehensive evaluation of both obfuscation methods, ..." However, the evaluation in the experiment part is not sufficient.
4. The threat model part is not clear. It is supposed to explain information on the adversary's side. Currently, the content in this part is not.
5. More experiments should be given to show the effectiveness of the method.

### Questions
See weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a privacy-preserving strategy for deep learning queries through domain shifting. Using encoder-decoder models, it obfuscates inputs without modifying the deep learning model, supporting both whitebox and blackbox scenarios. The approach demonstrates effective privacy protection with minimal impact on performance across multiple datasets.

### Strengths
1. The key idea by leveraging the encoder and decoder is intersting and clear.
2. The method accommodates both blackbox and whitebox access scenarios. Model-agnostic in-place domain shifting works without internal model access, while model-specific out-of-place shifting enhances privacy when access to model internals is available.
3.Compared to homomorphic encryption, this approach significantly reduces computational costs, achieving rapid inference times even on complex datasets. It maintains privacy effectively without the high latency typically associated with other cryptographic methods.

### Weaknesses
1. If an attacker repeatedly sends disguised data and observes the output patterns of the model, the label replacement scheme may be gradually compromised through an inference attack, thereby undermining privacy protection. It would be interesting to explore this extreme scenario.
2. While leveraging the encoder and decoder has been discussed in [1, 2], it is important to clarify the differences between them.
[1] InferDPT: Privacy-Preserving Inference for Black-box Large Language Model
[2] FedAdOb: Privacy-Preserving Federated Deep Learning with Adaptive Obfuscation
3. It would be more valuable to explore these methods with more sensitive data, such as facial images or address data in NLP.

### Questions
1.The method requires that the pseudo-data can maintain consistent classification results across different models, but in practice, how to ensure that the accuracy of the decoder is not affected by the fact that the classification labels may vary somewhat due to the structural differences of different models?
2.Could the authors provide formal privacy metrics, such as leveraging membership inference attack, to quantitatively evaluate privacy instead of relying solely on empirical SSIM?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper proposes a framework for protecting the privacy of inputs during inference. The key insight is to transform the input into a different domain using an encoder before sending it to an external service provider to perform inference. A decoder is then used to transform the model's prediction to recover the correct output. The authors evaluate their method on several image classification datasets (MNIST< CIFAR-10, Imagenet)

### Strengths
1. The paper is easy to understand
2. Evaluation is done on several image classification datasets.

### Weaknesses
In the model-agnostic setting, the authors propose to transform the input image from its original class (class $j$) to a different class(class $j'=j+i \mod M$). Since $i$ is chosen randomly, it wouldn't be possible for the service provider to know the original class. Unfortunately, there are two major flaws with this approach:

1.  Compute: Doing this requires a lot of compute (likely on the order of compute used to perform the inference itself!). This makes it impractical in most remote-inference settings, where the user does not have the compute resources necessary to perform inference (let alone complex input transformations) locally.

2. Access to Training data: The authors assume that the user has access to labeled training data. This data is required to train the GAN (section 4.3.1) to perform the transformation of the input from one class to another requires access to labeled data. Note that this labeled data could have been used to train an inference model and perform inference locally, sidestepping the problem of performing privacy-perserving remote inference! Additionally, the  accuracy of this whole pipeline is bottlenecked by the ability of the GAN to correctly transform the input image from one label to another, so I'm not convinced that the proposed approach is better than just training a model on the user's end and performing inference locally.


In the model specific setting, the authors propose using a network to transform the input in a way that it reduces the SSIM between the original and transformed image.Reduction of SSIM does not guarantee privacy. More rigorous privacy guarantees involve adding noise to the encoded input [1]. As far as I can tell, the proposed method provides no privacy guarantees.

[1] Maeng et al. "Bounding the Invertibility of Privacy-preserving Instance Encoding using Fisher Information" NeurIPS 2023

### Questions
Can you please address the points raised under weaknesses?

### Soundness
1

### Presentation
3

### Contribution
1

### Rating
1

### Confidence
5
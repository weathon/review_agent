# Context is All You Need

- Avg Score: 3.00
- Decision: Reject
- Scores: 6, 2, 2, 2

## Abstract
Artificial Neural Networks (ANNs) are increasingly deployed across diverse domains, often requiring them to generalize beyond their training conditions. This shift in context frequently leads to performance degradation, a central challenge in Domain Generalization (DG). While numerous techniques exist to mitigate this issue (e.g., fine-tuning, activation steering, meta-learning, adversarial training, normalization-based approaches, and parameter-efficient methods such as prompt tuning), they are often complex, resource-intensive, and difficult to scale; particularly for large models like Large Language Models (LLMs). In contrast, we introduce CONTXT (\emph{\textbf{C}ontextual augmentati\textbf{O}n for \textbf{N}eural fea\textbf{T}ure \textbf{X} \textbf{T}ransforms}): a simple, intuitive, and elegant method for contextual adaptation. CONTXT work by augmenting the model’s internal representations with lightweight, contextually relevant feature indexes through straightforward multiplicative and additive vector operations. Despite its simplicity, CONTXT significantly improves performance across both discriminative (e.g., classification with ANNs/CNNs) and generative (e.g., LLMs) tasks. With minimal computational overhead and straight forward integration, CONTXT layers offer a practical and effective solution to DG and a variety of problems facing ANNs, demonstrating that strong results need not come at the cost of complexity. More generally, CONTXT provides a compact mechanism to manipulate information flow and steer ANN processing in a desired direction without retraining the network.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
1

### Summary
In this manuscript, the authors aim to address the challenge of Domain Generalization (DG) in Artificial Neural Networks (ANNs), where models often suffer from performance degradation when deployed in contexts different from their training conditions. Existing solutions for DG (e.g., fine-tuning, adversarial training, activation steering) are typically complex, resource-intensive, and hard to scale. CONTXT operates through simple vector arithmetic: first, it computes a "context index" as the difference between a precomputed context vector and the current feature representation of the input at a chosen layer. Then, it adjusts the input features by adding a scaled version of this index. It also supports multi-context adjustment by linearly combining multiple context indices. Experimental results demonstrate the effectiveness of CONTXT across both discriminative and generative tasks.

### Strengths
The proposed CONTXT is a lightweight, brain-inspired technique that modifies intermediate network features to inject or remove contextual information without retraining.
CONTXT provides a simple, interpretable, and efficient solution to DG, avoiding the complexity and resource costs of retraining or fine-tuning while delivering substantial performance gains.

### Weaknesses
There are some concerns for the manuscript as follows:

1.The section METHODS is too brief. Some contents, e.g., the first paragraph in section RESULTS, can be omitted to extend section METHODS.
2.Thus, in the context vector computation, when should we select the feature of a representative sample or the mean feature over samples exhibiting? Since this selection may has great influence to the experimental results, what is the selection standard?
3.The image classification is introduced as the downstream task for the proposed method, so if the proposed method can be used in other tasks, e.g., text classification, how to calculate contextual references? And what will be the experimental results?

### Questions
See the Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a simple yet effective approach for domain generalization in Artificial Neural Networks. CONTXT modifies the intermediate features by linearly augmenting them with relevant contextual information, enabling better OOD performance without any auxiliary networks or retraining requirements. The method is evaluated on discriminative tasks as well as generative tasks and performs well, as reported by the paper.

### Strengths
1. The paper presents a simple yet powerful approach for domain generalization in Artificial Neural Networks. 

2. The proposed method is applied to both classification and generative tasks, which shows that the method can be applied to most of the existing ANNs.

3. The connection of work with the brain seems interesting and motivating.

### Weaknesses
1. The paper lacks a much deeper analysis of why their method actually works. For instance, some theoretical backing (as the method is quite straightforward, it will be easy to develop a nice theory on top of it), attention visualization, will make the work really strong.

2. The experimental analysis of the work is not sufficient; for instance, the t-SNE plots could tell the exact difference between how good the method is performing. Plotting them for the model using the proposed method and supervised fine-tuning can further clarify and solidify the findings of the paper.

3. The paper requires some rewriting as well. For instance, in the results section, the paper abruptly starts introducing the method again in line 180. An additional separate section on the theoretical analysis of the proposed method is highly suggested.

Overall, in its current state, I believe that the work is not sufficient for the main conference; however, the work still has a promising direction, and when extended with some theoretical analysis and has a clear intuition of why the method works can make it a really useful and important finding.

### Questions
See Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents CONTXT, a method of activation steering that helps neural networks perform better in new domains. CONTXT first computes a context vector, then uses this context vector at inference time (by simple vector operations such as addition or subtraction) to modify the model's internal activations. The paper experiements on OOD image classification with VGG and text generation with LLMs.

### Strengths
* Simplicity: The method does not require re-training the model, but rather, steering activations at inference. This is a major selling point since model re-training/finetuning can be cumbersome and expensive.
* The core idea of the paper is presented well, and Figure 1 (while not exactly visually appealing) does a good job of highlighting the method clearly.

### Weaknesses
* The fundamental idea of activation steering is not entirely new. This concept has been applied in various different domains, including text generation and even other image generative models. The paper claims the novelty on the specific formulation and the application of their method to OOD classification. However, I am not convinced that CONTXT provides a fundamentally new concept, but rather an application of an existing concept.
* While the paper claims that there is little hyperparameter tuning, it still needs to be tuned per-layer, which is not trivial. The authors suggest that this could be learned in the validation setting. Furthermore, the claim that CONTXT requires "minimal computational overhead" or "minimal latency" is not in good faith, since computing context vectors themselves could be expensive depending on the setting. This leads me to my next point:
* CONTXT is motivated by Domain Generalization (DG) and also claims to experiment on DG settings; however, the experimental settings do not align with the core issues in DG. In DG, we assume that the target domain is not available at any time. Thus, since CONTXT requires data from a target domain to create the context vector, it cannot be classified as tacking the DG problem. It is closer to domain adaptation or test-time adaptation. 
  * To add to the point above, the motivating example in Figure 2 is a bit misleading because it instills the idea that the "correct" context is known for the image. However, the correct context here depends entirely on the classification of the image ("cow on a beach"). So essentially, we need to have already classified the image to know which context to remove or add.

### Questions
I would appreciate authors' response to the weaknesses addressed.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes CONTXT, a lightweight “context steering” mechanism that edits intermediate activations by adding a scaled difference between a cached “context vector” and the current hidden state. The method aims to (a) improve domain generalization for vision classifiers and (b) steer LLM generations toward a desired attribute/persona without fine-tuning.

### Strengths
S1. Domain generalization using just a vector operation is a very solid research direction.

S2. The paper is generally clear apart from figures (refer weakness)

### Weaknesses
W1. "Brain-inspired" motivation is confusing: The biological framing (PFC top-down control) is used to justify adding a linear direction to features, but no concrete architectural, algorithmic, or intuitive notions are provided. I don't really understand how this is related to brain in any way.

W2. The construction of "context vectors" is largely a rehash of a very relevant line of literature on prototype-based/centroid-based learning, like [1] which is not discussed at all. Context vectors are similar to mean feature vectors which act as references and moves points toward/away from them via linear vectors - a clear similarity to using the similarity as the distance metric.

W3. "Context" needs to be better defined: The definition is changed across sections, making it hard to formalize what a valid context vector is. For language, it is defined as "The injected (in-domain) context vector comprised of the average feature representation across all training domain samples," - which does not make sense to me, is the entire training corpus a context? If this is true then why not do the same for images? Additionally, the paper gives formulas for single/multiple contexts, but not a rigorous selection/validation protocol for constructing them in general.

W4. The quality of figures is very poor, with no error bars, statistical tests, etc. reported.

[1] This Looks Like That: Deep Learning for Interpretable Image Recognition, NeurIPS 2019

### Questions
1. Can the authors explain in detail why their method is derived from brain functioning?

2. Were any other similar methods explored?

### Soundness
1

### Presentation
1

### Contribution
2

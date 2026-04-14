# PolaFormer: Polarity-aware Linear Attention for Vision Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8

## Abstract
Linear attention has emerged as a promising alternative to softmax-based attention, leveraging kernelized feature maps to reduce complexity from quadratic to linear in sequence length. However, the non-negative constraint on feature maps and the relaxed exponential function used in approximation lead to significant information loss compared to the original query-key dot products, resulting in less discriminative attention maps with higher entropy. To address the missing interactions driven by negative values in query-key pairs, we propose a polarity-aware linear attention mechanism that explicitly models both same-signed and opposite-signed query-key interactions, ensuring comprehensive coverage of relational information. Furthermore, to restore the spiky properties of attention maps, we provide a theoretical analysis proving the existence of a class of element-wise functions (with positive first and second derivatives) that can reduce entropy in the attention distribution. For simplicity, and recognizing the distinct contributions of each dimension, we employ a learnable power function for rescaling, allowing strong and weak attention signals to be effectively separated. Extensive experiments demonstrate that the proposed PolaFormer improves performance on various vision tasks, enhancing both expressiveness and efficiency by up to 4.6%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper aims to improve the state of the art linear transformers with linear complexity O(N) with N the sequence length. It focus on addressing two issues as identified: 1) Loss of negative values, and 2) loss of attention spikiness. The proposed method, PolaFormer, is based on the idea of separating the query-key pairs with their plarity into two branches, one for positive and one for negative. Other improvement includes making some previously hand-designed static parameters to be learnable. Experiments are conducted on image classification, object detection and segmentation, and the long range arena tasks.

### Strengths
Good and clear writing, with good background and context, detailed description of equations, and transformer mechnisam, and the literature review of previous linear transformers. 

Visualisation is easy to understand

The key idea is delivered clearly, with the assistance of well designed charts and graphs.

The experiments show good margin as compared to previous alternative methods.

### Weaknesses
First point unclear to me is about the definition of negative values (Line 201). As clearly stated in softmax kernel function (Line 16), they operate in the output end of softmax function where it is deemed that all values are not negative. It is not about the input end of softmax where values can be either negative or positive. Under this context, I do not see anywhere negative values are lost or overlooked as kernel based linear attention is dealing with the softmax-ed space where no negative values exist at all. 

Even if one wants to have negative values, what is the fundamental challenge with just changing the corresponding component, such as instead of using ReLU, one can simply use other activation functions allowing negative values such as leaky ReLU, Tanh etc. No discussion on this simple baseline choice is provided. This also implies that the motivation of going for more complex design is thus not strong. That being said, for being actionable, it is suggested that the authors compare their approach against baselines using activation functions that allow negative values, such as leaky ReLU or tanh or shift sigmoid with both negative and positive. This would help clarify the advantages of their more complex design over simpler alternatives.

It is hard to relate Eq (4) with the loss of negative values as discussed before such as Lines 190-194. Why ReLU based mapping will just operate on latter 3 terms and leave out the first term? This seems to be ad-hoc assumption. I would suggest that the authors provide a more detailed explanation or proof of how ReLU-based mappings specifically affect the terms in Eq (4), and why this leads to information loss. 

The other learnable parts to replacing static hand-designed values is good bit but hard to be claimed as addressing major challenges in this context.

As this work is about scalability of transformer along with the length of input sequences, what is the range of each experiment. In general, the sequences are not that long for vision tasks like image classification, object detection and segmentation. They are not the best suitable test applications. LRA may give longer sequences which is thus better for test. I would suggest the authors to provide specific sequence lengths for each experiment. Additionally, including experiments on tasks with longer sequences would be helpful to better demonstrate the scalability of this approach.

Also, the scalability of the proposed model along with the increase of sequence length is not evaluated, which should be a key aspect for this problem. Including an experiment or analysis that explicitly shows how this model's performance and efficiency scale with increasing sequence length, compared to baseline methods, would be important and insightful.

Similarly, the ablation study is best done with long sequence based tasks, but not short ones like image-net classification.

Table 4: while $\alpha$ in Eq (9) is learnable, why in this table it looks like the value is set to 3/5/7? Or I misunderstand? Besides, please show what are the learned value of $\alpha$ for each other experiment.

### Questions
Please check the weakness parts

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
4

### Summary
This paper introduces PolaFormer, a new attention mechanism designed for vision transformers that aims to improve both expressiveness and efficiency. Traditional linear attention methods reduce computational complexity by approximating the softmax function but often lose critical information by ignoring negative query-key interactions and producing less discriminative attention maps with higher entropy. PolaFormer tackles this issue by explicitly modeling both positive and negative query-key interactions, ensuring a more comprehensive capture of relational information. Additionally, it employs a learnable power function to adjust the attention distribution's entropy, restoring the sharpness characteristic of softmax-based attention. The authors back their approach with theoretical analysis and demonstrate its effectiveness through extensive experiments across various vision tasks, showing notable performance improvements while maintaining linear computational complexity.

### Strengths
1. The key strength is the introduction of polarity-aware attention. By decomposing the query and key vectors into positive and negative components (as in Equation (3)), they capture all possible interactions: positive-positive, negative-negative, positive-negative, and negative-positive. This is a significant departure from traditional methods that only consider positive interactions due to non-negative feature maps.
2. They provide solid theoretical analysis. For example, in Theorem 1, they prove that using a function \( g \) with positive first and second derivatives (like their learnable power function) reduces the entropy of the attention distribution. This mathematical proof gives credibility to their claim that their method restores the "spikiness" of the attention weights.
3. The experimental results are impressive. On ImageNet-1K classification, their PolaFormer variants outperform the baselines by significant margins. For instance, DeiT-T-PolaFormer improves Top-1 accuracy by up to 6.3% over other DeiT variants. In object detection tasks on COCO, they achieve improvements ranging from 2.3% to 4.6% in AP scores. These aren't just marginal gains—they're substantial improvements that demonstrate the practical value of their method.
4. Despite the added complexity of handling negative interactions and using a learnable power function, they manage to keep the computational complexity linear with respect to sequence length \( N \), as shown in their complexity analysis (Equation (10)). They also report faster inference speeds compared to other models with similar FLOPs, which is a big deal for real-world applications.

### Weaknesses
1. Introducing a learnable power function and additional parameters like the polarity coefficients \( G_s \) and \( G_o \) could introduce training challenges, such as sensitivity to initialization or convergence issues. The paper doesn't discuss whether they encountered any of these problems or how they addressed them.
2. Since attention mechanisms are also fundamental in NLP, it would have been interesting to see PolaFormer's performance on language tasks. The paper focuses solely on vision tasks, so we don't know if the benefits carry over to NLP applications.
3. While they do perform an ablation study (as shown in Table 3), it could be more comprehensive. For instance, they could explore how different values of the scaling factor in the learnable power function affect performance, or how sensitive the model is to the choice of convolutional modules used to increase the rank of the attention map.

### Questions
1. Did introducing the learnable power function and the polarity coefficients \( G_s \) and \( G_o \) affect training stability? For example, did you encounter issues like vanishing or exploding gradients? If so, how did you mitigate them?
2. How sensitive is the model's performance to the initialization and learning of the polarity coefficients \( G_s \) and \( G_o \)? Did you notice any significant performance drops or instability when varying these parameters?
3. Have you tried applying PolaFormer to NLP tasks like machine translation or language modeling? Since attention is crucial in these areas too, it would be interesting to see if your method provides benefits there.
4. You theoretically show that your method reduces entropy in the attention distribution. Did you measure the entropy empirically in your experiments to confirm this? It would be interesting to see a plot or some data showing how the entropy changes with your method compared to others.
5. The attention weight visualizations in Figure 1 are helpful. Could you provide more examples, perhaps showing how PolaFormer focuses on relevant parts of the input in different tasks or layers? This could provide more intuitive insight into how your method works.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
To overcome the shortcomings of current linearized self-attention, the paper proposes polarityaware linear attention mechanism that attends to both both same-signed and opposite-signed query-key interactions. the latter on is oftern ignored by existing methods. Besides, the proposed polarity-aware attention can also addresses the loss of attention spikeness.

### Strengths
1. The paper is well written and organized.
2. The paper provides solid motivation for the method and the proposed approach is well justified.
3. The experimental results show promising performance.

### Weaknesses
1. After reading the paper, I am still not sure how the non-negativity constraint is perserved. Especially when a learnable matrix $G$ is applied. Could authors provide more explanation on how the learnable matrix $G$ can perserve the non-negativity constraint?

2. Some latest baselines are missing in the paper. For instance, authors should consider incorporating the latest work [1] for comparsion. This baseline also proposes a new linear self-attention to achieve both high expressiveness capacity and low computation complexity.  The comparison on image classification or object detection would be helpful for a comprehensive assessment of the propsoed method.

3. One minor error in line 98. Negative-negative should be Negative-positive.

[1] Han, Dongchen, et al. "Agent attention: On the integration of softmax and linear attention." ECCV 2024.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Linear attention replaces the softmax over query-key dot products with a kernel function that makes the attention operation linear in kernel space, which breaks the quadratic complexity complexity in the sequence length. 
This paper proposes a variant of linear attention where (1) the positive and negative components of the query and key vectors are separated and (2) the individual components are made more "peaky" by applying a per-component $x \leadsto x^p$ transformation where the powers $p>1$ are learned.

### Strengths
S1.  The method presented in this paper is simple and well justified

S2. Paper generally easy to read 

S3. results outperform the full attention baseline (I merely expected it to be a good approximation)

### Weaknesses
W1. The main dawback of the experiments is that there is not experiment on truly high resolution image, where this method would fully benefit from the linear complexity

W2. Some parts are not clear (see below) 


Unclear points: 
- it would be useful to specify from the intro that the method is applied with full training -- it is unclear until the experiments that this is not fine-tuning and not a drop-in replacement for softmax attention at inference time

L133: better = more accurate or faster? 

L162: relationship between d and D ?

L248 it is unclear how G^s and G^o can be trained since they depend on the batch size N -- or is N assumed to be fixed? 

eq (7) maybe not necessary to define what entropy is... Also the argument about better accounting for negative correlation is repeated 4-6x in the paper. By shaving these repetitions you could make room to move the proof to the main paper.

L300 IIUC d' = d since g() does just a pointwise mapping -- please mention this explicitly

Maybe some useful related work on the power kernels : [Tolias et al, Particular Object Retrieval With Integral Max-Pooling of CNN Activations, ICLR'16]

### Questions
please clarify how G^s and G^o can be trained

### Soundness
3

### Presentation
3

### Contribution
3
